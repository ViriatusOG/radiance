/* Derived from libr4d/r4d_gemm_mxfp4a8_nt_m64.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_gemm_mxfp4a8_nt_m64.hip - halo_gemm_mxfp4a8_nt_m64: OCP-MXFP4 weight, fp8 activation
// skinny GEMM for M up to 64, gfx1151. C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, A quantised per
// row (or per (row, 128)) to e4m3, W an e2m1 element per nibble with one E8M0 exponent per
// 32 elements along K.
//
// The stored weight is libr4d's byte for byte (halo_layout.cpp's mx_slot): one 32-lane slot
// per (n-tile, k-step), lane (lh*16 + r)'s dword holding row r's eight codes from
// k + 8*lh. The gfx11 fragment wants sixteen consecutive K of one column in both halves of
// the wave, so each lane unpacks its own slot dword and the halves trade the results
// (mx_frag16x): one 128-byte load a slot and each code unpacked once. The unpack is the M <= 8
// rows' bound (VALU, not DRAM): the four-codes-a-pass form and the half split took 34816 x 5120
// at M = 1 from ~690 to ~520 us on gfx1151. The per-32 exponent is folded into the element
// against the row's reference exponent: e2m1 value * 2^(exp - ref) is an exact bf16 (the
// exponent field lands in [114, 129] for the checkpoint's spread), and one factor per row,
// 2^(ref - 127), is applied as the scales fold. On gfx1201 the same fold targets e4m3 and
// its fp8 WMMA; here both operands dequantise to bf16 and take the bf16 WMMA, whose products
// are exact either way.
//
// The fold is per 32-block (two WMMA steps): acc += tmp * (2^(ref-127) * a_scale), with
// a_scale the (row, 128) activation scale on the block-scaled grid and 1 on the per-row
// grid, whose scale multiplies once at the store. Split-K is within the workgroup, reduced
// in LDS in split order, so a run repeats exactly and the kernel needs no scratch.
#include "halo.h"
#include "halo_launch.h"   // HALO_DYN_LDS: tests/halo_emu runs this unit on the host
#include "halo_gemm_args.h"
#include "halo_common.h"
#include "halo_gdn_wmma.h"
#include "halo_fp8_wmma.h"
#include "halo_wmma.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

#define HALO_GEMM_MXFP4_MAX_M 64
#define HALO_GEMM_MXFP4_GROUP 32
#define WAVE32 32

typedef int      v8i    __attribute__((ext_vector_type(8)));
typedef float    v8f    __attribute__((ext_vector_type(8)));
typedef int      v4i    __attribute__((ext_vector_type(4)));
typedef int      int4v  __attribute__((ext_vector_type(4)));

/* Sixteen e4m3 bytes to the exact bf16 fragment (as in halo_gemm_fp8a8). */
__device__ __forceinline__ v16bf mx_deq16_exact(const uint32_t* __restrict__ w4) {
    union { uint32_t u[8]; v16bf f; } q;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        const v2f p0 = halo_fp8::e4m3_pk<false>(w4[d]);
        const v2f p1 = halo_fp8::e4m3_pk<true>(w4[d]);
        q.u[d * 2]     = halo::pk_bf16(p0.x, p0.y);
        q.u[d * 2 + 1] = halo::pk_bf16(p1.x, p1.y);
    }
    return q.f;
}

/* One e2m1 code and its block's (exp - ref) as a bf16 bit pattern: value (2+m) * 2^(E-2)
 * with E = mag>>1, m = mag&1, scaled by 2^(exp-ref); zero stays zero (signed). */
__device__ __forceinline__ unsigned mx_e2m1_bf16(unsigned code, int ediff) {
    const unsigned mag = code & 7u;
    if (!mag) return (code & 8u) << 12;
    unsigned E = mag >> 1, m = mag & 1u;
    if (mag == 1) m = 0;   /* the subnormal 0.5: (1 + 0) * 2^-1, not (1 + 1/2) * 2^-1 */
    const unsigned ef = E + (unsigned)(ediff + 126);
    return ((code & 8u) << 12) | (ef << 7) | (m << 6);
}

/* Eight e2m1 codes (one dword) at one (exp - ref) to four bf16-pair dwords, four codes a pass:
 * halo_gemm_mxfp4a8_tiled.hip's mxt_frag8 (that file says how), bit-identical to the scalar
 * mx_e2m1_bf16 for every code and every ediff with 0 <= E + ediff + 126 <= 255 (the range the
 * relayout's reference exponent keeps; host-verified for every byte at every position). */
typedef unsigned short mx_us2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ void mx_frag8(unsigned dw, int ediff, uint32_t* out) {
    const uint32_t ed = (uint32_t)(ediff << 7) & 0xFFFFu;
    const mx_us2 ED2 = __builtin_bit_cast(mx_us2, ed | (ed << 16));
    const uint32_t z = (0u - ed) & 0xFFFFu;
    const uint32_t T_LO = 0xC0800000u | (z & 0xFFu), T_HI = 0x3F3F3F00u | (z >> 8);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t x = __builtin_amdgcn_perm(dw, dw, h ? 0x03030202u : 0x01010000u);
        const uint32_t mag = (x & 0x00070007u) | ((x >> 4) & 0x07000700u);
        const uint32_t sg = ((x << 4) & 0x00800080u) | (x & 0x80008000u);
        const uint32_t lo = __builtin_amdgcn_perm(0xC0804000u, T_LO, mag);
        const uint32_t hi = __builtin_amdgcn_perm(0x40404040u, T_HI, mag);
        const mx_us2 p0 = __builtin_bit_cast(mx_us2, __builtin_amdgcn_perm(hi, lo, 0x05010400u)) + ED2;
        const mx_us2 p1 = __builtin_bit_cast(mx_us2, __builtin_amdgcn_perm(hi, lo, 0x07030602u)) + ED2;
        out[2 * h]     = __builtin_bit_cast(uint32_t, p0) | __builtin_amdgcn_perm(0u, sg, 0x010C000Cu);
        out[2 * h + 1] = __builtin_bit_cast(uint32_t, p1) | __builtin_amdgcn_perm(0u, sg, 0x030C020Cu);
    }
}

/* The B fragment from the slot's dword `dw` of THIS lane (lane l: row l & 15, codes
 * k + 8 * (l >> 4) .. + 8), to sixteen bf16 in consecutive-K order. The fragment wants both of
 * its column's dwords in both halves of the wave; each half unpacks its own and the halves trade
 * the results by v_permlanex16 (lane i reads lane i of the other half), so a slot is one 128-byte
 * load and each code is unpacked once rather than once a half. */
__device__ __forceinline__ v16bf mx_frag16x(unsigned dw, int exp, int ref, int lane) {
    uint32_t own[4], oth[4];
    mx_frag8(dw, exp - ref, own);
#pragma unroll
    for (int q = 0; q < 4; ++q)
        oth[q] = __builtin_amdgcn_permlanex16(own[q], own[q], 0x76543210u, 0xfedcba98u, false, false);
    const bool up = lane >= 16;
    union { uint32_t u[8]; v16bf f; } r;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        r.u[q] = up ? oth[q] : own[q];
        r.u[4 + q] = up ? own[q] : oth[q];
    }
    return r.f;
}

#define HALO_MX_BOUNDS(MT, NPW) ((MT) * (NPW) >= 8 ? 256 : ((MT) * (NPW) >= 4 ? 512 : 1024))

template <int MT, int NPW, bool ABLK>
__global__ __launch_bounds__(HALO_MX_BOUNDS(MT, NPW)) void halo_gemm_mxfp4a8_nt_m64_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int WV, int SK, int as_rs, int lda, int ldws, int ldc,
    const uint16_t* __restrict__ Adq = nullptr) {
  HALO_DYN_LDS(float, red);
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK, kbase = ksplit * Ks;
  const int nlane = lane & 15;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  const int ntmax = N / 16 - 1;
  int wtile[NPW], wrow[NPW];
  float rf[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int t = nt0 + j;
    wtile[j] = t < ntmax ? t : ntmax;
    wrow[j] = wtile[j] * 16 + nlane;
    rf[j] = __int_as_float((int)Wref[wrow[j]] << 23);   /* 2^(ref - 127) */
  }

  const unsigned int* __restrict__ Wp = (const unsigned int*)W;
  const int ksteps = K / 16;

  for (int k = kbase; k < kbase + Ks; k += 32) {
    const int blk = k / HALO_GEMM_MXFP4_GROUP;          /* this 32-block's exponent index */
    v8f tmp[NPW][MT];
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
      for (int i = 0; i < MT; i++) tmp[j][i] = (v8f)(0.f);
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int kg = k + u * 16, kstep = kg / 16;
      v16bf af[MT];
#pragma unroll
      for (int i = 0; i < MT; i++) {
        int r = (mt0 + i) * 16 + nlane;
        r = r < M - 1 ? r : M - 1;
        if (Adq) {
          /* The launch's activation plane: mx_deq16_exact's outputs in row-major
           * [M, K] bf16, read back as the fragment's eight dwords. */
          const uint16_t* src = Adq + (size_t)r * K + kg;
          uint32_t* dst = (uint32_t*)&af[i];
          *(uint4*)dst = *(const uint4*)src;
          *(uint4*)(dst + 4) = *(const uint4*)(src + 8);
        } else {
          const int4v av = *(const int4v*)(A + (size_t)r * lda + kg);
          af[i] = mx_deq16_exact((const uint32_t*)&av);
        }
      }
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const int n = wrow[j];
        const unsigned dw = Wp[((size_t)wtile[j] * ksteps + kstep) * WAVE32 + lane];
        const v16bf bf = mx_frag16x(dw, (int)Ws[(size_t)blk * ldws + n], (int)Wref[n], lane);
#pragma unroll
        for (int i = 0; i < MT; i++) tmp[j][i] = halo::mma(af[i], bf, tmp[j][i]);
      }
    }
    /* Fold this 32-block: per accumulator row, the (row, 128) scale on the block grid. */
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float f = rf[j];
          if constexpr (ABLK) {
            int r = (mt0 + i) * 16 + 2 * e + (lane >> 4);
            r = r < M - 1 ? r : M - 1;
            f *= As[(size_t)r * as_rs + (k >> 7)];
          }
          acc[j][i][e] = fmaf(tmp[j][i][e], f, acc[j][i][e]);
        }
  }

  const int ncols = WV * NPW;
  const int nthreads = WV * SK * WAVE32;
  if (SK == 1) {
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int n = nt0 * 16 + j * 16 + nlane;
      if (n >= N) continue;
#pragma unroll
      for (int i = 0; i < MT; i++) {
        const int mb = (mt0 + i) * 16;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int m = mb + accRow(lane, e);
          if (m < M)
            C[(size_t)m * ldc + (nt0 + j) * 16 + nlane] =
                (__hip_bfloat16)(ABLK ? acc[j][i][e] : acc[j][i][e] * As[m]);
        }
      }
    }
    return;
  }
#pragma unroll
  for (int i = 0; i < MT; i++) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
      for (int e = 0; e < 8; e++) red[slot + e * WAVE32 + lane] = acc[j][i][e];
    }
    __syncthreads();
    for (int idx = tid; idx < ncols * 256; idx += nthreads) {
      const int col = idx >> 8, off = idx & 255;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      const int l = off & 31;
      const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
      const int m = (mt0 + i) * 16 + accRow(l, off >> 5);
      if (n < N && m < M)
        C[(size_t)m * ldc + n] = (__hip_bfloat16)(ABLK ? v : v * As[m]);
    }
  }
}

/* The activation plane: mx_deq16_exact over the whole [M, K) activation, once per
 * launch, so the GEMM's waves read fragments instead of each re-dequantising the
 * same codes for its column. Values are mx_deq16_exact's own; products through
 * the plane are bit-identical. */
__global__ __launch_bounds__(256) void mxfp4_adq_kernel(const unsigned char* __restrict__ A,
                                                        uint16_t* __restrict__ Adq,
                                                        int M, int K, int lda) {
    const int idx = blockIdx.x * (int)blockDim.x + threadIdx.x;
    if (idx >= M * (K / 16)) return;
    const int r = idx / (K / 16), c = (idx % (K / 16)) * 16;
    const int4v av = *(const int4v*)(A + (size_t)r * lda + c);
    const v16bf f = mx_deq16_exact((const uint32_t*)&av);
    uint4* dst = (uint4*)(Adq + (size_t)r * K + c);
    const uint4* src = (const uint4*)&f;
    dst[0] = src[0];
    dst[1] = src[1];
}

static void gemm_mxfp4a8_m64_impl(long a, long ascale, long wq, long ws, long wref, long c,
                                  int M, int K, int N, int WV, int SK, int MB, int NPW,
                                  int as_group, int as_rs, long stream, int lda, int ldws,
                                  int ldc, long adq = 0) {
  if (M < 1 || M > HALO_GEMM_MXFP4_MAX_M)
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: M must be 1..64");
  if (N % 16) throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: N must be a multiple of 16");
  if (K % (SK * HALO_GEMM_MXFP4_GROUP))
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: K must be divisible by SK*32");
  if (WV * SK * WAVE32 > HALO_MX_BOUNDS(MB, NPW))
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: WV*SK*32 exceeds this tile's launch bound");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: MB must be 1..4");
  if (NPW != 1 && NPW != 2 && NPW != 4 && NPW != 8)
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: NPW must be 1, 2, 4 or 8");
  if (as_group != 0 && as_group != 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: the activation scale is per row or per 128 of K");
  if (!lda) lda = K;
  if (!ldws) ldws = N;
  if (!ldc) ldc = N;
  const bool ablk = as_group == 128;
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("r4d gemm_mxfp4a8_nt_m: WV*NPW*SK too large");
  auto Ap = (const unsigned char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Rp = (const unsigned char*)wref;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
  const uint16_t* Adq = nullptr;
  if (adq) {
      Adq = (const uint16_t*)(uintptr_t)adq;
      const int total = M * (K / 16);
      hipLaunchKernelGGL((mxfp4_adq_kernel), dim3((total + 255) / 256), 256, 0, st, Ap, (uint16_t*)Adq, M, K, lda);
      if (hipGetLastError() != hipSuccess) throw std::runtime_error("mxfp4 adq launch");
  }
#define MX_LAUNCH(MTv, NPv)                                                                        \
  do {                                                                                             \
    if (ablk) hipLaunchKernelGGL((halo_gemm_mxfp4a8_nt_m64_kernel<MTv, NPv, true>), grid, block,  \
                                 smem, st, Ap, Sp, Wp, Zp, Rp, Cp, M, K, N, WV, SK, as_rs, lda,    \
                                 ldws, ldc, Adq);                                                  \
    else      hipLaunchKernelGGL((halo_gemm_mxfp4a8_nt_m64_kernel<MTv, NPv, false>), grid, block, \
                                 smem, st, Ap, Sp, Wp, Zp, Rp, Cp, M, K, N, WV, SK, as_rs, lda,    \
                                 ldws, ldc, Adq);                                                  \
  } while (0)
#define MX_LAUNCH_MB(NPv) switch (MB) { \
    case 1: MX_LAUNCH(1, NPv); break; \
    case 2: MX_LAUNCH(2, NPv); break; \
    case 3: MX_LAUNCH(3, NPv); break; \
    case 4: MX_LAUNCH(4, NPv); break; \
    default: throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: MB unsupported"); }
  switch (NPW) {
    case 1: MX_LAUNCH_MB(1); break;
    case 2: MX_LAUNCH_MB(2); break;
    case 4: MX_LAUNCH_MB(4); break;
    case 8: MX_LAUNCH_MB(8); break;
    default: throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: NPW must be 1, 2, 4 or 8");
  }
#undef MX_LAUNCH_MB
#undef MX_LAUNCH
}

void halo_gemm_mxfp4a8_nt_m64_plane(long a, long ascale, long wq, long ws, long wref, long c,
                                    int M, int K, int N, int WV, int SK, int MB, int NPW,
                                    int as_group, int as_rs, long stream, int lda, int ldws,
                                    int ldc, long scratch, long scratch_bytes) {
  const long adq = (scratch && scratch_bytes >= (long)M * K * 2) ? scratch : 0;
  gemm_mxfp4a8_m64_impl(a, ascale, wq, ws, wref, c, M, K, N, WV, SK, MB, NPW, as_group, as_rs,
                        stream, lda, ldws, ldc, adq);
}
void halo_gemm_mxfp4a8_nt_m64(long a, long ascale, long wq, long ws, long wref, long c,
                              int M, int K, int N, int WV, int SK, int MB, int NPW,
                              int as_group, int as_rs, long stream, int lda, int ldws, int ldc) {
  gemm_mxfp4a8_m64_impl(a, ascale, wq, ws, wref, c, M, K, N, WV, SK, MB, NPW, as_group, as_rs,
                        stream, lda, ldws, ldc, 0);
}
int halo_gemm_mxfp4a8_nt_m64_max_m(void) { return HALO_GEMM_MXFP4_MAX_M; }
int halo_gemm_mxfp4a8_nt_m64_group(void) { return HALO_GEMM_MXFP4_GROUP; }

// ============================================================== the plugin ABI entry points
extern "C" int halo_rad_gemm_mxfp4a8_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1, /*parts=*/1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return HALO_NO(RAD_E_SHAPE);
    if (g.M > halo_gemm_mxfp4a8_nt_m64_max_m()) return HALO_NO(RAD_E_SHAPE);
    if (!g.ascale) return HALO_NO(RAD_E_INVAL);
    /* The per-row reference exponent arrives as gemm_nt_q's trailing optional operand. */
    if (!g.bref) return HALO_NO(RAD_E_INVAL);
    MxParts mp;
    rc = mx_parts(a, &g, &mp);
    if (rc != RAD_OK) return rc;
    const int group = halo_gemm_mxfp4a8_nt_m64_group();
    int as_group = 0, as_rs = 1;
    rc = mx_ascale_grid(g, &as_group, &as_rs, mp.n);
    if (rc != RAD_OK) return rc;
    Cfg c;
    rc = cfg_for(a, g.K, g.M, group, BR_FLAT, &c, 0, 2, 16);   /* NPW 2 untuned, as the gemv row */
    if (rc != RAD_OK) return rc;
    const long long kb = (g.K + 127) / 128;
    for (int p = 0; p < mp.n; ++p) {
        const long long n0 = mp.n0[p], np = mp.n1[p] - mp.n0[p];
        HALO_TRY("gemm_mxfp4a8_nt_m64",
            halo_gemm_mxfp4a8_nt_m64_plane(halo_addr(g.a) + p * g.K,
                                     halo_addr(g.ascale) + (long)(p * kb * 4),
                                     halo_addr(g.b) + n0 * g.K / 2, halo_addr(g.bscale) + n0,
                                     halo_addr(g.bref) + n0, halo_addr(g.y) + n0 * 2,
                                     (int)g.M, (int)g.K, (int)np, c.wv, c.sk, c.mb, c.npw,
                                     as_group, as_rs, (long)(uintptr_t)s,
                                     (int)(mp.lda ? mp.lda : g.K), (int)g.N, (int)g.N,
                                     (long)(uintptr_t)a->scratch, a->scratch_bytes));
    }
    return RAD_OK;
}

extern "C" int64_t halo_rad_gemm_mxfp4a8_m64_scratch(const RadArgs* a) {
    const long long M = halo_geti_or(a, "M", 0), K = halo_geti_or(a, "K", 0);
    if (M < 1 || K < 1) return 0;
    return M * K * 2;   /* the activation plane */
}
