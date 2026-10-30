/* Derived from libr4d/r4d_gemm_w2a8_nt_m64.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_gemm_w2a8_nt.hip - halo_gemm_w2a8_nt: 2-bit-weight, 8-bit-activation skinny GEMM, the
// DFlash2 draft head. C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, A quantised per row, W an unsigned
// 2-bit code with an f16 scale and an integer zero per (row, 128 K).
//
// The weight keeps libr4d's fragment order byte for byte (halo_layout.cpp's w2_slot): within a
// (16-row tile, 128 K block), lane r = n % 16 holds eight dwords -- (lh, d) in {0,1} x {0..3} --
// whose 2-bit fields are the row's 128 codes. For one 16-wide k step u a lane needs exactly
// the two dwords (lh = 0, 1) at d = u/2: field 4*(e%4) + (e/4) + 2*(u%2) of dword lh=(j%8)/4
// is code j, with e = (j/8)*4 + (j%4). Sixteen shifts unpack the step; the integer zero is
// folded into the byte (code - zero in [-15, 3] is an exact int8), so the iu8 WMMA accumulates
// the corrected product directly and no activation group sums are needed -- the same integer
// libr4d's asum correction forms, in the same order.
//
// The activation is quant_act_i8's pre-shuffled plane, read as halo_gemm_w8a8_nt_m64 reads it:
// one cross-half exchange reassembles the gfx11 fragment's consecutive sixteen K.
#include "halo.h"
#include "halo_gemm_args.h"
#include "halo_common.h"
#include "halo_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

// Not structural (grid.y tiles M, rows are clamped); a sanity bound only.
#define HALO_GEMM_W2A8_MAX_M 32768
#define WAVE32 32

typedef int      v8i    __attribute__((ext_vector_type(8)));
typedef float    v8f    __attribute__((ext_vector_type(8)));
typedef int      v4i    __attribute__((ext_vector_type(4)));
typedef unsigned uint2v __attribute__((ext_vector_type(2)));

typedef __attribute__((address_space(1))) const uint2v* halo_ga2;

/* The activation fragment, as halo_gemm_w8a8_nt_m64 reads it. */
__device__ __forceinline__ v4i frag_step(unsigned d0, unsigned d1, int lane) {
  const unsigned p0 = (unsigned)__shfl_xor((int)d0, 16);
  const unsigned p1 = (unsigned)__shfl_xor((int)d1, 16);
  const bool hi = lane >> 4;
  v4i r;
  r[0] = (int)(hi ? p0 : d0);
  r[1] = (int)(hi ? d0 : p0);
  r[2] = (int)(hi ? p1 : d1);
  r[3] = (int)(hi ? d1 : p1);
  return r;
}

__device__ __forceinline__ v4i fragA8w(halo_ga2 p, int u, int lane) {
  const uint2v w = p[2 * u];
  return frag_step(w[0], w[1], lane);
}

__device__ __forceinline__ v8i mmai8(v4i a, v4i b, v8i c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}

/* The lane's eight dwords of (tile t, group kb): (lh = 0, d = 0..3) and (lh = 1, d = 0..3), each
 * run contiguous, so two 16-byte loads. */
__device__ __forceinline__ void loadW2(const unsigned* __restrict__ Wq, int t, int kb, int lane,
                                       int kpb, uint4& lo, uint4& hi) {
  const size_t base = ((size_t)(t * kpb + kb) * WAVE32 + (lane & 15)) * 4;
  lo = *(const uint4*)(Wq + base);
  hi = *(const uint4*)(Wq + base + 64);
}

/* One k step's weight fragment: the lane's two dwords at d = u/2, sixteen 2-bit fields
 * unpacked and zero-corrected into the fragment's bytes. Fragment dword d takes fields
 * 4q + h (+1 for d >= 2) of w0 (d even) or w1 (d odd) into byte q, h = 2 (u & 1): the four codes
 * of a dword sit at one shift, so it is a shift and a mask. The zero comes off every byte at once
 * through a bias: `zb` is 0x80 - zero in each byte (zero is 0..15), so 0x80 + code - zero neither
 * carries nor borrows, and the xor leaves code - zero as the int8 the byte loop wrote. */
__device__ __forceinline__ v4i fragW2(const uint4& lo, const uint4& hi, int u, unsigned zb) {
  const int dsel = u >> 1;
  const unsigned w0 = dsel == 0 ? lo.x : dsel == 1 ? lo.y : dsel == 2 ? lo.z : lo.w;
  const unsigned w1 = dsel == 0 ? hi.x : dsel == 1 ? hi.y : dsel == 2 ? hi.z : hi.w;
  const int s = (u & 1) * 4;
  constexpr unsigned M = 0x03030303u, B = 0x80808080u;
  v4i r4;
  r4[0] = (int)((((w0 >> s) & M) + zb) ^ B);
  r4[1] = (int)((((w1 >> s) & M) + zb) ^ B);
  r4[2] = (int)((((w0 >> (s + 2)) & M) + zb) ^ B);
  r4[3] = (int)((((w1 >> (s + 2)) & M) + zb) ^ B);
  return r4;
}

#define HALO_W2A8_BOUNDS(MT, NPW) ((MT) * (NPW) >= 8 ? 256 : ((MT) * (NPW) >= 4 ? 512 : 1024))

template <int MT, int NPW>
__global__ __launch_bounds__(HALO_W2A8_BOUNDS(MT, NPW)) void halo_gemm_w2a8_kernel(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Ws,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK) {
  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK;
  const int groups = Ks / HALO_GEMM_W2A8_GROUP;
  const int kbase = ksplit * Ks;
  const int kpb = K / HALO_GEMM_W2A8_GROUP;      /* 128-blocks a weight row */
  const int nsz = kpb;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  if (n0 < N) {
    halo_ga2 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) {
      int r = (mt0 + i) * 16 + (lane & 15);
      r = r < M - 1 ? r : M - 1;
      const size_t off = (size_t)r * K + kbase + 8 * (lane >> 4);
      pa[i] = (halo_ga2)(const uint2v*)(A + off);
    }
    const int ntlast = N / 16 - 1;
    int tt[NPW];
#pragma unroll
    for (int j = 0; j < NPW; j++) tt[j] = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;

    for (int g = 0; g < groups; g++) {
      const int kb = kbase / HALO_GEMM_W2A8_GROUP + g;
      float sc[NPW];
      unsigned zb[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const unsigned d = Ws[((size_t)(tt[j] * nsz + kb)) * 16 + (lane & 15)];
        sc[j] = __half2float(__ushort_as_half((unsigned short)(d & 0xffffu)));
        const int zero = (int)(-__half2float(__ushort_as_half((unsigned short)(d >> 16))) - 1024.0f);
        zb[j] = 0x80808080u - (unsigned)zero * 0x01010101u;
      }
      uint4 wlo[NPW], whi[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) loadW2(Wq, tt[j], kb, lane, kpb, wlo[j], whi[j]);
      v8i g_acc[NPW][MT];
#pragma unroll
      for (int u = 0; u < 8; u++) {
        v4i afrag[MT];
#pragma unroll
        for (int i = 0; i < MT; i++) afrag[i] = fragA8w(pa[i], u, lane);
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const v4i bfrag = fragW2(wlo[j], whi[j], u, zb[j]);
#pragma unroll
          for (int i = 0; i < MT; i++)
            g_acc[j][i] = (u == 0) ? mmai8(afrag[i], bfrag, (v8i)(0))
                                   : mmai8(afrag[i], bfrag, g_acc[j][i]);
        }
      }
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int e = 0; e < 8; e++)
            acc[j][i][e] = fmaf((float)g_acc[j][i][e], sc[j], acc[j][i][e]);
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += HALO_GEMM_W2A8_GROUP / 8;
    }
  }

  const int ncols = WV * NPW;
  const int nthreads = WV * SK * WAVE32;
  if (SK == 1) {
    if (n0 >= N) return;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int n = (nt0 + j) * 16 + (lane & 15);
      if (n >= N) continue;
#pragma unroll
      for (int i = 0; i < MT; i++) {
        const int mb = (mt0 + i) * 16;
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int m = mb + accRow(lane, e);
          if (m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(acc[j][i][e] * Ascale[m]);
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
      if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(v * Ascale[m]);
    }
  }
}

static void gemm_w2a8_impl(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                           int WV, int SK, int MB, int NPW, long stream) {
  if (M < 1 || M > HALO_GEMM_W2A8_MAX_M)
    throw std::runtime_error("r4d gemm_w2a8_nt: M out of range");
  if (N % 16) throw std::runtime_error("r4d gemm_w2a8_nt: N must be a multiple of 16");
  if (K % (SK * HALO_GEMM_W2A8_GROUP))
    throw std::runtime_error("r4d gemm_w2a8_nt: K must be divisible by SK*group");
  if (WV * SK * WAVE32 > HALO_W2A8_BOUNDS(MB, NPW))
    throw std::runtime_error("r4d gemm_w2a8_nt: WV*SK*32 exceeds this tile's launch bound");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_w2a8_nt: MB must be 1..4");
  if (NPW != 1 && NPW != 2 && NPW != 4 && NPW != 8)
    throw std::runtime_error("r4d gemm_w2a8_nt: NPW must be 1, 2, 4 or 8");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("r4d gemm_w2a8_nt: WV*NPW*SK too large");
  auto Ap = (const signed char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned*)wq;
  auto Zp = (const unsigned*)ws;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH2(MT, NP) hipLaunchKernelGGL((halo_gemm_w2a8_kernel<MT, NP>), \
    grid, block, smem, st, Ap, Sp, Wp, Zp, Cp, M, K, N, WV, SK)
#define LAUNCH2_MB4(NP) switch (MB) { \
    case 1: LAUNCH2(1, NP); break; \
    case 2: LAUNCH2(2, NP); break; \
    case 3: LAUNCH2(3, NP); break; \
    case 4: LAUNCH2(4, NP); break; \
    default: throw std::runtime_error("r4d gemm_w2a8_nt: MB unsupported"); }
#define LAUNCH2_MB2(NP) switch (MB) { \
    case 1: LAUNCH2(1, NP); break; \
    case 2: LAUNCH2(2, NP); break; \
    default: throw std::runtime_error("r4d gemm_w2a8_nt: MB*NPW > 8 would spill"); }
#define LAUNCH2_MB1(NP) switch (MB) { \
    case 1: LAUNCH2(1, NP); break; \
    default: throw std::runtime_error("r4d gemm_w2a8_nt: MB*NPW > 8 would spill"); }
#define LAUNCH2_NPW switch (NPW) { \
    case 1: LAUNCH2_MB4(1); break; \
    case 2: LAUNCH2_MB4(2); break; \
    case 4: LAUNCH2_MB2(4); break; \
    case 8: LAUNCH2_MB1(8); break; \
    default: throw std::runtime_error("r4d gemm_w2a8_nt: NPW must be 1, 2, 4 or 8"); }
  LAUNCH2_NPW;
#undef LAUNCH2_NPW
#undef LAUNCH2_MB1
#undef LAUNCH2_MB2
#undef LAUNCH2_MB4
#undef LAUNCH2
}

/* libr4d's raw signature (halo.h, C linkage). The zero point is folded into the int8 fragment
 * bytes here, so `asum` is not read, and the load hint `NT` has no gfx1151 form. */
extern "C" void halo_gemm_w2a8_nt(long a, long ascale, long wq, long ws, long asum, long c,
                                  int M, int K, int N, int WV, int SK, int MB, int NPW, int NT,
                                  long stream) {
  (void)asum; (void)NT;
  gemm_w2a8_impl(a, ascale, wq, ws, c, M, K, N, WV, SK, MB, NPW, stream);
}
static int halo_gemm_w2a8_nt_max_m(void) { return HALO_GEMM_W2A8_MAX_M; }

// ============================================================== the plugin ABI entry points
extern "C" int halo_rad_gemm_w2a8(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return HALO_NO(RAD_E_SHAPE);
    if (g.M > halo_gemm_w2a8_nt_max_m()) return HALO_NO(RAD_E_SHAPE);
    if (!g.ascale) return HALO_NO(RAD_E_INVAL);
    Cfg c;
    /* With the tile count, an untuned split is the smallest that fills the part: at the draft
     * head's N (a vocabulary, ~15500 tiles) that is no split, 1.9x faster on gfx1151 than the
     * largest legal one (20 at K = 2560), which leaves a wave one scale group. */
    rc = cfg_for(a, g.K, g.M, HALO_GEMM_W2A8_GROUP, BR_W2A8, &c, g.N / 16);
    if (rc != RAD_OK) return rc;
    HALO_TRY("gemm_w2a8_nt",
        gemm_w2a8_impl(halo_addr(g.a), halo_addr(g.ascale), halo_addr(g.b),
                       halo_addr(g.bscale), halo_addr(g.y),
                       (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, c.npw,
                       (long)(uintptr_t)s));
    return RAD_OK;
}

/* The raw-pointer query libr4d exports beside the row (halo.h). */
extern "C" int halo_gemm_w2a8_nt_group(void) { return HALO_GEMM_W2A8_GROUP; }
