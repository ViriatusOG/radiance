/* Derived from libr4d/r4d_gemm_w8a16_nt_m64.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_gemm_w8a16_nt_m64.hip - halo_gemm_w8a16_nt_m64: 8-bit-weight skinny GEMM for M up to 64,
// gfx1151.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, f16 A, bf16 C, fp32 accumulate.
//
// Used for lm_head, quantised to eight bits rather than four because its error lands on the
// sampled token rather than on a residual later layers can absorb.
//
// A skinny kernel holds every row of M in registers, so the weight is read once regardless of M.
// General int8 GEMMs pad M <= 16 to two 16-row macro tiles and read the weight twice. The scale is
// applied to the in-register accumulator and stored as bf16, with no separate epilogue launch.
//
// Structure matches halo_gemm_w4a16_nt_m64: fragment order, 512 B packed block (32 lanes x 16 B),
// (scale, zero) dword per (row, group), split-K reduction and clamped tiles. Only the bits per
// element differ, so a packed block covers 32 K instead of 64.
//
// Dequant: `0x6400 | b` is exactly the f16 for 1024 + b for b in 0..255, so eight v_perm_b32 turn
// four dwords into a 16x16x16 fragment, and one v_pk_add_f16 per pair adds -(1024 + zero). The
// subtraction is exact (integers below 2048), so the group scale is applied to the f32 accumulator.
// The stored bytes remain libr4d's fragment order; dequant8_step exchanges K quarters across the
// wave's halves to form this part's consecutive-K fragment (see there).
//
// A is f16, not bf16, because bf16's eight significant bits cannot hold 1024 + b. The alternative
// (raw codes plus a per-group activation sum, as in halo_gemm_w4a8_asym_nt_m64) gains nothing here:
// the adds hide under the weight stream. The caller's bf16 -> f16 cast is exact for a post-norm
// hidden state (O(10)).
//
// Quantiser: asymmetric, per output channel, per group of HALO_GEMM_W8_GROUP contiguous K:
// w ~= scale * (q - zero), q in 0..255, integer zero. One dword per (row, group): the f16 scale low,
// the f16 of -(1024 + zero) high. The group costs 3.1% more bytes than a per-channel scale and
// halves the KL divergence against the unquantised layer.
#include "halo.h"
#include "halo_gemm_args.h"
#include "halo_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

#define HALO_GEMM_W8_MAX_M 64
#define HALO_GEMM_W8_KPB   32         // K per packed block: 2 k steps, 16 bytes per lane
#define WAVE32 32

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));

typedef __attribute__((address_space(1))) const uint4v*             halo_gw4;
typedef __attribute__((address_space(1))) const unsigned long long* halo_gu2;
typedef __attribute__((address_space(1))) const unsigned*           halo_gu1;

/* gfx11 fragment: the row's whole 16-wide k run, 32 contiguous bytes, where the gfx12 form was
 * two 8-byte halves sixteen bytes apart. The pointer step is 32 bytes either way. */
__device__ __forceinline__ v16h fragA8(halo_gu2 p, int u) {
  union { unsigned long long w[4]; v16h f; } x;
  x.w[0] = p[4 * u];
  x.w[1] = p[4 * u + 1];
  x.w[2] = p[4 * u + 2];
  x.w[3] = p[4 * u + 3];
  return x.f;
}

// Four dwords of four consecutive-K bytes -> one 16x16x16 f16 B fragment, minus the zero point.
// Each v_perm_b32 pairs two bytes with the high byte 0x64 (the f16 exponent for 1024). nz holds
// -(1024 + zero) in both halves.
__device__ __forceinline__ v16h dequant8(unsigned w0, unsigned w1, unsigned w2, unsigned w3,
                                         __half2 nz) {
  const unsigned K64 = 0x64646464u;
  const unsigned d[4] = { w0, w1, w2, w3 };
  union { unsigned u[8]; __half2 h2[8]; v16h f; } x;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    x.u[2 * i]     = __builtin_amdgcn_perm(K64, d[i], 0x05010400u);
    x.u[2 * i + 1] = __builtin_amdgcn_perm(K64, d[i], 0x05030402u);
  }
#pragma unroll
  for (int i = 0; i < 8; i++) x.h2[i] = __hadd2(x.h2[i], nz);
  return x.f;
}

/* The stored weight keeps libr4d's gfx12 fragment order byte for byte. For one k step a lane's
 * two dwords hold K quarters (0,2) in the low half-wave and (1,3) in the high half-wave for the
 * same row, so one cross-half exchange per dword reassembles the gfx11 fragment's consecutive
 * sixteen K in order. */
__device__ __forceinline__ v16h dequant8_step(unsigned w0, unsigned w1, __half2 nz, int lane) {
  const unsigned p0 = (unsigned)__shfl_xor((int)w0, 16);
  const unsigned p1 = (unsigned)__shfl_xor((int)w1, 16);
  const bool hi = lane >> 4;
  return dequant8(hi ? p0 : w0, hi ? w0 : p0, hi ? p1 : w1, hi ? w1 : p1, nz);
}

__device__ __forceinline__ v8f mmah8(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

template <int MT, int NPW, bool NT>
__global__ __launch_bounds__(1024) void halo_gemm_w8a16_nt_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned* __restrict__ Wq,
    const unsigned* __restrict__ Wsz, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int WV, int SK) {
  extern __shared__ float red8[];       // WV*NPW*SK*256 floats: a 16x16 tile per (column, split)
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK;
  const int groups = Ks / HALO_GEMM_W8_GROUP;
  const int kbase = ksplit * Ks;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  if (n0 < N) {
    unsigned aoff[MT];
#pragma unroll
    for (int i = 0; i < MT; i++)
      aoff[i] = rowGo((unsigned)K, (mt0 + i) * 16, M - 1, lane) + (unsigned)kbase;
    halo_gu2 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) pa[i] = (halo_gu2)(const unsigned long long*)(A + aoff[i]);

    const int kpb = K / HALO_GEMM_W8_KPB;                  // packed blocks in a full row of tiles
    const int nsz = K / HALO_GEMM_W8_GROUP;                // (scale, zero) dwords per row of tiles
    halo_gw4 pw[NPW];
    halo_gu1 ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (halo_gw4)(const uint4v*)(Wq) +
              (size_t)(t * kpb + kbase / HALO_GEMM_W8_KPB) * WAVE32 + lane;
      ps[j] = (halo_gu1)(const unsigned*)(Wsz) +
              (size_t)(t * nsz + kbase / HALO_GEMM_W8_GROUP) * 16 + (lane & 15);
    }

    const int bpg = HALO_GEMM_W8_GROUP / HALO_GEMM_W8_KPB;  // packed blocks per group

    for (int g = 0; g < groups; g++) {
      float sc[NPW];
      __half2 nz[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const unsigned sz = ps[j][(size_t)g * 16];
        sc[j] = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
        nz[j] = __half2half2(__ushort_as_half((unsigned short)(sz >> 16)));
      }

      v8f g_acc[NPW][MT];
#pragma unroll
      for (int b = 0; b < bpg; b++) {
        uint4v wv[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const size_t o = (size_t)(g * bpg + b) * WAVE32;
          wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
        }
#pragma unroll
        for (int s = 0; s < 2; s++) {
          const int u = b * 2 + s;
          v16h afrag[MT];
#pragma unroll
          for (int i = 0; i < MT; i++) afrag[i] = fragA8(pa[i], u);
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            const v16h bfrag = dequant8_step(wv[j][2 * s], wv[j][2 * s + 1], nz[j], lane);
#pragma unroll
            for (int i = 0; i < MT; i++)
              g_acc[j][i] = u == 0 ? mmah8(afrag[i], bfrag, (v8f)(0.f))
                                   : mmah8(afrag[i], bfrag, g_acc[j][i]);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int e = 0; e < 8; e++) acc[j][i][e] = fmaf(g_acc[j][i][e], sc[j], acc[j][i][e]);
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += (HALO_GEMM_W8_GROUP / 16) * 4;
    }
  }

  const int ncols = WV * NPW;
  const int nthreads = WV * SK * WAVE32;
#pragma unroll
  for (int i = 0; i < MT; i++) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
      for (int e = 0; e < 8; e++) red8[slot + e * WAVE32 + lane] = acc[j][i][e];
    }
    __syncthreads();
    for (int idx = tid; idx < ncols * 256; idx += nthreads) {
      const int col = idx >> 8, off = idx & 255;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red8[(col * SK + s) * 256 + off];
      const int l = off & 31;
      const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
      const int m = (mt0 + i) * 16 + accRow(l, off >> 5);
      if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)v;
    }
  }
}

void halo_gemm_w8a16_nt_m64(long a, long wq, long wsz, long c, int M, int K, int N,
                           int WV, int SK, int MB, int NPW, int NT, long stream) {
  // Any M is correct (grid.y covers row tiles past MT; rows past M are clamped and masked). 64 is
  // the band this shape suits, since each extra row-tile group re-reads the weight; max_m()
  // reports it.
  if (M < 1) throw std::runtime_error("r4d gemm_w8a16_nt_m64: M must be >= 1");
  if (N % 16) throw std::runtime_error("r4d gemm_w8a16_nt_m64: N must be a multiple of 16");
  if (K % (SK * HALO_GEMM_W8_GROUP))
    throw std::runtime_error("r4d gemm_w8a16_nt_m64: K must be divisible by SK*128");
  if (WV * SK * WAVE32 > 1024) throw std::runtime_error("r4d gemm_w8a16_nt_m64: threads/block > 1024");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_w8a16_nt_m64: MB must be 1..4");
  if (NPW != 1 && NPW != 2 && NPW != 4)
    throw std::runtime_error("r4d gemm_w8a16_nt_m64: NPW must be 1, 2 or 4");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("r4d gemm_w8a16_nt_m64: WV*NPW*SK too large");
  auto Ap = (const unsigned short*)a;
  auto Wp = (const unsigned*)wq;
  auto Sp = (const unsigned*)wsz;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH8(MT, NP, TT) hipLaunchKernelGGL((halo_gemm_w8a16_nt_m64_kernel<MT, NP, TT>), grid, \
                                               block, smem, st, Ap, Wp, Sp, Cp, M, K, N, WV, SK)
#define LAUNCH8_MB(NP, TT) switch (MB) { \
    case 1: LAUNCH8(1, NP, TT); break; \
    case 2: LAUNCH8(2, NP, TT); break; \
    case 3: LAUNCH8(3, NP, TT); break; \
    case 4: LAUNCH8(4, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w8a16_nt_m64: MB unsupported"); }
  if (NPW == 1) {
    if (NT) { LAUNCH8_MB(1, true); } else { LAUNCH8_MB(1, false); }
  } else if (NPW == 2) {
    if (NT) { LAUNCH8_MB(2, true); } else { LAUNCH8_MB(2, false); }
  } else {
    if (NT) { LAUNCH8_MB(4, true); } else { LAUNCH8_MB(4, false); }
  }
#undef LAUNCH8_MB
#undef LAUNCH8
}

int halo_gemm_w8a16_nt_m64_max_m(void) { return HALO_GEMM_W8_MAX_M; }
int halo_gemm_w8a16_nt_m64_group(void) { return HALO_GEMM_W8_GROUP; }

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in halo_gemm_args.h; see libhalo/halo_args.h.
extern "C" int halo_rad_gemm_w8a16_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return HALO_NO(RAD_E_SHAPE);
    /* Any other activation dtype would read plausible but wrong numbers. */
    if (g.a->dtype != RAD_F16) return HALO_NO(RAD_E_DTYPE);
    if (g.M > halo_gemm_w8a16_nt_m64_max_m()) return HALO_NO(RAD_E_SHAPE);
    const int group = halo_gemm_w8a16_nt_m64_group();
    Cfg c;
    rc = cfg_for(a, g.K, g.M, group, BR_FLAT, &c);
    if (rc != RAD_OK) return rc;
    HALO_TRY("gemm_w8a16_nt_m64",
        halo_gemm_w8a16_nt_m64(halo_addr(g.a), halo_addr(g.b), halo_addr(g.bscale), halo_addr(g.y),
                              (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, c.npw, c.nt,
                              (long)(uintptr_t)s));
    return RAD_OK;
}

