// r4d_gemm_mxfp4a8_gemv.hip - r4d_gemm_mxfp4a8_gemv: the OCP-MXFP4 x fp8 GEMM for one to eight rows
// (and, its WMMA form alone, up to 64: the gemv_wide row), gfx1201. C[M,N] = A[M,K] @
// dequant(Wq)[N,K]^T with the operands, weight layout and fold of r4d_gemm_mxfp4a8_decode (and
// _nt_m64, _tiled).
//
// At one stream's decode (M = 1, or 1 + the draft depth) the GEMM is a pure weight stream, and the
// decode kernel's WMMA tile spends its time staging the weight through LDS between two barriers a
// slab, which leaves narrow projections at a third to two thirds of DRAM bandwidth. This kernel
// keeps the weight out of LDS altogether: every lane loads the weight words it multiplies, folds
// them to e4m3 in registers by r4d_mxfp4_unpack8 exactly as the other MXFP4 kernels fold them (the
// same weight bytes in every M band), and the block's waves split K and meet once in LDS at the end.
//
// Two forms, chosen from K alone so that a row's output bits do not depend on how many rows the
// step carries (a verify step's first row is bit-identical to a plain decode of it):
//
// - K a multiple of 128: the WMMA form. One 16-column tile a block (two from 25 rows where
//   N >= 4096, K >= 3072 and the tile count is even), eight waves on its K in chunks of one 128
//   block, and up to four 16-row tiles (M <= 64), each weight word multiplied with every row
//   tile's fragment once it is loaded. A lane loads its own B-fragment word of each k-step
//   straight from the fragment-order weight (the order is the fragment's), the rows sit in the A
//   fragments (rows >= M zero), and v_wmma_f32_16x16x16_fp8_fp8 multiplies. The per-(row, 128)
//   activation scale is applied to each chunk's partial. At M = 1 the WMMA does sixteen times the
//   work it needs, which costs nothing next to the weight stream; from M = 2 up it is what keeps
//   the kernel on the stream (the dot form pays a dot per row).
// - Otherwise (K = 64 mod 128): the dot form. A lane loads 16 bytes -- four columns' eight codes of
//   one k-step -- so a wave reads four k-steps, 512 contiguous bytes, from each of its TN tiles, and
//   multiplies with v_dot4_f32_fp8_fp8; the activation a lane reads (eight k of each row) serves its
//   four columns in all TN tiles. A k-step group (64 of K) lies in one 128 block, so the activation
//   scale is applied as each partial is made. The lanes holding the same columns are summed by
//   shuffles. TN only widens a block; the sum order of a column is fixed by KW, which is constant.
//
// The column's 2^(ref - 127) is applied once, as in the decode kernel's epilogue. No scratch and no
// counters: there is no split across blocks.
#include "r4d.h"
#include "r4d_keyed.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_mxfp4.h"

#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#define R4D_MXG_MAX_M 8
#define R4D_MXG_WMMA_MAX_M 64   // the WMMA form's bound: four 16-row tiles a wave
#define R4D_MXG_KW 8     // waves on a block's K, both forms; a constant, so the sum order is too

typedef unsigned mxg_u4 __attribute__((ext_vector_type(4)));
typedef unsigned mxg_u2 __attribute__((ext_vector_type(2)));
typedef float mxg_v8f __attribute__((ext_vector_type(8)));
typedef int   mxg_v2i __attribute__((ext_vector_type(2)));

// The dot form: TN 16-column tiles a block, MT >= M rows. Needs K % 64 == 0.
template <int MT, int TN, bool ABLK>
__global__ __launch_bounds__(R4D_MXG_KW * 32)
void r4d_gemm_mxfp4a8_gemv_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, __hip_bfloat16* __restrict__ C, int M, int K,
    int as_rs, int lda, int ldws, int ldc) {
  constexpr int KW = R4D_MXG_KW;
  __shared__ float sP[KW][MT][16 * TN];
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);   // uniform: scalar loads
  const int nt0 = blockIdx.x * TN;               // the block's first 16-column tile
  const int g = lane & 7, j = lane >> 3;         // the slot quad in a k-step; the k-step of four
  const int khalf = g >> 2, cq = (g * 4) & 15;   // this lane's k half and first of its four columns
  const int ksteps = K >> 4, groups = K >> 6;
  const mxg_u4* __restrict__ Wv = (const mxg_u4*)W;

  int ref[TN][4];
#pragma unroll
  for (int t = 0; t < TN; ++t) {
    const unsigned r4 = *(const unsigned*)(Wref + (nt0 + t) * 16 + cq);
#pragma unroll
    for (int c = 0; c < 4; ++c) ref[t][c] = (int)((r4 >> (8 * c)) & 0xFFu);
  }

  float acc[MT][TN][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int t = 0; t < TN; ++t)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[m][t][c] = 0.f;

#pragma unroll 4
  for (int gi = wave; gi < groups; gi += KW) {
    const int kstep = gi * 4 + j;
    const int k0 = kstep * 16 + khalf * 8;
    const int kb = gi >> 1;                      // the wave's 128 block (a group is 64 of K)
    mxg_u4 wv[TN];
    unsigned e4[TN];
#pragma unroll
    for (int t = 0; t < TN; ++t) {
      wv[t] = __builtin_nontemporal_load(&Wv[((size_t)(nt0 + t) * ksteps + kstep) * 8 + g]);
      e4[t] = *(const unsigned*)(Ws + (size_t)(k0 >> 5) * ldws + (nt0 + t) * 16 + cq);
    }
    unsigned lo[TN][4], hi[TN][4];
#pragma unroll
    for (int t = 0; t < TN; ++t)
#pragma unroll
      for (int c = 0; c < 4; ++c)
        r4d_mxfp4_unpack8(wv[t][c], r4d_mxfp4_fold_d(ref[t][c], (int)((e4[t] >> (8 * c)) & 0xFFu)),
                          &lo[t][c], &hi[t][c]);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m < M) {
        const mxg_u2 av = *(const mxg_u2*)(A + (size_t)m * lda + k0);
        const float sc = ABLK ? As[(size_t)m * as_rs + kb] : 1.f;
#pragma unroll
        for (int t = 0; t < TN; ++t)
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            float d = __builtin_amdgcn_dot4_f32_fp8_fp8(lo[t][c], av.x, 0.f);
            d = __builtin_amdgcn_dot4_f32_fp8_fp8(hi[t][c], av.y, d);
            acc[m][t][c] = ABLK ? fmaf(d, sc, acc[m][t][c]) : acc[m][t][c] + d;
          }
      }
    }
  }

  // The lanes with the same columns: the other k half (lane ^ 4) and the other k-steps (^ 8, ^ 16).
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int t = 0; t < TN; ++t)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        float v = acc[m][t][c];
        v += __shfl_xor(v, 4, 32);
        v += __shfl_xor(v, 8, 32);
        v += __shfl_xor(v, 16, 32);
        acc[m][t][c] = v;
      }
  if (lane < 4) {
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int t = 0; t < TN; ++t)
#pragma unroll
        for (int c = 0; c < 4; ++c) sP[wave][m][t * 16 + cq + c] = acc[m][t][c];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < MT * 16 * TN; idx += KW * 32) {
    const int m = idx / (16 * TN), cc = idx % (16 * TN);
    if (m < M) {
      float sum = 0.f;
#pragma unroll
      for (int w = 0; w < KW; ++w) sum += sP[w][m][cc];
      const int col = nt0 * 16 + cc;
      const float rf = __int_as_float((int)Wref[col] << 23);
      C[(size_t)m * ldc + col] = (__hip_bfloat16)(ABLK ? sum * rf : sum * rf * As[(size_t)m * as_rs]);
    }
  }
}

// The WMMA form: TN 16-column tiles a block, RT 16-row tiles (M <= 16 RT). Needs K % 128 == 0 (a
// chunk is a whole 128 block; a tail would be dropped); the launcher guarantees it and picks RT and TN.
// A wave loads each weight word of a chunk once and runs one WMMA chain per row tile over it, so the
// weight streams once whatever M is, and a row's sum is its tile's chain alone -- the same at every RT
// and TN.
//
// TN: the activation is read per block, so with one column tile a block it is read N/16 times
// (356 MB from L2 at M = 64 on a 17408 x 5120 weight, against its 47 MB). Two tiles a block load
// each activation fragment once for both. On narrow or shallow weights two tiles lose (fewer
// blocks than the part has units, a short K a wave), so the launcher takes two only from N 4096
// and K 3072, past 24 rows. Four tiles spill past 32 rows.
template <bool ABLK, int RT, int TN = 1>
__global__ __launch_bounds__(R4D_MXG_KW * 32)
void r4d_gemm_mxfp4a8_gemvw_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, __hip_bfloat16* __restrict__ C, int M, int K,
    int as_rs, int lda, int ldws, int ldc) {
  constexpr int KW = R4D_MXG_KW;
  __shared__ float sP[KW][16 * RT][16];
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int nt0 = blockIdx.x * TN;
  const int col = lane & 15, khalf = lane >> 4;  // B fragment: this lane's column and k half
  const int ksteps = K >> 4, chunks = K >> 7;
  const unsigned* __restrict__ Wp = (const unsigned*)W;
  int ref[TN];
#pragma unroll
  for (int t = 0; t < TN; ++t) ref[t] = (int)Wref[(nt0 + t) * 16 + col];
  mxg_v8f tot[RT][TN];
#pragma unroll
  for (int r = 0; r < RT; ++r)
#pragma unroll
    for (int t = 0; t < TN; ++t) tot[r][t] = (mxg_v8f)(0.f);
  for (int ch = wave; ch < chunks; ch += KW) {
    unsigned wv[TN][8];
#pragma unroll
    for (int t = 0; t < TN; ++t)
#pragma unroll
      for (int s = 0; s < 8; ++s)
        wv[t][s] = __builtin_nontemporal_load(&Wp[((size_t)(nt0 + t) * ksteps + ch * 8 + s) * 32 + lane]);
    int e8[TN][4];
#pragma unroll
    for (int t = 0; t < TN; ++t)
#pragma unroll
      for (int q = 0; q < 4; ++q) e8[t][q] = (int)Ws[(size_t)(ch * 4 + q) * ldws + (nt0 + t) * 16 + col];
    mxg_v2i wf[TN][8];
#pragma unroll
    for (int t = 0; t < TN; ++t)
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        unsigned lo, hi;
        r4d_mxfp4_unpack8(wv[t][s], r4d_mxfp4_fold_d(ref[t], e8[t][s >> 1]), &lo, &hi);
        wf[t][s][0] = (int)lo; wf[t][s][1] = (int)hi;
      }
#pragma unroll
    for (int r = 0; r < RT; ++r) {
      const int row = r * 16 + col;               // A fragment: this lane's row in tile r
      const bool live = row < M;
      mxg_v2i af[8];
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        if (live) {
          const unsigned* pa = (const unsigned*)(A + (size_t)row * lda + (ch * 8 + s) * 16 + khalf * 8);
          af[s][0] = (int)pa[0]; af[s][1] = (int)pa[1];
        } else {
          af[s][0] = 0; af[s][1] = 0;
        }
      }
      const float sc = ABLK ? (live ? As[(size_t)row * as_rs + ch] : 0.f) : 1.f;
#pragma unroll
      for (int t = 0; t < TN; ++t) {
        mxg_v8f acc = (mxg_v8f)(0.f);
#pragma unroll
        for (int s = 0; s < 8; ++s)
          acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(wf[t][s], af[s], acc);
        // acc[e]: row (lane & 15) of tile r, column (lane >> 4) * 8 + e -- as the decode epilogue reads it.
#pragma unroll
        for (int e = 0; e < 8; ++e) tot[r][t][e] = fmaf(acc[e], sc, tot[r][t][e]);
      }
    }
  }
  // One column tile at a time through sP (TN of them would not fit in LDS at four row tiles).
#pragma unroll
  for (int t = 0; t < TN; ++t) {
    if (t) __syncthreads();
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int e = 0; e < 8; ++e) sP[wave][r * 16 + col][khalf * 8 + e] = tot[r][t][e];
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * RT * 16; idx += KW * 32) {
      const int m = idx >> 4, c = idx & 15;
      if (m < M) {
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < KW; ++w) sum += sP[w][m][c];
        const int cc = (nt0 + t) * 16 + c;
        const float rf = __int_as_float((int)Wref[cc] << 23);
        C[(size_t)m * ldc + cc] = (__hip_bfloat16)(ABLK ? sum * rf : sum * rf * As[(size_t)m * as_rs]);
      }
    }
  }
}

namespace {
void mxg_launch(long a, long ascale, long wq, long ws, long wref, long c, int M, int K, int N,
                int as_group, int as_rs, long stream, int lda, int ldws, int ldc) {
  if (M < 1 || M > (K % 128 == 0 ? R4D_MXG_WMMA_MAX_M : R4D_MXG_MAX_M))
    throw std::runtime_error("r4d gemm_mxfp4a8_gemv: M must be 1..8, or 1..64 when K % 128 == 0");
  if (N < 16 || N % 16) throw std::runtime_error("r4d gemm_mxfp4a8_gemv: N must be a multiple of 16");
  if (K < 64 || K % 64) throw std::runtime_error("r4d gemm_mxfp4a8_gemv: K must be a multiple of 64");
  if (as_group != 0 && as_group != 128)
    throw std::runtime_error("r4d gemm_mxfp4a8_gemv: the activation scale is per row or per 128 of K");
  const bool ablk = as_group == 128;
  const int tiles = N / 16;
  hipStream_t st = (hipStream_t)stream;
  auto Ap = (const unsigned char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Rp = (const unsigned char*)wref;
  auto Cp = (__hip_bfloat16*)c;
  const dim3 block(R4D_MXG_KW * 32);
  if (K % 128 == 0) {
    const int rt = (M + 15) / 16;
    // Two column tiles a block from 25 rows on a wide and deep enough weight (see the kernel); an
    // odd tile count keeps one.
    const bool tn2 = M > 24 && tiles % 2 == 0 && N >= 4096 && K >= 3072;
#define MXW_L(AB_, RT_, TN_) hipLaunchKernelGGL((r4d_gemm_mxfp4a8_gemvw_kernel<AB_, RT_, TN_>), \
                                                dim3(tiles / TN_), block, 0, st, Ap, Sp, Wp, Zp, Rp, \
                                                Cp, M, K, as_rs, lda, ldws, ldc)
#define MXW_TN(AB_, RT_) { if (tn2) MXW_L(AB_, RT_, 2); else MXW_L(AB_, RT_, 1); }
#define MXW_AB(RT_) { if (ablk) MXW_TN(true, RT_) else MXW_TN(false, RT_) }
    // One row tile is at most 16 rows, below the two-column threshold.
    if (rt <= 1) { if (ablk) MXW_L(true, 1, 1); else MXW_L(false, 1, 1); }
    else if (rt == 2) MXW_AB(2) else if (rt == 3) MXW_AB(3) else MXW_AB(4)
#undef MXW_AB
#undef MXW_TN
#undef MXW_L
    return;
  }
  // Tiles a block carries: enough to amortise the activation loads at several rows, the widest that
  // divides the tile count.
  int tn = M >= 3 ? 4 : (M == 2 ? 2 : 1);
  while (tn > 1 && tiles % tn) tn >>= 1;
  const dim3 grid(tiles / tn);
#define MXG_L(MT_, TN_, AB_)                                                                   \
  hipLaunchKernelGGL((r4d_gemm_mxfp4a8_gemv_kernel<MT_, TN_, AB_>), grid, block, 0, st, Ap, Sp, \
                     Wp, Zp, Rp, Cp, M, K, as_rs, lda, ldws, ldc)
#define MXG_AB(MT_, TN_) { if (ablk) MXG_L(MT_, TN_, true); else MXG_L(MT_, TN_, false); }
#define MXG_TN(MT_) { if (tn == 4) MXG_AB(MT_, 4) else if (tn == 2) MXG_AB(MT_, 2) else MXG_AB(MT_, 1) }
  if (M == 1) MXG_AB(1, 1)
  else if (M == 2) { if (tn == 2) MXG_AB(2, 2) else MXG_AB(2, 1) }
  else if (M <= 4) MXG_TN(4)
  else MXG_TN(8)
#undef MXG_TN
#undef MXG_AB
#undef MXG_L
}
}  // namespace

// ============================================================== the plugin ABI entry point
extern "C" int r4d_rad_gemm_mxfp4a8_gemv(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1, /*parts=*/1);
    if (rc != RAD_OK) return rc;
    if (g.N < 16 || g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale || !g.bref) return R4D_NO(RAD_E_INVAL);
    /* The dot form reads four columns' exponents and references as one word. */
    if (!r4d_aligned((const void*)r4d_addr(g.bscale), 4) || !r4d_aligned((const void*)r4d_addr(g.bref), 4))
        return R4D_NO(RAD_E_ALIGN);
    MxParts mp;
    rc = mx_parts(a, &g, &mp);
    if (rc != RAD_OK) return rc;
    if (g.K % 64) return R4D_NO(RAD_E_SHAPE);
    if (g.M > (g.K % 128 == 0 ? R4D_MXG_WMMA_MAX_M : R4D_MXG_MAX_M)) return R4D_NO(RAD_E_SHAPE);
    int as_group = 0, as_rs = 1;
    rc = mx_ascale_grid(g, &as_group, &as_rs, mp.n);
    if (rc != RAD_OK) return rc;
    const long long kb = (g.K + 127) / 128;
    for (int p = 0; p < mp.n; ++p) {
        const long long n0 = mp.n0[p], np = mp.n1[p] - mp.n0[p];
        R4D_TRY("gemm_mxfp4a8_gemv",
            mxg_launch(r4d_addr(g.a) + p * g.K, r4d_addr(g.ascale) + (long)(p * kb * 4),
                       r4d_addr(g.b) + n0 * g.K / 2, r4d_addr(g.bscale) + n0,
                       r4d_addr(g.bref) + n0, r4d_addr(g.y) + n0 * 2,
                       (int)g.M, (int)g.K, (int)np, as_group, as_rs, (long)(uintptr_t)s,
                       (int)(mp.lda ? mp.lda : g.K), (int)g.N, (int)g.N));
    }
    return RAD_OK;
}
