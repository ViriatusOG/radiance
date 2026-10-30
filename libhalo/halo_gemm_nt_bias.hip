/* Derived from libr4d/r4d_gemm_bf16_nt_bias.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_gemm_nt_bias.hip - gemm_bf16_nt_bias_tiled: biased GEMM C = res + act(A @ W^T + bias)
// over 64-deep K tiles, the K tail staged as zeros (halo_gemm_bf16_nt_tiled's tile, with the
// epilogue libr4d's row promises):
//
//     t = bf16(acc + bias);  if act: t = bf16(act(t));  if res: t = bf16(res + t)
//
// three RNE roundings in the store, as the unfused chain's tensors would round. The f32
// accumulation order is the kernel's own (the row's tolerance allows it); what is pinned is
// the roundings and the activation forms (gelu erf, gelu_tanh). `res` may be the output
// itself: each element is read by the lane that overwrites it.
#include "halo.h"
#include "halo_gemm_args.h"
#include "halo_common.h"
#include "halo_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>

#define NB_BK 64                // K per staged tile: four WMMA k steps
#define NB_PITCH (NB_BK + 8)
#define NB_WM 2                 // waves over M
#define NB_WN 4                 // waves over N

__device__ __forceinline__ float nb_act(int act, float t) {
  if (act == 1) return 0.5f * t * (1.0f + erff(t * 0.70710678118654752f));
  if (act == 2) {
    const float u = 0.7978845608028654f * (t + 0.044715f * t * t * t);
    return 0.5f * t * (1.0f + tanhf(u));
  }
  return t;
}

// TM x TN WMMA tiles a wave; BM = 2*TM*16, BN = 4*TN*16.
template <int TM, int TN>
__global__ __launch_bounds__(256) void halo_gemm_nt_bias_kernel(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    const void* __restrict__ Bias, const unsigned short* __restrict__ Res,
    __hip_bfloat16* __restrict__ C, int M, int K, int N,
    int lda, int ldw, int ldc, int ldr, int act, int bias_f32, int has_res) {
  constexpr int BM = NB_WM * TM * 16, BN = NB_WN * TN * 16;
  constexpr int CPR = NB_BK / 8;                  // 16-byte chunks a staged row
  constexpr int AL = BM * CPR / 256, WL = BN * CPR / 256;
  static_assert(AL >= 1 && WL >= 1 && BM * CPR % 256 == 0 && BN * CPR % 256 == 0,
                "a tile is whole 16-byte chunks for every thread");
  __shared__ __attribute__((aligned(16))) unsigned short sA[BM * NB_PITCH];
  __shared__ __attribute__((aligned(16))) unsigned short sW[BN * NB_PITCH];

  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int wm = wave / NB_WN, wn = wave % NB_WN;
  constexpr unsigned kGroupM = 8;
  const unsigned per = kGroupM * gridDim.x, lin = blockIdx.y * gridDim.x + blockIdx.x;
  const unsigned first = lin / per * kGroupM, r = lin % per;
  const unsigned gm = min(gridDim.y - first, kGroupM);
  const int m0 = (int)(first + r % gm) * BM, n0 = (int)(r / gm) * BN;

  unsigned aoff[AL], woff[WL];
  unsigned short* adst[AL];
  unsigned short* wdst[WL];
#pragma unroll
  for (int i = 0; i < AL; ++i) {
    const int c = tid + 256 * i, rr = c / CPR, e = (c % CPR) * 8;
    const int gr = m0 + rr < M ? m0 + rr : M - 1;
    aoff[i] = (unsigned)gr * (unsigned)lda + (unsigned)e;
    adst[i] = sA + rr * NB_PITCH + e;
  }
#pragma unroll
  for (int i = 0; i < WL; ++i) {
    const int c = tid + 256 * i, rr = c / CPR, e = (c % CPR) * 8;
    const int gr = n0 + rr < N ? n0 + rr : N - 1;
    woff[i] = (unsigned)gr * (unsigned)ldw + (unsigned)e;
    wdst[i] = sW + rr * NB_PITCH + e;
  }

  const unsigned short* fa = sA + (wm * TM * 16 + (lane & 15)) * NB_PITCH;
  const unsigned short* fw = sW + (wn * TN * 16 + (lane & 15)) * NB_PITCH;

  v8f acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (v8f)(0.f);

  const int nk = (K + NB_BK - 1) / NB_BK;
  // A staged chunk is eight elements; chunks starting past K stage zeros. K is a multiple
  // of 8, so a chunk is whole or absent.
  auto load_tile = [&](int kt, halo_u32x4* ra, halo_u32x4* rw) {
    const unsigned k0 = (unsigned)kt * NB_BK;
#pragma unroll
    for (int i = 0; i < AL; ++i) {
      const int c = tid + 256 * i, e = (c % CPR) * 8;
      ra[i] = (k0 + (unsigned)e < (unsigned)K)
                  ? *(const halo_u32x4*)(A + aoff[i] + k0) : (halo_u32x4)0u;
    }
#pragma unroll
    for (int i = 0; i < WL; ++i) {
      const int c = tid + 256 * i, e = (c % CPR) * 8;
      rw[i] = (k0 + (unsigned)e < (unsigned)K)
                  ? *(const halo_u32x4*)(W + woff[i] + k0) : (halo_u32x4)0u;
    }
  };

  halo_u32x4 ra[AL], rw[WL];
  load_tile(0, ra, rw);

  for (int kt = 0; kt < nk; ++kt) {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < AL; ++i) *(halo_u32x4*)adst[i] = ra[i];
#pragma unroll
    for (int i = 0; i < WL; ++i) *(halo_u32x4*)wdst[i] = rw[i];
    __syncthreads();
    if (kt + 1 < nk) load_tile(kt + 1, ra, rw);
#pragma unroll
    for (int s = 0; s < NB_BK / 16; ++s) {
      v16bf af[TM], bf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) af[i] = fragP(fa + i * 16 * NB_PITCH + s * 16);
#pragma unroll
      for (int j = 0; j < TN; ++j) bf[j] = fragP(fw + j * 16 * NB_PITCH + s * 16);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = mma(af[i], bf[j], acc[i][j]);
      __builtin_amdgcn_sched_group_barrier(0x100, TM + TN, 0);    /* DS reads */
      __builtin_amdgcn_sched_group_barrier(0x008, TM * TN, 0);    /* WMMA */
    }
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
      const int mb = m0 + (wm * TM + i) * 16;
      float bv = 0.f;
      if (n < N)
        bv = bias_f32 ? ((const float*)Bias)[n]
                      : bf16_to_f32(((const unsigned short*)Bias)[n]);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = mb + accRow(lane, e);
        if (n < N && m < M) {
          float t = bf16_to_f32(f32_to_bf16(acc[i][j][e] + bv));
          if (act) t = bf16_to_f32(f32_to_bf16(nb_act(act, t)));
          if (has_res)
            t = bf16_to_f32(f32_to_bf16(bf16_to_f32(Res[(size_t)m * ldr + n]) + t));
          C[(size_t)m * ldc + n] = (__hip_bfloat16)t;
        }
      }
    }
}

static int gemm_nt_bias_launch(long a, long w, long bias, long res, long c, int M, int K, int N,
                               int lda, int ldw, int ldc, int ldr, int act, int bias_f32,
                               int has_res, long stream) {
  if (M < 1 || N < 1 || K < 8 || (K % 8)) return -1;
  if ((long long)M * lda >= (1ll << 32) || (long long)N * ldw >= (1ll << 32)) return -1;
  const long long g = (long long)((N + 127) / 128) * ((M + 127) / 128);
  const int bt = g >= 32 ? 128 : 64;
  dim3 grid((N + bt - 1) / bt, (M + bt - 1) / bt), block(256);
  auto Ap = (const unsigned short*)a; auto Wp = (const unsigned short*)w;
  auto Rp = (const unsigned short*)res; auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define NB_LAUNCH(TM, TN) hipLaunchKernelGGL((halo_gemm_nt_bias_kernel<TM, TN>), grid, block, 0, \
    st, Ap, Wp, (const void*)bias, Rp, Cp, M, K, N, lda, ldw, ldc, ldr, act, bias_f32, has_res)
  if (bt == 128) NB_LAUNCH(4, 2); else NB_LAUNCH(2, 1);
#undef NB_LAUNCH
  return (int)hipGetLastError();
}

// ============================================================== the plugin ABI entry point
extern "C" int halo_rad_gemm_nt_bias(const RadArgs* a, RadStream s) {
    enum { GF_A = 0, GF_W, GF_BIAS, GF_RES, GF_Y };
    const RadTensor* A = halo_opt(a, GF_A);
    const RadTensor* W = halo_opt(a, GF_W);
    const RadTensor* B = halo_opt(a, GF_BIAS);
    const RadTensor* R = halo_opt(a, GF_RES);
    const RadTensor* Y = halo_opt(a, GF_Y);
    if (!A || !W || !B || !Y) return HALO_NO(RAD_E_INVAL);
    if (A->dtype != RAD_BF16 || W->dtype != RAD_BF16 || Y->dtype != RAD_BF16)
        return HALO_NO(RAD_E_DTYPE);
    if (B->dtype != RAD_F32 && B->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    const bool has_res = R && R->data;
    if (has_res && R->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    if (!halo_contig(W) || !halo_contig(B)) return HALO_NO(RAD_E_STRIDE);

    const char* act_s = halo_gets(a, "act");
    int act = 0;
    if (act_s && *act_s) {
        if (!std::strcmp(act_s, "gelu")) act = 1;
        else if (!std::strcmp(act_s, "gelu_tanh")) act = 2;
        else if (std::strcmp(act_s, "none")) return HALO_NO(RAD_E_SCHEMA);
    }
    const char* dt = halo_gets(a, "dtype");
    if (dt && *dt && std::strcmp(dt, "bf16")) return HALO_NO(RAD_E_DTYPE);

    const long long K = halo_geti_or(a, "K", halo_dim(W, (int)W->rank - 1));
    const long long N = halo_geti_or(a, "N", K > 0 ? halo_numel(W) / K : 0);
    if (K <= 0 || N <= 0 || halo_numel(W) < N * K || halo_numel(B) < N)
        return HALO_NO(RAD_E_SHAPE);
    long long M = 0, Ma = 0, Mw = 0, Mr2 = 0;
    const long long ldc = halo_rows_pitch(Y, N, &M);
    const long long lda = halo_rows_pitch(A, K, &Ma);
    const long long ldw = halo_rows_pitch(W, K, &Mw);
    const long long ldr = has_res ? halo_rows_pitch(R, N, &Mr2) : 0;
    if (ldc < 0 || lda < 0 || ldw < 0 || ldr < 0) return HALO_NO(RAD_E_STRIDE);
    if (M <= 0 || Ma < M || Mw < N || (has_res && Mr2 < M)) return HALO_NO(RAD_E_SHAPE);
    if (!halo_aligned(A->data, 16)) return HALO_NO(RAD_E_ALIGN);

    const int e = gemm_nt_bias_launch(halo_addr(A), halo_addr(W), halo_addr(B),
                                      has_res ? halo_addr(R) : 0, halo_addr(Y),
                                      (int)M, (int)K, (int)N, (int)lda, (int)ldw, (int)ldc,
                                      (int)ldr, act, B->dtype == RAD_F32, has_res ? 1 : 0,
                                      (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : HALO_NO(e < 0 ? RAD_E_SHAPE : RAD_E_DEVICE);
}
