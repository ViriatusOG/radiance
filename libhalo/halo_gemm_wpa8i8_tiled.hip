/* Derived from libr4d/r4d_gemm_wpa8i8_tiled.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_gemm_wpa8i8_tiled.hip - gemm_w4a8i8_tiled / gemm_w5a8i8_tiled: the 4- and 5-bit
// asymmetric weight x int8 activation GEMM for prefill row counts (ParoQuant int4 / int5).
// C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with dequant = scale[n][k/128] * (code - zero[n][k/128]),
// the activation int8 with an f32 scale per (row, part, 128 of K).
//
// The tile machinery is halo_gemm_w4a8_tiled's (128 x 256 blocks, the weight's packed dwords
// staged through LDS, the activation staged row-major so a fragment is four consecutive
// dwords); what differs is the fragment and the fold:
//   * the weight byte is (code - zero), formed in the fragment by libr4d's three-op SWAR
//     subtract, ((codes | 0x80808080) - zero) ^ 0x80808080 per dword -- exact for codes and
//     zeros in 0..31, so the iu8 WMMA accumulates the corrected product in int32;
//   * BITS 5 takes the code's fifth bit from the plane after the nibbles (one dword per
//     (n-tile, 64 of K, lane); byte s = k step s, bit e = fragment element e), spread into
//     bit 4 of each byte as halo_gemm_w5a16_nt_m64 spreads it;
//   * the fold is per (row, group): acc += g * (w_scale[col] * a_scale[row][g]).
// A partitioned weight (pb1/pb2) is one launch per part over windows of the planes (mx_parts).
#include "halo.h"
#include "halo_gemm_args.h"
#include "halo_common.h"
#include "halo_gdn_wmma.h"   // accRow: the gfx11 accumulator row mapping
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

#define HALO_WPT_KPB 64

typedef int      v2i_t __attribute__((ext_vector_type(2)));
typedef int      v4i_t __attribute__((ext_vector_type(4)));
typedef int      v8i_t __attribute__((ext_vector_type(8)));
typedef float    v8f_t __attribute__((ext_vector_type(8)));
typedef unsigned u4v_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ v8i_t wpt_mmai(v4i_t a, v4i_t b, v8i_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
}

__device__ __forceinline__ v2i_t wpt_dequant8(unsigned w) {
  v2i_t r;
  r[0] = (int)((w << 4) & 0xF0F0F0F0u);
  r[1] = (int)(w & 0xF0F0F0F0u);
  return r;
}

/* Four code bytes (0..31) with the zero in every byte -> the four bytes of (code - zero). */
__device__ __forceinline__ unsigned wpt_sub_zero(unsigned codes4, unsigned zz) {
  return ((codes4 | 0x80808080u) - zz) ^ 0x80808080u;
}

__device__ __forceinline__ unsigned wpt_spread16(unsigned n) {   // 4 bits -> bit 4 of each byte
  return ((n * 0x00204081u) & 0x01010101u) << 4;
}

/* The fragment from this lane's dword and its cross-half partner's, as in
 * halo_gemm_w4a8_tiled, with the zero subtracted per byte. */
__device__ __forceinline__ v4i_t wpt_step(unsigned w, unsigned pw, int lane, int zero) {
  const bool hi = lane >> 4;
  const v2i_t lo = wpt_dequant8(hi ? pw : w);
  const v2i_t hd = wpt_dequant8(hi ? w : pw);
  const unsigned zz = (unsigned)zero * 0x01010101u;
  v4i_t r;
  r[0] = (int)wpt_sub_zero((unsigned)(lo[0] >> 4) & 0x0F0F0F0Fu, zz);
  r[1] = (int)wpt_sub_zero((unsigned)(hd[0] >> 4) & 0x0F0F0F0Fu, zz);
  r[2] = (int)wpt_sub_zero((unsigned)(lo[1] >> 4) & 0x0F0F0F0Fu, zz);
  r[3] = (int)wpt_sub_zero((unsigned)(hd[1] >> 4) & 0x0F0F0F0Fu, zz);
  return r;
}

/* BITS 5: hb (this lane's) and phb (the partner's) are the fifth-bit bytes of the two dwords.
 * Fragment dword d takes its fifth bits from hb's low nibble (d < 2: nibbles 2b) or high
 * nibble (d >= 2: nibbles 2b+1), as halo_gemm_w5a16_nt_m64's dequant5_pairs reads them. */
__device__ __forceinline__ v4i_t wpt_step5(unsigned w, unsigned pw, unsigned hb, unsigned phb,
                                           int lane, int zero) {
  const bool hi = lane >> 4;
  const v2i_t lo = wpt_dequant8(hi ? pw : w);
  const v2i_t hd = wpt_dequant8(hi ? w : pw);
  const unsigned ahb = hi ? phb : hb, bhb = hi ? hb : phb;
  const unsigned zz = (unsigned)zero * 0x01010101u;
  v4i_t r;
  r[0] = (int)wpt_sub_zero(((unsigned)(lo[0] >> 4) & 0x0F0F0F0Fu) | wpt_spread16(ahb & 15u), zz);
  r[1] = (int)wpt_sub_zero(((unsigned)(hd[0] >> 4) & 0x0F0F0F0Fu) | wpt_spread16(bhb & 15u), zz);
  r[2] = (int)wpt_sub_zero(((unsigned)(lo[1] >> 4) & 0x0F0F0F0Fu) | wpt_spread16(ahb >> 4), zz);
  r[3] = (int)wpt_sub_zero(((unsigned)(hd[1] >> 4) & 0x0F0F0F0Fu) | wpt_spread16(bhb >> 4), zz);
  return r;
}

// WM x WN waves, each owning TM x TN WMMA tiles. BM = WM*TM*16, BN = WN*TN*16. UNR k blocks
// (half-groups of 64) are unrolled per scale group of 128.
template <int WM, int WN, int TM, int TN, int BITS, int UNR>
__global__ __launch_bounds__(WM * WN * 32) void halo_gemm_wpa8i8_tiled_kernel(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Whi,
    const unsigned* __restrict__ Ws, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int lda, int as_rs, int n0g, int nend) {
  constexpr int BK = 64;                            // half a scale group
  constexpr int GRP = HALO_GEMM_W4A8_GROUP;
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
  constexpr int NTILE = BN / 16;
  constexpr int AW = BM * BK / 4;                   // A dwords per buffer
  constexpr int WW = NTILE * 128;                   // W dwords per buffer
  constexpr int HW = NTILE * 32;                    // fifth-bit dwords per buffer

  __shared__ unsigned lds_a[2][AW];
  __shared__ unsigned lds_w[2][WW];
  __shared__ unsigned lds_h[2][HW];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int m0 = blockIdx.y * BM, n0 = n0g + blockIdx.x * BN;
  const int kpb = K / HALO_WPT_KPB, nsz = K / GRP;
  const int ntlast = N / 16 - 1;
  constexpr int NT_ = WM * WN * 32;

  v8f_t acc[TN][TM];
#pragma unroll
  for (int j = 0; j < TN; j++)
#pragma unroll
    for (int i = 0; i < TM; i++) acc[j][i] = (v8f_t)(0.f);
  v8i_t g[TN][TM];
  unsigned scv[TN];
  int zero[TN];
#pragma unroll
  for (int j = 0; j < TN; j++) { scv[j] = 0; zero[j] = 0; }

#define ASW(r, c) ((r) * (BK / 4) + ((c) ^ ((((r) >> 1) & 7) << 1)))

  constexpr int APT = (AW + NT_ - 1) / NT_, WPT = WW / NT_;
  constexpr bool APAD = (AW % NT_) != 0;
  unsigned ra[APT], rw[WPT > 0 ? WPT : 1];
  constexpr int WPT4 = (WW / 4) / NT_;
  constexpr bool W128 = WPT4 > 0 && (WW / 4) % NT_ == 0;
  u4v_t rw4[WPT4 > 0 ? WPT4 : 1];
  constexpr int HPT = (HW + NT_ - 1) / NT_;
  unsigned rh[HPT > 0 ? HPT : 1];
  static_assert(W128 || (WPT > 0 && WW % NT_ == 0), "no W staging path tiles this block shape");
  auto issue = [&](int kk) {
    if (W128) {
#pragma unroll
      for (int x = 0; x < WPT4; x++) {
        const int t = tid + x * NT_, tl = t / 32, off4 = t % 32;
        const int tg = (n0 / 16 + tl) < ntlast ? (n0 / 16 + tl) : ntlast;
        rw4[x] = *(const u4v_t*)(Wq + (unsigned)((tg * kpb + kk / HALO_WPT_KPB) * 128 + off4 * 4));
      }
    }
#pragma unroll
    for (int x = 0; x < APT; x++) {
      const int t = tid + x * NT_, r = t / (BK / 4), c = t % (BK / 4);
      if (APAD && t >= AW) continue;
      const int mm = m0 + r < M ? m0 + r : M - 1;
      ra[x] = *(const unsigned*)(A + (size_t)mm * lda + kk + c * 4);
    }
#pragma unroll
    for (int x = 0; x < WPT && !W128; x++) {
      const int t = tid + x * NT_, tl = t / 128, off = t % 128;
      const int tg = (n0 / 16 + tl) < ntlast ? (n0 / 16 + tl) : ntlast;
      rw[x] = Wq[(unsigned)((tg * kpb + kk / HALO_WPT_KPB) * 128 + off)];
    }
    if (BITS == 5) {
#pragma unroll
      for (int x = 0; x < HPT; x++) {
        const int t = tid + x * NT_;
        if (t >= HW) continue;
        const int tl = t / 32, ln = t % 32;
        const int tg = (n0 / 16 + tl) < ntlast ? (n0 / 16 + tl) : ntlast;
        rh[x] = Whi[(unsigned)((tg * kpb + kk / HALO_WPT_KPB) * 32 + ln)];
      }
    }
  };
  auto commit = [&](int buf) {
    if (W128) {
#pragma unroll
      for (int x = 0; x < WPT4; x++) *(u4v_t*)(lds_w[buf] + (tid + x * NT_) * 4) = rw4[x];
    }
#pragma unroll
    for (int x = 0; x < APT; x++) {
      const int t = tid + x * NT_, r = t / (BK / 4), c = t % (BK / 4);
      if (APAD && t >= AW) continue;
      lds_a[buf][ASW(r, c)] = ra[x];
    }
#pragma unroll
    for (int x = 0; x < WPT && !W128; x++) lds_w[buf][tid + x * NT_] = rw[x];
    if (BITS == 5) {
#pragma unroll
      for (int x = 0; x < HPT; x++) {
        const int t = tid + x * NT_;
        if (t < HW) lds_h[buf][t] = rh[x];
      }
    }
  };
#define HALO_WS(g_, j_) Ws[(unsigned)((((n0 / 16 + wn * TN + (j_)) < ntlast ? (n0 / 16 + wn * TN + (j_)) : ntlast) * nsz + (g_)) * 16 + (lane & 15))]

  issue(0);
  commit(0);
  __syncthreads();

  for (int kg = 0, it = 0; kg < K; kg += BK * UNR) {
#pragma unroll
    for (int h = 0; h < UNR; h++, it++) {
    const int k0 = kg + h * BK;
    const int buf = it & 1;
    if (k0 + BK < K) issue(k0 + BK);

    const bool first_half = (k0 % GRP) == 0;
    if (first_half) {
#pragma unroll
      for (int j = 0; j < TN; j++) {
        scv[j] = HALO_WS(k0 / GRP, j);
        zero[j] = (int)(-__half2float(__ushort_as_half((unsigned short)(scv[j] >> 16))) - 1024.0f);
      }
    }
#pragma unroll
    for (int u = 0; u < BK / 16; u++) {
      v4i_t af[TM];
#pragma unroll
      for (int i = 0; i < TM; i++) {
        const int row = (wm * TM + i) * 16 + (lane & 15);
#pragma unroll
        for (int d = 0; d < 4; d++)
          af[i][d] = (int)lds_a[buf][ASW(row, u * 4 + d)];
      }
#pragma unroll
      for (int j = 0; j < TN; j++) {
        const u4v_t wv = *(const u4v_t*)(lds_w[buf] + (size_t)(wn * TN + j) * 128 + lane * 4);
        const u4v_t pv = *(const u4v_t*)(lds_w[buf] + (size_t)(wn * TN + j) * 128 + (lane ^ 16) * 4);
        v4i_t bf;
        if (BITS == 5) {
          const unsigned hbw = lds_h[buf][(wn * TN + j) * 32 + lane];
          const unsigned hbp = lds_h[buf][(wn * TN + j) * 32 + (lane ^ 16)];
          bf = wpt_step5(wv[u], pv[u], (hbw >> (8 * u)) & 0xffu, (hbp >> (8 * u)) & 0xffu,
                         lane, zero[j]);
        } else {
          bf = wpt_step(wv[u], pv[u], lane, zero[j]);
        }
#pragma unroll
        for (int i = 0; i < TM; i++)
          g[j][i] = (first_half && u == 0) ? wpt_mmai(af[i], bf, (v8i_t)(0))
                                           : wpt_mmai(af[i], bf, g[j][i]);
      }
    }

    if ((k0 + BK) % GRP == 0) {                     // the group closed: fold it into the float acc
      float sc[TN];
#pragma unroll
      for (int j = 0; j < TN; j++)
        sc[j] = __half2float(__ushort_as_half((unsigned short)(scv[j] & 0xFFFFu)));
#pragma unroll
      for (int i = 0; i < TM; i++) {
        float as[8];
        const int rb = m0 + (wm * TM + i) * 16;
#pragma unroll
        for (int e = 0; e < 8; e++) {
          int m = rb + 2 * e + (lane >> 4);
          m = m < M ? m : M - 1;
          as[e] = Ascale[(size_t)m * as_rs + k0 / GRP];
        }
#pragma unroll
        for (int j = 0; j < TN; j++)
#pragma unroll
          for (int e = 0; e < 8; e++)
            acc[j][i][e] = fmaf((float)g[j][i][e], sc[j] * as[e], acc[j][i][e]);
      }
    }
    if (k0 + BK < K) commit(buf ^ 1);
    __syncthreads();
    }
  }
#undef HALO_WS
#undef ASW

#pragma unroll
  for (int j = 0; j < TN; j++)
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
        const int m = m0 + (wm * TM + i) * 16 + accRow(lane, e);
        if (n < nend && m < M)
          C[(size_t)m * N + n] = (__hip_bfloat16)acc[j][i][e];
      }
}

template <int BITS>
static void wpa8i8_launch(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int lda, int as_rs, int n0, int n1, long stream) {
  constexpr int BM = 128, BN = 128;
  const unsigned* Whi = BITS == 5 ? (const unsigned*)wq + (size_t)N * K / 8 : nullptr;
  dim3 grid((n1 - n0 + BN - 1) / BN, (M + BM - 1) / BM), block(512);
  hipLaunchKernelGGL((halo_gemm_wpa8i8_tiled_kernel<4, 4, 2, 2, BITS, 2>), grid, block, 0,
                     (hipStream_t)stream, (const signed char*)a, (const float*)ascale,
                     (const unsigned*)wq, Whi, (const unsigned*)ws, (__hip_bfloat16*)c,
                     M, K, N, lda, as_rs, n0, n1);
  if (hipGetLastError() != hipSuccess)
    throw std::runtime_error("r4d gemm_wpa8i8_tiled: launch failed");
}

template <int BITS>
static int rad_wpa8i8_tiled(const RadArgs* a, RadStream s, const char* dtype) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1, /*parts=*/1);
    if (rc != RAD_OK) return rc;
    if (!g.ascale) return HALO_NO(RAD_E_INVAL);
    MxParts mp;
    rc = mx_parts(a, &g, &mp);
    if (rc != RAD_OK) return rc;
    const char* dt = halo_gets(a, "dtype");
    if (!dt || std::strcmp(dt, dtype)) return HALO_NO(RAD_E_DTYPE);
    if ((g.K % 128) || (g.N % 16)) return HALO_NO(RAD_E_SHAPE);
    const long long PK = mp.lda > 0 ? mp.lda : g.K;          // activation row pitch (elements)
    const int PG = (int)(PK / 128);                          // scale groups a row, all parts
    for (int p = 0; p < mp.n; ++p) {
        HALO_TRY("gemm_wpa8i8_tiled",
            wpa8i8_launch<BITS>(halo_addr(g.a) + (long)p * g.K,
                                halo_addr(g.ascale) + (long)p * (g.K / 128) * (long)sizeof(float),
                                halo_addr(g.b), halo_addr(g.bscale), halo_addr(g.y),
                                (int)g.M, (int)g.K, (int)g.N, (int)PK, PG,
                                (int)mp.n0[p], (int)mp.n1[p], (long)(uintptr_t)s));
    }
    return RAD_OK;
}

extern "C" int halo_rad_gemm_w4a8i8_tiled(const RadArgs* a, RadStream s) {
    return rad_wpa8i8_tiled<4>(a, s, "w4a8i8_asym");
}
extern "C" int halo_rad_gemm_w5a8i8_tiled(const RadArgs* a, RadStream s) {
    return rad_wpa8i8_tiled<5>(a, s, "w5a8i8");
}
