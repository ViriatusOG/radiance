/* Derived from libr4d/r4d_gemm_w4a8_tiled.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_gemm_w4a8_tiled.hip - halo_gemm_w4a8_tiled: the W4A8 GEMM for prefill batches.
//
// halo_gemm_w4a8_nt_m64 covers at most 64 rows per block, so at M=4096 each of 64 M-blocks re-reads
// the weight slice and it holds a flat ~105 TF/s. The target is the block-fp8 prefill GEMM
// (~204 TF/s on the same shapes).
//
// Tile shape from traffic: a BM x BN tile reads M/BM copies of the weight and N/BN of the
// activation. For gate_proj at M=4096 (~600 GB/s DRAM, ~1049 GB/s from the 64 MB MALL):
//
//     64 x 128 (skinny)   4352 blocks   2.85 GB   2.7 ms
//     128 x 256           1088 blocks   1.43 GB   1.4 ms
//     256 x 256            544 blocks   1.07 GB   1.0 ms
//
// 256x256 needs TM=TN=4 per wave, but the group loop keeps the int32 and float accumulators live
// together (256 VGPRs), so TM*TN <= 8 and the tile is 128 x 256.
//
// BK is half the scale group, double-buffered. A single buffer at BK = the group exposes every
// global-load latency and is slower than the skinny kernel: the loop is latency-bound. Double
// buffering must stay under 32 KB of LDS to fit two workgroups per CU; at BK = 64 it is 8 KB of A
// and 8 KB of W per buffer. The int32 accumulator carries across the two iterations of a group.
//
// Padding A's LDS row stride is slower at M = 4096 (one extra dword per row exceeds 32 KB and
// drops a CU to one workgroup); the XOR swizzle below avoids the bank conflicts instead.
//
// Layouts are the skinny kernel's: the packed weight is in WMMA fragment order (lane L of a
// (tile, k-block) owns dwords [4L, 4L+3] of a 128-dword run), so staging is a contiguous copy, and
// the quantiser pre-shuffles the activation within each k step. Both kernels share one weight and
// one activation.
#include "halo.h"
#include "halo_gemm_args.h"
#include "halo_common.h"
#include "halo_gdn_wmma.h"   // accRow: the gfx11 accumulator row mapping
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

#define HALO_TILED_KPB 64

typedef int      v2i_t __attribute__((ext_vector_type(2)));
typedef int      v4i_t __attribute__((ext_vector_type(4)));
typedef int      v8i_t __attribute__((ext_vector_type(8)));
typedef float    v8f_t __attribute__((ext_vector_type(8)));
typedef unsigned u4v_t __attribute__((ext_vector_type(4)));
typedef float    v4f_t __attribute__((ext_vector_type(4)));

// BSIGNED=false reads the unpacked byte as 16*nibble (0..240) instead of 16*twos(nibble): the
// asymmetric grid, whose per-group zero is corrected with a row sum of the activation.
template <bool BSIGNED>
__device__ __forceinline__ v8i_t tmmai(v4i_t a, v4i_t b, v8i_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, BSIGNED, b, c, false);
}

__device__ __forceinline__ v2i_t tdequant8(unsigned w) {
  v2i_t r;
  r[0] = (int)((w << 4) & 0xF0F0F0F0u);
  r[1] = (int)(w & 0xF0F0F0F0u);
  return r;
}

/* The weight keeps libr4d's fragment order in LDS ([tile][lane][4 steps]): a lane's dword for a
 * step holds K quarters (0,2) in the low half-wave and (1,3) in the high, so the gfx11 fragment
 * unpacks this lane's dword and its cross-half partner's and interleaves the quarter dwords. */
__device__ __forceinline__ v4i_t tdequant_step(unsigned w, unsigned pw, int lane) {
  const bool hi = lane >> 4;
  const v2i_t lo = tdequant8(hi ? pw : w);
  const v2i_t hd = tdequant8(hi ? w : pw);
  v4i_t r;
  r[0] = lo[0]; r[1] = hd[0]; r[2] = lo[1]; r[3] = hd[1];
  return r;
}

// WM x WN waves, each owning TM x TN WMMA tiles. BM = WM*TM*16, BN = WN*TN*16. UNR k blocks are
// unrolled per scale group.
//
// The symmetric grid seeds each group's WMMA accumulator with 0x4B400000 (the magic bias), which
// removes the int32 -> float conversion from the group fold. For |x| < 2^22,
//
//     float(x) == __uint_as_float(0x4B400000 + x) - 12582912.0f
//
// since 0x4B400000 is 1.5*2^23 with unit mantissa steps. A group's int32 sum is at most
// 128 * 127 * 128 = 2,080,768. The -12582912*s per group is summed per column and subtracted once
// at the end. The asymmetric grid converts plainly: with the bias, z16, the row sums and csum on top
// of 128 accumulator registers, it spills.
// No minimum-occupancy bound: libr4d's 8 waves a SIMD caps a lane at 192 VGPRs on gfx1151 and
// spilled most arms (the 4x2 tile 158-165 slots); unbounded, the arms run 1.1-2.4x faster.
// PART: as in halo_gemm_w8a8_tiled.hip, the waves of the M-cut last row of blocks skip the WMMAs
// and folds of their padding fragments; only a ragged M's launch instantiates it.
template <int WM, int WN, int TM, int TN, int ASYM, int UNR, bool PART = false>
__global__ __launch_bounds__(WM * WN * 32) void halo_gemm_w4a8_tiled_kernel(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Ws,
    const float* __restrict__ Asum, __hip_bfloat16* __restrict__ C, int M, int K, int N) {
  constexpr int BK = 64;                            // half a scale group
  constexpr int GRP = HALO_GEMM_W4A8_GROUP;
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
  constexpr int NTILE = BN / 16;
  constexpr int AW = BM * BK / 4;                   // A dwords per buffer
  constexpr int WW = NTILE * 128;                   // W dwords per buffer (BK == KPB, so one run)
  constexpr bool BIAS = !ASYM;

  __shared__ unsigned lds_a[2][AW];
  __shared__ unsigned lds_w[2][WW];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  // Groups of 8 activation bands, M fastest (halo_gemm_bf16_nt_tiled's order), so the bands of a
  // column block share its weight through the cache.
  constexpr unsigned kGroupM = 8;
  const unsigned per = kGroupM * gridDim.x, lin = blockIdx.y * gridDim.x + blockIdx.x;
  const unsigned first = lin / per * kGroupM, rr = lin % per;
  const unsigned gm = min(gridDim.y - first, kGroupM);
  const int m0 = (int)(first + rr % gm) * BM, bx = (int)(rr / gm), n0 = bx * BN;
  const int live = PART ? __builtin_amdgcn_readfirstlane(
                              min(TM, max(0, (M + 15) / 16 - (m0 / 16 + wm * TM))))
                        : TM;
  const int kpb = K / HALO_TILED_KPB, nsz = K / GRP;
  constexpr int NT_ = WM * WN * 32;

  v8f_t acc[TN][TM];
#pragma unroll
  for (int j = 0; j < TN; j++)
#pragma unroll
    for (int i = 0; i < TM; i++) acc[j][i] = (v8f_t)(0.f);
  v8i_t g[TN][TM];
  float csum[TN];
#pragma unroll
  for (int j = 0; j < TN; j++) csum[j] = 0.f;
  unsigned scv[TN];

  // XOR swizzle instead of padding (padding would exceed 32 KB and drop a CU to one workgroup).
  // A row is BK/4 = 16 dwords over 32 banks, so the sixteen lanes of a fragment read (differing
  // only in the row) would collide. Bank = (r*16 + (c ^ mask)) % 32, so the row only contributes
  // its low bit: `(r & 7) << 1` leaves rows r and r+8 colliding (2-way conflict on every A read),
  // while `((r >> 1) & 7) << 1` is conflict-free. Both stay inside the row ((row & 15) << 1 would
  // reach 30 and corrupt the next row) and even (for ds_load_b64 alignment). The A LDS read is about
  // a third of the kernel at decode tiles. The asymmetric kernel keeps the 8-row form.
#define ASW(r, c) ((r) * (BK / 4) + ((c) ^ ((((ASYM ? (r) : ((r) >> 1)) & 7) << 1))))

  // Prefetch into registers, compute off the other buffer, then commit to LDS, so the load wait
  // lands after the WMMA work (a prefetch that writes LDS directly would stall the compute; >2x).
  // A staging may be ragged (a predicate on the last pass), so BM need not be a multiple of the
  // workgroup width (e.g. BM = 80, 112).
  constexpr int APT = (AW + NT_ - 1) / NT_, WPT = WW / NT_;
  constexpr bool APAD = (AW % NT_) != 0;
  unsigned ra[APT], rw[WPT > 0 ? WPT : 1];
  // The weight moves as one global_load_b128 and one ds_store_b128 per thread where that tiles the
  // block (four threads cover a 16-dword row), else as dwords.
  constexpr int WPT4 = (WW / 4) / NT_;
  constexpr bool W128 = WPT4 > 0 && (WW / 4) % NT_ == 0;
  u4v_t rw4[WPT4 > 0 ? WPT4 : 1];
  static_assert(W128 || (WPT > 0 && WW % NT_ == 0), "no W staging path tiles this block shape");
  // Global indices are 32-bit: a 64-bit address costs four extra instructions per load.
  auto issue = [&](int kk) {
    if (W128) {
#pragma unroll
      for (int x = 0; x < WPT4; x++) {
        const int t = tid + x * NT_, tl = t / 32, off4 = t % 32;
        rw4[x] = *(const u4v_t*)(Wq + (unsigned)(((n0 / 16 + tl) * kpb + kk / HALO_TILED_KPB) *
                                                     128 + off4 * 4));
      }
    }
#pragma unroll
    for (int x = 0; x < APT; x++) {
      const int t = tid + x * NT_, r = t / (BK / 4), c = t % (BK / 4);
      if (APAD && t >= AW) continue;
      const int mm = m0 + r < M ? m0 + r : M - 1;
      ra[x] = *(const unsigned*)(A + (unsigned)(mm * K + kk + c * 4));
    }
#pragma unroll
    for (int x = 0; x < WPT && !W128; x++) {
      const int t = tid + x * NT_, tl = t / 128, off = t % 128;
      rw[x] = Wq[(unsigned)(((n0 / 16 + tl) * kpb + kk / HALO_TILED_KPB) * 128 + off)];
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
  };
  // The group's scale dword: the f16 scale, and for the asymmetric grid the f16 of -(1024 + zero)
  // in the high half. 32-bit index: the table has at most (N/16)*(K/GRP)*16 elements.
#define HALO_WS(g_, j_) Ws[(unsigned)(((n0 / 16 + wn * TN + (j_)) * nsz + (g_)) * 16 + (lane & 15))]

  issue(0);
  commit(0);
  __syncthreads();

  // Unroll the K loop over the group (UNR = GRP/BK) so the group-boundary tests are compile-time per
  // copy; a dynamic branch around the epilogue stops the scheduler interleaving it with the WMMA.
  // Worth several percent.
  for (int kg = 0, it = 0; kg < K; kg += BK * UNR) {
#pragma unroll
    for (int h = 0; h < UNR; h++, it++) {
    const int k0 = kg + h * BK;
    const int buf = it & 1;
    if (k0 + BK < K) issue(k0 + BK);                // loads in flight across the WMMA work below

    const bool first_half = (k0 % GRP) == 0;
    // Fetch the group scale at the start of the group. Every wave reaches the boundary on the same
    // k0, so a load at the point of use stalls them all at once. Keep the raw dword and convert at
    // the group close: converting here would put the wait right after the load.
    if ((k0 % GRP) == 0) {
#pragma unroll
      for (int j = 0; j < TN; j++)
        scv[j] = HALO_WS(k0 / GRP, j);
    }
#pragma unroll
    for (int u = 0; u < BK / 16; u++) {
      v4i_t af[TM];
#pragma unroll
      for (int i = 0; i < TM; i++) {
        /* The A tile is staged row-major: the fragment is four consecutive LDS dwords. */
        const int row = (wm * TM + i) * 16 + (lane & 15);
#pragma unroll
        for (int d = 0; d < 4; d++)
          af[i][d] = (int)lds_a[buf][ASW(row, u * 4 + d)];
      }
#pragma unroll
      for (int j = 0; j < TN; j++) {
        const u4v_t wv = *(const u4v_t*)(lds_w[buf] + (size_t)(wn * TN + j) * 128 + lane * 4);
        const u4v_t pv = *(const u4v_t*)(lds_w[buf] + (size_t)(wn * TN + j) * 128 + (lane ^ 16) * 4);
        const v4i_t bf = tdequant_step(wv[u], pv[u], lane);
#pragma unroll
        for (int i = 0; i < TM; i++) {
          if (PART && i >= live) continue;              // wave-uniform: padding fragments past M
          g[j][i] = (first_half && u == 0)
                        ? tmmai<!ASYM>(af[i], bf, (v8i_t)(BIAS ? 0x4B400000 : 0))
                        : tmmai<!ASYM>(af[i], bf, g[j][i]);
        }
      }
    }

    if ((k0 + BK) % GRP == 0) {                     // the group closed: fold it into the float acc
      float sc[TN];
#pragma unroll
      for (int j = 0; j < TN; j++)
        sc[j] = __half2float(__ushort_as_half((unsigned short)(scv[j] & 0xFFFFu)));
      if (BIAS) {
#pragma unroll
        for (int j = 0; j < TN; j++) csum[j] += sc[j];
      }
      // Asymmetric grid: the group contributes s * (sum_k a_k q_k - z * sum_k a_k), using the
      // per-row per-group activation sums from halo_quant_act_i8_asum, laid out [G][Mpad] so a
      // lane's eight rows (m0 + (wm*TM+i)*16 + accRow(lane, e)) are gathered below. The zero is the
      // scale dword's high half, the f16 of -(1024 + z) (exact). One extra fma per element; the
      // loop is i-outer so only one tile's eight sums are live.
      float z16[TN];
      if (ASYM) {
#pragma unroll
        for (int j = 0; j < TN; j++) {
          const float nz = __half2float(__ushort_as_half((unsigned short)(scv[j] >> 16)));
          z16[j] = 16.f * (-nz - 1024.f);
        }
      }
      const int Mpad = ((M + 15) / 16) * 16;
#pragma unroll
      for (int i = 0; i < TM; i++) {
        if (PART && i >= live) continue;
        float as[8];
        if (ASYM) {
          /* The lane's eight rows stride by two through the tile on this part: load the
           * tile's sixteen sums and pick from registers. */
          int rb = m0 + (wm * TM + i) * 16;
          rb = rb < Mpad - 16 ? rb : Mpad - 16;        // rows past M are stored away, not read
          const v4f_t* pv = (const v4f_t*)(Asum + (size_t)(k0 / GRP) * Mpad + rb);
          float av[16];
#pragma unroll
          for (int t = 0; t < 4; t++) {
            const v4f_t v = pv[t];
#pragma unroll
            for (int e = 0; e < 4; e++) av[4 * t + e] = v[e];
          }
#pragma unroll
          for (int e = 0; e < 8; e++) as[e] = av[2 * e + (lane >> 4)];
        }
#pragma unroll
        for (int j = 0; j < TN; j++)
#pragma unroll
          for (int e = 0; e < 8; e++) {
            float v = BIAS ? __uint_as_float((unsigned)g[j][i][e]) : (float)g[j][i][e];
            if (ASYM) v = fmaf(-z16[j], as[e], v);
            acc[j][i][e] = fmaf(v, sc[j], acc[j][i][e]);
          }
      }
    }
    if (k0 + BK < K) commit(buf ^ 1);               // the wait for those loads lands here
    __syncthreads();
    }
  }
#undef HALO_WS
#undef ASW

  if (BIAS) {
#pragma unroll
    for (int j = 0; j < TN; j++) {
      const float d = -12582912.0f * csum[j];
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int e = 0; e < 8; e++) acc[j][i][e] += d;
    }
  }

  // The flat predicated store beat LDS transposes (bf16 or packed dwords) and hand-hoisted Ascale /
  // row-offset variants: the epilogue is not bandwidth-bound (a cache-resident output takes the same
  // time), and with j innermost the stores stride the register file and share one guard.
#pragma unroll
  for (int j = 0; j < TN; j++)
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
        const int m = m0 + (wm * TM + i) * 16 + accRow(lane, e);
        if (n < N && m < M)
          // 0.0625 undoes the 16 from the unpack's high-nibble placement.
          C[(size_t)m * N + n] = (__hip_bfloat16)(acc[j][i][e] * Ascale[m] * 0.0625f);
      }
}

// CANPART: this arm instantiates the PART kernel for a ragged M (the default arm only).
template <int WM, int WN, int TM, int TN, int ASYM, int UNR, bool CANPART = false>
static void halo_tiled_launch(long a, long ascale, long wq, long ws, long asum, long c, int M, int K,
                             int N, long stream) {
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
  if (N % BN)
    throw std::runtime_error("r4d gemm_w4a8_tiled: N is not a multiple of this variant's block");
  dim3 grid(N / BN, (M + BM - 1) / BM), block(WM * WN * 32);
  if (CANPART && M % BM)
    hipLaunchKernelGGL((halo_gemm_w4a8_tiled_kernel<WM, WN, TM, TN, ASYM, UNR, CANPART>), grid,
                       block, 0, (hipStream_t)stream, (const signed char*)a, (const float*)ascale,
                       (const unsigned*)wq, (const unsigned*)ws, (const float*)asum,
                       (__hip_bfloat16*)c, M, K, N);
  else
    hipLaunchKernelGGL((halo_gemm_w4a8_tiled_kernel<WM, WN, TM, TN, ASYM, UNR>), grid, block, 0,
                       (hipStream_t)stream, (const signed char*)a, (const float*)ascale,
                       (const unsigned*)wq, (const unsigned*)ws, (const float*)asum,
                       (__hip_bfloat16*)c, M, K, N);
}

static void halo_gemm_w4a8_tiled_impl(long a, long ascale, long wq, long ws, long asum, long c,
                                     int M, int K, int N, long stream, int variant, bool asym) {
  if (N % 256) throw std::runtime_error("r4d gemm_w4a8_tiled: N must be a multiple of 256");
  if (K % HALO_GEMM_W4A8_GROUP)
    throw std::runtime_error("r4d gemm_w4a8_tiled: K must be a multiple of the scale group");
  if (M < 1) throw std::runtime_error("r4d gemm_w4a8_tiled: M must be positive");
  // The unroll covers exactly one scale group: less leaves the boundary dynamic (~10% slower), more
  // doubles live ranges (over 2x slower). The K check above makes every K a whole number of them.
  static_assert(HALO_GEMM_W4A8_GROUP % HALO_TILED_KPB == 0, "the scale group must be whole k blocks");
  constexpr int UNRG = HALO_GEMM_W4A8_GROUP / HALO_TILED_KPB;
#define HALO_TILED_LAUNCH(...) \
  halo_tiled_launch<__VA_ARGS__>(a, ascale, wq, ws, asum, c, M, K, N, stream)
  // The unroll loses on the asymmetric path (186 VGPRs), so it keeps UNR = 1, and it has one tile.
  if (asym) {
    HALO_TILED_LAUNCH(2, 8, 4, 2, 1, 1);
    return;
  }
  // The arms tile_bmbn in halo_rows.cpp lists, each with the (BM, BN) it records there. 512-thread
  // tiles are <2, 8, TM, TN> (or <1, 16, ...> for an odd BM at BN 256); at 256 threads <1, 8, TM, 1>
  // gives BM = 16*TM for any TM, so the block can be the smallest multiple of 16 covering M.
  // Staging constrains shapes (W needs WM | 4*TN, A needs WN | 8*TM).
  switch (variant) {
    case 6:   HALO_TILED_LAUNCH(2, 8, 2, 2, 0, UNRG); break;   // BM 64,  BN 256
    case 10:  HALO_TILED_LAUNCH(2, 8, 3, 2, 0, UNRG); break;   // BM 96,  BN 256
    case 12:  HALO_TILED_LAUNCH(2, 8, 3, 1, 0, UNRG); break;   // BM 96,  BN 128
    case 57:  HALO_TILED_LAUNCH(2, 8, 5, 1, 0, UNRG); break;   // BM 160, BN 128
    case 59:  HALO_TILED_LAUNCH(2, 8, 4, 1, 0, UNRG, true); break;   // BM 128, BN 128
    case 114: HALO_TILED_LAUNCH(1, 8, 5, 1, 0, UNRG); break;   // BM 80,  BN 128
    case 124: HALO_TILED_LAUNCH(1, 8, 4, 1, 0, UNRG); break;   // BM 64,  BN 128
    case 125: HALO_TILED_LAUNCH(1, 8, 7, 1, 0, UNRG); break;   // BM 112, BN 128
    case 126: HALO_TILED_LAUNCH(1, 8, 10, 1, 0, UNRG); break;  // BM 160, BN 128
    case 128: HALO_TILED_LAUNCH(1, 8, 2, 1, 0, UNRG); break;   // BM 32,  BN 128
    case 129: HALO_TILED_LAUNCH(1, 8, 3, 1, 0, UNRG); break;   // BM 48,  BN 128
    case 138: HALO_TILED_LAUNCH(1, 16, 3, 1, 0, UNRG); break;  // BM 48,  BN 256
    case 139: HALO_TILED_LAUNCH(1, 16, 7, 1, 0, UNRG); break;  // BM 112, BN 256
    case 146: HALO_TILED_LAUNCH(1, 8, 9, 1, 0, UNRG); break;   // BM 144, BN 128
    case 900: HALO_TILED_LAUNCH(2, 8, 8, 1, 0, UNRG); break;  // BM 256, BN 128 (prefill arm 14)
    // The untuned default is arm 59's tile, 2x8 waves of 4x1 tiles: on gfx1151 the 4x2 tile
    // libr4d defaults to (BM 128 x BN 256) holds 128 accumulators a lane and runs at 15-19 TFLOP/s
    // against 28-30 here (--perfw8 sweep at N 16384 K 5120 M 2048, N 5120 K 17408 / 5120 M 512).
    // Every arm accumulates each output over K in the same order, so the bytes do not move.
    default:  HALO_TILED_LAUNCH(2, 8, 4, 1, 0, UNRG, true); break;
  }
#undef HALO_TILED_LAUNCH
}

void halo_gemm_w4a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                         long stream, int variant) {
  halo_gemm_w4a8_tiled_impl(a, ascale, wq, ws, 0, c, M, K, N, stream, variant, false);
}

// Asymmetric grid: the nibble is read unsigned and the per-group zero is corrected with the per-row
// per-group activation sums from halo_quant_act_i8_asum ([K/group][Mpad], Mpad = ceil(M/16)*16).
void halo_gemm_w4a8_asym_tiled(long a, long ascale, long wq, long ws, long asum, long c, int M,
                              int K, int N, long stream, int variant) {
  if (!asum) throw std::runtime_error("r4d gemm_w4a8_asym_tiled: asum is required");
  halo_gemm_w4a8_tiled_impl(a, ascale, wq, ws, asum, c, M, K, N, stream, variant, true);
}

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in halo_gemm_args.h; see libhalo/halo_args.h.
/* ------------------------------------------------------------------ the large-M kernels
 *
 * These take one opaque tile variant; see tile_ok in halo_gemm_args.h for the N % BN check. */

extern "C" int halo_rad_gemm_w4a8_tiled(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (!g.ascale) return HALO_NO(RAD_E_INVAL);
    if (g.N % 256) return HALO_NO(RAD_E_SHAPE);                       /* the entry point throws on this */
    if (g.K % HALO_GEMM_W4A8_GROUP) return HALO_NO(RAD_E_SHAPE);
    HaloTileCfg tv;
    halo_cfg_tiled(a, &tv);
    const HaloTileCfg* t = &tv;
    rc = tile_ok(t, g.N);
    if (rc != RAD_OK) return rc;

    if (is_asym(a)) {
        if (!g.asum) return HALO_NO(RAD_E_INVAL);
        HALO_TRY("gemm_w4a8_asym_tiled",
            halo_gemm_w4a8_asym_tiled(halo_addr(g.a), halo_addr(g.ascale), halo_addr(g.b),
                                     halo_addr(g.bscale), halo_addr(g.asum), halo_addr(g.y),
                                     (int)g.M, (int)g.K, (int)g.N, (long)(uintptr_t)s,
                                     t->variant));
        return RAD_OK;
    }
    HALO_TRY("gemm_w4a8_tiled",
        halo_gemm_w4a8_tiled(halo_addr(g.a), halo_addr(g.ascale), halo_addr(g.b), halo_addr(g.bscale),
                            halo_addr(g.y), (int)g.M, (int)g.K, (int)g.N, (long)(uintptr_t)s,
                            t->variant));
    return RAD_OK;
}

/* The prefill row: the same packed weight and pre-shuffled activation as the tiled row, so on
 * gfx1151 it runs on the tiled kernel above. Each prefill arm names a (BM, BN); arms sharing a
 * shape share the tiled arm of that shape. */
extern "C" int halo_rad_gemm_w4a8_prefill(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (!g.ascale) return HALO_NO(RAD_E_INVAL);
    if (g.N % 256) return HALO_NO(RAD_E_SHAPE);
    if (g.K % HALO_GEMM_W4A8_GROUP) return HALO_NO(RAD_E_SHAPE);
    if (is_asym(a)) return HALO_NO(RAD_E_INVAL);   /* no asymmetric prefill grid in libr4d either */
    HaloTileCfg pv;
    halo_cfg_prefill(a, &pv);
    const HaloTileCfg* t = &pv;
    rc = tile_ok(t, g.N);
    if (rc != RAD_OK) return rc;
    int variant = 0;                                /* BM 128 x BN 256, the default shape */
    if      (t->bm == 64  && t->bn == 256) variant = 6;
    else if (t->bm == 96  && t->bn == 256) variant = 10;
    else if (t->bm == 128 && t->bn == 128) variant = 59;
    else if (t->bm == 256 && t->bn == 128) variant = 900;
    HALO_TRY("gemm_w4a8_prefill",
        halo_gemm_w4a8_tiled(halo_addr(g.a), halo_addr(g.ascale), halo_addr(g.b), halo_addr(g.bscale),
                            halo_addr(g.y), (int)g.M, (int)g.K, (int)g.N, (long)(uintptr_t)s,
                            variant));
    return RAD_OK;
}

/* The raw-pointer prefill entry libr4d exports beside the rows (halo.h): on gfx1151 the
 * prefill arms run on the tiled kernel, as the prefill row above does. Negative codes,
 * as libr4d's: -1 a null pointer, -2 a non-positive extent, -4 a K not a whole scale
 * group or an N the tile cannot take, -6 the launch failed. */
extern "C" void halo_gemm_w4a8_prefill(long a, long ascale, long wq, long ws, long c, int M,
                                       int K, int N, long stream, int variant) {
    if (!a || !ascale || !wq || !ws || !c || M <= 0 || K <= 0 || N <= 0) return;
    if ((K % HALO_GEMM_W4A8_GROUP) || (N % 256)) return;
    /* libr4d's prefill arms name (BM, BN) like the tiled arms but number them
     * differently; map by shape, as the prefill row above does. The remaining
     * prefill arms are BM 128 x BN 256 with option bits, the tiled default. */
    int tv = 0;
    switch (variant) {
        case 8: case 13: tv = 6; break;    /* BM 64,  BN 256 */
        case 9:          tv = 10; break;   /* BM 96,  BN 256 */
        case 10:         tv = 59; break;   /* BM 128, BN 128 */
        case 14:         tv = 900; break;  /* BM 256, BN 128 */
        default:         tv = 0; break;    /* BM 128, BN 256 */
    }
    halo_gemm_w4a8_tiled(a, ascale, wq, ws, c, M, K, N, stream, tv);
}
