/* Derived from libr4d/r4d_gemm_mxfp4a8_tiled.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_gemm_mxfp4a8_tiled.hip - halo_gemm_mxfp4a8_tiled: the OCP-MXFP4 x fp8 GEMM for prefill row
// counts, gfx1151. C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with the operands, weight layout and fold
// of halo_gemm_mxfp4a8_nt_m64 and halo_gemm_mxfp4a8_decode, for the band above theirs.
//
// The decode kernels cover at most 128 rows per block, so at prefill they would re-read the weight
// ceil(M/128) times. Here a block is a 256 x (32*TN) tile and each weight byte is unpacked once per
// K slab into LDS, shared by the block's four row-waves.
//
// A goes from global straight into registers. A WMMA A fragment (gfx1151: lane l holds row l%16 and
// the whole 16-wide k run -- halo_wmma.h) spans 16 rows of a row-major plane, and staging it
// through LDS would cost a write, a barrier and a second read. So A is first re-tiled into the op's
// scratch in fragment order:
//
//     AT[(m/16) * (K/16) * 256 + (k/16) * 256 + (m%16) * 16 + k%16] = A[m][k]
//
// (libr4d writes row m%16 to slot (m%16) + 16*((k/8)%2), the gfx12 fragment's half-split layout;
// on gfx1151 the lane's k run is consecutive, so a fragment is sixteen consecutive fp8 bytes of one
// row and a fragment is 256 contiguous bytes.) A lane's fragment bytes are one aligned 16-byte
// load, so a slab's A fragments issue back to back off a wave-uniform SGPR base (SADDR form). The
// re-tile costs one read and write of A, far less than it saves on large projections, and is
// internal to the op. Rows past M in the last fragment are written as zeros.
//
// The same plane serves the w4a8i8/w5a8i8 rows (their int8 fragments want the same 16 consecutive
// bytes; fp8 and int8 are both one byte per element), which is why this op owns the scratch hook
// their rows declare.
//
// Weight and fold as in the skinny kernel: the fragment-order weight from halo_layout.cpp, its
// e2m1 codes unfolded to EXACT bf16 (m64's mx_e2m1_bf16 arithmetic, the (exp - ref) difference
// folded per 32-block), staged as bf16 pairs in LDS. On gfx1201 the staged form is the folded e4m3
// byte and the fp8 WMMA consumes it; gfx1151 has no fp8 WMMA, and since the gather must
// dequantise anyway, the staging dequantises once instead -- which also makes this row's
// accumulation the skinny kernel's exact-product arithmetic (no e4m3 re-rounding of 2^-d for
// d >= 9; the table's ties are gone). The per-row factor 2^(Wref[n]-127) is applied at the
// epilogue, exactly m64's.
//
// Tile: 8 waves as 4 (M) x 2 (N), each owning TM=4 x TN fragments, so a block covers BMF = 256
// rows and BNF = 2*TN*16 columns. TN=4 halves the column-block count and wins from M = 1024 on
// (on the reference part at a 27B's per-rank weights, mixed below), so tn=0 picks TN=4 from
// M >= 1024. The K slab is 128 at TN=2 (64 if K % 128 != 0) and 64 at TN=4, which holds twice the
// accumulators. TN does not change an output's bits: each fragment runs the same WMMAs in the same
// K order, and the staged bf16 weight does not depend on TN.
//
// Barriers fence LDS only: __syncthreads() also drains the global load counter, and nothing in
// global memory is shared between the waves, so the next slab's A loads stay in flight.
//
// The weight words and exponents a thread stages are loaded a slab ahead, into registers, right
// after the staging barrier, so their DRAM round trip runs under the slab's WMMAs instead of in
// front of the next unpack (+1 VGPR at TN=4). Loading the A fragments a slab ahead too is slower
// (and here impossible at bf16 width); the A load of step s+1 issues under step s's WMMAs (one
// step of lookahead, two 16-byte fragments a row in flight).
//
// A thin last row of blocks (1..160 rows, under 1024 rows in all) runs a second kernel whose
// blocks of that row skip the WMMAs of their padding fragments past M. Every other M runs the
// plain kernel, whose body declares its own LDS: a guard inside the one kernel, or the body
// taking the LDS by pointer, slows the full blocks.
//
// Staging is clamped, never predicated: a predicated load gets its own exec-mask region and the
// compiler cannot carry a counted s_wait_loadcnt across the merge. Clamped loads (tile past M ->
// last real tile, row past N -> last aligned group) issue back to back; the epilogue drops them.
//
// The accumulator is the direct C: the WMMA takes the activation as A and the weight as B, so a
// lane's accumulator slot e holds row 2e + (lane>>4) (accRow) and column lane&15 -- the skinny
// kernel's own store mapping, one single-bf16 store per slot, the per-column 2^(Wref-127) hoisted
// per lane (gfx1151 gives each lane one column, where gfx12's transposed store gave it eight
// consecutive ones and read the reference per slot).
//
// Two activation scale grids: one f32 per row (ABLK=false) or one per (row, 128 of K) (ABLK=true,
// as every fused producer in arch/common writes). With ABLK the accumulator is carried in units of
// the current block's scale: at each 128 boundary every row is multiplied by s_prev / s_next
// (~K/128 fp32 ulps of extra rounding), and the epilogue applies the last scale. Thread t owns row
// t of the block, loads the next boundary's two scales a boundary early and writes the ratio to
// LDS; a lane reads one ratio per slot row (accRow, not a lane constant as on gfx12).
//
// Benchmarking note: as for the skinny kernel, a looped weight stays Infinity-Cache resident.
#include "halo.h"
#include "halo_gemm_args.h"
#include "halo_common.h"
#include "halo_gdn_wmma.h"
#include "halo_fp8_wmma.h"
#include "halo_wmma.h"
#include "halo_mxfp4.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#define HALO_MXT_TM    4         // 16-row fragments per wave
#define HALO_MXT_WM    4         // waves along M
#define HALO_MXT_WN    2         // waves along N
#define HALO_MXT_PAD   8         // LDS row pad, in bf16 elements, and 8 because the staged weight is
                                 // read and written as 16-byte vectors: every row must start on a
                                 // 16-byte boundary. (LBK+8)*2 B per row also turns by 16 B an
                                 // element row, so the gather's 16 lanes stay bank-staggered.
#define HALO_MXT_NTHR  (HALO_MXT_WM * HALO_MXT_WN * 32)
#define HALO_MXT_BMF   (HALO_MXT_WM * HALO_MXT_TM * 16)
#define HALO_MXT_KUNIT 64        // the narrowest K slab, so K must be a multiple of it
#define HALO_MXT_TN4_MIN_M 1024  // tn=0's crossover, see above
#define HALO_MXT_GROUP 32        // K per E8M0 exponent (the OCP MX group)

typedef int      mxt_v2i  __attribute__((ext_vector_type(2)));
typedef unsigned mxt_u2   __attribute__((ext_vector_type(2)));
typedef unsigned mxt_u4   __attribute__((ext_vector_type(4)));
typedef int      mxt_v4i  __attribute__((ext_vector_type(4)));

/* Sixteen e4m3 bytes to the exact bf16 fragment (as halo_gemm_mxfp4a8_nt_m64's mx_deq16_exact). */
__device__ __forceinline__ v16bf mxt_deq16_exact(const uint32_t* __restrict__ w4) {
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

/* One e2m1 code and its block's (exp - ref) as a bf16 bit pattern (m64's mx_e2m1_bf16). NOTE the
 * argument is exp MINUS ref, as m64's mx_frag16x passes it -- NOT the e4m3 table's d = ref - exp
 * (halo_mxfp4_fold_d), whose sign is the other way because there the fold divides the magnitude
 * by 2^d. Passing the table's d here flips every staged exponent. */
/* Eight e2m1 codes (one dword) to four bf16-pair dwords, branchless, four codes a pass: one perm
 * spreads two bytes to [b, b, b', b'], so masks give each code's magnitude (the table selector)
 * and sign a byte; two table perms give the four codes' low and high bf16 bytes at ediff 0
 * (magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6; v_perm_b32 selects bytes 0-3 from its SECOND operand),
 * two perms interleave them into bf16 pairs, one packed 16-bit add shifts the exponents by ediff
 * (magnitude 0's entry is -(ediff << 7), which the add returns to zero), and the signs land at
 * bits 15 and 31 by one more perm each. Bit-identical to the scalar mxt_e2m1_bf16 (and to the
 * byte-a-pass perm form it replaces) for every code and every ediff with
 * 0 <= E + ediff + 126 <= 255: host-verified for every byte value at every position over that
 * range. About 40% fewer VALU than the byte-a-pass form; the skinny, decode and gated kernels
 * carry the same function. */
typedef unsigned short mxt_us2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ void mxt_frag8(unsigned dw, int ediff, uint32_t* out) {
  const uint32_t ed = (uint32_t)(ediff << 7) & 0xFFFFu;
  const mxt_us2 ED2 = __builtin_bit_cast(mxt_us2, ed | (ed << 16));
  const uint32_t z = (0u - ed) & 0xFFFFu;
  const uint32_t T_LO = 0xC0800000u | (z & 0xFFu), T_HI = 0x3F3F3F00u | (z >> 8);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t x = __builtin_amdgcn_perm(dw, dw, h ? 0x03030202u : 0x01010000u);
    const uint32_t mag = (x & 0x00070007u) | ((x >> 4) & 0x07000700u);
    const uint32_t sg = ((x << 4) & 0x00800080u) | (x & 0x80008000u);
    const uint32_t lo = __builtin_amdgcn_perm(0xC0804000u, T_LO, mag);
    const uint32_t hi = __builtin_amdgcn_perm(0x40404040u, T_HI, mag);
    const mxt_us2 p0 = __builtin_bit_cast(mxt_us2, __builtin_amdgcn_perm(hi, lo, 0x05010400u)) + ED2;
    const mxt_us2 p1 = __builtin_bit_cast(mxt_us2, __builtin_amdgcn_perm(hi, lo, 0x07030602u)) + ED2;
    out[2 * h]     = __builtin_bit_cast(uint32_t, p0) | __builtin_amdgcn_perm(0u, sg, 0x010C000Cu);
    out[2 * h + 1] = __builtin_bit_cast(uint32_t, p1) | __builtin_amdgcn_perm(0u, sg, 0x030C020Cu);
  }
}

__device__ __forceinline__ unsigned mxt_e2m1_bf16(unsigned code, int ediff) {
    const unsigned mag = code & 7u;
    if (!mag) return (code & 8u) << 12;
    unsigned E = mag >> 1, m = mag & 1u;
    if (mag == 1) m = 0;   /* the subnormal 0.5: (1 + 0) * 2^-1, not (1 + 1/2) * 2^-1 */
    const unsigned ef = E + (unsigned)(ediff + 126);
    return ((code & 8u) << 12) | (ef << 7) | (m << 6);
}

// The re-tile: one thread per 8-byte slot of the fragment-ordered plane; the slot map is r4d's
// (slot s of a fragment = row s&15, k-bytes 8*(s>>4)..+8 of the 16-wide step), so the reads keep
// r4d's schedule; only the store lands at gfx1151's consecutive-k slot.
__global__ __launch_bounds__(256) void halo_mxfp4a8_tile_a_kernel(
    const unsigned char* __restrict__ A, unsigned char* __restrict__ AT, int M, int K,
    long long slots, int lda) {
  const long long o = (long long)blockIdx.x * 256 + threadIdx.x;
  if (o >= slots) return;
  const int ks = K / 16;
  const int lane = (int)(o & 31);
  const long long f = o >> 5;
  const int mt = (int)(f / ks), kst = (int)(f % ks);
  const int r = mt * 16 + (lane & 15), k = kst * 16 + (lane >> 4) * 8;
  mxt_u2 v = { 0u, 0u };
  if (r < M) v = *(const mxt_u2*)(A + (size_t)r * lda + k);
  *(mxt_u2*)(AT + f * 256 + (lane & 15) * 16 + (lane >> 4) * 8) = v;
}

// The kernel's body. OWN: it declares its LDS arrays itself (the plain kernel); otherwise it takes
// the caller's (the two-body kernel, one set for both bodies). PART: a block the M edge cuts,
// which skips the WMMAs of its padding fragments past M.
template <int TN, int LBK, bool ABLK, bool PART, bool OWN>
__device__ __forceinline__ void halo_gemm_mxfp4a8_tiled_body(
    const unsigned char* __restrict__ AT, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int as_rs, int ldws, int ldc, unsigned short* __restrict__ sW_in,
    float* __restrict__ sR_in) {
  constexpr int TM = HALO_MXT_TM, WN = HALO_MXT_WN, NTHR = HALO_MXT_NTHR, BMF = HALO_MXT_BMF;
  constexpr int NS = LBK / 16, STR = LBK + HALO_MXT_PAD, BNF = WN * TN * 16;
  constexpr int NTILES_T = BNF / 16, TOT_SLOTS = NTILES_T * NS * 32;
  // 4 weight slots (one 16 B read) wherever the tile has enough of them, else 2 (8 B), else 1.
  constexpr int WSLOTS = (TOT_SLOTS >= NTHR * 4) ? 4 : (TOT_SLOTS >= NTHR * 2 ? 2 : 1);
  constexpr int NW = TOT_SLOTS / (NTHR * WSLOTS);
  static_assert(NW * NTHR * WSLOTS == TOT_SLOTS, "the W tile must split evenly over the block");
  __shared__ unsigned short sW_own[OWN ? BNF * STR : 1];   // the staged weight, exact bf16 pairs
  __shared__ float sR_own[OWN && ABLK ? BMF : 1];   // ABLK: a boundary's ratio a row, then each row's final scale
  unsigned short* __restrict__ sW = OWN ? sW_own : sW_in;
  float* __restrict__ sR = OWN ? sR_own : sR_in;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF;
  const int ksteps = K / 16, Mt = (M + 15) >> 4;
  // PART: this wave's M fragments that hold a row below M, 0..TM and wave-uniform; the rest are
  // padding whose WMMAs are skipped (their rows are dropped at the store). Their A loads stay,
  // clamped, one step in flight.
  const int live = PART ? __builtin_amdgcn_readfirstlane(min(TM, max(0, Mt - ((m0 >> 4) + wm * TM))))
                        : TM;

  // One A base per M fragment, wave-uniform: readfirstlane keeps it in an SGPR so loads take the
  // SADDR form with a 32-bit lane offset. A tile past M reads the last real one.
  const unsigned char* abase[TM];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int mt = (m0 >> 4) + wm * TM + i;
    mt = mt < Mt - 1 ? mt : Mt - 1;
    abase[i] = AT + (size_t)__builtin_amdgcn_readfirstlane(mt * ksteps * 256);
  }
  const unsigned int aoff = (unsigned int)col * 16;

  // The W slots this thread stages, fixed across slabs: WSLOTS adjacent rows of one (n-tile,
  // k-step), so the wave reads contiguous bytes; the clamp keeps the vector read aligned.
  int wrow[NW], wkloc[NW], wrc[NW], wkst[NW], wlanec[NW];
  unsigned int wref[NW];
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    const int sl = it * NTHR * WSLOTS + tid * WSLOTS;
    const int lane_ = sl & 31, rest = sl >> 5;
    const int kst = rest % NS, ntl = rest / NS;
    wrow[it] = ntl * 16 + (lane_ & 15);
    wkloc[it] = kst * 16 + (lane_ >> 4) * 8;
    const int gn = n0 + wrow[it];
    wrc[it] = gn < N - WSLOTS ? gn : N - WSLOTS;
    wlanec[it] = (lane_ & 16) | (wrc[it] & 15);
    wkst[it] = kst;
    wref[it] = WSLOTS == 4 ? *(const unsigned int*)(Wref + wrc[it])
             : WSLOTS == 2 ? *(const unsigned short*)(Wref + wrc[it])
                           : Wref[wrc[it]];
  }

  v8f acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (v8f)(0.f);

  // ABLK: thread t owns row m0 + t (BMF == NTHR) and loads the next 128 boundary's two scales (the
  // block starting there and the one before it) a boundary early.
  const int trow = m0 + tid < M ? m0 + tid : M - 1;
  const float* __restrict__ asrow = As + (size_t)trow * as_rs;
  float ns = 0.f, np = 0.f;
  const unsigned int* __restrict__ Wp = (const unsigned int*)W;

  // The per-lane reference factors: the lane's one column per N fragment (gfx1151's D mapping
  // keeps the column on lane&15). Clamped reads; the epilogue drops the padding columns.
  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j)
    rf[j] = __int_as_float((int)Wref[n0 + wn * TN * 16 + j * 16 + col < N
                                        ? n0 + wn * TN * 16 + j * 16 + col : N - 1] << 23);

  // The slab's weight words and exponents, loaded a slab ahead (the first one here).
  unsigned int wqn[NW][WSLOTS], wswn[NW];
  auto wload = [&](int k0) {
    const int kstep0 = k0 / 16;
#pragma unroll
    for (int it = 0; it < NW; ++it) {
      const unsigned int* src =
          &Wp[((size_t)(wrc[it] >> 4) * ksteps + kstep0 + wkst[it]) * 32 + wlanec[it]];
      const int blk = (k0 + wkst[it] * 16) / HALO_MXT_GROUP;
      if constexpr (WSLOTS == 4) {
        const mxt_u4 v = *(const mxt_u4*)src;
        wqn[it][0] = v[0]; wqn[it][1] = v[1]; wqn[it][2] = v[2]; wqn[it][3] = v[3];
        wswn[it] = *(const unsigned int*)(Ws + (size_t)blk * ldws + wrc[it]);
      } else if constexpr (WSLOTS == 2) {
        const mxt_u2 v = *(const mxt_u2*)src;
        wqn[it][0] = v[0]; wqn[it][1] = v[1];
        wswn[it] = *(const unsigned short*)(Ws + (size_t)blk * ldws + wrc[it]);
      } else {
        wqn[it][0] = *src;
        wswn[it] = Ws[(size_t)blk * ldws + wrc[it]];
      }
    }
  };
  wload(0);

  // The A fragments one step of lookahead ahead (int4v = 16 fp8 bytes = the lane's fragment row).
  mxt_v4i af_buf[2][TM];
  auto aload = [&](int step, int kb) {
#pragma unroll
    for (int i = 0; i < TM; ++i)
      af_buf[step & 1][i] =
          *(const mxt_v4i*)(abase[i] + aoff + (unsigned int)(kb * 16) + (unsigned int)(step * 256));
  };
  aload(0, 0);

  for (int k0 = 0; k0 < K; k0 += LBK) {
    // ABLK: at a 128 boundary each row's ratio goes to LDS, published by the staging barrier below
    // (sR's previous readers finished before the last slab's closing barrier). A zero block keeps
    // the carried scale; after a zero block the carried scale is looked up from the grid.
    const bool bnd = ABLK && k0 && (k0 & 127) == 0;
    if constexpr (ABLK) {
      if (bnd)
        sR[tid] = ns > 0.f ? (np > 0.f ? np : halo_mxfp4_carried(asrow, (k0 >> 7) - 2)) / ns : 1.f;
    }
    // W slab -> LDS, unfolded to exact bf16 with the 32-block's (exp - ref) folded in, from the
    // words loaded a slab ago.
#pragma unroll
    for (int it = 0; it < NW; ++it) {
#pragma unroll
      for (int q = 0; q < WSLOTS; ++q) {
        const int ed = (int)((wswn[it] >> (8 * q)) & 0xFF) - (int)((wref[it] >> (8 * q)) & 0xFF);
        union { uint32_t u[4]; mxt_u4 v; } pk;
        mxt_frag8(wqn[it][q], ed, pk.u);
        *(mxt_u4*)(&sW[(wrow[it] + q) * STR + wkloc[it]]) = pk.v;
      }
    }
    lds_barrier();
    if (k0 + LBK < K) wload(k0 + LBK);   // the next slab's weight flies across this slab's WMMAs

    if constexpr (ABLK) {
      // Re-express each row's partial sum in the new block's scale: one ratio a slot row.
      if (bnd) {
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e)
              acc[i][j][e] *= sR[wm * TM * 16 + i * 16 + accRow(lane, e)];
      }
      // The next boundary's scales, issued behind this slab's loads and needed a boundary later.
      if ((k0 & 127) == 0 && k0 + 128 < K) {
        ns = asrow[(k0 >> 7) + 1];
        np = asrow[k0 >> 7];
      }
    }

#pragma unroll
    for (int step = 0; step < NS; ++step) {
      // The weight fragments: the lane's column, sixteen consecutive k, straight out of LDS.
      v16bf wf[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned short* p = &sW[(wn * TN * 16 + j * 16 + col) * STR + step * 16];
        union { uint4 w[2]; v16bf f; } u;
        u.w[0] = *(const uint4*)p;
        u.w[1] = *(const uint4*)(p + 8);
        wf[j] = u.f;
      }
      // The activation fragments staged for this step (loaded under the previous step's WMMAs).
      v16bf af[TM];
#pragma unroll
      for (int i = 0; i < TM; ++i) af[i] = mxt_deq16_exact((const uint32_t*)&af_buf[step & 1][i]);
      // Keep one step's fragments live at a time; otherwise the compiler hoists every step's
      // loads above the first WMMA.
      __builtin_amdgcn_sched_barrier(0);
      if (step + 1 < NS) aload(step + 1, k0);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        if (PART && i >= live) break;   // wave-uniform: padding fragments past M
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = halo::mma(af[i], wf[j], acc[i][j]);
      }
    }
    // The next slab's first A step flies across this slab's closing barrier; the current slab's
    // last step is done, so buffer (NS - 1) & 1 is dead, and step 0's buffer 0 is dead too --
    // NS = LBK/16 is even here (LBK is 64 or 128), so the last step used buffer 1.
    if (k0 + LBK < K) aload(0, k0 + LBK);
    lds_barrier();
  }

  // Epilogue. A lane holds row m0 + 64*wm + 16i + accRow(lane, e) of M fragment i for slot e, and
  // the one column n0 + wn*TN*16 + j*16 + lane&15 of N fragment j. 2^(Wref[n]-127) is the float
  // whose exponent field is Wref[n] (hoisted above); N % 16 keeps a lane's column inside one
  // fragment, so a column is entirely inside or entirely outside N.
  float sc[TM][8];
  if constexpr (ABLK) {
    // the scale each row ends in, once a row; the K loop's closing barrier retired the ratio reads
    sR[tid] = halo_mxfp4_carried(asrow, (K - 1) >> 7);
    lds_barrier();
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        sc[i][e] = sR[wm * TM * 16 + i * 16 + accRow(lane, e)];
  }
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int nb = n0 + wn * TN * 16 + j * 16 + col;
    if (nb >= N) continue;
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * TM * 16 + i * 16 + accRow(lane, e);
        if (m >= M) continue;
        const float s = ABLK ? sc[i][e] : As[(size_t)m * as_rs];
        C[(size_t)m * ldc + nb] = (__hip_bfloat16)(acc[i][j][e] * rf[j] * s);
      }
  }
}

// The plain kernel: every block the unguarded body, over the body's own LDS.
template <int TN, int LBK, bool ABLK>
__global__ __launch_bounds__(HALO_MXT_NTHR) void halo_gemm_mxfp4a8_tiled_kernel(
    const unsigned char* __restrict__ AT, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int as_rs, int ldws, int ldc) {
  halo_gemm_mxfp4a8_tiled_body<TN, LBK, ABLK, false, true>(AT, As, W, Ws, Wref, C, M, K, N, as_rs,
                                                           ldws, ldc, nullptr, nullptr);
}

// The two-body kernel, for a thin last row of blocks: a block of that row takes the guarded body,
// every other block the unguarded one, uniform over the block, over one set of LDS arrays.
template <int TN, int LBK, bool ABLK>
__global__ __launch_bounds__(HALO_MXT_NTHR) void halo_gemm_mxfp4a8_tiled_part_kernel(
    const unsigned char* __restrict__ AT, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int as_rs, int ldws, int ldc) {
  constexpr int BNF = HALO_MXT_WN * TN * 16, STR = LBK + HALO_MXT_PAD;
  __shared__ unsigned short sW[BNF * STR];
  __shared__ float sR[ABLK ? HALO_MXT_BMF : 1];
  if ((int)(blockIdx.y + 1) * HALO_MXT_BMF > M)
    halo_gemm_mxfp4a8_tiled_body<TN, LBK, ABLK, true, false>(AT, As, W, Ws, Wref, C, M, K, N,
                                                             as_rs, ldws, ldc, (unsigned short*)sW,
                                                             sR);
  else
    halo_gemm_mxfp4a8_tiled_body<TN, LBK, ABLK, false, false>(AT, As, W, Ws, Wref, C, M, K, N,
                                                              as_rs, ldws, ldc,
                                                              (unsigned short*)sW, sR);
}

extern "C" int halo_gemm_mxfp4a8_tiled_group(void) { return HALO_MXT_GROUP; }

extern "C" long halo_mxfp4a8_tile_a_bytes(int M, int K) {
  return M < 1 || K < 16 ? 0 : (long)((M + 15) / 16) * 16 * K;
}

extern "C" void halo_mxfp4a8_tile_a(long a, long at, int M, int K, long stream, int lda) {
  if (!lda) lda = K;
  if (M < 1) throw std::runtime_error("halo mxfp4a8_tile_a: M must be >= 1");
  if (K < 16 || K % 16) throw std::runtime_error("halo mxfp4a8_tile_a: K must be a multiple of 16");
  const long long slots = (long long)((M + 15) / 16) * (K / 16) * 32;
  hipLaunchKernelGGL(halo_mxfp4a8_tile_a_kernel, dim3((unsigned)((slots + 255) / 256)), dim3(256),
                     0, (hipStream_t)stream, (const unsigned char*)a, (unsigned char*)at, M, K,
                     slots, lda);
}

extern "C" void halo_gemm_mxfp4a8_tiled(long at, long ascale, long wq, long ws, long wref, long c,
                                        int M, int K, int N, int TN, int as_group, int as_rs,
                                        long stream, int ldws, int ldc) {
  if (!ldws) ldws = N;
  if (!ldc) ldc = N;
  if (M < 1) throw std::runtime_error("halo gemm_mxfp4a8_tiled: M must be >= 1");
  if (N < 16 || N % 16) throw std::runtime_error("halo gemm_mxfp4a8_tiled: N must be a multiple of 16");
  if (K < HALO_MXT_KUNIT || K % HALO_MXT_KUNIT)
    throw std::runtime_error("halo gemm_mxfp4a8_tiled: K must be a multiple of 64");
  // tn=0: TN=4 from the M crossover on, unless its 128-wide tile leaves fewer than 128 blocks
  // (narrow N), where TN=2's extra blocks win.
  if (TN == 0)
    TN = M >= HALO_MXT_TN4_MIN_M && (long)((N + 127) / 128) * ((M + HALO_MXT_BMF - 1) / HALO_MXT_BMF) >= 128
             ? 4 : 2;
  if (TN != 2 && TN != 4) throw std::runtime_error("halo gemm_mxfp4a8_tiled: TN must be 2 or 4");
  if (as_group != 0 && as_group != 128)
    throw std::runtime_error("halo gemm_mxfp4a8_tiled: the activation scale is per row or per 128 of K");
  const bool ablk = as_group == 128;
  if (ablk && as_rs < (K + 127) / 128)
    throw std::runtime_error("halo gemm_mxfp4a8_tiled: activation scale rows shorter than ceil(K/128)");
  const bool wide_slab = TN == 2 && K % 128 == 0;
  const int bnf = HALO_MXT_WN * TN * 16;
  dim3 grid((N + bnf - 1) / bnf, (M + HALO_MXT_BMF - 1) / HALO_MXT_BMF), block(HALO_MXT_NTHR);
  auto Ap = (const unsigned char*)at;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Rp = (const unsigned char*)wref;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
  // The two-body kernel only for a thin last row of blocks (1..160 rows) under 1024 rows; see the
  // header.
  const int rem = M % HALO_MXT_BMF;
  const bool part = rem > 0 && rem <= 160 && M < 1024;
#define MXT_LAUNCH(TNv, LBKv, ABv) do {                                                          \
    if (part)                                                                                     \
      hipLaunchKernelGGL((halo_gemm_mxfp4a8_tiled_part_kernel<TNv, LBKv, ABv>), grid, block, 0, st,\
                         Ap, Sp, Wp, Zp, Rp, Cp, M, K, N, as_rs, ldws, ldc);                       \
    else                                                                                          \
      hipLaunchKernelGGL((halo_gemm_mxfp4a8_tiled_kernel<TNv, LBKv, ABv>), grid, block, 0, st, Ap, \
                         Sp, Wp, Zp, Rp, Cp, M, K, N, as_rs, ldws, ldc);                           \
  } while (0)
  if (TN == 4)        { if (ablk) MXT_LAUNCH(4, 64, true);  else MXT_LAUNCH(4, 64, false); }
  else if (wide_slab) { if (ablk) MXT_LAUNCH(2, 128, true); else MXT_LAUNCH(2, 128, false); }
  else                { if (ablk) MXT_LAUNCH(2, 64, true);  else MXT_LAUNCH(2, 64, false); }
#undef MXT_LAUNCH
}

// ============================================================== the LDS-staged row kernel
//
// What the row launches, and what the gemv rows launch past one row tile on a wide weight. The
// kernel above keeps the activation in fp8 and every wave re-runs the exact e4m3 decode (~180 VALU
// a fragment) for each of its row fragments at every k step -- about three times the WMMA work it
// feeds -- and spills at every instantiation on gfx1151. This one is halo_gemm_fp8a8.hip's prefill
// tile: eight waves stage a BM x 64 activation tile and a BN x 64 weight tile through LDS as exact
// bf16, the next tiles' loads in flight under the current one's WMMAs. The activation comes from a
// bf16 plane the launch decodes once (mxt2_adq_kernel, when the op's scratch holds it) or is
// decoded while staging; the weight is unpacked once a block with mxt_frag8.
//
// Contract: byte-identical to gemm_mxfp4a8_nt_m64 at the same split (one, for the tiled row; the
// gemv split law's, for the gemv rows). Each 32-block runs its two 16-K WMMAs from zero, then folds
// with fmaf(tmp, 2^(ref-127) [* As(row, k/128)], acc) in K order, each split's chain from zero and
// the chains summed from zero in split order, and the store is m64's (ABLK ? v : v * As[m]); the A
// values are mx_deq16_exact's, the weight values mx_frag16x's. Only where operands sit changes.
//
// The weight's fragment order (halo_layout.cpp): per (16-row tile, 16-k step), 32 dwords, dword l
// row l & 15 with k 8*(l >> 4) .. +8 in nibble order. Four steps are 512 contiguous bytes, so a
// thread's 16-byte chunk is one k run of four adjacent rows, whose exponents and reference
// exponents are four adjacent bytes.
#define MXT2_BK    64
#define MXT2_PITCH (MXT2_BK + 8)

__global__ __launch_bounds__(256) void mxt2_adq_kernel(const unsigned char* __restrict__ A,
                                                       uint16_t* __restrict__ Adq, int M, int K,
                                                       int lda) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)M * (K / 16)) return;
  const int r = (int)(idx / (K / 16)), c = (int)(idx % (K / 16)) * 16;
  const mxt_v4i av = *(const mxt_v4i*)(A + (size_t)r * lda + c);
  const v16bf f = mxt_deq16_exact((const uint32_t*)&av);
  uint4* dst = (uint4*)(Adq + (size_t)r * K + c);
  const uint4* src = (const uint4*)&f;
  dst[0] = src[0];
  dst[1] = src[1];
}

/* The tile is WMW x (8 / WMW) waves, each a TM x TN patch of 16 x 16 fragments: BM = WMW*TM*16
 * rows by BN = (8/WMW)*TN*16 columns. The prefill tiles are 2 x 4 waves; the decode band's
 * (gemv_wide, 17..64 rows) one block row tall, so the weight is read and unpacked once whatever
 * the row count (the skinny kernel reads and unpacks it once a row tile).
 *
 * SPLIT: the skinny kernel at SK splits, not one. nt_m64 runs each contiguous K/SK range as its
 * own chain of folds from zero and sums the chains' totals from zero in split order
 * (v = 0; v += acc_s); one wave walking all of K reproduces that exactly with the running chain
 * `acc` and a running total `v`: at each split's last fold, v += acc and acc restarts from zero.
 * At one split v = 0 + acc is acc's own bits (acc is never -0: it starts at +0, and a
 * round-to-nearest sum is -0 only of two -0s), so the decode band runs SPLIT at every split and
 * the prefill band, always one split, without it.
 *
 * The weight's raw words and exponents, and the activation's chunks, are loaded PD slabs ahead into
 * a register ring, so PD slabs of DRAM latency hide under the WMMAs (the barriers fence LDS only,
 * so the ring stays in flight across them). At PD 1 this is the one slab of lookahead the prefill
 * tiles keep (their accumulators leave no room for more).
 *
 * PART: a launch whose last row of blocks the M edge cuts; each wave skips the WMMAs and folds of
 * its fragments wholly past M (wave-uniform), so a thin tail band costs its staging, not a band of
 * WMMAs (M 129: 2487 -> 2116 us at N 5120, K 25600). */
template <int WMW, int TM, int TN, int PD, bool APLANE, bool ABLK, bool SPLIT, bool PART>
__global__ __launch_bounds__(256) void halo_gemm_mxfp4a8_tiled2_kernel(
    const unsigned char* __restrict__ A, const uint16_t* __restrict__ Adq,
    const float* __restrict__ As, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ Ws, const unsigned char* __restrict__ Wref,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int lda, int as_rs, int ldws, int ldc,
    int Ks) {
  constexpr int WNW = 8 / WMW;
  constexpr int BM = WMW * TM * 16, BN = WNW * TN * 16;
  constexpr int ACH = APLANE ? 8 : 16;                  // activation elements a 16-byte chunk
  constexpr int CPR = MXT2_BK / ACH;                    // ... chunks a row
  constexpr int ACN = BM * CPR;                         // chunks a tile
  constexpr int AL = (ACN + 255) / 256;                 // ... a thread (a small tile: the low threads)
  constexpr int WCH = BN * MXT2_BK / 2 / 16;            // 16-byte weight chunks a tile
  constexpr int WL = (WCH + 255) / 256;                  // ... a thread (a 64-wide tile: half)
  static_assert(WMW * WNW == 8 && (WCH % 256 == 0 || WCH < 256), "a tile is whole chunks");
  __shared__ __attribute__((aligned(16))) unsigned short sA[BM * MXT2_PITCH];
  __shared__ __attribute__((aligned(16))) unsigned short sW[BN * MXT2_PITCH];
  __shared__ __attribute__((aligned(16))) float sS[ABLK ? BM : 1];

  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
  const int wm = wave / WNW, wn = wave % WNW;
  constexpr unsigned kGroupM = 8;
  const unsigned per = kGroupM * gridDim.x, lin = blockIdx.y * gridDim.x + blockIdx.x;
  const unsigned first = lin / per * kGroupM, rr = lin % per;
  const unsigned gm = min(gridDim.y - first, kGroupM);
  const int m0 = (int)(first + rr % gm) * BM, n0 = (int)(rr / gm) * BN;
  const int ksteps = K / 16, ntmax = N / 16 - 1;

  /* Each thread's loads are a 32-bit byte offset off a slab base that is uniform over the block
   * (SGPRs), so a ring slot costs its data registers alone, not a 64-bit address a slab. */
  unsigned aoff[AL];
  unsigned short* adst[AL];
#pragma unroll
  for (int i = 0; i < AL; ++i) {
    /* A thread past the tile's chunks (a 16- or 48-row tile) loads a real one and stores none. */
    const int c = (tid + 256 * i) % ACN, r = c / CPR, e = (c % CPR) * ACH;
    const int gr = m0 + r < M ? m0 + r : M - 1;
    aoff[i] = APLANE ? ((unsigned)gr * (unsigned)K + e) * 2u : (unsigned)gr * (unsigned)lda + e;
    adst[i] = sA + r * MXT2_PITCH + e;
  }
  /* Weight chunk c: n tile c >> 5 of the block (clamped to the last real one, as m64 clamps),
   * dwords d0 .. d0+3 of its 512-byte four-step run: step d0 >> 5, rows (d0 & 15) .. +3, k run
   * 8 * ((d0 >> 4) & 1). */
  unsigned woff[WL], eoff[WL];
  unsigned short* wdst[WL];
  unsigned wref4[WL];
#pragma unroll
  for (int i = 0; i < WL; ++i) {
    const int c = WCH < 256 ? tid % WCH : tid + 256 * i, t = c >> 5, d0 = (c & 31) * 4;
    const int s = d0 >> 5, l0 = d0 & 31;
    int gt = (n0 >> 4) + t;
    gt = gt < ntmax ? gt : ntmax;
    const int n4 = gt * 16 + (l0 & 15);
    woff[i] = (((unsigned)gt * (unsigned)ksteps + s) * 32u + l0) * 4u;
    eoff[i] = (unsigned)(s >> 1) * (unsigned)ldws + n4;   /* the step's 32-block, past the slab's first */
    wref4[i] = *(const unsigned*)(Wref + n4);
    wdst[i] = sW + (t * 16 + (l0 & 15)) * MXT2_PITCH + s * 16 + (l0 >> 4) * 8;
  }
  const int srow = tid < BM ? (m0 + tid < M ? m0 + tid : M - 1) : 0;
  const unsigned soff = (unsigned)srow * (unsigned)as_rs * 4u;

  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    int t = (n0 >> 4) + wn * TN + j;
    t = t < ntmax ? t : ntmax;
    rf[j] = __int_as_float((int)Wref[t * 16 + (lane & 15)] << 23);   /* 2^(ref - 127) */
  }

  /* PART: this wave's fragments that hold a row below M, wave-uniform; past them (the padding of
   * a block the M edge cuts) the WMMAs and folds are skipped and the rows dropped at the store. */
  const int live = PART ? __builtin_amdgcn_readfirstlane(
                              min(TM, max(0, (M + 15) / 16 - (m0 / 16 + wm * TM))))
                        : TM;
  const unsigned short* fa = sA + (wm * TM * 16 + (lane & 15)) * MXT2_PITCH;
  const unsigned short* fw = sW + (wn * TN * 16 + (lane & 15)) * MXT2_PITCH;

  /* acc: the current split's chain; v: the splits' running total (SPLIT only); tmp: the open
   * 32-block, whose first WMMA starts from the zero accumulator rather than a cleared tmp. */
  v8f acc[TM][TN], tmp[TM][TN], v[SPLIT ? TM : 1][SPLIT ? TN : 1];
  const v8f zero = (v8f)0.f;
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = zero;
  if constexpr (SPLIT) {
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) v[i][j] = zero;
  }

  halo_u32x4 ra[PD][AL], rw[PD][WL];
  unsigned re[PD][WL];
  float rs[PD];
  /* Every load is unconditional (a fetch past the last slab re-reads it): a load issued on one
   * path only makes the wait-count pass assume the worst at the merge, an s_waitcnt vmcnt(0) at
   * the top of every slab that drains the whole ring. */
  const int nk = K / MXT2_BK;
  /* Both operands go through the ring: the load counter retires in order, so an activation
   * fetched nearer than the weight would make each slab's wait on it drain the ring too. */
  auto fetch = [&](int d, int kt) {
    const int kc = kt < nk ? kt : nk - 1, k = kc * MXT2_BK;
    const char* ab = APLANE ? (const char*)(Adq + k) : (const char*)(A + k);
    const char* wb = (const char*)W + (size_t)kc * 4 * 32 * 4;
    const unsigned char* eb = Ws + (size_t)(k / HALO_MXT_GROUP) * ldws;
#pragma unroll
    for (int i = 0; i < AL; ++i) ra[d][i] = *(const halo_u32x4*)(ab + aoff[i]);
    /* rows past BM read row 0's and never publish it */
    if constexpr (ABLK) rs[d] = *(const float*)((const char*)(As + (k >> 7)) + soff);
#pragma unroll
    for (int i = 0; i < WL; ++i) {
      /* A 64-wide tile's upper threads repeat the lower half's chunk; only the lower half stores. */
      rw[d][i] = *(const halo_u32x4*)(wb + woff[i]);
      re[d][i] = *(const unsigned*)(eb + eoff[i]);
    }
  };

#pragma unroll
  for (int d = 0; d < PD; ++d) fetch(d, d);
  const int fps = Ks / HALO_MXT_GROUP;   // folds a split
  int fc = 0;

  /* One slab: stage it from ring slot d, then refill the slot PD slabs ahead and run the WMMAs. */
  auto slab = [&](const int d, const int kt) {
    /* No code motion across slabs: the scheduler would otherwise unpack later slabs' ring
     * words early and hold them (spilling past 256 VGPRs at a 16-row tile). */
    __builtin_amdgcn_sched_barrier(0);
    lds_barrier();
#pragma unroll
    for (int i = 0; i < AL; ++i) {
      if (ACN % 256 && tid + 256 * i >= ACN) break;
      if constexpr (APLANE) {
        *(halo_u32x4*)adst[i] = ra[d][i];
      } else {
        const v16bf f = mxt_deq16_exact((const uint32_t*)&ra[d][i]);
        const halo_u32x4* q = (const halo_u32x4*)&f;
        *(halo_u32x4*)adst[i] = q[0];
        *(halo_u32x4*)(adst[i] + 8) = q[1];
      }
    }
#pragma unroll
    for (int i = 0; i < WL; ++i)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        if (WCH < 256 && tid >= WCH) break;
        const int ed = (int)((re[d][i] >> (8 * q)) & 0xFF) - (int)((wref4[i] >> (8 * q)) & 0xFF);
        union { uint32_t u[4]; halo_u32x4 v; } pk;
        mxt_frag8(rw[d][i][q], ed, pk.u);
        *(halo_u32x4*)(wdst[i] + q * MXT2_PITCH) = pk.v;
      }
    if constexpr (ABLK) {
      /* Row r at its accumulator slot's order: fragment r / 16, half r & 1, slot (r & 15) / 2,
       * so a lane's eight fold scales are two 16-byte reads. */
      if (tid < BM) sS[(tid & ~15) + (tid & 1) * 8 + ((tid & 15) >> 1)] = rs[d];
    }
    lds_barrier();
    fetch(d, kt + PD);
#pragma unroll
    for (int s = 0; s < MXT2_BK / 16; ++s) {
      v16bf af[TM], bf[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) af[i] = fragP(fa + i * 16 * MXT2_PITCH + s * 16);
#pragma unroll
      for (int j = 0; j < TN; ++j) bf[j] = fragP(fw + j * 16 * MXT2_PITCH + s * 16);
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        if (PART && i >= live) break;
#pragma unroll
        for (int j = 0; j < TN; ++j) tmp[i][j] = halo::mma(af[i], bf[j], (s & 1) ? tmp[i][j] : zero);
      }
      __builtin_amdgcn_sched_group_barrier(0x100, TM + TN, 0);    /* DS reads */
      __builtin_amdgcn_sched_group_barrier(0x008, TM * TN, 0);    /* WMMA */
      if (s & 1) {
        /* A 32-block closed: m64's fold. */
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          if (PART && i >= live) break;
          v8f asv = (v8f)1.f;
          if constexpr (ABLK) asv = *(const v8f*)&sS[(wm * TM + i) * 16 + (lane >> 4) * 8];
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float as = asv[e];   /* row 2e + (lane >> 4) of the fragment (accRow) */
#pragma unroll
            for (int j = 0; j < TN; ++j) {
              float f = rf[j];
              if constexpr (ABLK) f *= as;
              acc[i][j][e] = fmaf(tmp[i][j][e], f, acc[i][j][e]);
            }
          }
        }
        if constexpr (SPLIT) {
          /* The split's last fold: its chain joins the total, in split order. */
          if (++fc == fps) {
            fc = 0;
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
              for (int j = 0; j < TN; ++j) {
                v[i][j] += acc[i][j];
                acc[i][j] = zero;
              }
          }
        }
      }
    }
  };

  int kt0 = 0;
  for (; kt0 + PD <= nk; kt0 += PD) {
#pragma unroll
    for (int d = 0; d < PD; ++d) slab(d, kt0 + d);
  }
#pragma unroll
  for (int d = 0; d < PD - 1; ++d)
    if (kt0 + d < nk) slab(d, kt0 + d);

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
      if (n >= N) continue;
      const int mb = m0 + (wm * TM + i) * 16;
      const v8f r = SPLIT ? v[SPLIT ? i : 0][SPLIT ? j : 0] : acc[i][j];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = mb + accRow(lane, e);
        if (m < M) C[(size_t)m * ldc + n] = (__hip_bfloat16)(ABLK ? r[e] : r[e] * As[m]);
      }
    }
}

/* The activation plane fits the op's scratch up to this size; past it (a long prefill band at a
 * wide K) the kernel decodes while staging instead, rather than the program reserving the plane. */
#define MXT2_PLANE_CAP (256ll << 20)

/* Launch one part. `Ks` is the split length (K for one split). The prefill band picks the tile by
 * the row's `tn` tunable (2: 64 x 64, 4: 128 x 128); 0 is halo_gemm_bf16_nt_tiled's rule, the
 * 128 x 128 tile once its grid reaches 32 blocks. The two tiles' bytes are the same, so the tuner
 * chooses on time alone. The decode band (M <= 64, the plane present) takes a 1 x 8 wave tile
 * one to four row fragments tall. */
static int mxt2_launch(const unsigned char* A, const uint16_t* Adq, const float* As,
                       const unsigned char* W, const unsigned char* Ws, const unsigned char* Wref,
                       __hip_bfloat16* C, int M, int K, int N, int lda, bool ablk, int as_rs,
                       int ldws, int ldc, int tn, int Ks, hipStream_t st) {
#define MXT2_GO(WMW, TM, TN, PD, P, B, S, PT) hipLaunchKernelGGL(                                 \
    (halo_gemm_mxfp4a8_tiled2_kernel<WMW, TM, TN, PD, P, B, S, PT>), grid, dim3(256), 0, st, A,  \
    Adq, As, W, Ws, Wref, C, M, K, N, lda, as_rs, ldws, ldc, Ks)
  /* The decode band: SPLIT at any split count (one included), whole row tiles. */
#define MXT2_AB(WMW, TM, TN, PD, P) do {                                                          \
    if (ablk) MXT2_GO(WMW, TM, TN, PD, P, true, true, false);                                     \
    else      MXT2_GO(WMW, TM, TN, PD, P, false, true, false); } while (0)
  /* The prefill band: one split always; `part` when the M edge cuts the last row of blocks. */
#define MXT2_AB1(WMW, TM, TN, PD, P) do {                                                         \
    if (part) { if (ablk) MXT2_GO(WMW, TM, TN, PD, P, true, false, true);                        \
                else      MXT2_GO(WMW, TM, TN, PD, P, false, false, true); }                     \
    else      { if (ablk) MXT2_GO(WMW, TM, TN, PD, P, true, false, false);                       \
                else      MXT2_GO(WMW, TM, TN, PD, P, false, false, false); } } while (0)
  if (M > 16 && M <= 64 && Adq) {
    /* The decode band: a 1 x 8 wave tile 32 or 48 rows tall and 128 wide, deep in the ring; at
     * 49..64 rows the 2 x 4 one 64 square, whose accumulators leave room for the occupancy the
     * WMMAs then need. Measured on gfx1151 against each other (34816 x 5120: 769 us at 18 rows,
     * 956 at 36, 1140 at 64). */
    const int mt = (M + 15) / 16;
    if (mt == 2) {
      const dim3 grid((unsigned)((N + 127) / 128), 1);
      MXT2_AB(1, 2, 1, 4, true);
    } else if (mt == 3) {
      const dim3 grid((unsigned)((N + 127) / 128), 1);
      MXT2_AB(1, 3, 1, 4, true);
    } else {
      const dim3 grid((unsigned)((N + 63) / 64), 1);
      MXT2_AB(2, 2, 1, 1, true);
    }
  } else {
    const long long g128 = (long long)((N + 127) / 128) * ((M + 127) / 128);
    const bool big = tn == 4 || (tn != 2 && g128 >= 32);
    const int bt = big ? 128 : 64;
    const bool part = M % bt != 0;
    const dim3 grid((unsigned)((N + bt - 1) / bt), (unsigned)((M + bt - 1) / bt));
    if (big) { if (Adq) MXT2_AB1(2, 4, 2, 1, true); else MXT2_AB1(2, 4, 2, 1, false); }
    else     { if (Adq) MXT2_AB1(2, 2, 1, 1, true); else MXT2_AB1(2, 2, 1, 1, false); }
  }
#undef MXT2_AB1
#undef MXT2_AB
#undef MXT2_GO
  return hipGetLastError() == hipSuccess ? 0 : -5;
}

/* Every part of a gemm_nt_q MXFP4 launch through mxt2_launch at split length K / sk. The
 * activation plane is decoded into the op's scratch when it fits (else each block decodes while
 * staging). */
static int mxt2_run(const RadArgs* a, RadStream s, const GemmOps& g, const MxParts& mp,
                    int as_group, int as_rs, int tn, int sk) {
    const long long plane = g.M * g.K * 2;
    uint16_t* adq = (a->scratch && a->scratch_bytes >= plane) ? (uint16_t*)a->scratch : nullptr;
    const long kb = (long)((g.K + 127) / 128);
    hipStream_t st = (hipStream_t)s;
    for (int p = 0; p < mp.n; ++p) {
        const long long n0 = mp.n0[p], np = mp.n1[p] - mp.n0[p];
        const int lda = (int)(mp.lda ? mp.lda : g.K);
        const unsigned char* ap = (const unsigned char*)halo_addr(g.a) + p * g.K;
        if (adq) {
            const long long total = g.M * (g.K / 16);
            hipLaunchKernelGGL(mxt2_adq_kernel, dim3((unsigned)((total + 255) / 256)), dim3(256), 0,
                               st, ap, adq, (int)g.M, (int)g.K, lda);
        }
        if (mxt2_launch(ap, adq, (const float*)(halo_addr(g.ascale) + p * kb * 4),
                        (const unsigned char*)halo_addr(g.b) + n0 * g.K / 2,
                        (const unsigned char*)halo_addr(g.bscale) + n0,
                        (const unsigned char*)halo_addr(g.bref) + n0,
                        (__hip_bfloat16*)(halo_addr(g.y) + n0 * 2), (int)g.M, (int)g.K, (int)np,
                        lda, as_group == 128, as_rs, (int)g.N, (int)g.N, tn, (int)(g.K / sk), st))
            return HALO_NO(RAD_E_DEVICE);
    }
    return RAD_OK;
}

/* nt_m64's bytes at `sk` splits, past one row tile of the decode band: the gemv rows at the gemv
 * split law's SK (halo_gemm_mxfp4a8_gemv.hip) and the decode rows at one split
 * (halo_gemm_mxfp4a8_decode.hip). RAD_E_UNSUPPORTED when this kernel does not take the shape or
 * the scratch holds no plane (the caller then runs its own kernel). */
int halo_mxfp4a8_tiled_split(const RadArgs* a, RadStream s, const GemmOps& g, const MxParts& mp,
                             int as_group, int as_rs, int sk) {
    if (g.M > 64 || sk < 1 || g.K % MXT2_BK || g.K % ((long long)sk * HALO_MXT_GROUP) ||
        !a->scratch || a->scratch_bytes < g.M * g.K * 2)
        return RAD_E_UNSUPPORTED;
    return mxt2_run(a, s, g, mp, as_group, as_rs, 0, sk);
}

// ============================================================== the plugin ABI entry point
// For a partitioned weight (mx_parts), each part's activation copy is decoded and multiplied in
// turn. The `tn` tunable now picks the tile (see mxt2_launch); it never changes the bytes.
extern "C" int halo_rad_gemm_mxfp4a8_tiled(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1, /*parts=*/1);
    if (rc != RAD_OK) return rc;
    if (g.N < 16 || g.N % 16) return HALO_NO(RAD_E_SHAPE);
    if (!g.ascale) return HALO_NO(RAD_E_INVAL);
    if (!g.bref) return HALO_NO(RAD_E_INVAL);
    MxParts mp;
    rc = mx_parts(a, &g, &mp);
    if (rc != RAD_OK) return rc;
    if (g.K % HALO_MXT_KUNIT) return HALO_NO(RAD_E_SHAPE);
    int as_group = 0, as_rs = 1;
    rc = mx_ascale_grid(g, &as_group, &as_rs, mp.n);
    if (rc != RAD_OK) return rc;
    return mxt2_run(a, s, g, mp, as_group, as_rs, (int)halo_geti_or(a, "tn", 0), 1);
}

/* The row's scratch: the exact bf16 activation plane, M * K * 2, at the band's widest M, while it
 * fits MXT2_PLANE_CAP (one plane serves every part: parts run in order on the op's stream). */
extern "C" int64_t halo_rad_gemm_mxfp4a8_tiled_plane_scratch(const RadArgs* a) {
    const long long M = halo_geti_or(a, "M", 0), K = halo_geti_or(a, "K", 0);
    if (M < 1 || K < 1) return 0;
    const long long b = M * K * 2;
    return b <= MXT2_PLANE_CAP ? b : 0;
}

/* The fp8 fragment plane halo_mxfp4a8_tile_a writes, in whole 16-row fragments: what the
 * w4a8i8/w5a8i8 tiled rows size with this hook (the mxfp4 row itself now asks for
 * halo_rad_gemm_mxfp4a8_tiled_plane_scratch). */
extern "C" int64_t halo_rad_gemm_mxfp4a8_tiled_scratch(const RadArgs* a) {
    const long long M = halo_geti_or(a, "M", 0), K = halo_geti_or(a, "K", 0);
    return (int64_t)halo_mxfp4a8_tile_a_bytes((int)M, (int)K);
}
