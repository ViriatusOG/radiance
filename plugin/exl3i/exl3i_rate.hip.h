// exl3i_rate.hip.h -- the rate policies of exl3i's kernels: how a lane finds and decodes its fragment of one tile at
// rate m / 16 in codebook CB. The kernels (exl3i_nt.hip.h) take one as a template parameter R:
//   R::CBV              the codebook (the rate-free kernels are instantiated once a codebook translation unit)
//   R::NW               words a run of four positions reads (2 or 3)
//   R::K4               the K = 4 three-load form applies (integer K 4 only)
//   R::words(m)         u32 a tile (a constant for integer K, so the addressing folds)
//   R(lane, m)          the lane's plan: woff[2][NW] (word indices), the decode's state
//   decode(w)           the lane's v8h A fragment from its words w[run][x] = tile word woff[run][x]
//   decode8(w, d)       the same eight values as int8 grid indices (wdec8; mad: q): d[hw] byte e =
//                       element 4 hw + e
//   madh(w, h)          mad only: the same eight positions' masked hashes (s A + B) & 0x3F3F3F3F, h[e] for element e
//                       (k = exl3i_frag16_k(lane, e)): the byte-sum GEMV's operands (exl3i_mad.hip.h), q = bytesum - 126
//
// RateInt<BITS, CB>: integer K. Plan<BITS> (exl3r4d_common.hip.h) gives the lane's words and shifts; mad decodes by the
// same state extraction (exl3i_hash_run) and its byte-sum hash. The codebook parameter is mad's alone (one codebook).
// RateFrac<CB, MODE>: fractional and sub-1 rates, the plan (exl3i_plan.h, host-checked by tests/exl3i_plan_test.c) read
// from a table the compiler builds (exl3i_plan_tab.h, tests/exl3i_plan_tab_test.cpp): MODE 0 KA 1..4 (two words), 1 KA 0
// (two words, the 0-bit steps' salt), 2 KA 5..7 (three).
// Values: common/exl3i_decode.h's codebooks, bit for bit (tests/exl3i_gpu_test.hip).
#pragma once
#include "exl3r4d_common.hip.h"
#include "exl3i_plan.h"
#include "exl3i_plan_tab.h"
#include "exl3i_a8.h"
#include "exl3i_decode.h"   // EXL3I_MAD_A / _B

// mad, the one codebook (common/exl3i_decode.h).
enum { EXL3I_DEV_CB_MAD = 0 };

// mad (exl3i_decode.h): one 16-bit state's masked hash, ((s A + B) mod 2^32) & 0x3F3F3F3F, A = 0xD0474B < 2^24. Plain C:
// every caller's s is a ubfe / mask of 16 bits, so LLVM proves s A fits the u24 multiply and selects v_mad_u32_u24 itself,
// and, unlike an asm statement, models its latency (s_delay_alu; an asm form ran the GEMV about 15% slower).
static __device__ __forceinline__ unsigned exl3i_mad_hash(unsigned s) {
  return ((s & 0xFFFFu) * EXL3I_MAD_A + EXL3I_MAD_B) & 0x3F3F3F3Fu;
}

// RateFrac's lane plans, built by the compiler (exl3i_plan_tab.h; one copy a translation unit).
static __constant__ exl3i_pt::Tab exl3i_plan_tab = exl3i_pt::build();

// mad's grid step: its values are q 4 K_INV for the int8 index q (common/exl3i_decode.h).
#define EXL3I_I8_STEP 0x26EEu   /* fp16 4 K_INV */

// mad's fp16 pair from two hashes (h0 low): fp16(q 4 K_INV), q = S - 126 exact in fp16 (fp16(1024 + S) - 1150), one
// rounding of q times the step: the oracle's bits (q 4 K_INV is exact in f32; a zero is +0 both ways).
template <int CB>
static __device__ __forceinline__ unsigned exl3i_fin16(unsigned h0, unsigned h1) {
  static_assert(CB == EXL3I_DEV_CB_MAD, "the dense kernels serve mad");
  const unsigned a = sad_hi(h1, sad_lo(h0, 0x64006400u));   // fp16(1024 + S)
  const _Float16 ws = __builtin_bit_cast(_Float16, (unsigned short)0x26EE);   // 4 K_INV
  const v2h q = __builtin_bit_cast(v2h, a) - (v2h){(_Float16)1150.0f, (_Float16)1150.0f};
  return __builtin_bit_cast(unsigned, q * (v2h){ws, ws});
}

// DQ (the decode form, mad): the pair's int8 grid indices q = S - 126 as exact fp16 integers, unscaled -- the kernel
// multiplies its f32 sums by EXL3I_I8_STEP once, so the weight is q * step exactly. One packed subtract.
template <int CB>
static __device__ __forceinline__ unsigned exl3i_fin16q(unsigned h0, unsigned h1) {
  static_assert(CB == EXL3I_DEV_CB_MAD, "DQ is mad's");
  const unsigned a = sad_hi(h1, sad_lo(h0, 0x64006400u));   // fp16(1024 + S)
  return __builtin_bit_cast(unsigned, __builtin_bit_cast(v2h, a) - (v2h){(_Float16)1150.0f, (_Float16)1150.0f});
}
template <int CB, bool Q>
static __device__ __forceinline__ unsigned exl3i_fin16x(unsigned h0, unsigned h1) {
  if constexpr (Q) return exl3i_fin16q<CB>(h0, h1);
  else return exl3i_fin16<CB>(h0, h1);
}

// mad's int8 indices of two hashes as (n0 | n1 << 16), n = q + 128 = S + 2 in [2, 254] (exl3i_pack8 makes bytes).
template <int CB>
static __device__ __forceinline__ unsigned exl3i_fin8(unsigned h0, unsigned h1) {
  static_assert(CB == EXL3I_DEV_CB_MAD, "the int8 indices are mad's");
  return sad_hi(h1, sad_lo(h0, 0x00020002u));
}

// Two exl3i_fin8 pairs (elements 0, 1 in lo, 2, 3 in hi) -> four int8 bytes.
static __device__ __forceinline__ unsigned exl3i_pack8(unsigned lo, unsigned hi) {
  return __builtin_amdgcn_perm(hi, lo, 0x06040200u) ^ 0x80808080u;
}

// Four states -> their hashes (mad's one state at a time, masked).
template <int CB>
static __device__ __forceinline__ void exl3i_hash4(const unsigned (&st)[4], unsigned (&h)[4]) {
  static_assert(CB == EXL3I_DEV_CB_MAD, "the state hash is mad's");
#pragma unroll
  for (int i = 0; i < 4; i++) h[i] = exl3i_mad_hash(st[i]);
}

// The run's state extraction up to the hashes (the same words and shifts as the lane plan), for the codebook: the run's
// four positions q0 .. q0 + 3 -> h[0..3]. mad hashes a state at a time from its own 16 bits (a ubfe each).
template <int BITS, int NW, int CB = EXL3I_DEV_CB_MAD>
static __device__ __forceinline__ void exl3i_hash_run(const unsigned (&w)[NW], unsigned s, unsigned (&h)[4]) {
  const unsigned lo = __builtin_amdgcn_alignbit(w[NW - 2], w[NW - 1], s);   // the 32 stream bits ending at e3
  unsigned mid = 0;
  if constexpr (3 * BITS + 16 > 32) {
    const unsigned hi = NW == 3 ? __builtin_amdgcn_alignbit(w[0], w[1], s) : (w[0] >> s);
    mid = __builtin_amdgcn_alignbit(hi, lo, 16);   // bits 16..47 above e3 - 48
  }
  unsigned st[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int off = (3 - i) * BITS;   // state i ends off bits below e3
    st[i] = off + 16 <= 32 ? __builtin_amdgcn_ubfe(lo, off, 16) : __builtin_amdgcn_ubfe(mid, off - 16, 16);
  }
  exl3i_hash4<CB>(st, h);
}

// The byte-sum GEMV's order: run r's state i is element e = 2 r + i (i < 2) or 4 + 2 r + i - 2 (the fragment's order).
static __device__ __forceinline__ void exl3i_madh_put(const unsigned (&hr)[4], int r, unsigned (&h)[8]) {
  h[2 * r] = hr[0];
  h[2 * r + 1] = hr[1];
  h[4 + 2 * r] = hr[2];
  h[5 + 2 * r] = hr[3];
}

template <int BITS, int CB>
struct RateInt {
  static constexpr int CBV = CB;
  static constexpr int NW = Plan<BITS>::NW;
  static constexpr bool K4 = BITS == 4;
  static __device__ __forceinline__ constexpr int words(int) { return Plan<BITS>::WORDS; }
  int woff[2][NW];
  unsigned shift[2];
  __device__ __forceinline__ RateInt(int lane, int) {
    const Plan<BITS> P(lane);
#pragma unroll
    for (int r = 0; r < 2; r++) {
      shift[r] = P.shift[r];
#pragma unroll
      for (int x = 0; x < NW; x++) woff[r][x] = P.woff[r][x];
    }
  }
  // Q: mad's unscaled indices (exl3i_fin16q; the decode form's DQ), else the codebook's fp16 values.
  template <bool Q = false>
  __device__ __forceinline__ v8h decode(const unsigned (&w)[2][NW]) const {
    static_assert(!Q || CB == EXL3I_DEV_CB_MAD, "DQ is mad's");
    // element order of the fragment: run 0 -> u[0], u[2]; run 1 -> u[1], u[3]
    union { unsigned u[4]; v8h f; } x;
#pragma unroll
    for (int r = 0; r < 2; r++) {
      unsigned h[4];
      exl3i_hash_run<BITS, NW, CB>(w[r], shift[r], h);
      x.u[r] = exl3i_fin16x<CB, Q>(h[0], h[1]);
      x.u[2 + r] = exl3i_fin16x<CB, Q>(h[2], h[3]);
    }
    return x.f;
  }
  __device__ __forceinline__ void decode8(const unsigned (&w)[2][NW], unsigned (&d)[2]) const {
    unsigned p[4];
#pragma unroll
    for (int r = 0; r < 2; r++) {
      unsigned h[4];
      exl3i_hash_run<BITS, NW, CB>(w[r], shift[r], h);
      p[r] = exl3i_fin8<CB>(h[0], h[1]);
      p[2 + r] = exl3i_fin8<CB>(h[2], h[3]);
    }
    d[0] = exl3i_pack8(p[0], p[1]);
    d[1] = exl3i_pack8(p[2], p[3]);
  }
  // mad: the eight masked hashes in element order (the byte-sum GEMV)
  __device__ __forceinline__ void madh(const unsigned (&w)[2][NW], unsigned (&h)[8]) const {
    static_assert(CB == EXL3I_DEV_CB_MAD, "madh is mad's");
#pragma unroll
    for (int r = 0; r < 2; r++) {
      unsigned hr[4];
      exl3i_hash_run<BITS, NW, CB>(w[r], shift[r], hr);
      exl3i_madh_put(hr, r, h);
    }
  }
};

template <int CB, int MODE>
struct RateFrac {
  static constexpr int CBV = CB;
  static constexpr int NW = MODE == 2 ? 3 : 2;
  static constexpr bool K4 = false;
  static __device__ __forceinline__ int words(int m) { return m / 2; }
  // off[r][i]: state i's offset below e3 (i 0..2; state 3 ends at e3, offset 0); xp[r]: the run's salts as the packed
  // pairs (0, 1), (2, 3) (MODE 1; 0 otherwise)
  int woff[2][NW];
  unsigned shift[2], off[2][3], xp[2][2];
  __device__ __forceinline__ RateFrac(int lane, int m) {
    // the lane's four dwords (and the salts at sub-1), read once
    const uint4 e = *(const uint4*)exl3i_plan_tab.e[m >> 1][lane];
    const unsigned ev[4] = {e.x, e.y, e.z, e.w};
    unsigned xv[4] = {0, 0, 0, 0};
    if constexpr (MODE == 1) {
      const uint4 x = *(const uint4*)exl3i_plan_tab.xr[m >> 1][lane];
      xv[0] = x.x; xv[1] = x.y; xv[2] = x.z; xv[3] = x.w;
    }
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const unsigned w = ev[2 * r], o = ev[2 * r + 1];
      shift[r] = w >> 24;
#pragma unroll
      for (int x = 0; x < NW; x++) woff[r][x] = (int)((w >> (8 * (3 - NW + x))) & 0xFFu);
#pragma unroll
      for (int i = 0; i < 3; i++) off[r][i] = (o >> (8 * i)) & 0xFFu;
      xp[r][0] = xv[2 * r];
      xp[r][1] = xv[2 * r + 1];
    }
  }
  // The run's four states as the packed pairs the hash takes: p01 = s0 | s1 << 16, p23 = s2 | s3 << 16 (fractional K).
  // State i is the 16 bits at off[i] of the window ending at e3: lo (32 bits) at KA <= 4, {hi, lo} (64) at KA 5..7 (off up
  // to 24), so a_i = (window >> off[i]) is one shift / alignbit (v_alignbit reads the shift's low five bits: off < 32), and
  // each pair one v_perm of two low halves. Four ubfe and two lshl_or per run (KA <= 4) cost more, and at KA 5..7 the
  // offset's side of 16 is a lane's run-time value, so the ubfe forms would need a select per state.
  __device__ __forceinline__ void pairs(const unsigned (&w)[NW], int r, unsigned& p01, unsigned& p23) const {
    const unsigned lo = __builtin_amdgcn_alignbit(w[NW - 2], w[NW - 1], shift[r]);   // the 32 stream bits ending at e3
    unsigned a[3];
    if constexpr (MODE == 2) {
      const unsigned hi = __builtin_amdgcn_alignbit(w[0], w[1], shift[r]);           // the 32 above them
#pragma unroll
      for (int i = 0; i < 3; i++) a[i] = __builtin_amdgcn_alignbit(hi, lo, off[r][i]);
    } else {
#pragma unroll
      for (int i = 0; i < 3; i++) a[i] = lo >> off[r][i];
    }
    p01 = __builtin_amdgcn_perm(a[1], a[0], 0x05040100u);
    p23 = __builtin_amdgcn_perm(lo, a[2], 0x05040100u);
    if constexpr (MODE == 1) { p01 ^= xp[r][0]; p23 ^= xp[r][1]; }
  }
  // A packed pair of states -> their hashes (low, high).
  static __device__ __forceinline__ void hash_p(unsigned p, unsigned& ha, unsigned& hb) {
    static_assert(CB == EXL3I_DEV_CB_MAD, "the state hash is mad's");
    ha = exl3i_mad_hash(p & 0xFFFFu);
    hb = exl3i_mad_hash(p >> 16);
  }
  // The run's four hashes. mad at KA >= 1: each state its own 16 bits straight from the window (a ubfe of lo at KA <= 4,
  // where every offset is <= 15; an alignbit of {hi, lo} and a mask at KA 5..7), no packed pairs; the sub-1 rates (their
  // salts are packed pairs) through pairs().
  __device__ __forceinline__ void hashes(const unsigned (&w)[NW], int r, unsigned (&h)[4]) const {
    if constexpr (CB == EXL3I_DEV_CB_MAD && MODE != 1) {
      const unsigned lo = __builtin_amdgcn_alignbit(w[NW - 2], w[NW - 1], shift[r]);
      unsigned st[4];
      if constexpr (MODE == 2) {
        const unsigned hi = __builtin_amdgcn_alignbit(w[0], w[1], shift[r]);
#pragma unroll
        for (int i = 0; i < 3; i++) st[i] = __builtin_amdgcn_alignbit(hi, lo, off[r][i]) & 0xFFFFu;
      } else {
#pragma unroll
        for (int i = 0; i < 3; i++) st[i] = __builtin_amdgcn_ubfe(lo, off[r][i], 16);
      }
      st[3] = lo & 0xFFFFu;
      exl3i_hash4<CB>(st, h);
    } else {
      unsigned p01, p23;
      pairs(w, r, p01, p23);
      hash_p(p01, h[0], h[1]);
      hash_p(p23, h[2], h[3]);
    }
  }
  // mad: the eight masked hashes in element order (the byte-sum GEMV)
  __device__ __forceinline__ void madh(const unsigned (&w)[2][NW], unsigned (&h)[8]) const {
    static_assert(CB == EXL3I_DEV_CB_MAD, "madh is mad's");
#pragma unroll
    for (int r = 0; r < 2; r++) {
      unsigned hr[4];
      hashes(w[r], r, hr);
      exl3i_madh_put(hr, r, h);
    }
  }
  // element order of the fragment: run 0 -> u[0], u[2]; run 1 -> u[1], u[3]
  template <bool Q = false>
  __device__ __forceinline__ v8h decode(const unsigned (&w)[2][NW]) const {
    static_assert(!Q || CB == EXL3I_DEV_CB_MAD, "DQ is mad's");
    union { unsigned u[4]; v8h f; } x;
#pragma unroll
    for (int r = 0; r < 2; r++) {
      unsigned h[4];
      hashes(w[r], r, h);
      x.u[r] = exl3i_fin16x<CB, Q>(h[0], h[1]);
      x.u[2 + r] = exl3i_fin16x<CB, Q>(h[2], h[3]);
    }
    return x.f;
  }
  __device__ __forceinline__ void decode8(const unsigned (&w)[2][NW], unsigned (&d)[2]) const {
    unsigned p[4];
#pragma unroll
    for (int r = 0; r < 2; r++) {
      unsigned h[4];
      hashes(w[r], r, h);
      p[r] = exl3i_fin8<CB>(h[0], h[1]);
      p[2 + r] = exl3i_fin8<CB>(h[2], h[3]);
    }
    d[0] = exl3i_pack8(p[0], p[1]);
    d[1] = exl3i_pack8(p[2], p[3]);
  }
};
