/* Derived from libr4d/r4d_dt16.h by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_dt16.h -- 16-bit operand traits for the attention kernels, gfx1151.
//
// This is the gfx11 counterpart of libr4d's DT16 layer, and the difference is the
// whole porting story for attention. The two architectures disagree on what one
// lane holds (halo_wmma.h states the layouts and how they were measured):
//
//                        a lane's A / B fragment      the k its element e holds
//   gfx1201 (RDNA4)      8 x 16-bit (4 dwords)       k = 8*(half) + e, half = lane>>4
//   gfx1151 (RDNA3.5)   16 x 16-bit (8 dwords)       k = e   (lanes 16-31 replicate)
//
// libr4d's attention kernels are built around a transposed score matrix
// S^T = K.Q^T so that the D column -- the query -- stays (lane & 15) on both
// architectures; halo_wmma.h notes that this is why those kernels are portable
// at all. What changes is the ROW: on gfx12 a lane's accumulator element e is
// key 8*half + e of the 16-key tile, on gfx11 it is key 2*e + (lane>>4). Both
// spread the tile's sixteen keys over the half-pair, so the kernels' existing
// swap16 (lane ^ 16) row-max / row-sum reductions keep their meaning; a port
// changes the key index and the fragment width, not the softmax algebra.
//
//   * mk()        takes the lane's EIGHT dwords (16 values) instead of four.
//   * gather      reads 32 bytes at the un-split k offset (no 8*half term).
//   * P fragments the softmax's eight values are keys 2e+half; a 16-consecutive
//                 key fragment is assembled across the half-pair (see p16w).
//   * store       each lane writes its eight values at head dim 2e+half.
//
// Both sources are exact in either 16-bit format (an fp8 e4m3 K/V has 3 mantissa
// bits, bf16 7, f16 10), so DT16<0> (bf16) and DT16<1> (f16) stay interchangeable
// for correctness; they differ in range and in convert cost, as r4d_dt16.h says.
// gfx1151 has no v_cvt_pk_bf16_f32 either, so DT16<0>::pk is the software RTNE
// halo_wmma.h already uses, and f16 packs two floats in one instruction.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "halo_common.h"
#include "halo_wmma.h"       // v16bf / v16hf / v8f, halo::mma, pk_bf16, pk_f16, f32_to_bf16
#include "halo_fp8_wmma.h"   // e4m3_pk: the exact e4m3 unpack (no fp8 convert insts on gfx11)

typedef __fp16 v2hf_dt __attribute__((ext_vector_type(2)));

// GROW = how many octaves the running row max may exceed the reference max before the
// accumulator must be rescaled, exactly as r4d_dt16.h defines it for each format.
template <int F16> struct DT16;

template <> struct DT16<0> {                     // bf16: exponent range of f32
    typedef v16bf frag;
    static constexpr float SHIFT = 0.0f;
    static constexpr float GROW  = 60.0f;        // 2^60 * |v| * n stays far under f32 max

    // From the lane's eight dwords, i.e. its full sixteen-value k run in order.
    __device__ __forceinline__ static frag mk(const uint32_t w[8]) {
        union { uint32_t u[8]; frag f; } q;
#pragma unroll
        for (int i = 0; i < 8; ++i) q.u[i] = w[i];
        return q.f;
    }
    __device__ __forceinline__ static v8f wmma(frag a, frag b, v8f c) { return halo::mma(a, b, c); }
    __device__ __forceinline__ static uint32_t pk(float x, float y) { return halo::pk_bf16(x, y); }
    // Two f32 to two bf16 in a dword, element order (no lane-half permutation on gfx11).
    __device__ __forceinline__ static uint32_t pkv(v2f x) {
        union { v2f f; float e[2]; } s;
        s.f = x;
        return halo::pk_bf16(s.e[0], s.e[1]);
    }
    __device__ __forceinline__ static uint32_t from_bf16w(uint32_t w) { return w; }   // identity
};

template <> struct DT16<1> {                     // f16: 3 more mantissa bits, 1 convert instead of 6
    typedef v16hf frag;
    static constexpr float SHIFT = 0.0f;
    static constexpr float GROW  = 14.0f;        // p <= 2^14 = 16384 < 65504

    __device__ __forceinline__ static frag mk(const uint32_t w[8]) {
        union { uint32_t u[8]; frag f; } q;
#pragma unroll
        for (int i = 0; i < 8; ++i) q.u[i] = w[i];
        return q.f;
    }
    __device__ __forceinline__ static v8f wmma(frag a, frag b, v8f c) { return halo::mma(a, b, c); }
    __device__ __forceinline__ static uint32_t pk(float x, float y) { return halo::pk_f16(x, y); }
    __device__ __forceinline__ static uint32_t pkv(v2f x) {
        union { v2f f; float e[2]; } s;
        s.f = x;
        return halo::pk_f16(s.e[0], s.e[1]);
    }
    // two bf16 in a dword -> two f16 in a dword (bf16->f32 is a shift, so three VALU).
    __device__ __forceinline__ static uint32_t from_bf16w(uint32_t w) {
        return halo::pk_f16(__builtin_bit_cast(float, w << 16),
                            __builtin_bit_cast(float, w & 0xffff0000u));
    }
};

// Widen sixteen fp8 (four dwords) to the lane's eight 16-bit dwords, in element order.
// On gfx12 the same widening is eight values with a lane-half permute; on gfx11 every lane
// holds the whole 16-wide run, so this is the four dwords twice over and no permute.
template <class D>
__device__ __forceinline__ void fp8x16_to_16x8w(uint4 raw, uint32_t w8[8]) {
    const uint32_t d[4] = { raw.x, raw.y, raw.z, raw.w };
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        w8[q * 2 + 0] = D::pkv(halo_fp8::e4m3_pk<false>(d[q]));
        w8[q * 2 + 1] = D::pkv(halo_fp8::e4m3_pk<true>(d[q]));
    }
}

// Widen eight fp8 (two dwords) to four 16-bit dwords in element order -- the narrow form,
// kept for the kernels that stage fewer than sixteen k at a time (r4d's fp8x8_to_16x4w).
template <class D>
__device__ __forceinline__ void fp8x8_to_16x4w(uint2 raw, uint32_t w4[4]) {
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int wv = (q == 0) ? (int)raw.x : (int)raw.y;
        w4[q * 2 + 0] = D::pkv(halo_fp8::e4m3_pk<false>(wv));
        w4[q * 2 + 1] = D::pkv(halo_fp8::e4m3_pk<true>(wv));
    }
}

/* Eight k6_v4 codes as four f16 dwords, dims (2i, 2i+1) in dword i -- r4d_dt16.h's
 * k6x8_to_f16x4w, unchanged: the unpack is exact integer/scale arithmetic and touches no
 * WMMA layout, so it ports as it stands. `half` selects which eight of the sixteen dims a
 * caller wants; a gfx1151 16-wide fragment asks for both halves into w8[0..7]. */
__device__ __forceinline__ void k6x8_to_f16x4w(uint32_t lo, uint32_t hb, int half, uint32_t s2,
                                               uint32_t* w4) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    const uint32_t ev = ((lo & 0x0F0F0F0Fu) | ((half ? hb : hb << 4) & 0x30303030u)) ^ 0x20202020u;
    const uint32_t od = (((lo >> 4) & 0x0F0F0F0Fu) | ((half ? hb >> 2 : hb << 2) & 0x30303030u))
                      ^ 0x20202020u;
    const h2 s = __builtin_bit_cast(h2, s2);
    const h2 bias = { (_Float16)1056.0f, (_Float16)1056.0f };
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const uint32_t sel = (uint32_t)b | 0x0c00u | ((uint32_t)(4 + b) << 16) | 0x0c000000u;
        const uint32_t x = __builtin_amdgcn_perm(od, ev, sel) | 0x64006400u;
        w4[b] = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2, x) - bias) * s);
    }
}

/* Build a 16-consecutive-key P fragment from the lane's eight softmax values.
 *
 * On gfx1151 a lane's accumulator element e is key 2*e + (lane>>4) of the tile, so a lane holds
 * the keys of ONE parity and the other eight live in lane ^ 16 -- the same split the kernels
 * already reduce across for the row max and row sum. The WMMA's A operand wants the sixteen keys
 * in order, so the halves are interleaved here: four lane-16 exchanges move the partner's four
 * dwords across, then key k takes element k>>1 from whichever half owns parity k&1. `w` is the
 * lane's own four dwords (its eight keys in order); `p16` the assembled eight. */
template <class D>
__device__ __forceinline__ void p16w(const uint32_t w[4], uint32_t p16[8]) {
    (void)sizeof(D);
    const int mine_h = (int)((threadIdx.x >> 4) & 1);   // the parity this lane holds
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = halo_xor_lane_u32<16>(w[i]);
#pragma unroll
    for (int i = 0; i < 8; ++i) p16[i] = 0u;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int e = k >> 1;                                 // accumulator element index
        const uint32_t src = (((k & 1) == mine_h) ? w : o)[e >> 1];
        const uint32_t v16 = (e & 1) ? (src >> 16) : (src & 0xFFFFu);
        p16[k >> 1] |= v16 << ((k & 1) ? 16 : 0);
    }
}

/* The store side of the same mapping: project a 16-consecutive-dim accumulator onto the lane's
 * eight values. O's row is the head dim and its column the query (lane & 15), so a lane's
 * element e is head dim 2*e + (lane>>4) -- the values it packs are at even (or odd) dims in
 * order, and each is written at its own offset. Returned as the lane's eight 16-bit values in
 * element order; a caller stores element e at dim 2*e + (lane>>4). */
template <class D>
__device__ __forceinline__ void o8w(const float acc[8], float inv, uint32_t w4[4]) {
    uint32_t v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = D::pk(acc[e] * inv, 0.0f) & 0xFFFFu;
#pragma unroll
    for (int i = 0; i < 4; ++i) w4[i] = v[i * 2] | (v[i * 2 + 1] << 16);
}

/* Did any lane of the wave take the branch? The row-max test is wave-wide because the
 * accumulator rescale it guards must be taken by the whole half-pair together. */
__device__ __forceinline__ int wave_any(bool c) {
    return (int)__builtin_amdgcn_ballot_w32(c);
}
