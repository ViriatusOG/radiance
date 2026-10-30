/* Derived from libr4d/r4d_moe.hip by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. */
// halo_router_topk.h -- the softmax-plus-exact-top_k row of router_topk_bf16, shared by the
// standalone selection (halo_router_topk_bf16.hip) and its fused scatter forms
// (halo_router_scatter.hip, halo_router_gemm_scatter.hip). One definition, so a fused selection
// and the standalone one rank a row bit-identically; that is what halo_selftest's
// router_topk_scatter case asserts. It is a header for the reason PORTING.md gives: two units
// defining the same device function only works if the compiler sees the same body.
//
// Every reduction here is a wave butterfly on halo_xor_lane_u32 (halo_common.h), the same value on
// gfx1151 as on gfx1201 -- an unsigned minimum of a packed (key, ~id) word, so the combine order
// does not matter and the answer is the reference's, bit for bit. There is no WMMA leg and no
// fragment order in this op.
//
// Nothing here is a sort. Round r of the selection takes the largest key below round r-1's, which
// is exactly libref's repeated-maximum scan with its ties-to-the-lower-id rule, and parks the
// winner on lane r (top_k <= the wave, refused above). The keys are the logits' order-preserving
// bits over ~expert, so equal logits order to the lower expert id.
#ifndef HALO_ROUTER_TOPK_H
#define HALO_ROUTER_TOPK_H

#include <hip/hip_runtime.h>
#include <cstdint>

#include "halo_common.h"
#include "halo_moe_sort.h"

namespace halo_routertopk {
namespace {

constexpr uint32_t kWave = 32;

/* The expert-count ceiling and the sort's protected-expert list belong to the sort this op hands
 * its slots to, so they are halo_moe_sort.h's rather than a second copy. */
using halo_moesort::kMaxExpert;
using halo_moesort::Protect;
using halo_moesort::sort_place;

/* The key a lane compares: the logit's order-preserving 16-bit image over ~expert, so a larger key
 * is a larger logit and, on a tie, the lower expert. 0 is never selected (absent, NaN, or taken).
 *
 * Same answer as a two-register comparison, with two special cases handled: -0.0 compares equal to
 * +0.0 but has a different pattern, so it is folded to +0.0; NaN never wins a float comparison, but
 * a positive NaN's pattern sorts above +inf, so NaN gets key 0 (never selected; an all-NaN row
 * takes the `nothing left` exit). */
__device__ __forceinline__ uint32_t topk_key(uint16_t raw, uint32_t e) {
    const bool nan = ((raw & 0x7F80u) == 0x7F80u) && (raw & 0x7Fu);
    uint16_t u = (raw == 0x8000u) ? (uint16_t)0u : raw;          /* -0.0 folds to +0.0 */
    const uint16_t k = (u & 0x8000u) ? (uint16_t)~u : (uint16_t)(u | 0x8000u);
    return nan ? 0u : (((uint32_t)k << 16) | (~e & 0xFFFFu));
}
__device__ __forceinline__ uint16_t topk_key_raw(uint32_t key) {
    const uint16_t k = (uint16_t)(key >> 16);
    return (k & 0x8000u) ? (uint16_t)(k & 0x7FFFu) : (uint16_t)~k;
}

/* The selection's wave minimum on the DPP/permlane exchange rather than `__shfl_xor` (an LDS round
 * trip per stage, run top_k times a row with nothing to hide it). An unsigned minimum is
 * order-independent. */
__device__ __forceinline__ uint32_t wave_min_u32(uint32_t b) {
    uint32_t o;
    o = halo_xor_lane_u32<16>(b); b = b < o ? b : o;
    o = halo_xor_lane_u32<8>(b);  b = b < o ? b : o;
    o = halo_xor_lane_u32<4>(b);  b = b < o ? b : o;
    o = halo_xor_lane_u32<2>(b);  b = b < o ? b : o;
    o = halo_xor_lane_u32<1>(b);  b = b < o ? b : o;
    return b;
}

/* In the fused kernel a row also hands its slots to the sort through `e`: each slot's expert
 * relative to the slice, or -1 for another rank's. The standalone selection passes a null `e`. */
struct SlotSink {
    int32_t* e;
    int32_t  base;
    uint32_t n_local;
    Protect  prot;
};

/* MAXOWN is how many of a row's logits one lane holds (n_expert <= MAXOWN * kWave). It is a
 * template parameter because they are registers: 16 per lane at 512 experts, 8 at 256.
 *
 * The trip count is MAXOWN, not a lane-derived `n_own`: a lane-derived count makes every later
 * `v[i]` a divergent register index, which the machine implements as a v_movrels waterfall loop,
 * costing tens of microseconds to rank 512 numbers. Unrolled to the constant bound, the array stays
 * in registers. Slots past `n_expert` hold -INFINITY with key 0, which keeps the result
 * bit-identical: fmaxf ignores -inf, __expf(-inf - mx) is exactly zero, and the selection never
 * picks key 0 while a real entry remains.
 *
 * All MAXOWN loads issue before any is waited on. A guarded `if (ok) raw = row[e]` would put each
 * load in its own exec-masked branch and serialise the round trips, so a slot past `n_expert` reads
 * the row's first logit (always in bounds) and the select discards it.
 *
 * One row per wave, with nothing outside the wave: reductions are butterflies over the row's 32
 * lanes and stores are addressed by `m`, so the grid shape cannot change the answer. That lets the
 * fused kernels run this same code from a one-workgroup grid, byte-identically. */
template <uint32_t MAXOWN>
__device__ __forceinline__ void router_topk_row(
    const uint16_t* __restrict__ lg, int32_t* __restrict__ ids, uint16_t* __restrict__ w,
    uint32_t m, uint32_t n_expert, uint32_t top_k, uint32_t lg_ld, int norm, uint32_t lane,
    SlotSink snk) {

    const uint16_t* row = lg + (size_t)m * lg_ld;
    uint16_t raw[MAXOWN];
#pragma unroll
    for (uint32_t i = 0; i < MAXOWN; ++i) {
        const uint32_t e = lane + i * kWave;
        raw[i] = row[e < n_expert ? e : 0u];
    }
    float    v[MAXOWN];
    uint32_t key[MAXOWN];    /* (order-preserving value, ~id); 0 = absent or already chosen */
#pragma unroll
    for (uint32_t i = 0; i < MAXOWN; ++i) {
        const uint32_t e  = lane + i * kWave;
        const bool     ok = e < n_expert;
        v[i]   = ok ? bf16_to_f32(raw[i]) : -INFINITY;
        key[i] = topk_key(raw[i], e) & (0u - (uint32_t)ok);
    }

    float mx = -INFINITY;
#pragma unroll
    for (uint32_t i = 0; i < MAXOWN; ++i) mx = fmaxf(mx, v[i]);
    mx = fmaxf(mx, halo_xor_lane<16>(mx));
    mx = fmaxf(mx, halo_xor_lane<8>(mx));
    mx = fmaxf(mx, halo_xor_lane<4>(mx));
    mx = fmaxf(mx, halo_xor_lane<2>(mx));
    mx = fmaxf(mx, halo_xor_lane<1>(mx));

    float sum = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < MAXOWN; ++i) sum += __expf(v[i] - mx);
    sum = halo_wave_sum(sum);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    /* ---- the selection stays in registers.
     *
     * The reference stores each selected probability as bf16 and, with `norm`, rescales the stored
     * value. Doing that literally would chain top_k stores and dependent loads on one lane.
     * Instead round r parks its winner on lane r (top_k <= kWave; both launchers refuse more).
     *
     * Round r takes the largest key below round r-1's, which equals zeroing the last winner (keys
     * are unique) but costs one subtraction: with P the last winner minus one, P - key is at most P
     * for keys not above P and wraps past P for taken keys, so the smallest P - key names the
     * largest untaken key. Key 0 (absent or NaN) yields P itself and is chosen only when nothing
     * real is left, the `nothing left` exit.
     *
     * Probabilities are computed after the loop on each winner's lane with the same expression, so
     * the same bits. `wsum` adds them in round order, as the reference does, and the
     * renormalisation reads the produced bf16, not the f32 behind it. */
    uint32_t P     = 0xFFFFFFFFu;     /* round 0 may take any key */
    uint32_t mykey = 0u;
    uint32_t nsel  = 0;
    for (uint32_t r = 0; r < top_k; ++r) {
        uint32_t d = 0xFFFFFFFFu;
#pragma unroll
        for (uint32_t i = 0; i < MAXOWN; ++i) {
            const uint32_t t = P - key[i];
            d = d < t ? d : t;
        }
        const uint32_t best = P - wave_min_u32(d);
        if ((best & 0xFFFFu) == 0u) break;    /* fewer live entries than top_k: nothing left */
        if (lane == r) mykey = best;
        P = best - 1u;
        ++nsel;
    }
    const float    p  = __expf(bf16_to_f32(topk_key_raw(mykey)) - mx) * inv;
    const uint16_t pb = f32_to_bf16(p);
    float wsum = 0.0f;
    for (uint32_t r = 0; r < nsel; ++r)
        wsum += __builtin_bit_cast(float,
                    __builtin_amdgcn_readlane(__builtin_bit_cast(int, p), (int)r));
    if (lane < nsel) {
        ids[(size_t)m * top_k + lane] = (int32_t)(~mykey & 0xFFFFu);
        w[(size_t)m * top_k + lane]   = (norm && wsum > 0.0f)
                                      ? f32_to_bf16(bf16_to_f32(pb) * (1.0f / wsum)) : pb;
    }
    if (snk.e && lane < top_k) {
        const int32_t e  = (int32_t)(~mykey & 0xFFFFu) - snk.base;
        const bool    ok = lane < nsel && e >= 0 && e < (int32_t)snk.n_local;
        snk.e[m * top_k + lane] = ok ? sort_place(e, snk.n_local, snk.prot) : -1;
    }
}

/* The block's wave count for a one-workgroup fused form: a wave per row, rounded to a power of two
 * between 8 and 32. A wave selects its rows serially, so a 32-row verify step at eight waves runs
 * four selections back to back, while extra waves cost ~0.4 us at four rows. Above 32 rows the
 * waves wrap. Shared by router_topk_scatter and router_sigmoid_topk_scatter. */
constexpr uint32_t scatter_waves(uint32_t M) { return M <= 8 ? 8u : M <= 16 ? 16u : 32u; }

}  /* namespace */
}  /* namespace halo_routertopk */

#endif  /* HALO_ROUTER_TOPK_H */
