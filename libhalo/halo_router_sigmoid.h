/* Derived from libr4d/r4d_glm.hip by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. */
// halo_router_sigmoid.h -- the sigmoid-routing row of router_sigmoid_topk_f32, shared by the
// standalone selection (halo_router_sigmoid_topk_f32.hip) and its fused scatter form
// (halo_router_scatter.hip), so the two rank a row bit-identically.
//
// p = sigmoid(logit); the top_k largest of (p + bias) are taken, equal ranks to the lower expert,
// as a serial scan with a strict `>` takes them; then mult = norm ? route_scale / (sum p + 1e-20) :
// route_scale and a weight is p * mult. A NaN rank is never selected.
//
// The only architectural content is in the reductions, and both are wave butterflies
// (halo_xor_lane_u64 / readlane) whose value does not depend on the architecture. The selection is
// the same p passes of a wave maximum that libref's repeated-maximum scan makes.
#ifndef HALO_ROUTER_SIGMOID_H
#define HALO_ROUTER_SIGMOID_H

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

#include "halo_common.h"

namespace halo_routersigmoid {
namespace {

constexpr int kSigmoidWave = 32;

/* The rank's order-preserving bits (-0 folded to +0) over ~expert, so equal ranks order to the
 * lower expert; 0 for a NaN. */
__device__ __forceinline__ unsigned long long router_rank_key(float rank, int e) {
    uint32_t u = __float_as_uint(rank);
    if (u == 0x80000000u) u = 0;
    const uint32_t k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return rank == rank ? ((unsigned long long)k << 32) | (uint32_t)~e : 0ull;
}
__device__ __forceinline__ unsigned long long router_wave_max(unsigned long long b) {
    unsigned long long o;
    o = halo_xor_lane_u64<16>(b);
    b = b > o ? b : o;
    o = halo_xor_lane_u64<8>(b);
    b = b > o ? b : o;
    o = halo_xor_lane_u64<4>(b);
    b = b > o ? b : o;
    o = halo_xor_lane_u64<2>(b);
    b = b > o ? b : o;
    o = halo_xor_lane_u64<1>(b);
    b = b > o ? b : o;
    return b;
}

/* One row in one wave, from any grid. A lane below `k` returns its slot's expert as written to
 * `ids`, which is also the row's return value so a fused caller can hand the slot straight to the
 * sort. OWN is a template bound because the keys are registers; a lane-derived trip count would
 * index them divergently. */
template <int OWN>
__device__ __forceinline__ int router_sigmoid_row(const float* logits, const float* bias, int* ids,
                                                 float* weights, long lp, long bp, long ip,
                                                 long wp, int r, int ne, int k, int norm,
                                                 float factor, int lane) {
    float p[OWN];
    unsigned long long key[OWN];
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const int e = lane + 32 * i, s = e < ne ? e : 0;
        p[i] = 1.0f / (1.0f + expf(-logits[(long)r * lp + s]));
        key[i] = e < ne ? router_rank_key(p[i] + bias[(long)s * bp], e) : 0ull;
    }
    unsigned long long below = ~0ull, mine = 0;
    for (int j = 0; j < k; ++j) {
        unsigned long long best = 0;
#pragma unroll
        for (int i = 0; i < OWN; ++i) {
            const unsigned long long c = key[i] < below ? key[i] : 0ull;
            best = c > best ? c : best;
        }
        best = router_wave_max(best);
        if (lane == j) mine = best;
        below = best;
    }
    const int e = mine ? (int)~(uint32_t)mine : 0;
    float chosen = 0;
#pragma unroll
    for (int i = 0; i < OWN; ++i) {
        const float v = __shfl(p[i], e & 31, 32);
        if ((e >> 5) == i) chosen = v;
    }
    float total = 0;
    for (int j = 0; j < k; ++j)
        total += __int_as_float(__builtin_amdgcn_readlane(__float_as_int(chosen), j));
    const float mult = norm ? factor / (total + 1e-20f) : factor;
    if (lane < k) {
        ids[(long)r * ip + lane] = e;
        weights[(long)r * wp + lane] = chosen * mult;
    }
    return e;
}

/* A wave a row, up to 32 rows a workgroup: the standalone selection's grid rule. */
constexpr int sigmoid_waves(int M) { return M <= 8 ? 8 : M <= 16 ? 16 : 32; }

}  /* namespace */
}  /* namespace halo_routersigmoid */

#endif  /* HALO_ROUTER_SIGMOID_H */
