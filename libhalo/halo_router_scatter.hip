/* Derived from libr4d/r4d_moe.hip and libr4d/r4d_glm.hip by the libhalo rename transform
 * (2026-10-26); see libhalo/README.md. */
// halo_router_scatter.hip -- the two fused router rows: the selection and moe_scatter's stable sort
// in one workgroup.
//
//   router_topk_scatter          softmax + exact top_k over bf16 logits, then the sort
//   router_sigmoid_topk_scatter  sigmoid routing + correction bias over f32 logits, then the sort
//                                (and, where the layer has one, the ungated shared arm's gate of 1)
//
// At decode each half is a single workgroup moving about a kilobyte, so each launch is mostly
// dispatch floor and barriers, once per MoE layer per step. Neither op is WMMA code: the selection
// is the wave-local row function of halo_router_topk.h / halo_router_sigmoid.h and the sort is
// halo_moe_sort.h's, so the port is plumbing only.
//
// Byte-identical to the unfused pair (halo_selftest asserts it on all five outputs): the selection
// and placement are the same device functions, wave-local and stable. Slots reach the sort through
// LDS, not `expert_ids`: each row's lanes write their slot's expert (relative to the slice, or -1)
// into the sort's slot list, so the one barrier between the halves orders LDS only. There is no
// barrier before the rows (each thread zeroes the histogram on its way in), since a top barrier
// would wait for the workgroup's last wave to launch.
//
// The one difference from the pair is a row with fewer live experts than top_k (NaN logits): the
// standalone sort reads its unselected slots from whatever `expert_ids` held, this one records them
// as dropped. A sigmoid row with fewer finite ranks than top_k writes expert 0 into its slot, which
// is what the standalone selection writes, so that family has no such difference.
//
// The rows are a loop, not an early return, because every thread must reach the barrier. T =
// M * top_k must fit kScatterSmall, checked here and by the architecture before declaring the op.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_emit.h"
#include "halo_moe_sort.h"
#include "halo_router_topk.h"
#include "halo_router_sigmoid.h"
#include <cmath>
#include <cstdint>

#define HALO_RS_CHK() do { hipError_t e_ = hipGetLastError();                      \
                           if (e_ != hipSuccess) return HALO_NO(RAD_E_DEVICE); } while (0)

namespace {

using halo_moesort::kMaxExpert;
using halo_moesort::kScatterSmall;
using halo_moesort::Protect;
using halo_moesort::protect_of;
using halo_moesort::scatter_small_place;
using halo_moesort::ScatterLds;
using halo_moesort::sort_place;

constexpr uint32_t kWave = 32;

/* Two different expert counts: the selection runs over the whole (replicated) router, `n_expert`,
 * while the sort runs over this rank's slice, `n_local`, dropping the rest (512 against 254 at two
 * ranks here). Passing one count to both would let the sort index its histogram out of range. */
template <uint32_t MAXOWN, uint32_t NW>
__global__ __launch_bounds__(NW * kWave) void router_topk_scatter_kernel(
    const uint16_t* __restrict__ lg, int32_t* __restrict__ ids, uint16_t* __restrict__ w,
    int32_t* __restrict__ sorted, int32_t* __restrict__ off, int32_t* __restrict__ cnt,
    uint32_t M, uint32_t n_expert, uint32_t n_local, uint32_t top_k, uint32_t lg_ld, int norm,
    int32_t base, Protect prot) {
    __shared__ ScatterLds<NW> sl;
    const uint32_t wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
    for (uint32_t e = threadIdx.x; e < n_local; e += blockDim.x) sl.cnt[e] = 0;
    const halo_routertopk::SlotSink snk{sl.e, base, n_local, prot};
    for (uint32_t m = wave; m < M; m += NW)
        halo_routertopk::router_topk_row<MAXOWN>(lg, ids, w, m, n_expert, top_k, lg_ld, norm, lane,
                                                snk);
    lds_barrier();
    scatter_small_place(sorted, off, cnt, M * top_k, n_local, sl);
}

/* router_sigmoid_topk, then moe_scatter over its ids, then the `fill` of an ungated shared arm's
 * gate with 1, in one workgroup. INV also writes the inverse, as moe_scatter_inv does. */
template <int OWN, uint32_t NW, bool INV>
__global__ __launch_bounds__(NW * kWave) void
    router_sigmoid_scatter_kernel(const float* logits, const float* bias, int* ids, float* weights,
                                  int32_t* sorted, int32_t* off, int32_t* cnt, uint16_t* gate,
                                  int32_t* inv, long lp, long bp, long ip, long wp, long gp,
                                  int m, int ne, int n_local, int k, int norm, float factor,
                                  int base, Protect prot) {
    __shared__ ScatterLds<NW> sl;
    const uint32_t lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (int e = (int)threadIdx.x; e < n_local; e += (int)(NW * kWave)) sl.cnt[e] = 0;
    /* No loop-carried barrier: a row writes only its own LDS slots and the gate, so the single
     * barrier before the sort is enough. */
    for (int r = (int)wave; r < m; r += (int)NW) {
        const int e = halo_routersigmoid::router_sigmoid_row<OWN>(
                          logits, bias, ids, weights, lp, bp, ip, wp, r, ne, k, norm, factor,
                          (int)lane) - base;
        if (lane < (uint32_t)k)
            sl.e[r * k + (int)lane] =
                e >= 0 && e < n_local ? sort_place(e, (uint32_t)n_local, prot) : -1;
        if (gate && lane == 0) gate[(long)r * gp] = f32_to_bf16(1.0f);
    }
    lds_barrier();
    scatter_small_place<NW, INV>(sorted, off, cnt, (uint32_t)(m * k), (uint32_t)n_local, sl, inv);
}

}  /* namespace */

/* The pair in one launch. Operands are the two ops' own, selection first: logits -> expert_ids,
 * expert_w, sorted_tok, expert_offset, expert_count. The checks are the two shims' checks.
 *
 * It refuses, rather than falls back, when the sort would not fit one workgroup, so the op's cost
 * does not hinge on a hidden bound; the architecture checks the same before declaring it. */
extern "C" int halo_rad_router_topk_scatter(const RadArgs* a, RadStream s) {
    const RadTensor* lg     = halo_opt(a, 0);
    const RadTensor* ids    = halo_opt(a, 1);
    const RadTensor* w      = halo_opt(a, 2);
    const RadTensor* sorted = halo_opt(a, 3);
    const RadTensor* off    = halo_opt(a, 4);
    const RadTensor* cnt    = halo_opt(a, 5);
    if (!lg || !ids || !w || !sorted || !off || !cnt) return HALO_NO(RAD_E_INVAL);
    if (!lg->data || !ids->data || !w->data || !sorted->data || !off->data || !cnt->data)
        return HALO_NO(RAD_E_INVAL);

    const long long n_expert = halo_geti_or(a, "n_expert", 0);
    /* Absent means no expert parallelism: the slice is the whole router. */
    const long long n_local  = halo_geti_or(a, "n_local", n_expert);
    const long long top_k    = halo_geti_or(a, "top_k", 0);
    const long long norm     = halo_geti_or(a, "norm", 0);
    const long long base     = halo_geti_or(a, "expert_base", 0);
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert || top_k <= 0 || top_k > n_expert)
        return HALO_NO(RAD_E_SHAPE);
    if (top_k > (long long)kWave) return HALO_NO(RAD_E_SHAPE);
    if (n_local <= 0 || n_local > n_expert) return HALO_NO(RAD_E_SHAPE);
    if (base < 0 || base + n_local > n_expert) return HALO_NO(RAD_E_SHAPE);
    Protect prot;
    const int prc = protect_of(a, n_local, &prot);
    if (prc != RAD_OK) return prc;
    if (lg->dtype != RAD_BF16 || w->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    if (ids->dtype != RAD_I32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32) return HALO_NO(RAD_E_DTYPE);

    long long M = 0;
    const long long lg_ld = halo_rows_pitch(lg, n_expert, &M);
    if (lg_ld < 0 || M <= 0) return HALO_NO(RAD_E_STRIDE);
    if (!halo_contig(ids) || !halo_contig(w) || !halo_contig(sorted) || !halo_contig(off) ||
        !halo_contig(cnt)) return HALO_NO(RAD_E_STRIDE);

    const long long T = M * top_k;
    if (T > (long long)kScatterSmall) return HALO_NO(RAD_E_SHAPE);
    if (halo_numel(ids) < T || halo_numel(w) < T) return HALO_NO(RAD_E_SHAPE);
    if (halo_numel(sorted) < T) return HALO_NO(RAD_E_SHAPE);
    if (halo_numel(off) < n_local + 1 || halo_numel(cnt) < n_local) return HALO_NO(RAD_E_SHAPE);

    /* One wave per row, rounded to a power of two between 8 and 32 waves (halo_router_topk.h). The
     * narrowest instantiation, as in the standalone selection. */
    const uint32_t nw = halo_routertopk::scatter_waves((uint32_t)M);
#define HALO_RTS_L(OWN, NWV)                                                                     \
    HALO_EMIT(s, (router_topk_scatter_kernel<OWN, NWV>), dim3(1), dim3((NWV) * kWave), 0,        \
              (const uint16_t*)lg->data, (int32_t*)ids->data, (uint16_t*)w->data,                \
              (int32_t*)sorted->data, (int32_t*)off->data, (int32_t*)cnt->data, (uint32_t)M,     \
              (uint32_t)n_expert, (uint32_t)n_local, (uint32_t)top_k, (uint32_t)lg_ld, (int)norm, \
              (int32_t)base, prot)
    int rc;
    if (n_expert <= (long long)(8 * kWave)) {
        if      (nw == 8)  rc = HALO_RTS_L(8, 8);
        else if (nw == 16) rc = HALO_RTS_L(8, 16);
        else               rc = HALO_RTS_L(8, 32);
    } else {
        if      (nw == 8)  rc = HALO_RTS_L(16, 8);
        else if (nw == 16) rc = HALO_RTS_L(16, 16);
        else               rc = HALO_RTS_L(16, 32);
    }
#undef HALO_RTS_L
    if (rc != RAD_OK) return rc;
    HALO_RS_CHK();
    return RAD_OK;
}

/* Operands: router_sigmoid_topk's four, moe_scatter's three outputs, then the optional gate and the
 * optional inverse. The sort is the single-workgroup form, so M * top_k must fit it; a larger pass
 * is refused rather than served at another cost. */
extern "C" int halo_rad_router_sigmoid_topk_scatter(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 7) return HALO_NO(RAD_E_INVAL);
    const RadTensor &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &w = a->t[3];
    const RadTensor *sorted = &a->t[4], *off = &a->t[5], *cnt = &a->t[6];
    const RadTensor* gate = a->n_t > 7 ? halo_opt(a, 7) : nullptr;
    const RadTensor* inv  = a->n_t > 8 ? halo_opt(a, 8) : nullptr;
    const long ne = halo_geti_or(a, "n_expert", 0), k = halo_geti_or(a, "top_k", 0);
    const long nl = halo_geti_or(a, "n_local", ne), base = halo_geti_or(a, "expert_base", 0);
    const float factor = (float)halo_getf(a, "route_scale", 0);
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || ids.dtype != RAD_I32 ||
        w.dtype != RAD_F32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32 || (gate && gate->dtype != RAD_BF16) ||
        (inv && inv->dtype != RAD_I32))
        return HALO_NO(RAD_E_DTYPE);
    if (ne < 1 || ne > 512 || k < 1 || k > 32 || k > ne || !std::isfinite(factor) || factor <= 0 ||
        !lg.data || !bias.data || !ids.data || !w.data || !sorted->data || !off->data ||
        !cnt->data || lg.rank != 2 || lg.shape[0] < 1 || lg.shape[1] != ne || bias.rank != 1 ||
        bias.shape[0] != ne || ids.rank != 2 || w.rank != 2 || ids.shape[0] != lg.shape[0] ||
        w.shape[0] != lg.shape[0] || ids.shape[1] != k || w.shape[1] != k || lg.stride[1] != 1 ||
        ids.stride[1] != 1 || w.stride[1] != 1 || lg.stride[0] < ne || ids.stride[0] < k ||
        w.stride[0] < k || bias.stride[0] < 1)
        return HALO_NO(RAD_E_SHAPE);
    const long m = lg.shape[0], T = m * k;
    if (T > (long)kScatterSmall || nl < 1 || nl > ne || base < 0 || base + nl > ne ||
        !halo_contig(sorted) || !halo_contig(off) || !halo_contig(cnt) ||
        halo_numel(sorted) < T || halo_numel(off) < nl + 1 || halo_numel(cnt) < nl ||
        (inv && (!halo_contig(inv) || halo_numel(inv) < T)))
        return HALO_NO(RAD_E_SHAPE);
    long long grows = 0;
    const long long gp = gate ? halo_rows_pitch(gate, 1, &grows) : 0;
    if (gate && (gp < 1 || grows < m)) return HALO_NO(RAD_E_SHAPE);
    Protect prot;
    const int prc = protect_of(a, nl, &prot);
    if (prc != RAD_OK) return prc;
    const int norm = (int)(halo_geti_or(a, "norm", 0) != 0);

    const int nw = halo_routersigmoid::sigmoid_waves((int)m);
#define HALO_RSS_I(OWN, NW, INV)                                                                  \
    do {                                                                                          \
        rc = HALO_EMIT(s, (router_sigmoid_scatter_kernel<OWN, NW, INV>), dim3(1),                 \
                       dim3((NW) * kWave), 0, (const float*)lg.data,                              \
                       (const float*)bias.data, (int*)ids.data, (float*)w.data,                    \
                       (int32_t*)sorted->data, (int32_t*)off->data, (int32_t*)cnt->data,           \
                       gate ? (uint16_t*)gate->data : (uint16_t*)nullptr,                          \
                       inv ? (int32_t*)inv->data : (int32_t*)nullptr, lg.stride[0],                \
                       bias.stride[0], ids.stride[0], w.stride[0], (long)gp, (int)m, (int)ne,      \
                       (int)nl, (int)k, norm, factor, (int)base, prot);                            \
        if (rc != RAD_OK) return rc;                                                              \
    } while (0)
#define HALO_RSS(OWN, NW)                                                                         \
    do {                                                                                          \
        if (inv) HALO_RSS_I(OWN, NW, true);                                                       \
        HALO_RSS_I(OWN, NW, false);                                                               \
    } while (0)
    int rc = RAD_OK;
    if (ne <= 9 * 32) {
        if (nw == 8)       HALO_RSS(9, 8u);
        else if (nw == 16) HALO_RSS(9, 16u);
        else               HALO_RSS(9, 32u);
    } else {
        if (nw == 8)       HALO_RSS(16, 8u);
        else if (nw == 16) HALO_RSS(16, 16u);
        else               HALO_RSS(16, 32u);
    }
#undef HALO_RSS
#undef HALO_RSS_I
    if (rc != RAD_OK) return rc;
    HALO_RS_CHK();
    return RAD_OK;
}
