/* Derived from libr4d/r4d_moe.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_router_topk_bf16.hip - halo_router_topk: softmax over a row of expert logits, then the exact
// top_k of it, one wave a row.
//
// The port is the rename plus one architectural note: every reduction here is a wave butterfly on
// halo_xor_lane_u32 (halo_common.h), which is the same value on gfx1151 as on gfx1201 -- an unsigned
// minimum of a packed (key, ~id) word, so the combine order does not matter and the answer is the
// reference's, bit for bit. There is no WMMA leg and no fragment order in this op.
//
// Nothing here is a sort. Round r of the selection takes the largest key below round r-1's, which is
// exactly libref's repeated-maximum scan with its ties-to-the-lower-id rule, and parks the winner on
// lane r (top_k <= the wave, refused above). The keys are the logits' order-preserving bits over
// ~expert, so equal logits order to the lower expert id.
//
// The row itself is halo_router_topk.h, because the fused scatter forms (router_topk_scatter,
// router_gemm_topk_scatter -- halo_router_scatter.hip, halo_router_gemm_scatter.hip) run that same
// row function and hand their slots to moe_scatter's single-workgroup sort (halo_moe_sort.h). One
// definition, so a fused selection and this one rank a row identically; halo_selftest's
// router_topk_scatter case is what asserts it.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_emit.h"
#include "halo_router_topk.h"
#include <cstdint>

namespace {
using namespace halo_routertopk;

template <uint32_t NW, uint32_t MAXOWN>
__global__ __launch_bounds__(NW * kWave) void router_topk_kernel(
    const uint16_t* __restrict__ lg, int32_t* __restrict__ ids, uint16_t* __restrict__ w,
    uint32_t M, uint32_t n_expert, uint32_t top_k, uint32_t lg_ld, int norm) {
    const uint32_t wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
    const uint32_t m = blockIdx.x * NW + wave;
    if (m >= M) return;
    router_topk_row<MAXOWN>(lg, ids, w, m, n_expert, top_k, lg_ld, norm, lane,
                            SlotSink{nullptr, 0, 0, Protect{}});
}

}  /* namespace */

extern "C" int halo_rad_router_topk(const RadArgs* a, RadStream s) {
    const RadTensor* lg  = halo_opt(a, 0);
    const RadTensor* ids = halo_opt(a, 1);
    const RadTensor* w   = halo_opt(a, 2);
    if (!lg || !ids || !w || !lg->data || !ids->data || !w->data) return HALO_NO(RAD_E_INVAL);
    const long long n_expert = halo_geti_or(a, "n_expert", 0);
    const long long top_k    = halo_geti_or(a, "top_k", 0);
    const long long norm     = halo_geti_or(a, "norm", 0);
    if (n_expert <= 0 || n_expert > (long long)halo_routertopk::kMaxExpert || top_k <= 0 ||
        top_k > n_expert)
        return HALO_NO(RAD_E_SHAPE);
    /* The key packs `~e` into sixteen bits, which the kMaxExpert check above guarantees. top_k is
     * bounded by the wave, since round r parks its winner on lane r. */
    if (top_k > (long long)halo_routertopk::kWave) return HALO_NO(RAD_E_SHAPE);
    if (lg->dtype != RAD_BF16 || w->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    if (ids->dtype != RAD_I32) return HALO_NO(RAD_E_DTYPE);

    long long M = 0;
    const long long lg_ld = halo_rows_pitch(lg, n_expert, &M);
    if (lg_ld < 0 || M <= 0) return HALO_NO(RAD_E_STRIDE);
    if (!halo_contig(ids) || !halo_contig(w)) return HALO_NO(RAD_E_STRIDE);
    if (halo_numel(ids) < M * top_k || halo_numel(w) < M * top_k) return HALO_NO(RAD_E_SHAPE);

    constexpr uint32_t NW = 4;
    const dim3 grid((unsigned)((M + NW - 1) / NW)), blk(NW * halo_routertopk::kWave);
    /* The narrowest instantiation covering n_expert; the wider costs eight registers per lane. */
    if (n_expert <= (long long)(8 * halo_routertopk::kWave))
        hipLaunchKernelGGL((router_topk_kernel<NW, 8>), grid, blk, 0, (hipStream_t)s,
                           (const uint16_t*)lg->data, (int32_t*)ids->data, (uint16_t*)w->data,
                           (uint32_t)M, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)lg_ld,
                           (int)norm);
    else
        hipLaunchKernelGGL((router_topk_kernel<NW, 16>), grid, blk, 0, (hipStream_t)s,
                           (const uint16_t*)lg->data, (int32_t*)ids->data, (uint16_t*)w->data,
                           (uint32_t)M, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)lg_ld,
                           (int)norm);
    return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}
