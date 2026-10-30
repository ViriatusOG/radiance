/* Derived from libr4d/r4d_glm.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_router_sigmoid_topk_f32.hip - halo_router_sigmoid_topk: sigmoid routing with a correction
// bias, one wave per row, f32.
//
// p = sigmoid(logit); the top_k largest of (p + bias) are taken, equal ranks to the lower expert, as
// a serial scan with a strict `>` takes them; then mult = norm ? route_scale / (sum p + 1e-20) :
// route_scale and a weight is p * mult. A NaN rank is never selected.
//
// As in halo_router_topk_bf16, the only architectural content is in the reductions, and both are
// wave butterflies (halo_xor_lane_u64 / readlane) whose value does not depend on the architecture.
// The selection is the same p passes of a wave maximum that libref's repeated-maximum scan makes.
//
// The row itself is halo_router_sigmoid.h, because the fused scatter form
// (router_sigmoid_topk_scatter, halo_router_scatter.hip) runs that same row function and hands its
// slots to moe_scatter's single-workgroup sort (halo_moe_sort.h). One definition, so the fused
// selection and this one rank a row identically.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_emit.h"
#include "halo_router_sigmoid.h"
#include <cmath>
#include <cstdint>

namespace {
using namespace halo_routersigmoid;

template <int OWN>
__global__ void __launch_bounds__(128)
    router_sigmoid_wave_kernel(const float* logits, const float* bias, int* ids, float* weights,
                               long lp, long bp, long ip, long wp, int m, int ne, int k, int norm,
                               float factor) {
    const int lane = threadIdx.x & 31, r = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (r >= m) return;
    (void)router_sigmoid_row<OWN>(logits, bias, ids, weights, lp, bp, ip, wp, r, ne, k, norm,
                                  factor, lane);
}

}  /* namespace */

extern "C" int halo_rad_router_sigmoid_topk(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4) return HALO_NO(RAD_E_INVAL);
    const RadTensor &lg = a->t[0], &bias = a->t[1], &ids = a->t[2], &w = a->t[3];
    const long ne = halo_geti_or(a, "n_expert", 0), k = halo_geti_or(a, "top_k", 0);
    const float factor = (float)halo_getf(a, "route_scale", 0);
    if (lg.dtype != RAD_F32 || bias.dtype != RAD_F32 || ids.dtype != RAD_I32 || w.dtype != RAD_F32)
        return HALO_NO(RAD_E_DTYPE);
    if (ne < 1 || ne > 512 || k < 1 || k > 32 || k > ne || !std::isfinite(factor) || factor <= 0 ||
        !lg.data || !bias.data || !ids.data || !w.data || lg.rank != 2 || lg.shape[0] < 1 ||
        lg.shape[0] > 65535 || lg.shape[1] != ne || bias.rank != 1 || bias.shape[0] != ne ||
        ids.rank != 2 || w.rank != 2 || ids.shape[0] != lg.shape[0] || w.shape[0] != lg.shape[0] ||
        ids.shape[1] != k || w.shape[1] != k || lg.stride[1] != 1 || ids.stride[1] != 1 ||
        w.stride[1] != 1 || lg.stride[0] < ne || ids.stride[0] < k || w.stride[0] < k ||
        bias.stride[0] < 1)
        return HALO_NO(RAD_E_SHAPE);
    const int norm = (int)(halo_geti_or(a, "norm", 0) != 0);
    const dim3 grid((unsigned)((lg.shape[0] + 3) / 4));
    if (ne <= 9 * 32)
        return HALO_EMIT(stream, router_sigmoid_wave_kernel<9>, grid, dim3(128), 0,
                         (const float*)lg.data, (const float*)bias.data, (int*)ids.data,
                         (float*)w.data, lg.stride[0], bias.stride[0], ids.stride[0], w.stride[0],
                         (int)lg.shape[0], (int)ne, (int)k, norm, factor);
    return HALO_EMIT(stream, router_sigmoid_wave_kernel<16>, grid, dim3(128), 0,
                     (const float*)lg.data, (const float*)bias.data, (int*)ids.data, (float*)w.data,
                     lg.stride[0], bias.stride[0], ids.stride[0], w.stride[0], (int)lg.shape[0],
                     (int)ne, (int)k, norm, factor);
}
