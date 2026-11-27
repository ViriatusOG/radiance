/* r4d_situ.hip -- SiTU, Inkling 2's expert activation, as libref/ref_elementwise.cpp's ref_situ_mul
 * defines it: the gate and up halves of a fused gate_up, each capped by tanh when its cap is
 * positive, the sigmoid taken on the raw gate, one bf16 rounding of the product.
 *
 * One thread an element; the row's columns run across the workgroup. Rows may have their own
 * pitch, the last axis dense. Nothing here allocates or synchronises.
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "r4d_plugin.h"
#include "r4d_chain.h"
#include "r4d_launch.h"
#include "rad_plugin.h"

namespace r4d_situ {

constexpr int kThr = 256;

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }
__device__ __forceinline__ uint16_t tobf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);   /* NaN stays NaN */
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

struct Args {
    const uint16_t* gu; long long gu_rs;
    uint16_t* y; long long y_rs;
    int n;
    float gcap, ucap;
};

/* CH is r4dchain for the chained form (r4d_chain.h), which waits before its first read and adds
 * after its last write; no block returns early. */
template <class CH>
__global__ __launch_bounds__(kThr) void situ_kernel(Args A) {
    if (CH::on) r4d_chain_wait();
    const long long r = blockIdx.x;
    const uint16_t* g = A.gu + r * A.gu_rs;
    uint16_t* yr = A.y + r * A.y_rs;
    for (int i = (int)threadIdx.x; i < A.n; i += kThr) {
        const float gv = bf(g[i]), uv = bf(g[A.n + i]);
        const float ga = A.gcap > 0.0f ? A.gcap * tanhf(gv / A.gcap) : gv;
        const float ua = A.ucap > 0.0f ? A.ucap * tanhf(uv / A.ucap) : uv;
        yr[i] = tobf(ga * (1.0f / (1.0f + expf(-gv))) * ua);
    }
    if (CH::on) r4d_chain_done();
}

}  // namespace r4d_situ

int r4d_rad_situ_mul(const RadArgs* a, RadStream stream) {
    using namespace r4d_situ;
    if (!a || a->n_t < 2) return RAD_E_INVAL;
    const RadTensor* gu = rad_arg_in(a, 0);
    const RadTensor* y = rad_arg_in(a, 1);
    if (!gu || !y) return RAD_E_INVAL;
    if (gu->dtype != RAD_BF16 || y->dtype != RAD_BF16) return RAD_E_DTYPE;
    const long long n = rad_args_geti_or(a, "n", 0);
    const double gcap = rad_args_getf_or(a, "gate_cap", 0.0), ucap = rad_args_getf_or(a, "up_cap", 0.0);
    if (n < 1) return RAD_E_SHAPE;
    if (!(gcap >= 0.0) || !(ucap >= 0.0) || !std::isfinite(gcap) || !std::isfinite(ucap))
        return RAD_E_INVAL;
    /* The rows are y's: a launch sees the ranged `M` at its band's upper bound, so it is a ceiling
     * only (docs/OPS.md), and gate_up may hold more rows than y takes. */
    if (y->rank != 2) return RAD_E_SHAPE;
    const long long m_cap = rad_args_geti_or(a, "M", 0);
    long long M = y->shape[0];
    if (m_cap > 0 && m_cap < M) M = m_cap;
    if (M < 1) return RAD_E_SHAPE;
    if (gu->rank != 2 || gu->shape[1] < 2 * n || gu->stride[1] != 1 || gu->shape[0] < M)
        return RAD_E_SHAPE;
    if (y->rank != 2 || y->shape[1] != n || y->stride[1] != 1 || y->shape[0] < M)
        return RAD_E_SHAPE;

    Args A{};
    A.gu = (const uint16_t*)gu->data; A.gu_rs = gu->stride[0];
    A.y = (uint16_t*)y->data; A.y_rs = y->stride[0];
    A.n = (int)n;
    A.gcap = (float)gcap; A.ucap = (float)ucap;
    if (rad_args_geti_or(a, "chain", 0) != 0)
        R4D_KLAUNCH(R4D_KNAME(situ_kernel<r4dchain>), dim3((unsigned)M), dim3(kThr),
                    (hipStream_t)stream, A);
    else
        R4D_KLAUNCH(R4D_KNAME(situ_kernel<R4dUnchained>), dim3((unsigned)M), dim3(kThr),
                    (hipStream_t)stream, A);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
