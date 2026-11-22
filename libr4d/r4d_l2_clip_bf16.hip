// r4d_l2_clip_bf16.hip -- Inkling 2's per-group L2 clip with an optional per-column scale and a
// position tau, the QK and relative-bias clip of its attention.
//
//   v[r][i]  = x[r][i] * w[i]                       (w optional, one per column)
//   f[r][g]  = min(1, threshold / (||v[r][g]|| + 1e-6))   over the group g of i
//   tau[r]   = 1 + alpha * ln(max(1, (positions[r] + 1) / floor))   (1 with no positions)
//   y[r][i]  = v[r][i] * f[r][g] * tau[r]
//
// libref/ref_elementwise.cpp is the oracle and docs/OPS.md the contract.
//
// One thread per (row, group). The group's norm is taken before any of its outputs are stored, and
// each element is read before it is written, so y may alias x; a group belongs to one thread, so
// no two threads write the same element. Pitched x and y are read through their row pitch.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"

namespace {

constexpr int kL2Threads = 256;

__global__ __launch_bounds__(kL2Threads)
void l2_clip_kernel(const uint16_t* __restrict__ x, long long xld, const uint16_t* __restrict__ w,
                    const int32_t* __restrict__ pos, uint16_t* __restrict__ y, long long yld,
                    long long n, long long g, long long G, long long rows, double thr, double flr,
                    double alpha) {
    const long long idx = (long long)blockIdx.x * kL2Threads + threadIdx.x;
    if (idx >= rows * G) return;
    const long long r = idx / G, gi = idx % G;
    float tau = 1.0f;
    if (pos) {
        const double p = (double)pos[r];
        tau = 1.0f + (float)alpha * (float)log(fmax(1.0, (p + 1.0) / flr));
    }
    const uint16_t* xr = x + r * xld + gi * g;
    uint16_t* yr = y + r * yld + gi * g;
    const uint16_t* wg = w ? w + gi * g : nullptr;

    double ss = 0.0;
    for (long long j = 0; j < g; ++j) {
        const double v = (double)bf16_to_f32(xr[j]) * (wg ? (double)bf16_to_f32(wg[j]) : 1.0);
        ss += v * v;
    }
    const float f = (float)fmin(1.0, thr / (sqrt(ss) + 1e-6));
    for (long long j = 0; j < g; ++j) {
        const float v = bf16_to_f32(xr[j]) * (wg ? bf16_to_f32(wg[j]) : 1.0f);
        yr[j] = f32_to_bf16(v * f * tau);
    }
}

}  // namespace

/* Operands: x [rows, n], w [n] (optional), positions [rows] i32 (optional), y [rows, n].
 * Params: M, n, group, threshold, dtype (bf16 only), floor (1), alpha (0). */
extern "C" int r4d_rad_l2_clip(const RadArgs* a, RadStream s) {
    const RadTensor *x = r4d_opt(a, 0), *w = r4d_opt(a, 1), *pos = r4d_opt(a, 2), *y = r4d_opt(a, 3);
    if (!x || !y) return R4D_NO(RAD_E_INVAL);
    const long long n = r4d_geti_or(a, "n", 0), g = r4d_geti_or(a, "group", n);
    if (n <= 0 || g <= 0 || n % g != 0) return R4D_NO(RAD_E_SHAPE);
    const double thr = r4d_getf(a, "threshold", 0.0), flr = r4d_getf(a, "floor", 1.0);
    const double alpha = r4d_getf(a, "alpha", 0.0);
    if (!(thr > 0.0) || !(flr > 0.0)) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || y->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);

    long long rows = 0;
    const long long xld = r4d_rows_pitch(x, n, &rows);
    if (xld < 0) return R4D_NO(RAD_E_STRIDE);
    long long yrows = 0;
    const long long yld = r4d_rows_pitch(y, n, &yrows);
    if (yld < 0) return R4D_NO(RAD_E_STRIDE);
    if (rows <= 0 || yrows != rows) return R4D_NO(RAD_E_SHAPE);
    if (w && (w->dtype != RAD_BF16 || r4d_numel(w) < n || !r4d_contig(w))) return R4D_NO(RAD_E_SHAPE);
    if (pos && (pos->dtype != RAD_I32 || r4d_numel(pos) < rows || !r4d_contig(pos)))
        return R4D_NO(RAD_E_SHAPE);

    const long long G = n / g;
    const uint32_t blocks = (uint32_t)((rows * G + kL2Threads - 1) / kL2Threads);
    hipLaunchKernelGGL(l2_clip_kernel, dim3(blocks), dim3(kL2Threads), 0, (hipStream_t)s,
                       (const uint16_t*)r4d_addr(x), xld,
                       w ? (const uint16_t*)r4d_addr(w) : nullptr,
                       pos ? (const int32_t*)r4d_addr(pos) : nullptr,
                       (uint16_t*)r4d_addr(y), yld, n, g, G, rows, thr, flr, alpha);
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}
