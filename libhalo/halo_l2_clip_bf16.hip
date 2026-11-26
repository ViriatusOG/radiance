/* Derived from libr4d/r4d_l2_clip_bf16.hip by the libhalo rename transform (2026-11-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_l2_clip_bf16.hip -- Inkling 2's per-group L2 clip with an optional per-column scale and a
// position tau, the QK and relative-bias clip of its attention.
//
//   v[r][i]  = x[r][i] * w[i]                       (w optional, one per column)
//   f[r][g]  = min(1, threshold / (||v[r][g]|| + 1e-6))   over the group g of i
//   tau[r]   = 1 + alpha * ln(max(1, (positions[r] + 1) / floor))   (1 with no positions)
//   y[r][i]  = v[r][i] * f[r][g] * tau[r]
//
// libref/ref_elementwise.cpp is the oracle and docs/OPS.md the contract.
//
// Wave-per-(row, group): 32 lanes cooperate on each group in f32, caching up to 256 elements in
// registers (single global read, DPP wave sum) before storing y; safe for in-place y == x.
#include "halo.h"
#include "halo_args.h"
#include "halo_chain.h"
#include "halo_common.h"
#include "halo_emit.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kL2Wave = 32;

__device__ __forceinline__ void l2_clip_wave_body(
    const uint16_t* __restrict__ x, long long xld, const uint16_t* __restrict__ w,
    const int32_t* __restrict__ pos, uint16_t* __restrict__ y, long long yld,
    long long g, long long rows, float thr, float flr, float alpha) {
    const long long gi = (long long)blockIdx.x;
    const int lane = (int)threadIdx.x;
    const uint16_t* wg = w ? w + gi * g : nullptr;

    for (long long r = (long long)blockIdx.y; r < rows; r += (long long)gridDim.y) {
        float tau = 1.0f;
        if (pos) {
            const float p = (float)pos[r];
            tau = 1.0f + alpha * logf(fmaxf(1.0f, (p + 1.0f) / flr));
        }
        const uint16_t* xr = x + r * xld + gi * g;
        uint16_t* yr = y + r * yld + gi * g;

        if (g <= 256) {
            float v[8];
            float ss = 0.0f;
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                const long long j = lane + (long long)p * kL2Wave;
                if (j < g) {
                    const float xw = bf16_to_f32(xr[j]) * (wg ? bf16_to_f32(wg[j]) : 1.0f);
                    v[p] = xw;
                    ss += xw * xw;
                }
            }
            ss = halo_wave_sum(ss);
            const float f = fminf(1.0f, thr / (sqrtf(ss) + 1e-6f));
            const float s = f * tau;
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                const long long j = lane + (long long)p * kL2Wave;
                if (j < g) yr[j] = f32_to_bf16(v[p] * s);
            }
        } else {
            float ss = 0.0f;
            for (long long j = lane; j < g; j += kL2Wave) {
                const float xw = bf16_to_f32(xr[j]) * (wg ? bf16_to_f32(wg[j]) : 1.0f);
                ss += xw * xw;
            }
            ss = halo_wave_sum(ss);
            const float f = fminf(1.0f, thr / (sqrtf(ss) + 1e-6f));
            const float s = f * tau;
            for (long long j = lane; j < g; j += kL2Wave) {
                const float xw = bf16_to_f32(xr[j]) * (wg ? bf16_to_f32(wg[j]) : 1.0f);
                yr[j] = f32_to_bf16(xw * s);
            }
        }
    }
}

template <class CH>
__global__ __launch_bounds__(kL2Wave) void l2_clip_wave_kernel(
    const uint16_t* __restrict__ x, long long xld, const uint16_t* __restrict__ w,
    const int32_t* __restrict__ pos, uint16_t* __restrict__ y, long long yld,
    long long g, long long rows, float thr, float flr, float alpha) {
    if (CH::on) halo_chain_wait();
    l2_clip_wave_body(x, xld, w, pos, y, yld, g, rows, thr, flr, alpha);
    if (CH::on) halo_chain_done();
}

}  // namespace

/* Operands: x [rows, n], w [n] (optional), positions [rows] i32 (optional), y [rows, n].
 * Params: M, n, group, threshold, dtype (bf16 only), floor (1), alpha (0). */
extern "C" int halo_rad_l2_clip(const RadArgs* a, RadStream s) {
    const RadTensor *x = halo_opt(a, 0), *w = halo_opt(a, 1), *pos = halo_opt(a, 2), *y = halo_opt(a, 3);
    if (!x || !y) return HALO_NO(RAD_E_INVAL);
    const long long n = halo_geti_or(a, "n", 0), g = halo_geti_or(a, "group", n);
    if (n <= 0 || g <= 0 || n % g != 0) return HALO_NO(RAD_E_SHAPE);
    const double thr = halo_getf(a, "threshold", 0.0), flr = halo_getf(a, "floor", 1.0);
    const double alpha = halo_getf(a, "alpha", 0.0);
    if (!(thr > 0.0) || !(flr > 0.0)) return HALO_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || y->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);

    long long rows = 0;
    const long long xld = halo_rows_pitch(x, n, &rows);
    if (xld < 0) return HALO_NO(RAD_E_STRIDE);
    long long yrows = 0;
    const long long yld = halo_rows_pitch(y, n, &yrows);
    if (yld < 0) return HALO_NO(RAD_E_STRIDE);
    if (rows <= 0 || yrows != rows) return HALO_NO(RAD_E_SHAPE);
    if (w && (w->dtype != RAD_BF16 || halo_numel(w) < n || !halo_contig(w))) return HALO_NO(RAD_E_SHAPE);
    if (pos && (pos->dtype != RAD_I32 || halo_numel(pos) < rows || !halo_contig(pos)))
        return HALO_NO(RAD_E_SHAPE);

    const long long G = n / g;
    const unsigned gy = (unsigned)(rows < 65535LL ? rows : 65535LL);
    const bool chain = halo_geti_or(a, "chain", 0) != 0;
    const int rc = HALO_EMIT(s, (chain ? l2_clip_wave_kernel<r4dchain> : l2_clip_wave_kernel<HaloUnchained>),
                            dim3((unsigned)G, gy), dim3(kL2Wave), 0,
                            (const uint16_t*)halo_addr(x), xld,
                            w ? (const uint16_t*)halo_addr(w) : nullptr,
                            pos ? (const int32_t*)halo_addr(pos) : nullptr,
                            (uint16_t*)halo_addr(y), yld, g, rows,
                            (float)thr, (float)flr, (float)alpha);
    if (rc != RAD_OK) return rc;
    return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}
