/* Derived from libr4d/r4d_moe.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_scale_rows_bf16.hip -- the shared expert's one-scalar row gate, gfx1151.
//
// y[m, i] = x[m, i] * act(s[m]); with the optional `ad`, the fused gate-and-add form
// y = bf16(x * act(s)) + ad. No fragment, no LDS: one workgroup per row, the scalar read once.
// Extracted from libr4d's r4d_moe.hip, which other lanes own.
#include "halo.h"
#include "halo_args.h"
#include "halo_emit.h"
#include "halo_common.h"

#include <cstdint>
#include <cstring>

namespace {

/* This unit's own error check: libr4d's R4D_MOE_CHK returns a raw negative hipError_t, the halo
 * rows answer RAD_E_* codes. */
#define HALO_SR_CHK() do { hipError_t e_ = hipGetLastError();                    \
                           if (e_ != hipSuccess) return HALO_NO(RAD_E_DEVICE); } while (0)

/* ================================================================== scale_rows
 *
 * y[m, i] = x[m, i] * act(s[m]). One workgroup per row, grid-strided along it, with the scalar read
 * once. Used only for the shared expert's gate; docs/OPS.md explains why `mul` cannot serve it. */
/* With `ad`, the fused gate-and-add form `y = bf16(x * act(s)) + ad`. The middle narrow matches the
 * unfused pair, which stores the product as bf16 before adding; an f32 product would change the
 * last bit of every channel. */
template <uint32_t NTHR>
__global__ __launch_bounds__(NTHR) void scale_rows_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ sc,
    const uint16_t* __restrict__ ad, uint16_t* __restrict__ y,
    uint32_t M, uint32_t n, uint32_t x_ld, uint32_t s_ld, uint32_t a_ld, uint32_t y_ld, int sig) {
    const uint32_t m = blockIdx.x;
    if (m >= M) return;
    const float v0 = bf16_to_f32(sc[(size_t)m * s_ld]);
    const float v  = sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    for (uint32_t i = threadIdx.x; i < n; i += NTHR) {
        const uint16_t p = f32_to_bf16(bf16_to_f32(x[(size_t)m * x_ld + i]) * v);
        y[(size_t)m * y_ld + i] =
            ad ? f32_to_bf16(bf16_to_f32(p) + bf16_to_f32(ad[(size_t)m * a_ld + i])) : p;
    }
}


}  /* namespace */

extern "C" int halo_rad_scale_rows(const RadArgs* a, RadStream s) {
    const RadTensor* x  = halo_opt(a, 0);
    const RadTensor* sc = halo_opt(a, 1);
    const RadTensor* ad = halo_opt(a, 2);          /* optional: the fused gate-and-add form */
    const RadTensor* y  = halo_opt(a, 3);
    if (!x || !sc || !y || !x->data || !sc->data || !y->data) return HALO_NO(RAD_E_INVAL);
    if (ad && !ad->data) ad = nullptr;
    if (x->dtype != RAD_BF16 || sc->dtype != RAD_BF16 || y->dtype != RAD_BF16)
        return HALO_NO(RAD_E_DTYPE);
    if (ad && ad->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    const long long n = halo_geti_or(a, "n", 0);
    if (n <= 0) return HALO_NO(RAD_E_SHAPE);
    /* Read as a name, not via halo_act_code (whose vocabulary is silu/sigmoid, not sigmoid/none), so
     * "none" is distinguishable from a typo. An unknown name is refused. */
    const char* act = halo_gets(a, "act");
    const int sig = (act && !std::strcmp(act, "sigmoid")) ? 1 : 0;
    if (act && std::strcmp(act, "sigmoid") && std::strcmp(act, "none")) return HALO_NO(RAD_E_INVAL);

    long long M = 0;
    const long long y_ld = halo_rows_pitch(y, n, &M);
    if (y_ld < 0 || M <= 0) return HALO_NO(RAD_E_STRIDE);
    long long Mx = 0;
    const long long x_ld = halo_rows_pitch(x, n, &Mx);
    if (x_ld < 0 || Mx < M) return HALO_NO(RAD_E_STRIDE);
    if (halo_numel(sc) < M) return HALO_NO(RAD_E_SHAPE);
    long long a_ld = 0;
    if (ad) {
        long long Ma = 0;
        a_ld = halo_rows_pitch(ad, n, &Ma);
        if (a_ld < 0 || Ma < M) return HALO_NO(RAD_E_STRIDE);
    }
    /* One scalar per row, as [M] or [M, 1]; the row stride is dim 0's in both. */
    const long long s_ld = halo_stride(sc, 0) > 0 ? halo_stride(sc, 0) : 1;

    hipLaunchKernelGGL((scale_rows_kernel<256>), dim3((unsigned)M), dim3(256), 0, (hipStream_t)s,
                       (const uint16_t*)x->data, (const uint16_t*)sc->data,
                       ad ? (const uint16_t*)ad->data : nullptr, (uint16_t*)y->data,
                       (uint32_t)M, (uint32_t)n, (uint32_t)x_ld, (uint32_t)(s_ld ? s_ld : 1),
                       (uint32_t)a_ld, (uint32_t)y_ld, sig);
    HALO_SR_CHK();
    return RAD_OK;
}
