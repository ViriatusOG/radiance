/* r3d_had_quant_act_i8g.hip -- had_quant_act_i8g: a 128-wide block Hadamard, then per-group int8.
 *
 * The rotated int8 form for the int8 trunk GEMMs (i8a8h): the 128-wide Hadamard shrinks the group
 * maximum, and the codes are row-major (the i8a8 GEMMs read stored order), with one scale per
 * (row, group). This is quant_act_i8g with the rotation; the unrotated form is a separate op.
 *
 * A separate op, not a parameter: it writes different numbers and a different code layout than
 * quant_act_i8 (step-ordered) or had_quant_act_i8 (step-ordered), so a swapped graph would run as
 * a different model.
 *
 * The numbers are libref's (had_quant_act_q8, q8_group): the transform is unnormalised, the scale
 * is amax / 127 of the transformed group and 1 for an all-zero group, a code is
 * clamp(v * (1 / scale), +-127) rounded to nearest even. The 1/sqrt(128) of the rotation pair is
 * not in this scale: the i8a8h weight's scale carries the /128.
 *
 * gfx1102: the FWHT's cross-lane stages are DPP, and a DPP exchange in a loop body miscompiles here
 * (a non-first wave drops out). So there is no group loop: a workgroup is one wave and one
 * (row, group), the grid is (n / 128, M), and the wave runs the whole FWHT in registers once. n is
 * any multiple of 128.
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3d_fwht.h"
#include <cmath>

namespace {

constexpr float kTopI8 = 127.0f;

__device__ __forceinline__ float ld_bf(const uint16_t* p, uint32_t i) {
    return bf16_to_f32((uint16_t)ldg_u16(p + i));
}
__device__ __forceinline__ float clamp_top(float c, float top) {
    if (c > top) c = top;
    if (c < -top) c = -top;
    return c;
}
__device__ __forceinline__ uint32_t i8_codes4(const float* v, float inv) {
    const uint32_t b0 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[0] * inv, kTopI8));
    const uint32_t b1 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[1] * inv, kTopI8));
    const uint32_t b2 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[2] * inv, kTopI8));
    const uint32_t b3 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[3] * inv, kTopI8));
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}
__device__ __forceinline__ float wave_max_f(float v) {
    v = fmaxf(v, r3d_xor_lane<16>(v));
    v = fmaxf(v, r3d_xor_lane<8>(v));
    v = fmaxf(v, r3d_xor_lane<4>(v));
    v = fmaxf(v, r3d_xor_lane<2>(v));
    v = fmaxf(v, r3d_xor_lane<1>(v));
    return v;
}

__global__ __launch_bounds__(kWave) R3D_WAVES(8) void had_quant_act_i8g_kernel(
    const uint16_t* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s, uint32_t n,
    uint32_t ngrp) {
    constexpr int HAD = 128, VPL = HAD / 32;
    const uint32_t g = blockIdx.x, row = blockIdx.y, lane = threadIdx.x & 31;
    const size_t at = (size_t)row * n + (size_t)g * HAD + lane * VPL;

    float v[VPL];
#pragma unroll
    for (int j = 0; j < VPL; ++j) v[j] = ld_bf(x + at, (uint32_t)j);
    r3d_fwht_wave<HAD>(v, lane);
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < VPL; ++j) m = fmaxf(m, fabsf(v[j]));
    m = wave_max_f(m);
    const float sc = m > 0.0f ? m * (1.0f / kTopI8) : 1.0f;      /* libref's q8_group */
    const float inv = 1.0f / sc;
    if (lane == 0) s[(size_t)row * ngrp + g] = sc;
    *(uint32_t*)(q + at) = i8_codes4(v, inv);                    /* row-major */
}

}  // namespace

extern "C" {

int r3d_rad_had_quant_act_i8g(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, 0), *q = r3d_opt(a, 1), *sc = r3d_opt(a, 2);
    if (!r3d_p(x) || !r3d_wp(q) || !r3d_wp(sc)) return R3D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || q->dtype != RAD_I8 || sc->dtype != RAD_F32)
        return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(x) || !r3d_contig(q) || !r3d_contig(sc)) return R3D_NO(RAD_E_STRIDE);

    long long M = 0, K = 0;
    if (r3d_rows_of(x, &M, &K) != 0) return R3D_NO(RAD_E_INVAL);
    if (K <= 0 || K % 4) return R3D_NO(RAD_E_SHAPE);
    const long long grp = r3d_geti_or(a, "group", 128);
    if (grp != 128 || K % 128) return R3D_NO(RAD_E_SHAPE);    // whole 128-wide groups
    const long long G = K / 128;
    if (M <= 0 || r3d_numel(q) < M * K || r3d_numel(sc) < M * G)
        return R3D_NO(RAD_E_SHAPE);

    dim3 grid((unsigned)G, (unsigned)M), blk(kWave);
    R3D_KLAUNCH(had_quant_act_i8g_kernel, grid, blk, (hipStream_t)s, (const uint16_t*)r3d_p(x),
                (uint8_t*)r3d_wp(q), (float*)r3d_wp(sc), (uint32_t)K, (uint32_t)G);
    return (int)hipGetLastError();
}

}  // extern "C"
