/* plugin/glm/glm.hip -- swiglu_clamped_had_quant_i8g on the card.
 *
 * libr4d's gated quantiser body (r4d_fused_quant.hip, gated_quant_fp8_body with ROT and I8) with
 * swiglu_clamped's expression (r4d_glm.hip, swiglu_clamped_kernel) in place of the plain SiLU: a wave
 * a scale group, four consecutive columns a lane, which is r4d_fwht_wave<128>'s mapping, so the
 * rotation stays in registers. Each element is rounded to bf16 and read back before the rotation,
 * as had_quant_act_i8g reads the stored bf16 product; the absmax, the scale and the codes are
 * r4d_q8.h's. SUB 2 is IQ8R's 256-wide group: two 128-wide halves, each rotated on its own, under one
 * scale. Built with the library's flags (-ffp-contract=off, rad_add_plugin), so the bytes are the
 * two-op chain's (tests/glm_device_test.cpp compares them).
 */
#include "glm.h"

#include "../../libr4d/r4d_common.h"
#include "../../libr4d/r4d_emit.h"
#include "../../libr4d/r4d_fwht.h"
#include "../../libr4d/r4d_q8.h"
#include "../../libr4d/r4d_args.h"
#include "../../libr4d/r4d_chain.h"

namespace {

constexpr uint32_t kWave = 32;

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }

/* CHAIN brackets the body with the chained launch's wait and add (libr4d/r4d_chain.h); every block
 * runs to the end. The plain kernel and the chained one (r4dchain*) follow the body. */
template <uint32_t SUB, bool CHAIN>
__device__ __forceinline__ void swiglu_clamped_had_quant_body(
    const uint16_t* __restrict__ gu, int8_t* __restrict__ q, float* __restrict__ s,
    uint16_t* __restrict__ y, uint32_t n, uint32_t ngrp, uint32_t gu_ld, uint32_t q_ld,
    uint32_t s_ld, uint32_t y_ld, float limit) {
    if (CHAIN) r4d_chain_wait();
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const uint16_t* __restrict__ gr = gu + (size_t)row * gu_ld;
    uint16_t* __restrict__ yr = y ? y + (size_t)row * y_ld : nullptr;
    int8_t* __restrict__ qr = q + (size_t)row * q_ld;

    for (uint32_t kb = wave; kb < ngrp; kb += blockDim.x / kWave) {
        float v[SUB][4];
        uint32_t c0[SUB];
#pragma unroll
        for (uint32_t b = 0; b < SUB; ++b) {
            c0[b] = (kb * SUB + b) * 128u + lane * 4u;
            uint16_t gb[4], ub[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                gb[j] = gr[c0[b] + j];
                ub[j] = gr[n + c0[b] + j];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float g = fminf(bf2f(gb[j]), limit);
                const float u = fminf(fmaxf(bf2f(ub[j]), -limit), limit);
                const uint16_t h = f32_to_bf16((g / (1.0f + expf(-g))) * u);
                if (yr) yr[c0[b] + j] = h;
                v[b][j] = bf2f(h);
            }
            r4d_fwht_wave<128>(v[b], (int)lane);
        }
        float m = 0.0f;
#pragma unroll
        for (uint32_t b = 0; b < SUB; ++b)
            m = fmaxf(m, fmaxf(fmaxf(fabsf(v[b][0]), fabsf(v[b][1])),
                               fmaxf(fabsf(v[b][2]), fabsf(v[b][3]))));
        m = r4d_wave_max(m);
        const float sc = m > 0.0f ? m * (1.0f / q8_top<true>()) : 1.0f;
        if (lane == 0) s[(size_t)row * s_ld + kb] = sc;
        const float inv = 1.0f / sc;
#pragma unroll
        for (uint32_t b = 0; b < SUB; ++b)
            *(uint32_t*)(qr + c0[b]) = q8_pack4<true>(v[b][0], v[b][1], v[b][2], v[b][3], inv);
    }
    if (CHAIN) r4d_chain_done();
}

template <uint32_t SUB>
__global__ __launch_bounds__(256) void swiglu_clamped_had_quant_kernel(
    const uint16_t* __restrict__ gu, int8_t* __restrict__ q, float* __restrict__ s,
    uint16_t* __restrict__ y, uint32_t n, uint32_t ngrp, uint32_t gu_ld, uint32_t q_ld,
    uint32_t s_ld, uint32_t y_ld, float limit) {
    swiglu_clamped_had_quant_body<SUB, false>(gu, q, s, y, n, ngrp, gu_ld, q_ld, s_ld, y_ld, limit);
}
template <uint32_t SUB>
__global__ __launch_bounds__(256) void r4dchain_glm_swiglu_clamped_had_quant(
    const uint16_t* __restrict__ gu, int8_t* __restrict__ q, float* __restrict__ s,
    uint16_t* __restrict__ y, uint32_t n, uint32_t ngrp, uint32_t gu_ld, uint32_t q_ld,
    uint32_t s_ld, uint32_t y_ld, float limit) {
    swiglu_clamped_had_quant_body<SUB, true>(gu, q, s, y, n, ngrp, gu_ld, q_ld, s_ld, y_ld, limit);
}

}  /* namespace */

extern "C" int rad_glm_swiglu_quant_launch(const RadArgs* a, RadStream stream) {
    RadGlmSwigluQuantCall c{};
    const int rc = rad_glm_swiglu_quant_args(a, &c);
    if (rc != RAD_OK) return rc;
    if (c.gu_ld > UINT32_MAX || c.q_ld > UINT32_MAX || c.s_ld > UINT32_MAX || c.y_ld > UINT32_MAX)
        return RAD_E_SHAPE;
    /* The codes are written a dword at a time. */
    if (((uintptr_t)c.q & 3u) || (c.q_ld & 3)) return RAD_E_STRIDE;
    const uint32_t ngrp = (uint32_t)(c.n / c.group);
    /* A wave a group, at most eight a block: the width does not change the bits (no cross-wave
     * reduction, and the rotation is wave-local). */
    const uint32_t thr = (ngrp < 8u ? ngrp : 8u) * kWave;
    /* Chained at the decode width only (RadKernelChainFn), as libr4d's quantisers are. */
    const bool chain = r4d_geti_or(a, "chain", 0) != 0 && c.M <= 64;
#define GLM_SQ(SUB_)                                                                               \
    R4D_EMIT(stream, (chain ? r4dchain_glm_swiglu_clamped_had_quant<SUB_>                          \
                            : swiglu_clamped_had_quant_kernel<SUB_>), dim3((uint32_t)c.M), dim3(thr), 0, \
             c.gu, c.q, c.s, c.y, (uint32_t)c.n, ngrp, (uint32_t)c.gu_ld, (uint32_t)c.q_ld,        \
             (uint32_t)c.s_ld, (uint32_t)c.y_ld, c.limit)
    return c.group == 256 ? GLM_SQ(2u) : GLM_SQ(1u);
#undef GLM_SQ
}
