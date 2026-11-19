/* plugin/glm/ref_glm.cpp -- swiglu_clamped_had_quant_i8g on the host: the oracle rad-kbench checks the
 * device row against, and the only form a machine with no card runs.
 *
 * Each step is the device's, in the same order: the clamped SwiGLU in f32 and its bf16 round (round
 * to nearest even), the unnormalised Hadamard by ascending stride with the low index taking a + b
 * and the high a - b (libr4d/r4d_fwht.h; addition commutes and negation is exact, so the lane
 * exchange gives these bits), one absmax over the group, the scale amax / 127 (1 for an all-zero
 * group), and the codes rint(clamp(v * (1 / scale), -127, 127)). The host's expf may differ from
 * the device's in the last place, which a bf16 round usually absorbs; the device row is held to the
 * libr4d chain's bytes by tests/glm_device_test.cpp, and to this one within rad-kbench's tolerance. */
#include "glm.h"

#include <cmath>

namespace {

void fwht128(float* v) {
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; ++i)
            if ((i & h) == 0) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
}

}  /* namespace */

int rad_ref_glm_swiglu_quant(const RadArgs* a, RadStream) {
    RadGlmSwigluQuantCall c{};
    const int rc = rad_glm_swiglu_quant_args(a, &c);
    if (rc != RAD_OK) return rc;
    const float lim = c.limit;
    for (int64_t r = 0; r < c.M; ++r) {
        const uint16_t* gr = c.gu + r * c.gu_ld;
        for (int64_t g0 = 0; g0 < c.n; g0 += c.group) {
            float v[256];
            for (int64_t j = 0; j < c.group; ++j) {
                const float g = std::fmin(rad_bf16_to_f32(gr[g0 + j]), lim);
                const float u = std::fmin(std::fmax(rad_bf16_to_f32(gr[c.n + g0 + j]), -lim), lim);
                const uint16_t h = rad_f32_to_bf16((g / (1.0f + std::exp(-g))) * u);
                if (c.y) c.y[r * c.y_ld + g0 + j] = h;
                v[j] = rad_bf16_to_f32(h);
            }
            for (int64_t b = 0; b < c.group; b += 128) fwht128(v + b);
            float m = 0.0f;
            for (int64_t j = 0; j < c.group; ++j) m = std::fmax(m, std::fabs(v[j]));
            const float sc = m > 0.0f ? m * (1.0f / 127.0f) : 1.0f;
            c.s[r * c.s_ld + g0 / c.group] = sc;
            const float inv = 1.0f / sc;
            for (int64_t j = 0; j < c.group; ++j)
                c.q[r * c.q_ld + g0 + j] =
                    (int8_t)(int)std::rint(std::fmin(std::fmax(v[j] * inv, -127.0f), 127.0f));
        }
    }
    return RAD_OK;
}
