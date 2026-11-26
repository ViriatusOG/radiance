/* lq_ggml_dequant.cpp -- llama.cpp's dequantize_row_<type> (lq_ggml_dequant.h).
 *
 * Ported from ggml/src/ggml-quants.c (llama.cpp, MIT; the same upstream as lq_ggml_port), keeping
 * each loop's arithmetic in its order so every value is the one llama.cpp computes: d is widened
 * to f32 first, then multiplied as written there.
 */
#include "lq_ggml_dequant.h"
#include "lq_ggml_port.h"

#include <cstring>

namespace lq::ggml {

float f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, bits;
    if (e == 0x1F) {
        bits = s | 0x7F800000u | (m << 13);                 /* inf / nan */
    } else if (e != 0) {
        bits = s | ((e + 112u) << 23) | (m << 13);          /* normal */
    } else if (m == 0) {
        bits = s;                                            /* zero */
    } else {                                                 /* subnormal: normalise */
        e = 113;
        while (!(m & 0x400u)) {
            m <<= 1;
            --e;
        }
        bits = s | (e << 23) | ((m & 0x3FFu) << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

namespace {

inline void scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

void dq_q8_0(const void* v, float* y, int64_t k) {
    const block_q8_0* x = (const block_q8_0*)v;
    const int64_t nb = k / QK8_0;
    for (int64_t i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        for (int j = 0; j < QK8_0; ++j) y[i * QK8_0 + j] = x[i].qs[j] * d;
    }
}

void dq_q5_K(const void* v, float* y, int64_t k) {
    const block_q5_K* x = (const block_q5_K*)v;
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const float d = f16_to_f32(x[i].d);
        const float min = f16_to_f32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void dq_q6_K(const void* v, float* y, int64_t k) {
    const block_q6_K* x = (const block_q6_K*)v;
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void dq_iq4_nl(const void* v, float* y, int64_t k) {
    const block_iq4_nl* x = (const block_iq4_nl*)v;
    const int64_t nb = k / QK4_NL;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* qs = x[i].qs;
        const float d = f16_to_f32(x[i].d);
        for (int j = 0; j < QK4_NL / 2; ++j) {
            y[j + 0] = d * kvalues_iq4nl[qs[j] & 0xf];
            y[j + QK4_NL / 2] = d * kvalues_iq4nl[qs[j] >> 4];
        }
        y += QK4_NL;
    }
}

void dq_iq2_xxs(const void* v, float* y, int64_t k) {
    const block_iq2_xxs* x = (const block_iq2_xxs*)v;
    const int64_t nb = k / QK_K;
    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*)aux32;
    for (int64_t i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            std::memcpy(aux32, x[i].qs + 4 * ib32, 2 * sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j) y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                y += 8;
            }
        }
    }
}

void dq_iq3_xxs(const void* v, float* y, int64_t k) {
    const block_iq3_xxs* x = (const block_iq3_xxs*)v;
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* scales_and_signs = qs + QK_K / 4;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            uint32_t aux32;
            std::memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof aux32);
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

void dq_iq1_s(const void* v, float* y, int64_t k) {
    const block_iq1_s* x = (const block_iq1_s*)v;
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint16_t* qh = x[i].qh;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const float dl = d * (2 * ((qh[ib] >> 12) & 7) + 1);
            const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 4; ++l) {
                const int8_t* grid = (const int8_t*)(iq1s_grid + (qs[l] | (((qh[ib] >> 3 * l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) y[j] = dl * (grid[j] + delta);
                y += 8;
            }
            qs += 4;
        }
    }
}

struct Entry { ggml_type type; DequantFn fn; };
const Entry kDequant[] = {
    { GGML_TYPE_Q8_0,    dq_q8_0 },
    { GGML_TYPE_Q5_K,    dq_q5_K },
    { GGML_TYPE_Q6_K,    dq_q6_K },
    { GGML_TYPE_IQ4_NL,  dq_iq4_nl },
    { GGML_TYPE_IQ2_XXS, dq_iq2_xxs },
    { GGML_TYPE_IQ3_XXS, dq_iq3_xxs },
    { GGML_TYPE_IQ1_S,   dq_iq1_s },
};

}  /* namespace */

DequantFn dequant_of(uint32_t type) {
    for (const Entry& e : kDequant)
        if ((uint32_t)e.type == type) return e.fn;
    return nullptr;
}

}  /* namespace lq::ggml */
