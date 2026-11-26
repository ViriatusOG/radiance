/* exl3i_twopass_ref.c -- the two-pass prefill planes' C reference (exl3i_twopass_ref.h). Build with -ffp-contract=off. */
#include "exl3i_twopass_ref.h"
#include "exl3i_a8.h"
#include "exl3_decode.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


/* tile (tn, tk) of the container's record table: its values in ring order */
static void tile_values(const Exl3iView* v, int64_t tn, int64_t tk, uint16_t val[256]) {
    const int64_t rw = v->cols / 16 * (v->m / 2);   /* the record table's row words (exl3i_row_words) */
    exl3i_decode_tile(v->record + tn * rw + tk * (v->m / 2), v->m, v->cb, val, 0);
}

static int chunk_ok(const Exl3iView* v, int64_t c0, int64_t nc) {
    return v && v->record && exl3i_rate_ok(v->m) && v->rows % 16 == 0 && v->cols % 64 == 0 && c0 >= 0 && nc > 0 &&
           c0 % 16 == 0 && nc % 16 == 0 && c0 + nc <= v->rows;
}

float exl3i_ref_a8_ws(int cb) { (void)cb; return EXL3I_A8_WS; }   /* the A8 plane's step (4 K_INV; mad's grid) */

int exl3i_ref_i8plane(const Exl3iView* v, int64_t c0, int64_t nc, int8_t* out) {
    uint16_t val[256];
    int tr[256], tc[256], p;
    int64_t tn, tk;
    if (!chunk_ok(v, c0, nc) || v->cb != EXL3I_CB_MAD || !out) return -1;
    for (p = 0; p < 256; ++p) exl3_pos_to_tile(p, &tr[p], &tc[p]);
    for (tn = c0 / 16; tn < (c0 + nc) / 16; ++tn)
        for (tk = 0; tk < v->cols / 16; ++tk) {
            tile_values(v, tn, tk, val);
            for (p = 0; p < 256; ++p) {   /* position p is W_hat[k = 16 tk + r][n = 16 tn + c] */
                const int64_t n = 16 * tn + tc[p] - c0, k = 16 * tk + tr[p];
                out[exl3i_i8plane_off(n, k, v->cols)] = (int8_t)exl3i_a8_wq(exl3_half_to_float(val[p]));
            }
        }
    return 0;
}

int exl3i_ref_f16plane(const Exl3iView* v, int64_t c0, int64_t nc, uint16_t* out) {
    uint16_t val[256];
    int tr[256], tc[256], p;
    int64_t tn, tk;
    if (!chunk_ok(v, c0, nc) || !exl3i_codebook_name(v->cb) || !out) return -1;
    for (p = 0; p < 256; ++p) exl3_pos_to_tile(p, &tr[p], &tc[p]);
    for (tn = c0 / 16; tn < (c0 + nc) / 16; ++tn)
        for (tk = 0; tk < v->cols / 16; ++tk) {
            tile_values(v, tn, tk, val);
            for (p = 0; p < 256; ++p) {
                const int64_t n = 16 * tn + tc[p] - c0, k = 16 * tk + tr[p];
                out[exl3i_f16plane_off(n, k, nc / 16)] = val[p];
            }
        }
    return 0;
}

static float f16r(float f) { return exl3_half_to_float(exl3_half_from_double((double)f)); }   /* one RNE to fp16 */
static float bf16f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

void exl3i_ref_rotate128(const uint16_t* x, int x_bf16, const uint16_t* suh, uint16_t* out) {
    static const int order[7] = { 0, 1, 2, 3, 4, 5, 6 };   /* in-lane 0, 1; DPP 2..5; lane ^ 16: 6 */
    const float RS = 0.08838834764831845f;
    float v[128];
    int f, s;
    for (f = 0; f < 128; ++f) {
        const float xv = x_bf16 ? bf16f(x[f]) : exl3_half_to_float(x[f]);
        v[f] = f16r(xv * exl3_half_to_float(suh[f]));
    }
    for (s = 0; s < 7; ++s) {
        const int b = 1 << order[s];
        for (f = 0; f < 128; ++f)
            if (!(f & b)) {
                const float lo = v[f], hi = v[f | b];
                v[f] = lo + hi;
                v[f | b] = lo - hi;
            }
    }
    /* the scale's product rounded once: the device's (_Float16)(v * RS) is one v_fma_mixlo_f16 (exact product, one
     * rounding to fp16), not an f32 multiply then a conversion (the two differ on ~4e-5 of values by 1 ulp) */
    for (f = 0; f < 128; ++f) out[f] = exl3_half_from_double((double)v[f] * (double)RS);
}

void exl3i_ref_actq(const uint16_t* x, int64_t x_pitch, int x_bf16, const uint16_t* suh, const uint16_t* suh1,
                    int mats, int64_t M, int64_t K, int64_t group, int8_t* aq, float* sa) {
    const int64_t nkb = K / 128, nbg = group / 128;
    uint16_t* rot = (uint16_t*)malloc((size_t)K * sizeof *rot);
    int mat;
    int64_t r, g, b, k;
    for (mat = 0; mat < mats; ++mat)
        for (r = 0; r < M; ++r) {
            const uint16_t* su = mat ? suh1 : suh;
            for (b = 0; b < nkb; ++b) exl3i_ref_rotate128(x + r * x_pitch + b * 128, x_bf16, su + b * 128, rot + b * 128);
            for (g = 0; g < K / group; ++g) {
                float amax = 0.0f, inv, sc;
                for (k = g * group; k < (g + 1) * group; ++k) amax = fmaxf(amax, fabsf(exl3_half_to_float(rot[k])));
                inv = exl3i_a8_inv(amax);
                sc = exl3i_a8_scale(amax);
                for (k = g * group; k < (g + 1) * group; ++k)
                    aq[((int64_t)mat * M + r) * K + k] = (int8_t)exl3i_a8_q(exl3_half_to_float(rot[k]), inv);
                for (b = g * nbg; b < (g + 1) * nbg; ++b) sa[((int64_t)mat * M + r) * nkb + b] = sc;
            }
        }
    free(rot);
}

void exl3i_ref_a8_gemm(const int8_t* aq, const float* sa, const int8_t* w8, const uint16_t* svh, int mats, int64_t M,
                       int64_t N, int64_t K, double* out) {
    exl3i_ref_a8_gemm_cb(aq, sa, w8, svh, mats, M, N, K, EXL3I_CB_MAD, out);
}

void exl3i_ref_a8_gemm_cb(const int8_t* aq, const float* sa, const int8_t* w8, const uint16_t* svh, int mats, int64_t M,
                          int64_t N, int64_t K, int cb, double* out) {
    const int64_t nkb = K / 128;
    int8_t* wr = (int8_t*)malloc((size_t)(N * K));
    double z[128];
    int64_t n, k, m, nb, kb;
    for (n = 0; n < N; ++n)
        for (k = 0; k < K; ++k) wr[n * K + k] = w8[exl3i_i8plane_off(n, k, K)];
    for (m = 0; m < M; ++m)
        for (nb = 0; nb < N / 128; ++nb) {
            const int mat = mats == 2 && nb * 128 >= N / 2;
            const int8_t* a = aq + ((int64_t)mat * M + m) * K;
            const float* s = sa + ((int64_t)mat * M + m) * nkb;
            int i, b;
            for (i = 0; i < 128; ++i) {
                const int8_t* w = wr + (nb * 128 + i) * K;
                double acc = 0;
                for (kb = 0; kb < nkb; ++kb) {
                    long dot = 0;
                    for (k = kb * 128; k < kb * 128 + 128; ++k) dot += (long)a[k] * w[k];
                    acc += (double)s[kb] * (double)dot;
                }
                z[i] = acc * (double)exl3i_ref_a8_ws(cb);
            }
            for (b = 1; b < 128; b <<= 1)
                for (i = 0; i < 128; ++i)
                    if (!(i & b)) {
                        const double lo = z[i], hi = z[i | b];
                        z[i] = lo + hi;
                        z[i | b] = lo - hi;
                    }
            for (i = 0; i < 128; ++i)
                out[m * N + nb * 128 + i] = z[i] / sqrt(128.0) * (double)exl3_half_to_float(svh[nb * 128 + i]);
        }
    free(wr);
}

void exl3i_ref_a8_gemm_dev(const int8_t* aq, const float* sa, const int8_t* w8, const uint16_t* svh, int mats, int64_t M,
                           int64_t N, int64_t K, int y_dt, void* y, int64_t y_pitch) {
    exl3i_ref_a8_gemm_dev_cb(aq, sa, w8, svh, mats, M, N, K, EXL3I_CB_MAD, y_dt, y, y_pitch);
}

void exl3i_ref_a8_gemm_dev_cb(const int8_t* aq, const float* sa, const int8_t* w8, const uint16_t* svh, int mats,
                              int64_t M, int64_t N, int64_t K, int cb, int y_dt, void* y, int64_t y_pitch) {
    const int64_t nkb = K / 128;
    const float B = 12582912.0f, C = exl3i_ref_a8_ws(cb) * 0.08838834764831845f;   /* exl3i_a8_gemm's p.C, an f32 product */
    int8_t* wr = (int8_t*)malloc((size_t)(N * K));
    float z[128];
    int64_t n, k, m, nb, kb;
    for (n = 0; n < N; ++n)
        for (k = 0; k < K; ++k) wr[n * K + k] = w8[exl3i_i8plane_off(n, k, K)];
    for (m = 0; m < M; ++m)
        for (nb = 0; nb < N / 128; ++nb) {
            const int mat = mats == 2 && nb * 128 >= N / 2;
            const int8_t* a = aq + ((int64_t)mat * M + m) * K;
            const float* s = sa + ((int64_t)mat * M + m) * nkb;
            int i, b, t;
            for (i = 0; i < 128; ++i) {   /* the K loop: raw, the fold, the bias flushes */
                const int8_t* w = wr + (nb * 128 + i) * K;
                float out = 0.0f, corr = 0.0f;
                for (kb = 0; kb < nkb; ++kb) {
                    long dot = 0;
                    for (k = kb * 128; k < kb * 128 + 128; ++k) dot += (long)a[k] * w[k];
                    if (kb % 8 == 0 && kb) {
                        out = fmaf(-B, corr, out);
                        corr = 0.0f;
                    }
                    corr += s[kb];
                    out = fmaf((float)(12582912L + dot), s[kb], out);
                }
                z[i] = fmaf(-B, corr, out);
            }
            /* bits 0-2 (a lane's eight slots), 3 (lane ^ 16), 4 (a wave's two tiles): butterflies, low + high, low - high */
            for (b = 1; b < 32; b <<= 1)
                for (i = 0; i < 128; ++i)
                    if (!(i & b)) {
                        const float lo = z[i], hi = z[i | b];
                        z[i] = lo + hi;
                        z[i | b] = lo - hi;
                    }
            /* bits 5-6: output wave ww sums the four waves' values in wave order, signed by popcount(ww & w2) */
            for (i = 0; i < 128; ++i) {
                const int ww = i >> 5, r0 = i & 31;
                const int64_t n_ = nb * 128 + i;
                float r = 0.0f, o;
                for (t = 0; t < 4; ++t) r = fmaf((__builtin_popcount(ww & t) & 1) ? -1.0f : 1.0f, z[32 * t + r0], r);
                o = (r * C) * exl3_half_to_float(svh[n_]);
                if (y_dt == 1) memcpy((char*)y + (m * y_pitch + n_) * 4, &o, 4);
                else {
                    uint16_t h;
                    if (y_dt == 2) h = exl3_half_from_double((double)o);
                    else { uint32_t u; memcpy(&u, &o, 4); h = (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16); }
                    memcpy((char*)y + (m * y_pitch + n_) * 2, &h, 2);
                }
            }
        }
    free(wr);
}
