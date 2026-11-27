/* rerank_ref.c -- exl3i_rerank's host reference (include/exl3i_rerank.h): the device row's values bit for bit, from the
 * container's record through the oracle's decode (exl3import/common/exl3i_decode.c). Slow and plain on purpose; build
 * with -ffp-contract=off (every multiply-add below that should fuse is an explicit fmaf, as on the device). */
#include "exl3i_rerank_dev.h"
#include "exl3i_rerank.h"
#include "exl3i_decode.h"
#include "exl3_decode.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const float kRS = 0.08838834764831845f;   /* 1 / sqrt(128) */

static float act_f(uint16_t b, int f16) {
    if (f16) return exl3_half_to_float(b);
    {
        const uint32_t u = (uint32_t)b << 16;
        float f;
        memcpy(&f, &u, 4);
        return f;
    }
}
static float to_f16(float f) { return exl3_half_to_float(exl3_half_from_double((double)f)); }
static uint16_t h16d(double d) { return exl3_half_from_double(d); }

/* rotate128 (exl3r4d_common.hip.h) on each 128 features: f16(x suh), the stages h = 1 .. 64 in f32 (a +- b), f16(v RS)
 * -- the last as ONE rounding of the exact product: the device compiler fuses rotate128_math's f32 multiply and the f16
 * conversion (v_fma_mix: the product never rounded to f32), and measured on gfx1201 the double-rounded form differs in
 * 1 ulp on ~1 feature in 20,000 (tests/test_rerank.hip checks this rotation against the device row's). x suh is exact
 * in f32 (8 x 11 bits), so its one rounding is the same either way. */
void exl3i_rerank_ref_rotate(const uint16_t* x, int x_f16, const uint16_t* suh, int64_t K, uint16_t* xr) {
    for (int64_t g = 0; g < K; g += 128) {
        float v[128];
        for (int k = 0; k < 128; ++k) {
            const float p = act_f(x[g + k], x_f16) * exl3_half_to_float(suh[g + k]);
            v[k] = to_f16(p);
        }
        for (int h = 1; h < 128; h <<= 1)
            for (int i = 0; i < 128; ++i)
                if (!(i & h)) {
                    const float a = v[i], b = v[i + h];
                    v[i] = a + b;
                    v[i + h] = a - b;
                }
        for (int k = 0; k < 128; ++k) xr[g + k] = h16d((double)v[k] * (double)kRS);   /* one rounding (below) */
    }
}

/* better(a, b): a sorts before b -- the larger value, ties to the lower id, then the earlier slot (a NaN ranks as -inf) */
static int better(float va, int32_t ia, int64_t ca, float vb, int32_t ib, int64_t cb) {
    if (va != va) va = -INFINITY;
    if (vb != vb) vb = -INFINITY;
    return va > vb || (va == vb && (ia < ib || (ia == ib && ca < cb)));
}

int exl3i_rerank_ref(const uint16_t* x, int64_t x_pitch, int x_f16, const uint32_t* record, const uint16_t* suh,
                     const uint16_t* svh, const int32_t* idx_in, int32_t* idx, float* val, float* pairs, int64_t M,
                     int64_t R, int64_t N, int64_t K, int64_t voff, int m, int cb, int r16) {
    if (!x || !record || !suh || !svh || !idx_in || !idx || !val || M < 0 || R < 1 || R > EXL3I_RERANK_MAX_R ||
        N < 128 || N % 128 || K < 128 || K % 128 || !exl3i_rate_ok(m) || cb != EXL3I_CB_MAD)
        return -1;
    const int W = m / 2;
    const int64_t kt = K / 16;
    const int dq = cb == EXL3I_CB_MAD;
    const float s4 = 4.0f * exl3_half_to_float(0x1EEE), step = exl3_half_to_float(0x26EE);
    uint16_t* xr = (uint16_t*)malloc((size_t)K * 2);
    float* xf = (float*)malloc((size_t)K * 4);
    int32_t* id = (int32_t*)malloc((size_t)R * 4);
    float* sc = (float*)malloc((size_t)R * 4);
    int* ok = (int*)malloc((size_t)R * sizeof(int));
    if (!xr || !xf || !id || !sc || !ok) { free(xr); free(xf); free(id); free(sc); free(ok); return -1; }
    int tr[256], tc[256];
    for (int p = 0; p < 256; ++p) exl3_pos_to_tile(p, &tr[p], &tc[p]);
    for (int64_t mi = 0; mi < M; ++mi) {
        exl3i_rerank_ref_rotate(x + mi * x_pitch, x_f16, suh, K, xr);
        for (int64_t k = 0; k < K; ++k) xf[k] = exl3_half_to_float(xr[k]);
        for (int64_t r = 0; r < R; ++r) {
            const int64_t g = idx_in[mi * R + r], local = g - voff;
            id[r] = (int32_t)g;
            ok[r] = g >= 0 && local >= 0 && local < N;
            sc[r] = -INFINITY;
            if (!ok[r]) continue;
            const int64_t s = exl3i_rerank_row(local, N, r16), b = s / 128;
            const int j = (int)(s % 128);
            float z[128];
            for (int st = 0; st < 8; ++st) {
                const int64_t tn = b * 8 + st;
                float A[2][16];
                for (int hf = 0; hf < 2; ++hf) {
                    for (int ct = 0; ct < 16; ++ct) A[hf][ct] = 0.0f;
                    for (int64_t cc = 0; cc < K / 128; ++cc)
                        for (int tq = 0; tq < 4; ++tq) {
                            const int64_t tk = 8 * cc + 4 * hf + tq;
                            uint16_t v16[256];
                            float v[16][16];   /* [rt (k)][ct (n)] */
                            exl3i_decode_tile(record + (tn * kt + tk) * W, m, cb, v16, NULL);
                            for (int p = 0; p < 256; ++p) {
                                const float f = exl3_half_to_float(v16[p]);
                                v[tr[p]][tc[p]] = dq ? rintf(f / s4) : f;
                            }
                            for (int ct = 0; ct < 16; ++ct)
                                for (int rt = 0; rt < 16; ++rt) A[hf][ct] = fmaf(v[rt][ct], xf[16 * tk + rt], A[hf][ct]);
                        }
                }
                for (int ct = 0; ct < 16; ++ct) {
                    const float a = A[0][ct] + A[1][ct];
                    z[16 * st + ct] = dq ? a * step : a;
                }
            }
            for (int w = 64, h = 1; w >= 1; w >>= 1, h <<= 1)
                for (int i = 0; i < w; ++i) z[i] = (j & h) ? z[2 * i] - z[2 * i + 1] : z[2 * i] + z[2 * i + 1];
            {
                const float y = z[0] * kRS;
                sc[r] = y * exl3_half_to_float(svh[s]);
            }
        }
        /* the sort: each valid slot to its rank, then the pads */
        int64_t nv = 0;
        for (int64_t c = 0; c < R; ++c) nv += ok[c];
        for (int64_t c = 0; c < R; ++c) {
            if (!ok[c]) continue;
            int64_t rank = 0;
            for (int64_t d = 0; d < R; ++d)
                if (d != c && ok[d] && better(sc[d], id[d], d, sc[c], id[c], c)) ++rank;
            idx[mi * R + rank] = id[c];
            val[mi * R + rank] = sc[c];
            if (pairs) {
                memcpy(&pairs[(mi * R + rank) * 2], &id[c], 4);
                pairs[(mi * R + rank) * 2 + 1] = sc[c];
            }
        }
        for (int64_t o = nv; o < R; ++o) {
            const int32_t pad = -1;
            idx[mi * R + o] = pad;
            val[mi * R + o] = -INFINITY;
            if (pairs) {
                memcpy(&pairs[(mi * R + o) * 2], &pad, 4);
                pairs[(mi * R + o) * 2 + 1] = -INFINITY;
            }
        }
    }
    free(xr); free(xf); free(id); free(sc); free(ok);
    return 0;
}
