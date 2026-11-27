/* exl3i_decode.c -- the "exl3i" oracle (exl3i_decode.h). Plain C99; build with -ffp-contract=off. */
#include "exl3i_decode.h"
#include "exl3_decode.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int exl3i_mask(int n) {
    int m = 0, i;
    for (i = 0; i < 16; ++i) if (((i + 1) * n) / 16 - (i * n) / 16) m |= 1 << i;
    return m;
}

static uint16_t salt16(int i) { return (uint16_t)((0x9E37u * (uint32_t)((i & 15) + 1)) & 0xFFFFu); }

uint32_t exl3i_mad_sum(uint32_t state) {
    const uint32_t h = ((state & 0xFFFFFFu) * EXL3I_MAD_A + EXL3I_MAD_B) & 0x3F3F3F3Fu;
    return (h & 255u) + (h >> 8 & 255u) + (h >> 16 & 255u) + (h >> 24);
}

int exl3i_codebook_q(uint32_t state, int cb) {
    if (cb == EXL3I_CB_MAD) return (int)exl3i_mad_sum(state) - 126;
    return 0;
}

uint16_t exl3i_codebook_value(uint32_t state, int cb) {
    /* mad: fp16(q 4 K_INV): q 4 K_INV is exact in f32, one rounding (exl3hashm's __float2half_rn) */
    float s4;
    if (cb != EXL3I_CB_MAD) return 0;
    s4 = 4.0f * exl3_half_to_float(0x1EEE);
    return exl3_half_from_double((double)((float)exl3i_codebook_q(state, cb) * s4));
}

int exl3i_decode_tile(const uint32_t* words, int m, int cb, uint16_t values[256], uint16_t* states) {
    const int ka = m / 16, mask = exl3i_mask(m % 16), nw = m / 2, R = 32 * nw;
    int i, S = 0;
    if (!words || !values || !exl3i_rate_ok(m) || !exl3i_codebook_name(cb)) return -1;
    for (i = 0; i < 256; ++i) {
        const int d = ka + ((mask >> (i & 15)) & 1);
        int start, wi, o;
        uint64_t w2;
        uint32_t st;
        S += d;
        start = (S - 16) % R;
        if (start < 0) start += R;
        wi = start >> 5;
        o = start & 31;
        w2 = ((uint64_t)words[wi] << 32) | (uint64_t)words[(wi + 1) % nw];
        st = (uint32_t)((w2 >> (48 - o)) & 0xFFFFu);
        if (states) states[i] = (uint16_t)st;
        values[i] = exl3i_codebook_value(d ? st : (st ^ salt16(i)), cb);
    }
    return 0;
}

int exl3i_decode_what_rows(const Exl3iView* v, int64_t row0, int64_t n, float* out) {
    uint16_t val[256];
    int tr[256], tc[256], p;
    int64_t tn, tk, rw;
    if (!v || !v->record || !out || !exl3i_rate_ok(v->m) || !exl3i_codebook_name(v->cb)) return -1;
    if (v->rows <= 0 || v->cols <= 0 || v->rows % 16 || v->cols % 16 || row0 < 0 || n < 0 || row0 + n > v->rows)
        return -2;
    rw = v->cols / 16 * (v->m / 2);
    for (p = 0; p < 256; ++p) exl3_pos_to_tile(p, &tr[p], &tc[p]);
    for (tn = row0 / 16; tn * 16 < row0 + n; ++tn)
        for (tk = 0; tk < v->cols / 16; ++tk) {
            exl3i_decode_tile(v->record + tn * rw + tk * (v->m / 2), v->m, v->cb, val, 0);
            for (p = 0; p < 256; ++p) {   /* position p decodes W_hat[k = 16 tk + r][n = 16 tn + c] */
                const int64_t i = tn * 16 + tc[p];
                if (i < row0 || i >= row0 + n) continue;
                out[(i - row0) * v->cols + tk * 16 + tr[p]] = exl3_half_to_float(val[p]);
            }
        }
    return 0;
}

/* One 128-row block [b0, b0 + 128) of W into blk, f32 [128][cols], exl3_decode.c's order: W_hat, had over k (each
 * row), x suh, had over n (each column), x svh. */
static int decode_block(const Exl3iView* v, int64_t b0, float* blk, float* col) {
    const int64_t K = v->cols;
    int64_t r, k;
    const int rc = exl3i_decode_what_rows(v, b0, 128, blk);
    if (rc) return rc;
    for (r = 0; r < 128; ++r) {
        float* row = blk + r * K;
        const uint16_t* su = (v->suh1 && b0 + r >= v->rows / 2) ? v->suh1 : v->suh;
        exl3_had128(row, K);
        for (k = 0; k < K; ++k) row[k] *= exl3_half_to_float(su[k]);
    }
    for (k = 0; k < K; ++k) {
        for (r = 0; r < 128; ++r) col[r] = blk[r * K + k];
        exl3_had128(col, 128);
        for (r = 0; r < 128; ++r) blk[r * K + k] = col[r];
    }
    for (r = 0; r < 128; ++r) {
        const float s = exl3_half_to_float(v->svh[b0 + r]);
        for (k = 0; k < K; ++k) blk[r * K + k] *= s;
    }
    return 0;
}

int exl3i_decode_rows(const Exl3iView* v, int64_t row0, int64_t n, float* out) {
    float col[128];
    float* blk;
    int64_t b0;
    int rc = 0;
    if (!v || !v->record || !v->suh || !v->svh || !out) return -1;
    if (v->rows <= 0 || v->cols <= 0 || v->rows % 128 || v->cols % 128 || (v->suh1 && v->rows % 256) || row0 < 0 ||
        n < 0 || row0 + n > v->rows)
        return -2;
    if (n == 0) return 0;
    blk = (float*)malloc((size_t)(128 * v->cols) * sizeof *blk);
    if (!blk) return -3;
    for (b0 = row0 / 128 * 128; b0 < row0 + n && !rc; b0 += 128) {
        const int64_t lo = b0 > row0 ? b0 : row0, hi = b0 + 128 < row0 + n ? b0 + 128 : row0 + n;
        rc = decode_block(v, b0, blk, col);
        if (!rc) memcpy(out + (lo - row0) * v->cols, blk + (lo - b0) * v->cols, (size_t)((hi - lo) * v->cols) * sizeof *out);
    }
    free(blk);
    return rc;
}
