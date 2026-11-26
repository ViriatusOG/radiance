/* exl3i_rerank_host_test.c -- exl3i_rerank's host reference (rerank_ref.c, include/exl3i_rerank.h) against two
 * independent evaluations of the head, no device:
 *   (A) the decode form's value in double: every row of the head from the oracle's W_hat (exl3i_decode_what_rows; the
 *       grid index q and the step for mad), dotted with the same f16 rotated row, the whole output Hadamard
 *       (exl3_had128's sign rule), x 1/sqrt(128) x svh, read at the candidate's vocabulary row (R16 or natural): the
 *       reference must be within the f32 summation bound K u L1 of it (L1 = the sum of the terms' magnitudes);
 *   (B) the oracle's weight (exl3i_decode_rows: W = diag(svh) Hb W_hat Hb diag(suh)) times the bf16 row in double: the
 *       semantics (rows, scales, rotations, R16 order); off only by the f16 rounding of the rotated row, so within
 *       2^-9 of the row's L1 (|W| |x|);
 * plus the sort: idx / val descending with ties to the lower id, pads (-1, -inf) last, in place (idx = idx_in), the
 * pair plane, and the order of every candidate set equal to a sort of (A)'s values where they are apart by more than
 * the bound. Last line PASS / FAIL. */
#include "exl3i_rerank_dev.h"
#include "exl3i_rerank.h"
#include "exl3i_decode.h"
#include "exl3_decode.h"
#include "exl3i_encoding.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t RNG = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { RNG ^= RNG << 13; RNG ^= RNG >> 7; RNG ^= RNG << 17; return (uint32_t)(RNG >> 16); }
static int FAILS = 0;

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

static void had_d(double* v) {   /* exl3_had128's sign rule, unnormalised, in double */
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; ++i)
            if (!(i & h)) { const double a = v[i], b = v[i + h]; v[i] = a + b; v[i + h] = a - b; }
}

static void run(int64_t N, int64_t K, const char* rate, const char* cbn, int r16, int64_t M, int64_t R, int x_f16) {
    const int m = exl3i_rate_parse(rate), cb = exl3i_codebook_from_name(cbn);
    const int dq = cb == EXL3I_CB_MAD;
    const int64_t rw = exl3i_row_words(K, m);
    uint32_t* rec = malloc((size_t)(N / 16) * rw * 4);
    uint16_t *suh = malloc((size_t)K * 2), *svh = malloc((size_t)N * 2), *x = malloc((size_t)M * K * 2);
    int32_t *in = malloc((size_t)M * R * 4), *idx = malloc((size_t)M * R * 4), *idx2 = malloc((size_t)M * R * 4);
    float *val = malloc((size_t)M * R * 4), *val2 = malloc((size_t)M * R * 4), *pairs = malloc((size_t)M * R * 8);
    float *what = malloc((size_t)N * K * 4), *wfull = malloc((size_t)N * K * 4);
    double *A = malloc((size_t)N * 8), *L = malloc((size_t)N * 8), *B = malloc((size_t)N * 8), *BL = malloc((size_t)N * 8);
    uint16_t* xr = malloc((size_t)K * 2);
    for (int64_t i = 0; i < (N / 16) * rw; ++i) rec[i] = rnd() ^ (rnd() << 16);
    for (int64_t k = 0; k < K; ++k) suh[k] = exl3_half_from_double((rnd() & 1 ? -1.0 : 1.0) * (0.5 + (rnd() & 1023) / 1024.0));
    for (int64_t n = 0; n < N; ++n) svh[n] = exl3_half_from_double((rnd() & 1 ? -1.0 : 1.0) * (0.25 + (rnd() & 1023) / 2048.0));
    for (int64_t i = 0; i < M * K; ++i) {
        const float f = ((int)(rnd() & 0xFFFF) - 32768) / 8192.0f;
        x[i] = x_f16 ? exl3_half_from_double(f) : f2bf(f);
    }
    /* candidates: distinct ids, some pads (-1, past N), the rest drawn over the vocabulary; row 0 includes the rows
     * of one output block (a tie-rich set) */
    for (int64_t mi = 0; mi < M; ++mi)
        for (int64_t r = 0; r < R; ++r) {
            int32_t g;
            int dup;
            do {
                const uint32_t u = rnd() % 64;
                g = u == 0 ? -1 : u == 1 ? (int32_t)(N + (rnd() % 7)) : (int32_t)(rnd() % N);
                if (mi == 0 && r < 8) g = (int32_t)r;
                dup = 0;
                for (int64_t q = 0; q < r; ++q) dup |= in[mi * R + q] == g && g >= 0 && g < N;
            } while (dup);
            in[mi * R + r] = g;
        }
    memcpy(idx2, in, (size_t)M * R * 4);
    if (exl3i_rerank_ref(x, K, x_f16, rec, suh, svh, in, idx, val, pairs, M, R, N, K, 0, m, cb, r16) != 0 ||
        exl3i_rerank_ref(x, K, x_f16, rec, suh, svh, idx2, idx2, val2, NULL, M, R, N, K, 0, m, cb, r16) != 0) {
        printf("ref refused\n");
        ++FAILS;
        return;
    }
    /* in place == separate */
    if (memcmp(idx, idx2, (size_t)M * R * 4) || memcmp(val, val2, (size_t)M * R * 4)) { printf("in place differs\n"); ++FAILS; }
    Exl3iView v = { N, K, m, cb, rec, suh, 0, svh };
    if (exl3i_decode_what_rows(&v, 0, N, what) != 0 || exl3i_decode_rows(&v, 0, N, wfull) != 0) { printf("oracle refused\n"); ++FAILS; return; }
    const double s4 = 4.0 * exl3_half_to_float(0x1EEE), step = exl3_half_to_float(0x26EE), RS = 0.08838834764831845;
    double worstA = 0, worstB = 0;
    long nbad = 0, nord = 0;
    for (int64_t mi = 0; mi < M; ++mi) {
        exl3i_rerank_ref_rotate(x + mi * K, x_f16, suh, K, xr);
        for (int64_t n = 0; n < N; ++n) {   /* stored row n: z and its magnitude sum */
            double z = 0, l = 0, b = 0, bl = 0;
            for (int64_t k = 0; k < K; ++k) {
                const double w = dq ? rint(what[n * K + k] / s4) : what[n * K + k];
                const double t = w * exl3_half_to_float(xr[k]);
                z += t; l += fabs(t);
                const double xv = x_f16 ? exl3_half_to_float(x[mi * K + k]) : bf2f(x[mi * K + k]);
                b += (double)wfull[n * K + k] * xv; bl += fabs((double)wfull[n * K + k] * xv);
            }
            A[n] = dq ? z * step : z; L[n] = dq ? l * step : l;
            B[n] = b; BL[n] = bl;
        }
        for (int64_t b0 = 0; b0 < N; b0 += 128) {   /* the output rotation, in stored order */
            double l1 = 0;
            for (int i = 0; i < 128; ++i) l1 += L[b0 + i];
            had_d(A + b0);
            for (int i = 0; i < 128; ++i) { A[b0 + i] *= RS * exl3_half_to_float(svh[b0 + i]); L[b0 + i] = l1 * RS * fabs(exl3_half_to_float(svh[b0 + i])); }
        }
        /* B is in stored order too (the oracle's rows are the stored rows); its vocabulary row is exl3i_r16_row */
        int64_t nv = 0;
        for (int64_t r = 0; r < R; ++r) nv += in[mi * R + r] >= 0 && in[mi * R + r] < N;
        for (int64_t o = 0; o < R; ++o) {
            const int32_t g = idx[mi * R + o];
            const float f = val[mi * R + o];
            int32_t pg;
            memcpy(&pg, &pairs[(mi * R + o) * 2], 4);
            if (pg != g || memcmp(&pairs[(mi * R + o) * 2 + 1], &f, 4)) { ++nbad; continue; }
            if (o >= nv) { nbad += !(g == -1 && f == -INFINITY); continue; }
            if (g < 0 || g >= N) { ++nbad; continue; }
            const int64_t s = exl3i_rerank_row(g, N, r16);
            if (r16 && exl3i_r16_row(s, N) != g) { ++nbad; continue; }
            const double eA = fabs((double)f - A[s]) / ((double)K * ldexp(1.0, -24) * L[s] + 1e-300);
            const double eB = fabs((double)f - B[s]) / (ldexp(1.0, -9) * BL[s] + 1e-300);
            if (eA > worstA) worstA = eA;
            if (eB > worstB) worstB = eB;
            nbad += eA > 1.0 || eB > 1.0;
            if (o > 0) {   /* the order: by value, ties to the lower id; and (A) agrees where apart by more than the bound */
                const int32_t g0 = idx[mi * R + o - 1];
                const float f0 = val[mi * R + o - 1];
                if (f0 < f || (f0 == f && g0 > g)) ++nbad;
                const int64_t s0 = exl3i_rerank_row(g0, N, r16);
                const double tol = (double)K * ldexp(1.0, -24) * (L[s] + L[s0]);
                nord += A[s0] < A[s] - tol;
            }
        }
        /* every valid input id appears once in the output */
        for (int64_t r = 0; r < R; ++r) {
            const int32_t g = in[mi * R + r];
            if (g < 0 || g >= N) continue;
            int c = 0;
            for (int64_t o = 0; o < nv; ++o) c += idx[mi * R + o] == g;
            nbad += c != 1;
        }
    }
    printf("N %6lld K %5lld K%-6s %-6s %s M %lld R %3lld x %s: worst |ref - A| = %.3f u L1 (bound K u L1), |ref - B| = %.3f of 2^-9 L1; "
           "%ld bad, %ld misordered%s\n", (long long)N, (long long)K, rate, cbn, r16 ? "r16" : "nat", (long long)M,
           (long long)R, x_f16 ? "f16" : "bf16", worstA * (double)K, worstB, nbad, nord, nbad || nord ? "  FAIL" : "");
    FAILS += nbad != 0 || nord != 0;
    free(rec); free(suh); free(svh); free(x); free(in); free(idx); free(idx2); free(val); free(val2); free(pairs);
    free(what); free(wfull); free(A); free(L); free(B); free(BL); free(xr);
}

int main(void) {
    run(1024, 512, "4.125", "mad", 1, 3, 32, 0);
    run(1024, 512, "2.5", "mad", 0, 2, 32, 0);
    run(1024, 384, "4", "mad", 1, 3, 32, 0);
    run(1024, 256, "7", "mad", 0, 2, 16, 1);
    run(512, 256, "2.5", "mad", 1, 2, 32, 0);
    run(1024, 256, "0.5", "mad", 1, 2, 8, 0);
    run(2048, 256, "6.5", "mad", 1, 2, 64, 0);
    printf("%s (%d failures)\n", FAILS ? "EXL3I_RERANK_HOST FAIL" : "EXL3I_RERANK_HOST PASS", FAILS);
    return FAILS ? 1 : 0;
}
