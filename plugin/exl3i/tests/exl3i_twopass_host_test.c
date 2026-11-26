/* exl3i_twopass_host_test.c -- the two-pass prefill's host test: no device. The planes' layout byte for byte, the
 * device passes' index arithmetic emulated lane by lane, the arena plan at every trunk shape, the dispatch of the
 * real nt.c / rows.c (compiled in, the device launchers stubbed: they record their argument blocks), and the
 * refusals. Sections (each prints a line; the last line is PASS or FAIL with the count):
 *   A  codebook: mad's 65536 values on the int8 grid (q = value / WS exactly, fp16(q WS) = value);
 *      the device's closed forms (exl3i_rate.hip.h) emulated: int8 index and fp16 bits, every state
 *   B  the int8 plane: the reference (exl3i_ref_i8plane) against radiance's own fragment map (libr4d/r4d_fp8_frag.h
 *      frag_src, when built with it) and the oracle's W_hat, every rate; wdec8 emulated lane by lane (the fragment's
 *      element map, the lane ^ 16 exchange, the 32-byte store), byte for byte
 *   C  the f16 plane (T2's wdec): the reference against the oracle's W_hat and T2's wdec emulated lane by lane
 *   D  the activation side: rotate128's transcription against the oracle's exl3_had128; the quantiser's invariants
 *      (|q| <= 127, the group's max at +-127, the error within half a step, zero groups), a8t / a8g groups, both halves
 *   E  the A8 arithmetic: exl3i_ref_a8_gemm against x . W^T in double (the oracle's W), unstacked and stacked
 *   F  the arena at M = 1, 511, 512, 513, 2048, 4096 for every trunk shape (and the head's chunks): offsets aligned and
 *      disjoint, sizes, the threshold, the scratch hook's answer (params only, as the engine calls it)
 *   G  dispatch through rows.c's rows and nt.c's launch: f16 / T2 / A8 by act and real rows, the argument blocks,
 *      multi-chunk launches, the threshold knob
 *   H  refusals: act spellings, A8 on a codebook other than mad, a group that does not divide K, the rows' constraints, rows past the
 *      scratch, encodings */
#define _POSIX_C_SOURCE 200809L   /* posix_memalign */
#include "exl3i.h"
#include "exl3i_dev.h"
#include "exl3i_rerank_dev.h"
#include "exl3i_a8.h"
#include "exl3i_a8_dev.h"
#include "exl3i_pin.h"
#include "exl3i_decode.h"
#include "exl3_decode.h"
#include "exl3i_twopass_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int FAILS = 0, CHECKS = 0;
static void check(int ok, const char* what) {
    ++CHECKS;
    if (!ok) { ++FAILS; fprintf(stderr, "FAIL: %s\n", what); }
}

static uint64_t RNG = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void) { RNG ^= RNG << 13; RNG ^= RNG >> 7; RNG ^= RNG << 17; return (uint32_t)(RNG >> 16); }
static float rndn(void) { float s = 0; int i; for (i = 0; i < 4; ++i) s += (rnd() & 0xFFFF) / 65536.0f; return (s - 2.0f) * 1.7320508f; }
static uint16_t f2h(float f) { return exl3_half_from_double((double)f); }
static float h2f(uint16_t h) { return exl3_half_to_float(h); }

/* ---------------------------------------------------------------- the device launchers, stubbed */
static int n_nt[4], n_actq, n_wdec, n_gemm, n_mad;
static Exl3iGemm last_nt, last_mad;
static Exl3iA8Args last_actq, wdec_log[64], gemm_log[64];
static void reset_calls(void) { n_nt[3] = n_actq = n_wdec = n_gemm = n_mad = 0; }
int exl3i_launch_nt_mad(const Exl3iGemm* a, int npw, void* s) { (void)npw; (void)s; ++n_nt[3]; last_nt = *a; return RAD_OK; }
int exl3i_launch_mad_gemv(const Exl3iGemm* a, int npw, void* s) { (void)npw; (void)s; ++n_mad; last_mad = *a; return RAD_OK; }
/* exl3i_rerank's (rows.c lists its row; rerank.c is compiled in): not exercised here (tests/exl3i_rerank_host_test.c) */
int exl3i_launch_rerank(const Exl3iRerankCall* c, void* s) { (void)c; (void)s; return RAD_OK; }
/* the embedding gathers' device launchers (embed_exl3i.hip, embed_tq.hip; rows.c lists their rows): not built here, and not
 * exercised by this test (their host rows are checked by tests/embed_exl3i_test.cpp) */
int embed_exl3i_device_launch(const RadArgs* a, RadStream s) { (void)a; (void)s; return RAD_OK; }
int embed_tq_device_launch(const RadArgs* a, RadStream s) { (void)a; (void)s; return RAD_OK; }
int exl3i_a8_actq(const Exl3iA8Args* a, void* s) { (void)s; ++n_actq; last_actq = *a; return RAD_OK; }
int exl3i_a8_wdec8(const Exl3iA8Args* a, void* s) { (void)s; if (n_wdec < 64) wdec_log[n_wdec] = *a; ++n_wdec; return RAD_OK; }
int exl3i_a8_gemm(const Exl3iA8Args* a, void* s) { (void)s; if (n_gemm < 64) gemm_log[n_gemm] = *a; ++n_gemm; return RAD_OK; }
/* AQ: the producer-side quantiser (exl3i_actq), recorded */
static int n_aq;
static Exl3iAqArgs last_aq;
int exl3i_aq_actq(const Exl3iAqArgs* a, void* s) { (void)s; ++n_aq; last_aq = *a; return RAD_OK; }
/* FUSE: one launch on the device; here its two passes' records (the dispatch checks count both) */
int exl3i_a8_gemmd(const Exl3iA8Args* a, void* s) { int rc = exl3i_a8_wdec8(a, s); return rc == RAD_OK ? exl3i_a8_gemm(a, s) : rc; }

#ifndef EXL3I_NO_LIBR4D
void exl3i_test_frag_src(int64_t rt, int64_t ks, int64_t lane, int64_t t, int64_t j, int64_t* row, int64_t* k);
int exl3i_test_frag_consts(void);   /* kSuperK, kTileBytes, kLaneBytes as 64 / 1024 / 32 */
#endif

typedef struct { int m, cb; int64_t N, K; int stacked; uint32_t* rec; uint16_t *suh, *suh1, *svh; } Wt;

static Wt make_w(int m, int cb, int64_t N, int64_t K, int stacked) {
    Wt w;
    size_t n = (size_t)(N / 16) * (size_t)exl3i_row_words(K, m), i;
    w.m = m; w.cb = cb; w.N = N; w.K = K; w.stacked = stacked;
    w.rec = (uint32_t*)malloc(n * 4);
    for (i = 0; i < n; ++i) w.rec[i] = rnd() ^ (rnd() << 16);
    w.suh = (uint16_t*)malloc((size_t)K * 2); w.suh1 = (uint16_t*)malloc((size_t)K * 2); w.svh = (uint16_t*)malloc((size_t)N * 2);
    for (i = 0; i < (size_t)K; ++i) { w.suh[i] = f2h((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f));
                                     w.suh1[i] = f2h((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f)); }
    for (i = 0; i < (size_t)N; ++i) w.svh[i] = f2h((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f));
    return w;
}
static void free_w(Wt* w) { free(w->rec); free(w->suh); free(w->suh1); free(w->svh); }
static Exl3iView view_of(const Wt* w) {
    Exl3iView v;
    v.rows = w->N; v.cols = w->K; v.m = w->m; v.cb = w->cb; v.record = w->rec; v.suh = w->suh;
    v.suh1 = w->stacked ? w->suh1 : 0; v.svh = w->svh;
    return v;
}

/* ================================================================ A: codebook */
static void test_codebook(void) {
    int bad = 0, s;
    const float ws = 4.0f * h2f(0x1EEE);
    check(ws == EXL3I_A8_WS, "A: EXL3I_A8_WS is 4 x fp16 0x1EEE");
    for (s = 0; s < 65536; ++s) {
        const uint16_t v = exl3i_codebook_value((uint32_t)s, EXL3I_CB_MAD);
        const int q = exl3i_codebook_q((uint32_t)s, EXL3I_CB_MAD);
        /* as values: the oracle's q = rint(-0.3) is -0, so v may be fp16 -0 where q ws is +0 */
        if (q < -127 || q > 127 || exl3i_a8_wq(h2f(v)) != q || h2f(exl3_half_from_double((double)((float)q * ws))) != h2f(v)) ++bad;
    }
    check(bad == 0, "A: mad values on the int8 grid");
    check(exl3i_codebook_value(0u, 1) == 0, "A: a codebook other than mad has no value");
    printf("A codebook: 65536 mad states on the grid, %d off\n", bad);
}

/* decode element e of lane `lane` in a tile = the ring position (the fragment's order; exl3i_gpu_test's k_dec map) */
static int lane_pos(int lane, int e) {
    const int n = lane & 15, h = lane >> 4, base = 32 * (n & 7) + 4 * (n >> 3) + 16 * h;
    return base + 8 * ((e >> 1) & 1) + (e & 1) + 2 * (e >> 2);
}

/* ================================================================ B: the int8 plane */
static void test_i8plane(void) {
    int m, bad_ref = 0, bad_frag = 0, bad_emu = 0, bad_map = 0, rates = 0;
    const int64_t N = 256, K = 256, c0 = 128, nc = 128;   /* a chunk that does not start at 0 */
    {   /* the element map the emulation relies on: position of (lane, e) is W_hat[frag16_k(lane, e)][lane & 15] */
        int lane, e;
        for (lane = 0; lane < 32; ++lane)
            for (e = 0; e < 8; ++e) { int r, c; exl3_pos_to_tile(lane_pos(lane, e), &r, &c); bad_map += r != exl3i_frag16_k(lane, e) || c != (lane & 15); }
        check(bad_map == 0, "B: the fragment's element map = exl3_pos_to_tile");
    }
#ifndef EXL3I_NO_LIBR4D
    check(exl3i_test_frag_consts() == 1, "B: r4d_fp8_frag.h's kSuperK / kTileBytes / kLaneBytes are 64 / 1024 / 32");
#endif
    for (m = 2; m <= 128; m += 2, ++rates) {
        Wt w = make_w(m, EXL3I_CB_MAD, N, K, 0);
        Exl3iView v = view_of(&w);
        int8_t* pl = (int8_t*)malloc((size_t)(nc * K));
        int8_t* emu = (int8_t*)calloc((size_t)(nc * K), 1);
        float* what = (float*)malloc((size_t)(N * K) * sizeof(float));
        int64_t n, k;
        exl3i_ref_i8plane(&v, c0, nc, pl);
        exl3i_decode_what_rows(&v, 0, N, what);
        for (n = 0; n < nc; ++n)   /* every byte where exl3i_i8plane_off puts W_hat[c0 + n][k] */
            for (k = 0; k < K; ++k) bad_ref += pl[exl3i_i8plane_off(n, k, K)] != exl3i_a8_wq(what[(c0 + n) * K + k]);
#ifndef EXL3I_NO_LIBR4D
        {   /* radiance's own definition, walked forwards */
            int64_t rt, ks, lane, t, j;
            for (rt = 0; rt < nc / 16; ++rt)
                for (ks = 0; ks < K / 64; ++ks)
                    for (lane = 0; lane < 32; ++lane)
                        for (t = 0; t < 4; ++t)
                            for (j = 0; j < 8; ++j) {
                                int64_t row, kk;
                                exl3i_test_frag_src(rt, ks, lane, t, j, &row, &kk);
                                bad_frag += pl[(rt * (K / 64) + ks) * 1024 + lane * 32 + 8 * t + j] !=
                                            exl3i_a8_wq(what[(c0 + row) * K + kk]);
                            }
        }
#endif
        {   /* wdec8, lane by lane: tile (4 ks + t, c0/16 + rt) decoded, int8, the exchange, the 32-byte store */
            int64_t rt, ks;
            const int64_t rw = exl3i_row_words(K, m);
            for (rt = 0; rt < nc / 16; ++rt)
                for (ks = 0; ks < K / 64; ++ks) {
                    uint32_t o[32][8];
                    int t, lane, e;
                    for (t = 0; t < 4; ++t) {
                        uint16_t val[256];
                        uint32_t d0[32], d1[32];
                        exl3i_decode_tile(w.rec + (c0 / 16 + rt) * rw + (4 * ks + t) * (m / 2), m, w.cb, val, 0);
                        for (lane = 0; lane < 32; ++lane) {
                            uint32_t b[2] = { 0, 0 };
                            for (e = 0; e < 8; ++e)
                                b[e >> 2] |= (uint32_t)(exl3i_a8_wq(h2f(val[lane_pos(lane, e)])) & 0xFF) << (8 * (e & 3));
                            d0[lane] = b[0]; d1[lane] = b[1];
                        }
                        for (lane = 0; lane < 32; ++lane) {
                            const int p = lane ^ 16;
                            const uint32_t recv = exl3i_wdec8_send(p, d0[p], d1[p]);
                            exl3i_wdec8_frag(lane, d0[lane], d1[lane], recv, &o[lane][2 * t], &o[lane][2 * t + 1]);
                        }
                    }
                    for (lane = 0; lane < 32; ++lane) memcpy(emu + (rt * (K / 64) + ks) * 1024 + lane * 32, o[lane], 32);
                }
            bad_emu += memcmp(emu, pl, (size_t)(nc * K)) != 0;
        }
        free(pl); free(emu); free(what); free_w(&w);
    }
    check(bad_ref == 0, "B: the int8 plane holds W_hat's int8 index where exl3i_i8plane_off says");
    check(bad_frag == 0, "B: the int8 plane = r4d_fp8::frag_src's permutation");
    check(bad_emu == 0, "B: wdec8's lane arithmetic gives the reference plane");
#ifndef EXL3I_NO_LIBR4D
    printf("B int8 plane: %d rates x [%lld..%lld) x %lld, %d bytes off W_hat, %d off r4d_fp8::frag_src, %d rates' wdec8 "
           "emulation differ\n", rates, (long long)c0, (long long)(c0 + nc), (long long)K, bad_ref, bad_frag, bad_emu);
#else
    printf("B int8 plane: %d rates, %d bytes off W_hat, frag_src cross-check skipped (no libr4d), %d rates' wdec8 "
           "emulation differ\n", rates, bad_ref, bad_emu);
#endif
}

/* ================================================================ C: the f16 plane (T2) */
static void test_f16plane(void) {
    int m, cb, bad_ref = 0, bad_emu = 0, n_cases = 0;
    const int64_t N = 256, K = 128, c0 = 0, nc = 256;
    for (cb = EXL3I_CB_MAD; cb <= EXL3I_CB_MAD; ++cb)   /* mad: the one codebook */
        for (m = 2; m <= 128; m += 6, ++n_cases) {
            Wt w = make_w(m, cb, N, K, 0);
            Exl3iView v = view_of(&w);
            uint16_t* pl = (uint16_t*)malloc((size_t)(nc * K) * 2);
            uint16_t* emu = (uint16_t*)malloc((size_t)(nc * K) * 2);
            float* what = (float*)malloc((size_t)(N * K) * sizeof(float));
            const int64_t ntc = nc / 16, rw = exl3i_row_words(K, m);
            int64_t n, k, ks, nt;
            exl3i_ref_f16plane(&v, c0, nc, pl);
            exl3i_decode_what_rows(&v, 0, N, what);
            for (n = 0; n < nc; ++n)
                for (k = 0; k < K; ++k) bad_ref += h2f(pl[exl3i_f16plane_off(n, k, ntc)]) != what[(c0 + n) * K + k];
            /* T2's wdec: tile = ks * ntc + nt, ((v8h*)raw)[tile * 32 + lane] = decode(lane) */
            for (ks = 0; ks < K / 16; ++ks)
                for (nt = 0; nt < ntc; ++nt) {
                    uint16_t val[256];
                    int lane, e;
                    exl3i_decode_tile(w.rec + (c0 / 16 + nt) * rw + ks * (m / 2), m, cb, val, 0);
                    for (lane = 0; lane < 32; ++lane)
                        for (e = 0; e < 8; ++e) emu[((ks * ntc + nt) * 32 + lane) * 8 + e] = val[lane_pos(lane, e)];
                }
            bad_emu += memcmp(emu, pl, (size_t)(nc * K) * 2) != 0;
            free(pl); free(emu); free(what); free_w(&w);
        }
    check(bad_ref == 0, "C: the f16 plane holds W_hat where exl3i_f16plane_off says");
    check(bad_emu == 0, "C: T2's wdec lane order gives the reference plane");
    printf("C f16 plane (T2's wdec): %d cases (mad x 22 rates), %d values off W_hat, %d emulations differ\n", n_cases,
           bad_ref, bad_emu);
}

/* ================================================================ D: the activation side */
static void test_act(void) {
    int i, r, ulp1 = 0, worse = 0, n = 0, bad_q = 0, bad_max = 0, bad_err = 0, bad_rep = 0, bad_half = 0;
    for (r = 0; r < 64; ++r) {   /* the transcription is the transform: within 1 fp16 ulp of the oracle's */
        uint16_t x[128], su[128], o[128];
        float f[128];
        for (i = 0; i < 128; ++i) { x[i] = f2h(rndn() * (r % 7 + 1)); su[i] = f2h((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f)); }
        exl3i_ref_rotate128(x, 0, su, o);
        for (i = 0; i < 128; ++i) f[i] = h2f(f2h(h2f(x[i]) * h2f(su[i])));
        exl3_had128(f, 128);
        for (i = 0; i < 128; ++i, ++n) {
            const uint16_t want = f2h(f[i]);
            const int d = abs((int)(int16_t)o[i] - (int)(int16_t)want);
            if (o[i] != want) { if (d == 1) ++ulp1; else ++worse; }
        }
    }
    check(worse == 0 && ulp1 * 100 < n, "D: rotate128's transcription within 1 fp16 ulp of exl3_had128 (rarely)");
    {
        const int64_t M = 9, K = 1024;
        const int64_t groups[3] = { 128, 512, 1024 };   /* a8g128, a8g512, a8t at K = 1024 */
        uint16_t* x = (uint16_t*)malloc((size_t)(M * K) * 2);
        uint16_t* su = (uint16_t*)malloc((size_t)K * 2);
        uint16_t* su1 = (uint16_t*)malloc((size_t)K * 2);
        int8_t* aq = (int8_t*)malloc((size_t)(2 * M * K));
        float* sa = (float*)malloc((size_t)(2 * M * (K / 128)) * 4);
        uint16_t* rot = (uint16_t*)malloc((size_t)K * 2);
        int gi;
        for (i = 0; i < M * K; ++i) x[i] = f2h(rndn() * (1.0f + (float)(i % 5)));
        for (i = 0; i < 256; ++i) x[3 * K + i] = 0;   /* row 3: two zero blocks (a zero group at g128) */
        for (i = 0; i < K; ++i) { su[i] = f2h(0.75f + (rnd() & 255) / 1024.f); su1[i] = f2h(-1.25f + (rnd() & 255) / 1024.f); }
        for (gi = 0; gi < 3; ++gi) {
            const int64_t g = groups[gi];
            int mat;
            int64_t row, b, k;
            exl3i_ref_actq(x, K, 0, su, su1, 2, M, K, g, aq, sa);
            for (mat = 0; mat < 2; ++mat)
                for (row = 0; row < M; ++row) {
                    for (b = 0; b < K / 128; ++b) exl3i_ref_rotate128(x + row * K + b * 128, 0, (mat ? su1 : su) + b * 128, rot + b * 128);
                    for (b = 0; b < K / g; ++b) {
                        const float sc = sa[(mat * M + row) * (K / 128) + b * (g / 128)];
                        int qmax = 0;
                        int64_t bb;
                        for (bb = b * (g / 128); bb < (b + 1) * (g / 128); ++bb) bad_rep += sa[(mat * M + row) * (K / 128) + bb] != sc;
                        for (k = b * g; k < (b + 1) * g; ++k) {
                            const int q = aq[(mat * M + row) * K + k];
                            const float vv = h2f(rot[k]);
                            if (q < -127 || q > 127) ++bad_q;
                            if (abs(q) > qmax) qmax = abs(q);
                            if (fabsf((float)q * sc - vv) > 0.5001f * sc + 1e-30f) ++bad_err;
                        }
                        if (sc > 0 ? qmax != 127 : qmax != 0) ++bad_max;
                    }
                }
            /* the two halves differ (suh vs suh.1) */
            bad_half += memcmp(aq, aq + M * K, (size_t)(M * K)) == 0;
        }
        free(x); free(su); free(su1); free(aq); free(sa); free(rot);
    }
    check(bad_q == 0 && bad_max == 0 && bad_err == 0 && bad_rep == 0 && bad_half == 0, "D: the quantiser's invariants");
    printf("D activation: rotate128 transcription vs exl3_had128: %d of %d values 1 ulp off, %d worse; quantiser (g128 / "
           "g512 / a8t, 2 halves, zero groups): %d out of range, %d groups without +-127, %d over half a step, %d scales not "
           "repeated, %d halves equal\n", ulp1, n, worse, bad_q, bad_max, bad_err, bad_rep, bad_half);
}

/* ================================================================ E: the A8 arithmetic */
static void test_a8_math(void) {
    int st;
    for (st = 0; st < 2; ++st) {
        const int64_t M = 24, N = 256, K = 512;
        const int m = 64;   /* K4 */
        Wt w = make_w(m, EXL3I_CB_MAD, N, K, st);
        Exl3iView v = view_of(&w);
        float* W = (float*)malloc((size_t)(N * K) * sizeof(float));
        uint16_t* x = (uint16_t*)malloc((size_t)(M * K) * 2);
        int8_t* aq = (int8_t*)malloc((size_t)(2 * M * K));
        float* sa = (float*)malloc((size_t)(2 * M * (K / 128)) * 4);
        int8_t* w8 = (int8_t*)malloc((size_t)(N * K));
        double* y = (double*)malloc((size_t)(M * N) * sizeof(double));
        double se = 0, sr = 0;
        int64_t i, n, k;
        char msg[96];
        exl3i_decode_rows(&v, 0, N, W);
        for (i = 0; i < M * K; ++i) x[i] = f2h(rndn());
        exl3i_ref_actq(x, K, 0, w.suh, w.suh1, st ? 2 : 1, M, K, 128, aq, sa);
        exl3i_ref_i8plane(&v, 0, N, w8);
        exl3i_ref_a8_gemm(aq, sa, w8, w.svh, st ? 2 : 1, M, N, K, y);
        for (i = 0; i < M; ++i)
            for (n = 0; n < N; ++n) {
                double s = 0;
                for (k = 0; k < K; ++k) s += (double)h2f(x[i * K + k]) * W[n * K + k];
                sr += s * s;
                se += (y[i * N + n] - s) * (y[i * N + n] - s);
            }
        snprintf(msg, sizeof msg, "E: the A8 reference near x . W^T (%s): rel rms %.3g", st ? "stacked" : "plain", sqrt(se / sr));
        check(sqrt(se / sr) < 2e-2, msg);
        {   /* the device-arithmetic emulation (f32 y) against the exact fold of the same operands */
            float* yd = (float*)malloc((size_t)(M * N) * sizeof(float));
            double de = 0, dr = 0;
            exl3i_ref_a8_gemm_dev(aq, sa, w8, w.svh, st ? 2 : 1, M, N, K, 1, yd, N);
            for (i = 0; i < M * N; ++i) { de += ((double)yd[i] - y[i]) * ((double)yd[i] - y[i]); dr += y[i] * y[i]; }
            snprintf(msg, sizeof msg, "E: the device-arithmetic emulation near the exact fold (%s): rel rms %.3g",
                     st ? "stacked" : "plain", sqrt(de / dr));
            check(sqrt(de / dr) < 2e-3, msg);
            free(yd);
        }
        printf("E A8 arithmetic (%s, [%lld x %lld] K4, M %lld): rel rms err vs x . W^T %.4f (int8 g128 activations)\n",
               st ? "stacked gate_up" : "plain", (long long)N, (long long)K, (long long)M, sqrt(se / sr));
        free(W); free(x); free(aq); free(sa); free(w8); free(y); free_w(&w);
    }
}

/* ================================================================ F: the arena */
typedef struct { const char* name; int64_t N, K; int stacked; } Shape;
static const Shape TRUNK[] = {
    { "gate_up", 34816, 5120, 1 }, { "down", 5120, 17408, 0 }, { "gdn_qkv", 10240, 5120, 0 }, { "gdn_z", 6144, 5120, 0 },
    { "attn_qg", 12288, 5120, 0 }, { "attn_k", 1024, 5120, 0 }, { "attn_v", 1024, 5120, 0 }, { "attn_o", 5120, 6144, 0 },
    { "ssm_out", 5120, 6144, 0 },
};
#define N_TRUNK (int)(sizeof TRUNK / sizeof TRUNK[0])

static RadParam PBUF[6];
static char PRATE[16], PCB[16], PACT[16];
static const RadParam* params(int64_t M, int64_t N, int64_t K, const char* rate, const char* cb, const char* act) {
    snprintf(PRATE, sizeof PRATE, "%s", rate); snprintf(PCB, sizeof PCB, "%s", cb); snprintf(PACT, sizeof PACT, "%s", act);
    PBUF[0] = RAD_INT("M", M); PBUF[1] = RAD_INT("N", N); PBUF[2] = RAD_INT("K", K);
    PBUF[3] = RAD_STR("rate", PRATE); PBUF[4] = RAD_STR("codebook", PCB); PBUF[5] = RAD_STR("act", PACT);
    return PBUF;
}

static void test_plan(void) {
    const int64_t Ms[] = { 1, 511, 512, 513, 2048, 4096 };
    int s, i, bad = 0;
    int64_t worst = 0;
    printf("F arena (a8g128, one prefill step; MB = 2^20 B):\n  %-8s %6s %5s  %9s %9s %9s %7s %9s\n", "shape", "M", "mats",
           "aq MB", "sa MB", "plane MB", "chunks", "total MB");
    for (s = 0; s < N_TRUNK; ++s) {
        int64_t prev_total = -1;
        for (i = 0; i < 6; ++i) {
            const Shape* sh = &TRUNK[s];
            const int64_t M = Ms[i];
            Exl3iA8Plan p;
            RadArgs a;
            int64_t hook;
            const int rc = exl3i_a8_plan(M, sh->N, sh->K, sh->stacked, 128, EXL3I_A8_CHUNK_BYTES, EXL3I_A8_MIN_ROWS, &p);
            char msg[128];
            snprintf(msg, sizeof msg, "F %s M %lld", sh->name, (long long)M);
            if (rc) { check(0, msg); continue; }
            if (M >= EXL3I_A8_MIN_ROWS) {
                bad += p.aq_off % 256 || p.sa_off % 256 || p.w_off % 256;
                bad += p.aq_off + p.aq_bytes > p.sa_off || p.sa_off + p.sa_bytes > p.w_off || p.w_off + p.w_bytes > p.a8_bytes;
                bad += p.aq_bytes < p.mats * M * sh->K || p.sa_bytes < p.mats * M * (sh->K / 128) * 4 || p.w_bytes < p.chunk * sh->K;
                bad += p.chunk % 128 || p.chunk > sh->N || p.chunk * sh->K > EXL3I_A8_CHUNK_BYTES || p.mats != (sh->stacked ? 2 : 1);
                bad += p.a8_bytes >= p.f16_bytes ? p.total != p.a8_bytes : p.total != p.f16_bytes;
                bad += p.f16_bytes != exl3i_a8_align((int64_t)(EXL3I_A8_MIN_ROWS - 1) * 2 * sh->K * 2);
            } else {
                bad += p.a8_bytes != 0 || p.total != p.f16_bytes || p.f16_bytes != exl3i_a8_align(M * 2 * sh->K * 2);
            }
            if (prev_total > p.total) ++bad;   /* never shrinks with M */
            prev_total = p.total;
            /* the scratch hook, as the engine calls it: params only; covers both halves wherever N % 256 */
            memset(&a, 0, sizeof a);
            a.p = params(M, sh->N, sh->K, "4", "mad", "a8g128");
            a.n_p = 6;
            hook = exl3i_nt_scratch(&a);
            {
                Exl3iA8Plan q;
                exl3i_a8_plan(M, sh->N, sh->K, sh->N % 256 == 0, 128, EXL3I_A8_CHUNK_BYTES, EXL3I_A8_MIN_ROWS, &q);
                /* at M <= 16 the decode form's K split (KS) adds its partial sums, at most 32 slices of M x N f32 */
                bad += M > 16 ? hook != q.total || hook < p.total
                              : hook < q.total || hook > q.total + 32LL * M * sh->N * 4 || hook < p.total;
            }
            if (hook > worst) worst = hook;
            if (M == 512 || M == 2048 || M == 4096)
                printf("  %-8s %6lld %5d  %9.2f %9.2f %9.2f %7lld %9.2f%s\n", sh->name, (long long)M, p.mats,
                       p.aq_bytes / 1048576.0, p.sa_bytes / 1048576.0, p.w_bytes / 1048576.0,
                       (long long)((sh->N + p.chunk - 1) / p.chunk), p.total / 1048576.0,
                       hook != p.total ? "  (hook: both halves)" : "");
        }
    }
    {   /* the head's shape (f16 in serving; the plan still chunks it): 248320 x 5120 -> 192 MB chunks */
        Exl3iA8Plan p;
        exl3i_a8_plan(2048, 248320, 5120, 0, 128, EXL3I_A8_CHUNK_BYTES, EXL3I_A8_MIN_ROWS, &p);
        bad += p.chunk != 39296 || (248320 + p.chunk - 1) / p.chunk != 7;
        printf("  head-shaped [248320, 5120] at M 2048: chunk %lld columns, %lld chunks, plane %.1f MB\n", (long long)p.chunk,
               (long long)((248320 + p.chunk - 1) / p.chunk), p.w_bytes / 1048576.0);
    }
    {   /* the f16 row's answer is unchanged (the region + T2's chunk) */
        RadArgs a;
        memset(&a, 0, sizeof a);
        a.p = params(2048, 5120, 17408, "4", "mad", "f16");
        a.n_p = 6;
        bad += exl3i_nt_scratch(&a) != (2048LL * 2 * 17408 * 2 + 255) / 256 * 256 + EXL3R4D_T2_CHUNK_BYTES;
    }
    check(bad == 0, "F: the arena plan");
    printf("F arena: %d trunk shapes x M {1, 511, 512, 513, 2048, 4096}, %d wrong; the largest scratch answer %.1f MB\n",
           N_TRUNK, bad, worst / 1048576.0);
}

/* ================================================================ G / H: dispatch and refusals through the rows */
/* 256-B aligned and left untouched: the stubs never read them, so a 570 MB y costs no memory */
static void* amalloc(size_t n) { void* p = 0; if (posix_memalign(&p, 256, n ? n : 256)) return 0; return p; }

typedef struct {
    RadTensor t[6];
    void *x, *rec, *suh, *suh1, *svh, *y, *scratch;
} Ops;

/* host buffers shaped as the row wants them (the stubs never read them) */
static void make_ops(Ops* o, int64_t T, int64_t N, int64_t K, int m, int stacked, uint32_t ydt) {
    int i;
    memset(o, 0, sizeof *o);
    o->x = amalloc((size_t)(T * K * 2));
    o->rec = amalloc((size_t)(N / 16 * exl3i_row_words(K, m) * 4));
    o->suh = amalloc((size_t)K * 2); o->suh1 = amalloc((size_t)K * 2); o->svh = amalloc((size_t)N * 2);
    o->y = amalloc((size_t)(T * N * 4));
    for (i = 0; i < 6; ++i) { o->t[i].rank = 2; o->t[i].stride[1] = 1; }
    o->t[0].data = o->x; o->t[0].dtype = RAD_BF16; o->t[0].shape[0] = T; o->t[0].shape[1] = K; o->t[0].stride[0] = K;
    o->t[1].data = o->rec; o->t[1].dtype = EXL3I_DT_TILES; o->t[1].shape[0] = N / 16; o->t[1].shape[1] = exl3i_row_words(K, m);
    o->t[1].stride[0] = exl3i_row_words(K, m);
    o->t[2].data = o->suh; o->t[2].dtype = RAD_F16; o->t[2].shape[0] = 1; o->t[2].shape[1] = K; o->t[2].stride[0] = K;
    o->t[3].data = o->svh; o->t[3].dtype = RAD_F16; o->t[3].shape[0] = N; o->t[3].shape[1] = 1; o->t[3].stride[0] = 1;
    o->t[4] = o->t[2]; o->t[4].data = stacked ? o->suh1 : 0;
    o->t[5].data = o->y; o->t[5].dtype = ydt; o->t[5].shape[0] = T; o->t[5].shape[1] = N; o->t[5].stride[0] = N;
}
static void free_ops(Ops* o) { free(o->x); free(o->rec); free(o->suh); free(o->suh1); free(o->svh); free(o->y); free(o->scratch); }

/* One launch through row `row`: declared rows Md, real rows T. The scratch is the hook's answer at Md (or `sb`). */
static int run(const RadKernelInfo* row, int64_t Md, int64_t T, int64_t N, int64_t K, const char* rate, int m,
               const char* cb, const char* act, int stacked, uint32_t ydt, int64_t sb, Ops* o) {
    RadArgs a;
    int64_t need;
    make_ops(o, T, N, K, m, stacked, ydt);
    memset(&a, 0, sizeof a);
    a.p = params(Md, N, K, rate, cb, act);
    a.n_p = 6;
    need = row->scratch(&a);
    if (sb < 0) sb = need;
    if (sb < 0) return (int)RAD_E_SCRATCH - 1000;   /* the hook refused (declare-time) */
    o->scratch = amalloc((size_t)sb);
    a.t = o->t; a.n_t = 6; a.scratch = o->scratch; a.scratch_bytes = sb; a.world_size = 1;
    reset_calls();
    return row->launch(&a, 0);
}

static const RadKernelInfo* row_named(const char* name) {
    int i;
    for (i = 0; i < rad_kernel_count(); ++i) if (!strcmp(rad_kernel_at(i)->name, name)) return rad_kernel_at(i);
    return 0;
}

static void test_dispatch(void) {
    const RadKernelInfo* rf = row_named("exl3i_gemm_nt");
    const RadKernelInfo* ra = row_named("exl3i_gemm_nt_a8");
    Ops o;
    int rc, s, bad = 0, n = 0;
    char msg[160];
    check(rf && ra, "G: both rows present");
    if (!rf || !ra) return;
    check(rf->launch == ra->launch && rf->layout == ra->layout && rf->relayout == ra->relayout &&
          rf->unrelayout == ra->unrelayout, "G: one launch and one stored form for both rows");
    /* f16: T2 at 2048 rows (raw set), nt_pass at 511 */
    rc = run(rf, 2048, 2048, 5120, 17408, "4", 64, "mad", "f16", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_nt[3] == 1 && n_actq == 0 && last_nt.g.raw != 0, "G: f16 at 2048 rows: the f16 kernel with T2's chunk");
    free_ops(&o);
    rc = run(rf, 2048, 511, 5120, 17408, "4", 64, "mad", "f16", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_nt[3] == 1 && last_nt.g.raw == 0, "G: f16 at 511 rows: nt_pass");
    free_ops(&o);
    /* a8g128, every trunk shape, declared 4096: real rows 1 / 511 the f16 kernel, 512 / 513 / 2048 / 4096 the A8 path */
    for (s = 0; s < N_TRUNK; ++s) {
        const Shape* sh = &TRUNK[s];
        const int64_t Ts[] = { 1, 511, 512, 513, 2048, 4096 };
        int i;
        for (i = 0; i < 6; ++i, ++n) {
            const int64_t T = Ts[i];
            Exl3iA8Plan p;
            int ok;
            rc = run(ra, 4096, T, sh->N, sh->K, "2.5", 40, "mad", "a8g128", sh->stacked, i % 3 == 0 ? RAD_F32 : i % 3 == 1 ? RAD_F16 : RAD_BF16, -1, &o);
            exl3i_a8_plan(T, sh->N, sh->K, sh->stacked, 128, EXL3I_A8_CHUNK_BYTES, EXL3I_A8_MIN_ROWS, &p);
            if (T < EXL3I_A8_MIN_ROWS) {
                ok = rc == RAD_OK && n_nt[3] == 1 && n_actq == 0 && n_wdec == 0 && n_gemm == 0 && last_nt.g.raw == 0 &&
                     last_nt.g.T == T && last_nt.m == 40;
            } else {
                const Exl3iA8Args* q = &last_actq;
                const char* base = (const char*)o.scratch;
                ok = rc == RAD_OK && n_nt[3] == 0 && n_actq == 1 && n_wdec == n_gemm &&
                     n_wdec == (int)((sh->N + p.chunk - 1) / p.chunk);
                ok = ok && q->M == T && q->N == sh->N && q->K == sh->K && q->m == 40 && q->group == 128 &&
                     q->mats == (sh->stacked ? 2 : 1) && (q->suh1 != 0) == sh->stacked && q->x_bf16 == 1 &&
                     q->y_dt == (i % 3 == 0 ? 1 : i % 3 == 1 ? 2 : 0) && q->x == o.x && q->y == o.y && q->codes == o.rec &&
                     q->svh == o.svh && (const char*)q->aq == base + p.aq_off && (const char*)q->sa == base + p.sa_off &&
                     (const char*)q->wq == base + p.w_off && q->x_pitch == sh->K && q->y_pitch == sh->N;
                if (ok) {
                    int c;
                    int64_t c0 = 0;
                    for (c = 0; c < n_wdec && c < 64; ++c) {
                        ok = ok && wdec_log[c].c0 == c0 && gemm_log[c].c0 == c0 && wdec_log[c].nc == gemm_log[c].nc &&
                             wdec_log[c].nc == (sh->N - c0 < p.chunk ? sh->N - c0 : p.chunk);
                        c0 += wdec_log[c].nc;
                    }
                    ok = ok && c0 == sh->N;
                }
            }
            if (!ok) {
                snprintf(msg, sizeof msg, "G a8g128 %s T %lld (declared 4096): rc %d nt %d actq %d wdec %d gemm %d", sh->name,
                         (long long)T, rc, n_nt[3], n_actq, n_wdec, n_gemm);
                check(0, msg);
                ++bad;
            }
            free_ops(&o);
        }
    }
    check(bad == 0, "G: a8g128 dispatch at every trunk shape");
    /* a8t: the group is K; a8g1024 on gate_up */
    rc = run(ra, 2048, 2048, 5120, 17408, "3", 48, "mad", "a8t", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_actq == 1 && last_actq.group == 17408, "G: a8t: one scale a row (group K)");
    free_ops(&o);
    rc = run(ra, 2048, 2048, 34816, 5120, "3", 48, "mad", "a8g1024", 1, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_actq == 1 && last_actq.group == 1024 && last_actq.mats == 2, "G: a8g1024 stacked");
    free_ops(&o);
    /* several chunks (the budget knob): down's plane in 4 MB chunks = 240 columns -> 128-column chunks... */
    exl3i_test_pin(EXL3I_PIN_A8_CHUNK_MB, 3);   /* 3 MB / 5120 B a column = 614 -> 512 columns a chunk */
    rc = run(ra, 1024, 1024, 5120, 5120, "4", 64, "mad", "a8g128", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_wdec == 10 && n_gemm == 10 && wdec_log[9].c0 == 4608 && wdec_log[9].nc == 512,
          "G: a 3 MB chunk budget: 10 chunks of 512 columns");
    free_ops(&o);
    rc = run(ra, 1024, 1024, 5120 + 128 * 3, 5120, "4", 64, "mad", "a8g128", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_wdec == 11 && wdec_log[10].c0 == 5120 && wdec_log[10].nc == 384, "G: a ragged last chunk");
    free_ops(&o);
    exl3i_test_pin(EXL3I_PIN_A8_CHUNK_MB, 0);
    /* the threshold knob: A8 from 16 rows (the bench's M 16-511 sweep) */
    exl3i_test_pin(EXL3I_PIN_A8_MIN_ROWS, 16);
    rc = run(ra, 256, 16, 6144, 5120, "4", 64, "mad", "a8g128", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_actq == 1 && n_gemm == 1, "G: pin A8_MIN_ROWS 16: A8 at 16 rows");
    free_ops(&o);
    rc = run(ra, 256, 15, 6144, 5120, "4", 64, "mad", "a8g128", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_actq == 0 && n_nt[3] == 1, "G: pin A8_MIN_ROWS 16: f16 at 15 rows");
    free_ops(&o);
    exl3i_test_pin(EXL3I_PIN_A8_MIN_ROWS, 0);
    printf("G dispatch: %d trunk launches (9 shapes x real rows 1 / 511 / 512 / 513 / 2048 / 4096, declared 4096), %d "
           "wrong; a8t, a8g1024, multi-chunk, the threshold knob\n", n, bad);
}

/* the engine's constraint match, for the rows' lists (RAD_C_IN: the value one of the space-separated words) */
static int in_set(const char* set, const char* v) {
    const size_t n = strlen(v);
    const char* p = set;
    while (*p) {
        const char* e = strchr(p, ' ');
        const size_t l = e ? (size_t)(e - p) : strlen(p);
        if (l == n && !strncmp(p, v, n)) return 1;
        p += l + (e ? 1 : 0);
    }
    return 0;
}
static int row_takes(const RadKernelInfo* r, const char* cb, const char* act) {
    int i;
    for (i = 0; i < r->n_constraints; ++i) {
        const RadConstraint* c = &r->constraints[i];
        if (c->op != RAD_C_IN) continue;
        if (!strcmp(c->key, "codebook") && !in_set(c->sval, cb)) return 0;
        if (!strcmp(c->key, "act") && !in_set(c->sval, act)) return 0;
    }
    return 1;
}

static void test_refusals(void) {
    const RadKernelInfo* rf = row_named("exl3i_gemm_nt");
    const RadKernelInfo* ra = row_named("exl3i_gemm_nt_a8");
    static const char* const bad_act[] = { "a8g0128", "A8T", "a8g64", "a8g100", "a8", "a8g", "a8t ", "a8g128x", "f32", "" };
    static const char* const good_act[] = { "f16", "a8t", "a8g128", "a8g256", "a8g512", "a8g1024" };
    const int n0 = CHECKS;
    Ops o;
    int i, rc;
    Exl3Act am;
    if (!rf || !ra) return;
    for (i = 0; i < (int)(sizeof bad_act / sizeof bad_act[0]); ++i) {
        char msg[64];
        snprintf(msg, sizeof msg, "H: act '%s' refused by the parse", bad_act[i]);
        check(exl3_act_parse(bad_act[i], &am) != 0, msg);
    }
    for (i = 0; i < (int)(sizeof good_act / sizeof good_act[0]); ++i) {
        const int f = !strcmp(good_act[i], "f16");
        char msg[96];
        snprintf(msg, sizeof msg, "H: act %s: taken by %s (mad)", good_act[i], f ? "the f16 row" : "the A8 row");
        check(exl3_act_parse(good_act[i], &am) == 0 && row_takes(rf, "mad", good_act[i]) == f && row_takes(ra, "mad", good_act[i]) == !f, msg);
    }
    check(!row_takes(rf, "mad", "a8g2048") && !row_takes(ra, "mad", "a8g2048"), "H: a8g2048 (parses) has no row");
    /* a codebook other than mad: no dense or A8 row takes it */
    check(!row_takes(rf, "other", "f16") && !row_takes(ra, "other", "a8g128") && !row_takes(ra, "other", "a8t") &&
          row_takes(rf, "mad", "f16"),
          "H: no dense / A8 row takes a codebook other than mad");
    /* A8 on a codebook other than mad: refused by the scratch hook (declare) and at launch (the parse names no codebook) */
    rc = run(ra, 2048, 2048, 5120, 5120, "4", 64, "other", "a8g128", 0, RAD_BF16, 1 << 20, &o);
    check(rc == RAD_E_DTYPE && n_actq + n_nt[3] == 0, "H: A8 on a codebook other than mad refused at launch");
    free_ops(&o);
    {
        RadArgs a;
        memset(&a, 0, sizeof a);
        a.p = params(2048, 5120, 5120, "4", "other", "a8g128");
        a.n_p = 6;
        check(ra->scratch(&a) < 0, "H: A8 on a codebook other than mad refused by the scratch hook");
        a.p = params(2048, 5120, 1536, "4", "mad", "a8g1024");
        check(ra->scratch(&a) < 0, "H: a8g1024 with K 1536 refused by the scratch hook");
    }
    rc = run(ra, 2048, 600, 5120, 1536, "4", 64, "mad", "a8g1024", 0, RAD_BF16, 64 << 20, &o);
    check(rc == RAD_E_SHAPE && n_actq == 0, "H: a8g1024 with K 1536 refused at launch");
    free_ops(&o);
    /* more real rows than the scratch was sized for: refused, never a silent f16 (4096 rows in a band sized at 1024 for
     * both halves' planes) */
    rc = run(ra, 1024, 4096, 5120, 5120, "4", 64, "mad", "a8g128", 0, RAD_BF16, -1, &o);
    check(rc == RAD_E_SCRATCH && n_actq + n_nt[3] == 0, "H: rows past the scratch's refused");
    free_ops(&o);
    /* a declared band under the threshold has no A8 arena; a launch of 512 rows in it is refused */
    rc = run(ra, 511, 512, 5120, 5120, "4", 64, "mad", "a8g128", 0, RAD_BF16, -1, &o);
    check(rc == RAD_E_SCRATCH, "H: 512 rows in a band declared at 511 refused");
    free_ops(&o);
    /* the stacked seam with N % 256 */
    rc = run(ra, 2048, 2048, 384, 5120, "4", 64, "mad", "a8g128", 1, RAD_BF16, -1, &o);
    check(rc == RAD_E_SHAPE && n_actq == 0, "H: suh.1 with N % 256 refused");
    free_ops(&o);
    /* encodings the import must refuse (the layout hooks read them through exl3i_encoding_read) */
    {
        RadEncoding e;
        exl3i_encoding_make(40, EXL3I_CB_MAD, 0, 256, 256, &e);
        check(exl3i_encoding_read(&e, 0, 0, 0, 0, 0) == RAD_OK, "H: a good encoding reads");
        rad_enc_copy_str(e.transform, "mad/h128/k2.3");
        check(exl3i_encoding_read(&e, 0, 0, 0, 0, 0) == RAD_E_FORMAT, "H: rate 2.3 refused");
        rad_enc_copy_str(e.transform, "mad/h64/k2.5");
        check(exl3i_encoding_read(&e, 0, 0, 0, 0, 0) == RAD_E_FORMAT, "H: h64 refused");
        rad_enc_copy_str(e.transform, "other/h128/k2.5");
        check(exl3i_encoding_read(&e, 0, 0, 0, 0, 0) == RAD_E_FORMAT, "H: a codebook other than mad refused");
        rad_enc_copy_str(e.transform, "other/h128/k4/r16");
        check(exl3i_encoding_read(&e, 0, 0, 0, 0, 0) == RAD_E_FORMAT, "H: a codebook other than mad refused with the r16 suffix");
        rad_enc_copy_str(e.transform, "mad/h128/k2.5");
        check(exl3i_encoding_read(&e, 0, 0, 0, 0, 0) == RAD_OK, "H: a mad encoding reads");
        rad_enc_copy_str(e.transform, "mad/h128/k2.25");
        check(exl3i_encoding_read(&e, 0, 0, 0, 0, 0) == RAD_E_FORMAT, "H: a rate that disagrees with the record's width refused");
        {   /* version 3: the R16 row order */
            int r16 = -1, mm = 0;
            check(exl3i_encoding_make_r16(64, EXL3I_CB_MAD, 0, 248320, 5120, &e) == RAD_OK &&
                  !strcmp(e.transform, "mad/h128/k4/r16"), "H: an R16 encoding is made (transform .../r16)");
            check(exl3i_encoding_read_r16(&e, &mm, 0, 0, 0, 0, &r16) == RAD_OK && r16 == 1 && mm == 64, "H: an R16 encoding reads, r16 1");
            check(exl3i_encoding_read(&e, 0, 0, 0, 0, 0) == RAD_OK, "H: exl3i_encoding_read takes an R16 encoding too");
            check(exl3i_encoding_make_r16(64, EXL3I_CB_MAD, 0, 248320 - 16, 5120, &e) == RAD_E_INVAL, "H: R16 with rows not in 128s refused");
            check(exl3i_encoding_make_r16(64, EXL3I_CB_MAD, 1, 34816, 5120, &e) == RAD_E_INVAL, "H: R16 stacked refused");
            check(exl3i_encoding_make_r16(38, EXL3I_CB_MAD, 0, 256, 256, &e) == RAD_OK && exl3i_encoding_read_r16(&e, 0, 0, 0, 0, 0, &r16) == RAD_OK &&
                  r16 == 1, "H: R16 at K2.375 (the longest transform) fits");
            exl3i_encoding_make(64, EXL3I_CB_MAD, 0, 256, 256, &e);
            check(exl3i_encoding_read_r16(&e, 0, 0, 0, 0, 0, &r16) == RAD_OK && r16 == 0, "H: a natural encoding reads r16 0");
            strcpy(e.transform, "mad/h128/k4/r8");
            check(exl3i_encoding_read(&e, 0, 0, 0, 0, 0) == RAD_E_FORMAT, "H: an unknown row order refused");
            {   /* the closed form is a permutation of 0 .. N - 1 */
                static unsigned char seen[248320];
                int okp = 1;
                memset(seen, 0, sizeof seen);
                for (int64_t s = 0; s < 248320; ++s) { const int64_t v = exl3i_r16_row(s, 248320); if (v < 0 || v >= 248320 || seen[v]++) okp = 0; }
                check(okp && exl3i_r16_row(16, 248320) == 31040 && exl3i_r16_row(128, 248320) == 16, "H: exl3i_r16_row is a permutation (16 -> 31040, 128 -> 16)");
            }
        }
    }
    printf("H refusals: %d checks\n", CHECKS - n0);
}

/* ================================================================ L: AQ (exl3i_abi.h version 3) */
/* The dense op given aq + aq_plane / aq_planes takes the A8 path without its own actq, the GEMM pointed at its planes in
 * the producer's packed rows; without aq_planes it is as before; below 512 rows or on another group with aq it refuses
 * (`a` may be stale). exl3i_actq checks its operands, its pre stages' too, and hands the device its rows. */
static void test_aq(void) {
    const RadKernelInfo* ra = row_named("exl3i_gemm_nt_a8");
    const RadKernelInfo* rq = row_named("exl3i_actq");
    int bad = 0, n = 0;
    check(ra && rq, "L: rows exl3i_gemm_nt_a8 and exl3i_actq");
    if (!ra || !rq) return;
    {   /* gate_up (stacked) on planes 1, 2 of 3, K 5120, 2048 rows */
        const int64_t T = 2048, N = 34816, K = 5120, np = 3, rb = exl3i_aq_row_bytes(K, (int)np);
        Ops o;
        RadParam pp[8];
        RadTensor t7[7];
        RadArgs a;
        void* aq = amalloc((size_t)(T * rb));
        make_ops(&o, T, N, K, 64, 1, RAD_BF16);
        memcpy(pp, params(2048, N, K, "4", "mad", "a8g128"), 6 * sizeof(RadParam));
        pp[6] = (RadParam)RAD_INT("aq_plane", 1);
        pp[7] = (RadParam)RAD_INT("aq_planes", np);
        memcpy(t7, o.t, 6 * sizeof(RadTensor));
        memset(&t7[6], 0, sizeof t7[6]);
        t7[6].data = aq; t7[6].dtype = RAD_U8; t7[6].rank = 2; t7[6].shape[0] = T; t7[6].shape[1] = rb; t7[6].stride[0] = rb;
        t7[6].stride[1] = 1;
        memset(&a, 0, sizeof a);
        a.p = pp; a.n_p = 8; a.t = t7; a.n_t = 7; a.world_size = 1;
        a.scratch_bytes = ra->scratch(&a);
        o.scratch = amalloc((size_t)a.scratch_bytes); a.scratch = o.scratch;
        reset_calls(); n_aq = 0;
        bad += ra->launch(&a, 0) != RAD_OK;
        bad += n_actq != 0 || n_gemm < 1;
        bad += (char*)gemm_log[0].aq != (char*)aq + 1 * K || (char*)gemm_log[0].sa != (char*)aq + np * K + 4 * (1 * (K / 128));
        bad += gemm_log[0].a_ld != rb || gemm_log[0].a_ms != K || gemm_log[0].sa_ld != rb / 4 || gemm_log[0].sa_ms != K / 128;
        ++n;
        /* aq_planes 2 cannot hold planes 1, 2: refused */
        pp[7] = (RadParam)RAD_INT("aq_planes", 2);
        bad += ra->launch(&a, 0) == RAD_OK; ++n;
        /* no aq_planes: aq is not read, the op quantises itself */
        a.n_p = 6; reset_calls();
        bad += ra->launch(&a, 0) != RAD_OK || n_actq != 1 || gemm_log[0].a_ld != 0; ++n;
        /* 511 rows: below the A8 threshold, aq given: refused, nothing launched */
        a.n_p = 8; pp[7] = (RadParam)RAD_INT("aq_planes", np);
        t7[0].shape[0] = 511; t7[5].shape[0] = 511; t7[6].shape[0] = 511; reset_calls();
        bad += ra->launch(&a, 0) != RAD_E_SHAPE || n_actq != 0 || n_gemm != 0 || n_nt[3] != 0; ++n;
        /* ... and without aq there the f16 path as before */
        t7[6].data = 0; reset_calls();
        bad += ra->launch(&a, 0) != RAD_OK || n_actq != 0 || n_gemm != 0 || n_nt[3] != 1; ++n;
        free(aq); free_ops(&o);
    }
    {   /* a8g256 with aq: refused (AQ's planes are a8g128) */
        const int64_t T = 1024, N = 5120, K = 17408, rb = exl3i_aq_row_bytes(K, 1);
        Ops o;
        RadParam pp[8];
        RadTensor t7[7];
        RadArgs a;
        void* aq = amalloc((size_t)(T * rb));
        make_ops(&o, T, N, K, 64, 0, RAD_BF16);
        memcpy(pp, params(1024, N, K, "4", "mad", "a8g256"), 6 * sizeof(RadParam));
        pp[6] = (RadParam)RAD_INT("aq_plane", 0);
        pp[7] = (RadParam)RAD_INT("aq_planes", 1);
        memcpy(t7, o.t, 6 * sizeof(RadTensor));
        memset(&t7[6], 0, sizeof t7[6]);
        t7[6].data = aq; t7[6].dtype = RAD_U8; t7[6].rank = 2; t7[6].shape[0] = T; t7[6].shape[1] = rb; t7[6].stride[0] = rb;
        memset(&a, 0, sizeof a);
        a.p = pp; a.n_p = 8; a.t = t7; a.n_t = 7; a.world_size = 1;
        a.scratch_bytes = ra->scratch(&a);
        o.scratch = amalloc((size_t)a.scratch_bytes); a.scratch = o.scratch;
        bad += ra->launch(&a, 0) == RAD_OK; ++n;
        free(aq); free_ops(&o);
    }
    {   /* exl3i_actq: 3 planes of K 5120 over 2048 bf16 rows */
        const int64_t T = 2048, K = 5120, rb = exl3i_aq_row_bytes(K, 3);
        void* x = amalloc((size_t)(T * K * 2));
        void* sv[3] = { amalloc((size_t)K * 2), amalloc((size_t)K * 2), amalloc((size_t)K * 2) };
        void* out = amalloc((size_t)(T * rb));
        RadTensor t[EXL3I_AQ_N_OPD];
        RadParam pq[7] = { RAD_INT("M", 2048), RAD_INT("K", K), RAD_INT("planes", 3), RAD_STR("pre", "none"),
                           { "eps", RAD_P_F64, 0, 0, 0, 1e-6 }, { "wadd", RAD_P_F64, 0, 0, 0, 1.0 }, RAD_INT("res", 1) };
        void* gain = amalloc((size_t)K * 4);
        void* hb = amalloc((size_t)(T * K * 2));
        void* gu = amalloc((size_t)(T * 2 * K * 2));
        RadArgs a;
        int i;
        memset(t, 0, sizeof t);
        t[0].data = x; t[0].dtype = RAD_BF16; t[0].rank = 2; t[0].shape[0] = T; t[0].shape[1] = K; t[0].stride[0] = K; t[0].stride[1] = 1;
        for (i = 0; i < 3; ++i) {
            t[1 + i].data = sv[i]; t[1 + i].dtype = RAD_F16; t[1 + i].rank = 2; t[1 + i].shape[0] = 1; t[1 + i].shape[1] = K;
            t[1 + i].stride[0] = K; t[1 + i].stride[1] = 1;
        }
        t[5].data = out; t[5].dtype = RAD_U8; t[5].rank = 2; t[5].shape[0] = T; t[5].shape[1] = rb; t[5].stride[0] = rb; t[5].stride[1] = 1;
        memset(&a, 0, sizeof a);
        a.p = pq; a.n_p = 3; a.t = t; a.n_t = EXL3I_AQ_N_OPD; a.world_size = 1;
        n_aq = 0;
        bad += rq->launch(&a, 0) != RAD_OK || n_aq != 1 || last_aq.planes != 3 || last_aq.out_ld != rb || last_aq.M != T ||
               last_aq.K != K || last_aq.s[2] != sv[2] || !last_aq.x_bf16; ++n;
        t[3].data = 0;   /* a missing third scale vector: refused */
        bad += rq->launch(&a, 0) == RAD_OK; ++n;
        t[3].data = sv[2]; t[5].stride[0] = rb - 4;   /* rows too short: refused */
        bad += rq->launch(&a, 0) == RAD_OK; ++n;
        t[5].stride[0] = rb;
        /* pre rmsnorm: the gain (f32), eps, wadd and h through; without g refused; an f16 x refused */
        a.n_t = EXL3I_AQ_N_OPD;
        pq[3] = (RadParam)RAD_STR("pre", "rmsnorm"); a.n_p = 6;
        t[7].data = gain; t[7].dtype = RAD_F32; t[7].rank = 1; t[7].shape[0] = K; t[7].stride[0] = 1;
        t[8] = t[0]; t[8].data = hb;
        n_aq = 0;
        bad += rq->launch(&a, 0) != RAD_OK || n_aq != 1 || last_aq.pre != EXL3I_AQ_PRE_RMSNORM || last_aq.g != gain ||
               !last_aq.g_f32 || last_aq.h != hb || last_aq.eps != 1e-6f || last_aq.wadd != 1.0f || last_aq.h_pitch != K; ++n;
        t[7].data = 0;
        bad += rq->launch(&a, 0) != RAD_E_INVAL; ++n;
        t[7].data = gain; t[0].dtype = RAD_F16;
        bad += rq->launch(&a, 0) != RAD_E_DTYPE; ++n;
        t[0].dtype = RAD_BF16;
        /* pre silu_mul: x [M, 2K] wanted; [M, K] refused */
        pq[3] = (RadParam)RAD_STR("pre", "silu_mul");
        bad += rq->launch(&a, 0) != RAD_E_SHAPE; ++n;
        t[0].data = gu; t[0].shape[1] = 2 * K; t[0].stride[0] = 2 * K; n_aq = 0;
        bad += rq->launch(&a, 0) != RAD_OK || n_aq != 1 || last_aq.pre != EXL3I_AQ_PRE_SILU_MUL || last_aq.x_pitch != 2 * K; ++n;
        /* pre mul: x2 wanted */
        t[0].data = x; t[0].shape[1] = K; t[0].stride[0] = K;
        pq[3] = (RadParam)RAD_STR("pre", "mul");
        bad += rq->launch(&a, 0) != RAD_E_INVAL; ++n;
        t[6] = t[0]; t[6].data = hb; t[8].data = 0; n_aq = 0;
        bad += rq->launch(&a, 0) != RAD_OK || n_aq != 1 || last_aq.x2 != hb || last_aq.h != 0; ++n;
        /* an unknown pre: refused at the schema */
        pq[3] = (RadParam)RAD_STR("pre", "gelu");
        bad += rq->launch(&a, 0) != RAD_E_SCHEMA; ++n;
        /* pre none with h: refused (nothing to store) */
        pq[3] = (RadParam)RAD_STR("pre", "none"); t[8].data = hb;
        bad += rq->launch(&a, 0) != RAD_E_DTYPE; ++n;
        /* version 5, res = 1: pre rmsnorm with res and xo through (xo may be x, the residual stream in place; h may be
         * res); without xo, or at another pre, refused */
        pq[3] = (RadParam)RAD_STR("pre", "rmsnorm"); a.n_p = 7; t[6].data = 0;
        t[9] = t[0]; t[9].data = hb; t[10] = t[0];
        n_aq = 0;
        bad += rq->launch(&a, 0) != RAD_OK || n_aq != 1 || last_aq.res != hb || last_aq.xo != x || last_aq.h != hb ||
               last_aq.res_pitch != K || last_aq.xo_pitch != K || last_aq.pre != EXL3I_AQ_PRE_RMSNORM; ++n;
        t[10].data = 0;
        bad += rq->launch(&a, 0) != RAD_E_INVAL; ++n;
        t[10] = t[0]; pq[3] = (RadParam)RAD_STR("pre", "none"); t[8].data = 0;
        bad += rq->launch(&a, 0) != RAD_E_SCHEMA; ++n;
        a.n_p = 6; n_aq = 0;   /* res absent: res / xo not read */
        bad += rq->launch(&a, 0) != RAD_OK || n_aq != 1 || last_aq.res != 0 || last_aq.xo != 0; ++n;
        free(x); free(out); free(gain); free(hb); free(gu); for (i = 0; i < 3; ++i) free(sv[i]);
    }
    {   /* layout: a suh plane through exl3i_actq's hook is stored as exl3i_dense stores it; g (operand 7) as it is */
        const int64_t N = 1024, K = 5120;
        RadEncoding e;
        RadParam pd[6], pq[3] = { RAD_INT("M", 2048), RAD_INT("K", K), RAD_INT("planes", 3) };
        RadTensor pl;
        RadLayout L0, L1;
        int sel, i;
        exl3i_encoding_make(64, EXL3I_CB_MAD, 0, N, K, &e);
        memcpy(pd, params(2048, N, K, "4", "mad", "a8g128"), sizeof pd);
        sel = -1;
        for (i = 0; i < (int)e.n_planes; ++i) if (!strcmp(e.plane[i].role, "suh")) sel = i;
        memset(&pl, 0, sizeof pl);
        pl.dtype = RAD_F16; pl.rank = 2; pl.shape[0] = 1; pl.shape[1] = K; pl.stride[0] = K; pl.stride[1] = 1;
        bad += ra->layout(pd, 6, EXL3I_OPD_SUH, &e, &sel, &pl, 1, &L0) != RAD_OK;
        bad += rq->layout(pq, 3, EXL3I_AQ_S0 + 2, &e, &sel, &pl, 1, &L1) != RAD_OK;
        bad += strcmp(L0.tag, L1.tag) != 0 || L0.dtype != L1.dtype || L0.bytes != L1.bytes || L0.align != L1.align; ++n;
        bad += rq->layout(pq, 3, EXL3I_AQ_G, &e, &sel, &pl, 1, &L1) != RAD_E_UNSUPPORTED; ++n;
        pq[1] = (RadParam)RAD_INT("K", 4096);   /* a K not the weight's */
        bad += rq->layout(pq, 3, EXL3I_AQ_S0, &e, &sel, &pl, 1, &L1) == RAD_OK; ++n;
    }
    check(bad == 0, "L: AQ dispatch and exl3i_actq");
    printf("L AQ: %d cases, %d wrong (the A8 path on the producer's planes, no own actq; the refusals; exl3i_actq's rows)\n", n, bad);
}

/* ================================================================ M: mad */
static void test_mad(void) {
    const RadKernelInfo* rf = row_named("exl3i_gemm_nt");
    const RadKernelInfo* ra = row_named("exl3i_gemm_nt_a8");
    const float ws = 4.0f * h2f(0x1EEE);
    const int n0 = CHECKS;
    int s, bq = 0, bv = 0, b16 = 0, b16q = 0, b8 = 0, smin = 999, smax = -1, i, rc;
    Ops o;
    char msg[160];
    check(exl3i_codebook_from_name("mad") == EXL3I_CB_MAD && !strcmp(exl3i_codebook_name(EXL3I_CB_MAD), "mad") &&
          exl3i_codebook_from_name("mad8") < 0 && exl3i_codebook_from_name("ma") < 0, "M: the name mad");
    for (s = 0; s < 65536; ++s) {
        /* the definition, spelled out: ((s A + B) mod 2^32) & 0x3F3F3F3F, byte sum, - 126 */
        const uint32_t h = ((uint32_t)s * 0xD0474Bu + 0x2545F491u) & 0x3F3F3F3Fu;
        const int S = (int)(h & 255) + (int)(h >> 8 & 255) + (int)(h >> 16 & 255) + (int)(h >> 24), q = S - 126;
        const uint16_t v = exl3i_codebook_value((uint32_t)s, EXL3I_CB_MAD);
        if (S < smin) smin = S;
        if (S > smax) smax = S;
        bq += exl3i_codebook_q((uint32_t)s, EXL3I_CB_MAD) != q || (int)exl3i_mad_sum((uint32_t)s) != S;
        bv += v != f2h((float)q * ws) || exl3i_a8_wq(h2f(v)) != q;
        /* the device's closed forms (exl3i_rate.hip.h): fp16(1024 + S) - 1150 (fin16q), x the step (fin16), S + 2 (fin8) */
        {
            const double a = h2f(f2h(1024.0f + (float)S)), qd = h2f(f2h((float)(a - 1150.0)));
            b16q += qd != (double)q;
            b16 += f2h((float)(qd * (double)h2f(0x26EE))) != v;
            b8 += (int)(int8_t)(uint8_t)((S + 2) ^ 0x80) != q;
        }
    }
    check(bq == 0 && bv == 0, "M: mad's q and value = the definition, every state (value on the int8 grid, step 4 K_INV)");
    check(b16q == 0 && b16 == 0 && b8 == 0, "M: mad's closed forms (fin16q, fin16, fin8) = the oracle, every state");
    check(smin >= 0 && smax <= 252, "M: S in [0, 252] (|q| <= 126)");
    printf("M codebook: 65536 states, q %d / value %d / closed forms %d %d %d differ; S in [%d, %d]\n", bq, bv, b16q, b16, b8,
           smin, smax);
    {   /* encodings: version 4 */
        RadEncoding e;
        int cb = -1, mm = 0;
        check(exl3i_encoding_make(42, EXL3I_CB_MAD, 1, 34816, 5120, &e) == RAD_OK && !strcmp(e.transform, "mad/h128/k2.625") &&
              exl3i_encoding_read(&e, &mm, &cb, 0, 0, 0) == RAD_OK && cb == EXL3I_CB_MAD && mm == 42, "M: a mad encoding is made and reads");
        check(exl3i_encoding_make_r16(66, EXL3I_CB_MAD, 0, 248320, 5120, &e) == RAD_OK && !strcmp(e.transform, "mad/h128/k4.125/r16"),
              "M: a mad R16 encoding");
    }
    /* rows: mad takes the f16 row and the A8 row */
    for (i = 0; i < 6; ++i) {
        static const char* const acts[] = { "f16", "a8t", "a8g128", "a8g256", "a8g512", "a8g1024" };
        snprintf(msg, sizeof msg, "M: act %s with mad: exactly one row", acts[i]);
        check(row_takes(rf, "mad", acts[i]) + row_takes(ra, "mad", acts[i]) == 1, msg);
    }
    /* dispatch: a8g128 1 .. 8 rows the byte-sum GEMV (KS past its rows), 9 .. 511 nt_pass (mad's), 512+ the A8 path */
    {
        int bad = 0, n = 0, si;
        for (si = 0; si < N_TRUNK; ++si) {
            const Shape* sh = &TRUNK[si];
            for (i = 1; i <= 10; ++i, ++n) {
                const int64_t T = i <= 8 ? i : i == 9 ? 9 : 511;
                int ok;
                rc = run(ra, T <= 8 ? 8 : T, T, sh->N, sh->K, "2", 32, "mad", "a8g128", sh->stacked, RAD_BF16, -1, &o);
                if (T <= 2) {   /* the GEMV by itself: T 1, and T 2 below K4 (rate 2 here: past K2 only head-sized N) */
                    const int64_t at = 2 * T * exl3i_mad_row_bytes(sh->K);
                    ok = rc == RAD_OK && n_mad == 1 && n_nt[3] == 0 && n_actq == 0 && last_mad.g.T == T && last_mad.m == 32 &&
                         last_mad.cb == EXL3I_CB_MAD && last_mad.g.xr == o.scratch &&
                         (last_mad.ksn == 1 ? last_mad.part == 0 : (const char*)last_mad.part == (const char*)o.scratch + at);
                } else {
                    ok = rc == RAD_OK && n_mad == 0 && n_nt[3] == 1 && n_actq == 0 && last_nt.g.T == T;
                }
                if (!ok) {
                    snprintf(msg, sizeof msg, "M dispatch %s T %lld: rc %d mad %d nt %d", sh->name, (long long)T, rc, n_mad, n_nt[3]);
                    check(0, msg);
                    ++bad;
                }
                free_ops(&o);
            }
        }
        check(bad == 0, "M: mad a8g128 dispatch (GEMV at 1 / 2 rows below K4, nt_pass at 3 .. 511) at every trunk shape");
        printf("M dispatch: %d launches, %d wrong\n", n, bad);
    }
    rc = run(ra, 4096, 2048, 34816, 5120, "2.625", 42, "mad", "a8g128", 1, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_actq == 1 && n_gemm == 1 && gemm_log[0].cb == EXL3I_CB_MAD && n_mad == 0, "M: 2048 rows the A8 path, cb mad");
    free_ops(&o);
    rc = run(ra, 8, 1, 5120, 17408, "2.625", 42, "mad", "a8g128", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_mad == 0 && n_nt[3] == 1, "M: K2.625 trunk shape, 1 row, no producer's rows: nt_pass");
    free_ops(&o);
    rc = run(ra, 8, 2, 248320, 5120, "4", 64, "mad", "a8g128", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_mad == 0 && n_nt[3] == 1, "M: the head at K4, 2 rows: nt_pass (2 rows below K4 only)");
    free_ops(&o);
    rc = run(ra, 8, 1, 248320, 5120, "4.125", 66, "mad", "a8g128", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_mad == 1 && n_nt[3] == 0, "M: the head at K4.125, 1 row: the GEMV (a head-sized N)");
    free_ops(&o);
    exl3i_test_pin(EXL3I_PIN_MAD_GEMV_T, 8);
    rc = run(ra, 8, 8, 5120, 17408, "4", 64, "mad", "a8g128", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_mad == 1 && n_nt[3] == 0, "M: pin MAD_GEMV_T 8: the GEMV at 8 rows");
    free_ops(&o);
    exl3i_test_pin(EXL3I_PIN_MAD_GEMV_T, 0);
    rc = run(ra, 8, 1, 5120, 17408, "4", 64, "mad", "a8g256", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_mad == 0 && n_nt[3] == 1, "M: a8g256 at 1 row: nt_pass (the GEMV is a8g128's)");
    free_ops(&o);
    rc = run(rf, 8, 1, 5120, 17408, "4", 64, "mad", "f16", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_mad == 0 && n_nt[3] == 1, "M: f16 at 1 row: nt_pass");
    free_ops(&o);
    exl3i_test_pin(EXL3I_PIN_MAD_GEMV_OFF, 1);
    rc = run(ra, 8, 1, 5120, 17408, "4", 64, "mad", "a8g128", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_mad == 0 && n_nt[3] == 1, "M: pin MAD_GEMV_OFF: nt_pass");
    free_ops(&o);
    exl3i_test_pin(EXL3I_PIN_MAD_GEMV_OFF, 0);
    /* a scratch too small for the GEMV's rows: nt_pass (its region fits), never a write past the scratch */
    rc = run(ra, 1, 1, 5120, 17408, "4", 64, "mad", "a8g128", 0, RAD_BF16, 1 * 2 * 17408 * 2, &o);
    check(rc == RAD_OK && n_mad == 0 && n_nt[3] == 1, "M: no room for the GEMV's rows: nt_pass");
    free_ops(&o);
    rc = run(ra, 4096, 2048, 5120, 17408, "4", 64, "mad", "a8g128", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_mad == 0 && gemm_log[0].cb == EXL3I_CB_MAD, "M: mad unchanged (A8 path, cb mad)");
    free_ops(&o);
    rc = run(ra, 8, 1, 5120, 17408, "4", 64, "mad", "a8g128", 0, RAD_BF16, -1, &o);
    check(rc == RAD_OK && n_mad == 0 && n_nt[3] == 1, "M: mad K4 at 1 row, trunk N: nt_pass (the GEMV is off past K2 at trunk N)");
    free_ops(&o);
    printf("M mad: %d checks\n", CHECKS - n0);
}

int main(void) {
    test_codebook();
    test_i8plane();
    test_f16plane();
    test_act();
    test_a8_math();
    test_plan();
    test_dispatch();
    test_aq();
    test_refusals();
    test_mad();
    if (CHECKS == 0) { printf("EXL3I TWOPASS HOST TEST FAIL (0 checks)\n"); return 1; }
    printf("EXL3I TWOPASS HOST TEST %s (%d checks", FAILS ? "FAIL" : "PASS", CHECKS);
    if (FAILS) printf(", %d failed", FAILS);
    printf(")\n");
    return FAILS != 0;
}
