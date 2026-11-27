/* nt.c -- exl3i_dense's host side: the operand checks, the argument block (the GEMM kernels'
 * layout, plus the rate), the GSCR region's scratch, the A8 path's dispatch and arena (exl3i_a8.h), and the weight's layout
 * hooks.
 *
 * THE PATHS. act f16 (row exl3i_gemm_nt): nt_pass, T2 from EXL3R4D_T2_MIN rows. act a8t / a8g<G> (row
 * exl3i_gemm_nt_a8, codebook mad): at a launch of EXL3I_A8_MIN_ROWS real rows or more the A8 path (actq,
 * then per column chunk wdec8 and the i8a8 GEMM); below, the f16 path's nt_pass (never T2: the A8 arena is sized
 * for the A8 path), the same stored weight.
 *
 * R16 (the encoding's /r16 row order): a weight whose rows are stored in R16 order (the head) gets the record dtype
 * EXL3I_DT_TILES_R16 from the layout hook; the f16 row's launch then writes output s at row exl3i_r16_row(s, N)
 * (Exl3iGemm.r16, the epilogue's store). The A8 row refuses it, but for a mad weight at a8g128: the
 * byte-sum GEMV at 1 .. 8 rows (its epilogue is nt_pass's), the f16 path above (the A8 tile has no R16 store).
 *
 * THE STORED RECORD. The container's record plane (include/exl3i_encoding.h) is a table of 16-row groups, each its
 * cols/16 tiles in k order: tile (tk, tn) at word (tn kt + tk) W, W = m / 2 (row-sliceable). The kernel reads
 * the kernels' k-major order: tile (tk, tn) at (tk nt + tn) W. The relayout moves whole tiles (W words each), nothing
 * inside a tile changes; unrelayout moves them back. */
#include "exl3i.h"
#include "exl3i_dev.h"
#include "exl3i_a8.h"
#include "exl3i_a8_dev.h"
#include "exl3i_pin.h"
#include "rad_types.h"
#include <stdlib.h>
#include <string.h>

#ifndef EXL3I_NPW
#define EXL3I_NPW 2   /* column tiles a wave (the shipped arm) */
#endif
#ifndef EXL3I_MAD_NPW
#define EXL3I_MAD_NPW 2   /* the byte-sum GEMV's (exl3i_mad.hip.h; the same macro there) */
#endif

static int64_t flat_width(const RadTensor* t) {
    if (t->rank < 2 || t->rank > 3) return -1;
    if (t->stride[t->rank - 1] != 1) return -1;
    if (t->rank == 3 && t->stride[1] != t->shape[2]) return -1;
    const int64_t w = t->rank == 3 ? t->shape[1] * t->shape[2] : t->shape[1];
    return t->stride[0] >= w ? w : -1;
}

/* The geometry and the operands: the checks (K, N in 128s, N in 256s with suh.1, f16 / bf16 a, bf16 / f16 / f32
 * y, the alignments) plus the rate, the codebook and the act mode: f16; an A8 mode on mad (the int8 plane is
 * their grid) whose group divides K. */
static int geom(const RadArgs* a, Exl3iGeom* g, const RadTensor** x, const RadTensor** y, const RadTensor** tr,
                const RadTensor** ts, const RadTensor** ts1, const RadTensor** tv) {
    if (a->n_t < EXL3I_N_OPD_MIN) return RAD_E_INVAL;   /* aq (version 3) may be absent from an older launch */
    *x = rad_arg_in(a, EXL3I_OPD_A);
    *y = rad_arg_in(a, EXL3I_OPD_Y);
    *tr = rad_arg_in(a, EXL3I_OPD_RECORD);
    *ts = rad_arg_in(a, EXL3I_OPD_SUH);
    *ts1 = rad_arg_in(a, EXL3I_OPD_SUH1);
    *tv = rad_arg_in(a, EXL3I_OPD_SVH);
    if (!*x || !*y || !*tr || !*ts || !*tv) return RAD_E_INVAL;
    const int rc = exl3i_geom_parse(a->p, a->n_p, g);
    if (rc != RAD_OK) return rc;
    if (g->act.kind != EXL3_ACT_F16) {
        if (g->cb != EXL3I_CB_MAD) return RAD_E_UNSUPPORTED;
        if (!exl3i_a8_group(g->act.kind, g->act.group, g->K)) return RAD_E_SHAPE;
    }
    if (*ts1 && g->N % 256) return RAD_E_SHAPE;
    if ((*x)->dtype != RAD_BF16 && (*x)->dtype != RAD_F16) return RAD_E_DTYPE;
    if ((*y)->dtype != RAD_BF16 && (*y)->dtype != RAD_F32 && (*y)->dtype != RAD_F16) return RAD_E_DTYPE;
    const int64_t xw = flat_width(*x), yw = flat_width(*y);
    if (xw < 0 || yw < 0) return RAD_E_STRIDE;
    const int64_t M = (*x)->shape[0];
    if (M < 1 || (*y)->shape[0] < M) return RAD_E_SHAPE;
    if (yw != g->N || xw < g->K) return RAD_E_SHAPE;
    if (!rad_ptr_aligned((*x)->data, 8) || ((*x)->stride[0] * 2) % 8) return RAD_E_ALIGN;
    if (!rad_ptr_aligned((*y)->data, 16) || rad_dtype_bytes((*y)->dtype, (*y)->stride[0]) % 16) return RAD_E_ALIGN;
    /* The record, as the layout hook stored it. EXL3I_DT_TILES_R16 (rows in R16 order) is the f16
     * row's; a mad R16 weight also takes a8g128 (the byte-sum GEMV's epilogue stores R16 rows; the
     * A8 tile does not, so its launches past 8 rows take the f16 path). */
    const int mad_a8g128 = g->cb == EXL3I_CB_MAD && g->act.kind == EXL3_ACT_A8G &&
                           g->act.group == 128;
    if ((*tr)->dtype == EXL3I_DT_TILES_R16 &&
        ((g->act.kind != EXL3_ACT_F16 && !mad_a8g128) || *ts1 || g->N % 128))
        return RAD_E_UNSUPPORTED;
    if (((*tr)->dtype != EXL3I_DT_TILES && (*tr)->dtype != EXL3I_DT_TILES_R16) ||
        (*tr)->rank != 2 || (*tr)->shape[0] != g->N / 16 ||
        (*tr)->shape[1] != exl3i_row_words(g->K, g->m))
        return RAD_E_SHAPE;
    if (!rad_ptr_aligned((*tr)->data, 256)) return RAD_E_ALIGN;
    const RadTensor* v[3] = { *ts, *ts1, *tv };
    for (int i = 0; i < 3; ++i) {
        if (!v[i]) continue;
        const int64_t want = i == 2 ? g->N : g->K;
        if (v[i]->dtype != RAD_F16) return RAD_E_DTYPE;
        if (rad_tensor_numel(v[i]) != want) return RAD_E_SHAPE;
        if (!rad_ptr_aligned(v[i]->data, 2)) return RAD_E_ALIGN;
    }
    return RAD_OK;
}

/* KS (exl3i_dev.h): K slices across workgroups for a launch of T <= 16 rows, from the weight's shape alone (so the
 * scratch hook and the launch agree): enough slices that the grid has about EXL3I_KS_WG workgroups (N / 128 a slice;
 * 0 = off), each slice at least two 128-feature blocks, at most 32 slices. The test pin EXL3I_PIN_KSPLIT forces n
 * slices (1: off). The partial sums take ksn T N f32 past the GSCR region. */
#define EXL3I_KS_WG 512
static int ks_slices(int64_t T, int64_t N, int64_t K) {
    const int64_t force = exl3i_pins[EXL3I_PIN_KSPLIT], wg = EXL3I_KS_WG;
    if (T > 16 || T < 1) return 1;
    const int64_t nkb = K / 128, cbs = N / 128;
    int64_t s = force > 0 ? force : cbs > 0 ? (wg + cbs - 1) / cbs : 1;
    if (s > nkb / 2) s = nkb / 2;
    if (s > 32) s = 32;
    return s < 2 ? 1 : (int)s;
}
static int64_t region_bytes(int64_t T, int64_t K) { return (T * 2 * K * 2 + 255) / 256 * 256; }
/* The byte-sum GEMV (mad, a8g128, T <= EXL3I_MAD_MAX_ROWS): its rows for both matrices of a stacked weight. It does a
 * sudot4 a weight a row, so past a row or two nt_pass's WMMA (DQ, f16 rows) is faster: with the producer's rows
 * (aq, form xb) it takes any T up to 8 (their bf16 rows are stale); by itself T 1, and T 2 below K4 (measured on gfx1201, gate_up T 2:
 * K2.5 126 vs 144 us, K4.125 164 vs 159; T 3: K2.5 232 vs 150), where the pre-pass pays (mad_gemv_rows). */
#define EXL3I_MAD_MAX_ROWS 8
static int64_t mad_region_bytes(int64_t T, int64_t K) { return 2 * T * exl3i_mad_row_bytes(K); }
static int64_t ks_bytes(int64_t T, int64_t N, int64_t K) {
    const int s = ks_slices(T, N, K);
    return s > 1 ? (int64_t)s * T * N * 4 : 0;
}

/* Attach the K-split partial sums past the first `at` bytes of the arena when it holds them; else
 * leave the launch unsplit. */
static void ks_attach(Exl3iGemm* q, const RadArgs* a, int64_t T, int64_t N, int64_t K, int64_t at) {
    const int64_t kb = ks_bytes(T, N, K);
    q->ksn = 1;
    q->part = 0;
    if (kb > 0 && a->scratch && a->scratch_bytes >= at + kb) {
        q->ksn = ks_slices(T, N, K);
        q->part = (float*)((char*)a->scratch + at);
    }
}

static int fill(const RadArgs* a, Exl3iGemm* q, int* cb) {
    const RadTensor *x, *y, *tr, *ts, *ts1, *tv;
    Exl3iGeom g;
    const int rc = geom(a, &g, &x, &y, &tr, &ts, &ts1, &tv);
    if (rc != RAD_OK) return rc;
    memset(q, 0, sizeof *q);
    Exl3r4dGemm* d = &q->g;
    d->cls[0].codes = tr->data;
    d->cls[0].suh = ts->data;
    d->cls[0].suh1 = ts1 ? ts1->data : 0;
    d->cls[0].svh = tv->data;
    d->cls[0].codes_s0 = 1;
    d->cls[0].suh_s0 = 1;
    d->cls[0].suh1_s0 = ts1 ? 1 : 0;
    d->cls[0].svh_s0 = 1;
    d->cls[0].rows = (int32_t)g.N;
    d->cls[0].cols = (int32_t)g.K;
    d->cls[1] = d->cls[0];
    d->x = x->data;
    d->y = y->data;
    d->xr = a->scratch;
    {
        const int64_t xr_bytes = ((int64_t)x->shape[0] * 2 * g.K * 2 + 255) / 256 * 256;
        d->raw = x->shape[0] >= EXL3R4D_T2_MIN && a->scratch_bytes >= xr_bytes + EXL3R4D_T2_CHUNK_BYTES
                     ? (float*)((char*)a->scratch + xr_bytes) : 0;
    }
    ks_attach(q, a, x->shape[0], g.N, g.K, region_bytes(x->shape[0], g.K));
    d->xr_pitch = 2 * g.K;
    d->x_pitch = x->stride[0];
    d->y_pitch = y->stride[0];
    d->T = (int32_t)x->shape[0];
    d->n_tok = d->T;
    d->n_expert = 1;
    d->top_k = 1;
    d->n_class = 2;
    d->y_cols = (int32_t)g.N;
    d->x_bf16 = x->dtype == RAD_BF16;
    d->y_dt = y->dtype == RAD_F32 ? 1 : y->dtype == RAD_F16 ? 2 : 0;
    d->sk = 1;
    d->chunk = 16;
    d->nslot = 1;
    q->m = g.m;
    q->cb = g.cb;
    q->r16 = tr->dtype == EXL3I_DT_TILES_R16;
    *cb = g.cb;
    return RAD_OK;
}

/* The A8 threshold and chunk: EXL3I_A8_MIN_ROWS and EXL3I_A8_CHUNK_BYTES, or a test's pins (exl3i_pin.h). */
static int64_t a8_min_rows(void) {
    const int64_t p = exl3i_pins[EXL3I_PIN_A8_MIN_ROWS];
    return p > 0 ? p : EXL3I_A8_MIN_ROWS;
}
static int64_t a8_chunk_bytes(void) {
    const int64_t p = exl3i_pins[EXL3I_PIN_A8_CHUNK_MB];
    return p > 0 ? p << 20 : EXL3I_A8_CHUNK_BYTES;
}

/* The A8 path at a launch's real rows T (>= EXL3I_A8_MIN_ROWS): the arena laid out at T (no larger than at the
 * declared M, which the scratch hook sized), the rows quantised once, then each column chunk decoded and multiplied. */
static int a8_launch(const RadArgs* a, RadStream s) {
    const RadTensor *x, *y, *tr, *ts, *ts1, *tv;
    Exl3iGeom g;
    int rc = geom(a, &g, &x, &y, &tr, &ts, &ts1, &tv);
    if (rc != RAD_OK) return rc;
    const int64_t T = x->shape[0], group = exl3i_a8_group(g.act.kind, g.act.group, g.K);
    Exl3iA8Plan pl;
    if (exl3i_a8_plan(T, g.N, g.K, ts1 != 0, group, a8_chunk_bytes(), a8_min_rows(), &pl) != 0) return RAD_E_SHAPE;
    if (!a->scratch || a->scratch_bytes < pl.total) return RAD_E_SCRATCH;
    if (!rad_ptr_aligned(a->scratch, EXL3I_A8_ALIGN)) return RAD_E_ALIGN;
    if (!rad_ptr_aligned(tv->data, 16)) return RAD_E_ALIGN;   /* the store reads svh 8 at a time */
    Exl3iA8Args q;
    memset(&q, 0, sizeof q);
    q.x = x->data;
    q.codes = tr->data;
    q.suh = ts->data;
    q.suh1 = ts1 ? ts1->data : 0;
    q.svh = tv->data;
    q.y = y->data;
    q.aq = (int8_t*)((char*)a->scratch + pl.aq_off);
    q.sa = (float*)((char*)a->scratch + pl.sa_off);
    q.wq = (int8_t*)((char*)a->scratch + pl.w_off);
    q.x_pitch = x->stride[0];
    q.y_pitch = y->stride[0];
    q.M = (int32_t)T;
    q.N = (int32_t)g.N;
    q.K = (int32_t)g.K;
    q.m = g.m;
    q.group = (int32_t)group;
    q.mats = pl.mats;
    q.x_bf16 = x->dtype == RAD_BF16;
    q.y_dt = y->dtype == RAD_F32 ? 1 : y->dtype == RAD_F16 ? 2 : 0;
    q.cb = g.cb;
    /* AQ (version 3): the producer quantised the rows (exl3i_actq), this weight's planes at aq_plane of aq_planes */
    const RadTensor* aq = a->n_t > EXL3I_OPD_AQ ? rad_arg_in(a, EXL3I_OPD_AQ) : 0;
    const int64_t np = rad_param_geti(a->p, a->n_p, "aq_planes", 0), p0 = rad_param_geti(a->p, a->n_p, "aq_plane", -1);
    if (aq && aq->data && np > 0) {   /* without aq_planes (a tool's synthesised operand) aq is not read */
        const int64_t rb = exl3i_aq_row_bytes(g.K, (int)np), nkb = g.K / 128;
        if (T < a8_min_rows()) return RAD_E_SHAPE;   /* not reached: exl3i_nt_launch refuses it first */
        if (group != 128 || np < 1 || np > EXL3I_AQ_MAX_PLANES || p0 < 0 || p0 + pl.mats > np) return RAD_E_SHAPE;
        if (aq->rank != 2 || aq->shape[0] < T || aq->stride[0] < rb || aq->stride[0] % 4 || rad_dtype_bytes(aq->dtype, 1) != 1)
            return RAD_E_SHAPE;
        if (!rad_ptr_aligned(aq->data, 16)) return RAD_E_ALIGN;
        q.aq = (int8_t*)aq->data + p0 * g.K;
        q.sa = (float*)((char*)aq->data + np * g.K) + p0 * nkb;
        q.a_ld = aq->stride[0];
        q.a_ms = g.K;
        q.sa_ld = aq->stride[0] / 4;
        q.sa_ms = nkb;
    } else {
        rc = exl3i_a8_actq(&q, s);
    }
    for (int64_t c0 = 0; rc == RAD_OK && c0 < g.N; c0 += pl.chunk) {
        q.c0 = (int32_t)c0;
        q.nc = (int32_t)(g.N - c0 < pl.chunk ? g.N - c0 : pl.chunk);
        rc = exl3i_a8_gemmd(&q, s);   /* FUSE: the GEMM decodes its own weight tiles (mad) */
        if (rc != RAD_E_UNSUPPORTED) continue;
        rc = exl3i_a8_wdec8(&q, s);
        if (rc == RAD_OK) rc = exl3i_a8_gemm(&q, s);
    }
    return rc;
}

/* The byte-sum GEMV applies: a8g128 (its rows are the A8 path's a8g128 bytes). The test pin EXL3I_PIN_MAD_GEMV_OFF
 * turns it off (nt_pass with DQ on the same weight). */
static int mad_gemv_on(const RadArgs* a) {
    const char* ac = rad_param_gets(a->p, a->n_p, "act", 0);
    Exl3Act m;
    if (!ac || exl3_act_parse(ac, &m) != 0 || m.kind != EXL3_ACT_A8G || m.group != 128) return 0;
    return exl3i_pins[EXL3I_PIN_MAD_GEMV_OFF] == 0;
}
/* The most rows the GEMV takes by itself (no producer's rows: it quantises them in a pre-pass, a second launch) at rate
 * m / 16 on N output rows; 0: nt_pass. Past K2 a trunk shape's pre-pass costs more than the GEMV saves over nt_pass's DOT
 * form on mad (measured on gfx1201, T 1, layer: K2 248 vs 269 us, K2.5 298 vs 280, K4.125 397 vs 368); a head-sized N amortises it (the
 * head K2.5 747 vs 847, K4.125 1079 vs 1273). The test pin EXL3I_PIN_MAD_GEMV_T: up to n rows at every rate and shape. */
static int mad_gemv_rows(int m, int64_t N) {
    const int64_t t = exl3i_pins[EXL3I_PIN_MAD_GEMV_T];
    if (t > 0) return (int)(t < EXL3I_MAD_MAX_ROWS ? t : EXL3I_MAD_MAX_ROWS);
    if (m > 32 && N < 65536) return 0;
    return m < 64 ? 2 : 1;
}

/* The launch's act kind (EXL3_ACT_*), f16 when absent. */
static int act_kind(const RadArgs* a) {
    const char* ac = rad_param_gets(a->p, a->n_p, "act", 0);
    Exl3Act m;
    return ac && exl3_act_parse(ac, &m) == 0 ? m.kind : EXL3_ACT_F16;
}

int exl3i_nt_launch(const RadArgs* a, RadStream s) {
    Exl3iGemm q;
    int cb = 0;
    const int rc = fill(a, &q, &cb);
    if (rc != RAD_OK) return rc;
    const int kind = act_kind(a);
    /* AQ: a producer that wrote the planes may have left `a` stale (exl3i_abi.h), so aq is never passed over here */
    const RadTensor* aq = a->n_t > EXL3I_OPD_AQ ? rad_arg_in(a, EXL3I_OPD_AQ) : 0;
    const int aq_on = aq && aq->data && rad_param_geti(a->p, a->n_p, "aq_planes", 0) > 0;
    /* version 4: a mad a8g128 launch of 1 .. EXL3I_AQ_XB_MAX_ROWS rows reads aq in form xb (the GEMV's rows) */
    const int aq_xb = aq_on && cb == EXL3I_CB_MAD && kind == EXL3_ACT_A8G && q.g.T <= EXL3I_AQ_XB_MAX_ROWS;
    if (aq_on && !aq_xb && (kind != EXL3_ACT_A8G || q.g.T < a8_min_rows() || q.r16))   /* R16: no A8 path to read it */
        return RAD_E_SHAPE;
    if (aq_xb) {
        const int64_t np = rad_param_geti(a->p, a->n_p, "aq_planes", 0), p0 = rad_param_geti(a->p, a->n_p, "aq_plane", -1);
        const int mats = q.g.cls[0].suh1 ? 2 : 1;
        const int64_t T = q.g.T, N = q.g.y_cols, K = q.g.cls[0].cols;
        Exl3Act am;
        const char* ac = rad_param_gets(a->p, a->n_p, "act", 0);
        if (!ac || exl3_act_parse(ac, &am) != 0 || am.group != 128) return RAD_E_SHAPE;
        if (np < 1 || np > EXL3I_AQ_MAX_PLANES || p0 < 0 || p0 + mats > np) return RAD_E_SHAPE;
        if (aq->rank != 2 || aq->shape[0] < T || rad_dtype_bytes(aq->dtype, 1) != 1 ||
            aq->stride[0] < exl3i_aq_xb_row_bytes(K, (int)np) || aq->stride[0] % 256)
            return RAD_E_SHAPE;
        if (!rad_ptr_aligned(aq->data, 256)) return RAD_E_ALIGN;
        q.xq = (const unsigned char*)aq->data;
        q.xq_ld = aq->stride[0];
        q.xq_plane = (int32_t)p0;
        ks_attach(&q, a, T, N, K, mad_region_bytes(T, K));
        return exl3i_launch_mad_gemv(&q, EXL3I_MAD_NPW, s);
    }
    if (kind != EXL3_ACT_F16) {
        if (q.g.T >= a8_min_rows() && !q.r16) return a8_launch(a, s);
        if (q.r16 && a->scratch_bytes < region_bytes(q.g.T, q.g.cls[0].cols)) return RAD_E_SCRATCH;   /* f16 path's region */
        q.g.raw = 0;   /* below the threshold: nt_pass (the arena holds the region, not T2's chunk) */
        if (cb == EXL3I_CB_MAD && q.g.T <= mad_gemv_rows(q.m, q.g.y_cols) && mad_gemv_on(a)) {   /* the byte-sum GEMV */
            const int64_t T = q.g.T, N = q.g.y_cols, K = q.g.cls[0].cols;
            const int64_t at = mad_region_bytes(T, K);
            if (q.g.xr && a->scratch_bytes >= at) {
                ks_attach(&q, a, T, N, K, at);
                return exl3i_launch_mad_gemv(&q, EXL3I_MAD_NPW, s);
            }
        }
    }
    if (!q.g.xr) return RAD_E_INVAL;
    if (cb == EXL3I_CB_MAD) return exl3i_launch_nt_mad(&q, EXL3I_NPW, s);
    return RAD_E_UNSUPPORTED;   /* mad is the one codebook (rows.c's constraint refuses the rest at selection) */
}

/* At most 16 rows, room for KS's partial sums past the region (at the declared M, which bounds a launch's T). */
static int64_t ks_scratch(const Exl3iGeom* g, int64_t total) {
    const int64_t kb = ks_bytes(g->M, g->N, g->K), need = region_bytes(g->M, g->K) + kb;
    return kb > 0 && need > total ? need : total;
}

/* act f16: the region [T][2K] f16, 256-B aligned, and T2's decoded chunk from EXL3R4D_T2_MIN rows.
 * An A8 mode: the A8 arena at M (exl3i_a8.h; the region alone below EXL3I_A8_MIN_ROWS). -1: a geometry no path serves. */
int64_t exl3i_nt_scratch(const RadArgs* a) {
    Exl3iGeom g;
    if (exl3i_geom_parse(a->p, a->n_p, &g) != RAD_OK) return -1;
    if (g.act.kind != EXL3_ACT_F16) {
        Exl3iA8Plan pl;
        /* the hook sees no operands (rad_builder.cpp: a scratch hook depends only on geometry), so whether suh.1 is
         * there is unknown: a shape that could be stacked (N % 256) is sized for both halves' activation planes */
        const int stacked = g.N % 256 == 0;
        if (g.cb != EXL3I_CB_MAD) return -1;
        if (exl3i_a8_plan(g.M, g.N, g.K, stacked, exl3i_a8_group(g.act.kind, g.act.group, g.K), a8_chunk_bytes(),
                          a8_min_rows(), &pl) != 0)
            return -1;
        if (g.cb == EXL3I_CB_MAD && g.act.kind == EXL3_ACT_A8G && g.act.group == 128) {   /* the GEMV's rows + KS */
            const int64_t T = g.M < EXL3I_MAD_MAX_ROWS ? g.M : EXL3I_MAD_MAX_ROWS;
            const int64_t need = mad_region_bytes(T, g.K) + ks_bytes(T, g.N, g.K);
            if (need > pl.total) pl.total = need;
        }
        return ks_scratch(&g, pl.total);
    }
    return ks_scratch(&g, region_bytes(g.M, g.K) + (g.M >= EXL3R4D_T2_MIN ? EXL3R4D_T2_CHUNK_BYTES : 0));
}

/* ------------------------------------------------------------------ the layout hooks */

void exl3i_rows_to_tiles(const uint32_t* rec, int64_t rows, int64_t cols, int m, uint32_t* out) {
    const int64_t nt = rows / 16, kt = cols / 16, W = m / 2;
    for (int64_t tn = 0; tn < nt; ++tn)
        for (int64_t tk = 0; tk < kt; ++tk)
            memcpy(out + (tk * nt + tn) * W, rec + (tn * kt + tk) * W, (size_t)W * 4);
}

void exl3i_tiles_to_rows(const uint32_t* tiles, int64_t rows, int64_t cols, int m, uint32_t* out) {
    const int64_t nt = rows / 16, kt = cols / 16, W = m / 2;
    for (int64_t tn = 0; tn < nt; ++tn)
        for (int64_t tk = 0; tk < kt; ++tk)
            memcpy(out + (tn * kt + tk) * W, tiles + (tk * nt + tn) * W, (size_t)W * 4);
}

/* Operand -> its plane, checked against the encoding and the geometry: *q the operand, rows / cols / m the
 * weight's, in the op's (per rank) geometry. Under tensor parallelism the planes are this rank's slice of the
 * container's (the loader cuts a single-plane declaration in the plane's own elements, core/format/share.cpp):
 * a row shard's record is [N / 16, words(K)] of N the rank's rows, a column shard's words(K) of the rank's K, the
 * suh / suh.1 numel the rank's K, the svh numel the rank's N; the whole weight (world 1, a replicated R16 head)
 * has the container's extents. The op geometry is the slice, so the checks run against it. */
static int check(int operand, const RadParam* p, int n_p, const RadEncoding* enc, const int* sel,
                 const RadTensor* planes, int n_planes, int64_t* rows, int64_t* cols, int* m, int* r16) {
    static const char* const role[EXL3I_N_OPD] = { 0, "record", "suh", "svh", "suh.1", 0, 0 };
    if (operand < 0 || operand >= EXL3I_N_OPD || !role[operand]) return RAD_E_UNSUPPORTED;
    int em, ecb, stacked, o16;
    int64_t er, ec;
    if (exl3i_encoding_read_r16(enc, &em, &ecb, &stacked, &er, &ec, &o16) != RAD_OK) return RAD_E_DTYPE;
    if (r16) *r16 = o16;
    Exl3iGeom g;
    if (exl3i_geom_parse(p, n_p, &g) != RAD_OK) return RAD_E_SHAPE;
    if (g.m != em || g.cb != ecb) return RAD_E_DTYPE;
    if (g.N > er || g.K > ec || er % g.N || ec % g.K) return RAD_E_SHAPE;
    if (operand == EXL3I_OPD_SUH1 && !stacked) return RAD_E_SHAPE;
    if (n_planes != 1 || !sel || !planes || sel[0] < 0 || sel[0] >= (int)enc->n_planes) return RAD_E_SHAPE;
    if (strcmp(enc->plane[sel[0]].role, role[operand]) != 0) return RAD_E_SHAPE;
    if (planes[0].rank != 2) return RAD_E_SHAPE;
    if (operand == EXL3I_OPD_RECORD) {
        if (planes[0].dtype != RAD_U32) return RAD_E_DTYPE;
        if (planes[0].shape[0] != g.N / 16 || planes[0].shape[1] != exl3i_row_words(g.K, em)) return RAD_E_SHAPE;
    } else {
        if (planes[0].dtype != RAD_F16) return RAD_E_DTYPE;
        if (planes[0].shape[0] * planes[0].shape[1] != (operand == EXL3I_OPD_SVH ? g.N : g.K)) return RAD_E_SHAPE;
    }
    *rows = g.N;
    *cols = g.K;
    *m = em;
    return RAD_OK;
}

static int64_t record_bytes(int64_t rows, int64_t cols, int m) { return rows / 16 * exl3i_row_words(cols, m) * 4; }

int exl3i_nt_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                    const RadTensor* planes, int n_planes, RadLayout* out) {
    int64_t rows, cols;
    int m, r16;
    const int rc = check(operand, p, n_p, enc, sel, planes, n_planes, &rows, &cols, &m, &r16);
    if (rc != RAD_OK) return rc;
    if (!out) return RAD_E_INVAL;
    memset(out, 0, sizeof *out);
    out->rank = 2;
    out->shape[0] = planes[0].shape[0];
    out->shape[1] = planes[0].shape[1];
    out->align = 256;
    if (operand != EXL3I_OPD_RECORD) {
        out->tag = "exl3i.f16.vec";
        out->dtype = RAD_F16;
        out->bytes = planes[0].shape[0] * planes[0].shape[1] * 2;
        return RAD_OK;
    }
    out->tag = r16 ? "exl3i.tiles.kmajor.r16" : "exl3i.tiles.kmajor";
    out->dtype = r16 ? EXL3I_DT_TILES_R16 : EXL3I_DT_TILES;
    out->bytes = record_bytes(rows, cols, m);
    return RAD_OK;
}

int exl3i_nt_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                      const RadTensor* planes, int n_planes, void* dst, int64_t dst_bytes) {
    int64_t rows, cols;
    int m;
    const int rc = check(operand, p, n_p, enc, sel, planes, n_planes, &rows, &cols, &m, 0);
    if (rc != RAD_OK) return rc;
    if (operand != EXL3I_OPD_RECORD) {
        const int64_t b = planes[0].shape[0] * planes[0].shape[1] * 2;
        if (!dst || !planes[0].data || dst_bytes < b) return RAD_E_INVAL;
        memcpy(dst, planes[0].data, (size_t)b);
        return RAD_OK;
    }
    if (!dst || !planes[0].data || dst_bytes < record_bytes(rows, cols, m)) return RAD_E_INVAL;
    exl3i_rows_to_tiles((const uint32_t*)planes[0].data, rows, cols, m, (uint32_t*)dst);
    return RAD_OK;
}

int exl3i_nt_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                        const void* src, int64_t src_bytes, const RadTensor* planes, int n_planes) {
    int64_t rows, cols;
    int m;
    const int rc = check(operand, p, n_p, enc, sel, planes, n_planes, &rows, &cols, &m, 0);
    if (rc != RAD_OK) return rc;
    if (operand != EXL3I_OPD_RECORD) {
        const int64_t b = planes[0].shape[0] * planes[0].shape[1] * 2;
        if (!src || !planes[0].data || src_bytes < b) return RAD_E_INVAL;
        memcpy(planes[0].data, src, (size_t)b);
        return RAD_OK;
    }
    if (!src || !planes[0].data || src_bytes < record_bytes(rows, cols, m)) return RAD_E_INVAL;
    exl3i_tiles_to_rows((const uint32_t*)src, rows, cols, m, (uint32_t*)planes[0].data);
    return RAD_OK;
}

/* ------------------------------------------------------------------ exl3i_actq (AQ, exl3i_abi.h version 3) */

/* x [M, K] f16 | bf16 ([M, 2K] bf16 for silu_mul), s0 .. s(planes - 1) f16 [K], out u8 [>= M, >= exl3i_aq_row_bytes(K,
 * planes)], pitch % 4; pre rmsnorm: g f32 | bf16 [K]; pre mul: x2 bf16 [M, K]; h (optional) bf16 [M, K]; res = 1 (pre
 * rmsnorm, version 5): res and xo bf16 [M, K]. */
int exl3i_aq_launch(const RadArgs* a, RadStream s) {
    int64_t M, K;
    int planes, pre;
    int rc = exl3i_aq_parse(a->p, a->n_p, &M, &K, &planes, &pre);
    if (rc != RAD_OK) return rc;
    const int form = exl3i_aq_form_parse(rad_param_gets(a->p, a->n_p, "form", 0));
    const int64_t rowb = form == EXL3I_AQ_FORM_XB ? exl3i_aq_xb_row_bytes(K, planes) : exl3i_aq_row_bytes(K, planes);
    if (a->n_t < EXL3I_AQ_N_OPD_MIN) return RAD_E_INVAL;
    const RadTensor* x = rad_arg_in(a, EXL3I_AQ_X);
    const RadTensor* o = rad_arg_in(a, EXL3I_AQ_OUT);
    const RadTensor* x2 = a->n_t > EXL3I_AQ_X2 ? rad_arg_in(a, EXL3I_AQ_X2) : 0;
    const RadTensor* gn = a->n_t > EXL3I_AQ_G ? rad_arg_in(a, EXL3I_AQ_G) : 0;
    const RadTensor* h = a->n_t > EXL3I_AQ_H ? rad_arg_in(a, EXL3I_AQ_H) : 0;
    const RadTensor* res = a->n_t > EXL3I_AQ_RES ? rad_arg_in(a, EXL3I_AQ_RES) : 0;
    const RadTensor* xo = a->n_t > EXL3I_AQ_XO ? rad_arg_in(a, EXL3I_AQ_XO) : 0;
    if (!x || !o || !x->data || !o->data) return RAD_E_INVAL;
    if (x->dtype != RAD_BF16 && x->dtype != RAD_F16) return RAD_E_DTYPE;
    if (pre != EXL3I_AQ_PRE_NONE && x->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (rad_dtype_bytes(o->dtype, 1) != 1) return RAD_E_DTYPE;
    const int64_t xw = flat_width(x), T = x->shape[0];
    if (xw < (pre == EXL3I_AQ_PRE_SILU_MUL ? 2 * K : K) || T < 1 || o->rank != 2 || o->shape[0] < T ||
        o->stride[0] < rowb || o->stride[0] % (form == EXL3I_AQ_FORM_XB ? 256 : 4) ||
        (form == EXL3I_AQ_FORM_XB && T > EXL3I_AQ_XB_MAX_ROWS))
        return RAD_E_SHAPE;
    if (!rad_ptr_aligned(x->data, 8) || (x->stride[0] * 2) % 8 || !rad_ptr_aligned(o->data, form == EXL3I_AQ_FORM_XB ? 256 : 16))
        return RAD_E_ALIGN;
    Exl3iAqArgs q;
    memset(&q, 0, sizeof q);
    for (int i = 0; i < planes; ++i) {
        const RadTensor* v = rad_arg_in(a, EXL3I_AQ_S0 + i);
        if (!v || !v->data || v->dtype != RAD_F16 || rad_tensor_numel(v) != K) return RAD_E_SHAPE;
        if (!rad_ptr_aligned(v->data, 8)) return RAD_E_ALIGN;
        q.s[i] = v->data;
    }
    if (pre == EXL3I_AQ_PRE_MUL) {
        if (!x2 || !x2->data || x2->dtype != RAD_BF16) return RAD_E_INVAL;
        if (flat_width(x2) < K || x2->shape[0] < T) return RAD_E_SHAPE;
        if (!rad_ptr_aligned(x2->data, 8) || (x2->stride[0] * 2) % 8) return RAD_E_ALIGN;
        q.x2 = x2->data;
        q.x2_pitch = x2->stride[0];
    }
    if (pre == EXL3I_AQ_PRE_RMSNORM) {
        if (!gn || !gn->data || (gn->dtype != RAD_F32 && gn->dtype != RAD_BF16)) return RAD_E_INVAL;
        if (rad_tensor_numel(gn) != K) return RAD_E_SHAPE;
        q.g = gn->data;
        q.g_f32 = gn->dtype == RAD_F32;
        q.eps = (float)rad_param_getf(a->p, a->n_p, "eps", 1e-6);
        q.wadd = (float)rad_param_getf(a->p, a->n_p, "wadd", 0.0);
    }
    if (h && h->data) {
        if (pre == EXL3I_AQ_PRE_NONE || h->dtype != RAD_BF16) return RAD_E_DTYPE;
        if (flat_width(h) < K || h->shape[0] < T) return RAD_E_SHAPE;
        if (!rad_ptr_aligned(h->data, 8) || (h->stride[0] * 2) % 8) return RAD_E_ALIGN;
        q.h = h->data;
        q.h_pitch = h->stride[0];
    }
    if (exl3i_aq_res(a->p, a->n_p)) {   /* version 5: x + res into xo, then the norm */
        if (!res || !res->data || !xo || !xo->data) return RAD_E_INVAL;
        if (res->dtype != RAD_BF16 || xo->dtype != RAD_BF16) return RAD_E_DTYPE;
        if (flat_width(res) < K || res->shape[0] < T || flat_width(xo) < K || xo->shape[0] < T) return RAD_E_SHAPE;
        if (!rad_ptr_aligned(res->data, 8) || (res->stride[0] * 2) % 8 || !rad_ptr_aligned(xo->data, 8) ||
            (xo->stride[0] * 2) % 8)
            return RAD_E_ALIGN;
        q.res = res->data;
        q.res_pitch = res->stride[0];
        q.xo = xo->data;
        q.xo_pitch = xo->stride[0];
    }
    q.x = x->data;
    q.out = (uint8_t*)o->data;
    q.x_pitch = x->stride[0];
    q.out_ld = o->stride[0];
    q.M = (int32_t)T;
    q.K = (int32_t)K;
    q.planes = planes;
    q.x_bf16 = x->dtype == RAD_BF16;
    q.pre = pre;
    q.form = form;
    return exl3i_aq_actq(&q, s);
}

/* exl3i_actq's weights: s0 .. s3 are exl3i weights' suh / suh.1 planes, which exl3i_dense stores as "exl3i.f16.vec" (a
 * copy, 256-B aligned: exl3i_nt_layout); a weight is stored once, so this row asks for the same form. The rmsnorm gain g
 * is read as it is (the rmsnorm row reads it so too). */
static int aq_check(int operand, const RadParam* p, int n_p, const RadEncoding* enc, const int* sel,
                    const RadTensor* planes, int n_planes) {
    int64_t M, K, er, ec;
    int planes_n, pre, em, ecb, stacked, o16;
    if (operand < EXL3I_AQ_S0 || operand >= EXL3I_AQ_S0 + EXL3I_AQ_MAX_PLANES) return RAD_E_UNSUPPORTED;
    if (exl3i_aq_parse(p, n_p, &M, &K, &planes_n, &pre) != RAD_OK) return RAD_E_SHAPE;
    if (exl3i_encoding_read_r16(enc, &em, &ecb, &stacked, &er, &ec, &o16) != RAD_OK) return RAD_E_DTYPE;
    /* The producer's K is the consumer's K as the consumer's rank reads it (its suh's numel): a column-sharded
     * consumer's container has the full K, the plane holds the rank's slice, checked below. */
    (void) er; (void) ec;
    if (n_planes != 1 || !sel || !planes || sel[0] < 0 || sel[0] >= (int)enc->n_planes) return RAD_E_SHAPE;
    if (strcmp(enc->plane[sel[0]].role, "suh") != 0 && strcmp(enc->plane[sel[0]].role, "suh.1") != 0) return RAD_E_SHAPE;
    if (planes[0].rank != 2 || planes[0].dtype != RAD_F16) return RAD_E_DTYPE;
    if (planes[0].shape[0] * planes[0].shape[1] != K) return RAD_E_SHAPE;
    return RAD_OK;
}

int exl3i_aq_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                    const RadTensor* planes, int n_planes, RadLayout* out) {
    const int rc = aq_check(operand, p, n_p, enc, sel, planes, n_planes);
    if (rc != RAD_OK) return rc;
    if (!out) return RAD_E_INVAL;
    memset(out, 0, sizeof *out);
    out->rank = 2;
    out->shape[0] = planes[0].shape[0];
    out->shape[1] = planes[0].shape[1];
    out->align = 256;
    out->tag = "exl3i.f16.vec";
    out->dtype = RAD_F16;
    out->bytes = planes[0].shape[0] * planes[0].shape[1] * 2;
    return RAD_OK;
}

int exl3i_aq_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                      const RadTensor* planes, int n_planes, void* dst, int64_t dst_bytes) {
    const int rc = aq_check(operand, p, n_p, enc, sel, planes, n_planes);
    if (rc != RAD_OK) return rc;
    const int64_t b = planes[0].shape[0] * planes[0].shape[1] * 2;
    if (!dst || !planes[0].data || dst_bytes < b) return RAD_E_INVAL;
    memcpy(dst, planes[0].data, (size_t)b);
    return RAD_OK;
}

int exl3i_aq_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                        const void* src, int64_t src_bytes, const RadTensor* planes, int n_planes) {
    const int rc = aq_check(operand, p, n_p, enc, sel, planes, n_planes);
    if (rc != RAD_OK) return rc;
    const int64_t b = planes[0].shape[0] * planes[0].shape[1] * 2;
    if (!src || !planes[0].data || src_bytes < b) return RAD_E_INVAL;
    memcpy(planes[0].data, src, (size_t)b);
    return RAD_OK;
}

/* the actq op's operands as a tool synthesises them: x [M, K] f16, s_i [K], out u8 [M, row bytes] */
int exl3i_aq_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M, K;
    int planes;
    int pre;
    if (exl3i_aq_parse(p, n_p, &M, &K, &planes, &pre) != RAD_OK) return RAD_E_UNSUPPORTED;
    if (operand < 0 || operand >= EXL3I_AQ_N_OPD || !out) return RAD_E_SHAPE;
    memset(out, 0, sizeof *out);
    out->idx_const = -1;
    if (operand == EXL3I_AQ_X) {
        out->dtype = pre ? RAD_BF16 : RAD_F16; out->rank = 2; out->shape[0] = M;
        out->shape[1] = pre == EXL3I_AQ_PRE_SILU_MUL ? 2 * K : K; out->fill = RAD_FILL_NORMAL;
    }
    else if (operand == EXL3I_AQ_X2 || operand == EXL3I_AQ_H) {   /* absent unless the pre stage reads / writes it */
        out->dtype = RAD_BF16; out->rank = 2; out->shape[0] = M; out->shape[1] = K; out->fill = RAD_FILL_NORMAL;
        if ((operand == EXL3I_AQ_X2 && pre != EXL3I_AQ_PRE_MUL) || (operand == EXL3I_AQ_H && !pre)) out->flags = RAD_OPD_F_ABSENT;
    }
    else if (operand == EXL3I_AQ_RES || operand == EXL3I_AQ_XO) {   /* version 5: with res = 1 only */
        out->dtype = RAD_BF16; out->rank = 2; out->shape[0] = M; out->shape[1] = K; out->fill = RAD_FILL_NORMAL;
        if (!exl3i_aq_res(p, n_p)) out->flags = RAD_OPD_F_ABSENT;
    }
    else if (operand == EXL3I_AQ_G) {
        out->dtype = RAD_F32; out->rank = 1; out->shape[0] = K; out->fill = RAD_FILL_NORMAL;
        if (pre != EXL3I_AQ_PRE_RMSNORM) out->flags = RAD_OPD_F_ABSENT;
    }
    else if (operand == EXL3I_AQ_OUT) {
        const int form = exl3i_aq_form_parse(rad_param_gets(p, n_p, "form", 0));
        out->dtype = RAD_U8; out->rank = 2; out->shape[0] = M;
        out->shape[1] = form == EXL3I_AQ_FORM_XB ? exl3i_aq_xb_row_bytes(K, planes) : exl3i_aq_row_bytes(K, planes);
    }
    else {
        out->dtype = RAD_F16; out->rank = 1; out->shape[0] = K; out->fill = RAD_FILL_NORMAL;
        if (operand - EXL3I_AQ_S0 >= planes) out->flags = RAD_OPD_F_ABSENT;
    }
    return RAD_OK;
}
