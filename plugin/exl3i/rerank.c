/* rerank.c -- exl3i_rerank's host side (include/exl3i_rerank.h): the operand checks, the argument block, the shape
 * hook. The weight planes are exl3i_dense's at exl3i_dense's operand positions, so its layout hooks serve this row
 * unchanged (rows.c): the same tags, one stored copy of the head. */
#include "exl3i.h"
#include "exl3i_rerank.h"
#include "exl3i_rerank_dev.h"

#include <string.h>

/* The geometry: M (a range in a shape hook), R, N, K, rate, codebook, vocab_off. RAD_E_SCHEMA / RAD_E_DTYPE /
 * RAD_E_SHAPE as exl3i_geom_parse. */
static int geom(const RadParam* p, int n_p, int64_t* M, int64_t* R, Exl3iGeom* g, int64_t* voff) {
    const int rc = exl3i_geom_parse(p, n_p, g);
    if (rc != RAD_OK) return rc;
    if (!rad_param_find(p, n_p, "R")) return RAD_E_SCHEMA;
    *M = rad_param_getdim(p, n_p, "M", 0);
    *R = rad_param_geti(p, n_p, "R", 0);
    *voff = rad_param_geti(p, n_p, "vocab_off", 0);
    if (*M < 1 || *R < 1 || *R > EXL3I_RERANK_MAX_R || *voff < 0) return RAD_E_SHAPE;
    if (g->act.kind != EXL3_ACT_F16) return RAD_E_DTYPE;   /* the op has no act: the decode form's f16 rows */
    return RAD_OK;
}

static void desc(RadOpdDesc* d, uint32_t dt, uint32_t rank, const int64_t* shape, uint32_t fill) {
    memset(d, 0, sizeof *d);
    d->dtype = dt;
    d->rank = rank;
    for (uint32_t r = 0; r < rank; ++r) d->shape[r] = shape[r];
    d->fill = fill;
    d->idx_const = -1;
}

int exl3i_rerank_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M, R, voff;
    Exl3iGeom g;
    if (geom(p, n_p, &M, &R, &g, &voff) != RAD_OK || !out) return RAD_E_UNSUPPORTED;
    switch (operand) {
    case EXL3I_RR_X:      desc(out, RAD_BF16, 2, (int64_t[]){ M, g.K }, RAD_FILL_NORMAL); break;
    case EXL3I_RR_RECORD: desc(out, RAD_U32, 2, (int64_t[]){ g.N / 16, exl3i_row_words(g.K, g.m) }, RAD_FILL_INDEX);
                          out->idx_max = (int64_t)1 << 32; break;
    case EXL3I_RR_SUH:    desc(out, RAD_F16, 1, (int64_t[]){ g.K }, RAD_FILL_NORMAL); break;
    case EXL3I_RR_SVH:    desc(out, RAD_F16, 1, (int64_t[]){ g.N }, RAD_FILL_NORMAL); break;
    case EXL3I_RR_IDX_IN: desc(out, RAD_I32, 2, (int64_t[]){ M, R }, RAD_FILL_INDEX);   /* global ids; below vocab_off pads */
                          out->idx_max = voff + g.N; out->flags = RAD_OPD_F_IDX_UNIQUE; break;
    case EXL3I_RR_IDX:    desc(out, RAD_I32, 2, (int64_t[]){ M, R }, RAD_FILL_INDEX); out->idx_max = voff + g.N; break;
    case EXL3I_RR_VAL:    desc(out, RAD_F32, 2, (int64_t[]){ M, R }, RAD_FILL_NORMAL); break;
    case EXL3I_RR_PAIRS:  desc(out, RAD_F32, 3, (int64_t[]){ M, R, 2 }, RAD_FILL_NORMAL); break;
    default: return RAD_E_SHAPE;
    }
    return RAD_OK;
}

/* [rows, width] with unit last stride: the row pitch, or -1 */
static int64_t pitch2(const RadTensor* t, int64_t rows, int64_t width) {
    int64_t w = 1;
    if (!t || t->rank < 1 || t->stride[t->rank - 1] != 1) return -1;
    for (uint32_t r = 1; r < t->rank; ++r) w *= t->shape[r];
    if (t->rank == 1) return t->shape[0] >= rows * width ? width : -1;
    if (t->rank == 3 && t->stride[1] != t->shape[2]) return -1;
    if (t->shape[0] < rows || w != width) return -1;
    return t->stride[0];
}

static int fill(const RadArgs* a, Exl3iRerankCall* c) {
    int64_t Mx, R, voff;
    Exl3iGeom g;
    if (!a || a->n_t < EXL3I_RR_PAIRS) return RAD_E_INVAL;
    int rc = geom(a->p, a->n_p, &Mx, &R, &g, &voff);
    if (rc != RAD_OK) return rc;
    const RadTensor *x = rad_arg_in(a, EXL3I_RR_X), *tr = rad_arg_in(a, EXL3I_RR_RECORD);
    const RadTensor *su = rad_arg_in(a, EXL3I_RR_SUH), *sv = rad_arg_in(a, EXL3I_RR_SVH);
    const RadTensor *in = rad_arg_in(a, EXL3I_RR_IDX_IN), *ix = rad_arg_in(a, EXL3I_RR_IDX);
    const RadTensor *va = rad_arg_in(a, EXL3I_RR_VAL);
    const RadTensor *pr = a->n_t > EXL3I_RR_PAIRS ? rad_arg_in(a, EXL3I_RR_PAIRS) : 0;
    if (!x || !tr || !su || !sv || !in || !ix || !va) return RAD_E_INVAL;
    if (x->dtype != RAD_BF16 && x->dtype != RAD_F16) return RAD_E_DTYPE;
    if (in->dtype != RAD_I32 || ix->dtype != RAD_I32 || va->dtype != RAD_F32) return RAD_E_DTYPE;
    if (su->dtype != RAD_F16 || sv->dtype != RAD_F16) return RAD_E_DTYPE;
    if (rad_tensor_numel(su) != g.K || rad_tensor_numel(sv) != g.N) return RAD_E_SHAPE;
    /* the record as exl3i_dense's layout hook stored it (EXL3I_DT_TILES_R16: rows in R16 order) */
    if ((tr->dtype != EXL3I_DT_TILES && tr->dtype != EXL3I_DT_TILES_R16) || tr->rank != 2 || tr->shape[0] != g.N / 16 ||
        tr->shape[1] != exl3i_row_words(g.K, g.m))
        return RAD_E_SHAPE;
    /* rows come from the operand (M is a band's ceiling) */
    const int64_t M = x->shape[0];
    if (M < 0 || M > Mx) return RAD_E_SHAPE;
    const int64_t xp = pitch2(x, M, g.K), ip = pitch2(in, M, R), dp = pitch2(ix, M, R), vp = pitch2(va, M, R);
    if (xp < 0 || ip < 0 || dp < 0 || vp < 0) return RAD_E_STRIDE;
    if (pr) {
        if (pr->dtype != RAD_F32 || rad_tensor_numel(pr) < M * R * 2) return RAD_E_SHAPE;
        if (!rad_tensor_is_contiguous(pr)) return RAD_E_STRIDE;   /* [M, R, 2] packed, as row_topk writes it */
    }
    memset(c, 0, sizeof *c);
    c->x = x->data;
    c->tiles = (const uint32_t*)tr->data;
    c->suh = (const uint16_t*)su->data;
    c->svh = (const uint16_t*)sv->data;
    c->idx_in = (const int32_t*)in->data;
    c->idx = (int32_t*)ix->data;
    c->val = (float*)va->data;
    c->pairs = pr ? (float*)pr->data : 0;
    c->x_pitch = xp;
    c->in_pitch = ip;
    c->idx_pitch = dp;
    c->val_pitch = vp;
    c->M = M;
    c->R = R;
    c->N = g.N;
    c->K = g.K;
    c->voff = voff;
    c->m = g.m;
    c->cb = g.cb;
    c->r16 = tr->dtype == EXL3I_DT_TILES_R16;
    c->x_f16 = x->dtype == RAD_F16;
    return RAD_OK;
}

int exl3i_rerank_launch(const RadArgs* a, RadStream s) {
    Exl3iRerankCall c;
    const int rc = fill(a, &c);
    if (rc != RAD_OK) return rc;
    return exl3i_launch_rerank(&c, s);
}
