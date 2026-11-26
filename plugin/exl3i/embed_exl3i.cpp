/* embed_exl3i.cpp -- the embed_exl3i schema, shape and layout hooks, operand reader and host row (embed_exl3i.h). */
#include "embed_exl3i.h"
#include "exl3_decode.h"
#include "exl3i_encoding.h"
#include "rad_encoding.h"
#include "rad_plugin.h"

#include <cstring>

namespace {

const RadParamSpec kParams[] = {
    { "M",        RAD_P_INT, RAD_REQUIRED, 0 },
    { "n_embd",   RAD_P_INT, RAD_REQUIRED, 0 },
    { "n_vocab",  RAD_P_INT, RAD_REQUIRED, 0 },
    { "m",        RAD_P_INT, RAD_REQUIRED, 0 },
    { "codebook", RAD_P_STR, RAD_REQUIRED, 0 },
    { "r16",      RAD_P_INT, RAD_REQUIRED, 0 },
};

const RadOperandSpec kOperands[] = {
    { "tokens", RAD_OPD_IN,     0 },
    { "record", RAD_OPD_WEIGHT, 0 },
    { "suh",    RAD_OPD_WEIGHT, 0 },
    { "svh",    RAD_OPD_WEIGHT, 0 },
    { "x",      RAD_OPD_OUT,    0 },
};

const RadOpSchema kSchema = {
    EMBED_EXL3I_OP, kParams, 6, kOperands, EMBED_EXL3I_N_OPD,
    "x[m] = row tokens[m] of an exl3i (trellis) token table, decoded from the stored record, suh and svh: the 128 "
    "trellis rows of the token's output-Hadamard block folded with its signs, rotated over 128 columns, x 2^-7 x suh x "
    "svh, one bf16 rounding (embed_exl3i.h); an id outside [0, n_vocab) writes a zero row.",
};

struct Geom {
    int64_t ne, nv;
    int m, cb, r16;
};

int geom(const RadParam* p, int n_p, Geom* g) {
    g->ne = rad_param_geti(p, n_p, "n_embd", 0);
    g->nv = rad_param_geti(p, n_p, "n_vocab", 0);
    g->m = (int)rad_param_geti(p, n_p, "m", 0);
    g->cb = exl3i_codebook_from_name(rad_param_gets(p, n_p, "codebook", ""));
    g->r16 = (int)rad_param_geti(p, n_p, "r16", 0);
    if (g->ne <= 0 || g->ne % 128 || g->nv <= 0 || g->nv % 128 || !exl3i_rate_ok(g->m) || g->cb < 0 ||
        (g->r16 != 0 && g->r16 != 1))
        return RAD_E_SHAPE;
    return RAD_OK;
}

inline float half_f(uint16_t h) { return exl3_half_to_float(h); }

inline uint16_t f32bf16(float f) {   /* round to nearest even; a NaN stays a NaN */
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

}  /* namespace */

extern "C" {

const RadOpSchema* embed_exl3i_schema(void) { return &kSchema; }

int embed_exl3i_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    Geom g;
    if (!out || geom(p, n_p, &g) != RAD_OK) return RAD_E_UNSUPPORTED;
    const int64_t M = rad_param_getdim(p, n_p, "M", 0);
    if (M <= 0) return RAD_E_UNSUPPORTED;
    RadOpdDesc d{};
    d.idx_const = -1;
    d.fill = RAD_FILL_NORMAL;
    d.rank = 2;
    switch (operand) {
    case EMBED_EXL3I_TOKENS:
        d.dtype = RAD_I32; d.rank = 1; d.shape[0] = M; d.fill = RAD_FILL_INDEX; d.idx_max = g.nv;
        break;
    case EMBED_EXL3I_RECORD:   /* any bits are a trellis */
        d.dtype = RAD_U32; d.shape[0] = g.nv / 16; d.shape[1] = exl3i_row_words(g.ne, g.m); d.fill = RAD_FILL_INDEX;
        break;
    case EMBED_EXL3I_SUH: d.dtype = RAD_F16; d.shape[0] = 1; d.shape[1] = g.ne; break;
    case EMBED_EXL3I_SVH: d.dtype = RAD_F16; d.shape[0] = g.nv; d.shape[1] = 1; break;
    case EMBED_EXL3I_X:   d.dtype = RAD_BF16; d.shape[0] = M; d.shape[1] = g.ne; break;
    default: return RAD_E_SHAPE;
    }
    *out = d;
    return RAD_OK;
}

/* Every plane is read where it lies (the record row-major: one 16-row strip a table row, its tiles in k order, which
 * is the order a token's block wants); the hook only checks the encoding is the exl3i one the params name: rate,
 * codebook, row order, [n_vocab, n_embd], not stacked, and the plane the operand takes. */
int embed_exl3i_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                       const RadTensor* planes, int n_planes, RadLayout* out) {
    (void)planes; (void)n_planes; (void)out;
    Geom g;
    if (geom(p, n_p, &g) != RAD_OK) return RAD_E_SHAPE;
    int m = 0, cb = 0, stacked = 0, r16 = 0;
    int64_t rows = 0, cols = 0;
    if (!enc || !sel || exl3i_encoding_read_r16(enc, &m, &cb, &stacked, &rows, &cols, &r16) != RAD_OK)
        return RAD_E_DTYPE;
    if (m != g.m || cb != g.cb || r16 != g.r16 || stacked) return RAD_E_DTYPE;
    if (rows != g.nv || cols != g.ne) return RAD_E_SHAPE;
    const int want = operand == EMBED_EXL3I_RECORD ? EXL3I_PLANE_RECORD : operand == EMBED_EXL3I_SUH ? EXL3I_PLANE_SUH
                   : operand == EMBED_EXL3I_SVH ? EXL3I_PLANE_SVH : -1;
    if (want < 0 || sel[0] != want) return RAD_E_SHAPE;
    return RAD_E_UNSUPPORTED;   /* as stored */
}

int embed_exl3i_args(const RadArgs* a, EmbedExl3iCall* o) {
    if (!a || !o || !a->t || a->n_t != EMBED_EXL3I_N_OPD) return RAD_E_INVAL;
    Geom g;
    if (geom(a->p, a->n_p, &g) != RAD_OK) return RAD_E_SHAPE;
    const RadTensor* tk = rad_arg_in(a, EMBED_EXL3I_TOKENS);
    const RadTensor* r  = rad_arg_in(a, EMBED_EXL3I_RECORD);
    const RadTensor* su = rad_arg_in(a, EMBED_EXL3I_SUH);
    const RadTensor* sv = rad_arg_in(a, EMBED_EXL3I_SVH);
    const RadTensor* x  = rad_arg_in(a, EMBED_EXL3I_X);
    if (!tk || !r || !su || !sv || !x) return RAD_E_INVAL;
    if (tk->dtype != RAD_I32 || r->dtype != RAD_U32 || su->dtype != RAD_F16 || sv->dtype != RAD_F16 ||
        x->dtype != RAD_BF16)
        return RAD_E_DTYPE;
    int64_t M = 1;   /* tokens: [M] or [M, 1] */
    for (uint32_t i = 0; i < tk->rank; ++i) M *= tk->shape[i];
    const int64_t rw = exl3i_row_words(g.ne, g.m);
    if (r->rank != 2 || r->shape[0] != g.nv / 16 || r->shape[1] != rw || !rad_tensor_is_contiguous(r))
        return RAD_E_SHAPE;
    if (rad_tensor_numel(su) != g.ne || rad_tensor_numel(sv) != g.nv) return RAD_E_SHAPE;
    if (x->rank < 2 || x->shape[0] < M || x->shape[x->rank - 1] * (x->rank == 3 ? x->shape[1] : 1) != g.ne ||
        x->stride[x->rank - 1] != 1)
        return RAD_E_SHAPE;
    o->tokens = (const int32_t*)tk->data;
    o->record = (const uint32_t*)r->data;
    o->row_words = rw;
    o->suh = (const uint16_t*)su->data;
    o->svh = (const uint16_t*)sv->data;
    o->x = (uint16_t*)x->data;
    o->x_pitch = x->stride[0];
    o->M = M;
    o->n_embd = g.ne;
    o->n_vocab = g.nv;
    o->m = g.m;
    o->cb = g.cb;
    o->r16 = g.r16;
    return RAD_OK;
}

/* The host row: per token and 128 columns, exl3i_decode_rows' own operations on the token's block restricted to these
 * columns (the k rotation is per 128 columns, so they need nothing else): the 64 tiles decoded by exl3i_decode_tile,
 * exl3_had128 and x suh on each of the 128 rows, exl3_had128 on each column, the token's element x svh, RNE bf16. Slow on
 * purpose (a reference); no allocation (a fixed scratch). */
int embed_exl3i_host_launch(const RadArgs* a, RadStream s) {
    (void)s;
    EmbedExl3iCall c;
    memset(&c, 0, sizeof c);   /* no stack garbage in the padding: a recorded launch replays byte for byte */
    const int rc = embed_exl3i_args(a, &c);
    if (rc != RAD_OK) return rc;
    const int W = c.m / 2;
    int tr[256], tcol[256];
    for (int p = 0; p < 256; ++p) exl3_pos_to_tile(p, &tr[p], &tcol[p]);
    static thread_local float blk[128][128];   /* [block row r][column] of the current 128 columns */
    uint16_t val[256];
    float col[128];
    for (int64_t mi = 0; mi < c.M; ++mi) {
        uint16_t* xr = c.x + mi * c.x_pitch;
        const int64_t id = c.tokens[mi];
        if (id < 0 || id >= c.n_vocab) {
            std::memset(xr, 0, (size_t)c.n_embd * 2);
            continue;
        }
        const int64_t n = embed_exl3i_row(id, c.n_vocab, c.r16), b = n / 128;
        const int j = (int)(n % 128);
        const float sv = half_f(c.svh[n]);
        for (int64_t g = 0; g < c.n_embd; g += 128) {
            for (int st = 0; st < 8; ++st)          /* strip st of the block, its 8 tiles of these columns */
                for (int t = 0; t < 8; ++t) {
                    const uint32_t* w = c.record + (b * 8 + st) * c.row_words + (g / 16 + t) * W;
                    exl3i_decode_tile(w, c.m, c.cb, val, nullptr);
                    for (int p = 0; p < 256; ++p) blk[st * 16 + tcol[p]][t * 16 + tr[p]] = half_f(val[p]);
                }
            for (int r = 0; r < 128; ++r) {
                exl3_had128(blk[r], 128);
                for (int k = 0; k < 128; ++k) blk[r][k] *= half_f(c.suh[g + k]);
            }
            for (int k = 0; k < 128; ++k) {
                for (int r = 0; r < 128; ++r) col[r] = blk[r][k];
                exl3_had128(col, 128);
                xr[g + k] = f32bf16(col[j] * sv);
            }
        }
    }
    return RAD_OK;
}

}  /* extern "C" */
