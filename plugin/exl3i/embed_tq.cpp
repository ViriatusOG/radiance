/* embed_tq.cpp -- the embed_tq schema, shape and layout hooks, operand reader and host row (embed_tq.h). */
#include "embed_tq.h"
#include "rad_encoding.h"
#include "rad_plugin.h"

#include <cstring>

namespace {

const RadParamSpec kParams[] = {
    { "M",       RAD_P_INT, RAD_REQUIRED, 0 },
    { "n_embd",  RAD_P_INT, RAD_REQUIRED, 0 },
    { "n_vocab", RAD_P_INT, RAD_REQUIRED, 0 },
    { "bits",    RAD_P_INT, RAD_REQUIRED, 0 },
};

const RadOperandSpec kOperands[] = {
    { "tokens", RAD_OPD_IN,     0 },
    { "codes",  RAD_OPD_WEIGHT, 0 },
    { "scale",  RAD_OPD_WEIGHT, 0 },
    { "table",  RAD_OPD_WEIGHT, 0 },
    { "x",      RAD_OPD_OUT,    0 },
};

const RadOpSchema kSchema = {
    EMBED_TQ_OP, kParams, 4, kOperands, EMBED_TQ_N_OPD,
    "x[m] = bf16(H128 (table[codes[id]] * scale[id])) per 128 columns of row id = tokens[m]: the token "
    "embedding gathered from a codebook-quantised, fwht128-rotated table (rtn table= group=128 scale=bf16 "
    "transform=fwht128), decoded in the core's own order; an id outside [0, n_vocab) writes a zero row.",
};

const uint32_t kCodeDt[9] = { 0, RAD_U1, RAD_U2, RAD_U3, RAD_U4, RAD_U5, RAD_U6, RAD_U7, RAD_U8 };

int geom(const RadParam* p, int n_p, int64_t* ne, int64_t* nv, int* bits) {
    *ne = rad_param_geti(p, n_p, "n_embd", 0);
    *nv = rad_param_geti(p, n_p, "n_vocab", 0);
    *bits = (int)rad_param_geti(p, n_p, "bits", 0);
    if (*ne <= 0 || *ne % 128 || *nv <= 0 || *bits < 1 || *bits > 8) return RAD_E_SHAPE;
    return RAD_OK;
}

inline float bf16f(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t f32bf16(float f) {   /* round to nearest even; a NaN stays a NaN */
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

}  /* namespace */

extern "C" {

const RadOpSchema* embed_tq_schema(void) { return &kSchema; }

int embed_tq_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t ne, nv;
    int bits;
    if (!out || geom(p, n_p, &ne, &nv, &bits) != RAD_OK) return RAD_E_UNSUPPORTED;
    const int64_t M = rad_param_getdim(p, n_p, "M", 0);
    if (M <= 0) return RAD_E_UNSUPPORTED;
    RadOpdDesc d{};
    d.idx_const = -1;
    d.fill = RAD_FILL_NORMAL;
    d.rank = 2;
    switch (operand) {
    case EMBED_TQ_TOKENS:
        d.dtype = RAD_I32; d.rank = 1; d.shape[0] = M; d.fill = RAD_FILL_INDEX; d.idx_max = nv;
        break;
    case EMBED_TQ_CODES:
        d.dtype = kCodeDt[bits]; d.shape[0] = nv; d.shape[1] = ne; d.fill = RAD_FILL_INDEX;
        d.idx_max = (int64_t)1 << bits;
        break;
    case EMBED_TQ_SCALE: d.dtype = RAD_BF16; d.shape[0] = nv; d.shape[1] = ne / 128; break;
    case EMBED_TQ_TABLE: d.dtype = RAD_F32; d.shape[0] = 1; d.shape[1] = (int64_t)1 << bits; break;
    case EMBED_TQ_X:     d.dtype = RAD_BF16; d.shape[0] = M; d.shape[1] = ne; break;
    default: return RAD_E_SHAPE;
    }
    *out = d;
    return RAD_OK;
}

/* Every plane is read where it lies; the hook only checks the encoding is the one the row decodes: affine,
 * codes uB [1x1], a bf16 scale per [1x128], an f32 table of 2^B values, fwht128, nothing else. */
int embed_tq_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                    const RadTensor* planes, int n_planes, RadLayout* out) {
    (void)planes; (void)n_planes; (void)out;
    int64_t ne, nv;
    int bits;
    if (geom(p, n_p, &ne, &nv, &bits) != RAD_OK) return RAD_E_SHAPE;
    if (!enc || !sel || !rad_enc_is(enc, "affine") || enc->n_planes != 3 || rad_enc_fwht(enc) != 128)
        return RAD_E_DTYPE;
    const int ic = rad_enc_find(enc, "codes"), is = rad_enc_find(enc, "scale"), it = rad_enc_find(enc, "table");
    if (ic < 0 || is < 0 || it < 0) return RAD_E_DTYPE;
    const RadEncPlane &c = enc->plane[ic], &s = enc->plane[is], &t = enc->plane[it];
    if (c.dtype != kCodeDt[bits] || c.block[0] != 1 || c.block[1] != 1) return RAD_E_DTYPE;
    if (s.dtype != RAD_BF16 || s.block[0] != 1 || s.block[1] != 128) return RAD_E_DTYPE;
    if (t.dtype != RAD_F32 || t.kind != RAD_PLANE_TABLE || t.extent[0] != 1 || t.extent[1] != ((int64_t)1 << bits))
        return RAD_E_DTYPE;
    const int want = operand == EMBED_TQ_CODES ? ic : operand == EMBED_TQ_SCALE ? is
                   : operand == EMBED_TQ_TABLE ? it : -1;
    if (want < 0 || sel[0] != want) return RAD_E_SHAPE;
    return RAD_E_UNSUPPORTED;   /* as stored */
}

int embed_tq_args(const RadArgs* a, EmbedTqCall* o) {
    if (!a || !o || !a->t || a->n_t != EMBED_TQ_N_OPD) return RAD_E_INVAL;
    int64_t ne, nv;
    int bits;
    if (geom(a->p, a->n_p, &ne, &nv, &bits) != RAD_OK) return RAD_E_SHAPE;
    const RadTensor* tk = rad_arg_in(a, EMBED_TQ_TOKENS);
    const RadTensor* c  = rad_arg_in(a, EMBED_TQ_CODES);
    const RadTensor* s  = rad_arg_in(a, EMBED_TQ_SCALE);
    const RadTensor* t  = rad_arg_in(a, EMBED_TQ_TABLE);
    const RadTensor* x  = rad_arg_in(a, EMBED_TQ_X);
    if (!tk || !c || !s || !t || !x) return RAD_E_INVAL;
    if (tk->dtype != RAD_I32 || c->dtype != kCodeDt[bits] || s->dtype != RAD_BF16 || t->dtype != RAD_F32 ||
        x->dtype != RAD_BF16)
        return RAD_E_DTYPE;
    int64_t M = 1;   /* tokens: [M] or [M, 1] */
    for (uint32_t i = 0; i < tk->rank; ++i) M *= tk->shape[i];
    if (x->rank < 2 || x->shape[0] < M || x->shape[x->rank - 1] * (x->rank == 3 ? x->shape[1] : 1) != ne ||
        x->stride[x->rank - 1] != 1)
        return RAD_E_SHAPE;
    int64_t tc = 1;
    for (uint32_t i = 0; i < t->rank; ++i) tc *= t->shape[i];
    if (tc != ((int64_t)1 << bits)) return RAD_E_SHAPE;
    o->tokens = (const int32_t*)tk->data;
    o->codes = (const uint8_t*)c->data;
    o->codes_row_bytes = (ne * bits + 7) / 8;
    o->scale = (const uint16_t*)s->data;
    o->table = (const float*)t->data;
    o->x = (uint16_t*)x->data;
    o->x_pitch = x->stride[0];
    o->M = M;
    o->n_embd = ne;
    o->n_vocab = nv;
    o->bits = bits;
    return RAD_OK;
}

int embed_tq_host_launch(const RadArgs* a, RadStream s) {
    (void)s;
    EmbedTqCall c;
    memset(&c, 0, sizeof c);   /* no stack garbage in the padding: a recorded launch replays byte for byte */
    const int rc = embed_tq_args(a, &c);
    if (rc != RAD_OK) return rc;
    const uint32_t mask = (1u << c.bits) - 1u;
    float v[128];
    for (int64_t m = 0; m < c.M; ++m) {
        uint16_t* xr = c.x + m * c.x_pitch;
        const int64_t id = c.tokens[m];
        if (id < 0 || id >= c.n_vocab) {
            std::memset(xr, 0, (size_t)c.n_embd * 2);
            continue;
        }
        const uint8_t* cr = c.codes + id * c.codes_row_bytes;
        const uint16_t* sr = c.scale + id * (c.n_embd / 128);
        for (int64_t g = 0; g < c.n_embd; g += 128) {
            const float k = bf16f(sr[g / 128]);
            for (int j = 0; j < 128; ++j) {
                const int64_t bit = (g + j) * c.bits;
                uint32_t w = cr[bit >> 3];
                if (((bit & 7) + c.bits) > 8) w |= (uint32_t)cr[(bit >> 3) + 1] << 8;
                v[j] = c.table[(w >> (bit & 7)) & mask] * k;
            }
            for (int h = 1; h < 128; h <<= 1)
                for (int i = 0; i < 128; i += h << 1)
                    for (int j = i; j < i + h; ++j) {
                        const float p = v[j], q = v[j + h];
                        v[j] = p + q;
                        v[j + h] = p - q;
                    }
            for (int j = 0; j < 128; ++j) xr[g + j] = f32bf16(v[j]);
        }
    }
    return RAD_OK;
}

}  /* extern "C" */
