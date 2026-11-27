/* rad_block_inkling2_vision.h -- Inkling 2 Mini's vision tower: the patch embedding, the CLS
 * token, a pre-norm ViT with layer scale and a 2-D rotary, the 2 x 2 patch merge, and the two-layer
 * projector into the text width. Llama.cpp's inkling2v graph is the reference (its build_vit,
 * build_rope_2d and build_patch_merge_permute in mtmd-graph.cpp); tests/mkinkling2vision.py is the
 * float32 reference this block is checked against.
 *
 * WHAT IT COMPUTES, for a grid of gh x gw patches (n = gh * gw, M = n + 1 with the CLS row):
 *
 *   p   = pix_patch(image)                         [n, 3 p p], channel-major, (ph, pw) inside
 *   x   = p @ Wp^T                                 gemm_nt, no bias
 *   x  += pos                                      the learned side x side table on this grid
 *   seq = [cls + cls_pos ; x]                      the CLS row first
 *   seq = LN_pre(seq)
 *   per block:  seq += ls1 * proj(attn(rope(qkv(LN1(seq)))))  full attention over all M rows
 *               seq += ls2 * fc2(gelu(fc1(LN2(seq))))          exact erf GELU
 *   cells = pixtral_merge(seq[1:], merge 2)        [n / 4, 4 width], (channel, dy, dx) columns
 *   out   = proj_out(gelu(proj_in(cells)))         no bias, projection hidden -> text width
 *
 * The rotary is llama's build_rope_2d: each head's first half turns by the row position and its
 * second half by the column position, each over adjacent pairs (2j, 2j + 1) at theta^(-j / (head_dim
 * / 4)). rope's axial mode pairs across the halves instead, so q and k are stored with their head
 * dimensions permuted (see the layout note), and the same rotation is then rope's axial with
 * sections (head_dim / 4, head_dim / 4) and components (row, column). The CLS row is at (0, 0) and
 * patch (y, x) at (y + 1, x + 1) (mtmd's encoder fills them).
 *
 * LAYOUT FIXES, done once when the weights are packed, not per call. The checkpoint's patch weight
 * is HWC-flattened and proj_in's columns are ordered (dy, dx, channel) (the converter does not
 * permute either, and llama's graph reads them so). The existing ops produce channel-major
 * features, pix_patch's `hw` order and pixtral_merge's (channel, dy, dx), so the packed weights are
 * column-permuted to match. The rotary's pairs are adjacent inside each half of a head, and rope's
 * axial pairs (i, i + head_dim / 2), so q and k rows are permuted within each head: dimension 2j
 * goes to j, 2j + 1 to j + head_dim / 2, and the second half's 2j to head_dim / 4 + j and 2j + 1 to
 * 3 head_dim / 4 + j. Applied to both, that leaves every q . k unchanged; v is not permuted.
 *
 * A GRID OTHER THAN THE TABLE'S. The learned table is side x side. An image whose grid differs from
 * it gets the table resampled to its grid first, by posemb_resample (align_corners off, edge taps
 * clamped: llama's bilinear, vLLM's kernel), into a buffer of the workspace's own. A grid equal to
 * the table reads the table itself, so the checkpoint's grid is unchanged bit for bit. The CLS row
 * takes no table position: its position is (0, 0) in the rotary and it is not on the grid.
 * The grid is the workspace's: Workspace::init fixes it, and forward refuses an image of another.
 *
 * HOW IT RUNS. `forward` issues ops through a launch callback (the engine's issue, or a host
 * kernel table in a test), so the same sequence is the one the engine declares. The tensors are
 * f32 throughout, with each op's rounding as libref's. Buffers are sized once by Workspace::init,
 * so the step path allocates nothing.
 */
#ifndef RAD_BLOCK_INKLING2_VISION_H
#define RAD_BLOCK_INKLING2_VISION_H

#include "rad_types.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace rad {
namespace arch {
namespace inkling2v {

struct Geometry {
    int64_t depth = 0;    /* ViT blocks */
    int64_t width = 0;    /* the ViT width; head_dim = width / heads */
    int64_t heads = 0;
    int64_t patch = 0;    /* native patch side */
    int64_t side = 0;     /* the position table's grid side: the image is side * patch px square */
    int64_t merge = 2;    /* the spatial merge */
    int64_t proj = 0;     /* the projector's hidden width (4096 released) */
    int64_t text = 0;     /* the text width the projector writes */
    float   eps = 1e-5f;  /* every LayerNorm */
    float   theta = 1e4f;

    int64_t head_dim() const { return heads > 0 ? width / heads : 0; }
    int64_t ff() const { return 4 * width; }
    int64_t n() const { return side * side; }    /* the table's rows */
};

/* The checkpoint's tensors, f32, in checkpoint layout. Non-owning: the reader keeps them alive. */
struct LayerIn {
    const float *ln1_w, *ln1_b, *ln2_w, *ln2_b, *ls1, *ls2;
    const float *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
    const float *fc_w, *fc_b, *pr_w, *pr_b;
};
struct TowerIn {
    const float *cls, *cls_pos;             /* [width] each */
    const float *patch_w;                   /* [width, 3 p p], HWC-flattened */
    const float *pos;                       /* [side^2, width] */
    const float *pre_w, *pre_b;             /* ln_pre */
    std::vector<LayerIn> layers;
    const float *in_w;                      /* proj_in [proj, 4 width], (dy, dx, channel) columns */
    const float *out_w;                     /* proj_out [text, proj] */
};

/* The weights in the layout the ops read, owned. */
struct Packed {
    Geometry g;
    std::vector<float> cls, patch_w, pos, pre_w, pre_b, in_w, out_w;
    struct Layer {
        std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b, ls1, ls2;
        std::vector<float> qkv_w, qkv_b;    /* q, k, v stacked on rows: [3 width, width] */
        std::vector<float> o_w, o_b, fc_w, fc_b, pr_w, pr_b;
    };
    std::vector<Layer> layers;
};

/* The within-head permutation of q and k: where head dimension d goes. The first half's pairs
 * (2j, 2j + 1) become axial's pairs (j, j + head_dim / 2), the second half's become (head_dim / 4 + j,
 * 3 head_dim / 4 + j), so axial turns each section's j-th pair at the section's ladder. */
inline int64_t rope_perm(int64_t d, int64_t hd) {
    const int64_t half = hd / 2, quarter = hd / 4;
    if (d < half) return d % 2 == 0 ? d / 2 : d / 2 + half;
    const int64_t e = d - half;
    return e % 2 == 0 ? quarter + e / 2 : quarter + half + e / 2;
}

/* Packs the checkpoint into the op layout. Done at declare; the step path only reads Packed. */
inline int pack(const Geometry& g, const TowerIn& in, Packed* out) {
    if (g.depth <= 0 || g.width <= 0 || g.heads <= 0 || g.width % g.heads || g.head_dim() % 4 ||
        g.patch <= 0 || g.side <= 0 || g.merge != 2 || g.side % g.merge || g.proj <= 0 ||
        g.text <= 0 || (int64_t)in.layers.size() != g.depth)
        return RAD_E_SHAPE;
    const int64_t W = g.width, P = g.patch, pp = P * P, cpp = 3 * pp, PH = g.proj;
    out->g = g;
    out->cls.resize(W);
    for (int64_t i = 0; i < W; ++i) out->cls[i] = in.cls[i] + in.cls_pos[i];
    /* The patch weight is HWC inside each patch; pix_patch's `hw` order is channel-major. */
    out->patch_w.resize(W * cpp);
    for (int64_t o = 0; o < W; ++o)
        for (int64_t c = 0; c < 3; ++c)
            for (int64_t ph = 0; ph < P; ++ph)
                for (int64_t pw = 0; pw < P; ++pw)
                    out->patch_w[o * cpp + c * pp + ph * P + pw] =
                        in.patch_w[o * cpp + (ph * P + pw) * 3 + c];
    out->pos.assign(in.pos, in.pos + g.n() * W);
    out->pre_w.assign(in.pre_w, in.pre_w + W);
    out->pre_b.assign(in.pre_b, in.pre_b + W);
    /* pixtral_merge emits (channel, dy, dx) columns; the checkpoint has (dy, dx, channel). */
    out->in_w.resize(PH * 4 * W);
    for (int64_t o = 0; o < PH; ++o)
        for (int64_t q = 0; q < 4; ++q)
            for (int64_t c = 0; c < W; ++c)
                out->in_w[o * 4 * W + c * 4 + q] = in.in_w[o * 4 * W + q * W + c];
    out->out_w.assign(in.out_w, in.out_w + g.text * PH);

    out->layers.assign(g.depth, Packed::Layer{});
    for (int64_t l = 0; l < g.depth; ++l) {
        const LayerIn& s = in.layers[l];
        Packed::Layer& d = out->layers[l];
        d.ln1_w.assign(s.ln1_w, s.ln1_w + W);  d.ln1_b.assign(s.ln1_b, s.ln1_b + W);
        d.ln2_w.assign(s.ln2_w, s.ln2_w + W);  d.ln2_b.assign(s.ln2_b, s.ln2_b + W);
        d.ls1.assign(s.ls1, s.ls1 + W);        d.ls2.assign(s.ls2, s.ls2 + W);
        /* q and k take the rope's within-head permutation (see the layout note); v is stored as is. */
        const int64_t hd = g.head_dim();
        d.qkv_w.resize(3 * W * W);
        d.qkv_b.resize(3 * W);
        for (int64_t r = 0; r < W; ++r) {
            const int64_t dst = (r / hd) * hd + rope_perm(r % hd, hd);
            std::memcpy(&d.qkv_w[(0 * W + dst) * W], s.q_w + r * W, sizeof(float) * W);
            std::memcpy(&d.qkv_w[(1 * W + dst) * W], s.k_w + r * W, sizeof(float) * W);
            d.qkv_b[0 * W + dst] = s.q_b[r];
            d.qkv_b[1 * W + dst] = s.k_b[r];
        }
        std::memcpy(&d.qkv_w[2 * W * W], s.v_w, sizeof(float) * W * W);
        std::memcpy(&d.qkv_b[2 * W], s.v_b, sizeof(float) * W);
        d.o_w.assign(s.o_w, s.o_w + W * W);    d.o_b.assign(s.o_b, s.o_b + W);
        d.fc_w.assign(s.fc_w, s.fc_w + g.ff() * W);  d.fc_b.assign(s.fc_b, s.fc_b + g.ff());
        d.pr_w.assign(s.pr_w, s.pr_w + W * g.ff()); d.pr_b.assign(s.pr_b, s.pr_b + W);
    }
    return RAD_OK;
}

/* The buffers one forward writes, sized once at declare for the grid. */
struct Workspace {
    int64_t gh = 0, gw = 0;                       /* the grid this workspace is sized for */
    int64_t n = 0, M = 0, W = 0, ff = 0, cells = 0, PH = 0, text = 0, cpp = 0;
    std::vector<float> stream[2], h, qkv, att, t1, t2, ffn, emb, patches, merged, hid, hid2, ptab;
    std::vector<int32_t> pos, coord, cu;

    /* The grid is gh x gw patches; both sides must be whole merges. */
    int init(const Geometry& g, int64_t grid_h, int64_t grid_w) {
        if (g.side <= 0 || g.width <= 0 || g.patch <= 0 || g.merge <= 0 || grid_h <= 0 ||
            grid_w <= 0 || grid_h % g.merge || grid_w % g.merge)
            return RAD_E_SHAPE;
        gh = grid_h; gw = grid_w;
        n = gh * gw; M = n + 1; W = g.width; ff = g.ff(); PH = g.proj;
        cells = n / (g.merge * g.merge);
        text = g.text; cpp = 3 * g.patch * g.patch;
        stream[0].assign(M * W, 0.0f);  stream[1].assign(M * W, 0.0f);
        h.assign(M * W, 0.0f);   qkv.assign(M * 3 * W, 0.0f);  att.assign(M * W, 0.0f);
        t1.assign(M * W, 0.0f);  t2.assign(M * W, 0.0f);       ffn.assign(M * ff, 0.0f);
        emb.assign(n * W, 0.0f); patches.assign(n * cpp, 0.0f); ptab.assign(n * W, 0.0f);
        merged.assign(cells * 4 * W, 0.0f); hid.assign(cells * PH, 0.0f); hid2.assign(cells * PH, 0.0f);
        pos.assign(2 * M, 0); coord.assign(4 * n, 0); cu.assign(2, 0);
        return RAD_OK;
    }
};

/* The op launch: the engine's issue or a host kernel table. */
using Issue = int (*)(void* ctx, const char* op, const RadTensor* t, int nt, const RadParam* p, int np);

namespace detail {

inline RadTensor mat(void* d, uint32_t dt, int64_t rows, int64_t cols, int64_t row_stride) {
    RadTensor r{};
    r.data = d;
    r.dtype = dt;
    r.rank = 2;
    r.shape[0] = rows;
    r.shape[1] = cols;
    r.stride[0] = row_stride;
    r.stride[1] = 1;
    return r;
}

/* One op launch's operands and parameters, built on the stack. */
struct Call {
    RadTensor t[6];
    int nt = 0;
    RadParam p[10];
    int np = 0;
    Call& in(const RadTensor& x) { t[nt++] = x; return *this; }
    Call& i(const char* k, int64_t v) { p[np++] = RAD_INT(k, v); return *this; }
    Call& f(const char* k, double v) { p[np++] = RAD_F64(k, v); return *this; }
    Call& s(const char* k, const char* v) { p[np++] = RAD_STR(k, v); return *this; }
    int run(Issue issue, void* ctx, const char* op) const { return issue(ctx, op, t, nt, p, np); }
};

}  // namespace detail

/* One image: `img` is [gh * patch, gw * patch * 3] f32, normalised, on the workspace's grid.
 * `tokens` (n x width) and `out` ((n / merge^2) x text) are written; `tokens` may be null. */
inline int forward(const Packed& P, Workspace& ws, Issue issue, void* ctx, const float* img,
                   int64_t img_h, int64_t img_w, float* tokens, float* out) {
    using detail::Call;
    using detail::mat;
    const Geometry& g = P.g;
    if (!issue || !img || !out) return RAD_E_INVAL;
    const int64_t gh = ws.gh, gw = ws.gw;
    if (img_h != gh * g.patch || img_w != gw * g.patch) return RAD_E_SHAPE;
    const int64_t n = ws.n, M = ws.M, W = ws.W, ff = ws.ff, cells = ws.cells, PH = ws.PH;
    const int64_t hd = g.head_dim(), pp = g.patch * g.patch, cpp = 3 * pp;
    const int64_t side = g.side;
    const uint32_t F32 = RAD_F32, I32 = RAD_I32;
    const float* const pos_tab = P.pos.data();
    int rc = RAD_OK;

    /* Positions and merge coordinates: CLS at (0, 0), patch (y, x) at (y + 1, x + 1) for the rotary;
     * raster grid coordinates for the merge. */
    ws.pos[0] = 0;
    ws.pos[M] = 0;
    for (int64_t i = 0; i < n; ++i) {
        ws.pos[1 + i] = (int32_t)(i % gw + 1);
        ws.pos[M + 1 + i] = (int32_t)(i / gw + 1);
        ws.coord[0 * n + i] = (int32_t)(i / gw);
        ws.coord[1 * n + i] = (int32_t)(i % gw);
        ws.coord[2 * n + i] = (int32_t)gh;
        ws.coord[3 * n + i] = (int32_t)gw;
    }
    ws.cu[0] = 0;
    ws.cu[1] = (int32_t)M;

    /* Patch embedding: pix_patch's rows [gh, gw * cpp] are the patches in raster order. */
    {
        Call c;
        c.in(mat(const_cast<float*>(img), F32, img_h, img_w * 3, img_w * 3))
         .in(mat(ws.patches.data(), F32, gh, gw * cpp, gw * cpp))
         .i("M", img_h).i("p", g.patch).i("C", 3).s("order", "hw").s("dtype", "f32");
        rc = c.run(issue, ctx, "pix_patch");
        if (rc) return rc;
    }
    {
        Call c;
        c.in(mat(ws.patches.data(), F32, n, cpp, cpp))
         .in(mat(const_cast<float*>(P.patch_w.data()), F32, W, cpp, cpp))
         .in(mat(ws.emb.data(), F32, n, W, W))
         .i("M", n).i("N", W).i("K", cpp).s("dtype", "f32");
        rc = c.run(issue, ctx, "gemm_nt");
        if (rc) return rc;
    }

    /* The table on this grid: the checkpoint's own grid reads the table, any other gets it resampled
     * (posemb_resample, align_corners off), once per call into the workspace. */
    const float* pos_grid = pos_tab;
    if (gh != side || gw != side) {
        Call c;
        c.in(mat(const_cast<float*>(pos_tab), F32, side * side, W, W))
         .in(mat(ws.ptab.data(), F32, n, W, W))
         .i("S_h", side).i("S_w", side).i("T_h", gh).i("T_w", gw).i("n", W).s("dtype", "f32");
        rc = c.run(issue, ctx, "posemb_resample");
        if (rc) return rc;
        pos_grid = ws.ptab.data();
    }

    /* Stream: row 0 is the CLS, rows 1.. are the patches plus the position table. */
    float* cur = ws.stream[0].data();
    float* alt = ws.stream[1].data();
    std::memcpy(cur, P.cls.data(), sizeof(float) * W);
    {
        Call c;
        c.in(mat(ws.emb.data(), F32, n, W, W))
         .in(mat(const_cast<float*>(pos_grid), F32, n, W, W))
         .in(mat(cur + W, F32, n, W, W))
         .i("M", n).i("n", W).s("dtype", "f32");
        rc = c.run(issue, ctx, "add");
        if (rc) return rc;
    }
    {
        Call c;
        c.in(mat(cur, F32, M, W, W))
         .in(mat(const_cast<float*>(P.pre_w.data()), F32, 1, W, W))
         .in(mat(const_cast<float*>(P.pre_b.data()), F32, 1, W, W))
         .in(mat(alt, F32, M, W, W))
         .i("M", M).i("n", W).f("eps", g.eps).s("dtype", "f32");
        rc = c.run(issue, ctx, "layernorm");
        if (rc) return rc;
        std::swap(cur, alt);
    }

    for (int64_t l = 0; l < g.depth; ++l) {
        const Packed::Layer& L = P.layers[l];
        {
            Call c;
            c.in(mat(cur, F32, M, W, W))
             .in(mat(const_cast<float*>(L.ln1_w.data()), F32, 1, W, W))
             .in(mat(const_cast<float*>(L.ln1_b.data()), F32, 1, W, W))
             .in(mat(ws.h.data(), F32, M, W, W))
             .i("M", M).i("n", W).f("eps", g.eps).s("dtype", "f32");
            rc = c.run(issue, ctx, "layernorm");
            if (rc) return rc;
        }
        {
            Call c;
            c.in(mat(ws.h.data(), F32, M, W, W))
             .in(mat(const_cast<float*>(L.qkv_w.data()), F32, 3 * W, W, W))
             .in(mat(const_cast<float*>(L.qkv_b.data()), F32, 1, 3 * W, 3 * W))
             .in(RadTensor{})  /* res: the optional residual, absent */
             .in(mat(ws.qkv.data(), F32, M, 3 * W, 3 * W))
             .i("M", M).i("N", 3 * W).i("K", W).s("dtype", "f32");
            rc = c.run(issue, ctx, "gemm_nt_bias");
            if (rc) return rc;
        }
        {
            /* Two sections of head_dim / 4 pairs: the row then the column, each at theta^(-j /
             * (head_dim / 4)), paired across the halves on the permuted q and k. */
            char sec[48];
            std::snprintf(sec, sizeof sec, "%lld %lld", (long long)(hd / 4), (long long)(hd / 4));
            Call c;
            c.in(mat(ws.qkv.data(), F32, M, 3 * W, 3 * W))
             .in(mat(ws.pos.data(), I32, 2, M, M))
             .i("M", M).i("head_dim", hd).i("n_head", g.heads).i("n_head_kv", g.heads)
             .f("theta", g.theta).f("scale", 1.0).s("mode", "axial").s("sections", sec);
            rc = c.run(issue, ctx, "rope");
            if (rc) return rc;
        }
        {
            RadTensor q = mat(ws.qkv.data(), F32, M, W, 3 * W);
            RadTensor k = mat(ws.qkv.data() + W, F32, M, W, 3 * W);
            RadTensor v = mat(ws.qkv.data() + 2 * W, F32, M, W, 3 * W);
            Call c;
            c.in(q).in(k).in(v)
             .in(RadTensor{ ws.cu.data(), I32, 1, {2}, {1} })
             .in(mat(ws.att.data(), F32, M, W, W))
             .i("M", M).i("head_dim", hd).i("gqa", 1).i("causal", 0).s("dtype", "f32")
             .f("scale", 1.0 / std::sqrt((double)hd));
            rc = c.run(issue, ctx, "attn_dense");
            if (rc) return rc;
        }
        {
            Call c;
            c.in(mat(ws.att.data(), F32, M, W, W))
             .in(mat(const_cast<float*>(L.o_w.data()), F32, W, W, W))
             .in(mat(const_cast<float*>(L.o_b.data()), F32, 1, W, W))
             .in(RadTensor{})  /* res: the optional residual, absent */
             .in(mat(ws.t1.data(), F32, M, W, W))
             .i("M", M).i("N", W).i("K", W).s("dtype", "f32");
            rc = c.run(issue, ctx, "gemm_nt_bias");
            if (rc) return rc;
        }
        {
            Call c;
            c.in(mat(ws.t1.data(), F32, M, W, W))
             .in(mat(const_cast<float*>(L.ls1.data()), F32, 1, W, W))
             .in(mat(ws.t2.data(), F32, M, W, W))
             .i("M", M).i("n", W).s("dtype", "f32");
            rc = c.run(issue, ctx, "mul_weight");
            if (rc) return rc;
        }
        {
            Call c;
            c.in(mat(cur, F32, M, W, W)).in(mat(ws.t2.data(), F32, M, W, W))
             .in(mat(alt, F32, M, W, W))
             .i("M", M).i("n", W).s("dtype", "f32");
            rc = c.run(issue, ctx, "add");
            if (rc) return rc;
            std::swap(cur, alt);
        }
        {
            Call c;
            c.in(mat(cur, F32, M, W, W))
             .in(mat(const_cast<float*>(L.ln2_w.data()), F32, 1, W, W))
             .in(mat(const_cast<float*>(L.ln2_b.data()), F32, 1, W, W))
             .in(mat(ws.h.data(), F32, M, W, W))
             .i("M", M).i("n", W).f("eps", g.eps).s("dtype", "f32");
            rc = c.run(issue, ctx, "layernorm");
            if (rc) return rc;
        }
        {
            Call c;
            c.in(mat(ws.h.data(), F32, M, W, W))
             .in(mat(const_cast<float*>(L.fc_w.data()), F32, ff, W, W))
             .in(mat(const_cast<float*>(L.fc_b.data()), F32, 1, ff, ff))
             .in(RadTensor{})  /* res: the optional residual, absent */
             .in(mat(ws.ffn.data(), F32, M, ff, ff))
             .i("M", M).i("N", ff).i("K", W).s("dtype", "f32").s("act", "gelu");
            rc = c.run(issue, ctx, "gemm_nt_bias");
            if (rc) return rc;
        }
        {
            Call c;
            c.in(mat(ws.ffn.data(), F32, M, ff, ff))
             .in(mat(const_cast<float*>(L.pr_w.data()), F32, W, ff, ff))
             .in(mat(const_cast<float*>(L.pr_b.data()), F32, 1, W, W))
             .in(RadTensor{})  /* res: the optional residual, absent */
             .in(mat(ws.t1.data(), F32, M, W, W))
             .i("M", M).i("N", W).i("K", ff).s("dtype", "f32");
            rc = c.run(issue, ctx, "gemm_nt_bias");
            if (rc) return rc;
        }
        {
            Call c;
            c.in(mat(ws.t1.data(), F32, M, W, W))
             .in(mat(const_cast<float*>(L.ls2.data()), F32, 1, W, W))
             .in(mat(ws.t2.data(), F32, M, W, W))
             .i("M", M).i("n", W).s("dtype", "f32");
            rc = c.run(issue, ctx, "mul_weight");
            if (rc) return rc;
        }
        {
            Call c;
            c.in(mat(cur, F32, M, W, W)).in(mat(ws.t2.data(), F32, M, W, W))
             .in(mat(alt, F32, M, W, W))
             .i("M", M).i("n", W).s("dtype", "f32");
            rc = c.run(issue, ctx, "add");
            if (rc) return rc;
            std::swap(cur, alt);
        }
    }

    /* The CLS row leaves; the patch rows merge 2 x 2 into the projector's cells. */
    if (tokens) std::memcpy(tokens, cur + W, sizeof(float) * n * W);
    {
        Call c;
        c.in(mat(cur + W, F32, n, W, W))
         .in(mat(ws.coord.data(), I32, 4, n, n))
         .in(mat(ws.merged.data(), F32, cells, 4 * W, 4 * W))
         .i("M", n).i("n", W).i("merge", g.merge).s("dtype", "f32");
        rc = c.run(issue, ctx, "pixtral_merge");
        if (rc) return rc;
    }
    {
        Call c;
        c.in(mat(ws.merged.data(), F32, cells, 4 * W, 4 * W))
         .in(mat(const_cast<float*>(P.in_w.data()), F32, PH, 4 * W, 4 * W))
         .in(mat(ws.hid.data(), F32, cells, PH, PH))
         .i("M", cells).i("N", PH).i("K", 4 * W).s("dtype", "f32");
        rc = c.run(issue, ctx, "gemm_nt");
        if (rc) return rc;
    }
    {
        Call c;
        c.in(mat(ws.hid.data(), F32, cells, PH, PH)).in(mat(ws.hid2.data(), F32, cells, PH, PH))
         .i("M", cells).i("n", PH).s("dtype", "f32");
        rc = c.run(issue, ctx, "gelu");
        if (rc) return rc;
    }
    {
        Call c;
        c.in(mat(ws.hid2.data(), F32, cells, PH, PH))
         .in(mat(const_cast<float*>(P.out_w.data()), F32, g.text, PH, PH))
         .in(mat(out, F32, cells, g.text, g.text))
         .i("M", cells).i("N", g.text).i("K", PH).s("dtype", "f32");
        rc = c.run(issue, ctx, "gemm_nt");
        if (rc) return rc;
    }
    return RAD_OK;
}

}  // namespace inkling2v
}  // namespace arch
}  // namespace rad

#endif  /* RAD_BLOCK_INKLING2_VISION_H */
