/* rad_block_inkling2_tower.h -- Inkling 2's vision tower as the engine serves it: declared beside
 * the text backbone (arch/inkling2), run on an encoder pass (RadBatch::enc), its rows read by the
 * core and scattered over the placeholders of a later step. rad_block_inkling2_vision.h is the same
 * tower over host f32 buffers (the golden's test); this one reads the container's bf16 weights and
 * the pass the core stages.
 *
 * Per pass, the processor (core/mm, Family::Inkling2) gives each picture one zero row at
 * coordinate (-1, -1) ahead of its patches, which are (y, x, channel) in merge-block order:
 *
 *   x  = rows @ Wp^T                             the patch embedding, bias-less; the class row
 *                                                stays zero
 *   x += pos(table, row, col), or cls            grid_embed_hw: the 32 x 32 table resampled per
 *                                                axis onto the picture's patch grid (torch's
 *                                                bilinear, align_corners=False, clamped); the class
 *                                                row takes class_embedding as its prefix row
 *   x[class rows] += cls_pos                     table_add, one row a picture
 *   x  = LN_pre(x)
 *   depth times (pre-norm, layer scale):
 *     x += ls1 * proj(attn(rope(qkv(LN1(x)))))   full attention within a picture, class row
 *                                                included; rope's axial at theta 1e4, (column + 1,
 *                                                row + 1), the class row (0, 0)
 *     x += ls2 * c_proj(gelu(c_fc(LN2(x))))      exact GELU
 *   g  = each picture's patch rows               gather_rows: the class row dropped; a merge cell's
 *                                                four patches are consecutive, so g read as
 *                                                [P / 4, 4 width] is the reference's 2 x 2 merge,
 *                                                columns (dy, dx, channel)
 *   out = proj_out(gelu(proj_in(g)))             both bias-less (projector_bias false)
 *
 * The reference's rotary turns the adjacent pairs (2j, 2j + 1) of each half of a head, the first
 * half by the column and the second by the row, at theta^(-j / (head_dim / 4)) (vLLM's
 * inkling2/nvidia/ops/vision.py, llama.cpp's build_rope_2d). rope's `axial` pairs (i, i + head_dim
 * / 2) instead, so q and k are stored with their rows permuted within each head (row-piece maps,
 * done at convert: the container holds the permuted rows), which turns the same angles onto the
 * same pairs. Every q . k is unchanged; v is not permuted. rad_block_inkling2_vision.h's rope_perm
 * is the permutation.
 *
 * Weights are replicated per-request bf16 (RAD_ACCESS_PER_REQUEST), the blocks one group each from
 * `first_group`, as the other towers in arch/ do. Activations are pass-sized (RadBuildCtx::
 * max_enc_patches rows), except in a sizing declare (shape_probe), where they follow the probe's
 * step and the attention's longest segment stays at the real pass.
 */
#ifndef RAD_BLOCK_INKLING2_TOWER_H
#define RAD_BLOCK_INKLING2_TOWER_H

#include "rad_arch.h"
#include "rad_block_inkling2_vision.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace rad {
namespace arch {

struct Inkling2Tower {
    int64_t depth = 0, width = 0, heads = 0, head_dim = 0, ff = 0, patch = 0, merge = 0;
    int64_t side = 0, proj = 0, n_embd = 0, patch_dim = 0, P = 0, R = 0;
    double  eps = 1e-5;
    double  theta = 1e4;
    bool    on = false;
    char    sec[48] = {};

    struct Layer {
        rad_weight n1w = 0, n1b = 0, n2w = 0, n2b = 0, ls1 = 0, ls2 = 0;
        rad_weight qkv_w = 0, qkv_b = 0, o_w = 0, o_b = 0, fc_w = 0, fc_b = 0, pr_w = 0, pr_b = 0;
        rad_op ln1 = 0, qkv = 0, rope = 0, attn = 0, o = 0, ls1_op = 0, add1 = 0;
        rad_op ln2 = 0, fc = 0, pr = 0, ls2_op = 0, add2 = 0;
    };
    std::vector<Layer> L;
    rad_weight w_patch = 0, w_pos = 0, w_cls = 0, w_clspos = 0, w_pre = 0, w_preb = 0;
    rad_weight w_in = 0, w_out = 0;
    rad_op op_patch = 0, op_pos = 0, op_clspos = 0, op_lpre = 0, op_order = 0, op_seq = 0;
    rad_op op_gather = 0, op_in = 0, op_gelu = 0, op_out = 0;
    rad_buf b_x = 0, b_s = 0, b_h = 0, b_qkv = 0, b_att = 0, b_t = 0, b_ff = 0;
    rad_buf b_idx = 0, b_inv = 0, b_pos = 0, b_cuw = 0, b_seq = 0;
    rad_buf b_g = 0, b_m1 = 0, b_m2 = 0, b_out = 0;

    /* The container's vision_config names this tower. */
    static bool present(const RadModelMeta* m) {
        return !std::strcmp(rad_meta_gets(m, "vision_config.model_type", ""), "inkling2_vision") &&
               rad_meta_geti(m, "vision_config.num_hidden_layers", 0) > 0;
    }

    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, int first_group);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int Inkling2Tower::declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx,
                                  Names& nm, int64_t model_embd, int first_group) {
    on = false;
    if (!present(m) || ctx->max_enc_patches <= 0) return RAD_OK;

    depth  = rad_meta_geti(m, "vision_config.num_hidden_layers", 0);
    width  = rad_meta_geti(m, "vision_config.hidden_size", 0);
    heads  = rad_meta_geti(m, "vision_config.num_attention_heads", 0);
    ff     = (int64_t)std::llround(rad_meta_getf(m, "vision_config.mlp_ratio", 0.0) * (double)width);
    patch  = rad_meta_geti(m, "vision_config.native_patch_size", 0);
    merge  = rad_meta_geti(m, "vision_config.adapter_merge_size", 0);
    side   = rad_meta_geti(m, "vision_config.posemb_grid_size", 0);
    proj   = rad_meta_geti(m, "vision_config.projection_hidden_dim", 0);
    eps    = rad_meta_getf(m, "vision_config.layer_norm_eps", 1e-5);
    n_embd = model_embd;
    const int64_t text = rad_meta_geti(m, "vision_config.text_hidden_size", model_embd);
    if (depth <= 0 || depth > 128 || width <= 0 || heads <= 0 || width % heads || ff <= 0 ||
        patch <= 0 || merge <= 0 || side <= 0 || side > 512 || proj <= 0 || text != model_embd) {
        fprintf(stderr, "radiance: inkling2: the container's vision_config is incomplete or "
                        "inconsistent (layers %lld, hidden %lld, heads %lld, ff %lld, patch %lld, "
                        "merge %lld, position grid %lld, projector %lld, text width %lld against "
                        "the model's %lld)\n", (long long)depth, (long long)width,
                (long long)heads, (long long)ff, (long long)patch, (long long)merge,
                (long long)side, (long long)proj, (long long)text, (long long)model_embd);
        return RAD_E_INVAL;
    }
    /* What is served of the config: the 2 x 2 merge (a cell is four consecutive rows) and the
     * bias-less projector the release states. */
    if (merge != 2 || rad_meta_geti(m, "vision_config.projector_bias", 0) != 0) {
        fprintf(stderr, "radiance: inkling2: this vision tower states merge %lld and projector_bias "
                        "%lld; the 2 x 2 merge and a bias-less projector are served\n",
                (long long)merge, (long long)rad_meta_geti(m, "vision_config.projector_bias", 0));
        return RAD_E_UNSUPPORTED;
    }
    head_dim = width / heads;
    if (head_dim % 4) return RAD_E_SHAPE;
    patch_dim = 3 * patch * patch;
    const int64_t mm = merge * merge, mw = mm * width;
    std::snprintf(sec, sizeof sec, "%lld %lld", (long long)(head_dim / 4),
                  (long long)(head_dim / 4));

    /* A pass's rows, class rows included; its output rows are at most a quarter of them. */
    P = ctx->max_enc_patches;
    const int64_t max_seg = P;
    if (ctx->shape_probe) {
        const int64_t r = std::min<int64_t>(P, ctx->max_tok);
        P = std::max<int64_t>(mm, r / mm * mm);
    }
    R = P / mm;
    if (R <= 0) return RAD_E_SHAPE;
    const int ACC = RAD_ACCESS_PER_REQUEST;

    const char* vb = "model.visual.backbone";
    if (map_copy(b, nm.f("v.patch.weight"), nm.ckpt("%s.embeddings.patch_embedding.weight", vb)) < 0)
        return RAD_E_INVAL;
    /* Whether the tower is here at all: its stem's encoding, which convert answers from the
     * checkpoint and load from the container alike. A container converted without it (the text
     * recipe's) serves text; the maps left behind are inert. */
    {
        RadEncoding enc;
        if (rad_weight_encoding(b, nm.f("v.patch.weight"), &enc, nullptr, nullptr) != RAD_OK) {
            rad_note(b, "inkling2: vision_config names a tower its weights are not here for; "
                        "serving text");
            return RAD_OK;
        }
    }
    auto bw = [&](const char* dn, const char* src, std::initializer_list<int64_t> shp,
                  RadWeightGroup g) -> rad_weight {
        if (map_copy(b, dn, src) < 0) return 0;
        RadEncoding e{};
        if (weight_enc(b, dn, &e) &&
            !(rad_enc_is(&e, "plain") && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16)) {
            fprintf(stderr, "radiance: inkling2 vision: '%s' must be bf16\n", dn);
            return 0;
        }
        return decl_w(b, dn, RAD_BF16, shp, ACC, RAD_SHARD_NONE, g);
    };
    w_patch = decl_w(b, nm.f("v.patch.weight"), RAD_BF16, { width, patch_dim }, ACC,
                     RAD_SHARD_NONE, grp_model());
    w_pos    = bw(nm.f("v.pos.weight"), nm.ckpt("%s.embeddings.position_embedding", vb),
                  { side * side, width }, grp_model());
    w_cls    = bw(nm.f("v.cls.weight"), nm.ckpt("%s.embeddings.class_embedding", vb), { width },
                  grp_model());
    w_clspos = bw(nm.f("v.cls_pos.weight"), nm.ckpt("%s.embeddings.cls_position_embedding", vb),
                  { width }, grp_model());
    w_pre    = bw(nm.f("v.ln_pre.weight"), nm.ckpt("%s.ln_pre.weight", vb), { width }, grp_model());
    w_preb   = bw(nm.f("v.ln_pre.bias"), nm.ckpt("%s.ln_pre.bias", vb), { width }, grp_model());

    /* q | k | v joined into one projection, q's and k's rows permuted within each head onto
     * rope's axial pairing (see the header), one row piece a row; v whole. */
    std::vector<int64_t> src_row((size_t)width);
    for (int64_t r = 0; r < width; ++r) {
        const int64_t h = r / head_dim, d = r % head_dim;
        src_row[(size_t)(h * head_dim + inkling2v::rope_perm(d, head_dim))] = r;
    }
    auto qkv_map = [&](const char* dn, const char* qn, const char* kn, const char* vn) -> int {
        for (const char* srcn : { qn, kn }) {
            for (int64_t r = 0; r < width; r += 8) {
                RadNameMap mp = map_base(dn, RAD_MAP_ROWS);
                for (int64_t k = r; k < std::min<int64_t>(r + 8, width); ++k) {
                    mp.src[mp.n_src] = srcn;
                    mp.src_index[mp.n_src] = (int)src_row[(size_t)k];
                    mp.src_rows[mp.n_src] = 1;
                    ++mp.n_src;
                }
                if (rad_decl_name_map(b, &mp) < 0) return RAD_E_INVAL;
            }
        }
        return map_rows(b, dn, { { vn, 0, width } });
    };
    L.assign((size_t)depth, Layer{});
    for (int64_t i = 0; i < depth; ++i) {
        Layer& l = L[(size_t)i];
        const RadWeightGroup g = grp_layer(first_group + (int)i);
        const int li = (int)i;
        auto src = [&](const char* s) { return nm.ckpt("%s.layers.%d.%s", vb, li, s); };
        auto dn  = [&](const char* s) { return nm.f("v.blk.%d.%s", li, s); };
        l.n1w = bw(dn("ln_1.weight"), src("ln_1.weight"), { width }, g);
        l.n1b = bw(dn("ln_1.bias"), src("ln_1.bias"), { width }, g);
        l.n2w = bw(dn("ln_2.weight"), src("ln_2.weight"), { width }, g);
        l.n2b = bw(dn("ln_2.bias"), src("ln_2.bias"), { width }, g);
        l.ls1 = bw(dn("ls_1.gamma"), src("ls_1.gamma"), { width }, g);
        l.ls2 = bw(dn("ls_2.gamma"), src("ls_2.gamma"), { width }, g);
        {
            /* nm.ckpt returns a scratch string per call: keep the three names apart. */
            const std::string q = src("self_attn.q_proj.weight"), k = src("self_attn.k_proj.weight"),
                              v = src("self_attn.v_proj.weight");
            const std::string qb = src("self_attn.q_proj.bias"), kb = src("self_attn.k_proj.bias"),
                              vbn = src("self_attn.v_proj.bias");
            const std::string wn = dn("qkv.weight"), bn = dn("qkv.bias");
            RAD_ARCH_TRY(qkv_map(wn.c_str(), q.c_str(), k.c_str(), v.c_str()));
            RAD_ARCH_TRY(qkv_map(bn.c_str(), qb.c_str(), kb.c_str(), vbn.c_str()));
            l.qkv_w = decl_w(b, wn.c_str(), RAD_BF16, { 3 * width, width }, ACC, RAD_SHARD_NONE, g);
            l.qkv_b = decl_w(b, bn.c_str(), RAD_BF16, { 3 * width }, ACC, RAD_SHARD_NONE, g);
        }
        l.o_w  = bw(dn("out_proj.weight"), src("self_attn.out_proj.weight"), { width, width }, g);
        l.o_b  = bw(dn("out_proj.bias"), src("self_attn.out_proj.bias"), { width }, g);
        l.fc_w = bw(dn("c_fc.weight"), src("mlp.c_fc.weight"), { ff, width }, g);
        l.fc_b = bw(dn("c_fc.bias"), src("mlp.c_fc.bias"), { ff }, g);
        l.pr_w = bw(dn("c_proj.weight"), src("mlp.c_proj.weight"), { width, ff }, g);
        l.pr_b = bw(dn("c_proj.bias"), src("mlp.c_proj.bias"), { width }, g);
        if (!l.n1w || !l.n1b || !l.n2w || !l.n2b || !l.ls1 || !l.ls2 || !l.qkv_w || !l.qkv_b ||
            !l.o_w || !l.o_b || !l.fc_w || !l.fc_b || !l.pr_w || !l.pr_b)
            return RAD_E_INVAL;
    }
    w_in  = bw(nm.f("v.proj_in.weight"), nm.ckpt("model.visual.proj_in.weight"), { proj, mw },
               grp_model());
    w_out = bw(nm.f("v.proj_out.weight"), nm.ckpt("model.visual.proj_out.weight"),
               { n_embd, proj }, grp_model());
    if (!w_patch || !w_pos || !w_cls || !w_clspos || !w_pre || !w_preb || !w_in || !w_out)
        return RAD_E_INVAL;

    /* ---- activations. `g` holds the gathered patch rows at `width`, read by proj_in as R rows
     * of mm * width (the merge's view, no copy). */
    b_x   = decl_b(b, nm.f("v.x"),   RAD_BF16, { P, width });
    b_s   = decl_b(b, nm.f("v.s"),   RAD_BF16, { P, width });
    b_h   = decl_b(b, nm.f("v.h"),   RAD_BF16, { P, width });
    b_qkv = decl_b(b, nm.f("v.qkv"), RAD_BF16, { P, 3 * width });
    b_att = decl_b(b, nm.f("v.att"), RAD_BF16, { P, width });
    b_t   = decl_b(b, nm.f("v.t"),   RAD_BF16, { P, width });
    b_ff  = decl_b(b, nm.f("v.ff"),  RAD_BF16, { P, ff });
    b_idx = decl_b(b, nm.f("v.idx"), RAD_I32,  { P });
    b_inv = decl_b(b, nm.f("v.inv"), RAD_I32,  { P });
    b_pos = decl_b(b, nm.f("v.pos"), RAD_I32,  { 2, P });
    b_cuw = decl_b(b, nm.f("v.cuw"), RAD_I32,  { P + 1 });
    b_seq = decl_b(b, nm.f("v.seq"), RAD_I32,  { P });
    b_g   = decl_b(b, nm.f("v.g"),   RAD_BF16, { R, mw });
    b_m1  = decl_b(b, nm.f("v.m1"),  RAD_BF16, { R, proj });
    b_m2  = decl_b(b, nm.f("v.m2"),  RAD_BF16, { R, proj });
    b_out = decl_b(b, nm.f("v.out"), RAD_BF16, { R, n_embd });
    if (!b_x || !b_s || !b_h || !b_qkv || !b_att || !b_t || !b_ff || !b_idx || !b_inv || !b_pos ||
        !b_cuw || !b_seq || !b_g || !b_m1 || !b_m2 || !b_out)
        return RAD_E_INVAL;

    auto gemm_b = [&](int64_t N, int64_t K, const char* a, rad_weight w, rad_weight bias) {
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16"), RAD_STR("act", a)),
                      RAD_WEIGHTS(w, bias));
    };
    auto gemm = [&](int64_t rows, int64_t N, int64_t K, rad_weight w) {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto lnorm = [&](rad_weight w, rad_weight bias) {
        return RAD_OP(b, "layernorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width), RAD_F64("eps", eps),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w, bias));
    };
    auto ew = [&](const char* op, rad_weight w) {
        return w ? RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                            RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w))
                 : RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                            RAD_STR("dtype", "bf16")),
                          RAD_NOWEIGHTS);
    };

    op_patch  = rw(b, gemm(P, width, patch_dim, w_patch), {}, { b_x });
    op_pos    = rw(b, RAD_OP(b, "grid_embed_hw",
                             RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                        RAD_INT("side", side), RAD_INT("n_prefix", 1),
                                        RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(w_pos, w_cls)),
                  { b_x }, { b_x });
    op_clspos = rw(b, RAD_OP(b, "table_add",
                             RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width), RAD_INT("L", 1),
                                        RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(w_clspos)),
                   { b_x }, { b_x });
    op_lpre   = rw(b, lnorm(w_pre, w_preb), { b_x }, { b_s });
    /* window 0: the identity order, whose positions are (column + 1, row + 1) -- the class row's
     * (-1, -1) coordinate gives it the reference's (0, 0). */
    op_order  = rw(b, RAD_OP(b, "window_order",
                             RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("window", 0)),
                             RAD_NOWEIGHTS),
                   {}, { b_idx, b_inv, b_pos, b_cuw });
    op_seq    = rw(b, RAD_OP(b, "iota",
                             RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("start", 0), RAD_INT("step", 1)),
                             RAD_NOWEIGHTS),
                   {}, { b_seq });
    for (Layer& l : L) {
        l.ln1  = rw(b, lnorm(l.n1w, l.n1b), { b_s }, { b_h });
        l.qkv  = rw(b, gemm_b(3 * width, width, "none", l.qkv_w, l.qkv_b), { b_h }, { b_qkv });
        l.rope = rw(b, RAD_OP(b, "rope",
                              RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", head_dim),
                                         RAD_INT("n_head", heads), RAD_INT("n_head_kv", heads),
                                         RAD_F64("theta", theta), RAD_F64("scale", 1.0),
                                         RAD_STR("mode", "axial"),
                                         RAD_INT("rotary_dim", head_dim),
                                         RAD_STR("sections", sec)),
                              RAD_NOWEIGHTS),
                    { b_qkv, b_pos }, { b_qkv });
        l.attn = rw(b, RAD_OP(b, "attn_dense",
                              RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", head_dim),
                                         RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                         RAD_STR("dtype", "bf16"),
                                         RAD_F64("scale", 1.0 / std::sqrt((double)head_dim)),
                                         RAD_INT("max_seqlen", max_seg)),
                              RAD_NOWEIGHTS),
                    { b_qkv }, { b_att });
        l.o      = rw(b, gemm_b(width, width, "none", l.o_w, l.o_b), { b_att }, { b_t });
        l.ls1_op = rw(b, ew("mul_weight", l.ls1), { b_t }, { b_att });
        l.add1   = rw(b, ew("add", 0), { b_s, b_att }, { b_s });
        l.ln2    = rw(b, lnorm(l.n2w, l.n2b), { b_s }, { b_h });
        l.fc     = rw(b, gemm_b(ff, width, "gelu", l.fc_w, l.fc_b), { b_h }, { b_ff });
        l.pr     = rw(b, gemm_b(width, ff, "none", l.pr_w, l.pr_b), { b_ff }, { b_t });
        l.ls2_op = rw(b, ew("mul_weight", l.ls2), { b_t }, { b_att });
        l.add2   = rw(b, ew("add", 0), { b_s, b_att }, { b_s });
    }
    op_gather = rw(b, RAD_OP(b, "gather_rows",
                             RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { b_s, b_seq }, { b_g });
    op_in   = rw(b, gemm(R, proj, mw, w_in), { b_g }, { b_m1 });
    op_gelu = rw(b, RAD_OP(b, "gelu",
                           RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", proj),
                                      RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                 { b_m1 }, { b_m2 });
    op_out  = rw(b, gemm(R, n_embd, proj, w_out), { b_m2 }, { b_out });

    if (!ctx->shape_probe) {
        RadEncoderDecl e{};
        e.name        = "vision";
        e.modalities  = RAD_MM_IMAGE;
        e.patch_dim   = patch_dim;
        e.merge       = mm;
        e.max_patches = ctx->max_enc_patches;
        e.n_embd      = n_embd;
        e.out         = b_out;
        RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        rad_note(b, "inkling2 vision tower: %lld blocks of %lld (%lld heads x %lld), ff %lld, patch "
                    "%lld, %lld x %lld position table, one class row a picture, merge %lld; "
                    "projector %lld -> %lld -> %lld; %lld rows a pass",
                 (long long)depth, (long long)width, (long long)heads, (long long)head_dim,
                 (long long)ff, (long long)patch, (long long)side, (long long)side,
                 (long long)merge, (long long)mw, (long long)proj, (long long)n_embd,
                 (long long)P);
    }
    on = true;
    return RAD_OK;
}

inline void Inkling2Tower::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t Pn = batch->enc_n_patch, S = batch->enc_n_seg;
    const int64_t mm = merge * merge;
    /* Each picture is its class row and h x w patches, whole merge cells. */
    int64_t rows = 0, out_rows = 0;
    bool whole = on && Pn > 0 && Pn <= P && S > 0 && batch->enc_seg_h && batch->enc_seg_w &&
                 batch->enc_pixels && batch->enc_coord && batch->enc_cu;
    for (int64_t s = 0; whole && s < S; ++s) {
        const int64_t hw = (int64_t)batch->enc_seg_h[s] * batch->enc_seg_w[s];
        whole = hw > 0 && hw % mm == 0;
        rows += 1 + hw;
        out_rows += hw / mm;
    }
    if (!whole || rows != Pn || out_rows > R) {
        rad_step_fail(c, "inkling2: an encoder pass this program's vision tower cannot serve: no "
                         "tower, more rows than it was declared for, or pictures that are not a "
                         "class row and whole merge cells");
        return;
    }
    const RadOperand pix = praw2(batch->enc_pixels, RAD_BF16, Pn, patch_dim);
    const RadOperand crd = praw2(batch->enc_coord, RAD_I32, 4, Pn);
    const RadOperand cu  = praw(batch->enc_cu, RAD_I32, S + 1);
    RadOperand pos = RAD_B(b_pos);
    pos.rows = 2;
    pos.cols = Pn;
    RadOperand cuw = RAD_B(b_cuw);
    cuw.rows = S + 1;

    RAD_ISSUE_N(c, op_patch, Pn, pix, RAD_W(w_patch), brows(b_x, Pn));
    RAD_ISSUE_N(c, op_pos, Pn, brows(b_x, Pn), RAD_W(w_pos), crd, RAD_W(w_cls));
    for (int64_t s = 0, r0 = 0; s < S; ++s) {
        RAD_ISSUE_N(c, op_clspos, 1, brow_slice(b_x, r0, 1, width), RAD_W(w_clspos),
                    brow_slice(b_x, r0, 1, width));
        r0 += 1 + (int64_t)batch->enc_seg_h[s] * batch->enc_seg_w[s];
    }
    RAD_ISSUE_N(c, op_lpre, Pn, brows(b_x, Pn), RAD_W(w_pre), RAD_W(w_preb), brows(b_s, Pn));
    RAD_ISSUE_N(c, op_order, Pn, crd, cu, brows(b_idx, Pn), brows(b_inv, Pn), pos, cuw);
    RAD_ISSUE_N(c, op_seq, Pn, brows(b_seq, Pn));
    for (const Layer& l : L) {
        RAD_ISSUE_N(c, l.ln1, Pn, brows(b_s, Pn), RAD_W(l.n1w), RAD_W(l.n1b), brows(b_h, Pn));
        RAD_ISSUE_N(c, l.qkv, Pn, brows(b_h, Pn), RAD_W(l.qkv_w), RAD_W(l.qkv_b), RAD_NONE,
                    brows(b_qkv, Pn));
        RAD_ISSUE_N(c, l.rope, Pn, brows(b_qkv, Pn), pos);
        RAD_ISSUE_N(c, l.attn, Pn, bcol(b_qkv, 0, width, Pn), bcol(b_qkv, width, width, Pn),
                    bcol(b_qkv, 2 * width, width, Pn), cu, brows(b_att, Pn));
        RAD_ISSUE_N(c, l.o, Pn, brows(b_att, Pn), RAD_W(l.o_w), RAD_W(l.o_b), RAD_NONE,
                    brows(b_t, Pn));
        RAD_ISSUE_N(c, l.ls1_op, Pn, brows(b_t, Pn), RAD_W(l.ls1), brows(b_att, Pn));
        RAD_ISSUE_N(c, l.add1, Pn, brows(b_s, Pn), brows(b_att, Pn), brows(b_s, Pn));
        RAD_ISSUE_N(c, l.ln2, Pn, brows(b_s, Pn), RAD_W(l.n2w), RAD_W(l.n2b), brows(b_h, Pn));
        RAD_ISSUE_N(c, l.fc, Pn, brows(b_h, Pn), RAD_W(l.fc_w), RAD_W(l.fc_b), RAD_NONE,
                    brows(b_ff, Pn));
        RAD_ISSUE_N(c, l.pr, Pn, brows(b_ff, Pn), RAD_W(l.pr_w), RAD_W(l.pr_b), RAD_NONE,
                    brows(b_t, Pn));
        RAD_ISSUE_N(c, l.ls2_op, Pn, brows(b_t, Pn), RAD_W(l.ls2), brows(b_att, Pn));
        RAD_ISSUE_N(c, l.add2, Pn, brows(b_s, Pn), brows(b_att, Pn), brows(b_s, Pn));
    }
    /* Each picture's patch rows, behind its class row, into `g` at width: picture s's rows
     * [r0 + 1, r0 + 1 + hw) of the stream, named by a slice of the identity index. */
    for (int64_t s = 0, r0 = 0, o0 = 0; s < S; ++s) {
        const int64_t hw = (int64_t)batch->enc_seg_h[s] * batch->enc_seg_w[s];
        RAD_ISSUE_N(c, op_gather, hw, brows(b_s, Pn), brow_slice(b_seq, r0 + 1, hw, 1),
                    brow_slice(b_g, o0, hw / mm, mm * width));
        r0 += 1 + hw;
        o0 += hw / mm;
    }
    RAD_ISSUE_N(c, op_in, out_rows, brows(b_g, out_rows), RAD_W(w_in), brows(b_m1, out_rows));
    RAD_ISSUE_N(c, op_gelu, out_rows, brows(b_m1, out_rows), brows(b_m2, out_rows));
    RAD_ISSUE_N(c, op_out, out_rows, brows(b_m2, out_rows), RAD_W(w_out), brows(b_out, out_rows));
}

}  /* namespace arch */
}  /* namespace rad */

#endif  /* RAD_BLOCK_INKLING2_TOWER_H */
