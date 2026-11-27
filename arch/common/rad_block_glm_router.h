/* GLM noaux_tc router. Correction bias changes selection only; mixture weights
 * come from the uncorrected probabilities -- sigmoid, or the score Config::score names
 * (DeepSeek-V4's sqrt(softplus)). All routing arithmetic is F32. With `groups`
 * above 1 the selection is group-limited (router_sigmoid_group_topk, Ling-3.0's 8 groups, 4 kept). */
#ifndef RAD_BLOCK_GLM_ROUTER_H
#define RAD_BLOCK_GLM_ROUTER_H
#include "rad_glm_tp.h"
#include <cmath>

namespace rad {
namespace arch {
struct GlmRouter {
    struct Config {
        int64_t experts = 288, top_k = 8, rows = 32, groups = 1, topk_groups = 1;
        int norm = 1;
        double route_scale = 2.5;
        /* The score function by name (router_scored_topk: "sqrtsoftplus" for DeepSeek-V4); null is
         * GLM's sigmoid (router_sigmoid_topk). */
        const char* score = nullptr;
        /* Image-span routing: with `vl` set, `Src::correction_vl` (bias_vl) is the correction on
         * rows whose mask (RadBatch::vl_mask) is 1 and `correction` on the rest
         * (router_scored_topk_vl). Selection only, weights as on the text rows. */
        int vl = 0;
    };
    struct Src {
        const char *weight = nullptr, *correction = nullptr;
        const char *correction_vl = nullptr;
    };
    struct Wire {
        rad_buf input = 0, logits = 0, ids = 0, weights = 0;
    };
    Geom g{};
    Config c{};
    Wire w{};
    rad_weight projection = 0, correction = 0, correction_vl_w = 0;
    rad_op op_project = 0, op_select = 0, op_select_vl = 0;
    SplitF32Proj split{};

    int declare(RadBuilder* b, Names& nm, const Geom& geom, int layer, const char* name,
                const Config& cfg, const Wire& wire, const Src& src) {
        g = geom;
        c = cfg;
        w = wire;
        if (!b || !name || !src.weight || !src.correction || g.n_embd < 1 || c.experts < 1 ||
            c.experts > 512 || c.top_k < 1 || c.top_k > c.experts || c.rows < 1 ||
            c.rows > g.max_tok || !std::isfinite(c.route_scale) || c.route_scale <= 0 ||
            c.groups < 1 || c.experts % c.groups || c.topk_groups < 1 || c.topk_groups > c.groups ||
            !w.input || !w.logits || !w.ids || !w.weights || w.logits == w.weights)
            return RAD_E_INVAL;
        const char *pn = nm.f("%s.weight", name), *cn = nm.f("%s.correction", name);
        RAD_ARCH_TRY(map_copy(b, pn, src.weight));
        RAD_ARCH_TRY(map_copy(b, cn, src.correction));
        projection = decl_w(b, pn, RAD_F32, { c.experts, g.n_embd }, RAD_ACCESS_PER_TOKEN,
                            RAD_SHARD_NONE, grp_layer(layer));
        correction = decl_w(b, cn, RAD_F32, { c.experts }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(layer));
        if (!projection || !correction) return RAD_E_INVAL;
        op_project = rw(b,
                        RAD_OP(b, "gemm_nt_f32",
                               RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("N", c.experts),
                                          RAD_INT("K", g.n_embd), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(projection)),
                        { w.input }, { w.logits });
        RAD_ARCH_TRY(split.declare(b, nm, g, name, c.rows, c.experts, g.n_embd, projection, w.input,
                                   w.logits));
        if (c.score)
            op_select =
                rw(b,
                   RAD_OP(b, "router_scored_topk",
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.experts),
                                     RAD_INT("top_k", c.top_k), RAD_INT("norm", c.norm),
                                     RAD_F64("route_scale", c.route_scale),
                                     RAD_STR("score", c.score), RAD_STR("dtype", "f32")),
                          RAD_WEIGHTS(correction)),
                   { w.logits }, { w.ids, w.weights });
        else if (c.groups > 1)
            op_select =
                rw(b,
                   RAD_OP(b, "router_sigmoid_group_topk",
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.experts),
                                     RAD_INT("top_k", c.top_k), RAD_INT("n_group", c.groups),
                                     RAD_INT("topk_group", c.topk_groups), RAD_INT("norm", c.norm),
                                     RAD_F64("route_scale", c.route_scale), RAD_STR("dtype", "f32")),
                          RAD_WEIGHTS(correction)),
                   { w.logits }, { w.ids, w.weights });
        else
            op_select =
                rw(b,
                   RAD_OP(b, "router_sigmoid_topk",
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.experts),
                                     RAD_INT("top_k", c.top_k), RAD_INT("norm", c.norm),
                                     RAD_F64("route_scale", c.route_scale), RAD_STR("dtype", "f32")),
                          RAD_WEIGHTS(correction)),
                   { w.logits }, { w.ids, w.weights });
        if (c.vl) {
            if (!src.correction_vl || !c.score) return RAD_E_INVAL;
            const char* vn = nm.f("%s.correction_vl", name);
            RAD_ARCH_TRY(map_copy(b, vn, src.correction_vl));
            correction_vl_w = decl_w(b, vn, RAD_F32, { c.experts }, RAD_ACCESS_PER_TOKEN,
                                     RAD_SHARD_NONE, grp_layer(layer));
            if (!correction_vl_w) return RAD_E_INVAL;
            op_select_vl =
                rw(b,
                   RAD_OP(b, "router_scored_topk_vl",
                          RAD_PARAMS(RAD_RANGE("M", 1, c.rows), RAD_INT("n_expert", c.experts),
                                     RAD_INT("top_k", c.top_k), RAD_INT("norm", c.norm),
                                     RAD_F64("route_scale", c.route_scale),
                                     RAD_STR("score", c.score), RAD_STR("dtype", "f32")),
                          RAD_WEIGHTS(correction, correction_vl_w)),
                   { w.logits }, { w.ids, w.weights });
            if (!op_select_vl) return RAD_E_INVAL;
        }
        return op_project && op_select ? RAD_OK : RAD_E_INVAL;
    }
    bool masked() const { return op_select_vl != 0; }
    /* The masked select: same projection, then the vl op with the row mask (RadBatch::vl_mask;
     * 1 inside an image span). */
    void step_masked(RadCtx* ctx, int64_t first, int64_t rows, int64_t output_first,
                     RadOperand mask) const {
        project(ctx, first, rows, output_first);
        RAD_ISSUE_N(ctx, op_select_vl, rows, brow_slice(w.logits, output_first, rows, c.experts),
                    RAD_W(correction), RAD_W(correction_vl_w), mask,
                    brow_slice(w.ids, output_first, rows, c.top_k),
                    brow_slice(w.weights, output_first, rows, c.top_k));
    }
    void project(RadCtx* ctx, int64_t first, int64_t rows, int64_t output_first = 0) const {
        if (split.takes(rows))
            split.step(ctx, rows, brow_slice(w.input, first, rows, g.n_embd), projection,
                       brow_slice(w.logits, output_first, rows, c.experts));
        else
            RAD_ISSUE_N(ctx, op_project, rows, brow_slice(w.input, first, rows, g.n_embd),
                        RAD_W(projection), brow_slice(w.logits, output_first, rows, c.experts));
    }
    void step(RadCtx* ctx, int64_t first, int64_t rows, int64_t output_first = 0) const {
        project(ctx, first, rows, output_first);
        RAD_ISSUE_N(ctx, op_select, rows, brow_slice(w.logits, output_first, rows, c.experts),
                    RAD_W(correction), brow_slice(w.ids, output_first, rows, c.top_k),
                    brow_slice(w.weights, output_first, rows, c.top_k));
    }
};
}
} // namespace rad::arch
#endif
