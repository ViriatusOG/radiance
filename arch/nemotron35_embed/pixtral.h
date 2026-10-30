/* Nemotron 3.5's Pixtral tower and Mistral3 projector, BF16 at TP1.
 * The processor supplies complete raster grids. Image/text prompt integration is a separate
 * acceptance gate: a working encoder pass alone does not make image embeddings serviceable. */
#pragma once
#include <arch/rad_arch.h>

namespace nemotron35_embed {
using namespace rad::arch;

struct Pixtral {
    struct Norm {
        rad_weight w = 0;
        rad_op rms = 0, gain = 0;
    };
    struct Layer {
        Norm n1, n2;
        rad_weight qkv = 0, out = 0, gu = 0, down = 0;
        rad_op project = 0, rope = 0, attn = 0, output = 0, add1 = 0;
        rad_op gate = 0, act = 0, gate_mul = 0, ffn = 0, add2 = 0;
    };
    bool on = false;
    int64_t P = 0, width = 0, heads = 0, hd = 0, ff = 0, depth = 0, patch_dim = 0;
    int64_t merge = 0, embd = 0, max_grid = 0;
    rad_buf x = 0, h = 0, qkv = 0, att = 0, o = 0, gu = 0, ffn = 0, packed = 0, merged = 0,
            proj = 0, out = 0;
    rad_weight patch = 0, merger = 0, linear1 = 0, linear2 = 0;
    rad_op stem = 0, pack = 0, merge_op = 0, project1 = 0, gelu = 0, project2 = 0;
    Norm pre, post;
    std::vector<Layer> layers;

    int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx, Names& nm,
                Names& source, int64_t output_width, int first_group) {
        on = false;
        if (ctx->max_enc_patches <= 0) return RAD_OK;
        auto vi = [&](const char* k) {
            return rad_meta_geti(meta, (std::string("vision_config.") + k).c_str(), 0);
        };
        const char* bias = rad_meta_gets(meta, "multimodal_projector_bias", "");
        if (std::strcmp(rad_meta_gets(meta, "vision_config.model_type", ""), "pixtral") ||
            std::strcmp(rad_meta_gets(meta, "vision_config.hidden_act", ""), "silu") ||
            std::strcmp(rad_meta_gets(meta, "projector_hidden_act", ""), "gelu") ||
            (std::strcmp(bias, "0") && std::strcmp(bias, "false")) ||
            rad_meta_geti(meta, "vision_feature_layer", 0) != -1)
            return RAD_E_UNSUPPORTED;
        const char* rope = rad_meta_gets(meta, "vision_config.rope_parameters.rope_type", "");
        /* This checkpoint saves `default`; PixtralVisionConfig interprets it as axial. */
        if (std::strcmp(rope, "default") && std::strcmp(rope, "axial")) return RAD_E_UNSUPPORTED;
        width = vi("hidden_size");
        heads = vi("num_attention_heads");
        ff = vi("intermediate_size");
        depth = vi("num_hidden_layers");
        const int64_t ps = vi("patch_size"), channels = vi("num_channels"),
                      image_size = vi("image_size");
        merge = rad_meta_geti(meta, "spatial_merge_size", 0);
        embd = output_width;
        const double theta = rad_meta_getf(meta, "vision_config.rope_parameters.rope_theta", 0);
        const double eps = meta->rms_eps;
        if (width <= 0 || heads <= 0 || width % heads || (width / heads) % 4 || ff <= 0 ||
            depth <= 0 || ps <= 0 || channels != 3 || merge != 2 || embd <= 0 || image_size < ps ||
            !(theta > 0) || !std::isfinite(theta) || !(eps > 0) || !std::isfinite(eps))
            return RAD_E_INVAL;
        hd = width / heads;
        patch_dim = channels * ps * ps;
        max_grid = image_size / ps;
        if (vi("head_dim") && vi("head_dim") != hd) return RAD_E_INVAL;
        const int64_t mm = merge * merge, max_seg = ctx->max_enc_patches;
        P = max_seg;
        if (P % mm) return RAD_E_SHAPE;
        if (ctx->shape_probe) P = std::max(mm, std::min(P, ctx->max_tok) / mm * mm);
        if (P < mm) return RAD_E_INVAL;
        const int64_t R = P / mm;
        auto buf = [&](const char* name, int64_t rows, int64_t cols, uint32_t flags = 0) {
            return decl_b(b, nm.f("v.%s", name), RAD_BF16, {rows, cols}, flags);
        };
        x = buf("x", P, width);
        h = buf("h", P, width);
        qkv = buf("qkv", P, 3 * width);
        att = buf("attn", P, width);
        o = buf("o", P, width);
        gu = buf("gate_up", P, 2 * ff);
        ffn = buf("ff", P, ff);
        packed = buf("packed", R, mm * width);
        merged = buf("merged", R, width);
        proj = buf("proj", R, embd);
        out = buf("out", R, embd, RAD_BUF_PERSIST);
        auto weight = [&](const char* dest, const char* src, std::initializer_list<int64_t> shape,
                          RadWeightGroup group) {
            if (map_copy(b, dest, src) < 0) return rad_weight(0);
            return decl_w(b, dest, RAD_BF16, shape, RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, group);
        };
        auto gemm = [&](int64_t rows, int64_t N, int64_t K, rad_weight w) {
            return RAD_OP(b, "gemm_nt",
                          RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                     RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w));
        };
        auto norm = [&](Norm& n, const char* dest, const char* src, RadWeightGroup group, double e,
                        rad_buf from, rad_buf to) {
            n.w = weight(dest, src, {width}, group);
            /* The reference narrows the weightless norm before multiplying by the BF16 gain. */
            n.rms = rw(b,
                       RAD_OP(b, "rmsnorm_p",
                              RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                         RAD_F64("eps", e), RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                       {from}, {to});
            n.gain = rw(b,
                        RAD_OP(b, "mul_weight",
                               RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                          RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(n.w)),
                        {to}, {to});
        };
        auto add = [&] {
            return RAD_OP(
                b, "add",
                RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width), RAD_STR("dtype", "bf16")),
                RAD_NOWEIGHTS);
        };
        patch = weight(nm.f("v.patch.weight"), source.f("vision_tower.patch_conv.weight"),
                       {width, patch_dim}, grp_model());
        stem = rw(b, gemm(P, width, patch_dim, patch), {}, {x});
        norm(pre, nm.f("v.pre.weight"), source.f("vision_tower.ln_pre.weight"), grp_model(), 1e-5,
             x, x);
        layers.resize((size_t)depth);
        for (int64_t i = 0; i < depth; ++i) {
            auto& l = layers[(size_t)i];
            const long long li = i;
            const auto group = grp_layer(first_group + (int)i);
            norm(l.n1, nm.f("v.blk.%lld.n1.weight", li),
                 source.f("vision_tower.transformer.layers.%lld.attention_norm.weight", li), group,
                 1e-5, x, h);
            norm(l.n2, nm.f("v.blk.%lld.n2.weight", li),
                 source.f("vision_tower.transformer.layers.%lld.ffn_norm.weight", li), group, 1e-5,
                 x, h);
            RAD_ARCH_TRY(map_concat(
                b, nm.f("v.blk.%lld.qkv.weight", li),
                {source.f("vision_tower.transformer.layers.%lld.attention.q_proj.weight", li),
                 source.f("vision_tower.transformer.layers.%lld.attention.k_proj.weight", li),
                 source.f("vision_tower.transformer.layers.%lld.attention.v_proj.weight", li)},
                0));
            l.qkv = decl_w(b, nm.f("v.blk.%lld.qkv.weight", li), RAD_BF16, {3 * width, width},
                           RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, group);
            l.out =
                weight(nm.f("v.blk.%lld.out.weight", li),
                       source.f("vision_tower.transformer.layers.%lld.attention.o_proj.weight", li),
                       {width, width}, group);
            RAD_ARCH_TRY(map_concat(
                b, nm.f("v.blk.%lld.gate_up.weight", li),
                {source.f("vision_tower.transformer.layers.%lld.feed_forward.gate_proj.weight", li),
                 source.f("vision_tower.transformer.layers.%lld.feed_forward.up_proj.weight", li)},
                0));
            l.gu = decl_w(b, nm.f("v.blk.%lld.gate_up.weight", li), RAD_BF16, {2 * ff, width},
                          RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, group);
            l.down = weight(
                nm.f("v.blk.%lld.down.weight", li),
                source.f("vision_tower.transformer.layers.%lld.feed_forward.down_proj.weight", li),
                {width, ff}, group);
            l.project = rw(b, gemm(P, 3 * width, width, l.qkv), {h}, {qkv});
            l.rope = rw(b,
                        RAD_OP(b, "pixtral_rope",
                               RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", hd),
                                          RAD_INT("n_head", heads), RAD_F64("theta", theta),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                        {qkv}, {qkv});
            l.attn = rw(
                b,
                RAD_OP(b, "embed_attn",
                       RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", hd), RAD_INT("gqa", 1),
                                  RAD_INT("causal", 0), RAD_INT("max_seqlen", max_seg),
                                  RAD_F64("scale", 1 / std::sqrt((double)hd)),
                                  RAD_STR("dtype", "bf16")),
                       RAD_NOWEIGHTS),
                {qkv}, {att});
            l.output = rw(b, gemm(P, width, width, l.out), {att}, {o});
            l.add1 = rw(b, add(), {x, o}, {x});
            l.gate = rw(b, gemm(P, 2 * ff, width, l.gu), {h}, {gu});
            l.act = rw(
                b,
                RAD_OP(b, "silu",
                       RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", ff), RAD_STR("dtype", "bf16")),
                       RAD_NOWEIGHTS),
                {gu}, {gu});
            l.gate_mul = rw(
                b,
                RAD_OP(b, "mul",
                       RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", ff), RAD_STR("dtype", "bf16")),
                       RAD_NOWEIGHTS),
                {gu}, {ffn});
            l.ffn = rw(b, gemm(P, width, ff, l.down), {ffn}, {o});
            l.add2 = rw(b, add(), {x, o}, {x});
        }
        norm(post, nm.f("v.post.weight"), source.f("multi_modal_projector.norm.weight"),
             grp_model(), eps, x, h);
        merger = weight(nm.f("v.merger.weight"),
                        source.f("multi_modal_projector.patch_merger.merging_layer.weight"),
                        {width, mm * width}, grp_model());
        linear1 =
            weight(nm.f("v.linear1.weight"), source.f("multi_modal_projector.linear_1.weight"),
                   {embd, width}, grp_model());
        linear2 =
            weight(nm.f("v.linear2.weight"), source.f("multi_modal_projector.linear_2.weight"),
                   {embd, embd}, grp_model());
        pack = rw(b,
                  RAD_OP(b, "pixtral_merge",
                         RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                    RAD_INT("merge", merge), RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                  {h}, {packed});
        merge_op = rw(b, gemm(R, width, mm * width, merger), {packed}, {merged});
        project1 = rw(b, gemm(R, embd, width, linear1), {merged}, {proj});
        gelu = rw(
            b,
            RAD_OP(b, "gelu",
                   RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", embd), RAD_STR("dtype", "bf16")),
                   RAD_NOWEIGHTS),
            {proj}, {proj});
        project2 = rw(b, gemm(R, embd, embd, linear2), {proj}, {out});
        const RadEncoding plain = rad_enc_plain(RAD_BF16);
        auto bf16_only = [&](const char* name) -> int {
            RadEncoding enc{};
            const int rc = rad_weight_encoding(b, name, &enc, nullptr, nullptr);
            if (rc == RAD_E_NOTFOUND) return RAD_OK;
            if (rc < 0) return rc;
            return rad_enc_equal(&enc, &plain) ? RAD_OK : RAD_E_UNSUPPORTED;
        };
        for (const char* name : {"patch", "pre", "post", "merger", "linear1", "linear2"})
            RAD_ARCH_TRY(bf16_only(nm.f("v.%s.weight", name)));
        for (int64_t i = 0; i < depth; ++i)
            for (const char* name : {"n1", "n2", "qkv", "out", "gate_up", "down"})
                RAD_ARCH_TRY(bf16_only(nm.f("v.blk.%lld.%s.weight", (long long)i, name)));
        if (!ctx->shape_probe) {
            RadEncoderDecl e{};
            e.name = "vision";
            e.modalities = RAD_MM_IMAGE;
            e.patch_dim = patch_dim;
            e.merge = mm;
            e.max_patches = P;
            e.n_embd = embd;
            e.out = out;
            RAD_ARCH_TRY(rad_declare_encoder(b, &e));
        }
        on = true;
        return RAD_OK;
    }

    void step(RadCtx* c, const RadBatch* batch) const {
        const int64_t n = batch->enc_n_patch, s = batch->enc_n_seg, mm = merge * merge;
        if (!on || batch->enc != RAD_MM_IMAGE || n <= 0 || n > P || n % mm || s <= 0 ||
            s > n / mm || batch->enc_max_seg <= 0 || batch->enc_max_seg > P || !batch->enc_pixels ||
            !batch->enc_coord || !batch->enc_cu || !batch->enc_seg_h || !batch->enc_seg_w) {
            rad_step_fail(c, "nemotron35_embed: invalid Pixtral encoder pass");
            return;
        }
        int64_t total = 0;
        for (int64_t i = 0; i < s; ++i) {
            const int64_t hh = batch->enc_seg_h[i], ww = batch->enc_seg_w[i];
            if (hh <= 0 || ww <= 0 || hh % merge || ww % merge || hh > max_grid || ww > max_grid ||
                hh * ww > batch->enc_max_seg || hh * ww > n - total) {
                rad_step_fail(c, "nemotron35_embed: Pixtral expects complete even raster grids");
                return;
            }
            total += hh * ww;
        }
        if (total != n) {
            rad_step_fail(c, "nemotron35_embed: Pixtral grid sizes disagree with patches");
            return;
        }
        const int64_t r = n / mm;
        const auto coord = praw2(batch->enc_coord, RAD_I32, 2, n),
                   cu = praw(batch->enc_cu, RAD_I32, s + 1);
        auto norm = [&](const Norm& nrm, rad_buf from, rad_buf to) {
            RAD_ISSUE_N(c, nrm.rms, n, brows(from, n), brows(to, n));
            RAD_ISSUE_N(c, nrm.gain, n, brows(to, n), RAD_W(nrm.w), brows(to, n));
        };
        RAD_ISSUE_N(c, stem, n, praw2(batch->enc_pixels, RAD_BF16, n, patch_dim), RAD_W(patch),
                    brows(x, n));
        norm(pre, x, x);
        for (const auto& l : layers) {
            norm(l.n1, x, h);
            RAD_ISSUE_N(c, l.project, n, brows(h, n), RAD_W(l.qkv), brows(qkv, n));
            RAD_ISSUE_N(c, l.rope, n, brows(qkv, n), coord);
            RAD_ISSUE_N(c, l.attn, n, bcol(qkv, 0, width, n), bcol(qkv, width, width, n),
                        bcol(qkv, 2 * width, width, n), cu, brows(att, n));
            RAD_ISSUE_N(c, l.output, n, brows(att, n), RAD_W(l.out), brows(o, n));
            RAD_ISSUE_N(c, l.add1, n, brows(x, n), brows(o, n), brows(x, n));
            norm(l.n2, x, h);
            RAD_ISSUE_N(c, l.gate, n, brows(h, n), RAD_W(l.gu), brows(gu, n));
            RAD_ISSUE_N(c, l.act, n, bcol(gu, 0, ff, n), bcol(gu, 0, ff, n));
            RAD_ISSUE_N(c, l.gate_mul, n, bcol(gu, 0, ff, n), bcol(gu, ff, ff, n), brows(ffn, n));
            RAD_ISSUE_N(c, l.ffn, n, brows(ffn, n), RAD_W(l.down), brows(o, n));
            RAD_ISSUE_N(c, l.add2, n, brows(x, n), brows(o, n), brows(x, n));
        }
        norm(post, x, h);
        RAD_ISSUE_N(c, pack, n, brows(h, n), praw2(batch->enc_coord, RAD_I32, 4, n),
                    brows(packed, r));
        RAD_ISSUE_N(c, merge_op, r, brows(packed, r), RAD_W(merger), brows(merged, r));
        RAD_ISSUE_N(c, project1, r, brows(merged, r), RAD_W(linear1), brows(proj, r));
        RAD_ISSUE_N(c, gelu, r, brows(proj, r), brows(proj, r));
        RAD_ISSUE_N(c, project2, r, brows(proj, r), RAD_W(linear2), brows(out, r));
    }
};
} // namespace nemotron35_embed
