/* DeepSeek-V4.1's vision tower: the ViT over one image and the aligner that maps its 3x3 cells
 * into the language model's width, BF16 at TP1 (the tower is ~0.9 GiB; it is replicated, not
 * sharded). The reference is `inference/vision.py` (DeepSeek ViT: full bidirectional attention
 * over one image with 2D RoPE) and `model.py` 1218-1242 for the span.
 *
 * Geometry is fixed by config.json: 32 blocks, width 1024 = 16 heads x 64, SwiGLU 2816, patch
 * 14, theta 10000, RMSNorm eps 1e-6 with an F32 gain, 3x3 aligner into 5120. Nothing here
 * hard-codes those numbers: P (patches a pass), depth and widths all come from vision_config.
 *
 * Unlike the text path, no kernel normalises the weightless RMS with a separate gain: the tower
 * weight is F32, so `rmsnorm` carries the gain itself (it reads the gain in f32 and rounds the
 * row once, which is the reference's float32 arithmetic narrowed to the activation dtype).
 * The 2-D rotary is `ds41_vit_rope2d` -- both axes on ONE ladder, not Pixtral's interleaved
 * pair -- and the 3x3 pack is `ds41_pack3x3`, zero-padded over the raster grid, not a
 * channel-major merger. */
#pragma once
#include <arch/rad_arch.h>

namespace deepseek41 {
using namespace rad::arch;

struct Ds41Vision {
    struct Norm {
        rad_weight w = 0;
        rad_op op = 0;
    };
    struct Layer {
        Norm n1, n2;
        rad_weight qkv_w = 0, qkv_b = 0, out_w = 0, out_b = 0, gu = 0, down = 0;
        rad_op project = 0, rope = 0, attn = 0, output = 0, add1 = 0;
        rad_op gate = 0, act = 0, gate_mul = 0, ffn = 0, add2 = 0;
    };
    bool on = false;
    int64_t P = 0, width = 0, heads = 0, hd = 0, ff = 0, depth = 0, patch_dim = 0;
    int64_t merge = 0, embd = 0, max_grid = 0;
    float eps = 1e-6f, theta = 10000.0f;
    rad_buf x = 0, h = 0, qkv = 0, att = 0, o = 0, gu = 0, ffn = 0, packed = 0, proj = 0,
            out = 0;
    rad_weight patch = 0, patch_b = 0, merger1 = 0, merger1_b = 0, merger2 = 0, merger2_b = 0;
    /* The span's learned rows (image_start / image_end / image_newline) copied into `span_rows`
     * [4, n_embd]: 0, 3, 2; slot 1 is unused (the encoder's own rows). */
    rad_weight start_w = 0, end_w = 0, nl_w = 0;
    rad_buf span_rows = 0;
    rad_op op_start = 0;
    rad_op stem = 0, pack = 0, project1 = 0, gelu = 0, project2 = 0;
    Norm post;
    std::vector<Layer> layers;

    int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx, Names& nm,
                int64_t model_embd, int first_group) {
        on = false;
        if (ctx->max_enc_patches <= 0) return RAD_OK;
        auto vi = [&](const char* k) {
            return rad_meta_geti(meta, (std::string("vision_config.") + k).c_str(), 0);
        };
        width = vi("hidden_size");
        heads = vi("num_attention_heads");
        ff = vi("intermediate_size");
        depth = vi("num_hidden_layers");
        const int64_t ps = vi("patch_size");
        /* config.json: `downsample_ratio` 3, no per-tower eps or image_size; the reference fixes
         * RMSNorm eps at 1e-6 (vision.py's norm), temporal patch 1 and RGB. */
        merge = vi("downsample_ratio");
        embd = model_embd;
        const double theta_m = rad_meta_getf(meta, "vision_config.rope_theta", 0);
        const double eps_m = rad_meta_getf(meta, "vision_config.rms_norm_eps", 1e-6);
        if (width <= 0 || heads <= 0 || width % heads || (width / heads) % 4 || ff <= 0 ||
            depth <= 0 || ps <= 0 || merge < 2 || embd <= 0 || !(theta_m > 0) ||
            !std::isfinite(theta_m) || !(eps_m > 0) || !std::isfinite(eps_m))
            return RAD_E_INVAL;
        hd = width / heads;
        patch_dim = 3 * 1 * ps * ps;          /* temporal_patch_size 1: an image is one plane */
        /* gemm_nt_bias's device kernel stages 8-wide; 588 is not a multiple of 8, and the host
         * fallback reads the device pixel pointer and faults. Four zero columns make K legal. */
        patch_dim = (patch_dim + 7) & ~7;
        theta = (float)theta_m;
        eps = (float)eps_m;
        const int64_t mm = merge * merge, max_seg = ctx->max_enc_patches;
        P = max_seg;
        if (P % mm) return RAD_E_SHAPE;
        if (ctx->shape_probe) P = std::max(mm, std::min(P, ctx->max_tok) / mm * mm);
        if (P < mm) return RAD_E_INVAL;
        const int64_t R = P / mm;
        max_grid = ctx->max_enc_patches;

        auto buf = [&](const char* name, int64_t rows, int64_t cols, uint32_t flags = 0) {
            return decl_b(b, nm.f("v.%s", name), RAD_BF16, {rows, cols}, flags);
        };
        x   = buf("x", P, width);
        h   = buf("h", P, width);
        qkv = buf("qkv", P, 3 * width);
        att = buf("attn", P, width);
        o   = buf("o", P, width);
        gu  = buf("gate_up", P, 2 * ff);
        ffn = buf("ff", P, ff);
        packed = buf("packed", R, mm * width);
        proj = buf("proj", R, embd);
        out = buf("out", R, embd, RAD_BUF_PERSIST);
        if (!x || !h || !qkv || !att || !o || !gu || !ffn || !packed || !proj || !out)
            return RAD_E_INVAL;

        auto weight = [&](const char* dest, const char* src, uint32_t dt,
                          std::initializer_list<int64_t> shape, RadWeightGroup group) {
            if (map_copy(b, dest, src) < 0) return rad_weight(0);
            return decl_w(b, dest, dt, shape, RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, group);
        };
        auto gemm = [&](int64_t rows, int64_t N, int64_t K, const char* act, rad_weight w,
                        rad_weight bias) -> rad_op {
            if (bias)
                return RAD_OP(b, "gemm_nt_bias",
                              RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                         RAD_STR("dtype", "bf16"), RAD_STR("act", act)),
                              RAD_WEIGHTS(w, bias));
            return RAD_OP(b, "gemm_nt",
                          RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                     RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w));
        };
        auto norm = [&](Norm& n, const char* dest, const char* src, RadWeightGroup group,
                        rad_buf from, rad_buf to) {
            n.w = weight(dest, src, RAD_F32, {width}, group);
            n.op = rw(b,
                      RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                        RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(n.w)),
                      {from}, {to});
        };

        patch = weight(nm.f("v.patch.weight"), nm.ckpt("vision.patch_embed.proj.weight"), RAD_BF16,
                       {width, patch_dim}, grp_model());
        patch_b = weight(nm.f("v.patch.bias"), nm.ckpt("vision.patch_embed.proj.bias"), RAD_BF16,
                         {width}, grp_model());
        stem = rw(b, gemm(P, width, patch_dim, "none", patch, patch_b), {}, {x});

        layers.resize((size_t)depth);
        for (int64_t i = 0; i < depth; ++i) {
            auto& l = layers[(size_t)i];
            const long long li = i;
            const auto group = grp_layer(first_group + (int)i);
            norm(l.n1, nm.f("v.blk.%lld.n1.weight", li),
                 nm.ckpt("vision.blocks.%lld.norm1.weight", li), group, x, h);
            norm(l.n2, nm.f("v.blk.%lld.n2.weight", li),
                 nm.ckpt("vision.blocks.%lld.norm2.weight", li), group, x, h);
            l.qkv_w = weight(nm.f("v.blk.%lld.qkv.weight", li),
                             nm.ckpt("vision.blocks.%lld.attn.wqkv.weight", li), RAD_BF16,
                             {3 * width, width}, group);
            l.qkv_b = weight(nm.f("v.blk.%lld.qkv.bias", li),
                             nm.ckpt("vision.blocks.%lld.attn.wqkv.bias", li), RAD_BF16,
                             {3 * width}, group);
            l.out_w = weight(nm.f("v.blk.%lld.out.weight", li),
                             nm.ckpt("vision.blocks.%lld.attn.wo.weight", li), RAD_BF16,
                             {width, width}, group);
            l.out_b = weight(nm.f("v.blk.%lld.out.bias", li),
                             nm.ckpt("vision.blocks.%lld.attn.wo.bias", li), RAD_BF16, {width},
                             group);
            l.gu = weight(nm.f("v.blk.%lld.gate_up.weight", li),
                          nm.ckpt("vision.blocks.%lld.mlp.w1.weight", li), RAD_BF16,
                          {2 * ff, width}, group);
            l.down = weight(nm.f("v.blk.%lld.down.weight", li),
                            nm.ckpt("vision.blocks.%lld.mlp.w2.weight", li), RAD_BF16,
                            {width, ff}, group);
            if (!l.n1.w || !l.n2.w || !l.qkv_w || !l.qkv_b || !l.out_w || !l.out_b || !l.gu ||
                !l.down)
                return RAD_E_INVAL;
            l.project = rw(b, gemm(P, 3 * width, width, "none", l.qkv_w, l.qkv_b), {h}, {qkv});
            l.rope = rw(b,
                        RAD_OP(b, "ds41_vit_rope2d",
                               RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", hd),
                                          RAD_INT("n_head", heads), RAD_F64("theta", theta),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                        {qkv}, {qkv});
            l.attn = rw(b,
                        RAD_OP(b, "embed_attn",
                               RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", hd),
                                          RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                          RAD_INT("max_seqlen", max_seg),
                                          RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                        {qkv}, {att});
            l.output = rw(b, gemm(P, width, width, "none", l.out_w, l.out_b), {att}, {o});
            l.add1 = rw(b,
                        RAD_OP(b, "add",
                               RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                        {x, o}, {x});
            l.gate = rw(b, gemm(P, 2 * ff, width, "none", l.gu, 0), {h}, {gu});
            l.act = rw(b,
                       RAD_OP(b, "silu",
                              RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", ff),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                       {gu}, {gu});
            l.gate_mul = rw(b,
                            RAD_OP(b, "mul",
                                   RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", ff),
                                              RAD_STR("dtype", "bf16")),
                                   RAD_NOWEIGHTS),
                            {gu}, {ffn});
            l.ffn = rw(b, gemm(P, width, ff, "none", l.down, 0), {ffn}, {o});
            l.add2 = rw(b,
                        RAD_OP(b, "add",
                               RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                        {x, o}, {x});
        }
        norm(post, nm.f("v.post.weight"), nm.ckpt("vision.norm.weight"), grp_model(), x, h);
        merger1 = weight(nm.f("v.merger1.weight"), nm.ckpt("aligner.w1.weight"), RAD_BF16,
                         {embd, mm * width}, grp_model());
        merger1_b = weight(nm.f("v.merger1.bias"), nm.ckpt("aligner.w1.bias"), RAD_BF16, {embd},
                           grp_model());
        merger2 = weight(nm.f("v.merger2.weight"), nm.ckpt("aligner.w2.weight"), RAD_BF16,
                         {embd, embd}, grp_model());
        merger2_b = weight(nm.f("v.merger2.bias"), nm.ckpt("aligner.w2.bias"), RAD_BF16, {embd},
                           grp_model());
        if (!patch || !patch_b || !post.w || !merger1 || !merger1_b || !merger2 ||
            !merger2_b)
            return RAD_E_INVAL;
        start_w = weight(nm.f("v.image_start"), nm.ckpt("image_start"), RAD_BF16, {embd},
                         grp_model());
        end_w = weight(nm.f("v.image_end"), nm.ckpt("image_end"), RAD_BF16, {embd}, grp_model());
        nl_w = weight(nm.f("v.image_newline"), nm.ckpt("image_newline"), RAD_BF16, {embd},
                      grp_model());
        span_rows = decl_b(b, nm.f("v.span_rows"), RAD_BF16, {4, embd}, RAD_BUF_PERSIST);
        if (!start_w || !end_w || !nl_w || !span_rows) return RAD_E_INVAL;
        op_start = rw(b,
                      RAD_OP(b, "ds41_span_rows",
                             RAD_PARAMS(RAD_INT("n", embd), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(start_w, nl_w, end_w)),
                      {}, {span_rows});
        if (!op_start) return RAD_E_INVAL;
        pack = rw(b,
                  RAD_OP(b, "ds41_pack3x3",
                         RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width),
                                    RAD_INT("merge", merge), RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                  {h}, {packed});
        project1 = rw(b, gemm(R, embd, mm * width, "none", merger1, merger1_b), {packed}, {proj});
        gelu = rw(b,
                  RAD_OP(b, "gelu",
                         RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", embd),
                                    RAD_STR("dtype", "bf16")),
                         RAD_NOWEIGHTS),
                  {proj}, {proj});
        project2 = rw(b, gemm(R, embd, embd, "none", merger2, merger2_b), {proj}, {out});
        if (!ctx->shape_probe) {
            RadEncoderDecl e{};
            e.name = "vision";
            e.modalities = RAD_MM_IMAGE;
            e.patch_dim = patch_dim;
            e.merge = (int)mm;
            e.max_patches = P;
            e.n_embd = embd;
            e.out = out;
            e.span_rows = span_rows;
            RAD_ARCH_TRY(rad_declare_encoder(b, &e));
            rad_note(b, "DeepSeek-V4.1 vision tower: %lld blocks of %lld (%lld heads x %lld), "
                        "ff %lld, patch %lld, merge %lld; %lld patches a pass -> %lld rows of %lld",
                     (long long)depth, (long long)width, (long long)heads, (long long)hd,
                     (long long)ff, (long long)ps, (long long)merge, (long long)P, (long long)R,
                     (long long)embd);
        }
        on = true;
        return RAD_OK;
    }

    void step(RadCtx* c, const RadBatch* batch) const {
        const int64_t n = batch->enc_n_patch, s = batch->enc_n_seg, mm = merge * merge;
        if (!on || batch->enc != RAD_MM_IMAGE || n <= 0 || n > P || s <= 0 || !batch->enc_pixels ||
            !batch->enc_coord || !batch->enc_cu || !batch->enc_seg_h || !batch->enc_seg_w) {
            rad_step_fail(c, "deepseek41: invalid vision encoder pass");
            return;
        }
        int64_t total = 0, rows = 0;
        for (int64_t i = 0; i < s; ++i) {
            const int64_t hh = batch->enc_seg_h[i], ww = batch->enc_seg_w[i];
            if (hh <= 0 || ww <= 0 || hh * ww > n - total) {
                rad_step_fail(c, "deepseek41: vision expects complete raster grids");
                return;
            }
            total += hh * ww;
            rows += ((hh + merge - 1) / merge) * ((ww + merge - 1) / merge);
        }
        if (total != n || rows > P / mm) {
            rad_step_fail(c, "deepseek41: vision grid sizes disagree with patches");
            return;
        }
        const auto coord = praw2(batch->enc_coord, RAD_I32, 2, n),
                   crd4 = praw2(batch->enc_coord, RAD_I32, 4, n),
                   cu = praw(batch->enc_cu, RAD_I32, s + 1);
        auto norm = [&](const Norm& nrm, rad_buf from, rad_buf to) {
            RAD_ISSUE_N(c, nrm.op, n, brows(from, n), RAD_W(nrm.w), brows(to, n));
        };
        RAD_ISSUE_N(c, stem, n, praw2(batch->enc_pixels, RAD_BF16, n, patch_dim), RAD_W(patch),
                    RAD_W(patch_b), RAD_NONE, brows(x, n));
        RAD_ISSUE_N(c, op_start, embd, RAD_W(start_w), RAD_W(nl_w), RAD_W(end_w),
                    brows(span_rows, 4));
        for (const auto& l : layers) {
            norm(l.n1, x, h);
            RAD_ISSUE_N(c, l.project, n, brows(h, n), RAD_W(l.qkv_w), RAD_W(l.qkv_b), RAD_NONE,
                        brows(qkv, n));
            RAD_ISSUE_N(c, l.rope, n, brows(qkv, n), coord);
            RAD_ISSUE_N(c, l.attn, n, bcol(qkv, 0, width, n), bcol(qkv, width, width, n),
                        bcol(qkv, 2 * width, width, n), cu, brows(att, n));
            RAD_ISSUE_N(c, l.output, n, brows(att, n), RAD_W(l.out_w), RAD_W(l.out_b), RAD_NONE,
                        brows(o, n));
            RAD_ISSUE_N(c, l.add1, n, brows(x, n), brows(o, n), brows(x, n));
            norm(l.n2, x, h);
            RAD_ISSUE_N(c, l.gate, n, brows(h, n), RAD_W(l.gu), brows(gu, n));
            RAD_ISSUE_N(c, l.act, n, bcol(gu, 0, ff, n), bcol(gu, 0, ff, n));
            RAD_ISSUE_N(c, l.gate_mul, n, bcol(gu, 0, ff, n), bcol(gu, ff, ff, n), brows(ffn, n));
            RAD_ISSUE_N(c, l.ffn, n, brows(ffn, n), RAD_W(l.down), brows(o, n));
            RAD_ISSUE_N(c, l.add2, n, brows(x, n), brows(o, n), brows(x, n));
        }
        norm(post, x, h);
        RAD_ISSUE_N(c, pack, n, brows(h, n), crd4, brows(packed, (int64_t)rows));
        RAD_ISSUE_N(c, project1, (int64_t)rows, brows(packed, (int64_t)rows), RAD_W(merger1),
                    RAD_W(merger1_b), RAD_NONE, brows(proj, (int64_t)rows));
        RAD_ISSUE_N(c, gelu, (int64_t)rows, brows(proj, (int64_t)rows), brows(proj, (int64_t)rows));
        RAD_ISSUE_N(c, project2, (int64_t)rows, brows(proj, (int64_t)rows), RAD_W(merger2),
                    RAD_W(merger2_b), RAD_NONE, brows(out, (int64_t)rows));
    }
};
}  // namespace deepseek41
