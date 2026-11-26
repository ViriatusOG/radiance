/* flint -- Recursive Flint (39.8B-A2.5B): two dense layers then 47 sigmoid-routed MoE layers,
 * grouped-query attention with per-head Q/K RMSNorm, a learned per-head softmax sink, and a
 * 513-token sliding window on 35 of the 49 layers (the other 14 are full attention and NoPE).
 *
 *   2 dense layers:  rmsnorm -> q/k/v -> qk-norm -> [rope] -> paged attention + sink -> o
 *                    rmsnorm -> gate_up -> silu_mul -> down
 *   47 MoE layers:   same attention; rmsnorm -> router -> top-6 of 256 (sigmoid)
 *                    -> grouped gate_up -> silu_mul -> down -> gather
 *                    + ungated shared expert (2 x 512)
 *
 * Weights come from a BF16 or recipe-quantized Flint checkpoint; the blocks probe the stored
 * encoding. Two trunk forms, read from layer 0's q projection:
 *
 *   - a rotated int8 trunk (data/recipes/flint-iq8r.recipe: IQ8R linears, i8 codes rotated by a
 *     128-wide Hadamard): the norms, the gated products and the attention output write
 *     had_quant_act_i8g's codes themselves, every linear reads them at i8a8h, and the routed
 *     experts are rotated int8 (or another form with an int8 row) read by the grouped GEMM's int8
 *     rows from the same codes (MoeQ::a8), as qwen35moe_w8a8 serves Qwen3.6-35B-A3B;
 *   - any other (bf16, the four-bit attention of data/recipes/flint-bf16-q4.recipe, plain int8
 *     linears): each linear reads the bf16 plane or quantises it itself, and four-bit experts read
 *     the block input's E4M3 pair, as hyv3_bf16 serves its W4 and W8 recipes.
 *
 * No MTP head.
 */
#include <arch/rad_arch.h>
#include "flint_attention.h"
#include <arch/rad_block_mlp_q.h>
#include <arch/rad_block_moe_q.h>
#include "rad_abi_layout.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <vector>

namespace flint {
using namespace rad::arch;

struct Layer {
    bool     dense = false;
    bool     sliding = false;
    FlintAttn attn{};
    MlpQ   dense_mlp{};
    MoeQ   moe{};
};

struct Model {
    Names nm{};
    Geom  g{};
    int64_t n_ff_expert = 0, n_ff_shared = 0;
    rad_kvgroup kv_full = 0, kv_win = 0;
    /* Whether layer 0's probe found a rotated-int8 trunk, and its activation scale group: see
     * rad_linear.h's trunk_rotation. */
    bool    trunk_rot = false;
    int64_t trunk_g8 = RAD_SCALE_GROUP;

    rad_buf x = 0, h_out = 0, logits = 0;
    ActQ  h{};
    /* The routed experts' rotated int8 copy of the block input, where a rotated trunk's norms do
     * not write the codes themselves (MoeQ::Wire::hr). */
    ActQ  hr{};
    rad_buf q = 0, k = 0, v = 0;
    ActQ  attn{};
    rad_buf dense_gate_up = 0;
    ActQ  dense_ffn{};
    MoeQ::Wire mw{};

    rad_weight token = 0, final_norm = 0, lm_head = 0;
    rad_op embed = 0, embed_ar = 0, shared_one = 0, final_rms = 0, gather = 0, logits_op = 0;
    std::vector<Layer> layers;
};

/* Every per-instance state is indexed by slot (rad_build_slot at declare, rad_slot in step). */
static Model g_model[MAX_SLOTS];

/* Flint's layer schedule: full attention (and NoPE) at 0, 1, 5, 9, ..., 45, 48; the other 35
 * layers are sliding (window 513) and carry RoPE. */
static bool is_full(int64_t l) {
    return l == 0 || l == 1 || l == 48 || (l >= 5 && l <= 45 && (l - 5) % 4 == 0);
}

static int64_t meta_i(const RadModelMeta* meta, const char* a, const char* b, int64_t def) {
    int64_t v = meta_text_geti(meta, a, INT64_MIN);
    if (v == INT64_MIN && b) v = meta_text_geti(meta, b, INT64_MIN);
    return v == INT64_MIN ? def : v;
}

static int check_model(const RadModelMeta* meta, const Geom& g, Model& m) {
    const int64_t sliding = meta_i(meta, "sliding_window", "flint.attention.sliding_window", 513);
    const int64_t n_expert = meta_text_geti(meta, "num_experts", meta->n_expert);
    const int64_t top_k = meta_text_geti(meta, "num_experts_per_tok", meta->n_expert_used);
    const int64_t moe_ff = meta_i(meta, "moe_intermediate_size",
                                  "flint.expert_feed_forward_length", 512);
    const int64_t shared = meta_i(meta, "shared_expert_intermediate_size",
                                  "flint.expert_shared_feed_forward_length", 1024);
    const int64_t dense_ff = meta_i(meta, "intermediate_size",
                                    "flint.feed_forward_length", 6144);

    if (g.n_layer != 49 || g.n_embd != 2048 || g.n_head_all != 32 || g.n_head_kv_all != 4 ||
        g.head_dim != 128 || g.n_vocab_all != 150144 || dense_ff != 6144 ||
        sliding != 513 || n_expert != 256 || top_k != 6 || moe_ff != 512 || shared != 1024) {
        fprintf(stderr, "radiance: flint: unsupported geometry (49 layers, 2048-wide, 32/4 "
                        "heads x 128, vocab 150144, 256 experts top-6, expert width 512, shared "
                        "1024, sliding 513); got %lld layers, %lld-wide, %lld/%lld heads, "
                        "vocab %lld, %lld experts top-%lld, expert %lld, shared %lld, "
                        "sliding %lld, dense %lld\n",
                (long long)g.n_layer, (long long)g.n_embd, (long long)g.n_head_all,
                (long long)g.n_head_kv_all, (long long)g.n_vocab_all, (long long)n_expert,
                (long long)top_k, (long long)moe_ff, (long long)shared, (long long)sliding,
                (long long)dense_ff);
        return RAD_E_UNSUPPORTED;
    }
    if (g.world != 1 && g.world != 2) {
        fprintf(stderr, "radiance: flint supports tensor-parallel world sizes 1 and 2\n");
        return RAD_E_UNSUPPORTED;
    }
    m.n_ff_expert = moe_ff;
    m.n_ff_shared = shared;
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->rank < 0 || ctx->rank >= MAX_RANKS ||
        rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);

    int status = geom_from(m.g, meta, ctx, "bf16", RAD_BF16);
    if (status < 0) return status;
    Geom& g = m.g;
    Names& nm = m.nm;

    status = check_model(meta, g, m);
    if (status < 0) return status;

    if (ctx->kv_dtype != RAD_DT_INVALID && ctx->kv_dtype != RAD_BF16 && ctx->kv_dtype != RAD_I8) {
        fprintf(stderr, "radiance: flint serves a bf16 or an int8 KV cache, not %s\n",
                rad_dtype_name((uint32_t)ctx->kv_dtype));
        return RAD_E_UNSUPPORTED;
    }
    status = yarn_from(g, meta, "", g.head_dim);
    if (status < 0) return status;

    const int64_t sliding = meta_i(meta, "sliding_window", "flint.attention.sliding_window", 513);
    const int64_t n_expert_all = meta_text_geti(meta, "num_experts", meta->n_expert);
    const int64_t top_k = meta_text_geti(meta, "num_experts_per_tok", meta->n_expert_used);
    if (n_expert_all % g.world != 0) return RAD_E_SHAPE;
    const int64_t n_expert_local = n_expert_all / g.world;

    /* Two paged groups: the 14 full layers grow with the context, the 35 sliding layers keep
     * only 513 tokens. That is what makes 262K affordable. Both take the deployment's cache
     * dtype: bf16, or with --kv-cache-dtype i8 the int8 cache (abi/rad_builder.h), K and V as
     * int8 codes with one f16 scale each per token and head. */
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_FULL;
    kd.dtype = rad_kv_cache_dtype(ctx, RAD_BF16);
    kd.n_head_kv = g.n_head_kv;
    kd.head_dim = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_full"), &kd);
    if (!m.kv_full) return RAD_E_INVAL;

    RadKVGroupDecl kw = kd;
    kw.kind = RAD_KV_WINDOW;
    kw.window = sliding;
    m.kv_win = rad_decl_kv_group(b, nm.f("kv_win"), &kw);
    if (!m.kv_win) return RAD_E_INVAL;
    g.kv_dtype = rad_dtype_name(kd.dtype);

    const int64_t T = g.max_tok, qd = g.q_dim(), kvd = g.kv_dim();

    /* The trunk's stored form, probed from layer 0's q projection: rad_linear.h's trunk_rotation
     * answers whether it is rotated int8 and how wide the activation codes' scale group is. */
    m.layers.resize((size_t)g.n_layer);
    RAD_ARCH_TRY(m.layers[0].attn.q.probe(b, nm, g, nm.f("blk.0.attn_q"),
                                          { nm.ckpt("model.layers.0.self_attn.q_proj.weight") }));
    {
        RadEncoding enc{};
        m.trunk_rot = weight_enc(b, nm.f("blk.0.attn_q.weight"), &enc) &&
                      trunk_rotation(enc, &m.trunk_g8);
    }

    MoeQ::Config mc{};
    mc.n_expert_all = n_expert_all;
    mc.n_expert = n_expert_local;
    mc.expert_base = n_expert_local * g.rank;
    mc.probed_expert = mc.expert_base;
    mc.top_k = top_k;
    mc.n_ff_exp = m.n_ff_expert;
    mc.norm_topk = 1;
    mc.sigmoid_router = 1;
    mc.route_scale = meta_text_getf(meta, "router_scaling_factor", 1.0);
    mc.shared_ungated = 1;
    /* One combine at the gather, not two separately reduced arms: TP2 then matches the TP1
     * arithmetic (the local routed+shared sum is rounded to bf16 once and the closing
     * all-reduce reduces it), and that single collective folds into the next block's norm. */
    mc.shared_fp32_combine = 0;
    mc.shared_one_external = g.world == 1;
    mc.router_fuse = 1;
    mc.rows = T < 2048 ? T : 2048;
    block_slice_128(m.n_ff_shared, g.world, g.rank, &mc.shared_lo, &mc.shared_hi);
    mc.n_ff_shared = mc.shared_hi - mc.shared_lo;

    /* ---- the routed experts' stored form, read from the first routed layer's first owned
     * expert; every layer is declared with it (MoeQ::declare checks each layer's own). */
    const int64_t l_moe = 2;
    MoeQ::Src probe_src{};
    probe_src.experts = nm.ckpt("model.layers.%lld.mlp.experts", (long long)l_moe);
    RAD_ARCH_TRY(MoeQ::probe(b, nm, (int)l_moe, probe_src, &mc));
    if (m.trunk_rot && (mc.expert_bf16 || !mc.expert_rot)) {
        fprintf(stderr, "radiance: flint: the trunk is rotated int8 and the routed experts are "
                        "%s; a rotated trunk's experts read its int8 codes and are stored rotated "
                        "(data/recipes/flint-iq8r.recipe)\n",
                mc.expert_bf16 ? "plain bf16" : "unrotated");
        return RAD_E_UNSUPPORTED;
    }
    if (!m.trunk_rot && (mc.expert_i8 || mc.expert_iqr)) {
        fprintf(stderr, "radiance: flint: the routed experts are rotated int8 and the trunk is "
                        "not; they read the trunk's rotated int8 codes, so convert the trunk with "
                        "them (data/recipes/flint-iq8r.recipe)\n");
        return RAD_E_UNSUPPORTED;
    }

    /* ---- activation buffers. A rotated trunk's producers write had_quant_act_i8g's codes and
     * its linears declare no quantiser: the norms (rmsnorm_had_quant_i8g and, at two ranks, its
     * all-reduce form) where a loaded library serves them, the attention output (q_attn), and the
     * gated products (gated_had_quant_i8g). Where the norms are not served, each linear
     * quantises `h` itself and the experts read `hr`, written once a layer. Any other trunk keeps
     * both code pairs on `h`: E4M3 for four-bit experts, int8 for a plain int8 linear's own
     * quantiser. */
    const bool rot = m.trunk_rot;
    const int64_t g8 = m.trunk_g8;
    const bool h_fed = rot && rad_op_declared(b, "rmsnorm_had_quant_i8g") &&
                       (g.world != 2 || rad_op_declared(b, "ar_rmsnorm_had_quant_i8g"));
    m.x = decl_b(b, nm.f("x"), RAD_BF16, {T, g.n_embd});
    m.h.x = decl_b(b, nm.f("h"), RAD_BF16, {T, g.n_embd});
    RAD_ARCH_TRY(m.h.declare_qs(b, nm, g, "h", g.n_embd, 0, true, !h_fed,
                                rot ? g8 : (int64_t)RAD_SCALE_GROUP));
    m.h.q8_fed = m.h.q8_rot = h_fed;
    if (rot && !h_fed) {
        RAD_ARCH_TRY(m.hr.declare_qs(b, nm, g, "moe_hr", g.n_embd, 0, true, false,
                                     RAD_SCALE_GROUP));
        m.hr.q8_fed = m.hr.q8_rot = true;
    }
    m.q = decl_b(b, nm.f("attn_q"), RAD_BF16, {T, qd});
    m.k = decl_b(b, nm.f("attn_k"), RAD_BF16, {T, kvd});
    m.v = decl_b(b, nm.f("attn_v"), RAD_BF16, {T, kvd});
    m.attn.x = decl_b(b, nm.f("attn_out"), RAD_BF16, {T, qd});
    RAD_ARCH_TRY(m.attn.declare_qs(b, nm, g, "attn_out", qd, 0, true, false,
                                   rot ? g8 : (int64_t)RAD_SCALE_GROUP));
    m.attn.q8_fed = m.attn.q8_rot = rot;
    m.dense_gate_up = decl_b(b, nm.f("dense_gate_up"), RAD_BF16, {T, 2 * g.n_ff});
    m.dense_ffn.x = decl_b(b, nm.f("dense_ffn"), RAD_BF16, {T, g.n_ff});
    if (rot)
        RAD_ARCH_TRY(m.dense_ffn.declare_pair(b, nm, g, "dense_ffn", g.n_ff, 0, true, g8,
                                              "gated_had_quant_i8g"));
    else
        RAD_ARCH_TRY(m.dense_ffn.declare_qs(b, nm, g, "dense_ffn", g.n_ff, 0, true));
    m.h_out = decl_b(b, nm.f("h_out"), RAD_BF16, {g.max_logit_rows, g.n_embd});
    m.logits = decl_b(b, nm.f("logits"), RAD_F32, {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.logits));

    const int64_t R = T, S = mc.rows * top_k;
    m.mw.x = m.x;
    m.mw.h = m.h;
    m.mw.hr = m.hr;
    m.mw.logits = decl_b(b, nm.f("moe_logits"), RAD_F32, {R, n_expert_all});
    m.mw.ids = decl_b(b, nm.f("moe_ids"), RAD_I32, {R, top_k});
    m.mw.ew = decl_b(b, nm.f("moe_weights"), RAD_F32, {R, top_k});
    m.mw.eoff = decl_b(b, nm.f("moe_offsets"), RAD_I32, {n_expert_local + 1});
    m.mw.ecnt = decl_b(b, nm.f("moe_counts"), RAD_I32, {n_expert_local}, RAD_BUF_PERSIST);
    m.mw.sorted = decl_b(b, nm.f("moe_sorted"), RAD_I32, {S});
    m.mw.egu = decl_b(b, nm.f("moe_gate_up"), RAD_BF16, {S, 2 * mc.n_ff_exp});
    m.mw.eff.x = decl_b(b, nm.f("moe_ffn"), RAD_BF16, {S, mc.n_ff_exp});
    if (rot) {
        /* The routed arm of a rotated trunk is int8 throughout: the gated product's rotated codes
         * at one scale a 128 of K, which the grouped GEMM's int8 rows read (MoeQ::a8). */
        RAD_ARCH_TRY(m.mw.eff.declare_qs(b, nm, g, "moe_ffn", mc.n_ff_exp, S, true, false,
                                         RAD_SCALE_GROUP));
        m.mw.eff.q8_fed = m.mw.eff.q8_rot = true;
    } else {
        RAD_ARCH_TRY(m.mw.eff.declare_qs(b, nm, g, "moe_ffn", mc.n_ff_exp, S, true));
    }
    m.mw.edn = decl_b(b, nm.f("moe_down"), RAD_BF16, {S, g.n_embd});
    m.mw.sgu = decl_b(b, nm.f("shared_gate_up"), RAD_BF16, {T, 2 * mc.n_ff_shared});
    m.mw.sff.x = decl_b(b, nm.f("shared_ffn"), RAD_BF16, {T, mc.n_ff_shared});
    if (rot) {
        /* The shared expert's down projection is a trunk linear whose input codes take the group
         * its own weight is stored at, read from the first routed layer's through that layer's
         * own LinearQ (which then maps it once). */
        LinearQ& sd = m.layers[(size_t)l_moe].moe.sh_down;
        RAD_ARCH_TRY(sd.probe(b, nm, g, nm.f("blk.%lld.ffn_down_shexp", (long long)l_moe),
                              { nm.ckpt("model.layers.%lld.mlp.shared_expert.down_proj.weight",
                                        (long long)l_moe) }));
        const int64_t sg = sd.probed_i8 ? sd.probed_g8 : g8;
        RAD_ARCH_TRY(m.mw.sff.declare_pair(b, nm, g, "shared_ffn", mc.n_ff_shared, 0, true, sg,
                                           "gated_had_quant_i8g"));
    } else {
        RAD_ARCH_TRY(m.mw.sff.declare_qs(b, nm, g, "shared_ffn", mc.n_ff_shared, 0, true));
    }
    m.mw.sout = decl_b(b, nm.f("shared_out"), RAD_BF16, {T, g.n_embd});
    m.mw.sgate = decl_b(b, nm.f("shared_one"), RAD_BF16, {T, 1});
    if (mc.shared_one_external) {
        m.shared_one = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", 1),
                                        RAD_STR("dtype", g.dtype), RAD_F64("value", 1.0)),
                             RAD_NOWEIGHTS), {}, {m.mw.sgate});
        if (!m.shared_one) return RAD_E_INVAL;
    }

    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("model.norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.embed_tokens.weight")));
    m.token = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.final_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.lm_head = decl_w(b, nm.f("output.weight"),
                       weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                       {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.embed = rw(b, RAD_OP(b, "embed_lookup",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                               RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                               RAD_INT("vocab_offset", g.vocab_off)),
                    RAD_WEIGHTS(m.token)),
                {}, {m.x});
    if (g.world > 1)
        m.embed_ar = decl_all_reduce_exact(b, g, m.x);

    FlintAttn::Wire aw{};
    aw.x = m.x; aw.h = m.h; aw.q = m.q; aw.k = m.k; aw.v = m.v; aw.attn = m.attn;
    MlpQ::Wire dw{};
    dw.x = m.x; dw.h = m.h; dw.gate_up = m.dense_gate_up; dw.ffn = m.dense_ffn;

    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.dense = (l < 2);
        lay.sliding = !is_full(l);
        const char* pre = nm.ckpt("model.layers.%lld", (long long)l);

        FlintAttn::Src as{};
        as.norm   = nm.ckpt("%s.input_layernorm.weight", pre);
        as.q      = nm.ckpt("%s.self_attn.q_proj.weight", pre);
        as.k      = nm.ckpt("%s.self_attn.k_proj.weight", pre);
        as.v      = nm.ckpt("%s.self_attn.v_proj.weight", pre);
        as.q_norm = nm.ckpt("%s.self_attn.q_norm.weight", pre);
        as.k_norm = nm.ckpt("%s.self_attn.k_norm.weight", pre);
        as.o      = nm.ckpt("%s.self_attn.o_proj.weight", pre);
        as.sink   = nm.ckpt("%s.self_attn.softmax_offset", pre);
        as.use_rope = lay.sliding;
        as.window   = lay.sliding ? sliding : 0;

        const rad_kvgroup grp = lay.sliding ? m.kv_win : m.kv_full;
        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, grp));
        /* Every block after the first folds the previous block's delta and, at two ranks,
         * absorbs its all-reduce into its own norm; the last block keeps its add for the plain
         * final norm and issues its own collective. */
        const LayerFold f = layer_fold(g, l, g.n_layer, nullptr, 0, false);
        status = lay.attn.declare(b, nm, g, (int)l, grp, aw, as, f.mix_fold, false, f.mix_ar_in,
                                  f.mix_ar_out);
        if (status < 0) {
            fprintf(stderr, "radiance: flint: attention layer %lld declaration failed (%d)\n",
                    (long long)l, status);
            return status;
        }

        if (lay.dense) {
            MlpQ::Src ds{};
            const char* mlp = nm.ckpt("%s.mlp", pre);
            ds.norm = nm.ckpt("%s.post_attention_layernorm.weight", pre);
            ds.gate = nm.ckpt("%s.gate_proj.weight", mlp);
            ds.up   = nm.ckpt("%s.up_proj.weight", mlp);
            ds.down = nm.ckpt("%s.down_proj.weight", mlp);
            status = lay.dense_mlp.declare(b, nm, g, (int)l, dw, ds, true, f.mlp_add,
                                           f.mlp_ar_in, f.mlp_ar_out);
            if (status < 0) {
                fprintf(stderr, "radiance: flint: dense layer %lld declaration failed (%d)\n",
                        (long long)l, status);
                return status;
            }
        } else {
            MoeQ::Src ms{};
            ms.cfg = mc;
            const char* mlp = nm.ckpt("%s.mlp", pre);
            ms.norm = nm.ckpt("%s.post_attention_layernorm.weight", pre);
            ms.router = nm.ckpt("%s.gate.weight", mlp);
            /* Flint has no router correction bias. The sigmoid router's bias is
             * selection-only, so the converter merges a zero [n_expert] tensor under this
             * name (data/flint-zero-bias.safetensors, `rad-convert --add`). */
            ms.router_bias = nm.ckpt("%s.expert_bias", mlp);
            ms.experts = nm.ckpt("%s.experts", mlp);
            ms.shared  = nm.ckpt("%s.shared_expert", mlp);
            status = lay.moe.declare(b, nm, g, (int)l, m.mw, ms, true, f.mlp_add, f.mlp_ar_in,
                                     f.mlp_ar_out);
            if (status < 0) {
                fprintf(stderr, "radiance: flint: MoE layer %lld declaration failed (%d)\n",
                        (long long)l, status);
                return status;
            }
        }
    }

    m.final_rms = rw(b, RAD_OP(b, "rmsnorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(m.final_norm)),
                    {m.x}, {m.h.x});
    m.gather = rw(b, RAD_OP(b, "gather_rows",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd),
                                RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                  {m.h.x}, {m.h_out});
    m.logits_op = rw(b, RAD_OP(b, "logits_gemm",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(m.lm_head)),
                    {m.h_out}, {m.logits});

    rad_note(b, "Flint: %lld layers (2 dense, 47 top-%lld/%lld sigmoid MoE), %lld/%lld "
                "attention/KV heads x %lld; expert width %lld, shared %lld, window %lld on %s; "
                "per-head sink on every layer; %s trunk, %s KV cache",
             (long long)g.n_layer, (long long)top_k, (long long)n_expert_all,
             (long long)g.n_head_all, (long long)g.n_head_kv_all, (long long)g.head_dim,
             (long long)m.n_ff_expert, (long long)mc.n_ff_shared, (long long)sliding,
             g.world == 2 ? "both cards" : "one card",
             rot ? (g8 == RAD_I8_GROUP_WIDE ? "IQ8R (rotated int8)" : "rotated int8") : "unrotated",
             g.kv_dtype);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_slot(c)];
    RAD_ISSUE(c, m.embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.token),
              brows(m.x, batch->n_tok));
    if (m.embed_ar)
        RAD_ISSUE_N(c, m.embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.x, batch->n_tok), RAD_NONE);

    if (m.shared_one)
        RAD_ISSUE_N(c, m.shared_one, batch->n_tok, brows(m.mw.sgate, batch->n_tok));
    for (Layer& layer : m.layers) {
        layer.attn.step(c, batch);
        if (layer.dense) layer.dense_mlp.step(c, batch);
        else layer.moe.step(c, batch);
    }

    RAD_ISSUE(c, m.final_rms, brows(m.x, batch->n_tok), RAD_W(m.final_norm),
              brows(m.h.x, batch->n_tok));
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.gather, batch->n_out,
                    brows(m.h.x, batch->n_tok), praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.h_out, batch->n_out));
        RAD_ISSUE_N(c, m.logits_op, batch->n_out,
                    brows(m.h_out, batch->n_out), RAD_W(m.lm_head),
                    brows(m.logits, batch->n_out));
    }
}

static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->draft_depth = 0;
    return RAD_OK;
}

}  // namespace flint

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(flint)
static const char* const kFlintTriggers[] = { "<tool_call>" };
static const RadChatFormat kFlintChatFormat = {
    .struct_size = sizeof(RadChatFormat),
    .name = "flint",
    .reasoning_open = "<think>",
    .reasoning_close = "</think>",
    .reasoning_breaks = kFlintTriggers,
    .n_reasoning_breaks = 1,
    .reasoning_in_prompt = RAD_CHAT_THINK_FROM_PROMPT,
    .section_open = nullptr,
    .section_close = nullptr,
    .call_separator = nullptr,
    .call_open = nullptr,
    .call_close = nullptr,
    .name_prefix = "<tool_call>",
    .name_suffix = "",
    .call_end = "</tool_call>",
    .arg_prefix = "<arg_key>",
    .arg_name_suffix = "</arg_key>",
    .arg_value_prefix = "<arg_value>",
    .arg_value_suffix = "</arg_value>",
    .string_value = RAD_CHAT_VALUE_RAW,
    .other_value = RAD_CHAT_VALUE_JSON,
    .value_newlines = 1,
    .triggers = kFlintTriggers,
    .n_triggers = 1,
    .args_form = RAD_CHAT_ARGS_TAGGED,
};
extern "C" const RadChatFormat* rad_arch_chat_format(const RadModelMeta*) {
    return &kFlintChatFormat;
}
extern "C" const char* rad_arch_id(void) { return "flint"; }
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
extern "C" const char* rad_arch_quant(void) { return ""; }
extern "C" int rad_arch_declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* c) {
    return flint::declare(b, m, c);
}
extern "C" void rad_arch_step(struct RadCtx* c, const struct RadBatch* batch) {
    flint::step(c, batch);
}
extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
extern "C" const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = {
        RAD_PLUGIN_ARCH, "flint", "0.1.0",
        "Recursive Flint: 2 dense + 47 sigmoid MoE layers, sink attention, 513 window", ""
    };
    return &info;
}
#endif
