/* Shared HYV3 graph. Thin architecture wrappers select the source descriptor and source format. */
#ifndef HYV3_NS
#define HYV3_NS hyv3_fp8
#endif
#ifndef HYV3_QUANT
#define HYV3_QUANT "fp8"
#endif
#ifndef HYV3_FP8_SOURCE
#define HYV3_FP8_SOURCE 1
#endif
#ifndef HYV3_PLUGIN_NAME
#define HYV3_PLUGIN_NAME "hyv3_fp8"
#endif
#ifndef HYV3_GGUF_SOURCE
#define HYV3_GGUF_SOURCE 0
#endif
#include <arch/rad_arch.h>
#include "hyv3_attention.h"
#include "rad_abi_layout.h"
#include <arch/rad_block_mlp_fp8.h>
#include <arch/rad_block_moe_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace HYV3_NS {
using namespace rad::arch;

struct Layer {
    bool dense = false;
    AttnQKFP8 attn{};
    MlpFP8 dense_mlp{};
    MoeFP8 moe{};
};

struct Model {
    Names nm{};
    Geom g{};
    int64_t n_ff_expert = 0, n_ff_shared = 0;
    rad_kvgroup kv_full = 0;

    rad_buf x = 0, h_out = 0, logits = 0;
    ActFP8 h{};
    rad_buf q = 0, k = 0, v = 0;
    ActFP8 attn{};
    rad_buf dense_gate_up = 0;
    ActFP8 dense_ffn{};
    MoeFP8::Wire mw{};

    rad_weight token = 0, final_norm = 0, lm_head = 0;
    rad_op embed = 0, embed_ar = 0, shared_one = 0, final_rms = 0, gather = 0, logits_op = 0;
    std::vector<Layer> layers;
};

static Model models[MAX_RANKS];

static int check_format(const RadModelMeta* meta) {
    const char* q = meta && meta->quant ? meta->quant : "";
#if HYV3_GGUF_SOURCE
    if (std::strcmp(q, "gguf_ftype_15") != 0) {
        fprintf(stderr, "radiance: HYV3 GGUF adapter expects quant descriptor 'gguf_ftype_15'; "
                        "got '%s'\n", *q ? q : "(none)");
        return RAD_E_UNSUPPORTED;
    }
#elif HYV3_FP8_SOURCE
    if (std::strcmp(q, "fp8") != 0) {
        fprintf(stderr, "radiance: HYV3 FP8 expects quant descriptor 'fp8'; got '%s'\n",
                *q ? q : "(none)");
        return RAD_E_UNSUPPORTED;
    }
    const char* fmt = rad_meta_gets(meta, "quantization_config.fmt", nullptr);
    if (fmt && std::strcmp(fmt, "e4m3") != 0) return RAD_E_UNSUPPORTED;
    const char* method = rad_meta_gets(meta, "quantization_config.quant_method", nullptr);
    if (method && std::strcmp(method, "fp8") != 0) return RAD_E_UNSUPPORTED;
#else
    if (*q) {
        fprintf(stderr, "radiance: HYV3 BF16 expects an unquantized checkpoint; got '%s'\n", q);
        return RAD_E_UNSUPPORTED;
    }
#endif
    return RAD_OK;
}

static int check_model_config(const RadModelMeta* meta, const Geom& g, Model& model) {
    if (meta_text_geti(meta, "qk_norm", 0) != 1 ||
        meta_text_geti(meta, "moe_router_use_sigmoid", 0) != 1 ||
        meta_text_geti(meta, "moe_router_enable_expert_bias", 0) != 1 ||
        meta_text_geti(meta, "route_norm", 0) != 1 ||
        meta_text_geti(meta, "enable_moe_fp32_combine", 0) != 1 ||
        meta_text_geti(meta, "attention_bias", 0) != 0 ||
        meta_text_geti(meta, "mlp_bias", 0) != 0) {
        fprintf(stderr, "radiance: hyv3 requires QK norm, sigmoid routing with correction bias, "
                        "route normalization, FP32 MoE combine, and bias-free projections\n");
        return RAD_E_UNSUPPORTED;
    }

    const int64_t first_dense = meta_text_geti(meta, "first_k_dense_replace", -1);
    const int64_t n_expert = meta_text_geti(meta, "num_experts", 0);
    const int64_t top_k = meta_text_geti(meta, "num_experts_per_tok", 0);
    const int64_t n_shared = meta_text_geti(meta, "num_shared_experts", 0);
    const int64_t moe_intermediate = meta_text_geti(meta, "moe_intermediate_size", 0);
    const int64_t n_ff_shared = meta_text_geti(meta, "expert_hidden_dim", 0);
    model.n_ff_expert = meta_text_geti(meta, "expert_hidden_dim", 0);
    const int64_t route_norm = meta_text_geti(meta, "route_norm", 0);
    const double route_scale = meta_text_getf(meta, "router_scaling_factor", 1.0);
    const char* schedule = meta_text_gets(meta, "mlp_layer_types", nullptr);
    std::vector<int> kinds;

    if (first_dense != 1 || n_expert != meta->n_expert || top_k != meta->n_expert_used ||
        n_expert != 128 || top_k != 8 || n_shared != 1 || model.n_ff_expert != 768 ||
        n_ff_shared != 768 || moe_intermediate != 1536 || route_norm != 1 ||
        !std::isfinite(route_scale) || route_scale <= 0 ||
        std::fabs(route_scale - 2.826) > 1e-6 || model.n_ff_expert % RAD_FP8_BLOCK != 0) {
        fprintf(stderr, "radiance: hyv3: unsupported MoE geometry/config; expected 128 experts, "
                        "top-8, 768-wide routed and shared experts, normalized "
                        "sigmoid routing scaled by 2.826, and one dense first layer\n");
        return RAD_E_UNSUPPORTED;
    }
    if (schedule && *schedule) {
        RAD_ARCH_TRY(parse_layer_kinds("hyv3", schedule, {"sparse"}, "dense", &kinds));
        if ((int64_t)kinds.size() != g.n_layer) {
            fprintf(stderr, "radiance: hyv3: mlp_layer_types names %zu layers; the model has %lld\n",
                    kinds.size(), (long long)g.n_layer);
            return RAD_E_UNSUPPORTED;
        }
        for (int64_t l = 0; l < g.n_layer; ++l) {
            const int want = l == 0 ? 0 : 1;
            if (kinds[(size_t)l] != want) {
                fprintf(stderr, "radiance: hyv3: layer %lld has an unsupported MLP kind; expected "
                                "dense at layer 0 and sparse afterwards\n", (long long)l);
                return RAD_E_UNSUPPORTED;
            }
        }
    }

    const int64_t shared_width = n_ff_shared * n_shared;
    if (shared_width <= 0 || shared_width % RAD_FP8_BLOCK != 0) return RAD_E_SHAPE;
    model.n_ff_shared = shared_width;
    return RAD_OK;
}

static void block_slice_128(int64_t width, int world, int rank, int64_t* lo, int64_t* hi) {
    const int64_t blocks = width / RAD_FP8_BLOCK;
    const int64_t base = blocks / world, rem = blocks % world;
    const int64_t first = rank * base + std::min<int64_t>(rank, rem);
    const int64_t count = base + (rank < rem ? 1 : 0);
    *lo = first * RAD_FP8_BLOCK;
    *hi = (first + count) * RAD_FP8_BLOCK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    Model& m = models[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    int status = check_format(meta);
    if (status < 0) return status;
    status = geom_from(m.g, meta, ctx, "bf16", RAD_BF16);
    if (status < 0) return status;
    Geom& g = m.g;
    Names& nm = m.nm;

    if (g.n_layer < 2 || g.n_embd != 2048 || g.n_head_all != 32 || g.n_head_kv_all != 4 ||
        g.head_dim != 128 || g.n_vocab_all != 120832 ||
        (g.world != 1 && g.world != 2)) {
        fprintf(stderr, "radiance: hyv3 supports Hy-Memory's 2048-wide, 32/4-head, "
                        "head_dim=128 geometry at tensor-parallel world sizes 1 or 2\n");
        return RAD_E_UNSUPPORTED;
    }
    status = check_model_config(meta, g, m);
    if (status < 0) return status;
#if HYV3_FP8_SOURCE
    if ((ctx->kv_dtype != RAD_DT_INVALID && ctx->kv_dtype != RAD_BF16 &&
         ctx->kv_dtype != RAD_F8E4M3) || ctx->kv_quant != RAD_KVQ_NONE) {
        fprintf(stderr, "radiance: HYV3 FP8 supports BF16 or plain E4M3 KV cache, not this layout\n");
        return RAD_E_UNSUPPORTED;
    }
#else
    if ((ctx->kv_dtype != RAD_DT_INVALID && ctx->kv_dtype != RAD_BF16) ||
        ctx->kv_quant != RAD_KVQ_NONE) {
        fprintf(stderr, "radiance: HYV3 BF16 supports a BF16 KV cache only\n");
        return RAD_E_UNSUPPORTED;
    }
#endif
    status = yarn_from(g, meta, "", g.head_dim);
    if (status < 0) return status;

    const int64_t n_expert_all = meta_text_geti(meta, "num_experts", meta->n_expert);
    const int64_t top_k = meta_text_geti(meta, "num_experts_per_tok", meta->n_expert_used);
    if (n_expert_all % g.world != 0) return RAD_E_SHAPE;
    const int64_t n_expert_local = n_expert_all / g.world;

    /* ---- Full-attention KV cache. All layers share one group. */
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_FULL;
    kd.dtype = rad_kv_cache_dtype(ctx, RAD_BF16);
    kd.n_head_kv = g.n_head_kv;
    kd.head_dim = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &kd);
    if (!m.kv_full) return RAD_E_INVAL;
    g.kv_dtype = rad_dtype_name(kd.dtype);

    const int64_t T = g.max_tok, qd = g.q_dim(), kvd = g.kv_dim();
    m.x = decl_b(b, nm.f("x"), RAD_BF16, {T, g.n_embd});
    m.h.x = decl_b(b, nm.f("h"), RAD_BF16, {T, g.n_embd});
    RAD_ARCH_TRY(m.h.declare_qs(b, nm, g, "h", g.n_embd, 0, !HYV3_FP8_SOURCE));
    m.q = decl_b(b, nm.f("attn_q"), RAD_BF16, {T, qd});
    m.k = decl_b(b, nm.f("attn_k"), RAD_BF16, {T, kvd});
    m.v = decl_b(b, nm.f("attn_v"), RAD_BF16, {T, kvd});
    m.attn.x = decl_b(b, nm.f("attn_out"), RAD_BF16, {T, qd});
    RAD_ARCH_TRY(m.attn.declare_qs(b, nm, g, "attn_out", qd, 0, !HYV3_FP8_SOURCE));
    m.dense_gate_up = decl_b(b, nm.f("dense_gate_up"), RAD_BF16, {T, 2 * g.n_ff});
    m.dense_ffn.x = decl_b(b, nm.f("dense_ffn"), RAD_BF16, {T, g.n_ff});
    RAD_ARCH_TRY(m.dense_ffn.declare_qs(b, nm, g, "dense_ffn", g.n_ff, 0, !HYV3_FP8_SOURCE));
    m.h_out = decl_b(b, nm.f("h_out"), RAD_BF16, {g.max_logit_rows, g.n_embd});
    m.logits = decl_b(b, nm.f("logits"), RAD_F32, {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.logits));

    /* The router is replicated, while the routed experts are expert-parallel. The shared expert
     * is split only on whole FP8 scale blocks; the final block remainder is assigned to early
     * ranks. At TP2, routed and shared BF16 outputs are all-reduced separately before FP32 combine. */
    MoeFP8::Config mc{};
    mc.n_expert_all = n_expert_all;
    mc.n_expert = n_expert_local;
    mc.expert_base = n_expert_local * g.rank;
    mc.probed_expert = mc.expert_base;
    mc.top_k = top_k;
    mc.n_ff_exp = m.n_ff_expert;
    mc.norm_topk = 1;
    mc.sigmoid_router = 1;
    mc.route_scale = meta_text_getf(meta, "router_scaling_factor", 2.826);
    mc.shared_ungated = 1;
    mc.shared_fp32_combine = 1;
    mc.shared_one_external = g.world == 1;
    mc.router_fuse = 1;
    mc.rows = T < 2048 ? T : 2048;
    block_slice_128(m.n_ff_shared, g.world, g.rank, &mc.shared_lo, &mc.shared_hi);
    mc.n_ff_shared = mc.shared_hi - mc.shared_lo;

#if HYV3_FP8_SOURCE
    /* The source's per-tensor scales are imported by the recipe to format 9 (block FP8 with F32
     * 128x128 scales). The model probes the source before recipe planning, so state that encoding
     * explicitly here. */
    mc.expert_f32_scale = 1;
#endif

    const int64_t R = T, S = mc.rows * top_k;
    m.mw.x = m.x;
    m.mw.h = m.h;
    m.mw.logits = decl_b(b, nm.f("moe_logits"), RAD_F32, {R, n_expert_all});
    m.mw.ids = decl_b(b, nm.f("moe_ids"), RAD_I32, {R, top_k});
    m.mw.ew = decl_b(b, nm.f("moe_weights"), RAD_F32, {R, top_k});
    m.mw.eoff = decl_b(b, nm.f("moe_offsets"), RAD_I32, {n_expert_local + 1});
    m.mw.ecnt = decl_b(b, nm.f("moe_counts"), RAD_I32, {n_expert_local}, RAD_BUF_PERSIST);
    m.mw.sorted = decl_b(b, nm.f("moe_sorted"), RAD_I32, {S});
    m.mw.egu = decl_b(b, nm.f("moe_gate_up"), RAD_BF16, {S, 2 * mc.n_ff_exp});
    m.mw.eff.x = decl_b(b, nm.f("moe_ffn"), RAD_BF16, {S, mc.n_ff_exp});
    RAD_ARCH_TRY(m.mw.eff.declare_qs(b, nm, g, "moe_ffn", mc.n_ff_exp, S, !HYV3_FP8_SOURCE));
    m.mw.edn = decl_b(b, nm.f("moe_down"), RAD_BF16, {S, g.n_embd});
    m.mw.moe = decl_b(b, nm.f("moe_routed"), RAD_BF16, {T, g.n_embd});
    m.mw.sgu = decl_b(b, nm.f("shared_gate_up"), RAD_BF16, {T, 2 * mc.n_ff_shared});
    m.mw.sff.x = decl_b(b, nm.f("shared_ffn"), RAD_BF16, {T, mc.n_ff_shared});
    RAD_ARCH_TRY(m.mw.sff.declare_qs(b, nm, g, "shared_ffn", mc.n_ff_shared, 0,
                                     !HYV3_FP8_SOURCE));
    m.mw.sout = decl_b(b, nm.f("shared_out"), RAD_BF16, {T, g.n_embd});
    m.mw.sgate = decl_b(b, nm.f("shared_one"), RAD_BF16, {T, 1});
    if (mc.shared_one_external) {
        m.shared_one = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", 1),
                                        RAD_STR("dtype", g.dtype), RAD_F64("value", 1.0)),
                             RAD_NOWEIGHTS), {}, {m.mw.sgate});
        if (!m.shared_one) return RAD_E_INVAL;
    }

    /* ---- Token embedding and head. */
#if HYV3_GGUF_SOURCE
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("token_embd.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("output_norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("output.weight"),
                              nm.ckpt("token_embd.weight")));
#else
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("model.norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.embed_tokens.weight")));
#endif
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
        m.embed_ar = rw(b, RAD_OP(b, "all_reduce",
                           RAD_PARAMS(RAD_INT("world_size", g.world),
                                      RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                      RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                           RAD_NOWEIGHTS),
                       {m.x}, {m.x});

    AttnQKFP8::Wire aw{};
    aw.x = m.x; aw.h = m.h; aw.q = m.q; aw.k = m.k; aw.v = m.v; aw.attn = m.attn;
    MlpFP8::Wire dw{};
    dw.x = m.x; dw.h = m.h; dw.gate_up = m.dense_gate_up; dw.ffn = m.dense_ffn;

    m.layers.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.dense = (l == 0);
#if HYV3_GGUF_SOURCE
        const char* pre = nm.ckpt("blk.%lld", (long long)l);
#else
        const char* pre = nm.ckpt("model.layers.%lld", (long long)l);
#endif
        AttnQKFP8::Src as{};
#if HYV3_GGUF_SOURCE
        as.norm = nm.ckpt("%s.attn_norm.weight", pre);
        as.q = nm.ckpt("%s.attn_q.weight", pre);
        as.k = nm.ckpt("%s.attn_k.weight", pre);
        as.v = nm.ckpt("%s.attn_v.weight", pre);
        as.q_norm = nm.ckpt("%s.attn_q_norm.weight", pre);
        as.k_norm = nm.ckpt("%s.attn_k_norm.weight", pre);
        as.o = nm.ckpt("%s.attn_output.weight", pre);
#else
        as.norm = nm.ckpt("%s.input_layernorm.weight", pre);
        as.q = nm.ckpt("%s.self_attn.q_proj.weight", pre);
        as.k = nm.ckpt("%s.self_attn.k_proj.weight", pre);
        as.v = nm.ckpt("%s.self_attn.v_proj.weight", pre);
        as.q_norm = nm.ckpt("%s.self_attn.q_norm.weight", pre);
        as.k_norm = nm.ckpt("%s.self_attn.k_norm.weight", pre);
        as.o = nm.ckpt("%s.self_attn.o_proj.weight", pre);
#endif
#if HYV3_FP8_SOURCE
        if (kd.dtype == RAD_F8E4M3) {
            const std::string sk = "hyv3.kv_cache." + std::to_string(l) + ".k_scale";
            const std::string sv = "hyv3.kv_cache." + std::to_string(l) + ".v_scale";
            as.k_scale = meta_text_getf(meta, sk.c_str(), NAN);
            as.v_scale = meta_text_getf(meta, sv.c_str(), NAN);
            if (!std::isfinite(as.k_scale) || as.k_scale <= 0.0 ||
                !std::isfinite(as.v_scale) || as.v_scale <= 0.0) {
                fprintf(stderr, "radiance: hyv3_fp8: FP8 KV cache needs positive finite scales "
                                "for layer %lld (metadata keys %s and %s)\n",
                        (long long)l, sk.c_str(), sv.c_str());
                return RAD_E_UNSUPPORTED;
            }
        }
#endif
        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_full));
        // Each norm folds its predecessor's residual add. The first attention reads the
        // embedding directly; the final feed-forward keeps its add for the plain output norm.
        // Keep collective placement and both routed/shared BF16 rounding points.
        status = lay.attn.declare(b, nm, g, (int)l, m.kv_full, aw, as, l > 0, false);
        if (status < 0) {
            fprintf(stderr, "radiance: hyv3: attention layer %lld declaration failed (%d)\n",
                    (long long)l, status);
            return status;
        }

        if (lay.dense) {
            MlpFP8::Src ds{};
#if HYV3_GGUF_SOURCE
            ds.norm = nm.ckpt("%s.ffn_norm.weight", pre);
            ds.gate = nm.ckpt("%s.ffn_gate.weight", pre);
            ds.up = nm.ckpt("%s.ffn_up.weight", pre);
            ds.down = nm.ckpt("%s.ffn_down.weight", pre);
#else
            const char* mlp = nm.ckpt("%s.mlp", pre);
            ds.norm = nm.ckpt("%s.post_attention_layernorm.weight", pre);
            ds.gate = nm.ckpt("%s.gate_proj.weight", mlp);
            ds.up = nm.ckpt("%s.up_proj.weight", mlp);
            ds.down = nm.ckpt("%s.down_proj.weight", mlp);
#endif
            status = lay.dense_mlp.declare(b, nm, g, (int)l, dw, ds, true,
                                           l + 1 == g.n_layer);
            if (status < 0) {
                fprintf(stderr, "radiance: hyv3: dense MLP declaration failed (%d)\n", status);
                return status;
            }
        } else {
            MoeFP8::Src ms{};
            ms.cfg = mc;
#if HYV3_GGUF_SOURCE
            ms.norm = nm.ckpt("%s.ffn_norm.weight", pre);
            ms.router = nm.ckpt("%s.ffn_gate_inp.weight", pre);
            ms.router_bias = nm.ckpt("%s.exp_probs_b", pre);
            ms.experts_gate = nm.ckpt("%s.ffn_gate_exps.weight", pre);
            ms.experts_up = nm.ckpt("%s.ffn_up_exps.weight", pre);
            ms.experts_down = nm.ckpt("%s.ffn_down_exps.weight", pre);
            ms.shared_mlp_gate = nm.ckpt("%s.ffn_gate_shexp.weight", pre);
            ms.shared_mlp_up = nm.ckpt("%s.ffn_up_shexp.weight", pre);
            ms.shared_mlp_down = nm.ckpt("%s.ffn_down_shexp.weight", pre);
#else
            const char* mlp = nm.ckpt("%s.mlp", pre);
            ms.norm = nm.ckpt("%s.post_attention_layernorm.weight", pre);
            ms.router = nm.ckpt("%s.router.gate.weight", mlp);
            ms.router_bias = nm.ckpt("%s.expert_bias", mlp);
            ms.experts = nm.ckpt("%s.experts", mlp);
            ms.shared = nm.ckpt("%s.shared_mlp", mlp);
#endif
#if !HYV3_FP8_SOURCE
            status = MoeFP8::probe(b, nm, (int)l, ms, &ms.cfg);
            if (status < 0) {
                fprintf(stderr, "radiance: HYV3: expert format probe failed at layer %lld (%d)\n",
                        (long long)l, status);
                return status;
            }
#endif
            status = lay.moe.declare(b, nm, g, (int)l, m.mw, ms, true, l + 1 == g.n_layer);
            if (status < 0) {
                fprintf(stderr, "radiance: hyv3: MoE layer %lld declaration failed (%d)\n",
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

    rad_note(b, "HYV3: %lld layers (dense first, then top-%lld/%lld sigmoid MoE), %lld/%lld "
                "attention/KV heads x %lld; expert width %lld, shared width %lld, router scale %.3f",
             (long long)g.n_layer, (long long)top_k, (long long)n_expert_all,
             (long long)g.n_head_all, (long long)g.n_head_kv_all, (long long)g.head_dim,
             (long long)m.n_ff_expert, (long long)m.n_ff_shared, mc.route_scale);
#if HYV3_FP8_SOURCE
    rad_note(b, "E4M3 per-tensor scales are imported to 128x128 F32 planes; Q/K RMSNorm precedes "
                "NeoX RoPE and shared/routed outputs combine in FP32");
#elif HYV3_GGUF_SOURCE
    rad_note(b, "GGUF-source HYV3; Q4_K/Q6_K tensors are decoded and recipe-converted to native "
                "Radiance weights, adding quantization loss to an already-lossy source");
#else
    rad_note(b, "BF16-source HYV3; per-weight recipes may quantize linears or routed experts; "
                "Q/K RMSNorm precedes NeoX RoPE and shared/routed outputs combine in FP32");
#endif
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = models[rad_rank(c)];
    RAD_ISSUE(c, m.embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.token),
              brows(m.x, batch->n_tok));
    if (m.embed_ar)
        RAD_ISSUE_N(c, m.embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.x, batch->n_tok), RAD_NONE);

    // All TP1 ungated gathers read this same ones plane; no per-layer fill or combine launch.
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

}  // namespace HYV3_NS

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(HYV3_NS)
static const char* const kHyv3ToolTriggers[] = { "<tool_calls>" };
static const RadChatFormat kHyv3ChatFormat = {
    .struct_size = sizeof(RadChatFormat),
    .name = "hyv3",
    .reasoning_open = "<think>",
    .reasoning_close = "</think>",
    .reasoning_breaks = kHyv3ToolTriggers,
    .n_reasoning_breaks = 1,
    .reasoning_in_prompt = RAD_CHAT_THINK_FROM_PROMPT,
    .section_open = "<tool_calls>",
    .section_close = "</tool_calls>",
    .name_prefix = "<tool_call>",
    .name_suffix = "<tool_sep>",
    .call_end = "</tool_call>",
    .arg_prefix = "<arg_key>",
    .arg_name_suffix = "</arg_key>\n",
    .arg_value_prefix = "<arg_value>",
    .arg_value_suffix = "</arg_value>",
    .string_value = RAD_CHAT_VALUE_RAW,
    .other_value = RAD_CHAT_VALUE_JSON,
    .triggers = kHyv3ToolTriggers,
    .n_triggers = 1,
    .args_form = RAD_CHAT_ARGS_TAGGED,
};
extern "C" const RadChatFormat* rad_arch_chat_format(const RadModelMeta*) {
    return &kHyv3ChatFormat;
}
extern "C" const char* rad_arch_id(void) { return "hyv3"; }
extern "C" const char* rad_arch_quant(void) { return HYV3_QUANT; }
extern "C" int rad_arch_declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* c) {
    return HYV3_NS::declare(b, m, c);
}
extern "C" void rad_arch_step(struct RadCtx* c, const struct RadBatch* batch) {
    HYV3_NS::step(c, batch);
}
extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
extern "C" const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = {
        RAD_PLUGIN_ARCH, HYV3_PLUGIN_NAME, "0.2.0",
        "Tencent Hy-Memory HYV3: Q/K-normalized full attention and sigmoid MoE", ""
    };
    return &info;
}
#endif

#undef HYV3_NS
#undef HYV3_QUANT
#undef HYV3_FP8_SOURCE
#undef HYV3_PLUGIN_NAME
#undef HYV3_GGUF_SOURCE
