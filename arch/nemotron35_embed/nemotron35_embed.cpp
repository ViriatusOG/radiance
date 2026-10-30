/* Nemotron-3.5-Embed: bidirectional BF16 text and a Pixtral encoder at TP1.
 *
 * The wrapper's language_model.* tensors are a Ministral3Model, without a logits head or Q/K
 * norms. Whole inputs attend bidirectionally and are mean-pooled after the final norm. The
 * endpoint normalises the vector after any dimensional truncation. Image rows replace only
 * their typed placeholders; separator tokens retain their vocabulary embeddings.
 *
 * Keep model-specific rotary interpretation here: the shared YaRN reader deliberately rejects
 * mscale, and changing its defaults would change unrelated models. This checkpoint's equal
 * mscale numerator/denominator give magnitude 1, independently of its position-dependent Q scale.
 */
#include <arch/rad_arch.h>
#include "pixtral.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace nemotron35_embed {
using namespace rad::arch;

struct Layer {
    rad_weight norm = 0, qkv = 0, out = 0, ff_norm = 0, gate_up = 0, down = 0;
    rad_op in_norm = 0, in_gain = 0, project = 0, rope = 0, qscale = 0, attn = 0, output = 0;
    rad_op add_attn = 0, pre_ff = 0, ff_gain = 0, gate = 0, act = 0, gate_mul = 0, ffn = 0,
           add_ff = 0;
};
struct Model {
    bool rerank = false;
    Geom g{};
    Names nm{""}, source{""};
    rad_buf x = 0, h = 0, qkv = 0, attn = 0, o = 0, gu = 0, ff = 0, mean = 0, pooled = 0;
    rad_weight tok = 0, norm = 0, score = 0;
    rad_op embed = 0, inject = 0, final_norm = 0, final_gain = 0, pool = 0, cast = 0, head = 0;
    std::vector<Layer> layers;
    Pixtral vision;
};
static Model model;

/* JSON booleans reach plugins as 0/1, while out-of-tree callers may use their JSON spelling. */
static bool false_value(const char* s) {
    return s && (!std::strcmp(s, "0") || !std::strcmp(s, "false"));
}

static int read_rope(Geom& g, const RadModelMeta* meta) {
    constexpr const char* pre = "text_config.rope_parameters.";
    auto val = [&](const char* key, const char* fallback = nullptr) {
        return rad_meta_gets(meta, (std::string(pre) + key).c_str(), fallback);
    };
    const char* type = val("rope_type", val("type"));
    if (!type || std::strcmp(type, "yarn")) return RAD_E_UNSUPPORTED;
    const double factor = rad_meta_getf(meta, "text_config.rope_parameters.factor", 0);
    const double num = rad_meta_getf(meta, "text_config.rope_parameters.mscale", NAN);
    const double den = rad_meta_getf(meta, "text_config.rope_parameters.mscale_all_dim", NAN);
    if (!(factor >= 1) || !std::isfinite(factor) || !std::isfinite(num) || !std::isfinite(den) ||
        num < 0 || den < 0)
        return RAD_E_INVAL;
    auto magnitude = [&](double m) { return factor > 1 ? 1 + 0.1 * m * std::log(factor) : 1; };
    const double mag = num && den ? magnitude(num) / magnitude(den) : magnitude(1);
    if (!(mag > 0) || !std::isfinite(mag)) return RAD_E_INVAL;

    /* Canonicalise only this plugin's temporary metadata view; never change the checkpoint or
     * shared reader. An explicit attention_factor, if present, is authoritative in Transformers. */
    std::vector<const char*> keys, vals;
    for (int i = 0; i < meta->n_kv; ++i) {
        const char* k = meta->kv_key[i];
        if (!std::strcmp(k, "text_config.rope_parameters.mscale") ||
            !std::strcmp(k, "text_config.rope_parameters.mscale_all_dim"))
            continue;
        keys.push_back(k);
        vals.push_back(meta->kv_val[i]);
    }
    char mag_text[48];
    std::snprintf(mag_text, sizeof mag_text, "%.17g", mag);
    const std::string derived = mag_text;
    if (!val("attention_factor")) {
        keys.push_back("text_config.rope_parameters.attention_factor");
        vals.push_back(derived.c_str());
    }
    RadModelMeta view = *meta;
    view.n_kv = (int)keys.size();
    view.kv_key = keys.data();
    view.kv_val = vals.data();
    return yarn_from(g, &view, "", g.head_dim);
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || ctx->max_seqs <= 0) return RAD_E_INVAL;
    if (ctx->rank != 0 || ctx->world_size > 1 || ctx->is_draft || ctx->max_spec > 0) {
        std::fprintf(stderr, "radiance: nemotron35_embed serves BF16 at TP1, without drafting\n");
        return RAD_E_UNSUPPORTED;
    }
    if (ctx->max_audio_frames > 0) {
        std::fprintf(stderr, "radiance: nemotron35_embed does not serve audio\n");
        return RAD_E_UNSUPPORTED;
    }
    if (ctx->max_ctx <= 0 || ctx->max_ctx > 32768) {
        std::fprintf(
            stderr,
            "radiance: nemotron35_embed: the embedding limit is 32768 tokens; set --max-model-len 32768 or lower\n");
        return RAD_E_INVAL;
    }
    if (std::strcmp(rad_meta_gets(meta, "model_type", ""), "mistral3") ||
        std::strcmp(rad_meta_gets(meta, "text_config.model_type", ""), "ministral3") ||
        !false_value(rad_meta_gets(meta, "is_causal", nullptr)) ||
        !false_value(rad_meta_gets(meta, "text_config.is_causal", nullptr)) ||
        std::strcmp(rad_meta_gets(meta, "text_config.hidden_act", ""), "silu") ||
        (meta->quant && *meta->quant))
        return RAD_E_UNSUPPORTED;
    const char* window = rad_meta_gets(meta, "text_config.sliding_window", nullptr);
    if (window && std::strcmp(window, "null") && std::strcmp(window, "0")) return RAD_E_UNSUPPORTED;
    const double beta =
        rad_meta_getf(meta, "text_config.rope_parameters.llama_4_scaling_beta", NAN);
    const int64_t original =
        rad_meta_geti(meta, "text_config.rope_parameters.original_max_position_embeddings", 0);
    if (!std::isfinite(beta) || beta < 0 || original <= 0) return RAD_E_INVAL;

    model = Model{};
    Model& m = model;
    m.rerank =
        !std::strcmp(rad_meta_gets(meta, "architectures", ""), "Mistral3ForSequenceClassification");
    const char* architecture = rad_meta_gets(meta, "architectures", "");
    if (*architecture && !m.rerank && std::strcmp(architecture, "Mistral3Model"))
        return RAD_E_UNSUPPORTED;
    if (m.rerank && (std::strcmp(rad_meta_gets(meta, "pooling", ""), "avg") ||
                     std::strcmp(rad_meta_gets(meta, "text_config.pooling", ""), "avg")))
        return RAD_E_UNSUPPORTED;
    m.nm = Names(ctx->scope);
    // The classifier wrapper prefixes the backbone with model.; its score head stays at root.
    m.source = Names(m.rerank ? "model" : "");
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));
    Geom& g = m.g;
    Names& nm = m.nm;
    Names& source = m.source;
    if (g.n_head_kv <= 0 || g.n_head % g.n_head_kv || g.head_dim % 2) return RAD_E_INVAL;
    RAD_ARCH_TRY(read_rope(g, meta));
    const int64_t T = g.max_tok, E = g.n_embd, Q = g.q_dim(), K = g.kv_dim(), F = g.n_ff;
    m.x = decl_b(b, nm.f("x"), RAD_BF16, {T, E});
    m.h = decl_b(b, nm.f("h"), RAD_BF16, {T, E});
    m.qkv = decl_b(b, nm.f("qkv"), RAD_BF16, {T, Q + 2 * K});
    m.attn = decl_b(b, nm.f("attn"), RAD_BF16, {T, Q});
    m.o = decl_b(b, nm.f("o"), RAD_BF16, {T, E});
    m.gu = decl_b(b, nm.f("gate_up"), RAD_BF16, {T, 2 * F});
    m.ff = decl_b(b, nm.f("ff"), RAD_BF16, {T, F});
    m.mean = decl_b(b, nm.f("mean"), RAD_BF16, {g.max_seqs, E});
    m.pooled = decl_b(b, nm.f("pooled"), RAD_F32, {g.max_seqs, m.rerank ? 1 : E}, RAD_BUF_PERSIST);
    RAD_ARCH_TRY(rad_declare_pooled(b, m.pooled));
    auto weight = [&](const char* dest, const char* src, std::initializer_list<int64_t> shape,
                      RadWeightGroup group) {
        if (map_copy(b, dest, src) < 0) return rad_weight(0);
        return decl_w(b, dest, RAD_BF16, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, group);
    };
    auto gemm = [&](int64_t N, int64_t kk, rad_weight w) {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", N), RAD_INT("K", kk),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto rms = [&] {
        // Ministral rounds the weightless norm before multiplying by the BF16 gain.
        return RAD_OP(b, "rmsnorm_p",
                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E), RAD_F64("eps", g.eps),
                                 RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS);
    };
    auto gain = [&](rad_weight w) {
        return RAD_OP(b, "mul_weight",
                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto add = [&] {
        return RAD_OP(b, "add",
                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E), RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS);
    };
    RAD_ARCH_TRY(
        map_copy(b, nm.f("token_embd.weight"), source.f("language_model.embed_tokens.weight")));
    m.tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, E}, RAD_ACCESS_VOCAB,
                   RAD_SHARD_NONE, grp_model());
    m.norm = weight(nm.f("output_norm.weight"), source.f("language_model.norm.weight"), {E},
                    grp_model());
    m.embed = rw(b,
                 RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", E),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16")),
                        RAD_WEIGHTS(m.tok)),
                 {}, {m.x});
    m.layers.resize((size_t)g.n_layer);
    for (int64_t i = 0; i < g.n_layer; ++i) {
        Layer& l = m.layers[(size_t)i];
        const auto group = grp_layer((int)i);
        const auto n = (long long)i;
        l.norm =
            weight(nm.f("blk.%lld.attn_norm.weight", n),
                   source.f("language_model.layers.%lld.input_layernorm.weight", n), {E}, group);
        l.ff_norm = weight(
            nm.f("blk.%lld.ffn_norm.weight", n),
            source.f("language_model.layers.%lld.post_attention_layernorm.weight", n), {E}, group);
        RAD_ARCH_TRY(map_concat(b, nm.f("blk.%lld.qkv.weight", n),
                                {source.f("language_model.layers.%lld.self_attn.q_proj.weight", n),
                                 source.f("language_model.layers.%lld.self_attn.k_proj.weight", n),
                                 source.f("language_model.layers.%lld.self_attn.v_proj.weight", n)},
                                0));
        l.qkv = decl_w(b, nm.f("blk.%lld.qkv.weight", n), RAD_BF16, {Q + 2 * K, E},
                       RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, group);
        l.out = weight(nm.f("blk.%lld.attn_o.weight", n),
                       source.f("language_model.layers.%lld.self_attn.o_proj.weight", n), {E, Q},
                       group);
        RAD_ARCH_TRY(map_concat(b, nm.f("blk.%lld.gate_up.weight", n),
                                {source.f("language_model.layers.%lld.mlp.gate_proj.weight", n),
                                 source.f("language_model.layers.%lld.mlp.up_proj.weight", n)},
                                0));
        l.gate_up = decl_w(b, nm.f("blk.%lld.gate_up.weight", n), RAD_BF16, {2 * F, E},
                           RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, group);
        l.down =
            weight(nm.f("blk.%lld.ffn_down.weight", n),
                   source.f("language_model.layers.%lld.mlp.down_proj.weight", n), {E, F}, group);
        l.in_norm = rw(b, rms(), {m.x}, {m.h});
        l.in_gain = rw(b, gain(l.norm), {m.h}, {m.h});
        l.project = rw(b, gemm(Q + 2 * K, E, l.qkv), {m.h}, {m.qkv});
        l.rope = rw(
            b,
            decl_rotary(b, g, "ministral_rope",
                        RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", g.head_dim),
                                   RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", g.n_head_kv),
                                   RAD_F64("theta", g.theta), RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
            {m.qkv}, {m.qkv});
        l.qscale = rw(
            b,
            RAD_OP(b, "position_scale",
                   RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", Q), RAD_INT("original", original),
                              RAD_F64("beta", beta), RAD_STR("dtype", "bf16")),
                   RAD_NOWEIGHTS),
            {m.qkv}, {m.qkv});
        l.attn = rw(b,
                    RAD_OP(b, "embed_attn",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", g.head_dim),
                                      RAD_INT("gqa", g.gqa()), RAD_INT("causal", 0),
                                      RAD_INT("max_seqlen", std::min(T, g.max_ctx)),
                                      RAD_STR("dtype", "bf16"),
                                      RAD_F64("scale", 1.0 / std::sqrt((double)g.head_dim))),
                           RAD_NOWEIGHTS),
                    {m.qkv}, {m.attn});
        l.output = rw(b, gemm(E, Q, l.out), {m.attn}, {m.o});
        l.add_attn = rw(b, add(), {m.x, m.o}, {m.x});
        l.pre_ff = rw(b, rms(), {m.x}, {m.h});
        l.ff_gain = rw(b, gain(l.ff_norm), {m.h}, {m.h});
        l.gate = rw(b, gemm(2 * F, E, l.gate_up), {m.h}, {m.gu});
        l.act =
            rw(b,
               RAD_OP(b, "silu",
                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", F), RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS),
               {m.gu}, {m.gu});
        l.gate_mul =
            rw(b,
               RAD_OP(b, "mul",
                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", F), RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS),
               {m.gu}, {m.ff});
        l.ffn = rw(b, gemm(E, F, l.down), {m.ff}, {m.o});
        l.add_ff = rw(b, add(), {m.x, m.o}, {m.x});
    }
    m.final_norm = rw(b, rms(), {m.x}, {m.h});
    m.final_gain = rw(b, gain(m.norm), {m.h}, {m.h});
    m.pool = rw(b,
                RAD_OP(b, m.rerank ? "seg_mean_bf16_sum" : "seg_mean",
                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E), RAD_STR("dtype", "bf16")),
                       RAD_NOWEIGHTS),
                {m.h}, {m.mean});
    if (m.rerank) {
        m.score = weight(nm.f("score.weight"), nm.ckpt("language_model.score.weight"), {1, E},
                         grp_model());
        m.head = rw(b,
                    RAD_OP(b, "logits_gemm",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("n_vocab", 1),
                                      RAD_INT("n_embd", E), RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(m.score)),
                    {m.mean}, {m.pooled});
    } else {
        m.cast = rw(b,
                    RAD_OP(b, "cast",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("n", E),
                                      RAD_STR("from", "bf16"), RAD_STR("to", "f32")),
                           RAD_NOWEIGHTS),
                    {m.mean}, {m.pooled});
    }
    /* Recipes may quantise weights without changing the architecture descriptor. Do not read an
     * INT8 plane with a BF16-only graph, even when a container still advertises quant="". */
    const RadEncoding plain = rad_enc_plain(RAD_BF16);
    auto bf16_only = [&](const char* name) -> int {
        RadEncoding enc{};
        const int rc = rad_weight_encoding(b, name, &enc, nullptr, nullptr);
        if (rc == RAD_E_NOTFOUND) return RAD_OK; // conversion has not populated the encoding yet
        if (rc < 0) return rc;
        if (rad_enc_equal(&enc, &plain)) return RAD_OK;
        std::fprintf(
            stderr,
            "radiance: nemotron35_embed requires BF16 weight '%s'; quantised linears are not implemented yet\n",
            name);
        return RAD_E_UNSUPPORTED;
    };
    RAD_ARCH_TRY(bf16_only(nm.f("token_embd.weight")));
    RAD_ARCH_TRY(bf16_only(nm.f("output_norm.weight")));
    if (m.rerank)
        RAD_ARCH_TRY(bf16_only(nm.f("score.weight")));
    for (int64_t i = 0; i < g.n_layer; ++i)
        for (const char* suffix : {"attn_norm", "ffn_norm", "qkv", "attn_o", "gate_up", "ffn_down"})
            RAD_ARCH_TRY(bf16_only(nm.f("blk.%lld.%s.weight", (long long)i, suffix)));
    RAD_ARCH_TRY(m.vision.declare(b, meta, ctx, nm, source, E, (int)g.n_layer));
    if (m.vision.on)
        m.inject =
            rw(b,
               RAD_OP(b, "scatter_rows",
                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E), RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS),
               {m.x}, {m.x});
    if (m.rerank)
        rad_note(b,
                 "Nemotron-3.5-Rerank: BF16 bidirectional query/document scoring, TP1; "
                 "BF16 sum/division, FP32 score; Pixtral encoder %s",
                 m.vision.on ? "on" : "off");
    else
        rad_note(
            b,
            "Nemotron-3.5-Embed: BF16 bidirectional embeddings, TP1; mean-pooled to %lld; Pixtral encoder %s",
            (long long)E, m.vision.on ? "on" : "off");
    return RAD_OK;
}

static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    if (!std::strcmp(rad_meta_gets(meta, "vision_config.model_type", ""), "pixtral")) {
        if (rad_meta_geti(meta, "spatial_merge_size", 0) != 2) return RAD_E_UNSUPPORTED;
        out->enc_merge = 4;
    }
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = model;
    const Geom& g = m.g;
    if (batch->enc) {
        m.vision.step(c, batch);
        return;
    }
    const int64_t T = batch->n_tok, S = batch->n_seq, Q = g.q_dim(), K = g.kv_dim();
    if (batch->phase != RAD_PHASE_PREFILL || T <= 0 || T > g.max_tok || S <= 0 || S > g.max_seqs ||
        batch->max_q_len > g.max_ctx || !batch->token_ids || !batch->positions ||
        !batch->cu_seqlens) {
        rad_step_fail(
            c,
            "nemotron35_embed: expected whole text inputs within the declared token/context bounds");
        return;
    }
    if (batch->n_mm_rows < 0 || batch->n_mm_rows > T ||
        (batch->n_mm_rows && (!m.inject || !batch->mm_rows || !batch->mm_embd))) {
        rad_step_fail(
            c,
            "nemotron35_embed: image rows require a declared Pixtral encoder and valid row data");
        return;
    }
    const auto pos = praw(batch->positions, RAD_I32, T),
               cu = praw(batch->cu_seqlens, RAD_I32, S + 1);
    RAD_ISSUE(c, m.embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.tok), brows(m.x, T));
    if (batch->n_mm_rows)
        RAD_ISSUE_N(c, m.inject, batch->n_mm_rows,
                    praw2(batch->mm_embd, RAD_BF16, batch->n_mm_rows, g.n_embd),
                    praw(batch->mm_rows, RAD_I32, batch->n_mm_rows), brows(m.x, T));
    for (const Layer& l : m.layers) {
        RAD_ISSUE(c, l.in_norm, brows(m.x, T), brows(m.h, T));
        RAD_ISSUE(c, l.in_gain, brows(m.h, T), RAD_W(l.norm), brows(m.h, T));
        RAD_ISSUE(c, l.project, brows(m.h, T), RAD_W(l.qkv), brows(m.qkv, T));
        RAD_ISSUE(c, l.rope, brows(m.qkv, T), pos);
        RAD_ISSUE(c, l.qscale, bcol(m.qkv, 0, Q, T), pos);
        RAD_ISSUE(c, l.attn, bcol(m.qkv, 0, Q, T), bcol(m.qkv, Q, K, T), bcol(m.qkv, Q + K, K, T),
                  cu, brows(m.attn, T));
        RAD_ISSUE(c, l.output, brows(m.attn, T), RAD_W(l.out), brows(m.o, T));
        RAD_ISSUE(c, l.add_attn, brows(m.x, T), brows(m.o, T), brows(m.x, T));
        RAD_ISSUE(c, l.pre_ff, brows(m.x, T), brows(m.h, T));
        RAD_ISSUE(c, l.ff_gain, brows(m.h, T), RAD_W(l.ff_norm), brows(m.h, T));
        RAD_ISSUE(c, l.gate, brows(m.h, T), RAD_W(l.gate_up), brows(m.gu, T));
        RAD_ISSUE(c, l.act, bcol(m.gu, 0, g.n_ff, T), bcol(m.gu, 0, g.n_ff, T));
        RAD_ISSUE(c, l.gate_mul, bcol(m.gu, 0, g.n_ff, T), bcol(m.gu, g.n_ff, g.n_ff, T),
                  brows(m.ff, T));
        RAD_ISSUE(c, l.ffn, brows(m.ff, T), RAD_W(l.down), brows(m.o, T));
        RAD_ISSUE(c, l.add_ff, brows(m.x, T), brows(m.o, T), brows(m.x, T));
    }
    RAD_ISSUE(c, m.final_norm, brows(m.x, T), brows(m.h, T));
    RAD_ISSUE(c, m.final_gain, brows(m.h, T), RAD_W(m.norm), brows(m.h, T));
    RAD_ISSUE(c, m.pool, brows(m.h, T), cu, brows(m.mean, S));
    if (m.rerank)
        RAD_ISSUE_N(c, m.head, S, brows(m.mean, S), RAD_W(m.score), brows(m.pooled, S));
    else
        RAD_ISSUE_N(c, m.cast, S, brows(m.mean, S), brows(m.pooled, S));
}
} // namespace nemotron35_embed

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(nemotron35_embed, "mistral3", "", "0.5.0",
                "Nemotron-3.5-Embed/Rerank: experimental BF16 text/image retrieval, TP1")
RAD_ARCH_PROBE(nemotron35_embed)
#endif
