/* laya -- Convai Innovations' Laya decision models (laya, laya-typed-decisions,
 * laya-multilingual), served on /v1/systemone. bf16 weights and activations, f32 norms and biases.
 *
 * A Laya checkpoint is a ModernBERT encoder and a small decision head (DecisionModel in laya's
 * common.py). It reads one sequence a question,
 *     [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
 * which the server builds (core/server/laya.cpp), and scores each option at its [MASK]. Like
 * nomicbert it declares a pooled output and no logits, so the engine runs it with no sampler and
 * no KV cache, every sequence whole in one step, and returns each sequence's pooled row: here one
 * logit an option (abi/rad_laya.h). The request's words carry what the tokens do not: the
 * question's type and where its markers sit.
 *
 * ============================== THE SHAPE ==============================
 *
 *   encoder (ModernBERT): token embedding, then LayerNorm; every norm has no bias, no linear has one
 *     h += Wo attn(rope(Wqkv LN_a(h)))       no LN_a on layer 0
 *     h += Wo (gelu(Wi_in LN_m(h)) * Wi_gate LN_m(h))     Wi is one [2 ff, E] tensor, [in | gate]
 *     a final LayerNorm
 *   attention: a `full_attention` layer (layer_types) attends over the whole sequence with the full
 *   rotary theta, the others over |i - j| <= local_attention / 2 with the sliding one; NeoX rotary
 *   over the whole head, scale 1 / sqrt(head_dim)
 *   head: h + type_emb[type]; head_layers pre-norm transformer layers (torch's
 *   TransformerEncoderLayer with norm_first: E / 64 heads, a ReLU feed-forward of 4 E, every
 *   linear and norm with a bias); the rows at the markers; LayerNorm, Linear + gelu, Linear to
 *   one logit an option
 *
 * The release's escalation head (act_head) and its stored temperatures are not declared: the
 * answer is the option logits, and the server applies the fitted temperatures rad-convert writes
 * into the header from rl_agent_config.json.
 */
#include <arch/rad_arch.h>
#include <rad_laya.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace laya {

using namespace rad::arch;

struct Layer {
    bool sliding = false;
    rad_weight w_attn_norm = 0, w_qkv = 0, w_out = 0, w_mlp_norm = 0, w_wi = 0, w_wo = 0;
    rad_op op_attn_norm = 0, op_qkv = 0, op_rope = 0, op_attn = 0, op_out = 0, op_add_attn = 0;
    rad_op op_mlp_norm = 0, op_wi = 0, op_act = 0, op_wo = 0, op_add_mlp = 0;
};

struct HeadLayer {
    rad_weight n1_w = 0, n1_b = 0, qkv_w = 0, qkv_b = 0, out_w = 0, out_b = 0;
    rad_weight n2_w = 0, n2_b = 0, l1_w = 0, l1_b = 0, l2_w = 0, l2_b = 0;
    rad_op op_n1 = 0, op_qkv = 0, op_attn = 0, op_out = 0, op_n2 = 0, op_l1 = 0, op_relu = 0,
           op_l2 = 0;
};

struct Model {
    Names nm{""};
    Geom  g{};
    int64_t F = 0;          /* the encoder's feed-forward width */
    int64_t head_dim_h = 0; /* the head's attention width, E / max(1, E / 64) */

    rad_buf b_o = 0, b_x = 0, b_h = 0, b_t = 0, b_qkv = 0, b_att = 0, b_wi = 0, b_ff = 0,
            b_hff = 0, b_trow = 0, b_m = 0, b_m2 = 0, b_sc = 0, b_pooled = 0;

    rad_weight w_tok = 0, w_emb_norm = 0, w_final_norm = 0, w_type = 0;
    rad_weight w_sn = 0, b_sn = 0, w_s1 = 0, b_s1 = 0, w_s2 = 0, b_s2 = 0;
    rad_op op_embed = 0, op_emb_norm = 0, op_final_norm = 0, op_type = 0, op_add_type = 0;
    rad_op op_gather = 0, op_sn = 0, op_s1 = 0, op_s2 = 0, op_fill = 0, op_out = 0;

    std::vector<Layer>     layers;
    std::vector<HeadLayer> head;
};

static Model g_model[MAX_RANKS];

static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->draft_depth = 0;
    const char* dt = rad_meta_gets(meta, "decision.type", nullptr);
    if (!dt || std::strcmp(dt, "laya")) {
        fprintf(stderr, "radiance: laya: the container declares no Laya decision head "
                        "(decision.type=laya); convert the checkpoint directory with rad-convert, "
                        "which reads rl_agent_config.json beside it\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* Which layers attend over the sliding window: `layer_types` as the config states it, or every
 * layer but each global_attn_every_n_layers-th in configs written before transformers 5. */
static int sliding_layers(const RadModelMeta* meta, int64_t n_layer, std::vector<bool>* out) {
    out->assign((size_t)n_layer, false);
    if (const char* lt = rad_meta_gets(meta, "layer_types", nullptr)) {
        int64_t l = 0;
        for (const char* p = lt; *p;) {
            while (*p == ' ') ++p;
            const char* e = p;
            while (*e && *e != ' ') ++e;
            if (e == p) break;
            const std::string t(p, e);
            if (l >= n_layer || (t != "full_attention" && t != "sliding_attention")) return RAD_E_INVAL;
            (*out)[(size_t)l++] = t == "sliding_attention";
            p = e;
        }
        return l == n_layer ? RAD_OK : RAD_E_INVAL;
    }
    const int64_t every = rad_meta_geti(meta, "global_attn_every_n_layers", 0);
    if (every <= 0) return RAD_E_INVAL;
    for (int64_t l = 0; l < n_layer; ++l) (*out)[(size_t)l] = l % every != 0;
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    if (ctx->world_size > 1) {
        fprintf(stderr, "radiance: laya runs on one card (--tp 1); the largest checkpoint is "
                        "0.4B parameters\n");
        return RAD_E_UNSUPPORTED;
    }
    RadArchProbe pr{};
    RAD_ARCH_TRY(probe(meta, &pr));

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));
    Geom&  g  = m.g;
    Names& nm = m.nm;
    const int64_t T = g.max_tok, E = g.n_embd, S = g.max_seqs, P = RAD_LAYA_POOLED;
    m.F = g.n_ff;
    const int64_t F = m.F;

    /* The variant flags this plugin does not serve, refused by name. */
    for (const char* k : { "norm_bias", "attention_bias", "mlp_bias" })
        if (rad_meta_geti(meta, k, 0) != 0) {
            fprintf(stderr, "radiance: laya: the encoder sets %s; ModernBERT as Laya ships it has "
                            "no bias there\n", k);
            return RAD_E_UNSUPPORTED;
        }
    const char* act = rad_meta_gets(meta, "hidden_activation", "gelu");
    if (std::strcmp(act, "gelu")) {
        fprintf(stderr, "radiance: laya: hidden_activation '%s'; this plugin serves exact gelu\n", act);
        return RAD_E_UNSUPPORTED;
    }
    if (g.n_head_kv != g.n_head || g.head_dim * g.n_head != E) {
        fprintf(stderr, "radiance: laya: %lld heads of %lld (%lld for keys) do not tile hidden %lld\n",
                (long long)g.n_head, (long long)g.head_dim, (long long)g.n_head_kv, (long long)E);
        return RAD_E_INVAL;
    }
    std::vector<bool> sliding;
    if (sliding_layers(meta, g.n_layer, &sliding) != RAD_OK) {
        fprintf(stderr, "radiance: laya: neither layer_types (%lld entries of full_attention or "
                        "sliding_attention) nor global_attn_every_n_layers says which layers "
                        "attend locally\n", (long long)g.n_layer);
        return RAD_E_INVAL;
    }
    const int64_t local = rad_meta_geti(meta, "local_attention", 128);
    /* transformers masks the local layers to |i - j| <= local_attention / 2; attn_dense's window
     * keeps |i - j| < window. */
    const int64_t window = local / 2 + 1;
    const double theta_full = rad_meta_getf(meta, "rope_parameters.full_attention.rope_theta",
                                            rad_meta_getf(meta, "global_rope_theta", 160000.0));
    const double theta_slide = rad_meta_getf(meta, "rope_parameters.sliding_attention.rope_theta",
                                             rad_meta_getf(meta, "local_rope_theta", 10000.0));
    const int64_t n_head_layers = rad_meta_geti(meta, "decision.laya.head_layers", -1);
    if (local < 2 || !(theta_full > 0.0) || !(theta_slide > 0.0) || n_head_layers < 0) {
        fprintf(stderr, "radiance: laya: local_attention %lld, rotary thetas %g and %g, or "
                        "decision.laya.head_layers %lld is not a Laya model\n", (long long)local,
                theta_full, theta_slide, (long long)n_head_layers);
        return RAD_E_INVAL;
    }
    /* torch's MultiheadAttention in the head: E / 64 heads, at least one. */
    const int64_t hh = std::max<int64_t>(1, E / 64);
    m.head_dim_h = E / hh;
    if (m.head_dim_h * hh != E) {
        fprintf(stderr, "radiance: laya: hidden %lld does not split into %lld head heads\n",
                (long long)E, (long long)hh);
        return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(rad_declare_plugin_words(b, RAD_LAYA_MAX_WORDS));

    /* ---- activations. b_sc is P rows longer than any step's markers: the last sequence's pooled
     * row is copied from P scores, its own and then filler. */
    m.b_o    = decl_b(b, nm.f("o"),    RAD_BF16, { T, E });
    m.b_x    = decl_b(b, nm.f("x"),    RAD_BF16, { T, E });
    m.b_h    = decl_b(b, nm.f("h"),    RAD_BF16, { T, E });
    m.b_t    = decl_b(b, nm.f("t"),    RAD_BF16, { T, E });
    m.b_qkv  = decl_b(b, nm.f("qkv"),  RAD_BF16, { T, 3 * E });
    m.b_att  = decl_b(b, nm.f("attn"), RAD_BF16, { T, E });
    m.b_wi   = decl_b(b, nm.f("wi"),   RAD_BF16, { T, 2 * F });
    m.b_ff   = decl_b(b, nm.f("ff"),   RAD_BF16, { T, F });
    m.b_hff  = decl_b(b, nm.f("hff"),  RAD_BF16, { T, 4 * E });
    m.b_trow = decl_b(b, nm.f("type"), RAD_BF16, { S, E });
    m.b_m    = decl_b(b, nm.f("mark"), RAD_BF16, { T, E });
    m.b_m2   = decl_b(b, nm.f("mark2"), RAD_BF16, { T, E });
    m.b_sc   = decl_b(b, nm.f("score"), RAD_BF16, { T + P, 1 });
    m.b_pooled = decl_b(b, nm.f("pooled"), RAD_F32, { S, P }, RAD_BUF_PERSIST);
    if (!m.b_o || !m.b_x || !m.b_h || !m.b_t || !m.b_qkv || !m.b_att || !m.b_wi || !m.b_ff ||
        !m.b_hff || !m.b_trow || !m.b_m || !m.b_m2 || !m.b_sc || !m.b_pooled)
        return RAD_E_INVAL;
    RAD_ARCH_TRY(rad_declare_pooled(b, m.b_pooled));

    /* ---- weights. Linears and tables bf16, norms and biases f32. */
    const int A = RAD_ACCESS_PER_TOKEN, N = RAD_SHARD_NONE;
    auto w_model = [&](const char* name, const char* src, uint32_t dt,
                       std::initializer_list<int64_t> shape, int access = RAD_ACCESS_PER_TOKEN) {
        if (map_copy(b, nm.f("%s", name), nm.ckpt("%s", src)) != RAD_OK) return (rad_weight)0;
        return decl_w(b, nm.f("%s", name), dt, shape, access, N, grp_model());
    };
    m.w_tok        = w_model("token_embd.weight", "encoder.embeddings.tok_embeddings.weight",
                             RAD_BF16, { g.n_vocab, E }, RAD_ACCESS_VOCAB);
    m.w_emb_norm   = w_model("emb_norm.weight", "encoder.embeddings.norm.weight", RAD_F32, { E });
    m.w_final_norm = w_model("output_norm.weight", "encoder.final_norm.weight", RAD_F32, { E });
    m.w_type       = w_model("type_emb.weight", "type_emb.weight", RAD_BF16, { 3, E });
    m.w_sn         = w_model("scorer.norm.weight", "scorer.0.weight", RAD_F32, { E });
    m.b_sn         = w_model("scorer.norm.bias",   "scorer.0.bias",   RAD_F32, { E });
    m.w_s1         = w_model("scorer.fc1.weight",  "scorer.1.weight", RAD_BF16, { E, E });
    m.b_s1         = w_model("scorer.fc1.bias",    "scorer.1.bias",   RAD_F32, { E });
    m.w_s2         = w_model("scorer.fc2.weight",  "scorer.3.weight", RAD_BF16, { 1, E });
    m.b_s2         = w_model("scorer.fc2.bias",    "scorer.3.bias",   RAD_F32, { 1 });
    if (!m.w_tok || !m.w_emb_norm || !m.w_final_norm || !m.w_type || !m.w_sn || !m.b_sn ||
        !m.w_s1 || !m.b_s1 || !m.w_s2 || !m.b_s2)
        return RAD_E_INVAL;

    /* ---- ops, each kind stated once. */
    auto lnorm = [&](int64_t rows, bool bias, rad_weight w, rad_weight bw) {
        return bias ? RAD_OP(b, "layernorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", E),
                                        RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(w, bw))
                    : RAD_OP(b, "layernorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", E),
                                        RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(w));
    };
    auto gemm = [&](int64_t rows, int64_t n_out, int64_t n_in, rad_weight w) {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", n_out), RAD_INT("K", n_in),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto gemmb = [&](int64_t rows, int64_t n_out, int64_t n_in, const char* act_name, rad_weight w,
                     rad_weight bw) {
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", n_out), RAD_INT("K", n_in),
                                 RAD_STR("dtype", "bf16"), RAD_STR("act", act_name)),
                      RAD_WEIGHTS(w, bw));
    };
    auto unary = [&](const char* op, int64_t rows, int64_t n) {
        return RAD_OP(b, op,
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS);
    };
    auto attn = [&](int64_t head_dim, double scale, int64_t win) {
        return win > 0 ? RAD_OP(b, "attn_dense",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", head_dim),
                                           RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                           RAD_STR("dtype", "bf16"), RAD_F64("scale", scale),
                                           RAD_INT("max_seqlen", T), RAD_INT("window", win)),
                                RAD_NOWEIGHTS)
                       : RAD_OP(b, "attn_dense",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", head_dim),
                                           RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                           RAD_STR("dtype", "bf16"), RAD_F64("scale", scale),
                                           RAD_INT("max_seqlen", T)),
                                RAD_NOWEIGHTS);
    };
    auto rope = [&](double theta) {
        return RAD_OP(b, "rope",
                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", g.head_dim),
                                 RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", g.n_head_kv),
                                 RAD_F64("theta", theta), RAD_F64("scale", 1.0),
                                 RAD_STR("mode", "neox")),
                      RAD_NOWEIGHTS);
    };

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", E),
                                         RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", "bf16"),
                                         RAD_INT("vocab_offset", 0), RAD_F64("mult", 1.0)),
                              RAD_WEIGHTS(m.w_tok)),
                    {}, { m.b_o });
    m.op_emb_norm = rw(b, lnorm(T, false, m.w_emb_norm, 0), { m.b_o }, { m.b_x });

    m.layers.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& y = m.layers[(size_t)l];
        y.sliding = sliding[(size_t)l];
        const long long ll = (long long)l;
        const RadWeightGroup gl = grp_layer((int)l);
        auto src = [&](const char* s) { return nm.ckpt("encoder.layers.%lld.%s", ll, s); };
        if (l > 0) RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_norm.weight", ll), src("attn_norm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_qkv.weight", ll), src("attn.Wqkv.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_out.weight", ll), src("attn.Wo.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_norm.weight", ll), src("mlp_norm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_in_gate.weight", ll), src("mlp.Wi.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_down.weight", ll), src("mlp.Wo.weight")));
        if (l > 0)
            y.w_attn_norm = decl_w(b, nm.f("blk.%lld.attn_norm.weight", ll), RAD_F32, { E }, A, N, gl);
        y.w_qkv      = decl_w(b, nm.f("blk.%lld.attn_qkv.weight", ll), RAD_BF16, { 3 * E, E }, A, N, gl);
        y.w_out      = decl_w(b, nm.f("blk.%lld.attn_out.weight", ll), RAD_BF16, { E, E }, A, N, gl);
        y.w_mlp_norm = decl_w(b, nm.f("blk.%lld.ffn_norm.weight", ll), RAD_F32, { E }, A, N, gl);
        y.w_wi       = decl_w(b, nm.f("blk.%lld.ffn_in_gate.weight", ll), RAD_BF16, { 2 * F, E }, A, N, gl);
        y.w_wo       = decl_w(b, nm.f("blk.%lld.ffn_down.weight", ll), RAD_BF16, { E, F }, A, N, gl);
        if ((l > 0 && !y.w_attn_norm) || !y.w_qkv || !y.w_out || !y.w_mlp_norm || !y.w_wi || !y.w_wo)
            return RAD_E_INVAL;

        const rad_buf att_in = l > 0 ? m.b_h : m.b_x;
        if (l > 0) y.op_attn_norm = rw(b, lnorm(T, false, y.w_attn_norm, 0), { m.b_x }, { m.b_h });
        y.op_qkv  = rw(b, gemm(T, 3 * E, E, y.w_qkv), { att_in }, { m.b_qkv });
        y.op_rope = rw(b, rope(y.sliding ? theta_slide : theta_full), { m.b_qkv }, { m.b_qkv });
        y.op_attn = rw(b, attn(g.head_dim, 1.0 / std::sqrt((double)g.head_dim), y.sliding ? window : 0),
                       { m.b_qkv }, { m.b_att });
        y.op_out  = rw(b, gemm(T, E, E, y.w_out), { m.b_att }, { m.b_o });
        y.op_add_attn = rw(b, unary("add", T, E), { m.b_x, m.b_o }, { m.b_x });
        y.op_mlp_norm = rw(b, lnorm(T, false, y.w_mlp_norm, 0), { m.b_x }, { m.b_h });
        y.op_wi   = rw(b, gemm(T, 2 * F, E, y.w_wi), { m.b_h }, { m.b_wi });
        y.op_act  = rw(b, RAD_OP(b, "act_mul",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", F),
                                            RAD_STR("act", "gelu"), RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       { m.b_wi }, { m.b_ff });
        y.op_wo   = rw(b, gemm(T, E, F, y.w_wo), { m.b_ff }, { m.b_o });
        y.op_add_mlp = rw(b, unary("add", T, E), { m.b_x, m.b_o }, { m.b_x });
        if ((l > 0 && !y.op_attn_norm) || !y.op_qkv || !y.op_rope || !y.op_attn || !y.op_out ||
            !y.op_add_attn || !y.op_mlp_norm || !y.op_wi || !y.op_act || !y.op_wo || !y.op_add_mlp)
            return RAD_E_INVAL;
    }
    m.op_final_norm = rw(b, lnorm(T, false, m.w_final_norm, 0), { m.b_x }, { m.b_h });

    /* ---- the head, over the encoder's rows in b_h. */
    m.op_type = rw(b, RAD_OP(b, "embed_lookup",
                             RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n_embd", E), RAD_INT("n_vocab", 3),
                                        RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", 0)),
                             RAD_WEIGHTS(m.w_type)),
                   {}, { m.b_trow });
    m.op_add_type = rw(b, unary("add", T, E), { m.b_h, m.b_trow }, { m.b_h });
    m.head.resize((size_t)n_head_layers);
    for (int64_t j = 0; j < n_head_layers; ++j) {
        HeadLayer& h = m.head[(size_t)j];
        const long long jj = (long long)j;
        auto hw = [&](const char* name, const char* src, uint32_t dt,
                      std::initializer_list<int64_t> shape) {
            if (map_copy(b, nm.f("head.%lld.%s", jj, name), nm.ckpt("head.layers.%lld.%s", jj, src)) != RAD_OK)
                return (rad_weight)0;
            return decl_w(b, nm.f("head.%lld.%s", jj, name), dt, shape, A, N, grp_model());
        };
        h.n1_w  = hw("norm1.weight",    "norm1.weight",              RAD_F32,  { E });
        h.n1_b  = hw("norm1.bias",      "norm1.bias",                RAD_F32,  { E });
        h.qkv_w = hw("attn_qkv.weight", "self_attn.in_proj_weight",  RAD_BF16, { 3 * E, E });
        h.qkv_b = hw("attn_qkv.bias",   "self_attn.in_proj_bias",    RAD_F32,  { 3 * E });
        h.out_w = hw("attn_out.weight", "self_attn.out_proj.weight", RAD_BF16, { E, E });
        h.out_b = hw("attn_out.bias",   "self_attn.out_proj.bias",   RAD_F32,  { E });
        h.n2_w  = hw("norm2.weight",    "norm2.weight",              RAD_F32,  { E });
        h.n2_b  = hw("norm2.bias",      "norm2.bias",                RAD_F32,  { E });
        h.l1_w  = hw("ffn_up.weight",   "linear1.weight",            RAD_BF16, { 4 * E, E });
        h.l1_b  = hw("ffn_up.bias",     "linear1.bias",              RAD_F32,  { 4 * E });
        h.l2_w  = hw("ffn_down.weight", "linear2.weight",            RAD_BF16, { E, 4 * E });
        h.l2_b  = hw("ffn_down.bias",   "linear2.bias",              RAD_F32,  { E });
        if (!h.n1_w || !h.n1_b || !h.qkv_w || !h.qkv_b || !h.out_w || !h.out_b || !h.n2_w ||
            !h.n2_b || !h.l1_w || !h.l1_b || !h.l2_w || !h.l2_b)
            return RAD_E_INVAL;
        h.op_n1   = rw(b, lnorm(T, true, h.n1_w, h.n1_b), { m.b_h }, { m.b_t });
        h.op_qkv  = rw(b, gemmb(T, 3 * E, E, "none", h.qkv_w, h.qkv_b), { m.b_t }, { m.b_qkv });
        h.op_attn = rw(b, attn(m.head_dim_h, 1.0 / std::sqrt((double)m.head_dim_h), 0),
                       { m.b_qkv }, { m.b_att });
        h.op_out  = rw(b, gemmb(T, E, E, "none", h.out_w, h.out_b), { m.b_att, m.b_h }, { m.b_h });
        h.op_n2   = rw(b, lnorm(T, true, h.n2_w, h.n2_b), { m.b_h }, { m.b_t });
        h.op_l1   = rw(b, gemmb(T, 4 * E, E, "none", h.l1_w, h.l1_b), { m.b_t }, { m.b_hff });
        h.op_relu = rw(b, unary("relu", T, 4 * E), { m.b_hff }, { m.b_hff });
        h.op_l2   = rw(b, gemmb(T, E, 4 * E, "none", h.l2_w, h.l2_b), { m.b_hff, m.b_h }, { m.b_h });
        if (!h.op_n1 || !h.op_qkv || !h.op_attn || !h.op_out || !h.op_n2 || !h.op_l1 ||
            !h.op_relu || !h.op_l2)
            return RAD_E_INVAL;
    }
    /* The rows at the markers, a sequence at a time, then the scorer over all of them. */
    m.op_gather = rw(b, unary("gather_rows", RAD_LAYA_MAX_OPTIONS, E), { m.b_h }, { m.b_m });
    m.op_sn     = rw(b, lnorm(T, true, m.w_sn, m.b_sn), { m.b_m }, { m.b_t });
    m.op_s1     = rw(b, gemmb(T, E, E, "gelu", m.w_s1, m.b_s1), { m.b_t }, { m.b_m2 });
    m.op_s2     = rw(b, gemmb(T, 1, E, "none", m.w_s2, m.b_s2), { m.b_m2 }, { m.b_sc });
    /* Past the last marker's score, P filler rows the last pooled row's copy may read. */
    m.op_fill   = rw(b, RAD_OP(b, "fill",
                               RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", P),
                                          RAD_STR("dtype", "bf16"), RAD_F64("value", -1e4)),
                               RAD_NOWEIGHTS),
                     {}, { m.b_sc });
    m.op_out    = rw(b, RAD_OP(b, "cast",
                               RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", P),
                                          RAD_STR("from", "bf16"), RAD_STR("to", "f32")),
                               RAD_NOWEIGHTS),
                     { m.b_sc }, { m.b_pooled });
    if (!m.op_embed || !m.op_emb_norm || !m.op_final_norm || !m.op_type || !m.op_add_type ||
        !m.op_gather || !m.op_sn || !m.op_s1 || !m.op_s2 || !m.op_fill || !m.op_out)
        return RAD_E_INVAL;

    int64_t n_sliding = 0;
    for (bool s : sliding) n_sliding += s ? 1 : 0;
    rad_note(b, "laya decision model: ModernBERT %lld layers (%lld sliding over %lld, rotary %g "
                "and %g), hidden %lld, ffn %lld, %lld heads of %lld; head %lld layers of %lld "
                "heads; at most %d options a question, one logit each in the pooled row",
             (long long)g.n_layer, (long long)n_sliding, (long long)local, theta_full, theta_slide,
             (long long)E, (long long)F, (long long)g.n_head, (long long)g.head_dim,
             (long long)n_head_layers, (long long)hh, (int)RAD_LAYA_MAX_OPTIONS);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_rank(c)];
    const Geom& g = m.g;
    const int64_t T = batch->n_tok, S = batch->n_seq, E = g.n_embd, F = m.F;
    const int64_t P = RAD_LAYA_POOLED;
    if (T <= 0 || S <= 0) return;
    if (!batch->plugin_cu_h || !batch->plugin_words_h || !batch->plugin_words) {
        rad_step_fail(c, "laya: the step carries no words; every Laya sequence comes from "
                         "/v1/systemone with its question's type and markers");
        return;
    }
    /* Each sequence's first row from the words: an embedding input runs whole in one step, so
     * the sequences' token counts add up to the step's. */
    std::vector<int64_t> first((size_t)S + 1, 0);
    for (int64_t s = 0; s < S; ++s) {
        const int32_t w0 = batch->plugin_cu_h[s], w1 = batch->plugin_cu_h[s + 1];
        const int32_t* hw = batch->plugin_words_h + w0;
        const int64_t n = w1 - w0 > RAD_LAYA_W_K ? hw[RAD_LAYA_W_N] : -1;
        const int64_t k = w1 - w0 > RAD_LAYA_W_K ? hw[RAD_LAYA_W_K] : -1;
        const int64_t type = w1 - w0 > RAD_LAYA_W_K ? hw[RAD_LAYA_W_TYPE] : -1;
        bool ok = n > 0 && k >= 1 && k <= RAD_LAYA_MAX_OPTIONS && w1 - w0 >= RAD_LAYA_W_MARKERS + k &&
                  type >= RAD_LAYA_CHOICE && type <= RAD_LAYA_NOUL;
        for (int64_t i = 0; ok && i < k; ++i) {
            const int32_t at = hw[RAD_LAYA_W_MARKERS + i];
            ok = at >= 0 && at < n;
        }
        if (!ok) {
            rad_step_fail(c, "laya: a sequence's words are not a Laya question's (type, markers)");
            return;
        }
        first[(size_t)s + 1] = first[(size_t)s] + n;
    }
    if (first[(size_t)S] != T) {
        rad_step_fail(c, "laya: the words' token counts do not add up to the step's tokens");
        return;
    }
    const RadOperand cu  = praw(batch->cu_seqlens, RAD_I32, S + 1);
    const RadOperand pos = praw(batch->positions, RAD_I32, T);

    /* ---- the encoder over every sequence at once. */
    RAD_ISSUE_N(c, m.op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_o, T));
    RAD_ISSUE_N(c, m.op_emb_norm, T, brows(m.b_o, T), RAD_W(m.w_emb_norm), RAD_NONE, brows(m.b_x, T));
    for (size_t l = 0; l < m.layers.size(); ++l) {
        const Layer& y = m.layers[l];
        rad_buf att_in = m.b_x;
        if (l > 0) {
            RAD_ISSUE_N(c, y.op_attn_norm, T, brows(m.b_x, T), RAD_W(y.w_attn_norm), RAD_NONE,
                        brows(m.b_h, T));
            att_in = m.b_h;
        }
        RAD_ISSUE_N(c, y.op_qkv, T, brows(att_in, T), RAD_W(y.w_qkv), brows(m.b_qkv, T));
        RAD_ISSUE_N(c, y.op_rope, T, brows(m.b_qkv, T), pos);
        RAD_ISSUE_N(c, y.op_attn, T, bcol(m.b_qkv, 0, E, T), bcol(m.b_qkv, E, E, T),
                    bcol(m.b_qkv, 2 * E, E, T), cu, brows(m.b_att, T));
        RAD_ISSUE_N(c, y.op_out, T, brows(m.b_att, T), RAD_W(y.w_out), brows(m.b_o, T));
        RAD_ISSUE_N(c, y.op_add_attn, T, brows(m.b_x, T), brows(m.b_o, T), brows(m.b_x, T));
        RAD_ISSUE_N(c, y.op_mlp_norm, T, brows(m.b_x, T), RAD_W(y.w_mlp_norm), RAD_NONE,
                    brows(m.b_h, T));
        RAD_ISSUE_N(c, y.op_wi, T, brows(m.b_h, T), RAD_W(y.w_wi), brows(m.b_wi, T));
        RAD_ISSUE_N(c, y.op_act, T, bcol(m.b_wi, 0, F, T), bcol(m.b_wi, F, F, T), brows(m.b_ff, T));
        RAD_ISSUE_N(c, y.op_wo, T, brows(m.b_ff, T), RAD_W(y.w_wo), brows(m.b_o, T));
        RAD_ISSUE_N(c, y.op_add_mlp, T, brows(m.b_x, T), brows(m.b_o, T), brows(m.b_x, T));
    }
    RAD_ISSUE_N(c, m.op_final_norm, T, brows(m.b_x, T), RAD_W(m.w_final_norm), RAD_NONE,
                brows(m.b_h, T));

    /* ---- the head: each sequence's type row added over its rows, then the head's layers. */
    for (int64_t s = 0; s < S; ++s) {
        const int32_t* dw = batch->plugin_words + batch->plugin_cu_h[s];
        const int64_t r0 = first[(size_t)s], n = first[(size_t)s + 1] - r0;
        RAD_ISSUE_N(c, m.op_type, 1, praw(dw + RAD_LAYA_W_TYPE, RAD_I32, 1), RAD_W(m.w_type),
                    brow_slice(m.b_trow, s, 1, E));
        RAD_ISSUE_N(c, m.op_add_type, n, brow_slice(m.b_h, r0, n, E), brow_slice(m.b_trow, s, 1, E),
                    brow_slice(m.b_h, r0, n, E));
    }
    for (const HeadLayer& h : m.head) {
        RAD_ISSUE_N(c, h.op_n1, T, brows(m.b_h, T), RAD_W(h.n1_w), RAD_W(h.n1_b), brows(m.b_t, T));
        RAD_ISSUE_N(c, h.op_qkv, T, brows(m.b_t, T), RAD_W(h.qkv_w), RAD_W(h.qkv_b), RAD_NONE,
                    brows(m.b_qkv, T));
        RAD_ISSUE_N(c, h.op_attn, T, bcol(m.b_qkv, 0, E, T), bcol(m.b_qkv, E, E, T),
                    bcol(m.b_qkv, 2 * E, E, T), cu, brows(m.b_att, T));
        RAD_ISSUE_N(c, h.op_out, T, brows(m.b_att, T), RAD_W(h.out_w), RAD_W(h.out_b),
                    brows(m.b_h, T), brows(m.b_h, T));
        RAD_ISSUE_N(c, h.op_n2, T, brows(m.b_h, T), RAD_W(h.n2_w), RAD_W(h.n2_b), brows(m.b_t, T));
        RAD_ISSUE_N(c, h.op_l1, T, brows(m.b_t, T), RAD_W(h.l1_w), RAD_W(h.l1_b), RAD_NONE,
                    brows(m.b_hff, T));
        RAD_ISSUE_N(c, h.op_relu, T, brows(m.b_hff, T), brows(m.b_hff, T));
        RAD_ISSUE_N(c, h.op_l2, T, brows(m.b_hff, T), RAD_W(h.l2_w), RAD_W(h.l2_b),
                    brows(m.b_h, T), brows(m.b_h, T));
    }

    /* ---- the markers' rows, consecutive across sequences, and one score each. */
    std::vector<int64_t> mark((size_t)S + 1, 0);
    for (int64_t s = 0; s < S; ++s) {
        const int32_t* dw = batch->plugin_words + batch->plugin_cu_h[s];
        const int64_t k = batch->plugin_words_h[batch->plugin_cu_h[s] + RAD_LAYA_W_K];
        const int64_t r0 = first[(size_t)s], n = first[(size_t)s + 1] - r0;
        RAD_ISSUE_N(c, m.op_gather, k, brow_slice(m.b_h, r0, n, E),
                    praw(dw + RAD_LAYA_W_MARKERS, RAD_I32, k), brow_slice(m.b_m, mark[(size_t)s], k, E));
        mark[(size_t)s + 1] = mark[(size_t)s] + k;
    }
    const int64_t K = mark[(size_t)S];
    RAD_ISSUE_N(c, m.op_sn, K, brows(m.b_m, K), RAD_W(m.w_sn), RAD_W(m.b_sn), brows(m.b_t, K));
    RAD_ISSUE_N(c, m.op_s1, K, brows(m.b_t, K), RAD_W(m.w_s1), RAD_W(m.b_s1), RAD_NONE, brows(m.b_m2, K));
    RAD_ISSUE_N(c, m.op_s2, K, brows(m.b_m2, K), RAD_W(m.w_s2), RAD_W(m.b_s2), RAD_NONE, brows(m.b_sc, K));
    RAD_ISSUE_N(c, m.op_fill, 1, brow_slice(m.b_sc, K, P, 1));
    /* Each sequence's pooled row: its k scores and, past them, the next scores or filler, which
     * the server does not read. */
    for (int64_t s = 0; s < S; ++s)
        RAD_ISSUE_N(c, m.op_out, 1, brow_slice(m.b_sc, mark[(size_t)s], P, 1),
                    bcol_at(m.b_pooled, s, P, 0, P, 1));
}

}  /* namespace laya */

RAD_ARCH_PLUGIN(laya, "laya", "", "0.1.0",
                "Convai Innovations' Laya (laya, laya-typed-decisions, laya-multilingual): a "
                "ModernBERT encoder and a decision head scoring each option at its [MASK], one "
                "sequence a question, served on /v1/systemone; bf16")
RAD_ARCH_PROBE(laya)
