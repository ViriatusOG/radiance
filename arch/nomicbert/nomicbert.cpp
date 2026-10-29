/* nomicbert -- Nomic BERT (model_type nomic_bert), the encoder behind nomic-embed-text-v2-moe.
 * bf16 weights and activations, with f32 norms and biases.
 *
 * An embedding model: it declares a pooled output (rad_declare_pooled) and no logits, so the
 * engine runs it with no sampler and no KV cache, carries every input whole in one step, and
 * returns each sequence's pooled row. Attention is bidirectional on every layer, so nothing is
 * paged: a token attends to later tokens, and there is nothing to carry between steps.
 *
 * Shape:
 *
 *   12 layers, hidden 768, 12 heads of 64, ffn 3072, vocab 250048, type vocab 1
 *   post-norm block, the config's `prenorm: false`:
 *     a = out_proj(attn(rope(Wqkv h + b))) + b_out; h = LN1(h + a)
 *     dense layer: m = fc2(gelu(fc1 h + b1)) + b2; h = LN2(h + m)
 *     MoE layer (every second, moe_every_n_layers 2): router top-2 over eight experts, no
 *     renormalisation, no shared bias; h = LN2(h + moe(h))
 *   fused Wqkv [q|k|v], no GQA, non-causal, scale 1/sqrt(64) = 0.125
 *   NeoX rotary over all 64 dimensions, theta 1e4, on q and k only, no position table
 *   embeddings: word + token_type, then emb_ln; no final norm and no pooler tensor
 *   output: the mean of every real token, then cast to the f32 pooled buffer
 *
 * The expert tables are one concatenated tensor each: w1 [8 * 3072, 768] as [out, in], w2
 * [8 * 3072, 768] as [in, out] per expert, so w2 is sliced, then transposed (RAD_MAP_TRANSPOSE
 * with src_index and src_rows). Every expert's two tensors take distinct group slots.
 *
 * The block order and the omitted shared bias match llama.cpp's embedder for this model
 * (post-norm, no expert bias). The checkpoint's `experts.bias` tensors are not declared and stay
 * unconverted.
 */
#include <arch/rad_arch.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace nomicbert {

using namespace rad::arch;

/* Post-norm: attention on the raw stream, then LN1, then the feed-forward on LN1's output, then
 * LN2 over that output plus the feed-forward's. */

struct Layer {
    bool moe = false;
    rad_weight w_norm1 = 0, b_norm1 = 0, w_qkv = 0, b_qkv = 0, w_out = 0, b_out = 0;
    rad_weight w_norm2 = 0, b_norm2 = 0;
    rad_weight w_fc1 = 0, b_fc1 = 0, w_fc2 = 0, b_fc2 = 0;
    rad_weight w_router = 0;
    std::vector<rad_weight> w_w1, w_w2;

    rad_op op_qkv = 0, op_rope = 0, op_attn = 0, op_out = 0, op_norm1 = 0, op_norm2 = 0;
    rad_op op_fc1 = 0, op_fc2 = 0;
    rad_op op_router = 0, op_topk = 0, op_scatter = 0, op_egu = 0, op_gelu = 0, op_edn = 0;
    rad_op op_gather = 0, op_add = 0;
};

struct Model {
    Names nm{""};
    Geom  g{};
    int64_t n_expert = 0, n_expert_used = 0, n_type = 1;

    rad_buf b_x = 0, b_h = 0, b_o = 0, b_qkv = 0, b_att = 0, b_ff = 0;
    rad_buf b_egu = 0, b_eact = 0, b_edn = 0, b_emoe = 0;
    rad_buf b_logits = 0, b_ids = 0, b_ew = 0, b_sorted = 0, b_eoff = 0, b_ecnt = 0;
    rad_buf b_mean = 0, b_pooled = 0;

    rad_weight w_tok = 0, w_type = 0, w_emb_ln = 0, b_emb_ln = 0;
    rad_op op_embed = 0, op_type = 0, op_emb_ln = 0, op_mean = 0, op_cast = 0;

    std::vector<Layer> layers;
};

static Model g_model[MAX_RANKS];

/* The layer schedule, read from moe_every_n_layers rather than derived: a wrong interval would
 * declare a MoE layer whose tensors the checkpoint does not hold. Layer l is routed when
 * (l + 1) % moe_every_n_layers == 0, as nomic's modeling_hf_nomic_bert builds it. */
static bool is_moe_layer(int64_t l, int64_t every) {
    return every > 0 && (l + 1) % every == 0;
}

static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->draft_depth = 0;
    const char* mt = rad_meta_gets(meta, "model_type", nullptr);
    if (mt && *mt && std::strcmp(mt, "nomic_bert") && std::strcmp(mt, "nomicbert")) {
        fprintf(stderr, "radiance: nomicbert: model_type '%s' is not nomic_bert\n", mt);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;
    /* A 475M-parameter encoder fits one card; tensor parallelism would only add an all-reduce per
     * layer. */
    if (ctx->world_size > 1) {
        fprintf(stderr, "radiance: nomicbert runs on one card (--tp 1); the model is 475M "
                        "parameters\n");
        return RAD_E_UNSUPPORTED;
    }

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));
    Geom&  g  = m.g;
    Names& nm = m.nm;
    const int64_t T = g.max_tok, E = g.n_embd;

    /* The variant flags this plugin does not serve, refused by name. */
    if (rad_meta_geti(meta, "prenorm", 0) != 0) {
        fprintf(stderr, "radiance: nomicbert: the container sets prenorm; this plugin declares "
                        "the post-norm block\n");
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_geti(meta, "causal", 0) != 0) {
        fprintf(stderr, "radiance: nomicbert: the container sets causal; this encoder attends "
                        "over the whole input\n");
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_geti(meta, "rotary_emb_interleaved", 0) != 0 ||
        rad_meta_getf(meta, "rotary_emb_fraction", 1.0) != 1.0 ||
        rad_meta_gets(meta, "rotary_emb_scale_base", nullptr) ||
        rad_meta_gets(meta, "rotary_scaling_factor", nullptr)) {
        fprintf(stderr, "radiance: nomicbert: this plugin serves NeoX rotary over the whole head, "
                        "theta 1e4, with no scale base and no scaling factor\n");
        return RAD_E_UNSUPPORTED;
    }
    const char* act = rad_meta_gets(meta, "activation_function", nullptr);
    if (act && *act && std::strcmp(act, "gelu")) {
        fprintf(stderr, "radiance: nomicbert: activation_function '%s'; this plugin serves exact "
                        "gelu\n", act);
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_geti(meta, "moe_normalize_expert_weights", 0) != 0) {
        fprintf(stderr, "radiance: nomicbert: moe_normalize_expert_weights is set; the router's "
                        "top-k weights are used without renormalisation\n");
        return RAD_E_UNSUPPORTED;
    }
    if (rad_meta_geti(meta, "num_shared_experts", 0) != 0) {
        fprintf(stderr, "radiance: nomicbert: num_shared_experts is set; this checkpoint has no "
                        "shared expert\n");
        return RAD_E_UNSUPPORTED;
    }

    m.n_expert = meta->n_expert;
    m.n_expert_used = meta->n_expert_used;
    m.n_type = rad_meta_geti(meta, "type_vocab_size", 1);
    const int64_t every = rad_meta_geti(meta, "moe_every_n_layers", 0);
    const bool any_moe = m.n_expert > 0;
    if ((g.n_layer <= 0) || (E <= 0) || (g.n_head <= 0) || (g.head_dim <= 0) ||
        (g.n_ff <= 0) || (g.n_vocab <= 0) || (g.n_head_kv != g.n_head) ||
        (g.head_dim * g.n_head != E)) {
        fprintf(stderr, "radiance: nomicbert: the container does not describe the model "
                        "(layers %lld, hidden %lld, heads %lld of %lld, ffn %lld, vocab %lld)\n",
                (long long)g.n_layer, (long long)E, (long long)g.n_head,
                (long long)g.head_dim, (long long)g.n_ff, (long long)g.n_vocab);
        return RAD_E_INVAL;
    }
    if (any_moe && (every <= 0 || m.n_expert_used <= 0 || m.n_expert_used > m.n_expert)) {
        fprintf(stderr, "radiance: nomicbert: %lld experts with top_k %lld and "
                        "moe_every_n_layers %lld does not describe a routed layer\n",
                (long long)m.n_expert, (long long)m.n_expert_used, (long long)every);
        return RAD_E_INVAL;
    }
    if (m.n_type != 1) {
        fprintf(stderr, "radiance: nomicbert: type_vocab_size %lld; this checkpoint carries one "
                        "token-type row\n", (long long)m.n_type);
        return RAD_E_UNSUPPORTED;
    }
    const int64_t K = m.n_expert_used;
    const int64_t P = T * K;

    /* ---- activations. */
    m.b_x    = decl_b(b, nm.f("x"),        RAD_BF16, {T, E});
    m.b_h    = decl_b(b, nm.f("h"),        RAD_BF16, {T, E});
    m.b_o    = decl_b(b, nm.f("o"),        RAD_BF16, {T, E});
    m.b_qkv  = decl_b(b, nm.f("qkv"),      RAD_BF16, {T, 3 * E});
    m.b_att  = decl_b(b, nm.f("attn"),     RAD_BF16, {T, E});
    m.b_ff   = decl_b(b, nm.f("ff"),       RAD_BF16, {T, g.n_ff});
    if (any_moe) {
        m.b_egu    = decl_b(b, nm.f("moe_egu"),    RAD_BF16, {P, g.n_ff});
        m.b_eact   = decl_b(b, nm.f("moe_eact"),   RAD_BF16, {P, g.n_ff});
        m.b_edn    = decl_b(b, nm.f("moe_edn"),    RAD_BF16, {P, E});
        m.b_emoe   = decl_b(b, nm.f("moe_out"),    RAD_BF16, {T, E});
        m.b_logits = decl_b(b, nm.f("moe_logits"), RAD_BF16, {T, m.n_expert});
        m.b_ids    = decl_b(b, nm.f("moe_ids"),    RAD_I32,  {T, K});
        m.b_ew     = decl_b(b, nm.f("moe_w"),      RAD_BF16, {T, K});
        m.b_sorted = decl_b(b, nm.f("moe_sorted"), RAD_I32,  {P});
        m.b_eoff   = decl_b(b, nm.f("moe_eoff"),   RAD_I32,  {m.n_expert + 1});
        /* PERSIST: rad_route_report reads the histogram, not a declared op, so liveness would
         * otherwise treat the scatter's write as dead. */
        m.b_ecnt   = decl_b(b, nm.f("moe_ecnt"),   RAD_I32,  {m.n_expert}, RAD_BUF_PERSIST);
    }
    m.b_mean   = decl_b(b, nm.f("mean"),   RAD_BF16, {g.max_seqs, E});
    m.b_pooled = decl_b(b, nm.f("pooled"), RAD_F32,  {g.max_seqs, E}, RAD_BUF_PERSIST);
    RAD_ARCH_TRY(rad_declare_pooled(b, m.b_pooled));

    /* ---- the model-level weights. Norms and biases stay f32 where the kernels read a wider
     * gain; the embedding, the token-type table and the linears are bf16. table_add_bf16 reads a
     * bf16 table, and the CPU rows (libref, libavx) read any dtype the tensor carries. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), "embeddings.word_embeddings.weight"));
    RAD_ARCH_TRY(map_copy(b, nm.f("token_type.weight"),
                          "embeddings.token_type_embeddings.weight"));
    RAD_ARCH_TRY(map_copy(b, nm.f("emb_ln.weight"), "emb_ln.weight"));
    RAD_ARCH_TRY(map_copy(b, nm.f("emb_ln.bias"),   "emb_ln.bias"));
    m.w_tok    = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, E},
                        RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    m.w_type   = decl_w(b, nm.f("token_type.weight"), RAD_BF16, {1, E},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_emb_ln = decl_w(b, nm.f("emb_ln.weight"), RAD_F32, {E},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.b_emb_ln = decl_w(b, nm.f("emb_ln.bias"), RAD_F32, {E},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    auto layernorm_op = [&](rad_weight w, rad_weight bias) {
        return RAD_OP(b, "layernorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                 RAD_F64("eps", g.eps), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w, bias));
    };
    auto gemm_bias_op = [&](int64_t N, int64_t Width, rad_weight w, rad_weight bias,
                            const char* act_name) {
        if (act_name)
            return RAD_OP(b, "gemm_nt_bias",
                          RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", N),
                                     RAD_INT("K", Width), RAD_STR("dtype", "bf16"),
                                     RAD_STR("act", act_name)),
                          RAD_WEIGHTS(w, bias));
        return RAD_OP(b, "gemm_nt_bias",
                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", N),
                                 RAD_INT("K", Width), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w, bias));
    };
    auto gemm_op = [&](int64_t N, int64_t Width, rad_weight w) {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", N),
                                 RAD_INT("K", Width), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto table_add_op = [&](int64_t rows, rad_weight w) {
        return RAD_OP(b, "table_add",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", E), RAD_INT("L", 1),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto add_op = [&](rad_buf a, rad_buf c, rad_buf y) {
        return rw(b, RAD_OP(b, "add",
                            RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  {a, c}, {y});
    };
    auto moe_gemm_op = [&](int64_t N, int64_t Width, const char* order,
                           const std::vector<rad_weight>& tab, rad_buf a, rad_buf y) {
        rad_op h = rad_decl_op(b, "moe_gemm",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", N),
                                          RAD_INT("K", Width), RAD_INT("n_expert", m.n_expert),
                                          RAD_INT("top_k", K), RAD_STR("a_order", order),
                                          RAD_STR("dtype", "bf16")),
                               tab.data(), (int)tab.size());
        return rw(b, h, {a, m.b_sorted, m.b_eoff}, {y});
    };

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", E),
                                         RAD_INT("n_vocab", g.n_vocab),
                                         RAD_STR("dtype", "bf16"), RAD_INT("vocab_offset", 0),
                                         RAD_F64("mult", 1.0)),
                              RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_o});
    m.op_type = rw(b, table_add_op(T, m.w_type), {m.b_o}, {m.b_o});
    m.op_emb_ln = rw(b, layernorm_op(m.w_emb_ln, m.b_emb_ln), {m.b_o}, {m.b_x});

    m.layers.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& y = m.layers[(size_t)l];
        y.moe = any_moe && is_moe_layer(l, every);
        const long long ll = (long long)l;
        const RadWeightGroup gl = grp_layer((int)l);
        auto src = [&](const char* s) { return nm.ckpt("encoder.layers.%lld.%s", ll, s); };

        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_norm.weight", ll), src("norm1.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_norm.bias", ll),   src("norm1.bias")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_qkv.weight", ll), src("attn.Wqkv.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_qkv.bias", ll),   src("attn.Wqkv.bias")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_out.weight", ll),
                              src("attn.out_proj.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.attn_out.bias", ll),
                              src("attn.out_proj.bias")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_norm.weight", ll), src("norm2.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_norm.bias", ll),   src("norm2.bias")));

        const int A = RAD_ACCESS_PER_TOKEN, N = RAD_SHARD_NONE;
        y.w_norm1 = decl_w(b, nm.f("blk.%lld.attn_norm.weight", ll), RAD_F32, {E}, A, N, gl);
        y.b_norm1 = decl_w(b, nm.f("blk.%lld.attn_norm.bias", ll),   RAD_F32, {E}, A, N, gl);
        y.w_qkv   = decl_w(b, nm.f("blk.%lld.attn_qkv.weight", ll), RAD_BF16, {3 * E, E}, A, N, gl);
        y.b_qkv   = decl_w(b, nm.f("blk.%lld.attn_qkv.bias", ll),   RAD_F32, {3 * E}, A, N, gl);
        y.w_out   = decl_w(b, nm.f("blk.%lld.attn_out.weight", ll), RAD_BF16, {E, E}, A, N, gl);
        y.b_out   = decl_w(b, nm.f("blk.%lld.attn_out.bias", ll),   RAD_F32, {E}, A, N, gl);
        y.w_norm2 = decl_w(b, nm.f("blk.%lld.ffn_norm.weight", ll), RAD_F32, {E}, A, N, gl);
        y.b_norm2 = decl_w(b, nm.f("blk.%lld.ffn_norm.bias", ll),   RAD_F32, {E}, A, N, gl);

        if (!y.moe) {
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_fc1.weight", ll), src("mlp.fc1.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_fc1.bias", ll),   src("mlp.fc1.bias")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_fc2.weight", ll), src("mlp.fc2.weight")));
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_fc2.bias", ll),   src("mlp.fc2.bias")));
            y.w_fc1 = decl_w(b, nm.f("blk.%lld.ffn_fc1.weight", ll), RAD_BF16, {g.n_ff, E},
                             A, N, gl);
            y.b_fc1 = decl_w(b, nm.f("blk.%lld.ffn_fc1.bias", ll),   RAD_F32, {g.n_ff}, A, N, gl);
            y.w_fc2 = decl_w(b, nm.f("blk.%lld.ffn_fc2.weight", ll), RAD_BF16, {E, g.n_ff},
                             A, N, gl);
            y.b_fc2 = decl_w(b, nm.f("blk.%lld.ffn_fc2.bias", ll),   RAD_F32, {E}, A, N, gl);
        } else {
            RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ffn_router.weight", ll),
                                  src("mlp.router.layer.weight")));
            y.w_router   = decl_w(b, nm.f("blk.%lld.ffn_router.weight", ll), RAD_BF16,
                                  {m.n_expert, E}, A, N, gl);
            /* mlp.experts.bias is not mapped (no expert bias, see the header). */
            /* Each projection is a consecutive run of expert handles: all w1 in expert order,
             * then all w2. A grouped GEMM takes the run's first handle and its length. The name
             * maps come first: declare reads a weight's planned encoding when the weight is
             * declared, and a slice mapped afterwards would be read as plain bf16. */
            for (int64_t e = 0; e < m.n_expert; ++e) {
                const long long ee = (long long)e;
                RAD_ARCH_TRY(map_slice(b, nm.f("blk.%lld.ffn_exp_w1.%lld.weight", ll, ee),
                                       src("mlp.experts.mlp.w1"), (int)e, g.n_ff));
                RAD_ARCH_TRY(map_transpose_slice(b, nm.f("blk.%lld.ffn_exp_w2.%lld.weight", ll, ee),
                                                 src("mlp.experts.mlp.w2"), (int)e, g.n_ff));
            }
            y.w_w1.resize((size_t)m.n_expert);
            y.w_w2.resize((size_t)m.n_expert);
            for (int64_t e = 0; e < m.n_expert; ++e) {
                const long long ee = (long long)e;
                y.w_w1[(size_t)e] = decl_w(b, nm.f("blk.%lld.ffn_exp_w1.%lld.weight", ll, ee),
                                           RAD_BF16, {g.n_ff, E}, A, N,
                                           grp_expert_slot((int)l, (int)e, 0));
                if (!y.w_w1[(size_t)e]) return RAD_E_INVAL;
            }
            for (int64_t e = 0; e < m.n_expert; ++e) {
                const long long ee = (long long)e;
                y.w_w2[(size_t)e] = decl_w(b, nm.f("blk.%lld.ffn_exp_w2.%lld.weight", ll, ee),
                                           RAD_BF16, {E, g.n_ff}, A, N,
                                           grp_expert_slot((int)l, (int)e, 1));
                if (!y.w_w2[(size_t)e]) return RAD_E_INVAL;
            }
        }

        y.op_qkv = rw(b, gemm_bias_op(3 * E, E, y.w_qkv, y.b_qkv, nullptr),
                      {m.b_x}, {m.b_qkv});
        y.op_rope = rw(b, RAD_OP(b, "rope",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", g.head_dim),
                                            RAD_INT("n_head", g.n_head),
                                            RAD_INT("n_head_kv", g.n_head_kv),
                                            RAD_F64("theta", meta->rope_theta), RAD_F64("scale", 1.0),
                                            RAD_STR("mode", "neox")),
                                 RAD_NOWEIGHTS),
                       {m.b_qkv}, {m.b_qkv});
        y.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", g.head_dim),
                                            RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                            RAD_STR("dtype", "bf16"), RAD_F64("scale", 0.125),
                                            RAD_INT("max_seqlen", T)),
                                 RAD_NOWEIGHTS),
                       {m.b_qkv}, {m.b_att});
        y.op_out = rw(b, gemm_bias_op(E, E, y.w_out, y.b_out, nullptr),
                      {m.b_att, m.b_x}, {m.b_x});
        y.op_norm1 = rw(b, layernorm_op(y.w_norm1, y.b_norm1), {m.b_x}, {m.b_h});

        if (!y.moe) {
            y.op_fc1 = rw(b, gemm_bias_op(g.n_ff, E, y.w_fc1, y.b_fc1, "gelu"),
                          {m.b_h}, {m.b_ff});
            y.op_fc2 = rw(b, gemm_bias_op(E, g.n_ff, y.w_fc2, y.b_fc2, nullptr),
                          {m.b_ff, m.b_h}, {m.b_o});
        } else {
            y.op_router = rw(b, gemm_op(m.n_expert, E, y.w_router), {m.b_h}, {m.b_logits});
            y.op_topk = rw(b, RAD_OP(b, "router_topk",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T),
                                                RAD_INT("n_expert", m.n_expert),
                                                RAD_INT("top_k", K), RAD_INT("norm", 0),
                                                RAD_STR("dtype", "bf16")),
                                     RAD_NOWEIGHTS),
                           {m.b_logits}, {m.b_ids, m.b_ew});
            y.op_scatter = rw(b, RAD_OP(b, "moe_scatter",
                                        RAD_PARAMS(RAD_RANGE("M", 1, T),
                                                   RAD_INT("n_expert", m.n_expert),
                                                   RAD_INT("top_k", K),
                                                   RAD_INT("expert_base", 0)),
                                        RAD_NOWEIGHTS),
                              {m.b_ids}, {m.b_sorted, m.b_eoff, m.b_ecnt});
            y.op_egu = moe_gemm_op(g.n_ff, E, "token", y.w_w1, m.b_h, m.b_egu);
            y.op_gelu = rw(b, RAD_OP(b, "gelu",
                                     RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", g.n_ff),
                                                RAD_STR("dtype", "bf16")),
                                     RAD_NOWEIGHTS),
                           {m.b_egu}, {m.b_eact});
            y.op_edn = moe_gemm_op(E, g.n_ff, "sorted", y.w_w2, m.b_eact, m.b_edn);
            y.op_gather = rw(b, RAD_OP(b, "moe_gather",
                                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                                  RAD_INT("top_k", K), RAD_STR("dtype", "bf16"),
                                                  RAD_STR("act", "none")),
                                       RAD_NOWEIGHTS),
                             {m.b_edn, m.b_ew, m.b_sorted}, {m.b_emoe});
            y.op_add = add_op(m.b_h, m.b_emoe, m.b_o);
        }

        y.op_norm2 = rw(b, layernorm_op(y.w_norm2, y.b_norm2), {m.b_o}, {m.b_x});

        if (!y.op_qkv || !y.op_rope || !y.op_attn || !y.op_out || !y.op_norm1 || !y.op_norm2)
            return RAD_E_INVAL;
        if (y.moe && (!y.op_router || !y.op_topk || !y.op_scatter || !y.op_egu || !y.op_gelu ||
                      !y.op_edn || !y.op_gather || !y.op_add))
            return RAD_E_INVAL;
        if (!y.moe && (!y.op_fc1 || !y.op_fc2)) return RAD_E_INVAL;
    }

    m.op_mean = rw(b, RAD_OP(b, "seg_mean",
                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   {m.b_x}, {m.b_mean});
    m.op_cast = rw(b, RAD_OP(b, "cast",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("n", E),
                                        RAD_STR("from", "bf16"), RAD_STR("to", "f32")),
                             RAD_NOWEIGHTS),
                   {m.b_mean}, {m.b_pooled});

    int64_t n_routed = 0;
    for (const Layer& y : m.layers) n_routed += y.moe ? 1 : 0;
    rad_note(b, "embedding model: %lld layers (%lld routed, every %lld), hidden %lld, ffn %lld, "
                "%lld heads of %lld, %lld experts top %lld; post-norm, no expert bias; "
                "mean-pooled to %lld",
             (long long)g.n_layer, (long long)n_routed, (long long)every, (long long)E,
             (long long)g.n_ff, (long long)g.n_head, (long long)g.head_dim,
             (long long)m.n_expert, (long long)m.n_expert_used, (long long)E);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_rank(c)];
    const Geom& g = m.g;
    const int64_t T = batch->n_tok, S = batch->n_seq, E = g.n_embd;
    if (T <= 0 || S <= 0) return;
    const int64_t K = m.n_expert_used;
    const RadOperand cu  = praw(batch->cu_seqlens, RAD_I32, S + 1);
    const RadOperand pos = praw(batch->positions, RAD_I32, T);

    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok),
              brows(m.b_o, T));
    RAD_ISSUE(c, m.op_type, brows(m.b_o, T), RAD_W(m.w_type), brows(m.b_o, T));
    RAD_ISSUE(c, m.op_emb_ln, brows(m.b_o, T), RAD_W(m.w_emb_ln), RAD_W(m.b_emb_ln),
              brows(m.b_x, T));

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const Layer& y = m.layers[(size_t)l];

        RAD_ISSUE_N(c, y.op_qkv, T, brows(m.b_x, T), RAD_W(y.w_qkv), RAD_W(y.b_qkv),
                    RAD_NONE, brows(m.b_qkv, T));
        RAD_ISSUE(c, y.op_rope, brows(m.b_qkv, T), pos);
        RAD_ISSUE(c, y.op_attn, bcol(m.b_qkv, 0, E, T), bcol(m.b_qkv, E, E, T),
                  bcol(m.b_qkv, 2 * E, E, T), cu, brows(m.b_att, T));
        RAD_ISSUE_N(c, y.op_out, T, brows(m.b_att, T), RAD_W(y.w_out), RAD_W(y.b_out),
                    brows(m.b_x, T), brows(m.b_x, T));
        RAD_ISSUE(c, y.op_norm1, brows(m.b_x, T), RAD_W(y.w_norm1), RAD_W(y.b_norm1),
                  brows(m.b_h, T));

        if (!y.moe) {
            RAD_ISSUE_N(c, y.op_fc1, T, brows(m.b_h, T), RAD_W(y.w_fc1), RAD_W(y.b_fc1),
                        RAD_NONE, brows(m.b_ff, T));
            RAD_ISSUE_N(c, y.op_fc2, T, brows(m.b_ff, T), RAD_W(y.w_fc2),
                        RAD_W(y.b_fc2), brows(m.b_h, T), brows(m.b_o, T));
        } else {
            RAD_ISSUE(c, y.op_router, brows(m.b_h, T), RAD_W(y.w_router), brows(m.b_logits, T));
            RAD_ISSUE(c, y.op_topk, brows(m.b_logits, T), brows(m.b_ids, T), brows(m.b_ew, T));
            RAD_ISSUE(c, y.op_scatter, brows(m.b_ids, T), brows(m.b_sorted, T * K),
                      brows(m.b_eoff, m.n_expert + 1), brows(m.b_ecnt, m.n_expert));
            RAD_ISSUE_N(c, y.op_egu, T, brows(m.b_h, T), RAD_WTAB(y.w_w1[0], m.n_expert),
                        brows(m.b_sorted, T * K), brows(m.b_eoff, m.n_expert + 1),
                        brows(m.b_egu, T * K));
            RAD_ISSUE_N(c, y.op_gelu, T * K, brows(m.b_egu, T * K), brows(m.b_eact, T * K));
            RAD_ISSUE_N(c, y.op_edn, T, brows(m.b_eact, T * K), RAD_WTAB(y.w_w2[0], m.n_expert),
                        brows(m.b_sorted, T * K), brows(m.b_eoff, m.n_expert + 1),
                        brows(m.b_edn, T * K));
            RAD_ISSUE_N(c, y.op_gather, T, brows(m.b_edn, T * K), brows(m.b_ew, T),
                        brows(m.b_sorted, T * K), RAD_NONE, RAD_NONE, brows(m.b_emoe, T));
            RAD_ISSUE(c, y.op_add, brows(m.b_h, T), brows(m.b_emoe, T), brows(m.b_o, T));
        }
        RAD_ISSUE(c, y.op_norm2, brows(m.b_o, T), RAD_W(y.w_norm2), RAD_W(y.b_norm2),
                  brows(m.b_x, T));
    }

    RAD_ISSUE(c, m.op_mean, brows(m.b_x, T), cu, brows(m.b_mean, S));
    RAD_ISSUE_N(c, m.op_cast, S, brows(m.b_mean, S), brows(m.b_pooled, S));
}

}  /* namespace nomicbert */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(nomicbert, "nomicbert", "", "0.1.0",
                "Nomic BERT (nomic-embed-text-v2-moe): a bidirectional post-norm encoder with "
                "NeoX rotary attention, dense and routed GELU feed-forwards, mean-pooled to one "
                "vector an input; bf16")
RAD_ARCH_PROBE(nomicbert)
#endif
