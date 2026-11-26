/* embgemma2 -- EmbeddingGemma 2, a bidirectional Gemma encoder that pools a whole input into one
 * vector. bf16 weights and bf16 activations.
 *
 * An embedding model: it declares a pooled output (rad_declare_pooled) and no logits, so the
 * engine runs it with no sampler and no KV cache, carries every input whole in one step, and
 * returns each sequence's pooled row. Attention is bidirectional on every layer, so nothing is
 * paged: a token attends to later tokens, and there is nothing to carry between steps.
 *
 * ============================== THE SHAPE ==============================
 *
 *   24 layers, hidden 512, ffn 2048, vocab 262144, embedding scale sqrt(512) rounded to bf16
 *   layer_types: five `sliding_attention` then one `full_attention`, four times over
 *     sliding: 4 heads / 2 kv heads / head_dim 256, rope theta 1e4, |i - j| <= 512
 *     full:    4 heads / 1 kv head  / head_dim 512, rope theta 1e6, the whole input
 *   attention scale 1.0 (the q/k norms carry the temperature); q, k and v are RMS-normed per head,
 *   v with no gain
 *   Gemma's sandwich norms: a norm before and after attention and the feed-forward, all plain
 *   RMSNorm gains (x_hat * w, not 1 + w)
 *   a gated gelu-tanh feed-forward
 *   per-layer embeddings, projection-only: the token embedding projected to 24 x 512, normed, and
 *   each layer's slice gating that layer's residual through a 512 -> 512 -> 512 bottleneck
 *   a learned scalar per layer, multiplying the residual stream at the layer's end
 *   final norm, then the mean over every token of the input, then 512 -> 768
 *
 * The reference is transformers' modeling_embedding_gemma2 (model_type embedding_gemma2, which
 * rad-convert spells `embeddinggemma2`). Its mean pooling and normalisation are
 * sentence-transformers' modules on top; normalisation is the server's (it truncates to
 * `dimensions` first), so the pooled row here is the unnormalised mean.
 *
 * Everything the checkpoint carries is declared: the text model, the vision tower
 * (rad_block_gemma4_vision.h) and the audio encoder (rad_block_gemma4_audio.h).
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_gemma4_audio.h>
#include <arch/rad_block_gemma4_vision.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace embgemma2 {

using namespace rad::arch;

/* What a layer's attention is: the two kinds differ in head width, KV head count, rotary base and
 * window, and nothing else. */
struct Kind {
    int64_t head_dim  = 0;
    int64_t n_head    = 0;
    int64_t n_head_kv = 0;
    double  theta     = 0.0;
    int64_t window    = 0;      /* attn_dense's `window`: |i - j| < window, 0 the whole input */
    int64_t qkv() const { return (n_head + 2 * n_head_kv) * head_dim; }
    int64_t q()   const { return n_head * head_dim; }
};

struct Layer {
    int     full = 0;
    rad_weight w_in_norm = 0, w_qkv = 0, w_q_norm = 0, w_k_norm = 0, w_o = 0, w_post_attn = 0;
    rad_weight w_pre_ff = 0, w_gate_up = 0, w_down = 0, w_post_ff = 0;
    rad_weight w_ple_gate = 0, w_ple_proj = 0, w_post_ple = 0, w_scalar = 0;
    rad_op op_in_norm = 0, op_qkv = 0, op_qk = 0, op_attn = 0, op_o = 0, op_post_attn = 0;
    rad_op op_add_attn = 0, op_pre_ff = 0, op_gate_up = 0, op_act = 0, op_down = 0;
    rad_op op_post_ff = 0, op_add_ff = 0, op_ple_gate = 0, op_ple_act = 0, op_ple_proj = 0;
    rad_op op_post_ple = 0, op_add_ple = 0, op_scale = 0;
};

struct Model {
    Names nm{""};
    Geom  g{};
    Kind  ks{}, kf{};           /* sliding, full */
    int64_t n_ple = 0;          /* hidden_size_per_layer_input */
    int64_t n_out = 0;          /* embedding_dim */

    rad_buf b_x = 0, b_h = 0, b_o = 0, b_qkv_s = 0, b_qkv_f = 0, b_att_s = 0, b_att_f = 0;
    rad_buf b_gu = 0, b_ff = 0, b_p = 0, b_ple = 0, b_mean = 0, b_proj = 0, b_pooled = 0;

    rad_weight w_tok = 0, w_ple_proj = 0, w_ple_norm = 0, w_norm = 0, w_out = 0;
    rad_op op_embed = 0, op_ple = 0, op_ple_norm = 0, op_norm = 0, op_mean = 0, op_out = 0;
    rad_op op_cast = 0;
    rad_op op_mm = 0;           /* the media rows over their placeholders' embeddings */

    std::vector<Layer> layers;
    Gemma4Vision vis;
    Gemma4Audio aud;
};

static Model g_model[MAX_SLOTS];

/* The layer schedule, read rather than derived: layer_types states which layers attend to the
 * whole input, and an assumed pattern would silently be wrong for a sibling checkpoint that
 * differs. */
static int read_schedule(const RadModelMeta* meta, int64_t n_layer, std::vector<int>* full) {
    const char* s = rad_meta_gets(meta, "layer_types", nullptr);
    if (!s || !*s) {
        fprintf(stderr, "radiance: embgemma2: the container has no layer_types; it says which "
                        "layers are sliding and which see the whole input, and is not guessed\n");
        return RAD_E_INVAL;
    }
    full->clear();
    for (const char* p = s; *p; ) {
        while (*p == ' ') ++p;
        if (!*p) break;
        const char* t = p;
        while (*p && *p != ' ') ++p;
        const size_t len = (size_t)(p - t);
        if (len == 14 && !std::strncmp(t, "full_attention", 14))         full->push_back(1);
        else if (len == 17 && !std::strncmp(t, "sliding_attention", 17)) full->push_back(0);
        else {
            fprintf(stderr, "radiance: embgemma2: layer_types names '%.*s'; this plugin knows "
                            "full_attention and sliding_attention\n", (int)len, t);
            return RAD_E_UNSUPPORTED;
        }
    }
    if ((int64_t)full->size() != n_layer) {
        fprintf(stderr, "radiance: embgemma2: layer_types names %zu layers and the model has "
                        "%lld\n", full->size(), (long long)n_layer);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    /* A 270M encoder fits easily on one card; tensor parallelism would only add an all-reduce per
     * layer, so it is not offered. */
    if (ctx->world_size > 1) {
        fprintf(stderr, "radiance: embgemma2 runs on one card (--tp 1); the text model is 270M "
                        "parameters\n");
        return RAD_E_UNSUPPORTED;
    }

    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));
    Geom&  g  = m.g;
    Names& nm = m.nm;
    const int64_t T = g.max_tok, E = g.n_embd;

    std::vector<int> full;
    RAD_ARCH_TRY(read_schedule(meta, g.n_layer, &full));

    /* The two attention kinds. The sliding one is the config's top-level geometry; the full one
     * overrides head width and KV heads per layer (per_layer_config), and every full layer must
     * state the same override (checked below). */
    m.ks.head_dim  = g.head_dim;
    m.ks.n_head    = g.n_head;
    m.ks.n_head_kv = g.n_head_kv;
    m.ks.theta     = rad_meta_getf(meta, "rope_parameters.sliding_attention.rope_theta", 0.0);
    /* The reference's window is a radius (|i - j| <= sliding_window) and attn_dense's is a strict
     * bound, so one more. */
    const int64_t radius = meta_int2(meta, "", "sliding_window", 0);
    m.ks.window = radius + 1;
    m.kf.n_head = g.n_head;
    m.kf.theta  = rad_meta_getf(meta, "rope_parameters.full_attention.rope_theta", 0.0);
    m.kf.window = 0;
    for (int64_t l = 0; l < g.n_layer; ++l) {
        if (!full[(size_t)l]) continue;
        char k1[96], k2[96];
        std::snprintf(k1, sizeof k1, "per_layer_config.%02lld.head_dim", (long long)l);
        std::snprintf(k2, sizeof k2, "per_layer_config.%02lld.num_key_value_heads", (long long)l);
        const int64_t hd = meta_int2(meta, "", k1, 0), kv = meta_int2(meta, "", k2, 0);
        if (!m.kf.head_dim) { m.kf.head_dim = hd; m.kf.n_head_kv = kv; }
        if (hd <= 0 || kv <= 0 || hd != m.kf.head_dim || kv != m.kf.n_head_kv) {
            fprintf(stderr, "radiance: embgemma2: full-attention layer %lld states head_dim %lld "
                            "and %lld KV heads; this plugin serves one full geometry\n",
                    (long long)l, (long long)hd, (long long)kv);
            return RAD_E_INVAL;
        }
    }
    m.n_ple = meta_int2(meta, "", "hidden_size_per_layer_input", 0);
    m.n_out = meta_int2(meta, "", "embedding_dim", 0);
    if (!(m.ks.theta > 0) || !(m.kf.theta > 0) || radius <= 0 || m.kf.head_dim <= 0 ||
        m.n_ple <= 0 || m.n_out <= 0 || m.ks.n_head % m.ks.n_head_kv || m.kf.n_head % m.kf.n_head_kv) {
        fprintf(stderr, "radiance: embgemma2: the container does not describe the model (rope "
                        "thetas %g / %g, sliding_window %lld, full head_dim %lld, "
                        "hidden_size_per_layer_input %lld, embedding_dim %lld)\n",
                m.ks.theta, m.kf.theta, (long long)radius, (long long)m.kf.head_dim,
                (long long)m.n_ple, (long long)m.n_out);
        return RAD_E_INVAL;
    }
    const int64_t P = m.n_ple * g.n_layer;
    /* One input is at most one step (Request::embed), so the longest attention segment is the
     * step's token budget. */
    const int64_t max_seg = T;

    /* ---- activations. ple holds every layer's per-layer input for the whole step, read as one
     * column slice per layer. */
    m.b_x     = decl_b(b, nm.f("x"),       RAD_BF16, {T, E});
    m.b_h     = decl_b(b, nm.f("h"),       RAD_BF16, {T, E});
    m.b_o     = decl_b(b, nm.f("o"),       RAD_BF16, {T, E});
    m.b_qkv_s = decl_b(b, nm.f("qkv_s"),   RAD_BF16, {T, m.ks.qkv()});
    m.b_qkv_f = decl_b(b, nm.f("qkv_f"),   RAD_BF16, {T, m.kf.qkv()});
    m.b_att_s = decl_b(b, nm.f("att_s"),   RAD_BF16, {T, m.ks.q()});
    m.b_att_f = decl_b(b, nm.f("att_f"),   RAD_BF16, {T, m.kf.q()});
    m.b_gu    = decl_b(b, nm.f("gate_up"), RAD_BF16, {T, 2 * g.n_ff});
    m.b_ff    = decl_b(b, nm.f("ffn"),     RAD_BF16, {T, g.n_ff});
    m.b_p     = decl_b(b, nm.f("ple_p"),   RAD_BF16, {T, m.n_ple});
    m.b_ple   = decl_b(b, nm.f("ple"),     RAD_BF16, {T, P});
    m.b_mean  = decl_b(b, nm.f("mean"),    RAD_BF16, {g.max_seqs, E});
    m.b_proj  = decl_b(b, nm.f("proj"),    RAD_BF16, {g.max_seqs, m.n_out});
    m.b_pooled= decl_b(b, nm.f("pooled"),  RAD_F32,  {g.max_seqs, m.n_out}, RAD_BUF_PERSIST);
    RAD_ARCH_TRY(rad_declare_pooled(b, m.b_pooled));

    /* ---- the model-level weights, all at the checkpoint's bf16, norm gains included: the
     * reference widens a gain to f32 at use, which is exact, and a wider stored copy could not be
     * read in place on the host backend. */
    const char* L = "language_model";
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("%s.embed_tokens.weight", L)));
    RAD_ARCH_TRY(map_copy(b, nm.f("ple_proj.weight"),
                          nm.ckpt("%s.ple.per_layer_model_projection.weight", L)));
    RAD_ARCH_TRY(map_copy(b, nm.f("ple_norm.weight"),
                          nm.ckpt("%s.ple.per_layer_projection_norm.weight", L)));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("%s.norm.weight", L)));
    RAD_ARCH_TRY(map_copy(b, nm.f("embd_proj.weight"),
                          nm.ckpt("%s.embedding_projection.weight", L)));
    m.w_tok      = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, E},
                          RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    m.w_ple_proj = decl_w(b, nm.f("ple_proj.weight"), RAD_BF16, {P, E},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_ple_norm = decl_w(b, nm.f("ple_norm.weight"), RAD_BF16, {m.n_ple},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_norm     = decl_w(b, nm.f("output_norm.weight"), RAD_BF16, {E},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_out      = decl_w(b, nm.f("embd_proj.weight"), RAD_BF16, {m.n_out, E},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    auto rmsnorm = [&](int64_t rows, int64_t n, double eps, rad_weight w) {
        return RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                 RAD_F64("wadd", 0.0), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto gemm = [&](int64_t rows, int64_t N, int64_t K, rad_weight w) {
        return RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(w));
    };
    auto elem = [&](const char* op, int64_t n, rad_weight w) {
        if (w)
            return RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n),
                                            RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w));
        return RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n),
                                        RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS);
    };
    auto act_mul = [&](int64_t n) {
        return RAD_OP(b, "act_mul", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n),
                                               RAD_STR("act", "gelu_tanh"),
                                               RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS);
    };

    /* The embedding, scaled by sqrt(hidden) rounded as the reference rounds it (a bf16 buffer):
     * sqrt(512) = 22.6274 multiplies as 22.625. */
    const double escale = rad_bf16_to_f32(rad_f32_to_bf16((float)std::sqrt((double)E)));
    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", E),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", 0), RAD_F64("mult", escale)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    /* A media part's rows overwrite its placeholders' embeddings before anything reads them, so an
     * image token's per-layer inputs come from its image row (declared only with a tower). */
    if ((Gemma4Vision::present(meta) && ctx->max_enc_patches > 0) ||
        (Gemma4Audio::present(meta) && ctx->max_audio_frames > 0))
        m.op_mm = rw(b, RAD_OP(b, "scatter_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     {m.b_x}, {m.b_x});
    /* The per-layer inputs: the embedding projected to n_layer x n_ple and RMS-normed per layer.
     * The reference scales the projection by hidden^-0.5 before the norm; the RMS norm of s*x with
     * eps equals the norm of x with eps / s^2, which is what is declared, avoiding the bf16
     * rounding of s*x. */
    m.op_ple = rw(b, gemm(T, P, E, m.w_ple_proj), {m.b_x}, {m.b_ple});
    m.op_ple_norm = rw(b, rmsnorm(T * g.n_layer, m.n_ple, (double)g.eps * (double)E, m.w_ple_norm),
                       {m.b_ple}, {m.b_ple});

    m.layers.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& y = m.layers[(size_t)l];
        y.full = full[(size_t)l];
        const Kind& k = y.full ? m.kf : m.ks;
        const rad_buf qkv = y.full ? m.b_qkv_f : m.b_qkv_s;
        const rad_buf att = y.full ? m.b_att_f : m.b_att_s;
        const long long ll = (long long)l;
        const RadWeightGroup gl = grp_layer((int)l);
        auto src = [&](const char* s) { return nm.ckpt("%s.layers.%lld.%s", L, ll, s); };

        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.in_norm", ll),   src("input_layernorm.weight")));
        RAD_ARCH_TRY(map_concat(b, nm.f("blk.%lld.qkv", ll),
                                {src("self_attn.q_proj.weight"), src("self_attn.k_proj.weight"),
                                 src("self_attn.v_proj.weight")}));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.q_norm", ll),    src("self_attn.q_norm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.k_norm", ll),    src("self_attn.k_norm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.o", ll),         src("self_attn.o_proj.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.post_attn", ll),
                              src("post_attention_layernorm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.pre_ff", ll),
                              src("pre_feedforward_layernorm.weight")));
        RAD_ARCH_TRY(map_concat(b, nm.f("blk.%lld.gate_up", ll),
                                {src("mlp.gate_proj.weight"), src("mlp.up_proj.weight")}));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.down", ll),      src("mlp.down_proj.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.post_ff", ll),
                              src("post_feedforward_layernorm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ple_gate", ll),
                              src("ple_block.per_layer_input_gate.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.ple_proj", ll),
                              src("ple_block.per_layer_projection.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.post_ple", ll),
                              src("ple_block.post_per_layer_input_norm.weight")));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%lld.scalar", ll),    src("layer_scalar")));

        const int A = RAD_ACCESS_PER_TOKEN, N = RAD_SHARD_NONE;
        y.w_in_norm   = decl_w(b, nm.f("blk.%lld.in_norm", ll),   RAD_BF16, {E}, A, N, gl);
        y.w_qkv       = decl_w(b, nm.f("blk.%lld.qkv", ll),       RAD_BF16, {k.qkv(), E}, A, N, gl);
        y.w_q_norm    = decl_w(b, nm.f("blk.%lld.q_norm", ll),    RAD_BF16, {k.head_dim}, A, N, gl);
        y.w_k_norm    = decl_w(b, nm.f("blk.%lld.k_norm", ll),    RAD_BF16, {k.head_dim}, A, N, gl);
        y.w_o         = decl_w(b, nm.f("blk.%lld.o", ll),         RAD_BF16, {E, k.q()}, A, N, gl);
        y.w_post_attn = decl_w(b, nm.f("blk.%lld.post_attn", ll), RAD_BF16, {E}, A, N, gl);
        y.w_pre_ff    = decl_w(b, nm.f("blk.%lld.pre_ff", ll),    RAD_BF16, {E}, A, N, gl);
        y.w_gate_up   = decl_w(b, nm.f("blk.%lld.gate_up", ll),   RAD_BF16, {2 * g.n_ff, E}, A, N, gl);
        y.w_down      = decl_w(b, nm.f("blk.%lld.down", ll),      RAD_BF16, {E, g.n_ff}, A, N, gl);
        y.w_post_ff   = decl_w(b, nm.f("blk.%lld.post_ff", ll),   RAD_BF16, {E}, A, N, gl);
        y.w_ple_gate  = decl_w(b, nm.f("blk.%lld.ple_gate", ll),  RAD_BF16, {m.n_ple, E}, A, N, gl);
        y.w_ple_proj  = decl_w(b, nm.f("blk.%lld.ple_proj", ll),  RAD_BF16, {E, m.n_ple}, A, N, gl);
        y.w_post_ple  = decl_w(b, nm.f("blk.%lld.post_ple", ll),  RAD_BF16, {E}, A, N, gl);
        /* bf16 and one element: the reference multiplies the bf16 residual by it in bf16. */
        y.w_scalar    = decl_w(b, nm.f("blk.%lld.scalar", ll),    RAD_BF16, {1}, A, N, gl);

        y.op_in_norm = rw(b, rmsnorm(T, E, g.eps, y.w_in_norm), {m.b_x}, {m.b_h});
        y.op_qkv     = rw(b, gemm(T, k.qkv(), E, y.w_qkv), {m.b_h}, {qkv});
        y.op_qk      = rw(b, RAD_OP(b, "qk_norm_rope",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", k.head_dim),
                                               RAD_INT("n_head", k.n_head),
                                               RAD_INT("n_head_kv", k.n_head_kv),
                                               RAD_F64("theta", k.theta), RAD_F64("eps", g.eps),
                                               RAD_STR("mode", "neox"), RAD_INT("v_norm", 1)),
                                    RAD_WEIGHTS(y.w_q_norm, y.w_k_norm)),
                          {qkv}, {qkv});
        y.op_attn    = rw(b, RAD_OP(b, "attn_dense",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", k.head_dim),
                                               RAD_INT("gqa", k.n_head / k.n_head_kv),
                                               RAD_INT("causal", 0), RAD_STR("dtype", "bf16"),
                                               RAD_F64("scale", 1.0),
                                               RAD_INT("max_seqlen", max_seg),
                                               RAD_INT("window", k.window)),
                                    RAD_NOWEIGHTS),
                          {qkv}, {att});
        y.op_o        = rw(b, gemm(T, E, k.q(), y.w_o), {att}, {m.b_o});
        y.op_post_attn= rw(b, rmsnorm(T, E, g.eps, y.w_post_attn), {m.b_o}, {m.b_o});
        y.op_add_attn = rw(b, elem("add", E, 0), {m.b_x, m.b_o}, {m.b_x});
        y.op_pre_ff   = rw(b, rmsnorm(T, E, g.eps, y.w_pre_ff), {m.b_x}, {m.b_h});
        y.op_gate_up  = rw(b, gemm(T, 2 * g.n_ff, E, y.w_gate_up), {m.b_h}, {m.b_gu});
        y.op_act      = rw(b, act_mul(g.n_ff), {m.b_gu}, {m.b_ff});
        y.op_down     = rw(b, gemm(T, E, g.n_ff, y.w_down), {m.b_ff}, {m.b_o});
        y.op_post_ff  = rw(b, rmsnorm(T, E, g.eps, y.w_post_ff), {m.b_o}, {m.b_o});
        y.op_add_ff   = rw(b, elem("add", E, 0), {m.b_x, m.b_o}, {m.b_x});
        y.op_ple_gate = rw(b, gemm(T, m.n_ple, E, y.w_ple_gate), {m.b_x}, {m.b_p});
        y.op_ple_act  = rw(b, act_mul(m.n_ple), {m.b_p, m.b_ple}, {m.b_p});
        y.op_ple_proj = rw(b, gemm(T, E, m.n_ple, y.w_ple_proj), {m.b_p}, {m.b_o});
        y.op_post_ple = rw(b, rmsnorm(T, E, g.eps, y.w_post_ple), {m.b_o}, {m.b_o});
        y.op_add_ple  = rw(b, elem("add", E, 0), {m.b_x, m.b_o}, {m.b_x});
        y.op_scale    = rw(b, RAD_OP(b, "mul_weight",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                                RAD_STR("dtype", "bf16"), RAD_INT("scalar", 1)),
                                     RAD_WEIGHTS(y.w_scalar)),
                           {m.b_x}, {m.b_x});
    }

    m.op_norm = rw(b, rmsnorm(T, E, g.eps, m.w_norm), {m.b_x}, {m.b_h});
    m.op_mean = rw(b, RAD_OP(b, "seg_mean",
                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", E),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   {m.b_h}, {m.b_mean});
    m.op_out  = rw(b, gemm(g.max_seqs, m.n_out, E, m.w_out), {m.b_mean}, {m.b_proj});
    m.op_cast = cast_op(b, g.max_seqs, m.n_out, "bf16", "f32", m.b_proj, m.b_pooled);

    RAD_ARCH_TRY(m.vis.declare(b, meta, ctx, nm, E, (int)g.n_layer, "", "embgemma2"));
    RAD_ARCH_TRY(m.aud.declare(b, meta, ctx, nm, E, (int)g.n_layer + (int)m.vis.L.size(), "",
                               "embgemma2"));

    int n_full = 0;
    for (int f : full) n_full += f;
    rad_note(b, "embedding model: %lld layers (%d whole-input, %lld sliding |i-j|<=%lld), hidden "
                "%lld, ffn %lld; sliding %lldx%lld (kv %lld) theta %g, full %lldx%lld (kv %lld) "
                "theta %g; per-layer inputs %lld; mean-pooled to %lld",
             (long long)g.n_layer, n_full, (long long)(g.n_layer - n_full), (long long)radius,
             (long long)E, (long long)g.n_ff, (long long)m.ks.n_head, (long long)m.ks.head_dim,
             (long long)m.ks.n_head_kv, m.ks.theta, (long long)m.kf.n_head,
             (long long)m.kf.head_dim, (long long)m.kf.n_head_kv, m.kf.theta,
             (long long)m.n_ple, (long long)m.n_out);
    return RAD_OK;
}

/* Before declare: the vision tower's pooling window, so a pass size the core picks is a whole
 * number of windows. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    if (Gemma4Vision::present(meta)) {
        const int64_t k = rad_meta_geti(meta, "vision_config.pooling_kernel_size", 0);
        if (k > 0) out->enc_merge = k * k;
    }
    if (Gemma4Audio::present(meta)) out->audio_frames = Gemma4Audio::frames_for(meta);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_slot(c)];
    const Geom& g = m.g;
    const int64_t T = batch->n_tok, S = batch->n_seq, E = g.n_embd;
    if (batch->enc) {
        if (batch->enc == RAD_MM_AUDIO) m.aud.step(c, batch);
        else                            m.vis.step(c, batch);
        return;
    }
    if (T <= 0 || S <= 0) return;
    const RadOperand cu  = praw(batch->cu_seqlens, RAD_I32, S + 1);
    const RadOperand pos = praw(batch->positions, RAD_I32, T);

    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_x, T));
    if (m.op_mm && batch->n_mm_rows > 0)
        RAD_ISSUE_N(c, m.op_mm, batch->n_mm_rows,
                    praw2(batch->mm_embd, RAD_BF16, batch->n_mm_rows, E),
                    praw(batch->mm_rows, RAD_I32, batch->n_mm_rows), brows(m.b_x, T));
    RAD_ISSUE(c, m.op_ple, brows(m.b_x, T), RAD_W(m.w_ple_proj), brows(m.b_ple, T));
    /* [T, n_layer * n_ple] read as [T * n_layer, n_ple]: one norm row per (token, layer). */
    RAD_ISSUE_N(c, m.op_ple_norm, T * g.n_layer, brows(m.b_ple, T), RAD_W(m.w_ple_norm),
                brows(m.b_ple, T));

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const Layer& y = m.layers[(size_t)l];
        const Kind& k = y.full ? m.kf : m.ks;
        const rad_buf qkv = y.full ? m.b_qkv_f : m.b_qkv_s;
        const rad_buf att = y.full ? m.b_att_f : m.b_att_s;
        const int64_t q = k.q(), kv = k.n_head_kv * k.head_dim;

        RAD_ISSUE(c, y.op_in_norm, brows(m.b_x, T), RAD_W(y.w_in_norm), brows(m.b_h, T));
        RAD_ISSUE(c, y.op_qkv, brows(m.b_h, T), RAD_W(y.w_qkv), brows(qkv, T));
        RAD_ISSUE(c, y.op_qk, brows(qkv, T), pos, RAD_W(y.w_q_norm), RAD_W(y.w_k_norm));
        RAD_ISSUE(c, y.op_attn, bcol(qkv, 0, q, T), bcol(qkv, q, kv, T), bcol(qkv, q + kv, kv, T),
                  cu, brows(att, T));
        RAD_ISSUE(c, y.op_o, brows(att, T), RAD_W(y.w_o), brows(m.b_o, T));
        RAD_ISSUE(c, y.op_post_attn, brows(m.b_o, T), RAD_W(y.w_post_attn), brows(m.b_o, T));
        RAD_ISSUE(c, y.op_add_attn, brows(m.b_x, T), brows(m.b_o, T), brows(m.b_x, T));
        RAD_ISSUE(c, y.op_pre_ff, brows(m.b_x, T), RAD_W(y.w_pre_ff), brows(m.b_h, T));
        RAD_ISSUE(c, y.op_gate_up, brows(m.b_h, T), RAD_W(y.w_gate_up), brows(m.b_gu, T));
        RAD_ISSUE(c, y.op_act, bcol(m.b_gu, 0, g.n_ff, T), bcol(m.b_gu, g.n_ff, g.n_ff, T),
                  brows(m.b_ff, T));
        RAD_ISSUE(c, y.op_down, brows(m.b_ff, T), RAD_W(y.w_down), brows(m.b_o, T));
        RAD_ISSUE(c, y.op_post_ff, brows(m.b_o, T), RAD_W(y.w_post_ff), brows(m.b_o, T));
        RAD_ISSUE(c, y.op_add_ff, brows(m.b_x, T), brows(m.b_o, T), brows(m.b_x, T));
        RAD_ISSUE(c, y.op_ple_gate, brows(m.b_x, T), RAD_W(y.w_ple_gate), brows(m.b_p, T));
        RAD_ISSUE(c, y.op_ple_act, brows(m.b_p, T), bcol(m.b_ple, l * m.n_ple, m.n_ple, T),
                  brows(m.b_p, T));
        RAD_ISSUE(c, y.op_ple_proj, brows(m.b_p, T), RAD_W(y.w_ple_proj), brows(m.b_o, T));
        RAD_ISSUE(c, y.op_post_ple, brows(m.b_o, T), RAD_W(y.w_post_ple), brows(m.b_o, T));
        RAD_ISSUE(c, y.op_add_ple, brows(m.b_x, T), brows(m.b_o, T), brows(m.b_x, T));
        RAD_ISSUE(c, y.op_scale, brows(m.b_x, T), RAD_W(y.w_scalar), brows(m.b_x, T));
    }

    RAD_ISSUE(c, m.op_norm, brows(m.b_x, T), RAD_W(m.w_norm), brows(m.b_h, T));
    RAD_ISSUE(c, m.op_mean, brows(m.b_h, T), cu, brows(m.b_mean, S));
    RAD_ISSUE_N(c, m.op_out, S, brows(m.b_mean, S), RAD_W(m.w_out), brows(m.b_proj, S));
    RAD_ISSUE_N(c, m.op_cast, S, brows(m.b_proj, S), brows(m.b_pooled, S));
    (void)E;
}

}  /* namespace embgemma2 */

#ifndef RAD_ARCH_NO_EXPORTS
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(embgemma2, "embeddinggemma2", "", "0.1.0",
                "EmbeddingGemma 2: a bidirectional Gemma encoder with per-layer embeddings, "
                "mean-pooled to one vector an input; bf16")
RAD_ARCH_PROBE(embgemma2)
#endif
