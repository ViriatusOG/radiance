/* museglimmer.cpp -- Meta's Muse Glimmer 30B, bf16 weights and bf16 activations.
 *
 * Muse Glimmer (HF `meta-models/Muse-Glimmer-30B`, model_type `muse_glimmer`) is a dense text
 * model with a separate vision tower. The text trunk here is served by this plugin; the tower and
 * its projector are declared at the bottom of declare() when the deployment asked for media.
 *
 * ============================== WHAT IT SERVES ==============================
 *
 * The unquantised release (quantisation descriptor ""), every projection a bf16 GEMM. The lm_head
 * is declared at whatever the model holds, so a recipe-made block-fp8 head uses the fp8 logits
 * kernel.
 *
 * ============================== THE SHAPE, AT 30B ==============================
 *
 *   52 layers, `layer_types` full_attention on every 4th layer counted back from the last
 *                    (3, 7, ..., 51; the shipped config states the array, and this plugin checks
 *                    it against full_attention_interval 4 rather than trusting either alone)
 *   hidden 6656   ffn 19968   vocab 202048   (lm_head not tied)
 *   32 heads / 2 kv heads / head_dim 128 -> GQA 16
 *   sliding_window 2048: the sliding layers own a RAD_KV_WINDOW pool, the full layers a
 *                    RAD_KV_FULL pool; a sequence holds one state per bound layer of each group
 *   q and output gate are SEPARATE checkpoint tensors (`q_proj`, `gate_proj`), unlike Qwen3.5's
 *                    interleaved q_proj -- two GEMMs, two buffers, no fused q|gate row
 *   `layer_rope_theta`: 500000 on the sliding layers, 0 on the full ones. A 0 layer is NoPE: no
 *                    rotary is declared or issued there (the reference passes
 *                    position_embeddings=None). max_position_embeddings 131072, rope_type default
 *
 * ============================== WHAT DIFFERS FROM A LLAMA-LIKE ==============================
 *
 * Every deviation from the Qwen/Llama shape below is load-bearing; getting one wrong yields
 * fluent wrong text (spec §17 refuses instead).
 *
 *   * Four norms a layer, all Gemma-2 style "centered" norms, x_hat * (1 + w): the attention's
 *     input (rms_norm_eps 1e-5) and its output (post_norm_eps 1e-8), then the feed-forward's input
 *     (1e-5) and output (1e-8). The residual adds the *normed* branch, not the raw one.
 *   * The q/k "norm" is weightless: MuseGlimmerRMSNorm(with_scale=False), a plain RMS norm over
 *     head_dim at rms_norm_eps. The query is then multiplied by `qk_scale_factor` (3.87); the key
 *     is not. The attention scale stays 1/sqrt(head_dim). `rmsnorm` serves both: no gain operand,
 *     `mult` on the query.
 *   * The embedding is normed after the lookup: a weightless RMS norm at rms_norm_eps on the
 *     gathered rows (MuseGlimmerTextNormedEmbedding). It cannot be folded into the table because
 *     a drafter embeds without it, so it is an op.
 *   * The final norm (`model.language_model.norm.weight`) is a plain RMSNorm WITH a gain and
 *     without the centered 1+: wadd 0, unlike every layer norm.
 *   * The logits are pre-scaled by `output_multiplier` (1/sqrt(26)) before the Gemma-style tanh
 *     softcap: 20 * tanh(logits * mult / 20). The multiplier is carried on the final norm's
 *     `mult` (its output feeds only the head), the cap on `logit_softcap` over the f32 logits.
 *
 * ============================== WHAT THIS PLUGIN DOES NOT DECLARE ==============================
 *
 * The speculative drafter Meta ships beside the model (a DFlash block stack over layers
 * 1, 13, 25, 37, 49) is not declared: it lives in its own checkpoint with its own `draft.`
 * config, which this tree has no fixture for. Serving the trunk alone is a complete model.
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_museglimmer_vision.h>
#include <arch/rad_block_muse_dflash.h>
#include <arch/rad_calib_full.h>
#include <arch/rad_museglimmer_chat.h>
#include "dflash2.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace museglimmer {

using namespace rad::arch;

/* One trunk layer. The rope ops exist only where layer_rope_theta says so; the branch is fixed
 * per layer by the model. */
struct Layer {
    int full = 0;      /* full attention (no window, no rope in this family) */
    int rope = 0;      /* carries a rotary (the sliding layers) */

    rad_weight w_norm = 0, w_post = 0, w_q = 0, w_gate = 0, w_k = 0, w_v = 0, w_o = 0;
    rad_weight w_fnorm = 0, w_fpost = 0, w_gate_up = 0, w_down = 0;

    rad_op op_norm = 0, op_q = 0, op_gate = 0, op_k = 0, op_v = 0;
    rad_op op_qn = 0, op_kn = 0, op_rq = 0, op_rk = 0;
    rad_op op_kv = 0, op_attn = 0, op_sig = 0, op_mul = 0, op_o = 0, op_ar = 0;
    rad_op op_post = 0, op_add = 0;
    rad_op op_fnorm = 0, op_gu = 0, op_act = 0, op_down = 0, op_far = 0, op_fpost = 0, op_fadd = 0;

    /* The layer's calibration inputs (rad_calib_full.h): -1 where the run does not record it. */
    int cs_in = -1, cs_o = -1, cs_mlp = -1, cs_down = -1;
};

struct Model {
    Names nm{""};
    Geom  g{};        /* the trunk geometry, the rank's shard of it */
    Geom  g_rope{};   /* the same, with the sliding layers' rotary base (decl_rope reads theta) */
    float qk_mult  = 0;   /* qk_scale_factor */
    float post_eps = 0;   /* post_norm_eps */
    float out_mult = 0;   /* output_multiplier */
    float softcap  = 0;   /* final_logit_softcapping; 0: none */
    int64_t window = 0;   /* sliding_window */

    rad_kvgroup kv_win = 0, kv_full = 0;

    /* the residual stream and the two vocabulary edges */
    rad_buf b_x = 0, b_h = 0, b_hout = 0, b_logits = 0;
    rad_buf b_q = 0, b_gate = 0, b_k = 0, b_v = 0, b_attn = 0;
    rad_buf b_gu = 0, b_ffn = 0;

    rad_weight w_tok = 0, w_out_norm = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_embed_norm = 0;
    rad_op op_final_norm = 0, op_gather = 0, op_logits = 0, op_cap = 0;

    /* ---- media (arch/common/rad_block_museglimmer_vision.h): the vision tower and the scatter of
     * its rows, declared only when the container carries a tower and the deployment asked for
     * media. The tower is declared last, so its activations share the arena with every buffer
     * above. */
    bool      media = false;
    rad_op    op_media_rows = 0;
    MuseTower vit{};

    /* ---- Meta's DFlash or incoai's DFlash2, selected by draft.architectures. */
    bool        have_df = false, have_df2 = false;
    MuseDflash  df{};
    MuseDflash::Wire dfw{};
    Dflash2     df2{};
    Dflash2::Wire df2w{};
    MuseTaps    taps{};
    rad_kvgroup kv_draft = 0;

    std::vector<Layer> layers;

    /* --calib-dir: the whole Hessian of every linear's input over a --calib-corpus run. */
    CalibFull calib;
    int       cs_head = -1;
};

/* Declare runs once per rank of an instance, on the rank's own thread (spec §1, §9); every
 * instance's state is its slot's (rad_build_slot / rad_slot). */
static Model g_model[MAX_SLOTS];

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;

    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));

    /* A routed model is not this family. */
    if (meta->n_expert > 0) {
        fprintf(stderr, "radiance: museglimmer refuses a checkpoint with %lld experts\n",
                (long long)meta->n_expert);
        return RAD_E_UNSUPPORTED;
    }

    Geom&  g  = m.g;
    Names& nm = m.nm;

    /* ---- the family's own config. Everything that changes the mathematics is required or read
     * from the family defaults the reference config class states; a wrong default yields fluent
     * wrong text, so nothing here falls back to a "reasonable" value silently. */
    m.qk_mult  = (float)meta_text_getf(meta, "qk_scale_factor", 3.87);
    m.post_eps = (float)meta_text_getf(meta, "post_norm_eps", 1e-8);
    m.out_mult = (float)meta_text_getf(meta, "output_multiplier", 0.19611613513818404);
    m.softcap  = (float)meta_text_getf(meta, "final_logit_softcapping", 20.0);
    m.window   = meta_text_geti(meta, "sliding_window", 2048);
    const double theta_rope = meta_text_getf(meta, "rope_parameters.rope_theta",
                                    meta_text_getf(meta, "rope_theta", 0.0));
    if (!(m.qk_mult > 0.0f) || !(m.out_mult > 0.0f) || !(m.post_eps > 0.0f) || m.window <= 0 ||
        !(theta_rope > 0.0)) {
        fprintf(stderr, "radiance: museglimmer: qk_scale_factor %g output_multiplier %g "
                        "post_norm_eps %g sliding_window %lld rope theta %g are all required\n",
                (double)m.qk_mult, (double)m.out_mult, (double)m.post_eps,
                (long long)m.window, theta_rope);
        return RAD_E_INVAL;
    }
    if (g.rope_mc) {
        fprintf(stderr, "radiance: museglimmer: the trunk has one-component rotary; a container "
                        "with mrope_section is not this family\n");
        return RAD_E_UNSUPPORTED;
    }

    /* The rotary width is the full head: rope_parameters has no partial_rotary_factor and the
     * reference rotates all of head_dim. */
    const int64_t n_rot = g.head_dim;

    /* ---- layer kinds. The config ships `layer_types`; where it does not (a GGUF container), the
     * interval rule stands in. The two must agree where both exist: the schedule decides which
     * layers keep a long context, and a wrong layer still generates fluent text. */
    std::vector<int> full;
    const char* kinds = meta_text_gets(meta, "layer_types", nullptr);
    if (kinds && *kinds)
        RAD_ARCH_TRY(parse_layer_kinds("museglimmer", kinds, {"full_attention"},
                                       "sliding_attention", &full));
    const int64_t interval = 4;
    if ((int64_t)full.size() < g.n_layer) {
        full.assign((size_t)g.n_layer, 0);
        for (int64_t l = 0; l < g.n_layer; ++l) full[(size_t)l] = is_full_attention(l, interval);
        if (kinds && *kinds)
            rad_note(b, "the container's layer_types named %zu layers and the model has %lld; "
                        "building the full-attention layers by the interval %lld",
                    full.size() - (size_t)g.n_layer + (size_t)g.n_layer, (long long)g.n_layer,
                    (long long)interval);
    } else {
        for (int64_t l = 0; l < g.n_layer; ++l) {
            if ((full[(size_t)l] != 0) != is_full_attention(l, interval)) {
                fprintf(stderr, "radiance: museglimmer: layer_types makes layer %lld %s and "
                                "full_attention_interval %lld makes it %s\n",
                        (long long)l, full[(size_t)l] ? "full" : "sliding", (long long)interval,
                        is_full_attention(l, interval) ? "full" : "sliding");
                return RAD_E_UNSUPPORTED;
            }
        }
    }

    /* Per-layer rotary: the shipped `layer_rope_theta`, else the family pattern (theta on the
     * sliding layers, 0 -- NoPE -- on the full ones). */
    std::vector<float> theta((size_t)g.n_layer, 0.0f);
    const char* thetas = meta_text_gets(meta, "layer_rope_theta", nullptr);
    if (thetas && *thetas) {
        size_t i = 0;
        for_each_token(thetas, [&](const char* t, size_t n) {
            if (i >= theta.size()) return false;
            const std::string tok(t, n);
            char* e = nullptr;
            const double v = std::strtod(tok.c_str(), &e);
            theta[i++] = e == tok.c_str() ? -1.0f : (float)v;
            return true;
        });
        if (i < theta.size()) {
            fprintf(stderr, "radiance: museglimmer: layer_rope_theta names %zu layers and the "
                            "model has %lld\n", i, (long long)g.n_layer);
            return RAD_E_INVAL;
        }
    } else {
        for (int64_t l = 0; l < g.n_layer; ++l)
            theta[(size_t)l] = full[(size_t)l] ? 0.0f : (float)theta_rope;
    }
    for (int64_t l = 0; l < g.n_layer; ++l) {
        if (!(theta[(size_t)l] >= 0.0f)) {
            fprintf(stderr, "radiance: museglimmer: layer_rope_theta[%lld] is not a number\n",
                    (long long)l);
            return RAD_E_INVAL;
        }
    }

    /* Two Geoms: the trunk's for the buffers and shards, one with the sliding layers' rotary base
     * for the rope ops (decl_rope takes theta from the Geom it is given). The rotary scaling is
     * read into the rope Geom, against the sliding layers' own base: the extension reaches their
     * rope and leaves the NoPE full layers untouched (they have no rotary to scale), and the
     * mechanical extension past the trained 131072 positions -- which this family takes well, the
     * full layers carrying no positions and the sliding ones reading only in-window relative
     * offsets -- is the no-keys default this call returns. */
    m.g_rope          = g;
    m.g_rope.theta    = (float)theta_rope;
    RAD_ARCH_TRY(yarn_from(m.g_rope, meta, "", g.head_dim));

    const int64_t q_dim  = g.q_dim();
    const int64_t kv_dim = g.kv_dim();

    /* ---- two KV groups (spec §7.2): the sliding layers' pool recycles blocks below the window,
     * the full layers' does not. A group with no bound layers is a declare error. */
    const uint32_t kvdt = rad_kv_cache_dtype(ctx, RAD_BF16);
    RadKVGroupDecl sd{};
    sd.kind      = RAD_KV_WINDOW;
    sd.dtype     = kvdt;
    sd.window    = m.window;
    sd.n_head_kv = g.n_head_kv;
    sd.head_dim  = g.head_dim;
    m.kv_win = rad_decl_kv_group(b, nm.f("kv_sliding"), &sd);
    RadKVGroupDecl gd{};
    gd.kind      = RAD_KV_FULL;
    gd.dtype     = kvdt;
    gd.n_head_kv = g.n_head_kv;
    gd.head_dim  = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_full"), &gd);
    g.kv_dtype = rad_dtype_name(kvdt);
    if (m.kv_win == 0 || m.kv_full == 0) {
        fprintf(stderr, "radiance: museglimmer: a KV group failed to declare\n");
        return RAD_E_INVAL;
    }

    /* ---- activation buffers, sized at max_tok and shared by every layer; liveness over the
     * declared read and write sets fits them in one arena (spec §3.1). */
    m.b_x    = decl_b(b, nm.f("x"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.b_h    = decl_b(b, nm.f("h"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.b_q    = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, q_dim});
    m.b_gate = decl_b(b, nm.f("attn_gate"),g.act_dtype, {g.max_tok, q_dim});
    m.b_k    = decl_b(b, nm.f("attn_k"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_v    = decl_b(b, nm.f("attn_v"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_attn = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, q_dim});
    m.b_gu   = decl_b(b, nm.f("gate_up"),  g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.b_ffn  = decl_b(b, nm.f("ffn"),      g.act_dtype, {g.max_tok, g.n_ff});
    /* The lm_head runs over the sampled rows only (RadBatch::out_ids), not the whole step. */
    m.b_hout   = decl_b(b, nm.f("h_out"),   g.act_dtype, {g.max_logit_rows, g.n_embd});
    m.b_logits = decl_b(b, nm.f("logits"),  RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the drafter's taps. Declared before the trunk's layers: the buffer planner takes
     * lifetimes in declaration order, so a tap buffer declared after them could share bytes a
     * layer temporary lands on, and acceptance would collapse with no other symptom. */
    if (ctx->max_spec > 0 && m.taps.read(meta, g.n_layer) > 0) {
        RAD_ARCH_TRY(m.taps.declare(b, nm, g, g.max_tok, m.b_x, g.act_dtype));
        if (m.taps.n <= 0) {
            fprintf(stderr, "radiance: museglimmer: the container carries a drafter whose "
                            "draft.target_layer_ids name no layer of this %lld-layer trunk\n",
                    (long long)g.n_layer);
            return RAD_E_INVAL;
        }
    }


    /* ---- vocabulary edges, both row-sharded. The 30B does not tie its head; the second source
     * serves family members that do, first present wins. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"),
                          nm.ckpt("model.language_model.norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.language_model.embed_tokens.weight")));

    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    /* A plain RMSNorm with a gain: declared at f32, cast at load, read with wadd 0. */
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_lm_head  = decl_w(b, nm.f("output.weight"),
                          weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                          {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    if (g.world > 1)
        m.op_embed_ar = decl_all_reduce_exact(b, g, m.b_x);
    /* The embedding norm, after the vocabulary shard's reduction: a weightless RMS norm at
     * rms_norm_eps, in place. Weightless because MuseGlimmerTextNormedEmbedding has no gain. */
    m.op_embed_norm = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_STR("dtype", g.dtype)),
                             RAD_NOWEIGHTS),
                         {m.b_x}, {m.b_x});

    /* The encoder's rows overwrite the lookup before its reduction (as rad_block_vit.h's
     * MediaRows): declared only when the tower is served, issued on the passes that carry rows,
     * so the reduction counts each row once. */
    m.media = MuseTower::present(meta) && ctx->max_enc_patches > 0;
    if (m.media)
        m.op_media_rows = rw(b, RAD_OP(b, "scatter_rows",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                         RAD_STR("dtype", g.dtype)),
                              RAD_NOWEIGHTS),
                          {m.b_x}, {m.b_x});

    /* ---- calibration: the widest split input bounds the Hessian's width. */
    RAD_ARCH_TRY(m.calib.begin(b, nm, g, ctx, g.n_layer,
                               std::max({g.n_embd, q_dim * g.world, g.n_ff * g.world}),
                               std::max({g.n_embd, q_dim * g.world, g.n_ff * g.world})));

    /* ---- the layers. */
    /* The attention scale stays 1/sqrt(head_dim): the query's qk_scale_factor rides the q norm's
     * `mult`. */
    m.layers.resize((size_t)g.n_layer);
    for (int l = 0; l < g.n_layer; ++l) {
        Layer&   lay  = m.layers[(size_t)l];
        lay.full     = full[(size_t)l];
        lay.rope     = theta[(size_t)l] > 0.0f;
        const bool   glob = lay.full;
        rad_kvgroup  kg   = glob ? m.kv_full : m.kv_win;
        const char*  pre  = "model.language_model.layers";
        const Geom&  gr   = lay.rope ? m.g_rope : m.g;

        /* -- weights. Shapes are [out, in]; gemm_nt reads B transposed. */
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l),
                              nm.ckpt("%s.%lld.input_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_post.weight", l),
                              nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q.weight", l),
                              nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_gate.weight", l),
                              nm.ckpt("%s.%lld.self_attn.gate_proj.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k.weight", l),
                              nm.ckpt("%s.%lld.self_attn.k_proj.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_v.weight", l),
                              nm.ckpt("%s.%lld.self_attn.v_proj.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_output.weight", l),
                              nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_norm.weight", l),
                              nm.ckpt("%s.%lld.pre_feedforward_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_post.weight", l),
                              nm.ckpt("%s.%lld.post_feedforward_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_concat(b, nm.f("blk.%d.ffn_gate_up.weight", l),
                                {nm.ckpt("%s.%lld.mlp.gate_proj.weight", pre, (long long)l),
                                 nm.ckpt("%s.%lld.mlp.up_proj.weight", pre, (long long)l)},
                                0));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_down.weight", l),
                              nm.ckpt("%s.%lld.mlp.down_proj.weight", pre, (long long)l)));

        lay.w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        lay.w_post = decl_w(b, nm.f("blk.%d.attn_post.weight", l), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        lay.w_q    = decl_w(b, nm.f("blk.%d.attn_q.weight", l), RAD_BF16, {q_dim, g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
        lay.w_gate = decl_w(b, nm.f("blk.%d.attn_gate.weight", l), RAD_BF16, {q_dim, g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
        lay.w_k    = decl_w(b, nm.f("blk.%d.attn_k.weight", l), RAD_BF16, {kv_dim, g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
        lay.w_v    = decl_w(b, nm.f("blk.%d.attn_v.weight", l), RAD_BF16, {kv_dim, g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
        lay.w_o    = decl_w(b, nm.f("blk.%d.attn_output.weight", l), RAD_BF16, {g.n_embd, q_dim},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, grp_layer(l));
        lay.w_fnorm  = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_F32, {g.n_embd},
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        lay.w_fpost  = decl_w(b, nm.f("blk.%d.ffn_post.weight", l), RAD_F32, {g.n_embd},
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        lay.w_gate_up= decl_w(b, nm.f("blk.%d.ffn_gate_up.weight", l), RAD_BF16,
                              {2 * g.n_ff, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,
                              grp_layer(l), 0, {g.n_ff, g.n_ff});
        lay.w_down   = decl_w(b, nm.f("blk.%d.ffn_down.weight", l), RAD_BF16, {g.n_embd, g.n_ff},
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_COL, grp_layer(l));

        /* -- ops, in run order except attn_paged, declared before kv_store like every family: the
         * two touch disjoint buffers and the block size comes from the resolved kernel. */
        lay.op_norm = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", 1.0),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(lay.w_norm)),
                        {m.b_x}, {m.b_h});
        lay.op_q    = rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", q_dim),
                                       RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(lay.w_q)),
                        {m.b_h}, {m.b_q});
        lay.op_gate = rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", q_dim),
                                       RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(lay.w_gate)),
                        {m.b_h}, {m.b_gate});
        lay.op_k    = rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", kv_dim),
                                       RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(lay.w_k)),
                        {m.b_h}, {m.b_k});
        lay.op_v    = rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", kv_dim),
                                       RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(lay.w_v)),
                        {m.b_h}, {m.b_v});

        /* The q/k norms are weightless (no gain operand), the query carrying qk_scale_factor on
         * `mult`. Both run in place: a row's scale is computed before the row is written. */
        lay.op_qn = rw(b, RAD_OP(b, "rmsnorm",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                     RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                     RAD_F64("mult", (double)m.qk_mult), RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {m.b_q}, {m.b_q});
        lay.op_kn = rw(b, RAD_OP(b, "rmsnorm",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head_kv),
                                     RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {m.b_k}, {m.b_k});

        if (lay.rope) {
            lay.op_rq = rw(b, decl_rope(b, gr, g.max_tok, g.head_dim, g.n_head, 0, n_rot, "neox"),
                           {m.b_q}, {m.b_q});
            lay.op_rk = rw(b, decl_rope(b, gr, g.max_tok, g.head_dim, g.n_head_kv, 0, n_rot, "neox"),
                           {m.b_k}, {m.b_k});
        }

        lay.op_attn = rw(b, rad_decl_op(b, "attn_paged",
                             RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok),
                                        RAD_INT("head_dim", g.head_dim), RAD_INT("gqa", g.gqa()),
                                        RAD_INT("causal", 1),
                                        RAD_INT("window", glob ? 0 : m.window),
                                        RAD_INT("n_head", g.n_head),
                                        RAD_INT("block_size", RAD_KV_BLOCK),
                                        RAD_INT("max_seqs", g.max_seqs),
                                        RAD_INT("max_ctx", g.max_ctx),
                                        RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                             RAD_NOWEIGHTS),
                         {m.b_q}, {m.b_attn});

        lay.op_kv = rw(b, RAD_OP(b, "kv_store",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                      RAD_INT("n_head_kv", g.n_head_kv),
                                      RAD_INT("block_size", rad_kv_block_size(b, kg)),
                                      RAD_STR("kv_dtype", g.kv_dtype)),
                           RAD_NOWEIGHTS),
                       {m.b_k, m.b_v}, {});

        /* out = attn * sigmoid(gate): two ops, as in the gated-attention block. */
        lay.op_sig = rw(b, RAD_OP(b, "sigmoid",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                       RAD_INT("n", g.head_dim), RAD_STR("dtype", g.dtype)),
                            RAD_NOWEIGHTS),
                        {m.b_gate}, {m.b_gate});
        lay.op_mul = rw(b, RAD_OP(b, "mul",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_NOWEIGHTS),
                        {m.b_attn, m.b_gate}, {m.b_attn});

        lay.op_o = rw(b, RAD_OP(b, "gemm_nt",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                                     RAD_INT("K", q_dim), RAD_STR("dtype", g.dtype)),
                          RAD_WEIGHTS(lay.w_o)),
                      {m.b_attn}, {m.b_h});
        if (g.world > 1)
            lay.op_ar = decl_all_reduce(b, g, m.b_h);
        /* The branch's own norm, at post_norm_eps, before the residual add. */
        lay.op_post = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", m.post_eps), RAD_F64("wadd", 1.0),
                                        RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(lay.w_post)),
                         {m.b_h}, {m.b_h});
        lay.op_add = rw(b, RAD_OP(b, "add",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_NOWEIGHTS),
                        {m.b_x, m.b_h}, {m.b_x});

        /* ---- the feed-forward, the same double norm. */
        lay.op_fnorm = rw(b, RAD_OP(b, "rmsnorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                         RAD_F64("eps", g.eps), RAD_F64("wadd", 1.0),
                                         RAD_STR("dtype", g.dtype)),
                              RAD_WEIGHTS(lay.w_fnorm)),
                          {m.b_x}, {m.b_h});
        lay.op_gu  = rw(b, RAD_OP(b, "gemm_nt",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", 2 * g.n_ff),
                                     RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                          RAD_WEIGHTS(lay.w_gate_up)),
                      {m.b_h}, {m.b_gu});
        lay.op_act = rw(b, RAD_OP(b, "silu_mul",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {m.b_gu}, {m.b_ffn});
        lay.op_down = rw(b, RAD_OP(b, "gemm_nt",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                                     RAD_INT("K", g.n_ff), RAD_STR("dtype", g.dtype)),
                          RAD_WEIGHTS(lay.w_down)),
                      {m.b_ffn}, {m.b_h});
        if (g.world > 1)
            lay.op_far = decl_all_reduce(b, g, m.b_h);
        lay.op_fpost = rw(b, RAD_OP(b, "rmsnorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                         RAD_F64("eps", m.post_eps), RAD_F64("wadd", 1.0),
                                         RAD_STR("dtype", g.dtype)),
                              RAD_WEIGHTS(lay.w_fpost)),
                          {m.b_h}, {m.b_h});
        lay.op_fadd = rw(b, RAD_OP(b, "add",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                         RAD_STR("dtype", g.dtype)),
                              RAD_NOWEIGHTS),
                          {m.b_x, m.b_h}, {m.b_x});

        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, kg));

        /* -- calibration taps, the same four a dense family records. */
        if (m.calib.records(l)) {
            const std::string L = std::string(pre) + "." + std::to_string(l);
            lay.cs_in = (int)m.calib.sites.size();
            RAD_ARCH_TRY(m.calib.site(b, g, l, L + ".input",
                                      L + ".self_attn.q_proj," + L + ".self_attn.k_proj," +
                                      L + ".self_attn.v_proj," + L + ".self_attn.gate_proj",
                                      m.b_h, g.n_embd, g.n_embd));
            lay.cs_o = (int)m.calib.sites.size();
            RAD_ARCH_TRY(m.calib.site(b, g, l, L + ".self_attn.o_proj.input", L + ".self_attn.o_proj",
                                      m.b_attn, q_dim, q_dim * g.world));
            lay.cs_mlp = (int)m.calib.sites.size();
            RAD_ARCH_TRY(m.calib.site(b, g, l, L + ".pre_feedforward.input",
                                      L + ".mlp.gate_proj," + L + ".mlp.up_proj",
                                      m.b_h, g.n_embd, g.n_embd));
            lay.cs_down = (int)m.calib.sites.size();
            RAD_ARCH_TRY(m.calib.site(b, g, l, L + ".mlp.down_proj.input", L + ".mlp.down_proj",
                                      m.b_ffn, g.n_ff, g.n_ff * g.world));
        }
    }

    /* ---- the head. The final norm is a plain RMSNorm (wadd 0) that also carries the logits'
     * pre-scale: its output feeds only the head, so mult here is logits * output_multiplier
     * exactly. */
    m.op_final_norm = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                        RAD_F64("mult", (double)m.out_mult), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(m.w_out_norm)),
                         {m.b_x}, {m.b_h});
    if (m.calib.on && m.calib.hi == g.n_layer) {
        m.cs_head = (int)m.calib.sites.size();
        RAD_ARCH_TRY(m.calib.site(b, g, -1, "lm_head.input", "lm_head", m.b_h, g.n_embd, g.n_embd));
    }

    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {m.b_h}, {m.b_hout});
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", g.n_embd),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(m.w_lm_head)),
                     {m.b_hout}, {m.b_logits});
    /* cap * tanh(x / cap) over the f32 logits, after the pre-scale. */
    if (m.softcap > 0.0f) {
        m.op_cap = rw(b, RAD_OP(b, "logit_softcap",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                     RAD_INT("n", g.n_vocab), RAD_F64("cap", (double)m.softcap),
                                     RAD_STR("dtype", "f32")),
                          RAD_NOWEIGHTS),
                      {m.b_logits}, {m.b_logits});
        if (!m.op_cap)
            rad_note(b, "warning: no kernel serves `logit_softcap` (load libr4d or libavx), so "
                        "the logits skip %g * tanh(x / %g): greedy output is unchanged, sampled "
                        "output draws from a sharper distribution", (double)m.softcap,
                    (double)m.softcap);
    }

    /* ---- media: the vision tower, declared last so its activations share the arena with every
     * buffer above (the trunk's weights were mapped and declared first). */
    if (m.media)
        RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer));

    /* ---- the drafter is declared last so its buffers cannot overlap trunk temporaries. */
    if (m.taps.n > 0) {
        if (muse_draft_is_dflash2(meta)) {
            RAD_ARCH_TRY(declare_muse_dflash2(&m.df2, &m.df2w, &m.kv_draft, b, nm, meta, ctx, g,
                                              m.taps, (int)g.n_layer, m.w_tok, m.w_lm_head));
            m.have_df2 = true;
        } else {
            RAD_ARCH_TRY(declare_muse_dflash(&m.df, &m.dfw, &m.kv_draft, b, nm, meta, ctx, g,
                                             m.taps, (int)g.n_layer, m.w_tok, m.w_lm_head));
            m.have_df = true;
        }
    }

    /* ---- notes for --debug-graph. */
    int64_t nfull = 0, nrope = 0;
    for (int64_t l = 0; l < g.n_layer; ++l) {
        nfull += full[(size_t)l];
        nrope += theta[(size_t)l] > 0.0f;
    }
    rad_note(b, "rank %d/%d: %lld layers (%lld full attention, %lld sliding over a %lld-token "
                "window; %lld with rotary at theta %g), n_head=%lld n_head_kv=%lld head_dim=%lld "
                "n_ff=%lld vocab=[%lld,%lld)",
             g.rank, g.world, (long long)g.n_layer, (long long)nfull,
             (long long)(g.n_layer - nfull), (long long)m.window, (long long)nrope,
             theta_rope, (long long)g.n_head, (long long)g.n_head_kv, (long long)g.head_dim,
             (long long)g.n_ff, (long long)g.vocab_off, (long long)(g.vocab_off + g.n_vocab));
    rad_note(b, "centered norms (1 + w): input/pre at eps %g, post at eps %g; the final norm is a "
                "plain gain; the embedding and the q/k norms are weightless", (double)g.eps,
            (double)m.post_eps);
    rad_note(b, "attention scale 1/sqrt(%lld) with the query pre-scaled by qk_scale_factor %g; "
                "logits pre-scaled by output_multiplier %g%s%g",
             (long long)g.head_dim, (double)m.qk_mult, (double)m.out_mult,
             m.softcap > 0.0f ? " and soft-capped at " : "", (double)m.softcap);
    if (m.g_rope.theta != g.theta)
        rad_note(b, "rotary runs on the sliding layers only (theta %g); the full-attention layers "
                    "are NoPE", (double)m.g_rope.theta);
    rad_note(b, "logits for at most %lld sampled rows a step (of %lld tokens): gather_rows "
                "selects RadBatch::out_ids before the lm_head", (long long)g.max_logit_rows,
            (long long)g.max_tok);

    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_slot(c)];

    /* An encoder pass runs only the vision tower (RadBatch::enc). */
    if (batch->enc) {
        m.vit.step(c, batch);
        return;
    }

    /* A draft pass runs only the drafter: >0 the query pass on the staged block, <0 the context
     * pass that projects and stores K/V for the tokens the trunk just committed. */
    if (batch->draft_pass != 0) {
        if (m.have_df2) {
            if (batch->draft_pass > 0) m.df2.step(c, batch);
            else                       m.df2.context(c, batch);
            return;
        }
        if (!m.have_df) {
            rad_step_fail(c, "this serve stages a draft pass but the container carries no drafter");
            return;
        }
        if (batch->draft_pass > 0) m.df.step(c, batch);
        else                       m.df.context(c, batch);
        return;
    }

    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
              brows(m.b_x, batch->n_tok));
    /* The encoder's rows overwrite the lookup before the vocabulary shard's reduction, so the
     * reduction counts each row once. */
    if (m.op_media_rows && batch->n_mm_rows > 0)
        RAD_ISSUE_N(c, m.op_media_rows, batch->n_mm_rows,
                    praw2(batch->mm_embd, RAD_BF16, batch->n_mm_rows, m.g.n_embd),
                    praw(batch->mm_rows, RAD_I32, batch->n_mm_rows),
                    brows(m.b_x, batch->n_tok));
    /* The vocab shard's reduction; declared only when world > 1, so tp1 is unchanged. */
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);
    /* The embedding norm, after the reduction: it normalises whole rows. */
    RAD_ISSUE(c, m.op_embed_norm, brows(m.b_x, batch->n_tok), RAD_NONE, brows(m.b_x, batch->n_tok));

    const int64_t T = batch->n_tok;
    for (size_t li = 0; li < m.layers.size(); ++li) {
        const Layer& l = m.layers[li];
        const rad_kvgroup kg = l.full ? m.kv_full : m.kv_win;
        const RadKVGroupBatch* kvb = kv_batch(batch, kg);
        const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
        const int32_t* table = kvb ? kvb->block_table  : nullptr;
        const int32_t* used  = kvb ? kvb->seqused      : nullptr;

        /* -- attention. The first Hessian sees the normalized input consumed by q/gate/k/v. */
        RAD_ISSUE(c, l.op_norm, brows(m.b_x, T), RAD_W(l.w_norm), brows(m.b_h, T));
        if (l.cs_in >= 0) m.calib.tap(c, (size_t)l.cs_in, T);
        RAD_ISSUE(c, l.op_q,    brows(m.b_h, T), RAD_W(l.w_q),    brows(m.b_q, T));
        RAD_ISSUE(c, l.op_gate, brows(m.b_h, T), RAD_W(l.w_gate), brows(m.b_gate, T));
        RAD_ISSUE(c, l.op_k,    brows(m.b_h, T), RAD_W(l.w_k),    brows(m.b_k, T));
        RAD_ISSUE(c, l.op_v,    brows(m.b_h, T), RAD_W(l.w_v),    brows(m.b_v, T));
        RAD_ISSUE_N(c, l.op_qn, T * m.g.n_head,      brows(m.b_q, T), RAD_NONE, brows(m.b_q, T));
        RAD_ISSUE_N(c, l.op_kn, T * m.g.n_head_kv,   brows(m.b_k, T), RAD_NONE, brows(m.b_k, T));
        if (l.rope) {
            RAD_ISSUE(c, l.op_rq, brows(m.b_q, T), rope_posmc(m.g, batch, T));
            RAD_ISSUE(c, l.op_rk, brows(m.b_k, T), rope_posmc(m.g, batch, T));
        }
        RAD_ISSUE(c, l.op_kv,
                  brows(m.b_k, T), brows(m.b_v, T), praw(slot, RAD_I32, T),
                  kv_cache(kg, (int)li));
        /* The longest query length, not the token count, selects the attention band. */
        RAD_ISSUE_N(c, l.op_attn, batch->max_q_len,
                    brows(m.b_q, T), kv_cache(kg, (int)li),
                    praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(m.b_attn, T));
        RAD_ISSUE_N(c, l.op_sig, T * m.g.n_head, brows(m.b_gate, T), brows(m.b_gate, T));
        RAD_ISSUE(c, l.op_mul, brows(m.b_attn, T), brows(m.b_gate, T), brows(m.b_attn, T));
        if (l.cs_o >= 0) m.calib.tap(c, (size_t)l.cs_o, T);
        RAD_ISSUE(c, l.op_o,   brows(m.b_attn, T), RAD_W(l.w_o),    brows(m.b_h, T));
        if (l.op_ar) RAD_ISSUE_N(c, l.op_ar, T * m.g.n_embd, brows(m.b_h, T), RAD_NONE);
        RAD_ISSUE(c, l.op_post, brows(m.b_h, T), RAD_W(l.w_post), brows(m.b_h, T));
        RAD_ISSUE(c, l.op_add,  brows(m.b_x, T), brows(m.b_h, T), brows(m.b_x, T));

        /* -- feed-forward. */
        RAD_ISSUE(c, l.op_fnorm, brows(m.b_x, T), RAD_W(l.w_fnorm), brows(m.b_h, T));
        if (l.cs_mlp >= 0) m.calib.tap(c, (size_t)l.cs_mlp, T);
        RAD_ISSUE(c, l.op_gu,    brows(m.b_h, T), RAD_W(l.w_gate_up), brows(m.b_gu, T));
        RAD_ISSUE(c, l.op_act,   brows(m.b_gu, T), brows(m.b_ffn, T));
        if (l.cs_down >= 0) m.calib.tap(c, (size_t)l.cs_down, T);
        RAD_ISSUE(c, l.op_down,  brows(m.b_ffn, T), RAD_W(l.w_down), brows(m.b_h, T));
        if (l.op_far) RAD_ISSUE_N(c, l.op_far, T * m.g.n_embd, brows(m.b_h, T), RAD_NONE);
        RAD_ISSUE(c, l.op_fpost, brows(m.b_h, T), RAD_W(l.w_fpost), brows(m.b_h, T));
        RAD_ISSUE(c, l.op_fadd,  brows(m.b_x, T), brows(m.b_h, T), brows(m.b_x, T));

        /* The drafter reads this layer's completed residual, if it taps it. */
        m.taps.step(c, m.b_x, T, (int64_t)li, m.g.n_embd);
    }

    RAD_ISSUE(c, m.op_final_norm,
              brows(m.b_x, batch->n_tok), RAD_W(m.w_out_norm), brows(m.b_h, batch->n_tok));
    if (m.cs_head >= 0) m.calib.tap(c, (size_t)m.cs_head, T);
    m.calib.after_step(c, T);

    /* The lm_head runs only on the rows the sampler asked for; a mid-prompt prefill chunk has
     * n_out zero and skips it. */
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.b_h, batch->n_tok),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                    brows(m.b_logits, batch->n_out));
        if (m.op_cap)
            RAD_ISSUE_N(c, m.op_cap, batch->n_out, brows(m.b_logits, batch->n_out),
                        brows(m.b_logits, batch->n_out));
    }
}

/* What the plugin says before it declares anything: this family's tower merges 4 patches into a
 * row, so an automatic --mm-max-patches rounds to whole merged rows. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    if (MuseTower::present(meta)) out->enc_merge = 4;
    /* DFlash2 reads block_size from dflash_config; its anchor row proposes nothing. */
    if (muse_draft_is_dflash2(meta)) {
        const long long bs = rad_meta_geti(meta, "draft.dflash_config.block_size", 16);
        out->draft_depth = bs > 1 ? (int)(bs - 1) : 15;
        out->draft_depth_fixed = 1;
        return RAD_OK;
    }
    /* Meta DFlash denoises `block_size` positions; the anchor proposes nothing. */
    const long long bs = rad_meta_geti(meta, "draft.block_size", 0);
    if (bs > 1) {
        out->draft_depth = (int)(bs - 1);
        out->draft_depth_fixed = 1;
    }
    return RAD_OK;
}

static const RadChatFormat* chat_format(const RadModelMeta*) {
    return muse_chat_format();
}

}  /* namespace museglimmer */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(museglimmer)
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(museglimmer, "museglimmer", "", "0.1.0",
                "Meta Muse Glimmer 30B dense, bf16 weights and activations, gated attention "
                "alternating sliding-window (rotary) and NoPE full-attention layers, Gemma-2 "
                "style double norms")
RAD_ARCH_CHAT_FORMAT(museglimmer)
#endif
