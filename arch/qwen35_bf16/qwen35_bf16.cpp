/* qwen35_bf16 -- Qwen3.5 / 3.6 / 3.8 dense, bf16 weights and bf16 activations.
 *
 * Qwen3.5, 3.6 and 3.8 dense models report the architecture id `qwen35`; `qwen35moe` is the same
 * graph with a routed FFN. Dimensions come from metadata, so this file serves every size (0.8B to
 * 27B). docs/ARCHITECTURES.md has the geometry and its plausible-but-wrong traps.
 *
 * One plugin implements one architecture id for one quantisation scheme (spec §2.4); a quantised
 * variant is a separate plugin composing the same blocks against other ops (e.g.
 * rmsnorm_had_quant_i8, gemm_nt_q), so declare() below has no branch on quantisation.
 *
 * ============================== WHAT IT SERVES ==============================
 *
 * The unquantised release (quantisation descriptor ""), every projection a bf16 GEMM. The
 * block-scaled fp8 release is qwen35_fp8's. The lm_head is declared at whatever the model holds,
 * so a recipe-made block-fp8 head uses the fp8 logits kernel.
 *
 * ============================== THE SHAPE, AT 27B ==============================
 *
 *   64 layers, full_attention_interval 4 -> layers 3,7,...,63 are full attention (16 of them),
 *                                           the other 48 are gated delta net
 *   hidden 5120   ffn 17408   vocab 248320   (lm_head not tied)
 *   24 heads / 4 kv heads / head_dim 256  -> GQA 6; head_dim is read, not derived
 *                                            (n_embd / n_head would be 213)
 *   attn_output_gate -> q_proj is [n_embd, 2 * n_head * head_dim], query and gate interleaved
 *   partial_rotary_factor 0.25 -> MROPE over the first 64 of 256 head dims, theta 1e7
 *   linear: 16 key heads / 48 value heads of 128, conv width 4, fp32 recurrent state
 *
 * ============================== WHAT THIS PLUGIN DOES NOT DECLARE ==============================
 *
 * The MTP / NextN head (one extra decoder block plus fc, enorm, hnorm) is not declared here; it
 * is not needed for text generation. It sits beside the trunk under its own `mtp.` prefix, so
 * `nextn_predict_layers` is not subtracted from the layer count (see declare()); running it as a
 * trunk layer would be a different model.
 *
 * The vision tower (27 blocks, hidden 1152, 16 heads of 72, patch 16, out 5120) is declared
 * through arch/common/rad_block_vit.h, only when the container carries one and the deployment
 * asked for media.
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_attn_gated.h>
#include <arch/rad_block_gdn.h>
#include <arch/rad_block_mlp.h>
#include <arch/rad_block_vit.h>
#include <arch/rad_calib_full.h>

#include <vector>

namespace qwen35_bf16 {

using namespace rad::arch;

/* One trunk layer, of one of two kinds; both small structs are held rather than a union. `full`
 * is the only branch in step(), and it is fixed per layer by the model. */
struct Layer {
    int           full = 0;
    AttnGatedBF16 attn;
    GdnBF16       gdn;
    MlpBF16       mlp;
    /* The layer's calibration inputs (rad_calib_full.h): the attention or delta net's, its output
     * projection's, the feed-forward's and down_proj's; -1 where the run does not record it. */
    int cs_in = -1, cs_o = -1, cs_mlp = -1, cs_down = -1;
};

struct Model {
    Names           nm{""};
    Geom            g{};
    GdnBF16::Config gcfg{};
    int64_t         n_rot  = 0;
    int64_t         n_main = 0;    /* trunk layers, excluding the MTP block */

    rad_kvgroup kv_full = 0, kv_state = 0, kv_conv = 0;

    /* the residual stream and the two vocabulary edges */
    rad_buf b_x = 0, b_h = 0, b_hout = 0, b_logits = 0;
    /* full attention */
    rad_buf b_qg = 0, b_k = 0, b_v = 0, b_q = 0, b_gate = 0, b_attn = 0;
    /* gated delta net */
    rad_buf b_in = 0, b_z = 0, b_gq = 0, b_gk = 0, b_gv = 0;
    rad_buf b_gdec = 0, b_beta = 0, b_kkt = 0, b_go = 0;
    /* feed-forward */
    rad_buf b_gate_up = 0, b_ffn = 0;

    rad_weight w_tok = 0, w_out_norm = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_final_norm = 0, op_gather = 0, op_logits = 0;
    /* Set by a plugin built over this one whose head answers a request from its words
     * (RadBatch::plugin_words): a `fill` of the logits rows that plugin declared. A step every
     * sampled row of which carries words then skips the lm_head and leaves those zero rows for the
     * sampler, whose token the request discards. Zero, the lm_head runs as it always has, and the
     * graph this plugin declares is unchanged. */
    rad_op op_logits_fill = 0;

    /* ---- media (arch/common/rad_block_vit.h): the vision tower and the scatter of its rows,
     * declared only when the container carries a tower and the deployment asked for media. */
    bool        media = false;
    VisionTower vit{};
    MediaRows   mrows{};

    std::vector<Layer> layers;

    /* --calib-dir: the whole Hessian of every linear's input over a --calib-corpus run. */
    CalibFull calib;
    int       cs_head = -1;     /* the lm_head's input, recorded with the last layer */
};

/* Declare runs once per rank, in one process, on the rank's own thread (spec §1, §9), so the
 * model state is indexed by rank and read back in step() with rad_rank(). */
static Model g_model[MAX_RANKS];

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));

    /* A routed model is a different graph, served by the qwen35moe plugin. */
    if (meta->n_expert > 0) {
        fprintf(stderr, "radiance: qwen35_bf16 refuses a checkpoint with %lld experts; that is "
                        "qwen35moe\n", (long long)meta->n_expert);
        return RAD_E_UNSUPPORTED;
    }

    Geom&  g  = m.g;
    Names& nm = m.nm;

    /* ---- geometry RadModelMeta does not name, from the container's free-form header. Both
     * spellings are accepted: HuggingFace keys (from safetensors) and llama.cpp's (from GGUF). */
    bool have_interval = false;
    const int64_t interval = meta_int2(meta, "qwen35.full_attention_interval",
                                       "full_attention_interval", 4, &have_interval);
    const int64_t nextn    = meta_int2(meta, "qwen35.nextn_predict_layers",
                                       "mtp_num_hidden_layers", 0);
    if (interval <= 0 || nextn < 0 || nextn >= g.n_layer) {
        fprintf(stderr, "radiance: qwen35: full_attention_interval %lld / nextn %lld out of range "
                        "for %lld layers\n",
                (long long)interval, (long long)nextn, (long long)g.n_layer);
        return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(check_layer_schedule("qwen35", meta_text_gets(meta, "layer_types", nullptr),
                                      g.n_layer, interval, {"full_attention"}));
    /* Every declared layer is a trunk layer: `mtp_num_hidden_layers` is not part of
     * `num_hidden_layers` in this family (the MTP block ships separately under `mtp.`).
     * Subtracting it would drop full-attention layer 63 and silently run a trunk one layer short.
     * `nextn` is used only for the note below. */
    m.n_main = g.n_layer;

    /* Qwen3.5's norms are GemmaRMSNorm, `x_hat * (1 + w)` (vLLM imports GemmaRMSNorm as
     * Qwen3_5RMSNorm). This covers input_layernorm, post_attention_layernorm, the final norm and
     * the q/k head norms, but not the delta net's ssm_norm, a plain RMSNormGated (gains ~0.88).
     * The checkpoint confirms it: layer norm weights of 0.07..0.29 and partly negative q/k norm
     * weights only make sense as `1 + w`. The final norm (0.68..0.96) is ambiguous and follows the
     * reference implementation. */
    g.wadd    = 1.0f;
    g.wadd_qk = 1.0f;
    const float final_wadd = 1.0f;

    /* The rotary width: partial_rotary_factor 0.25 over head_dim 256 is 64. A container that
     * omits it is refused rather than defaulted to a full rotation, which would give fluent wrong
     * text (spec §17). */
    m.n_rot = meta_int2(meta, "qwen35.rope.dimension_count", "rope_dimension_count", 0);
    if (m.n_rot <= 0) {
        /* Both spellings occur in this family: Qwen3.8-27B-FP8 states `partial_rotary_factor`
         * at the top level and inside `rope_parameters`, Qwen3.5-0.8B only inside
         * `rope_parameters` (rad-convert flattens nested config with a dot). */
        double prf = rad_meta_getf(meta, "partial_rotary_factor", 0.0);
        if (!(prf > 0.0)) prf = rad_meta_getf(meta, "rope_parameters.partial_rotary_factor", 0.0);
        if (prf > 0.0) m.n_rot = (int64_t)(g.head_dim * prf);
    }
    if (m.n_rot <= 0 || m.n_rot > g.head_dim || (m.n_rot & 1)) {
        fprintf(stderr, "radiance: qwen35: no usable rotary width -- give the container "
                        "qwen35.rope.dimension_count or partial_rotary_factor (head_dim %lld)\n",
                (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    RAD_ARCH_TRY(yarn_from(g, meta, "", m.n_rot));

    /* The linear-attention geometry. llama.cpp maps the GGUF ssm keys onto these:
     *   ssm.state_size -> the key AND value head width, ssm.group_count -> key heads,
     *   ssm.time_step_rank -> value heads, ssm.inner_size -> value heads * value width. */
    GdnBF16::Config& gc = m.gcfg;
    const int64_t n_head_k_all = meta_int2(meta, "qwen35.ssm.group_count",
                                           "linear_num_key_heads", 0);
    const int64_t n_head_v_all = meta_int2(meta, "qwen35.ssm.time_step_rank",
                                           "linear_num_value_heads", 0);
    gc.head_k     = meta_int2(meta, "qwen35.ssm.state_size", "linear_key_head_dim", 0);
    gc.head_v     = meta_int2(meta, "qwen35.ssm.value_head_dim", "linear_value_head_dim", 0);
    gc.conv_width = meta_int2(meta, "qwen35.ssm.conv_kernel", "linear_conv_kernel_dim", 0);
    if (gc.head_v <= 0) {
        const int64_t inner = meta_int2(meta, "qwen35.ssm.inner_size", "linear_inner_size", 0);
        if (inner > 0 && n_head_v_all > 0) gc.head_v = inner / n_head_v_all;
    }
    if (n_head_k_all <= 0 || n_head_v_all <= 0 || gc.head_k <= 0 || gc.head_v <= 0 ||
        gc.conv_width <= 0) {
        fprintf(stderr, "radiance: qwen35: the container does not describe the linear layers "
                        "(key heads %lld, value heads %lld, key width %lld, value width %lld, "
                        "conv width %lld)\n",
                (long long)n_head_k_all, (long long)n_head_v_all, (long long)gc.head_k,
                (long long)gc.head_v, (long long)gc.conv_width);
        return RAD_E_INVAL;
    }
    /* The chunked scan's tile length, not a model parameter: libr4d's
     * gdn_chunk_scan_k128_v128_c64_bf16, and also the linear-state checkpoint interval
     * (spec §7.3). Note: gdn_conv_prep takes it while gdn_chunk_scan derives it from the kernel,
     * and nothing checks that the two agree. */
    gc.chunk = 64;

    int s;
    if ((s = divide_or_fail(n_head_k_all, g.world, "linear_num_key_heads",   &gc.n_head_k)) < 0)
        return s;
    if ((s = divide_or_fail(n_head_v_all, g.world, "linear_num_value_heads", &gc.n_head_v)) < 0)
        return s;

    const int64_t q_dim    = g.q_dim();
    const int64_t kv_dim   = g.kv_dim();
    const int64_t v_dim    = gc.v_dim();
    const int64_t conv_dim = gc.conv_dim();

    /* ---- three KV groups (spec §7.2). Every layer is bound below to the group it owns state
     * in; a group with no bound layers is a declare error, since the bound layer count is how
     * many state instances a sequence holds.
     *
     * The full-attention group's block size is not declared: it comes from the resolved attention
     * kernel and is read back with rad_kv_block_size() inside the block. */
    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    g.kv_dtype = rad_dtype_name(fd.dtype);

    /* The recurrent state is fp32, as `mamba_ssm_dtype: float32` asks; an f16 state changes the
     * generated text (qwen35_fp8.cpp). It costs 48 x 128 x 128 x 4 = 3 MiB per layer, 144 MiB
     * across the 48 linear layers per sequence, the largest item in the memory plan. */
    RadKVGroupDecl sd{};
    sd.kind         = RAD_KV_LINEAR;
    sd.dtype        = RAD_F32;
    sd.n_head_kv    = gc.n_head_v;
    sd.state_dim[0] = gc.head_v;
    sd.state_dim[1] = gc.head_k;
    m.kv_state = rad_decl_kv_group(b, nm.f("kv_gdn_state"), &sd);

    /* The conv window: conv_width-1 + n_spec entries of the whole convolved row, counted in heads
     * so the channel split never straddles a q/k/v boundary at any tensor-parallel width. */
    RadKVGroupDecl cd{};
    cd.kind       = RAD_KV_CONV;
    cd.dtype      = RAD_BF16;
    cd.n_head_kv  = 2 * gc.n_head_k + gc.n_head_v;
    cd.head_dim   = gc.head_k;
    cd.conv_width = gc.conv_width;
    m.kv_conv = rad_decl_kv_group(b, nm.f("kv_gdn_conv"), &cd);

    /* ---- activation buffers, sized at max_tok (the worst step) and shared by every layer of a
     * kind; liveness analysis over the declared read and write sets fits them in one arena
     * (spec §3.1). */
    m.b_x       = decl_b(b, nm.f("x"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.b_h       = decl_b(b, nm.f("h"),        g.act_dtype, {g.max_tok, g.n_embd});

    /* The query-and-gate projection is rank 3, [tokens, heads, 2*head_dim], so the per-head
     * interleaving is addressable: query and gate are the first and second head_dim columns of the
     * last dimension, i.e. ordinary column slices. Flattened, no operand could name them. */
    m.b_qg      = decl_b(b, nm.f("attn_qg"),  g.act_dtype, {g.max_tok, g.n_head, 2 * g.head_dim});
    m.b_k       = decl_b(b, nm.f("attn_k"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_v       = decl_b(b, nm.f("attn_v"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_q       = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, q_dim});
    m.b_gate    = decl_b(b, nm.f("attn_gate"),g.act_dtype, {g.max_tok, q_dim});
    m.b_attn    = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, q_dim});

    m.b_in      = decl_b(b, nm.f("gdn_in"),   g.act_dtype, {g.max_tok, gc.in_dim()});
    m.b_z       = decl_b(b, nm.f("gdn_z"),    g.act_dtype, {g.max_tok, v_dim});
    m.b_gq      = decl_b(b, nm.f("gdn_q"),    g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    m.b_gk      = decl_b(b, nm.f("gdn_k"),    g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    m.b_gv      = decl_b(b, nm.f("gdn_v"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});
    /* Decay and beta are f32: the decay is a cumulative sum whose intra-chunk span reaches
     * hundreds of nats on real prefill inputs, and a bf16 running sum would lose the low bits that
     * every decay factor exponentiates. */
    m.b_gdec    = decl_b(b, nm.f("gdn_g"),    RAD_F32,     {g.max_tok, gc.n_head_v});
    m.b_beta    = decl_b(b, nm.f("gdn_beta"), RAD_F32,     {g.max_tok, gc.n_head_v});
    /* bf16, not f32: gdn_kkt_solve writes this plane as bf16 and gdn_chunk_scan reads it as
     * bf16. */
    m.b_kkt     = decl_b(b, nm.f("gdn_kkt"),  RAD_BF16,    {g.max_tok, gc.n_head_v, gc.chunk});
    m.b_go      = decl_b(b, nm.f("gdn_o"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});

    m.b_gate_up = decl_b(b, nm.f("gate_up"),  g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.b_ffn     = decl_b(b, nm.f("ffn"),      g.act_dtype, {g.max_tok, g.n_ff});
    /* The lm_head runs over the sampled rows only: `max_logit_rows` is one row per sequence, or
     * per verified draft position. At 248320 vocab that is 63 MiB of logits instead of 8 GiB.
     * gather_rows selects the hidden states into the narrow `b_hout` (n_embd wide) before the
     * GEMM. */
    m.b_hout    = decl_b(b, nm.f("h_out"),    g.act_dtype, {g.max_logit_rows, g.n_embd});
    m.b_logits  = decl_b(b, nm.f("logits"),   RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- vocabulary edges, both row-sharded. The lm_head is sharded because the sampler runs a
     * distributed top-k and full logits are never gathered (spec §9, §13); replicating the
     * embedding would cost 248320 x 5120 bf16 (2.43 GiB) per rank. `embed_lookup` takes a
     * `vocab_offset`: a rank gathers the ids in its slice and writes zero rows for the rest, and
     * the all_reduce below sums the ranks into x. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"),
                          nm.ckpt("model.language_model.norm.weight")));
    /* The 27B does not tie its lm_head (`tie_word_embeddings: false`); the second source serves
     * the small family members that do. The first source present wins. */
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.language_model.embed_tokens.weight")));

    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    /* Row-sharded with the vocabulary: g.n_vocab is this rank's share (rad_arch.h). Declared at
     * its codes' dtype (bf16 as shipped, E4M3 where a recipe made it block fp8), which selects the
     * logits kernel. */
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    /* The encoder's rows overwrite the lookup before its reduction (see MediaRows). */
    m.media = VisionTower::present(meta) && ctx->max_enc_patches > 0;
    if (m.media) RAD_ARCH_TRY(m.mrows.declare(b, g.max_tok, g.n_embd, m.b_x));
    /* Always exact, not g.wire_exact: this sums disjoint rows (each token from one rank, zeros
     * from the others), so a lossy wire would only add error. */
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                           {m.b_x}, {m.b_x});

    /* ---- the layers. */
    AttnGatedBF16::Wire aw{};
    aw.x = m.b_x; aw.h = m.b_h; aw.qg = m.b_qg; aw.k = m.b_k; aw.v = m.b_v;
    aw.q = m.b_q; aw.gate = m.b_gate; aw.attn = m.b_attn;

    GdnBF16::Wire gw{};
    gw.x = m.b_x; gw.h = m.b_h; gw.in = m.b_in; gw.z = m.b_z;
    gw.q = m.b_gq; gw.k = m.b_gk; gw.v = m.b_gv;
    gw.gdec = m.b_gdec; gw.beta = m.b_beta; gw.kkt = m.b_kkt; gw.o = m.b_go;

    MlpBF16::Wire mw{};
    mw.x = m.b_x; mw.h = m.b_h; mw.gate_up = m.b_gate_up; mw.ffn = m.b_ffn;

    /* The calibration tap: the widest split input is down_proj's (the feed-forward's width), and
     * no input is wider. */
    RAD_ARCH_TRY(m.calib.begin(b, nm, g, ctx, m.n_main, g.n_ff * g.world,
                               std::max({ g.n_embd, q_dim * g.world, v_dim * g.world, g.n_ff * g.world })));

    m.layers.resize((size_t)m.n_main);
    for (int64_t l = 0; l < m.n_main; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.full = is_full_attention(l, interval) ? 1 : 0;

        const char* pre = "model.language_model.layers";
        if (lay.full) {
            AttnGatedBF16::Src as{};
            as.norm   = nm.ckpt("%s.%lld.input_layernorm.weight",     pre, (long long)l);
            as.qg     = nm.ckpt("%s.%lld.self_attn.q_proj.weight",    pre, (long long)l);
            as.k      = nm.ckpt("%s.%lld.self_attn.k_proj.weight",    pre, (long long)l);
            as.v      = nm.ckpt("%s.%lld.self_attn.v_proj.weight",    pre, (long long)l);
            as.q_norm = nm.ckpt("%s.%lld.self_attn.q_norm.weight",    pre, (long long)l);
            as.k_norm = nm.ckpt("%s.%lld.self_attn.k_norm.weight",    pre, (long long)l);
            as.o      = nm.ckpt("%s.%lld.self_attn.o_proj.weight",    pre, (long long)l);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_full));
            RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv_full, m.n_rot, aw, as));
        } else {
            GdnBF16::Src gs{};
            gs.norm     = nm.ckpt("%s.%lld.input_layernorm.weight",           pre, (long long)l);
            gs.in_qkv   = nm.ckpt("%s.%lld.linear_attn.in_proj_qkv.weight",   pre, (long long)l);
            gs.in_a     = nm.ckpt("%s.%lld.linear_attn.in_proj_a.weight",     pre, (long long)l);
            gs.in_b     = nm.ckpt("%s.%lld.linear_attn.in_proj_b.weight",     pre, (long long)l);
            gs.in_z     = nm.ckpt("%s.%lld.linear_attn.in_proj_z.weight",     pre, (long long)l);
            gs.conv1d   = nm.ckpt("%s.%lld.linear_attn.conv1d.weight",        pre, (long long)l);
            gs.a_log    = nm.ckpt("%s.%lld.linear_attn.A_log",                pre, (long long)l);
            gs.dt_bias  = nm.ckpt("%s.%lld.linear_attn.dt_bias",              pre, (long long)l);
            gs.out_norm = nm.ckpt("%s.%lld.linear_attn.norm.weight",          pre, (long long)l);
            gs.out      = nm.ckpt("%s.%lld.linear_attn.out_proj.weight",      pre, (long long)l);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_state));
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_conv));
            RAD_ARCH_TRY(lay.gdn.declare(b, nm, g, gc, (int)l, m.kv_state, m.kv_conv, gw, gs));
        }
        if (m.calib.records(l)) {
            const std::string L = std::string(pre) + "." + std::to_string(l);
            lay.cs_in = (int)m.calib.sites.size();
            RAD_ARCH_TRY(m.calib.site(b, g, l, L + ".input",
                                      lay.full ? L + ".self_attn.q_proj," + L + ".self_attn.k_proj," + L + ".self_attn.v_proj"
                                               : L + ".linear_attn.in_proj_qkv," + L + ".linear_attn.in_proj_z," +
                                                 L + ".linear_attn.in_proj_a," + L + ".linear_attn.in_proj_b",
                                      m.b_h, g.n_embd, g.n_embd));
            lay.cs_o = (int)m.calib.sites.size();
            if (lay.full)
                RAD_ARCH_TRY(m.calib.site(b, g, l, L + ".self_attn.o_proj.input", L + ".self_attn.o_proj",
                                          m.b_attn, q_dim, q_dim * g.world));
            else
                RAD_ARCH_TRY(m.calib.site(b, g, l, L + ".linear_attn.out_proj.input", L + ".linear_attn.out_proj",
                                          m.b_go, v_dim, v_dim * g.world));
        }

        MlpBF16::Src ms{};
        ms.norm = nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l);
        ms.gate = nm.ckpt("%s.%lld.mlp.gate_proj.weight",            pre, (long long)l);
        ms.up   = nm.ckpt("%s.%lld.mlp.up_proj.weight",              pre, (long long)l);
        ms.down = nm.ckpt("%s.%lld.mlp.down_proj.weight",            pre, (long long)l);
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms));
        if (m.calib.records(l)) {
            const std::string L = std::string(pre) + "." + std::to_string(l);
            lay.cs_mlp = (int)m.calib.sites.size();
            RAD_ARCH_TRY(m.calib.site(b, g, l, L + ".post_attention.input",
                                      L + ".mlp.gate_proj," + L + ".mlp.up_proj", m.b_h, g.n_embd, g.n_embd));
            lay.cs_down = (int)m.calib.sites.size();
            RAD_ARCH_TRY(m.calib.site(b, g, l, L + ".mlp.down_proj.input", L + ".mlp.down_proj",
                                      m.b_ffn, g.n_ff, g.n_ff * g.world));
        }
    }

    m.op_final_norm = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_F64("wadd", final_wadd), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(m.w_out_norm)),
                         {m.b_x}, {m.b_h});
    /* The lm_head's input, with the run that records the last layer. */
    if (m.calib.on && m.calib.hi == m.n_main) {
        m.cs_head = (int)m.calib.sites.size();
        RAD_ARCH_TRY(m.calib.site(b, g, -1, "lm_head.input", "lm_head", m.b_h, g.n_embd, g.n_embd));
    }

    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {m.b_h}, {m.b_hout});

    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n_vocab", g.n_vocab),
                                    RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(m.w_lm_head)),
                     {m.b_hout}, {m.b_logits});

    /* ---- notes for --debug-graph. */
    rad_note(b, "rank %d/%d: %lld trunk layers, %lld full attention + %lld gated delta net, "
                "n_head=%lld n_head_kv=%lld head_dim=%lld n_ff=%lld vocab=[%lld,%lld)",
             g.rank, g.world, (long long)m.n_main,
             (long long)(m.n_main / interval), (long long)(m.n_main - m.n_main / interval),
             (long long)g.n_head, (long long)g.n_head_kv, (long long)g.head_dim,
             (long long)g.n_ff, (long long)g.vocab_off, (long long)(g.vocab_off + g.n_vocab));
    rad_note(b, "gdn: %lld key heads x %lld, %lld value heads x %lld, conv width %lld, chunk %lld; "
                "conv row %lld channels, projection row %lld",
             (long long)gc.n_head_k, (long long)gc.head_k, (long long)gc.n_head_v,
             (long long)gc.head_v, (long long)gc.conv_width, (long long)gc.chunk,
             (long long)conv_dim, (long long)gc.in_dim());
    rad_note(b, "ssm_a_log holds A_log itself: the kernels form -exp(A_log) * softplus(a + dt), "
                "so a GGUF's ssm_a = -exp(A_log) must be converted back");
    rad_note(b, "rope: %s over %lld of %lld head dims%s%s", g.rope_mode, (long long)m.n_rot,
             (long long)g.head_dim, g.rope_mc ? ", sections " : "", g.rope_sections);
    if (!have_interval)
        rad_note(b, "the container did not state full_attention_interval; assuming %lld, which "
                    "makes layers 3,7,... full attention", (long long)interval);
    rad_note(b, "logits for at most %lld sampled rows a step (of %lld tokens): gather_rows "
                "selects RadBatch::out_ids before the lm_head", (long long)g.max_logit_rows,
             (long long)g.max_tok);
    if (nextn > 0)
        rad_note(b, "the %lld MTP/NextN layer(s) under `mtp.` are not declared: this plugin "
                    "serves the trunk only", (long long)nextn);

    /* The vision tower, last, so its activations share the arena with every buffer above. */
    if (m.media) RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer + 64));
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];

    /* An encoder pass runs only the vision tower (RadBatch::enc). */
    if (batch->enc) {
        m.vit.step(c, batch);
        return;
    }

    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
              brows(m.b_x, batch->n_tok));
    m.mrows.step(c, batch, m.b_x, batch->n_tok);
    /* The vocab shard's reduction; declared only when world > 1, so tp1 is unchanged. */
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);

    const int64_t T = batch->n_tok;
    for (const Layer& l : m.layers) {
        /* Under calibration a block's input norm is issued once ahead of the block, so its output
         * is read before the block overwrites `h`; the block computes it again, the same op on
         * the same bytes. */
        if (l.cs_in >= 0) {
            if (l.full) RAD_ISSUE(c, l.attn.op_norm, brows(m.b_x, T), RAD_W(l.attn.w_norm), brows(m.b_h, T));
            else        RAD_ISSUE(c, l.gdn.op_norm,  brows(m.b_x, T), RAD_W(l.gdn.w_norm),  brows(m.b_h, T));
            m.calib.tap(c, (size_t)l.cs_in, T);
        }
        if (l.full) l.attn.step(c, batch);
        else        l.gdn.step(c, batch);
        if (l.cs_o >= 0) m.calib.tap(c, (size_t)l.cs_o, T);
        if (l.cs_mlp >= 0) {
            RAD_ISSUE(c, l.mlp.op_norm, brows(m.b_x, T), RAD_W(l.mlp.w_norm), brows(m.b_h, T));
            m.calib.tap(c, (size_t)l.cs_mlp, T);
        }
        l.mlp.step(c, batch);
        if (l.cs_down >= 0) m.calib.tap(c, (size_t)l.cs_down, T);
    }

    RAD_ISSUE(c, m.op_final_norm,
              brows(m.b_x, batch->n_tok), RAD_W(m.w_out_norm), brows(m.b_h, batch->n_tok));
    if (m.cs_head >= 0) m.calib.tap(c, (size_t)m.cs_head, T);
    m.calib.after_step(c, T);

    /* The lm_head runs only on the rows the sampler asked for; a mid-prompt prefill chunk has
     * n_out zero and skips it. */
    if (batch->n_out > 0) {
        if (m.op_logits_fill && plugin_words_cover_rows(batch)) {
            RAD_ISSUE_N(c, m.op_logits_fill, batch->n_out, brows(m.b_logits, batch->n_out));
            return;
        }
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.b_h, batch->n_tok),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                    brows(m.b_logits, batch->n_out));
    }
}

}  /* namespace qwen35_bf16 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PLUGIN(qwen35_bf16, "qwen35", "", "0.1.0",
                "Qwen3.5 / 3.6 / 3.8 dense (0.8B..27B), bf16 weights and activations, gated "
                "attention over paged KV every fourth layer and gated delta net between")
#endif
