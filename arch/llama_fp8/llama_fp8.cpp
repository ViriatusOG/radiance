/* llama_fp8 -- the plain LLaMA graph with block-scaled FP8 weights and FP8 activations.
 *
 * ============================== WHAT IT SERVES ==============================
 *
 * `model_type: llama` with `LlamaForCausalLM`, as MiniCPM5-2B reports: 42 layers, hidden 2048, 16
 * query heads over 2 KV heads at head_dim 128, SwiGLU at 6144, RMS norms, full rotary at theta
 * 5e6, an untied 130560-row head, and no biases. Every layer is full attention; unlike
 * arch/qwen35_fp8 there is no gated delta net, layer schedule, output gate or per-head q/k norm.
 *
 * ============================== TWO CHECKPOINT FORMATS, ONE GRAPH ==============================
 *
 * The selection key is the (architecture, quantisation) pair (spec §2.4) and one .so claims one
 * pair, so this source builds twice:
 *
 *   llama_fp8        quant "fp8_e4m3"    the checkpoint holds E4M3 codes and a `weight_scale_inv`
 *                                        plane per linear, as the engine reads them.
 *   llama_dense_fp8  quant ""            the checkpoint is dense bf16, and a recipe quantises its
 *                                        linears at convert time
 *                                        (data/recipes/minicpm5-2b-iq8r.recipe: rotated int8, a
 *                                        scale per (row, 256 of K)) or leaves them bf16.
 *
 * MiniCPM5-2B needs the second, since it is published only in bf16. Both builds declare the same
 * weights and ops, so nothing downstream can tell which one ran.
 *
 * ============================== WHAT IS AND IS NOT FP8 ==============================
 *
 *   E4M3 + a bf16 [N/128][K/128] plane   self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj
 *   BF16, no plane                       every norm and the embedding
 *
 * The lm_head is declared at whatever the model holds: bf16 as llama checkpoints ship it, or block
 * fp8 or int8 where a recipe quantised it (the widest GEMM at half the bytes), and the logits
 * kernel for that encoding is selected. Both vocabulary edges are row-sharded with the vocabulary.
 *
 * ============================== AN INT8 TRUNK ==============================
 *
 * The dense build also serves a recipe that makes the linears rotated int8 (IQ8R:
 * data/recipes/minicpm5-2b-iq8r.recipe). Each is a LinearQ, which reads what the container holds,
 * and declare() learns the format from the first layer's q|k|v before it declares the activations:
 * they then carry int8 codes with one scale per (token, scale group) in place of the E4M3 pair,
 * written by the norms, the gated product and the attention output's quantiser. The DSpark
 * drafter's linears follow their own encodings the same way (rad_block_dspark.h).
 */
#ifndef LLAMA_DENSE_SRC
#define LLAMA_DENSE_SRC 0
#endif
#ifndef LLAMA_NS
#define LLAMA_NS llama_fp8
#endif
#ifndef LLAMA_QUANT
#define LLAMA_QUANT "fp8_e4m3"
#endif

#include <cstring>
#include <vector>

#include <arch/rad_arch.h>
#include <arch/rad_block_attn_q.h>
#include <arch/rad_block_dspark.h>
#include <arch/rad_block_mlp_q.h>

namespace LLAMA_NS {

using namespace rad::arch;

struct Layer {
    AttnQ attn;
    MlpQ  mlp;
};

struct Model {
    Names   nm{""};
    Geom    g{};
    int64_t n_rot = 0;
    int64_t n_main = 0;

    rad_kvgroup kv_full = 0;

    /* The trunk's linears are rotated int8 (IQ8R, or the 128-group int8 before it), read from the
     * first layer's fused q|k|v: the activations then carry int8 code pairs at `trunk_g8` columns
     * a scale, written by the norms, the gated product and the attention output's quantiser. */
    bool    trunk_i8 = false;
    int64_t trunk_g8 = RAD_SCALE_GROUP;

    rad_buf b_x = 0, b_hout = 0, b_logits = 0;
    ActQ  a_h{};
    rad_buf b_qkv = 0;
    rad_buf b_q   = 0;
    ActQ  a_attn{};
    rad_buf b_gate_up = 0;
    ActQ  a_ffn{};

    /* Auxiliary hidden-state taps, declared only when the container carries a drafter that wants
     * them. A DSpark or DFlash drafter reads the residual stream at several intermediate layers
     * and projects the concatenation down (`fc` is [n_embd, n_taps * n_embd]), so the input must
     * be one contiguous row. The trunk's one residual buffer is overwritten by every layer, so
     * each tap's rows are copied into its column slice as they pass. Same mechanism as
     * arch/qwen35_fp8. */
    enum { MAX_TAPS = 8 };
    int      n_taps = 0;
    int      tap_layer[MAX_TAPS] = {};
    rad_buf  b_dfsrc = 0;                 /* [max_tok, n_taps * n_embd] */
    rad_op   op_tap = 0;

    /* The drafter, declared only when the container carries one and this serve speculates. Its
     * layers get absolute indices after the trunk's, so the KV group's layer_slot lookup finds
     * them and its weights land in layer groups of their own. */
    Dspark       ds{};
    Dspark::Wire dsw{};
    rad_kvgroup  kv_draft = 0;
    bool         have_ds = false;

    rad_weight w_tok = 0, w_out_norm = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_final_norm = 0, op_gather = 0, op_logits = 0;

    std::vector<Layer> layers;
};

static Model g_model[MAX_SLOTS];

/* The quantisation descriptor is the selection key but is checked anyway: `--arch` can override
 * the architecture id at convert time, and a plugin handed the wrong checkpoint should say which
 * half disagrees rather than fail at a name map. */
static int check_format(const RadModelMeta* meta) {
    const char* q = meta->quant ? meta->quant : "";
#if LLAMA_DENSE_SRC
    if (*q) {
        fprintf(stderr, "radiance: llama_dense_fp8 serves an unquantised checkpoint and this one "
                        "declares '%s'; llama_fp8 serves it\n", q);
        return RAD_E_UNSUPPORTED;
    }
#else
    if (std::strncmp(q, "fp8", 3) != 0) {
        fprintf(stderr, "radiance: llama_fp8 serves a block-scaled fp8 checkpoint and this one "
                        "declares '%s'; llama_dense_fp8 serves a dense bf16 one\n",
                *q ? q : "(unquantised)");
        return RAD_E_UNSUPPORTED;
    }
#endif
    if (meta->n_expert > 0) {
        fprintf(stderr, "radiance: llama has no routed feed-forward and this checkpoint declares "
                        "%lld experts\n", (long long)meta->n_expert);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* The DSpark drafter. Its Geom is the trunk's with the drafter's layer geometry from `draft.*`
 * substituted; the vocabulary, deployment bounds, rank and wire stay the trunk's.
 *
 * Nothing is defaulted quietly: a `draft.` config that does not describe a drafter this file
 * implements is refused by name, because a silently ignored scale factor or head type would only
 * lower acceptance (spec §17).
 *
 * The confidence head (`AcceptRatePredictor`) is deliberately ignored: it drives adaptive block
 * length, this engine's depth is fixed at declare, and it changes nothing the drafter proposes
 * (rad_block_dspark.h). */
static int declare_dspark(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                          Model& m, int base_layer) {
    Names& nm = m.nm;
    Geom   dg{};
    /* Plain RMS norm: the reference uses Qwen3RMSNorm (`model_type: qwen3`). It matches this
     * MiniCPM5 target, but it is the drafter's convention, set on the drafter's own Geom. */
    RAD_ARCH_TRY(drafter_geom(dg, m.g, meta, "DSpark"));

    Dspark::Config dc{};
    dc.n_taps = m.n_taps;
    dc.rank   = rad_meta_geti(meta, "draft.markov_rank", 0);
    dc.causal = rad_meta_geti(meta, "draft.is_causal", 0) ? 1 : 0;
    if (rad_meta_geti(meta, "draft.use_sliding_window", 0))
        dc.window = rad_meta_geti(meta, "draft.sliding_window", 0);
    /* How many candidates the Markov walk re-ranks. The reference biases all 130560 logits and
     * takes the argmax; this re-ranks the base head's top K, the head's only approximation, which
     * can cost acceptance but not correctness (rad_block_dspark.h). The selector bounds K at 32;
     * the default 16 is 8 KiB of codebook per position, and a checkpoint may ask for more. */
    dc.top_k  = rad_meta_geti(meta, "draft.markov_top_k", 16);
    if (dc.top_k < 1 || dc.top_k > 32) {
        fprintf(stderr, "radiance: draft.markov_top_k is %lld; the Markov walk scores at most 32 "
                        "candidates a position\n", (long long)dc.top_k);
        return RAD_E_UNSUPPORTED;
    }

    /* The block is both the number of noise rows and the number of drafts: every row, the anchor
     * included, predicts the token after it. So the depth is `block_size`, where a DFlash2
     * container's is `block_size - 1` (see RadDrafterDecl::block). A wrong value runs but accepts
     * nothing.
     *
     * A block-diffusion drafter is trained for one block length and has no partial block, so any
     * other depth is refused. The probe reports `draft_depth_fixed`, so the engine normally picks
     * this already. */
    const long long trained = rad_meta_geti(meta, "draft.block_size", 0);
    if (trained < 2) {
        fprintf(stderr, "radiance: this DSpark checkpoint declares no draft.block_size; the "
                        "number of masked positions it denoises is not a number this engine may "
                        "choose.\n");
        return RAD_E_FORMAT;
    }
    if ((long long)ctx->max_spec != trained) {
        fprintf(stderr, "radiance: the DSpark drafter was trained at block_size %lld, which is "
                        "%lld draft tokens, and this serve asks for %d. A block-diffusion drafter "
                        "denoises a fixed number of mask positions; run it at %lld.\n",
                trained, trained, ctx->max_spec, trained);
        return RAD_E_UNSUPPORTED;
    }
    dc.block = trained;

    /* Two variants this implementation does not apply; each changes the proposals, so neither is
     * ignored quietly. */
    const char* mh = rad_meta_gets(meta, "draft.markov_head_type", "vanilla");
    if (dc.rank <= 0 || !mh || std::strcmp(mh, "vanilla") != 0) {
        fprintf(stderr, "radiance: this DSpark checkpoint has markov_rank %lld and "
                        "markov_head_type '%s'. Only the rank-N 'vanilla' bigram head is "
                        "implemented, and it is what the published checkpoints ship.\n",
                (long long)dc.rank, mh ? mh : "(none)");
        return RAD_E_UNSUPPORTED;
    }
    if (dc.rank % 32 || dc.rank > 512) {
        fprintf(stderr, "radiance: markov_rank %lld; the walk that applies the bigram bias takes "
                        "a multiple of 32 up to 512\n", (long long)dc.rank);
        return RAD_E_UNSUPPORTED;
    }

    /* The drafter's own stored format, which decides its activation pairs below. */
    RAD_ARCH_TRY(Dspark::probe_i8(b, nm, "draft", &dc.act_i8, &dc.act_g8));
    dc.fc_mapped = true;
    const uint32_t dqt = dc.act_i8 ? RAD_I8 : RAD_F8E4M3;
    const int64_t  dqg = dc.act_i8 ? dc.act_g8 : (int64_t)RAD_SCALE_GROUP;
    auto dqcols = [dqg](int64_t n) { return (n + dqg - 1) / dqg; };

    /* DSpark's published config sets `sliding_window: null`, so this is normally a full cache. */
    RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, dg, dc.window, base_layer, &m.kv_draft));

    /* Activations. The context pass has one row per committed token over a whole prefill chunk;
     * the query pass one row per draft position, `block` per sequence; the two are sized apart.
     * All are small: the drafter's cost is its weights. */
    const int64_t qd  = dg.n_head * dg.head_dim;
    const int64_t kvd = dg.n_head_kv * dg.head_dim;
    int64_t qr = dg.max_seqs * dc.block;
    if (qr > dg.max_tok) qr = dg.max_tok;
    const int64_t sr   = dg.max_seqs * dc.steps();
    const int64_t spad = ((sr + 15) / 16) * 16;

    Dspark::Wire& dw = m.dsw;
    dw.src  = m.b_dfsrc;
    dw.c    = decl_b(b, nm.f("ds_c"),  dg.act_dtype, {dg.max_tok, dg.n_embd});
    dw.ck   = decl_b(b, nm.f("ds_ck"), dg.act_dtype, {dg.max_tok, kvd});
    dw.cv   = decl_b(b, nm.f("ds_cv"), dg.act_dtype, {dg.max_tok, kvd});
    /* Eight-bit copies for the context pass, at the code type and scale group the drafter's
     * linears read. `srcq` is the largest (20 MiB at a 2048-token chunk, five taps, hidden 2048)
     * and halves the bytes of the context pass's dominant GEMM. */
    if (dc.act_i8 && (qd % dqg || dg.n_embd % dqg || dg.n_ff % dqg)) {
        fprintf(stderr, "radiance: the DSpark drafter's int8 linears keep one scale a %lld of K, "
                        "which does not divide its widths (hidden %lld, attention %lld, "
                        "feed-forward %lld)\n",
                (long long)dqg, (long long)dg.n_embd, (long long)qd, (long long)dg.n_ff);
        return RAD_E_UNSUPPORTED;
    }
    dw.srcq = decl_b(b, nm.f("ds_srcq"), dqt, {dg.max_tok, dc.n_taps * dg.n_embd});
    dw.srcs = decl_b(b, nm.f("ds_srcs"), RAD_F32, {dg.max_tok, dqcols(dc.n_taps * dg.n_embd)});
    dw.cq   = decl_b(b, nm.f("ds_cq"), dqt,     {dg.max_tok, dg.n_embd});
    dw.cs   = decl_b(b, nm.f("ds_cs"), RAD_F32, {dg.max_tok, dqcols(dg.n_embd)});

    dw.x      = decl_b(b, nm.f("ds_x"), dg.act_dtype, {qr, dg.n_embd});
    dw.h.x    = decl_b(b, nm.f("ds_h"), dg.act_dtype, {qr, dg.n_embd});
    dw.h.q    = decl_b(b, nm.f("ds_h.q"), dqt,     {qr, dg.n_embd});
    dw.h.s    = decl_b(b, nm.f("ds_h.s"), RAD_F32, {qr, dqcols(dg.n_embd)});
    dw.q      = decl_b(b, nm.f("ds_q"), dg.act_dtype, {qr, qd});
    dw.k      = decl_b(b, nm.f("ds_k"), dg.act_dtype, {qr, kvd});
    dw.v      = decl_b(b, nm.f("ds_v"), dg.act_dtype, {qr, kvd});
    dw.attn.x = decl_b(b, nm.f("ds_attn"), dg.act_dtype, {qr, qd});
    dw.attn.q = decl_b(b, nm.f("ds_attn.q"), dqt,     {qr, qd});
    dw.attn.s = decl_b(b, nm.f("ds_attn.s"), RAD_F32, {qr, dqcols(qd)});
    dw.gate_up = decl_b(b, nm.f("ds_gate_up"), dg.act_dtype, {qr, 2 * dg.n_ff});
    dw.ffn.x  = decl_b(b, nm.f("ds_ffn"), dg.act_dtype, {qr, dg.n_ff});
    dw.ffn.q  = decl_b(b, nm.f("ds_ffn.q"), dqt,     {qr, dg.n_ff});
    dw.ffn.s  = decl_b(b, nm.f("ds_ffn.s"), RAD_F32, {qr, dqcols(dg.n_ff)});

    dw.sel_h  = decl_b(b, nm.f("ds_selh"), dg.act_dtype, {sr, dg.n_embd});
    dw.dq     = decl_b(b, nm.f("ds_dq"), RAD_I8,  {sr, dg.n_embd});
    dw.ds     = decl_b(b, nm.f("ds_ds"), RAD_F32, {sr, 1});
    /* Group-major, [n_embd/group][ceil(M/16)*16]: quant_act_i8's asum plane, whose first
     * dimension is the K group, not the token. */
    dw.dsum   = decl_b(b, nm.f("ds_dsum"), RAD_F32, {dg.n_embd / 128, spad});
    /* This rank's plane. `head_rows` is the block's own answer to whether the 2-bit head is
     * vocab-sharded; asking it keeps the buffer consistent with the op declared over it. */
    dw.dlog   = decl_b(b, nm.f("ds_dlog"), dg.act_dtype, {sr, Dspark::head_rows(dg)});
    dw.cand   = decl_b(b, nm.f("dspark_cand"), RAD_I32, {sr, dc.top_k});
    dw.unary  = decl_b(b, nm.f("ds_unary"), RAD_F32, {sr, dc.top_k});
    if (Dspark::head_sharded(dg)) {
        dw.dpair = decl_b(b, nm.f("ds_dpair"), RAD_F32, {sr, dc.top_k, 2});
        dw.dgath = decl_b(b, nm.f("ds_dgath"), RAD_F32, {sr, (int64_t)dg.world * dc.top_k, 2});
    }
    /* The proposal buffer (RadDrafterDecl::proposal): a block pass proposes through the
     * drafter's own walk, never through the sampler's token buffer. */
    dw.tokens = decl_b(b, nm.f("dspark_tokens"), RAD_I32, {dg.max_seqs, dc.steps()});

    /* The same lm_head sources as the trunk's head (embedding as the tied alternative), so both
     * project onto one vocabulary. */
    RAD_ARCH_TRY(m.ds.declare(b, nm, dg, dc, base_layer, m.kv_draft, m.w_tok, "draft",
                              nm.ckpt("lm_head.weight"), nm.ckpt("model.embed_tokens.weight"),
                              dw));
    m.have_ds = true;

    /* Block drafter: a context pass mirroring the trunk step, then one pass filling the whole
     * block. `block` is `depth`, not `depth + 1` (this drafter's training convention). `window`
     * must be the one this plugin gated its own attention with, since the scheduler stages the
     * block pass against it. */
    RadDrafterDecl drf{};
    drf.name           = "dspark";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = dc.block;
    drf.proposal       = dw.tokens;
    drf.proposal_pitch = dc.steps();
    drf.window         = dc.window;
    drf.mask_token     = (int32_t)rad_meta_geti(meta, "draft.mask_token_id", -1);
    if (drf.mask_token < 0)
        drf.mask_token = (int32_t)rad_meta_geti(meta, "draft.dflash_config.mask_token_id", -1);
    if (drf.mask_token < 0) {
        fprintf(stderr, "radiance: this DSpark checkpoint declares no mask_token_id, which every "
                        "unfilled block position embeds\n");
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));

    rad_note(b, "DSpark drafter: %lld layers at absolute %d..%d, %lld heads / %lld kv x %lld, "
                "ffn %lld, %s attention with a %lld-token window, %d trunk tap(s); one forward "
                "over %lld noise row(s), the anchor included, proposes %lld tokens",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1,
             (long long)dg.n_head, (long long)dg.n_head_kv, (long long)dg.head_dim,
             (long long)dg.n_ff, dc.causal ? "causal" : "non-causal", (long long)dc.window,
             m.n_taps, (long long)dc.block, (long long)dc.steps());
    rad_note(b, "the rank-%lld Markov bigram head walks `unary + <markov_w1[prev], "
                "markov_w2[cand]>` over the draft head's top %lld, greedily; the confidence head "
                "(adaptive block length) is not applied",
             (long long)dc.rank, (long long)dc.top_k);
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;

    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(check_format(meta));
    /* "bf16" is the activation dtype: everything but the two GEMM operands runs at it. */
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));

    Geom&  g  = m.g;
    Names& nm = m.nm;
    m.n_main  = g.n_layer;

    /* The rotary width is the whole head unless `partial_rotary_factor` says otherwise (Phi-style
     * models set it). A config that states nothing means the full head. */
    {
        double prf = rad_meta_getf(meta, "partial_rotary_factor", 0.0);
        if (!(prf > 0.0)) prf = rad_meta_getf(meta, "rope_parameters.partial_rotary_factor", 0.0);
        m.n_rot = prf > 0.0 ? (int64_t)(g.head_dim * prf) : g.head_dim;
    }
    if (m.n_rot <= 0 || m.n_rot > g.head_dim || (m.n_rot & 1)) {
        fprintf(stderr, "radiance: llama: rotary width %lld is not an even width inside head_dim "
                        "%lld\n", (long long)m.n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(yarn_from(g, meta, "", m.n_rot));

    /* ---- the KV cache. One group: every layer is full attention. */
    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    if (!m.kv_full) return RAD_E_INVAL;
    g.kv_dtype = rad_dtype_name(fd.dtype);

    /* ---- the trunk's stored format, read from the first layer's fused q|k|v before any
     * activation is declared: only an int8 trunk's activations carry an int8 code pair. The probe
     * maps the weight through the layer's own LinearQ, which then does not map it again; the
     * resize before the layer loop keeps the layers sized here. */
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    m.layers.resize((size_t)m.n_main);
    if (m.n_main > 0) {
        const char* pre = "model.layers";
        RAD_ARCH_TRY(m.layers[0].attn.qkv.probe(
            b, nm, g, nm.f("blk.0.attn_qkv"),
            { nm.ckpt("%s.0.self_attn.q_proj.weight", pre),
              nm.ckpt("%s.0.self_attn.k_proj.weight", pre),
              nm.ckpt("%s.0.self_attn.v_proj.weight", pre) }));
        RadEncoding enc{};
        m.trunk_i8 = weight_enc(b, nm.f("blk.0.attn_qkv.weight"), &enc) &&
                     (linear_is_w8h(enc) || linear_is_wnh(enc));
        if (m.trunk_i8) m.trunk_g8 = linear_w8h_group(enc);
    }
    if (m.trunk_i8 && (g.n_embd % m.trunk_g8 || q_dim % m.trunk_g8 || g.n_ff % m.trunk_g8)) {
        fprintf(stderr, "radiance: llama: the int8 trunk keeps one scale a %lld of K, which does "
                        "not divide this rank's widths (hidden %lld, attention %lld, "
                        "feed-forward %lld)\n",
                (long long)m.trunk_g8, (long long)g.n_embd, (long long)q_dim, (long long)g.n_ff);
        return RAD_E_UNSUPPORTED;
    }

    /* ---- activations. An int8 trunk's quantisers sit in their producers where the kernels serve
     * the rotated forms: the norms write had_quant_act_i8g's codes for q|k|v and gate|up, the
     * gated product for the down projection, and the attention output's own quantiser for the
     * out-projection, so no projection declares one and the activation carries no E4M3 pair.
     * Where a form is not declared (the CPU kernels), the producer keeps its E4M3 pair and each
     * projection quantises its own input from the bf16 plane. */
    if (m.trunk_i8) {
        if (rad_op_declared(b, "rmsnorm_had_quant_i8g") &&
            (g.world != 2 || rad_op_declared(b, "ar_rmsnorm_had_quant_i8g")))
            m.a_h.q8_fed = m.a_h.q8_rot = true;
        if (rad_op_declared(b, "gated_had_quant_i8g")) m.a_ffn.q8_fed = m.a_ffn.q8_rot = true;
        if (rad_op_declared(b, "had_quant_act_i8g")) m.a_attn.q8_fed = m.a_attn.q8_rot = true;
    }
    m.b_x      = decl_b(b, nm.f("x"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.a_h.x    = decl_b(b, nm.f("h"),        g.act_dtype, {g.max_tok, g.n_embd});
    RAD_ARCH_TRY(m.a_h.declare_qs(b, nm, g, "h", g.n_embd, 0, m.trunk_i8, !m.a_h.q8_fed,
                                 m.trunk_g8));
    m.b_qkv    = decl_b(b, nm.f("attn_qkv"), g.act_dtype, {g.max_tok, q_dim + 2 * kv_dim});
    m.b_q      = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, q_dim});
    m.a_attn.x = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, q_dim});
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "attn_out", q_dim, 0, m.trunk_i8,
                                    !m.a_attn.q8_fed, m.trunk_g8));
    m.b_gate_up = decl_b(b, nm.f("gate_up"), g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.a_ffn.x   = decl_b(b, nm.f("ffn"),     g.act_dtype, {g.max_tok, g.n_ff});
    RAD_ARCH_TRY(m.a_ffn.declare_qs(b, nm, g, "ffn", g.n_ff, 0, m.trunk_i8, !m.a_ffn.q8_fed,
                                   m.trunk_g8));
    m.b_hout   = decl_b(b, nm.f("h_out"),    g.act_dtype, {g.max_logit_rows, g.n_embd});

    /* ---- the drafter's taps, only when the container carries a drafter that reads them and the
     * deployment speculates: the buffer is one residual stream per tap, 40 MiB at a 2048-token
     * chunk with five taps and hidden 2048.
     *
     * Two key spellings: a SpecForge DFlash config nests the wiring under `dflash_config`, while
     * the published DSpark configs (openbmb/MiniCPM5-2B-DSpark and siblings) put it at the top
     * level. The flat key is read first. */
    if (rad_ctx_draft_head(ctx)) {
        const char* da = rad_meta_gets(meta, "draft.architectures", "");
        if (da && (std::strstr(da, "DSpark") || std::strstr(da, "DFlash"))) {
            const char* ids = rad_meta_gets(meta, "draft.target_layer_ids", nullptr);
            if (!ids) ids = rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr);
            m.n_taps = parse_layer_ids(ids, m.n_main, m.tap_layer, Model::MAX_TAPS);
            if (m.n_taps > 0) {
                m.b_dfsrc = decl_b(b, nm.f("draft_src"), g.act_dtype,
                                   {g.max_tok, (int64_t)m.n_taps * g.n_embd});
                m.op_tap = cast_op(b, g.max_tok, g.n_embd, g.dtype, g.dtype, m.b_x, m.b_dfsrc);
            } else {
                fprintf(stderr, "radiance: llama: the container carries a '%s' drafter but names "
                                "no trunk layer this model has; it is paired with a different "
                                "target\n", da);
                return RAD_E_UNSUPPORTED;
            }
        }
    }

    m.b_logits = decl_b(b, nm.f("logits"),   RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the vocabulary edges, both row-sharded with the vocabulary. A tied head is the
     * alternative source: without `lm_head.weight` the embedding table serves as both, as in small
     * LLaMA derivatives. The head is declared at its codes' dtype (bf16, or E4M3 where a recipe
     * made it block fp8), queried after its name map entry. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"), nm.ckpt("model.norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.embed_tokens.weight")));
    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    /* Always exact: each token's embedding comes from one rank and the others contribute zero, so
     * a lossy wire would only add error. */
    if (g.world > 1)
        m.op_embed_ar = decl_all_reduce_exact(b, g, m.b_x);

    AttnQ::Wire aw{};
    aw.x = m.b_x; aw.h = m.a_h; aw.qkv = m.b_qkv; aw.q = m.b_q; aw.attn = m.a_attn;
    MlpQ::Wire mw{};
    mw.x = m.b_x; mw.h = m.a_h; mw.gate_up = m.b_gate_up; mw.ffn = m.a_ffn;

    for (int64_t l = 0; l < m.n_main; ++l) {
        Layer& lay = m.layers[(size_t)l];
        /* Residual fold and collective placement: rad_linear.h, layer_fold. */
        const LayerFold f = layer_fold(g, l, m.n_main, m.tap_layer, m.n_taps, m.op_embed_ar != 0);

        const char* pre = "model.layers";
        AttnQ::Src as{};
        as.norm = nm.ckpt("%s.%lld.input_layernorm.weight",  pre, (long long)l);
        as.q    = nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l);
        as.k    = nm.ckpt("%s.%lld.self_attn.k_proj.weight", pre, (long long)l);
        as.v    = nm.ckpt("%s.%lld.self_attn.v_proj.weight", pre, (long long)l);
        as.o    = nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l);
        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_full));
        RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv_full, m.n_rot, aw, as,
                                      f.mix_fold, false, f.mix_ar_in, f.mix_ar_out));

        MlpQ::Src ms{};
        ms.norm = nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l);
        ms.gate = nm.ckpt("%s.%lld.mlp.gate_proj.weight",            pre, (long long)l);
        ms.up   = nm.ckpt("%s.%lld.mlp.up_proj.weight",              pre, (long long)l);
        ms.down = nm.ckpt("%s.%lld.mlp.down_proj.weight",            pre, (long long)l);
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms, true, f.mlp_add,
                                     f.mlp_ar_in, f.mlp_ar_out));
    }

    m.op_final_norm = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                        RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(m.w_out_norm)),
                         {m.b_x}, {m.a_h.x});
    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {m.a_h.x}, {m.b_hout});
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n_vocab", g.n_vocab),
                                    RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(m.w_lm_head)),
                     {m.b_hout}, {m.b_logits});

    /* The drafter is declared last, after `op_logits`. Do not move it.
     *
     * Declaration order is the buffer planner's liveness: a transient lives over [first_def,
     * last_use] in op indices, and buffers with disjoint ranges may share bytes
     * (core/build/rad_bufplan.cpp). The drafter's passes run after the whole trunk, so they must
     * be declared after it; declared earlier, the trunk's tail ops would look live after the
     * drafter, and trunk and drafter buffers live in one step would share bytes.
     *
     * The failure is silent: the trunk output stays correct (row 0 of a verify step is computed
     * before the clobber) but every draft is rejected. The op oracle cannot see it, since device
     * and host compute the same wrong answer from a clobbered input. It can also depend on world
     * size, because sharding changes buffer sizes and so the packer's offsets. */
    if (m.n_taps > 0)
        RAD_ARCH_TRY(declare_dspark(b, meta, ctx, m, (int)m.n_main));

    rad_note(b, "rank %d/%d: %lld layers, all full attention; n_head=%lld n_head_kv=%lld "
                "head_dim=%lld rot=%lld n_ff=%lld vocab=[%lld,%lld)",
             g.rank, g.world, (long long)m.n_main, (long long)g.n_head, (long long)g.n_head_kv,
             (long long)g.head_dim, (long long)m.n_rot, (long long)g.n_ff,
             (long long)g.vocab_off, (long long)(g.vocab_off + g.n_vocab));
#if LLAMA_DENSE_SRC
    if (m.trunk_i8)
        rad_note(b, "a dense bf16 checkpoint, every linear quantised by the recipe to rotated "
                    "int8 with one scale a [1 x %lld] of K, read against int8 activations with a "
                    "scale per (token, %lld)", (long long)m.trunk_g8, (long long)m.trunk_g8);
    else
    rad_note(b, "a dense bf16 checkpoint, every linear quantised by the recipe to E4M3 with a "
                "bf16 [N/%d,K/%d] scale plane (scale = block amax / 448, rounded to bf16)",
             RAD_SCALE_GROUP, RAD_SCALE_GROUP);
#else
    rad_note(b, "weights are block-scaled fp8: E4M3 [N,K] with a bf16 [N/%d,K/%d] scale plane, "
                "one stored weight taken by gemm_nt_q as two operands",
             RAD_SCALE_GROUP, RAD_SCALE_GROUP);
#endif
    rad_note(b, "bf16 in the checkpoint: every norm, the embedding and the lm_head, which is "
                "declared at what the container holds");
    if (m.n_taps > 0) {
        char ids[64] = {};
        int  off = 0;
        for (int t = 0; t < m.n_taps && off < (int)sizeof(ids) - 8; ++t)
            off += snprintf(ids + off, sizeof(ids) - (size_t)off, t ? " %d" : "%d",
                            m.tap_layer[t]);
        rad_note(b, "the drafter reads the residual stream after layer(s) %s into a [%lld, %lld] "
                    "buffer; those layers keep their own residual add",
                 ids, (long long)g.max_tok, (long long)((int64_t)m.n_taps * g.n_embd));
    }
    rad_note(b, "q, k and v are one fused [%lld, %lld] projection, rotated in place by `rope` "
                "(q heads, then k heads); no per-head q/k norm and no output gate",
             (long long)(q_dim + 2 * kv_dim), (long long)g.n_embd);
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_slot(c)];
    /* A drafter pass; the sign gives the kind. Positive is the query pass, proposing the whole
     * block at once; negative is the context pass, turning the tokens the trunk just committed
     * into the drafter's own K and V. Zero is a trunk step. */
    if (batch->draft_pass != 0) {
        if (!m.have_ds) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container carries no `draft.` block, or declare ran at max_spec 0");
            return;
        }
        if (batch->draft_pass > 0) m.ds.step(c, batch);
        else                       m.ds.context(c, batch);
        return;
    }

    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
              brows(m.b_x, batch->n_tok));
    if (m.op_embed_ar && !ar_wire_is_exact(m.g, batch->n_tok))
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);

    for (size_t li = 0; li < m.layers.size(); ++li) {
        m.layers[li].attn.step(c, batch);
        m.layers[li].mlp.step(c, batch);
        /* The tap, into its column slice. Runs on prefill and decode alike: the drafter
         * conditions on every committed token's residual stream. */
        for (int t = 0; t < m.n_taps; ++t) {
            if (m.tap_layer[t] != (int)li) continue;
            RAD_ISSUE(c, m.op_tap, brows(m.b_x, batch->n_tok),
                      bcol(m.b_dfsrc, (int64_t)t * m.g.n_embd, m.g.n_embd, batch->n_tok));
        }
    }

    RAD_ISSUE(c, m.op_final_norm,
              brows(m.b_x, batch->n_tok), RAD_W(m.w_out_norm), brows(m.a_h.x, batch->n_tok));
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.a_h.x, batch->n_tok),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                    brows(m.b_logits, batch->n_out));
    }
}

/* The speculation depth this container supports: zero for a plain llama container (no MTP head
 * in this family), or the trained block length of a DSpark drafter added with `--draft-model`.
 *
 * `draft_depth` is `block_size`, not `block_size - 1` as in arch/qwen35_fp8: DSpark's anchor row
 * predicts too. `draft_depth_fixed` forbids any other depth, since the drafter was trained for
 * exactly one block length. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->draft_depth = 0;
#if LLAMA_DENSE_SRC
    /* Chained launches pay on an IQ8R trunk with and without its drafter: libr4d's
     * rad_kernel_chain holds the IQ8R decode rows. MiniCPM5-2B on one R9700, ms a decode step
     * unchained / chained: 5.57 / 5.17 at one sequence and 6.09 / 5.71 at eight without a drafter,
     * and with DSpark 7.32 / 6.87 at one and 8.15 / 7.92 at four. A block-fp8 trunk gains less
     * and loses nowhere: 5.45 / 5.40 and 5.92 / 5.89 without, 7.14 / 7.12 and 8.17 / 8.16 with. */
    out->chain_pays = RAD_CHAIN_ALWAYS;
#endif
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    if (da && (std::strstr(da, "DSpark") || std::strstr(da, "DFlash"))) {
        const long long bs = rad_meta_geti(meta, "draft.block_size", 0);
        if (bs > 1) {
            out->draft_depth       = (int)bs;
            out->draft_depth_fixed = 1;
        }
    }
    return RAD_OK;
}

}  /* namespace LLAMA_NS */

#ifndef RAD_ARCH_NO_EXPORTS
/* One more level of expansion so `#NS` sees the expanded namespace. */
#define LLAMA_PROBE_X(NS)        RAD_ARCH_PROBE(NS)
#define LLAMA_PLUGIN_X(NS, ...)  RAD_ARCH_PLUGIN(NS, __VA_ARGS__)
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
LLAMA_PROBE_X(LLAMA_NS)
#if LLAMA_DENSE_SRC
LLAMA_PLUGIN_X(LLAMA_NS, "llama", LLAMA_QUANT, "0.1.0",
               "LLaMA-shaped decoders (MiniCPM5, Llama 3, and every derivative that reports "
               "model_type llama) from a dense bf16 checkpoint, made block-scaled fp8 by a recipe "
               "at convert time and served fp8 weights / fp8 activations")
#else
LLAMA_PLUGIN_X(LLAMA_NS, "llama", LLAMA_QUANT, "0.1.0",
               "LLaMA-shaped decoders (MiniCPM5, Llama 3, and every derivative that reports "
               "model_type llama), block-scaled fp8 weights and fp8 activations")
#endif
#endif
