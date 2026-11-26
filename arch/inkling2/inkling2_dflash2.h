/* inkling2_dflash2.h -- a DFlash2 drafter (rad_block_dflash2.h) on Inkling 2's trunk: incoai's
 * Inkling-2-Mini-DFlash2, a `DFlash2DraftModel` read by `rad-convert --draft-model`, told apart by
 * draft.architectures (dflash2_present).
 *
 * The block is the one Qwen3.8 and Muse Glimmer serve. What this wiring adds is how the drafter was
 * trained against Inkling (the SGLang integration shipped with the checkpoint,
 * sglang_inkling_dflash2.py, over SGLang's Inkling 2 model):
 *
 *   taps        dflash_config.target_layer_ids (5, 23, 41, 48) are layer applications of the
 *               48-application loop: the running partial block entering application i (SGLang
 *               captures `state.partial` before the visit runs), and 48 the final norm's output.
 *               The trunk copies them (InkDflash2::tap, at the start of each tapped application
 *               and after the final norm), a token's streams side by side in `src` in the order the
 *               ids list them, which is fc's column order.
 *   embedding   the target's input embedding is embed_norm(embed(id)) (Inkling 2's
 *               get_input_embeddings, which the integration's drafter reads), so the query block's
 *               rows are normed after the lookup (Dflash2::op_post_embed) with the trunk's norm.
 *   head        the target's own head (int8, vocabulary-sharded above one rank), read by logits_gemm
 *               (Dflash2::w_thead): the drafter has no head of its own, and the trunk's is already
 *               a narrow read. No rerank, since its scores are that head's.
 *   vocabulary  candidates come from the real vocabulary (unpadded_vocab_size); the head's padded
 *               rows are filled with -1e30 before the top-K (Config::vocab_real), as the trunk fills
 *               its own logits.
 *   softcap     dflash_config.final_logit_softcapping under selector_softcap_mode
 *               "after_correction": the cap is taken over the unary logit plus the edge score, which
 *               the greedy walk's argmax does not see (Config::cap_after).
 *
 * Under tensor parallelism the block shards as on Qwen3.8 (its SHARDING section): q, k, v and
 * gate_up by row, o and down by column with an all-reduce, fc, the convolutions and the codebooks
 * replicated. The drafter's 8 KV heads divide over two ranks.
 */
#ifndef INKLING2_DFLASH2_H
#define INKLING2_DFLASH2_H

#include <arch/rad_arch.h>
#include <arch/rad_block_dflash2.h>
#include <arch/rad_linear.h>

#include <cstring>

namespace inkling2 {

using namespace rad::arch;

/* Whether the container carries a DFlash2 drafter: the importer merges its config.json under
 * `draft.` (Checkpoint::add_draft). */
inline bool dflash2_present(const RadModelMeta* meta) {
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    return da && std::strstr(da, "DFlash2");
}

/* The depth it was trained at: block_size - 1 mask rows, the anchor proposing nothing. */
inline int64_t dflash2_depth(const RadModelMeta* meta) {
    return rad_meta_geti(meta, "draft.dflash_config.block_size", 0) - 1;
}

/* The weight prefetch window beside the query pass's all-reduces: 12 MiB, the value measured for
 * the Qwen3.8 DFlash2 drafter on two R9700s (not re-measured on this one). */
constexpr int64_t kInkDf2Prefetch = (int64_t)12 << 20;

struct InkDflash2 {
    enum { kMaxTaps = 8 };

    bool    on = false;
    int     n_taps = 0;
    int     tap_app[kMaxTaps] = {};   /* the applications whose input stream is tapped; -1: the
                                       * final norm's output */
    int64_t E = 0;
    rad_buf src = 0;                  /* [max_tok, n_taps * E] bf16, written by the trunk */
    rad_kvgroup kv = 0;
    Dflash2       df{};
    Dflash2::Wire w{};
    rad_op op_tap = 0, op_tap_out = 0;

    int  declare(RadBuilder* b, Names& nm, const RadModelMeta* meta, const RadBuildCtx* ctx,
                 const Geom& target, int base_layer, int n_apps, rad_buf stream, rad_buf final_out,
                 rad_weight embed, rad_weight embed_norm, rad_weight head, int64_t vocab_real);
    /* The tap for the stream entering application a, or with a == -1 the final norm's output. */
    void tap(RadCtx* c, rad_buf s, int64_t T, int a) const {
        if (!on) return;
        for (int t = 0; t < n_taps; ++t)
            if (tap_app[t] == a)
                RAD_ISSUE_N(c, a < 0 ? op_tap_out : op_tap, T, brows(s, T),
                            bcol(src, (int64_t)t * E, E, T));
    }
    void context(RadCtx* c, const RadBatch* batch) const { df.context(c, batch); }
    void query(RadCtx* c, const RadBatch* batch) const { df.step(c, batch); }
};

inline int InkDflash2::declare(RadBuilder* b, Names& nm, const RadModelMeta* meta,
                               const RadBuildCtx* ctx, const Geom& target, int base_layer,
                               int n_apps, rad_buf stream, rad_buf final_out, rad_weight embed,
                               rad_weight embed_norm, rad_weight head, int64_t vocab_real) {
    Geom dg{};
    RAD_ARCH_TRY(drafter_geom(dg, target, meta, "Inkling 2 DFlash2"));
    E = dg.n_embd;
    if (rad_meta_geti(meta, "draft.hidden_size", E) != E) {
        fprintf(stderr, "radiance: inkling2: the DFlash2 drafter reads a %lld-wide trunk and this "
                        "target is %lld wide\n",
                (long long)rad_meta_geti(meta, "draft.hidden_size", 0), (long long)E);
        return RAD_E_UNSUPPORTED;
    }
    if (ctx->draft_coupling) {
        fprintf(stderr, "radiance: inkling2: the DFlash2 drafter's selector caps its scores after "
                        "the edge correction, which the greedy walk serves; --draft-coupling "
                        "samples the walk and is not served here\n");
        return RAD_E_UNSUPPORTED;
    }

    /* the taps: applications in [0, n_apps), sorted, the last the final norm (n_apps) */
    {
        int ids[kMaxTaps];
        const int n = parse_layer_ids(rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", ""),
                                      n_apps + 1, ids, kMaxTaps);
        bool ok = n >= 1 && ids[n - 1] == n_apps;
        for (int t = 0; ok && t + 1 < n; ++t) ok = ids[t] >= 0 && ids[t] < ids[t + 1];
        if (!ok) {
            fprintf(stderr, "radiance: inkling2: the DFlash2 drafter's target_layer_ids are not "
                            "sorted layer applications ending at %d (the final norm), at most %d\n",
                    n_apps, (int)kMaxTaps);
            return RAD_E_UNSUPPORTED;
        }
        n_taps = n;
        for (int t = 0; t < n; ++t) tap_app[t] = ids[t] == n_apps ? -1 : ids[t];
    }

    Dflash2::Config dc{};
    dc.short_block = 1;
    dc.prefetch    = kInkDf2Prefetch;
    dc.n_taps      = n_taps;
    dc.block       = ctx->max_spec + 1;
    dc.taps        = rad_meta_geti(meta, "draft.dflash_config.conv_kernel_size", 2);
    dc.group       = rad_meta_geti(meta, "draft.dflash_config.conv_group_size", 16);
    dc.sel_rank    = rad_meta_geti(meta, "draft.dflash_config.selector_rank", 0);
    dc.sel_topk    = rad_meta_geti(meta, "draft.dflash_config.selector_top_k", 0);
    dc.causal      = rad_meta_geti(meta, "draft.is_causal", 0) ? 1 : 0;
    if (rad_meta_geti(meta, "draft.use_sliding_window", 0))
        dc.window = rad_meta_geti(meta, "draft.dflash_config.swa_window_size",
                                  rad_meta_geti(meta, "draft.sliding_window", 0));
    if (dc.sel_rank <= 0 || dc.sel_topk <= 0) {
        fprintf(stderr, "radiance: inkling2: the DFlash2 drafter states no selector geometry "
                        "(rank %lld, top_k %lld)\n", (long long)dc.sel_rank, (long long)dc.sel_topk);
        return RAD_E_INVAL;
    }
    const long long trained = rad_meta_geti(meta, "draft.dflash_config.block_size", 0);
    if (dc.block < 2 || (dc.block & (dc.block - 1))) {
        fprintf(stderr, "radiance: inkling2: DFlash2's convolution masks the block position with "
                        "block_size - 1, so the query block has to be a power of two. This serve "
                        "asks for %d draft tokens, which makes it %lld; the checkpoint was trained "
                        "at %lld.\n", ctx->max_spec, (long long)dc.block, trained);
        return RAD_E_UNSUPPORTED;
    }
    if (trained > 0 && trained != dc.block)
        rad_note(b, "DFlash2 was trained at block_size %lld (%lld draft tokens) and runs at %lld; "
                    "it accepts less away from its trained block length",
                 trained, trained - 1, (long long)dc.block);
    dc.escale = rad_meta_getf(meta, "draft.dflash_config.input_embedding_scale", 1.0);
    dc.omul   = rad_meta_getf(meta, "draft.dflash_config.output_multiplier", 1.0);
    dc.scap   = rad_meta_getf(meta, "draft.dflash_config.final_logit_softcapping", 0.0);
    {
        const char* mode = rad_meta_gets(meta, "draft.dflash_config.selector_softcap_mode",
                                         "before_correction");
        if (!std::strcmp(mode, "after_correction")) {
            dc.cap_after = true;
        } else if (std::strcmp(mode, "before_correction") != 0) {
            fprintf(stderr, "radiance: inkling2: the DFlash2 drafter's selector_softcap_mode '%s' "
                            "is neither before_correction nor after_correction\n", mode);
            return RAD_E_UNSUPPORTED;
        }
    }
    dc.vocab_real = vocab_real;
    RAD_ARCH_TRY(Dflash2::probe_i8(b, nm, "draft", &dc.act_i8, &dc.act_g8));
    dc.fc_mapped = true;

    RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, dg, dc.window, base_layer, &kv));

    const int64_t T   = dg.max_tok;
    const int64_t qd  = dg.n_head * dg.head_dim;
    const int64_t kvd = dg.n_head_kv * dg.head_dim;
    const uint32_t act_dtype = dc.act_i8 ? RAD_I8 : RAD_F8E4M3;
    const int64_t act_group = dc.act_i8 ? dc.act_g8 : (int64_t)RAD_SCALE_GROUP;
    const auto act_groups = [act_group](int64_t n) { return (n + act_group - 1) / act_group; };
    int64_t qr = dg.max_seqs * dc.block;
    if (qr > T) qr = T;
    const int64_t sr = dg.max_seqs * dc.steps();

    /* The tap buffer, written by the trunk at three applications and the final norm and read by
     * both drafter passes, so it is live for the whole step. */
    src = decl_b(b, nm.f("df_src"), RAD_BF16, {T, (int64_t)n_taps * E});
    if (!src) return RAD_E_INVAL;
    RAD_ARCH_TRY(rad_buf_concurrent(b, src));

    w.src     = src;
    w.c       = decl_b(b, nm.f("df_c"),    dg.act_dtype, {T, E});
    w.ck      = decl_b(b, nm.f("df_ck"),   dg.act_dtype, {T, kvd});
    w.cv      = decl_b(b, nm.f("df_cv"),   dg.act_dtype, {T, kvd});
    w.x       = decl_b(b, nm.f("df_x"),    dg.act_dtype, {qr, E});
    w.h       = decl_b(b, nm.f("df_h"),    dg.act_dtype, {qr, E});
    w.h2      = decl_b(b, nm.f("df_h2"),   dg.act_dtype, {qr, E});
    w.coef    = decl_b(b, nm.f("df_coef"), dg.act_dtype, {qr, dc.coef_dim(E)});
    w.q       = decl_b(b, nm.f("df_q"),    dg.act_dtype, {qr, qd});
    w.k       = decl_b(b, nm.f("df_k"),    dg.act_dtype, {qr, kvd});
    w.v       = decl_b(b, nm.f("df_v"),    dg.act_dtype, {qr, kvd});
    w.attn    = decl_b(b, nm.f("df_attn"), dg.act_dtype, {qr, qd});
    w.gate_up = decl_b(b, nm.f("df_gate_up"), dg.act_dtype, {qr, 2 * dg.n_ff});
    w.ffn     = decl_b(b, nm.f("df_ffn"),  dg.act_dtype, {qr, dg.n_ff});
    w.srcq    = decl_b(b, nm.f("df_srcq"), act_dtype, {T, (int64_t)n_taps * E});
    w.srcs    = decl_b(b, nm.f("df_srcs"), RAD_F32, {T, act_groups((int64_t)n_taps * E)});
    w.cq      = decl_b(b, nm.f("df_cq"),   act_dtype, {T, E});
    w.cs      = decl_b(b, nm.f("df_cs"),   RAD_F32,   {T, act_groups(E)});
    w.h2q     = decl_b(b, nm.f("df_h2q"),  act_dtype, {qr, E});
    w.h2s     = decl_b(b, nm.f("df_h2s"),  RAD_F32,   {qr, act_groups(E)});
    w.atq     = decl_b(b, nm.f("df_atq"),  act_dtype, {qr, qd});
    w.ats     = decl_b(b, nm.f("df_ats"),  RAD_F32,   {qr, act_groups(qd)});
    w.ffq     = decl_b(b, nm.f("df_ffq"),  act_dtype, {qr, dg.n_ff});
    w.ffs     = decl_b(b, nm.f("df_ffs"),  RAD_F32,   {qr, act_groups(dg.n_ff)});
    w.sel_h   = decl_b(b, nm.f("df_selh"), dg.act_dtype, {sr, E});
    w.hp      = decl_b(b, nm.f("df_hp"),   dg.act_dtype, {sr, dc.sel_rank});
    /* logits_gemm's f32 logits over this rank's rows of the target's head */
    w.dlog    = decl_b(b, nm.f("df_dlog"), RAD_F32, {sr, Dflash2::head_rows(dg)});
    w.cand    = decl_b(b, nm.f("df_cand"), RAD_I32, {sr, dc.sel_topk});
    w.unary   = decl_b(b, nm.f("df_unary"), RAD_F32, {sr, dc.sel_topk});
    if (Dflash2::head_sharded(dg)) {
        w.dpair = decl_b(b, nm.f("df_dpair"), RAD_F32, {sr, dc.sel_topk, 2});
        w.dgath = decl_b(b, nm.f("df_dgath"), RAD_F32, {sr, (int64_t)dg.world * dc.sel_topk, 2});
        if (!w.dpair || !w.dgath) return RAD_E_INVAL;
    }
    w.tokens  = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {dg.max_seqs, dc.steps()});
    for (rad_buf h : { w.c, w.ck, w.cv, w.x, w.h, w.h2, w.coef, w.q, w.k, w.v, w.attn, w.gate_up,
                       w.ffn, w.srcq, w.srcs, w.cq, w.cs, w.h2q, w.h2s, w.atq, w.ats, w.ffq, w.ffs,
                       w.sel_h, w.hp, w.dlog, w.cand, w.unary, w.tokens })
        if (!h) return RAD_E_INVAL;

    /* The trunk's copies into the tap buffer, one column slice a stream: declared after the
     * trunk, so the final norm's output lives until its copy. */
    op_tap = cast_op(b, T, E, "bf16", "bf16", stream, src);
    op_tap_out = cast_op(b, T, E, "bf16", "bf16", final_out, src);
    /* embed_norm over the query block's looked-up rows, as the trunk norms its own (its eps) */
    df.op_post_embed = rw(b, RAD_OP(b, "rmsnorm",
                                    RAD_PARAMS(RAD_RANGE("M", 1, qr), RAD_INT("n", E),
                                               RAD_F64("eps", target.eps), RAD_STR("dtype", "bf16")),
                                    RAD_WEIGHTS(embed_norm)),
                          {w.x}, {w.x});
    df.post_embed_w = embed_norm;
    df.w_thead = head;
    if (!op_tap || !op_tap_out || !df.op_post_embed) return RAD_E_INVAL;

    RAD_ARCH_TRY(df.declare(b, nm, dg, dc, base_layer, kv, embed, "draft", nullptr, w));

    RadDrafterDecl drf{};
    drf.name           = "dflash2";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = dc.block;
    drf.proposal       = w.tokens;
    drf.proposal_pitch = dc.steps();
    drf.window         = dc.window;
    drf.mask_token     = (int32_t)rad_meta_geti(meta, "draft.dflash_config.mask_token_id", -1);
    if (drf.mask_token < 0 || drf.mask_token >= vocab_real) {
        fprintf(stderr, "radiance: inkling2: this DFlash2 checkpoint's dflash_config.mask_token_id "
                        "(%d) is not an id of the target's %lld-token vocabulary\n",
                drf.mask_token, (long long)vocab_real);
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));
    if (df.block_lo > 0) RAD_ARCH_TRY(rad_drafter_short_block(b, df.block_lo));
    on = true;

    rad_note(b, "Inkling 2 DFlash2 drafter: %lld layers at absolute %d..%d, %lld heads / %lld kv x "
                "%lld, ffn %lld, %s attention with a %lld-token window; taps at applications %d, "
                "%d, %d and the final norm (%d in all); one forward proposes %lld tokens through "
                "the candidate selector over the target's head and its %lld real tokens%s",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1,
             (long long)dg.n_head, (long long)dg.n_head_kv, (long long)dg.head_dim,
             (long long)dg.n_ff, dc.causal ? "causal" : "non-causal", (long long)dc.window,
             tap_app[0], n_taps > 1 ? tap_app[1] : -1, n_taps > 2 ? tap_app[2] : -1, n_taps,
             (long long)dc.steps(), (long long)vocab_real,
             dc.cap_after ? ", softcap after the edge correction" : "");
    return RAD_OK;
}

}  /* namespace inkling2 */

#endif /* INKLING2_DFLASH2_H */
