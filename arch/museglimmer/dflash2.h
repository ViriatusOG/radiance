/* dflash2.h -- wire rad_block_dflash2.h onto a Muse Glimmer trunk.
 *
 * Meta's assistant is a plain DFlash (rad_block_muse_dflash.h). incoai's Muse-Glimmer-30B-DFlash2
 * is the DFlash2 block: candidate selector, grouped convolutions, 8 KV heads. The two are
 * selected by draft.architectures. Muse's DFlash2 ships output_multiplier and
 * final_logit_softcapping; the block applies those (Config::omul / Config::scap).
 */
#pragma once

#include <arch/rad_block_dflash2.h>
#include <arch/rad_block_muse_dflash.h>
#include <arch/rad_linear.h>

#include <cstring>

namespace rad {
namespace arch {

inline bool muse_draft_is_dflash2(const RadModelMeta* meta) {
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    return da && std::strstr(da, "DFlash2");
}

/* The drafter's weight prefetch window: 12 MiB, the value measured for the Qwen3.8 DFlash2
 * drafter on two R9700s. */
constexpr int64_t kMuseDf2Prefetch = (int64_t)12 << 20;

inline int declare_muse_dflash2(Dflash2* ds, Dflash2::Wire* w, rad_kvgroup* kv,
                                RadBuilder* b, Names& nm, const RadModelMeta* meta,
                                const RadBuildCtx* ctx, const Geom& target,
                                const MuseTaps& taps, int base_layer, rad_weight tok_embd,
                                rad_weight head_w) {
    if (!ds || !w || !kv || !b) return RAD_E_INVAL;
    Geom dg{};
    RAD_ARCH_TRY(drafter_geom(dg, target, meta, "DFlash2", /*kv_repl_ok=*/true));
    const int64_t want_embd = rad_meta_geti(meta, "draft.hidden_size",
                                           rad_meta_geti(meta, "draft.dflash_config.hidden_size",
                                                         dg.n_embd));
    if (want_embd != dg.n_embd) {
        fprintf(stderr, "radiance: museglimmer: the DFlash2 drafter reads a %lld-wide trunk and "
                        "this target is %lld wide; they are not the same model\n",
                (long long)want_embd, (long long)dg.n_embd);
        return RAD_E_UNSUPPORTED;
    }

    Dflash2::Config dc{};
    dc.short_block = 1;
    dc.prefetch    = kMuseDf2Prefetch;
    dc.n_taps      = taps.n;
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
        fprintf(stderr, "radiance: museglimmer: the DFlash2 drafter states no selector geometry "
                        "(rank %lld, top_k %lld)\n",
                (long long)dc.sel_rank, (long long)dc.sel_topk);
        return RAD_E_INVAL;
    }

    const long long trained = rad_meta_geti(meta, "draft.dflash_config.block_size", 0);
    if (dc.block < 2 || (dc.block & (dc.block - 1))) {
        fprintf(stderr, "radiance: museglimmer: DFlash2's convolution masks the block position "
                        "with block_size - 1, so the query block has to be a power of two. This "
                        "serve asks for %d draft tokens, which makes it %lld; the checkpoint was "
                        "trained at %lld.\n",
                ctx->max_spec, (long long)dc.block, trained);
        return RAD_E_UNSUPPORTED;
    }
    if (trained > 0 && trained != dc.block)
        rad_note(b, "DFlash2 was trained at block_size %lld (%lld draft tokens) and runs at "
                    "%lld; it accepts less away from its trained block length",
                 trained, trained - 1, (long long)dc.block);

    dc.escale = rad_meta_getf(meta, "draft.dflash_config.input_embedding_scale", 1.0);
    dc.omul   = rad_meta_getf(meta, "draft.dflash_config.output_multiplier", 1.0);
    dc.scap   = rad_meta_getf(meta, "draft.dflash_config.final_logit_softcapping", 0.0);
    RAD_ARCH_TRY(Dflash2::probe_i8(b, nm, "draft", &dc.act_i8, &dc.act_g8));
    dc.fc_mapped = true;

    RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, dg, dc.window, base_layer, kv));

    const int64_t qd  = dg.n_head * dg.head_dim;
    const int64_t kvd = dg.n_head_kv * dg.head_dim;
    const uint32_t act_dtype = dc.act_i8 ? RAD_I8 : RAD_F8E4M3;
    const int64_t act_group = dc.act_i8 ? dc.act_g8 : (int64_t)RAD_SCALE_GROUP;
    const auto act_groups = [act_group](int64_t n) { return (n + act_group - 1) / act_group; };
    int64_t qr = dg.max_seqs * dc.block;
    if (qr > dg.max_tok) qr = dg.max_tok;
    const int64_t sr   = dg.max_seqs * dc.steps();
    const int64_t spad = ((sr + 15) / 16) * 16;

    Dflash2::Wire& dw = *w;
    dw.src     = taps.src;
    dw.c       = decl_b(b, nm.f("df_c"),    dg.act_dtype, {dg.max_tok, dg.n_embd});
    dw.ck      = decl_b(b, nm.f("df_ck"),   dg.act_dtype, {dg.max_tok, kvd});
    dw.cv      = decl_b(b, nm.f("df_cv"),   dg.act_dtype, {dg.max_tok, kvd});
    dw.x       = decl_b(b, nm.f("df_x"),    dg.act_dtype, {qr, dg.n_embd});
    dw.h       = decl_b(b, nm.f("df_h"),    dg.act_dtype, {qr, dg.n_embd});
    dw.h2      = decl_b(b, nm.f("df_h2"),   dg.act_dtype, {qr, dg.n_embd});
    dw.coef    = decl_b(b, nm.f("df_coef"), dg.act_dtype, {qr, dc.coef_dim(dg.n_embd)});
    dw.q       = decl_b(b, nm.f("df_q"),    dg.act_dtype, {qr, qd});
    dw.k       = decl_b(b, nm.f("df_k"),    dg.act_dtype, {qr, kvd});
    dw.v       = decl_b(b, nm.f("df_v"),    dg.act_dtype, {qr, kvd});
    dw.attn    = decl_b(b, nm.f("df_attn"), dg.act_dtype, {qr, qd});
    dw.gate_up = decl_b(b, nm.f("df_gate_up"), dg.act_dtype, {qr, 2 * dg.n_ff});
    dw.ffn     = decl_b(b, nm.f("df_ffn"),  dg.act_dtype, {qr, dg.n_ff});
    dw.srcq    = decl_b(b, nm.f("df_srcq"), act_dtype,
                        {dg.max_tok, dc.n_taps * dg.n_embd});
    dw.srcs    = decl_b(b, nm.f("df_srcs"), RAD_F32,
                        {dg.max_tok, act_groups(dc.n_taps * dg.n_embd)});
    dw.cq      = decl_b(b, nm.f("df_cq"),   act_dtype, {dg.max_tok, dg.n_embd});
    dw.cs      = decl_b(b, nm.f("df_cs"),   RAD_F32,    {dg.max_tok, act_groups(dg.n_embd)});
    dw.h2q     = decl_b(b, nm.f("df_h2q"),  act_dtype, {qr, dg.n_embd});
    dw.h2s     = decl_b(b, nm.f("df_h2s"),  RAD_F32,    {qr, act_groups(dg.n_embd)});
    dw.atq     = decl_b(b, nm.f("df_atq"),  act_dtype, {qr, qd});
    dw.ats     = decl_b(b, nm.f("df_ats"),  RAD_F32,    {qr, act_groups(qd)});
    dw.ffq     = decl_b(b, nm.f("df_ffq"),  act_dtype, {qr, dg.n_ff});
    dw.ffs     = decl_b(b, nm.f("df_ffs"),  RAD_F32,    {qr, act_groups(dg.n_ff)});
    dw.sel_h   = decl_b(b, nm.f("df_selh"), dg.act_dtype, {sr, dg.n_embd});
    dw.hp      = decl_b(b, nm.f("df_hp"),   dg.act_dtype, {sr, dc.sel_rank});
    dw.dq      = decl_b(b, nm.f("df_dq"),   RAD_I8,  {sr, dg.n_embd});
    dw.ds      = decl_b(b, nm.f("df_ds"),   RAD_F32, {sr, 1});
    dw.dsum    = decl_b(b, nm.f("df_dsum"), RAD_F32, {dg.n_embd / 128, spad});
    dw.dlog    = decl_b(b, nm.f("df_dlog"), dg.act_dtype, {sr, Dflash2::head_rows(dg)});
    {
        const uint32_t hd = weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16);
        if (head_w && (hd == RAD_F8E4M3 || hd == RAD_I8) &&
            (Dflash2::head_sharded(dg) || dg.world <= 1)) {
            ds->w_lm  = head_w;
            ds->top_w = Dflash2::kRerankR;
        }
    }
    const int64_t tw = ds->w_lm ? ds->top_w : dc.sel_topk;
    dw.cand    = decl_b(b, nm.f("df_cand"), RAD_I32, {sr, tw});
    dw.unary   = decl_b(b, nm.f("df_unary"), RAD_F32, {sr, tw});
    if (Dflash2::head_sharded(dg)) {
        dw.dpair = decl_b(b, nm.f("df_dpair"), RAD_F32, {sr, tw, 2});
        dw.dgath = decl_b(b, nm.f("df_dgath"), RAD_F32, {sr, (int64_t)dg.world * tw, 2});
    }
    dw.tokens  = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {dg.max_seqs, dc.steps()});
    if (ctx->draft_coupling) {
        if (dc.steps() > RAD_MTP_COUPLE_MAX_ROUNDS || dc.sel_topk > RAD_DFLASH_COUPLE_MAX_K ||
            dc.sel_rank % RAD_DFLASH_EDGE_PARTS || dc.sel_rank > RAD_DFLASH_COUPLE_MAX_R) {
            fprintf(stderr, "radiance: museglimmer: --draft-coupling on DFlash2 needs at most "
                            "%d drafts, %d candidates and a selector rank that is a multiple of "
                            "%d up to %d\n",
                    RAD_MTP_COUPLE_MAX_ROUNDS, RAD_DFLASH_COUPLE_MAX_K, RAD_DFLASH_EDGE_PARTS,
                    RAD_DFLASH_COUPLE_MAX_R);
            return RAD_E_UNSUPPORTED;
        }
        dc.q_temp = 0.6;
        dw.couple_draws = decl_b(b, nm.f("mtp_couple_draws"), RAD_U32,
                                 {dg.max_seqs, RAD_MTP_COUPLE_DRAW_WORDS}, RAD_BUF_PERSIST);
        dw.couple_state = decl_b(b, nm.f("mtp_couple_state"), RAD_U32,
                                 {dc.steps(), dg.max_seqs, RAD_MTP_COUPLE_STATE_WORDS},
                                 RAD_BUF_PERSIST);
        if (!dw.couple_draws || !dw.couple_state) return RAD_E_INVAL;
    }

    RAD_ARCH_TRY(ds->declare(b, nm, dg, dc, base_layer, *kv, tok_embd, "draft",
                             nm.ckpt("lm_head.weight"), dw));

    RadDrafterDecl drf{};
    drf.name           = "dflash2";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    drf.block          = dc.block;
    drf.proposal       = dw.tokens;
    drf.proposal_pitch = dc.steps();
    drf.window         = dc.window;
    drf.mask_token     = (int32_t)rad_meta_geti(meta, "draft.dflash_config.mask_token_id", -1);
    if (drf.mask_token < 0) {
        fprintf(stderr, "radiance: museglimmer: this DFlash2 checkpoint declares no "
                        "dflash_config.mask_token_id\n");
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));
    if (ds->block_lo > 0) RAD_ARCH_TRY(rad_drafter_short_block(b, ds->block_lo));

    rad_note(b, "DFlash2 drafter: %lld layers at absolute %d..%d, %lld heads / %lld kv x %lld, "
                "ffn %lld, %s attention with a %lld-token window, %d trunk tap(s); one forward "
                "proposes %lld tokens through the candidate selector",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1,
             (long long)dg.n_head, (long long)dg.n_head_kv, (long long)dg.head_dim,
             (long long)dg.n_ff, dc.causal ? "causal" : "non-causal", (long long)dc.window,
             taps.n, (long long)dc.steps());
    if (dc.omul != 1.0 || dc.scap > 0.0)
        rad_note(b, "DFlash2 logits: output_multiplier %g, softcap %g", dc.omul, dc.scap);
    return RAD_OK;
}

}  /* namespace arch */
}  /* namespace rad */
