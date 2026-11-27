/* qwen35_exl3.cpp -- Qwen3.5 / 3.6 / 3.8 dense architecture plugin with EXL3 weights.
 *
 * Implements out-of-tree architecture plugin for plugin ABI 44.
 * GDN, Attention, and MLP blocks derived directly from Radiance's BF16 blocks
 * (rad_block_gdn_exl3.h, rad_block_attn_gated_exl3.h, rad_block_mlp_exl3.h),
 * replacing projection gemm_nt declarations with the dense EXL3-V3 op (exl3i_dense).
 *
 * Media and speculation are stock radiance's, unchanged: the vision tower and the scatter
 * of its rows (arch/common/rad_block_vit.h, bf16, as qwen35_bf16 declares them: only when the container
 * carries a tower and the deployment asked for media), and the DFlash2 drafter (arch/common/rad_block_dflash2.h,
 * declared as qwen35_fp8 declares it: only when the container carries a `draft.` DFlash2 and the deployment
 * speculates; its linears block fp8 / MXFP4 and its 2-bit head as the recipe made them, the trunk's residual
 * stream tapped at the layers it names). The trunk's exl3 weights are not touched by either.
 */

#include "qwen35_exl3.h"

#include <arch/rad_arch.h>
#include <arch/rad_block_dflash2.h>
#include <arch/rad_block_vit.h>
#include "exl3_dense_op.h"
#include "exl3i_rerank.h"
#include "rad_block_attn_gated_exl3.h"
#include "rad_block_gdn_exl3.h"
#include "rad_block_mlp_exl3.h"
#include "exl3_fuse.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace qwen35_exl3 {

using namespace rad::arch;

struct Layer {
    int           full = 0;
    AttnGatedEXL3 attn;
    GdnEXL3       gdn;
    MlpEXL3       mlp;
};

struct Model {
    Names           nm{""};
    Geom            g{};
    GdnEXL3::Config gcfg{};
    int64_t         n_rot  = 0;
    int64_t         n_main = 0;

    rad_kvgroup kv_full = 0, kv_state = 0, kv_conv = 0;

    /* Residual stream & activations */
    rad_buf b_x = 0, b_h = 0, b_hout = 0, b_logits = 0;
    /* Full attention */
    rad_buf b_qg = 0, b_k = 0, b_v = 0, b_q = 0, b_gate = 0, b_attn = 0;
    /* exl3_fuse.h qk: the rope cos / sin table every attention layer reads ([exl3_rope_rows, n_rot] f32, PERSIST: its
     * rows outlive the step that wrote them) and its once-a-step fill; 0 when the fusion is off or not served */
    rad_buf b_rope_cs = 0;
    rad_op  op_rope_cs = 0;
    /* GDN */
    rad_buf b_in = 0, b_z = 0, b_gq = 0, b_gk = 0, b_gv = 0;
    rad_buf b_ga = 0, b_gb = 0;   /* the GDN gate and beta projections, plain bf16 */
    rad_buf b_gab = 0;            /* fused layers: ssm_ab's output, a | b | padding */
    std::vector<int64_t> ab_rows; /* per layer: ssm_ab's rows when it is a trellis weight, 0 for plain a / b */
    rad_buf b_gdec = 0, b_beta = 0, b_kkt = 0, b_go = 0;
    /* MLP */
    rad_buf b_mlp_gate_up = 0, b_ffn = 0;
    /* AQ: the producers' int8 rows for their exl3i A8 consumers (declared by the first site that turns on) */
    Exl3AqBuf aq{};

    /* ---- media (arch/common/rad_block_vit.h), as qwen35_bf16: the vision tower and the scatter of its rows */
    bool        media = false;
    VisionTower vit{};
    MediaRows   mrows{};

    /* ---- the DFlash2 drafter (arch/common/rad_block_dflash2.h), as qwen35_fp8: its block, KV group and wire,
     * and the trunk's taps (the residual stream after each named layer, copied into its column of dflash_src) */
    bool          have_df = false;
    Dflash2       df{};
    Dflash2::Wire dfw{};
    rad_kvgroup   kv_draft = 0;
    enum { MAX_TAPS = 8 };
    int      n_taps = 0;
    int      tap_layer[MAX_TAPS] = {};
    rad_buf  b_dfsrc = 0;   /* [max_tok, n_taps * n_embd] */
    rad_op   op_tap = 0;

    /* Vocabulary edges */
    rad_weight      w_tok = 0, w_tok_scale = 0, w_tok_table = 0, w_out_norm = 0;
    int             tok_bits = 0;    /* > 0: token_embd is table-quantised under fwht128 (embed_tq) */
    rad_weight      w_tok_suh = 0, w_tok_svh = 0;
    int             tok_m = 0, tok_cb = 0, tok_r16 = 0;   /* tok_m > 0: token_embd is an exl3i trellis at rate tok_m / 16
                                                           * (embed_exl3i; w_tok = its record) */
    Exl3DenseWeight wt_head{};
    rad_op          op_embed = 0, op_embed_ar = 0, op_final_norm = 0, op_gather = 0, op_logits = 0;
    /* exl3_fuse.h addnorm: each block leaves its residual add to the next norm (the last one's to op_final_addnorm);
     * a block whose norm is not served takes it as a plain add first, so `defer_add` needs only the switch */
    bool            defer_add = false;
    rad_op          op_final_add = 0, op_final_addnorm = 0;

    std::vector<Layer> layers;
};

static Model g_model[MAX_RANKS];

/* The DFlash2 drafter: qwen35_fp8.cpp's declare_dflash2, unchanged but for what this trunk does not
 * have: no prefetch beside all-reduces (the trunk's exl3i weights are sharded by the loader's plane slices), and
 * candidates are rescored against a copy of the original lm_head the recipe adds (`dflash.rerank_head`, below)
 * instead of the target's head (logit_rerank reads an fp8 or int8 head; this one is exl3i or bf16) -- or, with
 * no copy and an exl3i head, against that head itself by exl3i_rerank (tp 1 only: its kernel reads the head where
 * exl3i_dense's layout put it, and a replicated R16 head's per-rank geometry it does not know). The draft head
 * (`dflash.draft_head`, the block's) maps the
 * checkpoint's lm_head too, so both come from the bf16 original whatever head the trunk imports. Its geometry, KV
 * group (the window, at absolute layers base_layer..), activations, the selector's planes and the drafter declaration
 * are stock's. */
static int declare_dflash2(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                           Model& m, int base_layer) {
    Names& nm = m.nm;
    Geom   dg{};
    /* Plain RMS norm: the drafter uses the ordinary norm where the trunk uses GemmaRMSNorm
     * (rad_block_dflash2.h). */
    RAD_ARCH_TRY(drafter_geom(dg, m.g, meta, "DFlash2"));

    Dflash2::Config dc{};
    dc.short_block = 1;   /* rad_drafter_short_block below */
    dc.prefetch    = 0;
    dc.n_taps   = m.n_taps;
    dc.block    = ctx->max_spec + 1;
    dc.taps     = rad_meta_geti(meta, "draft.dflash_config.conv_kernel_size", 2);
    dc.group    = rad_meta_geti(meta, "draft.dflash_config.conv_group_size", 16);
    dc.sel_rank = rad_meta_geti(meta, "draft.dflash_config.selector_rank", 0);
    dc.sel_topk = rad_meta_geti(meta, "draft.dflash_config.selector_top_k", 0);
    dc.causal   = rad_meta_geti(meta, "draft.is_causal", 0) ? 1 : 0;
    if (rad_meta_geti(meta, "draft.use_sliding_window", 0))
        dc.window = rad_meta_geti(meta, "draft.dflash_config.swa_window_size",
                                  rad_meta_geti(meta, "draft.sliding_window", 0));
    if (dc.sel_rank <= 0 || dc.sel_topk <= 0) {
        fprintf(stderr, "radiance: the DFlash2 drafter states no selector geometry "
                        "(rank %lld, top_k %lld). Without it the block of drafts has no path "
                        "through it.\n", (long long)dc.sel_rank, (long long)dc.sel_topk);
        return RAD_E_INVAL;
    }
    const long long trained = rad_meta_geti(meta, "draft.dflash_config.block_size", 0);
    if (dc.block < 2 || (dc.block & (dc.block - 1))) {
        fprintf(stderr, "radiance: DFlash2's convolution masks the block position with "
                        "block_size - 1, so the query block has to be a power of two. This serve "
                        "asks for %d draft tokens, which makes it %lld; the checkpoint was trained "
                        "at %lld (%lld draft tokens).\n",
                ctx->max_spec, (long long)dc.block, trained, trained - 1);
        return RAD_E_UNSUPPORTED;
    }
    if (trained > 0 && trained != dc.block)
        rad_note(b, "DFlash2 was trained at block_size %lld (%lld draft tokens) and runs at "
                    "%lld; it accepts less away from its trained block length",
                 trained, trained - 1, (long long)dc.block);
    const double escale = rad_meta_getf(meta, "draft.dflash_config.input_embedding_scale", 1.0);
    const double omul   = rad_meta_getf(meta, "draft.dflash_config.output_multiplier", 1.0);
    const double scap   = rad_meta_getf(meta, "draft.dflash_config.final_logit_softcapping", 0.0);
    if (escale != 1.0 || omul != 1.0 || scap != 0.0) {
        fprintf(stderr, "radiance: this DFlash2 checkpoint scales its input embedding by %g and "
                        "its draft logits by %g, and soft-caps them at %g. None of the three is "
                        "implemented, and a drafter that ignored them would just accept less.\n",
                escale, omul, scap);
        return RAD_E_UNSUPPORTED;
    }

    RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, dg, dc.window, base_layer, &m.kv_draft));

    const int64_t qd  = dg.n_head * dg.head_dim;
    const int64_t kvd = dg.n_head_kv * dg.head_dim;
    int64_t qr = dg.max_seqs * dc.block;
    if (qr > dg.max_tok) qr = dg.max_tok;
    const int64_t sr   = dg.max_seqs * dc.steps();
    const int64_t spad = ((sr + 15) / 16) * 16;

    Dflash2::Wire& dw = m.dfw;
    dw.src     = m.b_dfsrc;
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
    dw.srcq    = decl_b(b, nm.f("df_srcq"), RAD_F8E4M3,
                        {dg.max_tok, dc.n_taps * dg.n_embd});
    dw.srcs    = decl_b(b, nm.f("df_srcs"), RAD_F32,
                        {dg.max_tok, scale_groups(dc.n_taps * dg.n_embd)});
    dw.cq      = decl_b(b, nm.f("df_cq"),   RAD_F8E4M3, {dg.max_tok, dg.n_embd});
    dw.cs      = decl_b(b, nm.f("df_cs"),   RAD_F32,    {dg.max_tok, scale_groups(dg.n_embd)});
    dw.h2q     = decl_b(b, nm.f("df_h2q"),  RAD_F8E4M3, {qr, dg.n_embd});
    dw.h2s     = decl_b(b, nm.f("df_h2s"),  RAD_F32,    {qr, scale_groups(dg.n_embd)});
    dw.atq     = decl_b(b, nm.f("df_atq"),  RAD_F8E4M3, {qr, qd});
    dw.ats     = decl_b(b, nm.f("df_ats"),  RAD_F32,    {qr, scale_groups(qd)});
    dw.ffq     = decl_b(b, nm.f("df_ffq"),  RAD_F8E4M3, {qr, dg.n_ff});
    dw.ffs     = decl_b(b, nm.f("df_ffs"),  RAD_F32,    {qr, scale_groups(dg.n_ff)});
    dw.sel_h   = decl_b(b, nm.f("df_selh"), dg.act_dtype, {sr, dg.n_embd});
    dw.hp      = decl_b(b, nm.f("df_hp"),   dg.act_dtype, {sr, dc.sel_rank});
    dw.dq      = decl_b(b, nm.f("df_dq"),   RAD_I8,  {sr, dg.n_embd});
    dw.ds      = decl_b(b, nm.f("df_ds"),   RAD_F32, {sr, 1});
    dw.dsum    = decl_b(b, nm.f("df_dsum"), RAD_F32, {dg.n_embd / 128, spad});
    dw.dlog    = decl_b(b, nm.f("df_dlog"), dg.act_dtype, {sr, Dflash2::head_rows(dg)});
    /* The rescoring head: logit_rerank reads fp8 or int8 rows and this trunk's head is exl3i, so
     * the container may carry its own copy, `dflash.rerank_head.weight`, made by the recipe from the checkpoint's
     * lm_head -- the original bf16 one, never the trunk's quantised output.weight (which is an import of its own) --
     * as `codes=i8 group=128 scale=f16 clamp=sym` or block fp8. Then the coarse head proposes kRerankR candidates and
     * the selector reads the best sel_topk by the exact logits, as qwen35_fp8 does against its fp8 head. Mapped for
     * every drafter and declared only when the recipe put it on one of those grids: a container without the line
     * writes no copy (a mapped name nobody declares is not written) and keeps the coarse head's scores. */
    {
        const char* rn = nm.f("dflash.rerank_head.weight");
        RAD_ARCH_TRY(map_copy(b, rn, nm.ckpt("lm_head.weight")));
        const uint32_t hd = weight_codes_dtype(b, rn, RAD_BF16);
        if ((hd == RAD_F8E4M3 || hd == RAD_I8) && (Dflash2::head_sharded(dg) || dg.world <= 1)) {
            m.df.w_lm  = decl_w(b, rn, hd, {dg.n_vocab, dg.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
            if (!m.df.w_lm) return RAD_E_INVAL;
            m.df.top_w = Dflash2::kRerankR;
        }
    }
    /* No copy and an exl3i head: exl3i_rerank rescores the candidates against the trunk's own
     * output.weight, read where exl3i_dense's layout put it (the same stored form, so no byte more of VRAM): the value
     * exl3i_dense's decode form -- what scores the verify rows -- writes at that row, to f32 rounding
     * (include/exl3i_rerank.h). Only where a kernel serves it, at tp 1 (an R16 head serves whole, but the rerank
     * kernel does not take a replicated head's geometry); the test pin rerank_off (Exl3TestPins)
     * keeps the coarse head's own scores. */
    if (!m.df.w_lm && m.wt_head.is_exl3i && dg.world <= 1 && rad_op_declared(b, EXL3I_OP_RERANK)) {
        if (!exl3_test_pins().rerank_off) {
            const rad_weight hw[3] = { m.wt_head.w_codes, m.wt_head.w_suh, m.wt_head.w_svh };
            RadParam rp[] = { RAD_RANGE("M", 1, sr), RAD_INT("R", Dflash2::kRerankR), RAD_INT("N", m.g.n_vocab),
                              RAD_INT("K", m.g.n_embd), RAD_STR("rate", m.wt_head.rate),
                              RAD_STR("codebook", m.wt_head.cb_name) };
            const rad_op o = rad_decl_op(b, EXL3I_OP_RERANK, rp, 6, hw, 3);
            if (o && rad_op_resolved(b, o)) {
                m.df.op_rr_ext = o;
                for (int i = 0; i < 3; ++i) m.df.rr_w[i] = hw[i];
                m.df.top_w = Dflash2::kRerankR;
            }
        }
    }
    /* A trellis token table: the drafter embeds its block (the anchor and the mask tokens) through
     * embed_exl3i, the rows the trunk's own embedding gives, decoded exactly from the stored record -- the mask token's
     * row too, so no bf16 row is kept beside the trellis. */
    if (m.tok_m) {
        const rad_weight ew[3] = { m.w_tok, m.w_tok_suh, m.w_tok_svh };
        const rad_op o = rw(b, RAD_OP(b, "embed_exl3i",
                                RAD_PARAMS(RAD_RANGE("M", 1, qr), RAD_INT("n_embd", m.g.n_embd),
                                           RAD_INT("n_vocab", m.g.n_vocab_all), RAD_INT("m", m.tok_m),
                                           RAD_STR("codebook", exl3i_codebook_name(m.tok_cb)),
                                           RAD_INT("r16", m.tok_r16)),
                                ew, 3),
                            {}, {dw.x});
        if (!o) return RAD_E_INVAL;
        m.df.op_embed_ext = o;
        for (int i = 0; i < 3; ++i) m.df.emb_w[i] = ew[i];
    }
    const int64_t tw = m.df.w_lm || m.df.op_rr_ext ? m.df.top_w : dc.sel_topk;
    dw.cand    = decl_b(b, nm.f("df_cand"), RAD_I32, {sr, tw});
    dw.unary   = decl_b(b, nm.f("df_unary"), RAD_F32, {sr, tw});
    if (m.df.op_rr_ext) {
        const rad_buf rd[2] = { dw.sel_h, dw.cand }, wr[2] = { dw.cand, dw.unary };
        rad_op_reads (b, m.df.op_rr_ext, rd, 2);
        rad_op_writes(b, m.df.op_rr_ext, wr, 2);
    }
    if (Dflash2::head_sharded(dg)) {
        dw.dpair = decl_b(b, nm.f("df_dpair"), RAD_F32, {sr, tw, 2});
        dw.dgath = decl_b(b, nm.f("df_dgath"), RAD_F32, {sr, (int64_t)dg.world * tw, 2});
    }
    dw.tokens  = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {dg.max_seqs, dc.steps()});

    if (ctx->draft_coupling) {   /* --draft-coupling: as qwen35_fp8 (abi/rad_spechub.h) */
        if (dc.steps() > RAD_MTP_COUPLE_MAX_ROUNDS || dc.sel_topk > RAD_DFLASH_COUPLE_MAX_K ||
            dc.sel_rank % RAD_DFLASH_EDGE_PARTS || dc.sel_rank > RAD_DFLASH_COUPLE_MAX_R) {
            rad_note(b, "--draft-coupling on: DFlash2's coupled walk takes at most %d drafts, %d "
                        "candidates and a selector rank that is a multiple of %d up to %d",
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

    RAD_ARCH_TRY(m.df.declare(b, nm, dg, dc, base_layer, m.kv_draft, m.w_tok, "draft",
                              nm.ckpt("lm_head.weight"), dw));
    m.have_df = true;

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
        fprintf(stderr, "radiance: this DFlash2 checkpoint declares no "
                        "dflash_config.mask_token_id, and every drafted position embeds it -- "
                        "there is nothing to run.\n");
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));
    if (m.df.block_lo > 0) RAD_ARCH_TRY(rad_drafter_short_block(b, m.df.block_lo));

    rad_note(b, "DFlash2 drafter: %lld layers at absolute %d..%d, %lld heads / %lld kv x %lld, "
                "ffn %lld, %s attention with a %lld-token window, %d trunk tap(s); one forward "
                "proposes %lld tokens through the candidate selector; %s head, %s; block embedded by %s",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1,
             (long long)dg.n_head, (long long)dg.n_head_kv, (long long)dg.head_dim,
             (long long)dg.n_ff, dc.causal ? "causal" : "non-causal", (long long)dc.window,
             m.n_taps, (long long)dc.steps(), m.df.head_i8 ? "int8" : "2-bit",
             m.df.w_lm ? "its top 32 rescored against dflash.rerank_head"
             : m.df.op_rr_ext ? "its top 32 rescored against output.weight itself (exl3i_rerank, no copy)"
                              : "its own scores (no rescoring head)",
             m.df.op_embed_ext ? "embed_exl3i (the trunk's trellis table)" : "embed_lookup");
    return RAD_OK;
}

/* The speculation depth the container's drafter supports, answered before declare (qwen35_fp8's probe): DFlash2
 * proposes its trained block in one pass, so the depth is block_size - 1 and fixed; 0 without one (this plugin
 * declares no MTP head). */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    if (da && std::strstr(da, "DFlash2")) {
        const long long bs = rad_meta_geti(meta, "draft.dflash_config.block_size", 8);
        out->draft_depth = bs > 1 ? (int)(bs - 1) : 7;
        out->draft_depth_fixed = 1;
    }
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    Model& m = g_model[ctx->rank];
    m = Model{};
    Exl3ActScope act_scope(meta);   /* the weights' activation modes (exl3_act.h keys) for decl_exl3_weight */
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));

    if (meta->n_expert > 0) {
        fprintf(stderr, "qwen35_exl3: refuses checkpoint with %lld experts (MoE)\n",
                (long long)meta->n_expert);
        return RAD_E_UNSUPPORTED;
    }

    Geom&  g  = m.g;
    Names& nm = m.nm;

    bool have_interval = false;
    const int64_t interval = meta_int2(meta, "qwen35.full_attention_interval",
                                       "full_attention_interval", 4, &have_interval);

    if (interval <= 0) {
        fprintf(stderr, "qwen35_exl3: invalid full_attention_interval %lld\n", (long long)interval);
        return RAD_E_INVAL;
    }

    /* Check layer schedule if layer_types string is present in metadata */
    const char* layer_types = meta_text_gets(meta, "layer_types", nullptr);
    if (!layer_types) layer_types = meta_text_gets(meta, "text_config.layer_types", nullptr);
    RAD_ARCH_TRY(check_layer_schedule("qwen35", layer_types, g.n_layer, interval, {"full_attention"}));

    /* Trunk layers: the 64 text layers. The vision tower and the DFlash2 drafter are declared beside them (below,
     * each only when the container carries it and the deployment asks for it); the MTP head is not served. */
    m.n_main = g.n_layer;

    /* Norm eps and GEMM additions */
    g.wadd    = 1.0f;
    g.wadd_qk = 1.0f;
    const float final_wadd = 1.0f;

    /* Rotary width (partial_rotary_factor 0.25 -> 64) */
    m.n_rot = meta_int2(meta, "qwen35.rope.dimension_count", "rope_dimension_count", 0);
    if (m.n_rot <= 0) {
        double prf = rad_meta_getf(meta, "partial_rotary_factor", 0.0);
        if (!(prf > 0.0)) prf = rad_meta_getf(meta, "rope_parameters.partial_rotary_factor", 0.0);
        if (!(prf > 0.0)) prf = rad_meta_getf(meta, "text_config.partial_rotary_factor", 0.0);
        if (prf > 0.0) m.n_rot = (int64_t)(g.head_dim * prf);
        else m.n_rot = (int64_t)(g.head_dim * 0.25); /* default for Qwen3.5 */
    }
    if (m.n_rot <= 0 || m.n_rot > g.head_dim || (m.n_rot & 1)) {
        fprintf(stderr, "qwen35_exl3: invalid rotary width %lld (head_dim %lld)\n",
                (long long)m.n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    RAD_ARCH_TRY(yarn_from(g, meta, "", m.n_rot));

    /* GDN geometry */
    GdnEXL3::Config& gc = m.gcfg;
    const int64_t n_head_k_all = meta_int2(meta, "qwen35.ssm.group_count", "linear_num_key_heads", 16);
    const int64_t n_head_v_all = meta_int2(meta, "qwen35.ssm.time_step_rank", "linear_num_value_heads", 48);
    gc.head_k     = meta_int2(meta, "qwen35.ssm.state_size", "linear_key_head_dim", 128);
    gc.head_v     = meta_int2(meta, "qwen35.ssm.value_head_dim", "linear_value_head_dim", 128);
    gc.conv_width = meta_int2(meta, "qwen35.ssm.conv_kernel", "linear_conv_kernel_dim", 4);
    if (gc.head_v <= 0) {
        const int64_t inner = meta_int2(meta, "qwen35.ssm.inner_size", "linear_inner_size", 0);
        if (inner > 0 && n_head_v_all > 0) gc.head_v = inner / n_head_v_all;
    }
    if (n_head_k_all <= 0 || n_head_v_all <= 0 || gc.head_k <= 0 || gc.head_v <= 0 || gc.conv_width <= 0) {
        fprintf(stderr, "qwen35_exl3: incomplete GDN parameters in metadata\n");
        return RAD_E_INVAL;
    }
    gc.chunk = 64;

    int s;
    if ((s = divide_or_fail(n_head_k_all, g.world, "linear_num_key_heads",   &gc.n_head_k)) < 0) return s;
    if ((s = divide_or_fail(n_head_v_all, g.world, "linear_num_value_heads", &gc.n_head_v)) < 0) return s;
    gc.n_head_v_all = n_head_v_all;   /* ssm_ab's a | b split: the rank's a / b column views within the full output */

    const int64_t q_dim    = g.q_dim();
    const int64_t kv_dim   = g.kv_dim();
    const int64_t v_dim    = gc.v_dim();

    /* KV groups */
    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    g.kv_dtype = rad_dtype_name(fd.dtype);

    RadKVGroupDecl sd{};
    sd.kind         = RAD_KV_LINEAR;
    sd.dtype        = RAD_F32;
    sd.n_head_kv    = gc.n_head_v;
    sd.state_dim[0] = gc.head_v;
    sd.state_dim[1] = gc.head_k;
    m.kv_state = rad_decl_kv_group(b, nm.f("kv_gdn_state"), &sd);

    RadKVGroupDecl cd{};
    cd.kind       = RAD_KV_CONV;
    cd.dtype      = RAD_BF16;
    cd.n_head_kv  = 2 * gc.n_head_k + gc.n_head_v;
    cd.head_dim   = gc.head_k;
    cd.conv_width = gc.conv_width;
    m.kv_conv = rad_decl_kv_group(b, nm.f("kv_gdn_conv"), &cd);

    /* The GDN's a/b (gate and beta) projections are plain bf16 in the checkpoint and in the
     * published EXL3 exports, and the fused ssm_in (q|k|v|a|b) is not in 128s at this model, so
     * the block serves ssm_qkv exl3 beside plain a/b. Their row counts are not in the geometry
     * (48 and 128 at 27B): map every GDN layer's a/b (the buffer plan needs them before the
     * layer loop) and read the source extents.
     *
     * A layer may instead hold them as one trellis weight, ssm_ab: a's rows, b's, then the first rows of a
     * again up to a multiple of 128 (the quantizer's LDLQ takes 128s; the padding is stored and its outputs unread),
     * one row per value head each. It is mapped for every GDN layer; a layer whose ssm_ab is an exl3i weight (a
     * recipe line at convert, the container's entry at load) serves it, every other its plain a / b. */
    const int64_t ab_pair = 2 * n_head_v_all;
    const int64_t ab_rows = (ab_pair + 127) / 128 * 128;
    m.ab_rows.assign((size_t)g.n_layer, 0);
    int64_t n_fused = 0, n_plain = 0, l_plain = -1;
    for (int l = 0; l < (int)g.n_layer; ++l) {
        if (is_full_attention(l, interval)) continue;
        const char* in_a = nm.ckpt("model.language_model.layers.%d.linear_attn.in_proj_a.weight", l);
        const char* in_b = nm.ckpt("model.language_model.layers.%d.linear_attn.in_proj_b.weight", l);
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ssm_a.weight", l), in_a));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ssm_b.weight", l), in_b));
        if (ab_rows - ab_pair <= n_head_v_all) {   /* the padding is a's first rows: at most all of them */
            if (ab_rows > ab_pair)
                RAD_ARCH_TRY(map_rows(b, nm.f("blk.%d.ssm_ab.weight", l),
                                      { { in_a, 0, n_head_v_all }, { in_b, 0, n_head_v_all }, { in_a, 0, ab_rows - ab_pair } }));
            else
                RAD_ARCH_TRY(map_rows(b, nm.f("blk.%d.ssm_ab.weight", l), { { in_a, 0, n_head_v_all }, { in_b, 0, n_head_v_all } }));
            RadEncoding enc{};
            int64_t sh[RAD_MAX_RANK] = {0};
            uint32_t rk = 0;
            if (rad_weight_encoding(b, nm.f("blk.%d.ssm_ab.weight", l), &enc, sh, &rk) == RAD_OK &&
                rad_enc_is(&enc, EXL3I_SCHEME)) {
                if (rk != 2 || sh[0] != ab_rows || sh[1] != g.n_embd) {
                    fprintf(stderr, "qwen35_exl3: blk.%d.ssm_ab is [%lld, %lld]; a | b | padding is [%lld, %lld]\n", l,
                            (long long)sh[0], (long long)sh[1], (long long)ab_rows, (long long)g.n_embd);
                    return RAD_E_SHAPE;
                }
                /* At world > 1 the weight serves whole on every rank (decl_exl3_weight; its GEMM is 128 rows
                 * and the rank's a / b rows are not a slice of the trellis record): the block reads its
                 * v heads' gate and beta as column views of the full output. */
                m.ab_rows[(size_t)l] = ab_rows;
                ++n_fused;
                continue;
            }
        }
        ++n_plain;
        if (l_plain < 0) l_plain = l;
    }
    gc.a_rows = gc.b_rows = n_head_v_all / g.world;   /* per rank: one row per v head, row-sharded with them */
    if (n_plain > 0) {
        RadEncoding enc{};
        int64_t sh[RAD_MAX_RANK] = {0};
        uint32_t rk = 0;
        RAD_ARCH_TRY(rad_weight_encoding(b, nm.f("blk.%d.ssm_a.weight", (int)l_plain), &enc, sh, &rk));
        if (rk != 2 || sh[0] <= 0 || sh[1] != g.n_embd) {
            fprintf(stderr, "qwen35_exl3: in_proj_a is not [rows, n_embd] in the checkpoint\n");
            return RAD_E_FORMAT;
        }
        if (sh[0] % g.world != 0) {
            fprintf(stderr, "qwen35_exl3: in_proj_a's %lld rows do not split over %d ranks\n", (long long)sh[0], g.world);
            return RAD_E_UNSUPPORTED;
        }
        gc.a_rows = sh[0] / g.world;   /* per rank: one row per v head, row-sharded with them */
        RAD_ARCH_TRY(rad_weight_encoding(b, nm.f("blk.%d.ssm_b.weight", (int)l_plain), &enc, sh, &rk));
        if (rk != 2 || sh[0] <= 0 || sh[1] != g.n_embd) {
            fprintf(stderr, "qwen35_exl3: in_proj_b is not [rows, n_embd] in the checkpoint\n");
            return RAD_E_FORMAT;
        }
        if (sh[0] % g.world != 0) {
            fprintf(stderr, "qwen35_exl3: in_proj_b's %lld rows do not split over %d ranks\n", (long long)sh[0], g.world);
            return RAD_E_UNSUPPORTED;
        }
        gc.b_rows = sh[0] / g.world;   /* per rank: one row per v head, row-sharded with them */
        if (n_fused > 0 && (gc.a_rows * g.world != n_head_v_all || gc.b_rows * g.world != n_head_v_all)) {
            fprintf(stderr, "qwen35_exl3: in_proj_a / in_proj_b are %lld / %lld rows, not one a value head (%lld): "
                            "ssm_ab's split would not be theirs\n", (long long)gc.a_rows, (long long)gc.b_rows,
                    (long long)n_head_v_all);
            return RAD_E_FORMAT;
        }
    }
    if (n_fused > 0)
        rad_note(b, "qwen35_exl3: %lld GDN layer(s) serve a | b as one trellis weight (ssm_ab, %lld rows), %lld plain a / b",
                 (long long)n_fused, (long long)ab_rows, (long long)n_plain);

    /* Activation buffers */
    m.b_x       = decl_b(b, nm.f("x"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.b_h       = decl_b(b, nm.f("h"),        g.act_dtype, {g.max_tok, g.n_embd});
    /* Media: the vision tower (rad_block_vit.h, bf16) and the scatter of its rows into x (MediaRows), as qwen35_bf16
     * declares them, when the deployment asked for media and the container carries the tower's weights -- not only its
     * config: a container packed text-only (an older build of this plugin, or a recipe without the tower) keeps the
     * checkpoint's vision_config. The tower's own gate (gate_weights: the stem's map, then its encoding, which convert
     * answers from the checkpoint and load from the container) decides, so it is declared here, ahead of the trunk
     * (the stock archs declare it last; its encoder pass runs alone and hands its rows over through the engine's media
     * staging, so the order only moves its activations in the arena plan), and MediaRows follows its answer. */
    if (VisionTower::present(meta) && ctx->max_enc_patches > 0) {
        RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer + 64, false, true));
        m.media = m.vit.on;
        if (!m.media)
            rad_note(b, "qwen35_exl3: vision_config names a tower this container does not carry; serving text only");
    }
    if (m.media) RAD_ARCH_TRY(m.mrows.declare(b, g.max_tok, g.n_embd, m.b_x));

    m.b_qg      = decl_b(b, nm.f("attn_qg"),  g.act_dtype, {g.max_tok, g.n_head, 2 * g.head_dim});
    m.b_k       = decl_b(b, nm.f("attn_k"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_v       = decl_b(b, nm.f("attn_v"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_q       = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, q_dim});
    m.b_gate    = decl_b(b, nm.f("attn_gate"),g.act_dtype, {g.max_tok, q_dim});
    m.b_attn    = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, q_dim});

    m.b_in      = decl_b(b, nm.f("gdn_in"),   g.act_dtype, {g.max_tok, gc.conv_dim()});
    if (n_plain > 0) {
        m.b_ga  = decl_b(b, nm.f("gdn_a"),    g.act_dtype, {g.max_tok, gc.a_rows});
        m.b_gb  = decl_b(b, nm.f("gdn_b"),    g.act_dtype, {g.max_tok, gc.b_rows});
    }
    if (n_fused > 0)
        m.b_gab = decl_b(b, nm.f("gdn_ab"),   g.act_dtype, {g.max_tok, ab_rows});
    m.b_z       = decl_b(b, nm.f("gdn_z"),    g.act_dtype, {g.max_tok, v_dim});
    m.b_gq      = decl_b(b, nm.f("gdn_q"),    g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    m.b_gk      = decl_b(b, nm.f("gdn_k"),    g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    m.b_gv      = decl_b(b, nm.f("gdn_v"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});
    m.b_gdec    = decl_b(b, nm.f("gdn_g"),    RAD_F32,     {g.max_tok, gc.n_head_v});
    m.b_beta    = decl_b(b, nm.f("gdn_beta"), RAD_F32,     {g.max_tok, gc.n_head_v});
    m.b_kkt     = decl_b(b, nm.f("gdn_kkt"),  RAD_BF16,    {g.max_tok, gc.n_head_v, gc.chunk});
    m.b_go      = decl_b(b, nm.f("gdn_o"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});

    m.b_mlp_gate_up = decl_b(b, nm.f("mlp_gate_up"), g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.b_ffn      = decl_b(b, nm.f("ffn"),      g.act_dtype, {g.max_tok, g.n_ff});

    /* AQ's widest row: the attention norm's three planes, down's (K n_ff), o's (K q_dim), GDN's / gate_up's two */
    m.aq.name  = nm.f("aq");
    m.aq.rows  = g.max_tok;
    m.aq.width = exl3_aq_row_width({ { g.n_embd, 3 }, { g.n_embd, 2 }, { g.n_ff, 1 }, { q_dim, 1 } });
    m.aq.name_xb  = nm.f("aq_xb");   /* the decode band's (mad, form xb) */
    m.aq.rows_xb  = std::min<int64_t>(g.max_tok, EXL3I_AQ_XB_ISSUE_ROWS);
    m.aq.width_xb = exl3_aq_xb_row_width({ { g.n_embd, 3 }, { g.n_embd, 2 }, { g.n_ff, 1 }, { q_dim, 1 } });

    m.b_hout    = decl_b(b, nm.f("h_out"),    g.act_dtype, {g.max_logit_rows, g.n_embd});

    /* The DFlash2 drafter's taps (qwen35_fp8), only when the container carries the drafter and the deployment
     * speculates: one residual row a tap, copied by `cast` into its column slice after the layer. Declared before
     * the layers, so dflash_src is live from the first tap's write. */
    if (ctx->max_spec > 0) {
        const char* da = rad_meta_gets(meta, "draft.architectures", "");
        if (da && std::strstr(da, "DFlash")) {
            m.n_taps = parse_layer_ids(rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr),
                                       m.n_main, m.tap_layer, Model::MAX_TAPS);
            if (m.n_taps > 0) {
                m.b_dfsrc = decl_b(b, nm.f("dflash_src"), g.act_dtype, {g.max_tok, (int64_t)m.n_taps * g.n_embd});
                m.op_tap = rw(b, RAD_OP(b, "cast",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                             RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                                  RAD_NOWEIGHTS),
                              {m.b_x}, {m.b_dfsrc});
            }
        }
    }

    /* Vocabulary edges */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"),
                          nm.ckpt("model.language_model.norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.language_model.embed_tokens.weight")));

    /* The token table as the container holds it: the checkpoint's bf16 (embed_lookup, sharded by
     * vocabulary rows), or codebook-quantised under fwht128 (recipe: `token_embd.weight rtn table=<2^B
     * values> group=128 scale=bf16 transform=fwht128 rule=search`) for embed_tq, which every rank reads whole
     * (so it is declared at the full n_vocab_all, not the rank's vocab shard). */
    {
        RadEncoding te{};
        if (weight_enc(b, nm.f("token_embd.weight"), &te) && rad_enc_is(&te, "affine") &&
            rad_enc_find(&te, "table") >= 0 && rad_enc_fwht(&te) == 128) {
            const int ic = rad_enc_find(&te, "codes");
            m.tok_bits = ic >= 0 ? (int)rad_dtype_bits(te.plane[ic].dtype) : 0;
            if (m.tok_bits < 1 || m.tok_bits > 8) return RAD_E_DTYPE;
        } else if (weight_enc(b, nm.f("token_embd.weight"), &te) && rad_enc_is(&te, EXL3I_SCHEME)) {
            /* recipe: `token_embd.weight exl3i src=<the embedding's level> codebook=mad [rows=r16]` */
            int st = 0;
            int64_t er = 0, ec = 0;
            if (exl3i_encoding_read_r16(&te, &m.tok_m, &m.tok_cb, &st, &er, &ec, &m.tok_r16) != RAD_OK || st ||
                er != g.n_vocab_all || ec != g.n_embd) {
                fprintf(stderr, "qwen35_exl3: token_embd is exl3i but not a [%lld, %lld] unstacked table (mad, any rate in "
                                "1/8 steps): refused at load\n", (long long)g.n_vocab_all, (long long)g.n_embd);
                return RAD_E_FORMAT;
            }
        }
    }
    if (m.tok_bits) {
        const char* src = nm.f("token_embd.weight");
        char cdt[4];
        std::snprintf(cdt, sizeof cdt, "u%d", m.tok_bits);
        m.w_tok = decl_view(b, nm.f("token_embd.codes"), src, "codes", rad_dtype_parse(cdt),
                            {g.n_vocab_all, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
        m.w_tok_scale = decl_view(b, nm.f("token_embd.scale"), src, "scale", RAD_BF16,
                                  {g.n_vocab_all, g.n_embd / 128}, RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
        m.w_tok_table = decl_view(b, nm.f("token_embd.table"), src, "table", RAD_F32,
                                  {1, (int64_t)1 << m.tok_bits}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    } else if (m.tok_m) {   /* the trellis as stored: every rank reads the whole table, as embed_tq's */
        const char* src = nm.f("token_embd.weight");
        m.w_tok = decl_view(b, nm.f("token_embd.record"), src, "record", RAD_U32,
                            {g.n_vocab_all / 16, exl3i_row_words(g.n_embd, m.tok_m)}, RAD_ACCESS_VOCAB, RAD_SHARD_NONE,
                            grp_model());
        m.w_tok_suh = decl_view(b, nm.f("token_embd.suh"), src, "suh", RAD_F16, {1, g.n_embd},
                                RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
        m.w_tok_svh = decl_view(b, nm.f("token_embd.svh"), src, "svh", RAD_F16, {g.n_vocab_all, 1},
                                RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    } else {
        m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                         RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    }
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    /* LM Head projection. An R16 exl3i head (the release heads: the rows stored permuted) cannot be
     * row-sharded -- a rank's vocab slice is not a slice of the stored rows -- so decl_exl3_weight serves it
     * whole on every rank at world > 1 (wt_head.replicated); the engine's whole-vocab logits layout then
     * reads rank 0 only (KLD) and every rank samples the same full distribution (serve). */
    RAD_ARCH_TRY(decl_exl3_weight(b, nm.f("output.weight"), g.n_vocab, g.n_embd,
                                  RAD_SHARD_ROW, grp_model(), 0, nullptr, &m.wt_head, g.world));
    const int64_t head_vocab = m.wt_head.replicated ? g.n_vocab_all : g.n_vocab;
    m.b_logits  = decl_b(b, nm.f("logits"),   RAD_F32,     {g.max_logit_rows, head_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    if (m.tok_bits) {
        const rad_weight wt[3] = { m.w_tok, m.w_tok_scale, m.w_tok_table };
        m.op_embed = rw(b, RAD_OP(b, "embed_tq",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                       RAD_INT("n_vocab", g.n_vocab_all), RAD_INT("bits", m.tok_bits)),
                            wt, 3),
                        {}, {m.b_x});
    } else if (m.tok_m) {
        const rad_weight wt[3] = { m.w_tok, m.w_tok_suh, m.w_tok_svh };
        m.op_embed = rw(b, RAD_OP(b, "embed_exl3i",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                       RAD_INT("n_vocab", g.n_vocab_all), RAD_INT("m", m.tok_m),
                                       RAD_STR("codebook", exl3i_codebook_name(m.tok_cb)), RAD_INT("r16", m.tok_r16)),
                            wt, 3),
                        {}, {m.b_x});
    } else {
        m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                       RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                       RAD_INT("vocab_offset", g.vocab_off)),
                            RAD_WEIGHTS(m.w_tok)),
                        {}, {m.b_x});
    }

    if (g.world > 1 && !m.tok_bits && !m.tok_m) {   /* embed_tq / embed_exl3i read whole rows on every rank: nothing to sum */
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                           {m.b_x}, {m.b_x});
    }

    /* exl3_fuse.h qk: the rope table and its fill, declared before the layers that read it; without a fill the
     * attention blocks keep their norm + rope + store chain. Its band is the fused prologue's (exl3_qk_rows). */
    if (exl3_fuse_on(EXL3_FUSE_QK) && rad_op_declared(b, "qk_norm_rope_gate_kv_store") &&
        rad_op_declared(b, "rope_table")) {
        m.b_rope_cs = decl_b(b, nm.f("rope_cs"), RAD_F32, {exl3_rope_rows(g), m.n_rot}, RAD_BUF_PERSIST);
        m.op_rope_cs = rw(b, decl_rotary(b, g, "rope_table",
                                 RAD_PARAMS(RAD_RANGE("M", 1, exl3_qk_rows(g)), RAD_INT("rot", m.n_rot),
                                            RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                            RAD_STR("dtype", "f32")),
                                 RAD_NOWEIGHTS),
                          {}, {m.b_rope_cs});
    }

    /* Trunk layers */
    AttnGatedEXL3::Wire aw{};
    aw.x = m.b_x; aw.h = m.b_h; aw.qg = m.b_qg; aw.k = m.b_k; aw.v = m.b_v;
    aw.q = m.b_q; aw.gate = m.b_gate; aw.attn = m.b_attn; aw.aq = &m.aq;
    aw.cos_sin = m.op_rope_cs ? m.b_rope_cs : 0;

    GdnEXL3::Wire gw{};
    gw.x = m.b_x; gw.h = m.b_h; gw.in = m.b_in; gw.a = m.b_ga; gw.b = m.b_gb; gw.ab = m.b_gab; gw.z = m.b_z;
    gw.q = m.b_gq; gw.k = m.b_gk; gw.v = m.b_gv;
    gw.gdec = m.b_gdec; gw.beta = m.b_beta; gw.kkt = m.b_kkt; gw.o = m.b_go; gw.aq = &m.aq;

    MlpEXL3::Wire mw{};
    mw.x = m.b_x; mw.h = m.b_h; mw.gate_up = m.b_mlp_gate_up; mw.ffn = m.b_ffn; mw.aq = &m.aq;

    m.layers.resize((size_t)m.n_main);
    for (int64_t l = 0; l < m.n_main; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.full = is_full_attention(l, interval) ? 1 : 0;

        const char* pre = "model.language_model.layers";
        if (lay.full) {
            AttnGatedEXL3::Src as{};
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
            GdnEXL3::Src gs{};
            gs.norm     = nm.ckpt("%s.%lld.input_layernorm.weight",           pre, (long long)l);
            gs.in_qkv   = nm.ckpt("%s.%lld.linear_attn.in_proj_qkv.weight",   pre, (long long)l);
            gs.in_z     = nm.ckpt("%s.%lld.linear_attn.in_proj_z.weight",     pre, (long long)l);
            gs.conv1d   = nm.ckpt("%s.%lld.linear_attn.conv1d.weight",        pre, (long long)l);
            gs.a_log    = nm.ckpt("%s.%lld.linear_attn.A_log",                pre, (long long)l);
            gs.dt_bias  = nm.ckpt("%s.%lld.linear_attn.dt_bias",              pre, (long long)l);
            gs.out_norm = nm.ckpt("%s.%lld.linear_attn.norm.weight",          pre, (long long)l);
            gs.out      = nm.ckpt("%s.%lld.linear_attn.out_proj.weight",      pre, (long long)l);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_state));
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_conv));
            GdnEXL3::Config lc = gc;
            lc.ab_rows = m.ab_rows[(size_t)l];
            RAD_ARCH_TRY(lay.gdn.declare(b, nm, g, lc, (int)l, m.kv_state, m.kv_conv, gw, gs));
        }

        MlpEXL3::Src ms{};
        ms.norm = nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l);
        ms.gate = nm.ckpt("%s.%lld.mlp.gate_proj.weight",            pre, (long long)l);
        ms.up   = nm.ckpt("%s.%lld.mlp.up_proj.weight",              pre, (long long)l);
        ms.down = nm.ckpt("%s.%lld.mlp.down_proj.weight",            pre, (long long)l);
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms));
    }
    /* exl3_fuse.h qk: no fill when no attention layer's prologue resolved */
    bool qk_any = false;
    for (const Layer& lay : m.layers) qk_any = qk_any || (lay.full && lay.attn.op_qk_kv);
    if (!qk_any) m.op_rope_cs = 0;

    /* Final norm and gather */
    m.op_final_norm = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_F64("wadd", final_wadd), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(m.w_out_norm)),
                         {m.b_x}, {m.b_h});
    m.op_final_addnorm = decl_addnorm(b, g, final_wadd, m.w_out_norm, m.b_x, m.b_h);
    m.defer_add = exl3_fuse_on(EXL3_FUSE_ADDNORM) && rad_op_declared(b, "add_rmsnorm");
    if (m.defer_add)   /* the last block's add, when the final norm is not served fused */
        m.op_final_add = rw(b, RAD_OP(b, "add",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                           RAD_STR("dtype", g.dtype)),
                                RAD_NOWEIGHTS),
                            {m.b_x, m.b_h}, {m.b_x});

    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {m.b_h}, {m.b_hout});

    /* LM Head projection */
    /* An unquantized head writes the f32 logits buffer: logits_gemm, as in the bf16 arch (gemm_nt's output is the
     * activation dtype, which would leave bf16 bytes in an f32 buffer). The issue call is the same for both. */
    if (m.wt_head.is_exl3)
        m.op_logits = decl_exl3_proj(b, g.max_logit_rows, head_vocab, g.n_embd, g.dtype, m.wt_head, m.b_hout, m.b_logits);
    else
        m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                        RAD_INT("n_vocab", g.n_vocab),
                                        RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(m.wt_head.w_bf16)),
                         {m.b_hout}, {m.b_logits});

    /* The DFlash2 drafter at absolute layers n_main.. of its own KV group (n_taps > 0 only when the container
     * names a DFlash drafter and max_spec is positive). It embeds its block through the target's token table, read
     * as a bf16 table, or an exl3i trellis one through embed_exl3i (op_embed_ext): a table-quantised one
     * (embed_tq) is refused by name. */
    if (m.n_taps > 0) {
        if (m.tok_bits) {
            fprintf(stderr, "qwen35_exl3: the DFlash2 drafter embeds through the bf16 or exl3i token table; this "
                            "container's token_embd is table-quantised (embed_tq, %d bits) -- convert it to speculate\n",
                    m.tok_bits);
            return RAD_E_UNSUPPORTED;
        }
        RAD_ARCH_TRY(declare_dflash2(b, meta, ctx, m, (int)m.n_main));
    }

    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    if (!c || !batch) return;
    const int rank = rad_rank(c);
    if (rank < 0 || rank >= MAX_RANKS) return;
    const Model& m = g_model[rank];

    /* An encoder pass runs only the vision tower (RadBatch::enc). */
    if (batch->enc) {
        m.vit.step(c, batch);
        return;
    }
    /* A drafter pass runs only the drafter: positive the query pass (the whole block), negative the context pass
     * (the committed tokens' taps into the drafter's K and V). */
    if (batch->draft_pass != 0) {
        if (!m.have_df) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the container has no "
                             "`draft.` DFlash2, or declare ran at max_spec 0");
            return;
        }
        if (batch->draft_pass > 0) m.df.step(c, batch);
        else                       m.df.context(c, batch);
        return;
    }

    if (m.tok_bits)
        RAD_ISSUE_N(c, m.op_embed, batch->n_tok,
                    praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok), RAD_W(m.w_tok_scale),
                    RAD_W(m.w_tok_table), brows(m.b_x, batch->n_tok));
    else if (m.tok_m)
        RAD_ISSUE_N(c, m.op_embed, batch->n_tok,
                    praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok), RAD_W(m.w_tok_suh),
                    RAD_W(m.w_tok_svh), brows(m.b_x, batch->n_tok));
    else
        RAD_ISSUE(c, m.op_embed,
                  praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
                  brows(m.b_x, batch->n_tok));

    m.mrows.step(c, batch, m.b_x, batch->n_tok);

    if (m.op_embed_ar) {
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);
    }

    /* exl3_fuse.h qk: the rope table's rows for this step, read by every attention layer's fused prologue */
    if (m.op_rope_cs && batch->n_tok <= exl3_qk_rows(m.g) && !rope_mixed(batch))
        RAD_ISSUE_N(c, m.op_rope_cs, batch->n_tok, rope_pos1(batch, batch->n_tok), RAD_B(m.b_rope_cs));

    /* pend: the last block left its residual add (x += h) to the next norm (exl3_fuse.h addnorm) */
    bool pend = false;
    for (int64_t li = 0; li < (int64_t)m.layers.size(); ++li) {
        const Layer& l = m.layers[(size_t)li];
        if (l.full) l.attn.step(c, batch, pend, m.defer_add);
        else        l.gdn.step(c, batch, pend, m.defer_add);
        /* a tapped layer finishes x itself: the tap reads it */
        bool tap = false;
        for (int t = 0; t < m.n_taps; ++t) tap = tap || m.tap_layer[t] == (int)li;
        l.mlp.step(c, batch, m.defer_add, m.defer_add && !tap);
        pend = m.defer_add && !tap;
        /* The tap, after the layer: the drafter conditions on this layer's output, which the next layer overwrites */
        for (int t = 0; t < m.n_taps; ++t) {
            if (m.tap_layer[t] != (int)li) continue;
            RAD_ISSUE(c, m.op_tap, brows(m.b_x, batch->n_tok),
                      bcol(m.b_dfsrc, (int64_t)t * m.g.n_embd, m.g.n_embd, batch->n_tok));
        }
    }

    if (pend && m.op_final_addnorm) {
        RAD_ISSUE(c, m.op_final_addnorm, brows(m.b_x, batch->n_tok), brows(m.b_h, batch->n_tok), RAD_W(m.w_out_norm),
                  brows(m.b_h, batch->n_tok));
    } else {
        if (pend) RAD_ISSUE(c, m.op_final_add, brows(m.b_x, batch->n_tok), brows(m.b_h, batch->n_tok),
                            brows(m.b_x, batch->n_tok));
        RAD_ISSUE(c, m.op_final_norm,
                  brows(m.b_x, batch->n_tok), RAD_W(m.w_out_norm), brows(m.b_h, batch->n_tok));
    }

    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.b_h, batch->n_tok),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        issue_exl3_proj(c, m.op_logits, batch->n_out, m.wt_head, m.b_hout, m.b_logits);
    }
}

} /* namespace qwen35_exl3 */

#ifndef RAD_ARCH_NO_EXPORTS
static const RadPluginInfo kPluginInfo = {
    RAD_PLUGIN_ARCH,
    "qwen35_exl3",
    "1.12.0",
    "Qwen3.5 / 3.6 / 3.8 dense architecture plugin with EXL3 block code",
    ""
};

#include "rad_abi_layout.h"
RAD_PLUGIN_ABI_LAYOUT_EXPORT

extern "C" {

uint32_t rad_plugin_abi_version(void) {
    return RAD_ABI_VERSION;
}

const RadPluginInfo* rad_plugin_info(void) {
    return &kPluginInfo;
}

const char* rad_arch_id(void) {
    return "qwen35";
}

const char* rad_arch_quant(void) {
    return "exl3";
}

/* Reads only exl3 containers: a checkpoint whose quantisation was imported weight by weight is
 * qwen35_fp8's, and a second taker for one would have the engine refuse it
 * (rad_arch_serves_imports, rad_builder.h). */
int rad_arch_serves_imports(void) {
    return 0;
}

int rad_arch_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    return qwen35_exl3::declare(b, meta, ctx);
}

void rad_arch_step(RadCtx* c, const RadBatch* batch) {
    qwen35_exl3::step(c, batch);
}

int rad_arch_probe(const RadModelMeta* meta, RadArchProbe* out) {
    return qwen35_exl3::probe(meta, out);
}

} /* extern "C" */
#endif
