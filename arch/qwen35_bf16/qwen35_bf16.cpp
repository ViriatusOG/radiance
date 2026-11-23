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
 * so a recipe-made block-fp8 head uses the fp8 logits kernel. Likewise each trunk linear: one
 * packed losslessly as `bf16x12` (data/recipes/q38-27b-bf16x12-df2.recipe) is read by gemm_nt at
 * that dtype, which gives the bf16 GEMM's bytes from about three quarters of the weight's
 * (decl_linear_bf16 in rad_arch.h).
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
 *
 * ============================== THE DRAFTER ==============================
 *
 * A container converted with `rad-convert --draft-model` carries a DFlash2 drafter under `draft.`
 * (data/recipes/q38-27b-bf16-df2.recipe), declared when the deployment speculates as qwen35_fp8
 * declares it: a second model at absolute layers n_layer.., reading the residual stream after the
 * layers `dflash_config.target_layer_ids` names (the taps). The drafter only proposes; the trunk
 * verifies every token, so what is emitted is this graph's. The blocks below serve the verify
 * steps (the delta net keeps an anchor state when speculating, rad_block_gdn.h).
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_attn_gated.h>
#include <arch/rad_block_dflash2.h>
#include <arch/rad_block_gdn.h>
#include <arch/rad_block_mlp.h>
#include <arch/rad_block_vit.h>
#include <arch/rad_calib_full.h>

#include <cstring>
#include <vector>

namespace qwen35_bf16 {

using namespace rad::arch;

/* The weight reads beside the decode all-reduces (prefetch_beside, rad_arch.h), at more than one
 * rank. A decode-height all-reduce leaves the card's memory idle for its round trip, so the next
 * GEMM's leading weight bytes are read into the on-die cache then, as qwen35_fp8 and gemma4_bf16
 * do. Each read is issued inside its block, after the block's last GEMM and before its own
 * all-reduce, so it is beside the reduction whether the block or the next norm (ar_rmsnorm) runs
 * it:
 *   kPfGateUpBytes  of gate_up, after the mixer's o or out
 *   kPfNextBytes    of the next layer's first GEMM (q|gate, or the delta net's in; the lm_head
 *                   after the last layer), after down
 *   kPfOutBytes     of the delta net's out-projection, beside its decode recurrent update, which
 *                   runs one workgroup per (sequence, head) and leaves the memory mostly idle; in a
 *                   pure decode step of at most kPfOutSeqs sequences (qwen35_fp8's bound); only
 *                   where launches do not chain (below)
 *   kPfDraftBytes  of each GEMM the DFlash2 drafter's reads name (Dflash2::Config::prefetch),
 *                   qwen35_fp8's size
 * Decode steps only, up to kPrefetchRows rows: one or two sequences' verify. Qwen3.8-27B bf16x12 +
 * DFlash2 at depth 7, TP2 on two R9700s, one stream, seven greedy prompts, two interleaved server
 * starts each: 43.95 ms a step without the reads, 41.76 with these sizes (-5.0 %), the text
 * byte-identical. A verify pass's packed GEMMs took 2.2 ms less and its all-reduces 0.66 ms more
 * (ar_rmsnorm 17.4 -> 22.7 us beside a read); each read starts ~6 us after its GEMM and reads at
 * ~330 GB/s beside the reduction; the drafter's pass took 2.51 -> 2.38 ms. Of the other (gate_up,
 * next, out) sizes in MiB, (12, 12, 8) was within 0.1 ms at more bytes a pass and (12, 6, 15)
 * 0.12 ms slower; in an earlier round (8, 16, 8) and (16, 20, 12) ran 0.44 and 0.34 ms slower than
 * (12, 12, 8): a next read longer than the reduction runs on into the GEMM it feeds.
 *
 * Where launches chain (rad_chain_on) every read goes on lane 0 as the chained form right behind
 * its GEMM (prefetch_beside). There the out read would sit in front of the recurrent update, a
 * plain launch, which waits for it, so the read runs in series with the update instead of beside
 * it; none is declared. The same bf16x12 + DFlash2 deployment, chained, three interleaved server
 * starts each: 41.10 ms a step with the out read, 40.72 without it (-0.38 ms, faster in each of
 * the three rounds), the text byte-identical. */
constexpr int64_t kPfGateUpBytes = (int64_t)12 << 20;
constexpr int64_t kPfNextBytes   = (int64_t)8 << 20;
constexpr int64_t kPfOutBytes    = (int64_t)12 << 20;
constexpr int64_t kPfOutSeqs     = 4;
constexpr int64_t kPfDraftBytes  = (int64_t)12 << 20;

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

    /* ---- the DFlash2 drafter (rad_block_dflash2.h), declared only when the container carries
     * one and the deployment speculates, and the taps it reads: each tap layer's output copied
     * into its column slice of `b_dfsrc` (`fc` is [n_embd, n_taps * n_embd]). */
    bool          have_df = false;
    Dflash2       df{};
    Dflash2::Wire dfw{};
    rad_kvgroup   kv_draft = 0;
    enum { MAX_TAPS = 8 };
    int      n_taps = 0;
    int      tap_layer[MAX_TAPS] = {};
    rad_buf  b_dfsrc = 0;                 /* [max_tok, n_taps * n_embd] */
    rad_op   op_tap = 0;

    std::vector<Layer> layers;

    /* --calib-dir: the whole Hessian of every linear's input over a --calib-corpus run. */
    CalibFull calib;
    int       cs_head = -1;     /* the lm_head's input, recorded with the last layer */
};

/* Declare runs once per rank, in one process, on the rank's own thread (spec §1, §9), so the
 * model state is per instance: indexed by the instance's slot (unique across every engine in the
 * process, rad_build_slot) and read back in step() with rad_slot(). */
static Model g_model[MAX_SLOTS];

/* The DFlash2 drafter: its geometry, KV group, buffers and block, as qwen35_fp8.cpp declares it
 * (each plugin keeps its own copy, as gemma3's does); only the rescoring test differs, for the
 * heads a bf16 container can hold.
 *
 * A second model, not a head: it has its own layer count, head count, feed-forward width, rope
 * base, norm convention and vocabulary claim. Its Geom starts as a copy of the trunk's (n_embd,
 * deployment bounds, rank, wire) and takes everything else from the drafter's config.
 *
 * Nothing is defaulted quietly: a `draft.` config that does not describe a drafter this engine
 * implements is refused by name, because a silently ignored scale factor would only lower
 * acceptance (spec §17). */
static int declare_dflash2(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                           Model& m, int base_layer) {
    Names& nm = m.nm;
    Geom   dg{};
    /* Plain RMS norm: the drafter uses the ordinary norm where the trunk uses GemmaRMSNorm
     * (rad_block_dflash2.h). */
    RAD_ARCH_TRY(drafter_geom(dg, m.g, meta, "DFlash2"));

    Dflash2::Config dc{};
    /* The drafter's reads beside its all-reduces (kPfDraftBytes; the block declares them at more
     * than one rank), none under --weight-prefetch off, as gemma4_bf16 declares them. */
    dc.prefetch = ctx->weight_prefetch_off ? 0 : kPfDraftBytes;
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

    /* The query block must be a power of two: the convolution masks a row's position with
     * `block_size - 1`, so any other value convolves one sequence's block into the next. The shim
     * also rejects it, but only after the drafter's weights have loaded. */
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

    /* Three scalars this implementation does not apply. Each changes the proposals and would show
     * up only as lower acceptance, so a non-default value is refused. */
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

    /* The window bounds the cache: the drafter's 2048-token window, not the context length. */
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
    /* The codes the drafter's six linears read: every activation feeding one has an 8-bit copy
     * and an f32 scale plane. E4M3 with a scale per 128 for block fp8 or MXFP4 linears (rad_fp8.h's
     * DenseFP8), int8 with a scale per 128 or IQ8R's 256 for rotated int8 ones, which the drafter's
     * `fc` tells apart.
     * `srcq` is the largest (50 MiB at a 2048-token chunk and five taps) and halves the bytes of
     * the context pass's dominant GEMM. */
    RAD_ARCH_TRY(Dflash2::probe_i8(b, nm, "draft", &dc.act_i8, &dc.act_g8));
    dc.fc_mapped = true;
    const uint32_t dqt = dc.act_i8 ? RAD_I8 : RAD_F8E4M3;
    const int64_t  dqg = dc.act_i8 ? dc.act_g8 : (int64_t)RAD_FP8_BLOCK;
    if (dc.act_i8 && ((dc.n_taps * dg.n_embd) % dqg || dg.n_embd % dqg || qd % dqg ||
                      dg.n_ff % dqg)) {
        fprintf(stderr, "radiance: a rotated int8 drafter takes widths in whole %lld-column scale groups; "
                        "this one is n_embd %lld, q %lld, n_ff %lld\n", (long long)dqg,
                (long long)dg.n_embd, (long long)qd, (long long)dg.n_ff);
        return RAD_E_INVAL;
    }
    auto dqs = [&](int64_t n) { return (n + dqg - 1) / dqg; };
    dw.srcq    = decl_b(b, nm.f("df_srcq"), dqt, {dg.max_tok, dc.n_taps * dg.n_embd});
    dw.srcs    = decl_b(b, nm.f("df_srcs"), RAD_F32, {dg.max_tok, dqs(dc.n_taps * dg.n_embd)});
    dw.cq      = decl_b(b, nm.f("df_cq"),   dqt,     {dg.max_tok, dg.n_embd});
    dw.cs      = decl_b(b, nm.f("df_cs"),   RAD_F32, {dg.max_tok, dqs(dg.n_embd)});
    dw.h2q     = decl_b(b, nm.f("df_h2q"),  dqt,     {qr, dg.n_embd});
    dw.h2s     = decl_b(b, nm.f("df_h2s"),  RAD_F32, {qr, dqs(dg.n_embd)});
    dw.atq     = decl_b(b, nm.f("df_atq"),  dqt,     {qr, qd});
    dw.ats     = decl_b(b, nm.f("df_ats"),  RAD_F32, {qr, dqs(qd)});
    dw.ffq     = decl_b(b, nm.f("df_ffq"),  dqt,     {qr, dg.n_ff});
    dw.ffs     = decl_b(b, nm.f("df_ffs"),  RAD_F32, {qr, dqs(dg.n_ff)});
    dw.sel_h   = decl_b(b, nm.f("df_selh"), dg.act_dtype, {sr, dg.n_embd});
    dw.hp      = decl_b(b, nm.f("df_hp"),   dg.act_dtype, {sr, dc.sel_rank});
    dw.dq      = decl_b(b, nm.f("df_dq"),   RAD_I8,  {sr, dg.n_embd});
    dw.ds      = decl_b(b, nm.f("df_ds"),   RAD_F32, {sr, 1});
    /* Group-major, [n_embd/group][ceil(M/16)*16]: quant_act_i8's asum plane, whose first
     * dimension is the K group, not the token. */
    dw.dsum    = decl_b(b, nm.f("df_dsum"), RAD_F32, {dg.n_embd / 128, spad});
    /* This rank's plane. `head_rows` is the block's own answer to whether the 2-bit head is
     * vocab-sharded; asking it keeps the buffer consistent with the op declared over it. */
    dw.dlog    = decl_b(b, nm.f("df_dlog"), dg.act_dtype, {sr, Dflash2::head_rows(dg)});
    /* Candidates are rescored against the target's head when logit_rerank can read its plane
     * (bf16 as the checkpoint holds it, logit_rerank_bf16; fp8 or int8 rows, as
     * data/recipes/q38-27b-bf16.recipe's int8 head) and every candidate a rank proposes is one it
     * owns; the planes are then kRerankR wide and the selector reads the first sel_topk. */
    {
        const uint32_t hd = weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16);
        if ((hd == RAD_BF16 || hd == RAD_F8E4M3 || hd == RAD_I8) &&
            (Dflash2::head_sharded(dg) || dg.world <= 1)) {
            m.df.w_lm  = m.w_lm_head;
            m.df.top_w = Dflash2::kRerankR;
        }
    }
    const int64_t tw = m.df.w_lm ? m.df.top_w : dc.sel_topk;
    dw.cand    = decl_b(b, nm.f("df_cand"), RAD_I32, {sr, tw});
    dw.unary   = decl_b(b, nm.f("df_unary"), RAD_F32, {sr, tw});
    /* The vocab-parallel candidate exchange, and only when there is one: this rank's top-R as
     * (int32 id, float value) pairs and the all-gathered world_size * R of them. */
    if (Dflash2::head_sharded(dg)) {
        dw.dpair = decl_b(b, nm.f("df_dpair"), RAD_F32, {sr, tw, 2});
        dw.dgath = decl_b(b, nm.f("df_dgath"), RAD_F32, {sr, (int64_t)dg.world * tw, 2});
    }
    /* The proposal buffer (RadDrafterDecl::proposal): a DFlash2 pass proposes through the
     * selector, never through the sampler's token buffer. */
    dw.tokens  = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {dg.max_seqs, dc.steps()});

    /* --draft-coupling (RadBuildCtx::draft_coupling), as qwen35_fp8 declares it: the walk samples
     * each draft from Q, its scores' softmax over the best eight candidates, and the sampler
     * couples the target's own draw to it (abi/rad_spechub.h). The planes are named as the MTP
     * head's, which is how the engine finds them. Q's temperature keeps the output exact at any
     * value and moves only acceptance; 0.6 is qwen35_fp8's. With logit_rerank above, the walk's
     * unary scores are the bf16 head's, as they are the fp8 head's there. */
    if (ctx->draft_coupling) {
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
        rad_note(b, "DFlash2 coupled: the walk samples x~Q (FP64 softmax of its edge scores / %g "
                    "over the best eight candidates) at each of %lld positions, and the sampler "
                    "couples the target's own draw to it; fixed-seed tokens are NOT preserved",
                 dc.q_temp, (long long)dc.steps());
    }

    RAD_ARCH_TRY(m.df.declare(b, nm, dg, dc, base_layer, m.kv_draft, m.w_tok, "draft",
                              nm.ckpt("lm_head.weight"), dw));
    m.have_df = true;

    /* Block drafter: a context pass mirroring the trunk step, then one pass filling the whole
     * block. `mask_token_id` is read only here and travels in the drafter declaration. `window`
     * must be the one this plugin gated its own attention with, since the scheduler stages the
     * block pass against it. */
    RadDrafterDecl drf{};
    drf.name           = "dflash2";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    /* The anchor row predicts nothing here, so the block is one row longer than the depth and the
     * head runs over rows 1..depth (rad_builder.h, RadDrafterDecl::block). */
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

    rad_note(b, "DFlash2 drafter: %lld layers at absolute %d..%d, %lld heads / %lld kv x %lld, "
                "ffn %lld, %s attention with a %lld-token window, %d trunk tap(s); one forward "
                "proposes %lld tokens through the candidate selector",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1,
             (long long)dg.n_head, (long long)dg.n_head_kv, (long long)dg.head_dim,
             (long long)dg.n_ff, dc.causal ? "causal" : "non-causal", (long long)dc.window,
             m.n_taps, (long long)dc.steps());
    return RAD_OK;
}


static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;

    Model& m = g_model[rad_build_slot(ctx)];
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

    /* The DFlash2 drafter's taps, only when the container carries the drafter and the deployment
     * speculates: one residual stream per tap, 100 MiB at a 2048-token chunk and five taps. `cast`
     * does the copy into a column slice (libr4d/r4d_rows.cpp). */
    if (ctx->max_spec > 0) {
        const char* da = rad_meta_gets(meta, "draft.architectures", "");
        if (da && std::strstr(da, "DFlash")) {
            m.n_taps = parse_layer_ids(
                rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr),
                m.n_main, m.tap_layer, Model::MAX_TAPS);
            if (m.n_taps > 0) {
                m.b_dfsrc = decl_b(b, nm.f("dflash_src"), g.act_dtype,
                                   {g.max_tok, (int64_t)m.n_taps * g.n_embd});
                m.op_tap = rw(b, RAD_OP(b, "cast",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                             RAD_INT("n", g.n_embd),
                                             RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                                  RAD_NOWEIGHTS),
                              {m.b_x}, {m.b_dfsrc});
            }
        }
    }
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

    /* ---- where the residual adds happen. At two ranks with ar_rmsnorm, each block's norm takes
     * the previous block's delta and all-reduce in one launch (rad_norm_bf16.h), placed by
     * layer_fold() as in the fp8 trunk: a mixer folds the previous feed-forward's unless it is
     * layer 0 or follows a drafter tap layer; a feed-forward always folds its mixer's, and keeps
     * its own add on a tap layer and on the last layer, where the tap and the final norm read `x`.
     * Otherwise (one rank, or a kernel library without the op) every block adds its own delta,
     * as before. Byte-identical either way: ar_rmsnorm is all_reduce + add + rmsnorm.
     *
     * Not under calibration (--calib-dir): step() issues each block's input norm ahead of the
     * block for the tap, which a folded norm would turn into a second all-reduce and add. */
    const bool fold_ar = g.world == 2 && rad_op_declared(b, "ar_rmsnorm") && !m.calib.on;

    m.layers.resize((size_t)m.n_main);
    for (int64_t l = 0; l < m.n_main; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.full = is_full_attention(l, interval) ? 1 : 0;
        const LayerFold f = fold_ar ? layer_fold(g, l, m.n_main, m.tap_layer, m.n_taps, false)
                                    : LayerFold{};
        const bool mix_fold = fold_ar && f.mix_fold;

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
            RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv_full, m.n_rot, aw, as,
                                          mix_fold, !fold_ar, mix_fold, fold_ar));
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
            RAD_ARCH_TRY(lay.gdn.declare(b, nm, g, gc, (int)l, m.kv_state, m.kv_conv, gw, gs,
                                         mix_fold, !fold_ar, mix_fold, fold_ar));
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
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms, fold_ar, !fold_ar || f.mlp_add,
                                     fold_ar, fold_ar && f.mlp_ar_out));
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

    /* ---- the weight reads beside the decode all-reduces (kPfGateUpBytes and the rest), after
     * every weight they read is declared. At more than one rank, where there are reductions to sit
     * beside, and none under --weight-prefetch off. Each takes at most its weight's first N x K
     * bytes (this rank's shape): r4d_rad_prefetch clamps a read to its tensor only where the stored
     * dtype has a width, a packed weight's dtype is libr4d's own (rad_dtype_bits answers 0), and
     * a bf16x12 weight's first region, a byte an element, is N x K long. */
    int64_t n_pf = 0;
    if (g.world > 1 && !ctx->weight_prefetch_off && rad_op_declared(b, "prefetch")) {
        auto read = [&](rad_weight w, int64_t want, int64_t n, int64_t k) {
            const rad_op op = decl_prefetch(b, w, std::min(want, n * k));
            n_pf += op != 0;
            return op;
        };
        for (int64_t l = 0; l < m.n_main; ++l) {
            Layer& lay = m.layers[(size_t)l];
            /* gate_up's, issued after the mixer's last GEMM */
            const rad_op gu = read(lay.mlp.w_gate_up, kPfGateUpBytes, 2 * g.n_ff, g.n_embd);
            if (lay.full) { lay.attn.op_pf = gu; lay.attn.pf_w = lay.mlp.w_gate_up; }
            else          { lay.gdn.op_pf  = gu; lay.gdn.pf_w  = lay.mlp.w_gate_up; }
            /* the next layer's first GEMM, issued after down; after the last layer the lm_head */
            if (l + 1 < m.n_main) {
                const Layer& nx = m.layers[(size_t)l + 1];
                lay.mlp.pf_w  = nx.full ? nx.attn.w_qg : nx.gdn.w_in;
                lay.mlp.op_pf = read(lay.mlp.pf_w, kPfNextBytes,
                                     nx.full ? 2 * g.q_dim() : gc.in_dim(), g.n_embd);
            } else {
                lay.mlp.pf_w  = m.w_lm_head;
                lay.mlp.op_pf = read(m.w_lm_head, kPfNextBytes, g.n_vocab, g.n_embd);
            }
            /* the delta net's out-projection, beside its recurrent update; none where launches
             * chain, which would put the read in series with the plain update (above) */
            if (!lay.full && !rad_chain_on()) {
                lay.gdn.op_pf_out   = read(lay.gdn.w_out, kPfOutBytes, g.n_embd, gc.v_dim());
                lay.gdn.pf_out_seqs = kPfOutSeqs;
            }
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

    /* ---- the DFlash2 drafter (rad_block_dflash2.h), at absolute layers n_main.. of its own KV
     * group. `n_taps` is non-zero only when `draft.architectures` names a DFlash model and
     * max_spec is positive. */
    if (m.n_taps > 0) RAD_ARCH_TRY(declare_dflash2(b, meta, ctx, m, (int)m.n_main));

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
    if (n_pf > 0 && rad_chain_on())
        rad_note(b, "%lld weight reads beside the decode all-reduces (the first %lld MiB of "
                    "gate_up, %lld of the next layer's first GEMM; none of the delta net's out, "
                    "since launches chain), on decode steps of up to %lld rows",
                 (long long)n_pf, (long long)(kPfGateUpBytes >> 20),
                 (long long)(kPfNextBytes >> 20), (long long)kPrefetchRows);
    else if (n_pf > 0)
        rad_note(b, "%lld weight reads beside the decode all-reduces and recurrent updates (the "
                    "first %lld MiB of gate_up, %lld of the next layer's first GEMM, %lld of the "
                    "delta net's out), on decode steps of up to %lld rows",
                 (long long)n_pf, (long long)(kPfGateUpBytes >> 20),
                 (long long)(kPfNextBytes >> 20), (long long)(kPfOutBytes >> 20),
                 (long long)kPrefetchRows);
    if (nextn > 0)
        rad_note(b, "the %lld MTP/NextN layer(s) under `mtp.` are not declared: the only drafter "
                    "this plugin runs is a DFlash2 one under `draft.`", (long long)nextn);

    /* The vision tower, last, so its activations share the arena with every buffer above. */
    if (m.media) RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer + 64));
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_slot(c)];

    /* An encoder pass runs only the vision tower (RadBatch::enc). */
    if (batch->enc) {
        m.vit.step(c, batch);
        return;
    }

    /* A drafter pass runs only the drafter: positive is DFlash2's query pass, proposing the whole
     * block at once, and negative its context pass, turning the tokens the trunk just committed
     * into the drafter's own K and V. Zero is a trunk step. */
    if (batch->draft_pass != 0) {
        if (!m.have_df) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container has no `draft.` block, or declare ran at max_spec 0");
            return;
        }
        if (batch->draft_pass > 0) m.df.step(c, batch);
        else                      m.df.context(c, batch);
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
    for (int64_t li = 0; li < (int64_t)m.layers.size(); ++li) {
        const Layer& l = m.layers[(size_t)li];
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

        /* The tap, after the layer: the drafter conditions on this layer's output, which the next
         * layer overwrites, so the rows are copied into this tap's column slice now. */
        for (int t = 0; t < m.n_taps; ++t) {
            if (m.tap_layer[t] != (int)li) continue;
            RAD_ISSUE(c, m.op_tap, brows(m.b_x, batch->n_tok),
                      bcol(m.b_dfsrc, (int64_t)t * m.g.n_embd, m.g.n_embd, batch->n_tok));
        }
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

/* ================================================================== the probe
 *
 * The speculation depth this container's drafter supports, answered before declare because
 * `max_spec` sizes the drafter's buffers, bands kernels and sets the scheduler's window; rad-convert
 * declares at it, which is how the container comes to carry the drafter's weights. DFlash2
 * proposes the whole block in one pass, so its depth is the block length it was trained at
 * (block_size 8: one anchor and 7 mask positions), and any other depth is refused. 0 when the
 * container carries no drafter.
 *
 * Chained launches pay on the plain-bf16 trunk with a drafter and without one: every kernel of
 * its decode step but the gated delta net's two and the vocabulary GEMM has a chained form
 * (libr4d's rad_kernel_chain). Ms a decode step unchained / chained: Qwen3.5-0.8B on one R9700
 * 3.89 / 3.60 at one sequence and 4.72 / 4.46 at four; Qwen3.8-27B as bf16 on two, where the
 * bf16 GEMMs are nearly all of the step, 48.04 / 47.41 and 49.57 / 48.81. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->chain_pays = RAD_CHAIN_ALWAYS;
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    if (da && std::strstr(da, "DFlash2")) {
        const long long bs = rad_meta_geti(meta, "draft.dflash_config.block_size", 8);
        out->draft_depth = bs > 1 ? (int)(bs - 1) : 7;
        out->draft_depth_fixed = 1;   /* trained at exactly this block; any other is a refusal */
    }
    return RAD_OK;
}

}  /* namespace qwen35_bf16 */

#ifndef RAD_ARCH_NO_EXPORTS
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PROBE(qwen35_bf16)
RAD_ARCH_PLUGIN(qwen35_bf16, "qwen35", "", "0.1.0",
                "Qwen3.5 / 3.6 / 3.8 dense (0.8B..27B), bf16 weights and activations, gated "
                "attention over paged KV every fourth layer and gated delta net between")
#endif
