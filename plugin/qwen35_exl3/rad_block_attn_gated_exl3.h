/* rad_block_attn_gated_exl3.h -- gated grouped-query attention with EXL3 dense projections.
 *
 * Derived from Radiance's arch/common/rad_block_attn_gated.h:
 * Replaces projection gemm_nt declarations (op_qg, op_kp, op_vp, op_o) with exl3i_dense.
 * Keeps RMSNorm, RoPE (M-RoPE partial rotary), Q/K head norms, sigmoid/gate, kv_store, attn_paged.
 * At decode heights the head norms, the rotary and kv_store may run as one qk_norm_rope_gate_kv_store (exl3_fuse.h qk),
 * and the output gate's sigmoid and product as one act_mul (exl3_fuse.h gate).
 */
#ifndef RAD_BLOCK_ATTN_GATED_EXL3_H
#define RAD_BLOCK_ATTN_GATED_EXL3_H

#include <arch/rad_arch.h>
#include "exl3_dense_op.h"
#include "exl3_fuse.h"

namespace qwen35_exl3 {

using namespace rad::arch;

struct AttnGatedEXL3 {
    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]              the residual stream */
        rad_buf h    = 0;   /* [max_tok, n_embd]              normed input, then the o output */
        rad_buf qg   = 0;   /* [max_tok, n_head, 2*head_dim]  query and gate, interleaved */
        rad_buf k    = 0;   /* [max_tok, kv_dim] */
        rad_buf v    = 0;   /* [max_tok, kv_dim] */
        rad_buf q    = 0;   /* [max_tok, q_dim]   de-interleaved, normed, rotated */
        rad_buf gate = 0;   /* [max_tok, q_dim]   sigmoid of the de-interleaved gate */
        rad_buf attn = 0;   /* [max_tok, q_dim]   attention output, then gated in place */
        Exl3AqBuf* aq = nullptr;   /* AQ's shared row buffer (exl3_dense_op.h) */
        rad_buf cos_sin = 0;  /* [exl3_rope_rows, rot] f32, filled once a step by the model's rope_table; 0: no qk fusion */
    };

    struct Src {
        const char* norm   = nullptr;   /* input_layernorm.weight */
        const char* qg     = nullptr;   /* self_attn.q_proj.weight -- query AND gate */
        const char* k      = nullptr;
        const char* v      = nullptr;
        const char* q_norm = nullptr;
        const char* k_norm = nullptr;
        const char* o      = nullptr;
    };

    Geom        g{};
    Wire        w{};
    int         layer   = 0;
    rad_kvgroup kv      = 0;
    int64_t     n_rot   = 0;
    const char* mode    = "mrope";

    rad_weight w_norm = 0, w_q_norm = 0, w_k_norm = 0;
    Exl3DenseWeight wt_qg{}, wt_k{}, wt_v{}, wt_o{};
    Exl3Aq aq_in{}, aq_o{};   /* AQ: the norm for q / k / v, the gate's mul for o */

    rad_op op_norm = 0, op_qg = 0, op_kp = 0, op_vp = 0;
    rad_op op_q_norm = 0, op_k_norm = 0, op_rope_q = 0, op_rope_k = 0;
    rad_op op_attn = 0, op_kv_store = 0, op_sigmoid = 0, op_gate = 0;
    rad_op op_o = 0, op_ar = 0, op_add = 0;
    rad_op op_addnorm = 0;   /* exl3_fuse.h addnorm: the previous block's add and this norm, 0 if not served */
    /* exl3_fuse.h qk: q / k norms, rotary and kv_store in one launch up to qk_rows rows, 0 if not served */
    rad_op  op_qk_kv = 0;
    int64_t qk_rows = 0;
    rad_op  op_gate_act = 0;   /* exl3_fuse.h gate: sigmoid(gate) * attn in one launch, 0 if not served */

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                 int64_t rotary_dim, const Wire& wire, const Src& src);
    /* pend: the previous block left its residual add to this norm; defer: leave this block's to the next */
    void step(RadCtx* c, const RadBatch* batch, bool pend = false, bool defer = false) const;
};

inline int AttnGatedEXL3::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                                  rad_kvgroup group, int64_t rotary_dim, const Wire& wire,
                                  const Src& src) {
    g = geom; w = wire; layer = l; kv = group; n_rot = rotary_dim;

    const int64_t q_dim   = g.q_dim();
    const int64_t kv_dim  = g.kv_dim();
    const int64_t qg_dim  = 2 * q_dim;

    if (n_rot <= 0 || n_rot > g.head_dim || (n_rot & 1)) {
        fprintf(stderr, "radiance: rotary_dim %lld is not an even width inside head_dim %lld\n",
                (long long)n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    if (g.world > g.n_head_kv_all) {
        fprintf(stderr, "radiance: %s declares no KV row span and cannot serve world_size = %d "
                        "with n_head_kv = %lld; the KV-head replication in geom_from needs the "
                        "span passed to its k/v weights.\n", __func__, g.world,
                (long long)g.n_head_kv_all);
        return RAD_E_UNSUPPORTED;
    }

    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l),   src.norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_qg.weight", l),     src.qg));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k.weight", l),      src.k));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_v.weight", l),      src.v));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l), src.q_norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l), src.k_norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_output.weight", l), src.o));

    w_norm   = decl_w(b, nm.f("blk.%d.attn_norm.weight", l),   RAD_F32,  {g.n_embd},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* EXL3 projection weights */
    RAD_ARCH_TRY(decl_exl3_weight(b, nm.f("blk.%d.attn_qg.weight", l), qg_dim, g.n_embd,
                                  RAD_SHARD_ROW, grp_layer(l), 0, nullptr, &wt_qg, g.world));
    RAD_ARCH_TRY(decl_exl3_weight(b, nm.f("blk.%d.attn_k.weight", l), kv_dim, g.n_embd,
                                  RAD_SHARD_ROW, grp_layer(l), 0, nullptr, &wt_k, g.world));
    RAD_ARCH_TRY(decl_exl3_weight(b, nm.f("blk.%d.attn_v.weight", l), kv_dim, g.n_embd,
                                  RAD_SHARD_ROW, grp_layer(l), 0, nullptr, &wt_v, g.world));
    RAD_ARCH_TRY(exl3_act_same("attn qg / k / v", l, {&wt_qg, &wt_k, &wt_v}));

    w_q_norm = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_F32,  {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_k_norm = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", l), RAD_F32,  {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    RAD_ARCH_TRY(decl_exl3_weight(b, nm.f("blk.%d.attn_output.weight", l), g.n_embd, q_dim,
                                  RAD_SHARD_COL, grp_layer(l), 0, nullptr, &wt_o, g.world));

    /* ops */
    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});
    op_addnorm = decl_addnorm(b, g, g.wadd, w_norm, w.x, w.h);

    RAD_ARCH_TRY(aq_in.declare(b, g.max_tok, g.n_embd, "rmsnorm", g.eps, g.wadd, g.act_dtype, {&wt_qg, &wt_k, &wt_v},
                               w.aq, w.x, 0, w_norm, 0));
    if (exl3_fuse_on(EXL3_FUSE_ADDNORM)) aq_in.declare_res(b, w.h);   /* the previous block's add (exl3_fuse.h) */
    op_qg = decl_exl3_proj(b, g.max_tok, qg_dim, g.n_embd, g.dtype, wt_qg, w.h, w.qg, &aq_in,
                           Exl3Aq::plane_of({&wt_qg, &wt_k, &wt_v}, &wt_qg));
    op_kp = decl_exl3_proj(b, g.max_tok, kv_dim, g.n_embd, g.dtype, wt_k, w.h, w.k, &aq_in,
                           Exl3Aq::plane_of({&wt_qg, &wt_k, &wt_v}, &wt_k));
    op_vp = decl_exl3_proj(b, g.max_tok, kv_dim, g.n_embd, g.dtype, wt_v, w.h, w.v, &aq_in,
                           Exl3Aq::plane_of({&wt_qg, &wt_k, &wt_v}, &wt_v));

    op_q_norm = rw(b, RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                  RAD_INT("n", g.head_dim),
                                  RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(w_q_norm)),
                   {w.qg}, {w.q});
    op_k_norm = rw(b, RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head_kv),
                                  RAD_INT("n", g.head_dim),
                                  RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(w_k_norm)),
                   {w.k}, {w.k});

    op_rope_q = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head, 0, n_rot, mode),
                   {w.q}, {w.q});
    op_rope_k = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head_kv, 0, n_rot, mode),
                   {w.k}, {w.k});

    op_attn = rw(b, RAD_OP(b, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", 0),
                                RAD_INT("n_head", g.n_head), RAD_INT("block_size", RAD_KV_BLOCK), RAD_INT("max_seqs", g.max_seqs),
                                RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {w.q}, {w.attn});

    op_kv_store = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     {w.k, w.v}, {});

    /* exl3_fuse.h qk: the prologue with the cache store folded in, declared after attn_paged (the cache block size is
     * read back from it). The kernel rotates NeoX pairs from a table filled at one rotary component a token, exact when
     * the components agree: step() takes it only then. No gate_out: sigmoid reads the gate in qg as before. */
    const bool rope_neox = !std::strcmp(mode, "neox") || !std::strcmp(mode, "mrope") || g.rope_mc;
    if (w.cos_sin && rope_neox && exl3_fuse_on(EXL3_FUSE_QK) && rad_op_declared(b, "qk_norm_rope_gate_kv_store")) {
        const int64_t rows = exl3_qk_rows(g);
        const rad_op o = RAD_OP(b, "qk_norm_rope_gate_kv_store",
                                RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", g.head_dim),
                                           RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", g.n_head_kv),
                                           RAD_INT("rot", n_rot), RAD_STR("q_dtype", g.dtype), RAD_F64("eps", g.eps),
                                           RAD_F64("wadd", g.wadd_qk), RAD_INT("cs_f32", 1),
                                           RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                           RAD_STR("kv_dtype", g.kv_dtype)),
                                RAD_WEIGHTS(w_q_norm, w_k_norm));
        if (o && rad_op_resolved(b, o)) {
            op_qk_kv = rw(b, o, {w.qg, w.k, w.cos_sin, w.v}, {w.q, w.k});
            qk_rows = rows;
        }
    }

    op_sigmoid = rw(b, RAD_OP(b, "sigmoid",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                   RAD_INT("n", g.head_dim), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.qg}, {w.gate});

    op_gate = rw(b, RAD_OP(b, "mul",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                 {w.attn, w.gate}, {w.attn});

    /* exl3_fuse.h gate: (token, head) rows of head_dim as the sigmoid's, the gate read in place in qg, the product
     * written over the attention output as mul writes it */
    if (exl3_fuse_on(EXL3_FUSE_GATE) && rad_op_declared(b, "act_mul")) {
        const rad_op o = RAD_OP(b, "act_mul",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head), RAD_INT("n", g.head_dim),
                                           RAD_STR("act", "sigmoid"), RAD_STR("dtype", g.dtype)),
                                RAD_NOWEIGHTS);
        if (o && rad_op_resolved(b, o)) op_gate_act = rw(b, o, {w.qg, w.attn}, {w.attn});
    }

    RAD_ARCH_TRY(aq_o.declare(b, g.max_tok, q_dim, "mul", 0.0, 0.0, g.act_dtype, {&wt_o}, w.aq, w.attn, w.gate, 0, 0));
    op_o = decl_exl3_proj(b, g.max_tok, g.n_embd, q_dim, g.dtype, wt_o, w.attn, w.h, &aq_o, 0);

    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h}, {w.h});

    op_add = rw(b, RAD_OP(b, "add",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.x, w.h}, {w.x});

    return RAD_OK;
}

inline void AttnGatedEXL3::step(RadCtx* c, const RadBatch* batch, bool pend, bool defer) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    const int64_t T = batch->n_tok;

    /* AQ: from EXL3I_AQ_MIN_ROWS rows the norm writes q / k / v's planes, not h */
    const Exl3Aq* aqi = issue_in_norm(c, T, pend, op_add, op_addnorm, op_norm, aq_in, w.x, w.h, w_norm);
    issue_exl3_proj(c, op_qg, T, wt_qg, w.h, w.qg, aqi);
    issue_exl3_proj(c, op_kp, T, wt_k,  w.h, w.k, aqi);
    issue_exl3_proj(c, op_vp, T, wt_v,  w.h, w.v, aqi);

    if (op_qk_kv && T <= qk_rows && !rope_mixed(batch)) {
        RAD_ISSUE_N(c, op_qk_kv, T,
                    brows(w.qg, T), brows(w.k, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_W(w_q_norm), RAD_W(w_k_norm),
                    brows(w.q, T), brows(w.k, T), RAD_NONE,
                    brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));
    } else {
        RAD_ISSUE_N(c, op_q_norm, T * g.n_head,
                    bcol(w.qg, 0, g.head_dim, T), RAD_W(w_q_norm), brows(w.q, T));
        RAD_ISSUE_N(c, op_k_norm, T * g.n_head_kv,
                    brows(w.k, T), RAD_W(w_k_norm), brows(w.k, T));
        RAD_ISSUE(c, op_rope_q, brows(w.q, T), rope_posmc(g, batch, T));
        RAD_ISSUE(c, op_rope_k, brows(w.k, T), rope_posmc(g, batch, T));

        RAD_ISSUE(c, op_kv_store,
                  brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));
    }

    RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                brows(w.q, T), kv_cache(kv, layer),
                praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                praw(used, RAD_I32, batch->n_seq),
                RAD_NONE, RAD_NONE,
                praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                brows(w.attn, T));

    /* AQ: the gate's product goes straight to o's plane; else one act_mul where served, else sigmoid then mul */
    const Exl3Aq* aqo = nullptr;
    if (op_gate_act && !aq_o.takes(T)) {
        RAD_ISSUE_N(c, op_gate_act, T * g.n_head,
                    bcol(w.qg, g.head_dim, g.head_dim, T), brows(w.attn, T), brows(w.attn, T));
    } else {
        RAD_ISSUE_N(c, op_sigmoid, T * g.n_head,
                    bcol(w.qg, g.head_dim, g.head_dim, T), brows(w.gate, T));
        aqo = aq_o.issue(c, T);
        if (!aqo) RAD_ISSUE(c, op_gate, brows(w.attn, T), brows(w.gate, T), brows(w.attn, T));
    }

    issue_exl3_proj(c, op_o, T, wt_o, w.attn, w.h, aqo);
    if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    if (!defer) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h, T), brows(w.x, T));
}

}  /* namespace qwen35_exl3 */

#endif /* RAD_BLOCK_ATTN_GATED_EXL3_H */
