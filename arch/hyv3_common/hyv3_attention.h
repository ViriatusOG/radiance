/* HYV3 full GQA attention with block-scaled FP8 projections and per-head
 * Q/K RMSNorm before NeoX RoPE (the HYV3 attention path).
 *
 *   rmsnorm -> q/k/v gemm_nt_q -> per-head q/k RMSNorm -> rope -> kv_store -> attn_paged
 *   -> quant_act_fp8 -> o gemm_nt_q -> [all_reduce] -> residual add
 *
 * Unlike AttnQ, the projections stay separate so Q and K can be normalized independently before
 * rotation. The weights and all non-attention activations use the same contracts as the other FP8
 * blocks in rad_linear.h.
 */
#ifndef HYV3_ATTENTION_H
#define HYV3_ATTENTION_H

#include <arch/rad_linear.h>

#include <cstdio>
#include <cstring>

namespace rad {
namespace arch {

struct AttnQK {
    struct Wire {
        rad_buf x = 0;       /* [max_tok, n_embd] residual stream */
        ActQ h{};          /* [max_tok, n_embd] normed input, then o output */
        rad_buf q = 0;       /* [max_tok, q_dim], projected, normed and rotated */
        rad_buf k = 0;       /* [max_tok, kv_dim], projected, normed and rotated */
        rad_buf v = 0;       /* [max_tok, kv_dim], projected */
        ActQ attn{};       /* [max_tok, q_dim] attention output */
    };

    struct Src {
        const char* norm = nullptr;
        const char* q = nullptr;
        const char* k = nullptr;
        const char* v = nullptr;
        const char* q_norm = nullptr;
        const char* k_norm = nullptr;
        const char* o = nullptr;
    };

    Geom g{};
    Wire w{};
    int layer = 0;
    rad_kvgroup kv = 0;

    rad_weight w_norm = 0, w_q_norm = 0, w_k_norm = 0;
    LinearQ q{}, k{}, v{}, o{};
    NormQuant nq_h{};
    QuantAct q_attn{};
    rad_op op_q_norm = 0, op_k_norm = 0;
    rad_op op_rope_q = 0, op_rope_k = 0, op_attn = 0, op_kv_store = 0;
    rad_op op_qkr_store = 0; /* Q/K norms, NeoX rotary and BF16 cache store, decode band */
    int64_t qkr_rows = 0;
    rad_op op_ar = 0, op_add = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                const Wire& wire, const Src& src, bool fold = false, bool add_out = true);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int AttnQK::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                              rad_kvgroup group, const Wire& wire, const Src& src,
                              bool fold, bool add_out) {
    g = geom;
    w = wire;
    layer = l;
    kv = group;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();

    if (g.world > g.n_head_kv_all) {
        fprintf(stderr, "radiance: %s declares no KV row span and cannot serve world_size = %d "
                        "with n_head_kv = %lld\n", __func__, g.world,
                (long long)g.n_head_kv_all);
        return RAD_E_UNSUPPORTED;
    }
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l), src.norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l), src.q_norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l), src.k_norm));
    w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_F32, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_q_norm = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_F32, {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_k_norm = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", l), RAD_F32, {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    if (!w_norm || !w_q_norm || !w_k_norm) return RAD_E_INVAL;

    // As in the other pre-norm blocks, absorb a preceding delta into the residual stream.
    // The fused kernel normalises the BF16-rounded sum, not an unrounded FP32 substitute.
    RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                              fold ? w.x : 0));
    RAD_ARCH_TRY(q.declare(b, nm, g, nm.f("blk.%d.attn_q", l), q_dim, g.n_embd,
                           RAD_SHARD_ROW, grp_layer(l), {src.q}, w.h, w.q));
    RAD_ARCH_TRY(k.declare(b, nm, g, nm.f("blk.%d.attn_k", l), kv_dim, g.n_embd,
                           RAD_SHARD_ROW, grp_layer(l), {src.k}, w.h, w.k));
    RAD_ARCH_TRY(v.declare(b, nm, g, nm.f("blk.%d.attn_v", l), kv_dim, g.n_embd,
                           RAD_SHARD_ROW, grp_layer(l), {src.v}, w.h, w.v));

    auto head_norm = [&](int64_t heads, rad_weight wt) {
        return RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * heads), RAD_INT("n", g.head_dim),
                                 RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(wt));
    };
    op_q_norm = rw(b, head_norm(g.n_head, w_q_norm), {w.q}, {w.q});
    op_k_norm = rw(b, head_norm(g.n_head_kv, w_k_norm), {w.k}, {w.k});
    op_rope_q = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head, 0, g.head_dim, "neox"),
                   {w.q}, {w.q});
    op_rope_k = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head_kv, 0, g.head_dim, "neox"),
                   {w.k}, {w.k});

    const rad_op attn = rad_decl_op(b, "attn_paged",
                         RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok),
                                    RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                    RAD_INT("window", 0), RAD_INT("n_head", g.n_head),
                                    RAD_INT("block_size", RAD_KV_BLOCK),
                                    RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                    RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS);
    op_attn = rw(b, attn, {w.q}, {w.attn.x});
    op_kv_store = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     {w.k, w.v}, {});

    /* The standalone chain stays available for prefill, scaled/multicomponent rotary, the int8
     * cache and libraries without the fold. A 128-channel BF16 decode prologue is otherwise five
     * tiny launches; the fused kernel preserves the BF16 norm rounding before the rotary. */
    if (g.head_dim == 128 && !std::strcmp(g.dtype, "bf16") &&
        !std::strcmp(g.kv_dtype, "bf16") &&
        !g.rope_mc && g.yarn.factor == 0.0f &&
        rad_op_declared(b, "qkv_norm_rope_kv_store")) {
        qkr_rows = qk_fuse_rows(g);
        op_qkr_store = RAD_OP(b, "qkv_norm_rope_kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, qkr_rows), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_F64("theta", g.theta), RAD_F64("eps", g.eps),
                                    RAD_F64("scale", g.rope_scale), RAD_STR("mode", "neox"),
                                    RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype),
                                    RAD_INT("v_norm", 0),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_WEIGHTS(w_q_norm, w_k_norm));
        if (!op_qkr_store || !rad_op_resolved(b, op_qkr_store)) op_qkr_store = 0;
        if (op_qkr_store)
            op_qkr_store = rw(b, op_qkr_store, {w.q, w.k, w.v}, {w.q, w.k, w.v});
    }

    RAD_ARCH_TRY(q_attn.declare(b, g, w.attn, q_dim));
    RAD_ARCH_TRY(o.declare(b, nm, g, nm.f("blk.%d.attn_output", l), g.n_embd, q_dim,
                           RAD_SHARD_COL, grp_layer(l), {src.o}, w.attn, w.h.x));
    if (g.world > 1)
        op_ar = decl_all_reduce(b, g, w.h.x);
    if (add_out) op_add = rw(b, RAD_OP(b, "add",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.x, w.h.x}, {w.x});
    return op_attn && op_kv_store && (!add_out || op_add) ? RAD_OK : RAD_E_INVAL;
}

inline void AttnQK::step(RadCtx* c, const RadBatch* batch) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table : nullptr;
    const int32_t* used = kvb ? kvb->seqused : nullptr;
    const int64_t T = batch->n_tok;

    nq_h.step(c, T);
    q.step(c, w.h, w.q, T);
    k.step(c, w.h, w.k, T);
    v.step(c, w.h, w.v, T);
    if (op_qkr_store && T <= qkr_rows) {
        RAD_ISSUE_N(c, op_qkr_store, T, brows(w.q, T), brows(w.k, T), brows(w.v, T),
                    rope_posmc(g, batch, T), RAD_W(w_q_norm), RAD_W(w_k_norm),
                    praw(slot, RAD_I32, T), kv_cache(kv, layer));
    } else {
        RAD_ISSUE_N(c, op_q_norm, T * g.n_head, brows(w.q, T), RAD_W(w_q_norm), brows(w.q, T));
        RAD_ISSUE_N(c, op_k_norm, T * g.n_head_kv, brows(w.k, T), RAD_W(w_k_norm), brows(w.k, T));
        RAD_ISSUE(c, op_rope_q, brows(w.q, T), rope_posmc(g, batch, T));
        RAD_ISSUE(c, op_rope_k, brows(w.k, T), rope_posmc(g, batch, T));
        RAD_ISSUE(c, op_kv_store, brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T),
                  kv_cache(kv, layer));
    }
    RAD_ISSUE_N(c, op_attn, batch->max_q_len, brows(w.q, T), kv_cache(kv, layer),
                praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                praw(used, RAD_I32, batch->n_seq), RAD_NONE, RAD_NONE,
                praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1), brows(w.attn.x, T));
    q_attn.step(c, T);
    o.step(c, w.attn, w.h.x, T);
    if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
    if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h.x, T), brows(w.x, T));
}

}  // namespace arch
}  // namespace rad

#endif /* HYV3_ATTENTION_H */
