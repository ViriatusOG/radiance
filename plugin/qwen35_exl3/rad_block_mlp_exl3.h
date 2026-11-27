/* rad_block_mlp_exl3.h -- SwiGLU feed-forward block, one stacked EXL3 gate_up GEMM.
 *
 * Derived from Radiance's arch/common/rad_block_mlp.h: the checkpoint's gate and up
 * projections are stacked at import (the exl3quant quantiser's ffn_gate_up rule: the gate
 * half in rows [0, n_ff) on the suh plane, the up half in rows [n_ff, 2n_ff) on the suh1
 * plane), served by one dense EXL3 op (N = 2*n_ff, the kernel's stacked path) feeding
 * silu_mul, plus op_down. Keeps RMSNorm, silu_mul, and all-reduce / residual add.
 * The residual add may be left to the next block's norm and this norm may take the previous block's (exl3_fuse.h).
 */
#ifndef RAD_BLOCK_MLP_EXL3_H
#define RAD_BLOCK_MLP_EXL3_H

#include <arch/rad_arch.h>
#include "exl3_dense_op.h"
#include "exl3_fuse.h"

namespace qwen35_exl3 {

using namespace rad::arch;

struct MlpEXL3 {
    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd] the residual stream */
        rad_buf h    = 0;   /* [max_tok, n_embd] normed input, then down output */
        rad_buf gate_up = 0; /* [max_tok, 2*n_ff] the stacked gate | up */
        rad_buf ffn  = 0;   /* [max_tok, n_ff] */
        Exl3AqBuf* aq = nullptr;   /* AQ's shared row buffer (exl3_dense_op.h) */
    };

    struct Src {
        const char* norm = nullptr;   /* post_attention_layernorm.weight */
        const char* gate = nullptr;   /* mlp.gate_proj -- map_concat with up into the stacked weight */
        const char* up   = nullptr;   /* mlp.up_proj */
        const char* down = nullptr;
    };

    Geom g{};
    Wire w{};
    int  layer = 0;

    rad_weight w_norm = 0;
    Exl3DenseWeight wt_gate_up{}, wt_down{};
    Exl3Aq aq_in{}, aq_down{};   /* AQ: the norm for gate_up (its two planes), silu_mul for down */

    rad_op op_norm = 0, op_gate_up = 0, op_act = 0, op_down = 0, op_ar = 0, op_add = 0;
    rad_op op_addnorm = 0;   /* exl3_fuse.h addnorm: the previous block's add and this norm, 0 if not served */

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                 const Src& src);
    /* pend: the previous block left its residual add to this norm; defer: leave this block's to the next */
    void step(RadCtx* c, const RadBatch* batch, bool pend = false, bool defer = false) const;
};

inline int MlpEXL3::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                            const Src& src) {
    g = geom; w = wire; layer = l;

    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_norm.weight", l), src.norm));
    /* The stacked entry is sized by map_concat (spec §2.4: the checkpoint's gate and up
     * concatenated along dim 0); its data is the exl3i import's stacked gate|up. */
    RAD_ARCH_TRY(map_concat(b, nm.f("blk.%d.ffn_gate_up.weight", l), { src.gate, src.up }, 0));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_down.weight", l), src.down));

    w_norm = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_F32, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* stacked gate | up: row-sharded as two parts, so each rank holds its slice of gate AND of up
     * (N = 2 * g.n_ff per rank; the kernel's stacked path reads the suh.1 plane for the up half) */
    const int64_t gu_parts[2] = {g.n_ff, g.n_ff};
    RAD_ARCH_TRY(decl_exl3_weight(b, nm.f("blk.%d.ffn_gate_up.weight", l), 2 * g.n_ff, g.n_embd,
                                  RAD_SHARD_ROW, grp_layer(l), 2, gu_parts, &wt_gate_up, g.world));

    /* down projection: col-sharded, N = g.n_embd, K = g.n_ff */
    RAD_ARCH_TRY(decl_exl3_weight(b, nm.f("blk.%d.ffn_down.weight", l), g.n_embd, g.n_ff,
                                  RAD_SHARD_COL, grp_layer(l), 0, nullptr, &wt_down, g.world));

    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});
    op_addnorm = decl_addnorm(b, g, g.wadd, w_norm, w.x, w.h);

    RAD_ARCH_TRY(aq_in.declare(b, g.max_tok, g.n_embd, "rmsnorm", g.eps, g.wadd, g.act_dtype, {&wt_gate_up}, w.aq, w.x,
                               0, w_norm, 0));
    if (exl3_fuse_on(EXL3_FUSE_ADDNORM)) aq_in.declare_res(b, w.h);   /* the previous block's add (exl3_fuse.h) */
    op_gate_up = decl_exl3_proj(b, g.max_tok, 2 * g.n_ff, g.n_embd, g.dtype, wt_gate_up, w.h, w.gate_up, &aq_in, 0);

    op_act = rw(b, RAD_OP(b, "silu_mul",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.gate_up}, {w.ffn});

    RAD_ARCH_TRY(aq_down.declare(b, g.max_tok, g.n_ff, "silu_mul", 0.0, 0.0, g.act_dtype, {&wt_down}, w.aq, w.gate_up,
                                 0, 0, 0));
    op_down = decl_exl3_proj(b, g.max_tok, g.n_embd, g.n_ff, g.dtype, wt_down, w.ffn, w.h, &aq_down, 0);

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

inline void MlpEXL3::step(RadCtx* c, const RadBatch* batch, bool pend, bool defer) const {
    const int64_t T = batch->n_tok;
    /* AQ: from EXL3I_AQ_MIN_ROWS rows the norm writes gate_up's planes, silu_mul down's */
    const Exl3Aq* aqi = issue_in_norm(c, T, pend, op_add, op_addnorm, op_norm, aq_in, w.x, w.h, w_norm);
    issue_exl3_proj(c, op_gate_up, T, wt_gate_up, w.h, w.gate_up, aqi);
    const Exl3Aq* aqd = aq_down.issue(c, T);
    if (!aqd) RAD_ISSUE(c, op_act, brows(w.gate_up, T), brows(w.ffn, T));
    issue_exl3_proj(c, op_down, T, wt_down, w.ffn, w.h, aqd);
    if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    if (!defer) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h, T), brows(w.x, T));
}

}  /* namespace qwen35_exl3 */

#endif /* RAD_BLOCK_MLP_EXL3_H */
