/* rad_block_mlp.h -- one SwiGLU feed-forward block, bf16 operands.
 *
 *   rmsnorm -> gate_up gemm -> silu_mul -> down gemm -> [all_reduce] -> residual add
 *
 * gate and up are fused into one weight at convert time (spec §2.4): the map concatenates the
 * checkpoint's gate and up along dim 0 and silu_mul splits [M, 2n] back into gate * up. One GEMM
 * launch and one movement unit instead of two.
 *
 * With a fold (rad_norm_bf16.h) the norm takes the previous block's delta and all-reduce, and
 * `add_out` / `ar_out` leave this block's own to the next block's norm. The defaults are the
 * block as above.
 */
#ifndef RAD_BLOCK_MLP_H
#define RAD_BLOCK_MLP_H

#include "rad_arch.h"
#include "rad_norm_bf16.h"

namespace rad {
namespace arch {

struct MlpBF16 {
    struct Wire {
        rad_buf x       = 0;   /* [max_tok, n_embd] the residual stream */
        rad_buf h       = 0;   /* [max_tok, n_embd] normed input, then the down-projection output */
        rad_buf gate_up = 0;   /* [max_tok, 2 * n_ff] */
        rad_buf ffn     = 0;   /* [max_tok, n_ff] */
    };

    /* Checkpoint-side names for one layer. The plugin supplies them because only it knows both
     * sides of the map (spec §2.4); spellings differ per checkpoint (`...layers.N.mlp.*`,
     * `blk.N.ffn_*`). */
    struct Src {
        const char* norm = nullptr;   /* post_attention_layernorm.weight */
        const char* gate = nullptr;
        const char* up   = nullptr;
        const char* down = nullptr;
    };

    Geom g{};
    Wire w{};
    int  layer = 0;

    rad_weight w_norm = 0, w_gate_up = 0, w_down = 0;
    rad_op op_norm = 0, op_gate_up = 0, op_act = 0, op_down = 0, op_ar = 0, op_add = 0;
    NormBF16 nq{};
    /* The next block's norm takes this block's all-reduce where it can (NormBF16::takes_ar). */
    bool ar_out = false;
    /* Set by the architecture after declare: a `prefetch` of the next GEMM's leading weight bytes
     * (decl_prefetch, rad_arch.h), issued after down and before this block's own all-reduce where
     * it issues one, so it is beside the reduction whether this block or the next norm runs it
     * (prefetch_beside). 0 where none is declared. */
    rad_op     op_pf = 0;
    rad_weight pf_w  = 0;

    /* `fold` / `ar_in`: the norm takes the previous block's delta / and its all-reduce.
     * `add_out`: issue this block's add here. `ar_out_`: leave the all-reduce to the next norm. */
    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                 const Src& src, bool fold = false, bool add_out = true, bool ar_in = false,
                 bool ar_out_ = false);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int MlpBF16::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                            const Src& src, bool fold, bool add_out, bool ar_in, bool ar_out_) {
    g = geom; w = wire; layer = l; ar_out = ar_out_;

    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ffn_norm.weight", l),    src.norm));
    RAD_ARCH_TRY(map_concat(b, nm.f("blk.%d.ffn_gate_up.weight", l), { src.gate, src.up }, 0));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ffn_down.weight", l),    src.down));

    w_norm    = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l),    RAD_F32,  {g.n_embd},
                       RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    const LinearBF16 gu = decl_linear_bf16(b, g, nm.f("blk.%d.ffn_gate_up.weight", l),
                                           {2 * g.n_ff, g.n_embd}, RAD_SHARD_ROW, grp_layer(l),
                                           {g.n_ff, g.n_ff});
    const LinearBF16 dn = decl_linear_bf16(b, g, nm.f("blk.%d.ffn_down.weight", l),
                                           {g.n_embd, g.n_ff}, RAD_SHARD_COL, grp_layer(l));
    w_gate_up = gu.w;
    w_down    = dn.w;

    RAD_ARCH_TRY(nq.declare(b, g, w_norm, w.x, w.h, fold, ar_in));
    op_norm = nq.op_norm;

    op_gate_up = rw(b, RAD_OP(b, "gemm_nt",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", 2 * g.n_ff),
                                   RAD_INT("K", g.n_embd), RAD_STR("dtype", gu.dtype)),
                        RAD_WEIGHTS(w_gate_up)),
                    {w.h}, {w.gate_up});

    op_act = rw(b, RAD_OP(b, "silu_mul",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.gate_up}, {w.ffn});

    op_down = rw(b, RAD_OP(b, "gemm_nt",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                                RAD_INT("K", g.n_ff), RAD_STR("dtype", dn.dtype)),
                     RAD_WEIGHTS(w_down)),
                 {w.ffn}, {w.h});

    /* Column-sharded down projection leaves a partial sum on every rank. The plugin emits the
     * reduction so the graph dump shows it (spec §9). */
    if (g.world > 1)
        op_ar = decl_all_reduce(b, g, w.h);

    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.x, w.h}, {w.x});

    return RAD_OK;
}

inline void MlpBF16::step(RadCtx* c, const RadBatch* batch) const {
    /* Operands carry the step's row count, not the buffer's: kernels read extents from the
     * operands because a ranged parameter is collapsed to a band before issue (rad_arch.h). */
    const int64_t T = batch->n_tok;
    nq.step(c, T);
    RAD_ISSUE(c, op_gate_up, brows(w.h, T),       RAD_W(w_gate_up), brows(w.gate_up, T));
    RAD_ISSUE(c, op_act,     brows(w.gate_up, T), brows(w.ffn, T));
    RAD_ISSUE(c, op_down,    brows(w.ffn, T),     RAD_W(w_down),    brows(w.h, T));
    /* Decode steps only: a prompt's rows never take the reads (rad_arch.h, kPrefetchRows). */
    prefetch_beside(c, op_pf, pf_w, batch->phase == RAD_PHASE_DECODE ? T : INT64_MAX);
    if (op_ar && !(ar_out && ar_wire_is_exact(g, T)))
        RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
    if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_MLP_H */
