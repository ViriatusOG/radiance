/* rad_norm_bf16.h -- a bf16 block's input norm, which can take the previous block's delta with it.
 *
 * The bf16 blocks (rad_block_attn_gated.h, rad_block_gdn.h, rad_block_mlp.h) each end with
 * `all_reduce(h)` and `add(x, h)` and start with `rmsnorm(x) -> h`. With a fold, the previous block
 * leaves its delta in `h` and issues neither. This norm then issues, at two ranks where the step's
 * all-reduce goes out on the exact wire, `ar_rmsnorm`: the reduce in place, `x += h` and
 * `rmsnorm(x) -> h` in one launch, the three ops' bytes (its replaces chain, libr4d/r4d_fuse.cpp).
 * Elsewhere the previous block issues its own `all_reduce` and this norm issues `add` then
 * `rmsnorm`: the three ops as before.
 *
 * The bf16 counterpart of rad_fp8.h's NormQuantFP8 fold. The previous block asks takes_ar() on the
 * same rows before skipping its collective (its `ar_out`), so both decide on ar_wire_is_exact.
 * Where the folds go is layer_fold()'s rule, as for the fp8 trunk.
 */
#ifndef RAD_NORM_BF16_H
#define RAD_NORM_BF16_H

#include "rad_arch.h"
#include "rad_fp8.h"   /* ar_wire_is_exact */

namespace rad {
namespace arch {

struct NormBF16 {
    Geom       g{};
    rad_weight w = 0;
    rad_buf    x = 0, h = 0;
    rad_op     op_norm = 0;   /* rmsnorm(x) -> h */
    rad_op     op_add  = 0;   /* with a fold, unfused: x += h first */
    rad_op     op_arn  = 0;   /* with a fold at two ranks: ar_rmsnorm(h, x) -> h */

    /* `fold`: take the previous block's delta from `h` into `x`. `fuse_ar`: and its all-reduce,
     * on steps whose wire is exact (two ranks only). */
    int  declare(RadBuilder* b, const Geom& geom, rad_weight gain, rad_buf x_, rad_buf h_,
                 bool fold = false, bool fuse_ar = false);
    /* Asked by the previous block before issuing its own all-reduce. */
    bool takes_ar(int64_t T) const { return op_arn && ar_wire_is_exact(g, T); }
    void step(RadCtx* c, int64_t T) const;
};

inline int NormBF16::declare(RadBuilder* b, const Geom& geom, rad_weight gain, rad_buf x_,
                             rad_buf h_, bool fold, bool fuse_ar) {
    g = geom; w = gain; x = x_; h = h_;
    op_add = op_arn = 0;
    if (fold) {
        op_add = rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {x, h}, {x});
        if (!op_add) return RAD_E_INVAL;
    }
    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w)),
                 {x}, {h});
    if (!op_norm) return RAD_E_INVAL;
    if (fold && fuse_ar && g.world == 2) {
        /* `h` is the message and the output, `x` the residual: both read and written. */
        op_arn = rw(b, RAD_OP(b, "ar_rmsnorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_F64("eps", g.eps), RAD_INT("world_size", g.world),
                                   RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd)),
                        RAD_WEIGHTS(w)),
                    {h, x}, {h, x});
        if (!op_arn) return RAD_E_INVAL;
    }
    return RAD_OK;
}

inline void NormBF16::step(RadCtx* c, int64_t T) const {
    if (takes_ar(T)) {
        RAD_ISSUE_N(c, op_arn, T, brows(h, T), brows(x, T), RAD_W(w), brows(h, T));
        return;
    }
    if (op_add) RAD_ISSUE(c, op_add, brows(x, T), brows(h, T), brows(x, T));
    RAD_ISSUE(c, op_norm, brows(x, T), RAD_W(w), brows(h, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_NORM_BF16_H */
