/* exl3_fuse.h -- the step's small-op fusions: radiance ops libr4d serves that replace a chain of the blocks' own
 * unfused ops, each declared only where a loaded library resolves it (the chain runs otherwise).
 *
 *   addnorm  a block's residual add (x += h) folded into the next block's input norm, or the final norm: add_rmsnorm,
 *            byte-identical to add then rmsnorm (libr4d r4d_fuse.cpp hook). Where the norm itself is folded into AQ's
 *            exl3i_actq (prefill from EXL3I_AQ_MIN_ROWS rows; with mad consumers also the 1-row decode band, form xb)
 *            the add goes into the site too (exl3i version 5: actq's res, add_rmsnorm's bytes; Exl3Aq::declare_res),
 *            or, where exl3i does not serve that, is issued alone before it.
 *   gdn      gdn_conv_update + gdn_recurrent_update at decode: gdn_conv_recurrent_update (byte-identical, hook), up
 *            to 8 tokens a sequence (its window is a register preload).
 *   qk       the attention prologue at decode heights (<= 64 rows, no media rows): q / k norms, the rotary of q and k
 *            and kv_store as one qk_norm_rope_gate_kv_store, reading a cos / sin table one rope_table fills a step.
 *            As qwen35_fp8 serves it (rad_linear.h: asserted byte-equal to the unfused chain in r4d_selftest).
 *   gate     the attention output gate, sigmoid of the gate then its product with the attention output: one act_mul
 *            (act sigmoid), bf16(bf16(sigmoid(g)) * attn) as the pair stores it (libr4d's sigmoid_bf16 and act_mul_bf16
 *            evaluate the same expression). Where AQ's site takes the product (prefill, the mad decode band) the
 *            pair's sigmoid stays.
 *
 * Each is on wherever a loaded library resolves its op; the test pin fuse_off (Exl3TestPins, exl3_dense_op.h) leaves
 * the chosen ones out, so a test can compare a fused step with the unfused chain. */
#ifndef EXL3_FUSE_H
#define EXL3_FUSE_H

#include <arch/rad_arch.h>
#include "exl3_dense_op.h"

#include <cstdlib>
#include <cstring>

namespace qwen35_exl3 {

enum { EXL3_FUSE_ADDNORM = 1, EXL3_FUSE_GDN = 2, EXL3_FUSE_QK = 4, EXL3_FUSE_GATE = 8 };

inline bool exl3_fuse_on(int which) { return !(exl3_test_pins().fuse_off & which); }

/* qk: the fused prologue's row band, shared by the attention blocks, the rope table's fill and the step deciding
 * whether to fill it (qwen35_fp8's qk_fuse_rows: at a prefill chunk the separate kernels already fill the device) */
inline int64_t exl3_qk_rows(const rad::arch::Geom& g) { return g.max_tok < 64 ? g.max_tok : 64; }
/* the rope table's height: a row for every position a step can name, speculative rounds past max_ctx included
 * (qwen35_fp8's rope_table_rows) */
inline int64_t exl3_rope_rows(const rad::arch::Geom& g) { return g.max_ctx + (g.max_spec > 0 ? (int64_t)g.max_spec : 0) + 1; }

/* add_rmsnorm over the residual stream x with the block output h as the addend, the norm written to h (the add
 * reads its row of h before the norm writes it, thread by thread: r4d_model_bf16.hip add_rmsnorm_body). 0 when the
 * fusion is off or no library resolves it. */
inline rad_op decl_addnorm(RadBuilder* b, const rad::arch::Geom& g, double wadd, rad_weight w_norm, rad_buf x,
                           rad_buf h) {
    if (!exl3_fuse_on(EXL3_FUSE_ADDNORM) || !rad_op_declared(b, "add_rmsnorm")) return 0;
    const rad_op o = RAD_OP(b, "add_rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd), RAD_F64("eps", g.eps),
                                       RAD_STR("dtype", g.dtype), RAD_F64("wadd", wadd)),
                            RAD_WEIGHTS(w_norm));
    if (!o || !rad_op_resolved(b, o)) return 0;
    return rad::arch::rw(b, o, { x, h }, { x, h });
}

/* A block's input norm. `pend`: the previous block left its residual add (x += h) to this one -- folded into
 * add_rmsnorm when declared and the norm is not AQ's (Exl3Aq::takes: the A8 rows or the decode band's xb), into AQ's
 * site when the form it takes has the add (Exl3Aq::res_at, exl3i version 5), else issued alone first. Returns the AQ site that served the norm
 * (Exl3Aq::issue), null when the bf16 norm was written to h. */
inline const Exl3Aq* issue_in_norm(RadCtx* c, int64_t T, bool pend, rad_op op_add, rad_op op_addnorm, rad_op op_norm,
                                   const Exl3Aq& aq, rad_buf x, rad_buf h, rad_weight w_norm) {
    using rad::arch::brows;
    if (pend && (aq.takes(T) ? !aq.res_at(T) : !op_addnorm)) {
        RAD_ISSUE(c, op_add, brows(x, T), brows(h, T), brows(x, T));
        pend = false;
    }
    if (const Exl3Aq* s = aq.issue(c, T, pend)) return s;
    if (pend) RAD_ISSUE(c, op_addnorm, brows(x, T), brows(h, T), RAD_W(w_norm), brows(h, T));
    else      RAD_ISSUE(c, op_norm, brows(x, T), RAD_W(w_norm), brows(h, T));
    return nullptr;
}

}  /* namespace qwen35_exl3 */

#endif /* EXL3_FUSE_H */
