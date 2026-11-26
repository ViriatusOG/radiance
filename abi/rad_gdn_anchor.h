/* rad_gdn_anchor.h -- the layout of a gated-delta-net state slot under the `anchor` state form.
 *
 * docs/OPS.md, `state_form`: a speculating sequence keeps one committed state (the anchor) and,
 * in a second slot of the same shape, one replay record per verified row: the row's normalised
 * key once per 32-row group of head_v, its raw value row, and its decay exponent and beta once
 * per group. The next step replays the first `num_accepted` records onto the anchor.
 *
 * Three parties read this layout and must agree: libr4d's kernels
 * (r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip), libref's reference
 * (libref/ref_gdn.cpp, AnchorLayout) and the core's acceptance of a tree verify
 * (rad_drafter_tree, core/sched/advance_impl.h, ADV_COMPACT), which moves the accepted path's
 * records to the front of the block so the next step's replay reads a chain. Everything here is
 * in elements of the state's dtype, inside one head's [head_v, head_k] block of the factor slot:
 *
 *     record t at   t * fstr
 *       [0, groups*K)              the normalised k, one copy per 32-row group
 *       [groups*K, groups*K + V)   the raw v row
 *       [groups*K + V, +2*groups)  (decay exponent, beta) per group
 *     the shift marker at tmax * fstr, past the last record
 *
 * libr4d's K = V = 128 kernel has fstr 648, tmax 16 and the marker at 10368. */
#ifndef RAD_GDN_ANCHOR_H
#define RAD_GDN_ANCHOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline int64_t rad_gdn_anchor_groups(int64_t head_v) { return (head_v + 31) / 32; }

/* Elements of one record. */
static inline int64_t rad_gdn_anchor_fstr(int64_t head_k, int64_t head_v) {
    const int64_t g = rad_gdn_anchor_groups(head_v);
    return g * head_k + head_v + 2 * g;
}

/* Records a head's block holds: 16, or as many as fit before the marker on a small head. */
static inline int64_t rad_gdn_anchor_tmax(int64_t head_k, int64_t head_v) {
    const int64_t fstr = rad_gdn_anchor_fstr(head_k, head_v);
    const int64_t fit = fstr > 0 ? (head_v * head_k - 1) / fstr : 0;
    return fit < 16 ? fit : 16;
}

/* The marker's element offset. */
static inline int64_t rad_gdn_anchor_mark(int64_t head_k, int64_t head_v) {
    return rad_gdn_anchor_tmax(head_k, head_v) * rad_gdn_anchor_fstr(head_k, head_v);
}

#ifdef __cplusplus
}
#endif
#endif  /* RAD_GDN_ANCHOR_H */
