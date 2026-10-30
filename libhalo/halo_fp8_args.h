/* Derived from libr4d/r4d_fp8_args.h by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. Op schemas, row names and stored weight layouts are
 * shared with libr4d by design. Everything else in this file is libhalo's own
 * and diverges as it is measured on gfx1151. */
/* halo_fp8_args.h -- operand list of the block-scaled fp8 GEMM family.
 *
 * Every kernel that reads `<linear>.weight` as the checkpoint stores it (F8_E4M3 [N][K] with a
 * BF16 scale on a [ceil(N/128)][ceil(K/128)] grid) implements the same op at a different M and
 * takes the same five operands in the same order. See libhalo/halo_args.h.
 */
#ifndef HALO_FP8_ARGS_H
#define HALO_FP8_ARGS_H

#include "halo.h"
#include "halo_args.h"

enum { F_A = 0, F_ASCALE = 1, F_B = 2, F_BSCALE = 3, F_Y = 4 };
/* ...and for `quant_act_fp8` and `sample_chain`. */
enum { QF_X = 0, QF_Q = 1, QF_S = 2 };
/* The fused forms follow their int8 siblings (`rmsnorm_had_quant_i8`, `gated_had_quant_i8`)
 * minus the rotation's operands, so switching activation format changes op names only. */
enum { RQ_X = 0, RQ_RES = 1, RQ_W = 2, RQ_Q = 3, RQ_S = 4, RQ_Y = 5 };
enum { GQ_GU = 0, GQ_Q = 1, GQ_S = 2, GQ_Y = 3 };
enum { TQ_G = 0, TQ_X = 1, TQ_Q = 2, TQ_S = 3, TQ_Y = 4 };
enum { SC_LOGITS = 0, SC_QUERY = 1, SC_U = 2, SC_DRAW = 3 };

struct Fp8Ops {
    const RadTensor *a, *ascale, *b, *bscale, *y;
    long long M, N, K, nkb;
    long a_ld, sa_ld, w_ld, sw_ld;    /* a_ld / w_ld in elements here; see the note at each use */
};

/* Reads this family's operands and pitches. libhalo takes the weight pitch in bytes and the scale
 * pitch in elements: an E4M3 row is bytes, a bf16 scale row is not. */
/* The weight's scale plane: per 128x128 tile (block E4M3), or per row per 128 K (the int8 twins). */
enum { FP8_OPS_BLOCK_SCALE = 0, FP8_OPS_ROW_SCALE = 1 };

static inline int fp8_ops(const RadArgs* a, Fp8Ops* g, int want_ascale,
                          int wscale = FP8_OPS_BLOCK_SCALE) {
    std::memset(g, 0, sizeof(*g));
    /* No fp8 or int8 block GEMM reads a partitioned weight's `pb1`/`pb2`; refuse rather than
     * ignore (see gemm_ops). */
    if (halo_geti_or(a, "pb1", 0) > 0 || halo_geti_or(a, "pb2", 0) > 0) return HALO_NO(RAD_E_UNSUPPORTED);
    g->a      = halo_opt(a, F_A);
    g->ascale = halo_opt(a, F_ASCALE);
    g->b      = halo_opt(a, F_B);
    g->bscale = halo_opt(a, F_BSCALE);
    g->y      = halo_opt(a, F_Y);
    if (!g->a || !g->b || !g->bscale || !g->y) return HALO_NO(RAD_E_INVAL);
    if (want_ascale && !g->ascale) return HALO_NO(RAD_E_INVAL);

    if (g->b->rank != 2 || g->bscale->rank != 2) return HALO_NO(RAD_E_SHAPE);
    /* Rows are dim 0 and everything after it is the row, as in halo_rad_quant_act_fp8. A gated
     * attention q|gate projection is declared [tokens, heads, 2*head_dim] for the de-interleave
     * that follows; to the GEMM its N is heads*2*head_dim, so rank > 2 must be accepted. */
    if (g->y->rank < 2) return HALO_NO(RAD_E_SHAPE);
    g->M = g->y->shape[0];
    if (g->M < 1 || halo_numel(g->y) % g->M) return HALO_NO(RAD_E_SHAPE);
    g->N = halo_numel(g->y) / g->M;
    g->K = g->b->shape[1];
    if (g->b->shape[0] != g->N) return HALO_NO(RAD_E_SHAPE);
    if (g->N < 1 || g->K < 1) return HALO_NO(RAD_E_SHAPE);

    /* The layout hook and the .rad it wrote use HALO_FP8_BLOCK and libhalo compiles its own copy;
     * a mismatch would be a silently wrong scale grid. */
    const long long blk = halo_gemm_fp8a8_block();
    if (blk != HALO_FP8_BLOCK) return HALO_NO(RAD_E_SHAPE);
    g->nkb = (g->K + blk - 1) / blk;
    /* Check the scale plane's extents: a wrong grid still reads plausible numbers. */
    if (g->bscale->shape[0] != (wscale == FP8_OPS_ROW_SCALE ? g->N : (g->N + blk - 1) / blk))
        return HALO_NO(RAD_E_SHAPE);
    if (g->bscale->shape[1] != g->nkb) return HALO_NO(RAD_E_SHAPE);

    const long long wp = halo_row_pitch(g->b), sp = halo_row_pitch(g->bscale);
    const long long yp = halo_row_pitch(g->y);
    if (wp < 0 || sp < 0 || yp < 0) return HALO_NO(RAD_E_STRIDE);
    /* The output has no pitch argument -- libhalo writes C[r*N + c]. */
    if (yp != g->N) return HALO_NO(RAD_E_STRIDE);
    g->w_ld = wp;      /* elements == bytes for E4M3 */
    g->sw_ld = sp;

    if (g->a->rank < 2) return HALO_NO(RAD_E_SHAPE);
    if (g->a->shape[0] != g->M || halo_numel(g->a) != g->M * g->K) return HALO_NO(RAD_E_SHAPE);
    const long long ap = halo_row_pitch(g->a);
    if (ap < 0) return HALO_NO(RAD_E_STRIDE);
    g->a_ld = ap;
    if (g->ascale) {
        if (g->ascale->rank != 2) return HALO_NO(RAD_E_SHAPE);
        if (g->ascale->shape[0] != g->M || g->ascale->shape[1] != g->nkb) return HALO_NO(RAD_E_SHAPE);
        const long long asp = halo_row_pitch(g->ascale);
        if (asp < 0) return HALO_NO(RAD_E_STRIDE);
        g->sa_ld = asp;
    }
    /* 16 bytes: the widest weight load in the family is a uint4 (matvec) or a 32-byte staged pair
     * (GEMM), both at a 16-byte offset from the row base. */
    if (!halo_aligned(g->a->data, 16) || !halo_aligned(g->b->data, 16)) return HALO_NO(RAD_E_ALIGN);
    return RAD_OK;
}


#endif  /* HALO_FP8_ARGS_H */
