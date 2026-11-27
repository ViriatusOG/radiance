/* r4d_ar_gather_admit.h -- the all-gather rows' operand admission (ar_gather_2rank / _3rank / _4rank / _8rank:
 * ar_gather_wide_setup in r4d_ar_entry.hip), host-callable, so a test can hold an architecture's issued gather to
 * the rule the card applies at launch without a card (tests/r4d_emu/test_ds41walk.cpp: the DSpark walk's key gather).
 *
 * x is a rank's contribution (numel n, at most the instance's declared bound), y the [world * n] result. A
 * row-interleaved gather (`row` > 0) moves whole rows of `row` elements, which must divide n and fill whole 16-byte
 * words; y may sit in the leading columns of a wider [rows][world * row] buffer and x be read at a wider pitch, each
 * pitch in whole 16-byte words. The message itself must be whole 16-byte words and both ends 16-byte aligned.
 * Returns RAD_OK with the output / input pitches (0 for tight), else the status the launch returns.
 * ar_gather_wide_setup calls this, so the launch and a host-side test apply the same rules. */
#ifndef R4D_AR_GATHER_ADMIT_H
#define R4D_AR_GATHER_ADMIT_H

#include "r4d_args.h"

static inline int r4d_ar_gather_admit(const RadTensor* x, const RadTensor* y, int world, long max_numel, int esize,
                                      long long row, long long* dpitch_out, long long* spitch_out) {
    if (!x || !y) return RAD_E_INVAL;             /* the gather has no in-place form */
    if (y->data == x->data) return RAD_E_INVAL;

    const long n = (long)r4d_numel(x);
    if (n <= 0 || n > max_numel) return RAD_E_SHAPE;
    /* y is [world_size * numel], rank r's contribution at offset r * numel (docs/OPS.md). */
    if ((long long)r4d_numel(y) != (long long)n * world) return RAD_E_SHAPE;

    if (row < 0 || (row > 0 && (n % row || (row * esize) % 16))) return RAD_E_SHAPE;
    /* A row-interleaved gather's [rows][world * row] may sit in the leading columns of a wider
     * buffer, the pitch in whole 16-byte words: rows land at that pitch, the rest of each row is
     * left alone. */
    long long dpitch = 0;
    if (!r4d_contig(y)) {
        const long long p = r4d_row_pitch(y);
        if (row <= 0 || y->rank != 2 || p <= 0 || y->shape[1] != (int64_t)row * world ||
            (p * esize) % 16)
            return RAD_E_STRIDE;
        dpitch = p;
    }
    /* Likewise x may be [rows][row] read at a wider pitch: a rank's window of a row it computed
     * wider (a split linear's block-aligned superset, arch/common/rad_block_dflash2.h). */
    long long spitch = 0;
    if (!r4d_contig(x)) {
        const long long p = r4d_row_pitch(x);
        if (row <= 0 || x->rank != 2 || p <= 0 || x->shape[1] != (int64_t)row ||
            (p * esize) % 16)
            return RAD_E_STRIDE;
        spitch = p;
    }

    if (((long long)n * esize) % 16) return RAD_E_ALIGN;
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(y->data, 16)) return RAD_E_ALIGN;
    if (dpitch_out) *dpitch_out = dpitch;
    if (spitch_out) *spitch_out = spitch;
    return RAD_OK;
}

#endif
