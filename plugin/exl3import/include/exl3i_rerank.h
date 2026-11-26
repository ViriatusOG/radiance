/* exl3i_rerank.h -- `exl3i_rerank`, logit_rerank over an exl3i (EXL3-V3 trellis) head read where it lies: the drafter's
 * candidates rescored against the trunk's own output.weight, no copy of the head. Header-only C; the op
 * lives in the exl3i kernel library (plugin/exl3i: rerank.c, exl3i_rerank.hip), next to exl3i_dense, and reads the
 * head's record in the form exl3i_dense's layout hook stores it (one stored copy: the same tags).
 *
 * THE OP. logit_rerank's semantics (libref ref_logit_rerank) with the head's value the logit exl3i_dense's decode form
 * (nt_pass at 1 .. 16 rows, act f16, or a mad head's A8 rows below the A8 threshold, which take nt_pass too
 * past the GEMV's 1 .. 2 rows) writes at that vocabulary row:
 *
 *   per row m:  xr = f16(had128(f16(x[m] x suh)) x 1/sqrt(128))         (rotate128 of exl3r4d_common.hip.h: the stages h = 1 .. 64
 *                                                                          in f32, each exact a +- b)
 *   per candidate id = idx_in[m, r] (global; local = id - vocab_off, a pad when outside [0, N)):
 *     s = the stored row of local (R16 order: (local % (N/8)) / 16 x 128 + local / (N/8) x 16 + local % 16, else
 *         local), b = s / 128, j = s % 128
 *     z_i (the 128 rows i of block b) = sum over k of v(i, k) xr_k, in f32: v the decoded trellis value -- for the
 *         the int8-grid codebook (mad) its grid index q, the sum then x the grid step (fp16 0x26EE), as nt_pass's
 *         DQ. Each product is exact in f32; the order is fixed (below)
 *     val = ((sum over i of (-1)^popcount(i & j) z_i) x 1/sqrt(128)) x svh[s]    (the output rotation's element j)
 *   then the R (id, val) re-sorted descending, ties to the lower id, pads (id -1, val -inf) last: idx, val, and the
 *   pair plane (f32 [M, R, 2], id bits then value) when given.
 *
 * Exactness: the same xr, the same products and the same epilogue form as exl3i_dense's decode form; only the f32
 * summation order of the 5120-term dots and of the 128-point output sum differ (nt_pass's WMMA / KS / EPIC order), so
 * the value equals exl3i_dense's logit at that row to f32 rounding (tests/test_rerank.hip measures it). The device row
 * and the host reference (exl3i_rerank_ref) use one order and agree bit for bit:
 *   z_i = (A_0 + A_1) [x step], A_h = fma-chained over the 128-feature chunks c = 0 .. K/128 - 1, its four tiles
 *         tk = 8 c + 4 h .. 8 c + 4 h + 3, each tile's 16 k rows rt = 0 .. 15 (feature 16 tk + rt): A = fmaf(v, xr, A);
 *   the output sum: adjacent pairs first, u_i = z_2i +- z_2i+1 (sign by bit 0 of j), then pairs of those (bit 1), ...
 *         to bit 6 -- the cone of element j of exl3_had128's in-place butterfly.
 *
 *   params    M (rows, a range), R (candidates a row), N (n_vocab rows of this rank's head), K (n_embd), rate (exl3i's
 *             canonical text), codebook (mad), vocab_off (optional, 0)
 *   operands  x (in bf16 | f16 [M, K]), record / suh / svh (weights: exl3i_dense's planes and positions, its stored
 *             form), idx_in (in i32 [M, R]), idx (out i32 [M, R]), val (out f32 [M, R]), pairs (out, optional,
 *             f32 [M, R, 2]). idx may be idx_in (in place) and val may be any f32 plane of [M, R].
 * The R16 row order is the record's stored dtype (EXL3I_DT_TILES_R16, plugin/exl3i/exl3i.h), as for exl3i_dense. */
#ifndef EXL3I_RERANK_H
#define EXL3I_RERANK_H

#include "rad_abi.h"
#include "rad_plugin.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EXL3I_OP_RERANK "exl3i_rerank"
#define EXL3I_RERANK_VERSION 1
#define EXL3I_RERANK_MAX_R 256   /* candidates a row the device row takes */

/* operand positions: record / suh / svh at exl3i_dense's (EXL3I_OPD_RECORD .. EXL3I_OPD_SVH), so its layout hooks read
 * the same operand numbers */
enum { EXL3I_RR_X = 0, EXL3I_RR_RECORD = 1, EXL3I_RR_SUH = 2, EXL3I_RR_SVH = 3, EXL3I_RR_IDX_IN = 4, EXL3I_RR_IDX = 5,
       EXL3I_RR_VAL = 6, EXL3I_RR_PAIRS = 7, EXL3I_RR_N_OPD = 8 };

static inline const RadOpSchema* exl3i_rerank_schema(void) {
    static const RadParamSpec p[] = {
        { "M",         RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "R",         RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "N",         RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "K",         RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "rate",      RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE },
        { "codebook",  RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE },
        { "vocab_off", RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE },
    };
    static const RadOperandSpec o[EXL3I_RR_N_OPD] = {
        { "x",      RAD_OPD_IN,     0 },
        { "record", RAD_OPD_WEIGHT, 0 },
        { "suh",    RAD_OPD_WEIGHT, 0 },
        { "svh",    RAD_OPD_WEIGHT, 0 },
        { "idx_in", RAD_OPD_IN,     0 },
        { "idx",    RAD_OPD_OUT,    0 },
        { "val",    RAD_OPD_OUT,    0 },
        { "pairs",  RAD_OPD_OUT,    1 },
    };
    static const RadOpSchema s = {
        EXL3I_OP_RERANK, p, (int)(sizeof p / sizeof p[0]), o, EXL3I_RR_N_OPD,
        "logit_rerank against an exl3i head read in exl3i_dense's stored form: R candidate ids a row (GLOBAL, vocab_off "
        "this rank's first row; outside its rows a pad) scored as exl3i_dense's decode form scores that row (x x suh "
        "rotated to f16, the candidate's 128-row output-Hadamard block decoded and dotted, its element of the output "
        "rotation x 1/sqrt(128) x svh; R16 rows by the record's dtype), re-sorted descending, ties to the lower id. "
        "include/exl3i_rerank.h." };
    return &s;
}

/* The stored row of local vocabulary row v in a head of n rows (R16 order or natural). */
#if defined(__HIPCC__)
__host__ __device__
#endif
static inline int64_t exl3i_rerank_row(int64_t v, int64_t n, int r16) {
    const int64_t e8 = n / 8;
    return r16 ? (v % e8) / 16 * 128 + v / e8 * 16 + v % 16 : v;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* EXL3I_RERANK_H */
