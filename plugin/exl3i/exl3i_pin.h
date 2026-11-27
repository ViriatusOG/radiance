/* exl3i_pin.h -- test pins: the forms a served engine does not select by itself at a given shape, reachable from the
 * device tests and benches, which compare each fast form with the form it replaces on the same weight.
 *
 * A served engine never calls exl3i_test_pin, so every pin reads 0 and every launch follows the dispatch rules of nt.c,
 * exl3i_nt.hip.h and exl3i_a8_prep.hip. The pins are process-wide and not synchronised: a test sets them between
 * launches, from one thread. A test that loads exl3i.so reaches the setter by dlsym(h, "exl3i_test_pin"); a test that
 * compiles the sources links pin.c. */
#ifndef EXL3I_PIN_H
#define EXL3I_PIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    EXL3I_PIN_A8_MIN_ROWS = 0,  /* n > 0: the A8 path from n real rows instead of EXL3I_A8_MIN_ROWS */
    EXL3I_PIN_A8_CHUNK_MB,      /* n > 0: the A8 int8 plane in column chunks of at most n MiB instead of EXL3I_A8_CHUNK_BYTES */
    EXL3I_PIN_KSPLIT,           /* n > 0: the decode form's K split into n slices (1: unsplit) instead of ks_slices' rule */
    EXL3I_PIN_MAD_GEMV_OFF,     /* 1: no byte-sum GEMV (nt_pass's DOT / WMMA forms on the same weight) */
    EXL3I_PIN_MAD_GEMV_T,       /* n > 0: the byte-sum GEMV up to n rows (at most 8) at every rate and shape */
    EXL3I_PIN_DEC_DOT_OFF,      /* 1: the one-row decode on the WMMA form instead of the v_dot2 form */
    EXL3I_PIN_T2_OFF,           /* 1: nt_pass at every row count (no two-pass prefill form) */
    EXL3I_PIN_AQ_RES_SPLIT_OFF, /* 1: exl3i_actq's residual rmsnorm one workgroup a row (no arrival counters) */
    EXL3I_PIN_A8_FUSE_OFF,      /* 1: the A8 path's two passes (wdec8, then the GEMM) instead of the fused GEMM */
    EXL3I_PIN_N
};

extern int64_t exl3i_pins[EXL3I_PIN_N];

/* Sets one pin (0 restores the served rule). An unknown pin is ignored. */
void exl3i_test_pin(int pin, int64_t value);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* EXL3I_PIN_H */
