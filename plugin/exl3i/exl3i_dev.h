/* exl3i_dev.h -- what exl3i's host side (nt.c) hands its device side (exl3i_nt_cb*.hip): the dense GEMM's argument
 * block (exl3r4d_dev.h: the dense op's fields) and the rate. Passed by value (< 768 B). */
#ifndef EXL3I_DEV_H
#define EXL3I_DEV_H

#include "exl3r4d_dev.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Exl3iGemm {
    Exl3r4dGemm g;   /* cls[0] the weight: codes = the record in tile order (k-major, nt.c's relayout) */
    int32_t     m;   /* rate m / 16 */
    int32_t     cb;  /* EXL3I_CB_* */
    /* KS (decode, T <= 16): the K split across workgroups. ksn slices (1: off), grid.z the slice; each workgroup writes
     * its raw f32 sums [ksn][T][N] to part (scratch past the GSCR region), the last of a column block's ksn to arrive
     * (ctr, the launch's stream's counter block, zero at entry and reset by that workgroup) sums them in slice order and
     * runs the epilogue. nt.c sets ksn and part (exl3i_ks_slices), the device side's launch kctr. */
    int32_t     ksn;
    float*      part;
    unsigned*   kctr;
    int32_t     r16;   /* the weight's rows in R16 order: output s is stored at row exl3i_r16_row(s, N) (nt.c, the layout dtype) */
    /* the mad GEMV's rows from the producer (AQ form xb, exl3i_abi.h version 4), or null (the GEMV quantises them):
     * row t at xq + t xq_ld bytes, this weight's first plane xq_plane (exl3i_aq_xb_plane_bytes(K) each) */
    const unsigned char* xq;
    int64_t     xq_ld;
    int32_t     xq_plane;
} Exl3iGemm;

/* The dense launch (mad, the one codebook; exl3i_nt_mad.hip). RAD_OK or a negative RAD_E_*; never allocates or
 * synchronises. npw: column tiles a wave (1, 2, 4). */
int exl3i_launch_nt_mad(const Exl3iGemm* a, int npw, void* stream);

/* The byte-sum GEMV (exl3i_mad.hip.h; mad, act a8g128, 1 .. 8 rows): the rows quantised into g.xr, per matrix (2 for a
 * stacked gate_up) and row exl3i_mad_row_bytes(K): xb int32 [K], sa f32 [K / 128], bias int32 [K / 128][2] (256-B
 * aligned rows). KS as the decode form's (ksn, part past the rows). npw: EXL3I_MAD_NPW of the build. */
#define EXL3I_MAD_ROW_BYTES(K) ((4 * (int64_t)(K) + 12 * ((int64_t)(K) / 128) + 255) / 256 * 256)   /* device side too */
static inline int64_t exl3i_mad_row_bytes(int64_t K) { return EXL3I_MAD_ROW_BYTES(K); }
int exl3i_launch_mad_gemv(const Exl3iGemm* a, int npw, void* stream);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* EXL3I_DEV_H */
