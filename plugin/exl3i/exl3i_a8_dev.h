/* exl3i_a8_dev.h -- what exl3i's host side (nt.c) hands the two-pass paths' device side: the A8 path (exl3i_a8_prep.hip,
 * exl3i_a8_gemm.hip; exl3i_a8.h): one plain argument block, passed by value (< 768 B), and their launches. No radiance
 * types cross this line. */
#ifndef EXL3I_A8_DEV_H
#define EXL3I_A8_DEV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Exl3iA8Args {
    const void*  x;         /* a [M][x_pitch] f16 | bf16 */
    const void*  codes;     /* the record in tile order (k-major, nt.c's relayout) */
    const void*  suh;       /* f16 [K] */
    const void*  suh1;      /* f16 [K], the up half's (stacked gate_up), or null */
    const void*  svh;       /* f16 [N] */
    void*        y;         /* [M][y_pitch] of y_dt */
    int8_t*      aq;        /* the arena (exl3i_a8.h Exl3iA8Plan at M): int8 [mats][M][K] */
    float*       sa;        /* f32 [mats][M][K / 128] */
    int8_t*      wq;        /* the chunk's int8 plane [nc / 16][K / 64][1024] */
    int64_t      x_pitch, y_pitch;   /* elements */
    int32_t      M, N, K;   /* real rows; the weight [N, K] */
    int32_t      m;         /* rate m / 16 */
    int32_t      group;     /* inputs an activation scale (128 .. K, dividing K) */
    int32_t      mats;      /* 2 with suh1, else 1 */
    int32_t      x_bf16;    /* a is bf16 (else f16) */
    int32_t      y_dt;      /* 0 bf16, 1 f32, 2 f16 */
    int32_t      c0, nc;    /* the current column chunk [c0, c0 + nc), both multiples of 128 */
    int32_t      cb;        /* EXL3I_CB_*: mad (the int8 plane's index and step) */
    /* A8: where the GEMM reads the activation planes -- row pitches (int8 elements of aq, floats of sa) and the second
     * plane's offset (a stacked gate_up's suh.1). 0: the arena's own [mats][M][K] / [mats][M][K / 128] (actq's). AQ
     * (exl3i_abi.h version 3): the producer's packed rows, aq / sa at this weight's first plane. */
    int64_t      a_ld, a_ms, sa_ld, sa_ms;
} Exl3iA8Args;

/* exl3i_actq (AQ): x [M, K] for up to four consumers' input scales s[i] -> out, packed per row as [planes][K] int8 then
 * [planes][K / 128] f32 (out_ld bytes a row) -- each plane the A8 path's own a8g128 quantisation (exl3i_a8_actq) of x,
 * or of the producer stage `pre` (EXL3I_AQ_PRE_*, exl3i_abi.h) applied to x: rmsnorm (gain g, eps, wadd), silu_mul
 * (x [M, 2K], gate first), mul (x x x2); its bf16 value also to h when h is set. */
typedef struct Exl3iAqArgs {
    const void*  x;          /* [M][x_pitch] f16 | bf16 (bf16 for every pre but none) */
    const void*  s[4];       /* f16 [K] each, planes of them */
    uint8_t*     out;
    const void*  x2;         /* pre mul: bf16 [M][x2_pitch] */
    const void*  g;          /* pre rmsnorm: the gain, f32 (g_f32) | bf16 [K] */
    void*        h;          /* optional: bf16 [M][h_pitch], the pre stage's value */
    int64_t      x_pitch, out_ld, x2_pitch, h_pitch;
    float        eps, wadd;
    int32_t      M, K, planes, x_bf16, pre, g_f32;
    int32_t      form;       /* EXL3I_AQ_FORM_* (exl3i_abi.h): int8 planes, or xb (the mad GEMV's rows; version 4) */
    /* version 5, pre rmsnorm: x + res (bf16 [M][res_pitch]) is written to xo ([M][xo_pitch]) and normalised; null: none.
     * rctr: the decode band's arrival counters (a row over several workgroups), set by exl3i_aq_actq */
    const void*  res;
    void*        xo;
    int64_t      res_pitch, xo_pitch;
    unsigned*    rctr;
} Exl3iAqArgs;
int exl3i_aq_actq(const Exl3iAqArgs* a, void* stream);

/* RAD_OK or a negative RAD_E_* (RAD_E_DEVICE a failed launch, RAD_E_SHAPE a geometry the kernels do not take); stream-
 * ordered, never allocate or synchronise. actq: the rows into aq / sa. wdec8: chunk [c0, c0 + nc) into wq (mad only:
 * the codebook the int8 plane exists for). gemm: the chunk's output columns. */
int exl3i_a8_actq(const Exl3iA8Args* a, void* stream);
int exl3i_a8_wdec8(const Exl3iA8Args* a, void* stream);
int exl3i_a8_gemm(const Exl3iA8Args* a, void* stream);
/* FUSE (mad): wdec8 + gemm of the chunk as one launch, the GEMM's blocks decoding their weight tiles first
 * (exl3i_a8_prep.hip).
 * RAD_E_UNSUPPORTED: not applicable (another codebook, see exl3i_a8_prep.hip) or pinned off by a test -- run the two
 * passes. */
int exl3i_a8_gemmd(const Exl3iA8Args* a, void* stream);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* EXL3I_A8_DEV_H */
