/* exl3i_rerank_dev.h -- what exl3i_rerank's host side (rerank.c) hands its device side (exl3i_rerank.hip), and the host
 * reference (rerank_ref.c) the tests hold the device row to. The op: include/exl3i_rerank.h. */
#ifndef EXL3I_RERANK_DEV_H
#define EXL3I_RERANK_DEV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One launch; passed by value (zeroed first: a recorded launch replays its argument bytes). */
typedef struct Exl3iRerankCall {
    const void*     x;         /* bf16 | f16 [M][x_pitch] */
    const uint32_t* tiles;     /* the record in exl3i_dense's stored form: tile (tk, tn) at (tk N/16 + tn) m/2 words */
    const uint16_t* suh;       /* f16 [K] */
    const uint16_t* svh;       /* f16 [N] */
    const int32_t*  idx_in;    /* [M][in_pitch] */
    int32_t*        idx;       /* [M][idx_pitch] */
    float*          val;       /* [M][val_pitch] */
    float*          pairs;     /* [M][R][2] or null */
    int64_t         x_pitch, in_pitch, idx_pitch, val_pitch;
    int64_t         M, R, N, K, voff;
    int32_t         m, cb, r16, x_f16;
} Exl3iRerankCall;

/* The two launches (scores a workgroup a (row, candidate), then the per-row sort). RAD_OK or a negative RAD_E_*. */
int exl3i_launch_rerank(const Exl3iRerankCall* c, void* stream);
/* The device row's first stage alone (tests): the M rows' rotated activations, f16 bits [M][K], into out (device). */
int exl3i_rerank_rotate_rows(const Exl3iRerankCall* c, uint16_t* out, void* stream);

/* The host reference: the same values bit for bit, from the container's record (row-major: table row t = weight rows
 * [16 t, 16 t + 16), its tiles in k order). x is bf16 / f16 bits; idx may alias idx_in. 0, or -1 (bad arguments). */
int exl3i_rerank_ref(const uint16_t* x, int64_t x_pitch, int x_f16, const uint32_t* record, const uint16_t* suh,
                     const uint16_t* svh, const int32_t* idx_in, int32_t* idx, float* val, float* pairs, int64_t M,
                     int64_t R, int64_t N, int64_t K, int64_t voff, int m, int cb, int r16);
/* The ref's first stage alone: row x's rotated f16 activation, xr[K] (fp16 bits). */
void exl3i_rerank_ref_rotate(const uint16_t* x, int x_f16, const uint16_t* suh, int64_t K, uint16_t* xr);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* EXL3I_RERANK_DEV_H */
