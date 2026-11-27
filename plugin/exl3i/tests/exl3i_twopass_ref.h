/* exl3i_twopass_ref.h -- the C reference of exl3i's two-pass prefill planes (exl3i_a8.h): plain C99, slow and readable,
 * every byte a device pass writes defined here from the oracle (common/exl3i_decode.h). Used by the host test
 * (tests/exl3i_twopass_host_test.c) and the device test (tests/test_twopass.hip), which compares the device's scratch
 * with these byte for byte. Build with -ffp-contract=off. */
#ifndef EXL3I_TWOPASS_REF_H
#define EXL3I_TWOPASS_REF_H

#include <stdint.h>
#include "exl3i_decode.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The int8 grid's step of the A8 plane: WS = 4 K_INV (mad's grid). */
float exl3i_ref_a8_ws(int cb);

/* W_hat's int8 indices of rows [c0, c0 + nc) (a chunk) of a mad weight in the i8a8 fragment plane:
 * [nc/16][K/64][1024] bytes, exl3i_i8plane_off. Built tile by tile from exl3i_decode_tile (ring order, then
 * exl3_pos_to_tile), each value's index recovered from its fp16 value (exl3i_a8_wq). 0, or -1 (arguments). */
int exl3i_ref_i8plane(const Exl3iView* v, int64_t c0, int64_t nc, int8_t* out);

/* T2's f16 plane of rows [c0, c0 + nc): halfs [K/16][nc/16][32 lanes][8], exl3i_f16plane_off; any codebook. */
int exl3i_ref_f16plane(const Exl3iView* v, int64_t c0, int64_t nc, uint16_t* out);

/* One 128-feature block of one row rotated as the device's rotate128_math (exl3r4d_common.hip.h) computes it, bit for
 * bit: v = f16(x x suh) (x f16 or bf16), the natural-order transform as single-rounded f32 adds in its stage order
 * (feature bits 0, 1 in a lane; 2..5 the DPP stages; 6 the lane ^ 16 stage), then f16(v x f32(1 / sqrt(128))) with
 * the product rounded once (gfx1201's compiled form is v_fma_mixlo_f16, even under -ffp-contract=off). */
void exl3i_ref_rotate128(const uint16_t* x, int x_bf16, const uint16_t* suh, uint16_t* out);

/* The A8 activation planes of rows [0, M) (exl3i_a8.h actq): aq int8 [mats][M][K], sa f32 [mats][M][K/128]; x
 * [M][x_pitch] (16-bit), suh / suh1 f16 [K] (suh1 used when mats = 2), group inputs a scale. */
void exl3i_ref_actq(const uint16_t* x, int64_t x_pitch, int x_bf16, const uint16_t* suh, const uint16_t* suh1,
                    int mats, int64_t M, int64_t K, int64_t group, int8_t* aq, float* sa);

/* The A8 GEMM in double over the quantised operands (what the device's arithmetic approximates; no rounding but the
 * output's): y[m][n] = svh[n] x 1/sqrt(128) x sum_n' H[n mod 128][n' mod 128] x WS x sum_kb sa[m][kb]
 * x sum_{k in kb} aq[m][k] W8[n'][k], n' over n's 128-row block, the activation plane of n's half (mats = 2). W8 the
 * whole weight's int8 plane (exl3i_ref_i8plane at c0 = 0, nc = N). out double [M][N]. */
void exl3i_ref_a8_gemm(const int8_t* aq, const float* sa, const int8_t* w8, const uint16_t* svh, int mats, int64_t M,
                       int64_t N, int64_t K, double* out);
/* the same for codebook cb's plane (WS replaced by exl3i_ref_a8_ws(cb)) */
void exl3i_ref_a8_gemm_cb(const int8_t* aq, const float* sa, const int8_t* w8, const uint16_t* svh, int mats, int64_t M,
                          int64_t N, int64_t K, int cb, double* out);

/* The same GEMM as the device computes it, operation for operation, so its output is the device's byte for byte: the
 * exact int8 sums of each 128-K block (WMMA, int32) read as f32 12582912 + sum, folded with one f32 FMA a block by the
 * row's scale, the bias 12582912 x the summed scales taken out every 8 blocks and at the end (the fold,
 * exl3i_8a8.hip.h), the output Hadamard in the epilogue's order (bits 0-2 butterflies, bit 3, bit 4, then bits 5-6
 * as one signed sum over the four weight waves in wave order), x C = f32(WS x f32(1/sqrt(128))), x svh, then y_dt
 * 0 bf16 (RNE), 1 f32, 2 f16 (RNE) into y [M][y_pitch]. */
void exl3i_ref_a8_gemm_dev(const int8_t* aq, const float* sa, const int8_t* w8, const uint16_t* svh, int mats, int64_t M,
                           int64_t N, int64_t K, int y_dt, void* y, int64_t y_pitch);
/* the same for codebook cb's plane: C = f32(exl3i_ref_a8_ws(cb) x f32(1/sqrt(128))) (exl3i_a8_gemm's) */
void exl3i_ref_a8_gemm_dev_cb(const int8_t* aq, const float* sa, const int8_t* w8, const uint16_t* svh, int mats,
                              int64_t M, int64_t N, int64_t K, int cb, int y_dt, void* y, int64_t y_pitch);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* EXL3I_TWOPASS_REF_H */
