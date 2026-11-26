/* exl3_decode.h -- the shared EXL3 numerics the exl3i decoders are built on: the fp16 conversions, the 128-point
 * Sylvester Hadamard, and the trellis tile <-> ring-position permutation. Plain C99, no dependencies.
 *
 * The oracle for what an exl3i weight's numbers are lives in exl3i_decode.h; everything here is the arithmetic
 * its decode and the device kernels (embed_exl3i.hip, exl3i_rerank.hip) transcribe, line by line.
 *
 * Thread safety: every function is reentrant (no static state). Build with -ffp-contract=off.
 */
#ifndef EXL3_DECODE_H
#define EXL3_DECODE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    EXL3_HAD = 128,   /* the rotation block; both feature counts are multiples of it */
};

/* Return codes. Negative is failure. */
enum { EXL3_OK = 0, EXL3_E_ARG = -1, EXL3_E_SHAPE = -2, EXL3_E_NOMEM = -3 };

/* ------------------------------------------------------------------ fp16 */
float    exl3_half_to_float(uint16_t h);
/* Round to nearest even, ONE rounding of the exact double; overflow to inf, nan to 0x7E00. */
uint16_t exl3_half_from_double(double x);

/* ------------------------------------------------------------------ the tile */
/* Position p (0..255) -> tile row r (k axis, 0..15) and tile column c (n axis, 0..15). */
void exl3_pos_to_tile(int p, int* r, int* c);
/* Tile (r, c) -> position. */
int  exl3_tile_to_pos(int r, int c);

/* ------------------------------------------------------------------ the rotation */
/* The normalised natural-order Sylvester Hadamard over each block of 128 of v[0..n), in place:
 * butterfly (a+b low, a-b high) in f32, then x (float)(1/sqrt(128)). EXL3_E_SHAPE unless n % 128 == 0. */
int exl3_had128(float* v, int64_t n);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* EXL3_DECODE_H */
