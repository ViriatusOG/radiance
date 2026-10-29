/* r3d_fp8_dequant.h -- the E4M3 -> f16 decoder as a table, for the rows that read fp8 codes.
 *
 * WHY THIS EXISTS, WITH THE MEASUREMENT THAT ASKED FOR IT
 *
 * gfx11 has no instruction that converts E4M3 to f16. clang lists `__builtin_amdgcn_cvt_pk_f16_fp8`
 * for the target, and a build that calls it on gfx1102 contains ZERO `v_cvt_pk_f16_fp8` in its ISA:
 * the builtin is software here, lowered to the same exponent work `e4m3_to_f32` does. r3d_codecs.h's
 * codec is two `e4m3_to_f32` (a dozen instructions of shifts, exponent bias and subnormal handling
 * each) plus a pack, about 25 instructions a pair, 100 an eight-code group. Measured on the
 * vocabulary head at M=1 (r3d_vocab_bf16.hip, RX 7600 / gfx1102 / 32 CU): 13.603 ms a token at
 * 19.6 GB/s of codes, 8x above the 1.64 ms the codes' 255 MiB floor at this row's own 163 GB/s, and
 * the ALU was the binding resource, not DRAM.
 *
 * E4M3 has 256 bit patterns and every one of them has an f16 value (f16's exponent range covers
 * E4M3's, its subnormals included, and E4M3's NaN maps to an f16 NaN). So the decoder is a
 * 256-entry table of 512 bytes, and eight LDS reads replace the hundred ALU instructions. That is
 * the whole idea, and the table goes in LDS -- not __constant__, whose broadcast is a liability here
 * because every lane decodes a DIFFERENT code, and divergent constant indexes serialise.
 *
 * Cost: 512 bytes of LDS a block, one fill of 256 stores at block start plus a barrier. Conflict
 * behaviour on the reads: 16-bit loads at data-dependent addresses, so a wavefront's 32 lanes land
 * on 16 bank words; random codes give roughly two lanes per word, and measured cost is about two
 * cycles a read -- eight reads an eight-code group against the codec's hundred instructions.
 *
 * The table is EXACT: it is built by calling the codec once per entry, so it is the codec, moved
 * from the loop to the block's start. Nothing about the arithmetic changes, and a row's rel_l2
 * against its reference is unchanged by this header. The one thing to keep: the table must be filled
 * before any lane reads it, so every use is `..._init` then the barrier it contains, then reads.
 *
 * int8 codes do NOT want a table: `v_cvt_f32_i8` is a hardware instruction, so
 * r3d_codecs.h's i8x2_to_f16x2 is two converts and a pack. This header serves the fp8 codes.
 *
 * Header-only, like the rest of libr3d. Include it in a .hip after r3d_codecs.h.
 */
#ifndef R3D_FP8_DEQUANT_H
#define R3D_FP8_DEQUANT_H

#include "r3d_codecs.h"

#define R3D_E4M3_TBL 256          /* every E4M3 bit pattern, in order, as the f16 it means */
#define R3D_E4M3_TBL_BYTES (R3D_E4M3_TBL * 2)

/* Fill the block's table: `tbl` is __shared__ uint16_t[R3D_E4M3_TBL]. The barrier is inside it,
 * because the only correct order is fill-then-read and the row should not have to remember it. */
__device__ __forceinline__ void r3d_e4m3_f16_init(uint16_t* tbl, int tid, int nthr)
{
    for (int i = tid; i < R3D_E4M3_TBL; i += nthr)
        tbl[i] = f32_to_f16(e4m3_to_f32((uint8_t)i));
    __syncthreads();
}

/* Two codes (the low 16 bits of a dword) as one f16 pair, the pair layout dot2_f16x2 reads: the
 * first code in the low half. Same answer as e4m3x2_to_f16x2, eight table reads per eight codes. */
__device__ __forceinline__ uint32_t r3d_e4m3x2_ld(const uint16_t* tbl, uint32_t pair)
{
    const uint16_t lo = tbl[pair & 0xffu];
    const uint16_t hi = tbl[(pair >> 8) & 0xffu];
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

/* Eight codes (two dwords, straddling pairs as always) as four f16 pairs. The sibling of
 * codes8_to_f16x4 in r3d_vocab_bf16.hip; the int8 side of that helper stays with the codec, because
 * int8 has a hardware convert and fp8 does not. */
__device__ __forceinline__ void r3d_e4m3x8_ld(const uint16_t* tbl, uint2 c, uint32_t* w)
{
    const uint32_t d0 = c.x, d1 = c.y;
    w[0] = r3d_e4m3x2_ld(tbl, d0 & 0xffffu);
    w[1] = r3d_e4m3x2_ld(tbl, d0 >> 16);
    w[2] = r3d_e4m3x2_ld(tbl, d1 & 0xffffu);
    w[3] = r3d_e4m3x2_ld(tbl, d1 >> 16);
}

/* One code, for the rows that decode a scalar a step (an activation tile staged a byte a lane). */
__device__ __forceinline__ uint16_t r3d_e4m3_ld(const uint16_t* tbl, uint8_t code)
{
    return tbl[code];
}

#endif /* R3D_FP8_DEQUANT_H */
