/* r3d_fp8_dequant.h -- E4M3 -> f16 decode through a 256-entry LDS table.
 *
 * gfx11 has no E4M3-to-f16 instruction: `__builtin_amdgcn_cvt_pk_f16_fp8` compiles on gfx1102 but
 * emits no `v_cvt_pk_f16_fp8`, only the same software exponent work as `e4m3_to_f32`. The codec in
 * r3d_codecs.h costs about 25 instructions a pair, 100 an eight-code group. On the M=1 vocabulary
 * head (r3d_vocab_bf16.hip, gfx1102, 32 CU) that makes the row ALU-bound at about 8x its DRAM floor.
 *
 * Every one of E4M3's 256 bit patterns has an exact f16 value (f16's exponent range covers E4M3's,
 * subnormals included, and NaN maps to NaN), so the decoder is a 512-byte table and eight LDS reads
 * replace about a hundred ALU instructions. The table lives in LDS, not __constant__: each lane
 * decodes a different code, and divergent constant indexes serialise.
 *
 * Cost: 512 bytes of LDS a block and a 256-store fill plus a barrier at block start. Reads are
 * 16-bit loads at data-dependent addresses; random codes put about two lanes on each bank word, for
 * about two cycles a read.
 *
 * The table is filled by calling the codec once per entry, so results are bit-identical to the
 * codec. Invariant: the table must be filled before any lane reads it; r3d_e4m3_f16_init contains
 * the barrier.
 *
 * int8 codes need no table: `v_cvt_f32_i8` is a hardware instruction (r3d_codecs.h: i8x2_to_f16x2).
 *
 * Include in a .hip after r3d_codecs.h.
 */
#ifndef R3D_FP8_DEQUANT_H
#define R3D_FP8_DEQUANT_H

#include "r3d_codecs.h"

#define R3D_E4M3_TBL 256          /* every E4M3 bit pattern, in order, as the f16 it means */
#define R3D_E4M3_TBL_BYTES (R3D_E4M3_TBL * 2)

/* Fill the block's table: `tbl` is __shared__ uint16_t[R3D_E4M3_TBL]. Ends with the barrier that
 * orders the fill before any read. */
__device__ __forceinline__ void r3d_e4m3_f16_init(uint16_t* tbl, int tid, int nthr)
{
    for (int i = tid; i < R3D_E4M3_TBL; i += nthr)
        tbl[i] = f32_to_f16(e4m3_to_f32((uint8_t)i));
    __syncthreads();
}

/* Two codes (the low 16 bits of a dword) as one f16 pair in dot2_f16x2 layout, the first code in
 * the low half. Same result as e4m3x2_to_f16x2. */
__device__ __forceinline__ uint32_t r3d_e4m3x2_ld(const uint16_t* tbl, uint32_t pair)
{
    const uint16_t lo = tbl[pair & 0xffu];
    const uint16_t hi = tbl[(pair >> 8) & 0xffu];
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

/* Eight codes (two dwords) as four f16 pairs; the fp8 counterpart of codes8_to_f16x4 in
 * r3d_vocab_bf16.hip. */
__device__ __forceinline__ void r3d_e4m3x8_ld(const uint16_t* tbl, uint2 c, uint32_t* w)
{
    const uint32_t d0 = c.x, d1 = c.y;
    w[0] = r3d_e4m3x2_ld(tbl, d0 & 0xffffu);
    w[1] = r3d_e4m3x2_ld(tbl, d0 >> 16);
    w[2] = r3d_e4m3x2_ld(tbl, d1 & 0xffffu);
    w[3] = r3d_e4m3x2_ld(tbl, d1 >> 16);
}

/* One code, for rows that decode a scalar per step (an activation tile staged a byte a lane). */
__device__ __forceinline__ uint16_t r3d_e4m3_ld(const uint16_t* tbl, uint8_t code)
{
    return tbl[code];
}

#endif /* R3D_FP8_DEQUANT_H */
