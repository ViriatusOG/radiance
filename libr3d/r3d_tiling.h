/* r3d_tiling.h -- the LDS swizzle shared by libr3d's tiled rows.
 *
 * A tiled row stages an [rows x pitch] f16 tile in LDS and reads it back in pairs for dot2_f16x2.
 * The store and the load must use the same swizzle, so both are defined here:
 *
 *   srow(row, c8, pitch)      element offset of the c8-th run of eight elements of `row`
 *   srow_pair(row, kc, pitch) element offset of the pair (kc, kc+1) of `row`, kc even
 *
 *  - Granularity is eight elements (16 bytes at f16) because staging writes one uint4 per step; a
 *    finer swizzle would split the store and make it unaligned.
 *  - The XOR with (row & 7) rotates the runs of eight consecutive rows by one slot each. A wavefront
 *    reads 32 rows at the same kc, and the rotation spreads those reads over 32 distinct 16-byte
 *    slots instead of the four a plain row-major tile uses, which removes the bank conflict.
 *  - The mask (pitch / 8 - 1) keeps the rotated index inside the row. pitch is in elements and
 *    pitch/8 must be a power of two (32 or 128 for every tile in the library). Without the mask the
 *    index reads a neighbouring row, a silent, data-dependent error.
 *  - srow_pair finds the run the pair lives in (kc >> 3) and adds the element inside it (kc & 7).
 *
 * With dot2_f16x2 accumulating into f32, a row using these functions matches ref_gemm_nt_q at a
 * bf16 output to rel_l2 1.7e-03 (bounded by the store's rounding); a wrong swizzle gives rel_l2 3e+01.
 *
 * Include at file scope in a .hip before the kernels. The definitions are in an unnamed namespace,
 * so each translation unit gets its own copy.
 */
#ifndef R3D_TILING_H
#define R3D_TILING_H

namespace {

/* The element offset of run `c8` of `row`. pitch is a row's element count, a power of two. */
__device__ __forceinline__ int srow(int row, int c8, int pitch)
{
    return row * pitch + (((c8 ^ (row & 7)) & (pitch / 8 - 1)) << 3);
}

/* The element offset of the f16 pair (kc, kc+1), kc even: the run the pair lives in, plus the
 * element's place inside that run. */
__device__ __forceinline__ int srow_pair(int row, int kc, int pitch)
{
    return srow(row, kc >> 3, pitch) + (kc & 7);
}

}  // namespace

#endif /* R3D_TILING_H */
