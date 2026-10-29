/* r3d_tiling.h -- the tiled rows' LDS row law, in one place.
 *
 * Every tiled row in libr3d stages an [rows x pitch] f16 tile in LDS and then reads it back two
 * elements at a time for dot2_f16x2. Two functions spell that out and they must agree exactly, so
 * they live here and the rows include this file.
 *
 *   srow(row, c8, pitch)      the element offset of the c8-th run of EIGHT elements of `row`
 *   srow_pair(row, kc, pitch) the element offset of the pair (kc, kc+1) of `row`, kc even
 *
 * The law, and why it is shaped like this:
 *
 *  - Granularity is EIGHT elements (16 bytes at f16): staging writes a uint4 a step, so the XOR has
 *    to move whole uint4s and never split one. A swizzle at pair granularity would make the store
 *    unaligned and the compiler would say so in bytes of spill.
 *  - The XOR is with (row & 7), so eight consecutive rows rotate the runs by one slot each. That is
 *    what kills the dot loop's bank conflict: a wavefront reads 32 rows at the same kc, and with the
 *    rotation the 32 reads land on 32 distinct 16-byte slots instead of the four a plain row-major
 *    tile puts them on.
 *  - The mask (pitch / 8 - 1) keeps the rotated index INSIDE the row. pitch is in elements, so a
 *    row has pitch/8 runs, and pitch/8 is a power of two for every tile in the library (32, 128).
 *    Without the mask a rotated index leaves the row and reads a neighbour's, which is the bug this
 *    header exists to prevent: it is silent, it is data-dependent, and it measures 30x off.
 *  - srow_pair takes the pair index kc>>3 -- the run the pair lives in -- and adds (kc & 7), the
 *    element inside the run. Reading pairs through srow is what keeps the store, the load and the
 *    swizzle one statement each.
 *
 * Measured law, in the family's numbers: a row that stages through srow and reads through
 * srow_pair, with dot2_f16x2 accumulating into f32, lands at rel_l2 1.7e-03 against
 * ref_gemm_nt_q's body at a bf16 output (the store's ulp binding, not the fold's) with 83% of
 * outputs bit-exact. A row that gets the swizzle wrong, in either direction, lands at rel_l2 3e+01.
 *
 * Header-only, like the rest of libr3d. Include it in a .hip at file scope, before the kernels; the
 * definitions are in an unnamed namespace, so each translation unit gets its own copy -- which is
 * what the rows had before this header existed, spelled the same way, so the host pass does not see
 * an unused static and the inliner sees the same function it always did.
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
