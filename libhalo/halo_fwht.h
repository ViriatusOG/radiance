/* Derived from libr4d/r4d_fwht.h by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. Op schemas, row names and stored weight layouts are
 * shared with libr4d by design. Everything else in this file is libhalo's own
 * and diverges as it is measured on gfx1151. */
// halo_fwht.h - a wave-local fast Walsh-Hadamard transform in registers, with no LDS and no barrier.
//
// A butterfly staged in LDS costs seven stages, each a read-modify-write of every element plus a
// __syncthreads() (about 28 LDS operations and 7 barriers per element), several times the rest of
// a rotated quantiser; the transform's arithmetic is a fraction of a percent.
//
// Mapping: one wave holds a `HAD`-wide block, VPL = HAD/32 contiguous values per lane, so lane l
// holds indices l*VPL .. l*VPL+VPL-1. A butterfly at stride h pairs index i with i+h:
//
//   h < VPL    both indices in the same lane     -- plain register arithmetic
//   h >= VPL   the partner is lane l + h/VPL     -- one lane exchange with mask h/VPL
//
// HAD/VPL is always 32, so the cross-lane masks are 1, 2, 4, 8, 16 at any HAD: five exchanges per
// register (halo_xor_lane, DPP) and no barrier. The lane holding the low index keeps (v + partner),
// the high one (partner - v). The caller still needs one barrier before a whole-row reduction.
#pragma once
#include <hip/hip_runtime.h>
#include "halo_common.h"

// `hi ? (o - v) : (v + o)` written as `o + (hi ? -v : v)`, with the sign flip one XOR against a
// stage-uniform mask, so there is no per-element select. Bit-identical: negation is exact and
// addition commutes.
template <int VPL, int MASK>
__device__ __forceinline__ void halo_fwht_xstage(float* v, int lane) {
    const unsigned int sm = (lane & MASK) ? 0x80000000u : 0u;
#pragma unroll
    for (int j = 0; j < VPL; j++) {
        const float o = halo_xor_lane<MASK>(v[j]);
        v[j] = __int_as_float(__float_as_int(v[j]) ^ sm) + o;
    }
}

// In-register FWHT of a HAD-wide block held VPL = HAD/32 values per lane, lane-major.
// Unnormalised: the caller folds 1/sqrt(HAD) into whatever scale it already applies.
template <int HAD>
__device__ __forceinline__ void halo_fwht_wave(float* v, int lane) {
    constexpr int VPL = HAD / 32;
    // in-lane stages: stride 1 .. VPL/2, entirely within this lane's registers
#pragma unroll
    for (int h = 1; h < VPL; h <<= 1) {
#pragma unroll
        for (int j = 0; j < VPL; j++) {
            if ((j & h) == 0) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
    // cross-lane stages: stride VPL*mask for mask = 1, 2, 4, 8, 16
    halo_fwht_xstage<VPL, 1>(v, lane);
    halo_fwht_xstage<VPL, 2>(v, lane);
    halo_fwht_xstage<VPL, 4>(v, lane);
    halo_fwht_xstage<VPL, 8>(v, lane);
    halo_fwht_xstage<VPL, 16>(v, lane);
}
