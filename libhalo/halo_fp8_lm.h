/* Derived from libr4d/r4d_fp8_lm.h by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. Op schemas, row names and stored weight layouts are
 * shared with libr4d by design. Everything else in this file is libhalo's own
 * and diverges as it is measured on gfx1151. */
// halo_fp8_lm.h -- stored row geometry of the fp8 lm_head. Pure C++ (no HIP, no halo.h) so the
// layout hook (halo_fp8_layout.cpp), the entry point (halo_gemm_fp8a16_nt_m1.hip) and the selftest
// derive the row width from one definition.
//
// `logits_gemm` carries one weight operand, so the block scales live in each row's tail (see
// halo_rad_layout_fp8_logits); a row-sharded lm_head stays a plain byte range under TP.
//
// Padding: both kernels that read this plane (the M <= 2 matvec and the WMMA tile) load the weight
// as uint4 at `row * w_ld + (a multiple of 16)`, so the row pitch must be a multiple of 16. The
// unpadded width is one only when
//
//     K + 2*(K/128) = 130m  (K = 128m)   is 0 mod 16   <=>   m is 0 mod 8   <=>   K is 0 mod 1024
//
// e.g. K = 5120 qualifies, K = 2560 does not. The pad costs under 16 bytes per row (0.3% at
// K = 2560) and is a no-op when K is a multiple of 1024, where the plane is byte-identical to the
// unpadded one.
//
// This is a 16-byte correctness floor, unrelated to the 128-byte row padding that
// halo_gemm_fp8a16_nt_m1.hip's header evaluates and rejects.
#pragma once
#include <cstdint>

namespace halo_fp8 {

// Bytes one stored lm_head row occupies: K E4M3 codes, then K/128 bf16 block scales, then the
// alignment tail. K must be a multiple of 128 (the scale block); the caller checks that.
inline int64_t lm_row_bytes(int64_t K) {
    const int64_t used = K + 2 * (K / 128);
    return (used + 15) & ~(int64_t)15;
}

}  // namespace halo_fp8
