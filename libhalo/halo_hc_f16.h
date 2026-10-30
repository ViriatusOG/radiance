// halo_hc_f16.h -- E4M3 rows as f16 WMMA operands, for the gated residual's matrices (hc_read's
// mixing pair and mtp_enter's fc pair, stored as halo_hc_fp8.h describes).
//
// An E4M3 byte's magnitude moved to an f16's bits 14..7 (its sign to bit 15) reads as 2^-8 times
// the code's value, subnormals included (exponent bias 15 against 7), so sixteen codes become a
// B fragment in about two VALU a value. A row's group scale times 2^8 then multiplies the step's
// f32 product into the accumulator: a D column is one lane, so it is one scale a lane. The
// activations these matrices meet are bf16 by contract, which f16 holds exactly in the range
// they live in.
#pragma once
#include "halo_wmma.h"
#include <cstdint>

namespace halo_hcf {

/* Four E4M3 codes (one dword) to two f16x2 dwords, each value 2^-8 times the code's. */
__device__ __forceinline__ void e4m3x4(uint32_t w, uint32_t& lo, uint32_t& hi) {
    const uint32_t m = w & 0x7F7F7F7Fu, sg = w & 0x80808080u;
    lo = (__builtin_amdgcn_perm(0u, m, 0x0c010c00u) << 7) | __builtin_amdgcn_perm(0u, sg, 0x010c000cu);
    hi = (__builtin_amdgcn_perm(0u, m, 0x0c030c02u) << 7) | __builtin_amdgcn_perm(0u, sg, 0x030c020cu);
}
__device__ __forceinline__ void codes16(uint4 r, uint32_t* d) {
    e4m3x4(r.x, d[0], d[1]);
    e4m3x4(r.y, d[2], d[3]);
    e4m3x4(r.z, d[4], d[5]);
    e4m3x4(r.w, d[6], d[7]);
}
__device__ __forceinline__ v16hf frag(const uint32_t* d) {
    union { uint32_t w[8]; v16hf f; } u;
#pragma unroll
    for (int i = 0; i < 8; ++i) u.w[i] = d[i];
    return u.f;
}
/* Two floats that are exact bf16 values to an f16 pair: exact in f16's normal range. */
__device__ __forceinline__ uint32_t pk16(float a, float b) {
    return __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(a, b));
}

}  // namespace halo_hcf
