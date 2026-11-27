/* exl3dense_device.hip.h -- the device helpers the token-embedding gathers (embed_exl3i.hip) share: the codebook's fp16
 * constant K_INV (mad's grid step is 4 K_INV), the fp16 and bf16 roundings, and the fp16 conversion. The gather's own decode and rotation are in
 * embed_exl3i.hip; the rounding functions here are the host's (rad_f32_to_bf16, rad_f32_to_f16), written out.
 */
#ifndef EXL3DENSE_DEVICE_HIP_H
#define EXL3DENSE_DEVICE_HIP_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "exl3dense.h"

namespace exl3dense_kernel {

/* K_INV, by its bits: the int8 grid's step is 4 K_INV (mad's value, exl3i_decode.h). */
constexpr unsigned short kKInv = 0x1EEEu;   /* ~1/147.7 */

__device__ __forceinline__ float half_to_float(unsigned short h) {
    return __half2float(__ushort_as_half(h));
}

/* rad_f32_to_bf16's IEEE RNE rule, bit for bit. */
__device__ __forceinline__ unsigned short f32_to_bf16(float f) {
    const unsigned u = __float_as_uint(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (unsigned short)((u >> 16) | 0x0040u);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

/* rad_f32_to_f16: IEEE RNE, subnormals kept, overflow to infinity -- the host's algorithm, written
 * out rather than left to __float2half_rn. The hardware conversion is the same rounding, but the
 * compiler is free to fuse the multiply that feeds it into one mixed-precision FMA
 * (`v_fma_mixlo_f16`), which rounds the exact product once where the contract rounds the f32
 * product and then the f16 value. That double rounding is visible: one element in 8448 differed by
 * 1 ulp against the host reference before this became integer arithmetic. */
__device__ __forceinline__ unsigned short f32_to_f16(float f) {
    const unsigned u = __float_as_uint(f);
    const unsigned s = (u >> 16) & 0x8000u;
    const unsigned a = u & 0x7fffffffu;
    if (a >= 0x7f800000u)
        return (unsigned short)(s | 0x7c00u | (a > 0x7f800000u ? ((a >> 13) & 0x3ffu) | 0x200u : 0u));
    if (a >= 0x47800000u) return (unsigned short)(s | 0x7c00u);
    if (a < 0x33000000u) return (unsigned short)s;
    {
        const int e = (int)(a >> 23) - 127;
        unsigned m = a & 0x7fffffu;
        if (e < -14) {
            const int sh = -14 - e + 13;
            unsigned lo, q, half;
            m |= 0x800000u;
            lo = m & ((1u << sh) - 1u);
            q = m >> sh;
            half = 1u << (sh - 1);
            if (lo > half || (lo == half && (q & 1u))) ++q;
            return (unsigned short)(s | q);
        }
        {
            unsigned q = m >> 13;
            const unsigned lo = m & 0x1fffu;
            unsigned he;
            if (lo > 0x1000u || (lo == 0x1000u && (q & 1u))) ++q;
            he = (unsigned)(e + 15);
            if (q == 0x400u) { q = 0; ++he; }
            if (he >= 0x1fu) return (unsigned short)(s | 0x7c00u);
            return (unsigned short)(s | (he << 10) | q);
        }
    }
}

}  /* namespace exl3dense_kernel */

#endif  /* EXL3DENSE_DEVICE_HIP_H */
