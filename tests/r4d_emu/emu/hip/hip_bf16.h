/* tests/r4d_emu (and halo_emu): HIP's __hip_bfloat16, round-to-nearest-even from float (NaN kept quiet), as ROCm's. */
#pragma once
#include <cstdint>
#include <cstring>
struct __hip_bfloat16 {
    uint16_t data;
    __hip_bfloat16() = default;
    __hip_bfloat16(float f) {
        uint32_t u; std::memcpy(&u, &f, 4);
        if ((u & 0x7fffffffu) > 0x7f800000u) data = (uint16_t)((u >> 16) | 0x40u);
        else data = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    operator float() const { uint32_t u = (uint32_t)data << 16; float f; std::memcpy(&f, &u, 4); return f; }
};
