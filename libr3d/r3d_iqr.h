#pragma once

#include <cstdint>

#if defined(__HIPCC__) || defined(__HIP__)
#define R3D_IQR_HD __host__ __device__
#else
#define R3D_IQR_HD
#endif

namespace r3d_iqr {

constexpr uint32_t weight_dtype = 0x33000100u;
constexpr uint32_t scale_dtype = 0x33000200u;
constexpr uint32_t group_scale_dtype = 0x33000300u;
constexpr uint32_t weight_magic = 0x33525149u;

struct Header {
    uint32_t magic;
    uint32_t bits;
    uint32_t run_rows;
    uint32_t state_bits;
    uint32_t multiplier;
    uint32_t increment;
    uint32_t rows;
    uint32_t columns;
};
static_assert(sizeof(Header) == 32);

R3D_IQR_HD inline int8_t trellis_mad(uint32_t state, uint32_t multiplier,
                                    uint32_t increment) {
    const uint32_t product = (state & 0x00ffffffu) * (multiplier & 0x00ffffffu);
    const uint32_t masked = (product + increment) & 0x3f3f3f3fu;
    const uint32_t pairs = (masked & 0x00ff00ffu) + ((masked >> 8) & 0x00ff00ffu);
    const int value = static_cast<int>((pairs & 0xffffu) + (pairs >> 16)) - 126;
    return static_cast<int8_t>(value);
}

R3D_IQR_HD inline uint32_t packed_code(const uint8_t* codes, uint64_t row_stride,
                                      uint64_t row, uint64_t column, unsigned bits) {
    const uint64_t bit = column * bits;
    const uint8_t* source = codes + row * row_stride + bit / 8;
    const unsigned shift = static_cast<unsigned>(bit % 8);
    uint32_t word = source[0];
    if (shift + bits > 8) word |= static_cast<uint32_t>(source[1]) << 8;
    return (word >> shift) & ((1u << bits) - 1u);
}

R3D_IQR_HD inline uint32_t trellis_state(const uint8_t* codes, uint64_t row_stride,
                                        uint64_t row, uint64_t column, unsigned bits,
                                        unsigned state_bits, unsigned run_rows) {
    const uint64_t run_mask = run_rows - 1u;
    const uint64_t run_start = row & ~run_mask;
    const unsigned position = static_cast<unsigned>(row - run_start);
    uint32_t state = 0;
    for (unsigned shift = 0, back = 0; shift < state_bits; shift += bits, ++back) {
        const unsigned previous = (position + run_rows - back) & static_cast<unsigned>(run_mask);
        state |= packed_code(codes, row_stride, run_start + previous, column, bits) << shift;
    }
    return state & ((1u << state_bits) - 1u);
}

}
