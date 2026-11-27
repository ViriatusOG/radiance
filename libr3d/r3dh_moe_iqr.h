#pragma once
#include "rad_plugin.h"
#include "rad_encoding.h"
#include <cstdint>
#if defined(__HIPCC__) || defined(__HIP__)
#define R3DH_MOE_IQR_HD __host__ __device__
#else
#define R3DH_MOE_IQR_HD
#endif
namespace r3dh_moe_iqr {
constexpr uint32_t weight_dtype = 0x334d0100u;
constexpr uint32_t scale_dtype = 0x334d0200u;
constexpr uint32_t magic = 0x314d5149u;
constexpr const char* weight_tag = "r3dh.moe.iqr.u4.l16.r128.v1";
constexpr const char* scale_tag = "r3dh.moe.iqr.rowcol.f32.v1";
struct Header {
    uint32_t signature, version, rows, columns;
    uint32_t multiplier, increment, run, bits;
};
static_assert(sizeof(Header) == 32);
struct Config {
    int tokens;
    int width[2];
    int reduction, experts, top_k, output_width;
};
R3DH_MOE_IQR_HD inline int8_t mad(uint32_t state, uint32_t multiplier, uint32_t increment) {
    const uint32_t value = (((state & 0x00ffffffu) * (multiplier & 0x00ffffffu))
                           + increment) & 0x3f3f3f3fu;
    return static_cast<int8_t>(static_cast<int>(value & 255u)
        + static_cast<int>((value >> 8) & 255u)
        + static_cast<int>((value >> 16) & 255u)
        + static_cast<int>(value >> 24) - 126);
}
R3DH_MOE_IQR_HD inline uint32_t state(const uint8_t* codes, int columns, int row, int column) {
    uint32_t result = 0;
    for (unsigned back = 0; back < 4; ++back) {
        const unsigned previous = (static_cast<unsigned>(row) & ~127u)
            | ((static_cast<unsigned>(row) - back) & 127u);
        const uint8_t packed = codes[static_cast<uint64_t>(previous) * (columns / 2) + column / 2];
        result |= ((packed >> ((column & 1) * 4)) & 15u) << (back * 4);
    }
    return result;
}
int config(const RadParam* params, int count, Config* output, bool allow_band = false);
}
extern "C" {
int r3dh_moe_iqr_shape(const RadParam* params, int count, int operand,
    RadOpdDesc* output);
int r3dh_moe_iqr_layout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const RadTensor* planes,
    int plane_count, RadLayout* output);
int r3dh_moe_iqr_relayout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const RadTensor* planes,
    int plane_count, void* destination, int64_t bytes);
int r3dh_moe_iqr_unrelayout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const void* source,
    int64_t bytes, const RadTensor* planes, int plane_count);
int r3dh_moe_iqr_launch(const RadArgs* arguments, RadStream stream);
}
