#pragma once
#include "../libr3d/r3dh_moe_iqr.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace iqr_fixture {
inline void check(bool ok, const char* expr, int line) {
    if (!ok) {
        std::fprintf(stderr, "IQR fixture line %d: %s\n", line, expr);
        std::exit(1);
    }
}
#define IQR_CHECK(expr) ::iqr_fixture::check((expr), #expr, __LINE__)
using Params = std::array<RadParam, 12>;
inline Params params(int even, int odd) {
    return {RAD_INT("M", 2), RAD_INT("N", even), RAD_INT("N_odd", odd), RAD_INT("K", 2560),
        RAD_INT("K_odd", 2560), RAD_INT("n_expert", 512), RAD_INT("top_k", 10), RAD_INT("group", 128),
        RAD_INT("a_group", 128), RAD_INT("parts", 2), RAD_STR("dtype", "i8a8h"), RAD_STR("a_order", "token")};
}
inline RadTensor matrix(void* data, uint32_t type, int rows, int columns) {
    RadTensor result{};
    result.data = data;
    result.dtype = type;
    result.rank = 2;
    result.shape[0] = rows;
    result.shape[1] = result.stride[0] = columns;
    result.stride[1] = 1;
    return result;
}
struct Expert {
    int width;
    RadEncoding enc{};
    uint32_t coeff[2];
    std::vector<uint8_t> codes;
    std::vector<int8_t> book;
    std::vector<float> row, column;
    Expert(int width, unsigned seed) : width(width), coeff{0x19660du + 2 * seed, 0x3c6ef35fu - seed},
        codes(size_t(width) * 1280), book(65536), row(width), column(20) {
        rad_enc_copy_str(enc.scheme, "trellis");
        rad_enc_copy_str(enc.transform, "fwht128");
        rad_enc_add_plane(&enc, "codes", RAD_U4, 1, 1);
        rad_enc_add_table(&enc, "trellis", RAD_I8, 65536, 1);
        enc.plane[1].block[0] = 128;
        enc.plane[1].block[1] = 1;
        rad_enc_add_table(&enc, "trellis.mad", RAD_U32, 2, 1);
        rad_enc_add_plane(&enc, "scale", RAD_F32, 1, 0);
        rad_enc_add_plane(&enc, "scale.col", RAD_F32, 0, 128);
        for (unsigned index = 0; index < book.size(); ++index) {
            const uint32_t word = index * coeff[0] + coeff[1];
            int sum = -126;
            for (int shift = 0; shift < 32; shift += 8) sum += (word >> shift) & 63u;
            book[index] = static_cast<int8_t>(sum);
        }
        uint32_t random = seed;
        for (auto& packed : codes) {
            random = 1664525u * random + 1013904223u;
            packed = random >> 24;
        }
        for (int index = 0; index < width; ++index) row[index] = float(1 + (index + seed) % 11) / 97;
        for (int group = 0; group < 20; ++group) column[group] = float(2 + (group + seed) % 7) / 53;
    }
    std::array<RadTensor, 5> planes() {
        return {matrix(codes.data(), RAD_U4, width, 2560), matrix(book.data(), RAD_I8, 65536, 1),
            matrix(coeff, RAD_U32, 2, 1), matrix(row.data(), RAD_F32, width, 1), matrix(column.data(), RAD_F32, 1, 20)};
    }
};
constexpr int plane_ids[]{0, 1, 2, 3, 4};
inline std::vector<uint8_t> pack(const Params& geometry, Expert& expert, int operand,
                                 RadLayoutFn layout, RadRelayoutFn relayout) {
    const bool weight = operand == 2 || operand == 6;
    const int first = weight ? 0 : 3, count = weight ? 3 : 2;
    const auto planes = expert.planes();
    RadLayout result{};
    IQR_CHECK(layout(geometry.data(), geometry.size(), operand, &expert.enc, plane_ids + first,
                     planes.data() + first, count, &result) == RAD_OK);
    IQR_CHECK(result.bytes > 0 && result.align == 256 && result.shape[0] == expert.width);
    std::vector<uint8_t> bytes(result.bytes);
    IQR_CHECK(relayout(geometry.data(), geometry.size(), operand, &expert.enc, plane_ids + first,
                       planes.data() + first, count, bytes.data(), bytes.size()) == RAD_OK);
    return bytes;
}
}