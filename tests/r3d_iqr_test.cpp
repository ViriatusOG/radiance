#include "rad_test.h"
#include "r3d_iqr.h"
#include "rad_plugin.h"
#include "rad_encoding.h"

extern "C" int r3d_rad_layout_iqr(const RadParam*, int, int, const RadEncoding*,
                                 const int*, const RadTensor*, int, RadLayout*);
extern "C" int r3d_rad_relayout_iqr(const RadParam*, int, int, const RadEncoding*,
                                   const int*, const RadTensor*, int, void*, int64_t);
extern "C" int r3d_rad_unrelayout_iqr(const RadParam*, int, int, const RadEncoding*,
                                     const int*, const void*, int64_t, const RadTensor*, int);

#include <array>
#include <cstdint>

TEST(r3d_iqr_bf16_group_scale_contract) {
    RadEncoding encoding{};
    rad_enc_copy_str(encoding.scheme, "trellis");
    rad_enc_copy_str(encoding.transform, "fwht128");
    rad_enc_add_plane(&encoding, "codes", RAD_U4, 1, 1);
    rad_enc_add_plane(&encoding, "scale", RAD_BF16, 1, 128);
    const int trellis = rad_enc_add_table(&encoding, "trellis", RAD_I8, 65536, 1);
    encoding.plane[trellis].block[0] = 128;
    encoding.plane[trellis].block[1] = 1;
    rad_enc_add_table(&encoding, "trellis.mad", RAD_U32, 2, 1);
    std::vector<uint16_t> scales(256 * 2, 0x3c01);
    RadTensor plane{};
    plane.data = scales.data(); plane.dtype = RAD_BF16; plane.rank = 2;
    plane.shape[0] = 256; plane.shape[1] = 2; rad_tensor_pack(&plane);
    const int selected[]{1};
    const RadParam params[]{RAD_INT("N", 256), RAD_INT("K", 256),
        RAD_INT("group", 128), RAD_STR("dtype", "i8a8h")};
    RadLayout layout{};
    REQUIRE_EQ(r3d_rad_layout_iqr(params, 4, 3, &encoding, selected, &plane, 1, &layout), RAD_OK);
    CHECK_EQ(layout.dtype, 0x33000300u);
    CHECK_EQ(layout.bytes, 1024);
}

namespace {

int reference_mad(uint32_t state, uint32_t multiplier, uint32_t increment) {
    const uint64_t product = static_cast<uint64_t>(state % 16777216u)
                           * (multiplier % 16777216u) + increment;
    const uint32_t wrapped = static_cast<uint32_t>(product);
    int total = -126;
    for (unsigned shift = 0; shift < 32; shift += 8)
        total += static_cast<int>((wrapped >> shift) & 63u);
    return total;
}

}

TEST(r3d_iqr_bf16_group_roundtrip_and_rejects) {
    constexpr int rows = 256, reduction = 32768, groups = reduction / 128;
    auto tensor = [](void* data, uint32_t dtype, int height, int width) {
        RadTensor result{}; result.data = data; result.dtype = dtype; result.rank = 2;
        result.shape[0] = height; result.shape[1] = width; rad_tensor_pack(&result);
        return result;
    };
    for (unsigned bits : {3u, 4u, 5u}) for (unsigned run : {128u, 256u}) {
        const uint32_t code_dtype = bits == 3 ? RAD_U3 : bits == 4 ? RAD_U4 : RAD_U5;
        RadEncoding encoding{};
        rad_enc_copy_str(encoding.scheme, "trellis"); rad_enc_copy_str(encoding.transform, "fwht128");
        rad_enc_add_plane(&encoding, "codes", code_dtype, 1, 1);
        rad_enc_add_plane(&encoding, "scale", RAD_BF16, 1, 128);
        const int trellis = rad_enc_add_table(&encoding, "trellis", RAD_I8, 65536, 1);
        encoding.plane[trellis].block[0] = run; encoding.plane[trellis].block[1] = 1;
        rad_enc_add_table(&encoding, "trellis.mad", RAD_U32, 2, 1);
        std::vector<uint8_t> codes(size_t(rows) * reduction * bits / 8);
        for (size_t index = 0; index < codes.size(); ++index) codes[index] = uint8_t(index * 197 + index / 97);
        std::vector<uint16_t> scales(rows * groups);
        for (size_t index = 0; index < scales.size(); ++index) scales[index] = uint16_t(index);
        uint32_t coefficients[]{0xffffffffu, 0xfffffffdu};
        std::vector<int8_t> table(65536);
        for (uint32_t state = 0; state < table.size(); ++state)
            table[state] = int8_t(reference_mad(state, coefficients[0], coefficients[1]));
        const auto original_codes = codes, original_table = std::vector<uint8_t>(table.begin(), table.end());
        const auto original_scales = scales;
        RadTensor weight_planes[]{tensor(codes.data(), code_dtype, rows, reduction),
            tensor(table.data(), RAD_I8, 65536, 1), tensor(coefficients, RAD_U32, 2, 1)};
        RadTensor scale_plane = tensor(scales.data(), RAD_BF16, rows, groups);
        const int selected[]{0, 2, 3}, scale_selected[]{1};
        const RadParam params[]{RAD_INT("N", rows), RAD_INT("K", reduction), RAD_INT("group", 128), RAD_STR("dtype", "i8a8h")};
        RadLayout weight_layout{}, scale_layout{};
        REQUIRE_EQ(r3d_rad_layout_iqr(params, 4, 2, &encoding, selected, weight_planes, 3, &weight_layout), RAD_OK);
        REQUIRE_EQ(r3d_rad_layout_iqr(params, 4, 3, &encoding, scale_selected, &scale_plane, 1, &scale_layout), RAD_OK);
        CHECK_EQ(weight_layout.dtype, r3d_iqr::weight_dtype);
        CHECK_EQ(scale_layout.dtype, r3d_iqr::group_scale_dtype);
        CHECK_EQ(scale_layout.bytes, int64_t(scales.size() * 2));
        std::vector<uint8_t> packed_weight(weight_layout.bytes + 128, 0x5a), packed_scale(scale_layout.bytes + 128, 0x5a);
        REQUIRE_EQ(r3d_rad_relayout_iqr(params, 4, 2, &encoding, selected, weight_planes, 3,
            packed_weight.data() + 64, weight_layout.bytes), RAD_OK);
        REQUIRE_EQ(r3d_rad_relayout_iqr(params, 4, 3, &encoding, scale_selected, &scale_plane, 1,
            packed_scale.data() + 64, scale_layout.bytes), RAD_OK);
        CHECK_EQ(std::memcmp(packed_scale.data() + 64, scales.data(), scale_layout.bytes), 0);
        std::fill(codes.begin(), codes.end(), 0); std::fill(table.begin(), table.end(), 0);
        std::fill(scales.begin(), scales.end(), 0); coefficients[0] = coefficients[1] = 0;
        REQUIRE_EQ(r3d_rad_unrelayout_iqr(params, 4, 2, &encoding, selected, packed_weight.data() + 64,
            weight_layout.bytes, weight_planes, 3), RAD_OK);
        REQUIRE_EQ(r3d_rad_unrelayout_iqr(params, 4, 3, &encoding, scale_selected, packed_scale.data() + 64,
            scale_layout.bytes, &scale_plane, 1), RAD_OK);
        CHECK(codes == original_codes); CHECK(scales == original_scales);
        CHECK_EQ(coefficients[0], 0xffffffffu); CHECK_EQ(coefficients[1], 0xfffffffdu);
        for (size_t index = 0; index < table.size(); ++index) CHECK_EQ(uint8_t(table[index]), original_table[index]);
        for (size_t index = 0; index < 64; ++index) {
            CHECK_EQ(packed_weight[index], 0x5a); CHECK_EQ(packed_weight[packed_weight.size() - 1 - index], 0x5a);
            CHECK_EQ(packed_scale[index], 0x5a); CHECK_EQ(packed_scale[packed_scale.size() - 1 - index], 0x5a);
        }
        auto reject = [&](auto mutate) {
            auto bad = encoding; mutate(bad);
            CHECK(r3d_rad_layout_iqr(params, 4, 2, &bad, selected, weight_planes, 3, &weight_layout) != RAD_OK);
            CHECK(r3d_rad_layout_iqr(params, 4, 3, &bad, scale_selected, &scale_plane, 1, &scale_layout) != RAD_OK);
        };
        reject([](auto& bad) { bad.n_planes = 3; });
        reject([](auto& bad) { bad.n_planes = 5; });
        reject([](auto& bad) { bad.n_planes = RAD_ENC_MAX_PLANES + 1; });
        reject([](auto& bad) { rad_enc_copy_str(bad.scheme, "affine"); });
        reject([](auto& bad) { rad_enc_copy_str(bad.plane[2].role, "codes"); });
        reject([](auto& bad) { rad_enc_copy_str(bad.plane[3].role, "scale"); });
        reject([](auto& bad) { bad.plane[0].dtype = RAD_U2; });
        reject([](auto& bad) { bad.plane[1].dtype = RAD_F32; });
        reject([](auto& bad) { bad.plane[1].block[1] = 64; });
        reject([](auto& bad) { bad.plane[1].block[0] = 128; });
        reject([](auto& bad) { bad.plane[2].extent[0] = 65535; });
        reject([](auto& bad) { rad_enc_copy_str(bad.transform, "fwht64"); });
        const int duplicate[]{0, 0, 3}, wrong_scale[]{0};
        CHECK_EQ(r3d_rad_layout_iqr(params, 4, 6, &encoding, selected, weight_planes, 3, &weight_layout), RAD_E_DTYPE);
        CHECK_EQ(r3d_rad_layout_iqr(params, 4, 2, &encoding, duplicate, weight_planes, 3, &weight_layout), RAD_E_SHAPE);
        CHECK_EQ(r3d_rad_layout_iqr(params, 4, 3, &encoding, wrong_scale, &scale_plane, 1, &scale_layout), RAD_E_SHAPE);
        CHECK_EQ(r3d_rad_relayout_iqr(params, 4, 3, &encoding, scale_selected, &scale_plane, 1,
            packed_scale.data() + 64, int64_t(original_scales.size() * 2 - 1)), RAD_E_SHAPE);
        table[65535] ^= 1;
        CHECK_EQ(r3d_rad_relayout_iqr(params, 4, 2, &encoding, selected, weight_planes, 3,
            packed_weight.data() + 64, int64_t(packed_weight.size() - 128)), RAD_E_FORMAT);
    }
}

TEST(r3d_iqr_mad_exhaustive_states) {
    const std::array<std::array<uint32_t, 2>, 6> coefficients{{
        {{0u, 0u}}, {{1u, 0u}}, {{0xffffffffu, 0xffffffffu}},
        {{0x9e3779b9u, 0x7f4a7c15u}}, {{0x01010101u, 0x12345678u}},
        {{0x80000001u, 0x80000000u}}
    }};
    for (const auto& coefficient : coefficients)
        for (uint32_t state = 0; state < 65536; ++state)
            CHECK_EQ(static_cast<int>(r3d_iqr::trellis_mad(
                         state, coefficient[0], coefficient[1])),
                     reference_mad(state, coefficient[0], coefficient[1]));
}

TEST(r3d_iqr_mad_unsigned_wrap_boundaries) {
    const std::array<uint32_t, 6> states{{
        0u, 65535u, 65536u, 0x7fffffffu, 0x80000000u, 0xffffffffu
    }};
    for (uint32_t state : states)
        CHECK_EQ(static_cast<int>(r3d_iqr::trellis_mad(state, 0xffffffffu, 0xfffffffeu)),
                 reference_mad(state, 0xffffffffu, 0xfffffffeu));
}

TEST(r3d_iqr_mad_ignores_high_multiplier_bytes) {
    for (uint32_t state = 0; state < 65536; ++state)
        CHECK_EQ(r3d_iqr::trellis_mad(state | 0xab000000u, 0xcd123457u, 0xfedcba98u),
                 r3d_iqr::trellis_mad(state, 0x00123457u, 0xfedcba98u));
}

TEST(r3d_iqr_packed_codes_and_tail_biting) {
    for (unsigned bits : {3u, 4u, 5u}) {
        for (unsigned run : {128u, 256u}) {
            const unsigned rows = run * 2;
            const unsigned columns = 131;
            const unsigned stride = (columns * bits + 7) / 8;
            std::vector<uint8_t> packed(static_cast<size_t>(rows) * stride, 0);
            std::vector<uint32_t> logical(static_cast<size_t>(rows) * columns);
            uint32_t random_state = 0xabcdef01u;
            for (unsigned row = 0; row < rows; ++row) {
                for (unsigned column = 0; column < columns; ++column) {
                    random_state = random_state * 1664525u + 1013904223u;
                    const uint32_t code = (random_state >> 16) & ((1u << bits) - 1u);
                    logical[row * columns + column] = code;
                    for (unsigned code_bit = 0; code_bit < bits; ++code_bit) {
                        const unsigned position = column * bits + code_bit;
                        if ((code >> code_bit) & 1u)
                            packed[row * stride + position / 8] |= static_cast<uint8_t>(1u << (position % 8));
                    }
                    CHECK_EQ(r3d_iqr::packed_code(packed.data(), stride, row, column, bits), code);
                }
            }
            for (unsigned start = 0; start < rows; start += run) {
                for (unsigned column = 0; column < columns; ++column) {
                    uint32_t sequential = 0;
                    const unsigned history = (16 + bits - 1) / bits;
                    for (unsigned tail = run - history; tail < run; ++tail)
                        sequential = ((sequential << bits)
                                      | logical[(start + tail) * columns + column]) & 65535u;
                    for (unsigned position = 0; position < run; ++position) {
                        const unsigned row = start + position;
                        sequential = ((sequential << bits) | logical[row * columns + column]) & 65535u;
                        CHECK_EQ(r3d_iqr::trellis_state(packed.data(), stride, row, column, bits, 16, run),
                                 sequential);
                    }
                }
            }
        }
    }
}

RAD_TEST_MAIN()
TEST(r3d_iqr_layout_roundtrip_and_strict_rejection) {
    for (unsigned bits : {3u, 4u, 5u}) {
        const uint32_t codes_type = bits == 3 ? RAD_U3 : bits == 4 ? RAD_U4 : RAD_U5;
        RadEncoding encoding{};
        rad_enc_copy_str(encoding.scheme, "trellis");
        rad_enc_copy_str(encoding.transform, "fwht128");
        rad_enc_add_plane(&encoding, "codes", codes_type, 1, 1);
        rad_enc_add_plane(&encoding, "scale", RAD_F32, 1, 0);
        const int trellis = rad_enc_add_table(&encoding, "trellis", RAD_I8, 65536, 1);
        encoding.plane[trellis].block[0] = 256;
        encoding.plane[trellis].block[1] = 1;
        rad_enc_add_table(&encoding, "trellis.mad", RAD_U32, 2, 1);
        rad_enc_add_plane(&encoding, "scale.col", RAD_F32, 0, 128);
        std::vector<uint8_t> codes(256 * 256 * bits / 8);
        std::vector<int8_t> table(65536);
        std::vector<float> scales(256), column_scales(2);
        uint32_t coefficients[2]{0xcd123457u, 0xfedcba98u};
        for (size_t index = 0; index < codes.size(); ++index) codes[index] = (index * 71 + 23) % 256;
        for (uint32_t state = 0; state < 65536; ++state)
            table[state] = static_cast<int8_t>(reference_mad(state, coefficients[0], coefficients[1]));
        for (size_t index = 0; index < scales.size(); ++index) scales[index] = (index + 1) / 8192.0f;
        column_scales[0] = 0.5f; column_scales[1] = 1.25f;
        auto make_tensor = [](void* data, uint32_t dtype, int64_t rows, int64_t columns) {
            RadTensor tensor{};
            tensor.data = data; tensor.dtype = dtype; tensor.rank = 2;
            tensor.shape[0] = rows; tensor.shape[1] = columns;
            rad_tensor_pack(&tensor); return tensor;
        };
        const int weight_selected[3]{0, 2, 3}, scale_selected[2]{1, 4};
        RadTensor weight_planes[3]{make_tensor(codes.data(), codes_type, 256, 256),
            make_tensor(table.data(), RAD_I8, 65536, 1), make_tensor(coefficients, RAD_U32, 2, 1)};
        RadTensor scale_planes[2]{make_tensor(scales.data(), RAD_F32, 256, 1),
            make_tensor(column_scales.data(), RAD_F32, 1, 2)};
        const RadParam parameters[]{RAD_INT("N", 256), RAD_INT("K", 256), RAD_INT("group", 128),
                                     RAD_STR("dtype", "i8a8h")};
        const int count = sizeof(parameters) / sizeof(parameters[0]);
        RadLayout weight_layout{}, scale_layout{};
        CHECK_EQ(r3d_rad_layout_iqr(parameters, count, 2, &encoding, weight_selected,
                                    weight_planes, 3, &weight_layout), RAD_OK);
        CHECK_EQ(r3d_rad_layout_iqr(parameters, count, 3, &encoding, scale_selected,
                                    scale_planes, 2, &scale_layout), RAD_OK);
        CHECK_EQ(weight_layout.bytes, static_cast<int64_t>(codes.size() + sizeof(r3d_iqr::Header)));
        CHECK_EQ(scale_layout.bytes, static_cast<int64_t>(258 * sizeof(float)));
        std::vector<uint8_t> packed_weight(weight_layout.bytes), packed_scale(scale_layout.bytes);
        CHECK_EQ(r3d_rad_relayout_iqr(parameters, count, 2, &encoding, weight_selected, weight_planes,
                                      3, packed_weight.data(), packed_weight.size()), RAD_OK);
        CHECK_EQ(r3d_rad_relayout_iqr(parameters, count, 3, &encoding, scale_selected, scale_planes,
                                      2, packed_scale.data(), packed_scale.size()), RAD_OK);
        const auto expected_codes = codes;
        const auto expected_table = table;
        const auto expected_scales = scales;
        const auto expected_column_scales = column_scales;
        std::fill(codes.begin(), codes.end(), 0);
        std::fill(table.begin(), table.end(), 0);
        std::fill(scales.begin(), scales.end(), 0);
        std::fill(column_scales.begin(), column_scales.end(), 0);
        coefficients[0] = 0; coefficients[1] = 0;
        CHECK_EQ(r3d_rad_unrelayout_iqr(parameters, count, 2, &encoding, weight_selected,
            packed_weight.data(), packed_weight.size(), weight_planes, 3), RAD_OK);
        CHECK_EQ(r3d_rad_unrelayout_iqr(parameters, count, 3, &encoding, scale_selected,
            packed_scale.data(), packed_scale.size(), scale_planes, 2), RAD_OK);
        CHECK(codes == expected_codes); CHECK(table == expected_table);
        CHECK(scales == expected_scales); CHECK(column_scales == expected_column_scales);
        table[12345] ^= 1;
        CHECK_EQ(r3d_rad_relayout_iqr(parameters, count, 2, &encoding, weight_selected, weight_planes,
            3, packed_weight.data(), packed_weight.size()), RAD_E_FORMAT);
        table[12345] ^= 1;
        CHECK_EQ(r3d_rad_relayout_iqr(parameters, count, 2, &encoding, weight_selected, weight_planes,
            3, packed_weight.data(), packed_weight.size() - 1), RAD_E_SHAPE);
        encoding.plane[trellis].block[1] = 8;
        CHECK_EQ(r3d_rad_layout_iqr(parameters, count, 2, &encoding, weight_selected,
            weight_planes, 3, &weight_layout), RAD_E_DTYPE);
        encoding.plane[trellis].block[1] = 1;
        encoding.plane[0].dtype = RAD_U2;
        CHECK_EQ(r3d_rad_layout_iqr(parameters, count, 2, &encoding, weight_selected,
            weight_planes, 3, &weight_layout), RAD_E_DTYPE);
        encoding.plane[0].dtype = codes_type;
        weight_planes[0].shape[0] = 255;
        CHECK_EQ(r3d_rad_layout_iqr(parameters, count, 2, &encoding, weight_selected,
            weight_planes, 3, &weight_layout), RAD_E_SHAPE);
    }
}
