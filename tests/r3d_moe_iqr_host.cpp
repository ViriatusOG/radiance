#include "r3d_moe_iqr_fixture.h"
#include <algorithm>
#include <limits>

using namespace iqr_fixture;
static void roundtrip(const Params& geometry, int parity) {
    const int operand = 2 + 4 * parity;
    const int width = geometry[1 + parity].ival;
    Expert original(width, 7), copy(width, 33);
    auto packed = pack(geometry, original, operand, r3dh_moe_iqr_layout, r3dh_moe_iqr_relayout);
    auto scales = pack(geometry, original, operand + 1, r3dh_moe_iqr_layout, r3dh_moe_iqr_relayout);
    auto planes = copy.planes();
    auto unpack = [&](int port, int first, int count, const std::vector<uint8_t>& data) {
        return r3dh_moe_iqr_unrelayout(geometry.data(), geometry.size(), port, &copy.enc, plane_ids + first,
            data.data(), data.size(), planes.data() + first, count);
    };
    IQR_CHECK(unpack(operand, 0, 3, packed) == RAD_OK && unpack(operand + 1, 3, 2, scales) == RAD_OK);
    IQR_CHECK(original.codes == copy.codes && original.book == copy.book && original.row == copy.row
        && original.column == copy.column && original.coeff[0] == copy.coeff[0] && original.coeff[1] == copy.coeff[1]);
    packed.front() ^= 1;
    IQR_CHECK(unpack(operand, 0, 3, packed) == RAD_E_FORMAT);
    packed.pop_back();
    IQR_CHECK(unpack(operand, 0, 3, packed) == RAD_E_SHAPE);
    const int duplicate[]{0, 1, 1};
    RadLayout layout{};
    auto inspect = [&](int port, const int* ids) {
        return r3dh_moe_iqr_layout(geometry.data(), geometry.size(), port, &copy.enc, ids, planes.data(), 3, &layout);
    };
    IQR_CHECK(inspect(operand, duplicate) == RAD_E_SHAPE && inspect(0, plane_ids) == RAD_E_UNSUPPORTED);
    copy.enc.plane[1].block[0] = 64;
    IQR_CHECK(inspect(operand, plane_ids) == RAD_E_DTYPE);
    copy.enc.plane[1].block[0] = 128;
    copy.book.back() ^= 1;
    packed.push_back(0);
    IQR_CHECK(r3dh_moe_iqr_relayout(geometry.data(), geometry.size(), operand, &copy.enc, plane_ids, planes.data(), 3,
        packed.data(), packed.size()) == RAD_E_FORMAT);
    copy.column.back() = std::numeric_limits<float>::infinity();
    IQR_CHECK(r3dh_moe_iqr_relayout(geometry.data(), geometry.size(), operand + 1, &copy.enc, plane_ids + 3,
        planes.data() + 3, 2, scales.data(), scales.size()) == RAD_E_FORMAT);
}
int main() {
    for (auto geometry : {params(768, 512), params(512, 768), params(1280, 1280)}) {
        RadOpdDesc desc{};
        IQR_CHECK(r3dh_moe_iqr_shape(geometry.data(), geometry.size(), 8, &desc) == RAD_OK);
        IQR_CHECK(desc.dtype == RAD_BF16 && desc.shape[0] == 20 && desc.shape[1] == std::max(geometry[1].ival, geometry[2].ival));
        for (int port : {2, 3, 6, 7}) {
            IQR_CHECK(r3dh_moe_iqr_shape(geometry.data(), geometry.size(), port, &desc) == RAD_OK);
            IQR_CHECK(desc.rank == 3 && desc.shape[0] == 256 && desc.shape[1] == geometry[1 + (port >= 6)].ival
                && desc.shape[2] == (port % 2 ? 20 : 2560));
        }
        IQR_CHECK(r3dh_moe_iqr_shape(geometry.data(), geometry.size(), 5, &desc) == RAD_OK
            && desc.flags == RAD_OPD_F_IDX_PREFIX && desc.idx_max == 21);
        IQR_CHECK(r3dh_moe_iqr_shape(geometry.data(), geometry.size(), 9, &desc) == RAD_OK && desc.flags == RAD_OPD_F_ABSENT);
        roundtrip(geometry, 0);
        roundtrip(geometry, 1);
    }
    auto rejects = [](const Params& geometry) {
        RadOpdDesc desc{};
        return r3dh_moe_iqr_shape(geometry.data(), geometry.size(), 8, &desc) != RAD_OK;
    };
    IQR_CHECK(rejects(params(512, 512)) && rejects(params(1280, 768)));
    auto geometry = params(768, 512);
    for (size_t index : {0u, 3u, 4u, 5u, 6u, 7u, 8u, 9u}) {
        auto invalid = geometry;
        invalid[index].ival = index == 5 ? 510 : 0;
        IQR_CHECK(rejects(invalid));
    }
    geometry[10] = RAD_STR("dtype", "w4nl64a8i8h");
    IQR_CHECK(rejects(geometry));
    geometry = params(512, 768);
    geometry[11] = RAD_STR("a_order", "sorted");
    IQR_CHECK(rejects(geometry));
    geometry = params(768, 512);
    geometry[0] = RAD_RANGE("M", 1, 2048);
    RadOpdDesc desc{};
    IQR_CHECK(r3dh_moe_iqr_shape(geometry.data(), geometry.size(), 8, &desc) == RAD_OK && desc.shape[0] == 20480);
    r3dh_moe_iqr::Config launch{};
    IQR_CHECK(r3dh_moe_iqr::config(geometry.data(), geometry.size(), &launch) == RAD_E_SHAPE);
    geometry[0] = RAD_RANGE("M", 0, 2);
    IQR_CHECK(rejects(geometry));
}