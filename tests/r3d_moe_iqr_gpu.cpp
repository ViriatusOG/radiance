#include "r3d_moe_iqr_fixture.h"
#include "../core/rad_internal.h"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <dlfcn.h>
#include <string>

namespace {
using namespace iqr_fixture;
void hip_check(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(status));
        std::exit(1);
    }
}
struct Buffers {
    std::vector<void*> owned;
    template<class Value> void* upload(const std::vector<Value>& values) {
        void* device = nullptr;
        const size_t bytes = values.size() * sizeof(Value);
        hip_check(hipMalloc(&device, bytes));
        owned.push_back(device);
        hip_check(hipMemcpy(device, values.data(), bytes, hipMemcpyHostToDevice));
        return device;
    }
    ~Buffers() { for (void* data : owned) hip_check(hipFree(data)); }
};
RadTensor table(void* data, uint32_t dtype, int rows, int columns, int pitch) {
    RadTensor result{};
    result.data = data;
    result.dtype = dtype;
    result.rank = 3;
    result.shape[0] = 256;
    result.shape[1] = rows;
    result.shape[2] = result.stride[1] = columns;
    result.stride[0] = pitch;
    result.stride[2] = 1;
    return result;
}
RadTensor vector_tensor(void* data, int count) {
    RadTensor result{};
    result.data = data;
    result.dtype = RAD_I32;
    result.rank = 1;
    result.shape[0] = count;
    result.stride[0] = 1;
    return result;
}
uint16_t narrow(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const unsigned lower = bits & 65535, upper = bits >> 16;
    return upper + (lower > 32768 || (lower == 32768 && (upper & 1)));
}
std::vector<int8_t> decode(const Expert& expert) {
    std::vector<int8_t> result(size_t(expert.width) * 2560);
    for (int column = 0; column < 2560; ++column) {
        auto nibble = [&](int row) { return (expert.codes[size_t(row) * 1280 + column / 2] >> (4 * (column % 2))) & 15; };
        for (int run = 0; run < expert.width; run += 128) {
            uint16_t state = 0;
            for (int row = run + 125; row < run + 128; ++row) state = (state << 4) | nibble(row);
            for (int row = run; row < run + 128; ++row) {
                state = (state << 4) | nibble(row);
                result[size_t(row) * 2560 + column] = expert.book[state];
            }
        }
    }
    return result;
}
void compare(const RadKernelInfo& kernel, int even, int odd, hipStream_t stream) {
    auto geometry = params(even, odd);
    if (even == 768) geometry[0] = RAD_INT("M", 7);
    const int width = std::max(even, odd), pitch = width + 7;
    constexpr int input_pitch = 2563, scale_pitch = 23, rows = 20;
    constexpr uint16_t sentinel = 0x7f95;
    std::vector<int8_t> input(2 * input_pitch);
    std::vector<float> factors(2 * scale_pitch);
    for (int index = 0; index < int(input.size()); ++index) input[index] = (index * 31 % 249) - 124;
    for (int index = 0; index < int(factors.size()); ++index) factors[index] = float(1 + index % 13) / 37;
    const std::vector<int32_t> slots{19, 2, 17, -1, 4, 13, 9, 11, 6, 0, 10, 1, 12, 3, 14, 5, 20, 7, 18, 8};
    const std::array<int, 3> ids{2, 5, 510}, ends{3, 8, 18};
    std::vector<int32_t> offsets(513, 0);
    for (size_t index = 0; index < ids.size(); ++index)
        std::fill(offsets.begin() + ids[index] + 1, offsets.end(), ends[index]);
    Buffers buffers;
    RadTensor tensors[10]{};
    std::vector<uint16_t> expected(rows * pitch, sentinel), actual = expected;
    for (int row = 0; row < rows; ++row) std::fill_n(expected.begin() + row * pitch, width, 0);
    tensors[0] = matrix(buffers.upload(input), RAD_I8, 2, 2560);
    tensors[0].stride[0] = input_pitch;
    tensors[1] = matrix(buffers.upload(factors), RAD_F32, 2, 20);
    tensors[1].stride[0] = scale_pitch;
    tensors[4] = vector_tensor(buffers.upload(slots), rows);
    tensors[5] = vector_tensor(buffers.upload(offsets), 513);
    tensors[8] = matrix(buffers.upload(actual), RAD_BF16, rows, width);
    tensors[8].stride[0] = pitch;
    for (int parity = 0; parity < 2; ++parity) {
        const int local = parity ? odd : even, port = 2 + 4 * parity;
        const int scale_bytes = (local + 20) * sizeof(float) + 16;
        std::vector<void*> weights(256), scales(256);
        std::vector<uint8_t> strided(256 * scale_bytes);
        for (size_t run = 0; run < ids.size(); ++run) {
            const int expert_id = ids[run];
            if ((expert_id & 1) != parity) continue;
            Expert expert(local, expert_id + 17);
            weights[expert_id / 2] = buffers.upload(pack(geometry, expert, port, kernel.layout, kernel.relayout));
            const auto scale = pack(geometry, expert, port + 1, kernel.layout, kernel.relayout);
            if (parity) scales[expert_id / 2] = buffers.upload(scale);
            else std::copy(scale.begin(), scale.end(), strided.begin() + (expert_id / 2) * scale_bytes);
            const auto decoded = decode(expert);
            for (int row = run ? ends[run - 1] : 0; row < ends[run]; ++row) {
                if (slots[row] < 0 || slots[row] >= rows) continue;
                const int token = slots[row] / 10;
                for (int weight_row = 0; weight_row < local; ++weight_row) {
                    float sum = 0;
                    for (int group = 0; group < 20; ++group) {
                        int dot = 0;
                        for (int inner = 128 * group; inner < 128 * (group + 1); ++inner)
                            dot += input[token * input_pitch + inner] * decoded[weight_row * 2560 + inner];
                        sum += (dot * factors[token * scale_pitch + group]) * expert.column[group];
                    }
                    const int column = weight_row < local / 2 ? weight_row : width / 2 + weight_row - local / 2;
                    expected[row * pitch + column] = narrow(sum * expert.row[weight_row]);
                }
            }
        }
        tensors[port] = table(buffers.upload(weights), r3dh_moe_iqr::weight_dtype, local, 2560, 0);
        tensors[port + 1] = table(parity ? buffers.upload(scales) : buffers.upload(strided),
            r3dh_moe_iqr::scale_dtype, local, parity ? 20 : 2560, parity ? 0 : scale_bytes);
    }
    RadArgs args{};
    args.t = tensors;
    args.n_t = even == 512 ? 10 : 9;
    args.p = geometry.data();
    args.n_p = geometry.size();
    tensors[2].dtype = RAD_I8;
    IQR_CHECK(kernel.launch(&args, stream) == RAD_E_DTYPE);
    tensors[2].dtype = r3dh_moe_iqr::weight_dtype;
    IQR_CHECK(kernel.launch(&args, stream) == RAD_OK);
    hip_check(hipStreamSynchronize(stream));
    hip_check(hipMemcpy(actual.data(), tensors[8].data, actual.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    for (size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != expected[index]) {
            std::fprintf(stderr, "%s sorted row %zu col %zu: got %04x expected %04x\n", kernel.name,
                         index / pitch, index % pitch, actual[index], expected[index]);
            std::exit(1);
        }
    }
    if (even == 768) {
        std::fill(offsets.begin(), offsets.end(), 0);
        std::fill(actual.begin(), actual.end(), sentinel);
        tensors[0].shape[0] = tensors[1].shape[0] = 1;
        tensors[4].shape[0] = tensors[8].shape[0] = 10;
        tensors[5].data = buffers.upload(offsets);
        tensors[8].data = buffers.upload(actual);
        IQR_CHECK(kernel.launch(&args, stream) == RAD_OK);
        hip_check(hipStreamSynchronize(stream));
        hip_check(hipMemcpy(actual.data(), tensors[8].data, actual.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < pitch; ++column)
                IQR_CHECK(actual[row * pitch + column] == (row < 10 && column < width ? 0 : sentinel));
    }
}
}

int main(int argc, char** argv) {
    const bool catalogue = argc == 3 && !std::strcmp(argv[2], "--catalogue-only");
    IQR_CHECK(argc == 2 || catalogue);
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
    const auto row_count = reinterpret_cast<decltype(&rad_kernel_count)>(dlsym(library, "rad_kernel_count"));
    const auto row_at = reinterpret_cast<decltype(&rad_kernel_at)>(dlsym(library, "rad_kernel_at"));
    IQR_CHECK(row_count && row_at);
    auto match = [&](const iqr_fixture::Params& params) {
        rad::Geometry geometry;
        for (const auto& param : params) {
            if (param.kind == RAD_P_INT) geometry.set_i(param.key, param.ival);
            else geometry.set_s(param.key, param.sval);
        }
        const RadKernelInfo* row = nullptr;
        for (int index = 0; index < row_count(); ++index) {
            const auto* candidate = row_at(index);
            if (candidate->domain != RAD_DOMAIN_DEVICE || std::strcmp(candidate->op, "moe_gemm_q")) continue;
            if (std::all_of(candidate->constraints, candidate->constraints + candidate->n_constraints,
                    [&](const RadConstraint& constraint) { return rad::constraint_holds(constraint, geometry); })) {
                IQR_CHECK(!row);
                row = candidate;
            }
        }
        return row;
    };
    constexpr std::array<int, 3> even_widths{512, 768, 1280};
    std::array<const RadKernelInfo*, even_widths.size()> rows{};
    for (size_t index = 0; index < rows.size(); ++index) {
        const int even = even_widths[index];
        const int odd = even == 1280 ? 1280 : 1280 - even;
        const auto geometry = iqr_fixture::params(even, odd);
        const std::string name = "model_moe_iqr_" + std::to_string(even) + "_" + std::to_string(odd) + "_strict";
        const auto* row = rows[index] = match(geometry);
        IQR_CHECK(row && name == row->name && row->layout && row->relayout && row->unrelayout
            && row->opd_shape && row->launch && !row->scratch);
        RadOpdDesc desc{};
        for (int operand : {0, 1}) {
            IQR_CHECK(row->opd_shape(geometry.data(), geometry.size(), operand, &desc) == RAD_OK);
            IQR_CHECK(desc.rank == 2 && desc.shape[0] == 2 && desc.shape[1] == (operand ? 20 : 2560)
                && desc.dtype == (operand ? RAD_F32 : RAD_I8));
        }
        IQR_CHECK(row->opd_shape(geometry.data(), geometry.size(), 4, &desc) == RAD_OK
            && desc.rank == 1 && desc.shape[0] == 20 && desc.dtype == RAD_I32
            && desc.flags == RAD_OPD_F_IDX_UNIQUE && desc.idx_max == 20);
        for (int operand : {-1, 10})
            IQR_CHECK(row->opd_shape(geometry.data(), geometry.size(), operand, &desc) == RAD_E_INVAL);
        std::array<RadTensor, 10> empty{};
        RadArgs args{};
        args.p = geometry.data();
        args.n_p = geometry.size();
        args.t = empty.data();
        for (int arity : {8, 9, 10, 11}) {
            args.n_t = arity;
            IQR_CHECK(row->launch(&args, nullptr) == (arity == 9 || arity == 10 ? RAD_E_SHAPE : RAD_E_INVAL));
        }
        auto invalid = geometry;
        for (int tokens : {0, 1, 2048, 2049}) {
            invalid[0] = RAD_INT("M", tokens);
            const bool supported = tokens == 1 || tokens == 2048;
            IQR_CHECK(match(invalid) == (supported ? row : nullptr));
            IQR_CHECK(row->opd_shape(invalid.data(), invalid.size(), 8, &desc) == (supported ? RAD_OK : RAD_E_SHAPE));
        }
        invalid = geometry;
        invalid[5] = RAD_INT("n_expert", 510);
        IQR_CHECK(!match(invalid));
        invalid = geometry;
        invalid[11] = RAD_STR("a_order", "sorted");
        IQR_CHECK(!match(invalid));
        invalid = geometry;
        invalid[10] = RAD_STR("dtype", "w4nl64a8i8h");
        IQR_CHECK(!match(invalid));
        invalid[10] = RAD_STR("dtype", "fp8a8f32");
        const auto* fallback = match(invalid);
        IQR_CHECK(fallback && !std::strcmp(fallback->name, "moe_gemm_q_e4m3"));
    }
    IQR_CHECK(!match(iqr_fixture::params(512, 512)));
    if (catalogue) { dlclose(library); return 0; }
    int count = 0;
    const auto status = hipGetDeviceCount(&count);
    if (status == hipErrorNoDevice) return 77;
    hip_check(status);
    int chosen = -1;
    for (int device = 0; device < count; ++device) {
        hipDeviceProp_t props{};
        hip_check(hipGetDeviceProperties(&props, device));
        if (std::string(props.gcnArchName).substr(0, 7) == "gfx1100") { chosen = device; break; }
    }
    if (chosen < 0) { std::fprintf(stderr, "no visible gfx1100\n"); return 77; }
    hip_check(hipSetDevice(chosen));
    hipStream_t stream{};
    hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    compare(*rows[0], 512, 768, stream);
    compare(*rows[1], 768, 512, stream);
    compare(*rows[2], 1280, 1280, stream);
    hip_check(hipStreamDestroy(stream));
    dlclose(library);
}