#include "rad_device.h"
#include "rad_encoding.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(int status, const char* operation) {
    if (status != RAD_OK) throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
}

struct Library {
    void* handle;
    explicit Library(const char* path) : handle(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
        if (!handle) throw std::runtime_error(dlerror());
    }
    ~Library() { dlclose(handle); }
    const RadKernelInfo& row(const char* name) const {
        auto count = reinterpret_cast<int (*)()>(dlsym(handle, "rad_kernel_count"));
        auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(handle, "rad_kernel_at"));
        if (!count || !at) throw std::runtime_error("missing public kernel catalogue");
        for (int index = 0; index < count(); ++index) {
            const auto* info = at(index);
            if (info && !std::strcmp(info->name, name) && !std::strcmp(info->op, "gemm_nt_q")
                && info->domain == RAD_DOMAIN_DEVICE && info->launch)
                return *info;
        }
        throw std::runtime_error(std::string("missing catalogue row: ") + name);
    }
};

struct Queue {
    RadStream stream{};
    Queue() { check(rad_stream_create(&stream, 0), "create stream"); }
    ~Queue() { rad_stream_destroy(stream); }
};

struct DeviceBuffer {
    void* data;
    template<class Value> DeviceBuffer(const std::vector<Value>& host, RadStream stream)
        : data(rad_dev_alloc(host.size() * sizeof(Value), RAD_MEM_DEVICE)) {
        if (!data) throw std::bad_alloc();
        check(rad_memcpy_async(data, host.data(), host.size() * sizeof(Value), stream), "upload");
    }
    ~DeviceBuffer() { rad_dev_free(data, RAD_MEM_DEVICE); }
};

RadTensor matrix(void* data, uint32_t dtype, int rows, int columns, int pitch = 0) {
    RadTensor result{};
    result.data = data; result.dtype = dtype; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = columns;
    result.stride[0] = pitch ? pitch : columns; result.stride[1] = 1;
    return result;
}

void differential(const RadKernelInfo& row, RadStream stream, unsigned bits, int run, bool grouped, int rows) {
    constexpr int groups = 3, reduction = 128 * groups;
    const int columns = 2 * run, input_pitch = reduction + 7, scale_pitch = groups + 2, out_pitch = columns + 9;
    const RadParam params[]{RAD_STR("dtype", "i8a8h"), RAD_INT("M", 16), RAD_INT("N", columns),
                            RAD_INT("K", reduction), RAD_INT("group", 128)};
    const uint32_t dtype = bits == 3 ? RAD_U3 : bits == 4 ? RAD_U4 : RAD_U5;
    RadEncoding encoding{};
    rad_enc_copy_str(encoding.scheme, "trellis"); rad_enc_copy_str(encoding.transform, "fwht128");
    const int code_id = rad_enc_add_plane(&encoding, "codes", dtype, 1, 1);
    const int scale_id = rad_enc_add_plane(&encoding, "scale", grouped ? RAD_BF16 : RAD_F32, 1, grouped ? 128 : 0);
    const int table_id = rad_enc_add_table(&encoding, "trellis", RAD_I8, 65536, 1);
    encoding.plane[table_id].block[0] = run; encoding.plane[table_id].block[1] = 1;
    const int mad_id = rad_enc_add_table(&encoding, "trellis.mad", RAD_U32, 2, 1);
    const int col_id = grouped ? -1 : rad_enc_add_plane(&encoding, "scale.col", RAD_F32, 0, 128);

    uint32_t coefficients[]{0x9e3779b9u, 0x7f4a7c15u};
    std::vector<int8_t> table(65536);
    for (unsigned state = 0; state < table.size(); ++state) {
        uint32_t word = state * (coefficients[0] & 0xffffffu) + coefficients[1];
        int value = -126;
        for (int byte = 0; byte < 4; ++byte, word >>= 8) value += int(word & 63);
        table[state] = int8_t(value);
    }
    std::mt19937 random(573 + bits + run);
    std::vector<uint8_t> symbols(columns * reduction), codes(symbols.size() * bits / 8);
    for (size_t index = 0; index < symbols.size(); ++index) {
        symbols[index] = random() % (1u << bits);
        for (unsigned bit = 0; bit < bits; ++bit)
            if ((symbols[index] >> bit) & 1u) codes[(index * bits + bit) / 8] |= 1u << ((index * bits + bit) % 8);
    }
    std::vector<int8_t> decoded(symbols.size());
    for (int start = 0; start < columns; start += run) for (int pos = 0; pos < reduction; ++pos) {
        uint16_t state = 0;
        for (int pass = 0; pass < 2; ++pass) for (int col = start; col < start + run; ++col) {
            state = uint16_t((uint32_t(state) << bits) | symbols[col * reduction + pos]);
            if (pass) decoded[col * reduction + pos] = table[state];
        }
    }
    auto scale = [&]() { return float(1 + random() % 4093) / 3901.f; };
    std::vector<int8_t> activation(rows * input_pitch);
    std::vector<float> activation_scales(rows * scale_pitch), row_scales(columns), col_scales(groups);
    std::vector<uint16_t> group_scales(columns * groups);
    for (auto& value : activation) value = int8_t(int(random() % 256) - 128);
    std::generate(activation_scales.begin(), activation_scales.end(), scale);
    std::generate(row_scales.begin(), row_scales.end(), scale);
    std::generate(col_scales.begin(), col_scales.end(), scale);
    for (auto& value : group_scales) value = rad_f32_to_bf16(scale());
    RadTensor weight_planes[]{matrix(codes.data(), dtype, columns, reduction), matrix(table.data(), RAD_I8, 65536, 1),
                              matrix(coefficients, RAD_U32, 2, 1)};
    RadTensor scale_planes[]{grouped ? matrix(group_scales.data(), RAD_BF16, columns, groups)
                                     : matrix(row_scales.data(), RAD_F32, columns, 1),
                             matrix(col_scales.data(), RAD_F32, 1, groups)};
    const int weight_ids[]{code_id, table_id, mad_id}, scale_ids[]{scale_id, col_id};
    RadLayout layout[2]{};
    auto pack = [&](int operand, const int* ids, const RadTensor* planes, int count) {
        auto& desc = layout[operand - 2];
        check(row.layout(params, 5, operand, &encoding, ids, planes, count, &desc), "layout");
        std::vector<uint8_t> result(desc.bytes);
        check(row.relayout(params, 5, operand, &encoding, ids, planes, count, result.data(), desc.bytes), "relayout");
        return result;
    };
    auto weight = pack(2, weight_ids, weight_planes, 3), scales = pack(3, scale_ids, scale_planes, grouped ? 1 : 2);
    constexpr int guard = 32;
    std::vector<uint16_t> output(2 * guard + rows * out_pitch, 0xdead), expected = output;
    for (int token = 0; token < rows; ++token) for (int col = 0; col < columns; ++col) {
        float sum = 0;
        for (int group = 0; group < groups; ++group) {
            int subtotal = 0;
            for (int pos = group * 128; pos < (group + 1) * 128; ++pos)
                subtotal += int(activation[token * input_pitch + pos]) * decoded[col * reduction + pos];
            float term = float(subtotal) * activation_scales[token * scale_pitch + group];
            sum += term * (grouped ? rad_bf16_to_f32(group_scales[col * groups + group]) : col_scales[group]);
        }
        expected[guard + token * out_pitch + col] = rad_f32_to_bf16(grouped ? sum : sum * row_scales[col]);
    }
    DeviceBuffer input(activation, stream), input_scale(activation_scales, stream), packed(weight, stream),
                 packed_scale(scales, stream), result(output, stream);
    const int scale_view_width = grouped ? groups : reduction;
    RadTensor tensors[7]{matrix(input.data, RAD_I8, rows, reduction, input_pitch),
                        matrix(input_scale.data, RAD_F32, rows, groups, scale_pitch),
                        matrix(packed.data, layout[0].dtype, columns, reduction),
                        matrix(packed_scale.data, layout[1].dtype, columns, scale_view_width),
                        matrix(static_cast<uint16_t*>(result.data) + guard, RAD_BF16, rows, columns, out_pitch)};
    if (rows > 1) {
        tensors[4].rank = 3; tensors[4].shape[1] = columns / 16; tensors[4].shape[2] = 16;
        tensors[4].stride[1] = 16; tensors[4].stride[2] = 1;
    }
    RadArgs args{}; args.t = tensors; args.n_t = 7; args.p = params; args.n_p = 5;
    check(row.launch(&args, stream), "launch");
    check(rad_memcpy_async(output.data(), result.data, output.size() * sizeof(uint16_t), stream), "download");
    check(rad_stream_sync(stream), "sync");
    const auto mismatch = std::mismatch(output.begin(), output.end(), expected.begin());
    if (mismatch.first != output.end()) {
        std::fprintf(stderr, "u%u run=%d scales=%s M=%d offset=%td: got %04x expected %04x\n", bits, run,
                     grouped ? "group" : "factored", rows, mismatch.first - output.begin(), *mismatch.first, *mismatch.second);
        throw std::runtime_error("output or guard mismatch");
    }
}
}

int main(int argc, char** argv) try {
    const bool catalogue = argc == 3 && !std::strcmp(argv[2], "--catalogue-only");
    if (argc != 2 && !catalogue) { std::fprintf(stderr, "usage: %s path/to/libr3d.so [--catalogue-only]\n", argv[0]); return 2; }
    Library library(argv[1]);
    const auto& row = library.row("gemm_nt_q_iqr");
    if (!row.layout || !row.relayout || !row.unrelayout || row.priority >= library.row("gemm_nt_q_i8a8").priority)
        throw std::runtime_error("invalid packed hooks or native int8 precedence");
    const RadParam params[]{RAD_STR("dtype", "i8a8h"), RAD_INT("M", 1), RAD_INT("N", 256), RAD_INT("K", 128)};
    const auto plane = matrix(nullptr, RAD_I8, 256, 128);
    const int selected = 0;
    RadLayout layout{};
    for (const auto& native : {rad_enc_plain(RAD_I8), rad_enc_affine(RAD_I8, RAD_BF16, 1, 128)})
        if (row.layout(params, 4, 2, &native, &selected, &plane, 1, &layout) != RAD_E_DTYPE)
            throw std::runtime_error("packed row claimed native int8 encoding");
    if (catalogue) { std::puts("passed dense IQR catalogue and native int8 exclusions"); return 0; }
    for (int device = 0; device < rad_dev_count(); ++device) {
        RadDeviceProps props{};
        check(rad_dev_props(device, &props), "device properties");
        if (props.is_host_backend || std::strcmp(props.arch, "gfx1100")) continue;
        check(rad_dev_set(device), "select visible gfx1100");
        Queue queue;
        for (unsigned bits : {3u, 4u, 5u}) for (int run : {128, 256})
            for (bool grouped : {false, true}) for (int rows : {1, 5})
                differential(row, queue.stream, bits, run, grouped, rows);
        std::printf("passed 24 dense IQR differential cases on %s\n", props.name);
        return 0;
    }
    std::fprintf(stderr, "skipped: no visible gfx1100 device\n");
    return 77;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what()); return 1;
}