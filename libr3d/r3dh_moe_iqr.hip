#include <hip/hip_runtime.h>
#include "r3dh_moe_iqr.h"

namespace {
using namespace r3dh_moe_iqr;
using Int4 = int __attribute__((ext_vector_type(4)));
using Int8 = int __attribute__((ext_vector_type(8)));

struct Table {
    const uint8_t* base;
    uint64_t pitch;
    __device__ const uint8_t* entry(int index) const {
        return pitch ? base + index * pitch : reinterpret_cast<const uint8_t* const*>(base)[index];
    }
};
struct Call {
    Config geometry;
    Table weight[2], scale[2];
    const int8_t* activation;
    const float* activation_scale;
    const int32_t *sorted, *offsets;
    uint16_t* output;
    int64_t activation_pitch, scale_pitch, output_pitch;
};
__device__ uint16_t bf16(float value) {
    const uint32_t word = __float_as_uint(value);
    if ((word & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((word >> 16) | 64u);
    return static_cast<uint16_t>((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
__device__ int token(const Call& call, int row, int begin, int end) {
    if (row < begin || row >= end) return -1;
    const int slot = call.sorted[row];
    return slot >= 0 && slot < call.geometry.tokens * call.geometry.top_k ? slot / call.geometry.top_k : -1;
}
__global__ __launch_bounds__(32) void gate_up(Call call) {
    const unsigned lane = threadIdx.x;
    const int first = blockIdx.y * 16;
    const int total = call.geometry.tokens * call.geometry.top_k;
    const int limit = first + 16 < total ? first + 16 : total;
    const int column = blockIdx.x * 16 + (lane & 15u);
    for (int begin = first; begin < limit;) {
        int lower = 0, upper = call.geometry.experts + 1;
        while (lower < upper) {
            const int middle = (lower + upper) / 2;
            if (call.offsets[middle] <= begin) lower = middle + 1;
            else upper = middle;
        }
        const int expert = lower - 1;
        const bool routed = expert >= 0 && expert < call.geometry.experts;
        const int boundary = routed ? call.offsets[expert + 1] : total;
        const int end = boundary < limit && boundary > begin ? boundary : limit;
        const int parity = routed ? expert & 1 : 0;
        const int width = call.geometry.width[parity];
        const int part = call.geometry.output_width / 2;
        const int local = column % part;
        const int weight_row = (column / part) * (width / 2) + local;
        const bool live = column < call.geometry.output_width && local < width / 2;
        const uint8_t* blob = routed ? call.weight[parity].entry(expert / 2) : nullptr;
        const auto* header = reinterpret_cast<const Header*>(blob);
        const auto* scales = routed ? reinterpret_cast<const float*>(call.scale[parity].entry(expert / 2)) : nullptr;
        const int input_row = token(call, first + (lane & 15u), begin, end);
        float accumulator[8]{};
        for (int group = 0; routed && group < call.geometry.reduction / 128; ++group) {
            Int8 subtotal{};
            for (int step = 0; step < 128; step += 16) {
                Int4 activation{}, weight{};
                for (int word = 0; word < 4; ++word) {
                    uint32_t inputs = 0, weights = 0;
                    for (int byte = 0; byte < 4; ++byte) {
                        const int pos = group * 128 + step + word * 4 + byte;
                        const int8_t input = input_row >= 0 ? call.activation[input_row * call.activation_pitch + pos] : 0;
                        const int8_t decoded = live ? mad(state(blob + sizeof(Header), call.geometry.reduction,
                            weight_row, pos), header->multiplier, header->increment) : 0;
                        inputs |= uint32_t(uint8_t(input)) << (byte * 8);
                        weights |= uint32_t(uint8_t(decoded)) << (byte * 8);
                    }
                    activation[word] = static_cast<int>(inputs);
                    weight[word] = static_cast<int>(weights);
                }
                subtotal = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, activation, true, weight, subtotal, false);
            }
            for (int index = 0; index < 8; ++index) {
                const int row = token(call, first + index * 2 + (lane >> 4), begin, end);
                const float factor = row >= 0 ? call.activation_scale[row * call.scale_pitch + group] : 0.0f;
                accumulator[index] += (static_cast<float>(subtotal[index]) * factor) * scales[width + group];
            }
        }
        for (int index = 0; index < 8; ++index) {
            const int row = first + index * 2 + (lane >> 4);
            if (row >= begin && row < end && column < call.geometry.output_width) {
                const bool valid = routed && live && token(call, row, begin, end) >= 0;
                call.output[row * call.output_pitch + column] = valid ? bf16(accumulator[index] * scales[weight_row]) : 0;
            }
        }
        begin = end;
    }
}
bool matrix(const RadTensor& tensor, uint32_t dtype, int rows, int columns) {
    return tensor.data && tensor.dtype == dtype && tensor.rank == 2 && tensor.shape[0] == rows
        && tensor.shape[1] == columns && tensor.stride[0] >= columns && tensor.stride[0] <= (1ll << 40)
        && tensor.stride[1] == 1;
}
int table(const RadTensor& tensor, bool weight, int rows, int entries, Table& result) {
    if (!tensor.data || tensor.rank != 3 || tensor.shape[0] != entries || tensor.shape[1] != rows
        || (tensor.shape[2] != 2560 && (weight || tensor.shape[2] != 20))) return RAD_E_SHAPE;
    if (tensor.dtype != (weight ? weight_dtype : scale_dtype)) return RAD_E_DTYPE;
    const int64_t bytes = weight ? sizeof(Header) + rows * 1280 : (rows + 20) * sizeof(float);
    if (tensor.stride[0] < 0 || (tensor.stride[0] && tensor.stride[0] < bytes)
        || tensor.stride[1] != tensor.shape[2] || tensor.stride[2] != 1) return RAD_E_STRIDE;
    result = {static_cast<const uint8_t*>(tensor.data), static_cast<uint64_t>(tensor.stride[0])};
    return RAD_OK;
}
}

extern "C" int r3dh_moe_iqr_launch(const RadArgs* args, RadStream stream) {
    if (!args || (args->n_t != 9 && args->n_t != 10) || !args->t) return RAD_E_INVAL;
    Call call{};
    int status = config(args->p, args->n_p, &call.geometry);
    if (status != RAD_OK) return status;
    const auto* tensors = args->t;
    Config& geometry = call.geometry;
    const int64_t rows = tensors[8].shape[0];
    if (rows < geometry.top_k || rows % geometry.top_k || rows > geometry.tokens * geometry.top_k) return RAD_E_SHAPE;
    geometry.tokens = static_cast<int>(rows / geometry.top_k);
    if (!matrix(tensors[0], RAD_I8, geometry.tokens, 2560) || !matrix(tensors[1], RAD_F32, geometry.tokens, 20)
        || !matrix(tensors[8], RAD_BF16, rows, geometry.output_width)) return RAD_E_SHAPE;
    for (int operand = 4; operand <= 5; ++operand) {
        const RadTensor& tensor = tensors[operand];
        if (!tensor.data || tensor.dtype != RAD_I32 || tensor.rank != 1
            || tensor.shape[0] != (operand == 4 ? rows : geometry.experts + 1) || tensor.stride[0] != 1) return RAD_E_SHAPE;
    }
    for (int parity = 0; parity < 2; ++parity) {
        status = table(tensors[2 + parity * 4], true, geometry.width[parity], geometry.experts / 2, call.weight[parity]);
        if (status != RAD_OK) return status;
        status = table(tensors[3 + parity * 4], false, geometry.width[parity], geometry.experts / 2, call.scale[parity]);
        if (status != RAD_OK) return status;
    }
    call.activation = static_cast<const int8_t*>(tensors[0].data);
    call.activation_scale = static_cast<const float*>(tensors[1].data);
    call.sorted = static_cast<const int32_t*>(tensors[4].data);
    call.offsets = static_cast<const int32_t*>(tensors[5].data);
    call.output = static_cast<uint16_t*>(tensors[8].data);
    call.activation_pitch = tensors[0].stride[0];
    call.scale_pitch = tensors[1].stride[0];
    call.output_pitch = tensors[8].stride[0];
    hipLaunchKernelGGL(gate_up, dim3((geometry.output_width + 15) / 16, (rows + 15) / 16),
                       dim3(32), 0, reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}