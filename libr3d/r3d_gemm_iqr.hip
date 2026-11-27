#include <hip/hip_runtime.h>
#include "r3d_iqr.h"
#include "rad_plugin.h"

#include <climits>
#include <cstring>

namespace {

template<bool GroupScale>
__global__ void iqr_scalar(const int8_t* activation, const float* activation_scale,
    const r3d_iqr::Header* header, const void* scales, uint16_t* output,
    int columns, int reduction, int64_t activation_pitch, int64_t scale_pitch, int64_t output_pitch) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (column >= columns || header->magic != r3d_iqr::weight_magic || header->state_bits != 16
        || header->bits < 3 || header->bits > 5 || (header->run_rows != 128 && header->run_rows != 256)
        || header->rows != unsigned(columns) || header->columns != unsigned(reduction)
        || columns % header->run_rows) return;
    const auto* codes = reinterpret_cast<const uint8_t*>(header + 1);
    const uint64_t pitch = uint64_t(reduction) * header->bits / 8;
    const int groups = reduction / 128;
    float accumulator = 0;
    for (int group = 0; group < groups; ++group) {
        int32_t subtotal = 0;
        for (int offset = 0; offset < 128; ++offset) {
            const int position = group * 128 + offset;
            const uint32_t state = r3d_iqr::trellis_state(codes, pitch, column, position,
                                                           header->bits, 16, header->run_rows);
            subtotal += int(activation[row * activation_pitch + position])
                      * int(r3d_iqr::trellis_mad(state, header->multiplier, header->increment));
        }
        float weight_scale;
        if constexpr (GroupScale)
            weight_scale = __uint_as_float(uint32_t(static_cast<const uint16_t*>(scales)[
                int64_t(column) * groups + group]) << 16);
        else
            weight_scale = static_cast<const float*>(scales)[int64_t(columns) + group];
        const float scaled = __fmul_rn(float(subtotal), activation_scale[row * scale_pitch + group]);
        accumulator = __fadd_rn(accumulator, __fmul_rn(scaled, weight_scale));
    }
    if constexpr (!GroupScale) accumulator = __fmul_rn(accumulator, static_cast<const float*>(scales)[column]);
    const uint32_t word = __float_as_uint(accumulator);
    output[row * output_pitch + column] = (word & 0x7fffffffu) > 0x7f800000u
        ? uint16_t((word >> 16) | 0x40u) : uint16_t((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}

bool matrix(const RadTensor& tensor, int64_t rows, int64_t width) {
    return tensor.rank == 2 && tensor.shape[0] == rows && tensor.shape[1] == width
        && tensor.stride[1] == 1 && tensor.stride[0] >= width;
}

bool addressable(const RadTensor& tensor, int64_t rows, int64_t width, int element_bytes) {
    return tensor.stride[0] >= width && (rows == 1 || tensor.stride[0] <=
        (INT64_MAX / element_bytes - width) / (rows - 1));
}

}

extern "C" int r3d_rad_gemm_iqr(const RadArgs* args, RadStream stream) {
    if (!args || !args->t || args->n_t < 5 || args->n_t > 7) return RAD_E_INVAL;
    const RadTensor* tensor = args->t;
    for (int index = 5; index < args->n_t; ++index)
        if (tensor[index].data || tensor[index].rank) return RAD_E_UNSUPPORTED;
    for (int index = 0; index < 5; ++index) if (!tensor[index].data) return RAD_E_INVAL;
    const bool grouped = tensor[3].dtype == r3d_iqr::group_scale_dtype;
    if (std::strcmp(rad_args_gets_or(args, "dtype", ""), "i8a8h")
        || tensor[0].dtype != RAD_I8 || tensor[1].dtype != RAD_F32
        || tensor[2].dtype != r3d_iqr::weight_dtype || (!grouped && tensor[3].dtype != r3d_iqr::scale_dtype)
        || tensor[4].dtype != RAD_BF16) return RAD_E_DTYPE;
    const int64_t rows = tensor[0].shape[0], band = rad_args_geti_or(args, "M", 0);
    const int64_t columns = rad_args_geti_or(args, "N", 0), reduction = rad_args_geti_or(args, "K", 0);
    if (rows < 1 || rows > band || band > 2048 || columns < 128 || columns > INT_MAX || columns % 128
        || reduction < 128 || reduction > INT_MAX || reduction % 128
        || rad_args_geti_or(args, "group", 128) != 128 || rad_args_geti_or(args, "w_rb", 1) != 1)
        return RAD_E_SHAPE;
    if (!matrix(tensor[0], rows, reduction) || !matrix(tensor[1], rows, reduction / 128)
        || !matrix(tensor[2], columns, reduction) || tensor[3].rank != 2 || tensor[3].stride[1] != 1
        || tensor[3].shape[0] != columns || (tensor[3].shape[1] != reduction / 128 && tensor[3].shape[1] != reduction))
        return RAD_E_SHAPE;
    const RadTensor& out = tensor[4];
    if (out.rank < 2 || out.rank > sizeof(out.shape) / sizeof(out.shape[0]) || out.shape[0] != rows)
        return RAD_E_SHAPE;
    int64_t width = 1;
    for (int axis = out.rank - 1; axis > 0; --axis) {
        if (out.shape[axis] < 1 || out.shape[axis] > columns / width) return RAD_E_SHAPE;
        if (out.stride[axis] != width) return RAD_E_STRIDE;
        width *= out.shape[axis];
    }
    if (width != columns) return RAD_E_SHAPE;
    if (!addressable(tensor[0], rows, reduction, 1) || !addressable(tensor[1], rows, reduction / 128, 4)
        || !addressable(out, rows, width, 2)) return RAD_E_STRIDE;
    auto launch = [&](auto kernel) {
        kernel<<<dim3((columns + 127) / 128, rows), 128, 0, reinterpret_cast<hipStream_t>(stream)>>>(
            static_cast<const int8_t*>(tensor[0].data), static_cast<const float*>(tensor[1].data),
            static_cast<const r3d_iqr::Header*>(tensor[2].data), tensor[3].data,
            static_cast<uint16_t*>(out.data), int(columns), int(reduction), tensor[0].stride[0],
            tensor[1].stride[0], out.stride[0]);
    };
    if (grouped) launch(iqr_scalar<true>); else launch(iqr_scalar<false>);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}