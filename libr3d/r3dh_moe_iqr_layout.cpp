#include <initializer_list>
#include "r3dh_moe_iqr.h"
#include <cmath>
#include <cstring>

namespace r3dh_moe_iqr {
int config(const RadParam* params, int count, Config* output, bool allow_band) {
    if (!output || count < 0 || (count && !params)) return RAD_E_INVAL;
    if (std::strcmp(rad_param_gets(params, count, "dtype", ""), "i8a8h")
        || std::strcmp(rad_param_gets(params, count, "a_order", "token"), "token"))
        return RAD_E_UNSUPPORTED;
    const RadParam* token_axis = rad_param_find(params, count, "M");
    if (token_axis && token_axis->kind == RAD_P_RANGE && (!allow_band
        || token_axis->ival < 1 || token_axis->ival > token_axis->ihi)) return RAD_E_SHAPE;
    const int64_t tokens = rad_param_getdim(params, count, "M", 0);
    const int64_t width = rad_param_geti(params, count, "N", 0);
    const int64_t odd = rad_param_geti(params, count, "N_odd", width);
    if (tokens < 1 || tokens > 2048 || !((width == 1280 && odd == 1280)
        || (width == 512 && odd == 768) || (width == 768 && odd == 512))
        || rad_param_geti(params, count, "K", 0) != 2560
        || rad_param_geti(params, count, "K_odd", 2560) != 2560
        || rad_param_geti(params, count, "n_expert", 0) != 512
        || rad_param_geti(params, count, "top_k", 0) != 10
        || rad_param_geti(params, count, "group", 128) != 128
        || rad_param_geti(params, count, "a_group", 128) != 128
        || rad_param_geti(params, count, "parts", 2) != 2) return RAD_E_SHAPE;
    *output = Config{static_cast<int>(tokens), {static_cast<int>(width), static_cast<int>(odd)},
                     2560, 512, 10, static_cast<int>(width > odd ? width : odd)};
    return RAD_OK;
}
}

namespace {
using namespace r3dh_moe_iqr;
struct View {
    int rows;
    bool weight;
    const RadTensor *codes = nullptr, *table = nullptr, *coeff = nullptr;
    const RadTensor *row = nullptr, *column = nullptr;
};
bool tensor(const RadTensor* value, uint32_t type, int rows, int columns) {
    return value && value->dtype == type && value->rank == 2
        && value->shape[0] == rows && value->shape[1] == columns
        && value->stride[0] == columns && value->stride[1] == 1;
}
int inspect(const RadParam* params, int count, int operand, const RadEncoding* encoding,
            const int* selected, const RadTensor* planes, int plane_count, View* view) {
    Config geometry;
    const int status = config(params, count, &geometry, true);
    if (status != RAD_OK) return status;
    if (operand != 2 && operand != 3 && operand != 6 && operand != 7)
        return RAD_E_UNSUPPORTED;
    if (!encoding || !selected || !planes) return RAD_E_INVAL;
    view->rows = geometry.width[operand >= 6];
    view->weight = operand == 2 || operand == 6;
    if (encoding->n_planes != 5 || std::strcmp(encoding->scheme, "trellis")
        || std::strcmp(encoding->transform, "fwht128")) return RAD_E_DTYPE;
    const RadEncPlane* codes = rad_enc_plane(encoding, "codes");
    const RadEncPlane* table = rad_enc_plane(encoding, "trellis");
    const RadEncPlane* coeff = rad_enc_plane(encoding, "trellis.mad");
    const RadEncPlane* row = rad_enc_plane(encoding, "scale");
    const RadEncPlane* column = rad_enc_plane(encoding, "scale.col");
    if (!codes || !table || !coeff || !row || !column
        || codes->dtype != RAD_U4 || codes->kind != RAD_PLANE_TILED
        || codes->block[0] != 1 || codes->block[1] != 1
        || table->dtype != RAD_I8 || table->kind != RAD_PLANE_TABLE
        || table->extent[0] != 65536 || table->extent[1] != 1
        || table->block[0] != 128 || table->block[1] != 1
        || coeff->dtype != RAD_U32 || coeff->kind != RAD_PLANE_TABLE
        || coeff->extent[0] != 2 || coeff->extent[1] != 1
        || row->dtype != RAD_F32 || row->kind != RAD_PLANE_TILED
        || row->block[0] != 1 || row->block[1] != 0
        || column->dtype != RAD_F32 || column->kind != RAD_PLANE_TILED
        || column->block[0] != 0 || column->block[1] != 128) return RAD_E_DTYPE;
    if (plane_count != (view->weight ? 3 : 2)) return RAD_E_SHAPE;
    for (int index = 0; index < plane_count; ++index) {
        if (selected[index] < 0 || selected[index] >= 5) return RAD_E_SHAPE;
        const char* role = encoding->plane[selected[index]].role;
        const RadTensor** destination = !std::strcmp(role, "codes") ? &view->codes
            : !std::strcmp(role, "trellis") ? &view->table
            : !std::strcmp(role, "trellis.mad") ? &view->coeff
            : !std::strcmp(role, "scale") ? &view->row
            : !std::strcmp(role, "scale.col") ? &view->column : nullptr;
        if (!destination || *destination) return RAD_E_SHAPE;
        *destination = planes + index;
    }
    return view->weight
        ? tensor(view->codes, RAD_U4, view->rows, 2560)
            && tensor(view->table, RAD_I8, 65536, 1)
            && tensor(view->coeff, RAD_U32, 2, 1) ? RAD_OK : RAD_E_SHAPE
        : tensor(view->row, RAD_F32, view->rows, 1)
            && tensor(view->column, RAD_F32, 1, 20) ? RAD_OK : RAD_E_SHAPE;
}
int64_t size(const View& view) {
    return view.weight ? sizeof(Header) + static_cast<int64_t>(view.rows) * 1280
                       : static_cast<int64_t>(view.rows + 20) * sizeof(float);
}
}

extern "C" int r3dh_moe_iqr_shape(const RadParam* params, int count, int operand,
                                 RadOpdDesc* output) {
    if (!output || operand < 0 || operand > 9) return RAD_E_INVAL;
    Config geometry;
    const int status = config(params, count, &geometry, true);
    if (status != RAD_OK) return status;
    *output = RadOpdDesc{};
    output->idx_const = -1;
    if (operand == 9) {
        output->flags = RAD_OPD_F_ABSENT;
        return RAD_OK;
    }
    output->rank = 2;
    if (operand == 2 || operand == 3 || operand == 6 || operand == 7) {
        output->dtype = operand == 2 || operand == 6 ? RAD_I8 : RAD_F32;
        output->rank = 3;
        output->shape[0] = geometry.experts / 2;
        output->shape[1] = geometry.width[operand >= 6];
        output->shape[2] = operand == 2 || operand == 6 ? 2560 : 20;
    } else if (operand == 4 || operand == 5) {
        output->dtype = RAD_I32;
        output->rank = 1;
        output->shape[0] = operand == 4 ? geometry.tokens * 10 : geometry.experts + 1;
        output->fill = RAD_FILL_INDEX;
        output->idx_max = geometry.tokens * geometry.top_k + (operand == 5);
        output->flags = operand == 4 ? RAD_OPD_F_IDX_UNIQUE : RAD_OPD_F_IDX_PREFIX;
    } else {
        output->dtype = operand == 0 ? RAD_I8 : operand == 1 ? RAD_F32 : RAD_BF16;
        output->shape[0] = operand == 8 ? geometry.tokens * 10 : geometry.tokens;
        output->shape[1] = operand == 0 ? 2560 : operand == 1 ? 20 : geometry.output_width;
    }
    return RAD_OK;
}

extern "C" int r3dh_moe_iqr_layout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const RadTensor* planes,
    int plane_count, RadLayout* output) {
    if (!output) return RAD_E_INVAL;
    View view{};
    const int status = inspect(params, count, operand, encoding, selected, planes, plane_count, &view);
    if (status != RAD_OK) return status;
    *output = RadLayout{};
    output->tag = view.weight ? weight_tag : scale_tag;
    output->dtype = view.weight ? weight_dtype : scale_dtype;
    output->rank = 2;
    output->shape[0] = view.rows;
    output->shape[1] = view.weight ? 2560 : 20;
    output->align = 256;
    output->bytes = size(view);
    return RAD_OK;
}

extern "C" int r3dh_moe_iqr_relayout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const RadTensor* planes,
    int plane_count, void* destination, int64_t bytes) {
    View view{};
    const int status = inspect(params, count, operand, encoding, selected, planes, plane_count, &view);
    if (status != RAD_OK) return status;
    if (!destination || bytes != size(view)) return RAD_E_SHAPE;
    auto* output = static_cast<uint8_t*>(destination);
    if (view.weight) {
        if (!view.codes->data || !view.table->data || !view.coeff->data) return RAD_E_INVAL;
        uint32_t coefficients[2];
        std::memcpy(coefficients, view.coeff->data, sizeof(coefficients));
        const auto* table = static_cast<const int8_t*>(view.table->data);
        for (uint32_t index = 0; index < 65536; ++index)
            if (table[index] != mad(index, coefficients[0], coefficients[1])) return RAD_E_FORMAT;
        const Header header{magic, 1, static_cast<uint32_t>(view.rows), 2560,
                            coefficients[0], coefficients[1], 128, 4};
        std::memcpy(output, &header, sizeof(header));
        std::memcpy(output + sizeof(header), view.codes->data, bytes - sizeof(header));
    } else {
        if (!view.row->data || !view.column->data) return RAD_E_INVAL;
        for (const RadTensor* plane : {view.row, view.column}) {
            const auto* values = static_cast<const float*>(plane->data);
            const int elements = plane == view.row ? view.rows : 20;
            for (int index = 0; index < elements; ++index)
                if (!std::isfinite(values[index])) return RAD_E_FORMAT;
        }
        std::memcpy(output, view.row->data, view.rows * sizeof(float));
        std::memcpy(output + view.rows * sizeof(float), view.column->data, 20 * sizeof(float));
    }
    return RAD_OK;
}

extern "C" int r3dh_moe_iqr_unrelayout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const void* source,
    int64_t bytes, const RadTensor* planes, int plane_count) {
    View view{};
    const int status = inspect(params, count, operand, encoding, selected, planes, plane_count, &view);
    if (status != RAD_OK) return status;
    if (!source || bytes != size(view)) return RAD_E_SHAPE;
    const auto* input = static_cast<const uint8_t*>(source);
    if (view.weight) {
        Header header;
        std::memcpy(&header, input, sizeof(header));
        if (header.signature != magic || header.version != 1 || header.rows != static_cast<uint32_t>(view.rows)
            || header.columns != 2560 || header.run != 128 || header.bits != 4) return RAD_E_FORMAT;
        if (!view.codes->data || !view.table->data || !view.coeff->data) return RAD_E_INVAL;
        std::memcpy(view.codes->data, input + sizeof(header), bytes - sizeof(header));
        const uint32_t coefficients[2] = {header.multiplier, header.increment};
        std::memcpy(view.coeff->data, coefficients, sizeof(coefficients));
        auto* table = static_cast<int8_t*>(view.table->data);
        for (uint32_t index = 0; index < 65536; ++index)
            table[index] = mad(index, header.multiplier, header.increment);
    } else {
        if (!view.row->data || !view.column->data) return RAD_E_INVAL;
        std::memcpy(view.row->data, input, view.rows * sizeof(float));
        std::memcpy(view.column->data, input + view.rows * sizeof(float), 20 * sizeof(float));
    }
    return RAD_OK;
}
