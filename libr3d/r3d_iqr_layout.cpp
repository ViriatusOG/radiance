#include "r3d_iqr.h"
#include "rad_plugin.h"
#include "rad_encoding.h"

#include <cstring>
#include <limits>

namespace {

struct View {
    int64_t rows = 0;
    int64_t columns = 0;
    unsigned bits = 0;
    unsigned run = 0;
    bool group_scale = false;
    const RadTensor* codes = nullptr;
    const RadTensor* mad = nullptr;
    const RadTensor* table = nullptr;
    const RadTensor* scale = nullptr;
    const RadTensor* column = nullptr;
};

bool dimensions(const RadTensor* tensor, uint32_t dtype, int64_t rows, int64_t columns) {
    return tensor && tensor->dtype == dtype && tensor->rank == 2
        && tensor->shape[0] == rows && tensor->shape[1] == columns;
}

int inspect(const RadParam* params, int count, int operand, const RadEncoding* encoding,
            const int* selected, const RadTensor* planes, int plane_count, View* view) {
    if (!encoding || !selected || !planes || plane_count <= 0) return RAD_E_INVAL;
    if (operand != 2 && operand != 3) return RAD_E_DTYPE;
    if (encoding->n_planes != 4 && encoding->n_planes != 5) return RAD_E_DTYPE;
    const char* dtype = rad_param_gets(params, count, "dtype", "");
    if (std::strcmp(dtype, "i8a8h") || std::strncmp(encoding->transform, "fwht128", RAD_ENC_STR)
        || !rad_enc_is(encoding, "trellis")) return RAD_E_DTYPE;
    view->rows = rad_param_geti(params, count, "N", 0);
    view->columns = rad_param_geti(params, count, "K", 0);
    if (view->rows <= 0 || view->rows > std::numeric_limits<int>::max()
        || view->columns <= 0 || view->columns > std::numeric_limits<int>::max()
        || view->rows % 128 || view->columns % 128
        || rad_param_geti(params, count, "group", 128) != 128
        || rad_param_geti(params, count, "w_rb", 1) != 1) return RAD_E_SHAPE;
    const RadEncPlane* codes = rad_enc_plane(encoding, "codes");
    const RadEncPlane* trellis = rad_enc_plane(encoding, "trellis");
    const RadEncPlane* mad = rad_enc_plane(encoding, "trellis.mad");
    const RadEncPlane* scale = rad_enc_plane(encoding, "scale");
    const RadEncPlane* column = rad_enc_plane(encoding, "scale.col");
    view->group_scale = scale && scale->dtype == RAD_BF16 && !column;
    if (!codes || !trellis || !mad || !scale
        || encoding->n_planes != (view->group_scale ? 4 : 5))
        return RAD_E_DTYPE;
    view->bits = codes->dtype == RAD_U3 ? 3 : codes->dtype == RAD_U4 ? 4
               : codes->dtype == RAD_U5 ? 5 : 0;
    if (!view->bits || codes->block[0] != 1 || codes->block[1] != 1
        || codes->kind != RAD_PLANE_TILED || scale->kind != RAD_PLANE_TILED
        || trellis->kind != RAD_PLANE_TABLE || mad->kind != RAD_PLANE_TABLE
        || trellis->dtype != RAD_I8 || trellis->extent[0] != 65536 || trellis->extent[1] != 1
        || (trellis->block[0] != 128 && trellis->block[0] != 256) || trellis->block[1] != 1
        || mad->dtype != RAD_U32 || mad->extent[0] != 2 || mad->extent[1] != 1
        || scale->block[0] != 1
        || (view->group_scale ? scale->block[1] != 128
            : scale->dtype != RAD_F32 || scale->block[1] != 0 || !column
                || column->kind != RAD_PLANE_TILED || column->dtype != RAD_F32
                || column->block[0] != 0 || column->block[1] != 128))
        return RAD_E_DTYPE;
    view->run = static_cast<unsigned>(trellis->block[0]);
    if (view->rows % view->run) return RAD_E_SHAPE;
    for (int index = 0; index < plane_count; ++index) {
        if (selected[index] < 0 || selected[index] >= encoding->n_planes) return RAD_E_SHAPE;
        const char* role = encoding->plane[selected[index]].role;
        const RadTensor** destination = !std::strcmp(role, "codes") ? &view->codes
            : !std::strcmp(role, "trellis") ? &view->table
            : !std::strcmp(role, "trellis.mad") ? &view->mad
            : !std::strcmp(role, "scale") ? &view->scale
            : !std::strcmp(role, "scale.col") ? &view->column : nullptr;
        if (!destination || *destination) return RAD_E_SHAPE;
        *destination = &planes[index];
    }
    if (operand == 2) {
        if (plane_count != 3 || !dimensions(view->codes, codes->dtype, view->rows, view->columns)
            || !dimensions(view->table, RAD_I8, 65536, 1)
            || !dimensions(view->mad, RAD_U32, 2, 1)) return RAD_E_SHAPE;
    } else {
        if (view->group_scale) {
            if (plane_count != 1 || !dimensions(view->scale, RAD_BF16,
                view->rows, view->columns / 128)) return RAD_E_SHAPE;
        } else if (plane_count != 2 || !dimensions(view->scale, RAD_F32, view->rows, 1)
            || !dimensions(view->column, RAD_F32, 1, view->columns / 128)) return RAD_E_SHAPE;
    }
    return RAD_OK;
}

int64_t stored_bytes(const View& view, int operand) {
    if (operand == 3 && view.group_scale)
        return view.rows * (view.columns / 128) * sizeof(uint16_t);
    return operand == 2 ? sizeof(r3d_iqr::Header) + view.rows * (view.columns / 8 * view.bits)
                        : (view.rows + view.columns / 128) * sizeof(float);
}

}

extern "C" int r3d_shape_gemm_iqr(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    if (!output || operand < 0 || operand > 6) return RAD_E_INVAL;
    const int64_t rows = rad_param_geti(params, count, "M", 0);
    const int64_t columns = rad_param_geti(params, count, "N", 0);
    const int64_t reduction = rad_param_geti(params, count, "K", 0);
    if (rows < 1 || rows > 2048 || columns < 128 || columns % 128 || reduction < 128
        || reduction % 128 || rad_param_geti(params, count, "group", 128) != 128)
        return RAD_E_SHAPE;
    *output = RadOpdDesc{};
    output->idx_const = -1;
    if (operand >= 5) { output->flags = RAD_OPD_F_ABSENT; return RAD_OK; }
    output->rank = 2;
    output->dtype = operand == 0 || operand == 2 ? RAD_I8 : operand == 4 ? RAD_BF16 : RAD_F32;
    output->shape[0] = operand == 2 || operand == 3 ? columns : rows;
    output->shape[1] = operand == 4 ? columns : operand == 1 || operand == 3 ? reduction / 128 : reduction;
    if (operand == 1 || operand == 3) output->fill = RAD_FILL_SIGMOID;
    return RAD_OK;
}

extern "C" int r3d_rad_layout_iqr(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const RadTensor* planes,
    int plane_count, RadLayout* output) {
    if (!output) return RAD_E_INVAL;
    View view;
    const int status = inspect(params, count, operand, encoding, selected, planes, plane_count, &view);
    if (status != RAD_OK) return status;
    *output = RadLayout{};
    output->tag = operand == 2 ? "r3d.iqr.mad.raw.v1"
        : view.group_scale ? "r3d.iqr.rowgroup.bf16.v1" : "r3d.iqr.rowcol.v1";
    output->dtype = operand == 2 ? r3d_iqr::weight_dtype
        : view.group_scale ? r3d_iqr::group_scale_dtype : r3d_iqr::scale_dtype;
    output->rank = 2;
    output->shape[0] = view.rows;
    output->shape[1] = operand == 2 ? view.columns : view.columns / 128;
    output->align = 256;
    output->bytes = stored_bytes(view, operand);
    return RAD_OK;
}

extern "C" int r3d_rad_relayout_iqr(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const RadTensor* planes,
    int plane_count, void* destination, int64_t bytes) {
    View view;
    const int status = inspect(params, count, operand, encoding, selected, planes, plane_count, &view);
    if (status != RAD_OK) return status;
    if (!destination || bytes != stored_bytes(view, operand)) return RAD_E_SHAPE;
    auto* output = static_cast<uint8_t*>(destination);
    if (operand == 2) {
        if (!view.codes->data || !view.mad->data || !view.table->data) return RAD_E_SHAPE;
        const auto* coefficients = static_cast<const uint32_t*>(view.mad->data);
        const auto* table = static_cast<const int8_t*>(view.table->data);
        for (uint32_t state = 0; state < 65536; ++state)
            if (r3d_iqr::trellis_mad(state, coefficients[0], coefficients[1]) != table[state])
                return RAD_E_FORMAT;
        const r3d_iqr::Header header{r3d_iqr::weight_magic, view.bits, view.run, 16,
            coefficients[0], coefficients[1], static_cast<uint32_t>(view.rows),
            static_cast<uint32_t>(view.columns)};
        std::memcpy(output, &header, sizeof(header));
        std::memcpy(output + sizeof(header), view.codes->data, bytes - sizeof(header));
    } else if (view.group_scale) {
        if (!view.scale->data) return RAD_E_SHAPE;
        std::memcpy(output, view.scale->data, bytes);
    } else {
        if (!view.scale->data || !view.column->data) return RAD_E_SHAPE;
        std::memcpy(output, view.scale->data, view.rows * sizeof(float));
        std::memcpy(output + view.rows * sizeof(float), view.column->data,
                    view.columns / 128 * sizeof(float));
    }
    return RAD_OK;
}

extern "C" int r3d_rad_unrelayout_iqr(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const void* source,
    int64_t bytes, const RadTensor* planes, int plane_count) {
    View view;
    const int status = inspect(params, count, operand, encoding, selected, planes, plane_count, &view);
    if (status != RAD_OK) return status;
    if (!source || bytes != stored_bytes(view, operand)) return RAD_E_SHAPE;
    const auto* input = static_cast<const uint8_t*>(source);
    if (operand == 2) {
        r3d_iqr::Header header;
        std::memcpy(&header, input, sizeof(header));
        if (header.magic != r3d_iqr::weight_magic || header.bits != view.bits
            || header.run_rows != view.run || header.state_bits != 16
            || header.rows != view.rows || header.columns != view.columns
            || !view.codes->data || !view.mad->data || !view.table->data) return RAD_E_SHAPE;
        std::memcpy(view.codes->data, input + sizeof(header), bytes - sizeof(header));
        auto* coefficients = static_cast<uint32_t*>(view.mad->data);
        coefficients[0] = header.multiplier;
        coefficients[1] = header.increment;
        auto* table = static_cast<int8_t*>(view.table->data);
        for (uint32_t state = 0; state < 65536; ++state)
            table[state] = r3d_iqr::trellis_mad(state, header.multiplier, header.increment);
    } else if (view.group_scale) {
        if (!view.scale->data) return RAD_E_SHAPE;
        std::memcpy(view.scale->data, input, bytes);
    } else {
        if (!view.scale->data || !view.column->data) return RAD_E_SHAPE;
        std::memcpy(view.scale->data, input, view.rows * sizeof(float));
        std::memcpy(view.column->data, input + view.rows * sizeof(float), view.columns / 128 * sizeof(float));
    }
    return RAD_OK;
}
