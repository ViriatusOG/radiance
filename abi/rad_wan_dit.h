/* Wan TI2V's latent patch layout and three-axis rotary dimensions. */
#pragma once

#include "rad_types.h"

namespace rad_wan_dit {

constexpr int head_width = 128;
constexpr int pairs = head_width / 2;
constexpr int spatial_pairs = pairs / 3;
constexpr int time_pairs = pairs - 2 * spatial_pairs;
constexpr int max_axis = 1024;

#if defined(__HIPCC__)
__host__ __device__
#endif
inline int pair_axis(int pair) {
    return pair < time_pairs ? 0 : pair < time_pairs + spatial_pairs ? 1 : 2;
}

#if defined(__HIPCC__)
__host__ __device__
#endif
inline int pair_in_axis(int pair, int axis) {
    return axis == 0 ? pair : pair - time_pairs - (axis - 1) * spatial_pairs;
}

#if defined(__HIPCC__)
__host__ __device__
#endif
inline bool grid_rows(int64_t t, int64_t h, int64_t w, int64_t& rows) {
    if (t < 1 || t > max_axis || h < 1 || h > max_axis || w < 1 || w > max_axis)
        return false;
    rows = t * h * w;
    return true;
}

#if defined(__HIPCC__)
__host__ __device__
#endif
inline bool latent_grid(const int32_t* dims, int64_t& height, int64_t& width, int64_t& rows) {
    if (dims[1] % 2 || dims[2] % 2) return false;
    height = dims[1] / 2;
    width = dims[2] / 2;
    return grid_rows(dims[0], height, width, rows);
}

#if defined(__HIPCC__)
__host__ __device__
#endif
inline int64_t coordinate(int64_t row, int axis, int64_t h, int64_t w) {
    return axis == 0 ? row / (h * w) : axis == 1 ? (row / w) % h : row % w;
}

#if defined(__HIPCC__)
__host__ __device__
#endif
inline int64_t unpack_offset(int64_t row, int64_t channel, int64_t height, int64_t width,
                             int64_t channels, int64_t pitch) {
    const int64_t x = row % (2 * width);
    const int64_t y = (row / (2 * width)) % (2 * height);
    const int64_t t = row / (4 * height * width);
    const int64_t patch = (t * height + y / 2) * width + x / 2;
    return patch * pitch + ((y % 2) * 2 + x % 2) * channels + channel;
}

inline int check_matrix(const RadTensor* t, uint32_t dtype, int64_t rows, int64_t width) {
    if (t->dtype != dtype) return RAD_E_DTYPE;
    if (t->rank != 2 || t->shape[0] != rows || t->shape[1] != width || rows < 1 || width < 1)
        return RAD_E_SHAPE;
    if (t->stride[1] != 1 || t->stride[0] < width ||
        t->stride[0] > (INT64_MAX / 4 - width) / rows) return RAD_E_STRIDE;
    return RAD_OK;
}

inline int check_grid(const RadTensor* grid) {
    if (grid->dtype != RAD_I32) return RAD_E_DTYPE;
    if (grid->rank != 1 || grid->shape[0] != 3) return RAD_E_SHAPE;
    return grid->stride[0] == 1 ? RAD_OK : RAD_E_STRIDE;
}

// A layer has six learned modulation rows. The head uses two, both
// conditioned by the same time embedding rather than six projected rows.
inline int check_modulation(const RadTensor* time, const RadTensor* table,
                            const RadTensor* split, int64_t width, bool head) {
    const int64_t steps = time->shape[0];
    if (steps != (split ? 2 : 1)) return RAD_E_SHAPE;
    const int rc = check_matrix(time, RAD_F32, steps, head ? width : 6 * width);
    if (rc < 0) return rc;
    if (table->dtype != RAD_F32 || (split && split->dtype != RAD_I32)) return RAD_E_DTYPE;
    if (table->rank != 3 || table->shape[0] != 1 || table->shape[1] != (head ? 2 : 6) ||
        table->shape[2] != width || (split && (split->rank != 1 || split->shape[0] != 1)))
        return RAD_E_SHAPE;
    if (table->stride[1] != width || table->stride[2] != 1)
        return RAD_E_STRIDE;
    return RAD_OK;
}

}  // namespace rad_wan_dit
