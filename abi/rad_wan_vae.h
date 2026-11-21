/* Wan's chunked causal convolution and decoder resampling layout. */
#pragma once

#include "rad_wan_dit.h"

namespace rad_wan_vae {

constexpr int64_t max_spatial_axis = 8192;
constexpr int64_t max_chunk_frames = 8;

#if defined(__HIPCC__)
__host__ __device__
#endif
inline bool frame_plane(const int32_t* geom, int64_t offsets, int64_t& plane) {
    if (geom[0] + 1LL != offsets || geom[0] < 1 || geom[0] > max_chunk_frames ||
        geom[1] < 1 || geom[1] > max_spatial_axis || geom[2] < 1 || geom[2] > max_spatial_axis)
        return false;
    plane = (int64_t)geom[1] * geom[2];
    return true;
}

inline int check_history(const RadTensor* x, const RadTensor* history, const RadTensor* out,
                         bool temporal, int64_t& plane, int64_t& frames) {
    const int64_t rows = x->shape[0], channels = x->shape[1], cached = history->shape[0];
    if (rows < 1 || rows > INT32_MAX || channels < 1 || channels > 4096 ||
        cached < 2 || cached > INT32_MAX || (cached & 1)) return RAD_E_SHAPE;
    plane = cached / 2;
    if (rows % plane) return RAD_E_SHAPE;
    frames = rows / plane;
    int rc = rad_wan_dit::check_matrix(x, RAD_BF16, rows, channels);
    if (rc < 0) return rc;
    rc = rad_wan_dit::check_matrix(history, RAD_BF16, cached, channels);
    if (rc < 0) return rc;
    return rad_wan_dit::check_matrix(out, RAD_BF16, temporal ? rows : rows + cached,
                                     temporal ? 3 * channels : channels);
}

template<bool Temporal>
#if defined(__HIPCC__)
__host__ __device__
#endif
inline void history_cell(const uint16_t* x, uint16_t* history, uint16_t* out,
                          int64_t pixel, int64_t c, int64_t plane, int64_t frames,
                          int64_t x_ld, int64_t h_ld, int64_t y_ld, bool reset) {
    uint16_t older = reset ? 0 : history[pixel * h_ld + c];
    uint16_t newer = reset ? 0 : history[(plane + pixel) * h_ld + c];
    if constexpr (!Temporal) {
        out[pixel * y_ld + c] = older;
        out[(plane + pixel) * y_ld + c] = newer;
    }
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t row = frame * plane + pixel;
        const uint16_t value = x[row * x_ld + c];
        if constexpr (Temporal) {
            uint16_t* tap = out + row * y_ld + c * 3;
            tap[0] = older;
            tap[1] = newer;
            tap[2] = value;
        } else {
            out[(row + 2 * plane) * y_ld + c] = value;
        }
        older = newer;
        newer = value;
    }
    history[pixel * h_ld + c] = older;
    history[(plane + pixel) * h_ld + c] = newer;
}

inline int check_expand(const RadTensor* x, const RadTensor* out, int64_t cin, int64_t cout,
                        int64_t st, int64_t s, int64_t drop, bool temporal) {
    const int64_t rows = x->shape[0], expanded = out->shape[0];
    if (cin < 1 || cin > 4096 || cout < 1 || cout > 4096 || st < 1 || st > 2 || s < 1 || s > 2 ||
        drop < 0 || drop >= st || rows > INT32_MAX || expanded > INT32_MAX)
        return RAD_E_SHAPE;
    if (temporal ? (st != 2 || s != 1 || cin != 2 * cout || drop != 0)
                 : ((cout * st * s * s) % cin != 0)) return RAD_E_SHAPE;
    const int rc = rad_wan_dit::check_matrix(x, RAD_BF16, rows, cin);
    if (rc < 0) return rc;
    return rad_wan_dit::check_matrix(out, RAD_BF16, expanded, cout);
}

#if defined(__HIPCC__)
__host__ __device__
#endif
inline bool check_expand_grid(const int32_t* geom, int64_t rows, int64_t expanded,
                              int64_t st, int64_t s, int64_t drop) {
    const int64_t t = geom[0], h = geom[1], w = geom[2];
    if (t < 1 || t > 1024 || h < 1 || h > max_spatial_axis || w < 1 || w > max_spatial_axis)
        return false;
    return rows == t * h * w && expanded == (t * st - drop) * h * w * s * s;
}

#if defined(__HIPCC__)
__host__ __device__
#endif
inline int64_t expand_offset(int64_t row, int64_t c, int64_t h, int64_t w, int64_t cin,
                             int64_t cout, int64_t st, int64_t s, int64_t drop, bool temporal,
                             int64_t pitch) {
    const int64_t out_w = w * s, out_h = h * s;
    const int64_t x = row % out_w, y = (row / out_w) % out_h;
    const int64_t t = row / (out_w * out_h) + drop;
    const int64_t src_row = ((t / st) * h + y / s) * w + x / s;
    if (temporal) return src_row * pitch + (t % st) * cout + c;
    const int64_t repeated = ((c * st + t % st) * s + y % s) * s + x % s;
    return src_row * pitch + repeated / (cout * st * s * s / cin);
}

}  // namespace rad_wan_vae
