/* Geometry and the relative-position buckets of Wan's UMT5 encoder. */
#pragma once

#include "rad_types.h"

namespace rad_umt5 {

constexpr int head_width = 64;
constexpr int max_tokens = 512;
constexpr int buckets = 32;

#if defined(__HIPCC__)
__host__ __device__
#endif
inline int bucket(int distance) {
    // Wan T5RelativeEmbedding's 32 bidirectional buckets, max_distance 128.
    // Integer thresholds are the first distance in each logarithmic bucket.
    const int sign = distance > 0 ? buckets / 2 : 0;
    const int64_t mag = distance >= 0 ? distance : -static_cast<int64_t>(distance);
    if (mag < 8) return sign + static_cast<int>(mag);
    return sign + 8 + (mag >= 12) + (mag >= 16) + (mag >= 23) + (mag >= 32) +
           (mag >= 46) + (mag >= 64) + (mag >= 91);
}

inline int validate(const RadTensor* const* matrices, const RadTensor* cu,
                    const RadTensor* bias, int64_t heads, int64_t segments, int64_t& rows) {
    if (heads < 1 || heads > 128 || segments < 1 || segments > INT32_MAX - 1)
        return RAD_E_SHAPE;
    if (cu->dtype != RAD_I32 || bias->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (cu->rank != 1 || cu->shape[0] != segments + 1 || bias->rank != 2 ||
        bias->shape[0] != buckets || bias->shape[1] != heads) return RAD_E_SHAPE;
    rows = matrices[0]->shape[0];
    const int64_t width = heads * head_width;
    if (rows < segments || rows > INT32_MAX || rows > segments * max_tokens) return RAD_E_SHAPE;
    if (cu->stride[0] != 1 || bias->stride[1] != 1 || bias->stride[0] < heads ||
        bias->stride[0] > (INT64_MAX - heads) / buckets) return RAD_E_STRIDE;
    for (int i = 0; i < 4; ++i) {
        const RadTensor* t = matrices[i];
        if (t->dtype != RAD_BF16) return RAD_E_DTYPE;
        if (t->rank != 2 || t->shape[0] != rows || t->shape[1] != width) return RAD_E_SHAPE;
        if (t->stride[1] != 1 || t->stride[0] < width ||
            t->stride[0] > (INT64_MAX - width) / rows) return RAD_E_STRIDE;
    }
    return RAD_OK;
}

}  // namespace rad_umt5
