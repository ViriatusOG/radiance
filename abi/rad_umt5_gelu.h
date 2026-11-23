/* The BF16 tanh GELU used by Wan's UMT5 feed-forward gate. */
#pragma once

#include <cmath>

namespace rad_umt5 {

template<class Round>
#if defined(__HIPCC__)
__device__
#endif
inline float gelu(float x, Round round) {
    // PyTorch's BF16 cube uses two BF16 products. The remaining eager
    // operations also materialize BF16 before the next operation consumes them.
    const float cube = round(round(x * x) * x);
    const float inner = round(x + round(0.044715f * cube));
    const float activation = round(::tanhf(round(0.7978845608028654f * inner)));
    return round(round(0.5f * x) * round(1.0f + activation));
}

}  // namespace rad_umt5
