/* lq_ggml_dequant.h -- llama.cpp's dequantize_row_<type> for the types a kernel reads natively,
 * over the native blocks lq_ggml_planes' pack_of writes back.
 *
 * The reference a kernel's dot product is checked against: these are the values llama.cpp itself
 * dequantises to, and the core's enc_decode_rows of the same blocks' planes gives the same f32
 * values bit for bit (lq_ggml_planes.cpp says why; checkpoint_test pins it). Shared like
 * lq_ggml_planes: the core and libref compile it from libquant's sources.
 */
#pragma once
#include <cstdint>

namespace lq::ggml {

/* `k` values (a whole number of blocks) of one row's native blocks -> f32 at y. */
typedef void (*DequantFn)(const void* blocks, float* y, int64_t k);

/* Null for a type without a native reader (the types with one: q8_0, q5_K, q6_K, iq4_nl, iq2_xxs,
 * iq3_xxs, iq1_s). */
DequantFn dequant_of(uint32_t type);

/* IEEE half -> float, exactly (subnormals, infinities and NaN included). */
float f16_to_f32(uint16_t h);

}  /* namespace lq::ggml */
