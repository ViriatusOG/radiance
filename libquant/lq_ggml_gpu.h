/* lq_ggml_gpu.h -- GGML IQ2_XXS/IQ3_XXS on the card. See lq_ggml_gpu.hip.
 *
 * The host quantiser gets blocks back from the card, then unpacks them into planes exactly as the
 * host loop does. The entry points are built only where libquant has a device build; lq_ggml.cpp
 * guards the call with LQ_CARD.
 */
#pragma once
#include "lq_ggml_port.h"

#include <string>

namespace lq::ggml {

/* Whether a device build with at least one selected card is available. */
bool gpu_available();

/* Quantise `nrow` rows of `n_per_row` f32 values from `src` into GGUF blocks at `dst`, on the
 * card this thread has been assigned. `imp` is the per-column imatrix slice (IQ2_XXS needs it;
 * IQ3_XXS uses x*x when it is null). Returns RAD_OK, RAD_E_UNSUPPORTED when the caller must use
 * the host path, or another failure with the reason in *why. */
int quantize_gpu(enum ggml_type type, const float * src, void * dst, int64_t nrow,
                 int64_t n_per_row, const float * imp, std::string * why);

}  /* namespace lq::ggml */
