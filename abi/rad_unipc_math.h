/* Shared scalar arithmetic for the flow_unipc_step host and device ops. */
#pragma once

namespace rad_unipc {

enum Coefficient {
    SIGMA, CORRECT_SAMPLE, CORRECT_CLEAN, CORRECT_PREVIOUS_DELTA, CORRECT_CURRENT_DELTA,
    PREDICT_SAMPLE, PREDICT_CLEAN, PREDICT_DELTA, COUNT,
};

#if defined(__HIPCC__)
__host__ __device__
#endif
inline float advance(const float* coefficients, float sample, float velocity,
                     float& last_sample, float& last_clean, float& older_clean) {
    const float clean = sample - coefficients[SIGMA] * velocity;
    if (coefficients[CORRECT_SAMPLE] > 0.0f) {
        sample = coefficients[CORRECT_SAMPLE] * last_sample +
                 coefficients[CORRECT_CLEAN] * last_clean +
                 coefficients[CORRECT_PREVIOUS_DELTA] * (older_clean - last_clean) +
                 coefficients[CORRECT_CURRENT_DELTA] * (clean - last_clean);
    }
    const float next = coefficients[PREDICT_SAMPLE] * sample +
                       coefficients[PREDICT_CLEAN] * clean +
                       coefficients[PREDICT_DELTA] * (last_clean - clean);
    last_sample = sample;
    older_clean = last_clean;
    last_clean = clean;
    return next;
}

}  // namespace rad_unipc
