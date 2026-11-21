/* The flow-prediction UniPC BH2 sampler of Wan 2.2 TI2V-5B. Coefficients
 * depend only on the sigma grid and can be shared by all latent elements. */
#pragma once

#include "rad_unipc_math.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace wan {

struct UniPCStep {
    float sigma;
    int32_t timestep;
    // Correct the previous prediction from its saved sample and clean estimates.
    float correct_sample;
    float correct_clean;
    float correct_previous_delta;
    float correct_current_delta;
    // Advance from the corrected sample to the next sigma.
    float predict_sample;
    float predict_clean;
    float predict_delta;

    void coefficients(float* values) const {
        values[rad_unipc::SIGMA] = sigma;
        values[rad_unipc::CORRECT_SAMPLE] = correct_sample;
        values[rad_unipc::CORRECT_CLEAN] = correct_clean;
        values[rad_unipc::CORRECT_PREVIOUS_DELTA] = correct_previous_delta;
        values[rad_unipc::CORRECT_CURRENT_DELTA] = correct_current_delta;
        values[rad_unipc::PREDICT_SAMPLE] = predict_sample;
        values[rad_unipc::PREDICT_CLEAN] = predict_clean;
        values[rad_unipc::PREDICT_DELTA] = predict_delta;
    }
};

inline float flow_log_snr(float sigma) {
    return std::log(1.0f - sigma) - std::log(sigma);
}

inline std::vector<UniPCStep> unipc_plan(int32_t steps, double shift) {
    constexpr int32_t training_steps = 1000;
    if (steps < 1 || steps > training_steps || !std::isfinite(shift) ||
        shift <= 0.0 || shift > 100.0) return {};

    std::vector<UniPCStep> plan(static_cast<std::size_t>(steps));
    std::vector<float> log_snr(plan.size());
    // Wan builds the training grid in float32, then computes the request
    // grid and shift in float64. The model timestep truncates before sigma
    // is rounded back to float32 for sampling. Execute by step index starting
    // at zero, equivalent to set_begin_index(0) in the reference. Integer
    // timesteps can repeat even when the underlying sigma grid is distinct.
    const double first = static_cast<float>(1.0 - 1.0 / training_steps);
    const double stride = -first / steps;
    for (int32_t i = 0; i < steps; ++i) {
        const double base = i * stride + first;
        const double sigma = shift * base / (1.0 + (shift - 1.0) * base);
        plan[i].sigma = static_cast<float>(sigma);
        plan[i].timestep = static_cast<int32_t>(sigma * training_steps);
        if (plan[i].sigma <= 0.0f || plan[i].sigma >= 1.0f ||
            (i > 0 && plan[i].sigma >= plan[i - 1].sigma)) return {};
        log_snr[i] = flow_log_snr(plan[i].sigma);
        if (!std::isfinite(log_snr[i]) || (i > 0 && log_snr[i] <= log_snr[i - 1]))
            return {};
    }

    for (int32_t i = 0; i < steps; ++i) {
        UniPCStep& step = plan[i];
        const float sigma = step.sigma;
        const float lambda = log_snr[i];
        if (i > 0) {
            const float prior = plan[i - 1].sigma;
            const float prior_lambda = log_snr[i - 1];
            const float hh = prior_lambda - lambda;
            const float bh = std::expm1(hh);
            step.correct_sample = sigma / prior;
            step.correct_clean = -(1.0f - sigma) * bh;
            if (i == 1) {
                step.correct_current_delta = step.correct_clean * 0.5f;
            } else {
                // The order-two corrector solves [1 1; rk 1] * rho = b.
                const float rk = (log_snr[i - 2] - prior_lambda) / -hh;
                const float phi2 = bh / hh - 1.0f;
                const float b0 = phi2 / bh;
                const float b1 = (phi2 / hh - 0.5f) * 2.0f / bh;
                const float rho_prior = (b0 - b1) / (1.0f - rk);
                step.correct_previous_delta = step.correct_clean * rho_prior / rk;
                step.correct_current_delta = step.correct_clean * (b0 - rho_prior);
            }
        }

        if (i == steps - 1) {
            // At terminal sigma zero Wan drops to first order and returns x0.
            step.predict_clean = 1.0f;
        } else {
            const float next = plan[i + 1].sigma;
            const float hh = lambda - log_snr[i + 1];
            step.predict_sample = next / sigma;
            step.predict_clean = -(1.0f - next) * std::expm1(hh);
            if (i > 0) {
                const float rk = (log_snr[i - 1] - lambda) / -hh;
                step.predict_delta = step.predict_clean * 0.5f / rk;
            }
        }
    }
    return plan;
}

struct UniPCState {
    float last_sample = 0;    // The sample after correction, before prediction.
    float last_clean = 0;
    float older_clean = 0;
};

// Scalar reference for the elementwise latent update. The state is private
// to a latent element and starts at zero for each new request.
inline float unipc_update(const UniPCStep& step, UniPCState& state, float sample,
                          float velocity) {
    float coefficients[rad_unipc::COUNT];
    step.coefficients(coefficients);
    return rad_unipc::advance(coefficients, sample, velocity, state.last_sample, state.last_clean,
                              state.older_clean);
}

}  // namespace wan
