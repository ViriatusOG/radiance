#include "rad_test.h"
#include "../core/gen/wan_unipc.h"

#include <array>
#include <cmath>
#include <limits>

// Scalar latents from the pinned Wan FlowUniPCMultistepScheduler (1ea34ff4),
// using PyTorch float32 and applying the same input-dependent velocity at every step.
TEST(shift_five_predictor_and_corrector_match_wan) {
    const auto steps = wan::unipc_plan(4, 5.0);
    REQUIRE_EQ(steps.size(), 4u);
    const std::array<int, 4> timesteps{ 999, 937, 833, 624 };
    const std::array<float, 4> sigmas{ .999799848f, .937265456f, .833055377f, .624687374f };
    const std::array<float, 4> expected{ .320561081f, .320648879f, .294746846f, .201659024f };
    wan::UniPCState state;
    float sample = .3125f;
    for (int i = 0; i < 4; ++i) {
        CHECK_EQ(steps[i].timestep, timesteps[i]);
        CHECK_EQ(steps[i].sigma, sigmas[i]);
        const float velocity = sample * .1875f + (i - 2) * .09375f;
        sample = wan::unipc_update(steps[i], state, sample, velocity);
        CHECK(std::abs(sample - expected[i]) < 2e-7f);
    }
}

TEST(a_single_evaluation_returns_the_clean_prediction) {
    const auto steps = wan::unipc_plan(1, 5.0);
    REQUIRE_EQ(steps.size(), 1u);
    wan::UniPCState state;
    const float sample = wan::unipc_update(steps[0], state, .3125f, -.12890625f);
    CHECK(std::abs(sample - .441380441f) < 1e-7f);
}

TEST(the_grid_rejects_unsupported_or_unrepresentable_requests) {
    CHECK(wan::unipc_plan(0, 5.0).empty());
    CHECK(wan::unipc_plan(1001, 5.0).empty());
    CHECK(wan::unipc_plan(50, 0.0).empty());
    CHECK(wan::unipc_plan(50, 101.0).empty());
    CHECK(wan::unipc_plan(50, std::numeric_limits<double>::quiet_NaN()).empty());
    CHECK(wan::unipc_plan(50, std::numeric_limits<double>::min()).empty());
}

TEST(equal_integer_timesteps_keep_both_sigma_steps) {
    const auto steps = wan::unipc_plan(1000, 5.0);
    REQUIRE_EQ(steps.size(), 1000u);
    CHECK_EQ(steps[0].timestep, 999);
    CHECK_EQ(steps[1].timestep, 999);
    CHECK(steps[0].sigma > steps[1].sigma);
    CHECK_EQ(steps[0].correct_sample, 0.0f);
    CHECK(steps[1].correct_sample > 0.0f);
}

RAD_TEST_MAIN()
