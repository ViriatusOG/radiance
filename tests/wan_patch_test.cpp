#include "rad_test.h"
#include "../arch/wan/wan_patch.h"

#include <array>

TEST(ti2v_patch_rows_follow_the_reference_flatten_order) {
    wan::PatchGrid grid{};
    REQUIRE(wan::patch_grid(2, 4, 6, grid));
    REQUIRE_EQ(grid.rows, 12);
    std::array<float, 36> positions{};
    wan::patch_positions(grid, positions.data());
    // Conv3d output [frame, height, width] flattened with width fastest.
    const std::array<float, 36> expected = {
        0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1,  0, 0, 0, 1, 1, 1,
        0, 1, 2, 0, 1, 2,  0, 1, 2, 0, 1, 2,
    };
    CHECK(positions == expected);
    CHECK(!wan::patch_grid(1, 3, 4, grid));
    CHECK(!wan::patch_grid(1025, 4, 6, grid));
    CHECK(!wan::patch_grid(1, 4, 2050, grid));
}

TEST(i2v_zeroes_the_first_patch_plane_only) {
    wan::PatchGrid grid{};
    REQUIRE(wan::patch_grid(2, 4, 6, grid));
    std::array<float, 12> times{};
    wan::patch_timesteps(grid, true, 781.0f, times.data());
    const std::array<float, 12> image_expected = {
        0, 0, 0, 0, 0, 0, 781, 781, 781, 781, 781, 781,
    };
    CHECK(times == image_expected);
    wan::patch_timesteps(grid, false, 781.0f, times.data());
    for (const float time : times) CHECK_EQ(time, 781.0f);
}

TEST(input_convolution_and_output_head_use_distinct_patch_columns) {
    using wan::PatchLayout;
    // The last pixel of a 2x2 patch, channel 5 of Wan's 48 latents.
    CHECK_EQ(wan::patch_element(PatchLayout::ConvInput, 5, 1, 1, 48), 23);
    CHECK_EQ(wan::patch_element(PatchLayout::HeadOutput, 5, 1, 1, 48), 149);
}

RAD_TEST_MAIN()
