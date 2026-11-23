/* Wan 2.2 TI2V-5B patch geometry. The DiT strides over latent space by
 * (1, 2, 2); the video VAE's stride is independent of this one. */
#pragma once

#include "rad_wan_dit.h"

#include <cstdint>

namespace wan {

struct PatchGrid {
    int64_t frames;
    int64_t height;
    int64_t width;
    int64_t rows;
};

inline bool patch_grid(int64_t latent_frames, int64_t latent_height,
                       int64_t latent_width, PatchGrid& grid) {
    if (latent_frames < 1 || latent_height < 2 || latent_width < 2 ||
        latent_height % 2 || latent_width % 2)
        return false;
    const int64_t height = latent_height / 2;
    const int64_t width = latent_width / 2;
    int64_t rows = 0;
    if (!rad_wan_dit::grid_rows(latent_frames, height, width, rows)) return false;
    grid = { latent_frames, height, width, rows };
    return true;
}

// The input convolution's flattened weights read a row in [channel, dy, dx]
// order. The head emits a row in [dy, dx, channel] order. Both traverse rows
// with x varying fastest, then y, then time.
enum class PatchLayout { ConvInput, HeadOutput };

inline int64_t patch_element(PatchLayout layout, int64_t channel, int64_t dy,
                             int64_t dx, int64_t channels) {
    const int64_t pixel = dy * 2 + dx;
    return layout == PatchLayout::ConvInput ? channel * 4 + pixel
                                            : pixel * channels + channel;
}

// Requires a validated grid and room for three planes of grid.rows floats.
// Axes are contiguous planes of the position tensor.
inline void patch_positions(const PatchGrid& grid, float* positions) {
    const int64_t plane = grid.height * grid.width;
    for (int64_t row = 0; row < grid.rows; ++row) {
        positions[row] = static_cast<float>(row / plane);
        positions[grid.rows + row] = static_cast<float>((row % plane) / grid.width);
        positions[2 * grid.rows + row] = static_cast<float>(row % grid.width);
    }
}

// Requires a validated grid and room for grid.rows floats. The first frame
// gets timestep zero in image-to-video, including every spatial patch.
inline void patch_timesteps(const PatchGrid& grid, bool image_to_video, float timestep,
                            float* timesteps) {
    const int64_t fixed_rows = image_to_video ? grid.height * grid.width : 0;
    for (int64_t row = 0; row < grid.rows; ++row)
        timesteps[row] = row < fixed_rows ? 0.0f : timestep;
}

}  // namespace wan
