/* Wan 2.2 TI2V-5B's first-frame VAE encoder. All causal caches start empty. */
#pragma once

#include <arch/rad_vae2d.h>

namespace wan {

struct VaeEncoder {
    static constexpr int64_t patch_channels = 12;
    static constexpr int64_t latent_channels = 48;

    int declare(RadBuilder* b, rad::arch::Names& names, int64_t max_latent_plane,
                rad_buf patches, rad_buf frame_offsets, int weight_group) {
        using namespace rad::arch;
        if (!patches || !frame_offsets || max_latent_plane < 1 ||
            max_latent_plane > INT32_MAX / (64 * 640)) return RAD_E_SHAPE;

        graph.b = b;
        graph.nm = &names;
        graph.scope = "vae.encode";
        graph.down = true;
        graph.base_px = max_latent_plane * 64;
        graph.first_group = weight_group;
        graph.form = VAE_CONV_CAUSAL;
        graph.round_norm_length = true;
        offsets = frame_offsets;
        capacity = max_latent_plane;

        VImg x = graph.conv({patches, patch_channels, 0}, 160, 3, 1, 1, "same", "vae.encoder.conv_in");
        const int64_t channels[] = {160, 320, 640, 640};
        const int temporal_strides[] = {1, 2, 2};
        for (int stage = 0; stage < 4; ++stage) {
            const VImg shortcut = x;
            for (int block = 0; block < 2; ++block)
                x = graph.resblock(x, channels[stage],
                    names.ckpt("vae.encoder.down_blocks.%d.resnets.%d", stage, block));
            if (stage == 3) {
                x = graph.add(x, shortcut);
            } else {
                // The first chunk bypasses the time convolution, but the shortcut still
                // averages the left zero padding when the temporal stride is two.
                x = graph.conv(x, channels[stage], 3, 2, 1, "down",
                    names.ckpt("vae.encoder.down_blocks.%d.downsampler.resample.1", stage), true);
                graph.shortcut(shortcut, x, temporal_strides[stage]);
            }
        }
        x = graph.resblock(x, 640, "vae.encoder.mid_block.resnets.0");
        x = graph.attention(x, "vae.encoder.mid_block.attentions.0", true);
        x = graph.resblock(x, 640, "vae.encoder.mid_block.resnets.1");
        x = graph.norm(x, true, "vae.encoder.norm_out", 4);
        x = graph.conv(x, 2 * latent_channels, 3, 1, 1, "same", "vae.encoder.conv_out");
        encoded = graph.conv(x, 2 * latent_channels, 1, 1, 1, "same", "vae.quant_conv");
        for (const auto& op : graph.ops)
            if (!op.op || !op.y.buf) return RAD_E_NOKERNEL;
        return graph.attn_reads(offsets);
    }

    // The input buffer holds width-first 2x2 RGB patches in the reference's channel order.
    // The offsets buffer holds {0, latent_height * latent_width}.
    int step(RadCtx* c, int64_t latent_height, int64_t latent_width) const {
        if (latent_height < 1 || latent_width < 1 || latent_height > capacity / latent_width) {
            rad_step_fail(c, "wan: invalid VAE encoder geometry");
            return RAD_E_SHAPE;
        }
        graph.issue(c, latent_height * 8, latent_width * 8, rad::arch::brows(offsets, 2), nullptr);
        return RAD_OK;
    }

    rad_buf output() const { return encoded.buf; }

    RadOperand mean(int64_t rows) const {
        return rad::arch::bcol(encoded.buf, 0, latent_channels, rows);
    }

private:
    rad::arch::Vae graph;
    rad::arch::VImg encoded{};
    int64_t capacity = 0;
    rad_buf offsets = 0;
};

}  // namespace wan
