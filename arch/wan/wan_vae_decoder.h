/* Wan 2.2 TI2V-5B's decoder. Each call decodes one latent plane. */
#pragma once

#include "wan_vae_time.h"

namespace wan {

static inline int declare_vae_spatial_weights(RadBuilder* b, rad::arch::Names& names,
                                             const char* source, int64_t cin, int64_t cout, int k,
                                             bool causal, RadWeightGroup group,
                                             rad_weight& weight, rad_weight& bias) {
    using namespace rad::arch;
    const char* source_w = names.ckpt("%s.weight", source);
    const char* source_b = names.ckpt("%s.bias", source);
    const char* w = names.f("%s.weight", source);
    const char* v = names.f("%s.bias", source);
    RAD_ARCH_TRY(causal ? map_tap(b, w, source_w, 2, -1) : map_copy(b, w, source_w));
    RAD_ARCH_TRY(map_copy(b, v, source_b));
    weight = decl_w(b, w, RAD_BF16, {cout, cin, k, k}, RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, group);
    bias = decl_w(b, v, RAD_BF16, {cout}, RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, group);
    return weight && bias ? RAD_OK : RAD_E_INVAL;
}

struct VaeFrameConv {
    int declare(RadBuilder* b, rad::arch::Names& names, const char* source, VaeChunkCapacity cap,
                int64_t in_channels, int64_t out_channels, rad_buf x, rad_buf y, int kernel,
                bool upsample, bool causal, RadWeightGroup group) {
        using namespace rad::arch;
        cin = in_channels;
        cout = out_channels;
        up = upsample ? 2 : 1;
        input = x;
        output = y;
        RAD_ARCH_TRY(declare_vae_spatial_weights(b, names, source, cin, cout, kernel, causal,
                                                group, weight, bias));
        op = rw(b, RAD_OP(b, "conv2d_nhwc",
                           RAD_PARAMS(RAD_RANGE("M", 1, cap.plane * up * up), RAD_INT("cin", cin),
                                      RAD_INT("cout", cout), RAD_INT("k", kernel), RAD_INT("stride", 1),
                                      RAD_STR("pad", "same"), RAD_INT("up", up), RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(weight, bias)), {input}, {output});
        return op ? RAD_OK : RAD_E_NOKERNEL;
    }

    void step(RadCtx* c, const VaeVolume& v) const {
        using namespace rad::arch;
        const int64_t in_plane = v.height * v.width;
        const int64_t out_plane = in_plane * up * up;
        const auto* x = static_cast<const uint16_t*>(rad_buf_ptr(c, input));
        auto* y = static_cast<uint16_t*>(rad_buf_ptr(c, output));
        for (int64_t f = 0; f < v.frames; ++f)
            RAD_ISSUE_N(c, op, out_plane,
                        praw2(x + f * in_plane * cin, RAD_BF16, v.height, v.width * cin),
                        RAD_W(weight), RAD_W(bias),
                        praw2(y + f * out_plane * cout, RAD_BF16, v.height * up, v.width * up * cout));
    }

private:
    rad_buf input = 0, output = 0;
    rad_weight weight = 0, bias = 0;
    rad_op op = 0;
    int64_t cin = 0, cout = 0;
    int up = 1;
};

static inline rad_buf declare_vae_volume(RadBuilder* b, rad::arch::Names& names, const char* layer,
                                         const char* suffix, VaeChunkCapacity cap, int64_t channels) {
    return rad::arch::decl_b(b, names.f("%s.%s", layer, suffix), RAD_BF16,
                             {cap.frames * cap.plane, channels});
}

struct VaeDecoderNorm {
    int declare(RadBuilder* b, rad::arch::Names& names, const char* source, VaeChunkCapacity cap,
                int64_t channels, rad_buf x, rad_buf y, bool silu, RadWeightGroup group) {
        using namespace rad::arch;
        input = x;
        output = y;
        const char* w = names.f("%s.gamma", source);
        RAD_ARCH_TRY(map_copy(b, w, names.ckpt("%s.gamma", source)));
        gamma = silu
            ? decl_w(b, w, RAD_BF16, {channels, 1, 1, 1}, RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, group)
            : decl_w(b, w, RAD_BF16, {channels, 1, 1}, RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, group);
        if (!gamma) return RAD_E_INVAL;
        op = rw(b, RAD_OP(b, "l2norm_chan",
                           RAD_PARAMS(RAD_RANGE("M", 1, cap.frames * cap.plane), RAD_INT("n", channels),
                                      RAD_STR("act", silu ? "silu" : "none"), RAD_STR("dtype", "bf16"),
                                      RAD_INT("round_length", 1)),
                           RAD_WEIGHTS(gamma)), {input}, {output});
        return op ? RAD_OK : RAD_E_NOKERNEL;
    }

    void step(RadCtx* c, int64_t rows) const {
        using namespace rad::arch;
        RAD_ISSUE_N(c, op, rows, brows(input, rows), RAD_W(gamma), brows(output, rows));
    }

private:
    rad_buf input = 0, output = 0;
    rad_weight gamma = 0;
    rad_op op = 0;
};

static inline rad_op declare_vae_add(RadBuilder* b, VaeChunkCapacity cap, int64_t channels,
                                     rad_buf a, rad_buf x, rad_buf y) {
    using namespace rad::arch;
    return rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, cap.frames * cap.plane), RAD_INT("n", channels),
                                   RAD_STR("dtype", "bf16")), RAD_NOWEIGHTS),
              {a, x}, {y});
}

struct VaeDecoderResidual {
    int declare(RadBuilder* b, rad::arch::Names& names, const char* source, VaeChunkCapacity cap,
                int64_t cin, int64_t cout, rad_buf x, rad_buf y, rad_buf geom, RadWeightGroup group) {
        input = x;
        output = y;
        shortcut = cin == cout ? x : declare_vae_volume(b, names, source, "skip", cap, cout);
        norm1_out = declare_vae_volume(b, names, source, "normalized1", cap, cin);
        conv1_out = declare_vae_volume(b, names, source, "convolved1", cap, cout);
        norm2_out = declare_vae_volume(b, names, source, "normalized2", cap, cout);
        conv2_out = declare_vae_volume(b, names, source, "convolved2", cap, cout);
        if (!shortcut || !norm1_out || !conv1_out || !norm2_out || !conv2_out || !y)
            return RAD_E_INVAL;
        if (shortcut != input)
            RAD_ARCH_TRY(project_shortcut.declare(b, names, names.ckpt("%s.conv_shortcut", source),
                                                  cap, cin, cout, x, shortcut, 1, false, true, group));
        RAD_ARCH_TRY(norm1.declare(b, names, names.ckpt("%s.norm1", source), cap, cin, x, norm1_out, true, group));
        const char* conv1_name = names.ckpt("%s.conv1", source);
        RAD_ARCH_TRY(conv1.declare(b, names, conv1_name, conv1_name, cap, cin, cout, norm1_out,
                                   conv1_out, geom, group));
        RAD_ARCH_TRY(norm2.declare(b, names, names.ckpt("%s.norm2", source), cap, cout, conv1_out,
                                   norm2_out, true, group));
        const char* conv2_name = names.ckpt("%s.conv2", source);
        RAD_ARCH_TRY(conv2.declare(b, names, conv2_name, conv2_name, cap, cout, cout, norm2_out,
                                   conv2_out, geom, group));
        add = declare_vae_add(b, cap, cout, conv2_out, shortcut, y);
        return add ? RAD_OK : RAD_E_NOKERNEL;
    }

    void step(RadCtx* c, const VaeVolume& v, int lane, bool first) const {
        using namespace rad::arch;
        const int64_t rows = v.frames * v.height * v.width;
        if (shortcut != input) project_shortcut.step(c, v);
        norm1.step(c, rows);
        conv1.step(c, v, lane, first, RAD_B(norm1_out), RAD_B(conv1_out));
        norm2.step(c, rows);
        conv2.step(c, v, lane, first, RAD_B(norm2_out), RAD_B(conv2_out));
        RAD_ISSUE_N(c, add, rows, brows(conv2_out, rows), brows(shortcut, rows), brows(output, rows));
    }

private:
    VaeFrameConv project_shortcut;
    VaeDecoderNorm norm1, norm2;
    VaeCausalConv conv1, conv2;
    rad_buf input = 0, output = 0, shortcut = 0;
    rad_buf norm1_out = 0, conv1_out = 0, norm2_out = 0, conv2_out = 0;
    rad_op add = 0;
};

struct VaeDecoderAttention {
    int declare(RadBuilder* b, rad::arch::Names& names, const char* source, VaeChunkCapacity cap,
                int64_t channels, rad_buf x, rad_buf y, rad_buf geom, RadWeightGroup group) {
        using namespace rad::arch;
        width = channels;
        input = x;
        output = y;
        const auto normalized = declare_vae_volume(b, names, source, "normalized", cap, width);
        qkv = declare_vae_volume(b, names, source, "qkv", cap, 3 * width);
        attended = declare_vae_volume(b, names, source, "attended", cap, width);
        projected = declare_vae_volume(b, names, source, "projected", cap, width);
        offsets = decl_b(b, names.f("%s.offsets", source), RAD_I32, {cap.frames + 1});
        if (!normalized || !qkv || !attended || !projected || !offsets || !output) return RAD_E_INVAL;
        RAD_ARCH_TRY(norm.declare(b, names, names.ckpt("%s.norm", source), cap, width, x,
                                  normalized, false, group));
        RAD_ARCH_TRY(project_in.declare(b, names, names.ckpt("%s.to_qkv", source), cap, width, 3 * width,
                                        normalized, qkv, 1, false, false, group));
        make_offsets = rw(b, RAD_OP(b, "wan_vae_frame_cu", RAD_PARAMS(RAD_RANGE("M", 2, cap.frames + 1)),
                                     RAD_NOWEIGHTS), {geom}, {offsets});
        attention = rw(b, RAD_OP(b, "attn_dense",
                                  RAD_PARAMS(RAD_RANGE("M", 1, cap.frames * cap.plane),
                                             RAD_INT("head_dim", width), RAD_INT("gqa", 1),
                                             RAD_INT("causal", 0), RAD_STR("dtype", "bf16")),
                                  RAD_NOWEIGHTS), {qkv, offsets}, {attended});
        RAD_ARCH_TRY(project_out.declare(b, names, names.ckpt("%s.proj", source), cap, width, width,
                                         attended, projected, 1, false, false, group));
        add = declare_vae_add(b, cap, width, projected, x, y);
        return make_offsets && attention && add ? RAD_OK : RAD_E_NOKERNEL;
    }

    void step(RadCtx* c, const VaeVolume& v) const {
        using namespace rad::arch;
        const int64_t rows = v.frames * v.height * v.width;
        norm.step(c, rows);
        project_in.step(c, v);
        RAD_ISSUE_N(c, make_offsets, v.frames + 1, v.geometry, brows(offsets, v.frames + 1));
        RAD_ISSUE_N(c, attention, rows, bcol(qkv, 0, width, rows), bcol(qkv, width, width, rows),
                    bcol(qkv, 2 * width, width, rows), brows(offsets, v.frames + 1), brows(attended, rows));
        project_out.step(c, v);
        RAD_ISSUE_N(c, add, rows, brows(projected, rows), brows(input, rows), brows(output, rows));
    }

private:
    VaeFrameConv project_in, project_out;
    VaeDecoderNorm norm;
    rad_buf input = 0, output = 0, offsets = 0, qkv = 0, attended = 0, projected = 0;
    rad_op make_offsets = 0, attention = 0, add = 0;
    int64_t width = 0;
};

struct VaeDecoderUpsample {
    int declare(RadBuilder* b, rad::arch::Names& names, const char* source, VaeChunkCapacity cap,
                int64_t cin, int64_t cout, rad_buf skip, rad_buf x, rad_buf y, rad_buf geom,
                rad_buf next_geom, bool up_time, RadWeightGroup group) {
        using namespace rad::arch;
        input = x;
        shortcut = skip;
        output = y;
        output_geometry = next_geom;
        temporal_scale = up_time ? 2 : 1;
        const VaeChunkCapacity temporal_cap{cap.plane, cap.frames * temporal_scale, cap.lanes};
        const VaeChunkCapacity out_cap{4 * temporal_cap.plane, temporal_cap.frames, temporal_cap.lanes};
        const auto spatial_input = up_time
            ? declare_vae_volume(b, names, source, "temporal", temporal_cap, cout) : input;
        const auto spatial_output = declare_vae_volume(b, names, source, "spatial", out_cap, cout);
        shortcut_output = declare_vae_volume(b, names, source, "skip", out_cap, cout);
        if (!spatial_input || !spatial_output || !shortcut_output || !output || !output_geometry)
            return RAD_E_INVAL;
        temporal_output = spatial_input;
        convolved = spatial_output;
        if (up_time) {
            const char* time = names.ckpt("%s.time_conv", source);
            RAD_ARCH_TRY(temporal.declare(b, names, time, time, cap, cout, input, temporal_output, geom, group));
        }
        RAD_ARCH_TRY(spatial.declare(b, names, names.ckpt("%s.resample.1", source), temporal_cap, cout, cout,
                                     spatial_input, spatial_output, 3, true, false, group));
        for (int first = 0; first <= 1; ++first) {
            // DupUp3D keeps only the final repeated temporal plane on a first chunk.
            expand[first] = rw(b, RAD_OP(b, "wan_vae_expand",
                RAD_PARAMS(RAD_RANGE("M", 1, cap.frames * cap.plane), RAD_INT("cin", cin),
                           RAD_INT("cout", cout), RAD_INT("st", temporal_scale), RAD_INT("s", 2),
                           RAD_INT("drop", first ? temporal_scale - 1 : 0),
                           RAD_INT("temporal", 0), RAD_STR("dtype", "bf16")), RAD_NOWEIGHTS),
                {skip, geom}, {shortcut_output});
            geometry[first] = rw(b, RAD_OP(b, "int_affine",
                RAD_PARAMS(RAD_RANGE("M", 3, 3), RAD_STR("src", "0 1 2"),
                           RAD_STR("mul", first || !up_time ? "1 2 2" : "2 2 2"),
                           RAD_STR("add", "0 0 0")), RAD_NOWEIGHTS), {geom}, {output_geometry});
            if (!expand[first] || !geometry[first]) return RAD_E_NOKERNEL;
        }
        add = declare_vae_add(b, out_cap, cout, spatial_output, shortcut_output, y);
        return add ? RAD_OK : RAD_E_NOKERNEL;
    }

    VaeVolume step(RadCtx* c, const VaeVolume& v, int lane, int chunk) const {
        using namespace rad::arch;
        const bool first = chunk == 0;
        VaeVolume features = v;
        if (temporal_scale > 1) {
            temporal.step(c, v, lane, chunk, RAD_B(input), RAD_B(temporal_output));
            if (!first) features.frames *= temporal_scale;
        }
        spatial.step(c, features);
        const int64_t in_rows = v.frames * v.height * v.width;
        VaeVolume out{features.frames, v.height * 2, v.width * 2, brows(output_geometry, 3)};
        const int64_t out_rows = out.frames * out.height * out.width;
        RAD_ISSUE_N(c, expand[first], in_rows, brows(shortcut, in_rows), v.geometry,
                    brows(shortcut_output, out_rows));
        RAD_ISSUE_N(c, geometry[first], 3, v.geometry, out.geometry);
        RAD_ISSUE_N(c, add, out_rows, brows(convolved, out_rows), brows(shortcut_output, out_rows),
                    brows(output, out_rows));
        return out;
    }

private:
    VaeTemporalUp temporal;
    VaeFrameConv spatial;
    int temporal_scale = 1;
    rad_buf input = 0, shortcut = 0, output = 0, output_geometry = 0;
    rad_buf temporal_output = 0, convolved = 0, shortcut_output = 0;
    rad_op expand[2]{}, geometry[2]{}, add = 0;
};

struct VaeDecoder {
    static constexpr int64_t latent_channels = 48;
    static constexpr int64_t patch_channels = 12;
    static constexpr int64_t spatial_scale = 8; // Before RGB unpatchification.

    int declare(RadBuilder* b, rad::arch::Names& names, int64_t max_plane, int64_t lanes,
                rad_buf latent, rad_buf geom, RadWeightGroup group) {
        using namespace rad::arch;
        limits = {max_plane, 1, lanes};
        // The first full-resolution residual takes 512 channels, four frames and two history
        // frames. The device convolution addresses its input with 32-bit byte offsets.
        constexpr int64_t max_plane_bytes = (4 + 2) * 512 * 2 * spatial_scale * spatial_scale;
        if (!limits.valid() || !latent || !geom || max_plane > 0xffff0000LL / max_plane_bytes)
            return RAD_E_SHAPE;

        input_geometry = geom;
        VaeChunkCapacity cap = limits;
        post_quant_out = declare_vae_volume(b, names, "vae.decode", "post_quant", cap, latent_channels);
        conv_in_out = declare_vae_volume(b, names, "vae.decode", "conv_in", cap, 1024);
        if (!post_quant_out || !conv_in_out) return RAD_E_INVAL;
        RAD_ARCH_TRY(post_quant.declare(b, names, "vae.post_quant_conv", cap, latent_channels,
                                       latent_channels, latent, post_quant_out, 1, false, true, group));
        RAD_ARCH_TRY(conv_in.declare(b, names, "vae.decoder.conv_in", "vae.decoder.conv_in", cap,
                                    latent_channels, 1024, post_quant_out, conv_in_out, geom, group));

        const auto mid0 = declare_vae_volume(b, names, "vae.decode", "mid0", cap, 1024);
        const auto mid_attended = declare_vae_volume(b, names, "vae.decode", "mid_attended", cap, 1024);
        const auto mid1 = declare_vae_volume(b, names, "vae.decode", "mid1", cap, 1024);
        if (!mid0 || !mid_attended || !mid1) return RAD_E_INVAL;
        RAD_ARCH_TRY(middle[0].declare(b, names, "vae.decoder.mid_block.resnets.0", cap, 1024, 1024,
                                      conv_in_out, mid0, geom, group));
        RAD_ARCH_TRY(attention.declare(b, names, "vae.decoder.mid_block.attentions.0", cap, 1024,
                                      mid0, mid_attended, geom, group));
        RAD_ARCH_TRY(middle[1].declare(b, names, "vae.decoder.mid_block.resnets.1", cap, 1024, 1024,
                                      mid_attended, mid1, geom, group));

        rad_buf input = mid1;
        int64_t width = 1024;
        const int64_t channels[] = {1024, 1024, 512, 256};
        for (int stage = 0; stage < 4; ++stage) {
            const rad_buf skip = input;
            const int64_t skip_width = width;
            for (int block = 0; block < 3; ++block) {
                const auto* name = names.ckpt("vae.decoder.up_blocks.%d.resnets.%d", stage, block);
                const auto output = declare_vae_volume(b, names, name, "out", cap, channels[stage]);
                RAD_ARCH_TRY(residual[stage][block].declare(b, names, name, cap, width, channels[stage],
                                                           input, output, geom, group));
                input = output;
                width = channels[stage];
            }
            if (stage == 3) break;
            const bool temporal = stage < 2;
            const VaeChunkCapacity out_cap{cap.plane * 4, cap.frames * (temporal ? 2 : 1), cap.lanes};
            const auto* name = names.ckpt("vae.decoder.up_blocks.%d.upsampler", stage);
            const auto out = declare_vae_volume(b, names, name, "out", out_cap, width);
            const auto out_geom = decl_b(b, names.f("%s.geom", name), RAD_I32, {3});
            RAD_ARCH_TRY(upsample[stage].declare(b, names, name, cap, skip_width, width, skip,
                                                input, out, geom, out_geom, temporal, group));
            cap = out_cap;
            input = out;
            geom = out_geom;
        }

        output_geometry = geom;
        normalized = declare_vae_volume(b, names, "vae.decode", "normalized", cap, width);
        patches = declare_vae_volume(b, names, "vae.decode", "patches", cap, patch_channels);
        if (!normalized || !patches) return RAD_E_INVAL;
        RAD_ARCH_TRY(final_norm.declare(b, names, "vae.decoder.norm_out", cap, width, input,
                                       normalized, true, group));
        return conv_out.declare(b, names, "vae.decoder.conv_out", "vae.decoder.conv_out", cap,
                                width, patch_channels, normalized, patches, geom, group);
    }

    int declare_rgb(RadBuilder* b, rad::arch::Names& names, rad_buf frames) {
        using namespace rad::arch;
        if (!frames || !patches || !output_geometry) return RAD_E_INVAL;
        rgb = frames;
        first_frame = decl_b(b, names.f("vae.decode.first_frame"), RAD_I32, {1});
        const int64_t max_patch_rows = 4 * spatial_scale * spatial_scale * limits.plane;
        rgb_patches = decl_b(b, names.f("vae.decode.rgb_patches"), RAD_F32, {max_patch_rows, patch_channels});
        if (!first_frame || !rgb_patches) return RAD_E_INVAL;
        // The reference widens the decoder's BF16 patches before pixel normalization.
        widen_rgb = rw(b, RAD_OP(b, "cast",
            RAD_PARAMS(RAD_RANGE("M", 1, max_patch_rows), RAD_INT("n", patch_channels),
                       RAD_STR("from", "bf16"), RAD_STR("to", "f32")), RAD_NOWEIGHTS),
            {patches}, {rgb_patches});
        set_first = rw(b, RAD_OP(b, "int_affine",
                                 RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_STR("src", "-1"),
                                            RAD_STR("mul", "0"), RAD_STR("add", "0")), RAD_NOWEIGHTS),
                       {output_geometry}, {first_frame});
        // The decoder emits the entire spatial plane. Use the untiled patch-store path.
        constexpr int64_t full_plane = rad_wan_vae::max_spatial_axis;
        store_rgb = rw(b, RAD_OP(b, "tile_stitch",
            RAD_PARAMS(RAD_RANGE("M", 1, 4 * 2 * spatial_scale * limits.plane),
                       RAD_INT("tile", full_plane), RAD_INT("stride", full_plane),
                       RAD_INT("scale", 1), RAD_INT("scale_t", 1), RAD_INT("p", 2),
                       RAD_INT("C", 3), RAD_STR("order", "wh"), RAD_STR("dtype", "f32")),
            RAD_NOWEIGHTS), {rgb_patches, output_geometry, first_frame}, {rgb});
        return widen_rgb && set_first && store_rgb ? RAD_OK : RAD_E_NOKERNEL;
    }

    // Calls share transient buffers and must run serially. Each lane starts at chunk zero,
    // then advances in order at fixed geometry until its next reset.
    int step(RadCtx* c, int64_t height, int64_t width, int lane, int chunk) const {
        using namespace rad::arch;
        VaeVolume volume{1, height, width, brows(input_geometry, 3)};
        if (chunk < 0 || !limits.fits(volume, lane) ||
            height > rad_wan_vae::max_spatial_axis / spatial_scale ||
            width > rad_wan_vae::max_spatial_axis / spatial_scale) {
            rad_step_fail(c, "wan: invalid VAE decoder chunk");
            return RAD_E_SHAPE;
        }
        const bool first = chunk == 0;
        post_quant.step(c, volume);
        conv_in.step(c, volume, lane, first, RAD_B(post_quant_out), RAD_B(conv_in_out));
        middle[0].step(c, volume, lane, first);
        attention.step(c, volume);
        middle[1].step(c, volume, lane, first);
        for (int stage = 0; stage < 4; ++stage) {
            for (const auto& block : residual[stage]) block.step(c, volume, lane, first);
            if (stage < 3) volume = upsample[stage].step(c, volume, lane, chunk);
        }
        final_norm.step(c, volume.frames * volume.height * volume.width);
        conv_out.step(c, volume, lane, first, RAD_B(normalized), RAD_B(patches));
        if (store_rgb) {
            const int64_t patch_rows = volume.frames * volume.height * volume.width;
            const int64_t pixel_rows = 2 * volume.frames * volume.height;
            RAD_ISSUE_N(c, widen_rgb, patch_rows, brows(patches, patch_rows), brows(rgb_patches, patch_rows));
            RAD_ISSUE_N(c, set_first, 1, volume.geometry, brows(first_frame, 1));
            RAD_ISSUE_N(c, store_rgb, pixel_rows, brows(rgb_patches, patch_rows),
                        volume.geometry, brows(first_frame, 1),
                        praw2(rad_buf_ptr(c, rgb), RAD_U8, pixel_rows, 2 * volume.width * 3));
        }
        return RAD_OK;
    }

    rad_buf output() const { return patches; }

private:
    VaeChunkCapacity limits{};
    VaeFrameConv post_quant;
    VaeCausalConv conv_in, conv_out;
    VaeDecoderResidual middle[2], residual[4][3];
    VaeDecoderAttention attention;
    VaeDecoderUpsample upsample[3];
    VaeDecoderNorm final_norm;
    rad_buf input_geometry = 0, post_quant_out = 0, conv_in_out = 0, normalized = 0, patches = 0;
    rad_buf output_geometry = 0, first_frame = 0, rgb_patches = 0, rgb = 0;
    rad_op widen_rgb = 0, set_first = 0, store_rgb = 0;
};

}  // namespace wan
