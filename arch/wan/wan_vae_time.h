/* Causal convolutions and temporal upsampling for Wan's chunked decoder. */
#pragma once

#include <arch/rad_arch.h>

#include "rad_wan_vae.h"

namespace wan {

struct VaeVolume {
    int64_t frames;
    int64_t height;
    int64_t width;
    RadOperand geometry; // Device I32 {frames, height, width}, matching the host dimensions.
};

struct VaeChunkCapacity {
    int64_t plane;
    int64_t frames;
    int64_t lanes;

    bool valid() const {
        return plane >= 1 && plane <= INT32_MAX / 256 && frames >= 1 && frames <= 8 &&
               lanes >= 1 && lanes <= 16;
    }

    bool fits(const VaeVolume& volume, int lane) const {
        const bool have_geometry = volume.geometry.kind == RAD_OPK_BUF ? volume.geometry.handle != 0
            : volume.geometry.kind == RAD_OPK_RAW && volume.geometry.raw;
        return lane >= 0 && lane < lanes && volume.frames >= 1 && volume.frames <= frames &&
               volume.height >= 1 && volume.height <= rad_wan_vae::max_spatial_axis &&
               volume.width >= 1 && volume.width <= rad_wan_vae::max_spatial_axis &&
               volume.height <= plane / volume.width && have_geometry;
    }
};

struct VaeCausalHistory {
    int declare(RadBuilder* b, rad::arch::Names& names, const char* name, VaeChunkCapacity capacity,
                int64_t channels, rad_buf input, rad_buf output, bool temporal) {
        using namespace rad::arch;
        limits = capacity;
        width = channels;
        saved = decl_b(b, names.f("%s.history", name), RAD_BF16,
                       {2 * limits.plane * limits.lanes, width}, RAD_BUF_PERSIST);
        if (!saved) return RAD_E_INVAL;
        for (int i = 0; i < 2; ++i) {
            op[i] = rw(b, RAD_OP(b, "wan_vae_history",
                                 RAD_PARAMS(RAD_RANGE("M", 1, limits.frames * limits.plane),
                                            RAD_INT("temporal", temporal), RAD_INT("reset", i),
                                            RAD_STR("dtype", "bf16")), RAD_NOWEIGHTS),
                       {input, saved}, {output, saved});
            if (!op[i]) return RAD_E_NOKERNEL;
        }
        return RAD_OK;
    }

    void step(RadCtx* c, RadOperand x, RadOperand y, int64_t plane, int64_t rows,
              int lane, bool reset) const {
        // Geometry stays fixed for a lane from its reset until the final chunk.
        const auto cache = rad::arch::brow_slice(saved, lane * 2 * limits.plane, 2 * plane, width);
        RAD_ISSUE_N(c, op[reset], rows, x, cache, y);
    }

private:
    VaeChunkCapacity limits{};
    int64_t width = 0;
    rad_buf saved = 0;
    rad_op op[2]{};
};

static inline int declare_vae_conv_weights(RadBuilder* b, rad::arch::Names& names, const char* name,
                                           const char* source, int64_t cin, int64_t cout, int64_t k,
                                           RadWeightGroup group, rad_weight& weight, rad_weight& bias) {
    using namespace rad::arch;
    const char* w = names.f("%s.weight", name);
    const char* v = names.f("%s.bias", name);
    RAD_ARCH_TRY(map_copy(b, w, names.ckpt("%s.weight", source)));
    RAD_ARCH_TRY(map_copy(b, v, names.ckpt("%s.bias", source)));
    weight = decl_w(b, w, RAD_BF16, {cout, cin, 3, k, k}, RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, group);
    bias = decl_w(b, v, RAD_BF16, {cout}, RAD_ACCESS_PER_REQUEST, RAD_SHARD_NONE, group);
    return weight && bias ? RAD_OK : RAD_E_INVAL;
}

struct VaeCausalConv {
    int declare(RadBuilder* b, rad::arch::Names& names, const char* name, const char* source,
                VaeChunkCapacity capacity, int64_t in_channels, int64_t out_channels,
                rad_buf input, rad_buf output, rad_buf geometry, RadWeightGroup group) {
        using namespace rad::arch;
        if (!capacity.valid() || !input || !output || in_channels < 8 || in_channels > 4096 ||
            in_channels % 8 || out_channels < 1 || out_channels > 4096) return RAD_E_SHAPE;
        limits = capacity;
        cout = out_channels;
        const int64_t max_rows = limits.frames * limits.plane;
        const int64_t padded_rows = max_rows + 2 * limits.plane;
        // conv3d_ndhwc's device gather uses 32-bit byte offsets.
        if (padded_rows * in_channels * 2 > 0xffff0000LL) return RAD_E_SHAPE;
        padded = decl_b(b, names.f("%s.padded", name), RAD_BF16, {padded_rows, in_channels});
        convolved = decl_b(b, names.f("%s.convolved", name), RAD_BF16, {padded_rows, cout});
        padded_geom = decl_b(b, names.f("%s.padded_geom", name), RAD_I32, {3});
        if (!padded || !convolved || !padded_geom) return RAD_E_INVAL;
        RAD_ARCH_TRY(declare_vae_conv_weights(b, names, name, source, in_channels, cout, 3,
                                             group, weight, bias));
        RAD_ARCH_TRY(history.declare(b, names, name, limits, in_channels, input, padded, false));
        const auto geom_op = RAD_OP(b, "int_affine",
                                     RAD_PARAMS(RAD_RANGE("M", 3, 3), RAD_STR("src", "0 1 2"),
                                                RAD_STR("mul", "1 1 1"), RAD_STR("add", "2 0 0")),
                                     RAD_NOWEIGHTS);
        geometry_op = geometry ? rw(b, geom_op, {geometry}, {padded_geom})
                               : rw(b, geom_op, {}, {padded_geom});
        conv = rw(b, RAD_OP(b, "conv3d_ndhwc",
                             RAD_PARAMS(RAD_RANGE("M", 1, padded_rows), RAD_INT("cin", in_channels),
                                        RAD_INT("cout", cout), RAD_INT("kt", 3), RAD_INT("k", 3),
                                        RAD_INT("causal", 1), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(weight, bias)), {padded, padded_geom}, {convolved});
        crop = rw(b, RAD_OP(b, "cast",
                             RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("n", cout),
                                        RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                             RAD_NOWEIGHTS), {convolved}, {output});
        return geometry_op && conv && crop ? RAD_OK : RAD_E_NOKERNEL;
    }

    void step(RadCtx* c, const VaeVolume& volume, int lane, bool first, RadOperand x,
              RadOperand y) const {
        using namespace rad::arch;
        if (!limits.fits(volume, lane)) {
            rad_step_fail(c, "wan: invalid causal VAE convolution volume");
            return;
        }
        const int64_t plane = volume.height * volume.width, rows = volume.frames * plane;
        const int64_t padded_rows = rows + 2 * plane;
        x.rows = y.rows = rows;
        history.step(c, x, brows(padded, padded_rows), plane, rows, lane, first);
        RAD_ISSUE_N(c, geometry_op, 3, volume.geometry, brows(padded_geom, 3));
        RAD_ISSUE_N(c, conv, padded_rows, brows(padded, padded_rows), RAD_W(weight), RAD_W(bias),
                    brows(padded_geom, 3), RAD_NONE, brows(convolved, padded_rows));
        // The first two convolution outputs are for the supplied history.
        RAD_ISSUE_N(c, crop, rows, brow_slice(convolved, 2 * plane, rows, cout), y);
    }

private:
    VaeChunkCapacity limits{};
    VaeCausalHistory history;
    int64_t cout = 0;
    rad_buf padded = 0, convolved = 0, padded_geom = 0;
    rad_weight weight = 0, bias = 0;
    rad_op geometry_op = 0, conv = 0, crop = 0;
};

struct VaeTemporalUp {
    int declare(RadBuilder* b, rad::arch::Names& names, const char* name, const char* source,
                VaeChunkCapacity capacity, int64_t channels, rad_buf input, rad_buf output,
                rad_buf geometry, RadWeightGroup group) {
        using namespace rad::arch;
        if (!capacity.valid() || !input || !output || channels < 8 || channels > 2048 ||
            channels % 8) return RAD_E_SHAPE;
        limits = capacity;
        width = channels;
        const int64_t max_rows = limits.frames * limits.plane;
        if (max_rows * 3 * width >= (1LL << 32)) return RAD_E_SHAPE;
        windows = decl_b(b, names.f("%s.windows", name), RAD_BF16, {max_rows, 3 * width});
        packed = decl_b(b, names.f("%s.projected", name), RAD_BF16, {max_rows, 2 * width});
        if (!windows || !packed) return RAD_E_INVAL;
        RAD_ARCH_TRY(declare_vae_conv_weights(b, names, name, source, width, 2 * width, 1,
                                             group, weight, bias));
        RAD_ARCH_TRY(history.declare(b, names, name, limits, width, input, windows, true));
        project = rw(b, RAD_OP(b, "gemm_nt_bias",
                                RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("N", 2 * width),
                                           RAD_INT("K", 3 * width), RAD_STR("dtype", "bf16")),
                                RAD_WEIGHTS(weight, bias)), {windows}, {packed});
        const auto expand = RAD_OP(b, "wan_vae_expand",
                                    RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("cin", 2 * width),
                                               RAD_INT("cout", width), RAD_INT("st", 2),
                                               RAD_INT("s", 1), RAD_INT("drop", 0),
                                               RAD_INT("temporal", 1), RAD_STR("dtype", "bf16")),
                                    RAD_NOWEIGHTS);
        shuffle = geometry ? rw(b, expand, {packed, geometry}, {output})
                           : rw(b, expand, {packed}, {output});
        bypass = rw(b, RAD_OP(b, "cast",
                               RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("n", width),
                                          RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                               RAD_NOWEIGHTS), {input}, {output});
        return project && shuffle && bypass ? RAD_OK : RAD_E_NOKERNEL;
    }

    void step(RadCtx* c, const VaeVolume& volume, int lane, int chunk, RadOperand x,
              RadOperand y) const {
        using namespace rad::arch;
        if (chunk < 0 || !limits.fits(volume, lane)) {
            rad_step_fail(c, "wan: invalid temporal VAE upsample volume or chunk");
            return;
        }
        const int64_t plane = volume.height * volume.width, rows = volume.frames * plane;
        x.rows = rows;
        if (chunk == 0) {
            // Wan uses a sentinel on the first chunk. It bypasses the temporal
            // convolution and does not put these features in its time cache.
            y.rows = rows;
            RAD_ISSUE_N(c, bypass, rows, x, y);
            return;
        }
        y.rows = 2 * rows;
        history.step(c, x, brows(windows, rows), plane, rows, lane, chunk == 1);
        RAD_ISSUE_N(c, project, rows, brows(windows, rows), RAD_W(weight), RAD_W(bias),
                    RAD_NONE, brows(packed, rows));
        RAD_ISSUE_N(c, shuffle, rows, brows(packed, rows), volume.geometry, y);
    }

private:
    VaeChunkCapacity limits{};
    VaeCausalHistory history;
    int64_t width = 0;
    rad_buf windows = 0, packed = 0;
    rad_weight weight = 0, bias = 0;
    rad_op project = 0, shuffle = 0, bypass = 0;
};

}  // namespace wan
