/* Patch projection and output layout at the ends of the Wan denoiser. */
#pragma once

#include <arch/rad_arch.h>

#include "rad_wan_dit.h"
#include "wan_patch.h"

namespace wan {

struct DitPatchIO {
    static constexpr int64_t channels = 48;
    static constexpr int64_t patch_width = 4 * channels;
    static constexpr int64_t hidden_width = 3072;

    rad_buf hidden = 0, rotary = 0;

    int declare_input(RadBuilder* b, rad::arch::Names& names, rad_buf latent,
                      int64_t max_patch_rows) {
        using namespace rad::arch;
        if (!latent || max_patch_rows <= 0 || max_patch_rows > INT32_MAX / 4)
            return RAD_E_SHAPE;
        max_rows = max_patch_rows;
        patches = decl_b(b, names.f("dit.input_patches"), RAD_BF16, {max_rows, patch_width});
        hidden = decl_b(b, names.f("dit.hidden"), RAD_BF16, {max_rows, hidden_width});
        rotary = decl_b(b, names.f("dit.rotary"), RAD_F32, {max_rows, rad_wan_dit::head_width});
        if (!patches || !hidden || !rotary) return RAD_E_INVAL;

        const char* weight = names.f("dit.patch_embedding.weight");
        const char* bias = names.f("dit.patch_embedding.bias");
        RAD_ARCH_TRY(map_copy(b, weight, "transformer.patch_embedding.weight"));
        RAD_ARCH_TRY(map_copy(b, bias, "transformer.patch_embedding.bias"));
        // Keep the source Conv3d shape. Biased GEMM folds the dense trailing
        // dimensions into K, in the same order s2d3d packs a latent patch.
        patch_weight = decl_w(b, weight, RAD_BF16, {hidden_width, channels, 1, 2, 2},
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
        patch_bias = decl_w(b, bias, RAD_BF16, {hidden_width}, RAD_ACCESS_PER_TOKEN,
                            RAD_SHARD_NONE, grp_model());
        if (!patch_weight || !patch_bias) return RAD_E_INVAL;

        pack = rw(b, RAD_OP(b, "s2d3d", RAD_PARAMS(RAD_RANGE("M", 1, max_rows),
                              RAD_INT("C", channels), RAD_INT("st", 1), RAD_INT("sh", 2),
                              RAD_INT("sw", 2), RAD_STR("mode", "copy"), RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS), {latent}, {patches});
        project = rw(b, RAD_OP(b, "gemm_nt_bias", RAD_PARAMS(RAD_RANGE("M", 1, max_rows),
                                 RAD_INT("N", hidden_width), RAD_INT("K", patch_width),
                                 RAD_STR("dtype", "bf16")), RAD_WEIGHTS(patch_weight, patch_bias)),
                       {patches}, {hidden});
        rope_table = rw(b, RAD_OP(b, "wan_rope_table", RAD_PARAMS(RAD_RANGE("M", 1, max_rows),
                                    RAD_STR("dtype", "f32")), RAD_NOWEIGHTS), {}, {rotary});
        return pack && project && rope_table ? RAD_OK : RAD_E_NOKERNEL;
    }

    int declare_output(RadBuilder* b, rad_buf head, rad_buf velocity, uint32_t head_dtype) {
        using namespace rad::arch;
        if (!head || !velocity || max_rows <= 0) return RAD_E_INVAL;
        if (head_dtype != RAD_F32 && head_dtype != RAD_BF16) return RAD_E_DTYPE;
        unpack = rw(b, RAD_OP(b, "wan_unpatch", RAD_PARAMS(RAD_RANGE("M", 1, max_rows),
                                RAD_INT("n", channels), RAD_STR("dtype", rad_dtype_name(head_dtype))),
                              RAD_NOWEIGHTS), {head}, {velocity});
        return unpack ? RAD_OK : RAD_E_NOKERNEL;
    }

    // The caller chooses the lane's F32 latent slice and passes its buffer handle
    // to declare_input for liveness. The view is [4 * patch rows, channels].
    int64_t step_input(RadCtx* c, const RadVideoGenBatch& video, RadOperand latent) const {
        using namespace rad::arch;
        PatchGrid grid{};
        if (!video.geom_d || !patch_grid(video.geom.lat_frames, video.geom.lat_h, video.geom.lat_w, grid) ||
            grid.rows > max_rows) {
            rad_step_fail(c, "wan: invalid denoiser patch grid");
            return RAD_E_SHAPE;
        }
        latent.rows = 4 * grid.rows;
        RAD_ISSUE_N(c, pack, grid.rows, latent, praw(video.geom_d + RAD_VGEOM_LAT_FRAMES, RAD_I32, 3),
                    brows(patches, grid.rows));
        RAD_ISSUE_N(c, project, grid.rows, brows(patches, grid.rows), RAD_W(patch_weight),
                    RAD_W(patch_bias), RAD_NONE, brows(hidden, grid.rows));
        RAD_ISSUE_N(c, rope_table, grid.rows,
                    praw(video.geom_d + RAD_VGEOM_LAT_FRAMES, RAD_I32, 3), brows(rotary, grid.rows));
        return grid.rows;
    }

    void step_output(RadCtx* c, const RadVideoGenBatch& video, int64_t rows,
                     RadOperand head, RadOperand velocity) const {
        using namespace rad::arch;
        if (!video.geom_d || rows <= 0 || rows > max_rows) {
            rad_step_fail(c, "wan: invalid denoiser output geometry");
            return;
        }
        head.rows = rows;
        velocity.rows = 4 * rows;
        RAD_ISSUE_N(c, unpack, rows, head, praw(video.geom_d + RAD_VGEOM_LAT_FRAMES, RAD_I32, 3),
                    velocity);
    }

private:
    int64_t max_rows = 0;
    rad_buf patches = 0;
    rad_weight patch_weight = 0, patch_bias = 0;
    rad_op pack = 0, project = 0, rope_table = 0, unpack = 0;
};

}  // namespace wan
