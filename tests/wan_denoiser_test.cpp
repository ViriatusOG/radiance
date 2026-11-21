#include "arch_fakes.h"
#include "../arch/wan/wan_patch.h"
#include "../arch/wan/wan_time.h"
#include "../arch/wan/wan_dit_io.h"
#include "../arch/wan/wan_denoiser.h"

TEST(wan_time_conditioning_uses_zero_only_for_the_reference_frame) {
    RadBuilder builder{};
    rad::arch::Names names("wan");
    wan::TimeConditioning time;
    REQUIRE_EQ(time.declare(&builder, names), RAD_OK);
    REQUIRE_EQ(builder.weights.size(), 6u);
    for (const auto& w : builder.weights) CHECK_EQ(w.d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(pflt(find_op(builder, "timestep_embed_f64"), "max_period"), 10000.0);
    CHECK_EQ(pint(find_op(builder, "gemm_nt_bias", 2), "N"), 18432);
    const float schedule[] = {999.0f, 0.999f};
    RadVideoGenBatch video{};
    video.sched = schedule;
    RadBatch batch{};
    for (int image : {0, 1}) {
        video.has_image = image;
        RadCtx run{};
        run.b = &builder;
        run.batch = &batch;
        time.step(&run, video);
        REQUIRE(run.step_fail.empty());
        REQUIRE_EQ(run.issues.size(), (size_t)(7 + image));
        if (image) CHECK(builder.ops.at(run.issues[0].op - 1).op == "fill");
        const auto& copy = run.issues.at(image);
        CHECK(copy.opd.at(0).raw == schedule);
        CHECK_EQ(copy.opd.at(1).offset, image);
        CHECK_EQ(run.issues.at(image + 1).n, 1 + image);
        CHECK_EQ(run.issues.back().opd.back().handle, time.modulation);
        CHECK_EQ(run.issues.back().opd.back().rows, 1 + image);
        CHECK_LINT(builder, &run);
    }
}

TEST(wan_denoiser_io_uses_latent_geometry_and_preserves_weight_shape) {
    using namespace rad::arch;
    RadBuilder builder{};
    Names names("wan");
    const rad_buf latent = decl_b(&builder, "latent", RAD_F32, {64, 48}, RAD_BUF_PERSIST);
    const rad_buf head = decl_b(&builder, "head", RAD_F32, {16, 192}, RAD_BUF_PERSIST);
    const rad_buf velocity = decl_b(&builder, "velocity", RAD_F32, {64, 48}, RAD_BUF_PERSIST);
    wan::DitPatchIO io;
    REQUIRE_EQ(io.declare_input(&builder, names, latent, 16), RAD_OK);
    REQUIRE_EQ(io.declare_output(&builder, head, velocity, RAD_F32), RAD_OK);
    const auto* weight = find_w(builder, "wan.dit.patch_embedding.weight");
    REQUIRE(weight);
    CHECK_EQ(weight->d.rank, 5u);
    CHECK_EQ(weight->d.shape[1], 48);
    CHECK_EQ(weight->d.shape[2], 1);
    CHECK_EQ(weight->d.shape[3], 2);
    CHECK_EQ(weight->d.shape[4], 2);
    CHECK_EQ(pint(find_op(builder, "gemm_nt_bias"), "K"), 192);

    int32_t dims[RAD_VGEOM_N]{};
    dims[RAD_VGEOM_LAT_FRAMES] = 2;
    dims[RAD_VGEOM_LAT_H] = 4;
    dims[RAD_VGEOM_LAT_W] = 6;
    RadVideoGenBatch video{};
    video.geom.lat_frames = 2;
    video.geom.lat_h = 4;
    video.geom.lat_w = 6;
    video.geom_d = dims;
    RadCtx run{};
    run.b = &builder;
    const int64_t rows = io.step_input(&run, video, RAD_B(latent));
    REQUIRE_EQ(rows, 12);
    io.step_output(&run, video, rows, RAD_B(head), RAD_B(velocity));
    REQUIRE(run.step_fail.empty());
    REQUIRE_EQ(run.issues.size(), 4u);
    CHECK_EQ(run.issues[0].opd.at(0).rows, 48);
    CHECK_EQ(run.issues[1].opd.back().handle, io.hidden);
    CHECK_EQ(run.issues[2].opd.back().handle, io.rotary);
    CHECK_EQ(run.issues[3].opd.at(0).rows, 12);
    CHECK_EQ(run.issues[3].opd.back().rows, 48);
    for (const auto& issue : run.issues)
        for (const auto& operand : issue.opd)
            if (operand.kind == RAD_OPK_RAW) CHECK(operand.raw == dims + RAD_VGEOM_LAT_FRAMES);
    CHECK_LINT(builder, &run);
}

TEST(wan_denoiser_records_all_blocks_with_fp32_residual_and_velocity) {
    using namespace rad::arch;
    RadBuilder builder{};
    Names names("wan");
    const rad_buf latent = decl_b(&builder, "latent", RAD_F32, {64, 48}, RAD_BUF_PERSIST);
    const rad_buf text = decl_b(&builder, "text", RAD_BF16, {512, 3072}, RAD_BUF_PERSIST);
    const rad_buf velocity = decl_b(&builder, "velocity", RAD_F32, {64, 48}, RAD_BUF_PERSIST);
    wan::Denoiser denoiser;
    REQUIRE_EQ(denoiser.declare(&builder, names, latent, text, velocity, 16), RAD_OK);
    CHECK_EQ(count_op(builder, "attn_dense"), 30);
    CHECK_EQ(count_op(builder, "attn_cross"), 30);
    CHECK_EQ(count_op(builder, "wan_gated_add"), 90);
    CHECK_EQ(count_op(builder, "wan_mod_norm"), 61);
    CHECK_EQ(pint(find_op(builder, "wan_mod_norm", 0), "round_norm"), 1);
    CHECK_EQ(pint(find_op(builder, "wan_mod_norm", 2), "round_norm"), 0);
    CHECK_EQ(pint(find_op(builder, "wan_mod_norm", 60), "group"), -1);
    const auto* head = find_w(builder, "wan.dit.head.weight");
    const auto* state = find_b(builder, "wan.dit.state");
    REQUIRE(head && state);
    CHECK_EQ(head->d.dtype, (uint32_t)RAD_F32);
    CHECK_EQ(head->d.shape[0], 192);
    CHECK_EQ(state->d.dtype, (uint32_t)RAD_F32);
    const auto* self_qkv = find_map(builder, "wan.dit.block.0.self.qkv.weight");
    REQUIRE(self_qkv);
    REQUIRE_EQ(self_qkv->src.size(), 3u);
    CHECK_EQ(self_qkv->src[1], std::string("transformer.blocks.0.attn1.to_k.weight"));
    const auto* last_norm = find_map(builder, "wan.dit.block.29.cross.norm_k.weight");
    REQUIRE(last_norm);
    CHECK_EQ(last_norm->src.at(0), std::string("transformer.blocks.29.attn2.norm_k.weight"));

    int32_t geometry[RAD_VGEOM_N]{};
    geometry[RAD_VGEOM_LAT_FRAMES] = 2;
    geometry[RAD_VGEOM_LAT_H] = 4;
    geometry[RAD_VGEOM_LAT_W] = 6;
    geometry[RAD_VGEOM_FIRST_CU + 1] = 6;
    int32_t video_cu[] = {0, 12};
    const float scheduler[] = {999, 1};
    RadVideoGenBatch video{};
    video.geom.lat_frames = 2;
    video.geom.lat_h = 4;
    video.geom.lat_w = 6;
    video.geom_d = geometry;
    video.lat_cu = video_cu;
    video.sched = scheduler;
    for (int image : {0, 1}) {
        video.has_image = image;
        RadCtx run{};
        run.b = &builder;
        CHECK_OK(denoiser.step(&run, video, RAD_B(latent), RAD_B(text), RAD_B(velocity)));
        REQUIRE(run.step_fail.empty());
        for (const auto& issue : run.issues) {
            const auto& op = builder.ops.at(issue.op - 1);
            if (op.op == "wan_mod_norm") {
                CHECK_EQ(issue.opd.at(1).rows, 1 + image);
                if (image)
                    CHECK(issue.opd.at(3).raw == geometry + RAD_VGEOM_FIRST_CU + 1);
                else
                    CHECK_EQ(issue.opd.at(3).kind, RAD_OPK_NONE);
            } else if (op.op == "attn_cross") {
                CHECK_EQ(issue.opd.at(0).rows, 12);
                CHECK_EQ(issue.opd.at(1).rows, 512);
            }
        }
        REQUIRE(!run.issues.empty());
        CHECK_EQ(run.issues.back().opd.back().handle, velocity);
        CHECK_EQ(run.issues.back().opd.back().rows, 48);
        CHECK_LINT(builder, &run);
    }
}

RAD_TEST_MAIN()
