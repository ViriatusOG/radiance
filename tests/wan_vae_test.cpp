#include "arch_fakes.h"
#include "../arch/wan/wan_vae_time.h"
#include "../arch/wan/wan_vae_encoder.h"
#include "../arch/wan/wan_vae_decoder.h"

TEST(wan_vae_time_cache_bypasses_first_chunk_and_isolates_lanes) {
    using namespace rad::arch;
    RadBuilder b{};
    Names names("wan");
    const rad_buf in = decl_b(&b, "in", RAD_BF16, {16, 8}, RAD_BUF_PERSIST);
    const rad_buf out = decl_b(&b, "out", RAD_BF16, {32, 8}, RAD_BUF_PERSIST);
    const int32_t dims[] = {1, 2, 2};
    const wan::VaeVolume volume{1, 2, 2, praw(dims, RAD_I32, 3)};
    const wan::VaeChunkCapacity limit{8, 2, 2};
    const wan::VaeChunkCapacity wide{9000, 2, 2};
    CHECK(!wide.fits({1, 1, 9000, volume.geometry}, 0));
    CHECK(!limit.fits({1, 2, 2, RAD_NONE}, 0));
    wan::VaeTemporalUp up;
    REQUIRE_EQ(up.declare(&b, names, "vae.up", "vae.decoder.up.time_conv",
                         limit, 8, in, out, 0, grp_layer(0)), RAD_OK);
    const auto* weight = find_w(b, "wan.vae.up.weight");
    REQUIRE(weight);
    CHECK_EQ(weight->d.rank, 5u);
    CHECK_EQ(weight->d.shape[0], 16);
    CHECK_EQ(weight->d.shape[1], 8);
    CHECK_EQ(weight->d.shape[2], 3);
    CHECK_EQ(weight->d.shape[3], 1);
    for (int chunk : {0, 1, 2}) {
        RadCtx run{};
        run.b = &b;
        up.step(&run, volume, 1, chunk, RAD_B(in), RAD_B(out));
        REQUIRE(run.step_fail.empty());
        if (chunk == 0) {
            REQUIRE_EQ(run.issues.size(), 1u);
            CHECK(b.ops.at(run.issues[0].op - 1).op == "cast");
            CHECK_EQ(run.issues[0].opd.back().rows, 4);
        } else {
            REQUIRE_EQ(run.issues.size(), 3u);
            const auto& issue = run.issues.front();
            const auto& op = b.ops.at(issue.op - 1);
            CHECK(op.op == "wan_vae_history");
            CHECK_EQ(pint(&op, "reset"), chunk == 1);
            CHECK_EQ(issue.opd.at(1).offset, 2 * 8 * 8);
            CHECK_EQ(issue.opd.at(1).rows, 8);
            CHECK_EQ(run.issues.back().opd.back().rows, 8);
        }
        CHECK_LINT(b, &run);
    }
}

TEST(wan_vae_causal_convolution_excludes_history_outputs) {
    using namespace rad::arch;
    RadBuilder b{};
    Names names("wan");
    const auto in = decl_b(&b, "in", RAD_BF16, {16, 8}, RAD_BUF_PERSIST);
    const auto out = decl_b(&b, "out", RAD_BF16, {16, 8}, RAD_BUF_PERSIST);
    wan::VaeCausalConv conv;
    REQUIRE_EQ(conv.declare(&b, names, "vae.conv", "vae.decoder.conv_in", {8, 2, 2},
                           8, 8, in, out, 0, grp_layer(0)), RAD_OK);
    int32_t dims[] = {2, 2, 2};
    RadCtx run{};
    run.b = &b;
    conv.step(&run, {2, 2, 2, praw(dims, RAD_I32, 3)}, 0, false, RAD_B(in), RAD_B(out));
    REQUIRE_EQ(run.issues.size(), 4u);
    CHECK_EQ(run.issues[2].n, 16);
    CHECK_EQ(run.issues[3].opd.at(0).offset, 2 * 4 * 8);
    CHECK_EQ(run.issues[3].opd.back().rows, 8);
    CHECK_LINT(b, &run);
}

TEST(wan_vae_encoder_uses_last_temporal_taps_and_spatial_checkpoint_weights) {
    using namespace rad::arch;
    RadBuilder b{};
    Names names("wan");
    const auto input = decl_b(&b, "patches", RAD_BF16, {12 * 64, 12}, RAD_BUF_PERSIST);
    const auto cu = decl_b(&b, "offsets", RAD_I32, {2}, RAD_BUF_PERSIST);
    wan::VaeEncoder enc;
    REQUIRE_EQ(enc.declare(&b, names, 12, input, cu, 2), RAD_OK);
    const auto* conv = find_map(b, "wan.vae.encoder.conv_in.weight");
    const auto* resample = find_map(b, "wan.vae.encoder.down_blocks.2.downsampler.resample.1.weight");
    const auto* attention = find_map(b, "wan.vae.encoder.mid_block.attentions.0.to_qkv.weight");
    REQUIRE(conv && resample && attention);
    CHECK_EQ(conv->mode, RAD_MAP_TAP);
    CHECK_EQ(resample->mode, RAD_MAP_COPY);
    CHECK_EQ(attention->mode, RAD_MAP_COPY);
    CHECK(attention->src == (std::vector<std::string>{"vae.encoder.mid_block.attentions.0.to_qkv.weight"}));
    const auto* w = find_w(b, "wan.vae.encoder.conv_in.weight");
    REQUIRE(w);
    CHECK_EQ(w->d.shape[1], 12);
    CHECK_EQ(w->d.rank, 4u);
    RadCtx run{};
    run.b = &b;
    REQUIRE_EQ(enc.step(&run, 2, 3), RAD_OK);
    int shortcuts = 0, attention_calls = 0;
    for (const auto& issue : run.issues) {
        const auto* op = b.op_of(issue.op);
        if (op->op == "l2norm_chan") CHECK_EQ(pint(op, "round_length"), 1);
        if (op->op == "s2d_avg_add") {
            CHECK_EQ(pint(op, "ft"), shortcuts == 0 ? 1 : 2);
            ++shortcuts;
        }
        if (op->op == "attn_dense") {
            ++attention_calls;
            CHECK_EQ(pint(op, "head_dim"), 640);
            CHECK_EQ(issue.n, 6);
            CHECK_EQ(issue.opd[3].handle, cu);
        }
    }
    CHECK_EQ(shortcuts, 3);
    CHECK_EQ(attention_calls, 1);
    CHECK_EQ(enc.mean(6).handle, enc.output());
    CHECK_EQ(enc.mean(6).cols, 48);
    CHECK_LINT(b, &run);
}

TEST(wan_vae_decoder_emits_one_frame_then_four_and_declares_causal_history) {
    using namespace rad::arch;
    RadBuilder b{};
    Names names("wan");
    const auto latent = decl_b(&b, "latent", RAD_BF16, {24, 48}, RAD_BUF_PERSIST);
    const auto geom = decl_b(&b, "geometry", RAD_I32, {3}, RAD_BUF_PERSIST);
    const auto rgb = decl_b(&b, "frames", RAD_U8, {4 * 24 * 256, 3}, RAD_BUF_PERSIST);
    wan::VaeDecoder decoder;
    REQUIRE_EQ(decoder.declare(&b, names, 24, 2, latent, geom, grp_layer(3)), RAD_OK);
    REQUIRE_EQ(decoder.declare_rgb(&b, names, rgb), RAD_OK);
    auto* mid = find_map(b, "wan.vae.decoder.mid_block.attentions.0.to_qkv.weight");
    REQUIRE(mid);
    CHECK_EQ(mid->mode, RAD_MAP_COPY);
    auto* after = find_w(b, "wan.vae.decoder.up_blocks.3.resnets.0.conv1.weight");
    REQUIRE(after);
    CHECK_EQ(after->d.shape[0], 256);
    CHECK_EQ(after->d.shape[1], 512);
    CHECK_EQ(after->d.shape[2], 3);
    auto* time = find_w(b, "wan.vae.decoder.up_blocks.1.upsampler.time_conv.weight");
    REQUIRE(time);
    CHECK_EQ(time->d.shape[0], 2048);
    CHECK_EQ(time->d.shape[1], 1024);
    CHECK(!find_w(b, "wan.vae.decoder.up_blocks.2.upsampler.time_conv.weight"));

    for (int chunk = 0; chunk < 3; ++chunk) {
        RadCtx run{};
        run.b = &b;
        REQUIRE_EQ(decoder.step(&run, 4, 6, 1, chunk), RAD_OK);
        CHECK(run.step_fail.empty());
        REQUIRE(!run.issues.empty());
        int spatial_history = 0, time_history = 0, shortcuts = 0;
        for (const auto& issue : run.issues) {
            const auto* op = b.op_of(issue.op);
            if (op->op == "wan_vae_history") {
                if (pint(op, "temporal")) {
                    ++time_history;
                    CHECK_EQ(pint(op, "reset"), chunk == 1);
                } else {
                    ++spatial_history;
                    CHECK_EQ(pint(op, "reset"), chunk == 0);
                }
            }
            if (op->op == "wan_vae_expand" && !pint(op, "temporal")) {
                CHECK_EQ(pint(op, "drop"), chunk == 0 && shortcuts < 2 ? 1 : 0);
                ++shortcuts;
            }
        }
        const int64_t frames = chunk == 0 ? 1 : 4;
        const auto& end = run.issues.back();
        CHECK(b.op_of(end.op)->op == "tile_stitch");
        const auto& cast = run.issues.at(run.issues.size() - 3);
        CHECK(b.op_of(cast.op)->op == "cast");
        CHECK_EQ(cast.opd.front().handle, decoder.output());
        CHECK_EQ(cast.opd.back().handle, end.opd.front().handle);
        CHECK_EQ(end.opd.front().rows, frames * 24 * 64);
        CHECK_EQ(end.opd.back().rows, frames * 4 * 16);
        CHECK_EQ(end.opd.back().cols, 6 * 16 * 3);
        CHECK(end.opd.back().raw == rad_buf_ptr(&run, rgb));
        CHECK_EQ(spatial_history, 30);
        CHECK_EQ(time_history, chunk == 0 ? 0 : 2);
        CHECK_EQ(shortcuts, 3);
        CHECK_LINT(b, &run);
    }
}

RAD_TEST_MAIN()
