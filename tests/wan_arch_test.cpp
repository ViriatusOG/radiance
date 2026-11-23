#include "arch_fakes.h"
#define RAD_ARCH_NO_EXPORTS
#include "../arch/wan/wan.cpp"
static RadVideoGenDecl declared_wan{};
extern "C" int rad_declare_video_generator(RadBuilder*, const RadVideoGenDecl* decl) {
    declared_wan = *decl;
    return RAD_OK;
}
TEST(wan_video_passes_use_both_guidance_branches_and_restore_the_first_frame) {
    using namespace rad::arch;
    RadBuilder b{};
    MetaKV fields(RadModelMeta{});
    std::string mean, stddev;
    for (int i = 0; i < 48; ++i) {
        mean += i ? " 0" : "0";
        stddev += i ? " 1" : "1";
    }
    fields.set("transformer.num_layers", "30")
          .set("transformer.num_attention_heads", "24")
          .set("transformer.attention_head_dim", "128")
          .set("transformer.in_channels", "48")
          .set("transformer.out_channels", "48")
          .set("vae.z_dim", "48")
          .set("vae.base_dim", "160")
          .set("vae.decoder_base_dim", "256")
          .set("vae.patch_size", "2")
          .set("vae.latents_mean", mean.c_str())
          .set("vae.latents_std", stddev.c_str());
    const auto* meta = fields.meta();
    RadArchProbe probe{};
    REQUIRE_EQ(wan::probe(meta, &probe), RAD_OK);
    auto build = build_ctx(0, 1);
    build.max_tok = probe.gen_rows;
    REQUIRE_EQ(wan::declare(&b, meta, &build), RAD_OK);
    CHECK_EQ(declared_wan.frames_tail, -3);
    CHECK_EQ(declared_wan.sched_kind, RAD_VIDEO_SCHEDULE_UNIPC);
    CHECK_EQ(declared_wan.guidance, RAD_VIDEO_GUIDANCE_CFG);
    const auto* velocities = find_b(b, "dit.velocities");
    REQUIRE(velocities);
    CHECK_EQ(velocities->d.kind, RAD_BUF_PERSIST);
    for (const auto& weight : b.weights) {
        if (weight.name.rfind("umt5.block.", 0) != 0) continue;
        CHECK_EQ(weight.d.access, RAD_ACCESS_PER_REQUEST);
        CHECK(weight.d.group.layer > 31);
    }

    float schedule[16]{}, guidance[8]{5.0f};
    uint32_t seed[4]{};
    int32_t geom[RAD_VGEOM_N]{}, span[4]{0, 3, 0, 0}, lane = 0, first = 1;
    RadVideoGenBatch v{};
    v.geom.lat_frames = 3;
    v.geom.lat_h = v.geom.lat_w = 2;
    v.geom_d = geom;
    v.lat_cu = span;
    v.lane_d = &lane;
    v.first_d = &first;
    v.seed = seed;
    v.sched = schedule;
    v.guidance_h = v.guidance = guidance;
    v.has_image = 1;
    RadBatch batch{};
    batch.gen = RAD_GEN_PASS_VSTEP;
    batch.vgen_b = &v;
    RadCtx run{};
    run.b = &b;
    run.batch = &batch;
    wan::step(&run, &batch);
    REQUIRE(run.step_fail.empty());

    int unpatches = 0;
    for (const auto& issue : run.issues)
        if (b.op_of(issue.op)->op == "wan_unpatch") ++unpatches;
    CHECK_EQ(unpatches, 2);
    REQUIRE(run.issues.size() > 2u);
    const auto& sampler = run.issues.at(run.issues.size() - 2);
    CHECK(b.op_of(sampler.op)->op == "flow_unipc_step");
    CHECK_EQ(sampler.n, 12);
    CHECK_EQ(sampler.opd[0].handle, sampler.opd[4].handle);
    CHECK_EQ(sampler.opd[4].offset, wan::kMaxVoxels * wan::kChannels);
    const auto& pin = run.issues.back();
    CHECK(b.op_of(pin.op)->op == "cast");
    CHECK_EQ(pin.n, 4);
    CHECK_EQ(pin.opd.back().handle, sampler.opd[3].handle);
    CHECK_LINT(b, &run);
}

RAD_TEST_MAIN()
