#include "arch_fakes.h"
#include "../arch/wan/wan_umt5.h"
#include "../arch/wan/wan_conditioning.h"

TEST(wan_umt5_declares_each_checkpoint_tensor_and_runs_packed_conditioning) {
    RadBuilder builder{};
    rad::arch::Names names("wan");
    wan::Umt5Encoder encoder;
    REQUIRE_EQ(encoder.declare(&builder, names), RAD_OK);
    const auto* token_weight = find_w(builder, "wan.umt5.token.weight");
    REQUIRE(token_weight);
    CHECK_EQ(token_weight->d.shape[0], 256384);
    CHECK_EQ(token_weight->d.shape[1], 4096);
    const auto* output = find_b(builder, "wan.umt5.encoded");
    REQUIRE(output);
    CHECK_EQ(output->d.shape[0], 1024);
    CHECK_EQ(output->d.shape[1], 4096);

    std::set<std::string> sources;
    size_t source_count = 0;
    for (const auto& map : builder.maps) {
        source_count += map.src.size();
        sources.insert(map.src.begin(), map.src.end());
    }
    CHECK_EQ(source_count, 242u);
    CHECK_EQ(sources.size(), source_count);
    const auto* token_map = find_map(builder, "wan.umt5.token.weight");
    REQUIRE(token_map);
    CHECK(token_map->src == std::vector<std::string>{"shared.weight"});
    const auto* qkv = find_map(builder, "wan.umt5.block.0.qkv.weight");
    const auto* ffn = find_map(builder, "wan.umt5.block.23.gate_up.weight");
    REQUIRE(qkv && ffn);
    CHECK(qkv->src == (std::vector<std::string>{
        "encoder.block.0.layer.0.SelfAttention.q.weight",
        "encoder.block.0.layer.0.SelfAttention.k.weight",
        "encoder.block.0.layer.0.SelfAttention.v.weight"}));
    CHECK(ffn->src == (std::vector<std::string>{
        "encoder.block.23.layer.1.DenseReluDense.wi_0.weight",
        "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"}));

    const int32_t tokens[] = {20, 21, 1, 57, 1};
    const int32_t boundaries[] = {0, 3, 5};
    RadBatch batch{};
    batch.n_tok = 5;
    batch.n_seq = 2;
    batch.token_ids = tokens;
    batch.cu_seqlens = boundaries;
    RadCtx run{};
    run.b = &builder;
    run.batch = &batch;
    encoder.step(&run, batch);
    REQUIRE(run.step_fail.empty());

    int layer = 0;
    for (const auto& issue : run.issues) {
        const auto& op = builder.ops.at(issue.op - 1);
        CHECK_EQ(issue.n, batch.n_tok);
        if (op.op != "attn_umt5") continue;
        REQUIRE_EQ(issue.opd.size(), 6u);
        CHECK_EQ(pint(&op, "heads"), 64);
        for (int i = 0; i < 3; ++i) {
            CHECK_EQ(issue.opd[i].handle, issue.opd[0].handle);
            CHECK_EQ(issue.opd[i].offset, i * 4096);
            CHECK_EQ(issue.opd[i].cols, 4096);
        }
        CHECK(issue.opd[3].raw == boundaries);
        CHECK_EQ(issue.opd[3].rows, 3);
        const auto& bias = builder.weights.at(issue.opd[4].handle - 1);
        CHECK_EQ(bias.d.group.layer, wan::Umt5Encoder::first_weight_group + layer++);
        CHECK_EQ(bias.d.shape[0], 32);
        CHECK_EQ(bias.d.shape[1], 64);
    }
    CHECK_EQ(layer, 24);
    REQUIRE(!run.issues.empty());
    CHECK_EQ(run.issues.back().opd.back().handle, encoder.encoded);
    CHECK_LINT(builder, &run);
}

TEST(wan_umt5_rejects_geometry_outside_one_or_two_tokenized_prompts) {
    RadCtx run{};
    RadBatch batch{};
    batch.n_tok = 513;
    batch.n_seq = 1;
    const int32_t token = 1;
    const int32_t boundaries[] = {0, 513};
    batch.token_ids = &token;
    batch.cu_seqlens = boundaries;
    wan::Umt5Encoder{}.step(&run, batch);
    CHECK(!run.step_fail.empty());
    CHECK(run.issues.empty());
}

TEST(wan_text_conditioning_projects_padded_prompts_into_the_right_video_lane) {
    RadBuilder builder{};
    rad::arch::Names names("wan");
    wan::TextConditioning text;
    REQUIRE_EQ(text.declare(&builder, names, 2), RAD_OK);
    const auto* w = find_w(builder, "wan.text.proj1.weight");
    REQUIRE(w);
    CHECK_EQ(w->d.shape[0], 3072);
    CHECK_EQ(w->d.shape[1], 4096);
    CHECK_EQ(w->d.dtype, (uint32_t)RAD_BF16);
    const auto* persistent = find_b(builder, "wan.text.projected");
    REQUIRE(persistent);
    CHECK_EQ(persistent->d.kind, RAD_BUF_PERSIST);
    CHECK_EQ(persistent->d.shape[0], 2048);
    CHECK_EQ(persistent->d.shape[1], 3072);

    const int32_t tokens[] = {20, 21, 1, 57, 1}, boundaries[] = {0, 3, 5};
    RadVideoGenBatch video{};
    video.lane = 1;
    video.n_neg = 2;
    RadBatch batch{};
    batch.n_tok = 5;
    batch.n_seq = 2;
    batch.token_ids = tokens;
    batch.cu_seqlens = boundaries;
    batch.vgen_b = &video;
    RadCtx run{};
    run.b = &builder;
    run.batch = &batch;
    text.step(&run, batch);
    REQUIRE(run.step_fail.empty());
    REQUIRE(run.issues.size() >= 5);
    const size_t end = run.issues.size();
    const auto& fill = run.issues[end - 5];
    const auto& positive = run.issues[end - 4];
    const auto& negative = run.issues[end - 3];
    const auto& proj1 = run.issues[end - 2];
    const auto& proj2 = run.issues[end - 1];
    CHECK(builder.ops.at(fill.op - 1).op == "fill");
    CHECK_EQ(fill.n, 1024);
    CHECK_EQ(positive.n, 3);
    CHECK_EQ(positive.opd.at(0).offset, 0);
    CHECK_EQ(positive.opd.at(1).offset, 0);
    CHECK_EQ(negative.n, 2);
    CHECK_EQ(negative.opd.at(0).offset, 3 * 4096);
    CHECK_EQ(negative.opd.at(1).offset, 512 * 4096);
    CHECK_EQ(pstr(&builder.ops.at(proj1.op - 1), "act"), std::string("gelu_tanh"));
    CHECK_EQ(proj1.n, 1024);
    CHECK_EQ(proj2.n, 1024);
    CHECK_EQ(proj2.opd.back().handle, text.buffer());
    CHECK_EQ(proj2.opd.back().offset, text.context(1, 0).offset);
    CHECK_EQ(text.context(1, 1).offset, 1536 * 3072);
    CHECK_LINT(builder, &run);

    video.n_neg = 0;
    batch.n_tok = 3;
    batch.n_seq = 1;
    RadCtx single{};
    single.b = &builder;
    single.batch = &batch;
    text.step(&single, batch);
    REQUIRE(single.step_fail.empty());
    CHECK_EQ(single.issues.size() + 1, run.issues.size());
    CHECK_EQ(single.issues.back().opd.back().rows, 512);
    CHECK_LINT(builder, &single);

    video.lane = 2;
    RadCtx invalid{};
    invalid.b = &builder;
    text.step(&invalid, batch);
    CHECK(!invalid.step_fail.empty());
    CHECK(invalid.issues.empty());
}

RAD_TEST_MAIN()
