/* xstats_test.cpp -- --expert-stats: the expert map across ranks, the per-rank tally and its merge,
 * and the file (place/xstats.h). Pure host code: the engine's drain hands the tally the same rows
 * these cases hand it. */
#include "rad_test.h"

#include "place/xstats.h"

#include "util/json.h"

#include <dirent.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace rad;

namespace {

using Ids = std::vector<std::vector<std::vector<int32_t>>>;   /* [rank][layer][local] */

std::vector<int32_t> iota_from(int32_t base, int32_t n) {
    std::vector<int32_t> v((size_t)n);
    for (int32_t i = 0; i < n; ++i) v[(size_t)i] = base + i;
    return v;
}

/* Credits one model-wide histogram `global` of `layer` at `step` on every rank, each rank's row
 * being its slice in the order its map names the experts, as moe_scatter would write it. */
void credit_all(ExpertStats& xs, const Ids& ids, int layer, int step,
                const std::vector<int32_t>& global, int reports, int top_k) {
    for (size_t r = 0; r < ids.size(); ++r) {
        XsTally* t = xs.tally((int)r);
        if (!t) continue;
        std::vector<int32_t> row;
        for (int32_t g : ids[r][(size_t)layer]) row.push_back(global[(size_t)g]);
        std::lock_guard<std::mutex> lk(t->mutex());
        t->credit(layer, step, row.data(), (int64_t)row.size(), reports, top_k);
    }
}

void issued_all(ExpertStats& xs, int n_rank, int step, int draft_pass, int64_t n_seq,
                int64_t n_seq_decode, int64_t n_tok, int64_t n_tok_decode) {
    for (int r = 0; r < n_rank; ++r)
        if (XsTally* t = xs.tally(r)) t->issued(step, draft_pass, n_seq, n_seq_decode, n_tok, n_tok_decode);
}

uint64_t sum(const std::vector<uint64_t>& v) {
    uint64_t s = 0;
    for (uint64_t x : v) s += x;
    return s;
}

}  /* namespace */

/* Every rank holds every expert (a width slice of each): the container gives each rank the same
 * ids, so rank 0 counts them all and the other ranks count nothing. */
TEST(experts_every_rank_holds_are_counted_on_rank_zero_only) {
    Ids ids(3, { iota_from(0, 4), iota_from(0, 4) });
    XsMap m;
    CHECK_OK(xs_map_build(ids, { 4, 4 }, 2, &m));
    REQUIRE_EQ(m.n_layer, 2);
    CHECK_EQ(m.n_expert[0], 4);
    CHECK_EQ(m.how[0], std::string("container"));
    CHECK(m.gid[0][1] == iota_from(0, 4));
    for (int r = 1; r < 3; ++r)
        for (int32_t g : m.gid[(size_t)r][0]) CHECK_EQ(g, XsMap::kElsewhere);
}

/* Expert parallelism as deepseek41 declares it at TP4: the trunk's 24 experts six a rank, the
 * DSpark head's 12 split 4/4/2/2 (the even floor, two more to the first ranks). Each rank's row is
 * its own slice from zero; the container's ids put each index at its model-wide expert. */
TEST(an_expert_parallel_split_maps_each_ranks_slice_to_the_models_ids) {
    const int32_t head_n[4] = { 4, 4, 2, 2 }, head_base[4] = { 0, 4, 8, 10 };
    Ids ids(4);
    for (int r = 0; r < 4; ++r)
        ids[(size_t)r] = { iota_from(6 * r, 6), iota_from(head_base[r], head_n[r]) };
    XsMap m;
    CHECK_OK(xs_map_build(ids, { 24, 0 }, 1, &m));
    CHECK_EQ(m.n_trunk, 1);
    CHECK_EQ(m.n_expert[0], 24);
    CHECK_EQ(m.n_expert[1], 12);
    for (int r = 0; r < 4; ++r) {
        CHECK(m.gid[(size_t)r][0] == iota_from(6 * r, 6));
        CHECK(m.gid[(size_t)r][1] == iota_from(head_base[r], head_n[r]));
    }

    /* Any assignment the container states is taken as stated, interleaved included. */
    Ids odd = { { { 1, 3, 5 } }, { { 0, 2, 4 } } };
    CHECK_OK(xs_map_build(odd, { 6 }, 1, &m));
    CHECK(m.gid[0][0] == std::vector<int32_t>({ 1, 3, 5 }));
    CHECK(m.gid[1][0] == std::vector<int32_t>({ 0, 2, 4 }));
    CHECK_EQ(m.n_expert[0], 6);
}

/* Without the container's ids (a checkpoint served directly, or ids that cannot be right), the
 * map falls back: the indices themselves at one rank or where every rank holds the model's count,
 * the ranks' rows in rank order otherwise. */
TEST(without_ids_the_map_falls_back_to_the_indices_or_rank_order) {
    XsMap m;
    CHECK_OK(xs_map_build({ { std::vector<int32_t>(5, -1) } }, { 5 }, 1, &m));
    CHECK_EQ(m.how[0], std::string("index"));
    CHECK(m.gid[0][0] == iota_from(0, 5));

    CHECK_OK(xs_map_build({ { std::vector<int32_t>(4, -1) }, { std::vector<int32_t>(4, -1) } }, { 4 }, 1, &m));
    CHECK_EQ(m.how[0], std::string("replicated"));
    CHECK(m.gid[0][0] == iota_from(0, 4));
    CHECK(m.gid[1][0] == std::vector<int32_t>(4, XsMap::kElsewhere));

    CHECK_OK(xs_map_build({ { std::vector<int32_t>(3, -1) }, { std::vector<int32_t>(3, -1) } }, { 6 }, 1, &m));
    CHECK_EQ(m.how[0], std::string("rank order"));
    CHECK(m.gid[1][0] == iota_from(3, 3));
    CHECK_EQ(m.n_expert[0], 6);

    /* One id given twice within a rank is not an assignment: the layer falls back. */
    CHECK_OK(xs_map_build({ { { 0, 1 } }, { { 2, 2 } } }, { 4 }, 1, &m));
    CHECK_EQ(m.how[0], std::string("rank order"));
    CHECK(m.gid[1][0] == iota_from(2, 2));
}

/* The tally files each histogram under its step's kind, counts the head's under draft, and merges
 * the ranks' slices into model-wide rows whose sums are the tokens times top_k. */
TEST(the_tally_files_by_kind_and_merges_the_ranks_slices) {
    /* Two ranks: layer 0 a trunk layer of 4 experts (2 a rank), layer 1 a head of 2 (1 a rank). */
    const Ids ids = { { { 0, 1 }, { 0 } }, { { 2, 3 }, { 1 } } };
    XsMap m;
    CHECK_OK(xs_map_build(ids, { 4, 0 }, 1, &m));
    ExpertStats xs;
    const std::string path = radtest::tmp_path("merge.json");
    XsSnapshot info;
    info.model = "tiny";
    CHECK_OK(xs.init(path, m, info));
    REQUIRE(xs.tally(0) && xs.tally(1));

    /* Step 0: a prefill of 3 tokens, top-2. */
    issued_all(xs, 2, 0, 0, 1, 0, 3, 0);
    credit_all(xs, ids, 0, 0, { 3, 1, 2, 0 }, 1, 2);
    /* Step 1: decode of 2 sequences, then two draft passes; the bank holds the last's 2 rows. */
    issued_all(xs, 2, 1, 0, 2, 2, 2, 2);
    issued_all(xs, 2, 1, -1, 2, 2, 2, 0);
    issued_all(xs, 2, 1, 2, 2, 2, 2, 0);
    credit_all(xs, ids, 0, 1, { 0, 2, 1, 1 }, 1, 2);
    credit_all(xs, ids, 1, 1, { 3, 1 }, 2, 2);
    /* Step 2: mixed, 1 decode row and a 4-token chunk. */
    issued_all(xs, 2, 2, 0, 2, 1, 5, 1);
    credit_all(xs, ids, 0, 2, { 4, 2, 3, 1 }, 1, 2);
    /* A histogram of a step nothing filed (it cannot be classified): mixed, and counted apart. */
    credit_all(xs, ids, 0, 7, { 1, 1, 0, 0 }, 1, 2);

    const XsSnapshot s = xs.snapshot();
    CHECK(s.counts[kXsPrefill][0] == std::vector<uint64_t>({ 3, 1, 2, 0 }));
    CHECK(s.counts[kXsDecode][0] == std::vector<uint64_t>({ 0, 2, 1, 1 }));
    CHECK(s.counts[kXsMixed][0] == std::vector<uint64_t>({ 5, 3, 3, 1 }));
    CHECK(s.counts[kXsDraft][1] == std::vector<uint64_t>({ 3, 1 }));
    CHECK(s.counts[kXsDraft][0] == std::vector<uint64_t>({ 0, 0, 0, 0 }));
    CHECK(s.counts[kXsPrefill][1] == std::vector<uint64_t>({ 0, 0 }));

    /* Selections are the tokens of the passes the banks hold times top_k. */
    for (int k : { kXsPrefill, kXsDecode })
        CHECK_EQ(s.selections[k][0], s.obs_tokens[k][0] * 2);
    CHECK_EQ(s.obs_tokens[kXsPrefill][0], (uint64_t)3);
    CHECK_EQ(s.obs_tokens[kXsDecode][0], (uint64_t)2);
    CHECK_EQ(s.obs_tokens[kXsMixed][0], (uint64_t)5);    /* the unclassified one adds no tokens */
    CHECK_EQ(s.selections[kXsMixed][0], (uint64_t)12);
    CHECK_EQ(s.obs_tokens[kXsDraft][1], (uint64_t)2);
    CHECK_EQ(s.selections[kXsDraft][1], (uint64_t)4);
    CHECK_EQ(s.passes[kXsDraft][1], (uint64_t)2);
    CHECK_EQ(s.histograms[kXsMixed][0], (uint64_t)2);

    /* Issued, once (not once a rank). */
    CHECK_EQ(s.steps[kXsPrefill], (uint64_t)1);
    CHECK_EQ(s.tokens[kXsPrefill], (uint64_t)3);
    CHECK_EQ(s.steps[kXsDecode], (uint64_t)1);
    CHECK_EQ(s.tokens[kXsDecode], (uint64_t)2);
    CHECK_EQ(s.tokens[kXsMixed], (uint64_t)5);
    CHECK_EQ(s.mixed_decode_tokens, (uint64_t)1);
    CHECK_EQ(s.draft_passes, (uint64_t)2);
    CHECK_EQ(s.draft_rows, (uint64_t)4);
    CHECK_EQ(s.top_k[0], 2);
    CHECK_EQ(s.rank_unclassified[0], (uint64_t)1);
    CHECK_EQ(s.rank_unclassified[1], (uint64_t)1);
    CHECK_EQ(sum(s.rank_unmapped), (uint64_t)0);
    CHECK_EQ(s.rank_credited[0], (uint64_t)5);
}

/* A replicated layer is drained on rank 0 only: the other ranks get no tally at all. And counts
 * past a rank's declared experts, or where the map has no id, are unmapped, not dropped silently. */
TEST(ranks_that_count_nothing_have_no_tally_and_strays_are_unmapped) {
    XsMap m;
    CHECK_OK(xs_map_build({ { iota_from(0, 3) }, { iota_from(0, 3) } }, { 3 }, 1, &m));
    ExpertStats xs;
    CHECK_OK(xs.init(radtest::tmp_path("repl.json"), m, XsSnapshot{}));
    CHECK(xs.tally(0) != nullptr);
    CHECK(xs.tally(1) == nullptr);

    XsTally* t = xs.tally(0);
    t->issued(4, 0, 1, 1, 1, 1);
    const int32_t row[5] = { 1, 0, 1, 7, -3 };   /* two past the declared three */
    {
        std::lock_guard<std::mutex> lk(t->mutex());
        t->credit(0, 4, row, 5, 1, 2);
    }
    const XsSnapshot s = xs.snapshot();
    CHECK(s.counts[kXsDecode][0] == std::vector<uint64_t>({ 1, 0, 1 }));
    CHECK_EQ(s.rank_unmapped[0], (uint64_t)7);
    CHECK_EQ(s.selections[kXsDecode][0], (uint64_t)2);
}

/* The file: JSON with the shape the report reads, there from boot, replaced whole through a
 * rename (no temporary left behind), and rewritten by the writer thread. */
TEST(the_file_is_json_written_at_boot_and_replaced_through_a_rename) {
    const Ids ids = { { { 0, 1 } }, { { 2, 3 } } };
    XsMap m;
    CHECK_OK(xs_map_build(ids, { 4 }, 1, &m));
    const std::string path = radtest::tmp_path("stats.json");
    XsSnapshot info;
    info.model = "a \"quoted\" name";
    info.arch = "deepseek41";
    ExpertStats xs;
    CHECK_OK(xs.init(path, m, info, 0.05));

    Json j = Json::parse(read_file(path), nullptr, false);
    REQUIRE(!j.is_discarded());
    CHECK_EQ(j["format"].get<std::string>(), std::string("radiance-expert-stats/1"));
    CHECK_EQ(j["model"].get<std::string>(), info.model);
    CHECK_EQ(j["final"].get<bool>(), false);
    CHECK_EQ(j["n_layers"].get<int>(), 1);
    CHECK_EQ(j["n_experts"].get<int>(), 4);
    CHECK_EQ(j["tp"].get<int>(), 2);
    REQUIRE(j["counts"]["decode"].size() == 1);
    CHECK_EQ(j["counts"]["decode"][0].size(), (size_t)4);

    issued_all(xs, 2, 0, 0, 1, 1, 1, 1);
    credit_all(xs, ids, 0, 0, { 0, 1, 1, 0 }, 1, 2);
    xs.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    xs.stop();
    j = Json::parse(read_file(path), nullptr, false);
    REQUIRE(!j.is_discarded());
    CHECK(j["counts"]["decode"][0] == Json({ 0, 1, 1, 0 }));
    CHECK_EQ(j["selections"]["decode"][0].get<uint64_t>(), (uint64_t)2);
    CHECK_EQ(j["observed_tokens"]["decode"][0].get<uint64_t>(), (uint64_t)1);
    CHECK_EQ(j["tokens"]["decode"].get<uint64_t>(), (uint64_t)1);
    CHECK(j["uptime_s"].get<double>() > 0.0);

    CHECK_OK(xs.write(true));
    j = Json::parse(read_file(path), nullptr, false);
    REQUIRE(!j.is_discarded());
    CHECK_EQ(j["final"].get<bool>(), true);

    /* Only the file itself in the directory: every temporary was renamed or removed. */
    int others = 0;
    if (DIR* d = opendir(radtest::tmpdir().c_str())) {
        while (dirent* e = readdir(d)) {
            const std::string n = e->d_name;
            if (n.find("stats.json.tmp") != std::string::npos) ++others;
        }
        closedir(d);
    }
    CHECK_EQ(others, 0);

    /* A path that cannot be written fails init. */
    ExpertStats bad;
    CHECK(bad.init(radtest::tmp_path("no/such/dir/x.json"), m, XsSnapshot{}) < 0);
}

/* The profile file is local data with a contract, and its tmp name is predictable. These pin
 * the loader's refusals -- a negative count (which read unsigned would wrap into a share near
 * 100%), an n_trunk_layers that truncates through int32 into exactly the "another model" match -- and
 * that the tmp name is created exclusively: a symlink planted there must survive untouched. */
TEST(the_profile_file_and_its_tmp_name_keep_their_contracts) {
    auto profile = [](const std::string& body) {
        const std::string p = radtest::tmp_path("xs_profile.json");
        if (FILE* f = std::fopen(p.c_str(), "wb")) {
            std::fwrite(body.data(), 1, body.size(), f);
            std::fclose(f);
        }
        return p;
    };

    const std::string ok = "{\"format\":\"radiance-expert-stats/1\",\"n_trunk_layers\":2,"
                           "\"counts\":{\"decode\":[[1,2],[0,0],[3,0]]}}";
    {
        std::vector<RadFileProfile> out;
        int32_t nt = -1;
        CHECK_EQ(xs_profile_load(profile(ok), &out, &nt), RAD_OK);
        CHECK_EQ(nt, 2);
        CHECK_EQ(out.size(), 3u);   /* the zero-sum row contributes nothing */
    }
    {   /* 2^32 + 2: read through int32 this lands on 2, and 2 layers is what the refusal
         * against would compare. */
        std::vector<RadFileProfile> out;
        int32_t nt = -1;
        CHECK(xs_profile_load(profile("{\"format\":\"radiance-expert-stats/1\","
                                      "\"n_trunk_layers\":4294967298,"
                                      "\"counts\":{\"decode\":[[1,2]]}}"),
                              &out, &nt) < 0);
    }
    {   std::vector<RadFileProfile> out;
        CHECK(xs_profile_load(profile("{\"format\":\"radiance-expert-stats/1\","
                                      "\"counts\":{\"decode\":[[-1,2]]}}"),
                              &out, nullptr) < 0);
    }

    /* The tmp name is the destination's plus this process's PID -- public. Planted as a
     * symlink, it must be cleared and re-created, never written through. */
    const std::string victim = radtest::tmp_path("xs_victim.txt");
    if (FILE* v = std::fopen(victim.c_str(), "w")) { std::fputs("keep", v); std::fclose(v); }
    const std::string dest = radtest::tmp_path("xs_stats.json");
    const std::string tmp = dest + ".tmp." + std::to_string((long long)getpid());
    ::unlink(tmp.c_str());
    if (::symlink(victim.c_str(), tmp.c_str()) != 0) SKIP("cannot plant a symlink here");
    CHECK_EQ(xs_write_atomic(dest, "{\"format\":\"radiance-expert-stats/1\"}\n"), RAD_OK);
    std::string kept;
    if (FILE* v = std::fopen(victim.c_str(), "r")) {
        char buf[16];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, v)) > 0) kept.append(buf, n);
        std::fclose(v);
    }
    CHECK_EQ(kept, std::string("keep"));
    CHECK(::access(tmp.c_str(), F_OK) != 0);   /* renamed away, not left behind */
}

RAD_TEST_MAIN()
