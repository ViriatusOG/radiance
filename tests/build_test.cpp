/* build_test.cpp -- the declare phase.
 *
 * Runs against a real Registry loading the real test plugin; a fake selector would be a second
 * implementation of the bucket table. core/plugin/testplugin/ carries spec §2.2's worked example
 * (t_gemm with rows at M<=16, M<=64 and unconstrained), a range with a hole, an op with no host
 * kernel, and an init() that refuses.
 *
 * Under test: the five products of declare (spec §3.1), and that the phase always completes and
 * decides everything once.
 */
#include "rad_test.h"
#include "rad_planes.h"

#include <nlohmann/json.hpp>

#include "build/rad_build.h"
#include "build/startup.h"
#include "rad_internal.h"

#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

using namespace rad;
using json = nlohmann::ordered_json;

namespace {

std::string kernels_dir() { return home() + "/testkernels"; }

int load(Registry& r, std::vector<std::string> hierarchy = { "radtest_over", "radtest_base" }) {
    const int st = r.load_dir(kernels_dir(), hierarchy);
    if (st == RAD_E_IO)
        fprintf(stderr, "    (no test plugins in %s -- build the radtest_* targets)\n",
                kernels_dir().c_str());
    return st;
}

/* t_attn as a paged-attention caller declares it: q_len 1..16, 8 heads of 128, 16-token blocks. */
rad_op decl_t_attn(Builder& b) {
    const RadParam p[] = { RAD_RANGE("q_len", 1, 16), RAD_INT("head_dim", 128),
                           RAD_INT("window", 0), RAD_INT("block_size", 16),
                           RAD_INT("causal", 1), RAD_INT("n_head", 8),
                           RAD_STR("kv_dtype", "bf16") };
    return b.decl_op("t_attn", p, 7, nullptr, 0);
}

/* A bf16 full-attention KV group, "kv_full", of 8 heads of 128. */
rad_kvgroup decl_kv_full(Builder& b) {
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_FULL;
    kd.dtype = RAD_BF16;
    kd.n_head_kv = 8;
    kd.head_dim = 128;
    return b.decl_kv_group("kv_full", &kd);
}

RadModelMeta make_meta() {
    RadModelMeta m{};
    m.arch_id  = "test_arch";
    m.name     = "test_model";
    m.quant    = "bf16";
    m.n_layers = 2;
    m.n_embd   = 4096;
    return m;
}

RadBuildCtx make_ctx() {
    RadBuildCtx c{};
    c.rank = 0;
    c.world_size = 1;
    c.max_tok  = 8192;
    c.max_seqs = 256;
    c.max_ctx  = 4096;
    c.scope = "";
    return c;
}

/* t_gemm's operands are a(in), b(weight), y(out), so one weight handle. */
rad_weight decl_w(Builder& b, const char* name, int64_t d0 = 4096, int64_t d1 = 4096) {
    RadWeightDecl d{};
    d.dtype = RAD_BF16;
    d.rank  = 2;
    d.shape[0] = d0;
    d.shape[1] = d1;
    d.access = RAD_ACCESS_PER_TOKEN;
    d.shard  = RAD_SHARD_ROW;
    d.group  = rad_group_layer(0);
    return b.decl_weight(name, &d);
}

/* The §2.2 worked example, as an architecture plugin would declare it. */
rad_op decl_gemm(Builder& b, rad_weight w, int64_t max_m = 8192) {
    const RadParam p[] = { RAD_RANGE("M", 1, max_m), RAD_INT("N", 4096), RAD_INT("K", 4096),
                           RAD_STR("dtype", "bf16") };
    return b.decl_op("t_gemm", p, 4, &w, 1);
}

BufferInfo make_buf(const char* name, int64_t bytes, int kind, int32_t lo, int32_t hi) {
    BufferInfo b;
    b.name = name;
    b.decl.dtype = RAD_I8;
    b.decl.rank  = 1;
    b.decl.shape[0] = bytes;
    b.decl.kind = kind;
    b.decl.domain = RAD_DOMAIN_DEVICE;
    b.bytes = bytes;
    b.first_def = lo;
    b.last_use  = hi;
    return b;
}

}  /* namespace */

/* Leaves a case when the radtest_* plugins do not load; otherwise the assertions would walk
 * handles the builder never issued and crash without a reason. This build makes them, so their
 * absence is a missing in-tree artifact (rad_test.h). */
#define NEED_PLUGIN(reg)                                                              \
    Registry reg;                                                                     \
    if (load(reg) < 0) ARTIFACT_MISSING("the test plugins in " + kernels_dir());

/* ================================================================== the bucket table */
TEST(bucket_table_resolves_per_band) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    rad_weight w = decl_w(b, "blk.0.ffn_gate.weight");
    CHECK(w == 1);
    rad_op h = decl_gemm(b, w);

    CHECK(h == 1);
    CHECK_EQ(b.op_resolved(h), 1);
    CHECK_EQ((int)b.errors().size(), 0);

    const OpInfo* o = b.program().op(h);
    CHECK_EQ((int)o->bands.size(), 3);
    if (o->bands.size() != 3) return;
    CHECK_EQ(o->bands[0].hi, 16);
    CHECK_EQ(o->bands[1].hi, 64);
    CHECK_EQ(o->bands[2].hi, 8192);

    /* The band picks the kernel, not the range: resolving once at max_tok would pick the
     * unconstrained prefill row and run it at M = 1, the loss §2.2 avoids. */
    CHECK_EQ(std::string(o->bands[0].dom[RAD_DOMAIN_DEVICE].row->info->name), "t_gemm_m16");
    CHECK_EQ(std::string(o->bands[1].dom[RAD_DOMAIN_DEVICE].row->info->name), "t_gemm_m64");
    CHECK_EQ(std::string(o->bands[2].dom[RAD_DOMAIN_DEVICE].row->info->name), "t_gemm_any");

    /* Both domains are resolved where both exist, so the execution site can change at runtime
     * without re-resolving (spec §2.1). */
    for (const Band& band : o->bands)
        CHECK_EQ(std::string(band.dom[RAD_DOMAIN_HOST].row->info->name), "t_gemm_host");

    /* Each band is frozen at its upper bound, its worst case. */
    long long m = 0;
    CHECK(o->bands[1].dom[RAD_DOMAIN_DEVICE].geom.get_i("M", &m));
    CHECK_EQ(m, 64);

    CHECK_EQ((int)b.program().misses.size(), 0);
    CHECK_OK(b.finish());

    /* Recorded weight use makes layer-offload prefetch exact rather than predicted. */
    CHECK_EQ(b.program().weight(w)->first_use_op, 1);
    CHECK_EQ(b.program().weight(w)->last_use_op, 1);
}

TEST(an_op_with_no_ranged_parameter_is_one_band) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    rad_weight w = decl_w(b, "w");
    const RadParam p[] = { RAD_INT("M", 8), RAD_INT("n", 4096) };
    rad_op h = b.decl_op("t_tie", p, 2, &w, 1);
    CHECK(h == 1);

    /* build_bands returns one band with hi = INT64_MAX, so the issue path's band scan needs no
     * special case. */
    const OpInfo* o = b.program().op(h);
    CHECK_EQ((int)o->bands.size(), 1);
    CHECK_EQ(o->bands[0].hi, INT64_MAX);
    CHECK(o->ranged_key.empty());
    /* Ties break on declaration order. */
    CHECK_EQ(std::string(o->bands[0].dom[RAD_DOMAIN_DEVICE].row->info->name), "t_tie_first");
}

/* ================================================================== the graph dump, exactly */
TEST(graph_dump_matches_the_spec) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    rad_weight w = decl_w(b, "blk.12.ffn_gate.weight");
    decl_gemm(b, w);
    CHECK_OK(b.finish());

    const std::string s = dump_graph_str(b.program(), &b.errors());

    /* The bucket-table block, byte for byte spec §2.2's format: the op, its fixed parameters,
     * then one line per band naming kernel, plugin and priority. If the format drifts, the spec
     * is right.
     *
     * Only t_gemm_m64 declares a `bn` axis, so only its line carries §15's tuning clause. */
    const char* expect =
        "t_gemm  N=4096 K=4096 dtype=bf16\n"
        "    M <=   16  ->  t_gemm_m16              (radtest_base, prio 30)\n"
        "    M <=   64  ->  t_gemm_m64              (radtest_base, prio 20, bn=64 from axis defaults)\n"
        "    M <= 8192  ->  t_gemm_any              (radtest_base, prio 10)\n";
    if (!has(s, expect)) {
        fprintf(stderr, "---- dump was ----\n%s------------------\n", s.c_str());
        CHECK(has(s, expect));
    }

    /* The op has a host kernel, so a labelled host table follows; the device table is unlabelled,
     * matching §2.2. */
    CHECK(has(s, "    host domain:\n"));
    CHECK(has(s, "    M <= 8192  ->  t_gemm_host             (radtest_base, prio 10)\n"));

    /* Matched constraints, touched weights, and the manifest with tier and site left for the
     * planner. */
    CHECK(has(s, "      matched  t_gemm_m16: M <= 16, K % 256 == 0, dtype in {bf16 w4a8}\n"));
    CHECK(has(s, "      matched  t_gemm_any: (unconstrained"));
    CHECK(has(s, "      weights  blk.12.ffn_gate.weight\n"));
    CHECK(has(s, "tier"));
    CHECK(has(s, "---- tuning requests: 1"));
    CHECK(!has(s, "---- unresolved"));
}

/* ================================================================== the buffer plan */
TEST(liveness_packing) {
    Program p;
    p.buffers.emplace_back();        /* the sentinel */
    p.ops.resize(6);                 /* five ops plus the sentinel */

    /* a and b are never live together, so they share bytes; c and d overlap at ops 2 and 3, so
     * they do not. */
    p.buffers.push_back(make_buf("a", 1024, RAD_BUF_TRANSIENT, 1, 2));
    p.buffers.push_back(make_buf("b", 1024, RAD_BUF_TRANSIENT, 3, 4));
    p.buffers.push_back(make_buf("c", 2048, RAD_BUF_TRANSIENT, 1, 4));
    p.buffers.push_back(make_buf("d", 2048, RAD_BUF_TRANSIENT, 2, 3));
    p.buffers.push_back(make_buf("keep", 512, RAD_BUF_PERSIST, -1, -1));

    BufPlanReport rep;
    CHECK_OK(plan_buffers(p, &rep));

    const BufferInfo& a = p.buffers[1];
    const BufferInfo& bb = p.buffers[2];
    const BufferInfo& c = p.buffers[3];
    const BufferInfo& d = p.buffers[4];
    const BufferInfo& k = p.buffers[5];

    CHECK_EQ(a.arena_offset, bb.arena_offset);
    CHECK(c.arena_offset != d.arena_offset);
    CHECK(c.arena_offset + c.bytes <= d.arena_offset || d.arena_offset + d.bytes <= c.arena_offset);

    /* The sum is the cost without sharing and the peak a perfect packer's; the arena lies
     * between. */
    CHECK_EQ(rep.sum_bytes, 6144);
    CHECK_EQ(rep.peak_live, 5120);
    CHECK_EQ(rep.transient_bytes, 5120);

    /* A persist buffer survives across steps, so it gets its own region at the bottom, where no
     * smaller step's plan moves it; transients go above. */
    CHECK_EQ(k.arena_offset, 0);
    CHECK_EQ(p.arena_fixed_bytes, 512);
    for (const BufferInfo* t : { &a, &bb, &c, &d }) CHECK(t->arena_offset >= p.arena_fixed_bytes);
    CHECK_EQ(p.arena_bytes, 5632);
    CHECK_EQ(rep.persist_bytes, 512);
    CHECK_EQ(rep.n_unknown_live, 0);
}

/* A smaller step's plan: the same buffers with rows scaled, the fixed region untouched and the
 * transients packed lower, so a step that small never touches the arena past `end`. */
TEST(a_smaller_step_packs_its_transients_lower_and_keeps_the_fixed_region) {
    auto mk = [](int64_t rows) {
        Program p;
        p.buffers.emplace_back();
        p.ops.resize(6);
        auto rowbuf = [&](const char* n, int64_t w, int32_t lo, int32_t hi) {
            BufferInfo b = make_buf(n, rows * w, RAD_BUF_TRANSIENT, lo, hi);
            b.decl.rank = 2;
            b.decl.shape[0] = rows;
            b.decl.shape[1] = w;
            return b;
        };
        p.buffers.push_back(rowbuf("x", 256, 1, 2));
        p.buffers.push_back(rowbuf("y", 512, 2, 4));
        p.buffers.push_back(rowbuf("z", 256, 3, 5));
        p.buffers.push_back(make_buf("state", 4096, RAD_BUF_PERSIST, -1, -1));
        return p;
    };
    Program full = mk(2048);
    CHECK_OK(plan_buffers(full));
    Program probe = mk(256);
    ArenaLevel lv;
    std::string why;
    CHECK_OK(plan_level(full, probe, 256, &lv, &why));
    CHECK_EQ(lv.rows, 256);
    CHECK_EQ(lv.offset[4], full.buffers[4].arena_offset);      /* the fixed region stays */
    for (int i = 1; i <= 3; ++i) {
        CHECK(lv.offset[(size_t)i] >= full.arena_fixed_bytes);
        CHECK(lv.offset[(size_t)i] + buffer_bytes(lv.decl[(size_t)i]) <= lv.end);
        CHECK_EQ(lv.decl[(size_t)i].shape[0], 256);
    }
    CHECK(lv.end < full.arena_bytes);
    CHECK(lv.end - full.arena_fixed_bytes <= (full.arena_bytes - full.arena_fixed_bytes) / 4);
    /* Simultaneously live transients never share bytes at the smaller size either. */
    CHECK(lv.offset[2] + buffer_bytes(lv.decl[2]) <= lv.offset[3] ||
          lv.offset[3] + buffer_bytes(lv.decl[3]) <= lv.offset[2]);

    /* Only the rows may shrink; a narrower second extent changes the stride. */
    Program wide = mk(256);
    wide.buffers[1].decl.shape[1] = 128;
    CHECK_EQ(plan_level(full, wide, 256, &lv, &why), RAD_E_STATE);
    CHECK(why.find("extent") != std::string::npos);
    /* A different buffer list is a different graph. */
    Program other = mk(256);
    other.buffers.pop_back();
    CHECK_EQ(plan_level(full, other, 256, &lv, &why), RAD_E_STATE);
    /* Persistent state sized by the step keeps its full size and place at a smaller step... */
    Program shrank = mk(256);
    shrank.buffers[4].decl.shape[0] = 1024;
    CHECK_OK(plan_level(full, shrank, 256, &lv, &why));
    CHECK_EQ(lv.decl[4].shape[0], full.buffers[4].decl.shape[0]);
    CHECK_EQ(lv.offset[4], full.buffers[4].arena_offset);
    /* ...but one larger at a smaller step was not sized by the step, and is refused. */
    Program grew = mk(256);
    grew.buffers[4].decl.shape[0] = 8192;
    CHECK_EQ(plan_level(full, grew, 256, &lv, &why), RAD_E_STATE);
}

TEST(unknown_liveness_is_conservative) {
    Program p;
    p.buffers.emplace_back();
    p.ops.resize(4);

    /* With no declared uses nothing may share: an op may issue any handle from declare, in any
     * order, so declaration position says nothing about when it is read. */
    p.buffers.push_back(make_buf("x", 1024, RAD_BUF_TRANSIENT, -1, -1));
    p.buffers.push_back(make_buf("y", 1024, RAD_BUF_TRANSIENT, -1, -1));

    BufPlanReport rep;
    CHECK_OK(plan_buffers(p, &rep));
    CHECK_EQ(rep.n_unknown_live, 2);
    CHECK_EQ(rep.transient_bytes, 2048);
    CHECK(p.buffers[1].arena_offset != p.buffers[2].arena_offset);
}

TEST(host_and_device_buffers_get_separate_offset_spaces) {
    Program p;
    p.buffers.emplace_back();
    p.ops.resize(4);

    BufferInfo d0 = make_buf("dev_a", 4096, RAD_BUF_TRANSIENT, 1, 3);
    BufferInfo h0 = make_buf("host_a", 4096, RAD_BUF_TRANSIENT, 1, 3);
    h0.decl.domain = RAD_DOMAIN_HOST;
    BufferInfo h1 = make_buf("host_b", 2048, RAD_BUF_TRANSIENT, 1, 3);
    h1.decl.domain = RAD_DOMAIN_HOST;
    p.buffers.push_back(d0);
    p.buffers.push_back(h0);
    p.buffers.push_back(h1);

    BufPlanReport rep;
    CHECK_OK(plan_buffers(p, &rep));

    /* A host-site activation cannot index device memory, so each domain is coloured separately
     * with offsets relative to its own arena. Live buffers in different domains may share an
     * offset; in the same domain they may not. */
    CHECK_EQ(p.buffers[1].arena_offset, 0);
    CHECK_EQ(p.buffers[2].arena_offset, 0);
    CHECK(p.buffers[3].arena_offset != p.buffers[2].arena_offset);
    CHECK_EQ(p.arena_bytes, 4096);
    CHECK_EQ(p.host_arena_bytes, 4096 + 2048);
    CHECK_EQ(rep.host_bytes, 4096 + 2048);
}

TEST(buffer_uses_drive_liveness) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    RadBufDecl bd{};
    bd.dtype = RAD_I8;
    bd.rank = 1;
    bd.shape[0] = 4096;
    bd.kind = RAD_BUF_TRANSIENT;
    bd.domain = RAD_DOMAIN_DEVICE;
    rad_buf x = b.decl_buffer("x", &bd);
    rad_buf y = b.decl_buffer("y", &bd);
    CHECK(x == 1 && y == 2);

    rad_weight w = decl_w(b, "w0");
    rad_op o1 = decl_gemm(b, w);
    rad_op o2 = decl_gemm(b, w);
    rad_op o3 = decl_gemm(b, w);

    CHECK_OK(b.op_writes(o1, &x, 1));
    CHECK_OK(b.op_reads (o2, &x, 1));
    CHECK_OK(b.op_writes(o3, &y, 1));
    CHECK_OK(b.finish());

    CHECK_EQ(b.program().buffer(x)->first_def, 1);
    CHECK_EQ(b.program().buffer(x)->last_use, 2);
    CHECK_EQ(b.program().buffer(y)->first_def, 3);
    CHECK_EQ(b.buf_report().n_unknown_live, 0);
    /* x dies at op 2 and y is born at op 3, so one 4 KiB slot serves both. */
    CHECK_EQ(b.program().buffer(x)->arena_offset, b.program().buffer(y)->arena_offset);
    CHECK_EQ(b.program().arena_bytes, 4096);
}

/* ================================================================== declare errors */
/* The parameter schema check belongs to Registry::validate. These tests assert that declare
 * refuses and that the registry's reason reaches the caller intact. */
TEST(two_ranged_parameters_is_an_error) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    rad_weight w = decl_w(b, "w");
    const RadParam p[] = { RAD_RANGE("M", 1, 8192), RAD_INT("N", 4096),
                           RAD_RANGE("K", 1, 4096), RAD_STR("dtype", "bf16") };
    rad_op h = b.decl_op("t_gemm", p, 4, &w, 1);

    CHECK_EQ(h, RAD_NULL_HANDLE);
    CHECK_EQ(b.status(), RAD_E_SCHEMA);
    CHECK_EQ((int)b.errors().size(), 1);
    /* Both offending parameters are named. */
    CHECK(has(b.errors()[0], "'M'"));
    CHECK(has(b.errors()[0], "'K'"));
    CHECK(has(b.errors()[0], "at most one ranged parameter"));
    /* The op is not recorded, so the run phase cannot reach it. */
    CHECK_EQ((int)b.program().ops.size(), 1);
}

TEST(schema_mismatches_are_named) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());
    rad_weight w = decl_w(b, "w");

    /* Missing a required key. */
    const RadParam miss[] = { RAD_RANGE("M", 1, 8192), RAD_INT("N", 4096),
                              RAD_STR("dtype", "bf16") };
    CHECK_EQ(b.decl_op("t_gemm", miss, 3, &w, 1), RAD_NULL_HANDLE);
    CHECK(has(b.errors().back(), "required parameter 'K' was not given"));

    /* A key nothing declared. */
    const RadParam extra[] = { RAD_RANGE("M", 1, 8192), RAD_INT("N", 4096), RAD_INT("K", 4096),
                               RAD_STR("dtype", "bf16"), RAD_INT("tile", 128) };
    CHECK_EQ(b.decl_op("t_gemm", extra, 5, &w, 1), RAD_NULL_HANDLE);
    CHECK(has(b.errors().back(), "no parameter 'tile' in this op's schema"));

    /* The right key, the wrong type. */
    const RadParam wrong[] = { RAD_RANGE("M", 1, 8192), RAD_INT("N", 4096), RAD_INT("K", 4096),
                               RAD_INT("dtype", 3) };
    CHECK_EQ(b.decl_op("t_gemm", wrong, 4, &w, 1), RAD_NULL_HANDLE);
    CHECK(has(b.errors().back(), "the schema says it is a string"));

    /* The wrong number of weight operands. Arguments are positional, so at issue this would be
     * silent garbage; it must be caught here. Registry::validate sees only parameters, so this
     * check stays in declare. */
    const RadParam ok[] = { RAD_RANGE("M", 1, 8192), RAD_INT("N", 4096), RAD_INT("K", 4096),
                            RAD_STR("dtype", "bf16") };
    const rad_weight two[] = { w, w };
    CHECK_EQ(b.decl_op("t_gemm", ok, 4, two, 2), RAD_NULL_HANDLE);
    CHECK(has(b.errors().back(), "2 weight operands declared"));

    CHECK_EQ(b.status(), RAD_E_SCHEMA);
}

TEST(no_schema_names_the_near_miss) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    const RadParam p[] = { RAD_INT("M", 8), RAD_INT("n", 4096) };
    CHECK_EQ(b.decl_op("t_tei", p, 2, nullptr, 0), RAD_NULL_HANDLE);
    CHECK_EQ(b.status(), RAD_E_NOSCHEMA);
    /* Without a near miss, a typo reads like an unimplemented op (spec §2.3). The suggestion is
     * the registry's edit-distance policy; this only checks it reaches the caller. */
    CHECK(has(b.errors().back(), "no loaded kernel plugin declares op 't_tei'"));
    CHECK(has(b.errors().back(), "did you mean"));
}

/* ================================================================== a weight list over its tables
 * A table operand consumes a run of the declared weights: fixed weight positions take one each,
 * and the rest is divided over the table positions in declaration order, the first positions
 * taking one extra when it does not divide. The layout pass, converter and issue path all read
 * this map (OpInfo::weight_opd), so it is asserted whole. t_moe has one fixed weight (operand 1)
 * and eight tables, class pairs at operands 2/3, 5/6, 7/8 and 9/10, as a four-class grouped GEMM
 * lays out its classes. */
namespace {

/* `n` weights declared in a row under `tag`, and t_moe over them. */
rad_op decl_moe(Builder& b, const char* tag, int n, std::vector<rad_weight>* w) {
    w->clear();
    for (int i = 0; i < n; ++i)
        w->push_back(decl_w(b, (std::string(tag) + ".e" + std::to_string(i)).c_str(), 64, 64));
    const RadParam p[] = { RAD_INT("M", 8), RAD_INT("n", 64) };
    return b.decl_op("t_moe", p, 2, w->data(), (int)w->size());
}

/* The operand each declared weight went to, as runs: "1x1 2x3" is one weight at operand 1, then
 * three at operand 2. */
std::string runs(const OpInfo* o) {
    std::string s;
    for (size_t i = 0; i < o->weight_opd.size();) {
        size_t j = i;
        while (j < o->weight_opd.size() && o->weight_opd[j] == o->weight_opd[i]) ++j;
        s += (s.empty() ? "" : " ") + std::to_string(o->weight_opd[i]) + "x" +
             std::to_string(j - i);
        i = j;
    }
    return s;
}

}  /* namespace */

TEST(a_weight_list_divides_over_its_tables_the_first_taking_the_remainder) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());
    std::vector<rad_weight> w;

    /* Ten experts' codes and scales after the fixed weight: twenty entries over eight tables is
     * two each with four left, so the first four positions (classes 0 and 1, codes and scales)
     * hold three and the rest two, as a four-class grouped GEMM over ten experts. */
    rad_op h = decl_moe(b, "ten", 21, &w);
    REQUIRE(h != RAD_NULL_HANDLE);
    const OpInfo* o = b.program().op(h);
    REQUIRE(o != nullptr);
    CHECK_EQ(runs(o), std::string("1x1 2x3 3x3 5x3 6x3 7x2 8x2 9x2 10x2"));
    /* The list is cut into runs in declaration order, not dealt round-robin. */
    CHECK(o->weights == w);

    /* An evenly dividing list: sixteen entries, two per table. */
    h = decl_moe(b, "eight", 17, &w);
    REQUIRE(h != RAD_NULL_HANDLE);
    CHECK_EQ(runs(b.program().op(h)), std::string("1x1 2x2 3x2 5x2 6x2 7x2 8x2 9x2 10x2"));

    /* One entry per table is the minimum. */
    h = decl_moe(b, "four", 9, &w);
    REQUIRE(h != RAD_NULL_HANDLE);
    CHECK_EQ(runs(b.program().op(h)), std::string("1x1 2x1 3x1 5x1 6x1 7x1 8x1 9x1 10x1"));
    CHECK_EQ((int)b.errors().size(), 0);
}

TEST(a_weight_list_shorter_than_its_tables_is_refused_by_name) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());
    std::vector<rad_weight> w;
    const size_t ops = b.program().ops.size();

    /* Seven entries for eight tables leaves one empty: refused at declare, with the counts. */
    CHECK_EQ(decl_moe(b, "short", 8, &w), RAD_NULL_HANDLE);
    CHECK_EQ(b.status(), RAD_E_SCHEMA);
    REQUIRE(!b.errors().empty());
    CHECK(has(b.errors().back(), "op 't_moe': 8 weight operands declared"));
    CHECK(has(b.errors().back(), "1 fixed weight operands and 8 weight TABLE operands"));
    CHECK(has(b.errors().back(), "so 7 leaves a table empty"));
    /* Nothing is recorded for the run phase. */
    CHECK_EQ(b.program().ops.size(), ops);
}

/* ================================================================== misses, reported whole */
TEST(a_hole_in_the_range_is_reported_as_that_band) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    /* t_skinny has rows at M <= 64 and M >= 4096 and nothing in between. */
    rad_weight w = decl_w(b, "w");
    const RadParam p[] = { RAD_RANGE("M", 1, 8192), RAD_INT("N", 4096), RAD_INT("K", 4096),
                           RAD_STR("dtype", "bf16") };
    rad_op h = b.decl_op("t_skinny", p, 4, &w, 1);

    /* No handle is returned, so the plugin can fall back to the unfused sequence; the op is still
     * recorded so the graph dump shows the hole. Declare always completes. */
    CHECK_EQ(h, RAD_NULL_HANDLE);
    CHECK_EQ((int)b.program().ops.size(), 2);
    CHECK_EQ(b.status(), RAD_OK);

    const OpInfo& o = b.program().ops[1];
    CHECK(o.bands.size() >= 3);
    if (o.bands.size() < 3) return;
    CHECK(o.bands[0].dom[RAD_DOMAIN_DEVICE]);          /* M <= 64 */
    CHECK(!o.bands[1].dom[RAD_DOMAIN_DEVICE]);         /* the hole */
    CHECK(o.bands[2].dom[RAD_DOMAIN_DEVICE]);          /* M >= 4096 */

    /* The missing-kernel report is per band: it names the hole, not the whole op. */
    bool named = false;
    for (const auto& m : b.program().misses)
        if (m.op == "t_skinny" && has(m.geom, "M=4095")) named = true;
    CHECK(named);

    const std::string s = dump_graph_str(b.program());
    CHECK(has(s, "(nothing)"));
    CHECK(has(s, "---- unresolved:"));
}

TEST(an_op_with_no_host_kernel_is_a_miss_per_band_not_a_failure) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    /* t_attn is device-only. A missing host kernel only rules out host-site placement for this
     * op, so declare succeeds and the report records the host miss. */
    rad_op h = decl_t_attn(b);
    CHECK(h == 1);
    CHECK_EQ(b.op_resolved(h), 1);
    CHECK_EQ(b.status(), RAD_OK);

    bool host_miss = false;
    for (const auto& m : b.program().misses) if (m.domain == RAD_DOMAIN_HOST) host_miss = true;
    CHECK(host_miss);
}

/* A device hole is decided per instance. Of two declarations of one op, the weighted one's device
 * miss is covered by its host kernel (the planner moves it to the host by placing its weight
 * there), so on a card only the weightless instance's miss is a hole. */
TEST(a_device_hole_is_decided_per_instance) {
    Program p;
    p.ops.resize(3);
    p.ops[1].op = "rmsnorm";                          /* no weight */
    p.ops[1].index = 1;
    p.ops[2].op = "rmsnorm";
    p.ops[2].index = 2;
    p.ops[2].weights = { 1 };
    auto miss = [](int32_t index, int domain) {
        Program::Miss m;
        m.op = "rmsnorm";
        m.span = "M <= 64";
        m.why = "no kernel matched";
        m.domain = domain;
        m.index = index;
        return m;
    };
    p.misses.push_back(miss(2, RAD_DOMAIN_DEVICE));   /* the weighted one; its host kernel exists */
    CHECK_EQ((int)p.device_holes(false).size(), 0);

    p.misses.push_back(miss(1, RAD_DOMAIN_DEVICE));   /* the weightless one */
    CHECK_EQ((int)p.device_holes(false).size(), 1);
    CHECK_EQ(p.device_holes(false)[0]->index, 1);
    CHECK_EQ((int)p.device_holes(true).size(), 0);    /* the host backend runs both on the host */

    p.misses.push_back(miss(2, RAD_DOMAIN_HOST));     /* no host kernel for the weighted one */
    CHECK_EQ((int)p.device_holes(true).size(), 1);
    CHECK_EQ(p.device_holes(true)[0]->index, 2);
}

/* ================================================================== init, after the whole graph */
TEST(a_refusing_init_becomes_a_miss_and_not_an_abort) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    /* t_eq_m32's init always fails. Declare must still complete and the following op still
     * resolve; init runs after the whole graph, so the miss report covers all of it. */
    rad_weight w = decl_w(b, "w");
    const RadParam eq[] = { RAD_INT("M", 32), RAD_INT("n", 4096) };
    rad_op h_eq = b.decl_op("t_eq", eq, 2, &w, 1);
    rad_op h_gemm = decl_gemm(b, w);
    CHECK(h_eq == 1);
    CHECK(h_gemm == 2);

    /* Resolution succeeded: the kernel matched the constraints. */
    CHECK(b.program().op(h_eq)->bands[0].dom[RAD_DOMAIN_DEVICE]);

    const int st = b.finish();
    CHECK(st < 0);
    CHECK(has(b.errors().back(), "init() refused the instance"));

    /* The band becomes a miss, so no kernel whose setup failed can launch; the rest of the graph
     * is intact. */
    CHECK(!b.program().op(h_eq)->bands[0].dom[RAD_DOMAIN_DEVICE]);
    CHECK(b.program().op(h_gemm)->bands[0].dom[RAD_DOMAIN_DEVICE]);
    CHECK_EQ(b.op_resolved(h_eq), 0);
    CHECK_EQ(b.op_resolved(h_gemm), 1);
}

/* ================================================================== scratch and tuning */
TEST(tune_requests_are_one_per_kernel_and_shape) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    rad_weight w = decl_w(b, "w");
    /* Repeated identical ops yield one tuning request. */
    decl_gemm(b, w);
    decl_gemm(b, w);
    CHECK_OK(b.finish());

    /* Only t_gemm_m64 declares an axis, and only its band's shape needs measuring. */
    CHECK_EQ((int)b.program().tune_requests.size(), 1);
    CHECK_EQ(b.program().tune_requests[0].kernel, std::string("t_gemm_m64"));
    CHECK_EQ(b.program().tune_requests[0].plugin, std::string("radtest_base"));
    CHECK_EQ(b.program().tune_requests[0].info->n_tunables, 1);
    CHECK(has(b.program().tune_requests[0].geom.str(), "M=64"));
}

/* ================================================================== KV groups */
/* A head without its own cache must bind none: it gets no history pass, so a group bound at one
 * of its layers (meta.n_layers and above) would be read but never written. The same head borrowing
 * only trunk layers finishes. */
TEST(a_head_with_no_cache_of_its_own_binds_no_group_of_its_own) {
    NEED_PLUGIN(reg)
    for (bool own : { false, true }) {
        Builder b(reg, make_meta(), make_ctx());
        const rad_kvgroup g = decl_kv_full(b);
        decl_t_attn(b);
        CHECK_OK(b.bind_layer_kv(0, g));
        CHECK_OK(b.bind_layer_kv(1, g));
        if (own) CHECK_OK(b.bind_layer_kv(2, g));   /* make_meta has two trunk layers; 2 is the head's */
        RadDrafterDecl d{};
        d.kind = RAD_DRAFT_SERIAL; d.name = "mtp"; d.depth = 3;
        CHECK_OK(rad_declare_drafter(&b, &d));
        CHECK_OK(rad_drafter_shared_kv(&b, 0));
        b.finish();
        if (own) {
            CHECK_EQ(b.status(), RAD_E_STATE);
            CHECK(has(b.errors().back(), "keeps no cache of its own"));
        } else {
            CHECK_OK(b.status());
        }
    }
}

TEST(kv_block_size_comes_off_the_kernel) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    rad_kvgroup g = decl_kv_full(b);
    CHECK(g == 1);

    /* With no consuming op there is no answer; this is an error because a silent zero would
     * become a division by zero in the block manager. */
    CHECK_EQ(b.kv_block_size(g), 0);
    CHECK_EQ(b.status(), RAD_E_STATE);
    CHECK(has(b.errors().back(), "no op that reads this group"));

    rad_op h = decl_t_attn(b);
    CHECK(h == 1);

    /* 16 because the resolved kernel's constraints say so, not a core choice (spec §7.2). */
    CHECK_EQ(b.kv_block_size(g), 16);

    CHECK_OK(b.bind_layer_kv(0, g));
    CHECK_OK(b.bind_layer_kv(1, g));
    b.finish();

    const KVGroupInfo& gi = b.program().kv_groups[g];
    CHECK_EQ(gi.block_size, 16);
    /* Invariant: one page covers every layer bound to the group, as KVManager defines a page. The
     * manager divides this figure by the layer count, so a per-layer size would make the pool
     * n_layers too small with a fractional layer stride. Hence the `* 2` for the two bound
     * layers. */
    CHECK_EQ(gi.bytes_per_block, 2 * 16 * 8 * 128 * 2 * 2);
    CHECK_EQ((int)gi.layers.size(), 2);
}

TEST(glm_index_cache_pages_are_sized_by_the_real_pool_consumer) {
    Registry reg;
    REQUIRE_EQ(reg.load_dir(home() + "/kernels", { "libref" }), RAD_OK);
    Builder b(reg, make_meta(), make_ctx());
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_FULL;
    kd.dtype = RAD_U8;
    kd.n_head_kv = 1;
    kd.head_dim = 17;
    const auto group = b.decl_kv_group("glm_index", &kd);
    RadWeightDecl wd{};
    wd.dtype = RAD_F32;
    wd.rank = 2;
    wd.shape[0] = 4;
    wd.shape[1] = 128;
    wd.access = RAD_ACCESS_PER_TOKEN;
    wd.shard = RAD_SHARD_NONE;
    const auto ape = b.decl_weight("pool_ape", &wd);
    const RadParam params[] = { RAD_RANGE("M", 1, 12), RAD_INT("n", 128), RAD_INT("ratio", 4),
                                RAD_STR("dtype", "bf16"), RAD_INT("block_size", 4) };
    REQUIRE(b.decl_op("dsa_pool", params, 5, &ape, 1));
    CHECK_EQ(b.kv_block_size(group), 4);
    CHECK_OK(b.bind_layer_kv(0, group));
    CHECK_OK(b.bind_layer_kv(1, group));
    REQUIRE_EQ(b.finish(), RAD_OK);
    CHECK_EQ(b.program().kv_groups[group].bytes_per_block, (int64_t)(2 * 136));
}

TEST(a_kv_group_with_no_layers_is_an_error) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    decl_kv_full(b);
    decl_t_attn(b);

    /* A sequence owns one state instance per bound layer, so a group with no layers has no page
     * size; declare refuses it. */
    CHECK(b.finish() < 0);
    CHECK(has(b.errors().back(), "kv group 'kv_full' has no layers bound"));
}

TEST(linear_state_is_sized_per_sequence) {
    NEED_PLUGIN(reg)
    RadBuildCtx ctx = make_ctx();
    ctx.max_spec = 4;
    Builder b(reg, make_meta(), ctx);

    RadKVGroupDecl ld{};
    ld.kind = RAD_KV_LINEAR;
    ld.dtype = RAD_F32;
    ld.n_head_kv = 32;
    ld.state_dim[0] = 128;
    ld.state_dim[1] = 128;
    rad_kvgroup gl = b.decl_kv_group("gdn_state", &ld);

    RadKVGroupDecl cd{};
    cd.kind = RAD_KV_CONV;
    cd.dtype = RAD_F32;
    cd.n_head_kv = 32;
    cd.head_dim = 128;
    cd.conv_width = 4;
    rad_kvgroup gc = b.decl_kv_group("gdn_conv", &cd);

    CHECK_OK(b.bind_layer_kv(0, gl));
    CHECK_OK(b.bind_layer_kv(0, gc));
    b.finish();

    /* 32 heads of 128x128 fp32 is 2 MiB per layer, too large to snapshot per block (spec §7.3).
     * A per-sequence group has no block size: it warns and returns 0; use bytes_per_state. */
    CHECK_EQ(b.program().kv_groups[gl].bytes_per_state, 32LL * 128 * 128 * 4);
    CHECK_EQ(b.program().kv_groups[gl].bytes_per_block, 0);

    /* A rolling window of conv_width-1 + n_spec entries, so a speculative rejection changes the
     * read offset instead of recomputing (spec §10).
     *
     * This must equal the state tensor's extent: kernels lay out a slot from it, and the KV
     * manager clears, checkpoints and restores the slot this figure describes. One extra entry
     * would spill into the next sequence's window. See rad_builder.cpp's note on `entries`. */
    CHECK_EQ(b.program().kv_groups[gc].bytes_per_state, 32LL * 128 * (4 - 1 + 4) * 4);
    CHECK_EQ(b.program().kv_groups[gc].bytes_per_state,
             32LL * 128 * conv_window_entries(cd, 4) * 4);
}

/* rad_kv_group_replicated marks a paged group the same on every rank (the tiers keep it once a
 * host slot). It is refused, by name, on a group that is not paged, beside rad_kv_group_zero in
 * either order, on a handle that names no group, and after declare. */
TEST(a_replicated_kv_group_is_paged_and_never_zero) {
    NEED_PLUGIN(reg)
    {
        Builder b(reg, make_meta(), make_ctx());
        const rad_kvgroup g = decl_kv_full(b);
        CHECK_OK(rad_kv_group_replicated(&b, g));
        CHECK(b.program().kv_groups[g].replicated);
        CHECK_EQ(rad_kv_group_zero(&b, g), (int)RAD_E_INVAL);
        CHECK(has(b.errors().back(), "kv group 'kv_full': kv_group_zero on a group declared "
                                     "replicated"));
        CHECK(!b.program().kv_groups[g].zero);
    }
    {
        Builder b(reg, make_meta(), make_ctx());
        const rad_kvgroup g = decl_kv_full(b);
        CHECK_OK(rad_kv_group_zero(&b, g));
        CHECK_EQ(rad_kv_group_replicated(&b, g), (int)RAD_E_INVAL);
        CHECK(has(b.errors().back(), "kv group 'kv_full': kv_group_replicated on a group declared "
                                     "zero"));
        CHECK(!b.program().kv_groups[g].replicated);
    }
    {
        Builder b(reg, make_meta(), make_ctx());
        RadKVGroupDecl ld{};
        ld.kind = RAD_KV_LINEAR;
        ld.dtype = RAD_F32;
        ld.n_head_kv = 4;
        ld.state_dim[0] = 16;
        ld.state_dim[1] = 16;
        const rad_kvgroup gl = b.decl_kv_group("gdn_state", &ld);
        CHECK_EQ(rad_kv_group_replicated(&b, gl), (int)RAD_E_INVAL);
        CHECK(has(b.errors().back(), "kv group 'gdn_state': kv_group_replicated on a group that is "
                                     "not paged"));
        CHECK_EQ(rad_kv_group_replicated(&b, 7), (int)RAD_E_INVAL);
        CHECK(has(b.errors().back(), "kv_group_replicated: 7 is not a declared KV group"));
    }
    {
        Builder b(reg, make_meta(), make_ctx());
        const rad_kvgroup g = decl_kv_full(b);
        decl_t_attn(b);
        CHECK_OK(b.bind_layer_kv(0, g));
        CHECK_OK(b.bind_layer_kv(1, g));
        REQUIRE_EQ(b.finish(), RAD_OK);
        CHECK_EQ(rad_kv_group_replicated(&b, g), (int)RAD_E_STATE);
        CHECK(!b.program().kv_groups[g].replicated);
    }
}

/* ================================================================== names, notes, the C surface */

/* decl_name_map keeps `src_index`: a set index survives and an unset one is -1, not slice 0. The
 * field value-initialises to 0 while its documented default is -1, so losing it would silently
 * make every map read sub-tensor 0. */
TEST(a_name_map_carries_its_sub_tensor_index) {
    NEED_PLUGIN(reg)
    RadBuildCtx ctx = make_ctx();
    Builder b(reg, make_meta(), ctx);

    {   /* One expert's slice of two stacked tensors, concatenated (map_slice_concat's shape). */
        std::string declared = "blk.0.ffn_gate_up.3.weight";
        std::string s0 = "blk.0.ffn_gate_exps.weight";
        std::string s1 = "blk.0.ffn_up_exps.weight";
        RadNameMap m{};
        m.declared = declared.c_str();
        m.mode = RAD_MAP_CONCAT;
        m.n_src = 2;
        m.src[0] = s0.c_str();  m.src_index[0] = 3;
        m.src[1] = s1.c_str();  m.src_index[1] = 3;
        m.concat_dim = 0;
        CHECK_OK(b.decl_name_map(&m));
    }
    {   /* An ordinary whole-tensor map, which must not come back as slice 0. */
        std::string declared = "token_embd.weight";
        std::string s0 = "model.embed_tokens.weight";
        RadNameMap m{};
        m.declared = declared.c_str();
        m.mode = RAD_MAP_COPY;
        m.n_src = 1;
        m.src[0] = s0.c_str();
        m.src_index[0] = -1;
        CHECK_OK(b.decl_name_map(&m));
    }

    CHECK_EQ((int)b.program().name_map.size(), 2);
    CHECK_EQ(b.program().name_map[0].src_index[0], 3);
    CHECK_EQ(b.program().name_map[0].src_index[1], 3);
    CHECK_EQ(b.program().name_map[1].src_index[0], -1);
    /* The unused tail is -1, not 0: a consumer that walks all eight must not see slice zero. */
    CHECK_EQ(b.program().name_map[1].src_index[1], -1);
    CHECK_EQ(b.program().name_map[1].src_index[7], -1);
}

/* A concat map may be declared in pieces (how a 128-shard tensor exceeds `src[8]`), but two other
 * repeats are errors. A converter lookup built with `std::map::emplace` keeps the first entry, so
 * a weight declared twice would silently convert from the first spelling; since concat pieces are
 * joined, repeats that must not be joined are refused at declare. */
TEST(a_concat_name_map_may_be_declared_in_pieces) {
    RadBuildCtx ctx = make_ctx();

    auto piece = [](const char* declared, std::initializer_list<const char*> src, int mode,
                    int dim) {
        RadNameMap m{};
        m.declared = declared;
        m.mode = mode;
        m.n_src = (int)src.size();
        int i = 0;
        for (const char* s : src) { m.src[i] = s; m.src_index[i] = -1; ++i; }
        m.concat_dim = dim;
        return m;
    };

    {   /* Twelve shards as two pieces: both are kept, in call order. */
        NEED_PLUGIN(reg)
        Builder b(reg, make_meta(), ctx);
        RadNameMap a = piece("ngram.weight", { "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7" },
                             RAD_MAP_CONCAT, 0);
        RadNameMap c = piece("ngram.weight", { "s8", "s9", "s10", "s11" }, RAD_MAP_CONCAT, 0);
        CHECK_OK(b.decl_name_map(&a));
        CHECK_OK(b.decl_name_map(&c));
        CHECK_EQ((int)b.program().name_map.size(), 2);
        CHECK_EQ(b.program().name_map[0].n_src, 8);
        CHECK_EQ(b.program().name_map[1].n_src, 4);
        /* Call order is concatenation order: shard 0 first, or the table is permuted and every
         * hashed row id addresses the wrong embedding. */
        CHECK(std::string(b.program().name_map[0].src[0]) == "s0");
        CHECK(std::string(b.program().name_map[1].src[0]) == "s8");
    }
    {   /* A repeated copy map is refused. */
        NEED_PLUGIN(reg)
        Builder b(reg, make_meta(), ctx);
        RadNameMap a = piece("w", { "x" }, RAD_MAP_COPY, 0);
        RadNameMap c = piece("w", { "y" }, RAD_MAP_COPY, 0);
        CHECK_OK(b.decl_name_map(&a));
        CHECK(b.decl_name_map(&c) != RAD_OK);
    }
    {   /* So is a piece joining along a different axis. */
        NEED_PLUGIN(reg)
        Builder b(reg, make_meta(), ctx);
        RadNameMap a = piece("w", { "x" }, RAD_MAP_CONCAT, 0);
        RadNameMap c = piece("w", { "y" }, RAD_MAP_CONCAT, 1);
        CHECK_OK(b.decl_name_map(&a));
        CHECK(b.decl_name_map(&c) != RAD_OK);
    }
}

TEST(name_map_and_notes_survive_the_plugin) {
    NEED_PLUGIN(reg)
    RadBuildCtx ctx = make_ctx();
    ctx.scope = "draft";
    Builder b(reg, make_meta(), ctx);

    {
        /* The plugin owns these strings and the Program outlives the call, so they are interned,
         * not borrowed. */
        std::string declared = "attn_qkv.weight";
        std::string s0 = "model.layers.0.q_proj.weight";
        std::string s1 = "model.layers.0.k_proj.weight";
        RadNameMap m{};
        m.declared = declared.c_str();
        m.mode = RAD_MAP_CONCAT;
        m.n_src = 2;
        m.src[0] = s0.c_str();
        m.src[1] = s1.c_str();
        m.concat_dim = 0;
        CHECK_OK(b.decl_name_map(&m));
    }

    CHECK_EQ((int)b.program().name_map.size(), 1);
    /* The scope prefixes declared names so a drafter and the target can share one builder. */
    CHECK_EQ(std::string(b.program().name_map[0].declared), std::string("draft.attn_qkv.weight"));
    CHECK_EQ(std::string(b.program().name_map[0].src[1]),
             std::string("model.layers.0.k_proj.weight"));

    rad_weight w = decl_w(b, "attn_qkv.weight", 16, 16);
    CHECK_EQ(b.program().weight(w)->name, std::string("draft.attn_qkv.weight"));

    rad_note(&b, "%d layers, gqa %d", 48, 8);
    CHECK_EQ(b.program().notes[0], std::string("48 layers, gqa 8"));
}

TEST(the_c_surface_drives_the_builder) {
    NEED_PLUGIN(reg)
    RadModelMeta meta = make_meta();
    Builder b(reg, meta, make_ctx());
    RadBuilder* h = &b;              /* the handle is the builder */

    RadWeightDecl wd{};
    wd.dtype = RAD_BF16; wd.rank = 2; wd.shape[0] = 4096; wd.shape[1] = 4096;
    wd.access = RAD_ACCESS_PER_TOKEN;
    rad_weight w = rad_decl_weight(h, "blk.0.ffn_gate.weight", &wd);
    CHECK(w != RAD_NULL_HANDLE);

    rad_op op = RAD_OP(h, "t_gemm",
                       RAD_PARAMS(RAD_RANGE("M", 1, 8192), RAD_INT("N", 4096), RAD_INT("K", 4096),
                                  RAD_STR("dtype", "bf16")),
                       RAD_WEIGHTS(w));
    CHECK(op != RAD_NULL_HANDLE);
    CHECK_EQ(rad_op_resolved(h, op), 1);
    CHECK_EQ(rad_op_resolved(h, 99), 0);
    /* Whether an op is in the vocabulary, asked without declaring it; never an error. */
    CHECK_EQ(rad_op_declared(h, "t_gemm"), 1);
    CHECK_EQ(rad_op_declared(h, "t_no_such_op"), 0);
    CHECK_EQ(rad_op_declared(h, nullptr), 0);
    CHECK_EQ(b.status(), RAD_OK);

    /* A weight declared twice is refused now; the container has one entry for it. */
    CHECK_EQ(rad_decl_weight(h, "blk.0.ffn_gate.weight", &wd), RAD_NULL_HANDLE);
    CHECK_EQ(b.status(), RAD_E_DUPLICATE);

    CHECK_EQ(rad_meta_geti(&meta, "nothing", 7), 7);
}

/* The program is moved out of the builder, and a short std::string keeps its characters inline
 * (SSO), so a pointer into it would dangle after the move. The name lives in drafter_name; the decl
 * carries none. */
TEST(the_drafter_name_survives_the_program_leaving_the_builder) {
    Registry reg;
    std::unique_ptr<Builder> b(new Builder(reg, make_meta(), make_ctx()));

    std::string plugin_owned = "mtp";
    RadDrafterDecl d{};
    d.kind  = RAD_DRAFT_SERIAL;
    d.name  = plugin_owned.c_str();
    d.depth = 3;
    CHECK_OK(b->declare_drafter(&d));
    plugin_owned.assign("xxx");

    Program p = std::move(b->program());
    b.reset();
    CHECK_EQ(p.drafter_name, std::string("mtp"));
    CHECK(p.drafter.name == nullptr);
    CHECK_EQ(p.drafter.kind, (int)RAD_DRAFT_SERIAL);
    CHECK_EQ(p.drafter.depth, 3);
}

/* rad_drafter_short_block applies only to a declared block drafter, and only to a block shorter
 * than its own that still proposes a token; anything else is refused at declare, since the core
 * would stage a block the plugin has no ops for. */
TEST(a_block_drafters_short_block_is_shorter_and_still_proposes) {
    Registry reg;
    {
        Builder b(reg, make_meta(), make_ctx());
        CHECK_EQ(rad_drafter_short_block(&b, 4), RAD_E_STATE);   /* nothing declared */
    }
    {
        Builder b(reg, make_meta(), make_ctx());
        RadDrafterDecl d{};
        d.kind = RAD_DRAFT_SERIAL; d.name = "mtp"; d.depth = 4;
        CHECK_OK(rad_declare_drafter(&b, &d));
        CHECK_EQ(rad_drafter_short_block(&b, 3), RAD_E_STATE);   /* a serial head */
    }
    {
        Builder b(reg, make_meta(), make_ctx());
        RadBufDecl bd{};
        bd.dtype = RAD_I32; bd.rank = 2; bd.shape[0] = 256; bd.shape[1] = 4;
        const rad_buf prop = rad_decl_buffer(&b, "prop", &bd);
        RadDrafterDecl d{};
        d.kind = RAD_DRAFT_BLOCK; d.name = "df"; d.depth = 4; d.mask_token = 5;
        d.proposal = prop; d.proposal_pitch = 4;               /* block 5: the anchor + 4 */
        CHECK_OK(rad_declare_drafter(&b, &d));
        CHECK_EQ(rad_drafter_short_block(&b, 5), RAD_E_INVAL);   /* not shorter */
        CHECK_EQ(rad_drafter_short_block(&b, 1), RAD_E_INVAL);   /* proposes nothing */
        CHECK_EQ(b.program().drafter_short_block, 0);
        CHECK_OK(rad_drafter_short_block(&b, 3));
        CHECK_EQ(b.program().drafter_short_block, 3);
    }
}

/* A block drafter's proposal pitch is its row pitch, which the engine reads row s at: left at 0 it
 * defaults to depth, and under depth it is refused, since a row could not hold the proposals. */
TEST(a_block_drafters_proposal_pitch_holds_its_proposals) {
    Registry reg;
    RadBufDecl bd{};
    bd.dtype = RAD_I32; bd.rank = 2; bd.shape[0] = 256; bd.shape[1] = 4;
    {
        Builder b(reg, make_meta(), make_ctx());
        RadDrafterDecl d{};
        d.kind = RAD_DRAFT_BLOCK; d.name = "df"; d.depth = 4; d.mask_token = 5;
        d.proposal = rad_decl_buffer(&b, "prop", &bd);
        CHECK_OK(rad_declare_drafter(&b, &d));
        CHECK_EQ(b.program().drafter.proposal_pitch, 4);
    }
    {
        Builder b(reg, make_meta(), make_ctx());
        RadDrafterDecl d{};
        d.kind = RAD_DRAFT_BLOCK; d.name = "df"; d.depth = 4; d.mask_token = 5;
        d.proposal = rad_decl_buffer(&b, "prop", &bd); d.proposal_pitch = 3;
        CHECK_EQ(rad_declare_drafter(&b, &d), RAD_E_INVAL);
    }
    {
        Builder b(reg, make_meta(), make_ctx());
        RadDrafterDecl d{};
        d.kind = RAD_DRAFT_SERIAL; d.name = "mtp"; d.depth = 4;
        CHECK_OK(rad_declare_drafter(&b, &d));
        CHECK_EQ(b.program().drafter.proposal_pitch, 1);
    }
}

/* rad_drafter_shared_kv applies only to a declared serial drafter. With none declared, a block
 * drafter, or a position no head occupies, it is refused at declare; a head staged the wrong way
 * silently costs acceptance. */
TEST(a_head_that_keeps_no_cache_of_its_own_is_a_serial_one) {
    Registry reg;
    {
        Builder b(reg, make_meta(), make_ctx());
        CHECK_EQ(rad_drafter_shared_kv(&b, 0), RAD_E_STATE);   /* nothing declared */
    }
    {
        Builder b(reg, make_meta(), make_ctx());
        RadBufDecl bd{};
        bd.dtype = RAD_I32; bd.rank = 2; bd.shape[0] = 256; bd.shape[1] = 4;
        const rad_buf prop = rad_decl_buffer(&b, "prop", &bd);
        RadDrafterDecl d{};
        d.kind = RAD_DRAFT_BLOCK; d.name = "df"; d.depth = 4; d.mask_token = 5;
        d.proposal = prop; d.proposal_pitch = 4;
        CHECK_OK(rad_declare_drafter(&b, &d));
        CHECK_EQ(rad_drafter_shared_kv(&b, 0), RAD_E_STATE);   /* a block drafter */
        CHECK(!b.program().drafter_shared_kv);
    }
    {
        Builder b(reg, make_meta(), make_ctx());
        RadDrafterDecl d{};
        d.kind = RAD_DRAFT_SERIAL; d.name = "mtp"; d.depth = 4;
        CHECK_OK(rad_declare_drafter(&b, &d));
        CHECK_EQ(rad_drafter_shared_kv(&b, 2), RAD_E_INVAL);   /* no position a head sits at */
        CHECK(!b.program().drafter_shared_kv);
        CHECK_OK(rad_drafter_shared_kv(&b, 1));
        CHECK(b.program().drafter_shared_kv);
        CHECK_EQ(b.program().drafter_pos_offset, 1);
    }
    {
        /* A drafter that does not ask keeps its own cache. */
        Builder b(reg, make_meta(), make_ctx());
        RadDrafterDecl d{};
        d.kind = RAD_DRAFT_SERIAL; d.name = "mtp"; d.depth = 4;
        CHECK_OK(rad_declare_drafter(&b, &d));
        CHECK(!b.program().drafter_shared_kv);
        CHECK_EQ(b.program().drafter_pos_offset, 0);
    }
}

/* ================================================================== the enum name tables
 *
 * These names appear in --debug-graph and the Model view, where an operator reads what a plugin
 * declared. Every known value has a name; "?" marks a value this core does not know.
 */
TEST(the_declared_enums_all_have_names_and_an_unknown_value_says_so) {
    CHECK_EQ(std::string(bld_domain_name(RAD_DOMAIN_DEVICE)), std::string("device"));
    CHECK_EQ(std::string(bld_domain_name(RAD_DOMAIN_HOST)), std::string("host"));

    CHECK_EQ(std::string(bld_access_name(RAD_ACCESS_PER_TOKEN)), std::string("per-token"));
    CHECK_EQ(std::string(bld_access_name(RAD_ACCESS_PER_REQUEST)), std::string("per-request"));
    CHECK_EQ(std::string(bld_access_name(RAD_ACCESS_CONDITIONAL)), std::string("conditional"));
    CHECK_EQ(std::string(bld_access_name(RAD_ACCESS_RARE)), std::string("rare"));
    CHECK_EQ(std::string(bld_access_name(RAD_ACCESS_VOCAB)), std::string("vocab"));

    CHECK_EQ(std::string(bld_buf_kind_name(RAD_BUF_TRANSIENT)), std::string("transient"));
    CHECK_EQ(std::string(bld_buf_kind_name(RAD_BUF_PERSIST)), std::string("persist"));
    CHECK_EQ(std::string(bld_buf_kind_name(RAD_BUF_DERIVED)), std::string("derived"));

    /* Every name is distinct, or two classes would read as one in the dump. */
    const char* access[] = { bld_access_name(RAD_ACCESS_PER_TOKEN),
                             bld_access_name(RAD_ACCESS_PER_REQUEST),
                             bld_access_name(RAD_ACCESS_CONDITIONAL),
                             bld_access_name(RAD_ACCESS_RARE),
                             bld_access_name(RAD_ACCESS_VOCAB) };
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j) CHECK(std::strcmp(access[i], access[j]) != 0);

    /* An unknown value is "?" rather than a plausible name. */
    CHECK_EQ(std::string(bld_access_name(9999)), std::string("?"));
    CHECK_EQ(std::string(bld_buf_kind_name(-1)), std::string("?"));
    /* Every KV kind the ABI defines is named. */
    for (int k : { RAD_KV_FULL, RAD_KV_WINDOW, RAD_KV_LINEAR, RAD_KV_CONV }) {
        const char* n = bld_kv_kind_name(k);
        CHECK(n && *n);
        CHECK(std::strcmp(n, "?") != 0);
    }
}

/* ================================================================== sizing a declared buffer
 *
 * Zero means "the core cannot size it": legal for a weight (a kernel's layout hook sizes it), a
 * declare-time failure for an activation. An ordinary buffer must never get zero.
 */
TEST(buffer_bytes_sizes_a_declaration_or_says_it_cannot) {
    RadBufDecl d{};
    d.dtype = RAD_BF16;
    d.rank  = 2;
    d.shape[0] = 16;
    d.shape[1] = 32;
    CHECK_EQ(buffer_bytes(d), (int64_t)(16 * 32 * 2));

    d.dtype = RAD_F32;
    CHECK_EQ(buffer_bytes(d), (int64_t)(16 * 32 * 4));
    d.dtype = RAD_I8;
    CHECK_EQ(buffer_bytes(d), (int64_t)(16 * 32));

    /* A rank-0 declaration has no elements; it is not a buffer of one. */
    RadBufDecl z{};
    z.dtype = RAD_BF16;
    z.rank  = 0;
    CHECK_EQ(buffer_bytes(z), 0ll);

    /* A zero extent in any dimension is zero bytes, so an unused optional plane reserves no
       arena. */
    RadBufDecl e{};
    e.dtype = RAD_BF16;
    e.rank  = 2;
    e.shape[0] = 0;
    e.shape[1] = 4096;
    CHECK_EQ(buffer_bytes(e), 0ll);

    /* An unknown dtype is 0, which plan_buffers turns into a named refusal for an activation. */
    RadBufDecl p{};
    p.dtype = 0xFFFF;
    p.rank  = 1;
    p.shape[0] = 1024;
    CHECK_EQ(buffer_bytes(p), 0ll);

    /* Large shapes must not overflow. */
    RadBufDecl big{};
    big.dtype = RAD_BF16;
    big.rank  = 2;
    big.shape[0] = 200000;
    big.shape[1] = 4096;
    CHECK_EQ(buffer_bytes(big), 200000ll * 4096ll * 2ll);
}

/* ================================================================== the Model view's data
 *
 * dump_graph_json feeds the dashboard's Model view. It groups identical declarations (a 64-layer
 * model declares one op 64 times with one bucket table) into one row with a count, and keeps the
 * declared order beside it as indices, since grouping loses the sequence a diagram needs.
 */
TEST(the_graph_json_groups_identical_declarations_and_keeps_the_order) {
    Program p;
    p.meta = make_meta();
    p.ctx.rank = 0;
    p.ctx.world_size = 1;
    p.weights.emplace_back();
    p.buffers.emplace_back();
    p.ops.emplace_back();                 /* the sentinel */

    /* Two ops alternating over sixteen layers: two groups of sixteen, thirty-two in order. */
    for (int l = 0; l < 16; ++l) {
        for (const char* name : { "attn", "ffn" }) {
            OpInfo o;
            o.op = name;
            o.index = (int32_t)p.ops.size();
            o.base.set_i("N", 4096);
            p.ops.push_back(o);
        }
    }

    const std::string s = dump_graph_json(p);
    const json j = json::parse(s, nullptr, false);
    CHECK(!j.is_discarded());

    CHECK_EQ(j["counts"].value("ops", (int64_t)0), (int64_t)32);
    CHECK_EQ(j["ops"].size(), 2u);                 /* not 32 */
    CHECK_EQ(j["ops"][0].value("count", (int64_t)0), (int64_t)16);
    CHECK_EQ(j["ops"][1].value("count", (int64_t)0), (int64_t)16);
    CHECK_EQ(j["ops"][0].value("op", std::string()), std::string("attn"));
    CHECK_EQ(j["ops"][1].value("op", std::string()), std::string("ffn"));

    /* The order: one group index per declared op. */
    CHECK_EQ(j["order"].size(), 32u);
    CHECK_EQ(j["order"][0].get<int>(), 0);
    CHECK_EQ(j["order"][1].get<int>(), 1);
    CHECK_EQ(j["order"][2].get<int>(), 0);

    /* Where each group's declarations sit, first and last. */
    CHECK_EQ(j["ops"][0].value("first_at", -1), 0);
    CHECK_EQ(j["ops"][0].value("last_at", -1), 30);
    CHECK_EQ(j["ops"][1].value("first_at", -1), 1);
    CHECK_EQ(j["ops"][1].value("last_at", -1), 31);

    /* The model block feeds the view's header and comes from the container metadata. */
    CHECK_EQ(j["model"].value("arch", std::string()), std::string("test_arch"));
    CHECK_EQ(j["model"].value("n_layers", (int64_t)0), (int64_t)2);
    CHECK_EQ(j["model"].value("world", (int64_t)0), (int64_t)1);
}

/* Two declarations of one op that resolved differently are two groups: the band table is part of
 * the grouping key. */
TEST(the_graph_json_splits_a_group_whose_band_table_differs) {
    Program p;
    p.meta = make_meta();
    p.weights.emplace_back();
    p.buffers.emplace_back();
    p.ops.emplace_back();

    for (int i = 0; i < 2; ++i) {
        OpInfo o;
        o.op = "gemm";
        o.index = (int32_t)p.ops.size();
        Band b;
        b.hi = i ? 64 : 16;                        /* a different table on the second one */
        b.miss[RAD_DOMAIN_DEVICE] = "no kernel";
        b.miss[RAD_DOMAIN_HOST]   = "no kernel";
        o.bands.push_back(b);
        p.ops.push_back(o);
    }

    const json j = json::parse(dump_graph_json(p), nullptr, false);
    CHECK(!j.is_discarded());
    CHECK_EQ(j["ops"].size(), 2u);
    CHECK_EQ(j["ops"][0].value("count", (int64_t)0), (int64_t)1);
    CHECK_EQ(j["ops"][1].value("count", (int64_t)0), (int64_t)1);
    /* A band with no kernel on either domain reports the reason, distinguishing "not resolved"
       from "not declared". */
    CHECK(j["ops"][0]["bands"][0].contains("device_miss"));
    CHECK(j["ops"][0]["bands"][0].contains("host_miss"));
}

TEST(the_graph_json_of_an_empty_program_is_still_valid_json) {
    Program p;
    p.weights.emplace_back();
    p.buffers.emplace_back();
    p.ops.emplace_back();
    const json j = json::parse(dump_graph_json(p), nullptr, false);
    CHECK(!j.is_discarded());
    CHECK(j["ops"].is_array());
    CHECK_EQ(j["ops"].size(), 0u);
    CHECK(j["order"].is_array());
    CHECK_EQ(j["order"].size(), 0u);
    CHECK_EQ(j["counts"].value("ops", (int64_t)-1), (int64_t)0);
}

/* ================================================================== the reference-kernel gate
 *
 * libref is both the fallback that lets an unsupported model run and the correctness oracle. In
 * production the fallback is a hazard: a model silently on reference GEMMs runs orders of
 * magnitude slower. rad_gate_reference_kernels (called from engine bringup and rad-convert) makes
 * it an explicit decision; if it counted zero ops, nothing else would fail. Its message (naming
 * ops and caller) is part of its contract, so the tests read it back from stderr.
 */
namespace {

/* A library standing in for one that declared itself a reference implementation. That is decided
 * by its rows' declaration (KernelRow::reference), never its name. */
const char* const kRef = "aref";

/* Rows live in static storage because a Resolved points at one, as plugin tables are static for
 * the process. */
KernelRow& row_from(const char* plugin) {
    static std::vector<std::unique_ptr<KernelRow>> rows;
    rows.push_back(std::make_unique<KernelRow>());
    rows.back()->plugin = plugin;
    rows.back()->reference = std::strcmp(plugin, kRef) == 0;
    return *rows.back();
}

/* An op with one band resolved in the device domain by `plugin`, or unresolved when null. */
OpInfo op_on(const char* op, const char* plugin) {
    OpInfo oi;
    oi.op = op;
    oi.bands.emplace_back();
    if (plugin) oi.bands[0].dom[RAD_DOMAIN_DEVICE].row = &row_from(plugin);
    return oi;
}

Program with_ops(std::vector<OpInfo> ops) {
    Program p;
    p.ops.emplace_back();            /* the sentinel: handles are 1-based */
    for (OpInfo& o : ops) p.ops.push_back(std::move(o));
    return p;
}

}  /* namespace */

TEST(an_op_a_real_library_serves_is_not_a_reference_fallback) {
    Program p = with_ops({ op_on("rmsnorm", "libr4d"), op_on("gemm", "libr4d") });
    std::vector<std::string> ops;
    CHECK_EQ(rad_reference_ops(p, &ops), 0);
    CHECK(ops.empty());
    CHECK_OK(rad_gate_reference_kernels(p, false, "serve"));
}

/* The sentinel at index 0 has no bands and is not an op; counting it would make the gate refuse
 * every program. */
TEST(the_op_sentinel_is_not_counted_as_a_reference_fallback) {
    Program p;
    p.ops.emplace_back();
    std::vector<std::string> ops;
    CHECK_EQ(rad_reference_ops(p, &ops), 0);
    CHECK(ops.empty());
    CHECK_OK(rad_gate_reference_kernels(p, false, "serve"));
}

TEST(an_op_only_the_reference_serves_refuses_the_serve) {
    Program p = with_ops({ op_on("rmsnorm", "libr4d"),
                           op_on("gemm", kRef) });
    std::vector<std::string> ops;
    CHECK_EQ(rad_reference_ops(p, &ops), 1);
    CHECK_EQ(ops.size(), (size_t)1);
    if (!ops.empty()) CHECK_EQ(ops[0], std::string("gemm"));

    int st = RAD_OK;
    std::string msg = stderr_of([&] { st = rad_gate_reference_kernels(p, false, "serve"); });
    CHECK_EQ(st, RAD_E_NOKERNEL);
    CHECK(has(msg, "gemm"));
    CHECK(has(msg, "serve"));
    CHECK(has(msg, "--debug-accept-reference-kernels"));

    /* With the debug flag the cost is accepted deliberately; the run continues and still
     * warns. */
    msg = stderr_of([&] { st = rad_gate_reference_kernels(p, true, "serve"); });
    CHECK_OK(st);
    CHECK(has(msg, "gemm"));
    CHECK(has(msg, "not a production configuration"));
}

/* The count is per declaration (how much of the program is on the slow path); the names are
 * deduplicated, since a layered model declares the same op many times. */
TEST(the_reference_count_is_per_declaration_and_the_names_are_not) {
    std::vector<OpInfo> ops;
    for (int i = 0; i < 8; ++i) {
        ops.push_back(op_on("rmsnorm", kRef));
        ops.push_back(op_on("gemm", kRef));
    }
    Program p = with_ops(std::move(ops));
    std::vector<std::string> names;
    CHECK_EQ(rad_reference_ops(p, &names), 16);
    CHECK_EQ(names.size(), (size_t)2);
    /* Names are in declaration order. */
    if (names.size() == 2) {
        CHECK_EQ(names[0], std::string("rmsnorm"));
        CHECK_EQ(names[1], std::string("gemm"));
    }
}

/* An op counts as on the reference only if no real library serves it in any band or domain.
 * Selection already prefers the real row wherever both apply. */
TEST(a_real_kernel_in_any_band_or_domain_clears_the_op) {
    OpInfo oi = op_on("gemm", kRef);
    oi.bands.emplace_back();
    oi.bands[1].dom[RAD_DOMAIN_DEVICE].row = &row_from("libr4d");
    Program p = with_ops({ oi });
    std::vector<std::string> names;
    CHECK_EQ(rad_reference_ops(p, &names), 0);

    /* Likewise across domains: a host row from a real library clears the op. */
    OpInfo oj = op_on("gemm", kRef);
    oj.bands[0].dom[RAD_DOMAIN_HOST].row = &row_from("libr4d");
    Program q = with_ops({ oj });
    names.clear();
    CHECK_EQ(rad_reference_ops(q, &names), 0);
}

/* An op with nothing resolved is counted too; otherwise the program would clear this gate and
 * fail at the first step that issues the op. */
TEST(an_op_nothing_resolved_is_counted_rather_than_waved_through) {
    Program p = with_ops({ op_on("gemm", nullptr) });
    std::vector<std::string> names;
    CHECK_EQ(rad_reference_ops(p, &names), 1);
    int st = RAD_OK;
    (void)stderr_of([&] { st = rad_gate_reference_kernels(p, false, "serve"); });
    CHECK_EQ(st, RAD_E_NOKERNEL);
}

/* The message names the caller, so it states the consequence. */
TEST(the_gate_names_its_caller) {
    Program p = with_ops({ op_on("gemm", kRef) });
    int st = RAD_OK;
    const std::string msg =
        stderr_of([&] { st = rad_gate_reference_kernels(p, false, "the container's layout"); });
    CHECK_EQ(st, RAD_E_NOKERNEL);
    CHECK(has(msg, "the container's layout"));
}

/* Shard spans and per-part spans survive the weight list reallocating. WeightInfo's copy is
 * hand-written (it holds an atomic), so a field the copy omits would be reset on reallocation and
 * the loader would misread an uneven share. */
TEST(shard_spans_survive_the_weight_list_reallocating) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());
    /* Two weights declared and marked first, then enough more to reallocate the list several
     * times. */
    RadWeightDecl d{};
    d.dtype = RAD_BF16; d.rank = 2; d.shape[0] = 36; d.shape[1] = 64;
    d.access = RAD_ACCESS_PER_TOKEN; d.shard = RAD_SHARD_ROW; d.group = rad_group_layer(0);
    d.n_row_parts = 2; d.row_parts[0] = 18; d.row_parts[1] = 18;
    const rad_weight ab = b.decl_weight("ab", &d);
    REQUIRE(ab != 0);
    const int64_t lo[2] = { 0, 0 }, hi[2] = { 18, 18 }, tot[2] = { 48, 48 };
    CHECK_EQ(b.weight_shard_span_parts(ab, lo, hi, tot, 2), RAD_OK);

    const rad_weight sp = decl_w(b, "span", 18, 64);
    CHECK_EQ(b.weight_shard_span(sp, 6, 24), RAD_OK);

    for (int i = 0; i < 300; ++i) {
        char n[32]; snprintf(n, sizeof n, "w%d", i);
        REQUIRE(decl_w(b, n, 64, 64) != 0);
    }

    const WeightInfo* w = b.program().weight(ab);
    REQUIRE(w != nullptr);
    CHECK_EQ(w->n_part_spans, 2);
    CHECK_EQ((long long)w->part_lo[1], 0LL);
    CHECK_EQ((long long)w->part_hi[1], 18LL);
    CHECK_EQ((long long)w->part_total[1], 48LL);
    w = b.program().weight(sp);
    CHECK_EQ((long long)w->shard_lo, 6LL);
    CHECK_EQ((long long)w->shard_hi, 24LL);
    CHECK_EQ((int)b.errors().size(), 0);
}

/* ================================================================== stored forms and encodings */
namespace {

std::string extra_dir() { return home() + "/testkernels/extra"; }

/* The base and override plugins with radtest_extra behind them. */
int load_extra(Registry& r) {
    return r.load_dirs({ kernels_dir(), extra_dir() },
                       { "radtest_over", "radtest_base", "radtest_extra" });
}

#define NEED_EXTRA(reg)                                                               \
    Registry reg;                                                                     \
    if (load_extra(reg) < 0) ARTIFACT_MISSING("the test plugins in " + extra_dir());

/* A [256, 256] weight of `dtype` read by t_lay over M in [lo, hi]. */
rad_op decl_lay(Builder& b, const char* name, uint32_t dtype, int64_t lo, int64_t hi,
                rad_weight* out = nullptr) {
    RadWeightDecl d{};
    d.dtype = dtype;
    d.rank = 2;
    d.shape[0] = 256;
    d.shape[1] = 256;
    d.access = RAD_ACCESS_PER_TOKEN;
    d.group = rad_group_layer(0);
    const rad_weight w = b.decl_weight(name, &d);
    if (out) *out = w;
    const RadParam p[] = { RAD_RANGE("M", lo, hi), RAD_INT("N", 256), RAD_INT("K", 256),
                           RAD_STR("dtype", "bf16") };
    return b.decl_op("t_lay", p, 4, &w, 1);
}

bool any_error(const Builder& b, const char* needle) {
    for (const std::string& e : b.errors()) if (has(e, needle)) return true;
    return false;
}

/* Each band as "hi:row ", with its device row. */
std::string band_names_of(const OpInfo& o) {
    std::string s;
    for (const Band& b : o.bands)
        s += fmt("%lld:%s ", (long long)b.hi,
                 b.dom[RAD_DOMAIN_DEVICE].row ? b.dom[RAD_DOMAIN_DEVICE].row->info->name : "-");
    return s;
}

const char* device_row(const OpInfo* o, size_t band) {
    const Resolved& r = o->bands[band].dom[RAD_DOMAIN_DEVICE];
    return r.row ? r.row->info->name : "<none>";
}

}  /* namespace */

/* Every band of an op reads the same stored bytes. A kernel that relayouts its weight in one band
 * beside a kernel that reads it as stored in another would misread it with a matching byte count
 * (an fp8 head read as bf16 gives empty completions once batches grow), so declare refuses the
 * pair and names both kernels. */
TEST(a_weight_wanted_in_two_stored_forms_is_refused_naming_both_kernels) {
    NEED_EXTRA(reg)
    Builder b(reg, make_meta(), make_ctx());
    const rad_op h = decl_lay(b, "w.bf16", RAD_BF16, 1, 4096);
    REQUIRE(h != 0);
    REQUIRE_EQ(b.program().op(h)->bands.size(), 2u);
    CHECK_EQ(std::string(device_row(b.program().op(h), 0)), "t_lay_tiled");
    CHECK_EQ(std::string(device_row(b.program().op(h), 1)), "t_lay_plain");
    int rc = 0;
    (void)stderr_of([&] { rc = b.finish(); });
    CHECK_EQ(rc, RAD_E_SHAPE);
    CHECK(any_error(b, "two stored forms"));
    CHECK(any_error(b, "t_lay_tiled") && any_error(b, "t_lay_plain"));
}

/* A kernel whose layout hook does not read the weight's encoding is not a candidate: the band
 * falls to the next kernel, so one declaration serves a bf16 checkpoint and its fp8 container. */
TEST(a_kernel_that_does_not_read_the_encoding_gives_its_band_to_the_next) {
    NEED_EXTRA(reg)
    Builder b(reg, make_meta(), make_ctx());
    const rad_op h = decl_lay(b, "w.fp8", RAD_F8E4M3, 1, 4096);
    REQUIRE(h != 0);
    const OpInfo* o = b.program().op(h);
    REQUIRE_EQ(o->bands.size(), 2u);
    CHECK_EQ(std::string(device_row(o, 0)), "t_lay_plain");
    CHECK_EQ(std::string(device_row(o, 1)), "t_lay_plain");
    CHECK_OK(b.finish());
    CHECK(b.errors().empty());
}

/* A hookless device kernel reads one plane as stored, so a weight of codes and scales is not its
 * to read: the band misses with the reason, while a kernel whose layout joins the planes serves
 * the band it covers. A host kernel without a hook reads canonically and never conflicts with the
 * device's stored form. */
TEST(a_two_plane_weight_needs_a_kernel_that_joins_its_planes) {
    NEED_EXTRA(reg)
    auto sources = [](const std::string& name, const Program&, WeightSource* out) -> int {
        if (name != "w.q") return RAD_E_NOTFOUND;
        out->enc = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
        out->rank = 2;
        out->shape[0] = 256;
        out->shape[1] = 256;
        return RAD_OK;
    };
    {
        Builder b(reg, make_meta(), make_ctx());
        b.set_sources(sources);
        const rad_op h = decl_lay(b, "w.q", RAD_F8E4M3, 17, 64);
        REQUIRE(h != 0);                              /* the host kernel serves it */
        CHECK(!b.program().op(h)->bands[0].dom[RAD_DOMAIN_DEVICE]);
        bool named = false;
        for (const Program::Miss& m : b.program().misses)
            if (m.domain == RAD_DOMAIN_DEVICE && m.op == "t_lay" && has(m.why, "2 planes")) named = true;
        CHECK(named);
    }
    {
        Builder b(reg, make_meta(), make_ctx());
        b.set_sources(sources);
        rad_weight w = 0;
        const rad_op h = decl_lay(b, "w.q", RAD_F8E4M3, 1, 16, &w);
        REQUIRE(h != 0);
        CHECK_EQ(std::string(device_row(b.program().op(h), 0)), "t_lay_tiled");
        CHECK_OK(b.finish());
        CHECK_EQ(b.program().weight(w)->layout_tag, std::string("tiled-fp8"));
    }
    {
        Builder b(reg, make_meta(), make_ctx());
        rad_weight w = 0;
        const rad_op h = decl_lay(b, "w.bf16", RAD_BF16, 1, 16, &w);
        REQUIRE(h != 0);
        CHECK_EQ(std::string(b.program().op(h)->bands[0].dom[RAD_DOMAIN_HOST].row->info->name),
                 "t_lay_host");
        CHECK_OK(b.finish());
        CHECK_EQ(b.program().weight(w)->layout_tag, std::string("tiled"));
    }
}

/* A kernel that reads a weight's stored bytes without interpreting them (libr4d's prefetch into
 * the cache) answers RAD_LAYOUT_ANY: it takes the form another kernel chose, declared before it or
 * after, and claims none, so it never conflicts; a weight only it reads is stored as it is. */
TEST(a_kernel_that_reads_any_stored_form_claims_none) {
    NEED_EXTRA(reg)
    const RadParam p[] = { RAD_RANGE("M", 1, 16), RAD_INT("N", 256), RAD_INT("K", 256),
                           RAD_STR("dtype", "bf16") };
    auto weight = [](Builder& b) {
        RadWeightDecl d{};
        d.dtype = RAD_BF16;
        d.rank = 2;
        d.shape[0] = 256;
        d.shape[1] = 256;
        d.access = RAD_ACCESS_PER_TOKEN;
        d.group = rad_group_layer(0);
        return b.decl_weight("w.bf16", &d);
    };
    for (bool any_first : { false, true }) {
        Builder b(reg, make_meta(), make_ctx());
        const rad_weight w = weight(b);
        rad_op a = 0, t = 0;
        if (any_first) a = b.decl_op("t_any", p, 4, &w, 1);
        t = b.decl_op("t_lay", p, 4, &w, 1);
        if (!any_first) a = b.decl_op("t_any", p, 4, &w, 1);
        REQUIRE(a != 0 && t != 0);
        CHECK_EQ(std::string(device_row(b.program().op(a), 0)), "t_any_read");
        CHECK_EQ(std::string(device_row(b.program().op(t), 0)), "t_lay_tiled");
        CHECK_OK(b.finish());
        CHECK(b.errors().empty());
        CHECK_EQ(b.program().weight(w)->layout_tag, std::string("tiled"));
    }
    Builder b(reg, make_meta(), make_ctx());
    const rad_weight w = weight(b);
    REQUIRE(b.decl_op("t_any", p, 4, &w, 1) != 0);
    CHECK_OK(b.finish());
    CHECK(b.errors().empty());
    CHECK(b.program().weight(w)->layout_tag.empty());
    CHECK(b.program().weight(w)->identity);
}

namespace {

/* A [256, 256] E4M3 weight with one f32 scale per 128x128 tile, declared over both planes. */
int fp8_tiles(const std::string& name, const Program&, WeightSource* out) {
    if (name != "w.q") return RAD_E_NOTFOUND;
    out->enc = rad_enc_affine(RAD_F8E4M3, RAD_F32, 128, 128);
    out->rank = 2;
    out->shape[0] = 256;
    out->shape[1] = 256;
    return RAD_OK;
}

/* `op` over that weight for M in [lo, hi]. */
rad_op decl_two_plane(Builder& b, const char* op, int64_t lo, int64_t hi, rad_weight* out) {
    RadWeightDecl d{};
    d.dtype = RAD_F8E4M3;
    d.rank = 2;
    d.shape[0] = 256;
    d.shape[1] = 256;
    d.access = RAD_ACCESS_PER_TOKEN;
    d.group = rad_group_layer(0);
    const rad_weight w = b.decl_weight("w.q", &d);
    *out = w;
    const RadParam p[] = { RAD_RANGE("M", lo, hi), RAD_INT("N", 256), RAD_INT("K", 256),
                           RAD_STR("dtype", "fp8") };
    return b.decl_op(op, p, 4, &w, 1);
}

}  /* namespace */

/* Where no device kernel reads a weight of several planes (the host backend, or a band only the
 * host serves), the host kernel's hook decides how it is stored: the joined form of
 * abi/rad_planes.h, a header and both planes, sized here from the plane extents. */
TEST(a_host_kernel_joins_the_planes_no_device_kernel_reads) {
    NEED_EXTRA(reg)
    Builder b(reg, make_meta(), make_ctx());
    b.set_sources(fp8_tiles);
    rad_weight w = 0;
    const rad_op h = decl_two_plane(b, "t_join", 17, 64, &w);
    REQUIRE(h != 0);
    CHECK(!b.program().op(h)->bands[0].dom[RAD_DOMAIN_DEVICE]);
    CHECK_OK(b.finish());
    CHECK(b.errors().empty());
    const WeightInfo* wi = b.program().weight(w);
    CHECK_EQ(wi->layout_tag, std::string(RAD_PLANES_TAG));
    CHECK_EQ(wi->lay_dom, (int)RAD_DOMAIN_HOST);
    /* 40 + 2 * 80 header bytes to 256, the 65536 codes, the four f32 scales, each padded to 64. */
    CHECK_EQ((long long)wi->stored_bytes, 256LL + 65536 + 64);
    CHECK(!wi->identity);
}

/* On a card the device kernel's stored form stands, and a host kernel that would join the same
 * planes is no conflict: it cannot take the op, which the runtime decides when an op is placed on
 * the host (Ctx::check_domain_layout). Refusing here would stop every model whose host fallback
 * reads a weight differently from its device kernel. */
TEST(a_device_kernels_stored_form_stands_beside_a_joining_host_kernel) {
    NEED_EXTRA(reg)
    Builder b(reg, make_meta(), make_ctx());
    b.set_sources(fp8_tiles);
    rad_weight w = 0;
    const rad_op h = decl_two_plane(b, "t_join", 1, 16, &w);
    REQUIRE(h != 0);
    CHECK_EQ(std::string(device_row(b.program().op(h), 0)), "t_join_tiled");
    CHECK_OK(b.finish());
    CHECK(b.errors().empty());
    CHECK_EQ(b.program().weight(w)->layout_tag, std::string("tiled-fp8"));
    CHECK_EQ(b.program().weight(w)->lay_dom, (int)RAD_DOMAIN_DEVICE);
}

/* A host kernel without a hook reads one plane as stored. Read by nothing else, a weight of two
 * planes has nobody to say where the second lies, and the refusal names the host kernel. */
TEST(a_two_plane_weight_read_only_by_a_hookless_host_kernel_is_refused_by_name) {
    NEED_EXTRA(reg)
    Builder b(reg, make_meta(), make_ctx());
    b.set_sources(fp8_tiles);
    rad_weight w = 0;
    const rad_op h = decl_two_plane(b, "t_lay", 17, 64, &w);
    REQUIRE(h != 0);
    int rc = 0;
    (void)stderr_of([&] { rc = b.finish(); });
    CHECK_EQ(rc, RAD_E_SHAPE);
    CHECK(any_error(b, "t_lay_host"));
    CHECK(any_error(b, "rad_planes.h"));
}

/* The loader widens a bf16 plane to a declared f32 or relayouts it for a kernel, not both: a
 * widened weight read by a relayouting kernel is refused at declare. */
TEST(a_widened_weight_cannot_also_be_relayouted) {
    NEED_EXTRA(reg)
    Builder b(reg, make_meta(), make_ctx());
    b.set_sources([](const std::string& name, const Program&, WeightSource* out) -> int {
        if (name != "w.wide") return RAD_E_NOTFOUND;
        out->enc = rad_enc_plain(RAD_BF16);
        out->rank = 2;
        out->shape[0] = 256;
        out->shape[1] = 256;
        return RAD_OK;
    });
    const rad_op h = decl_lay(b, "w.wide", RAD_F32, 1, 16);
    REQUIRE(h != 0);
    int rc = 0;
    (void)stderr_of([&] { rc = b.finish(); });
    CHECK_EQ(rc, RAD_E_DTYPE);
    CHECK(any_error(b, "a widening and a relayout"));
}

/* The words a request may hand the plugin are a bound the program records; a negative or absurd
 * bound is refused, and declaring none leaves it at zero. */
TEST(the_plugin_words_bound_is_recorded_and_checked) {
    NEED_EXTRA(reg)
    Builder b(reg, make_meta(), make_ctx());
    CHECK_EQ(b.program().plugin_words, (int64_t)0);
    CHECK_OK(b.declare_plugin_words(4096));
    CHECK_EQ(b.program().plugin_words, (int64_t)4096);
    CHECK(b.declare_plugin_words(-1) < 0);
    CHECK(b.declare_plugin_words((int64_t)1 << 21) < 0);
    CHECK_EQ(b.program().plugin_words, (int64_t)4096);
}

/* The other direction, the d3 readout: an f32 checkpoint tensor declared bf16 (what the device
 * logits kernel reads) is narrowed by the loader, and the stored form is one plain bf16 plane. */
TEST(an_f32_source_declared_bf16_is_narrowed_at_load) {
    NEED_EXTRA(reg)
    Builder b(reg, make_meta(), make_ctx());
    b.set_sources([](const std::string& name, const Program&, WeightSource* out) -> int {
        if (name != "w.readout") return RAD_E_NOTFOUND;
        out->enc = rad_enc_plain(RAD_F32);
        out->rank = 2;
        out->shape[0] = 32;
        out->shape[1] = 64;
        return RAD_OK;
    });
    RadWeightDecl d{};
    d.dtype = RAD_BF16;
    d.rank = 2;
    d.shape[0] = 32;
    d.shape[1] = 64;
    d.access = RAD_ACCESS_PER_TOKEN;
    d.group = rad_group_layer(0);
    const rad_weight w = b.decl_weight("w.readout", &d);
    REQUIRE(w != 0);
    CHECK_OK(b.finish());
    CHECK(b.errors().empty());
    const WeightInfo* wi = b.program().weight(w);
    CHECK(wi->narrow);
    CHECK(!wi->widen);
    CHECK_EQ(wi->stored_dtype, (uint32_t)RAD_BF16);
    CHECK_EQ((long long)wi->stored_bytes, 32LL * 64LL * 2);
    CHECK(!weight_file_direct(*wi, 1));
}

/* ================================================================== the sizing declare */

/* A smaller arena level is declared again at a smaller max_tok and takes the real declare's bands,
 * since its kernels are the ones that run. The plugin must declare the same graph: the same op at
 * every position, and every fixed parameter unchanged, because a fixed row count that followed
 * max_tok would be launched at the real size over the smaller level's buffers. A capacity the
 * kernel only bounds its writes by may shrink, never grow. */
TEST(a_sizing_declare_must_declare_the_real_graph_with_its_fixed_parameters) {
    NEED_EXTRA(reg)
    auto cap_op = [](Builder& b, int64_t cap, int64_t max_m) {
        const RadParam p[] = { RAD_RANGE("M", 1, max_m), RAD_INT("cap", cap), RAD_INT("n", 64) };
        return b.decl_op("t_cap", p, 3, nullptr, 0);
    };
    Builder real(reg, make_meta(), make_ctx());
    const rad_weight w = decl_w(real, "w0");
    REQUIRE(decl_gemm(real, w) != 0);
    REQUIRE(decl_gemm(real, w) != 0);
    REQUIRE(cap_op(real, 1024, 8192) != 0);
    REQUIRE_EQ(real.finish(), RAD_OK);
    const Program& ref = real.program();

    RadBuildCtx small = make_ctx();
    small.max_tok = 512;
    auto probe = [&](const std::function<void(Builder&, rad_weight)>& body, std::string* errs) {
        Builder b(reg, make_meta(), small);
        b.set_reference(&ref);
        const rad_weight wp = decl_w(b, "w0");
        int rc = 0;
        *errs = stderr_of([&] { body(b, wp); rc = b.finish(); });
        return rc;
    };
    std::string e;
    const size_t asked = reg.n_selections();
    CHECK_OK(probe([&](Builder& b, rad_weight wp) {
        decl_gemm(b, wp, 512);
        decl_gemm(b, wp, 512);
        cap_op(b, 1024, 512);
        for (size_t i = 1; i < b.program().ops.size(); ++i)
            CHECK_EQ(band_names_of(b.program().ops[i]), band_names_of(ref.ops[i]));
    }, &e));
    CHECK_EQ(reg.n_selections(), asked);                       /* nothing resolved again */

    CHECK_EQ(probe([&](Builder& b, rad_weight wp) {
        decl_gemm(b, wp, 512);
        const RadParam p[] = { RAD_RANGE("M", 1, 512), RAD_INT("n", 4096), RAD_STR("dtype", "bf16") };
        b.decl_op("t_norm", p, 3, &wp, 1);
    }, &e), RAD_E_STATE);
    CHECK(has(e, "op 2 is 't_norm', and the real declare had 't_gemm'"));

    CHECK_EQ(probe([&](Builder& b, rad_weight wp) {
        const RadParam p[] = { RAD_RANGE("M", 1, 512), RAD_INT("N", 2048), RAD_INT("K", 4096),
                               RAD_STR("dtype", "bf16") };
        b.decl_op("t_gemm", p, 4, &wp, 1);
    }, &e), RAD_E_STATE);
    CHECK(has(e, "changes 'N'"));

    CHECK_OK(probe([&](Builder& b, rad_weight wp) {
        decl_gemm(b, wp, 512);
        decl_gemm(b, wp, 512);
        cap_op(b, 256, 512);
    }, &e));
    CHECK_EQ(probe([&](Builder& b, rad_weight wp) {
        decl_gemm(b, wp, 512);
        decl_gemm(b, wp, 512);
        cap_op(b, 2048, 512);
    }, &e), RAD_E_STATE);
    CHECK(has(e, "changes 'cap'"));

    CHECK_EQ(probe([&](Builder& b, rad_weight wp) {
        decl_gemm(b, wp, 512);
        decl_gemm(b, wp, 512);
        cap_op(b, 1024, 512);
        decl_gemm(b, wp, 512);
    }, &e), RAD_E_STATE);
    CHECK(has(e, "had 'nothing' there"));
}

/* ================================================================== liveness the plan cannot see */

/* Some buffers are read after the declared op list or beside it. The logits and the pooled output
 * are read by the sampler and the engine after the last op, so they live to the end. A buffer the
 * second lane uses lives for the whole program, since its declared range does not bound when the
 * lane runs, and the second lane gets a scratch region of its own. */
TEST(the_logits_pooled_and_concurrent_buffers_live_past_their_declared_uses) {
    NEED_PLUGIN(reg)
    for (int mode = 0; mode < 3; ++mode) {
        Builder b(reg, make_meta(), make_ctx());
        RadBufDecl bd{};
        bd.dtype = RAD_I8;
        bd.rank = 1;
        bd.shape[0] = 4096;
        bd.kind = RAD_BUF_TRANSIENT;
        bd.domain = RAD_DOMAIN_DEVICE;
        /* The pooled output is one f32 row a sequence. */
        RadBufDecl od = bd;
        if (mode == 1) {
            od.dtype = RAD_F32;
            od.rank = 2;
            od.shape[0] = make_ctx().max_seqs;
            od.shape[1] = 4;
        }
        const rad_buf out = b.decl_buffer("out", &od);
        const rad_buf x = b.decl_buffer("x", &bd);
        const rad_buf y = b.decl_buffer("y", &bd);
        const rad_weight w = decl_w(b, "w0");
        rad_op o[4];
        for (rad_op& h : o) h = decl_gemm(b, w);
        CHECK_OK(b.op_writes(o[0], &out, 1));
        CHECK_OK(b.op_writes(o[1], &x, 1));
        CHECK_OK(b.op_reads(o[2], &x, 1));
        CHECK_OK(b.op_writes(o[3], &y, 1));
        if (mode == 0) CHECK_OK(b.declare_logits(out));
        if (mode == 1) CHECK_OK(b.declare_pooled(out));
        if (mode == 2) CHECK_OK(b.buf_concurrent(y));
        CHECK_OK(b.finish());
        Program& p = b.program();
        const BufferInfo* bo = p.buffer(out);
        const BufferInfo* bx = p.buffer(x);
        const BufferInfo* by = p.buffer(y);
        if (mode < 2) {
            CHECK_EQ(bo->first_def, 1);
            CHECK_EQ(bo->last_use, 4);
            /* Live together, so x and y cannot reuse the output's bytes. */
            for (const BufferInfo* t : { bx, by })
                CHECK(t->arena_offset >= bo->arena_offset + bo->bytes ||
                      bo->arena_offset >= t->arena_offset + t->bytes);
            CHECK_EQ(p.scratch_regions, 1);
        } else {
            CHECK_EQ(bo->last_use, 1);
            CHECK_EQ(by->first_def, 0);
            CHECK_EQ(by->last_use, 4);
            CHECK_EQ(p.scratch_regions, 2);
            /* The second lane's region is bounded by the ops it may issue, never the larger. */
            CHECK(p.scratch_bytes_lane1 >= 0 && p.scratch_bytes_lane1 <= p.scratch_bytes);
            /* Without the extension y (op 4) would take the bytes the output (op 1) left. */
            CHECK(by->arena_offset >= bo->arena_offset + bo->bytes ||
                  bo->arena_offset >= by->arena_offset + by->bytes);
        }
    }
}

/* Under a declared window (rad_lane1_window_begin) a concurrent buffer is live across the window and
 * its own uses, not the whole program: a buffer used only past the window can take its bytes. */
TEST(a_concurrent_buffer_under_a_window_lives_across_the_window_only) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());
    RadBufDecl bd{};
    bd.dtype = RAD_I8;
    bd.rank = 1;
    bd.shape[0] = 4096;
    bd.kind = RAD_BUF_TRANSIENT;
    bd.domain = RAD_DOMAIN_DEVICE;
    const rad_buf y = b.decl_buffer("y", &bd);
    const rad_buf z = b.decl_buffer("z", &bd);
    const rad_weight w = decl_w(b, "w0");
    CHECK_OK(rad_lane1_window_begin(&b));
    rad_op o1 = decl_gemm(b, w), o2 = decl_gemm(b, w);
    CHECK_OK(rad_lane1_window_end(&b));
    rad_op o3 = decl_gemm(b, w);
    CHECK_OK(b.op_writes(o1, &y, 1));
    CHECK_OK(b.op_reads(o2, &y, 1));
    CHECK_OK(b.op_writes(o3, &z, 1));
    CHECK_OK(b.buf_concurrent(y));
    CHECK_OK(b.finish());
    Program& p = b.program();
    CHECK_EQ(p.lane1_w0, (int32_t)o1);
    CHECK_EQ(p.lane1_w1, (int32_t)o2);
    const BufferInfo* by = p.buffer(y);
    const BufferInfo* bz = p.buffer(z);
    CHECK_EQ(by->first_def, (int32_t)o1);
    CHECK_EQ(by->last_use, (int32_t)o2);
    /* z lives past the window only, so it shares y's bytes (whole program, it could not). */
    CHECK_EQ(bz->first_def, (int32_t)o3);
    CHECK(bz->arena_offset < by->arena_offset + by->bytes &&
          by->arena_offset < bz->arena_offset + bz->bytes);
}

/* A window opened and never closed is refused at finish. */
TEST(a_second_lane_window_must_be_closed) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());
    const rad_weight w = decl_w(b, "w0");
    CHECK_OK(rad_lane1_window_begin(&b));
    decl_gemm(b, w);
    b.finish();
    CHECK_EQ(b.status(), RAD_E_STATE);
}

/* ================================================================== the buffer plan, at random */

/* The packer is greedy, so its output depends on sizes, and a liveness or offset mistake aliases
 * two live activations with no crash. Over random programs: same-domain buffers live at once never
 * share bytes, the fixed region holds every persistent buffer below every transient, offsets keep
 * the sub-block alignment, the arena covers the live peak, and planning is deterministic. A
 * smaller step's level keeps the same guarantees inside the full arena. */
TEST(the_buffer_plan_never_aliases_two_live_buffers) {
    const uint64_t seed = 0x243f6a8885a308d3ull;
    uint64_t x = seed;
    auto rnd = [&x](uint64_t n) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return n ? x % n : 0; };
    std::string failures;
    auto bad = [&](int prog, const std::string& what) {
        if (failures.size() < 2000) failures += fmt("    seed %#llx program %d: %s\n",
                                                    (unsigned long long)seed, prog, what.c_str());
    };
    for (int it = 0; it < 300; ++it) {
        Program p;
        p.buffers.emplace_back();
        const int32_t n_ops = 20 + (int32_t)rnd(380);
        p.ops.resize((size_t)n_ops + 1);
        const int n = 30 + (int)rnd(170);
        for (int i = 0; i < n; ++i) {
            const uint64_t kind = rnd(10);
            BufferInfo b;
            b.name = fmt("b%d", i);
            b.decl.dtype = RAD_I8;
            b.decl.rank = 2;
            b.decl.shape[0] = 1 + (int64_t)rnd(rnd(8) == 0 ? 4096 : 64);
            b.decl.shape[1] = 1 + (int64_t)rnd(rnd(4) == 0 ? 1024 : 16);
            b.decl.domain = rnd(5) == 0 ? RAD_DOMAIN_HOST : RAD_DOMAIN_DEVICE;
            b.decl.kind = kind < 7 ? RAD_BUF_TRANSIENT : RAD_BUF_PERSIST;
            if (b.decl.kind == RAD_BUF_TRANSIENT && rnd(10) != 0) {
                b.first_def = 1 + (int32_t)rnd((uint64_t)n_ops);
                b.last_use = b.first_def + (int32_t)rnd((uint64_t)(n_ops - b.first_def + 1));
            } else if (b.decl.kind == RAD_BUF_PERSIST) {
                b.first_def = 0;                    /* as the builder sets every persistent buffer */
                b.last_use = n_ops;
            }
            p.buffers.push_back(b);
        }
        BufPlanReport rep;
        if (plan_buffers(p, &rep) != RAD_OK) { bad(it, "plan_buffers refused"); continue; }

        struct Span { int32_t lo, hi; int64_t off, end; int dom; bool persist; };
        auto spans = [&](const Program& q, const std::vector<int64_t>* offs,
                         const std::vector<RadBufDecl>* decl) {
            std::vector<Span> v;
            for (size_t i = 1; i < q.buffers.size(); ++i) {
                const BufferInfo& b = q.buffers[i];
                const RadBufDecl& d = decl ? (*decl)[i] : b.decl;
                Span s;
                s.lo = b.first_def < 0 ? 0 : b.first_def;
                s.hi = b.first_def < 0 ? n_ops : b.last_use;
                s.off = offs ? (*offs)[i] : b.arena_offset;
                s.end = s.off + align_up(buffer_bytes(d), (int64_t)RAD_ALIGN_SUB);
                s.dom = d.domain;
                s.persist = d.kind != RAD_BUF_TRANSIENT;
                v.push_back(s);
            }
            return v;
        };
        auto aliased = [](const std::vector<Span>& v) {
            for (size_t i = 0; i < v.size(); ++i)
                for (size_t j = i + 1; j < v.size(); ++j) {
                    const Span& a = v[i];
                    const Span& b = v[j];
                    if (a.dom != b.dom) continue;
                    const bool live = a.persist || b.persist || (a.lo <= b.hi && b.lo <= a.hi);
                    if (live && a.off < b.end && b.off < a.end) return true;
                }
            return false;
        };
        const std::vector<Span> s = spans(p, nullptr, nullptr);
        if (aliased(s)) bad(it, "two live buffers share bytes");
        int64_t peak = 0;
        for (const Span& a : s) {
            if (a.off % (int64_t)RAD_ALIGN_SUB != 0) bad(it, "an offset off the sub-block");
            if (a.dom == RAD_DOMAIN_HOST) {
                if (a.end > p.host_arena_bytes) bad(it, "a host buffer past the host arena");
                continue;
            }
            if (a.persist && a.end > p.arena_fixed_bytes) bad(it, "a persistent buffer above the fixed region");
            if (!a.persist && a.off < p.arena_fixed_bytes) bad(it, "a transient in the fixed region");
            if (a.end > p.arena_bytes) bad(it, "a buffer past the arena");
        }
        for (int32_t t = 0; t <= n_ops; ++t) {
            int64_t live = 0;
            for (const Span& a : s)
                if (a.dom == RAD_DOMAIN_DEVICE && !a.persist && a.lo <= t && t <= a.hi) live += a.end - a.off;
            peak = std::max(peak, live);
        }
        if (p.arena_bytes < p.arena_fixed_bytes + peak) bad(it, "the arena is below the live peak");
        if (rep.peak_live != peak) bad(it, "the report's peak disagrees with a recount");

        Program again = p;
        for (size_t i = 1; i < again.buffers.size(); ++i) again.buffers[i].arena_offset = -7;
        plan_buffers(again);
        for (size_t i = 1; i < p.buffers.size(); ++i)
            if (again.buffers[i].arena_offset != p.buffers[i].arena_offset) {
                bad(it, "a second plan moved a buffer");
                break;
            }

        for (int64_t div : { 2, 8 }) {
            Program probe = p;
            for (size_t i = 1; i < probe.buffers.size(); ++i)
                if (probe.buffers[i].decl.kind == RAD_BUF_TRANSIENT)
                    probe.buffers[i].decl.shape[0] = std::max<int64_t>(1, p.buffers[i].decl.shape[0] / div);
            ArenaLevel lv;
            std::string why;
            if (plan_level(p, probe, 1, &lv, &why) != RAD_OK) { bad(it, "plan_level: " + why); continue; }
            const std::vector<Span> ls = spans(p, &lv.offset, &lv.decl);
            if (aliased(ls)) bad(it, fmt("two live buffers share bytes at 1/%lld", (long long)div));
            if (lv.end > p.arena_bytes) bad(it, "a level past the full arena");
            for (size_t i = 1; i < p.buffers.size(); ++i) {
                const BufferInfo& b = p.buffers[i];
                if (b.decl.kind != RAD_BUF_TRANSIENT || b.decl.domain == RAD_DOMAIN_HOST) {
                    if (lv.offset[i] != b.arena_offset) bad(it, "a level moved a fixed or host buffer");
                } else if (ls[i - 1].end > lv.end) {
                    bad(it, "a level's transient past its end");
                }
            }
        }
    }
    if (!failures.empty()) fprintf(stderr, "%s", failures.c_str());
    CHECK(failures.empty());
}

/* ================================================================== $RADIANCE_HOME */

/* $RADIANCE_HOME is a ':'-separated search path, first home first; empty entries are skipped and
 * an empty path is the working directory. A file is read from the first home that has it, and
 * written to the first home when none does. */
TEST(the_home_search_path_splits_on_colons_and_the_first_home_with_a_file_wins) {
    CHECK(rad_home_dirs("a::b:") == (std::vector<std::string>{ "a", "b" }));
    CHECK(rad_home_dirs("") == (std::vector<std::string>{ "." }));
    CHECK(rad_home_dirs(":") == (std::vector<std::string>{ "." }));
    CHECK(rad_home_dirs("/x") == (std::vector<std::string>{ "/x" }));

    const std::string root = radtest::tmp_path("home_search");
    REQUIRE(mkdir(root.c_str(), 0755) == 0);
    const std::string a = root + "/a", b = root + "/b";
    REQUIRE(mkdir(a.c_str(), 0755) == 0);
    REQUIRE(mkdir(b.c_str(), 0755) == 0);
    REQUIRE(mkdir((b + "/tune").c_str(), 0755) == 0);
    FILE* f = std::fopen((b + "/tune/m.tune").c_str(), "w");
    REQUIRE(f != nullptr);
    std::fclose(f);
    const std::string path = a + ":" + b;
    CHECK_EQ(rad_home_file(path, "tune/m.tune"), b + "/tune/m.tune");
    CHECK_EQ(rad_home_file(path, "tune/none.tune"), a + "/tune/none.tune");
    CHECK_EQ(rad_home_file(b + ":" + a, "tune/m.tune"), b + "/tune/m.tune");
}

/* ================================================================== picking the architecture plugin */

/* A family has one plugin per weight format, all answering one architecture id, so the plugin is
 * selected by the (id, quantisation) pair, exactly. A checkpoint whose quantisation was imported
 * weight by weight names its producing tool, which no plugin claims: the family's one quantised
 * plugin serves it (or its only plugin), and two quantised plugins are refused by name rather than
 * chosen between. A plugin that declines imports (rad_arch_serves_imports, radtest_archq_w8) is
 * still selected by its own descriptor, but is neither counted nor picked for an import. */
TEST(the_architecture_plugin_is_picked_by_id_and_quantisation) {
    const std::string archq = home() + "/testarchq";
    NEED_ARTIFACT(archq + "/radtest_archq_fp8.so");
    const std::string root = radtest::tmp_path("arch_pick");
    REQUIRE(mkdir(root.c_str(), 0755) == 0);

    /* A home with the test kernels and the named family members. */
    int n_home = 0;
    auto make_home = [&](std::initializer_list<const char*> members) {
        const std::string h = root + fmt("/h%d", n_home++);
        mkdir(h.c_str(), 0755);
        if (symlink(kernels_dir().c_str(), (h + "/kernels").c_str()) != 0) return std::string();
        mkdir((h + "/architectures").c_str(), 0755);
        for (const char* m : members) {
            const std::string so = std::string("radtest_archq_") + m + ".so";
            if (symlink((archq + "/" + so).c_str(), (h + "/architectures/" + so).c_str()) != 0)
                return std::string();
        }
        return h;
    };
    auto pick = [&](const std::string& h, const char* quant, bool imported, LoadedPlugins* lp,
                    std::string* log) {
        rad_tools_registry().close();
        int rc = 0;
        *log = stderr_of([&] {
            rc = rad_tools_load_plugins(h, { "radtest_over", "radtest_base" }, "radtest.q", quant,
                                        lp, imported);
        });
        return rc;
    };
    const std::string all = make_home({ "plain", "fp8", "w4" });
    const std::string one = make_home({ "plain", "fp8" });
    const std::string bare = make_home({ "plain" });
    const std::string w8 = make_home({ "plain", "fp8", "w8" });
    const std::string plain_w8 = make_home({ "plain", "w8" });
    REQUIRE(!all.empty() && !one.empty() && !bare.empty() && !w8.empty() && !plain_w8.empty());

    LoadedPlugins lp;
    std::string log;
    for (const char* q : { "", "fp8", "w4" }) {
        CHECK_OK(pick(all, q, false, &lp, &log));
        CHECK_EQ(lp.arch_plugin, std::string("radtest_archq_") + (*q ? q : "plain"));
        CHECK_EQ(lp.arch_quant, std::string(q));
        CHECK_EQ(lp.kernels.size(), 2u);
    }
    CHECK_EQ(pick(all, "int3", false, &lp, &log), RAD_E_NOTFOUND);
    CHECK(has(log, "is served at"));
    CHECK(has(log, "fp8") && has(log, "w4"));
    CHECK_OK(pick(one, "quark-mxfp4", true, &lp, &log));
    CHECK_EQ(lp.arch_plugin, std::string("radtest_archq_fp8"));
    CHECK_EQ(lp.arch_quant, std::string("fp8"));
    CHECK_EQ(pick(all, "quark-mxfp4", true, &lp, &log), RAD_E_NOTFOUND);
    CHECK(has(log, "more than one quantised plugin (fp8, w4)") ||
          has(log, "more than one quantised plugin (w4, fp8)"));
    CHECK_OK(pick(bare, "quark-mxfp4", true, &lp, &log));
    CHECK_EQ(lp.arch_plugin, std::string("radtest_archq_plain"));
    CHECK_EQ(pick(all, "fp8", false, &lp, &log), RAD_OK);

    /* A second quantised plugin that declines imports leaves them to the first. */
    CHECK_OK(pick(w8, "quark-mxfp4", true, &lp, &log));
    CHECK_EQ(lp.arch_plugin, std::string("radtest_archq_fp8"));
    CHECK(!has(log, "more than one quantised plugin"));
    CHECK_OK(pick(w8, "w8", false, &lp, &log));
    CHECK_EQ(lp.arch_plugin, std::string("radtest_archq_w8"));
    CHECK_EQ(lp.arch_quant, std::string("w8"));
    CHECK_OK(pick(plain_w8, "quark-mxfp4", true, &lp, &log));
    CHECK_EQ(lp.arch_plugin, std::string("radtest_archq_plain"));
    rad_tools_registry().close();
}

RAD_TEST_MAIN()
