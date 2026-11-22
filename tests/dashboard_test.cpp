/* tests/dashboard_test.cpp -- what the dashboard and its JSON routes show.
 *
 * The page at `/` and the process's sampler read dozens of the fields of /stats and /sessions by
 * name; a field renamed on either side blanks a panel with no error anywhere. So the routes are
 * checked for the values a populated engine gives them, for every field the page reads of them
 * and for every figure the sampler derives from them. The KV figures behind both come from
 * SchedulerBridge's aggregation over the scheduler's groups, checked here on a hybrid model's
 * groups, where summing or picking the wrong group goes wrong silently.
 */
#include "server_fakes.h"
#include "rad_internal.h"   /* rad_mono_ns: a synthetic bringup start */
#include "server/dashboard_page.h"
#include "server/history.h"

#include <cmath>
#include <functional>
#include <map>
#include <set>

namespace {

/* A scheduler with a live request table, stored sessions and idle-tier counters. */
struct StatsScheduler : FakeScheduler {
    int requests(std::vector<RequestStat>* out) const override {
        RequestStat r;
        r.id = 41;
        r.state = "decode";
        r.prompt_tokens = 900;
        r.computed_tokens = 900;
        r.cached_tokens = 512;
        r.output_tokens = 33;
        r.max_tokens = 4096;
        r.ctx_tokens = 933;
        r.kv_blocks = 120;
        r.kv_tokens = 960;
        r.n_draft = 3;
        r.slot = 2;
        r.age_s = 4.5;
        r.ttft_s = 0.25;
        r.prefill_s = 0.20;
        r.decode_tps = 61.0;
        out->assign(1, r);
        return 1;
    }
    int sessions(std::vector<SessionStat>* out, SessionOccupancy* occ) const override {
        SessionStat s;
        s.id = 7;
        s.model = "m";
        s.label = "conv-7";
        s.n_tokens = 5000;
        s.bytes = 1 << 20;
        s.idle_s = 12.0;
        s.age_s = 300.0;
        s.on_device = 40;
        s.on_host = 10;
        s.snapshots = 2;
        s.snap_bytes = 4096;
        s.resume_tokens = 4096;
        s.moving = true;
        out->assign(1, s);
        occ->host_bytes = 8 << 20;
        occ->host_cap_bytes = 64 << 20;
        return 1;
    }
    /* The one stored conversation is "conv-7" (engine id 7); "busy" has a request in flight. */
    std::vector<std::pair<std::string, int64_t>> released;
    int release_session(const std::string& label, uint64_t id, int64_t keep_tokens,
                        SessionReleased* out) override {
        if (label == "busy") return RAD_E_STATE;
        if (label != "conv-7" && id != 7) return RAD_E_NOTFOUND;
        released.push_back({ label.empty() ? "id" : label, keep_tokens });
        out->sessions = 1;
        out->entries = 48 - keep_tokens / 16;
        out->tokens = out->entries * 16;
        out->device_bytes = out->entries * 4096;
        out->host_bytes = out->entries * 2048;
        out->snapshots = 2;
        out->kept_tokens = keep_tokens;
        out->kept_shared = 2;
        return RAD_OK;
    }
    TierCounters tier_counters() const override {
        TierCounters c;
        c.to_host = 3; c.dropped = 2; c.promoted = 1; c.host_drops = 4; c.failures = 0;
        c.host_bytes = 8 << 20; c.ck_bytes = 2048; c.ck_slots = 16;
        c.restores = 5; c.restore_ns = 2000000; c.flushes = 6; c.flush_ns = 3000000;
        c.host_on = true;
        return c;
    }
};

/* Every value a snapshot carries, set, so every optional field is emitted. */
void populated(LiveSnapshot& s) {
    CardStat c;
    c.index = 0;
    c.name = "R9700";
    c.util = 0.5;
    c.vram_used = 10LL << 30;
    c.vram_total = 32LL << 30;
    c.power_w = 200;
    c.temp_c = 60;
    s.cards = { c };
    s.experts.n_layers = 2;
    s.experts.n_experts = 3;
    s.experts.tier = { 0, 1, 2, 0, 0, 1 };
    for (int t = 0; t < kLiveTiers; ++t) { s.experts.units[t] = t + 1; s.experts.bytes[t] = (t + 1) << 20; }
    s.experts.n_units = 6;
    s.experts.total_bytes = 6 << 20;
    s.experts.movable = 4;
    s.experts.layer_units = { 3, 3 };
    s.experts.layer_resident = { 2, 3 };
    s.experts.promotions = 9; s.experts.demotions = 8; s.experts.readmits = 2;
    s.experts.distinct_promoted = 7; s.experts.dispatches = 20; s.experts.refused = 3;
    s.experts.no_candidate = 1; s.experts.routed = 100; s.experts.routed_resident = 80;
    s.experts.flex_bytes = 1 << 20; s.experts.flex_capacity = 2 << 20;
    s.experts.heat_enabled = true;
    s.link.present = true;
    s.link.world = 2;
    s.link.ar_calls = 50; s.link.ar_bytes = 5000; s.link.h2d_bytes = 900; s.link.d2h_bytes = 800;
    s.link.ssd_reads = 4; s.link.ssd_bytes = 400; s.link.stream_bytes = 700;
    s.link.prefetched = 2; s.link.prefetch_bytes = 300; s.link.staged_bytes = 200;
    s.link.kv_streaming = true; s.link.kv_stream_active = true; s.link.kv_stream_bytes = 600;
    s.passes.present = true;
    s.passes.played = 10; s.passes.recorded = 2;
}

struct Dash {
    StatsScheduler sched;
    FakeTokenizer  tok;
    std::unique_ptr<Server> srv;
    explicit Dash(bool with_graph) {
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        d.live = populated;
        ServerOptions o;
        o.model_id = "dash";
        o.max_seqs = 4;
        if (with_graph) o.graph_json = [] { return std::string(R"({"order":[]})"); };
        sched.m.running = 1;
        sched.m.waiting = 2;
        sched.m.kv_tokens_used = 1000;
        sched.m.kv_tokens_total = 4000;
        sched.m.kv_tokens_cached = 300;
        sched.m.kv_tokens_max = 6000;
        sched.m.prefix_ckpt_lru = 3;
        sched.m.hit_cut_no_snapshot = 2;
        sched.m.hit_cut_tokens = 9000;
        sched.m.hit_cut_resumed_below = 4;
        sched.m.prefix_restore_cuts = 1;
        sched.m.draft_depth_seen = 2;
        sched.m.draft_pos_reached[0] = 10; sched.m.draft_pos_accepted[0] = 8;
        sched.m.draft_pos_reached[1] = 8;  sched.m.draft_pos_accepted[1] = 2;
        sched.m.encode_items = 3;
        sched.m.encode_patches = 12000;
        sched.m.encode_ns = 1500000000;
        sched.m.encode_pending = 1;
        srv.reset(new Server(d, o));
    }
    json get(const char* path) {
        HttpResponse r = srv->router().call("GET", path);
        CHECK_EQ(r.status, 200);
        return json::parse(r.body, nullptr, false);
    }
};

void each_field_is_emitted(const std::set<std::string>& names, const json& o, const char* what) {
    for (const std::string& n : names) {
        const bool there = o.is_object() && o.contains(n);
        CHECK(there);
        if (!there) fprintf(stderr, "    the page reads %s.%s, which the route does not emit\n", what, n.c_str());
    }
}

}  /* namespace */

/* ================================================================== the routes */

/* /stats carries the scheduler's figures, the live request table, the draft positions with their
 * reach, and the engine's snapshot (cards, experts by named tier, link, passes); /graph is the
 * provider's JSON, or a 501 naming the missing provider. */
TEST(stats_carries_the_engine_the_requests_and_the_snapshot) {
    Dash d(false);
    const json s = d.get("/stats");
    CHECK_EQ(s.value("model", ""), std::string("dash"));
    CHECK_EQ(s.value("queue_cap", (int64_t)0), (int64_t)32);
    CHECK_EQ(s.value("in_flight", (int64_t)-1), (int64_t)0);
    CHECK_EQ(s.value("running", (int64_t)0), (int64_t)1);
    CHECK_EQ(s.value("kv_tokens_cached", (int64_t)0), (int64_t)300);
    CHECK_EQ(s.value("kv_tokens_max", (int64_t)0), (int64_t)6000);
    REQUIRE(s["draft_positions"].size() == 2);
    CHECK_NEAR(s["draft_positions"][1]["rate"].get<double>(), 0.25, 1e-12);
    CHECK_EQ(s["draft_positions"][1]["reached"].get<int64_t>(), (int64_t)8);
    REQUIRE(s["requests"].size() == 1);
    const json& r = s["requests"][0];
    CHECK_EQ(r.value("state", ""), std::string("decode"));
    CHECK_EQ(r.value("kv_tokens", (int64_t)0), (int64_t)960);
    CHECK_NEAR(r.value("ttft_s", -1.0), 0.25, 1e-12);
    CHECK_NEAR(r.value("prefill_s", -1.0), 0.20, 1e-12);
    REQUIRE(s["cards"].size() == 1);
    CHECK_EQ(s["cards"][0].value("name", ""), std::string("R9700"));
    REQUIRE(s["experts"]["tiers"].size() == (size_t)kTierCount);
    for (int t = 0; t < kTierCount; ++t) {
        CHECK_EQ(s["experts"]["tiers"][t].value("name", ""), std::string(tier_name((Tier)t)));
        CHECK_EQ(s["experts"]["tiers"][t].value("units", (int64_t)0), (int64_t)(t + 1));
    }
    CHECK_EQ(s["link"].value("stream_bytes", (int64_t)0), (int64_t)700);
    CHECK_EQ(s["link"].value("kv_stream_bytes", (int64_t)0), (int64_t)600);
    CHECK(s["link"].value("kv_streaming", false));
    CHECK(s["link"].value("kv_stream_active", false));
    CHECK_EQ(s["passes"].value("played", (int64_t)0), (int64_t)10);
    CHECK_EQ(s["ckpt_evictions"].value("store_full", (int64_t)0), (int64_t)3);
    CHECK_EQ(s["hit_cuts"].value("no_snapshot", (int64_t)0), (int64_t)2);
    CHECK_EQ(s["hit_cuts"].value("resumed_below", (int64_t)0), (int64_t)4);
    CHECK_EQ(s.value("hit_cut_tokens", (int64_t)0), (int64_t)9000);
    CHECK_EQ(s.value("encode_items", (int64_t)0), (int64_t)3);
    CHECK_EQ(s.value("encode_patches", (int64_t)0), (int64_t)12000);
    CHECK_EQ(s.value("encode_ns", (int64_t)0), (int64_t)1500000000);
    CHECK_EQ(s.value("encode_pending", (int64_t)0), (int64_t)1);

    CHECK_EQ(d.srv->router().call("GET", "/graph").status, 501);
    Dash g(true);
    CHECK_EQ(g.get("/graph").value("order", json()).size(), (size_t)0);
}

/* /sessions lists each stored conversation with where its blocks are, the pool as the card holds
 * it (each block once), the tiers' counters and the snapshot store. */
TEST(sessions_lists_each_conversation_and_what_the_tiers_did) {
    Dash d(false);
    const json s = d.get("/sessions");
    REQUIRE(s["sessions"].size() == 1);
    const json& e = s["sessions"][0];
    CHECK_EQ(e.value("label", ""), std::string("conv-7"));
    CHECK_EQ(e.value("on_device", (int64_t)0), (int64_t)40);
    CHECK_EQ(e.value("resume_tokens", (int64_t)0), (int64_t)4096);
    CHECK_EQ(s["blocks_sum_overlapping"].value("host", (int64_t)0), (int64_t)10);
    CHECK_EQ(s["pool"].value("tokens_used", (int64_t)0), (int64_t)1000);
    CHECK_EQ(s["pool"].value("unshared_tokens", (int64_t)0), (int64_t)300);
    CHECK_EQ(s["moved"].value("to_host", (int64_t)0), (int64_t)3);
    CHECK_EQ(s["snapshot_store"].value("bytes_each", (int64_t)0), (int64_t)2048);
    CHECK_NEAR(s["stall"].value("restore_ms", 0.0), 2.0, 1e-9);
    CHECK_EQ(s["occupancy"].value("host_cap_bytes", (int64_t)0), (int64_t)(64 << 20));
    CHECK(s.value("enabled", false));
    CHECK_EQ(s["linear"]["evictions"].value("store_full", (int64_t)0), (int64_t)3);
    CHECK_EQ(s["linear"]["hit_cuts"].value("no_snapshot", (int64_t)0), (int64_t)2);
    CHECK_EQ(s["linear"]["hit_cuts"].value("resumed_below", (int64_t)0), (int64_t)4);
    CHECK_EQ(s["moved"].value("restore_cuts", (int64_t)0), (int64_t)1);
    CHECK_EQ(s["linear"].value("hit_cut_tokens", (int64_t)0), (int64_t)9000);
}

/* DELETE /sessions forgets one stored conversation by its label or engine id and answers with
 * what went and what stayed; an unknown session is 404, one with a request in flight 409, a
 * malformed count 400, and an engine without a prefix cache 501. */
TEST(a_session_is_released_by_label_or_id_and_the_answer_says_what_went) {
    Dash d(false);
    HttpResponse r = d.srv->router().call("DELETE", "/sessions/conv-7");
    CHECK_EQ(r.status, 200);
    json o = json::parse(r.body, nullptr, false);
    CHECK_EQ(o.value("label", ""), std::string("conv-7"));
    CHECK_EQ(o.value("entries", (int64_t)0), (int64_t)48);
    CHECK_EQ(o.value("tokens", (int64_t)0), (int64_t)768);
    CHECK_EQ(o.value("snapshots", (int64_t)0), (int64_t)2);
    CHECK_EQ(o["kept"].value("shared", (int64_t)0), (int64_t)2);
    CHECK_EQ(o["kept"].value("tokens", (int64_t)0), (int64_t)0);
    REQUIRE_EQ(d.sched.released.size(), 1u);
    CHECK_EQ(d.sched.released[0].second, (int64_t)0);

    /* keep_tokens reaches the scheduler and comes back in the answer. */
    r = d.srv->router().call("DELETE", "/sessions/conv-7?keep_tokens=160");
    CHECK_EQ(r.status, 200);
    o = json::parse(r.body, nullptr, false);
    CHECK_EQ(o.value("keep_tokens", (int64_t)0), (int64_t)160);
    CHECK_EQ(o.value("entries", (int64_t)0), (int64_t)38);
    CHECK_EQ(o["kept"].value("tokens", (int64_t)0), (int64_t)160);
    REQUIRE_EQ(d.sched.released.size(), 2u);
    CHECK_EQ(d.sched.released[1].second, (int64_t)160);

    /* By engine id, and by a label the path cannot carry. */
    r = d.srv->router().call("DELETE", "/sessions?id=7");
    CHECK_EQ(r.status, 200);
    o = json::parse(r.body, nullptr, false);
    CHECK_EQ(o.value("id", (int64_t)0), (int64_t)7);
    CHECK(!o.contains("label"));
    r = d.srv->router().call("DELETE", "/sessions?label=conv-7");
    CHECK_EQ(r.status, 200);

    CHECK_EQ(d.srv->router().call("DELETE", "/sessions/conv-8").status, 404);
    CHECK_EQ(d.srv->router().call("DELETE", "/sessions?id=8").status, 404);
    CHECK_EQ(d.srv->router().call("DELETE", "/sessions/busy").status, 409);
    CHECK_EQ(d.srv->router().call("DELETE", "/sessions").status, 400);
    CHECK_EQ(d.srv->router().call("DELETE", "/sessions?id=seven").status, 400);
    CHECK_EQ(d.srv->router().call("DELETE", "/sessions/conv-7?keep_tokens=-1").status, 400);
    CHECK_EQ(d.sched.released.size(), 4u);

    /* The stored-sessions view carries the counters beside the expiry ones. */
    const json s = d.get("/sessions");
    CHECK(s.contains("released"));
    CHECK(s["released"].contains("entries"));
    CHECK(s["released"].contains("snapshots"));
}

TEST(a_session_release_without_a_prefix_cache_is_not_implemented) {
    FakeScheduler sched;      /* the base: no release_session */
    FakeTokenizer tok;
    Deps dd;
    dd.sched = &sched;
    dd.tok = &tok;
    ServerOptions o;
    o.model_id = "dash";
    Server srv(dd, o);
    CHECK_EQ(srv.router().call("DELETE", "/sessions/conv-7").status, 501);
}

/* ================================================================== the page's field contract */

/* Every field the page's script reads from /stats (in its update and the request table) and from
 * /sessions (its loader and the session rows) is one the routes emit for a populated engine. A
 * rename on either side otherwise blanks a panel and reports nothing. */
/* /stats reports the measured startup time, the cost side of any decision to restart; the page
 * does not read it. A test starts in microseconds, where a real note_ready() would compute 0 (the
 * same as "never called"), so a synthetic start exercises the arithmetic through the real route. */
TEST(stats_reports_how_long_bringup_took_once_the_server_listens) {
    Dash d(false);
    /* Before the server has listened, there is no measured cost to report. */
    CHECK_EQ(d.get("/stats").value("startup_s", (int64_t)-1), (int64_t)0);

    d.srv->note_ready(rad_mono_ns() - 47ull * 1000000000ull);
    CHECK_EQ(d.get("/stats").value("startup_s", (int64_t)-1), (int64_t)47);

    /* A start in the future is not a negative cost: the figure is a duration, and a page that
     * divided by it would otherwise be handed a sign flip. */
    d.srv->note_ready(rad_mono_ns() + 10ull * 1000000000ull);
    CHECK_EQ(d.get("/stats").value("startup_s", (int64_t)-1), (int64_t)47);

    /* 0 means "not yet", so it must not overwrite a measured figure. */
    d.srv->note_ready(0);
    CHECK_EQ(d.get("/stats").value("startup_s", (int64_t)-1), (int64_t)47);
}

/* The page is built from typed sources: every field it reads of a route's object is a member of an
 * interface in dashboard/src/lib/api.ts, and the compiled-in header lists them (kPageReads). Each
 * must be one the route emits for a populated engine. */
TEST(every_field_the_page_reads_is_one_the_routes_emit) {
    Dash d(false);
    HttpResponse page = d.srv->router().call("GET", "/");
    REQUIRE(page.status == 200 && has(page.body, "<div id=\"root\">"));
    const json stats = d.get("/stats");
    const json sessions = d.get("/sessions");
    const std::map<std::string, const json*> objects = {
        { "stats", &stats },
        { "requests", &stats["requests"][0] },
        { "draft_positions", &stats["draft_positions"][0] },
        { "experts", &stats["experts"] },
        { "tiers", &stats["experts"]["tiers"][0] },
        { "sessions", &sessions },
        { "session", &sessions["sessions"][0] },
    };
    std::map<std::string, std::set<std::string>> reads;
    for (const char* const* r = kPageReads; *r; ++r) {
        const std::string e = *r;
        const size_t dot = e.find('.');
        REQUIRE(dot != std::string::npos);
        reads[e.substr(0, dot)].insert(e.substr(dot + 1));
    }
    CHECK(reads["stats"].size() > 12);
    for (const auto& [object, names] : reads) {
        const auto it = objects.find(object);
        CHECK(it != objects.end());
        if (it != objects.end()) each_field_is_emitted(names, *it->second, object.c_str());
    }
}

/* The charts draw what the process samples (History), which reads /stats by field name as the page
 * does. Fed a populated engine's /stats and an earlier one with every counter lower, the sampler
 * must yield each figure a chart draws; a field the route renamed would leave its column out. */
TEST(the_sampler_yields_every_charted_figure_from_the_stats_route) {
    Dash d(false);
    const json now = d.get("/stats");
    std::function<json(const json&)> earlier = [&](const json& j) -> json {
        if (j.is_number()) return json(j.get<double>() - 7.0);
        if (j.is_object()) { json o = json::object(); for (auto it = j.begin(); it != j.end(); ++it) o[it.key()] = earlier(it.value()); return o; }
        if (j.is_array()) { json a = json::array(); for (const json& e : j) a.push_back(earlier(e)); return a; }
        return j;
    };
    json prev = earlier(now);
    prev["started_at"] = now["started_at"];
    const History::Row row = History::model_row(now, &prev, 1.0);
    std::set<std::string> have;
    for (const auto& [name, v] : row)
        if (std::isfinite(v)) have.insert(name);
    for (const char* k : { "decode", "prefill", "running", "waiting", "preempted", "flight", "hit", "kvUsed", "kvCap",
                           "kvKept", "kvRead", "jobRun", "jobDone", "prefillRate", "reqRate", "stepAny", "step",
                           "accept", "tokStep", "lkAr", "lkUp", "lkDn", "lkSsd", "lkStream", "expHit", "expIn",
                           "expOut", "encBusy", "encRate", "encPatch", "encWait" }) {
        CHECK(have.count(k) == 1);
        if (!have.count(k)) fprintf(stderr, "    the sampler yields no '%s' from /stats\n", k);
    }
}

/* ================================================================== the bridge's KV figures */

/* SchedulerBridge's aggregation over a hybrid model's groups: two paged groups share one token
 * axis, so the token figures are the fuller one's, not their sum; a linear state has no token
 * axis and adds only bytes; a windowed group's carve bounds lookback, not fit, so it neither
 * binds the token figures (a drafter's window sits full for the server's life) nor bounds
 * kv_tokens_max. Utilisation is the maximum over every group, bytes their sum, the cache's
 * unshared share is clamped to what is used, and the draft depth to the table's size. */
TEST(the_bridge_reads_the_binding_paged_group_and_sums_only_bytes) {
    rad::SchedMetrics m;
    m.n_kv_groups = 4;
    auto group = [&](int i, int64_t bt, int64_t total, int64_t free_, bool window, int64_t bytes) {
        m.kv[i].block_tokens = bt;
        m.kv[i].blocks_total = total;
        m.kv[i].blocks_free = free_;
        m.kv[i].utilisation = total ? (float)(1.0 - (double)free_ / (double)total) : 0.0f;
        m.kv[i].window = window;
        m.kv[i].bytes_total = bytes;
        m.kv[i].bytes_used = bytes / 2;
        m.kv[i].bytes_carved = bytes * 2;
    };
    group(0, 16, 100, 40, false, 1000);   /* attention, 60% */
    group(1, 16, 100, 10, false, 200);    /* a second paged group on the same axis, 90% */
    group(2, 0, 8, 0, false, 500);        /* linear state, full, no token axis */
    group(3, 16, 4, 0, true, 64);         /* a drafter's window, always full */
    m.prefix_held_tokens = 5000;
    m.hit_cut_resumed_below = 6;
    m.prefix_restore_cuts = 7;
    m.draft_depth_seen = 70;
    for (int j = 0; j < 64; ++j) { m.draft_pos_reached[j] = 100 - j; m.draft_pos_accepted[j] = j; }

    const server::SchedMetrics o = bridge_metrics(m);
    CHECK_NEAR(o.kv_util, 1.0, 1e-6);
    CHECK_EQ(o.kv_tokens_total, (int64_t)(100 * 16));
    CHECK_EQ(o.kv_tokens_used, (int64_t)(90 * 16));
    CHECK_EQ(o.kv_tokens_max, (int64_t)(100 * 16));
    CHECK_EQ(o.kv_bytes_total, (int64_t)(1000 + 200 + 500 + 64));
    CHECK_EQ(o.kv_bytes_used, (int64_t)(500 + 100 + 250 + 32));
    CHECK_EQ(o.kv_bytes_carved, (int64_t)(2000 + 400 + 1000 + 128));
    CHECK_EQ(o.kv_tokens_cached, o.kv_tokens_used);
    CHECK_EQ(o.draft_depth_seen, server::SchedMetrics::MAX_DRAFT_POS);
    CHECK_EQ(o.draft_pos_reached[63], (int64_t)37);
    CHECK_EQ(o.hit_cut_resumed_below, (int64_t)6);
    CHECK_EQ(o.prefix_restore_cuts, (int64_t)7);

    /* The fuller group binds wherever it sits in the list. */
    group(0, 16, 200, 10, false, 1000);
    const server::SchedMetrics p = bridge_metrics(m);
    CHECK_EQ(p.kv_tokens_total, (int64_t)(200 * 16));
    CHECK_EQ(p.kv_tokens_used, (int64_t)(190 * 16));
    CHECK_EQ(p.kv_tokens_max, (int64_t)(100 * 16));
}

RAD_TEST_MAIN()
