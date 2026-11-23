/* Actual engine ownership boundaries on the CPU backend. A closed host gate makes a fence
 * fail deterministically; opening it tests recovery without a GPU, a timeout or a soak. */
#include "rad_test.h"
#include "engine_priv.h"
#include "place/expert_memory.h"
#include "gen/pass.h"
#include <chrono>
#include <sys/wait.h>
#include <unistd.h>

namespace rad {
struct EngineMemoryTestAccess {
    static int init(Engine& e, int n = 2) {
        for (int i = 0; i < n; ++i) {
            auto r = std::make_unique<Engine::Rank>();
            r->index = i;
            CtxDesc d;
            d.program = &r->program;
            RAD_TRY(r->ctx.init(d));
            e.ranks_.push_back(std::move(r));
        }
        return RAD_OK;
    }
    static RadStream stream(Engine& e, int rank) { return e.ranks_[rank]->ctx.stream(); }
    static int generate(Engine& e, std::unique_ptr<gen::IPassDriver> driver) {
        e.gen_ = std::move(driver); return e.gen_loop();
    }
    static int drain(Engine& e) { return e.drain_streams(); }
    static void hold(Engine& e) { e.hold_req_.store(true); e.hold_status_.store(RAD_OK); e.hold_point(); }
    static void clear_hold(Engine& e) { e.hold_release(); }
    static bool acknowledged(Engine& e) { return e.hold_ack_.load(); }
    static int hold_status(Engine& e) { return e.hold_status_.load(); }
    static int quiet(Engine& e, const std::function<int()>& fn) {
        e.quiet_fn_ = &fn; e.quiet_done_ = false; e.quiet_pending_.store(true);
        e.quiet_service(); e.quiet_pending_.store(false); e.quiet_fn_ = nullptr;
        return e.quiet_rc_;
    }
    static int flex(Engine& e) {
        e.flex_pending_.store(true); e.flex_done_ = false; e.flex_status_ = RAD_OK;
        e.flex_want_ = 4096; e.flex_service(); e.flex_pending_.store(false);
        return e.flex_status_;
    }
    static bool flex_done(Engine& e) { return e.flex_done_; }
    static int deferred(Engine& e) { return e.release_deferred_quietly(); }
    static int tier_stream(Engine& e, int rank) {
        auto& t = e.ranks_[rank]->tierx;
        t.enabled_ = true; t.device_ = 0; t.own_stream_ = true;
        return rad_stream_create(&t.stream_, 0);
    }
    static RadStream tier(Engine& e, int rank) { return e.ranks_[rank]->tierx.stream_; }
    static int leader_blocks(Engine& e, std::vector<int32_t>* blocks) {
        auto& rank = *e.ranks_[0];
        Config config;
        config.vram_weights_mib = 1; config.vram_kv_mib = 1;
        config.max_ctx = 64; config.max_seqs = 1; config.checkpoint_slots = 0;
        auto& allocator = rad_malloc_allocator();
        RAD_TRY(rank.pools.configure(config, 0, &allocator, &allocator));
        std::vector<KVGroupInfo> groups(2);
        auto& group = groups[1]; group.name = "guard-test";
        group.decl.kind = RAD_KV_FULL; group.decl.dtype = RAD_BF16;
        group.decl.n_head_kv = 1; group.decl.head_dim = 8;
        group.block_size = 16; group.layers = {0};
        RAD_TRY(rank.kv.configure(groups, config, rank.pools.kv()));
        if (rank.kv.reserve_blocks(1, 2, blocks) != 2) return RAD_E_STATE;
        auto& tier = rank.tierx;
        tier.leader_ = true; tier.kv_ = &rank.kv; tier.tiers_ = &rank.tiers;
        tier.gi_.resize(1); tier.gi_[0].g = 1;
        return RAD_OK;
    }
    static int64_t free_blocks(Engine& e) { return e.ranks_[0]->kv.free_blocks(1); }

    static VMem& backing(Engine& e, int rank) { return e.ranks_[rank]->xflex; }
    static void broker(Engine& e, VramBroker* broker) { e.broker_ = broker; }
    static int64_t shrink(Engine& e, int64_t bytes) { return e.xflex_give(bytes); }
    static void tick(Engine& e) { e.xflex_cap_ = e.ranks_[0]->xflex.size(); e.xflex_tick(true); }
    static void producer_state(Engine& e, bool stopped, bool stopping) {
        e.producers_stopped_.store(stopped); e.stopping_.store(stopping);
    }
    static int grow(Engine& e, int64_t bytes, ExpertMemoryOps& mappings) {
        return e.xflex_grow_to(bytes, mappings);
    }
    static void idle_tick(Engine& e) {
        e.xflex_cap_ = e.ranks_[0]->xflex.size();
        e.xflex_ns_ = 0;
        e.xflex_busy_ns_ = rad_mono_ns() - 6000000000ull;
        e.xflex_retry_ns_ = 0;
        e.xflex_tick(false);
    }
    static int64_t eviction_demand(Engine& e) { return e.xflex_want_.load(); }

    static int64_t charge(Engine& e) { return e.xflex_bytes_.load(); }
    static int64_t loan(Engine& e, int rank) {
        return e.ranks_[rank]->mover.loan_offered(Mover::kLendFleet);
    }
    static void set_loan(Engine& e, int64_t bytes) {
        for (auto& rank : e.ranks_) rank->mover.set_loan(Mover::kLendFleet, bytes);
        e.xflex_settle();
    }
    static int report(Engine& e, std::vector<TierJob>& jobs) {
        return e.report_tier_jobs(jobs, std::vector<char>(jobs.size(), 0), 1);
    }
};
}
using rad::Engine;
using Access = rad::EngineMemoryTestAccess;
struct Gate {
    void* gate = nullptr;
    explicit Gate(RadStream stream) { CHECK_OK(rad_stream_gate(stream, &gate)); }
    void open() { if (gate) { CHECK_OK(rad_gate_open(gate)); gate = nullptr; } }
    ~Gate() { open(); }
};
TEST(failed_fence_prevents_park_ack_quiet_flex_and_unload) {
    Engine e(rad::Config{});
    REQUIRE_EQ(Access::init(e), RAD_OK);
    Gate gate(Access::stream(e, 1));
    CHECK_EQ(Access::drain(e), RAD_E_STATE);
    CHECK_EQ(e.memory_fence_status(), RAD_E_STATE);
    const auto begin = std::chrono::steady_clock::now();
    Access::hold(e);
    CHECK(!Access::acknowledged(e));
    CHECK_EQ(Access::hold_status(e), RAD_E_STATE);
    CHECK(std::chrono::steady_clock::now() - begin < std::chrono::seconds(1));
    Access::clear_hold(e);
    bool ran = false;
    CHECK_EQ(Access::quiet(e, [&] { ran = true; return RAD_OK; }), RAD_E_STATE);
    CHECK(!ran);
    CHECK_EQ(Access::flex(e), RAD_E_STATE);
    CHECK(Access::flex_done(e));
    CHECK_EQ(Access::deferred(e), RAD_E_STATE);
    CHECK_EQ(e.prepare_unload(), RAD_E_STATE);
    gate.open();
    CHECK_OK(Access::drain(e));
    CHECK_EQ(e.memory_fence_status(), RAD_OK);
    CHECK_OK(Access::quiet(e, [&] { ran = true; return RAD_OK; }));
    CHECK(ran);
    CHECK_OK(e.prepare_unload());
}
struct RecordingGenerator final : rad::gen::IPassDriver {
    int next_calls = 0, shutdown_status = RAD_OK;
    std::vector<Readback> results;
    const RadBatch* next(int) override { ++next_calls; return nullptr; }
    const RadBatch* batch_for(int) const override { return nullptr; }
    const std::vector<Readback>& readbacks() const override { return results; }
    int done() override { return RAD_OK; }
    void shutdown(const char*, int status) override { shutdown_status = status; }
    int64_t queued() const override { return 0; }
    int64_t running() const override { return 0; }
};
TEST(generator_rejects_new_passes_after_a_failed_handoff) {
    Engine e(rad::Config{});
    REQUIRE_EQ(Access::init(e), RAD_OK);
    Gate gate(Access::stream(e, 1));
    CHECK_EQ(Access::drain(e), RAD_E_STATE);
    auto driver = std::make_unique<RecordingGenerator>();
    auto* observed = driver.get();
    CHECK_EQ(Access::generate(e, std::move(driver)), RAD_E_STATE);
    CHECK_EQ(observed->next_calls, 0);
    CHECK_EQ(observed->shutdown_status, RAD_E_STATE);
    gate.open();
}
TEST(follower_transfer_fence_prevents_shared_block_recycling) {
    Engine e(rad::Config{});
    REQUIRE_EQ(Access::init(e), RAD_OK);
    REQUIRE_EQ(Access::tier_stream(e, 0), RAD_OK);
    REQUIRE_EQ(Access::tier_stream(e, 1), RAD_OK);
    std::vector<int32_t> blocks;
    REQUIRE_EQ(Access::leader_blocks(e, &blocks), RAD_OK);
    const int64_t before = Access::free_blocks(e);
    Gate gate(Access::tier(e, 1));
    std::vector<rad::TierJob> jobs(2);
    for (size_t i = 0; i < jobs.size(); ++i) {
        jobs[i].kind = rad::TierJob::Kind::ToDevice;
        jobs[i].blocks = {blocks[i]};
    }
    CHECK_EQ(Access::report(e, jobs), RAD_E_STATE);
    CHECK_EQ(Access::free_blocks(e), before);
    CHECK_EQ(jobs[0].blocks[0], blocks[0]);
    gate.open();
    CHECK_OK(Access::drain(e));
    CHECK_OK(Access::report(e, jobs));
    CHECK_EQ(Access::free_blocks(e), before + 2);
    CHECK(jobs[0].blocks.empty() && jobs[1].blocks.empty());
}
TEST(persistent_teardown_fence_exits_before_freeing_device_memory) {
    const pid_t child = fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        // Create streams after fork: no inherited host worker threads are used in the child.
        auto* e = new Engine(rad::Config{});
        if (Access::init(*e, 1) < 0) _exit(99);
        void* gate = nullptr;
        if (rad_stream_gate(Access::stream(*e, 0), &gate) < 0) _exit(98);
        delete e;
        _exit(97);
    }
    int status = 0;
    REQUIRE_EQ(waitpid(child, &status, 0), child);
    CHECK(WIFEXITED(status));
    CHECK_EQ(WEXITSTATUS(status), 70);
}

struct Broker final : rad::VramBroker {
    int calls = 0, settled = 0;
    int64_t requested = 0;
    bool allow = true;
    int grant(const rad::VramAsk&, int64_t*, std::string*) override { return RAD_E_UNSUPPORTED; }
    bool flex_grow(const std::vector<int>&, int64_t bytes) override {
        ++calls; requested = bytes; return allow;
    }
    void flex_done() override { ++settled; }
};
struct Mappings final : rad::ExpertMemoryOps {
    Engine& engine;
    int fail_rank = -1;
    bool fail_settle = false;
    int commits = 0, settles = 0;
    explicit Mappings(Engine& e) : engine(e) {}
    size_t rank_count() const override { return 2; }
    rad::VMem& memory(size_t rank) override { return Access::backing(engine, (int)rank); }
    int select(size_t) override { return rad_dev_set(0); }
    int commit(size_t rank, int64_t offset, int64_t bytes) override {
        ++commits;
        return (int)rank == fail_rank ? RAD_E_NOMEM : memory(rank).commit(offset, bytes);
    }
    int settle() override { ++settles; return fail_settle ? RAD_E_DEVICE : rad::dev_settle_maps(); }
};
TEST(asymmetric_recall_is_repaired_before_either_loan_can_grow) {
    Broker broker;
    Engine e(rad::Config{});
    REQUIRE_EQ(Access::init(e), RAD_OK);
    Access::broker(e, &broker);
    const int64_t g = rad::VMem::granularity();
    REQUIRE(g > 0);
    for (int rank = 0; rank < 2; ++rank) {
        CHECK_OK(Access::backing(e, rank).reserve(8 * g));
        CHECK_OK(Access::backing(e, rank).commit(4 * g, 4 * g));
    }
    Access::set_loan(e, 2 * g);
    // One rank returned its lower pages; the other retained them after a failed recall.
    CHECK_OK(Access::backing(e, 0).decommit(4 * g, 2 * g));
    CHECK_EQ(Access::backing(e, 0).committed_suffix(), 2 * g);
    CHECK_EQ(Access::backing(e, 1).committed_suffix(), 4 * g);
    Mappings maps(e);
    CHECK_OK(Access::grow(e, 5 * g, maps));
    CHECK_EQ(broker.requested, 3 * g); // includes rank 0's missing old pages, not just one new page
    CHECK_EQ(broker.calls, broker.settled);
    for (int rank = 0; rank < 2; ++rank) {
        CHECK(Access::backing(e, rank).is_committed(3 * g, 5 * g));
        CHECK_EQ(Access::loan(e, rank), 5 * g);
    }
}
TEST(failed_second_rank_growth_keeps_old_loans_and_accounts_partial_pages) {
    Broker broker;
    Engine e(rad::Config{});
    REQUIRE_EQ(Access::init(e), RAD_OK);
    Access::broker(e, &broker);
    const int64_t g = rad::VMem::granularity();
    for (int rank = 0; rank < 2; ++rank) {
        CHECK_OK(Access::backing(e, rank).reserve(8 * g));
        CHECK_OK(Access::backing(e, rank).commit(6 * g, 2 * g));
    }
    Access::set_loan(e, 2 * g);
    Mappings maps(e); maps.fail_rank = 1;
    CHECK_EQ(Access::grow(e, 5 * g, maps), RAD_E_NOMEM);
    CHECK_EQ(Access::charge(e), 5 * g); // rank 0's successfully mapped pages remain charged
    CHECK_EQ(Access::backing(e, 1).committed(), 2 * g);
    CHECK_EQ(Access::loan(e, 0), 2 * g);
    CHECK_EQ(Access::loan(e, 1), 2 * g);
    CHECK(Access::shrink(e, g) >= 0);
    CHECK(Access::loan(e, 0) <= 2 * g);
    CHECK(Access::loan(e, 1) <= 2 * g);
    CHECK(Access::backing(e, 1).is_committed(8 * g - Access::loan(e, 1), Access::loan(e, 1)));
    maps.fail_rank = -1;
    CHECK_OK(Access::grow(e, 5 * g, maps));
    CHECK_EQ(Access::loan(e, 0), 5 * g);
    CHECK_EQ(Access::loan(e, 1), 5 * g);
    CHECK_EQ(broker.calls, broker.settled);
}
TEST(mapped_extents_do_not_publish_a_loan_before_translation_settlement) {
    Broker broker;
    Engine e(rad::Config{});
    REQUIRE_EQ(Access::init(e), RAD_OK);
    Access::broker(e, &broker);
    const int64_t g = rad::VMem::granularity();
    for (int rank = 0; rank < 2; ++rank) CHECK_OK(Access::backing(e, rank).reserve(8 * g));
    Mappings maps(e); maps.fail_settle = true;
    CHECK_EQ(Access::grow(e, 4 * g, maps), RAD_E_DEVICE);
    CHECK_EQ(Access::charge(e), 4 * g);
    CHECK_EQ(Access::loan(e, 0), 0);
    CHECK_EQ(Access::loan(e, 1), 0);
    CHECK_EQ(e.memory_fence_status(), RAD_E_DEVICE);
    const int calls = broker.calls;
    maps.fail_settle = false;
    CHECK_OK(Access::grow(e, 4 * g, maps)); // no new pages, but the settle must be retried
    CHECK(maps.settles >= 2);
    CHECK_EQ(broker.calls, calls);
    CHECK_EQ(e.memory_fence_status(), RAD_OK);
    CHECK_EQ(Access::loan(e, 0), 4 * g);
    CHECK_EQ(Access::loan(e, 1), 4 * g);
}
TEST(refused_growth_does_not_change_backing_or_loans) {
    Broker broker; broker.allow = false;
    Engine e(rad::Config{});
    REQUIRE_EQ(Access::init(e), RAD_OK);
    Access::broker(e, &broker);
    const int64_t g = rad::VMem::granularity();
    for (int rank = 0; rank < 2; ++rank) CHECK_OK(Access::backing(e, rank).reserve(8 * g));
    Mappings maps(e);
    CHECK_EQ(Access::grow(e, 4 * g, maps), RAD_E_FULL);
    CHECK_EQ(maps.commits, 0);
    CHECK_EQ(Access::charge(e), 0);
    CHECK_EQ(Access::loan(e, 0), 0);
    CHECK_EQ(Access::loan(e, 1), 0);
    CHECK_EQ(broker.settled, 0);
}

TEST(full_mapping_without_published_loans_is_repaired_by_the_normal_tick) {
    Broker broker;
    Engine e(rad::Config{});
    REQUIRE_EQ(Access::init(e), RAD_OK);
    Access::broker(e, &broker);
    const int64_t g = rad::VMem::granularity();
    for (int rank = 0; rank < 2; ++rank) {
        CHECK_OK(Access::backing(e, rank).reserve(8 * g));
        CHECK_OK(Access::backing(e, rank).commit(0, 8 * g));
    }
    Access::set_loan(e, 2 * g);
    Access::tick(e);
    CHECK_EQ(Access::loan(e, 0), 8 * g);
    CHECK_EQ(Access::loan(e, 1), 8 * g);
    CHECK_EQ(broker.calls, 0);
}
TEST(final_unload_fence_requires_an_explicit_producer_join) {
    Engine e(rad::Config{});
    REQUIRE_EQ(Access::init(e), RAD_OK);
    Access::producer_state(e, false, true);
    CHECK_EQ(e.prepare_unload(), RAD_E_STATE);
    CHECK_EQ(e.retry_memory_fence(), RAD_E_STATE);
    Access::producer_state(e, true, true);
    CHECK_OK(e.prepare_unload());
}
TEST(idle_restore_repairs_only_the_remembered_suffix_on_every_rank) {
    Broker broker;
    Engine e(rad::Config{});
    REQUIRE_EQ(Access::init(e), RAD_OK);
    Access::broker(e, &broker);
    const int64_t g = rad::VMem::granularity();
    REQUIRE(g > 0);
    for (int rank = 0; rank < 2; ++rank) {
        CHECK_OK(Access::backing(e, rank).reserve(8 * g));
        CHECK_OK(Access::backing(e, rank).commit(4 * g, 4 * g));
    }
    Access::set_loan(e, 4 * g);
    REQUIRE_EQ(Access::shrink(e, 2 * g), 2 * g);
    // Rank zero has already mapped the previous watermark after a partial grow.
    CHECK_OK(Access::backing(e, 0).commit(4 * g, 2 * g));
    Access::set_loan(e, 2 * g);
    CHECK_EQ(Access::charge(e), 4 * g);
    Access::idle_tick(e);
    CHECK_EQ(broker.requested, 2 * g);
    CHECK_EQ(broker.calls, 1);
    CHECK_EQ(broker.settled, 1);
    CHECK_EQ(Access::eviction_demand(e), 0);
    for (int rank = 0; rank < 2; ++rank) CHECK_EQ(Access::loan(e, rank), 4 * g);
    Access::idle_tick(e);
    CHECK_EQ(broker.calls, 1);
}

TEST(idle_restore_denied_by_the_broker_does_not_request_peer_eviction) {
    Broker broker;
    Engine e(rad::Config{});
    REQUIRE_EQ(Access::init(e), RAD_OK);
    Access::broker(e, &broker);
    const int64_t g = rad::VMem::granularity();
    REQUIRE(g > 0);
    for (int rank = 0; rank < 2; ++rank) {
        CHECK_OK(Access::backing(e, rank).reserve(8 * g));
        CHECK_OK(Access::backing(e, rank).commit(4 * g, 4 * g));
    }
    Access::set_loan(e, 4 * g);
    REQUIRE_EQ(Access::shrink(e, 4 * g), 4 * g);
    broker.allow = false;
    Access::idle_tick(e);
    CHECK_EQ(broker.calls, 1);
    CHECK_EQ(broker.settled, 0);
    CHECK_EQ(Access::eviction_demand(e), 0);
    CHECK_EQ(Access::charge(e), 0);
}
RAD_TEST_MAIN()
