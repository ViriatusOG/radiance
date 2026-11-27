/* p4b_kv_ring_test.cpp -- RAD_KV_RING, the declaration-side of R-SWA (Unlimited-OCR P4b).
 *
 * RAD_KV_RING is RAD_KV_WINDOW whose floor is each sequence's own prompt length: the group is
 * sized for the whole context plus one ring a sequence (the ring among the fixed costs), never
 * capped at a declared window, never given a window snapshot, and never checkpointed. These
 * tests pin the core's admission of the kind -- the builder, the KV plan, and the chunk geometry
 * -- and that a fixed-window (RAD_KV_WINDOW) group beside it is exactly what it was.
 */
#include "rad_test.h"

#include "build/rad_build.h"
#include "mem/kv.h"
#include "mem/pools.h"
#include "rad_internal.h"
#include "sched/geometry.h"

#include <cstring>
#include <vector>

using namespace rad;

/* Handle 0 is the null group. */
enum : int32_t { G_NULL = 0, G_RING = 1, G_FIXED = 2 };

static Config ring_config(int64_t max_tok = 128) {
    Config c;
    c.vram_weights_mib = 1;
    c.vram_kv_mib = 4;
    c.gpu_headroom_mib = 1;
    c.host_pool_mib = 0;
    c.max_seqs = 4;
    c.max_tok = max_tok;
    c.max_ctx = 4096;
    c.checkpoint_interval = 512;
    c.prefix_cache = true;
    return c;
}

/* One layer at `block`; the page is block * n_head_kv(2) * head_dim(8) * bf16(2) * K,V. */
static KVGroupInfo ring_group(int64_t block, int64_t w, int layer = 0) {
    KVGroupInfo k;
    k.name = "rswa";
    k.decl.kind = RAD_KV_RING;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = 2;
    k.decl.head_dim = 8;
    k.decl.window = w;
    k.block_size = block;
    k.layers = { layer };
    return k;
}

static KVGroupInfo fixed_window_group(int64_t block, int64_t w, int layer = 0) {
    KVGroupInfo k;
    k.name = "swa";
    k.decl.kind = RAD_KV_WINDOW;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = 2;
    k.decl.head_dim = 8;
    k.decl.window = w;
    k.block_size = block;
    k.layers = { layer };
    return k;
}

static std::vector<KVGroupInfo> one(const KVGroupInfo& g) {
    return { KVGroupInfo{}, g };
}

static int configure_pools(Pools& p, const Config& c) {
    return p.configure(c, 0, &rad_malloc_allocator(), &rad_malloc_allocator());
}

/* A RAD_KV_RING group plans as a runtime window without anyone declaring it first (the kind is
 * the declaration), and with the engine's declare_runtime_window() call it plans identically.
 * The ring is a fixed cost: one ring a sequence on top of the context the split sizes. */
TEST(a_ring_group_plans_as_a_runtime_window_with_the_ring_in_the_fixed_cost) {
    const int64_t B = 128, W = 128;
    const KVGroupInfo grp = ring_group(B, W);
    const std::vector<KVGroupInfo> g = one(grp);
    const Config cfg = ring_config();
    const int64_t page = B * 2 * 8 * 2 * 2;               /* 8192 bytes */
    const int64_t ring_cost = (int64_t)cfg.max_seqs * ((W + B - 1) / B) * page;

    /* Without declare_runtime_window: the kind alone turns the plan over to runtime windows. */
    {
        Pools pools;
        REQUIRE_EQ(configure_pools(pools, cfg), RAD_OK);
        KVManager kv;
        CHECK_OK(kv.configure(g, cfg, pools.kv()));
        const KVGroupPlan* p = kv.plan(G_RING);
        REQUIRE(p);
        CHECK(p->runtime_win);
        CHECK_EQ(p->runtime_tokens, W);
        CHECK_EQ(kv.runtime_window_tokens(G_RING), W);
        /* No window snapshot, so nothing to checkpoint: the window is each sequence's own. */
        CHECK(!p->win_snap);
        CHECK(!kv.has_window_snapshot());
        CHECK_EQ(kv.checkpoint_bytes(), 0ll);
        /* Sized for the whole context plus the ring a sequence, not capped at window_blocks. */
        CHECK_EQ(kv.ceiling(cfg.max_ctx),
                 kv.fixed_bytes() + (int64_t)cfg.max_seqs * (cfg.max_ctx / B) * page);
        /* The ring is in the fixed cost (the plan's one alignment unit is for the group). */
        CHECK_EQ(kv.fixed_bytes(), (int64_t)RAD_ALIGN_UNIT + ring_cost);
        /* The report calls it a ring, not a fixed window. */
        CHECK(has(kv.report(), "ring"));
        CHECK(has(kv.report(), "runtime window W=128"));
    }

    /* The engine's path: declare, then plan. Same plan, and the declaration is idempotent
     * against the kind's own. */
    {
        Pools pools;
        REQUIRE_EQ(configure_pools(pools, cfg), RAD_OK);
        KVManager kv;
        CHECK_OK(kv.declare_runtime_window(G_RING, W));
        CHECK_OK(kv.configure(g, cfg, pools.kv()));
        const KVGroupPlan* p = kv.plan(G_RING);
        REQUIRE(p);
        CHECK(p->runtime_win);
        CHECK_EQ(p->runtime_tokens, W);
        CHECK_EQ(kv.fixed_bytes(), (int64_t)RAD_ALIGN_UNIT + ring_cost);
        CHECK(!kv.has_window_snapshot());
        /* A ring carries no declared-window cap; window_blocks still reports the ring's own
         * rounding (rarely read: every plan path gates on runtime_win first). */
        CHECK_EQ(p->window_blocks, W / B + 1);
    }

    /* A second W would contradict the ring already in the fixed cost. */
    {
        KVManager kv;
        CHECK_OK(kv.plan_groups(g, cfg));
        int rc = 0;
        const std::string why = stderr_of([&] { rc = kv.declare_runtime_window(G_RING, W * 2); });
        CHECK_EQ(rc, RAD_E_INVAL);
        CHECK(has(why, "already carries a runtime window"));
        CHECK_EQ(kv.declare_runtime_window(G_RING, W), RAD_OK);
    }
}

/* A fixed-width window group beside a ring group keeps today's plan exactly, so gemma4_bf16's
 * recycling and window snapshot are untouched by R-SWA's presence. */
TEST(a_fixed_window_group_is_unaffected_beside_a_ring_group) {
    const KVGroupInfo ring = ring_group(128, 128, /*layer=*/0);
    const KVGroupInfo fixed = fixed_window_group(16, 64, /*layer=*/1);
    std::vector<KVGroupInfo> both = { KVGroupInfo{}, ring, fixed };
    std::vector<KVGroupInfo> alone = { KVGroupInfo{}, fixed };
    Config cfg = ring_config();
    cfg.max_tok = 64;

    Pools pa, pb;
    REQUIRE_EQ(configure_pools(pa, cfg), RAD_OK);
    REQUIRE_EQ(configure_pools(pb, cfg), RAD_OK);
    KVManager kva, kvb;
    CHECK_OK(kva.declare_runtime_window(G_RING, 128));
    CHECK_OK(kva.configure(both, cfg, pa.kv()));
    CHECK_OK(kvb.configure(alone, cfg, pb.kv()));

    const KVGroupPlan* p = kva.plan(G_FIXED);   /* index 2: {null, ring, fixed} */
    const KVGroupPlan* q = kvb.plan(1);         /* the fixed group alone: {null, fixed} */
    REQUIRE(p);
    REQUIRE(q);
    CHECK(!p->runtime_win);
    CHECK_EQ(p->runtime_tokens, 0ll);
    CHECK(p->win_snap);
    CHECK(q->win_snap);
    CHECK_EQ(p->window_blocks, q->window_blocks);
    CHECK_EQ(p->window_blocks, 64 / 16 + 1);
    CHECK_EQ(p->bytes_per_block, q->bytes_per_block);
    CHECK_EQ(p->block_size, q->block_size);
    /* Only the fixed group's window is checkpointed, and it is the same number either way. */
    CHECK_EQ(kva.checkpoint_bytes(), kvb.checkpoint_bytes());
    CHECK(kva.has_window_snapshot() == kvb.has_window_snapshot());
    /* The ring's own plan is the runtime window, not the fixed one. */
    CHECK(kva.plan(G_RING)->runtime_win);
    CHECK(!kva.plan(G_RING)->win_snap);
}

/* The chunk geometry must not advertise a window snapshot for a ring: there is none to take, and
 * an interval grid would only claim one. A fixed window group keeps it exactly. */
TEST(a_ring_group_gets_no_window_snapshot_geometry) {
    Config cfg = ring_config();
    Program prog;
    prog.kv_groups = one(ring_group(128, 128));
    ChunkGeometry geo;
    CHECK_OK(chunk_geometry_resolve(prog, cfg, &geo));
    CHECK_EQ(geo.attn_block, 128ll);
    CHECK_EQ(geo.quantum, 128ll);
    CHECK(!geo.window_ckpt);
    CHECK_EQ(geo.checkpoint_interval, 0ll);
    CHECK(!geo.checkpoints());

    /* The same group declared as a fixed window keeps the snapshot geometry (gemma4's). */
    Program w;
    w.kv_groups = one(fixed_window_group(128, 128));
    CHECK_OK(chunk_geometry_resolve(w, cfg, &geo));
    CHECK(geo.window_ckpt);
    CHECK_EQ(geo.checkpoint_interval, 512ll);
    CHECK(geo.checkpoints());

    /* A ring beside a fixed window still snapshots the fixed one, as a hybrid always did. */
    Program h;
    h.kv_groups = { KVGroupInfo{}, ring_group(128, 128, 0), fixed_window_group(128, 128, 1) };
    CHECK_OK(chunk_geometry_resolve(h, cfg, &geo));
    CHECK_EQ(geo.attn_block, 128ll);
    CHECK(geo.window_ckpt);
    CHECK_EQ(geo.checkpoint_interval, 512ll);
}

/* ================================================================== the declare phase */

static std::string kernels_dir() { return home() + "/testkernels"; }

static int load_plugins(Registry& r) {
    return r.load_dir(kernels_dir(), { "radtest_over", "radtest_base" });
}

static rad_op decl_t_attn(Builder& b) {
    const RadParam p[] = { RAD_RANGE("q_len", 1, 16), RAD_INT("head_dim", 128),
                           RAD_INT("window", 0), RAD_INT("block_size", 16),
                           RAD_INT("causal", 1), RAD_INT("n_head", 8),
                           RAD_STR("kv_dtype", "bf16") };
    return b.decl_op("t_attn", p, 7, nullptr, 0);
}

static RadModelMeta make_meta() {
    RadModelMeta m{};
    m.arch_id  = "test_arch";
    m.name     = "test_model";
    m.quant    = "bf16";
    m.n_layers = 2;
    m.n_embd   = 4096;
    return m;
}

static RadBuildCtx make_ctx() {
    RadBuildCtx c{};
    c.rank = 0;
    c.world_size = 1;
    c.max_tok  = 8192;
    c.max_seqs = 256;
    c.max_ctx  = 4096;
    c.scope = "";
    return c;
}

/* The ABI admits the kind, its block size still comes off the resolved attention kernel, and its
 * page is sized as a paged group's (so KVManager::plan_groups accepts it). */
TEST(the_builder_accepts_a_ring_group) {
    Registry reg;
    if (load_plugins(reg) < 0) ARTIFACT_MISSING("the test plugins in " + kernels_dir());

    Builder b(reg, make_meta(), make_ctx());
    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_RING;
    kd.dtype = RAD_BF16;
    kd.window = 128;
    kd.n_head_kv = 8;
    kd.head_dim = 128;
    const rad_kvgroup g = b.decl_kv_group("kv_ring", &kd);
    CHECK(g == 1);
    CHECK_EQ((int)b.program().kv_groups[g].decl.kind, (int)RAD_KV_RING);

    decl_t_attn(b);
    /* 16 because the resolved test kernel's constraints say so. */
    CHECK_EQ(b.kv_block_size(g), 16);
    CHECK_OK(b.bind_layer_kv(0, g));
    CHECK_OK(b.bind_layer_kv(1, g));
    b.finish();

    const KVGroupInfo& gi = b.program().kv_groups[g];
    CHECK_EQ(gi.block_size, 16ll);
    CHECK_EQ(gi.decl.window, 128ll);
    /* One page covers both bound layers, as for a full or fixed-window group. */
    CHECK_EQ(gi.bytes_per_block, 2 * 16 * 8 * 128 * 2 * 2);
    CHECK_EQ(gi.bytes_per_state, 0ll);
    /* Named, not "?". */
    CHECK(std::strcmp(kv_kind_name(RAD_KV_RING), "?") != 0);
}

RAD_TEST_MAIN()
