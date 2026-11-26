/* kv_window_test.cpp -- the runtime per-sequence window on a RAD_KV_WINDOW group (R-SWA).
 *
 * R-SWA (Unlimited-OCR) keeps a sequence's whole prefill as a retained reference and decodes into
 * a ring of W slots right after it: visible = [0, prefill) U [prefill, prefill + W). The reference
 * is never recycled, the ring rewrites its slots in place, and RoPE positions keep increasing --
 * only the block/slot mapping recycles. These tests pin the manager's side of that: the visible
 * token set, flat bytes after the floor is set, the recycling boundary, the exclusion from the
 * prefix cache and snapshots, and that a fixed-window group (gemma4's) is exactly what it was.
 */
#include "rad_test.h"

#include "mem/kv.h"
#include "mem/pools.h"
#include "sched/advance_impl.h"
#include "sched/mem_iface.h"
#include "rad_internal.h"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

using namespace rad;

/* Handle 0 is the null group (see mem_test): a real group at index 0 would read as "no group". */
enum : int32_t { G_NULL = 0, G_WIN = 1 };

static Config window_config() {
    Config c;
    c.vram_weights_mib = 1;
    c.vram_kv_mib = 4;             /* 4 MiB: an 8 KiB page at block 128, room for 4 x 4096 tokens */
    c.gpu_headroom_mib = 1;
    c.host_pool_mib = 0;
    c.max_seqs = 4;
    c.max_tok = 128;               /* one block a step, as a decode deployment */
    c.max_ctx = 4096;
    c.checkpoint_interval = 512;
    c.prefix_cache = true;
    return c;
}

/* A single windowed group of one layer at `block` (page block*2*8*2*2 bytes), so a test can use
 * the geometry R-SWA runs at: block_size 128 = W. */
static KVGroupInfo window_group(int64_t block, int64_t declared_window) {
    KVGroupInfo k;
    k.name = "rswa";
    k.decl.kind = RAD_KV_WINDOW;
    k.decl.dtype = RAD_BF16;
    k.decl.n_head_kv = 2;
    k.decl.head_dim = 8;
    k.decl.window = declared_window;
    k.block_size = block;
    k.layers = { 0 };
    return k;
}

static std::vector<KVGroupInfo> groups_of(const KVGroupInfo& g) {
    std::vector<KVGroupInfo> v;
    v.push_back(KVGroupInfo{});
    v.push_back(g);
    return v;
}

static int configure_pools(Pools& p, const Config& c) {
    return p.configure(c, 0, &rad_malloc_allocator(), &rad_malloc_allocator());
}

/* A sequence's slot for `pos`, through the rule the step batch uses (KVGeom::slot_ring). */
struct Ring {
    KVManager* kv;
    KVGeom     geo;
    int32_t    g = G_WIN;

    bool has_window() const { return kv->runtime_window(1, g, nullptr); }
    SeqWindow window() const {
        SeqWindow w;
        kv->runtime_window(1, g, &w);
        return w;
    }
    int32_t slot(int64_t pos) const {
        SeqWindow w;
        if (!kv->runtime_window(1, g, &w)) return -1;
        return geo.slot_ring(g, kv->block_table(1, g), kv->first_block_pos(1, g), pos, w);
    }
};

TEST(rswa_reference_and_ring_keeps_the_visible_set_and_flat_bytes) {
    const int64_t B = 128, W = 128, P = 400;      /* P is not a block multiple on purpose */
    const KVGroupInfo grp = window_group(B, W);
    const std::vector<KVGroupInfo> g = groups_of(grp);
    Config cfg = window_config();
    Pools pools;
    CHECK_OK(configure_pools(pools, cfg));
    KVManager kv;
    CHECK_OK(kv.declare_runtime_window(G_WIN, W));
    CHECK_OK(kv.configure(g, cfg, pools.kv()));

    /* A runtime-window group keeps no window snapshot: the window is the sequence's own data. */
    CHECK(kv.plan(G_WIN)->runtime_win);
    CHECK_EQ(kv.plan(G_WIN)->runtime_tokens, W);
    CHECK(!kv.plan(G_WIN)->win_snap);
    CHECK(!kv.has_window_snapshot());
    CHECK_EQ(kv.checkpoint_bytes(), 0ll);
    CHECK_EQ(kv.runtime_window_tokens(G_WIN), W);
    /* Sized for the context plus one ring a sequence, not capped at window_blocks. */
    CHECK(kv.total_blocks(G_WIN) >= cfg.max_seqs * (cfg.max_ctx / B + 1));

    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.set_window(1, P, W));
    const Ring r{ &kv, KVGeom::of(g), G_WIN };
    CHECK(r.has_window());
    const SeqWindow w = r.window();
    CHECK_EQ(w.visible_lo, P);
    /* Whole blocks: the ring starts at the first boundary at or after the floor, and the tokens
     * in [P, ring_pos) stay in the reference's last block. */
    CHECK_EQ(w.ring_pos, 512ll);
    CHECK_EQ(w.ring_blocks, 1ll);
    CHECK_EQ(w.ring_span(B), 128ll);

    /* Prefill: the reference is written linearly; the table is ceil(P / B) blocks. */
    CHECK_OK(kv.ensure(1, P));
    CHECK_EQ(kv.first_block_pos(1, G_WIN), 0ll);
    CHECK_EQ((long long)kv.block_table(1, G_WIN).size(), 4ll);
    std::map<int32_t, int64_t> content;             /* slot -> position it holds */
    for (int64_t p = 0; p < P; ++p) {
        const int32_t s = r.slot(p);
        REQUIRE(s >= 0);
        content[s] = p;
    }

    /* Generate L tokens. The visible set must be [0, P) plus, in the ring, the newest ring_span
     * positions: [0, P) U [P, ring_pos) (frozen: written once, kept) U the last `span` of
     * [ring_pos, n). Block granularity is the accepted consequence -- with W = 128 and block 128
     * the visible tail is 112 frozen + 128 ring = 240 tokens, always W..W+B-1 = 128..255 and a
     * superset of the reference implementation's last 128. Once n >= ring_pos + span the set is
     * exactly that, so the test asserts equality. */
    const int64_t L = 2000;
    int64_t flat_blocks = -1;
    int64_t bad_step = -1;
    for (int64_t p = P; p < P + L && bad_step < 0; ++p) {
        CHECK_OK(kv.ensure(1, p + 1, p));
        const int32_t s = r.slot(p);
        REQUIRE(s >= 0);
        content[s] = p;
        const int64_t n = p + 1;

        /* The table never moves and the floor stays 0: only slots recycle. */
        CHECK_EQ(kv.first_block_pos(1, G_WIN), 0ll);
        const int64_t want_blocks =
            n <= w.ring_pos ? (n + B - 1) / B
                            : w.ring_pos / B + std::min(w.ring_blocks,
                                                        (n - w.ring_pos + B - 1) / B);
        CHECK_EQ((long long)kv.block_table(1, G_WIN).size(), want_blocks);

        /* Bytes are flat once the ring is full: no block is allocated past it, ever. */
        if (n >= w.ring_pos + w.ring_span(B)) {
            if (flat_blocks < 0) flat_blocks = (int64_t)kv.block_table(1, G_WIN).size();
            CHECK_EQ((long long)kv.block_table(1, G_WIN).size(), flat_blocks);
            CHECK_EQ(kv.held_blocks(G_WIN), flat_blocks);
        }

        /* The slots hold exactly the rule's set. */
        std::set<int64_t> held, want;
        for (const auto& kvp : content) held.insert(kvp.second);
        for (int64_t q = 0; q < std::min<int64_t>(P, n); ++q) want.insert(q);
        for (int64_t q = P; q < std::min(w.ring_pos, n); ++q) want.insert(q);
        for (int64_t q = std::max<int64_t>(w.ring_pos, n - w.ring_span(B)); q < n; ++q)
            want.insert(q);
        if (held != want) bad_step = n;
    }
    CHECK_EQ(bad_step, -1ll);

    /* The reference's blocks were never recycled: blocks 0..2 are untouched and block 3's
     * reference slots (0..111 = positions 384..399) still hold their prefill values. */
    const std::vector<int32_t>& bt = kv.block_table(1, G_WIN);
    REQUIRE_EQ((long long)bt.size(), 5ll);
    for (int64_t p = 0; p < P; ++p) {
        const int64_t bi = p / B;
        const int32_t slot = (int32_t)(bt[(size_t)bi] * B + p % B);
        REQUIRE(content.count(slot) > 0);
        CHECK_EQ(content[slot], p);
    }

    /* The recycling boundary: the ring is one block, so a position and the position one span
     * later share a slot, while neighbours do not. ring_pos + span is the first token that
     * overwrites ring_pos. */
    CHECK_EQ(r.slot(w.ring_pos), r.slot(w.ring_pos + w.ring_span(B)));
    CHECK(r.slot(w.ring_pos) != r.slot(w.ring_pos + 1));
    const int32_t first_ring_slot = (int32_t)(bt[4] * B + w.ring_pos % B);
    CHECK_EQ(r.slot(w.ring_pos), first_ring_slot);

    /* Flat to the end, and the free list gets everything back on free. */
    CHECK_EQ((long long)kv.block_table(1, G_WIN).size(), flat_blocks);
    const int64_t free0 = kv.free_blocks(G_WIN);
    kv.free_sequence(1);
    CHECK_EQ(kv.free_blocks(G_WIN), free0 + flat_blocks);
}

/* A block-aligned floor is the exact reference rule: the ring starts at the floor, so the visible
 * tail is exactly the last W tokens. */
TEST(rswa_aligned_floor_shows_exactly_the_last_w_tokens) {
    const int64_t B = 128, W = 128, P = 512;
    const KVGroupInfo grp = window_group(B, W);
    const std::vector<KVGroupInfo> g = groups_of(grp);
    Config cfg = window_config();
    Pools pools;
    CHECK_OK(configure_pools(pools, cfg));
    KVManager kv;
    CHECK_OK(kv.declare_runtime_window(G_WIN, W));
    CHECK_OK(kv.configure(g, cfg, pools.kv()));
    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.set_window(1, P, W));
    const Ring r{ &kv, KVGeom::of(g), G_WIN };
    const SeqWindow w = r.window();
    CHECK_EQ(w.ring_pos, P);
    CHECK_EQ(w.ring_blocks, 1ll);
    CHECK_EQ(w.ring_span(B), W);

    CHECK_OK(kv.ensure(1, P));
    REQUIRE_EQ((long long)kv.block_table(1, G_WIN).size(), 4ll);
    for (int64_t p = P; p < P + 1024; ++p) CHECK_OK(kv.ensure(1, p + 1, p));
    /* Every tail position maps into the one ring block, at the position mod B. */
    REQUIRE_EQ((long long)kv.block_table(1, G_WIN).size(), 5ll);
    const int32_t ring_block = kv.block_table(1, G_WIN)[4];
    for (int64_t p = P; p < P + 1024; ++p) {
        const int32_t s = r.slot(p);
        REQUIRE(s >= 0);
        CHECK_EQ(s, (int32_t)(ring_block * B + p % B));
    }
    /* Held = 4 reference blocks + 1 ring, and nothing more however long the generation is. */
    CHECK_EQ(kv.held_blocks(G_WIN), 5ll);
    kv.free_sequence(1);
}

/* A group that is never given a runtime window keeps today's recycling exactly: the same numbers
 * as mem_test's kv_window_group_recycles_below_the_window, so gemma4_bf16 is untouched. */
TEST(fixed_window_group_is_unchanged) {
    const KVGroupInfo grp = window_group(16, 64);
    const std::vector<KVGroupInfo> g = groups_of(grp);
    Config cfg = window_config();
    cfg.max_tok = 64;
    Pools pools;
    CHECK_OK(configure_pools(pools, cfg));
    KVManager kv;
    CHECK_OK(kv.configure(g, cfg, pools.kv()));     /* no declare_runtime_window */

    CHECK(!kv.plan(G_WIN)->runtime_win);
    CHECK_EQ(kv.runtime_window_tokens(G_WIN), 0ll);
    /* Window 64 at block 16 is 4 blocks plus one for the straddle, as before. */
    CHECK_EQ(kv.plan(G_WIN)->window_blocks, 5ll);

    const int64_t n0 = kv.free_blocks(G_WIN);
    CHECK_OK(kv.add_sequence(1));
    CHECK(!kv.has_runtime_window(1));
    CHECK_OK(kv.ensure(1, 80));
    CHECK_EQ(kv.first_block_pos(1, G_WIN), 0ll);
    CHECK_EQ((long long)kv.block_table(1, G_WIN).size(), 5ll);
    CHECK_OK(kv.ensure(1, 400));
    CHECK_EQ(kv.first_block_pos(1, G_WIN), 16ll);
    CHECK_EQ((long long)kv.block_table(1, G_WIN).size(), 24ll);
    CHECK_OK(kv.ensure(1, 401));
    CHECK_EQ(kv.first_block_pos(1, G_WIN), 336ll);
    CHECK_EQ((long long)kv.block_table(1, G_WIN).size(), 5ll);
    kv.trim_window(1, 401);
    CHECK_EQ(kv.first_block_pos(1, G_WIN), 336ll);
    CHECK_EQ((long long)kv.block_table(1, G_WIN).size(), 5ll);
    CHECK_EQ(kv.free_blocks(G_WIN), n0 - 5);
    kv.free_sequence(1);
    CHECK_EQ(kv.free_blocks(G_WIN), n0);
}

/* A sequence with a runtime window is excluded from every reuse path that assumes the declare-time
 * window, and the manager refuses the ones a caller could still reach. */
TEST(rswa_sequence_is_excluded_from_prefix_and_snapshots) {
    const KVGroupInfo grp = window_group(128, 128);
    const std::vector<KVGroupInfo> g = groups_of(grp);
    Config cfg = window_config();
    Pools pools;
    CHECK_OK(configure_pools(pools, cfg));
    KVManager kv;
    CHECK_OK(kv.declare_runtime_window(G_WIN, 128));
    CHECK_OK(kv.configure(g, cfg, pools.kv()));
    /* The declare-time window is still what an unarmed sequence gets. */
    CHECK_OK(kv.add_sequence(1));
    CHECK(!kv.has_runtime_window(1));
    CHECK_OK(kv.set_window(1, 256, 128));
    CHECK(kv.has_runtime_window(1));
    CHECK_OK(kv.ensure(1, 256));
    CHECK_OK(kv.ensure(1, 300, 256));

    std::vector<int32_t> rows;
    CHECK(!kv.window_rows(1, 256, &rows));           /* no snapshot describes this window */
    CHECK(rows.empty());
    CHECK_EQ(kv.seed_window(1, 256), RAD_E_STATE);   /* no hit resumes into it */
    CHECK_EQ(kv.adopt(1, G_WIN, kv.block_table(1, G_WIN), 256), RAD_E_STATE);
    CHECK_EQ(kv.fork(1, 2), RAD_E_UNSUPPORTED);
    CHECK_EQ(kv.rollback(1, 200), RAD_E_UNSUPPORTED);
    /* A fresh retry: preempt clears the window so admission re-arms it. */
    CHECK_OK(kv.preempt(1));
    CHECK(!kv.has_runtime_window(1));
    kv.free_sequence(1);
}

/* Refusals: a runtime window needs a windowed group, a floor at or ahead of the KV it holds, and
 * blocks that can hold the reference plus the ring. */
TEST(rswa_refusals_are_explicit) {
    Config cfg = window_config();
    {
        /* A FULL group cannot carry one: the declaration is recorded, and the plan refuses it. */
        std::vector<KVGroupInfo> g = groups_of(window_group(128, 0));
        g[G_WIN].decl.kind = RAD_KV_FULL;
        Pools pools;
        CHECK_OK(configure_pools(pools, cfg));
        KVManager kv;
        CHECK_OK(kv.declare_runtime_window(G_WIN, 128));
        CHECK_EQ(kv.configure(g, cfg, pools.kv()), RAD_E_INVAL);
    }
    {
        /* A model with no windowed group has nothing to arm. */
        std::vector<KVGroupInfo> g = groups_of(window_group(128, 0));
        g[G_WIN].decl.kind = RAD_KV_FULL;
        Pools pools;
        CHECK_OK(configure_pools(pools, cfg));
        KVManager kv;
        CHECK_OK(kv.configure(g, cfg, pools.kv()));
        CHECK_OK(kv.add_sequence(1));
        CHECK_EQ(kv.set_window(1, 256, 128), RAD_E_UNSUPPORTED);
        kv.free_sequence(1);
    }
    {
        /* A windowed group plans; the refusals are about the sequence. */
        const std::vector<KVGroupInfo> g = groups_of(window_group(128, 128));
        Pools pools;
        CHECK_OK(configure_pools(pools, cfg));
        KVManager kv;
        CHECK_OK(kv.declare_runtime_window(G_WIN, 128));
        CHECK_OK(kv.configure(g, cfg, pools.kv()));
        CHECK_OK(kv.add_sequence(1));
        CHECK_EQ(kv.set_window(1, 256, 0), RAD_E_INVAL);
        CHECK_EQ(kv.set_window(1, -1, 128), RAD_E_INVAL);
        CHECK_OK(kv.ensure(1, 256));
        /* The floor cannot move behind KV already written. */
        CHECK_EQ(kv.set_window(1, 128, 128), RAD_E_STATE);
        CHECK(!kv.has_runtime_window(1));
        kv.free_sequence(1);
    }
}

/* The block rounding must never lose a token of the last W: the ring is W..W+B-1 tokens wide,
 * whatever W is relative to the block. */
TEST(rswa_ring_always_covers_the_last_w_tokens) {
    const int64_t B = 128;
    for (int64_t w_tokens : { 1, 64, 100, 128, 129, 200, 300 }) {
        for (int64_t P : { 400, 512 }) {
            const KVGroupInfo grp = window_group(B, 128);
            const std::vector<KVGroupInfo> g = groups_of(grp);
            Config cfg = window_config();
            Pools pools;
            REQUIRE_EQ(configure_pools(pools, cfg), RAD_OK);
            KVManager kv;
            REQUIRE_EQ(kv.declare_runtime_window(G_WIN, w_tokens), RAD_OK);
            REQUIRE_EQ(kv.configure(g, cfg, pools.kv()), RAD_OK);
            REQUIRE_EQ(kv.add_sequence(1), RAD_OK);
            REQUIRE_EQ(kv.set_window(1, P, w_tokens), RAD_OK);
            const Ring r{ &kv, KVGeom::of(g), G_WIN };
            const SeqWindow w = r.window();
            const int64_t span = w.ring_span(B);
            CHECK_EQ(w.ring_pos, (P + B - 1) / B * B);
            CHECK_EQ(w.ring_blocks, (w_tokens + B - 1) / B);
            /* The frozen tail [P, ring_pos) plus the ring always covers the last W: at least W
             * wide, and never more than the ring's rounding plus the frozen fraction (W + B - 1
             * when W is a block multiple, W + 2B - 2 otherwise). */
            CHECK(w.ring_pos - P + span >= w_tokens);
            CHECK(w.ring_pos - P + span <= w_tokens + 2 * B - 2);
            if (w_tokens % B == 0) CHECK(w.ring_pos - P + span <= w_tokens + B - 1);
            REQUIRE_EQ(kv.ensure(1, P), RAD_OK);
            const int64_t n = P + 3 * span;
            for (int64_t p = P; p < n; ++p) REQUIRE_EQ(kv.ensure(1, p + 1, p), RAD_OK);
            /* Held blocks: the reference's, then the ring's -- flat. */
            const int64_t blocks = (int64_t)kv.block_table(1, G_WIN).size();
            CHECK_EQ(blocks, w.ring_pos / B + w.ring_blocks);
            std::map<int32_t, int64_t> content;
            for (int64_t p = 0; p < n; ++p) {
                const int32_t s = r.slot(p);
                REQUIRE(s >= 0);
                content[s] = p;                    /* a ring slot's newest write wins */
            }
            std::set<int64_t> held;
            for (const auto& kvp : content) held.insert(kvp.second);
            /* The last W positions of the generation are among what the slots hold... */
            for (int64_t p = n - w_tokens; p < n; ++p) CHECK(held.count(p) > 0);
            /* ...and nothing outside the rule is. */
            std::set<int64_t> want;
            for (int64_t p = 0; p < std::min<int64_t>(P, n); ++p) want.insert(p);
            for (int64_t p = P; p < std::min(w.ring_pos, n); ++p) want.insert(p);
            for (int64_t p = std::max<int64_t>(w.ring_pos, n - span); p < n; ++p) want.insert(p);
            CHECK(held == want);
            kv.free_sequence(1);
        }
    }
}

/* The card's advance (core/sched/advance_impl.h, adv_groups) runs AFTER the batch builder and
 * overwrites every group's `seqused` and slot_mapping, so for a runtime window it must reproduce
 * the ring's mapping. Writing used = ctx + n (177 for a 176-token held span) and mapping a wrapped
 * position linearly would make the decode walk bt[ceil(used/16)-1] = bt[11], the first entry past
 * the 11-block table, which the builder pads with -1, and fault the card below the KV pool. With
 * block 16 and the ring 8 blocks (128 tokens), the table is the reference's 3 blocks plus the
 * ring's 8. */
TEST(rswa_advance_keeps_the_ring_span_and_slots) {
    const int64_t B = 16, PITCH = 32;
    const int32_t ring_pos = 48, ring_blocks = 8, table_len = 11;

    int32_t pos[5] = { 176, 177, 183, 191, 192 };   /* wrapped ring positions */
    const int32_t base = 176, n = 5;
    AdvanceArgs a;
    std::vector<char> slab(4096, 0);
    const int64_t used_off = 0, slot_off = 16, bt_off = slot_off + 8 * 4;
    const int64_t first_off = bt_off + PITCH * 4, pos_off = first_off + 4;
    const int64_t rpos_off = pos_off + 32, tlen_off = rpos_off + 4;
    const int64_t grp_off = tlen_off + 8;
    a.slab = slab.data();
    a.pos_off = pos_off; a.grp_off = grp_off;
    a.n_groups = 1;
    std::memcpy(slab.data() + pos_off, pos, sizeof pos);
    int32_t* bt = reinterpret_cast<int32_t*>(slab.data() + bt_off);
    for (int32_t b = 0; b < table_len; ++b) bt[b] = b;      /* blocks 0..10 */
    for (int64_t b = table_len; b < PITCH; ++b) bt[b] = -1; /* the builder's padding */
    *reinterpret_cast<int32_t*>(slab.data() + first_off) = 0;
    /* Per sequence, as the builder fills them: this row's floor and its table's length. */
    *reinterpret_cast<int32_t*>(slab.data() + rpos_off) = ring_pos;
    *reinterpret_cast<int32_t*>(slab.data() + tlen_off) = (int32_t)table_len;
    AdvGroup* g = reinterpret_cast<AdvGroup*>(slab.data() + grp_off);
    g->block_size = (int32_t)B; g->paged = 1; g->pitch = (int32_t)PITCH;
    g->slot_off = slot_off; g->used_off = used_off; g->bt_off = bt_off;
    g->first_off = first_off; g->rpos_off = rpos_off; g->tlen_off = tlen_off;
    g->ring = 1; g->ring_blocks = ring_blocks;

    adv_groups(a, 0, 0, base, n);

    /* used stops at the table: the ring holds 3 reference blocks + 8 ring blocks = 176 tokens,
     * while the context is 181. Without the clamp the decode reads bt[11] = -1. */
    CHECK_EQ(*reinterpret_cast<int32_t*>(slab.data() + used_off), 176);
    const int32_t* slot = reinterpret_cast<const int32_t*>(slab.data() + slot_off);
    /* A position at or past ring_pos maps into the ring, modulo its blocks: 176 -> block 3
     * (slot 48), 192 -> block 4 (slot 64), never the linear (p - first)/block = 11..12. */
    CHECK_EQ(slot[0], 48);   /* 176: ring block 3, offset 0 */
    CHECK_EQ(slot[1], 49);   /* 177 */
    CHECK_EQ(slot[2], 55);   /* 183 */
    CHECK_EQ(slot[3], 63);   /* 191: last slot of ring block 3 */
    CHECK_EQ(slot[4], 64);   /* 192: ring block 4, offset 0 */
    /* Every slot is a real slot of the table (never -1). */
    for (int32_t j = 0; j < n; ++j) CHECK(slot[j] >= 0);

    /* The same row without a runtime window keeps the linear mapping: used = base + n, and the
     * wrapped position lands past the table. */
    g->ring = 0;
    adv_groups(a, 0, 0, base, n);
    CHECK_EQ(*reinterpret_cast<int32_t*>(slab.data() + used_off), 181);
    CHECK_EQ(slot[0], -1);
}

/* The plan's P4 gate "flat KV bytes after prefill+128" at the KV-manager level, at the geometry
 * R-SWA deploys at (block 16, W 128) and the P4a prompt size (a 176-token prompt = the 11-block
 * table of the fault). The scheduler arms `visible_lo` = the prompt size at admission; the floor
 * here is block-aligned so the visible set is exactly [0, P) U the last W positions, count P + W.
 * The decode runs far past prefill + W and pins, every step:
 *   - the visible set is exactly [0, P) plus the newest W positions, and its count is P + W;
 *   - the block table is flat once the ring fills: ceil(P/B) reference + W/B ring blocks, never
 *     one more (the sequence's KV bytes stop growing after prefill + 128);
 *   - every ring write lands where adv_groups maps it (core/sched/advance_impl.h): block
 *     ring_pos/B + ((p - ring_pos)/B) % ring_blocks -- so the manager's rule and the card's rule
 *     are checked against the same table and positions;
 *   - no write ever lands on a slot index >= table_len * block: the table is allocated from a
 *     fresh pool so its physical ids are 0..len-1 (the min-heap), which makes that bound literal;
 *     the un-ring'ed linear mapping runs past it from position 304 on, which is the R-SWA fault
 *     class (used past the ring floor walking into the builder's -1 padding). */
TEST(rswa_flat_kv_bytes_after_prefill_plus_128) {
    const int64_t B = 16, W = 128, P = 176;         /* P is block-aligned: the set is exact */
    const int64_t ring_blocks = W / B;              /* 8 */
    const int64_t prefill_blocks = P / B;           /* 11: the P4a table */
    const int64_t flat_blocks = prefill_blocks + ring_blocks;   /* 19 = 304 tokens */
    const KVGroupInfo grp = window_group(B, W);
    const std::vector<KVGroupInfo> g = groups_of(grp);
    Config cfg = window_config();
    Pools pools;
    CHECK_OK(configure_pools(pools, cfg));
    KVManager kv;
    CHECK_OK(kv.declare_runtime_window(G_WIN, W));
    CHECK_OK(kv.configure(g, cfg, pools.kv()));
    CHECK_OK(kv.add_sequence(1));
    CHECK_OK(kv.set_window(1, P, W));
    const Ring r{ &kv, KVGeom::of(g), G_WIN };
    const SeqWindow w = r.window();
    CHECK_EQ(w.visible_lo, P);
    CHECK_EQ(w.ring_pos, P);                        /* aligned floor: the ring starts at P */
    CHECK_EQ(w.ring_blocks, ring_blocks);
    CHECK_EQ(w.ring_span(B), W);

    /* Prefill: 11 blocks, identity physical ids (fresh min-heap pool, one sequence). */
    CHECK_OK(kv.ensure(1, P));
    const std::vector<int32_t> bt0 = kv.block_table(1, G_WIN);
    REQUIRE_EQ((long long)bt0.size(), prefill_blocks);
    for (int64_t b = 0; b < prefill_blocks; ++b) CHECK_EQ(bt0[(size_t)b], (int32_t)b);

    std::map<int32_t, int64_t> content;
    for (int64_t p = 0; p < P; ++p) {
        const int32_t s = r.slot(p);
        REQUIRE(s >= 0);
        CHECK(s < prefill_blocks * B);
        content[s] = p;
    }

    /* Decode far past prefill + W. */
    const int64_t L = 1000;
    int64_t bad_step = -1, bad_map = -1, bad_pad = -1, bad_size = -1;
    int32_t max_slot = -1;
    for (int64_t p = P; p < P + L; ++p) {
        CHECK_OK(kv.ensure(1, p + 1, p));
        const int64_t n = p + 1;
        const std::vector<int32_t>& bt = kv.block_table(1, G_WIN);
        const int64_t table_len = (int64_t)bt.size();

        /* Table length: the reference's 11 blocks, plus the ring's filling one block at a time,
         * flat at 19 once the newest W positions span the whole ring. It never exceeds 19. */
        const int64_t want_blocks =
            n <= w.ring_pos ? (n + B - 1) / B
                            : w.ring_pos / B + std::min(ring_blocks, (n - w.ring_pos + B - 1) / B);
        if (table_len != want_blocks) bad_size = n;
        CHECK(table_len <= flat_blocks);
        if (n >= w.ring_pos + w.ring_span(B)) {
            CHECK_EQ(table_len, flat_blocks);
            CHECK_EQ(kv.held_blocks(G_WIN), flat_blocks);
        }

        /* The write itself: the ring rule, and never off the table. */
        const int64_t bi = w.ring_pos / B + ((p - w.ring_pos) / B) % ring_blocks;
        const int32_t s = r.slot(p);
        REQUIRE(s >= 0);
        if (s != (int32_t)(bt[(size_t)bi] * B + p % B)) bad_map = n;
        if (bi >= table_len || s >= table_len * B) bad_pad = n;
        if (s > max_slot) max_slot = s;
        content[s] = p;

        /* The visible set: exactly [0, P) plus the newest W positions; count P + min(n - P, W),
         * flat P + W once the ring is full. */
        std::set<int64_t> held, want;
        for (const auto& kvp : content) held.insert(kvp.second);
        for (int64_t q = 0; q < P; ++q) want.insert(q);
        for (int64_t q = std::max(w.ring_pos, n - w.ring_span(B)); q < n; ++q) want.insert(q);
        if (held != want) bad_step = n;
        CHECK_EQ((long long)held.size(), P + std::min(n - P, W));
    }
    CHECK_EQ(bad_step, -1ll);
    CHECK_EQ(bad_map, -1ll);
    CHECK_EQ(bad_pad, -1ll);
    CHECK_EQ(bad_size, -1ll);
    CHECK(max_slot < flat_blocks * B);              /* 303 < 304: no write leaves the flat KV */
    CHECK_EQ((long long)kv.block_table(1, G_WIN).size(), flat_blocks);
    CHECK_EQ(kv.first_block_pos(1, G_WIN), 0ll);

    /* The fault class: the linear rule the un-ring'ed advance used maps the same positions past
     * the 19-block table from position 304 (the second ring lap) on, where the table has nothing
     * -- the slot lookup returns -1 into the builder's padding. */
    const Ring r_lin{ &kv, KVGeom::of(g), G_WIN };
    for (int64_t p = 304; p < P + L; ++p)
        CHECK_EQ(r_lin.geo.slot(G_WIN, kv.block_table(1, G_WIN), 0, p), -1);
    /* The ring rule keeps every one of them on a real slot, cycling over the ring span. */
    for (int64_t p = w.ring_pos; p < P + L; ++p) {
        CHECK(r.slot(p) >= 0);
        CHECK_EQ(r.slot(p), r.slot(p + w.ring_span(B)));   /* one lap: same slot, in place */
        if (p + 1 < P + L) CHECK(r.slot(p) != r.slot(p + 1));
    }

    /* Cross-check with the card's rule (adv_groups) on the manager's own table: the same positions
     * through the slab path must produce the manager's slots, and `used` must stop at the table. */
    {
        const std::vector<int32_t>& bt = kv.block_table(1, G_WIN);
        const int32_t tlen = (int32_t)bt.size();
        std::vector<char> slab(32768, 0);
        const int64_t pos_off = 0, slot_off = (int64_t)L * 4;
        const int64_t bt_off = slot_off + (int64_t)L * 4;
        const int64_t first_off = bt_off + (int64_t)tlen * 4;
        const int64_t rpos_off = first_off + 8, tlen_off = rpos_off + 8, used_off = tlen_off + 8;
        const int64_t grp_off = (used_off + 8 + 7) & ~(int64_t)7;   /* AdvGroup holds int64 */
        AdvanceArgs a;
        a.slab = slab.data();
        a.pos_off = pos_off; a.grp_off = grp_off; a.n_groups = 1;
        int32_t* pos = reinterpret_cast<int32_t*>(slab.data() + pos_off);
        for (int64_t j = 0; j < L; ++j) pos[j] = (int32_t)(P + j);
        std::memcpy(slab.data() + bt_off, bt.data(), (size_t)tlen * 4);
        *reinterpret_cast<int32_t*>(slab.data() + first_off) = 0;
        *reinterpret_cast<int32_t*>(slab.data() + rpos_off) = (int32_t)w.ring_pos;
        *reinterpret_cast<int32_t*>(slab.data() + tlen_off) = tlen;
        AdvGroup* ag = reinterpret_cast<AdvGroup*>(slab.data() + grp_off);
        ag->block_size = (int32_t)B; ag->paged = 1; ag->pitch = tlen;
        ag->slot_off = slot_off; ag->used_off = used_off; ag->bt_off = bt_off;
        ag->first_off = first_off; ag->rpos_off = rpos_off; ag->tlen_off = tlen_off;
        ag->ring = 1; ag->ring_blocks = (int32_t)ring_blocks;

        adv_groups(a, 0, 0, (int32_t)P, (int32_t)L);
        CHECK_EQ(*reinterpret_cast<int32_t*>(slab.data() + used_off), (int32_t)(flat_blocks * B));
        const int32_t* slot = reinterpret_cast<const int32_t*>(slab.data() + slot_off);
        int64_t mismatches = 0, negatives = 0;
        for (int64_t j = 0; j < L; ++j) {
            if (slot[j] != r.slot(P + j)) ++mismatches;
            if (slot[j] < 0) ++negatives;
            CHECK(slot[j] < tlen * B);              /* every write inside the flat KV bytes */
        }
        CHECK_EQ(mismatches, 0ll);
        CHECK_EQ(negatives, 0ll);

        /* Without the ring the same advance walks off the padded table (the P4a fault). */
        ag->ring = 0;
        adv_groups(a, 0, 0, (int32_t)P, (int32_t)L);
        CHECK_EQ(*reinterpret_cast<int32_t*>(slab.data() + used_off), (int32_t)(P + L));
        CHECK_EQ(slot[304 - P], -1);                /* position 304: bt[19] is padding */
    }

    /* The reference is untouched by a thousand ring laps, and free returns the flat set. */
    const std::vector<int32_t>& bt = kv.block_table(1, G_WIN);
    for (int64_t p = 0; p < P; ++p) {
        const int32_t slot = (int32_t)(bt[(size_t)(p / B)] * B + p % B);
        REQUIRE(content.count(slot) > 0);
        CHECK_EQ(content[slot], p);
    }
    const int64_t free0 = kv.free_blocks(G_WIN);
    kv.free_sequence(1);
    CHECK_EQ(kv.free_blocks(G_WIN), free0 + flat_blocks);
}

RAD_TEST_MAIN()
