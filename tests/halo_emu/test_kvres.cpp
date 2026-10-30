/* tests/halo_emu (from tests/r4d_emu/test_kvres.cpp): kv_resolve (libhalo/halo_kv_resolve.hip) on the host SIMT emulator, every operand
 * held to a port of libref's ref_kv_resolve (libref/ref_kvstream.cpp) byte for byte: the rewritten
 * selection, both residency maps, the slot epochs and heat counters, the control block, the mirror
 * (untouched) and every byte of the ring.
 *
 * The fixtures are the shapes the claim list has to survive:
 *
 *   - a partly resident pool, an eviction, a second chance: the ordinary decode call;
 *   - identity placement and its two exits: ids that fit the ring and take their own slot, an id
 *     whose slot is occupied by a different block (a clash, which walks from the hand without
 *     evicting), ids past the ring, and a full ring of clashes (which is CLOCK, unchanged);
 *   - 2052 claims (four rows of 513, what `--max-num-seqs 4` selects) and 8000 claims at a launch
 *     wide enough to derive a list of 8192 ids;
 *   - 12 claims into 8 slots: more claims than the ring holds, so the unplaceable ones revert
 *     through the selection and the overflow latch is set;
 *   - the two-bit heat counter: a slot hit twice surviving two sweep passes where the one-bit
 *     CLOCK would evict it, a saturated counter halving 3 -> 1 on a pass, and a saturated counter
 *     outlasting a slot that was only placed last call;
 *   - one bump a slot a call however many rows name it, which is the one place the device's
 *     `atomicExch` and a per-entry reference would part company;
 *   - 8200 claims at a launch whose bound (32 rows x 513) exceeds the workgroup's LDS room, so the
 *     claims are gathered into the row's scratch and sorted there; the slots must still be
 *     libref's.
 *
 * A claim count is bounded by the launch's entries, `sum_q nsel[q] <= M * sel_pitch`, so every
 * fixture's claim count is a number the kernel derives rather than one it assumes. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "halo_plugin.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" int halo_rad_kv_resolve(const RadArgs* a, RadStream stream);

namespace {

enum { kNOB = 0xEE };

RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[i];
        t.stride[i] = st;
        st *= shape[i];
    }
    return t;
}

/* Block `b`'s bytes: a pattern whose block digit is recoverable, so a copy that landed in the wrong
 * slot shows up in the ring's bytes. */
uint8_t block_byte(int64_t b, int64_t i) { return (uint8_t)((29 * b + i) & 0xff); }

struct Fix {
    int64_t M = 1, blocks = 1, slots = 1, pitch = 1, bbytes = 8;
    std::vector<int32_t> sel, nsel, res, sblk, sst, ctl;
    std::vector<uint8_t> host, pool;

    /* A fresh pool: the maps say what is resident, a free slot holds the marker. */
    void shape(int64_t M_, int64_t blocks_, int64_t slots_, int64_t pitch_, int64_t bbytes_) {
        M = M_; blocks = blocks_; slots = slots_; pitch = pitch_; bbytes = bbytes_;
        sel.assign((size_t)(M * pitch), -1);
        nsel.assign((size_t)M, 0);
        res.assign((size_t)blocks, -1);
        sblk.assign((size_t)slots, -1);
        sst.assign((size_t)(slots * 2), 0);
        ctl.assign(16, 0);
        host.assign((size_t)(blocks * bbytes), 0);
        pool.assign((size_t)(slots * bbytes), (uint8_t)kNOB);
        for (int64_t b = 0; b < blocks; ++b)
            for (int64_t i = 0; i < bbytes; ++i) host[(size_t)(b * bbytes + i)] = block_byte(b, i);
    }
    /* Make slot `s` hold block `b`: the map and the bytes, as a resident block leaves them. */
    void resident(int64_t s, int64_t b) {
        sblk[(size_t)s] = (int32_t)b;
        res[(size_t)b] = (int32_t)s;
        for (int64_t i = 0; i < bbytes; ++i) pool[(size_t)(s * bbytes + i)] = block_byte(b, i);
    }
};

/* The heat counter's two writes, copied from libref: a use bumps by one and saturates at 3, a
 * sweep pass halves, and a negative value reads as cold. */
int32_t heat_up(int32_t h) { return h < 0 ? 1 : (h < 3 ? h + 1 : 3); }
int32_t heat_down(int32_t h) { return h > 0 ? (h >> 1) : 0; }

/* libref's ref_kv_resolve over a fixture, step for step (libref/ref_kvstream.cpp): the same epoch,
 * the same one claim per distinct block, the same sorted miss list, the same free-slot-then-CLOCK
 * sweep over the same two-bit heat counter, the same revert of what it could not place, the same
 * copies and rewrite. The fixtures are legal calls, so the refusals are not repeated here. */
void ref_resolve(Fix& f) {
    const int64_t M = f.M, blocks = f.blocks, slots = f.slots, pitch = f.pitch, bbytes = f.bbytes;
    int32_t* const R = f.res.data();
    int32_t* const SB = f.sblk.data();
    int32_t* const ST = f.sst.data();
    int32_t* const C = f.ctl.data();
    const int64_t stp = 2;

    std::vector<int64_t> off, ent;
    for (int64_t q = 0; q < M; ++q)
        for (int64_t i = 0; i < f.nsel[(size_t)q]; ++i) {
            off.push_back(q * pitch + i);
            ent.push_back(f.sel[(size_t)(q * pitch + i)]);
        }

    const int32_t epoch = (int32_t)((uint32_t)C[0] + 1u);

    std::vector<int32_t> miss;
    for (size_t e = 0; e < ent.size(); ++e) {
        const int32_t b = (int32_t)ent[e];
        const int32_t s = R[b];
        if (s >= 0) {
            /* One bump a slot a call, not one an entry (libref's step 2). */
            if (ST[(size_t)s * stp] != epoch) {
                ST[(size_t)s * stp] = epoch;
                ST[(size_t)s * stp + 1] = heat_up(ST[(size_t)s * stp + 1]);
            }
        } else if (s == -1) {
            R[b] = -2;
            miss.push_back(b);
        }
    }
    const int64_t lookups = (int64_t)ent.size();
    std::sort(miss.begin(), miss.end());

    int64_t free_left = 0;
    for (int64_t s = 0; s < slots; ++s)
        if (SB[s] < 0 && ST[(size_t)s * stp] != epoch) ++free_left;
    int64_t hand = C[1];
    int64_t placed = 0;
    bool overflow = false;
    for (size_t k = 0; k < miss.size(); ++k) {
        const int32_t b = miss[k];
        int64_t victim = -1;
        /* Identity first: the block's own slot if the id fits the ring and that slot is free (or
         * already holds it), hand untouched (libref's step 4). */
        if ((int64_t)b < slots && ST[(int64_t)b * stp] != epoch && (SB[b] < 0 || SB[b] == b))
            victim = b;
        if (victim < 0 && free_left > 0) {
            int64_t h = hand;
            for (int64_t scanned = 0; scanned < slots; ++scanned) {
                const int64_t s = h;
                h = h + 1 == slots ? 0 : h + 1;
                if (ST[(size_t)s * stp] == epoch) continue;
                if (SB[s] >= 0) continue;
                victim = s;
                hand = h;
                break;
            }
            if (victim < 0) free_left = 0;
        }
        if (victim < 0) {
            for (int64_t scanned = 0; scanned < 3 * slots; ++scanned) {
                const int64_t s = hand;
                hand = hand + 1 == slots ? 0 : hand + 1;
                if (ST[(size_t)s * stp] == epoch) continue;
                if (SB[s] >= 0 && ST[(size_t)s * stp + 1] > 0) {
                    ST[(size_t)s * stp + 1] = heat_down(ST[(size_t)s * stp + 1]);
                    continue;
                }
                victim = s;
                break;
            }
        }
        if (victim < 0) { overflow = true; break; }
        if (SB[victim] < 0 && free_left > 0) --free_left;
        if (SB[victim] >= 0) R[SB[victim]] = -1;
        SB[victim] = b;
        ST[(size_t)victim * stp] = epoch;
        ST[(size_t)victim * stp + 1] = heat_up(ST[(size_t)victim * stp + 1]);
        R[b] = (int32_t)victim;
        ++placed;
    }
    for (size_t k = (size_t)placed; k < miss.size(); ++k) R[miss[k]] = -1;

    for (int64_t k = 0; k < placed; ++k) {
        const int32_t b = miss[(size_t)k];
        std::memcpy(f.pool.data() + (int64_t)R[b] * bbytes, f.host.data() + (int64_t)b * bbytes,
                    (size_t)bbytes);
    }
    for (size_t e = 0; e < ent.size(); ++e) {
        const int32_t s = R[ent[e]];
        f.sel[(size_t)off[e]] = s >= 0 ? s : 0;
    }

    int64_t used = 0;
    for (int64_t s = 0; s < slots; ++s)
        if (ST[(size_t)s * stp] == epoch) ++used;
    const int64_t uni = used + (int64_t)miss.size() - placed;
    C[0] = epoch;
    C[1] = (int32_t)hand;
    C[2] = (int32_t)placed;
    if (overflow) C[3] = 1;
    if (uni > C[10]) C[10] = (int32_t)uni;
    if (placed > C[11]) C[11] = (int32_t)placed;
    if ((int64_t)miss.size() > C[12]) C[12] = (int32_t)miss.size();
    uint64_t* const ctr = (uint64_t*)(C + 4);
    ctr[0] += (uint64_t)placed;
    ctr[1] += (uint64_t)lookups;
    ctr[2] += 1u;
}

/* The device entry point over a fixture. Its scratch is the row's `next_pow2(slots)` miss-list ids
 * (the list is sorted there when a launch is too wide for the shared claim list) then `slots`
 * slots. */
int dev_resolve(Fix& f) {
    int64_t nlist = 1;
    while (nlist < f.slots) nlist <<= 1;
    std::vector<int32_t> scratch((size_t)(nlist + f.slots), 0);
    RadTensor t[8] = { tens(f.sel.data(), RAD_I32, { f.M, f.pitch }),
                       tens(f.nsel.data(), RAD_I32, { f.M }),
                       tens(f.res.data(), RAD_I32, { f.blocks }),
                       tens(f.sblk.data(), RAD_I32, { f.slots }),
                       tens(f.sst.data(), RAD_I32, { f.slots, 2 }),
                       tens(f.ctl.data(), RAD_I32, { 16 }),
                       tens(f.host.data(), RAD_U8, { f.blocks, f.bbytes }),
                       tens(f.pool.data(), RAD_U8, { f.slots, f.bbytes }) };
    RadParam p[5]{};
    p[0] = { "M", RAD_P_INT, f.M };
    p[1] = { "blocks", RAD_P_INT, f.blocks };
    p[2] = { "slots", RAD_P_INT, f.slots };
    p[3] = { "sel_pitch", RAD_P_INT, f.pitch };
    p[4] = { "block_bytes", RAD_P_INT, f.bbytes };
    RadArgs a{};
    a.t = t;
    a.n_t = 8;
    a.p = p;
    a.n_p = 5;
    a.scratch = scratch.data();
    a.scratch_bytes = (int64_t)scratch.size() * 4;
    return halo_rad_kv_resolve(&a, nullptr);
}

size_t mismatch(const std::vector<int32_t>& a, const std::vector<int32_t>& b, const char* what,
                const char* name) {
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            if (!bad)
                std::printf("  FAIL %-42s %s[%zu]: %d, libref %d\n", name, what, i, a[i], b[i]);
            ++bad;
        }
    return bad;
}

size_t mismatch_bytes(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                      const char* what, const char* name) {
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            if (!bad)
                std::printf("  FAIL %-42s %s[%zu]: %u, libref %u\n", name, what, i, a[i], b[i]);
            ++bad;
        }
    return bad;
}

/* One fixture, both ways, every operand compared. */
int check(const char* name, Fix f) {
    std::printf("  ..   %s\n", name);
    std::fflush(stdout);
    Fix ref = f;
    ref_resolve(ref);
    const int rc = dev_resolve(f);
    if (rc != RAD_OK) {
        std::printf("  FAIL %-42s refused (%d)\n", name, rc);
        return 1;
    }
    size_t bad = 0;
    bad += mismatch(f.sel, ref.sel, "sel", name);
    bad += mismatch(f.res, ref.res, "res", name);
    bad += mismatch(f.sblk, ref.sblk, "slot_block", name);
    bad += mismatch(f.sst, ref.sst, "slot_state", name);
    bad += mismatch(f.ctl, ref.ctl, "ctl", name);
    bad += mismatch_bytes(f.pool, ref.pool, "kv_cache", name);
    bad += mismatch_bytes(f.host, ref.host, "host", name);
    if (bad) return 1;
    std::printf("  ok   %-42s %lld claims, %lld placed\n", name, (long long)ref.ctl[12],
                (long long)ref.ctl[2]);
    return 0;
}

/* `n` distinct block ids across `M` rows of `pitch` entries, contiguously from `first`. Each row
 * declares exactly the entries this fills, since `nsel` is what says an entry is valid -- and a
 * declared entry holding -1 is a state fault, not padding. */
void fill_claims(Fix& f, int64_t first, int64_t n, int64_t per_row) {
    int64_t k = 0;
    for (int64_t q = 0; q < f.M; ++q) {
        int64_t take = per_row;
        if (take > n - k) take = n - k;
        if (take < 0) take = 0;
        f.nsel[(size_t)q] = (int32_t)take;
        for (int64_t i = 0; i < take; ++i, ++k) f.sel[(size_t)(q * f.pitch + i)] = (int32_t)(first + k);
    }
}

int run_all() {
    int fails = 0, cases = 0;
    ++cases;
    {
        Fix f;
        f.shape(2, 8, 4, 4, 8);
        f.resident(0, 2);
        f.resident(3, 5);
        f.sel = { 2, 5, 6, -1, 2, 4, -1, -1 };
        f.nsel = { 3, 2 };
        f.sst = { 0, 1, 0, 0, 0, 0, 0, 0 };
        fails += check("a decode call: two hits, two misses, ascending", f);
    }
    ++cases;
    {
        /* Every slot resident and referenced: the sweep clears four bits and takes the slot the
         * hand started on, one full turn later. */
        Fix f;
        f.shape(1, 8, 4, 2, 4);
        for (int64_t s = 0; s < 4; ++s) f.resident(s, s + 4);
        f.sel = { 0, 1 };
        f.nsel = { 2 };
        f.sst = { 0, 1, 0, 1, 0, 1, 0, 1 };
        fails += check("second chance: a full ring clears and wraps", f);
    }
    ++cases;
    {
        /* 2052 claims, the four-sequence decode shape. */
        Fix f;
        f.shape(4, 4096, 2100, 513, 8);
        fill_claims(f, 100, 2052, 513);
        fails += check("2052 claims, every slot free", f);
    }
    ++cases;
    {
        /* A launch wide enough that the list is 8192 ids (32 x 513 = 16416 is past the workgroup's
         * LDS room), with 8000 claims inside it. */
        Fix f;
        f.shape(32, 12000, 8100, 513, 8);
        fill_claims(f, 200, 8000, 250);
        fails += check("8000 claims inside a derived 8192-id list", f);
    }
    ++cases;
    {
        /* More claims than the ring: the eight smallest ids take the eight free slots, the rest
         * revert through the selection and the overflow latches. */
        Fix f;
        f.shape(1, 4096, 8, 12, 8);
        f.sel = { 4090, 3, 2000, 7, 1, 4000, 5, 9, 11, 2, 100, 500 };
        f.nsel = { 12 };
        fails += check("12 claims into 8 slots: the smallest eight", f);
    }
    ++cases;
    {
        /* 8200 claims against a list of 8192: the launch's bound is past the workgroup's LDS room, so
         * the kernel gathers the claims from the selection into the row's scratch and sorts them
         * there. Same slots as libref's sort of all 8200, and no walk of `blocks`. */
        Fix f;
        f.shape(32, 12000, 8300, 513, 8);
        fill_claims(f, 0, 8200, 257);
        fails += check("8200 claims: sorted in the scratch above the list", f);
    }
    ++cases;
    {
        /* Identity's two ways out in one call. Block 1's own slot holds block 6, so it is a clash
         * and takes the first free slot after the hand (0) -- never an eviction. Block 2's slot is
         * free, so it takes slot 2 even though the hand has moved to 1. */
        Fix f;
        f.shape(1, 12, 4, 4, 8);
        f.resident(1, 6);
        f.sel = { 1, 2, 8, -1 };
        f.nsel = { 3 };
        f.sst = { 0, 1, 0, 0, 0, 0, 0, 0 };
        fails += check("identity clash falls back to a free slot, then identity again", f);
    }
    ++cases;
    {
        /* Ids past the ring, the depth case: nothing can be identity, every miss walks from the
         * hand. */
        Fix f;
        f.shape(1, 16, 4, 5, 8);
        f.sel = { 9, 10, 11, 12, 13 };
        f.nsel = { 5 };
        fails += check("ids past the ring: the old free-slot walk, unchanged", f);
    }
    ++cases;
    {
        /* A full ring of clashing slots: identity has nowhere to go and CLOCK picks the victim the
         * hand walks to, halving one counter on the way. */
        Fix f;
        f.shape(1, 12, 3, 2, 8);
        f.resident(0, 5);
        f.resident(1, 6);
        f.resident(2, 7);
        f.sel = { 0, 1 };
        f.nsel = { 2 };
        f.sst = { 0, 1, 0, 1, 0, 1 };
        fails += check("identity clash on a full ring: CLOCK, unchanged", f);
    }
    ++cases;
    {
        /* A ring wider than one thread's chunk with its free slots far from the hand: the sweep
         * walks the chunk summaries (2100 slots is 3 a thread) and must still name the first free
         * slot from the hand. Slots 0..2048 hold blocks, 2049 and 2050 are free, and the misses
         * are ids past the ring, so identity cannot place them. */
        Fix f;
        f.shape(1, 6000, 2100, 3, 8);
        for (int64_t s = 0; s < 2049; ++s) f.resident(s, s);
        f.sel = { 4000, 4001, 4002 };
        f.nsel = { 3 };
        fails += check("free slots in the last chunk, hand at 0", f);
    }
    ++cases;
    {
        /* The same ring with the hand past the free region: the walk wraps through the chunk
         * summaries to slots 0 and 1, in that order, and leaves the hand at 2. */
        Fix f;
        f.shape(1, 6000, 2100, 3, 8);
        for (int64_t s = 2; s < 2100; ++s) f.resident(s, s);
        f.sel = { 4000, 4001, 4002 };
        f.nsel = { 3 };
        f.ctl[1] = 2099;
        fails += check("free slots at slot 0, hand at the end: the walk wraps", f);
    }
    ++cases;
    {
        /* Heat, the discriminating case: slot 0 was hit twice (counter 2), slots 1 and 2 once each.
         * A one-bit CLOCK clears all three and evicts slot 0 on the second pass; the counter halves
         * 2 -> 1 -> 0, so slot 0 survives two passes and slot 1 is the victim. */
        Fix f;
        f.shape(1, 32, 3, 1, 8);
        f.resident(0, 10);
        f.resident(1, 11);
        f.resident(2, 12);
        f.sst = { 0, 2, 0, 1, 0, 1 };
        f.sel = { 20 };
        f.nsel = { 1 };
        fails += check("heat: a slot hit twice survives two passes, CLOCK would take it", f);
    }
    ++cases;
    {
        /* The halving: a saturated 3 is 1 after one pass, not 0, and the cold slot behind it is the
         * victim. A one-bit CLOCK clears slot 0 and evicts it. */
        Fix f;
        f.shape(1, 32, 3, 1, 8);
        f.resident(0, 10);
        f.resident(1, 11);
        f.resident(2, 12);
        f.sst = { 0, 3, 0, 0, 0, 0 };
        f.sel = { 20 };
        f.nsel = { 1 };
        fails += check("heat: a saturated counter halves 3 -> 1 on one pass", f);
    }
    ++cases;
    {
        /* A saturated counter outlasts a slot that was only placed last call: {3,1,1,1} walks two
         * full laps, leaving the 3 at 1 then 0, and takes slot 1 (counter 1 -> 0 -> victim). Earning
         * the same state from placements is what test/kvstream's multi-call case does. */
        Fix f;
        f.shape(1, 32, 4, 1, 8);
        for (int64_t s = 0; s < 4; ++s) f.resident(s, 10 + s);
        f.sst = { 0, 3, 0, 1, 0, 1, 0, 1 };
        f.sel = { 20 };
        f.nsel = { 1 };
        fails += check("heat: a saturated counter outlasts a fresh placement", f);
    }
    ++cases;
    {
        /* One bump a slot a call: block 10 is resident and named by both entries, so its counter
         * gains one step (1 -> 2), not two, where a per-entry bump would disagree with the device's
         * first-stamp `atomicExch`. */
        Fix f;
        f.shape(1, 32, 3, 2, 8);
        f.resident(0, 10);
        f.resident(1, 11);
        f.resident(2, 12);
        f.sst = { 0, 1, 0, 0, 0, 0 };
        f.sel = { 10, 10 };
        f.nsel = { 2 };
        fails += check("heat: a block two entries name gains one step, not two", f);
    }
    ++cases;
    {
        /* Every miss takes its own slot with a hit and a hand that is not at 0 in play: the
         * identity fast path must leave the hand exactly where it was and still stamp its slots,
         * because the fallback never runs. */
        Fix f;
        f.shape(1, 12, 8, 6, 8);
        f.resident(3, 9);              /* a resident block the selection does not name */
        f.ctl[1] = 5;
        f.sel = { 0, 2, 4, 6, 7 };
        f.nsel = { 5 };
        fails += check("identity for every miss, hand untouched", f);
    }
    std::printf("%d of %d fixtures match libref\n", cases - fails, cases);
    return fails ? 1 : 0;
}

}  /* namespace */

int main() {
    const int fails = run_all();
    return fails;
}
