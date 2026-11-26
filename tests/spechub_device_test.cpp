/* tests/spechub_device_test.cpp -- the spechub plugin's DEVICE verify rows against their host rows,
 * on a card.
 *
 *   mtp_couple_verify   rows whose state drafted 1..4 children (the chain's maximal coupling for one
 *                       child, recursive rejection sampling for several, abi/rad_mtp_rrs.h), and
 *                       --draft-verify block's chains of 1..8 rows (abi/rad_mtp_block.h)
 *   mtp_trav_verify     --draft-verify traversal, abi/rad_mtp_traversal.h: requests of 1..21 rows
 *
 * The device rows and the host rows run the same shared header code, so the outputs must agree
 * token for token. The traversal walk and the block decision are also compared bit for bit in
 * what they write. Skipped without a card (the host backend), or without the device rows in the
 * plugin. Two tests read the plugin's code object (no card needed): the verify kernels' private
 * segment, since the queue sizes its scratch by it. Run on a card under the shared lock:
 *   flock gpu.lock build/tests/spechub_device_test
 */
#include "rad_test.h"
#include "device/aql.h"

#include "rad_abi.h"
#include "rad_device.h"
#include "rad_mtp_rrs.h"
#include "rad_mtp_traversal.h"
#include "rad_sample.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef RAD_SPECHUB_LIB
#define RAD_SPECHUB_LIB ""
#endif

namespace {

struct Plugin {
    void* handle = nullptr;
    int (*count)() = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;
    bool open(const char* path) {
        if (const char* o = std::getenv("RAD_SPECHUB_LIB_OVERRIDE")) path = o;   /* a baseline build, for RAD_BENCH */
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) return false;
        count = (int (*)())dlsym(handle, "rad_kernel_count");
        at    = (const RadKernelInfo* (*)(int))dlsym(handle, "rad_kernel_at");
        return count && at;
    }
    const RadKernelInfo* row(const char* name) const {
        for (int i = 0; i < count(); ++i) {
            const RadKernelInfo* r = at(i);
            if (r && r->name && std::string(r->name) == name) return r;
        }
        return nullptr;
    }
    ~Plugin() { if (handle) dlclose(handle); }
};

/* A tensor over a host buffer or its device twin. */
struct Buf {
    std::vector<uint8_t> host;
    void* dev = nullptr;
    RadTensor t{};
    Buf(uint32_t dtype, std::vector<int64_t> shape, size_t elem) {
        t.dtype = dtype;
        t.rank = (uint32_t)shape.size();
        int64_t n = 1;
        for (size_t i = 0; i < shape.size(); ++i) { t.shape[i] = shape[i]; n *= shape[i]; }
        int64_t acc = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= shape[i]; }
        host.assign((size_t)n * elem + 16, 0);
    }
    ~Buf() { if (dev) rad_dev_free(dev, RAD_MEM_DEVICE); }
    RadTensor on_host() { RadTensor x = t; x.data = host.data(); return x; }
    RadTensor on_dev(RadStream s) {
        if (!dev) dev = rad_dev_alloc((int64_t)host.size(), RAD_MEM_DEVICE);
        rad_memcpy_async(dev, host.data(), (int64_t)host.size(), s);
        RadTensor x = t; x.data = dev; return x;
    }
    void fetch(RadStream s) { rad_memcpy_async(host.data(), dev, (int64_t)host.size(), s); }
};

RadParam P(const char* k, long long v) { return RAD_INT(k, v); }

uint32_t g_seed = 424242;
uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return (g_seed >> 8) & 0xffffff; }

RadMtpCoupleDraw make_draw(uint64_t seed, uint64_t base, uint64_t seq, int card, uint32_t mode) {
    RadMtpCoupleDraw d{};
    d.seed = seed; d.base = base; d.seq = seq; d.card = card; d.mode = mode;
    return d;
}

/* A device other than the host backend is present. */
bool have_card() {
    RadDeviceProps dp{};
    return rad_dev_count() >= 1 && rad_dev_props(0, &dp) == RAD_OK && !dp.is_host_backend;
}

}  /* namespace */

/* The queue's scratch is sized by a kernel's private segment times every lane it may run. A
 * couple_verify carrying the block row's planes would need 17.8 KB a lane in a workgroup of 128
 * lanes of which one works, and a sampled tree of five or more rows would fault the queue with
 * OUT_OF_RESOURCES once HSA_NO_SCRATCH_RECLAIM keeps the scratch resident. So the code object is
 * pinned: the workgroup kernel keeps a small private segment (the serial helper that needs the big
 * one is a kernel of its own). Runs without a card: it reads the plugin's own code object. */
TEST(the_verify_kernels_ask_for_little_scratch) {
    FILE* f = std::fopen(RAD_SPECHUB_LIB, "rb");
    if (!f) ARTIFACT_MISSING(RAD_SPECHUB_LIB);
    std::vector<unsigned char> img;
    unsigned char buf[65536];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) img.insert(img.end(), buf, buf + n);
    std::fclose(f);
    static const char kMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
    auto at = std::search(img.begin(), img.end(), kMagic, kMagic + sizeof kMagic - 1);
    if (at == img.end()) SKIP("no device code in the plugin");
    const void* co = nullptr;
    size_t cosz = 0;
    std::string why;
    REQUIRE(rad::aql::bundle_code_object(&*at, { "hipv4-amdgcn-amd-amdhsa--gfx1201", "hip-amdgcn-amd-amdhsa--gfx1201",
                                                 "amdgcn-amd-amdhsa--gfx1201" }, &co, &cosz, &why));
    std::vector<rad::aql::KernelMeta> ks;
    REQUIRE(rad::aql::code_object_kernels(co, cosz, &ks, &why));
    int seen = 0, seen_trav = 0;
    for (const auto& k : ks) {
        if (k.symbol.find("trav_verify") != std::string::npos) {
            /* the traversal verify keeps a request in registers and LDS: no scratch at all, so its
             * dispatch never waits for the queue's scratch */
            ++seen_trav;
            CHECK_EQ((unsigned)k.private_size, 0u);
            fprintf(stderr, "  trav_verify private segment %u bytes\n", (unsigned)k.private_size);
            continue;
        }
        if (k.symbol.find("couple_verify") == std::string::npos || k.symbol.find("wide") != std::string::npos)
            continue;
        ++seen;
        CHECK(k.private_size <= 256u);
        fprintf(stderr, "  couple_verify private segment %u bytes\n", (unsigned)k.private_size);
    }
    CHECK_EQ(seen, 1);
    CHECK_EQ(seen_trav, 1);
}

TEST(the_device_verify_row_agrees_with_the_host_row) {
    if (!have_card()) SKIP("no card");
    Plugin pl;
    if (!pl.open(RAD_SPECHUB_LIB))
        ARTIFACT_MISSING(std::string(RAD_SPECHUB_LIB) + " does not load");
    const RadKernelInfo* dver = pl.row("spechub_mtp_couple_verify");
    const RadKernelInfo* hver = pl.row("ref_mtp_couple_verify");
    if (!dver) SKIP("the plugin has no device rows");
    REQUIRE(hver);
    RadStream s{};
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);
    const int SW = (int)RAD_MTP_COUPLE_STATE_WORDS;
    const int64_t VOC = 50000;

    /* ---- verify over states with 1..4 children, at a 24-slot plane and at the sampler's full 1024-slot
     * plane (top_k 0: every slot live on even rows, 40 live slots and holes on odd rows) -- the
     * workgroup's compaction and weights over several chunks, then the host's decision over them */
    for (const int NC : { 24, 1024, 1100 }) {
        const int M = 64;
        const bool wide = NC > 64;      /* 1100: past the workgroup's plane, one lane decides */
        Buf cid(RAD_I32, { M, NC }, 4), cval(RAD_F32, { M, NC }, 4), prm(RAD_U32, { M, 24 }, 4);
        Buf sts(RAD_U32, { M, SW }, 4), tok(RAD_I32, { M }, 4);
        int32_t* pi = (int32_t*)cid.host.data();
        float* pv = (float*)cval.host.data();
        RadSampleParams* pp = (RadSampleParams*)prm.host.data();
        RadMtpCoupleState* ps_ = (RadMtpCoupleState*)sts.host.data();
        int32_t* pt = (int32_t*)tok.host.data();
        for (int m = 0; m < M; ++m) {
            const int live = !wide || m % 2 == 0 ? NC : 40;
            for (int c = 0; c < NC; ++c) {
                if (c >= live) { pi[m * NC + c] = -1; pv[m * NC + c] = -INFINITY; continue; }
                int id;
                bool dup;
                do {
                    id = (int)(rnd() % (wide ? VOC : 40));
                    dup = false;
                    for (int q = 0; q < c; ++q) dup = dup || pi[m * NC + q] == id;
                } while (dup);
                pi[m * NC + c] = id;
                pv[m * NC + c] = 2.0f - (wide ? 0.01f : 0.25f) * c - (rnd() % 100) * 0.01f;
            }
            int32_t qid[8];
            float qv[8];
            for (int c = 0; c < 8; ++c) {
                /* Q's support among the candidates (wide) or the 40-token universe */
                qid[c] = wide ? pi[m * NC + (int)(rnd() % (uint32_t)(live < 60 ? live : 60))] : (int)(rnd() % 40);
                qv[c] = 2.0f - 0.3f * c;
            }
            for (int c = 0; c < 8; ++c)
                for (int q = 0; q < c; ++q) if (qid[q] == qid[c]) qid[c] = 100 + c;
            RadMtpCoupleState s0{};
            rad_mtp_couple_build_q(qid, qv, VOC, &s0);
            s0.draw = make_draw(0x900ull + (uint64_t)m, 3, 1, 0, 1u);
            s0.status = 1; s0.round = 1; s0.pos = 11 + (uint64_t)m;
            const int nk = 1 + m % 4;
            if (nk >= 2) rad_mtp_rrs_draw(&s0, 0, nk); else s0.x = s0.ids[0];
            ps_[m] = s0;
            pp[m].seed = s0.draw.seed;
            pp[m].pos = s0.pos;
            pp[m].flags = m % 16 == 15 ? RAD_SP_GREEDY : 0;
            pt[m] = -9;
        }
        const std::vector<RadParam> pr = { P("M", M), P("n_cand", NC), P("n_vocab", (int)VOC) };
        Buf htok = tok, hst2 = sts;
        {
            RadTensor ts[5] = { cid.on_host(), cval.on_host(), prm.on_host(), hst2.on_host(), htok.on_host() };
            RadArgs a{}; a.t = ts; a.n_t = 5; a.p = pr.data(); a.n_p = (int)pr.size();
            REQUIRE_EQ(hver->launch(&a, s), RAD_OK);
        }
        {
            RadTensor ts[5] = { cid.on_dev(s), cval.on_dev(s), prm.on_dev(s), sts.on_dev(s), tok.on_dev(s) };
            RadArgs a{}; a.t = ts; a.n_t = 5; a.p = pr.data(); a.n_p = (int)pr.size();
            REQUIRE_EQ(dver->launch(&a, s), RAD_OK);
            tok.fetch(s);
            sts.fetch(s);
            REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
        }
        int multi = 0, kept = 0;
        for (int m = 0; m < M; ++m) {
            const int32_t g = ((int32_t*)tok.host.data())[m], w = ((int32_t*)htok.host.data())[m];
            CHECK_EQ(g, w);
            /* the verify leaves the state plane alone */
            RadMtpCoupleState dg, hw;
            std::memcpy(&dg, sts.host.data() + (size_t)m * SW * 4, sizeof dg);
            std::memcpy(&hw, hst2.host.data() + (size_t)m * SW * 4, sizeof hw);
            CHECK(std::memcmp(&dg, &hw, sizeof dg) == 0);
            multi += ps_[m].nkids >= 2 && w != -9;
            kept += w == -9;
        }
        CHECK(multi >= 20);
        CHECK_EQ(kept, 4);   /* the greedy rows keep the ordinary token */
    }
    rad_stream_destroy(s);
}

/* --draft-verify block (abi/rad_mtp_block.h): the device verify row decides a whole chain from its
 * round-1 row and writes the tokens of rows 0 .. tau; the host row is the same shared code over the
 * dense plane. Blocks of 1..8 rows, planes of 24 (the common case), 1000 (live slots compacted over
 * several chunks of the workgroup) and 1100 (past the LDS plane: one thread runs the helper) slots,
 * with holes; tokens must agree row for row, rows the block does not decide keep their sentinel. */
TEST(the_device_block_verify_agrees_with_the_host_row) {
    if (!have_card()) SKIP("no card");
    Plugin pl;
    if (!pl.open(RAD_SPECHUB_LIB))
        ARTIFACT_MISSING(std::string(RAD_SPECHUB_LIB) + " does not load");
    const RadKernelInfo* dver = pl.row("spechub_mtp_couple_verify");
    const RadKernelInfo* hver = pl.row("ref_mtp_couple_verify");
    if (!dver) SKIP("the plugin has no device rows");
    REQUIRE(hver);
    RadStream s{};
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);
    const int SW = (int)RAD_MTP_COUPLE_STATE_WORDS;
    const int64_t VOC = 50000;
    int shortened = 0, whole = 0, writes = 0;
    for (int NC : { 24, 1000, 1100, 1001 }) {   /* 1001: wide, 300..500 live slots a row */
        const bool wide = NC == 1001;
        const int NSEQ = 40, MAXROWS = 9;
        const int M = NSEQ * MAXROWS;
        Buf cid(RAD_I32, { M, NC }, 4), cval(RAD_F32, { M, NC }, 4), prm(RAD_U32, { M, 24 }, 4);
        Buf sts(RAD_U32, { M, SW }, 4), tok(RAD_I32, { M }, 4);
        int32_t* pi = (int32_t*)cid.host.data();
        float* pv = (float*)cval.host.data();
        RadSampleParams* pp = (RadSampleParams*)prm.host.data();
        RadMtpCoupleState* ps_ = (RadMtpCoupleState*)sts.host.data();
        int32_t* pt = (int32_t*)tok.host.data();
        for (int m = 0; m < M; ++m) { pt[m] = -9; for (int c = 0; c < NC; ++c) { pi[m * NC + c] = -1; pv[m * NC + c] = -INFINITY; } }
        int row = 0;
        for (int q = 0; q < NSEQ; ++q) {
            const int g = 1 + q % 8;
            const uint64_t seed = 0xb10c0000ull + (uint64_t)q, pos0 = 20 + 3 * (uint64_t)q;
            for (int i = 0; i <= g; ++i, ++row) {
                /* live candidates at random slots */
                const int nlive = wide ? 300 + (int)(rnd() % 200) : 6 + (int)(rnd() % 19);
                int ids[512];
                for (int c = 0; c < nlive; ++c) {
                    bool dup;
                    do {
                        ids[c] = (int)(rnd() % (wide ? 20000u : 40u));
                        dup = false;
                        for (int k = 0; k < c; ++k) dup = dup || ids[k] == ids[c];
                    } while (dup);
                    int slot;
                    do { slot = (int)(rnd() % NC); } while (pi[row * NC + slot] >= 0);
                    pi[row * NC + slot] = ids[c];
                    pv[row * NC + slot] = wide ? 3.0f - 0.02f * c - (rnd() % 100) * 0.01f : 3.0f - 0.35f * c - (rnd() % 100) * 0.01f;
                }
                pp[row].seed = seed;
                pp[row].pos = pos0 + (uint64_t)i;
                if (i == g) continue;
                int32_t qid[8];
                float qv[8];
                const int nq = 3 + (int)(rnd() % 6);
                for (int c = 0; c < 8; ++c) {
                    qid[c] = c < nq ? (c < nlive && rnd() % 4 ? ids[c] : (int)(rnd() % 40)) : -1;
                    qv[c] = c < nq ? 2.5f - 0.4f * c - (rnd() % 100) * 0.01f : -INFINITY;
                    for (int k = 0; k < c; ++k) if (qid[c] >= 0 && qid[k] == qid[c]) qid[c] = 100 + c;
                }
                RadMtpCoupleState st{};
                REQUIRE(rad_mtp_couple_build_q(qid, qv, VOC, &st));
                st.draw = make_draw(seed, pos0, 7 + (uint64_t)q, 0, 1u);
                st.status = 1; st.round = i + 1; st.pos = pos0 + (uint64_t)i; st.blk = g;
                st.x = st.ids[rnd() % (uint32_t)st.count];
                ps_[row] = st;
            }
        }
        const std::vector<RadParam> pr = { P("M", M), P("n_cand", NC), P("n_vocab", (int)VOC) };
        Buf htok = tok;
        {
            RadTensor ts[5] = { cid.on_host(), cval.on_host(), prm.on_host(), sts.on_host(), htok.on_host() };
            RadArgs a{}; a.t = ts; a.n_t = 5; a.p = pr.data(); a.n_p = (int)pr.size();
            REQUIRE_EQ(hver->launch(&a, s), RAD_OK);
        }
        {
            RadTensor ts[5] = { cid.on_dev(s), cval.on_dev(s), prm.on_dev(s), sts.on_dev(s), tok.on_dev(s) };
            RadArgs a{}; a.t = ts; a.n_t = 5; a.p = pr.data(); a.n_p = (int)pr.size();
            REQUIRE_EQ(dver->launch(&a, s), RAD_OK);
            tok.fetch(s);
            REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
        }
        row = 0;
        for (int q = 0; q < NSEQ; ++q) {
            const int g = 1 + q % 8;
            int written = 0;
            for (int i = 0; i <= g; ++i) {
                const int32_t gv = ((int32_t*)tok.host.data())[row + i], wv = ((int32_t*)htok.host.data())[row + i];
                CHECK_EQ(gv, wv);
                written += wv != -9;
            }
            writes += written > 0;
            whole += written == g;      /* tau == g: every draft written, the bonus row alone untouched */
            shortened += written > 0 && written < g;
            row += g + 1;
        }
    }
    CHECK(writes >= 100);
    CHECK(shortened >= 10);
    CHECK(whole >= 10);
    rad_stream_destroy(s);
}

/* Traversal verification (abi/rad_mtp_traversal.h): the device row against the host row, over
 * requests whose rows form a small tree (a root with two children, one of them with two, and a
 * chain below the other). The device prepares the tree with a workgroup (integer masses, whose sums
 * do not depend on the order) and runs the header's walk over it, so the tokens must be identical;
 * planes of 12, 1024 and 1100 slots. */
TEST(the_device_traversal_verify_agrees_with_the_host_row) {
    if (!have_card()) SKIP("no card");
    Plugin pl;
    if (!pl.open(RAD_SPECHUB_LIB))
        ARTIFACT_MISSING(std::string(RAD_SPECHUB_LIB) + " does not load");
    const RadKernelInfo* dtr = pl.row("spechub_mtp_trav_verify");
    const RadKernelInfo* htr = pl.row("ref_mtp_trav_verify");
    if (!dtr) SKIP("the plugin has no traversal device row");
    REQUIRE(htr);
    RadStream s{};
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);
    const int SW = (int)RAD_MTP_COUPLE_STATE_WORDS;
    const int64_t VOC = 50000;
  for (const int NC : { 12, 1024, 1100 }) {
    const bool wide = NC > 64;
    const int RQ = 12, NR = 7, M = RQ * NR;
    const int par[NR] = { -1, 0, 0, 1, 1, 2, 5 };
    const int nkid[NR] = { 2, 2, 1, 0, 0, 1, 0 };
    Buf cid(RAD_I32, { M, NC }, 4), cval(RAD_F32, { M, NC }, 4), prm(RAD_U32, { M, 24 }, 4);
    Buf sts(RAD_U32, { M, SW }, 4), tok(RAD_I32, { M }, 4), topo(RAD_I32, { M, 2 }, 4);
    int32_t* pi = (int32_t*)cid.host.data();
    float* pv = (float*)cval.host.data();
    RadSampleParams* pp = (RadSampleParams*)prm.host.data();
    RadMtpCoupleState* ps_ = (RadMtpCoupleState*)sts.host.data();
    int32_t* pt = (int32_t*)tok.host.data();
    int32_t* ptopo = (int32_t*)topo.host.data();
    for (int q = 0; q < RQ; ++q) {
        const uint64_t seed = 0x700ull + (uint64_t)q, p0 = 20 + (uint64_t)q;
        int depth[NR] = {};
        int32_t childtok[NR][4] = {};
        for (int n = 0; n < NR; ++n) {
            const int m = q * NR + n;
            depth[n] = n ? depth[par[n]] + 1 : 0;
            /* wide: the sampler's full plane at top_k 0 on even requests, 40 live slots and holes on odd */
            const int live = !wide || q % 2 == 0 ? NC : 40;
            for (int c = 0; c < NC; ++c) {
                if (wide && c >= live) { pi[m * NC + c] = -1; pv[m * NC + c] = -INFINITY; continue; }
                int id;
                bool dup;
                do {
                    id = (int)(rnd() % (wide ? VOC : 14));
                    dup = false;
                    for (int z = 0; z < c; ++z) dup = dup || pi[m * NC + z] == id;
                } while (dup);
                pi[m * NC + c] = id;
                pv[m * NC + c] = wide ? 2.0f - 0.01f * c - (rnd() % 100) * 0.01f
                                      : (c < 10 ? 2.0f - 0.25f * c : -INFINITY) - (rnd() % 100) * 0.01f;
            }
            int32_t qid[8];
            float qv[8];
            for (int c = 0; c < 8; ++c) {
                qid[c] = wide ? pi[m * NC + (int)(rnd() % (uint32_t)(live < 60 ? live : 60))] : (int)(rnd() % 14);
                qv[c] = 2.0f - 0.3f * c - (rnd() % 50) * 0.01f;
            }
            for (int c = 0; c < 8; ++c)
                for (int z = 0; z < c; ++z) if (qid[z] == qid[c]) qid[c] = 100 + c;
            RadMtpCoupleState s0{};
            rad_mtp_couple_build_q(qid, qv, VOC, &s0);
            s0.draw = make_draw(seed, 3, 1, 0, 1u);
            s0.status = 1; s0.round = depth[n] + 1; s0.pos = p0 + (uint64_t)depth[n];
            if (nkid[n] >= 2) rad_mtp_rrs_draw(&s0, (uint64_t)n, nkid[n]);
            else { s0.x = s0.ids[0]; s0.nkids = 0; }
            for (int c = 0; c < nkid[n]; ++c) childtok[n][c] = s0.ids[c];
            if (nkid[n] > 0) ps_[m] = s0; else ps_[m].status = 0;
            pp[m].seed = seed;
            pp[m].pos = p0 + (uint64_t)depth[n];
            pp[m].flags = q == 5 ? RAD_SP_GREEDY : 0;   /* one request is greedy: untouched */
            pt[m] = -9;
        }
        int seen[NR] = {};
        for (int n = 0; n < NR; ++n) {
            const int m = q * NR + n;
            ptopo[2 * m] = n ? q * NR + par[n] : -1;
            ptopo[2 * m + 1] = n ? childtok[par[n]][seen[par[n]]++] : 0;
        }
    }
    const std::vector<RadParam> pr = { P("M", M), P("n_cand", NC), P("n_vocab", (int)VOC) };
    Buf htok = tok;
    {
        RadTensor ts[6] = { cid.on_host(), cval.on_host(), prm.on_host(), sts.on_host(), htok.on_host(),
                            topo.on_host() };
        RadArgs a{}; a.t = ts; a.n_t = 6; a.p = pr.data(); a.n_p = (int)pr.size();
        REQUIRE_EQ(htr->launch(&a, s), RAD_OK);
    }
    {
        RadTensor ts[6] = { cid.on_dev(s), cval.on_dev(s), prm.on_dev(s), sts.on_dev(s), tok.on_dev(s),
                            topo.on_dev(s) };
        RadArgs a{}; a.t = ts; a.n_t = 6; a.p = pr.data(); a.n_p = (int)pr.size();
        REQUIRE_EQ(dtr->launch(&a, s), RAD_OK);
        tok.fetch(s);
        REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
    }
    int written = 0, greedy_kept = 0;
    for (int m = 0; m < M; ++m) {
        const int32_t g = ((int32_t*)tok.host.data())[m], w = ((int32_t*)htok.host.data())[m];
        CHECK_EQ(g, w);
        written += w != -9;
        if (m / NR == 5) greedy_kept += w == -9;
    }
    CHECK(written >= RQ);
    CHECK_EQ(greedy_kept, NR);
  }
    rad_stream_destroy(s);
}

/* The traversal verify over what a step can stage, device against host, bit for bit: requests of 1,
 * 4, 11 and 21 rows in one launch (a workgroup's four waves each take every fourth row; a lone bonus
 * row; a tree wider than a wave's share), Gumbel and RheoSampling nodes, chain rows whose x is not
 * the shortlist's first id, greedy and stale rows, targets that
 * lack draft tokens, and planes of 24, 1024 (the sampler's), 2048 (two passes a lane), 1100 and 4100
 * slots (no whole chunks; the fixed-point unit one bit coarser) with holes and -inf entries. */
TEST(the_device_traversal_verify_agrees_with_the_host_row_on_mixed_requests) {
    if (!have_card()) SKIP("no card");
    Plugin pl;
    if (!pl.open(RAD_SPECHUB_LIB))
        ARTIFACT_MISSING(std::string(RAD_SPECHUB_LIB) + " does not load");
    const RadKernelInfo* dtr = pl.row("spechub_mtp_trav_verify");
    const RadKernelInfo* htr = pl.row("ref_mtp_trav_verify");
    if (!dtr) SKIP("the plugin has no traversal device row");
    REQUIRE(htr);
    RadStream s{};
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);
    const int SW = (int)RAD_MTP_COUPLE_STATE_WORDS;
    const int64_t VOC = 60000;
    /* the request shapes: parents by row (root -1) */
    struct Shape { int nr; int par[24]; };
    const Shape shapes[] = {
        { 1,  { -1 } },
        { 4,  { -1, 0, 1, 2 } },
        { 11, { -1, 0, 0, 1, 1, 2, 3, 3, 5, 6, 6 } },
        { 21, { -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 } },
        { 7,  { -1, 0, 1, 2, 0, 1, 2 } },
    };
    long written = 0, paths3 = 0, finals = 0, rheo_nodes = 0;
    for (const int NC : { 24, 1024, 2048, 1100, 4100 }) {
        const int REPS = NC > 2048 ? 6 : 24;
        int M = 0;
        for (int r = 0; r < REPS; ++r) M += shapes[r % 5].nr;
        Buf cid(RAD_I32, { M, NC }, 4), cval(RAD_F32, { M, NC }, 4), prm(RAD_U32, { M, 24 }, 4);
        Buf sts(RAD_U32, { M, SW }, 4), tok(RAD_I32, { M }, 4), topo(RAD_I32, { M, 2 }, 4);
        int32_t* pi = (int32_t*)cid.host.data();
        float* pv = (float*)cval.host.data();
        RadSampleParams* pp = (RadSampleParams*)prm.host.data();
        RadMtpCoupleState* ps_ = (RadMtpCoupleState*)sts.host.data();
        int32_t* pt = (int32_t*)tok.host.data();
        int32_t* ptopo = (int32_t*)topo.host.data();
        int row0 = 0;
        for (int r = 0; r < REPS; ++r) {
            const Shape& sh = shapes[r % 5];
            const uint64_t seed = 0x51000ull + 977ull * (uint64_t)r + (uint64_t)NC, pos0 = 40 + (uint64_t)r;
            const int kind = r % 4;          /* 0 close laws, 1 flat Q with a token the target lacks, 2 Q off the target, 3 close */
            const int rheo = r % 6 == 4 ? 2 : r % 6 == 5 ? 3 : 0;   /* scheme of the nodes with several children */
            int depth[24] = {}, nkid[24] = {}, seen[24] = {};
            int32_t childtok[24][8] = {};
            for (int n = 1; n < sh.nr; ++n) ++nkid[sh.par[n]];
            for (int n = 0; n < sh.nr; ++n) {
                const int m = row0 + n;
                depth[n] = n ? depth[sh.par[n]] + 1 : 0;
                /* the plane: a peaked head, a long tail, holes and -inf entries at random slots */
                const int live = NC <= 24 ? 14 : (r % 3 == 0 ? 20 : NC - NC / 7);
                std::vector<int> ids((size_t)NC, -1);
                for (int c = 0; c < NC; ++c) { pi[(size_t)m * NC + c] = -1; pv[(size_t)m * NC + c] = -INFINITY; }
                for (int c = 0, placed = 0; c < NC && placed < live; ++c) {
                    if (NC > 24 && r % 3 != 0 && rnd() % 7 == 0) continue;   /* a hole */
                    bool dup;
                    int id;
                    do { id = (int)(rnd() % (uint32_t)(VOC - 2000)); dup = false;
                         for (int k = 0; k < c; ++k) dup = dup || ids[(size_t)k] == id; } while (dup);
                    ids[(size_t)c] = id;
                    pi[(size_t)m * NC + c] = id;
                    pv[(size_t)m * NC + c] = rnd() % 23 == 0 ? -INFINITY
                        : placed < 20 ? 9.0f - 0.8f * (float)placed - (rnd() % 100) * 0.01f
                                      : 1.0f - 0.003f * (float)placed - (rnd() % 100) * 0.01f;
                    ++placed;
                }
                pp[m].seed = seed; pp[m].pos = pos0 + (uint64_t)depth[n];
                pp[m].flags = r == 7 ? RAD_SP_GREEDY : 0;
                pt[m] = -9;
                ps_[m] = RadMtpCoupleState{};
                if (nkid[n] == 0) continue;
                int32_t qid[8];
                float qv[8];
                int have = 0;
                for (int c = 0; c < NC && have < 8; ++c)
                    if (ids[(size_t)c] >= 0) qid[have++] = ids[(size_t)c];
                for (int c = have; c < 8; ++c) qid[c] = (int)(VOC - 100 + c);
                for (int c = 0; c < 8; ++c) {
                    if (kind == 2) qid[c] = (int)(VOC - 1000 + 8 * n + c);
                    qv[c] = (kind == 1 ? 7.0f - 0.4f * c : 8.0f - 0.8f * c) - (rnd() % 100) * 0.01f;
                }
                if (kind == 1) qid[1 + n % 3] = (int)(VOC - 500 + n);
                RadMtpCoupleState st{};
                REQUIRE(rad_mtp_couple_build_q(qid, qv, VOC, &st));
                st.draw = make_draw(seed, pos0, 7 + (uint64_t)r, 0, 1u);
                st.status = 1; st.round = depth[n] + 1; st.pos = pos0 + (uint64_t)depth[n];
                if (nkid[n] >= 2 && rheo) { rad_mtp_rrs_draw_rheo(&st, (uint64_t)n, nkid[n], rheo - 1); ++rheo_nodes; }
                else if (nkid[n] >= 2) rad_mtp_rrs_draw(&st, (uint64_t)n, nkid[n]);
                else { st.x = st.ids[rad_mtp_couple_pick(st, rad_spechub_uniform(seed, pos0, 77 + (uint64_t)n))]; st.nkids = 0; }
                const int nk_real = st.nkids >= 1 ? st.nkids : 1;
                for (int c = 0; c < nkid[n]; ++c) childtok[n][c] = c < nk_real ? (st.nkids >= 1 ? st.ids[c] : st.x) : (int)(VOC - 50 + c);
                if (r == 9 && n == 1) st.pos += 3;   /* a stale state: a leaf of the walk */
                ps_[m] = st;
            }
            for (int n = 0; n < sh.nr; ++n) {
                const int m = row0 + n;
                ptopo[2 * m] = n ? row0 + sh.par[n] : -1;
                ptopo[2 * m + 1] = n ? childtok[sh.par[n]][seen[sh.par[n]]++] : 0;
            }
            row0 += sh.nr;
        }
        const std::vector<RadParam> pr = { P("M", M), P("n_cand", NC), P("n_vocab", (int)VOC) };
        Buf htok = tok, hst = sts;
        {
            RadTensor ts[6] = { cid.on_host(), cval.on_host(), prm.on_host(), hst.on_host(), htok.on_host(), topo.on_host() };
            RadArgs a{}; a.t = ts; a.n_t = 6; a.p = pr.data(); a.n_p = (int)pr.size();
            REQUIRE_EQ(htr->launch(&a, s), RAD_OK);
        }
        {
            RadTensor ts[6] = { cid.on_dev(s), cval.on_dev(s), prm.on_dev(s), sts.on_dev(s), tok.on_dev(s), topo.on_dev(s) };
            RadArgs a{}; a.t = ts; a.n_t = 6; a.p = pr.data(); a.n_p = (int)pr.size();
            REQUIRE_EQ(dtr->launch(&a, s), RAD_OK);
            tok.fetch(s);
            sts.fetch(s);
            REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
        }
        row0 = 0;
        for (int r = 0; r < REPS; ++r) {
            const Shape& sh = shapes[r % 5];
            int wr = 0;
            for (int n = 0; n < sh.nr; ++n) {
                const int m = row0 + n;
                const int32_t g = ((int32_t*)tok.host.data())[m], w = ((int32_t*)htok.host.data())[m];
                CHECK_EQ(g, w);
                wr += w != -9;
                RadMtpCoupleState dg, hw;
                std::memcpy(&dg, sts.host.data() + (size_t)m * SW * 4, sizeof dg);
                std::memcpy(&hw, hst.host.data() + (size_t)m * SW * 4, sizeof hw);
                CHECK(std::memcmp(&dg, &hw, sizeof dg) == 0);   /* the walk leaves the plane alone */
            }
            if (r == 7) CHECK_EQ(wr, 0);   /* the greedy request keeps every pick */
            written += wr;
            paths3 += wr >= 3;
            finals += wr >= 1;
            row0 += sh.nr;
        }
    }
    fprintf(stderr, "  mixed requests: %ld tokens written, %ld requests decided, %ld with a path of 2+ drafts, %ld rheo nodes\n",
            written, finals, paths3, rheo_nodes);
    CHECK(written >= 100);
    CHECK(paths3 >= 10);
    CHECK(rheo_nodes >= 10);
    rad_stream_destroy(s);
}

/* Microbenchmark of the block verify op (set RAD_BLOCK_BENCH=1; RAD_BENCH_LIB=<spechub.so> to time
 * another build of the plugin). Wall time per launch over back-to-back launches of one block (g
 * drafts plus the bonus row, one sequence as at max-num-seqs 1), minus the same launch over rows the
 * op ignores (the launch floor). Planes of 1024 slots with `live` live candidates: 20 (top_k 20)
 * or 1024 (top_k 0). Eight different blocks are cycled. */
TEST(block_verify_microbenchmark) {
    if (!getenv("RAD_BLOCK_BENCH")) return;
    if (!have_card()) return;
    Plugin pl;
    const char* lib = getenv("RAD_BENCH_LIB") ? getenv("RAD_BENCH_LIB") : RAD_SPECHUB_LIB;
    REQUIRE(pl.open(lib));
    const RadKernelInfo* dver = pl.row("spechub_mtp_couple_verify");
    REQUIRE(dver);
    RadStream s{};
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);
    const int SW = (int)RAD_MTP_COUPLE_STATE_WORDS;
    const int64_t VOC = 150000;
    const int NC = 1024, NV = 8;
    const int nrep = getenv("RAD_BENCH_REPS") ? atoi(getenv("RAD_BENCH_REPS")) : 400;
    const int divide = getenv("RAD_BENCH_DIV") ? atoi(getenv("RAD_BENCH_DIV")) : 1;   /* the plugin's in-kernel repeat count */
    for (int live : { 20, 1024 }) for (int g : { 3, 5, 7 }) {
        const int M = g + 1;
        struct Var { Buf cid, cval, prm, sts, tok; Var(int M, int NC, int SW) :
            cid(RAD_I32, { M, NC }, 4), cval(RAD_F32, { M, NC }, 4), prm(RAD_U32, { M, 24 }, 4),
            sts(RAD_U32, { M, SW }, 4), tok(RAD_I32, { M }, 4) {} };
        std::vector<Var*> vars;
        for (int v = 0; v < NV; ++v) {
            Var* x = new Var(M, NC, SW);
            vars.push_back(x);
            int32_t* pi = (int32_t*)x->cid.host.data();
            float* pv = (float*)x->cval.host.data();
            RadSampleParams* pp = (RadSampleParams*)x->prm.host.data();
            RadMtpCoupleState* ps_ = (RadMtpCoupleState*)x->sts.host.data();
            for (int m = 0; m < M; ++m) for (int c = 0; c < NC; ++c) { pi[m * NC + c] = -1; pv[m * NC + c] = -INFINITY; }
            const uint64_t seed = 0xbe00ull + (uint64_t)v, pos0 = 100;
            for (int i = 0; i < M; ++i) {
                int ids[1024];
                for (int c = 0; c < live; ++c) {
                    bool dup;
                    do { ids[c] = (int)(rnd() % VOC); dup = false;
                         for (int k = 0; k < c; ++k) dup = dup || ids[k] == ids[c]; } while (dup);
                    pi[i * NC + c] = ids[c];
                    pv[i * NC + c] = 12.0f - 0.9f * (float)c * (20.0f / (float)live) - (rnd() % 100) * 0.01f;
                }
                pp[i].seed = seed; pp[i].pos = pos0 + (uint64_t)i;
                if (i == g) continue;
                int32_t qid[8]; float qv[8];
                for (int c = 0; c < 8; ++c) { qid[c] = ids[c]; qv[c] = 11.0f - 0.9f * c - (rnd() % 100) * 0.01f; }
                RadMtpCoupleState st{};
                REQUIRE(rad_mtp_couple_build_q(qid, qv, VOC, &st));
                st.draw = make_draw(seed, pos0, 7 + (uint64_t)v, 0, 1u);
                st.status = 1; st.round = i + 1; st.pos = pos0 + (uint64_t)i; st.blk = g;
                st.x = st.ids[rnd() % 3u];
                ps_[i] = st;
            }
        }
        const std::vector<RadParam> pr = { P("M", M), P("n_cand", NC), P("n_vocab", (int)VOC) };
        std::vector<RadTensor> ts[NV];
        std::vector<RadTensor> tn[NV];
        for (int v = 0; v < NV; ++v) {
            Var* x = vars[(size_t)v];
            ts[v] = { x->cid.on_dev(s), x->cval.on_dev(s), x->prm.on_dev(s), x->sts.on_dev(s), x->tok.on_dev(s) };
        }
        /* the floor: the same launch with every state inert (status 0) */
        std::vector<Var*> inert;
        for (int v = 0; v < NV; ++v) {
            Var* x = new Var(M, NC, SW);
            memcpy(x->cid.host.data(), vars[(size_t)v]->cid.host.data(), x->cid.host.size());
            memcpy(x->cval.host.data(), vars[(size_t)v]->cval.host.data(), x->cval.host.size());
            memcpy(x->prm.host.data(), vars[(size_t)v]->prm.host.data(), x->prm.host.size());
            inert.push_back(x);
            tn[v] = { x->cid.on_dev(s), x->cval.on_dev(s), x->prm.on_dev(s), x->sts.on_dev(s), x->tok.on_dev(s) };
        }
        auto run = [&](std::vector<RadTensor>* t, int reps) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) {
                RadArgs a{}; a.t = t[r % NV].data(); a.n_t = 5; a.p = pr.data(); a.n_p = (int)pr.size();
                if (dver->launch(&a, s) != RAD_OK) return -1.0;
            }
            rad_stream_sync(s);
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        run(ts, 50); run(tn, 50);
        double best = 1e9, floor_ = 1e9;
        for (int rep = 0; rep < 5; ++rep) { best = std::min(best, run(ts, nrep)); floor_ = std::min(floor_, run(tn, nrep)); }
        fprintf(stderr, "  BENCH live=%4d g=%d  block %.1f us/call  floor %.1f us  net per block decision %.1f us\n", live, g, best, floor_, (best - floor_) / divide);
        for (auto* x : vars) delete x;
        for (auto* x : inert) delete x;
    }
    rad_stream_destroy(s);
}

/* Microbenchmark of the traversal verify op (RAD_TRAV_BENCH=1). One request a launch (max-num-seqs 1):
 * a chain of 3 drafts and trees of 4, 6 and 10 nodes, on the sampler's 1024-slot plane with 20 live
 * candidates (top_k 20) or all of them (top_k 0); three kinds of data: `acc` (Q close to P: the walk
 * accepts its first leaf most of the time), `mix` (a flatter Q with one token the target lacks) and `rej`
 * (Q's support is off the target's: every node is visited and rejected, and the final token is an
 * off-support draw at the root -- the longest walk a tree has). Eight requests of a kind are cycled.
 *
 * RAD_BENCH_LIB is the plugin timed; RAD_BENCH_LIB_REP the same sources built with
 * -DRAD_TRAV_REPEAT=RAD_BENCH_REP, whose kernel decides the request that many times in one launch:
 * (t_rep - t_one) / (REP - 1) is the kernel's time per decision with the launch, the queue and the sync
 * taken out (the AQL backend reports no kernel times). Tokens of the timed plugin are checked against
 * the host row first. */
namespace {
struct TravShape { const char* name; int nr; int par[16]; };
const TravShape kTravShapes[] = {
    { "chain3", 4,  { -1, 0, 1, 2 } },
    { "tree4",  5,  { -1, 0, 1, 2, 0 } },                         /* chain 3 + a sibling of its first node */
    { "tree6",  7,  { -1, 0, 1, 2, 0, 1, 2 } },                   /* chain 3 + a sibling at every depth */
    { "tree10", 11, { -1, 0, 0, 1, 1, 2, 3, 3, 5, 6, 6 } },       /* a width-2 grower's tree */
};
struct TravReq {
    Buf cid, cval, prm, sts, tok, topo;
    TravReq(int M, int NC, int SW)
        : cid(RAD_I32, { M, NC }, 4), cval(RAD_F32, { M, NC }, 4), prm(RAD_U32, { M, 24 }, 4),
          sts(RAD_U32, { M, SW }, 4), tok(RAD_I32, { M }, 4), topo(RAD_I32, { M, 2 }, 4) {}
};
/* kind 0 acc, 1 mix, 2 rej, 3 inert (no bound state: the op's early out) */
void fill_trav_req(TravReq* x, const TravShape& sh, int NC, int live, int kind, uint64_t seed, int64_t VOC,
                   int deep = -1) {
    const int NR = sh.nr;
    int32_t* pi = (int32_t*)x->cid.host.data();
    float* pv = (float*)x->cval.host.data();
    RadSampleParams* pp = (RadSampleParams*)x->prm.host.data();
    RadMtpCoupleState* ps_ = (RadMtpCoupleState*)x->sts.host.data();
    int32_t* pt = (int32_t*)x->tok.host.data();
    int32_t* ptopo = (int32_t*)x->topo.host.data();
    int depth[16] = {}, nkid[16] = {}, seen[16] = {};
    int32_t childtok[16][8] = {};
    for (int n = 1; n < NR; ++n) ++nkid[sh.par[n]];
    const uint64_t pos0 = 100;
    for (int n = 0; n < NR; ++n) {
        depth[n] = n ? depth[sh.par[n]] + 1 : 0;
        int ids[1024];
        for (int c = 0; c < NC; ++c) { pi[n * NC + c] = -1; pv[n * NC + c] = -INFINITY; }
        for (int c = 0; c < live; ++c) {
            bool dup;
            do { ids[c] = (int)(rnd() % (uint32_t)(VOC - 1000)); dup = false;
                 for (int k = 0; k < c; ++k) dup = dup || ids[k] == ids[c]; } while (dup);
            pi[n * NC + c] = ids[c];
            /* a peaked head of 20 and, at top_k 0, a long flat tail carrying about 6 % of the mass */
            pv[n * NC + c] = c < 20 ? 12.0f - 0.9f * (float)c - (rnd() % 100) * 0.01f
                                    : 3.0f - 0.004f * (float)(c - 20) - (rnd() % 100) * 0.01f;
            /* `deep`: the row's mass in one candidate of pass `deep` (the final draw's scan, by depth);
             * RAD_TRAV_BENCH_DEEP=2 gives every other candidate a small mass of its own (2e-9 of it), so
             * that no pass of the scan is one of zero masses */
            if (deep >= 0)
                pv[n * NC + c] = c == 32 * deep + 5 ? 12.0f : (atoi(getenv("RAD_TRAV_BENCH_DEEP")) == 2 ? -8.0f : -30.0f);
        }
        pp[n].seed = seed; pp[n].pos = pos0 + (uint64_t)depth[n]; pp[n].flags = 0;
        pt[n] = -9;
        ps_[n] = RadMtpCoupleState{};
        if (nkid[n] == 0 || kind == 3) continue;
        int32_t qid[8]; float qv[8];
        for (int c = 0; c < 8; ++c) {
            qid[c] = kind == 2 ? (int)(VOC - 1000 + 8 * n + c) : ids[c];
            qv[c] = kind == 1 ? 11.0f - 0.5f * c - (rnd() % 100) * 0.01f : 11.0f - 0.9f * c - (rnd() % 100) * 0.01f;
        }
        if (kind == 1) qid[1 + n % 3] = (int)(VOC - 500 + n);     /* a draft token the target lacks */
        RadMtpCoupleState st{};
        REQUIRE(rad_mtp_couple_build_q(qid, qv, VOC, &st));
        st.draw = make_draw(seed, pos0, 7, 0, 1u);
        st.status = 1; st.round = depth[n] + 1; st.pos = pos0 + (uint64_t)depth[n];
        if (nkid[n] >= 2) rad_mtp_rrs_draw(&st, (uint64_t)n, nkid[n]);
        else { st.x = st.ids[rad_mtp_couple_pick(st, rad_spechub_uniform(seed, pos0, 77 + (uint64_t)n))]; st.nkids = 0; }
        for (int c = 0; c < nkid[n]; ++c) childtok[n][c] = nkid[n] >= 2 ? st.ids[c] : st.x;
        ps_[n] = st;
    }
    for (int n = 0; n < NR; ++n) {
        ptopo[2 * n] = n ? sh.par[n] : -1;
        ptopo[2 * n + 1] = n ? childtok[sh.par[n]][seen[sh.par[n]]++] : 0;
    }
}
}  /* namespace */

TEST(trav_verify_microbenchmark) {
    if (!getenv("RAD_TRAV_BENCH")) return;
    if (!have_card()) return;
    Plugin pl, plr;
    const char* lib = getenv("RAD_BENCH_LIB") ? getenv("RAD_BENCH_LIB") : RAD_SPECHUB_LIB;
    REQUIRE(pl.open(lib));
    const RadKernelInfo* dtr = pl.row("spechub_mtp_trav_verify");
    const RadKernelInfo* htr = pl.row("ref_mtp_trav_verify");
    REQUIRE(dtr && htr);
    const RadKernelInfo* drep = nullptr;
    const int rep = getenv("RAD_BENCH_REP") ? atoi(getenv("RAD_BENCH_REP")) : 0;
    if (getenv("RAD_BENCH_LIB_REP") && rep > 1) {
        REQUIRE(plr.open(getenv("RAD_BENCH_LIB_REP")));
        drep = plr.row("spechub_mtp_trav_verify");
        REQUIRE(drep);
    }
    RadStream s{};
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);
    const int SW = (int)RAD_MTP_COUPLE_STATE_WORDS;
    const int64_t VOC = 150000;
    const int NC = 1024, NV = 8;
    const int nrep = getenv("RAD_BENCH_REPS") ? atoi(getenv("RAD_BENCH_REPS")) : 200;
    const char* kinds[] = { "acc", "mix", "rej", "inert" };
    /* RAD_TRAV_BENCH_DEEP=1: instead, the chain's final draw off the support with the drawn candidate in
     * pass 0, 7, 8, .. 31 of a 1024-slot plane: what the scan for it costs by depth. */
    const bool deep_series = getenv("RAD_TRAV_BENCH_DEEP") != nullptr;
    const int deeps[] = { 0, 3, 7, 8, 15, 16, 23, 24, 31 };
    for (int live : { 20, 1024 }) for (const TravShape& sh : kTravShapes) for (int kind = 0; kind < 4; ++kind)
    for (int di = 0; di < (deep_series ? 9 : 1); ++di) {
        const int deep = deep_series ? deeps[di] : -1;
        if (deep_series && (live != 1024 || kind != 2 || &sh != &kTravShapes[0])) continue;
        const int M = sh.nr;
        std::vector<TravReq*> vars;
        std::vector<RadTensor> ts[NV];
        const std::vector<RadParam> pr = { P("M", M), P("n_cand", NC), P("n_vocab", (int)VOC) };
        int path_rows = 0, mism = 0;
        for (int v = 0; v < NV; ++v) {
            TravReq* x = new TravReq(M, NC, SW);
            vars.push_back(x);
            fill_trav_req(x, sh, NC, live, kind, 0x7a00ull + 131ull * (uint64_t)v + (uint64_t)live, VOC, deep);
            /* the host row, on copies */
            Buf htok = x->tok, hst = x->sts;
            RadTensor th[6] = { x->cid.on_host(), x->cval.on_host(), x->prm.on_host(), hst.on_host(), htok.on_host(), x->topo.on_host() };
            RadArgs ah{}; ah.t = th; ah.n_t = 6; ah.p = pr.data(); ah.n_p = (int)pr.size();
            REQUIRE_EQ(htr->launch(&ah, s), RAD_OK);
            ts[v] = { x->cid.on_dev(s), x->cval.on_dev(s), x->prm.on_dev(s), x->sts.on_dev(s), x->tok.on_dev(s), x->topo.on_dev(s) };
            RadArgs a{}; a.t = ts[v].data(); a.n_t = 6; a.p = pr.data(); a.n_p = (int)pr.size();
            REQUIRE_EQ(dtr->launch(&a, s), RAD_OK);
            x->tok.fetch(s);
            REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
            for (int m = 0; m < M; ++m) {
                const int32_t g = ((int32_t*)x->tok.host.data())[m], w = ((int32_t*)htok.host.data())[m];
                mism += g != w;
                path_rows += w != -9;
            }
        }
        CHECK_EQ(mism, 0);
        auto run = [&](const RadKernelInfo* k, int reps) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) {
                RadArgs a{}; a.t = ts[r % NV].data(); a.n_t = 6; a.p = pr.data(); a.n_p = (int)pr.size();
                if (k->launch(&a, s) != RAD_OK) return -1.0;
            }
            rad_stream_sync(s);
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        run(dtr, 40);
        if (drep) run(drep, 10);
        double one = 1e9, many = 1e9;
        for (int it = 0; it < 5; ++it) {
            one = std::min(one, run(dtr, nrep));
            if (drep) many = std::min(many, run(drep, std::max(8, nrep / 4)));
        }
        if (deep_series) fprintf(stderr, "  TRAVBENCH final draw in pass %2d:", deep);
        if (drep)
            fprintf(stderr, "  TRAVBENCH live=%4d %-6s %-5s rows=%2d  launch %.1f us  x%d %.1f us  kernel per decision %.2f us  (path rows/req %.2f, mism %d)\n",
                    live, sh.name, kinds[kind], M, one, rep, many, (many - one) / (rep - 1), (double)path_rows / NV, mism);
        else
            fprintf(stderr, "  TRAVBENCH live=%4d %-6s %-5s rows=%2d  launch %.1f us (path rows/req %.2f, mism %d)\n",
                    live, sh.name, kinds[kind], M, one, (double)path_rows / NV, mism);
        for (auto* x : vars) delete x;
    }
    rad_stream_destroy(s);
}

RAD_TEST_MAIN()
