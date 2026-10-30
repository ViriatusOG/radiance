/* quant_tcq_gpu_test.cpp -- trellis GPTQ on the card writes the host's bytes.
 *
 * libquant's card path (libquant/lq_gpu.h) promises the same planes as the host loop for every
 * trellis form, so a container does not depend on the machine that converted it. Each case
 * quantises one weight on the host and on card 0 (and split over cards 0 and 1 where two are
 * visible) and compares every plane byte for byte: the codes, the scales a dyn=1 block chose from
 * the columns the card left, and the table. The cases cover the shapes the kernel forms differ by
 * (2^L states, k-bit codes, lanes sharing a group's predecessors, the int8 row-block grid), a
 * short last run, a short last block, no Hessian, values small enough that the squared errors
 * are subnormal, and the computed code (tcq_code); then a weight split over two cards, and three
 * such quantised side by side.
 *
 * The unit entry runs with every card hidden and checks that a missing card is refused; the
 * _card entry runs the comparisons.
 */
#include "quant_fixture.h"

#include <dlfcn.h>

#include <thread>

using namespace rad;
using namespace qfix;

namespace {

RadQuantDevicesFn devices_hook() {
    const rad::Plugin* p = reg().plugin("libquant");
    return p && p->handle ? (RadQuantDevicesFn)dlsym(p->handle, "rad_quant_devices") : nullptr;
}

/* The input's Gram for K columns: per-channel energy over two decades, neighbours mixed. */
std::vector<float> gram(int64_t K, int64_t T, uint64_t seed) {
    const std::vector<float> z = draw(T * K, seed);
    std::vector<double> x((size_t)(T * K));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = 0; c < K; ++c) {
            const float* zr = &z[(size_t)(t * K)];
            const double amp = std::pow(10.0, (double)((c * 37) % K) / (double)K * 2.0);
            x[(size_t)(t * K + c)] = (zr[c] + 0.5 * zr[(c + 1) % K]) * amp;
        }
    std::vector<double> h((size_t)(K * K), 0.0);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < K; ++i) {
            const double xi = x[(size_t)(t * K + i)];
            for (int64_t j = 0; j < K; ++j) h[(size_t)(i * K + j)] += xi * x[(size_t)(t * K + j)];
        }
    return std::vector<float>(h.begin(), h.end());
}

/* A calibration directory with one Gram per (name, K) the cases use. */
const std::string& calib_dir() {
    static const struct Dir {
        TempDir t{ "rad_tcq_gpu" };
        Dir() {
            if (t.path.empty()) return;
            for (int64_t K : { 128, 256, 320, 384, 512 }) {
                const std::string stem = "blk.0.k" + std::to_string(K);
                write_gram(t.path + "/gram.r0." + stem + ".bin", K, 2048, gram(K, 2048, 300 + K));
            }
        }
    } dir;
    return dir.t.path;
}

struct Case {
    const char* what;
    int L, k, rb;
    int64_t N, K;
    bool calib, v2;      /* v2: E4M3 under one f32, fwht128, dyn=1 (the IQR recipes' form) */
    float scale = 1.0f;  /* times the weight */
    const char* code = nullptr;   /* tcq_code= (a computed table) */
};

Opts opts(const Case& c) {
    Opts o;
    o.s("codes", ("u" + std::to_string(c.k)).c_str()).i("tcq", c.L).i("tcq_rows", 256)
        .i("group", 128);
    if (c.rb) o.i("tcq_rb", c.rb);
    if (c.v2) {
        o.s("scale", "fp8_e4m3").s("scale2", "f32").s("block2", "*x*").s("transform", "fwht128")
            .i("dyn", 1).f("damp", 0.05);
    } else {
        o.s("scale", c.scale < 1e-6f ? "f32" : "bf16");
    }
    if (c.calib) o.s("calib", calib_dir().c_str());
    if (c.code) o.s("tcq_code", c.code);
    return o;
}

bool same(const Quantised& a, const Quantised& b, std::string* where) {
    if (a.enc.n_planes != b.enc.n_planes) { *where = "plane count"; return false; }
    for (int i = 0; i < a.enc.n_planes; ++i) {
        const std::vector<uint8_t>& x = a.planes[(size_t)i];
        const std::vector<uint8_t>& y = b.planes[(size_t)i];
        if (x.size() != y.size()) { *where = std::string(a.enc.plane[i].role) + " size"; return false; }
        for (size_t j = 0; j < x.size(); ++j)
            if (x[j] != y[j]) {
                *where = std::string(a.enc.plane[i].role) + " byte " + std::to_string(j);
                return false;
            }
    }
    return true;
}

}  /* namespace */

TEST(a_missing_card_is_refused) {
    NEED_LIBQUANT();
    const RadQuantDevicesFn fn = devices_hook();
    REQUIRE(fn != nullptr);
    const int none[] = { 1023 };
    const int st = fn(none, 1);
    CHECK(st == RAD_E_DEVICE || st == RAD_E_UNSUPPORTED);
    CHECK_EQ(fn(nullptr, 0), (int)RAD_OK);
}

TEST(the_card_writes_the_hosts_bytes) {
    NEED_LIBQUANT();
    const RadQuantDevicesFn fn = devices_hook();
    REQUIRE(fn != nullptr);
    const int zero[] = { 0 }, both[] = { 0, 1 };
    const int st = fn(zero, 1);
    if (st == RAD_E_UNSUPPORTED) SKIP("this libquant has no device path");
    if (st != RAD_OK) SKIP("no card visible");
    const bool two = fn(both, 2) == RAD_OK;
    REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);

    const Case cases[] = {
        { "L10 k2, no Hessian, a short last run",     10, 2, 0,  300, 256, false, false },
        { "L12 k4, the V2 form, a short last run",    12, 4, 0,  300, 384, true,  true  },
        { "L14 k4 rb32, the V3 form",                 14, 4, 32, 512, 512, true,  true  },
        { "L14 k4, no row blocks",                    14, 4, 0,  256, 256, true,  true  },
        { "L12 k4 rb8",                               12, 4, 8,  256, 256, true,  true  },
        { "L14 k2 rb8",                               14, 2, 8,  256, 256, true,  true  },
        { "L14 k3 rb32",                              14, 3, 32, 256, 256, true,  true  },
        { "L14 k5 rb32",                              14, 5, 32, 256, 256, true,  true  },
        { "L14 k6 rb32",                              14, 6, 32, 256, 256, true,  true  },
        { "L14 k7 rb32",                              14, 7, 32, 256, 128, true,  true  },
        { "L12 k2",                                   12, 2, 0,  256, 256, true,  true  },
        { "L12 k3 rb8",                               12, 3, 8,  256, 256, true,  true  },
        { "L12 k5",                                   12, 5, 0,  256, 256, true,  true  },
        { "L12 k6 rb8",                               12, 6, 8,  256, 256, true,  true  },
        { "L12 k7",                                   12, 7, 0,  256, 128, true,  true  },
        { "L12 k3, a short last block, bf16 scales",  12, 3, 0,  300, 320, true,  false },
        { "L10 k4, subnormal squared errors",         10, 4, 0,  256, 256, true,  false, 1e-21f },
        { "L16 k4 rb32",                              16, 4, 32, 256, 256, true,  true  },
        { "L16 k3 rb32, one cost buffer",             16, 3, 32, 256, 256, true,  true  },
        { "L16 k6 rb32",                              16, 6, 32, 256, 128, true,  true  },
        { "L16 k4 rb32, the computed code",           16, 4, 32, 256, 256, true,  true,  1.0f, "mad" },
        { "L16 k3 rb32, the computed code",           16, 3, 32, 256, 256, true,  true,  1.0f, "mad" },
        { "L15 k2, one cost buffer, computed",        15, 2, 0,  256, 256, true,  true,  1.0f, "mad" },
    };
    for (const Case& c : cases) {
        std::vector<float> w = draw(c.N * c.K, 500 + (uint64_t)c.L * 16 + (uint64_t)c.k, 6.0f);
        for (float& v : w) v *= c.scale;
        const std::string name = "blk.0.k" + std::to_string(c.K) + ".weight";
        Opts o = opts(c);
        REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);
        const Quantised host = quantise(quantizer("gptq"), o, name.c_str(), c.N, c.K, w);
        REQUIRE_EQ(fn(zero, 1), (int)RAD_OK);
        const Quantised card = quantise(quantizer("gptq"), o, name.c_str(), c.N, c.K, w);
        REQUIRE_EQ(host.status, RAD_OK);
        REQUIRE_EQ(card.status, RAD_OK);
        std::string where;
        const bool ok = same(host, card, &where);
        std::fprintf(stderr, "    %-44s %s%s\n", c.what, ok ? "same bytes" : "DIFFERS at ",
                     ok ? "" : where.c_str());
        CHECK(ok);
    }

    /* A weight of 32 runs or more is split over the cards; each card's rows are its own runs. */
    if (two) {
        const Case c = { "L12 k4 V2 form, 32 runs over two cards", 12, 4, 0, 32 * 256 + 100, 256,
                         true, true };
        const std::vector<float> w = draw(c.N * c.K, 777, 6.0f);
        Opts o = opts(c);
        REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);
        const Quantised host = quantise(quantizer("gptq"), o, "blk.0.k256.weight", c.N, c.K, w);
        REQUIRE_EQ(fn(both, 2), (int)RAD_OK);
        const Quantised card = quantise(quantizer("gptq"), o, "blk.0.k256.weight", c.N, c.K, w);
        REQUIRE_EQ(card.status, RAD_OK);
        std::string where;
        const bool ok = same(host, card, &where);
        std::fprintf(stderr, "    %-44s %s%s\n", c.what, ok ? "same bytes" : "DIFFERS at ",
                     ok ? "" : where.c_str());
        CHECK(ok);

        /* Weights quantised side by side, as rad-convert --gpus runs its large ones, each split
         * over both cards: copies to and from the cards from several threads at once. */
        constexpr int kSide = 3, kRounds = 3;
        std::vector<std::vector<float>> ws;
        std::vector<Quantised> hosts;
        REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);
        for (int x = 0; x < kSide; ++x) {
            ws.push_back(draw(c.N * c.K, 900 + (uint64_t)x, 6.0f));
            hosts.push_back(quantise(quantizer("gptq"), o, "blk.0.k256.weight", c.N, c.K, ws.back()));
            REQUIRE_EQ(hosts.back().status, RAD_OK);
        }
        REQUIRE_EQ(fn(both, 2), (int)RAD_OK);
        int bad = 0;
        for (int round = 0; round < kRounds; ++round) {
            std::vector<Quantised> cards(kSide);
            std::vector<std::thread> th;
            for (int x = 0; x < kSide; ++x)
                th.emplace_back([&, x] {
                    Opts ox = o;
                    cards[(size_t)x] = quantise(quantizer("gptq"), ox, "blk.0.k256.weight", c.N,
                                                c.K, ws[(size_t)x]);
                });
            for (std::thread& t : th) t.join();
            for (int x = 0; x < kSide; ++x)
                bad += cards[(size_t)x].status != RAD_OK || !same(hosts[(size_t)x], cards[(size_t)x], &where);
        }
        std::fprintf(stderr, "    %-44s %d of %d differ\n", "three weights side by side, both cards",
                     bad, kSide * kRounds);
        CHECK_EQ(bad, 0);
    }
    REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);
}

RAD_TEST_MAIN()
