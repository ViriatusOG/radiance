/* quant_ggml_gpu_test.cpp -- the GGML IQ2_XXS/IQ3_XXS card path writes the host's bytes.
 *
 * The host quantiser is the oracle: the same weight, imatrix and converter band are run once on
 * the host and once on a card, and every plane byte is compared. The four-card case runs four
 * weights in four threads, which is how rad-convert's large-weight workers occupy the cards.
 */
#include "quant_fixture.h"

#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace rad;
using namespace qfix;

namespace {

RadQuantDevicesFn devices_hook() {
    const rad::Plugin* p = reg().plugin("libquant");
    return p && p->handle ? (RadQuantDevicesFn)dlsym(p->handle, "rad_quant_devices") : nullptr;
}

/* A positive column importance with a few deliberately uneven columns. */
std::vector<float> imatrix(int64_t cols, uint64_t seed) {
    std::vector<float> v((size_t)cols);
    uint64_t x = seed;
    for (int64_t i = 0; i < cols; ++i) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        v[(size_t)i] = 0.25f + (float)((x >> 40) % 97) / 32.0f;
    }
    return v;
}

bool same(const Quantised& a, const Quantised& b, std::string* where) {
    if (a.enc.n_planes != b.enc.n_planes) {
        *where = "plane count";
        return false;
    }
    for (int i = 0; i < a.enc.n_planes; ++i) {
        const std::vector<uint8_t>& x = a.planes[(size_t)i];
        const std::vector<uint8_t>& y = b.planes[(size_t)i];
        if (x.size() != y.size()) {
            *where = std::string(a.enc.plane[i].role) + " size";
            return false;
        }
        for (size_t j = 0; j < x.size(); ++j)
            if (x[j] != y[j]) {
                *where = std::string(a.enc.plane[i].role) + " byte " + std::to_string(j);
                return false;
            }
    }
    return true;
}

struct Case {
    const char* type;
    int64_t rows;
    int64_t cols;
    bool has_imp;
};

const Case kCases[] = {
    { "iq2_xxs", 4, 2304, true },
    { "iq2_xxs", 8, 256, true },
    { "iq2_xxs", 2, 7168, true },
    { "iq3_xxs", 4, 2304, true },
    { "iq3_xxs", 4, 2304, false },
    { "iq3_xxs", 8, 256, false },
};

}  /* namespace */

TEST(a_missing_card_is_refused) {
    NEED_LIBQUANT();
    const RadQuantDevicesFn fn = devices_hook();
    REQUIRE(fn != nullptr);
    const int none[] = { 1023 };
    CHECK(fn(none, 1) == RAD_E_DEVICE || fn(none, 1) == RAD_E_UNSUPPORTED);
    CHECK_EQ(fn(nullptr, 0), (int)RAD_OK);
}

TEST(the_card_writes_the_host_bytes) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("ggml");
    REQUIRE(q != nullptr);
    const RadQuantDevicesFn fn = devices_hook();
    REQUIRE(fn != nullptr);
    const int zero[] = { 0 };
    const int st = fn(zero, 1);
    if (st == RAD_E_UNSUPPORTED) SKIP("this libquant has no device path");
    if (st != RAD_OK) SKIP("no card visible");
    REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);

    for (const Case& c : kCases) {
        std::vector<float> w = draw(c.rows * c.cols, 700 + (uint64_t)c.rows * 31 + c.cols);
        const std::vector<float> imp = imatrix(c.cols, 900 + (uint64_t)c.cols);
        const float* ip = c.has_imp ? imp.data() : nullptr;
        const std::string name = "blk.0.expert." + std::to_string(c.rows) + ".weight";

        Opts h;
        h.s("type", c.type).i("gpu", 0);
        const Quantised host = quantise(q, h, name.c_str(), c.rows, c.cols, w, ip);
        REQUIRE_EQ(host.status, RAD_OK);

        REQUIRE_EQ(fn(zero, 1), (int)RAD_OK);
        Opts g;
        /* The card path is gated on whole-weight f32 bytes (gpu_min_bytes, 32 MiB by default); these
         * synthetic weights are tiny, so every card rule here lowers the gate. */
        g.s("type", c.type).i("gpu", 1).i("gpu_min_bytes", 1);
        const Quantised card = quantise(q, g, name.c_str(), c.rows, c.cols, w, ip);
        REQUIRE_EQ(card.status, RAD_OK);
        std::string where;
        const bool ok = same(host, card, &where);
        std::fprintf(stderr, "    %-8s rows %lld cols %lld imp %d: %s%s\n", c.type,
                     (long long)c.rows, (long long)c.cols, (int)c.has_imp,
                     ok ? "same bytes" : "DIFFERS at ", ok ? "" : where.c_str());
        CHECK(ok);
        REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);
    }
}

TEST(four_concurrent_weights_use_four_cards) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("ggml");
    REQUIRE(q != nullptr);
    const RadQuantDevicesFn fn = devices_hook();
    REQUIRE(fn != nullptr);
    const int ids[] = { 0, 1, 2, 3 };
    int n = 0;
    if (fn(ids, 4) != RAD_OK) {
        if (fn(ids, 2) == RAD_OK) n = 2;
        else if (fn(ids, 1) == RAD_OK) n = 1;
        else SKIP("no card visible");
    } else {
        n = 4;
    }
    REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);

    const int64_t rows = 4, cols = 2304;
    std::vector<std::vector<float>> w((size_t)n);
    std::vector<Quantised> host((size_t)n);
    for (int i = 0; i < n; ++i) {
        w[(size_t)i] = draw(rows * cols, 1000 + (uint64_t)i);
        Opts o;
        o.s("type", "iq2_xxs").i("gpu", 0);
        host[(size_t)i] = quantise(q, o, "blk.0.expert.weight", rows, cols, w[(size_t)i],
                                   imatrix(cols, 1100 + (uint64_t)i).data());
        REQUIRE_EQ(host[(size_t)i].status, RAD_OK);
    }

    REQUIRE_EQ(fn(ids, n), (int)RAD_OK);
    std::vector<Quantised> card((size_t)n);
    std::vector<std::thread> th;
    for (int i = 0; i < n; ++i)
        th.emplace_back([&, i] {
            Opts o;
            o.s("type", "iq2_xxs").i("gpu", 1).i("gpu_min_bytes", 1);
            card[(size_t)i] = quantise(q, o, "blk.0.expert.weight", rows, cols, w[(size_t)i],
                                       imatrix(cols, 1100 + (uint64_t)i).data());
        });
    for (std::thread& t : th) t.join();
    for (int i = 0; i < n; ++i) {
        REQUIRE_EQ(card[(size_t)i].status, RAD_OK);
        std::string where;
        const bool ok = same(host[(size_t)i], card[(size_t)i], &where);
        std::fprintf(stderr, "    concurrent %d/%d: %s%s\n", i + 1, n, ok ? "same bytes" : "DIFFERS at ",
                     ok ? "" : where.c_str());
        CHECK(ok);
    }
    REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);
}

TEST(real_iq2_blocks_match_host) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("ggml");
    REQUIRE(q != nullptr);
    const RadQuantDevicesFn fn = devices_hook();
    REQUIRE(fn != nullptr);
    const int zero[] = { 0 };
    if (fn(zero, 1) != RAD_OK) SKIP("no card visible");
    REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);

    std::ifstream f(RAD_TEST_DATA_DIR "/real_iq2_cases.bin", std::ios::binary);
    REQUIRE(f.good());
    std::vector<float> all((7 * (256 + 256)));
    f.read(reinterpret_cast<char*>(all.data()), (std::streamsize)all.size() * sizeof(float));
    REQUIRE(f.good());
    f.close();
    for (int i = 0; i < 7; ++i) {
        const float* wp = all.data() + (size_t)i * 512;
        const float* ip = wp + 256;
        std::vector<float> w(wp, wp + 256), imp(ip, ip + 256);
        Opts h;
        h.s("type", "iq2_xxs").i("gpu", 0);
        const Quantised host = quantise(q, h, "blk.0.expert.weight", 1, 256, w, imp.data());
        REQUIRE_EQ(host.status, RAD_OK);
        REQUIRE_EQ(fn(zero, 1), (int)RAD_OK);
        Opts g;
        g.s("type", "iq2_xxs").i("gpu", 1).i("gpu_min_bytes", 1);
        const Quantised card = quantise(q, g, "blk.0.expert.weight", 1, 256, w, imp.data());
        REQUIRE_EQ(card.status, RAD_OK);
        std::string where;
        const bool ok = same(host, card, &where);
        std::fprintf(stderr, "    real iq2 block %d: %s%s\n", i, ok ? "same bytes" : "DIFFERS at ",
                     ok ? "" : where.c_str());
        CHECK(ok);
        REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);
    }
}

TEST(real_iq3_blocks_match_host) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("ggml");
    REQUIRE(q != nullptr);
    const RadQuantDevicesFn fn = devices_hook();
    REQUIRE(fn != nullptr);
    const int zero[] = { 0 };
    if (fn(zero, 1) != RAD_OK) SKIP("no card visible");
    REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);
    std::ifstream f(RAD_TEST_DATA_DIR "/real_iq3_cases.bin", std::ios::binary);
    REQUIRE(f.good());
    std::vector<float> all((1 * (256 + 256)));
    f.read(reinterpret_cast<char*>(all.data()), (std::streamsize)all.size() * sizeof(float));
    REQUIRE(f.good());
    std::vector<float> w(all.data(), all.data() + 256), imp(all.data() + 256, all.data() + 512);
    Opts h; h.s("type", "iq3_xxs").i("gpu", 0);
    const Quantised host = quantise(q, h, "blk.6.expert.weight", 1, 256, w, imp.data());
    REQUIRE_EQ(host.status, RAD_OK);
    REQUIRE_EQ(fn(zero, 1), (int)RAD_OK);
    Opts g; g.s("type", "iq3_xxs").i("gpu", 1).i("gpu_min_bytes", 1);
    const Quantised card = quantise(q, g, "blk.6.expert.weight", 1, 256, w, imp.data());
    REQUIRE_EQ(card.status, RAD_OK);
    std::string where;
    const bool ok = same(host, card, &where);
    std::fprintf(stderr, "    real iq3 block: %s%s\n", ok ? "same bytes" : "DIFFERS at ", ok ? "" : where.c_str());
    CHECK(ok);
    REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);
}

TEST(bench_card_vs_host) {
    NEED_LIBQUANT();
    const RadQuantizerInfo* q = quantizer("ggml");
    REQUIRE(q != nullptr);
    const RadQuantDevicesFn fn = devices_hook();
    REQUIRE(fn != nullptr);
    const int zero[] = { 0 };
    if (fn(zero, 1) == RAD_E_UNSUPPORTED) SKIP("this libquant has no device path");
    if (fn(zero, 1) != RAD_OK) SKIP("no card visible");
    REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);

    const Case cases[] = {
        { "iq2_xxs", 128, 2304, true },
        { "iq3_xxs", 128, 2304, true },
    };
    for (const Case& c : cases) {
        std::vector<float> w = draw(c.rows * c.cols, 1700 + (uint64_t)c.rows * 31 + c.cols);
        const std::vector<float> imp = imatrix(c.cols, 1900 + (uint64_t)c.cols);
        Opts h;
        h.s("type", c.type).i("gpu", 0);
        Opts g;
        g.s("type", c.type).i("gpu", 1).i("gpu_min_bytes", 1);
        REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);
        const auto t0 = std::chrono::steady_clock::now();
        const Quantised host = quantise(q, h, "blk.0.expert.weight", c.rows, c.cols, w, imp.data());
        const auto t1 = std::chrono::steady_clock::now();
        REQUIRE_EQ(host.status, RAD_OK);
        REQUIRE_EQ(fn(zero, 1), (int)RAD_OK);
        const Quantised card = quantise(q, g, "blk.0.expert.weight", c.rows, c.cols, w, imp.data());
        const auto t2 = std::chrono::steady_clock::now();
        REQUIRE_EQ(card.status, RAD_OK);
        const double th = std::chrono::duration<double>(t1 - t0).count();
        const double tc = std::chrono::duration<double>(t2 - t1).count();
        std::fprintf(stderr, "    %-8s rows %lld cols %lld: host %.3f s, one card %.3f s, %.2fx\n",
                     c.type, (long long)c.rows, (long long)c.cols, th, tc, th / tc);
        CHECK(th > 0 && tc > 0);
        REQUIRE_EQ(fn(nullptr, 0), (int)RAD_OK);
    }
}

RAD_TEST_MAIN()
