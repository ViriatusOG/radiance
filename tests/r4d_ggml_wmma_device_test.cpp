// r4d_ggml_wmma_device_test.cpp -- gemm_nt_ggml, logits_gemm_ggml and moe_gemm_ggml at decode
// heights over IQ2_XXS and Q8_0 blocks on the actual card (r4d_ggml_wmma.hip, reached through the
// rows' own entry).
//
// Builds random llama.cpp blocks (the stored form: blocks row after row), runs the rows at M = 1, 8,
// 17 and 48 and holds the result to a double dequant reference by llama.cpp's formulas at the output
// rounding. Then the decode band's two promises: a row has the same bits at every M, and an
// expert's moe output row has the bits of the gemm's for that row and expert. Ordinary host CTest
// skips this; the explicit card entry is the hardware gate. No model weights required.
#include "rad_test.h"
#include "rad_abi.h"
#include "rad_device.h"
#include "rad_plugin.h"
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
/* REQUIRE for the helpers that return a value (REQUIRE returns void): the runner reports the throw
 * as the case's failure. */
#define ENSURE_OK(expr)                                                                       \
    do { const int rc_ = (expr);                                                              \
         if (rc_ != RAD_OK) throw std::runtime_error(#expr " = " + std::to_string(rc_)); } while (0)

namespace {

struct Library {
    void* h;
    explicit Library(const char* name) {
        const char* home = std::getenv("RADIANCE_HOME");
        h = dlopen((std::string(home ? home : "") + "/kernels/" + name + ".so").c_str(),
                   RTLD_NOW | RTLD_LOCAL);
        if (!h) throw std::runtime_error(dlerror());
    }
    ~Library() { dlclose(h); }
    const RadKernelInfo* row(const char* name) {
        auto count = (int (*)())dlsym(h, "rad_kernel_count");
        auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        for (int i = 0; i < count(); ++i)
            if (!std::strcmp(at(i)->name, name)) return at(i);
        throw std::runtime_error(std::string("missing row ") + name);
    }
};

RadParam pi(const char* k, long long v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}
RadParam ps(const char* k, const char* v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_STR;
    p.sval = v;
    return p;
}

bool device() {
    RadDeviceProps p{};
    return rad_dev_props(0, &p) == RAD_OK && !p.is_host_backend && rad_dev_set(0) == RAD_OK;
}

/* llama.cpp's iq2xxs_grid (ggml-common.h): 256 rows of 8 bytes, each byte 8, 25 or 43. */
const uint64_t kGrid[256] = {
    0x0808080808080808ull, 0x080808080808082bull, 0x0808080808081919ull, 0x0808080808082b08ull,
    0x0808080808082b2bull, 0x0808080808190819ull, 0x0808080808191908ull, 0x08080808082b0808ull,
    0x08080808082b082bull, 0x08080808082b2b08ull, 0x08080808082b2b2bull, 0x0808080819080819ull,
    0x0808080819081908ull, 0x0808080819190808ull, 0x0808080819192b08ull, 0x08080808192b0819ull,
    0x08080808192b1908ull, 0x080808082b080808ull, 0x080808082b08082bull, 0x080808082b082b2bull,
    0x080808082b2b082bull, 0x0808081908080819ull, 0x0808081908081908ull, 0x0808081908190808ull,
    0x0808081908191919ull, 0x0808081919080808ull, 0x080808192b081908ull, 0x080808192b192b08ull,
    0x0808082b08080808ull, 0x0808082b0808082bull, 0x0808082b082b082bull, 0x0808082b2b08082bull,
    0x0808190808080819ull, 0x0808190808081908ull, 0x0808190808190808ull, 0x08081908082b0819ull,
    0x08081908082b1908ull, 0x0808190819080808ull, 0x080819081908082bull, 0x0808190819082b08ull,
    0x08081908192b0808ull, 0x080819082b080819ull, 0x080819082b081908ull, 0x080819082b190808ull,
    0x080819082b2b1908ull, 0x0808191908080808ull, 0x080819190808082bull, 0x0808191908082b08ull,
    0x08081919082b0808ull, 0x080819191908192bull, 0x08081919192b2b19ull, 0x080819192b080808ull,
    0x080819192b190819ull, 0x0808192b08082b19ull, 0x0808192b08190808ull, 0x0808192b19080808ull,
    0x0808192b2b081908ull, 0x0808192b2b2b1908ull, 0x08082b0808080808ull, 0x08082b0808081919ull,
    0x08082b0808082b08ull, 0x08082b0808191908ull, 0x08082b08082b2b08ull, 0x08082b0819080819ull,
    0x08082b0819081908ull, 0x08082b0819190808ull, 0x08082b081919082bull, 0x08082b082b082b08ull,
    0x08082b1908081908ull, 0x08082b1919080808ull, 0x08082b2b0808082bull, 0x08082b2b08191908ull,
    0x0819080808080819ull, 0x0819080808081908ull, 0x0819080808190808ull, 0x08190808082b0819ull,
    0x0819080819080808ull, 0x08190808192b0808ull, 0x081908082b081908ull, 0x081908082b190808ull,
    0x081908082b191919ull, 0x0819081908080808ull, 0x0819081908082b08ull, 0x08190819082b0808ull,
    0x0819081919190808ull, 0x0819081919192b2bull, 0x081908192b080808ull, 0x0819082b082b1908ull,
    0x0819082b19081919ull, 0x0819190808080808ull, 0x0819190808082b08ull, 0x08191908082b0808ull,
    0x08191908082b1919ull, 0x0819190819082b19ull, 0x081919082b080808ull, 0x0819191908192b08ull,
    0x08191919192b082bull, 0x0819192b08080808ull, 0x0819192b0819192bull, 0x08192b0808080819ull,
    0x08192b0808081908ull, 0x08192b0808190808ull, 0x08192b0819080808ull, 0x08192b082b080819ull,
    0x08192b1908080808ull, 0x08192b1908081919ull, 0x08192b192b2b0808ull, 0x08192b2b19190819ull,
    0x082b080808080808ull, 0x082b08080808082bull, 0x082b080808082b2bull, 0x082b080819081908ull,
    0x082b0808192b0819ull, 0x082b08082b080808ull, 0x082b08082b08082bull, 0x082b0819082b2b19ull,
    0x082b081919082b08ull, 0x082b082b08080808ull, 0x082b082b0808082bull, 0x082b190808080819ull,
    0x082b190808081908ull, 0x082b190808190808ull, 0x082b190819080808ull, 0x082b19081919192bull,
    0x082b191908080808ull, 0x082b191919080819ull, 0x082b1919192b1908ull, 0x082b192b2b190808ull,
    0x082b2b0808082b08ull, 0x082b2b08082b0808ull, 0x082b2b082b191908ull, 0x082b2b2b19081908ull,
    0x1908080808080819ull, 0x1908080808081908ull, 0x1908080808190808ull, 0x1908080808192b08ull,
    0x19080808082b0819ull, 0x19080808082b1908ull, 0x1908080819080808ull, 0x1908080819082b08ull,
    0x190808081919192bull, 0x19080808192b0808ull, 0x190808082b080819ull, 0x190808082b081908ull,
    0x190808082b190808ull, 0x1908081908080808ull, 0x19080819082b0808ull, 0x19080819192b0819ull,
    0x190808192b080808ull, 0x190808192b081919ull, 0x1908082b08080819ull, 0x1908082b08190808ull,
    0x1908082b19082b08ull, 0x1908082b1919192bull, 0x1908082b192b2b08ull, 0x1908190808080808ull,
    0x1908190808082b08ull, 0x19081908082b0808ull, 0x190819082b080808ull, 0x190819082b192b19ull,
    0x190819190819082bull, 0x19081919082b1908ull, 0x1908192b08080808ull, 0x19082b0808080819ull,
    0x19082b0808081908ull, 0x19082b0808190808ull, 0x19082b0819080808ull, 0x19082b0819081919ull,
    0x19082b1908080808ull, 0x19082b1919192b08ull, 0x19082b19192b0819ull, 0x19082b192b08082bull,
    0x19082b2b19081919ull, 0x19082b2b2b190808ull, 0x1919080808080808ull, 0x1919080808082b08ull,
    0x1919080808190819ull, 0x1919080808192b19ull, 0x19190808082b0808ull, 0x191908082b080808ull,
    0x191908082b082b08ull, 0x1919081908081908ull, 0x191908191908082bull, 0x191908192b2b1908ull,
    0x1919082b2b190819ull, 0x191919082b190808ull, 0x191919082b19082bull, 0x1919191908082b2bull,
    0x1919192b08080819ull, 0x1919192b19191908ull, 0x19192b0808080808ull, 0x19192b0808190819ull,
    0x19192b0808192b19ull, 0x19192b08192b1908ull, 0x19192b1919080808ull, 0x19192b2b08082b08ull,
    0x192b080808081908ull, 0x192b080808190808ull, 0x192b080819080808ull, 0x192b0808192b2b08ull,
    0x192b081908080808ull, 0x192b081919191919ull, 0x192b082b08192b08ull, 0x192b082b192b0808ull,
    0x192b190808080808ull, 0x192b190808081919ull, 0x192b191908190808ull, 0x192b19190819082bull,
    0x192b19192b081908ull, 0x192b2b081908082bull, 0x2b08080808080808ull, 0x2b0808080808082bull,
    0x2b08080808082b2bull, 0x2b08080819080819ull, 0x2b0808082b08082bull, 0x2b08081908081908ull,
    0x2b08081908192b08ull, 0x2b08081919080808ull, 0x2b08082b08190819ull, 0x2b08190808080819ull,
    0x2b08190808081908ull, 0x2b08190808190808ull, 0x2b08190808191919ull, 0x2b08190819080808ull,
    0x2b081908192b0808ull, 0x2b08191908080808ull, 0x2b0819191908192bull, 0x2b0819192b191908ull,
    0x2b08192b08082b19ull, 0x2b08192b19080808ull, 0x2b08192b192b0808ull, 0x2b082b080808082bull,
    0x2b082b1908081908ull, 0x2b082b2b08190819ull, 0x2b19080808081908ull, 0x2b19080808190808ull,
    0x2b190808082b1908ull, 0x2b19080819080808ull, 0x2b1908082b2b0819ull, 0x2b1908190819192bull,
    0x2b1908192b080808ull, 0x2b19082b19081919ull, 0x2b19190808080808ull, 0x2b191908082b082bull,
    0x2b19190819081908ull, 0x2b19191919190819ull, 0x2b192b082b080819ull, 0x2b192b19082b0808ull,
    0x2b2b08080808082bull, 0x2b2b080819190808ull, 0x2b2b08082b081919ull, 0x2b2b081908082b19ull,
    0x2b2b082b08080808ull, 0x2b2b190808192b08ull, 0x2b2b2b0819190808ull, 0x2b2b2b1908081908ull,
};

double h2d(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    const double v = e ? std::ldexp(1.0 + m / 1024.0, e - 15) : std::ldexp(m / 1024.0, -14);
    return s ? -v : v;
}

/* One weight in the stored form, and its values by llama.cpp's dequant. */
struct Weight {
    bool iq2;
    int N, K;
    long long rb;   /* row bytes */
    std::vector<uint8_t> w;
    Weight(bool iq2_, int N_, int K_, unsigned seed) : iq2(iq2_), N(N_), K(K_) {
        const int bb = iq2 ? 66 : 34, qk = iq2 ? 256 : 32;
        rb = (long long)(K / qk) * bb;
        w.resize((size_t)rb * N);
        std::mt19937 g(seed);
        for (auto& b : w) b = (uint8_t)(g() & 0xffu);
        for (int n = 0; n < N; ++n)
            for (int b = 0; b < K / qk; ++b) {   /* d: a positive normal f16, 2^-8 .. 2^-7 */
                uint8_t* p = &w[(size_t)n * rb + (size_t)b * bb];
                const uint16_t d = (uint16_t)(0x1c00u + g() % 0x0800u);
                p[0] = (uint8_t)(d & 0xffu);
                p[1] = (uint8_t)(d >> 8);
            }
    }
    double val(int n, int k) const {
        const uint8_t* r = &w[(size_t)n * rb];
        if (!iq2) {
            const uint8_t* b = r + (size_t)(k / 32) * 34;
            return h2d((uint16_t)(b[0] | (b[1] << 8))) * (double)(int8_t)b[2 + k % 32];
        }
        const uint8_t* b = r + (size_t)(k / 256) * 66;
        const int kk = k % 256, u = kk / 32, G = (kk % 32) / 8, t = kk % 8;
        uint32_t a0, a1;
        std::memcpy(&a0, b + 2 + 8 * u, 4);
        std::memcpy(&a1, b + 6 + 8 * u, 4);
        const double db = h2d((uint16_t)(b[0] | (b[1] << 8))) * (0.5 + (a1 >> 28)) * 0.25;
        const int gv = (int)((kGrid[(a0 >> (8 * G)) & 255u] >> (8 * t)) & 255u);
        const uint32_t sx = (a1 >> (7 * G)) & 127u;
        const uint32_t sg = sx | ((uint32_t)(__builtin_popcount(sx) & 1) << 7);
        return ((sg >> t) & 1u) ? -db * gv : db * gv;
    }
};

struct Dev {
    std::vector<void*> alloc;
    void* put(const void* host, size_t bytes, RadStream s) {
        void* d = rad_dev_alloc(bytes ? bytes : 1, RAD_MEM_DEVICE);
        if (!d) throw std::runtime_error("rad_dev_alloc failed");
        alloc.push_back(d);
        if (host && bytes) ENSURE_OK(rad_memcpy_async(d, host, bytes, s));
        return d;
    }
    ~Dev() { for (void* p : alloc) rad_dev_free(p, RAD_MEM_DEVICE); }
};

RadTensor tensor(void* d, int dtype, long long r, long long c) {
    RadTensor t{};
    t.data = d;
    t.dtype = dtype;
    t.rank = 2;
    t.shape[0] = r;
    t.shape[1] = c;
    t.stride[0] = c;
    t.stride[1] = 1;
    return t;
}

/* gemm_nt_ggml (bf16 out) or logits_gemm_ggml (f32 out) at M rows; returns the output bits. */
std::vector<uint8_t> gemm(const RadKernelInfo* kg, const Weight& W, const std::vector<uint16_t>& x,
                          int M, bool logits, RadStream s) {
    Dev d;
    const int N = W.N, K = W.K, ob = logits ? 4 : 2;
    void* dw = d.put(W.w.data(), W.w.size(), s);
    void* dx = d.put(x.data(), (size_t)M * K * 2, s);
    void* dy = d.put(nullptr, (size_t)M * N * ob, s);
    RadTensor t[3] = { tensor(dx, RAD_BF16, M, K), tensor(dw, RAD_U8, N, W.rb),
                       tensor(dy, logits ? RAD_F32 : RAD_BF16, M, N) };
    std::vector<RadParam> p = { ps("type", W.iq2 ? "iq2_xxs" : "q8_0"), pi("M", M) };
    if (logits) { p.push_back(pi("n_vocab", N)); p.push_back(pi("n_embd", K)); }
    else { p.push_back(pi("N", N)); p.push_back(pi("K", K)); }
    p.push_back(ps("dtype", "bf16"));
    RadArgs a{};
    a.t = t;
    a.n_t = 3;
    a.p = p.data();
    a.n_p = p.size();
    ENSURE_OK(kg->launch(&a, s));
    std::vector<uint8_t> got((size_t)M * N * ob);
    ENSURE_OK(rad_memcpy_async(got.data(), dy, got.size(), s));
    ENSURE_OK(rad_stream_sync(s));
    return got;
}

/* The double reference of the first M rows against the output: f32 sums miss the exact sum by at
 * most K * eps32 * l1, and both round to the output dtype. */
void check(const Weight& W, const std::vector<uint16_t>& x, const std::vector<uint8_t>& got, int M,
           bool logits) {
    const int N = W.N, K = W.K;
    const double eps32 = 1.1982616e-7, outr = logits ? 1e-6 : 0.008;
    double worst = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; n += (n % 97 == 0 ? 1 : 13)) {
            double r = 0, l1 = 0;
            for (int k = 0; k < K; ++k) {
                const double t = W.val(n, k) * rad_bf16_to_f32(x[(size_t)m * K + k]);
                r += t;
                l1 += std::fabs(t);
            }
            double g;
            if (logits) { float f; std::memcpy(&f, &got[((size_t)m * N + n) * 4], 4); g = f; }
            else g = rad_bf16_to_f32(*(const uint16_t*)&got[((size_t)m * N + n) * 2]);
            REQUIRE(std::isfinite(g));
            worst = std::max(worst, std::fabs(g - r) / (std::fabs(r) + 1e-6));
            CHECK(std::fabs(g - r) <= 2.0 * eps32 * K * l1 + outr * std::fmax(std::fabs(g), std::fabs(r)) + 1e-30);
        }
    fprintf(stderr, "    %s %s N=%d K=%d M=%d worstRel=%.3g\n", logits ? "logits" : "gemm",
            W.iq2 ? "iq2_xxs" : "q8_0", N, K, M, worst);
}

std::vector<uint16_t> acts(int M, int K, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<uint16_t> x((size_t)M * K);
    for (auto& v : x) v = rad_f32_to_bf16(u(g));
    return x;
}

/* M = 48 checked against the reference, then M = 1, 8, 17: their rows have the bits of M = 48's. */
void band(const RadKernelInfo* kg, const Weight& W, bool logits, RadStream s) {
    const auto x = acts(48, W.K, 0xacu + (unsigned)W.N);
    const auto g48 = gemm(kg, W, x, 48, logits, s);
    check(W, x, g48, 48, logits);
    for (int M : { 1, 8, 17 }) {
        const auto g = gemm(kg, W, x, M, logits, s);
        CHECK(std::memcmp(g.data(), g48.data(), g.size()) == 0);
    }
}

/* moe_gemm_ggml over 64 IQ2_XXS experts, 8 tokens at top 9 (a speculative verify of depth 7), a
 * dropped slot among them: every output row is the gemm's row for its token and expert, bit for
 * bit, and the dropped slot's row is zeros. */
void moe(const RadKernelInfo* km, const RadKernelInfo* kg, int N, int K, RadStream s) {
    const int ne = 64, top_k = 9, tok = 8, T = tok * top_k;
    std::vector<Weight> ex;
    std::vector<uint8_t> all;
    for (int e = 0; e < ne; ++e) {
        ex.emplace_back(true, N, K, 0x900u + (unsigned)e);
        all.insert(all.end(), ex.back().w.begin(), ex.back().w.end());
    }
    const long long rb = ex[0].rb;
    const auto x = acts(tok, K, 0x77u);
    std::mt19937 g(5);
    std::vector<std::vector<int>> by(ne);
    for (int t = 0; t < tok; ++t) {
        std::vector<int> p(ne);
        for (int i = 0; i < ne; ++i) p[i] = i;
        std::shuffle(p.begin(), p.end(), g);
        for (int j = 0; j < top_k; ++j) by[p[j]].push_back(t * top_k + j);
    }
    std::vector<int32_t> sorted, off(ne + 1, 0);
    for (int e = 0; e < ne; ++e) {
        off[e] = (int32_t)sorted.size();
        for (int f : by[e]) sorted.push_back(f);
    }
    off[ne] = (int32_t)sorted.size();
    const int dropped = 3;
    const int32_t dropped_flat = sorted[dropped];
    sorted[dropped] = -1;
    Dev d;
    void* dw = d.put(all.data(), all.size(), s);
    void* dx = d.put(x.data(), x.size() * 2, s);
    void* dso = d.put(sorted.data(), sorted.size() * 4, s);
    void* dof = d.put(off.data(), off.size() * 4, s);
    void* dy = d.put(nullptr, (size_t)T * N * 2, s);
    RadTensor W{};
    W.data = dw;
    W.dtype = RAD_U8;
    W.rank = 3;
    W.shape[0] = ne; W.shape[1] = N; W.shape[2] = rb;
    W.stride[0] = N * rb; W.stride[1] = rb; W.stride[2] = 1;
    RadTensor so{}, of{};
    so.data = dso; so.dtype = RAD_I32; so.rank = 1; so.shape[0] = T; so.stride[0] = 1;
    of.data = dof; of.dtype = RAD_I32; of.rank = 1; of.shape[0] = ne + 1; of.stride[0] = 1;
    RadTensor t[5] = { tensor(dx, RAD_BF16, tok, K), W, so, of, tensor(dy, RAD_BF16, T, N) };
    std::vector<RadParam> p = { ps("type", "iq2_xxs"), pi("N", N), pi("K", K), pi("n_expert", ne),
                                pi("top_k", top_k), ps("a_order", "token"), ps("dtype", "bf16") };
    RadArgs a{};
    a.t = t;
    a.n_t = 5;
    a.p = p.data();
    a.n_p = p.size();
    REQUIRE_OK(km->launch(&a, s));
    std::vector<uint16_t> y((size_t)T * N);
    REQUIRE_OK(rad_memcpy_async(y.data(), dy, y.size() * 2, s));
    REQUIRE_OK(rad_stream_sync(s));
    int rows = 0;
    for (int e = 0; e < ne; ++e) {
        if (off[e + 1] == off[e]) continue;
        const auto ge = gemm(kg, ex[e], x, tok, false, s);
        for (int i = off[e]; i < off[e + 1]; ++i) {
            const uint16_t* yr = &y[(size_t)i * N];
            if (i == dropped) {
                CHECK(std::all_of(yr, yr + N, [](uint16_t v) { return v == 0; }));
                continue;
            }
            CHECK(std::memcmp(yr, &ge[(size_t)(sorted[i] / top_k) * N * 2], (size_t)N * 2) == 0);
            ++rows;
        }
    }
    (void)dropped_flat;
    fprintf(stderr, "    moe iq2_xxs N=%d K=%d: %d rows equal the gemm's\n", N, K, rows);
}

}  // namespace

TEST(ggml_wmma_device_rows_match_the_dequant_and_keep_their_bits) {
    if (!device()) SKIP("explicit R9700 card run required");
    Library lib("libr4d");
    auto* g256 = lib.row("gemm_nt_ggml_qk256_gemv");
    auto* g32 = lib.row("gemm_nt_ggml_qk32_gemv");
    auto* l32 = lib.row("logits_gemm_ggml_qk32_gemv");
    auto* m256 = lib.row("moe_gemm_ggml_qk256_gemv");
    RadStream s = nullptr;
    REQUIRE_OK(rad_stream_create(&s, 0));
    /* Nemotron-3.5-Bolt-74B's shapes: the IQ2_XXS shared experts and a short-K expert, the Q8_0
     * latent projections and attention, and a Q8_0 head. */
    band(g256, Weight(true, 5376, 4096, 1), false, s);
    band(g256, Weight(true, 4096, 5376, 2), false, s);
    band(g256, Weight(true, 2560, 1024, 3), false, s);
    band(g256, Weight(true, 1024, 1536, 4), false, s);
    band(g32, Weight(false, 1024, 4096, 5), false, s);
    band(g32, Weight(false, 4096, 1024, 6), false, s);
    band(g32, Weight(false, 4608, 4096, 7), false, s);
    band(l32, Weight(false, 8192, 4096, 8), true, s);
    moe(m256, g256, 1024, 2560, s);
    moe(m256, g256, 2560, 1024, s);
    rad_stream_destroy(s);
}

RAD_TEST_MAIN()
