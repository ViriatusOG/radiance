// test_ks.hip -- the decode form's K split across workgroups (KS, exl3i_nt.hip.h / nt.c ks_slices) against the unsplit
// nt_pass, through the plugin ABI on random trellis weights at the trunk shapes. Usage:
//   test_ks EXL3I.so [--rates 0.5,1,2.5,4,6.5] [--codebooks mad]
// Per (shape, rate, codebook, T in 1 4 8 16): y with the KSPLIT pin at 1 (off) and with the shipped choice (auto), bf16 out;
// reported: the slices, max |y_ks - y_off|, that over max |y_off| (rel), and the count of bf16 values that differ.
// Checks: rel <= 2^-7 (a few bf16 ulps at the output's scale: one f32 reorder of a K-long sum), the KS run byte-identical
// when repeated 20 times back to back (the slice-order reduce; the counters reset by each launch), and the T = 1 row of
// a T = 4 launch identical to the T = 1 launch's (rows independent; with DOT, T = 1's form, only within the bound). And
// at T 1 the DOT form (four v_dot2 a tile) against the WMMA form (the DEC_DOT_OFF pin), KS off and on. Last line PASS / FAIL.
#include <hip/hip_runtime.h>
#include "exl3i.h"
#include "exl3i_decode.h"
#include "exl3i_pin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#define HIPCHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

struct Shape { const char* name; int N, K; bool stacked; };
static const Shape SHAPES[] = {
    { "gate_up", 34816, 5120, true }, { "down", 5120, 17408, false }, { "gdn_qkv", 10240, 5120, false },
    { "gdn_z", 6144, 5120, false }, { "attn_qg", 12288, 5120, false }, { "attn_k", 1024, 5120, false },
    { "attn_v", 1024, 5120, false }, { "attn_o", 5120, 6144, false }, { "ssm_out", 5120, 6144, false },
};

static uint64_t RNG = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() { RNG ^= RNG << 13; RNG ^= RNG >> 7; RNG ^= RNG << 17; return (uint32_t)(RNG >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

static const RadKernelInfo* RF = nullptr;
static void (*PIN)(int, int64_t) = nullptr;   // the plugin's exl3i_test_pin (exl3i_pin.h)
static int FAILS = 0;

static RadTensor tensor(void* d, uint32_t dt, int64_t r0, int64_t r1) {
    RadTensor t{};
    t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r0; t.shape[1] = r1; t.stride[0] = r1; t.stride[1] = 1;
    return t;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_ks EXL3I.so [--rates a,b] [--codebooks a,b]\n"); return 2; }
    std::string rates = "0.5,1,1.5,2.5,4,6.5", cbs = "mad";
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--rates")) rates = argv[i + 1];
        else if (!std::strcmp(argv[i], "--codebooks")) cbs = argv[i + 1];
    }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto inf = (const RadPluginInfo* (*)())dlsym(h, "rad_plugin_info");
    auto cnt = (int (*)())dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    for (int i = 0; i < cnt(); ++i) if (!std::strcmp(at(i)->name, "exl3i_gemm_nt")) RF = at(i);
    if (!RF) { std::fprintf(stderr, "no exl3i_gemm_nt\n"); return 2; }
    PIN = (void (*)(int, int64_t))dlsym(h, "exl3i_test_pin");
    if (!PIN) { std::fprintf(stderr, "no exl3i_test_pin\n"); return 2; }
    std::printf("exl3i %s -- test_ks\n", inf()->version);
    auto split = [](const std::string& v) { std::vector<std::string> o; size_t p = 0;
        while (p <= v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); o.push_back(v.substr(p, q - p)); p = q + 1; }
        return o; };
    std::printf("%-8s %5s %-7s %3s %4s %12s %10s %8s\n", "shape", "rate", "cb", "T", "ksn", "max_abs", "rel", "ndiff");
    double worst = 0;
    for (const std::string& cbn : split(cbs))
    for (const std::string& rs : split(rates)) {
        const int m = exl3i_rate_parse(rs.c_str());
        const int cb = exl3i_codebook_from_name(cbn.c_str());
        for (const Shape& s : SHAPES) {
            RadEncoding e;
            exl3i_encoding_make(m, cb, s.stacked, s.N, s.K, &e);
            const int W = exl3i_row_words(s.K, m);
            std::vector<uint32_t> rec((size_t)(s.N / 16) * W);
            for (auto& x : rec) x = rnd() ^ (rnd() << 16);
            char rt[16];
            exl3i_rate_str(m, rt);
            auto params = [&](int64_t M) {
                return std::vector<RadParam>{ RAD_INT("M", M), RAD_INT("N", s.N), RAD_INT("K", s.K), RAD_STR("rate", rt),
                                              RAD_STR("codebook", cbn.c_str()), RAD_STR("act", "f16") };
            };
            auto pa = params(16);
            RadTensor pl = tensor(rec.data(), RAD_U32, s.N / 16, W);
            RadLayout L;
            int sel = EXL3I_PLANE_RECORD;
            if (RF->layout(pa.data(), 6, EXL3I_OPD_RECORD, &e, &sel, &pl, 1, &L) != RAD_OK) { std::printf("layout refused\n"); return 2; }
            std::vector<char> buf((size_t)L.bytes);
            RF->relayout(pa.data(), 6, EXL3I_OPD_RECORD, &e, &sel, &pl, 1, buf.data(), L.bytes);
            void *drec, *suh, *suh1 = nullptr, *svh, *x, *y;
            HIPCHECK(hipMalloc(&drec, buf.size()));
            HIPCHECK(hipMemcpy(drec, buf.data(), buf.size(), hipMemcpyHostToDevice));
            auto vec = [](int n) {
                std::vector<uint16_t> v((size_t)n);
                for (auto& q : v) { _Float16 hh = (_Float16)((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f)); std::memcpy(&q, &hh, 2); }
                void* p; HIPCHECK(hipMalloc(&p, (size_t)n * 2)); HIPCHECK(hipMemcpy(p, v.data(), (size_t)n * 2, hipMemcpyHostToDevice));
                return p;
            };
            suh = vec(s.K); svh = vec(s.N);
            if (s.stacked) suh1 = vec(s.K);
            HIPCHECK(hipMalloc(&x, (size_t)16 * s.K * 2));
            HIPCHECK(hipMalloc(&y, (size_t)16 * s.N * 2));
            {
                std::vector<uint16_t> hx((size_t)16 * s.K);
                for (auto& v : hx) { float f = ((int)(rnd() & 0xFFFF) - 32768) / 16384.0f; uint32_t u; std::memcpy(&u, &f, 4); v = (uint16_t)(u >> 16); }
                HIPCHECK(hipMemcpy(x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
            }
            auto run = [&](int T, int ksplit, int reps, bool dot = true) {
                PIN(EXL3I_PIN_KSPLIT, ksplit);
                PIN(EXL3I_PIN_DEC_DOT_OFF, dot ? 0 : 1);
                auto p = params(T);
                RadTensor t[6]{};
                t[0] = tensor(x, RAD_BF16, T, s.K);
                t[1] = tensor(drec, EXL3I_DT_TILES, s.N / 16, W);
                t[2] = tensor(suh, RAD_F16, 1, s.K);
                t[3] = tensor(svh, RAD_F16, s.N, 1);
                t[4] = tensor(suh1, RAD_F16, 1, s.K);
                t[5] = tensor(y, RAD_BF16, T, s.N);
                RadArgs a{};
                a.t = t; a.n_t = 6; a.p = p.data(); a.n_p = 6; a.world_size = 1;
                const int64_t sb = RF->scratch(&a);
                void* sc = nullptr;
                if (sb > 0) HIPCHECK(hipMalloc(&sc, (size_t)sb));
                a.scratch = sc; a.scratch_bytes = sb;
                std::vector<std::vector<uint16_t>> outs;
                for (int r = 0; r < reps; ++r) {
                    HIPCHECK(hipMemset(y, 0xFF, (size_t)T * s.N * 2));
                    if (RF->launch(&a, nullptr) != RAD_OK) { std::printf("launch failed\n"); ++FAILS; }
                    HIPCHECK(hipDeviceSynchronize());
                    outs.emplace_back((size_t)T * s.N);
                    HIPCHECK(hipMemcpy(outs.back().data(), y, (size_t)T * s.N * 2, hipMemcpyDeviceToHost));
                }
                if (sc) HIPCHECK(hipFree(sc));
                PIN(EXL3I_PIN_KSPLIT, 0);
                PIN(EXL3I_PIN_DEC_DOT_OFF, 0);
                return outs;
            };
            // the slices the shipped rule picks (nt.c ks_slices, repeated here for the report)
            const int nkb = s.K / 128, cbs_ = s.N / 128;
            int ksn = (512 + cbs_ - 1) / cbs_;   // nt.c EXL3I_KS_WG
            ksn = std::min(std::min(ksn, nkb / 2), 32);
            if (ksn < 2) ksn = 1;
            std::vector<uint16_t> ks1;
            for (int T : { 1, 4, 8, 16 }) {
                const auto off = run(T, 1, 1)[0];
                const auto on = run(T, 0, 20);
                double mx = 0, ref = 0;
                long nd = 0;
                for (size_t i = 0; i < off.size(); ++i) {
                    const double a = bf2f(off[i]), b = bf2f(on[0][i]);
                    if (!std::isfinite(a) || !std::isfinite(b)) { ++FAILS; std::printf("non-finite\n"); break; }
                    mx = std::max(mx, std::fabs(a - b));
                    ref = std::max(ref, std::fabs(a));
                    nd += off[i] != on[0][i];
                }
                const double rel = ref > 0 ? mx / ref : 0;
                worst = std::max(worst, rel);
                bool rep_ok = true;
                for (auto& o : on) rep_ok &= o == on[0];
                if (T == 1) ks1 = on[0];
                bool row_ok = T == 1 || std::equal(ks1.begin(), ks1.end(), on[0].begin());
                if (!row_ok) {   // DOT: T 1 sums in another order than T 4's WMMA; within the bound instead
                    double m2 = 0, r2 = 0;
                    for (size_t i = 0; i < ks1.size(); ++i) { m2 = std::max(m2, (double)std::fabs(bf2f(ks1[i]) - bf2f(on[0][i]))); r2 = std::max(r2, (double)std::fabs(bf2f(ks1[i]))); }
                    row_ok = m2 <= r2 / 128;
                }
                std::printf("%-8s %5s %-7s %3d %4d %12.4g %10.3g %8ld%s%s\n", s.name, rs.c_str(), cbn.c_str(), T, ksn, mx, rel, nd,
                            rep_ok ? "" : "  NOT REPEATABLE", row_ok ? "" : "  ROW 0 DIFFERS FROM T=1");
                if (rel > 1.0 / 128 || !rep_ok || !row_ok) ++FAILS;
            }
            {   // DOT (T 1): the dot form vs the WMMA form, split and unsplit
                for (int ksp : { 1, 0 }) {
                    const auto wm = run(1, ksp, 1, false)[0], dt = run(1, ksp, 1, true)[0];
                    double mx = 0, ref = 0;
                    long nd = 0;
                    for (size_t i = 0; i < wm.size(); ++i) {
                        const double a = bf2f(wm[i]), b = bf2f(dt[i]);
                        if (!std::isfinite(b)) { ++FAILS; break; }
                        mx = std::max(mx, std::fabs(a - b)); ref = std::max(ref, std::fabs(a)); nd += wm[i] != dt[i];
                    }
                    const double rel = ref > 0 ? mx / ref : 0;
                    worst = std::max(worst, rel);
                    std::printf("%-8s %5s %-7s %3d %4s %12.4g %10.3g %8ld  (DOT vs WMMA, KS %s)\n", s.name, rs.c_str(), cbn.c_str(), 1,
                                ksp == 1 ? "1" : "auto", mx, rel, nd, ksp == 1 ? "off" : "on");
                    if (rel > 1.0 / 128) ++FAILS;
                }
            }
            HIPCHECK(hipFree(drec)); HIPCHECK(hipFree(suh)); HIPCHECK(hipFree(svh)); if (suh1) HIPCHECK(hipFree(suh1));
            HIPCHECK(hipFree(x)); HIPCHECK(hipFree(y));
        }
    }
    std::printf("worst rel %.3g\n%s (%d failures)\n", worst, FAILS ? "TEST_KS FAIL" : "TEST_KS PASS", FAILS);
    return FAILS ? 1 : 0;
}
