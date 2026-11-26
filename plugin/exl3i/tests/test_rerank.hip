// test_rerank.hip -- exl3i_rerank's device row (include/exl3i_rerank.h) through the plugin ABI. Usage:
//   test_rerank EXL3I.so [--quick] [--small]   (--small: the small shapes alone, not the 248320-row heads)
// Per (N, K, rate, codebook, row order, M): a random record laid out by the row's own layout hook (exl3i_dense's), the
// candidates drawn as the drafter's are (distinct ids, a few pads, some rows sharing an output block), then
//   1. the device row == the host reference (rerank_ref.c) bit for bit: idx, val, the pair plane; idx in place (= idx_in)
//      the same; two launches the same bytes;
//   2. val against exl3i_dense's own logits at those rows (exl3i_gemm_nt, act f16, launched at T = M on the same stored
//      record: nt_pass's decode form, DOT + KS at T 1): |d| within the f32 summation bound K u L1, L1 the sum of the
//      logit's terms' magnitudes (the oracle's decode of the candidate's block); the worst |d| / (u L1) printed;
//      for mad also against the a8g128 row at T 1 (the byte-sum GEMV, int8 rows: printed, not a check);
//   3. the time of a launch (M rows x R candidates) against exl3i_dense's full head at T = M.
// Last line PASS / FAIL.
#include <hip/hip_runtime.h>
#include "exl3i.h"
#include "exl3i_rerank.h"
#include "exl3i_rerank_dev.h"
#include "exl3i_decode.h"
#include "exl3_decode.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

#define HIPCHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

static uint64_t RNG = 0x2545F4914F6CDD1Dull;
static uint32_t rnd() { RNG ^= RNG << 13; RNG ^= RNG >> 7; RNG ^= RNG << 17; return (uint32_t)(RNG >> 16); }
static int FAILS = 0;
static const RadKernelInfo *RR = nullptr, *RF = nullptr, *RA = nullptr;
static int (*ROT)(const Exl3iRerankCall*, uint16_t*, void*) = nullptr;   // the device row's rotation alone

static RadTensor tensor(void* d, uint32_t dt, int64_t r0, int64_t r1) {
    RadTensor t{};
    t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r0; t.shape[1] = r1; t.stride[0] = r1; t.stride[1] = 1;
    return t;
}
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
template <class T> static T* dput(const std::vector<T>& v) {
    void* p; HIPCHECK(hipMalloc(&p, v.size() * sizeof(T) + 256)); HIPCHECK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return (T*)p;
}
template <class T> static std::vector<T> dget(const void* p, size_t n) {
    std::vector<T> v(n); HIPCHECK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return v;
}

struct Case { int64_t N, K; const char* rate; const char* cb; int r16; int x_f16; };

static void run(const Case& c, const std::vector<int>& Ms, bool time_it) {
    const int m = exl3i_rate_parse(c.rate), cb = exl3i_codebook_from_name(c.cb), W = m / 2;
    const bool dq = cb == EXL3I_CB_MAD;
    const int64_t N = c.N, K = c.K, rw = exl3i_row_words(K, m), R = 32;
    RadEncoding enc;
    if ((c.r16 ? exl3i_encoding_make_r16(m, cb, 0, N, K, &enc) : exl3i_encoding_make(m, cb, 0, N, K, &enc)) != RAD_OK) {
        std::printf("encoding refused\n"); ++FAILS; return;
    }
    std::vector<uint32_t> rec((size_t)(N / 16) * rw);
    for (auto& w : rec) w = rnd() ^ (rnd() << 16);
    std::vector<uint16_t> suh((size_t)K), svh((size_t)N);
    for (auto& q : suh) q = exl3_half_from_double((rnd() & 1 ? -1.0 : 1.0) * (0.5 + (rnd() & 1023) / 1024.0));
    for (auto& q : svh) q = exl3_half_from_double((rnd() & 1 ? -1.0 : 1.0) * (0.25 + (rnd() & 1023) / 2048.0));
    const int Mx = 16;
    std::vector<uint16_t> x((size_t)Mx * K);
    for (auto& q : x) { const float f = ((int)(rnd() & 0xFFFF) - 32768) / 8192.0f; q = c.x_f16 ? exl3_half_from_double(f) : f2bf(f); }
    // the stored form, by the row's own hook (= exl3i_dense's)
    std::vector<RadParam> pp = { RAD_INT("M", Mx), RAD_INT("R", R), RAD_INT("N", N), RAD_INT("K", K), RAD_STR("rate", c.rate),
                                 RAD_STR("codebook", c.cb) };
    RadTensor pl = tensor(rec.data(), RAD_U32, N / 16, rw);
    RadLayout L{};
    int sel = EXL3I_PLANE_RECORD;
    if (RR->layout(pp.data(), (int)pp.size(), EXL3I_RR_RECORD, &enc, &sel, &pl, 1, &L) != RAD_OK) { std::printf("layout refused\n"); ++FAILS; return; }
    {
        RadLayout Ld{};
        std::vector<RadParam> pd = { RAD_INT("M", Mx), RAD_INT("N", N), RAD_INT("K", K), RAD_STR("rate", c.rate),
                                     RAD_STR("codebook", c.cb), RAD_STR("act", "f16") };
        RF->layout(pd.data(), (int)pd.size(), EXL3I_OPD_RECORD, &enc, &sel, &pl, 1, &Ld);
        if (!L.tag || !Ld.tag || std::strcmp(L.tag, Ld.tag) || L.dtype != Ld.dtype || L.bytes != Ld.bytes) {
            std::printf("stored form differs from exl3i_dense's (%s / %s)  FAIL\n", L.tag ? L.tag : "-", Ld.tag ? Ld.tag : "-"); ++FAILS;
        }
    }
    std::vector<char> lay((size_t)L.bytes);
    RR->relayout(pp.data(), (int)pp.size(), EXL3I_RR_RECORD, &enc, &sel, &pl, 1, lay.data(), L.bytes);
    void* drec; HIPCHECK(hipMalloc(&drec, lay.size())); HIPCHECK(hipMemcpy(drec, lay.data(), lay.size(), hipMemcpyHostToDevice));
    uint16_t *dsuh = dput(suh), *dsvh = dput(svh), *dx = dput(x);
    float* dy; HIPCHECK(hipMalloc(&dy, (size_t)Mx * N * 4));
    const uint32_t recdt = L.dtype;
    // the oracle's W_hat for the L1 bounds (and the rotated rows the bound multiplies)
    Exl3iView v = { N, K, m, cb, rec.data(), suh.data(), nullptr, svh.data() };
    const double s4 = 4.0 * exl3_half_to_float(0x1EEE), step = exl3_half_to_float(0x26EE), RS = 0.08838834764831845;
    if (ROT) {   // the device row's rotation == the host reference's, every row and feature
        Exl3iRerankCall rc{};
        rc.x = dx; rc.x_pitch = K; rc.x_f16 = c.x_f16; rc.suh = dsuh; rc.M = Mx; rc.K = K;
        uint16_t* dxr; HIPCHECK(hipMalloc(&dxr, (size_t)Mx * K * 2));
        if (ROT(&rc, dxr, nullptr) != RAD_OK) { std::printf("rotate refused\n"); ++FAILS; }
        HIPCHECK(hipDeviceSynchronize());
        const auto got = dget<uint16_t>(dxr, (size_t)Mx * K);
        std::vector<uint16_t> want((size_t)K);
        long nd = 0;
        for (int mi = 0; mi < Mx; ++mi) {
            exl3i_rerank_ref_rotate(x.data() + (size_t)mi * K, c.x_f16, suh.data(), K, want.data());
            for (int64_t k = 0; k < K; ++k)
                if (want[k] != got[(size_t)mi * K + k] && nd++ < 6)
                    std::printf("    rotation row %d feature %lld: host %04x device %04x (x %04x suh %04x)\n", mi, (long long)k,
                                want[k], got[(size_t)mi * K + k], x[(size_t)mi * K + k], suh[k]);
        }
        if (nd) { std::printf("  rotation: %ld of %lld features differ  FAIL\n", nd, (long long)Mx * K); ++FAILS; }
        HIPCHECK(hipFree(dxr));
    }
    for (int M : Ms) {
        std::vector<int32_t> in((size_t)M * R);
        for (int mi = 0; mi < M; ++mi)
            for (int r = 0; r < R; ++r) {
                int32_t g; bool dup;
                do {
                    const uint32_t u = rnd() % 64;
                    g = u == 0 ? -1 : u == 1 ? (int32_t)(N + rnd() % 5) : (int32_t)(rnd() % N);
                    if (r < 6) g = (int32_t)((mi * 977 + r) % N);   // neighbours: shared blocks
                    dup = false;
                    for (int q = 0; q < r; ++q) dup |= in[(size_t)mi * R + q] == g && g >= 0 && g < N;
                } while (dup);
                in[(size_t)mi * R + r] = g;
            }
        std::vector<int32_t> hidx((size_t)M * R);
        std::vector<float> hval((size_t)M * R), hpair((size_t)M * R * 2);
        exl3i_rerank_ref(x.data(), K, c.x_f16, rec.data(), suh.data(), svh.data(), in.data(), hidx.data(), hval.data(),
                         hpair.data(), M, R, N, K, 0, m, cb, c.r16);
        int32_t *din = dput(in), *dix; float *dva, *dpr;
        HIPCHECK(hipMalloc(&dix, (size_t)M * R * 4)); HIPCHECK(hipMalloc(&dva, (size_t)M * R * 4)); HIPCHECK(hipMalloc(&dpr, (size_t)M * R * 8));
        auto launch = [&](int32_t* idx_in, int32_t* idx, bool pairs) {
            std::vector<RadParam> p = { RAD_INT("M", M), RAD_INT("R", R), RAD_INT("N", N), RAD_INT("K", K),
                                        RAD_STR("rate", c.rate), RAD_STR("codebook", c.cb) };
            RadTensor t[EXL3I_RR_N_OPD]{};
            t[0] = tensor(dx, c.x_f16 ? RAD_F16 : RAD_BF16, M, K);
            t[1] = tensor(drec, recdt, N / 16, rw);
            t[2] = tensor(dsuh, RAD_F16, 1, K);
            t[3] = tensor(dsvh, RAD_F16, N, 1);
            t[4] = tensor(idx_in, RAD_I32, M, R);
            t[5] = tensor(idx, RAD_I32, M, R);
            t[6] = tensor(dva, RAD_F32, M, R);
            if (pairs) { t[7] = tensor(dpr, RAD_F32, M, R * 2); }
            RadArgs a{};
            a.t = t; a.n_t = EXL3I_RR_N_OPD; a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1;
            return RR->launch(&a, nullptr);
        };
        HIPCHECK(hipMemset(dva, 0xFF, (size_t)M * R * 4));
        int rc = launch(din, dix, true);
        HIPCHECK(hipDeviceSynchronize());
        const auto didx = dget<int32_t>(dix, (size_t)M * R);
        const auto dval = dget<float>(dva, (size_t)M * R);
        const auto dpair = dget<float>(dpr, (size_t)M * R * 2);
        long bad = rc != RAD_OK;
        bad += std::memcmp(didx.data(), hidx.data(), didx.size() * 4) != 0;
        bad += std::memcmp(dval.data(), hval.data(), dval.size() * 4) != 0;
        bad += std::memcmp(dpair.data(), hpair.data(), dpair.size() * 4) != 0;
        for (size_t q = 0, shown = 0; q < didx.size() && shown < 4; ++q)
            if (didx[q] != hidx[q] || std::memcmp(&dval[q], &hval[q], 4)) {
                std::printf("    row %zu slot %zu: device %d %.9g, host %d %.9g\n", q / R, q % R, didx[q], dval[q], hidx[q], hval[q]);
                ++shown;
            }
        // in place, and again: the same bytes
        int32_t* dip = dput(in);
        rc = launch(dip, dip, false);
        HIPCHECK(hipDeviceSynchronize());
        const bool ip_ok = rc == RAD_OK && dget<int32_t>(dip, (size_t)M * R) == didx && dget<float>(dva, (size_t)M * R) == dval;
        if (!ip_ok) std::printf("    in place / second launch differs\n");
        bad += !ip_ok;
        // exl3i_dense's logits at the candidates' rows
        auto dense = [&](const RadKernelInfo* row, const char* act, int T) {
            std::vector<RadParam> p = { RAD_INT("M", T), RAD_INT("N", N), RAD_INT("K", K), RAD_STR("rate", c.rate),
                                        RAD_STR("codebook", c.cb), RAD_STR("act", act) };
            RadTensor t[EXL3I_N_OPD]{};
            t[0] = tensor(dx, c.x_f16 ? RAD_F16 : RAD_BF16, T, K);
            t[1] = tensor(drec, recdt, N / 16, rw);
            t[2] = tensor(dsuh, RAD_F16, 1, K);
            t[3] = tensor(dsvh, RAD_F16, N, 1);
            t[5] = tensor(dy, RAD_F32, T, N);
            RadArgs a{};
            a.t = t; a.n_t = EXL3I_N_OPD; a.p = p.data(); a.n_p = (int)p.size(); a.world_size = 1;
            const int64_t sb = row->scratch(&a);
            void* sc = nullptr;
            if (sb > 0) HIPCHECK(hipMalloc(&sc, (size_t)sb));
            a.scratch = sc; a.scratch_bytes = sb;
            const int r = row->launch(&a, nullptr);
            HIPCHECK(hipDeviceSynchronize());
            if (sc) HIPCHECK(hipFree(sc));
            return r;
        };
        double worst = 0, worst8 = -1;
        if (dense(RF, "f16", M) != RAD_OK) { std::printf("exl3i_dense refused\n"); ++bad; }
        const auto y = dget<float>(dy, (size_t)M * N);
        std::vector<float> y8;
        if (dq && cb == EXL3I_CB_MAD && RA && M == 1 && dense(RA, "a8g128", 1) == RAD_OK) y8 = dget<float>(dy, (size_t)N);
        std::vector<uint16_t> xr((size_t)K);
        std::vector<float> blk((size_t)128 * K);
        for (int mi = 0; mi < M; ++mi) {
            exl3i_rerank_ref_rotate(x.data() + (size_t)mi * K, c.x_f16, suh.data(), K, xr.data());
            for (int o = 0; o < R; ++o) {
                const int32_t g = didx[(size_t)mi * R + o];
                if (g < 0) continue;
                const int64_t s = exl3i_rerank_row(g, N, c.r16), b0 = s / 128 * 128;
                exl3i_decode_what_rows(&v, b0, 128, blk.data());
                double l1 = 0;
                for (int64_t q = 0; q < 128 * K; ++q) {
                    const double w = dq ? std::fabs(std::rint(blk[q] / s4)) * step : std::fabs(blk[q]);
                    l1 += w * std::fabs(exl3_half_to_float(xr[q % K]));
                }
                l1 *= RS * std::fabs(exl3_half_to_float(svh[s]));
                const double d = std::fabs((double)dval[(size_t)mi * R + o] - (double)y[(size_t)mi * N + g]) / (std::ldexp(1.0, -24) * l1);
                if (d > worst) worst = d;
                if (!y8.empty() && mi == 0) {
                    const double d8 = std::fabs((double)dval[o] - (double)y8[g]) / l1;
                    if (d8 > worst8) worst8 = d8;
                }
            }
        }
        bad += worst > (double)K;
        char t8[64] = "";
        if (worst8 >= 0) std::snprintf(t8, sizeof t8, "; vs a8g128 T 1 %.2e L1", worst8);
        char tt[160] = "";
        if (time_it) {
            hipEvent_t e0, e1; HIPCHECK(hipEventCreate(&e0)); HIPCHECK(hipEventCreate(&e1));
            float ms_r = 0, ms_d = 0;
            for (int w = 0; w < 3; ++w) launch(din, dix, false);
            HIPCHECK(hipEventRecord(e0));
            for (int w = 0; w < 20; ++w) launch(din, dix, false);
            HIPCHECK(hipEventRecord(e1)); HIPCHECK(hipEventSynchronize(e1)); HIPCHECK(hipEventElapsedTime(&ms_r, e0, e1));
            for (int w = 0; w < 3; ++w) dense(RF, "f16", M);
            HIPCHECK(hipEventRecord(e0));
            for (int w = 0; w < 10; ++w) dense(RF, "f16", M);
            HIPCHECK(hipEventRecord(e1)); HIPCHECK(hipEventSynchronize(e1)); HIPCHECK(hipEventElapsedTime(&ms_d, e0, e1));
            std::snprintf(tt, sizeof tt, "; %.1f us a launch (full head exl3i_dense T %d: %.1f us incl. sync)", ms_r * 1000 / 20, M, ms_d * 1000 / 10);
        }
        std::printf("N %6lld K %5lld K%-6s %-6s %s x %s M %2d R %lld: device %s host; vs exl3i_dense %.1f u L1 (bound %lld)%s%s%s\n",
                    (long long)N, (long long)K, c.rate, c.cb, c.r16 ? "r16" : "nat", c.x_f16 ? "f16" : "bf16", M, (long long)R,
                    bad ? "!=" : "==", worst, (long long)K, t8, tt, bad ? "  FAIL" : "");
        FAILS += bad != 0;
        HIPCHECK(hipFree(din)); HIPCHECK(hipFree(dip)); HIPCHECK(hipFree(dix)); HIPCHECK(hipFree(dva)); HIPCHECK(hipFree(dpr));
    }
    HIPCHECK(hipFree(drec)); HIPCHECK(hipFree(dsuh)); HIPCHECK(hipFree(dsvh)); HIPCHECK(hipFree(dx)); HIPCHECK(hipFree(dy));
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_rerank EXL3I.so [--quick] [--small]\n"); return 2; }
    bool quick = false, small_only = false;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--quick")) quick = true;
        if (!std::strcmp(argv[i], "--small")) small_only = true;
    }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto cnt = (int (*)())dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    auto scnt = (int (*)())dlsym(h, "rad_kernel_schema_count");
    auto sat = (const RadOpSchema* (*)(int))dlsym(h, "rad_kernel_schema_at");
    for (int i = 0; i < cnt(); ++i) {
        if (!std::strcmp(at(i)->name, "exl3i_rerank")) RR = at(i);
        if (!std::strcmp(at(i)->name, "exl3i_gemm_nt")) RF = at(i);
        if (!std::strcmp(at(i)->name, "exl3i_gemm_nt_a8")) RA = at(i);
    }
    ROT = (int (*)(const Exl3iRerankCall*, uint16_t*, void*))dlsym(h, "exl3i_rerank_rotate_rows");
    bool schema = false;
    for (int i = 0; i < scnt(); ++i) schema |= !std::strcmp(sat(i)->op, EXL3I_OP_RERANK);
    if (!RR || !RF || !schema) { std::fprintf(stderr, "no exl3i_rerank row / schema\n"); return 2; }
    const Case big[] = { { 248320, 5120, "4", "mad", 1, 0 }, { 248320, 5120, "4.125", "mad", 1, 0 } };
    const Case small[] = { { 4096, 5120, "2.5", "mad", 0, 0 }, { 2048, 1024, "7", "mad", 0, 1 }, { 4096, 2048, "2.5", "mad", 1, 0 },
                           { 4096, 1024, "0.5", "mad", 1, 0 }, { 2048, 1024, "6.5", "mad", 1, 0 } };
    if (!small_only)
        for (const Case& c : big) run(c, quick ? std::vector<int>{ 7 } : std::vector<int>{ 1, 7, 16 }, true);
    for (const Case& c : small) run(c, { 1, 7 }, false);
    std::printf("%s (%d failures)\n", FAILS ? "TEST_RERANK FAIL" : "TEST_RERANK PASS", FAILS);
    return FAILS ? 1 : 0;
}
