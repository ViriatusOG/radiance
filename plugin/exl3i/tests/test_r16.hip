// test_r16.hip -- the R16 row order (exl3i_encoding.h version 3, the head): the layout hook gives an R16 encoding's
// record the dtype EXL3I_DT_TILES_R16, and the launch writes output s at row exl3i_r16_row(s, N). Usage:
//   test_r16 EXL3I.so
// Per (N, K, rate, codebook, T, y dtype): the same random record launched as a natural weight and as an R16 one
// (the hooks of both encodings; the bytes are equal): y_r16[:, exl3i_r16_row(s, N)] must equal y_nat[:, s] byte for byte
// for every s and every row, through every path the f16 row takes (T 1 the decode form's DOT + KS, T 4 / 16 KS + DQ,
// T 17 MT 4, T 600 T2). Plus: a mad R16 record at a8g128 takes the byte-sum GEMV at T 1 / 3 / 8 (natural vs R16), and the
// f16 path above (the A8 path has no R16 store: an R16 weight never takes it, at any T). Last line PASS / FAIL.
#include <hip/hip_runtime.h>
#include "exl3i.h"
#include "exl3i_decode.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#define HIPCHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

static uint64_t RNG = 0x5DEECE66Dull;
static uint32_t rnd() { RNG ^= RNG << 13; RNG ^= RNG >> 7; RNG ^= RNG << 17; return (uint32_t)(RNG >> 16); }

static const RadKernelInfo *RF = nullptr, *RA = nullptr;
static int FAILS = 0;

static RadTensor tensor(void* d, uint32_t dt, int64_t r0, int64_t r1) {
    RadTensor t{};
    t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r0; t.shape[1] = r1; t.stride[0] = r1; t.stride[1] = 1;
    return t;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_r16 EXL3I.so\n"); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto inf = (const RadPluginInfo* (*)())dlsym(h, "rad_plugin_info");
    auto cnt = (int (*)())dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    for (int i = 0; i < cnt(); ++i) {
        if (!std::strcmp(at(i)->name, "exl3i_gemm_nt")) RF = at(i);
        if (!std::strcmp(at(i)->name, "exl3i_gemm_nt_a8")) RA = at(i);
    }
    if (!RF) { std::fprintf(stderr, "no exl3i_gemm_nt\n"); return 2; }
    std::printf("exl3i %s -- test_r16\n", inf()->version);
    struct Case { int N, K; const char* rate; const char* cb; };
    // a head-shaped N (rows / 8 a multiple of 16), the trunk's K; mad at K4, K2.5, K7 (the head) and K6.5
    const Case cases[] = { { 4096, 5120, "4", "mad" }, { 4096, 5120, "2.5", "mad" }, { 2048, 5120, "7", "mad" },
                           { 248320, 1024, "6.5", "mad" }, { 4096, 5120, "2.5", "mad" }, { 248320, 1024, "4.125", "mad" } };
    int n = 0;
    for (const Case& c : cases) {
        const int m = exl3i_rate_parse(c.rate), cb = exl3i_codebook_from_name(c.cb);
        RadEncoding en, er;
        exl3i_encoding_make(m, cb, 0, c.N, c.K, &en);
        if (exl3i_encoding_make_r16(m, cb, 0, c.N, c.K, &er) != RAD_OK) { std::printf("make_r16 refused\n"); ++FAILS; continue; }
        int r16 = -1;
        if (exl3i_encoding_read_r16(&er, nullptr, nullptr, nullptr, nullptr, nullptr, &r16) != RAD_OK || r16 != 1) {
            std::printf("read_r16 of %s failed\n", er.transform); ++FAILS;
        }
        const int W = exl3i_row_words(c.K, m);
        std::vector<uint32_t> rec((size_t)(c.N / 16) * W);
        for (auto& x : rec) x = rnd() ^ (rnd() << 16);
        auto params = [&](int64_t M, const char* act) {
            return std::vector<RadParam>{ RAD_INT("M", M), RAD_INT("N", c.N), RAD_INT("K", c.K), RAD_STR("rate", c.rate),
                                          RAD_STR("codebook", c.cb), RAD_STR("act", act) };
        };
        auto pa = params(16, "f16");
        RadTensor pl = tensor(rec.data(), RAD_U32, c.N / 16, W);
        RadLayout Ln, Lr;
        int sel = EXL3I_PLANE_RECORD;
        if (RF->layout(pa.data(), 6, EXL3I_OPD_RECORD, &en, &sel, &pl, 1, &Ln) != RAD_OK ||
            RF->layout(pa.data(), 6, EXL3I_OPD_RECORD, &er, &sel, &pl, 1, &Lr) != RAD_OK) { std::printf("layout refused\n"); return 1; }
        if (Ln.dtype != EXL3I_DT_TILES || Lr.dtype != EXL3I_DT_TILES_R16 || Ln.bytes != Lr.bytes) {
            std::printf("layout dtypes %u / %u (want %d / %d), bytes %lld / %lld\n", Ln.dtype, Lr.dtype, EXL3I_DT_TILES,
                        EXL3I_DT_TILES_R16, (long long)Ln.bytes, (long long)Lr.bytes);
            ++FAILS;
        }
        std::vector<char> bn((size_t)Ln.bytes), br((size_t)Lr.bytes);
        RF->relayout(pa.data(), 6, EXL3I_OPD_RECORD, &en, &sel, &pl, 1, bn.data(), Ln.bytes);
        RF->relayout(pa.data(), 6, EXL3I_OPD_RECORD, &er, &sel, &pl, 1, br.data(), Lr.bytes);
        if (bn != br) { std::printf("relayout bytes differ between the encodings\n"); ++FAILS; }
        void *drec, *suh, *svh, *x, *y;
        const int Mmax = 600;
        HIPCHECK(hipMalloc(&drec, bn.size()));
        HIPCHECK(hipMemcpy(drec, bn.data(), bn.size(), hipMemcpyHostToDevice));
        auto vec = [](int k) {
            std::vector<uint16_t> v((size_t)k);
            for (auto& q : v) { _Float16 hh = (_Float16)((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f)); std::memcpy(&q, &hh, 2); }
            void* p; HIPCHECK(hipMalloc(&p, (size_t)k * 2)); HIPCHECK(hipMemcpy(p, v.data(), (size_t)k * 2, hipMemcpyHostToDevice));
            return p;
        };
        suh = vec(c.K); svh = vec(c.N);
        HIPCHECK(hipMalloc(&x, (size_t)Mmax * c.K * 2));
        HIPCHECK(hipMalloc(&y, (size_t)Mmax * c.N * 4));
        {
            std::vector<uint16_t> hx((size_t)Mmax * c.K);
            for (auto& v : hx) { float f = ((int)(rnd() & 0xFFFF) - 32768) / 16384.0f; uint32_t u; std::memcpy(&u, &f, 4); v = (uint16_t)(u >> 16); }
            HIPCHECK(hipMemcpy(x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
        }
        auto run = [&](const RadKernelInfo* row, int T, uint32_t ydt, uint32_t recdt, const char* act, int* rc) {
            auto p = params(T, act);
            RadTensor t[6]{};
            t[0] = tensor(x, RAD_BF16, T, c.K);
            t[1] = tensor(drec, recdt, c.N / 16, W);
            t[2] = tensor(suh, RAD_F16, 1, c.K);
            t[3] = tensor(svh, RAD_F16, c.N, 1);
            t[5] = tensor(y, ydt, T, c.N);
            RadArgs a{};
            a.t = t; a.n_t = 6; a.p = p.data(); a.n_p = 6; a.world_size = 1;
            const int64_t sb = row->scratch(&a);
            void* sc = nullptr;
            if (sb > 0) HIPCHECK(hipMalloc(&sc, (size_t)sb));
            a.scratch = sc; a.scratch_bytes = sb;
            const size_t es = ydt == RAD_F32 ? 4 : 2;
            HIPCHECK(hipMemset(y, 0xFF, (size_t)T * c.N * es));
            *rc = row->launch(&a, nullptr);
            HIPCHECK(hipDeviceSynchronize());
            std::vector<char> o((size_t)T * c.N * es);
            HIPCHECK(hipMemcpy(o.data(), y, o.size(), hipMemcpyDeviceToHost));
            if (sc) HIPCHECK(hipFree(sc));
            return o;
        };
        const bool mad = cb == EXL3I_CB_MAD;
        for (int ai = 0; ai < (mad && RA ? 2 : 1); ++ai)
        for (int T : { 1, 3, 4, 8, 16, 17, 600 })
            for (uint32_t ydt : { (uint32_t)RAD_F32, (uint32_t)RAD_BF16 }) {
                if (!mad && (T == 3 || T == 8)) continue;
                const char* act = ai ? "a8g128" : "f16";
                int rc1, rc2;
                // a8g128 at 600 rows: the natural record takes the A8 path, the R16 one the f16 path -> compare with f16's
                const auto yn = run(ai && T <= 16 ? RA : RF, T, ydt, EXL3I_DT_TILES, ai && T <= 16 ? act : "f16", &rc1);
                const auto yr = run(ai ? RA : RF, T, ydt, EXL3I_DT_TILES_R16, act, &rc2);
                const size_t es = ydt == RAD_F32 ? 4 : 2;
                long bad = 0;
                if (rc1 != RAD_OK || rc2 != RAD_OK) { std::printf("launch rc %d / %d\n", rc1, rc2); ++FAILS; continue; }
                for (int r = 0; r < T; ++r)
                    for (int64_t s = 0; s < c.N; ++s)
                        bad += std::memcmp(yn.data() + ((size_t)r * c.N + s) * es, yr.data() + ((size_t)r * c.N + exl3i_r16_row(s, c.N)) * es, es) != 0;
                std::printf("N %6d K %5d K%-6s %-7s %-6s T %4d y %-4s: %ld of %lld outputs not at their R16 row%s\n", c.N, c.K,
                            c.rate, c.cb, act, T, ydt == RAD_F32 ? "f32" : "bf16", bad, (long long)T * c.N, bad ? "  FAIL" : "");
                FAILS += bad != 0;
                ++n;
            }
        HIPCHECK(hipFree(drec)); HIPCHECK(hipFree(suh)); HIPCHECK(hipFree(svh)); HIPCHECK(hipFree(x)); HIPCHECK(hipFree(y));
    }
    std::printf("%d comparisons\n%s (%d failures)\n", n, FAILS ? "TEST_R16 FAIL" : "TEST_R16 PASS", FAILS);
    return FAILS ? 1 : 0;
}
