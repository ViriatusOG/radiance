/* embed_exl3i_gpu_test.hip -- the device row of embed_exl3i against its host row, bit for bit, then its time at the 27B's
 * token table (n_vocab 248320, n_embd 5120) beside embed_tq's (the Lloyd table it replaces), on one card.
 *
 *   embed_exl3i_gpu_test [--bench-only] [--ids N] [--tokens FILE.i32] [--host]
 *
 * Check: random tables (every codebook; m 2 / 8 / 36 / 40 / 64 / 104 / 112 / 128; natural and R16 rows), every token,
 * then 300 repeats of earlier ids and out-of-range ids (the repeated-token path), device x == host x in every bit.
 * Bench: random records (the decode does not depend on the values), M = 1 / 16 / 128 / 1024 / 8192 tokens drawn
 * uniformly from [0, ids) (default the whole table: no id repeats and no block shared, the worst case), or the first
 * 8192 ids of a token file (--tokens: real text, e.g. a golden's tokens.i32), mean of 50 launches after 5 warm-up,
 * hipEvent timing. --host: the exl3i tables (record, suh, svh) in mapped pinned host memory, read in place over PCIe
 * (the host-RAM embedding case), embed_tq not run. */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "embed_exl3i.h"
#include "embed_tq.h"
#include "exl3_decode.h"
#include "exl3i_encoding.h"
#include "rad_plugin.h"

#define HIPCHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

namespace {

int failures = 0;

RadTensor tensor(void* data, uint32_t dt, std::initializer_list<int64_t> shape) {
    RadTensor t{};
    t.data = data;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int i = 0;
    for (int64_t s : shape) t.shape[i++] = s;
    int64_t acc = 1;
    for (int k = (int)t.rank - 1; k >= 0; --k) { t.stride[k] = acc; acc *= t.shape[k]; }
    return t;
}

bool g_host = false;

/* the table's copy the kernel reads: device memory, or (--host) mapped pinned host memory and its device alias */
template <class T> T* tab_copy(const std::vector<T>& h) {
    if (!g_host) {
        T* d = nullptr;
        HIPCHECK(hipMalloc(&d, h.size() * sizeof(T) + 16));
        HIPCHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
        return d;
    }
    T* p = nullptr;
    HIPCHECK(hipHostMalloc((void**)&p, h.size() * sizeof(T) + 16, hipHostMallocMapped));
    std::memcpy(p, h.data(), h.size() * sizeof(T));
    void* d = nullptr;
    HIPCHECK(hipHostGetDevicePointer(&d, p, 0));
    return (T*)d;
}

template <class T> void tab_free(T* d) {
    if (!g_host) { (void)hipFree(d); return; }
    void* p = nullptr;
    hipPointerAttribute_t at{};
    if (hipPointerGetAttributes(&at, d) == hipSuccess) p = at.hostPointer;
    (void)hipHostFree(p ? p : (void*)d);
}

template <class T> T* dev_copy(const std::vector<T>& h) {
    T* d = nullptr;
    HIPCHECK(hipMalloc(&d, h.size() * sizeof(T) + 16));
    HIPCHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

struct Table {
    int64_t nv, ne;
    int m, cb, r16;
    std::vector<uint32_t> rec;
    std::vector<uint16_t> suh, svh;
};

Table make(int64_t nv, int64_t ne, int m, int cb, int r16, unsigned seed) {
    std::mt19937 rng(seed);
    Table t{nv, ne, m, cb, r16, {}, {}, {}};
    t.rec.resize((size_t)(nv / 16 * exl3i_row_words(ne, m)));
    for (auto& w : t.rec) w = rng();
    std::uniform_real_distribution<double> mag(0.25, 2.0);
    auto sc = [&](int64_t n) {
        std::vector<uint16_t> v((size_t)n);
        for (auto& h : v) h = exl3_half_from_double((rng() & 1 ? -1.0 : 1.0) * mag(rng));
        return v;
    };
    t.suh = sc(ne);
    t.svh = sc(nv);
    return t;
}

struct Bound {
    std::vector<RadTensor> t;
    std::vector<RadParam> p;
    RadArgs a{};
};

void bind(Bound& b, const Table& T, int32_t* tok, int64_t M, const uint32_t* rec, const uint16_t* suh, const uint16_t* svh,
          uint16_t* x) {
    b.p = { RAD_INT("M", M), RAD_INT("n_embd", T.ne), RAD_INT("n_vocab", T.nv), RAD_INT("m", T.m),
            RAD_STR("codebook", exl3i_codebook_name(T.cb)), RAD_INT("r16", T.r16) };
    b.t = { tensor(tok, RAD_I32, {M}), tensor((void*)rec, RAD_U32, {T.nv / 16, exl3i_row_words(T.ne, T.m)}),
            tensor((void*)suh, RAD_F16, {1, T.ne}), tensor((void*)svh, RAD_F16, {T.nv, 1}), tensor(x, RAD_BF16, {M, T.ne}) };
    b.a.t = b.t.data();
    b.a.n_t = (int)b.t.size();
    b.a.p = b.p.data();
    b.a.n_p = (int)b.p.size();
}

void check_one(int64_t nv, int64_t ne, int m, int cb, int r16, unsigned seed) {
    const Table T = make(nv, ne, m, cb, r16, seed);
    std::vector<int32_t> tok;
    for (int64_t v = 0; v < nv; ++v) tok.push_back((int32_t)v);
    tok.push_back(-1);
    tok.push_back((int32_t)nv);
    std::mt19937 rr(seed + 1000);
    for (int i = 0; i < 300; ++i) tok.push_back(i % 50 == 7 ? (int32_t)nv + 3 : tok[rr() % tok.size()]);
    const int64_t M = (int64_t)tok.size();
    std::vector<uint16_t> xh((size_t)(M * ne), 0xFFFF), xd((size_t)(M * ne), 0xEEEE);
    Bound hb;
    bind(hb, T, tok.data(), M, T.rec.data(), T.suh.data(), T.svh.data(), xh.data());
    if (embed_exl3i_host_launch(&hb.a, nullptr) != RAD_OK) { std::printf("FAIL host launch\n"); ++failures; return; }
    int32_t* dtok = dev_copy(tok);
    uint32_t* drec = dev_copy(T.rec);
    uint16_t* dsuh = dev_copy(T.suh);
    uint16_t* dsvh = dev_copy(T.svh);
    uint16_t* dx = dev_copy(xd);
    Bound db;
    bind(db, T, dtok, M, drec, dsuh, dsvh, dx);
    const int rc = embed_exl3i_device_launch(&db.a, nullptr);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(xd.data(), dx, xd.size() * 2, hipMemcpyDeviceToHost));
    int64_t diff = 0;
    for (size_t i = 0; i < xd.size(); ++i) diff += xd[i] != xh[i];
    /* the small-batch grid (M < 64: a workgroup a token and 128 columns) on the first 40 tokens */
    std::vector<uint16_t> xs((size_t)(40 * ne), 0xDDDD);
    uint16_t* dxs = dev_copy(xs);
    Bound sb;
    bind(sb, T, dtok, 40, drec, dsuh, dsvh, dxs);
    const int rc2 = embed_exl3i_device_launch(&sb.a, nullptr);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(xs.data(), dxs, xs.size() * 2, hipMemcpyDeviceToHost));
    int64_t diff2 = 0;
    for (size_t i = 0; i < xs.size(); ++i) diff2 += xs[i] != xh[i];
    (void)hipFree(dxs);
    std::printf("m=%3d %-6s r16=%d: device rc %d / %d, %lld of %lld (M %lld) and %lld of %lld (M 40) bf16 values differ "
                "from the host row\n", m, exl3i_codebook_name(cb), r16, rc, rc2, (long long)diff, (long long)xd.size(),
                (long long)M, (long long)diff2, (long long)xs.size());
    if (rc != RAD_OK || rc2 != RAD_OK || diff || diff2) ++failures;
    (void)hipFree(dtok); (void)hipFree(drec); (void)hipFree(dsuh); (void)hipFree(dsvh); (void)hipFree(dx);
}

double time_ms(int (*launch)(const RadArgs*, RadStream), const RadArgs* a, int reps) {
    hipEvent_t e0, e1;
    HIPCHECK(hipEventCreate(&e0));
    HIPCHECK(hipEventCreate(&e1));
    for (int i = 0; i < 5; ++i) if (launch(a, nullptr) != RAD_OK) return -1;
    HIPCHECK(hipEventRecord(e0, nullptr));
    for (int i = 0; i < reps; ++i) launch(a, nullptr);
    HIPCHECK(hipEventRecord(e1, nullptr));
    HIPCHECK(hipEventSynchronize(e1));
    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, e0, e1));
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
    return ms / reps;
}

void bench(int64_t ids, const char* tokfile) {
    const int64_t nv = 248320, ne = 5120;
    std::mt19937 rng(11);
    const int64_t Ms[] = { 1, 16, 128, 1024, 8192 };
    std::vector<int32_t> tok(8192);
    for (auto& t : tok) t = (int32_t)(rng() % (uint32_t)ids);
    if (tokfile) {
        FILE* f = std::fopen(tokfile, "rb");
        if (!f || std::fread(tok.data(), 4, tok.size(), f) != tok.size()) { std::printf("cannot read 8192 ids from %s\n", tokfile); std::exit(2); }
        std::fclose(f);
    }
    {
        std::vector<int32_t> u(tok);
        std::sort(u.begin(), u.end());
        std::printf("bench ids: %s, %lld distinct of 8192\n", tokfile ? tokfile : "uniform",
                    (long long)(std::unique(u.begin(), u.end()) - u.begin()));
    }
    int32_t* dtok = dev_copy(tok);
    uint16_t* dx = nullptr;
    HIPCHECK(hipMalloc(&dx, (size_t)(8192 * ne * 2)));
    /* embed_tq, Lloyd l3 (V2 3.00's): codes u3, a bf16 scale a 128, an 8-value table */
    if (!g_host) {
        const int bits = 3;
        const int64_t rb = (ne * bits + 7) / 8;
        std::vector<uint8_t> codes((size_t)(nv * rb));
        for (auto& c : codes) c = (uint8_t)rng();
        std::vector<uint16_t> scale((size_t)(nv * ne / 128), 0x3c00);
        std::vector<float> table(8);
        for (int i = 0; i < 8; ++i) table[i] = (float)i - 3.5f;
        uint8_t* dc = dev_copy(codes);
        uint16_t* ds = dev_copy(scale);
        float* dt = dev_copy(table);
        for (int64_t M : Ms) {
            std::vector<RadParam> p = { RAD_INT("M", M), RAD_INT("n_embd", ne), RAD_INT("n_vocab", nv), RAD_INT("bits", bits) };
            std::vector<RadTensor> t = { tensor(dtok, RAD_I32, {M}), tensor(dc, RAD_U3, {nv, ne}), tensor(ds, RAD_BF16, {nv, ne / 128}),
                                         tensor(dt, RAD_F32, {1, 8}), tensor(dx, RAD_BF16, {M, ne}) };
            RadArgs a{};
            a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size();
            const double ms = time_ms(embed_tq_device_launch, &a, 50);
            std::printf("bench embed_tq    l3          M=%5lld: %9.1f us  (%.3f us a token)\n", (long long)M, ms * 1e3, ms * 1e3 / M);
        }
        (void)hipFree(dc); (void)hipFree(ds); (void)hipFree(dt);
    }
    for (int cb : { EXL3I_CB_MAD })
        for (int m : { 32, 40, 64, 104, 112, 128 }) {
            uint32_t* drec;
            uint16_t *dsuh, *dsvh;
            Table T;
            {
                Table F = make(nv, ne, m, cb, 0, 3);
                drec = tab_copy(F.rec);
                dsuh = tab_copy(F.suh);
                dsvh = tab_copy(F.svh);
                T = Table{F.nv, F.ne, F.m, F.cb, F.r16, {}, {}, {}};
            }
            for (int64_t M : Ms) {
                Bound b;
                bind(b, T, dtok, M, drec, dsuh, dsvh, dx);
                const double ms = time_ms(embed_exl3i_device_launch, &b.a, 50);
                std::printf("bench embed_exl3i %-6s K%-4g M=%5lld: %9.1f us  (%.3f us a token)%s\n", exl3i_codebook_name(cb),
                            m / 16.0, (long long)M, ms * 1e3, ms * 1e3 / M, g_host ? "  [host]" : "");
            }
            tab_free(drec); tab_free(dsuh); tab_free(dsvh);
        }
    (void)hipFree(dtok);
    (void)hipFree(dx);
}

}  // namespace

int main(int argc, char** argv) {
    bool bench_only = false;
    int64_t ids = 248320;
    const char* tokfile = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench-only")) bench_only = true;
        else if (!std::strcmp(argv[i], "--ids") && i + 1 < argc) ids = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) tokfile = argv[++i];
        else if (!std::strcmp(argv[i], "--host")) g_host = true;
    }
    if (!bench_only) {
        unsigned seed = 1;
        for (int cb : { EXL3I_CB_MAD })
            for (int m : { 2, 8, 36, 40, 64, 104, 112, 128 })
                for (int r16 = 0; r16 < 2; ++r16) check_one(512, 256, m, cb, r16, seed++);
        std::printf(failures ? "%d FAILURES\n" : "embed_exl3i device == host: all bit-exact\n", failures);
    }
    bench(ids, tokfile);
    return failures ? 1 : 0;
}
