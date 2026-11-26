// bench_twopass.hip -- the two-pass prefill's timing driver at Qwen3.8-27B's trunk shapes: the exl3i op
// through its plugin ABI, f16 row against A8 row, and the A8 path's three passes alone. Usage:
//   bench_twopass EXL3I.so [--rate 4] [--codebook mad] [--reps 20] [--shapes gate_up,down,...,head]
//                          [--m 2048[,M2...] | --sweep] [--modes f16,a8,a8x,phases,a8q] [--csv FILE]
//                          [--cold]
// Modes (each a row of the output):
//   f16     row exl3i_gemm_nt, act f16: nt_pass below 512 rows, T2 (two-pass f16 plane) from 512
//   a8      row exl3i_gemm_nt_a8, act a8g128: the A8 path from 512 rows, nt_pass below (what serving runs)
//   a8x     the same with the A8_MIN_ROWS pin at 16: the A8 path at every M >= 16 (is the 512 threshold right?)
//   a8q     codebook mad at M 1..8: the a8 row reading the producer's rows (AQ form xb, written by exl3i_actq once
//           before the timing, as serving's producer replaces its norm / silu_mul / mul): the GEMV alone
//   phases  the A8 path's passes timed alone (this binary's own copies of exl3i_a8_prep.hip / exl3i_a8_gemm.hip,
//           the plugin's sources): actq, wdec8 (the whole weight, one chunk where it fits), gemm
// Weights are random trellis codes at --rate in --codebook (mad; every value is a
// valid code), a and y bf16 (the graph's). mad's a8 row at M 1..8 is the byte-sum GEMV (exl3i_mad.hip.h). --cold: each
// launch reads its own copy of the record, enough copies for 256 MB (the R9700's 64 MB of L2 + MALL never holds the
// weight a launch reads, as in a step that streams every layer past it). The A8 and f16 rows take either.
// Each row also prints an FNV-1a digest of y after its last launch (codes and inputs are seeded: two builds whose outputs
// agree bit for bit print the same digest). Each timing is the median of --reps launches (events around each, after 3
// warm-ups), with the minimum. The summary at
// M 2048 sums a decoder layer: GDN (gdn_qkv, gdn_z, ssm_out, gate_up, down; the a/b projections are bf16 gemm_nt, not
// here) and attention (attn_qg, attn_k, attn_v, attn_o, gate_up, down), and the 48 : 16 mean. Default sweep:
// M 16 32 64 128 256 384 511 512 768 1024 1536 2048 3072 4096.
#include <hip/hip_runtime.h>
#include "exl3i.h"
#include "exl3i_a8.h"
#include "exl3i_a8_dev.h"
#include "exl3i_pin.h"
#include "exl3i_decode.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <string>
#include <vector>

#define HIPCHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

struct Shape { const char* name; int N, K; bool stacked; };
static const Shape SHAPES[] = {
    { "gate_up", 34816, 5120, true }, { "down", 5120, 17408, false }, { "gdn_qkv", 10240, 5120, false },
    { "gdn_z", 6144, 5120, false }, { "attn_qg", 12288, 5120, false }, { "attn_k", 1024, 5120, false },
    { "attn_v", 1024, 5120, false }, { "attn_o", 5120, 6144, false }, { "ssm_out", 5120, 6144, false },
    { "head", 248320, 5120, false },   // the LM head (row order aside); only when --shapes names it
};

static uint64_t RNG = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() { RNG ^= RNG << 13; RNG ^= RNG >> 7; RNG ^= RNG << 17; return (uint32_t)(RNG >> 16); }

static const RadKernelInfo *RF = nullptr, *RA = nullptr, *RQ = nullptr;
static std::string CB = "mad";
static const RadPluginInfo* INFO = nullptr;
static void (*PIN)(int, int64_t) = nullptr;   // the plugin's exl3i_test_pin (exl3i_pin.h)
static void test_pin(int k, int64_t v) { PIN(k, v); }

static void open_plugin(const char* path) {
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); std::exit(2); }
    auto ver = (uint32_t (*)())dlsym(h, "rad_plugin_abi_version");
    auto inf = (const RadPluginInfo* (*)())dlsym(h, "rad_plugin_info");
    auto cnt = (int (*)())dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!ver || !inf || !cnt || !at || ver() != RAD_ABI_VERSION) { std::fprintf(stderr, "%s: not an ABI %d plugin\n", path, RAD_ABI_VERSION); std::exit(2); }
    INFO = inf();
    for (int i = 0; i < cnt(); ++i) {
        if (!std::strcmp(at(i)->name, "exl3i_gemm_nt")) RF = at(i);
        if (!std::strcmp(at(i)->name, "exl3i_gemm_nt_a8")) RA = at(i);
        if (!std::strcmp(at(i)->name, "exl3i_actq")) RQ = at(i);
    }
    if (!RF || !RA) { std::fprintf(stderr, "%s: rows exl3i_gemm_nt and exl3i_gemm_nt_a8 wanted\n", path); std::exit(2); }
    PIN = (void (*)(int, int64_t))dlsym(h, "exl3i_test_pin");
    if (!PIN) { std::fprintf(stderr, "%s: no exl3i_test_pin\n", path); std::exit(2); }
}

static RadTensor tensor(void* d, uint32_t dt, int64_t r0, int64_t r1) {
    RadTensor t{};
    t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r0; t.shape[1] = r1; t.stride[0] = r1; t.stride[1] = 1;
    return t;
}

struct Params {
    char rate[16], act[16];
    std::vector<RadParam> p;
    Params(int64_t M, int N, int K, const char* rt, const char* ac) {
        std::snprintf(rate, sizeof rate, "%s", rt); std::snprintf(act, sizeof act, "%s", ac);
        p = { RAD_INT("M", M), RAD_INT("N", N), RAD_INT("K", K), RAD_STR("rate", rate), RAD_STR("codebook", CB.c_str()),
              RAD_STR("act", act) };
    }
};

struct DevW { void *rec = nullptr, *suh = nullptr, *suh1 = nullptr, *svh = nullptr; std::vector<void*> cold; };
static bool COLD = false;
static int BURST = 0;   // --burst N: the time of N back-to-back launches / N instead of each alone

static DevW store(const Shape& s, int m) {
    DevW d;
    RadEncoding e;
    exl3i_encoding_make(m, exl3i_codebook_from_name(CB.c_str()), s.stacked, s.N, s.K, &e);
    std::vector<uint32_t> rec((size_t)(s.N / 16) * exl3i_row_words(s.K, m));
    for (auto& x : rec) x = rnd() ^ (rnd() << 16);
    char rt[16];
    Params pa(2048, s.N, s.K, exl3i_rate_str(m, rt), "f16");
    RadTensor pl = tensor(rec.data(), RAD_U32, s.N / 16, exl3i_row_words(s.K, m));
    RadLayout L;
    int sel = EXL3I_PLANE_RECORD;
    if (RF->layout(pa.p.data(), 6, EXL3I_OPD_RECORD, &e, &sel, &pl, 1, &L) != RAD_OK) { std::fprintf(stderr, "layout refused\n"); std::exit(2); }
    std::vector<char> buf((size_t)L.bytes);
    RF->relayout(pa.p.data(), 6, EXL3I_OPD_RECORD, &e, &sel, &pl, 1, buf.data(), L.bytes);
    HIPCHECK(hipMalloc(&d.rec, buf.size()));
    HIPCHECK(hipMemcpy(d.rec, buf.data(), buf.size(), hipMemcpyHostToDevice));
    d.cold.push_back(d.rec);
    if (COLD)
        for (size_t tot = buf.size(); tot < (256u << 20); tot += buf.size()) {
            void* c; HIPCHECK(hipMalloc(&c, buf.size()));
            HIPCHECK(hipMemcpy(c, d.rec, buf.size(), hipMemcpyDeviceToDevice));
            d.cold.push_back(c);
        }
    auto vec = [](int n) {
        std::vector<uint16_t> v((size_t)n);
        for (auto& x : v) { _Float16 h = (_Float16)((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f)); std::memcpy(&x, &h, 2); }
        void* p; HIPCHECK(hipMalloc(&p, (size_t)n * 2)); HIPCHECK(hipMemcpy(p, v.data(), (size_t)n * 2, hipMemcpyHostToDevice));
        return p;
    };
    d.suh = vec(s.K); d.svh = vec(s.N);
    if (s.stacked) d.suh1 = vec(s.K);
    return d;
}

struct Stat { double med = 0, min = 0; int rc = 0; };

template <class F>
static Stat time_it(F&& f, int reps) {
    hipEvent_t a, b;
    HIPCHECK(hipEventCreate(&a)); HIPCHECK(hipEventCreate(&b));
    Stat s;
    for (int i = 0; i < 3; ++i) s.rc |= f();
    HIPCHECK(hipDeviceSynchronize());
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        HIPCHECK(hipEventRecord(a, 0));
        s.rc |= f();
        for (int j = 1; j < BURST; ++j) s.rc |= f();
        HIPCHECK(hipEventRecord(b, 0));
        HIPCHECK(hipEventSynchronize(b));
        float ms; HIPCHECK(hipEventElapsedTime(&ms, a, b));
        t.push_back(ms * 1000.0 / (BURST > 1 ? BURST : 1));
    }
    std::sort(t.begin(), t.end());
    s.med = t[t.size() / 2]; s.min = t[0];
    HIPCHECK(hipEventDestroy(a)); HIPCHECK(hipEventDestroy(b));
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: bench_twopass EXL3I.so [--rate 4] [--reps 20] [--shapes a,b] [--m M[,M...] | --sweep] [--modes f16,a8,a8x,phases] [--csv FILE]\n"); return 2; }
    std::string rate = "4", shapes = "", modes = "f16,a8,a8x,phases", csvp = "";
    int reps = 20;
    std::vector<int> Ms = { 2048 };
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "%s wants a value\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--rate") rate = val();
        else if (a == "--codebook") CB = val();
        else if (a == "--reps") reps = std::atoi(val().c_str());
        else if (a == "--shapes") shapes = val();
        else if (a == "--modes") modes = val();
        else if (a == "--csv") csvp = val();
        else if (a == "--cold") COLD = true;
        else if (a == "--burst") BURST = std::atoi(val().c_str());
        else if (a == "--m") {   // one M or a list: 1,4,16
            const std::string v = val();
            Ms.clear();
            for (size_t p = 0; p <= v.size();) {
                size_t q = v.find(',', p);
                if (q == std::string::npos) q = v.size();
                Ms.push_back(std::atoi(v.substr(p, q - p).c_str()));
                p = q + 1;
            }
        }
        else if (a == "--sweep") Ms = { 16, 32, 64, 128, 256, 384, 511, 512, 768, 1024, 1536, 2048, 3072, 4096 };
        else { std::fprintf(stderr, "unknown %s\n", a.c_str()); return 2; }
    }
    const int m = exl3i_rate_parse(rate.c_str());
    if (m < 0) { std::fprintf(stderr, "rate %s: not an exl3i rate\n", rate.c_str()); return 2; }
    auto has = [](const std::string& list, const char* x) { return list.empty() || ("," + list + ",").find(std::string(",") + x + ",") != std::string::npos; };
    open_plugin(argv[1]);
    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    const int cbi = exl3i_codebook_from_name(CB.c_str());
    if (cbi < 0) { std::fprintf(stderr, "codebook %s: not an exl3i codebook\n", CB.c_str()); return 2; }
    std::printf("exl3i %s -- %s; device %s %s; rate K%s %s, a / y bf16, %d reps\n", INFO->version, INFO->description, prop.name,
                prop.gcnArchName, rate.c_str(), CB.c_str(), reps);
    FILE* csv = csvp.empty() ? nullptr : std::fopen(csvp.c_str(), "w");
    if (csv) std::fprintf(csv, "shape,N,K,M,mode,us_median,us_min,tflops,rc,y_fnv\n");
    std::printf("%-8s %6s %6s %5s  %-7s %10s %10s %7s\n", "shape", "N", "K", "M", "mode", "us_med", "us_min", "TFLOP/s");
    const int Mmax = *std::max_element(Ms.begin(), Ms.end());
    std::map<std::string, std::map<int, std::map<std::string, double>>> T;   // shape -> M -> mode -> us
    for (const Shape& s : SHAPES) {
        if (!has(shapes, s.name) || (shapes.empty() && !std::strcmp(s.name, "head"))) continue;
        DevW d = store(s, m);
        void *x, *y;
        HIPCHECK(hipMalloc(&x, (size_t)Mmax * s.K * 2));
        HIPCHECK(hipMalloc(&y, (size_t)Mmax * s.N * 2));
        {   // x: bf16 ~N(0, 1)-ish bytes (any finite values do for timing)
            std::vector<uint16_t> h((size_t)Mmax * s.K);
            for (auto& v : h) { float f = ((int)(rnd() & 0xFFFF) - 32768) / 16384.0f; uint32_t u; std::memcpy(&u, &f, 4); v = (uint16_t)(u >> 16); }
            HIPCHECK(hipMemcpy(x, h.data(), h.size() * 2, hipMemcpyHostToDevice));
        }
        char rt[16];
        exl3i_rate_str(m, rt);
        for (int M : Ms) {
            const double flop = 2.0 * M * s.N * s.K;
            auto report = [&](const char* mode, const Stat& st) {
                const double tf = flop / (st.med * 1e-6) / 1e12;
                // y's bytes after the last launch (FNV-1a 64): the same inputs give the same digest on every build whose
                // output bits agree (a kernel change that keeps the bits keeps it)
                std::vector<uint8_t> yb((size_t)M * s.N * 2);
                HIPCHECK(hipMemcpy(yb.data(), y, yb.size(), hipMemcpyDeviceToHost));
                uint64_t fh = 0xcbf29ce484222325ull;
                for (uint8_t c : yb) fh = (fh ^ c) * 0x100000001b3ull;
                std::printf("%-8s %6d %6d %5d  %-7s %10.1f %10.1f %7.1f  y %016llx%s\n", s.name, s.N, s.K, M, mode, st.med, st.min, tf,
                            (unsigned long long)fh, st.rc ? "  (launch failed)" : "");
                if (csv) std::fprintf(csv, "%s,%d,%d,%d,%s,%.2f,%.2f,%.2f,%d,%016llx\n", s.name, s.N, s.K, M, mode, st.med, st.min, tf, st.rc,
                                      (unsigned long long)fh);
                T[s.name][M][mode] = st.med;
            };
            auto op = [&](const RadKernelInfo* row, const char* act) {
                Params pa(M, s.N, s.K, rt, act);
                RadTensor t[6]{};
                t[0] = tensor(x, RAD_BF16, M, s.K);
                t[1] = tensor(d.rec, EXL3I_DT_TILES, s.N / 16, exl3i_row_words(s.K, m));
                t[2] = tensor(d.suh, RAD_F16, 1, s.K);
                t[3] = tensor(d.svh, RAD_F16, s.N, 1);
                t[4] = tensor(d.suh1, RAD_F16, 1, s.K);
                t[5] = tensor(y, RAD_BF16, M, s.N);
                RadArgs a{};
                a.t = t; a.n_t = 6; a.p = pa.p.data(); a.n_p = 6; a.world_size = 1;
                const int64_t sb = row->scratch(&a);
                void* sc = nullptr;
                if (sb > 0) HIPCHECK(hipMalloc(&sc, (size_t)sb));
                a.scratch = sc; a.scratch_bytes = sb;
                size_t ci = 0;
                const Stat st = time_it([&]() { t[1].data = d.cold[ci++ % d.cold.size()]; return row->launch(&a, nullptr) != RAD_OK; }, reps);
                t[1].data = d.rec;
                if (sc) HIPCHECK(hipFree(sc));
                return st;
            };
            if (has(modes, "f16")) report("f16", op(RF, "f16"));
            if (has(modes, "a8q") && cbi == EXL3I_CB_MAD && M <= EXL3I_AQ_XB_MAX_ROWS && RQ) {
                // the producer's rows (form xb) for this weight's planes (suh, suh.1), then the dense op reading them
                const int np = s.stacked ? 2 : 1;
                const int64_t ld = exl3i_aq_xb_row_bytes(s.K, np);
                void* aqb; HIPCHECK(hipMalloc(&aqb, (size_t)M * ld));
                {
                    RadParam pq[4] = { RAD_INT("M", M), RAD_INT("K", s.K), RAD_INT("planes", np), RAD_STR("form", "xb") };
                    RadTensor t[EXL3I_AQ_N_OPD]{};
                    t[0] = tensor(x, RAD_BF16, M, s.K);
                    t[1] = tensor(d.suh, RAD_F16, 1, s.K);
                    t[2] = tensor(s.stacked ? d.suh1 : d.suh, RAD_F16, 1, s.K);
                    t[5] = tensor(aqb, RAD_U8, M, ld);
                    RadArgs a{};
                    a.p = pq; a.n_p = 4; a.t = t; a.n_t = EXL3I_AQ_N_OPD; a.world_size = 1;
                    if (RQ->launch(&a, nullptr) != RAD_OK) std::fprintf(stderr, "actq xb failed\n");
                    HIPCHECK(hipDeviceSynchronize());
                }
                Params pa(M, s.N, s.K, rt, "a8g128");
                pa.p.push_back(RAD_INT("aq_plane", 0));
                pa.p.push_back(RAD_INT("aq_planes", np));
                RadTensor t[7]{};
                t[0] = tensor(x, RAD_BF16, M, s.K);
                t[1] = tensor(d.rec, EXL3I_DT_TILES, s.N / 16, exl3i_row_words(s.K, m));
                t[2] = tensor(d.suh, RAD_F16, 1, s.K);
                t[3] = tensor(d.svh, RAD_F16, s.N, 1);
                t[4] = tensor(d.suh1, RAD_F16, 1, s.K);
                t[5] = tensor(y, RAD_BF16, M, s.N);
                t[6] = tensor(aqb, RAD_U8, M, ld);
                RadArgs a{};
                a.t = t; a.n_t = 7; a.p = pa.p.data(); a.n_p = (int)pa.p.size(); a.world_size = 1;
                const int64_t sb = RA->scratch(&a);
                void* sc = nullptr;
                if (sb > 0) HIPCHECK(hipMalloc(&sc, (size_t)sb));
                a.scratch = sc; a.scratch_bytes = sb;
                size_t ci = 0;
                const Stat st = time_it([&]() { t[1].data = d.cold[ci++ % d.cold.size()]; return RA->launch(&a, nullptr) != RAD_OK; }, reps);
                t[1].data = d.rec;
                if (sc) HIPCHECK(hipFree(sc));
                HIPCHECK(hipFree(aqb));
                report("a8q", st);
            }
            if (has(modes, "a8")) report("a8", op(RA, "a8g128"));
            if (has(modes, "a8x")) {
                test_pin(EXL3I_PIN_A8_MIN_ROWS, 16);
                if (M >= 16) report("a8x", op(RA, "a8g128"));
                test_pin(EXL3I_PIN_A8_MIN_ROWS, 0);
            }
            if (has(modes, "phases")) {
                Exl3iA8Plan pl;
                if (exl3i_a8_plan(M, s.N, s.K, s.stacked, 128, EXL3I_A8_CHUNK_BYTES, 1, &pl) == 0) {
                    void* ar; HIPCHECK(hipMalloc(&ar, (size_t)pl.total));
                    Exl3iA8Args q{};
                    q.x = x; q.codes = d.rec; q.suh = d.suh; q.suh1 = d.suh1; q.svh = d.svh; q.y = y;
                    q.aq = (int8_t*)((char*)ar + pl.aq_off); q.sa = (float*)((char*)ar + pl.sa_off); q.wq = (int8_t*)((char*)ar + pl.w_off);
                    q.x_pitch = s.K; q.y_pitch = s.N; q.M = M; q.N = s.N; q.K = s.K; q.m = m; q.group = 128; q.mats = pl.mats;
                    q.x_bf16 = 1; q.y_dt = 0; q.c0 = 0; q.nc = (int)std::min<int64_t>(pl.chunk, s.N); q.cb = cbi;
                    report("actq", time_it([&]() { return exl3i_a8_actq(&q, nullptr) != RAD_OK; }, reps));
                    report("wdec8", time_it([&]() {   // every chunk, as a launch decodes them
                        int rc = 0;
                        for (int c0 = 0; c0 < s.N; c0 += (int)pl.chunk) { Exl3iA8Args c = q; c.c0 = c0; c.nc = (int)std::min<int64_t>(pl.chunk, s.N - c0); rc |= exl3i_a8_wdec8(&c, nullptr); }
                        return rc != RAD_OK; }, reps));
                    report("gemm", time_it([&]() {
                        int rc = 0;
                        for (int c0 = 0; c0 < s.N; c0 += (int)pl.chunk) { Exl3iA8Args c = q; c.c0 = c0; c.nc = (int)std::min<int64_t>(pl.chunk, s.N - c0); rc |= exl3i_a8_gemm(&c, nullptr); }
                        return rc != RAD_OK; }, reps));
                    HIPCHECK(hipFree(ar));
                }
            }
        }
        HIPCHECK(hipFree(x)); HIPCHECK(hipFree(y));
        for (size_t i = 1; i < d.cold.size(); ++i) (void)hipFree(d.cold[i]);
        hipFree(d.rec); hipFree(d.suh); hipFree(d.svh); if (d.suh1) hipFree(d.suh1);
    }
    // the decoder layer at each M every shape was timed at
    const char* gdn[] = { "gdn_qkv", "gdn_z", "ssm_out", "gate_up", "down" };
    const char* att[] = { "attn_qg", "attn_k", "attn_v", "attn_o", "gate_up", "down" };
    for (int M : Ms)
        for (const char* mode : { "f16", "a8", "a8q", "a8x", "actq", "wdec8", "gemm" }) {
            double g = 0, a = 0;
            bool all = true;
            for (const char* n : gdn) { auto it = T[n][M].find(mode); if (it == T[n][M].end()) all = false; else g += it->second; }
            for (const char* n : att) { auto it = T[n][M].find(mode); if (it == T[n][M].end()) all = false; else a += it->second; }
            if (all)
                std::printf("layer M %5d %-4s: GDN %8.1f us, attention %8.1f us, mean (48 GDN : 16 attention) %8.1f us\n", M, mode, g, a,
                            (48 * g + 16 * a) / 64);
        }
    if (csv) std::fclose(csv);
    return 0;
}
