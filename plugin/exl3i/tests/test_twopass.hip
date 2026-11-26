// test_twopass.hip -- the two-pass prefill's device test (exl3i_a8.h): both rows through the plugin ABI (layout /
// relayout hooks, scratch hook, launch), the planes they leave in the op's scratch read back and compared with the C
// reference (tests/exl3i_twopass_ref.c) byte for byte, then the GEMMs against references. Usage:
//   test_twopass EXL3I.so [--quick] [--mad] [--nofuse]
// Sections (each prints its line; the last line is PASS or FAIL with the count, FAIL also when nothing was checked):
//   A  A8 planes, every rate m = 2..128 (even), mad, [256, 256], M 512: the int8 weight plane (wdec8), the int8
//      activations and their f32 scales (actq) byte for byte; y byte for byte against the device arithmetic's emulation
//      (exl3i_ref_a8_gemm_dev), against the quantised operands' exact GEMM (rel rms) and x . W^T (loose)
//   B  T2's planes (the f16 path), rates x both codebooks, M 512: the rotated rows (the region) against rotate128's
//      transcription and the f16 weight plane against the reference, byte for byte
//   C  the selftest matrix on the A8 row: real rows 1 / 15 / 511 / 512 / 513 / 952 / 1500 / 2048
//      in a band declared at 2048, y bf16 / f16 / f32, a f16 / bf16 with a padded pitch, the GDN ssm_out input
//      [T, 48, 128], the stacked seam, groups a8t / a8g256 / a8g512 / a8g1024, K2..K8 and fractional / sub-1 rates,
//      several chunks (pin A8_CHUNK_MB), two scratch sizes byte-identical, a replay byte-identical, finite outputs;
//      below 512 rows the A8 row's output byte-identical to the f16 row's
//   D  refusals through the ABI: A8 on a codebook other than mad, a group that does not divide K, rows past the scratch, act spellings
//   M  mad: A and B run mad too; the byte-sum GEMV (a8g128, 1 .. 8 rows, exl3i_mad.hip.h): M1 every rate,
//      [512, 1024] / stacked [512, 512]: its rows in the scratch (xb, sa, bias) byte for byte against the reference's
//      int8 activations, y (f32) against the exact GEMM of the quantised operands (|d| <= 2e-6 max |y|: the f32 fold
//      only); M2 the trunk shapes x rates x T 1..8 against the device's A8 path on the same rows (pin A8_MIN_ROWS 1;
//      the same integers, another f32 order); M3 a row of an 8-row launch = the 1-row launch of that row bit for bit,
//      a KS launch byte-identical over 20 repeats, and the f16 path on the same weight (pin MAD_GEMV_OFF) within the
//      int8 activations' error
// The plugin's build id is printed first: a result belongs to that build.
#include <hip/hip_runtime.h>
#include "exl3i.h"
#include "exl3i_a8.h"
#include "exl3i_pin.h"
#include "exl3i_decode.h"
#include "exl3_decode.h"
#include "exl3i_twopass_ref.h"
#include "exl3i_dev.h"   // exl3i_mad_row_bytes

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#define HIPCHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

static int FAILS = 0, CHECKED = 0;
static bool QUICK = false;
static void fail(const std::string& what) { ++FAILS; std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
static bool expect(bool ok, const std::string& what) { ++CHECKED; if (!ok) fail(what); return ok; }

static uint64_t RNG = 0x2545F4914F6CDD1Dull;
static uint32_t rnd() { RNG ^= RNG << 13; RNG ^= RNG >> 7; RNG ^= RNG << 17; return (uint32_t)(RNG >> 16); }
static float rndn() { float s = 0; for (int i = 0; i < 4; ++i) s += (rnd() & 0xFFFF) / 65536.0f; return (s - 2.0f) * 1.7320508f; }
static uint16_t f2h(float f) { return exl3_half_from_double((double)f); }
static float h2f(uint16_t u) { return exl3_half_to_float(u); }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static void dfree(void* p) { if (p) (void)hipFree(p); }
static void* upload(const void* h, size_t n) { void* d; HIPCHECK(hipMalloc(&d, n ? n : 256)); if (n) HIPCHECK(hipMemcpy(d, h, n, hipMemcpyHostToDevice)); return d; }
static std::string rate_s(int m) { char b[16]; return exl3i_rate_str(m, b); }
static const char* dtn(uint32_t dt) { return dt == RAD_F32 ? "f32" : dt == RAD_F16 ? "f16" : dt == RAD_BF16 ? "bf16" : "?"; }

// ============================================================================= the plugin
static void* PH = nullptr;
static const RadKernelInfo *RF = nullptr, *RA = nullptr;   // exl3i_gemm_nt, exl3i_gemm_nt_a8
static const RadPluginInfo* INFO = nullptr;
static void (*PIN)(int, int64_t) = nullptr;   // the plugin's exl3i_test_pin (exl3i_pin.h)
static int64_t PINS[EXL3I_PIN_N] = {};        // what this test pinned, for the arena it expects
static void test_pin(int k, int64_t v) { PINS[k] = v; PIN(k, v); }

static void open_plugin(const char* path) {
    PH = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!PH) { std::fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); std::exit(2); }
    auto ver = (uint32_t (*)())dlsym(PH, "rad_plugin_abi_version");
    auto inf = (const RadPluginInfo* (*)())dlsym(PH, "rad_plugin_info");
    auto cnt = (int (*)())dlsym(PH, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(PH, "rad_kernel_at");
    if (!ver || !inf || !cnt || !at || ver() != RAD_ABI_VERSION) { std::fprintf(stderr, "%s: not an ABI %d plugin\n", path, RAD_ABI_VERSION); std::exit(2); }
    INFO = inf();
    for (int i = 0; i < cnt(); ++i) {
        if (!std::strcmp(at(i)->name, "exl3i_gemm_nt")) RF = at(i);
        if (!std::strcmp(at(i)->name, "exl3i_gemm_nt_a8")) RA = at(i);
    }
    if (!RF || !RA) { std::fprintf(stderr, "%s: rows exl3i_gemm_nt and exl3i_gemm_nt_a8 wanted\n", path); std::exit(2); }
    PIN = (void (*)(int, int64_t))dlsym(PH, "exl3i_test_pin");
    if (!PIN) { std::fprintf(stderr, "%s: no exl3i_test_pin\n", path); std::exit(2); }
}

// ============================================================================= a weight, stored through the hooks
struct Weight {
    int m, cb, N, K;
    bool stacked;
    std::vector<uint32_t> rec;
    std::vector<uint16_t> suh, suh1, svh;
    std::vector<float> W;   // the oracle's decode [N][K] (made on demand)
    Exl3iView view() const {
        Exl3iView v{};
        v.rows = N; v.cols = K; v.m = m; v.cb = cb; v.record = rec.data(); v.suh = suh.data();
        v.suh1 = stacked ? suh1.data() : nullptr; v.svh = svh.data();
        return v;
    }
    const std::vector<float>& decoded() {
        if (W.empty()) {
            W.resize((size_t)N * K);
            Exl3iView v = view();
            if (exl3i_decode_rows(&v, 0, N, W.data()) != 0) { std::fprintf(stderr, "oracle decode failed\n"); std::exit(2); }
        }
        return W;
    }
};

static Weight make_weight(int m, int cb, int N, int K, bool stacked) {
    Weight w{m, cb, N, K, stacked, {}, {}, {}, {}, {}};
    w.rec.resize((size_t)(N / 16) * exl3i_row_words(K, m));
    for (auto& x : w.rec) x = rnd() ^ (rnd() << 16);
    auto scales = [](std::vector<uint16_t>& v, int n) {
        v.resize(n);
        for (auto& x : v) x = f2h((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f));
    };
    scales(w.suh, K); scales(w.svh, N); scales(w.suh1, K);
    return w;
}

struct Params {
    char rate[16], codebook[16], act[16];
    std::vector<RadParam> p;
    Params(int64_t M, int N, int K, const char* rt, const char* cb, const char* ac) {
        std::snprintf(rate, sizeof rate, "%s", rt); std::snprintf(codebook, sizeof codebook, "%s", cb);
        std::snprintf(act, sizeof act, "%s", ac);
        p = { RAD_INT("M", M), RAD_INT("N", N), RAD_INT("K", K), RAD_STR("rate", rate), RAD_STR("codebook", codebook),
              RAD_STR("act", act) };
    }
};

static RadTensor tensor(void* d, uint32_t dt, int64_t r0, int64_t r1, int64_t pitch = -1) {
    RadTensor t{};
    t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r0; t.shape[1] = r1; t.stride[0] = pitch < 0 ? r1 : pitch; t.stride[1] = 1;
    return t;
}

struct Dev {
    void *rec = nullptr, *suh = nullptr, *suh1 = nullptr, *svh = nullptr;
    ~Dev() { dfree(rec); dfree(suh); dfree(suh1); dfree(svh); }
};

static int store_weight(const RadKernelInfo* row, const Weight& w, const Params& pa, Dev* d) {
    RadEncoding e;
    if (exl3i_encoding_make(w.m, w.cb, w.stacked, w.N, w.K, &e) != RAD_OK) return RAD_E_INVAL;
    struct { int opd, plane; const void* data; uint32_t dt; int64_t r, c; void** out; } ops[4] = {
        { EXL3I_OPD_RECORD, EXL3I_PLANE_RECORD, w.rec.data(), RAD_U32, w.N / 16, exl3i_row_words(w.K, w.m), &d->rec },
        { EXL3I_OPD_SUH, EXL3I_PLANE_SUH, w.suh.data(), RAD_F16, 1, w.K, &d->suh },
        { EXL3I_OPD_SVH, EXL3I_PLANE_SVH, w.svh.data(), RAD_F16, w.N, 1, &d->svh },
        { EXL3I_OPD_SUH1, EXL3I_PLANE_SUH1, w.suh1.data(), RAD_F16, 1, w.K, &d->suh1 },
    };
    for (int i = 0; i < (w.stacked ? 4 : 3); ++i) {
        RadTensor pl = tensor((void*)ops[i].data, ops[i].dt, ops[i].r, ops[i].c);
        RadLayout L;
        int sel = ops[i].plane;
        int rc = row->layout(pa.p.data(), (int)pa.p.size(), ops[i].opd, &e, &sel, &pl, 1, &L);
        if (rc != RAD_OK) return rc;
        std::vector<char> buf((size_t)L.bytes);
        rc = row->relayout(pa.p.data(), (int)pa.p.size(), ops[i].opd, &e, &sel, &pl, 1, buf.data(), L.bytes);
        if (rc != RAD_OK) return rc;
        *ops[i].out = upload(buf.data(), buf.size());
    }
    return RAD_OK;
}

// ============================================================================= x, a launch, its scratch
struct Run {
    int M; uint32_t xdt, ydt; int xpad; bool x3d;
};

struct Xin {
    std::vector<uint16_t> raw;   // [M][pitch] as the device reads it
    std::vector<float> f;        // [M][K] the values
    int64_t pitch;
    void* dev = nullptr;
    ~Xin() { dfree(dev); }
};

static void make_x(const Run& r, int K, Xin* x) {
    x->pitch = K + r.xpad;
    x->raw.assign((size_t)r.M * x->pitch, 0);
    x->f.resize((size_t)r.M * K);
    for (int i = 0; i < r.M; ++i) {
        const float rowscale = 0.25f + (float)(rnd() % 16) / 4.0f;   // rows of different size (per-token scales differ)
        for (int k = 0; k < K; ++k) {
            const float v = rndn() * rowscale * ((k % 97 == 3) ? 6.0f : 1.0f);   // a few larger channels
            const uint16_t u = r.xdt == RAD_F16 ? f2h(v) : f2bf(v);
            x->raw[(size_t)i * x->pitch + k] = u;
            x->f[(size_t)i * K + k] = r.xdt == RAD_F16 ? h2f(u) : bf2f(u);
        }
    }
    x->dev = upload(x->raw.data(), x->raw.size() * 2);
}

struct Out {
    int rc = 0;
    std::vector<char> y;
    std::vector<char> scratch;   // the op's scratch after the launch (when asked)
    int64_t scratch_bytes = 0;
};

// One launch through `row`, the band declared at Md rows. extra: bytes past the hook's answer.
static Out launch(const RadKernelInfo* row, const Params& pa, const Weight& w, const Dev& d, const Run& r, const Xin& x,
                  bool keep_scratch, int64_t extra = 0, int repeat = 1) {
    Out o;
    RadTensor t[6]{};
    t[0] = tensor(x.dev, r.xdt, r.M, w.K, x.pitch);
    if (r.x3d) { t[0].rank = 3; t[0].shape[1] = w.K / 128; t[0].shape[2] = 128; t[0].stride[0] = x.pitch; t[0].stride[1] = 128; t[0].stride[2] = 1; }
    t[1] = tensor(d.rec, EXL3I_DT_TILES, w.N / 16, exl3i_row_words(w.K, w.m));
    t[2] = tensor(d.suh, RAD_F16, 1, w.K);
    t[3] = tensor(d.svh, RAD_F16, w.N, 1);
    t[4] = tensor(w.stacked ? d.suh1 : nullptr, RAD_F16, 1, w.K);
    const size_t yb = (size_t)r.M * w.N * rad_dtype_bytes(r.ydt, 1);
    void* dy; HIPCHECK(hipMalloc(&dy, yb)); HIPCHECK(hipMemset(dy, 0xCD, yb));
    t[5] = tensor(dy, r.ydt, r.M, w.N);
    RadArgs a{};
    a.t = t; a.n_t = 6; a.p = pa.p.data(); a.n_p = (int)pa.p.size(); a.world_size = 1;
    const int64_t sb = row->scratch ? row->scratch(&a) : 0;
    if (sb < 0) { o.rc = (int)sb; dfree(dy); return o; }
    o.scratch_bytes = sb + extra;
    void* sc = nullptr;
    if (o.scratch_bytes > 0) { HIPCHECK(hipMalloc(&sc, (size_t)o.scratch_bytes)); HIPCHECK(hipMemset(sc, 0x5A, (size_t)o.scratch_bytes)); }
    a.scratch = sc; a.scratch_bytes = o.scratch_bytes;
    for (int i = 0; i < repeat; ++i) o.rc = row->launch(&a, nullptr);
    HIPCHECK(hipDeviceSynchronize());
    o.y.resize(yb);
    HIPCHECK(hipMemcpy(o.y.data(), dy, yb, hipMemcpyDeviceToHost));
    if (keep_scratch && o.scratch_bytes > 0) {
        o.scratch.resize((size_t)o.scratch_bytes);
        HIPCHECK(hipMemcpy(o.scratch.data(), sc, (size_t)o.scratch_bytes, hipMemcpyDeviceToHost));
    }
    dfree(dy); dfree(sc);
    return o;
}

static float out_val(const std::vector<char>& y, uint32_t dt, size_t i) {
    if (dt == RAD_F32) { float f; std::memcpy(&f, y.data() + 4 * i, 4); return f; }
    uint16_t u; std::memcpy(&u, y.data() + 2 * i, 2);
    return dt == RAD_F16 ? h2f(u) : bf2f(u);
}
static double ulp_of(uint32_t dt, double v) {
    v = std::fabs(v);
    if (dt == RAD_F32) return 0.0;
    const int e = v > 0 ? (int)std::floor(std::log2(v)) : -14;
    return std::ldexp(1.0, (e < -14 ? -14 : e) - (dt == RAD_F16 ? 10 : 7));
}

// y against a reference: rel rms err, worst element over (ulp(y) + tol_abs x rms), finiteness
struct Cmp { double rel = 0, worst = 0; bool finite = true; };
static Cmp compare(const std::vector<char>& y, uint32_t dt, const std::vector<double>& ref, double tol_abs) {
    Cmp c;
    double sr = 0, se = 0;
    for (double v : ref) sr += v * v;
    const double rms = std::sqrt(sr / ref.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        const double v = out_val(y, dt, i), d = v - ref[i];
        if (!std::isfinite(v)) c.finite = false;
        se += d * d;
        const double q = std::fabs(d) / (ulp_of(dt, ref[i]) + tol_abs * rms);
        if (q > c.worst) c.worst = q;
    }
    c.rel = std::sqrt(se / ref.size()) / rms;
    return c;
}

static std::vector<double> exact_ref(Weight& w, const Xin& x, int M) {
    const std::vector<float>& W = w.decoded();
    std::vector<double> ref((size_t)M * w.N);
    for (int i = 0; i < M; ++i)
        for (int n = 0; n < w.N; ++n) {
            double s = 0;
            const float* xr = &x.f[(size_t)i * w.K];
            const float* wr = &W[(size_t)n * w.K];
            for (int k = 0; k < w.K; ++k) s += (double)xr[k] * wr[k];
            ref[(size_t)i * w.N + n] = s;
        }
    return ref;
}

// The A8 threshold as nt.c reads it (the pin, else EXL3I_A8_MIN_ROWS).
static int64_t a8_min_rows() { return PINS[EXL3I_PIN_A8_MIN_ROWS] > 0 ? PINS[EXL3I_PIN_A8_MIN_ROWS] : EXL3I_A8_MIN_ROWS; }

// The A8 arena at real rows T as nt.c lays it out (with the pins as nt.c reads them).
static Exl3iA8Plan a8_plan(int64_t T, const Weight& w, int64_t group) {
    const int64_t cm = PINS[EXL3I_PIN_A8_CHUNK_MB];
    Exl3iA8Plan p{};
    exl3i_a8_plan(T, w.N, w.K, w.stacked, group, cm > 0 ? cm << 20 : EXL3I_A8_CHUNK_BYTES, a8_min_rows(), &p);
    return p;
}

// x rows as one [M][K] 16-bit block (the reference's input; the padded pitch dropped)
static std::vector<uint16_t> x_rows(const Xin& x, int M, int K) {
    std::vector<uint16_t> r((size_t)M * K);
    for (int i = 0; i < M; ++i) std::memcpy(&r[(size_t)i * K], &x.raw[(size_t)i * x.pitch], (size_t)K * 2);
    return r;
}

// ============================================================================= an A8 case end to end
struct A8Case {
    int m = 64, N = 256, K = 256, Md = 2048, cb = EXL3I_CB_MAD;
    bool stacked = false;
    const char* act = "a8g128";
    int64_t group = 128;
    Run r{512, RAD_F16, RAD_F16, 0, false};
    bool planes = false;        // compare the scratch's planes byte for byte
    bool exact = true;          // also against x . W^T
    int64_t extra = 0;
};

struct A8Res { bool ok = false; std::vector<char> y; double rel_q = 0, rel_x = 0; };

static A8Res a8_case(const A8Case& c, const std::string& tag) {
    A8Res res;
    Weight w = make_weight(c.m, c.cb, c.N, c.K, c.stacked);
    Params pa(c.Md, c.N, c.K, rate_s(c.m).c_str(), exl3i_codebook_name(c.cb), c.act);
    Dev d;
    if (!expect(store_weight(RA, w, pa, &d) == RAD_OK, tag + ": the layout hooks refused")) return res;
    Xin x;
    make_x(c.r, c.K, &x);
    Out o = launch(RA, pa, w, d, c.r, x, c.planes, c.extra);
    if (!expect(o.rc == RAD_OK, tag + ": launch returned " + std::to_string(o.rc))) return res;
    res.y = o.y;
    const int M = c.r.M;
    const Exl3iA8Plan p = a8_plan(M, w, c.group);
    const bool a8 = M >= a8_min_rows();
    bool ok = true;
    if (a8) {
        // the activation planes the device wrote, and the reference's
        const int mats = c.stacked ? 2 : 1;
        std::vector<uint16_t> xr = x_rows(x, M, c.K);
        std::vector<int8_t> aq((size_t)mats * M * c.K);
        std::vector<float> sa((size_t)mats * M * (c.K / 128));
        exl3i_ref_actq(xr.data(), c.K, c.r.xdt == RAD_BF16, w.suh.data(), w.suh1.data(), mats, M, c.K, c.group, aq.data(), sa.data());
        std::vector<int8_t> w8((size_t)c.N * c.K);
        Exl3iView v = w.view();
        exl3i_ref_i8plane(&v, 0, c.N, w8.data());
        if (c.planes) {
            const char* s = o.scratch.data();
            ok &= expect(!std::memcmp(s + p.aq_off, aq.data(), aq.size()), tag + ": int8 activations differ from the reference");
            ok &= expect(!std::memcmp(s + p.sa_off, sa.data(), sa.size() * 4), tag + ": activation scales differ from the reference");
            if (p.chunk >= c.N)   // one chunk: the whole plane is in the scratch
                ok &= expect(!std::memcmp(s + p.w_off, w8.data(), w8.size()), tag + ": the int8 weight plane differs from the reference");
            else {                // the last chunk's
                const int64_t c0 = (c.N - 1) / p.chunk * p.chunk, nc = c.N - c0;
                std::vector<int8_t> last((size_t)(nc * c.K));
                exl3i_ref_i8plane(&v, c0, nc, last.data());
                ok &= expect(!std::memcmp(s + p.w_off, last.data(), last.size()), tag + ": the last chunk's int8 plane differs");
            }
        }
        // y byte for byte against the device's arithmetic emulated (the fold, the bias flushes, the epilogue's order)
        {
            const int ydc = c.r.ydt == RAD_F32 ? 1 : c.r.ydt == RAD_F16 ? 2 : 0;
            const size_t eb = rad_dtype_bytes(c.r.ydt, 1);
            std::vector<char> yd(o.y.size());
            exl3i_ref_a8_gemm_dev_cb(aq.data(), sa.data(), w8.data(), w.svh.data(), mats, M, c.N, c.K, c.cb, ydc, yd.data(), c.N);
            size_t nd = 0;
            for (size_t i = 0; i < yd.size() / eb; ++i) nd += std::memcmp(&yd[i * eb], &o.y[i * eb], eb) != 0;
            ok &= expect(nd == 0, tag + ": y differs from the device arithmetic's emulation in " + std::to_string(nd) + " of " +
                                      std::to_string(yd.size() / eb) + " values");
        }
        // ... and against the exact (double) GEMM of the same operands: the fold's f32 rounding, rel rms only (the worst
        // element grows with the activation scale's width and K -- a8t at K 1024+ passes 6e-3 of rms by ~1.2x -- and
        // the byte check above already pins every value)
        std::vector<double> ref((size_t)M * c.N);
        exl3i_ref_a8_gemm_cb(aq.data(), sa.data(), w8.data(), w.svh.data(), mats, M, c.N, c.K, c.cb, ref.data());
        const Cmp q = compare(o.y, c.r.ydt, ref, 6e-3);
        const double lim = c.r.ydt == RAD_BF16 ? 4e-3 : 1.5e-3;
        res.rel_q = q.rel;
        char s[200];
        std::snprintf(s, sizeof s, "%s: vs the quantised operands rel rms %.3g (limit %.3g), worst %.3g x (ulp + 6e-3 rms)%s",
                      tag.c_str(), q.rel, lim, q.worst, q.finite ? "" : ", NOT FINITE");
        ok &= expect(q.finite && q.rel < lim, s);
        if (q.worst > 1.0) std::printf("  note: %s\n", s);
    }
    if (c.exact) {
        const std::vector<double> ref = exact_ref(w, x, M);
        const Cmp e = compare(o.y, c.r.ydt, ref, a8 ? 0.12 : 4e-3);
        const double lim = a8 ? 2.5e-2 : (c.r.ydt == RAD_BF16 ? 4e-3 : 1.5e-3);
        res.rel_x = e.rel;
        char s[200];
        std::snprintf(s, sizeof s, "%s: vs x . W^T rel rms %.3g (limit %.3g), worst %.3g x tolerance%s", tag.c_str(), e.rel, lim,
                      e.worst, e.finite ? "" : ", NOT FINITE");
        ok &= expect(e.finite && e.rel < lim && e.worst <= 1.0, s);
    }
    res.ok = ok;
    return res;
}

// ============================================================================= A: the A8 planes at every rate
static void test_a8_planes() {
  for (int cb : { (int)EXL3I_CB_MAD }) {
    int n = 0, bad = 0;
    double wq = 0, wx = 0;
    for (int m = 2; m <= 128; m += QUICK ? 14 : 2) {
        A8Case c;
        c.m = m; c.cb = cb; c.r = Run{512, (m / 2) % 2 ? RAD_BF16 : RAD_F16, RAD_F16, 0, false}; c.planes = true; c.Md = 512;
        const A8Res r = a8_case(c, std::string("A ") + exl3i_codebook_name(cb) + " m " + std::to_string(m) + " (K" + rate_s(m) + ")");
        ++n; bad += !r.ok;
        wq = std::max(wq, r.rel_q); wx = std::max(wx, r.rel_x);
    }
    std::printf("A A8 planes %s: %d rates, [256, 256], M 512: %d failed (int8 weight plane, int8 activations, scales byte for "
                "byte; y byte for byte vs the device emulation; worst rel rms %.3g vs quantised operands, %.3g vs x . W^T)\n",
                exl3i_codebook_name(cb), n, bad, wq, wx);
  }
}

// ============================================================================= B: T2's planes
static void test_t2_planes() {
    int n = 0, bad = 0;
    const int rates[] = { 2, 14, 32, 40, 64, 84, 112, 128 };
    for (int cb : { (int)EXL3I_CB_MAD })
        for (int m : rates) {
            const int N = cb == (int)EXL3I_CB_MAD ? 512 : 256, K = 256, M = 512;   // stacked (mad): the seam on T2's 256-column block
            Weight w = make_weight(m, cb, N, K, cb == (int)EXL3I_CB_MAD);
            Params pa(M, N, K, rate_s(m).c_str(), exl3i_codebook_name(cb), "f16");
            Dev d;
            if (!expect(store_weight(RF, w, pa, &d) == RAD_OK, "B hooks")) continue;
            const Run r{M, cb ? RAD_BF16 : RAD_F16, RAD_BF16, 0, false};
            Xin x;
            make_x(r, K, &x);
            test_pin(EXL3I_PIN_T2_OFF, 0);
            Out o = launch(RF, pa, w, d, r, x, true);
            const std::string tag = "B m " + std::to_string(m) + " cb " + std::to_string(cb);
            bool ok = expect(o.rc == RAD_OK, tag + ": launch");
            if (ok) {
                // the region [T][2K] f16: row i's half h (suh, suh.1) at [i][h K]
                std::vector<uint16_t> xr = x_rows(x, M, K), rot(128);
                int64_t off = 0;
                for (int i = 0; i < M && ok; ++i)
                    for (int h = 0; h < (w.stacked ? 2 : 1); ++h)
                        for (int b = 0; b < K / 128; ++b) {
                            exl3i_ref_rotate128(&xr[(size_t)i * K + b * 128], r.xdt == RAD_BF16, (h ? w.suh1 : w.suh).data() + b * 128, rot.data());
                            off += std::memcmp(o.scratch.data() + ((size_t)i * 2 * K + h * K + b * 128) * 2, rot.data(), 256) != 0;
                        }
                ok &= expect(off == 0, tag + ": the rotated rows differ from rotate128's transcription (" + std::to_string(off) + " blocks)");
                std::vector<uint16_t> f16((size_t)N * K);
                Exl3iView v = w.view();
                exl3i_ref_f16plane(&v, 0, N, f16.data());
                const int64_t xr_bytes = ((int64_t)M * 2 * K * 2 + 255) / 256 * 256;
                ok &= expect(!std::memcmp(o.scratch.data() + xr_bytes, f16.data(), f16.size() * 2), tag + ": T2's f16 plane differs");
            }
            ++n; bad += !ok;
        }
    std::printf("B T2 planes: %d (8 rates x mad, M 512): %d failed (rotated rows and the f16 weight plane byte for byte)\n",
                n, bad);
}

// ============================================================================= C: the selftest matrix
static void test_matrix() {
    int n = 0, bad = 0;
    auto go = [&](const A8Case& c, const std::string& tag) { const A8Res r = a8_case(c, tag); ++n; bad += !r.ok; return r; };
    // real rows against a band declared at 2048, three store dtypes, both a dtypes, padded pitch
    const int Ms[] = { 1, 15, 511, 512, 513, 952, 1500, 2048 };
    const uint32_t ydts[] = { RAD_BF16, RAD_F16, RAD_F32 };
    int i = 0;
    for (int M : Ms) {
        if (QUICK && M > 1024) continue;
        A8Case c;
        c.m = 40; c.N = 512; c.K = 512; c.Md = 2048; c.stacked = i % 2 == 1;
        c.r = Run{M, i % 2 ? RAD_BF16 : RAD_F16, ydts[i % 3], i % 4 == 2 ? 64 : 0, false};
        c.planes = true; c.exact = M <= 1500;
        go(c, "C M " + std::to_string(M) + " in a 2048 band, y " + dtn(c.r.ydt) + (c.stacked ? ", stacked" : "") + (c.r.xpad ? ", padded" : ""));
        ++i;
    }
    // below 512: the A8 row's output = the f16 row's (the same kernel, the same stored weight)
    for (int M : { 1, 17, 511 }) {
        Weight w = make_weight(48, EXL3I_CB_MAD, 256, 512, false);
        Params pa(2048, 256, 512, "3", "mad", "a8g128"), pf(2048, 256, 512, "3", "mad", "f16");
        Dev da, df;
        store_weight(RA, w, pa, &da);
        store_weight(RF, w, pf, &df);
        const Run r{M, RAD_BF16, RAD_BF16, 0, false};
        Xin x;
        make_x(r, 512, &x);
        test_pin(EXL3I_PIN_T2_OFF, 1);   // the f16 row's form below 512 rows is nt_pass either way; keep T2 out of the f16 row
        Out oa = launch(RA, pa, w, da, r, x, false), of = launch(RF, pf, w, df, r, x, false);
        test_pin(EXL3I_PIN_T2_OFF, 0);
        ++n;
        bad += !expect(oa.rc == RAD_OK && of.rc == RAD_OK && oa.y == of.y, "C M " + std::to_string(M) + ": the A8 row below 512 rows is not the f16 row's output");
    }
    // the GDN ssm_out input, [T, 48, 128] read as [T, 6144]
    for (int M : { 600, 33 }) {
        A8Case c;
        c.m = 64; c.N = 256; c.K = 6144; c.r = Run{M, RAD_BF16, RAD_BF16, 0, true}; c.planes = M >= 512;
        go(c, "C ssm_out 3-D input M " + std::to_string(M));
    }
    // groups
    struct { const char* act; int64_t g; int K; } groups[] = { { "a8t", 1024, 1024 }, { "a8g256", 256, 1024 }, { "a8g512", 512, 2048 },
                                                               { "a8g1024", 1024, 2048 }, { "a8t", 2560, 2560 } };
    for (auto& g : groups) {
        A8Case c;
        c.m = 56; c.N = 256; c.K = g.K; c.act = g.act; c.group = g.g; c.r = Run{640, RAD_F16, RAD_BF16, 0, false}; c.planes = true;
        go(c, std::string("C ") + g.act + " K " + std::to_string(g.K));
    }
    // rates K2..K8 and fractional / sub-1 at M 1024, stacked (the seam), y f32
    for (int m : { 2, 8, 14, 32, 36, 40, 44, 48, 64, 72, 80, 96, 104, 112, 120, 128 }) {
        if (QUICK && m % 16) continue;
        A8Case c;
        c.m = m; c.N = 512; c.K = 512; c.stacked = true; c.r = Run{1024, RAD_BF16, RAD_F32, 0, false}; c.exact = !QUICK;
        go(c, "C rate K" + rate_s(m) + " stacked M 1024");
    }
    // several chunks: 1 MB of plane a chunk at K 1024 = 1024 columns; N 2560 -> 3 chunks (the last 512)
    test_pin(EXL3I_PIN_A8_CHUNK_MB, 1);
    {
        A8Case c;
        c.m = 48; c.N = 2560; c.K = 1024; c.stacked = true; c.r = Run{777, RAD_F16, RAD_BF16, 0, false}; c.planes = true; c.exact = !QUICK;
        go(c, "C three chunks (pin A8_CHUNK_MB 1), stacked");
    }
    test_pin(EXL3I_PIN_A8_CHUNK_MB, 0);
    // two scratch sizes and a replay: byte-identical
    {
        A8Case c;
        c.m = 64; c.N = 768; c.K = 1024; c.r = Run{2048, RAD_BF16, RAD_BF16, 0, false}; c.exact = false;
        const uint64_t seed = RNG;
        const A8Res a = go(c, "C M 2048 (16 bands, the raster groups)");
        RNG = seed;
        c.extra = 64 << 20;
        const A8Res b = go(c, "C M 2048, scratch + 64 MB");
        ++n;
        bad += !expect(a.ok && b.ok && a.y == b.y, "C two scratch sizes: outputs differ");
        Weight w = make_weight(64, EXL3I_CB_MAD, 768, 1024, false);
        Params pa(4096, 768, 1024, "4", "mad", "a8g128");
        Dev d;
        store_weight(RA, w, pa, &d);
        const Run r{4096, RAD_BF16, RAD_BF16, 0, false};
        Xin x;
        make_x(r, 1024, &x);
        Out o1 = launch(RA, pa, w, d, r, x, false), o2 = launch(RA, pa, w, d, r, x, false, 0, 3);
        ++n;
        bad += !expect(o1.rc == RAD_OK && o2.rc == RAD_OK && o1.y == o2.y, "C M 4096: a replay (three launches) differs from one");
    }
    // the threshold knob: A8 from 16 rows (what the bench's M 16-511 sweep runs)
    test_pin(EXL3I_PIN_A8_MIN_ROWS, 16);
    for (int M : { 16, 100, 256 }) {
        A8Case c;
        c.m = 64; c.N = 256; c.K = 512; c.Md = 256; c.r = Run{M, RAD_F16, RAD_BF16, 0, false}; c.planes = true;
        go(c, "C pin A8_MIN_ROWS 16, M " + std::to_string(M));
    }
    test_pin(EXL3I_PIN_A8_MIN_ROWS, 0);
    std::printf("C selftest matrix: %d cases, %d failed\n", n, bad);
}

// ============================================================================= D: refusals through the ABI
static void test_refusals() {
    int n = 0, bad = 0;
    auto want = [&](int got, int w, const std::string& what) { ++n; bad += !expect(got == w, "D " + what + ": got " + std::to_string(got) + ", want " + std::to_string(w)); };
    Exl3iGeom g;
    { Params p(1, 256, 256, "2.5", "mad", "a8g0128"); want(exl3i_geom_parse(p.p.data(), 6, &g), RAD_E_DTYPE, "act a8g0128"); }
    { Params p(1, 256, 256, "2.5", "mad", "A8T"); want(exl3i_geom_parse(p.p.data(), 6, &g), RAD_E_DTYPE, "act A8T"); }
    { Params p(1, 256, 256, "2.5", "mad", "a8g64"); want(exl3i_geom_parse(p.p.data(), 6, &g), RAD_E_DTYPE, "act a8g64"); }
    { Params p(2048, 256, 1536, "2.5", "mad", "a8g1024"); RadArgs a{}; a.p = p.p.data(); a.n_p = 6;
      want(RA->scratch(&a) < 0, 1, "scratch: a8g1024 with K 1536"); }
    { Params p(2048, 256, 256, "2.5", "other", "a8g128"); RadArgs a{}; a.p = p.p.data(); a.n_p = 6;
      want(RA->scratch(&a) < 0, 1, "scratch: A8 on a codebook other than mad"); }
    Weight w = make_weight(40, EXL3I_CB_MAD, 256, 256, false);
    Params pf(2048, 256, 256, "2.5", "mad", "f16");
    Dev d;
    store_weight(RF, w, pf, &d);
    const Run r{600, RAD_F16, RAD_BF16, 0, false};
    Xin x;
    make_x(r, 256, &x);
    {   // the f16 row's launch (the scratch the rows past the band are checked against below)
        Out o = launch(RF, pf, w, d, r, x, false);   // a sane launch first
        want(o.rc, RAD_OK, "the f16 row on mad");
        RadTensor t[6]{};
        t[0] = tensor(x.dev, RAD_F16, 600, 256); t[1] = tensor(d.rec, EXL3I_DT_TILES, 16, exl3i_row_words(256, 40));
        t[2] = tensor(d.suh, RAD_F16, 1, 256); t[3] = tensor(d.svh, RAD_F16, 256, 1); t[4] = tensor(nullptr, RAD_F16, 1, 256);
        void* dy; HIPCHECK(hipMalloc(&dy, 600 * 256 * 2)); t[5] = tensor(dy, RAD_BF16, 600, 256);
        void* sc; HIPCHECK(hipMalloc(&sc, 64 << 20));
        RadArgs a{}; a.t = t; a.n_t = 6; a.scratch = sc; a.scratch_bytes = 64 << 20; a.world_size = 1;
        // rows past the band the scratch was sized for (A8 on mad, declared 512, launched at 4096 rows)
        Weight w8 = make_weight(40, EXL3I_CB_MAD, 256, 256, false);
        Params p8(512, 256, 256, "2.5", "mad", "a8g128");
        Dev d8;
        store_weight(RA, w8, p8, &d8);
        void* dx; HIPCHECK(hipMalloc(&dx, 4096 * 256 * 2)); HIPCHECK(hipMemset(dx, 0, 4096 * 256 * 2));
        void* dy2; HIPCHECK(hipMalloc(&dy2, 4096 * 256 * 2));
        t[0] = tensor(dx, RAD_F16, 4096, 256); t[1] = tensor(d8.rec, EXL3I_DT_TILES, 16, exl3i_row_words(256, 40));
        t[2] = tensor(d8.suh, RAD_F16, 1, 256); t[3] = tensor(d8.svh, RAD_F16, 256, 1); t[5] = tensor(dy2, RAD_BF16, 4096, 256);
        a.p = p8.p.data();
        a.n_p = 6;
        const int64_t sb = RA->scratch(&a);
        a.scratch_bytes = sb;
        want(RA->launch(&a, nullptr), RAD_E_SCRATCH, "launch: 4096 rows in a band sized at 512");
        HIPCHECK(hipDeviceSynchronize());
        dfree(dy); dfree(sc); dfree(dx); dfree(dy2);
    }
    std::printf("D refusals: %d, %d wrong\n", n, bad);
}


// ============================================================================= M: mad's byte-sum GEMV
static float f32_at(const std::vector<char>& y, size_t i) { float f; std::memcpy(&f, y.data() + 4 * i, 4); return f; }

// The exact GEMM of the quantised operands for output rows [c0, c0 + 128) of a mad weight (the activation plane of the
// rows' half): double, as exl3i_ref_a8_gemm_cb over that block alone.
static std::vector<double> mad_ref_block(const Weight& w, const std::vector<int8_t>& aq, const std::vector<float>& sa, int T,
                                         int64_t c0) {
    Exl3iView v = w.view();
    std::vector<int8_t> w8((size_t)128 * w.K);
    exl3i_ref_i8plane(&v, c0, 128, w8.data());
    const int mat = w.stacked && c0 >= w.N / 2 ? 1 : 0;
    std::vector<double> ref((size_t)T * 128);
    exl3i_ref_a8_gemm_cb(aq.data() + (size_t)mat * T * w.K, sa.data() + (size_t)mat * T * (w.K / 128), w8.data(),
                         w.svh.data() + c0, 1, T, 128, w.K, EXL3I_CB_MAD, ref.data());
    return ref;
}

static void test_mad() {
    test_pin(EXL3I_PIN_MAD_GEMV_T, 8);   // the GEMV at every row count it takes (by itself it stops at 1 / 2 rows)
    // M1: every rate, small shapes, against the reference: the rows the GEMV quantised and y
    {
        int n = 0, bad = 0;
        double worst = 0;
        for (int m = 2; m <= 128; m += QUICK ? 14 : 2) {
            const bool st = (m / 2) % 3 == 1;
            const int N = 512, K = st ? 512 : 1024, T = 1 + (m / 2) % 8, mats = st ? 2 : 1;
            const std::string tag = "M1 m " + std::to_string(m) + " (K" + rate_s(m) + ") T " + std::to_string(T) + (st ? " stacked" : "");
            Weight w = make_weight(m, EXL3I_CB_MAD, N, K, st);
            Params pa(8, N, K, rate_s(m).c_str(), "mad", "a8g128");
            Dev d;
            if (!expect(store_weight(RA, w, pa, &d) == RAD_OK, tag + ": hooks")) { ++bad; continue; }
            const Run r{T, (m / 2) % 2 ? RAD_BF16 : RAD_F16, RAD_F32, (m / 2) % 4 == 3 ? 128 : 0, false};
            Xin x;
            make_x(r, K, &x);
            Out o = launch(RA, pa, w, d, r, x, true);
            bool ok = expect(o.rc == RAD_OK, tag + ": launch " + std::to_string(o.rc));
            if (ok) {
                std::vector<uint16_t> xr = x_rows(x, T, K);
                std::vector<int8_t> aq((size_t)mats * T * K);
                std::vector<float> sa((size_t)mats * T * (K / 128));
                exl3i_ref_actq(xr.data(), K, r.xdt == RAD_BF16, w.suh.data(), w.suh1.data(), mats, T, K, 128, aq.data(), sa.data());
                // the GEMV's rows (its pre-pass, AQ form xb's layout): row t, matrix i at (t mats + i) exl3i_mad_row_bytes(K)
                const int64_t rb = exl3i_mad_row_bytes(K), nkb = K / 128;
                int64_t off = 0;
                for (int i = 0; i < mats; ++i)
                    for (int t = 0; t < T; ++t) {
                        const char* row = o.scratch.data() + (size_t)(t * mats + i) * rb;
                        const int8_t* q = &aq[((size_t)i * T + t) * K];
                        for (int k = 0; k < K; ++k) {
                            uint32_t v; std::memcpy(&v, row + 4 * k, 4);
                            off += v != (uint32_t)(uint8_t)q[k] * 0x01010101u;
                        }
                        off += std::memcmp(row + 4 * K, &sa[((size_t)i * T + t) * nkb], nkb * 4) != 0;
                        for (int kb = 0; kb < nkb; ++kb)
                            for (int hh = 0; hh < 2; ++hh) {
                                int sum = 0;
                                for (int k = kb * 128; k < kb * 128 + 128; ++k) sum += ((k % 16) / 4) % 2 == hh ? q[k] : 0;
                                int32_t b; std::memcpy(&b, row + 4 * K + 4 * nkb + 4 * (2 * kb + hh), 4);
                                off += b != 126 * sum;
                            }
                    }
                ok &= expect(off == 0, tag + ": the GEMV's rows differ from the reference (" + std::to_string(off) + ")");
                double md = 0, mr = 0;
                for (int64_t c0 = 0; c0 < N; c0 += 128) {
                    const std::vector<double> ref = mad_ref_block(w, aq, sa, T, c0);
                    for (int t = 0; t < T; ++t)
                        for (int j = 0; j < 128; ++j) {
                            const double rv = ref[(size_t)t * 128 + j], yv = f32_at(o.y, (size_t)t * N + c0 + j);
                            md = std::max(md, std::fabs(yv - rv)); mr = std::max(mr, std::fabs(rv));
                        }
                }
                worst = std::max(worst, md / mr);
                char sb[160];
                std::snprintf(sb, sizeof sb, "%s: y vs the quantised operands' exact GEMM max |d| %.3g of max |y| %.3g (%.2g)", tag.c_str(), md, mr, md / mr);
                ok &= expect(md <= 1e-5 * mr, sb);
            }
            ++n; bad += !ok;
        }
        std::printf("M1 mad GEMV: %d rates ([512, 1024], stacked [512, 512] every third; T 1..8): %d failed (rows byte for byte; "
                    "worst max |d| / max |y| %.3g vs the exact GEMM of the quantised operands)\n", n, bad, worst);
    }
    // M2: the trunk shapes, against the reference on three 128-row blocks, and the A8 path's y on the same rows
    {
        static const struct { const char* name; int N, K; bool st; } SH[] = {
            { "gate_up", 34816, 5120, true }, { "down", 5120, 17408, false }, { "gdn_qkv", 10240, 5120, false },
            { "gdn_z", 6144, 5120, false }, { "attn_qg", 12288, 5120, false }, { "attn_k", 1024, 5120, false },
            { "attn_o", 5120, 6144, false }, { "head", 248320, 5120, false } };
        const std::vector<int> rates = QUICK ? std::vector<int>{ 40, 66 } : std::vector<int>{ 8, 16, 32, 40, 42, 48, 64, 66, 72, 104, 128 };
        int n = 0, bad = 0;
        double worst = 0, worst_a8 = 0;
        for (const auto& sh : SH)
            for (int m : rates) {
                if (!std::strcmp(sh.name, "head") && m != 40 && m != 66) continue;
                Weight w = make_weight(m, EXL3I_CB_MAD, sh.N, sh.K, sh.st);
                for (int T = 1; T <= 8; T += (QUICK ? 7 : (m == 40 || m == 66 ? 1 : 3))) {
                    const std::string tag = std::string("M2 ") + sh.name + " K" + rate_s(m) + " T " + std::to_string(T);
                    const int mats = sh.st ? 2 : 1;
                    Params pa(8, sh.N, sh.K, rate_s(m).c_str(), "mad", "a8g128");
                    Dev d;
                    if (!expect(store_weight(RA, w, pa, &d) == RAD_OK, tag + ": hooks")) { ++bad; continue; }
                    const Run r{T, RAD_BF16, RAD_F32, 0, false};
                    Xin x;
                    make_x(r, sh.K, &x);
                    Out o = launch(RA, pa, w, d, r, x, false);
                    bool ok = expect(o.rc == RAD_OK, tag + ": launch " + std::to_string(o.rc));
                    if (ok) {
                        std::vector<uint16_t> xr = x_rows(x, T, sh.K);
                        std::vector<int8_t> aq((size_t)mats * T * sh.K);
                        std::vector<float> sa((size_t)mats * T * (sh.K / 128));
                        exl3i_ref_actq(xr.data(), sh.K, 1, w.suh.data(), w.suh1.data(), mats, T, sh.K, 128, aq.data(), sa.data());
                        double md = 0, mr = 0;
                        for (int64_t c0 : { (int64_t)0, (int64_t)(sh.N / 256) * 128, (int64_t)sh.N - 128 }) {
                            const std::vector<double> ref = mad_ref_block(w, aq, sa, T, c0);
                            for (int t = 0; t < T; ++t)
                                for (int j = 0; j < 128; ++j) {
                                    const double rv = ref[(size_t)t * 128 + j], yv = f32_at(o.y, (size_t)t * sh.N + c0 + j);
                                    md = std::max(md, std::fabs(yv - rv)); mr = std::max(mr, std::fabs(rv));
                                }
                        }
                        worst = std::max(worst, md / mr);
                        char sb[200];
                        std::snprintf(sb, sizeof sb, "%s: y vs the exact GEMM (3 blocks) max |d| %.3g of %.3g (%.2g)", tag.c_str(), md, mr, md / mr);
                        ok &= expect(md <= 1e-5 * mr, sb);
                        if (T == 1 || T == 8) {   // the A8 path (prefill) on the same rows: the same integers, its own fold
                            test_pin(EXL3I_PIN_A8_MIN_ROWS, 1);
                            Params pa8(T, sh.N, sh.K, rate_s(m).c_str(), "mad", "a8g128");
                            Out o8 = launch(RA, pa8, w, d, r, x, false);
                            test_pin(EXL3I_PIN_A8_MIN_ROWS, 0);
                            double da = 0, ya = 0;
                            for (size_t i = 0; i < (size_t)T * sh.N; ++i) {
                                da = std::max(da, (double)std::fabs(f32_at(o.y, i) - f32_at(o8.y, i)));
                                ya = std::max(ya, (double)std::fabs(f32_at(o8.y, i)));
                            }
                            worst_a8 = std::max(worst_a8, da / ya);
                            ok &= expect(o8.rc == RAD_OK && da <= 2e-3 * ya, tag + ": GEMV vs the A8 path rel " + std::to_string(da / ya));
                        }
                    }
                    ++n; bad += !ok;
                }
            }
        std::printf("M2 mad GEMV, trunk shapes + head x %zu rates x T: %d cases, %d failed (worst max |d| / max |y| %.3g vs the "
                    "exact GEMM on 3 blocks; %.3g vs the A8 path's y)\n", rates.size(), n, bad, worst, worst_a8);
    }
    // M3: rows independent, KS repeatable, the f16 path on the same weight
    {
        int n = 0, bad = 0;
        for (int si = 0; si < 3; ++si) {
            const int N = si == 0 ? 34816 : si == 1 ? 5120 : 1024, K = si == 1 ? 17408 : 5120;
            const bool st = si == 0;
            const int m = si == 2 ? 42 : 40;
            Weight w = make_weight(m, EXL3I_CB_MAD, N, K, st);
            Params pa(8, N, K, rate_s(m).c_str(), "mad", "a8g128");
            Dev d;
            if (!expect(store_weight(RA, w, pa, &d) == RAD_OK, "M3 hooks")) continue;
            const Run r8{8, RAD_BF16, RAD_F32, 0, false};
            Xin x;
            make_x(r8, K, &x);
            Out o8 = launch(RA, pa, w, d, r8, x, false);
            Out o20 = launch(RA, pa, w, d, r8, x, false, 0, 20);
            bool ok = expect(o8.rc == RAD_OK && o20.rc == RAD_OK && o8.y == o20.y, "M3 N " + std::to_string(N) + ": 20 repeats byte-identical");
            int rd = 0;
            for (int t = 0; t < 8; ++t) {
                Xin x1;
                x1.pitch = x.pitch; x1.raw.assign(x.raw.begin() + (size_t)t * x.pitch, x.raw.begin() + (size_t)(t + 1) * x.pitch);
                x1.dev = upload(x1.raw.data(), x1.raw.size() * 2);
                const Run r1{1, RAD_BF16, RAD_F32, 0, false};
                Out o1 = launch(RA, pa, w, d, r1, x1, false);
                rd += o1.rc != RAD_OK || std::memcmp(o1.y.data(), o8.y.data() + (size_t)t * N * 4, (size_t)N * 4) != 0;
            }
            ok &= expect(rd == 0, "M3 N " + std::to_string(N) + ": each row of the 8-row launch = its 1-row launch (" + std::to_string(rd) + " differ)");
            test_pin(EXL3I_PIN_MAD_GEMV_OFF, 1);
            Out of = launch(RA, pa, w, d, r8, x, false);
            test_pin(EXL3I_PIN_MAD_GEMV_OFF, 0);
            double se = 0, sr = 0;
            for (size_t i = 0; i < (size_t)8 * N; ++i) { const double a = f32_at(o8.y, i), b = f32_at(of.y, i); se += (a - b) * (a - b); sr += b * b; }
            const double rel = std::sqrt(se / sr);
            ok &= expect(of.rc == RAD_OK && rel < 2.5e-2, "M3 N " + std::to_string(N) + ": vs the f16 path rel rms " + std::to_string(rel));
            std::printf("  M3 N %d K %d: GEMV vs the f16 path (f16 activations) rel rms %.3g\n", N, K, rel);
            ++n; bad += !ok;
        }
        std::printf("M3 mad GEMV: %d shapes, %d failed (rows independent bit for bit, KS repeatable, the f16 path within the int8 "
                    "activations' error)\n", n, bad);
    }
    test_pin(EXL3I_PIN_MAD_GEMV_T, 0);
    {   // M4: the default rule without the producer's rows: the GEMV at 1 row (2 below K4) on a trunk shape at K <= 2,
        // else the f16 path, byte for byte the MAD_GEMV_OFF pin's
        int bad = 0;
        for (int m : { 32, 42, 66 })
            for (int T : { 1, 2, 3 }) {
                Weight w = make_weight(m, EXL3I_CB_MAD, 1024, 5120, false);
                Params pa(8, 1024, 5120, rate_s(m).c_str(), "mad", "a8g128");
                Dev d;
                if (!expect(store_weight(RA, w, pa, &d) == RAD_OK, "M4 hooks")) continue;
                const Run r{T, RAD_BF16, RAD_F32, 0, false};
                Xin x;
                make_x(r, 5120, &x);
                Out o = launch(RA, pa, w, d, r, x, false);
                test_pin(EXL3I_PIN_MAD_GEMV_OFF, 1);
                Out of = launch(RA, pa, w, d, r, x, false);
                test_pin(EXL3I_PIN_MAD_GEMV_OFF, 0);
                const bool gemv = m <= 32 && (T == 1 || (T == 2 && m < 64));
                bad += !expect(o.rc == RAD_OK && of.rc == RAD_OK && (o.y == of.y) == !gemv,
                               "M4 K" + rate_s(m) + " T " + std::to_string(T) + (gemv ? ": the GEMV" : ": the f16 path"));
            }
        std::printf("M4 mad dispatch by rows: %d wrong\n", bad);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_twopass EXL3I.so [--quick] [--mad] [--nofuse]\n"); return 2; }
    bool mad_only = false, nofuse = false;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--quick")) QUICK = true;
        if (!std::strcmp(argv[i], "--mad")) mad_only = true;     // section M alone
        if (!std::strcmp(argv[i], "--nofuse")) nofuse = true;   // the A8 path's two passes, not the fused GEMM
    }
    open_plugin(argv[1]);
    if (nofuse) test_pin(EXL3I_PIN_A8_FUSE_OFF, 1);
    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("exl3i %s -- %s (%s); device %s %s%s\n", INFO->version, INFO->description, INFO->build_target, prop.name,
                prop.gcnArchName, QUICK ? " [quick]" : "");
    if (!mad_only) {
        test_a8_planes();
        test_t2_planes();
        test_matrix();
        test_refusals();
    }
    test_mad();
    if (!CHECKED) { std::printf("TWOPASS GPU TEST FAIL (0 checks)\n"); return 1; }
    std::printf("TWOPASS GPU TEST %s (%d checks", FAILS ? "FAIL" : "PASS", CHECKED);
    if (FAILS) std::printf(", %d failed", FAILS);
    std::printf(")\n");
    return FAILS != 0;
}
