// exl3i_gpu_test.hip -- the exl3i row's device selftest: the row through its plugin ABI (layout / relayout hooks,
// scratch hook, launch) against the C oracle (common/exl3i_decode.c), plus the decode policies alone. Usage:
//   exl3i_gpu_test EXL3I.so
// Sections (each prints its own line; the last line is PASS or FAIL with the count):
//   A  decode: every rate m = 2..128 (even) x mad (and its madh form), the policy the launcher picks (RateInt / RateFrac),
//      every lane's fragment of 128 random tiles, bit for bit against exl3i_decode_tile; mad's madh (the byte-sum GEMV's
//      hashes) too: bytesum - 126 = the oracle value's grid index at every position
//   B  GEMM at every rate (mad), [256, 256], M = 1 (the decode band) and M = 17 (MT tiles):
//      y against x . W^T
//      in double (W the oracle's decode); row 0 of the M = 1 run (the decode form: DOT, and DQ on mad) within 1/1024
//      of max |y| of the M = 17 run's (MT 4)
//   C  shapes and dtypes on a rate sample: M in {1, 15, 16, 17, 511, 512, 513, 952} (the T2 threshold's
//      neighbours), y bf16 / f16 / f32, a f16 / bf16 (a padded row pitch), the stacked seam (suh.1), the GDN
//      ssm_out input [T, 48, 128] read as [T, 6144]
//   D  T2 (two-pass prefill) and nt_pass byte-identical at M = 600 (the T2_OFF pin)
//   F  refusals: bad rate / act strings, a rate that disagrees with the encoding, suh.1 with N % 256,
//      operands of the wrong dtype or shape
// The plugin's build id (rad_plugin_info) is printed first: a result belongs to that build.
#include <hip/hip_runtime.h>
#include "exl3i_rate.hip.h"
#include "exl3i.h"
#include "exl3i_dev.h"
#include "exl3i_pin.h"
#include "exl3i_decode.h"
#include "exl3_decode.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#define HIPCHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

static void dfree(void* p) { if (p) (void)hipFree(p); }

static int FAILS = 0;
static void fail(const std::string& what) { ++FAILS; std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }

static uint64_t RNG = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() { RNG ^= RNG << 13; RNG ^= RNG >> 7; RNG ^= RNG << 17; return (uint32_t)(RNG >> 16); }
static float rndn() {   // ~N(0, 1)
    float s = 0; for (int i = 0; i < 4; ++i) s += (rnd() & 0xFFFF) / 65536.0f; return (s - 2.0f) * 1.7320508f;
}

static uint16_t f2h(float f) { _Float16 h = (_Float16)f; uint16_t u; std::memcpy(&u, &h, 2); return u; }
static float h2f(uint16_t u) { _Float16 h; std::memcpy(&h, &u, 2); return (float)h; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

// ============================================================================= A: the decode policies alone
template <class R>
__global__ void k_dec(const unsigned* words, int m, unsigned short* out) {
    const int lane = threadIdx.x, tile = blockIdx.x;
    const R P(lane, m);
    const unsigned* tw = words + (size_t)tile * R::words(m);
    unsigned w[2][R::NW];
    for (int r = 0; r < 2; r++)
        for (int x = 0; x < R::NW; x++) w[r][x] = tw[P.woff[r][x]];
    union { v8h f; unsigned short h[8]; } u;
    u.f = P.decode(w);
    const int n = lane & 15, hh = lane >> 4, base = 32 * (n & 7) + 4 * (n >> 3) + 16 * hh;
    for (int e = 0; e < 8; e++) out[(size_t)tile * 256 + base + 8 * ((e >> 1) & 1) + (e & 1) + 2 * (e >> 2)] = u.h[e];
}

// mad: madh's eight hashes, each as bytesum - 126 (the GEMV's q), at the same positions
template <class R>
__global__ void k_madh(const unsigned* words, int m, unsigned short* out) {
    const int lane = threadIdx.x, tile = blockIdx.x;
    const R P(lane, m);
    const unsigned* tw = words + (size_t)tile * R::words(m);
    unsigned w[2][R::NW], hh[8];
    for (int r = 0; r < 2; r++)
        for (int x = 0; x < R::NW; x++) w[r][x] = tw[P.woff[r][x]];
    P.madh(w, hh);
    const int n = lane & 15, h = lane >> 4, base = 32 * (n & 7) + 4 * (n >> 3) + 16 * h;
    for (int e = 0; e < 8; e++)
        out[(size_t)tile * 256 + base + 8 * ((e >> 1) & 1) + (e & 1) + 2 * (e >> 2)] =
            (unsigned short)(__builtin_amdgcn_sudot4(false, (int)hh[e], false, 0x01010101, 0, false) - 126);
}

template <class R> struct k_dec_t { static const void* f() { return (const void*)k_dec<R>; } };
template <class R> struct k_madh_t { static const void* f() { return (const void*)k_madh<R>; } };
template <int CB, template <class> class KF>
static const void* dec_fn(int m) {   // the launcher's choice (exl3i_nt.hip.h exl3i_launch_cb)
    if (m % 16 == 0) switch (m / 16) {
        case 1: return KF<RateInt<1, CB>>::f();
        case 2: return KF<RateInt<2, CB>>::f();
        case 3: return KF<RateInt<3, CB>>::f();
        case 4: return KF<RateInt<4, CB>>::f();
        case 5: return KF<RateInt<5, CB>>::f();
        case 6: return KF<RateInt<6, CB>>::f();
        case 7: return KF<RateInt<7, CB>>::f();
        default: return KF<RateInt<8, CB>>::f();
    }
    if (m < 16) return KF<RateFrac<CB, 1>>::f();
    if (m < 80) return KF<RateFrac<CB, 0>>::f();
    return KF<RateFrac<CB, 2>>::f();
}

static void test_decode() {
    const int T = 128;
    unsigned* dw; unsigned short* dout;
    HIPCHECK(hipMalloc(&dw, (size_t)T * 64 * 4));
    HIPCHECK(hipMalloc(&dout, (size_t)T * 256 * 2));
    long bad = 0, n = 0, badq = 0;
    for (int form = 0; form < 2; ++form)   // 0: the decode (fp16 values); 1: mad's madh (the masked-hash form)
        for (int m = 2; m <= 128; m += 2) {
            const int W = m / 2;
            std::vector<uint32_t> h((size_t)T * W);
            for (auto& x : h) x = rnd() ^ (rnd() << 16);
            HIPCHECK(hipMemcpy(dw, h.data(), h.size() * 4, hipMemcpyHostToDevice));
            void* args[] = { &dw, (void*)&m, &dout };
            const void* fn = form ? dec_fn<EXL3I_DEV_CB_MAD, k_madh_t>(m) : dec_fn<EXL3I_DEV_CB_MAD, k_dec_t>(m);
            HIPCHECK(hipLaunchKernel(fn, dim3(T), dim3(32), args, 0, 0));
            std::vector<uint16_t> o((size_t)T * 256);
            HIPCHECK(hipMemcpy(o.data(), dout, o.size() * 2, hipMemcpyDeviceToHost));
            long b = 0;
            for (int t = 0; t < T; ++t) {
                uint16_t va[256];
                exl3i_decode_tile(&h[(size_t)t * W], m, EXL3I_CB_MAD, va, nullptr);
                for (int p = 0; p < 256; ++p)
                    b += form ? (int)(int16_t)o[(size_t)t * 256 + p] != exl3i_a8_wq(h2f(va[p])) : o[(size_t)t * 256 + p] != va[p];
            }
            if (b) { char s[96]; std::snprintf(s, sizeof s, "A decode m %d form %d: %ld values differ", m, form, b); fail(s); }
            (form ? badq : bad) += b; n += form ? 0 : (long)T * 256;
        }
    dfree(dw); dfree(dout);
    std::printf("A decode: %ld values over 64 rates x mad, %ld differ; mad's madh: %ld of %ld grid indices differ\n", n, bad,
                badq, 64L * T * 256);
}

// ============================================================================= the row through its ABI
struct Plugin {
    void* h = nullptr;
    const RadKernelInfo* row = nullptr;
    const RadPluginInfo* info = nullptr;
};

static Plugin open_plugin(const char* path, const char* op) {
    Plugin P;
    P.h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!P.h) { std::fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); std::exit(2); }
    auto ver = (uint32_t (*)())dlsym(P.h, "rad_plugin_abi_version");
    auto inf = (const RadPluginInfo* (*)())dlsym(P.h, "rad_plugin_info");
    auto cnt = (int (*)())dlsym(P.h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(P.h, "rad_kernel_at");
    if (!ver || !inf || !cnt || !at || ver() != RAD_ABI_VERSION) { std::fprintf(stderr, "%s: not an ABI %d plugin\n", path, RAD_ABI_VERSION); std::exit(2); }
    P.info = inf();
    for (int i = 0; i < cnt(); ++i)   // the first row of the op: exl3i's f16 row (exl3i_gemm_nt_a8 follows it)
        if (!std::strcmp(at(i)->op, op) && !P.row) P.row = at(i);
    if (!P.row) { std::fprintf(stderr, "%s: no row for %s\n", path, op); std::exit(2); }
    return P;
}

static Plugin PI;   // exl3i
static void (*PIN)(int, int64_t) = nullptr;   // the plugin's exl3i_test_pin (exl3i_pin.h)
static void test_pin(int k, int64_t v) { PIN(k, v); }

struct Weight {   // host copies
    int m, cb, N, K;
    bool stacked;
    std::vector<uint32_t> rec;   // the container's record plane: [N/16][(K/16) m/2]
    std::vector<uint16_t> suh, suh1, svh;
    std::vector<float> W;        // the oracle's decode, [N][K]
};

static Weight make_weight(int m, int cb, int N, int K, bool stacked, bool decode = true) {
    Weight w{m, cb, N, K, stacked, {}, {}, {}, {}, {}};
    w.rec.resize((size_t)(N / 16) * exl3i_row_words(K, m));
    for (auto& x : w.rec) x = rnd() ^ (rnd() << 16);
    auto scales = [](std::vector<uint16_t>& v, int n) {   // signs x a scale near 1 (as the quantizer's)
        v.resize(n);
        for (auto& x : v) x = f2h((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f));
    };
    scales(w.suh, K); scales(w.svh, N);
    if (stacked) scales(w.suh1, K);
    if (!decode) return w;   // a weight only the refusals see (the oracle wants a stacked weight's rows in 256s)
    w.W.resize((size_t)N * K);
    Exl3iView v{};
    v.rows = N; v.cols = K; v.m = m; v.cb = cb; v.record = w.rec.data(); v.suh = w.suh.data();
    v.suh1 = stacked ? w.suh1.data() : nullptr; v.svh = w.svh.data();
    if (exl3i_decode_rows(&v, 0, N, w.W.data()) != 0) { std::fprintf(stderr, "oracle decode failed\n"); std::exit(2); }
    return w;
}

struct Params {
    char rate[16], codebook[16], act[16];
    std::vector<RadParam> p;
    Params(int M, int N, int K, const char* rt, const char* cb, const char* ac) {
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

struct Dev {   // the weight's operands on the device, as the row stores them
    void *rec = nullptr, *suh = nullptr, *suh1 = nullptr, *svh = nullptr;
    ~Dev() { dfree(rec); dfree(suh); dfree(suh1); dfree(svh); }
};

static void* upload(const void* h, size_t n) { void* d; HIPCHECK(hipMalloc(&d, n)); HIPCHECK(hipMemcpy(d, h, n, hipMemcpyHostToDevice)); return d; }

// The weight through exl3i's layout and relayout hooks (operand by operand, as the engine's loader does).
static int store_weight(const Weight& w, const Params& pa, Dev* d) {
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
        int rc = PI.row->layout(pa.p.data(), (int)pa.p.size(), ops[i].opd, &e, &sel, &pl, 1, &L);
        if (rc != RAD_OK) return rc;
        std::vector<char> buf((size_t)L.bytes);
        rc = PI.row->relayout(pa.p.data(), (int)pa.p.size(), ops[i].opd, &e, &sel, &pl, 1, buf.data(), L.bytes);
        if (rc != RAD_OK) return rc;
        // unrelayout gives the plane back exactly
        std::vector<char> back((size_t)rad_enc_row_bytes(ops[i].dt, ops[i].c) * ops[i].r);
        RadTensor pb = pl; pb.data = back.data();
        rc = PI.row->unrelayout(pa.p.data(), (int)pa.p.size(), ops[i].opd, &e, &sel, buf.data(), L.bytes, &pb, 1);
        if (rc != RAD_OK || std::memcmp(back.data(), ops[i].data, back.size())) fail("unrelayout does not give the plane back");
        *ops[i].out = upload(buf.data(), buf.size());
    }
    return RAD_OK;
}

struct Run {
    int M; uint32_t xdt, ydt; int xpad; bool x3d;
};

// One launch through the row. y comes back as raw bytes [M][N] of ydt. Returns the launch's status.
static int launch(const Plugin& P, const Params& pa, const Weight& w, void* rec, void* suh, void* suh1, void* svh,
                  const Run& r, const void* dx, std::vector<char>* y, uint32_t rec_dt, int64_t rec_r, int64_t rec_c) {
    const int64_t xp = w.K + r.xpad;
    RadTensor t[6]{};
    t[0] = tensor((void*)dx, r.xdt, r.M, w.K, xp);
    if (r.x3d) { t[0].rank = 3; t[0].shape[1] = w.K / 128; t[0].shape[2] = 128; t[0].stride[0] = xp; t[0].stride[1] = 128; t[0].stride[2] = 1; }
    t[1] = tensor(rec, rec_dt, rec_r, rec_c);
    t[2] = tensor(suh, RAD_F16, 1, w.K);
    t[3] = tensor(svh, RAD_F16, w.N, 1);
    t[4] = tensor(suh1, RAD_F16, 1, w.K);
    if (!suh1) t[4].data = nullptr;
    const size_t yb = (size_t)r.M * w.N * rad_dtype_bytes(r.ydt, 1);
    void* dy; HIPCHECK(hipMalloc(&dy, yb)); HIPCHECK(hipMemset(dy, 0xCD, yb));
    t[5] = tensor(dy, r.ydt, r.M, w.N);
    RadArgs a{};
    a.t = t; a.n_t = 6; a.p = pa.p.data(); a.n_p = (int)pa.p.size(); a.world_size = 1;
    const int64_t sb = P.row->scratch ? P.row->scratch(&a) : 0;
    void* sc = nullptr;
    if (sb > 0) HIPCHECK(hipMalloc(&sc, (size_t)sb));
    a.scratch = sc; a.scratch_bytes = sb;
    const int rc = P.row->launch(&a, nullptr);
    HIPCHECK(hipDeviceSynchronize());
    y->resize(yb);
    HIPCHECK(hipMemcpy(y->data(), dy, yb, hipMemcpyDeviceToHost));
    dfree(dy); dfree(sc);
    return rc;
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

static const char* dtn(uint32_t dt) { return dt == RAD_F32 ? "f32" : dt == RAD_F16 ? "f16" : dt == RAD_BF16 ? "bf16" : "?"; }

// x on the host (as floats the device sees) and the device copy
static std::vector<float> make_x(const Run& r, int K, void** dx) {
    const int64_t xp = K + r.xpad;
    std::vector<uint16_t> raw((size_t)r.M * xp, 0);
    std::vector<float> xf((size_t)r.M * K);
    for (int i = 0; i < r.M; ++i)
        for (int k = 0; k < K; ++k) {
            const float v = rndn();
            const uint16_t u = r.xdt == RAD_F16 ? f2h(v) : f2bf(v);
            raw[(size_t)i * xp + k] = u;
            xf[(size_t)i * K + k] = r.xdt == RAD_F16 ? h2f(u) : bf2f(u);
        }
    *dx = upload(raw.data(), raw.size() * 2);
    return xf;
}

// y against x . W^T in double: rms of the error over rms of y, and the worst element over its tolerance
// (one ulp of the output type at |y| + 4e-3 rms(y): the device rounds the rotated rows to f16)
static bool check_y(const Weight& w, const std::vector<float>& xf, const Run& r, const std::vector<char>& y,
                    const std::string& tag, double* rel_out = nullptr) {
    double se = 0, sr = 0;
    std::vector<double> ref((size_t)r.M * w.N);
    for (int i = 0; i < r.M; ++i)
        for (int n = 0; n < w.N; ++n) {
            double s = 0;
            const float* xr = &xf[(size_t)i * w.K];
            const float* wr = &w.W[(size_t)n * w.K];
            for (int k = 0; k < w.K; ++k) s += (double)xr[k] * wr[k];
            ref[(size_t)i * w.N + n] = s;
            sr += s * s;
        }
    const double rms = std::sqrt(sr / ref.size());
    double worst = 0;
    bool finite = true;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double v = out_val(y, r.ydt, i), d = v - ref[i];
        if (!std::isfinite(v)) finite = false;
        se += d * d;
        const double q = std::fabs(d) / (ulp_of(r.ydt, ref[i]) + 4e-3 * rms);
        if (q > worst) worst = q;
    }
    const double rel = std::sqrt(se / ref.size()) / rms;
    const double lim = r.ydt == RAD_BF16 ? 4e-3 : 1.5e-3;
    const bool ok = finite && rel < lim && worst <= 1.0;
    if (rel_out) *rel_out = rel;
    if (!ok) {
        char s[256];
        std::snprintf(s, sizeof s, "%s: rel rms err %.3g (limit %.3g), worst element %.3g x its tolerance%s", tag.c_str(),
                      rel, lim, worst, finite ? "" : ", NOT FINITE");
        fail(s);
    }
    return ok;
}

static std::string rate_s(int m) { char b[16]; return exl3i_rate_str(m, b); }

// One case end to end: weight -> hooks -> launch -> check. y returned for byte comparisons.
static bool gemm_case(int m, int cb, int N, int K, bool stacked, const Run& r, std::vector<char>* yout = nullptr,
                      Weight* wkeep = nullptr, std::vector<float>* xkeep = nullptr, void** dxkeep = nullptr, double* rel = nullptr) {
    Weight w = make_weight(m, cb, N, K, stacked);
    Params pa(r.M, N, K, rate_s(m).c_str(), exl3i_codebook_name(cb), "f16");
    Dev d;
    int rc = store_weight(w, pa, &d);
    char tag[160];
    std::snprintf(tag, sizeof tag, "m %d (K%s) %s [%d, %d]%s M %d a %s y %s%s%s", m, rate_s(m).c_str(), exl3i_codebook_name(cb),
                  N, K, stacked ? " stacked" : "", r.M, dtn(r.xdt), dtn(r.ydt), r.xpad ? " padded" : "", r.x3d ? " 3-D" : "");
    if (rc != RAD_OK) { fail(std::string(tag) + ": the layout hooks refused (" + std::to_string(rc) + ")"); return false; }
    void* dx = nullptr;
    std::vector<float> xf = make_x(r, K, &dx);
    std::vector<char> y;
    rc = launch(PI, pa, w, d.rec, d.suh, stacked ? d.suh1 : nullptr, d.svh, r, dx, &y, EXL3I_DT_TILES, N / 16,
                exl3i_row_words(K, m));
    bool ok = rc == RAD_OK;
    if (!ok) fail(std::string(tag) + ": launch returned " + std::to_string(rc));
    else ok = check_y(w, xf, r, y, tag, rel);
    if (yout) *yout = y;
    if (wkeep) *wkeep = w;
    if (xkeep) *xkeep = xf;
    if (dxkeep) *dxkeep = dx; else dfree(dx);
    return ok;
}

// ============================================================================= B: every rate x codebook
static void test_rates() {
    int n = 0, bad = 0;
    double worst = 0, dq_worst = 0;
    for (int cb : { EXL3I_CB_MAD })
        for (int m = 2; m <= 128; m += 2) {
            const uint64_t seed = RNG;
            std::vector<char> y1, y17;
            double r1 = 0, r17 = 0;
            const bool a = gemm_case(m, cb, 256, 256, false, Run{1, RAD_F16, RAD_F16, 0, false}, &y1, nullptr, nullptr, nullptr, &r1);
            RNG = seed;   // the same weight and the same first row of x
            const bool b = gemm_case(m, cb, 256, 256, false, Run{17, RAD_F16, RAD_F16, 0, false}, &y17, nullptr, nullptr, nullptr, &r17);
            bool same = y1.size() && y17.size() >= y1.size() && !std::memcmp(y1.data(), y17.data(), y1.size());
            if (!same && y1.size() && y17.size() >= y1.size()) {
                // M 1 is the decode form's DOT (v_dot2 sums, another order than MT 4's WMMA), on mad also DQ (the grid
                // indices, the sums scaled once): row 0 within one f16 ulp of the output's scale of MT 4's
                double mx = 0, ref = 0;
                for (size_t i = 0; i < y1.size() / 2; ++i) {
                    uint16_t u1, u17;
                    std::memcpy(&u1, y1.data() + 2 * i, 2); std::memcpy(&u17, y17.data() + 2 * i, 2);
                    mx = std::max(mx, (double)std::fabs(h2f(u1) - h2f(u17))); ref = std::max(ref, (double)std::fabs(h2f(u17)));
                }
                same = mx <= ref / 1024;
                if (mx > dq_worst * ref) dq_worst = ref > 0 ? mx / ref : 0;
            }
            if (!same) fail("B m " + std::to_string(m) + " cb " + std::to_string(cb) + ": row 0 differs between M = 1 and M = 17");
            n += 2; bad += !a + !b + !same;
            worst = std::max(worst, std::max(r1, r17));
        }
    std::printf("B every rate: %d GEMMs (64 rates x mad x M 1, 17), %d failed, worst rel rms err %.3g; MT 1 = MT 4 row bytes "
                "(else within %.3g of max |y|: the decode form's DOT / DQ)\n", n, bad, worst, dq_worst);
}

// ============================================================================= C: shapes and dtypes
static void test_shapes() {
    const int rates[] = { 2, 8, 14, 20, 36, 40, 48, 62, 64, 84, 100, 112, 126, 128 };
    const int Ms[] = { 1, 15, 16, 17, 511, 512, 513, 952 };
    const uint32_t ydts[] = { RAD_BF16, RAD_F16, RAD_F32 };
    int n = 0, bad = 0, i = 0;
    for (int m : rates)
        for (int j = 0; j < 3; ++j, ++i) {
            const int M = Ms[i % 8];
            const Run r{M, (i % 2) ? RAD_BF16 : RAD_F16, ydts[i % 3], (i % 4 == 1) ? 64 : 0, false};
            const bool stacked = j == 1;
            const int N = j == 2 ? 384 : 512, K = j == 0 ? 1024 : 256;
            bad += !gemm_case(m, EXL3I_CB_MAD, N, K, stacked, r);
            ++n;
        }
    // GDN ssm_out: [T, 48, 128] read as [T, 6144]; and a padded 3-D pitch
    for (int m : { 40, 64, 100 }) {
        bad += !gemm_case(m, EXL3I_CB_MAD, 256, 6144, false, Run{33, RAD_BF16, RAD_BF16, 0, true});
        bad += !gemm_case(m, EXL3I_CB_MAD, 256, 6144, false, Run{600, RAD_BF16, RAD_BF16, 0, true});
        n += 2;
    }
    std::printf("C shapes and dtypes: %d GEMMs (M 1..952 across the T2 threshold, y bf16 / f16 / f32, a f16 / bf16, "
                "padded pitch, stacked, GDN 3-D), %d failed\n", n, bad);
}

// ============================================================================= D: T2 and nt_pass byte-identical
static void test_t2() {
    int n = 0, bad = 0;
    for (int m : { 8, 36, 40, 64, 100, 128 })
        for (int cb : { EXL3I_CB_MAD }) {
            const uint64_t seed = RNG;
            std::vector<char> ya, yb;
            test_pin(EXL3I_PIN_T2_OFF, 0);
            const bool a = gemm_case(m, cb, 512, 512, cb == EXL3I_CB_MAD, Run{600, RAD_F16, RAD_BF16, 0, false}, &ya);
            RNG = seed;
            test_pin(EXL3I_PIN_T2_OFF, 1);
            const bool b = gemm_case(m, cb, 512, 512, cb == EXL3I_CB_MAD, Run{600, RAD_F16, RAD_BF16, 0, false}, &yb);
            test_pin(EXL3I_PIN_T2_OFF, 0);
            const bool same = a && b && ya == yb;
            if (!same) fail("D m " + std::to_string(m) + " cb " + std::to_string(cb) + ": T2 and nt_pass differ");
            bad += !same; ++n;
        }
    std::printf("D T2 vs nt_pass at M 600: %d pairs, %d not byte-identical\n", n, bad);
}

// ============================================================================= G: TP slices
// The op geometry is the rank's (rows over `world` for a row shard, K for a column shard), the planes the
// core's dense cut of the container's (the record by 16-row groups, by word per record row; suh / suh.1 / svh
// along their axes). The hooks accept the slice and transpose it (the tiles are block-local, so the slice's
// transpose is the transposed slice), and the launch computes the slice's rows (row shard) or the partial dot
// over the rank's K (column shard) of the full oracle.
static void test_tp_slices() {
    const int m = 40, cb = EXL3I_CB_MAD, N = 512, K = 512, W = 2;
    Weight w = make_weight(m, cb, N, K, false);
    RadEncoding e;
    exl3i_encoding_make(m, cb, 0, N, K, &e);
    const int64_t words = exl3i_row_words(K, m);
    int n = 0, bad = 0;
    for (bool col : { false, true })
    for (int rank : { 0, 1 }) {
        const int64_t nr = col ? N : N / W;   // the rank's rows
        const int64_t nk = col ? K / W : K;   // the rank's K
        // the core's dense share of each plane
        std::vector<uint32_t> rec((size_t)(nr / 16) * exl3i_row_words(nk, m));
        if (col)
            for (int64_t t = 0; t < N / 16; ++t)
                std::memcpy(rec.data() + t * exl3i_row_words(nk, m), w.rec.data() + t * words + rank * exl3i_row_words(nk, m),
                            exl3i_row_words(nk, m) * 4);
        else
            std::memcpy(rec.data(), w.rec.data() + rank * (nr / 16) * words, rec.size() * 4);
        std::vector<uint16_t> suh(col ? w.suh.begin() + rank * (K / W) : w.suh.begin(),
                                  col ? w.suh.begin() + (rank + 1) * (K / W) : w.suh.end());
        std::vector<uint16_t> svh(col ? w.svh.begin() : w.svh.begin() + rank * (N / W),
                                  col ? w.svh.end() : w.svh.begin() + (rank + 1) * (N / W));
        Params pa(1, nr, nk, rate_s(m).c_str(), exl3i_codebook_name(cb), "f16");
        // the hooks over the slice, operand by operand (as the loader runs them)
        struct { int opd, plane; const void* data; uint32_t dt; int64_t r, c; } ops[3] = {
            { EXL3I_OPD_RECORD, EXL3I_PLANE_RECORD, rec.data(), RAD_U32, nr / 16, exl3i_row_words(nk, m) },
            { EXL3I_OPD_SUH, EXL3I_PLANE_SUH, suh.data(), RAD_F16, 1, nk },
            { EXL3I_OPD_SVH, EXL3I_PLANE_SVH, svh.data(), RAD_F16, nr, 1 },
        };
        void* d[3] = { nullptr, nullptr, nullptr };
        bool ok = true;
        for (int i = 0; i < 3 && ok; ++i) {
            RadTensor pl = tensor((void*)ops[i].data, ops[i].dt, ops[i].r, ops[i].c);
            RadLayout L;
            int sel = ops[i].plane;
            if (PI.row->layout(pa.p.data(), (int)pa.p.size(), ops[i].opd, &e, &sel, &pl, 1, &L) != RAD_OK) { ok = false; break; }
            std::vector<char> buf((size_t)L.bytes);
            if (PI.row->relayout(pa.p.data(), (int)pa.p.size(), ops[i].opd, &e, &sel, &pl, 1, buf.data(), L.bytes) != RAD_OK) { ok = false; break; }
            std::vector<char> back(ops[i].r * rad_enc_row_bytes(ops[i].dt, ops[i].c));
            RadTensor pb = pl; pb.data = back.data();
            if (PI.row->unrelayout(pa.p.data(), (int)pa.p.size(), ops[i].opd, &e, &sel, buf.data(), L.bytes, &pb, 1) != RAD_OK) { ok = false; break; }
            if (std::memcmp(back.data(), ops[i].data, back.size())) { ok = false; break; }
            d[i] = upload(buf.data(), buf.size());
        }
        char tag[160];
        std::snprintf(tag, sizeof tag, "G %s shard rank %d [%lld, %lld]", col ? "col" : "row", rank, (long long)nr, (long long)nk);
        if (!ok) { fail(std::string(tag) + ": the layout hooks refused the slice"); ++bad; continue; }
        // the launch: the row shard reads the full x and writes the rank's rows; the column shard reads the
        // rank's own input (its per-rank K) and writes the partial over its K
        Run r{ 1, RAD_F16, RAD_F16, 0, false };
        void* dx = nullptr;
        std::vector<float> xf = make_x(r, (int)nk, &dx);
        Weight w2 = w; w2.N = nr; w2.K = nk;
        std::vector<char> y;
        int rc = launch(PI, pa, w2, d[0], d[1], nullptr, d[2], r, dx, &y, EXL3I_DT_TILES, nr / 16, exl3i_row_words(nk, m));
        dfree(d[0]); dfree(d[1]); dfree(d[2]); dfree(dx);
        if (rc != RAD_OK) { fail(std::string(tag) + ": the launch refused"); ++bad; continue; }
        double se = 0, sr = 0;
        std::vector<double> ref((size_t)nr);
        for (int64_t i = 0; i < nr; ++i) {
            double s = 0;
            for (int k = 0; k < nk; ++k)
                s += (double)xf[k] * w.W[(size_t)(col ? i : rank * nr + i) * K + (col ? rank * nk + k : k)];
            ref[i] = s; sr += s * s;
        }
        const double rms = std::sqrt(sr / ref.size());
        for (size_t i = 0; i < ref.size(); ++i) {
            const float v = out_val(y, RAD_F16, i);
            se += (double)(v - ref[i]) * (v - ref[i]);
        }
        const double rel = std::sqrt(se / ref.size()) / rms;
        if (rel > 2e-3) { char s2[128]; std::snprintf(s2, sizeof s2, "%s: rel rms err %.3g", tag, rel); fail(s2); ++bad; }
        ++n;
    }
    std::printf("G TP slices: %d launches (row / column shard, ranks 0 / 1), %d failed\n", n, bad);
}

// ============================================================================= F: refusals
static void test_refusals() {
    int n = 0, bad = 0;
    auto expect = [&](int got, int want, const char* what) {
        ++n;
        if (got != want) { ++bad; fail(std::string("F ") + what + ": got " + std::to_string(got) + ", want " + std::to_string(want)); }
    };
    Exl3iGeom g;
    { Params p(1, 256, 256, "2.3", "mad", "f16"); expect(exl3i_geom_parse(p.p.data(), 6, &g), RAD_E_DTYPE, "rate 2.3"); }
    { Params p(1, 256, 256, "2.50", "mad", "f16"); expect(exl3i_geom_parse(p.p.data(), 6, &g), RAD_E_DTYPE, "rate 2.50 (not canonical)"); }
    { Params p(1, 256, 256, "8.125", "mad", "f16"); expect(exl3i_geom_parse(p.p.data(), 6, &g), RAD_E_DTYPE, "rate 8.125"); }
    { Params p(1, 256, 256, "2.5", "other", "f16"); expect(exl3i_geom_parse(p.p.data(), 6, &g), RAD_E_DTYPE, "a codebook other than mad"); }
    { Params p(1, 256, 256, "2.5", "mad", "a8g100"); expect(exl3i_geom_parse(p.p.data(), 6, &g), RAD_E_DTYPE, "act a8g100"); }
    { Params p(1, 256, 200, "2.5", "mad", "f16"); expect(exl3i_geom_parse(p.p.data(), 6, &g), RAD_E_SHAPE, "K 200"); }
    { Params p(1, 256, 256, "2.5", "mad", "f16"); expect(exl3i_geom_parse(p.p.data(), 4, &g), RAD_E_SCHEMA, "no codebook"); }
    // the layout hook: the op's rate or codebook against the encoding's
    Weight w = make_weight(40, EXL3I_CB_MAD, 256, 256, false);
    RadEncoding e;
    exl3i_encoding_make(40, EXL3I_CB_MAD, 0, 256, 256, &e);
    RadTensor pl = tensor(w.rec.data(), RAD_U32, 16, exl3i_row_words(256, 40));
    RadLayout L;
    int sel = EXL3I_PLANE_RECORD;
    { Params p(1, 256, 256, "2.25", "mad", "f16"); expect(PI.row->layout(p.p.data(), 6, EXL3I_OPD_RECORD, &e, &sel, &pl, 1, &L), RAD_E_DTYPE, "layout: op rate 2.25, encoding 2.5"); }
    { Params p(1, 256, 256, "2.5", "other", "f16"); expect(PI.row->layout(p.p.data(), 6, EXL3I_OPD_RECORD, &e, &sel, &pl, 1, &L), RAD_E_SHAPE, "layout: op codebook other than mad: the parse names none"); }
    { Params p(1, 512, 256, "2.5", "mad", "f16"); expect(PI.row->layout(p.p.data(), 6, EXL3I_OPD_RECORD, &e, &sel, &pl, 1, &L), RAD_E_SHAPE, "layout: N 512 on a [256, 256] weight"); }
    { Params p(1, 256, 256, "2.5", "mad", "f16"); expect(PI.row->layout(p.p.data(), 6, EXL3I_OPD_A, &e, &sel, &pl, 1, &L), RAD_E_UNSUPPORTED, "layout: operand a"); }
    { Params p(1, 256, 256, "2.5", "mad", "f16"); RadTensor b = pl; b.dtype = RAD_I32;
      expect(PI.row->layout(p.p.data(), 6, EXL3I_OPD_RECORD, &e, &sel, &b, 1, &L), RAD_E_DTYPE, "layout: record as i32"); }
    { Params p(1, 256, 256, "2.5", "mad", "f16"); int s2 = EXL3I_PLANE_SUH;
      expect(PI.row->layout(p.p.data(), 6, EXL3I_OPD_RECORD, &e, &s2, &pl, 1, &L), RAD_E_SHAPE, "layout: record operand given the suh plane"); }
    { Params p(1, 256, 256, "2.5", "mad", "f16"); int s2 = EXL3I_PLANE_SUH;
      RadTensor b = tensor(w.suh.data(), RAD_F16, 1, 256);
      expect(PI.row->layout(p.p.data(), 6, EXL3I_OPD_SUH1, &e, &s2, &b, 1, &L), RAD_E_SHAPE, "layout: suh.1 of an unstacked weight"); }
    // launches the row must refuse
    {
        Params pa(1, 256, 256, "2.5", "mad", "f16");
        Dev d;
        store_weight(w, pa, &d);
        void* dx = nullptr;
        make_x(Run{1, RAD_F16, RAD_F16, 0, false}, 256, &dx);
        std::vector<char> y;
        expect(launch(PI, pa, w, d.rec, d.suh, nullptr, d.svh, Run{1, RAD_F16, RAD_F16, 0, false}, dx, &y, RAD_U32, 16,
                      exl3i_row_words(256, 40)), RAD_E_SHAPE, "launch: the record not in the stored form");
        expect(launch(PI, pa, w, d.rec, d.suh, nullptr, d.svh, Run{1, RAD_F16, RAD_I32, 0, false}, dx, &y, EXL3I_DT_TILES, 16,
                      exl3i_row_words(256, 40)), RAD_E_DTYPE, "launch: y i32");
        Weight w3 = make_weight(40, EXL3I_CB_MAD, 384, 256, true, false);
        Params p3(1, 384, 256, "2.5", "mad", "f16");
        Dev d3;
        expect(store_weight(w3, p3, &d3), RAD_OK, "store a [384, 256] stacked weight");
        expect(launch(PI, p3, w3, d3.rec, d3.suh, d3.suh1, d3.svh, Run{1, RAD_F16, RAD_F16, 0, false}, dx, &y, EXL3I_DT_TILES, 24,
                      exl3i_row_words(256, 40)), RAD_E_SHAPE, "launch: suh.1 with N % 256");
        dfree(dx);
    }
    std::printf("F refusals: %d, %d wrong\n", n, bad);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: exl3i_gpu_test EXL3I.so\n"); return 2; }
    PI = open_plugin(argv[1], EXL3I_OP_NT);
    PIN = (void (*)(int, int64_t))dlsym(PI.h, "exl3i_test_pin");
    if (!PIN) { std::fprintf(stderr, "%s: no exl3i_test_pin\n", argv[1]); return 2; }
    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("exl3i %s -- %s (%s); device %s %s\n", PI.info->version, PI.info->description, PI.info->build_target, prop.name,
                prop.gcnArchName);
    test_decode();
    test_rates();
    test_shapes();
    test_t2();
    test_tp_slices();
    test_refusals();
    std::printf("EXL3I GPU TEST %s", FAILS ? "FAIL" : "PASS");
    if (FAILS) std::printf(" (%d)", FAILS);
    std::printf("\n");
    return FAILS != 0;
}
