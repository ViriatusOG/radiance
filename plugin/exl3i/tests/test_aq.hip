// test_aq.hip -- AQ (exl3i_abi.h version 3) on the card: exl3i_actq's planes against the host reference quantiser
// (exl3i_ref_actq: the A8 path's own a8g128 arithmetic) byte for byte, and the dense op's A8 output with aq (its planes
// at aq_plane of aq_planes) against its output quantising itself, byte for byte. Usage: test_aq EXL3I.so
// Cases: gate_up (stacked, planes 1-2 of 3), gdn_qkv (plane 0 of 2), attn_k (plane 2 of 3), down (K 17408, 1 plane),
// at 512 / 600 / 2048 rows, a bf16 and f16. With radiance's libr4d.so (second argument) also the producer stages: pre
// rmsnorm / silu_mul / mul against libr4d's own unfused row (rmsnorm_bf16, silu_mul_bf16, mul_bf16) followed by the
// plain actq -- the planes and h byte for byte (the oracle r4d_fuse.cpp uses for libr4d's own fusions).
// Form xb (version 4, mad's decode GEMV): the same cases at 1 / 3 / 8 rows on a mad weight: exl3i_actq form xb's rows
// (xb = q 0x01010101, the scales, the -126 x bias sums) against the reference's q byte for byte, and the dense op's
// output reading them against the GEMV quantising itself, byte for byte.
// Residual (version 5): libr4d's add_rmsnorm row then the plain actq against actq pre rmsnorm with res (test_res).
// Usage: test_aq EXL3I.so [LIBR4D.so]. Last line PASS / FAIL.
#include <hip/hip_runtime.h>
#include "exl3i.h"
#include "exl3i_decode.h"
#include "exl3i_twopass_ref.h"
#include "exl3i_abi.h"
#include "exl3i_pin.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

#define HIPCHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

static void (*PIN)(int, int64_t) = nullptr;   // the plugin's exl3i_test_pin (exl3i_pin.h)
static void test_pin(int k, int64_t v) { PIN(k, v); }
static uint64_t RNG = 0xA5A5A5A55A5A5A5Aull;
static uint32_t rnd() { RNG ^= RNG << 13; RNG ^= RNG >> 7; RNG ^= RNG << 17; return (uint32_t)(RNG >> 16); }
static uint16_t f2h(float f) { _Float16 h = (_Float16)f; uint16_t u; std::memcpy(&u, &h, 2); return u; }
static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

static const RadKernelInfo *RA = nullptr, *RQ = nullptr;
static int FAILS = 0;

static RadTensor tensor(void* d, uint32_t dt, int64_t r0, int64_t r1) {
    RadTensor t{};
    t.data = d; t.dtype = dt; t.rank = 2; t.shape[0] = r0; t.shape[1] = r1; t.stride[0] = r1; t.stride[1] = 1;
    return t;
}
static void* dput(const void* h, size_t n) { void* d; HIPCHECK(hipMalloc(&d, n)); HIPCHECK(hipMemcpy(d, h, n, hipMemcpyHostToDevice)); return d; }

// ---- the producer stages against libr4d's unfused rows
static RadParam f64p(const char* k, double v) { RadParam p{}; p.key = k; p.kind = RAD_P_F64; p.dval = v; return p; }

static int test_pre(const char* r4d) {
    if (!r4d) { std::printf("pre stages: SKIP (no libr4d.so given)\n"); return 0; }
    void* h = dlopen(r4d, RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::printf("dlopen %s: %s  FAIL\n", r4d, dlerror()); ++FAILS; return 1; }
    auto cnt = (int (*)())dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    const RadKernelInfo *rn = nullptr, *rs = nullptr, *rm = nullptr;
    for (int i = 0; cnt && i < cnt(); ++i) {
        if (!std::strcmp(at(i)->name, "rmsnorm_bf16")) rn = at(i);
        if (!std::strcmp(at(i)->name, "silu_mul_bf16")) rs = at(i);
        if (!std::strcmp(at(i)->name, "mul_bf16")) rm = at(i);
    }
    if (!rn || !rs || !rm) { std::printf("libr4d rows rmsnorm_bf16 / silu_mul_bf16 / mul_bf16 not found  FAIL\n"); ++FAILS; return 1; }
    struct Case { const char* pre; int K, planes; bool hout; };
    const Case cases[] = { { "rmsnorm", 5120, 3, false }, { "rmsnorm", 5120, 2, true }, { "rmsnorm", 5120, 2, false },
                           { "silu_mul", 17408, 1, false }, { "mul", 6144, 1, false }, { "mul", 6144, 1, true } };
    int n = 0;
    for (const Case& c : cases)
        for (int T : { 1, 3, 8, 512, 600, 2048 }) {   // 1..8: the decode band's grid (rmsnorm: a row over several workgroups)
            const int K = c.K, np = c.planes, xw = !std::strcmp(c.pre, "silu_mul") ? 2 * K : K;
            const int64_t rb = exl3i_aq_row_bytes(K, np);
            // x: N(0, 1)-ish bf16 with a few large features (the residual stream's), x2 a gate in (0, 1)
            std::vector<uint16_t> xh((size_t)T * xw), x2h((size_t)T * K);
            for (size_t i = 0; i < xh.size(); ++i) {
                float f = ((int)(rnd() & 0xFFFF) - 32768) / 16384.0f;
                if ((rnd() & 1023) == 0) f *= 64.0f;
                xh[i] = f2bf(f);
            }
            for (auto& q : x2h) q = f2bf((rnd() & 0xFFFF) / 65536.0f);
            if (T == 600 || T == 8) for (int k = 0; k < xw; ++k) xh[(size_t)7 * xw + k] = 0;   // a zero row (rsqrt(eps))
            std::vector<float> gain(K);
            for (auto& g : gain) g = ((int)(rnd() & 0xFFFF) - 32768) / 65536.0f;   // zero-centred, wadd 1
            std::vector<std::vector<uint16_t>> sv(np, std::vector<uint16_t>(K));
            for (auto& v : sv) for (auto& q : v) q = f2h((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f));
            void *dx = dput(xh.data(), xh.size() * 2), *dx2 = dput(x2h.data(), x2h.size() * 2), *dg = dput(gain.data(), K * 4);
            std::vector<void*> ds(np);
            for (int i = 0; i < np; ++i) ds[i] = dput(sv[i].data(), K * 2);
            void *dy, *dh, *dq0, *dq1;
            HIPCHECK(hipMalloc(&dy, (size_t)T * K * 2));
            HIPCHECK(hipMalloc(&dh, (size_t)T * K * 2));
            HIPCHECK(hipMalloc(&dq0, (size_t)T * rb));
            HIPCHECK(hipMalloc(&dq1, (size_t)T * rb));
            HIPCHECK(hipMemset(dh, 0xA5, (size_t)T * K * 2));
            // 1. libr4d's row -> y
            int rc;
            {
                RadArgs a{};
                a.world_size = 1;
                if (!std::strcmp(c.pre, "rmsnorm")) {
                    RadParam p[5] = { RAD_INT("M", T), RAD_INT("n", K), f64p("eps", 1e-6), RAD_STR("dtype", "bf16"), f64p("wadd", 1.0) };
                    RadTensor t[3] = { tensor(dx, RAD_BF16, T, K), tensor(dg, RAD_F32, 1, K), tensor(dy, RAD_BF16, T, K) };
                    t[1].rank = 1; t[1].shape[0] = K; t[1].stride[0] = 1;
                    a.p = p; a.n_p = 5; a.t = t; a.n_t = 3;
                    rc = rn->launch(&a, nullptr);
                } else if (!std::strcmp(c.pre, "silu_mul")) {
                    RadParam p[3] = { RAD_INT("M", T), RAD_INT("n", K), RAD_STR("dtype", "bf16") };
                    RadTensor t[2] = { tensor(dx, RAD_BF16, T, 2 * K), tensor(dy, RAD_BF16, T, K) };
                    a.p = p; a.n_p = 3; a.t = t; a.n_t = 2;
                    rc = rs->launch(&a, nullptr);
                } else {
                    RadParam p[3] = { RAD_INT("M", T), RAD_INT("n", K), RAD_STR("dtype", "bf16") };
                    RadTensor t[3] = { tensor(dx, RAD_BF16, T, K), tensor(dx2, RAD_BF16, T, K), tensor(dy, RAD_BF16, T, K) };
                    a.p = p; a.n_p = 3; a.t = t; a.n_t = 3;
                    rc = rm->launch(&a, nullptr);
                }
            }
            // 2. the plain actq of y; 3. the fused actq of x
            auto actq = [&](const char* pre, void* x, int w, void* out, bool hout) {
                RadParam p[6] = { RAD_INT("M", 2048), RAD_INT("K", K), RAD_INT("planes", np), RAD_STR("pre", pre),
                                  f64p("eps", 1e-6), f64p("wadd", 1.0) };
                RadTensor t[EXL3I_AQ_N_OPD]{};
                t[0] = tensor(x, RAD_BF16, T, w);
                for (int i = 0; i < np; ++i) t[1 + i] = tensor(ds[i], RAD_F16, 1, K);
                t[5] = tensor(out, RAD_U8, T, rb);
                t[6] = tensor(!std::strcmp(pre, "mul") ? dx2 : nullptr, RAD_BF16, T, K);
                t[7] = tensor(!std::strcmp(pre, "rmsnorm") ? dg : nullptr, RAD_F32, 1, K);
                t[8] = tensor(hout ? dh : nullptr, RAD_BF16, T, K);
                RadArgs a{};
                a.p = p; a.n_p = 6; a.t = t; a.n_t = EXL3I_AQ_N_OPD; a.world_size = 1;
                return RQ->launch(&a, nullptr);
            };
            const int r0 = actq("none", dy, K, dq0, false), r1 = actq(c.pre, dx, xw, dq1, c.hout);
            HIPCHECK(hipDeviceSynchronize());
            std::vector<uint8_t> q0((size_t)T * rb), q1((size_t)T * rb);
            std::vector<uint16_t> y((size_t)T * K), hh((size_t)T * K);
            HIPCHECK(hipMemcpy(q0.data(), dq0, q0.size(), hipMemcpyDeviceToHost));
            HIPCHECK(hipMemcpy(q1.data(), dq1, q1.size(), hipMemcpyDeviceToHost));
            HIPCHECK(hipMemcpy(y.data(), dy, y.size() * 2, hipMemcpyDeviceToHost));
            HIPCHECK(hipMemcpy(hh.data(), dh, hh.size() * 2, hipMemcpyDeviceToHost));
            long dq = 0, dhv = 0;
            for (size_t i = 0; i < q0.size(); ++i) dq += q0[i] != q1[i];
            if (c.hout) for (size_t i = 0; i < y.size(); ++i) dhv += y[i] != hh[i];
            else for (size_t i = 0; i < hh.size(); ++i) dhv += hh[i] != 0xA5A5;   // h absent: untouched
            const bool bad = rc != RAD_OK || r0 != RAD_OK || r1 != RAD_OK || dq || dhv;
            std::printf("pre %-8s K %5d T %4d planes %d h %d: libr4d row + actq vs fused: %ld of %zu plane bytes differ, "
                        "h %ld differ%s\n", c.pre, K, T, np, (int)c.hout, dq, q0.size(), dhv, bad ? "  FAIL" : "");
            FAILS += bad; ++n;
            HIPCHECK(hipFree(dx)); HIPCHECK(hipFree(dx2)); HIPCHECK(hipFree(dg)); HIPCHECK(hipFree(dy)); HIPCHECK(hipFree(dh));
            HIPCHECK(hipFree(dq0)); HIPCHECK(hipFree(dq1));
            for (void* d : ds) HIPCHECK(hipFree(d));
        }
    return n;
}

// ---- the residual add (version 5): libr4d's add_rmsnorm row then the plain actq, against actq pre rmsnorm with res
// (xo = x in place, h separate or over res itself), byte for byte: the planes, the residual stream and h. Every row count
// of test_pre; at 1..8 rows the decode band's split form (the row's last workgroup writes xo / h) and, with
// the AQ_RES_SPLIT_OFF pin, one workgroup a row; there also form xb.
static int test_res(const char* r4d) {
    if (!r4d) { std::printf("residual: SKIP (no libr4d.so given)\n"); return 0; }
    void* hl = dlopen(r4d, RTLD_NOW | RTLD_LOCAL);
    if (!hl) { std::printf("dlopen %s: %s  FAIL\n", r4d, dlerror()); ++FAILS; return 1; }
    auto cnt = (int (*)())dlsym(hl, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(hl, "rad_kernel_at");
    const RadKernelInfo* ra = nullptr;
    for (int i = 0; cnt && i < cnt(); ++i)
        if (!std::strcmp(at(i)->name, "add_rmsnorm_bf16")) ra = at(i);
    if (!ra) { std::printf("libr4d row add_rmsnorm_bf16 not found  FAIL\n"); ++FAILS; return 1; }
    const int K = 5120;
    struct Case { int planes; int h; };   // h: 0 none, 1 its own buffer, 2 over res (the GDN site: the norm over h)
    const Case cases[] = { { 3, 0 }, { 2, 1 }, { 2, 2 } };
    int n = 0;
    for (const Case& c : cases)
        for (int T : { 1, 3, 8, 512, 600, 2048 })
            for (int form : { EXL3I_AQ_FORM_I8, EXL3I_AQ_FORM_XB })
                for (int split : { 1, 0 }) {
                    if (T > 8 && (form == EXL3I_AQ_FORM_XB || !split)) continue;
                    const int np = c.planes;
                    const int64_t rb = form == EXL3I_AQ_FORM_XB ? exl3i_aq_xb_row_bytes(K, np) : exl3i_aq_row_bytes(K, np);
                    std::vector<uint16_t> xh((size_t)T * K), rh((size_t)T * K);
                    for (auto& q : xh) {
                        float f = ((int)(rnd() & 0xFFFF) - 32768) / 16384.0f;
                        if ((rnd() & 1023) == 0) f *= 64.0f;
                        q = f2bf(f);
                    }
                    for (auto& q : rh) q = f2bf(((int)(rnd() & 0xFFFF) - 32768) / 8192.0f);
                    if (T == 600 || T == 8)   // a row whose sum is zero (rsqrt(eps)): res = -x
                        for (int k = 0; k < K; ++k) rh[(size_t)7 * K + k] = xh[(size_t)7 * K + k] ^ 0x8000;
                    std::vector<float> gain(K);
                    for (auto& g : gain) g = ((int)(rnd() & 0xFFFF) - 32768) / 65536.0f;
                    std::vector<std::vector<uint16_t>> sv(np, std::vector<uint16_t>(K));
                    for (auto& v : sv) for (auto& q : v) q = f2h((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f));
                    void *x0 = dput(xh.data(), xh.size() * 2), *r0 = dput(rh.data(), rh.size() * 2);   // libr4d's
                    void *x1 = dput(xh.data(), xh.size() * 2), *r1 = dput(rh.data(), rh.size() * 2);   // fused
                    void* dg = dput(gain.data(), (size_t)K * 4);
                    std::vector<void*> ds(np);
                    for (int i = 0; i < np; ++i) ds[i] = dput(sv[i].data(), (size_t)K * 2);
                    void *dy, *dh, *dq0, *dq1;
                    HIPCHECK(hipMalloc(&dy, (size_t)T * K * 2));
                    HIPCHECK(hipMalloc(&dh, (size_t)T * K * 2));
                    HIPCHECK(hipMalloc(&dq0, (size_t)T * rb));
                    HIPCHECK(hipMalloc(&dq1, (size_t)T * rb));
                    HIPCHECK(hipMemset(dq0, 0, (size_t)T * rb));   // form xb's planes end in padding neither launch writes
                    HIPCHECK(hipMemset(dq1, 0, (size_t)T * rb));
                    HIPCHECK(hipMemset(dh, 0xA5, (size_t)T * K * 2));
                    int rc;
                    {   // 1. libr4d's add_rmsnorm: x0 += r0, y = rmsnorm(x0)
                        RadParam p[5] = { RAD_INT("M", T), RAD_INT("n", K), f64p("eps", 1e-6), RAD_STR("dtype", "bf16"),
                                          f64p("wadd", 1.0) };
                        RadTensor t[4] = { tensor(x0, RAD_BF16, T, K), tensor(r0, RAD_BF16, T, K), tensor(dg, RAD_F32, 1, K),
                                           tensor(dy, RAD_BF16, T, K) };
                        t[2].rank = 1; t[2].shape[0] = K; t[2].stride[0] = 1;
                        RadArgs a{};
                        a.world_size = 1; a.p = p; a.n_p = 5; a.t = t; a.n_t = 4;
                        rc = ra->launch(&a, nullptr);
                    }
                    auto actq = [&](bool fused, void* out) {
                        RadParam p[8] = { RAD_INT("M", form == EXL3I_AQ_FORM_XB ? 8 : 2048), RAD_INT("K", K), RAD_INT("planes", np),
                                          RAD_STR("pre", fused ? "rmsnorm" : "none"), f64p("eps", 1e-6), f64p("wadd", 1.0),
                                          RAD_STR("form", form == EXL3I_AQ_FORM_XB ? "xb" : "int8"), RAD_INT("res", 1) };
                        RadTensor t[EXL3I_AQ_N_OPD]{};
                        t[0] = tensor(fused ? x1 : dy, RAD_BF16, T, K);
                        for (int i = 0; i < np; ++i) t[1 + i] = tensor(ds[i], RAD_F16, 1, K);
                        t[5] = tensor(out, RAD_U8, T, rb);
                        if (fused) {
                            t[7] = tensor(dg, RAD_F32, 1, K);
                            t[8] = tensor(c.h == 1 ? dh : c.h == 2 ? r1 : nullptr, RAD_BF16, T, K);
                            t[9] = tensor(r1, RAD_BF16, T, K);
                            t[10] = tensor(x1, RAD_BF16, T, K);
                        }
                        RadArgs a{};
                        a.p = p; a.n_p = fused ? 8 : 7; a.t = t; a.n_t = EXL3I_AQ_N_OPD; a.world_size = 1;
                        return RQ->launch(&a, nullptr);
                    };
                    if (!split) test_pin(EXL3I_PIN_AQ_RES_SPLIT_OFF, 1);
                    const int e0 = actq(false, dq0), e1 = actq(true, dq1);
                    test_pin(EXL3I_PIN_AQ_RES_SPLIT_OFF, 0);
                    HIPCHECK(hipDeviceSynchronize());
                    std::vector<uint8_t> q0((size_t)T * rb), q1((size_t)T * rb);
                    std::vector<uint16_t> y((size_t)T * K), xa((size_t)T * K), xb((size_t)T * K), hh((size_t)T * K);
                    HIPCHECK(hipMemcpy(q0.data(), dq0, q0.size(), hipMemcpyDeviceToHost));
                    HIPCHECK(hipMemcpy(q1.data(), dq1, q1.size(), hipMemcpyDeviceToHost));
                    HIPCHECK(hipMemcpy(y.data(), dy, y.size() * 2, hipMemcpyDeviceToHost));
                    HIPCHECK(hipMemcpy(xa.data(), x0, xa.size() * 2, hipMemcpyDeviceToHost));
                    HIPCHECK(hipMemcpy(xb.data(), x1, xb.size() * 2, hipMemcpyDeviceToHost));
                    HIPCHECK(hipMemcpy(hh.data(), c.h == 2 ? r1 : dh, hh.size() * 2, hipMemcpyDeviceToHost));
                    long dq = 0, dx = 0, dhv = 0;
                    for (size_t i = 0; i < q0.size(); ++i) dq += q0[i] != q1[i];
                    for (size_t i = 0; i < xa.size(); ++i) dx += xa[i] != xb[i];
                    if (c.h) for (size_t i = 0; i < y.size(); ++i) dhv += y[i] != hh[i];
                    else for (size_t i = 0; i < hh.size(); ++i) dhv += hh[i] != 0xA5A5;
                    const bool bad = rc != RAD_OK || e0 != RAD_OK || e1 != RAD_OK || dq || dx || dhv;
                    std::printf("res K %d T %4d planes %d h %s form %-4s %s: add_rmsnorm + actq vs fused: %ld of %zu plane bytes, "
                                "x %ld, h %ld differ%s\n", K, T, np, c.h == 2 ? "=res" : c.h ? "own " : "none",
                                form == EXL3I_AQ_FORM_XB ? "xb" : "int8", split ? "split" : "1 wg ", dq, q0.size(), dx, dhv,
                                bad ? "  FAIL" : "");
                    FAILS += bad; ++n;
                    for (void* d : { x0, r0, x1, r1, dg, dy, dh, dq0, dq1 }) HIPCHECK(hipFree(d));
                    for (void* d : ds) HIPCHECK(hipFree(d));
                }
    return n;
}

// ---- form xb: mad's decode GEMV reading the producer's rows
static int test_xb() {
    test_pin(EXL3I_PIN_MAD_GEMV_T, 8);   // without aq too, the GEMV at every row count (by itself it stops at 1 / 2)
    struct Case { const char* name; int N, K; bool stacked; int plane, planes; };
    const Case cases[] = { { "gate_up", 34816, 5120, true, 1, 3 }, { "gdn_qkv", 10240, 5120, false, 0, 2 },
                           { "attn_k", 1024, 5120, false, 2, 3 }, { "down", 5120, 17408, false, 0, 1 } };
    int n = 0;
    for (const Case& c : cases)
        for (int T : { 1, 3, 8 })
            for (int xbf : { 1, 0 }) {
                const int m = T == 3 ? 66 : 42, K = c.K, N = c.N, np = c.planes;
                const int64_t pb = exl3i_aq_xb_plane_bytes(K), rb = exl3i_aq_xb_row_bytes(K, np), nkb = K / 128;
                const int64_t ld = T == 3 ? rb + 4096 : rb;   // padded rows (a buffer shared by several sites)
                RadEncoding e;
                exl3i_encoding_make(m, EXL3I_CB_MAD, c.stacked, N, K, &e);
                const int W = exl3i_row_words(K, m);
                std::vector<uint32_t> rec((size_t)(N / 16) * W);
                for (auto& x : rec) x = rnd() ^ (rnd() << 16);
                char rt[16];
                exl3i_rate_str(m, rt);
                auto params = [&](bool with_aq) {
                    std::vector<RadParam> p = { RAD_INT("M", 8), RAD_INT("N", N), RAD_INT("K", K), RAD_STR("rate", rt),
                                                RAD_STR("codebook", "mad"), RAD_STR("act", "a8g128") };
                    if (with_aq) { p.push_back(RAD_INT("aq_plane", c.plane)); p.push_back(RAD_INT("aq_planes", np)); }
                    return p;
                };
                auto p0 = params(false);
                RadTensor pl = tensor(rec.data(), RAD_U32, N / 16, W);
                RadLayout L;
                int sel = EXL3I_PLANE_RECORD;
                if (RA->layout(p0.data(), 6, EXL3I_OPD_RECORD, &e, &sel, &pl, 1, &L) != RAD_OK) { std::printf("layout refused\n"); return ++FAILS, n; }
                std::vector<char> rl((size_t)L.bytes);
                RA->relayout(p0.data(), 6, EXL3I_OPD_RECORD, &e, &sel, &pl, 1, rl.data(), L.bytes);
                std::vector<std::vector<uint16_t>> sv(np, std::vector<uint16_t>(K));
                for (auto& v : sv) for (auto& q : v) q = f2h((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f));
                std::vector<uint16_t> svh(N);
                for (auto& q : svh) q = f2h((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f));
                std::vector<uint16_t> xh((size_t)T * K);
                for (auto& q : xh) { float f = ((int)(rnd() & 0xFFFF) - 32768) / 8192.0f; q = xbf ? f2bf(f) : f2h(f); }
                void *drec = dput(rl.data(), rl.size()), *dx = dput(xh.data(), xh.size() * 2), *dsvh = dput(svh.data(), N * 2);
                std::vector<void*> ds(np);
                for (int i = 0; i < np; ++i) ds[i] = dput(sv[i].data(), K * 2);
                void *daq, *dy0, *dy1;
                HIPCHECK(hipMalloc(&daq, (size_t)T * ld));
                HIPCHECK(hipMalloc(&dy0, (size_t)T * N * 4));
                HIPCHECK(hipMalloc(&dy1, (size_t)T * N * 4));
                {   // exl3i_actq form xb against the reference
                    RadParam pq[4] = { RAD_INT("M", 8), RAD_INT("K", K), RAD_INT("planes", np), RAD_STR("form", "xb") };
                    RadTensor t[EXL3I_AQ_N_OPD]{};
                    t[0] = tensor(dx, xbf ? RAD_BF16 : RAD_F16, T, K);
                    for (int i = 0; i < np; ++i) t[1 + i] = tensor(ds[i], RAD_F16, 1, K);
                    t[5] = tensor(daq, RAD_U8, T, ld);
                    RadArgs a{};
                    a.p = pq; a.n_p = 4; a.t = t; a.n_t = EXL3I_AQ_N_OPD; a.world_size = 1;
                    if (RQ->launch(&a, nullptr) != RAD_OK) { std::printf("actq xb launch failed\n"); ++FAILS; }
                    HIPCHECK(hipDeviceSynchronize());
                    std::vector<uint8_t> got((size_t)T * ld);
                    HIPCHECK(hipMemcpy(got.data(), daq, got.size(), hipMemcpyDeviceToHost));
                    long bad = 0;
                    std::vector<int8_t> rq((size_t)T * K);
                    std::vector<float> rs((size_t)T * nkb);
                    for (int i = 0; i < np; ++i) {
                        exl3i_ref_actq(xh.data(), K, xbf, sv[i].data(), sv[i].data(), 1, T, K, 128, rq.data(), rs.data());
                        for (int r = 0; r < T; ++r) {
                            const uint8_t* g = got.data() + (size_t)r * ld + (size_t)i * pb;
                            const int8_t* q = &rq[(size_t)r * K];
                            for (int k = 0; k < K; ++k) { uint32_t v; std::memcpy(&v, g + 4 * k, 4); bad += v != (uint32_t)(uint8_t)q[k] * 0x01010101u; }
                            bad += std::memcmp(g + 4 * K, &rs[(size_t)r * nkb], nkb * 4) != 0;
                            for (int kb = 0; kb < nkb; ++kb)
                                for (int hh = 0; hh < 2; ++hh) {
                                    int sum = 0;
                                    for (int k = kb * 128; k < kb * 128 + 128; ++k) sum += ((k % 16) / 4) % 2 == hh ? q[k] : 0;
                                    int32_t b; std::memcpy(&b, g + 4 * K + 4 * nkb + 4 * (2 * kb + hh), 4);
                                    bad += b != 126 * sum;
                                }
                        }
                    }
                    std::printf("%-8s T %d x %-4s planes %d: actq xb rows differing from the reference %ld%s\n", c.name, T, xbf ? "bf16" : "f16",
                                np, bad, bad ? "  FAIL" : "");
                    FAILS += bad != 0; ++n;
                }
                auto dense = [&](bool with_aq, void* y) {
                    auto p = params(with_aq);
                    RadTensor t[7]{};
                    t[0] = tensor(dx, xbf ? RAD_BF16 : RAD_F16, T, K);
                    t[1] = tensor(drec, EXL3I_DT_TILES, N / 16, W);
                    t[2] = tensor(ds[c.plane], RAD_F16, 1, K);
                    t[3] = tensor(dsvh, RAD_F16, N, 1);
                    t[4] = tensor(c.stacked ? ds[c.plane + 1] : nullptr, RAD_F16, 1, K);
                    t[5] = tensor(y, RAD_F32, T, N);
                    t[6] = tensor(with_aq ? daq : nullptr, RAD_U8, T, ld);
                    RadArgs a{};
                    a.p = p.data(); a.n_p = (int)p.size(); a.t = t; a.n_t = 7; a.world_size = 1;
                    const int64_t sb = RA->scratch(&a);
                    void* sc = nullptr;
                    if (sb > 0) HIPCHECK(hipMalloc(&sc, (size_t)sb));
                    HIPCHECK(hipMemset(sc, 0xA5, (size_t)sb));   // an aq launch must not read the scratch's rows
                    a.scratch = sc; a.scratch_bytes = sb;
                    const int rc = RA->launch(&a, nullptr);
                    HIPCHECK(hipDeviceSynchronize());
                    if (sc) HIPCHECK(hipFree(sc));
                    return rc;
                };
                const int r0 = dense(false, dy0), r1 = dense(true, dy1);
                std::vector<uint32_t> y0((size_t)T * N), y1((size_t)T * N);
                HIPCHECK(hipMemcpy(y0.data(), dy0, y0.size() * 4, hipMemcpyDeviceToHost));
                HIPCHECK(hipMemcpy(y1.data(), dy1, y1.size() * 4, hipMemcpyDeviceToHost));
                long diff = 0;
                for (size_t i = 0; i < y0.size(); ++i) diff += y0[i] != y1[i];
                std::printf("%-8s T %d x %-4s plane %d of %d: mad GEMV with aq (xb) vs quantising itself: %ld of %zu f32 outputs differ%s\n",
                            c.name, T, xbf ? "bf16" : "f16", c.plane, np, diff, y0.size(), (r0 || r1 || diff) ? "  FAIL" : "");
                FAILS += r0 != RAD_OK || r1 != RAD_OK || diff != 0; ++n;
                HIPCHECK(hipFree(drec)); HIPCHECK(hipFree(dx)); HIPCHECK(hipFree(dsvh)); HIPCHECK(hipFree(daq));
                HIPCHECK(hipFree(dy0)); HIPCHECK(hipFree(dy1));
                for (void* d : ds) HIPCHECK(hipFree(d));
            }
    test_pin(EXL3I_PIN_MAD_GEMV_T, 0);
    return n;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: test_aq EXL3I.so [LIBR4D.so]\n"); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    auto inf = (const RadPluginInfo* (*)())dlsym(h, "rad_plugin_info");
    auto cnt = (int (*)())dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    for (int i = 0; i < cnt(); ++i) {
        if (!std::strcmp(at(i)->name, "exl3i_gemm_nt_a8")) RA = at(i);
        if (!std::strcmp(at(i)->name, "exl3i_actq")) RQ = at(i);
    }
    if (!RA || !RQ) { std::fprintf(stderr, "rows exl3i_gemm_nt_a8 and exl3i_actq wanted\n"); return 2; }
    PIN = (void (*)(int, int64_t))dlsym(h, "exl3i_test_pin");
    if (!PIN) { std::fprintf(stderr, "no exl3i_test_pin\n"); return 2; }
    std::printf("exl3i %s -- test_aq\n", inf()->version);
    struct Case { const char* name; int N, K; bool stacked; int plane, planes; };
    const Case cases[] = { { "gate_up", 34816, 5120, true, 1, 3 }, { "gdn_qkv", 10240, 5120, false, 0, 2 },
                           { "attn_k", 1024, 5120, false, 2, 3 }, { "down", 5120, 17408, false, 0, 1 } };
    int n = 0;
    for (const Case& c : cases)
        for (int T : { 512, 600, 2048 })
            for (int xbf : { 1, 0 }) {
                const int m = 40, K = c.K, N = c.N, np = c.planes;
                const int64_t rb = exl3i_aq_row_bytes(K, np), nkb = K / 128;
                const int64_t ld = T == 600 ? rb + 2048 : rb;   // T 600: rows padded (a buffer shared by several sites)
                // the weight (random trellis, mad, K2.5) through the hooks
                RadEncoding e;
                exl3i_encoding_make(m, EXL3I_CB_MAD, c.stacked, N, K, &e);
                const int W = exl3i_row_words(K, m);
                std::vector<uint32_t> rec((size_t)(N / 16) * W);
                for (auto& x : rec) x = rnd() ^ (rnd() << 16);
                char rt[16];
                exl3i_rate_str(m, rt);
                auto params = [&](bool with_aq) {
                    std::vector<RadParam> p = { RAD_INT("M", 2048), RAD_INT("N", N), RAD_INT("K", K), RAD_STR("rate", rt),
                                                RAD_STR("codebook", "mad"), RAD_STR("act", "a8g128") };
                    if (with_aq) { p.push_back(RAD_INT("aq_plane", c.plane)); p.push_back(RAD_INT("aq_planes", np)); }
                    return p;
                };
                auto p0 = params(false);
                RadTensor pl = tensor(rec.data(), RAD_U32, N / 16, W);
                RadLayout L;
                int sel = EXL3I_PLANE_RECORD;
                if (RA->layout(p0.data(), 6, EXL3I_OPD_RECORD, &e, &sel, &pl, 1, &L) != RAD_OK) { std::printf("layout refused\n"); return 1; }
                std::vector<char> rl((size_t)L.bytes);
                RA->relayout(p0.data(), 6, EXL3I_OPD_RECORD, &e, &sel, &pl, 1, rl.data(), L.bytes);
                // np scale vectors (the consumer's at plane, plane + 1 when stacked), x
                std::vector<std::vector<uint16_t>> sv(np, std::vector<uint16_t>(K));
                for (auto& v : sv) for (auto& q : v) q = f2h((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f));
                std::vector<uint16_t> svh(N);
                for (auto& q : svh) q = f2h((rnd() & 1 ? -1.f : 1.f) * (0.5f + (rnd() & 1023) / 1024.f));
                std::vector<uint16_t> xh((size_t)T * K);
                for (auto& q : xh) { float f = ((int)(rnd() & 0xFFFF) - 32768) / 8192.0f; q = xbf ? f2bf(f) : f2h(f); }
                void *drec = dput(rl.data(), rl.size()), *dx = dput(xh.data(), xh.size() * 2), *dsvh = dput(svh.data(), N * 2);
                std::vector<void*> ds(np);
                for (int i = 0; i < np; ++i) ds[i] = dput(sv[i].data(), K * 2);
                void *daq, *dy0, *dy1;
                HIPCHECK(hipMalloc(&daq, (size_t)T * ld));
                HIPCHECK(hipMalloc(&dy0, (size_t)T * N * 2));
                HIPCHECK(hipMalloc(&dy1, (size_t)T * N * 2));
                // 1. exl3i_actq -> daq; against the host reference, plane by plane
                {
                    RadParam pq[3] = { RAD_INT("M", 2048), RAD_INT("K", K), RAD_INT("planes", np) };
                    RadTensor t[EXL3I_AQ_N_OPD]{};
                    t[0] = tensor(dx, xbf ? RAD_BF16 : RAD_F16, T, K);
                    for (int i = 0; i < np; ++i) t[1 + i] = tensor(ds[i], RAD_F16, 1, K);
                    t[5] = tensor(daq, RAD_U8, T, ld);
                    RadArgs a{};
                    a.p = pq; a.n_p = 3; a.t = t; a.n_t = EXL3I_AQ_N_OPD; a.world_size = 1;
                    if (RQ->launch(&a, nullptr) != RAD_OK) { std::printf("actq launch failed\n"); ++FAILS; }
                    HIPCHECK(hipDeviceSynchronize());
                    std::vector<uint8_t> got((size_t)T * ld);
                    HIPCHECK(hipMemcpy(got.data(), daq, got.size(), hipMemcpyDeviceToHost));
                    long bad = 0;
                    std::vector<int8_t> rq((size_t)T * K);
                    std::vector<float> rs((size_t)T * nkb);
                    for (int i = 0; i < np; ++i) {
                        exl3i_ref_actq(xh.data(), K, xbf, sv[i].data(), sv[i].data(), 1, T, K, 128, rq.data(), rs.data());
                        for (int r = 0; r < T; ++r) {
                            bad += std::memcmp(got.data() + (size_t)r * ld + (size_t)i * K, rq.data() + (size_t)r * K, K) != 0;
                            bad += std::memcmp(got.data() + (size_t)r * ld + (size_t)np * K + (size_t)i * nkb * 4, rs.data() + (size_t)r * nkb, nkb * 4) != 0;
                        }
                    }
                    std::printf("%-8s T %4d x %-4s planes %d: actq rows differing from the reference %ld%s\n", c.name, T, xbf ? "bf16" : "f16",
                                np, bad, bad ? "  FAIL" : "");
                    FAILS += bad != 0; ++n;
                }
                // 2. the dense op without and with aq: y byte for byte
                auto dense = [&](bool with_aq, void* y) {
                    auto p = params(with_aq);
                    RadTensor t[7]{};
                    t[0] = tensor(dx, xbf ? RAD_BF16 : RAD_F16, T, K);
                    t[1] = tensor(drec, EXL3I_DT_TILES, N / 16, W);
                    t[2] = tensor(ds[c.plane], RAD_F16, 1, K);
                    t[3] = tensor(dsvh, RAD_F16, N, 1);
                    t[4] = tensor(c.stacked ? ds[c.plane + 1] : nullptr, RAD_F16, 1, K);
                    t[5] = tensor(y, RAD_BF16, T, N);
                    t[6] = tensor(with_aq ? daq : nullptr, RAD_U8, T, ld);
                    RadArgs a{};
                    a.p = p.data(); a.n_p = (int)p.size(); a.t = t; a.n_t = 7; a.world_size = 1;
                    const int64_t sb = RA->scratch(&a);
                    void* sc = nullptr;
                    if (sb > 0) HIPCHECK(hipMalloc(&sc, (size_t)sb));
                    HIPCHECK(hipMemset(sc, 0xA5, (size_t)sb));   // stale arena bytes: an aq launch must not read them
                    a.scratch = sc; a.scratch_bytes = sb;
                    const int rc = RA->launch(&a, nullptr);
                    HIPCHECK(hipDeviceSynchronize());
                    if (sc) HIPCHECK(hipFree(sc));
                    return rc;
                };
                const int r0 = dense(false, dy0), r1 = dense(true, dy1);
                std::vector<uint16_t> y0((size_t)T * N), y1((size_t)T * N);
                HIPCHECK(hipMemcpy(y0.data(), dy0, y0.size() * 2, hipMemcpyDeviceToHost));
                HIPCHECK(hipMemcpy(y1.data(), dy1, y1.size() * 2, hipMemcpyDeviceToHost));
                long diff = 0;
                for (size_t i = 0; i < y0.size(); ++i) diff += y0[i] != y1[i];
                std::printf("%-8s T %4d x %-4s plane %d of %d: dense with aq vs quantising itself: %ld of %zu outputs differ%s\n",
                            c.name, T, xbf ? "bf16" : "f16", c.plane, np, diff, y0.size(), (r0 || r1 || diff) ? "  FAIL" : "");
                FAILS += r0 != RAD_OK || r1 != RAD_OK || diff != 0; ++n;
                HIPCHECK(hipFree(drec)); HIPCHECK(hipFree(dx)); HIPCHECK(hipFree(dsvh)); HIPCHECK(hipFree(daq));
                HIPCHECK(hipFree(dy0)); HIPCHECK(hipFree(dy1));
                for (void* d : ds) HIPCHECK(hipFree(d));
            }
    n += test_xb();
    n += test_pre(argc > 2 ? argv[2] : nullptr);
    n += test_res(argc > 2 ? argv[2] : nullptr);
    std::printf("%d checks\n%s (%d failures)\n", n, FAILS ? "TEST_AQ FAIL" : "TEST_AQ PASS", FAILS);
    return FAILS ? 1 : 0;
}
