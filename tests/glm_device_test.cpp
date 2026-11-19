/* tests/glm_device_test.cpp -- the GLM kernel plugin's device rows against libr4d's chain, on a card.
 *
 *   swiglu_clamped_had_quant_i8g   against libr4d's swiglu_clamped then had_quant_act_i8g (group 128)
 *                                  or had_quant_act_iq8r (group 256): the codes, the scales and the
 *                                  bf16 product byte for byte, at GLM's widths (an expert's TP4 and
 *                                  TP8 slice, the dense layers' TP4 slice) and up to a prefill
 *                                  chunk's rows; and against the plugin's host row, which must agree
 *                                  except where the host's expf rounds a product to the other bf16.
 *
 * Skipped without a card. The libraries are the build tree's (RAD_R4D_LIB, RAD_GLM_LIB), or the
 * environment's RAD_R4D_LIB_OVERRIDE / RAD_GLM_LIB_OVERRIDE. Run on a card under the shared lock:
 * flock gpu.lock build/tests/glm_device_test */
#include "rad_test.h"

#include "rad_abi.h"
#include "rad_device.h"

#include <dlfcn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

#ifndef RAD_R4D_LIB
#define RAD_R4D_LIB ""
#endif
#ifndef RAD_GLM_LIB
#define RAD_GLM_LIB ""
#endif
const char* lib(const char* env, const char* dflt) {
    const char* o = std::getenv(env);
    return o && *o ? o : dflt;
}

struct Plugin {
    void* handle = nullptr;
    int (*count)() = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;
    bool open(const char* path) {
        if (!path) return false;
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) return false;
        count = (int (*)())dlsym(handle, "rad_kernel_count");
        at = (const RadKernelInfo* (*)(int))dlsym(handle, "rad_kernel_at");
        return count && at;
    }
    const RadKernelInfo* row(const char* name) const {
        for (int i = 0; i < count(); ++i) {
            const RadKernelInfo* r = at(i);
            if (r && r->name && std::string(r->name) == name) return r;
        }
        return nullptr;
    }
    ~Plugin() { if (handle) dlclose(handle); }
};

struct Buf {
    std::vector<uint8_t> host;
    void* dev = nullptr;
    RadTensor t{};
    Buf(uint32_t dtype, int64_t rows, int64_t cols, size_t elem) {
        t.dtype = dtype;
        t.rank = 2;
        t.shape[0] = rows;
        t.shape[1] = cols;
        t.stride[1] = 1;
        t.stride[0] = cols;
        host.assign((size_t)(rows * cols) * elem, 0);
    }
    ~Buf() { if (dev) rad_dev_free(dev, RAD_MEM_DEVICE); }
    RadTensor up(RadStream s) {
        if (!dev) dev = rad_dev_alloc((int64_t)host.size(), RAD_MEM_DEVICE);
        rad_memcpy_async(dev, host.data(), (int64_t)host.size(), s);
        RadTensor x = t;
        x.data = dev;
        return x;
    }
    RadTensor on_host() { RadTensor x = t; x.data = host.data(); return x; }
    void down(RadStream s) { rad_memcpy_async(host.data(), dev, (int64_t)host.size(), s); }
};

uint32_t g_seed = 20261119;
uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }

/* bf16 of a normal draw scaled so the gate and up span the clamp: |x| up to ~4 * sigma, with
 * sigma 4, so a share of each row sits past limit 10, and an all-zero row now and then. */
uint16_t draw(float sigma) {
    float u1 = ((rnd() & 0xffffff) + 1) / 16777217.0f, u2 = (rnd() & 0xffffff) / 16777216.0f;
    const float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2) * sigma;
    uint32_t b;
    std::memcpy(&b, &z, 4);
    return (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}

int launch(const RadKernelInfo* k, std::vector<RadTensor> t, std::vector<RadParam> p, RadStream s) {
    RadArgs a{};
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    return k->launch(&a, s);
}

bool have_card() {
    RadDeviceProps dp{};
    return rad_dev_count() >= 1 && rad_dev_props(0, &dp) == RAD_OK && !dp.is_host_backend;
}

}  /* namespace */

TEST(the_fused_clamped_swiglu_quantiser_is_the_libr4d_chain_byte_for_byte) {
    if (!have_card()) { fprintf(stderr, "  SKIP %s: no card\n", ::radtest::current()); return; }
    Plugin r4d, glm;
    const char* pr = lib("RAD_R4D_LIB_OVERRIDE", RAD_R4D_LIB);
    const char* pg = lib("RAD_GLM_LIB_OVERRIDE", RAD_GLM_LIB);
    if (!r4d.open(pr) || !glm.open(pg)) {
        fprintf(stderr, "  SKIP: libraries did not load (%s, %s)\n", pr, pg);
        return;
    }
    const RadKernelInfo* fused = glm.row("glm_swiglu_clamped_had_quant_i8g");
    const RadKernelInfo* host = glm.row("ref_glm_swiglu_clamped_had_quant_i8g");
    const RadKernelInfo* clamp = r4d.row("swiglu_clamped_bf16");
    const RadKernelInfo* q128 = r4d.row("had_quant_act_i8g");
    const RadKernelInfo* q256 = r4d.row("had_quant_act_iq8r");
    if (!fused) { fprintf(stderr, "  SKIP: the plugin has no device rows\n"); return; }
    REQUIRE(host && clamp && q128 && q256);
    RadStream s{};
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);

    struct Case { int64_t M, n, group; };
    /* An expert's TP4 slice (512) and TP8 slice (256) at group 128, routed rows from one token to a
     * prefill chunk's 2048 x 8; the dense layers' TP4 slice (3072) and the shared expert's (512) at
     * IQ8R's 256. */
    const Case cases[] = { { 1, 512, 128 },    { 8, 512, 128 },    { 37, 256, 128 },
                           { 16384, 512, 128 }, { 2048, 256, 128 }, { 1, 3072, 256 },
                           { 64, 3072, 256 },  { 2048, 3072, 256 }, { 33, 512, 256 } };
    const double limit = 10.0;
    long long host_code_diff = 0, host_codes = 0;
    for (const Case& c : cases) {
        Buf gu(RAD_BF16, c.M, 2 * c.n, 2);
        uint16_t* g = (uint16_t*)gu.host.data();
        for (int64_t r = 0; r < c.M; ++r)
            for (int64_t j = 0; j < 2 * c.n; ++j) g[r * 2 * c.n + j] = (r % 97 == 5) ? 0 : draw(4.0f);
        Buf y0(RAD_BF16, c.M, c.n, 2), q0(RAD_I8, c.M, c.n, 1), s0(RAD_F32, c.M, c.n / c.group, 4);
        Buf y1(RAD_BF16, c.M, c.n, 2), q1(RAD_I8, c.M, c.n, 1), s1(RAD_F32, c.M, c.n / c.group, 4);
        Buf q2(RAD_I8, c.M, c.n, 1), s2(RAD_F32, c.M, c.n / c.group, 4);

        const RadTensor tgu = gu.up(s), ty0 = y0.up(s), tq0 = q0.up(s), ts0 = s0.up(s);
        const RadTensor ty1 = y1.up(s), tq1 = q1.up(s), ts1 = s1.up(s), tq2 = q2.up(s),
                        ts2 = s2.up(s);
        /* libr4d's chain */
        REQUIRE_EQ(launch(clamp, { tgu, ty0 },
                          { RAD_INT("M", c.M), RAD_INT("n", c.n), RAD_F64("limit", limit),
                            RAD_STR("dtype", "bf16") }, s), RAD_OK);
        REQUIRE_EQ(launch(c.group == 256 ? q256 : q128, { ty0, tq0, ts0 },
                          { RAD_INT("M", c.M), RAD_INT("n", c.n), RAD_INT("group", c.group),
                            RAD_STR("dtype", "bf16") }, s), RAD_OK);
        /* the fused row, with and without the bf16 product */
        const std::vector<RadParam> fp = { RAD_INT("M", c.M), RAD_INT("n", c.n),
                                           RAD_F64("limit", limit), RAD_INT("group", c.group),
                                           RAD_STR("dtype", "bf16") };
        REQUIRE_EQ(launch(fused, { tgu, tq1, ts1, ty1 }, fp, s), RAD_OK);
        RadTensor none{};
        REQUIRE_EQ(launch(fused, { tgu, tq2, ts2, none }, fp, s), RAD_OK);
        y0.down(s); q0.down(s); s0.down(s); y1.down(s); q1.down(s); s1.down(s); q2.down(s);
        s2.down(s);
        REQUIRE_EQ(rad_stream_sync(s), RAD_OK);

        CHECK(y1.host == y0.host);
        CHECK(q1.host == q0.host);
        CHECK(s1.host == s0.host);
        CHECK(q2.host == q0.host);
        CHECK(s2.host == s0.host);
        fprintf(stderr, "  M %lld n %lld group %lld: %s\n", (long long)c.M, (long long)c.n,
                (long long)c.group,
                (q1.host == q0.host && s1.host == s0.host && y1.host == y0.host) ? "identical"
                                                                                  : "DIFFERENT");

        /* the host row, on the host copies */
        if (c.M <= 2048) {
            Buf qh(RAD_I8, c.M, c.n, 1), sh(RAD_F32, c.M, c.n / c.group, 4);
            REQUIRE_EQ(launch(host, { gu.on_host(), qh.on_host(), sh.on_host(), none }, fp, s),
                       RAD_OK);
            for (size_t i = 0; i < qh.host.size(); ++i) {
                const int d = (int)(int8_t)qh.host[i] - (int)(int8_t)q0.host[i];
                host_code_diff += d != 0;
                CHECK(d >= -1 && d <= 1);
            }
            host_codes += (long long)qh.host.size();
        }
    }
    fprintf(stderr, "  host row: %lld of %lld codes differ by one\n", host_code_diff, host_codes);
    CHECK(host_code_diff * 1000 < host_codes);   /* the last-place expf differences only */
    rad_stream_destroy(s);
}

RAD_TEST_MAIN()
