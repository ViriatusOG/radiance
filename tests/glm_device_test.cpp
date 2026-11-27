/* tests/glm_device_test.cpp -- the GLM kernel plugin's device rows against libr4d's chain, on a card.
 *
 *   swiglu_clamped_had_quant_i8g   against libr4d's swiglu_clamped then had_quant_act_i8g (group 128)
 *                                  or had_quant_act_iq8r (group 256): the codes, the scales and the
 *                                  bf16 product byte for byte, at GLM's widths (an expert's TP4 and
 *                                  TP8 slice, the dense layers' TP4 slice) and up to a prefill
 *                                  chunk's rows; and against the plugin's host row, which must agree
 *                                  except where the host's expf rounds a product to the other bf16.
 *   mla_attn at kv_dtype i8        the plugin's int8-latent reader against libref's host reader
 *                                  over hand-built caches (codes, f16 half scales, zero halves),
 *                                  shuffled and missing pages, partial and empty histories, and
 *                                  histories long enough to split and merge; within bf16 rounding
 *                                  of the output, since the two reduce in different orders.
 *
 * Skipped without a card. The libraries are the build tree's (RAD_R4D_LIB, RAD_GLM_LIB), or the
 * environment's RAD_R4D_LIB_OVERRIDE / RAD_GLM_LIB_OVERRIDE. Run on a card under the shared lock:
 * flock gpu.lock build/tests/glm_device_test */
#include "rad_test.h"

#include "rad_abi.h"
#include "rad_device.h"

#include <dlfcn.h>

#include <algorithm>
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
#ifndef RAD_REF_LIB
#define RAD_REF_LIB ""
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
    if (!have_card()) SKIP("no card");
    Plugin r4d, glm;
    const char* pr = lib("RAD_R4D_LIB_OVERRIDE", RAD_R4D_LIB);
    const char* pg = lib("RAD_GLM_LIB_OVERRIDE", RAD_GLM_LIB);
    if (!r4d.open(pr) || !glm.open(pg))
        ARTIFACT_MISSING(std::string(pr) + " or " + pg);
    const RadKernelInfo* fused = glm.row("glm_swiglu_clamped_had_quant_i8g");
    const RadKernelInfo* host = glm.row("ref_glm_swiglu_clamped_had_quant_i8g");
    const RadKernelInfo* clamp = r4d.row("swiglu_clamped_bf16");
    const RadKernelInfo* q128 = r4d.row("had_quant_act_i8g");
    const RadKernelInfo* q256 = r4d.row("had_quant_act_iq8r");
    if (!fused) SKIP("the plugin has no device rows");
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

namespace {
uint16_t f32_to_f16(float f) {
    /* positive normals only (the scales drawn here), round to nearest even */
    uint32_t b;
    std::memcpy(&b, &f, 4);
    const int e = (int)((b >> 23) & 0xff) - 127 + 15;
    uint32_t m = b & 0x7fffff;
    if (e <= 0) return 0;
    uint32_t h = ((uint32_t)e << 10) | (m >> 13);
    const uint32_t rest = m & 0x1fff;
    if (rest > 0x1000 || (rest == 0x1000 && (h & 1))) ++h;
    return (uint16_t)h;
}
float bf16f(uint16_t h) { uint32_t b = (uint32_t)h << 16; float f; std::memcpy(&f, &b, 4); return f; }
}  /* namespace */

TEST(the_int8_latent_reader_agrees_with_the_host_reader) {
    if (!have_card()) SKIP("no card");
    Plugin glm, ref;
    const char* pg = lib("RAD_GLM_LIB_OVERRIDE", RAD_GLM_LIB);
    const char* pr = lib("RAD_REF_LIB_OVERRIDE", RAD_REF_LIB);
    if (!glm.open(pg) || !ref.open(pr))
        ARTIFACT_MISSING(std::string(pg) + " or " + pr);
    const RadKernelInfo* dev = glm.row("glm_mla_attn_i8kv");
    const RadKernelInfo* host = ref.row("ref_mla_attn");
    if (!dev) SKIP("the plugin has no device rows");
    REQUIRE(host);
    RadStream s{};
    REQUIRE_EQ(rad_stream_create(&s, 0), RAD_OK);
    const int64_t D = 512, ROWB = 516, BS = 4;
    const double scale = 1.0 / 16.0;
    struct Case { int64_t M, H, blocks; };
    const Case cases[] = { { 1, 64, 1 }, { 3, 64, 40 }, { 1, 64, 600 }, { 4, 16, 300 },
                           { 32, 8, 20 }, { 2, 1, 129 }, { 63, 16, 64 }, { 6, 16, 12 } };
    double worst = 0;
    for (const Case& c : cases) {
        const int64_t P = c.M * c.blocks + 3;
        Buf kv(RAD_I8, P * BS, ROWB, 1);
        kv.t.rank = 4;
        kv.t.shape[0] = P; kv.t.shape[1] = 1; kv.t.shape[2] = BS; kv.t.shape[3] = ROWB;
        kv.t.stride[3] = 1; kv.t.stride[2] = ROWB; kv.t.stride[1] = BS * ROWB; kv.t.stride[0] = BS * ROWB;
        for (int64_t t = 0; t < P * BS; ++t) {
            uint8_t* row = kv.host.data() + t * ROWB;
            for (int h = 0; h < 2; ++h) {
                const bool zero = (rnd() % 23) == 0;
                for (int j = 0; j < 256; ++j)
                    row[h * 256 + j] = zero ? 0 : (uint8_t)(int8_t)((int)(rnd() % 255) - 127);
                const float sc = zero ? 0.0f : 0.002f + (rnd() % 1000) * 0.00004f;
                const uint16_t hb = zero ? 0 : f32_to_f16(sc);
                std::memcpy(row + 512 + 2 * h, &hb, 2);
            }
        }
        Buf q(RAD_BF16, c.M * c.H, D, 2);
        q.t.rank = 3; q.t.shape[0] = c.M; q.t.shape[1] = c.H; q.t.shape[2] = D;
        q.t.stride[2] = 1; q.t.stride[1] = D; q.t.stride[0] = c.H * D;
        uint16_t* qh = (uint16_t*)q.host.data();
        for (int64_t i = 0; i < c.M * c.H * D; ++i) qh[i] = draw(0.5f);
        Buf bt(RAD_I32, c.M, c.blocks, 4), ln(RAD_I32, 1, c.M, 4);
        ln.t.rank = 1; ln.t.shape[0] = c.M; ln.t.stride[0] = 1;
        int32_t* b = (int32_t*)bt.host.data();
        std::vector<int32_t> pages((size_t)P);
        for (int64_t i = 0; i < P; ++i) pages[(size_t)i] = (int32_t)i;
        for (int64_t i = P - 1; i > 0; --i) std::swap(pages[(size_t)i], pages[(size_t)(rnd() % (i + 1))]);
        for (int64_t i = 0; i < c.M * c.blocks; ++i) b[i] = (i % 37 == 11) ? -1 : pages[(size_t)i];
        /* Row 2 (of four or more) is nonempty with every page absent; row 0 of a long history has
         * its first 16-token tile absent and later tiles present. */
        if (c.M > 3)
            for (int64_t j = 0; j < c.blocks; ++j) b[2 * c.blocks + j] = -1;
        if (c.blocks >= 8)
            for (int64_t j = 0; j < 4; ++j) b[j] = -1;
        int32_t* l = (int32_t*)ln.host.data();
        for (int64_t r = 0; r < c.M; ++r)
            l[r] = (int32_t)(r == 1 && c.M > 2 ? 0 : c.blocks * BS - (r % 4));
        Buf od(RAD_BF16, c.M * c.H, D, 2), oh(RAD_BF16, c.M * c.H, D, 2);
        for (Buf* o : { &od, &oh }) {
            o->t.rank = 3; o->t.shape[0] = c.M; o->t.shape[1] = c.H; o->t.shape[2] = D;
            o->t.stride[2] = 1; o->t.stride[1] = D; o->t.stride[0] = c.H * D;
        }
        const std::vector<RadParam> p = { RAD_INT("M", c.M), RAD_INT("latent_dim", D),
                                          RAD_INT("heads", c.H), RAD_INT("blocks", c.blocks),
                                          RAD_INT("block_size", BS), RAD_F64("scale", scale),
                                          RAD_STR("dtype", "bf16"), RAD_STR("kv_dtype", "i8") };
        std::vector<RadTensor> th = { q.on_host(), kv.on_host(), bt.on_host(), ln.on_host(), oh.on_host() };
        RadArgs ah{};
        ah.t = th.data(); ah.n_t = 5; ah.p = p.data(); ah.n_p = (int)p.size();
        REQUIRE_EQ(host->launch(&ah, s), RAD_OK);
        const uint16_t* y = (const uint16_t*)oh.host.data();
        /* The matrix form (sixteen heads a tile) and the wave form, which eight heads always take
         * and RADIANCE_GLM_MLA_WAVE forces. */
        const char* was = std::getenv("RADIANCE_GLM_MLA_WAVE");
        const std::string saved = was ? was : "";
        struct Restore {
            bool had; std::string v;
            ~Restore() { if (had) setenv("RADIANCE_GLM_MLA_WAVE", v.c_str(), 1); else unsetenv("RADIANCE_GLM_MLA_WAVE"); }
        } restore{ was != nullptr, saved };
        /* rows with no visible token: their output must be written as exact zeros */
        std::vector<bool> blank((size_t)c.M, false);
        for (int64_t r = 0; r < c.M; ++r) {
            bool any = false;
            for (int64_t t = 0; t < l[r] && !any; ++t) any = b[r * c.blocks + t / BS] >= 0;
            blank[(size_t)r] = !any;
        }
        for (const bool wave : { false, true }) {
            if (!wave && c.H % 16) continue;
            if (wave) setenv("RADIANCE_GLM_MLA_WAVE", "1", 1); else unsetenv("RADIANCE_GLM_MLA_WAVE");
            /* poisoned: a value the reader fails to write stays NaN and fails below */
            for (size_t i = 0; i + 1 < od.host.size(); i += 2) { od.host[i] = 0xc0; od.host[i + 1] = 0x7f; }
            RadArgs a{};
            std::vector<RadTensor> t = { q.up(s), kv.up(s), bt.up(s), ln.up(s), od.up(s) };
            a.t = t.data(); a.n_t = 5; a.p = p.data(); a.n_p = (int)p.size();
            const int64_t sb = dev->scratch ? dev->scratch(&a) : 0;
            void* scratch = sb ? rad_dev_alloc(sb, RAD_MEM_DEVICE) : nullptr;
            a.scratch = scratch; a.scratch_bytes = sb;
            REQUIRE_EQ(dev->launch(&a, s), RAD_OK);
            od.down(s);
            REQUIRE_EQ(rad_stream_sync(s), RAD_OK);
            if (scratch) rad_dev_free(scratch, RAD_MEM_DEVICE);
            const uint16_t* x = (const uint16_t*)od.host.data();
            double peak = 0, err = 0;
            long long bad = 0, nonzero_blank = 0, first_bad = -1;
            for (int64_t i = 0; i < c.M * c.H * D; ++i) {
                const float xv = bf16f(x[i]), yv = bf16f(y[i]);
                if (!std::isfinite(xv) || !std::isfinite(yv)) {
                    if (!bad++) first_bad = i;
                    continue;
                }
                if (blank[(size_t)(i / (c.H * D))] && (x[i] & 0x7fff)) ++nonzero_blank;
                peak = std::fmax(peak, std::fabs(yv));
                err = std::fmax(err, std::fabs((double)xv - yv));
            }
            if (bad)
                fprintf(stderr, "  %lld nonfinite values, first at row %lld head %lld col %lld\n", bad,
                        first_bad / (c.H * D), (first_bad / D) % c.H, first_bad % D);
            CHECK_EQ(bad, 0ll);
            CHECK_EQ(nonzero_blank, 0ll);
            const double rel = peak > 0 ? err / peak : err;
            worst = std::fmax(worst, rel);
            fprintf(stderr, "  %s M %lld heads %lld blocks %lld: max |dev - host| %.3g of peak %.3g (%.2g)\n",
                    wave ? "wave  " : "matrix", (long long)c.M, (long long)c.H, (long long)c.blocks,
                    err, peak, rel);
            CHECK(rel < 1.6e-2);   /* two bf16 rounding steps of the output's range */
        }
    }
    fprintf(stderr, "  worst relative error %.3g\n", worst);
    rad_stream_destroy(s);
}

RAD_TEST_MAIN()
