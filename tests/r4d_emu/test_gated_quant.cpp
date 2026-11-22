/* gated_quant_i8 (r4d_gated_quant_i8.hip) through the SIMT emulator: act(gate) * up over a packed
 * gate_up and over two pitched operands, for silu, sigmoid and SiTU, at group widths that divide n,
 * leave a partial last group, and default to one scale per row. Checked against a double-precision
 * port of libref's ref_gated_quant_i8 over the bf16 values the kernel reads.
 *
 * The tolerance is the codes' own rounding: a code may sit one step from the double value when its
 * quotient falls on a rounding boundary, so each code must lie within 1 of round(ref / scale), each
 * scale within 1e-5 relative of amax / 127 (1 for a zero group), and the dequantised value within one
 * step of the reference. Zero codes must be exact. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(7);
static float nrm() { static std::normal_distribution<float> d(0.f, 1.f); return d(rng); }

/* A tensor with `rows` rows of `width` elements at row pitch `pitch` (elements, >= width). */
static RadTensor view(void* p, uint32_t dt, int64_t rows, int64_t width, int64_t pitch) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = 2;
    t.shape[0] = rows; t.shape[1] = width; t.stride[0] = pitch; t.stride[1] = 1;
    return t;
}
static std::vector<uint16_t> bfv(size_t n, float sc, std::vector<float>* f) {
    std::vector<uint16_t> b(n);
    f->resize(n);
    for (size_t i = 0; i < n; ++i) { b[i] = rad_f32_to_bf16(nrm() * sc); (*f)[i] = rad_bf16_to_f32(b[i]); }
    return b;
}
struct Params {
    std::vector<RadParam> p;
    void i(const char* k, long long v) { RadParam x{}; x.key = k; x.kind = RAD_P_INT; x.ival = v; p.push_back(x); }
    void f(const char* k, double v) { RadParam x{}; x.key = k; x.kind = RAD_P_F64; x.dval = v; p.push_back(x); }
    void str(const char* k, const char* v) { RadParam x{}; x.key = k; x.kind = RAD_P_STR; x.sval = v; p.push_back(x); }
};

struct Case {
    const char* name;
    int64_t M, n;
    long long group;     /* < 0: absent, so the default applies */
    bool packed;         /* one [M, 2n] gate_up, or two [M, n] operands */
    int64_t pitch_pad;   /* extra elements between rows: a pitched view */
    const char* act;
    double gcap, ucap;
    double sigma;        /* input scale: sets how far the tanh caps are reached */
    int64_t m_cap = 0;   /* the ranged M as a launch sees it, its band's upper bound; 0 is M */
};

static double act_ref(const char* act, double g, double u, double gcap, double ucap) {
    const double sg = 1.0 / (1.0 + std::exp(-g));
    if (!std::strcmp(act, "situ")) {
        const double ga = gcap > 0 ? gcap * std::tanh(g / gcap) : g;
        const double ua = ucap > 0 ? ucap * std::tanh(u / ucap) : u;
        return ga * sg * ua;
    }
    if (!std::strcmp(act, "sigmoid")) return sg * u;
    return g * sg * u;
}

static int run(const Case& c) {
    const int64_t M = c.M, n = c.n;
    const int64_t gw = 2 * n;
    const int64_t pa = (c.packed ? gw : n) + c.pitch_pad;   /* row pitch of the gate operand */
    const int64_t rowsA = M + 2;                             /* two spare rows past the band */

    std::vector<float> af, uf;
    auto ga = bfv((size_t)(rowsA * pa), c.sigma, &af);
    std::vector<uint16_t> gb;
    if (!c.packed) gb = bfv((size_t)(rowsA * n), c.sigma, &uf);

    int64_t ngrp;
    /* libref's default group is 128; a group wider than the row is one scale per row. */
    int64_t group = c.group < 0 ? 128 : c.group;
    if (group <= 0 || group > n) group = n;
    ngrp = (n + group - 1) / group;

    std::vector<int8_t> q((size_t)(M * n), 0);
    std::vector<float> s((size_t)(M * ngrp), 0.0f);

    RadTensor t[4] = {
        c.packed ? view(ga.data(), RAD_BF16, rowsA, gw, pa) : view(ga.data(), RAD_BF16, rowsA, n, pa),
        c.packed ? RadTensor{} : view(gb.data(), RAD_BF16, rowsA, n, n),
        view(q.data(), RAD_I8, M, n, n),
        view(s.data(), RAD_F32, M, ngrp, ngrp),
    };
    Params P;
    P.i("M", c.m_cap > 0 ? c.m_cap : M); P.i("n", n); P.str("dtype", "bf16"); P.str("act", c.act);
    if (c.group >= 0) P.i("group", c.group);
    if (c.gcap > 0 || c.ucap > 0) { P.f("gate_cap", c.gcap); P.f("up_cap", c.ucap); }
    RadArgs a{}; a.t = t; a.n_t = 4; a.p = P.p.data(); a.n_p = (int)P.p.size();
    const int rc = r4d_rad_gated_quant_i8(&a, nullptr);
    if (rc != RAD_OK) { std::printf("  %-44s launch %d\n", c.name, rc); return 1; }

    /* The operands as the kernel reads them: the bf16 values, in double. */
    int bad = 0, off_by_one = 0;
    double worst_scale = 0, worst_deq = 0;
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t gi = 0; gi < ngrp; ++gi) {
            const int64_t k0 = gi * group, k1 = std::min(k0 + group, n);
            std::vector<double> v;
            double amax = 0;
            for (int64_t i = k0; i < k1; ++i) {
                const double g = rad_bf16_to_f32(ga[(size_t)(r * pa + i)]);
                const double u = c.packed ? rad_bf16_to_f32(ga[(size_t)(r * pa + n + i)])
                                          : rad_bf16_to_f32(gb[(size_t)(r * n + i)]);
                const double x = act_ref(c.act, g, u, c.gcap, c.ucap);
                v.push_back(x);
                amax = std::max(amax, std::fabs(x));
            }
            const double sref = amax > 0 ? amax / 127.0 : 1.0;
            const double got_s = s[(size_t)(r * ngrp + gi)];
            worst_scale = std::max(worst_scale, std::fabs(got_s - sref) / sref);
            for (int64_t i = k0; i < k1; ++i) {
                const double x = v[(size_t)(i - k0)];
                const double qref = std::max(-127.0, std::min(127.0, x / sref));
                const int got = q[(size_t)(r * n + i)];
                const double d = std::fabs(got - std::nearbyint(qref));
                if (d > 1.0) ++bad;
                else if (d > 0.0) ++off_by_one;
                worst_deq = std::max(worst_deq, std::fabs(got * got_s - x) / std::max(amax, 1e-30));
            }
        }
    }
    /* Half a step is the device's own rounding of its quotient; the reference rounds the same value. */
    const bool ok = bad == 0 && worst_scale < 1e-5 && worst_deq <= 0.51 / 127.0;
    std::printf("  %-44s codes off by 1: %d  bad %d  scale rel %.1e  deq/amax %.2e  %s\n", c.name,
                off_by_one, bad, worst_scale, worst_deq, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/* The refusals: a cap with silu, and an act name libref does not know. */
static int refusals() {
    std::vector<float> f;
    auto gu = bfv((size_t)(2 * 64), 1.0f, &f);
    std::vector<int8_t> q(64, 0);
    std::vector<float> s(1, 0.0f);
    RadTensor t[4] = { view(gu.data(), RAD_BF16, 1, 128, 128), RadTensor{},
                       view(q.data(), RAD_I8, 1, 64, 64), view(s.data(), RAD_F32, 1, 1, 1) };
    int fails = 0;
    {
        Params P;
        P.i("M", 1); P.i("n", 64); P.str("dtype", "bf16"); P.str("act", "silu"); P.f("gate_cap", 4.0);
        RadArgs a{}; a.t = t; a.n_t = 4; a.p = P.p.data(); a.n_p = (int)P.p.size();
        const int rc = r4d_rad_gated_quant_i8(&a, nullptr);
        const bool ok = rc == RAD_E_UNSUPPORTED;
        std::printf("  %-44s rc %d  %s\n", "a cap with silu is refused", rc, ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    {
        Params P;
        P.i("M", 1); P.i("n", 64); P.str("dtype", "bf16"); P.str("act", "gelu");
        RadArgs a{}; a.t = t; a.n_t = 4; a.p = P.p.data(); a.n_p = (int)P.p.size();
        const int rc = r4d_rad_gated_quant_i8(&a, nullptr);
        const bool ok = rc == RAD_E_INVAL;
        std::printf("  %-44s rc %d  %s\n", "an unknown act is refused", rc, ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    return fails;
}

int main() {
    int fails = 0;
    fails += run({ "silu, packed, group 128, partial last group", 3, 200, 128, true, 0, "silu", 0, 0, 2.0 });
    fails += run({ "silu, two operands, pitched, group 64", 4, 128, 64, false, 9, "silu", 0, 0, 2.0 });
    fails += run({ "sigmoid, packed, default group, one row of scales", 2, 96, -1, true, 0, "sigmoid", 0, 0, 2.0 });
    fails += run({ "situ, packed, both caps, group 128", 4, 192, 128, true, 0, "situ", 4.0, 25.0, 3.0 });
    fails += run({ "situ far past both caps, two operands", 3, 130, 64, false, 5, "situ", 4.0, 25.0, 60.0 });
    fails += run({ "situ, no gate softcap", 2, 64, 64, true, 0, "situ", 0.0, 25.0, 3.0 });
    fails += run({ "situ, no up softcap", 2, 64, 32, true, 0, "situ", 4.0, 0.0, 60.0 });
    fails += run({ "situ, no caps", 2, 96, 32, true, 0, "situ", 0.0, 0.0, 3.0 });
    fails += run({ "silu over zeros: zero groups keep scale 1", 2, 128, 128, true, 0, "silu", 0, 0, 0.0 });
    fails += run({ "silu, prefill-sized, M 200", 200, 256, 128, true, 0, "silu", 0, 0, 2.0 });
    /* The engine launches with every parameter at its band's upper bound: the rows are q's. */
    fails += run({ "situ, M at its band's bound 16384, 5 rows", 5, 1152, 128, true, 0, "situ", 4.0, 25.0, 3.0,
                   16384 });
    fails += refusals();
    std::printf("gated_quant_i8: %d failure(s)\n", fails);
    return fails ? 1 : 0;
}
