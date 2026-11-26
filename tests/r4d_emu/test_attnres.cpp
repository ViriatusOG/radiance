/* Block attention residuals (r4d_attn_res.hip) through the SIMT emulator: the depth softmax over
 * pitched state planes, at depth 0, 1 and 8, one, three and four queries, with and without the
 * addend and the gain, over row counts from one to a prefill-sized handful. Checked against a
 * double-precision port of libref's ref_attn_res.cpp over the bf16 values the kernel reads. The
 * output is bf16, so the tolerance is one rounding of it. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static std::mt19937 rng(123);
static float nrm() { static std::normal_distribution<float> d(0.f, 1.f); return d(rng); }

static RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.shape[i] = shape[i]; t.stride[i] = st; st *= shape[i]; }
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
/* rel_l2 and the worst element relative to the reference's scale */
static bool close(const char* what, const std::vector<double>& ref, const std::vector<float>& got, double tol) {
    double se = 0, sr = 0, worst = 0, scale = 1e-30;
    long nan = 0;
    for (double r : ref) scale = std::max(scale, std::fabs(r));
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::isnan(got[i])) { ++nan; continue; }
        const double e = std::fabs(got[i] - ref[i]);
        se += e * e; sr += ref[i] * ref[i];
        worst = std::max(worst, e / scale);
    }
    const double rel = std::sqrt(se / std::max(sr, 1e-30));
    const bool ok = nan == 0 && rel < tol && worst < 8 * tol;
    std::printf("  %-44s rel_l2 %.2e  worst %.2e  nan %ld  %s\n", what, rel, worst, nan, ok ? "PASS" : "FAIL");
    return ok;
}

/* One configuration: states [MB, MT, n] of which the first nS planes and M rows are read. */
static int run_case(int64_t M, int64_t n, int64_t nS, int64_t nQ, bool with, const char* name,
                    int64_t m_cap = 0) {
    const int64_t MB = nS + 2, MT = M + 3;
    const double eps = 1e-6;
    std::vector<float> stf, curf, addf, qf, wf;
    auto st = bfv((size_t)(MB * MT * n), 1.5f, &stf);
    auto cur = bfv((size_t)(M * n), 1.0f, &curf);
    auto add = bfv((size_t)(M * n), 0.3f, &addf);
    std::vector<float> qv((size_t)(nQ * n)), wv((size_t)(nQ * n));
    for (auto& x : qv) x = nrm() * 0.2f;
    for (size_t i = 0; i < wv.size(); ++i) wv[i] = 1.0f + 0.1f * nrm();
    std::vector<float> yo((size_t)(nQ * M * n), 0.0f);
    std::vector<uint16_t> yb((size_t)(nQ * M * n), 0);

    /* The reference, in double, over the values the kernel reads. */
    std::vector<double> ref((size_t)(nQ * M * n));
    for (int64_t t = 0; t < M; ++t) {
        std::vector<std::vector<double>> v;
        for (int64_t i = 0; i <= nS; ++i) {
            std::vector<double> x((size_t)n);
            for (int64_t c = 0; c < n; ++c)
                x[(size_t)c] = i < nS ? stf[(size_t)(i * MT * n + t * n + c)]
                                      : (double)curf[(size_t)(t * n + c)] +
                                            (with ? (double)addf[(size_t)(t * n + c)] : 0.0);
            double ms = 0;
            for (double e : x) ms += e * e;
            const double r = 1.0 / std::sqrt(ms / n + eps);
            for (double& e : x) e *= r;
            v.push_back(x);
        }
        for (int64_t q = 0; q < nQ; ++q) {
            std::vector<double> sc;
            double mx = -1e300;
            for (auto& x : v) {
                double d = 0;
                for (int64_t c = 0; c < n; ++c) d += x[(size_t)c] * qv[(size_t)(q * n + c)];
                sc.push_back(d / std::sqrt((double)n));
                mx = std::max(mx, sc.back());
            }
            double den = 0;
            for (double& s : sc) { s = std::exp(s - mx); den += s; }
            std::vector<double> m((size_t)n, 0.0);
            for (size_t i = 0; i < v.size(); ++i)
                for (int64_t c = 0; c < n; ++c) m[(size_t)c] += sc[i] / den * v[i][(size_t)c];
            if (with) {
                double ms = 0;
                for (double e : m) ms += e * e;
                const double r = 1.0 / std::sqrt(ms / n + eps);
                for (int64_t c = 0; c < n; ++c) m[(size_t)c] = m[(size_t)c] * r * wv[(size_t)(q * n + c)];
            }
            for (int64_t c = 0; c < n; ++c) ref[(size_t)((q * M + t) * n + c)] = m[(size_t)c];
        }
    }

    RadTensor t[6] = {
        nS > 0 ? tens(st.data(), RAD_BF16, { MB, MT, n }) : RadTensor{},
        tens(cur.data(), RAD_BF16, { M, n }),
        with ? tens(add.data(), RAD_BF16, { M, n }) : RadTensor{},
        tens(qv.data(), RAD_F32, { nQ, n }),
        with ? tens(wv.data(), RAD_F32, { nQ, n }) : RadTensor{},
        tens(yb.data(), RAD_BF16, { nQ, M, n }),
    };
    Params P;
    P.i("M", m_cap > 0 ? m_cap : M); P.i("n", n); P.i("n_state", nS); P.i("n_query", nQ); P.str("dtype", "bf16");
    P.f("eps", eps);
    RadArgs a{}; a.t = t; a.n_t = 6; a.p = P.p.data(); a.n_p = (int)P.p.size();
    const int rc = r4d_rad_attn_res_mix(&a, nullptr);
    if (rc != RAD_OK) { std::printf("  %s launch %d\n", name, rc); return 1; }
    std::vector<float> got((size_t)(nQ * M * n));
    for (size_t i = 0; i < got.size(); ++i) got[i] = rad_bf16_to_f32(yb[i]);
    return close(name, ref, got, 5e-3) ? 0 : 1;
}

/* SiTU over a fused gate_up at pitched rows, around both caps and far past them, against the
 * formula in double. The values are sized so the tanh saturates on the up side and not the gate. */
static int run_situ(int64_t M, int64_t n, double gcap, double ucap, double sigma, const char* name,
                    int64_t m_cap = 0) {
    std::vector<float> gf;
    auto gu = bfv((size_t)((M + 2) * 2 * n), sigma, &gf);   /* two spare rows of pitch */
    std::vector<uint16_t> yb((size_t)(M * n), 0);
    RadTensor t[2] = { tens(gu.data(), RAD_BF16, { M + 2, 2 * n }), tens(yb.data(), RAD_BF16, { M, n }) };
    Params P;
    P.i("M", m_cap > 0 ? m_cap : M); P.i("n", n); P.str("dtype", "bf16"); P.f("gate_cap", gcap);
    P.f("up_cap", ucap);
    RadArgs a{}; a.t = t; a.n_t = 2; a.p = P.p.data(); a.n_p = (int)P.p.size();
    const int rc = r4d_rad_situ_mul(&a, nullptr);
    if (rc != RAD_OK) { std::printf("  %s launch %d\n", name, rc); return 1; }
    std::vector<double> ref((size_t)(M * n));
    std::vector<float> got((size_t)(M * n));
    for (int64_t r = 0; r < M; ++r)
        for (int64_t i = 0; i < n; ++i) {
            const double g = gf[(size_t)(r * 2 * n + i)], u = gf[(size_t)(r * 2 * n + n + i)];
            const double ga = gcap > 0 ? gcap * std::tanh(g / gcap) : g;
            const double ua = ucap > 0 ? ucap * std::tanh(u / ucap) : u;
            ref[(size_t)(r * n + i)] = ga * (1.0 / (1.0 + std::exp(-g))) * ua;
            got[(size_t)(r * n + i)] = rad_bf16_to_f32(yb[(size_t)(r * n + i)]);
        }
    return close(name, ref, got, 5e-3) ? 0 : 1;
}

static int run_case_bf16w(int64_t M, int64_t n, int64_t nS, bool with_add, const char* name) {
    const int64_t MB = nS + 2, MT = M + 2;
    const double eps = 1e-6;
    std::vector<float> stf, curf, addf, wf;
    auto st = bfv((size_t)(MB * MT * n), 1.5f, &stf);
    auto cur = bfv((size_t)(M * n), 1.0f, &curf);
    auto add = bfv((size_t)(M * n), 0.3f, &addf);
    auto wb = bfv((size_t)n, 0.8f, &wf);
    for (size_t i = 0; i < (size_t)n; ++i) {
        wb[i] = rad_f32_to_bf16(1.0f + 0.1f * nrm());
        wf[i] = rad_bf16_to_f32(wb[i]);
    }
    std::vector<float> qv((size_t)n);
    for (auto& x : qv) x = nrm() * 0.2f;
    std::vector<uint16_t> yb((size_t)(M * n), 0);

    std::vector<double> ref((size_t)(M * n));
    for (int64_t t = 0; t < M; ++t) {
        std::vector<std::vector<double>> v;
        for (int64_t i = 0; i <= nS; ++i) {
            std::vector<double> x((size_t)n);
            for (int64_t c = 0; c < n; ++c)
                x[(size_t)c] = i < nS ? stf[(size_t)(i * MT * n + t * n + c)]
                                      : (double)curf[(size_t)(t * n + c)] +
                                            (with_add ? (double)addf[(size_t)(t * n + c)] : 0.0);
            double ms = 0;
            for (double e : x) ms += e * e;
            const double r = 1.0 / std::sqrt(ms / n + eps);
            for (double& e : x) e *= r;
            v.push_back(x);
        }
        std::vector<double> sc;
        double mx = -1e300;
        for (auto& x : v) {
            double d = 0;
            for (int64_t c = 0; c < n; ++c) d += x[(size_t)c] * qv[(size_t)c];
            sc.push_back(d / std::sqrt((double)n));
            mx = std::max(mx, sc.back());
        }
        double den = 0;
        for (double& s : sc) { s = std::exp(s - mx); den += s; }
        std::vector<double> m((size_t)n, 0.0);
        for (size_t i = 0; i < v.size(); ++i)
            for (int64_t c = 0; c < n; ++c) m[(size_t)c] += sc[i] / den * v[i][(size_t)c];
        double ms = 0;
        for (double e : m) ms += e * e;
        const double r = 1.0 / std::sqrt(ms / n + eps);
        for (int64_t c = 0; c < n; ++c)
            ref[(size_t)(t * n + c)] = m[(size_t)c] * r * (double)wf[(size_t)c];
    }

    RadTensor t[6] = {
        nS > 0 ? tens(st.data(), RAD_BF16, { MB, MT, n }) : RadTensor{},
        tens(cur.data(), RAD_BF16, { M, n }),
        with_add ? tens(add.data(), RAD_BF16, { M, n }) : RadTensor{},
        tens(qv.data(), RAD_F32, { 1, n }),
        tens(wb.data(), RAD_BF16, { n }),
        tens(yb.data(), RAD_BF16, { M, n }),
    };
    Params P;
    P.i("M", M); P.i("n", n); P.i("n_state", nS); P.i("n_query", 1); P.str("dtype", "bf16");
    P.f("eps", eps);
    RadArgs a{}; a.t = t; a.n_t = 6; a.p = P.p.data(); a.n_p = (int)P.p.size();
    const int rc = r4d_rad_attn_res_mix(&a, nullptr);
    if (rc != RAD_OK) { std::printf("  %s launch %d\n", name, rc); return 1; }
    std::vector<float> got((size_t)(M * n));
    for (size_t i = 0; i < got.size(); ++i) got[i] = rad_bf16_to_f32(yb[i]);
    return close(name, ref, got, 5e-3) ? 0 : 1;
}

int main() {
    int fails = 0;
    fails += run_situ(3, 96, 4.0, 25.0, 3.0, "situ around both caps");
    fails += run_situ(4, 130, 4.0, 25.0, 60.0, "situ far past both caps");
    fails += run_situ(2, 64, 0.0, 25.0, 3.0, "situ, no gate softcap");
    fails += run_situ(2, 64, 4.0, 0.0, 60.0, "situ, no up softcap");
    /* The engine launches with every parameter at its band's upper bound: the rows are y's. */
    fails += run_situ(5, 1152, 4.0, 25.0, 3.0, "situ, M at its band's bound 16384, 5 rows", 16384);
    /* Depth 0 has no states: the mix is the rms-normed cur. */
    fails += run_case(1, 96, 0, 1, false, "depth 0, one query, M 1");
    fails += run_case(3, 96, 0, 4, true, "depth 0, four queries, addend and gain, M 3");
    fails += run_case(4, 96, 1, 1, false, "depth 1, one query, M 4");
    fails += run_case(5, 300, 1, 3, true, "depth 1, three queries, addend and gain");
    fails += run_case(9, 96, 8, 3, false, "depth 8, three queries, M 9 of 12 rows");
    fails += run_case(16, 96, 8, 4, true, "depth 8, four queries, addend and gain, M 16");
    fails += run_case(37, 2304, 8, 1, true, "model width 2304, depth 8, M 37");
    fails += run_case(2, 40, 2, 3, true, "a narrow row, depth 2, M 2");
    /* The rows are cur's when the launch sees M at its band's upper bound. */
    fails += run_case(13, 2304, 4, 1, true, "M at its band's bound 2048, cur 13 rows", 2048);
    fails += run_case_bf16w(5, 2304, 4, true, "depth 4, 1 query, 1D bf16 gain, with add");
    fails += run_case_bf16w(3, 96, 1, false, "depth 1, 1 query, 1D bf16 gain, no add");
    std::printf("attn_res: %d failure(s)\n", fails);
    return fails ? 1 : 0;
}
