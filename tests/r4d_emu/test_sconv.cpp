/* Inkling 2's short convolution with its residual (r4d_ple_conv_bf16.hip, SCONV) through the SIMT
 * emulator: a ragged prefill with and without the input-dependent taps, then one sequence through a
 * speculative verify with partial acceptance and the steps after it. Each output is held to the
 * formula over the committed history, in double, from the bf16 values the kernel reads; the
 * kernel rounds once to bf16, so the bound is a bf16 ulp of the reference. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct Params {
    std::vector<RadParam> p;
    void i(const char* k, long long v) { RadParam x{}; x.key = k; x.kind = RAD_P_INT; x.ival = v; p.push_back(x); }
    void str(const char* k, const char* v) { RadParam x{}; x.key = k; x.kind = RAD_P_STR; x.sval = v; p.push_back(x); }
};

RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.shape[i] = shape[i]; t.stride[i] = st; st *= shape[i]; }
    return t;
}

/* A deterministic bf16 table, and its values as the kernel sees them. */
std::vector<uint16_t> bf(const std::vector<float>& v) {
    std::vector<uint16_t> b(v.size());
    for (size_t i = 0; i < v.size(); ++i) b[i] = rad_f32_to_bf16(v[i]);
    return b;
}
std::vector<float> fl(const std::vector<uint16_t>& b) {
    std::vector<float> v(b.size());
    for (size_t i = 0; i < b.size(); ++i) v[i] = rad_bf16_to_f32(b[i]);
    return v;
}
std::vector<float> wave(size_t count, float seed) {
    std::vector<float> v(count);
    for (size_t i = 0; i < count; ++i) v[i] = std::sin(0.7 * (double)i + seed);
    return v;
}

/* The formula over one whole sequence, zeros before its first row. */
std::vector<double> ref_seq(const std::vector<float>& x, int64_t L, int64_t n, const std::vector<float>& w,
                            const std::vector<float>& dyn, int64_t heads, int64_t k) {
    const bool has_dyn = !dyn.empty();
    const double div = std::sqrt((double)((has_dyn ? 2 : 1) * k + 1));
    std::vector<double> y((size_t)(L * n));
    for (int64_t t = 0; t < L; ++t)
        for (int64_t c = 0; c < n; ++c) {
            double acc = x[t * n + c];
            for (int64_t lag = 0; lag < k && t - lag >= 0; ++lag) {
                double tap = w[c * k + (k - 1 - lag)];
                if (has_dyn) tap += dyn[t * heads * k + (c / (n / heads)) * k + lag];
                acc += tap * x[(t - lag) * n + c];
            }
            y[t * n + c] = acc / div;
        }
    return y;
}

std::vector<float> rows(const std::vector<float>& v, const std::vector<int>& idx, int64_t width) {
    std::vector<float> out;
    for (int i : idx) out.insert(out.end(), v.begin() + i * width, v.begin() + (i + 1) * width);
    return out;
}

int failures = 0;

/* Launch on `M` rows (`cu` the sequence bounds) and check y against `want` row by row. */
bool launch_and_check(const char* what, int64_t M, int64_t n, int64_t k, int64_t heads,
                      const std::vector<uint16_t>& x, const std::vector<uint16_t>& w,
                      std::vector<uint16_t>& cs, const std::vector<int32_t>& cu,
                      const std::vector<uint16_t>* dyn, const std::vector<int32_t>* cidx,
                      const std::vector<int32_t>* hini, const std::vector<int32_t>* nacc,
                      int64_t slots, int64_t S, const std::vector<double>& want) {
    /* The launch takes writable pointers; these copies keep the caller's inputs untouched. */
    std::vector<uint16_t> xc = x, wc = w, dc = dyn ? *dyn : std::vector<uint16_t>();
    std::vector<int32_t> cuc = cu, ci = cidx ? *cidx : std::vector<int32_t>(), hc = hini ? *hini : std::vector<int32_t>(),
                         nc = nacc ? *nacc : std::vector<int32_t>();
    std::vector<uint16_t> y((size_t)(M * n), 0);
    const int64_t N = (int64_t)cu.size() - 1;
    const int64_t dw = heads * k;
    if (!cidx) { ci.resize((size_t)N); for (int64_t s = 0; s < N; ++s) ci[s] = (int32_t)s; }
    RadTensor t[9] = {
        tens(xc.data(), RAD_BF16, { M, n }), tens(wc.data(), RAD_BF16, { n, k }),
        tens(cs.data(), RAD_BF16, { slots, n, S }), tens(cuc.data(), RAD_I32, { N + 1 }),
        dyn ? tens(dc.data(), RAD_BF16, { M, dw }) : RadTensor{},
        tens(ci.data(), RAD_I32, { N }),
        hini ? tens(hc.data(), RAD_I32, { N }) : RadTensor{},
        nacc ? tens(nc.data(), RAD_I32, { N }) : RadTensor{},
        tens(y.data(), RAD_BF16, { M, n }),
    };
    Params P;
    P.i("M", M); P.i("n", n); P.i("width", k); P.i("heads", heads); P.str("dtype", "bf16");
    RadArgs a{}; a.t = t; a.n_t = 9; a.p = P.p.data(); a.n_p = (int)P.p.size();
    const int rc = r4d_rad_sconv_res(&a, nullptr);
    if (rc != RAD_OK) { std::printf("  %-44s launch %d\n", what, rc); ++failures; return false; }
    double worst = 0;
    bool ok = true;
    for (int64_t i = 0; i < M * n; ++i) {
        const double g = rad_bf16_to_f32(y[i]);
        const double e = std::fabs(g - want[i]);
        worst = std::max(worst, e / (std::fabs(want[i]) + 1e-2));
        if (e > 0.01 * std::fabs(want[i]) + 2e-3) ok = false;
    }
    std::printf("  %-44s worst %.2e  %s\n", what, worst, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
    return ok;
}

/* Ragged prefill: rows 0..2 one sequence in slot 1, rows 3..7 another in slot 0, both cold. */
void prefill(bool dyn) {
    const int64_t n = 8, k = 4, S = 5, heads = dyn ? 2 : 0, dw = heads * k;
    const std::vector<float> wf = wave((size_t)(n * k), 0.1f);
    std::vector<uint16_t> w = bf(wf);
    std::vector<float> xf = wave(8 * n, 0.4f), df = wave(8 * (dw ? dw : 1), 1.3f);
    std::vector<uint16_t> x = bf(xf), d = bf(df);
    std::vector<float> xq = fl(x), dq = fl(d);
    std::vector<uint16_t> cs((size_t)(2 * n * S), rad_f32_to_bf16(7.0f));
    std::vector<int32_t> cu = { 0, 3, 8 }, ci = { 1, 0 }, hi = { 0, 0 };
    const std::vector<float> xa = rows(xq, { 0, 1, 2 }, n), xb = rows(xq, { 3, 4, 5, 6, 7 }, n);
    const std::vector<float> da = dyn ? rows(dq, { 0, 1, 2 }, dw) : std::vector<float>();
    const std::vector<float> db = dyn ? rows(dq, { 3, 4, 5, 6, 7 }, dw) : std::vector<float>();
    const std::vector<float> wq = fl(w);
    std::vector<double> want;
    for (double v : ref_seq(xa, 3, n, wq, da, heads, k)) want.push_back(v);
    for (double v : ref_seq(xb, 5, n, wq, db, heads, k)) want.push_back(v);
    launch_and_check(dyn ? "ragged prefill, dynamic taps" : "ragged prefill, plain (heads 0)", 8, n, k,
                     heads, x, w, cs, cu, dyn ? &d : nullptr, &ci, &hi, nullptr, 2, S, want);
    /* The window left behind: each sequence's last k-1 rows, at offset zero. */
    long bad = 0;
    for (int64_t c = 0; c < n; ++c)
        for (int64_t i = 0; i < k - 1; ++i) {
            bad += std::fabs(rad_bf16_to_f32(cs[1 * n * S + c * S + i]) - xa[i * n + c]) > 1e-6;
            bad += std::fabs(rad_bf16_to_f32(cs[0 * n * S + c * S + i]) - xb[(2 + i) * n + c]) > 1e-6;
        }
    std::printf("  %-44s bad %ld  %s\n", "window after prefill", bad, bad ? "FAIL" : "PASS");
    failures += bad != 0;
}

/* One sequence in slot 0: a cold prefix of 3 rows, a verify of 3 drafts reading at offset 0, a
 * step after two drafts were kept (offset 1), and one after a single kept row (offset 0). */
void speculative(bool dyn) {
    const int64_t n = 8, k = 4, S = 5, heads = dyn ? 2 : 0, dw = heads * k;
    const std::vector<float> wq = fl(bf(wave((size_t)(n * k), 0.1f)));
    const std::vector<uint16_t> w = bf(wave((size_t)(n * k), 0.1f));
    const std::vector<uint16_t> x = bf(wave(16 * n, 0.9f)), d = bf(wave(16 * (dw ? dw : 1), 2.1f));
    const std::vector<float> xq = fl(x), dq = fl(d);
    std::vector<uint16_t> cs((size_t)(n * S), rad_f32_to_bf16(7.0f));
    std::vector<int32_t> zero = { 0 };
    std::vector<int> committed;
    auto step = [&](const std::vector<int>& r, int hinit, int nacc, const char* what) {
        const int64_t M = (int64_t)r.size();
        std::vector<int> seq = committed;
        seq.insert(seq.end(), r.begin(), r.end());
        const std::vector<float> xs = rows(xq, seq, n);
        const std::vector<float> ds = dyn ? rows(dq, seq, dw) : std::vector<float>();
        const std::vector<double> full = ref_seq(xs, (int64_t)seq.size(), n, wq, ds, heads, k);
        std::vector<double> want(full.end() - M * n, full.end());
        std::vector<uint16_t> xr = std::vector<uint16_t>();
        for (int i : r) xr.insert(xr.end(), x.begin() + i * n, x.begin() + (i + 1) * n);
        std::vector<uint16_t> dr;
        if (dyn) for (int i : r) dr.insert(dr.end(), d.begin() + i * dw, d.begin() + (i + 1) * dw);
        std::vector<int32_t> cu = { 0, (int32_t)M }, hi = { hinit }, na = { nacc };
        launch_and_check(what, M, n, k, heads, xr, w, cs, cu, dyn ? &dr : nullptr, &zero, &hi,
                         nacc > 0 ? &na : nullptr, 1, S, want);
    };
    step({ 0, 1, 2 }, 0, 0, "prefix, cold, every row commits");
    committed = { 0, 1, 2 };
    step({ 3, 4, 5 }, 1, 1, "verify of three, read at offset 0");
    committed.push_back(3);
    committed.push_back(4);
    step({ 6 }, 1, 2, "after two drafts kept, read at 1");
    committed.push_back(6);
    step({ 7 }, 1, 1, "after one row kept, read at 0");
}

void sconv_add_cases() {
    const int64_t M = 6, n = 16, k = 4, S = 5;
    const std::vector<float> wf = wave((size_t)(n * k), 0.2f), xf = wave((size_t)(M * n), 0.5f),
                             rf = wave((size_t)(M * n), 1.1f);
    std::vector<uint16_t> w = bf(wf), x = bf(xf), add_r = bf(rf);
    const std::vector<float> wq = fl(w), xq = fl(x), rq = fl(add_r);
    std::vector<int32_t> cu = { 0, 2, 6 }, ci = { 0, 1 }, hi = { 0, 0 };
    std::vector<double> sc0 = ref_seq(rows(xq, { 0, 1 }, n), 2, n, wq, {}, 0, k);
    std::vector<double> sc1 = ref_seq(rows(xq, { 2, 3, 4, 5 }, n), 4, n, wq, {}, 0, k);
    std::vector<double> want = sc0;
    want.insert(want.end(), sc1.begin(), sc1.end());
    for (int64_t i = 0; i < M * n; ++i)
        want[(size_t)i] = rq[(size_t)i] + (double)rad_bf16_to_f32(rad_f32_to_bf16((float)want[(size_t)i]));

    for (int pass = 0; pass < 2; ++pass) {
        const bool alias = pass == 1;
        const char* what = alias ? "sconv_res_add in-place (y == add_resid)" : "sconv_res_add out-of-place";
        std::vector<uint16_t> cs((size_t)(2 * n * S), 0);
        std::vector<uint16_t> rc_buf = add_r, y((size_t)(M * n), 0);
        uint16_t* dst = alias ? rc_buf.data() : y.data();
        RadTensor t[10] = {
            tens(x.data(), RAD_BF16, { M, n }),
            tens(w.data(), RAD_BF16, { n, k }),
            tens(cs.data(), RAD_BF16, { 2, n, S }),
            tens(cu.data(), RAD_I32, { 3 }),
            RadTensor{},
            tens(ci.data(), RAD_I32, { 2 }),
            tens(hi.data(), RAD_I32, { 2 }),
            RadTensor{},
            tens(dst, RAD_BF16, { M, n }),
            tens(rc_buf.data(), RAD_BF16, { M, n }),
        };
        Params P;
        P.i("M", M); P.i("n", n); P.i("width", k); P.i("heads", 0); P.str("dtype", "bf16");
        RadArgs a{}; a.t = t; a.n_t = 10; a.p = P.p.data(); a.n_p = (int)P.p.size();
        const int rc = r4d_rad_sconv_res_add(&a, nullptr);
        if (rc != RAD_OK) { std::printf("  %-44s launch %d\n", what, rc); ++failures; continue; }
        double worst = 0;
        bool ok = true;
        for (int64_t i = 0; i < M * n; ++i) {
            const double g = rad_bf16_to_f32(dst[(size_t)i]);
            const double e = std::fabs(g - want[(size_t)i]);
            worst = std::max(worst, e / (std::fabs(want[(size_t)i]) + 1e-2));
            if (e > 0.01 * std::fabs(want[(size_t)i]) + 2e-3) ok = false;
        }
        std::printf("  %-44s worst %.2e  %s\n", what, worst, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }
}

}  // namespace

int main() {
    std::printf("sconv_res (r4d emulator, bf16):\n");
    prefill(true);
    prefill(false);
    speculative(true);
    speculative(false);
    sconv_add_cases();
    std::printf("sconv_res: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
