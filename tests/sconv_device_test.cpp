/* sconv_res on the card: libr4d's exported launcher (r4d_rad_sconv_res) against the formula in
 * double, over the shapes the model issues. Build with hipcc; pass the path of libr4d.so, and
 * optionally the entries' prefix (`halo` for libhalo's halo_rad_sconv_res). Where the library
 * exports <prefix>_rad_sconv_res_add as well, every launch also runs it on a copy of the state:
 * its output is held to add_resid plus the formula, and the state it leaves must be the plain
 * kernel's, byte for byte.
 *
 * A ragged prefill of sequences of several lengths at the model's width (n 1024, 8 heads of 128
 * for the dynamic form, n 2304 for the plain one), then decode steps of one row a sequence, then a
 * speculative verify with partial acceptance and the steps after it. Each output is held to the
 * formula over the committed history, from the bf16 values the kernel reads. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>

using Fn = int (*)(const RadArgs*, RadStream);

namespace {

int failures = 0;
Fn g_add = nullptr;   /* <prefix>_rad_sconv_res_add, when the library exports it */

void hipcheck(hipError_t e) {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(e)); std::exit(1); }
}

template <class T>
struct Device {
    T* p = nullptr;
    size_t n = 0;
    explicit Device(const std::vector<T>& v) : n(v.size()) {
        hipcheck(hipMalloc(&p, n * sizeof(T) + 16));
        hipcheck(hipMemcpy(p, v.data(), n * sizeof(T), hipMemcpyHostToDevice));
    }
    ~Device() { (void)hipFree(p); }
    std::vector<T> get() const {
        std::vector<T> v(n);
        hipcheck(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
        return v;
    }
};

RadTensor tensor(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.shape[i] = shape[i]; t.stride[i] = st; st *= shape[i]; }
    return t;
}

uint16_t bf16(float f) { return rad_f32_to_bf16(f); }
float fbf16(uint16_t b) { return rad_bf16_to_f32(b); }
std::vector<uint16_t> to_bf16(const std::vector<float>& v) {
    std::vector<uint16_t> b(v.size());
    for (size_t i = 0; i < v.size(); ++i) b[i] = bf16(v[i]);
    return b;
}
/* A deterministic pattern in [-1, 1], varied by `seed`, rounded to bf16 as the kernel reads it, so
 * the reference and the kernel see the same inputs. */
std::vector<float> wave(size_t count, double seed) {
    std::vector<float> v(count);
    for (size_t i = 0; i < count; ++i) v[i] = fbf16(bf16((float)std::sin(0.37 * (double)i + seed)));
    return v;
}

/* The formula over one whole sequence, zeros before its first row. */
std::vector<double> reference(const std::vector<float>& x, int64_t L, int64_t n, const std::vector<float>& w,
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

struct Stream {
    int64_t n = 0, k = 4, heads = 0, S = 0;
    std::vector<float> w, rows_x, rows_d;   /* the weight; every token's row and its dyn row */
};

/* One launch of M rows over the sequences `cu`, state `cs` [slots, n, S], checked against `want`. */
void launch(const char* what, const Stream& st, const std::vector<uint16_t>& x, const std::vector<uint16_t>& w,
            std::vector<uint16_t>& cs, int64_t slots, const std::vector<int32_t>& cu,
            const std::vector<uint16_t>* dyn, const std::vector<int32_t>& cidx,
            const std::vector<int32_t>* hini, const std::vector<int32_t>* nacc,
            const std::vector<double>& want, Fn fn) {
    const int64_t M = (int64_t)x.size() / st.n, N = (int64_t)cu.size() - 1, dw = st.heads * st.k;
    Device<uint16_t> dx(x), dwt(w), dcs(cs), ddy(dyn ? *dyn : std::vector<uint16_t>(1, 0));
    Device<int32_t> dcu(cu), dci(cidx), dhi(hini ? *hini : std::vector<int32_t>(1, 1)),
                    dna(nacc ? *nacc : std::vector<int32_t>(1, 1));
    Device<uint16_t> dy(std::vector<uint16_t>((size_t)(M * st.n), 0));
    /* sconv_res_add on a copy of the state, before the plain launch moves it. */
    std::vector<uint16_t> resid, y_add, cs_add;
    int rc_add = RAD_OK;
    if (g_add) {
        resid.resize((size_t)(M * st.n));
        for (size_t i = 0; i < resid.size(); ++i) resid[i] = bf16((float)std::cos(0.11 * (double)i));
        Device<uint16_t> dcs2(cs), dr(resid), dy2(std::vector<uint16_t>((size_t)(M * st.n), 0));
        RadTensor t2[10] = {
            tensor(dx.p, RAD_BF16, { M, st.n }), tensor(dwt.p, RAD_BF16, { st.n, st.k }),
            tensor(dcs2.p, RAD_BF16, { slots, st.n, st.S }), tensor(dcu.p, RAD_I32, { N + 1 }),
            dyn ? tensor(ddy.p, RAD_BF16, { M, dw }) : RadTensor{},
            tensor(dci.p, RAD_I32, { N }),
            hini ? tensor(dhi.p, RAD_I32, { N }) : RadTensor{},
            nacc ? tensor(dna.p, RAD_I32, { N }) : RadTensor{},
            tensor(dy2.p, RAD_BF16, { M, st.n }), tensor(dr.p, RAD_BF16, { M, st.n }),
        };
        RadParam p2[5]{};
        p2[0].key = "M";      p2[0].kind = RAD_P_INT; p2[0].ival = M;
        p2[1].key = "n";      p2[1].kind = RAD_P_INT; p2[1].ival = st.n;
        p2[2].key = "width";  p2[2].kind = RAD_P_INT; p2[2].ival = st.k;
        p2[3].key = "heads";  p2[3].kind = RAD_P_INT; p2[3].ival = st.heads;
        p2[4].key = "dtype";  p2[4].kind = RAD_P_STR; p2[4].sval = "bf16";
        RadArgs a2{};
        a2.t = t2; a2.n_t = 10; a2.p = p2; a2.n_p = 5;
        rc_add = g_add(&a2, nullptr);
        hipcheck(hipDeviceSynchronize());
        y_add = dy2.get();
        cs_add = dcs2.get();
    }
    RadTensor t[9] = {
        tensor(dx.p, RAD_BF16, { M, st.n }), tensor(dwt.p, RAD_BF16, { st.n, st.k }),
        tensor(dcs.p, RAD_BF16, { slots, st.n, st.S }), tensor(dcu.p, RAD_I32, { N + 1 }),
        dyn ? tensor(ddy.p, RAD_BF16, { M, dw }) : RadTensor{},
        tensor(dci.p, RAD_I32, { N }),
        hini ? tensor(dhi.p, RAD_I32, { N }) : RadTensor{},
        nacc ? tensor(dna.p, RAD_I32, { N }) : RadTensor{},
        tensor(dy.p, RAD_BF16, { M, st.n }),
    };
    RadParam p[5]{};
    p[0].key = "M";      p[0].kind = RAD_P_INT; p[0].ival = M;
    p[1].key = "n";      p[1].kind = RAD_P_INT; p[1].ival = st.n;
    p[2].key = "width";  p[2].kind = RAD_P_INT; p[2].ival = st.k;
    p[3].key = "heads";  p[3].kind = RAD_P_INT; p[3].ival = st.heads;
    p[4].key = "dtype";  p[4].kind = RAD_P_STR; p[4].sval = "bf16";
    RadArgs a{};
    a.t = t; a.n_t = 9; a.p = p; a.n_p = 5;
    const int rc = fn(&a, nullptr);
    hipcheck(hipDeviceSynchronize());
    if (rc != RAD_OK) { std::printf("  %-52s launch %d\n", what, rc); ++failures; return; }
    const std::vector<uint16_t> y = dy.get();
    cs = dcs.get();
    double worst = 0;
    bool ok = true;
    for (int64_t i = 0; i < M * st.n; ++i) {
        const double g = fbf16(y[i]);
        const double e = std::fabs(g - want[i]);
        worst = std::max(worst, e / (std::fabs(want[i]) + 1e-2));
        if (e > 0.01 * std::fabs(want[i]) + 2e-3) ok = false;
    }
    std::printf("  %-52s worst %.2e  %s\n", what, worst, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
    if (g_add) {
        if (rc_add != RAD_OK) { std::printf("  %-52s add launch %d\n", what, rc_add); ++failures; return; }
        double worst_add = 0;
        bool ok_add = cs_add == cs;   /* the state is the plain kernel's */
        for (int64_t i = 0; i < M * st.n; ++i) {
            const double r = fbf16(resid[i]);
            const double e = std::fabs((double)fbf16(y_add[i]) - (r + want[i]));
            worst_add = std::max(worst_add, e / (std::fabs(r + want[i]) + 1e-2));
            if (e > 0.01 * std::fabs(want[i]) + 0.01 * std::fabs(r) + 2e-3) ok_add = false;
        }
        std::printf("  %-52s worst %.2e  %s%s\n", "  ... sconv_res_add", worst_add, ok_add ? "PASS" : "FAIL",
                    cs_add == cs ? "" : " (its state differs from sconv_res's)");
        if (!ok_add) ++failures;
    }
}

/* Ragged prefill: sequences of lengths 1, 37 and 300 in distinct slots, cold. */
void prefill(Fn fn, const Stream& st, const char* what) {
    const std::vector<int> lens = { 1, 37, 300 };
    std::vector<int> idx;
    std::vector<int32_t> cu = { 0 }, cidx, hi;
    std::vector<double> want;
    const int64_t dw = st.heads * st.k;
    int total = 0;
    for (size_t s = 0; s < lens.size(); ++s) {
        for (int i = 0; i < lens[s]; ++i) idx.push_back(total + i);
        total += lens[s];
        cu.push_back(total);
        cidx.push_back((int32_t)(lens.size() - 1 - s));   /* reversed slots */
        hi.push_back(0);
    }
    const std::vector<float> x = rows(st.rows_x, idx, st.n), d = st.heads ? rows(st.rows_d, idx, dw) : std::vector<float>();
    int off = 0;
    for (int64_t L : lens) {
        std::vector<float> xs(x.begin() + off * st.n, x.begin() + (off + L) * st.n);
        std::vector<float> ds = st.heads ? std::vector<float>(d.begin() + off * dw, d.begin() + (off + L) * dw) : std::vector<float>();
        const auto r = reference(xs, L, st.n, st.w, ds, st.heads, st.k);
        want.insert(want.end(), r.begin(), r.end());
        off += (int)L;
    }
    std::vector<uint16_t> cs((size_t)(lens.size() * st.n * st.S), bf16(7.0f));
    std::vector<uint16_t> xb = to_bf16(x), db = to_bf16(d), wb = to_bf16(st.w);
    launch(what, st, xb, wb, cs, (int64_t)lens.size(), cu, st.heads ? &db : nullptr, cidx, &hi, nullptr, want, fn);
}

/* One sequence: a cold prefix, a verify of 4 drafts reading at offset 0, a step after three were
 * kept (offset 2), a decode step after that one row was kept (offset 0), then decode rows. */
void speculative(Fn fn, const Stream& st, const char* what) {
    const int64_t dw = st.heads * st.k;
    std::vector<uint16_t> cs((size_t)(st.n * st.S), bf16(7.0f));
    const std::vector<uint16_t> wb = to_bf16(st.w);
    /* The rows the sequence has committed, which a step's reference reads before its own rows. */
    std::vector<int> committed;
    auto step = [&](const std::vector<int>& r, int hinit, int nacc) {
        const int64_t M = (int64_t)r.size();
        std::vector<int> seq = committed;
        seq.insert(seq.end(), r.begin(), r.end());
        const auto full = reference(rows(st.rows_x, seq, st.n), (int64_t)seq.size(), st.n, st.w,
                                    st.heads ? rows(st.rows_d, seq, dw) : std::vector<float>(), st.heads, st.k);
        std::vector<double> want(full.end() - M * st.n, full.end());
        std::vector<uint16_t> xr = to_bf16(rows(st.rows_x, r, st.n));
        std::vector<uint16_t> dr = st.heads ? to_bf16(rows(st.rows_d, r, dw)) : std::vector<uint16_t>();
        std::vector<int32_t> cu = { 0, (int32_t)M }, ci = { 0 }, hi = { hinit }, na = { nacc };
        launch(what, st, xr, wb, cs, 1, cu, st.heads ? &dr : nullptr, ci, &hi,
               nacc > 0 ? &na : nullptr, want, fn);
    };
    step({ 0, 1, 2 }, 0, 0);                       /* cold prefix: every row commits */
    committed = { 0, 1, 2 };
    step({ 3, 4, 5, 6 }, 1, 1);                    /* a verify of four drafts, read at offset 0 */
    committed = { 0, 1, 2, 3, 4, 5 };              /* drafts 3, 4 and 5 were kept, 6 rejected */
    step({ 7 }, 1, 3);                             /* read at offset 2: history ends at draft 5 */
    committed.push_back(7);
    step({ 8 }, 1, 1);                             /* the row before was kept: offset 0 */
    committed.push_back(8);
    step({ 9 }, 1, 1);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s libr4d.so [prefix]\n", argv[0]); return 77; }
    const std::string pfx = argc > 2 ? argv[2] : "r4d";
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    const std::string entry = pfx + "_rad_sconv_res";
    auto fn = reinterpret_cast<Fn>(dlsym(h, entry.c_str()));
    if (!fn) { std::fprintf(stderr, "dlsym %s: %s\n", entry.c_str(), dlerror()); return 1; }
    g_add = reinterpret_cast<Fn>(dlsym(h, (entry + "_add").c_str()));

    std::printf("sconv_res on the card:\n");
    /* The model's dynamic form: n 1024, 8 heads of 128, k 4. */
    Stream dyn{};
    dyn.n = 1024; dyn.k = 4; dyn.heads = 8; dyn.S = 6;
    dyn.w = wave((size_t)(dyn.n * dyn.k), 0.2);
    dyn.rows_x = wave((size_t)(340 * dyn.n), 0.5);
    dyn.rows_d = wave((size_t)(340 * dyn.heads * dyn.k), 1.7);
    prefill(fn, dyn, "prefill 1, 37, 300 rows, dynamic taps (n 1024)");
    speculative(fn, dyn, "speculative, dynamic taps (n 1024)");

    /* The plain form at n 2304, the other stream's width (no dyn operand). */
    Stream plain{};
    plain.n = 2304; plain.k = 4; plain.heads = 0; plain.S = 6;
    plain.w = wave((size_t)(plain.n * plain.k), 0.9);
    plain.rows_x = wave((size_t)(340 * plain.n), 2.3);
    prefill(fn, plain, "prefill 1, 37, 300 rows, plain (n 2304)");
    speculative(fn, plain, "speculative, plain (n 2304)");

    std::printf("sconv_res on the card: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
