/* Inkling 2's per-group L2 clip (r4d_l2_clip_bf16.hip) through the SIMT emulator: the scale before
 * the norm, the group norm, the position tau, pitched x and y, and y aliasing x. Each output is held
 * to the formula in double over the bf16 values the kernel reads; the kernel rounds once to bf16, so
 * the bound is a bf16 ulp of the reference. The reference is computed before the launch, since an
 * in-place launch overwrites x. */
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
    void d(const char* k, double v) { RadParam x{}; x.key = k; x.kind = RAD_P_F64; x.dval = v; p.push_back(x); }
    void str(const char* k, const char* v) { RadParam x{}; x.key = k; x.kind = RAD_P_STR; x.sval = v; p.push_back(x); }
};

RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape, std::vector<int64_t> stride) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    for (size_t i = 0; i < shape.size(); ++i) { t.shape[i] = shape[i]; t.stride[i] = stride[i]; }
    return t;
}

int failures = 0;
const double kThr = 1.5, kFloor = 2.0, kAlpha = 0.3;

/* One launch on M rows of n columns. x and y have row pitch P >= n; y is a separate buffer unless
 * `inplace`, when it is x. The scale and the positions are present or absent. */
void check(const char* what, int64_t M, int64_t n, int64_t g, int64_t P, bool has_w, bool has_pos,
           bool inplace) {
    std::vector<uint16_t> xb((size_t)(M * P), 0);
    for (int64_t r = 0; r < M; ++r)
        for (int64_t c = 0; c < n; ++c)
            xb[(size_t)(r * P + c)] = rad_f32_to_bf16(3.0f * (float)std::sin(0.7 * (double)(r * n + c) + 0.3));
    std::vector<uint16_t> wb((size_t)n);
    for (int64_t c = 0; c < n; ++c) wb[(size_t)c] = rad_f32_to_bf16(0.5f + 0.5f * (float)std::cos((double)c));
    std::vector<int32_t> pos = { 0, 3, 10, 100, 1000 };
    pos.resize((size_t)M);

    /* The reference, from the bf16 values the kernel reads. */
    std::vector<double> ref((size_t)(M * n));
    for (int64_t r = 0; r < M; ++r) {
        const double tau = has_pos ? 1.0 + kAlpha * std::log(std::max(1.0, (pos[r] + 1.0) / kFloor)) : 1.0;
        for (int64_t g0 = 0; g0 < n; g0 += g) {
            double ss = 0;
            for (int64_t i = g0; i < g0 + g; ++i) {
                const double v = (double)rad_bf16_to_f32(xb[(size_t)(r * P + i)]) *
                                 (has_w ? (double)rad_bf16_to_f32(wb[(size_t)i]) : 1.0);
                ss += v * v;
            }
            const double f = std::min(1.0, kThr / (std::sqrt(ss) + 1e-6));
            for (int64_t i = g0; i < g0 + g; ++i)
                ref[(size_t)(r * n + i)] = (double)rad_bf16_to_f32(xb[(size_t)(r * P + i)]) *
                                           (has_w ? (double)rad_bf16_to_f32(wb[(size_t)i]) : 1.0) * f * tau;
        }
    }

    std::vector<uint16_t> yb((size_t)(M * P), 0);
    RadTensor xt = tens(xb.data(), RAD_BF16, { M, n }, { P, 1 });
    RadTensor wt = tens(wb.data(), RAD_BF16, { n }, { 1 });
    RadTensor pt = tens(pos.data(), RAD_I32, { M }, { 1 });
    RadTensor yt = inplace ? xt : tens(yb.data(), RAD_BF16, { M, n }, { P, 1 });

    RadTensor ts[4] = { xt, has_w ? wt : RadTensor{}, has_pos ? pt : RadTensor{}, yt };
    Params prm;
    prm.i("M", M); prm.i("n", n); prm.i("group", g); prm.d("threshold", kThr); prm.str("dtype", "bf16");
    prm.d("floor", kFloor); prm.d("alpha", kAlpha);
    RadArgs a{}; a.t = ts; a.n_t = 4; a.p = prm.p.data(); a.n_p = (int)prm.p.size();
    const int rc = r4d_rad_l2_clip(&a, nullptr);
    if (rc != RAD_OK) { std::printf("  %-40s launch %d\n", what, rc); ++failures; return; }

    const std::vector<uint16_t>& out = inplace ? xb : yb;
    double worst = 0;
    bool ok = true;
    for (int64_t r = 0; r < M; ++r)
        for (int64_t c = 0; c < n; ++c) {
            const double got = rad_bf16_to_f32(out[(size_t)(r * P + c)]);
            const double want = ref[(size_t)(r * n + c)];
            const double e = std::fabs(got - want);
            worst = std::max(worst, e / (std::fabs(want) + 1e-2));
            if (e > 0.01 * std::fabs(want) + 2e-3) ok = false;
        }
    std::printf("  %-40s worst %.2e  %s\n", what, worst, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

}  // namespace

int main() {
    std::printf("l2_clip (r4d emulator, bf16):\n");
    check("scale and positions", 5, 24, 8, 24, true, true, false);
    check("plain clip, one group per row", 5, 24, 24, 24, false, false, false);
    check("pitched views", 5, 24, 8, 29, true, true, false);
    check("in place", 5, 24, 8, 24, true, true, true);
    check("in place, pitched, positions only", 5, 24, 8, 29, false, true, true);
    std::printf("l2_clip: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
