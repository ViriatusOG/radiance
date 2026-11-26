/* l2_clip on the card: libr4d's exported launcher (r4d_rad_l2_clip) against the formula in double,
 * at the model's shapes (per-head norms of 128 over 8 heads, with the scale and positions, and a
 * plain 1024-wide clip). Build with hipcc; pass the path of libr4d.so.
 *
 * Each case runs on the card, in place and pitched, and is held to the formula over the bf16 values
 * the kernel reads. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

using Fn = int (*)(const RadArgs*, RadStream);

namespace {

int failures = 0;

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

RadTensor tensor(void* p, uint32_t dt, std::vector<int64_t> shape, std::vector<int64_t> stride) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    for (size_t i = 0; i < shape.size(); ++i) { t.shape[i] = shape[i]; t.stride[i] = stride[i]; }
    return t;
}

uint16_t bf16(float f) { return rad_f32_to_bf16(f); }
float fbf16(uint16_t b) { return rad_bf16_to_f32(b); }

/* One launch: M rows of n columns in groups of g, x and y at row pitch P, the scale and positions
 * present or absent, y either its own buffer or x. Checked against the formula. */
void run_case(Fn fn, const char* what, int64_t M, int64_t n, int64_t g, int64_t P, bool has_w,
              bool has_pos, bool inplace) {
    const double thr = 1.5, flr = 2.0, alpha = 0.3;
    std::vector<uint16_t> xh((size_t)(M * P), 0);
    for (int64_t r = 0; r < M; ++r)
        for (int64_t c = 0; c < n; ++c)
            xh[(size_t)(r * P + c)] = bf16(3.0f * (float)std::sin(0.37 * (double)(r * n + c) + 0.3));
    std::vector<uint16_t> wh((size_t)n);
    for (int64_t c = 0; c < n; ++c) wh[(size_t)c] = bf16(0.5f + 0.5f * (float)std::cos((double)c));
    std::vector<int32_t> ph((size_t)M);
    for (int64_t r = 0; r < M; ++r) ph[(size_t)r] = (int32_t)((r * 37) % 5000);

    std::vector<double> ref((size_t)(M * n));
    for (int64_t r = 0; r < M; ++r) {
        const double tau = has_pos ? 1.0 + alpha * std::log(std::max(1.0, (ph[r] + 1.0) / flr)) : 1.0;
        for (int64_t g0 = 0; g0 < n; g0 += g) {
            double ss = 0;
            for (int64_t i = g0; i < g0 + g; ++i) {
                const double v = (double)fbf16(xh[(size_t)(r * P + i)]) *
                                 (has_w ? (double)fbf16(wh[(size_t)i]) : 1.0);
                ss += v * v;
            }
            const double f = std::min(1.0, thr / (std::sqrt(ss) + 1e-6));
            for (int64_t i = g0; i < g0 + g; ++i)
                ref[(size_t)(r * n + i)] = (double)fbf16(xh[(size_t)(r * P + i)]) *
                                           (has_w ? (double)fbf16(wh[(size_t)i]) : 1.0) * f * tau;
        }
    }

    Device<uint16_t> dx(xh), dw(wh), dy(std::vector<uint16_t>((size_t)(M * P), 0));
    Device<int32_t> dp(ph);
    RadTensor xt = tensor(dx.p, RAD_BF16, { M, n }, { P, 1 });
    RadTensor wt = tensor(dw.p, RAD_BF16, { n }, { 1 });
    RadTensor pt = tensor(dp.p, RAD_I32, { M }, { 1 });
    RadTensor yt = inplace ? xt : tensor(dy.p, RAD_BF16, { M, n }, { P, 1 });
    RadTensor ts[4] = { xt, has_w ? wt : RadTensor{}, has_pos ? pt : RadTensor{}, yt };

    RadParam p[7]{};
    p[0].key = "M";         p[0].kind = RAD_P_INT; p[0].ival = M;
    p[1].key = "n";         p[1].kind = RAD_P_INT; p[1].ival = n;
    p[2].key = "group";     p[2].kind = RAD_P_INT; p[2].ival = g;
    p[3].key = "threshold"; p[3].kind = RAD_P_F64; p[3].dval = thr;
    p[4].key = "dtype";     p[4].kind = RAD_P_STR; p[4].sval = "bf16";
    p[5].key = "floor";     p[5].kind = RAD_P_F64; p[5].dval = flr;
    p[6].key = "alpha";     p[6].kind = RAD_P_F64; p[6].dval = alpha;
    RadArgs a{};
    a.t = ts; a.n_t = 4; a.p = p; a.n_p = 7;
    const int rc = fn(&a, nullptr);
    hipcheck(hipDeviceSynchronize());
    if (rc != RAD_OK) { std::printf("  %-44s launch %d\n", what, rc); ++failures; return; }

    const std::vector<uint16_t> out = inplace ? dx.get() : dy.get();
    double worst = 0;
    bool ok = true;
    for (int64_t r = 0; r < M; ++r)
        for (int64_t c = 0; c < n; ++c) {
            const double got = fbf16(out[(size_t)(r * P + c)]);
            const double want = ref[(size_t)(r * n + c)];
            const double e = std::fabs(got - want);
            worst = std::max(worst, e / (std::fabs(want) + 1e-2));
            if (e > 0.01 * std::fabs(want) + 2e-3) ok = false;
        }
    std::printf("  %-44s worst %.2e  %s\n", what, worst, ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s libr4d.so\n", argv[0]); return 77; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    auto fn = reinterpret_cast<Fn>(dlsym(h, "r4d_rad_l2_clip"));
    if (!fn) { std::fprintf(stderr, "dlsym r4d_rad_l2_clip: %s\n", dlerror()); return 1; }

    std::printf("l2_clip on the card:\n");
    /* The model's QK clip: 8 heads of 128 per row, the scale, positions. */
    run_case(fn, "QK clip, 8 heads of 128, scale and positions", 300, 1024, 128, 1024, true, true, false);
    /* A pitched view, then in place, at the same shape. */
    run_case(fn, "pitched views, 8 heads of 128", 300, 1024, 128, 1100, true, true, false);
    run_case(fn, "in place, 8 heads of 128", 300, 1024, 128, 1024, true, true, true);
    /* A plain clip over whole rows, with no scale or positions. */
    run_case(fn, "plain clip, one group per row", 64, 2304, 2304, 2304, false, false, false);
    run_case(fn, "in place, pitched, positions only", 64, 1024, 128, 1100, false, true, true);

    std::printf("l2_clip on the card: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
