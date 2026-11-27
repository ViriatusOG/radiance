/* posemb_resample on the card: libr4d's row (r4d_posemb.hip, through the plugin's own launch entry)
 * against libref's row of the same op, over the vision tower's grids and the dtype pairs libr4d
 * serves (bf16 or f32 table, bf16 or f32 output). Then grid_embed_hw, the per-axis resample the
 * engine's tower adds in place (r4d_vit_bf16.hip), over pictures with their class row. Both plugins are dlopen'd and each row is found by
 * its name in the plugin's kernel table, so the device answer is the one the engine would reach.
 * Build with hipcc, pass libr4d.so and libref.so; exits 77 when no card is visible. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <vector>

namespace {

void hipcheck(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP %s: %s\n", what, hipGetErrorString(e)); std::exit(1); }
}

/* A device copy of a host byte buffer, freed with the object. */
struct Device {
    void* p = nullptr;
    size_t bytes = 0;
    explicit Device(const std::vector<uint8_t>& v) : bytes(v.size()) {
        hipcheck(hipMalloc(&p, std::max<size_t>(bytes, 1)), "malloc");
        hipcheck(hipMemcpy(p, v.data(), bytes, hipMemcpyHostToDevice), "h2d");
    }
    ~Device() { (void)hipFree(p); }
    std::vector<uint8_t> get() const {
        std::vector<uint8_t> v(bytes);
        hipcheck(hipMemcpy(v.data(), p, bytes, hipMemcpyDeviceToHost), "d2h");
        return v;
    }
};

struct Case {
    const char* name;
    int64_t S, Th, Tw, n;   /* the square source table of side S, the grid, the width */
    uint32_t tdt, ydt;      /* the table's and the output's dtypes */
};

/* A row of a plugin by its name in the kernel table. */
int (*find_row(void* h, const char* name))(const RadArgs*, RadStream) {
    auto count = reinterpret_cast<int (*)(void)>(dlsym(h, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(h, "rad_kernel_at"));
    if (!count || !at) return nullptr;
    for (int i = 0; i < count(); ++i) {
        const RadKernelInfo* k = at(i);
        if (k && std::strcmp(k->name, name) == 0) return k->launch;
    }
    return nullptr;
}

RadTensor mat(void* p, uint32_t dt, int64_t rows, int64_t cols) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.stride[0] = cols;
    t.stride[1] = 1;
    return t;
}

RadParam pint(const char* k, long long v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
RadParam pstr(const char* k, const char* v) { RadParam p{}; p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }

/* One launch of a row over the given table and output buffers. */
int launch(int (*fn)(const RadArgs*, RadStream), const Case& c, void* tab, void* out) {
    RadTensor t[2] = { mat(tab, c.tdt, c.S * c.S, c.n), mat(out, c.ydt, c.Th * c.Tw, c.n) };
    RadParam p[6] = { pint("S_h", c.S), pint("S_w", c.S), pint("T_h", c.Th), pint("T_w", c.Tw),
                      pint("n", c.n), pstr("dtype", c.ydt == RAD_F32 ? "f32" : "bf16") };
    RadArgs a{};
    a.t = t;
    a.n_t = 2;
    a.p = p;
    a.n_p = 6;
    return fn(&a, nullptr);
}

/* The values of a dtype's buffer, widened to f32. */
std::vector<float> widen(const std::vector<uint8_t>& raw, uint32_t dt) {
    std::vector<float> v;
    if (dt == RAD_F32) {
        v.resize(raw.size() / 4);
        std::memcpy(v.data(), raw.data(), raw.size());
    } else {
        v.resize(raw.size() / 2);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(raw.data());
        for (size_t i = 0; i < v.size(); ++i) v[i] = rad_bf16_to_f32(h[i]);
    }
    return v;
}

/* The table as stored: random normals, rounded to the table's dtype. */
std::vector<uint8_t> table_bytes(int64_t count, uint32_t dt, std::mt19937& rng) {
    std::normal_distribution<float> d(0.f, 1.f);
    if (dt == RAD_F32) {
        std::vector<float> v((size_t)count);
        for (auto& x : v) x = d(rng);
        std::vector<uint8_t> b(v.size() * 4);
        std::memcpy(b.data(), v.data(), b.size());
        return b;
    }
    std::vector<uint16_t> v((size_t)count);
    for (auto& x : v) x = rad_f32_to_bf16(d(rng));
    std::vector<uint8_t> b(v.size() * 2);
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s libr4d.so libref.so\n", argv[0]); return 2; }
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) { std::printf("no card visible: skipped\n"); return 77; }
    void* hr4d = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    void* href = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!hr4d || !href) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    int (*dev_fn)(const RadArgs*, RadStream) = find_row(hr4d, "posemb_resample");
    int (*ref_fn)(const RadArgs*, RadStream) = find_row(href, "ref_posemb_resample");
    if (!dev_fn || !ref_fn) { std::fprintf(stderr, "posemb_resample row missing (libr4d %p, libref %p)\n", (void*)dev_fn, (void*)ref_fn); return 1; }

    const uint32_t F = RAD_F32, B = RAD_BF16;
    const std::vector<Case> cases = {
        { "32 x 32 onto itself, f32", 32, 32, 32, 72, F, F },
        { "32 x 32 onto 15 x 25, bf16 table and output", 32, 15, 25, 72, B, B },
        { "32 x 32 onto 103 x 103 (a 4096 px image in 40 px cells), bf16 table, f32 out", 32, 103, 103, 64, B, F },
        { "32 x 32 onto 1 x 1 and 2 x 7, f32", 32, 1, 1, 72, F, F },
        { "32 x 32 onto 2 x 7, f32 table, bf16 out", 32, 2, 7, 72, F, B },
        { "6 x 6 onto 8 x 10 (rectangular), f32", 6, 8, 10, 72, F, F },
    };
    std::mt19937 rng(77);
    int fails = 0;
    for (const Case& c : cases) {
        const int64_t rows = c.Th * c.Tw, count = rows * c.n;
        const std::vector<uint8_t> tab = table_bytes(c.S * c.S * c.n, c.tdt, rng);
        /* libref's answer on the host buffers. */
        std::vector<uint8_t> ref_out((size_t)count * (c.ydt == RAD_F32 ? 4 : 2), 0);
        std::vector<uint8_t> tab_host = tab;
        if (launch(ref_fn, c, tab_host.data(), ref_out.data()) != RAD_OK) {
            std::printf("  %-74s libref refused\n", c.name); ++fails; continue;
        }
        /* The card's row over the same table. */
        Device dtab(tab), dout(std::vector<uint8_t>(ref_out.size(), 0xAB));
        if (launch(dev_fn, c, dtab.p, dout.p) != RAD_OK) {
            std::printf("  %-74s card row refused\n", c.name); ++fails; continue;
        }
        hipcheck(hipDeviceSynchronize(), "sync");
        const std::vector<float> want = widen(ref_out, c.ydt), got = widen(dout.get(), c.ydt);
        double se = 0, sr = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            se += ((double)got[i] - want[i]) * ((double)got[i] - want[i]);
            sr += (double)want[i] * want[i];
        }
        const double rel = std::sqrt(se / std::max(sr, 1e-30));
        /* bf16 output rounds once (2^-9 a value); f32 output must agree to the f32 summation order. A
         * zero reference would match a zero card output for free, so it must be a real answer. */
        const double tol = c.ydt == RAD_F32 ? 1e-5 : 4e-3;
        const bool ok = rel < tol && sr > 0;
        std::printf("  %-74s rel_l2 %.2e  (tol %.0e)  %s\n", c.name, rel, tol, ok ? "PASS" : "FAIL");
        fails += ok ? 0 : 1;
    }
    std::printf("posemb_resample on the card: %zu case(s), %d failure(s)\n", cases.size(), fails);

    /* grid_embed_hw: x [M, n] bf16 in place, the 32 x 32 table bf16, coord [4, M] i32 for one or
     * two pictures, each a class row at (-1, -1) then its h x w patches, and the class prefix row. */
    int (*hw_dev)(const RadArgs*, RadStream) = find_row(hr4d, "grid_embed_hw_bf16");
    int (*hw_ref)(const RadArgs*, RadStream) = find_row(href, "ref_grid_embed_hw");
    if (!hw_dev || !hw_ref) { std::fprintf(stderr, "grid_embed_hw row missing\n"); return 1; }
    struct Pics { const char* name; std::vector<std::pair<int, int>> grids; };
    const std::vector<Pics> pics = {
        { "32 x 32 (the table's own grid)", { { 32, 32 } } },
        { "22 x 32 and 8 x 12 (two pictures, shrink and grow)", { { 22, 32 }, { 8, 12 } } },
        { "64 x 40 (both axes grow, unequally)", { { 64, 40 } } },
    };
    const int64_t S = 32, n = 128;
    int hw_fails = 0;
    for (const Pics& pc : pics) {
        std::vector<int32_t> r, col, gh, gw;
        for (auto [h, w] : pc.grids) {
            r.push_back(-1); col.push_back(-1); gh.push_back(h); gw.push_back(w);
            for (int i = 0; i < h * w; ++i) {
                r.push_back(i / w); col.push_back(i % w); gh.push_back(h); gw.push_back(w);
            }
        }
        const int64_t M = (int64_t)r.size();
        std::vector<uint8_t> crd((size_t)M * 16);
        int32_t* cp = reinterpret_cast<int32_t*>(crd.data());
        for (int64_t i = 0; i < M; ++i) {
            cp[i] = r[(size_t)i]; cp[M + i] = col[(size_t)i]; cp[2 * M + i] = gh[(size_t)i];
            cp[3 * M + i] = gw[(size_t)i];
        }
        const std::vector<uint8_t> tab = table_bytes(S * S * n, B, rng), pre = table_bytes(n, B, rng);
        const std::vector<uint8_t> x0 = table_bytes(M * n, B, rng);
        auto run = [&](int (*fn)(const RadArgs*, RadStream), void* x, void* t, void* c, void* pr) {
            RadTensor ts[4] = { mat(x, B, M, n), mat(t, B, S * S, n), {}, mat(pr, B, 1, n) };
            ts[2].data = c; ts[2].dtype = RAD_I32; ts[2].rank = 2;
            ts[2].shape[0] = 4; ts[2].shape[1] = M; ts[2].stride[0] = M; ts[2].stride[1] = 1;
            RadParam p[5] = { pint("M", M), pint("n", n), pint("side", S), pint("n_prefix", 1),
                              pstr("dtype", "bf16") };
            RadArgs a{};
            a.t = ts; a.n_t = 4; a.p = p; a.n_p = 5;
            return fn(&a, nullptr);
        };
        std::vector<uint8_t> want_x = x0, tab_h = tab, crd_h = crd, pre_h = pre;
        if (run(hw_ref, want_x.data(), tab_h.data(), crd_h.data(), pre_h.data()) != RAD_OK) {
            std::printf("  %-74s libref refused\n", pc.name); ++hw_fails; continue;
        }
        Device dx(x0), dt(tab), dc(crd), dp(pre);
        if (run(hw_dev, dx.p, dt.p, dc.p, dp.p) != RAD_OK) {
            std::printf("  %-74s card row refused\n", pc.name); ++hw_fails; continue;
        }
        hipcheck(hipDeviceSynchronize(), "sync");
        const std::vector<float> want = widen(want_x, B), got = widen(dx.get(), B),
                                 base = widen(x0, B);
        /* Compared on the added term, which is what the op computes; the add rounds once more. */
        double se = 0, sr = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            se += ((double)got[i] - want[i]) * ((double)got[i] - want[i]);
            sr += ((double)want[i] - base[i]) * ((double)want[i] - base[i]);
        }
        const double rel = std::sqrt(se / std::max(sr, 1e-30));
        const bool ok = rel < 8e-3 && sr > 0;
        std::printf("  %-74s rel_l2 %.2e  (tol 8e-03)  %s\n", pc.name, rel, ok ? "PASS" : "FAIL");
        hw_fails += ok ? 0 : 1;
    }
    std::printf("grid_embed_hw on the card: %zu case(s), %d failure(s)\n", pics.size(), hw_fails);
    return fails || hw_fails ? 1 : 0;
}
