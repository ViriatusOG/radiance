/* Inkling 2's attn_paged_rel on the card: libr4d's rows (r4d_attn_rel.hip, reached through the
 * plugin's own launch entries) against libref's launch of the same op, on the cases of
 * tests/r4d_emu/test_rel.cpp plus a ragged prefill of three thousand rows. Both plugins are
 * dlopen'd and each row is found by its name in the plugin's kernel table, so the device answer is
 * the one the engine would reach. Build with hipcc, pass libr4d.so and libref.so; exits 77 when no
 * card is visible. */
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
#include <string>
#include <vector>

namespace {

constexpr int HD = 128, BS = 16, DR = 16, E = 1024;

std::mt19937 rng(91);
float nrm() { static std::normal_distribution<float> d(0.f, 1.f); return d(rng); }

void hipcheck(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP %s: %s\n", what, hipGetErrorString(e)); std::exit(1); }
}

/* A device copy of a host vector, freed with the object. */
template <class T> struct Device {
    T* p = nullptr;
    size_t n = 0;
    explicit Device(const std::vector<T>& v) : n(v.size()) {
        hipcheck(hipMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), "malloc");
        hipcheck(hipMemcpy(p, v.data(), n * sizeof(T), hipMemcpyHostToDevice), "h2d");
    }
    ~Device() { (void)hipFree(p); }
    std::vector<T> get() const {
        std::vector<T> v(n);
        hipcheck(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
        return v;
    }
};

struct Case {
    const char* name;
    std::vector<int> qlen, ctx;
    int kvh, gqa, window, extent;
    bool i8 = false;
    bool cu = true;
    int rpitch = 0;
    int sink_every = 0;   /* a sink of 40 in every such row: it must take the row */
    int used = 0;         /* host KV bound (RadArgs::kv_used_max_h); 0 leaves it unset */
    int max_ctx = 0;      /* model bound passed as a parameter; 0 omits it */
    int wgps = 0;
    int want_splits = -1; /* decode split count the card must report; -1 skips the check */
    bool scratch = false;
};

/* Every operand of one case on the host: the same values feed both plugins. */
struct Data {
    int rows = 0, nh = 0, n_seq = 0, maxb = 0, nblocks = 0, rstride = 0;
    std::vector<uint16_t> q, kvb, r, rp, out;
    std::vector<uint8_t> kvi;
    std::vector<float> sk;
    std::vector<int32_t> bt, su, cu;
};

Data make(const Case& c) {
    Data d;
    d.n_seq = (int)c.qlen.size();
    d.nh = c.kvh * c.gqa;
    std::vector<int> cu(d.n_seq + 1, 0), nb(d.n_seq);
    for (int s = 0; s < d.n_seq; ++s) cu[s + 1] = cu[s] + c.qlen[s];
    d.rows = cu[d.n_seq];
    for (int s = 0; s < d.n_seq; ++s) { nb[s] = (c.ctx[s] + BS - 1) / BS; d.maxb = std::max(d.maxb, nb[s]); }
    d.bt.assign((size_t)d.n_seq * d.maxb, -1);
    int base = 0;
    for (int s = 0; s < d.n_seq; ++s) {
        for (int b = 0; b < nb[s]; ++b) d.bt[(size_t)s * d.maxb + b] = base + (nb[s] - 1 - b);
        base += nb[s];
    }
    d.nblocks = base;
    d.su.assign(c.ctx.begin(), c.ctx.end());
    d.cu.assign(cu.begin(), cu.end());

    const size_t slots = (size_t)d.nblocks * c.kvh * BS;
    if (c.i8) {
        d.kvi.assign(slots * (2 * HD + 4), 0);
        for (size_t row = 0; row < slots; ++row)
            for (int half = 0; half < 2; ++half) {
                float x[HD], am = 0.f;
                for (int k = 0; k < HD; ++k) { x[k] = nrm() * (half ? 0.8f : 1.0f); am = std::max(am, std::fabs(x[k])); }
                const uint16_t sh = rad_f32_to_f16(std::min(am / 127.f, 515.f));
                const float sc = rad_f16_to_f32(sh);
                uint8_t* p = d.kvi.data() + row * (2 * HD + 4);
                for (int k = 0; k < HD; ++k) {
                    const float qv = sc > 0 ? std::nearbyint(x[k] / sc) : 0.f;
                    p[half * HD + k] = (uint8_t)(int8_t)std::max(-127.f, std::min(127.f, qv));
                }
                std::memcpy(p + 2 * HD + 2 * half, &sh, 2);
            }
    } else {
        d.kvb.resize(slots * 2 * HD);
        for (auto& v : d.kvb) v = rad_f32_to_bf16(nrm());
    }

    d.q.resize((size_t)d.rows * d.nh * HD);
    for (auto& v : d.q) v = rad_f32_to_bf16(nrm());
    d.rstride = d.nh * DR + c.rpitch;
    d.r.assign((size_t)d.rows * d.rstride, rad_f32_to_bf16(9.0f));   /* the pitch's extra columns */
    for (int t = 0; t < d.rows; ++t)
        for (int k = 0; k < d.nh * DR; ++k) d.r[(size_t)t * d.rstride + k] = rad_f32_to_bf16(nrm());
    d.rp.resize((size_t)DR * E);
    for (auto& v : d.rp) v = rad_f32_to_bf16(0.3f * nrm());
    d.sk.resize((size_t)d.rows * d.nh);
    for (int t = 0; t < d.rows; ++t)
        for (int h = 0; h < d.nh; ++h) {
            float v = 1.5f * nrm();
            if (c.sink_every && t % c.sink_every == 0) v = 40.f;
            d.sk[(size_t)t * d.nh + h] = v;
        }
    d.out.assign((size_t)d.rows * d.nh * HD, 0);
    return d;
}

/* The operands of one case as RadTensors over the given buffers (host or device). */
struct Bufs {
    const void *q, *kv, *bt, *su, *cu, *r, *rp, *sk;
    void* out;
};

RadTensor tensor(const void* p, uint32_t dt, std::vector<int64_t> shape, std::vector<int64_t> strides = {}) {
    RadTensor t{};
    t.data = const_cast<void*>(p);
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[i];
        t.stride[i] = strides.empty() ? st : strides[i];
        st *= shape[i];
    }
    return t;
}

struct Params {
    std::vector<RadParam> p;
    void i(const char* k, long long v) { RadParam x{}; x.key = k; x.kind = RAD_P_INT; x.ival = v; p.push_back(x); }
    void f(const char* k, double v) { RadParam x{}; x.key = k; x.kind = RAD_P_F64; x.dval = v; p.push_back(x); }
    void s(const char* k, const char* v) { RadParam x{}; x.key = k; x.kind = RAD_P_STR; x.sval = v; p.push_back(x); }
};

/* Launch one row (`fn`) over the case's operands. `scratch` is the card's split partials. */
int launch(const Case& c, const Data& d, const Bufs& b, int (*fn)(const RadArgs*, RadStream),
           void* scratch = nullptr, int64_t scratch_bytes = 0) {
    const int kv_w = c.i8 ? 2 * HD + 4 : 2 * HD;
    RadTensor t[11] = {};
    t[0] = tensor(b.q, RAD_BF16, {d.rows, d.nh, HD});
    t[1] = c.i8 ? tensor(b.kv, RAD_I8, {d.nblocks, c.kvh, BS, kv_w})
                : tensor(b.kv, RAD_BF16, {d.nblocks, c.kvh, BS, kv_w});
    t[2] = tensor(b.bt, RAD_I32, {d.n_seq, d.maxb});
    t[3] = tensor(b.su, RAD_I32, {d.n_seq});
    if (c.cu) t[6] = tensor(b.cu, RAD_I32, {d.n_seq + 1});
    t[7] = c.rpitch ? tensor(b.r, RAD_BF16, {d.rows, d.nh, DR}, {d.rstride, DR, 1})
                    : tensor(b.r, RAD_BF16, {d.rows, d.nh, DR});
    t[8] = tensor(b.rp, RAD_BF16, {DR, E});
    t[9] = tensor(b.sk, RAD_F32, {d.rows, d.nh});
    t[10] = tensor(b.out, RAD_BF16, {d.rows, d.nh, HD});

    Params P;
    const int qmax = *std::max_element(c.qlen.begin(), c.qlen.end());
    P.i("q_len", c.cu ? qmax : c.qlen[0]);
    P.i("head_dim", HD); P.i("gqa", c.gqa); P.i("block_size", BS); P.i("causal", 1);
    P.i("window", c.window); P.s("q_dtype", "bf16"); P.s("kv_dtype", c.i8 ? "i8" : "bf16");
    P.i("n_head", d.nh); P.i("max_seqs", d.n_seq); P.i("d_rel", DR); P.i("extent", E);
    P.f("scale", 1.0 / std::sqrt((double)HD));
    if (c.max_ctx > 0) P.i("max_ctx", c.max_ctx);

    RadArgs a{};
    a.t = t;
    a.n_t = 11;
    a.p = P.p.data();
    a.n_p = (int)P.p.size();
    a.scratch = scratch;
    a.scratch_bytes = scratch_bytes;
    a.kv_used_max_h = c.used;
    a.wgp_count = c.wgps;
    return fn(&a, nullptr);
}

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

/* rel_l2 of the device's output against libref's, in f32. */
double rel_l2(const std::vector<uint16_t>& ref, const std::vector<uint16_t>& got) {
    double se = 0, sr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double r = rad_bf16_to_f32(ref[i]), g = rad_bf16_to_f32(got[i]);
        se += (g - r) * (g - r);
        sr += r * r;
    }
    return std::sqrt(se / std::max(sr, 1e-30));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s libr4d.so libref.so\n", argv[0]); return 2; }
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) { std::printf("no card visible: skipped\n"); return 77; }

    void* hr4d = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    void* href = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!hr4d || !href) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    const std::vector<Case> cases = {
        { "decode 1500 keys, gqa 3, full (extent 1024)", { 1 }, { 1500 }, 1, 3, 0, E },
        { "decode 1500 keys, gqa 3, window 1024", { 1 }, { 1500 }, 1, 3, 1024, E },
        { "decode 1500 keys, gqa 8, window 1024", { 1 }, { 1500 }, 1, 8, 1024, E },
        { "decode x2 (1500, 90), gqa 1, 2 kv heads", { 1, 1 }, { 1500, 90 }, 2, 1, 1024, E },
        { "verify 4 and 8 rows, gqa 4, full", { 4, 8 }, { 300, 1500 }, 1, 4, 0, E },
        { "ragged prefill 1500, 1000, 500 rows (2000 keys), gqa 2, full", { 1500, 1000, 500 }, { 2000, 1100, 700 }, 1, 2, 0, E },
        { "ragged prefill 1500, 1000, 500 rows (2000 keys), gqa 2, window 1024", { 1500, 1000, 500 }, { 2000, 1100, 700 }, 1, 2, 1024, E },
        { "int8 cache, gqa 3, full", { 2, 1 }, { 900, 500 }, 1, 3, 0, E, true },
        { "r as a column slice of a wider row, gqa 3", { 3 }, { 700 }, 1, 3, 0, E, false, true, 5 },
        { "sink far above the scores, gqa 2", { 1, 1 }, { 300, 40 }, 1, 2, 0, E, false, true, 0, 1 },
        /* The model bound is 262144. The split must follow the live bound, and a 10000-key
         * one-row launch at four KV heads must take more than one wave of 32 splits. */
        { "split 10000 keys, 4 kv heads, live 10000 not max_ctx", { 1 }, { 10000 }, 4, 3, 0, E,
          false, true, 0, 0, 10000, 262144, 32, 39, true },
        { "split 64 keys stays one piece despite max_ctx 262144", { 1 }, { 64 }, 4, 3, 0, E,
          false, true, 0, 0, 64, 262144, 32, 1, true },
        { "window ignores a 250000 live bound", { 1 }, { 1500 }, 4, 3, 1024, E,
          false, true, 0, 0, 250000, 262144, 32, 4, true },
        { "split 1500 keys, one kv head, five pieces", { 1 }, { 1500 }, 1, 3, 0, E,
          false, true, 0, 0, 1500, 262144, 32, 5, true },
        /* A verify packs its rows and group heads into one WMMA block a KV head; its partials
         * are rows * kv heads * gqa slots, which bounds the splits (21 here), and the last keys
         * carry the bias. */
        { "packed verify 8 rows, 10000 keys, 8 kv heads, gqa 3, split", { 8 }, { 10000 }, 8, 3, 0, E,
          false, true, 0, 0, 10000, 262144, 32, 21, true },
        { "packed verify int8, 4 and 7 rows, window 1024, split", { 4, 7 }, { 3000, 2500 }, 2, 3,
          1024, E, true, true, 0, 0, 3000, 262144, 32, 4, true },
        { "packed verify 5 rows, gqa 8 (three waves), split", { 5 }, { 4000 }, 1, 8, 0, E,
          false, true, 0, 0, 4000, 262144, 32, 15, true },
    };

    int fails = 0;
    double worst = 0;
    for (const Case& c : cases) {
        const Data d = make(c);
        /* libref's answer on the host buffers. */
        int (*ref_fn)(const RadArgs*, RadStream) = find_row(href, "ref_attn_paged_rel");
        if (!ref_fn) { std::fprintf(stderr, "libref has no ref_attn_paged_rel\n"); return 1; }
        std::vector<uint16_t> ref_out(d.out.size(), 0);
        Bufs hb{ d.q.data(), c.i8 ? (const void*)d.kvi.data() : (const void*)d.kvb.data(), d.bt.data(),
                 d.su.data(), c.cu ? (const void*)d.cu.data() : nullptr, d.r.data(), d.rp.data(),
                 d.sk.data(), ref_out.data() };
        if (launch(c, d, hb, ref_fn) != RAD_OK) { std::printf("  %-64s libref refused\n", c.name); ++fails; continue; }

        /* The card's row for this geometry. */
        const std::string row = "attn_rel_h128_gqa" + std::to_string(c.gqa) + (c.i8 ? "_i8kv" : "_bf16kv");
        int (*dev_fn)(const RadArgs*, RadStream) = find_row(hr4d, row.c_str());
        if (!dev_fn) { std::printf("  %-64s no row %s in libr4d\n", c.name, row.c_str()); ++fails; continue; }
        Device<uint16_t> dq(d.q), dr(d.r), drp(d.rp), dout(d.out);
        Device<uint8_t> dkvi(d.kvi);
        Device<uint16_t> dkvb(d.kvb);
        Device<int32_t> dbt(d.bt), dsu(d.su), dcu(d.cu);
        Device<float> dsk(d.sk);
        Bufs db{ dq.p, c.i8 ? (const void*)dkvi.p : (const void*)dkvb.p, dbt.p, dsu.p,
                 c.cu ? (const void*)dcu.p : nullptr, dr.p, drp.p, dsk.p, dout.p };
        void* scratch = nullptr;
        int64_t scratch_bytes = 0;
        if (c.scratch) {
            auto scratch_fn = reinterpret_cast<int64_t (*)(const RadArgs*)>(dlsym(hr4d, "r4d_rad_attn_rel_scratch"));
            if (!scratch_fn) { std::printf("  %-64s no scratch hook\n", c.name); ++fails; continue; }
            scratch_bytes = scratch_fn(nullptr);
            hipcheck(hipMalloc(&scratch, (size_t)scratch_bytes), "scratch");
        }
        if (launch(c, d, db, dev_fn, scratch, scratch_bytes) != RAD_OK) {
            std::printf("  %-64s card row refused\n", c.name); ++fails;
            if (scratch) (void)hipFree(scratch);
            continue;
        }
        hipcheck(hipDeviceSynchronize(), "sync");
        int splits = -1;
        bool splits_ok = true;
        if (c.want_splits >= 0) {
            auto last = reinterpret_cast<int (*)()>(dlsym(hr4d, "r4d_attn_rel_last_splits"));
            splits = last ? last() : -1;
            splits_ok = splits == c.want_splits;
        }
        if (scratch) (void)hipFree(scratch);
        const double e = rel_l2(ref_out, dout.get());
        /* A zero reference would match a zero card output for free: it must be a real answer. */
        double ref_norm = 0;
        for (uint16_t v : ref_out) ref_norm += (double)rad_bf16_to_f32(v) * rad_bf16_to_f32(v);
        const bool ok = e < 2e-3 && ref_norm > 0 && splits_ok;
        worst = std::max(worst, e);
        std::printf("  %-64s rel_l2 %.2e  rows %d  splits %d  %s\n", c.name, e, d.rows, splits,
                    ok ? "PASS" : "FAIL");
        fails += ok ? 0 : 1;
    }
    std::printf("attn_paged_rel on the card: %zu case(s), %d failure(s), max rel_l2 %.3e\n",
                cases.size(), fails, worst);
    return fails ? 1 : 0;
}
