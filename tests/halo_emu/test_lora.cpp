/* tests/halo_emu (from tests/r4d_emu/test_lora.cpp): lora_shrink / lora_expand (halo_lora_bf16.hip) on the host SIMT emulator.
 *
 * The case matrix and every check are lora_cases.h's, shared with tests/lora_ops_test.cpp (libref and
 * libavx): base rows untouched by bytes, adapter rows against a double-precision sum within the
 * bounds stated there. Here the implementations are the emulated kernels (the scalar rows and the
 * tiled rows: WMMA shrink, LDS-staged expand), ref_host.cpp's restatement of libref's arithmetic,
 * and libref itself when $RADIANCE_HOME holds one (ctest sets it). Every emulated expand is held
 * to libref's restatement bit for bit, because its j-ascending sum and its roundings are libref's.
 * Shrink is held to the bound rather than to libref's bytes: the scalar row sums k in 32 lane
 * strides and a butterfly, the WMMA row in 32-wide steps over waves and a K split.
 *
 * Then what the tiled rows promise beyond the matrix, each proven here on the emulator:
 *
 *   shapes     the real ones (K 2048 / 5120 / 6144 / 17408, R 16 / 48 / 64 / 192, N 256 ..
 *              12288, odd M, S 1..4, bf16 and E4M3 with a scale every 128): shrink to the f64
 *              bound (and its worst error printed against f64), expand to libref's bytes.
 *   batch      a row computed alone (M = 1) and as row p of M in {7, 16, 17, 37, 300}, the others
 *              random slots and base rows: its t and y bits are the same. Every K split, z fan-out
 *              and pass split the tiled rows can take differs between those launches.
 *   layout     the same rows through aligned (16-byte loads) and unaligned (element loads)
 *              buffers: the same bits.
 *   base rows  y filled with -0, NaN payloads and random values, t with a sentinel: a base row's
 *              bytes are untouched, and a launch of base rows only touches nothing.
 *   decode     every fp8 E4M3 / E5M2 / i8 code through a one-hot pool: t is the code times its
 *              group's scale exactly, so the WMMA row's code -> bf16 widening is the ABI's.
 */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "halo_plugin.h"
#include "ref_host.h"
#include "lora_cases.h"

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <string>
#include <vector>

using lora_cases::Impl;
using lora_cases::LaunchFn;
using lora_cases::mk;
using lora_cases::Params;

static int host_shrink(const RadArgs* a, RadStream) { return host_lora_shrink(a); }
static int host_expand(const RadArgs* a, RadStream) { return host_lora_expand(a); }

/* libref's two rows, through the plugin's own exports, or nulls. */
static void load_libref(LaunchFn* shrink, LaunchFn* expand) {
    *shrink = *expand = nullptr;
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home || !*home) return;
    const std::string p = std::string(home) + "/kernels/libref.so";
    void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) return;
    auto open = (int (*)(void))dlsym(h, "rad_plugin_open");      /* as the loader does */
    if (open && open() < 0) return;
    auto cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt || !at) return;
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* k = at(i);
        if (!k || !k->op) continue;
        if (!std::strcmp(k->op, "lora_shrink")) *shrink = k->launch;
        if (!std::strcmp(k->op, "lora_expand")) *expand = k->launch;
    }
}

static int g_fails = 0;
static void fail(const char* what, const std::string& detail) {
    ++g_fails;
    if (g_fails <= 40) std::printf("FAIL %s: %s\n", what, detail.c_str());
}
static std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
static std::string fmt(const char* f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

static const uint32_t kSent = 0x7fc0deadu;

/* ------------------------------------------------------------------ one shrink problem
 * x rows at row stride `lda` (from element `off` of the buffer), the scale stream [M, groups] at
 * `lds`, dense pools, t dense. Built once a test and launched through any shrink row. */
struct Shrink {
    int M = 0, K = 0, R = 0, S = 0, group = 128, lda = 0, off = 0, lds = 0;
    uint32_t adt = RAD_BF16;
    std::vector<uint8_t> abuf;      /* (off + M * lda) elements of adt; 16-byte aligned base */
    std::vector<float> sbuf;
    std::vector<int32_t> slot;
    std::vector<uint16_t> la;
};

static size_t esz(uint32_t dt) { return (size_t)rad_dtype_bits(dt) / 8; }

/* The x element of row m, column k, as the contract defines it (code times scale, one rounding). */
static float x_of(const Shrink& P, int m, int k) {
    float x = lora_cases::get(P.adt, P.abuf.data(), (size_t)(P.off + (size_t)m * P.lda + k));
    if (lora_cases::coded(P.adt)) x = x * P.sbuf[(size_t)m * P.lds + k / P.group];
    return x;
}

/* Run `fn` (with its scratch hook) over P; t comes back [M, R], base rows holding kSent. */
static int run_shrink(const Shrink& P, LaunchFn fn, lora_cases::ScratchFn hook,
                      std::vector<float>* t) {
    const bool sc = lora_cases::coded(P.adt);
    const int groups = (P.K + P.group - 1) / P.group;
    RadTensor ts[5]{};
    ts[0] = mk((void*)(P.abuf.data() + (size_t)P.off * esz(P.adt)), P.adt, {P.M, P.K}, {P.lda, 1});
    if (sc) ts[1] = mk((void*)P.sbuf.data(), RAD_F32, {P.M, groups}, {P.lds, 1});
    ts[2] = mk((void*)P.slot.data(), RAD_I32, {P.M}, {1});
    ts[3] = mk((void*)P.la.data(), RAD_BF16, {P.S, P.R, P.K}, {(int64_t)P.R * P.K, P.K, 1});
    t->assign((size_t)P.M * P.R, 0.0f);
    for (auto& v : *t) std::memcpy(&v, &kSent, 4);
    ts[4] = mk(t->data(), RAD_F32, {P.M, P.R}, {P.R, 1});
    Params Q;
    Q.i("M", P.M); Q.i("K", P.K); Q.i("R", P.R); Q.i("S", P.S); Q.i("group", sc ? P.group : 0);
    Q.s("dtype", rad_dtype_name(P.adt));
    RadArgs a{};
    a.t = ts; a.n_t = 5; a.p = Q.p.data(); a.n_p = (int)Q.p.size(); a.world_size = 1;
    std::vector<uint8_t> scr;
    lora_cases::with_scratch(a, hook, scr);
    return fn(&a, nullptr);
}

/* Random x, scales, pools; slots as given. `aligned`: row stride a multiple of 16 elements and no
 * offset, so the tiled row's 16-byte loads are taken; else K + 7 and 3. */
static Shrink make_shrink(std::mt19937& rng, int M, int K, int R, int S, uint32_t adt,
                          const std::vector<int32_t>& slot, bool aligned) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.01f, 0.05f);
    Shrink P;
    P.M = M; P.K = K; P.R = R; P.S = S; P.adt = adt;
    P.lda = aligned ? (K + 15) / 16 * 16 + 16 : K + 7;
    P.off = aligned ? 0 : 3;
    const int groups = (K + P.group - 1) / P.group;
    P.lds = groups + 3;
    const float amp = adt == RAD_I8 ? 40.0f : (lora_cases::coded(adt) ? 4.0f : 1.0f);
    P.abuf.assign(((size_t)P.off + (size_t)M * P.lda) * esz(adt) + 64, 0);
    for (size_t i = 0; i < (size_t)P.off + (size_t)M * P.lda; ++i)
        lora_cases::put(adt, P.abuf.data(), i, amp * nd(rng));
    P.sbuf.resize((size_t)M * P.lds);
    for (auto& v : P.sbuf) v = ud(rng);
    P.slot = slot;
    P.la.resize((size_t)S * R * K);
    for (auto& v : P.la) v = rad_f32_to_bf16(0.05f * nd(rng));
    return P;
}

/* Copy row `src` of Q (x and scales) into row `dst` of P, which must have Q's K and dtype. */
static void copy_row(Shrink& P, int dst, const Shrink& Q, int src) {
    const size_t e = esz(P.adt);
    std::memcpy(P.abuf.data() + ((size_t)P.off + (size_t)dst * P.lda) * e,
                Q.abuf.data() + ((size_t)Q.off + (size_t)src * Q.lda) * e, (size_t)P.K * e);
    const int groups = (P.K + P.group - 1) / P.group;
    std::memcpy(&P.sbuf[(size_t)dst * P.lds], &Q.sbuf[(size_t)src * Q.lds], (size_t)groups * 4);
}

/* t against the f64 sum: base rows keep the sentinel, adapter rows within (K + 2) 2^-24 sum|terms|.
 * Returns the worst |err| / bound and the worst |err| / sum|terms| (the error the report quotes). */
static void check_shrink(const char* who, const Shrink& P, const std::vector<float>& t,
                         double* worst_bound, double* worst_rel) {
    for (int m = 0; m < P.M; ++m) {
        const int32_t s = P.slot[(size_t)m];
        const bool base = s < 0 || s >= P.S;
        for (int j = 0; j < P.R; ++j) {
            const float v = t[(size_t)m * P.R + j];
            if (base) {
                if (std::memcmp(&v, &kSent, 4))
                    fail(who, fmt("shrink wrote base row %d (M %d K %d R %d)", m, P.M, P.K, P.R));
                continue;
            }
            double sum = 0, mag = 0;
            const uint16_t* ar = &P.la[((size_t)s * P.R + j) * P.K];
            for (int k = 0; k < P.K; ++k) {
                const double term = (double)x_of(P, m, k) * rad_bf16_to_f32(ar[k]);
                sum += term; mag += std::fabs(term);
            }
            const double tol = (P.K + 2) * std::ldexp(1.0, -24) * mag + 1e-30;
            const double e = std::fabs((double)v - sum);
            *worst_bound = std::max(*worst_bound, e / tol);
            if (mag > 0) *worst_rel = std::max(*worst_rel, e / mag);
            if (!(e <= tol))
                fail(who, fmt("shrink value M %d K %d R %d %s row %d col %d: %g vs %g (tol %g)",
                              P.M, P.K, P.R, rad_dtype_name(P.adt), m, j, (double)v, sum, tol));
        }
    }
}

/* ------------------------------------------------------------------ one expand problem */
struct Expand {
    int M = 0, N = 0, R = 0, S = 0, ldy = 0, yoff = 0;
    uint32_t ydt = RAD_BF16;
    std::vector<float> t;            /* [M, R] */
    std::vector<int32_t> slot;
    std::vector<uint16_t> lb;        /* [S, N, R] */
    std::vector<float> sc;           /* [S, N] */
    std::vector<uint8_t> y0;         /* (yoff + M * ldy) elements */
};

static Expand make_expand(std::mt19937& rng, int M, int N, int R, int S, uint32_t ydt,
                          const std::vector<int32_t>& slot, bool aligned) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.5f, 2.0f);
    std::uniform_int_distribution<int> pick(0, 5);
    Expand P;
    P.M = M; P.N = N; P.R = R; P.S = S; P.ydt = ydt; P.slot = slot;
    P.ldy = aligned ? N : N + 5;
    P.yoff = aligned ? 0 : 2;
    P.t.resize((size_t)M * R);
    for (auto& v : P.t) v = 0.5f * nd(rng);
    P.lb.resize((size_t)S * N * R);
    for (auto& v : P.lb) v = rad_f32_to_bf16(0.05f * nd(rng));
    P.sc.resize((size_t)S * N);
    for (auto& v : P.sc) v = ud(rng);
    const size_t ny = (size_t)P.yoff + (size_t)M * P.ldy, e = esz(ydt);
    P.y0.resize(ny * e);
    /* -0, NaNs with payloads (quiet and signalling) and ordinary values: a base row must keep
     * every one of them, bit for bit. */
    for (size_t i = 0; i < ny; ++i) {
        const int k = pick(rng);
        uint8_t* p = &P.y0[i * e];
        if (k == 0) {
            lora_cases::put(ydt, P.y0.data(), i, -0.0f);
        } else if (k == 1 && ydt == RAD_F32) {
            const uint32_t u = 0x7fa00001u + (uint32_t)(i & 0xff);
            std::memcpy(p, &u, 4);
        } else if (k == 1) {
            const uint16_t u = ydt == RAD_BF16 ? (uint16_t)(0xffa1u + (i & 7)) : (uint16_t)0x7d01u;
            std::memcpy(p, &u, 2);
        } else {
            lora_cases::put(ydt, P.y0.data(), i, nd(rng));
        }
    }
    return P;
}

static int run_expand(const Expand& P, LaunchFn fn, std::vector<uint8_t>* y) {
    *y = P.y0;
    const size_t e = esz(P.ydt);
    RadTensor ts[5]{};
    ts[0] = mk((void*)P.t.data(), RAD_F32, {P.M, P.R}, {P.R, 1});
    ts[1] = mk((void*)P.slot.data(), RAD_I32, {P.M}, {1});
    ts[2] = mk((void*)P.lb.data(), RAD_BF16, {P.S, P.N, P.R}, {(int64_t)P.N * P.R, P.R, 1});
    ts[3] = mk((void*)P.sc.data(), RAD_F32, {P.S, P.N}, {P.N, 1});
    ts[4] = mk(y->data() + (size_t)P.yoff * e, P.ydt, {P.M, P.N}, {P.ldy, 1});
    Params Q;
    Q.i("M", P.M); Q.i("N", P.N); Q.i("R", P.R); Q.i("S", P.S);
    Q.s("dtype", rad_dtype_name(P.ydt));
    RadArgs a{};
    a.t = ts; a.n_t = 5; a.p = Q.p.data(); a.n_p = (int)Q.p.size(); a.world_size = 1;
    return fn(&a, nullptr);
}

/* y against reference bytes `want` (an implementation of libref's order on the same operands),
 * and base rows plus every element outside the slice against the input. */
static void check_expand(const char* who, const Expand& P, const std::vector<uint8_t>& y,
                         const std::vector<uint8_t>& want) {
    const size_t e = esz(P.ydt), ny = (size_t)P.yoff + (size_t)P.M * P.ldy;
    int64_t bad_base = 0, bad_val = 0;
    for (size_t i = 0; i < ny; ++i) {
        const long long q = (long long)i - P.yoff;
        const int m = q < 0 ? -1 : (int)(q / P.ldy), n = q < 0 ? -1 : (int)(q % P.ldy);
        const bool inside = m >= 0 && m < P.M && n < P.N && P.slot[(size_t)m] >= 0 &&
                            P.slot[(size_t)m] < P.S;
        if (!inside) bad_base += std::memcmp(&y[i * e], &P.y0[i * e], e) != 0;
        else bad_val += std::memcmp(&y[i * e], &want[i * e], e) != 0;
    }
    if (bad_base)
        fail(who, fmt("expand touched %lld base-row / outside bytes (M %d N %d R %d %s)",
                      (long long)bad_base, P.M, P.N, P.R, rad_dtype_name(P.ydt)));
    if (bad_val)
        fail(who, fmt("expand: %lld outputs not libref's bytes (M %d N %d R %d %s)",
                      (long long)bad_val, P.M, P.N, P.R, rad_dtype_name(P.ydt)));
}

/* ------------------------------------------------------------------ the tests */

struct Rows {
    LaunchFn ref_shrink, ref_expand;       /* libref or ref_host: the bytes expand is held to */
};

/* Slots for M rows over S adapters, with base rows (-1 and S) mixed in. */
static std::vector<int32_t> rand_slots(std::mt19937& rng, int M, int S, int base_pct) {
    std::uniform_int_distribution<int> u(0, 99), s(0, S - 1), b(0, 2);
    std::vector<int32_t> v((size_t)M);
    for (auto& x : v) {
        if (u(rng) < base_pct) {
            const int k = b(rng);
            x = k == 0 ? -1 : (k == 1 ? S : -7);
        } else {
            x = s(rng);
        }
    }
    return v;
}

/* Real shapes. Each pair is shrink -> expand, as the engine issues it: expand reads the t the
 * tiled shrink wrote, and is held to libref's bytes over that same t. */
static void test_shapes(const Rows& R) {
    std::mt19937 rng(23);
    struct Sh { int M, K, R, N, S; uint32_t adt, ydt; bool al; };
    const Sh shapes[] = {
        { 9,   2048, 48,  2560,  4, RAD_BF16,   RAD_BF16, true  },   /* qkv, decode-ish */
        { 37,  2048, 32,  12288, 3, RAD_BF16,   RAD_BF16, true  },   /* gate_up */
        { 17,  6144, 16,  2048,  4, RAD_BF16,   RAD_BF16, false },   /* down, unaligned */
        { 5,   17408, 16, 256,   2, RAD_BF16,   RAD_F32,  true  },   /* the deepest K */
        { 7,   5120, 48,  1024,  4, RAD_F8E4M3, RAD_BF16, true  },   /* fp8, group 128 */
        { 33,  5120, 64,  640,   1, RAD_F8E4M3, RAD_F16,  false },   /* S = 1 */
        { 3,   2048, 192, 384,   4, RAD_BF16,   RAD_BF16, true  },   /* 3 modules x 64 */
        { 300, 2048, 48,  256,   2, RAD_BF16,   RAD_BF16, true  },   /* prefill-ish */
        { 131, 1000, 64,  1000,  3, RAD_I8,     RAD_F32,  false },   /* partial group, odd */
        { 21,  6144, 48,  300,   4, RAD_F8E5M2, RAD_BF16, true  },
    };
    double wb = 0, wr = 0;
    for (const Sh& sh : shapes) {
        const std::vector<int32_t> slot = rand_slots(rng, sh.M, sh.S, 25);
        const Shrink P = make_shrink(rng, sh.M, sh.K, sh.R, sh.S, sh.adt, slot, sh.al);
        std::vector<float> t;
        const int rc = run_shrink(P, halo_rad_lora_shrink_wmma, halo_rad_lora_shrink_wmma_scratch, &t);
        if (rc != RAD_OK) { fail("shapes", fmt("shrink_wmma rc %d", rc)); continue; }
        check_shrink("shapes/shrink_wmma", P, t, &wb, &wr);
        /* expand over the t just computed; base rows' t is the sentinel, never read */
        Expand E = make_expand(rng, sh.M, sh.N, sh.R, sh.S, sh.ydt, slot, sh.al);
        E.t = t;
        std::vector<uint8_t> want, got;
        if (run_expand(E, R.ref_expand, &want) != RAD_OK) { fail("shapes", "reference expand"); continue; }
        if (run_expand(E, halo_rad_lora_expand_tiled, &got) != RAD_OK) {
            fail("shapes", "expand_tiled rc");
            continue;
        }
        check_expand("shapes/expand_tiled", E, got, want);
    }
    std::printf("lora shapes: %zu shrink -> expand pairs; tiled shrink worst |err|/bound %.4f, "
                "worst |err| / sum|terms| %.3g (2^-24 = %.3g)\n",
                sizeof(shapes) / sizeof(shapes[0]), wb, wr, std::ldexp(1.0, -24));
}

/* Batch invariance: a row alone and inside batches of other slots and base rows. */
static void test_invariance() {
    std::mt19937 rng(41);
    struct Sh { int K, R, N, S; uint32_t adt, ydt; };
    const Sh shapes[] = {
        { 2048, 48, 2560, 4, RAD_BF16,   RAD_BF16 },   /* 4 K splits x 3 j tiles */
        { 6144, 16, 640,  3, RAD_BF16,   RAD_F32  },   /* 12 K splits */
        { 1000, 64, 300,  4, RAD_F8E4M3, RAD_BF16 },   /* coded, partial group, 2 splits */
        { 300,  20, 130,  2, RAD_BF16,   RAD_F16  },   /* no split, ragged R and N */
    };
    const int Ms[] = { 7, 16, 17, 37, 300 };
    int compared = 0;
    for (const Sh& sh : shapes) {
        for (int al = 0; al < 2; ++al) {
            /* The target row: one per slot, alone. */
            for (int s = 0; s < sh.S; ++s) {
                const std::vector<int32_t> one = { s };
                const Shrink P1 = make_shrink(rng, 1, sh.K, sh.R, sh.S, sh.adt, one, al);
                std::vector<float> t1, t1s;
                if (run_shrink(P1, halo_rad_lora_shrink_wmma, halo_rad_lora_shrink_wmma_scratch,
                               &t1) != RAD_OK ||
                    run_shrink(P1, halo_rad_lora_shrink, nullptr, &t1s) != RAD_OK) {
                    fail("invariance", "M=1 launch");
                    continue;
                }
                Expand E1 = make_expand(rng, 1, sh.N, sh.R, sh.S, sh.ydt, one, al);
                std::vector<uint8_t> y1, y1s;
                run_expand(E1, halo_rad_lora_expand_tiled, &y1);
                run_expand(E1, halo_rad_lora_expand, &y1s);
                if (y1 != y1s) fail("invariance", "tiled and scalar expand differ at M=1");
                for (int M : Ms) {
                    const std::vector<int32_t> slot0 = rand_slots(rng, M, sh.S, 30);
                    for (int p : { 0, M / 2, M - 1 }) {
                        std::vector<int32_t> slot = slot0;
                        slot[(size_t)p] = s;
                        /* the same pools; x and t of row p are the solo row's */
                        Shrink PB = make_shrink(rng, M, sh.K, sh.R, sh.S, sh.adt, slot, al != 0);
                        PB.la = P1.la;
                        copy_row(PB, p, P1, 0);
                        std::vector<float> tb;
                        if (run_shrink(PB, halo_rad_lora_shrink_wmma,
                                       halo_rad_lora_shrink_wmma_scratch, &tb) != RAD_OK) {
                            fail("invariance", "batch launch");
                            continue;
                        }
                        if (std::memcmp(&tb[(size_t)p * sh.R], t1.data(), (size_t)sh.R * 4))
                            fail("invariance", fmt("shrink_wmma: row %d of M %d (K %d R %d %s, "
                                                   "%s) is not its M=1 bits", p, M, sh.K, sh.R,
                                                   rad_dtype_name(sh.adt),
                                                   al ? "aligned" : "unaligned"));
                        double wb = 0, wr = 0;
                        check_shrink("invariance/shrink_wmma", PB, tb, &wb, &wr);
                        Expand EB = make_expand(rng, M, sh.N, sh.R, sh.S, sh.ydt, slot, al != 0);
                        EB.lb = E1.lb;
                        EB.sc = E1.sc;
                        std::memcpy(&EB.t[(size_t)p * sh.R], E1.t.data(), (size_t)sh.R * 4);
                        const size_t e = esz(sh.ydt);
                        std::memcpy(&EB.y0[((size_t)EB.yoff + (size_t)p * EB.ldy) * e],
                                    &E1.y0[(size_t)E1.yoff * e], (size_t)sh.N * e);
                        std::vector<uint8_t> yb, ys;
                        run_expand(EB, halo_rad_lora_expand_tiled, &yb);
                        run_expand(EB, halo_rad_lora_expand, &ys);
                        if (std::memcmp(&yb[((size_t)EB.yoff + (size_t)p * EB.ldy) * e],
                                        &y1[(size_t)E1.yoff * e], (size_t)sh.N * e))
                            fail("invariance", fmt("expand_tiled: row %d of M %d (N %d R %d) is "
                                                   "not its M=1 bits", p, M, sh.N, sh.R));
                        check_expand("invariance/expand_tiled", EB, yb, ys);
                        compared += 2;
                    }
                }
            }
        }
    }
    /* The layouts against each other: one problem through aligned and unaligned buffers. */
    for (const Sh& sh : shapes) {
        const std::vector<int32_t> slot = rand_slots(rng, 37, sh.S, 20);
        const Shrink A = make_shrink(rng, 37, sh.K, sh.R, sh.S, sh.adt, slot, true);
        Shrink U = make_shrink(rng, 37, sh.K, sh.R, sh.S, sh.adt, slot, false);
        U.la = A.la;
        for (int m = 0; m < 37; ++m) copy_row(U, m, A, m);
        std::vector<float> ta, tu;
        run_shrink(A, halo_rad_lora_shrink_wmma, halo_rad_lora_shrink_wmma_scratch, &ta);
        run_shrink(U, halo_rad_lora_shrink_wmma, halo_rad_lora_shrink_wmma_scratch, &tu);
        if (std::memcmp(ta.data(), tu.data(), ta.size() * 4))
            fail("layout", fmt("shrink_wmma: aligned and unaligned x differ (K %d R %d)", sh.K, sh.R));
        ++compared;
    }
    std::printf("lora batch invariance: %d row comparisons (M=1 vs M in {7,16,17,37,300}, three "
                "positions, aligned and unaligned)\n", compared);
}

/* Base rows only: a prefill-sized launch of nothing but base rows touches no byte of t or y. */
static void test_all_base() {
    std::mt19937 rng(5);
    std::vector<int32_t> slot((size_t)2048);
    for (size_t i = 0; i < slot.size(); ++i) slot[i] = (i % 3 == 0) ? -1 : (i % 3 == 1 ? 4 : -100);
    const Shrink P = make_shrink(rng, 2048, 256, 48, 4, RAD_BF16, slot, true);
    std::vector<float> t;
    double wb = 0, wr = 0;
    if (run_shrink(P, halo_rad_lora_shrink_wmma, halo_rad_lora_shrink_wmma_scratch, &t) != RAD_OK)
        fail("all-base", "shrink rc");
    check_shrink("all-base/shrink_wmma", P, t, &wb, &wr);
    const Expand E = make_expand(rng, 2048, 640, 48, 4, RAD_BF16, slot, false);
    std::vector<uint8_t> y;
    if (run_expand(E, halo_rad_lora_expand_tiled, &y) != RAD_OK) fail("all-base", "expand rc");
    if (y != E.y0) fail("all-base", "expand_tiled touched a base-only launch");
    std::printf("lora base rows only: M 2048, t and y untouched\n");
}

/* Every code through a one-hot pool: A[s, j, k] = (k == j), so t[m, j] = x(m, j) = code * scale
 * exactly (one exact product, the rest +-0), and the WMMA row's code widening is checked code by
 * code. Codes whose value is NaN or Inf would turn every j's zero products into NaN, so they
 * get rows of their own, where every t must be NaN as in the f64 sum. */
static void test_codes() {
    int checked = 0;
    for (uint32_t adt : { (uint32_t)RAD_F8E4M3, (uint32_t)RAD_F8E5M2, (uint32_t)RAD_I8 }) {
        const int K = 256, R = 256, S = 1;
        std::vector<int> special;
        std::vector<uint8_t> row(K);
        for (int c = 0; c < 256; ++c) {
            uint8_t b = (uint8_t)c;
            const float v = lora_cases::get(adt, &b, 0);
            if (std::isfinite(v)) row[(size_t)c] = b;
            else { row[(size_t)c] = 0; special.push_back(c); }
        }
        const int M = 1 + (int)special.size();
        Shrink P;
        P.M = M; P.K = K; P.R = R; P.S = S; P.adt = adt; P.lda = K; P.off = 0;
        P.lds = 2;
        P.abuf.assign((size_t)M * K + 64, 0);
        std::memcpy(P.abuf.data(), row.data(), (size_t)K);
        for (size_t i = 0; i < special.size(); ++i)
            P.abuf[(size_t)(1 + i) * K + (size_t)(i * 37 % K)] = (uint8_t)special[i];
        P.sbuf.resize((size_t)M * 2);
        for (int m = 0; m < M; ++m) { P.sbuf[(size_t)m * 2] = 0.0371f; P.sbuf[(size_t)m * 2 + 1] = 1.75f; }
        P.slot.assign((size_t)M, 0);
        P.la.assign((size_t)R * K, 0);
        for (int j = 0; j < R; ++j) P.la[(size_t)j * K + j] = rad_f32_to_bf16(1.0f);
        std::vector<float> t;
        if (run_shrink(P, halo_rad_lora_shrink_wmma, halo_rad_lora_shrink_wmma_scratch, &t) != RAD_OK) {
            fail("codes", "rc");
            continue;
        }
        for (int j = 0; j < R; ++j) {
            const float want = x_of(P, 0, j);
            if (!(t[(size_t)j] == want))
                fail("codes", fmt("%s code 0x%02x: t %.9g, want %.9g", rad_dtype_name(adt),
                                  (unsigned)row[(size_t)j], (double)t[(size_t)j], (double)want));
            ++checked;
        }
        for (int m = 1; m < M; ++m) {
            const int pos = (int)((size_t)(m - 1) * 37 % K);
            for (int j = 0; j < R; ++j) {
                /* j == pos: the code's own value times 1 plus zeros (an Inf stays Inf); every
                 * other j: the code times 0, which is NaN for both NaN and Inf. */
                const float v = t[(size_t)m * R + j], want = x_of(P, m, pos);
                const bool okv = (j == pos && !std::isnan(want)) ? v == want : std::isnan(v);
                if (!okv)
                    fail("codes", fmt("%s code 0x%02x: t[%d] = %g", rad_dtype_name(adt),
                                      (unsigned)special[(size_t)m - 1], j, (double)v));
            }
        }
    }
    std::printf("lora codes: %d codes widened exactly (E4M3, E5M2, i8), NaN / Inf rows NaN\n",
                checked);
}

/* lora_expand_mm (the generators' expand): t rounded to bf16 and summed on the WMMA in its own
 * order, so each output is held to the double sum over bf16(t) within one rounding of y plus the
 * f32 sum's bound; base rows and everything outside the slice untouched, bit for bit. Mixed slots
 * in a tile, an odd M and N, R not a multiple of 32, every y dtype, aligned and not. */
static void test_expand_mm() {
    std::mt19937 rng(41);
    struct Sh { int M, N, R, S; uint32_t ydt; bool al; int base_pct; };
    const Sh shapes[] = {
        { 1,    20,    16,  1, RAD_BF16, true,  0  },
        { 64,   64,    32,  1, RAD_BF16, true,  0  },
        { 130,  300,   48,  2, RAD_BF16, false, 20 },
        { 77,   9216,  192, 1, RAD_BF16, true,  0  },   /* a fused qkv, one adapter */
        { 200,  1000,  64,  3, RAD_F32,  false, 30 },
        { 65,   257,   24,  2, RAD_F16,  true,  10 },
        { 512,  2048,  128, 1, RAD_BF16, true,  0  },
    };
    int64_t worst_n = 0;
    double worst = 0;
    for (const Sh& sh : shapes) {
        const std::vector<int32_t> slot = rand_slots(rng, sh.M, sh.S, sh.base_pct);
        Expand E = make_expand(rng, sh.M, sh.N, sh.R, sh.S, sh.ydt, slot, sh.al);
        std::vector<uint8_t> got;
        if (run_expand(E, halo_rad_lora_expand_mm, &got) != RAD_OK) {
            fail("expand_mm", fmt("rc (M %d N %d R %d)", sh.M, sh.N, sh.R));
            continue;
        }
        const size_t e = esz(sh.ydt), ny = (size_t)E.yoff + (size_t)sh.M * E.ldy;
        int64_t bad_base = 0, bad_val = 0;
        for (size_t i = 0; i < ny; ++i) {
            const long long q = (long long)i - E.yoff;
            const int m = q < 0 ? -1 : (int)(q / E.ldy), n = q < 0 ? -1 : (int)(q % E.ldy);
            const int sl = m >= 0 && m < sh.M ? slot[(size_t)m] : -1;
            if (m < 0 || m >= sh.M || n >= sh.N || sl < 0 || sl >= sh.S) {
                bad_base += std::memcmp(&got[i * e], &E.y0[i * e], e) != 0;
                continue;
            }
            double acc = 0, terms = 0;
            for (int j = 0; j < sh.R; ++j) {
                const double tv = rad_bf16_to_f32(rad_f32_to_bf16(E.t[(size_t)m * sh.R + j]));
                const double bv = rad_bf16_to_f32(E.lb[((size_t)sl * sh.N + n) * sh.R + j]);
                acc += tv * bv;
                terms += std::fabs(tv * bv);
            }
            const double sc = E.sc[(size_t)sl * sh.N + n];
            const double y0 = lora_cases::get(sh.ydt, E.y0.data(), i);
            const double want = y0 + sc * acc;
            const double g = lora_cases::get(sh.ydt, got.data(), i);
            if (std::isnan(y0)) { bad_val += !std::isnan(g); continue; }
            const double ulp = sh.ydt == RAD_BF16 ? std::ldexp(1.0, -8)
                             : sh.ydt == RAD_F16  ? std::ldexp(1.0, -10) : std::ldexp(1.0, -23);
            const double tol = std::fabs(want) * ulp + sc * terms * (sh.R + 2) * std::ldexp(1.0, -24) +
                               1e-30;
            const double r = std::fabs(g - want) / tol;
            worst = std::max(worst, r);
            if (r > 1.0) ++bad_val;
            ++worst_n;
        }
        if (bad_base)
            fail("expand_mm", fmt("touched %lld base-row / outside bytes (M %d N %d R %d %s)",
                                  (long long)bad_base, sh.M, sh.N, sh.R, rad_dtype_name(sh.ydt)));
        if (bad_val)
            fail("expand_mm", fmt("%lld outputs outside the bound (M %d N %d R %d %s)",
                                  (long long)bad_val, sh.M, sh.N, sh.R, rad_dtype_name(sh.ydt)));
    }
    std::printf("lora expand_mm: %zu shapes, %lld outputs, worst |err|/bound %.3f\n",
                sizeof(shapes) / sizeof(shapes[0]), (long long)worst_n, worst);
}

int main() {
    LaunchFn rs = nullptr, re = nullptr;
    load_libref(&rs, &re);
    std::printf("lora: libref %s\n", rs && re
                ? "loaded: the emulated kernels are held to it as well"
                : "not found ($RADIANCE_HOME): held to ref_host.cpp and the double sums only");
    std::vector<Impl> impls = { { "host", host_shrink, host_expand, true, nullptr } };
    if (rs && re) impls.push_back({ "libref", rs, re, true, nullptr });
    impls.push_back({ "r4d-emu", halo_rad_lora_shrink, halo_rad_lora_expand, true, nullptr });
    impls.push_back({ "r4d-tile", halo_rad_lora_shrink_wmma, halo_rad_lora_expand_tiled, true,
                      halo_rad_lora_shrink_wmma_scratch });
    lora_cases::Tally T;
    lora_cases::run_matrix(T, impls, 11);
    std::printf("lora: %d implementation-cases, %d failure(s); worst |err|/tol shrink %.3f, "
                "expand %.3f\n", T.cases, T.fails, T.worst_shrink, T.worst_expand);

    Rows R{ rs ? rs : host_shrink, re ? re : host_expand };
    test_shapes(R);
    test_invariance();
    test_all_base();
    test_codes();
    test_expand_mm();
    std::printf("lora tiled rows: %d failure(s)\n", g_fails);
    return (T.fails != 0 || g_fails != 0) ? 1 : 0;
}
