/* gemma4_exl3_embed_test.cpp -- the token-table path of gemma4_exl3 in numbers, on the host.
 *
 * A quantised table (tq at 4 and 5 bits, an exl3i trellis at rate 4 in both row orders) is gathered by its op's
 * host row (plugin/exl3i, linked here) and scaled as the plugin declares it: a fill of one row with
 * bf16(sqrt(n_embd)) and a mul (gemma4_exl3_test pins the declared value, 62.0 at 3840 and 73.5 at 5376). That
 * must give, in every bit, what the bf16 path gives: embed_lookup with the same `mult` over the bf16 table that the
 * same gather decodes. One token, then a short batch with a repeated id and a padding id (-1, a zero row in both).
 * The same gathers at E4B's per-layer width (42 x 256 = 10752, scale 16) stand for the per-layer table.
 * A rank past 0 gathers its rows too and then writes zeros over them: the last case shows why zeros and not the rows
 * times 0, with a table that holds inf and NaN.
 * fill, mul and embed_lookup are libref's (the host reference rad-kbench holds every kernel to), dlopen'd from
 * $RADIANCE_HOME/kernels, which CMake sets; without libref the test exits 77, which CTest reports as skipped.
 */
#undef NDEBUG   /* the checks below are assert()s: keep them in a Release build */

#include "embed_exl3i.h"
#include "embed_tq.h"
#include "exl3i_encoding.h"
#include "rad_plugin.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <random>
#include <string>
#include <vector>

namespace {

/* ======== libref */

struct Ref {
    void*                handle = nullptr;
    int                  (*count)(void) = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;

    const RadKernelInfo* host(const char* op) const {
        for (int i = 0; i < count(); ++i) {
            const RadKernelInfo* k = at(i);
            if (k && k->domain == RAD_DOMAIN_HOST && !std::strcmp(k->op, op) && k->launch) return k;
        }
        return nullptr;
    }
};

bool load_ref(Ref* r) {
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home) return false;
    const std::string p = std::string(home) + "/kernels/libref.so";
    r->handle = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!r->handle) return false;
    r->count = (int (*)(void))dlsym(r->handle, "rad_kernel_count");
    r->at = (const RadKernelInfo* (*)(int))dlsym(r->handle, "rad_kernel_at");
    return r->count && r->at;
}

/* ======== tensors and calls */

uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t to_f16(float f) {   /* normal values only: the scales below stay in [0.25, 2] */
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    const int32_t  e = (int32_t)((u >> 23) & 0xff) - 127 + 15;
    const uint32_t m = (u >> 13) & 0x3ffu;
    return (uint16_t)(s | ((uint32_t)e << 10) | m);
}

/* A host tensor of `bytes` bytes, viewed as dtype [shape...], contiguous. */
struct T {
    std::vector<uint8_t> mem;
    RadTensor            t{};
    T(uint32_t dt, std::vector<int64_t> shape, size_t bytes) : mem(bytes + 8, 0) {
        t.data = mem.data();
        t.dtype = dt;
        t.rank = (uint32_t)shape.size();
        int64_t acc = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            t.shape[i] = shape[i];
            t.stride[i] = acc;
            acc *= shape[i];
        }
    }
    /* a copy or a move points its view at its own bytes */
    T(const T& o) : mem(o.mem), t(o.t) { t.data = mem.data(); }
    T(T&& o) noexcept : mem(std::move(o.mem)), t(o.t) { t.data = mem.data(); }
    uint16_t*       u16()       { return (uint16_t*)mem.data(); }
    const uint16_t* u16() const { return (const uint16_t*)mem.data(); }
};
T bf16_t(int64_t rows, int64_t cols) { return T(RAD_BF16, { rows, cols }, (size_t)(rows * cols * 2)); }
T ids_t(const std::vector<int32_t>& ids) {
    T x(RAD_I32, { (int64_t)ids.size() }, ids.size() * 4);
    std::memcpy(x.mem.data(), ids.data(), ids.size() * 4);
    return x;
}

/* The arguments of one launch: operands in schema order, the resolved params. */
struct A {
    std::vector<RadTensor>  ts;
    std::vector<RadParam>   ps;
    std::deque<std::string> strs;
    RadArgs                 a{};
    A& t(const T& x) { ts.push_back(x.t); return *this; }
    A& i(const char* k, long long v) { ps.push_back(RAD_INT(k, v)); return *this; }
    A& d(const char* k, double v) { ps.push_back(RAD_F64(k, v)); return *this; }
    A& s(const char* k, const char* v) {
        strs.emplace_back(v);
        ps.push_back(RAD_STR(k, strs.back().c_str()));
        return *this;
    }
    const RadArgs* get() {
        a.t = ts.data(); a.n_t = (int)ts.size(); a.p = ps.data(); a.n_p = (int)ps.size();
        a.rank = 0; a.world_size = 1;
        return &a;
    }
};

/* ======== the two paths */

/* The plugin's quantised path after the gather: fill one row with the scale, then x = x * row. */
void scale_like_the_plugin(const Ref& ref, T& x, int64_t M, int64_t n_embd, double scale) {
    T row = bf16_t(1, n_embd);
    A f; f.t(row).i("M", 1).i("n", n_embd).s("dtype", "bf16").d("value", scale);
    assert(ref.host("fill")->launch(f.get(), nullptr) == RAD_OK);
    A m; m.t(x).t(row).t(x).i("M", M).i("n", n_embd).s("dtype", "bf16");
    assert(ref.host("mul")->launch(m.get(), nullptr) == RAD_OK);
}

/* The bf16 path: embed_lookup with `mult` over `table` [n_vocab, n_embd]. */
T bf16_path(const Ref& ref, const T& table, const std::vector<int32_t>& ids, int64_t n_vocab, int64_t n_embd,
            double scale) {
    T tok = ids_t(ids), x = bf16_t((int64_t)ids.size(), n_embd);
    A e; e.t(tok).t(table).t(x).i("M", (long long)ids.size()).i("n_embd", n_embd).i("n_vocab", n_vocab)
          .s("dtype", "bf16").i("vocab_offset", 0).d("mult", scale);
    assert(ref.host("embed_lookup")->launch(e.get(), nullptr) == RAD_OK);
    return x;
}

/* Both paths over the same ids: equal in every bit, and the scale really applied (a nonzero row differs from the
 * unscaled gather). `gather(ids)` runs the quantised op's host row into a fresh [M, n_embd] tensor. */
template <class G>
void check_paths(const Ref& ref, const char* what, G gather, int64_t n_vocab, int64_t n_embd, double scale) {
    std::vector<int32_t> all;
    for (int32_t v = 0; v < (int32_t)n_vocab; ++v) all.push_back(v);
    const T table = gather(all);   /* the bf16 table the gather decodes, row by row */
    const std::vector<std::vector<int32_t>> cases = { { 3 }, { 2, 0, 2, -1, 5 } };
    for (const std::vector<int32_t>& ids : cases) {
        const int64_t M = (int64_t)ids.size();
        T q = gather(ids);
        const T unscaled = q;
        scale_like_the_plugin(ref, q, M, n_embd, scale);
        const T b = bf16_path(ref, table, ids, n_vocab, n_embd, scale);
        int64_t diff = 0, moved = 0, nonzero = 0;
        for (int64_t i = 0; i < M * n_embd; ++i) {
            const uint16_t qv = q.u16()[i];
            diff += qv != b.u16()[i];
            moved += qv != unscaled.u16()[i];
            nonzero += (qv & 0x7fff) != 0;
        }
        for (int64_t m = 0; m < M; ++m)   /* the padding id: a zero row in both */
            if (ids[(size_t)m] < 0)
                for (int64_t k = 0; k < n_embd; ++k) assert((q.u16()[m * n_embd + k] & 0x7fff) == 0);
        printf("  %-26s M %lld: %lld of %lld values differ from the bf16 path, %lld scaled, %lld nonzero\n",
               what, (long long)M, (long long)diff, (long long)(M * n_embd), (long long)moved, (long long)nonzero);
        assert(diff == 0);
        assert(nonzero > n_embd / 2 && moved > n_embd / 2);
    }
}

/* tq: random B-bit codes, a 2^B table in [-1, 1], a bf16 scale per (row, 128) around 1/128 (the butterfly is
 * unnormalised, embed_tq.h). */
void check_tq(const Ref& ref, int bits, int64_t n_embd, double scale) {
    const int64_t nv = 6, row_bytes = (n_embd * bits + 7) / 8;
    std::mt19937 rng(1234u + (unsigned)bits);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f), us(0.004f, 0.012f);
    const uint32_t cdt[9] = { 0, RAD_U1, RAD_U2, RAD_U3, RAD_U4, RAD_U5, RAD_U6, RAD_U7, RAD_U8 };
    T codes(cdt[bits], { nv, n_embd }, (size_t)(nv * row_bytes));
    for (int64_t i = 0; i < nv * row_bytes; ++i) codes.mem[(size_t)i] = (uint8_t)(rng() & 0xff);
    T sc = bf16_t(nv, n_embd / 128);
    for (int64_t i = 0; i < nv * (n_embd / 128); ++i) sc.u16()[i] = to_bf16(us(rng));
    T tab(RAD_F32, { 1, (int64_t)1 << bits }, (size_t)(4 << bits));
    for (int i = 0; i < (1 << bits); ++i) ((float*)tab.mem.data())[i] = u(rng);
    auto gather = [&](const std::vector<int32_t>& ids) {
        T tok = ids_t(ids), x = bf16_t((int64_t)ids.size(), n_embd);
        A a; a.t(tok).t(codes).t(sc).t(tab).t(x).i("M", (long long)ids.size()).i("n_embd", n_embd)
              .i("n_vocab", nv).i("bits", bits);
        assert(embed_tq_host_launch(a.get(), nullptr) == RAD_OK);
        return x;
    };
    char what[48];
    std::snprintf(what, sizeof what, "tq %d bits, n_embd %lld", bits, (long long)n_embd);
    check_paths(ref, what, gather, nv, n_embd, scale);
}

/* exl3i: a random record (any bits decode), suh and svh in [0.25, 2] with random signs; mad at rate 4 (m 64). */
void check_exl3i(const Ref& ref, int r16, int64_t n_embd, double scale) {
    const int64_t nv = 128, m = 64, rw = exl3i_row_words(n_embd, (int)m);
    std::mt19937 rng(77u + (unsigned)r16);
    std::uniform_real_distribution<float> us(0.25f, 2.0f);
    T rec(RAD_U32, { nv / 16, rw }, (size_t)(nv / 16 * rw * 4));
    for (int64_t i = 0; i < nv / 16 * rw; ++i) ((uint32_t*)rec.mem.data())[i] = (uint32_t)rng();
    T suh(RAD_F16, { 1, n_embd }, (size_t)(n_embd * 2)), svh(RAD_F16, { nv, 1 }, (size_t)(nv * 2));
    for (int64_t i = 0; i < n_embd; ++i) suh.u16()[i] = to_f16((rng() & 1 ? -1.0f : 1.0f) * us(rng));
    for (int64_t i = 0; i < nv; ++i) svh.u16()[i] = to_f16((rng() & 1 ? -1.0f : 1.0f) * us(rng) * 0.01f * 64);
    auto gather = [&](const std::vector<int32_t>& ids) {
        T tok = ids_t(ids), x = bf16_t((int64_t)ids.size(), n_embd);
        A a; a.t(tok).t(rec).t(suh).t(svh).t(x).i("M", (long long)ids.size()).i("n_embd", n_embd)
              .i("n_vocab", nv).i("m", m).s("codebook", "mad").i("r16", r16);
        assert(embed_exl3i_host_launch(a.get(), nullptr) == RAD_OK);
        return x;
    };
    char what[48];
    std::snprintf(what, sizeof what, "exl3i K4%s, n_embd %lld", r16 ? " r16" : "", (long long)n_embd);
    check_paths(ref, what, gather, nv, n_embd, scale);
}

/* A rank past 0: the gather's rows, then `fill` 0 over them (gemma4_exl3's op_embed_zero). A tq table may hold any
 * f32, inf and NaN too; the zeros must not depend on it, since the all-reduce adds this rank's rows to rank 0's. The
 * rows times a zero row would not do: inf * 0 and NaN * 0 are NaN. */
void check_rank_past_0_writes_zeros(const Ref& ref) {
    const int64_t nv = 4, n_embd = 3840, bits = 4, row_bytes = n_embd * bits / 8;
    T codes(RAD_U4, { nv, n_embd }, (size_t)(nv * row_bytes));
    for (int64_t i = 0; i < nv * row_bytes; ++i) codes.mem[(size_t)i] = (uint8_t)(i * 37 + 11);   /* every code */
    T sc = bf16_t(nv, n_embd / 128);
    for (int64_t i = 0; i < nv * (n_embd / 128); ++i) sc.u16()[i] = to_bf16(0.008f);
    T tab(RAD_F32, { 1, 16 }, 64);
    for (int i = 0; i < 16; ++i) ((float*)tab.mem.data())[i] = 0.1f * (float)(i - 8);
    ((float*)tab.mem.data())[3] = INFINITY;
    ((float*)tab.mem.data())[7] = NAN;
    const std::vector<int32_t> ids = { 1, 0, 3, -1, 2 };
    const int64_t M = (int64_t)ids.size();
    T tok = ids_t(ids), x = bf16_t(M, n_embd);
    A a; a.t(tok).t(codes).t(sc).t(tab).t(x).i("M", M).i("n_embd", n_embd).i("n_vocab", nv).i("bits", bits);
    assert(embed_tq_host_launch(a.get(), nullptr) == RAD_OK);
    int64_t nonfinite = 0;
    for (int64_t i = 0; i < M * n_embd; ++i) nonfinite += (x.u16()[i] & 0x7f80) == 0x7f80;
    T times0 = x;
    scale_like_the_plugin(ref, times0, M, n_embd, 0.0);
    int64_t nan0 = 0;
    for (int64_t i = 0; i < M * n_embd; ++i) nan0 += (times0.u16()[i] & 0x7f80) == 0x7f80;
    A f; f.t(x).i("M", M).i("n", n_embd).s("dtype", "bf16").d("value", 0.0);
    assert(ref.host("fill")->launch(f.get(), nullptr) == RAD_OK);
    int64_t notzero = 0;
    for (int64_t i = 0; i < M * n_embd; ++i) notzero += x.u16()[i] != 0;
    printf("  rank past 0, a table with inf and NaN: %lld of %lld gathered values not finite, %lld after * 0, "
           "%lld not +0 after the zero fill\n", (long long)nonfinite, (long long)(M * n_embd), (long long)nan0,
           (long long)notzero);
    assert(nonfinite > 0 && nan0 > 0 && notzero == 0);
}

}  /* namespace */

int main() {
    Ref ref;
    if (!load_ref(&ref) || !ref.host("fill") || !ref.host("mul") || !ref.host("embed_lookup")) {
        printf("gemma4_exl3_embed_test: libref (fill, mul, embed_lookup) not found under $RADIANCE_HOME/kernels: "
               "skipped\n");
        return 77;
    }
    assert(from_bf16(to_bf16(std::sqrt(3840.0f))) == 62.0f && from_bf16(to_bf16(std::sqrt(5376.0f))) == 73.5f);
    printf("[TEST] a quantised token table gathered and scaled as gemma4_exl3 does = the bf16 path, in every bit...\n");
    for (int bits : { 4, 5 }) check_tq(ref, bits, 3840, 62.0);
    check_tq(ref, 4, 5376, 73.5);
    for (int bits : { 4, 5, 8 }) check_tq(ref, bits, 42 * 256, 16.0);   /* E4B's per-layer table, sqrt(256) */
    for (int r16 : { 0, 1 }) check_exl3i(ref, r16, 3840, 62.0);
    printf("[TEST] a rank past 0 writes +0 over its gathered rows, whatever the table holds...\n");
    check_rank_past_0_writes_zeros(ref);
    printf("ALL TESTS PASSED.\n");
    return 0;
}
