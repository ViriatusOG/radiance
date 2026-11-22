/* tests/r4d_emu: moe_gemm_q at nvfp4a8i8 (r4d_moe_nvfp4.hip) on the host SIMT emulator, against an
 * f64 oracle of the value e2m1 * e4m3 * level times an int8 activation under its f32 scale.
 *
 * The scale operand is the blob the layout hook produces (r4d_layout.cpp, nvfp4): per expert N rows
 * of K / 16 E4M3 bytes, then one row whose first four bytes are the expert's f32 level. The codes
 * are the e2m1 plane as stored. Stacked and pointer-table operands both run, with a token-order and
 * a sorted-order gather, a dropped slot, and rows past the last expert's offset. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kE2m1Mag2[8] = { 0, 1, 2, 3, 4, 6, 8, 12 };   /* e2m1 magnitudes, doubled */

float e4m3_value(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? (float)m * std::ldexp(1.0f, -9)
                           : (1.0f + (float)m / 8.0f) * std::ldexp(1.0f, e - 7);
    return s ? -v : v;
}

RadTensor tensor(void* data, uint32_t dtype, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = data;
    t.dtype = dtype;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[(size_t)i];
        t.stride[i] = st;
        st *= shape[(size_t)i];
    }
    return t;
}

RadParam pint(const char* k, long long v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}
RadParam pstr(const char* k, const char* v) {
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_STR;
    p.sval = v;
    return p;
}

struct Case {
    int N, K, E, top_k, rows;   /* output width, K, experts, slots per token, sorted rows */
    bool pointer_scales;        /* the scale operands as a pointer table, not stacked */
    bool token_order;           /* a_order "token" (else "sorted") */
    bool declared_extent;       /* the pointer table at the plugin's declared [N, K], as the core
                                 * issues a view whose stored blob is [N + 1, K / 16] */
};

int run_case(const Case& c, unsigned seed) {
    const int N = c.N, K = c.K, E = c.E, T = c.rows, top_k = c.top_k;
    const int ne0 = (E + 1) / 2, ne1 = E / 2;
    /* the activation: one row per token (token order) or per sorted row (sorted order) */
    const int M = c.token_order ? (T + top_k - 1) / top_k : T;
    const int AG = 64;   /* the activation's scale covers 64 of K */
    std::mt19937 rng(seed);
    auto rnd = [&rng](unsigned n) { return (int)(rng() % n); };

    std::vector<uint8_t> codes((size_t)E * N * K);            /* [expert][n][k], 0..15 */
    std::vector<uint8_t> scales((size_t)E * N * (K / 16));    /* E4M3 per (n, 16 of K) */
    std::vector<float>   level((size_t)E);
    for (auto& x : codes) x = (uint8_t)rnd(16);
    for (auto& x : scales) x = (uint8_t)(0x30 + rnd(0x20));   /* E4M3 near 2^-3 .. 2^1 */
    for (auto& x : level) x = 0.5f + 0.25f * (float)rnd(8);

    std::vector<int8_t> act((size_t)M * K);
    for (auto& x : act) x = (int8_t)(rnd(255) - 127);
    std::vector<float> sa((size_t)M * (K / AG));
    for (auto& x : sa) x = 0.03125f * (float)(1 + rnd(16));

    /* the routing: sorted rows split between the experts in order; the last row is dropped */
    std::vector<int32_t> sorted((size_t)T), off((size_t)E + 1, 0);
    for (int i = 0; i < T; ++i)
        sorted[(size_t)i] = i == T - 1 ? -1 : (c.token_order ? rnd((unsigned)(M * top_k)) : i);
    for (int e = 1; e <= E; ++e) off[(size_t)e] = (int32_t)((long long)T * e / E);
    off[(size_t)E] = T - 1;
    for (int e = 1; e <= E; ++e)
        if (off[(size_t)e] < off[(size_t)e - 1]) off[(size_t)e] = off[(size_t)e - 1];

    /* the oracle: a row's expert is the last one whose offset is at most its row */
    std::vector<double> want((size_t)T * N, 0.0);
    for (int i = 0; i < T; ++i) {
        const int flat = sorted[(size_t)i];
        if (flat < 0) continue;
        int e = 0;
        for (int j = 0; j < E; ++j)
            if (off[(size_t)j] <= i) e = j;
        const int tok = c.token_order ? flat / top_k : i;
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                const uint8_t cd = codes[(size_t)((e * N + n) * K + k)];
                const double lv = (cd & 8 ? -kE2m1Mag2[cd & 7] : kE2m1Mag2[cd & 7]) * 0.5;
                const double sc = e4m3_value(scales[(size_t)((e * N + n) * (K / 16) + k / 16)]);
                acc += lv * sc * level[(size_t)e] * act[(size_t)(tok * K + k)] *
                       sa[(size_t)(tok * (K / AG) + k / AG)];
            }
            want[(size_t)(i * N + n)] = acc;
        }
    }

    /* the stored operands: each class's codes as a stacked e2m1 plane, and each expert's scale blob
     * (N rows of K / 16 bytes, then the level in the next row's first four bytes) */
    const int64_t blob = (int64_t)(N + 1) * (K / 16);
    std::vector<uint8_t> cw[2], sw[2];
    cw[0].assign((size_t)(ne0 * N * K / 2) + 1, 0);
    cw[1].assign((size_t)(ne1 * N * K / 2) + 1, 0);
    sw[0].assign((size_t)(ne0 * blob) + 4, 0);
    sw[1].assign((size_t)(ne1 * blob) + 4, 0);
    for (int e = 0; e < E; ++e) {
        const int q = e % 2, ix = e / 2;
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; k += 2) {
                const uint8_t lo = codes[(size_t)((e * N + n) * K + k)];
                const uint8_t hi = codes[(size_t)((e * N + n) * K + k + 1)];
                cw[q][(size_t)(ix * N * K / 2 + n * K / 2 + k / 2)] = (uint8_t)(lo | (hi << 4));
            }
        uint8_t* b = sw[q].data() + (size_t)(ix * blob);
        std::memcpy(b, scales.data() + (size_t)(e * N * (K / 16)), (size_t)(N * (K / 16)));
        std::memcpy(b + N * (K / 16), &level[(size_t)e], sizeof(float));
    }

    /* the scale operands: stacked, or a pointer table of one blob per expert (stride 0) */
    std::vector<void*> ptr[2];
    RadTensor ws[2];
    for (int q = 0; q < 2; ++q) {
        const int64_t ne = q == 0 ? ne0 : (ne1 > 0 ? ne1 : 1);
        if (c.pointer_scales) {
            for (int e = q; e < E; e += 2)
                ptr[q].push_back(sw[q].data() + (size_t)((e / 2) * blob));
            if (ptr[q].empty()) ptr[q].push_back(sw[q].data());
            ws[q] = c.declared_extent ? tensor(ptr[q].data(), RAD_U8, { ne, N, K })
                                      : tensor(ptr[q].data(), RAD_U8, { ne, N + 1, K / 16 });
            ws[q].stride[0] = 0;
        } else {
            ws[q] = tensor(sw[q].data(), RAD_U8, { ne, N + 1, K / 16 });
        }
    }
    RadTensor w0 = tensor(cw[0].data(), RAD_FP4E2M1, { ne0, N, K });
    RadTensor w1 = tensor(cw[1].data(), RAD_FP4E2M1, { ne1 > 0 ? ne1 : 1, N, K });
    RadTensor a = tensor(act.data(), RAD_I8, { M, K });
    RadTensor as = tensor(sa.data(), RAD_F32, { M, K / AG });
    RadTensor srt = tensor(sorted.data(), RAD_I32, { T });
    RadTensor ofs = tensor(off.data(), RAD_I32, { E + 1 });
    std::vector<uint16_t> y((size_t)T * N, 0);
    RadTensor yt = tensor(y.data(), RAD_BF16, { T, N });

    RadTensor ts[9] = { a, as, w0, ws[0], srt, ofs, w1, ws[1], yt };
    RadParam ps[] = { pint("N", N), pint("K", K), pint("N_odd", N), pint("K_odd", K),
                      pint("parts", 1), pint("n_expert", E), pint("top_k", top_k),
                      pint("group", AG), pstr("a_order", c.token_order ? "token" : "sorted"),
                      pstr("dtype", "nvfp4a8i8") };
    RadArgs args{};
    args.t = ts;
    args.n_t = 9;
    args.p = ps;
    args.n_p = (int)(sizeof ps / sizeof ps[0]);
    args.world_size = 1;
    const int rc = r4d_rad_moe_gemm_nvfp4a8i8(&args, nullptr);
    if (rc != RAD_OK) {
        std::fprintf(stderr, "    launch: rc %d\n", rc);
        return 1;
    }
    int bad = 0;
    for (int i = 0; i < T * N; ++i) {
        const uint32_t u = (uint32_t)y[(size_t)i] << 16;
        float got;
        std::memcpy(&got, &u, sizeof got);
        const double w = want[(size_t)i];
        if (std::fabs(got - w) > 0.01 * std::fabs(w) + 1e-3) {
            if (bad < 4) std::fprintf(stderr, "    row %d col %d: %g, want %g\n", i / N, i % N, got, w);
            ++bad;
        }
    }
    return bad ? 1 : 0;
}

}  // namespace

int main() {
    const Case cases[] = {
        { 16, 64, 2, 1, 5, false, true },    /* two experts, stacked scales, token order */
        { 16, 64, 2, 1, 5, true, true },     /* the same with a pointer table of scales */
        { 16, 128, 3, 1, 6, false, false },  /* sorted order, three experts */
        { 32, 64, 4, 2, 9, false, true },    /* top 2, four experts */
        { 16, 64, 2, 1, 5, true, true, true },   /* the pointer table as the core issues it */
    };
    int fails = 0, n = 0;
    for (const Case& c : cases) {
        ++n;
        const int rc = run_case(c, 77u + (unsigned)n);
        if (rc) ++fails;
        std::printf("  case %d (N %d K %d experts %d top %d rows %d%s%s%s): %s\n", n, c.N, c.K, c.E,
                    c.top_k, c.rows, c.pointer_scales ? " pointer" : "",
                    c.declared_extent ? " at the declared extent" : "",
                    c.token_order ? "" : " sorted", rc ? "FAIL" : "ok");
    }
    std::printf("%d case(s), %d failure(s)\n", n, fails);
    return fails ? 1 : 0;
}
