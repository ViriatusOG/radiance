// tests/halo_emu/test_gemm.cpp -- libhalo's prefill GEMM tiles and the MXFP4 decode plane, run on
// the host by tests/r4d_emu's SIMT emulator (its gfx11 WMMA forms), with no card.
//
// Each tile promises the bytes of a row the card already verified, so each case is a comparison:
//   gemm_fp8a8_tiled / gemm_i8a8_tiled  == gemm_fp8a8_nt_m16 / gemm_i8a8_nt_m16 over 64-row slices,
//                                          with the activation plane and without it (decoded while
//                                          staging), on both tile arms
//   gemm_mxfp4a8_tiled                  == gemm_mxfp4a8_nt_m64 at sk=1 over 64-row slices, both
//                                          activation scale grids, ragged N
//   gemm_mxfp4a8_decode                 with its activation plane == without it, at every split
// plus the narrow fp8 row against a float64 reference, which holds the emulator's gfx11 lane map
// to the arithmetic (the narrow kernel is card-verified; a wrong lane map would fail there).
// halo_selftest runs the same comparisons on the card at larger shapes.
#include "hip/hip_runtime.h"   // tests/r4d_emu/emu: the emulator
#include "rad_abi.h"
#include "rad_plugin.h"   // rad_f32_to_fp8e4m3: the saturating encoder every fp8 store is held to
#include "halo_fp8_frag.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" int halo_rad_gemm_fp8a8_tiled(const RadArgs*, RadStream);
extern "C" int64_t halo_rad_gemm_fp8a8_tiled_scratch(const RadArgs*);
extern "C" int halo_rad_gemm_fp8a8_m16(const RadArgs*, RadStream);
extern "C" int64_t halo_rad_gemm_fp8a8_m16_scratch(const RadArgs*);
extern "C" int halo_rad_gemm_i8a8_tiled(const RadArgs*, RadStream);
extern "C" int halo_rad_gemm_fp8a8_gated_m16(const RadArgs*, RadStream);
extern "C" int halo_rad_gemm_bf16_tiled(const RadArgs*, RadStream);
extern "C" int halo_rad_gemm_bf16_m64(const RadArgs*, RadStream);
extern "C" int halo_rad_gemm_i8a8_m16(const RadArgs*, RadStream);
extern "C" int halo_rad_gemm_mxfp4a8_tiled(const RadArgs*, RadStream);
extern "C" int64_t halo_rad_gemm_mxfp4a8_tiled_plane_scratch(const RadArgs*);
extern "C" int halo_rad_gemm_mxfp4a8_m64(const RadArgs*, RadStream);
extern "C" int64_t halo_rad_gemm_mxfp4a8_m64_scratch(const RadArgs*);
extern "C" int halo_rad_gemm_mxfp4a8_decode(const RadArgs*, RadStream);
extern "C" int64_t halo_rad_gemm_mxfp4a8_decode_scratch(const RadArgs*);

static int g_fail = 0;
static void ok(const std::string& what, bool c, const std::string& d = "") {
    std::printf("  %s  %s%s%s\n", c ? "ok  " : "FAIL", what.c_str(), d.empty() ? "" : "  -- ", d.c_str());
    if (!c) ++g_fail;
}
static uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x;
}
static RadTensor T(void* d, uint32_t dt, std::vector<int64_t> sh) {
    RadTensor t; std::memset(&t, 0, sizeof t);
    t.data = d; t.dtype = dt; t.rank = (uint32_t)sh.size();
    int64_t acc = 1;
    for (size_t i = 0; i < sh.size(); ++i) t.shape[i] = sh[i];
    for (int i = (int)sh.size() - 1; i >= 0; --i) { t.stride[i] = acc; acc *= sh[i]; }
    return t;
}
static RadParam PI(const char* k, long long v) { RadParam p; std::memset(&p, 0, sizeof p); p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }
static RadParam PS(const char* k, const char* v) { RadParam p; std::memset(&p, 0, sizeof p); p.key = k; p.kind = RAD_P_STR; p.sval = v; return p; }
static float bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t to_bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static double e4m3(uint8_t b) {
    const double s = (b & 0x80) ? -1 : 1; const int ex = (b >> 3) & 15, mn = b & 7;
    return ex == 0 ? s * mn / 512.0 : s * (1 + mn / 8.0) * std::ldexp(1.0, ex - 7);
}

static int run(int (*launch)(const RadArgs*, RadStream), int64_t (*hook)(const RadArgs*),
               std::vector<RadTensor> ts, std::vector<RadParam> ps) {
    RadArgs a; std::memset(&a, 0, sizeof a);
    a.t = ts.data(); a.n_t = (int)ts.size(); a.p = ps.data(); a.n_p = (int)ps.size(); a.world_size = 1;
    std::vector<char> scratch;
    if (hook) {
        const int64_t need = hook(&a);
        if (need > 0) { scratch.assign((size_t)need, (char)0x5a); a.scratch = scratch.data(); a.scratch_bytes = need; }
    }
    return launch(&a, nullptr);
}

/* fp8 / i8: tiled (plane), tiled (none), the narrow row over 64-row slices. */
static void fp8_case(int64_t M, int64_t N, int64_t K, bool i8) {
    const std::string tag = std::string(i8 ? "i8a8" : "fp8a8") + " tiled M=" + std::to_string(M) +
                            " N=" + std::to_string(N) + " K=" + std::to_string(K);
    const int64_t sk = K / 128, sn = i8 ? N : N / 128;
    std::vector<uint8_t> a((size_t)(M * K)), w((size_t)(N * K));
    std::vector<float> as((size_t)(M * sk));
    std::vector<uint16_t> ws((size_t)(sn * sk));
    auto code = [i8](uint32_t i) { uint8_t c = (uint8_t)mix(i); return !i8 && (c & 0x7f) == 0x7f ? (uint8_t)(c ^ 1) : c; };
    for (size_t i = 0; i < a.size(); ++i) a[i] = code((uint32_t)i * 2u + 1u);
    for (size_t i = 0; i < w.size(); ++i) w[i] = code((uint32_t)i * 2u + 0x5bd1e995u);
    for (size_t i = 0; i < as.size(); ++i) as[i] = std::ldexp(1.0f + (mix((uint32_t)i + 77) & 255) / 256.0f, -(int)(mix((uint32_t)i + 99) % 9) - 4);
    for (size_t i = 0; i < ws.size(); ++i) ws[i] = to_bf(std::ldexp(1.0f + (mix((uint32_t)i + 5) & 127) / 128.0f, -(int)(mix((uint32_t)i + 7) % 7) - 6));
    std::vector<uint16_t> y[3];
    for (auto& v : y) v.assign((size_t)(M * N), 0xffff);
    const uint32_t cdt = i8 ? (uint32_t)RAD_I8 : (uint32_t)RAD_F8E4M3;
    auto go = [&](int (*l)(const RadArgs*, RadStream), int64_t (*h)(const RadArgs*), int64_t m0, int64_t mm, std::vector<uint16_t>& out) {
        std::vector<RadTensor> ts = {
            T(a.data() + m0 * K, cdt, { mm, K }), T(as.data() + m0 * sk, RAD_F32, { mm, sk }),
            T(w.data(), cdt, { N, K }), T(ws.data(), RAD_BF16, { sn, sk }),
            T(out.data() + m0 * N, RAD_BF16, { mm, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }) };
        ts[5].data = nullptr; ts[5].rank = 0; ts[6].data = nullptr; ts[6].rank = 0;
        return run(l, h, ts, { PI("M", mm), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", i8 ? "i8a8" : "fp8a8") });
    };
    int rc = i8 ? go(halo_rad_gemm_i8a8_tiled, nullptr, 0, M, y[0]) : go(halo_rad_gemm_fp8a8_tiled, halo_rad_gemm_fp8a8_tiled_scratch, 0, M, y[0]);
    rc |= i8 ? go(halo_rad_gemm_i8a8_tiled, nullptr, 0, M, y[1]) : go(halo_rad_gemm_fp8a8_tiled, nullptr, 0, M, y[1]);
    for (int64_t m0 = 0; m0 < M; m0 += 64)
        rc |= i8 ? go(halo_rad_gemm_i8a8_m16, nullptr, m0, std::min<int64_t>(64, M - m0), y[2])
                 : go(halo_rad_gemm_fp8a8_m16, halo_rad_gemm_fp8a8_m16_scratch, m0, std::min<int64_t>(64, M - m0), y[2]);
    if (rc) { ok(tag, false, "rc " + std::to_string(rc)); return; }
    size_t d0 = 0, d1 = 0;
    for (size_t e = 0; e < y[2].size(); ++e) { d0 += y[0][e] != y[2][e]; d1 += y[1][e] != y[2][e]; }
    if (i8) {
        /* gemm_i8a8_tiled has no scratch hook: both runs take the same path, so one line. */
        ok(tag + " == narrow slices", d0 == 0 && d1 == 0, std::to_string(d0) + " differ");
        return;
    }
    ok(tag + " (plane) == narrow slices", d0 == 0, std::to_string(d0) + " differ");
    ok(tag + " (no plane) == narrow slices", d1 == 0, std::to_string(d1) + " differ");
    /* The narrow row against float64: checks the emulator's WMMA layout (the narrow kernel is
     * hardware-verified, so an emulator with a wrong lane map would fail here). */
    std::vector<double> wl((size_t)(N * K));
    for (int64_t rt = 0; rt < N / 16; ++rt)
        for (int64_t ks = 0; ks < K / 64; ++ks)
            for (int ln = 0; ln < 32; ++ln)
                for (int t = 0; t < 4; ++t)
                    for (int j = 0; j < 8; ++j) {
                        int64_t row, k;
                        halo_fp8::frag_src(rt, ks, ln, t, j, &row, &k);
                        wl[(size_t)(row * K + k)] = e4m3(w[(size_t)(((rt * (K / 64) + ks) * 32 + ln) * 32 + t * 8 + j)]);
                    }
    double num = 0, den = 0;
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0;
            for (int64_t g = 0; g < sk; ++g) {
                double s = 0;
                for (int64_t k = g * 128; k < g * 128 + 128; ++k) s += e4m3(a[(size_t)(m * K + k)]) * wl[(size_t)(n * K + k)];
                acc += s * as[(size_t)(m * sk + g)] * bf(ws[(size_t)((n / 128) * sk + g)]);
            }
            const double d = bf(y[2][(size_t)(m * N + n)]) - acc;
            num += d * d; den += acc * acc;
        }
    const double rel = std::sqrt(num / den);
    ok(tag + " narrow vs float64 reference", rel < 1e-2, "rel_l2 " + std::to_string(rel));
}

/* The gated narrow row's contract (its row table): the bytes of gemm_nt_q followed by
 * gated_quant_fp8. The narrow row computes the [M, N] gate|up product; the quantiser is restated
 * here in its own arithmetic -- gate and up rounded to bf16, silu(g) * u rounded to bf16, the group
 * amax over 128 columns, scale amax / 448 (1 at zero), codes of v * (1 / scale) clamped to +-448 by
 * the saturating encoder -- and the gated row's codes, scales and bf16 product must match. */
static void gated_ref_case(int64_t M, int64_t N, int64_t K) {
    const std::string tag = "fp8a8 gated == gemm_nt_q + gated_quant_fp8, M=" + std::to_string(M) +
                            " N=" + std::to_string(N) + " K=" + std::to_string(K);
    const int64_t sk = K / 128, sn = N / 128, nout = N / 2;
    std::vector<uint8_t> a((size_t)(M * K)), w((size_t)(N * K));
    std::vector<float> as((size_t)(M * sk));
    std::vector<uint16_t> ws((size_t)(sn * sk));
    auto code = [](uint32_t i) { uint8_t c = (uint8_t)mix(i); return (c & 0x7f) == 0x7f ? (uint8_t)(c ^ 1) : c; };
    for (size_t i = 0; i < a.size(); ++i) a[i] = code((uint32_t)i * 2u + 7u);
    for (size_t i = 0; i < w.size(); ++i) w[i] = code((uint32_t)i * 2u + 0x1234567u);
    for (size_t i = 0; i < as.size(); ++i) as[i] = std::ldexp(1.0f + (mix((uint32_t)i + 3) & 255) / 256.0f, -(int)(mix((uint32_t)i + 9) % 5) - 6);
    for (size_t i = 0; i < ws.size(); ++i) ws[i] = to_bf(std::ldexp(1.0f + (mix((uint32_t)i + 15) & 127) / 128.0f, -(int)(mix((uint32_t)i + 21) % 5) - 8));
    std::vector<uint16_t> y((size_t)(M * N), 0xffff);
    {
        std::vector<RadTensor> ts = {
            T(a.data(), RAD_F8E4M3, { M, K }), T(as.data(), RAD_F32, { M, sk }), T(w.data(), RAD_F8E4M3, { N, K }),
            T(ws.data(), RAD_BF16, { sn, sk }), T(y.data(), RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(nullptr, RAD_U8, { 1 }) };
        ts[5].data = nullptr; ts[5].rank = 0; ts[6].data = nullptr; ts[6].rank = 0;
        if (run(halo_rad_gemm_fp8a8_m16, nullptr, ts, { PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a8") })) {
            ok(tag, false, "the narrow row failed"); return;
        }
    }
    std::vector<uint8_t> q((size_t)(M * nout), 0xee), rq((size_t)(M * nout));
    std::vector<float> qs((size_t)(M * nout / 128), -1.f), rqs((size_t)(M * nout / 128));
    std::vector<uint16_t> yg((size_t)(M * nout), 0xffff), ry((size_t)(M * nout));
    for (int64_t m = 0; m < M; ++m)
        for (int64_t g = 0; g < nout / 128; ++g) {
            float v[128], amax = 0.f;
            for (int c = 0; c < 128; ++c) {
                const int64_t n = g * 128 + c;
                const float gv = bf(y[(size_t)(m * N + n)]), uv = bf(y[(size_t)(m * N + n + nout)]);
                const uint16_t hb = to_bf(gv / (1.0f + std::exp(-gv)) * uv);
                ry[(size_t)(m * nout + n)] = hb;
                v[c] = bf(hb);
                amax = std::fmax(amax, std::fabs(v[c]));
            }
            const float sc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
            rqs[(size_t)(m * (nout / 128) + g)] = sc;
            for (int c = 0; c < 128; ++c)
                rq[(size_t)(m * nout + g * 128 + c)] =
                    rad_f32_to_fp8e4m3(std::fmin(std::fmax(v[c] * (1.0f / sc), -448.0f), 448.0f));
        }
    std::vector<RadTensor> ts = {
        T(a.data(), RAD_F8E4M3, { M, K }), T(as.data(), RAD_F32, { M, sk }), T(w.data(), RAD_F8E4M3, { N, K }),
        T(ws.data(), RAD_BF16, { sn, sk }), T(q.data(), RAD_F8E4M3, { M, nout }),
        T(qs.data(), RAD_F32, { M, nout / 128 }), T(yg.data(), RAD_BF16, { M, nout }) };
    if (run(halo_rad_gemm_fp8a8_gated_m16, nullptr, ts,
            { PI("M", M), PI("N", N), PI("K", K), PI("group", 128), PS("dtype", "fp8a8"), PS("act", "silu") })) {
        ok(tag, false, "the gated row failed"); return;
    }
    size_t dq = 0, ds = 0, dy = 0;
    for (size_t i = 0; i < q.size(); ++i) { dq += q[i] != rq[i]; dy += yg[i] != ry[i]; }
    for (size_t i = 0; i < qs.size(); ++i) ds += std::memcmp(&qs[i], &rqs[i], 4) != 0;
    ok(tag, dq == 0 && ds == 0 && dy == 0,
       std::to_string(dq) + " codes, " + std::to_string(ds) + " scales, " + std::to_string(dy) + " products differ");
}

/* gemm_bf16_nt_tiled's contract (its header): byte-identical to gemm_bf16_nt_m64 at the same
 * shape -- both take SK from cfg_for, accumulate each slice from zero one WMMA per 16 K in order
 * and sum the slices from 0.f. Both tile arms (tile 1: 128 x 128, tile 2: 64 x 64), a K the split
 * law splits, ragged M and N. */
static void bf16_case(int64_t M, int64_t N, int64_t K, int tile) {
    const std::string tag = "bf16 tiled(" + std::to_string(tile) + ") == m64, M=" + std::to_string(M) +
                            " N=" + std::to_string(N) + " K=" + std::to_string(K);
    std::vector<uint16_t> a((size_t)(M * K)), w((size_t)(N * K));
    for (size_t i = 0; i < a.size(); ++i) a[i] = to_bf((float)((int)(mix((uint32_t)i + 41) % 2001) - 1000) / 400.0f);
    for (size_t i = 0; i < w.size(); ++i) w[i] = to_bf((float)((int)(mix((uint32_t)i + 0x9e37) % 2001) - 1000) / 900.0f);
    std::vector<uint16_t> y[2];
    for (auto& v : y) v.assign((size_t)(M * N), 0xffff);
    int rc = 0;
    for (int v = 0; v < 2; ++v) {
        std::vector<RadTensor> ts = { T(a.data(), RAD_BF16, { M, K }), T(w.data(), RAD_BF16, { N, K }),
                                      T(y[v].data(), RAD_BF16, { M, N }) };
        std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("K", K), PS("dtype", "bf16") };
        if (v == 0) ps.push_back(PI("tile", tile));
        rc |= run(v == 0 ? halo_rad_gemm_bf16_tiled : halo_rad_gemm_bf16_m64, nullptr, ts, ps);
    }
    if (rc) { ok(tag, false, "rc " + std::to_string(rc)); return; }
    size_t d = 0;
    for (size_t i = 0; i < y[0].size(); ++i) d += y[0][i] != y[1][i];
    ok(tag, d == 0, std::to_string(d) + " of " + std::to_string(y[0].size()) + " differ");
}

/* mxfp4: tiled (plane), tiled (none), m64 at sk=1 over 64-row slices; both scale grids. */
/* pb1/pb2 > 0: a partitioned weight (mx_parts) -- the activation one K-wide copy a part, side by
 * side, each part a column window of the output. */
static void mx_case(int64_t M, int64_t N, int64_t K, bool ablk, int64_t pb1 = 0, int64_t pb2 = 0) {
    const int segs = 1 + (pb1 > 0) + (pb2 > 0);
    const std::string tag = "mxfp4a8 tiled M=" + std::to_string(M) + " N=" + std::to_string(N) + " K=" +
                            std::to_string(K) + (ablk ? " ascale=row x 128" : " ascale=row") +
                            (segs > 1 ? " parts=" + std::to_string(segs) : "");
    const int64_t kb = K / 128, sa = ablk ? kb * segs : 1, KA = K * segs;
    std::vector<uint8_t> a((size_t)(M * KA)), w((size_t)(N * K / 2)), e((size_t)(K / 32 * N)), rf((size_t)N);
    std::vector<float> as((size_t)(M * sa));
    for (size_t i = 0; i < a.size(); ++i) { uint8_t c = (uint8_t)mix((uint32_t)i * 2u + 3u); a[i] = (c & 0x7f) == 0x7f ? (uint8_t)(c ^ 1) : c; }
    for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t)mix((uint32_t)i + 0x2545f491u);
    for (int64_t n = 0; n < N; ++n) rf[(size_t)n] = (uint8_t)(120 + mix((uint32_t)n + 11) % 15);
    for (int64_t b = 0; b < K / 32; ++b)
        for (int64_t n = 0; n < N; ++n) e[(size_t)(b * N + n)] = (uint8_t)(rf[(size_t)n] - 10 + (int)(mix((uint32_t)(b * N + n) + 31) % 21));
    for (size_t i = 0; i < as.size(); ++i) as[i] = std::ldexp(1.0f + (mix((uint32_t)i + 13) & 255) / 256.0f, -(int)(mix((uint32_t)i + 17) % 9) - 4);
    std::vector<uint16_t> y[3];
    for (auto& v : y) v.assign((size_t)(M * N), 0xffff);
    auto go = [&](int (*l)(const RadArgs*, RadStream), int64_t (*h)(const RadArgs*), int64_t m0, int64_t mm, std::vector<uint16_t>& out, bool one) {
        std::vector<RadTensor> ts = {
            T(a.data() + m0 * KA, RAD_F8E4M3, { mm, KA }),
            ablk ? T(as.data() + m0 * sa, RAD_F32, { mm, sa }) : T(as.data() + m0, RAD_F32, { mm }),
            T(w.data(), RAD_U8, { (int64_t)w.size() }), T(e.data(), RAD_U8, { (int64_t)e.size() }),
            T(out.data() + m0 * N, RAD_BF16, { mm, N }), T(nullptr, RAD_F32, { 1 }), T(rf.data(), RAD_U8, { N }) };
        ts[5].data = nullptr; ts[5].rank = 0;
        std::vector<RadParam> ps = { PI("M", mm), PI("N", N), PI("K", K), PI("group", 32), PS("dtype", "mxfp4a8") };
        if (one) ps.push_back(PI("sk", 1));
        if (pb1) ps.push_back(PI("pb1", pb1));
        if (pb2) ps.push_back(PI("pb2", pb2));
        return run(l, h, ts, ps);
    };
    int rc = go(halo_rad_gemm_mxfp4a8_tiled, halo_rad_gemm_mxfp4a8_tiled_plane_scratch, 0, M, y[0], false);
    rc |= go(halo_rad_gemm_mxfp4a8_tiled, nullptr, 0, M, y[1], false);
    for (int64_t m0 = 0; m0 < M; m0 += 64)
        rc |= go(halo_rad_gemm_mxfp4a8_m64, halo_rad_gemm_mxfp4a8_m64_scratch, m0, std::min<int64_t>(64, M - m0), y[2], true);
    if (rc) { ok(tag, false, "rc " + std::to_string(rc)); return; }
    size_t d0 = 0, d1 = 0;
    for (size_t q = 0; q < y[2].size(); ++q) { d0 += y[0][q] != y[2][q]; d1 += y[1][q] != y[2][q]; }
    ok(tag + " (plane) == m64 sk=1 slices", d0 == 0, std::to_string(d0) + " differ");
    ok(tag + " (no plane) == m64 sk=1 slices", d1 == 0, std::to_string(d1) + " differ");
}

/* The decode row with its hook's scratch (split partials + the activation plane) against the same
 * call with the plane's bytes withheld (partials only: the in-kernel decode), at every split. */
static void mxd_case(int64_t M, int64_t N, int64_t K, bool ablk, int ks) {
    const std::string tag = "mxfp4a8_decode M=" + std::to_string(M) + " N=" + std::to_string(N) + " K=" +
                            std::to_string(K) + " ks=" + std::to_string(ks) + (ablk ? " ascale=row x 128" : " ascale=row");
    const int64_t kb = K / 128, sa = ablk ? kb : 1;
    std::vector<uint8_t> a((size_t)(M * K)), w((size_t)(N * K / 2)), e((size_t)(K / 32 * N)), rf((size_t)N);
    std::vector<float> as((size_t)(M * sa));
    for (size_t i = 0; i < a.size(); ++i) { uint8_t c = (uint8_t)mix((uint32_t)i * 2u + 3u); a[i] = (c & 0x7f) == 0x7f ? (uint8_t)(c ^ 1) : c; }
    for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t)mix((uint32_t)i + 0x2545f491u);
    for (int64_t n = 0; n < N; ++n) rf[(size_t)n] = (uint8_t)(120 + mix((uint32_t)n + 11) % 15);
    for (int64_t b = 0; b < K / 32; ++b)
        for (int64_t n = 0; n < N; ++n) e[(size_t)(b * N + n)] = (uint8_t)(rf[(size_t)n] - 10 + (int)(mix((uint32_t)(b * N + n) + 31) % 21));
    for (size_t i = 0; i < as.size(); ++i) as[i] = std::ldexp(1.0f + (mix((uint32_t)i + 13) & 255) / 256.0f, -(int)(mix((uint32_t)i + 17) % 9) - 4);
    std::vector<uint16_t> y[3];
    for (auto& v : y) v.assign((size_t)(M * N), 0xffff);
    int rc = 0;
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<RadTensor> ts = {
            T(a.data(), RAD_F8E4M3, { M, K }),
            ablk ? T(as.data(), RAD_F32, { M, kb }) : T(as.data(), RAD_F32, { M }),
            T(w.data(), RAD_U8, { (int64_t)w.size() }), T(e.data(), RAD_U8, { (int64_t)e.size() }),
            T(y[pass].data(), RAD_BF16, { M, N }), T(nullptr, RAD_F32, { 1 }), T(rf.data(), RAD_U8, { N }) };
        ts[5].data = nullptr; ts[5].rank = 0;
        std::vector<RadParam> ps = { PI("M", M), PI("N", N), PI("K", K), PI("group", 32), PS("dtype", "mxfp4a8"), PI("ks", ks) };
        RadArgs ra; std::memset(&ra, 0, sizeof ra);
        ra.t = ts.data(); ra.n_t = (int)ts.size(); ra.p = ps.data(); ra.n_p = (int)ps.size(); ra.world_size = 1;
        int64_t need = halo_rad_gemm_mxfp4a8_decode_scratch(&ra);
        if (pass == 1) need -= M * K * 2;                    /* partials only: no plane */
        std::vector<char> sc((size_t)std::max<int64_t>(need, 1), (char)0x5a);
        ra.scratch = need > 0 ? sc.data() : nullptr; ra.scratch_bytes = need > 0 ? need : 0;
        rc |= halo_rad_gemm_mxfp4a8_decode(&ra, nullptr);
    }
    if (rc) { ok(tag, false, "rc " + std::to_string(rc)); return; }
    size_t d = 0;
    for (size_t q = 0; q < y[0].size(); ++q) d += y[0][q] != y[1][q];
    ok(tag + " plane == no plane", d == 0, std::to_string(d) + " differ");
    /* At one split the decode row's per-element arithmetic is the skinny kernel's (its header):
     * the same bytes as gemm_mxfp4a8_nt_m64 at sk=1, over 64-row slices. */
    if (ks != 1) return;
    for (int64_t m0 = 0; m0 < M && !rc; m0 += 64) {
        const int64_t mm = std::min<int64_t>(64, M - m0);
        std::vector<RadTensor> ts = {
            T(a.data() + m0 * K, RAD_F8E4M3, { mm, K }),
            ablk ? T(as.data() + m0 * sa, RAD_F32, { mm, kb }) : T(as.data() + m0, RAD_F32, { mm }),
            T(w.data(), RAD_U8, { (int64_t)w.size() }), T(e.data(), RAD_U8, { (int64_t)e.size() }),
            T(y[2].data() + m0 * N, RAD_BF16, { mm, N }), T(nullptr, RAD_F32, { 1 }), T(rf.data(), RAD_U8, { N }) };
        ts[5].data = nullptr; ts[5].rank = 0;
        rc |= run(halo_rad_gemm_mxfp4a8_m64, halo_rad_gemm_mxfp4a8_m64_scratch, ts,
                  { PI("M", mm), PI("N", N), PI("K", K), PI("group", 32), PS("dtype", "mxfp4a8"), PI("sk", 1) });
    }
    if (rc) { ok(tag + " == nt_m64 sk=1", false, "rc " + std::to_string(rc)); return; }
    d = 0;
    for (size_t q = 0; q < y[0].size(); ++q) d += y[0][q] != y[2][q];
    ok(tag + " == nt_m64 sk=1 slices", d == 0, std::to_string(d) + " differ");
}

int main(int argc, char** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    if (only.empty() || only == "fp8") {
        fp8_case(100, 256, 384, false);    /* 64 x 64 arm */
        fp8_case(520, 1024, 384, false);   /* 128 x 128 arm, ragged M */
    }
    if (only.empty() || only == "bf16") {
        bf16_case(40, 272, 640, 2);
        bf16_case(64, 512, 2560, 1);
        bf16_case(17, 1040, 5120, 2);
        bf16_case(64, 384, 1280, 1);
    }
    if (only.empty() || only == "gated") {
        gated_ref_case(5, 512, 256);
        gated_ref_case(40, 1024, 256);
        gated_ref_case(64, 512, 256);
    }
    if (only.empty() || only == "i8") {
        fp8_case(100, 256, 384, true);
        fp8_case(520, 1024, 384, true);
    }
    if (only.empty() || only == "mx") {
        mx_case(100, 272, 640, true);
        mx_case(100, 272, 640, false);
        mx_case(330, 1040, 384, true);     /* 128 x 128 arm: (1040/128 -> 9) x 3 = 27 < 32 ... */
        mx_case(520, 1040, 256, false);    /* 9 x 5 = 45 >= 32: 128 x 128, ragged N */
        mx_case(100, 272, 384, true, 96, 192);    /* three parts (a fused q|k|v), ragged last */
        mx_case(300, 1040, 256, true, 512);       /* two parts, 128 x 128 arm (parts need the 128 grid) */
    }
    if (only.empty() || only == "mxd") {
        for (int ks : { 1, 2, 4 }) {
            mxd_case(8, 400, 1024, true, ks);
            mxd_case(72, 272, 640, false, ks == 4 ? 2 : ks);
        }
    }
    std::printf("%s (%d failure(s))\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
