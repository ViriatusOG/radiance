/* tests/r4d_emu: the multi-row decode-band form of gg_gemm (r4d_ds41_ggml_mr.hip) against the stock decode kernel
 * (gg_gemv_kernel, the same unit's copy of r4d_ds41_ggml.hip's source) on the host SIMT emulator, byte for byte.
 *
 * Every (H, R) pair the form has, at its largest MT cap and at small caps (so M splits over grid.y), at one, two
 * and eight waves a block, over q5_K / q6_K / q8_0 rows of DeepSeek-V4.1's dense K values (512 768 1280 2048
 * 4096 5120 6144) and an N that leaves partial row groups and row slots, at M rows from 1 to 48. Weights: random
 * quant bytes, scales random finite f16 (normal, subnormal, near the f16 top); activations: random bf16 with
 * zeros of both signs, subnormals, huge values (products and sums that overflow to inf, inf - inf NaNs) and
 * infinities. Also, as information: the stock kernel's row m of an M-row call equals the same row alone (M = 1).
 *
 *   r4d_emu_ds41mr [quick]      quick: three K values, fewer M */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_ds41_ggml_mr.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int r4d_gg_mr_emu_stock(const R4dGgMrCall& c, int rpw);
extern "C" void r4d_gg_mr_pin(int h, int r, int mt, int wn);
extern "C" const char* r4d_gg_mr_last();

namespace {

enum { Q8_0 = 0, Q5_K = 1, Q6_K = 2 };   /* gg_gemm's GG_* codes */

uint64_t g_rs = 0x9E3779B97F4A7C15ull;
uint64_t rnd() { g_rs ^= g_rs << 13; g_rs ^= g_rs >> 7; g_rs ^= g_rs << 17; return g_rs; }

uint16_t f2bf(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
/* a random finite f16: mostly normal around 2^-7, some subnormal, some near the top */
uint16_t rnd_f16() {
    const uint32_t k = (uint32_t)(rnd() % 100);
    const uint16_t sign = (uint16_t)((rnd() & 1) << 15);
    uint16_t e;
    if (k < 8) e = 0;                                  /* subnormal (or zero) */
    else if (k < 12) e = (uint16_t)(28 + rnd() % 3);   /* up to 2^15 */
    else e = (uint16_t)(5 + rnd() % 8);                /* 2^-10 .. 2^-3 */
    return (uint16_t)(sign | (e << 10) | (uint16_t)(rnd() & 0x3ff));
}
/* an activation row: random values in [-4, 4] with zeros of both signs and bf16 subnormals sprinkled; one row in
 * eight also takes an infinity, one in eight a few huge values (products and sums past f32: inf, inf - inf) */
void fill_row(uint16_t* x, int K, int mode) {
    for (int k = 0; k < K; ++k) {
        const uint32_t c = (uint32_t)(rnd() % 1000);
        if (c < 5) x[k] = 0x0000;
        else if (c < 10) x[k] = 0x8000;
        else if (c < 15) x[k] = (uint16_t)((rnd() & 1 ? 0x8000 : 0) | (rnd() % 0x7f + 1));
        else x[k] = f2bf((float)((int)(rnd() % 20001) - 10000) / 2500.0f);
    }
    if (mode == 1) x[rnd() % K] = (uint16_t)((rnd() & 1 ? 0x8000 : 0) | 0x7f80);
    if (mode == 2)
        for (int e = 0; e < 3; ++e)
            x[rnd() % K] = f2bf((rnd() & 1 ? -1.0f : 1.0f) * std::ldexp(1.0f + (float)(rnd() % 1000) / 1000.0f, 110 + (int)(rnd() % 17)));
}

int g_fail = 0, g_cases = 0;
long long g_fin = 0, g_inf = 0, g_nan = 0;   /* the stock f32 outputs' classes: how much of the check is finite */
/* the f32 output (logits_gemm_ggml's) at every type: the emulator's unit instantiates it for all three (the plugin
 * for q6_K, the head's type), and an f32 output shows a changed operation order that bf16's rounding may hide */
constexpr bool kAllF32 = true;

struct Ty { int ty; const char* name; int qk, bb; };
const Ty kTypes[] = { { Q5_K, "q5_K", 256, 176 }, { Q6_K, "q6_K", 256, 210 }, { Q8_0, "q8_0", 32, 34 } };

std::vector<uint8_t> make_w(const Ty& t, int N, int K) {
    const long long rb = (long long)K / t.qk * t.bb;
    std::vector<uint8_t> w((size_t)(N * rb));
    for (auto& b : w) b = (uint8_t)rnd();
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < K / t.qk; ++b) {
            uint8_t* blk = w.data() + n * rb + (long long)b * t.bb;
            const uint16_t d = rnd_f16(), dm = rnd_f16();
            if (t.ty == Q6_K) std::memcpy(blk + 208, &d, 2);
            else { std::memcpy(blk, &d, 2); if (t.ty == Q5_K) std::memcpy(blk + 2, &dm, 2); }
        }
    return w;
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "quick");
    std::vector<int> Ks = quick ? std::vector<int>{ 768, 1280, 5120 } : std::vector<int>{ 512, 768, 1280, 2048, 4096, 5120, 6144 };
    std::vector<int> Ms = quick ? std::vector<int>{ 1, 3, 8, 9, 24, 48 }
                                : std::vector<int>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 24, 25, 32, 40, 47, 48 };
    struct Pin { int h, r, mt, wn; };
    const Pin pins[] = { { 1, 1, 0, 0 }, { 1, 1, 4, 1 }, { 2, 1, 0, 0 }, { 2, 1, 6, 2 }, { 2, 2, 0, 2 }, { 2, 2, 6, 0 }, { 4, 1, 0, 0 }, { 4, 1, 8, 1 },
                         { 4, 2, 0, 8 }, { 4, 2, 3, 0 }, { -1, 0, 0, 0 } };
    const int N = 45;   /* partial row groups, slots and lanes' rows at every form */
    int info_same = 0, info_cases = 0;
    for (const Ty& t : kTypes) {
        for (int K : Ks) {
            if (K % t.qk) continue;
            const std::vector<uint8_t> w = make_w(t, N, K);
            const long long w_ld = (long long)K / t.qk * t.bb;
            for (int M : Ms) {
                std::vector<uint16_t> x((size_t)M * K);
                for (int m = 0; m < M; ++m) { const int r8 = (int)(rnd() % 8); fill_row(x.data() + (size_t)m * K, K, r8 == 0 ? 1 : r8 == 1 ? 2 : 0); }
                const int x_al = (M & 1) && (((uintptr_t)x.data() & 15u) == 0) ? 1 : 0;   /* both activation loads */
                for (int f32 = 0; f32 < 2; ++f32) {
                    if (f32 && t.ty != Q6_K && !kAllF32) continue;
                    const size_t ob = f32 ? 4 : 2;
                    std::vector<uint8_t> ref((size_t)M * N * ob, 0xA5), got((size_t)M * N * ob);
                    R4dGgMrCall c{ t.ty, f32 != 0, x.data(), (long long)K, x_al, w.data(), w_ld, ref.data(), (long long)N, M, N, K };
                    if (r4d_gg_mr_emu_stock(c, 1) != 0) { std::printf("FAIL stock launch %s K=%d M=%d\n", t.name, K, M); ++g_fail; continue; }
                    if (f32)
                        for (size_t e = 0; e < ref.size(); e += 4) {
                            float v; std::memcpy(&v, &ref[e], 4);
                            ++(std::isnan(v) ? g_nan : std::isinf(v) ? g_inf : g_fin);
                        }
                    /* the stock form's own invariance: rows a wave 2 gives the same bytes */
                    {
                        std::vector<uint8_t> r2((size_t)M * N * ob, 0x5A);
                        c.y = r2.data();
                        r4d_gg_mr_emu_stock(c, 2);
                        ++g_cases;
                        if (r2 != ref) { std::printf("FAIL stock rpw2 != rpw1 %s K=%d M=%d\n", t.name, K, M); ++g_fail; }
                    }
                    /* information: a row of the M-row call vs the same row alone */
                    if (!f32) {
                        int same = 1;
                        for (int m = 0; m < M && same; m += (M > 4 ? M / 4 : 1)) {
                            std::vector<uint8_t> one((size_t)N * ob, 0x33);
                            R4dGgMrCall c1 = c;
                            c1.x = x.data() + (size_t)m * K; c1.M = 1; c1.y = one.data();
                            r4d_gg_mr_emu_stock(c1, 1);
                            if (std::memcmp(one.data(), ref.data() + (size_t)m * N * ob, (size_t)N * ob)) same = 0;
                        }
                        ++info_cases; info_same += same;
                    }
                    for (const Pin& p : pins) {
                        std::fill(got.begin(), got.end(), 0xC3);
                        c.y = got.data();
                        r4d_gg_mr_pin(p.h, p.r, p.mt, p.wn);
                        const int rc = r4d_gg_mr_gemm(nullptr, c);
                        r4d_gg_mr_pin(0, 0, 0, 0);
                        ++g_cases;
                        if (rc != 0) { std::printf("FAIL %s K=%d M=%d pin %d/%d/%d/%d: rc %d\n", t.name, K, M, p.h, p.r, p.mt, p.wn, rc); ++g_fail; continue; }
                        if (got != ref) {
                            size_t bad = 0, first = 0;
                            for (size_t e = 0; e < got.size(); e += ob)
                                if (std::memcmp(&got[e], &ref[e], ob)) { if (!bad) first = e / ob; ++bad; }
                            std::printf("FAIL %s%s K=%d M=%d pin %d/%d/%d/%d (%s): %zu of %zu outputs differ, first m=%zu n=%zu\n",
                                        t.name, f32 ? " f32" : "", K, M, p.h, p.r, p.mt, p.wn, r4d_gg_mr_last(), bad,
                                        got.size() / ob, first / N, first % N);
                            ++g_fail;
                        }
                    }
                }
            }
            std::printf("%s K=%d: done (%d cases, %d failures so far)\n", t.name, K, g_cases, g_fail);
            std::fflush(stdout);
        }
    }
    std::printf("INFO stock: a row of an M-row call equals the row alone in %d of %d (type, K, M) cases\n", info_same, info_cases);
    std::printf("INFO f32 reference outputs: %lld finite, %lld inf, %lld NaN\n", g_fin, g_inf, g_nan);
    std::printf("%s: %d cases, %d failures\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
    return g_fail ? 1 : 0;
}
