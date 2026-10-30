/* tests/halo_emu (from tests/r4d_emu/test_moemx.cpp): the MXFP4 expert GEMM (halo_moe_mxfp4.hip) on the host SIMT emulator. The form the
 * rule picks at each case (named per case, so a moved bound shows) must match a host restatement
 * of the per-element sequence bit for bit: per 128 of K in order, per 32-wide group a fresh
 * 32-product sum in k order (the emulated WMMA pair), folded as raw * activation scale * E8M0
 * scale with two roundings and an add. Away from GLM's slices the expert-major forms (K 256 and a
 * longer K) and the row groups; at GLM's TP4 down slice the GLM prefill form, at its TP8 gate_up
 * slice the P64 form with streamed A. Runs of 0, 1, 15, 16, 17, 63, 64, 65 and 130 rows, dropped
 * slots inside runs and past the routed rows, columns an odd expert does not have, a shorter K for
 * odd experts, E8M0 codes 0 and 255, parts 2, token and sorted rows, stacked planes and pointer
 * tables. At GLM's TP8 slices below P64 the small-M form, with runs longer than one tile. The
 * table decode is also held to the arithmetic one over every code.
 *
 * libhalo serves every one of these shapes with one kernel (mx_moe_kernel in halo_moe_mxfp4.hip),
 * so this copy does not assert which libr4d form ran; each case's output is still held to the host
 * restatement bit for bit. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "halo_plugin.h"
#include "halo_mxfp4.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern const char* emu_last_kernel;

/* The bytewise arithmetic decode, the reference of halo_mxfp4_unpack8_exact: per byte, magnitudes
 * from 2 map to (m + 12) * 4, 0 and 1 to 0 and 48, the sign to bit 7; then the low and high
 * nibbles' bytes interleave in k order. */
static uint32_t fp4x4_ref(uint32_t n) {
    const uint32_t mag = n & 0x07070707u, flag = (mag + 0x06060606u) & 0x08080808u;
    const uint32_t mask = (flag << 5) - (flag >> 3);
    return (((mag + 0x0c0c0c0cu) << 2) & mask) | (((mag & 0x01010101u) * 0x30u) & ~mask) |
           ((n & 0x08080808u) << 4);
}
static int check_decode() {
    int fails = 0;
    auto one = [&](uint32_t w) {
        const uint32_t low = fp4x4_ref(w & 0x0f0f0f0fu), high = fp4x4_ref((w >> 4) & 0x0f0f0f0fu);
        const uint32_t lo_ref = __byte_perm(low, high, 0x5140),
                       hi_ref = __byte_perm(low, high, 0x7362);
        uint32_t lo, hi, tlo, thi;
        halo_mxfp4_unpack8_exact(w, &lo, &hi);
        halo_mxfp4_unpack8(w, 0, &tlo, &thi);
        if (lo != lo_ref || hi != hi_ref || lo != tlo || hi != thi) {
            if (fails < 4)
                std::printf("FAIL decode of %08x: table %08x %08x, arithmetic %08x %08x\n", w, hi,
                            lo, hi_ref, lo_ref);
            ++fails;
        }
        /* each byte against the e2m1 value it encodes */
        for (int e = 0; e < 8; ++e) {
            const uint32_t c = (w >> (4 * e)) & 15u;
            static const float mag[8] = { 0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f };
            const float want = (c & 8u ? -1.f : 1.f) * mag[c & 7u];
            const uint8_t b = (uint8_t)(((e < 4 ? lo : hi) >> (8 * (e & 3))) & 0xffu);
            if (rad_fp8e4m3_to_f32(b) != want || (b & 0x80u) != (c & 8u ? 0x80u : 0u)) {
                if (fails < 4) std::printf("FAIL code %x at element %d -> byte %02x\n", c, e, b);
                ++fails;
            }
        }
    };
    /* the bytes decode independently: every pair of adjacent bytes exhaustively, at both halves
     * of the word, and random words over all four */
    for (uint32_t v = 0; v < 65536; ++v) {
        one(v);
        one(v << 16);
        one(v << 8);
    }
    std::mt19937 rng(7);
    for (int i = 0; i < 200000; ++i) one((uint32_t)rng());
    return fails;
}

struct Case {
    const char* name;
    int N, K, N_odd, K_odd, parts, top;
    bool sorted_a, tables, exceptional;
    std::vector<int> runs;
    int tail;           // dropped slots past the routed rows
    const char* kernel; // the form the rule picks
};

/* Byte equality with every bf16 NaN taken as one value. On x86 inf - inf gives the negative
 * default NaN, and an add of two NaNs returns its first operand, whose side the compiler may swap;
 * the card generates the positive quiet NaN, so a NaN's sign carries no information here. */
static bool nan16(uint16_t v) { return (v & 0x7f80u) == 0x7f80u && (v & 0x7fu); }
static size_t first_diff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        if (a[i] != b[i] && !(nan16(a[i]) && nan16(b[i]))) return i;
    return a.size() == b.size() ? a.size() : std::min(a.size(), b.size());
}

static RadTensor ten(void* p, uint32_t dt, std::vector<int64_t> shape,
                     std::vector<int64_t> stride) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
        t.shape[i] = shape[i];
        t.stride[i] = stride[i];
    }
    return t;
}

/* GLM's 288 experts: the given runs at experts spread over both classes, every other one empty. */
static std::vector<int> glm_runs(std::vector<int> live) {
    static const int at[] = { 2, 3, 50, 101, 150, 199, 200, 241, 286, 287 };
    std::vector<int> runs(288, 0);
    for (size_t i = 0; i < live.size() && i < sizeof at / sizeof at[0]; ++i) runs[at[i]] = live[i];
    return runs;
}

int main() {
    int fails = check_decode(), cases = 1;
    if (fails) std::printf("FAIL table decode: %d mismatches\n", fails);

    const std::vector<int> runs9 = { 0, 1, 15, 16, 17, 63, 64, 65, 130, 141 };
    const char* const staged = "(moe_mxfp4_expert_staged_kernel<128, 4>)";
    const char* const expert = "(moe_mxfp4_expert_kernel<4, 256>)";
    const char* const groups = "moe_mxfp4_fp8_kernel";
    const Case all[] = {
        { "gate_up-like, token rows", 128, 384, 128, 384, 2, 4, false, false, false, runs9, 6,
          groups },
        { "odd experts narrower and shorter", 128, 384, 64, 256, 2, 4, false, true, false, runs9,
          2, groups },
        { "down-like, sorted rows", 64, 256, 64, 256, 1, 2, true, true, false, runs9, 4, groups },
        { "exceptional E8M0 codes", 64, 256, 64, 256, 2, 4, false, false, true, runs9, 4, groups },
        { "expert-major, K 384",
          64,
          384,
          64,
          384,
          2,
          2,
          false,
          true,
          false,
          { 0, 1, 15, 16, 17, 30 },
          3,
          expert },
        { "expert-major, K 256, exceptional codes",
          64,
          256,
          32,
          256,
          1,
          2,
          true,
          false,
          true,
          { 1, 0, 3, 15, 1 },
          2,
          staged },
        { "expert-major, K 512:384, runs past 64 rows, N under a block",
          96,
          512,
          96,
          384,
          1,
          2,
          false,
          false,
          false,
          { 32, 33, 31, 96, 128, 129, 0, 2 },
          9,
          expert },
        { "expert-major, K 256:128, column tiles cut",
          192,
          256,
          128,
          128,
          2,
          4,
          true,
          true,
          true,
          { 64, 63, 65, 1, 0, 127, 17 },
          5,
          staged },
        { "GLM TP4 down, the GLM prefill form",
          4096,
          512,
          4096,
          512,
          1,
          8,
          true,
          true,
          true,
          glm_runs({ 63, 96, 1, 130, 32, 31, 33, 64, 65 }),
          13,
          "moe_mxfp4_prefill_kernel" },
        { "GLM TP8 gate_up, P64 BM32 BN64 K256",
          512,
          4096,
          512,
          4096,
          2,
          8,
          false,
          true,
          false,
          glm_runs({ 200, 33, 300, 31, 200 }),
          20,
          "(moe_mxfp4_p64_kernel<32, 64, 256, 1, false>)" },
        { "GLM TP8 gate_up, small-M form, four tokens",
          512,
          4096,
          512,
          4096,
          2,
          8,
          false,
          true,
          false,
          glm_runs({ 1, 2, 1, 3, 1, 6, 1, 2, 9, 1 }),
          5,
          "(moe_mxfp4_small_kernel<512, 1>)" },
        { "GLM TP8 gate_up, small-M form, runs past one tile",
          512,
          4096,
          512,
          4096,
          2,
          8,
          false,
          true,
          true,
          glm_runs({ 12, 1, 16, 17, 3, 0, 5, 9, 2, 33 }),
          2,
          "(moe_mxfp4_small_kernel<512, 1>)" },
        { "GLM TP8 down, small-M form, a run past one tile",
          4096,
          256,
          4096,
          256,
          1,
          8,
          true,
          true,
          true,
          glm_runs({ 3, 1, 17, 2, 1, 1, 1 }),
          5,
          "(moe_mxfp4_small_kernel<256, 4>)" },
        { "GLM TP8 down, P64 BM32 BN64 K128 x4, A resident",
          4096,
          256,
          4096,
          256,
          1,
          8,
          true,
          true,
          true,
          glm_runs({ 900, 801, 779, 700, 700, 581, 600, 600, 63, 1 }),
          40,
          "(moe_mxfp4_p64_kernel<32, 64, 128, 4, true>)" },
    };

    for (const Case& c : all) {
        const int E = (int)c.runs.size(), NK = std::max(c.K, c.K_odd), OUT = std::max(c.N, c.N_odd);
        int routed = 0;
        for (int r : c.runs) routed += r;
        const int T = (routed + c.tail + c.top - 1) / c.top * c.top;
        // An odd pitch on every other case: the LDS output's 16-byte stores need an aligned row.
        const int Arows = c.sorted_a ? T : T / c.top, ap = NK + 16, asp = NK / 128 + 1,
                  yp = OUT + ((&c - all) % 2 ? 5 : 8);
        std::mt19937 rng(1000 + cases);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::uniform_real_distribution<float> oct(-6.f, 6.f);
        std::vector<uint8_t> a((size_t)Arows * ap);
        for (auto& v : a) v = rad_f32_to_fp8e4m3(nd(rng) * 2.f);
        std::vector<float> as((size_t)Arows * asp);
        for (auto& v : as) v = std::exp2(oct(rng));
        std::vector<int> offsets(E + 1, 0), sorted(T, -1), owner(T, -1);
        for (int e = 0; e < E; ++e) offsets[e + 1] = offsets[e] + c.runs[e];
        for (int e = 0; e < E; ++e)
            for (int r = offsets[e]; r < offsets[e + 1]; ++r) {
                owner[r] = e;
                // A dropped slot about every eleventh row, the first rows of some runs included.
                sorted[r] = (r * 13 + e) % 11 == 0 ? -1 : (int)(rng() % (uint32_t)T);
            }
        /* Stacked planes hold every expert; a pointer table names one of at most three slices a
         * class, so GLM's 288 experts stay small. */
        const int n[2] = { c.N, c.N_odd }, k[2] = { c.K, c.K_odd },
                  nexp[2] = { (E + 1) / 2, E / 2 };
        std::vector<uint8_t> w[2], s[2];
        std::vector<const void*> wt[2], st[2];
        std::vector<std::vector<float>> wf[2];
        int slices[2];
        for (int q = 0; q < 2; ++q) {
            slices[q] = c.tables ? std::min(nexp[q], 3) : nexp[q];
            w[q].resize((size_t)slices[q] * n[q] * k[q] / 2);
            s[q].resize((size_t)slices[q] * n[q] * k[q] / 32);
            for (auto& v : w[q]) v = (uint8_t)rng();
            static const uint8_t special[] = { 0, 255, 1, 254, 127 };
            for (size_t i = 0; i < s[q].size(); ++i)
                s[q][i] = c.exceptional && rng() % 9 == 0 ? special[rng() % 5]
                                                          : (uint8_t)(118 + rng() % 16);
            for (int x = 0; x < nexp[q]; ++x) {
                wt[q].push_back(w[q].data() + (size_t)(x % slices[q]) * n[q] * k[q] / 2);
                st[q].push_back(s[q].data() + (size_t)(x % slices[q]) * n[q] * k[q] / 32);
            }
            /* each slice's codes as the E4M3 values the table decode gives */
            for (int x = 0; x < slices[q]; ++x) {
                std::vector<float> f((size_t)n[q] * k[q]);
                const uint8_t* src = w[q].data() + (size_t)x * n[q] * k[q] / 2;
                for (size_t i = 0; i < f.size(); ++i) {
                    uint32_t lo, hi;
                    halo_mxfp4_unpack8_exact((src[i / 2] >> (4 * (i & 1))) & 15u, &lo, &hi);
                    f[i] = rad_fp8e4m3_to_f32((uint8_t)(lo & 0xffu));
                }
                wf[q].push_back(std::move(f));
            }
        }
        auto plane = [&](int q, bool scale) {
            const int64_t cols = scale ? k[q] / 32 : k[q];
            void* p = c.tables ? (void*)(scale ? st[q].data() : wt[q].data())
                               : (void*)(scale ? s[q].data() : w[q].data());
            RadTensor t = ten(p, scale ? RAD_E8M0 : RAD_FP4E2M1, { nexp[q], n[q], cols },
                              { c.tables ? 0 : (int64_t)n[q] * cols, cols, 1 });
            return t;
        };

        /* the per-element sequence, on the host */
        std::vector<float> af(a.size());
        for (size_t i = 0; i < a.size(); ++i) af[i] = rad_fp8e4m3_to_f32(a[i]);
        std::vector<uint16_t> ref((size_t)T * yp, 0x7777);
        for (int r = 0; r < T; ++r)
            for (int col = 0; col < OUT; ++col) {
                uint16_t& out = ref[(size_t)r * yp + col];
                out = 0;
                if (owner[r] < 0 || sorted[r] < 0) continue;
                const int e = owner[r], q = e & 1, ar = c.sorted_a ? r : sorted[r] / c.top;
                const int pw = OUT / c.parts, cw = n[q] / c.parts, within = col % pw;
                const bool valid = within < cw;
                const int wn = (col / pw) * cw + within;
                const float* wr = wf[q][(size_t)((e / 2) % slices[q])].data() + (size_t)wn * k[q];
                const uint8_t* sr = (const uint8_t*)st[q][(size_t)(e / 2)] + (size_t)wn * k[q] / 32;
                const float* arow = af.data() + (size_t)ar * ap;
                float total = 0.f;
                for (int kb = 0; kb < k[q]; kb += 128)
                    for (int g = 0; g < 4; ++g) {
                        float raw = 0.f;
                        for (int kk = kb + g * 32; kk < kb + g * 32 + 32; ++kk)
                            raw += arow[kk] * (valid ? wr[kk] : 0.f);
                        const uint32_t x = valid ? sr[kb / 32 + g] : 127u;
                        uint32_t bits = x == 255 ? 0x7fc00000u : x ? x << 23 : 0x00400000u;
                        float scale;
                        std::memcpy(&scale, &bits, 4);
                        float t = raw * as[(size_t)ar * asp + kb / 128];
                        t = t * scale;
                        total = total + t;
                    }
                uint32_t u;
                std::memcpy(&u, &total, 4);
                out = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >>
                                 16);   // f32_to_bf16, halo_common.h
            }
        int live = 0;
        for (uint16_t v : ref) live += v != 0 && v != 0x7777;
        if (live < T * OUT / 3) {
            std::printf("FAIL %s: output mostly zero\n", c.name);
            ++fails;
        }

        RadTensor t[9] = { ten(a.data(), RAD_F8E4M3, { Arows, NK }, { ap, 1 }),
                           ten(as.data(), RAD_F32, { Arows, NK / 128 }, { asp, 1 }),
                           plane(0, false),
                           plane(0, true),
                           ten(sorted.data(), RAD_I32, { T }, { 1 }),
                           ten(offsets.data(), RAD_I32, { E + 1 }, { 1 }),
                           plane(1, false),
                           plane(1, true),
                           RadTensor{} };
        RadParam p[11]{};
        auto pi = [](const char* key, long long v) {
            RadParam x{};
            x.key = key;
            x.kind = RAD_P_INT;
            x.ival = v;
            return x;
        };
        auto ps = [](const char* key, const char* v) {
            RadParam x{};
            x.key = key;
            x.kind = RAD_P_STR;
            x.sval = v;
            return x;
        };
        p[0] = pi("M", T / c.top);
        p[1] = pi("N", c.N);
        p[2] = pi("K", c.K);
        p[3] = pi("N_odd", c.N_odd);
        p[4] = pi("K_odd", c.K_odd);
        p[5] = pi("n_expert", E);
        p[6] = pi("top_k", c.top);
        p[7] = pi("parts", c.parts);
        p[8] = pi("group", 128);
        p[9] = ps("a_order", c.sorted_a ? "sorted" : "token");
        p[10] = ps("dtype", "mxfp4a8");
        std::vector<uint16_t> got((size_t)T * yp, 0x7777);
        t[8] = ten(got.data(), RAD_BF16, { T, OUT }, { yp, 1 });
        RadArgs ar{};
        ar.t = t;
        ar.n_t = 9;
        ar.p = p;
        ar.n_p = 11;
        emu_last_kernel = "";
        ++cases;
        const int rc = halo_rad_moe_mxfp4(&ar, nullptr);
        const size_t i = first_diff(got, ref);
        if (rc != RAD_OK || i != got.size()) {
            std::printf("FAIL %s (T %d): ran %s (want %s), rc %d, first word %zu (row %zu col %zu: "
                        "%04x, want %04x)\n",
                        c.name, T, emu_last_kernel, c.kernel, rc, i, i / yp, i % yp,
                        i < got.size() ? got[i] : 0, i < ref.size() ? ref[i] : 0);
            ++fails;
        }
    }
    std::printf("%d case(s), %d failure(s)\n", cases, fails);
    return fails ? 1 : 0;
}
