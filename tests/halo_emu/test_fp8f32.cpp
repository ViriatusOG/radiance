/* tests/halo_emu (from tests/r4d_emu/test_fp8f32.cpp): the dense E4M3 GEMM with F32 block scales (halo_fp8_f32scale.hip) on the host SIMT
 * emulator. The form the rule picks at each shape (the decode form at one and two column tiles a
 * wave and one to eight waves a block, the 16- and 32-row tiles) must match the legacy 16x64 tile,
 * which serves an activation pitch off 16 bytes, bit for bit: the same fragments, the same mma
 * chain per 128-K block and the same fold order. Ragged row counts, pitched rows and a K that
 * leaves the decode form's last wave group partly idle are covered; the output's pitch padding
 * stays untouched. The grouped row at GLM's MTP slices is held to a host restatement
 * (grouped_cases).
 *
 * libhalo's unit (halo_gemm_fp8_f32scale.hip) is gfx1151's own design, not a port of these forms,
 * so this copy does not assert which kernel ran (libr4d's names: the legacy tile, the decode and
 * small forms); every output comparison stands -- the pitched activation against the unpitched one
 * byte for byte, and the grouped row against the host restatement byte for byte. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "halo_plugin.h"
#include "ref_host.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern const char* emu_last_kernel;

static RadTensor mat(void* p, uint32_t dt, int64_t rows, int64_t cols, int64_t ld) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.stride[0] = ld;
    t.stride[1] = 1;
    return t;
}

/* The grouped row at GLM's MTP slices (288 experts, top 8) below 384 routed rows -- the small-M
 * form at gate_up, the expert-major form at down -- against a host restatement of the
 * per-element sequence: per 128 of K in order, the
 * 128-product sum of the emulated WMMA chain from zero, folded as raw * (activation scale *
 * weight scale) and added. Dropped slots inside runs and past the routed rows, runs longer than an
 * A tile, token and sorted rows, pointer tables over three slices a class. */
static int grouped_cases(int& cases) {
    struct G {
        const char* name;
        int N, K;
        bool sorted_a;
        std::vector<int> live;
        int tail;
        const char* kernel;
    };
    const G all[] = {
        { "MTP gate_up, one token", 512, 4096, false, { 1, 1, 1, 1, 1, 1, 1, 1 }, 0,
          "fp8_f32scale_small<512>" },
        { "MTP gate_up, runs past one tile", 512, 4096, false, { 12, 1, 16, 17, 3, 0, 5, 9, 2, 33 }, 2,
          "fp8_f32scale_small<512>" },
        { "MTP down, six tokens", 4096, 256, true, { 3, 1, 10, 2, 1, 5, 1, 1, 17, 4 },
          3, "(fp8_f32scale_experts<128, 4>)" },
    };
    static const int at[] = { 2, 3, 50, 101, 150, 199, 200, 241, 286, 287 };
    int fails = 0;
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> oct(-6.f, 6.f);
    for (const G& g : all) {
        const int E = 288, top = 8, N = g.N, K = g.K, kb = K / 128;
        std::vector<int> runs(E, 0);
        for (size_t i = 0; i < g.live.size(); ++i) runs[at[i]] = g.live[i];
        int routed = 0;
        for (int r : runs) routed += r;
        const int T = (routed + g.tail + top - 1) / top * top;
        const int Arows = g.sorted_a ? T : T / top, ap = K + 16, yp = N + 8;
        std::vector<uint8_t> a((size_t)Arows * ap);
        for (auto& v : a) v = rad_f32_to_fp8e4m3(nd(rng));
        std::vector<float> as((size_t)Arows * kb);
        for (auto& v : as) v = std::exp2(oct(rng));
        std::vector<int> offsets(E + 1, 0), sorted(T, -1), owner(T, -1);
        for (int e = 0; e < E; ++e) offsets[e + 1] = offsets[e] + runs[e];
        for (int e = 0; e < E; ++e)
            for (int r = offsets[e]; r < offsets[e + 1]; ++r) {
                owner[r] = e;
                sorted[r] = (r * 13 + e) % 11 == 0 ? -1 : (int)(rng() % (uint32_t)T);
            }
        std::vector<uint8_t> w[2];
        std::vector<float> s[2];
        std::vector<const void*> wt[2], st[2];
        for (int q = 0; q < 2; ++q) {
            w[q].resize((size_t)3 * N * K);
            s[q].resize((size_t)3 * (N / 128) * kb);
            for (auto& v : w[q]) v = rad_f32_to_fp8e4m3(nd(rng) * 0.5f);
            for (auto& v : s[q]) v = std::exp2(oct(rng)) * 0.01f;
            for (int x = 0; x < E / 2; ++x) {
                wt[q].push_back(w[q].data() + (size_t)(x % 3) * N * K);
                st[q].push_back(s[q].data() + (size_t)(x % 3) * (N / 128) * kb);
            }
        }
        std::vector<uint16_t> ref((size_t)T * yp, 0x7777), got((size_t)T * yp, 0x7777);
        for (int r = 0; r < T; ++r)
            for (int n = 0; n < N; ++n) {
                uint16_t& out = ref[(size_t)r * yp + n];
                out = 0;
                if (owner[r] < 0) continue;
                const int e = owner[r], q = e & 1;
                const bool rv = sorted[r] >= 0;
                const int ar = !rv ? -1 : g.sorted_a ? r : sorted[r] / top;
                const uint8_t* wr = (const uint8_t*)wt[q][(size_t)(e / 2)] + (size_t)n * K;
                const float* sr = (const float*)st[q][(size_t)(e / 2)] + (size_t)(n / 128) * kb;
                float acc = 0.f;
                for (int b = 0; b < kb; ++b) {
                    float raw = 0.f;
                    if (rv)
                        for (int k = b * 128; k < b * 128 + 128; ++k)
                            raw += rad_fp8e4m3_to_f32(a[(size_t)ar * ap + k]) *
                                   rad_fp8e4m3_to_f32(wr[k]);
                    const float scale = rv ? as[(size_t)ar * kb + b] * sr[b] : 0.f;
                    const float t = raw * scale;
                    acc = acc + t;
                }
                uint32_t u;
                std::memcpy(&u, &acc, 4);
                out = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
            }
        auto ten3 = [](const void* const* tab, uint32_t dt, int64_t n0, int64_t n1, int64_t n2) {
            RadTensor t{};
            t.data = (void*)tab;
            t.dtype = dt;
            t.rank = 3;
            t.shape[0] = n0;
            t.shape[1] = n1;
            t.shape[2] = n2;
            t.stride[0] = 0;
            t.stride[1] = n2;
            t.stride[2] = 1;
            return t;
        };
        RadTensor t[9] = { mat(a.data(), RAD_F8E4M3, Arows, K, ap),
                           mat(as.data(), RAD_F32, Arows, kb, kb),
                           ten3(wt[0].data(), RAD_F8E4M3, E / 2, N, K),
                           ten3(st[0].data(), RAD_F32, E / 2, N / 128, kb),
                           RadTensor{},
                           RadTensor{},
                           ten3(wt[1].data(), RAD_F8E4M3, E / 2, N, K),
                           ten3(st[1].data(), RAD_F32, E / 2, N / 128, kb),
                           mat(got.data(), RAD_BF16, T, N, yp) };
        t[4].data = sorted.data();
        t[4].dtype = RAD_I32;
        t[4].rank = 1;
        t[4].shape[0] = T;
        t[4].stride[0] = 1;
        t[5].data = offsets.data();
        t[5].dtype = RAD_I32;
        t[5].rank = 1;
        t[5].shape[0] = E + 1;
        t[5].stride[0] = 1;
        RadParam p[10]{};
        p[0] = { "M", RAD_P_INT, T / top };
        p[1] = { "N", RAD_P_INT, N };
        p[2] = { "K", RAD_P_INT, K };
        p[3] = { "N_odd", RAD_P_INT, N };
        p[4] = { "K_odd", RAD_P_INT, K };
        p[5] = { "n_expert", RAD_P_INT, E };
        p[6] = { "top_k", RAD_P_INT, top };
        p[7] = { "group", RAD_P_INT, 128 };
        p[8].key = "a_order";
        p[8].kind = RAD_P_STR;
        p[8].sval = g.sorted_a ? "sorted" : "token";
        p[9].key = "dtype";
        p[9].kind = RAD_P_STR;
        p[9].sval = "fp8a8f32";
        RadArgs ar{};
        ar.t = t;
        ar.n_t = 9;
        ar.p = p;
        ar.n_p = 10;
        emu_last_kernel = "";
        ++cases;
        const int rc = halo_rad_moe_fp8_f32scale(&ar, nullptr);
        size_t i = 0;
        while (i < got.size() && got[i] == ref[i]) ++i;
        int live = 0;
        for (uint16_t v : ref) live += v != 0 && v != 0x7777;
        if (rc != RAD_OK || i != got.size() || live < routed * N / 2) {
            std::printf("FAIL %s (T %d): ran %s (want %s), rc %d, first word %zu of %zu, %d live\n",
                        g.name, T, emu_last_kernel, g.kernel, rc, i, got.size(), live);
            ++fails;
        }
    }
    return fails;
}

int main() {
    std::mt19937 rng(31);
    std::normal_distribution<float> nd(0.f, 1.f);
    struct Shape {
        int M, N, K;
        const char* kernel;
    };
    const char* const dec1 = "(fp8_f32scale_decode<1, 1>)";
    const char* const dec2 = "(fp8_f32scale_decode<1, 2>)";
    const char* const rows = "fp8_f32scale_rows<R>";
    // Decode at eight, four, two and one waves a block (K blocks 12/32, 4, 2, 1), at two column
    // tiles a wave from 17 rows over N 1024 and 640; the 16-row tile, and the 32-row one at N 4096.
    const Shape shapes[] = { { 1, 256, 512, dec1 },   { 6, 128, 1536, dec1 },
                             { 17, 256, 512, dec1 },  { 48, 128, 512, dec1 },
                             { 2, 128, 256, dec1 },   { 5, 256, 128, dec1 },
                             { 3, 128, 4096, dec1 },  { 40, 1024, 512, dec2 },
                             { 20, 640, 256, dec2 },  { 70, 128, 384, rows },
                             { 130, 128, 256, rows }, { 33, 4096, 512, rows },
                             { 70, 4096, 128, rows } };
    int fails = 0, cases = 0;
    for (const Shape& sh : shapes) {
        const int M = sh.M, N = sh.N, K = sh.K, kb = K / 128, ap = K + 16, lp = K + 8, yp = N + 8;
        std::vector<uint8_t> a((size_t)M * ap), al((size_t)M * lp), w((size_t)N * K);
        for (auto& v : a) v = rad_f32_to_fp8e4m3(nd(rng));
        for (int r = 0; r < M; ++r) std::memcpy(&al[(size_t)r * lp], &a[(size_t)r * ap], K);
        for (auto& v : w) v = rad_f32_to_fp8e4m3(nd(rng) * 0.5f);
        std::vector<float> as((size_t)M * kb), ws((size_t)(N / 128) * kb);
        // Scales over twelve octaves: K blocks of very different weight cancel, so a changed
        // summation order shows through the bf16 output.
        std::uniform_real_distribution<float> oct(-6.f, 6.f);
        for (auto& v : as) v = std::exp2(oct(rng));
        for (auto& v : ws) v = std::exp2(oct(rng)) * 0.01f;
        RadTensor t[7] = { mat(a.data(), RAD_F8E4M3, M, K, ap),
                           mat(as.data(), RAD_F32, M, kb, kb),
                           mat(w.data(), RAD_F8E4M3, N, K, K),
                           mat(ws.data(), RAD_F32, N / 128, kb, kb),
                           RadTensor{},
                           RadTensor{},
                           RadTensor{} };
        RadParam p[5]{};
        p[0] = { "M", RAD_P_INT, M };
        p[1] = { "N", RAD_P_INT, N };
        p[2] = { "K", RAD_P_INT, K };
        p[3] = { "group", RAD_P_INT, 128 };
        p[4].key = "dtype";
        p[4].kind = RAD_P_STR;
        p[4].sval = "fp8a8f32";
        // The pitch off 16 bytes takes the legacy tile.
        auto run = [&](bool legacy, std::vector<uint16_t>& y) {
            y.assign((size_t)M * yp, 0x7777);
            t[0] = legacy ? mat(al.data(), RAD_F8E4M3, M, K, lp)
                          : mat(a.data(), RAD_F8E4M3, M, K, ap);
            t[4] = mat(y.data(), RAD_BF16, M, N, yp);
            RadArgs ar{};
            ar.t = t;
            ar.n_t = 7;
            ar.p = p;
            ar.n_p = 5;
            return halo_rad_fp8_f32scale(&ar, nullptr);
        };
        std::vector<uint16_t> legacy, got;
        ++cases;
        if (run(true, legacy) != RAD_OK) {
            std::printf("FAIL legacy M=%d N=%d K=%d ran %s\n", M, N, K, emu_last_kernel);
            ++fails;
            continue;
        }
        int nonzero = 0;
        for (int r = 0; r < M; ++r)
            for (int c = 0; c < N; ++c) nonzero += legacy[(size_t)r * yp + c] != 0;
        if (nonzero < M * N / 2) {
            std::printf("FAIL legacy output mostly zero at M=%d\n", M);
            ++fails;
        }
        /* libhalo's two runs are one kernel at two pitches, so the comparison below checks the
         * pitch, not the arithmetic: the dense row's numeric anchor is this float64 restatement,
         * per 128 of K the product sum times (activation scale * weight scale), summed. */
        {
            double num = 0, den = 0;
            for (int r = 0; r < M; ++r)
                for (int n = 0; n < N; ++n) {
                    double acc = 0;
                    for (int g = 0; g < kb; ++g) {
                        double sum = 0;
                        for (int k = g * 128; k < g * 128 + 128; ++k)
                            sum += (double)rad_fp8e4m3_to_f32(a[(size_t)r * ap + k]) *
                                   (double)rad_fp8e4m3_to_f32(w[(size_t)n * K + k]);
                        acc += sum * (double)as[(size_t)r * kb + g] * (double)ws[(size_t)(n / 128) * kb + g];
                    }
                    const double d = (double)rad_bf16_to_f32(legacy[(size_t)r * yp + n]) - acc;
                    num += d * d;
                    den += acc * acc;
                }
            const double rel = std::sqrt(num / den);
            ++cases;
            if (!(rel < 1e-2)) {
                std::printf("FAIL dense M=%d N=%d K=%d against float64: rel_l2 %.3e\n", M, N, K, rel);
                ++fails;
            }
        }
        ++cases;
        const int rc = run(false, got);
        if (rc != RAD_OK || got != legacy) {
            size_t i = 0;
            while (rc == RAD_OK && i < got.size() && got[i] == legacy[i]) ++i;
            std::printf("FAIL %s (want %s) at M=%d N=%d K=%d (rc %d, first word %zu)\n",
                        emu_last_kernel, sh.kernel, M, N, K, rc, i);
            ++fails;
        }
    }
    fails += grouped_cases(cases);
    {
        /* A run of rows and a slice of columns give the bytes the whole product has there: GLM's
         * MLA split by rows computes q_b and o over a rank's rows at full width, the head form
         * over every row at a slice of the columns (or of K, as eight rank partials). */
        const int M = 130, N = 1024, K = 512, kb = K / 128, r0 = 37, mr = 33, c0 = 256, nc = 256;
        std::vector<uint8_t> a((size_t)M * K), w((size_t)N * K);
        for (auto& v : a) v = rad_f32_to_fp8e4m3(nd(rng));
        for (auto& v : w) v = rad_f32_to_fp8e4m3(nd(rng) * 0.5f);
        std::vector<float> as((size_t)M * kb), ws((size_t)(N / 128) * kb);
        std::uniform_real_distribution<float> oct(-6.f, 6.f);
        for (auto& v : as) v = std::exp2(oct(rng));
        for (auto& v : ws) v = std::exp2(oct(rng)) * 0.01f;
        auto gemm = [&](int rows, int cols, int row0, int col0, std::vector<uint16_t>& y) {
            y.assign((size_t)rows * cols, 0x7777);
            RadTensor t[7] = { mat(a.data() + (size_t)row0 * K, RAD_F8E4M3, rows, K, K),
                               mat(as.data() + (size_t)row0 * kb, RAD_F32, rows, kb, kb),
                               mat(w.data() + (size_t)col0 * K, RAD_F8E4M3, cols, K, K),
                               mat(ws.data() + (size_t)(col0 / 128) * kb, RAD_F32, cols / 128, kb,
                                   kb),
                               mat(y.data(), RAD_BF16, rows, cols, cols), RadTensor{}, RadTensor{} };
            RadParam p[5]{};
            p[0] = { "M", RAD_P_INT, rows };
            p[1] = { "N", RAD_P_INT, cols };
            p[2] = { "K", RAD_P_INT, K };
            p[3] = { "group", RAD_P_INT, 128 };
            p[4].key = "dtype";
            p[4].kind = RAD_P_STR;
            p[4].sval = "fp8a8f32";
            RadArgs ar{};
            ar.t = t;
            ar.n_t = 7;
            ar.p = p;
            ar.n_p = 5;
            return halo_rad_fp8_f32scale(&ar, nullptr);
        };
        std::vector<uint16_t> whole, part;
        ++cases;
        bool same = gemm(M, N, 0, 0, whole) == RAD_OK && gemm(mr, nc, r0, c0, part) == RAD_OK;
        for (int r = 0; same && r < mr; ++r)
            for (int c = 0; c < nc; ++c)
                same = same && part[(size_t)r * nc + c] == whole[(size_t)(r0 + r) * N + c0 + c];
        std::printf("%s rows %d..%d x columns %d..%d alone against the %dx%d product\n",
                    same ? "ok  " : "FAIL", r0, r0 + mr, c0, c0 + nc, M, N);
        fails += !same;
    }
    std::printf("%d case(s), %d failure(s)\n", cases, fails);
    return fails ? 1 : 0;
}
