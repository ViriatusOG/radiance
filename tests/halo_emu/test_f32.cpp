/* tests/halo_emu (from tests/r4d_emu/test_f32.cpp): gemm_nt_f32 (halo_glm.hip) on the host SIMT emulator. The form the launcher picks
 * at each shape (a tile of rows x columns a wave with its steps of k in flight, by M and N; the
 * shapes reach every one, with K past and short of whole groups of steps) must match the
 * arithmetic written out here bit for bit: lane l folds k = l, l+32, ... with fmaf, then the xor
 * butterfly over 16, 8, 4, 2, 1. Tail rows and columns and the pitch's padding stay untouched.
 *
 * router_sigmoid_topk from the same unit: the wave-per-row select must match a serial scan written
 * out here bit for bit, on random and on tied ranks, with the outputs' pitch padding untouched. */
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

static float gemm_model(const uint16_t* x, const float* w, int k) {
    float lane[32];
    for (int l = 0; l < 32; ++l) {
        float s = 0;
        for (int j = l; j < k; j += 32) s = std::fmaf(b2f(x[j]), w[j], s);
        lane[l] = s;
    }
    for (int mask = 16; mask; mask >>= 1) {
        float next[32];
        for (int l = 0; l < 32; ++l) next[l] = lane[l] + lane[l ^ mask];
        std::memcpy(lane, next, sizeof lane);
    }
    return lane[0];
}

static void router_model(const float* lg, const float* bias, int bp, int ne, int k, int norm,
                         float factor, int* ids, float* w) {
    float prob[512], rank[512], chosen[32], total = 0;
    int index[32];
    for (int e = 0; e < ne; ++e) {
        prob[e] = 1.0f / (1.0f + expf(-lg[e]));
        rank[e] = prob[e] + bias[(size_t)e * bp];
    }
    for (int j = 0; j < k; ++j) {
        int best = 0;
        for (int e = 1; e < ne; ++e)
            if (rank[e] > rank[best]) best = e;
        index[j] = best;
        chosen[j] = prob[best];
        total += chosen[j];
        rank[best] = -INFINITY;
    }
    const float mult = norm ? factor / (total + 1e-20f) : factor;
    for (int j = 0; j < k; ++j) {
        ids[j] = index[j];
        w[j] = chosen[j] * mult;
    }
}

static void router_cases(std::mt19937& rng, int& cases, int& fails) {
    struct Shape {
        int M, ne, k, norm;
        bool ties;
    };
    const Shape shapes[] = { { 1, 288, 8, 1, false }, { 6, 288, 8, 1, false },
                             { 20, 288, 8, 1, true }, { 48, 288, 8, 1, false },
                             { 6, 288, 8, 1, true },  { 5, 8, 8, 0, true },
                             { 5, 32, 32, 0, true },  { 7, 300, 8, 1, false },
                             { 3, 512, 32, 1, true } };
    std::uniform_int_distribution<int> ud(-1000, 1000);
    for (const Shape& sh : shapes) {
        const int M = sh.M, ne = sh.ne, k = sh.k, lp = ne + 5, bp = 2, ip = k + 3, wp = k + 1;
        const float factor = 2.5f;
        std::vector<float> lg((size_t)M * lp), bias((size_t)ne * bp);
        for (auto& v : lg) v = sh.ties ? (float)(ud(rng) % 8) * 0.5f : (float)ud(rng) / 200;
        for (auto& v : bias) v = sh.ties ? (float)(ud(rng) % 5) * 0.125f : (float)ud(rng) / 2000;
        RadTensor t[4]{};
        t[0].data = lg.data();
        t[0].dtype = RAD_F32;
        t[0].rank = 2;
        t[0].shape[0] = M;
        t[0].shape[1] = ne;
        t[0].stride[0] = lp;
        t[0].stride[1] = 1;
        t[1].data = bias.data();
        t[1].dtype = RAD_F32;
        t[1].rank = 1;
        t[1].shape[0] = ne;
        t[1].stride[0] = bp;
        t[2].dtype = RAD_I32;
        t[2].rank = 2;
        t[2].shape[0] = M;
        t[2].shape[1] = k;
        t[2].stride[0] = ip;
        t[2].stride[1] = 1;
        t[3].dtype = RAD_F32;
        t[3].rank = 2;
        t[3].shape[0] = M;
        t[3].shape[1] = k;
        t[3].stride[0] = wp;
        t[3].stride[1] = 1;
        RadParam p[6]{};
        p[0] = { "M", RAD_P_INT, M };
        p[1] = { "n_expert", RAD_P_INT, ne };
        p[2] = { "top_k", RAD_P_INT, k };
        p[3] = { "norm", RAD_P_INT, sh.norm };
        p[4].key = "route_scale";
        p[4].kind = RAD_P_F64;
        p[4].dval = factor;
        p[5].key = "dtype";
        p[5].kind = RAD_P_STR;
        p[5].sval = "f32";
        RadArgs a{};
        a.t = t;
        a.n_t = 4;
        a.p = p;
        a.n_p = 6;
        std::vector<int> ids((size_t)M * ip, -77);
        std::vector<float> w((size_t)M * wp, -7.25f);
        t[2].data = ids.data();
        t[3].data = w.data();
        ++cases;
        const int rc = halo_rad_router_sigmoid_topk(&a, nullptr);
        int bad = 0;
        for (int r = 0; r < M && rc == RAD_OK; ++r) {
            int mi[32];
            float mw[32];
            router_model(&lg[(size_t)r * lp], bias.data(), bp, ne, k, sh.norm, factor, mi, mw);
            bad += std::memcmp(mi, &ids[(size_t)r * ip], (size_t)k * 4) != 0;
            bad += std::memcmp(mw, &w[(size_t)r * wp], (size_t)k * 4) != 0;
            for (int c = k; c < ip; ++c) bad += ids[(size_t)r * ip + c] != -77;
            for (int c = k; c < wp; ++c) bad += w[(size_t)r * wp + c] != -7.25f;
        }
        if (rc != RAD_OK || bad) {
            std::printf("FAIL router wave select against the scan at M=%d ne=%d k=%d%s (rc %d): "
                        "%d\n",
                        M, ne, k, sh.ties ? " tied" : "", rc, bad);
            ++fails;
        }
    }
}

int main() {
    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.f, 1.f);
    struct Shape {
        int M, N, K;
    };
    // One wave an output (M 1), the 2x1, 4x1 (M under 16 or N up to 8), 4x4 and 8x4 tiles.
    const Shape shapes[] = { { 1, 288, 512 },  { 2, 288, 129 },  { 3, 36, 129 },
                             { 8, 288, 129 },  { 33, 32, 512 },  { 70, 36, 129 },
                             { 70, 4, 512 },   { 7, 31, 129 },   { 65, 13, 130 },
                             { 16, 288, 1056 }, { 24, 288, 1040 }, { 40, 36, 1024 } };
    int fails = 0, cases = 0;
    for (const Shape& sh : shapes) {
        const int M = sh.M, N = sh.N, K = sh.K, xp = K + 5, yp = N + 3;
        std::vector<uint16_t> x((size_t)M * xp);
        std::vector<float> w((size_t)N * K);
        for (auto& v : x) v = f2b(nd(rng));
        for (auto& v : w) v = nd(rng) * 0.03f;
        RadTensor t[3]{};
        t[0].data = x.data();
        t[0].dtype = RAD_BF16;
        t[0].rank = 2;
        t[0].shape[0] = M;
        t[0].shape[1] = K;
        t[0].stride[0] = xp;
        t[0].stride[1] = 1;
        t[1].data = w.data();
        t[1].dtype = RAD_F32;
        t[1].rank = 2;
        t[1].shape[0] = N;
        t[1].shape[1] = K;
        t[1].stride[0] = K;
        t[1].stride[1] = 1;
        t[2].dtype = RAD_F32;
        t[2].rank = 2;
        t[2].shape[0] = M;
        t[2].shape[1] = N;
        t[2].stride[0] = yp;
        t[2].stride[1] = 1;
        RadParam p[4]{};
        p[0] = { "M", RAD_P_INT, M };
        p[1] = { "N", RAD_P_INT, N };
        p[2] = { "K", RAD_P_INT, K };
        p[3].key = "dtype";
        p[3].kind = RAD_P_STR;
        p[3].sval = "bf16";
        RadArgs a{};
        a.t = t;
        a.n_t = 3;
        a.p = p;
        a.n_p = 4;
        std::vector<float> y((size_t)M * yp, -7.25f);
        t[2].data = y.data();
        ++cases;
        const int rc = halo_rad_gemm_nt_f32(&a, nullptr);
        int bad = 0;
        for (int r = 0; r < M && rc == RAD_OK; ++r) {
            for (int c = 0; c < N; ++c) {
                const float e = gemm_model(&x[(size_t)r * xp], &w[(size_t)c * K], K);
                bad += std::memcmp(&e, &y[(size_t)r * yp + c], 4) != 0;
            }
            for (int c = N; c < yp; ++c) bad += y[(size_t)r * yp + c] != -7.25f;
        }
        if (rc != RAD_OK || bad) {
            std::printf("FAIL %s at M=%d N=%d K=%d against its model (rc %d): %d words\n",
                        emu_last_kernel, M, N, K, rc, bad);
            ++fails;
        }
    }
    router_cases(rng, cases, fails);
    std::printf("%d case(s), %d failure(s)\n", cases, fails);
    return fails ? 1 : 0;
}
