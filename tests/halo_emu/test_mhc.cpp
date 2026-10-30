/* tests/halo_emu (from tests/r4d_emu/test_mhc.cpp): GLM's mHC read split by residual chunks (halo_mhc.hip) on the host SIMT emulator.
 * Each rank's mhc_pre_partial over its chunks, the partials joined row by row as the
 * row-interleaved all_gather lays them out, then mhc_pre_finish, must give mhc_pre's x, post and
 * comb bit for bit; the partial matrix's padding slots are zero. On 16-byte rows the vector forms
 * run, and they are held bit for bit to the same operands on pitched rows, where the scalar forms
 * run. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "halo_plugin.h"
#include "ref_host.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

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
static RadTensor vec(void* p, int64_t n) {
    RadTensor t{};
    t.data = p;
    t.dtype = RAD_F32;
    t.rank = 1;
    t.shape[0] = n;
    t.stride[0] = 1;
    return t;
}

int main() {
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int64_t n = 4096, hc = 4, W = hc * n, mix = hc * (hc + 2), S = 28, total = 8;
    std::vector<float> fn((size_t)(mix * W)), scale = { 0.7f, 0.9f, 1.1f }, base((size_t)mix);
    for (auto& v : fn) v = nd(rng) * 0.01f;
    for (auto& v : base) v = nd(rng) * 0.5f;
    int fails = 0, cases = 0;
    for (int64_t rows : { 1, 5, 37 }) {
        const int64_t hp = W + 6, xp = n + 2;
        std::vector<uint16_t> h((size_t)(rows * hp));
        for (auto& v : h) v = f2b(nd(rng));
        RadParam coef[8]{};
        coef[0] = { "M", RAD_P_INT, rows };
        coef[1] = { "n", RAD_P_INT, n };
        coef[2] = { "hc", RAD_P_INT, hc };
        coef[3] = { "iterations", RAD_P_INT, 20 };
        coef[4].key = "eps";
        coef[4].kind = RAD_P_F64;
        coef[4].dval = 1e-5;
        coef[5].key = "hc_eps";
        coef[5].kind = RAD_P_F64;
        coef[5].dval = 1e-6;
        coef[6].key = "dtype";
        coef[6].kind = RAD_P_STR;
        coef[6].sval = "bf16";
        std::vector<uint16_t> x((size_t)(rows * xp), 0x7777);
        std::vector<float> post((size_t)(rows * hc)), comb((size_t)(rows * hc * hc));
        RadTensor t[7] = { mat(h.data(), RAD_BF16, rows, W, hp),
                           mat(fn.data(), RAD_F32, mix, W, W),
                           vec(scale.data(), 3),
                           vec(base.data(), mix),
                           mat(x.data(), RAD_BF16, rows, n, xp),
                           mat(post.data(), RAD_F32, rows, hc, hc),
                           mat(comb.data(), RAD_F32, rows, hc * hc, hc * hc) };
        RadArgs a{};
        a.t = t;
        a.n_t = 7;
        a.p = coef;
        a.n_p = 7;
        if (halo_rad_mhc_pre(&a, nullptr) != RAD_OK) {
            std::printf("FAIL mhc_pre M=%lld\n", (long long)rows);
            ++fails;
            continue;
        }
        for (int world : { 2, 4, 8 }) {
            ++cases;
            const int64_t mine = total / world;
            std::vector<float> gathered((size_t)(rows * total * S));
            bool padded = true, launched = true;
            for (int rank = 0; rank < world; ++rank) {
                std::vector<float> part((size_t)(rows * mine * S), 7.0f);
                RadTensor pt[3] = { t[0], t[1],
                                    mat(part.data(), RAD_F32, rows, mine * S, mine * S) };
                RadParam pp[7]{};
                pp[0] = { "M", RAD_P_INT, rows };
                pp[1] = { "n", RAD_P_INT, n };
                pp[2] = { "hc", RAD_P_INT, hc };
                pp[3] = { "chunk", RAD_P_INT, 2048 };
                pp[4] = { "first", RAD_P_INT, rank * mine };
                pp[5] = { "chunks", RAD_P_INT, mine };
                pp[6].key = "dtype";
                pp[6].kind = RAD_P_STR;
                pp[6].sval = "bf16";
                RadArgs pa{};
                pa.t = pt;
                pa.n_t = 3;
                pa.p = pp;
                pa.n_p = 7;
                launched = launched && halo_rad_mhc_pre_partial(&pa, nullptr) == RAD_OK;
                for (int64_t r = 0; r < rows; ++r) {
                    for (int64_t c = 0; c < mine; ++c)
                        for (int64_t j = mix + 1; j < S; ++j)
                            padded = padded && part[(size_t)(r * mine * S + c * S + j)] == 0.0f;
                    std::memcpy(&gathered[(size_t)(r * total * S + rank * mine * S)],
                                &part[(size_t)(r * mine * S)], (size_t)(mine * S * 4));
                }
            }
            std::vector<uint16_t> x2((size_t)(rows * xp), 0x7777);
            std::vector<float> post2(post.size()), comb2(comb.size());
            RadTensor ft[7] = { mat(gathered.data(), RAD_F32, rows, total * S, total * S),
                                t[0],
                                t[2],
                                t[3],
                                mat(x2.data(), RAD_BF16, rows, n, xp),
                                mat(post2.data(), RAD_F32, rows, hc, hc),
                                mat(comb2.data(), RAD_F32, rows, hc * hc, hc * hc) };
            RadParam fp[8];
            std::memcpy(fp, coef, sizeof coef);
            fp[6] = { "chunk", RAD_P_INT, 2048 };
            fp[7].key = "dtype";
            fp[7].kind = RAD_P_STR;
            fp[7].sval = "bf16";
            RadArgs fa{};
            fa.t = ft;
            fa.n_t = 7;
            fa.p = fp;
            fa.n_p = 8;
            launched = launched && halo_rad_mhc_pre_finish(&fa, nullptr) == RAD_OK;
            const bool same = launched && x2 == x &&
                              std::memcmp(post2.data(), post.data(), post.size() * 4) == 0 &&
                              std::memcmp(comb2.data(), comb.data(), comb.size() * 4) == 0;
            if (!same || !padded) {
                std::printf("FAIL M=%lld world %d: %s\n", (long long)rows, world,
                            !launched ? "launch"
                            : !padded ? "padding"
                                      : "differs");
                ++fails;
            }
        }
    }
    /* The vector forms (four-slot partials, the finish's 16-byte collapse, the eight-column
     * mhc_post) on 16-byte rows against the scalar forms (one slot a thread, four rows and four
     * coefficients a block from 32 rows, the per-output mhc_post) on the same operands with the
     * residual pitched by two elements: mhc_pre's x, post and comb, mhc_pre_partial's matrix and
     * mhc_post's output, bit for bit. */
    for (int64_t rows : { 1, 5, 16, 37, 64 }) {
        std::vector<uint16_t> h((size_t)(rows * W)), y((size_t)(rows * n));
        for (auto& v : h) v = f2b(nd(rng));
        for (auto& v : y) v = f2b(nd(rng));
        const int64_t pitched = W + 2;
        std::vector<uint16_t> hp((size_t)(rows * pitched), 0x7777);
        for (int64_t r = 0; r < rows; ++r)
            std::memcpy(&hp[(size_t)(r * pitched)], &h[(size_t)(r * W)], (size_t)W * 2);
        RadParam coef[7]{};
        coef[0] = { "M", RAD_P_INT, rows };
        coef[1] = { "n", RAD_P_INT, n };
        coef[2] = { "hc", RAD_P_INT, hc };
        coef[3] = { "iterations", RAD_P_INT, 20 };
        coef[4].key = "eps";
        coef[4].kind = RAD_P_F64;
        coef[4].dval = 1e-6;
        coef[5].key = "hc_eps";
        coef[5].kind = RAD_P_F64;
        coef[5].dval = 1e-6;
        coef[6].key = "dtype";
        coef[6].kind = RAD_P_STR;
        coef[6].sval = "bf16";
        struct Out {
            std::vector<uint16_t> x, o;
            std::vector<float> p, c, part;
        };
        auto run = [&](Out* r, const uint16_t* hd, int64_t ld) {
            r->x.assign((size_t)(rows * n), 0x7777);
            r->o.assign((size_t)(rows * W), 0x7777);
            r->p.assign((size_t)(rows * hc), 0);
            r->c.assign((size_t)(rows * hc * hc), 0);
            r->part.assign((size_t)(rows * total * S), 7.0f);
            RadTensor t[7] = { mat((void*)hd, RAD_BF16, rows, W, ld),
                               mat(fn.data(), RAD_F32, mix, W, W),
                               vec(scale.data(), 3),
                               vec(base.data(), mix),
                               mat(r->x.data(), RAD_BF16, rows, n, n),
                               mat(r->p.data(), RAD_F32, rows, hc, hc),
                               mat(r->c.data(), RAD_F32, rows, hc * hc, hc * hc) };
            RadArgs a{};
            a.t = t;
            a.n_t = 7;
            a.p = coef;
            a.n_p = 7;
            bool okay = halo_rad_mhc_pre(&a, nullptr) == RAD_OK;
            RadTensor pt[3] = { t[0], t[1],
                                mat(r->part.data(), RAD_F32, rows, total * S, total * S) };
            RadParam pp[7]{};
            pp[0] = { "M", RAD_P_INT, rows };
            pp[1] = { "n", RAD_P_INT, n };
            pp[2] = { "hc", RAD_P_INT, hc };
            pp[3] = { "chunk", RAD_P_INT, 2048 };
            pp[4] = { "first", RAD_P_INT, 0 };
            pp[5] = { "chunks", RAD_P_INT, total };
            pp[6].key = "dtype";
            pp[6].kind = RAD_P_STR;
            pp[6].sval = "bf16";
            RadArgs pa{};
            pa.t = pt;
            pa.n_t = 3;
            pa.p = pp;
            pa.n_p = 7;
            okay = okay && halo_rad_mhc_pre_partial(&pa, nullptr) == RAD_OK;
            RadTensor ot[5] = { mat(y.data(), RAD_BF16, rows, n, n), t[0], t[5], t[6],
                                mat(r->o.data(), RAD_BF16, rows, W, W) };
            RadArgs oa{};
            oa.t = ot;
            oa.n_t = 5;
            oa.p = coef;
            oa.n_p = 7;
            return okay && halo_rad_mhc_post(&oa, nullptr) == RAD_OK;
        };
        Out ref, got;
        ++cases;
        const bool launched = run(&ref, hp.data(), pitched) && run(&got, h.data(), W);
        const char* bad = !launched                                                   ? "launch"
                          : got.x != ref.x                                            ? "x"
                          : std::memcmp(got.p.data(), ref.p.data(), ref.p.size() * 4) ? "post"
                          : std::memcmp(got.c.data(), ref.c.data(), ref.c.size() * 4) ? "comb"
                          : std::memcmp(got.part.data(), ref.part.data(), ref.part.size() * 4)
                              ? "partials"
                          : got.o != ref.o ? "mhc_post"
                                           : nullptr;
        if (bad) {
            std::printf("FAIL M=%lld 16-byte rows against pitched rows: %s differs\n",
                        (long long)rows, bad);
            ++fails;
        }
    }
    std::printf("%d case(s), %d failure(s)\n", cases, fails);
    return fails ? 1 : 0;
}
