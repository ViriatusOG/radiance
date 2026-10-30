/* tests/halo_emu (from tests/r4d_emu/test_dsa.cpp): GLM's DSA scoring and selection split over the tensor-parallel ranks by query
 * rows (halo_dsa.hip, then halo_qsa_select_bf16.hip) on the host SIMT emulator.
 *
 * Each rank scores its run of R = 4 * ceil(M / (4 * world)) rows with `row_part`, selects over
 * the rows of that run that exist with the rebased bounds the scorer wrote (`cu_part`), and the
 * runs are laid end to end as the all_gather concatenates them. The joined table, nsel and
 * seqused must equal the unsplit scorer and selection byte for byte on every row of the step; a
 * run's scores must equal the unsplit scores' rows, and its rows past the step score -inf. The
 * fixtures mix decode rows with a prefill chunk, a missing page, duplicated keys (exact ties) and
 * signed gates, at world sizes where every rank has rows and where the last ranks have none. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "halo_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kHeads = 32, kN = 128, kRatio = 4, kPitch = 136;

RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape) {
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        t.shape[i] = shape[i];
        t.stride[i] = st;
        st *= shape[i];
    }
    return t;
}

struct Fix {
    int S = 0, M = 0, nb = 0, pages = 0, topk = 0;
    std::vector<int32_t> cu, nc, pos, bt;
    std::vector<uint8_t> q, cache;   /* [M*32, 128] codes; [pages, 1, 4, 34] packed keys */
    std::vector<float> qs, gate;     /* [M*32, 1], [M, 32] */
};

/* `decode` one-row sequences, then one prefill chunk of `chunk` rows at `start`. */
Fix make(uint32_t seed, int decode, int chunk, int start, int topk) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255), coin(0, 99);
    std::normal_distribution<float> nd(0.f, 1.f);
    Fix f;
    f.S = decode + 1;
    f.M = decode + chunk;
    f.topk = topk;
    f.cu.push_back(0);
    for (int s = 0; s < decode; ++s) {
        const int p = 40 + 37 * s + byte(rng) % 50;
        f.pos.push_back(p);
        f.cu.push_back(f.cu.back() + 1);
        f.nc.push_back((p + 1) / kRatio);
    }
    for (int i = 0; i < chunk; ++i) f.pos.push_back(start + i);
    f.cu.push_back(f.M);
    f.nc.push_back((start + chunk) / kRatio);
    for (int c : f.nc) f.nb = std::max(f.nb, c + 2);
    f.pages = f.S * f.nb + 3;
    f.bt.assign((size_t)f.S * f.nb, -1);
    for (int s = 0; s < f.S; ++s)
        for (int b = 0; b <= f.nc[s] && b < f.nb; ++b)
            f.bt[(size_t)s * f.nb + b] = (s * f.nb + b * 7) % f.pages;
    f.bt[(size_t)(f.S - 1) * f.nb + 3] = -1;
    auto code = [&] {
        int v;
        do v = byte(rng);
        while ((v & 0x7f) == 0x7f);
        return (uint8_t)v;
    };
    f.cache.assign((size_t)f.pages * kPitch, 0);
    for (int p = 0; p < f.pages; ++p) {
        uint8_t* page = &f.cache[(size_t)p * kPitch];
        /* Every fifth page repeats its predecessor's key: equal scores on one row. */
        if (p % 5 == 4)
            std::memcpy(page, page - kPitch, kPitch);
        else {
            for (int d = 0; d < kN; ++d) page[d] = code();
            const float scale = std::ldexp(1.0f, -6 - p % 3);
            std::memcpy(page + kN, &scale, 4);
        }
    }
    f.q.resize((size_t)f.M * kHeads * kN);
    for (auto& v : f.q) v = code();
    f.qs.resize((size_t)f.M * kHeads);
    for (size_t i = 0; i < f.qs.size(); ++i) f.qs[i] = std::ldexp(1.0f, -5 - (int)(i % 3));
    f.gate.resize((size_t)f.M * kHeads);
    for (auto& v : f.gate) v = nd(rng) * (coin(rng) < 10 ? 0.0f : 1.0f);
    return f;
}

RadParam pint(const char* k, long long v) { return { k, RAD_P_INT, v, 0, nullptr }; }

/* Scores `rows` score rows from step row `part * rows` (part < 0: the whole step, no row_part). */
int score(const Fix& f, const uint8_t* q, const float* qs, const float* gate, int64_t qrows,
          int part, int rows, float* out, int32_t* cu_part) {
    RadTensor t[10] = {
        tens((void*)q, RAD_F8E4M3, { qrows * kHeads, kN }),
        tens((void*)qs, RAD_F32, { qrows * kHeads, 1 }),
        tens((void*)gate, RAD_F32, { qrows, kHeads }),
        tens((void*)f.cache.data(), RAD_U8, { f.pages, 1, 4, 34 }),
        RadTensor{},
        tens((void*)f.bt.data(), RAD_I32, { f.S, f.nb }),
        tens((void*)f.cu.data(), RAD_I32, { f.S + 1 }),
        tens((void*)f.nc.data(), RAD_I32, { f.S }),
        tens(out, RAD_F32, { rows, f.nb }),
        cu_part ? tens(cu_part, RAD_I32, { f.S + 1 }) : RadTensor{},
    };
    t[3].stride[0] = kPitch;
    RadParam p[7] = { pint("M", rows), pint("n", kN), pint("heads", kHeads), pint("blocks", f.nb),
                      pint("seqs", f.S), { "dtype", RAD_P_STR, 0, 0, "fp8_e4m3" },
                      pint("row_part", part) };
    RadArgs a{};
    a.t = t;
    a.n_t = 10;
    a.p = p;
    a.n_p = part < 0 ? 6 : 7;
    return halo_rad_dsa_score(&a, nullptr);
}

int select(const Fix& f, const float* sc, int rows, const int32_t* pos, const int32_t* cu,
           int32_t* sel, int32_t* nsel, int32_t* seqused) {
    RadTensor t[8] = {
        tens((void*)sc, RAD_F32, { rows, f.nb }),
        tens((void*)f.bt.data(), RAD_I32, { f.S, f.nb }),
        tens((void*)f.nc.data(), RAD_I32, { f.S }),
        tens((void*)pos, RAD_I32, { rows }),
        tens((void*)cu, RAD_I32, { f.S + 1 }),
        tens(sel, RAD_I32, { rows, f.topk + 1 }),
        tens(nsel, RAD_I32, { rows }),
        tens(seqused, RAD_I32, { rows }),
    };
    RadParam p[6] = { pint("M", rows),          pint("blocks", f.nb), pint("topk", f.topk),
                      pint("ratio", kRatio),    pint("seqs", f.S),    pint("signed", 1) };
    RadArgs a{};
    a.t = t;
    a.n_t = 8;
    a.p = p;
    a.n_p = 6;
    const int64_t need = halo_rad_qsa_select_scratch(&a);
    std::vector<uint32_t> scratch((size_t)(need / 4 + 1), 0xA5A5A5A5u);
    a.scratch = scratch.data();
    a.scratch_bytes = need;
    return halo_rad_qsa_select(&a, nullptr);
}

int check(const char* name, const Fix& f) {
    const int M = f.M, K = f.topk + 1;
    std::vector<float> want_sc((size_t)M * f.nb);
    std::vector<int32_t> want_sel((size_t)M * K, 0x55555555), want_n(M), want_su(M);
    if (score(f, f.q.data(), f.qs.data(), f.gate.data(), M, -1, M, want_sc.data(), nullptr) ||
        select(f, want_sc.data(), M, f.pos.data(), f.cu.data(), want_sel.data(), want_n.data(),
               want_su.data())) {
        std::printf("FAIL %s: the unsplit launch\n", name);
        return 1;
    }
    int fails = 0;
    for (int world : { 2, 4, 8 }) {
        const int R = 4 * ((M + 4 * world - 1) / (4 * world));
        const int64_t padded = (int64_t)world * R;
        /* The step's query rows, then rows past the step that a last run reads no further than. */
        std::vector<uint8_t> q((size_t)padded * kHeads * kN, 0x38);
        std::vector<float> qs((size_t)padded * kHeads, 1.0f), gate((size_t)padded * kHeads, 3.0f);
        std::memcpy(q.data(), f.q.data(), f.q.size());
        std::memcpy(qs.data(), f.qs.data(), f.qs.size() * 4);
        std::memcpy(gate.data(), f.gate.data(), f.gate.size() * 4);
        std::vector<int32_t> sel((size_t)padded * K, 0x55555555), ns(padded, 0x55555555),
            su(padded, 0x55555555);
        bool ok = true;
        const char* why = "";
        for (int part = 0; part < world && ok; ++part) {
            const int r0 = part * R, have = std::min(R, M - r0);
            std::vector<float> sc((size_t)R * f.nb, 7.0f);
            std::vector<int32_t> cu_part(f.S + 1, 0x55555555);
            if (score(f, q.data(), qs.data(), gate.data(), padded, part, R, sc.data(),
                      cu_part.data())) {
                ok = false, why = "score launch";
                break;
            }
            for (int s = 0; s <= f.S; ++s) ok = ok && cu_part[s] == f.cu[s] - r0;
            if (!ok) {
                why = "cu_part";
                break;
            }
            for (int i = 0; i < R && ok; ++i)
                for (int b = 0; b < f.nb && ok; ++b) {
                    const float v = sc[(size_t)i * f.nb + b];
                    ok = i < have ? std::memcmp(&v, &want_sc[(size_t)(r0 + i) * f.nb + b], 4) == 0
                                  : v == -INFINITY;
                }
            if (!ok) {
                why = "scores";
                break;
            }
            if (have <= 0) continue;
            if (select(f, sc.data(), have, f.pos.data() + r0, cu_part.data(), &sel[(size_t)r0 * K],
                       &ns[r0], &su[r0])) {
                ok = false, why = "select launch";
                break;
            }
        }
        if (ok) {
            ok = std::memcmp(sel.data(), want_sel.data(), want_sel.size() * 4) == 0;
            why = "selection";
        }
        if (ok) {
            ok = std::memcmp(ns.data(), want_n.data(), (size_t)M * 4) == 0 &&
                 std::memcmp(su.data(), want_su.data(), (size_t)M * 4) == 0;
            why = "nsel or seqused";
        }
        if (!ok) {
            std::printf("FAIL %s world %d (R %d): %s differs\n", name, world, R, why);
            ++fails;
        }
    }
    return fails;
}

}  // namespace

int main() {
    int fails = 0, cases = 0;
    struct Case {
        const char* name;
        uint32_t seed;
        int decode, chunk, start, topk;
    } cases_[] = {
        { "prefill chunk alone", 1, 0, 64, 300, 16 },
        { "decode rows beside a chunk", 2, 6, 61, 517, 32 },
        { "chunk from position 0, short history", 3, 2, 37, 0, 8 },
        { "rows below topk blocks", 4, 3, 29, 20, 64 },
    };
    for (const auto& c : cases_) {
        fails += check(c.name, make(c.seed, c.decode, c.chunk, c.start, c.topk));
        ++cases;
    }
    std::printf("dsa split: %d fixture(s), %d failure(s)\n", cases, fails);
    return fails ? 1 : 0;
}
