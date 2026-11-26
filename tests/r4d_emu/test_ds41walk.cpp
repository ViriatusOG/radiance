/* tests/r4d_emu: DeepSeek-V4.1's DSpark Markov walk over the whole vocabulary (the `dspark-walk` setting), on the
 * host SIMT emulator.
 *
 * dspark_walk_step (libr4d/r4d_ds41_walk.hip) runs one walk step on one rank's vocabulary shard; deepseek41
 * issues it `steps` + 1 times with an all_gather of the per-rank keys between steps. This test runs that
 * sequence for W = 1 .. 4 ranks (the shards as deepseek41 splits the vocabulary, shard_begin / shard_extent;
 * the gather as all_gather moves the bytes: rank w's key of sequence s at [s][w]) and holds every rank's
 * tokens to a plain host walk byte for byte:
 *
 *   prev = anchor; for l: u = pred[prev] (zeros for an id outside the vocabulary);
 *     tok[l] = the first v (lowest id) with the greatest logits[l][v] + fma-chain_r(u[r], succ[v][r]),
 *              NaN scores skipped, no score at all -> 0;  prev = tok[l]
 *
 * Cases: random tables and logits (bf16-valued logits, so equal logits are common) at ranks 32 / 64 / 256, one
 * to eleven sequences (two passes of eight), anchors in and out of the vocabulary; forced ties -- the same score
 * at two ids in different shards and in one shard, -0 against +0, an out-of-vocabulary predecessor (scores =
 * logits); NaN and -inf logits, a row of NaN only. Each case also checks the tokens' padding columns are left
 * alone. With RADIANCE_DS41_WALK_TABLES=<dir> (markov_w1.bf16 / markov_w2.bf16: the checkpoint's
 * mtp.2.markov_head embed / head tables, [129280, 256] bf16 each, extracted from the .rad) the real tables run
 * at the real vocabulary over four ranks, random seeds, plus a forced cross-shard tie on them, and the test
 * prints where in the head's own order each walked token sat (how often the walk leaves the head's top 32).
 *
 * Every case runs both forms of the kernel -- form 0 (a thread reads its own succ row) and form 1 (the block's rows
 * staged through LDS in 64-element chunks as whole lines) -- and requires every rank's key of every step (the
 * score's bits) to be the same in both, besides the tokens matching the plain walk.
 *
 * THE GATHER: libr4d's all-gather moves whole 16-byte words a row, so an 8-byte key a row is refused at launch
 * (ar_gather_4rank: "refused op 'all_gather' ... numel=16 dtype=f32 row=2"). The keys travel in 16-byte slots, and
 * every case above runs the planes at deepseek41's geometry (ds41_walk_geom.h). The last section builds, from that
 * same header, the all_gather operands deepseek41 issues between steps -- every world 2 / 3 / 4 / 8, sequences a
 * pass 1 .. 16, every sequence count S and step, at a 256-byte-aligned plane as the arena places it -- and holds
 * each to libr4d's own launch-time admission of the gather rows (r4d_ar_gather_admit.h, which ar_gather_wide_setup
 * calls); the 8-byte geometry must be refused there. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include "r4d_ar_gather_admit.h"
#include "../../arch/deepseek41/ds41_walk_geom.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

int64_t per_rank(int64_t total, int world) {   /* arch/common/rad_arch.h vocab_per_rank */
    if (total % world == 0) return total / world;
    const int64_t per = (total + world - 1) / world;
    return (per + 127) / 128 * 128;
}
int64_t sh_begin(int64_t total, int world, int rank) {
    const int64_t b = per_rank(total, world) * rank;
    return b < total ? b : total;
}
int64_t sh_extent(int64_t total, int world, int rank) {
    const int64_t b = sh_begin(total, world, rank), e = std::min(total, b + per_rank(total, world));
    return e > b ? e - b : 0;
}

struct Walk {
    int V = 0, R = 0, S = 0, steps = 5, astride = 5;
    const uint16_t* w1 = nullptr;          /* pred [V, R] */
    const uint16_t* w2 = nullptr;          /* succ [V, R] */
    std::vector<float> logits;             /* [S * steps, V] */
    std::vector<int32_t> anchor;           /* [(S - 1) * astride + 1] */
};

/* The plain walk: every id, in order. */
std::vector<int32_t> ref_walk(const Walk& w) {
    std::vector<int32_t> out((size_t)w.S * w.steps);
    std::vector<float> u((size_t)w.R);
    for (int s = 0; s < w.S; ++s) {
        int32_t prev = w.anchor[(size_t)s * w.astride];
        for (int l = 0; l < w.steps; ++l) {
            const bool ok = prev >= 0 && prev < w.V;
            for (int r = 0; r < w.R; ++r) u[(size_t)r] = ok ? b2f(w.w1[(size_t)prev * w.R + r]) : 0.0f;
            const float* lg = w.logits.data() + (size_t)(s * w.steps + l) * w.V;
            bool have = false;
            float best = 0.0f;
            int32_t tok = 0;
            for (int v = 0; v < w.V; ++v) {
                const uint16_t* row = w.w2 + (size_t)v * w.R;
                float acc = 0.0f;
                for (int r = 0; r < w.R; ++r) acc = std::fma(u[(size_t)r], b2f(row[r]), acc);
                const float sc = lg[v] + acc;
                if (std::isnan(sc)) continue;
                if (!have || sc > best) { have = true; best = sc; tok = v; }
            }
            out[(size_t)s * w.steps + l] = tok;
            prev = tok;
        }
    }
    return out;
}

RadTensor mat(void* d, uint32_t dt, int64_t r, int64_t c, int64_t pitch) {
    RadTensor t{};
    t.data = d;
    t.dtype = dt;
    t.rank = 2;
    t.shape[0] = r;
    t.shape[1] = c;
    t.stride[0] = pitch;
    t.stride[1] = 1;
    return t;
}
RadTensor vec(void* d, uint32_t dt, int64_t n) {
    RadTensor t{};
    t.data = d;
    t.dtype = dt;
    t.rank = 1;
    t.shape[0] = n;
    t.stride[0] = 1;
    return t;
}
RadParam pint(const char* k, int64_t v) { RadParam p{}; p.key = k; p.kind = RAD_P_INT; p.ival = v; return p; }

/* The sharded walk over W ranks as deepseek41 issues it; every rank's tokens (pitch steps + 2, padding
 * sentinel -77) into `tok[w]`. Returns false on a refused launch. */
bool sharded_walk(const Walk& w, int W, std::vector<std::vector<int32_t>>& tok, std::string* why, int form = 0,
                  std::vector<uint64_t>* keylog = nullptr) {
    const int S = w.S, steps = w.steps, tp = steps + 2;
    std::vector<std::vector<float>> lg((size_t)W);
    std::vector<int64_t> beg((size_t)W), ext((size_t)W), lp((size_t)W);
    for (int r = 0; r < W; ++r) {
        beg[(size_t)r] = sh_begin(w.V, W, r);
        ext[(size_t)r] = sh_extent(w.V, W, r);
        lp[(size_t)r] = std::max<int64_t>(1, ext[(size_t)r] + 3);       /* a padded pitch */
        lg[(size_t)r].assign((size_t)(S * steps) * lp[(size_t)r], 1e30f);
        for (int i = 0; i < S * steps; ++i)
            for (int64_t j = 0; j < ext[(size_t)r]; ++j)
                lg[(size_t)r][(size_t)(i * lp[(size_t)r] + j)] = w.logits[(size_t)i * w.V + beg[(size_t)r] + j];
    }
    tok.assign((size_t)W, std::vector<int32_t>((size_t)S * tp, -77));
    /* 16-byte key slots (ds41_walk_geom.h): keys [S, kSlot] f32 = [S][2] u64, gath [S, W * kSlot] = [S][2 W] u64 */
    const int64_t KC = ds41walk::keys_cols(), GC = ds41walk::gath_cols(W), U = ds41walk::kSlot / 2;
    std::vector<std::vector<uint64_t>> keys((size_t)W, std::vector<uint64_t>((size_t)S * U));
    std::vector<uint64_t> gath((size_t)S * W * U, 0);
    for (int step = 0; step <= steps; ++step) {
        for (int r = 0; r < W; ++r) {
            std::fill(keys[(size_t)r].begin(), keys[(size_t)r].end(), 0ull);
            RadTensor t[7] = {
                mat(lg[(size_t)r].data(), RAD_F32, (int64_t)S * steps, ext[(size_t)r], lp[(size_t)r]),
                vec((void*)w.anchor.data(), RAD_I32, (int64_t)w.anchor.size()),
                step > 0 ? mat(gath.data(), RAD_F32, S, GC, GC) : RadTensor{},
                mat((void*)w.w1, RAD_BF16, w.V, w.R, w.R),
                mat((void*)w.w2, RAD_BF16, w.V, w.R, w.R),
                step < steps ? mat(keys[(size_t)r].data(), RAD_F32, S, KC, KC) : RadTensor{},
                mat(tok[(size_t)r].data(), RAD_I32, S, steps, tp),
            };
            RadParam p[11] = { pint("M", S), pint("step", step), pint("steps", steps), pint("rank", w.R),
                               pint("n_vocab", w.V), pint("N", ext[(size_t)r]), pint("vocab_off", beg[(size_t)r]),
                               pint("world", W), pint("anchor_stride", w.astride), RadParam{} };
            p[9].key = "dtype";
            p[9].kind = RAD_P_STR;
            p[9].sval = "f32";
            p[10] = pint("form", form);
            RadArgs a{};
            a.t = t;
            a.n_t = 7;
            a.p = p;
            a.n_p = 11;
            const int rc = r4d_rad_dspark_walk_step(&a, nullptr);
            if (rc != RAD_OK) {
                if (why) *why = "rank " + std::to_string(r) + " step " + std::to_string(step) + " rc " + std::to_string(rc);
                return false;
            }
        }
        if (step < steps && keylog)   /* every rank's keys of the step: the scores' bits, for the form comparison */
            for (int r = 0; r < W; ++r) keylog->insert(keylog->end(), keys[(size_t)r].begin(), keys[(size_t)r].end());
        if (step < steps)   /* all_gather, row-interleaved by slot: y[s][w] = rank w's x[s], whole slots */
            for (int s = 0; s < S; ++s)
                for (int r = 0; r < W; ++r)
                    for (int64_t q = 0; q < U; ++q)
                        gath[(size_t)((s * W + r) * U + q)] = keys[(size_t)r][(size_t)(s * U + q)];
    }
    return true;
}

int g_cases = 0, g_fails = 0;

/* Run one walk at every W in `worlds` against the plain walk. */
void check(const char* what, const Walk& w, std::initializer_list<int> worlds, const std::vector<int32_t>* want_in = nullptr) {
    const std::vector<int32_t> want = want_in ? *want_in : ref_walk(w);
    for (int W : worlds) {
        ++g_cases;
        std::vector<std::vector<int32_t>> tok, tok1;
        std::vector<uint64_t> k0, k1;
        std::string why, why1;
        int bad = 0, pad = 0, first = -1;
        /* form 0 (a thread reads its row) and form 1 (rows staged through LDS) -- the same keys, bit for bit */
        const bool ran0 = sharded_walk(w, W, tok, &why, 0, &k0);
        const bool ran1 = sharded_walk(w, W, tok1, &why1, 1, &k1);
        const bool ran = ran0 && ran1;
        if (!ran1 && why.empty()) why = "form 1: " + why1;
        const bool same_keys = ran && k0 == k1 && tok == tok1;
        if (ran)
            for (int r = 0; r < W; ++r)
                for (int s = 0; s < w.S; ++s) {
                    for (int l = 0; l < w.steps; ++l)
                        if (tok[(size_t)r][(size_t)s * (w.steps + 2) + l] != want[(size_t)s * w.steps + l]) {
                            if (first < 0) first = r * 1000000 + s * 100 + l;
                            ++bad;
                        }
                    for (int l = w.steps; l < w.steps + 2; ++l) pad += tok[(size_t)r][(size_t)s * (w.steps + 2) + l] != -77;
                }
        const bool ok = ran && !bad && !pad && same_keys;
        std::printf("%-58s V %6d R %3d S %2d W %d: %s", what, w.V, w.R, w.S, W,
                    !ran ? "LAUNCH REFUSED" : bad ? "DIFFERS" : pad ? "PADDING TOUCHED"
                    : !same_keys ? "FORM 1 KEYS DIFFER FROM FORM 0" : "byte-identical to the plain walk (forms 0 and 1, every key)");
        if (!ran) std::printf(" (%s)", why.c_str());
        if (bad) std::printf(" (%d token(s) of %d x %d ranks; first rank %d seq %d pos %d)", bad, w.S * w.steps, W,
                             first / 1000000, first / 100 % 10000, first % 100);
        std::printf("  tokens[0] %d %d %d %d %d\n", want[0], want[1 % want.size()], want[2 % want.size()],
                    want[3 % want.size()], want[4 % want.size()]);
        if (!ok) ++g_fails;
    }
}

/* Random tables at magnitude `ts`, logits ~ N(0, ls) rounded to bf16 (ties among equal logits). */
void fill_random(Walk& w, std::vector<uint16_t>& w1, std::vector<uint16_t>& w2, std::mt19937& rng, float ts, float ls,
                 bool bad_anchor) {
    std::normal_distribution<float> nd(0.f, 1.f);
    w1.resize((size_t)w.V * w.R);
    w2.resize((size_t)w.V * w.R);
    for (auto& x : w1) x = f2b(nd(rng) * ts);
    for (auto& x : w2) x = f2b(nd(rng) * ts);
    w.w1 = w1.data();
    w.w2 = w2.data();
    w.logits.resize((size_t)w.S * w.steps * w.V);
    for (auto& x : w.logits) x = rb(nd(rng) * ls);
    w.anchor.assign((size_t)(w.S - 1) * w.astride + 1, 0);
    std::uniform_int_distribution<int> id(0, w.V - 1);
    for (int s = 0; s < w.S; ++s) w.anchor[(size_t)s * w.astride] = id(rng);
    if (bad_anchor && w.S > 1) {
        w.anchor[0] = -1;
        w.anchor[(size_t)(w.S - 1) * w.astride] = w.V + 3;
    }
}

}  // namespace

int main(int argc, char** argv) {
    const bool quick = argc > 1 && !std::strcmp(argv[1], "--quick");
    std::mt19937 rng(4101);

    /* ---- random tables and logits */
    struct Rnd { int V, R, S; float ts, ls; bool bad; };
    const Rnd rnd[] = { { 1000, 32, 1, 0.5f, 2.f, false }, { 1537, 64, 3, 0.4f, 3.f, true },
                        { 2048, 256, 8, 0.4f, 2.5f, false }, { 3001, 256, 11, 0.4f, 2.5f, true },
                        { 777, 512, 2, 0.3f, 1.f, false },   { 4096, 96, 5, 0.4f, 0.5f, true } };
    for (const Rnd& c : rnd) {
        for (int seed = 0; seed < (quick ? 1 : 3); ++seed) {
            Walk w;
            w.V = c.V; w.R = c.R; w.S = c.S;
            std::vector<uint16_t> w1, w2;
            fill_random(w, w1, w2, rng, c.ts, c.ls, c.bad);
            check(c.bad ? "random, anchors out of the vocabulary" : "random", w, { 1, 2, 3, 4 });
        }
    }

    /* ---- forced ties */
    {
        Walk w;
        w.V = 2500; w.R = 64; w.S = 4;
        std::vector<uint16_t> w1, w2;
        fill_random(w, w1, w2, rng, 0.4f, 2.f, false);
        /* (a) ids a < b in different shards (at W 4: 625 a shard) with the same succ row and the same logit, both
         * far above the rest at every position of sequence 0: the lower id wins whatever the shard order. */
        const int a = 300, b = 2100, c = 310;
        for (int r = 0; r < w.R; ++r) { w2[(size_t)b * w.R + r] = w2[(size_t)a * w.R + r]; w2[(size_t)c * w.R + r] = w2[(size_t)a * w.R + r]; }
        for (int l = 0; l < w.steps; ++l) {
            float* lg = w.logits.data() + (size_t)(0 * w.steps + l) * w.V;
            lg[a] = lg[b] = 60.f;
        }
        /* (b) sequence 1: the same with b = 310 in a's own shard, and the tie at the higher id first in the row */
        for (int l = 0; l < w.steps; ++l) {
            float* lg = w.logits.data() + (size_t)(1 * w.steps + l) * w.V;
            lg[c] = lg[a] = 61.f;
        }
        /* (c) sequence 2: the anchor out of the vocabulary and pred row of id 0 zero, so its scores are the
         * logits themselves at step 0: two equal maxima, 1900 and 40 (the lower wins) */
        w.anchor[(size_t)2 * w.astride] = -5;
        {
            float* lg = w.logits.data() + (size_t)(2 * w.steps + 0) * w.V;
            lg[1900] = lg[40] = 70.f;
        }
        /* (d) sequence 3: -0 against +0 as the best score -- the anchor's pred row zero (dot = +0), every logit of
         * step 0 negative but -0 at 2200 and +0 at 900: they tie, 900 wins; then -0 at 77 and +0 at 2300 at step 1
         * after a zero pred row for 900: 77 wins */
        const int an = 1234;
        w.anchor[(size_t)3 * w.astride] = an;
        for (int r = 0; r < w.R; ++r) { w1[(size_t)an * w.R + r] = 0; w1[(size_t)900 * w.R + r] = 0; }
        {
            float* lg0 = w.logits.data() + (size_t)(3 * w.steps + 0) * w.V;
            float* lg1 = w.logits.data() + (size_t)(3 * w.steps + 1) * w.V;
            for (int v = 0; v < w.V; ++v) { lg0[v] = -1.f - std::fabs(lg0[v]); lg1[v] = -1.f - std::fabs(lg1[v]); }
            lg0[2200] = -0.0f; lg0[900] = 0.0f;
            lg1[77] = -0.0f; lg1[2300] = 0.0f;
        }
        const std::vector<int32_t> want = ref_walk(w);
        const bool expect = want[0] == a && want[(size_t)1 * w.steps] == a && want[(size_t)2 * w.steps] == 40 &&
                            want[(size_t)3 * w.steps] == 900 && want[(size_t)3 * w.steps + 1] == 77;
        ++g_cases;
        std::printf("%-58s the plain walk takes the lower id at each forced tie: %s\n", "forced ties (reference)",
                    expect ? "yes" : "NO");
        if (!expect) ++g_fails;
        check("forced ties: cross-shard, same shard, logits only, -0/+0", w, { 1, 2, 3, 4 }, &want);
    }
    {
        /* NaN and -inf logits; sequence 1 step 2 all NaN (token 0), sequence 0 step 1 all -inf but one NaN */
        Walk w;
        w.V = 1800; w.R = 32; w.S = 3;
        std::vector<uint16_t> w1, w2;
        fill_random(w, w1, w2, rng, 0.4f, 2.f, false);
        for (size_t i = 0; i < w.logits.size(); i += 37) w.logits[i] = std::numeric_limits<float>::quiet_NaN();
        for (size_t i = 5; i < w.logits.size(); i += 53) w.logits[i] = -std::numeric_limits<float>::infinity();
        float* all_nan = w.logits.data() + (size_t)(1 * w.steps + 2) * w.V;
        for (int v = 0; v < w.V; ++v) all_nan[v] = std::numeric_limits<float>::quiet_NaN();
        float* all_inf = w.logits.data() + (size_t)(0 * w.steps + 1) * w.V;
        for (int v = 0; v < w.V; ++v) all_inf[v] = -std::numeric_limits<float>::infinity();
        all_inf[1500] = std::numeric_limits<float>::quiet_NaN();
        check("NaN / -inf logits, a row of NaN only", w, { 1, 2, 3, 4 });
    }

    /* ---- the checkpoint's tables at the real vocabulary */
    const char* td = std::getenv("RADIANCE_DS41_WALK_TABLES");
    if (td && *td && !quick) {
        const int V = 129280, R = 256;
        std::vector<uint16_t> w1((size_t)V * R), w2((size_t)V * R);
        bool loaded = true;
        for (int k = 0; k < 2; ++k) {
            const std::string p = std::string(td) + (k ? "/markov_w2.bf16" : "/markov_w1.bf16");
            FILE* f = std::fopen(p.c_str(), "rb");
            std::vector<uint16_t>& dst = k ? w2 : w1;
            if (!f || std::fread(dst.data(), 2, dst.size(), f) != dst.size()) { loaded = false; std::printf("real tables: cannot read %s\n", p.c_str()); }
            if (f) std::fclose(f);
        }
        if (!loaded) {
            ++g_cases; ++g_fails;
        } else {
            int picks = 0, out32 = 0, out16 = 0, worst = 0;
            for (int seed = 0; seed < 4; ++seed) {
                Walk w;
                w.V = V; w.R = R; w.S = seed == 3 ? 3 : 1 + seed % 2;
                w.w1 = w1.data(); w.w2 = w2.data();
                std::normal_distribution<float> nd(0.f, 1.f);
                std::uniform_int_distribution<int> id(0, V - 1);
                w.logits.resize((size_t)w.S * w.steps * V);
                /* a head's logits: a broad N(0, 2.5) body and a few tokens 3-9 above it */
                for (auto& x : w.logits) x = nd(rng) * 2.5f;
                for (int i = 0; i < w.S * w.steps; ++i)
                    for (int k = 0; k < 6; ++k) w.logits[(size_t)i * V + id(rng)] += 3.f + 6.f * std::fabs(nd(rng)) / 2.f;
                w.anchor.assign((size_t)(w.S - 1) * w.astride + 1, 0);
                for (int s = 0; s < w.S; ++s) w.anchor[(size_t)s * w.astride] = id(rng);
                const std::vector<int32_t> want = ref_walk(w);
                check("checkpoint tables, real vocabulary", w, { 4 }, &want);
                if (seed == 0) check("checkpoint tables, real vocabulary", w, { 3 }, &want);
                /* where the walked token sits in the head's own order */
                for (int i = 0; i < w.S * w.steps; ++i) {
                    const float* lg = w.logits.data() + (size_t)i * V;
                    const float t = lg[want[(size_t)i]];
                    int above = 0;
                    for (int v = 0; v < V; ++v) above += lg[v] > t || (lg[v] == t && v < want[(size_t)i]);
                    ++picks; out32 += above >= 32; out16 += above >= 16; worst = std::max(worst, above);
                }
            }
            std::printf("checkpoint tables: %d walked tokens, %d outside the head's top 16, %d outside its top 32 "
                        "(the deepest at head rank %d) -- synthetic logits, information only\n", picks, out16, out32, worst + 1);
            /* a forced cross-shard tie on the real tables: id 129000's succ row copied to 500, equal logits on top */
            Walk w;
            w.V = V; w.R = R; w.S = 1;
            std::vector<uint16_t> w2t(w2);
            for (int r = 0; r < R; ++r) w2t[(size_t)129000 * R + r] = w2t[(size_t)500 * R + r];
            w.w1 = w1.data(); w.w2 = w2t.data();
            std::normal_distribution<float> nd(0.f, 1.f);
            w.logits.resize((size_t)w.steps * V);
            for (auto& x : w.logits) x = nd(rng) * 2.5f;
            for (int l = 0; l < w.steps; ++l) w.logits[(size_t)l * V + 500] = w.logits[(size_t)l * V + 129000] = 80.f;
            w.anchor.assign(1, 4242);
            const std::vector<int32_t> want = ref_walk(w);
            ++g_cases;
            const bool lower = want[0] == 500;
            std::printf("%-58s the plain walk takes id 500 over 129000: %s\n", "checkpoint tables, cross-shard tie", lower ? "yes" : "NO");
            if (!lower) ++g_fails;
            check("checkpoint tables, cross-shard tie (500 vs 129000)", w, { 4 }, &want);
        }
    } else {
        std::printf("checkpoint tables: not run (RADIANCE_DS41_WALK_TABLES unset%s)\n", quick ? ", or --quick" : "");
    }
    /* ---- the gather between steps, as deepseek41 issues it, against libr4d's admission of the gather rows */
    {
        ++g_cases;
        const bool same = ds41walk::kSlot == R4D_DS41_WALK_SLOT;
        std::printf("%-58s deepseek41's slot %lld f32 = libr4d's R4D_DS41_WALK_SLOT %d: %s\n", "key slot", (long long)ds41walk::kSlot,
                    R4D_DS41_WALK_SLOT, same ? "yes" : "NO");
        if (!same) ++g_fails;
        const int64_t block = 5;
        int calls = 0, refused = 0;
        std::string first;
        for (int W : { 2, 3, 4, 8 })
            for (int64_t wseqs : { 1, 2, 3, 5, 8, 16 }) {
                const int64_t KR = ds41walk::plane_rows(block, wseqs), KC = ds41walk::keys_cols(), GC = ds41walk::gath_cols(W);
                /* the arena's planes, 256-byte aligned */
                float* kp = (float*)std::aligned_alloc(256, (size_t)((KR * KC * 4 + 255) / 256 * 256));
                float* gp = (float*)std::aligned_alloc(256, (size_t)((KR * GC * 4 + 255) / 256 * 256));
                for (int64_t S = 1; S <= wseqs; ++S)
                    for (int64_t l = 0; l < block; ++l) {
                        const int64_t r0 = ds41walk::step_row(l, wseqs);
                        RadTensor x = mat(kp + r0 * KC, RAD_F32, S, KC, KC), y = mat(gp + r0 * GC, RAD_F32, S, GC, GC);
                        long long dp = -1, sp = -1;
                        const int rc = r4d_ar_gather_admit(&x, &y, W, (long)ds41walk::gather_numel(wseqs), 4,
                                                           ds41walk::gather_row(), &dp, &sp);
                        ++calls;
                        if (rc != RAD_OK || dp != 0 || sp != 0) {
                            if (!refused++) first = "world " + std::to_string(W) + " wseqs " + std::to_string(wseqs) + " S " +
                                                    std::to_string(S) + " step " + std::to_string(l) + ": rc " + std::to_string(rc);
                        }
                    }
                std::free(kp);
                std::free(gp);
            }
        ++g_cases;
        std::printf("%-58s %d issues (world 2/3/4/8, 1-16 sequences a pass, every S and step): %s%s\n",
                    "the walk's all_gather vs libr4d's gather admission", calls, refused ? "REFUSED " : "all admitted",
                    refused ? first.c_str() : "");
        if (refused) ++g_fails;
        /* the geometry the gather refuses: an 8-byte key a row */
        float* kp = (float*)std::aligned_alloc(256, 4096);
        float* gp = (float*)std::aligned_alloc(256, 4096);
        RadTensor x = mat(kp, RAD_F32, 1, 2, 2), y = mat(gp, RAD_F32, 1, 8, 8);
        const int old = r4d_ar_gather_admit(&x, &y, 4, 16, 4, 2, nullptr, nullptr);
        std::free(kp);
        std::free(gp);
        ++g_cases;
        std::printf("%-58s refused: %s (rc %d)\n", "an 8-byte key row (row 2 f32)", old != RAD_OK ? "yes" : "NO", old);
        if (old == RAD_OK) ++g_fails;
    }
    std::printf("ds41 walk: %d case(s), %d failure(s)\n", g_cases, g_fails);
    return g_fails ? 1 : 0;
}
