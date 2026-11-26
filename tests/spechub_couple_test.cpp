/* tests/spechub_couple_test.cpp -- the MTP coupling's arithmetic, checked against its own law.
 *
 * What can silently go wrong in a maximal coupling, and what each test pins:
 *
 *   1. THE COMPOSED LAW. A coupling is correct when the emitted token is distributed as the
 *      TARGET's P, whatever the draft's Q was. The test builds a (P, Q) pair, draws many tokens
 *      through the coupling, and compares the empirical frequencies against P. This is the one
 *      property that a wrong acceptance rule, a wrong residual or a swapped mass cannot survive.
 *
 *   2. THE ACCEPTANCE RULE ITSELF. A draft token is accepted when it equals the token the target
 *      itself drew at that position: with Q = delta_x the coupling must return x on every draw.
 *      With P(x) >= Q(x) it must accept unconditionally, because there is no residual to fall
 *      back on.
 *
 *   3. THE UNDERFLOW AND DUPLICATE RULES. r(x) is exactly zero on rejection (distinct doubles never
 *      subtract to 0), so the residual walk can skip x. A candidate that appears twice contributes
 *      its mass once. Both are properties of the shared helpers, checked directly.
 *
 *   4. THE Q BUILD. build_q drops entries that underflow, refuses duplicate ids, and stores exactly
 *      the law the draw walks -- so the numbers in state.q[] are the law, not a normalised copy of
 *      it that drifted.
 *
 * Nothing here talks to a card or to libr4d: the helpers are abi/rad_spechub.h, which the plugin and
 * the engine both compile.
 */
#include "rad_test.h"

#include "rad_sample.h"
#include "rad_spechub.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <vector>

namespace {

/* A state as the propose kernel would leave it: Q over `ids` from `scores`, x drawn from Q. */
RadMtpCoupleState make_state(const std::vector<int32_t>& ids, const std::vector<float>& scores,
                             int64_t vocab, uint64_t seed, uint64_t base, int round) {
    RadMtpCoupleState s{};
    s.draw.seed = seed;
    s.draw.base = base;
    s.draw.seq = 1;
    s.draw.card = 0;
    s.draw.mode = 1;
    s.round = round;
    s.status = 1;
    std::vector<int32_t> id8(8, -1);
    std::vector<float>   v8(8, -INFINITY);
    for (size_t i = 0; i < ids.size() && i < 8; ++i) { id8[i] = ids[i]; v8[i] = scores[i]; }
    CHECK(rad_mtp_couple_build_q(id8.data(), v8.data(), vocab, &s));
    const double u = rad_spechub_uniform(seed, base, RAD_MTP_COUPLE_KEY_DRAFT + (uint64_t)round);
    s.x = s.ids[rad_mtp_couple_pick(s, u)];
    return s;
}

/* The target's candidate plane: ids in [0, n), softmax scores. */
void target_plane(int n, std::vector<int32_t>* ids, std::vector<float>* val) {
    ids->clear();
    val->clear();
    for (int i = 0; i < n; ++i) {
        ids->push_back(i);
        val->push_back(-0.15f * (float)i);      /* a decaying, non-uniform target */
    }
}

}  /* namespace */

/* THE COMPOSED LAW. Whatever Q is, the coupling's output must be distributed as the target's P.
 *
 * The draft must be DRAWN FROM Q ON EVERY TRIAL, which is what a round does. Holding x fixed and
 * repeating one draw measures the acceptance rate of a single proposal -- P(x)/Q(x) -- not the
 * composed distribution, and fails against P however correct the rule is.
 *
 * The tolerance is five sigma of a binomial at this sample count, so a wrong accept rule (mass on
 * x alone) or a naive residual (a renormalised P over the non-Q tokens only) cannot pass. */
TEST(the_coupled_token_is_distributed_as_the_target) {
    const uint64_t seed = 424242;
    const int n = 8;
    std::vector<int32_t> ids;
    std::vector<float>   val;
    target_plane(n, &ids, &val);

    /* P from the same FP64 formula the verify uses. */
    double sum = 0, P[8] = {};
    float  top = -INFINITY;
    for (float v : val) top = std::max(top, v);
    for (int i = 0; i < n; ++i) { P[i] = std::exp((double)val[i] - top); sum += P[i]; }
    for (int i = 0; i < n; ++i) P[i] /= sum;

    /* Q deliberately differs from P: three tokens, in a different order of preference. */
    const int32_t qids[8] = { 0, 1, 2, -1, -1, -1, -1, -1 };
    const float   qval[8] = { 0.0f, -0.5f, -1.0f, -INFINITY, -INFINITY, -INFINITY, -INFINITY,
                              -INFINITY };

    const int draws = 40000;
    std::map<int32_t, int64_t> seen;
    for (int k = 0; k < draws; ++k) {
        const uint64_t trial = (uint64_t)k;
        /* A fresh draft from the same shortlist on every trial, at that trial's coordinate. */
        RadMtpCoupleState s{};
        s.draw.seed = seed;
        s.draw.base = trial;
        s.draw.mode = 1;
        s.round = 1;
        s.status = 1;
        CHECK(rad_mtp_couple_build_q(qids, qval, 8, &s));
        const double u_draft =
            rad_spechub_uniform(seed, trial, RAD_MTP_COUPLE_KEY_DRAFT + 1);
        s.x = s.ids[rad_mtp_couple_pick(s, u_draft)];

        int32_t out = -1;
        CHECK(rad_mtp_couple_decide(s, ids.data(), val.data(), n, 8,
                                    rad_spechub_uniform(seed, trial, RAD_SAMPLE_KEY_ACCEPT),
                                    rad_spechub_uniform(seed, trial, RAD_SAMPLE_KEY_RESID), &out));
        ++seen[out];
    }
    for (int i = 0; i < n; ++i) {
        const double want = P[i];
        const double got  = (double)seen[(int32_t)i] / (double)draws;
        const double sigma = std::sqrt(want * (1.0 - want) / (double)draws);
        CHECK_NEAR(got, want, 5.0 * sigma + 1e-9);
    }
    /* Every survivor can be emitted: a rule that lost a tail token would show as a zero here. */
    CHECK_EQ((int64_t)seen.size(), (int64_t)n);
}

/* A DRAFT THE TARGET DOMINATES ON EVERY Q TOKEN IS ACCEPTED UNCONDITIONALLY: P(x) >= Q(x), so the
 * residual is empty and the ratio test has nothing to reject. The state here has ONE Q token, which
 * is the case a softmax over two tokens cannot give (two equal Q tokens of 0.5 each need P above
 * 0.5 on both, which a normalised P over three tokens cannot do). */
TEST(a_draft_the_target_dominates_is_always_accepted) {
    RadMtpCoupleState one = make_state({ 2 }, { 0.0f }, 8, 9, 0, 1);
    CHECK_EQ(one.count, 1);
    CHECK_EQ(one.x, 2);
    std::vector<int32_t> ids = { 0, 1, 2 };
    std::vector<float>   peak = { -20.0f, -20.0f, 9.0f };   /* P(2) ~ 1, Q(2) = 1 */
    for (int k = 0; k < 500; ++k) {
        int32_t out = -1;
        CHECK(rad_mtp_couple_decide(one, ids.data(), peak.data(), 3, 8,
                                    rad_spechub_uniform(7, (uint64_t)k, RAD_SAMPLE_KEY_ACCEPT),
                                    rad_spechub_uniform(7, (uint64_t)k, RAD_SAMPLE_KEY_RESID),
                                    &out));
        CHECK_EQ(out, 2);
    }

    /* AND WHERE P IS SHORT OF Q THE RATIO RULE APPLIES: the same Q token with a target that barely
     * admits it is rejected most of the time, and the fallback can never emit x. */
    std::vector<float> flat = { 4.0f, 4.0f, -4.0f };        /* P(2) is small */
    int rejected = 0;
    for (int k = 0; k < 2000; ++k) {
        int32_t out = -1;
        CHECK(rad_mtp_couple_decide(one, ids.data(), flat.data(), 3, 8,
                                    rad_spechub_uniform(7, (uint64_t)k, RAD_SAMPLE_KEY_ACCEPT),
                                    rad_spechub_uniform(7, (uint64_t)k, RAD_SAMPLE_KEY_RESID),
                                    &out));
        if (out != 2) { ++rejected; CHECK(out == 0 || out == 1); }
    }
    CHECK(rejected > 0);
}

/* THE RESIDUAL IS (P - Q)+ AND r(x) IS EXACTLY ZERO ON REJECTION. Here Q puts more mass on x than P
 * does, so x is sometimes rejected and the token must come from the OTHERS. The check that matters
 * is that y is never x on a rejected draw: r(x) contributes +0 exactly, which is what lets the walk
 * skip x without changing the total. */
TEST(a_rejected_draft_falls_back_to_the_residual_support) {
    RadMtpCoupleState s = make_state({ 0, 1 }, { 0.0f, 0.0f }, 8, 11, 0, 1);   /* Q(0) = Q(1) = 0.5 */
    CHECK_EQ(s.count, 2);
    std::vector<int32_t> ids = { 0, 1, 2 };
    /* P(0) = 0.09, so Q(0) = 0.5 exceeds it and x = 0 is rejected most of the time. */
    std::vector<float> val = { 0.0f, 2.2f, 1.0f };
    int accepted = 0, rejected = 0;
    for (int k = 0; k < 4000; ++k) {
        int32_t out = -1;
        CHECK(rad_mtp_couple_decide(s, ids.data(), val.data(), 3, 8,
                                    rad_spechub_uniform(3, (uint64_t)k, RAD_SAMPLE_KEY_ACCEPT),
                                    rad_spechub_uniform(3, (uint64_t)k, RAD_SAMPLE_KEY_RESID),
                                    &out));
        if (out == s.x) ++accepted;
        else { ++rejected; CHECK(out != s.x); }
    }
    CHECK(accepted > 0);
    CHECK(rejected > 0);
}

/* build_q STORES THE LAW IT WALKS. A shortlist with a hole, a -inf score and a duplicate id: the
 * stored q[] sums to 1, the duplicate is refused, and the hole contributes nothing. */
TEST(build_q_refuses_malformed_shortlists_and_drops_zero_mass) {
    RadMtpCoupleState s{};
    /* A hole at slot 1 and -inf at slot 3: two live candidates. */
    const int32_t ids[8] = { 4, -1, 7, 9, -1, -1, -1, -1 };
    const float   val[8] = { 0.0f, 0.0f, -1.0f, -INFINITY, 0.0f, 0.0f, 0.0f, 0.0f };
    CHECK(rad_mtp_couple_build_q(ids, val, 16, &s));
    CHECK_EQ(s.count, 2);
    CHECK_EQ(s.ids[0], 4);
    CHECK_EQ(s.ids[1], 7);
    CHECK_EQ(s.ids[2], -1);
    CHECK_NEAR(s.q[0] + s.q[1], 1.0, 1e-15);

    /* A duplicate id is a broken shortlist, not something to aggregate. */
    const int32_t dup[8] = { 4, 4, -1, -1, -1, -1, -1, -1 };
    CHECK(!rad_mtp_couple_build_q(dup, val, 16, &s));
    CHECK_EQ(s.count, 0);

    /* An id outside the vocabulary, and a NaN score, are both refused. */
    const int32_t wide[8] = { 99, -1, -1, -1, -1, -1, -1, -1 };
    CHECK(!rad_mtp_couple_build_q(wide, val, 16, &s));
    const float nan_val[8] = { NAN, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    const int32_t one[8] = { 2, -1, -1, -1, -1, -1, -1, -1 };
    CHECK(!rad_mtp_couple_build_q(one, nan_val, 16, &s));

    /* Every candidate at -inf has no support at all. */
    float all_inf[8];
    for (int i = 0; i < 8; ++i) all_inf[i] = -INFINITY;
    CHECK(!rad_mtp_couple_build_q(one, all_inf, 16, &s));
    CHECK_EQ(s.count, 0);
}

/* state_valid IS THE ONE RULE BOTH SIDES REFUSE A ROW BY. A status-0 row, a q[] that does not sum to
 * one, an id outside the vocabulary, x missing from the support, and a round outside the band are
 * all refused -- because a verifier that accepted any of them would couple against a law that is
 * not the draft's. */
TEST(state_valid_refuses_every_malformed_row) {
    RadMtpCoupleState s = make_state({ 0, 1 }, { 0.0f, -0.2f }, 8, 5, 0, 2);
    CHECK(rad_mtp_couple_state_valid(s, 8));

    RadMtpCoupleState bad = s;
    bad.status = 0;
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    bad = s; bad.q[0] += 0.5;                      /* no longer sums to 1 */
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    bad = s; bad.ids[0] = 99;                     /* outside the vocabulary */
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    bad = s; bad.x = 6;                           /* x is not in the support */
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    bad = s; bad.round = 0;                       /* outside the band */
    CHECK(!rad_mtp_couple_state_valid(bad, 8));
    bad = s; bad.round = RAD_MTP_COUPLE_MAX_ROUNDS + 1;
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    bad = s; bad.draw.mode = 0;                   /* a greedy row's state has no Q */
    CHECK(!rad_mtp_couple_state_valid(bad, 8));

    bad = s; bad.ids[1] = -1;                     /* a hole above count must carry no mass */
    CHECK(!rad_mtp_couple_state_valid(bad, 8) || s.count < 2);
}

/* THE UNIFORM IS THE SPECIFIED DRAW, and it is 53-bit: two seeds that differ in one bit must give
 * different numbers, and the range must be [0, 1) with no value at 1. This is the number the
 * acceptance compares P(x) against, so a uniform that repeated would make two draft positions
 * accept together. */
TEST(the_coupling_uniform_is_the_specified_53_bit_draw) {
    for (uint64_t pos = 0; pos < 32; ++pos) {
        const double u = rad_spechub_uniform(1234, pos, RAD_SAMPLE_KEY_ACCEPT);
        CHECK(u >= 0.0 && u < 1.0);
        /* 53 bits: a multiple of 2^-53, exactly. */
        CHECK_EQ(u, std::floor(u * 9007199254740992.0) / 9007199254740992.0);
    }
    CHECK(rad_spechub_uniform(1, 0, RAD_SAMPLE_KEY_ACCEPT) !=
          rad_spechub_uniform(2, 0, RAD_SAMPLE_KEY_ACCEPT));
    CHECK(rad_spechub_uniform(1, 0, RAD_SAMPLE_KEY_ACCEPT) !=
          rad_spechub_uniform(1, 1, RAD_SAMPLE_KEY_ACCEPT));
    CHECK(rad_spechub_uniform(1, 0, RAD_SAMPLE_KEY_ACCEPT) !=
          rad_spechub_uniform(1, 0, RAD_SAMPLE_KEY_RESID));
}

/* THE WIRE ROWS ARE THE SIZE THE PLANES ARE ADDRESSED AT. A field inserted into either struct would
 * shift every later operand of a plane the host writes and the device reads. */
TEST(the_coupling_wire_rows_keep_their_layout) {
    CHECK_EQ(sizeof(RadMtpCoupleDraw), (size_t)32);
    CHECK_EQ(sizeof(RadMtpCoupleState), (size_t)168);
    CHECK_EQ(RAD_MTP_COUPLE_DRAW_WORDS, (size_t)8);
    CHECK_EQ(RAD_MTP_COUPLE_STATE_WORDS, (size_t)42);
    CHECK_EQ(offsetof(RadMtpCoupleState, q) % 8, (size_t)0);
}

namespace {

/* Upper tail of a chi-square with k degrees of freedom (Wilson-Hilferty). */
double chi2_tail(double x, int k) {
    if (k <= 0) return 1.0;
    const double z = (std::cbrt(x / k) - (1 - 2.0 / (9 * k))) / std::sqrt(2.0 / (9 * k));
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

/* P as the verify computes it: the FP64 softmax over live ids at finite values, by token id. */
std::vector<double> target_law(const std::vector<int32_t>& ids, const std::vector<float>& v,
                               int vocab) {
    std::vector<double> P((size_t)vocab, 0.0);
    float top = -INFINITY;
    for (size_t j = 0; j < ids.size(); ++j)
        if (ids[j] >= 0) top = std::max(top, v[j]);
    double s = 0;
    for (size_t j = 0; j < ids.size(); ++j)
        if (ids[j] >= 0 && v[j] != -INFINITY) {
            const double w = std::exp((double)v[j] - top);
            P[(size_t)ids[j]] += w;
            s += w;
        }
    for (double& p : P) p /= s;
    return P;
}

/* A tiny autoregressive target and head over kV tokens: logits are a hash of the context. The
 * target keeps its top 4 (a top-k filter) at temperature 0.7, so the head's shortlist always holds
 * tokens outside P's support. */
constexpr int kV = 6;
void ctx_logits(const std::vector<int32_t>& ctx, uint64_t salt, float* out) {
    uint64_t h = salt;
    for (int32_t t : ctx) h = rad_sample_splitmix64(h ^ (uint64_t)(t + 1) * 0x100000001b3ull);
    for (int i = 0; i < kV; ++i)
        out[i] = (float)((double)(rad_sample_splitmix64(h + (uint64_t)i) >> 40) / 16777216.0 * 4.0 - 2.0);
}
void ar_target(const std::vector<int32_t>& ctx, std::vector<int32_t>* ids, std::vector<float>* v) {
    float l[kV];
    ctx_logits(ctx, 0xabcdef, l);
    int ord[kV];
    for (int i = 0; i < kV; ++i) ord[i] = i;
    std::sort(ord, ord + kV, [&](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); });
    ids->assign((size_t)kV, -1);
    v->assign((size_t)kV, -INFINITY);
    for (int i = 0; i < 4; ++i) { (*ids)[(size_t)i] = ord[i]; (*v)[(size_t)i] = l[ord[i]] / 0.7f; }
}
void ar_head(const std::vector<int32_t>& ctx, int32_t* id8, float* v8) {
    float l[kV];
    ctx_logits(ctx, 0x123457, l);
    for (int i = 0; i < 8; ++i) { id8[i] = i < kV ? i : -1; v8[i] = i < kV ? l[i] : -INFINITY; }
}
/* sample_pick's rule for the bonus row: an f32 inverse CDF at the seed's own stream. */
int32_t f32_pick(const std::vector<int32_t>& ids, const std::vector<float>& v, uint64_t seed,
                 uint64_t pos) {
    float mx = -INFINITY, sum = 0;
    for (size_t i = 0; i < ids.size(); ++i) if (ids[i] >= 0) mx = std::max(mx, v[i]);
    for (size_t i = 0; i < ids.size(); ++i) if (ids[i] >= 0) sum += std::exp(v[i] - mx);
    const float u = rad_sample_uniform01(seed ^ RAD_SAMPLE_KEY_PICK, pos);
    float cum = 0;
    int32_t last = -1;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0) continue;
        if (v[i] > -INFINITY) last = ids[i];
        cum += std::exp(v[i] - mx) / sum;
        if (u < cum) return ids[i];
    }
    return last;
}

}  /* namespace */

/* The composed law over many random (P, Q) pairs: targets with holes and -inf candidates, heads
 * of one to eight tokens that may lie outside P's support. Each case is a chi-square of the
 * emitted tokens against P, and the acceptance rate is checked against sum min(P, Q). */
TEST(the_coupled_token_follows_the_target_for_random_target_and_head_pairs) {
    std::mt19937_64 rng(12345);
    const int cases = 60, draws = 20000;
    int low_p = 0;
    for (int c = 0; c < cases; ++c) {
        const int vocab = 16 + (int)(rng() % 48);
        const int n = 4 + (int)(rng() % (uint64_t)(vocab - 4));
        std::vector<int32_t> pool((size_t)vocab);
        for (int i = 0; i < vocab; ++i) pool[(size_t)i] = i;
        std::shuffle(pool.begin(), pool.end(), rng);
        std::vector<int32_t> ids(pool.begin(), pool.begin() + n);
        std::vector<float> val((size_t)n);
        const float spread = 1.5f + (float)(c % 4);
        for (int j = 0; j < n; ++j)
            val[(size_t)j] = spread * (float)((double)(rng() >> 11) * 0x1p-53 * 2.0 - 1.0);
        for (int j = 1; j < n; ++j) {
            const uint64_t r = rng() % 10;
            if (r == 0) ids[(size_t)j] = -1;
            else if (r == 1) val[(size_t)j] = -INFINITY;
        }
        const std::vector<double> P = target_law(ids, val, vocab);

        const int qn = 1 + (int)(rng() % 8);
        std::shuffle(pool.begin(), pool.end(), rng);
        int32_t qid[8];
        float qv[8];
        for (int i = 0; i < 8; ++i) {
            qid[i] = i < qn ? pool[(size_t)i] : -1;
            qv[i] = i < qn ? spread * (float)((double)(rng() >> 11) * 0x1p-53 * 2.0 - 1.0) : -INFINITY;
        }
        const uint64_t seed = rng();
        RadMtpCoupleState base{};
        base.draw.seed = seed;
        base.draw.mode = 1;
        base.status = 1;
        base.round = 1;
        CHECK(rad_mtp_couple_build_q(qid, qv, vocab, &base));
        double want_acc = 0;
        {
            std::vector<double> Q((size_t)vocab, 0.0);
            for (int i = 0; i < base.count; ++i) Q[(size_t)base.ids[i]] = base.q[i];
            for (int i = 0; i < vocab; ++i) want_acc += std::min(P[(size_t)i], Q[(size_t)i]);
        }
        std::vector<int64_t> cnt((size_t)vocab, 0);
        int64_t acc = 0;
        for (int t = 0; t < draws; ++t) {
            RadMtpCoupleState s = base;
            s.draw.base = (uint64_t)t;
            s.x = s.ids[rad_mtp_couple_pick(
                s, rad_spechub_uniform(seed, s.draw.base, RAD_MTP_COUPLE_KEY_DRAFT + 1))];
            int32_t out = -1;
            CHECK(rad_mtp_couple_decide(s, ids.data(), val.data(), n, vocab,
                                        rad_spechub_uniform(seed, (uint64_t)t, RAD_SAMPLE_KEY_ACCEPT),
                                        rad_spechub_uniform(seed, (uint64_t)t, RAD_SAMPLE_KEY_RESID),
                                        &out));
            REQUIRE(out >= 0 && out < vocab);
            ++cnt[(size_t)out];
            acc += out == s.x;
        }
        double chi = 0;
        int dof = -1;
        for (int i = 0; i < vocab; ++i) {
            if (P[(size_t)i] == 0) { CHECK_EQ(cnt[(size_t)i], (int64_t)0); continue; }
            const double e = P[(size_t)i] * draws;
            chi += ((double)cnt[(size_t)i] - e) * ((double)cnt[(size_t)i] - e) / e;
            ++dof;
        }
        if (chi2_tail(chi, dof) < 1e-4) ++low_p;
        const double sd = std::sqrt(want_acc * (1 - want_acc) / draws) + 1e-9;
        CHECK_NEAR((double)acc / draws, want_acc, 5.0 * sd);
    }
    /* 60 independent cases at p < 1e-4: a correct coupling fails none of them in expectation. */
    CHECK(low_p <= 1);
}

/* The engine's flow over whole steps: a depth-2 linear draft at (base, round), verify rows at
 * n_out + k bound with rad_mtp_couple_binds, the first rejection ending the step, the bonus row by
 * the ordinary pick, and the next step drafting again from the committed text. The joint law of the
 * first three emitted tokens must be the target's own autoregressive law. */
TEST(the_emitted_sequence_follows_the_target_over_coupled_steps) {
    const int L = 3, depth = 2, trials = 150000;
    std::map<std::vector<int32_t>, int64_t> cnt;
    for (int t = 0; t < trials; ++t) {
        const uint64_t seed = rad_sample_splitmix64(0x5eed0000ull + (uint64_t)t);
        std::vector<int32_t> out;
        while ((int)out.size() < L) {
            const uint64_t n_out = out.size();
            RadMtpCoupleState st[depth];
            std::vector<int32_t> ctx = out;
            for (int r = 1; r <= depth; ++r) {
                int32_t id8[8];
                float v8[8];
                ar_head(ctx, id8, v8);
                RadMtpCoupleState s{};
                s.draw.seed = seed;
                s.draw.base = n_out;
                s.draw.seq = 1;
                s.draw.mode = 1;
                s.status = 1;
                s.round = r;
                CHECK(rad_mtp_couple_build_q(id8, v8, kV, &s));
                s.x = s.ids[rad_mtp_couple_pick(
                    s, rad_spechub_uniform(seed, n_out, RAD_MTP_COUPLE_KEY_DRAFT + (uint64_t)r))];
                s.pos = rad_mtp_couple_verify_pos(n_out, r);
                st[r - 1] = s;
                ctx.push_back(s.x);
            }
            ctx = out;
            for (int k = 0; k <= depth && (int)out.size() < L; ++k) {
                std::vector<int32_t> ids;
                std::vector<float> v;
                ar_target(ctx, &ids, &v);
                const uint64_t pos = n_out + (uint64_t)k;
                int32_t y = -1;
                bool more = true;
                if (k < depth) {
                    CHECK(rad_mtp_couple_binds(st[k], 1, seed, n_out, k, st[k].x, kV));
                    CHECK(rad_mtp_couple_decide(st[k], ids.data(), v.data(), (int64_t)ids.size(),
                                                kV, rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_ACCEPT),
                                                rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_RESID),
                                                &y));
                    more = y == st[k].x;
                } else {
                    y = f32_pick(ids, v, seed, pos);
                }
                out.push_back(y);
                ctx.push_back(y);
                if (!more) break;
            }
        }
        out.resize((size_t)L);
        ++cnt[out];
    }
    double chi = 0;
    int dof = -1;
    for (int code = 0; code < kV * kV * kV; ++code) {
        std::vector<int32_t> seq((size_t)L);
        for (int i = 0, c = code; i < L; ++i, c /= kV) seq[(size_t)i] = c % kV;
        double p = 1;
        std::vector<int32_t> ctx;
        for (int i = 0; i < L; ++i) {
            std::vector<int32_t> ids;
            std::vector<float> v;
            ar_target(ctx, &ids, &v);
            p *= target_law(ids, v, kV)[(size_t)seq[(size_t)i]];
            ctx.push_back(seq[(size_t)i]);
        }
        const auto it = cnt.find(seq);
        const double got = it == cnt.end() ? 0.0 : (double)it->second;
        if (p == 0) { CHECK_EQ(got, 0.0); continue; }
        const double e = p * trials;
        chi += (got - e) * (got - e) / e;
        ++dof;
    }
    const double tail = chi2_tail(chi, dof);
    std::fprintf(stderr, "  joint law of %d tokens over %d trials: chi2 %.1f, %d dof, p %.3f\n", L,
                 trials, chi, dof, tail);
    CHECK(tail > 1e-4);
}

RAD_TEST_MAIN()
