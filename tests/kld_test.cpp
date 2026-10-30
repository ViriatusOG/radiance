/* kld_test.cpp -- the KL scorer (core/kld.h): its arithmetic and its reference files.
 *
 * No device and no model. A reference is recorded from synthetic logits split over two vocabulary
 * shards, as two ranks hand them over, and a candidate is scored against it. Each position's KL,
 * argmax agreement and NLL must match a direct double computation from the f32 logits. The
 * reference's f16 storage is the only approximation and sets the tolerance; a candidate identical
 * to the reference scores zero. The sparse form (--kld-top) is checked the same way against the
 * KL of the two distributions coarsened to the reference's head and one tail bucket.
 */
#include "rad_test.h"

#include "kld.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using namespace rad;

namespace {

constexpr int64_t V = 1000;          /* two shards of 500 */
constexpr int64_t W = V / 2;

/* An empty directory of the case's own, under the test's temporary directory: a reference is
 * recorded once into a path, and a second recording into it is refused. */
struct TmpDir {
    std::string path;
    TmpDir() {
        std::string t = radtest::tmp_path("case.XXXXXX");
        path = mkdtemp(t.data()) ? t : "";
    }
};

/* Logits with a confident head: one token a few nats above a normal spread, the shape where an
 * f16 head would lose the most precision. */
std::vector<float> make_logits(int64_t rows, uint32_t seed) {
    std::mt19937 g(seed);
    std::normal_distribution<float> n(0.0f, 2.5f);
    std::uniform_int_distribution<int> pick(0, (int)V - 1);
    std::vector<float> l((size_t)(rows * V));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < V; ++i) l[(size_t)(r * V + i)] = n(g);
        l[(size_t)(r * V + pick(g))] += (r % 3 == 0) ? 2.0f : 9.0f;
    }
    return l;
}

/* The two shards a step hands over, from full rows. */
struct Shards {
    std::vector<float> a, b;
    std::vector<KldShard> s;
    Shards(const std::vector<float>& full, int64_t rows) : a((size_t)(rows * W)), b((size_t)(rows * W)) {
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t i = 0; i < W; ++i) {
                a[(size_t)(r * W + i)] = full[(size_t)(r * V + i)];
                b[(size_t)(r * W + i)] = full[(size_t)(r * V + W + i)];
            }
        s = { { a.data(), 0, W }, { b.data(), W, W } };
    }
};

double lse(const float* l) {
    double m = l[0];
    for (int64_t i = 1; i < V; ++i) m = std::max(m, (double)l[i]);
    double s = 0;
    for (int64_t i = 0; i < V; ++i) s += std::exp((double)l[i] - m);
    return m + std::log(s);
}

double kl_direct(const float* p, const float* q) {
    const double zp = lse(p), zq = lse(q);
    double kl = 0;
    for (int64_t i = 0; i < V; ++i) {
        const double lp = (double)p[i] - zp;
        kl += std::exp(lp) * (lp - ((double)q[i] - zq));
    }
    return kl;
}

int64_t argmax(const float* l) {
    int64_t a = 0;
    for (int64_t i = 1; i < V; ++i) if (l[i] > l[a]) a = i;
    return a;
}

/* Two documents: 40 tokens scored from 8 and 25 scored from 0, so 31 + 24 rows. */
const std::vector<int32_t> kDocA = [] { std::vector<int32_t> t(40); for (int i = 0; i < 40; ++i) t[i] = (i * 37) % V; return t; }();
const std::vector<int32_t> kDocB = [] { std::vector<int32_t> t(25); for (int i = 0; i < 25; ++i) t[i] = (i * 91 + 5) % V; return t; }();

/* Positions [0, n) of a document, one logits row each, handed over in two chunks as a chunked
 * prefill would, the first ending mid-document. */
void feed(Kld& k, uint64_t id, int64_t n, const std::vector<float>& full) {
    const int64_t cut = n / 2;
    {
        std::vector<float> part(full.begin(), full.begin() + cut * V);
        Shards sh(part, cut);
        CHECK_OK(k.score({ { id, 0, cut, 0 } }, sh.s));
    }
    {
        std::vector<float> part(full.begin() + cut * V, full.begin() + n * V);
        Shards sh(part, n - cut);
        CHECK_OK(k.score({ { id, cut, n - cut, 0 } }, sh.s));
    }
}

std::vector<float> read_rows(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(0, std::ios::end);
    std::vector<float> v((size_t)in.tellg() / sizeof(float));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(v.size() * sizeof(float)));
    return v;
}

void record(const std::string& dir, const std::vector<float>& la, const std::vector<float>& lb) {
    Kld rec;
    CHECK_OK(rec.add_doc("a", kDocA, 8));
    CHECK_OK(rec.add_doc("b", kDocB, 0));
    CHECK_OK(rec.record_into(dir, V, "reference"));
    CHECK_EQ(rec.n_rows(), (int64_t)(31 + 24));
    feed(rec, 1, 40, la);
    feed(rec, 2, 25, lb);
    CHECK_OK(rec.finish());
}

}  /* namespace */

/* Every scored position of a perturbed candidate matches a direct double computation, and rows
 * outside each document's scored range are left out. */
TEST(a_candidate_scores_what_a_direct_computation_gives) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    const std::vector<float> ra = make_logits(40, 1), rb = make_logits(25, 2);
    record(t.path + "/ref", ra, rb);

    /* The candidate: the reference plus noise, enough to flip some argmaxes. */
    std::vector<float> ca = ra, cb = rb;
    std::mt19937 g(7);
    std::normal_distribution<float> n(0.0f, 0.6f);
    for (float& v : ca) v += n(g);
    for (float& v : cb) v += n(g);

    Kld k;
    CHECK_OK(k.score_against(t.path + "/ref", t.path + "/rep.json", V, "candidate"));
    feed(k, 1, 40, ca);
    feed(k, 2, 25, cb);
    CHECK_OK(k.finish());

    const std::vector<float> rows = read_rows(t.path + "/rep.json.rows");
    REQUIRE_EQ(rows.size(), (size_t)(55 * 4));
    int64_t r = 0, flips = 0;
    auto check_doc = [&](const std::vector<float>& ref, const std::vector<float>& cand,
                         const std::vector<int32_t>& tok, int64_t from) {
        for (int64_t p = from; p + 1 < (int64_t)tok.size(); ++p, ++r) {
            const float* lp = ref.data() + p * V;
            const float* lq = cand.data() + p * V;
            const double want = kl_direct(lp, lq);
            CHECK_NEAR(rows[(size_t)(r * 4 + 0)], want, 2e-5 + 1e-3 * want);
            CHECK_NEAR(rows[(size_t)(r * 4 + 1)], lse(lq) - lq[tok[(size_t)p + 1]], 1e-4);
            CHECK_NEAR(rows[(size_t)(r * 4 + 2)], lse(lp) - lp[tok[(size_t)p + 1]], 1e-4);
            const bool same = argmax(lp) == argmax(lq);
            CHECK_EQ(rows[(size_t)(r * 4 + 3)], same ? 1.0f : 0.0f);
            flips += !same;
        }
    };
    check_doc(ra, ca, kDocA, 8);
    check_doc(rb, cb, kDocB, 0);
    CHECK_EQ(r, (int64_t)55);
    CHECK(flips > 0);         /* the noise must flip some argmax for the check to mean anything */
}

/* Control: a candidate identical to the reference scores a KL of only the f16 storage's
 * second-order residue, and every argmax agrees. */
TEST(the_reference_against_itself_scores_zero) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    const std::vector<float> ra = make_logits(40, 3), rb = make_logits(25, 4);
    record(t.path + "/ref", ra, rb);
    Kld k;
    CHECK_OK(k.score_against(t.path + "/ref", t.path + "/rep.json", V, "same"));
    feed(k, 1, 40, ra);
    feed(k, 2, 25, rb);
    CHECK_OK(k.finish());
    const std::vector<float> rows = read_rows(t.path + "/rep.json.rows");
    REQUIRE_EQ(rows.size(), (size_t)(55 * 4));
    for (size_t r = 0; r < 55; ++r) {
        CHECK(std::fabs(rows[r * 4 + 0]) < 1e-6f);
        CHECK_EQ(rows[r * 4 + 3], 1.0f);
        CHECK_NEAR(rows[r * 4 + 1], rows[r * 4 + 2], 1e-5);
    }
}

/* A reference is never overwritten, and one whose recording did not finish is never read. */
TEST(a_reference_is_written_once_and_read_only_when_complete) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    const std::vector<float> ra = make_logits(40, 5), rb = make_logits(25, 6);
    record(t.path + "/ref", ra, rb);
    {
        Kld again;
        CHECK_OK(again.add_doc("a", kDocA, 8));
        CHECK_EQ(again.record_into(t.path + "/ref", V, "x"), RAD_E_INVAL);
    }
    /* A recording that stops short writes no manifest, finish() reports the gap, and the
     * directory is not a reference. */
    {
        Kld half;
        CHECK_OK(half.add_doc("a", kDocA, 8));
        CHECK_OK(half.add_doc("b", kDocB, 0));
        CHECK_OK(half.record_into(t.path + "/half", V, "x"));
        feed(half, 1, 40, ra);
        CHECK_EQ(half.finish(), RAD_E_STATE);
    }
    Kld k;
    CHECK(k.score_against(t.path + "/half", "", V, "x") < 0);
    /* A vocabulary size that is not the reference's. */
    Kld v;
    CHECK_EQ(v.score_against(t.path + "/ref", "", V + 1, "x"), RAD_E_INVAL);
}

/* A document with nothing to score is refused when added, not found later as a missing row. */
TEST(a_document_must_leave_a_position_to_score) {
    Kld k;
    CHECK_EQ(k.add_doc("x", { 1 }, 0), RAD_E_INVAL);
    CHECK_EQ(k.add_doc("x", { 1, 2, 3 }, 2), RAD_E_INVAL);
    CHECK_EQ(k.add_doc("x", { 1, 2, 3 }, -1), RAD_E_INVAL);
    CHECK_OK(k.add_doc("x", { 1, 2, 3 }, 1));
    CHECK_EQ(k.n_rows(), (int64_t)1);
}

/* A position fed twice, as a re-prefill after preemption feeds it, is scored again from the
 * second feed and counted once: the report holds the later logits' KL, and a document never fed
 * is still missing. Counted twice, a repeated document would cover for one the engine dropped. */
TEST(a_position_fed_twice_is_counted_once_and_scored_from_the_second_feed) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    const std::vector<float> ra = make_logits(40, 11), rb = make_logits(25, 12);
    record(t.path + "/ref", ra, rb);
    std::vector<float> stale = ra;
    for (float& v : stale) v = -v;
    {
        Kld k;
        CHECK_OK(k.score_against(t.path + "/ref", "", V, "candidate"));
        feed(k, 1, 40, ra);
        feed(k, 1, 40, ra);
        CHECK_EQ(k.rows_done(), (int64_t)31);
        CHECK_EQ(k.finish(), RAD_E_STATE);           /* document b never came back */
    }
    Kld k;
    CHECK_OK(k.score_against(t.path + "/ref", t.path + "/rep.json", V, "candidate"));
    feed(k, 1, 40, stale);
    feed(k, 1, 40, ra);                              /* the re-prefill: the reference itself */
    feed(k, 2, 25, rb);
    CHECK_EQ(k.rows_done(), (int64_t)55);
    CHECK_OK(k.finish());
    const std::vector<float> rows = read_rows(t.path + "/rep.json.rows");
    REQUIRE_EQ(rows.size(), (size_t)(55 * 4));
    for (size_t r = 0; r < 55; ++r) CHECK(std::fabs(rows[r * 4 + 0]) < 1e-6f);
}

/* Spans reaching outside a document's scored range, or naming no document, are skipped: only
 * positions [score_from, n - 1) of a known document are rows. */
TEST(spans_outside_a_documents_scored_range_are_skipped) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    const std::vector<float> ra = make_logits(40, 13), rb = make_logits(25, 14);
    record(t.path + "/ref", ra, rb);
    Kld k;
    CHECK_OK(k.score_against(t.path + "/ref", "", V, "candidate"));
    /* Positions 0..9 of document a: only 8 and 9 are scored. */
    {
        std::vector<float> part(ra.begin(), ra.begin() + 10 * V);
        Shards sh(part, 10);
        CHECK_OK(k.score({ { 1, 0, 10, 0 } }, sh.s));
    }
    CHECK_EQ(k.rows_done(), (int64_t)2);
    /* Positions 35..44 of a 40-token document: 35..38 are scored, the last predicts nothing. */
    {
        std::vector<float> part(10 * V);
        std::copy(ra.begin() + 35 * V, ra.end(), part.begin());
        Shards sh(part, 10);
        CHECK_OK(k.score({ { 1, 35, 10, 0 } }, sh.s));
    }
    CHECK_EQ(k.rows_done(), (int64_t)6);
    /* No such documents. */
    {
        std::vector<float> part(ra.begin(), ra.begin() + 4 * V);
        Shards sh(part, 4);
        CHECK_OK(k.score({ { 0, 10, 4, 0 }, { 3, 10, 4, 0 } }, sh.s));
    }
    CHECK_EQ(k.rows_done(), (int64_t)6);
}

/* The report's statistics are those of its own rows: the mean, the percentiles (linear
 * interpolation between order statistics, as the report says), top-1 agreement and perplexity
 * exp(mean NLL), overall and per source. A miscounted statistic biases every comparison the
 * instrument is used for. */
TEST(the_reports_statistics_are_those_of_its_rows) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    const std::vector<float> ra = make_logits(40, 15), rb = make_logits(25, 16);
    record(t.path + "/ref", ra, rb);
    std::vector<float> ca = ra, cb = rb;
    std::mt19937 g(17);
    std::normal_distribution<float> n(0.0f, 0.8f);
    for (float& v : ca) v += n(g);
    for (float& v : cb) v += n(g);
    Kld k;
    CHECK_OK(k.score_against(t.path + "/ref", t.path + "/rep.json", V, "candidate"));
    feed(k, 1, 40, ca);
    feed(k, 2, 25, cb);
    CHECK_OK(k.finish());
    const std::vector<float> rows = read_rows(t.path + "/rep.json.rows");
    REQUIRE_EQ(rows.size(), (size_t)(55 * 4));
    std::ifstream in(t.path + "/rep.json");
    const nlohmann::json rep = nlohmann::json::parse(in, nullptr, false);
    REQUIRE(!rep.is_discarded());

    auto expect = [&](const nlohmann::json& j, size_t r0, size_t r1) {
        std::vector<double> kl;
        double sum = 0, nq = 0, np = 0, t1 = 0;
        for (size_t r = r0; r < r1; ++r) {
            kl.push_back(rows[r * 4]);
            sum += rows[r * 4];
            nq += rows[r * 4 + 1];
            np += rows[r * 4 + 2];
            t1 += rows[r * 4 + 3];
        }
        std::sort(kl.begin(), kl.end());
        const double m = (double)kl.size();
        auto pct = [&](double q) {
            const double x = q * (m - 1);
            const size_t i = (size_t)std::floor(x);
            return i + 1 >= kl.size() ? kl.back() : kl[i] + (x - (double)i) * (kl[i + 1] - kl[i]);
        };
        CHECK_EQ(j["positions"].get<int64_t>(), (int64_t)kl.size());
        CHECK_NEAR(j["kld"]["mean"].get<double>(), sum / m, 1e-6);
        CHECK_NEAR(j["kld"]["median"].get<double>(), pct(0.5), 1e-6);
        CHECK_NEAR(j["kld"]["p90"].get<double>(), pct(0.9), 1e-6);
        CHECK_NEAR(j["kld"]["p99"].get<double>(), pct(0.99), 1e-6);
        CHECK_NEAR(j["kld"]["max"].get<double>(), kl.back(), 1e-6);
        CHECK_NEAR(j["top1_agreement"].get<double>(), t1 / m, 1e-9);
        CHECK_NEAR(j["ppl"]["candidate"].get<double>(), std::exp(nq / m), 1e-4);
        CHECK_NEAR(j["ppl"]["reference"].get<double>(), std::exp(np / m), 1e-4);
    };
    expect(rep["all"], 0, 55);
    expect(rep["by_source"]["a"], 0, 31);
    expect(rep["by_source"]["b"], 31, 55);
    CHECK_EQ(rep["docs"].get<int64_t>(), (int64_t)2);
}

/* ------------------------------------------------------------------ the sparse form */

void record_top(const std::string& dir, const std::vector<float>& la, const std::vector<float>& lb, int top) {
    Kld rec;
    CHECK_OK(rec.add_doc("a", kDocA, 8));
    CHECK_OK(rec.add_doc("b", kDocB, 0));
    CHECK_OK(rec.record_into(dir, V, "reference", top));
    feed(rec, 1, 40, la);
    feed(rec, 2, 25, lb);
    CHECK_OK(rec.finish());
}

/* KL between P and Q coarsened to P's `top` most likely tokens (ties to the lower id) and one
 * bucket for the rest, in double from the f32 logits. */
double kl_coarse(const float* p, const float* q, int top) {
    const double zp = lse(p), zq = lse(q);
    std::vector<int64_t> ids((size_t)V);
    for (int64_t i = 0; i < V; ++i) ids[(size_t)i] = i;
    std::stable_sort(ids.begin(), ids.end(), [&](int64_t a, int64_t b) { return p[a] > p[b]; });
    double kl = 0, pt = 0, qt = 0;
    for (int64_t k = 0; k < V; ++k) {
        const int64_t i = ids[(size_t)k];
        const double lp = (double)p[i] - zp, lq = (double)q[i] - zq;
        if (k < top) kl += std::exp(lp) * (lp - lq);
        else { pt += std::exp(lp); qt += std::exp(lq); }
    }
    if (pt > 0) kl += pt * (std::log(pt) - std::log(qt));
    return kl;
}

/* A candidate scored against a top-64 reference gets, at every position, the coarsened KL a
 * direct double computation gives; that KL is at most the full one (data processing) and, with a
 * confident head, close to it. The manifest names the form, and the files hold no f16 rows. */
TEST(a_sparse_reference_scores_the_coarsened_kl) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    constexpr int kTop = 64;
    const std::vector<float> ra = make_logits(40, 21), rb = make_logits(25, 22);
    record_top(t.path + "/ref", ra, rb, kTop);
    CHECK(!std::filesystem::exists(t.path + "/ref/logp.f16"));
    CHECK_EQ((int64_t)std::filesystem::file_size(t.path + "/ref/top.bin"), (int64_t)(55 * kTop * 8));
    CHECK_EQ((int64_t)std::filesystem::file_size(t.path + "/ref/rows.f32"), (int64_t)(55 * 3 * 4));
    {
        std::ifstream in(t.path + "/ref/kld.json");
        const nlohmann::json m = nlohmann::json::parse(in, nullptr, false);
        CHECK_EQ(m["format"].get<std::string>(), std::string("radiance-kld-top-1"));
        CHECK_EQ(m["top"].get<int>(), kTop);
    }

    std::vector<float> ca = ra, cb = rb;
    std::mt19937 g(23);
    std::normal_distribution<float> n(0.0f, 0.6f);
    for (float& v : ca) v += n(g);
    for (float& v : cb) v += n(g);
    Kld k;
    CHECK_OK(k.score_against(t.path + "/ref", t.path + "/rep.json", V, "candidate"));
    feed(k, 1, 40, ca);
    feed(k, 2, 25, cb);
    CHECK_OK(k.finish());

    const std::vector<float> rows = read_rows(t.path + "/rep.json.rows");
    REQUIRE_EQ(rows.size(), (size_t)(55 * 4));
    int64_t r = 0, flips = 0;
    double sum_c = 0, sum_f = 0;
    auto check_doc = [&](const std::vector<float>& ref, const std::vector<float>& cand,
                         const std::vector<int32_t>& tok, int64_t from) {
        for (int64_t p = from; p + 1 < (int64_t)tok.size(); ++p, ++r) {
            const float* lp = ref.data() + p * V;
            const float* lq = cand.data() + p * V;
            const double want = kl_coarse(lp, lq, kTop), full = kl_direct(lp, lq);
            CHECK_NEAR(rows[(size_t)(r * 4 + 0)], want, 1e-6 + 1e-5 * want);
            CHECK(want <= full + 1e-9);
            sum_c += want;
            sum_f += full;
            CHECK_NEAR(rows[(size_t)(r * 4 + 1)], lse(lq) - lq[tok[(size_t)p + 1]], 1e-4);
            CHECK_NEAR(rows[(size_t)(r * 4 + 2)], lse(lp) - lp[tok[(size_t)p + 1]], 1e-4);
            const bool same = argmax(lp) == argmax(lq);
            CHECK_EQ(rows[(size_t)(r * 4 + 3)], same ? 1.0f : 0.0f);
            flips += !same;
        }
    };
    check_doc(ra, ca, kDocA, 8);
    check_doc(rb, cb, kDocB, 0);
    CHECK_EQ(r, (int64_t)55);
    CHECK(flips > 0);
    CHECK(sum_c > 0.5 * sum_f);     /* the head carries most of the divergence */
    std::ifstream in(t.path + "/rep.json");
    const nlohmann::json rep = nlohmann::json::parse(in, nullptr, false);
    CHECK_EQ(rep["ref_form"].get<std::string>(), std::string("top-64 + tail"));
}

/* Control: the reference against its own sparse form scores zero at every position. A head that
 * covers the whole vocabulary leaves an empty tail, which must cost nothing. */
TEST(the_reference_against_its_sparse_self_scores_zero) {
    for (int top : { 64, (int)V }) {
        TmpDir t;
        REQUIRE(!t.path.empty());
        const std::vector<float> ra = make_logits(40, 24), rb = make_logits(25, 25);
        record_top(t.path + "/ref", ra, rb, top);
        Kld k;
        CHECK_OK(k.score_against(t.path + "/ref", t.path + "/rep.json", V, "same"));
        feed(k, 1, 40, ra);
        feed(k, 2, 25, rb);
        CHECK_OK(k.finish());
        const std::vector<float> rows = read_rows(t.path + "/rep.json.rows");
        REQUIRE_EQ(rows.size(), (size_t)(55 * 4));
        for (size_t r = 0; r < 55; ++r) {
            CHECK(std::fabs(rows[r * 4 + 0]) < 1e-6f);
            CHECK_EQ(rows[r * 4 + 3], 1.0f);
        }
    }
}

/* The recorded tail is the mass outside the head summed from the row, so a confident position's
 * tail is its true small value, not the noise one minus the head would leave. */
TEST(a_sparse_reference_records_the_tail_mass_from_the_row) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    constexpr int kTop = 16;
    std::vector<float> ra = make_logits(40, 26), rb = make_logits(25, 27);
    for (int64_t r = 0; r < 25; ++r) rb[(size_t)(r * V + 3)] += 40.0f;   /* p_tail ~ e^-40 */
    record_top(t.path + "/ref", ra, rb, kTop);
    const std::vector<float> rows = read_rows(t.path + "/ref/rows.f32");
    REQUIRE_EQ(rows.size(), (size_t)(55 * 3));
    auto tail = [&](const float* l) {
        const double z = lse(l);
        std::vector<double> lp((size_t)V);
        for (int64_t i = 0; i < V; ++i) lp[(size_t)i] = (double)l[i] - z;
        std::sort(lp.begin(), lp.end(), std::greater<double>());
        double s = 0;
        for (int64_t i = kTop; i < V; ++i) s += std::exp(lp[(size_t)i]);
        return s;
    };
    for (int64_t p = 8; p < 39; ++p) {
        const double want = tail(ra.data() + p * V);
        CHECK_NEAR(rows[(size_t)((p - 8) * 3 + 2)], want, 1e-5 * want);
    }
    for (int64_t p = 0; p < 24; ++p) {
        const double want = tail(rb.data() + p * V);
        CHECK(want < 1e-9);    /* below what one minus the head resolves in f32 */
        CHECK_NEAR(rows[(size_t)((31 + p) * 3 + 2)], want, 1e-3 * want);
    }
}

/* A head larger than the vocabulary, or past the cap, is refused when the recording opens. */
TEST(a_sparse_head_must_fit_the_vocabulary) {
    TmpDir t;
    REQUIRE(!t.path.empty());
    Kld k;
    CHECK_OK(k.add_doc("a", kDocA, 8));
    CHECK_EQ(k.record_into(t.path + "/a", V, "x", (int)V + 1), RAD_E_INVAL);
    CHECK_EQ(k.record_into(t.path + "/b", V, "x", -1), RAD_E_INVAL);
}

RAD_TEST_MAIN()
