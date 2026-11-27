/* draft.cpp -- the static draft-depth policy and the adaptive one (`--draft-window adaptive`).
 * See draft.h.
 */
#include "draft.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace rad {

namespace {

/* The prior weighs as much as this many verifies. The (n+1)-th verify moves an estimate by
 * 1 / (n + k + 1), a running mean over prior and evidence, until that falls below the half-life's
 * step and the moving average takes over. Without it a sequence would spend its first ~16
 * verifies at the prior's window, most of a short answer. */
constexpr double kPriorWeight = 4.0;

/* The weight of the (n+1)-th sample in an estimate that is a running mean until 1 / (n + 1)
 * falls below a moving average's step at half-life `h` samples, and that average after. */
double settle_weight(double h, int64_t n) {
    return std::fmax(1.0 - std::exp2(-1.0 / h), 1.0 / (double)(n + 1));
}

}  /* namespace */

int DraftController::configure(const Params& p) {
    if (p.max_depth < 0 || p.max_depth > RAD_SCHED_MAX_SPEC) {
        RAD_ERR("draft depth %d is outside 0..%d -- the verify kernel band is %d query rows",
                p.max_depth, RAD_SCHED_MAX_SPEC, RAD_SCHED_MAX_SPEC);
        return RAD_E_INVAL;
    }
    p_ = p;
    tot_drafted_ = tot_accepted_ = 0;
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) { reach_[j] = 0; hit_[j] = 0; }
    seen_ = 0;

    adapt_ = p.adapt && p.max_depth > 0;
    alpha_ = 1.0 - std::exp2(-1.0 / kHalfLife);
    step_d_ = 0;
    step_w_ = 0;
    step_held_ = 0;
    prior_n_ = 0;
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) prior_[j] = 0.0f;
    for (int d = 0; d <= RAD_SCHED_MAX_SPEC; ++d) by_depth_[d] = 0;
    cost_.reset(kRowCost);
    timed_ = 0;
    /* Depth 0 (draft.h): reset whether or not it is on, so off it never leaves these values. */
    depth0_ = p.depth0;
    probe_every_ = p.probe_every > 0 ? p.probe_every : kProbeEvery;
    zero_ = false;
    zero_held_ = kHold;
    probe_left_ = 0;
    probe_d_ = 0;
    since_probe_ = 0;
    zero_run_ = 0;
    cycles_ = 0;
    plain_.reset(kRowCost);
    for (int c = 0; c < kCostClasses; ++c) {
        plain_n_[c] = plain_at_[c] = 0;
        dn_[c] = dtn_[c] = 0;
        dtok_[c] = dms_[c] = 0.0;
    }
    plain_probe_at_ = -(int64_t)RAD_SCHED_MAX_SPEC * 4096;
    n_zero_ = n_probe_ = n_zero_moves_ = 0;
    set_backoff(p.backoff);
    if (adapt_)
        RAD_INFO("draft depth: adaptive, %d..%d; a sequence stops at the first draft emitted less "
                 "than %.0f%% of the time, and the depth moves at +%.0f%% at most every %d steps",
                 depth0_ ? 0 : kMinDepth, p_.max_depth, 100.0 * kDraftMinChance, 100.0 * kHyst,
                 kHold);
    if (adapt_ && depth0_)
        RAD_INFO("draft depth 0: a step drafts nothing and verifies nothing where a plain step, "
                 "measured, emits %.0f%% more tokens a millisecond than the window; at 0, %d "
                 "cycles of the drafting depth every %d keep its estimates measured",
                 100.0 * kHyst, kProbeSteps, probe_every_);
    if (adapt_ && depth0_ && backoff_max_ > 0)
        RAD_INFO("draft probe back-off: each probe that leaves depth 0 in place doubles the "
                 "interval to the next, up to %d cycles; a move between 0 and drafting starts it "
                 "over", (int)std::min<int64_t>((int64_t)probe_every_ << backoff_max_, 4096));
    return RAD_OK;
}

void DraftController::observe(int drafted, int accepted) {
    if (drafted <= 0) return;
    if (drafted > RAD_SCHED_MAX_SPEC) drafted = RAD_SCHED_MAX_SPEC;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;
    tot_drafted_  += drafted;
    tot_accepted_ += accepted;

    /* Position j is evaluated only if every earlier position was accepted: 0..accepted-1 are
     * evaluated and accepted, position `accepted` (if drafted) evaluated and rejected, and nothing
     * past it is reached. */
    const int last = accepted < drafted ? accepted : drafted - 1;
    for (int j = 0; j <= last; ++j) {
        ++reach_[j];
        if (j < accepted) ++hit_[j];
    }
    if (drafted > seen_) seen_ = drafted;
    if (adapt_) ++by_depth_[drafted];
}

/* ------------------------------------------------------------------ the step's cost */
void DraftController::step_time(int q_rows, double ms, int n_rows, double tokens) {
    if (!adapt_) return;
    cost_.sample(q_rows, ms);
    if (q_rows < 1 || !(ms > 0.0) || !std::isfinite(ms)) return;
    ++timed_;
    /* Depth 0's view of the drafting steps, per row class (draft.h). */
    if (n_rows < 1) return;
    const int c = cost_class(n_rows);
    ++dn_[c];
    if (tokens > 0.0 && std::isfinite(tokens)) {
        double t = ms;
        if (dtn_[c] >= StepCost::kClipAfter && t > StepCost::kOutlier * dms_[c])
            t = StepCost::kOutlier * dms_[c];
        const double w = settle_weight(StepCost::kHalfLife, dtn_[c]);
        dtok_[c] = dtn_[c] ? dtok_[c] + w * (tokens - dtok_[c]) : tokens;
        dms_[c]  = dtn_[c] ? dms_[c] + w * (t - dms_[c]) : t;
        ++dtn_[c];
    }
}

void DraftController::plain_time(int n_rows, double ms) {
    if (!adapt_ || !depth0_ || n_rows < 1 || !(ms > 0.0) || !std::isfinite(ms)) return;
    /* A plain step still warming up (kPlainWarm) is no sample. */
    if (zero_run_ <= 1 + kPlainWarm) return;
    const int c = cost_class(n_rows);
    plain_.sample(n_rows, ms);
    ++plain_n_[c];
    plain_at_[c] = cycles_;
}

int DraftController::cost_class(int n_rows) {
    int c = 0;
    for (int r = n_rows; r > 1 && c < kCostClasses - 1; r >>= 1) ++c;
    return c;
}

int64_t DraftController::cost_samples(int q_rows) const {
    return q_rows >= 1 ? cost_.samples(q_rows) : 0;
}

double DraftController::step_cost(int q_rows) const { return cost_.cost(q_rows); }

/* ------------------------------------------------------------------ one sequence */
int DraftController::stop_of(const float* a) const {
    double p = 1.0;
    int d = 0;
    while (d < p_.max_depth) {
        p *= (double)a[d];
        if (p < kDraftMinChance) break;
        ++d;
    }
    return d < kMinDepth ? kMinDepth : d;
}

double DraftController::prior_at(int j) const {
    if (j < 0 || j >= RAD_SCHED_MAX_SPEC) return 0.0;
    return prior_n_ >= kPriorMinSeqs ? (double)prior_[j] : kPrior;
}

void DraftController::start(DraftAdapt& s) const {
    /* From finished sequences once there are enough (adapt_retire), else the fixed kPrior. */
    for (int j = 0; j < p_.max_depth; ++j) {
        s.a[j] = (float)prior_at(j);
        s.n[j] = 0;
    }
    s.depth = stop_of(s.a);
    s.held  = 0;
}

void DraftController::adapt_retire(const DraftAdapt& s) {
    if (!adapt_ || s.depth <= 0 || (double)s.n[0] < kHalfLife) return;
    const double w = settle_weight(kPriorHalfLifeSeqs, prior_n_);
    for (int j = 0; j < p_.max_depth; ++j)
        prior_[j] = prior_n_ ? (float)((double)prior_[j] + w * ((double)s.a[j] - (double)prior_[j]))
                             : s.a[j];
    ++prior_n_;
}

double DraftController::yield(const DraftAdapt& s, int d) const {
    if (d > p_.max_depth) d = p_.max_depth;
    double e = 1.0, run = 1.0;
    for (int k = 1; k <= d; ++k) { run *= (double)s.a[k - 1]; e += run; }
    return e;
}

int DraftController::window_of(const DraftAdapt& s) const {
    if (!adapt_) return p_.max_depth;
    if (s.depth <= 0 || zero_now()) return 0;
    const int d = step_depth();
    const int D = d > 0 ? d : p_.max_depth;
    return s.depth < D ? s.depth : D;
}

int DraftController::window_for(DraftAdapt& s) const {
    if (!adapt_) return next_depth();
    if (s.depth <= 0) start(s);
    return window_of(s);
}

int DraftController::adapt_observe(DraftAdapt& s, int drafted, int accepted) const {
    if (!adapt_) return p_.max_depth;
    if (s.depth <= 0) start(s);
    if (drafted <= 0) return s.depth;
    const int D = p_.max_depth;
    if (drafted > D) drafted = D;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;

    /* The positions this verify evaluated, by observe()'s rule. Positions past the first
     * rejection were never evaluated and are not updated here. */
    const int last = accepted < drafted ? accepted : drafted - 1;
    for (int j = 0; j <= last; ++j) {
        const double w = std::fmax(alpha_, 1.0 / ((double)s.n[j] + kPriorWeight + 1.0));
        const double x = j < accepted ? 1.0 : 0.0;
        s.a[j] = (float)((double)s.a[j] + w * (x - (double)s.a[j]));
        if (s.n[j] < UINT16_MAX) ++s.n[j];
    }
    /* Positions past the window drift toward its deepest one. Otherwise a sequence held shallow
     * by prose that turns into code would keep its stale deep estimates and never draft deep
     * enough to update them. Assumes acceptance decays smoothly with position; when wrong it costs
     * a move out and back. */
    const double edge = (double)s.a[drafted - 1];
    for (int j = drafted; j < D; ++j)
        s.a[j] = (float)((double)s.a[j] + alpha_ * (edge - (double)s.a[j]));

    /* The stop moves only past the hold: it is this sequence's share of the step's shape. */
    if (s.held < INT32_MAX) ++s.held;
    if (s.held < kHold) return s.depth;
    const int d = stop_of(s.a);
    if (d != s.depth) {
        s.depth = d;
        s.held  = 0;
    }
    return s.depth;
}

/* ------------------------------------------------------------------ the step */
int DraftController::plan_step(DraftAdapt* const* seq, int n_seq, int n_rows) {
    if (!adapt_ || n_seq <= 0) return step_d_;
    if (n_rows < n_seq) n_rows = n_seq;
    const int M = p_.max_depth;

    /* Expected tokens at each head depth, sum_i E_i(min(D, stop_i)) plus one per non-drafting
     * row, and the drafts the step would then verify, sum_i min(D, stop_i). Candidates end at the
     * deepest stop, since a deeper D changes nothing. */
    double tok[RAD_SCHED_MAX_SPEC + 1] = {};
    int    drafts[RAD_SCHED_MAX_SPEC + 1] = {};
    int top = kMinDepth;
    for (int i = 0; i < n_seq; ++i) {
        DraftAdapt& s = *seq[i];
        if (s.depth <= 0) start(s);
        if (s.depth > top) top = s.depth;
        double e = 1.0, run = 1.0;
        for (int d = 1; d <= M; ++d) {
            if (d <= s.depth) { run *= (double)s.a[d - 1]; e += run; }
            tok[d] += e;
            drafts[d] += d <= s.depth ? d : s.depth;
        }
    }
    const double plain = (double)(n_rows - n_seq);
    if (top > M) top = M;

    /* A step's query rows at depth d: a row a decode sequence and its drafts. */
    auto gain = [&](int d) { return (tok[d] + plain) / step_cost(n_rows + drafts[d]); };
    int best = kMinDepth;
    double best_g = -1.0;
    for (int d = kMinDepth; d <= top; ++d) {
        const double g = gain(d);
        if (g > best_g) { best_g = g; best = d; }
    }

    /* Move only after kHold steps and only for kHyst more yield than the current depth. The
     * first drafting step takes the best outright. */
    if (step_d_ <= 0) {
        step_d_ = best;
        step_held_ = 0;
    } else {
        if (step_held_ < INT32_MAX) ++step_held_;
        const int cur = step_d_ < top ? step_d_ : top;
        const double cur_g = gain(cur);
        if (best != cur && step_held_ >= kHold && best_g > cur_g * (1.0 + kHyst)) {
            RAD_DEBUG("draft window: head depth %d -> %d over %d drafting of %d rows (%.3f -> "
                      "%.3f tokens per unit of step cost)", step_d_, best, n_seq, n_rows, cur_g,
                      best_g);
            step_d_ = best;
            step_held_ = 0;
        }
    }
    step_w_ = step_d_ < top ? step_d_ : top;
    if (depth0_) {
        plan_zero(tok, drafts, plain, n_rows);
        zero_run_ = zero_now() ? (zero_run_ < INT32_MAX ? zero_run_ + 1 : zero_run_) : 0;
    }
    return step_depth();
}

/* Depth 0, after the drafting depth is chosen: whether this cycle runs at D = 0 instead. The
 * drafting depth step_w_ is the window the step would verify; its yield per millisecond is weighed
 * against the measured plain step's, with the same hold and hysteresis as a move of D. Both sides
 * are priced on the row-count costs (StepCost), not on a separate table per depth and row class. */
void DraftController::plan_zero(const double* tok, const int* drafts, double plain, int n_rows) {
    ++cycles_;
    if (zero_held_ < INT32_MAX) ++zero_held_;
    const int c  = cost_class(n_rows);
    const int dd = step_w_ > 0 ? step_w_ : kMinDepth;

    /* A probe runs its cycles out first, and the decision below reads what it measured. */
    bool probed = false;   /* a probe at the drafting depth ran out at this cycle */
    if (probe_left_ > 0) {
        if (--probe_left_ > 0) {
            if (probe_d_ > 0) probe_d_ = dd;
            ++n_probe_;
            return;
        }
        probed = probe_d_ > 0;
    }

    /* The hold counts every cycle, probes included, and starts spent: the first measured plain
     * step decides at once, as the first drafting step takes the best depth outright. Both sides
     * in ms: a plain step measured in this row class, and drafting steps timed. */
    const bool measured = plain_n_[c] > 0 && timed_ > 0;
    bool moved = false;
    if (measured) {
        /* The drafting side is the lower of the model's yield at the drafting depth and the
         * yield drafting steps measured here: E(D) comes from estimates a new sequence starts
         * optimistic, and a plain step's yield is n_rows a step by construction. Erring toward 0
         * corrects itself, since at 0 every plain step is a sample and every probe a drafting
         * one; erring toward drafting does not, since while drafting only plain probes sample
         * the plain step. */
        double g_d = (tok[dd] + plain) / step_cost(n_rows + drafts[dd]);
        if (dtn_[c] >= StepCost::kClipAfter && dms_[c] > 0.0)
            g_d = std::fmin(g_d, dtok_[c] / dms_[c]);
        const double g_0 = (double)n_rows / plain_.cost(n_rows);
        const bool want = zero_ ? !(g_d > g_0 * (1.0 + kHyst)) : g_0 > g_d * (1.0 + kHyst);
        if (want != zero_ && zero_held_ >= kHold) {
            RAD_DEBUG("draft window: head depth %d -> %d over %d rows (%.3f -> %.3f tokens a ms)",
                      zero_ ? 0 : dd, want ? 0 : dd, n_rows, want ? g_d : g_0, want ? g_0 : g_d);
            zero_ = want;
            zero_held_ = 0;
            since_probe_ = 0;
            ++n_zero_moves_;
            backoff_k_ = 0;
            moved = true;
        }
    }
    /* Probe back-off: a probe that ran out with 0 still chosen doubles the interval to the next. */
    if (probed && !moved && zero_ && backoff_k_ < backoff_max_) ++backoff_k_;

    if (zero_) {
        /* At 0: every probe_interval() cycles (probe_every, or more under the back-off),
         * kProbeSteps at the drafting depth. */
        if (++since_probe_ >= probe_interval()) {
            since_probe_ = 0;
            probe_d_ = dd;
            probe_left_ = kProbeSteps;
            ++n_probe_;
            return;
        }
        ++n_zero_;
        return;
    }
    /* Drafting: a plain probe where this class has no plain step measured, or only an old one,
     * once drafting steps are measured here (otherwise the comparison has nothing to weigh it
     * against). Its first kPlainWarm plain steps are no samples; it takes 3 the first time in a
     * class and 1 at a refresh. At most one every probe_every cycles: a probe whose steps all
     * carried a prefill chunk measured nothing, and must not be followed by another at once. */
    const bool stale = plain_n_[c] <= 0 ||
                       cycles_ - plain_at_[c] > (int64_t)kPlainStale * probe_every_;
    if (stale && dn_[c] >= StepCost::kClipAfter && cycles_ - plain_probe_at_ >= probe_every_) {
        plain_probe_at_ = cycles_;
        probe_d_ = 0;
        probe_left_ = 1 + kPlainWarm + (plain_n_[c] > 0 ? 1 : 3);
        ++n_probe_;
    }
}

}  /* namespace rad */
