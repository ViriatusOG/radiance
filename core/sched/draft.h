/* draft.h -- the speculative draft-depth controller.
 *
 * Static by default: the depth is `--num-speculative-tokens`. A dynamic policy that gates on the
 * drafter's own top-1 confidences cannot work here, because the scheduler picks the depth before
 * the drafter runs; measured acceptance arrives a step late and pooled across sequences. With a
 * draft head whose cost is nearly flat in depth (one extra verify row on a weight-bound step), the
 * best depth is a property of the model pair and is known up front, so it is a flag. The
 * controller still keeps acceptance accounting, a reported metric (spec §14).
 *
 * `--draft-window adaptive` is for when that premise fails: with streamed weights (a MoE reading
 * experts per verify row, a serial head running one pass per round) each draft costs a head round
 * and a verify row, and the best depth depends on the text (deep on code and JSON, shallow on
 * prose). The adaptive policy has two halves:
 *
 *   Per sequence (DraftAdapt): a smoothed estimate of each draft position's conditional
 *   acceptance, fed by the verify outcomes, and the sequence's stop -- the deepest draft whose
 *   chance of being emitted still pays for its verify row (kDraftMinChance).
 *
 *   Per step (plan_step): the head depth D. A serial head runs as many rounds as the deepest
 *   window in the step, so D maximises sum_i E_i(min(D, stop_i)) / T(R(D)), with
 *   E(d) = 1 + sum_k prod_{j<=k} a_j, R(D) the query rows the step would then carry (a row a
 *   sequence plus sum_i min(D, stop_i)) and T measured online by row count (step_time,
 *   step_cost.h). Each sequence then drafts min(D, stop_i).
 *
 * Both move only past a hold, and D also past a hysteresis, because a new window is a new pass
 * shape that must be recorded before it plays (runtime/ctx.h). `max_depth` stays the ceiling and
 * everything is sized for it.
 *
 * D pays for serial head rounds; a block drafter proposes its block in one pass and prompt lookup
 * runs none, so for those the policy prices only the verify rows (Scheduler::init): each sequence
 * verifies the first min(D, its stop) of its proposal, which on a MoE that reads each row's routed
 * experts is still the cost that matters.
 *
 * Depth 0 (Params::depth0, --draft-depth0) lets the step choose D = 0 as well: no head pass and
 * no verify row, each sequence emits one token, so the step is a plain decode step (a block
 * drafter's context pass still runs: it fills the drafter's window cache, which a hole would leave
 * unusable for a window). Under load a verify row costs about what it buys -- each row reads its
 * own routed experts -- and a plain step can emit more tokens a millisecond than the best window.
 * A plain step has no head round, so it is priced on a cost of its own (plain_, by decode rows,
 * fed by plain_time), never on the drafting steps' T(R), and never on a prior: D = 0 is taken only
 * where the measured plain step beats the drafting depth's yield by kHyst, with D's hold. The
 * drafting side is the lower of the estimates' E(D) / T(R(D)) and the tokens a millisecond
 * drafting steps in this row class actually emitted (step_time's tokens), since a new sequence's
 * estimate starts optimistic. While D is 0 nothing is verified, so nothing updates the acceptance
 * estimates or the drafting steps' times; a probe (kProbeSteps cycles at the drafting depth every
 * probe_every cycles at 0, --draft-probe) keeps both alive, and a plain probe (cycles at 0, its
 * first kPlainWarm plain steps not sampled) measures the plain step of a row class that has none,
 * or whose last is older than kPlainStale probe intervals.
 *
 * Probe back-off (Params::backoff, --draft-probe-backoff N): each probe at the drafting depth that
 * ends with depth 0 still winning doubles the interval to the next, up to probe_every << N; a move
 * between 0 and drafting starts it over at probe_every. A long run at 0 then pays few probes, at
 * the price of finding a turn into drafting text up to that many cycles late.
 *
 * Off (the default): depth 0's state never leaves its reset values, and step_depth(),
 * step_window(), window_of() and head_runs() follow the drafting policy alone.
 *
 * It changes only how many drafts a step verifies. Acceptance follows core/sample/accept.h, so
 * every emitted token is the target's own; at temperature 0 the text matches a plain run as far
 * as the logits do, and a verify row's logits are computed at an M that moves with the window
 * (see accept.h).
 */
#pragma once
#include "rad_core.h"
#include "step_cost.h"

namespace rad {

/* libr4d's decode kernel serves up to 64 query rows, the verify-step band (spec §10), so 64 is
 * the largest window any batch can carry. */
enum { RAD_SCHED_MAX_SPEC = 64 };

/* One sequence's half of the adaptive policy: the drafter's record on this sequence's text and the
 * resulting stop. Stored in the scheduler's per-request state; the controller does the arithmetic.
 *
 * `depth` 0 means not started: the first ask fills the estimate from the prior and sets the stop.
 * It belongs to the request, not its cache: preemption keeps it, a recycled slot resets it. */
struct DraftAdapt {
    int32_t  depth = 0;                       /* the sequence's stop; 0 = not started */
    int32_t  held  = 0;                       /* verifies since `depth` last moved */
    float    a[RAD_SCHED_MAX_SPEC] = {};      /* conditional acceptance at each draft position */
    uint16_t n[RAD_SCHED_MAX_SPEC] = {};      /* verifies that evaluated each one, saturating */
};

class DraftController {
public:
    struct Params {
        int  max_depth = 0;       /* the deployment's window; 0 disables speculation entirely */
        /* The head's own rounds when the window is wider than they are (--draft-suffix: a suffix
         * step verifies more rows than the head drafts); 0 = max_depth. */
        int  head_depth = 0;
        /* The adaptive policy: `--draft-window adaptive` with a serial drafter, decided by
         * Scheduler::init. Off, the depth is max_depth. */
        bool adapt = false;
        /* Depth 0 (see the top of this file): the adaptive policy may also choose D = 0. Inert
         * while the policy is off. */
        bool depth0 = false;
        /* Cycles at D = 0 between probes; 0 is kProbeEvery. */
        int  probe_every = 0;
        /* Probe back-off: at most this many doublings of the interval (0 off, 1..kBackoffMax). */
        int  backoff = 0;
    };

    /* ---------------------------------------------------------- the policy's constants
     * Not configurable: each is a property of the policy, not of a deployment. The one
     * deployment property, step cost, is measured (step_time). */

    /* A window moves only for 2% more yield. One draft more or less moves E/T by several percent,
     * while a smoothed estimate wobbles about 1% per verify; a lower threshold would flip D
     * between near-equal neighbours and pay a recording each time. */
    static constexpr double kHyst = 0.02;
    /* And not within 8 verifies (steps, for D) of the last move. A pass shape is recorded on its
     * second sighting and played from the third (runtime/ctx.h), so a shape held 8 steps mostly
     * plays from the recording; 8 verifies at kHalfLife move an estimate about a third of the way
     * to a new rate, a trend rather than one rejection. */
    static constexpr int kHold = 8;
    /* Each position's estimate has a half-life of 16 verifies that evaluate it (about 50 tokens of
     * text): short enough to follow a reply from prose into code and back, long enough that one
     * rejection moves an estimate by about 4%. */
    static constexpr double kHalfLife = 16.0;
    /* An unmeasured position starts at 0.8, optimistic on purpose: a window that starts deep
     * learns every position, one that starts shallow never learns the deep ones. 0.8 lies between
     * typical first-position acceptance on prose and on code, so a wrong start corrects within a
     * hold. */
    static constexpr double kPrior = 0.8;
    /* Once 4 sequences have finished, new ones start from the mean of their final estimates
     * (adapt_retire): a running mean, then a moving average with a half-life of 32 sequences, so
     * the prior follows a change of traffic within a few dozen requests. A sequence counts only
     * after kHalfLife verifies; before that its estimate is mostly the prior itself. */
    static constexpr int    kPriorMinSeqs = 4;
    static constexpr double kPriorHalfLifeSeqs = 32.0;
    /* Never fewer than one draft: at zero no position is evaluated, so the estimate could never
     * recover. */
    static constexpr int kMinDepth = 1;
    /* A sequence stops at the first draft whose chance of being emitted, prod_{j<=d} a_j, is
     * under 1 in 10. Once the head runs D rounds for the deepest neighbour, a sequence's only
     * added cost is its verify row, and a draft pays while p_d / E exceeds the share of the step
     * that row adds. That share is a few percent at any load and E is two to three tokens a
     * sequence, so this is never stricter than D's own trade; it only keeps a sequence that
     * accepts little from riding its neighbours' rounds with rows that are almost never emitted. */
    static constexpr double kDraftMinChance = 0.1;
    /* The shape an untimed row count is priced on (StepCost): each query row adds 12% of a
     * one-row step, a verify row of streamed experts and its share of a head round. Qwen3.8
     * Flash-Next on two R9700s, ms a step at 3 / 4 / 5 rows of one sequence: 11.2 / 12.3 / 13.6.
     * Only ratios are used, and a timed row count replaces it. */
    static constexpr double kRowCost = 0.12;
    /* Depth 0's bookkeeping is kept per decode-row class: 1, 2-3, 4-7, 8-15, 16+ rows. Whether
     * the plain step and the drafting steps are measured, and how long ago, is a property of the
     * load; the prices themselves come from the row-count costs. */
    static constexpr int kCostClasses = 5;
    /* Depth 0's probes. Every 16 cycles at D = 0 the step drafts at the drafting depth for 2
     * cycles: the first proposes (its step verifies nothing), the second verifies and proposes, so
     * its step time is a full drafting step's, and the step after it verifies once more. At
     * DeepSeek-V4.1's cost ratio at four sequences (a depth-5 adaptive step about 73 ms for 8.8
     * tokens, a plain one about 28 ms for 4) a probe costs about 15 % of two steps, about 1.7 %
     * over 18 cycles. Each probe moves a sequence's estimate by two verifies, so a turn from prose into
     * a copied block shows within a few probes. */
    static constexpr int kProbeEvery = 16;
    static constexpr int kProbeSteps = 2;
    /* A class's plain step is measured again when its last sample is older than 16 probe
     * intervals (256 cycles; at one sequence 2 slower steps in 256, under 1 %): a context that
     * grows changes the plain step and the window's alike, but only the window's is refreshed by
     * the probes from D = 0. */
    static constexpr int kPlainStale = 16;
    /* The first plain steps after the head stops are not samples of a plain step: the first two
     * sightings of a pass shape run eagerly and record (runtime/ctx.h), several times slower than
     * a played one, and the step right after a move still carries the transition. So the first
     * kPlainWarm plain steps of every run at 0 are left out, and a plain probe is 1 + kPlainWarm
     * cycles longer than the samples it takes: 3 the first time in a class, 1 at a refresh. */
    static constexpr int kPlainWarm = 2;
    /* Probe back-off's ceiling: 8 doublings of 16 cycles is 4096, the most --draft-probe
     * takes. */
    static constexpr int kBackoffMax = 8;

    int  configure(const Params& p);
    const Params& params() const { return p_; }

    /* The draft depth for the next step; 0 means do not speculate. Constant: under the adaptive
     * policy it is the ceiling, and window_for() is the window. */
    int  next_depth() const { return p_.head_depth > 0 ? p_.head_depth : p_.max_depth; }

    /* ---------------------------------------------------------- the adaptive policy */
    bool adaptive() const { return adapt_; }

    /* ---------------------------------------------------------- depth 0 (see the top)
     * Set at load from Params; the setters are for tests. Off returns to the drafting depth at
     * once and ends a probe; the measurements stay. */
    bool depth0() const { return depth0_; }
    void set_depth0(bool on) {
        if (on && !depth0_) zero_held_ = kHold;
        depth0_ = on;
        if (!on) { zero_ = false; probe_left_ = 0; since_probe_ = 0; backoff_k_ = 0; }
    }
    int  probe_every() const { return probe_every_; }
    void set_probe_every(int n) { probe_every_ = n > 0 ? n : kProbeEvery; }
    /* Probe back-off: the most doublings, 0 off; a change starts over. */
    int  backoff() const { return backoff_max_; }
    void set_backoff(int n) {
        backoff_max_ = n < 0 ? 0 : n > kBackoffMax ? kBackoffMax : n;
        backoff_k_ = 0;
    }
    int  backoff_k() const { return backoff_k_; }
    /* Cycles at 0 between probes now: probe_every << backoff_k, at most 4096. */
    int  probe_interval() const {
        if (backoff_k_ <= 0) return probe_every_;
        const int64_t p = (int64_t)probe_every_ << backoff_k_;
        return p > 4096 ? 4096 : (int)p;
    }
    /* Whether the head drafts this cycle: false only while depth 0 holds D at 0, or a plain probe
     * runs. A block drafter's context pass runs either way; its query pass and every verify row
     * do not. With depth 0 off it is next_depth() > 0. */
    bool head_runs() const { return p_.max_depth > 0 && !zero_now(); }
    /* Depth 0's counters: cycles planned at D = 0, cycles of probes, and moves between 0 and the
     * drafting depth. For tests and metrics. */
    int64_t zero_cycles() const { return n_zero_; }
    int64_t probe_cycles() const { return n_probe_; }
    int64_t zero_moves() const { return n_zero_moves_; }
    /* The window this sequence may draft into: min(D, its stop), starting its estimate on the
     * first ask; max_depth when the policy is off. Before any step chooses D, D is the ceiling. */
    int  window_for(DraftAdapt& s) const;
    /* As window_for() but never starts an estimate; 0 if not started. For reporting. */
    int  window_of(const DraftAdapt& s) const;
    /* One verify of this sequence: `drafted` proposed, a prefix of `accepted` taken. Updates the
     * estimate, and the stop once past the hold. Returns the stop. */
    int  adapt_observe(DraftAdapt& s, int drafted, int accepted) const;
    /* A finished sequence: its estimate joins the prior that new sequences start from.
     *
     * Per sequence, not per verify. Under this policy, pooled accepted(j) / reached(j) is biased:
     * only well-predicted texts get windows deep enough to measure deep positions. Here each
     * finished sequence counts once at every position, using its edge extrapolation where it never
     * reached (adapt_observe), so shallow texts pull deep positions down as much as deep ones hold
     * them up. */
    void adapt_retire(const DraftAdapt& s);
    /* The prior a sequence starting now is given at position `j`. For tests and logs. */
    double prior_at(int j) const;

    /* D for the step being opened. `seq` holds the estimates of the `n_seq` drafting sequences
     * (started here if needed), `n_rows` all decode rows (the rest emit one token and draft
     * nothing). Moves D past the hold and hysteresis; with nothing to draft, D stays. Returns D. */
    int  plan_step(DraftAdapt* const* seq, int n_seq, int n_rows);
    /* The step depth in force: 0 before the first step chose one, and while depth 0 holds D at 0
     * (head_runs() tells the two apart). */
    int  step_depth() const {
        if (zero_now()) return 0;
        return probe_left_ > 0 && probe_d_ > 0 ? probe_d_ : step_d_;
    }
    /* The deepest window a step at depth D carries: D capped at the deepest stop it was chosen
     * over; the ceiling before the first choice or with the policy off. Proposals from outside
     * the head (prompt lookup) are held to it, so they add no deeper pass shape. */
    int  step_window() const {
        if (zero_now()) return 0;
        return adapt_ && step_w_ > 0 ? step_w_ : p_.max_depth;
    }

    /* T(R): a drafting decode step of `q_rows` query rows (every sequence's verified token and
     * its drafts) took `ms` since the end of the previous step, the head's rounds included. Fed
     * by Scheduler::step_finished; adaptive policy only. `n_rows` (the step's decode rows) and
     * `tokens` (what it emitted), when given, also feed depth 0's measured drafting yield, which
     * nothing reads while depth 0 is off. */
    void step_time(int q_rows, double ms, int n_rows = 0, double tokens = -1.0);
    /* A plain step at D = 0 of `n_rows` decode rows took `ms`: depth 0's price. Only under depth
     * 0, and not for the first kPlainWarm plain steps of a run at 0. */
    void plain_time(int n_rows, double ms);
    /* Plain steps measured at a row count. */
    int64_t plain_samples(int n_rows) const { return n_rows >= 1 ? plain_.samples(n_rows) : 0; }
    /* Expected cost of a step of `q_rows` query rows (StepCost::cost): ms once anything is timed,
     * else the unitless prior. */
    double step_cost(int q_rows) const;
    /* Samples behind a row count: for tests and logs. */
    int64_t cost_samples(int q_rows) const;

    /* E(d) for `s`'s current estimate: expected tokens emitted by a verify of d drafts, the
     * verified token included. For tests and logs. */
    double yield(const DraftAdapt& s, int d) const;
    /* Histogram of verifies by window, counted only under the adaptive policy. */
    int64_t verified_at(int d) const {
        return (d >= 0 && d <= RAD_SCHED_MAX_SPEC) ? by_depth_[d] : 0;
    }

    /* One verify's outcome: `drafted` proposed, a prefix of `accepted` taken. Metrics only. */
    void observe(int drafted, int accepted);

    /* Metrics (spec §14). Cumulative and exact. */
    int64_t drafted()  const { return tot_drafted_; }
    int64_t accepted() const { return tot_accepted_; }
    float   acceptance() const {
        return tot_drafted_ ? (float)((double)tot_accepted_ / (double)tot_drafted_) : 0.0f;
    }

    /* Per-position acceptance, a diagnostic rather than a policy input. The pooled ratio cannot
     * tell a drafter that decays gradually from one that fails at a specific position.
     *
     * `reached`: every earlier position was accepted, so this one was evaluated. `hits`: and it
     * was accepted. Their ratio is the conditional acceptance. Unreached positions are not
     * charged, so a rejection at position 0 does not count against later positions. */
    int64_t reached(int j) const {
        return (j >= 0 && j < RAD_SCHED_MAX_SPEC) ? reach_[j] : 0;
    }
    int64_t hits(int j) const {
        return (j >= 0 && j < RAD_SCHED_MAX_SPEC) ? hit_[j] : 0;
    }
    /* The deepest position any draft has reached; bounds the meaningful part of both arrays. */
    int     depth_seen() const { return seen_; }

private:
    Params  p_;
    int64_t tot_drafted_ = 0, tot_accepted_ = 0;
    int64_t reach_[RAD_SCHED_MAX_SPEC] = {};
    int64_t hit_  [RAD_SCHED_MAX_SPEC] = {};
    int     seen_ = 0;

    /* The adaptive policy as configure() resolved it. */
    bool    adapt_ = false;
    double  alpha_ = 0.0;                         /* the smoothing step from kHalfLife */
    int64_t by_depth_[RAD_SCHED_MAX_SPEC + 1] = {};
    int32_t step_d_ = 0;                          /* D; 0 until a step chose one */
    int32_t step_w_ = 0;                          /* min(D, the deepest stop it saw) */
    int32_t step_held_ = 0;                       /* steps since D last moved */
    /* The prior from finished sequences (adapt_retire), and how many are behind it. */
    float   prior_[RAD_SCHED_MAX_SPEC] = {};
    int64_t prior_n_ = 0;
    /* T(R): drafting steps' times by query rows. */
    StepCost cost_{kRowCost};
    int64_t timed_ = 0;                           /* samples cost_ took */
    /* Depth 0. step_d_ stays the drafting depth (the policy above); zero_ says D is 0
     * instead. A probe runs probe_left_ more cycles at probe_d_ (0: a plain probe from drafting;
     * else the drafting depth, from 0). */
    bool    depth0_ = false;
    int32_t probe_every_ = kProbeEvery;
    bool    zero_ = false;
    int32_t zero_held_ = 0;                       /* cycles since zero_ last moved */
    int32_t probe_left_ = 0;
    int32_t probe_d_ = 0;
    int32_t since_probe_ = 0;                     /* cycles at 0 since the last probe */
    int32_t zero_run_ = 0;                        /* consecutive cycles the head has not run */
    int32_t backoff_max_ = 0, backoff_k_ = 0;
    int64_t cycles_ = 0;                          /* cycles planned */
    /* The plain step's time by decode rows (no head round), and per row class how many plain
     * samples it holds and the cycle of the last. */
    StepCost plain_{kRowCost};
    int64_t plain_n_[kCostClasses] = {};
    int64_t plain_at_[kCostClasses] = {};
    int64_t plain_probe_at_ = 0;                  /* cycles_ at the last plain probe's start */
    /* The drafting steps per row class: how many were timed, and the MEASURED tokens and time of
     * those that said what they emitted (moving averages, any depth). */
    int64_t dn_[kCostClasses] = {};
    double  dtok_[kCostClasses] = {};
    double  dms_[kCostClasses] = {};
    int64_t dtn_[kCostClasses] = {};
    int64_t n_zero_ = 0, n_probe_ = 0, n_zero_moves_ = 0;
    bool    zero_now() const {
        return adapt_ && depth0_ && (probe_left_ > 0 ? probe_d_ == 0 : zero_);
    }
    static int cost_class(int n_rows);
    /* Depth 0's half of plan_step: tok[d] the expected tokens and drafts[d] the drafts at each
     * depth, `plain` the rows that draft nothing. */
    void    plan_zero(const double* tok, const int* drafts, double plain, int n_rows);

    /* Fill a sequence's estimate from the prior and set its first stop. */
    void    start(DraftAdapt& s) const;
    /* The deepest draft whose chance of being emitted is at least kDraftMinChance. */
    int     stop_of(const float* a) const;
};

}  /* namespace rad */
