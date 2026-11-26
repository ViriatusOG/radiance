/* rad_mtp_block.h -- BLOCK VERIFICATION of a linear MTP draft chain (--draft-verify block).
 *
 * Sun, Mendlovic, Leviathan, Aharoni, Beirami, Ro, Suresh, "Block Verification Accelerates
 * Speculative Decoding" (arXiv 2403.10444): the chain's draft tokens X_1 .. X_g are accepted or
 * rejected JOINTLY. Token-wise speculative sampling (what --draft-coupling does at each row) is
 * optimal per token; block verification is optimal for the whole block: the expected number of
 * accepted tokens is never lower and the output law is the target's, exactly.
 *
 * With P_i = the target's law after X^i (the sampler's filtered candidate softmax at verify row i,
 * i = 0 .. g-1) and Q_{i+1} = the head's law the draft X_{i+1} was sampled from (the coupled state
 * of round i + 1, the FP64 softmax of the reranked top 8):
 *
 *     p_0 = 1,   p_i = min(1, p_{i-1} P_{i-1}(X_i) / Q_i(X_i))                  i = 1 .. g
 *     S_i = sum_x max(p_i P_i(x) - Q_{i+1}(x), 0)                                i = 0 .. g-1
 *     h_i = S_i / (S_i + 1 - p_i)   (i < g),    h_g = p_g
 *
 *     for i = g, g-1, ..., 0:  accept the block's first i tokens with probability h_i; the first
 *     success is tau. (h_0 = 1: it always stops.)
 *     tau = g:  X_1 .. X_g are emitted and the next token is the ordinary draw from P_g.
 *     tau < g:  X_1 .. X_tau are emitted, then Y ~ max(p_tau P_tau - Q_{tau+1}, 0) normalised.
 *
 * Exactness is checked in tests/mtp_block_test.cpp by exact enumeration of the draft tree of a toy
 * model (the law of the emitted tokens computed in closed form, to 1e-12) and by a chi-square of
 * the real device-shared code, with a negative control (the same rule with a wrong residual).
 *
 * Y may equal X_{tau+1}; the core then takes that draft as accepted (ADV_ACCEPT: "the draft is the
 * sampler's token at this row") and reads the next row, which keeps its ordinary pick: a draw from
 * the target's law given exactly the tokens emitted so far, so the extra tokens continue the same
 * autoregressive sample. The verify op therefore writes only rows 0 .. tau of the block and leaves
 * every later row alone.
 *
 * Relation to --draft-coupling: the DRAFT half is shared (x ~ Q at temperature 1 from the head's
 * top 8, coupled states with their FP64 Q); only the verify half differs. Maximal coupling decides
 * row k from P_k and Q_{k+1} alone, so it is a per-row rule the core's token comparison walks
 * unchanged. Block verification decides row k from the rows after it (h_i for i > k), so the two
 * rules cannot be applied to the same row; `--draft-verify block` replaces the coupling's verify
 * and requires the coupled drafting (it switches it on), and it is for a LINEAR chain; Traversal
 * Verification (abi/rad_mtp_traversal.h, 2505.12398) is the walk that also covers a node with
 * several children.
 */
#ifndef RAD_MTP_BLOCK_H
#define RAD_MTP_BLOCK_H

#include "rad_spechub.h"

#ifdef __cplusplus

#ifdef __HIPCC__
#define RAD_BLK_HD __host__ __device__
#else
#define RAD_BLK_HD
#endif

#define RAD_MTP_BLOCK_MAX      RAD_MTP_COUPLE_MAX_ROUNDS   /* g: the chain's longest block */
#define RAD_MTP_BLOCK_KEY_STOP 0x424c4b53544f5000ull       /* the h_i uniforms, per block */

/* Summation structure. The device decides a block with one workgroup of RAD_MTP_BLOCK_LANES
 * threads, so every FP64 sum over a row's candidate plane is DEFINED as this fixed tree, which the
 * host reference mirrors operation for operation (FP64 addition is not associative: the tree is
 * the definition, not an approximation):
 *   - plane slot j belongs to lane j % LANES; a lane adds its live slots in ascending j;
 *   - the lanes are reduced inside each group of RAD_MTP_BLOCK_WAVE by the butterfly
 *     a[l] = a[l] + a[l ^ d], d = 16, 8, 4, 2, 1 (every lane of a group ends with the same bits),
 *     and the LANES / WAVE group sums are added as ((g0 + g1) + (g2 + g3)).
 * The exp() of a slot is the same libm / device function on both sides. */
#define RAD_MTP_BLOCK_LANES 128
#define RAD_MTP_BLOCK_WAVE  32

/* One verify row's candidate plane, dense (n slots, holes id < 0 and -inf scores skipped), with
 * `valid` and `top` the scan of the whole plane (rad_mtp_block_scan makes it on the host; the
 * device reduces the same two values across its workgroup: a max and an OR are order free). */
struct RadBlockRow {
    const int32_t* ids;
    const float*   val;
    int64_t        n;
    int            valid;
    float          top;
};

/* The dense row's validity and maximum, as rad_mtp_couple_decide scans them. */
static inline RAD_BLK_HD void rad_mtp_block_scan(RadBlockRow* r, int64_t vocab) {
    float top = -INFINITY;
    int valid = 1;
    for (int64_t j = 0; j < r->n; ++j) {
        if (r->ids[j] < 0) continue;
        if (r->ids[j] >= vocab || r->val[j] != r->val[j] || r->val[j] == INFINITY) valid = 0;
        else if (r->val[j] > top) top = r->val[j];
    }
    r->valid = valid;
    r->top = top;
}

static inline RAD_BLK_HD bool rad_mtp_block_live(int32_t id, float v) {
    return id >= 0 && v != -INFINITY;
}

/* A lane's partial sums of one row: v[0..7] the mass on Q's ids (by their place in the state),
 * v[8] the mass off Q's support, v[9] the total. */
struct RadBlockAcc { double v[10]; };

static inline RAD_BLK_HD int rad_mtp_block_find(const RadMtpCoupleState& s, int32_t id) {
    int k = 0;
    while (k < s.count && id != s.ids[k]) ++k;
    return k;
}

/* Adds the live slot (id, v) to a lane's partial. */
static inline RAD_BLK_HD void rad_mtp_block_acc_add(RadBlockAcc* a, const RadMtpCoupleState& s,
                                                    int32_t id, float v, float top) {
    const double w = ::exp((double)v - top);
    const int k = rad_mtp_block_find(s, id);
    for (int kk = 0; kk < 8; ++kk) a->v[kk] += kk == k ? w : 0.0;
    a->v[8] += k >= s.count ? w : 0.0;
    a->v[9] += w;
}

/* The lanes' partials reduced by the tree above (the host form; the device does the butterfly with
 * shuffles and adds the four group sums the same way). */
static inline RAD_BLK_HD RadBlockAcc rad_mtp_block_tree(const RadBlockAcc* lane) {
    RadBlockAcc grp[RAD_MTP_BLOCK_LANES / RAD_MTP_BLOCK_WAVE];
    for (int w = 0; w < RAD_MTP_BLOCK_LANES / RAD_MTP_BLOCK_WAVE; ++w) {
        RadBlockAcc a[RAD_MTP_BLOCK_WAVE], t[RAD_MTP_BLOCK_WAVE];
        for (int l = 0; l < RAD_MTP_BLOCK_WAVE; ++l) a[l] = lane[w * RAD_MTP_BLOCK_WAVE + l];
        for (int d = RAD_MTP_BLOCK_WAVE / 2; d > 0; d >>= 1) {
            for (int l = 0; l < RAD_MTP_BLOCK_WAVE; ++l) t[l] = a[l];
            for (int l = 0; l < RAD_MTP_BLOCK_WAVE; ++l)
                for (int c = 0; c < 10; ++c) a[l].v[c] = t[l].v[c] + t[l ^ d].v[c];
        }
        grp[w] = a[0];
    }
    RadBlockAcc o;
    for (int c = 0; c < 10; ++c) o.v[c] = (grp[0].v[c] + grp[1].v[c]) + (grp[2].v[c] + grp[3].v[c]);
    return o;
}

/* P_i over Q_{i+1}'s support (pq, normalised, duplicate candidates aggregated), the mass off that
 * support (pout), the normaliser `sum` and the index `ix` of the draft token in the state's ids. */
struct RadBlockStat {
    double pq[8];
    double pout, sum;
    int    ix;
};

/* The stat from the reduced partials `t` (what the device builds from its workgroup's tree). */
static inline RAD_BLK_HD bool rad_mtp_block_stat_from(const RadBlockAcc& t, int valid, float top,
                                                      const RadMtpCoupleState& s, RadBlockStat* o) {
    if (!valid || top == -INFINITY) return false;
    const double sum = t.v[9];
    if (!(sum > 0) || !rad_spechub_finite(sum)) return false;
    o->ix = -1;
    for (int k = 0; k < 8; ++k) o->pq[k] = k < s.count ? t.v[k] : 0.0;
    for (int k = 0; k < s.count; ++k) {
        o->pq[k] /= sum;
        if (s.ids[k] == s.x) o->ix = k;
    }
    o->pout = t.v[8] / sum;
    o->sum = sum;
    return o->ix >= 0;
}

static inline RAD_BLK_HD bool rad_mtp_block_stat(const RadBlockRow& r, const RadMtpCoupleState& s,
                                                 RadBlockStat* o) {
    if (!r.valid || r.top == -INFINITY) return false;
    RadBlockAcc lane[RAD_MTP_BLOCK_LANES];
    for (int l = 0; l < RAD_MTP_BLOCK_LANES; ++l)
        for (int c = 0; c < 10; ++c) lane[l].v[c] = 0;
    for (int64_t j = 0; j < r.n; ++j)
        if (rad_mtp_block_live(r.ids[j], r.val[j]))
            rad_mtp_block_acc_add(&lane[j % RAD_MTP_BLOCK_LANES], s, r.ids[j], r.val[j], r.top);
    return rad_mtp_block_stat_from(rad_mtp_block_tree(lane), r.valid, r.top, s, o);
}

/* The block's weights. p[i] (i = 0 .. g), h[i] (i = 0 .. g). */
struct RadBlockPlan {
    int    g;
    double p[RAD_MTP_BLOCK_MAX + 1];
    double h[RAD_MTP_BLOCK_MAX + 1];
    double s[RAD_MTP_BLOCK_MAX + 1];   /* S_i, the residual's mass (s[g] unused) */
};

/* The residual mass of one token of row i at weight p: max(p P(x) - Q(x), 0) on Q's support,
 * p P(x) off it. */
static inline RAD_BLK_HD double rad_mtp_block_resid_sum(const RadMtpCoupleState& s,
                                                        const RadBlockStat& t, double p) {
    double r = p * t.pout;
    for (int k = 0; k < s.count; ++k) r += rad_spechub_positive(p * t.pq[k] - s.q[k]);
    return r;
}

/* st[i] / stat[i] for i = 0 .. g-1. */
static inline RAD_BLK_HD void rad_mtp_block_plan(const RadMtpCoupleState* st,
                                                 const RadBlockStat* stat, int g,
                                                 RadBlockPlan* o) {
    o->g = g;
    o->p[0] = 1.0;
    for (int i = 0; i < g; ++i) {
        const double ratio = stat[i].pq[stat[i].ix] / st[i].q[stat[i].ix];
        const double v = o->p[i] * ratio;
        o->p[i + 1] = v < 1.0 ? v : 1.0;
    }
    for (int i = 0; i < g; ++i) {
        const double S = rad_mtp_block_resid_sum(st[i], stat[i], o->p[i]);
        o->s[i] = S;
        const double den = S + (1.0 - o->p[i]);
        o->h[i] = (den > 0 && rad_spechub_finite(den)) ? S / den : 1.0;
    }
    o->s[g] = 0;
    o->h[g] = o->p[g];
}

/* tau: the first i, counting down from g, whose uniform falls below h_i. The uniforms are keyed by
 * the block's first output position, so no row's own draws are reused. */
static inline RAD_BLK_HD int rad_mtp_block_tau(const RadBlockPlan& pl, uint64_t seed,
                                               uint64_t pos0) {
    for (int i = pl.g; i > 0; --i)
        if (rad_spechub_uniform(seed, pos0, RAD_MTP_BLOCK_KEY_STOP + (uint64_t)i) < pl.h[i])
            return i;
    return 0;
}

/* One lane's partials of the residual draw: `off` the off-support masses p * (w / sum) (slots whose
 * id is not among Q's), `all` the plain weights (the fallback draw from P itself), and the last
 * positive id of each, in the lane's slot order. */
struct RadBlockLane { double off, all; int32_t last_off, last_all; };

static inline RAD_BLK_HD void rad_mtp_block_lane_add(RadBlockLane* a, const RadMtpCoupleState& s,
                                                     const RadBlockStat& t, double p, int32_t id,
                                                     float v, float top) {
    const double w = ::exp((double)v - top);
    a->all += w;
    if (w > 0) a->last_all = id;
    if (rad_mtp_block_find(s, id) >= s.count) {
        const double m = p * (w / t.sum);
        a->off += m;
        if (m > 0) a->last_off = id;
    }
}

/* Y ~ max(p P - Q, 0) normalised (state s, row r, stat t) with the uniform u in [0, 1). The mass
 * is walked in a fixed order: the ids of Q (place k = 0 .. count-1, mass max(p pq[k] - q[k], 0)),
 * then the other live slots lane by lane (the lanes of the tree above, each lane's slots in
 * ascending order; the lane totals are added in lane order). The walk subtracts the masses from
 * c = u * total and stops at the first negative; inside the lane it stops in, the walk restarts
 * from the c at the lane's start, so the structure is exactly what the workgroup computes. When
 * the total underflows to zero (FP64 cancellation with p = 1: P == Q) Y is drawn from P itself
 * over all live slots in the same lane order, a branch that depends on (P, Q) only. False: no live
 * candidate. */
static inline RAD_BLK_HD bool rad_mtp_block_resid_walk(const RadBlockRow& r,
                                                       const RadMtpCoupleState& s,
                                                       const RadBlockStat& t, double p, double u,
                                                       const RadBlockLane* lane, int32_t* out) {
    double on[8];
    double total = 0;
    for (int k = 0; k < s.count; ++k) {
        on[k] = rad_spechub_positive(p * t.pq[k] - s.q[k]);
        total += on[k];
    }
    double offt = 0;
    for (int l = 0; l < RAD_MTP_BLOCK_LANES; ++l) offt += lane[l].off;
    total += offt;
    const bool resid = total > 0 && rad_spechub_finite(total);
    double c = resid ? u * total : u * t.sum;
    int32_t last = -1;
    if (resid) {
        for (int k = 0; k < s.count; ++k)
            if (on[k] > 0) { last = s.ids[k]; c -= on[k]; if (c < 0) { *out = last; return true; } }
    }
    for (int l = 0; l < RAD_MTP_BLOCK_LANES; ++l) {
        const double T = resid ? lane[l].off : lane[l].all;
        const int32_t ll = resid ? lane[l].last_off : lane[l].last_all;
        if (!(T > 0)) continue;
        if (ll >= 0) last = ll;
        if (c - T < 0) {
            double cl = c;
            int32_t lastl = -1;
            for (int64_t j = l; j < r.n; j += RAD_MTP_BLOCK_LANES) {
                if (!rad_mtp_block_live(r.ids[j], r.val[j])) continue;
                const double w = ::exp((double)r.val[j] - r.top);
                double m;
                if (resid) {
                    if (rad_mtp_block_find(s, r.ids[j]) < s.count) continue;
                    m = p * (w / t.sum);
                } else {
                    m = w;
                }
                if (m > 0) { lastl = r.ids[j]; cl -= m; if (cl < 0) { *out = lastl; return true; } }
            }
            *out = lastl >= 0 ? lastl : ll;
            return *out >= 0;
        }
        c -= T;
    }
    *out = last;
    return last >= 0;
}

static inline RAD_BLK_HD bool rad_mtp_block_resid_draw(const RadBlockRow& r,
                                                       const RadMtpCoupleState& s,
                                                       const RadBlockStat& t, double p, double u,
                                                       int32_t* out) {
    RadBlockLane lane[RAD_MTP_BLOCK_LANES];
    for (int l = 0; l < RAD_MTP_BLOCK_LANES; ++l) lane[l] = RadBlockLane{ 0, 0, -1, -1 };
    for (int64_t j = 0; j < r.n; ++j)
        if (rad_mtp_block_live(r.ids[j], r.val[j]))
            rad_mtp_block_lane_add(&lane[j % RAD_MTP_BLOCK_LANES], s, t, p, r.ids[j], r.val[j],
                                   r.top);
    return rad_mtp_block_resid_walk(r, s, t, p, u, lane, out);
}

/* The whole decision on the host: rows[0 .. g-1] / st[0 .. g-1] are the block's verify rows and
 * their coupled states (round i + 1 at row i), pos0 the first row's output position. Writes tau and
 * *y; *y = -1 when tau == g (the g-th row's ordinary draw stands). False for an invalid target. */
static inline RAD_BLK_HD bool rad_mtp_block_decide(const RadMtpCoupleState* st,
                                                   const RadBlockRow* rows, int g, uint64_t seed,
                                                   uint64_t pos0, int* tau, int32_t* y) {
    RadBlockStat stat[RAD_MTP_BLOCK_MAX];
    RadBlockPlan pl;
    if (g < 1 || g > RAD_MTP_BLOCK_MAX) return false;
    for (int i = 0; i < g; ++i)
        if (!rad_mtp_block_stat(rows[i], st[i], &stat[i])) return false;
    rad_mtp_block_plan(st, stat, g, &pl);
    const int t = rad_mtp_block_tau(pl, seed, pos0);
    *tau = t;
    *y = -1;
    if (t >= g) return true;
    const double u = rad_spechub_uniform(seed, pos0 + (uint64_t)t, RAD_SAMPLE_KEY_RESID);
    return rad_mtp_block_resid_draw(rows[t], st[t], stat[t], pl.p[t], u, y);
}

#endif /* __cplusplus */
#endif /* RAD_MTP_BLOCK_H */
