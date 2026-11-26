/* rad_mtp_rrs.h -- recursive rejection sampling over a node's SIBLING drafts under
 * --draft-coupling on: the draft side (children sampled without replacement from Q) and the
 * verify side (the walk over a node's children against the target's P), as plain code the host
 * oracle, the device kernels and the tests share, so they cannot drift. Traversal verification
 * (abi/rad_mtp_traversal.h) walks nodes that drafted several children; the serial head of this
 * engine drafts one (nkids <= 1, the chain), so nothing in the engine draws siblings yet: the
 * header is the verify side's definition and its tests (tests/mtp_rrs_test.cpp). The short proof:
 *
 * Draft. A node whose head pass left the law Q (the FP64 T1 softmax of the reranked top 8,
 * rad_mtp_couple_build_q) drafts its children x_1, x_2, ... by Gumbel-top-k over log Q: add an
 * independent Gumbel to every log q_i and take the entries in descending order. That is sampling
 * without replacement (Plackett-Luce): x_1 ~ Q, x_2 ~ Q(. | not x_1), ... The first m of them are
 * the node's children, m chosen by anything that does not read the values x_j it applies to
 * (a fixed count, or a rule over the slot priorities, the j-th largest q of Q, never the drawn
 * token).
 *
 * Verify (Sequoia, Chen et al. 2024, Alg. 1; SpecInfer's multi-step verification with the draft
 * law renormalised after each removal). P_1 = the target's law, q_1 = Q. For i = 1..m:
 *     accept x_i with probability min(1, P_i(x_i) / q_i(x_i));
 *     on rejection  P_{i+1} = norm(max(P_i - q_i, 0)),  q_{i+1} = q_i without x_i, renormalised.
 * All rejected: the token is a draw from P_{m+1}. Exactness (induction on m): after a rejection of
 * the first draft the continuation depends on x_1 only through q_2, and P_2 = norm(max(P - q, 0))
 * does not depend on x_1 at all, so by the hypothesis it emits ~ P_2 whatever x_1 was; then
 * P(emit y) = sum_x q(x) min(1, P(x)/q(x)) [x = y] + P(reject) P_2(y) = min(P, q)(y) +
 * max(P - q, 0)(y) = P(y). The emitted token is the accepted child's token, or a residual draw that
 * has no mass on any rejected child (P_{i+1}(x_i) = 0 after a rejection), so the core's rule
 * "follow the first child whose token is the sampler's token at the row" (ADV_ACCEPT) walks exactly
 * the decision made here and needs no change.
 *
 * Wire: the node's RadMtpCoupleState holds Q (ids, q), the children in draft order in ids[0 ..
 * nkids), x = ids[0]. A state with nkids <= 1 is the chain's (rad_mtp_couple_decide, bit for bit).
 */
#ifndef RAD_MTP_RRS_H
#define RAD_MTP_RRS_H

#include "rad_spechub.h"

#ifdef __cplusplus

#ifdef __HIPCC__
#define RAD_RRS_HD __host__ __device__
#else
#define RAD_RRS_HD
#endif

#define RAD_MTP_RRS_KEY_TREE   0x4d54524545474d42ull   /* the draft's Gumbel stream */
#define RAD_MTP_RRS_KEY_ACCEPT 0x41435054ull

static inline RAD_RRS_HD uint64_t rad_mtp_rrs_mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

/* The open-interval uniform of candidate `c` of draft node `node`: (seed, base) name the draft
 * stream (the request's seed and its pre-commit output length, which no later step reuses). */
static inline RAD_RRS_HD double rad_mtp_rrs_gumbel_u(uint64_t seed, uint64_t base,
                                                         uint64_t node, int c) {
    uint64_t z = rad_mtp_rrs_mix(seed ^ RAD_MTP_RRS_KEY_TREE);
    z = rad_mtp_rrs_mix(z ^ base);
    z = rad_mtp_rrs_mix(z + node * 16u + (uint64_t)c);
    return ((double)(z >> 11) + 0.5) * 0x1p-53;
}

/* Gumbel-top-k order of Q's support: order[0 .. s.count) are indices into s.ids, the draft order
 * (descending of log q + Gumbel, ties to the lower index). Draws one uniform per support entry from
 * the state's own draw, node `node`. */
static inline RAD_RRS_HD void rad_mtp_rrs_order(const RadMtpCoupleState& s, uint64_t node,
                                                    int* order) {
    double key[8];
    for (int i = 0; i < s.count; ++i) {
        const double u = rad_mtp_rrs_gumbel_u(s.draw.seed, s.draw.base, node, i);
        key[i] = ::log(s.q[i]) - ::log(-::log(u));
        order[i] = i;
    }
    for (int i = 1; i < s.count; ++i) {   /* insertion sort, stable */
        const int o = order[i];
        int j = i;
        while (j > 0 && key[order[j - 1]] < key[o]) { order[j] = order[j - 1]; --j; }
        order[j] = o;
    }
}

/* Draw the node's children: reorder Q's support into the draft order (ids and q together), set
 * x = ids[0] and nkids = `nkids`. */
static inline RAD_RRS_HD void rad_mtp_rrs_draw(RadMtpCoupleState* s, uint64_t node, int nkids) {
    int order[8];
    rad_mtp_rrs_order(*s, node, order);
    int32_t ids[8];
    double  q[8];
    for (int i = 0; i < s->count; ++i) { ids[i] = s->ids[order[i]]; q[i] = s->q[order[i]]; }
    for (int i = 0; i < s->count; ++i) { s->ids[i] = ids[i]; s->q[i] = q[i]; }
    s->x = s->ids[0];
    s->nkids = nkids < s->count ? nkids : s->count;
}

/* The SLOT priorities of a node's law: slot j carries the log of the j-th LARGEST q of Q
 * (descending, ties by position), -inf past the support. They are a function of Q alone, never of
 * which token was drawn into which slot, so a rule that reads only them admits children by a
 * quantity independent of the values the children take. `s` is read before the draw reorders it
 * or after (the multiset of q is the same). */
static inline RAD_RRS_HD void rad_mtp_rrs_slot_logq(const RadMtpCoupleState& s, float* ov) {
    double qs[8];
    const int n = s.count < 8 ? s.count : 8;
    for (int i = 0; i < n; ++i) qs[i] = s.q[i];
    for (int i = 1; i < n; ++i) {
        const double v = qs[i];
        int j = i;
        while (j > 0 && qs[j - 1] < v) { qs[j] = qs[j - 1]; --j; }
        qs[j] = v;
    }
    for (int c = 0; c < 8; ++c) ov[c] = c < n ? (float)::log(qs[c]) : -INFINITY;
}

/* The longest prefix of the state's children that are the first `n` of `kid[]` (the tree's
 * children of this row, in node order, repeats dropped): what a verify may keep when a window cut
 * the tree short. 0 when even the first differs. */
static inline RAD_RRS_HD int rad_mtp_rrs_prefix(const RadMtpCoupleState& s, const int32_t* kid,
                                                    int n) {
    const int have = s.nkids >= 1 ? s.nkids : 1;
    int m = 0;
    while (m < have && m < n && m < s.count && s.ids[m] == kid[m]) ++m;
    return s.nkids >= 2 ? m : (n >= 1 && s.x == kid[0] ? 1 : 0);
}

/* The acceptance uniform of the i-th child at output position `pos`: the first is the chain's. */
static inline RAD_RRS_HD double rad_mtp_rrs_uaccept(uint64_t seed, uint64_t pos, int i) {
    return rad_spechub_uniform(seed, pos,
                               i == 0 ? (uint64_t)RAD_SAMPLE_KEY_ACCEPT
                                      : RAD_MTP_RRS_KEY_ACCEPT ^ rad_mtp_rrs_mix((uint64_t)i));
}

/* The verify of a node with nkids >= 2 children: the recursion above over the sampler's surviving
 * candidates (ids/val, n of them, FP64 P = softmax of val). Writes the emitted token to *out and
 * returns true, or false for an invalid target (the row keeps the ordinary token).
 *
 * The decision reads the live candidates (id >= 0, value not -inf) in ascending order through a
 * view `C::each(f)`, f(id, w, k) -> bool (false stops): w = exp(val - top) in FP64, k the id's slot
 * in the state's support (the first match) or s.count when it is outside. The host view
 * (RadRrsHostCands) computes them on the fly; the device verify (plugin/spechub/spechub.hip)
 * compacts the live candidates and computes every w and k with the whole workgroup, then runs this
 * same function on one lane over the stored values, so the ordered sums, and with them the token,
 * are the host's. `pq_out` and `sum_out` (optional) hand back P's masses on the support and their
 * normaliser (the export). */
template <class C>
static inline RAD_RRS_HD bool rad_mtp_rrs_decide_over(const RadMtpCoupleState& s, const C& cs,
                                                      uint64_t seed, uint64_t pos, int32_t* out,
                                                      double* pq_out = nullptr,
                                                      double* sum_out = nullptr) {
    double sum = 0, pq[8] = {}, wout = 0;
    cs.each([&](int32_t, double w, int k) {
        sum += w;
        if (k < s.count) pq[k] += w; else wout += w;
        return true;
    });
    if (!(sum > 0) || !rad_spechub_finite(sum)) return false;
    if (pq_out) for (int k = 0; k < 8; ++k) pq_out[k] = pq[k];
    if (sum_out) *sum_out = sum;
    /* P over Q's support (normalised), and the off-support mass's scale `so` (P_i(y) = so * w / sum
     * there). qr: the draft law with the drawn children removed. */
    double r[8], qr[8], so = 1.0;
    for (int k = 0; k < s.count; ++k) { r[k] = pq[k] / sum; qr[k] = s.q[k]; }
    const double pout = wout / sum;
    bool direct = false;
    for (int i = 0; i < s.nkids && !direct; ++i) {
        double sq = 0;
        for (int k = 0; k < s.count; ++k) sq += qr[k];
        if (!(sq > 0)) break;
        const double qx = qr[i] / sq;
        if (r[i] >= qx || rad_mtp_rrs_uaccept(seed, pos, i) * qx < r[i]) {
            *out = s.ids[i];
            return true;
        }
        double z = so * pout;
        for (int k = 0; k < s.count; ++k) {
            const double d = r[k] - qr[k] / sq;
            r[k] = d > 0 ? d : 0;
            z += r[k];
        }
        if (!(z > 0) || !rad_spechub_finite(z)) { direct = true; break; }
        for (int k = 0; k < s.count; ++k) r[k] /= z;
        so /= z;
        qr[i] = 0;
    }
    const double u = rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_RESID);
    double  c;
    int32_t last = -1;
    if (direct) {
        c = u * sum;
        bool hit = false;
        cs.each([&](int32_t id, double w, int) {
            if (w > 0) { last = id; c -= w; if (c < 0) { hit = true; return false; } }
            return true;
        });
        *out = last;
        (void)hit;
        return true;
    }
    double total = 0;
    for (int k = 0; k < s.count; ++k) total += r[k];
    total += so * pout;
    c = u * total;
    unsigned seen = 0;
    bool hit = false;
    cs.each([&](int32_t id, double w, int k) {
        double m;
        if (k < s.count) {
            if (seen & (1u << k)) return true;
            seen |= 1u << k;
            m = r[k];
        } else {
            m = so * (w / sum);
        }
        if (m > 0) { last = id; c -= m; if (c < 0) { hit = true; return false; } }
        return true;
    });
    *out = last;
    (void)hit;
    return last >= 0;
}

/* The host view: the candidates as the sampler wrote them, weights computed as they are read. */
struct RadRrsHostCands {
    const RadMtpCoupleState& s;
    const int32_t* ids; const float* val; int64_t n; float top;
    template <class F> RAD_RRS_HD void each(F&& f) const {
        for (int64_t j = 0; j < n; ++j) {
            if (ids[j] < 0 || val[j] == -INFINITY) continue;
            int k = 0;
            while (k < s.count && ids[j] != s.ids[k]) ++k;
            if (!f(ids[j], ::exp((double)val[j] - top), k)) return;
        }
    }
};

static inline RAD_RRS_HD bool rad_mtp_rrs_decide(const RadMtpCoupleState& s, const int32_t* ids,
                                                     const float* val, int64_t n, int64_t vocab,
                                                     uint64_t seed, uint64_t pos, int32_t* out) {
    float top = -INFINITY;
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] < 0) continue;
        if (ids[j] >= vocab || val[j] != val[j] || val[j] == INFINITY) return false;
        if (val[j] > top) top = val[j];
    }
    if (top == -INFINITY) return false;
    return rad_mtp_rrs_decide_over(s, RadRrsHostCands{ s, ids, val, n, top }, seed, pos, out);
}

/* ================================================================== RheoSampling siblings
 *
 * --draft-siblings rheo (--draft-rheo-m m). Hu et al. 2025 (arXiv 2502.18779, Thm 8, the greedy
 * multi-draft verifier) and RheoSampling (arXiv 2609.21827): instead of Gumbel-top-k siblings, a
 * node's children are
 *     T  = the m tokens of largest Q, in descending q order (deterministic, slots 0 .. m-1), and
 *     u  ~ Qt, ONE token sampled from the tail law Qt(x) = Q(x) / z over the support outside T,
 *          z = Q(outside T), in slot m;
 * the draft order stored in the state is ids = [T..., u, the rest of the support in descending q],
 * scheme = m + 1. A node whose support has no more than m tokens has no tail child (all of it
 * is T).
 *
 * Verify (derived here, not quoted; the law is checked by tests/mtp_rrs_test.cpp by exhaustive
 * enumeration). P = the target's law, children present = the first nkids of the draft order.
 *   tail child present (nkids > |T|, i.e. with a prefix draft order every T token is a child):
 *       accept u with probability min(1, P(u) / Qt(u))                       -> emit u
 *       else emit y ~ R / |R|,   R(y) = P(y)               y in T or off Q's support,
 *                                R(y) = max(P(y) - Qt(y), 0)  y in the tail support (R(u) = 0 here)
 *   otherwise (the tail slot was pruned or cut by a window, or Q has no tail): emit y ~ P directly.
 * Exactness: P(emit y) = min(P, Qt)(y) [y in tail] + (1 - A) R(y) / |R| with |R| = 1 - A,
 * A = sum_tail min(P, Qt), and min(P, Qt) + R = P on the tail, R = P elsewhere: P(emit y) = P(y).
 * It needs only that u ~ Qt independently of the verifier's uniforms (a separate draft stream), and
 * that WHICH slots exist depends on Q and the tree's budget alone (never on the value of u): both
 * hold, the slot priorities are rad_mtp_rrs_slot_logq_rheo. The emitted token is a child with
 * probability P(T) + A = sum_T P + sum_tail min(P, Qt), Thm 8's value, and a residual draw never
 * lands on u (R(u) = 0 when u was rejected), so the core's "follow the first child whose token is
 * the sampler's" walks the decision made here.
 * m = 0 is the single-draft maximal coupling of the chain (rad_mtp_couple_decide) with the sampled
 * child in slot 0.
 */
/* The candidate index of the tail draw in the draft stream (0..7 are Gumbel's). */
#define RAD_MTP_RRS_KEY_TAIL_C 8

/* Slot priorities of a rheo node (functions of Q alone, non-increasing): slots 0 .. m-1 carry the
 * log of the j-th largest q, slot m RheoSampling's proxy log min{q_(m-1), z} (z = the tail mass;
 * with m = 0, z = 1), -inf for the other slots and for slots past the support. */
static inline RAD_RRS_HD void rad_mtp_rrs_slot_logq_rheo(const RadMtpCoupleState& s, int m,
                                                         float* ov) {
    double qs[8];
    const int n = s.count < 8 ? s.count : 8;
    for (int i = 0; i < n; ++i) qs[i] = s.q[i];
    for (int i = 1; i < n; ++i) {
        const double v = qs[i];
        int j = i;
        while (j > 0 && qs[j - 1] < v) { qs[j] = qs[j - 1]; --j; }
        qs[j] = v;
    }
    const int nt = m < n ? m : n;
    for (int c = 0; c < 8; ++c) ov[c] = c < nt ? (float)::log(qs[c]) : -INFINITY;
    if (n > nt) {
        double z = 0;
        for (int i = nt; i < n; ++i) z += qs[i];
        if (z > 1) z = 1;
        const double ref = nt > 0 ? qs[nt - 1] : 1.0;
        ov[nt] = (float)::log(z < ref ? z : ref);
    }
}

/* Draw a rheo node's children: reorder Q's support into [top-m by q desc, the sampled tail child,
 * the rest by q desc], set x = ids[0], scheme = m + 1 and nkids = min(nkids, children possible).
 * The tail child is the inverse-CDF draw of Qt with the draft stream's uniform of (seed, base,
 * node). */
static inline RAD_RRS_HD void rad_mtp_rrs_draw_rheo_u(RadMtpCoupleState* s, double ut, int nkids,
                                                      int m) {
    int ord[8];
    const int n = s->count;
    for (int i = 0; i < n; ++i) ord[i] = i;
    for (int i = 1; i < n; ++i) {   /* descending q, stable */
        const int o = ord[i];
        int j = i;
        while (j > 0 && s->q[ord[j - 1]] < s->q[o]) { ord[j] = ord[j - 1]; --j; }
        ord[j] = o;
    }
    const int nt = m < n ? m : n;
    if (n > nt) {
        double z = 0;
        for (int i = nt; i < n; ++i) z += s->q[ord[i]];
        double c = ut * z;
        int pick = n - 1;
        for (int i = nt; i < n; ++i) { c -= s->q[ord[i]]; if (c < 0) { pick = i; break; } }
        const int o = ord[pick];
        for (int i = pick; i > nt; --i) ord[i] = ord[i - 1];
        ord[nt] = o;
    }
    int32_t ids[8];
    double  q[8];
    for (int i = 0; i < n; ++i) { ids[i] = s->ids[ord[i]]; q[i] = s->q[ord[i]]; }
    for (int i = 0; i < n; ++i) { s->ids[i] = ids[i]; s->q[i] = q[i]; }
    const int cap = nt + (n > nt ? 1 : 0);
    s->x = s->ids[0];
    s->scheme = m + 1;
    s->nkids = nkids < cap ? nkids : cap;
}
static inline RAD_RRS_HD void rad_mtp_rrs_draw_rheo(RadMtpCoupleState* s, uint64_t node, int nkids,
                                                    int m) {
    const double ut =
        rad_mtp_rrs_gumbel_u(s->draw.seed, s->draw.base, node, RAD_MTP_RRS_KEY_TAIL_C);
    rad_mtp_rrs_draw_rheo_u(s, ut, nkids, m);
}

/* The verify of a rheo node (any nkids >= 0): the rule above. Same contract as
 * rad_mtp_rrs_decide. */
static inline RAD_RRS_HD bool rad_mtp_rrs_decide_rheo_u(const RadMtpCoupleState& s,
                                                        const int32_t* ids, const float* val,
                                                        int64_t n, int64_t vocab, double u_accept,
                                                        double u_resid, int32_t* out) {
    float top = -INFINITY;
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] < 0) continue;
        if (ids[j] >= vocab || val[j] != val[j] || val[j] == INFINITY) return false;
        if (val[j] > top) top = val[j];
    }
    if (top == -INFINITY) return false;
    double sum = 0, pq[8] = {};
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] < 0 || val[j] == -INFINITY) continue;
        const double w = ::exp((double)val[j] - top);
        sum += w;
        for (int k = 0; k < s.count; ++k)
            if (ids[j] == s.ids[k]) { pq[k] += w; break; }
    }
    if (!(sum > 0) || !rad_spechub_finite(sum)) return false;
    const int m = s.scheme - 1;
    const int nt = m < s.count ? m : s.count;
    const bool tail = m >= 0 && s.count > nt && s.nkids > nt;
    double qt[8] = {}, r[8] = {};
    bool dominated = true;
    for (int k = 0; k < s.count; ++k) pq[k] /= sum;
    if (tail) {
        double z = 0;
        for (int k = nt; k < s.count; ++k) z += s.q[k];
        for (int k = nt; k < s.count; ++k) {
            const double v = nt == 0 ? s.q[k] : s.q[k] / z;
            qt[k] = v > 1 ? 1 : v;   /* clamp: rounding must not push a one-token tail past 1 */
            dominated = dominated && pq[k] >= qt[k];
        }
        for (int k = 0; k < s.count; ++k)
            r[k] = k < nt ? pq[k] : rad_spechub_positive(pq[k] - qt[k]);
    }
    /* mass of candidate j in R (the tail case) or in P (otherwise); Q tokens count once. */
    unsigned seen = 0;
    auto mass = [&](int64_t j) -> double {
        if (ids[j] < 0 || val[j] == -INFINITY) return 0;
        for (int k = 0; k < s.count; ++k)
            if (s.ids[k] == ids[j]) {
                if (seen & (1u << k)) return 0;
                seen |= 1u << k;
                return tail ? r[k] : pq[k];
            }
        return ::exp((double)val[j] - top) / sum;
    };
    double resid = 0;
    if (tail) {
        for (int64_t j = 0; j < n; ++j) resid += mass(j);
        seen = 0;
    }
    if (!tail || (!dominated && !(resid > 0 && rad_spechub_finite(resid)))) {
        double  c = u_resid * sum;   /* P itself, in the sampler's candidate order */
        int32_t last = -1;
        for (int64_t j = 0; j < n; ++j)
            if (ids[j] >= 0 && val[j] != -INFINITY) {
                const double w = ::exp((double)val[j] - top);
                if (w > 0) { last = ids[j]; c -= w; if (c < 0) { *out = last; return true; } }
            }
        *out = last;
        return last >= 0;
    }
    const int iu = nt;
    if (pq[iu] >= qt[iu] || u_accept * qt[iu] < pq[iu]) {
        *out = s.ids[iu];
        return true;
    }
    double  c = u_resid * resid;
    int32_t last = -1;
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] == s.ids[iu]) continue;
        const double mm = mass(j);
        if (mm > 0) { last = ids[j]; c -= mm; if (c < 0) { *out = last; return true; } }
    }
    *out = last;
    return last >= 0;
}
static inline RAD_RRS_HD bool rad_mtp_rrs_decide_rheo(const RadMtpCoupleState& s,
                                                      const int32_t* ids, const float* val,
                                                      int64_t n, int64_t vocab, uint64_t seed,
                                                      uint64_t pos, int32_t* out) {
    return rad_mtp_rrs_decide_rheo_u(s, ids, val, n, vocab,
                                     rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_ACCEPT),
                                     rad_spechub_uniform(seed, pos, RAD_SAMPLE_KEY_RESID), out);
}

#endif /* __cplusplus */
#endif /* RAD_MTP_RRS_H */
