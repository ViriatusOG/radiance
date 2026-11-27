/* rad_mtp_traversal.h -- Traversal Verification (Weng et al., arXiv 2505.12398) of a draft under
 * sampling (--draft-verify traversal, needs --draft-coupling on). The walk is defined on a tree;
 * the serial head of this engine drafts a chain, a tree of one child a node, which the sampler
 * stages (core/sample/sampler.cpp). Plain code shared by the host oracle, the device kernel and the
 * tests; the short form of the argument:
 *
 * Row-wise verification (rad_mtp_rrs_decide) decides each node on its own. Traversal verification
 * decides a whole ROOT-TO-LEAF PATH at once, bottom-up, and so can accept a longer prefix from the
 * same tree (for a chain it is Block Verification, never worse than token-level rejection
 * sampling).
 *
 * State of a node v during the walk: p_v (its acceptance rate, a sequence-level ratio), the target
 * law M_b(.|v) restricted to what has not been rejected below it (r), and the law the NEXT child of
 * v was drawn from.
 *   - A child c of v enters with p_c = min(1, p_v * r(c)/qn(c)), qn the law c was drawn from GIVEN
 *     the children tried before it.
 *   - The walk descends to the first leaf and tests eta < p_leaf. Accept: the path is accepted and
 *     the final token is drawn from that node's current residual law r (the plain target if none of
 *     its children was rejected). Reject: the leaf is deleted and its parent v is updated:
 *         S    = sum_x [ p_v r(x) - qn(x) ]+
 *         p_v' = S / (S + 1 - p_v)
 *         r'   = [ p_v r - qn ]+ / S
 *     and the walk continues with v's next child, or tests v itself once it has none.
 * The walk keeps the PRODUCT m(x) = p_v r(x) (on the draft support per slot, and one number for
 * everything off it, where qn is zero), which is all the rules read: p_c = min(1, m(c)/qn(c)), and
 * the update is one division, m' = [m - qn]+ / (S + 1 - p_v), p_v' = S / (S + 1 - p_v), the
 * off-support mass scaled alike; the final token is drawn in proportion to m. A node entered with
 * rate p keeps its target law unscaled and the factor p beside it (sc) until its first child is
 * rejected, so entering a node costs one division (the child's rate) and no pass over its law.
 * Exactness (Theorem 3.3 of the paper, Lemma A.2 by induction over a node's descendants): the
 * subtree of v is accepted with probability p_v and, given that, the token after v follows r. One
 * step of the induction, for a child drawn from ANY law qn that may depend on the children before
 * it: P(accepted through a child x) = qn(x) min(1, p_v r(x)/qn(x)) = min(qn(x), p_v r(x));
 * P(the child's subtree is rejected) = 1 - p_v + S; the rest then accepts with p_v' and r', which
 * adds (1 - p_v + S) p_v' r'(x) = [p_v r(x) - qn(x)]+, and the two sum to p_v r(x). It needs the
 * tree's shape to be independent of the drawn tokens (a chain's is), and each child's own law:
 *   - Gumbel-top-k siblings (the default, Plackett-Luce without replacement): qn is the node's Q
 *     with the children already tried removed, renormalised;
 *   - RheoSampling siblings (state.scheme = m + 1, abi/rad_mtp_rrs.h): the
 *     sampled tail child is tried FIRST with qn = Qt, the tail law, then the deterministic top-q
 *     children in their order, each with qn = a point mass on its token. A point mass accepts with
 *     p_v r(c) and leaves r without c; tried in this order the node's children take
 *     sum_tail min(P, Qt) + P(T) of the target, the greedy multi-draft verifier's value (Hu et al.
 *     2025, Thm 8).
 * tests/mtp_rrs_test.cpp checks the walk by exact enumeration and by chi-square, and shows that
 * plausible mistakes (token-level ratios, no (1 - p) term, with-replacement update for
 * without-replacement drafts, a rheo node walked as a Gumbel one) fail the same statistic.
 *
 * THE TARGET LAW IN FIXED POINT. The walk reads a row's target law only through its mass on the
 * node's draft support (eight ids), the mass off it and the total; the final token needs the
 * off-support candidates in order. Each live candidate's weight is an INTEGER:
 *
 *     m_j = floor(2^40 * e(val_j - top)),   e(x) = rad_mtp_trav_mass's FP32 exponential
 *
 * (top the row's largest value; the scale drops a bit per doubling of the plane past 4096 slots so
 * a row's total stays below 2^53). Integer addition is associative, so the sums are the same bits
 * whatever order a workgroup adds them in: the device kernel (plugin/spechub/spechub.hip) splits a
 * row over its threads any way it likes and still equals this header's serial loops, with no
 * reduction tree to mirror. e() is basic FP32 arithmetic and integer shifts, identical on host and
 * device (no libm call, no contraction: -ffp-contract=off, and the attribute below for a host
 * build that enables FMA).
 * Why FP32 suffices: the algorithm is exact for whatever positive weights define the law, as long
 * as every decision reads the same ones, which the integers guarantee. The law it realises is P~(y)
 * = m_y / sum m, within 2^-22 relative per candidate (e's error, tests/mtp_rrs_test.cpp checks it)
 * plus 2^-40 absolute of the softmax of the sampler's FP32 candidates: the accuracy class of the
 * uncoupled sample_pick (FP32 expf, FP32 running sum, a 24-bit uniform), which is the law a request
 * gets without speculation. The 8-wide FP64 sums of the walk are a fixed tree (rad_mtp_trav_sum8),
 * so a device may run them on one lane or across eight.
 *
 * Wire: no new state. The walk reads the same RadMtpCoupleState rows as the row-wise verify and
 * writes the SAME plane the core already walks (ADV_ACCEPT: the first child whose token is the
 * sampler's token at the row): on the accepted path every row's token is its path child's token and
 * the last row's token is the final one; the other rows are not on the walk and keep the pick.
 */
#ifndef RAD_MTP_TRAVERSAL_H
#define RAD_MTP_TRAVERSAL_H

#include "rad_spechub.h"
#include "rad_mtp_rrs.h"

#ifdef __cplusplus

#define RAD_MTP_TRAV_MAX_DEPTH 16
#define RAD_MTP_TRAV_MAX_ROWS  64              /* rows of one request the walk reads (first ones) */
#define RAD_MTP_TRAV_KEY       0x54524156ull   /* "TRAV": the walk's acceptance uniforms */
#define RAD_MTP_TRAV_QBITS     40              /* a candidate's mass unit: 2^-40 of the top's */

/* A host build with FMA enabled must not contract the exponential's products into its sums: the
 * device does not (-ffp-contract=off), and the masses have to be the same bits. */
#if defined(__GNUC__) && !defined(__clang__)
#define RAD_TRAV_STRICT __attribute__((optimize("fp-contract=off")))
#else
#define RAD_TRAV_STRICT
#endif

/* Test-only switches; the engine passes the zero value. */
struct RadMtpTravOpt {
    int  ctl = 0;   /* negative controls: 1 token-level ratios, 2 no (1 - p) term, 3 q never
                       renormalised (the with-replacement update on without-replacement drafts),
                       4 a RheoSampling node walked as a Gumbel one */
    /* Enumeration (tests): >= 0 replaces the uniforms -- test number `accept_at` (1-based) accepts
     * and every earlier one rejects; `p_test[i]` receives the acceptance rate of test i + 1 and
     * `*n_tests` their count. */
    int     accept_at = -1;
    double* p_test = nullptr;
    int*    n_tests = nullptr;
    int     max_tests = 0;
    double  u_final = -1.0;   /* >= 0: the final token's uniform (rad_mtp_trav_walk) */
};

struct RadMtpTravOut {
    int     n = 0;                           /* nodes on the path, root first */
    int     node[RAD_MTP_TRAV_MAX_DEPTH + 2] = {};
    int32_t y = -1;                          /* the final token ... */
    bool    has_y = false;                   /* ... when the last node carries a valid state */
};

/* ---------------------------------------------------------------- the target law in fixed point */

/* The mass unit of a plane of n candidates: 2^-qbits of the top candidate's weight, so that a
 * row's total stays below 2^53 (exact in FP64). */
static inline RAD_RRS_HD int rad_mtp_trav_qbits(int64_t n) {
    int q = RAD_MTP_TRAV_QBITS;
    for (int64_t c = 4096; c < n && q > 20; c <<= 1) --q;
    return q;
}

/* floor(2^qbits * e^(v - top)) for a live candidate (v <= top, both finite): the FP32 exponential
 * of Cephes' expf (argument reduction by n ln 2 in two parts, a degree-5 polynomial in the reduced
 * argument), taken as a 24-bit mantissa and shifted. Zero below 2^-41 of the top. Basic IEEE FP32
 * operations and integer shifts only. */
RAD_TRAV_STRICT static inline RAD_RRS_HD uint64_t rad_mtp_trav_mass(float v, float top, int qbits) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float x = v - top;
    if (!(x > -28.5f)) return 0;
    const int   n  = (int)(x * 1.44269504088896341f - 0.5f);   /* round(x / ln 2), x <= 0 */
    const float fn = (float)n;
    float r = x - fn * 0.693359375f;
    r = r - fn * -2.12194440e-4f;
    const float z = r * r;
    float y = 1.9875691500e-4f;
    y = y * r + 1.3981999507e-3f;
    y = y * r + 8.3334519073e-3f;
    y = y * r + 4.1665795894e-2f;
    y = y * r + 1.6666665459e-1f;
    y = y * r + 5.0000001201e-1f;
    y = y * z + r;
    y = y + 1.0f;                                               /* e^r, in [0.70, 1.42] */
    const uint32_t mant = (uint32_t)(y * 8388608.0f);           /* y * 2^23, below 2^24 */
    const int sh = qbits + n - 23;
    if (sh >= 0) return (uint64_t)mant << sh;
    return sh > -32 ? (uint64_t)(mant >> -sh) : 0;
}

/* An integer below 2^53 as a double, exactly, the same way on host and device. */
static inline RAD_RRS_HD double rad_mtp_trav_u2d(uint64_t x) {
    return (double)(uint32_t)(x >> 32) * 4294967296.0 + (double)(uint32_t)x;
}
/* The integer part of a double in [0, 2^53). */
static inline RAD_RRS_HD uint64_t rad_mtp_trav_d2u(double d) {
    const uint32_t hi = (uint32_t)(d * (1.0 / 4294967296.0));
    const double   lo = d - (double)hi * 4294967296.0;
    return ((uint64_t)hi << 32) + (uint64_t)(uint32_t)lo;
}

/* Candidate id's slot in a node's support (the first match), or cnt when it is outside. */
static inline RAD_RRS_HD int rad_mtp_trav_slot(const int32_t* sid, int cnt, int32_t id) {
    int k = 0;
    while (k < cnt && id != sid[k]) ++k;
    return k;
}

/* A row's validity and largest value, as rad_mtp_couple_decide scans them: false for an id past the
 * vocabulary, a NaN or +inf among the ids >= 0, or no finite value at all. */
static inline RAD_RRS_HD bool rad_mtp_trav_row_top(const int32_t* ids, const float* val, int64_t n,
                                                   int64_t vocab, float* top_out) {
    float top = -INFINITY;
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] < 0) continue;
        if (ids[j] >= vocab || val[j] != val[j] || val[j] == INFINITY) return false;
        if (val[j] > top) top = val[j];
    }
    *top_out = top;
    return top != -INFINITY;
}

/* One row's target law over a node's support `sid[0 .. cnt)`: the masses on the support slots
 * (duplicate candidates aggregated at the first matching slot), off it, and the total. The serial
 * definition; a workgroup adds the same integers in its own order. */
struct RadTravLaw {
    uint64_t pq[8];
    uint64_t off, sum;
};
static inline RAD_RRS_HD void rad_mtp_trav_row_law(const int32_t* ids, const float* val, int64_t n,
                                                   float top, const int32_t* sid, int cnt,
                                                   RadTravLaw* o) {
    const int qb = rad_mtp_trav_qbits(n);
    for (int k = 0; k < 8; ++k) o->pq[k] = 0;
    o->off = 0; o->sum = 0;
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] < 0 || val[j] == -INFINITY) continue;
        const uint64_t m = rad_mtp_trav_mass(val[j], top, qb);
        o->sum += m;
        const int k = rad_mtp_trav_slot(sid, cnt, ids[j]);
        if (k < cnt) o->pq[k] += m; else o->off += m;
    }
}

/* The off-support candidate that holds unit `t` of the off-support mass, candidates in ascending
 * slot order (t < the row's off-support mass). -1 when there is none. */
static inline RAD_RRS_HD int32_t rad_mtp_trav_off_token(const int32_t* ids, const float* val,
                                                        int64_t n, float top, const int32_t* sid,
                                                        int cnt, uint64_t t) {
    const int qb = rad_mtp_trav_qbits(n);
    for (int64_t j = 0; j < n; ++j) {
        if (ids[j] < 0 || val[j] == -INFINITY) continue;
        if (rad_mtp_trav_slot(sid, cnt, ids[j]) < cnt) continue;
        const uint64_t m = rad_mtp_trav_mass(val[j], top, qb);
        if (t < m) return ids[j];
        t -= m;
    }
    return -1;
}

/* ---------------------------------------------------------------- the prepared tree */

/* The eight-wide FP64 sums of the walk, as one fixed tree: the xor-4, xor-2, xor-1 butterfly of
 * eight lanes (IEEE addition is commutative, so every lane ends with the same bits). */
static inline RAD_RRS_HD double rad_mtp_trav_sum8(const double* v) {
    return ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
}

/* A node's draft side in the walk's canonical order: the children first, in draft order (a chain's
 * one child is x). `nk` the children the walk may try, `nt` -1 for Gumbel-top-k siblings, else the
 * number of deterministic top-q children of a RheoSampling node (its tail child is slot nt, `tail`
 * whether the node carries it), `swap` the stored slot that changed places with slot 0 (a chain row
 * whose x was not first; else 0).
 * False when the state names no child (unreachable for a rad_mtp_couple_state_valid state). */
static inline RAD_RRS_HD bool rad_mtp_trav_canon(const RadMtpCoupleState& s, int ctl,
                                                 int32_t* ids, double* q, int* nk, int* nt,
                                                 int* tail, int* swap) {
    /* Every index below is a constant: a device lane keeps the state and result in registers. */
    *swap = 0;
    *tail = 0;
    if (s.scheme > 0 && ctl != 4) {
        for (int k = 0; k < 8; ++k) {
            ids[k] = k < s.count ? s.ids[k] : -1;
            q[k] = k < s.count ? s.q[k] : 0.0;
        }
        const int m = s.scheme - 1;
        const int t = m < s.count ? m : s.count;
        const int cap = t + (s.count > t ? 1 : 0);
        *nt = t;
        *nk = s.nkids < cap ? (s.nkids > 0 ? s.nkids : 0) : cap;
        *tail = *nk > t;
        return true;
    }
    *nt = -1;
    *nk = s.nkids >= 2 ? (s.nkids < s.count ? s.nkids : s.count) : 1;
    /* a chain row: x to the front (the first slot that holds it changes places with slot 0) */
    int kx = 0;
    if (s.nkids < 2) {
        kx = 8;
        for (int k = 7; k >= 0; --k)
            if (k < s.count && s.ids[k] == s.x) kx = k;
    }
    const bool found = kx < 8;
    int32_t idx = -1;
    double  qx = 0.0;
    for (int k = 0; k < 8; ++k) {
        idx = k == kx ? s.ids[k] : idx;
        qx = k == kx ? s.q[k] : qx;
    }
    for (int k = 0; k < 8; ++k) {
        const bool in = found && k < s.count;
        const int32_t idk = k == 0 ? idx : (k == kx ? s.ids[0] : s.ids[k]);
        const double  qk  = k == 0 ? qx  : (k == kx ? s.q[0]  : s.q[k]);
        ids[k] = in ? idk : -1;
        q[k] = in ? qk : 0.0;
    }
    *swap = found ? kx : 0;
    return found;
}

/* One request's tree as the walk reads it, node-indexed (node 0 the root, parents before
 * children). Plain data: the device keeps it in LDS and fills it with a workgroup. */
struct RadTravTree {
    int      n;                                   /* nodes */
    int      ok[RAD_MTP_TRAV_MAX_ROWS];           /* an eligible state with a valid target law */
    int      nk[RAD_MTP_TRAV_MAX_ROWS];           /* children the walk tries */
    int      nt[RAD_MTP_TRAV_MAX_ROWS];           /* -1 Gumbel; else a rheo node's top children */
    int      tail[RAD_MTP_TRAV_MAX_ROWS];         /* a rheo node that carries its tail child */
    int      ci[RAD_MTP_TRAV_MAX_ROWS];           /* children tried so far */
    int      child[RAD_MTP_TRAV_MAX_ROWS][8];     /* slot k's child node, or -1 */
    int32_t  ids[RAD_MTP_TRAV_MAX_ROWS][8];       /* the support, canonical order, -1 past it */
    double   qr[RAD_MTP_TRAV_MAX_ROWS][8];        /* the draft law (Gumbel: tried ones zeroed) */
    double   m[RAD_MTP_TRAV_MAX_ROWS][8];         /* rate * residual on the support slots (the
                                                     target's law itself until the node is
                                                     entered) */
    double   moff[RAD_MTP_TRAV_MAX_ROWS];         /* the same off the support, in all */
    double   sc[RAD_MTP_TRAV_MAX_ROWS];           /* the factor m and moff still owe: the node's
                                                     rate from its entry to its first rejected
                                                     child, then 1 */
    double   p[RAD_MTP_TRAV_MAX_ROWS];            /* the node's acceptance rate */
    uint64_t off[RAD_MTP_TRAV_MAX_ROWS];          /* the off-support mass in units (final draw) */
    float    top[RAD_MTP_TRAV_MAX_ROWS];          /* the row's largest value */
    int      stack[RAD_MTP_TRAV_MAX_DEPTH + 1];
};

/* Node v's laws from its canonical draft side and its row's integer law (`valid`: the row scanned
 * clean and its total is positive). */
static inline RAD_RRS_HD void rad_mtp_trav_set_law(RadTravTree* t, int v, bool valid, float top,
                                                   const RadTravLaw& w) {
    const bool ok = valid && w.sum > 0;
    const double sum = ok ? rad_mtp_trav_u2d(w.sum) : 1.0;
    for (int k = 0; k < 8; ++k) t->m[v][k] = ok ? rad_mtp_trav_u2d(w.pq[k]) / sum : 0.0;
    t->moff[v] = ok ? rad_mtp_trav_u2d(w.off) / sum : 0.0;
    t->off[v] = ok ? w.off : 0;
    t->top[v] = top;
    t->ci[v] = 0;
    t->ok[v] = ok;
}

/* The slot of node v's j-th tried child. Gumbel: the draft order. Rheo: the sampled tail child
 * first (when the node carries it), then the deterministic ones. */
static inline RAD_RRS_HD int rad_mtp_trav_try_slot(const RadTravTree& t, int v, int j) {
    const int nt = t.nt[v];
    if (nt < 0) return j;
    return t.tail[v] ? (j == 0 ? nt : j - 1) : j;
}

/* The law node v's j-th tried child was drawn from, at support slot k. `isq` is the reciprocal of
 * rad_mtp_trav_try_norm: the sum of the node's remaining draft law (Gumbel) or of its tail's
 * (rheo). */
static inline RAD_RRS_HD double rad_mtp_trav_try_norm(const RadTravTree& t, int v) {
    const int nt = t.nt[v];
    if (nt < 0) return rad_mtp_trav_sum8(t.qr[v]);
    double tail[8];
    for (int k = 0; k < 8; ++k) tail[k] = k >= nt ? t.qr[v][k] : 0.0;
    return rad_mtp_trav_sum8(tail);
}
static inline RAD_RRS_HD double rad_mtp_trav_try_q(const RadTravTree& t, int v, int j, double isq,
                                                   int k) {
    const int nt = t.nt[v];
    if (nt < 0) return t.qr[v][k] * isq;
    const int slot = rad_mtp_trav_try_slot(t, v, j);
    if (slot != nt) return k == slot ? 1.0 : 0.0;          /* a deterministic child: a point mass */
    if (k < nt) return 0.0;
    const double q = nt == 0 ? t.qr[v][k] : t.qr[v][k] * isq;
    return q > 1.0 ? 1.0 : q;   /* clamp: rounding must not push a one-token tail past 1 */
}
/* m / q for node v's j-th tried child at its own slot i, m the child's mass in the node's law: the
 * child's rate before the cap, in one division (m * sq / qr rather than m / (qr / sq)). Negative
 * when the law has no mass there (the child is not tried). */
static inline RAD_RRS_HD double rad_mtp_trav_try_ratio(const RadTravTree& t, int v, int j, int i,
                                                       double m) {
    const int nt = t.nt[v];
    const double q = t.qr[v][i];
    if (nt >= 0 && i != nt) return m;                       /* a point mass */
    if (!(q > 0)) return -1.0;
    if (nt == 0) return q < 1.0 ? m / q : m;
    const double sq = rad_mtp_trav_try_norm(t, v);
    return q < sq ? m * sq / q : m;                         /* the law's value is at most 1 */
}

/* The walk over a prepared tree: the accepted path into path[0 .. *n_path). Returns false when
 * nothing was decided (an invalid root state: every row keeps the pick). `pos0` is the root row's
 * output position, the coordinate of the acceptance uniforms. Updates the tree in place: the last
 * node's m / moff are its residual law (up to the node's factor sc), which
 * rad_mtp_trav_final_pick reads. */
static inline RAD_RRS_HD bool rad_mtp_trav_walk_tree(RadTravTree& t, uint64_t seed, uint64_t pos0,
                                                     int* path, int* n_path,
                                                     const RadMtpTravOpt& opt = RadMtpTravOpt{}) {
    int top = 0, tests = 0;
    t.stack[0] = 0;
    t.p[0] = 1.0;
    t.sc[0] = 1.0;   /* the root's m is the target's law itself */
    *n_path = 0;
    if (!t.ok[0]) return false;
    for (;;) {
        const int v = t.stack[top];
        if (t.ok[v] && t.ci[v] < t.nk[v]) {
            const int j = t.ci[v];
            const int i = rad_mtp_trav_try_slot(t, v, j);
            /* p_c = min(1, p_v r(c) / qn(c)); control 1 drops p_v (token-level ratios) */
            const double f = opt.ctl == 1 ? (t.p[v] > 0 ? t.sc[v] / t.p[v] : 0.0) : t.sc[v];
            const double ratio = rad_mtp_trav_try_ratio(t, v, j, i, f * t.m[v][i]);
            const int c = top < RAD_MTP_TRAV_MAX_DEPTH && ratio >= 0 ? t.child[v][i] : -1;
            if (c < 0) {   /* a child the verify no longer carries: cut here */
                t.nk[v] = j;
                continue;
            }
            const double pc = ratio < 1.0 ? ratio : 1.0;
            t.p[c] = pc;
            t.sc[c] = pc;   /* the child's law becomes rate * law, when it is first needed */
            t.stack[++top] = c;
            continue;
        }
        /* a leaf: the sequence-level test */
        ++tests;
        bool accept;
        if (opt.accept_at >= 0) {
            if (opt.p_test && tests <= opt.max_tests) opt.p_test[tests - 1] = t.p[v];
            if (opt.n_tests) *opt.n_tests = tests;
            accept = tests == opt.accept_at;
        } else {
            const double u = rad_spechub_uniform(
                seed, pos0, RAD_MTP_TRAV_KEY ^ rad_mtp_rrs_mix((uint64_t)tests));
            accept = u < t.p[v];
        }
        if (accept) {
            *n_path = top + 1;
            for (int i = 0; i <= top; ++i) path[i] = t.stack[i];
            return true;
        }
        if (top == 0) return false;   /* the root always accepts (p = 1); only roundoff gets here */
        --top;
        const int g = t.stack[top];
        const int j = t.ci[g];
        const double isq = 1.0 / rad_mtp_trav_try_norm(t, g);
        const double pold = t.p[g], sc = t.sc[g];
        double d[8];
        for (int k = 0; k < 8; ++k) {
            const double x = sc * t.m[g][k] - rad_mtp_trav_try_q(t, g, j, isq, k);
            d[k] = x > 0 ? x : 0;
        }
        const double moff = sc * t.moff[g];
        const double S = moff + rad_mtp_trav_sum8(d);
        if (!(S > 0) || !rad_spechub_finite(S)) {
            /* nothing of the target is left above the draft: this node cannot accept, nor can a
             * child of it */
            t.p[g] = 0;
            t.sc[g] = 0;
        } else {
            const double den = opt.ctl == 2 ? S : S + 1.0 - pold;
            const double inv = 1.0 / den;
            const double pn = S * inv;
            t.p[g] = pn < 1.0 ? pn : 1.0;   /* a rate is at most 1: rounding may not push it past */
            for (int k = 0; k < 8; ++k) t.m[g][k] = d[k] * inv;
            t.moff[g] = moff * inv;
            t.sc[g] = 1.0;
        }
        /* Gumbel siblings are drawn WITHOUT replacement: the remaining Q renormalises */
        if (t.nt[g] < 0 && opt.ctl != 3) t.qr[g][j] = 0;
        ++t.ci[g];
    }
}

/* The final token of node v's residual law (in proportion to m on the support slots and moff off
 * it: the node's factor sc is common to both and cancels), by the uniform u: the support slots
 * first, in the canonical order, then the off-support candidates in slot order. kind 0: no token
 * (the row keeps the pick); 1: the token of support slot `slot`; 2: the off-support candidate
 * holding unit `t` of the row's off-support mass (rad_mtp_trav_off_token). */
struct RadTravPick {
    int      kind, slot;
    uint64_t t;
};
static inline RAD_RRS_HD RadTravPick rad_mtp_trav_final_pick(const RadTravTree& t, int v,
                                                             double u) {
    RadTravPick o;
    o.kind = 0; o.slot = -1; o.t = 0;
    const double woff = t.moff[v];
    const double total = rad_mtp_trav_sum8(t.m[v]) + woff;
    if (!(total > 0) || !rad_spechub_finite(total)) return o;
    double c = u * total;
    int last = -1;
    for (int k = 0; k < 8; ++k) {
        const double m = t.m[v][k];
        if (m > 0) {
            last = k;
            c -= m;
            if (c < 0) { o.kind = 1; o.slot = k; return o; }
        }
    }
    const uint64_t off = t.off[v];
    if (woff > 0 && off > 0) {
        const double frac = c / woff;
        uint64_t tt = frac < 1.0 ? rad_mtp_trav_d2u(frac * rad_mtp_trav_u2d(off)) : off - 1;
        if (tt >= off) tt = off - 1;
        o.kind = 2; o.t = tt;
        return o;
    }
    if (last >= 0) { o.kind = 1; o.slot = last; }
    return o;
}

/* ---------------------------------------------------------------- a tree behind an interface
 *
 * The tree the walk reads (a Tree type provides):
 *   const RadMtpCoupleState* state(int v)   the node's bound state, or null (node is a leaf)
 *   bool cand(int v, const int32_t** ids, const float** val, int64_t* n)   target candidates
 *   int  child(int v, int32_t token)        the node's first child row with that token, or -1
 *   int64_t vocab()
 * rad_mtp_trav_walk prepares the nodes the walk can reach (in discovery order, at most
 * RAD_MTP_TRAV_MAX_ROWS), walks and draws the final token: the host reference and the tests. */
template <class Tree>
static inline bool rad_mtp_trav_walk(const Tree& T, int root, uint64_t seed, uint64_t pos0,
                                     RadMtpTravOut* out, RadMtpTravOpt opt = RadMtpTravOpt{},
                                     RadTravTree* keep = nullptr, int* ids_out = nullptr) {
    RadTravTree  local;
    RadTravTree& t = keep ? *keep : local;
    int id[RAD_MTP_TRAV_MAX_ROWS];           /* node -> the tree's own id */
    struct Row { const int32_t* ids; const float* val; int64_t n; };
    Row rows[RAD_MTP_TRAV_MAX_ROWS];
    t.n = 1;
    id[0] = root;
    for (int a = 0; a < t.n; ++a) {
        const int v = id[a];
        t.ok[a] = 0; t.nk[a] = 0; t.nt[a] = -1; t.tail[a] = 0; t.ci[a] = 0;
        t.moff[a] = 0; t.sc[a] = 1.0; t.p[a] = 0; t.off[a] = 0; t.top[a] = -INFINITY;
        for (int k = 0; k < 8; ++k) {
            t.child[a][k] = -1; t.ids[a][k] = -1; t.qr[a][k] = 0; t.m[a][k] = 0;
        }
        rows[a] = Row{ nullptr, nullptr, 0 };
        const RadMtpCoupleState* sp = T.state(v);
        if (!sp) continue;
        int swap = 0;
        if (!rad_mtp_trav_canon(*sp, opt.ctl, t.ids[a], t.qr[a], &t.nk[a], &t.nt[a], &t.tail[a],
                                &swap))
            continue;
        const int32_t* ids = nullptr; const float* val = nullptr;
        int64_t n = 0;
        if (!T.cand(v, &ids, &val, &n)) continue;
        rows[a] = Row{ ids, val, n };
        float top = -INFINITY;
        RadTravLaw w{};
        const bool valid = rad_mtp_trav_row_top(ids, val, n, T.vocab(), &top);
        if (valid) rad_mtp_trav_row_law(ids, val, n, top, t.ids[a], sp->count, &w);
        rad_mtp_trav_set_law(&t, a, valid, top, w);
        if (!t.ok[a]) continue;
        for (int k = 0; k < t.nk[a]; ++k) {
            const int c = T.child(v, t.ids[a][k]);
            if (c < 0 || t.n >= RAD_MTP_TRAV_MAX_ROWS) continue;
            t.child[a][k] = t.n;
            id[t.n++] = c;
        }
    }
    int path[RAD_MTP_TRAV_MAX_DEPTH + 1], np = 0;
    if (ids_out) for (int a = 0; a < t.n; ++a) ids_out[a] = id[a];
    if (!rad_mtp_trav_walk_tree(t, seed, pos0, path, &np, opt)) return false;
    out->n = np;
    for (int i = 0; i < np; ++i) out->node[i] = id[path[i]];
    out->has_y = false;
    const int f = path[np - 1];
    if (t.ok[f]) {
        const double ur = opt.u_final >= 0
            ? opt.u_final
            : rad_spechub_uniform(seed, pos0 + (uint64_t)(np - 1), RAD_SAMPLE_KEY_RESID);
        const RadTravPick pk = rad_mtp_trav_final_pick(t, f, ur);
        if (pk.kind == 1) { out->y = t.ids[f][pk.slot]; out->has_y = true; }
        if (pk.kind == 2) {
            int cnt = 0;
            while (cnt < 8 && t.ids[f][cnt] >= 0) ++cnt;
            out->y = rad_mtp_trav_off_token(rows[f].ids, rows[f].val, rows[f].n, t.top[f],
                                            t.ids[f], cnt, pk.t);
            out->has_y = out->y >= 0;
        }
    }
    return true;
}

#endif /* __cplusplus */
#endif /* RAD_MTP_TRAVERSAL_H */
