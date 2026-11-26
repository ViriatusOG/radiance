/* ref_attn_res.cpp -- block attention residuals (Inkling 2's AttnRes): a softmax over depth.
 *
 *   attn_res_mix  for each row t, the values V = [states[0][t], .., states[n_state-1][t], cur[t] +
 *                 addend[t]] are rms-normalised without a gain; for each query q the scores are
 *                 (v_i . query[q]) / sqrt(n), the softmax over i weights the normalised values, and
 *                 with `w` the mix is rms-normalised again and gained by w[q].
 *
 * The planes of `states` may sit further apart than M rows: a caller keeps a [max_blocks, max_tok,
 * n] buffer and passes its first n_state planes, so the plane stride is the tensor's own.
 */
#include <cstdint>
#include "ref_common.h"
#include "ref_ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace ref;

int ref_attn_res_mix(const RadArgs* a, RadStream) {
    if (a->n_t < 6) return RAD_E_INVAL;
    const RadTensor *st = &a->t[0], *cur = &a->t[1], *qry = &a->t[3], *y = &a->t[5];
    const RadTensor* add = t_in(a, 2);
    const RadTensor* wt = t_in(a, 4);
    if (!cur->data || !qry->data || !y->data) return RAD_E_INVAL;
    const int64_t m_cap = p_int(a, "M", 0), n = p_int(a, "n", 0);
    /* The rows are cur's, `M` only a ceiling: a launch sees it at its band's upper bound. */
    const int64_t M = n > 0 ? std::min<int64_t>(rows_of(cur, n), m_cap > 0 ? m_cap : INT64_MAX) : 0;
    const int64_t nS = p_int(a, "n_state", 0), nQ = p_int(a, "n_query", 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    if (M <= 0 || n <= 0 || nS < 0 || nQ <= 0) return RAD_E_SHAPE;
    if (nS > 0 && (!st->data || st->rank != 3 || st->shape[2] != n || st->shape[1] < M ||
                   st->shape[0] < nS))
        return RAD_E_SHAPE;
    /* y is [nQ, rows, n], or a plain [rows, n] with one query. */
    const bool y2 = y->rank == 2 && nQ == 1;
    if (y2 ? (y->shape[0] < M || y->shape[1] != n)
           : (y->rank != 3 || y->shape[0] != nQ || y->shape[1] < M || y->shape[2] != n))
        return RAD_E_SHAPE;
    if (numel(cur) < M * n || numel(qry) < nQ * n) return RAD_E_SHAPE;
    if (add && numel(add) < M * n) return RAD_E_SHAPE;
    const bool wt1 = wt && wt->rank == 1 && nQ == 1 && wt->shape[0] == n;
    if (qry->rank != 2 || qry->shape[1] != n ||
        (wt && !wt1 && (wt->rank != 2 || wt->shape[1] != n)))
        return RAD_E_SHAPE;
    if (wt && numel(wt) < nQ * n) return RAD_E_SHAPE;

    const int64_t K = nS + 1;
    const float inv_sqrt_n = 1.0f / std::sqrt((float)n);
    const int64_t cs = laststride(cur), qs = laststride(qry), ys = laststride(y);
    const int64_t ss = st && st->data && st->rank == 3 ? laststride(st) : 1;
    const int64_t as = add ? laststride(add) : 1, ws = wt ? laststride(wt) : 1;

    REF_PAR_FOR(M * K * n * nQ)
    for (int64_t t = 0; t < M; ++t) {
        /* f32 throughout; the one rounding is the store into y. */
        std::vector<float> v((size_t)(K * n));
        for (int64_t i = 0; i < K; ++i) {
            for (int64_t c = 0; c < n; ++c) {
                float x;
                if (i < nS) {
                    x = ldt(st, i * st->stride[0] + t * st->stride[1] + c * ss);
                } else {
                    const int64_t cb = rowoff(cur, t, n) + c * cs;
                    x = ldt(cur, cb);
                    if (add) x += ldt(add, rowoff(add, t, n) + c * as);
                }
                v[(size_t)(i * n + c)] = x;
            }
        }
        /* The rms of each value, as rmsnorm's own: x * rsqrt(mean(x^2) + eps). */
        for (int64_t i = 0; i < K; ++i) {
            float ss2 = 0.0f;
            for (int64_t c = 0; c < n; ++c) ss2 += v[(size_t)(i * n + c)] * v[(size_t)(i * n + c)];
            const float r = 1.0f / std::sqrt(ss2 / (float)n + eps);
            for (int64_t c = 0; c < n; ++c) v[(size_t)(i * n + c)] *= r;
        }
        for (int64_t q = 0; q < nQ; ++q) {
            std::vector<float> p((size_t)K), m((size_t)n, 0.0f);
            float mx = -INFINITY;
            for (int64_t i = 0; i < K; ++i) {
                float d = 0.0f;
                for (int64_t c = 0; c < n; ++c)
                    d += v[(size_t)(i * n + c)] * ldt(qry, q * qry->stride[0] + c * qs);
                p[(size_t)i] = d * inv_sqrt_n;
                mx = std::max(mx, p[(size_t)i]);
            }
            float den = 0.0f;
            for (int64_t i = 0; i < K; ++i) {
                p[(size_t)i] = std::exp(p[(size_t)i] - mx);
                den += p[(size_t)i];
            }
            for (int64_t i = 0; i < K; ++i) {
                const float w = p[(size_t)i] / den;
                for (int64_t c = 0; c < n; ++c) m[(size_t)c] += w * v[(size_t)(i * n + c)];
            }
            if (wt) {
                float ss2 = 0.0f;
                for (int64_t c = 0; c < n; ++c) ss2 += m[(size_t)c] * m[(size_t)c];
                const float r = 1.0f / std::sqrt(ss2 / (float)n + eps);
                for (int64_t c = 0; c < n; ++c)
                    m[(size_t)c] = m[(size_t)c] * r * ldt(wt,
                                                          (wt1 ? 0 : q * wt->stride[0]) + c * ws);
            }
            const int64_t yb = y2 ? t * y->stride[0] : q * y->stride[0] + t * y->stride[1];
            for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, m[(size_t)c]);
        }
    }
    return RAD_OK;
}
