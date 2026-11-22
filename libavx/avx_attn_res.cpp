/* avx_attn_res.cpp -- block attention residuals, the contract of libref/ref_attn_res.cpp.
 *
 * One row t at a time: the depth's values are normalised into the thread's scratch, each query's
 * scores and softmax are taken over that depth, and the mix is written once. The row's values are
 * read once per query, which is cheap next to the depth-times-n multiply that dominates. f32 all
 * the way, so the one rounding is the store into y.
 */
#include <cstdint>
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <cmath>

using namespace avx;

AVX_KERNEL(attn_res_mix) {
    if (a->n_t < 6) return RAD_E_INVAL;
    const RadTensor *st = &a->t[0], *cur = &a->t[1], *qry = &a->t[3], *y = &a->t[5];
    const RadTensor* add = rad_arg_in(a, 2);
    const RadTensor* wt = rad_arg_in(a, 4);
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
    if (qry->rank != 2 || qry->shape[1] != n || (wt && (wt->rank != 2 || wt->shape[1] != n)))
        return RAD_E_SHAPE;
    if (add && numel(add) < M * n) return RAD_E_SHAPE;
    if (wt && numel(wt) < nQ * n) return RAD_E_SHAPE;

    const int64_t K = nS + 1;
    /* The depth's scores sit on the stack; a model has at most a dozen blocks. */
    if (K > 64) return RAD_E_UNSUPPORTED;
    const float inv_sqrt_n = 1.0f / std::sqrt((float)n);
    const int64_t cs = laststride(cur), qs = laststride(qry), ys = laststride(y);
    const int64_t ss = nS > 0 ? laststride(st) : 1;
    const int64_t as = add ? laststride(add) : 1, ws = wt ? laststride(wt) : 1;

    AVX_PARALLEL_FOR_IF(M * K * n * nQ >= AVX_PAR_MIN)
    for (int64_t t = 0; t < M; ++t) {
        float* v = scratch_f32((size_t)(K * n));
        float* m = scratch_f32_b((size_t)n);
        for (int64_t i = 0; i < K; ++i) {
            float* vi = v + i * n;
            if (i < nS) {
                const int64_t base = i * st->stride[0] + t * st->stride[1];
                for (int64_t c = 0; c < n; ++c) vi[c] = ldt(st, base + c * ss);
            } else {
                const int64_t cb = rowoff(cur, t, n), ab = add ? rowoff(add, t, n) : 0;
                for (int64_t c = 0; c < n; ++c) {
                    float x = ldt(cur, cb + c * cs);
                    if (add) x += ldt(add, ab + c * as);
                    vi[c] = x;
                }
            }
            float ss2 = 0.0f;
            for (int64_t c = 0; c < n; ++c) ss2 += vi[c] * vi[c];
            const float r = 1.0f / std::sqrt(ss2 / (float)n + eps);
            for (int64_t c = 0; c < n; ++c) vi[c] *= r;
        }
        for (int64_t q = 0; q < nQ; ++q) {
            /* The scores over the depth, then the softmax with its maximum subtracted. */
            float sc[64];
            float mx = -INFINITY;
            for (int64_t i = 0; i < K; ++i) {
                const float* vi = v + i * n;
                float d = 0.0f;
                for (int64_t c = 0; c < n; ++c) d += vi[c] * ldt(qry, q * qry->stride[0] + c * qs);
                sc[i] = d * inv_sqrt_n;
                mx = std::max(mx, sc[i]);
            }
            float den = 0.0f;
            for (int64_t i = 0; i < K; ++i) {
                sc[i] = std::exp(sc[i] - mx);
                den += sc[i];
            }
            for (int64_t c = 0; c < n; ++c) m[c] = 0.0f;
            for (int64_t i = 0; i < K; ++i) {
                const float p = sc[i] / den;
                const float* vi = v + i * n;
                for (int64_t c = 0; c < n; ++c) m[c] += p * vi[c];
            }
            if (wt) {
                float ss2 = 0.0f;
                for (int64_t c = 0; c < n; ++c) ss2 += m[c] * m[c];
                const float r = 1.0f / std::sqrt(ss2 / (float)n + eps);
                for (int64_t c = 0; c < n; ++c)
                    m[c] = m[c] * r * ldt(wt, q * wt->stride[0] + c * ws);
            }
            const int64_t yb = y2 ? t * y->stride[0] : q * y->stride[0] + t * y->stride[1];
            for (int64_t c = 0; c < n; ++c) stt(y, yb + c * ys, m[c]);
        }
    }
    return RAD_OK;
}
