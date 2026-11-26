/* DeepSeek-V4.1 vision host kernels. The device path is libr4d; these exist so a host
 * backend and avx_check cover the same ops libref does, with libref's rounding. */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <cmath>

using namespace avx;

namespace {
float softplus20(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }
enum class Score { Sigmoid, SqrtSoftplus, Softmax };
}

AVX_KERNEL(ds41_vit_rope2d) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *rc = &a->t[1];
    const int64_t M = x->rank == 2 ? x->shape[0] : 0;
    const int64_t D = p_int(a, "head_dim", 0), H = p_int(a, "n_head", 0);
    const float theta = p_f32(a, "theta", 0);
    if (M <= 0 || M > p_int(a, "M", 0) || D <= 0 || D % 4 || H <= 0 || !(theta > 0) ||
        !std::isfinite(theta) || x->rank != 2 || x->shape[0] < M || x->shape[1] != 3 * H * D ||
        rc->rank != 2 || rc->shape[0] < 2 || rc->shape[1] < M)
        return RAD_E_SHAPE;
    if (rc->dtype != RAD_I32 || (x->dtype != RAD_BF16 && x->dtype != RAD_F32)) return RAD_E_DTYPE;
    const int64_t rot = D / 2, quad = D / 4;
    for (int64_t p = 0; p < M; ++p)
        for (int64_t i = 0; i < rot; ++i) {
            const int64_t axis = i / quad, j = i % quad;
            const float pos = (float)ldi(rc, axis * rc->stride[0] + p * rc->stride[1]);
            const float inv = 1.0f / sc_pow(theta, (float)(2 * j) / (float)rot);
            const float ang = pos * inv;
            const float c = sc_cos(ang), s = sc_sin(ang);
            for (int64_t h = 0; h < 2 * H; ++h) {
                const int64_t u = rowoff(x, p, 3 * H * D) + (h * D + i) * laststride(x);
                const int64_t v = u + rot * laststride(x);
                const float x0 = ldt(x, u), x1 = ldt(x, v);
                stt(x, u, narrow1(x->dtype, x0 * c - x1 * s));
                stt(x, v, narrow1(x->dtype, x1 * c + x0 * s));
            }
        }
    return RAD_OK;
}

AVX_KERNEL(ds41_pack3x3) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *rc = &a->t[1], *y = &a->t[2];
    const int64_t M = x->rank == 2 ? x->shape[0] : 0;
    const int64_t N = p_int(a, "n", 0), K = p_int(a, "merge", 0);
    if (M <= 0 || M > p_int(a, "M", 0) || N <= 0 || K < 2 || x->rank != 2 || x->shape[0] != M ||
        x->shape[1] != N || rc->rank != 2 || rc->shape[0] != 4 || rc->shape[1] != M ||
        y->rank != 2 || y->shape[1] != N * K * K)
        return RAD_E_SHAPE;
    if (rc->dtype != RAD_I32 || x->dtype != y->dtype ||
        (x->dtype != RAD_BF16 && x->dtype != RAD_F32))
        return RAD_E_DTYPE;
    int64_t r0 = 0;
    for (int64_t p = 0; p < M;) {
        const int64_t h = ldi(rc, 2 * rc->stride[0] + p * rc->stride[1]);
        const int64_t w = ldi(rc, 3 * rc->stride[0] + p * rc->stride[1]);
        if (h <= 0 || w <= 0 || h * w > M - p) return RAD_E_INVAL;
        const int64_t oh = (h + K - 1) / K, ow = (w + K - 1) / K;
        if (r0 + oh * ow > y->shape[0]) return RAD_E_SHAPE;
        for (int64_t oy = 0; oy < oh; ++oy)
            for (int64_t ox = 0; ox < ow; ++ox) {
                const int64_t r = r0 + oy * ow + ox;
                for (int64_t j = 0; j < N; ++j)
                    for (int64_t dy = 0; dy < K; ++dy)
                        for (int64_t dx = 0; dx < K; ++dx) {
                            const int64_t yy = oy * K + dy, xx = ox * K + dx;
                            const int64_t dst =
                                rowoff(y, r, N * K * K) + (j * K * K + dy * K + dx) * laststride(y);
                            if (yy < h && xx < w)
                                stt(y, dst, ldt(x, rowoff(x, p + yy * w + xx, N) + j * laststride(x)));
                            else
                                stt(y, dst, 0.0f);
                        }
            }
        r0 += oh * ow;
        p += h * w;
    }
    if (r0 != y->shape[0]) return RAD_E_SHAPE;
    return RAD_OK;
}

AVX_KERNEL(ds41_span_rows) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *s = &a->t[0], *nl = &a->t[1], *e = &a->t[2], *y = &a->t[3];
    const int64_t n = p_int(a, "n", 0);
    if (n <= 0 || rad_tensor_numel(s) < n || rad_tensor_numel(nl) < n || rad_tensor_numel(e) < n ||
        y->rank != 2 || y->shape[0] != 4 || y->shape[1] != n)
        return RAD_E_SHAPE;
    if (y->dtype != RAD_BF16 || s->dtype != y->dtype || nl->dtype != y->dtype ||
        e->dtype != y->dtype)
        return RAD_E_DTYPE;
    const RadTensor* src[3] = { s, nl, e };
    const int64_t row[3] = { 0, 2, 3 };
    for (int k = 0; k < 3; ++k)
        for (int64_t i = 0; i < n; ++i)
            stt(y, rowoff(y, row[k], n) + i * laststride(y), ldt(src[k], i * src[k]->stride[0]));
    return RAD_OK;
}

AVX_KERNEL(router_scored_topk_vl) {
    if (!rad_args_have(a, 6)) return RAD_E_INVAL;
    const char* name = p_str(a, "score", "");
    Score score;
    if (!std::strcmp(name, "sigmoid")) score = Score::Sigmoid;
    else if (!std::strcmp(name, "sqrtsoftplus")) score = Score::SqrtSoftplus;
    else if (!std::strcmp(name, "softmax")) score = Score::Softmax;
    else return RAD_E_UNSUPPORTED;
    const auto &lg = a->t[0], &bias = a->t[1], &vl = a->t[2], &mask = a->t[3], &ids = a->t[4],
               &w = a->t[5];
    const int64_t ne = p_int(a, "n_expert", 0), k = p_int(a, "top_k", 0);
    const float factor = p_f32(a, "route_scale", 0);
    const bool norm = p_int(a, "norm", 0) != 0;
    if (ne < 1 || ne > 512 || k < 1 || k > ne || !std::isfinite(factor) || factor <= 0 ||
        lg.rank != 2 || lg.shape[0] < 1 || lg.shape[1] != ne || bias.rank != 1 ||
        bias.shape[0] != ne || bias.dtype != RAD_F32 || vl.rank != 1 || vl.shape[0] != ne ||
        vl.dtype != RAD_F32 || mask.rank != 1 || mask.shape[0] < lg.shape[0] ||
        mask.dtype != RAD_I32 || ids.dtype != RAD_I32 || ids.rank != 2 || w.rank != 2 ||
        ids.shape[0] != lg.shape[0] || w.shape[0] != lg.shape[0] || ids.shape[1] != k ||
        w.shape[1] != k || !lg.data || !bias.data || !vl.data || !mask.data)
        return RAD_E_SHAPE;
    for (int64_t r = 0; r < lg.shape[0]; ++r) {
        const RadTensor* bsel = ldi(&mask, r * laststride(&mask)) != 0 ? &vl : &bias;
        float scores[512], ranked[512], chosen[512];
        int index[512];
        float mx = -INFINITY, den = 0;
        for (int64_t e = 0; e < ne; ++e) {
            scores[e] = ldt(&lg, rowoff(&lg, r, ne) + e * laststride(&lg));
            mx = std::max(mx, scores[e]);
        }
        for (int64_t e = 0; e < ne; ++e) {
            if (score == Score::Sigmoid) scores[e] = 1.0f / (1.0f + sc_exp(-scores[e]));
            else if (score == Score::SqrtSoftplus) scores[e] = std::sqrt(softplus20(scores[e]));
            else { scores[e] = std::exp(scores[e] - mx); den += scores[e]; }
        }
        for (int64_t e = 0; e < ne; ++e) {
            if (score == Score::Softmax) scores[e] /= den;
            ranked[e] = scores[e] + ldt(bsel, e * laststride(bsel));
        }
        float total = 0;
        for (int64_t j = 0; j < k; ++j) {
            int winner = -1;
            for (int64_t e = 0; e < ne; ++e)
                if (ranked[e] > -INFINITY && (winner < 0 || ranked[e] > ranked[winner]))
                    winner = (int)e;
            if (winner < 0) return RAD_E_SHAPE;
            index[j] = winner;
            chosen[j] = scores[winner];
            total += chosen[j];
            ranked[winner] = -INFINITY;
        }
        const float mult = norm ? factor / (total + 1e-20f) : factor;
        for (int64_t j = 0; j < k; ++j) {
            st_int_dt(ids.dtype, ids.data, rowoff(&ids, r, k) + j * laststride(&ids), index[j]);
            stt(&w, rowoff(&w, r, k) + j * laststride(&w), chosen[j] * mult);
        }
    }
    return RAD_OK;
}
