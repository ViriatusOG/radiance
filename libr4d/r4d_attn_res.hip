/* r4d_attn_res.hip -- block attention residuals (libref/ref_attn_res.cpp has the contract).
 *
 * One workgroup a token row. The depth's values are read from global memory three times rather
 * than held in LDS: at the model's width (2304) nine of them in f32 are 80 KiB, past a workgroup's
 * 64 KiB. Each pass is a block reduction, serial in its order, so the sums are the same at any
 * block size, and the only rounding is the bf16 store of y. The workgroup owns its row end to end,
 * so nothing crosses a row and a prefill's thousands of rows are thousands of independent blocks.
 *
 * Activations are bf16; the query and the gain are f32. States may sit at any plane pitch, the
 * last axis dense. Nothing here allocates or synchronises across workgroups (docs/PLUGIN.md §5).
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "r4d_plugin.h"
#include "r4d_launch.h"
#include "rad_plugin.h"

namespace r4d_attn_res {

constexpr int kThr = 256;
constexpr int kMaxK = 16;   /* depth: the states and cur (n_state <= 15) */
constexpr int kMaxQ = 4;    /* query rows */

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }
__device__ __forceinline__ uint16_t tobf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);   /* NaN stays NaN */
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

/* A sum over the workgroup, the same tree at every call. The trailing barrier keeps a later
 * reduction from overwriting `red` while a lagging thread still reads its total. */
__device__ float block_sum(float v, float* red) {
    const int tid = (int)threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (int s = kThr / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

struct Args {
    const uint16_t* st; long long st_ps, st_rs;   /* state planes: plane pitch, row pitch */
    const uint16_t* cur; long long cur_rs;
    const uint16_t* add; long long add_rs;        /* null when absent */
    const float* q;     long long q_rs;
    const float* w;     long long w_rs;           /* null when absent */
    uint16_t* y; long long y_ps, y_rs;
    int M, n, nS, nQ;
    float eps;
};

/* The depth's i-th value at column c of this row: a state plane, or cur plus addend. */
__device__ __forceinline__ float value(const Args& A, int t, int i, int c) {
    if (i < A.nS) return bf(A.st[i * A.st_ps + t * A.st_rs + c]);
    float x = bf(A.cur[t * A.cur_rs + c]);
    if (A.add) x += bf(A.add[t * A.add_rs + c]);
    return x;
}

/* sum_i p[q][i] * (v_i[c] * rstd[i]): the mix at column c, before the gain. */
__device__ __forceinline__ float mix_at(const Args& A, int t, int q, int c, int K,
                                        const float* rstd, const float (*p)[kMaxK]) {
    float m = 0.0f;
    for (int i = 0; i < K; ++i) m += p[q][i] * (value(A, t, i, c) * rstd[i]);
    return m;
}

__global__ __launch_bounds__(kThr) void attn_res_kernel(Args A) {
    __shared__ float red[kThr];
    __shared__ float rstd[kMaxK];       /* 1 / rms of each depth value */
    __shared__ float rq[kMaxQ];         /* 1 / rms of each mixed row, with w */
    __shared__ float p[kMaxQ][kMaxK];   /* scores, then the softmax over the depth */
    const int t = (int)blockIdx.x, tid = (int)threadIdx.x;
    const int K = A.nS + 1, n = A.n, nQ = A.nQ;

    for (int i = 0; i < K; ++i) {
        float ss = 0.0f;
        for (int c = tid; c < n; c += kThr) {
            const float x = value(A, t, i, c);
            ss += x * x;
        }
        const float tot = block_sum(ss, red);
        if (tid == 0) rstd[i] = 1.0f / sqrtf(tot / (float)n + A.eps);
        __syncthreads();
    }

    /* Scores (v_hat . query) / sqrt(n): the dot product of the unnormalised value, scaled by its
     * rstd, so v_hat is never stored. */
    const float inv_sqrt_n = 1.0f / sqrtf((float)n);
    for (int q = 0; q < nQ; ++q)
        for (int i = 0; i < K; ++i) {
            float d = 0.0f;
            for (int c = tid; c < n; c += kThr) d += value(A, t, i, c) * A.q[q * A.q_rs + c];
            const float tot = block_sum(d, red);
            if (tid == 0) p[q][i] = tot * rstd[i] * inv_sqrt_n;
            __syncthreads();
        }
    if (tid == 0) {
        for (int q = 0; q < nQ; ++q) {
            float mx = p[q][0];
            for (int i = 1; i < K; ++i) mx = fmaxf(mx, p[q][i]);
            float den = 0.0f;
            for (int i = 0; i < K; ++i) {
                p[q][i] = expf(p[q][i] - mx);
                den += p[q][i];
            }
            for (int i = 0; i < K; ++i) p[q][i] /= den;
        }
    }
    __syncthreads();

    for (int q = 0; q < nQ; ++q) {
        if (A.w) {
            /* The mixed row's own rms, before its gain: one reduction, then the store pass. */
            float ss = 0.0f;
            for (int c = tid; c < n; c += kThr) {
                const float m = mix_at(A, t, q, c, K, rstd, p);
                ss += m * m;
            }
            const float tot = block_sum(ss, red);
            if (tid == 0) rq[q] = 1.0f / sqrtf(tot / (float)n + A.eps);
            __syncthreads();
            for (int c = tid; c < n; c += kThr) {
                const float m = mix_at(A, t, q, c, K, rstd, p);
                A.y[q * A.y_ps + t * A.y_rs + c] = tobf(m * rq[q] * A.w[q * A.w_rs + c]);
            }
        } else {
            for (int c = tid; c < n; c += kThr)
                A.y[q * A.y_ps + t * A.y_rs + c] = tobf(mix_at(A, t, q, c, K, rstd, p));
        }
        __syncthreads();
    }
}

}  // namespace r4d_attn_res

/* ------------------------------------------------------------------ attn_res_mix */
using namespace r4d_attn_res;

int r4d_rad_attn_res_mix(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 6) return RAD_E_INVAL;
    const RadTensor* st = rad_arg_in(a, 0);
    const RadTensor* cur = rad_arg_in(a, 1);
    const RadTensor* add = rad_arg_in(a, 2);
    const RadTensor* q = rad_arg_in(a, 3);
    const RadTensor* w = rad_arg_in(a, 4);
    const RadTensor* y = rad_arg_in(a, 5);
    if (!cur || !q || !y) return RAD_E_INVAL;
    const long long m_cap = rad_args_geti_or(a, "M", 0), n = rad_args_geti_or(a, "n", 0);
    const long long nS = rad_args_geti_or(a, "n_state", 0), nQ = rad_args_geti_or(a, "n_query", 0);
    const float eps = (float)rad_args_getf_or(a, "eps", 1e-6);
    /* The rows are cur's: a launch sees the ranged `M` at its band's upper bound, so it is a
     * ceiling only (docs/OPS.md), and the states and the output may hold more rows. */
    long long M = cur->rank == 2 ? cur->shape[0] : 0;
    if (m_cap > 0 && m_cap < M) M = m_cap;
    if (M < 1 || n < 1 || nS < 0 || nS > kMaxK - 1 || nQ < 1 || nQ > kMaxQ) return RAD_E_SHAPE;
    if (cur->dtype != RAD_BF16 || y->dtype != RAD_BF16 || q->dtype != RAD_F32)
        return RAD_E_DTYPE;
    if ((add && add->dtype != RAD_BF16) || (w && w->dtype != RAD_F32)) return RAD_E_DTYPE;
    if (nS > 0 && !st) return RAD_E_INVAL;
    if (nS > 0 && st->dtype != RAD_BF16) return RAD_E_DTYPE;

    /* Rows at their own pitch, the last axis dense. */
    if (cur->rank != 2 || cur->shape[1] != n || cur->stride[1] != 1 || cur->shape[0] < M)
        return RAD_E_SHAPE;
    if (add && (add->rank != 2 || add->shape[1] != n || add->stride[1] != 1 || add->shape[0] < M))
        return RAD_E_SHAPE;
    if (q->rank != 2 || q->shape[1] != n || q->stride[1] != 1 || q->shape[0] < nQ)
        return RAD_E_SHAPE;
    if (w && (w->rank != 2 || w->shape[1] != n || w->stride[1] != 1 || w->shape[0] < nQ))
        return RAD_E_SHAPE;
    /* y is [nQ, rows, n]; with one query it may be a plain [rows, n], which a caller can trim to
     * the step's rows (a rank-3 view trims only its leading axis). */
    const bool y2 = y->rank == 2 && nQ == 1;
    if (y2 ? (y->shape[0] < M || y->shape[1] != n || y->stride[1] != 1)
           : (y->rank != 3 || y->shape[0] != nQ || y->shape[1] < M || y->shape[2] != n ||
              y->stride[2] != 1))
        return RAD_E_SHAPE;
    if (nS > 0 && (st->rank != 3 || st->shape[2] != n || st->stride[2] != 1 || st->shape[1] < M ||
                   st->shape[0] < nS))
        return RAD_E_SHAPE;

    Args A{};
    A.st = nS > 0 ? (const uint16_t*)st->data : nullptr;
    A.st_ps = nS > 0 ? st->stride[0] : 0;
    A.st_rs = nS > 0 ? st->stride[1] : 0;
    A.cur = (const uint16_t*)cur->data; A.cur_rs = cur->stride[0];
    A.add = add ? (const uint16_t*)add->data : nullptr;
    A.add_rs = add ? add->stride[0] : 0;
    A.q = (const float*)q->data; A.q_rs = q->stride[0];
    A.w = w ? (const float*)w->data : nullptr;
    A.w_rs = w ? w->stride[0] : 0;
    A.y = (uint16_t*)y->data; A.y_ps = y2 ? 0 : y->stride[0]; A.y_rs = y2 ? y->stride[0] : y->stride[1];
    A.M = (int)M; A.n = (int)n; A.nS = (int)nS; A.nQ = (int)nQ;
    A.eps = eps;
    R4D_KLAUNCH(R4D_KNAME(attn_res_kernel), dim3((unsigned)M), dim3(kThr), (hipStream_t)stream, A);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
