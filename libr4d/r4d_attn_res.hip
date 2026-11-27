/* r4d_attn_res.hip -- block attention residuals (libref/ref_attn_res.cpp has the contract).
 *
 * One workgroup a token row (256 threads = 8 waves). Three kernels serve the shape:
 *   0. Vector path (one query, no addend, `K <= 5`, `n % 8 == 0` and `n <= 4096`, 16-byte aligned
 *      rows): the fast path's scheme with each thread loading eight values at a time.
 *   1. Fast register-cached path (`K <= 5 && n <= 3072`): reads states, cur (+ addend), and query
 *      from global memory ONCE into thread registers, reduces all K + nQ * K scalars together via
 *      DPP wave sums + 1 LDS wave merge barrier, broadcasts softmax * rstd in 1 barrier, and emits
 *      y directly from registers.
 *   2. General fallback (`K > 5 || n > 3072`): arbitrary depth up to 16 and width, using DPP wave
 *      sums + 8-wave LDS merge (2 barriers per reduction).
 * All three support r4dchain when requested; each sums in its own order, so the kernel chosen
 * for a shape is part of its result's bits.
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "r4d_chain.h"
#include "r4d_common.h"
#include "r4d_launch.h"
#include "r4d_plugin.h"
#include "rad_plugin.h"

namespace r4d_attn_res {

constexpr int kThr      = 256;
constexpr int kWaves    = kThr / 32;
constexpr int kMaxK       = 16;   /* depth: the states and cur (n_state <= 15) */
constexpr int kMaxQ       = 4;    /* query rows */
constexpr int kFastK      = 5;
constexpr int kMaxElem    = 12;   /* 12 * 256 = 3072 */
constexpr int kVec8Elem   = 2;    /* 2 * 8 * 256 = 4096 */

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }
__device__ __forceinline__ uint16_t tobf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);   /* NaN stays NaN */
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

struct Args {
    const uint16_t* st; long long st_ps, st_rs;   /* state planes: plane pitch, row pitch */
    const uint16_t* cur; long long cur_rs;
    const uint16_t* add; long long add_rs;        /* null when absent */
    const float* q;     long long q_rs;
    const void*  w;     long long w_rs;           /* f32 or bf16 gain, null when absent */
    uint16_t* y; long long y_ps, y_rs;
    int M, n, nS, nQ, w_bf16;
    float eps;
};

__device__ __forceinline__ float load_w(const Args& A, long long idx) {
    return A.w_bf16 ? bf(((const uint16_t*)A.w)[idx]) : ((const float*)A.w)[idx];
}

__device__ __forceinline__ float value(const Args& A, int t, int i, int c) {
    if (i < A.nS) return bf(A.st[i * A.st_ps + t * A.st_rs + c]);
    float x = bf(A.cur[t * A.cur_rs + c]);
    if (A.add) x += bf(A.add[t * A.add_rs + c]);
    return x;
}

__device__ __forceinline__ float mix_at(const Args& A, int t, int q, int c, int K,
                                        const float* rstd, const float (*p)[kMaxK]) {
    float m = 0.0f;
    for (int i = 0; i < K; ++i) m += p[q][i] * (value(A, t, i, c) * rstd[i]);
    return m;
}

__device__ __forceinline__ void attn_res_fast_body(Args A) {
    constexpr int kRedCols = kFastK + kMaxQ * kFastK;
    __shared__ float s_red[kWaves][kRedCols];
    __shared__ float s_pr[kMaxQ][kFastK];
    __shared__ float s_wsm[kMaxQ][kWaves];
    __shared__ float s_rq[kMaxQ];

    const int t = (int)blockIdx.x, tid = (int)threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int K = A.nS + 1, n = A.n, nQ = A.nQ;

    float val[kFastK][kMaxElem];
    float ss[kFastK];
    float dot[kMaxQ][kFastK];
#pragma unroll
    for (int i = 0; i < kFastK; ++i) {
        ss[i] = 0.0f;
#pragma unroll
        for (int q = 0; q < kMaxQ; ++q) dot[q][i] = 0.0f;
    }

#pragma unroll
    for (int e = 0; e < kMaxElem; ++e) {
        const int c = tid + e * kThr;
        if (c < n) {
            float qv[kMaxQ];
#pragma unroll
            for (int q = 0; q < kMaxQ; ++q)
                if (q < nQ) qv[q] = A.q[q * A.q_rs + c];
#pragma unroll
            for (int i = 0; i < kFastK; ++i) {
                if (i < K) {
                    const float x = value(A, t, i, c);
                    val[i][e] = x;
                    ss[i] += x * x;
#pragma unroll
                    for (int q = 0; q < kMaxQ; ++q)
                        if (q < nQ) dot[q][i] += x * qv[q];
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < kFastK; ++i) {
        if (i < K) {
            const float rs = r4d_wave_sum(ss[i]);
            if (lane == 0) s_red[wave][i] = rs;
#pragma unroll
            for (int q = 0; q < kMaxQ; ++q) {
                if (q < nQ) {
                    const float ds = r4d_wave_sum(dot[q][i]);
                    if (lane == 0) s_red[wave][kFastK + q * kFastK + i] = ds;
                }
            }
        }
    }
    lds_barrier();

    if (tid == 0) {
        float rstd[kFastK];
        for (int i = 0; i < K; ++i) {
            float tot = 0.0f;
            for (int w = 0; w < kWaves; ++w) tot += s_red[w][i];
            rstd[i] = 1.0f / sqrtf(tot / (float)n + A.eps);
        }
        const float inv_sqrt_n = 1.0f / sqrtf((float)n);
        for (int q = 0; q < nQ; ++q) {
            float sc[kFastK];
            for (int i = 0; i < K; ++i) {
                float tot = 0.0f;
                for (int w = 0; w < kWaves; ++w) tot += s_red[w][kFastK + q * kFastK + i];
                sc[i] = tot * rstd[i] * inv_sqrt_n;
            }
            float mx = sc[0];
            for (int i = 1; i < K; ++i) mx = fmaxf(mx, sc[i]);
            float den = 0.0f;
            for (int i = 0; i < K; ++i) {
                sc[i] = expf(sc[i] - mx);
                den += sc[i];
            }
            const float inv_den = 1.0f / den;
            for (int i = 0; i < K; ++i) s_pr[q][i] = (sc[i] * inv_den) * rstd[i];
        }
    }
    lds_barrier();

    for (int q = 0; q < nQ; ++q) {
        if (A.w) {
            float mval[kMaxElem];
            float sm = 0.0f;
#pragma unroll
            for (int e = 0; e < kMaxElem; ++e) {
                const int c = tid + e * kThr;
                if (c < n) {
                    float m = 0.0f;
#pragma unroll
                    for (int i = 0; i < kFastK; ++i)
                        if (i < K) m += s_pr[q][i] * val[i][e];
                    mval[e] = m;
                    sm += m * m;
                }
            }
            sm = r4d_wave_sum(sm);
            if (lane == 0) s_wsm[q][wave] = sm;
            lds_barrier();
            if (tid == 0) {
                float tot = 0.0f;
                for (int w = 0; w < kWaves; ++w) tot += s_wsm[q][w];
                s_rq[q] = 1.0f / sqrtf(tot / (float)n + A.eps);
            }
            lds_barrier();
            const float rq = s_rq[q];
#pragma unroll
            for (int e = 0; e < kMaxElem; ++e) {
                const int c = tid + e * kThr;
                if (c < n)
                    A.y[q * A.y_ps + t * A.y_rs + c] =
                        tobf(mval[e] * rq * load_w(A, q * A.w_rs + c));
            }
        } else {
#pragma unroll
            for (int e = 0; e < kMaxElem; ++e) {
                const int c = tid + e * kThr;
                if (c < n) {
                    float m = 0.0f;
#pragma unroll
                    for (int i = 0; i < kFastK; ++i)
                        if (i < K) m += s_pr[q][i] * val[i][e];
                    A.y[q * A.y_ps + t * A.y_rs + c] = tobf(m);
                }
            }
        }
    }
}

template <class CH>
__global__ __launch_bounds__(kThr) void attn_res_fast_kernel(Args A) {
    if (CH::on) r4d_chain_wait();
    attn_res_fast_body(A);
    if (CH::on) r4d_chain_done();
}

__device__ __forceinline__ void attn_res_vec8_body(Args A) {
    constexpr int kRedCols = kFastK + kFastK;
    __shared__ float s_red[kWaves][kRedCols];
    __shared__ float s_pr[kFastK];
    __shared__ float s_wsm[kWaves];
    __shared__ float s_rq;

    const int t = (int)blockIdx.x, tid = (int)threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int K = A.nS + 1, n = A.n;

    uint4 raw[kFastK][kVec8Elem];
    float ss[kFastK], dot[kFastK];
#pragma unroll
    for (int i = 0; i < kFastK; ++i) {
        ss[i] = 0.0f;
        dot[i] = 0.0f;
    }

#pragma unroll
    for (int e = 0; e < kVec8Elem; ++e) {
        const int c8 = (tid + e * kThr) * 8;
        if (c8 < n) {
            const float4 q0 = *(const float4*)(A.q + c8);
            const float4 q1 = *(const float4*)(A.q + c8 + 4);
#pragma unroll
            for (int i = 0; i < kFastK; ++i) {
                if (i < K) {
                    const uint16_t* ptr = (i < A.nS)
                        ? (A.st + i * A.st_ps + t * A.st_rs + c8)
                        : (A.cur + t * A.cur_rs + c8);
                    const uint4 v = *(const uint4*)ptr;
                    raw[i][e] = v;
                    const float x0 = bf16w_lo(v.x), x1 = bf16w_hi(v.x);
                    const float x2 = bf16w_lo(v.y), x3 = bf16w_hi(v.y);
                    const float x4 = bf16w_lo(v.z), x5 = bf16w_hi(v.z);
                    const float x6 = bf16w_lo(v.w), x7 = bf16w_hi(v.w);
                    ss[i] += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3 +
                             x4 * x4 + x5 * x5 + x6 * x6 + x7 * x7;
                    dot[i] += x0 * q0.x + x1 * q0.y + x2 * q0.z + x3 * q0.w +
                              x4 * q1.x + x5 * q1.y + x6 * q1.z + x7 * q1.w;
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < kFastK; ++i) {
        if (i < K) {
            const float rs = r4d_wave_sum(ss[i]);
            if (lane == 0) s_red[wave][i] = rs;
            const float ds = r4d_wave_sum(dot[i]);
            if (lane == 0) s_red[wave][kFastK + i] = ds;
        }
    }
    lds_barrier();

    if (tid == 0) {
        float rstd[kFastK], sc[kFastK];
        const float inv_sqrt_n = 1.0f / sqrtf((float)n);
        for (int i = 0; i < K; ++i) {
            float tot = 0.0f, dtot = 0.0f;
            for (int w = 0; w < kWaves; ++w) {
                tot  += s_red[w][i];
                dtot += s_red[w][kFastK + i];
            }
            rstd[i] = 1.0f / sqrtf(tot / (float)n + A.eps);
            sc[i] = dtot * rstd[i] * inv_sqrt_n;
        }
        float mx = sc[0];
        for (int i = 1; i < K; ++i) mx = fmaxf(mx, sc[i]);
        float den = 0.0f;
        for (int i = 0; i < K; ++i) {
            sc[i] = expf(sc[i] - mx);
            den += sc[i];
        }
        const float inv_den = 1.0f / den;
        for (int i = 0; i < K; ++i) s_pr[i] = (sc[i] * inv_den) * rstd[i];
    }
    lds_barrier();

    auto mix8 = [&](int e, float (&m)[8]) {
#pragma unroll
        for (int k = 0; k < 8; ++k) m[k] = 0.0f;
#pragma unroll
        for (int i = 0; i < kFastK; ++i) {
            if (i < K) {
                const float pr = s_pr[i];
                const uint4 v  = raw[i][e];
                m[0] += pr * bf16w_lo(v.x); m[1] += pr * bf16w_hi(v.x);
                m[2] += pr * bf16w_lo(v.y); m[3] += pr * bf16w_hi(v.y);
                m[4] += pr * bf16w_lo(v.z); m[5] += pr * bf16w_hi(v.z);
                m[6] += pr * bf16w_lo(v.w); m[7] += pr * bf16w_hi(v.w);
            }
        }
    };

    if (A.w) {
        float sm = 0.0f;
#pragma unroll
        for (int e = 0; e < kVec8Elem; ++e) {
            const int c8 = (tid + e * kThr) * 8;
            if (c8 < n) {
                float m[8];
                mix8(e, m);
#pragma unroll
                for (int k = 0; k < 8; ++k) sm += m[k] * m[k];
            }
        }
        sm = r4d_wave_sum(sm);
        if (lane == 0) s_wsm[wave] = sm;
        lds_barrier();
        if (tid == 0) {
            float tot = 0.0f;
            for (int w = 0; w < kWaves; ++w) tot += s_wsm[w];
            s_rq = 1.0f / sqrtf(tot / (float)n + A.eps);
        }
        lds_barrier();
        const float rq = s_rq;
#pragma unroll
        for (int e = 0; e < kVec8Elem; ++e) {
            const int c8 = (tid + e * kThr) * 8;
            if (c8 < n) {
                float m[8];
                mix8(e, m);
                if (A.w_bf16) {
                    const uint4 vw = *(const uint4*)((const uint16_t*)A.w + c8);
                    *(uint4*)(A.y + t * A.y_rs + c8) = make_uint4(
                        bf16x2_rne(m[0] * rq * bf16w_lo(vw.x), m[1] * rq * bf16w_hi(vw.x)),
                        bf16x2_rne(m[2] * rq * bf16w_lo(vw.y), m[3] * rq * bf16w_hi(vw.y)),
                        bf16x2_rne(m[4] * rq * bf16w_lo(vw.z), m[5] * rq * bf16w_hi(vw.z)),
                        bf16x2_rne(m[6] * rq * bf16w_lo(vw.w), m[7] * rq * bf16w_hi(vw.w)));
                } else {
                    const float4 w0 = *(const float4*)((const float*)A.w + c8);
                    const float4 w1 = *(const float4*)((const float*)A.w + c8 + 4);
                    *(uint4*)(A.y + t * A.y_rs + c8) = make_uint4(
                        bf16x2_rne(m[0] * rq * w0.x, m[1] * rq * w0.y),
                        bf16x2_rne(m[2] * rq * w0.z, m[3] * rq * w0.w),
                        bf16x2_rne(m[4] * rq * w1.x, m[5] * rq * w1.y),
                        bf16x2_rne(m[6] * rq * w1.z, m[7] * rq * w1.w));
                }
            }
        }
    } else {
#pragma unroll
        for (int e = 0; e < kVec8Elem; ++e) {
            const int c8 = (tid + e * kThr) * 8;
            if (c8 < n) {
                float m[8];
                mix8(e, m);
                *(uint4*)(A.y + t * A.y_rs + c8) = make_uint4(
                    bf16x2_rne(m[0], m[1]), bf16x2_rne(m[2], m[3]),
                    bf16x2_rne(m[4], m[5]), bf16x2_rne(m[6], m[7]));
            }
        }
    }
}

template <class CH>
__global__ __launch_bounds__(kThr) void attn_res_vec8_kernel(Args A) {
    if (CH::on) r4d_chain_wait();
    attn_res_vec8_body(A);
    if (CH::on) r4d_chain_done();
}

__device__ float block_sum_dpp(float v, float* red) {
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const float wv = r4d_wave_sum(v);
    if (lane == 0) red[wave] = wv;
    lds_barrier();
    float r = 0.0f;
    for (int w = 0; w < kWaves; ++w) r += red[w];
    lds_barrier();
    return r;
}

__device__ __forceinline__ void attn_res_general_body(Args A) {
    __shared__ float red[kWaves];
    __shared__ float rstd[kMaxK];
    __shared__ float rq[kMaxQ];
    __shared__ float p[kMaxQ][kMaxK];
    const int t = (int)blockIdx.x, tid = (int)threadIdx.x;
    const int K = A.nS + 1, n = A.n, nQ = A.nQ;

    for (int i = 0; i < K; ++i) {
        float ss = 0.0f;
        for (int c = tid; c < n; c += kThr) {
            const float x = value(A, t, i, c);
            ss += x * x;
        }
        const float tot = block_sum_dpp(ss, red);
        if (tid == 0) rstd[i] = 1.0f / sqrtf(tot / (float)n + A.eps);
        lds_barrier();
    }

    const float inv_sqrt_n = 1.0f / sqrtf((float)n);
    for (int q = 0; q < nQ; ++q)
        for (int i = 0; i < K; ++i) {
            float d = 0.0f;
            for (int c = tid; c < n; c += kThr) d += value(A, t, i, c) * A.q[q * A.q_rs + c];
            const float tot = block_sum_dpp(d, red);
            if (tid == 0) p[q][i] = tot * rstd[i] * inv_sqrt_n;
            lds_barrier();
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
    lds_barrier();

    for (int q = 0; q < nQ; ++q) {
        if (A.w) {
            float ss = 0.0f;
            for (int c = tid; c < n; c += kThr) {
                const float m = mix_at(A, t, q, c, K, rstd, p);
                ss += m * m;
            }
            const float tot = block_sum_dpp(ss, red);
            if (tid == 0) rq[q] = 1.0f / sqrtf(tot / (float)n + A.eps);
            lds_barrier();
            for (int c = tid; c < n; c += kThr) {
                const float m = mix_at(A, t, q, c, K, rstd, p);
                A.y[q * A.y_ps + t * A.y_rs + c] =
                    tobf(m * rq[q] * load_w(A, q * A.w_rs + c));
            }
        } else {
            for (int c = tid; c < n; c += kThr)
                A.y[q * A.y_ps + t * A.y_rs + c] = tobf(mix_at(A, t, q, c, K, rstd, p));
        }
        lds_barrier();
    }
}

template <class CH>
__global__ __launch_bounds__(kThr) void attn_res_kernel(Args A) {
    if (CH::on) r4d_chain_wait();
    attn_res_general_body(A);
    if (CH::on) r4d_chain_done();
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
    long long M = cur->rank == 2 ? cur->shape[0] : 0;
    if (m_cap > 0 && m_cap < M) M = m_cap;
    if (M < 1 || n < 1 || nS < 0 || nS > kMaxK - 1 || nQ < 1 || nQ > kMaxQ) return RAD_E_SHAPE;
    if (cur->dtype != RAD_BF16 || y->dtype != RAD_BF16 || q->dtype != RAD_F32)
        return RAD_E_DTYPE;
    if ((add && add->dtype != RAD_BF16) || (w && w->dtype != RAD_F32 && w->dtype != RAD_BF16))
        return RAD_E_DTYPE;
    if (nS > 0 && !st) return RAD_E_INVAL;
    if (nS > 0 && st->dtype != RAD_BF16) return RAD_E_DTYPE;

    if (cur->rank != 2 || cur->shape[1] != n || cur->stride[1] != 1 || cur->shape[0] < M)
        return RAD_E_SHAPE;
    if (add && (add->rank != 2 || add->shape[1] != n || add->stride[1] != 1 || add->shape[0] < M))
        return RAD_E_SHAPE;
    if (q->rank != 2 || q->shape[1] != n || q->stride[1] != 1 || q->shape[0] < nQ)
        return RAD_E_SHAPE;
    const bool w1 = w && w->rank == 1 && nQ == 1;
    if (w && (w1 ? (w->shape[0] != n || w->stride[0] != 1)
                 : (w->rank != 2 || w->shape[1] != n || w->stride[1] != 1 || w->shape[0] < nQ)))
        return RAD_E_SHAPE;
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
    A.w = w ? w->data : nullptr;
    A.w_rs = (w && !w1) ? w->stride[0] : 0;
    A.w_bf16 = (w && w->dtype == RAD_BF16) ? 1 : 0;
    A.y = (uint16_t*)y->data; A.y_ps = y2 ? 0 : y->stride[0]; A.y_rs = y2 ? y->stride[0] : y->stride[1];
    A.M = (int)M; A.n = (int)n; A.nS = (int)nS; A.nQ = (int)nQ;
    A.eps = eps;

    const bool chain = rad_args_geti_or(a, "chain", 0) != 0;
    hipStream_t hs = (hipStream_t)stream;
    const bool vec8_ok = !add && nQ == 1 && (nS + 1) <= kFastK && (n % 8) == 0 &&
                         n <= kVec8Elem * 8 * kThr &&
                         rad_ptr_aligned(cur->data, 16) && rad_ptr_aligned(y->data, 16) &&
                         rad_ptr_aligned(q->data, 16) && (cur->stride[0] % 8) == 0 &&
                         (A.y_rs % 8) == 0 &&
                         (nS == 0 || (rad_ptr_aligned(st->data, 16) &&
                                      (st->stride[0] % 8) == 0 && (st->stride[1] % 8) == 0)) &&
                         (!w || rad_ptr_aligned(w->data, 16));
    if (vec8_ok) {
        if (chain) R4D_KLAUNCH(R4D_KNAME(attn_res_vec8_kernel<r4dchain>), dim3((unsigned)M), dim3(kThr), hs, A);
        else       R4D_KLAUNCH(R4D_KNAME(attn_res_vec8_kernel<R4dUnchained>), dim3((unsigned)M), dim3(kThr), hs, A);
    } else if (nS + 1 <= kFastK && n <= kMaxElem * kThr) {
        if (chain) R4D_KLAUNCH(R4D_KNAME(attn_res_fast_kernel<r4dchain>), dim3((unsigned)M), dim3(kThr), hs, A);
        else       R4D_KLAUNCH(R4D_KNAME(attn_res_fast_kernel<R4dUnchained>), dim3((unsigned)M), dim3(kThr), hs, A);
    } else {
        if (chain) R4D_KLAUNCH(R4D_KNAME(attn_res_kernel<r4dchain>), dim3((unsigned)M), dim3(kThr), hs, A);
        else       R4D_KLAUNCH(R4D_KNAME(attn_res_kernel<R4dUnchained>), dim3((unsigned)M), dim3(kThr), hs, A);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
