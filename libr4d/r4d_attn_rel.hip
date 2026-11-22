/* r4d_attn_rel.hip -- Inkling 2's attention, attn_paged_rel (libref/ref_attn.cpp's
 * ref_attn_paged_rel is the contract): attn_paged with a relative-position bias on each score and
 * one sink logit per query row and head.
 *
 *   bias_j  = r[i,h] . rel_proj[:, d]   at d = pos_i - pos_j, for 0 <= d < extent, else 0
 *   logit_j = scale * q . k_j + bias_j  (the bias is not scaled)
 *   out     = sum_j exp(logit_j) v_j / (exp(sink) + sum_j exp(logit_j))
 *
 * Plain on purpose: one workgroup per (query row, query head) at head_dim 128, one thread a
 * dimension. A tile is 128 keys, one a thread: each thread takes its key's dot with the query
 * from LDS and its bias (d_rel MACs against rel_proj, never a table), thread 0 does the tile's
 * maximum and weights, and every thread accumulates its own dimension of the value rows under the
 * online softmax. The sink is folded in after the last tile. Whatever the group (gqa 1 to 8) the
 * kernel reads the query head's own key head, so it serves every gqa with the same code; the
 * speed work is later. Operands are bf16 or f32 where the caller's dtype says, the cache bf16 or
 * int8 (codes with an f16 scale of K and of V per slot, as libref reads it), and everything
 * accumulates in f32 and rounds once to the output.
 *
 * Nothing here allocates or synchronises beyond the block's own barriers (docs/PLUGIN.md §5).
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "r4d_plugin.h"
#include "r4d_launch.h"
#include "rad_plugin.h"

namespace r4d_rel {

constexpr int kThr = 128;      /* one thread a dimension: head_dim 128 */
constexpr int kHd  = 128;
constexpr int kTile = 128;     /* keys a tile, one a thread */
constexpr int kMaxDRel = 64;

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }
__device__ __forceinline__ uint16_t tobf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);   /* NaN stays NaN */
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

/* IEEE half to float, the int8 cache's scales (abi/rad_builder.h). */
__device__ __forceinline__ float f16_to_f32(uint16_t h) {
    const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 31u;
    uint32_t m = h & 1023u, u;
    if (e == 0) {
        if (m == 0) {
            u = s << 31;
        } else {
            /* Subnormal: shift the leading bit up into the exponent. */
            int32_t ee = 113;
            while (!(m & 1024u)) { m <<= 1; --ee; }
            u = (s << 31) | ((uint32_t)ee << 23) | ((m & 1023u) << 13);
        }
    } else if (e == 31) {
        u = (s << 31) | (0xFFu << 23) | (m << 13);
    } else {
        u = (s << 31) | ((e + 112u) << 23) | (m << 13);
    }
    return __uint_as_float(u);
}

/* A scalar of a bf16 or f32 operand. */
__device__ __forceinline__ float ld_bf_or_f(const void* p, bool f32, long long i) {
    return f32 ? ((const float*)p)[i] : bf(((const uint16_t*)p)[i]);
}

struct RelArgs {
    const void* q;  long long q_rs, q_hs, q_ds;  bool q_f32;
    const uint8_t* kv;  long long sb, sh, so;       /* elements (bf16) or bytes (int8) */
    const int32_t* bt;  long long bt_rs, bt_cs;
    const int32_t* seqused;
    const int32_t* cu;                              /* null: uniform q_len */
    const float* kdes;  const float* vdes;          /* null: 1 */
    const void* r;  long long r_rs, r_hs, r_ds;  bool r_f32;
    const void* rp; long long rp_c, rp_d;        bool rp_f32;
    const void* sk; long long s_rs, s_hs;        bool s_f32;
    void* out;      long long o_rs, o_hs, o_ds;  bool o_f32;
    long long n_seq, q_len, kv_heads, gqa, bs, max_blocks, window, extent, d_rel;
    int causal;
    float scale;
};

/* The score of the query (in LDS) against key `kb`, and the key's K scale for the int8 cache. */
template <bool I8>
__device__ __forceinline__ float key_scale(const uint8_t* kv, long long kb, float kds, int half) {
    if (!I8) return kds;
    const uint16_t h = (uint16_t)kv[kb + 2 * kHd + 2 * half] |
                       ((uint16_t)kv[kb + 2 * kHd + 2 * half + 1] << 8);
    return f16_to_f32(h);
}

template <bool I8>
__device__ __forceinline__ float kv_elem(const uint8_t* kv, long long i) {
    if (I8) return (float)((const int8_t*)kv)[i];
    return bf(((const uint16_t*)kv)[i]);
}

template <bool I8>
__global__ __launch_bounds__(kThr) void rel_kernel(RelArgs A) {
    __shared__ float s_q[kHd];
    __shared__ float s_r[kMaxDRel];
    __shared__ float s_w[kTile];
    __shared__ long long s_kb[kTile];
    __shared__ float s_m, s_l, s_alpha;

    const int tok = blockIdx.x, h = blockIdx.y, t = threadIdx.x;

    /* The owning sequence and the row's index within its query: the same on every thread, so the
     * early return below leaves the whole block. */
    long long s = 0, qlen = A.q_len, i = 0;
    if (A.cu) {
        while (s + 1 < A.n_seq && A.cu[s + 1] <= tok) ++s;
        const long long beg = A.cu[s];
        qlen = A.cu[s + 1] - beg;
        i = tok - beg;
    } else {
        s = tok / A.q_len;
        i = tok - s * A.q_len;
    }
    if (qlen <= 0) return;

    const long long kh  = h / A.gqa;
    const long long ctx = A.seqused[s];
    const long long pos = ctx - qlen + i;
    const float kds = (!I8 && A.kdes) ? A.kdes[s * A.kv_heads + kh] : 1.0f;
    const float vds = (!I8 && A.vdes) ? A.vdes[s * A.kv_heads + kh] : 1.0f;

    s_q[t] = ld_bf_or_f(A.q, A.q_f32, tok * A.q_rs + h * A.q_hs + t * A.q_ds);
    if (t < A.d_rel) s_r[t] = ld_bf_or_f(A.r, A.r_f32, tok * A.r_rs + h * A.r_hs + t * A.r_ds);
    if (t == 0) { s_m = -__builtin_inff(); s_l = 0.0f; }
    __syncthreads();

    /* The keys this row can see (attn_paged's predicate, as a range): causal keys up to pos, a
     * window bounding them behind (and ahead, when non-causal). Every key in the range is checked
     * for its block below. */
    long long lo = 0, hi = ctx < A.max_blocks * A.bs ? ctx : A.max_blocks * A.bs;
    if (A.causal) hi = hi < pos + 1 ? hi : pos + 1;
    if (A.window > 0) {
        lo = pos - A.window + 1 > 0 ? pos - A.window + 1 : 0;
        if (!A.causal) hi = hi < pos + A.window ? hi : pos + A.window;
    }

    float acc = 0.0f;
    for (long long j0 = lo; j0 < hi; j0 += kTile) {
        const long long j = j0 + t;
        float sc = -__builtin_inff();
        s_kb[t] = -1;
        if (j < hi) {
            const long long blk = A.bt[s * A.bt_rs + (j / A.bs) * A.bt_cs];
            if (blk >= 0) {
                const long long kb = blk * A.sb + kh * A.sh + (j % A.bs) * A.so;
                s_kb[t] = kb;
                const float km = key_scale<I8>(A.kv, kb, kds, 0);
                float dot = 0.0f;
                for (int dd = 0; dd < kHd; ++dd)
                    dot += s_q[dd] * (kv_elem<I8>(A.kv, kb + dd) * km);
                /* The bias: the query's r against rel_proj's column at the key's distance. */
                const long long dist = pos - j;
                float bias = 0.0f;
                if (dist >= 0 && dist < A.extent)
                    for (long long c = 0; c < A.d_rel; ++c)
                        bias += s_r[c] * ld_bf_or_f(A.rp, A.rp_f32, c * A.rp_c + dist * A.rp_d);
                sc = dot * A.scale + bias;
            }
        }
        s_w[t] = sc;
        __syncthreads();

        if (t == 0) {
            float tmax = -__builtin_inff();
            for (int k = 0; k < kTile; ++k) tmax = fmaxf(tmax, s_w[k]);
            const float mnew = fmaxf(s_m, tmax);
            if (mnew == -__builtin_inff()) {
                /* Nothing visible yet, nothing in this tile: no weights, no rescale. */
                for (int k = 0; k < kTile; ++k) s_w[k] = 0.0f;
                s_alpha = 0.0f;
            } else {
                float tsum = 0.0f;
                for (int k = 0; k < kTile; ++k) {
                    const float w = s_w[k] == -__builtin_inff() ? 0.0f : expf(s_w[k] - mnew);
                    s_w[k] = w;
                    tsum += w;
                }
                s_alpha = s_m == -__builtin_inff() ? 0.0f : expf(s_m - mnew);
                s_l = s_l * s_alpha + tsum;
                s_m = mnew;
            }
        }
        __syncthreads();

        /* This thread's dimension of the tile's values, under the new maximum. */
        acc *= s_alpha;
        for (int k = 0; k < kTile; ++k) {
            const long long kb = s_kb[k];
            if (kb < 0) continue;
            const float vm = key_scale<I8>(A.kv, kb, vds, 1);
            acc += s_w[k] * (kv_elem<I8>(A.kv, kb + kHd + t) * vm);
        }
        __syncthreads();
    }

    /* No visible key: a zero row, the sink or not. */
    if (s_m == -__builtin_inff()) {
        if (A.o_f32) ((float*)A.out)[tok * A.o_rs + h * A.o_hs + t * A.o_ds] = 0.0f;
        else ((uint16_t*)A.out)[tok * A.o_rs + h * A.o_hs + t * A.o_ds] = 0;
        return;
    }
    /* The sink joins the maximum and the denominator, and carries no value. */
    if (t == 0) {
        const float sv = ld_bf_or_f(A.sk, A.s_f32, tok * A.s_rs + h * A.s_hs);
        const float mnew = fmaxf(s_m, sv);
        s_alpha = expf(s_m - mnew);
        s_l = s_l * s_alpha + expf(sv - mnew);
        s_m = mnew;
    }
    __syncthreads();
    acc *= s_alpha;
    const float v = acc / s_l;
    const long long oi = tok * A.o_rs + h * A.o_hs + t * A.o_ds;
    if (A.o_f32) ((float*)A.out)[oi] = v;
    else         ((uint16_t*)A.out)[oi] = tobf(v);
}

/* Shared host side: the operands' geometry, then one launch. */
static int launch_rel(const RadArgs* a, RadStream stream, bool i8) {
    if (!a || a->n_t < 11) return RAD_E_INVAL;
    const RadTensor *q = rad_arg_in(a, 0), *kv = rad_arg_in(a, 1), *bt = rad_arg_in(a, 2);
    const RadTensor *su = rad_arg_in(a, 3), *kd = rad_arg_in(a, 4), *vd = rad_arg_in(a, 5);
    const RadTensor *cu = rad_arg_in(a, 6), *rr = rad_arg_in(a, 7), *rp = rad_arg_in(a, 8);
    const RadTensor *sk = rad_arg_in(a, 9), *o = rad_arg_in(a, 10);
    if (!q || !kv || !bt || !su || !rr || !rp || !sk || !o) return RAD_E_INVAL;

    const long long hd = rad_args_geti_or(a, "head_dim", 0), gqa = rad_args_geti_or(a, "gqa", 0);
    const long long bs = rad_args_geti_or(a, "block_size", 0), causal = rad_args_geti_or(a, "causal", 1);
    const long long window = rad_args_geti_or(a, "window", 0);
    const long long d_rel = rad_args_geti_or(a, "d_rel", 0), extent = rad_args_geti_or(a, "extent", 0);
    if (hd != kHd || gqa < 1 || bs < 1 || window < 0 || d_rel < 1 || d_rel > kMaxDRel || extent < 1)
        return RAD_E_SHAPE;
    if (q->rank < 2 || kv->rank != 4 || rr->rank < 2 || sk->rank < 2 || rp->rank != 2 || o->rank < 2)
        return RAD_E_SHAPE;

    /* Dtypes: activations bf16 or f32, the cache bf16 (or int8 for the int8 entry), the index
     * operands i32, the descales f32. */
    const bool q_f32 = q->dtype == RAD_F32, o_f32 = o->dtype == RAD_F32;
    if ((q->dtype != RAD_BF16 && !q_f32) || (o->dtype != RAD_BF16 && !o_f32)) return RAD_E_DTYPE;
    if (kv->dtype != (i8 ? RAD_I8 : RAD_BF16)) return RAD_E_DTYPE;
    if (bt->dtype != RAD_I32 || su->dtype != RAD_I32 || (cu && cu->dtype != RAD_I32)) return RAD_E_DTYPE;
    if ((kd && kd->dtype != RAD_F32) || (vd && vd->dtype != RAD_F32)) return RAD_E_DTYPE;
    const bool r_f32 = rr->dtype == RAD_F32, rp_f32 = rp->dtype == RAD_F32, s_f32 = sk->dtype == RAD_F32;
    if ((rr->dtype != RAD_BF16 && !r_f32) || (rp->dtype != RAD_BF16 && !rp_f32) ||
        (sk->dtype != RAD_BF16 && !s_f32)) return RAD_E_DTYPE;

    /* Contiguous within a row except the row pitch, and the last axis dense: the cache's K and V
     * are adjacent (stride 1 along head_dim). */
    if (q->stride[q->rank - 1] < 1 || o->stride[o->rank - 1] != 1 || rr->stride[rr->rank - 1] != 1 ||
        sk->stride[sk->rank - 1] < 1 || rp->stride[1] != 1 || kv->stride[3] != 1)
        return RAD_E_STRIDE;
    if (i8 && kv->shape[3] != 2 * kHd + 4) return RAD_E_SHAPE;
    if (!i8 && kv->shape[3] < 2 * kHd) return RAD_E_SHAPE;

    const long long kv_heads = kv->shape[1];
    const long long nh = gqa * kv_heads;
    /* A rank-2 q or out is [rows, nh * head_dim]; only a rank-3 one names its head axis. */
    if (kv->shape[2] != bs || (q->rank >= 3 && q->shape[q->rank - 2] != nh) ||
        (o->rank >= 3 && o->shape[o->rank - 2] != nh) || rp->shape[0] != d_rel ||
        rp->shape[1] != extent) return RAD_E_SHAPE;
    const long long n_seq = rad_tensor_numel(su);
    const long long rows  = rad_tensor_numel(q) / (nh * kHd);
    if (n_seq < 1 || rows < 1) return RAD_E_SHAPE;
    if (!cu && rows % n_seq) return RAD_E_SHAPE;
    if (cu && rad_tensor_numel(cu) != n_seq + 1) return RAD_E_SHAPE;
    const long long q_len = cu ? 0 : rows / n_seq;
    if (rad_tensor_numel(o) < rows * nh * kHd) return RAD_E_SHAPE;
    if (rad_tensor_numel(rr) < rows * nh * d_rel) return RAD_E_SHAPE;
    if (rad_tensor_numel(sk) < rows * nh) return RAD_E_SHAPE;

    RelArgs A{};
    A.q = q->data; A.q_f32 = q_f32;
    A.q_rs = q->stride[0];
    A.q_hs = q->rank >= 3 ? q->stride[q->rank - 2] : kHd * q->stride[q->rank - 1];
    A.q_ds = q->stride[q->rank - 1];
    A.kv = (const uint8_t*)kv->data; A.sb = kv->stride[0]; A.sh = kv->stride[1]; A.so = kv->stride[2];
    A.bt = (const int32_t*)bt->data;
    A.bt_rs = bt->rank >= 2 ? bt->stride[bt->rank - 2] : 0;
    A.bt_cs = bt->rank >= 2 ? bt->stride[bt->rank - 1] : 1;
    A.max_blocks = bt->rank >= 2 ? bt->shape[bt->rank - 1] : rad_tensor_numel(bt) / n_seq;
    A.seqused = (const int32_t*)su->data;
    A.cu = cu ? (const int32_t*)cu->data : nullptr;
    A.kdes = kd ? (const float*)kd->data : nullptr;
    A.vdes = vd ? (const float*)vd->data : nullptr;
    A.r = rr->data; A.r_f32 = r_f32;
    A.r_rs = rr->stride[0];
    A.r_hs = rr->rank >= 3 ? rr->stride[rr->rank - 2] : d_rel * rr->stride[rr->rank - 1];
    A.r_ds = rr->stride[rr->rank - 1];
    A.rp = rp->data; A.rp_f32 = rp_f32; A.rp_c = rp->stride[0]; A.rp_d = rp->stride[1];
    A.sk = sk->data; A.s_f32 = s_f32;
    A.s_rs = sk->stride[0];
    A.s_hs = sk->stride[sk->rank - 1];
    A.out = o->data; A.o_f32 = o_f32;
    A.o_rs = o->stride[0];
    A.o_hs = o->rank >= 3 ? o->stride[o->rank - 2] : kHd * o->stride[o->rank - 1];
    A.o_ds = o->stride[o->rank - 1];
    A.n_seq = n_seq; A.q_len = q_len; A.kv_heads = kv_heads; A.gqa = gqa; A.bs = bs;
    A.window = window; A.extent = extent; A.d_rel = d_rel; A.causal = (int)causal;
    A.scale = (float)rad_args_getf_or(a, "scale", 1.0 / __builtin_sqrt((double)kHd));

    if (i8) R4D_KLAUNCH(R4D_KNAME(rel_kernel<true>), dim3((unsigned)rows, (unsigned)nh), dim3(kThr),
                       (hipStream_t)stream, A);
    else    R4D_KLAUNCH(R4D_KNAME(rel_kernel<false>), dim3((unsigned)rows, (unsigned)nh), dim3(kThr),
                       (hipStream_t)stream, A);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* namespace r4d_rel */

extern "C" {

int r4d_rad_attn_rel_h128_bf16kv(const RadArgs* a, RadStream stream) {
    return r4d_rel::launch_rel(a, stream, false);
}

int r4d_rad_attn_rel_h128_i8kv(const RadArgs* a, RadStream stream) {
    return r4d_rel::launch_rel(a, stream, true);
}

}  /* extern "C" */
