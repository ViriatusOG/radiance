/* r4d_attn_rel.hip -- Inkling 2's attention, attn_paged_rel (libref/ref_attn.cpp's
 * ref_attn_paged_rel is the contract): attn_paged with a relative-position bias on each score and
 * one sink logit per query row and head.
 *
 *   bias_j  = r[i,h] . rel_proj[:, d]   at d = pos_i - pos_j, for 0 <= d < extent, else 0
 *   logit_j = scale * q . k_j + bias_j  (the bias is not scaled)
 *   out     = sum_j exp(logit_j) v_j / (exp(sink) + sum_j exp(logit_j))
 *
 * Four kernels serve the shape across every regime:
 *   1. GQA-shared split-KV decode/verify (`rel_dec_kernel` + `rel_merge_kernel`, 128 threads,
 *      BK=32): one (tok, kh, split) block covers every gqa<=8 query head, so K and V load once.
 *      Splits follow the live key bound, at least eight tiles each, up to the scratch budget.
 *      The merge walks those partials 32 a wave.
 *   1b. Packed verify (`rel_vfy_kernel` + the same merge): a verify of two or more rows a sequence
 *      makes each (row, group head) a WMMA column of one (sequence, kh, split) block, so the rows
 *      share every K and V tile instead of each walking the cache on its own.
 *   2. WMMA prefill (`rel_wmma_kernel`, 256 threads / 8 waves, BQ=128): stages 16-key paged tiles
 *      and touched rel_proj columns into bank-conflict-free LDS and executes S^T = K*Q^T,
 *      relative bias, and O^T = V^T*P^T via RDNA4 16x16x16 WMMA.
 *   3. Generic fallback (`rel_kernel`): serves f32 operands, gqa > 8, bs != 16, or unaligned views.
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "r4d_chain.h"
#include "r4d_common.h"
#include "r4d_launch.h"
#include "r4d_plugin.h"
#include "rad_plugin.h"

namespace r4d_rel {

constexpr int kThr          = 128;   /* generic fallback & decode/merge threads per block */
constexpr int kHd           = 128;
constexpr int kTile         = 128;   /* generic fallback key tile */
constexpr int kMaxDRel      = 64;
constexpr int kMaxGqa       = 8;

constexpr int kDecBK        = 32;    /* two 16-key paged blocks per tile in decode kernel */
constexpr int kRelTilesMin  = 8;     /* keys a split keeps, in tiles: rad_prefill_splits_law's floor */
constexpr int kRelMaxSplit  = 128;   /* merge width; scratch holds this at four (tok, kv-head) pairs */
constexpr int kRelPartBudget = 512;  /* max (tok, kv_head, split) partial slots in scratch */
constexpr int kRelBlocksPerWgp = 16; /* 128-thread blocks wanted on a WGP before the scratch cuts it */

constexpr int kWThreads     = 256;   /* WMMA prefill: 8 waves x 16 qrows = 128 qrows per block */
constexpr int kWBQ          = 128;
constexpr int kWKS          = kHd * 2 + 16;   /* 272 bytes per key row of K tile (bank-free) */
constexpr int kWVS          = 48;             /* 48 bytes per dim row of V^T tile: 16 f16 + pad */

typedef short    rel_v8s __attribute__((ext_vector_type(8)));
typedef float    rel_v8f __attribute__((ext_vector_type(8)));
typedef _Float16 rel_v8h __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float((uint32_t)v << 16); }
__device__ __forceinline__ uint16_t tobf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);   /* NaN stays NaN */
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
__device__ __forceinline__ unsigned pack2(float a, float b) {
    return (unsigned)tobf(a) | ((unsigned)tobf(b) << 16);
}

__device__ __forceinline__ float f16_to_f32(uint16_t h) {
    const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 31u;
    uint32_t m = h & 1023u, u;
    if (e == 0) {
        if (m == 0) {
            u = s << 31;
        } else {
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

__device__ __forceinline__ float ld_bf_or_f(const void* p, bool f32, long long i) {
    return f32 ? ((const float*)p)[i] : bf(((const uint16_t*)p)[i]);
}

__device__ __forceinline__ uint32_t ld_u32(const void* p) {
#ifdef __HIP__
    return *(const uint32_t*)p;
#else
    uint32_t v;
    __builtin_memcpy(&v, p, sizeof v);
    return v;
#endif
}

__device__ __forceinline__ uint2 ld_u2(const void* p) {
#ifdef __HIP__
    return *(const uint2*)p;
#else
    uint2 v;
    __builtin_memcpy(&v, p, sizeof v);
    return v;
#endif
}

__device__ __forceinline__ uint4 ld_u4(const void* p) {
#ifdef __HIP__
    return *(const uint4*)p;
#else
    uint4 v;
    __builtin_memcpy(&v, p, sizeof v);
    return v;
#endif
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
    float* part;
    long long n_seq, q_len, kv_heads, gqa, bs, max_blocks, window, extent, d_rel;
    int causal, n_split, tps;
    float scale;
};

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

/* ---- Generic fallback kernel (arbitrary strides, dtypes, gqa > 8, bs != 16) ------------------ */

template <bool I8>
__global__ __launch_bounds__(kThr) void rel_kernel(RelArgs A) {
    __shared__ float s_q[kHd];
    __shared__ float s_r[kMaxDRel];
    __shared__ float s_w[kTile];
    __shared__ long long s_kb[kTile];
    __shared__ float s_m, s_l, s_alpha;

    const int tok = blockIdx.x, h = blockIdx.y, t = threadIdx.x;

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

        acc *= s_alpha;
        for (int k = 0; k < kTile; ++k) {
            const long long kb = s_kb[k];
            if (kb < 0) continue;
            const float vm = key_scale<I8>(A.kv, kb, vds, 1);
            acc += s_w[k] * (kv_elem<I8>(A.kv, kb + kHd + t) * vm);
        }
        __syncthreads();
    }

    if (s_m == -__builtin_inff()) {
        if (A.o_f32) ((float*)A.out)[tok * A.o_rs + h * A.o_hs + t * A.o_ds] = 0.0f;
        else ((uint16_t*)A.out)[tok * A.o_rs + h * A.o_hs + t * A.o_ds] = 0;
        return;
    }
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

/* ---- Fast Path A: GQA-shared Split-KV Decode/Verify (128 threads, BK=32) --------------------- */

/* Four bytes of an 8-byte lane word. `high` selects the upper half. All lanes call it. */
__device__ __forceinline__ uint32_t rel_shfl_half(uint32_t lo, uint32_t hi, int src, int high) {
    const uint32_t x = __shfl(lo, src, 32);
    const uint32_t y = __shfl(hi, src, 32);
    return high ? y : x;
}

/* Keep these loads issued and finished before the score, without a later wait catching the next tile.
 * The fence pins VGPRs, so it exists in the device compile only; the host emulation has none. */
__device__ __forceinline__ void rel_keep8(uint2 raw[8], uint32_t sw[8]) {
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(raw[0].x), "+v"(raw[0].y), "+v"(raw[1].x), "+v"(raw[1].y),
                       "+v"(raw[2].x), "+v"(raw[2].y), "+v"(raw[3].x), "+v"(raw[3].y),
                       "+v"(raw[4].x), "+v"(raw[4].y), "+v"(raw[5].x), "+v"(raw[5].y),
                       "+v"(raw[6].x), "+v"(raw[6].y), "+v"(raw[7].x), "+v"(raw[7].y),
                       "+v"(sw[0]), "+v"(sw[1]), "+v"(sw[2]), "+v"(sw[3]),
                       "+v"(sw[4]), "+v"(sw[5]), "+v"(sw[6]), "+v"(sw[7]) ::);
#endif
}

template <bool I8>
__device__ __forceinline__ void rel_load8(const uint8_t* kv, long long base, long long jw, long long so,
                                          int lane, uint2 raw[8], uint32_t sw[8], long long kb[8]) {
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        kb[q] = base + ((jw + q) & 15) * so;
        if constexpr (I8) {
            raw[q] = ld_u2(kv + kb[q] + lane * 8);
            sw[q] = ld_u32(kv + kb[q] + 2 * kHd);
        } else {
            raw[q] = ld_u2(((const uint16_t*)kv) + kb[q] + lane * 4);
            sw[q] = 0;
        }
    }
}

/* f16 bits to f32. The conversion is exact, so it is f16_to_f32 on every value but a NaN's payload. */
__device__ __forceinline__ float rel_f16_bits(uint32_t h) {
    return (float)__builtin_bit_cast(_Float16, (uint16_t)h);
}

/* Score this wave's eight keys of a full tile, K already in registers and no key biased. Each lane
 * forms every key's four-element partial for one head, then one butterfly sums all eight: the
 * exchanges and adds are r4d_xor_sum<32>'s per key (offsets 16, 8, 4, 2, 1), so each score is
 * rel_dec_score_loaded's bit for bit. At offsets 16, 8 and 4 a lane keeps the half of the keys its
 * partner drops, so lane l ends with key l >> 2. Scoring one key at a time would cost a reduction,
 * a q reload and a masked store per key and head, and that issue rate bounds the decode. */
template <bool I8>
__device__ __forceinline__ void rel_dec_score8(int lane, int wave, int gqa, float qscale, float vds,
                                               long long base, long long jw, long long so,
                                               float s_q[][kHd], float s_sc[][kDecBK],
                                               long long* s_kb, float* s_vs, const uint2 raw[8],
                                               const uint32_t sw[8]) {
    float kf[8][4], ks[8];
    uint32_t my_sw = sw[0];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        if constexpr (I8) {
            const uint32_t w = rel_shfl_half(raw[q].x, raw[q].y, lane >> 1, lane & 1);
            kf[q][0] = (float)(int8_t)(w & 0xffu);
            kf[q][1] = (float)(int8_t)((w >> 8) & 0xffu);
            kf[q][2] = (float)(int8_t)((w >> 16) & 0xffu);
            kf[q][3] = (float)(int8_t)(w >> 24);
            ks[q] = rel_f16_bits(sw[q] & 0xffffu) * qscale;
        } else {
            kf[q][0] = bf((uint16_t)(raw[q].x & 0xffffu));
            kf[q][1] = bf((uint16_t)(raw[q].x >> 16));
            kf[q][2] = bf((uint16_t)(raw[q].y & 0xffffu));
            kf[q][3] = bf((uint16_t)(raw[q].y >> 16));
            ks[q] = qscale;
        }
        my_sw = lane == q ? sw[q] : my_sw;
    }
    const bool h16 = lane & 16, h8 = lane & 8, h4 = lane & 4;
    const int d0 = lane * 4;
    for (int g = 0; g < gqa; ++g) {
        const float q0 = s_q[g][d0 + 0], q1 = s_q[g][d0 + 1], q2 = s_q[g][d0 + 2], q3 = s_q[g][d0 + 3];
        float a[8], b[4], c[2];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const float dot = q0 * kf[q][0] + q1 * kf[q][1] + q2 * kf[q][2] + q3 * kf[q][3];
            a[q] = dot * ks[q];
            a[q] += r4d_xor_lane<16>(a[q]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) b[j] = r4d_dpp_xor_add<0x168>(r4d_pick(h16, a[4 + j], a[j]));
#pragma unroll
        for (int j = 0; j < 2; ++j) c[j] = r4d_dpp_xor_add<0x164>(r4d_pick(h8, b[2 + j], b[j]));
        float d = r4d_pick(h4, c[1], c[0]);
        d = r4d_dpp_xor_add<0x162>(d);
        d = r4d_dpp_xor_add<0x161>(d);
        if ((lane & 3) == 0) s_sc[g][wave * 8 + (lane >> 2)] = d;
    }
    if (lane < 8) {
        s_kb[wave * 8 + lane] = base + ((jw + lane) & 15) * so;
        s_vs[wave * 8 + lane] = I8 ? rel_f16_bits(my_sw >> 16) * vds : vds;
    }
}

/* Score a key whose K word is already in registers. A hole (have == false) writes an empty partial.
 * noinline: the caller's loads must not sink into these reductions, or each key waits on its own
 * HBM round trip. */
template <bool I8>
__device__ __noinline__ void rel_dec_score_loaded(
    int lane, int gqa, int k, long long kb, long long j, long long pos, int d_rel, int extent,
    long long rp_c, const uint16_t* rp16, float qscale, float vds,
    float s_q[][kHd], float s_r[][kMaxDRel], float s_sc[][kDecBK], long long* s_kb, float* s_vs,
    uint32_t w0, uint32_t w1, uint32_t scale_w, bool have) {
    if (!have) {
        if (lane == 0) {
            for (int g = 0; g < gqa; ++g) s_sc[g][k] = -__builtin_inff();
            s_kb[k] = -1;
            s_vs[k] = 0.0f;
        }
        return;
    }
    float kf[4];
    if constexpr (I8) {
        kf[0] = (float)(int8_t)(w0 & 0xffu);
        kf[1] = (float)(int8_t)((w0 >> 8) & 0xffu);
        kf[2] = (float)(int8_t)((w0 >> 16) & 0xffu);
        kf[3] = (float)(int8_t)(w0 >> 24);
    } else {
        kf[0] = bf((uint16_t)(w0 & 0xffffu));
        kf[1] = bf((uint16_t)(w0 >> 16));
        kf[2] = bf((uint16_t)(w1 & 0xffffu));
        kf[3] = bf((uint16_t)(w1 >> 16));
    }
    /* scale_w is the two f16 scales, already loaded. A load here would wait out the caller's burst. */
    const float ks = (I8 ? f16_to_f32((uint16_t)(scale_w & 0xffffu)) : 1.0f) * qscale;
    const float vs = I8 ? f16_to_f32((uint16_t)(scale_w >> 16)) * vds : vds;
    const long long dist = pos - j;
    const bool has_bias = dist >= 0 && dist < extent;
    const float rp0 = (has_bias && lane < d_rel) ? bf(rp16[(long long)lane * rp_c + dist]) : 0.0f;
    const float rp1 = (has_bias && d_rel > 32 && lane + 32 < d_rel)
                          ? bf(rp16[(long long)(lane + 32) * rp_c + dist])
                          : 0.0f;
    for (int g = 0; g < gqa; ++g) {
        const int d0 = lane * 4;
        const float dot = s_q[g][d0 + 0] * kf[0] + s_q[g][d0 + 1] * kf[1] +
                          s_q[g][d0 + 2] * kf[2] + s_q[g][d0 + 3] * kf[3];
        const float pb = (lane < d_rel ? s_r[g][lane] * rp0 : 0.0f) +
                         (d_rel > 32 && lane + 32 < d_rel ? s_r[g][lane + 32] * rp1 : 0.0f);
        const float sc = r4d_xor_sum<32>(dot * ks + pb);
        if (lane == 0) s_sc[g][k] = sc;
    }
    if (lane == 0) {
        s_kb[k] = kb;
        s_vs[k] = vs;
    }
}

template <bool I8>
__device__ __forceinline__ void rel_dec_body(RelArgs A) {
    __shared__ float     s_q[kMaxGqa][kHd];
    __shared__ float     s_r[kMaxGqa][kMaxDRel];
    __shared__ float     s_sc[kMaxGqa][kDecBK];
    __shared__ float     s_m[kMaxGqa];
    __shared__ float     s_l[kMaxGqa];
    __shared__ float     s_alpha[kMaxGqa];
    __shared__ long long s_kb[kDecBK];
    __shared__ float     s_vs[kDecBK];
    __shared__ float     s_part[4][kHd];
    __shared__ int       s_hole;

    const int tok = (int)blockIdx.x, kh = (int)blockIdx.y, sp = (int)blockIdx.z;
    const int t = (int)threadIdx.x, lane = t & 31, wave = t >> 5;
    const int gqa = (int)A.gqa;

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

    const long long ctx = A.seqused[s];
    const long long pos = ctx - qlen + i;
    const float kds = (!I8 && A.kdes) ? A.kdes[s * A.kv_heads + kh] : 1.0f;
    const float vds = (!I8 && A.vdes) ? A.vdes[s * A.kv_heads + kh] : 1.0f;
    const float qscale = A.scale * kds;

    const uint16_t* q_tok = (const uint16_t*)A.q + (long long)tok * A.q_rs +
                            (long long)kh * gqa * A.q_hs;
    const uint16_t* r_tok = (const uint16_t*)A.r + (long long)tok * A.r_rs +
                            (long long)kh * gqa * A.r_hs;
    for (int g = 0; g < gqa; ++g) {
        s_q[g][t] = bf(q_tok[g * A.q_hs + t]);
        if (t < A.d_rel) s_r[g][t] = bf(r_tok[g * A.r_hs + t]);
    }
    if (t < gqa) {
        s_m[t] = -__builtin_inff();
        s_l[t] = 0.0f;
    }
    if (t == 0) s_hole = 0;
    lds_barrier();

    long long lo = 0, hi = ctx < A.max_blocks * 16 ? ctx : A.max_blocks * 16;
    if (A.causal) hi = hi < pos + 1 ? hi : pos + 1;
    if (A.window > 0) {
        lo = pos - A.window + 1 > 0 ? pos - A.window + 1 : 0;
        if (!A.causal) hi = hi < pos + A.window ? hi : pos + A.window;
    }

    const long long nkt = hi > lo ? (hi - lo + kDecBK - 1) / kDecBK : 0;
    float acc[kMaxGqa];
    for (int g = 0; g < kMaxGqa; ++g) acc[g] = 0.0f;

    const uint16_t* rp16 = (const uint16_t*)A.rp;
    uint2 pend_raw[8] = {};
    uint32_t pend_sw[8] = {};
    long long pend_kb[8] = {};
    long long pend_j = -1;

    for (long long kt = sp; kt < nkt; kt += A.n_split) {
        const long long j0 = lo + kt * kDecBK;

        /* A full 32-key tile is page-aligned, so each wave's eight keys are one page. A wave
         * pulls 8 bytes a lane: int8 is the key and the value together (bytes 0-255), bf16 is the
         * key. A holed or partial tile keeps the one-key loop. */
        const bool tile_full = ((j0 & 31) == 0) && (j0 + kDecBK <= hi);
        const long long jw = j0 + (long long)wave * 8;
        const long long blk = (tile_full && jw < hi) ? A.bt[s * A.bt_rs + (jw >> 4)] : -1;
        uint2 raw[8];
        uint32_t sw[8];
        long long kb8[8];
        if (tile_full && blk >= 0) {
            const long long base = blk * A.sb + (long long)kh * A.sh;
            if (pend_j == j0) {
                rel_keep8(pend_raw, pend_sw);
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    raw[q] = pend_raw[q];
                    sw[q] = pend_sw[q];
                    kb8[q] = pend_kb[q];
                }
            } else {
                rel_load8<I8>(A.kv, base, jw, A.so, lane, raw, sw, kb8);
                rel_keep8(raw, sw);
            }
            pend_j = -1;
            {
                const long long jn = j0 + A.n_split * kDecBK;
                if ((jn & 31) == 0 && jn + kDecBK <= hi && kt + A.n_split < nkt) {
                    const long long jnw = jn + (long long)wave * 8;
                    const long long nblk = jnw < hi ? A.bt[s * A.bt_rs + (jnw >> 4)] : -1;
                    if (nblk >= 0) {
                        rel_load8<I8>(A.kv, nblk * A.sb + (long long)kh * A.sh, jnw, A.so, lane,
                                      pend_raw, pend_sw, pend_kb);
                        pend_j = jn;
                    }
                }
            }
            /* A wave with a key inside the bias extent scores one key at a time. */
            const bool biased = jw <= pos && jw + 7 > pos - A.extent;
            if (!biased) {
                rel_dec_score8<I8>(lane, wave, gqa, qscale, vds, base, jw, A.so, s_q, s_sc, s_kb,
                                   s_vs, raw, sw);
            } else {
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    uint32_t w0, w1;
                    if constexpr (I8) {
                        w0 = rel_shfl_half(raw[q].x, raw[q].y, lane >> 1, lane & 1);
                        w1 = 0;
                    } else {
                        w0 = raw[q].x;
                        w1 = raw[q].y;
                    }
                    rel_dec_score_loaded<I8>(lane, gqa, wave * 8 + q, kb8[q], jw + q, pos,
                                             (int)A.d_rel, (int)A.extent, A.rp_c, rp16, qscale,
                                             vds, s_q, s_r, s_sc, s_kb, s_vs, w0, w1, sw[q], true);
                }
            }
        } else {
            pend_j = -1;
            if (tile_full && lane == 0) s_hole = 1;
            for (int q = 0; q < 8; ++q) {
                const int k = wave * 8 + q;
                const long long j = j0 + k;
                long long kb = -1;
                if (j < hi) {
                    const long long pb = A.bt[s * A.bt_rs + (j >> 4)];
                    if (pb >= 0) kb = pb * A.sb + (long long)kh * A.sh + (j & 15) * A.so;
                }
                uint32_t w0 = 0, w1 = 0, sw = 0;
                if (kb >= 0) {
                    if constexpr (I8) {
                        w0 = ld_u32(A.kv + kb + lane * 4);
                        sw = ld_u32(A.kv + kb + 2 * kHd);
                    } else {
                        const uint2 w = ld_u2(((const uint16_t*)A.kv) + kb + lane * 4);
                        w0 = w.x;
                        w1 = w.y;
                    }
                }
                rel_dec_score_loaded<I8>(lane, gqa, k, kb, j, pos, (int)A.d_rel, (int)A.extent,
                                         A.rp_c, rp16, qscale, vds, s_q, s_r, s_sc, s_kb,
                                         s_vs, w0, w1, sw, kb >= 0);
            }
        }
        lds_barrier();

        /* Stage 2: Parallel 32-key online softmax across waves. */
        for (int g = wave; g < gqa; g += 4) {
            const float sc = s_sc[g][lane];
            const float tmax = r4d_wave_max(sc);
            const float m_prev = s_m[g];
            const float mnew = fmaxf(m_prev, tmax);
            const float w = (mnew == -__builtin_inff() || sc == -__builtin_inff())
                                ? 0.0f
                                : expf(sc - mnew);
            const float tsum = r4d_wave_sum(w);
            s_sc[g][lane] = w;
            if (lane == 0) {
                const float alpha = (m_prev == -__builtin_inff() || mnew == -__builtin_inff())
                                        ? 0.0f
                                        : expf(m_prev - mnew);
                s_alpha[g] = alpha;
                s_l[g] = s_l[g] * alpha + tsum;
                s_m[g] = mnew;
            }
        }
        lds_barrier();

        /* Stage 3. Previous partials take the new max. A full tile's value was loaded 8 bytes
         * a lane; each lane holds four elements of its wave's keys and hands them to the thread
         * that owns the element. */
        for (int g = 0; g < gqa; ++g) acc[g] *= s_alpha[g];
        if (tile_full && s_hole == 0) {
            uint2 vraw[8];
            if constexpr (!I8) {
#pragma unroll
                for (int q = 0; q < 8; ++q)
                    vraw[q] = ld_u2(((const uint16_t*)A.kv) + s_kb[wave * 8 + q] + kHd + lane * 4);
#if defined(__HIP_DEVICE_COMPILE__)
                asm volatile("" : "+v"(vraw[0].x), "+v"(vraw[0].y), "+v"(vraw[1].x), "+v"(vraw[1].y),
                                   "+v"(vraw[2].x), "+v"(vraw[2].y), "+v"(vraw[3].x), "+v"(vraw[3].y),
                                   "+v"(vraw[4].x), "+v"(vraw[4].y), "+v"(vraw[5].x), "+v"(vraw[5].y),
                                   "+v"(vraw[6].x), "+v"(vraw[6].y), "+v"(vraw[7].x), "+v"(vraw[7].y) ::);
#endif
            }
            /* The scaled values do not depend on the head: decode them once a tile. */
            float vf[8][4];
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const float vs = s_vs[wave * 8 + q];
                if constexpr (I8) {
                    const uint32_t vw = rel_shfl_half(raw[q].x, raw[q].y, 16 + (lane >> 1), lane & 1);
                    vf[q][0] = (float)(int8_t)(vw & 0xffu) * vs;
                    vf[q][1] = (float)(int8_t)((vw >> 8) & 0xffu) * vs;
                    vf[q][2] = (float)(int8_t)((vw >> 16) & 0xffu) * vs;
                    vf[q][3] = (float)(int8_t)(vw >> 24) * vs;
                } else {
                    vf[q][0] = bf((uint16_t)(vraw[q].x & 0xffffu)) * vs;
                    vf[q][1] = bf((uint16_t)(vraw[q].x >> 16)) * vs;
                    vf[q][2] = bf((uint16_t)(vraw[q].y & 0xffffu)) * vs;
                    vf[q][3] = bf((uint16_t)(vraw[q].y >> 16)) * vs;
                }
            }
            for (int g = 0; g < gqa; ++g) {
                float tv0 = 0.f, tv1 = 0.f, tv2 = 0.f, tv3 = 0.f;
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const float w = s_sc[g][wave * 8 + q];
                    tv0 = fmaf(w, vf[q][0], tv0);
                    tv1 = fmaf(w, vf[q][1], tv1);
                    tv2 = fmaf(w, vf[q][2], tv2);
                    tv3 = fmaf(w, vf[q][3], tv3);
                }
                s_part[wave][lane * 4 + 0] = tv0;
                s_part[wave][lane * 4 + 1] = tv1;
                s_part[wave][lane * 4 + 2] = tv2;
                s_part[wave][lane * 4 + 3] = tv3;
                lds_barrier();
                acc[g] += s_part[0][t] + s_part[1][t] + s_part[2][t] + s_part[3][t];
                lds_barrier();
            }
        } else {
            for (int k = 0; k < kDecBK; ++k) {
                const long long kb = s_kb[k];
                if (kb < 0) continue;
                const float v = kv_elem<I8>(A.kv, kb + kHd + t) * s_vs[k];
                for (int g = 0; g < gqa; ++g) acc[g] = fmaf(s_sc[g][k], v, acc[g]);
            }
        }
        lds_barrier();
        if (t == 0) s_hole = 0;
        lds_barrier();
    }

    if (A.n_split == 1) {
        uint16_t* o_tok = (uint16_t*)A.out + (long long)tok * A.o_rs +
                          (long long)kh * gqa * A.o_hs;
        for (int g = 0; g < gqa; ++g) {
            const float ms = s_m[g];
            uint16_t out_b = 0;
            if (ms != -__builtin_inff()) {
                const float sv = ld_bf_or_f(A.sk, A.s_f32,
                                            (long long)tok * A.s_rs + (kh * gqa + g) * A.s_hs);
                const float mnew = fmaxf(ms, sv);
                const float alpha = expf(ms - mnew);
                const float den = s_l[g] * alpha + expf(sv - mnew);
                out_b = tobf((acc[g] * alpha) / den);
            }
            o_tok[g * A.o_hs + t] = out_b;
        }
    } else {
        for (int g = 0; g < gqa; ++g) {
            const long long slot = (((long long)tok * A.kv_heads + kh) * A.n_split + sp) * gqa + g;
            float* dst = A.part + slot * (kHd + 4);
            dst[t] = acc[g];
            if (t == 0) {
                dst[kHd]     = s_m[g];
                dst[kHd + 1] = s_l[g];
            }
        }
    }
}

template <bool I8, class CH>
__global__ __launch_bounds__(kThr) void rel_dec_kernel(RelArgs A) {
    if (CH::on) r4d_chain_wait();
    rel_dec_body<I8>(A);
    if (CH::on) r4d_chain_done();
}

__device__ __forceinline__ void rel_merge_body(RelArgs A) {
    __shared__ float s_mt, s_den;
    __shared__ float s_w[kRelMaxSplit];
    const int tok = (int)blockIdx.x, kh = (int)blockIdx.y;
    const int t = (int)threadIdx.x, lane = t & 31, wave = t >> 5;
    const int gqa = (int)A.gqa, ns = A.n_split;
    uint16_t* o_tok = (uint16_t*)A.out + (long long)tok * A.o_rs + (long long)kh * gqa * A.o_hs;

    for (int g = 0; g < gqa; ++g) {
        if (wave == 0) {
            /* One wave holds 32 partials. A long context launches more, so the max and the
             * weighted sum walk the splits in waves. The sink is added once, not once a wave. */
            float mt = -__builtin_inff();
            for (int base = 0; base < ns; base += 32) {
                float ms = -__builtin_inff();
                const int i = base + lane;
                if (i < ns) {
                    const long long slot = (((long long)tok * A.kv_heads + kh) * ns + i) * gqa + g;
                    ms = A.part[slot * (kHd + 4) + kHd];
                }
                mt = fmaxf(mt, r4d_wave_max(ms));
            }
            if (mt == -__builtin_inff()) {
                if (lane == 0) { s_mt = mt; s_den = 1.0f; }
                for (int base = 0; base < ns; base += 32)
                    if (base + lane < ns) s_w[base + lane] = 0.0f;
            } else {
                const float sv = ld_bf_or_f(A.sk, A.s_f32,
                                            (long long)tok * A.s_rs + (kh * gqa + g) * A.s_hs);
                const float mnew = fmaxf(mt, sv);
                float ltot = expf(sv - mnew);
                for (int base = 0; base < ns; base += 32) {
                    float ms = -__builtin_inff(), ls = 0.0f;
                    const int i = base + lane;
                    if (i < ns) {
                        const long long slot = (((long long)tok * A.kv_heads + kh) * ns + i) * gqa + g;
                        const float* p = A.part + slot * (kHd + 4);
                        ms = p[kHd];
                        ls = p[kHd + 1];
                    }
                    const float wsp = (ms == -__builtin_inff()) ? 0.0f : expf(ms - mnew);
                    if (i < ns) s_w[i] = wsp;
                    ltot += r4d_wave_sum(ls * wsp);
                }
                if (lane == 0) { s_mt = mnew; s_den = ltot; }
            }
        }
        lds_barrier();

        uint16_t out_b = 0;
        if (s_mt != -__builtin_inff()) {
            float sum = 0.0f;
            const long long base = (((long long)tok * A.kv_heads + kh) * ns) * gqa + g;
            for (int sp = 0; sp < ns; ++sp) {
                const float w = s_w[sp];
                if (w != 0.0f) {
                    const float* p = A.part + (base + (long long)sp * gqa) * (kHd + 4);
                    sum = fmaf(w, p[t], sum);
                }
            }
            out_b = tobf(sum / s_den);
        }
        o_tok[g * A.o_hs + t] = out_b;
        lds_barrier();
    }
}

template <class CH>
__global__ __launch_bounds__(kThr) void rel_merge_kernel(RelArgs A) {
    if (CH::on) r4d_chain_wait();
    rel_merge_body(A);
    if (CH::on) r4d_chain_done();
}

/* ---- Fast Path B: WMMA Prefill (256 threads / 8 waves, BQ=128, 1 query head per block) ------ */

__device__ __forceinline__ float wshfl_xor(float v, int m) {
    return __int_as_float(__builtin_amdgcn_ds_bpermute((((int)threadIdx.x & 31) ^ m) << 2,
                                                       __float_as_int(v)));
}
__device__ __forceinline__ int wshfl_up(int v, int d) {
    const int l = (int)threadIdx.x & 31;
    return __builtin_amdgcn_ds_bpermute((l >= d ? l - d : l) << 2, v);
}
__device__ __forceinline__ float ex2(float x) {
#if __has_builtin(__builtin_amdgcn_exp2f)
    return __builtin_amdgcn_exp2f(x);
#else
    return exp2f(x);
#endif
}
__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }

__device__ __forceinline__ rel_v8f wmma_bf16(rel_v8s a, rel_v8s b, rel_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ rel_v8f wmma_f16(rel_v8h a, rel_v8h b, rel_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ unsigned f16_bits(float f) {
    if (f == f) f = __builtin_fminf(__builtin_fmaxf(f, -65504.0f), 65504.0f);
    return (unsigned)__builtin_bit_cast(unsigned short, (_Float16)f);
}
__device__ __forceinline__ unsigned f16x2_exact(float a, float b) {
#if defined(__GFX12__)
    return __builtin_bit_cast(unsigned, __builtin_amdgcn_cvt_pkrtz(a, b));
#else
    return f16_bits(a) | (f16_bits(b) << 16);
#endif
}
__device__ __forceinline__ void bf16x8_f16(const uint4 w, unsigned (&h)[8]) {
    const unsigned in[4] = { w.x, w.y, w.z, w.w };
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        h[2 * q]     = f16_bits(__uint_as_float(in[q] << 16));
        h[2 * q + 1] = f16_bits(__uint_as_float(in[q] & 0xffff0000u));
    }
}
template <bool I8>
__device__ __forceinline__ void kv8x4(unsigned w, float (&f)[4]) {
    if constexpr (I8) {
        f[0] = (float)(signed char)(w & 0xffu);
        f[1] = (float)(signed char)((w >> 8) & 0xffu);
        f[2] = (float)(signed char)((w >> 16) & 0xffu);
        f[3] = (float)(signed char)(w >> 24);
    }
}
__device__ __forceinline__ void i8x16_bf16(const uint4 w, unsigned (&h)[16]) {
    const unsigned in[4] = { w.x, w.y, w.z, w.w };
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        float f[4];
        kv8x4<true>(in[q], f);
#pragma unroll
        for (int e = 0; e < 4; ++e) h[q * 4 + e] = __float_as_uint(f[e]) >> 16;
    }
}

__device__ void find_rel_tile(const int32_t* cu, int n_seq, int tile, int* s_scan, int* s_res,
                              int& s_out, int& lt_out) {
    constexpr int NW = kWThreads / 32;
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    if (tid == 0) { s_res[0] = -1; s_res[1] = 0; }
    __syncthreads();
    int base = 0;
    for (int c0 = 0; c0 < n_seq; c0 += kWThreads) {
        const int s = c0 + tid;
        int nt = 0;
        if (s < n_seq) {
            const int ql = cu[s + 1] - cu[s];
            nt = ql > 0 ? (ql + kWBQ - 1) / kWBQ : 0;
        }
        int x = nt;
        for (int d = 1; d < 32; d <<= 1) {
            const int y = wshfl_up(x, d);
            if (lane >= d) x += y;
        }
        if (lane == 31) s_scan[wave] = x;
        __syncthreads();
        int before = 0, total = 0;
        for (int w = 0; w < NW; ++w) {
            const int v = s_scan[w];
            before += (w < wave) ? v : 0;
            total += v;
        }
        const int incl = base + before + x;
        if (nt > 0 && tile >= incl - nt && tile < incl) {
            s_res[0] = s;
            s_res[1] = tile - (incl - nt);
        }
        base += total;
        __syncthreads();
        if (s_res[0] >= 0 || base > tile) break;
    }
    s_out = s_res[0];
    lt_out = s_res[1];
}

template <bool I8, bool DR16>
__global__ __launch_bounds__(kWThreads, kWBQ >= 128 ? 1 : 2) void rel_wmma_kernel(RelArgs A) {
    constexpr int ESZ = I8 ? 1 : 2;
    constexpr int SLOT = 2 * kHd * ESZ + (I8 ? 4 : 0);
    constexpr int CW = 16 / ESZ;
    constexpr int NCH = kHd / CW;
    constexpr int NKI = 16 * NCH;
    constexpr int NVI = 8 * NCH;
    constexpr int KPT = (NKI + kWThreads - 1) / kWThreads;
    constexpr int VPT = (NVI + kWThreads - 1) / kWThreads;

    __shared__ uint4    s_k[16 * kWKS / 16];
    __shared__ uint4    s_vt[kHd * kWVS / 16];
    alignas(16) __shared__ uint16_t s_rp[kWBQ + 16][DR16 ? 24 : 66]; /* 24 shorts: 16B aligned + zero bank conflicts */
    __shared__ uint16_t s_r[DR16 ? 1 : kWBQ][DR16 ? 1 : 66];
    __shared__ float    s_bias[DR16 ? (kWThreads / 32) : 1][DR16 ? 16 : 1][DR16 ? 33 : 1];
    __shared__ unsigned s_sc[I8 ? 16 : 1];
    __shared__ int      s_scan[kWThreads / 32];
    __shared__ int      s_res[2];

    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int tile = (int)blockIdx.x, h_head = (int)blockIdx.y;
    const int kh = h_head / (int)A.gqa;

    int s = -1, lt = 0;
    if (A.cu) {
        find_rel_tile(A.cu, (int)A.n_seq, tile, s_scan, s_res, s, lt);
    } else {
        s = tile / A.tps;
        lt = tile - s * A.tps;
        if (s >= A.n_seq) s = -1;
    }
    int qlen = 0, tokb = 0, ctx = 0;
    if (s >= 0) {
        if (A.cu) { tokb = A.cu[s]; qlen = A.cu[s + 1] - tokb; }
        else      { tokb = s * (int)A.q_len; qlen = (int)A.q_len; }
        ctx = A.seqused[s];
    }
    const int pfirst = lt * kWBQ;
    const int npos = s >= 0 ? imax(0, imin(kWBQ, qlen - pfirst)) : 0;
    if (npos <= 0) return;

    const int tok0 = tokb + pfirst;
    const int qbase = ctx - qlen + pfirst;
    const int plast = qbase + npos - 1;
    const int causal = A.causal, window = (int)A.window;
    int hi = causal ? plast + 1 : ctx;
    if (!causal && window > 0) hi = imin(hi, plast + window);
    hi = imin(imin(hi, ctx), (int)A.max_blocks * 16);
    const int lo = window > 0 ? imax(0, qbase - window + 1) : 0;
    const int lo_a = lo & ~15;
    const int nkt = hi > lo_a ? (hi - lo_a + 15) / 16 : 0;

    const float kds = (!I8 && A.kdes) ? A.kdes[(long long)s * A.kv_heads + kh] : 1.0f;
    const float vds = (!I8 && A.vdes) ? A.vdes[(long long)s * A.kv_heads + kh] : 1.0f;
    const float sscale = A.scale * kds;
    constexpr float kLog2e = 1.4426950408889634f;

    /* Load this lane's Q^T fragment (all 128 dims) and query row coordinates. */
    const int n = lane & 15, h = lane >> 4;
    const int qrow = wave * 16 + n;
    const bool rval = qrow < npos;
    const int pos = qbase + qrow;

    const uint16_t* r16 = (const uint16_t*)A.r + (long long)tok0 * A.r_rs +
                          (long long)h_head * A.r_hs;
    rel_v8s rf{};
    if constexpr (DR16) {
        const uint16_t* rp_q = r16 + (long long)(rval ? qrow : 0) * A.r_rs + 8 * h;
        rf = __builtin_bit_cast(rel_v8s, rval ? ld_u4(rp_q) : make_uint4(0u, 0u, 0u, 0u));
    } else {
        for (int qr = wave; qr < npos; qr += kWThreads / 32) {
            if (lane < A.d_rel) s_r[qr][lane] = r16[(long long)qr * A.r_rs + lane];
            if (A.d_rel > 32 && lane + 32 < A.d_rel)
                s_r[qr][lane + 32] = r16[(long long)qr * A.r_rs + lane + 32];
        }
        __syncthreads();
    }

    rel_v8s qf[kHd / 16];
    {
        const uint16_t* qp = (const uint16_t*)A.q + (long long)(tok0 + (rval ? qrow : 0)) * A.q_rs +
                             (long long)h_head * A.q_hs + 8 * h;
#pragma unroll
        for (int kc = 0; kc < kHd / 16; ++kc) {
            const uint4 w = rval ? ld_u4(qp + kc * 16) : make_uint4(0u, 0u, 0u, 0u);
            qf[kc] = __builtin_bit_cast(rel_v8s, w);
        }
    }
    rel_v8f oacc[kHd / 16];
#pragma unroll
    for (int mc = 0; mc < kHd / 16; ++mc)
#pragma unroll
        for (int e = 0; e < 8; ++e) oacc[mc][e] = 0.0f;

    /* Start m = -inf, l = 0 to match zero-visible-keys == 0.0f contract; fold sink after loop. */
    float m = -__builtin_inff(), l = 0.0f;

    uint4 pk[KPT], pv[VPT][2];
    unsigned psc = 0u;
    bool pre_ok = false;
    auto fetch = [&](int kt) __attribute__((always_inline)) {
        const int k0 = lo_a + kt * 16, lb = k0 >> 4;
        int blk = -1;
        if (lb < A.max_blocks && k0 < hi) blk = A.bt[(long long)s * A.bt_rs + lb];
        pre_ok = blk >= 0;
        const uint8_t* base = A.kv + ((long long)(blk >= 0 ? blk : 0) * A.sb + (long long)kh * A.sh) * ESZ;
#pragma unroll
        for (int j = 0; j < KPT; ++j) {
            const int idx = tid + j * kWThreads;
            pk[j] = make_uint4(0u, 0u, 0u, 0u);
            if (pre_ok && idx < NKI) {
                const int key = idx / NCH, ch = idx - key * NCH;
                pk[j] = ld_u4(base + key * SLOT + ch * 16);
            }
        }
        if constexpr (I8) {
            psc = 0u;
            if (pre_ok && tid < 16) psc = ld_u32(base + tid * SLOT + 2 * kHd);
        }
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            const int idx = tid + j * kWThreads;
            pv[j][0] = pv[j][1] = make_uint4(0u, 0u, 0u, 0u);
            if (pre_ok && idx < NVI) {
                const int kp = idx & 7, ch = idx >> 3;
                const uint8_t* v0 = base + (2 * kp) * SLOT + kHd * ESZ + ch * 16;
                pv[j][0] = ld_u4(v0);
                pv[j][1] = ld_u4(v0 + SLOT);
            }
        }
    };

    const uint16_t* rp16 = (const uint16_t*)A.rp;
    uint8_t* kb = (uint8_t*)s_k;
    uint8_t* vb = (uint8_t*)s_vt;
    if (nkt > 0) fetch(0);
    for (int kt = 0; kt < nkt; ++kt) {
        const int k0 = lo_a + kt * 16;
        const bool tok = pre_ok;
        lds_barrier();

        /* Stage K: [key][dim] as bf16 */
#pragma unroll
        for (int j = 0; j < KPT; ++j) {
            const int idx = tid + j * kWThreads;
            if (idx < NKI) {
                const int key = idx / NCH, ch = idx - key * NCH;
                const int kk = k0 + key;
                const bool keep = tok && kk >= lo && kk < hi;
                const uint4 w = keep ? pk[j] : make_uint4(0u, 0u, 0u, 0u);
                if constexpr (I8) {
                    unsigned hb[16];
                    i8x16_bf16(w, hb);
                    uint4* dst = (uint4*)(kb + key * kWKS + ch * 32);
                    dst[0] = make_uint4(hb[0] | (hb[1] << 16), hb[2] | (hb[3] << 16),
                                        hb[4] | (hb[5] << 16), hb[6] | (hb[7] << 16));
                    dst[1] = make_uint4(hb[8] | (hb[9] << 16), hb[10] | (hb[11] << 16),
                                        hb[12] | (hb[13] << 16), hb[14] | (hb[15] << 16));
                } else {
                    *(uint4*)(kb + key * kWKS + ch * 16) = w;
                }
            }
        }
        if constexpr (I8) {
            if (tid < 16) s_sc[tid] = psc;
        }
        /* Stage V: [dim][key] as f16 */
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            const int idx = tid + j * kWThreads;
            if (idx < NVI) {
                const int kp = idx & 7, ch = idx >> 3;
                const int ka = k0 + 2 * kp, kb1 = ka + 1;
                const uint4 wa = (tok && ka >= lo && ka < hi) ? pv[j][0] : make_uint4(0u, 0u, 0u, 0u);
                const uint4 wb = (tok && kb1 >= lo && kb1 < hi) ? pv[j][1] : make_uint4(0u, 0u, 0u, 0u);
                unsigned d[CW];
                if constexpr (I8) {
                    const unsigned ia[4] = { wa.x, wa.y, wa.z, wa.w };
                    const unsigned ib[4] = { wb.x, wb.y, wb.z, wb.w };
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        float fa[4], fb[4];
                        kv8x4<true>(ia[q], fa);
                        kv8x4<true>(ib[q], fb);
#pragma unroll
                        for (int e = 0; e < 4; ++e) d[q * 4 + e] = f16x2_exact(fa[e], fb[e]);
                    }
                } else {
                    unsigned ha[8], hb2[8];
                    bf16x8_f16(wa, ha);
                    bf16x8_f16(wb, hb2);
#pragma unroll
                    for (int e = 0; e < CW; ++e) d[e] = ha[e] | (hb2[e] << 16);
                }
                const int rot = 2 * ((lane >> 3) & 3);
                unsigned* col = (unsigned*)(vb + (ch * CW) * kWVS) + kp;
#pragma unroll
                for (int t = 0; t < CW; ++t) {
                    const unsigned v0 = d[t], v2 = d[(t + 2) % CW], v4 = d[(t + 4) % CW],
                                   v6 = d[(t + 6) % CW];
                    const unsigned v = rot == 0 ? v0 : rot == 2 ? v2 : rot == 4 ? v4 : v6;
                    const int e = (t + rot) % CW;
                    col[e * (kWVS / 4)] = v;
                }
            }
        }
        /* Stage rel_proj columns [d_min .. d_min + kWBQ + 14] (plus zero tail row for DR16 WMMA). */
        const int d_min = qbase - (k0 + 15);
        const int d_max = plast - k0;
        const bool need_rp = tok && d_max >= 0 && d_min < (int)A.extent;
        if constexpr (DR16) {
            if (need_rp && tid < kWBQ + 16) {
                const int dist = d_min + tid;
                const bool dv = tid < kWBQ + 15 && dist >= 0 && dist < (int)A.extent;
                uint32_t w0[4], w1[4];
                if (dv) {
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const uint32_t lo0 = rp16[(long long)(2 * p) * A.rp_c + dist];
                        const uint32_t hi0 = rp16[(long long)(2 * p + 1) * A.rp_c + dist];
                        w0[p] = lo0 | (hi0 << 16);
                        const uint32_t lo1 = rp16[(long long)(8 + 2 * p) * A.rp_c + dist];
                        const uint32_t hi1 = rp16[(long long)(9 + 2 * p) * A.rp_c + dist];
                        w1[p] = lo1 | (hi1 << 16);
                    }
                } else {
#pragma unroll
                    for (int p = 0; p < 4; ++p) w0[p] = w1[p] = 0u;
                }
                *(uint4*)(&s_rp[tid][0]) = make_uint4(w0[0], w0[1], w0[2], w0[3]);
                *(uint4*)(&s_rp[tid][8]) = make_uint4(w1[0], w1[1], w1[2], w1[3]);
            }
        } else {
            if (need_rp && tid < kWBQ + 15) {
                const int dist = d_min + tid;
                const bool dv = dist >= 0 && dist < (int)A.extent;
                for (int c = 0; c < (int)A.d_rel; ++c)
                    s_rp[tid][c] = dv ? rp16[(long long)c * A.rp_c + dist] : 0;
            }
        }
        lds_barrier();
        if (kt + 1 < nkt) fetch(kt + 1);
        if constexpr (DR16) {
            if (need_rp) {
                rel_v8f b0 = {}, b1 = {};
                const uint4 a0 = ld_u4(&s_rp[wave * 16 + n][8 * h]);
                const uint4 a1 = ld_u4(&s_rp[wave * 16 + 16 + n][8 * h]);
                b0 = wmma_bf16(__builtin_bit_cast(rel_v8s, a0), rf, b0);
                b1 = wmma_bf16(__builtin_bit_cast(rel_v8s, a1), rf, b1);
                float* sb = &s_bias[wave][n][0];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    sb[8 * h + e]      = b0[e];
                    sb[16 + 8 * h + e] = b1[e];
                }
                lds_barrier();
            }
        }

        /* S^T = K * Q^T over all 128 dims (8 WMMAs, NS=1). */
        rel_v8f sacc;
#pragma unroll
        for (int e = 0; e < 8; ++e) sacc[e] = 0.0f;
        {
            const uint8_t* ka = kb + n * kWKS + (8 * h) * 2;
#pragma unroll
            for (int kc = 0; kc < kHd / 16; ++kc)
                sacc = wmma_bf16(__builtin_bit_cast(rel_v8s, *(const uint4*)(ka + kc * 32)),
                                 qf[kc], sacc);
        }

        if constexpr (I8) {
#pragma unroll
            for (int e = 0; e < 8; ++e)
                sacc[e] *= (float)__builtin_bit_cast(
                    _Float16, (unsigned short)(s_sc[8 * h + e] & 0xffffu));
        }

        /* Add relative bias r[qrow] . rp[:, dist] and convert score to log2 domain. */
        float sv[8], mx = -__builtin_inff();
        const int u0 = qrow + 15 - 8 * h;
        const int d_rel = (int)A.d_rel;
        const bool interior = rval && tok && k0 >= lo && (k0 + 15) < hi &&
                              (!causal || (k0 + 15) <= pos) &&
                              (window <= 0 || (pos - k0 < window && (causal || (k0 + 15) - pos < window)));
        bool fast = false;
        if (interior && !need_rp) {
            const float sscale_l2 = sscale * kLog2e;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                sv[e] = sacc[e] * sscale_l2;
                mx = __builtin_fmaxf(mx, sv[e]);
            }
            fast = true;
        } else if constexpr (DR16) {
            if (interior && need_rp && (pos - (k0 + 15)) >= 0 && (pos - k0) < (int)A.extent) {
                const float* sb = &s_bias[wave][n][0];
                const int m0 = 15 + n - 8 * h;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    sv[e] = (sacc[e] * sscale + sb[m0 - e]) * kLog2e;
                    mx = __builtin_fmaxf(mx, sv[e]);
                }
                fast = true;
            }
        }
        if (!fast) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int kk = k0 + 8 * h + e;
                bool vis = rval && tok && kk >= lo && kk < hi;
                if (causal) vis = vis && kk <= pos;
                if (window > 0) vis = vis && (pos - kk < window) && (causal || kk - pos < window);
                if (vis) {
                    float bias = 0.0f;
                    const int dist = pos - kk;
                    if (need_rp && dist >= 0 && dist < (int)A.extent) {
                        const int u = u0 - e;
                        if constexpr (DR16) {
                            bias = s_bias[wave][n][15 + n - 8 * h - e];
                        } else {
                            for (int c = 0; c < d_rel; ++c)
                                bias += bf(s_r[qrow][c]) * bf(s_rp[u][c]);
                        }
                    }
                    sv[e] = (sacc[e] * sscale + bias) * kLog2e;
                } else {
                    sv[e] = -__builtin_inff();
                }
                mx = __builtin_fmaxf(mx, sv[e]);
            }
        }
        mx = __builtin_fmaxf(mx, wshfl_xor(mx, 16));
        const float mn = __builtin_fmaxf(m, mx);
        const float corr = mn == -__builtin_inff() ? 1.0f : (m == -__builtin_inff() ? 0.0f : ex2(m - mn));
        float ps = 0.0f;
        rel_v8h pf;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float pe = (sv[e] == -__builtin_inff() || mn == -__builtin_inff())
                                 ? 0.0f
                                 : ex2(sv[e] - mn);
            pf[e] = (_Float16)pe;
            ps += (float)pf[e];
        }
        l = l * corr + ps;
        m = mn;
        if constexpr (I8) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float pe = (float)pf[e];
                const float vs = (float)__builtin_bit_cast(
                    _Float16, (unsigned short)(s_sc[8 * h + e] >> 16));
                pf[e] = pe != 0.0f ? (_Float16)(pe * vs) : (_Float16)0.0f;
            }
        }
        const uint8_t* va = vb + n * kWVS + h * 16;
        if (__builtin_amdgcn_ballot_w32(corr != 1.0f) != 0) {
#pragma unroll
            for (int mc = 0; mc < kHd / 16; ++mc) oacc[mc] *= corr;
        }
#pragma unroll
        for (int mc = 0; mc < kHd / 16; ++mc)
            oacc[mc] = wmma_f16(__builtin_bit_cast(rel_v8h, *(const uint4*)(va + mc * 16 * kWVS)),
                                pf, oacc[mc]);
    }

    const float lt2 = l + wshfl_xor(l, 16);
    if (rval) {
        uint16_t* op = (uint16_t*)A.out + (long long)(tok0 + qrow) * A.o_rs +
                       (long long)h_head * A.o_hs + 8 * h;
        if (m == -__builtin_inff()) {
#pragma unroll
            for (int mc = 0; mc < kHd / 16; ++mc)
                *(uint4*)(op + mc * 16) = make_uint4(0u, 0u, 0u, 0u);
        } else {
            const float sk_log2 = ld_bf_or_f(A.sk, A.s_f32,
                                             (long long)(tok0 + qrow) * A.s_rs +
                                             (long long)h_head * A.s_hs) *
                                  kLog2e;
            const float mfinal = __builtin_fmaxf(m, sk_log2);
            const float alpha = ex2(m - mfinal);
            const float den = lt2 * alpha + ex2(sk_log2 - mfinal);
            const float inv = den > 0.0f ? (vds * alpha) / den : 0.0f;
#pragma unroll
            for (int mc = 0; mc < kHd / 16; ++mc)
                *(uint4*)(op + mc * 16) = make_uint4(
                    pack2(oacc[mc][0] * inv, oacc[mc][1] * inv),
                    pack2(oacc[mc][2] * inv, oacc[mc][3] * inv),
                    pack2(oacc[mc][4] * inv, oacc[mc][5] * inv),
                    pack2(oacc[mc][6] * inv, oacc[mc][7] * inv));
        }
    }
}

/* Packed verify: rel_wmma_kernel's tile walk for a verify's few rows. The decode kernel gives each
 * verify row its own blocks, and each scores the group's heads one key at a time, so eight rows
 * cost eight times one. Here a block is one sequence's verify rows for one KV head, and a WMMA
 * column is a (row, head of the group) pair, qrow = row * gqa + head: every row and head shares each
 * staged K and V tile. blockIdx.z takes a contiguous run of the key tiles and, when there is more
 * than one split, leaves rel_dec_body's partial for rel_merge_kernel. A column's position, masks and
 * bias distance follow its row (prow). Kept apart from rel_wmma_kernel: sharing one body moved the
 * prefill's code and cost it 3-5%. */
template <bool I8, bool DR16>
__device__ __forceinline__ void rel_vfy_body(RelArgs A) {
    constexpr int ESZ = I8 ? 1 : 2;
    constexpr int SLOT = 2 * kHd * ESZ + (I8 ? 4 : 0);
    constexpr int CW = 16 / ESZ;
    constexpr int NCH = kHd / CW;
    constexpr int NKI = 16 * NCH;
    constexpr int NVI = 8 * NCH;
    constexpr int KPT = (NKI + kWThreads - 1) / kWThreads;
    constexpr int VPT = (NVI + kWThreads - 1) / kWThreads;

    __shared__ uint4    s_k[16 * kWKS / 16];
    __shared__ uint4    s_vt[kHd * kWVS / 16];
    alignas(16) __shared__ uint16_t s_rp[kWBQ + 16][DR16 ? 24 : 66]; /* 24 shorts: 16B aligned + zero bank conflicts */
    __shared__ uint16_t s_r[DR16 ? 1 : kWBQ][DR16 ? 1 : 66];
    __shared__ float    s_bias[DR16 ? (kWThreads / 32) : 1][DR16 ? 16 : 1][DR16 ? 33 : 1];
    __shared__ unsigned s_sc[I8 ? 16 : 1];

    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int gqa = (int)A.gqa;
    const int s = (int)blockIdx.x, kh = (int)blockIdx.y, sp = (int)blockIdx.z;
    if (s >= A.n_seq) return;
    int tokb, qlen;
    if (A.cu) { tokb = A.cu[s]; qlen = A.cu[s + 1] - tokb; }
    else      { tokb = s * (int)A.q_len; qlen = (int)A.q_len; }
    const int ctx = A.seqused[s];
    /* The launch keeps qlen * gqa within the block's 128 columns. */
    const int npos = imin(kWBQ, imax(0, qlen) * gqa);
    if (npos <= 0) return;

    const int tok0 = tokb;
    const int qbase = ctx - qlen;
    const int plast = qbase + qlen - 1;
    const int causal = A.causal, window = (int)A.window;
    int hi = causal ? plast + 1 : ctx;
    if (!causal && window > 0) hi = imin(hi, plast + window);
    hi = imin(imin(hi, ctx), (int)A.max_blocks * 16);
    const int lo = window > 0 ? imax(0, qbase - window + 1) : 0;
    const int lo_a = lo & ~15;
    const int nkt = hi > lo_a ? (hi - lo_a + 15) / 16 : 0;
    /* This split's key tiles [kt0, kt1). */
    const int per = (nkt + A.n_split - 1) / A.n_split;
    const int kt0 = imin(nkt, sp * per), kt1 = imin(nkt, kt0 + per);

    const float kds = (!I8 && A.kdes) ? A.kdes[(long long)s * A.kv_heads + kh] : 1.0f;
    const float vds = (!I8 && A.vdes) ? A.vdes[(long long)s * A.kv_heads + kh] : 1.0f;
    const float sscale = A.scale * kds;
    constexpr float kLog2e = 1.4426950408889634f;

    /* This lane's column, its row and head, and its Q^T fragment (all 128 dims). pw is the first
     * row of this wave's columns, where its 32-distance bias window starts. A wave with no columns
     * (wact false) only stages. */
    const int n = lane & 15, h = lane >> 4;
    const int qrow = wave * 16 + n;
    const bool rval = qrow < npos;
    const int prow = (rval ? qrow : 0) / gqa;
    const int head = kh * gqa + (rval ? qrow : 0) % gqa;
    const int pw = (wave * 16) / gqa;
    const int pos = qbase + prow;
    const bool wact = wave * 16 < npos;

    rel_v8s rf{};
    if constexpr (DR16) {
        const uint16_t* rp_q = (const uint16_t*)A.r + (long long)(tok0 + prow) * A.r_rs +
                               (long long)head * A.r_hs + 8 * h;
        rf = __builtin_bit_cast(rel_v8s, rval ? ld_u4(rp_q) : make_uint4(0u, 0u, 0u, 0u));
    } else {
        for (int qr = wave; qr < npos; qr += kWThreads / 32) {
            const uint16_t* rq = (const uint16_t*)A.r + (long long)(tok0 + qr / gqa) * A.r_rs +
                                 (long long)(kh * gqa + qr % gqa) * A.r_hs;
            if (lane < A.d_rel) s_r[qr][lane] = rq[lane];
            if (A.d_rel > 32 && lane + 32 < A.d_rel) s_r[qr][lane + 32] = rq[lane + 32];
        }
        __syncthreads();
    }

    rel_v8s qf[kHd / 16];
    {
        const uint16_t* qp = (const uint16_t*)A.q + (long long)(tok0 + prow) * A.q_rs +
                             (long long)head * A.q_hs + 8 * h;
#pragma unroll
        for (int kc = 0; kc < kHd / 16; ++kc) {
            const uint4 w = rval ? ld_u4(qp + kc * 16) : make_uint4(0u, 0u, 0u, 0u);
            qf[kc] = __builtin_bit_cast(rel_v8s, w);
        }
    }
    rel_v8f oacc[kHd / 16];
#pragma unroll
    for (int mc = 0; mc < kHd / 16; ++mc)
#pragma unroll
        for (int e = 0; e < 8; ++e) oacc[mc][e] = 0.0f;

    /* Start m = -inf, l = 0 to match zero-visible-keys == 0.0f contract; fold sink after loop. */
    float m = -__builtin_inff(), l = 0.0f;

    uint4 pk[KPT], pv[VPT][2];
    unsigned psc = 0u;
    bool pre_ok = false;
    auto fetch = [&](int kt) __attribute__((always_inline)) {
        const int k0 = lo_a + kt * 16, lb = k0 >> 4;
        int blk = -1;
        if (lb < A.max_blocks && k0 < hi) blk = A.bt[(long long)s * A.bt_rs + lb];
        pre_ok = blk >= 0;
        const uint8_t* base = A.kv + ((long long)(blk >= 0 ? blk : 0) * A.sb + (long long)kh * A.sh) * ESZ;
#pragma unroll
        for (int j = 0; j < KPT; ++j) {
            const int idx = tid + j * kWThreads;
            pk[j] = make_uint4(0u, 0u, 0u, 0u);
            if (pre_ok && idx < NKI) {
                const int key = idx / NCH, ch = idx - key * NCH;
                pk[j] = ld_u4(base + key * SLOT + ch * 16);
            }
        }
        if constexpr (I8) {
            psc = 0u;
            if (pre_ok && tid < 16) psc = ld_u32(base + tid * SLOT + 2 * kHd);
        }
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            const int idx = tid + j * kWThreads;
            pv[j][0] = pv[j][1] = make_uint4(0u, 0u, 0u, 0u);
            if (pre_ok && idx < NVI) {
                const int kp = idx & 7, ch = idx >> 3;
                const uint8_t* v0 = base + (2 * kp) * SLOT + kHd * ESZ + ch * 16;
                pv[j][0] = ld_u4(v0);
                pv[j][1] = ld_u4(v0 + SLOT);
            }
        }
    };

    const uint16_t* rp16 = (const uint16_t*)A.rp;
    uint8_t* kb = (uint8_t*)s_k;
    uint8_t* vb = (uint8_t*)s_vt;
    if (kt0 < kt1) fetch(kt0);
    for (int kt = kt0; kt < kt1; ++kt) {
        const int k0 = lo_a + kt * 16;
        const bool tok = pre_ok;
        lds_barrier();

        /* Stage K: [key][dim] as bf16 */
#pragma unroll
        for (int j = 0; j < KPT; ++j) {
            const int idx = tid + j * kWThreads;
            if (idx < NKI) {
                const int key = idx / NCH, ch = idx - key * NCH;
                const int kk = k0 + key;
                const bool keep = tok && kk >= lo && kk < hi;
                const uint4 w = keep ? pk[j] : make_uint4(0u, 0u, 0u, 0u);
                if constexpr (I8) {
                    unsigned hb[16];
                    i8x16_bf16(w, hb);
                    uint4* dst = (uint4*)(kb + key * kWKS + ch * 32);
                    dst[0] = make_uint4(hb[0] | (hb[1] << 16), hb[2] | (hb[3] << 16),
                                        hb[4] | (hb[5] << 16), hb[6] | (hb[7] << 16));
                    dst[1] = make_uint4(hb[8] | (hb[9] << 16), hb[10] | (hb[11] << 16),
                                        hb[12] | (hb[13] << 16), hb[14] | (hb[15] << 16));
                } else {
                    *(uint4*)(kb + key * kWKS + ch * 16) = w;
                }
            }
        }
        if constexpr (I8) {
            if (tid < 16) s_sc[tid] = psc;
        }
        /* Stage V: [dim][key] as f16 */
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            const int idx = tid + j * kWThreads;
            if (idx < NVI) {
                const int kp = idx & 7, ch = idx >> 3;
                const int ka = k0 + 2 * kp, kb1 = ka + 1;
                const uint4 wa = (tok && ka >= lo && ka < hi) ? pv[j][0] : make_uint4(0u, 0u, 0u, 0u);
                const uint4 wb = (tok && kb1 >= lo && kb1 < hi) ? pv[j][1] : make_uint4(0u, 0u, 0u, 0u);
                unsigned d[CW];
                if constexpr (I8) {
                    const unsigned ia[4] = { wa.x, wa.y, wa.z, wa.w };
                    const unsigned ib[4] = { wb.x, wb.y, wb.z, wb.w };
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        float fa[4], fb[4];
                        kv8x4<true>(ia[q], fa);
                        kv8x4<true>(ib[q], fb);
#pragma unroll
                        for (int e = 0; e < 4; ++e) d[q * 4 + e] = f16x2_exact(fa[e], fb[e]);
                    }
                } else {
                    unsigned ha[8], hb2[8];
                    bf16x8_f16(wa, ha);
                    bf16x8_f16(wb, hb2);
#pragma unroll
                    for (int e = 0; e < CW; ++e) d[e] = ha[e] | (hb2[e] << 16);
                }
                const int rot = 2 * ((lane >> 3) & 3);
                unsigned* col = (unsigned*)(vb + (ch * CW) * kWVS) + kp;
#pragma unroll
                for (int t = 0; t < CW; ++t) {
                    const unsigned v0 = d[t], v2 = d[(t + 2) % CW], v4 = d[(t + 4) % CW],
                                   v6 = d[(t + 6) % CW];
                    const unsigned v = rot == 0 ? v0 : rot == 2 ? v2 : rot == 4 ? v4 : v6;
                    const int e = (t + rot) % CW;
                    col[e * (kWVS / 4)] = v;
                }
            }
        }
        /* Stage rel_proj columns [d_min .. d_min + kWBQ + 14] (plus zero tail row for DR16 WMMA). */
        const int d_min = qbase - (k0 + 15);
        const int d_max = plast - k0;
        const bool need_rp = tok && d_max >= 0 && d_min < (int)A.extent;
        if constexpr (DR16) {
            if (need_rp && tid < kWBQ + 16) {
                const int dist = d_min + tid;
                const bool dv = tid < kWBQ + 15 && dist >= 0 && dist < (int)A.extent;
                uint32_t w0[4], w1[4];
                if (dv) {
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const uint32_t lo0 = rp16[(long long)(2 * p) * A.rp_c + dist];
                        const uint32_t hi0 = rp16[(long long)(2 * p + 1) * A.rp_c + dist];
                        w0[p] = lo0 | (hi0 << 16);
                        const uint32_t lo1 = rp16[(long long)(8 + 2 * p) * A.rp_c + dist];
                        const uint32_t hi1 = rp16[(long long)(9 + 2 * p) * A.rp_c + dist];
                        w1[p] = lo1 | (hi1 << 16);
                    }
                } else {
#pragma unroll
                    for (int p = 0; p < 4; ++p) w0[p] = w1[p] = 0u;
                }
                *(uint4*)(&s_rp[tid][0]) = make_uint4(w0[0], w0[1], w0[2], w0[3]);
                *(uint4*)(&s_rp[tid][8]) = make_uint4(w1[0], w1[1], w1[2], w1[3]);
            }
        } else {
            if (need_rp && tid < kWBQ + 15) {
                const int dist = d_min + tid;
                const bool dv = dist >= 0 && dist < (int)A.extent;
                for (int c = 0; c < (int)A.d_rel; ++c)
                    s_rp[tid][c] = dv ? rp16[(long long)c * A.rp_c + dist] : 0;
            }
        }
        lds_barrier();
        if (kt + 1 < kt1) fetch(kt + 1);
        if constexpr (DR16) {
            if (need_rp) {
                if (wact) {
                    rel_v8f b0 = {}, b1 = {};
                    const uint4 a0 = ld_u4(&s_rp[pw + n][8 * h]);
                    const uint4 a1 = ld_u4(&s_rp[pw + 16 + n][8 * h]);
                    b0 = wmma_bf16(__builtin_bit_cast(rel_v8s, a0), rf, b0);
                    b1 = wmma_bf16(__builtin_bit_cast(rel_v8s, a1), rf, b1);
                    float* sb = &s_bias[wave][n][0];
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        sb[8 * h + e]      = b0[e];
                        sb[16 + 8 * h + e] = b1[e];
                    }
                }
                lds_barrier();
            }
        }
        /* A wave with no columns only stages: its scores would all be masked. No barrier below. */
        if (!wact) continue;

        /* S^T = K * Q^T over all 128 dims (8 WMMAs, NS=1). */
        rel_v8f sacc;
#pragma unroll
        for (int e = 0; e < 8; ++e) sacc[e] = 0.0f;
        {
            const uint8_t* ka = kb + n * kWKS + (8 * h) * 2;
#pragma unroll
            for (int kc = 0; kc < kHd / 16; ++kc)
                sacc = wmma_bf16(__builtin_bit_cast(rel_v8s, *(const uint4*)(ka + kc * 32)),
                                 qf[kc], sacc);
        }

        if constexpr (I8) {
#pragma unroll
            for (int e = 0; e < 8; ++e)
                sacc[e] *= (float)__builtin_bit_cast(
                    _Float16, (unsigned short)(s_sc[8 * h + e] & 0xffffu));
        }

        /* Add relative bias r[qrow] . rp[:, dist] and convert score to log2 domain. */
        float sv[8], mx = -__builtin_inff();
        const int u0 = prow + 15 - 8 * h;
        const int d_rel = (int)A.d_rel;
        const bool interior = rval && tok && k0 >= lo && (k0 + 15) < hi &&
                              (!causal || (k0 + 15) <= pos) &&
                              (window <= 0 || (pos - k0 < window && (causal || (k0 + 15) - pos < window)));
        bool fast = false;
        if (interior && !need_rp) {
            const float sscale_l2 = sscale * kLog2e;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                sv[e] = sacc[e] * sscale_l2;
                mx = __builtin_fmaxf(mx, sv[e]);
            }
            fast = true;
        } else if constexpr (DR16) {
            if (interior && need_rp && (pos - (k0 + 15)) >= 0 && (pos - k0) < (int)A.extent) {
                const float* sb = &s_bias[wave][n][0];
                const int m0 = 15 + (prow - pw) - 8 * h;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    sv[e] = (sacc[e] * sscale + sb[m0 - e]) * kLog2e;
                    mx = __builtin_fmaxf(mx, sv[e]);
                }
                fast = true;
            }
        }
        if (!fast) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int kk = k0 + 8 * h + e;
                bool vis = rval && tok && kk >= lo && kk < hi;
                if (causal) vis = vis && kk <= pos;
                if (window > 0) vis = vis && (pos - kk < window) && (causal || kk - pos < window);
                if (vis) {
                    float bias = 0.0f;
                    const int dist = pos - kk;
                    if (need_rp && dist >= 0 && dist < (int)A.extent) {
                        const int u = u0 - e;
                        if constexpr (DR16) {
                            bias = s_bias[wave][n][15 + (prow - pw) - 8 * h - e];
                        } else {
                            for (int c = 0; c < d_rel; ++c)
                                bias += bf(s_r[qrow][c]) * bf(s_rp[u][c]);
                        }
                    }
                    sv[e] = (sacc[e] * sscale + bias) * kLog2e;
                } else {
                    sv[e] = -__builtin_inff();
                }
                mx = __builtin_fmaxf(mx, sv[e]);
            }
        }
        mx = __builtin_fmaxf(mx, wshfl_xor(mx, 16));
        const float mn = __builtin_fmaxf(m, mx);
        const float corr = mn == -__builtin_inff() ? 1.0f : (m == -__builtin_inff() ? 0.0f : ex2(m - mn));
        float ps = 0.0f;
        rel_v8h pf;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float pe = (sv[e] == -__builtin_inff() || mn == -__builtin_inff())
                                 ? 0.0f
                                 : ex2(sv[e] - mn);
            pf[e] = (_Float16)pe;
            ps += (float)pf[e];
        }
        l = l * corr + ps;
        m = mn;
        if constexpr (I8) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float pe = (float)pf[e];
                const float vs = (float)__builtin_bit_cast(
                    _Float16, (unsigned short)(s_sc[8 * h + e] >> 16));
                pf[e] = pe != 0.0f ? (_Float16)(pe * vs) : (_Float16)0.0f;
            }
        }
        const uint8_t* va = vb + n * kWVS + h * 16;
        if (__builtin_amdgcn_ballot_w32(corr != 1.0f) != 0) {
#pragma unroll
            for (int mc = 0; mc < kHd / 16; ++mc) oacc[mc] *= corr;
        }
#pragma unroll
        for (int mc = 0; mc < kHd / 16; ++mc)
            oacc[mc] = wmma_f16(__builtin_bit_cast(rel_v8h, *(const uint4*)(va + mc * 16 * kWVS)),
                                pf, oacc[mc]);
    }

    const float lt2 = l + wshfl_xor(l, 16);
    if (A.n_split > 1) {
        /* rel_dec_body's partial: the unnormalised sum with the value descale, m in natural
         * units, the denominator. An empty split leaves -inf, 0 and zeros. */
        if (rval) {
            const long long slot =
                (((long long)(tok0 + prow) * A.kv_heads + kh) * A.n_split + sp) * gqa + qrow % gqa;
            float* dst = A.part + slot * (kHd + 4);
#pragma unroll
            for (int mc = 0; mc < kHd / 16; ++mc)
#pragma unroll
                for (int e = 0; e < 8; ++e) dst[mc * 16 + 8 * h + e] = oacc[mc][e] * vds;
            if (h == 0) {
                dst[kHd]     = m == -__builtin_inff() ? m : m * 0.69314718055994531f;
                dst[kHd + 1] = lt2;
            }
        }
        return;
    }
    if (rval) {
        uint16_t* op = (uint16_t*)A.out + (long long)(tok0 + prow) * A.o_rs +
                       (long long)head * A.o_hs + 8 * h;
        if (m == -__builtin_inff()) {
#pragma unroll
            for (int mc = 0; mc < kHd / 16; ++mc)
                *(uint4*)(op + mc * 16) = make_uint4(0u, 0u, 0u, 0u);
        } else {
            const float sk_log2 = ld_bf_or_f(A.sk, A.s_f32,
                                             (long long)(tok0 + prow) * A.s_rs +
                                             (long long)head * A.s_hs) *
                                  kLog2e;
            const float mfinal = __builtin_fmaxf(m, sk_log2);
            const float alpha = ex2(m - mfinal);
            const float den = lt2 * alpha + ex2(sk_log2 - mfinal);
            const float inv = den > 0.0f ? (vds * alpha) / den : 0.0f;
#pragma unroll
            for (int mc = 0; mc < kHd / 16; ++mc)
                *(uint4*)(op + mc * 16) = make_uint4(
                    pack2(oacc[mc][0] * inv, oacc[mc][1] * inv),
                    pack2(oacc[mc][2] * inv, oacc[mc][3] * inv),
                    pack2(oacc[mc][4] * inv, oacc[mc][5] * inv),
                    pack2(oacc[mc][6] * inv, oacc[mc][7] * inv));
        }
    }
}

/* Grid (sequences, KV heads, splits); see rel_vfy_body. */
template <bool I8, bool DR16, class CH>
__global__ __launch_bounds__(kWThreads, 1) void rel_vfy_kernel(RelArgs A) {
    if (CH::on) r4d_chain_wait();
    rel_vfy_body<I8, DR16>(A);
    if (CH::on) r4d_chain_done();
}

static inline int64_t rel_scratch_bytes() {
    return (int64_t)kRelPartBudget * kMaxGqa * (kHd + 4) * (int64_t)sizeof(float);
}

/* Last decode launch's split count, for the device test. 0 if that launch did not take this path. */
int g_rel_last_splits = 0;

/* Decode split count. `span` is the keys this launch can read. A split keeps kRelTilesMin tiles, so
 * a short decode is not a wide reduction of empty partials (the same floor as rad_prefill_splits_law,
 * and for the same reason: fewer tiles changes the reduction order and does not hide the walk).
 * Sixteen 128-thread blocks a WGP is the fill; scratch holds 128 splits at four KV heads, which is
 * a 32-WGP card's fill at one query row. */
static int rel_decode_splits(int pairs, int span, int wgps) {
    if (pairs <= 0 || span <= 1) return 1;
    const int nkt = (span + kDecBK - 1) / kDecBK;
    if (nkt <= 1) return 1;
    int ns = kRelPartBudget / pairs;
    if (ns > kRelMaxSplit) ns = kRelMaxSplit;
    const int by_work = nkt / kRelTilesMin;
    if (ns > by_work) ns = by_work;
    const int cards = wgps > 0 ? wgps : 32;
    const int by_fill = (cards * kRelBlocksPerWgp + pairs - 1) / pairs;
    if (ns > by_fill) ns = by_fill;
    return ns < 1 ? 1 : ns;
}

/* Packed verify split count. Its partials are (row, KV head, split, head) slots, so the scratch
 * that holds kRelPartBudget slots of kMaxGqa heads bounds the splits at rows * kv_heads * gqa. A
 * split keeps sixteen 16-key tiles, and eight 256-thread blocks a WGP is the fill. */
static int rel_verify_splits(long long n_seq, long long rows, long long kv_heads, long long gqa,
                             long long span, int wgps) {
    const long long slots = rows * kv_heads * gqa, blocks = n_seq * kv_heads;
    if (slots <= 0 || blocks <= 0 || span <= 1) return 1;
    long long ns = (long long)kRelPartBudget * kMaxGqa / slots;
    if (ns > kRelMaxSplit) ns = kRelMaxSplit;
    const long long by_work = ((span + 15) / 16) / 16;
    if (ns > by_work) ns = by_work;
    const long long by_fill = ((wgps > 0 ? wgps : 32) * 8LL + blocks - 1) / blocks;
    if (ns > by_fill) ns = by_fill;
    return ns < 1 ? 1 : (int)ns;
}

template <bool I8, bool DR16>
static void launch_vfy(bool chain, dim3 grid, hipStream_t hs, const RelArgs& A) {
    if (chain)
        R4D_KLAUNCH(R4D_KNAME(rel_vfy_kernel<I8, DR16, r4dchain>), grid, dim3(kWThreads), hs, A);
    else
        R4D_KLAUNCH(R4D_KNAME(rel_vfy_kernel<I8, DR16, R4dUnchained>), grid, dim3(kWThreads), hs, A);
}

/* Shared host side: validate geometry, dispatch WMMA prefill, split-KV decode, or fallback. */
static int launch_rel(const RadArgs* a, RadStream stream, bool i8) {
    g_rel_last_splits = 0;
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

    const bool q_f32 = q->dtype == RAD_F32, o_f32 = o->dtype == RAD_F32;
    if ((q->dtype != RAD_BF16 && !q_f32) || (o->dtype != RAD_BF16 && !o_f32)) return RAD_E_DTYPE;
    if (kv->dtype != (i8 ? RAD_I8 : RAD_BF16)) return RAD_E_DTYPE;
    if (bt->dtype != RAD_I32 || su->dtype != RAD_I32 || (cu && cu->dtype != RAD_I32)) return RAD_E_DTYPE;
    if ((kd && kd->dtype != RAD_F32) || (vd && vd->dtype != RAD_F32)) return RAD_E_DTYPE;
    const bool r_f32 = rr->dtype == RAD_F32, rp_f32 = rp->dtype == RAD_F32, s_f32 = sk->dtype == RAD_F32;
    if ((rr->dtype != RAD_BF16 && !r_f32) || (rp->dtype != RAD_BF16 && !rp_f32) ||
        (sk->dtype != RAD_BF16 && !s_f32)) return RAD_E_DTYPE;

    if (q->stride[q->rank - 1] < 1 || o->stride[o->rank - 1] != 1 || rr->stride[rr->rank - 1] != 1 ||
        sk->stride[sk->rank - 1] < 1 || rp->stride[1] != 1 || kv->stride[3] != 1)
        return RAD_E_STRIDE;
    if (i8 && kv->shape[3] != 2 * kHd + 4) return RAD_E_SHAPE;
    if (!i8 && kv->shape[3] < 2 * kHd) return RAD_E_SHAPE;

    const long long kv_heads = kv->shape[1];
    const long long nh = gqa * kv_heads;
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

    const bool fast_ok = !q_f32 && !r_f32 && !rp_f32 && !o_f32 &&
                         A.q_ds == 1 && A.r_ds == 1 && A.o_ds == 1 && A.rp_d == 1 &&
                         A.bt_cs == 1 && bs == 16 && gqa <= kMaxGqa &&
                         rad_ptr_aligned(q->data, 8) && rad_ptr_aligned(o->data, 8) &&
                         ((A.q_rs | A.q_hs | A.o_rs | A.o_hs) % 4) == 0;

    if (fast_ok) {
        const long long esz = i8 ? 1 : 2;
        long long qmax = cu ? rad_args_geti_or(a, "q_len", rows) : q_len;
        if (qmax <= 0 || qmax > rows) qmax = rows;

        const bool wmma_aligned =
            rad_ptr_aligned(q->data, 16) && rad_ptr_aligned(o->data, 16) &&
            rad_ptr_aligned(kv->data, 16) && ((A.q_rs | A.q_hs | A.o_rs | A.o_hs) % 8) == 0 &&
            (A.sb * esz) % 16 == 0 && (A.sh * esz) % 16 == 0 &&
            kv->stride[2] == (i8 ? (2 * kHd + 4) : (2 * kHd));
        if (wmma_aligned && qmax >= 32 && rows >= n_seq * 16) {
            long long ntw = 0, tps = 0;
            if (!cu) {
                tps = (q_len + kWBQ - 1) / kWBQ;
                ntw = n_seq * tps;
            } else {
                const long long b1 = (rows + n_seq * (kWBQ - 1)) / kWBQ;
                const long long b2 = n_seq * ((qmax + kWBQ - 1) / kWBQ);
                ntw = b1 < b2 ? b1 : b2;
            }
            if (ntw > 0 && ntw <= 0x7fffffff) {
                A.tps = (int)tps;
                const bool dr16 = (d_rel == 16);
                if (i8) {
                    if (dr16) R4D_KLAUNCH(R4D_KNAME(rel_wmma_kernel<true, true>),
                                          dim3((unsigned)ntw, (unsigned)nh), dim3(kWThreads),
                                          (hipStream_t)stream, A);
                    else      R4D_KLAUNCH(R4D_KNAME(rel_wmma_kernel<true, false>),
                                          dim3((unsigned)ntw, (unsigned)nh), dim3(kWThreads),
                                          (hipStream_t)stream, A);
                } else {
                    if (dr16) R4D_KLAUNCH(R4D_KNAME(rel_wmma_kernel<false, true>),
                                          dim3((unsigned)ntw, (unsigned)nh), dim3(kWThreads),
                                          (hipStream_t)stream, A);
                    else      R4D_KLAUNCH(R4D_KNAME(rel_wmma_kernel<false, false>),
                                          dim3((unsigned)ntw, (unsigned)nh), dim3(kWThreads),
                                          (hipStream_t)stream, A);
                }
                return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
            }
        }

        const bool kv_dec_aligned = i8
            ? (rad_ptr_aligned(kv->data, 4) && ((A.sb | A.sh | A.so) % 4) == 0)
            : (rad_ptr_aligned(kv->data, 8) && ((A.sb | A.sh | A.so) % 4) == 0);
        if (kv_dec_aligned) {
            /* The window is the kernel's own bound. Past that, the runtime's host maximum
             * (kv_used_max_h: the decode pass's keyed context bucket, or the exact prefill span)
             * cuts the model max_ctx. max_ctx alone would launch the full split budget for an 8k
             * decode on a 262144 context, and the wider reduction would not hide a walk. */
            long long span = window > 0 ? window
                                        : rad_args_geti_or(a, "max_ctx", A.max_blocks * 16);
            if (a->kv_used_max_h > 0 && a->kv_used_max_h < span) span = a->kv_used_max_h;
            const long long pairs = rows * kv_heads;
            /* A verify (two or more rows a sequence) whose rows and group heads fit one WMMA
             * block shares each K and V tile across them; one row stays on the decode kernel. Its
             * 16-key tile pays a fixed staging cost the decode kernel does not, so a narrow verify
             * (under 12 columns) takes it only over a short span: on an R9700 at gqa 3, 8 rows
             * run 1.8-2.2x faster from 1k to 131k keys and 4 rows 1.0-1.7x, while 2 rows win
             * 1.7x at 1k keys and lose 0.6x from 32k. */
            const bool packed = wmma_aligned && qmax >= 2 && qmax * gqa <= kWBQ &&
                                (qmax * gqa >= 12 || span <= 8192);
            long long ns = 1;
            if (packed)
                ns = rel_verify_splits(n_seq, rows, kv_heads, gqa, span, a->wgp_count);
            else if (rows <= 32 && pairs > 0 && pairs < 128)
                ns = rel_decode_splits((int)pairs, (int)span, a->wgp_count);
            if (ns > 1 && (!a->scratch || a->scratch_bytes < rel_scratch_bytes() ||
                           !rad_ptr_aligned(a->scratch, 16)))
                ns = 1;

            A.n_split = (int)ns;
            A.part = ns > 1 ? (float*)a->scratch : nullptr;
            g_rel_last_splits = (int)ns;

            const bool chain = rad_args_geti_or(a, "chain", 0) != 0 && rows <= 16;
            hipStream_t hs = (hipStream_t)stream;
            if (packed) {
                const dim3 g((unsigned)n_seq, (unsigned)kv_heads, (unsigned)ns);
                const bool dr16 = d_rel == 16;
                if (i8) {
                    if (dr16) launch_vfy<true, true>(chain, g, hs, A);
                    else      launch_vfy<true, false>(chain, g, hs, A);
                } else {
                    if (dr16) launch_vfy<false, true>(chain, g, hs, A);
                    else      launch_vfy<false, false>(chain, g, hs, A);
                }
            } else if (i8) {
                if (chain)
                    R4D_KLAUNCH(R4D_KNAME(rel_dec_kernel<true, r4dchain>),
                                dim3((unsigned)rows, (unsigned)kv_heads, (unsigned)ns),
                                dim3(kThr), hs, A);
                else
                    R4D_KLAUNCH(R4D_KNAME(rel_dec_kernel<true, R4dUnchained>),
                                dim3((unsigned)rows, (unsigned)kv_heads, (unsigned)ns),
                                dim3(kThr), hs, A);
            } else {
                if (chain)
                    R4D_KLAUNCH(R4D_KNAME(rel_dec_kernel<false, r4dchain>),
                                dim3((unsigned)rows, (unsigned)kv_heads, (unsigned)ns),
                                dim3(kThr), hs, A);
                else
                    R4D_KLAUNCH(R4D_KNAME(rel_dec_kernel<false, R4dUnchained>),
                                dim3((unsigned)rows, (unsigned)kv_heads, (unsigned)ns),
                                dim3(kThr), hs, A);
            }
            if (ns > 1) {
                if (chain)
                    R4D_KLAUNCH(R4D_KNAME(rel_merge_kernel<r4dchain>),
                                dim3((unsigned)rows, (unsigned)kv_heads), dim3(kThr), hs, A);
                else
                    R4D_KLAUNCH(R4D_KNAME(rel_merge_kernel<R4dUnchained>),
                                dim3((unsigned)rows, (unsigned)kv_heads), dim3(kThr), hs, A);
            }
            return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
        }
    }

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

int64_t r4d_rad_attn_rel_scratch(const RadArgs*) {
    return r4d_rel::rel_scratch_bytes();
}

int r4d_attn_rel_last_splits(void) {
    return r4d_rel::g_rel_last_splits;
}

}  /* extern "C" */
