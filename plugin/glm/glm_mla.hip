/* plugin/glm/glm_mla.hip -- mla_attn over an int8 latent cache (kv_dtype i8), on the card.
 *
 * GLM-5.3-Flash's absorbed MLA keeps one 512-wide latent a token. The int8 cache is the ordinary
 * paged i8 row at head_dim 256 (abi/rad_builder.h): 256 codes of the first half, 256 of the second,
 * then the halves' f16 scales, 516 bytes a token. Both halves are key and value: a token's latent
 * is z[c] = code[c] * scale[c / 256], and score = scale_attn * q.z, out = sum p z / sum p over the
 * unscaled probabilities (docs/OPS.md, mla_attn; libref/ref_mla.cpp reads it the same way).
 *
 * The wave form is libr4d's `attend`/`merge` (libr4d/r4d_mla.hip) with the load replaced: a wave a
 * (row, head), lane l holding latent columns 16l .. 16l + 15, so lanes 0-15 take the first half's
 * scale and 16-31 the second's. A key is decoded to f32 once and that vector feeds both the dot and
 * the value sum. A cache row is only 4-byte aligned (516 bytes), so the codes are four dword loads;
 * the scales are read as halves at bytes 512 and 514. The history splits over waves as there and
 * the f32 partials merge the same way.
 *
 * The matrix form is libr4d's unpaired `attend_matrix` (sixteen heads of one row the WMMA tile's
 * rows, a 16-token chunk its columns), which the head form serves at four ranks and fewer. Its key
 * tile holds the raw codes, each exactly a bf16 (|code| <= 127), and the scales stay outside the
 * products: the scores are two dot products, columns 0-255 and 256-511, each times its half's scale
 * before they add; the probabilities stay unscaled for the maximum and the denominator, and the
 * value product takes p * scale as its high/low bf16 operand, half 0's for waves 0-1 (columns
 * 0-255) and half 1's for waves 2-3. So no dequantised key is ever rounded to bf16. Eight heads
 * (eight ranks), whose tile pairs two rows, take the wave form.
 */
#include "glm.h"

#include <hip/hip_fp16.h>

#include "../../libr4d/r4d_common.h"
#include "../../libr4d/r4d_emit.h"
#include "../../libr4d/r4d_gdn_wmma.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>

namespace {

constexpr int D = 512, PART = D + 2, ROW = D + 4;
constexpr int64_t kScratch = 512ll * PART * 4;
/* The matrix form's partials, as libr4d's matrix_scratch_bytes. */
constexpr int64_t kMatrixScratch = 2048ll * PART * 4;

struct Meta {
    long qr, qh, kp, kt, bp, bc, lp, orr, oh;
    int rows, heads, blocks, bs, pages, splits;
    float scale;
};

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

/* Sixteen latent columns of a query row (bf16), as libr4d's load_latent. */
__device__ __forceinline__ void load_query(const uint16_t* p, float (&v)[16]) {
#pragma unroll
    for (int j = 0; j < 16; ++j) v[j] = bf16_to_f32(p[j]);
}

/* Columns 16 * lane .. + 15 of a cache row: four dword loads of signed codes, times the lane's
 * half's f16 scale. `row` is the token's row base, dword aligned. */
__device__ __forceinline__ void load_key(const int8_t* row, int lane, float (&v)[16]) {
    const uint32_t* w = (const uint32_t*)(row + lane * 16);
    const uint16_t sb = *(const uint16_t*)(row + D + 2 * (lane >> 4));
    const float s = h2f(sb);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t x = w[k];
#pragma unroll
        for (int b = 0; b < 4; ++b) v[4 * k + b] = (float)(int8_t)(x >> (8 * b)) * s;
    }
}

__global__ __launch_bounds__(128) void attend_i8(const uint16_t* q, const int8_t* kv, const int* bt,
                                                 const int* len, uint16_t* out, float* scratch,
                                                 Meta m) {
    const int lane = threadIdx.x & 31, head = blockIdx.y * 4 + (threadIdx.x >> 5), r = blockIdx.x,
              split = blockIdx.z;
    if (head >= m.heads) return;
    const int live = min(max(len[(long)r * m.lp], 0), m.blocks * m.bs);
    const int begin = (int)((long)live * split / m.splits),
              end = (int)((long)live * (split + 1) / m.splits);
    float query[16], value[16] = {};
    load_query(q + (long)r * m.qr + (long)head * m.qh + lane * 16, query);
    float max_score = -INFINITY, den = 0;
    for (int token = begin; token < end; ++token) {
        const int page = bt[(long)r * m.bp + (token / m.bs) * m.bc];
        if (page < 0 || page >= m.pages) continue;
        float key[16];
        load_key(kv + (long)page * m.kp + (long)(token % m.bs) * m.kt, lane, key);
        float score = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) score += query[j] * key[j];
        score = r4d_wave_sum(score) * m.scale;
        const bool higher = score > max_score;
        const float weight = expf(higher ? max_score - score : score - max_score);
        if (higher) {
#pragma unroll
            for (int j = 0; j < 16; ++j) value[j] = value[j] * weight + key[j];
            den = den * weight + 1;
            max_score = score;
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j) value[j] += key[j] * weight;
            den += weight;
        }
    }
    if (m.splits == 1) {
#pragma unroll
        for (int j = 0; j < 16; ++j)
            out[(long)r * m.orr + (long)head * m.oh + lane * 16 + j] =
                f32_to_bf16(den > 0 ? value[j] / den : 0);
    } else {
        float* dst = scratch + (((long)r * m.heads + head) * m.splits + split) * PART;
#pragma unroll
        for (int j = 0; j < 16; ++j) dst[lane * 16 + j] = value[j];
        if (!lane) {
            dst[D] = max_score;
            dst[D + 1] = den;
        }
    }
}

/* Eight consecutive codes of a cache row as four bf16 pairs, exactly. `p` is dword aligned. */
__device__ __forceinline__ uint4 codes8(const int8_t* p) {
    const uint32_t a = ((const uint32_t*)p)[0], b = ((const uint32_t*)p)[1];
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t x = i < 2 ? a : b;
        const int sh = 16 * (i & 1);
        const float lo = (float)(int8_t)(x >> sh), hi = (float)(int8_t)(x >> (sh + 8));
        w[i] = (uint32_t)f32_to_bf16(lo) | (uint32_t)f32_to_bf16(hi) << 16;
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

__global__ __launch_bounds__(128) void attend_matrix_i8(const uint16_t* q, const int8_t* kv,
                                                        const int* bt, const int* len,
                                                        uint16_t* out, float* scratch, Meta m) {
    __shared__ __align__(16) uint16_t query[16 * 520], keys[16 * 520], prob_hi[2][16 * 24],
        prob_lo[2][16 * 24];
    __shared__ float score[16 * 16], maximum[16], denominator[16], alpha[16], kscale[16][2];
    __shared__ long source[16];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int r = blockIdx.x, h0 = blockIdx.y * 16, split = blockIdx.z;
    const int live = min(max(len[(long)r * m.lp], 0), m.blocks * m.bs);
    for (int j = tid; j < 16 * 512; j += 128)
        query[(j / 512) * 520 + j % 512] = q[(long)r * m.qr + (long)(h0 + j / 512) * m.qh + j % 512];
    if (tid < 16) {
        maximum[tid] = -INFINITY;
        denominator[tid] = 0;
    }
    v8f value[8];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) value[nt] = (v8f){ 0 };
    const int begin = (int)((long)live * split / m.splits),
              end = (int)((long)live * (split + 1) / m.splits);
    __syncthreads();
    for (int start = begin; start < end; start += 16) {
        if (tid < 16) {
            const int token = start + tid;
            const int page = token < end ? bt[(long)r * m.bp + (token / m.bs) * m.bc] : -1;
            const long at =
                page >= 0 && page < m.pages ? (long)page * m.kp + (long)(token % m.bs) * m.kt : -1;
            source[tid] = at;
            kscale[tid][0] = at >= 0 ? h2f(*(const uint16_t*)(kv + at + D)) : 0.0f;
            kscale[tid][1] = at >= 0 ? h2f(*(const uint16_t*)(kv + at + D + 2)) : 0.0f;
        }
        __syncthreads();
        {
            /* Unconditional loads (an absent row reads row 0 and is zeroed after), all eight in
             * flight before the first wait. */
            uint4 part[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int flat = tid + j * 128;
                const long at = source[flat >> 6];
                part[j] = codes8(kv + (at >= 0 ? at : 0) + (flat & 63) * 8);
                if (at < 0) part[j] = make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int flat = tid + j * 128;
                *(uint4*)(keys + (flat >> 6) * 520 + (flat & 63) * 8) = part[j];
            }
        }
        __syncthreads();
        if (!wave) {
            v8f dot0 = { 0 }, dot1 = { 0 };
#pragma unroll
            for (int d = 0; d < 256; d += 16)
                dot0 = mma(frag(query, 520, 0, d, lane), frag(keys, 520, 0, d, lane), dot0);
#pragma unroll
            for (int d = 256; d < 512; d += 16)
                dot1 = mma(frag(query, 520, 0, d, lane), frag(keys, 520, 0, d, lane), dot1);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int head = (lane >> 4) * 8 + j, col = lane & 15;
                score[head * 16 + col] =
                    source[col] >= 0
                        ? (dot0[j] * kscale[col][0] + dot1[j] * kscale[col][1]) * m.scale
                        : -INFINITY;
            }
        }
        __syncthreads();
        if (tid < 16) {
            float mx = maximum[tid];
            for (int j = 0; j < 16; ++j) mx = fmaxf(mx, score[tid * 16 + j]);
            const float a = isfinite(mx) ? expf(maximum[tid] - mx) : 1;
            float den = denominator[tid] * a;
            for (int j = 0; j < 16; ++j) {
                const float p = isfinite(score[tid * 16 + j]) ? expf(score[tid * 16 + j] - mx) : 0;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float x = p * kscale[j][h];
                    const uint16_t hi = f32_to_bf16(x);
                    prob_hi[h][tid * 24 + j] = hi;
                    prob_lo[h][tid * 24 + j] = f32_to_bf16(x - bf16_to_f32(hi));
                }
                den += p;
            }
            maximum[tid] = mx;
            denominator[tid] = den;
            alpha[tid] = a;
        }
        /* Read before the in-place transpose overwrites keys. Query remains resident. */
        uint16_t held[64];
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            const int flat = tid + j * 128;
            held[j] = keys[(flat / 512) * 520 + flat % 512];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            const int flat = tid + j * 128;
            keys[(flat % 512) * 16 + flat / 512] = held[j];
        }
        __syncthreads();
        const int half = wave >> 1;
        const auto hi = frag(prob_hi[half], 24, 0, 0, lane), lo = frag(prob_lo[half], 24, 0, 0, lane);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
            for (int j = 0; j < 8; ++j) value[nt][j] *= alpha[(lane >> 4) * 8 + j];
            const auto v = frag(keys, 16, wave * 128 + nt * 16, 0, lane);
            value[nt] = mma(hi, v, value[nt]);
            value[nt] = mma(lo, v, value[nt]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int col = wave * 128 + nt * 16 + (lane & 15);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = (lane >> 4) * 8 + j, head = h0 + row;
            if (m.splits == 1) {
                const float den = denominator[row];
                out[(long)r * m.orr + (long)head * m.oh + col] =
                    f32_to_bf16(den > 0 ? value[nt][j] / den : 0);
            } else
                scratch[(((long)r * m.heads + head) * m.splits + split) * PART + col] =
                    value[nt][j];
        }
    }
    if (m.splits > 1 && tid < 16) {
        float* dst = scratch + (((long)r * m.heads + h0 + tid) * m.splits + split) * PART;
        dst[D] = maximum[tid];
        dst[D + 1] = denominator[tid];
    }
}

/* libr4d's merge, unchanged: the splits' f32 partials under one max. */
__global__ __launch_bounds__(128) void merge(const float* scratch, uint16_t* out, Meta m) {
    const int lane = threadIdx.x & 31, head = blockIdx.y * 4 + (threadIdx.x >> 5), r = blockIdx.x;
    if (head >= m.heads) return;
    const float* src = scratch + ((long)r * m.heads + head) * m.splits * PART;
    float max_score = -INFINITY;
    for (int s = 0; s < m.splits; ++s)
        if (src[s * PART + D + 1] > 0) max_score = fmaxf(max_score, src[s * PART + D]);
    float value[16] = {}, den = 0;
    for (int s = 0; s < m.splits; ++s) {
        const float part_den = src[s * PART + D + 1];
        if (!(part_den > 0)) continue;
        const float w = expf(src[s * PART + D] - max_score);
        den += part_den * w;
#pragma unroll
        for (int j = 0; j < 16; ++j) value[j] += src[s * PART + lane * 16 + j] * w;
    }
#pragma unroll
    for (int j = 0; j < 16; ++j)
        out[(long)r * m.orr + (long)head * m.oh + lane * 16 + j] =
            f32_to_bf16(den > 0 ? value[j] / den : 0);
}

}  /* namespace */

extern "C" int64_t rad_glm_mla_i8_scratch(const RadArgs*) { return kMatrixScratch; }

/* mla_attn's operands: q [M, heads, 512] bf16, kv [pages, 1, block_size, 516] i8, block table
 * [M, blocks] i32, lengths [M] i32, out [M, heads, 512] bf16. */
extern "C" int rad_glm_mla_i8_launch(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const auto &q = a->t[0], &kv = a->t[1], &bt = a->t[2], &len = a->t[3], &o = a->t[4];
    if (q.dtype != RAD_BF16 || kv.dtype != RAD_I8 || o.dtype != RAD_BF16 || bt.dtype != RAD_I32 ||
        len.dtype != RAD_I32)
        return RAD_E_DTYPE;
    for (int i = 0; i < 5; ++i)
        if (!a->t[i].data) return RAD_E_INVAL;
    const long heads = rad_args_geti_or(a, "heads", 0), bs = rad_args_geti_or(a, "block_size", 0);
    const float scale = (float)rad_args_getf_or(a, "scale", 0);
    if (rad_args_geti_or(a, "latent_dim", 0) != D || heads < 1 || heads > 64 || bs < 1 ||
        bs > 256 || !std::isfinite(scale) || scale <= 0 || q.rank != 3 || q.shape[0] < 1 ||
        q.shape[0] > 65535 || q.shape[1] != heads || q.shape[2] != D || q.stride[2] != 1 ||
        q.stride[1] < D || q.stride[0] < heads * q.stride[1] || kv.rank != 4 || kv.shape[0] < 1 ||
        kv.shape[0] > INT_MAX || kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] != ROW ||
        kv.stride[3] != 1 || kv.stride[2] < ROW || kv.stride[0] < bs * kv.stride[2] ||
        bt.rank != 2 || bt.shape[0] != q.shape[0] || bt.shape[1] < 1 ||
        bt.shape[1] > INT_MAX / bs || bt.stride[1] < 1 ||
        bt.stride[0] < bt.shape[1] * bt.stride[1] || len.rank != 1 || len.shape[0] != q.shape[0] ||
        len.stride[0] < 1 || o.rank != 3 || o.shape[0] != q.shape[0] || o.shape[1] != heads ||
        o.shape[2] != D || o.stride[2] != 1 || o.stride[1] < D || o.stride[0] < heads * o.stride[1])
        return RAD_E_SHAPE;
    /* The codes are dword loads and the scales halfword loads: every row base dword aligned. */
    if (((uintptr_t)kv.data & 3) || (kv.stride[2] & 3) || (kv.stride[0] & 3)) return RAD_E_STRIDE;
    /* Sixteen heads a tile: the matrix form, as libr4d's unpaired attend_matrix splits. Otherwise
     * (eight heads, or fewer than eight) the wave form. */
    const bool matrix = heads % 16 == 0 && std::getenv("RADIANCE_GLM_MLA_WAVE") == nullptr;
    int splits;
    if (matrix) {
        const long jobs = q.shape[0] * heads / 16;
        splits = (int)std::min(64l, (64 + jobs - 1) / jobs);
        splits = (int)std::max(1l, std::min((long)splits, (bt.shape[1] * bs + 15) / 16));
        if (!a->scratch || a->scratch_bytes < kMatrixScratch) splits = 1;
    } else {
        const long jobs = q.shape[0] * heads;
        splits = (int)std::min(16l, (256 + jobs - 1) / jobs);
        splits = (int)std::max(1l, std::min((long)splits, (bt.shape[1] * bs + 63) / 64));
        if (!a->scratch || a->scratch_bytes < kScratch) splits = 1;
    }
    Meta m{ q.stride[0],  q.stride[1],     kv.stride[0],  kv.stride[2],
            bt.stride[0], bt.stride[1],    len.stride[0], o.stride[0],
            o.stride[1],  (int)q.shape[0], (int)heads,    (int)bt.shape[1],
            (int)bs,      (int)kv.shape[0], splits,       scale };
    int rc = matrix
                 ? R4D_EMIT(stream, attend_matrix_i8, dim3(m.rows, m.heads / 16, m.splits),
                            dim3(128), 0, (const uint16_t*)q.data, (const int8_t*)kv.data,
                            (const int*)bt.data, (const int*)len.data, (uint16_t*)o.data,
                            (float*)a->scratch, m)
                 : R4D_EMIT(stream, attend_i8, dim3(m.rows, (m.heads + 3) / 4, m.splits),
                            dim3(128), 0, (const uint16_t*)q.data, (const int8_t*)kv.data,
                            (const int*)bt.data, (const int*)len.data, (uint16_t*)o.data,
                            (float*)a->scratch, m);
    if (rc == RAD_OK && splits > 1)
        rc = R4D_EMIT(stream, merge, dim3(m.rows, (m.heads + 3) / 4), dim3(128), 0,
                      (const float*)a->scratch, (uint16_t*)o.data, m);
    return rc;
}
