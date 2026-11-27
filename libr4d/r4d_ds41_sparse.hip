// R4D -- DeepSeek-V4.1's sparse ops. Each kernel rounds where its libref function does
// (libref/ref_sparse_kv.cpp) and sums in libref's order, so the outputs are libref's bits:
//
//   index_score       one thread an entry (or HS waves an entry group, each a slice of the heads,
//                     the head sum then closed in head order from LDS): the dot in d order,
//                     rounded to bf16, relu x the rounded head weight, rounded, the head sum in head
//                     order, rounded.
//   index_candidates  one 1024-thread block a row: block maxima, the newest block pinned, an
//   index_topk        exact radix select of the k largest (value desc, position asc: libref's
//                     partial_sort order), then an ordered compaction. index_topk at decode
//                     heights over long rows splits a row into chunks (each keeps its exact top-k,
//                     which holds every global winner) and merges them in a second launch.
//   attn_sparse_sink  one workgroup a (d half, head group, row): every score q.k in d order (keys
//                     transposed through a wave-private LDS tile so the loads stay coalesced), the
//                     max, p = exp(s - max) by glibc's expf algorithm (the same double table and
//                     polynomial, so p is std::exp's float), sum p and sum p k in key order, then
//                     exp(sink - max), the division and one bf16 rounding.
//
// Block-scaled E4M3 caches (attn_sparse_sink latent_format e4m3_e4m3_16, index_score key_format
// e4m3_ue8m0_32; rad_kv_fp8.h) are widened as they are read to the very bf16 words a bf16 cache
// holds for an exact entry (code x scale is exact), and everything after the read is the bf16
// form's: such a cache gives the bf16 cache's bits in every form, the WMMA ones included.
//
// q.k and q_h.k_j multiply two bf16 values, exact in f32, so a fused multiply-add rounds exactly
// where libref's multiply-then-add does; p k is not exact and stays an unfused multiply and add.
//
// Prefill heights (more than kWmRows = 48 rows a pass) take WMMA forms instead, which are NOT
// libref's bits (kbench tolerance): attn_sparse_sink with an online softmax, v_exp_f32 and P as
// bf16 hi + lo; index_score with only the dot's summation order changed (every rounding after it
// is libref's). The decode band keeps the exact kernels, so speculative verify reproduces decode.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_emit.h"
#include "r4d_plugin.h"
#include "rad_kv_fp8.h"

#include <climits>
#include <cmath>
#include <cstdint>

extern __shared__ __align__(16) unsigned char r4d_ds41_sparse_dyn[];

namespace {

constexpr float kDsInf = __builtin_huge_valf();

/* ================================================================== scalar helpers */

/* rad_f32_to_bf16 (abi/rad_plugin.h): NaN quieted, everything else to nearest even. */
__device__ __forceinline__ uint32_t ds_bf16_bits(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return ((u >> 16) | 0x0040u) & 0xffffu;
    return ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16) & 0xffffu;
}
__device__ __forceinline__ float ds_round_bf16(float f) { return __uint_as_float(ds_bf16_bits(f) << 16); }
__device__ __forceinline__ float ds_bf(uint32_t h) { return __uint_as_float((h & 0xffffu) << 16); }

/* libref's seq_of_row: the s with cu[s] <= row < cu[s+1], by bisection. */
__device__ __forceinline__ int ds_seq_of_row(const int* cu, long cs, int S, long row) {
    int lo = 0, hi = S;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if ((long)cu[(long)mid * cs] <= row) lo = mid;
        else hi = mid;
    }
    return lo;
}

/* glibc's expf (sysdeps/ieee754/flt-32/e_expf.c, EXP2F_TABLE_BITS 5): 2^(k/32) from a table and a
 * cubic for 2^(r/32), in double, rounded once to float. A host replica of exactly these steps
 * matched libm's expf on 685k arguments (subnormal and overflow edges included). */
__constant__ unsigned long long kDsExp2fTab[32] = {
    0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull,
    0x3fef72b83c7d517bull, 0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull,
    0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
    0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull, 0x3feea47eb03a5585ull,
    0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull,
    0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
    0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull,
    0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full, 0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
};
__device__ __forceinline__ float ds_expf(float x) {
    const uint32_t ux = __float_as_uint(x);
    const uint32_t abstop = (ux >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {                          /* |x| >= 88, or inf/nan */
        if (ux == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8u) return x + x;
        if (x > 0x1.62e42ep6f) return kDsInf;
        if (x < -0x1.9fe368p6f) return 0.0f;
    }
    const double xd = (double)x;
    double z = 0x1.71547652b82fep+5 * xd;            /* x * 32 / ln 2 */
    double kd = z + 0x1.8p+52;
    const unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd -= 0x1.8p+52;
    const double r = z - kd;
    unsigned long long t = kDsExp2fTab[ki % 32];
    t += ki << 47;
    const double s = __longlong_as_double((long long)t);
    z = 0x1.c6af84b912394p-20 * r + 0x1.ebfce50fac4f3p-13;
    const double r2 = r * r;
    double y = 0x1.62e42ff0c52d6p-6 * r + 1.0;
    y = z * r2 + y;
    y = y * s;
    return (float)y;
}

/* An order-preserving image of a float with -0 folded onto +0 (libref's comparator calls them
 * equal). A candidate is > -inf, so its key is > 0x007fffff; 0 marks "not a candidate". */
__device__ __forceinline__ uint32_t ds_key(float v) {
    const uint32_t u = __float_as_uint(v + 0.0f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ void ds_wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

/* ================================================================== block-scaled E4M3 entries
 *
 * rad_kv_fp8.h's entries: n E4M3 codes, then a scale byte a block (the latent: E4M3 a block of 16,
 * attn_sparse_sink latent_format e4m3_e4m3_16; the index keys: a power of two a block of 32,
 * index_score key_format e4m3_ue8m0_32), zeros to 16 bytes. A value is code x scale in f32,
 * exact, and its bf16 is the product's high half, so a widened entry is bit for bit the bf16
 * cache's row and every kernel below computes on the very words it reads from a bf16 cache. */
constexpr int kDsLatScales = 512;                    /* the latent entry's scale bytes start here */
constexpr int kDsIdxScales = 128;                    /* the index entry's */

/* Two E4M3 codes (bytes 0,1 or, HI, 2,3 of c) to f32 (v_cvt_pk_f32_fp8, exact). */
template <bool HI>
__device__ __forceinline__ float2 ds_e4m3x2(uint32_t c) {
#if __has_builtin(__builtin_amdgcn_cvt_pk_f32_fp8)
    const auto v = __builtin_amdgcn_cvt_pk_f32_fp8((int)c, HI);
    return make_float2(v[0], v[1]);
#else
    const uint32_t s = HI ? 16u : 0u;
    return make_float2(rad_kvf_e4m3((uint8_t)(c >> s)), rad_kvf_e4m3((uint8_t)(c >> (s + 8u))));
#endif
}
__device__ __forceinline__ float ds_e4m3s(uint32_t b) {      /* an E4M3 scale byte (low byte of b) */
#if __has_builtin(__builtin_amdgcn_cvt_f32_fp8)
    return __builtin_amdgcn_cvt_f32_fp8((int)(b & 0xffu), 0);
#else
    return rad_kvf_e4m3((uint8_t)b);
#endif
}
__device__ __forceinline__ float ds_ue8m0s(uint32_t b) { return __uint_as_float((b & 0xffu) << 23); }

/* Two codes x sc as a bf16 pair word (lo = the first). The products are exact; their low halves
 * are zero, so the high halves are the bf16 values. */
template <bool HI>
__device__ __forceinline__ uint32_t ds_f8w(uint32_t c, float sc) {
    const float2 v = ds_e4m3x2<HI>(c);
    return __builtin_amdgcn_perm(__float_as_uint(v.y * sc), __float_as_uint(v.x * sc), 0x07060302u);
}
/* Eight codes (c0: values 0-3, c1: 4-7) x sc as the 16 bytes a bf16 row holds for them. */
__device__ __forceinline__ uint4 ds_f8x8(uint32_t c0, uint32_t c1, float sc) {
    return make_uint4(ds_f8w<false>(c0, sc), ds_f8w<true>(c0, sc), ds_f8w<false>(c1, sc), ds_f8w<true>(c1, sc));
}

/* A latent key's 16-byte piece pc (columns 8 pc .. 8 pc + 7) as bf16 words: the row's own bytes,
 * or (f8, an E4M3 entry at p) widened. With a branch: for the forms that may load under one. */
template <bool LF8>
__device__ __forceinline__ uint4 ds_kpiece(const uint16_t* p, int pc, bool f8) {
    if (!LF8 || !f8) return *(const uint4*)(p + pc * 8);
    const unsigned char* b = (const unsigned char*)p;
    const uint2 c = *(const uint2*)(b + pc * 8);
    return ds_f8x8(c.x, c.y, ds_e4m3s(b[kDsLatScales + (pc >> 1)]));
}
/* A latent key's DPT (1 or 2) columns at dcol (even when 2) as a word, the first in the low half. */
template <int DPT, bool LF8>
__device__ __forceinline__ uint32_t ds_kcols(const uint16_t* p, int dcol, bool f8) {
    if (!LF8 || !f8) return DPT == 2 ? *(const uint32_t*)(p + dcol) : (uint32_t)p[dcol];
    const unsigned char* b = (const unsigned char*)p;
    const float sc = ds_e4m3s(b[kDsLatScales + dcol / 16]);
    const uint32_t c = DPT == 2 ? (uint32_t)*(const uint16_t*)(b + dcol) : (uint32_t)b[dcol];
    return ds_f8w<false>(c, sc) & (DPT == 2 ? 0xffffffffu : 0xffffu);
}

/* Global (not flat) loads at a byte offset. */
typedef unsigned int ds_gv4u __attribute__((ext_vector_type(4)));
__device__ __forceinline__ uint4 ds_gld16(const void* p, long off) {
    const ds_gv4u v = *(__attribute__((address_space(1))) const ds_gv4u*)((const unsigned char*)p + off);
    return make_uint4(v.x, v.y, v.z, v.w);
}
__device__ __forceinline__ uint32_t ds_gld32(const void* p, long off) {
    return *(__attribute__((address_space(1))) const uint32_t*)((const unsigned char*)p + off);
}
__device__ __forceinline__ uint32_t ds_gld8(const void* p, long off) {
    return *(__attribute__((address_space(1))) const unsigned char*)((const unsigned char*)p + off);
}

/* Branch-free dual loads (the decode band and WMMA forms keep no load under a branch): the raw
 * bytes of pieces c .. c+7 (c a multiple of 8) of a key, a bf16 row's 8 x 16 bytes or (f8) an
 * E4M3 entry's 64 code bytes (r[0..3]; r[4..7] reread them) and its 4 scale bytes in s. Every
 * lane loads the same shapes; ds_dual_cvt8 then picks per lane. */
__device__ __forceinline__ void ds_dual_ld8(uint4 (&r)[8], uint32_t& s, const uint16_t* p, int c, bool f8) {
#pragma unroll
    for (int i = 0; i < 8; ++i) r[i] = ds_gld16(p, f8 ? c * 8 + (i & 3) * 16 : (c + i) * 16);
    s = ds_gld32(p, f8 ? kDsLatScales + c / 2 : 0);
}
__device__ __forceinline__ void ds_dual_cvt8(uint4 (&o)[8], const uint4 (&r)[8], uint32_t s, bool f8) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint4 w = r[i >> 1];
        const uint4 x = ds_f8x8((i & 1) ? w.z : w.x, (i & 1) ? w.w : w.y, ds_e4m3s(s >> (8 * (i >> 1))));
        o[i] = f8 ? x : r[i];
    }
}

/* ================================================================== index_score */
/* q, w and pos start at the run's first row (row_part); row0 is that row's index in cu_seqlens.
 * M is the score's rows (the pass height, which picks the form), Mv how many of them the step
 * holds: a row from Mv on is past the step and scores -inf without a read. */
struct DsScoreArgs {
    const uint16_t* q;
    const uint16_t* w;
    const uint16_t* kc;
    const int* bt;
    const int* pos;
    const int* cu;
    float* sc;
    long qr, wr, wc, btr, btc, nblk, ps, cs, sr, scs, row0;
    int nbt, S, M, Mv, C, H, ratio, cbs, slot;     /* slot: cache elements (KF8: bytes) a slot */
    float wscale;
};

/* Entry j's key at block blk: bf16 elements, or (KF8) the E4M3 entry's bytes. */
template <bool KF8>
__device__ __forceinline__ const uint16_t* ds_iptr(const DsScoreArgs& a, long blk, long j) {
    const long off = (blk * a.cbs + (j * a.ratio) % a.cbs) * (long)a.slot;
    return KF8 ? (const uint16_t*)((const unsigned char*)a.kc + off) : a.kc + off;
}

/* One thread an entry; HS waves of a block share 32 entries, each wave a slice of the heads (its
 * q then uniform, read through scalar loads). A thread holds its key as 128 floats. */
template <int HS, bool KF8>
__global__ __launch_bounds__(256) void ds41_index_score_kernel(DsScoreArgs a) {
    constexpr int DM = 128, EPB = 256 / HS;
    float* part = (float*)r4d_ds41_sparse_dyn;       /* [H][EPB], HS > 1 */
    /* The wave index through readfirstlane, so a wave's head slice (and its q) is provably uniform
     * and q stays in scalar loads. */
    const int lane = threadIdx.x & 31, wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    const int grp = wave / HS, sub = wave % HS;
    const int e = grp * 32 + lane;
    const long jb = (long)blockIdx.x * EPB, j = jb + e;
    const int hp = (a.H + HS - 1) / HS, h0 = sub * hp, h1 = min(a.H, h0 + hp);
    for (long m = blockIdx.y; m < a.M; m += gridDim.y) {
        float* srow = a.sc + m * a.sr;
        const long reach = m < a.Mv ? ((long)a.pos[m * a.ps] + 1) / a.ratio : 0;
        if (jb >= reach) {                           /* block-uniform: no key in this block */
            if (sub == 0 && j < a.C) srow[j * a.scs] = -kDsInf;
            continue;
        }
        const int s = ds_seq_of_row(a.cu, a.cs, a.S, a.row0 + m);
        const uint16_t* kp = nullptr;
        if (j < a.C && j < reach) {
            const long col = (j * a.ratio) / a.cbs;
            if (col < a.nbt) {
                const long blk = a.bt[(long)s * a.btr + col * a.btc];
                if (blk >= 0 && blk < a.nblk)
                    kp = ds_iptr<KF8>(a, blk, j);
            }
        }
        float acc = 0.0f;
        if (kp) {
            float k[DM];
            if constexpr (KF8) {
                /* 128 codes, then a power-of-two scale a block of 32: code x scale, the bf16 value */
                const unsigned char* b = (const unsigned char*)kp;
                const uint32_t s4 = *(const uint32_t*)(b + kDsIdxScales);
#pragma unroll
                for (int c = 0; c < DM / 16; ++c) {
                    const uint4 v = *(const uint4*)(b + c * 16);
                    const float sc = ds_ue8m0s(s4 >> (8 * (c >> 1)));
                    const uint32_t cw[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float2 lo = ds_e4m3x2<false>(cw[e]), hi = ds_e4m3x2<true>(cw[e]);
                        k[c * 16 + e * 4 + 0] = lo.x * sc; k[c * 16 + e * 4 + 1] = lo.y * sc;
                        k[c * 16 + e * 4 + 2] = hi.x * sc; k[c * 16 + e * 4 + 3] = hi.y * sc;
                    }
                }
            } else {
#pragma unroll
                for (int c = 0; c < DM / 8; ++c) {
                    const uint4 v = *(const uint4*)(kp + c * 8);
                    k[c * 8 + 0] = bf16w_lo(v.x); k[c * 8 + 1] = bf16w_hi(v.x);
                    k[c * 8 + 2] = bf16w_lo(v.y); k[c * 8 + 3] = bf16w_hi(v.y);
                    k[c * 8 + 4] = bf16w_lo(v.z); k[c * 8 + 5] = bf16w_hi(v.z);
                    k[c * 8 + 6] = bf16w_lo(v.w); k[c * 8 + 7] = bf16w_hi(v.w);
                }
            }
            const uint32_t* qrow = (const uint32_t*)(a.q + m * a.qr);
            const uint16_t* wrow = a.w + m * a.wr;
            for (int h = h0; h < h1; h += 4) {
                const int nh = min(4, h1 - h);
                const uint32_t* q0 = qrow + (long)h * (DM / 2);
                const uint32_t* q1 = qrow + (long)(nh > 1 ? h + 1 : h) * (DM / 2);
                const uint32_t* q2 = qrow + (long)(nh > 2 ? h + 2 : h) * (DM / 2);
                const uint32_t* q3 = qrow + (long)(nh > 3 ? h + 3 : h) * (DM / 2);
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
#pragma unroll
                for (int i = 0; i < DM / 2; ++i) {
                    const uint32_t w0 = q0[i], w1 = q1[i], w2 = q2[i], w3 = q3[i];
                    d0 = __builtin_fmaf(bf16w_lo(w0), k[2 * i], d0);
                    d1 = __builtin_fmaf(bf16w_lo(w1), k[2 * i], d1);
                    d2 = __builtin_fmaf(bf16w_lo(w2), k[2 * i], d2);
                    d3 = __builtin_fmaf(bf16w_lo(w3), k[2 * i], d3);
                    d0 = __builtin_fmaf(bf16w_hi(w0), k[2 * i + 1], d0);
                    d1 = __builtin_fmaf(bf16w_hi(w1), k[2 * i + 1], d1);
                    d2 = __builtin_fmaf(bf16w_hi(w2), k[2 * i + 1], d2);
                    d3 = __builtin_fmaf(bf16w_hi(w3), k[2 * i + 1], d3);
                }
                const float dd[4] = { d0, d1, d2, d3 };
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    if (u >= nh) break;
                    const int hh = h + u;
                    const float dot = ds_round_bf16(dd[u]);
                    const float wh = ds_round_bf16(ds_bf(wrow[(long)hh * a.wc]) * a.wscale);
                    const float c = ds_round_bf16((dot < 0.0f ? 0.0f : dot) * wh);
                    if (HS == 1) acc = acc + c;
                    else part[hh * EPB + e] = c;
                }
            }
        }
        if (HS > 1) {
            __syncthreads();
            if (sub == 0 && kp) {
                acc = 0.0f;
                for (int hh = 0; hh < a.H; ++hh) acc = acc + part[hh * EPB + e];
            }
            __syncthreads();
        }
        if (sub == 0 && j < a.C) srow[j * a.scs] = kp ? ds_round_bf16(acc) : -kDsInf;
    }
}

/* ================================================================== exact top-k by radix select
 *
 * One block of 1024 threads selects, among n keys (0 = not a candidate), the K largest under (key
 * desc, index asc). Three histogram passes (11, 11, 10 bits) find the K-th key T and how many keys
 * equal to T are taken (`need`, the lowest-indexed ones); an emit pass then walks the keys in index
 * order with a block scan, so each taken key knows its rank. A thread folds runs of one digit into
 * a single LDS add, which keeps the first pass (most keys share their top bits) off the atomic
 * unit. Up to kDsCache keys stay in LDS after the first pass. */
constexpr int kDsSelThreads = 1024;
constexpr int kDsCache = 8192;
constexpr int kDsCacheUse = kDsCache;               /* rows up to this many keys run from LDS */

struct DsSelShared {
    uint32_t hist[2048];
    uint32_t ws[33];
    uint32_t prefix, need, total, all, digit;
};

__device__ __forceinline__ uint32_t ds_block_excl_scan(uint32_t v, uint32_t* ws, uint32_t* total) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const uint32_t inc = r4d_wave_prefix_u32(v, lane);
    if (lane == 31) ws[wave] = inc;
    __syncthreads();
    if (wave == 0) {
        const uint32_t w = ws[lane];
        const uint32_t wi = r4d_wave_prefix_u32(w, lane);
        ws[lane] = wi - w;
        if (lane == 31) ws[32] = wi;
    }
    __syncthreads();
    const uint32_t r = ws[wave] + inc - v;
    *total = ws[32];
    __syncthreads();
    return r;
}

/* The digit (from the top) whose cumulative count reaches `need`; updates need and total. */
__device__ __forceinline__ void ds_find_digit(DsSelShared* s, int nbins) {
    const int t = threadIdx.x;
    const uint32_t need = s->need;
    const int r0 = 2 * t, r1 = 2 * t + 1;
    const uint32_t c0 = r0 < nbins ? s->hist[nbins - 1 - r0] : 0u;
    const uint32_t c1 = r1 < nbins ? s->hist[nbins - 1 - r1] : 0u;
    uint32_t total;
    const uint32_t above = ds_block_excl_scan(c0 + c1, s->ws, &total);
    if (c0 && above < need && need <= above + c0) {
        s->digit = (uint32_t)(nbins - 1 - r0);
        s->need = need - above;
    } else if (c1 && above + c0 < need && need <= above + c0 + c1) {
        s->digit = (uint32_t)(nbins - 1 - r1);
        s->need = need - above - c0;
    }
    if (t == 0) s->total = total;
    __syncthreads();
}

template <class Get>
__device__ void ds_select(const Get& get, uint32_t n, uint32_t K, uint32_t* cache, DsSelShared* s) {
    const int t = threadIdx.x;
    for (int i = t; i < 2048; i += kDsSelThreads) s->hist[i] = 0u;
    if (t == 0) { s->need = K; s->prefix = 0u; s->all = 0u; }
    __syncthreads();
    {
        uint32_t cur = ~0u, cnt = 0u;
        for (uint32_t i = t; i < n; i += kDsSelThreads) {
            const uint32_t key = get(i);
            if (cache) cache[i] = key;
            if (key) {
                const uint32_t d = key >> 21;
                if (d == cur) ++cnt;
                else {
                    if (cnt) atomicAdd(&s->hist[cur], cnt);
                    cur = d;
                    cnt = 1u;
                }
            }
        }
        if (cnt) atomicAdd(&s->hist[cur], cnt);
    }
    __syncthreads();
    ds_find_digit(s, 2048);
    if (s->total <= K) {                             /* every candidate is taken */
        __syncthreads();
        if (t == 0) s->all = 1u;
        __syncthreads();
        return;
    }
    uint32_t prefix = s->digit;
    for (int pass = 1; pass < 3; ++pass) {
        const int shift = pass == 1 ? 10 : 0, nbits = pass == 1 ? 11 : 10, hi = shift + nbits;
        const uint32_t mask = (1u << nbits) - 1u;
        __syncthreads();
        for (int i = t; i < 2048; i += kDsSelThreads) s->hist[i] = 0u;
        __syncthreads();
        uint32_t cur = ~0u, cnt = 0u;
        for (uint32_t i = t; i < n; i += kDsSelThreads) {
            const uint32_t key = cache ? cache[i] : get(i);
            if (key && (key >> hi) == prefix) {
                const uint32_t d = (key >> shift) & mask;
                if (d == cur) ++cnt;
                else {
                    if (cnt) atomicAdd(&s->hist[cur], cnt);
                    cur = d;
                    cnt = 1u;
                }
            }
        }
        if (cnt) atomicAdd(&s->hist[cur], cnt);
        __syncthreads();
        ds_find_digit(s, 1 << nbits);
        prefix = (prefix << nbits) | s->digit;
    }
    __syncthreads();
    if (t == 0) s->prefix = prefix;
    __syncthreads();
}

/* Walks the keys in index order: emit(i, take, out) for every i < n, `out` the taken key's rank in
 * index order; then finish(count taken). */
template <class Get, class Emit, class Finish>
__device__ void ds_emit(const Get& get, uint32_t n, const uint32_t* cache, DsSelShared* s,
                        const Emit& emit, const Finish& finish) {
    const bool all = s->all != 0u;
    const uint32_t T = all ? 0u : s->prefix, need = all ? 0u : s->need;
    uint32_t base_gt = 0u, base_eq = 0u;
    for (uint32_t i0 = 0; i0 < n; i0 += kDsSelThreads) {
        const uint32_t i = i0 + threadIdx.x;
        const uint32_t key = i < n ? (cache ? cache[i] : get(i)) : 0u;
        const uint32_t gt = key > T ? 1u : 0u;
        const uint32_t eq = (T != 0u && key == T) ? 1u : 0u;
        uint32_t tot;
        const uint32_t ex = ds_block_excl_scan(gt | (eq << 16), s->ws, &tot);
        const uint32_t eq_rank = base_eq + (ex >> 16);
        const bool take = gt || (eq && eq_rank < need);
        const uint32_t out = base_gt + (ex & 0xffffu) + min(eq_rank, need);
        if (i < n) emit(i, take, out);
        base_gt += tot & 0xffffu;
        base_eq += tot >> 16;
    }
    finish(base_gt + min(base_eq, need));
}

/* ------------------------------------------------------------------ index_topk */
/* keep: index_candidates' bits, a bit a block of `block` positions (bshift = log2(block) when
 * block is a power of two, else -1). */
struct DsTopkArgs {
    const float* sc;
    const int* keep;
    const int* pos;
    int* sel;
    long sr, scs, kr, kcs, ps, selr, selc;
    int M, C, K, ratio, block, bshift;
};

__device__ __forceinline__ long ds_topk_reach(const DsTopkArgs& a, long m) {
    long reach = ((long)a.pos[m * a.ps] + 1) / a.ratio;
    if (reach > a.C) reach = a.C;
    return reach < 0 ? 0 : reach;
}

/* The key of position j0 + i of a row: not a candidate outside keep's blocks or at -inf (or NaN). */
struct DsTopkKey {
    const float* row;
    const int* krow;
    long scs, kcs, j0;
    int block, bshift;
    __device__ __forceinline__ uint32_t operator()(uint32_t i) const {
        const long j = j0 + (long)i;
        if (krow) {
            const uint32_t b = bshift >= 0 ? (uint32_t)j >> bshift : (uint32_t)j / (uint32_t)block;
            if (!(((uint32_t)krow[(long)(b >> 5) * kcs] >> (b & 31u)) & 1u)) return 0u;
        }
        const float v = row[j * scs];
        return v > -kDsInf ? ds_key(v) : 0u;
    }
};

__global__ __launch_bounds__(kDsSelThreads) void ds41_index_topk_kernel(DsTopkArgs a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    for (long m = blockIdx.x; m < a.M; m += gridDim.x) {
        const uint32_t n = (uint32_t)ds_topk_reach(a, m);
        const DsTopkKey get{ a.sc + m * a.sr, a.keep ? a.keep + m * a.kr : nullptr, a.scs, a.kcs, 0,
                             a.block, a.bshift };
        uint32_t* c = n <= (uint32_t)kDsCacheUse ? cache : nullptr;
        ds_select(get, n, (uint32_t)a.K, c, &s);
        int* srow = a.sel + m * a.selr;
        const long selc = a.selc;
        const int K = a.K;
        ds_emit(get, n, c, &s,
                [&](uint32_t i, bool take, uint32_t out) { if (take) srow[(long)out * selc] = (int)i; },
                [&](uint32_t tot) {
                    for (long t = (long)tot + threadIdx.x; t < K; t += kDsSelThreads) srow[t * selc] = -1;
                });
        __syncthreads();
    }
}

/* The decode form over long rows. Stage 1: each (chunk, row) keeps its chunk's exact top-K as
 * (key << 32 | position), in position order, and its count. A row's winner is among its chunk's
 * K best, so stage 2's top-K over the chunks in order (position order again) is the row's. */
struct DsSplitArgs {
    DsTopkArgs t;
    unsigned long long* cand;
    int* cnt;
    int nch, chunk;
};

__global__ __launch_bounds__(kDsSelThreads) void ds41_index_topk_split1_kernel(DsSplitArgs a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    const int c = blockIdx.x;
    const long m = blockIdx.y;
    const long j0 = (long)c * a.chunk;
    long n = ds_topk_reach(a.t, m) - j0;
    n = n < 0 ? 0 : (n > a.chunk ? a.chunk : n);
    const DsTopkKey get{ a.t.sc + m * a.t.sr, a.t.keep ? a.t.keep + m * a.t.kr : nullptr, a.t.scs,
                         a.t.kcs, j0, a.t.block, a.t.bshift };
    ds_select(get, (uint32_t)n, (uint32_t)a.t.K, cache, &s);
    unsigned long long* out = a.cand + (m * a.nch + c) * (long)a.t.K;
    int* cnt = a.cnt + m * a.nch + c;
    ds_emit(get, (uint32_t)n, cache, &s,
            [&](uint32_t i, bool take, uint32_t o) {
                if (take)
                    out[o] = ((unsigned long long)cache[i] << 32) |
                             (unsigned long long)(uint32_t)(j0 + (long)i);
            },
            [&](uint32_t tot) { if (threadIdx.x == 0) *cnt = (int)tot; });
}

struct DsCandKey {
    const unsigned long long* cr;
    const int* cn;
    uint32_t K;
    __device__ __forceinline__ uint32_t operator()(uint32_t i) const {
        return (int)(i % K) < cn[i / K] ? (uint32_t)(cr[i] >> 32) : 0u;
    }
};

__global__ __launch_bounds__(kDsSelThreads) void ds41_index_topk_split2_kernel(DsSplitArgs a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    for (long m = blockIdx.x; m < a.t.M; m += gridDim.x) {
        const unsigned long long* cr = a.cand + m * a.nch * (long)a.t.K;
        const uint32_t n = (uint32_t)a.nch * (uint32_t)a.t.K;
        const DsCandKey get{ cr, a.cnt + m * a.nch, (uint32_t)a.t.K };
        uint32_t* c = n <= (uint32_t)kDsCacheUse ? cache : nullptr;
        ds_select(get, n, (uint32_t)a.t.K, c, &s);
        int* srow = a.t.sel + m * a.t.selr;
        const long selc = a.t.selc;
        const int K = a.t.K;
        ds_emit(get, n, c, &s,
                [&](uint32_t i, bool take, uint32_t out) {
                    if (take) srow[(long)out * selc] = (int)(uint32_t)(cr[i] & 0xffffffffull);
                },
                [&](uint32_t tot) {
                    for (long t = (long)tot + threadIdx.x; t < K; t += kDsSelThreads) srow[t * selc] = -1;
                });
        __syncthreads();
    }
}

/* ------------------------------------------------------------------ index_candidates */
/* keep: a bit a block, word b / 32 of a row holding blocks [32 (b / 32), 32 (b / 32) + 32). */
struct DsCandArgs {
    const float* sc;
    const int* pos;
    int* keep;
    long sr, scs, ps, kr, kcs;
    int M, C, block, KB, ratio;
};

/* Block b's score: the max of its positions (std::max(cur, v): a NaN never replaces), +inf when
 * pinned. A -inf block is never kept, so it is not a candidate. */
struct DsBlockKey {
    const float* row;
    long scs, C, block, pin;
    __device__ __forceinline__ uint32_t operator()(uint32_t b) const {
        float bs = -kDsInf;
        const long j0 = (long)b * block;
        const long j1 = j0 + block < C ? j0 + block : C;
        for (long j = j0; j < j1; ++j) {
            const float v = row[j * scs];
            bs = bs < v ? v : bs;
        }
        if ((long)b == pin) bs = kDsInf;
        return bs > -kDsInf ? ds_key(bs) : 0u;
    }
};

static_assert(kDsSelThreads % 32 == 0, "index_candidates writes a wave's ballot as a keep word");

__global__ __launch_bounds__(kDsSelThreads) void ds41_index_candidates_kernel(DsCandArgs a) {
    __shared__ DsSelShared s;
    __shared__ uint32_t cache[kDsCache];
    const long nb = ((long)a.C + a.block - 1) / a.block;
    for (long m = blockIdx.x; m < a.M; m += gridDim.x) {
        const long reach = ((long)a.pos[m * a.ps] + 1) / a.ratio;
        const long pin = (reach >= 1 && (reach - 1) / a.block < nb) ? (reach - 1) / a.block : -1;
        const DsBlockKey get{ a.sc + m * a.sr, a.scs, a.C, a.block, pin };
        uint32_t* c = nb <= kDsCacheUse ? cache : nullptr;
        ds_select(get, (uint32_t)nb, (uint32_t)a.KB, c, &s);
        int* krow = a.keep + m * a.kr;
        const long kcs = a.kcs;
        /* ds_emit walks the blocks kDsSelThreads at a time, thread t on block i0 + t, so a wave's
         * 32 lanes hold one word's blocks and its lane 0 the word's first: the wave's ballot of
         * `take` is the word. Past the last block the lanes do not run and their bits are 0; a
         * wave whose lane 0 does not run holds no block. */
        ds_emit(get, (uint32_t)nb, c, &s,
                [&](uint32_t b, bool take, uint32_t) {
                    const uint32_t word = __builtin_amdgcn_ballot_w32(take);
                    if ((threadIdx.x & 31u) == 0u) krow[(long)(b >> 5) * kcs] = (int)word;
                },
                [&](uint32_t) {});
        __syncthreads();
    }
}

/* ================================================================== attn_sparse_sink */
struct DsAttnArgs {
    const uint16_t* q;
    const uint16_t* wc;
    const int* wbt;
    const int* wused;
    const uint16_t* cc;
    const int* cbt;
    const int* sel;
    const int* pos;
    const int* cu;
    const float* sink;
    uint16_t* out;
    long qr, outr, wbtr, wbtc, nwblk, cbtr, cbtc, ncblk, selr, selc, wus, ps, cs, sks;
    int nwbt, bs, ncbt, cbs, ratio, S, M, H, win, nk, nkp;
    int cslot;                                       /* compressed cache elements (LF8: bytes) a slot */
    float scale;
};

/* Selected entry c at block blk: bf16 elements, or (LF8) the E4M3 entry's bytes. */
template <bool LF8>
__device__ __forceinline__ const uint16_t* ds_cptr(const DsAttnArgs& a, long blk, long c) {
    const long off = (blk * a.cbs + (c * a.ratio) % a.cbs) * (long)a.cslot;
    return LF8 ? (const uint16_t*)((const unsigned char*)a.cc + off) : a.cc + off;
}

constexpr int kDsAttnD = 512;
constexpr int kDsTileRow = 80;                       /* 32 bf16 of a key + 16 bytes of padding */
constexpr int kDsTileBytes = 8 * 32 * kDsTileRow;    /* one 32-key tile a wave */

__host__ __device__ constexpr long ds_attn_tile_off(int nkp, int hg) {
    return ((long)nkp * 8 + (long)hg * nkp * 4 + (long)hg * 8 + 15) / 16 * 16;
}

/* Workgroup (d half, head group, row). LDS: the row's key pointers (null = no key), scores then p
 * [HG][nkp], max and den [HG], and the waves' transpose tiles. */
template <int HG, int DS, bool LF8>
__global__ __launch_bounds__(256) void ds41_attn_sparse_sink_kernel(DsAttnArgs a) {
    constexpr int D = kDsAttnD, NW = 8, DPT = D / DS / 256;
    static_assert(DPT == 1 || DPT == 2, "one or two columns a thread");
    const uint16_t** kp = (const uint16_t**)r4d_ds41_sparse_dyn;
    float* sp = (float*)(r4d_ds41_sparse_dyn + (long)a.nkp * 8);
    float* smx = sp + (long)HG * a.nkp;
    float* sden = smx + HG;
    unsigned char* tile = r4d_ds41_sparse_dyn + ds_attn_tile_off(a.nkp, HG);
    const int tid = threadIdx.x, lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane((int)(tid >> 5));
    const int hg0 = blockIdx.y * HG;
    const int hmax = min(HG, a.H - hg0);
    const int dcol = blockIdx.x * (D / DS) + tid * DPT;
    for (long m = blockIdx.z; m < a.M; m += gridDim.z) {
        /* ---- the row's keys, in libref's order: the window, then the selections */
        const int s = ds_seq_of_row(a.cu, a.cs, a.S, m);
        const long p = a.pos[m * a.ps];
        long lr = (long)a.cu[(long)(s + 1) * a.cs] - 1;
        lr = lr < 0 ? 0 : (lr >= a.M ? a.M - 1 : lr);
        const long last = a.pos[lr * a.ps];
        const long first = last + 1 - (long)a.wused[(long)s * a.wus];
        for (int k = tid; k < a.nkp; k += 256) {
            const uint16_t* ptr = nullptr;
            if (k < a.win) {
                const long j = p - a.win + 1 + k, r = j - first;
                if (j >= 0 && r >= 0 && r / a.bs < a.nwbt) {
                    const long blk = a.wbt[(long)s * a.wbtr + (r / a.bs) * a.wbtc];
                    if (blk >= 0 && blk < a.nwblk) ptr = a.wc + (blk * a.bs + r % a.bs) * (long)D;
                }
            } else if (k < a.nk) {
                const long c = a.sel[m * a.selr + (long)(k - a.win) * a.selc];
                if (c >= 0 && (c * a.ratio) / a.cbs < a.ncbt) {
                    const long blk = a.cbt[(long)s * a.cbtr + ((c * a.ratio) / a.cbs) * a.cbtc];
                    if (blk >= 0 && blk < a.ncblk)
                        ptr = ds_cptr<LF8>(a, blk, c);
                }
            }
            kp[k] = ptr;
        }
        __syncthreads();

        /* ---- scores: a lane a key, each dot in d order. A wave loads its 32 keys 32 columns at a
         * time (four lanes a key, 16 bytes each), transposes them through its tile and reads its own
         * key's row back; q is uniform, so it comes through scalar loads. */
        {
            const uint32_t* qrow = (const uint32_t*)(a.q + m * a.qr) + (long)hg0 * (D / 2);
            unsigned char* tb = tile + wave * (32 * kDsTileRow);
            for (int kb = wave * 32; kb < a.nk; kb += NW * 32) {
                const uint16_t* lp[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int kk = kb + i * 8 + (lane >> 2);
                    lp[i] = kk < a.nk ? kp[kk] : nullptr;
                }
                float acc[HG];
#pragma unroll
                for (int h = 0; h < HG; ++h) acc[h] = 0.0f;
                for (int c = 0; c < D / 32; ++c) {
                    uint4 v[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        v[i] = lp[i] ? ds_kpiece<LF8>(lp[i], c * 4 + (lane & 3), kb + i * 8 + (lane >> 2) >= a.win)
                                     : make_uint4(0, 0, 0, 0);
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        *(uint4*)(tb + (i * 8 + (lane >> 2)) * kDsTileRow + (lane & 3) * 16) = v[i];
                    ds_wave_sync();
                    /* Eight columns at a time keeps HG x 4 q words live, within the scalar file. */
#pragma unroll 1
                    for (int i = 0; i < 4; ++i) {
                        const uint4 ri = *(const uint4*)(tb + lane * kDsTileRow + i * 16);
                        const uint32_t wk[4] = { ri.x, ri.y, ri.z, ri.w };
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const float k0 = bf16w_lo(wk[e]), k1 = bf16w_hi(wk[e]);
                            const int qi = c * 16 + i * 4 + e;
#pragma unroll
                            for (int h = 0; h < HG; ++h) {
                                const uint32_t qw = qrow[(long)(h < hmax ? h : 0) * (D / 2) + qi];
                                acc[h] = __builtin_fmaf(bf16w_lo(qw), k0, acc[h]);
                                acc[h] = __builtin_fmaf(bf16w_hi(qw), k1, acc[h]);
                            }
                        }
                    }
                    ds_wave_sync();                  /* the tile's reads before the next chunk's writes */
                }
                const int k = kb + lane;
                if (k < a.nk) {
                    const bool have = kp[k] != nullptr;
#pragma unroll
                    for (int h = 0; h < HG; ++h) sp[(long)h * a.nkp + k] = have ? acc[h] * a.scale : -kDsInf;
                }
            }
        }
        __syncthreads();

        /* ---- per head (one wave each): max floored at -1e30, then p in place (0 for no key) */
        if (wave < HG) {
            float* sh = sp + (long)wave * a.nkp;
            float mx = -1e30f;
            for (int k = lane; k < a.nk; k += 32) {
                const float v = sh[k];
                mx = mx < v ? v : mx;
            }
            mx = r4d_wave_max(mx);
            for (int k = lane; k < a.nkp; k += 32)
                sh[k] = (k < a.nk && kp[k]) ? ds_expf(sh[k] - mx) : 0.0f;
            if (lane == 0) smx[wave] = mx;
        }
        __syncthreads();

        /* ---- sum p k and sum p, key by key in libref's order. A key with no entry adds p = 0 times
         * k = 0, which leaves both sums as they were (they start at +0 and never reach -0). */
        float acc[HG][DPT], den[HG];
#pragma unroll
        for (int h = 0; h < HG; ++h) {
            den[h] = 0.0f;
#pragma unroll
            for (int d = 0; d < DPT; ++d) acc[h][d] = 0.0f;
        }
        for (int k = 0; k < a.nkp; k += 4) {
            float kv[4][DPT];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const uint16_t* ptr = kp[k + u];
                if constexpr (LF8) {
                    const uint32_t w = ptr ? ds_kcols<DPT, true>(ptr, dcol, k + u >= a.win) : 0u;
                    kv[u][0] = bf16w_lo(w);
                    kv[u][DPT - 1] = DPT == 2 ? bf16w_hi(w) : bf16w_lo(w);
                } else if constexpr (DPT == 2) {
                    const uint32_t w = ptr ? *(const uint32_t*)(ptr + dcol) : 0u;
                    kv[u][0] = bf16w_lo(w);
                    kv[u][DPT - 1] = bf16w_hi(w);
                } else {
                    kv[u][0] = ptr ? bf16_to_f32((uint16_t)ldg_u16(ptr + dcol)) : 0.0f;
                }
            }
#pragma unroll
            for (int h = 0; h < HG; ++h) {
                const float4 pv = *(const float4*)(sp + (long)h * a.nkp + k);
                const float pk[4] = { pv.x, pv.y, pv.z, pv.w };
#pragma unroll
                for (int u = 0; u < 4; ++u) {
#pragma unroll
                    for (int d = 0; d < DPT; ++d) {
                        const float prod = pk[u] * kv[u][d];
                        acc[h][d] = acc[h][d] + prod;
                    }
                }
                if (wave == 0) {
#pragma unroll
                    for (int u = 0; u < 4; ++u) den[h] = den[h] + pk[u];
                }
            }
        }
        if (wave == 0 && lane < hmax) {
            float dn = den[0];
#pragma unroll
            for (int h = 1; h < HG; ++h)
                if (lane == h) dn = den[h];
            sden[lane] = dn + ds_expf(a.sink[(long)(hg0 + lane) * a.sks] - smx[lane]);
        }
        __syncthreads();
        uint16_t* orow = a.out + m * a.outr + (long)hg0 * D + dcol;
#pragma unroll
        for (int h = 0; h < HG; ++h) {
            if (h >= hmax) break;
            const float dn = sden[h];
            if constexpr (DPT == 2) {
                const uint32_t lo = ds_bf16_bits(acc[h][0] / dn), hi = ds_bf16_bits(acc[h][DPT - 1] / dn);
                *(uint32_t*)(orow + (long)h * D) = lo | (hi << 16);
            } else {
                orow[(long)h * D] = (uint16_t)ds_bf16_bits(acc[h][0] / dn);
            }
        }
        __syncthreads();
    }
}

/* ================================================================== attn_sparse_sink, decode band
 *
 * Passes of at most kWmRows rows (decode, speculative verify) run the chains of the kernel above,
 * each in its order and with its roundings, so its bits; only which thread runs a chain changes.
 * (Where two NaNs of different sign meet in one add, which one propagates follows the operand order
 * the compiler picked, in the kernel above as here: such outputs are NaN in both, their sign bit
 * may differ.) A workgroup is a (head group of HG, row) of NT threads:
 *   keys     the row's key pointers in LDS (a zero row for no key), q as f32 in LDS, the expf
 *            table, the sinks.
 *   scores   a lane a key, HS heads at a time: each dot in d order, the key read straight from
 *            global 16 bytes at a time, two 128-byte lines in flight, q broadcast from LDS (four
 *            heads: from its bf16 row by scalar loads). Each
 *            wave keeps its 32 keys' maximum per head (floored at -1e30; a NaN score never wins,
 *            as in `mx < v ? v : mx`); the max is order-free, a -0/+0 choice included (s - max
 *            and sink - max give the same exp either way).
 *   softmax  p = glibc expf(s - max), a thread a key.
 *   PV       a thread owns DPT latent columns x HPT heads and sums p k over the keys in key order;
 *            sum p, key by key, in the first wave of each head part. The keys come either straight
 *            from global (PVD: a wave walks 32-key blocks, one block's loads in flight, no barrier)
 *            or through LDS tiles of NT/64 keys (a thread one 16-byte piece, P tiles in flight in
 *            registers, a barrier a tile). A no-key slot adds p = 0 times a zero row: +0, which
 *            leaves a sum that starts at +0 unchanged.
 * No global load may sit under a branch: the wait-count pass then waits for every load in flight
 * (s_wait_loadcnt 0) and the pipelining is gone; loads past the end read the zero row instead. */
__device__ __align__(16) uint16_t kDsZeroKey[kDsAttnD];

/* 16 bytes at p[i] by a global (not flat) load. */
typedef unsigned int ds_v4u __attribute__((ext_vector_type(4)));
__device__ __forceinline__ uint4 ds_ldg16(const uint16_t* p, int i) {
    const ds_v4u v = ((__attribute__((address_space(1))) const ds_v4u*)p)[i];
    return make_uint4(v.x, v.y, v.z, v.w);
}

struct DsDecLds {
    long kp, sp, smx, sden, tab, u, sq, bmax, total;
};
/* nkq: keys rounded up to 32. kp [nkq] pointers, sp [HG][nkq] scores then p, smx / sden [HG] (sden
 * holds the sinks until the den), tab the 32 doubles of the expf table; then, in one region,
 * sq [HG][512] q and bmax [HG][nkq/32] block maxima (scores) or the two key tiles of a tiled PV
 * (2 x nt x 16 bytes; nt = 0 for the direct PV). */
__host__ __device__ constexpr DsDecLds ds_dec_lds(int nkq, int hg, int nt) {
    DsDecLds l{};
    l.kp = 0;
    l.sp = (long)nkq * 8;
    l.smx = l.sp + (long)hg * nkq * 4;
    l.sden = l.smx + (long)hg * 4;
    l.tab = (l.sden + (long)hg * 4 + 15) / 16 * 16;
    l.u = l.tab + 32 * 8;
    l.sq = l.u;
    l.bmax = l.sq + (long)hg * kDsAttnD * 4;
    const long sc = l.bmax + (long)hg * (nkq / 32) * 4, pv = l.u + 2L * nt * 16;
    l.total = sc > pv ? sc : pv;
    return l;
}

/* ds_expf with the table read from `tab` (an LDS copy of kDsExp2fTab): the same operations. */
__device__ __forceinline__ float ds_expf_tab(float x, const unsigned long long* tab) {
    const uint32_t ux = __float_as_uint(x);
    const uint32_t abstop = (ux >> 20) & 0x7ffu;
    if (abstop >= 0x42bu) {
        if (ux == 0xff800000u) return 0.0f;
        if (abstop >= 0x7f8u) return x + x;
        if (x > 0x1.62e42ep6f) return kDsInf;
        if (x < -0x1.9fe368p6f) return 0.0f;
    }
    const double xd = (double)x;
    double z = 0x1.71547652b82fep+5 * xd;
    double kd = z + 0x1.8p+52;
    const unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
    kd -= 0x1.8p+52;
    const double r = z - kd;
    unsigned long long t = tab[ki % 32];
    t += ki << 47;
    const double s = __longlong_as_double((long long)t);
    z = 0x1.c6af84b912394p-20 * r + 0x1.ebfce50fac4f3p-13;
    const double r2 = r * r;
    double y = 0x1.62e42ff0c52d6p-6 * r + 1.0;
    y = z * r2 + y;
    y = y * s;
    return (float)y;
}

/* An offset of 0 the compiler cannot see through, made after the running sums and before the
 * next 16 key bytes are unpacked: the q reads and key conversions of 8 columns cannot be hoisted
 * above the previous columns' fmas (HS x 64 live floats spilled otherwise). */
template <int HS>
__device__ __forceinline__ int ds_dec_step(const float (&acc)[HS], uint32_t (&w)[4]) {
    int z;
    if constexpr (HS == 1)
        asm volatile("v_mov_b32 %0, 0" : "=v"(z), "+v"(w[0]), "+v"(w[1]), "+v"(w[2]), "+v"(w[3]) : "v"(acc[0]));
    else if constexpr (HS == 2)
        asm volatile("v_mov_b32 %0, 0" : "=v"(z), "+v"(w[0]), "+v"(w[1]), "+v"(w[2]), "+v"(w[3])
                     : "v"(acc[0]), "v"(acc[1]));
    else
        asm volatile("v_mov_b32 %0, 0" : "=v"(z), "+v"(w[0]), "+v"(w[1]), "+v"(w[2]), "+v"(w[3])
                     : "v"(acc[0]), "v"(acc[1]), "v"(acc[2]), "v"(acc[3]));
    return z;
}

/* 64 latent columns of one key (8 x 16 bytes) into the HS dots, in d order. With more than one
 * head, step i+1's q reads go out (behind the fence of step i-1's sums) before step i's fmas. */
template <int HS>
__device__ __forceinline__ void ds_dec_dot(float (&acc)[HS], const uint4 (&r)[8], const float* qp) {
    static_assert(HS == 1 || HS == 2 || HS == 4, "head subgroup");
    if constexpr (HS == 1) {
        /* one head: 64 hoisted q floats fit (no spill), and the hoist hides the LDS latency */
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t wk[4] = { r[i].x, r[i].y, r[i].z, r[i].w };
            const float4 qa = *(const float4*)(qp + i * 8);
            const float4 qb = *(const float4*)(qp + i * 8 + 4);
            const float qv[8] = { qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w };
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[0] = __builtin_fmaf(qv[2 * e], bf16w_lo(wk[e]), acc[0]);
                acc[0] = __builtin_fmaf(qv[2 * e + 1], bf16w_hi(wk[e]), acc[0]);
            }
        }
    } else {
        float4 qc[HS][2], qn[HS][2];
        uint32_t wc[4] = { r[0].x, r[0].y, r[0].z, r[0].w }, wn[4];
        {
            const float* qi = qp + ds_dec_step<HS>(acc, wc);
#pragma unroll
            for (int h = 0; h < HS; ++h) {
                qc[h][0] = *(const float4*)(qi + h * kDsAttnD);
                qc[h][1] = *(const float4*)(qi + h * kDsAttnD + 4);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i + 1 < 8) {
                wn[0] = r[i + 1].x; wn[1] = r[i + 1].y; wn[2] = r[i + 1].z; wn[3] = r[i + 1].w;
                const float* qi = qp + ds_dec_step<HS>(acc, wn) + (i + 1) * 8;
#pragma unroll
                for (int h = 0; h < HS; ++h) {
                    qn[h][0] = *(const float4*)(qi + h * kDsAttnD);
                    qn[h][1] = *(const float4*)(qi + h * kDsAttnD + 4);
                }
            }
#pragma unroll
            for (int h = 0; h < HS; ++h) {
                const float qv[8] = { qc[h][0].x, qc[h][0].y, qc[h][0].z, qc[h][0].w,
                                      qc[h][1].x, qc[h][1].y, qc[h][1].z, qc[h][1].w };
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[h] = __builtin_fmaf(qv[2 * e], bf16w_lo(wc[e]), acc[h]);
                    acc[h] = __builtin_fmaf(qv[2 * e + 1], bf16w_hi(wc[e]), acc[h]);
                }
            }
            if (i + 1 < 8) {
#pragma unroll
                for (int h = 0; h < HS; ++h) { qc[h][0] = qn[h][0]; qc[h][1] = qn[h][1]; }
#pragma unroll
                for (int e = 0; e < 4; ++e) wc[e] = wn[e];
            }
        }
    }
}

/* The HS-head dot of ds_dec_dot with q read straight from its bf16 row by scalar loads (wave-uniform
 * addresses: qw the row's words at this chunk, ho[h] head h's word offset), so the q values take
 * neither LDS bandwidth nor VGPRs. The same fmas in the same order. */
template <int HS>
__device__ __forceinline__ void ds_dec_dot_s(float (&acc)[HS], const uint4 (&r)[8], const uint32_t* qw,
                                             const int (&ho)[HS]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t wk[4] = { r[i].x, r[i].y, r[i].z, r[i].w };
        const int z = __builtin_amdgcn_readfirstlane(ds_dec_step<HS>(acc, wk));
        const uint32_t* qi = qw + z + i * 4;
#pragma unroll
        for (int h = 0; h < HS; ++h) {
            const uint32_t* qh = qi + ho[h];
            const uint32_t qq[4] = { qh[0], qh[1], qh[2], qh[3] };
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[h] = __builtin_fmaf(bf16w_lo(qq[e]), bf16w_lo(wk[e]), acc[h]);
                acc[h] = __builtin_fmaf(bf16w_hi(qq[e]), bf16w_hi(wk[e]), acc[h]);
            }
        }
    }
}

/* Four keys (their DPT columns in kw) into the HPT heads' sums p k, and sum p in the den wave,
 * each chain in key order (libref's mul, then add). */
template <int HPT, int DPT>
__device__ __forceinline__ void ds_dec_pv4(float (&acc)[HPT][DPT], float (&den)[HPT], const uint32_t (&kw)[4],
                                           const float* ph, int nkq, int k0, bool dw) {
    float kf[4][DPT];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        kf[u][0] = bf16w_lo(kw[u]);
        if constexpr (DPT == 2) kf[u][DPT - 1] = bf16w_hi(kw[u]);
    }
#pragma unroll
    for (int j = 0; j < HPT; ++j) {
        const float4 pv = *(const float4*)(ph + (long)j * nkq + k0);
        const float pk[4] = { pv.x, pv.y, pv.z, pv.w };
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int d = 0; d < DPT; ++d) {
                const float prod = pk[u] * kf[u][d];
                acc[j][d] = acc[j][d] + prod;
            }
        }
        if (dw) {
#pragma unroll
            for (int u = 0; u < 4; ++u) den[j] = den[j] + pk[u];
        }
    }
}

/* The DPT columns at dcol of the 32 keys whose pointers lanes 0..31 hold (a wave-uniform pointer a
 * key, through readlane), straight from global. */
template <int DPT>
__device__ __forceinline__ void ds_dec_pvld(uint32_t (&kv)[32], const uint16_t* lp, int dcol) {
    const unsigned long long v = (unsigned long long)lp;
    const int lo = (int)(uint32_t)v, hi = (int)(uint32_t)(v >> 32);
#pragma unroll
    for (int u = 0; u < 32; ++u) {
        const unsigned long long pu = ((unsigned long long)(uint32_t)__builtin_amdgcn_readlane(hi, u) << 32) |
                                      (uint32_t)__builtin_amdgcn_readlane(lo, u);
        const uint16_t* p = (const uint16_t*)pu + dcol;
        if constexpr (DPT == 2) kv[u] = *(__attribute__((address_space(1))) const uint32_t*)p;
        else kv[u] = ldg_u16(p);
    }
}

template <int HPT, int DPT>
__device__ __forceinline__ void ds_dec_pvsum(float (&acc)[HPT][DPT], float (&den)[HPT], const uint32_t (&kv)[32],
                                             const float* ph, int nkq, int k0, bool dw) {
#pragma unroll
    for (int u0 = 0; u0 < 32; u0 += 4) {
        const uint32_t kw[4] = { kv[u0], kv[u0 + 1], kv[u0 + 2], kv[u0 + 3] };
        ds_dec_pv4<HPT, DPT>(acc, den, kw, ph, nkq, k0 + u0, dw);
    }
}


/* LF8: a tile piece's raw bytes (a bf16 row's 16 or an E4M3 entry's 16 code bytes and its scale
 * byte), the same loads on every lane; ds_tile_cvt widens the piece at the LDS store. */
template <bool LF8>
__device__ __forceinline__ void ds_tile_ld(uint4& r, uint32_t& s, const uint16_t* p, int lc, bool f8) {
    if constexpr (!LF8) {
        r = ds_ldg16(p, lc);
    } else {
        r = ds_gld16(p, f8 ? (lc >> 1) * 16 : lc * 16);
        s = ds_gld8(p, f8 ? kDsLatScales + (lc >> 1) : 0);
    }
}
template <bool LF8>
__device__ __forceinline__ uint4 ds_tile_cvt(uint4 r, uint32_t s, int lc, bool f8) {
    if constexpr (!LF8) return r;
    const uint4 x = ds_f8x8((lc & 1) ? r.z : r.x, (lc & 1) ? r.w : r.y, ds_e4m3s(s));
    return f8 ? x : r;
}

template <int HG, int NT, int DPT, int P, bool PVD, bool LF8>
__global__ __launch_bounds__(NT) void ds41_attn_dec_kernel(DsAttnArgs a) {
    constexpr int D = kDsAttnD, NW = NT / 32, NCG = D / DPT, TK = NT / 64;
    constexpr int NHP = (NT / NCG) < HG ? (NT / NCG) : HG;   /* head parts in PV */
    constexpr int HPT = HG / NHP;                           /* heads a PV thread */
    constexpr int HS = HG < 4 ? HG : 4, NSG = HG / HS;      /* heads a score task, tasks a key block */
    constexpr bool QS = HS == 4;                            /* q by scalar loads in the scores */
    static_assert(HG % HS == 0, "head subgroups");
    static_assert(NT % 64 == 0 && 32 % TK == 0 && TK % 4 == 0 && NT >= NCG && HG % NHP == 0 &&
                  (DPT == 1 || DPT == 2) && P >= 1, "layout");
    static_assert(!(LF8 && PVD), "an E4M3 latent takes the tiled PV");
    const int nkq = (a.nk + 31) / 32 * 32, nkb = nkq / 32, ntl = nkq / TK;
    const DsDecLds L = ds_dec_lds(nkq, HG, PVD ? 0 : NT);
    const uint16_t** kp = (const uint16_t**)(r4d_ds41_sparse_dyn + L.kp);
    float* sp = (float*)(r4d_ds41_sparse_dyn + L.sp);
    float* sq = (float*)(r4d_ds41_sparse_dyn + L.sq);
    float* bmax = (float*)(r4d_ds41_sparse_dyn + L.bmax);
    float* smx = (float*)(r4d_ds41_sparse_dyn + L.smx);
    float* sden = (float*)(r4d_ds41_sparse_dyn + L.sden);
    unsigned long long* tab = (unsigned long long*)(r4d_ds41_sparse_dyn + L.tab);
    unsigned char* tiles = r4d_ds41_sparse_dyn + L.u;
    const int tid = threadIdx.x, lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane((int)(tid >> 5));
    const int hg0 = blockIdx.x * HG;
    const int hmax = min(HG, a.H - hg0);
    const long m = blockIdx.y;
    const uint16_t* zero = kDsZeroKey;

    /* ---- the row's keys, in libref's order: the window, then the selections */
    {
        const int s = ds_seq_of_row(a.cu, a.cs, a.S, m);
        const long p = a.pos[m * a.ps];
        long lr = (long)a.cu[(long)(s + 1) * a.cs] - 1;
        lr = lr < 0 ? 0 : (lr >= a.M ? a.M - 1 : lr);
        const long last = a.pos[lr * a.ps];
        const long first = last + 1 - (long)a.wused[(long)s * a.wus];
        for (int k = tid; k < nkq; k += NT) {
            const uint16_t* ptr = zero;
            if (k < a.win) {
                const long j = p - a.win + 1 + k, r = j - first;
                if (j >= 0 && r >= 0 && r / a.bs < a.nwbt) {
                    const long blk = a.wbt[(long)s * a.wbtr + (r / a.bs) * a.wbtc];
                    if (blk >= 0 && blk < a.nwblk) ptr = a.wc + (blk * a.bs + r % a.bs) * (long)D;
                }
            } else if (k < a.nk) {
                const long c = a.sel[m * a.selr + (long)(k - a.win) * a.selc];
                if (c >= 0 && (c * a.ratio) / a.cbs < a.ncbt) {
                    const long blk = a.cbt[(long)s * a.cbtr + ((c * a.ratio) / a.cbs) * a.cbtc];
                    if (blk >= 0 && blk < a.ncblk)
                        ptr = ds_cptr<LF8>(a, blk, c);
                }
            }
            kp[k] = ptr;
        }
        const uint32_t* qrow = (const uint32_t*)(a.q + m * a.qr) + (long)hg0 * (D / 2);
        for (int i = tid; i < (QS ? 0 : HG * (D / 2)); i += NT) {
            const int h = i / (D / 2), c = i % (D / 2);
            const uint32_t w = qrow[(long)(h < hmax ? h : 0) * (D / 2) + c];
            *(float2*)(sq + h * D + 2 * c) = make_float2(bf16w_lo(w), bf16w_hi(w));
        }
        if (tid < 32) tab[tid] = kDsExp2fTab[tid];
        if (tid < hmax) sden[tid] = a.sink[(long)(hg0 + tid) * a.sks];   /* the sinks, until the den */
    }
    lds_barrier();

    /* ---- scores: a lane a key, every dot in d order. A wave task is (32 keys, HS heads); the
     * NSG tasks of a key block go to neighbouring waves, so the block's second read hits L0.
     * Four heads read q by scalar loads (QS; LDS-bound otherwise), fewer from LDS. */
    for (int t = wave; t < nkb * NSG; t += NW) {
        const int kb = t / NSG, g = t % NSG;
        const int k = kb * 32 + lane;
        const uint16_t* kk = kp[k];
        const bool have = kk != zero;
        const float* qg = sq + g * HS * D;
        float acc[HS];
#pragma unroll
        for (int h = 0; h < HS; ++h) acc[h] = 0.0f;
        const uint32_t* qw = (const uint32_t*)(a.q + m * a.qr) + (long)hg0 * (D / 2);
        int ho[HS];
#pragma unroll
        for (int h = 0; h < HS; ++h) ho[h] = (g * HS + h < hmax ? g * HS + h : 0) * (D / 2);
        if constexpr (LF8) {
            /* The same chain on the widened words: both formats' loads on every lane (dual), the
             * widening after the previous chunk's dots. */
            const bool f8 = k >= a.win;
            uint4 ra[8], rb[8], x[8];
            uint32_t sa, sb;
            ds_dual_ld8(ra, sa, kk, 0, f8);
#pragma unroll 1
            for (int c = 0; c < D / 8; c += 16) {
                ds_dual_ld8(rb, sb, kk, c + 8, f8);
                ds_dual_cvt8(x, ra, sa, f8);
                if constexpr (QS) ds_dec_dot_s<HS>(acc, x, qw + c * 4, ho);
                else ds_dec_dot<HS>(acc, x, qg + c * 8);
                const int cn = c + 16 < D / 8 ? c + 16 : c;
                ds_dual_ld8(ra, sa, kk, cn, f8);
                ds_dual_cvt8(x, rb, sb, f8);
                if constexpr (QS) ds_dec_dot_s<HS>(acc, x, qw + (c + 8) * 4, ho);
                else ds_dec_dot<HS>(acc, x, qg + (c + 8) * 8);
            }
        } else {
        uint4 ra[8], rb[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) ra[i] = ds_ldg16(kk, i);
        /* No branch around a load (the wait counts would fall back to 0): the last pass reloads
         * its own first half, unused. */
#pragma unroll 1
        for (int c = 0; c < D / 8; c += 16) {
#pragma unroll
            for (int i = 0; i < 8; ++i) rb[i] = ds_ldg16(kk, c + 8 + i);
            if constexpr (QS) ds_dec_dot_s<HS>(acc, ra, qw + c * 4, ho);
            else ds_dec_dot<HS>(acc, ra, qg + c * 8);
            const int cn = c + 16 < D / 8 ? c + 16 : c;
#pragma unroll
            for (int i = 0; i < 8; ++i) ra[i] = ds_ldg16(kk, cn + i);
            if constexpr (QS) ds_dec_dot_s<HS>(acc, rb, qw + (c + 8) * 4, ho);
            else ds_dec_dot<HS>(acc, rb, qg + (c + 8) * 8);
        }
        }
#pragma unroll
        for (int h = 0; h < HS; ++h) {
            const int hh = g * HS + h;
            const float v = have ? acc[h] * a.scale : -kDsInf;
            sp[(long)hh * nkq + k] = v;
            const float t2 = r4d_wave_max(-1e30f < v ? v : -1e30f);
            if (lane == 0) bmax[hh * nkb + kb] = t2;
        }
    }

    /* PV through LDS tiles: the first tiles' loads go out now, under the softmax. A thread carries
     * one 16-byte piece (key lk of the tile, piece lc) of each of P tiles. No load sits under a
     * branch (the wait counts would fall back to 0): past the last tile they read the zero row. */
    const int lk = tid >> 6, lc = tid & 63;
    uint4 rg[PVD ? 1 : P];
    uint32_t rs[PVD ? 1 : P];                            /* LF8: the pieces' scale bytes */
    if constexpr (!PVD) {
#pragma unroll
        for (int s = 0; s < P; ++s) {
            const uint16_t* pp = kp[min(s, ntl - 1) * TK + lk];
            ds_tile_ld<LF8>(rg[s], rs[s], s < ntl ? pp : zero, lc, s * TK + lk >= a.win);
        }
    }
    lds_barrier();

    /* ---- per head: the max, then p in place (0 for no key) */
#pragma unroll 1
    for (int h = 0; h < HG; ++h) {
        float t = -1e30f;
        for (int j = lane; j < nkb; j += 32) t = fmaxf(t, bmax[h * nkb + j]);
        const float mx = r4d_wave_max(t);
        if (tid == 0) smx[h] = mx;
        float* sh = sp + (long)h * nkq;
        for (int k = tid; k < nkq; k += NT) sh[k] = kp[k] != zero ? ds_expf_tab(sh[k] - mx, tab) : 0.0f;
    }
    lds_barrier();                                       /* sq and bmax are dead: tiles may land */

    /* ---- sum p k and sum p, key by key in libref's order */
    const int cg = tid % NCG, hp = tid / NCG;
    const int dcol = cg * DPT;
    float acc[HPT][DPT], den[HPT];
#pragma unroll
    for (int j = 0; j < HPT; ++j) {
        den[j] = 0.0f;
#pragma unroll
        for (int d = 0; d < DPT; ++d) acc[j][d] = 0.0f;
    }
    const bool pvt = hp < NHP;                           /* wave-uniform: NCG is a multiple of 32 */
    const bool dw = cg < 32;
    const float* ph = sp + (long)(pvt ? hp * HPT : 0) * nkq;
    if constexpr (PVD) {
        /* Straight from global: a wave walks 32-key blocks, lane L holding block key L's pointer
         * (handed to the whole wave by readlane), one block's loads in flight while the previous
         * block is summed. Past the last block the loads read the zero row. */
        if (pvt) {
            uint32_t kv[2][32];
            const uint16_t* lp = kp[lane];
            ds_dec_pvld<DPT>(kv[0], lp, dcol);
#pragma unroll 1
            for (int kb = 0; kb < nkb; kb += 2) {
                lp = kb + 1 < nkb ? kp[(kb + 1) * 32 + lane] : zero;
                ds_dec_pvld<DPT>(kv[1], lp, dcol);
                ds_dec_pvsum<HPT, DPT>(acc, den, kv[0], ph, nkq, kb * 32, dw);
                lp = kb + 2 < nkb ? kp[(kb + 2) * 32 + lane] : zero;
                ds_dec_pvld<DPT>(kv[0], lp, dcol);
                if (kb + 1 < nkb) ds_dec_pvsum<HPT, DPT>(acc, den, kv[1], ph, nkq, (kb + 1) * 32, dw);
            }
        }
    } else {
        *(uint4*)(tiles + lk * (D * 2) + lc * 16) = ds_tile_cvt<LF8>(rg[0], rs[0], lc, lk >= a.win);
        {
            const uint16_t* pp = kp[min(P, ntl - 1) * TK + lk];
            ds_tile_ld<LF8>(rg[0], rs[0], P < ntl ? pp : zero, lc, P * TK + lk >= a.win);
        }
        lds_barrier();
#pragma unroll 1
        for (int t0 = 0; t0 < ntl; t0 += P) {
#pragma unroll
            for (int s = 0; s < P; ++s) {
                const int t = t0 + s;
                /* the pointer of the ring's next load first, off the tail of the step */
                const uint16_t* pp = kp[min(t + 1 + P, ntl - 1) * TK + lk];
                if (pvt && t < ntl) {
                    const unsigned char* tb = tiles + (t & 1) * (TK * D * 2) + dcol * 2;
#pragma unroll
                    for (int u0 = 0; u0 < TK; u0 += 4) {
                        uint32_t kw[4];
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            if constexpr (DPT == 2) kw[u] = *(const uint32_t*)(tb + (u0 + u) * (D * 2));
                            else kw[u] = *(const uint16_t*)(tb + (u0 + u) * (D * 2));
                        }
                        ds_dec_pv4<HPT, DPT>(acc, den, kw, ph, nkq, t * TK + u0, dw);
                    }
                }
                /* tile t+1 into the other buffer (read last before the previous barrier) */
                const int sn = (s + 1) % P;
                *(uint4*)(tiles + ((t + 1) & 1) * (TK * D * 2) + lk * (D * 2) + lc * 16) =
                    ds_tile_cvt<LF8>(rg[sn], rs[sn], lc, (t + 1) * TK + lk >= a.win);
                ds_tile_ld<LF8>(rg[sn], rs[sn], t + 1 + P < ntl ? pp : zero, lc, (t + 1 + P) * TK + lk >= a.win);
                lds_barrier();
            }
        }
    }
    if (pvt && cg == 0) {
#pragma unroll
        for (int j = 0; j < HPT; ++j) {
            const int h = hp * HPT + j;
            if (h < hmax) sden[h] = den[j] + ds_expf_tab(sden[h] - smx[h], tab);
        }
    }
    lds_barrier();
    if (pvt) {
        uint16_t* orow = a.out + m * a.outr + (long)hg0 * D + dcol;
#pragma unroll
        for (int j = 0; j < HPT; ++j) {
            const int h = hp * HPT + j;
            if (h >= hmax) break;
            const float dn = sden[h];
            if constexpr (DPT == 2) {
                const uint32_t lo = ds_bf16_bits(acc[j][0] / dn), hi = ds_bf16_bits(acc[j][DPT - 1] / dn);
                *(uint32_t*)(orow + (long)h * D) = lo | (hi << 16);
            } else {
                orow[(long)h * D] = (uint16_t)ds_bf16_bits(acc[j][0] / dn);
            }
        }
    }
}

/* ================================================================== attn_sparse_sink on WMMA
 *
 * Passes of more than kWmRows rows (prefill) on v_wmma_f32_16x16x16_bf16. A workgroup is a
 * (16-head tile, row); its four waves own 128 latent columns each and walk the row's keys 16 at
 * a time:
 *   S^T = K Q^T     over the wave's 128 columns: A = 16 keys as stored (8 columns a lane), B = Q^T
 *                   held in registers. The four partials meet in LDS and every wave sums them in
 *                   wave order, so all four hold the same scores: lane L head L%16, keys 8(L/16)+e.
 *   softmax         online: a running max per head (floored at -1e30), p = exp2(s log2e - max) by
 *                   v_exp_f32, the accumulators rescaled when the max moves, den summing f32 p.
 *   O^T += K^T P^T  A = K^T by global_load_tr_b128 (lane L column L%16, keys 8(L/16)+e), B = P^T
 *                   straight from the score layout, P = hi + lo in bf16 (two WMMAs; hi truncated,
 *                   lo = p - hi rounded, so |p - hi - lo| <= 2^-17 p).
 * Not libref's bits: q.k products are exact but sum in WMMA order, the exponential is v_exp_f32
 * and the softmax online; the decode band keeps the exact kernels above. A key
 * with no entry reads a zero row and scores -inf (0 * NaN from a recycled slot would be NaN). */
typedef __bf16 ds_v8bf __attribute__((ext_vector_type(8)));
typedef float ds_v8f __attribute__((ext_vector_type(8)));
typedef short ds_v8s __attribute__((ext_vector_type(8)));
typedef __attribute__((address_space(1))) ds_v8s ds_gv8s;

constexpr int kWmRows = 48;                          /* passes up to this many rows stay scalar */
constexpr int kWmWaves = 4;
constexpr int kWmCols = kDsAttnD / kWmWaves;         /* latent columns a wave */
constexpr int kWmSbufBytes = 2 * kWmWaves * 32 * 8 * 4;
/* LF8: a wave's 16 widened key rows of its 128 columns, through which K^T is read (an E4M3 entry
 * cannot take global_load_tr's transpose): 16 rows of 256 bytes + 16 of padding. */
constexpr int kWmTrPitch = 272;
constexpr int kWmTrBytes = 16 * kWmTrPitch;

__host__ __device__ constexpr long ds_wm_mask_off(int nkt) { return (long)nkt * 16 * 8; }
__host__ __device__ constexpr long ds_wm_sbuf_off(int nkt) {
    return (ds_wm_mask_off(nkt) + (long)nkt * 4 + 15) / 16 * 16;
}

__device__ __forceinline__ ds_v8f ds_wmma(uint4 a, uint4 b, ds_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(ds_v8bf, a),
                                                             __builtin_bit_cast(ds_v8bf, b), c);
}
/* global_load_tr_b128: lane 8g+e gives the address of 8 consecutive bf16 and lane 8g+i receives
 * value i of lanes 8g..8g+7, in lane order (an 8x8 transpose within each group of eight lanes). */
__device__ __forceinline__ uint4 ds_load_tr(const uint16_t* p) {
    return __builtin_bit_cast(uint4, __builtin_amdgcn_global_load_tr_b128_v8i16((ds_gv8s*)p));
}
__device__ __forceinline__ float ds_xor16(float v) {
    const int x = __builtin_bit_cast(int, v);
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false));
}
__device__ __forceinline__ float ds_ex2(float x) { return __builtin_amdgcn_exp2f(x); }
__device__ __forceinline__ uint32_t ds_rne_hi(float f) {        /* bf16 of a finite f, high half */
    const uint32_t u = __float_as_uint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
}

template <bool LO, bool LF8>
__global__ __launch_bounds__(128) void ds41_attn_sparse_sink_wm_kernel(DsAttnArgs a, float sl2) {
    constexpr int D = kDsAttnD;
    constexpr float kL2e = 1.4426950408889634f;
    const int nkt = (a.nk + 15) >> 4, nkk = nkt * 16;
    const uint16_t** kp = (const uint16_t**)r4d_ds41_sparse_dyn;
    uint32_t* vmask = (uint32_t*)(r4d_ds41_sparse_dyn + ds_wm_mask_off(nkt));
    float4* sbuf = (float4*)(r4d_ds41_sparse_dyn + ds_wm_sbuf_off(nkt));
    const int tid = threadIdx.x, lane = tid & 31, hf = lane >> 4;
    const int wave = __builtin_amdgcn_readfirstlane((int)(tid >> 5));
    const int hl = blockIdx.x * 16 + (lane & 15);
    const bool hok = hl < a.H;
    const int c0 = wave * kWmCols;
    const int kq = lane & 15;                        /* S^T's A row: key kq of the tile */
    const int ktr = 8 * hf + (lane & 7);             /* K^T's transposing row: key ktr */
    const int dtr = c0 + 8 * ((lane >> 3) & 1);
    for (long m = blockIdx.y; m < a.M; m += gridDim.y) {
        /* ---- the row's keys in libref's order (the window, then the selections), a null one
         * pointing at a zero row; a 16-bit mask a tile of which are keys */
        const int s = ds_seq_of_row(a.cu, a.cs, a.S, m);
        const long p = a.pos[m * a.ps];
        long lr = (long)a.cu[(long)(s + 1) * a.cs] - 1;
        lr = lr < 0 ? 0 : (lr >= a.M ? a.M - 1 : lr);
        const long last = a.pos[lr * a.ps];
        const long first = last + 1 - (long)a.wused[(long)s * a.wus];
        for (int k0 = 0; k0 < nkk; k0 += 128) {
            const int k = k0 + tid;
            const uint16_t* ptr = nullptr;
            if (k < a.win) {
                const long j = p - a.win + 1 + k, r = j - first;
                if (j >= 0 && r >= 0 && r / a.bs < a.nwbt) {
                    const long blk = a.wbt[(long)s * a.wbtr + (r / a.bs) * a.wbtc];
                    if (blk >= 0 && blk < a.nwblk) ptr = a.wc + (blk * a.bs + r % a.bs) * (long)D;
                }
            } else if (k < a.nk) {
                const long c = a.sel[m * a.selr + (long)(k - a.win) * a.selc];
                if (c >= 0 && (c * a.ratio) / a.cbs < a.ncbt) {
                    const long blk = a.cbt[(long)s * a.cbtr + ((c * a.ratio) / a.cbs) * a.cbtc];
                    if (blk >= 0 && blk < a.ncblk)
                        ptr = ds_cptr<LF8>(a, blk, c);
                }
            }
            if (k < nkk) kp[k] = ptr ? ptr : kDsZeroKey;
            const uint32_t b = (uint32_t)__builtin_amdgcn_ballot_w32(ptr != nullptr);
            const int t0 = (k0 + wave * 32) >> 4;
            if (lane == 0) {
                if (t0 < nkt) vmask[t0] = b & 0xffffu;
                if (t0 + 1 < nkt) vmask[t0 + 1] = b >> 16;
            }
        }
        __syncthreads();

        /* ---- Q^T for this lane's head and the wave's columns: step i, half hf = columns
         * c0 + 64 hf + 8 i .. + 7 (the same split as the key rows below) */
        uint4 qf[8];
        {
            const uint16_t* qp = a.q + m * a.qr + (long)(hok ? hl : 0) * D + c0 + 64 * hf;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                qf[i] = *(const uint4*)(qp + 8 * i);
                if (!hok) qf[i] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        ds_v8f o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = (ds_v8f)(0.0f);
        float mrun = -1e30f, den = 0.0f;
        uint4 kn[8], kr[8];
        uint32_t ks = 0;
        if constexpr (LF8) {
            ds_dual_ld8(kr, ks, kp[kq], (c0 + 64 * hf) / 8, kq >= a.win);
        } else {
            const uint16_t* pk = kp[kq] + c0 + 64 * hf;
#pragma unroll
            for (int i = 0; i < 8; ++i) kn[i] = *(const uint4*)(pk + 8 * i);
        }
        for (int t = 0; t < nkt; ++t) {
            uint4 kt[8];
            if constexpr (LF8) {
                /* tile t's rows widened (lane: key kq, columns 64 hf + 8 i), through the wave's
                 * LDS rows into K^T as global_load_tr gives it: lane L column 16 j + L%16, keys
                 * 8 (L/16) + e */
                ds_dual_cvt8(kn, kr, ks, t * 16 + kq >= a.win);
                unsigned char* tb = r4d_ds41_sparse_dyn + ds_wm_sbuf_off(nkt) + kWmSbufBytes + wave * kWmTrBytes;
#pragma unroll
                for (int i = 0; i < 8; ++i) *(uint4*)(tb + kq * kWmTrPitch + hf * 128 + i * 16) = kn[i];
                ds_wave_sync();
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    uint32_t w[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const unsigned char* c = tb + (16 * j + (lane & 15)) * 2;
                        const uint32_t e0 = *(const uint16_t*)(c + (8 * hf + 2 * q) * kWmTrPitch);
                        const uint32_t e1 = *(const uint16_t*)(c + (8 * hf + 2 * q + 1) * kWmTrPitch);
                        w[q] = e0 | (e1 << 16);
                    }
                    kt[j] = make_uint4(w[0], w[1], w[2], w[3]);
                }
            } else {
                const uint16_t* pt = kp[t * 16 + ktr] + dtr;
#pragma unroll
                for (int j = 0; j < 8; ++j) kt[j] = ds_load_tr(pt + 16 * j);
            }
            ds_v8f sp = (ds_v8f)(0.0f);
#pragma unroll
            for (int i = 0; i < 8; ++i) sp = ds_wmma(kn[i], qf[i], sp);
            if (t + 1 < nkt) {
                if constexpr (LF8) {
                    ds_dual_ld8(kr, ks, kp[(t + 1) * 16 + kq], (c0 + 64 * hf) / 8, (t + 1) * 16 + kq >= a.win);
                } else {
                    const uint16_t* pk = kp[(t + 1) * 16 + kq] + c0 + 64 * hf;
#pragma unroll
                    for (int i = 0; i < 8; ++i) kn[i] = *(const uint4*)(pk + 8 * i);
                }
            }
            float4* sb = sbuf + (t & 1) * (kWmWaves * 64);
            sb[wave * 64 + lane] = make_float4(sp[0], sp[1], sp[2], sp[3]);
            sb[wave * 64 + 32 + lane] = make_float4(sp[4], sp[5], sp[6], sp[7]);
            lds_barrier();
            float x[8];
            {
                const float4 l0 = sb[lane], h0 = sb[32 + lane];
                x[0] = l0.x; x[1] = l0.y; x[2] = l0.z; x[3] = l0.w;
                x[4] = h0.x; x[5] = h0.y; x[6] = h0.z; x[7] = h0.w;
#pragma unroll
                for (int w = 1; w < kWmWaves; ++w) {
                    const float4 l = sb[w * 64 + lane], h = sb[w * 64 + 32 + lane];
                    x[0] += l.x; x[1] += l.y; x[2] += l.z; x[3] += l.w;
                    x[4] += h.x; x[5] += h.y; x[6] += h.z; x[7] += h.w;
                }
            }
            const uint32_t vm = vmask[t] >> (8 * hf);
            float tmax = -kDsInf;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                x[e] = ((vm >> e) & 1u) ? x[e] * sl2 : -kDsInf;
                tmax = fmaxf(tmax, x[e]);
            }
            tmax = fmaxf(tmax, ds_xor16(tmax));
            const float mnew = fmaxf(mrun, tmax);
            const float alpha = ds_ex2(mrun - mnew);
            float pe[8], ps = 0.0f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                pe[e] = ds_ex2(x[e] - mnew);
                ps += pe[e];
            }
            den = den * alpha + ps;
            if (__builtin_amdgcn_ballot_w32(alpha != 1.0f)) {
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] *= alpha;
            }
            mrun = mnew;
            uint32_t hw[4], lw[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float p0 = pe[2 * q], p1 = pe[2 * q + 1];
                if constexpr (LO) {
                    const uint32_t u0 = __float_as_uint(p0) & 0xffff0000u, u1 = __float_as_uint(p1) & 0xffff0000u;
                    hw[q] = __builtin_amdgcn_perm(u1, u0, 0x07060302u);
                    lw[q] = __builtin_amdgcn_perm(ds_rne_hi(p1 - __uint_as_float(u1)),
                                                  ds_rne_hi(p0 - __uint_as_float(u0)), 0x07060302u);
                } else {
                    hw[q] = __builtin_amdgcn_perm(ds_rne_hi(p1), ds_rne_hi(p0), 0x07060302u);
                }
            }
            const uint4 ph = make_uint4(hw[0], hw[1], hw[2], hw[3]);
#pragma unroll
            for (int j = 0; j < 8; ++j) o[j] = ds_wmma(kt[j], ph, o[j]);
            if constexpr (LO) {
                const uint4 pl = make_uint4(lw[0], lw[1], lw[2], lw[3]);
#pragma unroll
                for (int j = 0; j < 8; ++j) o[j] = ds_wmma(kt[j], pl, o[j]);
            }
        }
        /* ---- den over both key halves, the sink, one rounding: lane L holds head L%16 at
         * columns c0 + 16 j + 8 hf + e */
        float dt = den + ds_xor16(den);
        if (hok) {
            dt += ds_ex2(a.sink[(long)hl * a.sks] * kL2e - mrun);
            uint16_t* op = a.out + m * a.outr + (long)hl * D + c0 + 8 * hf;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[4];
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    w[q] = ds_bf16_bits(o[j][2 * q] / dt) | (ds_bf16_bits(o[j][2 * q + 1] / dt) << 16);
                *(uint4*)(op + 16 * j) = make_uint4(w[0], w[1], w[2], w[3]);
            }
        }
        __syncthreads();
    }
}

/* ================================================================== index_score on WMMA
 *
 * Passes of more than kWmRows rows. A wave is a row and walks 16-entry tiles of its workgroup's
 * kIwEB entries: S = Q K^T (A = the row's 16-head tiles of q, B = 16 entries as stored; 8 steps of
 * 16 dims, q held in registers), D lane L = entry L%16, heads 8(L/16)+e. Then libref's roundings:
 * the dot to bf16, relu x the rounded head weight, rounded, summed in head order (the running sum
 * hops lanes L <-> L^16 between 8-head runs), rounded. Only the dot's summation order differs
 * from libref, so a score moves only where the f32 dot sat at a bf16 rounding boundary. */
constexpr int kIwEB = 128;                           /* entries a workgroup */

template <int HT, bool KF8>
__global__ __launch_bounds__(256) void ds41_index_score_wm_kernel(DsScoreArgs a) {
    constexpr int DM = 128;
    const int lane = threadIdx.x & 31, hf = lane >> 4;
    const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    const long m = (long)blockIdx.y * 8 + wave;
    if (m >= a.M) return;
    const long jb = (long)blockIdx.x * kIwEB;
    const long jend = jb + kIwEB < (long)a.C ? jb + kIwEB : (long)a.C;
    float* srow = a.sc + m * a.sr;
    const long reach = m < a.Mv ? ((long)a.pos[m * a.ps] + 1) / a.ratio : 0;
    if (jb >= reach) {
        for (long j = jb + lane; j < jend; j += 32) srow[j * a.scs] = -kDsInf;
        return;
    }
    const int s = ds_seq_of_row(a.cu, a.cs, a.S, a.row0 + m);
    uint4 qf[HT][8];
    float wh[HT][8];
#pragma unroll
    for (int t = 0; t < HT; ++t) {
        const int h = t * 16 + (lane & 15);
        const uint16_t* qp = a.q + m * a.qr + (long)(h < a.H ? h : 0) * DM + 64 * hf;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qf[t][i] = *(const uint4*)(qp + 8 * i);
            if (h >= a.H) qf[t][i] = make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int hh = t * 16 + 8 * hf + e;
            wh[t][e] = hh < a.H ? ds_round_bf16(ds_bf(a.w[m * a.wr + (long)hh * a.wc]) * a.wscale) : 0.0f;
        }
    }
    for (long j0 = jb; j0 < jend; j0 += 16) {
        if (j0 >= reach) {                           /* wave-uniform: the rest is past the row */
            for (long j = j0 + lane; j < jend; j += 32) srow[j * a.scs] = -kDsInf;
            break;
        }
        const long j = j0 + (lane & 15);
        const uint16_t* kp = nullptr;
        if (j < jend && j < reach) {
            const long col = (j * a.ratio) / a.cbs;
            if (col < a.nbt) {
                const long blk = a.bt[(long)s * a.btr + col * a.btc];
                if (blk >= 0 && blk < a.nblk) kp = ds_iptr<KF8>(a, blk, j);
            }
        }
        uint4 kb[8];
        if constexpr (KF8) {
            /* columns 64 hf + 8 i: code bytes 64 hf .. + 63, scales 2 hf + i / 4 (a block of 32) */
            const unsigned char* b = (const unsigned char*)(kp ? kp : kDsZeroKey);
            uint4 r[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) r[t] = *(const uint4*)(b + 64 * hf + 16 * t);
            const uint32_t s4 = *(const uint32_t*)(b + kDsIdxScales) >> (16 * hf);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint4 w = r[i >> 1];
                kb[i] = ds_f8x8((i & 1) ? w.z : w.x, (i & 1) ? w.w : w.y, ds_ue8m0s(s4 >> (8 * (i >> 2))));
            }
        } else {
            const uint16_t* pk = (kp ? kp : kDsZeroKey) + 64 * hf;
#pragma unroll
            for (int i = 0; i < 8; ++i) kb[i] = *(const uint4*)(pk + 8 * i);
        }
        ds_v8f d[HT];
#pragma unroll
        for (int t = 0; t < HT; ++t) {
            d[t] = (ds_v8f)(0.0f);
#pragma unroll
            for (int i = 0; i < 8; ++i) d[t] = ds_wmma(qf[t][i], kb[i], d[t]);
        }
        /* heads 0-7 in lane L, 8-15 in L^16, then 16-23 back in L and 24-31 in L^16 */
        float acc = 0.0f;
#pragma unroll
        for (int t = 0; t < HT; ++t) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                if (t + half > 0) acc = ds_xor16(acc);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const float dot = ds_round_bf16(d[t][e]);
                    acc = acc + ds_round_bf16((dot < 0.0f ? 0.0f : dot) * wh[t][e]);
                }
            }
        }
        if (hf == 1 && j < jend) srow[j * a.scs] = kp ? ds_round_bf16(acc) : -kDsInf;
    }
}

/* ================================================================== launch shims */
/* Linear addressing of an index operand: rank 1 by its stride, anything else only if dense. */
long lin_stride(const RadTensor& t) {
    if (t.rank == 1) return (long)t.stride[0];
    return r4d_contig(&t) ? 1 : -1;
}

bool aligned(const void* p, unsigned bytes) { return ((uintptr_t)p & (uintptr_t)(bytes - 1)) == 0; }

constexpr int kDsChunk = 8192;                       /* index_topk's split: positions a chunk */
constexpr int kDsSplitRows = 16;                     /* ... at up to this many rows */
constexpr int kDsSplitMinChunks = 4;                 /* ... over at least this many chunks */
constexpr int kDsSplitMaxK = 2048;

/* index_candidates' keep words a row: a bit a block of `block` positions over `width`. */
long long keep_words(long long width, long long block) {
    return block < 1 || width < 1 ? 0 : ((width + block - 1) / block + 31) / 32;
}

int64_t topk_split_bytes(long rows, long C, long K) {
    const long nch = (C + kDsChunk - 1) / kDsChunk;
    if (rows < 1 || nch < kDsSplitMinChunks || K < 1 || K > kDsSplitMaxK) return 0;
    return (int64_t)rows * nch * K * 8 + (int64_t)rows * nch * 4 + 16;
}

int index_score_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 7 || !a->t) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &w = a->t[1], &kc = a->t[2], &bt = a->t[3];
    const RadTensor &pos = a->t[4], &cu = a->t[5], &sc = a->t[6];
    const long long H = r4d_geti_or(a, "heads", 0), Dm = r4d_geti_or(a, "dim", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    const long long part = r4d_geti_or(a, "row_part", -1);
    const float wscale = (float)r4d_getf(a, "wscale", 0.0);
    /* key_format e4m3_ue8m0_32: E4M3 entries (rad_kv_fp8.h), kslots bytes */
    const int kf = rad_kvf_parse(r4d_gets(a, "key_format"));
    if (kf != RAD_KVF_BF16 && kf != RAD_KVF_E4M3_UE8M0_32) return R4D_NO(RAD_E_UNSUPPORTED);
    const bool kf8 = kf == RAD_KVF_E4M3_UE8M0_32;
    const long long kslots = kf8 ? rad_kvf_bytes(kf, Dm) : Dm;
    for (int i = 0; i < 7; ++i)
        if (!a->t[i].data) return R4D_NO(RAD_E_INVAL);
    if (H < 1 || Dm < 1 || ratio < 1 || Dm % ratio || kslots < 1 || kslots % ratio ||
        !std::isfinite(wscale) || q.rank != 2 ||
        q.shape[0] < 1 || q.shape[1] != H * Dm || w.rank != 2 || w.shape[0] != q.shape[0] ||
        w.shape[1] != H || kc.rank != 4 || kc.shape[1] != 1 || kc.shape[3] * ratio != kslots ||
        kc.shape[2] % ratio || bt.rank != 2 || r4d_numel(&pos) != q.shape[0] || r4d_numel(&cu) < 2 ||
        sc.rank != 2 || sc.shape[0] < 1 || (part < 0 && sc.shape[0] != q.shape[0]) || sc.shape[1] < 1)
        return R4D_NO(RAD_E_SHAPE);
    if (q.dtype != RAD_BF16 || w.dtype != RAD_BF16 || kc.dtype != (kf8 ? RAD_F8E4M3 : RAD_BF16) ||
        bt.dtype != RAD_I32 ||
        pos.dtype != RAD_I32 || cu.dtype != RAD_I32 || sc.dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    /* This row: 128-wide keys read 16 bytes at a time, q read as bf16 pairs. */
    if (Dm != 128 || ratio > 16 || (kf8 && !rad_tensor_is_contiguous(&kc))) return R4D_NO(RAD_E_UNSUPPORTED);
    const long ps = lin_stride(pos), cs = lin_stride(cu);
    if (ps < 0 || cs < 0 || q.stride[1] != 1 || (q.stride[0] & 1) || !aligned(q.data, 4) ||
        !aligned(kc.data, 16) || q.shape[0] > INT_MAX || sc.shape[1] > INT_MAX / 2 ||
        r4d_numel(&cu) - 1 > INT_MAX || bt.shape[1] > INT_MAX || H > INT_MAX / 128)
        return R4D_NO(RAD_E_SHAPE);
    if (sc.shape[0] > INT_MAX || (part > 0 && part > (long long)INT_MAX)) return R4D_NO(RAD_E_SHAPE);
    /* A run (row_part p) scores step rows [p R, p R + R), R the score's rows: q, w and pos are read
     * from row p R on; the rows the step does not hold score -inf. */
    const long row0 = part > 0 ? (long)part * (long)sc.shape[0] : 0;
    const long Mv = row0 < (long)q.shape[0]
                        ? ((long)q.shape[0] - row0 < (long)sc.shape[0] ? (long)q.shape[0] - row0
                                                                       : (long)sc.shape[0])
                        : 0;
    const long r0 = Mv > 0 ? row0 : 0;               /* no pointer past the operands */
    DsScoreArgs s{};
    s.q = (const uint16_t*)q.data + r0 * (long)q.stride[0];
    s.w = (const uint16_t*)w.data + r0 * (long)w.stride[0];
    s.kc = (const uint16_t*)kc.data;
    s.bt = (const int*)bt.data;
    s.pos = (const int*)pos.data + r0 * ps;
    s.cu = (const int*)cu.data;
    s.sc = (float*)sc.data;
    s.row0 = row0;
    s.qr = (long)q.stride[0];
    s.wr = (long)w.stride[0];
    s.wc = (long)w.stride[1];
    s.btr = (long)bt.stride[0];
    s.btc = (long)bt.stride[1];
    s.nblk = (long)kc.shape[0];
    s.ps = ps;
    s.cs = cs;
    s.sr = (long)sc.stride[0];
    s.scs = (long)sc.stride[1];
    s.nbt = (int)bt.shape[1];
    s.S = (int)(r4d_numel(&cu) - 1);
    s.M = (int)sc.shape[0];
    s.Mv = (int)Mv;
    s.C = (int)sc.shape[1];
    s.H = (int)H;
    s.ratio = (int)ratio;
    s.cbs = (int)kc.shape[2];
    s.slot = (int)(kslots / ratio);
    s.wscale = wscale;
    /* Prefill heights on WMMA (up to 32 heads, 16-byte q rows); the decode band on the scalar
     * kernel below (libref's bits). */
    if (s.M > kWmRows && H <= 32 && s.qr % 8 == 0 && aligned(s.q, 16) &&
        (s.M + 7) / 8 <= 65535) {
        const dim3 grid((unsigned)((s.C + kIwEB - 1) / kIwEB), (unsigned)((s.M + 7) / 8));
        if (kf8) {
            if (H > 16) return R4D_EMIT(stream, (ds41_index_score_wm_kernel<2, true>), grid, dim3(256), 0u, s);
            return R4D_EMIT(stream, (ds41_index_score_wm_kernel<1, true>), grid, dim3(256), 0u, s);
        }
        if (H > 16) return R4D_EMIT(stream, (ds41_index_score_wm_kernel<2, false>), grid, dim3(256), 0u, s);
        return R4D_EMIT(stream, (ds41_index_score_wm_kernel<1, false>), grid, dim3(256), 0u, s);
    }
    /* Head split: more workgroups while the grid is short of the card. The head sum is closed in
     * head order either way, so every split gives the same bits. */
    int HS = 1;
    while (HS < 8 && HS * 2 <= H && ((long)s.C * HS * 2 + 255) / 256 * s.M <= 1024 &&
           (long)H * (256 / (HS * 2)) * 4 <= 32768)
        HS *= 2;
    const unsigned gy = (unsigned)(s.M < 65535 ? s.M : 65535);
    const int epb = 256 / HS;
    const dim3 grid((unsigned)((s.C + epb - 1) / epb), gy);
    const uint32_t lds = HS > 1 ? (uint32_t)(H * epb * 4) : 0u;
    if (kf8) {
        switch (HS) {
            case 1: return R4D_EMIT(stream, (ds41_index_score_kernel<1, true>), grid, dim3(256), lds, s);
            case 2: return R4D_EMIT(stream, (ds41_index_score_kernel<2, true>), grid, dim3(256), lds, s);
            case 4: return R4D_EMIT(stream, (ds41_index_score_kernel<4, true>), grid, dim3(256), lds, s);
            default: return R4D_EMIT(stream, (ds41_index_score_kernel<8, true>), grid, dim3(256), lds, s);
        }
    }
    switch (HS) {
        case 1: return R4D_EMIT(stream, (ds41_index_score_kernel<1, false>), grid, dim3(256), lds, s);
        case 2: return R4D_EMIT(stream, (ds41_index_score_kernel<2, false>), grid, dim3(256), lds, s);
        case 4: return R4D_EMIT(stream, (ds41_index_score_kernel<4, false>), grid, dim3(256), lds, s);
        default: return R4D_EMIT(stream, (ds41_index_score_kernel<8, false>), grid, dim3(256), lds, s);
    }
}

int index_candidates_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 3 || !a->t) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[1], &keep = a->t[2];
    const long long block = r4d_geti_or(a, "block", 0), kb = r4d_geti_or(a, "topk_blocks", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    if (!sc.data || !pos.data || !keep.data) return R4D_NO(RAD_E_INVAL);
    if (block < 1 || kb < 1 || ratio < 1 || sc.rank != 2 || sc.shape[0] < 1 || sc.shape[1] < 1 ||
        r4d_numel(&pos) != sc.shape[0] || keep.rank != 2 || keep.shape[0] != sc.shape[0] ||
        keep.shape[1] != keep_words(sc.shape[1], block))
        return R4D_NO(RAD_E_SHAPE);
    if (sc.dtype != RAD_F32 || pos.dtype != RAD_I32 || keep.dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    const long ps = lin_stride(pos);
    if (ps < 0 || sc.shape[0] > INT_MAX || sc.shape[1] > INT_MAX / 2 || block > INT_MAX)
        return R4D_NO(RAD_E_SHAPE);
    DsCandArgs c{};
    c.sc = (const float*)sc.data;
    c.pos = (const int*)pos.data;
    c.keep = (int*)keep.data;
    c.sr = (long)sc.stride[0];
    c.scs = (long)sc.stride[1];
    c.ps = ps;
    c.kr = (long)keep.stride[0];
    c.kcs = (long)keep.stride[1];
    c.M = (int)sc.shape[0];
    c.C = (int)sc.shape[1];
    c.block = (int)block;
    c.KB = (int)(kb < INT_MAX ? kb : INT_MAX);
    c.ratio = (int)(ratio < INT_MAX ? ratio : INT_MAX);
    const unsigned g = (unsigned)(c.M < 65535 ? c.M : 65535);
    return R4D_EMIT(stream, ds41_index_candidates_kernel, dim3(g), dim3(kDsSelThreads), 0, c);
}

int index_topk_params(const RadArgs* a, long long* topk, long long* ratio) {
    *topk = r4d_geti_or(a, "topk", 0);
    *ratio = r4d_geti_or(a, "ratio", 0);
    return (*topk < 1 || *ratio < 1) ? RAD_E_SHAPE : RAD_OK;
}

int index_topk_body(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 4 || !a->t) return RAD_E_INVAL;
    const RadTensor &sc = a->t[0], &pos = a->t[2], &sel = a->t[3];
    const RadTensor* keep = a->t[1].data ? &a->t[1] : nullptr;
    long long topk, ratio;
    if (index_topk_params(a, &topk, &ratio) != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const long long block = r4d_geti_or(a, "block", 0);
    if (!sc.data || !pos.data || !sel.data || sc.rank != 2 || sc.shape[0] < 1 ||
        r4d_numel(&pos) != sc.shape[0] || sel.rank != 2 || sel.shape[0] != sc.shape[0] ||
        sel.shape[1] != topk ||
        (keep && (block < 1 || block > INT_MAX || keep->rank != 2 || keep->shape[0] != sc.shape[0] ||
                  keep->shape[1] != keep_words(sc.shape[1], block))))
        return R4D_NO(RAD_E_SHAPE);
    if (sc.dtype != RAD_F32 || pos.dtype != RAD_I32 || sel.dtype != RAD_I32 ||
        (keep && keep->dtype != RAD_I32))
        return R4D_NO(RAD_E_DTYPE);
    const long ps = lin_stride(pos);
    if (ps < 0 || sc.shape[0] > INT_MAX || sc.shape[1] < 0 || sc.shape[1] > INT_MAX / 2 ||
        topk > INT_MAX / 2)
        return R4D_NO(RAD_E_SHAPE);
    DsTopkArgs t{};
    t.sc = (const float*)sc.data;
    t.keep = keep ? (const int*)keep->data : nullptr;
    t.pos = (const int*)pos.data;
    t.sel = (int*)sel.data;
    t.sr = (long)sc.stride[0];
    t.scs = (long)sc.stride[1];
    t.kr = keep ? (long)keep->stride[0] : 0;
    t.kcs = keep ? (long)keep->stride[1] : 0;
    t.ps = ps;
    t.selr = (long)sel.stride[0];
    t.selc = (long)sel.stride[1];
    t.M = (int)sc.shape[0];
    t.C = (int)sc.shape[1];
    t.K = (int)topk;
    t.ratio = (int)(ratio < INT_MAX ? ratio : INT_MAX);
    t.block = keep ? (int)block : 1;
    t.bshift = -1;
    for (int sh = 0; sh < 31; ++sh)
        if ((1LL << sh) == (long long)t.block) t.bshift = sh;
    /* Few rows over a long row: chunks in parallel, then one merge a row. The same selection, so
     * the same bits, as the one-block-a-row kernel. */
    const int64_t need = t.M <= kDsSplitRows ? topk_split_bytes(t.M, t.C, t.K) : 0;
    if (need > 0 && a->scratch && a->scratch_bytes >= need && aligned(a->scratch, 8)) {
        DsSplitArgs sp{};
        sp.t = t;
        sp.chunk = kDsChunk;
        sp.nch = (t.C + kDsChunk - 1) / kDsChunk;
        sp.cand = (unsigned long long*)a->scratch;
        sp.cnt = (int*)((unsigned char*)a->scratch + (int64_t)t.M * sp.nch * t.K * 8);
        const int rc = R4D_EMIT(stream, ds41_index_topk_split1_kernel,
                                dim3((unsigned)sp.nch, (unsigned)t.M), dim3(kDsSelThreads), 0, sp);
        if (rc != RAD_OK) return rc;
        return R4D_EMIT(stream, ds41_index_topk_split2_kernel, dim3((unsigned)t.M),
                        dim3(kDsSelThreads), 0, sp);
    }
    const unsigned g = (unsigned)(t.M < 65535 ? t.M : 65535);
    return R4D_EMIT(stream, ds41_index_topk_kernel, dim3(g), dim3(kDsSelThreads), 0, t);
}

template <int HG, int DS, bool LF8>
int attn_emit(RadStream stream, const DsAttnArgs& s, unsigned gz) {
    const uint32_t lds = (uint32_t)(ds_attn_tile_off(s.nkp, HG) + kDsTileBytes);
    const dim3 grid((unsigned)DS, (unsigned)((s.H + HG - 1) / HG), gz);
    return R4D_EMIT(stream, (ds41_attn_sparse_sink_kernel<HG, DS, LF8>), grid, dim3(256), lds, s);
}

template <int HG, int NT, int DPT, bool PVD, bool LF8>
int attn_dec_go(RadStream stream, const DsAttnArgs& s) {
    constexpr int P = 4;                                /* key tiles in flight in PV */
    const int nkq = (s.nk + 31) / 32 * 32;
    const long lds = ds_dec_lds(nkq, HG, PVD ? 0 : NT).total;
    if (lds > 65536) return RAD_E_UNSUPPORTED;
    const dim3 grid((unsigned)((s.H + HG - 1) / HG), (unsigned)s.M);
    return R4D_EMIT(stream, (ds41_attn_dec_kernel<HG, NT, DPT, P, PVD, LF8>), grid, dim3(NT), (uint32_t)lds, s);
}

/* The decode band's kernel. The form by how many (row, head) pairs the pass has, any choice giving
 * the same bits: a smaller head group for fewer pairs, a larger one as the pairs grow, each faster
 * at its band than the scalar kernel (by several times at one row). On an LDS refusal (long rows),
 * the next smaller form; the one-head form takes every row this op accepts. */
template <bool LF8>
int attn_dec_emit(RadStream stream, const DsAttnArgs& s) {
    if (s.M > 65535 || (s.H + 0L) > 65535L * 8) return RAD_E_UNSUPPORTED;
    const long hr = (long)s.M * s.H;
    int c = hr <= 32 ? 0 : (hr <= 64 ? 1 : (hr <= 128 ? 2 : (hr <= 256 ? 3 : 4)));
    static const int next[8] = { -1, 0, 1, 2, 5, 6, 1, -1 };
    /* An E4M3 latent takes the tiled PV forms only (its loads stay branch-free there), one head
     * group size a band. */
    static const int tiled[5] = { 7, 6, 5, 4, 4 };
    static const int next8[8] = { -1, -1, -1, -1, 5, 6, 7, -1 };
    if (LF8) c = tiled[c];
    while (c >= 0) {
        int rc = RAD_E_UNSUPPORTED;
        if constexpr (LF8) {
            switch (c) {
                case 4: rc = attn_dec_go<8, 512, 2, false, true>(stream, s); break;
                case 5: rc = attn_dec_go<4, 512, 2, false, true>(stream, s); break;
                case 6: rc = attn_dec_go<2, 512, 2, false, true>(stream, s); break;
                case 7: rc = attn_dec_go<1, 512, 2, false, true>(stream, s); break;
                default: break;
            }
        } else {
            switch (c) {
                case 0: rc = attn_dec_go<1, 1024, 2, true, false>(stream, s); break;
                case 1: rc = attn_dec_go<2, 1024, 2, true, false>(stream, s); break;
                case 2: rc = attn_dec_go<4, 1024, 2, true, false>(stream, s); break;
                case 3: rc = attn_dec_go<8, 1024, 2, true, false>(stream, s); break;
                case 4: rc = attn_dec_go<8, 512, 2, false, false>(stream, s); break;
                case 5: rc = attn_dec_go<4, 512, 2, false, false>(stream, s); break;
                case 6: rc = attn_dec_go<2, 512, 2, false, false>(stream, s); break;
                default: break;
            }
        }
        if (rc != RAD_E_UNSUPPORTED) return rc;
        c = LF8 ? next8[c] : next[c];
    }
    return RAD_E_UNSUPPORTED;
}

/* The scalar kernel's head group and d split by how many workgroups the rows give. */
template <bool LF8>
int attn_scalar_emit(RadStream stream, const DsAttnArgs& s) {
    const long rows = s.M;
    int HG = 8, DS = 1;
    if (rows * s.H < 2048) HG = rows * s.H >= 512 ? 4 : 2;
    if (HG == 2) DS = 2;
    while (HG > 1 && ds_attn_tile_off(s.nkp, HG) + kDsTileBytes > 65536) HG /= 2;
    if (ds_attn_tile_off(s.nkp, HG) + kDsTileBytes > 65536) return R4D_NO(RAD_E_UNSUPPORTED);
    const unsigned gz = (unsigned)(s.M < 65535 ? s.M : 65535);
    if (DS == 2) {
        if (HG == 2) return attn_emit<2, 2, LF8>(stream, s, gz);
        return attn_emit<1, 2, LF8>(stream, s, gz);
    }
    switch (HG) {
        case 8: return attn_emit<8, 1, LF8>(stream, s, gz);
        case 4: return attn_emit<4, 1, LF8>(stream, s, gz);
        case 2: return attn_emit<2, 1, LF8>(stream, s, gz);
        default: return attn_emit<1, 1, LF8>(stream, s, gz);
    }
}

int attn_body(const RadArgs* a, RadStream stream) {
    /* Operands 4-6 are optional (null without a compressed source), as in libref. */
    if (!a || a->n_t < 11 || !a->t) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &wc = a->t[1], &wbt = a->t[2], &wused = a->t[3];
    const RadTensor* cc = a->t[4].data ? &a->t[4] : nullptr;
    const RadTensor* cbt = a->t[5].data ? &a->t[5] : nullptr;
    const RadTensor* sel = a->t[6].data ? &a->t[6] : nullptr;
    const RadTensor &pos = a->t[7], &cu = a->t[8], &sink = a->t[9], &out = a->t[10];
    const long long H = r4d_geti_or(a, "heads", 0), D = r4d_geti_or(a, "dim", 0);
    const long long win = r4d_geti_or(a, "window", 0), ratio = r4d_geti_or(a, "ratio", 0);
    const long long topk = r4d_geti_or(a, "topk", 0);
    const float scale = (float)r4d_getf(a, "scale", 0.0);
    const bool comp = ratio > 0;
    /* latent_format e4m3_e4m3_16: the compressed entries as E4M3 (rad_kv_fp8.h), cslots bytes */
    const int lf = rad_kvf_parse(r4d_gets(a, "latent_format"));
    if (lf != RAD_KVF_BF16 && lf != RAD_KVF_E4M3_E4M3_16) return R4D_NO(RAD_E_UNSUPPORTED);
    const bool lf8 = comp && lf == RAD_KVF_E4M3_E4M3_16;
    const long long cslots = lf8 ? rad_kvf_bytes(lf, D) : D;
    if (H < 1 || D < 2 || D > 4096 || win < 1 || ratio < 0 || topk < 0 || !std::isfinite(scale) ||
        scale <= 0 || (comp && (!cc || !cbt || !sel || topk < 1 || D % ratio || cslots < 1 ||
                                cslots % ratio)))
        return R4D_NO(RAD_E_SHAPE);
    if (!q.data || !wc.data || !wbt.data || !wused.data || !pos.data || !cu.data || !sink.data ||
        !out.data || q.rank != 2 || q.shape[0] < 1 || q.shape[1] != H * D || out.rank != 2 ||
        out.shape[0] != q.shape[0] || out.shape[1] != H * D || wc.rank != 4 || wc.shape[1] != 1 ||
        wc.shape[3] != D || wbt.rank != 2 || r4d_numel(&pos) != q.shape[0] || r4d_numel(&sink) != H ||
        sink.dtype != RAD_F32 || r4d_numel(&cu) < 2 || r4d_numel(&wused) < r4d_numel(&cu) - 1)
        return R4D_NO(RAD_E_SHAPE);
    const long long cbs = comp ? cc->shape[2] : 0;
    if (comp && (cc->rank != 4 || cc->shape[1] != 1 || cc->shape[3] * ratio != cslots || cbs % ratio ||
                 cbt->rank != 2 || sel->rank != 2 || sel->shape[0] != q.shape[0] ||
                 sel->shape[1] != topk || cbt->shape[0] != wbt.shape[0]))
        return R4D_NO(RAD_E_SHAPE);
    if (q.dtype != RAD_BF16 || wc.dtype != RAD_BF16 || out.dtype != RAD_BF16 || wbt.dtype != RAD_I32 ||
        wused.dtype != RAD_I32 || pos.dtype != RAD_I32 || cu.dtype != RAD_I32 ||
        (comp && (cc->dtype != (lf8 ? RAD_F8E4M3 : RAD_BF16) || cbt->dtype != RAD_I32 ||
                  sel->dtype != RAD_I32)))
        return R4D_NO(RAD_E_DTYPE);
    if (lf8 && !rad_tensor_is_contiguous(cc)) return R4D_NO(RAD_E_SHAPE);
    /* This row: the 512-wide latent, read 16 bytes at a time; at most 1024 + 2048 keys a row. */
    if (D != kDsAttnD || win > 1024 || topk > 2048 || ratio > 64) return R4D_NO(RAD_E_UNSUPPORTED);
    const long S = (long)(r4d_numel(&cu) - 1);
    const long ps = lin_stride(pos), cs = lin_stride(cu), wus = lin_stride(wused), sks = lin_stride(sink);
    if (ps < 0 || cs < 0 || wus < 0 || sks < 0 || q.stride[1] != 1 || out.stride[1] != 1 ||
        (q.stride[0] & 1) || (out.stride[0] & 1) || !aligned(q.data, 4) || !aligned(out.data, 4) ||
        !aligned(wc.data, 16) || (comp && !aligned(cc->data, 16)) || wbt.shape[0] < S ||
        q.shape[0] > INT_MAX || S > INT_MAX || wbt.shape[1] > INT_MAX ||
        (comp && cbt->shape[1] > INT_MAX) || wc.shape[2] < 1 || wc.shape[2] > INT_MAX || cbs > INT_MAX)
        return R4D_NO(RAD_E_SHAPE);
    DsAttnArgs s{};
    s.q = (const uint16_t*)q.data;
    s.wc = (const uint16_t*)wc.data;
    s.wbt = (const int*)wbt.data;
    s.wused = (const int*)wused.data;
    s.cc = comp ? (const uint16_t*)cc->data : nullptr;
    s.cbt = comp ? (const int*)cbt->data : nullptr;
    s.sel = comp ? (const int*)sel->data : nullptr;
    s.pos = (const int*)pos.data;
    s.cu = (const int*)cu.data;
    s.sink = (const float*)sink.data;
    s.out = (uint16_t*)out.data;
    s.qr = (long)q.stride[0];
    s.outr = (long)out.stride[0];
    s.wbtr = (long)wbt.stride[0];
    s.wbtc = (long)wbt.stride[1];
    s.nwblk = (long)wc.shape[0];
    s.cbtr = comp ? (long)cbt->stride[0] : 0;
    s.cbtc = comp ? (long)cbt->stride[1] : 0;
    s.ncblk = comp ? (long)cc->shape[0] : 0;
    s.selr = comp ? (long)sel->stride[0] : 0;
    s.selc = comp ? (long)sel->stride[1] : 0;
    s.wus = wus;
    s.ps = ps;
    s.cs = cs;
    s.sks = sks;
    s.nwbt = (int)wbt.shape[1];
    s.bs = (int)wc.shape[2];
    s.ncbt = comp ? (int)cbt->shape[1] : 0;
    s.cbs = comp ? (int)cbs : 1;
    s.ratio = comp ? (int)ratio : 1;
    s.S = (int)S;
    s.M = (int)q.shape[0];
    s.H = (int)H;
    s.win = (int)win;
    s.nk = (int)(win + (comp ? topk : 0));
    s.nkp = (s.nk + 3) / 4 * 4;
    s.cslot = comp ? (int)(cslots / ratio) : 0;
    s.scale = scale;
    /* Prefill heights on WMMA (16-byte q and out rows); the decode band on the scalar kernels
     * below (libref's bits). */
    {
        const int nkt = (s.nk + 15) / 16;
        const long lds = ds_wm_sbuf_off(nkt) + kWmSbufBytes + (lf8 ? kWmWaves * kWmTrBytes : 0);
        if (s.M > kWmRows && s.qr % 8 == 0 && s.outr % 8 == 0 && aligned(q.data, 16) &&
            aligned(out.data, 16) && lds <= 65536 && (H + 15) / 16 <= 65535) {
            const unsigned gy = (unsigned)(s.M < 65535 ? s.M : 65535);
            const dim3 grid((unsigned)((H + 15) / 16), gy);
            const float sl2 = (float)((double)scale * 1.4426950408889634);
            if (lf8)
                return R4D_EMIT(stream, (ds41_attn_sparse_sink_wm_kernel<true, true>), grid, dim3(128),
                                (uint32_t)lds, s, sl2);
            return R4D_EMIT(stream, (ds41_attn_sparse_sink_wm_kernel<true, false>), grid, dim3(128),
                            (uint32_t)lds, s, sl2);
        }
    }
    /* Head group and d split by how many workgroups the rows give. Every choice sums each output
     * in the same order, so the bits do not depend on it. */
    const long rows = s.M;
    if (rows <= kWmRows) {
        const int rc = lf8 ? attn_dec_emit<true>(stream, s) : attn_dec_emit<false>(stream, s);
        if (rc != RAD_E_UNSUPPORTED) return rc;
    }
    return lf8 ? attn_scalar_emit<true>(stream, s) : attn_scalar_emit<false>(stream, s);
}

}  // namespace

extern "C" int r4d_rad_ds41_index_score(const RadArgs* a, RadStream stream) {
    return index_score_body(a, stream);
}
extern "C" int r4d_rad_ds41_index_candidates(const RadArgs* a, RadStream stream) {
    return index_candidates_body(a, stream);
}
extern "C" int r4d_rad_ds41_index_topk(const RadArgs* a, RadStream stream) {
    return index_topk_body(a, stream);
}
/* Called at declare with the band's bound and no tensors: the split's partials for up to
 * kDsSplitRows rows of `width` positions; 0 (no split) when a row is short. */
extern "C" int64_t r4d_rad_ds41_index_topk_scratch(const RadArgs* a) {
    long long topk, ratio;
    if (index_topk_params(a, &topk, &ratio) != RAD_OK) return 0;
    long long M = r4d_geti_or(a, "M", kDsSplitRows);
    if (M < 1 || M > kDsSplitRows) M = kDsSplitRows;
    return topk_split_bytes((long)M, (long)r4d_geti_or(a, "width", 0), (long)topk);
}
extern "C" int r4d_rad_ds41_attn_sparse_sink(const RadArgs* a, RadStream stream) {
    return attn_body(a, stream);
}
