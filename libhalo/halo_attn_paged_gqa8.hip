/* Derived from libr4d/r4d_attn_paged_gqa8.hip by the libhalo rename transform (2026-10-26), then
 * ported to gfx1151; see libhalo/README.md. */
// Halo paged attention dispatch at eight queries per KV head, head_dim 256 and 128.
//
// Instantiations of the h256/gqa6 kernel templates (templated on head_dim and gqa), for the gqa-8
// models; the constraint tables gate on the exact ratio:
//
//   Qwen3.6-35B-A3B   16 query heads over 2 KV heads, head_dim 256
//   MiniCPM5-2B       16 query heads over 2 KV heads, head_dim 128
//
// Gqa changes only two bounds. A decode workgroup holds 64 rows of q_len * gqa, so the verify
// width is 8 here (10 at gqa 6); the selector bands on that value, stated in the row. A prefill
// block carries NWARPS * 16 / gqa query positions (48 here, 64 at gqa 6).
//
// Head_dim changes how K is fetched. O_KWIDE's thread-to-(key, channel) map covers exactly 16 x 16
// channels (head_dim 256); at 128 it would stage past the tile, so the h128 prefill runs without it.
// All other option bits are as defined in halo_attn_prefill_h256_gqa6.hip.
//
// gfx1151: both bands. The split-KV decode kernel is a body of its own
// (halo_attn_decode_h256_gqa6.hip); this unit includes it, instantiates it at this ratio and defines
// the ratio's decode entry points, plain and gated.
//
// No halo_attn_dims here: that entry point belongs to the h256/gqa6 unit, and a second definition
// would not link. Geometry is gated by the row constraints and the launchers below.
#include "halo.h"
#include "halo_args.h"
#include "halo_attn_args.h"
#include "halo_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "halo_attn_prefill_h256_gqa6.hip"
// The split-KV decode body. Both bands of a geometry belong to its unit, and this one is the unit
// for every ratio that does not have a file of its own (this one, and the wrappers that define
// HALO_ATTN_GQA before including it).
#include "halo_attn_decode_h256_gqa6.hip"

#define CHK8() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

/* Queries per KV head, and the symbol names built from it.
 *
 * Another ratio reuses this file: it defines these macros and includes this one (e.g.
 * halo_attn_paged_h256_gqa12.hip), and the entry points paste the ratio into their names. Only
 * the two bounds change: decode verify width 64/gqa and prefill block 384/gqa query positions.
 * Both must divide, and both are stated in the row so the selector bands on them. */
#ifndef HALO_ATTN_GQA
#define HALO_ATTN_GQA  8
#define HALO_ATTN_H128 1      /* this ratio also serves head_dim 128 -- MiniCPM5-2B is 16 over 2 */
#endif
/* Head_dim 64 prefill leg, off unless the including unit enables it (LFM2: 32 query heads over 8,
 * halo_attn_paged_h256_gqa4.hip). Uses the h128 option set (no O_KWIDE). */
#ifndef HALO_ATTN_H64
#define HALO_ATTN_H64 0
#endif
/* Head_dim 256 leg, on unless the including unit serves only head_dim 128. */
#ifndef HALO_ATTN_H256
#define HALO_ATTN_H256 1
#endif
/* The fp8_v4 and k6_v4 caches at head_dim 256, off unless the including unit serves a model whose
 * plugin offers them: Qwen3.8-Flash-Next's QSA layers (gqa12, and gqa6 at four ranks). */
#ifndef HALO_ATTN_V4
#define HALO_ATTN_V4 0
#endif
/* The fused attn_paged_gate_quant rows on the decode leg (gate, codes and scales as the merge's
 * epilogue), off unless the including unit's model declares them beside its unfused pair: the QSA
 * layers above, at both ratios. */
#ifndef HALO_ATTN_GQ
#define HALO_ATTN_GQ 0
#endif
/* The union form of the gated decode rows at prefill (g8_union_launch below), off unless the
 * including unit serves the QSA sparse attention (gqa12 over 4-token pages). */
#ifndef G8_UNION
#define G8_UNION 0
#endif
#define GQ_JOIN_(a, b, c) a##b##c
#define GQ_JOIN(a, b, c)  GQ_JOIN_(a, b, c)
#define GQ_SYM(pre, suf)  GQ_JOIN(pre, HALO_ATTN_GQA, suf)

#define G8_GQA         HALO_ATTN_GQA
/* KV page size in tokens; the kernel body is generic in it, so an including unit may override.
 * Qwen4-Exp's QSA uses 4: its indexer selects 4-token blocks, so the selection is just a block
 * table (docs/QSA.md). */
#ifndef G8_BLOCK_SIZE
#define G8_BLOCK_SIZE 16
#endif

#define G8_P_NWARPS 24
#define G8_P_TILE   48
// K prefetch 2 at head_dim 256, as in the h256/gqa6 prefill (P_OPT there): 4 spills the m-tile
// loop. The head_dim-128 set keeps prefetch 4, which fits.
#define G8_P_OPT_256 3954187
#define G8_P_OPT_128 3430923        // without O_KWIDE, prefetch 4
#define G8_P_OPT_64  G8_P_OPT_128

// ---- prefill -----------------------------------------------------------------------------------
template <int HEAD_DIM, int OPT, int KVP>
static int g8_prefill_launch(const HaloArgs* a, hipStream_t s) {
    if (a->head_dim != HEAD_DIM || a->block_size != G8_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != G8_GQA) return -2;
    // Split-KV when the caller set a->splits > 1 with scratch (halo_prefill_split_setup).
    return halo_prefill_launch<G8_P_NWARPS, G8_P_TILE, HEAD_DIM, G8_GQA, G8_BLOCK_SIZE, KVP, OPT>(
        a, s);
}
#define G8_SPLIT(HD, a, r) halo_prefill_split_setup<G8_P_NWARPS, G8_P_TILE, HD, G8_GQA>((a), (r))

// ============================================================== the plugin ABI entry points

/* The split-KV partials of the prefill rows (halo_prefill_scratch in the body). */
#if HALO_ATTN_H256
extern "C" int64_t GQ_SYM(halo_rad_attn_prefill_h256_gqa, _scratch)(const RadArgs* a) {
    return halo_prefill_scratch<G8_P_NWARPS, G8_P_TILE, 256, G8_GQA>(a);
}
#endif
#if HALO_ATTN_H128
extern "C" int64_t GQ_SYM(halo_rad_attn_prefill_h128_gqa, _scratch)(const RadArgs* a) {
    return halo_prefill_scratch<G8_P_NWARPS, G8_P_TILE, 128, G8_GQA>(a);
}
#endif
#if HALO_ATTN_H64
extern "C" int64_t GQ_SYM(halo_rad_attn_prefill_h64_gqa, _scratch)(const RadArgs* a) {
    return halo_prefill_scratch<G8_P_NWARPS, G8_P_TILE, 64, G8_GQA>(a);
}
#endif

#if HALO_ATTN_H256
extern "C" int GQ_SYM(halo_rad_attn_prefill_h256_gqa, _fp8)(const RadArgs* a, RadStream s) {
    HaloArgs r;
    const int rc = fill_paged(a, &r, 256);
    if (rc != RAD_OK) return rc;
    G8_SPLIT(256, a, &r);
    /* Absent descales mean 1.0, as on the gqa6 leg. */
    return g8_prefill_launch<256, G8_P_OPT_256, 0>(&r, (hipStream_t)s);
}
extern "C" int GQ_SYM(halo_rad_attn_prefill_h256_gqa, _bf16)(const RadArgs* a, RadStream s) {
    HaloArgs r;
    const int rc = fill_paged(a, &r, 256);
    if (rc != RAD_OK) return rc;
    G8_SPLIT(256, a, &r);
    return g8_prefill_launch<256, G8_P_OPT_256, 1>(&r, (hipStream_t)s);
}
#endif
#if HALO_ATTN_H128
extern "C" int GQ_SYM(halo_rad_attn_prefill_h128_gqa, _fp8)(const RadArgs* a, RadStream s) {
    HaloArgs r;
    const int rc = fill_paged(a, &r, 128);
    if (rc != RAD_OK) return rc;
    G8_SPLIT(128, a, &r);
    return g8_prefill_launch<128, G8_P_OPT_128, 0>(&r, (hipStream_t)s);
}
extern "C" int GQ_SYM(halo_rad_attn_prefill_h128_gqa, _bf16)(const RadArgs* a, RadStream s) {
    HaloArgs r;
    const int rc = fill_paged(a, &r, 128);
    if (rc != RAD_OK) return rc;
    G8_SPLIT(128, a, &r);
    return g8_prefill_launch<128, G8_P_OPT_128, 1>(&r, (hipStream_t)s);
}
#endif
#if HALO_ATTN_H64
extern "C" int GQ_SYM(halo_rad_attn_prefill_h64_gqa, _fp8)(const RadArgs* a, RadStream s) {
    HaloArgs r;
    const int rc = fill_paged(a, &r, 64);
    if (rc != RAD_OK) return rc;
    G8_SPLIT(64, a, &r);
    return g8_prefill_launch<64, G8_P_OPT_64, 0>(&r, (hipStream_t)s);
}
extern "C" int GQ_SYM(halo_rad_attn_prefill_h64_gqa, _bf16)(const RadArgs* a, RadStream s) {
    HaloArgs r;
    const int rc = fill_paged(a, &r, 64);
    if (rc != RAD_OK) return rc;
    G8_SPLIT(64, a, &r);
    return g8_prefill_launch<64, G8_P_OPT_64, 1>(&r, (hipStream_t)s);
}
#endif

#if HALO_ATTN_H256 && HALO_ATTN_V4
/* ---- the fp8_v4 cache (a token's row 25 * 256 / 16 = 400 bytes: E4M3 K, int4 V, f16 scales) and
 * the k6_v4 cache (352 bytes: int6 K, int4 V), prefill. */
#define G8_V4_ENTRIES(TAG, ROWB, XOPT)                                                         \
extern "C" int GQ_SYM(halo_rad_attn_prefill_h256_gqa, _##TAG)(const RadArgs* a, RadStream s) { \
    HaloArgs r;                                                                                \
    const int rc = fill_paged_row(a, &r, 256, ROWB);                                           \
    if (rc != RAD_OK) return rc;                                                               \
    G8_SPLIT(256, a, &r);                                                                      \
    return g8_prefill_launch<256, G8_P_OPT_256 | (XOPT), 0>(&r, (hipStream_t)s);               \
}                                                                                              \
extern "C" int64_t GQ_SYM(halo_rad_attn_prefill_h256_gqa, _##TAG##_scratch)(const RadArgs* a) {\
    return halo_prefill_scratch<G8_P_NWARPS, G8_P_TILE, 256, G8_GQA>(a, ROWB);                 \
}
G8_V4_ENTRIES(fp8v4, 400, O_V4)
G8_V4_ENTRIES(k6v4, 352, O_V4 | O_K6)
#undef G8_V4_ENTRIES
#endif

// ============================================================== the split-KV decode leg
//
// The h256/gqa6 split law and its merge constants (halo_attn_decode_h256_gqa6.hip). Gqa changes two
// things here: the decode verify width (64 / gqa query rows a workgroup) and the ratio the launch
// gates on. The option set is the h256 decode set for every geometry: libr4d's h128 decode set adds
// O_KRAW, a gfx12 K round the port does not instantiate, and its GPREV turns itself off at head_dim
// 128 anyway (a V prefetch there is past the three registers it fits in).
#define G8_D_OPT    HALO_D_OPT
#define G8_MAX_DECODE_ROWS 64       // q_len * gqa a single decode workgroup can hold

template <int HEAD_DIM, int KVP, int XOPT = 0>
/* The attention kernel and the merge; `gq` non-null adds the output gate and the quantiser to the
 * merge's epilogue (attn_paged_gate_quant). XOPT adds the cache-layout bits (O_V4, O_K6). */
static int g8_decode_launch(const HaloArgs* a, RadStream stream,
                            const HaloGateQuant* gq = nullptr) {
    return decode_launch<HEAD_DIM, G8_GQA, G8_BLOCK_SIZE, G8_D_OPT | XOPT, KVP, 0>(
        a, (hipStream_t)stream, gq);
}

/* Split-KV partial buffer size at declare time, from the band's upper bound (decode_scratch in the
 * body): parameters first, tensors the fallback. */
#if G8_UNION
static int64_t g8_union_scratch(const RadArgs* a);
#endif
static int64_t g8_decode_scratch(const RadArgs* a, int head_dim, long long row = 0) {
#if G8_UNION
    {
        const int64_t base = decode_scratch<256, G8_GQA, G8_BLOCK_SIZE, G8_D_OPT>(a, row);
        if (base < 0 || head_dim != 256) return base;
        return std::max<int64_t>(base, g8_union_scratch(a));
    }
#endif
#if HALO_ATTN_H256 && HALO_ATTN_H128
    return head_dim == 256 ? decode_scratch<256, G8_GQA, G8_BLOCK_SIZE, G8_D_OPT>(a, row)
                           : decode_scratch<128, G8_GQA, G8_BLOCK_SIZE, G8_D_OPT>(a, row);
#elif HALO_ATTN_H256
    (void)head_dim;
    return decode_scratch<256, G8_GQA, G8_BLOCK_SIZE, G8_D_OPT>(a, row);
#else
    (void)head_dim;
    return decode_scratch<128, G8_GQA, G8_BLOCK_SIZE, G8_D_OPT>(a, row);
#endif
}

static int g8_decode_common(const RadArgs* a, HaloArgs* r, int head_dim, long long row = 0) {
#if HALO_ATTN_H256 && HALO_ATTN_H128
    return head_dim == 256 ? decode_common<256, G8_GQA, G8_BLOCK_SIZE, G8_D_OPT>(a, r, row)
                           : decode_common<128, G8_GQA, G8_BLOCK_SIZE, G8_D_OPT>(a, r, row);
#elif HALO_ATTN_H256
    (void)head_dim;
    return decode_common<256, G8_GQA, G8_BLOCK_SIZE, G8_D_OPT>(a, r, row);
#else
    (void)head_dim;
    return decode_common<128, G8_GQA, G8_BLOCK_SIZE, G8_D_OPT>(a, r, row);
#endif
}

#if HALO_ATTN_GQ && HALO_ATTN_H256
/* ---- attn_paged_gate_quant: attn_paged's eight operands, then the gate, the codes and their
 * per-128-column scales. A gated row's merge row is one (token, head) pair, so the gate is a
 * [T, n_head, head_dim] view of the [q|gate] projection whose (token, head) rows are one pitch
 * apart when the view narrowed the last axis alone (gate_quant_fp8's test, repeated here because
 * this unit cannot link that unit). */
enum { AP_GQ_G = AP_OUT + 1, AP_GQ_Q, AP_GQ_S };

static long long g8_gq_row_pitch(const RadTensor* t, long long width) {
    if (!t || t->rank < 1) return -1;
    if (t->shape[t->rank - 1] != width) return halo_contig(t) ? width : -1;
    if (t->stride[t->rank - 1] != 1) return -1;
    if (t->rank == 1) return width;
    for (int i = (int)t->rank - 2; i >= 1; --i)
        if (t->stride[i - 1] != t->shape[i] * t->stride[i]) return -1;
    return t->stride[t->rank - 2];
}

/* The gate/codes/scales operands, validated as gate_quant_fp8 validates them: a bf16 gate, E4M3 (or
 * int8, for an int8 out projection) codes, f32 scales, act sigmoid, a scale group of 128. */
static int g8_gate_quant_args(const RadArgs* a, const HaloArgs& r, HaloGateQuant* gq) {
    const RadTensor* g = halo_opt(a, AP_GQ_G);
    const RadTensor* q = halo_opt(a, AP_GQ_Q);
    const RadTensor* s = halo_opt(a, AP_GQ_S);
    if (!g || !q || !s) return HALO_NO(RAD_E_INVAL);
    if (g->dtype != RAD_BF16 || (q->dtype != RAD_F8E4M3 && q->dtype != RAD_I8) ||
        s->dtype != RAD_F32)
        return HALO_NO(RAD_E_DTYPE);
    const char* act = halo_gets(a, "act");
    if (!act || std::strcmp(act, "sigmoid")) return HALO_NO(RAD_E_UNSUPPORTED);
    if (halo_geti_or(a, "group", kHaloGqGroup) != kHaloGqGroup) return HALO_NO(RAD_E_SHAPE);
    const long long n = r.head_dim;
    if (n % kHaloGqGroup) return HALO_NO(RAD_E_SHAPE);
    const long long nkb  = n / kHaloGqGroup;
    const long long rows = (long long)halo_q_total(r) * r.q_heads;
    if (halo_numel(g) < rows * n || halo_numel(q) < rows * n || halo_numel(s) < rows * nkb)
        return HALO_NO(RAD_E_SHAPE);
    const long long gp = g8_gq_row_pitch(g, n), qp = g8_gq_row_pitch(q, n),
                    sp = g8_gq_row_pitch(s, nkb);
    if (gp < 0 || qp < 0 || sp < 0) return HALO_NO(RAD_E_STRIDE);
    /* The vector merge reads a lane's gate in one 16-byte access and stores its codes in one 8-byte
     * access. */
    if ((halo_addr(g) & 15) || (gp % 8) || (halo_addr(q) & 7) || (qp % 8))
        return HALO_NO(RAD_E_STRIDE);
    gq->g    = (const uint16_t*)halo_addr(g);
    gq->q    = (uint8_t*)halo_addr(q);
    gq->s    = (float*)halo_addr(s);
    gq->g_ld = (unsigned)gp;
    gq->q_ld = (unsigned)qp;
    gq->s_ld = (unsigned)sp;
    gq->i8   = q->dtype == RAD_I8 ? 1 : 0;
    return RAD_OK;
}
#endif  /* HALO_ATTN_GQ && HALO_ATTN_H256 */


#if G8_UNION && HALO_ATTN_GQ && HALO_ATTN_H256
/* ---- the QSA prefill form: query rows in groups, each group's pages read once -------------------
 *
 * At prefill the sparse attention is issued one row per query (docs/QSA.md), each row with its own
 * 513-entry selection, and the decode kernel serves a row alone: 12 live rows of a 32-row tile pair
 * and every selected page widened for them, so a 2048-row launch moved ~4.3 GB of K/V through the
 * caches for 103 GFLOP. Consecutive queries select mostly the same pages (on Flash-Next prose, five
 * consecutive rows' selections have ~1.5x one row's pages in their union), so here five rows are
 * one workgroup's "sequence" (60 of the kernel's 64 rows) over the union of their pages:
 *
 *   1. g8_union_plan_kernel, a block a group: the union in a fixed order (the rows in turn, each
 *      row's pages not yet in it in that row's table order; an LDS hash maps page -> union index)
 *      and a word a union page, four bits a row: the tokens of the page the row attends to (4, the
 *      row's own count of its last, partial page, or 0 when it did not select the page).
 *   2. halo_attn_decode_kernel under O_UNION over (group, kv head), one split: the mask replaces
 *      the causal limit. 13.6 -> 6.5 ms a 2048-row call on an 8.9K prompt.
 *   3. the gated merge, per row, unchanged (each row's one partial).
 *
 * A row's softmax runs over its selected keys in union order rather than table order, so the bits
 * differ from the per-row form within rounding; the set, the scale and the epilogue are the same.
 * The page ids need not be ordered (a pool's free list need not be): the union is by lookup. */
constexpr int kG8UnionR = 5;            /* rows a group: 60 of the 64-row workgroup at gqa 12 */
constexpr int kG8UnionMinRows = 64;     /* below this the per-row form (decode) */
constexpr int kG8UnionMaxSp = 520;      /* the selection's pitch it plans for (topk + 1 = 513) */
constexpr int kG8UnionHash = 4096;      /* LDS hash slots: >= 1.5x the largest union */
/* Keys a staging round: 32 ran ~1% faster end to end than 16 (8.9K prefill, 1590 against 1580
 * tok/s); in isolation the two are within 2%. */
#ifndef G8_UNION_TILE
#define G8_UNION_TILE 32
#endif
constexpr int kG8UnionTile = G8_UNION_TILE;
constexpr int kG8UnionUp = (kG8UnionR * kG8UnionMaxSp + 7) & ~7;

struct G8UnionPlan {
    int* u; uint32_t* m; int* gctx; int* cu; int* fs;
    long long bytes;
};
static G8UnionPlan g8_union_plan(void* base, long long rows, int up) {
    const long long G = (rows + kG8UnionR - 1) / kG8UnionR;
    G8UnionPlan p;
    int* w = (int*)base;
    p.u = w;                       w += G * up;
    p.m = (uint32_t*)w;            w += G * up;
    p.gctx = w;                    w += G;
    p.cu = w;                      w += G + 1;
    p.fs = w;                      w += rows;
    p.bytes = (long long)((char*)w - (char*)base);
    return p;
}

__device__ __forceinline__ uint32_t g8_union_hash(int pg) {
    return ((uint32_t)pg * 2654435761u) >> (32 - 12);
}
static_assert(kG8UnionHash == 4096, "g8_union_hash keeps 12 bits");

__global__ __launch_bounds__(256) void g8_union_plan_kernel(
    const int* __restrict__ sel, int sp, const int* __restrict__ su, int rows, int up,
    int* __restrict__ U, uint32_t* __restrict__ M, int* __restrict__ gctx, int* __restrict__ cu,
    int* __restrict__ fs, int fsv) {
    __shared__ int hk[kG8UnionHash], hv[kG8UnionHash];
    __shared__ uint32_t sm[kG8UnionUp];
    __shared__ int wcnt[8];
    __shared__ int s_base;
    const int g = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    for (int i = tid; i < kG8UnionHash; i += 256) hk[i] = -1;
    for (int i = tid; i < up; i += 256) sm[i] = 0u;
    if (tid == 0) s_base = 0;
    __syncthreads();
    auto lookup = [&](int pg) {
        uint32_t h = g8_union_hash(pg);
        for (;;) {
            const int k = hk[h];
            if (k == pg) return hv[h];
            if (k == -1) return -1;
            h = (h + 1) & (kG8UnionHash - 1);
        }
    };
    /* The union, row by row: a chunk's new pages are numbered by a block scan in table order. */
    for (int r = 0; r < kG8UnionR; ++r) {
        const int row = g * kG8UnionR + r;
        if (row >= rows) break;
        const int c = su[row];
        const int n = c > 0 ? min((c + 3) / 4, sp) : 0;
        for (int j0 = 0; j0 < n; j0 += 256) {
            const int j = j0 + tid;
            const int pg = j < n ? sel[(size_t)row * sp + j] : -1;
            const bool isnew = pg >= 0 && lookup(pg) < 0;
            const unsigned bal = (unsigned)__ballot(isnew);
            if (lane == 0) wcnt[wave] = __popc(bal);
            __syncthreads();
            int pre = 0, tot = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) { pre += w < wave ? wcnt[w] : 0; tot += wcnt[w]; }
            if (isnew) {
                const int idx = s_base + pre + __popc(bal & ((1u << lane) - 1u));
                U[(size_t)g * up + idx] = pg;
                uint32_t h = g8_union_hash(pg);
                for (;;) {
                    const int prev = atomicCAS(&hk[h], -1, pg);
                    if (prev == -1) { hv[h] = idx; break; }
                    h = (h + 1) & (kG8UnionHash - 1);
                }
            }
            __syncthreads();
            if (tid == 0) s_base += tot;
            __syncthreads();
        }
    }
    const int cnt = s_base;
    /* Each row's tokens of each of its pages, four bits a row. */
    for (int r = 0; r < kG8UnionR; ++r) {
        const int row = g * kG8UnionR + r;
        if (row >= rows) break;
        const int c = su[row];
        const int n = c > 0 ? min((c + 3) / 4, sp) : 0;
        for (int j = tid; j < n; j += 256) {
            const int pg = sel[(size_t)row * sp + j];
            if (pg < 0) continue;
            const int u = lookup(pg);
            const int tk = j == n - 1 ? min(c - 4 * j, 4) : 4;
            atomicOr(&sm[u], (uint32_t)tk << (4 * r));
        }
        if (tid == 0) fs[row] = c > 0 ? fsv : 0;
    }
    __syncthreads();
    const int padded = (cnt + kG8UnionTile / 4 - 1) & ~(kG8UnionTile / 4 - 1);
    for (int i = tid; i < padded; i += 256) M[(size_t)g * up + i] = i < cnt ? sm[i] : 0u;
    if (tid == 0) {
        gctx[g] = 4 * cnt;
        cu[g] = g * kG8UnionR;
        if (g == (int)gridDim.x - 1) cu[g + 1] = rows;
    }
}

/* Declare-time bytes: the per-row partials at one split for max_seqs rows, the plan after them. */
static int64_t g8_union_scratch(const RadArgs* a) {
    const long long n_head = halo_geti_or(a, "n_head", 0), rows = halo_geti_or(a, "max_seqs", 0);
    if (n_head <= 0 || rows < kG8UnionMinRows) return 0;
    const long long part = rows * n_head * (256 * 2 + 8);
    return (int64_t)(((part + 255) & ~255LL) + g8_union_plan(nullptr, rows, kG8UnionUp).bytes);
}

/* 1 when the launch is not the union form's (the per-row form takes it), else 0 or the error. */
static int g8_union_launch(const HaloArgs* a, long long scratch_bytes, RadStream stream,
                           const HaloGateQuant* gq) {
    if (a->cu_q || a->q_len != 1 || a->num_seqs < kG8UnionMinRows || a->k_descale ||
        a->v_descale || a->window != 0 || a->kv_rep > 1 || a->q_ld != 0 || a->chain ||
        a->head_dim != 256 || a->block_size != 4 || a->kv_heads <= 0 ||
        a->q_heads / a->kv_heads != G8_GQA || a->max_blocks > kG8UnionMaxSp || a->splits > 1)
        return 1;
    constexpr int TILE = HALO_D_TILE, JPB = HALO_D_JPB;
    constexpr int PF16 = (G8_D_OPT & O_PF16) ? 1 : 0;
    static_assert(PF16, "the partials are f16");
    const long long rows = a->num_seqs, G = (rows + kG8UnionR - 1) / kG8UnionR;
    const int up = (kG8UnionR * a->max_blocks + 7) & ~7;
    const long long part = rows * a->q_heads * (256 * 2 + 8);
    const long long off = (part + 255) & ~255LL;
    G8UnionPlan pl = g8_union_plan((char*)a->scratch + off, rows, up);
    if (!a->scratch || off + pl.bytes > scratch_bytes) return 1;
    hipStream_t s = (hipStream_t)stream;

    hipLaunchKernelGGL(g8_union_plan_kernel, dim3((unsigned)G), dim3(256), 0, s, a->block_table,
                       a->max_blocks, a->seqused_k, (int)rows, up, pl.u, pl.m, pl.gctx, pl.cu, pl.fs,
                       TILE);
    CHK8();
    HaloArgs u = *a;
    u.num_seqs = (int)G;
    u.q_len = kG8UnionR;
    u.cu_q = pl.cu;
    u.total_q = (int)rows;
    u.block_table = pl.u;
    u.max_blocks = up;
    u.seqused_k = pl.gctx;
    u.umask = pl.m;
    u.splits = 1;
    hipLaunchKernelGGL((halo_attn_decode_kernel<4, kG8UnionTile, 256, G8_GQA, 4, 0, G8_D_OPT | O_UNION>),
                       dim3(1, a->kv_heads, (unsigned)G), dim3(4 * halo_dec_ds(256) * 32), 0, s, u, 1);
    CHK8();
    /* The merge reads one partial a row: `fs` gives a live row one tile of context. */
    HaloArgs m = *a;
    m.seqused_k = pl.fs;
    const int jobs = (int)rows * a->q_heads, blocks = (jobs + JPB - 1) / JPB;
    if (gq->i8)
        hipLaunchKernelGGL((halo_attn_splitkv_combine_vec_gq_kernel<256, 4, PF16, JPB, 0, 2>),
                           dim3(blocks), dim3(32 * JPB), 0, s, m, 1, TILE, *gq);
    else
        hipLaunchKernelGGL((halo_attn_splitkv_combine_vec_gq_kernel<256, 4, PF16, JPB, 0, 1>),
                           dim3(blocks), dim3(32 * JPB), 0, s, m, 1, TILE, *gq);
    CHK8();
    return 0;
}
#endif  /* G8_UNION && HALO_ATTN_GQ && HALO_ATTN_H256 */

// ---- decode entry points ------------------------------------------------------------------------
#if HALO_ATTN_H256
extern "C" int64_t GQ_SYM(halo_rad_attn_decode_h256_gqa, _scratch)(const RadArgs* a) {
    return g8_decode_scratch(a, 256);
}
extern "C" int GQ_SYM(halo_rad_attn_decode_h256_gqa, _fp8)(const RadArgs* a, RadStream s) {
    HaloArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    return g8_decode_launch<256, 0>(&r, s);
}
extern "C" int GQ_SYM(halo_rad_attn_decode_h256_gqa, _bf16)(const RadArgs* a, RadStream s) {
    HaloArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    return g8_decode_launch<256, 1>(&r, s);
}
#if HALO_ATTN_GQ
/* Gated: the merge's epilogue carries the gate and the quantiser. */
extern "C" int GQ_SYM(halo_rad_attn_decode_gq_h256_gqa, _fp8)(const RadArgs* a, RadStream s) {
    HaloArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    HaloGateQuant gq{};
    const int gc = g8_gate_quant_args(a, r, &gq);
    if (gc != RAD_OK) return gc;
#if G8_UNION
    const int uc = g8_union_launch(&r, (long long)a->scratch_bytes, s, &gq);
    if (uc <= 0) return uc;
#endif
    return g8_decode_launch<256, 0>(&r, s, &gq);
}
extern "C" int GQ_SYM(halo_rad_attn_decode_gq_h256_gqa, _bf16)(const RadArgs* a, RadStream s) {
    HaloArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    HaloGateQuant gq{};
    const int gc = g8_gate_quant_args(a, r, &gq);
    if (gc != RAD_OK) return gc;
    return g8_decode_launch<256, 1>(&r, s, &gq);
}
#endif  /* HALO_ATTN_GQ */
#endif  /* HALO_ATTN_H256 */

#if HALO_ATTN_H128
extern "C" int64_t GQ_SYM(halo_rad_attn_decode_h128_gqa, _scratch)(const RadArgs* a) {
    return g8_decode_scratch(a, 128);
}
extern "C" int GQ_SYM(halo_rad_attn_decode_h128_gqa, _fp8)(const RadArgs* a, RadStream s) {
    HaloArgs r;
    const int rc = g8_decode_common(a, &r, 128);
    if (rc != RAD_OK) return rc;
    return g8_decode_launch<128, 0>(&r, s);
}
extern "C" int GQ_SYM(halo_rad_attn_decode_h128_gqa, _bf16)(const RadArgs* a, RadStream s) {
    HaloArgs r;
    const int rc = g8_decode_common(a, &r, 128);
    if (rc != RAD_OK) return rc;
    return g8_decode_launch<128, 1>(&r, s);
}
#endif  /* HALO_ATTN_H128 */

#if HALO_ATTN_H256 && HALO_ATTN_V4
/* The fp8_v4 (a token's row 400 bytes: E4M3 K, int4 V, f16 scales) and k6_v4 (352 bytes: int6 K,
 * int4 V) caches on the decode leg, plain and gated. */
#define G8_V4_DECODE(TAG, ROWB, XOPT)                                                          \
extern "C" int64_t GQ_SYM(halo_rad_attn_decode_h256_gqa, _##TAG##_scratch)(const RadArgs* a) { \
    return g8_decode_scratch(a, 256, ROWB);                                                    \
}                                                                                              \
extern "C" int GQ_SYM(halo_rad_attn_decode_h256_gqa, _##TAG)(const RadArgs* a, RadStream s) {  \
    HaloArgs r;                                                                                \
    const int rc = g8_decode_common(a, &r, 256, ROWB);                                         \
    if (rc != RAD_OK) return rc;                                                               \
    return g8_decode_launch<256, 0, (XOPT)>(&r, s);                                            \
}                                                                                              \
extern "C" int GQ_SYM(halo_rad_attn_decode_gq_h256_gqa, _##TAG)(const RadArgs* a, RadStream s) { \
    HaloArgs r;                                                                                \
    const int rc = g8_decode_common(a, &r, 256, ROWB);                                         \
    if (rc != RAD_OK) return rc;                                                               \
    HaloGateQuant gq{};                                                                        \
    const int gc = g8_gate_quant_args(a, r, &gq);                                              \
    if (gc != RAD_OK) return gc;                                                               \
    return g8_decode_launch<256, 0, (XOPT)>(&r, s, &gq);                                       \
}
G8_V4_DECODE(fp8v4, 400, O_V4)
G8_V4_DECODE(k6v4, 352, O_V4 | O_K6)
#undef G8_V4_DECODE
#endif  /* HALO_ATTN_H256 && HALO_ATTN_V4 */
