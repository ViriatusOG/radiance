/* Derived from libr4d/r4d_moe.hip's moe_gemm_bf16 unit (2026-10-26), then ported to gfx1151;
 * see libhalo/README.md. */
// halo_moe_gemm_bf16.hip -- moe_gemm_bf16: the grouped GEMM with unquantised experts.
//
//     y[i, n] = sum_k a[row(i), k] * w[expert(i), n, k]     bf16 in, f32 accumulate, bf16 out
//
// over expert-sorted routed rows: `sorted_tok` permutes the live slots and `expert_offset` gives
// each expert's run of them, so one block owns one run (or one segment of it) and stages one
// expert's weight from the stacked `[n_expert, N, K]` plane or through the pointer table. Rows past
// the live count are dropped slots and must come out zero.
//
// Two launch forms, by the routed-row count T = M * top_k, libr4d's rule unchanged:
//   - decode, `2 * T <= 3 * n_expert`: 16 rows a block and one expert run per block, the run found
//     by a block-wide search over the offsets. No map and no scratch.
//   - prefill: 32 rows a block over a run-aligned block map (one block per (expert, first row), so
//     a streamed weight crosses the link once), 128 columns over eight waves where N allows it and
//     64 over four otherwise.
// Every form is bit-identical: each output element is one wave's accumulation over K in one order.
//
// The port from RDNA4 is the fragment layer and nothing else (halo_wmma.h states the algebra):
//
//   * gfx1151 replicates an operand row across both halves of the wave, so a lane holds the WHOLE
//     16-wide k run. A fragment is one contiguous 32-byte read of the row at
//     `tile + (lane & 15) * kP16 + ks` -- two uint4 -- and libr4d's `4 * (lane >> 4)` half term
//     disappears.
//   * a D fragment gives lane l element e = D[2 * e + (l >> 4)][l & 15], not gfx12's
//     D[8 * (l >> 4) + e][l & 15], so the output store walks rows 2e + (l >> 4) with the column
//     still on (l & 15).
//
// Everything else -- the LDS staging shape and its pitch of 72, the register prefetch of the next
// K pass, the K loop's `lds_barrier` and its one release fence, the block map and its parallel
// scan, the (tile, run) search, the live-row masking and the dropped-slot zeroing -- is libr4d's.
//
// Pinned form. This unit defines no `halo_moe_gemm_pin_*` entry point: halo_stubs.cpp keeps the
// weak no-op definitions and other lanes own them, so `g_form` below is the only form state here
// and every pin degrades to the rule. That is safe because the forms are bit-identical anyway.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_wmma.h"
#include "halo_emit.h"

#include <cstdint>
#include <cstring>

namespace {

constexpr uint32_t kMgWave = 32;              // wave32: the fragment layout is per wave
constexpr uint32_t kMgBK   = 64;              // bf16 K elements staged per pass
constexpr uint32_t kMgP16  = kMgBK + 8;       // the LDS row pitch, in elements (144 bytes)

/* A block-map entry naming the dropped-slot tail rather than an expert. */
constexpr uint32_t kMgMapTail = 0xffffffffu;
/* The block map's LDS size, shared with halo_moe_sort.h's kMaxExpert: wider expert counts take
 * the uniform grid, which needs no map. */
constexpr uint32_t kMgMaxExpert = 512;

/* The prefill tile: 32 rows, and 128 columns over eight waves where a column tile cannot straddle
 * two stacked weight parts. Each wave's share is unchanged (one 16-row by two 16-column tile), so
 * every element's arithmetic is the same; the wider block stages the activation once for twice the
 * columns. 128 columns over only four waves is no better. */
constexpr long long kMgPrefillRows = 32;
constexpr long long kMgPrefillCols = 128;

/* The pinned launch form. 0 is the launcher's rule; a pin is accepted and ignored, so every pinned
 * form takes the rule (see the file header). */
static int g_form = 0;

/* An entry of a weight-table operand: `tab` is the pointer array when `s0` is 0, else the base of a
 * stacked plane. Resolved on the device -- the table is in device memory, the mover may rewrite it,
 * and a host read would need a sync. */
__device__ __forceinline__ const uint8_t* mg_wtab_at(const void* const* tab, size_t s0,
                                                     uint32_t e) {
    if (s0 == 0) return (const uint8_t*)tab[e];
    return (const uint8_t*)tab + (size_t)e * s0;
}

/* A 16x16 bf16 operand fragment from an LDS tile whose row 0 is `tile`, advanced to the k step.
 * Lane l holds row (l & 15) and the whole 16-wide k run, so this is the one 32-byte contiguous read
 * gfx1151 wants; on gfx12 it was two 8-byte reads 16 bytes apart. */
__device__ __forceinline__ v16bf mg_frag(const uint16_t* __restrict__ tile, uint32_t lane) {
    return halo::mk_ldg<v16bf>(tile + (size_t)(lane & 15u) * kMgP16);
}

/* ---- the grouped GEMM -------------------------------------------------------------------------
 *
 * MT is the tile's 16-row count, NTL its 16-column tiles, WGM x WGN the wave grid over it.
 * `ES` selects the decode form (one run a block, run index in blockIdx.x % FM), `AL` the
 * run-aligned prefill form (one map entry a block), `SK` the row-masked weight staging. */
template <uint32_t MT, uint32_t NTL, uint32_t WGM, uint32_t WGN, bool ES = false, bool AL = false,
          bool SK = false>
__global__ __launch_bounds__(WGM * WGN * kMgWave) void mg_gemm_kernel(
    const uint16_t* __restrict__ A, const void* const* __restrict__ Wt, size_t w_s0,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    uint16_t* __restrict__ Y,
    uint32_t T, uint32_t N, uint32_t K, uint32_t n_expert, uint32_t top_k, uint32_t a_sorted,
    uint32_t a_ld, uint32_t w_ld, uint32_t y_ld,
    const uint32_t* __restrict__ map) {
    constexpr uint32_t FM = MT * 16, FN = NTL * 16;
    constexpr uint32_t NTHR = WGM * WGN * kMgWave;
    constexpr uint32_t WM = MT / WGM, WN = NTL / WGN;
    /* Sixteen-byte chunks: eight per staged row, and how many a thread carries per tile. */
    constexpr uint32_t kCPR = kMgBK / 8;
    constexpr uint32_t kAC = (FM * kCPR + NTHR - 1) / NTHR;
    constexpr uint32_t kBC = (FN * kCPR + NTHR - 1) / NTHR;
    static_assert(MT % WGM == 0 && NTL % WGN == 0, "the wave grid must divide the tile");

    __shared__ __attribute__((aligned(16))) uint16_t s_a[FM * kMgP16];
    __shared__ __attribute__((aligned(16))) uint16_t s_b[FN * kMgP16];
    __shared__ int32_t s_tok[FM];

    const uint32_t tid = threadIdx.x, lane = tid % kMgWave, wave = tid / kMgWave;
    const uint32_t wgm = wave / WGN, wgn = wave % WGN;
    /* The block's rows and run. */
    const uint32_t jrun = ES ? (blockIdx.x % FM) : 0u;
    uint32_t r0, rN, e_al = 0;
    bool tail = false;
    if constexpr (AL) {
        if (blockIdx.x >= map[0]) return;
        const uint32_t* ent = map + 1 + 2 * blockIdx.x;
        e_al = ent[0];
        r0   = ent[1];
        tail = (e_al == kMgMapTail);
        const uint32_t lim = tail ? T : (uint32_t)off[e_al + 1];
        rN = (r0 + FM < lim) ? (r0 + FM) : lim;
    } else {
        r0 = (ES ? (blockIdx.x / FM) : blockIdx.x) * FM;
        if (r0 >= T) return;
        rN = (r0 + FM < T) ? (r0 + FM) : T;
    }
    const uint32_t n0 = blockIdx.y * FN;

    for (uint32_t r = tid; r < FM; r += NTHR) {
        const uint32_t i = r0 + r;
        int32_t t = -1;
        if (i < rN) {
            const int32_t f = sorted[i];
            t = f < 0 ? -1 : (a_sorted ? (int32_t)i : f / (int32_t)top_k);
        }
        s_tok[r] = t;
    }
    __syncthreads();

    /* Dropped slots past the live count, zeroed by exactly one of a tile's blocks. */
    const uint32_t live = (uint32_t)off[n_expert];
    if (!ES || jrun == 0)
    for (uint32_t r = (r0 > live ? r0 : live); r < rN; ++r)
        for (uint32_t c = tid; c < FN && n0 + c < N; c += NTHR)
            Y[(size_t)r * y_ld + n0 + c] = 0;
    if constexpr (AL) { if (tail) return; }

    /* The expert of the block's first run, found the three ways: the map names it, the decode form
     * numbers the runs crossing the tile, and the uniform grid bisects the offsets. */
    uint32_t e = 0;
    if constexpr (AL) {
        e = e_al;
    } else if constexpr (ES) {
        __shared__ int32_t  s_run[FM];
        __shared__ uint32_t s_wc[NTHR / kMgWave];
        uint32_t nrun = 0;
        for (uint32_t c = 0; c < n_expert && nrun <= jrun; c += NTHR) {
            const uint32_t ce = c + NTHR < n_expert ? c + NTHR : n_expert;
            if ((uint32_t)off[ce] <= r0) continue;
            if ((uint32_t)off[c] >= rN) break;
            const uint32_t x = c + tid;
            bool p = false;
            if (x < ce) {
                const uint32_t lo = (uint32_t)off[x], hi = (uint32_t)off[x + 1];
                p = lo < hi && lo < rN && hi > r0;
            }
            const unsigned long long mk = __ballot(p);
            if (lane == 0) s_wc[wave] = (uint32_t)__popcll(mk);
            __syncthreads();
            uint32_t before = nrun, tot = 0;
            for (uint32_t w = 0; w < NTHR / kMgWave; ++w) {
                if (w < wave) before += s_wc[w];
                tot += s_wc[w];
            }
            const uint32_t pos = before + (uint32_t)__popcll(mk & ((1ull << lane) - 1ull));
            if (p && pos < FM) s_run[pos] = (int32_t)x;
            __syncthreads();
            nrun += tot;
        }
        if (jrun >= nrun) return;
        e = (uint32_t)s_run[jrun];
    } else {
        uint32_t lo = 0, hi = n_expert;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if ((uint32_t)off[mid] <= r0) lo = mid; else hi = mid;
        }
        e = lo;
    }

    uint32_t rs = r0, run = 0;
    if constexpr (ES) {
        const uint32_t s0 = (uint32_t)off[e];
        rs = s0 > r0 ? s0 : r0;
        run = jrun;
    } else {
        /* A call may serve a trailing segment of the sort (a layer's protected experts, after the
         * four-bit GEMM served the rest); then off[0] > 0 and the rows below belong to another
         * call. The block map and run search never reach them; the plain row grid must skip them. */
        const uint32_t s0 = (uint32_t)off[0];
        rs = s0 > r0 ? s0 : r0;
    }
    while (rs < rN && rs < live) {
        while (e < n_expert && (uint32_t)off[e + 1] <= rs) ++e;
        if (e >= n_expert) break;
        const uint32_t ehi = (uint32_t)off[e + 1];
        const uint32_t re  = rN < live ? (rN < ehi ? rN : ehi) : (live < ehi ? live : ehi);
        if (re <= rs) break;
        if (ES && run != jrun) { rs = re; ++run; continue; }

        const uint16_t* Wp = (const uint16_t*)mg_wtab_at(Wt, w_s0, e);
        const bool wlive = !SK || (r0 + wgm * WM * 16 < re);
        v8f acc[WM][WN];
#pragma unroll
        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
            for (uint32_t j = 0; j < WN; ++j) acc[i][j] = (v8f)(0.0f);

        /* Chunk c of a tile is row c / kCPR, elements (c % kCPR) * 8. A row outside the run, a
         * dropped slot or a column past N stages zeros. Every load is issued (a dead row reads an
         * existing row) and the store selects the zero, since a conditional load would serialise
         * the pass's loads. */
        halo_u32x4 apre[kAC], bpre[kBC];
        bool aok[kAC], bok[kBC];
        auto issue = [&](uint32_t k0) {
#pragma unroll
            for (uint32_t p = 0; p < kAC; ++p) {
                const uint32_t c = tid + p * NTHR;
                if (c >= FM * kCPR) continue;
                const uint32_t r = c / kCPR, kk = (c % kCPR) * 8;
                const int32_t tok = s_tok[r];
                aok[p] = tok >= 0 && (r0 + r) >= rs && (r0 + r) < re;
                apre[p] = *(const halo_u32x4*)(A + (size_t)(aok[p] ? tok : 0) * a_ld + k0 + kk);
            }
#pragma unroll
            for (uint32_t p = 0; p < kBC; ++p) {
                const uint32_t c = tid + p * NTHR;
                if (c >= FN * kCPR) continue;
                const uint32_t r = c / kCPR, kk = (c % kCPR) * 8;
                bok[p] = (n0 + r) < N;
                bpre[p] = *(const halo_u32x4*)(Wp + (size_t)(bok[p] ? n0 + r : N - 1) * w_ld +
                                               k0 + kk);
            }
        };
        issue(0);

        /* The K loop's barriers order LDS only, so they are lds_barrier() (__syncthreads() would
         * wait for the prefetch right after issuing it). The prefetch is collected by one global
         * release fence where the stores begin. */
        const halo_u32x4 zero = (halo_u32x4)(0u);
        for (uint32_t k0 = 0; k0 < K; k0 += kMgBK) {
            lds_barrier();                         /* the previous pass's fragments are read */
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");
#pragma unroll
            for (uint32_t p = 0; p < kAC; ++p) {
                const uint32_t c = tid + p * NTHR;
                if (c >= FM * kCPR) continue;
                *(halo_u32x4*)(s_a + (c / kCPR) * kMgP16 + (c % kCPR) * 8) =
                    aok[p] ? apre[p] : zero;
            }
#pragma unroll
            for (uint32_t p = 0; p < kBC; ++p) {
                const uint32_t c = tid + p * NTHR;
                if (c >= FN * kCPR) continue;
                *(halo_u32x4*)(s_b + (c / kCPR) * kMgP16 + (c % kCPR) * 8) =
                    bok[p] ? bpre[p] : zero;
            }
            if (k0 + kMgBK < K) issue(k0 + kMgBK);
            lds_barrier();

            if (!wlive) continue;
#pragma unroll
            for (uint32_t ks = 0; ks < kMgBK; ks += 16) {
                v16bf af[WM], bf[WN];
#pragma unroll
                for (uint32_t i = 0; i < WM; ++i) {
                    if (SK && r0 + (wgm * WM + i) * 16 >= re) continue;
                    af[i] = mg_frag(s_a + (wgm * WM + i) * 16 * kMgP16 + ks, lane);
                }
#pragma unroll
                for (uint32_t j = 0; j < WN; ++j)
                    bf[j] = mg_frag(s_b + (wgn * WN + j) * 16 * kMgP16 + ks, lane);
#pragma unroll
                for (uint32_t i = 0; i < WM; ++i) {
                    if (SK && r0 + (wgm * WM + i) * 16 >= re) continue;
#pragma unroll
                    for (uint32_t j = 0; j < WN; ++j) acc[i][j] = halo::mma(af[i], bf[j], acc[i][j]);
                }
            }
        }

        /* Lane l holds column (l & 15) and rows 2 * e + (l >> 4) (the gfx1151 D fragment,
         * halo_wmma.h). Rounded to nearest-even. */
#pragma unroll
        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
            for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    const uint32_t r = (wgm * WM + i) * 16 + 2 * t + (lane >> 4);
                    const uint32_t c = (wgn * WN + j) * 16 + (lane & 15u);
                    const uint32_t row = r0 + r;
                    if (r >= FM || row < rs || row >= re || n0 + c >= N) continue;
                    Y[(size_t)row * y_ld + n0 + c] = f32_to_bf16(acc[i][j][t]);
                }

        rs = re;
        ++run;
        if (ES) break;
    }
}

/* ---- the run-aligned block map ----------------------------------------------------------------
 *
 * The uniform row grid serves a run straddling a tile boundary with two blocks, staging its weight
 * twice. For a resident expert the second staging is an L2 hit, but for a pooled one it is another
 * pass over PCIe. So a block map gives one run per block: entry i names (expert, first row) and the
 * block serves rows [first, min(first + FM, off[e + 1])). The grid is an upper bound,
 * `n_expert + T/FM`; the map's first slot holds the real count and extra blocks exit at the head.
 *
 * Rows past `off[n_expert]` (dropped slots) still have to be zeroed: the map carries them as
 * entries with expert `kMgMapTail`, whose block zeroes and returns. */

/* The map's upper bound, defined once so the scratch hook and the launcher agree exactly (or the
 * kernel writes past its arena). */
static inline long long mg_blockmap_cap(long long T, long long n_expert, long long FM) {
    return n_expert + (T + FM - 1) / FM + 1;
}
/* One count slot, then two uint32 per entry. */
static inline long long mg_blockmap_bytes(long long cap) {
    return (1 + 2 * cap) * (long long)sizeof(uint32_t);
}

/* One workgroup: the job is a scan over at most kMgMaxExpert counters, cheaper than a second
 * launch. NT threads, thread tid owning experts tid, tid + NT, ...; offsets are read once, in
 * parallel. The prefix is parallel (a DPP scan per wave and one exchange of wave totals), not
 * hundreds of dependent LDS round trips on one thread ahead of a GEMM that waits for the map.
 * Integer sums are order-independent. */
__global__ __launch_bounds__(256) void mg_blockmap_kernel(
    const int32_t* __restrict__ off, uint32_t n_expert, uint32_t T, uint32_t FM, uint32_t cap,
    uint32_t* __restrict__ map) {
    constexpr uint32_t NT = 256, NW = NT / kMgWave, kTiles = (kMgMaxExpert + NT) / NT;
    __shared__ uint32_t s_cnt[kMgMaxExpert + 1];
    __shared__ int32_t  s_wt[kTiles][NW];
    const uint32_t tid = threadIdx.x, lane = tid % kMgWave, wave = tid / kMgWave;

    /* Every offset this thread needs in one round of loads: both ends of each expert it owns
     * (e = tid + i * NT), index clamped so every load is unconditional, all issued before the first
     * wait. Counts and map rows come from these registers; offsets never pass through LDS. */
    int32_t lo[kTiles], hi[kTiles];
#pragma unroll
    for (uint32_t i = 0; i < kTiles; ++i) {
        const uint32_t e = tid + i * NT;
        lo[i] = off[e < n_expert ? e : n_expert];
        hi[i] = off[e + 1 < n_expert ? e + 1 : n_expert];
    }
    const uint32_t live = (uint32_t)off[n_expert];
    /* Blocks this expert needs; a zero-row run needs none. */
#pragma unroll
    for (uint32_t i = 0; i < kTiles; ++i) {
        const uint32_t e = tid + i * NT;
        if (e < n_expert) s_cnt[e] = ((uint32_t)(hi[i] - lo[i]) + FM - 1) / FM;
    }
    /* The tail of dropped slots, carried as one more "expert" so the scan places it too. */
    if (tid == 0) s_cnt[n_expert] = T > live ? (T - live + FM - 1) / FM : 0u;
    __syncthreads();

    /* Exclusive scan over the n_expert + 1 counts, NT at a time. Each tile has its own exchange
     * slots, so a tile's writes never meet the previous tile's reads. */
    uint32_t carry = 0;
    for (uint32_t b = 0, tl = 0; b <= n_expert; b += NT, ++tl) {
        const uint32_t e = b + tid;
        const int32_t  v = e <= n_expert ? (int32_t)s_cnt[e] : 0;
        const int32_t  x = (int32_t)halo_wave_prefix_u32((uint32_t)v, (int)lane);
        if (lane == kMgWave - 1) s_wt[tl][wave] = x;
        lds_barrier();
        int32_t below = 0, tot = 0;
#pragma unroll
        for (uint32_t k = 0; k < NW; ++k) {
            const int32_t c = s_wt[tl][k];
            below += (k < wave) ? c : 0;
            tot   += c;
        }
        if (e <= n_expert) s_cnt[e] = carry + (uint32_t)(below + x - v);
        carry += (uint32_t)tot;
    }
    if (tid == 0) map[0] = carry < cap ? carry : cap;    /* the count, clamped to the grid */
    __syncthreads();

#pragma unroll
    for (uint32_t i = 0; i < kTiles; ++i) {
        const uint32_t e = tid + i * NT;
        if (e >= n_expert) continue;
        const uint32_t run = (uint32_t)(hi[i] - lo[i]);
        uint32_t o = s_cnt[e];
        for (uint32_t r = 0; r < run; r += FM, ++o) {
            if (o >= cap) break;
            map[1 + 2 * o + 0] = e;
            map[1 + 2 * o + 1] = (uint32_t)lo[i] + r;
        }
    }
    if (tid == 0) {
        uint32_t o = s_cnt[n_expert];
        for (uint32_t r = live; r < T; r += FM, ++o) {
            if (o >= cap) break;
            map[1 + 2 * o + 0] = kMgMapTail;
            map[1 + 2 * o + 1] = r;
        }
    }
}

/* Where the (tile, run) decode grid ends and the run-aligned prefill grid begins: an average of 1.5
 * routed rows per expert. At file scope because the scratch hook and the launcher must agree on the
 * band. Decode is at most `max_num_seqs * (1 + n_spec) * top_k`, a few hundred rows. */
static inline bool mg_decode_band(long long T, long long n_expert) {
    return 2 * T <= 3 * n_expert;
}

/* The weight-table operand from the core's tensor. A zero leading stride marks a pointer array
 * (abi/rad_runtime.h, RAD_WTAB); otherwise it is a stacked plane with the stride in elements of the
 * weight's dtype. bf16 is sixteen bits, so the stacked stride in bytes is twice the element
 * stride. */
static int mg_wtab_of(const RadTensor* t, uint32_t dtype, int bits, long long n_expert,
                      long long rows, long long cols, const void* const** tab, size_t* stride0,
                      long long* ld) {
    if (!t || !t->data || t->rank != 3) return HALO_NO(RAD_E_SHAPE);
    if (t->dtype != dtype) return HALO_NO(RAD_E_DTYPE);
    if (t->shape[0] < n_expert || t->shape[1] < rows || t->shape[2] < cols)
        return HALO_NO(RAD_E_SHAPE);
    if (t->stride[2] != 1) return HALO_NO(RAD_E_STRIDE);
    *tab     = (const void* const*)t->data;
    *stride0 = (size_t)((t->stride[0] * (int64_t)bits) / 8);
    *ld      = t->stride[1];
    return RAD_OK;
}

/* Bytes for the run map, zero where it is not used.
 *
 * T comes from the parameters (`M * top_k`, the routed row count), not an operand, since the hook
 * runs at declare before tensors are bound; the launcher derives the same count from Y. If they
 * ever disagree it is safe: the launcher checks the scratch it was handed and falls back to the
 * uniform grid when it is short. */
static int64_t mg_map_scratch(const RadArgs* a) {
    const long long n_expert = halo_geti_or(a, "n_expert", 0);
    const long long top_k    = halo_geti_or(a, "top_k", 0);
    const long long M        = halo_geti_or(a, "M", 0);
    if (n_expert <= 0 || n_expert > (long long)kMgMaxExpert || top_k <= 0 || M <= 0) return 0;
    const long long T = M * top_k;
    if (mg_decode_band(T, n_expert)) return 0;      /* the (tile, run) grid needs no map */
    return mg_blockmap_bytes(mg_blockmap_cap(T, n_expert, kMgPrefillRows));
}

}  // namespace

/* ============================================================== the plugin ABI entry points */

/* `moe_gemm` at bf16: operands a, w (one table), sorted_tok, expert_offset -> y. Band and forms
 * follow the rule above, so the same scratch hook sizes the run map.
 *
 * Every load is sixteen bytes, hence the alignment checks: row pitches must be multiples of eight
 * elements and stacked experts 16-byte aligned. Pointer-table entries live in device memory and
 * cannot be checked here; their allocators hand out movement-unit-aligned addresses. */
extern "C" int halo_rad_moe_gemm_bf16(const RadArgs* a, RadStream stream) {
    int emit_rc = RAD_OK;
    const RadTensor* A      = halo_opt(a, 0);
    const RadTensor* W      = halo_opt(a, 1);
    const RadTensor* sorted = halo_opt(a, 2);
    const RadTensor* off    = halo_opt(a, 3);
    const RadTensor* Y      = halo_opt(a, 4);
    if (!A || !W || !sorted || !off || !Y) return HALO_NO(RAD_E_INVAL);
    if (!A->data || !sorted->data || !off->data || !Y->data) return HALO_NO(RAD_E_INVAL);
    const char* dt = halo_gets(a, "dtype");
    if (!dt || std::strcmp(dt, "bf16")) return HALO_NO(RAD_E_DTYPE);

    const long long N = halo_geti_or(a, "N", 0);
    const long long K = halo_geti_or(a, "K", 0);
    const long long n_expert = halo_geti_or(a, "n_expert", 0);
    const long long top_k = halo_geti_or(a, "top_k", 0);
    if (N <= 0 || K <= 0 || n_expert <= 0 || top_k <= 0) return HALO_NO(RAD_E_SHAPE);
    /* Column tiles are 64 wide and all weight; a K pass is 64 deep. */
    if (N % 64 || K % (long long)kMgBK) return HALO_NO(RAD_E_SHAPE);
    if (A->dtype != RAD_BF16 || Y->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    if (sorted->dtype != RAD_I32 || off->dtype != RAD_I32) return HALO_NO(RAD_E_DTYPE);

    long long T = 0;
    const long long y_ld = halo_rows_pitch(Y, N, &T);
    if (y_ld < 0 || T <= 0) return HALO_NO(RAD_E_STRIDE);
    long long M = 0;
    const long long a_ld = halo_rows_pitch(A, K, &M);
    if (a_ld < 0 || M <= 0 || a_ld % 8 || !halo_aligned(A->data, 16)) return HALO_NO(RAD_E_STRIDE);
    if (halo_numel(sorted) < T || halo_numel(off) < n_expert + 1) return HALO_NO(RAD_E_SHAPE);

    /* The activation's row count is checked, not inferred. */
    const char* ord = halo_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return HALO_NO(RAD_E_UNSUPPORTED);
    if (M < (a_sorted ? T : (T + top_k - 1) / top_k)) return HALO_NO(RAD_E_SHAPE);

    const void* const* tab = nullptr;
    size_t w_s0 = 0;
    long long w_ld = 0;
    const int rc = mg_wtab_of(W, RAD_BF16, 16, n_expert, N, K, &tab, &w_s0, &w_ld);
    if (rc != RAD_OK) return rc;
    if (w_ld % 8 || w_s0 % 16) return HALO_NO(RAD_E_STRIDE);

    const uint32_t ny = (uint32_t)(N / 64);
    if (mg_decode_band(T, n_expert)) {
        const unsigned tiles = (unsigned)((T + 15) / 16);
        emit_rc = HALO_EMIT(stream, (mg_gemm_kernel<1, 4, 1, 4, true>),
                            dim3(tiles * 16u, ny), dim3(4 * kMgWave), 0,
                            (const uint16_t*)A->data, tab, w_s0,
                            (const int32_t*)sorted->data, (const int32_t*)off->data,
                            (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)K,
                            (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)(a_sorted ? 1 : 0),
                            (uint32_t)a_ld, (uint32_t)w_ld, (uint32_t)y_ld, nullptr);
    } else {
        const long long cap = mg_blockmap_cap(T, n_expert, kMgPrefillRows);
        uint32_t* map = (g_form != 3 && n_expert <= (long long)kMgMaxExpert && a->scratch &&
                         a->scratch_bytes >= mg_blockmap_bytes(cap))
                            ? (uint32_t*)a->scratch : nullptr;
        /* The 128-column, eight-wave tile wherever the width allows. */
        const bool wide_n = N % kMgPrefillCols == 0;
        const unsigned nyw = (unsigned)(N / kMgPrefillCols);
        const unsigned gx = map ? (unsigned)cap : (unsigned)((T + kMgPrefillRows - 1) / kMgPrefillRows);
        if (map) {
            emit_rc = HALO_EMIT(stream, mg_blockmap_kernel, dim3(1), dim3(256), 0,
                                (const int32_t*)off->data, (uint32_t)n_expert, (uint32_t)T,
                                (uint32_t)kMgPrefillRows, (uint32_t)cap, map);
            if (emit_rc != RAD_OK) return emit_rc;
        }
#define HALO_MOE16(AL_, NTL_, WGN_, NYG_)                                                       \
        HALO_EMIT(stream,                                                                      \
                 (mg_gemm_kernel<kMgPrefillRows / 16, NTL_, 2, WGN_, false, AL_, true>),        \
                 dim3(gx, NYG_), dim3(2 * WGN_ * kMgWave), 0,                                    \
                 (const uint16_t*)A->data, tab, w_s0,                                            \
                 (const int32_t*)sorted->data, (const int32_t*)off->data,                        \
                 (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)K,                      \
                 (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)(a_sorted ? 1 : 0),              \
                 (uint32_t)a_ld, (uint32_t)w_ld, (uint32_t)y_ld, (const uint32_t*)map)
        if (map)  emit_rc = wide_n ? HALO_MOE16(true, kMgPrefillCols / 16, 4, nyw)
                                   : HALO_MOE16(true, 4, 2, ny);
        else      emit_rc = wide_n ? HALO_MOE16(false, kMgPrefillCols / 16, 4, nyw)
                                   : HALO_MOE16(false, 4, 2, ny);
#undef HALO_MOE16
    }
    if (emit_rc != RAD_OK) return emit_rc;
    const hipError_t eh = hipGetLastError();
    if (eh != hipSuccess) return HALO_NO(-100 - (int)eh);
    return RAD_OK;
}

extern "C" int64_t halo_rad_moe_gemm_bf16_scratch(const RadArgs* a) {
    return mg_map_scratch(a);
}
