/* Derived from libr4d/r4d_attn_paged_window.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// Halo paged attention at head_dim 128, 256 and 512, 1/2/4/8 queries per KV head, any sliding
// window, causal or not, bf16 or fp8-e4m3 cache, q at any row pitch.
//
// head_dim 128 is Unlimited-OCR's decoder (10 query heads over 10 KV heads): the plain MHA case,
// served here so its q slice of the fused q|k|v plane is read in place instead of being lifted into
// a dense buffer. Gemma 4's text layers: sliding layers are 16 query heads over 8 KV heads at
// head_dim 256 with a 1024-token window, full layers 16 over 2 at head_dim 512. Where this overlaps
// libhalo's other h256 legs, the row table decides (halo_rows.cpp).
//
// Two kernels behind one entry point, as in libr4d:
//   - Scalar, for decode and verify: a workgroup owns one (sequence, KV head, slice of BQ query
//     positions, split of the key range); each wave owns RW of the tile's R = BQ*gqa query rows.
//     K/V go through LDS one or two paged blocks at a time, prefetched into registers while the
//     previous tile is consumed. Dot products are scalar FMAs with a butterfly reduce-scatter that
//     leaves key j's score in lane j; online softmax in the log2 domain. Memory-bound at decode.
//   - WMMA, for prefill-shaped launches (see its header below).
//
// Both compute libref's ref_attn_paged, predicate for predicate:
//   pos     = seqused[s] - q_len(s) + i                  (seqused INCLUDES this step's queries)
//   causal  key j visible when j <= pos;  non-causal when j < seqused[s]
//   window  causal: pos - j < window;  non-causal: |pos - j| < window;  0 = none
//   a block-table entry < 0 is a missing block: its keys are not visible
//   no visible key at all -> a zero row
//   k_descale / v_descale per (sequence, kv head), absent = 1
//
// Paged layout: kv_cache[n_blocks, kv_heads, 16, 2*head_dim], K then V per slot, so one block of
// one KV head is a contiguous run of 16 * 2 * head_dim elements; a key tile is one or two runs,
// fetched as 16-byte loads.
//
// Split-KV: with few (tile, kv head) pairs (decode, short verify) each tile's key range is cut into
// `n_split` pieces, each writing an unnormalised partial (acc, max, sum) to scratch, merged by a
// second kernel. The split count comes from host-side numbers only (rows, sequences, window or
// max_ctx); each workgroup divides its own key range, so a split with no keys writes (0, -inf, 0)
// and the merge needs no device-side context.
//
// ---------------------------------------------------------------------------------------------
// The gfx1151 port, in libhalo/halo_dt16.h's terms.
//
// libr4d keeps the score matrix transposed (S^T = K.Q^T) so that a fragment's COLUMN -- the query
// -- is (lane & 15) on both architectures. That is why this body ports at all: the softmax state is
// still one query row per lane, the row max and row sum are still one exchange with lane ^ 16, and
// the slice partials still meet in LDS per lane. What changes is the ROW, the key:
//
//                        a lane's S^T element e        a lane's A/B fragment k
//   gfx1201 (RDNA4)      key 8*(lane>>4) + e          k = 8*(lane>>4) + e
//   gfx1151 (RDNA3.5)    key 2*e + (lane>>4)          k = e   (lanes 16-31 replicate)
//
// So four things differ, and they are the four halo_dt16.h names:
//   * every fragment gather reads 32 bytes at the un-split k offset: the `+ 8*h` lane-half term is
//     gone (WQ::mk / WP::mk take the lane's eight dwords);
//   * the mask indexes key = k0 + 2*e + h, and the sixteen keys of a tile still split across the
//     half-pair, so the existing swap16 row max and row sum keep their meaning;
//   * the P fragment is assembled from the pair with p16w() -- gfx1151's B fragment wants the
//     sixteen keys in order, and a lane holds one parity of them;
//   * the store scatters at head dim 2*e + h, one 16-bit write a value (O^T's row is the head dim).
// Q,K run on the bf16 WMMA and P,V on the f16 one, exactly as libr4d splits them, both through
// halo_dt16.h's DT16<>::wmma. An fp8-e4m3 cache widens through halo_fp8::e4m3_pk, gfx1151 having
// no fp8 convert instruction; the scalar kernel's LDS tile layout, its FMA dots and its
// reduce-scatter are architecture-neutral and come over unchanged.
//
// Nothing here allocates, synchronises, or uses anything but halo_launch.h's launch. Wave32: the
// shuffles and the reduce-scatter are written for 32 lanes.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

#include "halo.h"
#include "halo_args.h"
#include "halo_attn_args.h"
#include "halo_chain.h"
#include "halo_common.h"
#include "halo_dt16.h"
#include "halo_fp8_wmma.h"
#include "halo_launch.h"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "this attention is written for wave32 (RDNA): build without -mwavefrontsize64"
#endif

namespace halo_win {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 32;

/* Split-KV bookkeeping: at most kPartBudget (tile, kv head, split) triples carry partials and the
 * tile table holds at most kTinfoMax tiles. The launch stays under both; scratch is sized for them. */
constexpr int kPartBudget = 192;
constexpr int kTinfoMax = 128;
constexpr int kMaxSplit = 64;

template <int HD> struct Geo;
/* R query rows per workgroup (R / 8 per wave), BK keys per tile. BK * 2 * HD bf16 = 32 KiB of LDS
 * either way: two paged blocks at 256, one at 512, four at 128. */
template <> struct Geo<128> { static constexpr int R = 32, BK = 32; };
template <> struct Geo<256> { static constexpr int R = 32, BK = 32; };
template <> struct Geo<512> { static constexpr int R = 16, BK = 16; };

/* A 16-byte staging register carried across a loop (halo_common.h's halo_u32x4): HIP's uint4 is a
 * class template wrapping a union, and an array of them kept live across the key-tile loop is
 * demoted to scratch by this toolchain -- a scratch store and load per chunk, which is exactly the
 * prefetch this kernel is built around. */
typedef halo_u32x4 w4_t;

/* ------------------------------------------------------------------ device helpers */
__device__ __forceinline__ int lane_id() { return (int)(threadIdx.x & 31u); }

/* Lane ^ N, as halo_common.h spells it: a DPP/xor-lane modifier rather than a `ds_bpermute_b32`
 * and its wait for every step of a butterfly or a reduce-scatter. */
template <int N> __device__ __forceinline__ float xor_lane(float v) { return halo_xor_lane<N>(v); }

__device__ __forceinline__ int shfl_up(int v, int d) {
    const int l = lane_id();
    return __builtin_amdgcn_ds_bpermute((l >= d ? l - d : l) << 2, v);
}
__device__ __forceinline__ float readlane(float v, int l) {
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), l));
}
/* The row max and row sum over the sixteen keys of a tile, whose two halves are in lane and lane ^
 * 16 on both architectures (halo_common.h's wave-wide forms are the shuffle loop's own order). */
__device__ __forceinline__ float wave_max(float v) { return halo_wave_max(v); }
__device__ __forceinline__ float wave_sum(float v) { return halo_wave_sum(v); }
__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }

/* f(Ic<0>{}), ..., f(Ic<N-1>{}): a loop whose index is a compile-time constant in every body.
 * `#pragma unroll` is dropped past a size threshold (the fp8 row loop crosses it), and a register
 * array indexed by a surviving loop variable is demoted to scratch. This guarantees the unroll. */
template <int I> struct Ic { static constexpr int value = I; };
template <int I, int N> struct StaticFor {
    template <typename F> __device__ __forceinline__ static void run(F& f) {
        if constexpr (I < N) {
            f(Ic<I>{});
            StaticFor<I + 1, N>::run(f);
        }
    }
};

__device__ __forceinline__ void bf16x4(uint2 w, float (&f)[4]) {
    f[0] = bf16w_lo(w.x);
    f[1] = bf16w_hi(w.x);
    f[2] = bf16w_lo(w.y);
    f[3] = bf16w_hi(w.y);
}
/* Four e4m3 bytes as four floats, exact in either form: gfx1151 has no v_cvt_pk_f32_fp8, so this is
 * halo_fp8::e4m3_pk's software decode (halo_fp8_wmma.h), the same bytes libr4d's RDNA4 path
 * returns for every value the cache can hold. */
__device__ __forceinline__ void fp8x4(unsigned w, float (&f)[4]) {
    const v2f a = halo_fp8::e4m3_pk<false>(w);
    const v2f b = halo_fp8::e4m3_pk<true>(w);
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}
/* f32 -> bf16 pair, as libr4d's pack2: round to nearest even, a NaN kept quiet (halo_common.h's
 * bf16x2_rne_nan, which is radiance's rad_f32_to_bf16 with the NaN select). */
__device__ __forceinline__ unsigned pack2(float a, float b) { return bf16x2_rne_nan(a, b); }
__device__ __forceinline__ uint16_t bf16_of(float f) { return (uint16_t)(bf16_hi_rne_nan(f) >> 16); }

/* Four elements of one slot row, at element offset `e`, out of the LDS tile. */
template <bool KV8>
__device__ __forceinline__ void lds4(const unsigned char* row, int e, float (&f)[4]) {
    if constexpr (KV8) fp8x4(*(const unsigned*)(row + e), f);
    else bf16x4(*(const uint2*)(row + 2 * e), f);
}

/* Reduce-scatter: each lane holds a partial dot for every key of the tile; afterwards lane l holds
 * key l's full sum. Each step keeps the half its lane bit selects and adds the partner's copy:
 * 16 + 8 + 4 + 2 + 1 = 31 shuffles for 32 keys, instead of 32 * 5 for separate reductions. */
template <int H, int N>
__device__ __forceinline__ void reduce_scatter(float (&v)[N], int lane) {
    if constexpr (H >= 1) {
        const bool up = (lane & H) != 0;
#pragma unroll
        for (int i = 0; i < H; ++i) {
            /* Select values, not lvalues: `up ? v[i + H] : v[i]` selects an address, and an array
             * addressed by a lane-dependent pointer is demoted to scratch. */
            const float lo = v[i], hi = v[i + H];
            const float keep = up ? hi : lo;
            const float send = up ? lo : hi;
            v[i] = keep + xor_lane<H>(send);
        }
        reduce_scatter<H / 2, N>(v, lane);
    }
}

/* ------------------------------------------------------------------ the launch's arguments */
struct AttnArgs {
    const uint16_t*      q;      long long q_rs, q_hs;     /* elements */
    const unsigned char* kv;     long long kv_sb, kv_sh;   /* elements of the cache's dtype */
    const int*           bt;     long long bt_rs;
    const int*           seqused;
    const int*           cu;                               /* null: uniform q_len */
    const float*         kdes;
    const float*         vdes;
    uint16_t*            out;    long long o_rs, o_hs;
    float*               part;                             /* split-KV partials, or null */
    int*                 tinfo;                            /* [tile] = (first row, positions) */
    float qscale;                                          /* softmax scale * log2(e) */
    int   max_blocks, n_seq, q_len, tps, kv_heads, causal, window, n_split;
};

/* Tile `tile` of a ragged batch: its sequence and which BQ-position slice of that sequence's query
 * it is. Tiles are numbered sequence by sequence, ceil(q_len(s) / BQ) each, so this is a block-wide
 * prefix sum over cu_seqlens, NT sequences per pass. All threads get the same answer from LDS. */
template <int BQ, int NT = kThreads>
__device__ void find_tile(const int* cu, int n_seq, int tile, int* s_scan, int* s_res, int& s_out,
                          int& lt_out) {
    constexpr int NW = NT / 32;
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    if (tid == 0) { s_res[0] = -1; s_res[1] = 0; }
    __syncthreads();
    int base = 0;
    for (int c0 = 0; c0 < n_seq; c0 += NT) {
        const int s = c0 + tid;
        int nt = 0;
        if (s < n_seq) {
            const int ql = cu[s + 1] - cu[s];
            nt = ql > 0 ? (ql + BQ - 1) / BQ : 0;
        }
        int x = nt;
        for (int d = 1; d < 32; d <<= 1) {
            const int y = shfl_up(x, d);
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
        if (nt > 0 && tile >= incl - nt && tile < incl) { s_res[0] = s; s_res[1] = tile - (incl - nt); }
        base += total;
        __syncthreads();
        if (s_res[0] >= 0 || base > tile) break;
    }
    s_out = s_res[0];
    lt_out = s_res[1];
}

/* ------------------------------------------------------------------ the attention kernel */
/* The body; attn_kernel and its chained form halo_win_r4dchain_attn (halo_chain.h) wrap it below,
 * since its blocks return early. */
template <int HD, int GQA, bool KV8>
__device__ __forceinline__ void attn_body(AttnArgs A) {
    constexpr int R = Geo<HD>::R, BK = Geo<HD>::BK, RW = R / kWaves, BQ = R / GQA;
    constexpr int NC = HD / 128;                     /* 4-element chunks a lane holds of a row */
    constexpr int ESZ = KV8 ? 1 : 2;
    constexpr int KEY_BYTES = 2 * HD * ESZ;          /* one slot: K then V */
    constexpr int TILE_U4 = BK * KEY_BYTES / 16;
    constexpr int NPT = TILE_U4 / kThreads;          /* 16-byte loads a thread, per tile */
    constexpr int BLK_U4 = 16 * KEY_BYTES / 16;      /* one paged block of one kv head */
    constexpr int NB = BK / 16;                      /* paged blocks a key tile spans */
    static_assert(R % kWaves == 0 && R % GQA == 0, "rows must split over waves and heads");
    static_assert(TILE_U4 % kThreads == 0 && BLK_U4 % kThreads == 0, "tile must split over threads");
    static_assert(BK == 32 || BK == 16, "the reduce-scatter is written for 16 or 32 keys");

    __shared__ uint4 s_kv[TILE_U4];
    __shared__ int s_scan[kWaves];
    __shared__ int s_res[2];

    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int tile = (int)blockIdx.x, kh = (int)blockIdx.y, z = (int)blockIdx.z;

    /* ---- which sequence, and which slice of its query */
    int s = -1, lt = 0;
    if (A.cu) {
        find_tile<BQ>(A.cu, A.n_seq, tile, s_scan, s_res, s, lt);
    } else {
        s = tile / A.tps;
        lt = tile - s * A.tps;
        if (s >= A.n_seq) s = -1;
    }
    int qlen = 0, tokb = 0, ctx = 0;
    if (s >= 0) {
        if (A.cu) { tokb = A.cu[s]; qlen = A.cu[s + 1] - tokb; }
        else      { tokb = s * A.q_len; qlen = A.q_len; }
        ctx = A.seqused[s];
    }
    const int pfirst = lt * BQ;
    const int npos = s >= 0 ? imax(0, imin(BQ, qlen - pfirst)) : 0;
    if (A.n_split > 1 && kh == 0 && z == 0 && tid == 0) {
        A.tinfo[2 * tile] = tokb + pfirst;
        A.tinfo[2 * tile + 1] = npos;
    }
    if (npos <= 0) return;                           /* uniform across the block */

    const int tok0 = tokb + pfirst;
    const int qbase = ctx - qlen + pfirst;           /* absolute position of the tile's first query */
    const int plast = qbase + npos - 1;
    const int causal = A.causal, window = A.window;

    /* ---- the tile's key range, and this split's share of it */
    int hi = causal ? plast + 1 : ctx;
    if (!causal && window > 0) hi = imin(hi, plast + window);
    hi = imin(imin(hi, ctx), A.max_blocks * 16);
    const int lo = window > 0 ? imax(0, qbase - window + 1) : 0;
    const int lo_a = lo & ~(BK - 1);
    const int nkt = hi > lo_a ? (hi - lo_a + BK - 1) / BK : 0;
    const int per = (nkt + A.n_split - 1) / A.n_split;
    const int kt_lo = imin(nkt, z * per), kt_hi = imin(nkt, kt_lo + per);

    const float kds = A.kdes ? A.kdes[(long long)s * A.kv_heads + kh] : 1.0f;
    const float vds = A.vdes ? A.vdes[(long long)s * A.kv_heads + kh] : 1.0f;

    /* ---- this wave's query rows: row r is position r / GQA, head kh * GQA + r % GQA */
    float qf[RW][NC][4], o[RW][NC][4], m[RW], l[RW];
    int pos[RW];
    bool val[RW];
#pragma unroll
    for (int i = 0; i < RW; ++i) {
        const int r = wave * RW + i, p = r / GQA, g = r - p * GQA;
        val[i] = p < npos;
        pos[i] = qbase + p;
        m[i] = -__builtin_inff();
        l[i] = 0.0f;
        const uint16_t* qp = A.q + (long long)(tok0 + p) * A.q_rs + (long long)(kh * GQA + g) * A.q_hs;
        const float qs = A.qscale * kds;             /* k_descale folds into the query */
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (val[i]) bf16x4(*(const uint2*)(qp + c * 128 + lane * 4), f);
#pragma unroll
            for (int e = 0; e < 4; ++e) { qf[i][c][e] = f[e] * qs; o[i][c][e] = 0.0f; }
        }
    }

    /* ---- the key tiles, prefetched one ahead through registers */
    w4_t pre[NPT];
    const w4_t zero4 = {0u, 0u, 0u, 0u};
    bool pre_ok[NB];
    auto fetch = [&](int kt) __attribute__((always_inline)) {
        const int k0 = lo_a + kt * BK;
        const unsigned char* src[NB];
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const int lb = (k0 >> 4) + b;
            int blk = -1;
            if (lb < A.max_blocks && (lb << 4) < hi) blk = A.bt[(long long)s * A.bt_rs + lb];
            pre_ok[b] = blk >= 0;
            /* Never null: selecting against null makes the pointer flat instead of global, and every
             * tile load a flat_load. A missing block points at the cache base; its value is
             * replaced by zeros below. */
            src[b] = A.kv + ((long long)(blk >= 0 ? blk : 0) * A.kv_sb + (long long)kh * A.kv_sh) * ESZ;
        }
#pragma unroll
        for (int i = 0; i < NPT; ++i) {
            const int b = (i * kThreads) / BLK_U4;   /* constant: BLK_U4 is a multiple of 256 */
            const int off = tid + i * kThreads - b * BLK_U4;
            if (pre_ok[b]) pre[i] = *(const w4_t*)((const uint4*)src[b] + off);
            else           pre[i] = zero4;
        }
    };

    const unsigned char* sb = (const unsigned char*)s_kv;
    if (kt_lo < kt_hi) fetch(kt_lo);
    for (int kt = kt_lo; kt < kt_hi; ++kt) {
        lds_barrier();                               /* the previous tile is consumed */
#pragma unroll
        for (int i = 0; i < NPT; ++i) ((w4_t*)s_kv)[tid + i * kThreads] = pre[i];
        const bool ok0 = pre_ok[0], ok1 = pre_ok[NB - 1];   /* values: see reduce_scatter */
        lds_barrier();
        const int k0 = lo_a + kt * BK;
        if (kt + 1 < kt_hi) fetch(kt + 1);

        const int jl = lane & (BK - 1);              /* the key this lane scores */
        const int kk = k0 + jl;
        const bool kok = (jl < 16 ? ok0 : ok1) && kk >= lo && kk < hi;

        auto row_step = [&](auto ic) __attribute__((always_inline)) {
            constexpr int i = decltype(ic)::value;
            if (!val[i]) return;                     /* uniform: one row, all lanes */
            float sc[BK];
#pragma unroll
            for (int j = 0; j < BK; ++j) {
                const unsigned char* row = sb + j * KEY_BYTES;
                float acc = 0.0f;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    float k[4];
                    lds4<KV8>(row, c * 128 + lane * 4, k);
#pragma unroll
                    for (int e = 0; e < 4; ++e) acc = __builtin_fmaf(qf[i][c][e], k[e], acc);
                }
                sc[j] = acc;
            }
            if constexpr (BK == 32) {
                reduce_scatter<16, BK>(sc, lane);
            } else {
#pragma unroll
                for (int j = 0; j < BK; ++j) sc[j] += xor_lane<16>(sc[j]);
                reduce_scatter<8, BK>(sc, lane);
            }
            const int pq = pos[i];
            bool vis = kok && (causal ? kk <= pq : true);
            if (window > 0) vis = vis && (pq - kk < window) && (causal || kk - pq < window);
            const float sv = vis ? sc[0] : -__builtin_inff();
            const float mn = __builtin_fmaxf(m[i], wave_max(sv));
            if (mn == -__builtin_inff()) return;     /* nothing visible yet: uniform */
            const float corr = ex2(m[i] - mn);
            const float p = vis ? ex2(sv - mn) : 0.0f;
            l[i] = l[i] * corr + wave_sum((BK == 16 && lane >= 16) ? 0.0f : p);
#pragma unroll
            for (int c = 0; c < NC; ++c)
#pragma unroll
                for (int e = 0; e < 4; ++e) o[i][c][e] *= corr;
#pragma unroll
            for (int j = 0; j < BK; ++j) {
                const float pj = readlane(p, j);     /* uniform; a masked key's V is never read */
                if (pj != 0.0f) {
                    const unsigned char* row = sb + j * KEY_BYTES;
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        float v[4];
                        lds4<KV8>(row, HD + c * 128 + lane * 4, v);
#pragma unroll
                        for (int e = 0; e < 4; ++e) o[i][c][e] = __builtin_fmaf(pj, v[e], o[i][c][e]);
                    }
                }
            }
            m[i] = mn;
        };
        StaticFor<0, RW>::run(row_step);
    }

    /* ---- out: normalised bf16 rows, or this split's partials */
#pragma unroll
    for (int i = 0; i < RW; ++i) {
        if (!val[i]) continue;
        const int r = wave * RW + i, p = r / GQA, g = r - p * GQA;
        if (A.n_split == 1) {
            const float inv = l[i] > 0.0f ? vds / l[i] : 0.0f;
            uint16_t* op = A.out + (long long)(tok0 + p) * A.o_rs + (long long)(kh * GQA + g) * A.o_hs;
#pragma unroll
            for (int c = 0; c < NC; ++c)
                *(uint2*)(op + c * 128 + lane * 4) =
                    make_uint2(pack2(o[i][c][0] * inv, o[i][c][1] * inv),
                               pack2(o[i][c][2] * inv, o[i][c][3] * inv));
        } else {
            const long long slot = (((long long)tile * A.kv_heads + kh) * A.n_split + z) * R + r;
            float* pp = A.part + slot * (HD + 4);
            if (lane == 0) { pp[HD] = m[i]; pp[HD + 1] = l[i]; }
            /* no key visible to this row in this split: the merge weighs it 0 and never reads its
             * accumulator, so it is not written (uniform: m is the wave's) */
            if (m[i] == -__builtin_inff()) continue;
#pragma unroll
            for (int c = 0; c < NC; ++c)
                *(float4*)(pp + c * 128 + lane * 4) =
                    make_float4(o[i][c][0] * vds, o[i][c][1] * vds, o[i][c][2] * vds, o[i][c][3] * vds);
        }
    }
}

/* ================================================================== the WMMA prefill kernel
 *
 * The scalar kernel is memory- and latency-bound on a long prefill, where a global layer's attention
 * is quadratic. This kernel uses the 16x16x16 WMMA instead, chosen only for prefill-shaped steps (a
 * sequence with >= kWmmaMinQ query rows); it computes the same function.
 *
 * Layout (wave32 v_wmma_f32_16x16x16, halo_wmma.h): the f32 result gives lane L column
 * N = L % 16 and rows M = 2 * e + (L / 16). So computing S^T = K * Q^T puts one query row on
 * (lane & 15) and gives a lane the eight keys of one parity, the other eight in lane ^ 16 (the
 * softmax statistics are one exchange, as in the scalar kernel), and those values as f16 are the B
 * operand of O^T = V^T * P^T. O^T's lane L also holds only column L % 16, so the rescale is a
 * per-lane scalar.
 *
 * Head split: O^T for 16 rows of a 512-dim head is 256 f32 per lane, the whole register file, so a
 * wave owns 16 rows by a 128-dim slice (kDQ): Q^T in 32 VGPRs, O^T in 64. Slices compute partial
 * S^T over their own dims; partials meet in LDS and every slice sums them in the same order, so all
 * see bit-identical scores. 16 waves: h512 is 4 row groups x 4 slices (64 rows), h256 8 x 2 (128).
 *
 * Tile = one paged block, 16 keys. K goes to LDS as [key][dim] bf16 with a 16-byte row pad so
 * A-fragment reads hit 32 banks; a gfx1151 A fragment is the whole 16-wide dim run (k = e), so the
 * gather is 32 bytes at the un-split offset. V goes transposed ([dim][key], 48-byte rows) because
 * P.V reduces over keys; two keys are packed per dword so a gfx1151 A fragment is eight dwords of
 * sixteen keys in order, and the dword index is rotated per lane group so a wave's 32 writes hit 32
 * banks. An fp8 cache is widened on load (exact). Keys outside the range or in a missing block are
 * written as zeros, since 0 * NaN from a recycled slot is still NaN.
 *
 * P and V in f16, Q and K in bf16: bf16 P leaves up to 2^-9 |v| error per output (up to 0.25% of the
 * V range against f64, 4x the scalar kernel), while f16 runs at the same rate with 11 bits. bf16 V
 * is exact in f16 within [2^-14, 65504] (clamped beyond, NaN kept); fp8 V is always exact. The
 * denominator sums P as rounded, so a row is a true weighted mean of V.
 */
constexpr int kWThreads = 512;
constexpr int kWWaves = kWThreads / 32;
constexpr int kDQ = 128;            /* dims of a wave's slice of the head */
constexpr int kWmmaMinQ = 64;       /* a launch whose longest query is shorter stays scalar */

template <int HD> struct WGeo {
    static constexpr int NS = HD / kDQ;           /* slices: waves sharing one 16-row group */
    static constexpr int RG = kWWaves / NS;       /* row groups a workgroup */
    static constexpr int R = RG * 16;             /* rows a workgroup */
    static constexpr int KS = HD * 2 + 16;        /* bytes a key row of the K tile */
    static constexpr int VS = 48;                 /* bytes a dim row of the V^T tile: 16 keys + pad */
};

/* Q,K are bf16 fragments and P,V are f16 ones; both go through halo_dt16.h's traits. */
typedef DT16<0> WQ;
typedef DT16<1> WP;

/* f32 -> f16 bits, round to nearest even; a finite value past f16's range saturates, NaN stays */
__device__ __forceinline__ unsigned f16_bits(float f) {
    if (f == f) f = __builtin_fminf(__builtin_fmaxf(f, -65504.0f), 65504.0f);
    return (unsigned)__builtin_bit_cast(unsigned short, (_Float16)f);
}
/* Eight bf16 (as bits) out of one 16-byte load of the cache, to the f16 dwords the V tile keeps. */
__device__ __forceinline__ void bf16x8_f16(const w4_t w, unsigned (&h)[8]) {
    const unsigned in[4] = { w.x, w.y, w.z, w.w };
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        h[2 * q] = f16_bits(bf16w_lo(in[q]));
        h[2 * q + 1] = f16_bits(bf16w_hi(in[q]));
    }
}
/* fp8 e4m3 -> bf16 bits (the K tile), exact: the f32 a byte widens to has nothing below bit 16 */
__device__ __forceinline__ void fp8x16_bf16(const w4_t w, unsigned (&h)[16]) {
    const unsigned in[4] = { w.x, w.y, w.z, w.w };
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        float f[4];
        fp8x4(in[q], f);
#pragma unroll
        for (int e = 0; e < 4; ++e) h[q * 4 + e] = __float_as_uint(f[e]) >> 16;
    }
}

template <int HD, int GQA, bool KV8>
__global__ __launch_bounds__(kWThreads) void attn_wmma_kernel(AttnArgs A) {
    using G = WGeo<HD>;
    constexpr int NS = G::NS, R = G::R, BQ = R / GQA, KS = G::KS, VS = G::VS;
    constexpr int ESZ = KV8 ? 1 : 2;
    constexpr int SLOT = 2 * HD * ESZ;            /* bytes of one key's slot: K then V */
    constexpr int CW = 16 / ESZ;                  /* cache elements one 16-byte load carries */
    constexpr int NCH = HD / CW;                  /* such chunks a head */
    constexpr int NKI = 16 * NCH;                 /* K items a tile: (key, chunk) */
    constexpr int NVI = 8 * NCH;                  /* V items a tile: (key pair, chunk) */
    constexpr int KPT = (NKI + kWThreads - 1) / kWThreads;
    constexpr int VPT = (NVI + kWThreads - 1) / kWThreads;
    constexpr int NKS = kDQ / 16;                 /* 16-wide dim runs a wave walks */
    static_assert(R % GQA == 0 && NS * G::RG == kWWaves, "rows must split over heads and waves");
    static_assert(KS % 16 == 0 && VS % 16 == 0, "16-byte LDS rows");

    __shared__ uint4 s_k[16 * KS / 16];
    __shared__ uint4 s_vt[HD * VS / 16];
    /* The slices' partial scores meet here. Head_dim 128 is one slice -- there is nothing to
     * exchange -- so the stage costs neither the 16 KiB nor a barrier per key tile there. */
    __shared__ float4 s_x[NS > 1 ? kWWaves * 32 * 2 : 1];
    __shared__ int s_scan[kWWaves];
    __shared__ int s_res[2];

    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int rg = wave / NS, sl = wave - rg * NS;
    const int tile = (int)blockIdx.x, kh = (int)blockIdx.y;

    /* ---- which sequence, and which slice of its query: as the scalar kernel, at this BQ */
    int s = -1, lt = 0;
    if (A.cu) {
        find_tile<BQ, kWThreads>(A.cu, A.n_seq, tile, s_scan, s_res, s, lt);
    } else {
        s = tile / A.tps;
        lt = tile - s * A.tps;
        if (s >= A.n_seq) s = -1;
    }
    int qlen = 0, tokb = 0, ctx = 0;
    if (s >= 0) {
        if (A.cu) { tokb = A.cu[s]; qlen = A.cu[s + 1] - tokb; }
        else      { tokb = s * A.q_len; qlen = A.q_len; }
        ctx = A.seqused[s];
    }
    const int pfirst = lt * BQ;
    const int npos = s >= 0 ? imax(0, imin(BQ, qlen - pfirst)) : 0;
    if (npos <= 0) return;                           /* uniform across the block */

    const int tok0 = tokb + pfirst;
    const int qbase = ctx - qlen + pfirst;
    const int plast = qbase + npos - 1;
    const int causal = A.causal, window = A.window;
    int hi = causal ? plast + 1 : ctx;
    if (!causal && window > 0) hi = imin(hi, plast + window);
    hi = imin(imin(hi, ctx), A.max_blocks * 16);
    const int lo = window > 0 ? imax(0, qbase - window + 1) : 0;
    const int lo_a = lo & ~15;
    const int nkt = hi > lo_a ? (hi - lo_a + 15) / 16 : 0;

    const float kds = A.kdes ? A.kdes[(long long)s * A.kv_heads + kh] : 1.0f;
    const float vds = A.vdes ? A.vdes[(long long)s * A.kv_heads + kh] : 1.0f;
    const float sscale = A.qscale * kds;             /* log2-domain softmax scale, k_descale in */

    /* ---- this lane's query row: column n of every S^T and O^T tile. gfx1151 gives a lane the whole
     * 16-wide dim run of its row (k = e), so the gather is 32 bytes at the slice's offset and the
     * lane half `h` selects nothing. */
    const int n = lane & 15, h = lane >> 4;
    const int r = rg * 16 + n, p = r / GQA, g = r - p * GQA;
    const bool rval = p < npos;
    const int pos = qbase + p;

    WQ::frag qf[NKS];
    {
        const uint16_t* qp = A.q + (long long)(tok0 + (rval ? p : 0)) * A.q_rs +
                             (long long)(kh * GQA + g) * A.q_hs + sl * kDQ;
#pragma unroll
        for (int kc = 0; kc < NKS; ++kc) {
            const uint4 u0 = rval ? *(const uint4*)(qp + kc * 16) : make_uint4(0u, 0u, 0u, 0u);
            const uint4 u1 = rval ? *(const uint4*)(qp + kc * 16 + 8) : make_uint4(0u, 0u, 0u, 0u);
            const uint32_t w8[8] = { u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w };
            qf[kc] = WQ::mk(w8);
        }
    }
    v8f oacc[NKS];
#pragma unroll
    for (int mc = 0; mc < NKS; ++mc)
#pragma unroll
        for (int e = 0; e < 8; ++e) oacc[mc][e] = 0.0f;
    float m = -__builtin_inff(), l = 0.0f;

    /* ---- the key tiles, one paged block each, prefetched one ahead through registers */
    w4_t pk[KPT], pv[VPT][2];
    const w4_t zero4 = {0u, 0u, 0u, 0u};
    bool pre_ok = false;
    auto fetch = [&](int kt) __attribute__((always_inline)) {
        const int k0 = lo_a + kt * 16, lb = k0 >> 4;
        int blk = -1;
        if (lb < A.max_blocks && k0 < hi) blk = A.bt[(long long)s * A.bt_rs + lb];
        pre_ok = blk >= 0;
        /* never a null: see the scalar kernel's fetch */
        const unsigned char* base =
            A.kv + ((long long)(blk >= 0 ? blk : 0) * A.kv_sb + (long long)kh * A.kv_sh) * ESZ;
#pragma unroll
        for (int j = 0; j < KPT; ++j) {
            const int idx = tid + j * kWThreads;
            pk[j] = zero4;
            if (pre_ok && idx < NKI) {
                const int key = idx / NCH, ch = idx - key * NCH;
                pk[j] = *(const w4_t*)(base + key * SLOT + ch * 16);
            }
        }
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            const int idx = tid + j * kWThreads;
            pv[j][0] = pv[j][1] = zero4;
            if (pre_ok && idx < NVI) {
                const int kp = idx & 7, ch = idx >> 3;
                const unsigned char* v0 = base + (2 * kp) * SLOT + HD * ESZ + ch * 16;
                pv[j][0] = *(const w4_t*)v0;
                pv[j][1] = *(const w4_t*)(v0 + SLOT);
            }
        }
    };

    unsigned char* kb = (unsigned char*)s_k;
    unsigned char* vb = (unsigned char*)s_vt;
    if (nkt > 0) fetch(0);
    for (int kt = 0; kt < nkt; ++kt) {
        const int k0 = lo_a + kt * 16;
        const bool tok = pre_ok;                     /* this tile's block is present */
        lds_barrier();                               /* the previous tile's LDS is consumed */

        /* ---- K: [key][dim] as bf16 */
#pragma unroll
        for (int j = 0; j < KPT; ++j) {
            const int idx = tid + j * kWThreads;
            if (idx < NKI) {
                const int key = idx / NCH, ch = idx - key * NCH;
                const int kk = k0 + key;
                const bool keep = tok && kk >= lo && kk < hi;
                const w4_t w = keep ? pk[j] : zero4;
                if constexpr (KV8) {
                    unsigned hb[16];
                    fp8x16_bf16(w, hb);
                    uint4* dst = (uint4*)(kb + key * KS + ch * 32);
                    dst[0] = make_uint4(hb[0] | hb[1] << 16, hb[2] | hb[3] << 16,
                                        hb[4] | hb[5] << 16, hb[6] | hb[7] << 16);
                    dst[1] = make_uint4(hb[8] | hb[9] << 16, hb[10] | hb[11] << 16,
                                        hb[12] | hb[13] << 16, hb[14] | hb[15] << 16);
                } else {
                    *(w4_t*)(kb + key * KS + ch * 16) = w;
                }
            }
        }
        /* ---- V: [dim][key] as f16, two keys a dword; the row order rotated by 2 per 8-lane
         * group, so the four groups of a wave write four different rows mod 8 -- 32 banks */
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            const int idx = tid + j * kWThreads;
            if (idx < NVI) {
                const int kp = idx & 7, ch = idx >> 3;
                const int ka = k0 + 2 * kp, kb1 = ka + 1;
                const w4_t wa = (tok && ka >= lo && ka < hi) ? pv[j][0] : zero4;
                const w4_t wb = (tok && kb1 >= lo && kb1 < hi) ? pv[j][1] : zero4;
                unsigned d[CW];                      /* f16 pairs: key 2kp low, 2kp + 1 high */
                if constexpr (KV8) {
                    /* every e4m3 value is exact in f16 (|v| <= 448, subnormals 2^-9 up), so the
                     * pack's round-toward-zero leaves the same bits */
                    const unsigned ia[4] = {wa.x, wa.y, wa.z, wa.w};
                    const unsigned ib[4] = {wb.x, wb.y, wb.z, wb.w};
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        float fa[4], fb[4];
                        fp8x4(ia[q], fa);
                        fp8x4(ib[q], fb);
#pragma unroll
                        for (int e = 0; e < 4; ++e) d[q * 4 + e] = WP::pk(fa[e], fb[e]);
                    }
                } else {
                    unsigned ha[8], hb2[8];
                    bf16x8_f16(wa, ha);
                    bf16x8_f16(wb, hb2);
#pragma unroll
                    for (int e = 0; e < CW; ++e) d[e] = ha[e] | hb2[e] << 16;
                }
                const int rot = 2 * ((lane >> 3) & 3);
                unsigned* col = (unsigned*)(vb + (ch * CW) * VS) + kp;
#pragma unroll
                for (int t = 0; t < CW; ++t) {
                    /* element (t + rot) % CW, chosen by value: rot is per lane */
                    const unsigned v0 = d[t], v2 = d[(t + 2) % CW], v4 = d[(t + 4) % CW];
                    const unsigned v6 = d[(t + 6) % CW];
                    const unsigned v = rot == 0 ? v0 : rot == 2 ? v2 : rot == 4 ? v4 : v6;
                    const int e = (t + rot) % CW;
                    col[e * (VS / 4)] = v;
                }
            }
        }
        lds_barrier();
        if (kt + 1 < nkt) fetch(kt + 1);

        /* ---- S^T partial over this wave's slice: A = K rows (key n), B = Q^T (row n) */
        v8f sacc;
#pragma unroll
        for (int e = 0; e < 8; ++e) sacc[e] = 0.0f;
        {
            /* One accumulator: a second hides the chain latency but costs 12 VGPRs, and the extra
             * waves hide latency better than the shorter chain does. */
            const unsigned char* ka = kb + n * KS + sl * kDQ * 2;
#pragma unroll
            for (int kc = 0; kc < NKS; ++kc) {
                const uint4 u0 = *(const uint4*)(ka + kc * 32);
                const uint4 u1 = *(const uint4*)(ka + kc * 32 + 16);
                const uint32_t w8[8] = { u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w };
                sacc = WQ::wmma(WQ::mk(w8), qf[kc], sacc);
            }
        }
        /* ---- the slices' partials meet: every slice sums them in slice order */
        float tot[8];
        if constexpr (NS == 1) {
#pragma unroll
            for (int e = 0; e < 8; ++e) tot[e] = sacc[e];
        } else {
            s_x[(wave * 2 + 0) * 32 + lane] = make_float4(sacc[0], sacc[1], sacc[2], sacc[3]);
            s_x[(wave * 2 + 1) * 32 + lane] = make_float4(sacc[4], sacc[5], sacc[6], sacc[7]);
            __syncthreads();
#pragma unroll
            for (int j = 0; j < NS; ++j) {
                float v[8];
                if (j == sl) {                       /* uniform: the wave's own, as written */
#pragma unroll
                    for (int e = 0; e < 8; ++e) v[e] = sacc[e];
                } else {
                    const float4 a = s_x[((rg * NS + j) * 2 + 0) * 32 + lane];
                    const float4 b = s_x[((rg * NS + j) * 2 + 1) * 32 + lane];
                    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
                    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) tot[e] = j == 0 ? v[e] : tot[e] + v[e];
            }
        }

        /* ---- online softmax, row n: eight keys here, eight in lane ^ 16. Element e is key
         * 2*e + h on gfx1151 (halo_dt16.h), the two halves covering a tile between them. */
        float sv[8], mx = -__builtin_inff();
        /* A tile wholly inside [lo, hi) and, causal, at or before the block's first position
         * (every row's pos >= qbase) is visible to every valid row: the same scores, unmasked. */
        const bool interior = window <= 0 && k0 >= lo && k0 + 16 <= hi && (!causal || k0 + 15 <= qbase);
        if (interior) {
            const bool vis = rval && tok;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                sv[e] = vis ? tot[e] * sscale : -__builtin_inff();
                mx = __builtin_fmaxf(mx, sv[e]);
            }
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int kk = k0 + 2 * e + h;
                bool vis = rval && tok && kk >= lo && kk < hi;
                if (causal) vis = vis && kk <= pos;
                if (window > 0) vis = vis && (pos - kk < window) && (causal || kk - pos < window);
                sv[e] = vis ? tot[e] * sscale : -__builtin_inff();
                mx = __builtin_fmaxf(mx, sv[e]);
            }
        }
        mx = __builtin_fmaxf(mx, swap16(mx));
        const float mn = __builtin_fmaxf(m, mx);
        const float corr = mn == -__builtin_inff() ? 1.0f : ex2(m - mn);
        /* P in f16, and the denominator summed from P as rounded (see the head of this kernel) */
        uint32_t pw[4];
#pragma unroll
        for (int e = 0; e < 8; e += 2) {
            const float p0 = sv[e] == -__builtin_inff() ? 0.0f : ex2(sv[e] - mn);
            const float p1 = sv[e + 1] == -__builtin_inff() ? 0.0f : ex2(sv[e + 1] - mn);
            pw[e >> 1] = WP::pk(p0, p1);             /* in [0, 1]: no saturation to worry about */
        }
        /* The running denominator is rescaled BEFORE this tile's P joins it: l = l*corr + sum(P),
         * summed from the f16 words P.V actually multiplies. */
        l *= corr;
        {
            const v2hf_dt one = {(__fp16)1.0f, (__fp16)1.0f};
#pragma unroll
            for (int e = 0; e < 4; ++e)
                l = __builtin_amdgcn_fdot2(__builtin_bit_cast(v2hf_dt, pw[e]), one, l, false);
        }
        m = mn;
        /* gfx1151's B fragment wants the sixteen keys of the tile in order and a lane holds one
         * parity of them: p16w (halo_dt16.h) interleaves the half-pair's eight values. */
        uint32_t pw16[8];
        p16w<WP>(pw, pw16);
        const WP::frag pf = WP::mk(pw16);

        /* ---- O^T += V^T * P^T in f16: A = V^T rows (dim n of the chunk), the sixteen keys */
        const unsigned char* va = vb + (sl * kDQ + n) * VS;
        if constexpr (KV8) {
            /* x * 1 = x, so a wave whose rows all kept their max skips the rescale. With an fp8
             * cache that is ~4% at head_dim 512; with bf16 the split loop measured ~10% slower,
             * so it keeps the fused one. */
            if (__builtin_amdgcn_ballot_w32(corr != 1.0f) != 0) {
#pragma unroll
                for (int mc = 0; mc < NKS; ++mc) oacc[mc] *= corr;
            }
#pragma unroll
            for (int mc = 0; mc < NKS; ++mc) {
                const uint4 u0 = *(const uint4*)(va + mc * 16 * VS);
                const uint4 u1 = *(const uint4*)(va + mc * 16 * VS + 16);
                const uint32_t w8[8] = { u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w };
                oacc[mc] = WP::wmma(WP::mk(w8), pf, oacc[mc]);
            }
        } else {
#pragma unroll
            for (int mc = 0; mc < NKS; ++mc) {
                const uint4 u0 = *(const uint4*)(va + mc * 16 * VS);
                const uint4 u1 = *(const uint4*)(va + mc * 16 * VS + 16);
                const uint32_t w8[8] = { u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w };
                oacc[mc] *= corr;
                oacc[mc] = WP::wmma(WP::mk(w8), pf, oacc[mc]);
            }
        }
    }

    /* ---- out: the row's denominator is its two lanes' partial sums. O^T's element e is head dim
     * 2*e + h (halo_dt16.h), so the store is a scatter of eight 16-bit values a chunk. */
    const float lt2 = l + swap16(l);
    if (rval) {
        const float inv = lt2 > 0.0f ? vds / lt2 : 0.0f;
        uint16_t* op = A.out + (long long)(tok0 + p) * A.o_rs + (long long)(kh * GQA + g) * A.o_hs +
                       sl * kDQ;
#pragma unroll
        for (int mc = 0; mc < NKS; ++mc)
#pragma unroll
            for (int e = 0; e < 8; ++e) op[mc * 16 + 2 * e + h] = bf16_of(oacc[mc][e] * inv);
    }
}

/* ------------------------------------------------------------------ the split-KV merge
 *
 * One wave per row, splits across lanes: lane z holds split z's (max, sum) and z + 32's
 * (kMaxSplit is 64), so the row max and denominator are one wave reduction each. Then one sweep
 * over the splits that saw a key, four at a time, each lane accumulating its 4 * NC dims. */
template <int HD, int GQA>
__device__ __forceinline__ void merge_body(AttnArgs A) {
    constexpr int R = Geo<HD>::R, NC = HD / 128;
    static_assert(kMaxSplit <= 64, "two splits a lane");
    const int tile = (int)blockIdx.x, kh = (int)blockIdx.y;
    const int lane = lane_id(), wave = (int)(threadIdx.x >> 5);
    const int tok0 = A.tinfo[2 * tile], npos = A.tinfo[2 * tile + 1];
    const int ns = A.n_split;
    const float ninf = -__builtin_inff();
    for (int r = wave; r < npos * GQA; r += kWaves) {
        const int p = r / GQA, g = r - p * GQA;
        const long long base = ((long long)tile * A.kv_heads + kh) * ns * R + r;
        uint16_t* op = A.out + (long long)(tok0 + p) * A.o_rs + (long long)(kh * GQA + g) * A.o_hs;
        float m0 = ninf, m1 = ninf, l0 = 0.0f, l1 = 0.0f;
        if (lane < ns) {
            const float* pp = A.part + (base + (long long)lane * R) * (HD + 4);
            m0 = pp[HD]; l0 = pp[HD + 1];
        }
        if (lane + 32 < ns) {
            const float* pp = A.part + (base + (long long)(lane + 32) * R) * (HD + 4);
            m1 = pp[HD]; l1 = pp[HD + 1];
        }
        const float mt = wave_max(__builtin_fmaxf(m0, m1));
        if (mt == ninf) {                            /* no key visible to the row: a zero row */
#pragma unroll
            for (int c = 0; c < NC; ++c) *(uint2*)(op + c * 128 + lane * 4) = make_uint2(0u, 0u);
            continue;
        }
        const float w0 = ex2(m0 - mt), w1 = ex2(m1 - mt);    /* an empty split: ex2(-inf) = 0 */
        const float lt = wave_sum(__builtin_fmaf(l0, w0, l1 * w1));
        const float inv = lt > 0.0f ? 1.0f / lt : 0.0f;
        float acc[NC][4];
#pragma unroll
        for (int c = 0; c < NC; ++c)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[c][e] = 0.0f;
        /* Splits with weight, as a wave-uniform 64-bit set, taken four at a time so 4 * NC loads
         * are in flight before the first FMA. A short group repeats its first split at weight 0
         * (a written split, so 0 * finite). An empty split's accumulator is never read. */
        uint64_t live = (uint64_t)__builtin_amdgcn_ballot_w32(w0 != 0.0f) |
                        ((uint64_t)__builtin_amdgcn_ballot_w32(w1 != 0.0f) << 32);
        /* Group g + 1's loads go out before group g's FMAs, so a group's wait overlaps the one
         * before it instead of following it; the FMAs keep their order, so the bytes do too. */
        auto take = [&](int (&zi)[4], float (&wj)[4]) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const bool have = live != 0;
                zi[j] = have ? __builtin_ctzll(live) : zi[0];
                live &= have ? live - 1 : live;
                wj[j] = readlane(zi[j] < 32 ? w0 : w1, zi[j] & 31);
                if (!have) wj[j] = 0.0f;
            }
        };
        auto load = [&](const int (&zi)[4], float4 (&v)[4][NC]) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float* pp = A.part + (base + (long long)zi[j] * R) * (HD + 4);
#pragma unroll
                for (int c = 0; c < NC; ++c) v[j][c] = *(const float4*)(pp + c * 128 + lane * 4);
            }
        };
        auto fold = [&](const float (&wj)[4], const float4 (&v)[4][NC]) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    acc[c][0] = __builtin_fmaf(wj[j], v[j][c].x, acc[c][0]);
                    acc[c][1] = __builtin_fmaf(wj[j], v[j][c].y, acc[c][1]);
                    acc[c][2] = __builtin_fmaf(wj[j], v[j][c].z, acc[c][2]);
                    acc[c][3] = __builtin_fmaf(wj[j], v[j][c].w, acc[c][3]);
                }
        };
        if (live) {
            int za[4], zb[4];
            float wa[4], wb[4];
            float4 va[4][NC], vb[4][NC];
            take(za, wa);
            load(za, va);
            while (true) {
                const bool more_b = live != 0;
                if (more_b) { take(zb, wb); load(zb, vb); }
                fold(wa, va);
                if (!more_b) break;
                const bool more_a = live != 0;
                if (more_a) { take(za, wa); load(za, va); }
                fold(wb, vb);
                if (!more_a) break;
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c)
            *(uint2*)(op + c * 128 + lane * 4) =
                make_uint2(pack2(acc[c][0] * inv, acc[c][1] * inv),
                           pack2(acc[c][2] * inv, acc[c][3] * inv));
    }
}

/* ------------------------------------------------------------------ host: the launch adapter */

/* q and out as [rows, heads, head_dim] (or [rows, heads * head_dim]) at their own row pitch: a
 * column slice of a fused projection, in whole heads. The last axis must be dense. */
template <int HD>
static constexpr int64_t part_bytes() {
    return (int64_t)kPartBudget * Geo<HD>::R * (HD + 4) * (int64_t)sizeof(float);
}
template <int HD>
static constexpr int64_t scratch_need() {
    return part_bytes<HD>() + (int64_t)kTinfoMax * 2 * (int64_t)sizeof(int);
}

/* The plain kernels and their chained forms (halo_chain.h, RadKernelChainFn): the chained ones wait
 * for the kernels before them on the queue in their prologue and count themselves in at the end. */
template <int HD, int GQA, bool KV8>
__global__ __launch_bounds__(kThreads) void attn_kernel(AttnArgs A) { attn_body<HD, GQA, KV8>(A); }
template <int HD, int GQA, bool KV8>
__global__ __launch_bounds__(kThreads) void halo_win_r4dchain_attn(AttnArgs A) {
    halo_chain_wait();
    attn_body<HD, GQA, KV8>(A);
    halo_chain_done();
}
template <int HD, int GQA>
__global__ __launch_bounds__(kThreads) void merge_kernel(AttnArgs A) { merge_body<HD, GQA>(A); }
template <int HD, int GQA>
__global__ __launch_bounds__(kThreads) void halo_win_r4dchain_merge(AttnArgs A) {
    halo_chain_wait();
    merge_body<HD, GQA>(A);
    halo_chain_done();
}

template <int HD, int GQA, bool KV8>
static int launch_attn(const RadArgs* a, RadStream stream) {
    constexpr int R = Geo<HD>::R, BK = Geo<HD>::BK, BQ = R / GQA, ESZ = KV8 ? 1 : 2;
    if (!a || a->n_t < 8) return HALO_NO(RAD_E_INVAL);
    const RadTensor* q  = halo_opt(a, AP_Q);
    const RadTensor* kv = halo_opt(a, AP_KV);
    const RadTensor* bt = halo_opt(a, AP_BT);
    const RadTensor* su = halo_opt(a, AP_SEQ);
    const RadTensor* kd = halo_opt(a, AP_KDS);
    const RadTensor* vd = halo_opt(a, AP_VDS);
    const RadTensor* cu = halo_opt(a, AP_CU);
    const RadTensor* o  = halo_opt(a, AP_OUT);
    if (!q || !kv || !bt || !su || !o) return HALO_NO(RAD_E_INVAL);

    if (halo_geti_or(a, "head_dim", HD) != HD || halo_geti_or(a, "gqa", GQA) != GQA ||
        halo_geti_or(a, "block_size", 16) != 16)
        return HALO_NO(RAD_E_SHAPE);
    const long long causal = halo_geti_or(a, "causal", 1);
    const long long window = halo_geti_or(a, "window", 0);
    if ((causal != 0 && causal != 1) || window < 0 || window > (1ll << 30)) return HALO_NO(RAD_E_SHAPE);

    if (q->dtype != RAD_BF16 || o->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    if (kv->dtype != (uint32_t)(KV8 ? RAD_F8E4M3 : RAD_BF16)) return HALO_NO(RAD_E_DTYPE);
    if (bt->dtype != RAD_I32 || su->dtype != RAD_I32 || (cu && cu->dtype != RAD_I32))
        return HALO_NO(RAD_E_DTYPE);
    if ((kd && kd->dtype != RAD_F32) || (vd && vd->dtype != RAD_F32)) return HALO_NO(RAD_E_DTYPE);

    /* the cache: [n_blocks, kv_heads, 16, 2 * head_dim], slots dense */
    if (kv->rank != 4 || kv->shape[2] != 16 || kv->shape[3] != 2 * HD) return HALO_NO(RAD_E_SHAPE);
    if (kv->stride[3] != 1 || kv->stride[2] != 2 * HD) return HALO_NO(RAD_E_STRIDE);
    const int64_t kvh = kv->shape[1];
    if (kvh <= 0 || kvh > 65535) return HALO_NO(RAD_E_SHAPE);
    const int64_t qh = kvh * GQA;

    /* q and out: [rows, heads * head_dim], each at its own row pitch (attn_paged_qrow reads q as a
     * column slice of a fused q|k|v projection). */
    long long rows = 0, q_rs = 0, q_hs = 0, orows = 0, o_rs = 0, o_hs = 0, qhn = 0, ohn = 0;
    if (!halo_head_rows(q, HD, &rows, &qhn, &q_rs) || !halo_head_rows(o, HD, &orows, &ohn, &o_rs))
        return HALO_NO(RAD_E_SHAPE);
    q_hs = o_hs = HD;
    if (qhn != qh || ohn != qh) return HALO_NO(RAD_E_SHAPE);
    if (orows < rows) return HALO_NO(RAD_E_SHAPE);
    const int64_t n_seq = halo_numel(su);
    if (rows == 0) return RAD_OK;
    if (n_seq <= 0 || n_seq > (1 << 24)) return HALO_NO(RAD_E_SHAPE);

    /* the block table: [n_seq, max_blocks] with a row pitch, or flat */
    int64_t max_blocks = 0, bt_rs = 0;
    if (bt->rank >= 2) {
        if (bt->stride[bt->rank - 1] != 1) return HALO_NO(RAD_E_STRIDE);
        max_blocks = bt->shape[bt->rank - 1];
        bt_rs = bt->stride[bt->rank - 2];
        if (halo_numel(bt) / max_blocks < n_seq) return HALO_NO(RAD_E_SHAPE);
    } else {
        max_blocks = halo_numel(bt) / n_seq;
        bt_rs = max_blocks;
    }
    if (max_blocks <= 0 || max_blocks > (1 << 26)) return HALO_NO(RAD_E_SHAPE);
    if ((kd && halo_numel(kd) < n_seq * kvh) || (vd && halo_numel(vd) < n_seq * kvh))
        return HALO_NO(RAD_E_SHAPE);

    /* 8-byte query/output chunks, 16-byte cache loads */
    if (!halo_aligned(q->data, 8) || !halo_aligned(o->data, 8) || (q_rs | q_hs | o_rs | o_hs) % 4)
        return HALO_NO(RAD_E_ALIGN);
    if (!halo_aligned(kv->data, 16) || (kv->stride[0] * ESZ) % 16 || (kv->stride[1] * ESZ) % 16)
        return HALO_NO(RAD_E_ALIGN);

    /* the tiles: uniform, or an upper bound over a ragged batch (empty tiles exit at once) */
    int64_t nt = 0, tps = 0, qlen_u = 0;
    if (!cu) {
        if (rows % n_seq) return HALO_NO(RAD_E_SHAPE);
        qlen_u = rows / n_seq;
        tps = (qlen_u + BQ - 1) / BQ;
        nt = n_seq * tps;
    } else {
        if (halo_numel(cu) != n_seq + 1) return HALO_NO(RAD_E_SHAPE);
        int64_t qmax = halo_geti_or(a, "q_len", rows);
        if (qmax <= 0 || qmax > rows) qmax = rows;
        const int64_t b1 = (rows + n_seq * (BQ - 1)) / BQ;
        const int64_t b2 = n_seq * ((qmax + BQ - 1) / BQ);
        nt = b1 < b2 ? b1 : b2;
    }
    if (nt <= 0) return RAD_OK;
    if (nt > 0x7fffffff) return HALO_NO(RAD_E_SHAPE);

    AttnArgs A{};
    A.q = (const uint16_t*)q->data;           A.q_rs = q_rs;  A.q_hs = q_hs;
    A.kv = (const unsigned char*)kv->data;    A.kv_sb = kv->stride[0];  A.kv_sh = kv->stride[1];
    A.bt = (const int*)bt->data;              A.bt_rs = bt_rs;
    A.seqused = (const int*)su->data;
    A.cu = cu ? (const int*)cu->data : nullptr;
    A.kdes = kd ? (const float*)kd->data : nullptr;
    A.vdes = vd ? (const float*)vd->data : nullptr;
    A.out = (uint16_t*)o->data;               A.o_rs = o_rs;  A.o_hs = o_hs;
    A.max_blocks = (int)max_blocks;
    A.n_seq = (int)n_seq;
    A.q_len = (int)qlen_u;
    A.kv_heads = (int)kvh;
    A.causal = (int)causal;
    A.window = (int)window;
    const double scale = halo_getf(a, "scale", 1.0 / __builtin_sqrt((double)HD));
    A.qscale = (float)(scale * 1.4426950408889634);
    A.n_split = 1;

    /* WMMA path for prefill-shaped steps: longest query >= kWmmaMinQ rows and mean >= half that.
     * Decode and verify batches keep the scalar kernel and its split-KV. Needs 16-byte aligned
     * query and output rows. Both agree with libref's ref_attn_paged to a few 1e-3 rel_l2 (the
     * standalone check in the lane's log) and differ from each other only in summation order. */
    {
        int64_t qmax_w = qlen_u;
        if (cu) {
            qmax_w = halo_geti_or(a, "q_len", rows);
            if (qmax_w <= 0 || qmax_w > rows) qmax_w = rows;
        }
        const bool shaped = qmax_w >= kWmmaMinQ && rows >= n_seq * (kWmmaMinQ / 2);
        const bool aligned = halo_aligned(q->data, 16) && halo_aligned(o->data, 16) &&
                             (q_rs | q_hs | o_rs | o_hs) % 8 == 0;
        if (shaped && aligned) {
            constexpr int BQW = WGeo<HD>::R / GQA;
            int64_t ntw = 0, tpsw = 0;
            if (!cu) {
                tpsw = (qlen_u + BQW - 1) / BQW;
                ntw = n_seq * tpsw;
            } else {
                const int64_t b1 = (rows + n_seq * (BQW - 1)) / BQW;
                const int64_t b2 = n_seq * ((qmax_w + BQW - 1) / BQW);
                ntw = b1 < b2 ? b1 : b2;
            }
            if (ntw <= 0) return RAD_OK;
            if (ntw > 0x7fffffff) return HALO_NO(RAD_E_SHAPE);
            AttnArgs W = A;
            W.tps = (int)tpsw;
            HALO_KLAUNCH(HALO_KNAME(attn_wmma_kernel<HD, GQA, KV8>),
                         dim3((unsigned)ntw, (unsigned)kvh, 1u), dim3(kWThreads),
                         (hipStream_t)stream, W);
            return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
        }
    }

    /* Split-KV only when (tile, kv head) pairs are too few to fill the GPU */
    int64_t ns = 1;
    const int64_t pairs = nt * kvh;
    if (pairs < kTinfoMax) {
        ns = kPartBudget / pairs;
        if (ns > kMaxSplit) ns = kMaxSplit;
        const int64_t span = window > 0 ? window + BQ
                                        : halo_geti_or(a, "max_ctx", max_blocks * 16);
        const int64_t nkt = (span + BK - 1) / BK + 1;
        if (ns > nkt / 2) ns = nkt / 2;
        if (ns < 1) ns = 1;
        if (ns > 1 && (!a->scratch || a->scratch_bytes < scratch_need<HD>() ||
                       !halo_aligned(a->scratch, 16)))
            ns = 1;   /* never wrong, only slower: the unsplit kernel writes the rows itself */
    }

    A.part = ns > 1 ? (float*)a->scratch : nullptr;
    A.tinfo = ns > 1 ? (int*)((char*)a->scratch + part_bytes<HD>()) : nullptr;
    A.tps = (int)tps;
    A.n_split = (int)ns;

    hipStream_t s = (hipStream_t)stream;
    /* The chained forms where the row resolved with `chain` (launches chain), at decode heights
     * only: the 31B's full layers at a verify. */
    const bool chain = halo_geti_or(a, "chain", 0) != 0 && rows <= 64;
    if (chain)
        HALO_KLAUNCH(HALO_KNAME(halo_win_r4dchain_attn<HD, GQA, KV8>),
                     dim3((unsigned)nt, (unsigned)kvh, (unsigned)ns), dim3(kThreads), s, A);
    else
        HALO_KLAUNCH(HALO_KNAME(attn_kernel<HD, GQA, KV8>),
                     dim3((unsigned)nt, (unsigned)kvh, (unsigned)ns), dim3(kThreads), s, A);
    if (ns > 1) {
        if (chain)
            HALO_KLAUNCH(HALO_KNAME(halo_win_r4dchain_merge<HD, GQA>),
                         dim3((unsigned)nt, (unsigned)kvh), dim3(kThreads), s, A);
        else
            HALO_KLAUNCH(HALO_KNAME(merge_kernel<HD, GQA>), dim3((unsigned)nt, (unsigned)kvh),
                         dim3(kThreads), s, A);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

}  /* namespace halo_win */

/* ------------------------------------------------------------------ the exported entry points */
extern "C" {

#define HALO_WIN_DEF(HD, G)                                                                        \
    int halo_rad_attn_window_h##HD##_gqa##G##_bf16(const RadArgs* a, RadStream s) {                \
        return halo_win::launch_attn<HD, G, false>(a, s);                                          \
    }                                                                                              \
    int halo_rad_attn_window_h##HD##_gqa##G##_fp8(const RadArgs* a, RadStream s) {                 \
        return halo_win::launch_attn<HD, G, true>(a, s);                                           \
    }
HALO_WIN_DEF(128, 1)
HALO_WIN_DEF(256, 1) HALO_WIN_DEF(256, 2) HALO_WIN_DEF(256, 4) HALO_WIN_DEF(256, 8)
HALO_WIN_DEF(512, 1) HALO_WIN_DEF(512, 2) HALO_WIN_DEF(512, 4) HALO_WIN_DEF(512, 8)
#undef HALO_WIN_DEF

/* Called at declare with no tensors. Partials are capped at kPartBudget (tile, head, split)
 * triples for any batch, so the size is a constant. */
int64_t halo_rad_attn_window_h128_scratch(const RadArgs*) { return halo_win::scratch_need<128>(); }
int64_t halo_rad_attn_window_h256_scratch(const RadArgs*) { return halo_win::scratch_need<256>(); }
int64_t halo_rad_attn_window_h512_scratch(const RadArgs*) { return halo_win::scratch_need<512>(); }

}  /* extern "C" */
