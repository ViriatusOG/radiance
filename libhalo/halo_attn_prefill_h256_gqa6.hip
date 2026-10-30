/* Derived from libr4d/r4d_attn_prefill_h256_gqa6.hip by the libhalo rename transform
 * (2026-10-26), then ported to gfx1151; see libhalo/README.md and libhalo/halo_dt16.h. */
// Halo prefill. Transposed core: S^T = K.Q^T, one lane owns one query row, ROWS = 16*NWARPS,
// 128-VGPR accumulator. The query is bf16 and the KV cache is fp8 or bf16; the 16-bit operand
// format for both matmuls is f16 (see halo_dt16.h for why).
//
// gfx1151 port of libr4d's body. The score fragment keeps its COLUMN on (lane & 15) -- the query --
// exactly as on gfx1201, so the softmax state, the swap16 row reductions and the whole control flow
// are unchanged. What moves is the ROW: on gfx12 a lane's accumulator element e is key
// 8*half + e of the sixteen-key tile, on gfx11 it is key 2*e + half. Four sites change:
//
//   gather   one 32-byte read of the whole sixteen-value k run, with no 8*half term (both the
//            O_CONTIG and the split map collapse into this one form);
//   mask     key = kbase + 2e + half;
//   P        the WMMA's A operand wants the sixteen keys in order, so the two parities are
//            assembled across the half-pair (halo_dt16.h's p16w);
//   store    element e is head dim 2e + half, so each lane scatters its eight values.
//
// Two gfx1151 facts the gfx12 body had no use for, both probed on the part (a 16x16x16 WMMA with
// garbage planted in chosen lanes): the low half-wave computes D's even rows from its own lanes'
// A and B, the high half the odd rows from theirs. So B must be replicated in both halves, but A
// row r is read only from lane r (r even) or 16 + r (r odd). Two uses:
//
//   QX       at head_dim 256 the query is held once, not twice (see QX below): the query and the
//            accumulator no longer fill the whole register file, and the m-tile loop stops
//            re-reading ~70 spilled query VGPRs from scratch every sixteen keys.
//   arow     a lane reads A's row (c & ~1) | half, the row its lane pair's target needs: the pair
//            reads one address (an LDS broadcast), so a fragment costs 512 bytes of LDS bandwidth
//            rather than 1 KB (QX's per-half dims would otherwise make the halves' reads distinct).
//
// O_QK8 and O_PV8 run Q.K^T and P.V over e4m3 operands, which gfx1201 does as an fp8 WMMA and
// gfx1151 has no instruction for (halo_fp8_wmma.h). They are EMULATED here instead of dropped:
// every operand is rounded to e4m3 exactly as the gfx12 leg rounds it, then widened back to the
// f16 the WMMA takes. e4m3 has 3 mantissa bits, so the widening is exact and each product is
// representable in f16 (<= 8 significand bits), which makes this the same arithmetic the fp8 WMMA
// did -- the same loss, to the last rounding, and no loss of its own.
//
//   QK8  Q rounded to e4m3 once at load, after the folded scale (the gfx12 site, moved from the
//        pk_fp8x4 there to halo_qcvt's E8 path here). K needs no rounding over the fp8 and fp8_v4
//        caches, whose K bytes ARE e4m3 and whose exact staging already holds those values; over
//        k6_v4, whose K is int6, the widen-then-round-once the byte staging does is applied at
//        the f16 staging instead (storeK).
//   PV8  P rounded to e4m3 from the f32 the softmax produced (pk_fp8x4's rounding), and V rounded
//        to e4m3 where the staging widens it with a scale (fp8_v4/k6_v4's int4 -> f16 * group
//        scale -> e4m3); the fp8 cache's V bytes are already e4m3. GROW is 8 octaves (e4m3's
//        largest value is 448 = 2^8.8) and DOT2 is off, both as in libr4d.
//
// The cost against the exact row is the rounding (the row's own accuracy, measured by the
// selftest) and one e4m3 encode + decode per operand value; the WMMA count is unchanged, so the
// leg buys no speed on this part, where the exact row already runs the same 16-bit WMMA.
//
// Each option below can be turned off on its own after a geometry change:
//
//   LAZY  (always on) Softmax against a reference max instead of a running one:
//         p = 2^(s - m_ref + SHIFT), with m_ref updated and the 128-VGPR accumulator rescaled only
//         when a row max exceeds m_ref by more than the P format absorbs (DT16::GROW octaves),
//         instead of on every m-tile. Same relative weighting in an f32 accumulator.
//   F16   f16 as the 16-bit operand format. gfx1151 has no bf16 convert either; v_cvt_pkrtz_f16_f32
//         converts two values in one. Exact for these operands (fp8 has 3 mantissa bits, bf16 7,
//         f16 10).
//   MSKIP Causal masking only on m-tiles that straddle the limit.
//   FOLDQ scale * k_descale * log2(e) folded into the query once at load.
//   CONTIG The WMMA k axis is only contracted, so relabelling it on both operands is free. On
//         gfx1151 the gather reads the lane's whole sixteen-value run either way; the bit is kept
//         in the option masks because the launchers still carry it, and it selects the padded row
//         stride (the 32-byte reads want it).
//   PF/SGB Load PF fragments before the matching PF WMMAs, with a scheduling barrier between the
//         groups. Without the barrier PF has no effect.
//   DOT2  v_dot2_f32_f16 row sum from the packed f16 P: 4 instructions instead of 8, and the
//         denominator uses the p values the PV matmul used.
//   BTS/KWIDE/GPREV/KLATE  tile staging; see each define.
//   QK8/PV8  Lossy e4m3 legs, emulated on the f16 WMMA (see the header).
#include "halo.h"
#include "halo_common.h"
#include "halo_dt16.h"

#include <algorithm>
#include <cmath>

#define O_F16     1
#define O_MSKIP   2
#define O_QK8    32
#define O_PV8    64
#define O_FOLDQ   8
#define O_CONTIG  512
#define O_PF2    1024
#define O_PF4    2048
#define O_PF8    3072
#define O_SGB    4096
#define O_DOT2  16384
// BTS: a tile's TILE/BS block-table entries are at a wave-uniform index, so load them into SGPRs
// once per tile. Otherwise each staging pass does two serialised memory round trips (block table,
// then KV) inside a barrier-bounded phase with nothing else to run.
#define O_BTS   262144
// KWIDE: one 16-byte K fetch per thread instead of two 8-byte ones. 768 threads x 16 B is exactly
// a 48-key fp8 tile, so staging is straight-line with half the loads and block-table lookups.
// key = tid/16 would make a warp write one row at a 32-byte stride (4-way LDS bank conflict, since
// 8*ch mod 32 takes four values). key = 2*warp + (lane&1), ch = (lane&31)>>1 interleaves two keys
// so the writes cover all 32 banks, while global reads stay two contiguous 256-byte runs.
#define O_KWIDE 524288
// GPREV: fetch tile ti+1's V into registers right after the second barrier, so its latency is
// hidden by tile ti's 96 WMMAs and only the LDS stores remain between barriers. V only: at
// 240 VGPRs a K prefetch would be spilled, and a spilled prefetch is waited on immediately.
#define O_GPREV 1048576
// LDSB: an LDS-scoped barrier instead of __syncthreads() (lds_barrier() in halo_common.h). The
// default fence invalidates the vector cache twice per tile, though only LDS needs ordering.
#define O_LDSB  2097152
// KLATE: on the fp8 and k6_v4 caches (the wide K fetch), tile ti+1's K is fetched once tile ti's
// last m-tile has its scores (its K fragments are dead there, so kreg adds no live range across the
// WMMAs), and the staging between the barriers only widens and stores. Otherwise every wave of the
// workgroup (the only one on its WGP) waits out a global round trip each tile.
#define O_KLATE 4194304
// O_V4: the fp8_v4 cache, O_V4 | O_K6 the k6_v4 cache (they share the int4 V layout; K6 adds the
// int6 K one).
#define O_V4    33554432
#define O_K6    67108864

// One f32 through an e4m3 round trip: the value an e4m3 code holds after pk_fp8x4's rounding.
__device__ __forceinline__ float halo_e4m3_rt(float x) {
    return halo_fp8::e4m3(halo_fp8::f32_to_e4m3(x));
}

// Four f16 dwords (eight values) through the same round trip in place -- the lossy legs' rounding
// of an operand the staging has already widened into f16.
template<class D>
__device__ __forceinline__ void halo_e4m3_rt4(uint32_t w4[4]) {
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const h2 v = __builtin_bit_cast(h2, w4[q]);
        w4[q] = D::pk(halo_e4m3_rt((float)v[0]), halo_e4m3_rt((float)v[1]));
    }
}

// Eight e4m3 bytes to four 16-bit dwords in element order: fp8x8_to_16x4w's result, cheaper in
// the f16 format. A byte's magnitude bits moved to f16 bits 14..7 read as 2^-8 times the e4m3 value
// (exponent bias 15 against 7; e4m3 subnormals land on f16 subnormals of the same mantissa), the
// sign goes to bit 15, and one packed multiply by 256 scales both values exactly. About 2.5 VALU a
// value against ~12 for the f32 route, and bit-identical to it on all 256 codes (checked on the
// part), the NaN codes included (+-480, as halo_fp8::e4m3 decodes them).
template<class D, int F16>
__device__ __forceinline__ void halo_fp8x8_w4(uint2 raw, uint32_t w4[4]) {
    if constexpr (!F16) {
        fp8x8_to_16x4w<D>(raw, w4);
    } else {
        typedef _Float16 h2 __attribute__((ext_vector_type(2)));
        const h2 k256 = { (_Float16)256.0f, (_Float16)256.0f };
        #pragma unroll
        for (int q = 0; q < 2; ++q) {
            const uint32_t w = q ? raw.y : raw.x;
            const uint32_t m = w & 0x7F7F7F7Fu, sg = w & 0x80808080u;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                // Bytes 2j and 2j+1: magnitudes to bits 0-7 and 16-23, signs to bytes 1 and 3
                // (perm selector 0x0c is a zero byte).
                const uint32_t lo = 0x0c000c00u | (uint32_t)(2 * j) | ((uint32_t)(2 * j + 1) << 16);
                const uint32_t hi = 0x000c000cu | ((uint32_t)(2 * j) << 8) | ((uint32_t)(2 * j + 1) << 24);
                const uint32_t n = (__builtin_amdgcn_perm(0u, m, lo) << 7) |
                                   __builtin_amdgcn_perm(0u, sg, hi);
                w4[2 * q + j] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2, n) * k256);
            }
        }
    }
}

// A fragment with its two half-waves exchanged: lane L takes lane L ^ 16's eight dwords.
template<class D>
__device__ __forceinline__ typename D::frag halo_swap16_frag(typename D::frag f) {
    union { typename D::frag f; uint32_t u[8]; } x;
    x.f = f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        // v_permlanex16 with identity selectors, as swap16. Its `old` operand is never read (every
        // lane is active and has a partner). Left undefined, the destination needs no prior copy of
        // the source, which passing the source as `old` costs (one v_mov a dword); a defined value
        // (even __builtin_nondeterministic_value) is materialised the same way.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
        int old;
        x.u[i] = (uint32_t)__builtin_amdgcn_permlanex16(old, (int)x.u[i], 0x76543210u,
                                                        0xFEDCBA98u, false, false);
#pragma clang diagnostic pop
    }
    return x.f;
}

// bf16 dword -> 16-bit dword in the target format, optionally scaled. Used once per kernel on Q.
// E8 rounds each value once to e4m3 first (the QK8 leg's query operand).
template<class D, int FOLDQ, int F16, int E8 = 0>
__device__ __forceinline__ uint32_t halo_qcvt(uint32_t w, float s) {
    const float lo = __builtin_bit_cast(float, w << 16);
    const float hi = __builtin_bit_cast(float, w & 0xffff0000u);
    if (E8) return D::pk(halo_e4m3_rt(FOLDQ ? lo * s : lo), halo_e4m3_rt(FOLDQ ? hi * s : hi));
    if (!FOLDQ) return F16 ? D::from_bf16w(w) : w;
    return D::pk(lo * s, hi * s);
}

// ---- the split-KV law (the launch side is at the end of this file) ----------------------------
#define HALO_P_SLOTS     20         // workgroups resident at once: one 24-wave workgroup a WGP
#define HALO_P_UNITS_MAX 160        // grids this large lose < 1/8 to the last round: never split
#define HALO_P_MIN_TILES 4          // tiles a split walks at least
#define HALO_P_MAX_SPLIT 64
#define HALO_P_OVH_TILES 4          // a split's fixed cost (query load, partial, merge), in tiles
#define HALO_P_SCRATCH_MAX (64LL << 20)

// Bytes of one split's partials: f32 O plus (m, l) per (token, q head).
static inline long long halo_prefill_split_bytes(long long q_total, long long q_heads, int head_dim) {
    return q_total * q_heads * (long long)(head_dim + 2) * 4LL;
}

// The split law: the count that minimises (rounds of HALO_P_SLOTS workgroups) x (tiles a
// workgroup walks + its fixed cost), taking a larger count only for a 3 % gain, and keeping the
// partials under HALO_P_SCRATCH_MAX.
// The kernel reruns it per workgroup on the tile count it really walks (the host knows only
// max_ctx), so a short context does not pay for partials it does not need.
__host__ __device__ inline int halo_prefill_splits(int units, int tiles, long long per_split) {
    if (units <= 0 || units >= HALO_P_UNITS_MAX || tiles < 2 * HALO_P_MIN_TILES) return 1;
    int smax = tiles / HALO_P_MIN_TILES;
    if (smax > HALO_P_MAX_SPLIT) smax = HALO_P_MAX_SPLIT;
    if (per_split > 0 && HALO_P_SCRATCH_MAX / per_split < smax) smax = (int)(HALO_P_SCRATCH_MAX / per_split);
    int best = 1;
    float best_cost = (float)((units + HALO_P_SLOTS - 1) / HALO_P_SLOTS) *
                      (float)(tiles + HALO_P_OVH_TILES);
    for (int s = 2; s <= smax; ++s) {
        const float rounds = (float)((units * s + HALO_P_SLOTS - 1) / HALO_P_SLOTS);
        const float cost = rounds * ((float)tiles / (float)s + (float)HALO_P_OVH_TILES);
        if (cost < 0.97f * best_cost) { best_cost = cost; best = s; }
    }
    return best;
}

// WIN enables a sliding window (a.window positions ending at each row's causal limit), as in the
// decode kernel. Keys below the window are masked, and the KV walk starts at the tile holding the
// workgroup's earliest window token. window <= 0 (or WIN off) is full attention.
template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT, int WIN = 0>
__global__ __launch_bounds__(NWARPS * 32)
void halo_attn_prefill_kernel(const HaloArgs a)
{
    constexpr int F16   = (OPT & O_F16)   ? 1 : 0;
    constexpr int MSKIP = (OPT & O_MSKIP) ? 1 : 0;
    constexpr int FOLDQ = (OPT & O_FOLDQ) ? 1 : 0;
    constexpr int CTG   = (OPT & O_CONTIG) ? 1 : 0;
    constexpr int PF    = 1 << ((OPT >> 10) & 3);
    constexpr int SGB   = (OPT & O_SGB) ? 1 : 0;
    constexpr int QK8   = ((OPT & O_QK8) && !KVP) ? 1 : 0;
    constexpr int PV8   = ((OPT & O_PV8) && !KVP) ? 1 : 0;
    static_assert(!(OPT & (O_QK8 | O_PV8)) || !KVP,
                  "the e4m3 legs need an 8-bit cache, or they compile to the exact form");
    static_assert(!(QK8 || PV8) || F16, "the e4m3 legs emulate on the f16 operand format");
    // PV8 keeps the denominator on the f32 p values the softmax formed (libr4d's choice): with
    // DOT2 on it would sum the rounded e4m3 ones.
    constexpr int DOT2  = ((OPT & O_DOT2) && (OPT & O_F16) && !PV8) ? 1 : 0;
    constexpr int BTS   = (OPT & O_BTS)   ? 1 : 0;
    constexpr int V4    = ((OPT & O_V4) && !KVP) ? 1 : 0;
    constexpr int K6    = ((OPT & O_K6) && V4) ? 1 : 0;
    // A token's row in a page, and where in it V's codes, V's scales and K's scales start.
    constexpr int ROWB  = K6 ? HEAD_DIM * 22 / 16 : V4 ? HEAD_DIM * 25 / 16 : 2 * HEAD_DIM;
    constexpr int VOFF  = K6 ? HEAD_DIM * 3 / 4 : HEAD_DIM;
    constexpr int VSOFF = K6 ? HEAD_DIM * 21 / 16 : HEAD_DIM * 3 / 2;
    constexpr int KSOFF = HEAD_DIM * 5 / 4;
    constexpr int GPREV = (OPT & O_GPREV) ? 1 : 0;
    constexpr int LDSB  = (OPT & O_LDSB)  ? 1 : 0;
    constexpr int KWIDE = (OPT & O_KWIDE) ? 1 : 0;
    // fp8_v4 takes none of the three (KLATE, the assume, the scale pack): they gain at 100k but lose
    // 3-7 % at 25k and 50k.
    constexpr int OLDV4 = V4 && !K6;
    constexpr int KLATE = ((OPT & O_KLATE) && KWIDE && !KVP && !OLDV4) ? 1 : 0;
    typedef DT16<F16> D;
    typedef typename D::frag frag16;

    // QX (head_dim 256): the query is held once, not twice. A B operand is replicated in both
    // half-waves, but each half only multiplies its own copy (header), so register i carries
    // fragment 2i in the low half and 2i+1 in the high one: one WMMA contracts each half against
    // its own fragment, a second against the half-swapped copy (eight v_permlanex16), with K read
    // at the matching dims per half. Q drops from 128 to 64 VGPRs; beside the 128-VGPR accumulator
    // the full query left ~90 spill slots, re-read from scratch every m-tile. The Q.K^T loop takes
    // its K fragments a pair at a time here; PF still sets the P.V loop's.
    constexpr int QX      = (HEAD_DIM >= 256) ? 1 : 0;
    constexpr int NQF     = QX ? HEAD_DIM / 32 : HEAD_DIM / 16;   // query registers

    constexpr int ROWS    = NWARPS * 16;
    constexpr int BLOCK_Q = ROWS / GQA;
    constexpr int NKS     = HEAD_DIM / 16;
    constexpr int MT      = TILE / 16;
    // Row strides padded by 8 shorts: 32-byte-aligned enough for the 16-byte gather pairs, and off
    // the bank period.
    constexpr int KSTR    = HEAD_DIM + 8;
    constexpr int VSTR    = TILE + 8;
    constexpr int NTHR    = NWARPS * 32;
    constexpr int NB      = TILE / BS;                       // block-table entries per tile
    constexpr int KEL     = KWIDE ? 16 : 8;                  // K elements fetched per thread-pass
    constexpr int KITEMS  = TILE * (HEAD_DIM / KEL);          // K passes needed for one tile
    constexpr int KPASS   = (KITEMS + NTHR - 1) / NTHR;        // passes per thread
    constexpr int KREGS   = KPASS * (KVP ? 2 : 1);             // uint4 of raw K per thread
    constexpr int NTR     = (TILE / 8) * (HEAD_DIM / (KVP ? 32 : 64));
    constexpr int VREGS   = (NTR + NWARPS - 1) / NWARPS;        // uint4 of raw V per thread
    constexpr float PSHIFT = D::SHIFT;
    constexpr float PGROW  = PV8 ? 8.0f : D::GROW;   // e4m3 max is 448 = 2^8.8
    static_assert(!KWIDE || KPASS == 1, "the wide K fetch is one straight-line pass");
    static_assert(!K6 || KWIDE, "k6_v4 K is staged sixteen dims a thread");

    __shared__ __align__(16) uint16_t sK[TILE * KSTR];
    __shared__ __align__(16) uint16_t sV16[HEAD_DIM * VSTR];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // The launch is NTHR threads. Stated, so the wide K fetch's `key >= TILE` test (never true:
    // KITEMS == NTHR) folds: as a branch its join made the compiler wait on every load in flight right
    // after the K fetch, the V stores' cover included.
    if constexpr (!OLDV4) __builtin_assume(tid < NTHR);
    const int c = lane & 15, h = lane >> 4;
    // The A-operand row this lane reads: its lane pair's target row (header), so both lanes of the
    // pair read one address.
    const int arow = (c & ~1) | h;

    // Split-KV (a.splits > 1, partials in a.scratch): grid.z is sequence-major over the splits, and
    // split sp walks its share of the workgroup's tile range, then writes its unnormalised O and its
    // (m, l) for halo_attn_prefill_merge to combine. One split is the plain kernel.
    const int nsplit = (a.splits > 1 && a.scratch) ? a.splits : 1;
    const int qb = blockIdx.x, kvh = blockIdx.y;
    const int seq = (int)blockIdx.z / nsplit, sp = (int)blockIdx.z - seq * nsplit;
    const int ctx = a.seqused_k[seq];

    // Ragged batches: this sequence owns rows [qoff, qoff+qlen) of q and out. grid.x is sized from
    // the batch maximum q_len, so blocks past a shorter sequence's end exit here.
    const int qoff = halo_q_off(a.cu_q, seq, a.q_len);
    const int qlen = halo_q_len(a.cu_q, seq, a.q_len);
    if (qb * BLOCK_Q >= qlen) return;

    const int row = warp * 16 + c;
    const int qi = row / GQA, hi = row - qi * GQA;
    const int qpos = qb * BLOCK_Q + qi;
    const bool live = (qpos < qlen);
    const int qrow = live ? qpos : (qlen - 1);
    const int qhead = kvh * GQA + hi;
    const size_t qtot = (size_t)halo_q_total(a);
    // Where this split's partial row lives: O as f32 [split][token][q_head][HEAD_DIM], then (m, l)
    // as f32 pairs [split][token][q_head].
    float* const pO = (float*)a.scratch
                    + (((size_t)sp * qtot + qoff + qpos) * a.q_heads + qhead) * HEAD_DIM;
    float* const pML = (float*)a.scratch + (size_t)nsplit * qtot * a.q_heads * HEAD_DIM
                     + (((size_t)sp * qtot + qoff + qpos) * a.q_heads + qhead) * 2;
    if (ctx <= 0) {
        // The plain kernel leaves these rows untouched; a negative l tells the merge to as well.
        if (nsplit > 1 && sp == 0 && live && h == 0) pML[1] = -1.0f;
        return;
    }
    // Warps whose sixteen rows are all past the sequence's queries only stage tiles (their rows
    // are never stored): at a short q_len most of the workgroup's WMMAs were for nothing.
    const bool wdead = __builtin_amdgcn_readfirstlane(qb * BLOCK_Q + (warp * 16) / GQA) >= qlen;

    const int* __restrict__ bt = a.block_table + (size_t)seq * a.max_blocks;

    const float kdesc = a.k_descale ? a.k_descale[seq * a.kv_heads + kvh] : 1.0f;
    const float vdesc = a.v_descale ? a.v_descale[seq * a.kv_heads + kvh] : 1.0f;
    const float mul   = a.scale * kdesc * 1.44269504089f;

    // gfx1151: a lane holds the whole sixteen-value k run of its query row, so the fragment is one
    // 32-byte read with no lane-half term (the O_CONTIG and split gathers are the same read here).
    frag16 qf[NQF];
    {
        const uint16_t* qp = (const uint16_t*)a.q
                           + ((size_t)(qoff + qrow) * a.q_heads + qhead) * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < NQF; ++i) {
            const int t = QX ? 2 * i + h : i;       // QX: fragment 2i + h in this half
            const uint4 u0 = *(const uint4*)(qp + 16 * t);
            const uint4 u1 = *(const uint4*)(qp + 16 * t + 8);
            const uint32_t w8[8] = { halo_qcvt<D, FOLDQ, F16, QK8>(u0.x, mul),
                                     halo_qcvt<D, FOLDQ, F16, QK8>(u0.y, mul),
                                     halo_qcvt<D, FOLDQ, F16, QK8>(u0.z, mul),
                                     halo_qcvt<D, FOLDQ, F16, QK8>(u0.w, mul),
                                     halo_qcvt<D, FOLDQ, F16, QK8>(u1.x, mul),
                                     halo_qcvt<D, FOLDQ, F16, QK8>(u1.y, mul),
                                     halo_qcvt<D, FOLDQ, F16, QK8>(u1.z, mul),
                                     halo_qcvt<D, FOLDQ, F16, QK8>(u1.w, mul) };
            qf[i] = D::mk(w8);
        }
    }

    const int klimit = ctx - qlen + qpos;
    // First key in this row's window, floored at zero. Same formula as the decode unit's window
    // floor.
    int wlo = (WIN && a.window > 0) ? (ctx - qlen + qpos - a.window + 1) : 0;
    if (WIN && a.window > 0 && wlo < 0) wlo = 0;

    float m_ref = -3.0e38f;
    float off   = PSHIFT + 3.0e38f;
    float l_i   = 0.0f;
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0,0,0,0,0,0,0,0};

    const int qmax_cta   = min(qb * BLOCK_Q + BLOCK_Q - 1, qlen - 1);
    const int klimit_cta = ctx - qlen + qmax_cta;
    const int ntiles_cta = (min(klimit_cta + 1, ctx) + TILE - 1) / TILE;
    // First window tile, taken at this workgroup's first query row (the one reaching furthest
    // back), not the chunk's row 0, so fully masked tiles are never walked. Starts at 0 without a
    // window.
    const int wlo0 = (WIN && a.window > 0) ? (ctx - qlen + qb * BLOCK_Q - a.window + 1) : 0;
    const int tstart_cta = (WIN && a.window > 0 && wlo0 > 0) ? (wlo0 / TILE) : 0;
    // The host sized the split from max_ctx, a bound; the walk this workgroup really has decides
    // how many splits it uses: ne, the split law at its real tile count. Splits past ne only
    // say they are empty (m at its floor, l = 0, which the merge skips without reading O); with
    // ne == 1 split 0 stores the output itself and tells the merge to leave the row (l < 0).
    // Split sp walks its contiguous share of [tstart_cta, ntiles_cta).
    const int nt_cta = max(ntiles_cta - tstart_cta, 0);
    const int ne = (nsplit > 1)
        ? min(nsplit, halo_prefill_splits((int)(gridDim.x * gridDim.y) * a.num_seqs, nt_cta, 0)) : 1;
    if (sp >= ne) {
        if (ne > 1 && live && h == 0) { pML[0] = -3.0e38f; pML[1] = 0.0f; }
        return;
    }
    const int tstart = tstart_cta + (int)(((long)nt_cta * sp) / ne);
    const int ntiles = tstart_cta + (int)(((long)nt_cta * (sp + 1)) / ne);

    // ---- staging, split into a global fetch (registers) and an LDS store ------------------------
    // The split lets GPREV move the fetch out from between the barriers.
    const int vg = lane >> 3, vj = lane & 7;
    uint4 kreg[KREGS], vreg[VREGS];

    // ---- fp8_v4 / k6_v4: V is int4 in groups of 32 with an f16 scale (absmax / 7): 16 bytes of
    // codes a group at VOFF (dim d at bits 4*(d%8) of word d/8), the group scales at VSOFF. A thread
    // stages one key pair's 8 dims: two code words and the pair's scale, fetched a tile ahead like
    // GPREV's V, so the loads are in flight across the m-tile loop. They widen into sV16 ([dim][key])
    // as a dword per dim holding both keys: with u = code ^ 8 (offset binary), 0x6400 | u is the f16
    // 1024 + u, so (x - 1032) * s is the value times s, rounded once. Key pairs run fastest across
    // lanes, which keeps the dword stores of one dim row in distinct banks.
    static_assert(!V4 || F16, "fp8_v4 widens V to f16");
    constexpr int NP4 = (TILE / 2) * (HEAD_DIM / 8);
    constexpr int U4  = V4 ? (NP4 + NTHR - 1) / NTHR : 1;
    // The pair's two scales stay apart until storeV4 packs them: packed here, the prefetch a tile
    // ahead would be waited on right after it is issued (fp8_v4 packs here, see KLATE).
    uint32_t v4c[U4][2], v4s[U4][2];
    auto fetchV4 = [&](int k0, const int* blks) {
        #pragma unroll
        for (int r = 0; r < U4; ++r) {
            const int u = tid + r * NTHR;
            if (NP4 % NTHR == 0 || u < NP4) {
                const int kp = u % (TILE / 2), o = u / (TILE / 2);
                uint32_t sc[2];
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int kk = min(k0 + 2 * kp + j, ctx - 1);
                    const int bk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                    const uint8_t* base = (const uint8_t*)a.kv + (size_t)bk * a.kv_block_stride
                                        + (size_t)kvh * a.kv_head_stride + (size_t)(kk % BS) * ROWB;
                    v4c[r][j] = *(const uint32_t*)(base + VOFF + o * 4);
                    sc[j]     = *(const uint16_t*)(base + VSOFF + (o >> 2) * 2);
                }
                if constexpr (OLDV4) { v4s[r][0] = sc[0] | (sc[1] << 16); v4s[r][1] = 0; }
                else                 { v4s[r][0] = sc[0]; v4s[r][1] = sc[1]; }
            }
        }
    };
    auto storeV4 = [&]() {
        typedef _Float16 h2 __attribute__((ext_vector_type(2)));
        #pragma unroll
        for (int r = 0; r < U4; ++r) {
            const int u = tid + r * NTHR;
            if (NP4 % NTHR == 0 || u < NP4) {
                const int kp = u % (TILE / 2), o = u / (TILE / 2);
                // Codes are two's complement nibbles; ^ 8 makes them offset binary (value + 8).
                const uint32_t n0 = (v4c[r][0] & 0x0F0F0F0Fu) ^ 0x08080808u;
                const uint32_t m0 = ((v4c[r][0] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
                const uint32_t n1 = (v4c[r][1] & 0x0F0F0F0Fu) ^ 0x08080808u;
                const uint32_t m1 = ((v4c[r][1] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
                const h2 s2 = __builtin_bit_cast(h2, OLDV4 ? v4s[r][0] : (v4s[r][0] | (v4s[r][1] << 16)));
                const h2 bias = { (_Float16)1032.0f, (_Float16)1032.0f };
                uint16_t* dst = &sV16[(o * 8) * VSTR + 2 * kp];
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    // Byte b of the first key's word low, of the second key's word in the high half:
                    // perm selects source bytes 0-3 from its second operand, 4-7 from its first,
                    // and 0x0c is a zero byte.
                    const uint32_t sel = (uint32_t)b | 0x0c00u | ((uint32_t)(4 + b) << 16) | 0x0c000000u;
                    const uint32_t lo = __builtin_amdgcn_perm(n1, n0, sel) | 0x64006400u;   // dim 2b
                    const uint32_t hi = __builtin_amdgcn_perm(m1, m0, sel) | 0x64006400u;   // dim 2b+1
                    const h2 vl = (__builtin_bit_cast(h2, lo) - bias) * s2;
                    const h2 vh = (__builtin_bit_cast(h2, hi) - bias) * s2;
                    if constexpr (PV8) {
                        // V as e4m3 bytes on gfx12: the widened value, rounded once
                        *(uint32_t*)(dst + (2 * b) * VSTR) =
                            D::pk(halo_e4m3_rt((float)vl[0]), halo_e4m3_rt((float)vl[1]));
                        *(uint32_t*)(dst + (2 * b + 1) * VSTR) =
                            D::pk(halo_e4m3_rt((float)vh[0]), halo_e4m3_rt((float)vh[1]));
                    } else {
                        *(uint32_t*)(dst + (2 * b) * VSTR)     = __builtin_bit_cast(uint32_t, vl);
                        *(uint32_t*)(dst + (2 * b + 1) * VSTR) = __builtin_bit_cast(uint32_t, vh);
                    }
                }
            }
        }
    };

    auto fetchK = [&](int k0, const int* blks) {
        #pragma unroll
        for (int p = 0; p < KPASS; ++p) {
            const int i = tid + p * NTHR;
            if (KITEMS % NTHR == 0 || i < KITEMS) {
                const int key = KWIDE ? (2 * (i >> 5) + (i & 1)) : (i / (HEAD_DIM / KEL));
                const int ch  = KWIDE ? ((i & 31) >> 1) : (i - key * (HEAD_DIM / KEL));
                if (KWIDE && key >= TILE) continue;
                const int kk  = min(k0 + key, ctx - 1);
                const int blk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)blk * a.kv_block_stride
                               + (size_t)kvh * a.kv_head_stride + (size_t)(kk % BS) * ROWB;
                if (KVP) {
                    const uint16_t* g = (const uint16_t*)a.kv + o + ch * KEL;
                    #pragma unroll
                    for (int q = 0; q < (KVP ? (KEL / 8) : 1); ++q)
                        kreg[p * (KEL / 8) + q] = *(const uint4*)(g + 8 * q);
                } else if (K6) {
                    // k6_v4: the chunk's sixteen low nibbles, its high-bit word, its group's scale.
                    const uint8_t* rp = (const uint8_t*)a.kv + o;
                    const uint2 lo = *(const uint2*)(rp + ch * 8);
                    kreg[p] = make_uint4(lo.x, lo.y, *(const uint32_t*)(rp + HEAD_DIM / 2 + ch * 4),
                                         *(const uint16_t*)(rp + KSOFF + (ch >> 1) * 2));
                } else if (KWIDE) {
                    kreg[p] = *(const uint4*)((const uint8_t*)a.kv + o + ch * 16);
                } else {
                    const uint2 r = *(const uint2*)((const uint8_t*)a.kv + o + ch * 8);
                    kreg[p] = make_uint4(r.x, r.y, 0, 0);
                }
            }
        }
    };

    auto fetchV = [&](int k0, const int* blks) {
        #pragma unroll
        for (int r = 0; r < VREGS; ++r) {
            const int idx = warp + r * NWARPS;
            if (NTR % NWARPS == 0 || idx < NTR) {
                constexpr int DG = HEAD_DIM / (KVP ? 32 : 64);
                const int kg = idx / DG, dg = idx - kg * DG;
                const int kk = min(k0 + kg * 8 + vj, ctx - 1);
                const int d0 = dg * (KVP ? 32 : 64) + vg * (KVP ? 8 : 16);
                const int blk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)blk * a.kv_block_stride
                               + (size_t)kvh * a.kv_head_stride
                               + (size_t)(kk % BS) * ROWB + HEAD_DIM + d0;
                const void* p = KVP ? (const void*)((const uint16_t*)a.kv + o)
                                    : (const void*)((const uint8_t*)a.kv + o);
                vreg[r] = __builtin_bit_cast(uint4, load_tr_b128(p));
            }
        }
    };

    auto storeK = [&]() {
        #pragma unroll
        for (int p = 0; p < KPASS; ++p) {
            const int i = tid + p * NTHR;
            if (KITEMS % NTHR == 0 || i < KITEMS) {
                const int key = KWIDE ? (2 * (i >> 5) + (i & 1)) : (i / (HEAD_DIM / KEL));
                const int ch  = KWIDE ? ((i & 31) >> 1) : (i - key * (HEAD_DIM / KEL));
                if (KWIDE && key >= TILE) continue;
                uint16_t* kd = &sK[key * KSTR + ch * KEL];
                #pragma unroll
                for (int q = 0; q < KEL / 8; ++q) {
                    uint32_t w4[4];
                    if (KVP) {
                        const uint4 r = kreg[p * (KEL / 8) + q];
                        w4[0] = D::from_bf16w(r.x); w4[1] = D::from_bf16w(r.y);
                        w4[2] = D::from_bf16w(r.z); w4[3] = D::from_bf16w(r.w);
                    } else {
                        const uint4 r = kreg[p];
                        if constexpr (K6) {
                            k6x8_to_f16x4w(q ? r.y : r.x, r.z, q, r.w | (r.w << 16), w4);
                            // QK8 over k6_v4: the byte staging rounds the widened int6 to e4m3
                            if constexpr (QK8) halo_e4m3_rt4<D>(w4);
                        } else halo_fp8x8_w4<D, F16>(q ? make_uint2(r.z, r.w) : make_uint2(r.x, r.y), w4);
                    }
                    *(uint2*)(kd + 8 * q)     = make_uint2(w4[0], w4[1]);
                    *(uint2*)(kd + 8 * q + 4) = make_uint2(w4[2], w4[3]);
                }
            }
        }
    };

    auto storeV = [&]() {
        #pragma unroll
        for (int r = 0; r < VREGS; ++r) {
            const int idx = warp + r * NWARPS;
            if (NTR % NWARPS == 0 || idx < NTR) {
                constexpr int DG = HEAD_DIM / (KVP ? 32 : 64);
                const int kg = idx / DG, dg = idx - kg * DG;
                const int d0 = dg * (KVP ? 32 : 64) + vg * (KVP ? 8 : 16);
                const uint4 v = vreg[r];
                if (KVP) {
                    uint16_t* dst = &sV16[(d0 + vj) * VSTR + kg * 8];
                    *(uint2*)(dst)     = make_uint2(D::from_bf16w(v.x), D::from_bf16w(v.y));
                    *(uint2*)(dst + 4) = make_uint2(D::from_bf16w(v.z), D::from_bf16w(v.w));
                } else {
                    // tr_b128 transposes 16-bit lanes, so each short holds two fp8 from adjacent
                    // dims; v_perm_b32 splits them into two dim rows.
                    const uint32_t lo0 = byte_gather(v.x, v.y, 0), lo1 = byte_gather(v.z, v.w, 0);
                    const uint32_t hi0 = byte_gather(v.x, v.y, 1), hi1 = byte_gather(v.z, v.w, 1);
                    uint32_t wl[4], wh[4];
                    halo_fp8x8_w4<D, F16>(make_uint2(lo0, lo1), wl);
                    halo_fp8x8_w4<D, F16>(make_uint2(hi0, hi1), wh);
                    uint16_t* a0 = &sV16[(d0 + 2 * vj)     * VSTR + kg * 8];
                    uint16_t* a1 = &sV16[(d0 + 2 * vj + 1) * VSTR + kg * 8];
                    *(uint2*)(a0)     = make_uint2(wl[0], wl[1]);
                    *(uint2*)(a0 + 4) = make_uint2(wl[2], wl[3]);
                    *(uint2*)(a1)     = make_uint2(wh[0], wh[1]);
                    *(uint2*)(a1 + 4) = make_uint2(wh[2], wh[3]);
                }
            }
        }
    };

    // Tile ti's block-table entries are loaded during tile ti-1, off the KV address chain.
    // A multi-block tile can run past the sequence's last block (on the last row, past the end of
    // the table). Keys are clamped to ctx - 1, so entries past lastb are never used; only tiles
    // reaching past lastb take the clamped load. With NB == 1 the test folds away.
    const int lastb = (ctx - 1) / BS;
    auto load_bt = [&](int t0, int* out) {
        if (NB > 1 && t0 + NB - 1 > lastb) {
            #pragma unroll
            for (int i = 0; i < NB; ++i) out[i] = bt[min(t0 + i, lastb)];
        } else {
            #pragma unroll
            for (int i = 0; i < NB; ++i) out[i] = bt[t0 + i];
        }
    };
    int blk[NB];
    // An empty split's tstart can lie past the context, where the clamped key would index blk
    // below its start: it fetches nothing.
    if (tstart < ntiles) {
        if (BTS) load_bt(tstart * NB, blk);
        if (GPREV) fetchV(tstart * TILE, blk);
        if (V4) fetchV4(tstart * TILE, blk);
        if (KLATE) fetchK(tstart * TILE, blk);
    }

    for (int ti = tstart; ti < ntiles; ++ti) {
        const int k0 = ti * TILE;
        int blkn[NB];
        if (BTS && ti + 1 < ntiles) {
            load_bt((ti + 1) * NB, blkn);
        }
        if (LDSB) lds_barrier(); else __syncthreads();
        if (!KLATE) fetchK(k0, blk);
        // V first: already in registers.
        if constexpr (V4) { storeV4(); } else { if (!GPREV) fetchV(k0, blk); storeV(); }
        storeK();
        if (LDSB) lds_barrier(); else __syncthreads();
        // Issue the next tile's V loads before the WMMAs below; they are consumed after the next
        // barrier.
        if (GPREV && ti + 1 < ntiles) fetchV(k0 + TILE, blkn);
        if (V4 && ti + 1 < ntiles) fetchV4(k0 + TILE, blkn);

        if (wdead && KLATE && ti + 1 < ntiles) fetchK(k0 + TILE, blkn);
        #pragma unroll 1
        for (int mt = 0; mt < (wdead ? 0 : MT); ++mt) {
            v8f s = (v8f){0,0,0,0,0,0,0,0};
            {
                // gfx1151: the A operand's row is the key and its element e is k = e, so this lane
                // reads the whole sixteen-dim run of key row arow -- no lane-half term.
                const uint16_t* krow = &sK[(mt * 16 + arow) * KSTR];
                if constexpr (QX) {
                    // Register i is fragment 2i + h here and 2i + 1 - h in its swapped copy, and
                    // each half reads its keys at the same dims, so the pair of WMMAs contracts all
                    // 32 dims in both halves (the low half in order 2i, 2i+1, the high 2i+1, 2i).
                    const uint16_t* kp0 = krow + 16 * h;
                    const uint16_t* kp1 = krow + 16 * (1 - h);
                    #pragma unroll
                    for (int i = 0; i < NQF; ++i) {
                        frag16 kb[2];
                        #pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            const uint16_t* kp = (j ? kp1 : kp0) + 32 * i;
                            const uint4 u0 = *(const uint4*)kp;
                            const uint4 u1 = *((const uint4*)kp + 1);
                            const uint32_t w8[8] = { u0.x, u0.y, u0.z, u0.w,
                                                     u1.x, u1.y, u1.z, u1.w };
                            kb[j] = D::mk(w8);
                        }
                        const frag16 qs = halo_swap16_frag<D>(qf[i]);
                        if (SGB) sched_barrier();
                        s = D::wmma(kb[0], qf[i], s);
                        s = D::wmma(kb[1], qs, s);
                    }
                } else
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    frag16 kb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        const uint16_t* kp = krow + 16 * (g + j);
                        const uint4 u0 = *(const uint4*)kp;
                        const uint4 u1 = *((const uint4*)kp + 1);
                        const uint32_t w8[8] = { u0.x, u0.y, u0.z, u0.w,
                                                 u1.x, u1.y, u1.z, u1.w };
                        kb[j] = D::mk(w8);
                    }
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) s = D::wmma(kb[j], qf[g + j], s);
                }
            }
            if (KLATE && mt == MT - 1 && ti + 1 < ntiles) fetchK(k0 + TILE, blkn);
            float sc[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) sc[e] = FOLDQ ? s[e] : s[e] * mul;

            // gfx1151: element e of the score fragment is key 2e + (lane >> 4) of the tile.
            const int kbase = k0 + mt * 16;
            // Mask keys past klimit and, with WIN, before wlo. MSKIP skips tiles straddling neither.
            if (!MSKIP || wave_any(kbase + 15 > klimit) || (WIN && wave_any(kbase < wlo))) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int key = kbase + 2 * e + h;
                    sc[e] = ((key <= klimit) && (!WIN || key >= wlo)) ? sc[e] : -INFINITY;
                }
            }

            float smax = sc[0];
            #pragma unroll
            for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);
            if (wave_any(smax > m_ref + PGROW)) {
                const float rmax  = fmaxf(smax, swap16(smax));
                const float mnew  = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
                const float alpha = ex2(m_ref - mnew);
                #pragma unroll
                for (int dt = 0; dt < NKS; ++dt)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) acc[dt][e] *= alpha;
                l_i *= alpha;
                m_ref = mnew;
                off   = PSHIFT - mnew;
            }

            float pf[8], lsum = 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float p = ex2(sc[e] + off);
                pf[e] = PV8 ? halo_e4m3_rt(p) : p;
                if (!DOT2) lsum += p;
            }
            if (!DOT2) l_i += lsum;

            {
                uint32_t pw[4];
                #pragma unroll
                for (int e = 0; e < 8; e += 2) pw[e >> 1] = D::pk(pf[e], pf[e + 1]);
                if (DOT2) {
                    typedef __fp16 v2hf_attn __attribute__((ext_vector_type(2)));
                    const v2hf_attn one = {(__fp16)1.0f, (__fp16)1.0f};
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                        l_i = __builtin_amdgcn_fdot2(__builtin_bit_cast(v2hf_attn, pw[e]), one, l_i, false);
                }
                // The P fragment's element e is key e, but this lane's eight values are keys
                // 2e + (lane >> 4): interleave the half-pair's copies into key order.
                uint32_t pw16[8];
                p16w<D>(pw, pw16);
                const frag16 p16 = D::mk(pw16);
                // V^T is the A operand: its row is the head dim and element e is k = e, so each lane
                // reads the whole sixteen-key run of dim row arow.
                const uint16_t* vbase = &sV16[arow * VSTR + mt * 16];
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    frag16 vb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        const uint16_t* vp = vbase + (g + j) * 16 * VSTR;
                        const uint4 u0 = *(const uint4*)vp;
                        const uint4 u1 = *((const uint4*)vp + 1);
                        const uint32_t w8[8] = { u0.x, u0.y, u0.z, u0.w,
                                                 u1.x, u1.y, u1.z, u1.w };
                        vb[j] = D::mk(w8);
                    }
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) acc[g + j] = D::wmma(vb[j], p16, acc[g + j]);
                }
            }
        }
        if (BTS) {
            #pragma unroll
            for (int i = 0; i < NB; ++i) blk[i] = blkn[i];
        }
    }

    if (!live) return;
    l_i += swap16(l_i);
    if (nsplit > 1 && ne == 1 && h == 0) pML[1] = -1.0f;     // stored below, merge skips it
    if (ne > 1) {
        // The partial: O times the V descale, unnormalised; m_ref is the reference both O and l
        // are exponentiated against (each carries the same 2^PSHIFT, which the merge cancels).
        #pragma unroll
        for (int dt = 0; dt < NKS; ++dt)
            #pragma unroll
            for (int e = 0; e < 8; ++e)
                pO[dt * 16 + 2 * e + h] = acc[dt][e] * vdesc;
        if (h == 0) { pML[0] = m_ref; pML[1] = l_i; }
        return;
    }
    const float inv = (l_i > 0.0f) ? (vdesc / l_i) : 0.0f;
    uint16_t* op = (uint16_t*)a.out
                 + ((size_t)(qoff + qpos) * a.q_heads + qhead) * HEAD_DIM;
    // gfx1151: accumulator element e is head dim 2*e + (lane >> 4), one value a lane, so the store
    // is a scatter of eight 16-bit values rather than one 16-byte write.
    #pragma unroll
    for (int dt = 0; dt < NKS; ++dt)
        #pragma unroll
        for (int e = 0; e < 8; ++e)
            op[dt * 16 + 2 * e + h] = f32_to_bf16(acc[dt][e] * inv);
}

// ================================================================== split-KV prefill
//
// The query-tiled grid is q tiles x KV heads x sequences. A short chunk over a long context -- a
// q_len of 11 at 16K tokens, four KV heads -- is four workgroups, each walking the whole context
// alone on one of the 20 WGPs (one 768-thread workgroup fills a WGP), so it ran at the speed of a
// q_len-64 chunk: 6.4 ms where reading the cache once is ~0.2. Splitting the tile range across
// workgroups (flash-decoding) fills the part; halo_attn_prefill_merge combines the partials. The
// partial buffer is the row's scratch, sized by its hook; without it (or with too little) the
// launch runs unsplit, the kernel as before.

// The count a launch of these args takes, before the scratch check.
template <int BQ, int TILE>
static inline int halo_prefill_splits_for(const HaloArgs& a) {
    const int qtiles = (a.q_len + BQ - 1) / BQ;
    const int units  = qtiles * a.kv_heads * a.num_seqs;
    const int tiles  = (a.max_ctx + TILE - 1) / TILE;
    return halo_prefill_splits(units, tiles,
                               halo_prefill_split_bytes(halo_q_total(a), a.q_heads, a.head_dim));
}

// Picks r->splits for a launch: the law's count when the caller's scratch holds its partials,
// else as many as fit, else 1 (unsplit, no scratch read). Call after fill_paged*.
template <int NWARPS, int TILE, int HEAD_DIM, int GQA>
static inline void halo_prefill_split_setup(const RadArgs* ra, HaloArgs* r) {
    constexpr int BQ = (NWARPS * 16) / GQA;
    r->splits = 1;
    const int s = halo_prefill_splits_for<BQ, TILE>(*r);
    if (s <= 1 || !ra->scratch || ra->scratch_bytes <= 0) return;
    const long long per = halo_prefill_split_bytes(halo_q_total(*r), r->q_heads, HEAD_DIM);
    const long long fit = per > 0 ? ra->scratch_bytes / per : 0;
    const int n = (int)std::min<long long>(s, fit);
    if (n > 1) { r->splits = n; r->scratch = ra->scratch; }
}

// The scratch hook: the largest partial buffer the law asks for over every q_len up to the band's
// bound and every sequence count up to max_seqs (parameters at declare time); from the tensors
// when the parameters are absent. 0 when no launch in the band splits.
template <int NWARPS, int TILE, int HEAD_DIM, int GQA>
static inline int64_t halo_prefill_scratch(const RadArgs* a, long long row = 0) {
    constexpr int BQ = (NWARPS * 16) / GQA;
    const long long n_head   = halo_geti_or(a, "n_head", 0);
    const long long max_seqs = halo_geti_or(a, "max_seqs", 0);
    const long long max_ctx  = halo_geti_or(a, "max_ctx", 0);
    const long long band     = halo_geti_or(a, "q_len", 0);
    if (n_head > 0 && max_seqs > 0 && max_ctx > 0 && band > 0) {
        const long long gqa = halo_geti_or(a, "gqa", GQA);
        const long long kvh = gqa > 0 ? n_head / gqa : n_head;
        const int tiles = (int)((max_ctx + TILE - 1) / TILE);
        int64_t worst = 0;
        for (long long n = 1; n <= max_seqs; ++n)
            for (long long qt = 1; (qt - 1) * BQ < band; ++qt) {
                const long long units = qt * kvh * n;
                if (units >= HALO_P_UNITS_MAX) break;
                const long long q = std::min<long long>(qt * BQ, band);
                const long long per = halo_prefill_split_bytes(n * q, n_head, HEAD_DIM);
                const int s = halo_prefill_splits((int)units, tiles, per);
                if (s > 1 && s * per > worst) worst = s * per;
            }
        return worst;
    }
    HaloArgs r;
    const int rc = fill_paged_row(a, &r, HEAD_DIM, row > 0 ? row : 2LL * HEAD_DIM);
    if (rc != RAD_OK) return 0;      // no tensors either: the launch runs unsplit
    const int s = halo_prefill_splits_for<BQ, TILE>(r);
    return s > 1 ? (int64_t)s * halo_prefill_split_bytes(halo_q_total(r), r.q_heads, HEAD_DIM) : 0;
}

// Combine the splits' partials into bf16 out: one wave a (token, q head) row, HEAD_DIM / 32 dims a
// lane. A row whose split 0 says l < 0 belongs to an empty sequence and is left as the plain kernel
// leaves it.
template <int HEAD_DIM>
__global__ __launch_bounds__(256)
void halo_attn_prefill_merge(const float* __restrict__ part, uint16_t* __restrict__ out,
                             int rows, int nsplit)
{
    constexpr int DPL = HEAD_DIM / 32;
    const int row = (int)blockIdx.x * 8 + (int)(threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float* ml = part + (size_t)nsplit * rows * HEAD_DIM;
    if (ml[(size_t)row * 2 + 1] < 0.0f) return;
    float mx = -3.0e38f;
    for (int s = 0; s < nsplit; ++s) mx = fmaxf(mx, ml[((size_t)s * rows + row) * 2]);
    float o[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) o[i] = 0.0f;
    float l = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        const float m_s = ml[((size_t)s * rows + row) * 2];
        const float l_s = ml[((size_t)s * rows + row) * 2 + 1];
        if (!(l_s > 0.0f)) continue;           // an empty split: O was never written
        const float w = exp2f(m_s - mx);
        l += w * l_s;
        const float* src = part + ((size_t)s * rows + row) * HEAD_DIM + lane * DPL;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) o[i] += w * src[i];
    }
    const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    uint16_t* dst = out + (size_t)row * HEAD_DIM + lane * DPL;
    #pragma unroll
    for (int i = 0; i < DPL; ++i) dst[i] = f32_to_bf16(o[i] * inv);
}

// The launch every prefill geometry shares: the attention kernel over q tiles x KV heads x
// (sequences x splits), then the merge when it split. Returns 0 or -100 - hipError_t.
template <int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT, int WIN = 0>
static int halo_prefill_launch(const HaloArgs* a, hipStream_t s) {
    constexpr int BQ = (NWARPS * 16) / GQA;
    HaloArgs r = *a;
    if (!(r.splits > 1 && r.scratch)) r.splits = 1;
    dim3 grid((r.q_len + BQ - 1) / BQ, r.kv_heads, r.num_seqs * r.splits);
    hipLaunchKernelGGL((halo_attn_prefill_kernel<NWARPS, TILE, HEAD_DIM, GQA, BS, KVP, OPT, WIN>),
                       grid, dim3(NWARPS * 32), 0, s, r);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -100 - (int)e;
    if (r.splits > 1) {
        const int rows = halo_q_total(r) * r.q_heads;
        hipLaunchKernelGGL((halo_attn_prefill_merge<HEAD_DIM>), dim3((rows + 7) / 8), dim3(256),
                           0, s, (const float*)r.scratch, (uint16_t*)r.out, rows, r.splits);
        e = hipGetLastError();
        if (e != hipSuccess) return -100 - (int)e;
    }
    return 0;
}
