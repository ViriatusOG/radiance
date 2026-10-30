/* Derived from libr4d/r4d_attn_decode_h256_gqa6.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// Halo split-KV paged decode (also the MTP-verify shape), head_dim 256, 6 queries per KV head,
// paged block 16. Same core as libr4d's (lazy reference-max softmax, f16 operands, folded query
// scale, contiguous-k fragments, prefetch + scheduling barrier, dot2 row sum), and the same two
// decode-only options:
//
//   KLDS0  reads K straight from the paged cache (bf16, fp8, fp8_v4 or k6_v4), widening once per
//          warp instead of once per workgroup. Costs VALU on a bandwidth-bound leg and frees sK,
//          which at TILE 16 is 8.3 KB of the 14.5 KB that limits resident workgroups.
//   PF16   split-KV partials stored as f16. At 16K context the partial round trip is ~14 MB
//          against ~17 MB of KV. Storing O/l instead of O bounds values by max|v|, so f16 cannot
//          overflow; the 11-bit mantissa costs ~5e-4 relative.
//
// ---- the gfx1151 port (halo_dt16.h is the authority; read its header first) ---------------------
//
// libr4d keeps the score fragment's COLUMN on (lane & 15) -- the query -- on both architectures,
// which is why this body ports at all. What differs is the ROW: gfx1201 gives a lane eight k of a
// sixteen-key tile (k = 8*(lane>>4) + e), gfx1151 gives it all sixteen of a 16-wide head-dim run
// and splits the tile's keys even/odd across the lane half-pair (element e = key 2e + (lane>>4)).
// Every fragment gather therefore reads 32 bytes at the un-split offset (two uint4, eight dwords),
// the mask and the causal test use key 2e + h, the P fragment is assembled across the half-pair
// with p16w(), and the partial store is written at head dim 2e + h. The swap16 (lane ^ 16) row-max
// and row-sum reductions mean the same thing on both, so that algebra is libr4d's unchanged.
//
// Dropped, each a gfx12-only or unused instantiation here (static_asserted rather than left to
// rot): the split (non-CONTIG) cache map -- the paged cache is read contiguous, KSTR HEAD_DIM + 8
// -- O_KRAW's one-round raw K (its shape changes with the fragment width and nothing asks for it),
// and O_V8, the fp8 code tile in LDS. The fp8_v4 / k6_v4 V staging (fetchV4 / storeV4) is
// LDS-layout code and ports as it stands.
//
// KLDS0 does not port as it stands. Read straight from the cache, every row-tile wave paid the
// tile's DRAM round trip after the staging barrier and widened its K itself, so each 16-row tile of
// a verify added 65-170 us at 16K context (q_len 6 at gqa 6 took 2.1x q_len 1). Here K goes through
// sK after all, widened once per workgroup: its codes are fetched a tile ahead into registers, as
// GPREV fetches V, and widened into sK at the staging barrier (KST below). The per-wave read is
// kept only where that prefetch does not fit the registers. Likewise V: gfx1151 has no
// global_load_tr_b128, and load_tr_b128's emulation shuffles the loaded dwords at once, so GPREV's
// prefetch waited on its own load. fetchV reads four keys' dwords instead and storeV transposes
// them with v_perm at the barrier. Both stage the same f16 values as before, so a launch's outputs
// are unchanged bit for bit; q_len 10 at gqa 6 now takes 1.25x q_len 1 (both 16K, 166 / 208 us).
#include "halo_launch.h"   // HALO_DYN_LDS: tests/halo_emu runs this body on the host
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_dt16.h"
#include "halo_q8.h"
#include "halo_chain.h"
#include "halo_attn_args.h"
#include <algorithm>
#include <cmath>
#include <type_traits>

#define O_F16    1
#define O_MSKIP  2
#define O_FOLDQ  8
#define O_KLDS0 16
#define O_PF16  32
#define O_CONTIG 512
#define O_PF2   1024
#define O_PF4   2048
#define O_PF8   3072
#define O_SGB   4096
#define O_DOT2 16384
// Tile-staging options, shared with libr4d's prefill unit:
//   BTS   the tile's block-table entries in SGPRs, loaded a tile ahead. With KLDS0 this also takes
//         the dependent bt[kk/BS] load out of the m-tile loop.
//   GPREV tile ti+1's V fetched into registers before the m-tile loop.
//   LDSB  an LDS-scoped barrier, so the barrier does not invalidate the vector cache.
#define O_BTS   262144
#define O_GPREV 1048576
#define O_LDSB  2097152
#define O_V8    8388608
#define O_KRAW  16777216
// O_V4: the fp8_v4 cache (E4M3 K, int4 V in groups of 32 with an f16 scale), and with O_K6 the
// k6_v4 cache (int6 K, int4 V); a token's row layout is RadBuildCtx::kv_quant's (abi/rad_builder.h).
#define O_V4    33554432
#define O_K6    67108864
// The two views of HaloArgs compiled in only where a row asks for them:
//   QLD    q rows at HaloArgs::q_ld (attn_paged_qrow), not dense.
//   KVREP  query group g reads cache head g / HaloArgs::kv_rep, descales per cache head.
#define O_QLD   134217728
#define O_KVREP 268435456
// UNION  a workgroup's "sequence" is a group of consecutive query rows (cu_q), its block table the
//        union of their pages and HaloArgs::umask which of a page's tokens each row attends to; the
//        mask replaces the causal limit (halo_attn_qsa_union in the paged unit).
#define O_UNION 536870912

__host__ __device__ __forceinline__ int halo_attn_tiles(int ctx, int tile) {
    return (ctx + tile - 1) / tile;
}
// First KV position query row `qrow` may attend to under a sliding window: the row's causal limit,
// less the window, floored at zero. window <= 0 means no window.
__host__ __device__ __forceinline__ int halo_attn_wlo(int ctx, int q_len, int qrow, int window) {
    if (window <= 0) return 0;
    const int lo = ctx - q_len + qrow - window + 1;
    return lo > 0 ? lo : 0;
}
/* One query dword -- two bf16 -- to the operand format. libr4d's r4d_qcvt lives in its prefill
 * unit because the paged dispatch includes prefill first; the halo decode body is a unit of its
 * own, so the helper lives here (halo_dt16.h stays the traits layer). */
template <class D, int FOLDQ, int F16>
__device__ __forceinline__ uint32_t halo_attn_qcvt(uint32_t w, float s) {
    if (!FOLDQ) return F16 ? D::from_bf16w(w) : w;
    return D::pk(__builtin_bit_cast(float, w << 16) * s,
                 __builtin_bit_cast(float, w & 0xffff0000u) * s);
}

// Waves a 16-row tile is split over along the head (split-D). At head 256 one wave would hold the
// query (16 fragments, 128 VGPRs) and the accumulator (16 x v8f, 128 VGPRs) at once, which is the
// whole register file: every instantiation spilled 180-450 VGPR slots (~1.4 KB of scratch a lane,
// re-read every tile) and ran at ~40 GB/s of KV. Two waves of 128 dims each hold 64 + 64 and add
// their partial S tiles through LDS (in wave order, so both see the same S and take the same softmax).
__host__ __device__ constexpr int halo_dec_ds(int head_dim) {
    return (head_dim >= 256 && head_dim % 128 == 0) ? head_dim / 128 : 1;
}

// WIN enables a sliding window (a.window positions ending at each row's causal limit). As a
// template parameter defaulting to 0, full attention compiles with no window code.
// The body: its blocks return early (empty slots, tiles past the window), so the chained form
// (halo_chain.h) wraps it below rather than bracketing it in place.
template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT,
         int WIN = 0>
__device__ __forceinline__
void halo_attn_decode_body(const HaloArgs& a, int splits)
{
    constexpr int F16   = (OPT & O_F16)   ? 1 : 0;
    constexpr int MSKIP = (OPT & O_MSKIP) ? 1 : 0;
    constexpr int FOLDQ = (OPT & O_FOLDQ) ? 1 : 0;
    constexpr int KLDS  = (OPT & O_KLDS0) ? 0 : 1;
    constexpr int PF16  = (OPT & O_PF16)  ? 1 : 0;
    constexpr int CTG   = (OPT & O_CONTIG) ? 1 : 0;
    constexpr int PF    = 1 << ((OPT >> 10) & 3);
    constexpr int SGB   = (OPT & O_SGB) ? 1 : 0;
    constexpr int DOT2  = ((OPT & O_DOT2) && (OPT & O_F16)) ? 1 : 0;
    constexpr int BTS   = (OPT & O_BTS)  ? 1 : 0;
    constexpr int LDSB  = (OPT & O_LDSB) ? 1 : 0;
    constexpr int V4    = ((OPT & O_V4) && !KVP) ? 1 : 0;
    constexpr int K6    = ((OPT & O_K6) && V4) ? 1 : 0;
    // A token's row in a page, and where in it V's codes, V's scales and K's scales start.
    constexpr int ROWB  = K6 ? HEAD_DIM * 22 / 16 : V4 ? HEAD_DIM * 25 / 16 : 2 * HEAD_DIM;
    constexpr int VOFF  = K6 ? HEAD_DIM * 3 / 4 : HEAD_DIM;
    constexpr int VSOFF = K6 ? HEAD_DIM * 21 / 16 : HEAD_DIM * 3 / 2;
    constexpr int KSOFF = HEAD_DIM * 5 / 4;
    constexpr int QLD   = (OPT & O_QLD)   ? 1 : 0;
    constexpr int KVREP = (OPT & O_KVREP) ? 1 : 0;
    constexpr int UNION = (OPT & O_UNION) ? 1 : 0;
    static_assert(!UNION || (BS == 4 && TILE % 16 == 0 && !WIN && !KVREP),
                  "the union mask is a word a 4-token page, four pages a 16-key m-tile");
    static_assert(CTG, "gfx1151 reads the paged cache in the contiguous layout (KSTR HEAD_DIM + 8)");
    static_assert(!(OPT & O_KRAW), "the one-round raw K round is not instantiated here");
    static_assert(!(OPT & O_V8), "the fp8 code tile in LDS is not instantiated here");
    static_assert(!K6 || !KLDS, "k6_v4 K is widened from the paged cache, a lane's 16 dims contiguous");
    static_assert(!V4 || F16, "fp8_v4 widens to f16");
    typedef DT16<F16> D;
    typedef typename D::frag frag16;

    constexpr int NKS   = HEAD_DIM / 16;
    constexpr int DS    = halo_dec_ds(HEAD_DIM);    // waves along the head a row tile
    constexpr int NKD   = NKS / DS;                 // a wave's 16-dim fragments
    constexpr int NWT   = NWARPS * DS;              // waves a block
    constexpr int MT    = TILE / 16;
    constexpr int KSTR  = HEAD_DIM + 8;
    constexpr int VSTR  = TILE + 8;
    constexpr int NTHR  = NWT * 32;
    constexpr int NB    = TILE / BS;
    // V is staged in units of four keys by DPU dims, one dword a key (four E4M3 or two bf16).
    constexpr int DPU   = KVP ? 2 : 4;
    constexpr int NVU   = (TILE / 4) * (HEAD_DIM / DPU);
    constexpr int VREGS = (NVU + NTHR - 1) / NTHR;
    // A V prefetch costs VREGS x 4 VGPRs on top of a 128-VGPR accumulator and a 64-VGPR query.
    // Up to three uint4 it fits in 255 VGPRs without spilling; at four or more it spills and
    // loses ~10%, so the geometry gates it, not just the flag.
    constexpr int GPREV = ((OPT & O_GPREV) && VREGS <= 3 && !V4) ? 1 : 0;
    // KST: under KLDS0, the tile's K codes (KCH 16-byte chunks) fetched a tile ahead, KREGS uint4 a
    // thread, and widened into sK once. Up to four fit beside the rest (two waves at head 256 fp8,
    // one at 128); bf16 at one h128 wave would need eight, and reads K per wave as libr4d does.
    // k6_v4 K is widened from the cache (its static_assert above).
    constexpr int KCH   = TILE * HEAD_DIM * (KVP ? 2 : 1) / 16;
    constexpr int KREGS = (KCH + NTHR - 1) / NTHR;
    constexpr int KST   = (!KLDS && !K6 && KREGS <= 4) ? 1 : 0;
    constexpr float PSHIFT = D::SHIFT;
    constexpr float PGROW  = D::GROW;

    __shared__ uint16_t sK[(KLDS || KST) ? TILE * KSTR : 1];
    // V4 at TILE 16: dim octet o's 8 rows start S(o) = 16 (o & 3) + 8 ((o >> 2) & 1) + 128 (o >> 2)
    // halfwords later, so the 4 (8) octets of a storeV4 store land on distinct banks. The skew is an
    // LDS bank offset, identical on both architectures.
    constexpr int VSK  = (V4 && TILE == 16 && NWARPS >= 2) ? 1 : 0;
    constexpr int VSKN = HEAD_DIM / 8 - 1;
    constexpr int VSKX = VSK ? 16 * (VSKN & 3) + 8 * ((VSKN >> 2) & 1) + 128 * (VSKN >> 2) : 0;
    __shared__ uint16_t sV16[HEAD_DIM * VSTR + VSKX];
    // Split-D: each wave's partial S tile, [row tile][dh][element][lane].
    __shared__ float sSx[DS > 1 ? NWT * 8 * 32 : 1];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // The launch is NTHR threads. Stated (k6_v4), so the V4 rounds wholly below NP4 lose their
    // range test: as exec-masked blocks, each round's loads issued only after the previous returned.
    if constexpr (K6) __builtin_assume(tid < NTHR);
    const int c = lane & 15, h = lane >> 4;

    const int sp = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z;
    // kvh is the query group; under KVREP the cache head it reads is kvh / kv_rep
    const int kvc = KVREP ? kvh / a.kv_rep : kvh;
    const int ctx = a.seqused_k[seq];
    if (ctx <= 0) return;

    // Ragged batches (HaloArgs::cu_q): this sequence's q/out rows and query length.
    const int qoff = halo_q_off(a.cu_q, seq, a.q_len);
    const int qlen = halo_q_len(a.cu_q, seq, a.q_len);
    if (qlen <= 0) return;                 // a slot carrying no query rows this step

    const int ntl  = halo_attn_tiles(ctx, TILE);
    // A sliding-window layer must not read below the window: those blocks may already be freed and
    // reassigned to another sequence. The walk starts at the tile holding row 0's first window
    // token (row 0 reaches furthest back), and the splits divide that range.
    const int wlo0 = WIN ? halo_attn_wlo(ctx, qlen, 0, a.window) : 0;
    const int tb   = WIN ? wlo0 / TILE : 0;
    const int neff = ntl - tb;
    const int tps  = (neff + splits - 1) / splits;
    const int t_lo = tb + sp * tps;
    if (t_lo >= ntl) return;
    const int t_hi = min(t_lo + tps, ntl);

    // Row tile and head slice; at DS 1 the wave is the row tile, as before the split.
    const int rw = DS > 1 ? warp % NWARPS : warp, dh = DS > 1 ? warp / NWARPS : 0;
    const int row = rw * 16 + c;
    const int qi = row / GQA, hi = row - qi * GQA;
    const bool live = (qi < qlen);
    const int qrow = live ? qi : (qlen - 1);
    const int qhead = kvh * GQA + hi;

    const int* __restrict__ bt = a.block_table + (size_t)seq * a.max_blocks;
    const float kdesc = a.k_descale
                      ? a.k_descale[seq * (KVREP ? a.kv_heads / a.kv_rep : a.kv_heads) + kvc]
                      : 1.0f;
    const float mul   = a.scale * kdesc * 1.44269504089f;

    frag16 qf[NKD];
    {
        const uint16_t* qp = QLD ? (const uint16_t*)a.q + (size_t)(qoff + qrow) * a.q_ld
                                                        + (size_t)qhead * HEAD_DIM
                                 : (const uint16_t*)a.q
                                   + ((size_t)(qoff + qrow) * a.q_heads + qhead) * HEAD_DIM;
        // gfx1151: the lane holds all sixteen dims of the run, so one 32-byte read and no
        // lane-half term.
#pragma unroll
        for (int t = 0; t < NKD; ++t) {
            const uint4 u0 = *(const uint4*)(qp + 16 * (dh * NKD + t));
            const uint4 u1 = *(const uint4*)(qp + 16 * (dh * NKD + t) + 8);
            const uint32_t w8[8] = { halo_attn_qcvt<D, FOLDQ, F16>(u0.x, mul),
                                     halo_attn_qcvt<D, FOLDQ, F16>(u0.y, mul),
                                     halo_attn_qcvt<D, FOLDQ, F16>(u0.z, mul),
                                     halo_attn_qcvt<D, FOLDQ, F16>(u0.w, mul),
                                     halo_attn_qcvt<D, FOLDQ, F16>(u1.x, mul),
                                     halo_attn_qcvt<D, FOLDQ, F16>(u1.y, mul),
                                     halo_attn_qcvt<D, FOLDQ, F16>(u1.z, mul),
                                     halo_attn_qcvt<D, FOLDQ, F16>(u1.w, mul) };
            qf[t] = D::mk(w8);
        }
    }

    // The row's absolute position. Causal masking bounds the key by it; non-causal masking bounds it
    // by the sequence, with the window applied on both sides (a block-diffusion drafter's own
    // block).
    const int pos = ctx - qlen + qrow;
    const int cz  = a.causal_seq ? (int)((const unsigned char*)a.causal_seq)[seq] : a.causal;
    const int klimit = cz ? pos
                          : ((WIN && a.window > 0) ? min(ctx - 1, pos + a.window - 1)
                                                   : ctx - 1);
    const int wlo    = (WIN && a.window > 0) ? (pos - a.window + 1 > 0
                                               ? pos - a.window + 1 : 0) : 0;
    float m_ref = -3.0e38f;
    float off   = PSHIFT + 3.0e38f;
    float l_i   = 0.0f;
    v8f acc[NKD];
#pragma unroll
    for (int t = 0; t < NKD; ++t) acc[t] = (v8f){0,0,0,0,0,0,0,0};

    uint4 vreg[VREGS];

    // ---- fp8_v4 / k6_v4: V's int4 codes into sV16 ([dim][key]) as libr4d stages them: a key pair's
    // 8 dims a thread, two code words and the pair's scale, widened with packed f16 into one dword a
    // dim, fetched in the tile: no prefetch.
    constexpr int NP4 = (TILE / 2) * (HEAD_DIM / 8);
    constexpr int U4  = V4 ? (NP4 + NTHR - 1) / NTHR : 1;
    // The pair's two scales stay apart until storeV4 packs them: packed here, the fetch is waited
    // on right after it is issued. fp8_v4 packs here, as libr4d's prefill unit does.
    constexpr int OLDV4 = V4 && !K6;
    uint32_t v4c[U4][2], v4s[U4][2];
    auto fetchV4 = [&](int k0, const int* blks) {
#pragma unroll
        for (int r = 0; r < U4; ++r) {
            const int u = tid + r * NTHR;
            // k6_v4, a partial round (NWARPS 3: the third, warps 0-1): u < NP4 <=> warp <
            // (NP4 - r NTHR) / 32 when NP4 is a multiple of 32, tested on the wave-uniform warp.
            if (NP4 % NTHR == 0 || (K6 ? ((r + 1) * NTHR <= NP4
                || (NP4 % 32 == 0 ? __builtin_amdgcn_readfirstlane(warp) * 32 < NP4 - r * NTHR
                                  : u < NP4)) : u < NP4)) {
                const int kp = u % (TILE / 2), o = u / (TILE / 2);
                uint32_t sc[2];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    int kk = min(k0 + 2 * kp + j, ctx - 1);
                    if (WIN && kk < wlo0) kk = wlo0;
                    const int bk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                    const uint8_t* base = (const uint8_t*)a.kv + (size_t)bk * a.kv_block_stride
                                        + (size_t)kvc * a.kv_head_stride + (size_t)(kk % BS) * ROWB;
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
            if (NP4 % NTHR == 0 || (K6 ? ((r + 1) * NTHR <= NP4
                || (NP4 % 32 == 0 ? __builtin_amdgcn_readfirstlane(warp) * 32 < NP4 - r * NTHR
                                  : u < NP4)) : u < NP4)) {
                const int kp = u % (TILE / 2), o = u / (TILE / 2);
                // Two's complement nibbles; ^ 8 makes them offset binary (value + 8).
                const uint32_t n0 = (v4c[r][0] & 0x0F0F0F0Fu) ^ 0x08080808u;
                const uint32_t m0 = ((v4c[r][0] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
                const uint32_t n1 = (v4c[r][1] & 0x0F0F0F0Fu) ^ 0x08080808u;
                const uint32_t m1 = ((v4c[r][1] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
                const h2 s2 = __builtin_bit_cast(h2, OLDV4 ? v4s[r][0] : (v4s[r][0] | (v4s[r][1] << 16)));
                const h2 bias = { (_Float16)1032.0f, (_Float16)1032.0f };
                uint16_t* dst = &sV16[(o * 8) * VSTR + 2 * kp
                                      + (VSK ? 16 * (o & 3) + 8 * ((o >> 2) & 1) + 128 * (o >> 2) : 0)];
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const uint32_t sel = (uint32_t)b | 0x0c00u | ((uint32_t)(4 + b) << 16) | 0x0c000000u;
                    const uint32_t lo = __builtin_amdgcn_perm(n1, n0, sel) | 0x64006400u;
                    const uint32_t hi = __builtin_amdgcn_perm(m1, m0, sel) | 0x64006400u;
                    const h2 vl = (__builtin_bit_cast(h2, lo) - bias) * s2;
                    const h2 vh = (__builtin_bit_cast(h2, hi) - bias) * s2;
                    *(uint32_t*)(dst + (2 * b) * VSTR)     = __builtin_bit_cast(uint32_t, vl);
                    *(uint32_t*)(dst + (2 * b + 1) * VSTR) = __builtin_bit_cast(uint32_t, vh);
                }
            }
        }
    };

    // ---- V into sV16 ([dim][key]): unit u is keys 4 kq .. 4 kq + 3 at dims DPU dq .., kq the fast
    // index, so a wave's 8-byte stores cover 16 of the 32 banks (twice the minimum cycles). fetchV
    // leaves the four dwords raw, a prefetch that does not wait on itself; storeV transposes them to
    // one dword a dim with v_perm (a 4x4 byte transpose for E4M3, then widened; a 2x2 halfword one
    // for bf16), four keys of a dim in one 8-byte store.
    auto fetchV = [&](int k0, const int* blks) {
#pragma unroll
        for (int r = 0; r < VREGS; ++r) {
            const int u = tid + r * NTHR;
            if (NVU % NTHR == 0 || u < NVU) {
                const int kq = u % (TILE / 4), dq = u / (TILE / 4);
                uint32_t w[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int kk = min(k0 + 4 * kq + i, ctx - 1);
                    if (WIN && kk < wlo0) kk = wlo0;
                    const int blk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                    const size_t o = ((size_t)blk * a.kv_block_stride + (size_t)kvc * a.kv_head_stride
                                   + (size_t)(kk % BS) * ROWB + HEAD_DIM + dq * DPU) * (KVP ? 2 : 1);
                    w[i] = *(const uint32_t*)((const uint8_t*)a.kv + o);
                }
                vreg[r] = make_uint4(w[0], w[1], w[2], w[3]);
            }
        }
    };
    // ---- KST: chunk idx is key idx / CPK, bytes 16 (idx % CPK) of its K; stored widened, f16 at
    // KSTR, the layout the KLDS fragment read below expects.
    uint4 kreg[KST ? KREGS : 1];
    auto fetchK = [&](int k0, const int* blks) {
#pragma unroll
        for (int r = 0; r < KREGS; ++r) {
            const int idx = tid + r * NTHR;
            if (KCH % NTHR == 0 || idx < KCH) {
                constexpr int CPK = KCH / TILE;
                const int key = idx / CPK, ch = idx - key * CPK;
                int kk = min(k0 + key, ctx - 1);
                if (WIN && kk < wlo0) kk = wlo0;
                const int kb = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = ((size_t)kb * a.kv_block_stride + (size_t)kvc * a.kv_head_stride
                               + (size_t)(kk % BS) * ROWB) * (KVP ? 2 : 1) + ch * 16;
                kreg[r] = *(const uint4*)((const uint8_t*)a.kv + o);
            }
        }
    };
    auto storeK = [&]() {
#pragma unroll
        for (int r = 0; r < KREGS; ++r) {
            const int idx = tid + r * NTHR;
            if (KCH % NTHR == 0 || idx < KCH) {
                constexpr int CPK = KCH / TILE;
                const int key = idx / CPK, ch = idx - key * CPK;
                const uint4 u = kreg[r];
                if (KVP) {
                    uint16_t* kd = &sK[key * KSTR + ch * 8];
                    *(uint4*)kd = make_uint4(D::from_bf16w(u.x), D::from_bf16w(u.y),
                                             D::from_bf16w(u.z), D::from_bf16w(u.w));
                } else {
                    uint32_t w8[8];
                    fp8x16_to_16x8w<D>(u, w8);
                    uint16_t* kd = &sK[key * KSTR + ch * 16];
                    *(uint4*)(kd)     = make_uint4(w8[0], w8[1], w8[2], w8[3]);
                    *(uint4*)(kd + 8) = make_uint4(w8[4], w8[5], w8[6], w8[7]);
                }
            }
        }
    };

    auto storeV = [&]() {
#pragma unroll
        for (int r = 0; r < VREGS; ++r) {
            const int u = tid + r * NTHR;
            if (NVU % NTHR == 0 || u < NVU) {
                const int kq = u % (TILE / 4), dq = u / (TILE / 4);
                const uint4 v = vreg[r];
                uint16_t* dst = &sV16[(dq * DPU) * VSTR + 4 * kq];
                if (KVP) {
                    const uint32_t f0 = D::from_bf16w(v.x), f1 = D::from_bf16w(v.y);
                    const uint32_t f2 = D::from_bf16w(v.z), f3 = D::from_bf16w(v.w);
                    *(uint2*)(dst) = make_uint2(__builtin_amdgcn_perm(f1, f0, 0x05040100u),
                                                __builtin_amdgcn_perm(f3, f2, 0x05040100u));
                    *(uint2*)(dst + VSTR) = make_uint2(__builtin_amdgcn_perm(f1, f0, 0x07060302u),
                                                       __builtin_amdgcn_perm(f3, f2, 0x07060302u));
                } else {
                    const uint32_t a01 = __builtin_amdgcn_perm(v.y, v.x, 0x05010400u);
                    const uint32_t b01 = __builtin_amdgcn_perm(v.y, v.x, 0x07030602u);
                    const uint32_t a23 = __builtin_amdgcn_perm(v.w, v.z, 0x05010400u);
                    const uint32_t b23 = __builtin_amdgcn_perm(v.w, v.z, 0x07030602u);
                    const uint32_t t[4] = { __builtin_amdgcn_perm(a23, a01, 0x05040100u),
                                            __builtin_amdgcn_perm(a23, a01, 0x07060302u),
                                            __builtin_amdgcn_perm(b23, b01, 0x05040100u),
                                            __builtin_amdgcn_perm(b23, b01, 0x07060302u) };
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        *(uint2*)(dst + j * VSTR) =
                            make_uint2(D::pkv(halo_fp8::e4m3_pk<false>(t[j])),
                                       D::pkv(halo_fp8::e4m3_pk<true>(t[j])));
                }
            }
        }
    };

    // A tile's NB entries are read whole, and a multi-block tile can run past the sequence's last
    // block (on the last row, past the end of the table). Keys are clamped to ctx - 1, so entries
    // past lastb are never used; only tiles reaching past lastb take the clamped load, and the rest
    // keep one wide scalar load. With NB == 1 the test folds away.
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
    if (BTS) load_bt(t_lo * NB, blk);
    if (GPREV) fetchV(t_lo * TILE, blk);
    if (KST) fetchK(t_lo * TILE, blk);

    for (int ti = t_lo; ti < t_hi; ++ti) {
        const int k0 = ti * TILE;
        int blkn[NB];
        if (BTS && ti + 1 < t_hi) {
            load_bt((ti + 1) * NB, blkn);
        }
        if (LDSB) lds_barrier(); else __syncthreads();

        if (KLDS) {
#pragma unroll 1
            for (int i = tid; i < TILE * (HEAD_DIM / 8); i += NTHR) {
                const int key = i / (HEAD_DIM / 8), ch = i - key * (HEAD_DIM / 8);
                int kk = min(k0 + key, ctx - 1);
                if (WIN && kk < wlo0) kk = wlo0;
                const int kb = BTS ? blk[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)kb * a.kv_block_stride
                               + (size_t)kvc * a.kv_head_stride + (size_t)(kk % BS) * ROWB;
                uint16_t* kd = &sK[key * KSTR];
                uint32_t w4[4];
                if (KVP) {
                    const uint4 r = *(const uint4*)((const uint16_t*)a.kv + o + ch * 8);
                    w4[0] = D::from_bf16w(r.x); w4[1] = D::from_bf16w(r.y);
                    w4[2] = D::from_bf16w(r.z); w4[3] = D::from_bf16w(r.w);
                } else {
                    fp8x8_to_16x4w<D>(*(const uint2*)((const uint8_t*)a.kv + o + ch * 8), w4);
                }
                *(uint2*)(kd + ch * 8)     = make_uint2(w4[0], w4[1]);
                *(uint2*)(kd + ch * 8 + 4) = make_uint2(w4[2], w4[3]);
            }
        }
        if (KST) storeK();
        if constexpr (V4) { fetchV4(k0, blk); storeV4(); }
        else { if (!GPREV) fetchV(k0, blk); storeV(); }
        if (LDSB) lds_barrier(); else __syncthreads();
        if (GPREV && ti + 1 < t_hi) fetchV(k0 + TILE, blkn);
        if (KST && ti + 1 < t_hi) fetchK(k0 + TILE, blkn);

#pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            const uint8_t* grow = nullptr;
            if (!KLDS && !KST) {
                int kk = min(k0 + mt * 16 + c, ctx - 1);
                if (WIN && kk < wlo0) kk = wlo0;
                const int kb = BTS ? blk[(kk - k0) / BS] : bt[kk / BS];
                grow = (const uint8_t*)a.kv + ((size_t)kb * a.kv_block_stride
                     + (size_t)kvc * a.kv_head_stride + (size_t)(kk % BS) * ROWB)
                     * (KVP ? 2 : 1);
            }
            v8f s = (v8f){0,0,0,0,0,0,0,0};
#pragma unroll
            for (int g = 0; g < NKD; g += PF) {
                frag16 kb[PF];
#pragma unroll
                for (int j = 0; j < PF; ++j) {
                    const int t = dh * NKD + g + j;
                    if (KLDS || KST) {
                        // A lane's whole 16-dim run of key (mt*16 + c): two uint4, no lane-half.
                        const uint16_t* kp = &sK[(mt * 16 + c) * KSTR + 16 * t];
                        const uint4 u0 = *(const uint4*)(kp);
                        const uint4 u1 = *(const uint4*)(kp + 8);
                        const uint32_t w8[8] = { u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w };
                        kb[j] = D::mk(w8);
                    } else if (KVP) {
                        const uint16_t* kp = (const uint16_t*)grow + 16 * t;
                        const uint4 u0 = *(const uint4*)(kp);
                        const uint4 u1 = *(const uint4*)(kp + 8);
                        const uint32_t w8[8] = { D::from_bf16w(u0.x), D::from_bf16w(u0.y),
                                                 D::from_bf16w(u0.z), D::from_bf16w(u0.w),
                                                 D::from_bf16w(u1.x), D::from_bf16w(u1.y),
                                                 D::from_bf16w(u1.z), D::from_bf16w(u1.w) };
                        kb[j] = D::mk(w8);
                    } else if constexpr (K6) {
                        // k6_v4: this lane's sixteen dims (16t ..) from their low nibbles, the
                        // chunk's high-bit word and the group's scale -- both halves of the unpack,
                        // since a gfx1151 lane holds the whole run (halo_dt16.h).
                        uint32_t w8[8];
                        const uint32_t sb = *(const uint16_t*)(grow + KSOFF + (t >> 1) * 2);
                        const uint32_t hb = *(const uint32_t*)(grow + HEAD_DIM / 2 + 4 * t);
                        const uint32_t s2 = sb | (sb << 16);
                        k6x8_to_f16x4w(*(const uint32_t*)(grow + 8 * t),     hb, 0, s2, w8 + 0);
                        k6x8_to_f16x4w(*(const uint32_t*)(grow + 8 * t + 4), hb, 1, s2, w8 + 4);
                        kb[j] = D::mk(w8);
                    } else {
                        // fp8-e4m3 cache: 16 codes, one 16-byte load, widened in element order.
                        const uint8_t* kp = grow + 16 * t;
                        uint32_t w8[8];
                        fp8x16_to_16x8w<D>(*(const uint4*)kp, w8);
                        kb[j] = D::mk(w8);
                    }
                }
                if (SGB) sched_barrier();
#pragma unroll
                for (int j = 0; j < PF; ++j) s = D::wmma(kb[j], qf[g + j], s);
            }
            if constexpr (DS > 1) {
                // The tile's S is the sum of the DS head slices' partials, added in slice order
                // from LDS by every wave of the row tile, so all of them hold the same S.
                float* xs = sSx + rw * (DS * 256);
#pragma unroll
                for (int e = 0; e < 8; ++e) xs[dh * 256 + e * 32 + lane] = s[e];
                if (LDSB) lds_barrier(); else __syncthreads();
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    float v = xs[e * 32 + lane];
#pragma unroll
                    for (int q = 1; q < DS; ++q) v += xs[q * 256 + e * 32 + lane];
                    s[e] = v;
                }
                // The next write is after the next tile's staging barriers; a second m-tile
                // within this one needs its own.
                if (MT > 1) { if (LDSB) lds_barrier(); else __syncthreads(); }
            }
            float sc[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) sc[e] = FOLDQ ? s[e] : s[e] * mul;

            // gfx1151: element e of this lane is key 2e + h of the 16-key tile (halo_dt16.h).
            const int kbase = k0 + mt * 16;
            if constexpr (UNION) {
                // Key 2e + h is in page e / 2 of the tile, token 2 (e & 1) + h; the row attends to
                // it when its four bits in the page's word exceed that token.
                const uint32_t* um = a.umask + (size_t)seq * a.max_blocks + (kbase >> 2);
                uint32_t mr[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) mr[i] = (um[i] >> (4 * qrow)) & 15u;
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    sc[e] = mr[e >> 1] > (uint32_t)(2 * (e & 1) + h) ? sc[e] : -INFINITY;
            } else
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
                for (int dt = 0; dt < NKD; ++dt)
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[dt][e] *= alpha;
                l_i *= alpha;
                m_ref = mnew;
                off   = PSHIFT - mnew;
            }

            float pf[8], lsum = 0.0f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float p = ex2(sc[e] + off); pf[e] = p;
                if (!DOT2) lsum += p;
            }
            if (!DOT2) l_i += lsum;

            {
                uint32_t pw[4];
#pragma unroll
                for (int e = 0; e < 8; e += 2) pw[e >> 1] = D::pk(pf[e], pf[e + 1]);
                if (DOT2) {
                    const v2hf_dt one = {(__fp16)1.0f, (__fp16)1.0f};
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        l_i = __builtin_amdgcn_fdot2(__builtin_bit_cast(v2hf_dt, pw[e]), one, l_i, false);
                }
                // P's sixteen keys are one parity a lane, so the 16-consecutive-key B fragment is
                // assembled across the half-pair (halo_dt16.h's p16w).
                uint32_t pw16[8];
                p16w<D>(pw, pw16);
                const frag16 p16 = D::mk(pw16);
                // Skewed tile: row 16 G + c is in octet 2 G + (c >> 3), so its S is 16 (c >> 3) here
                // plus 32 (G & 1) + 8 ((G >> 1) & 1) + 128 (G >> 1), a compile-time offset below.
                const uint16_t* vbase = &sV16[c * VSTR + mt * 16] + (VSK ? 16 * (c >> 3) : 0);
#pragma unroll
                for (int g = 0; g < NKD; g += PF) {
                    frag16 vb[PF];
#pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        const int gd = dh * NKD + g + j;      // the head's 16-dim fragment
                        const uint16_t* vp = vbase + gd * 16 * VSTR
                            + (VSK ? 32 * (gd & 1) + 8 * ((gd >> 1) & 1) + 128 * (gd >> 1) : 0);
                        // The lane's whole 16-key run of dim c + 16 (g + j): two uint4.
                        const uint4 u0 = *(const uint4*)vp;
                        const uint4 u1 = *(const uint4*)(vp + 8);
                        const uint32_t w8[8] = { u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w };
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
    const float lrow = l_i + swap16(l_i);
    const int tok = qoff + qrow;
    const size_t idx = (size_t)(tok * a.q_heads + qhead) * splits + sp;
    const size_t nrow = (size_t)halo_q_total(a) * a.q_heads * splits;
    if (PF16) {
        // O/l in f16: bounded by max|v|, so no overflow, and the merge weight becomes w*l.
        uint16_t* so = (uint16_t*)a.scratch;
        float* sm = (float*)(so + nrow * HEAD_DIM);
        float* sl = sm + nrow;
        uint16_t* op = so + idx * HEAD_DIM + dh * NKD * 16;
        const float inv = (lrow > 0.0f) ? (1.0f / lrow) : 0.0f;
#pragma unroll
        for (int dt = 0; dt < NKD; ++dt) {
            uint32_t tw[4];
#pragma unroll
            for (int e = 0; e < 8; e += 2) tw[e >> 1] = DT16<1>::pk(acc[dt][e] * inv,
                                                                    acc[dt][e + 1] * inv);
            // gfx1151: the lane's eight values sit at head dims 2e + h, one parity, so the tile's
            // sixteen dims are interleaved across the half-pair and one half writes the 32 bytes.
            uint32_t o16[8];
            p16w<D>(tw, o16);
            if (h == 0) {
                *(uint4*)(op + dt * 16)     = make_uint4(o16[0], o16[1], o16[2], o16[3]);
                *(uint4*)(op + dt * 16 + 8) = make_uint4(o16[4], o16[5], o16[6], o16[7]);
            }
        }
        if (h == 0 && dh == 0) { sm[idx] = m_ref; sl[idx] = lrow; }
    } else {
        float* so = (float*)a.scratch;
        float* sm = so + nrow * HEAD_DIM;
        float* sl = sm + nrow;
        float* op = so + idx * HEAD_DIM + dh * NKD * 16;
        // Element e is head dim 2e + h, so the f32 form is a scatter of eight values.
#pragma unroll
        for (int dt = 0; dt < NKD; ++dt)
#pragma unroll
            for (int e = 0; e < 8; ++e) op[dt * 16 + 2 * e + h] = acc[dt][e];
        if (h == 0 && dh == 0) { sm[idx] = m_ref; sl[idx] = lrow; }
    }
}

// ---- Output gate and fp8 quantiser as the merge's epilogue (attn_paged_gate_quant) ----
//
// A gated attention's merged row feeds gate_quant_fp8: bf16(sigmoid(gate)) times the row, rounded to
// bf16, then E4M3 codes with one scale per 128 columns. A merge row is one (token, head) pair and a
// scale group is 128 columns, so the `_gq` forms finish the row in registers instead of in a second
// dispatch.
//
// Byte-identical to gate_quant_fp8: gate rounded to bf16 before the multiply, product rounded, scale
// = group absmax / 448 (1 at zero), codes clamped to +-448, and the bf16 product stored to `out`. A
// padded slot merges to zeros and takes the same epilogue. GQ == 2 is the int8 form (scale amax /
// 127, round to nearest-even, halo_q8.h).
struct HaloGateQuant {
    const uint16_t* g;          // gate, row `job` at g + job * g_ld
    uint8_t*        q;          // E4M3 (or int8, `i8`) codes, row `job` at q + job * q_ld
    float*          s;          // one scale per 128 columns, row `job` at s + job * s_ld
    unsigned        g_ld, q_ld, s_ld;
    int             i8;         // read by the launcher only: which combine to instantiate
};
constexpr int kHaloGqGroup = 128;

__device__ __forceinline__ uint16_t halo_gq_gate(uint16_t gb, uint16_t xb) {
    const float    gv = bf16_to_f32(gb);
    const uint16_t ab = f32_to_bf16(1.0f / (1.0f + expf(-gv)));
    return f32_to_bf16(bf16_to_f32(ab) * bf16_to_f32(xb));
}
__device__ __forceinline__ float halo_gq_clamp(float v) { return fminf(fmaxf(v, -448.0f), 448.0f); }

// The plain decode kernel, and the chained one (halo_chain.h) over the same body.
template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT,
         int WIN = 0>
__global__ __launch_bounds__(NWARPS * halo_dec_ds(HEAD_DIM) * 32)
void halo_attn_decode_kernel(const HaloArgs a, int splits)
{
    halo_attn_decode_body<NWARPS, TILE, HEAD_DIM, GQA, BS, KVP, OPT, WIN>(a, splits);
}
template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT,
         int WIN = 0>
__global__ __launch_bounds__(NWARPS * halo_dec_ds(HEAD_DIM) * 32)
void halo_r4dchain_attn_decode(const HaloArgs a, int splits)
{
    halo_chain_wait();
    halo_attn_decode_body<NWARPS, TILE, HEAD_DIM, GQA, BS, KVP, OPT, WIN>(a, splits);
    halo_chain_done();
}

// Vectorised merge, used by the h256 launcher whenever the split count fits a wave. Bit-identical
// to splitkv_combine_body below; only the thread map differs. One wave owns one output row and each
// lane HEAD_DIM/32 dims, so a segment is one 16-byte load per lane; a block carries JPB rows. No
// barrier and no LDS, so rows in one block may have different contexts and segment counts.
template<int HEAD_DIM, int CHAINS, int PF16, int JPB, int WIN, int GQ>
__device__ __forceinline__
void splitkv_combine_vec_body(const HaloArgs& a, int splits, int tile, const HaloGateQuant& gq)
{
    constexpr int DPL = HEAD_DIM / 32;                 // dims a lane owns
    const int lane = threadIdx.x & 31;
    const int job  = blockIdx.x * JPB + (int)(threadIdx.x >> 5);
    if (job >= halo_q_total(a) * a.q_heads) return;
    const int tok = job / a.q_heads;
    const int seq = halo_q_seq(a.cu_q, a.num_seqs, a.q_len, tok);
    const int ctx = a.seqused_k[seq];
    uint16_t* outp = (uint16_t*)a.out + (size_t)job * HEAD_DIM + lane * DPL;

    /* Plain store, or gated + quantised. A lane's DPL columns lie in one 128-column group of
     * 128 / DPL lanes, so the group absmax is a butterfly over those lanes only. */
    auto finish = [&](const uint16_t (&ob)[DPL]) {
        if constexpr (!GQ) {
            union { uint4 u4; uint2 u2; uint16_t w[DPL]; } o_;
#pragma unroll
            for (int j = 0; j < DPL; ++j) o_.w[j] = ob[j];
            if (DPL == 8) *(uint4*)outp = o_.u4; else *(uint2*)outp = o_.u2;
        } else {
            constexpr int LPG = kHaloGqGroup / DPL;     // lanes a scale group
            static_assert(DPL == 8 || DPL == 4, "a lane's columns are one or two dwords of codes");
            const uint16_t* gp = gq.g + (size_t)job * gq.g_ld + lane * DPL;
            union { uint4 u4; uint2 u2; uint16_t w[DPL]; } gi, o_;
            if (DPL == 8) gi.u4 = *(const uint4*)gp; else gi.u2 = *(const uint2*)gp;
            float v[DPL];
            float m = 0.0f;
#pragma unroll
            for (int j = 0; j < DPL; ++j) {
                o_.w[j] = halo_gq_gate(gi.w[j], ob[j]);
                v[j] = bf16_to_f32(o_.w[j]);
                m = fmaxf(m, fabsf(v[j]));
            }
            if (DPL == 8) *(uint4*)outp = o_.u4; else *(uint2*)outp = o_.u2;
            if constexpr (LPG > 16) m = fmaxf(m, halo_xor_lane<16>(m));
            if constexpr (LPG > 8)  m = fmaxf(m, halo_xor_lane<8>(m));
            m = fmaxf(m, halo_xor_lane<4>(m));
            m = fmaxf(m, halo_xor_lane<2>(m));
            m = fmaxf(m, halo_xor_lane<1>(m));
            const float sc  = m > 0.0f ? m * (1.0f / q8_top<GQ == 2>()) : 1.0f;
            const float inv = 1.0f / sc;
            if ((lane % LPG) == 0) gq.s[(size_t)job * gq.s_ld + (lane * DPL) / kHaloGqGroup] = sc;
            uint32_t pk[DPL / 4];
#pragma unroll
            for (int w = 0; w < DPL / 4; ++w)
                pk[w] = q8_pack4<GQ == 2>(v[4 * w + 0], v[4 * w + 1], v[4 * w + 2], v[4 * w + 3],
                                          inv);
            uint8_t* qp = gq.q + (size_t)job * gq.q_ld + lane * DPL;
            if (DPL == 8) *(uint2*)qp = make_uint2(pk[0], pk[DPL / 4 - 1]);
            else          *(uint32_t*)qp = pk[0];
        }
    };

    if (ctx <= 0) {                                    // padded slot: zeros, read nothing
        uint16_t ob[DPL];
#pragma unroll
        for (int i = 0; i < DPL; ++i) ob[i] = 0;
        finish(ob);
        return;
    }
    const int nef = halo_attn_tiles(ctx, tile)
                  - (WIN ? halo_attn_wlo(ctx, halo_q_len(a.cu_q, seq, a.q_len), 0, a.window) / tile : 0);
    const int tp_ = (nef + splits - 1) / splits;
    const int u   = (nef + tp_ - 1) / tp_;
    const size_t nrow = (size_t)halo_q_total(a) * a.q_heads * splits;
    const size_t base = (size_t)job * splits;

    const float* sm; const float* sl;
    if (PF16) { const uint16_t* so = (const uint16_t*)a.scratch;
                sm = (const float*)(so + nrow * HEAD_DIM); sl = sm + nrow; }
    else      { const float* so = (const float*)a.scratch;
                sm = so + nrow * HEAD_DIM; sl = sm + nrow; }

    // max is order-independent and wave-uniform, so it is a scalar walk. The sum reproduces the
    // LDS tree's association (stride-32 pre-accumulation, then xor butterfly 16..1) so results
    // match the non-vector merge bit for bit.
    float gmax = -INFINITY;
    for (int s = 0; s < u; ++s) gmax = fmaxf(gmax, sm[base + s]);
    float tot = 0.0f;
    for (int s = lane; s < u; s += 32) {
        const float mm = sm[base + s];
        const float w  = (mm <= -3.0e38f) ? 0.0f : ex2(mm - gmax);
        tot += w * sl[base + s];
    }
    tot = halo_wave_sum(tot);

    float ch[CHAINS][DPL];
#pragma unroll
    for (int i = 0; i < CHAINS; ++i)
#pragma unroll
        for (int j = 0; j < DPL; ++j) ch[i][j] = 0.0f;

    auto wgt = [&](int sx) {
        const float mm = sm[base + sx];
        const float w  = (mm <= -3.0e38f) ? 0.0f : ex2(mm - gmax);
        return PF16 ? w * sl[base + sx] : w;
    };

    int s = 0;
    if (PF16) {
        const uint16_t* op = (const uint16_t*)a.scratch + base * HEAD_DIM + lane * DPL;
        // One aligned access per lane per segment: 16 bytes at head_dim 256, 8 at 128.
        union { uint4 u4; uint2 u2; __fp16 h[DPL]; } r;
        auto rd = [&](int sx) {
            if (DPL == 8) r.u4 = *(const uint4*)(op + (size_t)sx * HEAD_DIM);
            else          r.u2 = *(const uint2*)(op + (size_t)sx * HEAD_DIM);
        };
        for (; s + CHAINS <= u; s += CHAINS) {
#pragma unroll
            for (int i = 0; i < CHAINS; ++i) {
                rd(s + i);
                const float w = wgt(s + i);
#pragma unroll
                for (int j = 0; j < DPL; ++j) ch[i][j] = fmaf(w, (float)r.h[j], ch[i][j]);
            }
        }
        for (; s < u; ++s) {
            rd(s);
            const float w = wgt(s);
#pragma unroll
            for (int j = 0; j < DPL; ++j) ch[0][j] = fmaf(w, (float)r.h[j], ch[0][j]);
        }
    } else {
        const float* op = (const float*)a.scratch + base * HEAD_DIM + lane * DPL;
        for (; s + CHAINS <= u; s += CHAINS) {
#pragma unroll
            for (int i = 0; i < CHAINS; ++i) {
                const float* p = op + (size_t)(s + i) * HEAD_DIM;
                const float w = wgt(s + i);
#pragma unroll
                for (int j = 0; j < DPL; ++j) ch[i][j] = fmaf(w, p[j], ch[i][j]);
            }
        }
        for (; s < u; ++s) {
            const float* p = op + (size_t)s * HEAD_DIM;
            const float w = wgt(s);
#pragma unroll
            for (int j = 0; j < DPL; ++j) ch[0][j] = fmaf(w, p[j], ch[0][j]);
        }
    }

    const int kvh = (job % a.q_heads) / (a.q_heads / a.kv_heads);   /* the query group */
    const int krep = a.kv_rep > 1 ? a.kv_rep : 1;                    /* its cache head: kvh / krep */
    const float vd = a.v_descale ? a.v_descale[seq * (a.kv_heads / krep) + kvh / krep] : 1.0f;
    uint16_t ob[DPL];
#pragma unroll
    for (int j = 0; j < DPL; ++j) {
        float av = 0.0f;
#pragma unroll
        for (int i = 0; i < CHAINS; ++i) av += ch[i][j];
        // (av * vd) / tot in this order; folding vd/tot first rounds differently (one bf16 ULP).
        ob[j] = f32_to_bf16(tot > 0.0f ? av * vd / tot : 0.0f);
    }
    finish(ob);
}

template<int HEAD_DIM, int CHAINS, int PF16, int JPB, int WIN = 0>
__global__ __launch_bounds__(32 * JPB)
void halo_attn_splitkv_combine_vec_kernel(const HaloArgs a, int splits, int tile)
{
    splitkv_combine_vec_body<HEAD_DIM, CHAINS, PF16, JPB, WIN, 0>(a, splits, tile,
                                                                  HaloGateQuant{});
}
template<int HEAD_DIM, int CHAINS, int PF16, int JPB, int WIN = 0>
__global__ __launch_bounds__(32 * JPB)
void halo_r4dchain_attn_splitkv_combine_vec(const HaloArgs a, int splits, int tile)
{
    halo_chain_wait();
    splitkv_combine_vec_body<HEAD_DIM, CHAINS, PF16, JPB, WIN, 0>(a, splits, tile,
                                                                  HaloGateQuant{});
    halo_chain_done();
}
template<int HEAD_DIM, int CHAINS, int PF16, int JPB, int WIN = 0, int GQ = 1>
__global__ __launch_bounds__(32 * JPB)
void halo_attn_splitkv_combine_vec_gq_kernel(const HaloArgs a, int splits, int tile,
                                             const HaloGateQuant gq)
{
    splitkv_combine_vec_body<HEAD_DIM, CHAINS, PF16, JPB, WIN, GQ>(a, splits, tile, gq);
}

// Merge. PF16=1 reads normalised f16 partials and folds l into the weight; PF16=0 reads
// unnormalised f32 partials with l summed separately. Both walk segments in fixed index order so
// rounding never depends on workgroup scheduling.
template<int HEAD_DIM, int CHAINS, int PF16, int WIN, int GQ>
__device__ __forceinline__
void splitkv_combine_body(const HaloArgs& a, int splits, int tile, const HaloGateQuant& gq)
{
    HALO_DYN_LDS(float, sw);
    __shared__ float red[HEAD_DIM];
    const int job = blockIdx.x, d = threadIdx.x;
    const int tok = job / a.q_heads;
    const int seq = halo_q_seq(a.cu_q, a.num_seqs, a.q_len, tok);
    const int ctx = a.seqused_k[seq];

    /* One thread per column, so a 128-column scale group is four waves, combined through `red`.
     * Reusing `red` is safe: its last read (the total) is behind a barrier before the merge loop,
     * and a padded slot never used it. */
    auto finish = [&](uint16_t ob) {
        uint16_t* op = (uint16_t*)a.out + (size_t)job * HEAD_DIM + d;
        if constexpr (!GQ) {
            *op = ob;
        } else {
            constexpr int WPG = kHaloGqGroup / 32;         // waves a scale group
            const uint16_t h = halo_gq_gate(gq.g[(size_t)job * gq.g_ld + d], ob);
            *op = h;
            const float v = bf16_to_f32(h);
            const float m = halo_wave_max(fabsf(v));
            if ((d & 31) == 0) red[d / 32] = m;
            __syncthreads();
            const int g0 = (d / kHaloGqGroup) * WPG;
            float gm = red[g0];
#pragma unroll
            for (int w = 1; w < WPG; ++w) gm = fmaxf(gm, red[g0 + w]);
            const float sc  = gm > 0.0f ? gm * (1.0f / q8_top<GQ == 2>()) : 1.0f;
            const float inv = 1.0f / sc;
            if ((d % kHaloGqGroup) == 0) gq.s[(size_t)job * gq.s_ld + d / kHaloGqGroup] = sc;
            if constexpr (GQ == 2) {
                gq.q[(size_t)job * gq.q_ld + d] =
                    (uint8_t)(int)rintf(fminf(fmaxf(v * inv, -127.0f), 127.0f));
            } else {
                // gfx1151 has no v_cvt_pk_fp8_f32; halo_fp8_wmma.h's twin has its shape.
                const int p = halo_fp8::cvt_pk_fp8_f32(halo_gq_clamp(v * inv), 0.0f, 0, false);
                gq.q[(size_t)job * gq.q_ld + d] = (uint8_t)(p & 0xff);
            }
        }
    };

    // A padded slot (ctx 0) would divide by zero below; write zeros and stop. ctx is block-uniform,
    // so the barriers below stay matched.
    if (ctx <= 0) { finish(0); return; }
    // Segment count, derived exactly as in the decode kernel, including the window's tile floor;
    // otherwise the merge would read partials no segment wrote.
    const int nef = halo_attn_tiles(ctx, tile)
                  - (WIN ? halo_attn_wlo(ctx, halo_q_len(a.cu_q, seq, a.q_len), 0, a.window) / tile : 0);
    const int tp_ = (nef + splits - 1) / splits;
    const int u   = (nef + tp_ - 1) / tp_;
    const size_t nrow = (size_t)halo_q_total(a) * a.q_heads * splits;
    const size_t base = (size_t)job * splits;

    const float* sm; const float* sl;
    if (PF16) { const uint16_t* so = (const uint16_t*)a.scratch;
                sm = (const float*)(so + nrow * HEAD_DIM); sl = sm + nrow; }
    else      { const float* so = (const float*)a.scratch;
                sm = so + nrow * HEAD_DIM; sl = sm + nrow; }

    float gmax = -INFINITY;
    for (int s = d; s < u; s += HEAD_DIM) gmax = fmaxf(gmax, sm[base + s]);
    red[d] = gmax; __syncthreads();
    for (int off = HEAD_DIM / 2; off; off >>= 1) {
        if (d < off) red[d] = fmaxf(red[d], red[d + off]);
        __syncthreads();
    }
    gmax = red[0]; __syncthreads();

    float tot = 0.0f;
    for (int s = d; s < u; s += HEAD_DIM) {
        const float mm = sm[base + s];
        float w = (mm <= -3.0e38f) ? 0.0f : ex2(mm - gmax);
        if (PF16) { w *= sl[base + s]; sw[s] = w; tot += w; }
        else      { sw[s] = w;         tot += w * sl[base + s]; }
    }
    red[d] = tot; __syncthreads();
    for (int off = HEAD_DIM / 2; off; off >>= 1) {
        if (d < off) red[d] += red[d + off];
        __syncthreads();
    }
    tot = red[0]; __syncthreads();

    float ch[CHAINS];
#pragma unroll
    for (int i = 0; i < CHAINS; ++i) ch[i] = 0.0f;
    int s = 0;
    if (PF16) {
        const uint16_t* op = (const uint16_t*)a.scratch + base * HEAD_DIM + d;
        for (; s + CHAINS <= u; s += CHAINS) {
#pragma unroll
            for (int i = 0; i < CHAINS; ++i)
                ch[i] = fmaf(sw[s + i], (float)__builtin_bit_cast(
                    __fp16, op[(size_t)(s + i) * HEAD_DIM]), ch[i]);
        }
        for (; s < u; ++s)
            ch[0] = fmaf(sw[s], (float)__builtin_bit_cast(__fp16, op[(size_t)s * HEAD_DIM]), ch[0]);
    } else {
        const float* op = (const float*)a.scratch + base * HEAD_DIM + d;
        for (; s + CHAINS <= u; s += CHAINS) {
#pragma unroll
            for (int i = 0; i < CHAINS; ++i)
                ch[i] = fmaf(sw[s + i], op[(size_t)(s + i) * HEAD_DIM], ch[i]);
        }
        for (; s < u; ++s) ch[0] = fmaf(sw[s], op[(size_t)s * HEAD_DIM], ch[0]);
    }
    float av = 0.0f;
#pragma unroll
    for (int i = 0; i < CHAINS; ++i) av += ch[i];

    const int kvh = (job % a.q_heads) / (a.q_heads / a.kv_heads);   /* the query group */
    const int krep = a.kv_rep > 1 ? a.kv_rep : 1;                    /* its cache head: kvh / krep */
    const float vd = a.v_descale ? a.v_descale[seq * (a.kv_heads / krep) + kvh / krep] : 1.0f;
    finish(f32_to_bf16(tot > 0.0f ? av * vd / tot : 0.0f));
}

template<int HEAD_DIM, int CHAINS, int PF16, int WIN = 0>
__global__ __launch_bounds__(HEAD_DIM)
void halo_attn_splitkv_combine_kernel(const HaloArgs a, int splits, int tile)
{
    splitkv_combine_body<HEAD_DIM, CHAINS, PF16, WIN, 0>(a, splits, tile, HaloGateQuant{});
}
template<int HEAD_DIM, int CHAINS, int PF16, int WIN = 0>
__global__ __launch_bounds__(HEAD_DIM)
void halo_r4dchain_attn_splitkv_combine(const HaloArgs a, int splits, int tile)
{
    halo_chain_wait();
    splitkv_combine_body<HEAD_DIM, CHAINS, PF16, WIN, 0>(a, splits, tile, HaloGateQuant{});
    halo_chain_done();
}
template<int HEAD_DIM, int CHAINS, int PF16, int WIN = 0, int GQ = 1>
__global__ __launch_bounds__(HEAD_DIM)
void halo_attn_splitkv_combine_gq_kernel(const HaloArgs a, int splits, int tile,
                                         const HaloGateQuant gq)
{
    splitkv_combine_body<HEAD_DIM, CHAINS, PF16, WIN, GQ>(a, splits, tile, gq);
}

// The gated merges are instantiated by the bs4 and gqa12 units that include this body rather than by
// this unit (no BS-16 h256/gqa6 gq row exists); compile-check them here so a change to the body
// cannot break those units silently. Taking a kernel template's address instantiates it.
__attribute__((used)) static void* const kHaloGqInst[] = {
    (void*)&halo_attn_splitkv_combine_vec_gq_kernel<256, 4, 1, 2, 0, 1>,
    (void*)&halo_attn_splitkv_combine_vec_gq_kernel<256, 4, 1, 2, 0, 2>,
    (void*)&halo_attn_splitkv_combine_gq_kernel<256, 4, 1, 0, 1>,
    (void*)&halo_attn_splitkv_combine_gq_kernel<256, 4, 1, 0, 2>,
};

// ---- host helpers shared by the paged decode dispatches -----------------------------------------

// Split law of the full-attention legs (h256/gqa6 and the gqa8 family): as many workgroups as fit
// under 128 (grid = splits * kv_heads * num_seqs), at least two tiles per segment. It depends on
// gqa only through the row count, and on head_dim only through split-D's wave count.
//
// floor(128 / units), not a power of two: a power of two leaves 96-112 workgroups at 3, 5, 6 and 7
// requests, and floor is 2-20% faster there. Staying under a multiple of 64 matters (at five
// requests, 13 splits = 130 workgroups is three rounds of 64 and slower than 12).
//
// Fewer splits mainly save partial-buffer traffic (seqs x q_len x q_heads x splits rows of head_dim
// f16, written once and read once). The target is 256 workgroups where a workgroup has one or two
// waves (q_len * gqa <= 32), 128 otherwise: a one- or two-wave workgroup keeps too few KV bytes in
// flight to load the card's DRAM at 128. Under split-D (head 256) the same band is a four-wave
// workgroup (the launch takes the two-tile kernel for one tile), five of which fit a WGP's LDS:
// 192 is about two rounds of them: against 256, within 2% at 16K context and 5-26% faster at 2K.
static inline int halo_decode_splits(const HaloArgs* a, int tile) {
    if (a->splits > 0) return a->splits;
    const int tiles = (a->max_ctx + tile - 1) / tile;
    const int units = std::max(1, a->kv_heads * a->num_seqs);
    const int gqa = a->kv_heads > 0 ? std::max(1, a->q_heads / a->kv_heads) : 1;
    int s = (a->q_len * gqa > 32 ? 128 : halo_dec_ds(a->head_dim) > 1 ? 192 : 256) / units;
    const int cap = tiles / 2;
    if (s > cap) s = cap;
    if (s > tiles) s = tiles;
    if (s < 1) s = 1;
    return s;
}

// Split law of the windowed legs (the DFlash2 drafter at h128/gqa4, Gemma 4 at h256/gqa2): target
// 192 workgroups, round down to a power of two (min 16), keep at least two tiles per segment, then
// step up off splits * units == 256, a slow spot for these geometries. The tile count comes from
// the window when one is set, since that is the span the kernel walks.
[[maybe_unused]] static inline int halo_decode_splits_win(const HaloArgs* a, int tile) {
    if (a->splits > 0) return a->splits;
    const int span  = (a->window > 0 && a->window + a->q_len < a->max_ctx)
                    ? a->window + a->q_len : a->max_ctx;
    const int tiles = (span + tile - 1) / tile;
    const int units = std::max(1, a->kv_heads * a->num_seqs);
    const int raw   = 192 / units;
    int s = 128;
    while (s > 16 && s > raw) s >>= 1;
    const int cap = tiles / 2;
    while (s > 16 && s > cap) s >>= 1;
    if (s > tiles) s = tiles;
    if (s < 1) s = 1;
    if (s * units == 256 && s * 2 <= tiles) s <<= 1;
    return s;
}

// Bytes of split-KV partials one decode launch needs, the same for both cache dtypes: f16 O/l plus
// m and l under O_PF16, else f32 O plus m and l.
static inline long halo_decode_scratch(const HaloArgs* a, int splits, int opt) {
    const long rows = (long)halo_q_total(*a) * a->q_heads;
    return (opt & O_PF16) ? rows * splits * (long)(a->head_dim * 2 + 8)
                          : rows * splits * (long)(a->head_dim + 2) * 4L;
}

// ---- the launch ---------------------------------------------------------------------------------
// Everything below this line is shared host code: the option set and merge constants of the paged
// decode, one launch helper and one scratch helper per geometry. This file is a BODY -- a wrapper
// unit `#include`s it as text and is what HALO_UNITS compiles (PORTING.md §3) -- so it defines no
// `extern "C"` entry point of its own: a geometry's entries (`<geom>_*kv`, `*_scratch_bytes`, its
// plugin rows) live in the paged unit that owns that geometry, because two units may not define the
// same symbol.
//
// Pooled earlier geometry is a hard gate: anything else is rejected. Decode prefetches two K
// fragments per WMMA group, not four: four costs 8 more VGPRs at the peak of the QK loop and spills
// 18, reloading part of the query from scratch on every tile.
#define HALO_D_TILE 16
// The h256/gqa6 option set; a geometry whose registers differ says so in its own unit. F16 16-bit
// operands, mask skip, folded query scale, K read straight from the paged cache, f16 partials,
// contiguous-k fragments, prefetch 2 + scheduling barrier, dot2 row sum, block table in SGPRs, V
// prefetched across the m-tile loop, LDS-scoped barrier.
#define HALO_D_OPT  3429947
// Output rows (waves) per block in the vectorised merge. 1, 2 and 4 are within 1%; 2 is best on
// average.
#define HALO_D_JPB 2
// Minimum output rows for the vectorised merge. It gives a row 32 lanes instead of 256, trading
// parallelism for 8x fewer loads, which pays only when there are enough rows to fill the GPU; the
// crossover is ~192 rows.
#define HALO_D_VECMIN 256

/* Bytes of the split-KV partials one launch of this option set needs, from the split law the
 * launcher will use: the windowed law under WIN (its tile count comes from the span the kernel
 * walks, not from max_ctx), the full-attention one otherwise. The cache's row bytes and the
 * merge's PF16 enter through the option set, head_dim through the args. */
template <int OPT, int WIN = 0>
static inline long decode_scratch_bytes(const HaloArgs* a) {
    return halo_decode_scratch(a, WIN ? halo_decode_splits_win(a, HALO_D_TILE)
                                      : halo_decode_splits(a, HALO_D_TILE), OPT);
}

/* Split-KV partial buffer size at declare time, for a plugin row's scratch hook.
 *
 * Declare passes parameters and no tensors (the batch is not known yet), so parameters are the
 * primary source and tensors the fallback. Reading tensors first would refuse every attn_paged row
 * with "invalid argument". q_len is the band's upper bound, since the region is sized once for the
 * worst step.
 *
 * The size is the maximum over every sequence count up to max_seqs: S * splits(S) is not monotone
 * in S (the law is a closed form, so the sweep is cheap).
 *
 * `row`: a token's bytes in a (block, head) page (fill_paged_row), 0 for 2 * head_dim; `rep`: the
 * launch maps each cache head onto `rep` query groups (HaloArgs::kv_rep), so the split law counts
 * groups, `rep` of them a cache head; `q_pitched`: q may be a slice of a wider row
 * (attn_paged_qrow). */
template <int HEAD_DIM, int GQA, int BS, int OPT, int WIN = 0>
static int64_t decode_scratch(const RadArgs* a, long long row = 0, int rep = 1,
                              bool q_pitched = false) {
    const int sp = split_of(a);
    if (sp < 0) return HALO_NO(RAD_E_INVAL);

    HaloArgs r;
    const long long n_head   = halo_geti_or(a, "n_head", 0);
    const long long max_seqs = halo_geti_or(a, "max_seqs", 0);
    if (n_head > 0 && max_seqs > 0) {
        std::memset(&r, 0, sizeof(r));
        r.head_dim  = HEAD_DIM;
        r.q_heads   = (int)n_head;
        const long long gqa = halo_geti_or(a, "gqa", GQA);
        r.kv_heads  = (int)(gqa > 0 ? n_head / gqa : n_head) * rep;
        r.kv_rep    = rep;
        r.num_seqs  = (int)max_seqs;
        r.q_len     = (int)halo_geti_or(a, "q_len", 1);
        r.max_ctx   = (int)halo_geti_or(a, "max_ctx", 0);
        r.block_size = (int)halo_geti_or(a, "block_size", BS);
        /* The windowed law's span, so a windowed geometry is not sized against the full one. */
        r.window    = (int)halo_geti_or(a, "window", 0);
        r.splits    = sp;
        /* max_ctx feeds only the split law. Zero would pick one split and undersize the buffer for
         * a launch that later picks more, so refuse. */
        if (r.max_ctx <= 0) return HALO_NO(RAD_E_INVAL);
        int64_t worst = 0;
        for (long long n = 1; n <= max_seqs; ++n) {
            r.num_seqs = (int)n;
            const int64_t need = (int64_t)decode_scratch_bytes<OPT, WIN>(&r);
            if (need > worst) worst = need;
        }
        return worst;
    }

    int rc = fill_paged_row(a, &r, HEAD_DIM, row > 0 ? row : 2LL * HEAD_DIM, q_pitched);
    if (rc != RAD_OK) return rc;
    r.kv_heads *= rep;
    r.kv_rep = rep;
    r.splits = sp;
    return (int64_t)decode_scratch_bytes<OPT, WIN>(&r);
}

/* Operands plus the scratch check, shared by every decode plugin wrapper. `rep` and `q_pitched` are
 * the two views the decode rows add to attn_paged's eight operands (HaloArgs::kv_rep, q_ld). */
template <int HEAD_DIM, int GQA, int BS, int OPT, int WIN = 0>
static int decode_common(const RadArgs* a, HaloArgs* r, long long row = 0, int rep = 1,
                         bool q_pitched = false) {
    const int sp = split_of(a);
    if (sp < 0) return HALO_NO(RAD_E_INVAL);
    int rc = fill_paged_row(a, r, HEAD_DIM, row > 0 ? row : 2LL * HEAD_DIM, q_pitched);
    if (rc != RAD_OK) return rc;
    /* A QLD instance reads q_ld whatever q is: dense is the pitch q_heads * head_dim. */
    if (q_pitched && r->q_ld == 0) r->q_ld = (long)r->q_heads * HEAD_DIM;
    /* Each cache head serves `rep` query groups: the kernel's grid and its gqa test count groups. */
    r->kv_heads *= rep;
    r->kv_rep = rep;
    r->splits = sp;
    const int64_t need = (int64_t)decode_scratch_bytes<OPT, WIN>(r);
    if (!a->scratch || a->scratch_bytes < need) return HALO_NO(RAD_E_SCRATCH);
    return RAD_OK;
}

// The launch. One helper for every split-KV decode geometry this body serves: head_dim, queries per
// KV head, KV page size, the option set and the window bit are the template's, and the geometry's
// unit instantiates it. The three geometry parameters are a hard gate (a mismatch is a selection
// bug and returns the code that names which); WIN picks the split law and compiles the window walk
// in; KVP picks the cache reader (0 = E4M3 codes with per-(seq,head) descales, 1 = bf16).
//
// A non-null `gq` adds the output gate and the fp8/int8 quantiser to the merge's epilogue
// (attn_paged_gate_quant) instead of a second dispatch; a gated row has no chained form, so it
// always takes the plain launch, as libr4d's does.
//
// Returns 0, a negative code for a shape this instantiation does not serve, or -100 - hipError_t.
template <int HEAD_DIM, int GQA, int BS, int OPT, int KVP, int WIN = 0>
static int decode_launch(const HaloArgs* a, hipStream_t s, const HaloGateQuant* gq = nullptr) {
    constexpr int TILE = HALO_D_TILE, JPB = HALO_D_JPB, VECMIN = HALO_D_VECMIN;
    constexpr int PF16 = (OPT & O_PF16) ? 1 : 0;
    if (a->head_dim != HEAD_DIM || a->block_size != BS) return -1;
    if (a->q_heads / a->kv_heads != GQA) return -2;
    const int rows = a->q_len * GQA;
    if (rows > 64) return -4;                   // verify width 64 / gqa, i.e. q_len <= 64 / GQA
    const int nw = (rows + 15) / 16;            // gqa-aligned height (split-D rounds it below)
    const int splits = WIN ? halo_decode_splits_win(a, TILE) : halo_decode_splits(a, TILE);

#define HALO_D_LAUNCH(NW)                                                                           \
    if (a->chain)                                                                                   \
        hipLaunchKernelGGL((halo_r4dchain_attn_decode<NW, TILE, HEAD_DIM, GQA, BS, KVP, OPT, WIN>),     \
                           dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * halo_dec_ds(HEAD_DIM) * 32), 0, s, *a, splits); \
    else                                                                                            \
        hipLaunchKernelGGL((halo_attn_decode_kernel<NW, TILE, HEAD_DIM, GQA, BS, KVP, OPT, WIN>),   \
                           dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * halo_dec_ds(HEAD_DIM) * 32), 0, s, *a, splits)
    // Split-D (head 256) launches only the two- and four-tile kernels: one row tile takes two, three
    // take four, the extra tile's rows clamped to the last query and never stored. A one-tile
    // workgroup is two waves staging a whole K and V tile, eight uint4 a thread, too many for the V
    // prefetch: 3-5% slower at q_len 1-2 and 16K context, though 2-14% faster at 2K, where the
    // dead tile's WMMAs are not hidden behind the KV stream. A three-tile one is six waves, which
    // fit a WGP worse than eight: 6-7% slower at q_len 5-8 and 16K, 2% at 2K.
    constexpr int NW1 = halo_dec_ds(HEAD_DIM) > 1 ? 2 : 1, NW3 = halo_dec_ds(HEAD_DIM) > 1 ? 4 : 3;
    if      (nw == 1) { HALO_D_LAUNCH(NW1); }
    else if (nw == 2) { HALO_D_LAUNCH(2); }
    else if (nw == 3) { HALO_D_LAUNCH(NW3); }
    else              { HALO_D_LAUNCH(4); }
#undef HALO_D_LAUNCH
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -100 - (int)e;

    // The merge. Both forms produce identical bytes; the vectorised one wins only when there are
    // enough rows to fill the GPU and the split count fits a wave -- its wave sum reproduces the LDS
    // tree's association only while splits <= 64. The gated form carries the gate and the quantiser
    // in the merge's epilogue; GQ 2 is the int8 quantiser and GQ 1 E4M3.
    const int jobs3  = halo_q_total(*a) * a->q_heads;
    const bool vec   = splits <= 64 && jobs3 >= VECMIN;
    const int blocks = (jobs3 + JPB - 1) / JPB;
    if (gq) {
        if (vec && gq->i8)
            hipLaunchKernelGGL((halo_attn_splitkv_combine_vec_gq_kernel<HEAD_DIM, 4, PF16, JPB, WIN, 2>),
                               dim3(blocks), dim3(32 * JPB), 0, s, *a, splits, TILE, *gq);
        else if (vec)
            hipLaunchKernelGGL((halo_attn_splitkv_combine_vec_gq_kernel<HEAD_DIM, 4, PF16, JPB, WIN, 1>),
                               dim3(blocks), dim3(32 * JPB), 0, s, *a, splits, TILE, *gq);
        else if (gq->i8)
            hipLaunchKernelGGL((halo_attn_splitkv_combine_gq_kernel<HEAD_DIM, 4, PF16, WIN, 2>),
                               dim3(jobs3), dim3(HEAD_DIM), splits * sizeof(float), s, *a,
                               splits, TILE, *gq);
        else
            hipLaunchKernelGGL((halo_attn_splitkv_combine_gq_kernel<HEAD_DIM, 4, PF16, WIN, 1>),
                               dim3(jobs3), dim3(HEAD_DIM), splits * sizeof(float), s, *a,
                               splits, TILE, *gq);
    } else if (vec && a->chain) {
        hipLaunchKernelGGL((halo_r4dchain_attn_splitkv_combine_vec<HEAD_DIM, 4, PF16, JPB, WIN>),
                           dim3(blocks), dim3(32 * JPB), 0, s, *a, splits, TILE);
    } else if (vec) {
        hipLaunchKernelGGL((halo_attn_splitkv_combine_vec_kernel<HEAD_DIM, 4, PF16, JPB, WIN>),
                           dim3(blocks), dim3(32 * JPB), 0, s, *a, splits, TILE);
    } else if (a->chain) {
        hipLaunchKernelGGL((halo_r4dchain_attn_splitkv_combine<HEAD_DIM, 4, PF16, WIN>), dim3(jobs3),
                           dim3(HEAD_DIM), splits * sizeof(float), s, *a, splits, TILE);
    } else {
        hipLaunchKernelGGL((halo_attn_splitkv_combine_kernel<HEAD_DIM, 4, PF16, WIN>), dim3(jobs3),
                           dim3(HEAD_DIM), splits * sizeof(float), s, *a, splits, TILE);
    }
    e = hipGetLastError();
    return e == hipSuccess ? 0 : -100 - (int)e;
}
