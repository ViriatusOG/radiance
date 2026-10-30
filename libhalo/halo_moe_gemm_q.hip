/* The grouped quantised GEMM is libr4d/r4d_moe.hip's moe_gemm_q_kernel (there at gfx1201, where
 * both operands stage into LDS as E4M3 bytes and one v_wmma_f32_16x16x16_fp8_fp8 consumes them).
 * gfx1151 has no fp8 matrix instruction (see halo_fp8_wmma.h), so this port dequantises both
 * operands to f16 on the way into LDS and runs halo_wmma.h's v16hf WMMA instead. Everything the
 * numerics depend on -- the code tables and their nibble permutes, the scale grids and their fold,
 * the run walk, the dropped-slot zeroing, the launcher's forms -- is libr4d's, unchanged.
 *
 *   halo_rad_moe_gemm_q         W4 0, two classes      (block E4M3 weight, bf16 scale per 128x128)
 *   halo_rad_moe_gemm_q_mod4    W4 0, four classes
 *   halo_rad_moe_gemm_w4a8      W4 1-4 by `dtype`, two classes
 *   halo_rad_moe_gemm_w4a8_mod4 W4 1-4 by `dtype`, four classes
 *   halo_rad_moe_gemm_{q,w4a8}_scratch   the run map's bytes outside the decode band, else 0
 *
 * ---- the launcher's forms (libr4d's, and the pins the selftest drives them with) ---------
 *
 *   rule (pin 0)   the (tile, run) decode grid inside r4d_moe_decode_band, the run-aligned
 *                  prefill form outside it
 *   pin 1          the run-aligned prefill form: moe_blockmap_kernel builds the run map in the
 *                  caller's scratch, then the GEMM names its block by the map's entry, so one
 *                  block serves one run's rows and the grid is (map capacity, column tiles)
 *   pin 2          the (tile, run) decode grid: 16-row tiles, one run a block
 *   pin 3          the uniform prefill grid: 32-row tiles, a binary search per block, and each
 *                  wave skipping the M tiles its run does not reach
 *   pin_fk         the decode form's staged K: one 128-wide block a pass, or two (bit-identical,
 *                  W4 2-4 only, whose fold already sums scale groups)
 *   pin_down_tile  the aligned form's column tile on the down projection: the 256-column tile where
 *                  the shape is the sorted operand with one part and every class whole 256-column
 *                  tiles (block fp8: every 128-column group of the width in turn, every class whole
 *                  128-column groups), refused where it is not; 1 never takes it, 2 always (or
 *                  refuses)
 *   pin_down_dec   the sorted operand's decode form: moe_down_dec_kernel with n 64-column tiles a
 *                  block, -1 the (tile, run) form
 *
 * ---- what the fragment change costs ------------------------------------------------
 *
 * halo_wmma.h's note is the whole story: on gfx1151 a lane holds a fragment's WHOLE sixteen-wide k
 * run, in one vt16bf, and lanes 16-31 replicate it. So a fragment read here is one 32-byte read per
 * lane at (row0 + (lane & 15), ks) -- libr4d's `8*(lane>>4)` half term disappears -- and the
 * accumulator element e is output row `2*e + (lane>>4)`, column `lane & 15` (libr4d: row
 * `8*(lane>>4) + e`, same column). A staged row is therefore FK uint16 (256 B at FK 128) rather
 * than FK bytes, a thread stages the SAME 32 K elements as before (32 E4M3 bytes for the fp8
 * plane, 16 packed bytes for the four-bit ones, plus the five-bit form's sign dword), and produces
 * 64 B of bf16.
 *
 * ---- LDS pitch ---------------------------------------------------------------------
 *
 * libr4d's lds_pitch<FK>() = FK + 8 (bytes) separates sixteen 1-byte rows on 4-byte banks for its
 * 8-byte-per-lane fragment read. Here a lane reads 32 B and the rows are 2-byte elements apart, so
 * the pitch is kFK*2 + 16 = 272 B (136 uint16): 16-byte aligned (a ds_read_b128 needs 16, not 32)
 * and 68 dwords, so row r starts at bank 4r % 32 -- eight distinct starts, two rows each, for a
 * two-way conflict per phase. The alternatives are worse on this read: a 32-byte-aligned pad
 * (FK*2 + 32 = 288 B, 16-way) or FK*2 exactly (all rows on bank 0), and the conflict-free pitch
 * (FK*2 + 8 = 264 B, 2r % 32, sixteen distinct starts) is only 8-byte aligned and would force the
 * fragment read onto four ds_read_b64. +8 elements is also libr4d's own pad, restated for a
 * 2-byte element. Measured only for correctness here; see the unit note.
 *
 * ---- the 256-column down tile on a 64 KB LDS budget ---------------------------------
 *
 * libr4d's aligned down tile is 32 rows by 256 columns over sixteen waves, one staging of the
 * activation for twice the columns. Staged as bf16 that weight tile is 256 * 272 = 68 KB, over
 * gfx1151's 64 KB workgroup LDS, so this port's 256-column block takes its two 128-column halves
 * in turn: same block count as libr4d's rule (the grid's y axis is N/256), the same per-element
 * WMMA chain and fold, and the two halves share the block's run search, table load and token
 * resolution. It is a column tile of 256, staged in halves -- not a second kernel form with
 * different numbers, which is what the selftest holds it to.
 *
 * ---- E4M3 -> f16, exactly ----------------------------------------------------------
 *
 * Both operands stage as f16 through halo_hc_f16.h's e4m3x4: an E4M3 byte's magnitude moved to an
 * f16's bits 14..7 and its sign to bit 15 reads as 2^-8 times the code's value, for every E4M3 value
 * including zero and the subnormals, in about two VALU a value. libr4d's hardware fp8 WMMA decodes
 * E4M3 exactly too, so the products are libr4d's; the 2^16 the pair leaves out is folded into the
 * weight's scale (kMoeF16Fix, where libr4d folds the block scale). Every four-bit code table
 * (kW4IntPerm, kW4nlPerm, kW5nlMagPerm) is E4M3-exact, so the four-bit forms are exact as well.
 *
 * The w4nl table is a sign (the code's bit 3) over eight magnitudes, so its codes skip the E4M3
 * step: one byte permute each gives a magnitude's two f16 bytes (moeq_stage_nl, about 1.75 VALU a
 * value against 3.75 through the E4M3 bytes). The bits are the ones e4m3x4 makes of the code's E4M3
 * byte, so the two paths stage the same tile.
 *
 * ---- the two grids -------------------------------------------------------------------
 *
 * As libr4d: inside the decode band (runs averaging under a row and a half, r4d_moe_decode_band)
 * the grid is (tile, run) with 16-row tiles, one run a block, its run found by the one-round-trip
 * wave search; outside it, 32-row tiles over the uniform row grid with each wave skipping the M
 * tiles its run does not reach, or -- with scratch to hold the map -- the run-aligned form, one
 * block a run. Every form agrees bit for bit (the same WMMA chain per element, in the same K
 * order), which is what the selftest checks them against each other for. The sorted operand's
 * decode form additionally has moe_down_dec_kernel (one block a run tile, G 64-column tiles of
 * output, the same chain); the (tile, run) form serves it too, and the two agree byte for byte.
 *
 * ---- the K rotation (this port's, not libr4d's) ----------------------------------------
 *
 * A tile's K loop does not start at K block 0: it starts at moeq_rot(expert, 128-column block)
 * and wraps, so the tile's passes are the same 128-wide blocks in a rotated order. The reason is
 * DRAM, not arithmetic. A weight row is K bytes (2 KB at the gate/up projection), and every tile
 * of every block in flight reads the same 128-byte slice of its rows at the same moment when they
 * all start at block 0: those slices share their low address bits, so the whole grid hammers the
 * few channels those bits select and the rest idle. Measured on gfx1151 with a bare read of the
 * same pattern (64 rows by 128 bytes a pass, 64 MiB of experts): 69 GB/s in lockstep, 209 GB/s
 * with the start staggered; at 512 MiB, 149 against 214. The tile's own bytes are unchanged.
 *
 * The rotation is a function of what every form sees for an element -- the run's expert and the
 * output's 128-column block -- so every form still runs each element's WMMA chain and fold over
 * the same blocks in the same order and the forms agree bit for bit, as before. Against libr4d
 * (block 0 first) the per-element sum is reassociated by whole 128-blocks, inside the fp32
 * tolerance the host checks hold it to. Where a form may stage two blocks a pass the start is
 * kept even, so the two-block pass stays two consecutive blocks of the one-block order.
 */
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_emit.h"
#include "halo_fp8_wmma.h"
#include "halo_hc_f16.h"
#include "halo_moe_sort.h"
#include "halo_wmma.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace {

constexpr uint32_t kWave  = 32;
constexpr uint32_t kBlock = 128;   /* the fp8 scale block, along K and along N */
constexpr uint32_t kFK    = 128;   /* K elements staged per pass: exactly one scale block */

/* 2^16: what e4m3x4 leaves out of the two operands' product (2^-8 each; see the file header).
 * Folded into the weight's scale. */
constexpr float kMoeF16Fix = 65536.0f;
/* `mode` bit 4: the code table is the w4nl table, staged by moeq_stage_nl from MoeTab's f16 bytes
 * (moeq_nl_tab) rather than expanded to E4M3. */
constexpr uint32_t kModeNl = 1u << 4;

using halo_moesort::kMaxExpert;
using halo_moesort::wave_incl_scan_i32;

typedef unsigned int v4u __attribute__((ext_vector_type(4)));

/* The weight and its scales through global-space pointers. An expert's plane comes out of a
 * pointer table (moeq_wtab_at), so the compiler cannot place it and would issue flat loads, which
 * count on the LDS counter too: every wait for a fragment read would then wait for the prefetch
 * as well. The pointers address global memory. */
typedef __attribute__((address_space(1))) const v4u*      moeq_g4;
typedef __attribute__((address_space(1))) const uint32_t* moeq_g1;

/* The run map's sentinel expert: the block owns no run, only the dropped-slot rows to zero. */
constexpr uint32_t kMapTail = 0xffffffffu;

/* The aligned prefill tile's rows (libr4d's kPrefillRows), and its columns where no column tile
 * straddles two stacked weight parts (kPrefillCols). */
constexpr long long kPrefillRows = 32;
constexpr long long kPrefillCols = 128;
/* The run-aligned form's rows a block (MTP 4): one block a run of up to 64 rows. A prefill chunk's
 * runs average T / n_expert rows (40 at a 2048-token chunk of Qwen3.8-Flash-Next), so at 32 most
 * runs took two blocks, each expanding the run's whole weight tile and filling its last 16-row
 * tiles with dead rows; at 64 most take one, and the waves skip the dead tiles. The same WMMA chain
 * and fold per element, so the bytes are unchanged. */
constexpr long long kAlignRows = 64;

/* One class of experts (libr4d's MoeCls, unchanged): `n` rows and K `k`, the packed weight row
 * pitch `w_ld` in BYTES and the scale plane's row pitch `s_ld` in elements (dwords at W4 3, whose
 * plane packs four E4M3 group scales into one word). `w_s0`/`s_s0` are a stacked plane's entry
 * stride in bytes, 0 marking a pointer table. */
struct MoeCls {
    const void* const* w;
    const void* const* s;
    uint32_t w_s0, s_s0, n, k, w_ld, s_ld;
};

/* The code table, shared by all classes (halo_args.h's kW4IntPerm / kW4nlPerm / kW5nlMagPerm). */
struct MoeTab { uint32_t v[4]; };

/* An entry of a weight-table operand: `tab` is the pointer array when stride0 is 0, else the base
 * of a stacked plane. Resolved on the device, since the mover may rewrite the table. */
__device__ __forceinline__ const uint8_t* moeq_wtab_at(const void* const* tab, size_t stride0,
                                                       uint32_t e) {
    if (stride0 == 0) return (const uint8_t*)tab[e];
    return (const uint8_t*)tab + (size_t)e * stride0;
}

/* The LDS tile's row pitch in bytes, for a 2-byte element and a 32-byte-per-lane fragment read:
 * see the file header. Rows are 16-byte aligned, so the read is two ds_read_b128. FKP is the K
 * staged per pass (one 128-wide block, or two in the decode form's pinnable variant). */
constexpr uint32_t moeq_pitch(uint32_t fkp) { return fkp * 2u + 16u; }

/* The 128-wide K block a tile's K loop starts at: see "the K rotation" in the file header. A
 * function of the run's expert `e` (its global number, so the two- and four-class launches agree)
 * and the tile's 128-column block `cb`, nothing a form chooses, so every form walks an element's K
 * in the same order. `pairs` (the forms whose decode grid may stage two blocks a pass) keeps the
 * start even where the block count is, so a two-block pass is two consecutive blocks and the one-
 * and two-block variants fold in the same order. */
__device__ __forceinline__ uint32_t moeq_rot(uint32_t e, uint32_t cb, uint32_t nkb, bool pairs) {
    if (pairs && nkb % 2u == 0u) return 2u * ((e + cb) % (nkb / 2u));
    return (e + cb) % nkb;
}

/* libr4d's w4_e4m3: sixteen levels from two eight-byte permutes. V_PERM_B32 selects from a 64-bit
 * source, so one table holds eight entries; two tables and a selector between them cover a nibble:
 *
 *   m   = nib & 7                        the index within a half, the same for both
 *   lo  = perm(tab[1], tab[0], m)        E4M3 of entries 0..7   (codes 0..7)
 *   hi  = perm(tab[3], tab[2], m)        E4M3 of entries 8..15  (codes 8..15)
 *   sel = 0x03020100 + ((nib & 8) >> 1)  byte i picks lo[i], or hi[i] four bytes further on
 *   out = perm(hi, lo, sel)
 *
 * The add cannot carry: each selector byte is 0..3 before it and 4..7 after. Byte b of a packed
 * dword holds element b in its low nibble and b+4 in its high one, so the two masks yield elements
 * 0..3 and 4..7 in order, with no interleave. Ported as it stands: gfx1151 has the same VALU and
 * the same v_perm_b32. */
__device__ __forceinline__ uint32_t moeq_w4_e4m3(uint32_t nib, const uint32_t (&tab)[4]) {
    const uint32_t m   = nib & 0x07070707u;
    const uint32_t lo  = __builtin_amdgcn_perm(tab[1], tab[0], m);
    const uint32_t hi  = __builtin_amdgcn_perm(tab[3], tab[2], m);
    const uint32_t sel = 0x03020100u + ((nib & 0x08080808u) >> 1);
    return __builtin_amdgcn_perm(hi, lo, sel);
}

/* The 32 codes of sixteen packed bytes as E4M3 bytes in k order: o[m] holds elements 4m..4m+3.
 * libr4d's Stage8 was two uint4 (the fragment read's shape there); a dword array is what deq16
 * wants here, one call per sixteen elements. */
__device__ __forceinline__ void moeq_w4_expand(const uint32_t* p, const uint32_t (&tab)[4],
                                              uint32_t (&o)[8]) {
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        o[2 * i]     = moeq_w4_e4m3(p[i] & 0x0F0F0F0Fu, tab);
        o[2 * i + 1] = moeq_w4_e4m3((p[i] >> 4) & 0x0F0F0F0Fu, tab);
    }
}

/* w5nl64a8h's 32 codes: magnitudes through w4_expand (the table holds the sixteen magnitudes), then
 * each byte's sign from `sg`. Output dword m holds elements 4m..4m+3, and the relayout stores
 * element 4m+b's sign at bit 7-m of byte b of `sg` (halo_layout.cpp), so shifting by m puts dword
 * m's signs at bit 7 of their bytes (E4M3's sign bit) and the mask keeps them. */
__device__ __forceinline__ void moeq_w5_expand(const uint32_t* p, uint32_t sg,
                                               const uint32_t (&tab)[4], uint32_t (&o)[8]) {
    moeq_w4_expand(p, tab, o);
#pragma unroll
    for (uint32_t m = 0; m < 8; ++m) o[m] |= (sg << m) & 0x80808080u;
}

/* A staged row's 16 or 32 K elements: ND dwords of E4M3 (the bytes the fp8 plane holds, or what
 * the four-bit code tables expand to) become 4 * ND f16 at (row, k0) of an LDS tile, 2^-8 times
 * their values (the file header) -- two 16-byte stores per sixteen. */
template <uint32_t PITCH, uint32_t ND>
__device__ __forceinline__ void moeq_stage(uint16_t* __restrict__ lds, uint32_t row, uint32_t k0,
                                           const uint32_t (&w)[ND]) {
    static_assert(ND == 4 || ND == 8, "sixteen or thirty-two elements");
    uint16_t* p = lds + (size_t)row * (PITCH / 2u) + k0;
#pragma unroll
    for (uint32_t i = 0; i < ND / 4u; ++i) {
        uint32_t d[8];
#pragma unroll
        for (uint32_t x = 0; x < 4; ++x) halo_hcf::e4m3x4(w[i * 4 + x], d[2 * x], d[2 * x + 1]);
        *(v4u*)(p + i * 16)     = v4u{ d[0], d[1], d[2], d[3] };
        *(v4u*)(p + i * 16 + 8) = v4u{ d[4], d[5], d[6], d[7] };
    }
}

/* The w4nl codes of 32 K elements (four packed dwords, moeq_w4_expand's order: dword i holds
 * elements 8i..8i+3 in its low nibbles and 8i+4..8i+7 in its high ones) straight to f16 at (row,
 * k0). `tab` is moeq_nl_tab's: the eight magnitudes' f16 high bytes in v[0..1] and low bytes in
 * v[2..3], so a nibble's low three bits pick both bytes and its bit 3 is the sign. */
template <uint32_t PITCH>
__device__ __forceinline__ void moeq_stage_nl(uint16_t* __restrict__ lds, uint32_t row, uint32_t k0,
                                              const uint32_t (&p)[4], const uint32_t (&tab)[4]) {
    uint16_t* q = lds + (size_t)row * (PITCH / 2u) + k0;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t mlo = p[i] & 0x07070707u, mhi = (p[i] >> 4) & 0x07070707u;
        const uint32_t hlo = __builtin_amdgcn_perm(tab[1], tab[0], mlo) | ((p[i] << 4) & 0x80808080u);
        const uint32_t llo = __builtin_amdgcn_perm(tab[3], tab[2], mlo);
        const uint32_t hhi = __builtin_amdgcn_perm(tab[1], tab[0], mhi) | (p[i] & 0x80808080u);
        const uint32_t lhi = __builtin_amdgcn_perm(tab[3], tab[2], mhi);
        *(v4u*)(q + i * 8) = v4u{ __builtin_amdgcn_perm(hlo, llo, 0x05010400u),
                                  __builtin_amdgcn_perm(hlo, llo, 0x07030602u),
                                  __builtin_amdgcn_perm(hhi, lhi, 0x05010400u),
                                  __builtin_amdgcn_perm(hhi, lhi, 0x07030602u) };
    }
}

/* A fragment: lane l holds row (row0 + (l & 15)) at k [ks, ks + 16), all sixteen in one v16bf --
 * one 32-byte read, and lanes 16-31 read the same address (halo_wmma.h's fragment map). */
template <uint32_t PITCH>
__device__ __forceinline__ v16hf moeq_frag(const uint16_t* __restrict__ lds, uint32_t row0,
                                           uint32_t ks, uint32_t lane) {
    return halo::mk_ldg<v16hf>(lds + (size_t)(row0 + (lane & 15u)) * (PITCH / 2u) + ks);
}

/* W4 is the weight's form (libr4d's numbering): 0 block E4M3 with a bf16 scale per 128x128 tile; 1
 * four-bit codes with a bf16 scale per (row, 128 of K); 2 the same codes with an E4M3 scale per
 * (row, 64 of K) times the fixed kW4G64Scale (w4nl64a8h); 3 that per 32 (w4nl32a8h); 4 five-bit
 * codes, form 2's magnitudes and scales plus a sign plane (w5nl64a8h). At 2-4 a staged 128-wide
 * pass holds kSub scale groups, each in its own accumulator.
 *
 * ES is libr4d's (tile, run) decode grid: 16-row tiles, one run a block (MT 1, one wave grid step),
 * with the run found by the wave search below. Otherwise 32-row tiles over four or eight waves.
 *
 * NTL is the block's N tiles of 16 columns, WGN the waves along N; HC 2 has a block cover several
 * such column groups in turn, as many as the grid's y axis leaves it (N / FN / gridDim.y): two for
 * the aligned form's 256-column down tile (see the file header), all of them for block fp8's down
 * form. FKP is the K staged per pass: kFK, or two blocks in the decode form's pinnable variant.
 *
 * `map` is the run map (moe_blockmap_kernel), null where the block takes the uniform grid; with a
 * map the block is one run, named by its entry, and the grid is the map's capacity.
 *
 * MTP is the row grid's M tiles a block (FM = 16 * MTP), one wave row each: 2 on the uniform grid,
 * 4 on the run-aligned form (kAlignRows), where a block is one run and the waves of the 16-row
 * tiles past the run's end only stage, so a run's weight tile is expanded once for up to 64 rows. A
 * wave keeps one M tile at either size: two would double its accumulators and spill. At MTP 4 a
 * block is sixteen waves, and two of them share a WGP only at 192 VGPRs or fewer (eight waves a
 * SIMD), hence the waves-per-EU floor: the down tile's two-half form drew 196 and ran at half the
 * occupancy, 18% slower. The floor is two blocks a WGP (MTP * WGN / 2 waves a SIMD); not at W4 3,
 * whose four scale groups' accumulators would spill under it. */
template <int W4, bool ES, uint32_t NTL, uint32_t WGN, uint32_t HC, uint32_t FKP, uint32_t MTP = 2,
          uint32_t WMP = 1>
__global__ __launch_bounds__((ES ? 1u : MTP / WMP) * WGN * kWave)
__attribute__((amdgpu_waves_per_eu(!ES && MTP >= 4u && W4 != 3 ? MTP / WMP * WGN / 2u : 1u)))
void moe_gemm_q_kernel(
    const uint8_t* __restrict__ Aq, const float* __restrict__ As,
    const MoeCls c_even, const MoeCls c_odd, const MoeCls c_2, const MoeCls c_3,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    uint16_t* __restrict__ Y, const uint32_t* __restrict__ map,
    uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k,
    uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t mode, const MoeTab tab) {
    /* `mode`: bit 0 `a_sorted`, bits 1-2 the class shift (1 for two classes, 2 for four), bit 4
     * kModeNl, bits 8+ `parts`. */
    const uint32_t a_sorted = mode & 1u, cshift = (mode >> 1) & 3u, parts = mode >> 8;
    const bool nl = (mode & kModeNl) != 0u;
    constexpr uint32_t MT = ES ? 1u : MTP, WGM = ES ? 1u : MTP / WMP;
    constexpr uint32_t FM = MT * 16u, FN = NTL * 16u, NTHR = WGM * WGN * kWave;
    constexpr uint32_t WM = MT / WGM, WN = NTL / WGN;
    static_assert(MT % WGM == 0 && NTL % WGN == 0, "the wave grid must divide the tile");
    /* The block-fp8 weight has one scale per 128x128 tile; a w4 weight one per row, so its block
     * may span two fp8 scale rows. */
    static_assert(W4 != 0 || kBlock % FN == 0,
                  "a block's N rows must sit in one 128-row weight scale row");
    /* Scale groups per 128-wide block, and 128-wide blocks per staged pass (one, or two at FKP
     * 256, which only the forms whose fold already sums scale groups take). */
    constexpr uint32_t kSub  = W4 == 3 ? 4u : (W4 == 2 || W4 == 4) ? 2u : 1u;
    constexpr uint32_t kBpS  = FKP / kBlock;
    constexpr uint32_t kSubT = kSub * kBpS;
    static_assert(FKP == kFK || (FKP == 2u * kFK && kSub > 1),
                  "FKP is one scale block, or two where the fold already sums groups");
    static_assert(FKP % (kSubT * 16u) == 0, "a scale group is whole WMMA K steps");
    static_assert(FKP % 32u == 0, "a thread stages 32 K elements of a row");
    static_assert(WGM * WGN * kWave % (FKP / 32u) == 0, "the staging pass divides the block");
    static_assert(FM + FN <= NTHR, "a thread stages each row's and each column's scale: see the "
                                   "scale note at the K loop");
    static_assert(!ES || HC == 1u, "the decode grid's block is one column group wide");
    constexpr uint32_t kPitch = moeq_pitch(FKP);
    /* Passes in flight: the K loop's register ring (see the pipelining note at the loop). */
    constexpr uint32_t kPD = WM > 1u ? 1u : 2u;

    __shared__ uint16_t s_a[FM * (kPitch / 2u)];
    __shared__ uint16_t s_b[FN * (kPitch / 2u)];
    __shared__ int32_t  s_tok[FM];
    /* The pass's scales: one per activation row, and one per weight column (w4) or one for the
     * block (fp8), for each 128-wide block of the pass. See "the scales ride with the tile" at the
     * K loop. */
    __shared__ float    s_as[kBpS * FM];
    __shared__ float    s_ws[kSubT * FN];

    /* The wave index through readfirstlane: the compiler does not know tid / 32 is wave-uniform, and
     * the per-M-tile run test below would otherwise branch on exec around the WMMA, with every
     * accumulator live across both paths (300+ VGPRs of spill at two M tiles a wave). */
    const uint32_t tid = threadIdx.x, lane = tid % kWave,
                   wave = (uint32_t)__builtin_amdgcn_readfirstlane((int)(tid / kWave));
    const uint32_t wgm = wave / WGN, wgn = wave % WGN;
    /* With ES the x axis is (tile, run) with run minor, so a tile's blocks, which share activation
     * rows, land on neighbouring CUs. */
    const uint32_t jrun = ES ? (blockIdx.x % FM) : 0u;

    /* The block's rows and, where the prologue knows it, its expert. With the map (the run-aligned
     * form) entry `blockIdx.x` is (expert, first row) and the block serves only that run, up to the
     * run's end clamped to the output's rows; the map's tail entry owns the dropped-slot rows. */
    uint32_t r0, rN;
    bool tail = false;
    if constexpr (ES) {
        r0 = (blockIdx.x / FM) * FM;
        if (r0 >= T) return;
        rN = (r0 + FM < T) ? (r0 + FM) : T;
    } else if (map) {
        if (blockIdx.x >= map[0]) return;
        const uint32_t* ent = map + 1 + 2 * blockIdx.x;
        r0 = ent[1];
        tail = ent[0] == kMapTail;
        rN = (r0 + FM < T) ? (r0 + FM) : T;
        if (!tail) {
            const uint32_t oe = (uint32_t)off[ent[0] + 1];
            const uint32_t al_end = oe < T ? oe : T;
            rN = (r0 + FM < al_end) ? (r0 + FM) : al_end;
        }
    } else {
        r0 = blockIdx.x * FM;
        if (r0 >= T) return;
        rN = (r0 + FM < T) ? (r0 + FM) : T;
    }

    /* The column groups a block covers in turn (HC above). */
    const uint32_t hcn = HC == 1u ? 1u : (N / FN + gridDim.y - 1u) / gridDim.y;

    /* The map's tail block: it owns no run, only the rows past the live count (dropped slots, or
     * protected experts' rows their own GEMM writes afterwards). No LDS, so it leaves at once. */
    if (tail) {
        const uint32_t live = (uint32_t)off[n_expert];
#pragma unroll 1
        for (uint32_t h = 0; h < hcn; ++h) {
            const uint32_t n0 = (blockIdx.y * hcn + h) * FN;
            for (uint32_t r = (r0 > live ? r0 : live); r < rN; ++r)
                for (uint32_t c = tid; c < FN && n0 + c < N; c += NTHR)
                    Y[(size_t)r * y_ld + n0 + c] = 0;
        }
        return;
    }

    /* ES: this thread's NPT experts' offsets (plus the next), which of them hold a run meeting the
     * tile, and the wave's running count of those, carried to the run search below. */
    constexpr uint32_t NPT = ES ? kMaxExpert / NTHR : 1u;
    static_assert(!ES || (kMaxExpert % NTHR == 0 && FM <= NTHR),
                  "ES is the decode grid's search: the offsets divide over the block's threads");
    __shared__ uint32_t s_rc[ES ? NTHR / kWave : 1];
    __shared__ uint32_t s_sel[ES ? 3 : 1];
    /* The chunked run search's list (above kMaxExpert experts) and its per-wave counts. */
    __shared__ int32_t  s_run[ES ? FM : 1];
    __shared__ uint32_t s_wc[ES ? NTHR / kWave : 1];
    uint32_t rs_ov[NPT + 1], rs_pm = 0, rs_cnt = 0, rs_inc = 0;
    uint32_t live = 0, e_sel = 0, rs_first = 0, rs_end = 0;

    if constexpr (ES) {
        /* Every load the search needs, issued at once (sorted rows, this thread's offsets, live
         * count): one round trip, then one barrier publishes tokens and wave counts together. */
        int32_t f = -1;
        if (tid < FM && r0 + tid < rN) f = sorted[r0 + tid];
#pragma unroll
        for (uint32_t k = 0; k <= NPT; ++k) {
            const uint32_t x = tid * NPT + k;
            rs_ov[k] = x <= n_expert ? (uint32_t)off[x] : 0u;
        }
        live = (uint32_t)off[n_expert];
        if (tid < FM)
            s_tok[tid] = f < 0 ? -1
                              : (a_sorted ? (int32_t)(r0 + tid) : f / (int32_t)top_k);
#pragma unroll
        for (uint32_t k = 0; k < NPT; ++k) {
            const uint32_t x = tid * NPT + k;
            const bool p = x < n_expert && rs_ov[k] < rs_ov[k + 1] && rs_ov[k] < rN &&
                           rs_ov[k + 1] > r0;
            rs_pm |= (uint32_t)p << k;
            rs_cnt += p ? 1u : 0u;
        }
        rs_inc = (uint32_t)wave_incl_scan_i32((int32_t)rs_cnt, lane);
        if (lane == kWave - 1) s_rc[wave] = rs_inc;
        /* The search's barriers order LDS only. */
        lds_barrier();
    } else {
        /* The token each of the block's sorted rows reads its activation from, resolved once. -1
         * marks a dropped slot, whose output row is zero. */
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
        live = (uint32_t)off[n_expert];
    }

    /* Rows past the live count are dropped slots, zeroed here -- before the run search, since a
     * tile whose rows are all past the live count finds no run and leaves at the search. No run owns
     * them, so under ES only one of the tile's blocks writes them (identical concurrent writes are
     * still a race). With the map the tail entry owns them; the row grid has no such block, so it
     * zeroes them here (see the per-column-group pass below, which the aligned form's halves need). */
    if constexpr (ES) {
        const uint32_t n0 = blockIdx.y * FN;
        if (jrun == 0)
            for (uint32_t r = (r0 > live ? r0 : live); r < rN; ++r)
                for (uint32_t c = tid; c < FN && n0 + c < N; c += NTHR)
                    Y[(size_t)r * y_ld + n0 + c] = 0;
    }

    /* The expert of the block's first run. Under ES the (tile, run) grid would make the walk
     * dominate -- a few dozen slots spread over hundreds of experts, one dependent load an expert
     * -- so the block finds its tile's runs in one parallel pass: each thread tests NPT consecutive
     * experts for rows meeting this tile, and a wave scan plus the wave totals place hits in expert
     * (= row) order, so run `jrun` is entry `jrun`. The owner hands the expert and the run's bounds
     * over LDS. With the map the entry names it. Under the row grid a binary search for the tile's
     * first row finds the expert the loop below starts from. */
    if constexpr (ES) {
        if (n_expert <= (uint32_t)kMaxExpert) {
            uint32_t before = rs_inc - rs_cnt, nrun = 0;
#pragma unroll
            for (uint32_t w = 0; w < NTHR / kWave; ++w) {
                if (w < wave) before += s_rc[w];
                nrun += s_rc[w];
            }
            if (jrun >= before && jrun < before + rs_cnt) {
                uint32_t m = jrun - before;
#pragma unroll
                for (uint32_t k = 0; k < NPT; ++k) {
                    if (!((rs_pm >> k) & 1u)) continue;
                    if (m == 0) {
                        s_sel[0] = tid * NPT + k;
                        s_sel[1] = rs_ov[k] > r0 ? rs_ov[k] : r0;
                        s_sel[2] = rs_ov[k + 1];
                    }
                    --m;
                }
            }
            lds_barrier();
            if (jrun >= nrun) return;   /* the tile has fewer runs than this block's index */
            e_sel = s_sel[0];
            rs_first = s_sel[1];
            rs_end = s_sel[2];
        } else {
            /* Above kMaxExpert experts the search walks NTHR experts at a time (libr4d's chunked
             * pass): a chunk wholly before or after the tile is skipped with one broadcast load,
             * and hits are numbered in expert order, the same numbering as above. */
            uint32_t nrun = 0;
            for (uint32_t c = 0; c < n_expert && nrun <= jrun; c += NTHR) {
                const uint32_t ce = c + NTHR < n_expert ? c + NTHR : n_expert;
                if ((uint32_t)off[ce] <= r0) continue;   /* every expert here ends before it */
                if ((uint32_t)off[c] >= rN) break;       /* and every one from here after it */
                const uint32_t x = c + tid;
                bool p = false;
                if (x < ce) {
                    const uint32_t a = (uint32_t)off[x], b = (uint32_t)off[x + 1];
                    p = a < b && a < rN && b > r0;
                }
                const unsigned long long mk = __ballot(p);
                if (lane == 0) s_wc[wave] = (uint32_t)__popcll(mk);
                __syncthreads();
                uint32_t before2 = nrun, tot = 0;
                for (uint32_t w = 0; w < NTHR / kWave; ++w) {
                    if (w < wave) before2 += s_wc[w];
                    tot += s_wc[w];
                }
                const uint32_t pos = before2 + (uint32_t)__popcll(mk & ((1ull << lane) - 1ull));
                if (p && pos < FM) s_run[pos] = (int32_t)x;
                __syncthreads();
                nrun += tot;
            }
            if (jrun >= nrun) return;   /* the tile has fewer runs than this block's index */
            e_sel = (uint32_t)s_run[jrun];
            const uint32_t a = (uint32_t)off[e_sel];
            rs_first = a > r0 ? a : r0;
            rs_end = (uint32_t)off[e_sel + 1];
        }
    } else if (map) {
        e_sel = map[1 + 2 * blockIdx.x];
    } else {
        uint32_t lo = 0, hi = n_expert;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if ((uint32_t)off[mid] <= r0) lo = mid; else hi = mid;
        }
        e_sel = lo;
    }

    /* The output is `parts` stacked parts of N / parts columns, as is every weight (n / parts rows
     * per part). A weight part narrower than the output's fills the part's leading columns and the
     * rest are zero: a rank's two-block slice of a gate/up pair in the three-block layout the gated
     * quantiser reads, whose output there must be zero. With one part this is the whole weight at
     * column n0. */
    const uint32_t fy = N / parts;

    /* K elements a thread stages of a row: 32 (a 16-byte load of four-bit codes), or 16 for the
     * fp8 plane, so a load instruction's lanes cover whole 128-byte lines -- eight lanes a row, 16
     * bytes each -- rather than the 16-byte pieces of four lines that 32 per lane leave (two loads
     * a lane, each at a 32-byte stride). Measured on a bare read of the pattern: whole lines take
     * 7-8% more of DRAM's bandwidth. */
    constexpr uint32_t kSeg = W4 ? 32u : 16u;
    constexpr uint32_t kTPR = FKP / kSeg;             /* threads covering one staged row */
    constexpr uint32_t kRPP = NTHR / kTPR;            /* rows one staging pass covers */
    constexpr uint32_t kAP  = (FM + kRPP - 1) / kRPP;
    constexpr uint32_t kBP  = (FN + kRPP - 1) / kRPP;
    const uint32_t sn = tid / kTPR, sk = (tid % kTPR) * kSeg;

    /* The block's column groups, in turn: one, or `hcn` (the aligned down tile's two 128-column
     * halves, or block fp8's whole width). Each is a whole tile -- staging, K loop, fold, store --
     * so its bytes are the one-group tile's, and only the run search, the table load and the
     * tokens are shared. */
#pragma unroll 1
    for (uint32_t hcol = 0; hcol < hcn; ++hcol) {
        const uint32_t n0 = (blockIdx.y * hcn + hcol) * FN;
        if (n0 >= N) break;
        const uint32_t part = n0 / fy, j0 = n0 - part * fy;

        /* The row grid's dropped-slot rows, for this column group (the decode grid did them above,
         * the aligned form's tail entry does them for it). */
        if (!map && !ES)
            for (uint32_t r = (r0 > live ? r0 : live); r < rN; ++r)
                for (uint32_t c = tid; c < FN && n0 + c < N; c += NTHR)
                    Y[(size_t)r * y_ld + n0 + c] = 0;

        /* Under ES the loop starts at the block's run: its first row in the tile, run count
         * `jrun`. With the map the entry's row is the run's start and the loop runs once. */
        uint32_t e = e_sel;
        uint32_t rs = ES ? rs_first : r0, run = ES ? jrun : 0u;
        while (rs < rN && rs < live) {
            /* This run: rows [rs, re) all belong to expert `e`. The advance is at the top so `e`
             * never passes the last expert (`off[e + 1]` is read only for e < n_expert). */
            if constexpr (!ES) {
                while (!map && e < n_expert && (uint32_t)off[e + 1] <= rs) ++e;
                if (e >= n_expert) break;
            }
            const uint32_t ehi = ES ? rs_end : (uint32_t)off[e + 1];
            const uint32_t re  = rN < live ? (rN < ehi ? rN : ehi) : (live < ehi ? live : ehi);
            if (re <= rs) break;

            /* The expert's class and where this block's columns sit in its weight. Block-uniform:
             * a pad tile zeroes the run's rows and moves on without LDS or a barrier. `cshift` is 1
             * for two classes, 2 for four; c_2 and c_3 are unused at two. */
            const uint32_t cls = e & ((1u << cshift) - 1u);
            const MoeCls& C = cls == 0u ? c_even : cls == 1u ? c_odd : cls == 2u ? c_2 : c_3;
            const uint32_t fc = C.n / parts;
            if (j0 >= fc) {
                const uint32_t nr = re - rs;
                for (uint32_t x = tid; x < nr * FN; x += NTHR) {
                    const uint32_t c = x % FN;
                    if (n0 + c < N) Y[(size_t)(rs + x / FN) * y_ld + n0 + c] = 0;
                }
                rs = re;
                ++run;
                if (ES || map) break;
                continue;
            }
            const uint32_t wn0 = part * fc + j0;      /* the weight row of the block's column 0 */
            const uint32_t K = C.k, w_ld = C.w_ld;
            const uint32_t ws_ld = C.s_ld;
            const uint32_t kb_n = wn0 / kBlock;       /* the weight scale's N row (fp8 only) */
            if (W4) (void)kb_n;                       /* w4 scales vary per ROW, not per N block */

            const uint8_t*  Wp  = moeq_wtab_at(C.w, C.w_s0, e >> cshift);
            const uint16_t* Wsp = (const uint16_t*)moeq_wtab_at(C.s, C.s_s0, e >> cshift);

            /* The K loop's passes start at block `rot` and wrap (the file header's K rotation):
             * pass ps stages the 128-wide blocks from kpos(ps), consecutive within a pass. */
            const uint32_t nkb = K / kBlock;
            const uint32_t rot = moeq_rot(e, n0 / kBlock, nkb, kSub > 1);
            auto kpos = [&](uint32_t ps) {
                const uint32_t q = ps * kBpS + rot;
                return (q < nkb ? q : q - nkb) * kBlock;
            };

            /* A wave whose M tiles are all past the run has nothing to accumulate (wave-uniform
             * test); it still stages and meets every barrier, skipping only fragment reads, WMMA
             * and the fold. */
            const bool wlive = ES || (r0 + wgm * WM * 16 < re);
            v8f raw[WM][WN], out[WM][WN];
            /* The other scale groups' accumulators (`raw` is the first's). Each starts fresh at its
             * group's first WMMA, so none is zeroed. */
            v8f rx[kSubT > 1 ? kSubT - 1 : 1][kSubT > 1 ? WM : 1][kSubT > 1 ? WN : 1];
#pragma unroll
            for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                for (uint32_t j = 0; j < WN; ++j) {
                    raw[i][j] = v8f(0.f);
                    out[i][j] = v8f(0.f);
                }

            /* The K loop is software pipelined over a ring of kPD register slots: a pass's global
             * loads are issued kPD passes ahead, into the slot its pass frees, and collected at
             * that pass's LDS store. The loop's barriers are LDS-only (lds_barrier), so the loads
             * stay in flight across them -- __syncthreads' workgroup fence would wait for every
             * one of them (s_waitcnt vmcnt(0)) twice a pass, and with it the prefetch would cover
             * only the WMMA block. Bit-identical (the same sum in the same order).
             *
             * The scales ride with the tile: the fold needs each row's activation scale and each
             * column's weight scale, which loaded after the WMMA would be scattered dependent
             * loads. So the thread staging a row or column loads its scale with the tile, one pass
             * ahead, and the fold reads them from LDS. Bit-identical. */
            struct Stage { uint32_t d[kSeg / 4u]; };  /* kSeg K elements, the plane's packed form */
            Stage apre[kPD][kAP];
            Stage bpre[kPD][kBP];                     /* W4: only d[0..3] (16 packed bytes) used */
            uint32_t spre[kPD][W4 == 4 ? kBP : 1];    /* W4 4: the same 32 elements' sign bits */
            /* thread tid < FM: its row's activation scale; thread FM + c: column c's weight scale
             * (w4) or the block's (fp8). At W4 3 it is 32 bits (four packed E4M3 groups), otherwise
             * 16, held zero-extended in a whole register; one per 128-wide block of the pass. */
            using PwsT = uint32_t;
            float pas[kPD][kBpS];
            PwsT  pws[kPD][kBpS];

            /* Every load is unconditional, so the waitcnt pass can count them across the ring: a
             * row or scale with nothing to load reads a valid stand-in (activation row 0, the
             * tile's first weight row) and is zeroed at the LDS store, where the flags below are
             * applied. The sources and flags are fixed for the run, so they are resolved once. */
            const uint8_t* asrc[kAP];
            bool aok[kAP];
#pragma unroll
            for (uint32_t p = 0; p < kAP; ++p) {
                const uint32_t r = sn + p * kRPP;
                const int32_t tok = s_tok[r < FM ? r : 0u];
                aok[p] = r < FM && tok >= 0 && (r0 + r) >= rs && (r0 + r) < re;
                asrc[p] = Aq + (size_t)(aok[p] ? tok : 0) * a_ld + sk;
            }
            /* W4: half the bytes of the row, expanded on the way in; `w_ld` is already the packed
             * byte pitch. At W4 4 the row's sign plane (one bit per element) follows its K/2 bytes
             * of nibbles. */
            const uint8_t* bsrc[kBP];
            bool bok[kBP];
#pragma unroll
            for (uint32_t p = 0; p < kBP; ++p) {
                const uint32_t r = sn + p * kRPP;
                bok[p] = r < FN && (j0 + r) < fc;
                bsrc[p] = Wp + (size_t)(wn0 + (bok[p] ? r : 0u)) * w_ld;
            }
            /* Thread tid < FM: its row's activation scale; thread FM + c: column c's weight scale
             * (w4) or the block's (fp8, column 0 only). Every thread loads both. */
            const uint32_t nn = tid - FM;             /* wraps below FM: those threads stage A */
            bool sok;
            const float* asp;
            const uint8_t* wsp;
            {
                const int32_t tok = s_tok[tid < FM ? tid : 0u];
                const bool aw = tok >= 0 && (r0 + tid) >= rs && (r0 + tid) < re;
                const bool ww = nn < FN && (W4 ? j0 + nn < fc : nn == 0u);
                sok = tid < FM ? aw : ww;
                asp = As + (size_t)(tid < FM && aw ? tok : 0) * as_ld;
                const size_t wrow = W4 ? (size_t)(wn0 + (ww ? nn : 0u)) : (size_t)kb_n;
                wsp = (const uint8_t*)Wsp + wrow * ws_ld * (W4 == 3 ? 4u : 2u);
            }

            /* Pass `cc`'s loads into ring slot `sl`. */
            auto issue = [&](uint32_t sl, uint32_t cc) {
                const uint32_t kbk = cc / kBlock;
#pragma unroll
                for (uint32_t b = 0; b < kBpS; ++b) {
                    pas[sl][b] = asp[kbk + b];
                    if constexpr (W4 == 3) pws[sl][b] = ((moeq_g1)wsp)[kbk + b];
                    else                   pws[sl][b] = ldg_u16((const uint16_t*)wsp + kbk + b);
                }
#pragma unroll
                for (uint32_t p = 0; p < kAP; ++p)
#pragma unroll
                    for (uint32_t q = 0; q < kSeg / 16u; ++q) {
                        const v4u a0 = *(const v4u*)(asrc[p] + cc + 16u * q);
#pragma unroll
                        for (int x = 0; x < 4; ++x) apre[sl][p].d[4 * q + x] = a0[x];
                    }
#pragma unroll
                for (uint32_t p = 0; p < kBP; ++p) {
                    if constexpr (W4) {
                        const v4u b0 = *(moeq_g4)(bsrc[p] + ((cc + sk) >> 1));
#pragma unroll
                        for (int x = 0; x < 4; ++x) bpre[sl][p].d[x] = b0[x];
                        if constexpr (W4 == 4)
                            spre[sl][p] = *(moeq_g1)(bsrc[p] + (K >> 1) + ((cc + sk) >> 3));
                    } else {
                        const v4u b0 = *(moeq_g4)(bsrc[p] + cc + sk);
#pragma unroll
                        for (int x = 0; x < 4; ++x) bpre[sl][p].d[x] = b0[x];
                    }
                }
            };
            /* `s_tok` is settled by the barrier above the run loop and never rewritten, so priming
             * before the loop's own barrier is safe. */
            const uint32_t np = K / FKP;
            /* Every slot is issued every pass, past the tile's end too (the last pass again, which
             * the caches already hold): an issue under a condition would leave the waitcnt pass
             * unsure which loads are the newest, and it would wait for all of them. */
#pragma unroll
            for (uint32_t sl = 0; sl < kPD; ++sl) issue(sl, kpos(sl < np ? sl : np - 1));

            /* The K loop's barriers order LDS only; no thread reads another's global writes inside
             * it. Pass ps is ring slot ps % kPD, so the loop steps kPD passes with the slot loop
             * unrolled inside, every slot index a constant. */
            for (uint32_t ps0 = 0; ps0 < np; ps0 += kPD) {
#pragma unroll
            for (uint32_t sl = 0; sl < kPD; ++sl) {
                const uint32_t ps = ps0 + sl;
                if (ps >= np) break;
                lds_barrier();                   /* the previous pass's reads are done */
#pragma unroll
                for (uint32_t p = 0; p < kAP; ++p) {
                    const uint32_t r = sn + p * kRPP;
                    if (r >= FM) continue;
                    uint32_t w[kSeg / 4u];
#pragma unroll
                    for (uint32_t x = 0; x < kSeg / 4u; ++x) w[x] = aok[p] ? apre[sl][p].d[x] : 0u;
                    moeq_stage<kPitch>(s_a, r, sk, w);
                }
#pragma unroll
                for (uint32_t p = 0; p < kBP; ++p) {
                    const uint32_t r = sn + p * kRPP;
                    if (r >= FN) continue;
                    uint32_t o[kSeg / 4u];
                    if constexpr (W4) {
                        uint32_t w[4];
#pragma unroll
                        for (int x = 0; x < 4; ++x) w[x] = bok[p] ? bpre[sl][p].d[x] : 0u;
                        if constexpr (W4 != 4) {
                            if (nl) { moeq_stage_nl<kPitch>(s_b, r, sk, w, tab.v); continue; }
                        }
                        if constexpr (W4 == 4)
                            moeq_w5_expand(w, bok[p] ? spre[sl][p] : 0u, tab.v, o);
                        else
                            moeq_w4_expand(w, tab.v, o);
                    } else {
#pragma unroll
                        for (uint32_t x = 0; x < kSeg / 4u; ++x)
                            o[x] = bok[p] ? bpre[sl][p].d[x] : 0u;
                    }
                    moeq_stage<kPitch>(s_b, r, sk, o);
                }
                if (tid < FM) {
#pragma unroll
                    for (uint32_t b = 0; b < kBpS; ++b)
                        s_as[b * FM + tid] = sok ? pas[sl][b] : 0.0f;
                } else if (tid < FM + FN) {
                    if constexpr (kSub > 1) {
                        /* The tile's scale groups in column order, first in the low byte; at FKP
                         * 256 the first block's groups, then the second's. */
#pragma unroll
                        for (uint32_t b = 0; b < kBpS; ++b)
#pragma unroll
                            for (uint32_t q = 0; q < kSub; ++q)
                                s_ws[(b * kSub + q) * FN + nn] =
                                    halo_fp8::e4m3((uint32_t)(((sok ? pws[sl][b] : 0u) >> (8 * q)) &
                                                              0xFFu)) *
                                    (kW4G64Scale * kMoeF16Fix);
                    } else {
                        /* The 2^120 deq16_e4m3 leaves out rides here (see the file header). */
                        s_ws[nn] = sok ? halo_fp8::bscale(pws[sl][0]) * kMoeF16Fix : 0.0f;
                    }
                }
                issue(sl, kpos(ps + kPD < np ? ps + kPD : np - 1));
                lds_barrier();

                if (!wlive) continue;
                /* One K step of the tile into `acc`; `fresh` starts from zero instead of the held
                 * value (same sum, no reset). */
                /* NM is the wave's live M tiles: all WM, or at WM 2 only the first when the run ends
                 * inside the wave's rows. It is decided once a pass (below), so a K step is
                 * straight-line code: a run test per tile and step would split every step into
                 * branches, and the compiler hoists every step's fragment reads above them (24
                 * fragments at two M tiles, a 300-VGPR spill). Under ES and at WM 1 the wave test
                 * above already covers it. */
                auto kstep = [&](auto nmc, uint32_t ks, v8f (&acc)[WM][WN], bool fresh) {
                    constexpr uint32_t NM = decltype(nmc)::value;
                    v16hf af[NM], bf[WN];
#pragma unroll
                    for (uint32_t i = 0; i < NM; ++i)
                        af[i] = moeq_frag<kPitch>(s_a, (wgm * WM + i) * 16, ks, lane);
#pragma unroll
                    for (uint32_t j = 0; j < WN; ++j)
                        bf[j] = moeq_frag<kPitch>(s_b, (wgn * WN + j) * 16, ks, lane);
#pragma unroll
                    for (uint32_t i = 0; i < NM; ++i)
#pragma unroll
                        for (uint32_t j = 0; j < WN; ++j)
                            acc[i][j] = halo::mma(af[i], bf[j], fresh ? v8f(0.f) : acc[i][j]);
                };
                auto ksteps = [&](auto nmc) {
                /* At W4 2-4 each scale group of the tile has its own accumulator, so no group's
                 * WMMA waits on the previous group's fold; all fold once at the end of the tile, and
                 * each group's first step starts its accumulator. At FKP 256 group g of the pass is
                 * group g % kSub of block g / kSub, in K order. */
                if constexpr (kSubT > 1) {
                    constexpr uint32_t kG = FKP / kSubT;
                    /* At two M tiles a wave, a scheduling barrier a K step keeps the scheduler from
                     * hoisting every step's fragment reads ahead of the first WMMA (see the
                     * one-group note below). */
                    constexpr bool kFence = !ES && WM > 1u;
#pragma unroll
                    for (uint32_t ks = 0; ks < kG; ks += 16) {
                        kstep(nmc, ks, raw, ks == 0);
                        if constexpr (kFence) __builtin_amdgcn_sched_barrier(0);
                    }
#pragma unroll
                    for (uint32_t q = 1; q < kSubT; ++q)
#pragma unroll
                        for (uint32_t ks = q * kG; ks < (q + 1) * kG; ks += 16) {
                            kstep(nmc, ks, rx[q - 1], ks == q * kG);
                            if constexpr (kFence) __builtin_amdgcn_sched_barrier(0);
                        }
                } else {
                    /* The 128-column one-group tile takes a compiler fence every two K steps: left
                     * alone the scheduler hoists all eight steps' fragment reads ahead of the first
                     * WMMA and the block spills under the ring. The fence only orders. */
#pragma unroll
                    for (uint32_t ks = 0; ks < FKP; ks += 16) {
                        kstep(nmc, ks, raw, false);
                        if constexpr (!ES && NTL == 8 && HC == 1 && WM == 1)
                            if (ks % 32u == 16u) asm volatile("" ::: "memory");
                        if constexpr (!ES && WM > 1u) __builtin_amdgcn_sched_barrier(0);
                    }
                }
                };
                if constexpr (!ES && WM > 1u) {
                    static_assert(WM == 2u, "a pass runs all of a wave's M tiles or its first");
                    if (r0 + (wgm * WM + 1u) * 16u < re)
                        ksteps(std::integral_constant<uint32_t, 2u>{});
                    else
                        ksteps(std::integral_constant<uint32_t, 1u>{});
                } else {
                    ksteps(std::integral_constant<uint32_t, WM>{});
                }
                {
                    /* The forms differ only in what the weight scale varies over:
                     *
                     * fp8:  one bf16 per 128x128 tile, uniform over the block, folded into the
                     *       activation scale before the accumulate.
                     * w4:   one bf16 per (row, 128 of K); a 128x128 block scale would spend all
                     *       sixteen levels on its largest row. A fragment's lanes hold sixteen
                     *       consecutive output columns, so the scale is a per-lane constant per N
                     *       tile.
                     * W4 2-4: groups of 64 or 32 K, two or four per block, each sum under its
                     *       scale.
                     *
                     * The row of an accumulator element is 2*e + (lane>>4) here (halo_wmma.h),
                     * where libr4d's is 8*(lane>>4) + e; the column is `lane & 15` on both. */
                    float asv[kBpS][WM][8];
#pragma unroll
                    for (uint32_t b = 0; b < kBpS; ++b)
#pragma unroll
                        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t)
                                asv[b][i][t] = s_as[b * FM + (wgm * WM + i) * 16 +
                                                    halo::d_row(lane, t)];
                    if constexpr (kSub > 1) {
                        float wq[kSubT][WN];
#pragma unroll
                        for (uint32_t q = 0; q < kSubT; ++q)
#pragma unroll
                            for (uint32_t j = 0; j < WN; ++j)
                                wq[q][j] = s_ws[q * FN + (wgn * WN + j) * 16 +
                                                halo::d_col(lane)];
#pragma unroll
                        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                            for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                                for (uint32_t t = 0; t < 8; ++t) {
                                    /* One 128-wide block at a time, in K order: the update a
                                     * separate 128 pass would make, so FKP 256 is the same bytes. */
#pragma unroll
                                    for (uint32_t b = 0; b < kBpS; ++b) {
                                        const uint32_t base = b * kSub;
                                        const v8f* a0 = base == 0 ? &raw[i][j] : &rx[base - 1][i][j];
                                        float v = (*a0)[t] * wq[base][j];
#pragma unroll
                                        for (uint32_t q = 1; q < kSub; ++q)
                                            v += rx[base + q - 1][i][j][t] * wq[base + q][j];
                                        out[i][j][t] += v * asv[b][i][t];
                                    }
                                }
                    } else if constexpr (W4) {
                        float wsj[WN];
#pragma unroll
                        for (uint32_t j = 0; j < WN; ++j)
                            wsj[j] = s_ws[(wgn * WN + j) * 16 + halo::d_col(lane)];
#pragma unroll
                        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                            for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                                for (uint32_t t = 0; t < 8; ++t) {
                                    out[i][j][t] += raw[i][j][t] * wsj[j] * asv[0][i][t];
                                    raw[i][j][t] = 0.0f;
                                }
                    } else {
                        const float ws = s_ws[0];
#pragma unroll
                        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) asv[0][i][t] *= ws;
#pragma unroll
                        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                            for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                                for (uint32_t t = 0; t < 8; ++t) {
                                    out[i][j][t] += raw[i][j][t] * asv[0][i][t];
                                    raw[i][j][t] = 0.0f;
                                }
                    }
                }
            }
            }

            /* The store: for a fixed accumulator slot sixteen lanes hold sixteen consecutive
             * columns, so it is coalesced across the wave. */
#pragma unroll
            for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) {
                        const uint32_t r = (wgm * WM + i) * 16 + halo::d_row(lane, t);
                        const uint32_t c = (wgn * WN + j) * 16 + halo::d_col(lane);
                        const uint32_t row = r0 + r;
                        if (r >= FM || row < rs || row >= re || n0 + c >= N) continue;
                        Y[(size_t)row * y_ld + n0 + c] = halo::f32_to_bf16(out[i][j][t]);
                    }

            rs = re;
            ++run;
            (void)run;
            /* One run per block under ES and with the map (a map entry names a run). */
            if (ES || map) break;
        }
    }
}

/* Where the (tile, run) decode grid ends and the run-aligned prefill grid begins: an average of 1.5
 * routed rows per expert (libr4d's r4d_moe_decode_band, same measurement). */
inline bool moeq_decode_band(long long T, long long n_expert) { return 2 * T <= 3 * n_expert; }

/* ---- the run map (libr4d's moe_blockmap_kernel and its bounds) -----------------------
 *
 * One row-range a block at prefill is mostly dead rows, so the run-aligned form gives a block one
 * run: the map names, per block, (expert, first row), including the dropped-slot tail under
 * kMapTail. The grid is the map's upper bound and the true count is the first slot, so the blocks
 * past it exit at the head. */

/* The map's upper bound, defined once so the scratch hook and the launcher agree exactly (or the
 * kernel writes past its arena). Each expert contributes one block plus one per FM rows beyond the
 * first; the dropped-slot tail at most T/FM plus one. */
inline long long moeq_blockmap_cap(long long T, long long n_expert, long long FM) {
    return n_expert + (T + FM - 1) / FM + 1;
}
/* One count slot, then two uint32 per entry. */
inline long long moeq_blockmap_bytes(long long cap) {
    return (1 + 2 * cap) * (long long)sizeof(uint32_t);
}

/* One workgroup: the job is a scan over at most kMaxExpert counters, cheaper than a second launch.
 * NT threads, thread tid owning experts tid, tid + NT, ...; offsets are read once, in parallel.
 *
 * The prefix is parallel (a DPP scan per wave and one exchange of wave totals, 256 counts per tile;
 * three tiles at 512 experts), not 513 dependent LDS round trips on one thread ahead of a GEMM that
 * waits for the map. Integer sums are order-independent. */
__global__ __launch_bounds__(256) void moe_blockmap_kernel(
    const int32_t* __restrict__ off, uint32_t n_expert, uint32_t T, uint32_t FM, uint32_t cap,
    uint32_t* __restrict__ map) {
    constexpr uint32_t NT = 256, NW = NT / kWave, kTiles = (kMaxExpert + NT) / NT;
    __shared__ uint32_t s_cnt[kMaxExpert + 1];
    __shared__ int32_t  s_wt[kTiles][NW];
    const uint32_t tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;

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
        const int32_t  x = wave_incl_scan_i32(v, lane);
        if (lane == kWave - 1) s_wt[tl][wave] = x;
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
            map[1 + 2 * o + 0] = kMapTail;
            map[1 + 2 * o + 1] = r;
        }
    }
}

/* A code table that is a sign (bit 3) over eight E4M3 magnitudes -- kW4nlPerm -- as moeq_stage_nl
 * reads it: the magnitudes' f16 high bytes (entries 0..3, 4..7) then their low bytes, each f16 the
 * bits e4m3x4 makes of the magnitude's E4M3 byte. False, `out` untouched, for any other table. */
static bool moeq_nl_tab(const uint32_t* tab, MoeTab* out) {
    uint8_t e[16];
    for (int i = 0; i < 16; ++i) e[i] = (uint8_t)(tab[i / 4] >> (8 * (i % 4)));
    for (int i = 0; i < 8; ++i)
        if ((e[i] & 0x80u) || e[i + 8] != (uint8_t)(e[i] | 0x80u)) return false;
    MoeTab t{};
    for (int i = 0; i < 8; ++i) {
        const uint32_t h = (uint32_t)(e[i] & 0x7Fu) << 7;
        t.v[i / 4]     |= ((h >> 8) & 0xFFu) << (8 * (i % 4));
        t.v[2 + i / 4] |= (h & 0xFFu) << (8 * (i % 4));
    }
    *out = t;
    return true;
}

/* ---- the pins (test hooks; the selftest drives every form through them) ---------------- */

/* 0 the rule; 1 the run-aligned prefill form, 2 the (tile, run) decode form, 3 the uniform prefill
 * grid, whatever the band. Every form's bytes are the same; the selftest compares them. */
static int g_moe_prefill_form = 0;
/* The decode form's staged K: 0 the rule, 1 one 128-wide block a pass, 2 two wherever allowed. */
static int g_moe_fk_pin = 0;
/* The aligned form's column tile on the down projection: 0 the rule, 1 never the 256-column tile,
 * 2 it (refused where the shape cannot take it or the launch is not aligned). */
static int g_moe_down_tile_pin = 0;
/* The sorted operand's decode form: 0 the rule, -1 the (tile, run) form, n > 0 moe_down_dec_kernel
 * with n 64-column tiles a block (capped at kMoeDownDecMaxG). A pin on a shape that cannot take it
 * is refused. */
static int g_moe_down_dec_pin = 0;

/* The decode form's column tiles per block on the sorted operand: T >= 480 takes four, T >= 200
 * two, else one (libr4d's kMoeDownDecG2Rows / kMoeDownDecG4Rows). */
constexpr long long kMoeDownDecG2Rows = 200;
constexpr long long kMoeDownDecG4Rows = 480;
constexpr long long kMoeDownDecMaxG = 16;
static inline long long moeq_down_dec_g(long long T) {
    if (g_moe_down_dec_pin < 0) return 0;
    if (g_moe_down_dec_pin > 0) return g_moe_down_dec_pin;
    return T >= kMoeDownDecG4Rows ? 4 : T >= kMoeDownDecG2Rows ? 2 : 1;
}

/* The sorted operand's decode kernel, defined below (its own note there). */
template <int W4>
__global__ __launch_bounds__(4 * kWave) void moe_down_dec_kernel(
    const uint8_t* __restrict__ Aq, const float* __restrict__ As,
    const MoeCls c_even, const MoeCls c_odd, const MoeCls c_2, const MoeCls c_3,
    const int32_t* __restrict__ off, uint16_t* __restrict__ Y,
    uint32_t T, uint32_t N, uint32_t n_expert, uint32_t G,
    uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t mode, const MoeTab tab);

/* One body for the fp8 and w4 launchers, which differ only in the weight dtype and its pitches, so
 * their shape checks cannot drift apart. `ncls` is 2 for the two-class op (the odd class at N_odd /
 * K_odd) or 4 for the four-class one (classes 1..3 at N_1/K_1..N_3/K_3, `y` at operand 12). */
template <int W4>
int moe_gemm_launch(const RadArgs* a, RadStream stream, int ncls = 2) {
    if (ncls != 2 && ncls != 4) return HALO_NO(RAD_E_INVAL);
    const RadTensor* Aq     = halo_opt(a, 0);
    const RadTensor* As     = halo_opt(a, 1);
    const RadTensor* sorted = halo_opt(a, 4);
    const RadTensor* off    = halo_opt(a, 5);
    const RadTensor* Wc[4]  = { halo_opt(a, 2), halo_opt(a, 6),
                                ncls == 4 ? halo_opt(a, 8) : nullptr,
                                ncls == 4 ? halo_opt(a, 10) : nullptr };
    const RadTensor* Wsc[4] = { halo_opt(a, 3), halo_opt(a, 7),
                                ncls == 4 ? halo_opt(a, 9) : nullptr,
                                ncls == 4 ? halo_opt(a, 11) : nullptr };
    const RadTensor* Y      = halo_opt(a, ncls == 4 ? 12 : 8);
    if (!Aq || !As || !sorted || !off || !Y) return HALO_NO(RAD_E_INVAL);
    for (int q = 0; q < ncls; ++q)
        if (!Wc[q] || !Wsc[q]) return HALO_NO(RAD_E_INVAL);
    if (!Aq->data || !As->data || !sorted->data || !off->data || !Y->data)
        return HALO_NO(RAD_E_INVAL);

    /* Each class's weight geometry (defaulting to class 0's); the output is as wide as the widest
     * class and the activation as long as the longest K. */
    static const char* const kNk[4][2] = { { "N", "K" }, { "N_1", "K_1" }, { "N_2", "K_2" },
                                           { "N_3", "K_3" } };
    long long Nc[4] = { 0, 0, 0, 0 }, Kc[4] = { 0, 0, 0, 0 };
    Nc[0] = halo_geti_or(a, "N", 0);
    Kc[0] = halo_geti_or(a, "K", 0);
    for (int q = 1; q < ncls; ++q) {
        Nc[q] = halo_geti_or(a, ncls == 4 ? kNk[q][0] : "N_odd", Nc[0]);
        Kc[q] = halo_geti_or(a, ncls == 4 ? kNk[q][1] : "K_odd", Kc[0]);
    }
    const long long parts = halo_geti_or(a, "parts", 1);
    const long long n_expert = halo_geti_or(a, "n_expert", 0);
    const long long top_k = halo_geti_or(a, "top_k", 0);
    const long long group = halo_geti_or(a, "group", kBlock);
    if (parts <= 0 || n_expert <= 0 || top_k <= 0) return HALO_NO(RAD_E_SHAPE);
    if (group != (long long)kBlock) return HALO_NO(RAD_E_SHAPE);
    long long N = 0, K = 0;
    for (int q = 0; q < ncls; ++q) {
        if (Nc[q] <= 0 || Kc[q] <= 0) return HALO_NO(RAD_E_SHAPE);
        /* A part of every class is whole 64-column tiles, so a tile is all weight or all padding. */
        if (Kc[q] % kBlock || Nc[q] % (64 * parts)) return HALO_NO(RAD_E_SHAPE);
        if (Nc[q] > N) N = Nc[q];
        if (Kc[q] > K) K = Kc[q];
    }
    if (Aq->dtype != RAD_F8E4M3 || As->dtype != RAD_F32 || Y->dtype != RAD_BF16)
        return HALO_NO(RAD_E_DTYPE);
    if (sorted->dtype != RAD_I32 || off->dtype != RAD_I32) return HALO_NO(RAD_E_DTYPE);

    long long T = 0;
    const long long y_ld = halo_rows_pitch(Y, N, &T);
    if (y_ld < 0 || T <= 0) return HALO_NO(RAD_E_STRIDE);
    long long M = 0;
    const long long a_ld = halo_rows_pitch(Aq, K, &M);
    if (a_ld < 0 || M <= 0) return HALO_NO(RAD_E_STRIDE);
    long long Ms = 0;
    const long long as_ld = halo_rows_pitch(As, K / kBlock, &Ms);
    if (as_ld < 0 || Ms < M) return HALO_NO(RAD_E_STRIDE);
    if (halo_numel(sorted) < T || halo_numel(off) < n_expert + 1) return HALO_NO(RAD_E_SHAPE);

    /* The activation's row count is checked, not inferred: a short operand means the caller passed
     * the gate_up projection's buffer to the down projection. */
    const char* ord = halo_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return HALO_NO(RAD_E_UNSUPPORTED);
    if (M < (a_sorted ? T : (T + top_k - 1) / top_k)) return HALO_NO(RAD_E_SHAPE);

    /* The code table, from the dtype the layout hook accepted: w4nla8h's levels, w5nl64a8h's
     * magnitudes, or two's complement for w4a8 and w4a8h. */
    const char* wdt = halo_gets(a, "dtype");
    const uint32_t* tab = W4 == 4 ? kW5nlMagPerm
                        : (W4 && wdt && (!std::strcmp(wdt, "w4nla8h") ||
                                         !std::strcmp(wdt, "w4nl64a8h") ||
                                         !std::strcmp(wdt, "w4nl32a8h"))) ? kW4nlPerm
                                                                        : kW4IntPerm;

    /* One class's tables. The kernel wants the weight pitch in bytes: half the element count for a
     * 4-bit plane, five eighths for the 5-bit one. `wtab_of` reports the declared element stride, so
     * the scaling happens here. */
    struct ClsTab { const void* const* w; const void* const* s; size_t w_s0, s_s0;
                    long long n, k, w_ld, ws_ld; };
    auto wtab_of = [](const RadTensor* t, uint32_t dtype, int bits, long long cnt, long long rows,
                      long long cols, const void* const** tp, size_t* stride0,
                      long long* ld) -> int {
        if (!t || !t->data || t->rank != 3) return HALO_NO(RAD_E_SHAPE);
        if (t->dtype != dtype) return HALO_NO(RAD_E_DTYPE);
        if (t->shape[0] < cnt || t->shape[1] < rows || t->shape[2] < cols)
            return HALO_NO(RAD_E_SHAPE);
        if (t->stride[2] != 1) return HALO_NO(RAD_E_STRIDE);
        *tp      = (const void* const*)t->data;
        /* The element width is passed in: for a plugin-private dtype such as the 4-bit plane,
         * rad_dtype_bits() returns 0 and a stacked stride would collapse to the zero that marks a
         * pointer array. Bits are multiplied before the divide so a 4-bit plane keeps its half
         * byte. */
        *stride0 = (size_t)((t->stride[0] * (int64_t)bits) / 8);
        *ld      = t->stride[1];
        return RAD_OK;
    };
    auto cls_of = [&](const RadTensor* w, const RadTensor* ws, long long cnt, long long n,
                      long long k, MoeCls* c) -> int {
        long long w_ld = 0, ws_ld = 0;
        size_t w_s0 = 0, s_s0 = 0;
        int rc = wtab_of(w, W4 == 4 ? (uint32_t)HALO_DT_MOEW5
                            : W4 ? (uint32_t)HALO_DT_MOEW4 : (uint32_t)RAD_F8E4M3,
                         W4 == 4 ? 5 : W4 ? 4 : 8, cnt, n, k, &c->w, &w_s0, &w_ld);
        if (rc != RAD_OK) return rc;
        if constexpr (W4 >= 2) {
            /* One E4M3 byte per (row, group of 64 or 32 K). The pitch is derived, not read: the
             * scale view arrives at the weight's logical [n, k] (row stride k) while the stored row
             * is k/64 or k/32 bytes (halo_layout.cpp). A tile's scales are one 16- or 32-bit word,
             * so the kernel's pitch is k/128 words. */
            rc = wtab_of(ws, RAD_F8E4M3, 8, cnt, n, k / (kBlock / (W4 == 3 ? 4 : 2)), &c->s,
                         &s_s0, &ws_ld);
            if (rc != RAD_OK) return rc;
            /* A 5-bit row is its nibbles then its sign bits; the kernel finds the signs K/2 bytes
             * in. */
            if (W4 == 4 ? (w_ld % 8 || k % 32) : (w_ld % 2)) return HALO_NO(RAD_E_STRIDE);
            w_ld = W4 == 4 ? w_ld / 8 * 5 : w_ld / 2;
            ws_ld = k / kBlock;
        } else if constexpr (W4) {
            /* The scale plane is one bf16 per (row, 128 of K), not per 128x128 tile. */
            rc = wtab_of(ws, RAD_BF16, 16, cnt, n, k / kBlock, &c->s, &s_s0, &ws_ld);
            if (rc != RAD_OK) return rc;
            if (w_ld % 2) return HALO_NO(RAD_E_STRIDE);
            w_ld /= 2;
        } else {
            rc = wtab_of(ws, RAD_BF16, 16, cnt, (n + kBlock - 1) / kBlock, k / kBlock, &c->s,
                         &s_s0, &ws_ld);
            if (rc != RAD_OK) return rc;
        }
        /* A stacked entry is one expert's plane, far below 4 GiB; a pointer table is 0. */
        if (w_s0 > 0xFFFFFFFFull || s_s0 > 0xFFFFFFFFull) return HALO_NO(RAD_E_STRIDE);
        c->w_s0 = (uint32_t)w_s0;
        c->s_s0 = (uint32_t)s_s0;
        c->n    = (uint32_t)n;
        c->k    = (uint32_t)k;
        c->w_ld = (uint32_t)w_ld;
        c->s_ld = (uint32_t)ws_ld;
        return RAD_OK;
    };
    /* Class q holds experts e with e % ncls == q at entry e / ncls, ceil((n - q) / ncls) of them.
     * At two classes the last two stay zero and unread. */
    MoeCls cl[4]{};
    for (int q = 0; q < ncls; ++q) {
        const int rc = cls_of(Wc[q], Wsc[q], (n_expert + ncls - 1 - q) / ncls, Nc[q], Kc[q], &cl[q]);
        if (rc != RAD_OK) return rc;
    }
    const MoeCls& ce = cl[0];
    const MoeCls& co = cl[1];
    const MoeCls& c2 = cl[2];
    const MoeCls& c3 = cl[3];
    const uint32_t cshift = ncls == 4 ? 2u : 1u;
    if (parts > 0xFFFFFF) return HALO_NO(RAD_E_SHAPE);
    MoeTab tabv{};
    for (int i = 0; i < 4; ++i) tabv.v[i] = tab[i];
    uint32_t mode_nl = 0;
    if constexpr (W4 >= 1 && W4 <= 3)
        if (moeq_nl_tab(tab, &tabv)) mode_nl = kModeNl;
    const uint32_t mode = (a_sorted ? 1u : 0u) | (cshift << 1) | ((uint32_t)parts << 8) |
                          mode_nl;

    const unsigned ny = (unsigned)(N / 64);
    /* The run map: the caller's scratch, when it is roomy enough for the bound above and the map's
     * LDS tables hold the expert count (kMaxExpert). Its absence is not an error -- the uniform grid
     * serves the prefill band -- but form 3 asks for it never. */
    const long long mapcap = moeq_blockmap_cap(T, n_expert, kAlignRows);
    const long long mapneed = moeq_blockmap_bytes(mapcap);
    const uint32_t* map = (g_moe_prefill_form != 3 && n_expert <= (long long)kMaxExpert &&
                          a->scratch && a->scratch_bytes >= mapneed)
                             ? (const uint32_t*)a->scratch : nullptr;
    auto go = [&](auto kern, dim3 grid, dim3 block, const uint32_t* m) {
        return HALO_EMIT(stream, kern, grid, block, 0,
                         (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3,
                         (const int32_t*)sorted->data, (const int32_t*)off->data,
                         (uint16_t*)Y->data, m, (uint32_t)T, (uint32_t)N, (uint32_t)n_expert,
                         (uint32_t)top_k, (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode,
                         tabv);
    };
    int rc = RAD_OK;
    /* The band: a decode step takes the (tile, run) form (the kernel's ES note) and runs of several
     * rows take the prefill forms; moeq_decode_band has the crossover. */
    const int form = g_moe_prefill_form;
    if ((form == 0 && moeq_decode_band(T, n_expert)) || form == 2) {
        /* The sorted operand's decode kernel: one block a run tile, G column tiles of output. */
        const bool dn_dec = a_sorted && parts == 1 && n_expert <= (long long)kMaxExpert;
        const long long g2 = moeq_down_dec_g(T);
        if (!dn_dec && g_moe_down_dec_pin > 0) return HALO_NO(RAD_E_UNSUPPORTED);
        if (dn_dec && g2 > 0) {
            const unsigned slots  = (unsigned)((T < n_expert ? T : n_expert) + (T + 15) / 16);
            const unsigned groups = (unsigned)((N + 64 * g2 - 1) / (64 * g2));
            rc = HALO_EMIT(stream, (moe_down_dec_kernel<W4>), dim3(slots + 1u, groups),
                           dim3(4 * kWave), 0,
                           (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3,
                           (const int32_t*)off->data, (uint16_t*)Y->data,
                           (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)g2,
                           (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode, tabv);
            if (rc != RAD_OK) return rc;
            return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
        }
        /* The decode grid is (tile, run) wide (the kernel's ES note); decode has one tile of rows,
         * which alone would give only N/64 workgroups. */
        const unsigned tiles = (unsigned)((T + 15) / 16);
        /* Two 128-wide K blocks per pass where the format folds scale groups, every class's K is
         * whole 256-wide passes and the pin (or the rule's row limit) says so; bit-identical. */
        bool fk2 = false;
        if constexpr (W4 >= 2) {
            fk2 = !a_sorted && (g_moe_fk_pin == 2 ||
                                (g_moe_fk_pin == 0 &&
                                 T <= (W4 == 4 ? (1ll << 40) : 80ll)));
            for (int q = 0; q < ncls; ++q)
                fk2 = fk2 && Kc[q] % (2 * (long long)kBlock) == 0;
        }
        if constexpr (W4 >= 2) {
            if (fk2)
                rc = go(&moe_gemm_q_kernel<W4, true, 4, 4, 1, 2 * kFK>,
                        dim3(tiles * 16u, ny), dim3(4 * kWave), nullptr);
            else
                rc = go(&moe_gemm_q_kernel<W4, true, 4, 4, 1, kFK>,
                        dim3(tiles * 16u, ny), dim3(4 * kWave), nullptr);
        } else if constexpr (W4 == 0) {
            /* Block fp8 at 32 columns a block, twice the blocks of the 64-column tile: a decode
             * step's few runs otherwise leave the grid too short to keep every CU streaming
             * (3-5% at four to thirty-two tokens; the same chain per element). */
            rc = go(&moe_gemm_q_kernel<W4, true, 2, 2, 1, kFK>,
                    dim3(tiles * 16u, ny * 2u), dim3(2 * kWave), nullptr);
        } else {
            rc = go(&moe_gemm_q_kernel<W4, true, 4, 4, 1, kFK>,
                    dim3(tiles * 16u, ny), dim3(4 * kWave), nullptr);
        }
    } else {
        /* The prefill form is run-aligned when map scratch is present, uniform otherwise (e.g. a
         * caller that never asked the scratch hook); the answer is the same.
         *
         * The map builder's LDS tables hold kMaxExpert + 1 entries, so wider expert counts take the
         * uniform grid regardless of scratch. */
        /* The 128-column tile where no column tile straddles two parts of the weight. Block fp8
         * with a long K (eight passes or more a tile) takes the 64-column tile instead: twice the
         * blocks at 26 KB of LDS against 44, four resident a WGP rather than two, which keeps more
         * of a long K loop's weight in flight (14% at the gate/up projection, K 2048). */
        const bool wide_n = (N / parts) % kPrefillCols == 0 &&
                            !(W4 == 0 && K >= 8 * (long long)kFK);
        const unsigned nyw = (unsigned)(N / kPrefillCols);
        /* The down projection's 256-column tile: the sorted operand, one part, a w4 weight with
         * per-row scales and every class's width whole tiles take 256 columns over half as many
         * blocks as the 128-column tile. Each wave's share is unchanged, so every element is the
         * same sum (the file header has the two halves, which is how this port fits the tile in
         * 64 KB of LDS). Other shapes keep the 128- or 64-column tile.
         *
         * Block fp8 takes the same form further: one block a run, every 128-column group of the
         * output in turn (the grid's y axis is 1, so the kernel's group count is N / 128). A down
         * tile is a short K loop (four passes at K 512), so the run map, tokens and pipeline fill
         * a block pays dominate it; a block that walks the whole width pays them once a run
         * (9-12% at 128 to 512 tokens). */
        bool dn_wide = a_sorted && parts == 1 &&
                       N % ((W4 == 0 ? 1 : 2) * kPrefillCols) == 0;
        for (int q = 0; q < ncls; ++q)
            dn_wide = dn_wide && Nc[q] % ((W4 == 0 ? 1 : 2) * kPrefillCols) == 0;
        const int tpin = g_moe_down_tile_pin;
        if (tpin < 0 || tpin > 2) return HALO_NO(RAD_E_INVAL);
        if (tpin == 2 && (!map || !dn_wide)) return HALO_NO(RAD_E_UNSUPPORTED);
        if (tpin == 1) dn_wide = false;
        const unsigned gx = (unsigned)mapcap;
        const unsigned nrow = (unsigned)((T + kPrefillRows - 1) / kPrefillRows);
        if (map) {
            rc = HALO_EMIT(stream, moe_blockmap_kernel, dim3(1), dim3(256), 0,
                           (const int32_t*)off->data, (uint32_t)n_expert, (uint32_t)T,
                           (uint32_t)kAlignRows, (uint32_t)mapcap, (uint32_t*)a->scratch);
        }
        if (rc == RAD_OK) {
            constexpr uint32_t kAT = (uint32_t)(kAlignRows / 16);
            if (dn_wide) {
                rc = go(&moe_gemm_q_kernel<W4, false, 8, 4, 2, kFK, kAT, 2>,
                        dim3(gx, W4 == 0 ? 1u : (unsigned)(N / (2 * kPrefillCols))),
                        dim3(2 * kAT * kWave), map);
            } else if (wide_n) {
                if (map)
                    rc = go(&moe_gemm_q_kernel<W4, false, 8, 4, 1, kFK, kAT, 2>, dim3(gx, nyw),
                            dim3(2 * kAT * kWave), map);
                else
                    rc = go(&moe_gemm_q_kernel<W4, false, 8, 4, 1, kFK>, dim3(nrow, nyw),
                            dim3(8 * kWave), map);
            } else {
                if (map)
                    rc = go(&moe_gemm_q_kernel<W4, false, 4, 2, 1, kFK, kAT>, dim3(gx, ny),
                            dim3(2 * kAT * kWave), map);
                else
                    rc = go(&moe_gemm_q_kernel<W4, false, 4, 2, 1, kFK>, dim3(nrow, ny),
                            dim3(4 * kWave), map);
            }
        }
    }
    if (rc != RAD_OK) return rc;
    return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}

/* ==================================================================== moe_gemm_q, the down decode form
 *
 * libr4d's moe_down_dec_kernel: the sorted operand's projection at decode, one block per run tile
 * and G column tiles of output.
 *
 * The (tile, run) grid gives down (K 128 or 256 per rank, N the model width) a block per 16-row
 * tile, run slot and 64 columns: 25 600 blocks at sixteen sequences for ~63 MB of weight, most
 * finding no run, the rest staging 4-8 KB behind a search, a table load and a barrier pair. A run
 * crossing a tile boundary is staged twice.
 *
 * Here a block owns one 16-row tile of one run (tiles start at the run's first row, as in the
 * run-aligned prefill form) and G 64-column tiles of output:
 *
 *   - The grid is (run tile, column group). A run of n rows is ceil(n / 16) tiles, at most
 *     min(T, n_expert) + T/16 in all; block j finds the j-th in expert order in one round trip
 *     (four offsets per thread, a DPP count and one exchange). One extra last block zeroes the rows
 *     past the live count.
 *   - A run of up to sixteen rows is always one tile, so an expert's weight is staged once per
 *     column tile; a longer run gets one block per tile, side by side.
 *   - All G column tiles' passes form one pipeline over (column tile, K block), each loaded one
 *     pass ahead, so the search/table/first-load chain is paid once per G tiles.
 *
 * Bit-identical to the (tile, run) form: each output element is the same WMMA chain and fold, in the
 * same K order; the selftest compares them byte for byte. Here a lane holds the whole 16-wide k run
 * (halo_wmma.h), so a fragment read loses libr4d's 8*(lane>>4) half term and D element e of lane l
 * is row 2*e + (l>>4), column l & 15; a staged row is 16 bf16, dequantised from the plane's E4M3
 * bytes (or the four-bit code tables) as in moe_gemm_q_kernel.
 *
 * Sorted operand only: a run's activation rows are [rs, re) of `Aq` with no gather. Every row in a
 * run is a placed slot (the scatter fills exactly [0, off[n_expert])), so `sorted` is not read. */
template <int W4>
__global__ __launch_bounds__(4 * kWave) void moe_down_dec_kernel(
    const uint8_t* __restrict__ Aq, const float* __restrict__ As,
    const MoeCls c_even, const MoeCls c_odd, const MoeCls c_2, const MoeCls c_3,
    const int32_t* __restrict__ off, uint16_t* __restrict__ Y,
    uint32_t T, uint32_t N, uint32_t n_expert, uint32_t G,
    uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t mode, const MoeTab tab) {
    constexpr uint32_t FM = 16, FN = 64, NTHR = 4 * kWave;
    constexpr uint32_t kSub = W4 == 3 ? 4u : (W4 == 2 || W4 == 4) ? 2u : 1u;
    constexpr uint32_t NPT = kMaxExpert / NTHR;
    constexpr uint32_t kPitch = moeq_pitch(kFK);
    static_assert(kMaxExpert % NTHR == 0, "the offsets divide over the block's threads");
    static_assert(kFK % (kSub * 16u) == 0, "a scale group is whole WMMA K steps");
    static_assert(FM + FN <= NTHR, "a thread stages each row's and each column's scale");
    const uint32_t cshift = (mode >> 1) & 3u, parts = mode >> 8;
    const bool nl = (mode & kModeNl) != 0u;

    __shared__ uint16_t s_a[FM * (kPitch / 2u)];
    __shared__ uint16_t s_b[FN * (kPitch / 2u)];
    __shared__ float    s_as[FM];
    __shared__ float    s_ws[kSub * FN];
    __shared__ uint32_t s_rc[NTHR / kWave];
    __shared__ uint32_t s_sel[3];

    const uint32_t tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
    /* The block's columns: G 64-wide tiles from cg0, the last group cut at N. */
    const uint32_t cg0 = blockIdx.y * G * FN;
    if (cg0 >= N) return;               /* the launcher's grid has none; block-uniform */
    const uint32_t cend = cg0 + G * FN < N ? cg0 + G * FN : N;
    const uint32_t ng = (cend - cg0 + FN - 1) / FN;

    /* The last block owns no run: it zeroes every column of the rows past the live count (dropped
     * slots, or protected experts' rows their own GEMM writes afterwards). */
    if (blockIdx.x == gridDim.x - 1) {
        const uint32_t live = (uint32_t)off[n_expert];
        for (uint32_t r = live; r < T; ++r)
            for (uint32_t c = cg0 + tid; c < cend; c += NTHR) Y[(size_t)r * y_ld + c] = 0;
        return;
    }

    /* The tile: the j-th 16-row run tile in expert, then row, order. Each thread holds NPT
     * consecutive experts' offsets plus the next; tiles are numbered by thread, then over the wave
     * (DPP) and the waves before it. Both barriers order LDS.
     *
     * Every offset is clamped to T, so no run extends past the output (an operand may name fewer
     * rows than its buffer, never more) and the tile count stays within the grid. */
    uint32_t ov[NPT + 1];
#pragma unroll
    for (uint32_t k = 0; k <= NPT; ++k) {
        const uint32_t x = tid * NPT + k;
        const uint32_t raw = (uint32_t)off[x <= n_expert ? x : n_expert];
        const uint32_t o = x <= n_expert ? raw : 0u;
        ov[k] = o < T ? o : T;
    }
    uint32_t nt[NPT], cnt = 0;
#pragma unroll
    for (uint32_t k = 0; k < NPT; ++k) {
        nt[k] = tid * NPT + k < n_expert ? (ov[k + 1] - ov[k] + FM - 1) / FM : 0u;
        cnt += nt[k];
    }
    const uint32_t inc = (uint32_t)wave_incl_scan_i32((int32_t)cnt, lane);
    if (lane == kWave - 1) s_rc[wave] = inc;
    lds_barrier();
    uint32_t before = inc - cnt, ntile = 0;
#pragma unroll
    for (uint32_t w = 0; w < NTHR / kWave; ++w) {
        const uint32_t c = s_rc[w];
        before += w < wave ? c : 0u;
        ntile += c;
    }
    const uint32_t j = blockIdx.x;
    if (j >= ntile) return;             /* fewer tiles than slots: block-uniform */
    if (j >= before && j < before + cnt) {
        uint32_t m = j - before;
        bool hit = false;
#pragma unroll
        for (uint32_t k = 0; k < NPT; ++k) {
            if (!hit && m < nt[k]) {
                s_sel[0] = tid * NPT + k;
                s_sel[1] = ov[k] + m * FM;
                s_sel[2] = ov[k + 1];
                hit = true;
            }
            if (!hit) m -= nt[k];
        }
    }
    lds_barrier();
    /* Block-uniform from here (readfirstlane), so the class, table entry and pass bounds are
     * scalar. The block's rows are [rs, re), at most a tile. */
    const uint32_t e  = (uint32_t)__builtin_amdgcn_readfirstlane((int)s_sel[0]);
    const uint32_t rs = (uint32_t)__builtin_amdgcn_readfirstlane((int)s_sel[1]);
    const uint32_t r1 = (uint32_t)__builtin_amdgcn_readfirstlane((int)s_sel[2]);
    const uint32_t re = rs + FM < r1 ? rs + FM : r1;

    const uint32_t cls = e & ((1u << cshift) - 1u);
    const MoeCls& C = cls == 0u ? c_even : cls == 1u ? c_odd : cls == 2u ? c_2 : c_3;
    const uint32_t fy = N / parts, fc = C.n / parts;
    const uint32_t K = C.k, w_ld = C.w_ld, ws_ld = C.s_ld;
    const uint8_t*  Wp  = moeq_wtab_at(C.w, C.w_s0, e >> cshift);
    const uint16_t* Wsp = (const uint16_t*)moeq_wtab_at(C.s, C.s_s0, e >> cshift);

    /* A column tile past the class's width holds no weight: its rows are zeroed and it takes no
     * pass. The others are marked in `npm`. */
    uint32_t npm = 0;
    for (uint32_t g = 0; g < ng; ++g) {
        const uint32_t n0 = cg0 + g * FN, part = n0 / fy, j0 = n0 - part * fy;
        if (j0 < fc) { npm |= 1u << g; continue; }
        const uint32_t nr = re - rs;
        for (uint32_t x = tid; x < nr * FN; x += NTHR) {
            const uint32_t c = x % FN;
            if (n0 + c < N) Y[(size_t)(rs + x / FN) * y_ld + n0 + c] = 0;
        }
    }
    if (!npm) return;

    /* Passes over (column tile, K block), K innermost. The last pass's successor is never issued. */
    const uint32_t nK = K / kFK;
    const uint32_t P = (uint32_t)__builtin_popcount(npm) * nK;
    struct Pass { uint32_t g, kp; };
    auto advance = [&](Pass s) {
        if (++s.kp < nK) return s;
        s.kp = 0;
        const uint32_t rest = npm & ~((2u << s.g) - 1u);
        s.g = rest ? (uint32_t)__builtin_ctz(rest) : s.g;
        return s;
    };

    constexpr uint32_t kSeg = 32;                     /* K elements a thread stages */
    constexpr uint32_t kTPR = kFK / kSeg;             /* threads covering one staged row */
    constexpr uint32_t kRPP = NTHR / kTPR;            /* rows one staging pass covers */
    constexpr uint32_t kAP  = (FM + kRPP - 1) / kRPP;
    constexpr uint32_t kBP  = (FN + kRPP - 1) / kRPP;
    const uint32_t sn = tid / kTPR, sk = (tid % kTPR) * kSeg;

    /* Operands one pass ahead: unconditional scale loads, the zero selected at the LDS store. */
    struct Stage32 { uint32_t d[8]; };
    Stage32 apre[kAP];
    Stage32 bpre[kBP];
    uint32_t spre[W4 == 4 ? kBP : 1];
    float pau = 0.0f;
    uint32_t pwu = 0u;
    bool pok = false;
    auto issue = [&](const Pass& s) {
        const uint32_t n0 = cg0 + s.g * FN, part = n0 / fy, j0 = n0 - part * fy;
        /* The tile's K loop starts at moe_gemm_q_kernel's block for the same element (the file
         * header's K rotation), so the two forms fold in the same order. */
        const uint32_t kq = s.kp + moeq_rot(e, n0 / kBlock, nK, kSub > 1);
        const uint32_t kbk = kq < nK ? kq : kq - nK;
        const uint32_t cc = kbk * kFK;                 /* kFK is one 128-wide scale block */
        const uint32_t wn0 = part * fc + j0;           /* the weight row of the tile's column 0 */
        if (tid < FM) {
            const uint32_t row = rs + tid;
            pok = row < re;
            pau = As[(size_t)(pok ? row : 0u) * as_ld + kbk];
        } else if (tid < FM + FN) {
            const uint32_t nn = tid - FM;
            pok = j0 + nn < fc;
            const size_t wr = (size_t)(pok ? wn0 + nn : wn0) * ws_ld + kbk;
            if constexpr (W4 == 3)
                pwu = pok ? ((const uint32_t*)Wsp)[wr] : 0u;
            else if constexpr (W4)
                pwu = pok ? (uint32_t)Wsp[wr] : 0u;
            else
                pwu = (uint32_t)Wsp[(size_t)(wn0 / kBlock) * ws_ld + kbk];
        }
#pragma unroll
        for (uint32_t p = 0; p < kAP; ++p) {
            const uint32_t r = sn + p * kRPP;
            if (r >= FM) continue;
            const uint32_t row = rs + r;
            const bool ok = row < re;
            const uint8_t* src = Aq + (size_t)(ok ? row : 0u) * a_ld + cc + sk;
            if (ok) {
                const v4u a0 = *(const v4u*)src, a1 = *(const v4u*)(src + 16);
                for (int x = 0; x < 4; ++x) { apre[p].d[x] = a0[x]; apre[p].d[x + 4] = a1[x]; }
            } else {
#pragma unroll
                for (int x = 0; x < 8; ++x) apre[p].d[x] = 0u;
            }
        }
#pragma unroll
        for (uint32_t p = 0; p < kBP; ++p) {
            const uint32_t r = sn + p * kRPP;
            if (r >= FN) continue;
            const bool ok = (j0 + r) < fc;
            const uint8_t* wr = Wp + (size_t)(wn0 + r) * w_ld;
            if constexpr (W4) {
                if (ok) {
                    const v4u b0 = *(const v4u*)(wr + ((cc + sk) >> 1));
                    for (int x = 0; x < 4; ++x) bpre[p].d[x] = b0[x];
                } else {
                    for (int x = 0; x < 4; ++x) bpre[p].d[x] = 0u;
                }
                if constexpr (W4 == 4)
                    spre[p] = ok ? *(const uint32_t*)(wr + (K >> 1) + ((cc + sk) >> 3)) : 0u;
            } else {
                if (ok) {
                    const v4u b0 = *(const v4u*)(wr + cc + sk);
                    const v4u b1 = *(const v4u*)(wr + cc + sk + 16);
                    for (int x = 0; x < 4; ++x) { bpre[p].d[x] = b0[x]; bpre[p].d[x + 4] = b1[x]; }
                } else {
                    for (int x = 0; x < 8; ++x) bpre[p].d[x] = 0u;
                }
            }
        }
    };

    v8f raw = v8f(0.f), out = v8f(0.f);
    /* The other scale groups' accumulators at W4 2 to 4; each starts fresh at its group's first
     * WMMA, so none is zeroed. */
    v8f rx[kSub > 1 ? kSub - 1 : 1];
    Pass cur{(uint32_t)__builtin_ctz(npm), 0u};
    issue(cur);
    for (uint32_t p = 0; p < P; ++p) {
        __syncthreads();                         /* the previous pass's reads are done */
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");
#pragma unroll
        for (uint32_t q = 0; q < kAP; ++q) {
            const uint32_t r = sn + q * kRPP;
            if (r >= FM) continue;
            moeq_stage<kPitch>(s_a, r, sk, apre[q].d);
        }
#pragma unroll
        for (uint32_t q = 0; q < kBP; ++q) {
            const uint32_t r = sn + q * kRPP;
            if (r >= FN) continue;
            uint32_t o[8];
            if constexpr (W4 == 4) {
                moeq_w5_expand(bpre[q].d, spre[q], tab.v, o);
                moeq_stage<kPitch>(s_b, r, sk, o);
            } else if constexpr (W4) {
                if (nl) {
                    const uint32_t (&w4)[4] = *(const uint32_t (*)[4])bpre[q].d;
                    moeq_stage_nl<kPitch>(s_b, r, sk, w4, tab.v);
                } else {
                    moeq_w4_expand(bpre[q].d, tab.v, o);
                    moeq_stage<kPitch>(s_b, r, sk, o);
                }
            } else {
                moeq_stage<kPitch>(s_b, r, sk, bpre[q].d);
            }
        }
        if (tid < FM) {
            s_as[tid] = pok ? pau : 0.0f;
        } else if (tid < FM + FN) {
            const uint32_t pws = pok ? pwu : 0u;
            if constexpr (kSub > 1) {
#pragma unroll
                for (uint32_t q = 0; q < kSub; ++q)
                    s_ws[q * FN + tid - FM] =
                        halo_fp8::e4m3((uint32_t)((pws >> (8 * q)) & 0xFFu)) *
                        (kW4G64Scale * kMoeF16Fix);
            } else {
                s_ws[tid - FM] = (W4 || tid == FM)
                                     ? halo_fp8::bscale(pws) * kMoeF16Fix : 0.0f;
            }
        }
        const Pass nxt = advance(cur);
        if (p + 1 < P) issue(nxt);
        __syncthreads();

        /* moe_gemm_q_kernel's K step and fold at one M tile by one N tile a wave. */
        auto kstep = [&](uint32_t ks, v8f& acc, bool fresh = false) {
            const v16hf af = moeq_frag<kPitch>(s_a, 0, ks, lane);
            const v16hf bf = moeq_frag<kPitch>(s_b, wave * 16, ks, lane);
            acc = halo::mma(af, bf, fresh ? v8f(0.f) : acc);
        };
        if constexpr (kSub > 1) {
            constexpr uint32_t kG = kFK / kSub;
#pragma unroll
            for (uint32_t ks = 0; ks < kG; ks += 16) kstep(ks, raw, ks == 0);
#pragma unroll
            for (uint32_t q = 1; q < kSub; ++q)
#pragma unroll
                for (uint32_t ks = q * kG; ks < (q + 1) * kG; ks += 16)
                    kstep(ks, rx[q - 1], ks == q * kG);
        } else {
#pragma unroll
            for (uint32_t ks = 0; ks < kFK; ks += 16) kstep(ks, raw);
        }
        {
            float as[8];
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) as[t] = s_as[halo::d_row(lane, t)];
            if constexpr (kSub > 1) {
                float wq[kSub];
#pragma unroll
                for (uint32_t q = 0; q < kSub; ++q) wq[q] = s_ws[q * FN + wave * 16 +
                                                             halo::d_col(lane)];
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    float v = raw[t] * wq[0];
#pragma unroll
                    for (uint32_t q = 1; q < kSub; ++q) v += rx[q - 1][t] * wq[q];
                    out[t] += v * as[t];
                }
            } else if constexpr (W4) {
                const float wsj = s_ws[wave * 16 + halo::d_col(lane)];
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    out[t] += raw[t] * wsj * as[t];
                    raw[t] = 0.0f;
                }
            } else {
                const float ws = s_ws[0];
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) as[t] *= ws;
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    out[t] += raw[t] * as[t];
                    raw[t] = 0.0f;
                }
            }
        }

        /* The tile's last K block: its rows of the run, then a fresh accumulator. */
        if (cur.kp + 1 == nK) {
            const uint32_t n0 = cg0 + cur.g * FN;
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) {
                const uint32_t row = rs + halo::d_row(lane, t);
                const uint32_t c = wave * 16 + halo::d_col(lane);
                if (row >= re || n0 + c >= N) continue;
                Y[(size_t)row * y_ld + n0 + c] = halo::f32_to_bf16(out[t]);
            }
            out = v8f(0.f);
        }
        cur = nxt;
    }
}

}  // namespace

/* ================================================================== the entries */

extern "C" int halo_rad_moe_gemm_q(const RadArgs* a, RadStream s) {
    return moe_gemm_launch<0>(a, s);
}

/* `moe_gemm_q_mod4`: the same launches over four classes, by expert number mod 4. */
extern "C" int halo_rad_moe_gemm_q_mod4(const RadArgs* a, RadStream s) {
    return moe_gemm_launch<0>(a, s, 4);
}

/* The four-bit forms (an E4M3 scale per 64 or 32 of K for w4nl64a8h / w4nl32a8h, else a bf16 scale
 * per 128) and the five-bit w5nl64a8h. */
extern "C" int halo_rad_moe_gemm_w4a8(const RadArgs* a, RadStream stream) {
    const char* wdt = halo_gets(a, "dtype");
    if (wdt && !std::strcmp(wdt, "w4nl64a8h")) return moe_gemm_launch<2>(a, stream);
    if (wdt && !std::strcmp(wdt, "w4nl32a8h")) return moe_gemm_launch<3>(a, stream);
    if (wdt && !std::strcmp(wdt, "w5nl64a8h")) return moe_gemm_launch<4>(a, stream);
    return moe_gemm_launch<1>(a, stream);
}

extern "C" int halo_rad_moe_gemm_w4a8_mod4(const RadArgs* a, RadStream stream) {
    const char* wdt = halo_gets(a, "dtype");
    if (wdt && !std::strcmp(wdt, "w4nl64a8h")) return moe_gemm_launch<2>(a, stream, 4);
    if (wdt && !std::strcmp(wdt, "w4nl32a8h")) return moe_gemm_launch<3>(a, stream, 4);
    if (wdt && !std::strcmp(wdt, "w5nl64a8h")) return moe_gemm_launch<4>(a, stream, 4);
    return moe_gemm_launch<1>(a, stream, 4);
}

/* The run map's bytes outside the decode band (zero inside it: the (tile, run) grid indexes runs off
 * the offsets and the uniform grid binary-searches them). T comes from the parameters (`M *
 * top_k`, the routed row count), not an operand, since the hook runs at declare before tensors are
 * bound; the launcher derives the same count from Y and checks the scratch it is handed, falling
 * back to the uniform grid when it is short. */
static int64_t moeq_map_scratch(const RadArgs* a) {
    const long long n_expert = halo_geti_or(a, "n_expert", 0);
    const long long top_k    = halo_geti_or(a, "top_k", 0);
    const long long M        = halo_geti_or(a, "M", 0);
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert || top_k <= 0 || M <= 0) return 0;
    const long long T = M * top_k;
    if (moeq_decode_band(T, n_expert)) return 0;      /* the (tile, run) grid needs no map */
    return moeq_blockmap_bytes(moeq_blockmap_cap(T, n_expert, kPrefillRows));
}

extern "C" int64_t halo_rad_moe_gemm_q_scratch(const RadArgs* a) { return moeq_map_scratch(a); }
extern "C" int64_t halo_rad_moe_gemm_w4a8_scratch(const RadArgs* a) { return moeq_map_scratch(a); }

/* The four form pins: test hooks, not part of the plugin ABI (the selftest dlsyms them). */
extern "C" void halo_moe_gemm_pin_prefill(int form) { g_moe_prefill_form = form; }
extern "C" void halo_moe_gemm_pin_fk(int pin) { g_moe_fk_pin = pin; }
extern "C" void halo_moe_gemm_pin_down_tile(int pin) { g_moe_down_tile_pin = pin; }
extern "C" void halo_moe_gemm_pin_down_dec(int pin) {
    g_moe_down_dec_pin = pin < (int)kMoeDownDecMaxG ? pin : (int)kMoeDownDecMaxG;
}
