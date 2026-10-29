/* r3d_gemm_fp8a8_tiled.hip -- gemm_nt_q at prefill M, fp8a8, on a tiled f16 dot.
 *
 * ---- why this row exists ------------------------------------------------------------------------
 *
 * The graph, at the vocabulary head:
 *
 *   W placement: 'blk.0.attn_norm.weight' is grouped with a weight only a host kernel can read, so
 *                the whole group stays in the container's mapping
 *   E rank 0: 'blk.0.attn_norm.weight' was placed on the mapped tier, which reads the container
 *             where it lies, and its stored form is not the container's (widened at load)
 *
 * The chain is arithmetic, not policy: `gemm_nt_q` has no device row at prefill M, so its bands
 * resolve to a host kernel, so the layer's weights stay in the container's mapping, so the layer's
 * NORM weights are grouped with them, so the mapped tier is asked to perform the widening the norms'
 * stored form needs, so startup stops. `--placement all_vram` is already the default and there is no
 * grouping flag. The row that breaks the chain is the one the README names: the tiled
 * quantised GEMM at prefill M. This file is that row.
 *
 * ---- the bands (42 layers + the drafter, per --debug-graph) -------------------------------------
 *
 *   gemm_nt_q N=12288 K= 2048 group=128 dtype=fp8a8 M=2048  x44    the gated MLP, 2 x 6144
 *   gemm_nt_q N= 2560 K= 2048 group=128 dtype=fp8a8 M=2048  x44    qkv: 2048 + 2 x 256
 *   gemm_nt_q N= 2048 K= 6144 group=128 dtype=fp8a8 M=2048  x44    o-proj
 *   gemm_nt_q N= 2048 K= 2048 group=128 dtype=fp8a8 M=2048  x44
 *   gemm_nt_q N= 2048 K=10240 group=128 dtype=fp8a8 M=2048  x3     the drafter
 *
 * M=16 and M=1 are served, by gemm_nt_q_fp8a8_m16. The vocab prefill (N=130560 dtype=w2a8 M=1792) is
 * the 2-bit grid and not this row's. Constraints bind the BAND's keys: dtype in {fp8a8}, group ==
 * 128, M >= 17, K % 128. Below M=17 the m16 rows are the served path and this row says so. The floor
 * is `kMMin`, and it is the sibling row's declared ceiling plus one; the w2a8 arm below carries the
 * same floor, with the reason for it.
 *
 * ---- the law, transcribed -----------------------------------------------------------------------
 *
 * gemm_nt_q's law, in the registry: "The group sub-sum is accumulated from the raw codes and scaled
 * once per group, which is what an int8 or fp8 WMMA kernel does." In code, the reference's
 * quant_dot -- the same body r3d_gemm_fp8a8_gated.hip and r3d_gemm_fp8a16_nt_m1.hip carry:
 *
 *     for g in groups:  sub = 0
 *                       for k in g, ascending:  sub = sub + code_a[m][k] * code_b[n][k]
 *                       acc = acc + (sub * a_scale[m][g]) * b_scale[n/128][g]
 *
 * The codes are the raw E4M3 values, widened exactly into f32; the scale multiplies the finished
 * block ONCE, and the parentheses are the reference's. Operands, from gemm_q_shape, in schema
 * order: a E4M3 {M,K}; a_scale F32 {M, ceil(K/group)}; b E4M3 {N,K} (the checkpoint's own bytes);
 * b_scale BF16 {ceil(N/group), ceil(K/group)} -- f32 at fp8a8f32; y BF16 {M,N}; a_sum ABSENT for
 * fp8a8 because fp8a8 is symmetric; b_ref absent. y is bf16 with one narrowing at the store, f32
 * accumulate throughout.
 *
 * ---- the fold, and where it goes ----------------------------------------------------------------
 *
 * The tile is f16, so the two scale streams enter as f16 multiplies at staging: the activation's
 * `a_scale[m][g]` per staged row, the weight's `b_scale[nb][g]` once a chunk -- TN is 128, the scale
 * plane's row extent, so a tile is one scale-row, and kKC 32 divides 128, so a chunk sits inside one
 * k-block. The products are separable: (a*sa)*(b*sb) = (a*b)*sa*sb, so folding each side is the same
 * sum, formed a different way, and the accumulator stays one f32 register per output -- 64 a lane.
 * The alternative, a per-group partial scaled into the accumulator, needs 64 more registers on an arm
 * that is at 237 VGPRs, and spills.
 *
 * The fold goes through v_mul_f16, never through f16x2_pack. That builtin is v_cvt_pkrtz_f16_f32,
 * rounds toward zero, and r3d_codecs.h restricts it to values f16 represents exactly; a folded
 * product is not one, and on the vocabulary head the truncation measured rel_l2 2.8e-04 against the
 * fold law with 0.53 of outputs inside 2 ulps, where the f16 multiply gives 4.5e-06 and 1.000.
 *
 * Against the reference's law the fold costs the two f16 roundings of the staged values: measured
 * rel_l2 2.77e-04 at the head's band, which is half an f16 ulp (2^-12 = 2.44e-04) and 250x finer than
 * the E4M3 code's own quantum (2^-4). Against a model of the fold itself it is 4.5e-06 -- the dot's
 * fused rounding, the pair-dot folding two products into the accumulator per step. This row's board
 * reports both numbers, per band.
 *
 * ---- the decoder is a table, and that is measured ----------------------------------------------
 *
 * gfx11 has no fp8->f16 instruction: clang lists __builtin_amdgcn_cvt_pk_f16_fp8 and a gfx1102 build
 * that calls it contains zero of them in its ISA, so the builtin is software. r3d_fp8_dequant.h
 * replaces the codec with a 256-entry f16 table in LDS, 512 bytes, eight reads an eight-code group.
 * Measured on the vocabulary head's bands (the A/B is one source, -DR3D_NO_E4M3_TBL for the codec):
 * M=8 speculation 2.500 -> 0.924 ms a token, 2.70x; M=2048 prefill 323.978 -> 322.661 ms, flat; and
 * M=1 12.999 -> 13.293 ms, NOT moved: at M=1 the decoder is not the binding resource, and the
 * limiting access there is the 8-byte code load, not the ALU.
 *
 * ---- what the board measures --------------------------------------------------------------------
 *
 * Five bands, M=2048, GPU 2 (RX 7600, gfx1102, 32 CU), the container managed with its pages
 * prefetched to the card, every number on a placement line:
 *
 *   N=12288 K= 2048   28.171 ms   3659 GFLOP/s   0.0138 ms/token   the gated MLP, x44
 *   N= 2560 K= 2048    5.911 ms   3633 GFLOP/s   0.0029 ms/token   qkv 2048 + 2 x 256
 *   N= 2048 K= 6144   14.699 ms   3506 GFLOP/s   0.0072 ms/token   o-proj, x44
 *   N= 2048 K= 2048    4.381 ms   3922 GFLOP/s   0.0021 ms/token   x44
 *   N= 2048 K=10240   29.798 ms   2883 GFLOP/s   0.0145 ms/token   the drafter, x3
 *
 * 2883-3922 GFLOP/s is 27-36% of this card's dense f16 peak (10.8 TFLOP/s), above the vocabulary
 * head's fp8 3369 and within 4% of its bf16 4067. The bytes floor is 0.051-0.257 ms a band at the
 * 163 GB/s the head's row streams, and 2MNK/(K+N) is ~1900 flops a byte at M=2048: prefill is
 * compute-bound here, as it should be, and these bands are the box's fp8a8 prefill numbers.
 *
 * The floors, as the gate uses them. The fold's is 2^-11 a staged rounding, twice, on S =
 * sqrt(sum_k (a_k b_k)^2) -- the quadrature scale of the DEQUANTISED terms -- so 9.8e-4 S. The
 * store's is bf16's own ulp, 3.9e-3 of the output's binade, and at a bf16 output the store is the
 * binding one: rel_l2 lands at 1.71e-03 against BOTH the reference's per-group law and the fold
 * model, 83% of outputs are bit-exact bf16, 99.93% are within the fold's floor plus one store ulp,
 * and 0 outputs are beyond eight fold floors plus two store ulps, on every band. |y| alone is not a
 * denominator: a sum of 2048 random products is as often near zero as not, and a relative test
 * against it is a coin flip. S is the denominator, and it is why the per-output gate is
 * fold-bound + store-ulp and not a multiple of |y|'s ulp.
 *
 * ---- the open item, with numbers, and what is NOT the kernel -------------------------------------
 *
 * The grid is (ceil(N/128), ceil(M/128)) at one block a CU (64 f32 accumulators a lane). Measured
 * per block, the fast shapes cost a constant 18.4-18.5 us a block (16 tiles: 4.72 ms/256; 20 tiles:
 * 5.911/320). Column-tile counts 19 and 21 do not: at the same M and K, N=2432 (19 tiles) is
 * 64.283 ms and 64.301 ms in two isolated processes (211 us a block, 11.4x) and N=2688 (21 tiles) is
 * 129.383 ms (385 us a block, 20.9x). Deterministic to 0.1%, and it is a grid-shape cliff, not
 * arithmetic -- the board is CLEAN at both. Every column-tile count the graph issues is even (96, 20,
 * 16, 16, 16), so no band the graph runs lands on it; the first thing to try is the block-index
 * swizzle that tile-count cliffs usually answer to, and `nt` streaming loads second.
 *
 * Separately, and named so nobody reads it as the kernel: the qkv shape measures 5.911, 5.919 and
 * 6.002 ms as the FIRST case of a process and 85.669 / 85.847 ms as the SECOND case of the band
 * sequence, same binary, same seed, same grid, CLEAN board both ways, 250 GFLOP/s when slow. A
 * 14.5x deterministic placement artefact of a fixture that allocates and frees a 24 MiB managed
 * container in the case before it; at 0.2 GB/s the slow run is reading b over PCIe, in the mapped
 * tier's class, which is the placement law this file's header states. The row's number for the qkv
 * band is 5.911 ms; the sequence's 85.8 ms is the fixture's, and the engine ships the weights to
 * VRAM at load and reads them 44 times, which is the case the 5.911 measures.
 *
 * ---- where the bytes live, named on every line --------------------------------------------------
 *
 * The head's bands, one kernel, three placements, measured: VRAM 13.6 ms a token; the mapped tier
 * (hipHostMalloc) 643 ms; managed and left alone 2071 ms, 122 MB/s, because managed pages do not
 * migrate for a read-mostly device access on this box. This row's operands are host-readable for the
 * same reason the head's are -- the shim reads the container's header on the host, the kernel reads
 * the same bytes on the card -- so the board allocates with hipMallocManaged and prefetches, and says
 * which placement produced which number. Plain malloc faults 700.
 *
 * ---- what is refused, and where -----------------------------------------------------------------
 *
 * The shim, at its own lines, each naming its fix: n_t < 5 (E_INVAL); a_sum present on an fp8a8 band
 * (E_DTYPE -- fp8a8 is symmetric, a zero belongs to an asymmetric grid, and that is
 * r3d_gemm_nt_q_w2a8.hip); a container carrying a zero plane (E_DTYPE, the same reason); planes that
 * are not codes + scale, a codes plane tiled by anything but one code a cell, sub-byte codes
 * (w4a16's ground), e5m2 and the rest, a scale that is a table, an integer scale (E_DTYPE, each its
 * own line); a scale block that is not {group, group} (E_SHAPE); a plane that does not cover the
 * band, ceil() arithmetic included (E_SHAPE); a plane whose row_bytes is shorter than its row
 * (E_STRIDE); dtype of a not E4M3, a_scale not f32, b codes not E4M3, b_scale not bf16/f16/f32, y
 * not bf16 (E_DTYPE); M < 17 (E_UNSUPPORTED: gemm_nt_q_fp8a8_m16 serves that); group != 128
 * (E_SHAPE: the fold indexes 128 and kKC 32 divides it); K % 128 (E_SHAPE); pitches that a vector
 * load cannot walk (E_STRIDE); addresses not 16-byte aligned (E_ALIGN); absurd shapes (E_INVAL).
 * The kernel's shape rule is stated here, because "shape this kernel does not serve" is a coverage
 * claim, and coverage is caught by the fixture, not the board.
 * The tile is 128 wide, so a ragged N is covered by ceil(): the last column tile stages the columns
 * past N as zeros, writes the columns it owns and guards its stores column by column -- N needs NO
 * divisibility, and `N % 128` is not a band key gemm_nt_q carries (r3d_rows.cpp's pGemmQ is M, N,
 * K, group, dtype, pb1, pb2, split_n), so a check that wanted one could only ever refuse a shape
 * the row advertised. M is the same in rows: ceil(M/128) row-tiles, rows past M stage zeros, write
 * nothing, and M=65, 129, 258 and 512 -- the 27B fixture's geometries -- are served.
 * K % 128 IS a rule, and it is the kernel's and not a coverage one: a 32-code chunk must sit inside
 * one scale group (kKC 32 divides the group 128) and the scale plane is indexed by group, so a
 * ragged K reads the scale plane past its columns. The scale plane is read at ceil(N/128) x
 * ceil(K/128), exactly as the reference's `scaled_rows` covers it; the codes plane at rows == N,
 * cols == K. Those two are equalities because that IS the reference's coverage rule, and every
 * other shape test on this path is a covering one.
 *
 * And the keys, the same class of rule: the band's shape arrives as M, N, K -- gemm_q_shape reads
 * exactly those and pGemmQ names them -- and n_vocab and n_embd are logits_gemm's keys, not this
 * op's: read them and N and K are 0, the axis test fires, and a row whose constraints match the
 * bands refuses them all at launch, invisible at planning, because the graph reports constraint
 * matches and never asks the shim. N and K therefore fall back to r3d_dim(B, 0) and r3d_dim(B, 1)
 * -- the weight IS [N, K], so when the band names no key the operand answers, and the operand is
 * the authority the engine filled.
 *
 * The launcher's private codes are -1 a null plane, -2 a shape, -3 a pitch, -4 HIP, -5 an address,
 * and the driver names them apart from the RAD codes, which collide numerically.
 *
 * ---- relayout: none, and it is a claim ----------------------------------------------------------
 *
 * The codes plane is read row-major [N, K], the order the container stores it in and the order
 * libref reads. libr4d keeps fp8a8 weights in WMMA fragment order (r4d.fp8.w.frag64,
 * r3d_gemm_fp8a8_gated.hip's header): the two readings are ~1.35 apart, a relative error, not noise.
 * So this row carries no relayout and no unrelayout, and the hook is a reader, not a mover -- the
 * same claim the gated row makes, for the same reason.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_fp8_dequant.h"
#include "r3d_tiling.h"          /* the LDS row law: srow, srow_pair */
#include "r3d_launch.h"
#include "rad_planes.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kTM           = 128;                     /* rows a tile */
constexpr int kTN           = 128;                     /* columns a tile: one scale-row */
constexpr int kKC           = 32;                      /* k a chunk: divides the group */
constexpr int kGroup        = 128;                     /* R3D_FP8_BLOCK: the band's group */
/* The complement of gemm_nt_q_fp8a8_m16's declared ceiling, read from the row table: its name
 * says M <= 16, so this floor is 17 and M=17..64 is covered. M=17..64 is a batched decode step,
 * the most common shape a serving card sees; a floor above it leaves those bands to the host --
 * 209 bands a pass, six of them -- and one host band is what drags the whole model into one host
 * run. The tile is 128 rows, so a band of 17 computes 128: 2x wasted at M=64, 7.5x at M=17, and
 * 1.8 TFLOP/s on the card is still two orders above the host loop. */
constexpr int kMMin         = 17;

/* LDS row: 8-element granularity keeps the swizzle uint4-aligned, the rule every tiled row here
 * follows (r3d_vocab_bf16.hip's srow, the same function for the same reason). */
/* srow / srow_pair: r3d_tiling.h, the family's law in one place. */

/* The staged side of the same family, typed next to the row law so a kernel's staging and its
 * swizzle are read together. u16v is a 16-byte streamed load -- 32 U4 codes, 16 E4M3 codes, and
 * the uint4 the tile is written in -- memcpy'd in, never cast: an unaligned cast is UB and the
 * compilers say so in bytes of spill. u8v, its 8-byte sibling (8 E4M3 codes, 32 U2 codes, the f32
 * pair a scale row is read in), is declared with the streamed-load helper below.
 *
 * The scale row, which is why the two arms read a different number of scalars a staged tile:
 *   fp8a8  b_scale's block is {group, group} = {128, 128}, the block the shape function names, and
 *          the tile is 128 columns wide, so a staged chunk of 128 rows shares ONE scale: a single
 *          scalar read a tile, indexed by (n0/128, k0/128).
 *   w2a8   the grid's block is {1, group}, the same rule that fixes the scale's dtype at f16, so
 *          the tile's 128 rows are 128 scale rows: 128 scalar reads a chunk, and
 *          the zero rides with them, 128 more. That is the decode row's pattern -- its scale_at per
 *          staged 8-group IS the {1, group} case -- and it is kept: the scale is read where the grid
 *          puts it, not where the tile would like it. A staged row's pair is (scale, zero) and the
 *          value it stages is (code - zero) * scale, in f16 arithmetic, one v_mul_f16 and one
 *          v_sub_f16 a code. */
typedef uint32_t __attribute__((ext_vector_type(4))) u16v;   /* a 16-byte streamed load */

/* The two operands' scale planes, read the way the container hands them. */
struct ActQ {
    const float* scale;          /* a_scale, f32: [M][ceil(K/group)], or [M] with gmode 0 */
    long long row;               /* elements a row: ceil(K/group) in group mode, 1 in row mode */
    int gmode;                   /* 1: one scale per (row, group) -- `a_scale[m][g]`;
                                  * 0: one per ROW -- `a_scale[m]`, the ZP_I8_ROW form */
};

struct WQ {
    const uint8_t* codes;        /* b's codes plane, [N][K] */
    const uint8_t* scale;        /* b_scale's plane: the {128,128} block, or the {1,group} row */
    const uint8_t* zero;         /* the asym grid's zero plane, one per (row, group); null if none */
    long long code_row;          /* bytes a codes row */
    long long scale_row;         /* bytes a scale row */
    long long zero_row;          /* bytes a zero row */
    int sdt;                     /* 0 bf16, 1 f16, 2 f32 */
    int cdt;                     /* 0 E4M3, 1 int8, 2 U2, 3 U4 */
    int srows;                    /* 0: scale rows are 128-blocks (fp8a8); 1: scale rows are rows (w2a8, w4a8_asym) */
};

__device__ __forceinline__ int i8_dw(int L)
{
    return (L & ~3) | ((L & 1) << 1) | ((L & 2) >> 1);
}

/* (code - zero) * scale, in f16 arithmetic, for a sub-byte code read through the block's table:
 * the table holds f16(i) for the grid's code values, so the code's decode is one LDS read, the
 * scale one v_mul_f16 and the zero one v_sub_f16. This IS the reference's `sub(g) = is(g) -
 * z[n][g] * a_sum[g][m]`, formed per element: sum over a group of (a * (q - z) * sb) is the
 * reference's bracketed (sum a*q - z * sum a) * sb, term for term. */
__device__ __forceinline__ uint32_t fold_pair_q(uint32_t codepair, const uint16_t* tbl,
                                                _Float16 s, _Float16 zf, int bits)
{
    const uint32_t m = (1u << bits) - 1u;
    /* The table entry is an f16 BIT PATTERN, so it is bit_cast, not converted: a converting cast
     * stages 16384 for the code 0x4000, where the value is zero (the board's number was 217). The
     * zero comes off BEFORE the scale -- the grid's zero is an integer code and code minus zero is
     * exact in f16 for a 4-bit code and a U8 zero, so the staged value costs ONE rounding (the
     * multiply) where (code*s) - (z*s) costs two and cancels. */
    const _Float16 a = (__builtin_bit_cast(_Float16, tbl[codepair & m]) - zf) * s;
    const _Float16 b = (__builtin_bit_cast(_Float16, tbl[(codepair >> bits) & m]) - zf) * s;
    return (uint32_t)__builtin_bit_cast(uint16_t, a) |
           ((uint32_t)__builtin_bit_cast(uint16_t, b) << 16);
}

__device__ __forceinline__ float scale_w(const WQ h, int nb, int g)
{
    const uint8_t* p = h.scale + (long long)nb * h.scale_row + (long long)g * (h.sdt == 2 ? 4 : 2);
    if (h.sdt == 2) { float f; __builtin_memcpy(&f, p, 4); return f; }
    const uint16_t u = *(const uint16_t*)p;
    return h.sdt == 1 ? f16_to_f32(u) : bf16_to_f32(u);
}

/* The scale of (row n, group g), in the grid the plane is laid out in. This one line is the whole
 * difference between the two grids' scale rows, so it is written once and named for the ROWS:
 *   fp8a8  b_scale's rows are 128-row blocks ({group, group}) -> the tile's row n lives in block
 *          n>>7, one scalar serves the tile's 128 rows.
 *   w2a8   b_scale's rows are the weight's rows ({1, group}) -> row n, 128 scalars a staged chunk.
 * The srows flag must match the grid: a scale read from the wrong row is not noise, it is a
 * systematic few percent -- the head's bands at rel_l2 1.6e-02 against a fold floor of 1.8e-03 --
 * and it passes any gate set loosely enough. */
__device__ __forceinline__ float scale_at_w(const WQ h, int n, int g)
{
    return scale_w(h, h.srows ? n : (n >> 7), g);
}

__device__ __forceinline__ float zero_at_w(const WQ h, int n, int g)
{
    if (!h.zero) return 0.0f;
    return (float)h.zero[(long long)n * h.zero_row + g];   /* U8: the plane's practical form */
}

/* quant_act_i8's step order: column c sits at (c & ~12) | ((c & 4) << 1) | ((c & 8) >> 1), the
 * swap of bits 2 and 3, which is its own inverse. Bits 2 and 3 of a column are
 * bits 0 and 1 of its DWORD index, so a dword's four codes stay four contiguous bytes and the
 * staging reads a 16-code run's dwords in the order 0, 2, 1, 3. Nothing straddles a dword. */


/* code * scale, in f16 arithmetic: one v_mul_f16, round to nearest, subnormals kept. NOT
 * f16x2_pack, which truncates -- see the header and r3d_codecs.h. */
__device__ __forceinline__ uint32_t fold_pair(uint32_t w, float sc)
{
    const uint16_t lo16 = (uint16_t)(w & 0xffffu), hi16 = (uint16_t)(w >> 16);
    const _Float16 a = __builtin_bit_cast(_Float16, lo16);
    const _Float16 b = __builtin_bit_cast(_Float16, hi16);
    const _Float16 s = (_Float16)sc;
    const _Float16 rl = a * s, rh = b * s;
    return (uint32_t)__builtin_bit_cast(uint16_t, rl) |
           ((uint32_t)__builtin_bit_cast(uint16_t, rh) << 16);
}

/* Eight codes -> eight f16 pairs, through the table. int8 codes take the codec's hardware convert,
 * which is two v_cvt_f32_i8 and a pack; there is no such instruction for E4M3. */
__device__ __forceinline__ void codes8(const uint2 c, int cdt, const uint16_t* tbl, uint32_t* w)
{
    if (!cdt) { r3d_e4m3x8_ld(tbl, c, w); return; }
    const uint32_t d[2] = { c.x, c.y };
#pragma unroll
    for (int j = 0; j < 4; ++j)
        w[j] = i8x2_to_f16x2((d[j >> 1] >> (16 * (j & 1))) & 0xffffu);
}

template <int NT>
__device__ __forceinline__ uint2 loadc8s(const uint8_t* p)
{
    typedef uint32_t __attribute__((ext_vector_type(2))) u8v;
    u8v v;
    if (NT) v = __builtin_nontemporal_load((const u8v*)p);
    else __builtin_memcpy(&v, p, 8);
    uint2 r;
    __builtin_memcpy(&r, &v, 8);
    return r;
}

/*
 * The tile: 128 rows x 128 columns of y, 256 threads, eight waves, eight outputs a side a lane --
 * 64 f32 accumulators a lane, grid (N/kTN) x ceil(M/kTM). k is walked in chunks of 32; both sides
 * arrive as E4M3 codes, decode through the block's table and leave staging as f16 with their scale
 * folded in, so the dot loop is the vocabulary head's dot loop and nothing in it scales anything.
 */
template <int NT, int ARM, int CDT>   /* CDT: 0 E4M3, 1 int8, 2 U2, 3 U4 */
__global__ void __launch_bounds__(kBlockThreads) R3D_WAVES(2)
fp8a8_tiled_kernel(const uint8_t* __restrict__ a, ActQ aq, const uint8_t* __restrict__ b, WQ h,
                   void* __restrict__ y, int M, int N, int K, int a_ld, int y_ld)
{
    __shared__ __align__(16) uint16_t sA[kTM * kKC];
    __shared__ __align__(16) uint16_t sB[kTN * kKC];
    __shared__ __align__(16) uint16_t sTbl[R3D_E4M3_TBL];

    const int tid = (int)threadIdx.x;
    if (ARM == 0) {
        r3d_e4m3_f16_init(sTbl, tid, kBlockThreads);        /* E4M3 -> f16, the decoder table */
    } else {
        /* The asym grid's codes are unsigned integers: U2 has 4 values, U4 has 16. The same
         * 512 bytes, filled with f16(i) -- exact for every code the grid can hold. */
        for (int i = tid; i < 16; i += kBlockThreads) sTbl[i] = f32_to_f16((float)i);
        __syncthreads();
    }
    const int m0 = (int)blockIdx.y * kTM;
    const int n0 = (int)blockIdx.x * kTN;
    const int rg = tid >> 4, cg = tid & 15;

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;

    const int m_r = m0 + rg * 8, n_c = n0 + cg * 8;
    const int nb = n0 / kGroup;                              /* TN == the scale plane's row */

    /* The asym arm's staged row, one per thread for the whole tile: the B staging's row loop is
     * `nn = tid; nn < kTN; nn += kBlockThreads`, so thread tid stages row n0+tid in every chunk.
     * That is what makes the {1, group} grid's scalars cheap to read: the pair a row needs changes
     * only when the k-group changes (kKC 32 divides kGroup 128, so every fourth chunk), and the
     * load can be issued a chunk ahead, where its round trip overlaps the dot.
     *
     * At the 2-bit head's band M=64 N=12288 K=2048, the pair read inside the fold is 210.835 ms,
     * 15.3 GFLOP/s; read once a group with a chunk's lead, 2.4 ms, 1344 GFLOP/s.
     * The scalar is 128 a chunk -- the grid puts the scale at the row, so that is where it is read,
     * and the decode row's pattern is kept -- with one row per thread there is no second load to
     * overlap it with, and a round trip on the staging critical path is 64 of them a tile, paid in
     * full. The values are identical either way; this is placement of a load, not arithmetic. */
    const int  b_row  = n0 + tid;
    const bool b_has  = (ARM == 1) && (tid < kTN) && (b_row < N);
    _Float16 s_cur = (_Float16)0.0f, zf_cur = (_Float16)0.0f;
    if (b_has) {
        s_cur  = (_Float16)scale_at_w(h, b_row, 0);
        zf_cur = (_Float16)zero_at_w(h, b_row, 0);
    }
    /* NAMED NON-RESULT: the codes stream is NOT limited by loads in flight. A one-chunk lead on
     * the row's codes load -- 8 bytes (U2) or 16 (U4) a row a chunk, issued before the fold --
     * measures 24.430 ms against 24.585 ms at M=64 N=130560 K=2048, 0.6%, bit-identical output.
     * The scalar load is the cost because it sits on the critical path with nothing to overlap; the
     * stream is not waiting for latency, it is paying sector count: kKC 32 codes a row a chunk is
     * 8 bytes drawn from a 512-byte-strided plane, so every row costs a full 64-byte sector and the
     * band moves 8x the bytes it uses. 3.7 GB/s useful, 30 GB/s of sectors, and the card is not
     * near the sector limit: the shape is the limit. The lever is a wider k-chunk (kKC 64 = 16 bytes
     * a row = 4x amplification, kKC 128 = one group a chunk = 2x), paid for in LDS and occupancy --
     * a tiling pass on this arm, not a load lead. */

    for (int k0 = 0; k0 < K; k0 += kKC) {
        const int g = k0 / kGroup;                           /* one group a chunk: 32 | 128 */
        const float sb = scale_w(h, nb, g);                  /* one scale for the whole tile */

#pragma unroll
        for (int j = tid; j < kTM * (kKC / 8); j += kBlockThreads) {
            const int mm = j / (kKC / 8), c8 = j % (kKC / 8);
            const int m = m0 + mm, k = k0 + c8 * 8;
            uint4 staged = make_uint4(0, 0, 0, 0);
            if (m < M) {
                if (ARM == 0) {
                    /* A row's scale, once a staged run: a_scale is (row, group) and the run is
                     * inside one group. */
                    /* The row form contributes no g: `aq.gmode` is 0 and the term is a select, so
                     * one scale a row costs one IADD and no branch in the staging loop. */
                    const float sa = aq.scale[(long long)m * aq.row + (aq.gmode ? g : 0)];
                    uint32_t w[4];
                    codes8(loadc8s<NT>(a + (long long)m * a_ld + k), 0, sTbl, w);
                    staged = make_uint4(fold_pair(w[0], sa), fold_pair(w[1], sa),
                                        fold_pair(w[2], sa), fold_pair(w[3], sa));
                } else {
                    /* The w2a8 activation: int8 codes in quant_act_i8's step order, read four codes
                     * a dword at the permuted dword address, and ONE scale per ROW (ZP_I8_ROW:
                     * quant_act_i8 scales a whole row, not a group). */
                    const float sa = aq.scale[(long long)m * aq.row];
                    const int L0 = k >> 2;
                    const uint8_t* ap = a + (long long)m * a_ld;
                    const uint32_t d0 = *(const uint32_t*)(ap + (i8_dw(L0) << 2));
                    const uint32_t d1 = *(const uint32_t*)(ap + (i8_dw(L0 + 1) << 2));
                    const uint32_t d[2] = { d0, d1 };
                    uint32_t w[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                        w[q] = i8x2_to_f16x2((d[q >> 1] >> (16 * (q & 1))) & 0xffffu);
                    staged = make_uint4(fold_pair(w[0], sa), fold_pair(w[1], sa),
                                        fold_pair(w[2], sa), fold_pair(w[3], sa));
                }
            }
            *(uint4*)&sA[srow(mm, c8, kKC)] = staged;
        }
        if (ARM == 0) {
#pragma unroll
            for (int j = tid; j < kTN * (kKC / 8); j += kBlockThreads) {
                const int nn = j / (kKC / 8), c8 = j % (kKC / 8);
                const int n = n0 + nn, k = k0 + c8 * 8;
                uint4 staged = make_uint4(0, 0, 0, 0);
                if (n < N) {
                    uint32_t w[4];
                    codes8(loadc8s<NT>(h.codes + (long long)n * h.code_row + k), h.cdt, sTbl, w);
                    staged = make_uint4(fold_pair(w[0], sb), fold_pair(w[1], sb),
                                        fold_pair(w[2], sb), fold_pair(w[3], sb));
                }
                *(uint4*)&sB[srow(nn, c8, kKC)] = staged;
            }
        } else {
            /* The sub-byte weight, w2a8 (U2, four codes a byte, a row is K>>2 bytes) and
             * w4a8_asym (U4, two a byte, K>>1). A lane takes the whole 32-code chunk in one load:
             * 8 bytes of U2, 16 of U4, k0 is a multiple of 128 so both are aligned. The scale AND
             * the zero are per (row, group) -- the grid's block is {1, group} -- so every staged
             * row reads its own pair of scalars: the decode row's pattern, 128 scalar reads a chunk
             * where the fp8a8 arm reads one. */
            constexpr int BITS = (CDT == 2) ? 2 : 4;
            constexpr int FIELD = 2 * BITS;                 /* bits a pair of codes occupies */
#pragma unroll
            for (int nn = tid; nn < kTN; nn += kBlockThreads) {
                const int n = n0 + nn;
                uint4 st[4] = { make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0),
                                make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0) };
                if (n < N) {
                    const _Float16 s = s_cur, zf = zf_cur;   /* the group's pair, prefetched */
                    if (b_has && k0 + kKC < K && (k0 + kKC) / kGroup != g) {
                        const int gn = (k0 + kKC) / kGroup;
                        s_cur  = (_Float16)scale_at_w(h, b_row, gn);
                        zf_cur = (_Float16)zero_at_w(h, b_row, gn);
                    }
                    uint32_t d[4] = { 0, 0, 0, 0 };
                    const uint8_t* cp = h.codes + (long long)n * h.code_row
                                        + (k0 >> (CDT == 2 ? 2 : 1));
                    if (CDT == 2) { uint2 v; __builtin_memcpy(&v, cp, 8); d[0] = v.x; d[1] = v.y; }
                    else { u16v v; __builtin_memcpy(&v, cp, 16);
                           d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = v.w; }
                    uint32_t w[16];
#pragma unroll
                    for (int q = 0; q < 16; ++q) {
                        /* Pair q of the chunk sits at bit FIELD*q of the row: a 4-bit field for U2
                         * (eight to a dword), an 8-bit field for U4 (four to a dword). Neither
                         * straddles a dword, which is why the field is read from one word. */
                        const int bit = q * FIELD;
                        w[q] = fold_pair_q((d[bit >> 5] >> (bit & 31)) & ((1u << FIELD) - 1u),
                                          sTbl, s, zf, BITS);
                    }
                    st[0] = make_uint4(w[0], w[1], w[2], w[3]);
                    st[1] = make_uint4(w[4], w[5], w[6], w[7]);
                    st[2] = make_uint4(w[8], w[9], w[10], w[11]);
                    st[3] = make_uint4(w[12], w[13], w[14], w[15]);
                }
#pragma unroll
                for (int q = 0; q < 4; ++q) *(uint4*)&sB[srow(nn, q, kKC)] = st[q];
            }
        }
        __syncthreads();

#pragma unroll
        for (int kc = 0; kc < kKC; kc += 2) {
            uint32_t af[8], bf[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                af[i] = *(const uint32_t*)&sA[srow_pair(rg * 8 + i, kc, kKC)];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                bf[j] = *(const uint32_t*)&sB[srow_pair(cg * 8 + j, kc, kKC)];
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] = dot2_f16x2(af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

    /* y is bf16: one narrowing, 16-byte runs of eight columns. Rows past M write nothing. */
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m = m_r + i;
        if (m >= M) break;
        uint16_t* row = (uint16_t*)y + (long long)m * y_ld;
        if (n_c + 8 <= N) {
            uint32_t p[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                p[j] = (uint32_t)f32_to_bf16(acc[i][2 * j]) |
                       ((uint32_t)f32_to_bf16(acc[i][2 * j + 1]) << 16);
            *(uint4*)(row + n_c) = make_uint4(p[0], p[1], p[2], p[3]);
        } else {
            /* The ragged N tail, unrolled: a dynamic index into acc[i] would put the 64
             * accumulators in local memory, which is the trap this family of kernels documents. */
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (n_c + j < N) row[n_c + j] = f32_to_bf16(acc[i][j]);
        }
    }
}

/* ------------------------------------------------------------------ the launcher ------------------ */
/* The launcher speaks the internal codes the shim maps (-3 stride, -4 device, -5 align), so a
 * refusal here prints what it saw and returns -3, not the ABI code: the shim would translate -4 into
 * E_DEVICE and the message the user reads would name the wrong thing. */
#define L_STR(why, a1, v1, a2, v2)                                                    \
    do {                                                                              \
        if (r3d_trace_on())                                                           \
            std::fprintf(stderr, "[r3d] %s:%d %s: %s = %lld, %s = %lld\\n",           \
                         __FILE__, __LINE__, (why), (a1), (long long)(v1), (a2),      \
                         (long long)(v2));                                            \
        return R3D_NO(-3);                                                            \
    } while (0)

int launch(long av, long asv, long bvc, long bsv, long yv, int M, int N, int K, int group,
           int a_ld, long long b_row, int as_ld, long long bs_row, int y_ld, int nt, long stream,
           int sdt, int cdt, long zv = 0, long long z_row = 0, int srows = 0, int arm = 0)
{
    if (!av || !asv || !bvc || !bsv || !yv) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (group != kGroup) return -2;              /* the fold indexes 128; kKC 32 divides it */
    if (K % kGroup) return -2;                   /* a ragged k-block reads past the scale plane */
    if (nt < 0 || nt > 1) return -2;
    if (a_ld % 8) L_STR("a pitch: 8-byte code loads, 8 codes a lane", "a_ld", a_ld, "K", K);
    if (b_row & 7) L_STR("b row pitch is not 8-aligned", "b_row", b_row, "N", N);
    /* The activation's scale row, which is a DIFFERENT thing in the two arms: fp8a8 carries one
     * scale per (row, group) so the row is ceil(K/group) columns; w2a8 and w4a8_asym scale the
     * whole row (ZP_I8_ROW: quant_act_i8's law), so it is one column a row and as_ld 1 is the
     * legal pitch, not a broken one. The check is per arm for the same reason the staging is. */
    /* The activation's scales arrive in two forms, both the reference's: one per (row, group) at
     * width ceil(K/group), and one per ROW at width 1 (the ZP_I8_ROW form -- which is what a per-row
     * quantiser produces, and what this checkpoint's plan binds for the served model). Row mode is
     * `as_ld == 1` and the kernel drops the group term. */
    if (arm == 0 && as_ld != 1 && as_ld < (K + kGroup - 1) / kGroup) L_STR(
            "a_scale: neither one per row nor one per (row, group)", "as_ld", as_ld,
            "ceil(K/group)", (K + kGroup - 1) / kGroup);
    if (arm == 1 && as_ld < 1) return -3;
    if (y_ld % 8) L_STR("y pitch: 16-byte bf16 store runs", "y_ld", y_ld, "N", N);
    if (a_ld < K) L_STR("a pitch shorter than K", "a_ld", a_ld, "K", K);
    if (y_ld < N) L_STR("y pitch shorter than N", "y_ld", y_ld, "N", N);
    if ((av & 7) || (bvc & 7) || (yv & 7) || (asv & 3) || (bsv & 3)) return -5;
    if (bs_row & (sdt == 2 ? 3 : 1)) return -3;  /* a scale row is 4 bytes a column for f32, 2 for
                                                    bf16/f16: ceil(K/128) = 3 columns is 6 bytes. */
    if (arm == 1) {
        if (!(arm == 1 && (cdt == 2 || cdt == 3))) return -2;
        if (!zv) return -1;                      /* the asym grid has no spelling without a zero */
        if (z_row < 1) return -3;
        if (zv & 1) return -5;
        if (cdt == 3 && (bvc & 15)) return -5;    /* a 16-byte U4 load; U2 loads 8 */
    }

    const dim3 blk(kBlockThreads);
    const dim3 grd((unsigned)((N + kTN - 1) / kTN), (unsigned)((M + kTM - 1) / kTM));
    ActQ aq{ (const float*)(uintptr_t)asv, (long long)as_ld, as_ld != 1 };
    WQ h{ (const uint8_t*)(uintptr_t)bvc, (const uint8_t*)(uintptr_t)bsv,
          (const uint8_t*)(uintptr_t)zv, b_row, bs_row, z_row, sdt, cdt, srows };
    hipStream_t s = (hipStream_t)(uintptr_t)stream;
#define R3D_RUN(ARM, CD)                                                        \
    do {                                                                        \
        if (nt) fp8a8_tiled_kernel<1, ARM, CD><<<grd, blk, 0, s>>>(             \
            (const uint8_t*)(uintptr_t)av, aq, (const uint8_t*)(uintptr_t)bvc,   \
            h, (void*)(uintptr_t)yv, M, N, K, a_ld, y_ld);                       \
        else fp8a8_tiled_kernel<0, ARM, CD><<<grd, blk, 0, s>>>(                \
            (const uint8_t*)(uintptr_t)av, aq, (const uint8_t*)(uintptr_t)bvc,   \
            h, (void*)(uintptr_t)yv, M, N, K, a_ld, y_ld);                       \
    } while (0)
    if (arm == 0) { if (cdt == 1) R3D_RUN(0, 1); else R3D_RUN(0, 0); }
    else          { if (cdt == 3) R3D_RUN(1, 3); else R3D_RUN(1, 2); }
#undef R3D_RUN
    const hipError_t le = hipGetLastError();
    if (le != hipSuccess) {
        if (r3d_trace_on())
            std::fprintf(stderr, "[launch] arm=%d cdt=%d nt=%d grid(%u,%u) block(%u) M=%d N=%d K=%d: %s\n",
                         arm, cdt, nt, grd.x, grd.y, kBlockThreads, M, N, K, hipGetErrorString(le));
        return -4;
    }
    return 0;
}

/* libref's float_scale_dt and enc_scaled_rows, inlined: libref keeps them static inline in its own
 * header, so a row that transcribes a layout hook transcribes its helpers with it. Checks and
 * order unchanged. */
bool float_scale_dt(uint32_t dt) { return dt == RAD_BF16 || dt == RAD_F16 || dt == RAD_F32; }

bool enc_scaled_rows(const RadEncoding* enc, const int* sel, int n)
{
    if (!rad_enc_is(enc, "affine") || enc->transform[0] || enc->n_planes != 2) return false;
    if (n != 2 || !sel) return false;                     /* sel_roles(enc, sel, n, "codes","scale") */
    const char* want[2] = { "codes", "scale" };
    for (int i = 0; i < 2; ++i) {
        if (sel[i] < 0 || sel[i] >= enc->n_planes) return false;
        if (std::strncmp(enc->plane[sel[i]].role, want[i], RAD_ENC_STR) != 0) return false;
    }
    const RadEncPlane& c = enc->plane[sel[0]];
    const RadEncPlane& sc = enc->plane[sel[1]];
    return c.kind == RAD_PLANE_TILED && c.block[0] == 1 && c.block[1] == 1 &&
           (c.dtype == RAD_F8E4M3 || c.dtype == RAD_I8) && sc.kind == RAD_PLANE_TILED &&
           float_scale_dt(sc.dtype);
}

}  // namespace

/* ==================================================================================================
 * The layout hook: b (operand 2) as codes + one scale.
 *
 * `ref_layout_gemm_q` is the w2a8/w4a8 grid's hook: it is `layout_scaled(operand, 3, -1, ...)` over
 * `sel_roles("scale","zero")` with block {1, group}, and it reads `zp_grid(dtype)` first -- and
 * zp_grid has no "fp8a8" entry (it lists w2a8, w4a8_asym, w4a16_asym, w5a16, w4a8i8_asym, w5a8i8),
 * so the reference's own gemm_nt_q hook ANSWERS NOTHING for dtype=fp8a8: it returns E_DTYPE, the
 * placement finds no reader, and the group stays mapped. That is the chain's first link, and it is
 * why this hook is the head's class and not that one: two planes, "codes" and "scale", NO zero
 * plane (fp8a8 is symmetric), codes tiled one a cell, the scale on the block the shape function
 * names -- {group, group}. Body: layout_scaled(operand, 2, -1, ...) as ref_layout_head's is.
 * ================================================================================================== */
/* The writer for the layout this hook names -- the same rad_planes_relayout, the same operand guard,
 * the same reason: the plan asks for the container's codes and per-block scales as one operand, and
 * the loader converts at load with the row's help and nobody else's. */
extern "C" int r3d_relayout_gemm_q_fp8_weight(const RadParam* p, int n_p, int operand,
                                              const RadEncoding* enc, const int* sel,
                                              const RadTensor* planes, int n_planes, void* dst,
                                              int64_t dst_bytes)
{
    /* The query and the writer are one plan; see r3d_layout_gemm_q_fp8_weight. */
    const char* dt = r3d_pgets(p, n_p, "dtype");
    if (dt && (std::strcmp(dt, "w2a8") == 0 || std::strcmp(dt, "w4a8_asym") == 0))
        return r3d_relayout_gemm_nt_q_w2a8(p, n_p, operand, enc, sel, planes, n_planes, dst,
                                           dst_bytes);   /* the decode row's plan, one writer */
    if (operand != 2) return R3D_NO(RAD_E_UNSUPPORTED);
    return rad_planes_relayout(enc, sel, planes, n_planes, dst, dst_bytes);
}

/* The grids this row reads behind operand 2, in two branches that never borrow each other's
 * checks. fp8a8 is the symmetric pair (codes, scale) on {1,1} + {group,group}. w2a8 and
 * w4a8_asym are the ZERO-POINT triple (codes, scale, zero) on {1,1} + {1,group} + {1,group}:
 * ref_layout_gemm_q's plane rules with the codes plane riding along, because this row's plan puts
 * all three behind b. The dtype names which grid, and the codes dtype follows it (zp_grid:
 * w2a8 -> U2, w4a8_asym -> U4) -- a grid and a code width that disagree is a container this row
 * cannot read, at its own line. */
/* A layout query answers in the engine's own vocabulary and leaves no trace, and a model that will
 * not load says only that two rows want one weight in two forms. This prints the question and the
 * answer -- operand, band dtype, planes offered, code returned -- under RAD_R3D_TRACE=1. */
static int r3d_layout_trace(const char* who, int operand, const char* dt, int n_planes, int rc) {
    if (r3d_trace_on())
        std::fprintf(stderr, "[r3d] %s: operand %d dtype %s planes %d -> %d\n", who, operand,
                     dt ? dt : "(none)", n_planes, rc);
    return rc;
}

extern "C" int r3d_layout_gemm_q_fp8_weight(const RadParam* p, int n_p, int operand,
                                            const RadEncoding* enc, const int* sel,
                                            const RadTensor* planes, int n, RadLayout* out)
{
    const char* dt = r3d_pgets(p, n_p, "dtype");
    if (dt && (std::strcmp(dt, "w2a8") == 0 || std::strcmp(dt, "w4a8_asym") == 0))
        return r3d_layout_trace("r3d_layout_gemm_q_fp8_weight", operand, r3d_pgets(p, n_p, "dtype"), n, r3d_layout_gemm_nt_q_w2a8(p, n_p, operand, enc, sel, planes, n, out));

    if (n < 2 || operand != 2) return r3d_layout_trace("r3d_layout_gemm_q_fp8_weight", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_UNSUPPORTED);   /* b is operand 2 of gemm_nt_q */

    /* The packed grids ask the DECODE ROW what the container looks like, and get the answer by
     * calling its hook -- `r3d_layout_gemm_nt_q_w2a8`: codes plain on operand 2, the (scale, zero)
     * pair joined on operand 3, the {1, group} block, the f16 scale, the unsigned zero. The engine
     * settles one stored form per weight across every row of the op that reads it and says so at
     * declare time -- `weight 'dspark.draft_head.scale' is wanted in two stored forms:
     * 'rad.planes' by gemm_nt_q_w2a8_m16 and '(as it is)' by gemm_nt_q_w2a8_tiled. A weight is
     * stored once` -- so a tile that wants the trio joined behind `b` is not a different plan, it is
     * a model that will not load. Two rows describing one weight in one function cannot drift; the
     * arm's board measures this arrangement as `decodeform`, 1.660e-03 against the joined fixture's
     * 1.677e-03, the same law's noise. */

    if (!enc_scaled_rows(enc, sel, n)) return r3d_layout_trace("r3d_layout_gemm_q_fp8_weight", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);
    const long long group = r3d_pgeti_or(p, n_p, "group", 0);
    if (group <= 0) return r3d_layout_trace("r3d_layout_gemm_q_fp8_weight", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);                    /* the band names the block */
    if (enc->plane[sel[1]].block[0] != group || enc->plane[sel[1]].block[1] != group)
        return r3d_layout_trace("r3d_layout_gemm_q_fp8_weight", operand, r3d_pgets(p, n_p, "dtype"), n, RAD_E_DTYPE);                                /* {group, group} */
    return r3d_layout_trace("r3d_layout_gemm_q_fp8_weight", operand, r3d_pgets(p, n_p, "dtype"), n, rad_planes_layout(enc, sel, planes, n, out));
}

/* ==================================================================================================
 * The planes' entry: the caller has resolved the two planes and hands their addresses. The band is
 * fp8a8, group 128, y bf16. Codes row-major, scales [ceil(N/128)][ceil(K/128)], read as bf16 --
 * fp8a8's stated dtype; the f32 spelling (fp8a8f32) arrives through the shim, which reads the dtype
 * off the plane. b_row and bs_row of 0 mean the dense pitch.
 * ================================================================================================== */
extern "C" int r3d_gemm_nt_q_fp8_tiled(long a, long a_scale, long b_codes, long b_scale, long y,
                                       int M, int N, int K, int group, int a_ld, long long b_row,
                                       int as_ld, long long bs_row, int y_ld, int nt, long stream)
{
    const int st = launch(a, a_scale, b_codes, b_scale, y, M, N, K, group, a_ld, b_row ? b_row : K,
                          as_ld, bs_row ? bs_row : (long long)((K + kGroup - 1) / kGroup) * 2, y_ld,
                          nt, stream, 0, 0);
    if (st == -1) return R3D_NO(RAD_E_INVAL);
    if (st == -2) return R3D_NO(RAD_E_SHAPE);
    if (st == -3) return R3D_NO(RAD_E_STRIDE);
    if (st == -5) return R3D_NO(RAD_E_ALIGN);
    if (st == -4) return R3D_NO(RAD_E_DEVICE);
    return RAD_OK;
}

/* ==================================================================================================
 * The RadArgs shim. Schema positions: a 0, a_scale 1, b 2, b_scale 3, y 4, a_sum 5, b_ref 6. The
 * placement joins b and b_scale into one container, so 3 is absent and the container's header is
 * read here, on the host; a caller that resolved the planes itself passes 3 and takes the plain
 * path. Same kernel, same law, one branch.
 * ================================================================================================== */

/* RadStream, the ABI's type: RadLaunchFn is int (*)(const RadArgs*, RadStream) and RadStream is
 * void* (rad_abi.h, rad_types.h). A shim that takes a long runs -- same width, same convention --
 * and is a function-pointer type the row table has to be lied to about. The entry below keeps long
 * because it takes addresses as longs; the cast is here, at the boundary. */

/* The plan line's latch: one print per distinct shape this shim sees, keyed on what the line prints
 * -- M, N, K, the activation's rank, the scale's rank and the two pitches -- because the served plan
 * runs this shim 176 times a forward pass and 42 copies of one line are noise. A plain static array
 * with no lock: the worst a race does is print a shape twice, and the flag-off path never reaches
 * it. The sibling decode row (gemm_nt_q_fp8a8_m16) has no such line: the shapes below M=17 are its
 * served path, and the ones above it are what this line is for. */
struct GemmPlanKey { long long M, N, K; unsigned a_rank, as_rank; long long a_ld, y_ld; };

static bool gemm_plan_once(long long M, long long N, long long K, unsigned a_rank, unsigned as_rank,
                           long long a_ld, long long y_ld)
{
    static GemmPlanKey seen[96];
    static int n = 0;
    for (int i = 0; i < n; ++i)
        if (seen[i].M == M && seen[i].N == N && seen[i].K == K && seen[i].a_rank == a_rank &&
            seen[i].as_rank == as_rank && seen[i].a_ld == a_ld && seen[i].y_ld == y_ld)
            return false;
    if (n < 96) seen[n++] = GemmPlanKey{M, N, K, a_rank, as_rank, a_ld, y_ld};
    return true;
}

extern "C" int r3d_rad_gemm_nt_q_fp8_tiled(const RadArgs* args, RadStream s)
{
    if (!args || args->n_t < 5) return R3D_NO(RAD_E_INVAL);
    const RadTensor* A = r3d_opt(args, 0);
    const RadTensor* AS = r3d_opt(args, 1);
    const RadTensor* B = r3d_opt(args, 2);
    const RadTensor* BS = r3d_opt(args, 3);
    const RadTensor* Y = r3d_opt(args, 4);
    const RadTensor* ASUM = r3d_opt(args, 5);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(Y)) return R3D_NO(RAD_E_INVAL);

    /* The band's keys, from the schema this row is declared against: gemm_nt_q carries M, N, K,
     * group, dtype (r3d_rows.cpp pGemmQ) -- NOT n_vocab/n_embd, which belong to logits_gemm. Read
     * the wrong keys and N and K are 0, every launch is refused below at the shape check, the op
     * resolves to a host kernel for every band, the planner puts the whole model in one host run,
     * and a norm weight two tiers away refuses to load. Falling back to the weight's own dims keeps
     * a caller that names no N/K working, and says which operand is the authority. */
    const long long M_band = r3d_geti_or(args, "M", 0), N0 = r3d_geti_or(args, "N", 0);
    const long long K0 = r3d_geti_or(args, "K", 0);
    const long long N = N0 > 0 ? N0 : r3d_dim(B, 0);
    const long long K = K0 > 0 ? K0 : r3d_dim(B, 1);
    const long long group = r3d_geti_or(args, "group", 0);
    if (N <= 0 || K <= 0) return R3D_NO(RAD_E_SHAPE);
    if (K > (1 << 24)) return R3D_NO(RAD_E_INVAL);   /* absurd: a 16M-wide K is a 2 GiB code plane per row */
    if (group != kGroup) return R3D_NO(RAD_E_SHAPE);   /* the fold indexes 128 */
    if (K % kGroup) return R3D_NO(RAD_E_SHAPE);
    if (ASUM && r3d_p(ASUM))
        return R3D_NO(RAD_E_DTYPE);       /* fp8a8 is symmetric: a_sum is an asymmetric grid's, and
                                             r3d_gemm_nt_q_w2a8.hip reads that one */

    /* ---- b: a container of codes + scale, or the two planes as tensors -------------------------- */
    const void* bvc = r3d_p(B);
    const void* bsv = nullptr;
    long long b_row = 0, bs_row = 0;
    int sdt = 0, cdt = 0;
    const RadPlanesHeader* ph = (const RadPlanesHeader*)rad_planes_header(bvc);
    if (ph) {
        /* find answers with the bytes and points into the container; no copy. */
        const RadPlanesEntry* ce = nullptr;
        const RadPlanesEntry* se = nullptr;
        if (rad_planes_find(bvc, "zero", nullptr))
            return R3D_NO(RAD_E_DTYPE);   /* codes+scale+zero is gemm_nt_q's asymmetric grid: w2a8 */
        if (ph->n_planes != 2) return R3D_NO(RAD_E_DTYPE);
        const void* cb = rad_planes_find(bvc, "codes", &ce);
        const void* sb = rad_planes_find(bvc, "scale", &se);
        if (!cb || !sb) return R3D_NO(RAD_E_DTYPE);      /* a joined weight that is not codes + scale */
        if (ce->kind != RAD_PLANE_TILED || ce->block[0] != 1 || ce->block[1] != 1)
            return R3D_NO(RAD_E_DTYPE);                  /* codes: one a cell, tiled plain */
        if (ce->dtype != RAD_F8E4M3 && ce->dtype != RAD_I8) {
            if (rad_dtype_bytes(ce->dtype, 8) < 8)
                return R3D_NO(RAD_E_DTYPE);              /* sub-byte codes: w4a16's ground */
            return R3D_NO(RAD_E_DTYPE);                  /* e5m2 and the rest: no decoder here */
        }
        if (se->kind != RAD_PLANE_TILED) return R3D_NO(RAD_E_DTYPE);
        if (se->dtype == RAD_BF16) sdt = 0;
        else if (se->dtype == RAD_F16) sdt = 1;
        else if (se->dtype == RAD_F32) sdt = 2;
        else return R3D_NO(RAD_E_DTYPE);                 /* an integer scale is not float_scale_dt */
        if (se->block[0] != group || se->block[1] != group)
            return R3D_NO(RAD_E_SHAPE);   /* per-(row, group) is gemm_nt_q's w2a8 grid; per-row is
                                             not cover's indexing of the plane */
        bvc = cb; bsv = sb; b_row = ce->row_bytes; bs_row = se->row_bytes;
        cdt = ce->dtype == RAD_I8;
        const long long sb_r = (N + group - 1) / group, sb_c = (K + group - 1) / group;
        if (ce->rows != N || ce->cols != K) return R3D_NO(RAD_E_SHAPE);
        if (se->rows != sb_r || se->cols != sb_c) return R3D_NO(RAD_E_SHAPE);
        if (ce->row_bytes < K) return R3D_NO(RAD_E_STRIDE);   /* a packed row is not this form */
        if (se->row_bytes < sb_c * (sdt == 2 ? 4 : 2)) return R3D_NO(RAD_E_STRIDE);
    } else {
        if (!BS || !r3d_p(BS))
            return R3D_NO(RAD_E_DTYPE);   /* a weight that is neither a container nor a scale
                                             tensor: gemm_nt_q's shape hook names the two */
        b_row = K;
        const long long sb_r = (N + group - 1) / group, sb_c = (K + group - 1) / group;
        /* b_scale as a CONTAINER -- the codes plain on b, the scale joined behind b_scale. The
         * sibling decode row uses that spelling for the shared weight, and
         * r3d_relayout_gemm_q_fp8_weight writes it for a band whose encoding names its container at
         * operand 3, so the plan can hand it here. The header is read on the host, where it lives
         * (the same call b's arm makes above); the kernel gets the scale plane's BYTES and the row
         * pitch off the ENTRY, which is the identical answer the two-tensor spelling gives.
         * A `zero` plane stays refused: fp8a8's activation quantiser is symmetric and this arm has
         * no zero to fold -- the reference refuses the grid for it outright -- so a row that
         * computed through a zero plane would be inventing a number the oracle has no law for. The
         * asymmetric grid is gemm_nt_q's w2a8 form and r3d_gemm_nt_q_w2a8 owns that spelling. */
        const RadPlanesHeader* sph = (const RadPlanesHeader*)rad_planes_header(r3d_p(BS));
        if (sph) {
            if (rad_planes_find(r3d_p(BS), "zero", nullptr))
                return R3D_NO(RAD_E_DTYPE);   /* codes + scale + zero: r3d_gemm_nt_q_w2a8's grid,
                                                 not this arm's -- nor libref's */
            const RadPlanesEntry* se = nullptr;
            const void* sb = rad_planes_find(r3d_p(BS), "scale", &se);
            if (!sb || !se)
                return R3D_NO(RAD_E_DTYPE);   /* a joined stream with no `scale` plane:
                                                 r3d_relayout_gemm_q_fp8_weight is its writer */
            if (se->kind != RAD_PLANE_TILED) return R3D_NO(RAD_E_DTYPE);
            if (se->dtype == RAD_BF16) sdt = 0;
            else if (se->dtype == RAD_F16) sdt = 1;
            else if (se->dtype == RAD_F32) sdt = 2;
            else return R3D_NO(RAD_E_DTYPE);  /* an integer scale is not float_scale_dt */
            if (se->block[0] != group || se->block[1] != group)
                return R3D_NO(RAD_E_SHAPE);   /* per-row scales are the w2a8 grid; the 128x128
                                                 quantiser writes one scale a (128 rows x 128 codes) */
            if (se->rows != sb_r || se->cols != sb_c) return R3D_NO(RAD_E_SHAPE);
            if (se->row_bytes < sb_c * (sdt == 2 ? 4 : 2)) return R3D_NO(RAD_E_STRIDE);
            bsv = sb;
            bs_row = se->row_bytes;
        } else {
            bsv = r3d_p(BS);
            long long bs_rows = 0;
            const long long bs_pitch = r3d_rows_pitch(BS, sb_c, &bs_rows);
            if (bs_pitch < 0 || bs_rows < sb_r) return R3D_NO(RAD_E_STRIDE);
            if (BS->dtype == RAD_BF16) sdt = 0;
            else if (BS->dtype == RAD_F16) sdt = 1;
            else if (BS->dtype == RAD_F32) sdt = 2;
            else return R3D_NO(RAD_E_DTYPE);
            if (r3d_dim(BS, 0) < sb_r || r3d_dim(BS, 1) < sb_c) return R3D_NO(RAD_E_SHAPE);
            bs_row = bs_pitch * (sdt == 2 ? 4 : 2);      /* the kernel walks the plane's bytes */
        }
        if (B->dtype == RAD_F8E4M3) cdt = 0;
        else if (B->dtype == RAD_I8) cdt = 1;
        else if (rad_dtype_bytes(B->dtype, 8) < 8) return R3D_NO(RAD_E_DTYPE);
        else return R3D_NO(RAD_E_DTYPE);
        if (r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);
        if (r3d_dim(B, 0) != N || r3d_dim(B, 1) != K) return R3D_NO(RAD_E_SHAPE);
    }

    if (A->dtype != RAD_F8E4M3) return R3D_NO(RAD_E_DTYPE);
    if (AS->dtype != RAD_F32) return R3D_NO(RAD_E_DTYPE);
    if (Y->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);   /* gemm_q_shape: y is bf16, one narrowing */
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);

    long long a_rows = 0;
    long long a_ld = r3d_rows_pitch(A, K, &a_rows);
    /* THE STEP'S M IS THE ACTIVATION'S ROW COUNT. The band key `M` is an upper bound, and the
     * engine's own line says so -- its refusal names the step's M against the band's declared bound
     * -- so reading the key as the value would refuse a step shorter than the band. libref derives
     * M from the operands (the plane, not the plan), and the m16 row does it the same way. The key
     * keeps its job as the ceiling the plan promised not to exceed. */
    if (a_ld < K || a_rows < 1) {
        /* A group-quantised activation is the same row-major K codes the kernel wants, described in
         * the rank the quantiser writes them in: [M][ceil(K/group)][group], contiguous, the row
         * exactly K wide. `r3d_rows_pitch` asks for the width by name and a rank-3 plane whose last
         * axis is the group does not answer it, so the plane is asked directly: contiguous and
         * dividing by K is a plane of rows K wide. */
        if (rad_tensor_is_contiguous(A) && K > 0 && r3d_numel(A) % K == 0) {
            const long long r = r3d_numel(A) / K;
            if (r >= 1) { a_ld = K; a_rows = r; }
        }
    }
    if (a_ld < K || a_rows < 1) { r3d_trace_desc("a", A);
        std::fprintf(stderr, "[r3d] a: pitch %lld rows %lld, want pitch >= %lld and at least one row\n",
                     a_ld, a_rows, (long long)K);
        return R3D_NO(RAD_E_STRIDE); }
    const long long M = a_rows;
    if (M_band > 0 && M > M_band) { r3d_trace_desc("a", A);
        std::fprintf(stderr, "[r3d] a: %lld rows in a band that declared M <= %lld\n", M, M_band);
        return R3D_NO(RAD_E_SHAPE); }
    if (M * N > (1ll << 34)) return R3D_NO(RAD_E_INVAL);   /* 32-bit tile indices, at the step's
                                                           * height, not the band's bound */
    if (M < kMMin) return R3D_NO(RAD_E_UNSUPPORTED);   /* gemm_nt_q_fp8a8_m16 serves M <= 16, and
                                                        * this arm's floor is that declared
                                                        * ceiling + 1, read from its array */
    const long long kb = (K + group - 1) / group;
    /* Two forms of the activation's scales, and the plan binds the one its quantiser produces:
     * [M][ceil(K/group)] for a group quantiser, [M] for a per-row one (`a_scale[m]`, the ZP_I8_ROW
     * form). `r3d_rows_pitch(AS, kb, ...)` returns -1 for the rank-1 plane -- the scales do not
     * divide into rows of kb -- so the width asked for depends on the form, which the rank states;
     * both forms must pass, the row form included. */
    const int as_row = (AS->rank <= 1);
    const long long as_w = as_row ? 1 : kb;
    long long as_rows = 0;
    const long long as_ld = r3d_rows_pitch(AS, as_w, &as_rows);
    if (as_ld < as_w || as_rows < M) { r3d_trace_desc("a_scale", AS);
        std::fprintf(stderr, "[r3d] a_scale: want width %lld (rank %d), pitch %lld, rows %lld, M %lld\n",
                     as_w, AS->rank, as_ld, as_rows, M);
        return R3D_NO(RAD_E_STRIDE); }
    long long y_rows = 0;
    const long long y_ld = r3d_rows_pitch(Y, N, &y_rows);
    if (y_ld < N || y_rows < M) { r3d_trace_desc("y", Y);
        std::fprintf(stderr, "[r3d] y: pitch %lld rows %lld, want pitch >= %lld rows >= %lld\n",
                     y_ld, y_rows, (long long)N, M);
        return R3D_NO(RAD_E_STRIDE); }
    if (!r3d_aligned(r3d_p(A), 8) || !r3d_aligned(bvc, 8) || !r3d_aligned(r3d_p(Y), 8))
        return R3D_NO(RAD_E_ALIGN);
    if (!r3d_aligned(r3d_p(AS), 4) || !r3d_aligned(bsv, 4)) return R3D_NO(RAD_E_ALIGN);

    /* ---- the plan line, the GEMM's side of the same instrument ---------------------------------
     *
     * The step's M, off the activation (the header's law: the band key `M` is a BOUND, the plane is
     * the value), with the forms the served plan hands this row -- the codes' rank, pitch and numel,
     * the scale's rank and pitch, the output's pitch. The fixture draws its descriptors from the
     * shape hooks and does not cover the plan's own rank-1 `[M]` scale and rank-3 codes at a
     * M >= 17 launch; this line is the instrument for that form: the step's rows against the
     * arena's padding, ceil(M/16)*16, and the strides the producer and the consumer each read the
     * same plane through.
     *
     * Inert when the flag is unset -- the cached `r3d_trace_on()` every refusal already pays, and
     * the descriptors print through the existing `r3d_trace_desc`. Latched per shape, because this
     * shim runs 176 times a forward pass and 42 copies of one line are noise. Printed AFTER every
     * check, so a refused step stays the refusal lines' business. */
    if (r3d_trace_on() && gemm_plan_once(M, N, K, A->rank, AS->rank, a_ld, y_ld)) {
        std::fprintf(stderr,
                     "[r3d] plan: gemm M=%lld (band M<=%lld) N=%lld K=%lld group=%lld "
                     "a rank=%u numel=%lld stride=[%lld,%lld] pitch=%lld rows=%lld "
                     "a_scale rank=%u numel=%lld pitch=%lld form=%s "
                     "y rank=%u numel=%lld stride=[%lld,%lld] pitch=%lld Mpad=%lld\n",
                     M, M_band, N, K, group, (unsigned)A->rank, r3d_numel(A),
                     r3d_stride(A, 0), r3d_stride(A, 1), a_ld, a_rows, (unsigned)AS->rank,
                     r3d_numel(AS), as_ld, as_row ? "row[M]" : "group[M][kb]", (unsigned)Y->rank,
                     r3d_numel(Y), r3d_stride(Y, 0), r3d_stride(Y, 1), y_ld, (M + 15) / 16 * 16);
        r3d_trace_desc("gemm a", A);
        r3d_trace_desc("gemm a_scale", AS);
        r3d_trace_desc("gemm y", Y);
    }

    const int nt = (int)r3d_geti_or(args, "nt", 0);
    const int st = launch(r3d_addr(A), r3d_addr(AS), (long)(uintptr_t)bvc, (long)(uintptr_t)bsv,
                          r3d_addr(Y), (int)M, (int)N, (int)K, (int)group, (int)a_ld, b_row,
                          (int)as_ld, bs_row, (int)y_ld, nt, (long)s, sdt, cdt);
    if (st == -1) return R3D_NO(RAD_E_INVAL);
    if (st == -2) return R3D_NO(RAD_E_SHAPE);
    if (st == -3) return R3D_NO(RAD_E_STRIDE);
    if (st == -5) return R3D_NO(RAD_E_ALIGN);
    if (st == -4) return R3D_NO(RAD_E_DEVICE);
    return RAD_OK;
}

/* ==================================================================================================
 * The w2a8 arm's shim: gemm_nt_q at dtype=w2a8 (codes U2) and dtype=w4a8_asym (codes U4), the same
 * tile above the decode heights. The band that gates start-up is N=130560 K=2048 group=128
 * dtype=w2a8 at M=1792 -- the drafter's draft head, read three times a step -- and the graph prints
 * it as the last band with no device row. Below kMMin the m16 row (gemm_nt_q_w2a8_m16,
 * C_LE("M", 16) in its constraint array) is the served path; above it the tile. The floor is the
 * sibling's CEILING, read from its constraints, and it is 17: M=17..64 is a batched decode step,
 * the shape a serving card lives in, and 1.8 TFLOP/s of 128-row waste is two orders above the host.
 *
 * The law is the reference's gemm_zp, ZP_I8_ROW arm, and the kernel runs it in a folded form:
 *
 *   q[k]   = the grid's code of weight row n, column k                  U2: bits [2k,2k+2) of k/4
 *   sub(g) = SUM_{k in g} (int)a[m][i8_step_pos(k)] * ((int)q[k] - (int)z[n][g])   integers
 *   acc    = acc + (sub(g) * sa[m]) * sb[n][g]                          g ASCENDING
 *   y[m,n] = bf16(acc)
 *
 * which is the reference's `sub = is - z * a_sum` term for term: sum a*(q - z) = sum a*q - z*sum a,
 * exactly, over the same group. So the staged tile carries (q - z) * sb and the a-side carries
 * a * sa[m], and a_sum -- which exists to take the 128 bias off an integer dot, because this ISA
 * has no int8 dot (r3d_gemm_nt_q_w2a8.hip) -- has no term to take off here: the zero is
 * removed per element, before the dot. a_sum is still REQUIRED of a w2a8 band (gemm_zp reads it and
 * the schema carries it at slot 5; a band without it is not this band) and the refusal is this
 * shim's own line, not the fp8a8 arm's, which REFUSES the same tensor: the two arms' lines are
 * deliberately not shared, and the driver's board fires each at its own arm. The bracketing
 * differs from the reference (a per-group bracket there, one chain here) by the fold's floor, which
 * is what the board measures: rel_l2 against the transcribed body, at a bf16 output, where the
 * store's own ulp is the wider of the two.
 *
 * The activation arrives in quant_act_i8's step order, and its scale is ONE f32 PER ROW (ZP_I8_ROW:
 * quant_act_i8 scales a whole row), not per group like fp8a8's. The scale and the zero are per
 * (row, group): 128 scalar reads a staged chunk where the fp8a8 arm reads one.
 * ================================================================================================== */
extern "C" int r3d_rad_gemm_nt_q_w2a8_tiled(const RadArgs* args, RadStream s)
{
    if (!args || args->n_t < 5) return R3D_NO(RAD_E_INVAL);

    const char* dt = r3d_gets(args, "dtype");
    int cdt = -1;
    if (dt && std::strcmp(dt, "w2a8") == 0) cdt = 2;
    else if (dt && std::strcmp(dt, "w4a8_asym") == 0) cdt = 3;
    if (cdt < 0)
        return R3D_NO(RAD_E_DTYPE);   /* the zero-point grids on an int8 activation; w4a8i8_asym and
                                       * w5a8i8 quantise the activation to int8 with a DIFFERENT
                                       * trunk (ZP_I8_GROUP, no per-row scale) and this arm does not
                                       * read them. */

    const RadTensor* A = r3d_opt(args, 0);
    const RadTensor* AS = r3d_opt(args, 1);
    const RadTensor* B = r3d_opt(args, 2);
    const RadTensor* BS = r3d_opt(args, 3);
    const RadTensor* Y = r3d_opt(args, 4);
    const RadTensor* ASUM = r3d_opt(args, 5);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(Y)) return R3D_NO(RAD_E_INVAL);
    if (!r3d_p(ASUM))
        return R3D_NO(RAD_E_DTYPE);   /* a w2a8 band without a_sum is not a w2a8 band: gemm_zp's
                                       * ZP_I8_ROW arm reads it. This kernel folds the zero into the
                                       * staged tile and never indexes it -- see the header. */

    const long long M_band = r3d_geti_or(args, "M", 0), N0 = r3d_geti_or(args, "N", 0);
    const long long K0 = r3d_geti_or(args, "K", 0);
    const long long N = N0 > 0 ? N0 : r3d_dim(B, 0);       /* the weight IS [N, K]: name the operand
                                                            * when the band names no key */
    const long long K = K0 > 0 ? K0 : r3d_dim(B, 1);
    const long long group = r3d_geti_or(args, "group", 0);
    if (N <= 0 || K <= 0) return R3D_NO(RAD_E_SHAPE);
    /* The step's M, off the codes plane, exactly as in the fp8a8 shim above: the band key `M` is the
     * ceiling the plan promised not to exceed, not the height of this step, and a draft pass whose
     * chunk is shorter than the band it matched is refused by the bound below. */
    long long M = 0;
    if (K > 0 && r3d_numel(A) % K == 0) M = r3d_numel(A) / K;
    if (M < 1) M = M_band;
    if (M_band > 0 && M > M_band) { r3d_trace_desc("a", A);
        std::fprintf(stderr, "[r3d] w2a8 a: %lld rows in a band that declared M <= %lld\n", M, M_band);
        return R3D_NO(RAD_E_SHAPE); }
    if (group != kGroup)
        return R3D_NO(RAD_E_SHAPE);   /* the fold indexes the group as kGroup; kKC 32 divides it */
    if (K % kGroup) return R3D_NO(RAD_E_SHAPE);            /* a ragged k-chunk reads past the plane */
    if (M < kMMin)
        return R3D_NO(RAD_E_UNSUPPORTED);   /* gemm_nt_q_w2a8_m16's array says M <= 16; the floor of
                                             * this arm is that ceiling + 1, read from the array */
    const long long G = (K + kGroup - 1) / kGroup;
    const long long mpad = ((M + 15) / 16) * 16;           /* a_sum is group-major, 16 columns min */

    if (A->dtype != RAD_I8) return R3D_NO(RAD_E_DTYPE);    /* quant_act_i8's codes */
    if (Y->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(AS) || !r3d_is_f32(ASUM)) return R3D_NO(RAD_E_DTYPE);
    if (r3d_numel(AS) < M) return R3D_NO(RAD_E_INVAL);     /* one scale per ROW (ZP_I8_ROW) */
    if (r3d_numel(ASUM) < G * mpad) return R3D_NO(RAD_E_INVAL);
    if (r3d_numel(A) < M * K) return R3D_NO(RAD_E_SHAPE);
    if (M * N > (1ll << 34)) return R3D_NO(RAD_E_INVAL);   /* 32-bit tile indices */

    /* The weight's planes. This row's plan puts codes, scale and zero behind b (operand 2), which is
     * what r3d_layout_gemm_q_fp8_weight's asym branch describes. The decode row's plan puts the
     * scale+zero pair behind b_scale and leaves the codes plain on b, and the builder settles ONE
     * stored form per weight, so read whichever the container is and find the planes by role: a row
     * that insists on the arrangement it declared will refuse the settled bytes of a weight it
     * shares with the sibling. */
    const void* bbase = r3d_p(B);
    const RadPlanesHeader* bh = rad_planes_header(bbase);
    bool codes_joined = false;                 /* the codes plane inside b's container? */
    const RadPlanesEntry *ce = nullptr, *se = nullptr, *ze = nullptr;
    const void* sp = nullptr; const void* zp = nullptr;
    long long c_row = 0, s_row = 0, z_row = 0;
    const void* sv = nullptr; const void* zv = nullptr;
    if (bh) {
        const long long cb = (cdt == 2) ? (K >> 2) : (K >> 1);
        sp = rad_planes_find(bbase, "scale", &se);
        zp = rad_planes_find(bbase, "zero", &ze);
        const void* cp = rad_planes_find(bbase, "codes", &ce);
        if (!cp || !sp || !zp) return R3D_NO(RAD_E_DTYPE);  /* three planes or nothing */
        if (ce->dtype != (uint32_t)(cdt == 2 ? RAD_U2 : RAD_U4)) return R3D_NO(RAD_E_DTYPE);
        if (ce->row_bytes < cb) return R3D_NO(RAD_E_STRIDE);
        c_row = ce->row_bytes;
        codes_joined = true;
    } else {
        if (!r3d_p(BS)) return R3D_NO(RAD_E_DTYPE);         /* no zero, no plain spelling: the zero
                                                             * point has nowhere to live outside the
                                                             * container. */
        const RadPlanesHeader* h3 = rad_planes_header(r3d_p(BS));
        if (!h3) return R3D_NO(RAD_E_DTYPE);               /* the decode row's arrangement: the
                                                            * scale+zero pair joined behind b_scale */
        sp = rad_planes_find(r3d_p(BS), "scale", &se);
        zp = rad_planes_find(r3d_p(BS), "zero", &ze);
        if (!sp || !zp) return R3D_NO(RAD_E_DTYPE);
        if (B->dtype != (uint32_t)(cdt == 2 ? RAD_U2 : RAD_U4)) return R3D_NO(RAD_E_DTYPE);
        if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);
        c_row = rad_dtype_bytes(B->dtype, r3d_dim(B, 1));   /* the row in BYTES, the same measure
                                                             * the container's entry carries on the
                                                             * joined side: K/4 for U2, K/2 for U4.
                                                             * A hand-rolled `K * (1) / 2`
                                                             * integer-divides the U2 case to zero
                                                             * and the kernel then reads every row
                                                             * from the start of the plane: the
                                                             * decode form comes out rel_l2 1.9. */
        (void)ce;
    }
    if (!se || !ze) return R3D_NO(RAD_E_DTYPE);
    if (se->dtype != RAD_F16)
        return R3D_NO(RAD_E_DTYPE);   /* the grid's scale is f16; a bf16 or f32
                                       * scale plane is the fp8a8 grid's, not this one. */
    if (ze->dtype != RAD_U8)
        return R3D_NO(RAD_E_DTYPE);   /* DECLINED, at the site: a sub-byte zero plane needs the bit
                                       * extract (rad_load_bits(zb, bits, g), ten lines) that the m16
                                       * row also declines. U8 is w2a8's practical form. */
    if (se->rows != N || se->cols != G || ze->rows != N || ze->cols != G)
        return R3D_NO(RAD_E_SHAPE);   /* {1, group}: the scale's rows are the weight's rows */
    if ((se->row_bytes & 1) || se->row_bytes < 2 * G || ze->row_bytes < G)
        return R3D_NO(RAD_E_STRIDE);
    if (bh) {
        if (ce->rows != N || ce->cols != K) return R3D_NO(RAD_E_SHAPE);
        if (ce->row_bytes < c_row) return R3D_NO(RAD_E_STRIDE);
    } else if (r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);

    long long a_rows = 0;
    const long long a_ld = r3d_rows_pitch(A, K, &a_rows);
    if (a_ld < K || (a_ld & 3)) return R3D_NO(RAD_E_STRIDE);  /* four codes a dword, step order */
    long long yrows = 0;
    const long long y_ld = r3d_rows_pitch(Y, N, &yrows);
    if (y_ld < N) return R3D_NO(RAD_E_STRIDE);
    if (yrows > 0 && yrows < M) return R3D_NO(RAD_E_STRIDE);
    const long long as_ld = 1;                               /* a_scale is rank 1: one per row */
    sv = se ? (const char*)sp : nullptr;
    zv = ze ? (const char*)zp : nullptr;
    s_row = se->row_bytes; z_row = ze->row_bytes;
    const long long bvc = codes_joined
                        ? (long long)(uintptr_t)((const char*)bbase + ce->offset)  /* the container's
                                                                                     * codes plane */
                        : (long long)(uintptr_t)r3d_p(B);

    if (!r3d_aligned(r3d_p(A), 8) || !r3d_aligned(r3d_p(Y), 8) ||
        !r3d_aligned(sv, 2) || !r3d_aligned(zv, 1)) return R3D_NO(RAD_E_ALIGN);

    const int nt = (int)r3d_geti_or(args, "nt", 0);
    const int st = launch(r3d_addr(A), r3d_addr(AS), bvc, (long)(uintptr_t)sv, r3d_addr(Y),
                          (int)M, (int)N, (int)K, (int)group, (int)a_ld, c_row, (int)as_ld,
                          s_row, (int)y_ld, nt, (long)(uintptr_t)s, /*sdt*/ 1, cdt,
                          (long)(uintptr_t)zv, z_row, /*srows*/ 1, /*arm*/ 1);
    if (st == -1) return R3D_NO(RAD_E_INVAL);
    if (st == -2) return R3D_NO(RAD_E_SHAPE);
    if (st == -3) return R3D_NO(RAD_E_STRIDE);
    if (st == -5) return R3D_NO(RAD_E_ALIGN);
    if (st == -4) return R3D_NO(RAD_E_DEVICE);
    return RAD_OK;
}
