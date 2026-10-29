/* r3d_vocab_bf16.hip -- the lm_head projection: `logits_gemm`, and the same tiling under `gemm_nt`
 * at prefill sizes.
 *
 *   y[M,N] = x[M,K] @ w[N,K]^T, f32 accumulate, k ascending.
 *
 * `logits_gemm` spells N and K as n_vocab and n_embd and its y is **f32**: the schema says so
 * (`ref_logits_gemm_shape` writes `o.t(RAD_F32, {M, nv})`, "f32 logits, as every sampler stage reads
 * them"), and none of those stages wants a bf16 logit. `gemm_nt` is the same arithmetic with a bf16
 * y, narrowed once by the store. Both entries are the same two kernels; `YF32` is the only
 * difference in the code and the only difference in what the board has to report.
 *
 * ---- why this row exists ------------------------------------------------------------------------
 *
 * The probe of the served checkpoint refuses exactly one op: `logits_gemm`. libref's
 * `ref_logits_gemm` is
 * `gemm_nt_impl(a, false, "n_vocab", "n_embd")` on 32 OpenMP threads: 1.1 TFLOP for the band's shape
 * (M=2048, n_vocab=130560, n_embd=2048), which this card does in about a second and the CPU in
 * tens. For decode it is the 535 MB vocabulary head read once a token -- one call per token, the
 * head the model cannot answer without. This row is not a speed decision. It is
 * a read of the container's own form, and the head is stored in two of them: bf16 plain, with
 * `layout`, `relayout` and `unrelayout` all null; and E4M3 codes with a bf16 scale per 128x128 block
 * as two planes of one `rad.planes` container, which is the form the served checkpoint ships and
 * which is answered by `r3d_layout_logits_head`, this file's transcription of `ref_layout_head`.
 * Nothing is repacked in either case, and no operand description is declined.
 *
 * ---- the prefill arm (M > 16) -------------------------------------------------------------------
 *
 * A block is TM x TN = 128 x 128 outputs, 256 threads, eight waves, eight outputs a side a lane: 64
 * f32 accumulators a lane, grid (N/TN) x (M/TM). Lane `tid` owns rows m0 + (tid/16)*8 .. +7 by
 * columns n0 + (tid%16)*8 .. +7, so its 64 outputs are one 8x8 square of the tile and its stores are
 * 32-byte runs of f32 (16-byte runs of bf16).
 *
 * k is walked in KC = 32 code chunks. Both tiles are stored row-major in f16 -- a row is the 32
 * k-values of one m, 64 bytes -- because the global load that fills them is 8 contiguous k of one
 * row: one `uint4` a thread, sixteen bytes, widened on the way in by four `bf16x2_to_f16x2`, which
 * is exact for everything f16 can hold (bf16 has eight mantissa bits, f16 ten) and arrives as an inf
 * for what it cannot, which the board counts rather than assumes away. The compute then reads those
 * 64-byte rows in f16x2 pairs: eight activation pairs and eight weight pairs per two codes,
 * sixty-four `fdot2` per two codes -- one pair-dot per output per two codes, 16 LDS reads against
 * 64 dots.
 *
 * The tile pitch is KC elements and the rows are XOR-swizzled in eight-element units, which is the
 * only arrangement that satisfies both ends at once. The staging store is a `uint4`, so a row's
 * chunk offset has to stay a multiple of eight elements; an unswizzled 64-byte row stride puts all
 * eight rows a lane reads on the same bank, 32-way. `srow` moves whole eight-element chunks by
 * `row & 7`, which preserves the 16-byte alignment and brings the compute reads to two-way. Two-way
 * and not conflict-free: the eight rows a lane owns sit at stride eight, so a period-8 swizzle
 * repeats inside a lane's own set. The cost is 512 LDS cycles a block-chunk against 2048 of
 * dots, a quarter, which is what the vectorised staging store buys.
 *
 * At the band, M=2048 n_vocab=130560 n_embd=2048, 16 x 1020 blocks of 256 threads:
 * 272.24 ms = 4023 GFLOP/s, which is 37% of the 10.8 TFLOP/s this card's 32 CUs do dense f16 with
 * f32 accumulate. At M=46 -- one tile row, 1020 blocks -- the same kernel does 16.64 ms =
 * 1478 GFLOP/s, and the shortfall is the shape, not the kernel: 46 rows against a 128-row tile
 * leaves 5/8 of every block's arithmetic idle. The host row for this op is `ref_logits_gemm`: 1.1
 * TFLOP per forward pass across the whole vocabulary, on OpenMP threads; this row is the card's
 * answer to that comparison.
 *
 * ---- the decode arm (M <= 16) -------------------------------------------------------------------
 *
 * The prefill tile reads the head M/TM times; at M=1 it would read 127/128 of it for nothing. At
 * M <= 16 the block is the decode shape `r3d_gemm_bf16_nt_m16.hip` uses: a wave is MT activation
 * rows by 32/MT weight rows, lane l owns (m = l % MT, n = n0 + l / MT), one accumulator a lane from
 * k=0 to K, and no wave reduction anywhere -- so each output's sum is the reference's ascending
 * chain, summand for summand, in the pair-dot's grouping. The activation arrives in 128-code chunks
 * and stays in LDS for the chunk (MT x 128 f16, 4 KiB at MT=16, the same eight-element swizzle at
 * the wider pitch); the weight goes straight from global to registers in 16-byte loads, eight codes
 * at a time, widened where it lands. A wave keeps 32/MT weight rows in flight -- 4 KiB at MT=1 --
 * which is what makes 16-byte strided reads across rows cost what they fetch. `nt=1` asks for
 * non-temporal head loads, the honest hint for bytes read once a token; on this card it is worth
 * nothing: 3.316 ms against 3.290 ms at M=1, 0.8% the other way. It is implemented
 * because the ABI names it, and left off by default because the number says to.
 *
 * At n_vocab = 130560, n_embd = 2048 the arm is M = 1 in 3.226 ms = 166 GB/s of head, and M = 16 in
 * 8.151 ms = 1050 GFLOP/s, 66 GB/s of bytes. The w2a8 row's 22 GB/s is not this arm's bound: it is
 * that kernel's own limit -- an ALU-bound pair-dot with one 16-byte weight load per 64 codes -- and
 * the card's DRAM is 256 GB/s. This arm is the streaming shape, 16 bytes a lane a step with the
 * activation in LDS, and it runs at two thirds of the card.
 *
 * So the bf16 head costs 3.23 ms per decode token, and M = 16 (a speculative verify) costs 0.51 ms
 * a token. The quantised head is the same op in half the bytes: the codes are 130560 x 2048 =
 * 267,386,880 bytes (255.0 MiB) and the scales 1020 x 16 bf16 = 32,640 bytes (31.9 KiB), so a decode
 * reads 255.03 MiB where the bf16 head reads 510.0 MiB. At the 163 GB/s this row's bf16 arm runs at
 * the floor that is 1.64 ms a token, and at the card's 256 GB/s, 1.04 ms. (The 66.8 MB figure is the
 * w2a8 row's 2-bit grid, a different head form.) Every number below says which of the three
 * placements the bytes were in, because the placement is worth more than anything in the kernel, and
 * all three are on this page so the choice between the head's forms is made on measurement.
 *
 * ---- the accumulation, and what it is exact against ---------------------------------------------
 *
 * libref's law per output is `acc += a*b` over ascending k from zero, and for this op with no bias.
 * One accumulator per output, k ascending, no reassociation across lanes, in both arms. The pair-dot
 * folds two products into the accumulator per step (`acc + a0*b0 + a1*b1` in one rounding) where the
 * reference rounds each product in, so the sums agree closely and are not bit-equal. f32 output,
 * K = 2048: rel_l2 1.77e-06 against the reference's law, identical -- 1.77e-06, to the
 * digits -- against the same law with the operands widened to f16 first, so the widening
 * contributes nothing and the fusion is the whole of it. 1.77e-06 is about 30
 * f32 ulps on a sum of 2048 terms, and a random walk of one rounding per 1024 steps predicts
 * sqrt(1024) * 6e-8 = 1e-6, which is where it lands. At K = 320 it is 2.0e-07, which scales the same
 * way. What it means for the sampler: the head's bf16 values carry 4.9e-4 of quantisation each, so
 * this is 300x finer than the input, and a top-k decision flips only on a tie tighter than 1e-6.
 *
 * Nothing here is integer arithmetic, so the v_fdot2_f32_f16 mantissa rule that bounds
 * r3d_gemm_nt_q_w2a8.hip does not bite: the products are not integers, so there is no 2^20 wall, and
 * the board's tail is ordinary rounding, counted with the plane pre-filled at 0x7fc0 -- no non-finite
 * output at any size, including the 268-million-output band.
 *
 * ---- the head's stored form, and who answers it ---------------------------------------------------
 *
 * The band's `dtype=bf16` is the ACTIVATION's dtype. The head is quantised: the graph reads
 * 'output.weight' as two planes of fp8_e4m3 * bf16[128x128], and the mapped tier reads the
 * container where it lies, and its stored form is not the container's (rad.planes). The placement
 * needs a reader for the head's stored form, and a layout hook is what a reader is. So:
 *
 *   r3d_layout_logits_head  (operand 1 -- `ref_layout_head` is `layout_scaled(operand, 1, -1, ...)`)
 *       tag `rad.planes`; codes E4M3 (libref's `scaled_rows` also admits int8) with a float scale --
 *       bf16, f16 or f32 -- on a block of {128, 128} (`R3D_FP8_BLOCK`); ends in `rad_planes_layout`.
 *       It moves no byte and `relayout` stays null. `ref_layout_head` is libref-internal-static, so
 *       its body, and `layout_scaled`'s, and `enc_scaled_rows`' and `float_scale_dt`' with it, are
 *       transcribed here as the w2a8 hook transcribed `ref_layout_gemm_q`: checks and their order,
 *       unchanged. Two fields are worth naming: `rad_enc_is`
 *       compares `scheme`, and `transform` is the optional pre-transform. The
 *       encoding is affine with an EMPTY transform; a fixture that writes "affine" into `transform`
 *       makes the hook decline every head.
 *
 *   The law. `ScaledRows::at(r, c) = codes.at(r, c) * scale.cover(r, c)`: the code widened exactly,
 *   multiplied by the scale covering it, one f32 rounding, then the plain ascending-k GEMM in f32.
 *   The widening is exact both ways -- E4M3 is 4 exponent and 3 mantissa bits, f16 is 5 and 10, and
 *   int8 lands in f16 whole -- so the codes reach the dot as the numbers the checkpoint means, and
 *   the only divergence stays the dot's fused rounding: rel_l2 2.77e-04 at M=1, K=2048,
 *   which is one f16 ulp's worth (2^-12 = 2.44e-04) and is the fold, not the dot; against the fold
 *   law, below, it is 4.5e-06, and `<=2 ulps` is 1.000 with `>6 ulps` 0 at every size,
 *   268-million-output band included. y stays f32: logits are f32 and the reference's store is f32.
 *
 *   The fold, and the hardware rule it obeys. The reference dequantises each element into f32; the
 *   tile is f16, so the scale enters as a f16 multiply at staging -- the tiled arm once per staged
 *   8-column group, the decode arm once per 128-code chunk, and a chunk of 32 sits inside one
 *   128-column block (128 % 32 == 0, 128 % 128 == 0), so one scale per staged row. That costs no
 *   registers: the prefill arm is the register file's at 237 VGPRs, and per-k-block partials would
 *   need 64 more accumulators and spill. The multiply must be a hardware f16 multiply
 *   (`v_mul_f16`, round to nearest, subnormals kept), not `f16x2_pack`: that builtin is
 *   `v_cvt_pkrtz_f16_f32`, ROUNDS TOWARD ZERO, and r3d_codecs.h says so and says to call it only
 *   with values f16 already represents exactly. A folded weight is not one of those -- code 431 by
 *   a scale of 4.5e-5 is eleven significant bits -- and a toward-zero packer biases every product
 *   toward zero and flushes the subnormal end: rel_l2 2.8e-04 against the fold law with only 0.53 of
 *   outputs within 2 ulps. The f16 multiply is one rounding in the right direction and one
 *   instruction: 4.5e-06 and 1.000. The number and the verdict: 2.77e-04 against the reference's
 *   exact law is the fold's cost, 250x finer than the E4M3 code's own quantum (2^-4 = 6.25e-2), so
 *   the fold does not move the logits by anything the quantisation did not already move. An f32
 *   scale plane costs one more rounding (the scale to f16, 1.75e-04 against a model that folds
 *   after the multiply); the shipped form's bf16 scale is f16-exact and pays nothing.
 *
 *   With the head's bytes where the engine would put them (VRAM, with the header page
 *   read by the shim on the host): prefill M=2048 in 325.06 ms = 3369 GFLOP/s against the bf16 head's
 *   269.31 ms = 4067 -- 17% slower, all of it staging, and the case for the quantised head at prefill
 *   is the 255 MiB of weight traffic, not the FLOPs. M=46: 22.43 ms = 0.488 ms a token against
 *   0.361. Decode M=1: 13.603 ms a token, 19.7 GB/s of codes, 8x above the 1.64 ms floor, and the
 *   cause is the E4M3 decoder. `__builtin_amdgcn_cvt_pk_f16_fp8` exists in clang and expands
 *   to software on gfx1102 -- zero `v_cvt_pk_f16_fp8` in the ISA -- so eight codes cost
 *   ~100 instructions of `e4m3_to_f32` against the bf16 arm's four, and the arm is ALU-bound. The
 *   fix has a size: a 256-entry f16 table in LDS is 512 bytes, 8 LDS reads a staged group against
 *   ~100 ALU, which puts the arm back near its floor; that is a tiling pass, not this row's work.
 *   The speculation block is where the form already pays: M=8 (7 drafts + 1) reads the head ONCE and
 *   costs 21.67 ms = 2.708 ms a token, 5x cheaper per token than M=1 on the same kernel.
 *
 *   Where the bytes live, on this box, at M=1: in VRAM 13.6 ms; on the mapped tier
 *   (hipHostMalloc, read over PCIe by 8-byte strided loads) 643 ms a token; managed and left alone
 *   2071 ms a token, 122 MB/s, because managed pages do not migrate for a read-mostly device access
 *   on this ROCm and the kernel faults page by page. The head is placed on
 *   the mapped tier. A hook makes that placement legal; it does not make it fast, and these three
 *   numbers are the argument for uploading the head. The board takes the timed number after
 *   `hipMemPrefetchAsync` and says so on every line.
 *
 *   Both forms are served, chosen by the header: `rad_planes_header(w)` at the shim's head branch
 *   dispatches on it, and a plain bf16 operand goes down the bf16 path untouched. The choice is
 *   the container's, not the row's. The shim reads the container's header
 *   (which is why the operand's bytes must be host-readable: `scaled_rows()` does the same host read,
 *   so this is the engine's convention, not this row's invention); the kernel reads the same bytes on
 *   the card, so the fixture is hipMallocManaged + prefetch, or hipHostMalloc, never plain malloc --
 *   plain malloc faults 700 on the kernel's first read.
 *
 * ---- the decoder is a table, and what the table moved --------------------------------------------
 *
 * r3d_fp8_dequant.h replaces the E4M3 codec with a 256-entry f16 table in LDS (512 bytes, eight
 * reads an eight-code group) because clang lists __builtin_amdgcn_cvt_pk_f16_fp8 for gfx1102 and a
 * build that calls it contains ZERO v_cvt_pk_f16_fp8: the builtin is software here. The A/B is one
 * source built twice (-DR3D_NO_E4M3_TBL selects the codec), same fixture, same seed:
 *
 *   M=8    speculation  2.500 -> 0.924 ms/token   (20.002 -> 7.395 ms)   2.70x   THE WIN
 *   M=2048 prefill    323.978 -> 322.661 ms      (3380 -> 3394 GFLOP/s)  flat, -0.4%
 *   M=1    decode      12.999 -> 13.293 ms/token (20.6 -> 20.2 GB/s)     NOT moved, 2.3% worse
 *   M=46   prefill      21.603 -> 21.792 ms      (1139 -> 1129 GFLOP/s)  flat
 *
 * The table is exact by construction -- the codec, evaluated once per bit pattern at block start --
 * and rel_l2 is bit-identical between the two binaries (2.770e-04 vs the fold law / 4.452e-06 vs the
 * fold model at prefill; 2.758e-04 / 4.559e-06 at M=1), which is the check that the swap moved work
 * and not arithmetic. It is a 2.70x win where the decoder was the binding resource (MT=8: four
 * decodes and four folds a lane per eight codes, no memory wall in reach) and a NAMED NON-RESULT at
 * M=1: 13 ms at 20 GB/s of codes is 8x off the 1.64 ms floor this row's own 163 GB/s implies, and
 * removing ~100 ALU instructions an eight-code group does not move it. The suspect at M=1 is the
 * load width -- eight codes a lane through an 8-byte load, where the bf16 arm takes sixteen bytes a
 * lane and streams at 163 GB/s -- so the shape to try there is a 16-byte code load (16 codes, eight
 * pairs, half the wavefronts), not more ALU. 44 kernels, all vgpr_spill_count: 0, both builds.
 *
 * ---- what is refused, and where ------------------------------------------------------------------
 *
 * sk > 1: the grid is vocab-wide at every M -- 1020 blocks in x at the band for the prefill arm,
 * 8160 for the decode arm at MT=16 -- so no shape this row serves is short of parallelism, and a
 * split would buy a second kernel and a documented order change for nothing. kb: KC = 32 only; a
 * tile pitched at anything else is a different swizzle, not a knob, and the launcher enumerates the
 * values it accepts and refuses the rest. K not a multiple of KC, an x pitch not a multiple of 8, a
 * y pitch not a multiple of 4 (f32) or 8 (bf16), a weight that is not contiguous, and quantised
 * heads that are not this form. The head branch, in its order:
 *   a quantised head on the bf16-y row -- logits are f32; the tiled row keeps a bf16 head
 *   a `zero` plane on the head -- codes+scale+zero is gemm_nt_q's asymmetric grid, and the row
 *        that reads it is r3d_gemm_nt_q_w2a8.hip; that is w2a8's grid, not this op
 *   a joined operand that is not two planes
 *   a joined head whose planes are not codes + scale (a `table`, a `bias`: names it)
 *   codes tiled by anything but one code a cell
 *   sub-byte codes (4-bit: w4a16's ground, and the row that reads it is w2a8's neighbour)
 *   e5m2 and every other code dtype: no decoder here
 *   a scale plane that is a table, not a tiling
 *   an integer scale: not float_scale_dt's set
 *   a scale block that is not 128x128 -- per-(row, group) scales are gemm_nt_q's grid, and a
 *        per-row scale is a different indexing of the plane than ScaledRows::cover is
 *   a plane that does not cover the band (codes must be N x K, scales ceil(N/128) x
 *        ceil(K/128)); a plane whose row_bytes is shorter than its row
 * and the bf16 branch's: a plain head that is not bf16, x and y dtypes,
 * no axes, pitches, the head's numel and contiguity, absurd shapes,
 * alignment, sk > 1. The launcher's own: a head with a null plane, a dtype
 * outside the served set, a row pitch a vector load cannot walk (a 16-bit scale row is 2
 * bytes a column, so K = 320 -- 3 scale columns, 6 bytes -- is legal), addresses.
 * 24 refusals, 24 as stated, and the hook's two served encodings answer RAD_OK with
 * tag `rad.planes` and the codes' dtype.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_fp8_dequant.h"   /* the E4M3 -> f16 table: 512 bytes of LDS for ~100 ALU a group */
#include "r3d_tiling.h"          /* the LDS row law: srow, srow_pair */
#include "r3d_args.h"
#include "r3d_launch.h"

#include "rad_planes.h"

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWaves        = kBlockThreads / kWave;   /* 8 */
constexpr int kTM           = 128;                     /* activation rows a block */
constexpr int kTN           = 128;                     /* weight rows a block */
constexpr int kKC           = 32;                      /* codes a tile step */
constexpr int kSkinnyMax    = 16;                      /* the MT ceiling of the decode arm */
constexpr int kChunkK       = 128;                     /* the decode arm's activation chunk */

#if defined(__has_builtin)
#  if __has_builtin(__builtin_nontemporal_load)
#    define R3D_HAS_NT 1
#  endif
#endif
#ifndef R3D_HAS_NT
#  define R3D_HAS_NT 0
#endif

/* The tile address, in f16 elements: `pitch`-wide rows whose eight-element chunks are XORed by the
 * row's low bits. Multiples of eight in, so a uint4 store and an f16x2 read are both aligned by
 * construction, at both pitches this file uses (32 and 128). */
/* srow / srow_pair: r3d_tiling.h, the family's law in one place. */

/* Eight bf16 -> eight f16, exact for every value f16 can hold. */
__device__ __forceinline__ uint4 widen8(uint4 w)
{
    return make_uint4(bf16x2_to_f16x2(w.x), bf16x2_to_f16x2(w.y), bf16x2_to_f16x2(w.z),
                      bf16x2_to_f16x2(w.w));
}

constexpr int kFp8Block = 128;        /* the head's scale block: R3D_FP8_BLOCK, llama_dense_fp8 */

/* The head as the container stores it: E4M3 codes with one float scale per (128 x 128) block.
 * Passed by value; every pointer in it is a device address, resolved by the shim out of the joined
 * bytes (rad_planes_find), which is what a layout hook's absence asks of a kernel -- the container's
 * own form, read where it lies, no byte moved. */
struct HeadQ {
    const uint8_t* codes;        /* [N][K] */
    const uint8_t* scale;        /* [ceil(N/128)][ceil(K/128)] */
    long long code_row;          /* bytes a codes row: the plane's row_bytes, not K's element count */
    long long scale_row;         /* bytes a scale row */
    int sdt;                     /* the scale's dtype: 0 bf16, 1 f16, 2 f32 (float_scale_dt's set) */
    int cdt;                     /* the codes' dtype: 0 E4M3, 1 int8 */
};

/* The scale covering logical (n, k): `scale.cover` from the reference's ScaledRows, one scalar load a
 * row a chunk. The plane is 32 KiB at the band, so it is in cache from the first block that touches
 * a row and the load is not the traffic; the codes are. */
__device__ __forceinline__ float scale_at(const HeadQ h, int n, int k)
{
    const uint8_t* p = h.scale + (long long)(n >> 7) * h.scale_row
                     + (long long)(k >> 7) * (h.sdt == 2 ? 4 : 2);
    if (h.sdt == 2) { float f; __builtin_memcpy(&f, p, 4); return f; }
    const uint16_t u = *(const uint16_t*)p;
    return h.sdt == 1 ? f16_to_f32(u) : bf16_to_f32(u);
}

/* Eight codes -> eight f16, exact. E4M3 is four exponent and three mantissa bits and f16 is five and
 * ten, and int8 lands in f16 whole, so the decoder is a widening, not a rounding: the numbers the
 * checkpoint means are the numbers multiplied (r3d_gemm_fp8a16_nt_m1.hip says the same about the
 * same codec). e4m3x2_to_f16x2 decodes the two codes in a dword's LOW half, so the pairs straddle
 * dwords: two dwords in, four pairs out. Feeding it whole dwords decodes eight codes and leaves the
 * rest of the step multiplying registers that were never written -- the shape the sibling unit
 * documents, and the reason this helper takes the pair the way it does. */
#if defined(__has_builtin)
#  if __has_builtin(__builtin_amdgcn_cvt_pk_f16_fp8)
#    define R3D_HAS_FP8CVT 1        /* gfx11+: v_cvt_pk_f16_fp8, two codes to two f16, one instruction */
#  endif
#endif
#ifndef R3D_HAS_FP8CVT
#  define R3D_HAS_FP8CVT 0
#endif

__device__ __forceinline__ uint32_t cvt_pk_f16_fp8(uint32_t b)
{
#if R3D_HAS_FP8CVT
    return __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pk_f16_fp8(b));
#else
    return f16x2_pack(e4m3_to_f32((uint8_t)(b & 0xffu)), e4m3_to_f32((uint8_t)((b >> 8) & 0xffu)));
#endif
}

__device__ __forceinline__ void codes8_to_f16x4(const uint2 c, int cdt, const uint16_t* tbl,
                                               uint32_t* w)
{
    const uint32_t d[2] = { c.x, c.y };
    if (tbl && !cdt) {
        /* The table: eight LDS reads for eight codes. Exact by construction -- the table is built
         * from the codec below, one entry per bit pattern -- so a row's numbers do not move; what
         * moves is where the work happens. r3d_fp8_dequant.h carries the reason and the measurement.
         * Passing null tbl asks for the codec, which is how the board A/Bs the two in one binary. */
        r3d_e4m3x8_ld(tbl, c, w);
        return;
    }
    if (!cdt) {
        /* The instruction this wants: two codes to two f16, one op. clang has the builtin on this
         * target's list and gfx1102 does not have the instruction: the ISA of this build contains
         * zero v_cvt_pk_f16_fp8, so the builtin expands to the software decode below and
         * eight codes cost ~100 instructions of exponent work. At M=1: 19.7 GB/s of codes,
         * 8x above the 1.64 ms floor, ALU-bound. The builtin stays because it says what the operation
         * is and a card with the instruction gets it for free; the fix for THIS card is a 256-entry
         * f16 table in LDS, 512 bytes, 8 LDS reads where the ~100 ALU are, which is a tiling pass. */
        w[0] = cvt_pk_f16_fp8(d[0]);
        w[1] = cvt_pk_f16_fp8(d[0] >> 16);
        w[2] = cvt_pk_f16_fp8(d[1]);
        w[3] = cvt_pk_f16_fp8(d[1] >> 16);
        return;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t pair = (d[j >> 1] >> (16 * (j & 1))) & 0xffffu;
        w[j] = i8x2_to_f16x2(pair);
    }
}

/* The fold: the decoded code times the scale covering it, in f16 pairs.
 *
 * The reference dequantises per element into f32 (`ScaledRows::at` = code * scale, one rounding) and
 * then multiplies. Folding into the staged f16 costs one f16 rounding per weight -- half an ulp,
 * 2.4e-4 relative, on a code whose own quantum is 2^-4 = 6.3e-2, a factor of 250. It buys the whole
 * structure: the tile stays f16, the dot loop is untouched, the accumulators do not double, and the
 * register file, which is the tightest thing in the prefill arm at 237 VGPRs, does not move. The
 * alternative -- a per-block partial summed separately and scaled -- needs 64 more accumulators and
 * spills. The number the board prints is this fold's. */
__device__ __forceinline__ uint32_t scale_pair(uint32_t w, float sc)
{
#if defined(__HIP_DEVICE_COMPILE__)
    /* One f16 multiply, hardware, round-to-nearest.
     *
     * It is NOT f16x2_pack of an f32 product: that builtin is v_cvt_pkrtz_f16_f32, rounds toward
     * zero, and r3d_codecs.h says out loud to call it only with values f16 already represents
     * exactly. A folded weight is not one of those -- code 431 times a scale of 4.5e-5 is a product
     * with eleven significant bits -- so the packer truncates, and truncation is a bias: every
     * product moves toward zero, and it shows as rel_l2 2.8e-04 (half an f16 ulp, the
     * whole of it) with the subnormal end flushed to zero on top. F16 arithmetic is one v_mul_f16
     * an element and rounds the way the reference's f32 multiply rounds.
     */
    const uint16_t lo16 = (uint16_t)(w & 0xffffu), hi16 = (uint16_t)(w >> 16);
    const _Float16 a = __builtin_bit_cast(_Float16, lo16);
    const _Float16 b = __builtin_bit_cast(_Float16, hi16);
    const _Float16 s = (_Float16)sc;                 /* f32 -> f16, round to nearest even */
    const _Float16 rl = a * s, rh = b * s;           /* v_mul_f16: one rounding, subnormals kept */
    return (uint32_t)__builtin_bit_cast(uint16_t, rl) |
           ((uint32_t)__builtin_bit_cast(uint16_t, rh) << 16);
#else
    return (uint32_t)f32_to_f16(f16_to_f32((uint16_t)(w & 0xffffu)) * sc) |
           ((uint32_t)f32_to_f16(f16_to_f32((uint16_t)(w >> 16)) * sc) << 16);
#endif
}

/* The head's 16-byte load, ordinary or streamed-once. */
template <int NT>
__device__ __forceinline__ uint4 loadw8(const uint16_t* p)
{
#if R3D_HAS_NT
    if (NT) {
        /* The builtin wants a native vector, not HIP's class: four unsigned ints, bit-cast. */
        typedef unsigned int v4u __attribute__((ext_vector_type(4)));
        const v4u t = __builtin_nontemporal_load((const v4u*)p);
        uint4 r;
        __builtin_memcpy(&r, &t, 16);
        return r;
    }
#else
    (void)NT;
#endif
    return *(const uint4*)p;
}

/* Eight codes, the quantised head's 8-byte load, ordinary or streamed-once: the same builtin
 * requirement as the 16-byte one, a native vector. Half the bytes of the bf16 head is the whole
 * point of the form, and it is the load that changes. */
template <int NT>
__device__ __forceinline__ uint2 loadc8(const uint8_t* p)
{
#if R3D_HAS_NT
    if (NT) {
        typedef unsigned int v2u __attribute__((ext_vector_type(2)));
        const v2u t = __builtin_nontemporal_load((const v2u*)p);
        uint2 r;
        __builtin_memcpy(&r, &t, 8);
        return r;
    }
#else
    (void)NT;
#endif
    return *(const uint2*)p;
}

/* ============================================================ the prefill arm, M > 16
 *
 * y is f32 for logits_gemm and bf16 for gemm_nt; `YF32` is that choice, and it is the only one.
 */
template <bool YF32, bool WQ>
__global__ void __launch_bounds__(kBlockThreads) R3D_WAVES(2)
gemm_tiled_kernel(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, HeadQ h,
                  void* __restrict__ y, int M, int N, int K, int a_ld, int y_ld)
{
    (void)b;   /* the plain head; WQ reads the container through h */
    __shared__ __align__(16) uint16_t sA[kTM * kKC];   /* f16, TM rows x KC, swizzled */
    __shared__ __align__(16) uint16_t sB[kTN * kKC];   /* f16, TN rows x KC, swizzled */

    const int tid = (int)threadIdx.x;
    __shared__ __align__(16) uint16_t sTbl[R3D_E4M3_TBL];   /* the E4M3 decoder, 512 bytes */
#ifndef R3D_NO_E4M3_TBL
    if (WQ) r3d_e4m3_f16_init(sTbl, tid, kBlockThreads);    /* fills, then the barrier it holds */
    const uint16_t* const r3d_tbl = WQ ? sTbl : (const uint16_t*)nullptr;
#else
    (void)sTbl;                                             /* the board's A/B: the codec, unbuilt */
    const uint16_t* const r3d_tbl = (const uint16_t*)nullptr;
#endif
    const int m0 = (int)blockIdx.y * kTM;
    const int n0 = (int)blockIdx.x * kTN;
    const int rg = tid >> 4, cg = tid & 15;           /* this lane's 8x8 square */

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;

    const int m_r = m0 + rg * 8, n_c = n0 + cg * 8;

    for (int k0 = 0; k0 < K; k0 += kKC) {
        /* Stage 128 rows x 4 chunks of eight a side: one uint4 a thread a side, two a side. Rows
         * past M and codes past K stage as zeros, which is the clamping a ragged tail needs, keeps
         * the loads straight-line and contributes nothing to a sum. */
#pragma unroll
        for (int j = tid; j < kTM * (kKC / 8); j += kBlockThreads) {
            const int mm = j / (kKC / 8), c8 = j % (kKC / 8);
            const int m = m0 + mm, k = k0 + c8 * 8;
            uint4 w = make_uint4(0, 0, 0, 0);
            if (m < M && k < K) w = *(const uint4*)(a + (long long)m * a_ld + k);
            *(uint4*)&sA[srow(mm, c8, kKC)] = widen8(w);
        }
#pragma unroll
        for (int j = tid; j < kTN * (kKC / 8); j += kBlockThreads) {
            const int nn = j / (kKC / 8), c8 = j % (kKC / 8);
            const int n = n0 + nn, k = k0 + c8 * 8;
            uint4 staged = make_uint4(0, 0, 0, 0);
            if (n < N && k < K) {
                if (WQ) {
                    /* Eight codes, one scale: a group of 8 sits inside one 128-column block, so
                     * the whole staged run shares the scale, and the tile's 32-code chunk sits
                     * inside one block too (128 % 32 == 0). The scale indexes the tile's row, not
                     * the lane's: the plane is (n-block, k-block), and this row is n. */
                    const uint2 c = *(const uint2*)(h.codes + (long long)n * h.code_row + k);
                    uint32_t w[4];
                    codes8_to_f16x4(c, h.cdt, r3d_tbl, w);
                    const float sc = scale_at(h, n, k);
                    staged = make_uint4(scale_pair(w[0], sc), scale_pair(w[1], sc),
                                        scale_pair(w[2], sc), scale_pair(w[3], sc));
                } else {
                    staged = widen8(*(const uint4*)(b + (long long)n * K + k));
                }
            }
            *(uint4*)&sB[srow(nn, c8, kKC)] = staged;
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

    /* The store, eight columns at a time: 32 bytes of f32, 16 of bf16. Guarded row by row, so a
     * ragged M writes nothing and a ragged N writes whole runs of columns it owns. */
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m = m_r + i;
        if (m >= M) break;
        if (n_c + 8 <= N) {
            if (YF32) {
                float* row = (float*)y + (long long)m * y_ld;
#pragma unroll
                for (int j = 0; j < 8; j += 4)
                    *(uint4*)(row + n_c + j) =
                        make_uint4(__float_as_uint(acc[i][j]), __float_as_uint(acc[i][j + 1]),
                                   __float_as_uint(acc[i][j + 2]), __float_as_uint(acc[i][j + 3]));
            } else {
                uint16_t* row = (uint16_t*)y + (long long)m * y_ld;
                uint32_t p[4];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    p[j] = (uint32_t)f32_to_bf16(acc[i][2 * j]) |
                           ((uint32_t)f32_to_bf16(acc[i][2 * j + 1]) << 16);
                *(uint4*)(row + n_c) = make_uint4(p[0], p[1], p[2], p[3]);
            }
        } else {
            /* The ragged N tail, unrolled: a dynamic index into acc would put the 64 accumulators
             * in local memory, and the compiler says so -- 80 KiB of spill against a 64 KiB limit. */
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (n_c + j < N) {
                    if (YF32) ((float*)y)[(long long)m * y_ld + n_c + j] = acc[i][j];
                    else ((uint16_t*)y)[(long long)m * y_ld + n_c + j] = f32_to_bf16(acc[i][j]);
                }
            }
        }
    }
}

/* ============================================================ the decode arm, M <= 16
 *
 * A wave is MT activation rows by KLW = 32/MT weight rows, one accumulator a lane, the whole of k in
 * ascending order. The activation tile is block-wide, so every wave stages it -- a wave whose row is
 * past the vocabulary stages too and simply does not read, which is the rule that keeps a tile from
 * coming up with holes.
 */
template <int MT, bool YF32, int NT, bool WQ>
__global__ void __launch_bounds__(kBlockThreads) R3D_WAVES(4)
logits_skinny_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, HeadQ h,
                     void* __restrict__ y, int M, int N, int K, int x_ld, int y_ld)
{
    constexpr int KLW = kWave / MT;
    constexpr int STEPS = MT * (kChunkK / 8);          /* uint4 staging steps a chunk */

    __shared__ __align__(16) uint16_t sX[MT * kChunkK];   /* f16, MT rows x chunk, swizzled */
    __shared__ __align__(16) uint16_t sTbl[R3D_E4M3_TBL];
#ifndef R3D_NO_E4M3_TBL
    if (WQ) r3d_e4m3_f16_init(sTbl, (int)threadIdx.x, kBlockThreads);
    const uint16_t* const r3d_tbl = WQ ? sTbl : (const uint16_t*)nullptr;
#else
    (void)sTbl;
    const uint16_t* const r3d_tbl = (const uint16_t*)nullptr;
#endif

    const int lane = (int)threadIdx.x & (kWave - 1);
    const int warp = (int)threadIdx.x / kWave;
    const int m = lane & (MT - 1);
    const int n = (int)blockIdx.x * (kWaves * KLW) + warp * KLW + lane / MT;
    const bool live = n < N;

    /* A dead lane reads row N-1, a row that exists: the loads stay uniform and nothing is read out
     * of bounds; its result is dropped at the store. */
    const uint16_t* wrow = w + (long long)(live ? n : N - 1) * K;
    const int nsteps = M < MT ? M : MT;
    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += kChunkK) {
        for (int j = (int)threadIdx.x; j < STEPS; j += kBlockThreads) {
            const int r = j / (kChunkK / 8), c8 = j % (kChunkK / 8);
            const int k = k0 + c8 * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (r < nsteps && k < K) v = *(const uint4*)(x + (long long)r * x_ld + k);
            *(uint4*)&sX[srow(r, c8, kChunkK)] = widen8(v);
        }
        __syncthreads();

        if (live) {
            const uint16_t* wp = wrow + k0;
            const uint8_t* cp = h.codes + (long long)(live ? n : N - 1) * h.code_row + k0;
            /* One scale a chunk: kChunkK is the scale block, so the lane's 128 codes are inside a
             * single (n-block, k-block) and the load is one scalar per 128 codes. */
            const float sc = WQ ? scale_at(h, live ? n : N - 1, k0) : 0.0f;
            const int kend = K - k0 < kChunkK ? K - k0 : kChunkK;   /* a multiple of 8: the
                                                                     * launcher takes K % 8 == 0 */
            for (int kc = 0; kc < kend; kc += 8) {
                uint32_t w0, w1, w2, w3;
                if (WQ) {
                    uint32_t q[4];
                    codes8_to_f16x4(loadc8<NT>(cp + kc), h.cdt, r3d_tbl, q);
                    w0 = scale_pair(q[0], sc); w1 = scale_pair(q[1], sc);
                    w2 = scale_pair(q[2], sc); w3 = scale_pair(q[3], sc);
                } else {
                    const uint4 wv = loadw8<NT>(wp + kc);
                    w0 = bf16x2_to_f16x2(wv.x); w1 = bf16x2_to_f16x2(wv.y);
                    w2 = bf16x2_to_f16x2(wv.z); w3 = bf16x2_to_f16x2(wv.w);
                }
                const int s = srow_pair(m, kc, kChunkK);
                acc = dot2_f16x2(*(const uint32_t*)&sX[s], w0, acc);
                acc = dot2_f16x2(*(const uint32_t*)&sX[s + 2], w1, acc);
                acc = dot2_f16x2(*(const uint32_t*)&sX[s + 4], w2, acc);
                acc = dot2_f16x2(*(const uint32_t*)&sX[s + 6], w3, acc);
            }
        }
        __syncthreads();
    }

    if (live && m < M) {
        if (YF32) ((float*)y)[(long long)m * y_ld + n] = acc;
        else ((uint16_t*)y)[(long long)m * y_ld + n] = f32_to_bf16(acc);
    }
}

/* ------------------------------------------------------------------ the shared launcher
 *
 * Returns 0, -1 a null pointer, -2 a shape it does not serve, -3 a pitch it cannot read, -4 the HIP
 * launch failed, -5 an address it cannot read at 16 bytes.
 */
template <bool YF32>
int gemm_launch(long xv, long wv, long yv, int M, int N, int K, int x_ld, int y_ld, int sk, int kb,
                int nt, long stream, const HeadQ* hq = nullptr)
{
    if (!xv || !wv || !yv) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (sk > 1) return -2;                       /* the grid is vocab-wide at every M, see header */
    if (kb == 0) kb = kKC;
    if (kb != kKC) return -2;                    /* 32 is the tile's k-chunk and the only one */
    if (nt < 0 || nt > 1) return -2;
    if (K % kKC) return -2;                      /* the tile's chunk; 2048 at the band */
    if (K % 8) return -3;                        /* 16-byte weight loads */
    if (!x_ld) x_ld = K;
    if (!y_ld) y_ld = N;
    if (x_ld % 8) return -3;                     /* 16-byte activation loads (bf16) */
    if (y_ld % (YF32 ? 4 : 8)) return -3;        /* 16-byte store runs (f32: 4, bf16: 8) */
    if (x_ld < K || y_ld < N) return -3;
    if ((xv & 15) || (wv & 15) || (yv & 15)) return -5;

    const uint16_t* x = (const uint16_t*)(uintptr_t)xv;
    const uint16_t* w = (const uint16_t*)(uintptr_t)wv;
    void* y = (void*)(uintptr_t)yv;
    hipStream_t s = (hipStream_t)(uintptr_t)stream;

    /* The head's form is the shim's answer, resolved out of the container before the launch; this
     * is where its pointers are checked as pointers. The declared entries pass null and get the
     * plain bf16 head, which is what r3d.h's signature for them describes. */
    HeadQ h;
    std::memset(&h, 0, sizeof h);
    const int wq = hq != nullptr;
    if (hq) {
        h = *hq;
        if (!h.codes || !h.scale) return -1;
        if (h.sdt < 0 || h.sdt > 2 || h.cdt < 0 || h.cdt > 1) return -2;
        if (h.code_row & 7) return -3;        /* 8-byte code loads, one row a stride */
        if (h.scale_row & (h.sdt == 2 ? 3 : 1))
            return -3;                        /* a scale row is 4 bytes a column for f32, 2 for
                                               * bf16/f16: a plane of 3 columns is 6 bytes and the
                                               * loads land on even addresses. K = 320 is legal. */
        if (((long long)(uintptr_t)h.codes & 7) || ((long long)(uintptr_t)h.scale & 3)) return -5;
    }

    if (M <= kSkinnyMax) {
#define R3D_SKINNY(MTV, NTV, WQV)                                                                \
    do {                                                                                         \
        const int klw = kWave / (MTV);                                                           \
        const int per = kWaves * klw;                                                            \
        const dim3 grid((unsigned)((N + per - 1) / per)), block(kBlockThreads);                  \
        R3D_KLAUNCH(R3D_KNAME((logits_skinny_kernel<MTV, YF32, NTV, WQV>)), grid, block, s, x, w, \
                    h, y, M, N, K, x_ld, y_ld);                                                  \
    } while (0)
#define R3D_SKINNY_MT(MTV)                                                                       \
    do {                                                                                         \
        if (wq) { if (nt) R3D_SKINNY(MTV, 1, 1); else R3D_SKINNY(MTV, 0, 1); }                     \
        else    { if (nt) R3D_SKINNY(MTV, 1, 0); else R3D_SKINNY(MTV, 0, 0); }                     \
    } while (0)
        if (M == 1) R3D_SKINNY_MT(1);
        else if (M <= 2) R3D_SKINNY_MT(2);
        else if (M <= 4) R3D_SKINNY_MT(4);
        else if (M <= 8) R3D_SKINNY_MT(8);
        else R3D_SKINNY_MT(16);
#undef R3D_SKINNY_MT
#undef R3D_SKINNY
    } else {
        const dim3 grid((unsigned)((N + kTN - 1) / kTN), (unsigned)((M + kTM - 1) / kTM));
        const dim3 block(kBlockThreads);
        if (wq)
            R3D_KLAUNCH(R3D_KNAME((gemm_tiled_kernel<YF32, true>)), grid, block, s, x, w, h, y, M,
                        N, K, x_ld, y_ld);
        else
            R3D_KLAUNCH(R3D_KNAME((gemm_tiled_kernel<YF32, false>)), grid, block, s, x, w, h, y, M,
                        N, K, x_ld, y_ld);
    }
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The shim's shared body: the checks the two ops agree on, then the launch. `y_is_f32` is the op's
 * schema (logits are f32 because every sampler stage reads f32; gemm_nt's y is bf16), nkey/kkey the
 * axis names libref spells them with ("n_vocab"/"n_embd" and "N"/"K" -- ref_logits_gemm is
 * gemm_nt_impl with the axes renamed, so this body is shared by construction, not by imitation). */
int rad_gemm_run(const RadArgs* a, RadStream s, int y_is_f32, const char* nkey, const char* kkey)
{
    if (a->n_t < 3 || !a->t[0].data || !a->t[1].data || !a->t[2].data)
        return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = &a->t[0];
    const RadTensor* w = &a->t[1];
    const RadTensor* y = &a->t[2];

    const RadPlanesEntry* ce = nullptr;
    const RadPlanesEntry* se = nullptr;
    HeadQ h;
    std::memset(&h, 0, sizeof h);
    int joined = 0;

    /* ---- the head's stored form ---------------------------------------------------------------
     * The band's `dtype` key is the ACTIVATION's dtype: on this graph `dtype=bf16` with the head
     * stored as E4M3 codes and a bf16 scale per 128x128 block, in two planes of one container. The
     * stored form is answered by a layout hook -- `ref_layout_head`, transcribed below as
     * r3d_layout_logits_head, the way the w2a8 hook transcribed ref_layout_gemm_q -- and this is
     * its launch-side twin: read the header, read the two planes, dispatch on what they are. The
     * checks and their order are the reference's `scaled_rows`: the roles, then the dtypes and
     * blocks, then the coverage (which needs N and K, so it sits below, where the reference has
     * them already).
     *
     * The bf16 head is served as is: the container's header, not the row's preference, picks
     * the form. Two forms, one row, chosen by the bytes.
     */
    const RadPlanesHeader* ph = rad_planes_header(r3d_p(w));
    if (ph) {
        if (!y_is_f32) return R3D_NO(RAD_E_DTYPE);   /* logits are f32; the bf16-y row keeps a bf16 head */
        if (rad_planes_find(r3d_p(w), "zero", nullptr))
            return R3D_NO(RAD_E_DTYPE);   /* codes+scale+zero is gemm_nt_q's asymmetric grid: w2a8 */
        if (ph->n_planes != 2) return R3D_NO(RAD_E_DTYPE);
        const uint8_t* cb = (const uint8_t*)rad_planes_find(r3d_p(w), "codes", &ce);
        const uint8_t* sb = (const uint8_t*)rad_planes_find(r3d_p(w), "scale", &se);
        if (!cb || !sb || !ce || !se)
            return R3D_NO(RAD_E_DTYPE);   /* a joined head that is not codes + one scale */
        if (ce->kind != RAD_PLANE_TILED || ce->block[0] != 1 || ce->block[1] != 1)
            return R3D_NO(RAD_E_DTYPE);   /* codes must be one code an element, tiled plain */
        if (ce->dtype != RAD_F8E4M3 && ce->dtype != RAD_I8) {
            if (rad_dtype_bytes(ce->dtype, 8) < 8)
                return R3D_NO(RAD_E_DTYPE);   /* sub-byte codes: w4a16's ground, not this form */
            return R3D_NO(RAD_E_DTYPE);       /* e5m2 and the rest: no decoder here */
        }
        if (se->kind != RAD_PLANE_TILED) return R3D_NO(RAD_E_DTYPE);
        if (se->dtype == RAD_BF16) h.sdt = 0;
        else if (se->dtype == RAD_F16) h.sdt = 1;
        else if (se->dtype == RAD_F32) h.sdt = 2;
        else return R3D_NO(RAD_E_DTYPE);        /* an integer scale is not a float_scale_dt */
        if (se->block[0] != kFp8Block || se->block[1] != kFp8Block)
            return R3D_NO(RAD_E_SHAPE);   /* per-(row,group) scales are gemm_nt_q's grid; per-row
                                           * is not the indexing this kernel does */
        h.codes = cb; h.scale = sb; h.code_row = ce->row_bytes; h.scale_row = se->row_bytes;
        h.cdt = ce->dtype == RAD_I8 ? 1 : 0;
        joined = 1;
    } else if (w->dtype != RAD_BF16) {
        return R3D_NO(RAD_E_DTYPE);
    }
    if (x->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (y->dtype != (y_is_f32 ? (uint32_t)RAD_F32 : (uint32_t)RAD_BF16)) return R3D_NO(RAD_E_DTYPE);

    long long N = r3d_geti_or(a, nkey, 0);
    long long K = r3d_geti_or(a, kkey, 0);
    if (N <= 0) N = r3d_dim(w, 0);
    if (K <= 0) K = r3d_dim(w, 1);
    if (N <= 0 || K <= 0) return R3D_NO(RAD_E_SHAPE);

    long long M = 0, x_rows = 0;
    const long long y_pitch = r3d_rows_pitch(y, N, &M);      /* M = numel(y)/N, the reference's fold */
    if (y_pitch < N || M <= 0) return R3D_NO(RAD_E_STRIDE);
    const long long x_pitch = r3d_rows_pitch(x, K, &x_rows);
    if (x_pitch < K || x_rows < M) return R3D_NO(RAD_E_STRIDE);
    if (r3d_numel(w) < N * K) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(w)) return R3D_NO(RAD_E_STRIDE);         /* N rows of K contiguous codes */
    if (joined) {
        /* The coverage checks, `scaled_rows`' last block: the planes must cover the band, and the
         * scale plane covers it by blocks -- ceil, so a ragged vocabulary is fine (131 rows need 2
         * scale rows) and a plane one row short is not. */
        const long long sb_r = (N + kFp8Block - 1) / kFp8Block;
        const long long sb_c = (K + kFp8Block - 1) / kFp8Block;
        if (ce->rows != N || ce->cols != K) return R3D_NO(RAD_E_SHAPE);
        if (se->rows != sb_r || se->cols != sb_c) return R3D_NO(RAD_E_SHAPE);
        if (ce->row_bytes < ce->cols || se->row_bytes < se->cols * (h.sdt == 2 ? 4 : 2))
            return R3D_NO(RAD_E_STRIDE);
    }
    if (M > (long long)1 << 24 || N > (long long)1 << 28 || K > (long long)1 << 28)
        return R3D_NO(RAD_E_SHAPE);
    if (!r3d_aligned(r3d_p(x), 16) || !r3d_aligned(r3d_p(w), 16) || !r3d_aligned(r3d_p(y), 16))
        return R3D_NO(RAD_E_ALIGN);

    const int sk = (int)r3d_geti_or(a, "sk", 1);
    if (sk > 1) return R3D_NO(RAD_E_SHAPE);   /* the grid is vocab-wide at every M, see the header */
    const int kb = (int)r3d_geti_or(a, "kb", 0);
    const int nt = (int)r3d_geti_or(a, "nt", 0);

    const HeadQ* hp = joined ? &h : nullptr;
    const int e = y_is_f32
        ? gemm_launch<true>(r3d_addr(x), r3d_addr(w), r3d_addr(y), (int)M, (int)N, (int)K,
                            (int)x_pitch, (int)y_pitch, sk, kb, nt, (long)(uintptr_t)s, hp)
        : gemm_launch<false>(r3d_addr(x), r3d_addr(w), r3d_addr(y), (int)M, (int)N, (int)K,
                             (int)x_pitch, (int)y_pitch, sk, kb, nt, (long)(uintptr_t)s, hp);
    if (e == 0) return RAD_OK;
    return R3D_NO(e == -1 ? RAD_E_INVAL : e == -2 ? RAD_E_SHAPE
                        : e == -3 ? RAD_E_STRIDE : e == -5 ? RAD_E_ALIGN : RAD_E_DEVICE);
}

/* The reference's `enc_scaled_rows` and `float_scale_dt`, inlined: libref keeps them static inline in
 * its own header, so a plugin that transcribes ref_layout_head transcribes them with it. Checks and
 * order are its, unchanged: affine, untransformed, two planes, the roles in order, codes tiled with
 * block {1,1} and a code dtype it admits, the scale tiled with a float dtype. */
bool float_scale_dt(uint32_t dt)
{
    return dt == RAD_BF16 || dt == RAD_F16 || dt == RAD_F32;
}

bool enc_scaled_rows(const RadEncoding* enc, const int* sel, int n)
{
    if (!rad_enc_is(enc, "affine") || enc->transform[0] || enc->n_planes != 2) return false;
    if (n != 2 || !sel) return false;                 /* sel_roles(enc, sel, n, "codes", "scale") */
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

/* logits[M, n_vocab] = x[M, n_embd] @ lm_head[n_vocab, n_embd]^T, the declared entry (r3d.h's
 * r3d_vocab_bf16.hip section). M <= 16 takes the decode arm, above it the prefill tile. */
extern "C" int r3d_logits_gemm_bf16(long x, long lm, long y, int M, int n_vocab, int n_embd,
                                    int x_ld, int y_ld, int sk, int nt, long scratch, long stream)
{
    (void)scratch;                       /* no split, so no partials: see the header */
    return gemm_launch<true>(x, lm, y, M, n_vocab, n_embd, x_ld, y_ld, sk, kKC, nt, stream);
}

/* The same tiling with a bf16 y, for gemm_nt at prefill sizes (M >= 17; M <= 16 is the m16 row). */
extern "C" int r3d_gemm_nt_bf16_tiled(long a, long b, long y, int M, int N, int K, int a_ld,
                                      int y_ld, int sk, int kb, int nt, long scratch, long stream)
{
    (void)scratch;
    return gemm_launch<false>(a, b, y, M, N, K, a_ld, y_ld, sk, kb, nt, stream);
}

extern "C" int r3d_rad_logits_gemm_bf16(const RadArgs* a, RadStream s)
{
    return rad_gemm_run(a, s, 1, "n_vocab", "n_embd");
}

extern "C" int r3d_rad_gemm_nt_bf16_tiled(const RadArgs* a, RadStream s)
{
    return rad_gemm_run(a, s, 0, "N", "K");
}

/* The head's layout hook, `ref::ref_layout_head` transcribed: operand 1 of
 * logits_gemm, two planes, codes and one scale, untransformed -- and the answer is the container's
 * own form. It moves no byte, and its `relayout` writer (below) exists for the round-trip: this is
 * a reader, which is what
 * the mapped tier asks for when a weight's stored form is not the container's. Operand 0 (the
 * activations) and a one-plane selection are declined as `layout_scaled` declines them, and so is
 * every other encoding: a silent no-op on a head is the worst failure a row can have. */
/* The head's WRITER. A layout query that answers RAD_OK is a promise the ABI makes to the loader --
 * "out describes the stored form, and relayout will produce it" (rad_abi.h) -- and the load step
 * holds the row to it. The container stores the quantiser's pair; the plan wants one operand; the
 * row that owns the band assembles it, losslessly, with rad_planes_relayout: the header, the
 * entries and each plane's rows, copied.
 * No code is transformed and no number is derived -- a kernel doing arithmetic
 * here would make the served numbers depend on which library loaded the container (rad_abi.h).
 * The query and the writer share the plan through rad_planes_plan, so they cannot disagree. The
 * operand guard is the hook's own rule repeated: a call routed to the wrong slot writes nothing. */
extern "C" int r3d_relayout_logits_head(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* planes, int n_planes, void* dst,
                                        int64_t dst_bytes)
{
    (void)p; (void)n_p;
    if (operand != 1) return R3D_NO(RAD_E_UNSUPPORTED);
    return rad_planes_relayout(enc, sel, planes, n_planes, dst, dst_bytes);
}

extern "C" int r3d_layout_logits_head(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel,
                                      const RadTensor* planes, int n, RadLayout* out)
{
    (void)p; (void)n_p;                                        /* layout_scaled ignores both */
    if (n < 2 || operand != 1) return RAD_E_UNSUPPORTED;       /* layout_scaled(operand, 1, -1, ...) */
    if (!enc_scaled_rows(enc, sel, n)) return RAD_E_DTYPE;
    return rad_planes_layout(enc, sel, planes, n, out);
}
