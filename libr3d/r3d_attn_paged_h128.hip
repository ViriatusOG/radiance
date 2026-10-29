/* r3d_attn_paged_h128.hip -- paged attention at head_dim 128, the two bands MiniCPM5-2B declares.
 *
 *     out[tok, h, :] = softmax_j( scale * q[tok, h, :] . K[j, h/gqa, :] ) . V[j, h/gqa, :]
 *
 * over a paged cache, in libref's order. The order is the point of the file: a
 * tolerance on an attention output is not where a summation-order difference hides, so this kernel
 * keeps the chain the reference writes, and says out loud where it cannot.
 *
 * ============================== the layout ==============================
 *
 *     kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]      K then V inside a slot
 *
 * A slot's K is 128 elements and V the 128 after it, so a key's row is a 512-byte pair and both
 * halves are 16-byte aligned at every slot. `block_table[seq, j / block_size]` names the block
 * holding key j; the fixture *draws* its entries over [0, max_blocks), so ids repeat and one block
 * serves several keys. A negative entry is an unmapped block and hides its key -- the engine's
 * allocator frees a sequence by writing -1 into rows it is done with, so the kernel keeps the test
 * even though the fixture never sets it. `seqused[seq]` is the sequence's TOTAL key count, this
 * step's query rows included, so query row i of a len_s-row step sits at absolute position
 * `seqused[s] - len_s + i`. One kernel serves decode (q_len 1), a speculative verify (q_len 7) and a
 * prefill chunk (q_len 2048).
 *
 * ============================== the ragged batch ==============================
 *
 * A step's sequences do not have to contribute the same number of query rows. `cu_seqlens`, when
 * the caller has it, is [n_seq + 1] of i32 with cu[0] == 0 and cu[n_seq] == the query plane's row
 * count, and sequence s owns the rows [cu[s], cu[s+1]) -- `len_s = cu[s+1] - cu[s]` rows, whose
 * positions run from `seqused[s] - len_s`. That is libref's reading and not a local invention:
 * its shape hook (`cu[n_seq] == rows`, `q_len = cus ? 0 : rows / n_seq`) and its walk, where
 * the row's sequence, its length and its index within the sequence come from the descriptor and the
 * position from the sequence's own length. So `q_len` is a PER-SEQUENCE number in the reference and
 * in this file; the total row count is the plane's, and a ragged batch has no q_len at all.
 *
 * One grid covers a ragged batch because the clamp the width section documents is per sequence: a
 * tile is `ceil(len_s / rows_per_tile)` tiles for sequence s, and the slots of its last tile past
 * `len_s` compute a duplicate that stores nothing. The rows are laid out by `map_seq`, which walks
 * the descriptor once a workgroup -- a rectangular batch is the same walk with len_s = q_len and
 * cu[s] = s * q_len, so there is one mapping, one mask, one walk and one store for both forms.
 * The causal mask, the window, the block-table walk and the descales all key off the sequence the
 * row belongs to, exactly as the reference's `s = seq_of(cus, n_seq, tok)` keys them.
 *
 * ============================== the mask ==============================
 *
 *   causal = 1   key j visible when j <= pos
 *   causal = 0   key j visible when j <  ctx
 *   window > 0   causal: additionally pos - j < window; non-causal: |pos - j| < window, the
 *                two-sided reading libref's prose and libr4d's window kernel both state.
 *
 * A masked key is skipped, never scored to -inf: the denominator gets exactly the terms the
 * reference has, and a row with no visible key stores zeros, which is libref's `mx == -INFINITY`
 * case and every other kernel in the project.
 *
 * ============================== the tile ==============================
 *
 * An output row is (query row, query head): 128 dims, one f32 chain per dim. A lane owns one output
 * row, exactly as the skinny GEMM rows do, and for the same reason -- a lane's chain over keys is
 * the reference's chain, and no cross-lane reduction touches the denominator. A wave32 is 32 output
 * rows, and they must be 32 rows that read the same KV, which fixes the mapping:
 *
 *     kh = blockIdx.y      h = kh * GQA + (lane & (GQA-1))      tok = base + (lane >> lg GQA)
 *
 * so a wave is the GQA query heads of one KV head over 32/GQA query rows -- 8 heads x 4 rows at
 * gqa = 8. Every lane of a wave then reads the same cache row at the same moment, which makes every
 * cache read in this file an LDS broadcast (one `ds_read_b128` serves the wave) and makes the paged
 * walk a per-wave event rather than a per-lane one. A workgroup is WAVES waves = WAVES x 32 threads
 * = WAVES x (32/GQA) query rows of one sequence, one KV head: 16 rows at gqa 8 with four waves.
 * WAVES is a template parameter and the launcher picks it -- see "the width of a workgroup" below.
 *
 * Rows per cache read is 32. It is also the ceiling the register file sets: a lane carrying a second
 * output row needs 2 x 64 VGPRs of widened query and 2 x 128 of f32 accumulator, 384, over the 256
 * that four waves a SIMD allows. So the traffic is what gqa and the register file say it is.
 *
 * ============================== the width of a workgroup ==============================
 *
 * A row slot whose row is past its sequence's length computes a clamped duplicate: the walk, the
 * tile and the store all run for it and the result is dropped. At the decode band's q_len = 7 with
 * gqa = 8 a four-wave workgroup carries 16 row slots for 7 rows -- twice the walk for half the
 * outputs. So the launcher picks the narrowest workgroup that carries the caller's q_len (pick_waves:
 * one wave to 32/GQA rows, two to 64/GQA, four above), and prefill keeps four because its q_len is the
 * whole chunk. At the decode band that is two waves: 512 workgroups of 64 threads, 7 of 8 slots used.
 * A ragged batch has no one q_len to pick on: the launcher picks on the average sequence
 * (`ceil(rows / n_seq)`), which is a performance choice and nothing else -- the tile width moves
 * duplicate rows between tiles, never a result, and no host code reads the descriptor.
 *
 * Measured on the card: medians of 15 per-launch event times, both widths in one session, so the
 * clock drifts together (both read the prefill band at 44.9):
 *
 *     decode 256 x 7 @4096       294.7 ms -> 158.4 ms   -46%
 *     decode 256 x 7 @4096 sp4   268.0 ms -> 128.4 ms   -52%
 *     decode   8 x 7 @2048 sp8     4.33 ->   2.06 ms    -53%
 *     decode   1 x 7 @4096 sp8     4.27 ->   3.97 ms     -7%
 *     prefill 2048 x 2048         44.95 ->  44.78 ms     the path is unchanged
 *
 * The tile is the same 8 KiB at every width, so nothing about the cache traffic changes; the win is
 * the duplicated rows, and the reason it is this large is the one the counters show: the kernel
 * issues 15% of the SIMD's slots at the band (SQ_INSTS_VALU 4.5G against 30G of capacity,
 * SQ_INSTS_LDS 769M) and 3.7% at eight sequences. Halving the arithmetic on a kernel that is
 * stalling is not supposed to buy 46%; it does, and that says the stall is a function of how many
 * waves a workgroup has to bring to a barrier -- two arriving is not twice as cheap as four, it is
 * four times cheaper. The width is a run-time choice because it is the only lever on this op that
 * moves the band.
 *
 * ============================== measured, and not the answer ==============================
 *
 * Four things that look like they should work, each measured to not (medians of 15 per-launch
 * hipEventElapsedTime, A/B in one process), so the next reader does not spend an hour on them:
 *
 *   s_setprio 1 around the consume leg of the walk, 0 after: |delta| <= 0.7% on the bands. The
 *   knob is fine -- `s_setprio` takes an immediate, not an SGPR, so `s_setprio %0` will not link --
 *   but priority does not move this op.
 *
 *   R3D_WAVES(2) to lift the scalar-register ceiling: the bf16 arms report sgpr_count 107 and
 *   sgpr_spill_count ~98 at BOTH 4 and 2 waves a SIMD, and the timings do not move. The
 *   compute-shader scalar budget is not set by waves/SIMD the way the VGPR one is, so this is not
 *   a lever for the spills.
 *
 *   __noinline__ on the stage leg: the spills drop (98 -> 72) and a 160-byte private frame
 *   appears, timings unchanged. The spills are a load-safety matter, not a time matter: while the
 *   bf16 arms spill SGPRs, a kernel in this TU that also needs a private segment fails to launch
 *   with "the operation cannot be performed in the present state".
 *
 *   A two-stage pipeline -- two tiles, the next block's global loads issued before the current
 *   block is consumed, one barrier a block instead of two: a wash on the bands. The walk's
 *   per-block structure is the cost, and the two legs do not overlap at all -- the workgroup's
 *   width is the lever.
 *
 * ============================== the walk ==============================
 *
 * block_size = 16, so keys advance 16 at a time and one step stages a block of the KV head: 16 slots
 * x 128 elements, K and V, widened to f16 pairs on the way in, 4 KiB each. The four waves of a
 * workgroup share the stage, so a block of cache is one global read per workgroup per pass, and the
 * walk costs 8 KiB of the 64 KiB of LDS. The second pass re-reads the same blocks; a sequence's
 * cache for one KV head is ctx x 256 B, which at 4096 keys is 1 MiB, so the second pass is an L2
 * hit and the card's DRAM sees each byte once.
 *
 * The mask sits where the geometry puts it: a wave's query rows are consecutive, so at gqa 8 its
 * four positions span 3 and its diagonals lie in one 16-key block, two at a block boundary. Blocks
 * entirely below the wave's minimum position are walked with no mask code at all; the one or two
 * diagonal blocks test `j <= pos` per lane; blocks past the wave's maximum position are not walked.
 * In a 2048-token causal prefill that is one masked block in sixty-four.
 *
 * ============================== the two passes ==============================
 *
 * The reference is two-pass: pass one every visible key's score and the maximum; pass two, ascending
 * keys, `w = exp(score - max)`, `sum += w`, `acc[d] += w * v[d]`, then `acc * (1/sum)` and one
 * narrowing at the store. Online softmax -- the flash form, rescaling a running accumulator by
 * `exp(m_old - m_new)` -- brackets the denominator and every acc[d] differently, so it is not in
 * this file. Keeping the scores as the reference does costs a workspace of jcap floats a row: 32
 * rows x 4096 keys = 512 KiB a wave against 64 KiB of LDS. So the two passes are two walks, and
 * pass two recomputes the scores it needs -- same order, same operands, bit-identical the second
 * time. QK is paid twice and the op costs half again what keeping the scores would; that is the
 * price of the reference's parentheses.
 *
 * QK is 64 `v_dot2acc_f32_f16` a key a lane over widened f16 pairs, ascending d. A dot2 folds a
 * pair's two products into the accumulator in one rounding where the reference rounds per product,
 * so a lane's score chain brackets the reference's: the same one-ulp departure the bf16 GEMM row
 * documents, on a fraction of a percent of outputs, one bf16 store ulp each.
 *
 * P . V is 128 f32 multiply-accumulates a key a lane, `acc[d] += w * v[d]` with -ffp-contract=off,
 * which is the reference's expression with vds already folded into the staged V. dot2 cannot do this
 * leg in this layout: w is a per-lane scalar and the accumulation is per dim, so the only route to
 * the matrix path is to fold w into V -- and w is f32, so every weight of the sum would be read to
 * 11 bits, a ~1e-4 relative change on the output that survives the bf16 store as a few percent of
 * outputs moved one ulp. It is a third of the instruction count, and it is not the default; the
 * numbers this unit is measured with are the f32 leg's.
 *
 * ============================== exp ==============================
 *
 * `ref_exp` is `(float)std::exp((double)x)`: the reference's exponent is correctly rounded to f32
 * through a double. RDNA3 has no vector f64 -- double is soft-float, hundreds of cycles a call, and
 * a call per key per lane is the whole kernel. The device libm's `expf` is a ~1-ulp polynomial and
 * `__expf` is two orders of magnitude worse. `exp_ref` is a software expf: two-term range reduction
 * against a split ln2, a degree-8 polynomial, and 2^k assembled in the integer domain (no libm call
 * in the hot path). It is the only place in this library where an explicit `__fmaf_rn` is allowed,
 * and the rule it is an exception to is the no-FMA rule: that rule exists because libref's
 * expressions contain no FMA and a contracted multiply-add changes the order of rounding in
 * arithmetic we are required to reproduce. Here the thing we must reproduce is a *function* libref
 * computes in double precision on hardware that has f64 and this card does not. The polynomial's
 * FMA is arithmetic we chose, scoped to this helper, to get the closest legal transcription of that
 * function; it contracts no expression the reference wrote.
 *
 * `exp(0)` is exactly 1.0: k is 0, the reduction leaves r exactly 0, and the polynomial's constant
 * term passes through untouched. The argument here is always <= 0 (the max is taken over the same
 * set the exponentials are taken of), so the domain that matters is x <= 0. Below -103.98 the
 * result is under f32's smallest subnormal and the helper returns 0; the reference's
 * `(float)exp(-100)` is a denormal of order 3.7e-44, and a term that small added to a denominator
 * of order 1 is not a change of any kind, let alone one that survives a bf16 store -- which is the
 * reason the flush is legal and not an approximation to defend. Between -103.98 and -87.3 the
 * result is subnormal, and the helper reaches it by one hardware multiply by 2^-126, so it is
 * rounded the way the hardware rounds and not by hand.
 *
 * ============================== split-KV ==============================
 *
 * The decode entry takes `splits`: a sequence's blocks are cut into slices of a whole number of
 * blocks, each slice accumulating its own (acc[128], sum, max) into scratch, and a second kernel
 * combines the slices before the single narrowing. Combining needs the slices' maximum, so each
 * slice's partial is rescaled by `exp(m_slice - M)`: the denominator becomes
 * `sum = Σ_slices sum_s * exp(m_s - M)`, ascending in slice, and every term arrives as
 * `exp(s - m_s) * exp(m_s - M)` where the reference has `exp(s - M)`. Both are last-bit at the f32
 * level -- ~2 ulp per term, ~1 ulp on the sum -- and both vanish under the bf16 store except on
 * outputs sitting on a rounding boundary. The row's prose says `splits > 1` is not bit-identical to
 * the reference and `splits = 1` is.
 *
 * `splits` is a maximum, resolved per sequence from its own `seqused`: `nslice = ceil(nblk /
 * ceil(nblk / splits))` with nblk the sequence's block count, so a short sequence uses fewer slices
 * and there is no per-sequence legality check to fail. `nslice` is recomputed by the reducer from
 * the same two numbers, so the partials need no header beyond (acc, sum, max).
 *
 * ============================== what the shims refuse ==============================
 *
 * head_dim other than 128 (`RAD_E_SHAPE`; this file is named for it); a cache dtype other than bf16
 * and fp8_e4m3 -- an f32 cache has no dot2 path, and u8/u4/int8 codes are not this cache form
 * (`RAD_E_DTYPE`); a query dtype other than bf16 or f16, and an out dtype that is not the query's
 * (`RAD_E_DTYPE`); gqa outside {1,2,4,8} or not dividing n_head (the wave's head split is a shift);
 * block_size other than 16 (the stage is sized for one); q_len < 1 in a rectangular batch, and a
 * row count < 1 in a ragged one; a row count that does not fit the plane's 32-bit row indexing
 * (num_seqs * q_len, or cu[n_seq], past 2^31); a q, kv or out stride that puts a slot's halves off
 * a 16-byte boundary, or a row pitch that is not a multiple of 8; a `cu_seqlens` that is not i32
 * (`RAD_E_DTYPE` -- the kernel reads int, and the fix is an i32 plane, which is what the op
 * describes), is not contiguous (`RAD_E_STRIDE`), is not [n_seq + 1] (`RAD_E_SHAPE`), or is not
 * 4-byte aligned (`RAD_E_ALIGN`); descales attached to a bf16 cache, which the reference ignores
 * and this kernel refuses rather than drops, and an fp8 cache missing them (`RAD_E_SHAPE`); an
 * output shorter than rows * n_head * head_dim; and a scratch arena too small for the partials the
 * requested `splits` needs, which the launcher answers by lowering `splits` to what the arena
 * holds, down to 1, which needs none.
 *
 * What the descriptor carries that the host cannot see is clamped, not refused, because a launcher
 * reads no device memory: cu[0] == 0, monotonicity and cu[n_seq] == the plane's row count are the
 * description's promises (the reference's shape hook states all three), and the kernel's answer to each
 * is the reference's own. Rows past the plane are dropped (`row_ok` tests the plane's row count, so
 * a descriptor that runs long cannot write outside `out`); a segment with cu[s+1] <= cu[s] has no
 * tiles and contributes no rows, which is the reference's `if (qlen <= 0) return;`
 * (the reference's walk); and rows under cu[0] belong to sequence 0 with a negative index into its
 * query, which is what `seq_of` gives them.
 */
#include "r3d.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_args.h"
#include "r3d_launch.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>   /* size_t, in the arena indexing and the shim's scratch sizing */
#include <cstdio>    /* the device print, behind the budget; the shims already use std::fprintf */
#include <cstdlib>   /* std::getenv / std::atoi, in the arming */

/* ================================================================== the device record ring
 *
 * THE LAW. A kernel that calls device `printf` is given an extra implicit argument --
 * `hidden_hostcall_buffer` -- and the engine's AQL dispatcher does not provide it, so a kernel that
 * prints is UNDISPATCHABLE BY THE ENGINE: the instrument is not merely silent, the row is refused,
 * and a gate that hipLaunches by hand cannot see it. The kernel therefore writes; it does not speak.
 * The shim drains the ring to stderr, which is a host facility and costs the kernel nothing.
 *
 * Four plugin-owned device globals, and nothing else is added to the kernel: the launch parameters
 * are fixed, `RadArgs` is the row's ABI and the plan builds it, so a field cannot join it (and `abi/`
 * is untouched). The kernel owns these; the shim resolves them with `hipGetSymbolAddress` (once,
 * cached) and, ONLY under `RAD_R3D_TRACE`, writes the budget with `hipMemcpyToSymbol`.
 *
 *     r3d_dev_budget    records left to write; 0 -- its initial value -- means record nothing
 *     r3d_dev_launches  launches seen, incremented once per launch by the same thread that records
 *     r3d_dev_widx      the ring's write index, bumped once per record written
 *     r3d_dev_ring      kDevRing records, a fixed struct, 216 bytes each: 55 KiB, allocated once
 *
 * Why they exist: the engine's recorded pass plays a decode step by writing queue packets, and a
 * play never reaches a plugin shim. The shim is not in the path of a played step. The kernel is. So
 * the kernel writes down what it was handed: the step's `ctx` as the row reads it off `seqused`, the
 * block ids as the walk reads them off the table, the operand addresses baked into the packet, and
 * three content folds -- the query row, the K row of the FIRST block the row walks, the K row of the
 * LAST one -- so a block whose BYTES move underneath a step is visible: the adopted prefix whose
 * recycled block still holds the previous tenant's KV, and the address slip that serves fluent,
 * plausible, wrong text.
 *
 * A played step's record is written on the device and printed at the NEXT shim call, so a served
 * log's lines land one step behind the step they describe. The
 * record's own `L` and `ctx` restore the order, and the shim's plan line quotes the same counter, so
 * the gap between the two prints is the played pass, measured.
 *
 * Inert when the flag is unset: the budget is 0, so a launch pays one uniform scalar load of a
 * global and one not-taken branch, on one thread of one workgroup, plus the atomic increment of the
 * counter. The bound is decremented with `atomicSub` -- 16 CUs cannot record 16 copies of one launch
 * -- and the counter is `atomicAdd` before it, so a launch counted is a launch that ran, played ones
 * included. The ring is bounded (kDevRing slots): a harness that arms more than the ring holds loses
 * records, never the count, and the shim says so on one line.
 *
 * The counter counts every `attn_kernel` launch, budget or not. It does NOT count the split-KV
 * reduce: that kernel is a second launch of a step's work, not a step. */
enum { kDevRing = 256, kDevIds = 8 };

/* One launch, as the launch saw it. A fixed plain-old struct: no pointers, no float, no padding the
 * reader has to guess. The ids are the row's first kDevIds columns -- `block_size` 16 and a served
 * ladder's 21 to 33 token contexts make that two columns, so the whole row fits -- and `idsum` folds
 * every column, so a 4096-token context still has one comparable number for its row. */
struct DevRec
{
    unsigned long long L;                        /* the launch's own ordinal                */
    long long ctx, pos, slot0, slotL, idsum;     /* the step as the row reads it, and where */
    unsigned long long qh, kh, khL;              /* three content folds                     */
    unsigned long long kv, bt, su, cu, q, out;   /* the six addresses baked into the packet */
    int dec, n_seq, rows, q_len, s, len, span, start;
    int jcap, splits, waves, ncols, rowok, spare, spare2, spare3;
    int b[kDevIds];                              /* the row's ids; `ncols` of them are real */
};

extern "C" __device__ int r3d_dev_budget = 0;
extern "C" __device__ unsigned long long r3d_dev_launches = 0ULL;
extern "C" __device__ unsigned r3d_dev_widx = 0u;
/* `= {}` is load-bearing: an uninitialised `__device__` variable emits no host-side symbol, and
 * the engine's loader dlopens the plugin with the symbol live, so it stops with
 * `undefined symbol: r3d_dev_ring` before a single row is declared. The zero fill is the device
 * side's init, and it is what the flag-unset state is. */
extern "C" __device__ DevRec r3d_dev_ring[kDevRing] = {};


namespace {

typedef float r3d_f32x4 __attribute__((ext_vector_type(4)));

constexpr int kHeadDim  = 128;
constexpr int kPairs    = kHeadDim / 2;        // f16 pairs a head: one dword is two dims
constexpr int kBlock    = 16;                  // block_size: a step stages one of these
constexpr int kWaves    = 4;                   // 128 threads
constexpr int kThreads  = kWaves * 32;
constexpr int kPitch    = kPairs;              // dwords a staged row; rows are read as broadcasts
constexpr int kTileDw   = kBlock * kPitch;     // 1024 dwords = 4 KiB
constexpr int kPartStride = kHeadDim + 4;      // acc[128], sum, max, pad -- 528 B, whole 16 B rows

/* ------------------------------------------------------------------ exp */

/* exp(x) narrowed to f32, as close to `(float)std::exp((double)x)` as an ALU with no f64 gets.
 *
 * Measured against `(float)std::exp((double)x)` over 1.31M points of [-100, 0] and the boundary
 * cases: every point is within one f32 ulp, 76.1% bit-identical (the device libm's expf is 94.0%
 * identical and no better in ulps), exp(0) is exactly 1. The tail below 2^-126 -- x under about
 * -87.3 -- comes out zero, not subnormal: the scaling multiply that assembles the subnormal is
 * flushed by the hardware. That is invisible at the store, which is where the reference is
 * compared: a weight of 1e-38 next to a maximum weight of exactly 1 cannot move an accumulator, and
 * the bf16 store has no resolution at that ratio.
 *
 * Reduce by ln2 against a two-term constant: k = rn(x / ln2), r = x - k*LN_HI - k*LO with LN_HI the
 * high 24 bits of ln2 and LO what is left. A single-term reduction is what makes the naive version
 * 1.5 ulp -- k*ln2 is not representable, and the subtraction's cancellation is the whole error
 * budget. |r| <= 0.35, where a degree-8 polynomial reaches ~0.55 ulp, then 2^k by exponent
 * assembly. */
__device__ __forceinline__ float exp_ref(float x)
{
    /* Above the overflow point the reference's own double path yields +inf, so this one must too;
     * a NaN in, a NaN out. Both cases are outside the softmax's reach (the argument is a score
     * minus the row's maximum, never positive), and are here so the function is an exp, not a
     * softmax helper with an undefined edge. */
    if (!(x < 88.7228390525719f))
        return x == x ? __int_as_float(0x7f800000) : __int_as_float(0x7fc00000);
    if (x < -103.9720770839916f) return 0.0f;                  /* under the subnormals: see header */
    const int k = (int)__float2int_rn(x * 1.4426950408889634f);
    const float LN_HI = 0.693145751953125f;                    /* ln2, high bits, exactly a float */
    const float LN_LO = 1.42860681039557e-06f;                 /* ln2 - LN_HI */
    float r = x - (float)k * LN_HI;
    r = r - (float)k * LN_LO;
    /* e^r, |r| <= 0.35: 1 + r + r^2/2 + sum(r^i / i!), i = 3..8, Estrin over r^2. */
    const float c8 = 1.98412698412698e-04f, c7 = 1.38888888888889e-03f;
    const float c6 = 8.33333333333333e-03f, c5 = 4.16666666666667e-02f;
    const float c4 = 1.38888888888889e-01f, c3 = 1.66666666666667e-01f;
    const float r2 = r * r;
    const float p4 = __fmaf_rn(__fmaf_rn(__fmaf_rn(c8, r, c7), r, c6), r, c5);
    const float p2 = __fmaf_rn(p4, r2, __fmaf_rn(c3, r, 5.0e-01f));
    const float e = __fmaf_rn(p2, r2, __fmaf_rn(r, 1.0f, 1.0f));   /* in [0.70, 1.42] */
    /* Scale by 2^k in the exponent field; x <= 0 here, so the normal case is the whole story and
     * the subnormal one is a single hardware multiply. */
    const int be = __float_as_int(e);
    const int field = ((be >> 23) & 0xff) + k;
    if (field >= 1) return __int_as_float((be & 0x007fffff) | (field << 23));
    if (field <= -23) return 0.0f;
    const float t = __int_as_float((be & 0x007fffff) | ((field + 126) << 23));
    return t * __int_as_float((127 - 126) << 23);              /* one rounding, the hardware's */
}

/* ------------------------------------------------------------------ the kernel's view */

struct Args {
    const uint16_t* q;                 /* bf16 or f16 (q16), rows x n_head x head_dim */
    const uint16_t* kv;                /* bf16 cache */
    const uint8_t*  kv8;               /* E4M3 cache, when FP8 */
    const int*      bt;                /* block table, i32: the shim refuses i64 indices */
    const int*      su;                /* seqused, i32 */
    const int*      cu;                /* cu_seqlens, n_seq + 1, i32; null = q_len rows each */
    const float*    kds;
    const float*    vds;
    uint16_t*       out;
    float*          part;              /* split-KV partials */

    /* Strides are elements, and 32-bit: a cache is blocks x kv_heads x 16 x 256 elements, a row
     * pitch is n_head x head_dim, a block table pitch is max_blocks, and the launcher checks that
     * each fits. Every address in this kernel is a 64-bit base plus a 32-bit product, and the
     * register allocator is out of scalar registers long before it is out of vector ones -- the
     * wavefront runs in compute-shader mode, where a scalar spill is not a slow path, it is a
     * kernel the runtime refuses to load. */
    int sb, sh, so, sd;                /* cache strides */
    int q_rs, o_rs;                    /* row pitches; the head stride is head_dim */
    int bt_rs;   /* the block table row pitch, in entries: the tensor own stride */
    int bt_cs;   /* its column stride: 1 for every dense table the engine builds */                         /* block table pitch */
    int n_seq, q_len, rows, n_head, kv_heads, causal, window, jcap, splits, q16, waves;
    float scale;
    DevRec* trace_buf;               /* one record slot, a plain hipMalloc buffer: the device
                                       * globals are stubs in a dlopened plugin, the buffer is not */
};

__device__ __forceinline__ long long ld_ix(const int* p, size_t i) { return p[i]; }

/* ------------------------------------------------------------------ the row map
 *
 * One map for both batch forms. With a descriptor, sequence s owns the query rows [cu[s], cu[s+1])
 * -- `len_s` rows, the row's index within the sequence `i = row - cu[s]`, its absolute position
 * `seqused[s] - len_s + i`. Without one, the same walk with len_s = q_len and cu[s] = s * q_len,
 * which is the reference's own rectangular reading, not a parallel implementation
 * of it.
 *
 * A grid row is resolved by subtracting each sequence's tile count from blockIdx.x until the
 * remainder fits: n_seq int loads and n_seq subtractions, once per workgroup, against a walk of
 * nblk blocks with two barriers each. The descriptor is device memory and a launcher reads none of
 * it, so the grid is sized on the bound in `launch` and a grid row past the batch's last tile comes
 * back with tile < 0 and leaves, having staged nothing.
 *
 * `start` is the sequence's first dense row and `i0` the row its positions count from. They differ
 * only when cu[0] > 0: the reference's `seq_of` hands rows under cu[0] to sequence
 * 0 with a negative index into its query, and that is transcribed rather than refused because a
 * descriptor that does not start at 0 is a caller's arithmetic slip, not a geometry this kernel has
 * to be wider than. Both are 32-bit: they index rows of a plane whose element offsets the launcher
 * has already checked fit in 31 bits. */
struct RowMap { int s, tile, start, i0, len, span; };

__device__ __forceinline__ RowMap map_seq(const Args& A, const int tpb, int g)
{
    const int n = A.n_seq;
    for (int s = 0; s < n; ++s) {
        const int c0 = A.cu ? (int)ld_ix(A.cu, (size_t)s) : s * A.q_len;
        const int c1 = A.cu ? (int)ld_ix(A.cu, (size_t)(s + 1)) : c0 + A.q_len;
        const int start = (s == 0 && c0 > 0) ? 0 : c0;
        const int nr = c1 - start;
        const int nt = nr > 0 ? (nr + tpb - 1) / tpb : 0;      /* tpb is a template constant */
        /* TWO counts that are easy to conflate. `len` is the sequence's query length,
         * cu[s+1] - cu[s], and it is what the positions count from (the reference: qlen, and
         * pos = ctx - qlen + i). `span` is the number of rows this sequence's TILES carry,
         * cu[s+1] - start, and the two differ by exactly cu[0] on sequence 0 of a descriptor that
         * does not start at row 0: `start` reaches below cu[0] because seq_of hands those rows to
         * sequence 0 with a negative index, so the tiles must cover `span` rows, and only `len` of
         * them are inside the sequence. The store below is checked against the span and the
         * positions against the length: bounding the tile slots by `len` drops the last cu[0]
         * rows of the QUERY PLANE -- a silent, partial, in-bounds shortfall. */
        if (g < nt) return RowMap{ s, g, start, c0, c1 - c0, nr };
        if (s + 1 >= n) return RowMap{ s, -1, start, c0, c1 - c0, nr };  /* past the batch's tiles */
        g -= nt;
    }
    return RowMap{ n - 1, -1, 0, 0, 0, 0 };
}

/* The same walk for a row, not a tile: which sequence owns dense query row `r`. The reducer needs
 * it because a partial is indexed by (row, head) and its slice count is the SEQUENCE's block count. */
__device__ __forceinline__ int seq_of_row(const Args& A, int r)
{
    const int n = A.n_seq;
    for (int s = 0; s < n; ++s) {
        const int c1 = A.cu ? (int)ld_ix(A.cu, (size_t)(s + 1)) : (s + 1) * A.q_len;
        if (r < c1 || s + 1 >= n) return s;
    }
    return n - 1;
}

/* ------------------------------------------------------------------ staging */

/* Sixteen E4M3 codes -- one 16-byte load, the aligned unit a slot's 128-byte half divides into -- as
 * eight f16 pairs, descaled on the way. A code is exact in f16, so a power-of-two descale (every real
 * cache scale) costs a multiply and no rounding at all; a general descale is rounded once, toward
 * zero by the packing instruction, where the reference keeps it in f32. The bf16 path has no descale,
 * so it has no rounding to report. */
__device__ __forceinline__ void codes16_to_f16x8(const uint8_t* p, float s, uint32_t* w)
{
    const r3d_u32x4 c = *(const __attribute__((address_space(1))) r3d_u32x4*)(const void*)p;
    const uint32_t d[4] = { c.x, c.y, c.z, c.w };
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float a0 = e4m3_to_f32((uint8_t)(d[j] & 0xffu)) * s;
        const float a1 = e4m3_to_f32((uint8_t)((d[j] >> 8) & 0xffu)) * s;
        const float a2 = e4m3_to_f32((uint8_t)((d[j] >> 16) & 0xffu)) * s;
        const float a3 = e4m3_to_f32((uint8_t)(d[j] >> 24)) * s;
        w[j * 2] = f16x2_pack(a0, a1);
        w[j * 2 + 1] = f16x2_pack(a2, a3);
    }
}

/* Stage one block of one KV head: 16 slots x 128 elements, K and V, widened to f16 pairs.
 *
 * A slot's halves are 512 bytes apart and each 256 of them, so a bf16 chunk of 8 elements is 16
 * bytes and lands on a 16-byte boundary; an E4M3 chunk of 16 codes is 16 bytes and lands there too.
 * A chunk of 8 codes is 8 bytes and does not -- a misaligned 16-byte load on this hardware returns
 * garbage, which is the whole difference between the two chunk sizes below. 128 threads, one chunk of
 * K and one of V each for the cache, two each for the bf16: 256 chunks of 16 bytes a half, 8 KiB of
 * tile, two loads and two stores a thread.
 *
 * Every offset is 32-bit and the launcher has checked that the whole cache fits in 2^31 elements:
 * the wavefront runs in compute-shader mode, where a scalar spill is not a slow path but a kernel the
 * runtime refuses to load, and 64-bit index arithmetic across a loop like this one is what fills the
 * scalar bank. */
template <bool FP8, int NTH>
__device__ __forceinline__ void stage(Args const& A, uint32_t* sK, uint32_t* sV, int blk, int kh,
                                      float kds, float vds, int tid)
{
    constexpr int kC = FP8 ? 16 : 8;                            /* elements a 16-byte chunk */
    constexpr int kCPH = kHeadDim / kC;                         /* chunks a half, a slot */
    constexpr int kChunks = kBlock * kCPH;                      /* 128 code chunks, 256 bf16 */
    constexpr int kIter = kChunks / NTH;   /* bf16: 8, 4, 2 at a wave, two, four */
    static_assert(kIter * NTH == kChunks, "the stage must divide into whole per-thread trips: "
                                               "a bounds check inside the unrolled loop keeps the "
                                               "loop from unrolling, and the loop-invariant operands "
                                               "then get hoisted into scalar registers");
    const int base = blk * A.sb + kh * A.sh;                    /* this head's block, elements */

#pragma unroll
    for (int u = 0; u < kIter; ++u) {
        const int t = tid + u * NTH;
        const int slot = t / kCPH;
        const int c = (t % kCPH) * kC;                          /* element offset in the half */
        const int k0 = base + slot * A.so + c;
        const int v0 = k0 + kHeadDim;
        uint32_t* dk = sK + slot * kPitch + c / 2;
        uint32_t* dv = sV + slot * kPitch + c / 2;
        if constexpr (FP8) {                                    /* codes: element and byte agree */
            codes16_to_f16x8(A.kv8 + k0, kds, dk);
            codes16_to_f16x8(A.kv8 + v0, vds, dv);
        } else {
            const r3d_u32x4 ka =
                *(const __attribute__((address_space(1))) r3d_u32x4*)(const void*)(A.kv + k0);
            const r3d_u32x4 va =
                *(const __attribute__((address_space(1))) r3d_u32x4*)(const void*)(A.kv + v0);
            const uint32_t kd[4] = { ka.x, ka.y, ka.z, ka.w };
            const uint32_t vd[4] = { va.x, va.y, va.z, va.w };
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                dk[j] = bf16x2_to_f16x2(kd[j]);
                dv[j] = bf16x2_to_f16x2(vd[j]);
            }
        }
    }
}

/* ------------------------------------------------------------------ the walks */

/* The keys a lane may see: j is a candidate for the wave, the predicate is per lane. */
__device__ __forceinline__ bool visible(int j, long long pos, int causal, int window)
{
    if (causal && (long long)j > pos) return false;
    if (window > 0) {
        const long long d = pos - j;
        if (causal ? (d >= window) : (d >= window || d <= -window)) return false;
    }
    return true;
}

__device__ __forceinline__ float score_of(const uint32_t* qf, const uint32_t* kp, float scale)
{
    float dot = 0.0f;
#pragma unroll
    for (int p = 0; p < kPairs; ++p) dot = dot2_f16x2(qf[p], kp[p], dot);
    return dot * scale;
}

/* ------------------------------------------------------------------ the launch's own voice
 *
 * Two content folds and one line, printed from ONE thread of ONE workgroup of a launch.
 *
 * `fold64` is FNV-1a over 64-bit words -- multiply, xor, per word; no float arithmetic, no
 * reduction that a compiler could contract. The spelling is exact and the board's host-side fold
 * copies it byte for byte, because a hash the driver cannot recompute is a number nobody can check.
 * It is order-sensitive by construction (a permuted row is a different hash), which is the property
 * that makes the layer-major addressing slip visible: the same 128 elements read at a layer-major
 * offset fold to something else.
 *
 * The words are the row's OWN bytes, read as the walk reads them: the query row at
 * `q + row * q_rs` for head 0 -- head 0 is the first head of kv-head 0, and 128 bf16 elements are
 * the 32 words -- and the K row of the block the walk names FIRST, at `blk * sb + kh * sh +
 * slot * so`, the same arithmetic `stage()` uses (this file has no `slot_of` helper; `slot_k_off`
 * spells it out here so the print prints it). 128 elements are 16 words of E4M3 codes and 32 of
 * bf16. Both rows are 16-byte aligned: the shim checks a 16-aligned cache, query and output base
 * and a row pitch that is a multiple of 8 elements, and the header's layout note puts a slot's K on
 * a 16-byte boundary at every slot.
 */
__device__ __forceinline__ unsigned long long fold64(const void* p, int words)
{
    const unsigned long long* w = (const unsigned long long*)p;
    unsigned long long h = 14695981039346656037ULL;                 /* FNV-1a 64 offset basis */
    for (int i = 0; i < words; ++i) h = (h * 1099511628211ULL) ^ w[i];
    return h;
}

__device__ __forceinline__ long long slot_k_off(const Args& A, int blk, int kh, int slot)
{
    return (long long)blk * A.sb + (long long)kh * A.sh + (long long)slot * A.so;
}

/* One record per launch, from block (0,0,0), thread 0, into the ring. `noinline` on purpose: the
 * walk's register budget is the scarcest thing in this file (see the stride note on Args -- a scalar
 * spill is a kernel the runtime refuses to load), and a writer that a traced run runs once a launch
 * must not be inlined into the middle of the mapping code that every launch runs.
 *
 * This function reads memory, so its bounds are the subject. THREE reads, three bounds:
 *
 *   the query row      bound: the PLANE's row count `A.rows`, the same bound the kernel checks
 *                      before it stores (`M.start + tok < A.rows`, in the caller). `row` is the
 *                      caller's clamped row index: the grid is sized on a bound, so a tile can name
 *                      rows the plane does not have, and `row_ok` says whether this one is real. A
 *                      band key (`q_len`) and the sequence's length are bounds on the SHAPE the shim
 *                      accepted, not on the plane, and the clamp exists because the grid exceeds
 *                      them; reading through a band key is the out-of-bounds this guard kills. A tile
 *                      the batch has no rows for records `qh=0, rowok=0` -- the record says it did
 *                      not read, and `L` still counts the launch.
 *   the table's row    bound: the columns the WALK can name, ceil(jcap / block_size), where
 *                      jcap = min(ctx, bt_cols * block_size), so a column under it is a column the
 *                      shim proved when it sized the table from the tensor (row pitch plus bt_cols
 *                      columns fit, at the table's own pitches), and the row is `M.s * bt_rs` with
 *                      `M.s` a sequence index the mapping clamped.
 *   the cache          bound: a block id is one the table names, and the shim insists every index
 *                      the walk can reach is inside the cache's plane (`0 <= id < num_blocks`, plus
 *                      the dense-stride requirement). The fold reads the slot-0 K row of kv-head 0 at
 *                      the offset `stage()` computes, kHeadDim elements: 256 bytes of bf16, 128 of
 *                      fp8, both inside a slot's 2*kHeadDim. A negative id -- an unmapped slot prints
 *                      -1 -- folds to 0 and is not dereferenced. */
template <bool FP8>
__device__ __noinline__ void dev_record(const Args& A, const RowMap& M, long long row, bool row_ok,
                                        long long ctx, int jend_all, long long pos)
{
    /* A launch that runs is a launch that is counted, budget or not: this is the number that stops
     * the decode steps of a request being inferred from shim lines. */
    const unsigned long long n = atomicAdd(&r3d_dev_launches, 1ULL) + 1ULL;
    /* The bound, decremented atomically, so 16 CUs cannot record 16 copies of one launch. A caller
     * that armed 3 gets 3 records, wherever in the run they fall. The buffer channel bypasses the
     * bound: it is one slot per launch, drained at the launch, and the budget is a device global
     * that the engine process cannot arm. */
    if (!A.trace_buf && atomicSub(&r3d_dev_budget, 1) <= 0) return;

    DevRec r;
    r.L = n;
    r.dec = A.rows == 1 ? 1 : 0;
    r.n_seq = A.n_seq; r.rows = A.rows; r.q_len = A.q_len;
    r.s = M.s; r.len = M.len; r.span = M.span; r.start = M.start;
    r.ctx = ctx; r.pos = pos; r.jcap = jend_all; r.splits = A.splits; r.waves = A.waves;
    r.kv = (unsigned long long)(uintptr_t)(FP8 ? (const void*)A.kv8 : (const void*)A.kv);
    r.bt = (unsigned long long)(uintptr_t)A.bt;
    r.su = (unsigned long long)(uintptr_t)A.su;
    r.cu = (unsigned long long)(uintptr_t)A.cu;
    r.q = (unsigned long long)(uintptr_t)A.q;
    r.out = (unsigned long long)(uintptr_t)A.out;

    const int cols = (A.jcap + kBlock - 1) / kBlock;      /* <= bt_cols, by jcap's bound */
    const long long r0 = (long long)M.s * A.bt_rs;
    r.ncols = cols;
    unsigned long long is = 14695981039346656037ULL;      /* the fold64 basis, so a sum of ids is a
                                                           * fold of them, spelled the same way */
    for (int c = 0; c < cols; ++c) {
        const int b = (int)ld_ix(A.bt, (size_t)r0 + (size_t)c * (size_t)A.bt_cs);
        is = (is * 1099511628211ULL) ^ (unsigned long long)(unsigned int)b;
        if (c < kDevIds) r.b[c] = b;
    }
    for (int c = cols; c < kDevIds; ++c) r.b[c] = -2;     /* a column the row does not have. -2 is
                                                           * not -1, which is an unmapped block */
    r.idsum = (long long)is;

    /* The query row, under the plane's bound, named above. */
    const bool q_ok = row_ok && row >= 0 && row < A.rows;
    r.rowok = q_ok ? 1 : 0;
    r.qh = q_ok ? fold64(A.q + row * (long long)A.q_rs, kHeadDim / 4) : 0ULL;

    /* The head K row of the first block the row WALKS, and of the last block it walks. The walk's
     * extent is `jend_all = min(ctx, A.jcap)` keys, so the blocks it visits are `ceil(jend_all /
     * kBlock)` — NOT the table's `cols`: the table row is the plane's full capacity (64 columns at
     * a 4096 cap) and its far columns are `-1` for a short context, a fold of which is a fold of
     * nothing. `ncols` and `idsum` keep the whole table row — that is the census — while `kh`,
     * `khL`, `slot0`, `slotL` and `bL` are the walk's two ends: the head of an adopted prefix and
     * the tail the step is writing, where a recycled block shows up. */
    const int walk_cols = (jend_all + kBlock - 1) / kBlock;
    const int b0 = walk_cols > 0 ? r.b[0] : -2;
    const int blast = walk_cols > 0
          ? (int)ld_ix(A.bt, (size_t)r0 + (size_t)(walk_cols - 1) * (size_t)A.bt_cs)
          : -2;
    const long long k0 = b0 >= 0 ? slot_k_off(A, b0, 0, 0) : 0;
    const long long kL = blast >= 0 ? slot_k_off(A, blast, 0, 0) : 0;
    r.slot0 = k0; r.slotL = kL;
    r.kh  = b0 < 0 ? 0ULL : (FP8 ? fold64(A.kv8 + k0, kHeadDim / 8) : fold64(A.kv + k0, kHeadDim / 4));
    r.khL = (kL == 0) ? 0ULL
          : (FP8 ? fold64(A.kv8 + kL, kHeadDim / 8) : fold64(A.kv + kL, kHeadDim / 4));
    r.spare = blast;                                          /* the id the last fold is of, printed */
    r.spare2 = cols; r.spare3 = 0;

    /* One slot, claimed by one atomic. Past the ring's end a record is counted and dropped: the
     * counter keeps the truth, and the shim prints one line saying how many were lost. The struct is
     * filled in a local and copied in one statement so the writer is one block of stores; a drain
     * runs at a shim call, after the engine has synced the step, so it reads slots that are written
     * whole. */
    const unsigned w = atomicAdd(&r3d_dev_widx, 1u);
    if (w < (unsigned)kDevRing) r3d_dev_ring[w] = r;
    if (A.trace_buf) A.trace_buf[0] = r;   /* the engine-process channel: no symbol between us */
}

/* One wave's work: GQA heads x 32/GQA rows, over its slice of one sequence's blocks, for one KV
 * head. CAUSAL and SPLIT are template parameters because both change the loop, not just a flag. */
template <bool FP8, bool CAUSAL, bool SPLIT, int GQA, int WAVES>
__global__ __launch_bounds__(kThreads) R3D_WAVES(4)
void attn_kernel(const Args A)
{
    __shared__ uint32_t sK[kTileDw];
    __shared__ uint32_t sV[kTileDw];

    constexpr int TOKS = 32 / GQA;                       /* query rows a wave carries */
    constexpr int kNThr = WAVES * 32;                    /* 32, 64, 128 threads */
    constexpr int ROWSB = WAVES * TOKS;                  /* rows a workgroup carries */
    constexpr int LG = (GQA == 1) ? 0 : (GQA == 2) ? 1 : (GQA == 4) ? 2 : 3;

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5, lane = tid & 31;
    const RowMap M = map_seq(A, ROWSB, (int)blockIdx.x);
    /* A grid row the batch has no tiles for (the grid is sized on a bound), and a sequence with no
     * rows (cu[s+1] <= cu[s], the reference's `qlen <= 0`), stage nothing and store nothing. A
     * sequence that starts past the query plane -- a descriptor running beyond the operand -- is
     * nobody's rows: the plane's row count is the bound the store is checked against below. */
    if (M.tile < 0 || M.len <= 0 || M.start >= A.rows) return;
    const int s = M.s;
    const int tile = M.tile;
    const int kh = (int)blockIdx.y;
    const int slice = SPLIT ? (int)blockIdx.z : 0;

    const int h = kh * GQA + (lane & (GQA - 1));
    const int tb = tile * ROWSB + wave * TOKS;        /* the wave's first row of the sequence */
    const int tok = tb + (lane >> LG);                /* this lane's, counted inside the sequence */
    /* A row the operands do not have: past the rows this sequence's tiles carry (the last tile of
     * any sequence, and every tile of a short one in a ragged step), or past the query plane (a
     * descriptor that runs long). Both are the clamp the width section documents. The bound is the
     * SPAN, not the length: with cu[0] > 0 the span reaches below cu[0], and the rows there are
     * the reference's rows too (seq_of gives them to sequence 0), so they are stored. */
    const bool row_ok = tok < M.span && M.start + tok < A.rows;
    /* The clamp target is a row that EXISTS: the wave's first row if the sequence has one, else its
     * last, and never past the plane. An idle lane loads a real query row so the walk stays
     * branch-free; it never stores. */
    int tclamp = tb < M.span ? tb : M.span - 1;
    if (M.start + tclamp >= A.rows) tclamp = A.rows - 1 - M.start;
    const long long row = (long long)M.start + (row_ok ? tok : tclamp);

    const long long ctx = ld_ix(A.su, (size_t)s);
    const int jend_all = (int)(ctx < A.jcap ? ctx : A.jcap);
    /* `ib` is 0 for every descriptor the description allows (cu[0] == 0); it is the negative index
     * into sequence 0's query that libref's seq_of gives rows under cu[0]. */
    const long long ib = (long long)M.start - M.i0;
    const long long pos = ctx - M.len + tok + ib;

    /* THE LAUNCH WRITES. The shim is not in the path of a played step -- a play writes queue
     * packets and never calls `rad_arch_step` (the recorded pass's packet path) -- and this kernel
     * is, so the row writes down what the row was
     * handed: `ctx` off `seqused`, the block ids off the table at the table's own pitches, the six
     * operand addresses baked into the packet, and three content folds. ONE thread of ONE workgroup
     * (`blockIdx.z` is the split-KV slice: one launch, not `splits` of them), gated on a global the
     * shim writes only under `RAD_R3D_TRACE`. With the flag unset the gate is one load of a zero
     * global and a not-taken branch, and the launch records nothing. It does not printf: a kernel
     * that does takes the implicit `hidden_hostcall_buffer`, which the AQL backend does not supply,
     * and the row stops being dispatchable at all (see the ring's header).
     *
     * Placed HERE, before the query is widened and long before the walk, so the record covers the
     * launch even when the walk's rows are all clamped -- which is why the writer's own read of the
     * query row is bounded by the PLANE's rows and not by the band key, and why a clamped tile
     * records `qh=0 rowok=0` -- and so the writer's registers are live over the fewest live values
     * in the kernel. */
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        dev_record<FP8>(A, M, row, row_ok, ctx, jend_all, pos);

    /* The WORKGROUP's bounds, not the wave's. The stage is one tile shared by the four waves, so
     * every wave of a workgroup must walk exactly the same blocks: a range narrowed to a wave's own
     * four positions makes the waves stage different blocks over the same LDS, and the last writer
     * per 16-byte chunk wins -- a corrupted tile that looks like a numerical error. The positions of
     * a workgroup's 4 x TOKS consecutive rows span 4 x TOKS - 1, and keys past the last are nobody's;
     * a window cuts the front off as well. The mask that this widens past is per lane and free. */
    const int gb = tile * ROWSB;
    const long long wmin = ctx - M.len + gb + ib;
    const long long wmax = wmin + ROWSB - 1;
    int jlo = 0, jhi = jend_all;
    if (CAUSAL) {
        if (wmax + 1 < jhi) jhi = (int)(wmax + 1);
    }
    if (A.window > 0) {
        const long long lo = wmin - A.window + 1;
        if (lo > jlo) jlo = (int)(lo > 0 ? lo : 0);
        if (!CAUSAL) {
            const long long hi = wmax + A.window;
            if (hi < jhi) jhi = (int)(hi > 0 ? hi : 0);
        }
    }
    const int nblk = (jend_all + kBlock - 1) / kBlock;
    int b0 = jlo / kBlock, b1 = (jhi + kBlock - 1) / kBlock;
    if (SPLIT) {
        const int tps = (nblk + A.splits - 1) / A.splits;
        const int s0 = slice * tps;
        const int s1 = s0 + tps < nblk ? s0 + tps : nblk;
        if (b0 < s0) b0 = s0;
        if (b1 > s1) b1 = s1;
    }

    const float kds = A.kds ? A.kds[(size_t)s * A.kv_heads + kh] : 1.0f;
    const float vds = A.vds ? A.vds[(size_t)s * A.kv_heads + kh] : 1.0f;

    /* The query row, widened once and kept for both walks: 64 dwords, the lane's own 256 bytes, the
     * eight heads of a token contiguous in memory. An idle lane loads its wave's first row so the
     * registers hold something and the walk stays branch-free; it never stores. */
    uint32_t qf[kPairs];
    {
        const uint16_t* qp = A.q + row * (long long)A.q_rs + h * kHeadDim;
#pragma unroll
        for (int u = 0; u < kPairs / 4; ++u) {                 /* eight elements a 16-byte load */
            const r3d_u32x4 raw =
                *(const __attribute__((address_space(1))) r3d_u32x4*)(const void*)(qp + u * 8);
            const uint32_t d[4] = { raw.x, raw.y, raw.z, raw.w };
#pragma unroll
            for (int j = 0; j < 4; ++j) qf[u * 4 + j] = A.q16 ? d[j] : bf16x2_to_f16x2(d[j]);
        }
    }

    float acc[kHeadDim];
#pragma unroll
    for (int d = 0; d < kHeadDim; ++d) acc[d] = 0.0f;
    float sum = 0.0f, mx = -INFINITY;

    /* Walk one: the scores' maximum. Blocks entirely behind every row of the workgroup need no
     * mask, and the ones straddling a row's position mask per lane. */
    for (int b = b0; b < b1; ++b) {
        const int blk = (int)ld_ix(A.bt, (size_t)s * (size_t)A.bt_rs + (size_t)b * (size_t)A.bt_cs);
        group_barrier();
        if (blk >= 0) stage<FP8, kNThr>(A, sK, sV, blk, kh, kds, vds, tid);
        group_barrier();
        if (blk < 0) continue;
        const int j0 = b * kBlock;
        const int nu = j0 + kBlock <= jhi ? kBlock : (jhi > j0 ? jhi - j0 : 0);
        const bool nomask = !CAUSAL && A.window <= 0;
        if (nomask) {
            for (int u = 0; u < nu; ++u) {
                const float sc = score_of(qf, sK + u * kPitch, A.scale);
                if (sc > mx) mx = sc;
            }
        } else {
            for (int u = 0; u < nu; ++u) {
                if (!visible(j0 + u, pos, CAUSAL, A.window)) continue;
                const float sc = score_of(qf, sK + u * kPitch, A.scale);
                if (sc > mx) mx = sc;
            }
        }
    }

    /* Walk two: the same scores, exponentiated against the maximum, accumulated ascending. */
    if (mx > -INFINITY) {
        for (int b = b0; b < b1; ++b) {
            const int blk = (int)ld_ix(A.bt, (size_t)s * (size_t)A.bt_rs + (size_t)b * (size_t)A.bt_cs);
            group_barrier();
            if (blk >= 0) stage<FP8, kNThr>(A, sK, sV, blk, kh, kds, vds, tid);
            group_barrier();
            if (blk < 0) continue;
            const int j0 = b * kBlock;
            const int nu = j0 + kBlock <= jhi ? kBlock : (jhi > j0 ? jhi - j0 : 0);
            const bool nomask = !CAUSAL && A.window <= 0;
            for (int u = 0; u < nu; ++u) {
                if (!nomask && !visible(j0 + u, pos, CAUSAL, A.window)) continue;
                const float w = exp_ref(score_of(qf, sK + u * kPitch, A.scale) - mx);
                sum += w;
                const uint32_t* vp = sV + u * kPitch;
#pragma unroll
                for (int p = 0; p < kPairs; ++p) {
                    const uint32_t q2 = vp[p];
                    const int d = 2 * p;
                    acc[d] = acc[d] + w * f16_to_f32((uint16_t)(q2 & 0xffffu));
                    acc[d + 1] = acc[d + 1] + w * f16_to_f32((uint16_t)(q2 >> 16));
                }
            }
        }
    }

    /* A row slot past its sequence's length, or past the query plane, is a clamped duplicate: it
     * computed a walk, and it stores nothing -- the partial store included, which needs the same
     * test BEFORE it. A padding row's `row` clamps to a REAL row, so an unguarded partial store
     * puts a second writer into (row, h, slice) -- inside the declared arena, so no bounds check
     * sees it -- and the two writers' 132-float records interleave: a slot can end with one lane's
     * acc and another lane's (sum, max). The reduce then rescales a full slice's accumulator by
     * the other lane's maximum, the combined sum comes out thousands of times too small, and
     * `1/sum` hands back a power of two where a logit belongs. The mechanism needs a padding row
     * -- q_len < 32/gqa -- so a band that fills its waves (q_len 7 at gqa 8) has none. The slots a
     * padding row leaves unwritten are exactly the slots the reduce does not read: its index runs
     * to `nrows * n_head`, over real rows only. */
    if (!row_ok) return;

    if (SPLIT) {
        float* pp = A.part + ((size_t)row * A.n_head + h) * ((size_t)A.splits * kPartStride)
                    + (size_t)slice * kPartStride;
#pragma unroll
        for (int u = 0; u < kHeadDim / 8; ++u) {
            const r3d_f32x4 v = { acc[8 * u], acc[8 * u + 1], acc[8 * u + 2], acc[8 * u + 3] };
            const r3d_f32x4 w = { acc[8 * u + 4], acc[8 * u + 5], acc[8 * u + 6], acc[8 * u + 7] };
            *(r3d_f32x4*)(pp + 8 * u) = v;
            *(r3d_f32x4*)(pp + 8 * u + 4) = w;
        }
        pp[kHeadDim] = sum;
        pp[kHeadDim + 1] = mx;
        return;
    }

    uint16_t* op = A.out + row * (long long)A.o_rs + h * kHeadDim;
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;           /* no visible key: a zero row */
#pragma unroll
    for (int u = 0; u < kHeadDim / 8; ++u) {
        uint32_t q4[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = 8 * u + 2 * j;
            q4[j] = bf16x2_rne(acc[d] * inv, acc[d + 1] * inv);
        }
        *(r3d_u32x4*)(void*)(op + 8 * u) = r3d_u32x4{ q4[0], q4[1], q4[2], q4[3] };
    }
}

/* ------------------------------------------------------------------ split-KV reduce
 *
 * A wave per (query row, head): 32 lanes x 4 dims. The slices combine ascending, each rescaled by
 * exp(m_slice - M) with M the slices' maximum, which is where split-KV leaves the reference's
 * parentheses; see the header. */
__global__ __launch_bounds__(128) R3D_WAVES(4)
void attn_reduce(const Args A, long long nrows)
{
    const int w = (int)threadIdx.x >> 5, lane = (int)threadIdx.x & 31;
    const long long idx = (long long)blockIdx.x * 4 + w;
    const long long total = nrows * A.n_head;
    if (idx >= total) return;
    const long long row = idx / A.n_head, h = idx - row * A.n_head;
    const int s = seq_of_row(A, (int)row);            /* row / q_len with a null descriptor */
    const long long ctx = ld_ix(A.su, (size_t)s);
    const int jend_all = (int)(ctx < A.jcap ? ctx : A.jcap);
    const int nblk = (jend_all + kBlock - 1) / kBlock;
    const int tps = (nblk + A.splits - 1) / A.splits;
    const int nslice = tps > 0 ? (nblk + tps - 1) / tps : 0;

    float* pp = A.part + (size_t)idx * ((size_t)A.splits * kPartStride);
    float M = -INFINITY;
    for (int u = 0; u < nslice; ++u) {
        const float m = pp[(size_t)u * kPartStride + kHeadDim + 1];
        if (m > M) M = m;
    }
    uint16_t* o = A.out + row * (long long)A.o_rs + h * kHeadDim;
    const int d0 = lane * 4;
    float a[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float sum = 0.0f;
    if (M > -INFINITY) {
        for (int sl = 0; sl < nslice; ++sl) {
            const float* q = pp + (size_t)sl * kPartStride;
            const float f = exp_ref(q[kHeadDim + 1] - M);
            sum += q[kHeadDim] * f;
            a[0] += q[d0 + 0] * f;
            a[1] += q[d0 + 1] * f;
            a[2] += q[d0 + 2] * f;
            a[3] += q[d0 + 3] * f;
        }
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    *(r3d_u32x2*)(void*)(o + d0) = r3d_u32x2{ bf16x2_rne(a[0] * inv, a[1] * inv),
                                              bf16x2_rne(a[2] * inv, a[3] * inv) };
}

}  // namespace

/* ================================================================== launchers
 *
 * Both entries take raw addresses and extents, in the library's style, and both assume the dense
 * layouts the shim has just checked: the cache at [blocks, kv_heads, 16, 256], a block table whose
 * row pitch is exactly ceil(max_ctx / block_size), seqused one i32 a sequence. That is what the
 * signatures carry, so the shims refuse anything they cannot describe with it -- a strided cache, a
 * wider block table, an i64 index plane.
 *
 * `cu` is the ragged query descriptor, and with it `q_len` changes meaning: it is the query PLANE's
 * row count, the total, which the descriptor's cu[num_seqs] states and libref reads off the tensor
 * (`rows = numel(q) / (q_heads * head_dim)`, `cu[n_seq] == rows`). Absent, it
 * keeps its per-sequence meaning and the total is num_seqs * q_len. No entry reads the plane: the
 * grid is sized on a bound, and the kernel does the walking.
 */
#define R3D_NEED(cond) do { if (!(cond)) return R3D_NO(RAD_E_SHAPE); } while (0)

namespace {

int fill(Args& A, long q, long kv, long bt, long seqused, long k_desc, long v_desc, long cu,
         long rows, int num_seqs, int q_len, int head_dim, int gqa, int n_head, int block_size,
         int bt_cols, int bt_rs, int bt_cs, int max_ctx, int window, int causal, float scale,
         int kv_dt, long q_ld, long out_ld, int splits, long scratch, int waves)
{
    A = Args{};
    if (head_dim != kHeadDim) return -2;                            /* RAD_E_SHAPE */
    if (num_seqs <= 0 || n_head <= 0) return -2;
    /* A rectangular batch is counted per sequence; a ragged one has no per-sequence length, and its
     * `rows` is the query plane's row count. Both have to be at least one row. */
    if (cu) { if (rows < 1 || rows > 2147483000LL) return -2; }
    else if (q_len < 1 || (long long)num_seqs * q_len > 2147483000LL) return -2;
    if (block_size != kBlock) return -2;
    if (gqa != 1 && gqa != 2 && gqa != 4 && gqa != 8) return -2;    /* the wave split is a shift */
    if (n_head % gqa) return -2;
    if (window < 0 || max_ctx < 1) return -2;
    if (kv_dt != RAD_BF16 && kv_dt != RAD_F8E4M3) return -1;        /* RAD_E_DTYPE */
    if (!q || !kv || !bt || !seqused) return -1;

    const int kv_heads = n_head / gqa;
    /* Two counts that must not be conflated. The CACHE holds
     * ceil(max_ctx / block_size) blocks, which bounds the byte offset of a slot address. The TABLE
     * has `bt_cols` columns, and the engine sizes it to the step: a five-token prompt with nothing
     * behind it is one block, a table of [n_seq, 1]. The walk is bounded by the TABLE --
     * the reference takes max_blocks from `bt->shape[last]`, the row pitch from
     * `stride[rank-2]` and the column stride from `stride[rank-1]`, and jend = min(ctx, max_blocks *
     * block_size) -- so a context longer than the table has columns is truncated by the reference,
     * which is the oracle, and by us, to the same keys. */
    const int cache_blocks = (max_ctx + block_size - 1) / block_size;
    if (bt_cols < 1 || bt_rs < 1 || bt_cs < 1) return -2;   /* RAD_E_STRIDE, said where the shim sees it */

    A.q = (const uint16_t*)(uintptr_t)q;
    A.kv = kv_dt == RAD_BF16 ? (const uint16_t*)(uintptr_t)kv : nullptr;
    A.kv8 = kv_dt == RAD_F8E4M3 ? (const uint8_t*)(uintptr_t)kv : nullptr;
    A.bt = (const int*)(uintptr_t)bt;
    A.su = (const int*)(uintptr_t)seqused;
    A.cu = (const int*)(uintptr_t)cu;
    A.kds = (const float*)(uintptr_t)k_desc;
    A.vds = (const float*)(uintptr_t)v_desc;
    /* The dense cache: a slot's K is 128 elements, V the 128 after it. Every stride has to fit the
     * 32-bit form the kernel's addressing uses. */
    A.so = 2 * kHeadDim;
    A.sh = block_size * A.so;
    A.sb = kv_heads * A.sh;
    A.sd = 1;
    if ((long long)A.sb * cache_blocks > 2147483000LL) return -2;
    if (q_ld > 2147483000LL || out_ld > 2147483000LL) return -2;
    A.q_rs = (int)q_ld;
    A.o_rs = (int)out_ld;
    A.bt_rs = bt_rs;
    A.bt_cs = bt_cs;
    A.n_seq = num_seqs; A.q_len = cu ? 0 : q_len;
    A.rows = cu ? (int)rows : num_seqs * q_len;
    A.n_head = n_head; A.kv_heads = kv_heads;
    A.causal = causal ? 1 : 0; A.window = window;
    A.jcap = max_ctx;
    if ((long long)bt_cols * block_size < max_ctx) A.jcap = bt_cols * block_size;
    if (r3d_trace_on())   /* the walk's reach is a number the shim can print: a context that
                                     * comes back short is a table narrower than the sequence, and
                                     * that is a plan fact, not a kernel fact */
        std::fprintf(stderr, "[r3d] attn: q_len %d rows %ld n_seq %d bs %d bt cols %d pitch %d "
                             "cstride %d max_ctx %d -> jcap %d\n", q_len, rows, num_seqs,
                     block_size, bt_cols, bt_rs, bt_cs, max_ctx, A.jcap);
    {   /* the record's slot, for the engine process: a plain buffer the kernel can write and the
         * shim can read back, no symbol table between them. Inert when the flag is unset. */
        static void* g_trace_buf = nullptr;
        A.trace_buf = nullptr;
        if (r3d_trace_on() && !g_trace_buf)
            (void)hipMalloc(&g_trace_buf, sizeof(DevRec));
        if (g_trace_buf) A.trace_buf = (DevRec*)g_trace_buf;
    }
    if (r3d_trace_on())   /* the device's own view, at the launch: the table row, the
                                     * seqused, the KV base, and the first block's bytes -- the
                                     * four things a wrong answer can come from, spoken together.
                                     * The sync is the price: the staging may ride the engine's
                                     * stream, and a read on the default stream is unordered to it. */
    {
        (void)hipDeviceSynchronize();
        const char* kvb = A.kv8 ? (const char*)A.kv8 : (const char*)A.kv;
        for (int s = 0; s < (int)num_seqs && s < 4; ++s) {
            int row[16];
            if (hipMemcpy(row, (const char*)A.bt + (size_t)s * (size_t)bt_rs * sizeof(int),
                          16 * sizeof(int), hipMemcpyDeviceToHost) == hipSuccess) {
                std::fprintf(stderr, "[r3d] devstate s=%d bt=", s);
                for (int c = 0; c < 16; ++c) std::fprintf(stderr, "%s%d", c ? " " : "", row[c]);
                std::fprintf(stderr, "\n");
            }
            int su = -1;
            if (A.su &&
                hipMemcpy(&su, (const char*)A.su + (size_t)s * sizeof(int), sizeof(int),
                          hipMemcpyDeviceToHost) == hipSuccess)
                std::fprintf(stderr, "[r3d] devstate s=%d seqused=%d kvbase=%p\n", s, su,
                             (const void*)kvb);
        }
        int first = -1;
        if (kvb &&
            hipMemcpy(&first, (const char*)A.bt, sizeof(int), hipMemcpyDeviceToHost) == hipSuccess &&
            first >= 0) {
            /* The block's bytes are spoken by address only: a D2H of the cache base page faults
             * the engine process where the kernel's own read is fine, so the content check stays
             * with the board, and the table + base are what the served run speaks. */
            std::fprintf(stderr, "[r3d] devstate first_block=%d (content not read: see the board)\n",
                         first);
        }
    }
    /* The table bounds the walk, as in the reference: a context with more blocks than the
     * table has columns stops at the table, which is what the reference does and therefore
     * what the fixture and the served numbers say. */
    A.scale = scale > 0.0f ? scale : 1.0f / std::sqrt((float)kHeadDim);
    A.splits = splits > 0 ? splits : 1;
    A.waves = waves;
    A.part = A.splits > 1 ? (float*)(uintptr_t)scratch : nullptr;
    if (A.splits > 1 && !scratch) return -5;                        /* RAD_E_SCRATCH */
    return 0;
}

/* The narrowest workgroup that carries the caller's q_len. A wave carries 32 / gqa query rows, and
 * a row slot whose row is past q_len computes a clamped duplicate: at the decode band's q_len = 7
 * with gqa = 8 a four-wave workgroup spends 9 of its 16 row slots on duplicates, twice the walk for
 * half the outputs. The tile is the same 8 KiB at every width -- 16 tokens a block, staged once per
 * workgroup -- so a narrower workgroup is the same cache read spread over more workgroups, and
 * 512 -> 1024 of them is the fill the counters say the op is short of (15% issue at the band,
 * 3.7% at eight sequences, where splits=8 buys 3.6x). Prefill keeps four waves: its q_len is the
 * whole chunk, every slot is used, and a smaller workgroup would only re-read the cache.
 * The accumulation order does not depend on the width: the softmax sum is over keys, ascending,
 * inside one lane, and the rows a wave holds are the same rows shifted by wave * TOKS. */
int pick_waves(int q_len, int gqa)
{
    const int rows = 32 / (gqa > 0 ? gqa : 1);
    const int w = (q_len + rows - 1) / rows;
    return w <= 1 ? 1 : (w <= 2 ? 2 : 4);      /* the ladder the grid is cut on: 1, 2, 4 */
}

/* The rows one sequence of a batch contributes, as well as a host can know them: the plane's row
 * count over the sequence count, rounded up. With a descriptor the true lengths are device memory
 * and a launcher reads none of it, so this is what the width and the split count are picked on for a
 * ragged step. Both choices move duplicated row slots and grid blocks; neither can move a result,
 * which is why the average is legal as a proxy and a per-sequence length is not needed. */
int avg_rows(int rows, int num_seqs) { return num_seqs > 0 ? (rows + num_seqs - 1) / num_seqs : rows; }

/* Every width the ladder can return, at every gqa, must be instantiated and dispatched: pick_waves
 * is `ceil(q_len / (32/gqa))` clamped to the ladder, so gqa = 4 asks for W = 4 at q_len > 16,
 * gqa = 2 at q_len > 32, gqa = 1 at q_len > 64 -- and the prefill entry passes kWaves = 4
 * unconditionally. The launcher dispatches through `if constexpr (waves_ok<...>::value)`, so a
 * missing instantiation compiles the kernel call out: the entry returns RAD_OK, hipGetLastError()
 * is clean, and the output plane keeps what the caller put there. A silent no-op is the worst
 * failure a kernel row can have: the engine reads its own input back as the attention output. The
 * gate therefore asserts the whole ladder, at every gqa -- every width it admits is a width some
 * q_len can ask for. */
template <int GQA, int W> struct waves_ok {
    static const bool value = (W == 1) || (W == 2) || (W == 4);
};

int tok_per_block(int gqa, int waves) { return waves * (32 / (gqa > 0 ? gqa : 1)); }

/* The decode entry's `splits`, when the caller asked for the widest legal: enough workgroups to
 * fill the card (16 CUs x two workgroups at four waves a SIMD) and no more, on a doubling ladder. */
int pick_splits(int tiles, int num_seqs, int kv_heads)
{
    const long long blocks = (long long)tiles * num_seqs * kv_heads;
    int sp = 1;
    while (sp < 8 && blocks * sp < 32) sp *= 2;
    return sp;
}

static void dev_speak(const DevRec& r);   /* the record, spoken; defined with the ring below */

int launch(const Args& A, long out, long stream)
{
    Args B = A;
    B.out = (uint16_t*)(uintptr_t)out;
    const int gqa = A.n_head / A.kv_heads;
    const int waves = A.waves > 0 ? A.waves : kWaves;
    const int tpb = tok_per_block(gqa, waves);
    /* The grid's row count. Rectangular: every sequence has the same tile count, so the product is
     * exact and is the grid. Ragged: the tile count is
     * sum_s ceil(len_s / tpb), which lives in device memory, so the launch uses the bound
     *     sum_s ceil(len_s / tpb)  <=  ceil(rows / tpb) + n_seq - 1
     * (ceil(x) + ceil(y) <= ceil(x + y) + 1, inducted over the sequences), and `map_seq` sends a grid
     * row past the last tile straight home. The slack is at most n_seq - 1 empty workgroups, and an
     * empty one stages nothing: it reads n_seq descriptor entries and returns. */
    const int tiles = A.cu ? (A.rows + tpb - 1) / tpb + A.n_seq - 1
                           : (A.q_len + tpb - 1) / tpb * A.n_seq;
    dim3 grid((unsigned)tiles, (unsigned)A.kv_heads,
              (unsigned)(A.splits > 1 ? A.splits : 1));
    const bool fp8 = A.kv8 != nullptr;
    const bool causal = A.causal != 0;
    hipStream_t st = (hipStream_t)(uintptr_t)stream;

#define R3D_ATTN(FP8V, CAUSALV, SPLITV, GQAV, WAVESV)                          \
    R3D_KLAUNCH((attn_kernel<FP8V, CAUSALV, SPLITV, GQAV, WAVESV>), grid,       \
                dim3(WAVESV * 32), st, B)

#define R3D_ATTN_SEL(GQAV, WAVESV)                                             \
    do {                                                                       \
        if constexpr (waves_ok<GQAV, WAVESV>::value) {                          \
            /* `sel` is three bits, and all eight combinations are reachable: a single-sequence     \
             * causal decode picks splits 4 on its own (pick_splits(1, 1, 8) = 4), so sel 6 and 7   \
             * are the engine's most common split shape. Each combination needs its own arm: a      \
             * causal-with-splits launch routed into the non-causal instantiation takes its maximum \
             * over keys the row must not see, the softmax loses its top weight to a future key,   \
             * and the reduce amplifies the loss into a 1/sum of the wrong size. All eight arms    \
             * exist. */                                                                        \
            if (sel == 0) R3D_ATTN(false, false, false, GQAV, WAVESV);         \
            else if (sel == 1) R3D_ATTN(true, false, false, GQAV, WAVESV);     \
            else if (sel == 2) R3D_ATTN(false, true, false, GQAV, WAVESV);     \
            else if (sel == 3) R3D_ATTN(true, true, false, GQAV, WAVESV);      \
            else if (sel == 4) R3D_ATTN(false, false, true, GQAV, WAVESV);     \
            else if (sel == 5) R3D_ATTN(true, false, true, GQAV, WAVESV);      \
            else if (sel == 6) R3D_ATTN(false, true, true, GQAV, WAVESV);      \
            else R3D_ATTN(true, true, true, GQAV, WAVESV);                     \
        }                                                                      \
    } while (0)

#define R3D_ATTN_GQA(GQAV)                                                     \
    do {                                                                       \
        if (waves == 1) R3D_ATTN_SEL(GQAV, 1);                                 \
        else if (waves == 2) R3D_ATTN_SEL(GQAV, 2);                            \
        else R3D_ATTN_SEL(GQAV, 4);                                            \
    } while (0)

    const int sel = (fp8 ? 1 : 0) | (causal ? 2 : 0) | (A.splits > 1 ? 4 : 0);
    /* The flag is cleared HERE, after the arguments are staged and before the chevron, and read
     * straight back after it. `hipGetLastError()` reports the first error raised since it was last
     * cleared, so a check that does not clear first attributes to this launch whatever the process
     * raised before it: in a served process, the op that ran last. A synchronous launch clears a
     * pending asynchronous error before the check, so `HIP_LAUNCH_BLOCKING=1` hides a
     * misattribution of exactly this kind. A healthy step refused by a symptom it did not cause is
     * the failure mode; the launch is not. */
    (void)hipGetLastError();
    switch (gqa) {
    case 1: R3D_ATTN_GQA(1); break;
    case 2: R3D_ATTN_GQA(2); break;
    case 4: R3D_ATTN_GQA(4); break;
    default: R3D_ATTN_GQA(8); break;
    }
#undef R3D_ATTN_GQA
#undef R3D_ATTN_SEL
#undef R3D_ATTN
    {
        const hipError_t ke = hipGetLastError();
        if (ke != hipSuccess) {
            /* A launch failure names what it saw: the error, the grid, the workgroup and the width
             * the launcher chose. */
            if (r3d_trace_on())
                std::fprintf(stderr, "[r3d] attn: %s on the attn launch: grid (%u, %u, %u) block "
                                     "%d waves %d gqa %d splits %d rows %d n_seq %d jcap %d\n",
                             hipGetErrorName(ke), grid.x, grid.y, grid.z, waves * 32, waves, gqa,
                             A.splits, A.rows, A.n_seq, A.jcap);
            return -4;                                          /* RAD_E_DEVICE */
        }
    }

    if (B.trace_buf) {   /* the record, spoken at the launch that wrote it: one sync, under the flag */
        (void)hipStreamSynchronize(st);
        DevRec rec;
        if (hipMemcpy(&rec, (const void*)B.trace_buf, sizeof rec, hipMemcpyDeviceToHost) == hipSuccess)
            dev_speak(rec);
    }

    if (A.splits > 1) {
        const long long rows = A.rows;              /* the plane's rows: cu[n_seq], or n_seq * q_len */
        const long long total = rows * A.n_head;
        dim3 rg((unsigned)((total + 3) / 4));
        (void)hipGetLastError();                                /* the same rule, the same reason */
        R3D_KLAUNCH(attn_reduce, rg, dim3(128), st, B, rows);
        const hipError_t re = hipGetLastError();
        if (re != hipSuccess) {
            if (r3d_trace_on())
                std::fprintf(stderr, "[r3d] attn: %s on the reduce launch: grid (%u, 1, 1) block "
                                     "128 rows %lld heads %d splits %d\n", hipGetErrorName(re),
                             rg.x, rows, A.n_head, A.splits);
            return -4;
        }
    }
    return 0;
}

}  // namespace

extern "C" int r3d_attn_decode_h128(long q, long kv, long bt, long seqused, long k_desc,
                                    long v_desc, long cu, long out, int num_seqs, int q_len,
                                    int head_dim, int gqa, int n_head, int block_size, int bt_cols, int bt_rs, int bt_cs, int max_ctx,
                                    int window, int causal, float scale, int kv_dt, long q_ld,
                                    long out_ld, int splits, long scratch, long stream)
{
    /* `cu` is the ragged query descriptor and it is served: with it, `q_len` is the query plane's
     * row count (see the launchers' header) and the rows per sequence are cu[s+1] - cu[s]. The
     * width and the split count are picked on the average sequence, because the descriptor is on the
     * device and this function reads no device memory -- both numbers choose how many row slots are
     * duplicated, and neither touches a result. */
    if (splits < 0) return -2;
    if (gqa <= 0 || n_head <= 0 || n_head % gqa) return -2;
    if (cu && num_seqs <= 0) return -2;
    Args A;
    const int kv_heads = n_head / gqa;
    const int per = cu ? avg_rows(q_len, num_seqs) : (q_len > 0 ? q_len : 1);
    const int waves = pick_waves(per, gqa);
    const int tpb = tok_per_block(gqa, waves);
    const int tiles = per > 0 ? (per + tpb - 1) / tpb : 1;
    int sp = splits > 0 ? splits : pick_splits(tiles, num_seqs, kv_heads);
    const int e = fill(A, q, kv, bt, seqused, k_desc, v_desc, cu, cu ? q_len : 0, num_seqs,
                       cu ? 0 : q_len, head_dim, gqa, n_head, block_size, bt_cols, bt_rs, bt_cs,
                       max_ctx, window, causal, scale, kv_dt, q_ld, out_ld, sp, scratch, waves);
    if (e) return e;
    return launch(A, out, stream);
}

extern "C" int r3d_attn_prefill_h128(long q, long kv, long bt, long seqused, long k_desc,
                                     long v_desc, long cu, long out, int num_seqs, int q_len,
                                     int head_dim, int gqa, int n_head, int block_size, int bt_cols, int bt_rs, int bt_cs,
                                     int max_ctx, int window, int causal, float scale, int kv_dt,
                                     long q_ld, long out_ld, long scratch, long stream)
{
    (void)scratch;                       /* the prefill kernel has no split pass, so no partials */
    /* The width is kWaves with a descriptor or without one. A rectangular prefill's q_len is the
     * whole chunk, so every slot of a four-wave workgroup is used; a ragged step's rows are laid
     * out by the same workgroup, and the width that matters on a step with only a few rows in it is
     * how many waves the GPU gets, not how many slots go unused. A 5-row step at gqa 8 is one
     * workgroup per KV head: two waves is 2 x 2 wavefronts over 16 CUs, four waves is 2 x 4, and
     * the measured 5-row launch at 2048 keys is 16.7 ms at four waves against 19.3 at two, with the
     * duplicate slots costing nothing because a wave walks its tile once whatever rows it carries.
     * A step whose rows are one per sequence is a decode step and goes to the decode entry, which
     * picks its width on the average sequence -- that is where a narrow workgroup is the win. */
    Args A;
    const int waves = kWaves;
    const int e = fill(A, q, kv, bt, seqused, k_desc, v_desc, cu, cu ? q_len : 0, num_seqs,
                       cu ? 0 : q_len, head_dim, gqa, n_head, block_size, bt_cols, bt_rs, bt_cs, max_ctx, window, causal,
                       scale, kv_dt, q_ld, out_ld, 1, 0, waves);
    if (e) return e;
    A.splits = 1;
    A.part = nullptr;
    return launch(A, out, stream);
}

/* ================================================================== shims
 *
 * The paged geometry as the tensors state it: n_seq from `seqused`, n_head and head_dim from the
 * query, kv_heads from the cache, block_size from the cache, max_ctx from the cache's block count,
 * q_len from the query's row count over n_seq. Params fill in what the tensors cannot say (gqa,
 * causal, window, scale). */
namespace {

enum { P_Q = 0, P_KV = 1, P_BT = 2, P_SU = 3, P_KDS = 4, P_VDS = 5, P_CU = 6, P_OUT = 7 };

/* ---- arming the device record ring ----------------------------------------------------------
 *
 * The kernel's four globals (`r3d_dev_budget`, `r3d_dev_launches`, `r3d_dev_widx`, `r3d_dev_ring`)
 * are plugin-owned, at file scope, `extern "C"` so the names are the ones the code object carries.
 * `RadArgs` is the row's ABI and the plan builds it, so nothing joins it; the kernel's launch
 * parameters are fixed, so nothing joins them. The shim's whole instrument surface is therefore four
 * symbols:
 *
 *   hipGetSymbolAddress, once, cached            -- the counter, for the plan line to quote
 *   hipMemcpyToSymbol of the budget, under RAD_R3D_TRACE ONLY -- 0 means the kernel records nothing
 *   hipMemcpy of the ring, under RAD_R3D_TRACE   -- the drain: the records to stderr, host-side
 *   hipMemcpyToSymbol of the write index         -- the drain's reset, so a record is spoken once
 *
 * The budget is re-written on every traced call, not once per process, and `RAD_R3D_DEV` is its
 * size (default 64). Re-arming is one 4-byte copy under a flag that already does four device reads
 * a line, and it buys the property the board needs: a harness can arm exactly -- arm 3, launch 10,
 * count 3 -- between launches, with the kernel untouched. The last write wins, so a served ladder
 * re-arms at every prefill and the budget never runs dry mid-request.
 *
 * The drain runs on every traced call too, BEFORE the arming: it speaks the records the kernel has
 * written since the last call. For a played step there is no shim call in between, so its lines land
 * at the next step's shim call, one step behind the step they describe. The record's `L` and `ctx`
 * restore the order, and the plan line's `dev_launches=` quote is the counter the kernel itself ran,
 * so the two prints of one step line up and the gap between them is the played pass, measured.
 *
 * Inert when the flag is unset: `dev_ring_drain` and `dev_trace_arm` return before the runtime call,
 * the budget keeps its initial 0, and every kernel gates on it. Nothing here is on the served path
 * at all: the shim calls them from inside its existing `r3d_trace_on()` arm.
 */
enum { kDevPrintDefault = 64 };

template <typename T>
void* dev_sym(void** slot, const char* name, const T& symbol)
{
    if (*slot) return *slot;
    void* p = nullptr;
    const hipError_t e = hipGetSymbolAddress(&p, HIP_SYMBOL(symbol));   /* the T& overload */
    *slot = (e == hipSuccess) ? p : nullptr;
    if (e != hipSuccess)
        std::fprintf(stderr, "[r3d dev] %s: hipGetSymbolAddress -> %s\n", name,
                     hipGetErrorName(e));
    return *slot;
}

void* g_dev_budget = nullptr;                      /* hipGetSymbolAddress cache, resolved once */
void* g_dev_launches = nullptr;

unsigned long long dev_launch_count()
{
    /* `hipMemcpyFromSymbol`, not `hipGetSymbolAddress` + `hipMemcpy`: the arming's
     * `hipMemcpyToSymbol` resolves the stub the runtime owns, and in the engine process a
     * `hipGetSymbolAddress` on a dlopened plugin's symbol can hand back an address the kernel's
     * module does not write to -- the read is a copy of the initial value, forever. The same stub
     * resolution the arming uses is the one the read must use. */
    unsigned long long n = 0;
    (void)hipMemcpyFromSymbol(&n, HIP_SYMBOL(r3d_dev_launches), sizeof n);
    return n;
}

void* g_dev_widx = nullptr;
void* g_dev_ring = nullptr;

/* One record, spoken. The format is the line the board parses: the same field names in the same
 * order, `ids=` the row's columns, `kh` the fold of the FIRST block it walks and `khL` the fold of
 * the LAST, `idsum` a fold of every column so a long row is one comparable number. */
static void dev_speak(const DevRec& r)
{
    std::fprintf(stderr, "[r3d dev] L=%llu dec=%d n_seq=%d rows=%d q_len=%d s=%d len=%d span=%d "
                 "start=%d ctx=%lld pos=%lld jcap=%d splits=%d waves=%d kv=%p bt=%p su=%p cu=%p "
                 "q=%p out=%p bt0=%d bt1=%d bt2=%d slot0=%lld qh=%016llx kh=%016llx ids=",
                 (unsigned long long)r.L, r.dec, r.n_seq, r.rows, r.q_len, r.s, r.len, r.span,
                 r.start, (long long)r.ctx, (long long)r.pos, r.jcap, r.splits, r.waves,
                 (const void*)r.kv, (const void*)r.bt, (const void*)r.su, (const void*)r.cu,
                 (const void*)r.q, (const void*)r.out, r.b[0], r.b[1], r.b[2], (long long)r.slot0,
                 (unsigned long long)r.qh, (unsigned long long)r.kh);
    for (int c = 0; c < r.ncols && c < kDevIds; ++c)
        std::fprintf(stderr, "%s%d", c ? "," : "", r.b[c]);
    std::fprintf(stderr, " ncols=%d bL=%d slotL=%lld khL=%016llx idsum=%lld rowok=%d\n", r.ncols,
                 r.spare, (long long)r.slotL, (unsigned long long)r.khL, (long long)r.idsum,
                 r.rowok);
}

/* The ring, spoken. The drain is a D2H of at most kDevRing records (55 KiB) on a traced call: one
 * sync per shim call, under the flag only. The index is the kernel's own, so the drain can never
 * read past what the kernel wrote, and the reset is to 0, not to a moving base: a record is spoken
 * exactly once, and the counter -- which the drain does not touch -- is the launch census. */
void dev_ring_drain()
{
    if (!r3d_trace_on()) return;
    /* The reads go through `hipMemcpyFromSymbol`, the same stub resolution the arming's
     * `hipMemcpyToSymbol` uses (see `dev_launch_count`): in the engine process a
     * `hipGetSymbolAddress` on a dlopened plugin's symbol is not the address the kernel writes. */
    unsigned w = 0;
    if (hipMemcpyFromSymbol(&w, HIP_SYMBOL(r3d_dev_widx), sizeof w) != hipSuccess) return;
    if (w == 0) return;
    static DevRec buf[kDevRing];
    const unsigned keep = w < (unsigned)kDevRing ? w : (unsigned)kDevRing;
    if (hipMemcpyFromSymbol(buf, HIP_SYMBOL(r3d_dev_ring), (size_t)keep * sizeof(DevRec)) != hipSuccess)
        return;
    for (unsigned i = 0; i < keep; ++i) dev_speak(buf[i]);
    if (w > keep)
        std::fprintf(stderr, "[r3d dev] ring: %u records written, %u kept, %u counted and dropped "
                             "-- the counter is the truth\n", w, keep, w - keep);
    const unsigned zero = 0u;
    (void)hipMemcpyToSymbol(HIP_SYMBOL(r3d_dev_widx), &zero, sizeof zero);
}

void dev_trace_arm()
{
    if (!r3d_trace_on()) return;                   /* the gate: with the flag unset, nothing here */
    int want = kDevPrintDefault;
    if (const char* e = std::getenv("RAD_R3D_DEV")) {
        const int v = std::atoi(e);
        if (v > 0) want = v;                       /* a harness arms exactly; 0 disables */
        else if (v == 0) want = 0;
    }
    static bool said = false;
    const unsigned long long seen = dev_launch_count();
    if (!dev_sym(&g_dev_budget, "r3d_dev_budget", HIP_SYMBOL(r3d_dev_budget))) return;
    const hipError_t e = hipMemcpyToSymbol(HIP_SYMBOL(r3d_dev_budget), &want, sizeof want);
    if (!said || e != hipSuccess) {
        said = true;
        std::fprintf(stderr, "[r3d dev] armed: budget=%d launches_at_arm=%llu %s\n", want,
                     (unsigned long long)seen, e == hipSuccess ? "ok" : hipGetErrorName(e));
    }
}


/* ---- the plan line: the one number a launcher has never been able to read ----------------------
 *
 * `cu_seqlens` is device memory -- the header's rule is that a launcher reads none of it -- and the
 * walk's bound is a single value inside it: `cu[0]`. The served descriptors start at row 0. Under
 * RAD_R3D_TRACE=1 this stops being an inference: the shim copies the descriptor's `n_seq + 1` i32
 * and prints them, once per shape.
 *
 * Inert when the flag is unset. Everything below sits behind `r3d_trace_on()`, the cached flag load
 * `R3D_NO` already pays on a path that is failing anyway: no copy, no latch, no syscall, and no
 * operand the served path did not already touch.
 *
 * The reads, all of them bounded. The descriptor's plane holds exactly `n_seq + 1` ints -- the shim
 * has proved `r3d_numel(CU) == n_seq + 1` above -- and the copy is capped at `kCuPrint` of them,
 * with the LAST entry taken by its own one-int read when the plane is wider, so the printed total is
 * the total and nothing past the plane is touched. `seqused` is read for ONE int, its first, because
 * `ctx` is `seqused[0]` and a plan line that cannot say where the walk bounds is
 * half a line. A null pointer is never dereferenced: the descriptor is read only after `r3d_has(CU)`
 * said it carries bytes, and `SU->data` was required at the top of the shim. `hipMemcpyDefault` is
 * placement-agnostic, so the same line prints for a vram plane and for a host-mapped one.
 *
 * The latch is keyed on the shape -- n_seq, the plane's rows, the kernel's q_len, the table's
 * columns -- because the engine calls this shim once a layer a step and 42 identical lines are
 * noise. It is a plain static array with no lock: the worst a race can do here is print a shape
 * twice, and nothing at all when the flag is off. */
enum { kCuPrint = 64, kPlanShapes = 512 };

/* RAD_R3D_TRACE=2 is the same instrument with ctx in the shape key (see plan_once). Read once. */
bool r3d_trace_wide()
{
    static int wide = -1;
    if (wide < 0) {
        const char* e = std::getenv("RAD_R3D_TRACE");
        wide = (e && e[0] == '2') ? 1 : 0;
    }
    return wide != 0;
}

/* The shape key: n_seq, the plane's rows, the kernel's q_len, the table's
 * columns. `ctx` rides along ONLY under RAD_R3D_TRACE=2, which is the wide form of the same
 * instrument: a serving engine's requests all share the shape (rows 1, one sequence, 128 columns)
 * and differ only in how deep the context has grown, so with the narrow key the block table prints
 * once for the FIRST request and the requests that matter -- the ones after six blocks are in play
 * -- never print. The default (=1) is the narrow key. */
struct PlanKey { long long n_seq, rows, q_len, cols, ctx; };

bool plan_once(long long n_seq, long long rows, long long q_len, long long cols, long long ctx,
               bool by_ctx)
{
    static PlanKey seen[kPlanShapes];
    static int n = 0;
    for (int i = 0; i < n; ++i)
        if (seen[i].n_seq == n_seq && seen[i].rows == rows && seen[i].q_len == q_len &&
            seen[i].cols == cols && (!by_ctx || seen[i].ctx == ctx))
            return false;
    if (n < kPlanShapes) seen[n++] = PlanKey{n_seq, rows, q_len, cols, ctx};
    return true;
}

void plan_line(bool decode, const RadTensor* Q, const RadTensor* KV, const RadTensor* O,
               const RadTensor* CU, const RadTensor* SU, const RadTensor* BT, const void* scratch,
               long long n_seq, long long rows, long long q_len, long long bt_cols, long long bt_rs,
               long long bt_cs, int block_size, int max_ctx, int waves, int splits, bool by_ctx)
{
    /* The walk's bound, the same expression fill() uses: the TABLE's columns times the block,
     * capped by the cache's context (jend = min(ctx, max_blocks * block_size)). */
    const long long reach = (long long)bt_cols * block_size;
    const long long jcap = reach < max_ctx ? reach : (long long)max_ctx;

    long long ctx = -1;                              /* seqused[0], the first sequence's context */
    int ctx0 = 0;
    if (r3d_p(SU) && hipMemcpy(&ctx0, r3d_p(SU), sizeof(int), hipMemcpyDefault) == hipSuccess)
        ctx = ctx0;

    /* The latch lives here, after the one-int ctx read, because the wide key is keyed on it. This
     * whole function is called only under the flag, so the served path pays none of it. */
    if (!plan_once(n_seq, rows, q_len, bt_cols, ctx, by_ctx)) return;

    std::fprintf(stderr,
                 "[r3d] plan: %s n_seq=%lld rows=%lld q_len=%lld cu rank=%u numel=%lld ctx=%lld "
                 "bt cols=%lld pitch=%lld max_ctx=%d -> jcap=%lld waves=%d splits=%d",
                 decode ? "decode" : "prefill", n_seq, rows, q_len,
                 r3d_has(CU) ? (unsigned)CU->rank : 0u, r3d_has(CU) ? r3d_numel(CU) : 0LL, ctx,
                 bt_cols, bt_rs, max_ctx, jcap, waves, splits);

    /* The ADDRESSES the launch bakes. The engine captures this step into a graph after two eager
     * decode steps -- one request generating twelve tokens calls this shim for the prefill and for
     * two decode steps, and for nothing after that -- and a replay re-runs the kernels with the
     * ARGUMENTS frozen at capture. The planes a step reads are named by address in those arguments,
     * so if a later request's plan lives at different addresses, a replay is reading the earlier
     * request's planes and no per-row fixture, ladder or descriptor print can see it. `%p` of the
     * five operands this step reads and the one it writes, nothing dereferenced. */
    std::fprintf(stderr, " q=%p kv=%p bt=%p cu=%p su=%p out=%p part=%p",
                 r3d_p(Q), r3d_p(KV), r3d_p(BT), r3d_p(CU), r3d_p(SU), r3d_p(O), scratch);

    /* The kernel's own launch counter, quoted on the shim's line so the two prints of one step line
     * up: the shim sees a served request's prefill and its first two decode steps, the kernel sees
     * every one, and the gap between the two numbers IS the played pass. One 8-byte device read,
     * under the flag, on a line that is already reading three planes. */
    std::fprintf(stderr, " dev_launches=%llu", (unsigned long long)dev_launch_count());

    if (!r3d_has(CU)) {           /* an absent optional operand is ABSENT, not a refusal: the
                                     rectangular step is the common case and it says so */
        std::fprintf(stderr, " cu=absent (rectangular)");
    } else {
        int cuh[kCuPrint];
        const int take = (int)(n_seq + 1 < kCuPrint ? n_seq + 1 : kCuPrint);
        if (hipMemcpy(cuh, r3d_p(CU), (size_t)take * sizeof(int), hipMemcpyDefault) !=
            hipSuccess) {
            std::fprintf(stderr, " cu=copy failed");
        } else {
            long long cu_last = -1;
            if (n_seq + 1 <= kCuPrint) cu_last = cuh[(int)n_seq];
            else {
                int last = 0;                        /* the plane's last int, nothing beyond it */
                if (hipMemcpy(&last, (const int*)r3d_p(CU) + n_seq, sizeof(int),
                              hipMemcpyDefault) == hipSuccess)
                    cu_last = last;
            }
            std::fprintf(stderr, " cu[0]=%d cu[n_seq]=%lld", cuh[0], cu_last);
            if (n_seq + 1 <= 8)                      /* the whole descriptor, when it is short */
                for (int i = 0; i <= (int)n_seq; ++i)
                    std::fprintf(stderr, " cu[%d]=%d", i, cuh[i]);
        }
    }

    /* The block table's own entries for the row this step walks. The walk's arithmetic -- which
     * block holds key j -- is the one part of the plan this file cannot check from the descriptor,
     * and a reused block id inside one row, or an id the cache does not have, is invisible to every
     * per-row fixture (the drivers deal their ids from a shuffled pool, so they cannot produce the
     * first). Bounded three ways: at most `kCuPrint` columns of ONE row, only columns the shim has
     * already proved the plane covers (the reach check above), and row 1's first entry only when
     * there is a row 1. A -1 is an unmapped block and prints as -1, which is what the kernel sees. */
    if (r3d_has(BT)) {
        const int take = (int)(bt_cols < kCuPrint ? bt_cols : kCuPrint);
        int bth[kCuPrint];
        const int* btp = (const int*)r3d_p(BT);
        bool got = true;
        for (int i = 0; i < take; ++i)
            if (hipMemcpy(&bth[i], btp + (long long)i * bt_cs, sizeof(int),
                          hipMemcpyDefault) != hipSuccess) { got = false; break; }
        std::fprintf(stderr, " bt0=");
        if (!got) std::fprintf(stderr, "(copy failed)");
        else {
            std::fprintf(stderr, "[");
            for (int i = 0; i < take; ++i) std::fprintf(stderr, "%s%d", i ? "," : "", bth[i]);
            std::fprintf(stderr, "]");
            if (take < bt_cols) std::fprintf(stderr, " (%lld cols)", bt_cols);
            if (n_seq >= 2) {
                int r1 = 0;
                if (hipMemcpy(&r1, btp + bt_rs, sizeof(int), hipMemcpyDefault) == hipSuccess)
                    std::fprintf(stderr, " bt1[0]=%d", r1);
            }
        }
    }
    std::fprintf(stderr, "\n");
}

int attn_shim(const RadArgs* a, RadStream s, bool decode)
{
    if (a->n_t < 8) return R3D_NO(RAD_E_INVAL);
    const RadTensor* Q = r3d_opt(a, P_Q);
    const RadTensor* KV = r3d_opt(a, P_KV);
    const RadTensor* BT = r3d_opt(a, P_BT);
    const RadTensor* SU = r3d_opt(a, P_SU);
    const RadTensor* O = r3d_opt(a, P_OUT);
    const RadTensor* KDS = r3d_opt(a, P_KDS);
    const RadTensor* VDS = r3d_opt(a, P_VDS);
    const RadTensor* CU = r3d_opt(a, P_CU);
    if (!Q || !KV || !BT || !SU || !O) return R3D_NO(RAD_E_INVAL);
    if (!Q->data || !KV->data || !BT->data || !SU->data || !O->data) return R3D_NO(RAD_E_INVAL);
    /* `cu_seqlens` present is a ragged batch, and this file serves it: the row map, the positions,
     * the mask and the walk all key off the sequence a row belongs to, which is what the descriptor
     * says it is (see "the ragged batch" in the header). The descriptor is device memory, so what
     * the shim can check here is its shape and dtype; what it cannot see -- cu[0] == 0, monotonicity,
     * cu[n_seq] == the plane's row count -- the kernel clamps, exactly as libref's row loop does. */
    const bool ragged = r3d_has(CU);

    const int head_dim = (int)r3d_geti_or(a, "head_dim", r3d_dim(Q, (int)Q->rank - 1));
    if (head_dim != kHeadDim) return R3D_NO(RAD_E_SHAPE);
    const int gqa = (int)r3d_geti_or(a, "gqa", 1);
    if (gqa != 1 && gqa != 2 && gqa != 4 && gqa != 8) return R3D_NO(RAD_E_SHAPE);
    const int causal = (int)r3d_geti_or(a, "causal", 1);
    const int window = (int)r3d_geti_or(a, "window", 0);
    const float scale = (float)r3d_getf(a, "scale", 0.0);

    /* The kernel's f16 query path is written -- `A.q16` makes the widening a copy -- but the
     * launcher's signature carries one dtype slot, `kv_dt`, and it is the cache's, so the shim has
     * no way to tell the kernel that q is f16: the flag would sit at zero and the kernel would
     * widen f16 codes as bf16, a 2^9 scale error, not a rounding difference. Refused until `q_dt`
     * joins the signature. A row set that widens `q_dtype` to "bf16 f16" must land with that
     * parameter. */
    if (Q->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (O->dtype != Q->dtype) return R3D_NO(RAD_E_DTYPE);
    if (KV->dtype != RAD_BF16 && KV->dtype != RAD_F8E4M3) return R3D_NO(RAD_E_DTYPE);
    if (BT->dtype != RAD_I32 || SU->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);
    const bool quant = KV->dtype == RAD_F8E4M3;
    /* A quantised cache with no descale plane is a cache with scale 1.0, not a refused geometry.
     * The reference reads the planes only when they are there:
     *   const float kds = kdes ? ldt(kdes, s * kv_heads + kh) : 1.0f;
     *   const float vds = vdes ? ldt(vdes, s * kv_heads + kh) : 1.0f;
     * and says it in prose ("the scales are the reader's descales, default 1.0").
     * The op's operand description describes both planes as ABSENT, so a fixture that draws this
     * op hands every implementation null descales; refusing them is refusing the case class. The
     * kernel already carries the null through and widens at 1.0 (`A.kds ? ... : 1.0f`). */
    const bool kgiven = KDS && KDS->data, vgiven = VDS && VDS->data;
    if (!quant && (kgiven || vgiven))
        return R3D_NO(RAD_E_SHAPE);   /* a bf16 cache has no scales: the caller believes something
                                         the op does not have, and dropping that quietly is worse */
    if (quant && kgiven && !r3d_is_f32(KDS)) return R3D_NO(RAD_E_DTYPE);
    if (quant && vgiven && !r3d_is_f32(VDS)) return R3D_NO(RAD_E_DTYPE);

    const int n_head = (int)r3d_geti_or(a, "n_head", r3d_dim(Q, (int)Q->rank - 2));
    if (n_head <= 0 || n_head % gqa) return R3D_NO(RAD_E_SHAPE);
    const int kv_heads = n_head / gqa;
    const int block_size = (int)r3d_geti_or(a, "block_size", r3d_dim(KV, 2));
    if (block_size != kBlock) return R3D_NO(RAD_E_SHAPE);
    if (r3d_dim(KV, 1) != kv_heads) return R3D_NO(RAD_E_SHAPE);
    if (r3d_dim(KV, 3) < 2 * kHeadDim) return R3D_NO(RAD_E_SHAPE);
    const long long nb = r3d_dim(KV, 0);
    const int max_ctx = (int)(nb * block_size);
    const int max_blocks = (int)((max_ctx + block_size - 1) / block_size);

    const long long n_seq = r3d_numel(SU);
    if (n_seq < 1 || n_seq > 65536) return R3D_NO(RAD_E_SHAPE);
    /* The descriptor is [n_seq + 1] i32, contiguous: the kernel reads it with int indices, so an
     * i64 plane is not a wider form of the same operand, it is the wrong operand, and the fix is the
     * plane the op's description names. `seqused` is read the same way and is checked below. */
    if (ragged) {
        if (CU->dtype != RAD_I32) { r3d_trace_desc("cu_seqlens", CU); return R3D_NO(RAD_E_DTYPE); }
        if (!r3d_contig(CU)) { r3d_trace_desc("cu_seqlens", CU); return R3D_NO(RAD_E_STRIDE); }
        if (r3d_numel(CU) != n_seq + 1) { r3d_trace_desc("cu_seqlens", CU); return R3D_NO(RAD_E_SHAPE); }
        if (!r3d_aligned(CU->data, 4)) return R3D_NO(RAD_E_ALIGN);
    }
    long long qh = 0, qt = 0, qld = 0;
    if (!r3d_head_rows(Q, kHeadDim, &qt, &qh, &qld) || qh != n_head) { r3d_trace_desc("q", Q);
        r3d_trace_desc("out", O); return R3D_NO(RAD_E_STRIDE); }
    long long oh = 0, ot = 0, old = 0;
    if (!r3d_head_rows(O, kHeadDim, &ot, &oh, &old) || oh != n_head) { r3d_trace_desc("out", O);
        return R3D_NO(RAD_E_STRIDE); }
    if (qld % 8 || old % 8) return R3D_NO(RAD_E_STRIDE);
    /* Rows per sequence: the plane's row count divided by the sequence count when the batch is
     * rectangular, and nothing at all when it is ragged -- the reference's `q_len = cus ? 0 :
     * rows / n_seq` (the reference's reading), with the plane's row count (its `rows`, and the number a
     * descriptor's cu[n_seq] equals) carried to the launcher as the row count. A ragged batch is not
     * required to divide: 5 rows over 3 sequences is the case the engine sends, and `qt % n_seq` is
     * exactly the shape it arrives in. */
    if (qt < 1) return R3D_NO(RAD_E_SHAPE);
    if (qt > 2147483000LL) return R3D_NO(RAD_E_SHAPE);   /* the kernel's row index is 32-bit */
    if (!ragged && qt % n_seq) return R3D_NO(RAD_E_SHAPE);
    const long long q_len = ragged ? qt : qt / n_seq;      /* the per-sequence length, rectangular */
    const long long rows = qt;                             /* the plane's rows, both forms */
    if (q_len < 1) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(O) < rows * n_head * kHeadDim) return R3D_NO(RAD_E_SHAPE);
    /* The descales are per (sequence, KV head), the reference's indexing: `s * kv_heads + kh`. */
    if (quant && kgiven && (r3d_numel(KDS) < n_seq * kv_heads || !r3d_contig(KDS)))
        return R3D_NO(RAD_E_SHAPE);
    if (quant && vgiven && (r3d_numel(VDS) < n_seq * kv_heads || !r3d_contig(VDS)))
        return R3D_NO(RAD_E_SHAPE);
    if (quant && (kgiven != vgiven)) return R3D_NO(RAD_E_SHAPE);   /* half a pair is a caller bug */
    /* The cache is dense and the launcher assumes it: a slot's halves must land on a 16-byte
     * boundary at every slot, so the cache's strides have to be the dense ones. */
    if (!r3d_contig(KV)) { r3d_trace_desc("kv", KV); return R3D_NO(RAD_E_STRIDE); }
    /* The table is the engine's, sized to the step, so its geometry comes from the tensor and not
     * from the band: `max_blocks` is the TABLE's columns, the row pitch is `stride[rank-2]` and the
     * column stride `stride[rank-1]` (the reference's table read). A five-token prompt carries
     * [n_seq, 1]: one block per sequence, because nothing is behind it. What the shim insists on is
     * that every index the walk can reach is inside the plane -- the last row's pitch plus the last
     * column it can name: a contiguity check asks for a table as wide as the cache and refuses a
     * step the engine sends. */
    const long long bt_cols = BT->rank >= 2 ? r3d_dim(BT, (int)BT->rank - 1) : (n_seq ? r3d_numel(BT) / n_seq : 0);
    const long long bt_rs   = BT->rank >= 2 ? BT->stride[BT->rank - 2] : bt_cols;
    const long long bt_cs   = BT->rank >= 2 ? BT->stride[BT->rank - 1] : 1;
    if (bt_cols < 1 || bt_rs < 1 || bt_cs < 1) { r3d_trace_desc("block_tables", BT);
        return R3D_NO(RAD_E_STRIDE); }
    const long long bt_reach = (n_seq - 1) * bt_rs + (bt_cols - 1) * bt_cs + 1;
    if (r3d_numel(BT) < bt_reach) { r3d_trace_desc("block_tables", BT);
        std::fprintf(stderr, "[r3d] block table: %lld entries, the walk reaches %lld\n",
                     (long long)r3d_numel(BT), bt_reach);
        return R3D_NO(RAD_E_STRIDE); }
    if (!r3d_contig(SU)) { r3d_trace_desc("seqused", SU); return R3D_NO(RAD_E_STRIDE); }
    if (!r3d_aligned(Q->data, 16) || !r3d_aligned(KV->data, 16) || !r3d_aligned(O->data, 16))
        return R3D_NO(RAD_E_ALIGN);

    int splits = decode ? (int)r3d_geti_or(a, "splits", 0) : 1;
    if (splits < 0) return R3D_NO(RAD_E_SHAPE);
    /* The width the launcher will cut the grid on. A ragged batch has no per-sequence length to pick
     * on -- the descriptor is on the device -- so the average sequence decides, as it does in the
     * entry itself. It moves duplicated row slots between tiles and no result. */
    const long long per = ragged ? (rows + n_seq - 1) / n_seq : q_len;
    const int waves = pick_waves((int)per, gqa);
    const int tiles = (int)((per + tok_per_block(gqa, waves) - 1) / tok_per_block(gqa, waves));
    if (splits == 0) splits = decode ? pick_splits(tiles, (int)n_seq, kv_heads) : 1;
    if (splits > 64) splits = 64;
    if (splits > 1) {
        /* The arena the engine offers is the bound: halve the split count until the partials fit,
         * down to 1, which needs none -- and one slice is the whole walk in one block, the same
         * keys in the same order, so it is not a degraded answer, it is the answer without the
         * optimisation. A row that refuses a band because an optimisation has no room is a row that
         * tells the user to go away; the number is the same either way. */
        for (;;) {
            if (splits <= 1) { splits = 1; break; }
            const long long need = qt * n_head * (long long)splits * kPartStride * 4;
            if (a->scratch && a->scratch_bytes >= (size_t)need) break;
            splits /= 2;
        }
        if (r3d_trace_on() && splits > 1)
            std::fprintf(stderr, "[r3d] attn: %d slices into %zu of %zu bytes needed\n",
                         splits, a->scratch_bytes, (size_t)(qt * n_head *
                                                            (long long)splits * kPartStride * 4));
    }

    /* The plan line: what the descriptor actually says, once per shape, under the flag only. The
     * `q_len` printed is the one the kernel is given -- 0 with a descriptor, the per-sequence length
     * without one (the reference's reading) -- and `rows` is the plane's row count, which is what
     * cu[n_seq] is required to equal. Printed AFTER every check: a refused step's shape is the
     * refusal lines' business, not this one's. */
    /* The kernel is the one instrument that a played step runs through: its records sit in the ring
     * until a shim call drains them, so drain FIRST -- the records written since the last call, the
     * played steps included -- and THEN arm the budget this launch will spend, under the flag only
     * (both functions gate themselves on the same cached load). */
    dev_ring_drain();
    dev_trace_arm();

    if (r3d_trace_on())
        plan_line(decode, Q, KV, O, CU, SU, BT, splits > 1 ? (const void*)a->scratch : nullptr,
                  n_seq, rows, ragged ? 0LL : q_len, bt_cols, bt_rs, bt_cs, block_size, max_ctx,
                  waves, splits, r3d_trace_wide());

    /* The entries take the descriptor's row count where a rectangular batch has its per-sequence
     * length: `rows` is the plane's, which is what cu[n_seq] equals and what libref counts off the
     * tensor, and `q_len` is the per-sequence length only when there is no descriptor. */
    const int e = decode
        ? r3d_attn_decode_h128(r3d_addr(Q), r3d_addr(KV), r3d_addr(BT), r3d_addr(SU),
                               r3d_addr(KDS), r3d_addr(VDS), r3d_addr(CU), r3d_addr(O), (int)n_seq,
                               (int)(ragged ? rows : q_len),
                               head_dim, gqa, n_head, block_size, (int)bt_cols, (int)bt_rs,
                               (int)bt_cs, max_ctx, window, causal, scale,
                               (int)KV->dtype, qld, old, splits,
                               (long)(uintptr_t)(splits > 1 ? a->scratch : nullptr),
                               (long)(uintptr_t)s)
        : r3d_attn_prefill_h128(r3d_addr(Q), r3d_addr(KV), r3d_addr(BT), r3d_addr(SU),
                                r3d_addr(KDS), r3d_addr(VDS), r3d_addr(CU), r3d_addr(O), (int)n_seq,
                                (int)(ragged ? rows : q_len), head_dim, gqa, n_head, block_size,
                                (int)bt_cols, (int)bt_rs, (int)bt_cs, max_ctx, window,
                                causal, scale, (int)KV->dtype, qld, old,
                                (long)(uintptr_t)nullptr, (long)(uintptr_t)s);
    if (e == 0) return RAD_OK;
    return R3D_NO(e == -1 ? RAD_E_DTYPE : e == -2 ? RAD_E_SHAPE
                        : e == -3 ? RAD_E_ALIGN : e == -4 ? RAD_E_DEVICE
                        : e == -5 ? RAD_E_SCRATCH : RAD_E_INVAL);
}

}  // namespace

extern "C" int r3d_rad_attn_decode_h128(const RadArgs* a, RadStream s)
{
    return attn_shim(a, s, true);
}

extern "C" int r3d_rad_attn_prefill_h128(const RadArgs* a, RadStream s)
{
    return attn_shim(a, s, false);
}
