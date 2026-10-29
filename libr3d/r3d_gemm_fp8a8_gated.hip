/* r3d_gemm_fp8a8_gated.hip -- gemm_nt_q_gated: the gated MLP's GEMM, its gate and its quantiser in
 * one launch. 48 bands on one 2B model (N=12288 K=2048 group=128 dtype=fp8a8 act=silu, M=64 x47
 * and M=16), and the reason the [M, N] bf16 intermediate never exists.
 *
 *   y[M,n] = gate(a[M,K] @ b[N,K]^T), quantised:  n = N/2, gate half = b's rows [0, n),
 *   up half = [n, N). `q` is E4M3 codes over y, `scale` one f32 per (row, `group` of n), and
 *   `out_bf16`, when bound, is exactly the bf16 the codes were taken over.
 *
 * The law is ref_gemm_nt_q_gated and it is transcribed in its order:
 *
 *   row[j] = quant_dot(ma, mb, SA, SB, m, j, K)              over the FULL N, j ascending
 *   gt = round_bf16(row[i]),  up = round_bf16(row[n + i])    BOTH halves, before the gate
 *   sg = 1 / (1 + exp(-gt))                                  the gate is the FIRST half
 *   h  = f32_to_bf16((act == silu ? gt * sg : sg) * up)      the activation is not narrowed
 *   fp8_row(h, n, group)                                     amax -> scale[m, group], codes
 *
 * Both halves are narrowed before the gate because the unfused pair writes a bf16 [M, N] tensor and
 * re-reads it: the fused form must be byte-identical to the pair it replaces, so the narrowing that
 * the store performed is kept, and `out_bf16` is what the pair's tensor held. The activation itself
 * is NOT narrowed -- `gt * sg` is formed in f32 from the narrowed gate, and the product with `up`
 * rounds once, at the store.
 *
 * ---- the lane map: one lane owns a PAIR of dots -----------------------------------------------
 *
 * The gate couples column i with column n + i, and a wave covers 32 consecutive columns, so the
 * partner column is 192 waves away for the band's n = 6144: no lane-to-lane exchange reaches it, and
 * staging a whole row through LDS to reach it costs a round trip of the [M, N] tensor this op exists
 * to avoid. So the lane owns BOTH products of its column: it walks k ascending with two accumulators
 * and reads two weight rows, `i` and `n + i`. The weight traffic is unchanged -- the pair of rows is
 * exactly the traffic the unfused GEMM spent on the two columns -- and the coupling is free.
 *
 * A block covers ONE group of `group` output columns and RP = 256 / group rows, so a lane is
 *
 *     i = blockIdx.x * group + (tid % group)      m = blockIdx.y * RP + tid / group
 *
 * and a wave's 32 lanes are 32 consecutive columns of one group of one row, for group >= 32. That
 * alignment is what closes the epilogue inside the block: the group's amax is a wave max folded over
 * the group's group/32 waves in LDS, the scale is one store, the codes are one byte a lane. The
 * whole op -- product, gate, quantiser -- is one launch and no scratch, which is why the row is
 * concurrent (`rad_kernel_concurrent(i)` is `scratch == nullptr`) and why the bf16 row of the
 * intermediate is not written, read, or allocated.
 *
 * ---- the two streams: fp8a8 at M <= 8 reads the codes, the rest stages them -------------------
 *
 * The band's served case is a decode: one row, the whole 12288 x 2048 operand, 42 calls a token.
 * The shape above is built for the band's declared ceiling (M = 64, 47 of the 48 bands) and serves
 * that height well; at M = 1 it spends 987 us on 24 MiB of codes, 25 GB/s, and the reason is not
 * the loads. It is the path a code takes: global -> LDS stage -> decode -> dot, with two
 * `group_barrier`s per k-block holding the block in step, and a decoder of ten VALU a code (RDNA3
 * has no fp8 instruction: r3d_codecs.h, and the builtins that look like one are rejected at
 * instruction selection on gfx1102).
 *
 * So the arm is chosen by the row count, in the launcher, from the same band key:
 *
 *   M <= 8 and fp8a8   read the activation and both weight rows straight from memory, 16 bytes at
 *                      a time, decode them out of a 256-entry table the block builds once, and
 *                      multiply. No shared memory between a load and a dot, no barrier in the
 *                      k-loop, six loads in flight.
 *   M >= 9, and every fp8a16 case   the shape above, with the same table decoder in place of the
 *                      arithmetic one. At these heights the grid is 48 x M/2 blocks and occupancy,
 *                      not the window, is what feeds the stream; the stage pays for itself.
 *
 * Measured on the band, vram, GPU 2, nine launches timed on their own, best/median:
 *
 *   M = 1     987.4 / 1154.9  ->   171.3 /  181.6 us     (147 GB/s of codes; the card's 163 GB/s
 *   M = 2    1120.6 / 1196.5  ->   166.2 /  177.1 us      floor for 24 MiB is 154 us, and the same
 *   M = 4    1417.3 / 1581.2  ->   221.0 /  229.6 us      sixteen loads with no decode and no dot
 *   M = 8    1874.0 / 2318.0  ->   403.4 /  412.3 us      run in 139 us)
 *   M = 16   2949.3 / 3226.1  ->  1055.0 / 1160.6 us
 *   M = 32   5123.4 / 5405.1  ->  1660.6 / 1705.0 us
 *   M = 64   8395.3 / 9051.0  ->  2786.5 / 2877.3 us
 *
 * The two numbers behind the shape, both from the band's own stream (24 MiB of codes, vram, 11
 * launches, this kernel's loop with the epilogue off): staged through LDS and decoded arithmetically
 * 497 us; read straight and decoded from the table 187 us; the read alone 139 us. The table alone
 * moves M = 64 from 8395 us to 2786 us -- two thirds of the tall arm's gain is the decoder, and it
 * costs nothing in registers (109 a lane, four waves a SIMD, zero spills).
 *
 * Two things in this file are load-bearing. The table's argument is a PAIR, not an index (see
 * build_e4m3_table). And the decoded pairs go from the table into the dot inline, never through an
 * array (see the tall arm's step_body).
 *
 * ---- where the scale multiplies -------------------------------------------------------------
 *
 * quant_dot, unchanged from r3d_gemm_fp8a16_nt_m1.hip: per group of k a sub-sum from zero over
 * ascending k of the CODES' products, then `acc = acc + (sub * sa) * sb` -- the parentheses are the
 * reference's, the scale multiplies the finished block once, and `sa` is `a_scale[m][gi]` (1 when
 * the activation carries no scale plane). Each lane's two chains are the two calls the unfused pair
 * makes, so the bracketing is the pair's, element for element.
 *
 * b_scale is bf16: this op's shape hook writes `o.t(RAD_BF16, {nb, kb})` with no f32 spelling, so
 * the plain entry reads the plane as bf16 and the `_ex` twin takes the dtype, as the fp8 GEMM's
 * does. bf16 -> f32 widening is exact, so an f32 plane carrying the same values must produce
 * bit-identical output; the driver tests exactly that.
 *
 * ---- the epilogue's one transcendental -------------------------------------------------------
 *
 * libref's sigmoid is `1.0f / (1.0f + (float)std::exp((double)(-x)))`. There is no f64 ALU on this
 * card (1:64, emulated), and the band asks for M*n = 393216 exponentials a layer, which is
 * attention's regime and not the sampler's. `expf` is the accurate device form, as in
 * r3d_gated_quant_fp8's kernel (r3d_quant_fp8.hip), not `__expf`; what it feeds is a bf16
 * narrowing, whose ulp is 2^-8 against the transcendental's ~1e-7, so the narrowing lands the same
 * bf16 everywhere but on a tie. The count of those ties is in the row's prose, measured.
 *
 * ---- what the launcher refuses ----------------------------------------------------------------
 *
 * M > 64 (the bands' ceiling, and past it the weight stream -- M reads of an [N, K] plane from L2 --
 * is the wall, which is a tiled kernel's problem, not this one's); N odd (the reference's own
 * `N & 1` test); a group that is not a power of two in [32, 256] (a wave must sit inside one group,
 * and the block is one group wide); group > n, where the reference defaults the QUANTISER's group to
 * n and leaves the weight's K block at the stated width: two widths this kernel reads as one, so it
 * refuses rather than serve a geometry whose two grids disagree; K % group != 0, because a ragged K
 * block reads past the scale plane; sk > 1, because no K slab closes an epilogue that needs the
 * whole row. Refusals are codes: -1 null, -2 shape, -3 pitch, -4 HIP.
 *
 * ---- relayout: none, and it is a claim ---------------------------------------------------------
 *
 * libref reads the gated weight [N, K] row-major; libr4d stores fp8a8 weights in WMMA fragment order
 * (`r4d.fp8.w.frag64`). The two readings are ~1.35 apart, a relative error, not noise, so this row
 * carries no relayout and says so: a row that inherits the fp8 GEMM's unrelayout scores 1.35 against
 * the per-op oracle and reads as a broken kernel. The stored order is the reading order here, which
 * is the library's rule for every row.
 */

#include <hip/hip_runtime.h>          // __forceinline__, which r3d_codecs.h's R3D_HD spells

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <algorithm>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWaves        = kBlockThreads / kWave;
constexpr int kStep         = 16;     // codes a lane takes per step: one 16-byte load
constexpr int kWin          = 64;     // codes a lane takes per WINDOW: four loads per stream
constexpr int kGroupMin     = 32;     // a wave's 32 columns must sit inside one group
constexpr int kGroupMax     = 256;    // a block is one group wide, at 256 threads
constexpr int kMMax         = 64;     // the bands' ceiling: 47 of the 48 are M = 64
constexpr float kTopE4M3    = 448.0f;  // q8_group's top for E4M3: the largest finite code

/* The E4M3 decoder as a table: 256 f16, one entry per code. The codec's arithmetic decode is ten
 * VALU a code (r3d_codecs.h -- RDNA3 has no fp8 instruction, and the builtins that look like one
 * are rejected at instruction selection), and this kernel decodes every weight code it multiplies,
 * twice over at RP = 2, which at M = 1 is every code in the operand. The table costs two
 * `ds_read_u16` and a pack a PAIR, and it is exact by construction: the entry for code c is the
 * dword the arithmetic path returns for c, computed by that path at the block's first instruction.
 * The unit test reads all 256 entries back and compares them with the codec. */
__device__ __forceinline__ void build_e4m3_table(uint16_t* tab, uint32_t tid)
{
    for (uint32_t u = tid; u < 128; u += kBlockThreads)
        /* The codec's argument is a PAIR, low code in the low byte; the entry is indexed by code.
         * Feeding it the index alone fills the even entries and decodes code 0 into every odd one,
         * so the kernel multiplies half the plane by zero -- the board's `codes256` case, whose
         * weight is the code enumeration, shows it: rel_l2 9.6e-01, not a NaN. */
        *(uint32_t*)&tab[2 * u] = e4m3x2_to_f16x2((u * 2) | ((u * 2 + 1) << 8));
}

__device__ __forceinline__ uint32_t tab_pair(const uint16_t* tab, uint32_t pair)
{
    const uint32_t lo = *(const uint16_t __attribute__((address_space(3))) *)(tab + (pair & 0xffu));
    const uint32_t hi = *(const uint16_t __attribute__((address_space(3))) *)(tab + (pair >> 8));
    return lo | (hi << 16);
}

/* One step of the skinny arm: sixteen codes of activation and sixteen of weight, both coded, both
 * out of the same table. This is the whole of the reference's inner step -- `sub += a[k] * b[k]`,
 * k ascending -- with the pairs decoded as each dot consumes them. Four dot4, the pairs in order,
 * one accumulator; nothing about it depends on where the bytes came from. */
__device__ __forceinline__ float dot_coded_step(const uint16_t* tab, const r3d_u32x4 a,
                                                const r3d_u32x4 c, float sub)
{
    sub = dot4_f16x4(tab_pair(tab, a.x & 0xffffu), tab_pair(tab, a.x >> 16),
                     tab_pair(tab, c.x & 0xffffu), tab_pair(tab, c.x >> 16), sub);
    sub = dot4_f16x4(tab_pair(tab, a.y & 0xffffu), tab_pair(tab, a.y >> 16),
                     tab_pair(tab, c.y & 0xffffu), tab_pair(tab, c.y >> 16), sub);
    sub = dot4_f16x4(tab_pair(tab, a.z & 0xffffu), tab_pair(tab, a.z >> 16),
                     tab_pair(tab, c.z & 0xffffu), tab_pair(tab, c.z >> 16), sub);
    sub = dot4_f16x4(tab_pair(tab, a.w & 0xffffu), tab_pair(tab, a.w >> 16),
                     tab_pair(tab, c.w & 0xffffu), tab_pair(tab, c.w >> 16), sub);
    return sub;
}

/* One step's sixteen codes into the accumulator: four dot4, the pairs decoded out of the table as
 * each dot consumes them. Materialising the eight pairs first and feeding the dots from the array
 * is what the register file cannot afford here: the scheduler hoists the sixteen table reads, so
 * all eight decoded pairs are live at once on top of the window's loads. Consuming a pair in the
 * dot that wants it keeps two registers live where the array keeps eight -- which is the
 * difference between this arm compiling at zero spills and spilling 374 bytes against the
 * 128-register budget a 256-thread block gets at four waves a SIMD. The order is unchanged: the
 * pairs ascend, four dot4 a step, into one accumulator. */
__device__ __forceinline__ float dot_step(const uint16_t* tab, const r3d_u32x4 c,
                                          const r3d_u32x4 a0, const r3d_u32x4 a1, float sub)
{
    sub = dot4_f16x4(a0.x, a0.y, tab_pair(tab, c.x & 0xffffu), tab_pair(tab, c.x >> 16), sub);
    sub = dot4_f16x4(a0.z, a0.w, tab_pair(tab, c.y & 0xffffu), tab_pair(tab, c.y >> 16), sub);
    sub = dot4_f16x4(a1.x, a1.y, tab_pair(tab, c.z & 0xffffu), tab_pair(tab, c.z >> 16), sub);
    sub = dot4_f16x4(a1.z, a1.w, tab_pair(tab, c.w & 0xffffu), tab_pair(tab, c.w >> 16), sub);
    return sub;
}


template <bool NT>
__device__ __forceinline__ r3d_u32x4 ldc(const uint8_t* p)      // 16 codes
{
    if constexpr (NT) return *(const __attribute__((address_space(1))) r3d_u32x4*)p;
    else return __builtin_bit_cast(r3d_u32x4, *(const uint4*)p);
}

__device__ __forceinline__ r3d_u32x4 lds_a(const uint32_t* p) { return *(const r3d_u32x4*)p; }

/* q8_group's clamp, with q8_group's comparisons: fminf/fmaxf replace a NaN with the bound, and a NaN
 * activation is a value the reference encodes as the format's NaN, not as +-top. */
__device__ __forceinline__ float clamp_top(float c)
{
    if (c > kTopE4M3) c = kTopE4M3;
    if (c < -kTopE4M3) c = -kTopE4M3;
    return c;
}

/* y[M,n] = gate(a @ b^T), with the quantiser. GROUP is the block's width in output columns AND the
 * weight's K block AND the quantiser's group: one `group` parameter, three grids, which is what the
 * op states: kb, nb and ng are derived from the one key. ACT is the gate, A8 the coded activation,
 * NT the weight load's lowering.
 *
 * The activation is staged in LDS per K block, widened to f16 pairs on the way in -- E4M3 -> f16 and
 * bf16 -> f16 are both exact over f16's normal range. A block stages RP * GROUP values, 512 bytes at
 * every supported GROUP, and a wave reads its row's 16 values per step as one broadcast ds_read: all
 * 32 lanes of a wave own the same m, so all 32 read the same address. The weight is not staged: a
 * lane streams two rows of it and no other lane reads them with it. */
template <int GROUP, int ACT, bool A8, bool NT, bool WIN>
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(WIN ? 3 : 8) void gated_kernel(
    const uint8_t* __restrict__ A, const float* __restrict__ AScale,
    const uint8_t* __restrict__ B, const void* __restrict__ BScale, int bs16,
    uint8_t* __restrict__ Q, float* __restrict__ S, uint16_t* __restrict__ Y,
    int M, int N, int K, int a_ld, int q_ld, int y_ld, int nkb, int ng)
{
    constexpr int RP = kBlockThreads / GROUP;   // rows a block carries
    constexpr int AP = GROUP / 2;               // dwords a staged activation row (f16 pairs)
    constexpr int WPG = GROUP / kWave;          // waves covering one group of columns
    static_assert(RP >= 1 && WPG >= 1 && RP * WPG == kWaves, "a block is one group wide");

    __shared__ uint32_t s_a[RP * AP];
    __shared__ float s_mx[kWaves];
    __shared__ uint16_t s_tab[256];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;

    /* The block's decoder, filled before any lane reads it: 128 dwords, one per pair of codes, by
     * the codec's own arithmetic. The cost is 128 decodes a block -- at the band's M = 64 that is
     * 0.02% of the block's decodes, and at M = 1 it is paid once per block for 128 codes x 16 k
     * blocks of table reads. */
    build_e4m3_table(s_tab, tid);
    group_barrier();

    const uint32_t gidx = blockIdx.x;                     // the group of output columns
    const uint32_t mloc = tid / GROUP;                    // this lane's row within the block
    const uint32_t i    = gidx * GROUP + (tid % GROUP);   // the fused column
    const uint32_t m    = (uint32_t)blockIdx.y * RP + mloc;

    const int n      = N / 2;
    /* A ragged last group runs past n: the pair's rows are clamped, so no lane reads past the weight
     * plane, and its stores are guarded below. A row past M re-reads the last real row. */
    const int i_real = (int)(i < (uint32_t)n ? i : (uint32_t)(n - 1));
    const int m_real = (int)(m < (uint32_t)M ? m : (uint32_t)(M - 1));

    const uint8_t* wg = B + (size_t)i_real * K;           // the gate's row
    const uint8_t* wu = B + (size_t)(n + i_real) * K;     // the up's row, n rows further on
    /* b_scale is [ceil(N/group)][ceil(K/group)]: make_scales gives rblk = group (fewer scale rows
     * than operand rows is a block grid), so a row's scale row is its row index over the block. */
    const size_t bsg = (size_t)((uint32_t)i_real / GROUP) * (size_t)nkb;
    const size_t bsu = (size_t)((uint32_t)(n + i_real) / GROUP) * (size_t)nkb;
    const float* bsF  = bs16 ? nullptr : (const float*)BScale;
    const uint16_t* bsH = bs16 ? (const uint16_t*)BScale : nullptr;
    const float* as = A8 ? AScale + (size_t)m_real * nkb : nullptr;

    float acc_g = 0.0f, acc_u = 0.0f;

    /* One scale block per round, `nkb` of them: stage the block's activation, walk the block's k
     * with two chains, fold the block's two scales in at the block's end. K is a multiple of GROUP,
     * which the launcher requires, so every block is whole and the staging loop has no tail. */
    /* The skinny arm's activation pointer: the row, in bytes. */
    const uint8_t* ar = (A8 && WIN) ? (const uint8_t*)A + (size_t)m_real * a_ld : nullptr;

    for (int kb = 0; kb < K; kb += GROUP) {
        const int gi = kb / GROUP;
        /* The staged round, and the barrier around it, belong to the tall arms. At the skinny height
         * -- M <= 8, the threshold `R3D_GT_G` selects on, 16 KiB of activation at the band's K --
         * the whole activation is M*K bytes and the 48 column groups of a row all
         * read the same slice of it, so it lives in L1 from the first k-block on and staging it
         * through shared memory buys nothing but a barrier a block a k-block. The numbers: the
         * same stream read straight from memory 187 us, the same stream staged through LDS 497 us,
         * at 48 blocks, 24 MiB of codes, on a part of this family. */
        if constexpr (!WIN) {
        constexpr int kUnit = 8;                       // codes (fp8a8) or elements (bf16) a unit
        constexpr int nu = GROUP / kUnit;
#pragma unroll
        for (int u = (int)tid; u < RP * nu; u += kBlockThreads) {
            const int r = u / nu, c = (u % nu) * kUnit;
            const int rm = (int)((uint32_t)blockIdx.y * RP + (uint32_t)r);
            const int rr = rm < M ? rm : M - 1;
            uint32_t* dst = s_a + (size_t)r * AP + c / 2;
            if constexpr (A8) {
                const r3d_u32x2 b8 = *(const __attribute__((address_space(1))) r3d_u32x2*)(
                    A + (size_t)rr * a_ld + kb + c);
                dst[0] = e4m3x2_to_f16x2(b8.x & 0xffffu);
                dst[1] = e4m3x2_to_f16x2(b8.x >> 16);
                dst[2] = e4m3x2_to_f16x2(b8.y & 0xffffu);
                dst[3] = e4m3x2_to_f16x2(b8.y >> 16);
            } else {
                /* Cast to the element pointer FIRST: the row and k offsets are ELEMENT offsets,
                 * which for bf16 is twice the byte count (r3d_gemm_fp8a16_nt_m1.hip). */
                const r3d_u32x4 b16 = *(const __attribute__((address_space(1))) r3d_u32x4*)(
                    (const uint16_t*)A + (size_t)rr * a_ld + kb + c);
                dst[0] = bf16x2_to_f16x2(b16.x);
                dst[1] = bf16x2_to_f16x2(b16.y);
                dst[2] = bf16x2_to_f16x2(b16.z);
                dst[3] = bf16x2_to_f16x2(b16.w);
            }
        }
        group_barrier();

        }

        const uint32_t* ap = s_a + (size_t)mloc * AP;
        /* One chain at a time. The gate's product, k ascending into `sub_g`, then the up's into
         * `sub_u` -- the two chains are the two calls the unfused pair makes, they share nothing,
         * and each one's bracketing is that call's, element for element.
         *
         * Sequencing them, rather than interleaving them step by step, is what the register file
         * asks for. The window is eight 16-byte loads a stream; holding both streams' windows
         * live at once is 64 registers of loads on top of the decode's temporaries, the staged
         * activation's reads, the two accumulators and the block's pointers, and at 256 threads
         * (four waves a SIMD, 128 registers a lane) it spills 365 bytes. One chain's window is
         * 32 registers, and the arm compiles clean.
         *
         * On the band's stream -- 24 MiB of codes, vram, 11 launches, GPU 2, this kernel's loop
         * with the epilogue off: 16-byte step with the arithmetic decoder 584 us; table decoder,
         * same step 453 us; table, window 64 283 us; table, window 128 187 us; the same sixteen
         * loads with no decode and no dot 139 us. The op without the table decoder and the window
         * spends 1474 us a call. */
        float sub_g, sub_u;
        if constexpr (WIN) {
            /* The skinny arm, M <= 8 (the threshold `R3D_GT_G` selects on): read the codes, decode
             * them out of the block's table, multiply
             * them. No shared memory between the load and the dot, and no barrier in the k-loop --
             * the two chains, gate then up, k ascending inside each, the scales folded at the block's
             * end, the epilogue below untouched.
             *
             * Three loads a step, sixteen bytes each: the activation's sixteen codes, the gate's,
             * the up's. Six of them are in flight at the depth below, against the 163 GB/s the card
             * does; the tall arms feed the same stream with occupancy, at 32 waves a CU.
             *
             * On this band's stream (24 MiB of codes, vram, 11 launches, GPU 2, epilogue off):
             * the staged shape 497 us; the same codes read straight 187 us; the same sixteen loads
             * with no decode and no dot 139 us. The op without the table decoder and the window
             * spends 1474 us a call. */
            constexpr int kNS = GROUP / kStep;                // steps a block
            constexpr int kD  = 2;                            // loads a chain in flight
            auto chain = [&](const uint8_t* w, float sub) {
                auto ld = [&](int st) {
                    return ldc<NT>(w + kb + (st < kNS ? st : kNS - 1) * kStep);
                };
                /* Named registers, not an array: an array indexed across a round -- even with the
                 * round count constant and the loop marked for unrolling -- puts the codes in the
                 * private segment, 572 bytes a lane, and every code crosses scratch on its way from
                 * the load to the dot. Two named dwords-in-four, prefetched two steps ahead, keep
                 * the same six loads in flight and the frame at zero. */
                r3d_u32x4 c0 = ld(0), c1 = ld(1);
#pragma unroll
                for (int st = 0; st < kNS; st += kD) {
                    __builtin_amdgcn_sched_barrier(0);
                    const r3d_u32x4 a0 = ldc<NT>(ar + kb + (st + 0) * kStep);
                    const r3d_u32x4 a1 = ldc<NT>(ar + kb + (st + 1) * kStep);
                    const r3d_u32x4 n0 = ld(st + kD + 0), n1 = ld(st + kD + 1);
                    sub = dot_coded_step(s_tab, a0, c0, sub);
                    sub = dot_coded_step(s_tab, a1, c1, sub);
                    c0 = n0;
                    c1 = n1;
                }
                return sub;
            };
            sub_g = chain(wg, 0.0f);
            sub_u = chain(wu, 0.0f);
        } else {
            /* The tall arms: one 16-code step a round, both streams in the same round.
             * At M >= 8 the block grid is 48 x M/2 blocks, the wave slots are full, and the stream
             * is fed by occupancy -- 109 registers a lane, four waves a SIMD, 32 waves a CU. The
             * window is a skinny-arm fix: it buys its loads per lane at 32 registers, which on
             * this arm is a fifth wave a SIMD given up, and on the 47 bands at M = 64 that is the
             * whole op. Measured: windowed at M = 64 costs more than it saves. */
            sub_g = 0.0f;
            sub_u = 0.0f;
            /* The round's unroll depth is a register decision, and at group = 32 it is the whole
             * decision: the block is two steps wide, so `unroll 2` is a full unroll, the scheduler
             * takes all thirty-two table reads of the two chains and hoists them above the dots,
             * and the arm lands at 128 registers with 12 spilled (the same arm at group >= 64, in
             * the same file, is 106 to 112 with none). Left as a loop the body is one step. */
            /* Inline, from the table into the dot, on purpose. Materialising the eight decoded
             * pairs into an array on the way -- the shape the arithmetic decoder above feeds --
             * puts the plane about a third away from the reference at every tall height (M = 7:
             * rel_l2 6.6e-01; M = 64: -nan), with the table itself exact on the device, entry by
             * entry and pair by pair. The array form is the one that miscomputes here; do not
             * restore it as a tidy-up. */
            auto step_body = [&](int k) {
                const r3d_u32x4 a0 = lds_a(ap + k / 2);
                const r3d_u32x4 a1 = lds_a(ap + k / 2 + 4);
                sub_g = dot_step(s_tab, ldc<NT>(wg + kb + k), a0, a1, sub_g);
                sub_u = dot_step(s_tab, ldc<NT>(wu + kb + k), a0, a1, sub_u);
            };
            if constexpr (GROUP <= 32) {
#pragma unroll 1
                for (int k = 0; k < GROUP; k += kStep) step_body(k);
            } else {
#pragma unroll 2
                for (int k = 0; k < GROUP; k += kStep) step_body(k);
            }
        }
        if constexpr (!WIN)
            group_barrier();             // the next block's stage must not race this one's reads

        /* The two finished sub-sums, each scaled once, in libref's parentheses. */
        const float sa = A8 ? as[gi] : 1.0f;
        acc_g = acc_g + (sub_g * sa) * (bs16 ? bf16_to_f32(bsH[bsg + gi]) : bsF[bsg + gi]);
        acc_u = acc_u + (sub_u * sa) * (bs16 ? bf16_to_f32(bsH[bsu + gi]) : bsF[bsu + gi]);
    }

    /* ---- the epilogue: narrow both halves, gate, quantise the group -------------------------- */
    const float gt = bf16_to_f32(f32_to_bf16(acc_g));
    const float up = bf16_to_f32(f32_to_bf16(acc_u));
    const float sg = 1.0f / (1.0f + expf(-gt));
    const uint16_t h = f32_to_bf16((ACT == R3D_ACT_SILU ? gt * sg : sg) * up);

    const bool keep = (i < (uint32_t)n) && (m < (uint32_t)M);
    if (keep && Y) Y[(size_t)m * y_ld + i] = h;
    /* A column past n, and a row past M, contribute 0.0f: padding takes no part in the amax and
     * takes no store, which is quant_act_fp8's rule for a ragged row. */
    const float am = keep ? fabsf(bf16_to_f32(h)) : 0.0f;

    float mx = wave_max(am);             // fmaxf skips a NaN, as `if (m > amax)` does
    if (lane == 0) s_mx[wave] = mx;
    group_barrier();
    mx = s_mx[mloc * WPG];
#pragma unroll
    for (int w = 1; w < WPG; ++w) mx = fmaxf(mx, s_mx[mloc * WPG + w]);

    const float sc = mx > 0.0f ? mx * (1.0f / kTopE4M3) : 1.0f;
    /* One store per (row, group): the group's WPG waves all hold the same max, so the first writes
     * and the rest stand down. `scale` is dense -- the launcher takes no pitch for it -- so a row is
     * `ng` scales and the group index is the block's. */
    if (keep && lane == 0 && (wave % WPG) == 0) S[(size_t)m * ng + gidx] = sc;

    if (keep) {
        const float inv = 1.0f / sc;
        Q[(size_t)m * q_ld + i] = f32_to_e4m3(clamp_top(bf16_to_f32(h) * inv));
    }
}

}  // namespace

/* y[M,n] = gate(a[M,K] @ dequant(b)[N,K]^T), E4M3 codes out, n = N/2 inside.
 *
 *   a         [M][K]                E4M3 codes (fp8a8, with `a_scale`) or bf16 (fp8a16)
 *   a_scale   [M][ceil(K/group)]    f32, dense, or 0
 *   b         [N][K]                E4M3 codes, row-major, k contiguous: gate rows [0,n), up rows
 *                                   [n,N)
 *   b_scale   [ceil(N/group)][ceil(K/group)]   bf16 (the op's stated dtype) or f32, block-major
 *   q         [M][n]                E4M3 codes, `q_ld` elements a row (0 = n)
 *   scale     [M][ceil(n/group)]    f32, DENSE (no pitch is taken)
 *   out_bf16  [M][n]                bf16, `y_ld` elements a row (0 = n), optional: the pre-gate...
 *                                   the gated bf16 tensor the codes were taken over
 *
 * `group` is the weight's K block, the weight's row block and the output quantiser's group at once,
 * which is what the op's single key states; 0 takes the checkpoint's 128. `act` is R3D_ACT_SILU
 * (down_proj's input) or R3D_ACT_SIGMOID (a gated attention o_proj's). `sk` is accepted as 0/1 and
 * refused above it: the epilogue needs a whole row. `nt` selects the weight load's lowering.
 * `scratch` is unused -- the op needs no arena, which is what keeps the row concurrent.
 *
 * Returns 0, -1 a null pointer, -2 a shape it does not serve, -3 a pitch it cannot read, -4 HIP
 * failed. */
extern "C" int r3d_gemm_nt_q_gated_ex(long a, long a_scale, long b, long b_scale, long q, long scale,
                                      long out_bf16, int M, int N, int K, int group, int act,
                                      int a_ld, int q_ld, int y_ld, int sk, int nt, long scratch,
                                      int bs_dt, long stream)
{
    if (!a || !b || !b_scale || !q || !scale) return -1;
    if (M < 1 || N < 2 || (N & 1) || K < 1) return -2;
    if (M > kMMax) return -2;
    if (act != R3D_ACT_SILU && act != R3D_ACT_SIGMOID) return -2;
    const int n = N / 2;
    if (group <= 0) group = R3D_FP8_BLOCK;
    if (group > n) return -2;                                   // two grids, one key: see the header
    if (group < kGroupMin || group > kGroupMax || (group & (group - 1))) return -2;
    if (K % group) return -2;                                   // a ragged block reads past the scales
    const int nkb = K / group, ng = (n + group - 1) / group;
    if (!a_ld) a_ld = K;
    if (!q_ld) q_ld = n;
    if (!y_ld) y_ld = n;
    if (a_ld % 8) return -3;                                    // 8-code (and 8-element) staging
    if (sk == 0) sk = 1;
    if (sk > 1) return -2;                                      // no K slab closes this epilogue
    (void)scratch;

    const int bs16 = (bs_dt == RAD_BF16);      /* anything else, 0 included, reads the plane as f32 */
    const int a8 = a_scale != 0;
    const int rp = kBlockThreads / group;
    const dim3 grid((unsigned)ng, (unsigned)((M + rp - 1) / rp)), blk(kBlockThreads);

/* The skinny arms -- the decode's height -- take the windowed stream; the tall arms keep
 * the step. One number, two kernels, and the band key that names them is the same one.
 *
 * M <= 8 is the shipped threshold and it is a tunable, not a fact. With the table decoder in both
 * arms, N=12288 K=2048 group=128 vram: at M=8 skinny 402 against staged 754, at M=16 skinny 955
 * against staged 1030 -- so the crossover is above 16; M=9..15 is unmeasured. Decode lives at M=1
 * and a draft's verify at M<=8, which is what the threshold is chosen for; a prefill at 9..16 rows
 * would take the skinny arm on the M=16 numbers, for about 7%. */
#define R3D_GT_G(G, A, A8, NTV)                                                                  \
    do {                                                                                         \
        if (A8 && M <= 8) R3D_GT_W(G, A, A8, NTV, true);                                          \
        else R3D_GT_W(G, A, A8, NTV, false);                                                     \
    } while (0)
#define R3D_GT_W(G, A, A8, NTV, WV)                                                              \
    R3D_KLAUNCH(R3D_KNAME((gated_kernel<G, A, A8, NTV, WV>)), grid, blk, (hipStream_t)stream,    \
                (const uint8_t*)a, (const float*)a_scale, (const uint8_t*)b,                     \
                (const void*)b_scale, bs16, (uint8_t*)q, (float*)scale, (uint16_t*)out_bf16,     \
                M, N, K, a_ld, q_ld, y_ld, nkb, ng)
#define R3D_GT_NT(G, A, A8) do { if (nt) R3D_GT_G(G, A, A8, true); else R3D_GT_G(G, A, A8, false); } while (0)
#define R3D_GT_A8(G, A) do { if (a8) R3D_GT_NT(G, A, true); else R3D_GT_NT(G, A, false); } while (0)
#define R3D_GT_ACT(G) do { if (act == R3D_ACT_SILU) R3D_GT_A8(G, R3D_ACT_SILU);                    \
                           else                    R3D_GT_A8(G, R3D_ACT_SIGMOID); } while (0)
#define R3D_GT(G) do { switch (group) { case G: R3D_GT_ACT(G); break; default: break; } } while (0)
    R3D_GT(32);
    R3D_GT(64);
    R3D_GT(128);
    R3D_GT(256);
#undef R3D_GT
#undef R3D_GT_ACT
#undef R3D_GT_A8
#undef R3D_GT_NT
#undef R3D_GT_W
#undef R3D_GT_G
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The declaration in r3d.h carries no scale dtype, and this op's shape hook states bf16 -- the hook
 * has no f32 spelling at all -- so the plain entry reads bf16 and the `_ex` twin above carries the
 * key. A caller with an f32 scale plane (the `fp8a8f32` reading of the unfused GEMM) lands on `_ex`
 * with RAD_F32; the kernel widens bf16 -> f32 exactly, so the two agree to the bit. */
extern "C" int r3d_gemm_nt_q_gated(long a, long a_scale, long b, long b_scale, long q, long scale,
                                   long out_bf16, int M, int N, int K, int group, int act,
                                   int a_ld, int q_ld, int y_ld, int sk, int nt, long scratch,
                                   long stream)
{
    return r3d_gemm_nt_q_gated_ex(a, a_scale, b, b_scale, q, scale, out_bf16, M, N, K, group, act,
                                  a_ld, q_ld, y_ld, sk, nt, scratch, RAD_BF16, stream);
}

/* ============================================================== the plugin ABI entry point
 *
 * gemm_nt_q_gated's operands are [a, a_scale, b, b_scale, q, scale, out_bf16?] and its parameters
 * M, N, K, group, dtype, act (libref's pGemmQGated/oGemmQGated: `act` REQUIRED, `a_scale` REQUIRED,
 * `out_bf16` optional). The shape hook declines every scheme but fp8a8, so the row binds fp8a8 and
 * the shim reads the coded activation off the operands. */
enum { G_A = 0, G_ASCALE, G_B, G_BSCALE, G_Q, G_SCALE, G_OB };

#define R3D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R3D_NO(_s); } while (0)
static int dt_f32(const RadTensor* t) { return r3d_is_f32(t) ? RAD_OK : RAD_E_DTYPE; }
static int dt_bf16(const RadTensor* t) { return r3d_is_bf16(t) ? RAD_OK : RAD_E_DTYPE; }
static int dt_e4m3(const RadTensor* t) { return r3d_is_e4m3(t) ? RAD_OK : RAD_E_DTYPE; }

extern "C" int r3d_rad_gemm_nt_q_gated(const RadArgs* a, RadStream s)
{
    const RadTensor* A  = r3d_opt(a, G_A);
    const RadTensor* AS = r3d_opt(a, G_ASCALE);
    const RadTensor* B  = r3d_opt(a, G_B);
    const RadTensor* BS = r3d_opt(a, G_BSCALE);
    const RadTensor* Q  = r3d_opt(a, G_Q);
    const RadTensor* SC = r3d_opt(a, G_SCALE);
    const RadTensor* OB = r3d_opt(a, G_OB);
    if (!A || !B || !BS || !Q || !SC) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_e4m3(B)) return R3D_NO(RAD_E_DTYPE);
    R3D_NEED(dt_e4m3(Q));
    R3D_NEED(dt_f32(SC));
    if (OB) R3D_NEED(dt_bf16(OB));
    /* The activation: E4M3 codes with a per-(row, group) f32 scale (fp8a8, the scheme this op's
     * shape hook and every band states), or bf16 with no scale (fp8a16, which the hook declines, so
     * no band arrives: the kernel carries the path because the launcher does, and the shim accepts
     * it on the operands). E5M2 has no decoder here. */
    const bool a8 = AS != nullptr;
    if (a8) {
        if (!r3d_is_e4m3(A)) return R3D_NO(RAD_E_DTYPE);
        R3D_NEED(dt_f32(AS));
    } else {
        R3D_NEED(dt_bf16(A));
    }
    /* b_scale is bf16 -- `o.t(RAD_BF16, {nb, kb})` is this op's own line in the hook, and the
     * container's `weight_scale_inv` for the whole quantised family. f32 is accepted because the
     * widening is exact and the kernel reads both: the driver's bf16/f32 pair must agree element
     * for element. */
    const bool bs16 = r3d_is_bf16(BS);
    if (!bs16 && !r3d_is_f32(BS)) return R3D_NO(RAD_E_DTYPE);

    /* `act` is required by the schema (P_STR, not P_STRO): an implementation may compile one gate,
     * and an optional parameter would let the row resolve for the other. libref's default is -1 and
     * it refuses it; so does this shim, with no `mode` integer spelling to fall back on. */
    const int act = r3d_act_code(a, "act", nullptr, -1);
    if (act != R3D_ACT_SILU && act != R3D_ACT_SIGMOID) return R3D_NO(RAD_E_INVAL);

    const long long K = r3d_geti_or(a, "K", r3d_dim(B, (int)B->rank - 1));
    const long long N = r3d_geti_or(a, "N", r3d_dim(B, (int)B->rank - 2));
    if (K < 1 || N < 2 || (N & 1)) return R3D_NO(RAD_E_SHAPE);
    const long long n = N / 2;

    /* Rows come from `q`, never from the `M` parameter, which is the band's upper bound. libref
     * reads numel(q) / n; on a plane whose pitch exceeds n that counts the padding, so the shape's
     * own row count clamps it. A dense plane -- every case the fixture draws -- reads the same
     * number either way. */
    const long long qnel = r3d_numel(Q);
    if (qnel < n || qnel % n) return R3D_NO(RAD_E_SHAPE);
    long long M = qnel / n;
    if (Q->rank >= 2 && r3d_dim(Q, (int)Q->rank - 2) < M) M = r3d_dim(Q, (int)Q->rank - 2);
    if (M < 1 || M > kMMax) return R3D_NO(RAD_E_SHAPE);

    /* The group: the weight's K block, the weight's row block and the quantiser's group, from the
     * one key the op states. The reference defaults 0 to `group`-as-stated for the grids and
     * `g > n` to n for the quantiser alone; this kernel reads one width, so a group wider than the
     * fused width is refused rather than served on a grid the two sides do not share. */
    const long long group = r3d_geti_or(a, "group", R3D_FP8_BLOCK);
    if (group < kGroupMin || group > kGroupMax || (group & (group - 1))) return R3D_NO(RAD_E_SHAPE);
    if (group > n) return R3D_NO(RAD_E_SHAPE);
    if (K % group) return R3D_NO(RAD_E_SHAPE);
    const long long nkb = K / group, nb = (N + group - 1) / group, ng = (n + group - 1) / group;
    if (r3d_numel(A) < M * K || r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(BS) < nb * nkb) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(SC) < M * ng) return R3D_NO(RAD_E_SHAPE);
    if (AS && r3d_numel(AS) < M * nkb) return R3D_NO(RAD_E_SHAPE);
    if (OB && r3d_numel(OB) < M * n) return R3D_NO(RAD_E_SHAPE);

    long long arows = 0, qrows = 0, srows = 0, yrows = 0;
    const long long a_ld = r3d_rows_pitch(A, K, &arows);
    const long long q_ld = r3d_rows_pitch(Q, n, &qrows);
    const long long y_ld = OB ? r3d_rows_pitch(OB, n, &yrows) : n;
    if (a_ld < 0 || q_ld < 0 || y_ld < 0) return R3D_NO(RAD_E_STRIDE);
    /* `yrows` is only a promise when the plane is there. `r3d_rows_pitch` leaves the count at zero
     * for a null operand, and an absent optional output is a null operand by contract (r3d_args.h),
     * so testing `yrows` unconditionally reads "the row is missing" from "the caller asked for no
     * bf16 mirror". */
    if (arows < M || qrows < M || (OB && yrows < M)) return R3D_NO(RAD_E_SHAPE);
    if (a_ld % 8) return R3D_NO(RAD_E_STRIDE);            // 8-code / 8-element staging
    /* The scale planes travel without a pitch argument, so they must be dense, and `scale`'s grid
     * must be the one the kernel walks: `ng` scales a row. */
    if (!r3d_contig(B) || !r3d_contig(BS)) return R3D_NO(RAD_E_STRIDE);
    if (AS && !r3d_contig(AS)) return R3D_NO(RAD_E_STRIDE);
    if (r3d_rows_pitch(SC, ng, &srows) != ng) return R3D_NO(RAD_E_STRIDE);
    if (srows < M) return R3D_NO(RAD_E_SHAPE);
    if (r3d_dim(BS, 1) != 0 && BS->shape[BS->rank - 1] != nkb && r3d_numel(BS) != nb * nkb)
        return R3D_NO(RAD_E_SHAPE);
    if (!r3d_aligned(A->data, 16) || !r3d_aligned(B->data, 16)) return R3D_NO(RAD_E_ALIGN);

    R3dGemmCfg c;
    r3d_cfg_gemm(a, &c);
    /* Split-K is the axis this kernel cannot honour: a K slab produces a partial product, and the
     * epilogue needs the finished row -- its bf16 narrowing, the gate's partner column, the group's
     * amax. A caller that states sk > 1 is refused here, at the row, not answered slowly. */
    if (c.sk > 1) return R3D_NO(RAD_E_SHAPE);

    const int e = r3d_gemm_nt_q_gated_ex(
        r3d_addr(A), r3d_addr(AS), r3d_addr(B), r3d_addr(BS), r3d_addr(Q), r3d_addr(SC),
        r3d_addr(OB), (int)M, (int)N, (int)K, (int)group, act, (int)a_ld, (int)q_ld, (int)y_ld, 1,
        c.nt, 0L, (int)BS->dtype, (long)(uintptr_t)s);
    if (e == 0) return RAD_OK;
    return R3D_NO(e == -1 ? RAD_E_INVAL : e == -2 ? RAD_E_SHAPE
                        : e == -3 ? RAD_E_STRIDE : RAD_E_DEVICE);
}
