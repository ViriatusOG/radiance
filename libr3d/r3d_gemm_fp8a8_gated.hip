/* r3d_gemm_fp8a8_gated.hip -- gemm_nt_q_gated: the gated MLP's GEMM, gate and quantiser in one
 * launch, so the [M, N] bf16 intermediate is never materialised. On the 2B model it serves
 * (N=12288 K=2048 group=128 dtype=fp8a8 act=silu) 47 bands are M=64 and one is M=16.
 *
 *   y[M,n] = gate(a[M,K] @ b[N,K]^T), quantised:  n = N/2, gate half = b's rows [0, n),
 *   up half = [n, N). `q` is E4M3 codes over y, `scale` one f32 per (row, `group` of n), and
 *   `out_bf16`, when bound, is the bf16 the codes were taken over.
 *
 * Transcribed from ref_gemm_nt_q_gated, in its order:
 *
 *   row[j] = quant_dot(ma, mb, SA, SB, m, j, K)              over the full N, j ascending
 *   gt = round_bf16(row[i]),  up = round_bf16(row[n + i])    both halves, before the gate
 *   sg = 1 / (1 + exp(-gt))                                  the gate is the first half
 *   h  = f32_to_bf16((act == silu ? gt * sg : sg) * up)      the activation is not narrowed
 *   fp8_row(h, n, group)                                     amax -> scale[m, group], codes
 *
 * Both halves are narrowed before the gate because the unfused pair writes and re-reads a bf16
 * [M, N] tensor, and the fused form must be byte-identical to it; `out_bf16` is what that tensor
 * held. The activation is not narrowed: `gt * sg` is formed in f32 and the product with `up`
 * rounds once, at the store.
 *
 * ---- the lane map: one lane owns a pair of dots -----------------------------------------------
 *
 * The gate couples column i with column n + i, which for n = 6144 is 192 waves away: no lane
 * exchange reaches it, and staging the row through LDS would cost the round trip this op avoids.
 * So each lane computes both products of its column, walking k ascending with two accumulators over
 * weight rows `i` and `n + i`. The weight traffic equals the unfused GEMM's for the two columns.
 *
 * A block covers one group of `group` output columns and RP = 256 / group rows:
 *
 *     i = blockIdx.x * group + (tid % group)      m = blockIdx.y * RP + tid / group
 *
 * so for group >= 32 a wave's 32 lanes are 32 consecutive columns of one group of one row. The
 * epilogue therefore closes inside the block: the group's amax is a wave max folded over group/32
 * waves in LDS, the scale is one store, the codes one byte per lane. The op is one launch with no
 * scratch, so the row is concurrent (`rad_kernel_concurrent(i)` is `scratch == nullptr`).
 *
 * ---- the two arms: fp8a8 at M <= 8 reads the codes directly, the rest stage them --------------
 *
 * The staged shape below suits the bands' ceiling (M = 64) but not decode (M = 1, the whole
 * 12288 x 2048 operand, 42 calls per token). Its cost there is the path a code takes: global ->
 * LDS stage -> decode -> dot, with two `group_barrier`s per k-block, and an arithmetic decoder of
 * ten VALU per code (RDNA3 has no fp8 instructions; see r3d_codecs.h). The launcher picks the arm
 * by row count:
 *
 *   M <= 8 and fp8a8   read the activation and both weight rows directly, 16 bytes at a time,
 *                      decode them from a 256-entry table the block builds once, and multiply. No
 *                      shared memory between load and dot, no barrier in the k-loop, six loads in
 *                      flight.
 *   M >= 9, and every fp8a16 case   the staged shape, with the same table decoder. At these
 *                      heights the grid is 48 x M/2 blocks and occupancy feeds the stream.
 *
 * On the band's stream (24 MiB of codes, kernel loop without the epilogue) the staged arithmetic
 * path takes 497 us, the direct table path 187 us, and the bare loads 139 us. The table decoder
 * also cuts M = 64 by about two thirds at no register cost (109 per lane, four waves per SIMD, no
 * spills).
 *
 * Two invariants: the table builder's codec argument is a pair, not an index (build_e4m3_table),
 * and decoded pairs go from the table into the dot inline, never through an array (step_body).
 *
 * ---- where the scale multiplies -------------------------------------------------------------
 *
 * quant_dot, as in r3d_gemm_fp8a16_nt_m1.hip: per group of k a sub-sum from zero over ascending k
 * of the codes' products, then `acc = acc + (sub * sa) * sb` with the reference's parentheses, `sa`
 * being `a_scale[m][gi]` (1 without a scale plane). Each lane's two chains are the unfused pair's
 * two calls, so the bracketing matches element for element.
 *
 * b_scale is bf16: this op's shape hook writes `o.t(RAD_BF16, {nb, kb})` with no f32 form, so the
 * plain entry reads bf16 and the `_ex` entry takes the dtype, as for the fp8 GEMM. bf16 -> f32
 * widening is exact, so an f32 plane with the same values gives bit-identical output (the test
 * driver checks this).
 *
 * ---- the epilogue's one transcendental -------------------------------------------------------
 *
 * libref's sigmoid is `1.0f / (1.0f + (float)std::exp((double)(-x)))`. f64 runs at 1:64 on this
 * card, and the band needs M*n = 393216 exponentials per layer, so the kernel uses the accurate
 * device `expf` (not `__expf`), as r3d_gated_quant_fp8 (r3d_quant_fp8.hip) does. Its ~1e-7 error is
 * far under the bf16 ulp of 2^-8 the result is narrowed to, so the bf16 differs only on a tie; the
 * row's text records the tie count.
 *
 * ---- what the launcher refuses ----------------------------------------------------------------
 *
 * M > 64 (the bands' ceiling; beyond it the M reads of an [N, K] plane from L2 are a tiled
 * kernel's problem); N odd (the reference's `N & 1` test); a group that is not a power of two in
 * [32, 256] (a wave must sit inside one group, and the block is one group wide); group > n, where
 * the reference would use n for the quantiser's group but keep the stated K block, two widths this
 * kernel reads as one; K % group != 0, because a ragged K block reads past the scale plane; sk > 1,
 * because the epilogue needs the whole row. Codes: -1 null, -2 shape, -3 pitch, -4 HIP.
 *
 * ---- no relayout ------------------------------------------------------------------------------
 *
 * This row reads the gated weight [N, K] row-major, as libref does. libr4d stores fp8a8 weights in
 * WMMA fragment order (`r4d.fp8.w.frag64`); the two readings differ by a relative error of ~1.35,
 * so a row that inherited the fp8 GEMM's relayout would fail the per-op oracle. The stored order
 * is the reading order, as for every row in the library.
 */

#include <hip/hip_runtime.h>          // __forceinline__, which r3d_codecs.h's R3D_HD uses

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <algorithm>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWaves        = kBlockThreads / kWave;
constexpr int kStep         = 16;     // codes per lane step: one 16-byte load
constexpr int kGroupMin     = 32;     // a wave's 32 columns must sit inside one group
constexpr int kGroupMax     = 256;    // a block is one group wide, at 256 threads
constexpr int kMMax         = 64;     // the bands' ceiling
constexpr float kTopE4M3    = 448.0f;  // q8_group's top for E4M3: the largest finite code

/* The E4M3 decoder as a table: 256 f16, one entry per code. The codec's arithmetic decode is ten
 * VALU per code (r3d_codecs.h), and this kernel decodes every weight code it multiplies (twice at
 * RP = 2). The table costs two `ds_read_u16` and a pack per pair, and is exact: entry c is what the
 * arithmetic path returns for c, computed by that path at block start. The unit test reads all 256
 * entries back and compares them with the codec. */
__device__ __forceinline__ void build_e4m3_table(uint16_t* tab, uint32_t tid)
{
    for (uint32_t u = tid; u < 128; u += kBlockThreads)
        /* The codec's argument is a pair, low code in the low byte; the entry is indexed by code.
         * Passing the index alone would decode code 0 into every odd entry and silently zero half
         * the products (the `codes256` test case catches this). */
        *(uint32_t*)&tab[2 * u] = e4m3x2_to_f16x2((u * 2) | ((u * 2 + 1) << 8));
}

__device__ __forceinline__ uint32_t tab_pair(const uint16_t* tab, uint32_t pair)
{
    const uint32_t lo = *(const uint16_t __attribute__((address_space(3))) *)(tab + (pair & 0xffu));
    const uint32_t hi = *(const uint16_t __attribute__((address_space(3))) *)(tab + (pair >> 8));
    return lo | (hi << 16);
}

/* One step of the skinny arm: sixteen activation codes and sixteen weight codes, both decoded from
 * the same table as each dot consumes them. The reference's `sub += a[k] * b[k]`, k ascending: four
 * dot4, pairs in order, one accumulator. */
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

/* One step's sixteen weight codes into the accumulator: four dot4, pairs decoded from the table as
 * each dot consumes them. Materialising the eight pairs in an array first lets the scheduler hoist
 * all sixteen table reads, keeping eight decoded pairs live instead of two, which spills against
 * the 128-register budget of a 256-thread block at four waves per SIMD. Pairs ascend into one
 * accumulator. */
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

/* q8_group's clamp, with q8_group's comparisons rather than fminf/fmaxf, which would replace a NaN
 * with the bound; the reference encodes a NaN activation as the format's NaN, not +-top. */
__device__ __forceinline__ float clamp_top(float c)
{
    if (c > kTopE4M3) c = kTopE4M3;
    if (c < -kTopE4M3) c = -kTopE4M3;
    return c;
}

/* y[M,n] = gate(a @ b^T), with the quantiser. GROUP is at once the block's width in output columns,
 * the weight's K block and the quantiser's group: the op states one `group` key, and kb, nb and ng
 * derive from it. ACT is the gate, A8 the coded activation, NT the weight load's lowering, WIN the
 * skinny (direct-read) arm.
 *
 * The tall arm stages the activation in LDS per K block, widened exactly to f16 pairs. A block
 * stages RP * GROUP values (512 bytes at every supported GROUP). All 32 lanes of a wave share one
 * m, so a wave reads its row's 16 values per step as one broadcast ds_read. The weight is not
 * staged: each lane streams two rows no other lane reads. */
template <int GROUP, int ACT, bool A8, bool NT, bool WIN>
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(WIN ? 3 : 8) void gated_kernel(
    const uint8_t* __restrict__ A, const float* __restrict__ AScale,
    const uint8_t* __restrict__ B, const void* __restrict__ BScale, int bs16,
    uint8_t* __restrict__ Q, float* __restrict__ S, uint16_t* __restrict__ Y,
    int M, int N, int K, int a_ld, int q_ld, int y_ld, int nkb, int ng)
{
    constexpr int RP = kBlockThreads / GROUP;   // rows per block
    constexpr int AP = GROUP / 2;               // dwords per staged activation row (f16 pairs)
    constexpr int WPG = GROUP / kWave;          // waves covering one group of columns
    static_assert(RP >= 1 && WPG >= 1 && RP * WPG == kWaves, "a block is one group wide");

    __shared__ uint32_t s_a[RP * AP];
    __shared__ float s_mx[kWaves];
    __shared__ uint16_t s_tab[256];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;

    /* The block's decoder table, filled before any lane reads it: 128 dwords, one per pair of
     * codes, 128 arithmetic decodes per block. */
    build_e4m3_table(s_tab, tid);
    group_barrier();

    const uint32_t gidx = blockIdx.x;                     // the group of output columns
    const uint32_t mloc = tid / GROUP;                    // this lane's row within the block
    const uint32_t i    = gidx * GROUP + (tid % GROUP);   // the fused column
    const uint32_t m    = (uint32_t)blockIdx.y * RP + mloc;

    const int n      = N / 2;
    /* A ragged last group runs past n: the pair's rows are clamped so no lane reads past the weight
     * plane, and its stores are guarded below. A row past M re-reads the last real row. */
    const int i_real = (int)(i < (uint32_t)n ? i : (uint32_t)(n - 1));
    const int m_real = (int)(m < (uint32_t)M ? m : (uint32_t)(M - 1));

    const uint8_t* wg = B + (size_t)i_real * K;           // the gate's row
    const uint8_t* wu = B + (size_t)(n + i_real) * K;     // the up's row, n rows further on
    /* b_scale is [ceil(N/group)][ceil(K/group)] (make_scales uses rblk = group), so a row's scale
     * row is its row index divided by the block. */
    const size_t bsg = (size_t)((uint32_t)i_real / GROUP) * (size_t)nkb;
    const size_t bsu = (size_t)((uint32_t)(n + i_real) / GROUP) * (size_t)nkb;
    const float* bsF  = bs16 ? nullptr : (const float*)BScale;
    const uint16_t* bsH = bs16 ? (const uint16_t*)BScale : nullptr;
    const float* as = A8 ? AScale + (size_t)m_real * nkb : nullptr;

    float acc_g = 0.0f, acc_u = 0.0f;

    /* One scale block per iteration, `nkb` of them: stage the block's activation, walk its k with
     * two chains, apply its two scales at the end. The launcher requires K % GROUP == 0, so every
     * block is whole and staging has no tail. */
    /* The skinny arm's activation row, in bytes. */
    const uint8_t* ar = (A8 && WIN) ? (const uint8_t*)A + (size_t)m_real * a_ld : nullptr;

    for (int kb = 0; kb < K; kb += GROUP) {
        const int gi = kb / GROUP;
        /* Staging and its barrier belong to the tall arm. At the skinny height (M <= 8, at most
         * 16 KiB of activation at the band's K) all 48 column groups read the same activation
         * slice, so it stays in L1 and staging would only add a barrier per k-block. */
        if constexpr (!WIN) {
        constexpr int kUnit = 8;                       // codes (fp8a8) or elements (bf16) per unit
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
                /* Cast to the element pointer first: the row and k offsets are bf16 element
                 * offsets. */
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
        /* The two chains, gate into `sub_g` and up into `sub_u`, each k ascending: the unfused
         * pair's two calls, each with that call's bracketing.
         *
         * The skinny arm runs the chains one after the other rather than interleaved: holding
         * both streams' windows live (64 registers of loads, on top of everything else) spills at
         * 256 threads (four waves per SIMD, 128 registers per lane), while one chain's window is
         * 32 registers and does not. */
        float sub_g, sub_u;
        if constexpr (WIN) {
            /* The skinny arm, M <= 8 (selected by `R3D_GT_G`): load the codes, decode them from
             * the block's table and multiply, with no shared memory between load and dot and no
             * barrier in the k-loop. Each step loads 16 bytes of activation and 16 of weight;
             * prefetching two steps ahead keeps loads in flight, where the tall arm relies on
             * occupancy (32 waves per CU). */
            constexpr int kNS = GROUP / kStep;                // steps per block
            constexpr int kD  = 2;                            // loads in flight per chain
            auto chain = [&](const uint8_t* w, float sub) {
                auto ld = [&](int st) {
                    return ldc<NT>(w + kb + (st < kNS ? st : kNS - 1) * kStep);
                };
                /* Named registers, not an array: an array indexed across iterations, even with a
                 * constant trip count and unrolling, is placed in private (scratch) memory. Two
                 * named 16-byte registers, prefetched two steps ahead, keep the frame at zero. */
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
            /* The tall arm: one 16-code step per iteration, both streams in the same iteration.
             * At M >= 9 the grid is 48 x M/2 blocks and occupancy feeds the stream (109 registers
             * per lane, four waves per SIMD, 32 waves per CU). A load window would cost 32
             * registers per lane here, and at M = 64 it costs more than it saves. */
            sub_g = 0.0f;
            sub_u = 0.0f;
            /* Unroll depth is a register decision. At group = 32 the block is two steps wide, so
             * `unroll 2` is a full unroll: the scheduler hoists all 32 table reads of both chains
             * above the dots and the arm spills at 128 registers. group = 32 therefore stays a
             * one-step loop; group >= 64 unrolls by 2 without spilling. */
            /* Do not materialise the eight decoded pairs into an array on the way to the dot:
             * that form miscomputes on this arm (rel_l2 6.6e-01 at M = 7, NaN at M = 64) although
             * the table itself is exact. Keep the decode inline. */
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
            group_barrier();             // the next block's stage must not race these reads

        /* The two sub-sums, each scaled once, in libref's parentheses. */
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
    /* A column past n or a row past M contributes 0.0f to the amax and is not stored, as in
     * quant_act_fp8 for a ragged row. */
    const float am = keep ? fabsf(bf16_to_f32(h)) : 0.0f;

    float mx = wave_max(am);             // fmaxf skips a NaN, as `if (m > amax)` does
    if (lane == 0) s_mx[wave] = mx;
    group_barrier();
    mx = s_mx[mloc * WPG];
#pragma unroll
    for (int w = 1; w < WPG; ++w) mx = fmaxf(mx, s_mx[mloc * WPG + w]);

    const float sc = mx > 0.0f ? mx * (1.0f / kTopE4M3) : 1.0f;
    /* One store per (row, group): the group's WPG waves hold the same max and the first writes it.
     * `scale` is dense (the launcher takes no pitch for it): `ng` scales per row, indexed by the
     * block's group. */
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
 *   q         [M][n]                E4M3 codes, `q_ld` elements per row (0 = n)
 *   scale     [M][ceil(n/group)]    f32, dense (no pitch is taken)
 *   out_bf16  [M][n]                bf16, `y_ld` elements per row (0 = n), optional: the gated
 *                                   bf16 tensor the codes were taken over
 *
 * `group` is at once the weight's K block, the weight's row block and the output quantiser's group,
 * as the op's single key states; 0 takes the checkpoint's 128. `act` is R3D_ACT_SILU (down_proj's
 * input) or R3D_ACT_SIGMOID (a gated attention o_proj's). `sk` is accepted as 0/1 and refused above
 * it: the epilogue needs a whole row. `nt` selects the weight load's lowering. `scratch` is unused;
 * needing no arena keeps the row concurrent.
 *
 * Returns 0, -1 a null pointer, -2 an unsupported shape, -3 an unreadable pitch, -4 HIP failed. */
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

/* fp8a8 at M <= 8 takes the skinny (direct-read) arm, everything else the staged arm.
 *
 * The threshold is a tunable chosen for decode (M = 1) and draft verify (M <= 8). With the table
 * decoder in both arms, at N=12288 K=2048 group=128 the skinny arm takes 402 us against 754 at
 * M = 8 and 955 against 1030 at M = 16, so the crossover lies above 16 (M = 9..15 unmeasured);
 * raising the threshold to 16 would gain about 7% for a prefill of 9..16 rows. */
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

/* The entry without a scale dtype reads bf16, the only form this op's shape hook states. A caller
 * with an f32 scale plane (the `fp8a8f32` reading of the unfused GEMM) uses `_ex` with RAD_F32;
 * bf16 widens exactly, so the two agree bit for bit. */
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
 * M, N, K, group, dtype, act (libref's pGemmQGated/oGemmQGated: `act` and `a_scale` required,
 * `out_bf16` optional). The shape hook declines every scheme but fp8a8, so the row binds fp8a8 and
 * the shim reads the coded activation from the operands. */
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
    /* The activation: E4M3 codes with a per-(row, group) f32 scale (fp8a8, the scheme every band
     * states), or bf16 with no scale (fp8a16, which the shape hook declines, so no band reaches it;
     * the launcher supports it and the shim accepts it). E5M2 has no decoder here. */
    const bool a8 = AS != nullptr;
    if (a8) {
        if (!r3d_is_e4m3(A)) return R3D_NO(RAD_E_DTYPE);
        R3D_NEED(dt_f32(AS));
    } else {
        R3D_NEED(dt_bf16(A));
    }
    /* b_scale is bf16 (the hook's `o.t(RAD_BF16, {nb, kb})`, the container's `weight_scale_inv`).
     * f32 is also accepted; the widening is exact, so both give the same output. */
    const bool bs16 = r3d_is_bf16(BS);
    if (!bs16 && !r3d_is_f32(BS)) return R3D_NO(RAD_E_DTYPE);

    /* `act` is required by the schema (P_STR, not P_STRO), because an implementation may compile
     * only one gate. libref defaults it to -1 and refuses that; so does this shim, with no integer
     * `mode` fallback. */
    const int act = r3d_act_code(a, "act", nullptr, -1);
    if (act != R3D_ACT_SILU && act != R3D_ACT_SIGMOID) return R3D_NO(RAD_E_INVAL);

    const long long K = r3d_geti_or(a, "K", r3d_dim(B, (int)B->rank - 1));
    const long long N = r3d_geti_or(a, "N", r3d_dim(B, (int)B->rank - 2));
    if (K < 1 || N < 2 || (N & 1)) return R3D_NO(RAD_E_SHAPE);
    const long long n = N / 2;

    /* Rows come from `q`, never from the `M` parameter (the band's upper bound). libref reads
     * numel(q) / n, which on a plane with pitch > n counts the padding, so the shape's row count
     * clamps it. Dense planes give the same number either way. */
    const long long qnel = r3d_numel(Q);
    if (qnel < n || qnel % n) return R3D_NO(RAD_E_SHAPE);
    long long M = qnel / n;
    if (Q->rank >= 2 && r3d_dim(Q, (int)Q->rank - 2) < M) M = r3d_dim(Q, (int)Q->rank - 2);
    if (M < 1 || M > kMMax) return R3D_NO(RAD_E_SHAPE);

    /* The group: the weight's K block, row block and the quantiser's group, from the op's one key.
     * The reference would use n for the quantiser alone when group > n; this kernel reads one
     * width, so such a group is refused. */
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
    /* `yrows` is only meaningful when the plane exists: `r3d_rows_pitch` leaves it zero for a null
     * operand, and an absent optional output is a null operand (r3d_args.h). */
    if (arows < M || qrows < M || (OB && yrows < M)) return R3D_NO(RAD_E_SHAPE);
    if (a_ld % 8) return R3D_NO(RAD_E_STRIDE);            // 8-code / 8-element staging
    /* The scale planes take no pitch argument, so they must be dense, and `scale` must hold `ng`
     * scales per row. */
    if (!r3d_contig(B) || !r3d_contig(BS)) return R3D_NO(RAD_E_STRIDE);
    if (AS && !r3d_contig(AS)) return R3D_NO(RAD_E_STRIDE);
    if (r3d_rows_pitch(SC, ng, &srows) != ng) return R3D_NO(RAD_E_STRIDE);
    if (srows < M) return R3D_NO(RAD_E_SHAPE);
    if (r3d_dim(BS, 1) != 0 && BS->shape[BS->rank - 1] != nkb && r3d_numel(BS) != nb * nkb)
        return R3D_NO(RAD_E_SHAPE);
    if (!r3d_aligned(A->data, 16) || !r3d_aligned(B->data, 16)) return R3D_NO(RAD_E_ALIGN);

    R3dGemmCfg c;
    r3d_cfg_gemm(a, &c);
    /* No split-K: the epilogue needs the finished row (its bf16 narrowing, the gate's partner
     * column, the group's amax), so sk > 1 is refused. */
    if (c.sk > 1) return R3D_NO(RAD_E_SHAPE);

    const int e = r3d_gemm_nt_q_gated_ex(
        r3d_addr(A), r3d_addr(AS), r3d_addr(B), r3d_addr(BS), r3d_addr(Q), r3d_addr(SC),
        r3d_addr(OB), (int)M, (int)N, (int)K, (int)group, act, (int)a_ld, (int)q_ld, (int)y_ld, 1,
        c.nt, 0L, (int)BS->dtype, (long)(uintptr_t)s);
    if (e == 0) return RAD_OK;
    return R3D_NO(e == -1 ? RAD_E_INVAL : e == -2 ? RAD_E_SHAPE
                        : e == -3 ? RAD_E_STRIDE : RAD_E_DEVICE);
}
