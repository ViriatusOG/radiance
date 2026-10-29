/* r3d_gemm_fp8a16_nt_m1.hip -- the same skinny shape against a block-scaled weight.
 *
 *   y[M,N] = a[M,K] @ b[N,K]^T, b stored as E4M3 codes with one f32 scale per (pb1 x pb2) block,
 *   f32 accumulate, narrowed once by the store. `a_scale`, when present, means the activation is
 *   coded too -- quant_act_fp8's E4M3 codes with one f32 per (row, group) -- which is the fp8a8
 *   form the row table names `fp8a8`.
 *
 * RDNA3 has no fp8 instruction: the builtins are declared for gfx11 and die at instruction
 * selection. So the codes are decoded in software, and the decode is free of cost in the sense
 * that matters here: E4M3 is four exponent and three mantissa bits, f16 is five and ten, so
 * e4m3x2_to_f16x2 is EXACT and the dot2 multiplies the numbers the checkpoint means. Nothing is
 * rounded between the code and the product, which is why this row's output is libref's to the bit
 * at sk = 1 and not merely inside fp8 quantisation noise.
 *
 * ---- the lane map --------------------------------------------------------------------------
 *
 * The same as r3d_gemm_bf16_nt_m16's: lane l owns the output (m = l % MT, n = n0 + l / MT) for the
 * template's MT = ceil(M) in {1,2,4,8,16}, walks k ascending and holds its own accumulator. One
 * difference is forced by the format, and it is the one that makes this kernel what it is: a lane
 * consumes sixteen codes per step, because that is what sixteen bytes are, and sixteen bytes is
 * one global_load_b128. Eight dot2 per step, so the weight stream is one 16-byte load per eight
 * matrix instructions -- the decode (eight v_perms and shifts, exact) sits between them and is the
 * reason this kernel's rate is not the bf16 row's.
 *
 * A lane owns one (m, n) pair, so a lane owns one scale per block: `n / pb1` picks b_scale's row
 * and the block index picks its column, and both are hoisted out of the block's sixteen-element
 * steps. The activation's scale, when the activation is coded, is `m`'s row at the same block
 * index. Lanes sharing an n agree on it, which is why no lane-to-lane exchange is needed.
 *
 * ---- where the scale multiplies -------------------------------------------------------------
 *
 * ref_gemm.h's quant_tile, and quant_dot beside it, is the spec, and the order it writes is the
 * order this kernel runs:
 *
 *     for gi ascending:  sub = 0
 *                        for k in the group, ascending:  sub += a[k] * b[k]     (from the CODES)
 *                        acc = acc + (sub * sa) * sb
 *
 * with sa = a_scale[m][gi] or 1, and sb = b_scale[n / pb1][gi]. The scale multiplies the finished
 * block, once, at the block's end -- not every product: scaling each product is equal in exact
 * arithmetic and differs in the last bits, which moves an fp8 code and so moves a token. And
 * `(sub * sa) * sb` is an order, not a formula: the parentheses are libref's, and the products are
 * summed from the raw codes with no scale folded into them. This kernel accumulates a block's
 * sub-sum into a register of its own, so the bracketing is the reference's, and the widening is
 * exact, so at sk = 1 the output is bit-identical to it.
 *
 * ---- the activation ---------------------------------------------------------------------------
 *
 * Staged in LDS per block of the weight's K, widened on the way in: bf16 -> f16 for fp8a16, E4M3 ->
 * f16 for fp8a8, both exact. One row of a is M <= 16 elements wide per k, so a staged block is
 * MT * 128 values, 4.25 KiB at MT = 16, and the pitch is 68 dwords: 68 mod 32 = 4, so a
 * ds_read_b128's eight lanes land on banks four apart, and 272 bytes a row keeps the 16-byte reads
 * aligned. The weight is not staged: with M <= 16 a staged tile has 32/MT readers, so staging
 * trades a wave's worth of L1 residency for a round trip through LDS and buys nothing.
 *
 * ---- split-K --------------------------------------------------------------------------------
 *
 * `sk` contiguous slices, one per blockIdx.y, each writing its f32 partial to `scratch` and a
 * second kernel adding them per output in ascending slice order. Each partial is a chain of
 * `(sub * sa) * sb` terms over its own slices of blocks, so at sk = 1 the chain is libref's
 * exactly; sk > 1 brackets the block terms by slice, which is the split's doing and agrees to the
 * last bits. Legality is K % (sk * group) == 0 -- a slice that ends mid-block cannot apply that
 * block's scale -- which is the row's gemm_tune_valid rule, unit = group.
 *
 * ---- `nt` -----------------------------------------------------------------------------------
 *
 * No transposed load on this ISA and no cache field on a global load. nt = 1 issues the weight
 * reads through an address-space-1 pointer, which lowers to global_load_b128 with no cache
 * modifier and no swizzle; nt = 0 issues the same 16-byte reads through the generic pointer. The
 * axis measures the two lowerings, and on this card it measures the same stream twice.
 */

#include <hip/hip_runtime.h>          // __forceinline__, which r3d_codecs.h's R3D_HD spells

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <algorithm>

namespace {

constexpr int kBlockThreads = 256;
constexpr int kWaves        = kBlockThreads / kWave;
constexpr int kMTMax        = 16;
constexpr int kStep         = 16;      // codes a lane takes per step: one 16-byte load
constexpr int kAPitch       = 68;      // dwords a staged activation row (128 values + 4)
constexpr int kGroupMax     = 128;     // the stage is sized for one 128-wide block per row

/* A slice is legal when it is a whole number of scale blocks, so the split ladder is capped by
 * the scratch hook's widest split and stepped by the block the weight carries. */
int pick_sk(long long K, int group, int maxsk)
{
    static const int kLadder[] = { 16, 8, 4, 2, 1 };
    for (int s : kLadder)
        if (s <= maxsk && K % ((long long)s * group) == 0) return s;
    return 1;
}

template <bool NT>
__device__ __forceinline__ r3d_u32x4 ldc(const uint8_t* p)      // 16 codes
{
    if constexpr (NT) return *(const __attribute__((address_space(1))) r3d_u32x4*)p;
    else return __builtin_bit_cast(r3d_u32x4, *(const uint4*)p);
}

__device__ __forceinline__ r3d_u32x4 lds_a(const uint32_t* p)
{
    return *(const r3d_u32x4*)p;
}

__device__ __forceinline__ uint32_t ldg_u8(const uint8_t* p)
{
    return *(const __attribute__((address_space(1))) uint8_t*)p;
}

/* The weight's 16 codes as eight f16 pairs, exact.
 *
 * e4m3x2_to_f16x2 decodes the TWO codes in a dword's low half, so a 16-byte load -- four dwords,
 * sixteen codes -- is eight decodes, each fed the 16-bit pair it wants: the pairs straddle the
 * dwords, four dwords in, eight pairs out. Feeding the codec whole dwords decodes eight codes and
 * leaves the rest of the step multiplying registers that were never written, which is the quiet
 * version of this bug: the answer comes back a fraction of its size, not a NaN. */
__device__ __forceinline__ void codes_to_f16x8(const r3d_u32x4 c, uint32_t* w)
{
    const uint32_t d[4] = { c.x, c.y, c.z, c.w };
#pragma unroll
    for (int j = 0; j < 8; ++j)
        w[j] = e4m3x2_to_f16x2((d[j >> 1] >> (16 * (j & 1))) & 0xffffu);
}

/* y[M,N] = a[M,K] @ dequant(b)[N,K]^T. `A8` is the fp8a8 form: a carries E4M3 codes and a_scale
 * carries one f32 per (row, block); without it a is bf16 and read directly. */
template <int MT, bool A8, bool NT, bool SPLIT>
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemm_q_kernel(
    const uint8_t* __restrict__ A, const float* __restrict__ AScale,
    const uint8_t* __restrict__ B, const void* __restrict__ BScale, int bs16,
    uint16_t* __restrict__ Y, float* __restrict__ Part,
    int M, int N, int K, int group, int pb1, int a_ld, int y_ld, int kslab, int nkb)
{
    __shared__ uint32_t s_a[kMTMax * kAPitch];

    const uint32_t tid   = threadIdx.x;
    const uint32_t lane  = tid & (kWave - 1);
    const uint32_t wave  = tid / kWave;
    const uint32_t waves = blockDim.x / kWave;
    constexpr uint32_t kNT = kWave / MT;

    const uint32_t n0  = ((uint32_t)blockIdx.x * waves + wave) * kNT + lane / MT;
    const uint32_t m   = lane & (MT - 1);
    const int n_real = (int)(n0 < (uint32_t)N ? n0 : (uint32_t)(N - 1));
    const int m_real = (int)(m < (uint32_t)M ? m : (uint32_t)(M - 1));

    const int kbeg = (int)blockIdx.y * kslab;
    const int kend = kbeg + kslab < K ? kbeg + kslab : K;

    const uint8_t* w = B + (size_t)n_real * K;
    /* b_scale is the container's `weight_scale_inv`, and the op's shape hook writes it bf16 for the
     * whole quantised family -- `o.t(f32s ? RAD_F32 : RAD_BF16, {nb, kb})`; f32 belongs to the
     * separate `fp8a8f32` spelling. Both are read here; bf16 widens to f32 exactly, since every
     * bf16 value is representable in f32, so the widening is free and the products are unchanged. */
    const size_t bsrow = (size_t)(n_real / pb1) * nkb;
    const float* bsF = bs16 ? nullptr : (const float*)BScale + bsrow;
    const uint16_t* bsH = bs16 ? (const uint16_t*)BScale + bsrow : nullptr;
    const float* as = A8 ? AScale + (size_t)m_real * nkb : nullptr;

    float acc = 0.0f;

    /* One scale block per round: stage the block's activation, walk the block's k, fold the block's
     * scale in at the end. `kb` is the block's first k, which is where the scales are indexed. */
    for (int kb = kbeg; kb < kend; kb += group) {
        const int gl = std::min(group, kend - kb);
        const int gi = kb / group;

        /* Stage gl elements of MT activation rows, widened to f16 pairs. Eight values a unit: a
         * 16-byte load for a bf16 row, an 8-byte one for a coded row, which is why the coded row's
         * pitch check is eight, not sixteen. A row past M re-reads the last real row. */
        constexpr int kUnit = 8;
        const int nu = (gl + kUnit - 1) / kUnit;
        for (int i = (int)tid; i < MT * nu; i += (int)blockDim.x) {   // see the m16 unit's note on wv
            const int r = i / nu, c = (i % nu) * kUnit;
            const int rr = r < M ? r : M - 1;
            const int cnt = std::min(kUnit, gl - c);
            uint32_t* dst = s_a + (size_t)r * kAPitch + c / 2;
            if constexpr (A8) {
                const uint8_t* src = A + (size_t)rr * a_ld + kb + c;      // codes: bytes and elements agree
                for (int e = 0; e < cnt; e += 8) {
                    const uint8_t* sp = src + e;
                    uint32_t q[2];
                    if (cnt - e >= 8) {
                        const r3d_u32x2 b8 =
                            *(const __attribute__((address_space(1))) r3d_u32x2*)sp;
                        q[0] = b8.x; q[1] = b8.y;
                    } else {
                        uint32_t pack[2] = { 0, 0 };
                        for (int t = 0; t < cnt - e; ++t)
                            pack[t >> 2] |= (uint32_t)ldg_u8(sp + t) << (8 * (t & 3));
                        q[0] = pack[0]; q[1] = pack[1];
                    }
                    uint32_t p[4] = { e4m3x2_to_f16x2(q[0] & 0xffffu),
                                      e4m3x2_to_f16x2(q[0] >> 16),
                                      e4m3x2_to_f16x2(q[1] & 0xffffu),
                                      e4m3x2_to_f16x2(q[1] >> 16) };
#pragma unroll
                    for (int t = 0; t < 4; ++t)
                        if (e + 2 * t < gl) dst[e / 2 + t] = p[t];
                }
            } else {
                /* A is a byte pointer here because the coded form codes it: cast first, then the
                 * row and k offsets are ELEMENT offsets, which for bf16 is twice the byte count. */
                const uint16_t* src = (const uint16_t*)A + (size_t)rr * a_ld + kb + c;
                for (int e = 0; e < cnt; e += 8) {
                    uint32_t p[4];
                    if (cnt - e >= 8) {
                        const r3d_u32x4 b16 =
                            *(const __attribute__((address_space(1))) r3d_u32x4*)(src + e);
                        p[0] = bf16x2_to_f16x2(b16.x);
                        p[1] = bf16x2_to_f16x2(b16.y);
                        p[2] = bf16x2_to_f16x2(b16.z);
                        p[3] = bf16x2_to_f16x2(b16.w);
                    } else {
#pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            const uint32_t lo = ldg_u16(src + e + 2 * t);
                            const uint32_t hi = (e + 2 * t + 1 < gl)
                                                    ? ldg_u16(src + e + 2 * t + 1) : 0u;
                            p[t] = bf16x2_to_f16x2(lo | (hi << 16));
                        }
                    }
#pragma unroll
                    for (int t = 0; t < 4; ++t)
                        if (e + 2 * t < gl) dst[e / 2 + t] = p[t];
                }
            }
        }
        group_barrier();

        const uint32_t* ap = s_a + (size_t)m * kAPitch;
        float sub = 0.0f;
        const int gf = (gl / kStep) * kStep;
        for (int k = 0; k < gf; k += kStep) {
            const r3d_u32x4 a0 = lds_a(ap + k / 2);
            const r3d_u32x4 a1 = lds_a(ap + k / 2 + 4);
            uint32_t b[8];
            codes_to_f16x8(ldc<NT>(w + kb + k), b);
            sub = dot4_f16x4(a0.x, a0.y, b[0], b[1], sub);
            sub = dot4_f16x4(a0.z, a0.w, b[2], b[3], sub);
            sub = dot4_f16x4(a1.x, a1.y, b[4], b[5], sub);
            sub = dot4_f16x4(a1.z, a1.w, b[6], b[7], sub);
        }
        /* The last partial block, when K is not a whole block: ascending, two k at a time, each
         * code decoded on its own. The launcher refuses a K that is not a multiple of the block
         * (a ragged block would read past the scale plane), so this is a shape the kernel carries
         * for completeness and the row never issues. */
        for (int k = gf; k < gl; k += 2) {
            const uint32_t aw = ap[k / 2];
            const uint32_t c0 = ldg_u8(w + kb + k);
            const uint32_t c1 = (k + 1 < gl) ? ldg_u8(w + kb + k + 1) : 0u;
            const uint32_t pw = e4m3x2_to_f16x2(c0 | (c1 << 8));
            sub = dot2_f16x2(aw, pw, sub);
        }
        group_barrier();                 // the next block's stage must not race this one's reads

        /* The block's finished sub-sum, scaled once, in libref's parentheses. */
        const float sa = A8 ? as[gi] : 1.0f;
        acc = acc + (sub * sa) * (bs16 ? bf16_to_f32(bsH[gi]) : bsF[gi]);
    }

    const bool keep = (n0 < (uint32_t)N) && (m < (uint32_t)M);
    if (!keep) return;
    if constexpr (SPLIT) {
        Part[((size_t)blockIdx.y * M + m) * N + n0] = acc;
    } else {
        Y[(size_t)m * y_ld + n0] = f32_to_bf16(acc);
    }
}

/* The split reduction, the m16 unit's, in the same order: v = ((p0 + p1) + p2) + ... ascending by
 * slice, then one narrowing. Ascending slice order is libref's ascending sum only at sk = 1; past
 * that the block terms are bracketed by slice, which is the split's cost and is stated as such. */
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemm_q_reduce(
    const float* __restrict__ Part, uint16_t* __restrict__ Y, int M, int N, int y_ld, int sk)
{
    const int mn = M * N;
    const int stride = gridDim.x * kBlockThreads;
    for (int i = blockIdx.x * kBlockThreads + threadIdx.x; i < mn; i += stride) {
        const int m = i / N, n = i - m * N;
        const float* p = Part + (size_t)m * N + n;
        float v = p[0];
        for (int s = 1; s < sk; ++s) v = v + p[(size_t)s * M * N];
        Y[(size_t)m * y_ld + n] = f32_to_bf16(v);
    }
}

}  // namespace

/* y[M,N] = a[M,K] @ dequant(b)[N,K]^T over E4M3 codes, M <= 16, f32 accumulate, bf16 out.
 *
 *   a        [M][K]        bf16 (fp8a16) or E4M3 codes (fp8a8, when `a_scale` is present)
 *   a_scale  [M][ceil(K/group)]  f32, or 0
 *   b        [N][K]        E4M3 codes, row-major, k contiguous
 *   b_scale  [ceil(N/pb1)][ceil(K/pb2)]  bf16 (the container's weight_scale_inv) or f32, block-major
 *   y        [M][N]        bf16, `y_ld` elements a row (0 = N)
 *
 * `group` is the code group along K (pb2, the checkpoint's block width); `pb1` the rows sharing a
 * scale row (0 = group). `sk` splits K (0 = the widest legal), `nt` the weight load's lowering,
 * `scratch` the f32 partial arena, sk * M * N * 4 bytes, required when the split is on. Returns 0,
 * -1 a null pointer, -2 a shape it does not serve, -3 a pitch it cannot read, -4 HIP failed. */
extern "C" int r3d_gemm_nt_q_fp8_ex(long a, long a_scale, long b, long b_scale, long y, int M,
                                    int N, int K, int group, int pb1, int pb2, int a_ld, int y_ld,
                                    int sk, int nt, long scratch, int bs_dt, long stream)
{
    if (!a || !b || !b_scale || !y) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (M > kMTMax) return -2;
    if (group <= 0) group = R3D_FP8_BLOCK;
    if (pb1 <= 0) pb1 = group;
    if (pb2 == 0) pb2 = group;
    if (pb2 != group) return -2;                    // one block: the K block IS the code group
    if (group > kGroupMax) return -2;
    if (group % kStep) return -2;                   // a block must be whole lane steps
    if (K % group) return -2;                       // a ragged block reads past the scale plane
    const int nkb = K / group;
    if (!a_ld) a_ld = K;
    if (!y_ld) y_ld = N;
    const int a8 = a_scale != 0;
    if (a8 && (a_ld % 8)) return -3;                // 8-byte code staging
    if (!a8 && (a_ld % 8)) return -3;               // 16-byte widened staging
    if (sk == 0) sk = pick_sk(K, group, 16);
    if (sk < 1) return -2;
    if (K % ((long long)sk * group)) return -2;
    if (sk > 1 && !scratch) return -1;

    const int bs16 = (bs_dt == RAD_BF16);      /* anything else, 0 included, reads the plane as f32 */
    const int mt = M <= 1 ? 1 : M <= 2 ? 2 : M <= 4 ? 4 : M <= 8 ? 8 : kMTMax;
    const int kslab = K / sk;
    const int nrowb = (N + pb1 - 1) / pb1;
    if ((long long)nrowb * nkb <= 0) return -2;

#define R3D_Q_NT(MTV, A8V, NTV)                                                                  \
    do {                                                                                         \
        const int nt_ = (N + (kWave / MTV) - 1) / (kWave / MTV);                                 \
        const dim3 grid((nt_ + kWaves - 1) / kWaves, (unsigned)sk), dim3_blk(kBlockThreads);      \
        if (sk > 1) {                                                                            \
            R3D_KLAUNCH(R3D_KNAME((gemm_q_kernel<MTV, A8V, NTV, true>)), grid, dim3_blk,         \
                        (hipStream_t)stream, (const uint8_t*)a, (const float*)a_scale,           \
                        (const uint8_t*)b, (const void*)b_scale, bs16, (uint16_t*)y,        \
                        (float*)scratch, M, N, K, group, pb1, a_ld, y_ld, kslab, nkb);           \
            const int mn = M * N;                                                                \
            const int nb = std::min((mn + kBlockThreads - 1) / kBlockThreads, 4096);             \
            R3D_KLAUNCH(gemm_q_reduce, dim3(nb), dim3(kBlockThreads), (hipStream_t)stream,       \
                        (const float*)scratch, (uint16_t*)y, M, N, y_ld, sk);                    \
        } else {                                                                                 \
            R3D_KLAUNCH(R3D_KNAME((gemm_q_kernel<MTV, A8V, NTV, false>)), grid, dim3_blk,        \
                        (hipStream_t)stream, (const uint8_t*)a, (const float*)a_scale,           \
                        (const uint8_t*)b, (const void*)b_scale, bs16, (uint16_t*)y,        \
                        (float*)nullptr, M, N, K, group, pb1, a_ld, y_ld, kslab, nkb);           \
        }                                                                                        \
    } while (0)
#define R3D_Q_A8(MTV, A8V) do { if (nt) R3D_Q_NT(MTV, A8V, true); else R3D_Q_NT(MTV, A8V, false); } while (0)
#define R3D_Q_MT(MTV) do { if (a8) R3D_Q_A8(MTV, true); else R3D_Q_A8(MTV, false); } while (0)
    switch (mt) {
        case 1:  R3D_Q_MT(1);  break;
        case 2:  R3D_Q_MT(2);  break;
        case 4:  R3D_Q_MT(4);  break;
        case 8:  R3D_Q_MT(8);  break;
        default: R3D_Q_MT(16); break;
    }
#undef R3D_Q_MT
#undef R3D_Q_A8
#undef R3D_Q_NT
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The declaration in r3d.h carries no scale dtype, and the weight's block scales are bf16 for the
 * whole quantised family (see the entry above), so the family needs a second spelling. `_ex` adds
 * `bs_dt`, the b_scale plane's RadDataType: RAD_BF16 reads it as bf16, anything else -- 0 included
 * -- reads it as f32, which is exactly what the entry below, and every caller written before the
 * bf16 discovery, asks for. The shim takes the dtype off the operand and calls `_ex`; the old entry
 * forwards, so a raw caller of the old name sees the behaviour it was written against. */
extern "C" int r3d_gemm_nt_q_fp8(long a, long a_scale, long b, long b_scale, long y, int M, int N,
                                 int K, int group, int pb1, int pb2, int a_ld, int y_ld, int sk,
                                 int nt, long scratch, long stream)
{
    return r3d_gemm_nt_q_fp8_ex(a, a_scale, b, b_scale, y, M, N, K, group, pb1, pb2, a_ld, y_ld, sk,
                                nt, scratch, RAD_F32, stream);
}

/* ============================================================== the plugin ABI entry point
 *
 * gemm_nt_q's operands are [a, a_scale, b, b_scale, y, a_sum?, b_ref?] and its parameters M, N, K,
 * group, dtype, pb1?, pb2?, split_n? (libref's pGemmQ/oGemmQ). */
enum { Q_A = 0, Q_ASCALE, Q_B, Q_BSCALE, Q_Y, Q_ASUM, Q_BREF };

#define R3D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R3D_NO(_s); } while (0)
static int dt_f32(const RadTensor* t) { return r3d_is_f32(t) ? RAD_OK : RAD_E_DTYPE; }
static int dt_bf16(const RadTensor* t) { return r3d_is_bf16(t) ? RAD_OK : RAD_E_DTYPE; }

/* The trace latch for this file: distinct GEMM shapes, printed once each. A static array and a
 * counter, no lock -- the flag is a diagnostic and a doubled line is a cosmetic cost, a lock in
 * the served path is not. */
struct GemmPlanKey { long long M, N, K; unsigned a_rank, as_rank; long long a_ld, y_ld; };

static bool gemm_plan_once(long long M, long long N, long long K, unsigned a_rank,
                           unsigned as_rank, long long a_ld, long long y_ld)
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

extern "C" int r3d_rad_gemm_nt_q_fp8(const RadArgs* a, RadStream s)
{
    const RadTensor* A  = r3d_opt(a, Q_A);
    const RadTensor* AS = r3d_opt(a, Q_ASCALE);
    const RadTensor* B  = r3d_opt(a, Q_B);
    const RadTensor* BS = r3d_opt(a, Q_BSCALE);
    const RadTensor* Y  = r3d_opt(a, Q_Y);
    if (!A || !B || !BS || !Y) return R3D_NO(RAD_E_INVAL);
    /* The schemes this kernel decodes: E4M3 codes on the weight, bf16 activations (fp8a16) or
     * E4M3 codes with their own scale (fp8a8). E5M2 has no decoder here. */
    if (!r3d_is_e4m3(B)) return R3D_NO(RAD_E_DTYPE);
    /* The weight's block scales are bf16 for the whole quantised family -- the shape hook's
     * `o.t(f32s ? RAD_F32 : RAD_BF16, {nb, kb})` is the container's `weight_scale_inv`, and f32 is
     * the separate `fp8a8f32` spelling, which no row here claims (those bands stay with the
     * reference). Accept both; the kernel widens bf16 -> f32 at the load, exactly and for free. */
    const bool bs16 = r3d_is_bf16(BS);
    if (!bs16 && !r3d_is_f32(BS)) return R3D_NO(RAD_E_DTYPE);
    R3D_NEED(dt_bf16(Y));
    const bool a8 = AS != nullptr;
    if (a8) {
        if (!r3d_is_e4m3(A)) return R3D_NO(RAD_E_DTYPE);
        R3D_NEED(dt_f32(AS));
    } else {
        R3D_NEED(dt_bf16(A));
    }

    const long long K = r3d_geti_or(a, "K", r3d_dim(B, (int)B->rank - 1));
    const long long N = r3d_geti_or(a, "N", r3d_dim(B, (int)B->rank - 2));
    if (K < 1 || N < 1) return R3D_NO(RAD_E_SHAPE);
    const long long nel = r3d_numel(Y);
    if (nel < N || nel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = nel / N;                       // never from the `M` parameter
    if (M < 1 || M > kMTMax) return R3D_NO(RAD_E_SHAPE);

    /* The block geometry, and the disagreement libref refuses: `group` is what the selector
     * matched on, and pb2 is the checkpoint's, and they describe the same bytes. */
    const long long group = r3d_geti_or(a, "group", R3D_FP8_BLOCK);
    const long long pb1 = r3d_geti_or(a, "pb1", group), pb2 = r3d_geti_or(a, "pb2", group);
    if (group != R3D_FP8_BLOCK) return R3D_NO(RAD_E_SHAPE);
    if (pb2 != 0 && pb2 != group) return R3D_NO(RAD_E_SHAPE);
    if (pb1 <= 0) return R3D_NO(RAD_E_SHAPE);
    if (K % group) return R3D_NO(RAD_E_SHAPE);
    const long long nkb = (K + group - 1) / group, nrowb = (N + pb1 - 1) / pb1;
    if (r3d_numel(A) < M * K || r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(BS) < nrowb * nkb) return R3D_NO(RAD_E_SHAPE);
    if (AS && r3d_numel(AS) < M * nkb) return R3D_NO(RAD_E_SHAPE);
    /* `a_sum` removes an activation zero point this grid does not have, and `b_ref` is an mxfp4
     * scale reference: libref refuses both beside these schemes (gemm_ntq_impl), so we do. */
    if (r3d_opt(a, Q_ASUM)) return R3D_NO(RAD_E_UNSUPPORTED);
    if (r3d_opt(a, Q_BREF)) return R3D_NO(RAD_E_UNSUPPORTED);

    long long arows = 0, yrows = 0;
    const long long a_ld = r3d_rows_pitch(A, K, &arows);
    const long long y_ld = r3d_rows_pitch(Y, N, &yrows);
    if (a_ld < 0 || y_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (arows < M || yrows < M) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(B) || !r3d_contig(BS)) return R3D_NO(RAD_E_STRIDE);
    if (AS && !r3d_contig(AS)) return R3D_NO(RAD_E_STRIDE);
    /* The scale planes have no pitch argument in the launcher, so they must be dense, and their
     * stated grid must be the one the kernel walks. */
    if (r3d_dim(BS, 1) != 0 && BS->shape[BS->rank - 1] != nkb && r3d_numel(BS) != nrowb * nkb)
        return R3D_NO(RAD_E_SHAPE);
    if ((a8 && (a_ld % 8)) || (!a8 && (a_ld % 8))) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(A->data, 16) || !r3d_aligned(B->data, 16)) return R3D_NO(RAD_E_ALIGN);

    /* The plan's own numbers for the M<=16 band, under RAD_R3D_TRACE=1 and once per distinct
     * shape -- the same instrument the tiled row carries, in the row that serves the short steps.
     * It lives here because the served collapse begins at 15 prompt tokens, a step the tiled shim
     * never sees: the ladder's numbers below 17 are M=5..16 and the tiled row's line cannot speak
     * for them. M is derived off the activation's output the way the row derives it (line 430:
     * numel(y)/N, never the `M` parameter), and nothing is read that the checks above did not
     * already require. With the flag unset this is one cached getenv and a return: no copy, no
     * latch slot, no new operand read, no syscall. */
    if (r3d_trace_on() && gemm_plan_once(M, N, K, A->rank, AS ? AS->rank : 0u, a_ld, y_ld)) {
        std::fprintf(stderr, "[r3d] plan: gemm M=%lld (band M<=%d) N=%lld K=%lld group=%lld "
                     "a rank=%u numel=%lld stride=[%lld,%lld] pitch=%lld rows=%lld "
                     "a_scale %s numel=%lld y rank=%u numel=%lld stride=[%lld,%lld] pitch=%lld "
                     "Mpad=%lld\n",
                     M, kMTMax, N, K, group, (unsigned)A->rank, r3d_numel(A),
                     r3d_dim(A, 0), r3d_dim(A, 1), a_ld, arows,
                     AS ? "rank=2" : "absent (fp8a16, bf16 activations)",
                     AS ? r3d_numel(AS) : 0LL, (unsigned)Y->rank, r3d_numel(Y), r3d_dim(Y, 0),
                     r3d_dim(Y, 1), y_ld, (M + 15) / 16 * 16);
        r3d_trace_desc("gemm a", A);
        r3d_trace_desc("gemm y", Y);
        if (AS) r3d_trace_desc("gemm a_scale", AS);
    }

    R3dGemmCfg c;
    r3d_cfg_gemm(a, &c);
    int sk = c.sk > 0 ? c.sk : pick_sk(K, (int)group, 16);
    if (sk > 16) return R3D_NO(RAD_E_SHAPE);           // the scratch hook sizes for 16
    if (K % ((long long)sk * group)) return R3D_NO(RAD_E_SHAPE);
    if (sk > 1) {
        const long long need = (long long)sk * M * N * 4;
        if (!a->scratch || a->scratch_bytes < need) return R3D_NO(RAD_E_SCRATCH);
    }

    const int e = r3d_gemm_nt_q_fp8_ex(r3d_addr(A), r3d_addr(AS), r3d_addr(B), r3d_addr(BS),
                                    r3d_addr(Y), (int)M, (int)N, (int)K, (int)group, (int)pb1,
                                    (int)pb2, (int)a_ld, (int)y_ld, sk, c.nt,
                                    (long)(uintptr_t)(sk > 1 ? a->scratch : nullptr),
                                    (int)BS->dtype,        /* bf16 weight_scale_inv or f32 */
                                    (long)(uintptr_t)s);
    if (e == 0) return RAD_OK;
    return R3D_NO(e == -1 ? RAD_E_INVAL : e == -2 ? RAD_E_SHAPE
                        : e == -3 ? RAD_E_STRIDE : RAD_E_DEVICE);
}
