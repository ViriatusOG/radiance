/* r3d_gemm_i8a8.hip -- the int8 trunk GEMM: gemm_nt_q at dtype i8a8 and i8a8h, every M.
 *
 *   y[M,N] = a[M,K] @ b[N,K]^T, int8 a against signed int8 codes, the scale group 128.
 *
 * The operands are quant_act_i8g's (i8a8) or had_quant_act_i8g's (i8a8h): the activation's codes
 * are row-major, with one f32 scale per (row, group); the weight is the container's I8 [N, K] as
 * stored, with one bf16 scale per (row, group). i8a8h is the same launch: the rotation is in the
 * stored weight and in the activation's quantiser, and the weight's scale carries the /128. This
 * is the w8a8 wave-matrix arm (r3d_gemm_w8a8_nt_m64.hip) with those three differences: no step
 * order to unpermute, the activation scale read per group, and a bf16 weight scale.
 *
 * ---- the group law --------------------------------------------------------------------------
 *
 * libref's quant_dot: a group's 128 products are |a * code| <= 127 * 127, their sum 2,064,512
 * an exact integer far inside i32, so the eight wmma steps of a group chain an i32 sub-sum with
 * no rounding, and the group term is
 *
 *   acc += (sub * a_scale[m, g]) * b_scale[n, g]
 *
 * in f32, in ascending group order, then one bf16 store. Each of the three operations rounds
 * as the reference's does (two multiplies, then the add), so the row is bit-identical to
 * libref's quant_dot, not merely inside gemm_nt_q's tolerance.
 *
 * ---- the shape of a block -------------------------------------------------------------------
 *
 * A wave is one 16-column tile of N over MT 16-row tiles of M; a block is eight waves, and the
 * grid is (N / 128, M / (16 * MT)). The B fragment (the weight column) is loaded once per 16-k
 * step and reused over the MT row tiles. MT is 1 for M <= 16 (decode: no second tile of clamped
 * duplicate rows) and 2 above. A block reads its 128 weight rows once, so the weight is read
 * ceil(M / (16 * MT)) times in all: a prefill chunk is served, at the cost of that re-read.
 *
 * ---- the B pointer --------------------------------------------------------------------------
 *
 * The weight row base `w = B + n_real * b_rb` carries NO group offset; the group's k-base
 * enters the fragment read exactly once, as `w + kt + k` with kt = g * 128.
 *
 * ---- the register budget --------------------------------------------------------------------
 *
 * As the w8a8 arm at MT = 2: about 66 VGPRs a lane, 528 for eight waves of the 1024 a gfx11
 * SIMD keeps.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"

#include <algorithm>

namespace {

constexpr int kWaves        = 8;
constexpr int kBlockThreads = kWaves * kWave;
constexpr int kGroup        = 128;            /* the scale group: one k tile, eight wmma steps */
constexpr int kPitch        = 33;             /* dwords a staged a row: 33 mod 32 = 1, bank-free */


/* Signed i8 in, signed i32 out: the builtin's sign selectors take the operands' signedness as
 * compile-time constants, both signed for this grid. */

template <bool NT>
__device__ __forceinline__ r3d_u32x4 ldw(const uint8_t* p)
{
    if constexpr (NT) return *(const __attribute__((address_space(1))) r3d_u32x4*)p;
    else return __builtin_bit_cast(r3d_u32x4, *(const uint4*)p);
}

/* y[M,N] = a[M,K] @ b[N,K]^T, MT 16-row tiles a wave, one 16-column tile a wave, one scale
 * group a k tile. The dot2 arm's contract: K a multiple of the group, the group term the
 * reference's, in ascending group order. */
template <bool NT, int MT>
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemm_i8a8_wmma_kernel(
    const int8_t* __restrict__ a, const float* __restrict__ sa, int sa_rb,
    const uint8_t* __restrict__ B, int b_rb, const uint16_t* __restrict__ sb,
    int sb_rb, uint16_t* __restrict__ Y,
    int M, int N, int K, int a_ld, int y_ld)
{
    __shared__ __align__(16) uint32_t s_a[MT * 16 * kPitch];

    const uint32_t tid  = threadIdx.x;
    const uint32_t lane = tid & (kWave - 1);
    const uint32_t wave = tid / kWave;
    const uint32_t r = lane & 15;                   // the replicated row/column index

    const uint32_t n0 = (uint32_t)blockIdx.x * kWaves * 16 + wave * 16 + r;
    const int n_real = (int)(n0 < (uint32_t)N ? n0 : (uint32_t)(N - 1));
    const int mbase = (int)blockIdx.y * MT * 16;    // this block's first row

    const uint8_t* w = B + (size_t)n_real * b_rb;   // the row base: NO group offset
    v8float acc[MT];
    #pragma unroll
    for (int i = 0; i < MT; ++i) acc[i] = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    for (int g = 0; g < K / kGroup; ++g) {
        const int kt = g * kGroup;

        /* Stage MT*16 rows x 128 i8, row-major as the quantiser stored them. A row past M
         * re-reads the last real row, whose products the store drops. */
        for (int i = (int)tid; i < MT * 16 * (kGroup / 4); i += kBlockThreads) {
            const int row = i / (kGroup / 4), d = i % (kGroup / 4);
            const int rr = mbase + row;
            const uint32_t src =
                *(const uint32_t*)(a + (size_t)(rr < M ? rr : M - 1) * a_ld + kt + 4 * d);
            s_a[(size_t)row * kPitch + d] = src;
        }
        group_barrier();

        #pragma unroll
        for (int i = 0; i < MT; ++i) {
            const uint32_t* ap = s_a + (size_t)(i * 16 + r) * kPitch;
            v8int sub = v8int{ 0, 0, 0, 0, 0, 0, 0, 0 };
            for (int k = 0; k < kGroup; k += 16) {
                const r3d_u32x4 aq = *(const r3d_u32x4*)(ap + k / 4);
                // The group's k-base enters once (kt); the row base carries none.
                const r3d_u32x4 bq = ldw<NT>(w + kt + k);
                sub = wmma_i8(v4int{ (int)aq.x, (int)aq.y, (int)aq.z, (int)aq.w },
                              v4int{ (int)bq.x, (int)bq.y, (int)bq.z, (int)bq.w }, sub);
            }
            /* The group term, the reference's parentheses and the reference's roundings: the
             * product (sub * a_scale) * b_scale is rounded to f32 and then added, a v_mul and a
             * v_add, never a v_fmac. A fused multiply-add keeps the product unrounded, which
             * lands about half the launches of a model one bf16 step from quant_dot at their
             * worst element; libref and every unit here are built -ffp-contract=off, and an asm
             * fmac is outside that flag's reach. The b_scale load, its drain, its convert and
             * the eight terms run as one asm block: the load goes straight from global, the
             * drain is a vmcnt this asm owns, and the scale's register is written only by this
             * block's global_load, so no fragment load can clobber it mid-group. v20 (the scale)
             * and v21 (the rounded product) are named in the clobber because this LLVM aliases a
             * constrained asm output with a dead input register; the clobber keeps every
             * t/acc/sub register out of both across the block. The weight's scale is bf16: its
             * f32 is the word shifted up 16. The activation's scale is this group's, one per
             * (row, group). */
            float t[8];
            #pragma unroll
            for (int g2 = 0; g2 < 8; ++g2) {
                const int m = mbase + i * 16 + g2 * 2 + (int)(lane >> 4);
                t[g2] = (float)sub[g2] * sa[(size_t)(m < M ? m : M - 1) * sa_rb + g];
            }
            {
                const uint64_t ga = (uint64_t)(sb + (size_t)n_real * sb_rb + g);
                asm volatile(
                    "global_load_u16 v20, %[ga], off\n\t"
                    "s_waitcnt vmcnt(0)\n\t"
                    "v_lshlrev_b32_e32 v20, 16, v20\n\t"
                    "v_mul_f32_e32 v21, %[t0], v20\n\t"
                    "v_add_f32_e32 %[a0], %[a0], v21\n\t"
                    "v_mul_f32_e32 v21, %[t1], v20\n\t"
                    "v_add_f32_e32 %[a1], %[a1], v21\n\t"
                    "v_mul_f32_e32 v21, %[t2], v20\n\t"
                    "v_add_f32_e32 %[a2], %[a2], v21\n\t"
                    "v_mul_f32_e32 v21, %[t3], v20\n\t"
                    "v_add_f32_e32 %[a3], %[a3], v21\n\t"
                    "v_mul_f32_e32 v21, %[t4], v20\n\t"
                    "v_add_f32_e32 %[a4], %[a4], v21\n\t"
                    "v_mul_f32_e32 v21, %[t5], v20\n\t"
                    "v_add_f32_e32 %[a5], %[a5], v21\n\t"
                    "v_mul_f32_e32 v21, %[t6], v20\n\t"
                    "v_add_f32_e32 %[a6], %[a6], v21\n\t"
                    "v_mul_f32_e32 v21, %[t7], v20\n\t"
                    "v_add_f32_e32 %[a7], %[a7], v21"
                    : [a0] "+v"(acc[i][0]), [a1] "+v"(acc[i][1]), [a2] "+v"(acc[i][2]),
                      [a3] "+v"(acc[i][3]), [a4] "+v"(acc[i][4]), [a5] "+v"(acc[i][5]),
                      [a6] "+v"(acc[i][6]), [a7] "+v"(acc[i][7])
                    : [ga] "v"(ga),
                      [t0] "v"(t[0]), [t1] "v"(t[1]), [t2] "v"(t[2]), [t3] "v"(t[3]),
                      [t4] "v"(t[4]), [t5] "v"(t[5]), [t6] "v"(t[6]), [t7] "v"(t[7])
                    : "v20", "v21");
            }
        }
        group_barrier();                      // the next tile's stage must not race this one's reads
    }

    const bool keep = (n0 < (uint32_t)N);
    if (!keep) return;
    #pragma unroll
    for (int i = 0; i < MT; ++i)
    #pragma unroll
        for (int g2 = 0; g2 < 8; ++g2) {
            const int m = mbase + i * 16 + g2 * 2 + (int)(lane >> 4);
            if (m < M) Y[(size_t)m * y_ld + n0] = f32_to_bf16(acc[i][g2]);
        }
}

}  // namespace

/* ==================================================================================================
 * The RadArgs shim. Schema positions: a 0, a_scale 1, b 2, b_scale 3, y 4, a_sum 5, b_ref 6.
 * ================================================================================================== */
extern "C" int r3d_rad_gemm_nt_q_i8a8(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(a, "dtype");
    if (!dt || (std::strcmp(dt, "i8a8") != 0 && std::strcmp(dt, "i8a8h") != 0))
        return R3D_NO(RAD_E_DTYPE);
    if (r3d_geti_or(a, "group", kGroup) != kGroup) return R3D_NO(RAD_E_SHAPE);

    const RadTensor* A = r3d_opt(a, 0);
    const RadTensor* AS = r3d_opt(a, 1);
    const RadTensor* B = r3d_opt(a, 2);
    const RadTensor* BS = r3d_opt(a, 3);
    const RadTensor* Y = r3d_opt(a, 4);
    const RadTensor* ASUM = r3d_opt(a, 5);
    const RadTensor* BREF = r3d_opt(a, 6);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(BS) || !r3d_p(Y))
        return R3D_NO(RAD_E_INVAL);
    if (r3d_has(ASUM) || r3d_has(BREF)) return R3D_NO(RAD_E_DTYPE);

    if (!r3d_is_i8(A)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(AS)) return R3D_NO(RAD_E_DTYPE);
    if (B->dtype != RAD_I8) return R3D_NO(RAD_E_DTYPE);
    if (BS->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_bf16(Y)) return R3D_NO(RAD_E_DTYPE);

    const long long N = r3d_geti_or(a, "N", 0), K = r3d_geti_or(a, "K", 0);
    if (N < 1 || K < 1) return R3D_NO(RAD_E_SHAPE);
    if (K % kGroup) return R3D_NO(RAD_E_SHAPE);
    const long long G = K / kGroup;

    const long long ynel = r3d_numel(Y);
    if (ynel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = ynel / N;
    if (M < 1 || M > (1 << 24)) return R3D_NO(RAD_E_SHAPE);

    if (r3d_numel(A) < M * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(AS) < M * G) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(BS) < N * G) return R3D_NO(RAD_E_SHAPE);

    const long long ap = r3d_row_pitch(A);
    if (ap < K || ap % 4) return R3D_NO(RAD_E_STRIDE);
    const long long asp = M > 1 ? r3d_row_pitch(AS) : G;
    if (asp < G) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);
    const long long bp = r3d_row_pitch(B), bsp = r3d_row_pitch(BS);
    if (bp < K || (bp & 15) || bsp < G) return R3D_NO(RAD_E_STRIDE);
    long long yrows = 0;
    const long long yp = r3d_rows_pitch(Y, N, &yrows);
    if (yp < N) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(r3d_p(A), 4) || !r3d_aligned(r3d_p(B), 16)) return R3D_NO(RAD_E_ALIGN);

    R3dGemmCfg c;
    r3d_cfg_gemm(a, &c);

    const int mt = M <= 16 ? 1 : 2;
    const dim3 grid((unsigned)((N + kWaves * 16 - 1) / (kWaves * 16)),
                    (unsigned)((M + mt * 16 - 1) / (mt * 16))), block(kBlockThreads);
#define R3D_I8A8(NTV, MTV)                                                                        \
    R3D_KLAUNCH((gemm_i8a8_wmma_kernel<NTV, MTV>), grid, block, (hipStream_t)s,                   \
                (const int8_t*)r3d_p(A), (const float*)r3d_p(AS), (int)asp,                       \
                (const uint8_t*)r3d_p(B), (int)bp, (const uint16_t*)r3d_p(BS), (int)bsp,          \
                (uint16_t*)r3d_wp(Y), (int)M, (int)N, (int)K, (int)ap, (int)yp)
    if (c.nt) { if (mt == 1) R3D_I8A8(true, 1); else R3D_I8A8(true, 2); }
    else      { if (mt == 1) R3D_I8A8(false, 1); else R3D_I8A8(false, 2); }
#undef R3D_I8A8
    return hipGetLastError() == hipSuccess ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}
