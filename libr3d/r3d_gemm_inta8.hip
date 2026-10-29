/* r3d_gemm_inta8.hip -- gemm_nt_q at dtype=w4a8 and w8a8: the symmetric int4 and int8 weight
 * grids at decode heights, one kernel per grid.
 *
 * The grids, libref's descriptions: int8 activations in quant_act_i8's step order against
 * signed weight codes (I4, two a byte, or I8, one a byte) with one f16 scale per (row, 128 of
 * K), an f32 activation scale per row, and no zero point -- the dequant is code times scale,
 * so there is no a_sum to carry and no bias to take off. The group sub-sum is formed on the
 * integer grid and scaled once per group, the reference's order:
 *
 *   sub = sum over the group of a * code, an exact integer
 *   acc += (sub * a_scale[m]) * scale[n, g]
 *
 * ---- exactness, and why the dots may run in f16 --------------------------------------------------
 *
 * A lane's 64 products are |a * code| <= 127 * 7 (w4a8) or 127 * 127 (w8a8), and the group's
 * 128 of them sum to 113,792 (w4a8) or 2,064,512 (w8a8): both under 2^24, the f32 mantissa, so
 * the lane partial and the pair fold are exact, and the group terms land in the accumulator
 * in the reference's ascending order. The codes widen to f16 through a sign extension, which
 * is exact for both grids; the activations widen through the i8 codec the w2a8 arm uses.
 *
 * The activation is the step order: within each sixteen-byte step the middle two dwords are
 * exchanged, so the staging reads through the same position map the w2a8 arm does, and a
 * weight row's codes are plain row-major, two a byte at I4 and one a byte at I8.
 *
 * ---- the tile --------------------------------------------------------------------------------------
 *
 * One workgroup serves kWaves weight rows; a wavefront takes one row, lane l owns rows
 * l & (MT-1) and k-lane l / MT, and the chunk is 2048 / MT codes: a multiple of the 128
 * group at every MT, so a chunk carries CG = 16 / MT whole groups and the group's scale is
 * one read. The lane's 64 codes are 32 bytes at I4, two 16-byte loads, and 64 at I8, four.
 * The lane's 64 sit in the group's two halves the w2a8 arm folds with one shfl_xor(MT), and
 * the group's term -- scaled by the activation's row scale and the weight's group scale --
 * goes to shared, where the row's owner lane sums the chunk's groups in ascending order.
 *
 * M > 16 is the tiled row's; the w2a8 arm serves the 2- and 4-bit container grids there, and
 * the 8-bit grid at prefill heights is declared out of this unit.
 *
 * ---- refusals --------------------------------------------------------------------------------------
 *
 * Every structural violation is a refusal, in the shim: the dtypes of all five operands,
 * K % 128, the band's M, the b row pitch's 16-byte alignment, the a and y pitches. There is
 * no data that is a refusal: both grids have no reserved value.
 */

#include "r3d_common.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"
#include "r3d_args.h"

namespace {

constexpr int kWave          = 32;
constexpr int kWaves         = 8;
constexpr int kBlockThreads  = kWaves * kWave;
constexpr int kMTMax         = 16;
constexpr int kGroup         = 128;
constexpr int kTileQuads     = 512;        /* MT rows x CHUNK/4 f16, constant over MT */

/* Column c sits at byte step_pos(c), bits 2 and 3 of c swapped, the map its own inverse.
 * Applied to the four columns of one dword it leaves them four contiguous bytes and turns
 * the map on the DWORD index into a swap of bits 0 and 1 -- the middle two dwords of a
 * 16-byte step exchanged. */
__device__ __forceinline__ int64_t dword_pos(int64_t d)
{
    return 4 * ((d & ~(int64_t)3) | ((d & 1) << 1) | ((d & 2) >> 1));
}

/* The LDS swizzle, at the granularity the tile is touched: a quad of four f16, two dwords,
 * moved as one 8-byte word. Quads are numbered q = r*(CHUNK/4) + c, and p_ix XORs q's low
 * four bits with bits 4..7 -- a bijection on 16-quad groups. */
__device__ __forceinline__ int p_ix(int q) { return q ^ ((q >> 4) & 15); }

/* Four int8 codes -> two f16x2 dwords. Exact: an integer of |a| <= 127 needs no rounding. */
__device__ __forceinline__ void i8x4_to_f16x2(uint32_t w, uint32_t& lo, uint32_t& hi)
{
    lo = f16x2_pack((float)(int32_t)(int8_t)(w & 0xffu), (float)(int32_t)(int8_t)((w >> 8) & 0xffu));
    hi = f16x2_pack((float)(int32_t)(int8_t)((w >> 16) & 0xffu),
                    (float)(int32_t)(int8_t)((w >> 24) & 0xffu));
}

/* Four signed int4 codes (two bytes, the low code in the low nibble) -> two f16x2 dwords.
 * Exact: the sign extension lands on an integer, and f16 steps by 1/8 at exponent 2^-3.
 * A nibble n is the value (n & 7) - (n & 8): 0..7 pass through and 8..15 land on -8..-1.
 * The gfx1102 backend folds the bias-8 spelling ((n ^ 8) - 8) to n ^ 16 once it has proven
 * the nibble range, so the sign is kept as a bit term. */
__device__ __forceinline__ void i4x4_to_f16x2(uint32_t w, uint32_t& lo, uint32_t& hi)
{
    const uint32_t n0 = w & 0xFu;
    const uint32_t n1 = (w >> 4) & 0xFu;
    const uint32_t n2 = (w >> 8) & 0xFu;
    const uint32_t n3 = (w >> 12) & 0xFu;
    const int32_t c0 = (int32_t)((n0 & 7u) - (n0 & 8u));
    const int32_t c1 = (int32_t)((n1 & 7u) - (n1 & 8u));
    const int32_t c2 = (int32_t)((n2 & 7u) - (n2 & 8u));
    const int32_t c3 = (int32_t)((n3 & 7u) - (n3 & 8u));
    lo = f16x2_pack((float)c0, (float)c1);
    hi = f16x2_pack((float)c2, (float)c3);
}

__device__ __forceinline__ float dot4_f16x4(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1,
                                            float acc)
{
    const _Float16* fa = (const _Float16*)&a0;
    const _Float16* fb = (const _Float16*)&b0;
    acc = fmaf(fa[0], fb[0], acc);
    acc = fmaf(fa[1], fb[1], acc);
    const _Float16* fa1 = (const _Float16*)&a1;
    const _Float16* fb1 = (const _Float16*)&b1;
    acc = fmaf(fa1[0], fb1[0], acc);
    acc = fmaf(fa1[1], fb1[1], acc);
    return acc;
}

template <int MT, int CDT>
__global__ void __launch_bounds__(kBlockThreads) R3D_WAVES(2)
gemm_inta8_kernel(const int8_t* __restrict__ a, const float* __restrict__ sa,
                  const uint8_t* __restrict__ b, int b_rb, const uint16_t* __restrict__ sb,
                  int sb_rb, uint16_t* __restrict__ y, int M, int N, int K, int chunk,
                  int a_pitch, int y_pitch)
{
    constexpr int KL = kWave / MT;                    /* k-lanes per activation row */
    constexpr int CG = (2048 / MT) / kGroup;          /* groups per chunk: 16 / MT */
    constexpr int CL = 64;                            /* codes per lane per chunk */
    constexpr int NLOAD = (CL * CDT) / 128;           /* 16-byte loads: the lane's window */

    __shared__ __align__(16) uint2 sA[kTileQuads];    /* MT rows x CHUNK/4 f16, swizzled */
    __shared__ float sT[kWaves * 16];                 /* warp*16 + m*CG + j */
    __shared__ float sSa[MT];

    const int lane = threadIdx.x & (kWave - 1);
    const int warp = (int)(threadIdx.x / kWave);
    const int n = blockIdx.x * kWaves + warp;
    const bool live = n < N;
    if ((int)blockIdx.x * kWaves >= N) return;

    const int m = lane & (MT - 1);
    const int kl = lane / MT;
    const int row_bytes = b_rb;                     /* the codes' byte pitch, from the shim */
    const int cq = (CDT == 4) ? (chunk >> 1) : chunk;   /* the chunk's code bytes */
    const int nchunk = K / chunk;

    if (lane < MT) sSa[lane] = lane < M ? sa[lane] : 0.0f;
    __syncthreads();

    float acc = 0.0f;
    for (int ch = 0; ch < nchunk; ++ch) {
        /* Stage the chunk's activation rows, widened and step-unpermuted on the way in. Cells
         * are four codes: MT * CHUNK/4 = 512 of them at every MT, two per thread. */
        for (int i = threadIdx.x; i < MT * (chunk >> 2); i += kBlockThreads) {
            const int r = i / (chunk >> 2), c = i % (chunk >> 2);
            uint32_t w = 0;
            if (r < M)
                w = *(const uint32_t*)(a + (long long)r * a_pitch +
                                       dword_pos((int64_t)ch * (chunk >> 2) + c));
            const int q = r * (chunk >> 2) + c;
            uint32_t lo, hi;
            i8x4_to_f16x2(w, lo, hi);
            sA[p_ix(q)] = make_uint2(lo, hi);
        }
        __syncthreads();

        if (live) {
            /* 64 codes: 32 bytes at I4, 64 at I8. The lane's codes are kl * 64 within the
             * chunk, and the row pitch keeps every 16-byte load aligned. */
            const uint8_t* bp = b + (long long)n * row_bytes + (int64_t)ch * cq +
                                kl * (CL / 2) * (CDT / 4);
            uint4 qw[NLOAD];
#pragma unroll
            for (int L = 0; L < NLOAD; ++L)
                __builtin_memcpy(&qw[L], bp + 16 * L, 16);
            /* One value per quad: I4 the two-byte window as a 16-bit element, I8 the dword.
             * The I4 window is taken as a half at its own address; shifting a dword array by
             * a byte amount miscompiles on the gfx1102 backend. */
            uint32_t wd[(CL >> 2)];
            if (CDT == 4) {
                uint16_t wq[CL / 2];
#pragma unroll
                for (int L = 0; L < NLOAD; ++L)
                    __builtin_memcpy(&wq[8 * L], &qw[L], 16);
#pragma unroll
                for (int v = 0; v < (CL >> 2); ++v)
                    wd[v] = wq[v];
            } else {
#pragma unroll
                for (int L = 0; L < NLOAD; ++L)
                    { wd[4 * L + 0] = qw[L].x; wd[4 * L + 1] = qw[L].y;
                      wd[4 * L + 2] = qw[L].z; wd[4 * L + 3] = qw[L].w; }
            }

            const int qb = m * (chunk >> 2) + kl * (CL >> 2);
            /* 16 quads, decoded before the dot. */
            uint32_t cb[(CL >> 2)][2];
#pragma unroll
            for (int v = 0; v < (CL >> 2); ++v) {
                if (CDT == 4) i4x4_to_f16x2(wd[v] & 0xFFFFu, cb[v][0], cb[v][1]);
                else           i8x4_to_f16x2(wd[v], cb[v][0], cb[v][1]);
            }
            float p = 0.0f;                            /* the lane's half-group, an exact integer */
#pragma unroll
            for (int v = 0; v < (CL >> 2); ++v) {
                const uint2 av = sA[p_ix(qb + v)];
                p = dot4_f16x4(av.x, av.y, cb[v][0], cb[v][1], p);
            }

            /* Fold the pair: lanes l and l^MT are the same row, the two halves of one group. */
            const float isg = p + __shfl_xor(p, MT);

            const int g = ch * CG + (kl >> 1);
            const float sc = f16_to_f32(sb[(long long)n * sb_rb + g]);
            const float term = (isg * sSa[m]) * sc;    /* the reference's parentheses */
            /* One slot per (row, group-in-chunk): MT*CG is 16 at every MT. The group's two
             * half-lanes write the same value, which is what the pair fold above is for. */
            sT[warp * 16 + m * CG + (kl >> 1)] = term;
        }
        __syncthreads();
        /* The chunk's groups, in ascending order, into the lane that owns the row's answer:
         * the reference walks g in ascending order and f32 adds are not associative, so the
         * order is part of the answer, not a detail. */
        if (live && kl == 0) {
#pragma unroll
            for (int j = 0; j < CG; ++j)
                if (m < M) acc += sT[warp * 16 + m * CG + j];
        }
        __syncthreads();
    }

    if (live && kl == 0 && m < M) y[(long long)m * y_pitch + n] = f32_to_bf16(acc);
}

int mt_for(int M)
{
    int mt = 1;
    while (mt < M && mt < kMTMax) mt *= 2;
    return mt;
}

#define R3D_INTA8_LAUNCH(MT, CDT)                                                                                  \
    do {                                                                                                           \
        const dim3 grid((N + kWaves - 1) / kWaves), block(kBlockThreads);                                          \
        R3D_KLAUNCH(R3D_KNAME((gemm_inta8_kernel<MT, CDT>)), grid, block, (hipStream_t)stream,                     \
                    (const int8_t*)a, (const float*)a_scale, (const uint8_t*)b, (int)b_rb, (const uint16_t*)b_scale, \
                    (int)sb_rb, (uint16_t*)y, M, N, K, chunk, a_ld, y_ld);                                          \
    } while (0)

extern "C" int r3d_gemm_nt_q_w4a8_m16(long a, long a_scale, long b, long b_scale, long y,
                                      int M, int N, int K, int a_ld, int y_ld, long long b_rb,
                                      long long sb_rb, long stream)
{
    if (!a || !a_scale || !b || !b_scale || !y) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (M > kMTMax) return -2;
    if (K % kGroup) return -2;
    const int mt = mt_for(M);
    const int chunk = 2048 / mt;
    if (K % chunk) return -2;
    if (!a_ld) a_ld = K;
    if (!y_ld) y_ld = N;
    if (a_ld % 4) return -3;
    if (y_ld < N) return -3;
    if (!b_rb) b_rb = K / 2;
    if (b_rb < K / 2) return -3;
    if (b_rb & 15) return -5;
    if (!sb_rb) sb_rb = K / kGroup;
    if (sb_rb < K / kGroup) return -3;
    if (((uintptr_t)a & 3u) || ((uintptr_t)b & 15u)) return -5;

    switch (mt) {
        case 1:  R3D_INTA8_LAUNCH(1, 4);  break;
        case 2:  R3D_INTA8_LAUNCH(2, 4);  break;
        case 4:  R3D_INTA8_LAUNCH(4, 4);  break;
        case 8:  R3D_INTA8_LAUNCH(8, 4);  break;
        default: R3D_INTA8_LAUNCH(16, 4); break;
    }
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_gemm_nt_q_w8a8_m16(long a, long a_scale, long b, long b_scale, long y,
                                      int M, int N, int K, int a_ld, int y_ld, long long b_rb,
                                      long long sb_rb, long stream)
{
    if (!a || !a_scale || !b || !b_scale || !y) return -1;
    if (M < 1 || N < 1 || K < 1) return -2;
    if (M > kMTMax) return -2;
    if (K % kGroup) return -2;
    const int mt = mt_for(M);
    const int chunk = 2048 / mt;
    if (K % chunk) return -2;
    if (!a_ld) a_ld = K;
    if (!y_ld) y_ld = N;
    if (a_ld % 4) return -3;
    if (y_ld < N) return -3;
    if (!b_rb) b_rb = K;
    if (b_rb < K) return -3;
    if (b_rb & 15) return -5;
    if (!sb_rb) sb_rb = K / kGroup;
    if (sb_rb < K / kGroup) return -3;
    if (((uintptr_t)a & 3u) || ((uintptr_t)b & 15u)) return -5;

    switch (mt) {
        case 1:  R3D_INTA8_LAUNCH(1, 8);  break;
        case 2:  R3D_INTA8_LAUNCH(2, 8);  break;
        case 4:  R3D_INTA8_LAUNCH(4, 8);  break;
        case 8:  R3D_INTA8_LAUNCH(8, 8);  break;
        default: R3D_INTA8_LAUNCH(16, 8); break;
    }
    return hipGetLastError() == hipSuccess ? 0 : -4;
}
#undef R3D_INTA8_LAUNCH

/* ================================================================== the shims
 *
 * gemm_nt_q's slots, as the two shims take them: 0 a (I8, step order), 1 a_scale (f32, one
 * per row), 2 b (I4 or I8 codes), 3 b_scale (f16, one per (row, 128 of K)), 4 y (bf16),
 * 5 a_sum (absent: symmetric), 6 b_ref (absent).
 */

extern "C" int r3d_rad_gemm_nt_q_w4a8_m16(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(a, "dtype");
    if (dt && std::strcmp(dt, "w4a8") != 0) return R3D_NO(RAD_E_DTYPE);

    const RadTensor* A = r3d_opt(a, 0);
    const RadTensor* AS = r3d_opt(a, 1);
    const RadTensor* B = r3d_opt(a, 2);
    const RadTensor* BS = r3d_opt(a, 3);
    const RadTensor* Y = r3d_opt(a, 4);
    const RadTensor* ASUM = r3d_opt(a, 5);
    const RadTensor* BREF = r3d_opt(a, 6);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(BS) || !r3d_p(Y))
        return R3D_NO(RAD_E_INVAL);
    if (r3d_has(ASUM) || r3d_has(BREF)) return R3D_NO(RAD_E_DTYPE);   /* symmetric: nothing to remove */

    if (!r3d_is_i8(A)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(AS)) return R3D_NO(RAD_E_DTYPE);
    if (B->dtype != RAD_I4) return R3D_NO(RAD_E_DTYPE);
    if (BS->dtype != RAD_F16) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_bf16(Y)) return R3D_NO(RAD_E_DTYPE);

    const long long N = r3d_geti_or(a, "N", 0), K = r3d_geti_or(a, "K", 0);
    if (N < 1 || K < 1) return R3D_NO(RAD_E_SHAPE);
    if (K % kGroup) return R3D_NO(RAD_E_SHAPE);

    const long long ynel = r3d_numel(Y);
    if (ynel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = ynel / N;
    if (M < 1 || M > kMTMax) return R3D_NO(RAD_E_SHAPE);

    if (r3d_numel(A) < M * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(AS) < M) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(BS) < N * (K / kGroup)) return R3D_NO(RAD_E_SHAPE);

    const long long ap = r3d_row_pitch(A);
    if (ap < K || ap % 4) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);
    long long yrows = 0;
    const long long yp = r3d_rows_pitch(Y, N, &yrows);
    if (yp < N) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(r3d_p(A), 4) || !r3d_aligned(r3d_p(B), 16)) return R3D_NO(RAD_E_ALIGN);

    const long long bbr = r3d_row_pitch(B) / 2;   /* I4 counts elements, two a byte */
    const int rc = r3d_gemm_nt_q_w4a8_m16(r3d_addr(A), r3d_addr(AS), r3d_addr(B), r3d_addr(BS),
                                          r3d_addr(Y), (int)M, (int)N, (int)K, (int)ap, (int)yp,
                                          bbr, (long long)r3d_row_pitch(BS), (long)s);
    if (rc == -1) return R3D_NO(RAD_E_INVAL);
    if (rc == -2) return R3D_NO(RAD_E_SHAPE);
    if (rc == -3) return R3D_NO(RAD_E_STRIDE);
    if (rc == -5) return R3D_NO(RAD_E_ALIGN);
    if (rc == -4) return R3D_NO(RAD_E_DEVICE);
    return RAD_OK;
}

extern "C" int r3d_rad_gemm_nt_q_w8a8_m16(const RadArgs* a, RadStream s)
{
    if (!a) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(a, "dtype");
    if (dt && std::strcmp(dt, "w8a8") != 0) return R3D_NO(RAD_E_DTYPE);

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
    if (BS->dtype != RAD_F16) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_bf16(Y)) return R3D_NO(RAD_E_DTYPE);

    const long long N = r3d_geti_or(a, "N", 0), K = r3d_geti_or(a, "K", 0);
    if (N < 1 || K < 1) return R3D_NO(RAD_E_SHAPE);
    if (K % kGroup) return R3D_NO(RAD_E_SHAPE);

    const long long ynel = r3d_numel(Y);
    if (ynel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = ynel / N;
    if (M < 1 || M > kMTMax) return R3D_NO(RAD_E_SHAPE);

    if (r3d_numel(A) < M * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(AS) < M) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(BS) < N * (K / kGroup)) return R3D_NO(RAD_E_SHAPE);

    const long long ap = r3d_row_pitch(A);
    if (ap < K || ap % 4) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);
    long long yrows = 0;
    const long long yp = r3d_rows_pitch(Y, N, &yrows);
    if (yp < N) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(r3d_p(A), 4) || !r3d_aligned(r3d_p(B), 16)) return R3D_NO(RAD_E_ALIGN);

    const int rc = r3d_gemm_nt_q_w8a8_m16(r3d_addr(A), r3d_addr(AS), r3d_addr(B), r3d_addr(BS),
                                          r3d_addr(Y), (int)M, (int)N, (int)K, (int)ap, (int)yp,
                                          (long long)r3d_row_pitch(B), (long long)r3d_row_pitch(BS),
                                          (long)s);
    if (rc == -1) return R3D_NO(RAD_E_INVAL);
    if (rc == -2) return R3D_NO(RAD_E_SHAPE);
    if (rc == -3) return R3D_NO(RAD_E_STRIDE);
    if (rc == -5) return R3D_NO(RAD_E_ALIGN);
    if (rc == -4) return R3D_NO(RAD_E_DEVICE);
    return RAD_OK;
}

}   // namespace
