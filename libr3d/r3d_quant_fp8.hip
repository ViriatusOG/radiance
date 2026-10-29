/* r3d_quant_fp8.hip -- the activation quantisers and the dequantise that reads a container's own
 * codes.
 *
 * The arithmetic is libref's, transcribed. The group core is `q8_group`: an amax over
 * the group's consecutive columns, `sc = amax * (1.0f / top)` -- a MULTIPLY by the reciprocal, not a
 * divide -- codes `v * (1/sc)` clamped to +-top and encoded by the codec, and scale 1 for an
 * all-zero group, whose reciprocal would be an infinity. The five bodies are libref's
 * quant_act_fp8, quant_act_i8, quant_act_i8g, gated_quant_fp8 and dequant, loop order and rounding
 * points as written. The clamp is required and not defensive: `amax * (1/scale)` is
 * 448 exactly and can round an ulp above once the reciprocal is rounded, and a code out of range
 * encodes as NaN, which then spreads through every row that reads it.
 *
 * RDNA3 has no fp8 instruction (r3d_common.h), so both directions are the software codecs in
 * r3d_codecs.h, which are bit-exact mirrors of rad_plugin.h's; the absmax is a butterfly
 * (r3d_xor_max), which reproduces a `__shfl_xor` loop bit for bit, and a max is exact and
 * order-free, so the bracketing is q8_group's ascending one whatever the lane mapping.
 *
 * No kernel here carries a layout hook, and the codes are the bytes their consumer reads, which is
 * the reason a container converted on this card is readable anywhere. Four of the five planes are
 * plain row-major with one f32 scale per (row, group), because a dot2 GEMM reads codes in the order
 * they are stored. The fifth is the op whose NAME carries the order: libref stores quant_act_i8's
 * codes in the int8 WMMA's step order (`i8_step_pos`, the middle two dwords of every 16-byte step
 * exchanged) because that is what the w4a8, w8a8 and w2a8 GEMMs read, so this kernel stores them
 * that way too and its n is a multiple of 16; quant_act_i8g is the row-major int8 plane, and it is
 * the one this library's i8a8 GEMM consumes. The two are not interchangeable.
 *
 * The store address costs nothing either way: a lane owns FOUR CONSECUTIVE COLUMNS, so its codes are
 * one dword, a wave's are 128 contiguous bytes, and a wave covers exactly one 128-column group;
 * `i8_step_pos` preserves c % 4, so it permutes a wave's 128 bytes without splitting a lane. The
 * launchers therefore refuse `n % 4` -- and quant_act_i8 `n % 16` -- and a ragged last group
 * (n % group != 0) is written zero-padded: padding is a zero code and takes no part in the amax.
 *
 * Group widths: the rows ask for 128, the checkpoint's weight-scale block width, and the kernels
 * also serve any other power of two from 4 to 512, so a selftest can drive a narrow group and a
 * future row on a 64-wide grid needs no new kernel. A group is a power of two because the amax
 * butterfly is over the lanes covering it, GROUP/4 of them.
 */

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves   = kThreads / kWave;

enum : uint32_t {
    kColsPerLane = 4,                          /* one dword of codes a lane */
    kWaveCols    = kWave * kColsPerLane        /* 128: the columns a wave covers */
};

/* The largest code each quantiser saturates at. E4M3 has no infinity, so 448 is the largest finite
 * value, and the scale is a dequant multiplier like the checkpoint's weight_scale_inv: the GEMM
 * multiplies the activation's scale by the weight's block scale and never divides. */
constexpr float kTopE4M3 = 448.0f;
constexpr float kTopI8   = 127.0f;

/* The floor quant_act_i8's amax is raised to before its DIVIDE by 127, where quant_act_fp8
 * multiplies by a reciprocal, so the two quantisers round one ulp apart here; that is libref's
 * spelling and the GEMMs read the scale as it is. */
constexpr float kI8Floor = 1e-8f;

/* The LDS code-sum accumulator's width, in groups, as in libr4d's r4d_quant_act_i8. */
enum : uint32_t { kI8MaxGroups = 256 };

/* A bf16 element of a row: a whole-register 16-bit load (r3d_common.h) widened by a shift. */
__device__ __forceinline__ float ld_bf(const uint16_t* p, uint32_t i) {
    return bf16_to_f32((uint16_t)ldg_u16(p + i));
}

/* q8_group's clamp, in q8_group's order and with q8_group's comparisons, at the codec's top -- 448
 * for E4M3, 127 for int8. fminf/fmaxf are NOT the same function here: they replace a NaN with the
 * bound, and a NaN in an activation is a value the reference encodes as the format's NaN, not as
 * +-top. */
__device__ __forceinline__ float clamp_top(float c, float top) {
    if (c > top) c = top;
    if (c < -top) c = -top;
    return c;
}

/* The four codes a lane's four columns produce, in column order: byte 0 is column c0. */
__device__ __forceinline__ uint32_t e4m3_codes4(const float* v, float inv) {
    const uint32_t lo = e4m3x2_pack(clamp_top(v[0] * inv, kTopE4M3), clamp_top(v[1] * inv, kTopE4M3));
    const uint32_t hi = e4m3x2_pack(clamp_top(v[2] * inv, kTopE4M3), clamp_top(v[3] * inv, kTopE4M3));
    return lo | (hi << 16);
}

/* The same four codes as int8, at the same rounding points: clamp to +-127, then the codec's
 * round-to-nearest-even. quant_act_i8g's store; quant_act_i8's is the same bytes at a permuted
 * address. */
__device__ __forceinline__ uint32_t i8_codes4(const float* v, float inv) {
    const uint32_t b0 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[0] * inv, kTopI8));
    const uint32_t b1 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[1] * inv, kTopI8));
    const uint32_t b2 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[2] * inv, kTopI8));
    const uint32_t b3 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[3] * inv, kTopI8));
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

/* q8_group's scale from the group's amax: the reciprocal of `top` MULTIPLIED, not a divide -- the
 * one place the two int8 quantisers and the fp8 one agree, and the reason the per-row
 * quant_act_i8 spells its scale as a divide and is a separate body. */
__device__ __forceinline__ float group_scale(float am, float top) {
    return am > 0.0f ? am * (1.0f / top) : 1.0f;
}

/* The map libref uses, verbatim: where a row's column c is stored in the step-ordered int8 plane.
 * Bits 2 and 3 of c exchanged = the middle two dwords of every 16-byte step swapped, which is the
 * order the int8 WMMA reads an A fragment. The map is its own inverse, and it preserves c % 4, so a
 * lane's four columns stay contiguous and its store stays one dword. */
__device__ __forceinline__ uint32_t i8_step_pos(uint32_t c) {
    return (c & ~12u) | ((c & 4u) << 1) | ((c & 8u) >> 1);
}

/* block_sum's twin for a maximum -- r3d_common.h has the sum. A max needs no order argument at all:
 * it is exact, so the wave partials combine in any order -- but it is spelled the same way, so a
 * reader sees one reduction shape in this library and not two. `lds` holds kWaves floats. */
__device__ __forceinline__ float block_max(float v, float* lds) {
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    v = wave_max(v);
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float t = lds[0];
#pragma unroll
    for (uint32_t w = 1; w < kWaves; ++w) t = fmaxf(t, lds[w]);
    return t;
}

/* ================================================================== quant_act_fp8
 *
 * Per-(row, group) symmetric E4M3 of an activation, over q8_group. One workgroup
 * a row, a wave a group, a lane four columns -- libr4d's layout, because the reason for
 * it (one dword a lane, 128 contiguous bytes a wave) is the store and not the instruction set.
 *
 * GROUP is the group's width. At 128 a wave covers one group and the amax is r3d_xor_max<32>. A
 * narrower group is covered by GROUP/4 lanes of the wave and the butterfly is over those lanes, so
 * one wave quantises 128/GROUP groups in a pass; a wider one is covered by one wave in sub-blocks
 * of 128 whose maxima are fmaxf'd. All three are q8_group's bracketing.
 *
 * Rows are dim 0, so a rank-3 activation ([tokens, heads, width], a GDN block's out-projection
 * input) is one row per token, as libref reads it: its consuming GEMM's K is heads*width, and one
 * row per head would put scales on a grid that GEMM does not read.
 */
template <int GROUP>
__global__ __launch_bounds__(kThreads) R3D_WAVES(8) void quant_act_fp8_kernel(
    const uint16_t* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s, uint32_t n,
    uint32_t ngrp, uint32_t x_ld, uint32_t s_ld) {
    constexpr int kSub   = GROUP >= (int)kWaveCols ? GROUP / (int)kWaveCols : 1;
    constexpr int kPer   = GROUP < (int)kWaveCols ? (int)kWaveCols / GROUP : 1;
    constexpr int kLanes = GROUP < (int)kWaveCols ? (int)(GROUP / kColsPerLane) : (int)kWave;
    static_assert(kLanes >= 1 && kLanes <= (int)kWave, "a group spans at most one wave");

    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const uint16_t* __restrict__ xr = x + (size_t)row * x_ld;
    uint8_t* __restrict__ qr = q + (size_t)row * n;

    const uint32_t nunit = (ngrp + kPer - 1) / kPer;
    for (uint32_t u = wave; u < nunit; u += kWaves) {
        const uint32_t g0 = u * kPer, cbase = g0 * GROUP;
        float v[kSub][kColsPerLane];
        float am = 0.0f;
#pragma unroll
        for (int sb = 0; sb < kSub; ++sb) {
            const uint32_t c0 = cbase + (uint32_t)sb * kWaveCols + lane * kColsPerLane;
#pragma unroll
            for (int j = 0; j < kColsPerLane; ++j) {
                const uint32_t c = c0 + (uint32_t)j;
                /* Past the row's end a lane contributes 0.0f, so padding is a zero code and takes
                 * no part in the amax. n is a multiple of 4, so a lane's four columns are all in
                 * the row or all outside it: the ragged edge never splits a lane. */
                v[sb][j] = c < n ? ld_bf(xr, c) : 0.0f;
            }
            float m = 0.0f;
#pragma unroll
            for (int j = 0; j < kColsPerLane; ++j) m = fmaxf(m, fabsf(v[sb][j]));
            am = fmaxf(am, r3d_xor_max<kLanes>(m));
        }
        const float sc = group_scale(am, kTopE4M3);
        /* The scale store is guarded by the group count, not just by the lane: a wave covers
         * kPer = 128/GROUP groups and a row's group count is ceil(n/group), which is not a multiple
         * of kPer when the last group is ragged. Unguarded, those lanes wrote one row past the
         * scale row -- and on a ragged n, past the plane's end. */
        if ((lane & (kLanes - 1)) == 0) {
            const uint32_t gi = (kSub > 1) ? g0 : g0 + lane / kLanes;
            if (gi < ngrp) s[(size_t)row * s_ld + gi] = sc;
        }
        const float inv = 1.0f / sc;
#pragma unroll
        for (int sb = 0; sb < kSub; ++sb) {
            const uint32_t c0 = cbase + (uint32_t)sb * kWaveCols + lane * kColsPerLane;
            if (c0 < n) *(uint32_t*)(qr + c0) = e4m3_codes4(v[sb], inv);
        }
    }
}

/* ================================================================== quant_act_i8g
 *
 * The same group core at int8, q8_group called with top 127. One scale
 * per (row, group), `sc = amax * (1.0f / 127)` -- the reciprocal MULTIPLIED, as in the fp8 path, so
 * this is quant_act_fp8 with a different codec and not quant_act_i8 at a finer granularity -- codes
 * clamp(v * (1/sc), +-127) to nearest even, stored ROW-MAJOR, and scale 1 for an all-zero group.
 *
 * The row-major store is the point of the op: libr3d's only int8 GEMM is the i8a8 one, and a dot2
 * GEMM reads a row in the order it is stored. The step-ordered plane belongs to quant_act_i8, whose
 * consumers are libr4d's WMMA GEMMs, and the two are not interchangeable.
 *
 * Same lane mapping and same ragged rules as quant_act_fp8_kernel, so the two bodies read as one
 * shape with the codec swapped.
 */
template <int GROUP>
__global__ __launch_bounds__(kThreads) R3D_WAVES(8) void quant_act_i8g_kernel(
    const uint16_t* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s, uint32_t n,
    uint32_t ngrp, uint32_t x_ld, uint32_t s_ld) {
    constexpr int kSub   = GROUP >= (int)kWaveCols ? GROUP / (int)kWaveCols : 1;
    constexpr int kPer   = GROUP < (int)kWaveCols ? (int)kWaveCols / GROUP : 1;
    constexpr int kLanes = GROUP < (int)kWaveCols ? (int)(GROUP / kColsPerLane) : (int)kWave;
    static_assert(kLanes >= 1 && kLanes <= (int)kWave, "a group spans at most one wave");

    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const uint16_t* __restrict__ xr = x + (size_t)row * x_ld;
    uint8_t* __restrict__ qr = q + (size_t)row * n;

    const uint32_t nunit = (ngrp + kPer - 1) / kPer;
    for (uint32_t u = wave; u < nunit; u += kWaves) {
        const uint32_t g0 = u * kPer, cbase = g0 * GROUP;
        float v[kSub][kColsPerLane];
        float am = 0.0f;
#pragma unroll
        for (int sb = 0; sb < kSub; ++sb) {
            const uint32_t c0 = cbase + (uint32_t)sb * kWaveCols + lane * kColsPerLane;
#pragma unroll
            for (int j = 0; j < kColsPerLane; ++j) {
                const uint32_t c = c0 + (uint32_t)j;
                v[sb][j] = c < n ? ld_bf(xr, c) : 0.0f;    /* padding is a zero code, as above */
            }
            float m = 0.0f;
#pragma unroll
            for (int j = 0; j < kColsPerLane; ++j) m = fmaxf(m, fabsf(v[sb][j]));
            am = fmaxf(am, r3d_xor_max<kLanes>(m));
        }
        const float sc = group_scale(am, kTopI8);
        if ((lane & (kLanes - 1)) == 0) {
            const uint32_t gi = (kSub > 1) ? g0 : g0 + lane / kLanes;
            if (gi < ngrp) s[(size_t)row * s_ld + gi] = sc;
        }
        const float inv = 1.0f / sc;
#pragma unroll
        for (int sb = 0; sb < kSub; ++sb) {
            const uint32_t c0 = cbase + (uint32_t)sb * kWaveCols + lane * kColsPerLane;
            if (c0 < n) *(uint32_t*)(qr + c0) = i8_codes4(v[sb], inv);   /* row-major */
        }
    }
}

}  /* namespace */

/*   x  [M][n]                 bf16, `x_ld` elements a row (0 = n)
 *   q  [M][n]                 E4M3, DENSE: the launcher takes no code pitch, so the plane is
 *                             row-major with no hole in it, rounded up to 4 bytes for the padding
 *   s  [M][ceil(n/group)]     f32,  `s_ld` elements a row (0 = ceil(n/group))
 *
 * A launcher checks what a `__global__` cannot and issues one kernel: 0, -1 a null pointer, -2 a bad
 * extent (including a group with no instantiation), -3 a pitch that does not describe the operand,
 * -4 the HIP launch failed. It allocates nothing and synchronises on nothing. */
extern "C" int r3d_quant_act_fp8(long x, long q, long s, int M, int n, int group, int x_ld,
                                 int s_ld, long stream) {
    if (!x || !q || !s) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % (int)kColsPerLane) return -3;          /* a lane owns four columns; see the kernel */
    if (group < 4 || group > 512 || (group & (group - 1))) return -2;
    const int ngrp = (n + group - 1) / group;
    if (!x_ld) x_ld = n;
    if (!s_ld) s_ld = ngrp;
    if (x_ld < n || s_ld < ngrp) return -3;

    const hipStream_t st = (hipStream_t)(uintptr_t)stream;
    const dim3 grid((unsigned)M), block(kThreads);
#define R3D_QAF(G)                                                                                  \
    R3D_KLAUNCH(R3D_KNAME(quant_act_fp8_kernel<G>), grid, block, st, (const uint16_t*)x,            \
                (uint8_t*)q, (float*)s, (uint32_t)n, (uint32_t)ngrp, (uint32_t)x_ld,                \
                (uint32_t)s_ld)
    switch (group) {
        case 4:   R3D_QAF(4);   break;
        case 8:   R3D_QAF(8);   break;
        case 16:  R3D_QAF(16);  break;
        case 32:  R3D_QAF(32);  break;
        case 64:  R3D_QAF(64);  break;
        case 128: R3D_QAF(128); break;
        case 256: R3D_QAF(256); break;
        case 512: R3D_QAF(512); break;
        default:  return -2;
    }
#undef R3D_QAF
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

namespace {

/* ================================================================== quant_act_i8
 *
 * The activation of the w4a8, w8a8 and w2a8 GEMMs, ONE SCALE PER ROW,
 * max(amax, 1e-8) / 127, codes x * (1/scale) rounded to nearest even by the int8 codec.
 *
 * The codes go out in libref's STEP ORDER, `i8_step_pos`: the middle two
 * dwords of every 16-byte step exchanged, which is why this op's n is a multiple of 16 and not of
 * 4. That order is what the op MEANS -- the w4a8, w8a8 and w2a8 GEMMs read the plane that way, and
 * kbench compares the plane element by element -- so it is kept on a card that has no WMMA, and the
 * row-major int8 plane is a separate op, quant_act_i8g below, which is the one this library's i8a8
 * GEMM consumes. The permutation is per-lane and preserves c % 4, so a lane's four codes stay
 * contiguous and the store stays one dword; a wave's 128 bytes are a permutation of the same 128.
 *
 * One divergence from the reference's arithmetic, and it is inert: libref reduces a row with
 * std::max and this with fmaxf. They agree on every input the op can see -- std::max(a, NaN) is a, so
 * the reference's amax is never a NaN, and fmaxf drops the NaN the same way; an all-NaN row has amax
 * 0 in both and scale 1e-8/127.
 *
 * ASUM, when present, also takes each row's code sums per `group` columns, group-major
 * [n/group][Mpad] with Mpad = ceil(M/16)*16: an asymmetric weight grid needs sum_k a_k to subtract
 * z_g * sum_k a_k from a group's dot product once instead of N times, and it reads whole 16-row
 * tiles. The sums are integers, so the LDS accumulator's atomic adds are exact and their order is
 * free. Rows M..Mpad-1 are not written; the caller zeroes them, as libref does.
 */
template <bool ASUM>
__global__ __launch_bounds__(kThreads) R3D_WAVES(8) void quant_act_i8_kernel(
    const uint16_t* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s,
    float* __restrict__ asum, uint32_t n, uint32_t group, uint32_t x_ld, uint32_t s_ld,
    uint32_t mpad) {
    __shared__ float red[kWaves];
    __shared__ int gs[kI8MaxGroups];

    const uint32_t row = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)row * x_ld;
    uint8_t* __restrict__ qr = q + (size_t)row * n;
    const uint32_t ngrp = ASUM ? n / group : 0;

    float am = 0.0f;
    for (uint32_t c = threadIdx.x * kColsPerLane; c < n; c += kThreads * kColsPerLane) {
#pragma unroll
        for (int j = 0; j < kColsPerLane; ++j) {
            const uint32_t cc = c + (uint32_t)j;
            if (cc < n) am = fmaxf(am, fabsf(ld_bf(xr, cc)));
        }
    }
    const float scale = fmaxf(block_max(am, red), kI8Floor) / kTopI8;
    const float inv = 1.0f / scale;
    if (threadIdx.x == 0) s[(size_t)row * s_ld] = scale;
    if (ASUM) {
        for (uint32_t g = threadIdx.x; g < ngrp; g += kThreads) gs[g] = 0;
        __syncthreads();                    /* the zeroing has to land before the atomics */
    }

    for (uint32_t c = threadIdx.x * kColsPerLane; c < n; c += kThreads * kColsPerLane) {
        uint32_t w = 0;
#pragma unroll
        for (int j = 0; j < kColsPerLane; ++j) {
            const uint32_t cc = c + (uint32_t)j;
            const int8_t code = (cc < n) ? f32_to_i8_rtne(ld_bf(xr, cc) * inv) : (int8_t)0;
            w |= (uint32_t)(uint8_t)code << (8 * j);
            /* n % group == 0 and group % 4 == 0, so a lane's four columns are in one group. The
             * sum is over column order, and the map is its own inverse, so summing the codes of
             * columns [gi*group, (gi+1)*group) here is libref reading qr[i8_step_pos(i)] --
             * same bytes, same groups. */
            if (ASUM && cc < n) atomicAdd(&gs[cc / group], (int)code);
        }
        *(uint32_t*)(qr + i8_step_pos(c)) = w;      /* the step order, one dword a lane */
    }
    if (ASUM) {
        __syncthreads();
        for (uint32_t g = threadIdx.x; g < ngrp; g += kThreads)
            asum[(size_t)g * mpad + row] = (float)gs[g];
    }
}

}  /* namespace */

/*   x     [M][n]              bf16, `x_ld` elements a row (0 = n)
 *   q      [M][n]             i8, DENSE, in libref's STEP ORDER (i8_step_pos: bits 2 and 3 of the
 *                             column exchanged), so n is a multiple of 16, not of 4
 *   s      [M]                f32, ONE scale a row, written at element pitch `s_ld` (0 = 1). A
 *                             plane with more room than rows is legal and filled in libref's flat
 *                             order -- the row's scale at s[r], the tail the caller's -- because
 *                             that is what libref does with whatever capacity it is handed, and a
 *                             byte comparison of the tail then agrees.
 *   asum   [n/group][Mpad]    f32, optional, group-major, Mpad = ceil(M/16)*16
 *
 * `group` sizes asum and is ignored without it, as libref ignores it: it is the width of the code
 * sums, not the granularity of the scale, which is the whole row. */
extern "C" int r3d_quant_act_i8(long x, long q, long s, long asum, int M, int n, int group,
                                int x_ld, int s_ld, long stream) {
    if (!x || !q || !s) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % 16) return -3;                         /* a whole 16-byte step; see i8_step_pos */
    uint32_t mpad = 0;
    if (asum) {
        if (group < (int)kColsPerLane || group % (int)kColsPerLane || n % group) return -2;
        if (n / group > (int)kI8MaxGroups) return -2;
        mpad = (uint32_t)((M + 15) / 16 * 16);
    } else {
        group = n;
    }
    if (!x_ld) x_ld = n;
    if (!s_ld) s_ld = 1;
    if (x_ld < n || s_ld < 1) return -3;

    const hipStream_t st = (hipStream_t)(uintptr_t)stream;
    const dim3 grid((unsigned)M), block(kThreads);
    if (asum)
        R3D_KLAUNCH(R3D_KNAME(quant_act_i8_kernel<true>), grid, block, st, (const uint16_t*)x,
                    (uint8_t*)q, (float*)s, (float*)asum, (uint32_t)n, (uint32_t)group,
                    (uint32_t)x_ld, (uint32_t)s_ld, mpad);
    else
        R3D_KLAUNCH(R3D_KNAME(quant_act_i8_kernel<false>), grid, block, st, (const uint16_t*)x,
                    (uint8_t*)q, (float*)s, (float*)nullptr, (uint32_t)n, (uint32_t)group,
                    (uint32_t)x_ld, (uint32_t)s_ld, mpad);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/*   x  [M][n]                 bf16, `x_ld` elements a row (0 = n)
 *   q  [M][n]                 i8, DENSE and ROW-MAJOR: the plane this library's i8a8 GEMM reads
 *   s  [M][ceil(n/group)]     f32, `s_ld` elements a row (0 = ceil(n/group))
 *
 * `group` is the scale's width and is clamped to n when absent, as libref clamps it: `group >= n`
 * and `group` absent are both one scale a row. Codes clamp to +-127 before
 * the codec, in q8_group's order. 0, -1, -2 a bad extent or an uninstantiated group, -3 a pitch
 * that does not describe the operand, -4 the HIP launch failed. */
extern "C" int r3d_quant_act_i8g(const void* x, void* q, void* s, int M, int n, int group,
                                 long long x_ld, long long s_ld, long stream) {
    if (!x || !q || !s) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % (int)kColsPerLane) return -3;          /* a lane owns four columns; see the kernel */
    if (group < 4 || group > 512 || (group & (group - 1))) return -2;
    const int ngrp = (n + group - 1) / group;
    if (!x_ld) x_ld = n;
    if (!s_ld) s_ld = ngrp;
    if (x_ld < n || s_ld < ngrp) return -3;

    const hipStream_t st = (hipStream_t)(uintptr_t)stream;
    const dim3 grid((unsigned)M), block(kThreads);
#define R3D_QAI8G(G)                                                                                \
    R3D_KLAUNCH(R3D_KNAME(quant_act_i8g_kernel<G>), grid, block, st, (const uint16_t*)x,            \
                (uint8_t*)q, (float*)s, (uint32_t)n, (uint32_t)ngrp, (uint32_t)x_ld,                \
                (uint32_t)s_ld)
    switch (group) {
        case 4:   R3D_QAI8G(4);   break;
        case 8:   R3D_QAI8G(8);   break;
        case 16:  R3D_QAI8G(16);  break;
        case 32:  R3D_QAI8G(32);  break;
        case 64:  R3D_QAI8G(64);  break;
        case 128: R3D_QAI8G(128); break;
        case 256: R3D_QAI8G(256); break;
        case 512: R3D_QAI8G(512); break;
        default:  return -2;
    }
#undef R3D_QAI8G
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

namespace {

/* ================================================================== gated_quant_fp8
 *
 * act(gate) * up, then quant_act_fp8's quantiser, in one pass. `gate_up` is
 * [M, 2n], gate in the first n and up in the second, which is what a fused gate_up projection
 * writes; the product is what gets quantised, and `out_bf16`, when bound, receives it.
 *
 * The product is narrowed to bf16 and read back before the absmax. That is the op and not an
 * optimisation: the unfused pair writes a bf16 [M, n] tensor and quantises those bytes, so the
 * codes are taken over the narrowed value and out_bf16 holds exactly the value they came from.
 * Within the product the rounding points are libref's: sg = 1/(1+exp(-gate)) in f32, then
 * (gate*sg) for silu -- MULTIPLIED, not divided, which is a different f32 -- then * up, then one
 * bf16 narrowing. `expf` is the accurate form and not __expf (r3d_common.h); it is the one value in
 * this unit the reference evaluates in double, so a code can move by one where the bf16 narrowing
 * lands on a tie. Every other number here is bit-for-bit.
 *
 * No row reduction, so no first pass and no barrier: a wave a group, a lane four columns.
 */
template <int GROUP, int ACT>
__global__ __launch_bounds__(kThreads) R3D_WAVES(8) void gated_quant_fp8_kernel(
    const uint16_t* __restrict__ gu, uint16_t* __restrict__ y, uint8_t* __restrict__ q,
    float* __restrict__ s, uint32_t n, uint32_t ngrp, uint32_t s_ld) {
    constexpr int kSub   = GROUP >= (int)kWaveCols ? GROUP / (int)kWaveCols : 1;
    constexpr int kPer   = GROUP < (int)kWaveCols ? (int)kWaveCols / GROUP : 1;
    constexpr int kLanes = GROUP < (int)kWaveCols ? (int)(GROUP / kColsPerLane) : (int)kWave;

    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const uint16_t* __restrict__ gr = gu + (size_t)row * (2 * n);
    uint16_t* __restrict__ yr = y ? y + (size_t)row * n : nullptr;
    uint8_t* __restrict__ qr = q + (size_t)row * n;

    const uint32_t nunit = (ngrp + kPer - 1) / kPer;
    for (uint32_t u = wave; u < nunit; u += kWaves) {
        const uint32_t g0 = u * kPer, cbase = g0 * GROUP;
        float v[kSub][kColsPerLane];
        float am = 0.0f;
#pragma unroll
        for (int sb = 0; sb < kSub; ++sb) {
            const uint32_t c0 = cbase + (uint32_t)sb * kWaveCols + lane * kColsPerLane;
#pragma unroll
            for (int j = 0; j < kColsPerLane; ++j) {
                const uint32_t c = c0 + (uint32_t)j;
                if (c < n) {
                    const float gt = ld_bf(gr, c);
                    const float up = ld_bf(gr, n + c);
                    const float sg = 1.0f / (1.0f + expf(-gt));
                    const uint16_t h = f32_to_bf16((ACT == R3D_ACT_SILU ? gt * sg : sg) * up);
                    if (yr) yr[c] = h;
                    v[sb][j] = bf16_to_f32(h);
                } else {
                    v[sb][j] = 0.0f;                    /* padding: see quant_act_fp8_kernel */
                }
            }
            float m = 0.0f;
#pragma unroll
            for (int j = 0; j < kColsPerLane; ++j) m = fmaxf(m, fabsf(v[sb][j]));
            am = fmaxf(am, r3d_xor_max<kLanes>(m));
        }
        const float sc = group_scale(am, kTopE4M3);
        if ((lane & (kLanes - 1)) == 0) {                       /* the ragged guard, as above */
            const uint32_t gi = (kSub > 1) ? g0 : g0 + lane / kLanes;
            if (gi < ngrp) s[(size_t)row * s_ld + gi] = sc;
        }
        const float inv = 1.0f / sc;
#pragma unroll
        for (int sb = 0; sb < kSub; ++sb) {
            const uint32_t c0 = cbase + (uint32_t)sb * kWaveCols + lane * kColsPerLane;
            if (c0 < n) *(uint32_t*)(qr + c0) = e4m3_codes4(v[sb], inv);
        }
    }
}

}  /* namespace */

/*   gu  [M][2n]               bf16, gate first then up, DENSE (the launcher takes no pitch)
 *   y   [M][n]                bf16, the product, optional, dense
 *   q   [M][n]                E4M3, dense
 *   s   [M][ceil(n/group)]    f32,  dense
 *
 * `act` is R3D_ACT_SILU (down_proj's input) or R3D_ACT_SIGMOID (a gated attention o_proj's). */
extern "C" int r3d_gated_quant_fp8(long gate_up, long q, long s, long out_bf16, int M, int n,
                                   int group, int act, long stream) {
    if (!gate_up || !q || !s) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (n % (int)kColsPerLane) return -3;
    if (act != R3D_ACT_SILU && act != R3D_ACT_SIGMOID) return -2;
    if (group < 4 || group > 512 || (group & (group - 1))) return -2;
    const int ngrp = (n + group - 1) / group;

    const hipStream_t st = (hipStream_t)(uintptr_t)stream;
    const dim3 grid((unsigned)M), block(kThreads);
#define R3D_GQ(G, A)                                                                                \
    R3D_KLAUNCH(R3D_KNAME(gated_quant_fp8_kernel<G, A>), grid, block, st, (const uint16_t*)gate_up, \
                (uint16_t*)out_bf16, (uint8_t*)q, (float*)s, (uint32_t)n, (uint32_t)ngrp,           \
                (uint32_t)ngrp)
#define R3D_GQ_ACT(A)                                                                               \
    switch (group) {                                                                                \
        case 4:   R3D_GQ(4, A);   break;                                                            \
        case 8:   R3D_GQ(8, A);   break;                                                            \
        case 16:  R3D_GQ(16, A);  break;                                                            \
        case 32:  R3D_GQ(32, A);  break;                                                            \
        case 64:  R3D_GQ(64, A);  break;                                                            \
        case 128: R3D_GQ(128, A); break;                                                            \
        case 256: R3D_GQ(256, A); break;                                                            \
        case 512: R3D_GQ(512, A); break;                                                            \
        default:  return -2;                                                                        \
    }
    /* The gate is a template argument, so the multiply below is one instruction and not a branch in
     * the inner loop; the switch is over the group widths, each with its own lane mapping. */
    if (act == R3D_ACT_SILU) { R3D_GQ_ACT(R3D_ACT_SILU); }
    else                     { R3D_GQ_ACT(R3D_ACT_SIGMOID); }
#undef R3D_GQ_ACT
#undef R3D_GQ
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

namespace {

/* ================================================================== dequant
 *
 * y = code * scale, the read of a container's own quantised weight. Elementwise,
 * so there is no reduction and no order to match; the code decodes exactly (E4M3's four exponent
 * bits and three mantissa bits are a subset of f16's, and of f32's; E5M2 is f16's top byte,
 * rad_plugin.h; int8 is exact in f32), the product is one f32 multiply per element, and the
 * store narrows it to bf16 -- the width every consumer of a dequantised weight on this card reads,
 * which is why the launcher names the codes' dtype and not the output's.
 *
 * The scale stream is [M, ceil(n/group)] and the shim requires it to carry every group, so the
 * clamp below never fires on a well-formed call; libref clamps a short stream's index to its last
 * scale instead of reading off the end, and a kernel row refuses that geometry instead.
 */
template <int QDT>
__device__ __forceinline__ float code_value(uint8_t b) {
    if (QDT == R3D_DT_U8) return e4m3_to_f32(b);
    if (QDT == R3D_DT_E5M2) return f16_to_f32((uint16_t)b << 8);
    return (float)(int8_t)b;
}

template <int QDT>
__global__ __launch_bounds__(kThreads) R3D_WAVES(8) void dequant_kernel(
    const uint8_t* __restrict__ q, const float* __restrict__ s, uint16_t* __restrict__ y,
    uint32_t n, uint32_t group, uint32_t ngrp, uint32_t s_ld, uint32_t y_ld) {
    const uint32_t row = blockIdx.x;
    const uint8_t* __restrict__ qr = q + (size_t)row * n;
    const float* __restrict__ sr = s + (size_t)row * s_ld;
    uint16_t* __restrict__ yr = y + (size_t)row * y_ld;

    for (uint32_t c = threadIdx.x * kColsPerLane; c < n; c += kThreads * kColsPerLane) {
        const uint32_t gi = min(c / group, ngrp - 1u);
        const float sc = ldg_f32(sr + gi);
        /* n is a multiple of 4 and the plane is dense, so a lane's four codes are one dword. */
        const uint32_t w = ldg_u32((const uint32_t*)(qr + c));
        /* One 8-byte store a lane: bf16x2_rne is software RTNE, because gfx11 has no
         * v_cvt_pk_bf16_f32 (r3d_common.h), and the narrowing is the store's single round. */
        uint32_t* yp = (uint32_t*)(yr + c);
        yp[0] = bf16x2_rne(code_value<QDT>((uint8_t)(w & 0xffu)) * sc,
                           code_value<QDT>((uint8_t)(w >> 8)) * sc);
        yp[1] = bf16x2_rne(code_value<QDT>((uint8_t)(w >> 16)) * sc,
                           code_value<QDT>((uint8_t)(w >> 24)) * sc);
    }
}

}  /* namespace */

/*   q  [M][n]                     codes, DENSE: E4M3, E5M2 or i8, one byte apiece
 *   s  [M][ceil(n/group)]         f32, `s_ld` elements a row (0 = ceil(n/group))
 *   y  [M][n]                     bf16, `y_ld` elements a row (0 = n)
 *
 * `y` is bf16: the launcher describes the codes' dtype and not the output's, because a dequantised
 * weight here is read by a bf16 dot2 GEMM, which is the same reason r3d.h gives this launcher one
 * dtype argument. An f32 plane is r3d_cast's, which carries two. */
extern "C" int r3d_dequant(long q, long s, long y, int M, int n, int group, int q_dt, int s_ld,
                           int y_ld, long stream) {
    if (!q || !s || !y) return -1;
    if (M <= 0 || n <= 0 || group < 1) return -2;
    if (n % (int)kColsPerLane) return -3;
    if (q_dt != R3D_DT_U8 && q_dt != R3D_DT_E5M2 && q_dt != R3D_DT_I8) return -2;
    const int ngrp = (n + group - 1) / group;
    if (!s_ld) s_ld = ngrp;
    if (!y_ld) y_ld = n;
    if (s_ld < ngrp || y_ld < n) return -3;

    const hipStream_t st = (hipStream_t)(uintptr_t)stream;
    const dim3 grid((unsigned)M), block(kThreads);
#define R3D_DQ(D)                                                                                   \
    R3D_KLAUNCH(R3D_KNAME(dequant_kernel<D>), grid, block, st, (const uint8_t*)q, (const float*)s,  \
                (uint16_t*)y, (uint32_t)n, (uint32_t)group, (uint32_t)ngrp, (uint32_t)s_ld,         \
                (uint32_t)y_ld)
    switch (q_dt) {
        case R3D_DT_U8:   R3D_DQ(R3D_DT_U8);   break;
        case R3D_DT_E5M2: R3D_DQ(R3D_DT_E5M2); break;
        case R3D_DT_I8:   R3D_DQ(R3D_DT_I8);   break;
        default: return -2;
    }
#undef R3D_DQ
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ================================================================== the plugin ABI entry points
 *
 * Three things every shim here reads off the operands and not off the parameters:
 *
 *   * the dtype. A row's `dtype` constraint is a necessary condition on the geometry, not a promise
 *     about the tensors, and an f32 activation read as bf16 is a value that passes every other
 *     check.
 *   * the row count. `M` is the band's upper bound, so the args carry a bound and the operand
 *     carries the call.
 *   * the pitch. r3d_rows_pitch() answers in elements, or -1 for an inner axis that is strided.
 *
 * The code planes are dense: none of the four launchers takes a code pitch, because a dot2 GEMM
 * reads codes row-major and the engine hands them over in the order it stores them. A plane with a
 * hole in it is a stride refusal and not a slow path. */

/* bf16 is the only activation width these kernels read. An f32 activation is r3d_cast's problem,
 * which takes the dtypes as arguments; a quantiser with a dtype switch in its inner loop is the
 * wrong shape for an op whose whole point is the store. */
static inline int dt_bf16(const RadTensor* t) { return r3d_is_bf16(t) ? RAD_OK : RAD_E_DTYPE; }
static inline int dt_f32(const RadTensor* t)  { return r3d_is_f32(t)  ? RAD_OK : RAD_E_DTYPE; }
/* The codes' dtype picks the codec, so it is checked exactly and never as "some byte". */
static inline int dt_e4m3(const RadTensor* t) { return r3d_is_e4m3(t) ? RAD_OK : RAD_E_DTYPE; }
static inline int dt_i8(const RadTensor* t)   { return r3d_is_i8(t)   ? RAD_OK : RAD_E_DTYPE; }

#define R3D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R3D_NO(_s); } while (0)

/* A launcher checked the geometry and answered with a code; translate it to a RAD status. */
static inline int r3d_launch_status(int e) {
    switch (e) {
        case  0: return RAD_OK;
        case -1: return RAD_E_INVAL;
        case -2: return RAD_E_SHAPE;
        case -3: return RAD_E_STRIDE;
        default: return RAD_E_DEVICE;
    }
}

/* The group must be a power of two from 4 to 512: the amax butterfly is over the lanes covering a
 * group, GROUP/4 of them, and the kernels are instantiated for those widths. libref clamps
 * `group <= 0 || group > n` to n, so "one scale a row" arrives as 0 and leaves as n;
 * a width with no instantiation is refused and not rounded, because a rounded group would quantise
 * on a grid the weight's scales are not on. */
static inline int r3d_group_of(long long g, long long n) {
    if (g <= 0 || g > n) g = n;
    return (g >= 4 && g <= 512 && (g & (g - 1)) == 0) ? (int)g : 0;
}

/* ------------------------------------------------------------------ quant_act_fp8 */

enum { QAF_X = 0, QAF_Q, QAF_S };

extern "C" int r3d_rad_quant_act_fp8(const RadArgs* a, RadStream stream) {
    const RadTensor* x  = r3d_opt(a, QAF_X);
    const RadTensor* q  = r3d_opt(a, QAF_Q);
    const RadTensor* sc = r3d_opt(a, QAF_S);
    if (!x || !q || !sc) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(x));
    R3D_NEED(dt_e4m3(q));
    R3D_NEED(dt_f32(sc));

    const long long n = r3d_geti_or(a, "n", r3d_dim(x, (int)x->rank - 1));
    if (n <= 0 || n % kColsPerLane) return R3D_NO(RAD_E_SHAPE);
    const int group = r3d_group_of(r3d_geti_or(a, "group", R3D_ACT_GROUP), n);
    if (!group) return R3D_NO(RAD_E_SHAPE);
    const long long ngrp = (n + group - 1) / group;

    long long rows = 0, qrows = 0, srows = 0;
    const long long x_ld = r3d_rows_pitch(x, n, &rows);
    const long long q_ld = r3d_rows_pitch(q, n, &qrows);
    const long long s_ld = r3d_rows_pitch(sc, ngrp, &srows);
    if (x_ld < 0 || s_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (rows <= 0 || rows > 0x7fffffffLL || n > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);
    if (q_ld != n || qrows < rows || r3d_numel(q) < rows * n) return R3D_NO(RAD_E_SHAPE);
    if (srows < rows || r3d_numel(sc) < rows * ngrp) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_aligned(r3d_p(q), 4) || !r3d_aligned(r3d_p(sc), 4)) return R3D_NO(RAD_E_ALIGN);

    return r3d_launch_status(r3d_quant_act_fp8(r3d_addr(x), r3d_addr(q), r3d_addr(sc), (int)rows,
                                              (int)n, group, (int)x_ld, (int)s_ld,
                                              (long)(uintptr_t)stream));
}

/* ------------------------------------------------------------------ quant_act_i8 */

enum { QAI_X = 0, QAI_Q, QAI_S, QAI_ASUM };

extern "C" int r3d_rad_quant_act_i8(const RadArgs* a, RadStream stream) {
    const RadTensor* x  = r3d_opt(a, QAI_X);
    const RadTensor* q  = r3d_opt(a, QAI_Q);
    const RadTensor* sc = r3d_opt(a, QAI_S);
    const RadTensor* as = r3d_opt(a, QAI_ASUM);      /* the asymmetric grid's code sums */
    if (!x || !q || !sc) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(x));
    R3D_NEED(dt_i8(q));
    R3D_NEED(dt_f32(sc));
    if (r3d_has(as)) R3D_NEED(dt_f32(as));   /* a slot with no bytes is the operand being absent */

    const long long n = r3d_geti_or(a, "n", r3d_dim(x, (int)x->rank - 1));
    if (n <= 0 || n % 16) return R3D_NO(RAD_E_SHAPE);   /* a whole 16-byte step: see i8_step_pos */

    /* One scale a row: this op's scale granularity IS the row (the amax runs over the whole row,
     * and the store is `stt(s, rowoff(s, r, 1), scale)`, a FLAT stream at element pitch 1), and
     * libref lists `group` as the width of the row SUMS, not of the scale. A plane with more
     * elements than rows is therefore accepted and filled in libref's order: the row's scale
     * lands at s[r], the plane's first M elements, and the rest of it stays the caller's. That
     * is exactly the plane the row's operand description gives --
     * [M, ceil(n/group)], quant_act_i8g's width, of which this op writes the first column -- and
     * refusing it would be the shim tightening its own row past the reference: libref requires
     * numel(s) >= rows and nothing more, and the bytes it never touches compare equal. */

    long long rows = 0, qrows = 0, srows = 0;
    const long long x_ld = r3d_rows_pitch(x, n, &rows);
    const long long q_ld = r3d_rows_pitch(q, n, &qrows);
    const long long s_ld = r3d_rows_pitch(sc, 1, &srows);
    if (x_ld < 0 || s_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (rows <= 0 || rows > 0x7fffffffLL || n > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);
    if (q_ld != n || qrows < rows || r3d_numel(q) < rows * n) return R3D_NO(RAD_E_SHAPE);
    if (srows < rows || r3d_numel(sc) < rows) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_aligned(r3d_p(q), 4) || !r3d_aligned(r3d_p(sc), 4)) return R3D_NO(RAD_E_ALIGN);

    /* `group` sizes asum only, and is not read without it. n % group == 0, because a sum is over
     * whole columns of a row, and the LDS accumulator is 256 groups wide. */
    int group = 0;
    if (as) {
        group = (int)r3d_geti_or(a, "group", R3D_W8_GROUP);
        if (group < (int)kColsPerLane || group % kColsPerLane || n % group)
            return R3D_NO(RAD_E_SHAPE);
        if (n / group > (long long)kI8MaxGroups) return R3D_NO(RAD_E_SHAPE);
        if (!r3d_contig(as)) return R3D_NO(RAD_E_STRIDE);
        const long long mpad = ((rows + 15) / 16) * 16;
        if (r3d_numel(as) < (n / group) * mpad) return R3D_NO(RAD_E_SHAPE);
    }

    return r3d_launch_status(r3d_quant_act_i8(
        r3d_addr(x), r3d_addr(q), r3d_addr(sc), r3d_addr(as), (int)rows, (int)n, group, (int)x_ld,
        (int)s_ld, (long)(uintptr_t)stream));
}

/* ------------------------------------------------------------------ quant_act_i8g */

/* The int8 plane at group granularity, and the one this library's i8a8 GEMM reads. Operands are
 * libref's oQuantFp8 shape -- {x, q, scale} -- with the int8 codes' dtype. */
enum { QAG_X = 0, QAG_Q, QAG_S };

extern "C" int r3d_rad_quant_act_i8g(const RadArgs* a, RadStream stream) {
    const RadTensor* x  = r3d_opt(a, QAG_X);
    const RadTensor* q  = r3d_opt(a, QAG_Q);
    const RadTensor* sc = r3d_opt(a, QAG_S);
    if (!x || !q || !sc) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(x));
    R3D_NEED(dt_i8(q));
    R3D_NEED(dt_f32(sc));

    const long long n = r3d_geti_or(a, "n", r3d_dim(x, (int)x->rank - 1));
    if (n <= 0 || n % kColsPerLane) return R3D_NO(RAD_E_SHAPE);
    /* libref's clamp of `group`: absent, zero or wider than the row is one
     * scale a row, so the default here is 0 and the clamp turns it into n. A width the kernels have
     * no instantiation for is refused, not rounded to one. */
    const int group = r3d_group_of(r3d_geti_or(a, "group", 0), n);
    if (!group) return R3D_NO(RAD_E_SHAPE);
    const long long ngrp = (n + group - 1) / group;

    long long rows = 0, qrows = 0, srows = 0;
    const long long x_ld = r3d_rows_pitch(x, n, &rows);
    const long long q_ld = r3d_rows_pitch(q, n, &qrows);
    const long long s_ld = r3d_rows_pitch(sc, ngrp, &srows);
    if (x_ld < 0 || s_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (rows <= 0 || rows > 0x7fffffffLL || n > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);
    if (q_ld != n || qrows < rows || r3d_numel(q) < rows * n) return R3D_NO(RAD_E_SHAPE);
    if (srows < rows || r3d_numel(sc) < rows * ngrp) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_aligned(r3d_p(q), 4) || !r3d_aligned(r3d_p(sc), 4)) return R3D_NO(RAD_E_ALIGN);

    return r3d_launch_status(r3d_quant_act_i8g(x->data, q->data, sc->data, (int)rows, (int)n, group,
                                              x_ld, s_ld, (long)(uintptr_t)stream));
}

/* ------------------------------------------------------------------ gated_quant_fp8 */

enum { GQT_GU = 0, GQT_Q, GQT_S, GQT_Y };

extern "C" int r3d_rad_gated_quant_fp8(const RadArgs* a, RadStream stream) {
    const RadTensor* gu = r3d_opt(a, GQT_GU);
    const RadTensor* q  = r3d_opt(a, GQT_Q);
    const RadTensor* sc = r3d_opt(a, GQT_S);
    const RadTensor* y  = r3d_opt(a, GQT_Y);       /* the product, for a bf16 consumer */
    if (!gu || !q || !sc) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(gu));
    R3D_NEED(dt_e4m3(q));
    R3D_NEED(dt_f32(sc));
    if (r3d_has(y)) R3D_NEED(dt_bf16(y));   /* the same: an empty slot is no operand at all */

    /* "silu" (down_proj's input) or "sigmoid" (a gated attention o_proj's), `mode` the integer
     * spelling of the same choice, `act` winning where both are given. libref's `act` enumerates
     * those two, so a gelu-gated feed-forward is not this row's: it asks for a gate the oracle has
     * no definition for here. */
    const int act = r3d_act_code(a, "act", "mode", R3D_ACT_SILU);
    if (act < 0) return R3D_NO(RAD_E_UNSUPPORTED);

    const long long n = r3d_geti_or(a, "n", r3d_dim(q, (int)q->rank - 1));
    if (n <= 0 || n % kColsPerLane) return R3D_NO(RAD_E_SHAPE);
    const int group = r3d_group_of(r3d_geti_or(a, "group", R3D_ACT_GROUP), n);
    if (!group) return R3D_NO(RAD_E_SHAPE);
    const long long ngrp = (n + group - 1) / group;

    /* The fused gate_up projection writes the plane dense, gate half first, [M, 2n] with no hole: a
     * sliced gate is r3d_act_mul_bf16's, which takes the two pitches. */
    long long rows = 0, qrows = 0, srows = 0, yrows = 0;
    const long long gu_ld = r3d_rows_pitch(gu, 2 * n, &rows);
    const long long q_ld  = r3d_rows_pitch(q, n, &qrows);
    const long long s_ld  = r3d_rows_pitch(sc, ngrp, &srows);
    const long long y_ld  = y ? r3d_rows_pitch(y, n, &yrows) : n;
    if (gu_ld < 0 || q_ld < 0 || s_ld < 0 || y_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (rows <= 0 || rows > 0x7fffffffLL || n > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);
    if (gu_ld != 2 * n) return R3D_NO(RAD_E_SHAPE);
    if (q_ld != n || qrows < rows || r3d_numel(q) < rows * n) return R3D_NO(RAD_E_SHAPE);
    if (s_ld != ngrp || srows < rows || r3d_numel(sc) < rows * ngrp) return R3D_NO(RAD_E_SHAPE);
    if (y && (y_ld != n || yrows < rows || r3d_numel(y) < rows * n)) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_aligned(r3d_p(q), 4) || !r3d_aligned(r3d_p(sc), 4)) return R3D_NO(RAD_E_ALIGN);

    return r3d_launch_status(r3d_gated_quant_fp8(
        r3d_addr(gu), r3d_addr(q), r3d_addr(sc), r3d_addr(y), (int)rows, (int)n, group, act,
        (long)(uintptr_t)stream));
}

/* ------------------------------------------------------------------ dequant */

enum { DQT_Q = 0, DQT_S, DQT_Y };

extern "C" int r3d_rad_dequant(const RadArgs* a, RadStream stream) {
    const RadTensor* q  = r3d_opt(a, DQT_Q);
    const RadTensor* sc = r3d_opt(a, DQT_S);
    const RadTensor* y  = r3d_opt(a, DQT_Y);
    if (!q || !sc || !y) return R3D_NO(RAD_E_INVAL);
    /* The codes' dtype IS the encoding, so it is read off the tensor: E4M3 and E5M2 are one byte
     * apiece and nothing in the bytes tells them apart. A plain u8 plane carries no encoding, so it
     * is not accepted under the E4M3 code. */
    if (q->dtype == RAD_U8) return R3D_NO(RAD_E_DTYPE);
    const int q_dt = r3d_dt_code(q->dtype);
    if (q_dt != R3D_DT_U8 && q_dt != R3D_DT_E5M2 && q_dt != R3D_DT_I8)
        return R3D_NO(RAD_E_DTYPE);
    R3D_NEED(dt_f32(sc));
    R3D_NEED(dt_bf16(y));                           /* the launcher has no output dtype; see above */

    const long long n = r3d_geti_or(a, "n", r3d_dim(y, (int)y->rank - 1));
    if (n <= 0 || n % kColsPerLane) return R3D_NO(RAD_E_SHAPE);

    /* Rows come off `y`, as in ref_dequant: `y` is the plane the consumer reads. */
    long long rows = 0, qrows = 0, srows = 0;
    const long long y_ld = r3d_rows_pitch(y, n, &rows);
    const long long q_ld = r3d_rows_pitch(q, n, &qrows);
    if (y_ld < 0 || q_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (rows <= 0 || rows > 0x7fffffffLL || n > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);
    if (q_ld != n || qrows < rows || r3d_numel(q) < rows * n) return R3D_NO(RAD_E_SHAPE);

    /* The scale stream's extent gives the group count and `group` each group's width, as libref
     * reads them; with no `group`, the stream is taken as one scale per group of ceil(n/ngrp). */
    const long long ngrp_param = r3d_numel(sc) / rows > 0 ? r3d_numel(sc) / rows : 1;
    long long group = r3d_geti_or(a, "group", 0);
    if (group <= 0) group = (n + ngrp_param - 1) / ngrp_param;
    if (group < 1 || group > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);
    const long long ngrp = (n + group - 1) / group;
    const long long s_ld = r3d_rows_pitch(sc, ngrp, &srows);
    if (s_ld < 0 || srows < rows || r3d_numel(sc) < rows * ngrp) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_aligned(r3d_p(q), 4) || !r3d_aligned(r3d_p(sc), 4) || !r3d_aligned(r3d_p(y), 8))
        return R3D_NO(RAD_E_ALIGN);

    return r3d_launch_status(r3d_dequant(r3d_addr(q), r3d_addr(sc), r3d_addr(y), (int)rows, (int)n,
                                        (int)group, q_dt, (int)s_ld, (int)y_ld,
                                        (long)(uintptr_t)stream));
}
