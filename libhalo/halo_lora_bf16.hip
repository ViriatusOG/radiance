/* Derived from libr4d/r4d_lora_bf16.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
/* halo_lora_bf16.hip -- runtime LoRA's two halves, lora_shrink and lora_expand, per-row slot.
 *
 *     lora_shrink   t[m, j]  = sum_k x(m, k) * A[s, j, k]                    s = slot[m]
 *     lora_expand   y[m, n]  = y[m, n] + scale[s, n] * sum_j t[m, j] * B[s, n, j]
 *
 * The core issues the pair right after a targeted trunk GEMM, in the GEMM's own row slice, with
 * `a`/`a_scale` the GEMM's activation operands and `y` its output (docs/OPS.md). A row whose slot
 * is outside [0, S) is a base row and both kernels leave it before touching anything: shrink
 * writes no `t`, expand neither reads nor writes `y` (`y + 0` would turn -0 into +0), so a base
 * row stays bit for bit what the GEMM wrote.
 *
 * Invariants every row here keeps:
 *   - No split that moves with M: a row's arithmetic does not depend on the batch it arrives in,
 *     so a request alone and the same request in a mixed batch get the same bytes.
 *   - The grid is M and the declared parameters. Which adapters are active is read on the device
 *     from `slot` and the pools, so a recorded pass replays the same packets whatever they hold.
 *   - x(m, k) is gemm_nt_q's activation value: a code times its group's scale (one f32 product),
 *     or a bf16/f16/f32 element as it stands, the scale stream's grid read as libref's make_scales
 *     reads it. The decodes below are the ABI's conversions (rad_plugin.h) restated for the device
 *     bit for bit; gfx1201 has no bf16 convert, so the narrowing is the ABI's RTNE with its
 *     quiet-NaN rule. Every multiply-add rounds twice (-ffp-contract=off, RadAddPlugin.cmake).
 *
 * The scalar rows (lora_shrink_scalar, lora_expand_scalar): shrink is one block a row, one wave a
 * column j, lane l summing the k that are l mod 32 in ascending order, then a fixed xor butterfly;
 * expand is one thread an (m, n), j ascending, so its bytes are libref's. No scratch and no LDS. They
 * serve every dtype, and are the only rows for f16 and f32 activations.
 *
 * The tiled rows (lora_shrink_wmma, lora_shrink_wmma_q, lora_expand_tiled), preferred:
 *   - The grid's unit is a row tile (16 rows shrink, 32 expand). Every wave reads the tile's slots
 *     and a ballot decides; a tile of base rows leaves before any barrier or operand load. A tile
 *     holding rows of several adapters runs one pass per distinct slot, ascending, each with its
 *     own slot's pools; a row's value comes from its own slot's pass alone. The passes may be
 *     spread over blockIdx.z when the grid is small, which decides which block runs a pass, never
 *     its arithmetic.
 *   - Shrink is WMMA bf16 (16 x 16 x 16) with the pool fragment as the A operand and x as B, so a
 *     lane's accumulator is one row. Its order is fixed by K and R alone: K in 32-wide steps, a
 *     block's steps split over its 4 waves (wave w takes every 4th), the waves summed 0..3 in LDS,
 *     and for a long K the steps split over KS blocks whose f32 partials the last to arrive sums
 *     0..KS-1 (sh_plan). fp8 and i8 codes are exact in bf16 and feed the WMMA as codes; each step's
 *     partial is scaled by its group's f32 scale, which needs the group to be whole steps (a
 *     multiple of 32, or one group across K). Every rounding is a sum or product of the contract's
 *     terms, so the result is inside the scalar row's (K + 2) 2^-24 sum|terms| bound.
 *   - Expand is the scalar arithmetic in a tile: B[s]'s [128 columns, 32 j] slab is staged in LDS
 *     as f32 (exact) and reused by the tile's 32 rows; each output keeps the scalar sequence (acc =
 *     0, j ascending acc = acc + t*B, d = scale*acc, one rounding into y), so its bytes are
 *     libref's.
 *
 * Kernels are written against halo_launch.h so tests/halo_emu runs this source on the host.
 */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <utility>

#include "halo_plugin.h"
#include "halo_launch.h"
#include "halo_wmma.h"          /* v16bf, halo::mma: the gfx1151 fragment layer */
#include "rad_plugin.h"

namespace halo_lora {

/* What `a` holds; a template argument so the decode is chosen once per launch, not per element. */
enum { kBF16 = 0, kF16 = 1, kF32 = 2, kE4M3 = 3, kE5M2 = 4, kI8 = 5 };

struct ShrinkArgs {
    const void*     a;    long long a_rs, a_cs;        /* element strides of `a` */
    const float*    as;   long long s_rs, s_cs;        /* null: no scale stream */
    const int32_t*  slot; long long slot_st;
    const uint16_t* la;   long long la_s0, la_s1, la_s2;
    float*          t;    long long t_rs, t_cs;
    int             K, group, R, S;                  /* K < 2^31: 32-bit k and k / group */
};

struct ExpandArgs {
    const float*    t;    long long t_rs, t_cs;
    const int32_t*  slot; long long slot_st;
    const uint16_t* lb;   long long lb_s0, lb_s1, lb_s2;
    const float*    sc;   long long sc_rs, sc_cs;
    void*           y;    long long y_rs, y_cs;
    long long       N, ntiles;
    int             R, S;
};

/* ------------------------------------------------------- the ABI's conversions, on the device */
__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float((unsigned)h << 16); }

__device__ __forceinline__ uint16_t f2bf(float f) {           /* rad_f32_to_bf16 */
    const unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__device__ inline float f16_to_f32(uint16_t h) {             /* rad_f16_to_f32 */
    const unsigned s = (unsigned)(h >> 15) << 31;
    const unsigned e = ((unsigned)h >> 10) & 0x1fu;
    unsigned m = (unsigned)h & 0x3ffu, u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x400u)) { m <<= 1; ++sh; }
            m &= 0x3ffu;
            u = s | ((unsigned)(127 - 15 - sh + 1) << 23) | (m << 13);
        }
    } else if (e == 0x1fu) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e - 15u + 127u) << 23) | (m << 13);
    }
    return __uint_as_float(u);
}

__device__ inline uint16_t f32_to_f16(float f) {             /* rad_f32_to_f16 */
    const unsigned u = __float_as_uint(f);
    const unsigned s = (u >> 16) & 0x8000u, a = u & 0x7fffffffu;
    if (a >= 0x7f800000u)
        return (uint16_t)(s | 0x7c00u | (a > 0x7f800000u ? ((a >> 13) & 0x3ffu) | 0x200u : 0u));
    if (a >= 0x47800000u) return (uint16_t)(s | 0x7c00u);
    if (a < 0x33000000u) return (uint16_t)s;
    const int e = (int)(a >> 23) - 127;
    unsigned m = a & 0x7fffffu;
    if (e < -14) {
        const int sh = -14 - e + 13;
        m |= 0x800000u;
        const unsigned lo = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        unsigned q = m >> sh;
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return (uint16_t)(s | q);
    }
    unsigned q = m >> 13, he = (unsigned)(e + 15);
    const unsigned lo = m & 0x1fffu;
    if (lo > 0x1000u || (lo == 0x1000u && (q & 1u))) ++q;
    if (q == 0x400u) { q = 0; ++he; }
    if (he >= 0x1fu) return (uint16_t)(s | 0x7c00u);
    return (uint16_t)(s | (he << 10) | q);
}

__device__ inline float e4m3_to_f32(uint8_t b) {             /* rad_fp8e4m3_to_f32 */
    const unsigned s = (unsigned)(b >> 7) << 31;
    const unsigned e = ((unsigned)b >> 3) & 0xfu;
    unsigned m = (unsigned)b & 0x7u, u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x8u)) { m <<= 1; ++sh; }
            m &= 0x7u;
            u = s | ((unsigned)(127 - 6 - sh) << 23) | (m << 20);
        }
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    return __uint_as_float(u);
}

template <int ADT>
__device__ __forceinline__ float ld_act(const void* p, long long i) {
    if (ADT == kBF16) return bf2f(((const uint16_t*)p)[i]);
    if (ADT == kF16)  return f16_to_f32(((const uint16_t*)p)[i]);
    if (ADT == kF32)  return ((const float*)p)[i];
    if (ADT == kE4M3) return e4m3_to_f32(((const uint8_t*)p)[i]);
    if (ADT == kE5M2) return f16_to_f32((uint16_t)((unsigned)((const uint8_t*)p)[i] << 8));
    return (float)((const int8_t*)p)[i];
}

/* The sum of a wave's 32 values, the same on every lane and in a fixed order: an xor butterfly over
 * ds_bpermute. Each step adds the same two values on both partners (a + b == b + a), so the lanes
 * never drift apart and no lane's answer depends on the schedule. */
__device__ __forceinline__ float wave_sum(float v, int lane) {
    for (int off = 16; off >= 1; off >>= 1) {
        const int o = __builtin_amdgcn_ds_bpermute((lane ^ off) << 2, __float_as_int(v));
        v = v + __int_as_float(o);
    }
    return v;
}

/* ---------------------------------------------------------------- the kernels */

/* One block a row, eight waves; wave w owns columns j = w, w + 8, ... The x row is re-read per
 * column from cache rather than staged: K reaches 17408, past what LDS holds as f32. */
template <int ADT>
__global__ __launch_bounds__(256) void lora_shrink_kernel(ShrinkArgs A) {
    const long long m = blockIdx.x;
    const int s = A.slot[m * A.slot_st];
    if (s < 0 || s >= A.S) return;                   /* a base row: no t written */
    const int lane = (int)(threadIdx.x & 31u), wave = (int)(threadIdx.x >> 5);
    const long long x0 = m * A.a_rs, s0 = m * A.s_rs;
    for (int j = wave; j < A.R; j += 8) {
        const uint16_t* arow = A.la + (long long)s * A.la_s0 + (long long)j * A.la_s1;
        float acc = 0.0f;
        for (int k = lane; k < A.K; k += 32) {
            float x = ld_act<ADT>(A.a, x0 + (long long)k * A.a_cs);
            if (A.as) x = x * A.as[s0 + (long long)(k / A.group) * A.s_cs];
            acc += x * bf2f(arow[(long long)k * A.la_s2]);
        }
        acc = wave_sum(acc, lane);
        if (lane == 0) A.t[m * A.t_rs + (long long)j * A.t_cs] = acc;
    }
}

/* One thread an output element over a flat grid of M row-strips of 256 columns. */
template <int YDT>
__global__ __launch_bounds__(256) void lora_expand_kernel(ExpandArgs A) {
    const long long b = blockIdx.x;
    const long long m = b / A.ntiles;
    const long long n = (b - m * A.ntiles) * 256 + threadIdx.x;
    if (n >= A.N) return;
    const int s = A.slot[m * A.slot_st];
    if (s < 0 || s >= A.S) return;                   /* a base row: y not read, not written */
    const float* trow = A.t + m * A.t_rs;
    const uint16_t* brow = A.lb + (long long)s * A.lb_s0 + n * A.lb_s1;
    float acc = 0.0f;
    for (int j = 0; j < A.R; ++j)
        acc += trow[(long long)j * A.t_cs] * bf2f(brow[(long long)j * A.lb_s2]);
    const float d = A.sc[(long long)s * A.sc_rs + n * A.sc_cs] * acc;
    const long long o = m * A.y_rs + n * A.y_cs;
    if (YDT == kBF16) {
        uint16_t* y = (uint16_t*)A.y + o;
        *y = f2bf(bf2f(*y) + d);
    } else if (YDT == kF16) {
        uint16_t* y = (uint16_t*)A.y + o;
        *y = f32_to_f16(f16_to_f32(*y) + d);
    } else {
        float* y = (float*)A.y + o;
        *y = *y + d;
    }
}


/* ---------------------------------------------------------------- operand reading (host) */

/* `t` read as [rows, cols]: the trailing axes that span `cols` must be one dense run, the leading
 * ones fold to rows, and the row stride is the axis just outside the run -- libref/ref_gemm.h's
 * row_stride, so a rank-3 [tokens, heads, width] activation and a column slice of a fused plane are
 * read the way the oracle reads them. False when no single (rows, row stride, column stride)
 * describes the tensor. */
static bool as_rows(const RadTensor* t, int64_t cols, int64_t* rows, int64_t* rs, int64_t* cs) {
    if (!t || t->rank < 1 || cols <= 0) return false;
    const int r = (int)t->rank;
    int i = r - 1;
    int64_t span = t->shape[i];
    while (span < cols && i > 0) {
        if (t->stride[i - 1] != t->stride[i] * t->shape[i]) return false;
        --i;
        span *= t->shape[i];
    }
    if (span != cols) return false;
    int64_t n = 1;
    for (int d = 0; d < i; ++d) n *= t->shape[d];
    for (int d = 0; d + 1 < i; ++d)
        if (t->shape[d + 1] > 1 && t->stride[d] != t->stride[d + 1] * t->shape[d + 1]) return false;
    *rows = n;
    *rs = i > 0 ? t->stride[i - 1] : cols;
    *cs = t->stride[r - 1];
    return true;
}

/* The per-row slot array: [M], or [M, 1] -- its length and element stride. */
static bool as_vec(const RadTensor* t, int64_t* n, int64_t* st) {
    if (!t) return false;
    if (t->rank == 1) { *n = t->shape[0]; *st = t->stride[0]; return true; }
    int64_t cs = 0;
    return as_rows(t, 1, n, st, &cs);
}

/* A pool [S, rows, cols]: its three strides; anything but rank 3 is read dense. */
static void pool3(const RadTensor* t, int64_t rows, int64_t cols, long long* s0, long long* s1,
                  long long* s2) {
    if (t->rank == 3) { *s0 = t->stride[0]; *s1 = t->stride[1]; *s2 = t->stride[2]; return; }
    *s0 = rows * cols; *s1 = cols; *s2 = 1;
}

/* libref/ref_gemm.h's grid_block: the power of two whose ceiling division gives `blocks`. */
static int64_t grid_block(int64_t ext, int64_t blocks) {
    if (ext <= 0) return 1;
    if (blocks <= 1) return ext;
    const int64_t lo = (ext + blocks - 1) / blocks;
    int64_t b = 1;
    while (b < lo) b <<= 1;
    return (ext + b - 1) / b == blocks ? b : lo;
}

/* The dtype parameter is the tensor's own name (rad_dtype_name); a row resolved for one dtype and
 * handed another would compute the wrong decode, so a disagreement is refused, not followed. */
static bool dtype_agrees(const RadArgs* a, uint32_t dt) {
    const char* d = rad_args_gets(a, "dtype");
    return !d || rad_dtype_parse(d) == dt;
}

/* The scalar rows' launches, one per dtype instantiation: M blocks for shrink and M strips of 256
 * columns for expand, 256 threads each, whatever the slots hold. */
template <int ADT>
static void launch_shrink(const ShrinkArgs& P, int64_t blocks, RadStream stream) {
    (void)stream;                                 /* the host emulator's launch takes none */
    HALO_KLAUNCH(HALO_KNAME(lora_shrink_kernel<ADT>), dim3((unsigned)blocks), dim3(256),
                (hipStream_t)stream, P);
}
template <int YDT>
static void launch_expand(const ExpandArgs& P, int64_t blocks, RadStream stream) {
    (void)stream;
    HALO_KLAUNCH(HALO_KNAME(lora_expand_kernel<YDT>), dim3((unsigned)blocks), dim3(256),
                (hipStream_t)stream, P);
}

/* lora_shrink's operands, read and checked once for every row: on RAD_OK, `P` and `adt` describe
 * the launch over *M > 0 rows. Zero rows is refused, as every op refuses it (docs/OPS.md). */
static int shrink_operands(const RadArgs* a, ShrinkArgs* out, int* adt_out, int64_t* M_out) {
    *M_out = 0;
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const RadTensor *X = rad_arg_in(a, 0), *XS = rad_arg_in(a, 1), *SL = rad_arg_in(a, 2);
    const RadTensor *LA = rad_arg_in(a, 3), *T = rad_arg_in(a, 4);
    if (!X || !SL || !LA || !T) return RAD_E_INVAL;
    if (XS && XS->dtype == RAD_DT_INVALID) XS = nullptr;     /* absent, as libref */
    const int64_t K = rad_args_geti_or(a, "K", 0), R = rad_args_geti_or(a, "R", 0);
    const int64_t S = rad_args_geti_or(a, "S", 0);
    if (K <= 0 || K > 0x7fffffff || R <= 0 || S <= 0 || R > (1 << 16) || S > (1 << 20))
        return RAD_E_SHAPE;
    int adt;
    switch (X->dtype) {
        case RAD_BF16:   adt = kBF16; break;
        case RAD_F16:    adt = kF16;  break;
        case RAD_F32:    adt = kF32;  break;
        case RAD_F8E4M3: adt = kE4M3; break;
        case RAD_F8E5M2: adt = kE5M2; break;
        case RAD_I8:     adt = kI8;   break;
        default:         return RAD_E_DTYPE;
    }
    if (!dtype_agrees(a, X->dtype)) return RAD_E_DTYPE;
    if (adt >= kE4M3 && !XS) return RAD_E_INVAL;     /* a code without its scale has no value */
    if (LA->dtype != RAD_BF16 || T->dtype != RAD_F32 || SL->dtype != RAD_I32) return RAD_E_DTYPE;

    int64_t M = 0, t_rs = 0, t_cs = 0, xr = 0, x_rs = 0, x_cs = 0, sr = 0, sl_st = 0;
    if (!as_rows(T, R, &M, &t_rs, &t_cs)) return RAD_E_SHAPE;     /* the step's rows: t's view */
    if (M <= 0) return RAD_E_SHAPE;                               /* zero rows is not a call */
    const int64_t band = rad_args_geti_or(a, "M", 0);             /* libref's band_rows rule */
    if (band > 0 && M > band) return RAD_E_SHAPE;
    if (!as_rows(X, K, &xr, &x_rs, &x_cs) || xr < M) return RAD_E_SHAPE;
    if (!as_vec(SL, &sr, &sl_st) || sr < M) return RAD_E_SHAPE;
    if (rad_tensor_numel(LA) < S * R * K) return RAD_E_SHAPE;
    if (M > 0x7fffffff) return RAD_E_SHAPE;

    ShrinkArgs P{};
    P.a = X->data; P.a_rs = x_rs; P.a_cs = x_cs;
    if (XS) {
        /* gemm_nt_q's reading of the stream: [rows, groups] with its own strides, or a flat
         * [rows * groups]; the group is what the stream's extent implies. A grid with fewer rows
         * than the activation is a weight's block grid, never an activation's. */
        if (XS->dtype != RAD_F32) return RAD_E_DTYPE;
        int64_t groups = 0;
        if (XS->rank >= 2) {
            if (XS->shape[XS->rank - 2] < M) return RAD_E_SHAPE;
            groups = XS->shape[XS->rank - 1];
            P.s_rs = XS->stride[XS->rank - 2];
            P.s_cs = XS->stride[XS->rank - 1];
        } else {
            const int64_t nel = rad_tensor_numel(XS);
            if (nel < M) return RAD_E_SHAPE;
            groups = nel / M;
            P.s_rs = groups;
            P.s_cs = 1;
        }
        if (groups <= 0) return RAD_E_SHAPE;
        const int64_t group = grid_block(K, groups);
        const int64_t gp = rad_args_geti_or(a, "group", 0);         /* the grid, as libref */
        if (gp > 0 && (K + gp - 1) / gp != (K + group - 1) / group) return RAD_E_SHAPE;
        P.group = (int)group;
        P.as = (const float*)XS->data;
    }
    P.slot = (const int32_t*)SL->data; P.slot_st = sl_st;
    P.la = (const uint16_t*)LA->data;
    pool3(LA, R, K, &P.la_s0, &P.la_s1, &P.la_s2);
    P.t = (float*)T->data; P.t_rs = t_rs; P.t_cs = t_cs;
    P.K = (int)K; P.R = (int)R; P.S = (int)S;
    if (!P.as) P.group = (int)K;
    *out = P;
    *adt_out = adt;
    *M_out = M;
    return RAD_OK;
}

/* lora_expand's, the same way. */
static int expand_operands(const RadArgs* a, ExpandArgs* out, int* ydt_out, int64_t* M_out) {
    *M_out = 0;
    if (!a || a->n_t < 5 || !rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *T = rad_arg_in(a, 0), *SL = rad_arg_in(a, 1), *LB = rad_arg_in(a, 2);
    const RadTensor *SC = rad_arg_in(a, 3), *Y = rad_arg_in(a, 4);
    const int64_t N = rad_args_geti_or(a, "N", 0), R = rad_args_geti_or(a, "R", 0);
    const int64_t S = rad_args_geti_or(a, "S", 0);
    if (N <= 0 || R <= 0 || S <= 0 || R > (1 << 16) || S > (1 << 20)) return RAD_E_SHAPE;
    int ydt;
    switch (Y->dtype) {
        case RAD_BF16: ydt = kBF16; break;
        case RAD_F16:  ydt = kF16;  break;
        case RAD_F32:  ydt = kF32;  break;
        default:       return RAD_E_DTYPE;
    }
    if (!dtype_agrees(a, Y->dtype)) return RAD_E_DTYPE;
    if (T->dtype != RAD_F32 || SC->dtype != RAD_F32 || LB->dtype != RAD_BF16 ||
        SL->dtype != RAD_I32)
        return RAD_E_DTYPE;

    int64_t M = 0, y_rs = 0, y_cs = 0, tr = 0, t_rs = 0, t_cs = 0, sr = 0, sl_st = 0;
    if (!as_rows(Y, N, &M, &y_rs, &y_cs)) return RAD_E_SHAPE;     /* the step's rows: y's view */
    if (M <= 0) return RAD_E_SHAPE;                               /* zero rows is not a call */
    const int64_t band = rad_args_geti_or(a, "M", 0);             /* libref's band_rows rule */
    if (band > 0 && M > band) return RAD_E_SHAPE;
    if (!as_rows(T, R, &tr, &t_rs, &t_cs) || tr < M) return RAD_E_SHAPE;
    if (!as_vec(SL, &sr, &sl_st) || sr < M) return RAD_E_SHAPE;
    if (rad_tensor_numel(LB) < S * N * R || rad_tensor_numel(SC) < S * N) return RAD_E_SHAPE;
    if (M > 0x7fffffff) return RAD_E_SHAPE;

    ExpandArgs P{};
    P.t = (const float*)T->data; P.t_rs = t_rs; P.t_cs = t_cs;
    P.slot = (const int32_t*)SL->data; P.slot_st = sl_st;
    P.lb = (const uint16_t*)LB->data;
    pool3(LB, N, R, &P.lb_s0, &P.lb_s1, &P.lb_s2);
    P.sc = (const float*)SC->data;
    P.sc_rs = SC->rank >= 2 ? SC->stride[SC->rank - 2] : N;
    P.sc_cs = SC->rank >= 2 ? SC->stride[SC->rank - 1] : 1;
    P.y = Y->data; P.y_rs = y_rs; P.y_cs = y_cs;
    P.N = N; P.ntiles = (N + 255) / 256; P.R = (int)R; P.S = (int)S;
    *out = P;
    *ydt_out = ydt;
    *M_out = M;
    return RAD_OK;
}

static int shrink_scalar(const ShrinkArgs& P, int adt, int64_t M, RadStream stream) {
    switch (adt) {
        case kBF16: launch_shrink<kBF16>(P, M, stream); break;
        case kF16:  launch_shrink<kF16>(P, M, stream);  break;
        case kF32:  launch_shrink<kF32>(P, M, stream);  break;
        case kE4M3: launch_shrink<kE4M3>(P, M, stream); break;
        case kE5M2: launch_shrink<kE5M2>(P, M, stream); break;
        default:    launch_shrink<kI8>(P, M, stream);   break;
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* ================================================================ the tiled rows
 *
 * gfx1151 port of the three tiled rows. What changed, and why (halo_wmma.h, halo_dt16.h):
 *
 *   - A and B fragments are the lane's WHOLE sixteen-wide k run, not gfx12's eight values at
 *     k = 8*(e>>2) + 4*(lane>>4) + (e&3). So a 32-wide k step is still two WMMAs of sixteen k,
 *     and every gather loses libr4d's `+ half * 16` term: the lane reads all sixteen values
 *     and lanes 16-31 read the same ones (the hardware replicates the row across the halves).
 *   - The accumulator's ROW is 2*e + (lane>>4), not 8*(lane>>4) + e. A lane still holds one
 *     COLUMN (lane & 15) -- which is what keeps libr4d's per-lane group scale, its one-row
 *     accumulator and its epilogue indexing unchanged -- but its eight values are every
 *     second row of the tile instead of eight consecutive ones, so a scatter replaces the
 *     single 16-byte store.
 *   - Everything the rows promise about the arithmetic (which sum, in which order) is a
 *     property of the loop structure, not of the fragment layout, so it carries over: the
 *     k steps a wave takes, the wave reduction in LDS, the K split's arrival, expand_tiled's
 *     scalar j-ascending sequence (no WMMA there at all, so it is libref's bytes unchanged).
 */


enum {  /* the two rows' plans */
    kShMT = 16,
    kShJT = 16,
    kShStep = 32,
    kShWaves = 4,
    kShThreads = kShWaves * 32,
    kShU = 4,
    kShSplitMin = 16,
    kShFill = 24,
    kExMT = 32,
    kExNT = 128,
    kExJC = 32,
    kExThreads = 256,
    kZMax = 16,
    kZGrid = 256,
    kMmMT = 128,
    kMmNT = 256,
    kMmJC = 32,
    kMmPitch = 48,   /* kMmJC rounded up to a multiple of 8 shorts: every LDS row 16B aligned */
    kMmWaves = 8,
    kMmThreads = kMmWaves * 32,
};

typedef unsigned v4u __attribute__((ext_vector_type(4)));    /* 16 codes */

struct ShrinkTArgs {
    ShrinkArgs b;
    float*     part;       /* [ks][M][R] split partials, the op's scratch; null when ks == 1 */
    unsigned*  ctr;        /* arrival counters, one a (row tile, j tile, z); null when ks == 1 */
    int        M, jt, ks, spb, nsteps, Z;
};

struct ExpandTArgs {
    ExpandArgs b;
    int        M, Z;
};

__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }

/* The tile's distinct adapter slots, ascending; -1 entries are base rows. One thread. */
__device__ inline int distinct_slots(const int* s, int* out, int n) {
    int k = 0;
    for (int i = 0; i < n; ++i) {
        const int v = s[i];
        if (v < 0) continue;
        int p = 0;
        while (p < k && out[p] < v) ++p;
        if (p < k && out[p] == v) continue;
        for (int q = k; q > p; --q) out[q] = out[q - 1];
        out[p] = v;
        ++k;
    }
    return k;
}

/* A code as the bf16 that holds it exactly: E4M3, E5M2 and i8 all fit bf16's 8-bit significand. */
template <int ADT>
__device__ __forceinline__ uint16_t code_bf16(unsigned b) {
    if (ADT == kE4M3) {
        const unsigned s = (b & 0x80u) << 8, e = (b >> 3) & 0xfu, m = b & 7u;
        if (e == 0u) return (uint16_t)(s | (__float_as_uint((float)m * 0.001953125f) >> 16));
        if (e == 0xfu && m == 7u) return (uint16_t)(s | 0x7fc0u);
        return (uint16_t)(s | ((e + 120u) << 7) | (m << 4));
    }
    if (ADT == kE5M2) {
        const unsigned s = (b & 0x80u) << 8, e = (b >> 2) & 0x1fu, m = b & 3u;
        if (e == 0u) return (uint16_t)(s | (__float_as_uint((float)m * 1.52587890625e-05f) >> 16));
        if (e == 0x1fu) return (uint16_t)(s | 0x7f80u | (m << 5));
        return (uint16_t)(s | ((e + 112u) << 7) | (m << 5));
    }
    return (uint16_t)(__float_as_uint((float)(int8_t)(uint8_t)b) >> 16);
}

__device__ __forceinline__ v16bf zero_frag() {
    union { uint16_t h[16]; v16bf f; } u;
#pragma unroll
    for (int e = 0; e < 16; ++e) u.h[e] = 0;
    return u.f;
}

/* One gfx1151 A or B fragment: sixteen contiguous 16-bit values at `p`, which is already advanced
 * to this lane's row and k. Element e is k + e, so the whole k run is in ONE lane. Zeros past K.
 * DENSE: the row is dense from `p` and 16-byte aligned, so two 16-byte loads (gfx12's fragment is
 * two 8-byte groups 16 bytes apart and needs only 8). */
template <bool DENSE>
__device__ __forceinline__ v16bf ld_u16(const uint16_t* p, int rem, long long cs) {
    union { uint16_t h[16]; uint4 w[2]; v16bf f; } u;
    if (DENSE && rem >= 16) {
        u.w[0] = ((const uint4*)p)[0];
        u.w[1] = ((const uint4*)p)[1];
        return u.f;
    }
#pragma unroll
    for (int e = 0; e < 16; ++e) u.h[e] = e < rem ? p[(long long)e * cs] : (uint16_t)0;
    return u.f;
}

/* A pool row's fragment: sixteen k from k0 on, zeros past K and for a lane that is off (j >= R).
 * Zeros and not whatever follows: the next row's values times a nonzero x could be a NaN. */
template <bool V>
__device__ __forceinline__ v16bf ld_pool_frag(const uint16_t* row, int k0, int K, long long cs,
                                             bool on) {
    if (!on) return zero_frag();
    return ld_u16<V>(row + (long long)k0 * cs, K - k0, cs);
}

/* An activation row's fragment: bf16 as it is, a code as the bf16 that holds it exactly. */
template <int ADT, bool V>
__device__ __forceinline__ v16bf ld_act_frag(const void* a, long long row, int k0, int K,
                                            long long cs, bool on) {
    if (!on) return zero_frag();
    if (ADT == kBF16) {
        const uint16_t* p = (const uint16_t*)a + row;
        return ld_u16<V>(p + (long long)k0 * cs, K - k0, cs);
    }
    const uint8_t* p = (const uint8_t*)a + row;
    union { uint16_t h[16]; uint4 w[1]; v16bf f; } u;
    if (V && k0 + 16 <= K) {
        u.w[0] = *(const uint4*)(p + k0);
        uint16_t h[16];
#pragma unroll
        for (int e = 0; e < 16; ++e) h[e] = (uint16_t)code_bf16<ADT>(((const uint8_t*)&u.w[0])[e]);
#pragma unroll
        for (int e = 0; e < 16; ++e) u.h[e] = h[e];
        return u.f;
    }
#pragma unroll
    for (int e = 0; e < 16; ++e)
        u.h[e] = k0 + e < K ? (uint16_t)code_bf16<ADT>(p[(long long)(k0 + e) * cs]) : (uint16_t)0;
    return u.f;
}

/* Shrink, one block a (j tile x K split, row tile, z), 4 waves. The pool fragment is the WMMA's A
 * operand (16 j x 16 k) and the x fragment its B (16 k x 16 rows). On gfx1151 lane l's eight
 * outputs are j = 2 (l/16) + e .. of ONE column m = l % 16, so a lane's group scale is still one
 * value. Per element the order is, all of it a function of K, R and the group alone:
 *
 *   wave w:  acc = 0; for its steps i ascending (i = w, w + 4, ... of the block's run)
 *              bf16:  acc = wmma(A1, X1, wmma(A0, X0, acc))
 *              coded: acc = acc + scale(m, group of i) * wmma(A1, X1, wmma(A0, X0, 0))
 *   block:   v = ((w0 + w1) + w2) + w3
 *   t:       v of the single block, or ((p0 + p1) + ...) + p[ks-1] over the K splits
 */
template <int ADT, bool VX, bool VA>
__global__ __launch_bounds__(kShThreads) void lora_shrink_wmma_kernel(ShrinkTArgs T) {
    __shared__ float red[kShWaves][kShJT * kShMT];      /* [wave][j][m] */
    __shared__ int sh_slot[kShMT], sh_list[kShMT];
    __shared__ int sh_n;
    __shared__ unsigned sh_arr;
    const ShrinkArgs& A = T.b;
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int col = lane & 15, half = lane >> 4;
    const int jt = (int)blockIdx.x / T.ks, ks = (int)blockIdx.x - jt * T.ks;
    const int m0 = (int)blockIdx.y * kShMT, z = (int)blockIdx.z;

    /* The tile's slots, read by every wave, so a tile of base rows leaves here: no barrier, no
     * operand load. */
    int my = -1;
    if (lane < kShMT && m0 + lane < T.M) {
        const int s = A.slot[(long long)(m0 + lane) * A.slot_st];
        if (s >= 0 && s < A.S) my = s;
    }
    if (__builtin_amdgcn_ballot_w32(my >= 0) == 0u) return;
    if (wave == 0 && lane < kShMT) sh_slot[lane] = my;
    __syncthreads();
    if (tid == 0) sh_n = distinct_slots(sh_slot, sh_list, kShMT);
    __syncthreads();
    const int npass = sh_n;
    if (z >= npass) return;

    const int st0 = ks * T.spb, st1 = imin(st0 + T.spb, T.nsteps);
    const int j = jt * kShJT + col;                    /* this lane's pool row (A) */
    const bool aon = j < A.R;
    const long long xrow = (long long)(m0 + col) * A.a_rs;   /* this lane's x row (B) */
    const long long srow = (long long)(m0 + col) * A.s_rs;
    for (int p = z; p < npass; p += T.Z) {
        const int s = sh_list[p];
        const bool xon = sh_slot[col] == s;            /* this lane's row is in the pass */
        const uint16_t* ap = A.la + (long long)s * A.la_s0 + (long long)j * A.la_s1;
        v8f acc = (v8f)(0.0f);
        for (int st = st0 + wave; st < st1; st += kShWaves * kShU) {
            v16bf af[kShU][2], xf[kShU][2];
            float sc[kShU];
#pragma unroll
            for (int u = 0; u < kShU; ++u) {
                const int su = st + u * kShWaves;
                const bool in = su < st1;
                const int k = su * kShStep;
                af[u][0] = ld_pool_frag<VA>(ap, k, A.K, A.la_s2, aon && in);
                af[u][1] = ld_pool_frag<VA>(ap, k + 16, A.K, A.la_s2, aon && in);
                xf[u][0] = ld_act_frag<ADT, VX>(A.a, xrow, k, A.K, A.a_cs, xon && in);
                xf[u][1] = ld_act_frag<ADT, VX>(A.a, xrow, k + 16, A.K, A.a_cs, xon && in);
                sc[u] = 0.0f;
                if (ADT != kBF16 && xon && in)
                    sc[u] = A.as[srow + (long long)((su * kShStep) / A.group) * A.s_cs];
            }
#pragma unroll
            for (int u = 0; u < kShU; ++u) {
                if (st + u * kShWaves >= st1) break;   /* wave-uniform */
                if (ADT == kBF16) {
                    acc = halo::mma(af[u][0], xf[u][0], acc);
                    acc = halo::mma(af[u][1], xf[u][1], acc);
                } else {
                    v8f d = halo::mma(af[u][0], xf[u][0], (v8f)(0.0f));
                    d = halo::mma(af[u][1], xf[u][1], d);
                    acc = acc + d * sc[u];
                }
            }
        }
        /* gfx1151: element e is row 2*e + (lane>>4) of the j tile, column (lane & 15). */
#pragma unroll
        for (int e = 0; e < 8; ++e) red[wave][(2 * e + half) * kShMT + col] = acc[e];
        __syncthreads();
        for (int i = tid; i < kShJT * kShMT; i += kShThreads) {
            const int jj = i / kShMT, mm = i - jj * kShMT, jg = jt * kShJT + jj;
            float v = red[0][i];
            for (int w = 1; w < kShWaves; ++w) v = v + red[w][i];
            if (sh_slot[mm] != s || jg >= A.R) continue;
            if (T.ks == 1)
                A.t[(long long)(m0 + mm) * A.t_rs + (long long)jg * A.t_cs] = v;
            else
                T.part[((long long)ks * T.M + m0 + mm) * A.R + jg] = v;
        }
        __syncthreads();
    }
    if (T.ks == 1) return;

    /* The K split's sum, in the last block to arrive. */
    __threadfence();
    __syncthreads();
    const int cidx = ((int)blockIdx.y * T.jt + jt) * kZMax + z;
    if (tid == 0)
        sh_arr = __hip_atomic_fetch_add(&T.ctr[cidx], 1u, __ATOMIC_ACQ_REL,
                                        __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (sh_arr != (unsigned)(T.ks - 1)) return;
    if (tid == 0) T.ctr[cidx] = 0u;
    __threadfence();
    for (int i = tid; i < kShJT * kShMT; i += kShThreads) {
        const int jj = i / kShMT, mm = i - jj * kShMT, jg = jt * kShJT + jj;
        const int s = sh_slot[mm];
        if (s < 0 || jg >= A.R) continue;
        int r = 0;
        while (sh_list[r] != s) ++r;
        if (r % T.Z != z) continue;                    /* another block's pass */
        const long long o = (long long)(m0 + mm) * A.R + jg, pl = (long long)T.M * A.R;
        float v = T.part[o];
        for (int q = 1; q < T.ks; ++q) v = v + T.part[(long long)q * pl + o];
        A.t[(long long)(m0 + mm) * A.t_rs + (long long)jg * A.t_cs] = v;
    }
}

/* Expand, one block a (128 columns, 32 rows, z), 8 waves: wave w owns rows 4w..4w+3 and lane l
 * columns 4l..4l+3. No WMMA: it is the scalar row's arithmetic in a tile, so its bytes are
 * libref's unchanged by the port. B[s]'s [128 columns][32 j] slab is staged as f32 and t's [32
 * rows][32 j] beside it. V: B is dense, 16-byte aligned and R a multiple of 8. */
template <int YDT, bool V>
__global__ __launch_bounds__(kExThreads) void lora_expand_tiled_kernel(ExpandTArgs T) {
    __shared__ float bs[kExJC][kExNT];
    __shared__ float ts[kExJC][kExMT];
    __shared__ int ex_slot[kExMT], ex_list[kExMT];
    __shared__ int ex_n;
    const ExpandArgs& A = T.b;
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const long long n0 = (long long)blockIdx.x * kExNT;
    const int m0 = (int)blockIdx.y * kExMT, z = (int)blockIdx.z;

    int my = -1;
    if (m0 + lane < T.M) {
        const int s = A.slot[(long long)(m0 + lane) * A.slot_st];
        if (s >= 0 && s < A.S) my = s;
    }
    if (__builtin_amdgcn_ballot_w32(my >= 0) == 0u) return;     /* a tile of base rows */
    if (wave == 0) ex_slot[lane] = my;
    __syncthreads();
    if (tid == 0) ex_n = distinct_slots(ex_slot, ex_list, kExMT);
    __syncthreads();
    const int npass = ex_n;
    if (z >= npass) return;

    for (int p = z; p < npass; p += T.Z) {
        const int s = ex_list[p];
        const uint16_t* bp = A.lb + (long long)s * A.lb_s0;
        const bool live = ex_slot[4 * wave] == s || ex_slot[4 * wave + 1] == s ||
                          ex_slot[4 * wave + 2] == s || ex_slot[4 * wave + 3] == s;
        float acc[4][4];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) acc[r][c] = 0.0f;
        for (int j0 = 0; j0 < A.R; j0 += kExJC) {
            const int jc = imin(kExJC, A.R - j0);
            __syncthreads();                           /* the last slab's readers are done */
            if (V) {
                const int nv = jc >> 3;                /* four lanes cover a column's 32 j */
                for (int v = tid; v < kExNT * nv; v += kExThreads) {
                    const int nl = v / nv, jv = v - nl * nv;
                    const long long n = n0 + nl;
                    v8s q = (v8s)(short)0;
                    if (n < A.N) q = *(const v8s*)(bp + n * A.lb_s1 + j0 + jv * 8);
                    for (int e = 0; e < 8; ++e) bs[jv * 8 + e][nl] = bf2f((uint16_t)q[e]);
                }
            } else {
                for (int v = tid; v < kExNT * jc; v += kExThreads) {
                    const int jj = v / kExNT, nl = v - jj * kExNT;
                    const long long n = n0 + nl;
                    bs[jj][nl] = n < A.N ? bf2f(bp[n * A.lb_s1 + (long long)(j0 + jj) * A.lb_s2])
                                         : 0.0f;
                }
            }
            for (int v = tid; v < kExMT * jc; v += kExThreads) {
                const int jj = v / kExMT, r = v - jj * kExMT;
                ts[jj][r] = ex_slot[r] == s
                                ? A.t[(long long)(m0 + r) * A.t_rs + (long long)(j0 + jj) * A.t_cs]
                                : 0.0f;
            }
            __syncthreads();
            if (!live) continue;
            int jj = 0;
            for (; jj + 4 <= jc; jj += 4) {
                float tv[4][4], bv[4][4];
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    for (int r = 0; r < 4; ++r) tv[q][r] = ts[jj + q][4 * wave + r];
                    for (int c = 0; c < 4; ++c) bv[q][c] = bs[jj + q][4 * lane + c];
                }
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    for (int r = 0; r < 4; ++r)
                        for (int c = 0; c < 4; ++c)
                            acc[r][c] = acc[r][c] + tv[q][r] * bv[q][c];
            }
            for (; jj < jc; ++jj) {
                float tv[4], bv[4];
                for (int r = 0; r < 4; ++r) tv[r] = ts[jj][4 * wave + r];
                for (int c = 0; c < 4; ++c) bv[c] = bs[jj][4 * lane + c];
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c) acc[r][c] = acc[r][c] + tv[r] * bv[c];
            }
        }
        for (int r = 0; r < 4; ++r) {
            const int row = 4 * wave + r;
            if (ex_slot[row] != s) continue;           /* another pass's row, or a base row */
            const long long m = m0 + row;
            for (int c = 0; c < 4; ++c) {
                const long long n = n0 + 4 * lane + c;
                if (n >= A.N) continue;
                const float d = A.sc[(long long)s * A.sc_rs + n * A.sc_cs] * acc[r][c];
                const long long o = m * A.y_rs + n * A.y_cs;
                if (YDT == kBF16) {
                    uint16_t* y = (uint16_t*)A.y + o;
                    *y = f2bf(bf2f(*y) + d);
                } else if (YDT == kF16) {
                    uint16_t* y = (uint16_t*)A.y + o;
                    *y = f32_to_f16(f16_to_f32(*y) + d);
                } else {
                    float* y = (float*)A.y + o;
                    *y = *y + d;
                }
            }
        }
    }
}

/* lora_expand_mm: the same delta on the WMMA, t rounded to bf16, for the passes whose rows are
 * many and mostly of one slot. One block a (256 columns, 128 rows), eight waves, wave w the 32
 * rows at 32 (w >> 1) and the 128 columns at 128 (w & 1): 2 x 8 WMMA tiles. B's fragment is the
 * A operand (16 n x 16 j) and t's the B (16 j x 16 m). On gfx1151 the lane's eight outputs are n
 * = 2 (lane>>4) + e of ONE column m, so the epilogue is eight strided stores. */
template <int YDT, bool V>
__global__ __launch_bounds__(kMmThreads) void lora_expand_mm_kernel(ExpandTArgs T) {
    __shared__ uint16_t ts[kMmMT][kMmPitch];
    __shared__ uint16_t bs[kMmNT][kMmPitch];
    __shared__ int mm_slot[kMmMT], mm_list[kMmMT];
    __shared__ int mm_n;
    const ExpandArgs& A = T.b;
    const int tid = (int)threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int col = lane & 15, half = lane >> 4;
    const long long nb = (long long)blockIdx.x * kMmNT;
    const int mb = (int)blockIdx.y * kMmMT;
    const int wm = 32 * (wave >> 1), wn = 128 * (wave & 1);

    for (int i = tid; i < kMmMT; i += kMmThreads) {
        int s = -1;
        if (mb + i < T.M) {
            s = A.slot[(long long)(mb + i) * A.slot_st];
            if (s < 0 || s >= A.S) s = -1;
        }
        mm_slot[i] = s;
    }
    __syncthreads();
    if (tid == 0) mm_n = distinct_slots(mm_slot, mm_list, kMmMT);
    __syncthreads();
    const int npass = mm_n;

    for (int p = 0; p < npass; ++p) {
        const int s = mm_list[p];
        const uint16_t* bp = A.lb + (long long)s * A.lb_s0;
        v8f acc[2][8];
        for (int b = 0; b < 2; ++b)
            for (int a = 0; a < 8; ++a) acc[b][a] = (v8f)(0.0f);
        for (int j0 = 0; j0 < A.R; j0 += kMmJC) {
            __syncthreads();
            for (int i = tid; i < kMmMT * 2; i += kMmThreads) {
                const int r = i >> 1, h = (i & 1) * 16;
                const bool on = mm_slot[r] == s;
                const long long row = (long long)(mb + r) * A.t_rs;
                for (int e = 0; e < 16; ++e) {
                    const int j = j0 + h + e;
                    ts[r][h + e] = on && j < A.R ? f2bf(A.t[row + (long long)j * A.t_cs]) : (uint16_t)0;
                }
            }
            for (int i = tid; i < kMmNT * 2; i += kMmThreads) {
                const int c = i >> 1, h = (i & 1) * 16;
                const long long n = nb + c;
                union { uint16_t h[16]; v16bf f; } u;
                u.f = ld_u16<V>(bp + (n < A.N ? n : 0) * A.lb_s1 + j0 + h,
                                n < A.N ? A.R - (j0 + h) : 0, A.lb_s2);
                for (int e = 0; e < 16; ++e) bs[c][h + e] = u.h[e];
            }
            __syncthreads();
            v16bf tf[2][2];
            for (int b = 0; b < 2; ++b)
                for (int q = 0; q < 2; ++q) {
                    union { uint16_t h[16]; v16bf f; } u;
                    for (int e = 0; e < 16; ++e) u.h[e] = ts[wm + 16 * b + col][16 * q + e];
                    tf[b][q] = u.f;
                }
            for (int a = 0; a < 8; ++a) {
                v16bf bf[2];
                for (int q = 0; q < 2; ++q) {
                    union { uint16_t h[16]; v16bf f; } u;
                    for (int e = 0; e < 16; ++e) u.h[e] = bs[wn + 16 * a + col][16 * q + e];
                    bf[q] = u.f;
                }
                for (int b = 0; b < 2; ++b) {
                    acc[b][a] = halo::mma(bf[0], tf[b][0], acc[b][a]);
                    acc[b][a] = halo::mma(bf[1], tf[b][1], acc[b][a]);
                }
            }
        }
        for (int b = 0; b < 2; ++b) {
            const int ml = wm + 16 * b + col;
            if (mm_slot[ml] != s) continue;
            const long long m = mb + ml;
            for (int a = 0; a < 8; ++a) {
                const long long n0 = nb + wn + 16 * a;
                for (int e = 0; e < 8; ++e) {
                    const long long n = n0 + 2 * e + half;   /* gfx1151's D row */
                    if (n >= A.N) continue;
                    const float d = A.sc[(long long)s * A.sc_rs + n * A.sc_cs] * acc[b][a][e];
                    const long long o = m * A.y_rs + n * A.y_cs;
                    if (YDT == kBF16) {
                        uint16_t* y = (uint16_t*)A.y + o;
                        *y = f2bf(bf2f(*y) + d);
                    } else if (YDT == kF16) {
                        uint16_t* y = (uint16_t*)A.y + o;
                        *y = f32_to_f16(f16_to_f32(*y) + d);
                    } else {
                        float* y = (float*)A.y + o;
                        *y = *y + d;
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------------- the tiled rows' plans (host) */

/* The shrink split, from K and R alone and never M, because it is the summation order. */
struct ShPlan { int jt, ks, spb, nsteps; };
static ShPlan sh_plan(int64_t K, int64_t R) {
    ShPlan p;
    p.nsteps = (int)((K + kShStep - 1) / kShStep);
    p.jt = (int)((R + kShJT - 1) / kShJT);
    int ks = (kShFill + p.jt - 1) / p.jt;
    const int kmax = (p.nsteps + kShSplitMin - 1) / kShSplitMin;
    if (ks > kmax) ks = kmax;
    if (ks < 1) ks = 1;
    p.spb = (p.nsteps + ks - 1) / ks;
    p.ks = (p.nsteps + p.spb - 1) / p.spb;
    return p;
}
static int64_t sh_part_bytes(int64_t M, int64_t R, const ShPlan& p) {
    return p.ks > 1 ? (int64_t)p.ks * M * R * (int64_t)sizeof(float) : 0;
}

/* How many blocks one tile's passes spread over -- which block runs a pass, never its arithmetic. */
static int z_fan(int64_t base, int64_t S, int64_t M, int64_t tile) {
    int64_t z = base >= kZGrid ? 1 : kZGrid / base;
    if (z > S) z = S;
    if (z > kZMax) z = kZMax;
    if (z > M) z = M;
    if (z > tile) z = tile;
    return z < 1 ? 1 : (int)z;
}

/* The split's arrival counters, one block a (device, `t`); left zero for the next launch. */
static unsigned* shrink_ctr(const void* t, int64_t n, RadStream stream) {
    static std::mutex mu;
    static std::map<std::pair<int, const void*>, std::pair<unsigned*, int64_t>> blocks;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return nullptr;
    std::lock_guard<std::mutex> lk(mu);
    const auto key = std::make_pair(dev, t);
    auto it = blocks.find(key);
    if (it != blocks.end() && it->second.second >= n) return it->second.first;
    int64_t cap = it != blocks.end() ? 2 * it->second.second : 4096;
    while (cap < n) cap *= 2;
    void* p = nullptr;
    if (hipMalloc(&p, (size_t)cap * sizeof(unsigned)) != hipSuccess) return nullptr;
    if (hipMemsetAsync(p, 0, (size_t)cap * sizeof(unsigned), (hipStream_t)stream) != hipSuccess)
        return nullptr;
    blocks[key] = std::make_pair((unsigned*)p, cap);
    return (unsigned*)p;
}

template <int ADT, bool VX, bool VA>
static void launch_shrink_wmma(const ShrinkTArgs& P, dim3 grid, RadStream stream) {
    (void)stream;
    HALO_KLAUNCH(HALO_KNAME(lora_shrink_wmma_kernel<ADT, VX, VA>), grid, dim3(kShThreads),
                 (hipStream_t)stream, P);
}
template <int ADT>
static void launch_shrink_wmma_v(const ShrinkTArgs& P, dim3 grid, bool vx, bool va,
                                 RadStream stream) {
    if (vx && va)  launch_shrink_wmma<ADT, true, true>(P, grid, stream);
    else if (vx)   launch_shrink_wmma<ADT, true, false>(P, grid, stream);
    else if (va)   launch_shrink_wmma<ADT, false, true>(P, grid, stream);
    else           launch_shrink_wmma<ADT, false, false>(P, grid, stream);
}
template <int YDT, bool V>
static void launch_expand_tiled(const ExpandTArgs& P, dim3 grid, RadStream stream) {
    (void)stream;
    HALO_KLAUNCH(HALO_KNAME(lora_expand_tiled_kernel<YDT, V>), grid, dim3(kExThreads),
                 (hipStream_t)stream, P);
}
template <int YDT, bool V>
static void launch_expand_mm(const ExpandTArgs& P, dim3 grid, RadStream stream) {
    (void)stream;
    HALO_KLAUNCH(HALO_KNAME(lora_expand_mm_kernel<YDT, V>), grid, dim3(kMmThreads),
                 (hipStream_t)stream, P);
}

}  /* namespace halo_lora */

extern "C" {

/* operands: a, a_scale?, slot, lora_a, t */
int halo_rad_lora_shrink(const RadArgs* a, RadStream stream) {
    using namespace halo_lora;
    ShrinkArgs P{};
    int adt = 0;
    int64_t M = 0;
    const int rc = shrink_operands(a, &P, &adt, &M);
    if (rc != RAD_OK || M <= 0) return rc;
    return shrink_scalar(P, adt, M, stream);
}

/* operands: t, slot, lora_b, scale, y (inout) */
int halo_rad_lora_expand(const RadArgs* a, RadStream stream) {
    using namespace halo_lora;
    ExpandArgs P{};
    int ydt = 0;
    int64_t M = 0;
    const int rc = expand_operands(a, &P, &ydt, &M);
    if (rc != RAD_OK || M <= 0) return rc;
    if (M * P.ntiles > 0x7fffffff) return RAD_E_SHAPE;
    switch (ydt) {
        case kBF16: launch_expand<kBF16>(P, M * P.ntiles, stream); break;
        case kF16:  launch_expand<kF16>(P, M * P.ntiles, stream);  break;
        default:    launch_expand<kF32>(P, M * P.ntiles, stream);  break;
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* The tiled shrink: lora_shrink_wmma (bf16 a) and lora_shrink_wmma_q (fp8 / i8 codes whose group
 * is whole 32-wide steps) share this entry point. What the WMMA cannot hold exactly (f16 and f32
 * activations, a scale group that ends inside a step) goes to the scalar kernel, decided by the
 * dtype and the stream's grid and never by M. */
int halo_rad_lora_shrink_wmma(const RadArgs* a, RadStream stream) {
    using namespace halo_lora;
    ShrinkArgs P{};
    int adt = 0;
    int64_t M = 0;
    const int rc = shrink_operands(a, &P, &adt, &M);
    if (rc != RAD_OK || M <= 0) return rc;
    if (adt == kF16 || adt == kF32) return shrink_scalar(P, adt, M, stream);
    if (adt != kBF16 && P.group % kShStep != 0 && P.group < P.K)
        return shrink_scalar(P, adt, M, stream);

    const ShPlan pl = sh_plan(P.K, P.R);
    const int64_t rt = (M + kShMT - 1) / kShMT;
    if (rt > 65535 || (int64_t)pl.jt * pl.ks > 0x7fffffff) return RAD_E_SHAPE;
    ShrinkTArgs T{};
    T.b = P;
    T.M = (int)M; T.jt = pl.jt; T.ks = pl.ks; T.spb = pl.spb; T.nsteps = pl.nsteps;
    T.Z = z_fan(rt * pl.jt * pl.ks, P.S, M, kShMT);
    if (pl.ks > 1) {
        const int64_t need = sh_part_bytes(M, P.R, pl);
        if (!a->scratch || a->scratch_bytes < need) return RAD_E_SCRATCH;
        const int64_t band = rad_args_geti_or(a, "M", 0);
        const int64_t rows = band > M ? band : M;
        T.part = (float*)a->scratch;
        T.ctr = shrink_ctr(P.t, ((rows + kShMT - 1) / kShMT) * pl.jt * kZMax, stream);
        if (!T.ctr) return RAD_E_DEVICE;
    }
    /* 16-byte loads where the rows allow them; the arithmetic is the same either way. */
    const uintptr_t xa = (uintptr_t)P.a, pa = (uintptr_t)P.la;
    const bool vx = P.a_cs == 1 && xa % 16 == 0 && P.a_rs % (adt == kBF16 ? 8 : 16) == 0;
    const bool va = P.la_s2 == 1 && pa % 16 == 0 && P.la_s1 % 8 == 0 && P.la_s0 % 8 == 0;
    const dim3 grid((unsigned)(pl.jt * pl.ks), (unsigned)rt, (unsigned)T.Z);
    switch (adt) {
        case kBF16: launch_shrink_wmma_v<kBF16>(T, grid, vx, va, stream); break;
        case kE4M3: launch_shrink_wmma_v<kE4M3>(T, grid, vx, va, stream); break;
        case kE5M2: launch_shrink_wmma_v<kE5M2>(T, grid, vx, va, stream); break;
        default:    launch_shrink_wmma_v<kI8>(T, grid, vx, va, stream);   break;
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* The K split's partials, sized at declare for the band's widest M. */
int64_t halo_rad_lora_shrink_wmma_scratch(const RadArgs* a) {
    using namespace halo_lora;
    const int64_t M = rad_args_geti_or(a, "M", 0), K = rad_args_geti_or(a, "K", 0);
    const int64_t R = rad_args_geti_or(a, "R", 0);
    if (M <= 0 || K <= 0 || R <= 0 || K > 0x7fffffff || R > (1 << 16)) return 0;
    const char* d = rad_args_gets(a, "dtype");
    if (d && (rad_dtype_parse(d) == RAD_F16 || rad_dtype_parse(d) == RAD_F32)) return 0;
    return sh_part_bytes(M, R, sh_plan(K, R));
}

/* The tiled expand: one block a (128 columns, 32 rows). */
int halo_rad_lora_expand_tiled(const RadArgs* a, RadStream stream) {
    using namespace halo_lora;
    ExpandArgs P{};
    int ydt = 0;
    int64_t M = 0;
    const int rc = expand_operands(a, &P, &ydt, &M);
    if (rc != RAD_OK || M <= 0) return rc;
    const int64_t rt = (M + kExMT - 1) / kExMT, nt = (P.N + kExNT - 1) / kExNT;
    if (rt > 65535 || nt > 0x7fffffff) return RAD_E_SHAPE;
    ExpandTArgs T{};
    T.b = P;
    T.M = (int)M;
    T.Z = z_fan(rt * nt, P.S, M, kExMT);
    const bool v = P.lb_s2 == 1 && (uintptr_t)P.lb % 16 == 0 && P.lb_s1 % 8 == 0 &&
                   P.lb_s0 % 8 == 0 && P.R % 8 == 0;
    const dim3 grid((unsigned)nt, (unsigned)rt, (unsigned)T.Z);
    switch (ydt) {
        case kBF16: v ? launch_expand_tiled<kBF16, true>(T, grid, stream)
                      : launch_expand_tiled<kBF16, false>(T, grid, stream); break;
        case kF16:  v ? launch_expand_tiled<kF16, true>(T, grid, stream)
                      : launch_expand_tiled<kF16, false>(T, grid, stream);  break;
        default:    v ? launch_expand_tiled<kF32, true>(T, grid, stream)
                      : launch_expand_tiled<kF32, false>(T, grid, stream);  break;
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* lora_expand_mm: one block a (256 columns, 128 rows). */
int halo_rad_lora_expand_mm(const RadArgs* a, RadStream stream) {
    using namespace halo_lora;
    ExpandArgs P{};
    int ydt = 0;
    int64_t M = 0;
    const int rc = expand_operands(a, &P, &ydt, &M);
    if (rc != RAD_OK || M <= 0) return rc;
    const int64_t rt = (M + kMmMT - 1) / kMmMT, nt = (P.N + kMmNT - 1) / kMmNT;
    if (rt > 65535 || nt > 0x7fffffff) return RAD_E_SHAPE;
    ExpandTArgs T{};
    T.b = P;
    T.M = (int)M;
    T.Z = 1;
    const bool v = P.lb_s2 == 1 && (uintptr_t)P.lb % 16 == 0 && P.lb_s1 % 8 == 0 &&
                   P.lb_s0 % 8 == 0;
    const dim3 grid((unsigned)nt, (unsigned)rt, 1u);
    switch (ydt) {
        case kBF16: v ? launch_expand_mm<kBF16, true>(T, grid, stream)
                      : launch_expand_mm<kBF16, false>(T, grid, stream); break;
        case kF16:  v ? launch_expand_mm<kF16, true>(T, grid, stream)
                      : launch_expand_mm<kF16, false>(T, grid, stream);  break;
        default:    v ? launch_expand_mm<kF32, true>(T, grid, stream)
                      : launch_expand_mm<kF32, false>(T, grid, stream);  break;
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
}  /* extern "C" */
