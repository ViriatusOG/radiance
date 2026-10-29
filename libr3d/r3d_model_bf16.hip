/* r3d_model_bf16.hip -- the small ops the engine issues between the GEMMs: the norms, the
 * elementwise arithmetic, the activations, cast, softmax, the embedding gather, gather_rows and
 * scatter_rows, rope and the KV store. Every one of them needs a device launch, because the
 * engine's buffers live in VRAM and `rad_issue` refuses an op with no launch for its operands'
 * domain: on an RDNA3 card this unit is the difference between a model that runs and one that does
 * not (R3D_CONTRACT, "What libr3d is").
 *
 * The arithmetic is libref's -- the oracle every row is checked against at kbench -- and each
 * kernel names the ref function it must equal, its operation order and its rounding points:
 *
 *   * every reduction accumulates in f32 and only the store narrows, RTNE (r3d_codecs.h is
 *     rad_plugin.h's rounders restated for device code, bit for bit);
 *   * a row's sum is taken over lanes covering consecutive elements, butterflies in the order a
 *     `__shfl_xor` loop runs and wave partials added in wave order: `block_sum` in r3d_common.h,
 *     so the bracketing is the one the contract fixes and not a per-thread accident;
 *   * bf16 is kept in whole registers from load to store (`ldg_u16`) and a lane that owns two
 *     consecutive columns stores them as one dword through `bf16x2_rne` -- gfx11 has no
 *     v_cvt_pk_bf16_f32, so the pack is two software roundings and a v_perm_b32;
 *   * no FMA and no reassociation: -ffp-contract=off is on every HIP source (libr3d/CMakeLists.txt)
 *     and the expressions below are written in the reference's order, with the temporaries the
 *     reference has;
 *   * a negative index -- token id, gather row, KV slot -- is padding: a zero row, or no store.
 *     It is not clamped to zero, which would duplicate row 0 across every padding row.
 *
 * One deliberate divergence from ref, the same one libr4d states: ref validates its index arrays
 * on the host and refuses an id at or past the row count. A kernel cannot do that without a copy
 * and a sync on the step path, so a row count is an argument and an out-of-range id takes the
 * padding path (zero row, no store) instead of faulting. kv_store's slot bound is the one place
 * the reference has an argument the launcher does not (see below).
 *
 * `expf`, `powf` and `sincosf` are the accurate forms, not the `__` approximations: rope positions
 * reach 262144, past __sincosf's argument reduction, and libref evaluates every transcendental in
 * double and rounds once (libref's ref_exp/ref_pow/ref_sin/ref_cos/ref_erf). None of these
 * paths is the step's bottleneck, so the difference between libm and a device routine -- a few
 * ULP of f32, against the 1/256 of a bf16 store -- is the whole of the tolerance.
 *
 * No fusion: `rmsnorm_add` and the gated quantisers are separate entry points the architecture
 * plugin selects (spec §2.3), and live in other units.
 *
 * Two layers per op, as r3d.h states: a launcher that takes pointers as longs and geometry as
 * ints, checks what a `__global__` cannot, computes the grid and issues one kernel on the stream it
 * was handed; and the shim the row table names, which reads the operands with r3d_args.h, takes
 * the dtype from the tensor and the row count from the operand, and returns a RAD status. Neither
 * allocates, neither synchronises. A launcher returns 0 or -1 (null), -2 (extent), -3 (pitch),
 * -4 (the HIP launch failed); a shim maps any non-zero code to RAD_E_DEVICE.
 */

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <cmath>
#include <cstring>
#include <type_traits>

namespace {

constexpr uint32_t kThreads = 256;   // eight wave32s: the wide row, the hidden state
constexpr uint32_t kNarrow  = 32;    // one wave32: the narrow row, a head of a QK norm
constexpr uint32_t kNarrowN = 256;   // a row this narrow is a wave's, eight columns a lane

/* The workgroup a row gets: one wave for the rows a wave already covers -- an attention head, a
 * KV head, a QK norm, where a 256-thread group would spend three quarters of its lanes on a
 * barrier -- and 256 for everything wider. */
static inline int row_threads(int n) { return n <= (int)kNarrowN ? (int)kNarrow : (int)kThreads; }

/* ================================================================== the seams every kernel here
 * shares
 *
 * A whole-register store beside r3d_common.h's whole-register load. The dword form is the pair
 * store; the caller has already proved the address even, in elements, from a base alignment of 4.
 * The address-space cast is r3d_common.h's own, a C-style cast to the qualified type:
 * reinterpret_cast refuses to change address spaces.
 */
typedef uint32_t __attribute__((address_space(1))) *r3d_gu32;
typedef float    __attribute__((address_space(1))) *r3d_gf32;
typedef int32_t  __attribute__((address_space(1))) *r3d_gi32;
typedef int64_t  __attribute__((address_space(1))) *r3d_gi64;

__device__ __forceinline__ void stg_u32(uint16_t* p, uint32_t v) { *(r3d_gu32)p = v; }
__device__ __forceinline__ void stg_f32(float* p, float v) { *(r3d_gf32)p = v; }
/* A dword of four codes at a byte address: the fp8 quantiser's store. */
__device__ __forceinline__ void stg_code4(uint8_t* p, uint32_t v) { *(r3d_gu32)p = v; }
__device__ __forceinline__ int32_t ldg_i32(const int32_t* p) { return *(r3d_gi32)(uintptr_t)p; }
__device__ __forceinline__ int64_t ldg_i64(const int64_t* p) { return *(r3d_gi64)(uintptr_t)p; }
__device__ __forceinline__ void stg_i32(int32_t* p, int32_t v) { *(r3d_gi32)p = v; }
__device__ __forceinline__ void stg_i64(int64_t* p, int64_t v) { *(r3d_gi64)p = v; }
__device__ __forceinline__ uint16_t ldg_u16_at(const uint16_t* p, uint32_t i) { return (uint16_t)ldg_u16(p + i); }

/* The max across a workgroup, in the shape block_sum has: a wave butterfly and one LDS round trip.
 * The order is free -- a max is associative, commutative and idempotent, so softmax's maximum is
 * the one reduction whose bracketing cannot change the answer -- and every wave has to reach the
 * LDS before any thread reads it, which is what the barrier inside is for. */
template <int NTHREADS>
__device__ __forceinline__ float block_max(float v, float* lds) {
    constexpr int NWAVES = NTHREADS / kWave;
    const uint32_t lane = threadIdx.x & (kWave - 1);
    const uint32_t w = threadIdx.x / kWave;
    v = wave_max(v);
    if (lane == 0) lds[w] = v;
    __syncthreads();
    float t = lds[0];
#pragma unroll
    for (int i = 1; i < NWAVES; ++i) t = fmaxf(t, lds[i]);
    return t;
}

/* ================================================================== one element, any dtype
 *
 * cast, fill, gather_rows, scatter_rows, embed_lookup and kv_store are one kernel over several
 * dtypes, because their dtype is a runtime value in the launcher's argument list. These are
 * the reference's ld_dt and st_dt over libr3d's dtype codes, and they are deliberately as
 * expensive -- one switch per element, "the price of being generic" in ref's own words. None of
 * them is a matmul; none of them reads one element per FLOP.
 *
 * The integer stores round to nearest even, saturate, and store NaN as 0 (sat_round, the
 * reference's): wrapping an overflowing activation into a plausible small number is the failure
 * this library has to catch, not hide.
 */
__device__ __forceinline__ float sat_round(float v, float lo, float hi) {
    if (v != v) return 0.0f;
    const float r = rintf(v);
    return r < lo ? lo : (r > hi ? hi : r);
}

__device__ __forceinline__ float ld_elem(int dt, const void* p, size_t i) {
    switch (dt) {
        case R3D_DT_BF16: return bf16_to_f32(ldg_u16_at((const uint16_t*)p, (uint32_t)i));
        case R3D_DT_F16:  return f16_to_f32((uint16_t)ldg_u16((const uint16_t*)p + i));
        case R3D_DT_F32:  return ldg_f32((const float*)p + i);
        case R3D_DT_I32:  return (float)ldg_i32((const int32_t*)p + i);
        case R3D_DT_I64:  return (float)ldg_i64((const int64_t*)p + i);
        case R3D_DT_U8:   return e4m3_to_f32(((const uint8_t*)p)[i]);   // the container's fp8 byte
        case R3D_DT_I8:   return (float)((const int8_t*)p)[i];
        default:          return 0.0f;   // the shims refuse every other code; mirrors ref's ld_dt
    }
}

__device__ __forceinline__ void st_elem(int dt, void* p, size_t i, float v) {
    switch (dt) {
        case R3D_DT_BF16: ((uint16_t*)p)[i] = f32_to_bf16(v); break;
        case R3D_DT_F16:  ((uint16_t*)p)[i] = f32_to_f16(v); break;
        case R3D_DT_F32:  stg_f32((float*)p + i, v); break;
        case R3D_DT_U8:   ((uint8_t*)p)[i] = f32_to_e4m3(v); break;   // saturating, RTNE
        case R3D_DT_I8:   ((int8_t*)p)[i] = (int8_t)sat_round(v, -128.0f, 127.0f); break;
        case R3D_DT_I32: {
            const float r = sat_round(v, -0x1p31f, 0x1p31f);
            stg_i32((int32_t*)p + i, r >= 0x1p31f ? INT32_MAX : (int32_t)r);
            break;
        }
        case R3D_DT_I64: {
            const float r = sat_round(v, -0x1p63f, 0x1p63f);
            stg_i64((int64_t*)p + i, r >= 0x1p63f ? INT64_MAX : (int64_t)r);
            break;
        }
        default: break;
    }
}

/* The integer pair cast, which must not pass through f32: a token id above 2^24 does not survive
 * one, and an i64 -> i32 copy of a table offset that rounds is a wrong row, not a near miss. */
__device__ __forceinline__ int64_t ld_int(int dt, const void* p, size_t i) {
    switch (dt) {
        case R3D_DT_I32: return (int64_t)ldg_i32((const int32_t*)p + i);
        case R3D_DT_I64: return ldg_i64((const int64_t*)p + i);
        case R3D_DT_I8:  return (int64_t)((const int8_t*)p)[i];
        case R3D_DT_U8:  return (int64_t)((const uint8_t*)p)[i];
        default:         return 0;
    }
}

__device__ __forceinline__ void st_int(int dt, void* p, size_t i, int64_t v) {
    switch (dt) {
        case R3D_DT_I32: stg_i32((int32_t*)p + i, (int32_t)v); break;
        case R3D_DT_I64: stg_i64((int64_t*)p + i, v); break;
        case R3D_DT_I8:  ((int8_t*)p)[i] = (int8_t)v; break;
        case R3D_DT_U8:  ((uint8_t*)p)[i] = (uint8_t)v; break;
        default:         st_elem(dt, p, i, (float)v); break;
    }
}

__device__ __host__ __forceinline__ bool int_pair(int dt) {
    return dt == R3D_DT_I32 || dt == R3D_DT_I64 || dt == R3D_DT_I8 || dt == R3D_DT_U8;
}

/* The bytes an element of a dtype code occupies, on both sides of the launch: r3d_dt_size is a host
 * inline, and the gather kernels walk rows in bytes. */
__device__ __forceinline__ int dt_bytes(int dt) {
    switch (dt) {
        case R3D_DT_BF16: case R3D_DT_F16: return 2;
        case R3D_DT_F32:  case R3D_DT_I32: return 4;
        case R3D_DT_I64:  return 8;
        case R3D_DT_U8:   case R3D_DT_I8:  return 1;
        default:          return 0;
    }
}

/* The activation, off libref's act_* set. The reference evaluates the transcendental in double
 * and rounds once to f32 (libref's ref_exp/ref_erf); the device's accurate expf/erff are a
 * few ULP of f32 away, and the store to bf16 keeps 8 mantissa bits. gelu is the exact erf form --
 * "gelu" means this in libref, and the tanh approximation has to be asked for by name. */
template <int ACT>
__device__ __forceinline__ float act_of(float x) {
    if constexpr (ACT == R3D_ACT_SIGMOID) return 1.0f / (1.0f + expf(-x));
    if constexpr (ACT == R3D_ACT_GELU)
        return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
    return x * (1.0f / (1.0f + expf(-x)));   // silu: x * sigmoid(x), the product rounded once
}

/* ================================================================== rmsnorm
 * rms_rows:
 *
 *   ss = sum_i x[i]^2                        f32, ascending index order
 *   s  = 1 / sqrt(ss/n + eps)
 *   y[i] = x[i] * s * (w ? w[i] + wadd : 1) * mult
 *
 * The mean is over the last n of the row and the gain rides every row. An absent gain is the
 * constant 1 -- the shim refuses `wadd` without one, as ref does -- and the gain's dtype is read
 * off the operand (WDT: R3D_DT_BF16, R3D_DT_F32, R3D_DT_N for none), because a checkpoint that
 * stores its gains in f32 (a QK-norm's, an audio tower's) is read as it is. Forming 1+w in bf16
 * first loses 0.3% per channel, which is why `wadd` is added in f32 here.
 *
 * `mult` scales the normed row before its one rounding, for a residual branch weighted by a
 * constant (Gemma 4's audio feed-forward uses 0.5); 1 leaves the product chain alone.
 */
template <int THR, int WDT, int F32>
__global__ __launch_bounds__(THR) void rmsnorm_kernel(
    const uint16_t* __restrict__ x, const void* __restrict__ w, uint16_t* __restrict__ y,
    uint32_t n, float eps, float wadd, float mult, uint32_t x_ld, uint32_t y_ld, int pair) {
    typedef typename std::conditional<F32 != 0, float, uint16_t>::type T;
    __shared__ float lds[THR / kWave];
    const uint32_t m = blockIdx.x;
    const T* __restrict__ xr = (const T*)x + (size_t)m * x_ld;
    T* __restrict__ yr = (T*)y + (size_t)m * y_ld;
    const auto ld = [](const T* p, uint32_t i) {
        return F32 ? ldg_f32((const float*)p + i) : bf16_to_f32(ldg_u16_at((const uint16_t*)p, i));
    };

    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const float v = ld(xr, i);
        ss += v * v;
    }
    ss = block_sum<THR>(ss, lds);
    const float s = 1.0f / sqrtf(ss / (float)n + eps);

    const uint16_t* __restrict__ wr = (const uint16_t*)w;
    const float* __restrict__ wf = (const float*)w;
    /* One expression, the reference's order: ((x * s) * (w + wadd)) * mult. */
    const auto scaled = [&](uint32_t i) {
        const float g = WDT == R3D_DT_N ? 1.0f
                      : (WDT == R3D_DT_F32 ? ldg_f32(wf + i) : bf16_to_f32(ldg_u16_at(wr, i))) + wadd;
        return ld(xr, i) * s * g * mult;
    };
    if (pair) {
        for (uint32_t i = 2u * threadIdx.x; i < n; i += 2u * THR)
            stg_u32((uint16_t*)yr + i, bf16x2_rne(scaled(i), scaled(i + 1)));
    } else {
        for (uint32_t i = threadIdx.x; i < n; i += THR) {
            const float v = scaled(i);
            if (F32) stg_f32((float*)yr + i, v);
            else ((uint16_t*)yr)[i] = f32_to_bf16(v);
        }
    }
}

/* ================================================================== layernorm
 * Two passes -- the mean, then the squared deviations about that
 * mean -- and not one pass of E[x^2] - E[x]^2, which cancels badly when the mean is large against
 * the spread. Biased variance (divide by n), as transformers use. The store is the third pass:
 *
 *   y[i] = ((x[i] - mean) * s * w[i]) + b[i]
 *
 * with `b` optional and the gain bf16 (the launcher has no gain dtype; libref reads w's through
 * its own dtype, so an f32 gain is refused by the shim, not read as bf16).
 */
template <int THR>
__global__ __launch_bounds__(THR) void layernorm_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, const uint16_t* __restrict__ b,
    uint16_t* __restrict__ y, uint32_t n, float eps, uint32_t x_ld, uint32_t y_ld, int pair) {
    __shared__ float lds[THR / kWave];
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) sum += bf16_to_f32(ldg_u16_at(xr, i));
    const float mean = block_sum<THR>(sum, lds) / (float)n;

    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const float d = bf16_to_f32(ldg_u16_at(xr, i)) - mean;
        ss += d * d;
    }
    const float s = 1.0f / sqrtf(block_sum<THR>(ss, lds) / (float)n + eps);

    const auto scaled = [&](uint32_t i) {
        float v = (bf16_to_f32(ldg_u16_at(xr, i)) - mean) * s * bf16_to_f32(ldg_u16_at(w, i));
        if (b) v += bf16_to_f32(ldg_u16_at(b, i));
        return v;
    };
    if (pair) {
        for (uint32_t i = 2u * threadIdx.x; i < n; i += 2u * THR)
            stg_u32(yr + i, bf16x2_rne(scaled(i), scaled(i + 1)));
    } else {
        for (uint32_t i = threadIdx.x; i < n; i += THR)
            ((uint16_t*)yr)[i] = f32_to_bf16(scaled(i));
    }
}

/* ================================================================== add / mul
 * elementwise2, one rule per line:
 *
 *   u = a[i]; v = b[i]; s = mul ? u*v : u+v;
 *   if (div != 1) s = round_to_y(s) / div;            two roundings, y's dtype
 *   y[i] = round_to_y(s)
 *
 * `div` is the last sum of a mean taken pairwise -- (b0+b1+b2)/3, as a vocoder averages its
 * residual blocks -- and the sum is narrowed before it is divided because the unfused pair stores
 * it. 1 is the plain sum.
 *
 * The broadcast: b carrying exactly n elements rides every row (`bcast`, which the shim sets off
 * the operand's element count, as ref does); nothing else broadcasts, so a shape bug cannot pass
 * as a plausible number. The f32 pair narrows nothing on the way (round_dt(f32) is the identity),
 * and `y` may be `a`: every element is read by the thread that writes it.
 */
template <int THR, int MUL, int PAIR, int F32>
__global__ __launch_bounds__(THR) void binary_kernel(
    const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, uint16_t* __restrict__ y,
    uint32_t n, uint32_t a_ld, uint32_t b_ld, uint32_t y_ld, float div) {
    typedef typename std::conditional<F32 != 0, float, uint16_t>::type T;
    const uint32_t m = blockIdx.x;
    const T* __restrict__ ar = (const T*)a + (size_t)m * a_ld;
    const T* __restrict__ br = (const T*)b + (size_t)m * b_ld;
    T* __restrict__ yr = (T*)y + (size_t)m * y_ld;

    const auto ld = [](const T* p, uint32_t i) {
        return F32 ? ldg_f32((const float*)p + i) : bf16_to_f32(ldg_u16_at((const uint16_t*)p, i));
    };
    const auto at = [&](uint32_t i) {
        const float u = ld(ar, i), v = ld(br, i);
        float s = MUL ? u * v : u + v;
        if (div != 1.0f) s = F32 ? s / div : bf16_to_f32(f32_to_bf16(s)) / div;
        return s;
    };
    if (PAIR) {
        for (uint32_t i = 2u * threadIdx.x; i < n; i += 2u * THR) {
            const float lo = at(i), hi = at(i + 1);
            if (F32) {
                stg_f32((float*)yr + i, lo);
                stg_f32((float*)yr + i + 1, hi);
            } else {
                stg_u32((uint16_t*)yr + i, bf16x2_rne(lo, hi));
            }
        }
    } else {
        for (uint32_t i = threadIdx.x; i < n; i += THR) {
            const float v = at(i);
            if (F32) stg_f32((float*)yr + i, v);
            else ((uint16_t*)yr)[i] = f32_to_bf16(v);
        }
    }
}

/* ================================================================== cast
 * `from`/`to` are the selector's; the bytes are what the tensors say,
 * which is what ref reads and what this kernel does. An integer pair goes straight through, so a
 * token id above 2^24 survives an i64 -> i32 copy; everything else widens to f32 and narrows once
 * on the way out. One switch per element: cast is a bandwidth op between stages, not a hot loop.
 */
template <int THR>
__global__ __launch_bounds__(THR) void cast_kernel(
    const void* __restrict__ x, void* __restrict__ y, uint32_t n, uint32_t x_ld, uint32_t y_ld,
    int from_dt, int to_dt, int intp) {
    const uint32_t m = blockIdx.x;
    /* Both pitches are in elements of their own dtype, and the row base is in bytes. */
    const char* __restrict__ xr = (const char*)x + (size_t)m * x_ld * (size_t)dt_bytes(from_dt);
    char* __restrict__ yr = (char*)y + (size_t)m * y_ld * (size_t)dt_bytes(to_dt);
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        if (intp) st_int(to_dt, yr, i, ld_int(from_dt, xr, i));
        else      st_elem(to_dt, yr, i, ld_elem(from_dt, xr, i));
    }
}

/* ================================================================== silu / gelu / sigmoid
 * unary: one element, one activation, one narrowing by the store.
 * The three are one kernel with the activation as a template argument, because libr3d's rows serve
 * all three and the body differs by one expression. */
template <int THR, int ACT, int PAIR, int F32>
__global__ __launch_bounds__(THR) void unary_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ y, uint32_t n, uint32_t x_ld,
    uint32_t y_ld) {
    typedef typename std::conditional<F32 != 0, float, uint16_t>::type T;
    const uint32_t m = blockIdx.x;
    const T* __restrict__ xr = (const T*)x + (size_t)m * x_ld;
    T* __restrict__ yr = (T*)y + (size_t)m * y_ld;
    const auto ld = [](const T* p, uint32_t i) {
        return F32 ? ldg_f32((const float*)p + i) : bf16_to_f32(ldg_u16_at((const uint16_t*)p, i));
    };
    if (PAIR) {
        for (uint32_t i = 2u * threadIdx.x; i < n; i += 2u * THR) {
            const float lo = act_of<ACT>(ld(xr, i)), hi = act_of<ACT>(ld(xr, i + 1));
            if (F32) {
                stg_f32((float*)yr + i, lo);
                stg_f32((float*)yr + i + 1, hi);
            } else {
                stg_u32((uint16_t*)yr + i, bf16x2_rne(lo, hi));
            }
        }
    } else {
        for (uint32_t i = threadIdx.x; i < n; i += THR) {
            const float v = act_of<ACT>(ld(xr, i));
            if (F32) stg_f32((float*)yr + i, v);
            else ((uint16_t*)yr)[i] = f32_to_bf16(v);
        }
    }
}

/* ================================================================== act_mul
 * y = act(a) * b, with the activation narrowed to y's dtype first,
 * because the unfused pair stores it -- the rounding is the contract, not a detail:
 *
 *   v = act(a[i]);  y[i] = round_to_y( round_to_y(v) * b[i] )
 *
 * `a` and `b` are the two halves of a fused projection, so each has its own pitch. The shims serve
 * silu, gelu (the exact erf form) and sigmoid; gelu_tanh and quick_gelu are libref's other two
 * kActs entries and are refused, so the row's constraints state which gated MLPs reach this card.
 */
template <int THR, int ACT, int PAIR>
__global__ __launch_bounds__(THR) void act_mul_kernel(
    const uint16_t* __restrict__ g, const uint16_t* __restrict__ u, uint16_t* __restrict__ y,
    uint32_t n, uint32_t g_ld, uint32_t u_ld, uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ gr = g + (size_t)m * g_ld;
    const uint16_t* __restrict__ ur = u + (size_t)m * u_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    /* The activation, narrowed; then the product, narrowed by the store. */
    const auto at = [&](uint32_t i) {
        const float v = bf16_to_f32(f32_to_bf16(act_of<ACT>(bf16_to_f32(ldg_u16_at(gr, i)))));
        return v * bf16_to_f32(ldg_u16_at(ur, i));
    };
    if (PAIR) {
        for (uint32_t i = 2u * threadIdx.x; i < n; i += 2u * THR)
            stg_u32(yr + i, bf16x2_rne(at(i), at(i + 1)));
    } else {
        for (uint32_t i = threadIdx.x; i < n; i += THR)
            ((uint16_t*)yr)[i] = f32_to_bf16(at(i));
    }
}

/* ================================================================== silu_mul
 * y[i] = silu(gate_up[i]) * gate_up[n + i], over ONE fused operand:
 * the gate is the first half, which is the order rad-convert fuses gate and up into one weight in.
 *
 * Nothing is narrowed between the activation and the product. act_mul rounds the activation to y's
 * dtype first because it replaces an unfused pair that stored it; silu_mul is the fused pair, so
 * its product is the f32 activation times the f32 up, rounded once by the store.
 */
template <int THR, int PAIR>
__global__ __launch_bounds__(THR) void silu_mul_kernel(
    const uint16_t* __restrict__ gu, uint16_t* __restrict__ y, uint32_t n, uint32_t gu_ld,
    uint32_t y_ld) {
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ gr = gu + (size_t)m * gu_ld;
    uint16_t* __restrict__ yr = y + (size_t)m * y_ld;
    const auto at = [&](uint32_t i) {
        return act_of<R3D_ACT_SILU>(bf16_to_f32(ldg_u16_at(gr, i))) *
               bf16_to_f32(ldg_u16_at(gr, n + i));
    };
    if (PAIR) {
        for (uint32_t i = 2u * threadIdx.x; i < n; i += 2u * THR)
            stg_u32(yr + i, bf16x2_rne(at(i), at(i + 1)));
    } else {
        for (uint32_t i = threadIdx.x; i < n; i += THR) ((uint16_t*)yr)[i] = f32_to_bf16(at(i));
    }
}

/* ================================================================== rmsnorm_quant_fp8
 * The residual add, the norm, and the E4M3 quantiser of the result in one
 * pass, so the normalised activation never goes back to VRAM. The arithmetic of the quantiser is
 * r3d_quant_fp8.hip's, and `q8_group` before it, bit for bit: `sc = amax * (1.0f /
 * 448.0f)` (the reciprocal MULTIPLIED), `inv = 1.0f / sc`, the codes `v * inv` clamped to +-448 by
 * comparisons and not by fminf/fmaxf (those replace a NaN with the bound; a NaN activation encodes
 * as the format's NaN), and a group whose amax is zero takes scale 1.
 *
 * Two properties of ref's order are load-bearing and are kept:
 *   the residual sum is STORED (narrowed to bf16) before the variance reads it, so the norm is over
 *   the bf16 sum. That is what makes this the `add` + `rmsnorm` pair and not `rmsnorm_add`, which
 *   normalises the unrounded sum;
 *   the quantiser codes the bf16 normalised value, not the f32 one -- `out_bf16`, when asked for,
 *   holds exactly the bytes the codes were taken over.
 *
 * A group is a wave at a time, four consecutive columns a lane per 128 columns, so the amax is a
 * register max plus one wave butterfly (a maximum is exact in any order) and the codes leave as
 * dwords. The normalised row is RECOMPUTED in the second pass from the same operands -- hot in L2,
 * and it keeps the row's length out of LDS, which is what lets one workgroup take a 5120-column row.
 */
template <int THR, int GROUP, int WDT>
__global__ __launch_bounds__(THR) void rmsnorm_quant_fp8_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ res, const void* __restrict__ w,
    uint8_t* __restrict__ q, float* __restrict__ s, uint16_t* __restrict__ ob, uint32_t n,
    float eps, float wadd, uint32_t x_ld, uint32_t res_ld, uint32_t ob_ld, uint32_t q_ld,
    uint32_t s_ld) {
    constexpr uint32_t kChunk = 128u;                     /* a lane's four columns per chunk */
    constexpr uint32_t kNChunk = (uint32_t)GROUP / kChunk; /* chunks a group is: 1, 2 or 4 */
    constexpr float kTopE4M3 = 448.0f;
    __shared__ float lds[THR / kWave];
    const uint32_t m = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)m * x_ld;
    uint16_t* __restrict__ rr = res ? res + (size_t)m * res_ld : nullptr;
    uint8_t* __restrict__ qr = q + (size_t)m * q_ld;
    float* __restrict__ sr = s + (size_t)m * s_ld;
    uint16_t* __restrict__ obr = ob ? ob + (size_t)m * ob_ld : nullptr;

    /* The row the norm is taken over: the residual plane, once the sum has been written to it. */
    const auto src = [&](uint32_t i) {
        return bf16_to_f32((uint16_t)ldg_u16(rr ? rr + i : xr + i));
    };
    if (rr) {
        for (uint32_t i = threadIdx.x; i < n; i += THR)
            rr[i] = f32_to_bf16(bf16_to_f32(ldg_u16_at(xr, i)) + bf16_to_f32(ldg_u16_at(rr, i)));
    }
    __syncthreads();

    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) { const float v = src(i); ss += v * v; }
    ss = block_sum<THR>(ss, lds);
    const float rs = 1.0f / sqrtf(ss / (float)n + eps);
    __syncthreads();

    const uint32_t wave = threadIdx.x / kWave, lane = threadIdx.x & (kWave - 1);
    const uint32_t ngrp = n / (uint32_t)GROUP;
    for (uint32_t g = wave; g < ngrp; g += (uint32_t)(THR / kWave)) {
        const uint32_t base = g * (uint32_t)GROUP + 4u * lane;
        float v[kNChunk][4];
        float am = 0.0f;
        for (uint32_t j = 0; j < kNChunk; ++j) {
            for (uint32_t c = 0; c < 4u; ++c) {
                const uint32_t i = base + j * kChunk + c;
                /* The gain's dtype is the checkpoint's, read off the operand -- WDT, the plain
                 * rmsnorm's leg: the gains are read as stored, and the op's `dtype` names the
                 * ACTIVATION, never the weight. */
                const float gain = (!w ? 1.0f
                                  : WDT == R3D_DT_F32 ? ldg_f32((const float*)w + i)
                                                      : bf16_to_f32(ldg_u16_at((const uint16_t*)w, i)))
                                 + wadd;
                /* The normalised value narrowed to bf16, which is the value ref quantises and the
                 * value out_bf16 holds. */
                const float h = bf16_to_f32(f32_to_bf16(src(i) * rs * gain));
                v[j][c] = h;
                const float t = std::fabs(h);
                if (t > am) am = t;              /* ref's `if (m > amax) amax = m`: NaN is skipped */
            }
        }
        am = wave_max(am);
        const float sc = am > 0.0f ? am * (1.0f / kTopE4M3) : 1.0f;
        const float inv = 1.0f / sc;
        if (lane == 0) stg_f32(sr + g, sc);
        for (uint32_t j = 0; j < kNChunk; ++j) {
            float c0 = v[j][0] * inv, c1 = v[j][1] * inv, c2 = v[j][2] * inv, c3 = v[j][3] * inv;
            if (c0 > kTopE4M3) c0 = kTopE4M3; else if (c0 < -kTopE4M3) c0 = -kTopE4M3;
            if (c1 > kTopE4M3) c1 = kTopE4M3; else if (c1 < -kTopE4M3) c1 = -kTopE4M3;
            if (c2 > kTopE4M3) c2 = kTopE4M3; else if (c2 < -kTopE4M3) c2 = -kTopE4M3;
            if (c3 > kTopE4M3) c3 = kTopE4M3; else if (c3 < -kTopE4M3) c3 = -kTopE4M3;
            const uint32_t lo = e4m3x2_pack(c0, c1), hi = e4m3x2_pack(c2, c3);
            stg_code4(qr + base + j * kChunk, lo | (hi << 16));
            if (obr) {
                stg_u32(obr + base + j * kChunk, bf16x2_rne(v[j][0], v[j][1]));
                stg_u32(obr + base + j * kChunk + 2u, bf16x2_rne(v[j][2], v[j][3]));
            }
        }
    }
}

/* ================================================================== softmax
 * In f32:
 *
 *   mx  = max_i x[i]
 *   e   = exp(x[i] - mx);  y[i] = e;  sum += e            ascending
 *   inv = 1 / sum;         y[i] = y[i] * inv              read back from y
 *
 * The maximum is subtracted first (a 60-nat row overflows f32's exp otherwise); the denominator is
 * the ascending sum, which is what `block_sum` brackets. The kernel keeps `e` in a register
 * between the two passes, which is exact: ref reads it back from an f32 plane, and an f32 store
 * and load round nothing. The shim therefore requires y to be f32 -- a bf16 y would round `e`
 * before the multiply and ref would not, and libref itself casts f32 to f32 for this op.
 */
template <int THR>
__global__ __launch_bounds__(THR) void softmax_kernel(
    const float* __restrict__ x, float* __restrict__ y, uint32_t n, uint32_t x_ld, uint32_t y_ld) {
    __shared__ float lds[THR / kWave];
    const uint32_t m = blockIdx.x;
    const float* __restrict__ xr = x + (size_t)m * x_ld;
    float* __restrict__ yr = y + (size_t)m * y_ld;

    float mx = -INFINITY;
    for (uint32_t i = threadIdx.x; i < n; i += THR) mx = fmaxf(mx, ldg_f32(xr + i));
    mx = block_max<THR>(mx, lds);

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const float e = expf(ldg_f32(xr + i) - mx);
        sum += e;
        stg_f32(yr + i, e);
    }
    const float inv = 1.0f / block_sum<THR>(sum, lds);
    for (uint32_t i = threadIdx.x; i < n; i += THR) stg_f32(yr + i, ldg_f32(yr + i) * inv);
}

/* ================================================================== scale / fill
 * scale is y = x * s, s one scalar; fill is y = v over the whole
 * plane, and both narrow once, at the store.
 *
 * scale's scalar is the op's [1] weight plane, in device memory, and the kernel reads it: a launch
 * may not copy device memory to the host, and ref reads the same single value because its operands
 * are host planes. Every lane loads the same address, so it is one uniform load per block; the
 * product is taken in f32 and the store narrows, which is ref's `x * s` with round_dt(f32) as the
 * store.
 */
template <int THR, int PAIR, int F32>
__global__ __launch_bounds__(THR) void scale_kernel(
    const uint16_t* __restrict__ x, const void* __restrict__ sp, uint16_t* __restrict__ y,
    uint32_t n, int s_dt, uint32_t x_ld, uint32_t y_ld) {
    typedef typename std::conditional<F32 != 0, float, uint16_t>::type T;
    const uint32_t m = blockIdx.x;
    const T* __restrict__ xr = (const T*)x + (size_t)m * x_ld;
    T* __restrict__ yr = (T*)y + (size_t)m * y_ld;
    const float v = s_dt == R3D_DT_F32 ? ldg_f32((const float*)sp)
                                       : bf16_to_f32(ldg_u16((const uint16_t*)sp));
    const auto ld = [](const T* p, uint32_t i) {
        return F32 ? ldg_f32((const float*)p + i) : bf16_to_f32(ldg_u16_at((const uint16_t*)p, i));
    };
    if (PAIR) {
        for (uint32_t i = 2u * threadIdx.x; i < n; i += 2u * THR)
            stg_u32((uint16_t*)yr + i, bf16x2_rne(ld(xr, i) * v, ld(xr, i + 1) * v));
    } else {
        for (uint32_t i = threadIdx.x; i < n; i += THR) {
            const float r = ld(xr, i) * v;
            if (F32) stg_f32((float*)yr + i, r);
            else ((uint16_t*)yr)[i] = f32_to_bf16(r);
        }
    }
}

template <int THR>
__global__ __launch_bounds__(THR) void fill_kernel(void* __restrict__ y, uint32_t n, uint32_t y_ld,
                                                   int dt, float v) {
    const uint32_t m = blockIdx.x;
    char* __restrict__ yr = (char*)y + (size_t)m * y_ld * (size_t)dt_bytes(dt);
    for (uint32_t i = threadIdx.x; i < n; i += THR) st_elem(dt, yr, i, v);
}

/* ================================================================== rope
 * rope_freqs, rope_head, ref_rope -- in place on the fused qkv row, q heads
 * then k heads then v untouched:
 *
 *   inv[i] = 1 / pow(theta, (2i)/rot)                     the ladder, den == rot
 *   ang    = (pos[m] / scale) * inv[i]
 *   c, s   = cos(ang), sin(ang)
 *   i0, i1 = (i, i + rot/2) neox, (2i, 2i + 1) gptj
 *   h[i0]  = x0*c - x1*s ;  h[i1] = x0*s + x1*c           each narrowed once
 *
 * The pair lives in one thread, so the in-place store needs no exchange. `scale` is the position
 * scale and divides the position, as ref's rope_pos does (the value that multiplies the rotated
 * pair is YaRN's magnitude, and the shims refuse YaRN, so the magnitude is 1).
 *
 * The frequency is recomputed per pair, not built in a plane: powf runs once per pair per head,
 * off the step's critical path, and no scratch is allocated. nfreq == rot/2 always, so no pair
 * sits past the ladder (ref zeroes inv[] past nfreq for the proportional ladder it supports and
 * libr3d refuses).
 */
template <int THR, int GPTJ>
__global__ __launch_bounds__(THR) void rope_kernel(
    uint16_t* __restrict__ qkv, const int32_t* __restrict__ pos, uint32_t head_dim,
    uint32_t n_rope_heads, uint32_t rot, float theta, float scale, uint32_t row_pitch) {
    const uint32_t m = blockIdx.x;
    const float div = scale != 0.0f ? scale : 1.0f;
    const float p = (float)ldg_i32(pos + m) / div;
    const uint32_t half = rot / 2;
    const uint32_t total = n_rope_heads * half;
    uint16_t* __restrict__ row = qkv + (size_t)m * row_pitch;

    for (uint32_t j = threadIdx.x; j < total; j += THR) {
        const uint32_t h = j / half, i = j - h * half;
        /* ref's ladder: `1.0f / ref_pow(theta, (float)(2i) / den)`, and ref_pow is the double pow
         * narrowed once. powf here would put the reciprocal a ULP off, and at position 262144 a
         * ULP of a frequency is a hundredth of a radian of rotation -- the ladder is the one place
         * in this kernel where f32 is not enough. The angle itself stays f32, as ref's is. */
        const float inv = 1.0f / (float)std::pow((double)theta, (double)((float)(2 * i) / (float)rot));
        float s, c;
        sincosf(p * inv, &s, &c);
        const uint32_t i0 = h * head_dim + (GPTJ ? 2u * i : i);
        const uint32_t i1 = h * head_dim + (GPTJ ? 2u * i + 1u : i + half);
        const float x0 = bf16_to_f32(ldg_u16_at(row, i0));
        const float x1 = bf16_to_f32(ldg_u16_at(row, i1));
        ((uint16_t*)row)[i0] = f32_to_bf16(x0 * c - x1 * s);
        ((uint16_t*)row)[i1] = f32_to_bf16(x0 * s + x1 * c);
    }
}

/* ================================================================== kv_store
 * With the cache's strides read off the operand (kv_view):
 *
 *   s = slot[t];  if (s < 0) skip;                      a padded row has no cache slot
 *   h, d = the kv head and the element; j = h*head_dim + d
 *   base = (s/block)*sb + h*sh + (s%block)*so
 *   cache[base + d*sd]           = k[t*k_ld + j] * k_inv
 *   cache[base + (hd + d)*sd]    = v[t*v_ld + j] * v_inv
 *
 * k and v rows are not dense: a de-interleaved projection gives [tokens, kv_heads, head_dim], and
 * otherwise they are a column slice of the fused QKV plane -- 2560 wide against a 256-wide row for
 * 16 heads and 2 kv heads. Deriving the stride from the shape would read every token after the
 * first from the wrong rows, so the pitch is an argument.
 *
 * k_inv and v_inv are the reciprocal of the reader's descales and are 1 for every cache that is
 * not fp8, so the multiply below is exact -- and for the fp8 cache it is the divide ref does, in
 * the one place where a reciprocal costs nothing: a value times 1/scale against ref's
 * `ldt(k,..) * ks` differs by at most the rounding of the reciprocal, far under the fp8 code step.
 */
template <int THR>
__global__ __launch_bounds__(THR) void kv_store_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ v, const int32_t* __restrict__ sm,
    void* __restrict__ kv, uint32_t head_dim, uint32_t kv_heads, uint32_t block_size, int kv_dt,
    float k_inv, float v_inv, uint32_t k_ld, uint32_t v_ld, int64_t sb, int64_t sh, int64_t so,
    int64_t sd, int64_t slot_max) {
    const uint32_t t = blockIdx.x;
    const int32_t s = ldg_i32(sm + t);
    /* A negative slot is padding and a slot at or past the cache's own span names a block the
     * cache does not have: ref refuses such a launch on the host, where it can see the block
     * count, and a kernel that has read the slot can only drop it. A slot lost, not a step lost. */
    if (s < 0 || (int64_t)s >= slot_max) return;
    const int64_t blk = (int64_t)s / (int64_t)block_size, off = (int64_t)s % (int64_t)block_size;

    const uint32_t total = kv_heads * head_dim;
    const uint16_t* __restrict__ kr = k + (size_t)t * k_ld;
    const uint16_t* __restrict__ vr = v + (size_t)t * v_ld;
    const int es = kv_dt == R3D_DT_BF16 ? 2 : kv_dt == R3D_DT_F16 ? 2 : 1;
    char* __restrict__ plane = (char*)kv;

    for (uint32_t j = threadIdx.x; j < total; j += THR) {
        const uint32_t h = j / head_dim, d = j - h * head_dim;
        const int64_t cb = (blk * sb + (int64_t)h * sh + off * so) * es;
        st_elem(kv_dt, plane + cb, (size_t)d * sd, bf16_to_f32(ldg_u16_at(kr, j)) * k_inv);
        st_elem(kv_dt, plane + cb, (size_t)(head_dim + d) * sd,
                bf16_to_f32(ldg_u16_at(vr, j)) * v_inv);
    }
}

/* ================================================================== embed_lookup
 * embed_gather:
 *
 *   id = tok[m] - vocab_offset
 *   x[m][i] = (0 <= id < n_vocab) ? wte[id*w_ld + i] * wscale * mult : 0
 *
 * A negative id, or one at or past the table, is a zero row: padding's row, and in the
 * vocab-parallel split another rank's token -- summing the ranks reconstructs x. ref validates the
 * ids on the host when the table is whole and the world is one rank; a kernel cannot, so the table
 * bound is an argument and an out-of-range id writes zeros.
 *
 * The table's dtype is the launcher's `dt`: bf16, f16, f32, or E4M3 codes with the table's one
 * scale in `wscale` (embed_lookup_q's byte plane, whose scale is a parameter because quantising
 * the table sees the whole table and the gather sees one row). `mult` is Gemma's sqrt(hidden),
 * applied after the scale, so with wscale 1 the product is ref's single rounding.
 */
template <int THR>
__global__ __launch_bounds__(THR) void embed_lookup_kernel(
    const int32_t* __restrict__ ids, const void* __restrict__ wte, void* __restrict__ x,
    uint32_t n_embd, int32_t n_vocab, int32_t vocab_off, float wscale, float mult, uint32_t x_ld,
    int64_t w_ld, int dt, int pair) {
    const uint32_t m = blockIdx.x;
    const int32_t id = ldg_i32(ids + m) - vocab_off;
    char* __restrict__ xr = (char*)x + (size_t)m * x_ld * (size_t)dt_bytes(dt);

    if (id < 0 || id >= n_vocab) {
        for (uint32_t i = threadIdx.x; i < n_embd; i += THR) st_elem(dt, xr, i, 0.0f);
        return;
    }
    const char* __restrict__ wr = (const char*)wte + (int64_t)id * w_ld * (int64_t)dt_bytes(dt);
    if (pair) {
        uint16_t* __restrict__ xp = (uint16_t*)xr;
        for (uint32_t i = 2u * threadIdx.x; i < n_embd; i += 2u * THR)
            stg_u32(xp + i, bf16x2_rne(ld_elem(dt, wr, i) * wscale * mult,
                                       ld_elem(dt, wr, i + 1) * wscale * mult));
    } else {
        for (uint32_t i = threadIdx.x; i < n_embd; i += THR)
            st_elem(dt, xr, i, ld_elem(dt, wr, i) * wscale * mult);
    }
}

/* ================================================================== gather_rows / scatter_rows
 * y[r] = x[idx[r]] and x[idx[r]] = v[r], rows of n, and a row move
 * with no arithmetic: ld_elem/st_elem is the identity for bf16, f16, f32, i32 and i64, and the
 * fp8 code round trip is exact (every code decodes and re-encodes to itself).
 *
 * A negative index is padding: gather writes a zero row, scatter writes nothing -- an unsampled row
 * of a step, a slot with no counterpart. An index at or past the operand's row count takes the
 * same path, where ref refuses the launch (see the file header).
 *
 * scatter_rows' destination is INOUT and the write is partial: unnamed rows keep their contents,
 * because the destination is often a pool other sequences are in. Its indices must be distinct --
 * RAD_OPD_F_IDX_UNIQUE says so in the schema -- since two rows landing on one destination have no
 * defined winner in parallel, and an op whose oracle and kernel may disagree is not checkable.
 */
template <int THR>
__global__ __launch_bounds__(THR) void gather_rows_kernel(
    const void* __restrict__ x, const int32_t* __restrict__ idx, void* __restrict__ y, uint32_t n,
    uint32_t x_ld, uint32_t y_ld, int dt, int32_t es, int64_t src_rows) {
    const uint32_t m = blockIdx.x;
    const int32_t r = ldg_i32(idx + m);
    char* __restrict__ yr = (char*)y + (size_t)m * y_ld * (size_t)es;
    /* A negative index is padding and an index at or past the source plane is one libref refuses on
     * the host, where it can read the plane. Both take the zero row: reading x past its end is the
     * failure this bound exists to stop. `src_rows` 0 says the caller had no plane count, which is a
     * shim bug, not a bound. */
    if (r < 0 || (src_rows > 0 && (int64_t)r >= src_rows)) {
        for (uint32_t i = threadIdx.x; i < n; i += THR) st_elem(dt, yr, i, 0.0f);
        return;
    }
    const char* __restrict__ xr = (const char*)x + (size_t)r * x_ld * (size_t)es;
    for (uint32_t i = threadIdx.x; i < n; i += THR) st_elem(dt, yr, i, ld_elem(dt, xr, i));
}

template <int THR>
__global__ __launch_bounds__(THR) void scatter_rows_kernel(
    const void* __restrict__ v, const int32_t* __restrict__ idx, void* __restrict__ x, uint32_t n,
    uint32_t v_ld, uint32_t x_ld, int dt, int32_t es, int64_t dst_rows) {
    const uint32_t m = blockIdx.x;
    const int32_t r = ldg_i32(idx + m);
    /* A negative index is padding, one at or past the destination plane is libref's refused step:
     * dropped, so no row of x is written outside the plane. `dst_rows` 0 is unchecked. */
    if (r < 0 || (dst_rows > 0 && (int64_t)r >= dst_rows)) return;
    const char* __restrict__ vr = (const char*)v + (size_t)m * v_ld * (size_t)es;
    char* __restrict__ xr = (char*)x + (size_t)r * x_ld * (size_t)es;
    for (uint32_t i = threadIdx.x; i < n; i += THR) st_elem(dt, xr, i, ld_elem(dt, vr, i));
}

}  /* namespace */

/* ================================================================== the launchers
 *
 * Pointers as longs, geometry as ints, the stream as a long. Each checks what a `__global__`
 * cannot -- nulls, non-positive extents, a pitch that does not describe the operand -- takes the
 * thread count from the row width, decides the pair layout from the pitches and the base
 * alignments, and issues one kernel on the stream it was handed. A pitch of 0 means "dense, so n".
 */

/* The pair layout: a lane owning two consecutive columns, its result stored as one dword. Legal
 * when n, both pitches and the base are even, so every element of every row lands on an even
 * element index -- 4-byte alignment at 2 bytes an element. */
static inline int row_pair(long base, int n, int ld1, int ld2) {
    return ((n & 1) == 0) && ((ld1 & 1) == 0) && ((ld2 & 1) == 0) && ((base & 3) == 0);
}

/* One grid, one launch, per launcher: `grid` is the row count (blockIdx.x is the row), and the
 * two thread counts are the template arguments the grid was sized for. */
#define R3D_ROW2(kernel, thr, grid, stream, ...)                                                \
    do {                                                                                        \
        if ((thr) == (int)kNarrow)                                                             \
            R3D_KLAUNCH(R3D_KNAME(kernel<kNarrow>), dim3((uint32_t)(grid)), dim3(kNarrow),      \
                        (hipStream_t)(uintptr_t)(stream), __VA_ARGS__);                         \
        else                                                                                    \
            R3D_KLAUNCH(R3D_KNAME(kernel<kThreads>), dim3((uint32_t)(grid)), dim3(kThreads),    \
                        (hipStream_t)(uintptr_t)(stream), __VA_ARGS__);                         \
    } while (0)

#define R3D_ROPE2(thr, grid, stream, ...)                                                       \
    do {                                                                                        \
        if ((thr) == (int)kNarrow) {                                                            \
            if (gptj) R3D_ROPE_K(kNarrow, 1, grid, stream, __VA_ARGS__);                        \
            else      R3D_ROPE_K(kNarrow, 0, grid, stream, __VA_ARGS__);                        \
        } else {                                                                                \
            if (gptj) R3D_ROPE_K(kThreads, 1, grid, stream, __VA_ARGS__);                       \
            else      R3D_ROPE_K(kThreads, 0, grid, stream, __VA_ARGS__);                       \
        }                                                                                       \
    } while (0)

#define R3D_ROPE_K(THR, GPTJ, grid, stream, ...)                                                \
    R3D_KLAUNCH(R3D_KNAME(rope_kernel<THR, GPTJ>), dim3((uint32_t)(grid)), dim3(THR),           \
                (hipStream_t)(uintptr_t)(stream), __VA_ARGS__)

/* A row count a dim3 can carry and a grid-strided loop cannot be asked to exceed. */
static inline int row_count_ok(long long M) { return M > 0 && M <= 0x7fffffffLL; }

static int rmsnorm_launch(long x, long w, long y, int M, int n, float eps, float wadd, float mult,
                          int x_ld, int y_ld, int wdt, int f32, long stream) {
    if (!x || !y || (wdt != R3D_DT_N && !w)) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (wdt != R3D_DT_N && wdt != R3D_DT_BF16 && wdt != R3D_DT_F32) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    if (x_ld < n || y_ld < n) return -3;
    const int thr = row_threads(n);
    const int pair = f32 ? 0 : row_pair(y, n, x_ld, y_ld);
    const uint16_t* xp = (const uint16_t*)(uintptr_t)x;
    uint16_t* yp = (uint16_t*)(uintptr_t)y;
    const void* wp = (const void*)(uintptr_t)w;
#define R3D_RMS(THR, WDT)                                                                       \
    R3D_KLAUNCH(R3D_KNAME(rmsnorm_kernel<THR, WDT, 0>), dim3((uint32_t)M), dim3(THR),           \
                (hipStream_t)(uintptr_t)stream, xp, wp, yp, (uint32_t)n, eps, wadd, mult,       \
                (uint32_t)x_ld, (uint32_t)y_ld, pair)
    if (wdt == R3D_DT_F32) {
        if (thr == (int)kNarrow) R3D_RMS(kNarrow, R3D_DT_F32); else R3D_RMS(kThreads, R3D_DT_F32);
    } else if (wdt == R3D_DT_BF16) {
        if (thr == (int)kNarrow) R3D_RMS(kNarrow, R3D_DT_BF16); else R3D_RMS(kThreads, R3D_DT_BF16);
    } else {
        if (thr == (int)kNarrow) R3D_RMS(kNarrow, R3D_DT_N); else R3D_RMS(kThreads, R3D_DT_N);
    }
#undef R3D_RMS
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The f32 leg: the same two passes over an f32 row, and a store that narrows nothing. */
extern "C" int r3d_rmsnorm_f32(long x, long w, long y, int M, int n, float eps, float wadd,
                               float mult, int x_ld, int y_ld, int wdt, long stream) {
    if (!x || !y || (wdt != R3D_DT_N && !w)) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (wdt != R3D_DT_N && wdt != R3D_DT_BF16 && wdt != R3D_DT_F32) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    if (x_ld < n || y_ld < n) return -3;
#define R3D_RMSF(THR, WDT)                                                                      \
    R3D_KLAUNCH(R3D_KNAME(rmsnorm_kernel<THR, WDT, 1>), dim3((uint32_t)M), dim3(THR),           \
                (hipStream_t)(uintptr_t)stream, (const uint16_t*)(uintptr_t)x,                   \
                (const void*)(uintptr_t)w, (uint16_t*)(uintptr_t)y, (uint32_t)n, eps, wadd,      \
                mult, (uint32_t)x_ld, (uint32_t)y_ld, 0)
    if (row_threads(n) == (int)kNarrow) {
        if (wdt == R3D_DT_F32) R3D_RMSF(kNarrow, R3D_DT_F32);
        else if (wdt == R3D_DT_BF16) R3D_RMSF(kNarrow, R3D_DT_BF16);
        else R3D_RMSF(kNarrow, R3D_DT_N);
    } else {
        if (wdt == R3D_DT_F32) R3D_RMSF(kThreads, R3D_DT_F32);
        else if (wdt == R3D_DT_BF16) R3D_RMSF(kThreads, R3D_DT_BF16);
        else R3D_RMSF(kThreads, R3D_DT_N);
    }
#undef R3D_RMSF
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The bf16 entry point, in terms of the same body. */
extern "C" int r3d_rmsnorm_bf16(long x, long w, long y, int M, int n, float eps, float wadd,
                                float mult, int x_ld, int y_ld, int wdt, long stream) {
    return rmsnorm_launch(x, w, y, M, n, eps, wadd, mult, x_ld, y_ld, wdt, 0, stream);
}

/* The fused add + norm + E4M3 quantiser. `res` is the residual plane, INOUT, and 0 when the op has
 * no residual; `ob` is the optional bf16 copy of exactly what was quantised. `group` is the
 * quantiser's block and must divide n: a partial group needs the zero-padded store the standalone
 * quantiser writes, and this row's band is a checkpoint whose block divides its hidden width.
 * Pitches are in elements of their own operand; `s_ld` in scales, so a dense scale plane is n/group.
 * 0, -1, -2, -3, -4. */
extern "C" int r3d_rmsnorm_quant_fp8(long x, long res, long w, long q, long s, long ob, int M,
                                     int n, int group, float eps, float wadd, int x_ld, int res_ld,
                                     int ob_ld, int q_ld, int s_ld, int wdt, long stream) {
    if (!x || !w || !q || !s) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (group != 128 && group != 256 && group != 512) return -2;
    if (n % group) return -2;
    if (!x_ld) x_ld = n;
    if (!res_ld) res_ld = n;
    if (!ob_ld) ob_ld = n;
    if (!q_ld) q_ld = n;
    if (!s_ld) s_ld = n / group;
    if (x_ld < n || res_ld < n || ob_ld < n || q_ld < n || s_ld < n / group) return -3;
    const int thr = row_threads(n);
#define R3D_RQ(THR, GROUP, WDT)                                                                 \
    R3D_KLAUNCH(R3D_KNAME(rmsnorm_quant_fp8_kernel<THR, GROUP, WDT>), dim3((uint32_t)M),         \
                dim3(THR), (hipStream_t)(uintptr_t)stream, (const uint16_t*)(uintptr_t)x,        \
                (uint16_t*)(uintptr_t)res, (const void*)(uintptr_t)w,                             \
                (uint8_t*)(uintptr_t)q, (float*)(uintptr_t)s, (uint16_t*)(uintptr_t)ob,           \
                (uint32_t)n, eps, wadd, (uint32_t)x_ld, (uint32_t)res_ld, (uint32_t)ob_ld,        \
                (uint32_t)q_ld, (uint32_t)s_ld)
#define R3D_RQ_G(THR, WDT)                                                                       \
    do {                                                                                         \
        if (group == 128) R3D_RQ(THR, 128, WDT);                                                 \
        else if (group == 256) R3D_RQ(THR, 256, WDT);                                            \
        else R3D_RQ(THR, 512, WDT);                                                              \
    } while (0)
#define R3D_RQ_W(THR)                                                                            \
    do {                                                                                         \
        if (wdt == R3D_DT_F32) R3D_RQ_G(THR, R3D_DT_F32);                                        \
        else R3D_RQ_G(THR, R3D_DT_BF16);                                                         \
    } while (0)
    if (thr == (int)kNarrow) R3D_RQ_W(kNarrow); else R3D_RQ_W(kThreads);
#undef R3D_RQ_W
#undef R3D_RQ_G
#undef R3D_RQ
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_layernorm_bf16(long x, long w, long b, long y, int M, int n, float eps,
                                  int x_ld, int y_ld, long stream) {
    if (!x || !w || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    if (x_ld < n || y_ld < n) return -3;
    const int thr = row_threads(n);
    const int pair = row_pair(y, n, x_ld, y_ld);
    R3D_ROW2(layernorm_kernel, thr, M, stream, (const uint16_t*)(uintptr_t)x,
             (const uint16_t*)(uintptr_t)w, (const uint16_t*)(uintptr_t)b,
             (uint16_t*)(uintptr_t)y, (uint32_t)n, eps, (uint32_t)x_ld, (uint32_t)y_ld, pair);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The binary pair: one kernel over add/mul, bf16/f32 and the pair layout, with the broadcast
 * expressed as b's row pitch (0 makes one row ride every row). */
static int binary_launch(long a, long b, long y, int M, int n, int a_ld, int b_ld, int y_ld,
                         float div, int bcast, int mul, int f32, long stream) {
    if (!a || !b || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!(div != 0.0f) || !std::isfinite(div)) return -2;
    if (!a_ld) a_ld = n;
    if (!y_ld) y_ld = n;
    if (!bcast) {
        if (!b_ld) b_ld = n;
    } else {
        b_ld = 0;
    }
    if (a_ld < n || y_ld < n || (!bcast && b_ld < n)) return -3;
    const int thr = row_threads(n);
    const int pair = f32 ? 0 : row_pair(y, n, a_ld, y_ld) && (bcast || (b_ld & 1) == 0);
#define R3D_BIN(THR, MUL, PAIR, F32)                                                            \
    R3D_KLAUNCH(R3D_KNAME(binary_kernel<THR, MUL, PAIR, F32>), dim3((uint32_t)M), dim3(THR),    \
                (hipStream_t)(uintptr_t)stream, (const uint16_t*)(uintptr_t)a,                   \
                (const uint16_t*)(uintptr_t)b, (uint16_t*)(uintptr_t)y, (uint32_t)n,             \
                (uint32_t)a_ld, (uint32_t)b_ld, (uint32_t)y_ld, div)
#define R3D_BIN_M(THR)                                                                          \
    do {                                                                                        \
        if (mul) { if (pair) R3D_BIN(THR, 1, 1, 0); else R3D_BIN(THR, 1, 0, 0); }               \
        else     { if (pair) R3D_BIN(THR, 0, 1, 0); else R3D_BIN(THR, 0, 0, 0); }               \
    } while (0)
#define R3D_BIN_MF(THR)                                                                         \
    do {                                                                                        \
        if (mul) R3D_BIN(THR, 1, 0, 1); else R3D_BIN(THR, 0, 0, 1);                             \
    } while (0)
    if (f32) { if (thr == (int)kNarrow) R3D_BIN_MF(kNarrow); else R3D_BIN_MF(kThreads); }
    else     { if (thr == (int)kNarrow) R3D_BIN_M(kNarrow);  else R3D_BIN_M(kThreads); }
#undef R3D_BIN_MF
#undef R3D_BIN_M
#undef R3D_BIN
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_add_bf16(long a, long b, long y, int M, int n, int a_ld, int b_ld, int y_ld,
                            float div, int bcast, long stream) {
    return binary_launch(a, b, y, M, n, a_ld, b_ld, y_ld, div, bcast, 0, 0, stream);
}

extern "C" int r3d_add_f32(long a, long b, long y, int M, int n, int a_ld, int b_ld, int y_ld,
                           float div, int bcast, long stream) {
    return binary_launch(a, b, y, M, n, a_ld, b_ld, y_ld, div, bcast, 0, 1, stream);
}

extern "C" int r3d_mul_bf16(long a, long b, long y, int M, int n, int a_ld, int b_ld, int y_ld,
                            int bcast, long stream) {
    return binary_launch(a, b, y, M, n, a_ld, b_ld, y_ld, 1.0f, bcast, 1, 0, stream);
}

extern "C" int r3d_mul_f32(long a, long b, long y, int M, int n, int a_ld, int b_ld, int y_ld,
                           int bcast, long stream) {
    return binary_launch(a, b, y, M, n, a_ld, b_ld, y_ld, 1.0f, bcast, 1, 1, stream);
}

/* The cast's dtype codes: every pair of the dtypes the library moves. The shim translates the
 * tensor dtypes with r3d_dt_code and checks them here, so a call from a tool with an unlisted
 * pair is refused rather than run over a switch's default. */
static int cast_pair_ok(int from, int to) {
    static const int kOk[] = { R3D_DT_BF16, R3D_DT_F16, R3D_DT_F32, R3D_DT_I32, R3D_DT_I64,
                               R3D_DT_U8, R3D_DT_I8 };
    int okf = 0, okt = 0;
    for (int i = 0; i < (int)(sizeof kOk / sizeof kOk[0]); ++i) {
        okf = okf || from == kOk[i];
        okt = okt || to == kOk[i];
    }
    return okf && okt;
}

extern "C" int r3d_cast(long x, long y, int M, int n, int x_ld, int y_ld, int from_dt, int to_dt,
                        long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!cast_pair_ok(from_dt, to_dt)) return -2;
    const int fs = r3d_dt_size(from_dt), ts = r3d_dt_size(to_dt);
    if (!fs || !ts) return -2;
    /* The pitches are in elements of each operand's own dtype; 0 is dense. */
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    if (x_ld < n || y_ld < n) return -3;
    const int thr = row_threads(n);
    R3D_ROW2(cast_kernel, thr, M, stream, (const void*)(uintptr_t)x, (void*)(uintptr_t)y,
             (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld, from_dt, to_dt,
             int_pair(from_dt) && int_pair(to_dt) ? 1 : 0);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The unary activations: one launcher body, four template arguments. */
static int unary_launch(long x, long y, int M, int n, int x_ld, int y_ld, int act, int f32,
                        long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    if (x_ld < n || y_ld < n) return -3;
    const int thr = row_threads(n);
    const int pair = f32 ? 0 : row_pair(y, n, x_ld, y_ld);
#define R3D_UN(THR, ACT, PAIR, F32)                                                             \
    R3D_KLAUNCH(R3D_KNAME(unary_kernel<THR, ACT, PAIR, F32>), dim3((uint32_t)M), dim3(THR),     \
                (hipStream_t)(uintptr_t)stream, (const uint16_t*)(uintptr_t)x,                   \
                (uint16_t*)(uintptr_t)y, (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld)
#define R3D_UN_A(THR)                                                                           \
    do {                                                                                        \
        if (act == R3D_ACT_SILU) { if (pair) R3D_UN(THR, R3D_ACT_SILU, 1, 0);                    \
                                   else      R3D_UN(THR, R3D_ACT_SILU, 0, 0); }                 \
        else if (act == R3D_ACT_GELU) { if (pair) R3D_UN(THR, R3D_ACT_GELU, 1, 0);               \
                                        else      R3D_UN(THR, R3D_ACT_GELU, 0, 0); }            \
        else { if (pair) R3D_UN(THR, R3D_ACT_SIGMOID, 1, 0);                                     \
               else      R3D_UN(THR, R3D_ACT_SIGMOID, 0, 0); }                                  \
    } while (0)
#define R3D_UN_AF(THR)                                                                          \
    do {                                                                                        \
        if (act == R3D_ACT_SILU) R3D_UN(THR, R3D_ACT_SILU, 0, 1);                                \
        else if (act == R3D_ACT_GELU) R3D_UN(THR, R3D_ACT_GELU, 0, 1);                           \
        else R3D_UN(THR, R3D_ACT_SIGMOID, 0, 1);                                                \
    } while (0)
    if (f32) { if (thr == (int)kNarrow) R3D_UN_AF(kNarrow); else R3D_UN_AF(kThreads); }
    else     { if (thr == (int)kNarrow) R3D_UN_A(kNarrow);  else R3D_UN_A(kThreads); }
#undef R3D_UN_AF
#undef R3D_UN_A
#undef R3D_UN
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_silu_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream) {
    return unary_launch(x, y, M, n, x_ld, y_ld, R3D_ACT_SILU, 0, stream);
}

extern "C" int r3d_gelu_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream) {
    return unary_launch(x, y, M, n, x_ld, y_ld, R3D_ACT_GELU, 0, stream);
}

extern "C" int r3d_sigmoid_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream) {
    return unary_launch(x, y, M, n, x_ld, y_ld, R3D_ACT_SIGMOID, 0, stream);
}

extern "C" int r3d_silu_f32(long x, long y, int M, int n, int x_ld, int y_ld, long stream) {
    return unary_launch(x, y, M, n, x_ld, y_ld, R3D_ACT_SILU, 1, stream);
}

extern "C" int r3d_act_mul_bf16(long gate, long up, long y, int M, int n, int g_ld, int u_ld,
                                int y_ld, int act, long stream) {
    if (!gate || !up || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (act != R3D_ACT_SILU && act != R3D_ACT_GELU && act != R3D_ACT_SIGMOID) return -2;
    if (!g_ld) g_ld = n;
    if (!u_ld) u_ld = n;
    if (!y_ld) y_ld = n;
    if (g_ld < n || u_ld < n || y_ld < n) return -3;
    const int thr = row_threads(n);
    const int pair = row_pair(y, n, g_ld, y_ld) && (u_ld & 1) == 0;
#define R3D_AM(THR, ACT, PAIR)                                                                  \
    R3D_KLAUNCH(R3D_KNAME(act_mul_kernel<THR, ACT, PAIR>), dim3((uint32_t)M), dim3(THR),        \
                (hipStream_t)(uintptr_t)stream, (const uint16_t*)(uintptr_t)gate,                \
                (const uint16_t*)(uintptr_t)up, (uint16_t*)(uintptr_t)y, (uint32_t)n,            \
                (uint32_t)g_ld, (uint32_t)u_ld, (uint32_t)y_ld)
#define R3D_AM_A(THR)                                                                           \
    do {                                                                                        \
        if (act == R3D_ACT_SILU) { if (pair) R3D_AM(THR, R3D_ACT_SILU, 1);                       \
                                   else      R3D_AM(THR, R3D_ACT_SILU, 0); }                    \
        else if (act == R3D_ACT_GELU) { if (pair) R3D_AM(THR, R3D_ACT_GELU, 1);                  \
                                        else      R3D_AM(THR, R3D_ACT_GELU, 0); }               \
        else { if (pair) R3D_AM(THR, R3D_ACT_SIGMOID, 1);                                        \
               else      R3D_AM(THR, R3D_ACT_SIGMOID, 0); }                                     \
    } while (0)
    if (thr == (int)kNarrow) R3D_AM_A(kNarrow); else R3D_AM_A(kThreads);
#undef R3D_AM_A
#undef R3D_AM
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* y = silu(gate_up[.., i]) * gate_up[.., n + i]. The gate and the up share one plane, so one pitch
 * carries both halves: `gu_ld` is the FUSED row's pitch (>= 2n), `y_ld` the output's. */
extern "C" int r3d_silu_mul_bf16(long gate_up, long y, int M, int n, int gu_ld, int y_ld,
                                 long stream) {
    if (!gate_up || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!gu_ld) gu_ld = 2 * n;
    if (!y_ld) y_ld = n;
    if (gu_ld < 2 * n || y_ld < n) return -3;
    const int thr = row_threads(n);
    const int pair = row_pair(y, n, gu_ld, y_ld);   /* n even puts the up half on a 2-byte boundary */
#define R3D_SM(THR, PAIR)                                                                         \
    R3D_KLAUNCH(R3D_KNAME(silu_mul_kernel<THR, PAIR>), dim3((uint32_t)M), dim3(THR),             \
                (hipStream_t)(uintptr_t)stream, (const uint16_t*)(uintptr_t)gate_up,             \
                (uint16_t*)(uintptr_t)y, (uint32_t)n, (uint32_t)gu_ld, (uint32_t)y_ld)
    if (pair) {
        if (thr == (int)kNarrow) R3D_SM(kNarrow, 1); else R3D_SM(kThreads, 1);
    } else {
        if (thr == (int)kNarrow) R3D_SM(kNarrow, 0); else R3D_SM(kThreads, 0);
    }
#undef R3D_SM
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_softmax_f32(long x, long y, int M, int n, int x_ld, int y_ld, long stream) {
    if (!x || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    if (x_ld < n || y_ld < n) return -3;
    const int thr = row_threads(n);
    R3D_ROW2(softmax_kernel, thr, M, stream, (const float*)(uintptr_t)x, (float*)(uintptr_t)y,
             (uint32_t)n, (uint32_t)x_ld, (uint32_t)y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* scale's two legs: the scalar plane's dtype is `s_dt` (bf16 or f32), the row's dtype is the
 * launcher's. A null `s` is -1: the shim refuses a scale with no plane, so a call that reaches
 * here without one is a caller bug, not a scale of zero. */
static int scale_launch(long x, long s, long y, int M, int n, int s_dt, int x_ld, int y_ld, int f32,
                        long stream) {
    if (!x || !s || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (s_dt != R3D_DT_BF16 && s_dt != R3D_DT_F32) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    if (x_ld < n || y_ld < n) return -3;
    const int thr = row_threads(n);
    const int pair = f32 ? 0 : row_pair(y, n, x_ld, y_ld);
#define R3D_SC(THR, PAIR, F32)                                                                  \
    R3D_KLAUNCH(R3D_KNAME(scale_kernel<THR, PAIR, F32>), dim3((uint32_t)M), dim3(THR),           \
                (hipStream_t)(uintptr_t)stream, (const uint16_t*)(uintptr_t)x,                   \
                (const void*)(uintptr_t)s, (uint16_t*)(uintptr_t)y, (uint32_t)n, s_dt,           \
                (uint32_t)x_ld, (uint32_t)y_ld)
#define R3D_SC_P(THR, F32)                                                                      \
    do {                                                                                        \
        if (pair) R3D_SC(THR, 1, F32); else R3D_SC(THR, 0, F32);                                \
    } while (0)
    if (f32) { if (thr == (int)kNarrow) R3D_SC_P(kNarrow, 1); else R3D_SC_P(kThreads, 1); }
    else     { if (thr == (int)kNarrow) R3D_SC_P(kNarrow, 0); else R3D_SC_P(kThreads, 0); }
#undef R3D_SC_P
#undef R3D_SC
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_scale_bf16(long x, long s, long y, int M, int n, int s_dt, int x_ld, int y_ld,
                              long stream) {
    return scale_launch(x, s, y, M, n, s_dt, x_ld, y_ld, 0, stream);
}

extern "C" int r3d_scale_f32(long x, long s, long y, int M, int n, int s_dt, int x_ld, int y_ld,
                             long stream) {
    return scale_launch(x, s, y, M, n, s_dt, x_ld, y_ld, 1, stream);
}

extern "C" int r3d_fill(long y, int M, int n, int y_ld, int dt, float v, long stream) {
    if (!y) return -1;
    if (M <= 0 || n <= 0) return -2;
    if (r3d_dt_size(dt) == 0) return -2;
    if (!y_ld) y_ld = n;
    if (y_ld < n) return -3;
    const int thr = row_threads(n);
    R3D_ROW2(fill_kernel, thr, M, stream, (void*)(uintptr_t)y, (uint32_t)n, (uint32_t)y_ld, dt, v);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_rope_bf16(long qkv, long pos, int M, int head_dim, int n_head, int n_head_kv,
                             float theta, float scale, int mode, int rot, int qkv_ld,
                             long stream) {
    if (!qkv || !pos) return -1;
    if (M <= 0 || head_dim <= 0 || n_head <= 0 || n_head_kv < 0) return -2;
    if (rot <= 0) rot = head_dim;
    if (rot > head_dim || (rot & 1)) return -3;
    if (mode != R3D_ROPE_NEOX && mode != R3D_ROPE_GPTJ) return -3;
    if (head_dim & 1) return -3;                       // the two halves of a head, evenly split
    if (!(theta > 0.0f) || !std::isfinite(theta)) return -2;
    if (!qkv_ld) qkv_ld = (n_head + 2 * n_head_kv) * head_dim;
    if (qkv_ld < (n_head + 2 * n_head_kv) * head_dim) return -3;
    /* rope is a per-head-of-64-pairs op: one wave covers a token's rope heads when the head is
     * narrow, and 256 threads take the wide ones. */
    const int gptj = mode == R3D_ROPE_GPTJ;
    const int thr = (n_head + n_head_kv) * (rot / 2) <= 256u ? (int)kNarrow : (int)kThreads;
    R3D_ROPE2(thr, M, stream, (uint16_t*)(uintptr_t)qkv, (const int32_t*)(uintptr_t)pos,
              (uint32_t)head_dim, (uint32_t)(n_head + n_head_kv), (uint32_t)rot, theta, scale,
              (uint32_t)qkv_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_kv_store(long k, long v, long sm, long kv, int M, int head_dim, int n_head_kv,
                            int block_size, int kv_dt, float k_scale, float v_scale, int k_ld,
                            int v_ld, long long slot_max, long sb, long sh, long so, long sd,
                            long stream) {
    if (!k || !v || !sm || !kv) return -1;
    if (M <= 0 || head_dim <= 0 || n_head_kv <= 0 || block_size <= 0) return -2;
    if (kv_dt != R3D_DT_BF16 && kv_dt != R3D_DT_F16 && kv_dt != R3D_DT_U8) return -2;
    const int fp8 = kv_dt == R3D_DT_U8;
    if (fp8 && !(k_scale > 0.0f && v_scale > 0.0f)) return -2;
    const float k_inv = fp8 ? 1.0f / k_scale : 1.0f;
    const float v_inv = fp8 ? 1.0f / v_scale : 1.0f;
    if (!k_ld) k_ld = n_head_kv * head_dim;
    if (!v_ld) v_ld = n_head_kv * head_dim;
    if (k_ld < n_head_kv * head_dim || v_ld < n_head_kv * head_dim) return -3;
    if (sb < 0 || sh < 0 || so < 0 || sd <= 0) return -3;
    /* The cache's span, as r3d_shape_kv_store sizes it: M tokens need ceil(M/block_size) blocks,
     * and the shape adds two so a step that resolves its slots before the next block is written
     * has somewhere to put them. Keep the two formulas in step. */
    if (slot_max <= 0)
        slot_max = (long long)((M + block_size - 1) / block_size + 2) * block_size;
    const int thr = row_threads(n_head_kv * head_dim);
    R3D_ROW2(kv_store_kernel, thr, M, stream, (const uint16_t*)(uintptr_t)k,
             (const uint16_t*)(uintptr_t)v, (const int32_t*)(uintptr_t)sm, (void*)(uintptr_t)kv,
             (uint32_t)head_dim, (uint32_t)n_head_kv, (uint32_t)block_size, kv_dt, k_inv, v_inv,
             (uint32_t)k_ld, (uint32_t)v_ld, (int64_t)sb, (int64_t)sh, (int64_t)so, (int64_t)sd,
             (int64_t)slot_max);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_embed_lookup(long ids, long wte, long x, int M, int n_embd, int n_vocab,
                                int vocab_offset, float wscale, float mult, int x_ld, int w_ld,
                                int dt, long stream) {
    if (!ids || !wte || !x) return -1;
    if (M <= 0 || n_embd <= 0 || n_vocab <= 0) return -2;
    if (dt != R3D_DT_BF16 && dt != R3D_DT_F16 && dt != R3D_DT_F32 && dt != R3D_DT_U8) return -2;
    if (!x_ld) x_ld = n_embd;
    if (!w_ld) w_ld = n_embd;
    if (x_ld < n_embd || w_ld < n_embd) return -3;
    const int thr = row_threads(n_embd);
    const int pair = (dt == R3D_DT_BF16) ? row_pair(x, n_embd, x_ld, x_ld) : 0;
    R3D_ROW2(embed_lookup_kernel, thr, M, stream, (const int32_t*)(uintptr_t)ids,
             (const void*)(uintptr_t)wte, (void*)(uintptr_t)x, (uint32_t)n_embd, (int32_t)n_vocab,
             (int32_t)vocab_offset, wscale, mult, (uint32_t)x_ld, (int64_t)w_ld, dt, pair);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The row moves: the pitch is in elements and the element size in bytes, both off the dtype, so
 * the kernel's row base is one multiply. */
extern "C" int r3d_gather_rows(long x, long idx, long y, int M, int n, int x_ld, int y_ld, int dt,
                               long long src_rows, long stream) {
    if (!x || !idx || !y) return -1;
    if (M <= 0 || n <= 0) return -2;
    const int es = r3d_dt_size(dt);
    if (!es) return -2;
    if (!x_ld) x_ld = n;
    if (!y_ld) y_ld = n;
    if (x_ld < n || y_ld < n) return -3;
    R3D_ROW2(gather_rows_kernel, row_threads(n), M, stream, (const void*)(uintptr_t)x,
             (const int32_t*)(uintptr_t)idx, (void*)(uintptr_t)y, (uint32_t)n, (uint32_t)x_ld,
             (uint32_t)y_ld, dt, es, (int64_t)src_rows);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_scatter_rows(long v, long idx, long x, int M, int n, int v_ld, int x_ld, int dt,
                                long long dst_rows, long stream) {
    if (!v || !idx || !x) return -1;
    if (M <= 0 || n <= 0) return -2;
    const int es = r3d_dt_size(dt);
    if (!es) return -2;
    if (!v_ld) v_ld = n;
    if (!x_ld) x_ld = n;
    if (v_ld < n || x_ld < n) return -3;
    R3D_ROW2(scatter_rows_kernel, row_threads(n), M, stream, (const void*)(uintptr_t)v,
             (const int32_t*)(uintptr_t)idx, (void*)(uintptr_t)x, (uint32_t)n, (uint32_t)v_ld,
             (uint32_t)x_ld, dt, es, (int64_t)dst_rows);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ================================================================== the plugin ABI entry points
 *
 * Unpacking RadArgs and deriving the pitches belongs to the kernel library (r3d_args.h). The
 * kernels take raw pointers and one pitch per operand, so anything unpitched must be tight
 * (spec §2.1). Before each launch a shim checks:
 *
 *   * the dtype, off the operand: a row's `dtype` constraint is a necessary condition on the
 *     geometry, not a promise about the tensors, and an f32 operand read as bf16 is a value the
  *     geometry checks will not catch;
 *   * the row pitch, r3d_rows_pitch(): elements, or -1 (RAD_E_STRIDE) for a strided inner axis;
 *   * the row count, off the operand, never the `M` parameter: `M` is ranged, so the args carry
 *     the band's upper bound;
 *   * the parameters, with libref's defaults, and the forms the kernels cannot match -- refused
 *     with RAD_E_UNSUPPORTED so the row's constraints can state the band and a model that needs
 *     the refused form is not routed here at all.
 *
 * Every refusal goes through R3D_NO(), so a run with RAD_R3D_TRACE set says which line said no.
 */

static inline int dt_bf16(const RadTensor* t) { return r3d_is_bf16(t) ? RAD_OK : RAD_E_DTYPE; }
static inline int dt_f32(const RadTensor* t) { return r3d_is_f32(t) ? RAD_OK : RAD_E_DTYPE; }
static inline int dt_i32(const RadTensor* t) { return r3d_is_i32(t) ? RAD_OK : RAD_E_DTYPE; }

/* A bf16-or-f32 pair, so the two dtype legs of add/mul read as one test. */
static inline int dt_same(const RadTensor* a, const RadTensor* b) {
    return a && b && a->dtype == b->dtype ? RAD_OK : RAD_E_DTYPE;
}

#define R3D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R3D_NO(_s); } while (0)

/* A launcher's return: 0 is RAD_OK, anything else is a device failure -- the arguments were the
 * shim's to check, and it checked them. */
static inline int r3d_launch_ok(int rc) { return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE); }

/* ------------------------------------------------------------------ rmsnorm */

enum { RMS_X = 0, RMS_W, RMS_Y };

extern "C" int r3d_rad_rmsnorm(const RadArgs* a, RadStream s) {
    const RadTensor* x = r3d_opt(a, RMS_X);
    const RadTensor* w = r3d_opt(a, RMS_W);
    const RadTensor* y = r3d_opt(a, RMS_Y);
    if (!x || !y) return R3D_NO(RAD_E_INVAL);
    /* The row's dtype band is bf16 or f32, and the two legs are the same two passes: an f32
     * checkpoint's residual stream normalizes without a store that narrows. */
    const bool f32 = r3d_is_f32(x) && r3d_is_f32(y);
    if (!f32) {
        R3D_NEED(dt_bf16(x));
        R3D_NEED(dt_bf16(y));
    }

    const float wadd = (float)r3d_getf(a, "wadd", 0.0);
    const float mult = (float)r3d_getf(a, "mult", 1.0);
    const float eps = (float)r3d_getf(a, "eps", 1e-6);
    /* `wadd` without a gain is refused, as ref does: 1 + wadd of a gain that does not exist is a
     * constant scale wearing a norm's name. */
    if (!w && wadd != 0.0f) return R3D_NO(RAD_E_INVAL);
    /* The gain's dtype comes off the gain: bf16, f32, or none. */
    int wdt = R3D_DT_N;
    if (w) {
        if (r3d_is_bf16(w)) wdt = R3D_DT_BF16;
        else if (r3d_is_f32(w)) wdt = R3D_DT_F32;
        else return R3D_NO(RAD_E_DTYPE);
        if (!r3d_contig(w)) return R3D_NO(RAD_E_STRIDE);
    }

    const long long n = r3d_geti_or(a, "n", w ? r3d_dim(w, (int)w->rank - 1)
                                              : r3d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r3d_rows_pitch(x, n, &M);
    const long long y_ld = r3d_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!row_count_ok(M) || My < M) return R3D_NO(RAD_E_SHAPE);
    if (w && r3d_numel(w) < n) return R3D_NO(RAD_E_SHAPE);

    const int rc = f32
        ? r3d_rmsnorm_f32(r3d_addr(x), r3d_addr(w), r3d_addr(y), (int)M, (int)n, eps, wadd, mult,
                          (int)x_ld, (int)y_ld, wdt, (long)(uintptr_t)s)
        : r3d_rmsnorm_bf16(r3d_addr(x), r3d_addr(w), r3d_addr(y), (int)M, (int)n, eps, wadd, mult,
                           (int)x_ld, (int)y_ld, wdt, (long)(uintptr_t)s);
    return r3d_launch_ok(rc);
}

/* ------------------------------------------------------------------ rmsnorm_quant_fp8
 * The residual add, the norm and the E4M3 quantiser in one launch: {x, residual?, w, q, scale,
 * out_bf16?} with {M, n, eps, group, dtype, wadd?}, libref's. The row's reason to exist is the
 * traffic: the normalised row is written once as codes and one bf16 copy, not written as bf16 and
 * read back by a separate quantiser.
 */
extern "C" int r3d_rad_rmsnorm_quant_fp8(const RadArgs* a, RadStream s) {
    const RadTensor* x = r3d_opt(a, 0);
    const RadTensor* res = r3d_opt(a, 1);
    const RadTensor* w = r3d_opt(a, 2);
    const RadTensor* q = r3d_opt(a, 3);
    const RadTensor* sc = r3d_opt(a, 4);
    const RadTensor* ob = r3d_opt(a, 5);
    if (!x || !w || !q || !sc) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(x));
    /* The gain comes in the dtype the checkpoint stores it in -- bf16 or f32, read off the operand.
     * The band's `dtype` is the activation's: a checkpoint that stores its gains as f32 is read as
     * it is, and a shim that dtype-checks the weight against the activation band serves one
     * checkpoint and refuses the next. The kernel's WDT leg is the plain rmsnorm's. */
    int wdt = R3D_DT_N;
    if (r3d_is_bf16(w)) wdt = R3D_DT_BF16;
    else if (r3d_is_f32(w)) wdt = R3D_DT_F32;
    else return R3D_NO(RAD_E_DTYPE);
    /* `residual` is validated below, where the flag that makes it read is read. `out_bf16` is
     * checked because this kernel WRITES it, so it is required of the row that asks for it (the
     * reference writes it when the operand is there) and absent when the slot carries no bytes. */
    if (r3d_has(ob)) R3D_NEED(dt_bf16(ob));   /* no bytes = the plan asked for no bf16 copy */
    if (r3d_dt_code(q->dtype) != R3D_DT_U8) return R3D_NO(RAD_E_DTYPE);   // E4M3 codes
    R3D_NEED(dt_f32(sc));

    const long long n = r3d_geti_or(a, "n", w->rank ? r3d_dim(w, (int)w->rank - 1) : 0);
    const long long group = r3d_geti_or(a, "group", 0);
    if (n <= 0 || group <= 0) return R3D_NO(RAD_E_SHAPE);
    /* A wave covers a group, four columns a lane per 128 columns, and a partial group is the
     * standalone quantiser's zero-padded store. A checkpoint whose block divides its hidden width
     * is the band this row is for. */
    if (group != 128 && group != 256 && group != 512) return R3D_NO(RAD_E_UNSUPPORTED);
    if (n % group) return R3D_NO(RAD_E_UNSUPPORTED);
    const float eps = (float)r3d_getf(a, "eps", 1e-6);
    const float wadd = (float)r3d_getf(a, "wadd", 0.0);
    /* The reference reads `residual` under `wadd` and nowhere else, and so do we -- checked here,
     * after the flag, and REQUIRED by it. The engine hands the slot a tensor even when the band
     * says wadd=0, in any dtype; a row that validates an operand the op will not read is a row
     * that refuses a checkpoint. */
    if (wadd != 0.0f) {
        if (!r3d_has(res)) return R3D_NO(RAD_E_INVAL);
        R3D_NEED(dt_bf16(res));
    }

    const long long ngrp = n / group;
    long long M = 0, Mres = 0, Mob = 0, Mq = 0, Ms = 0;
    const long long x_ld = r3d_rows_pitch(x, n, &M);
    const long long q_ld = r3d_rows_pitch(q, n, &Mq);
    const long long s_ld = r3d_rows_pitch(sc, ngrp, &Ms);
    if (x_ld < 0 || q_ld < 0 || s_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!row_count_ok(M) || Mq < M || Ms < M) return R3D_NO(RAD_E_SHAPE);
    long long res_ld = 0, ob_ld = 0;
    if (res) { res_ld = r3d_rows_pitch(res, n, &Mres); if (res_ld < 0 || Mres < M)
                   return Mres < 0 ? R3D_NO(RAD_E_STRIDE) : R3D_NO(RAD_E_SHAPE); }
    if (ob)  { ob_ld = r3d_rows_pitch(ob, n, &Mob);   if (ob_ld < 0 || Mob < M)
                   return Mob < 0 ? R3D_NO(RAD_E_STRIDE) : R3D_NO(RAD_E_SHAPE); }
    if (r3d_numel(w) < n) return R3D_NO(RAD_E_SHAPE);

    return r3d_launch_ok(r3d_rmsnorm_quant_fp8(
        r3d_addr(x), r3d_addr(res), r3d_addr(w), r3d_addr(q), r3d_addr(sc), r3d_addr(ob), (int)M,
        (int)n, (int)group, eps, wadd, (int)x_ld, (int)res_ld, (int)ob_ld, (int)q_ld, (int)s_ld,
        wdt, (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ layernorm */

enum { LN_X = 0, LN_W, LN_B, LN_Y };

extern "C" int r3d_rad_layernorm(const RadArgs* a, RadStream s) {
    const RadTensor* x = r3d_opt(a, LN_X);
    const RadTensor* w = r3d_opt(a, LN_W);
    const RadTensor* b = r3d_opt(a, LN_B);
    const RadTensor* y = r3d_opt(a, LN_Y);
    if (!x || !w || !y) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(x));
    R3D_NEED(dt_bf16(w));
    R3D_NEED(dt_bf16(y));
    if (b) {
        /* libref reads a bias in any float dtype; the kernel reads bf16 (the launcher has one
         * dtype for gain and bias), so an f32 bias is refused, not read wrong. */
        if (!r3d_is_bf16(b)) return R3D_NO(RAD_E_DTYPE);
        R3D_NEED(dt_bf16(b));
    }
    if (!r3d_contig(w) || (b && !r3d_contig(b))) return R3D_NO(RAD_E_STRIDE);

    const long long n = r3d_geti_or(a, "n", r3d_dim(w, (int)w->rank - 1));
    if (n <= 0) return R3D_NO(RAD_E_SHAPE);
    const float eps = (float)r3d_getf(a, "eps", 1e-5);
    long long M = 0, My = 0;
    const long long x_ld = r3d_rows_pitch(x, n, &M);
    const long long y_ld = r3d_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!row_count_ok(M) || My < M) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(w) < n || (b && r3d_numel(b) < n)) return R3D_NO(RAD_E_SHAPE);

    return r3d_launch_ok(r3d_layernorm_bf16(r3d_addr(x), r3d_addr(w), r3d_addr(b), r3d_addr(y),
                                            (int)M, (int)n, eps, (int)x_ld, (int)y_ld,
                                            (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ add / mul
 * The broadcast rule is libref's, read off the operand: a `b` of exactly n elements against more
 * than one row rides every row, and nothing else broadcasts. `div` is `add`'s only (mul's schema
 * has no div key), and a zero, NaN or infinity is refused -- the reference refuses it too. */
static int binary_shim(const RadArgs* a, RadStream s, int mul) {
    const RadTensor* A = r3d_opt(a, 0);
    const RadTensor* B = r3d_opt(a, 1);
    const RadTensor* Y = r3d_opt(a, 2);
    if (!A || !B || !Y) return R3D_NO(RAD_E_INVAL);
    const bool f32 = r3d_is_f32(Y);
    if (f32) {
        R3D_NEED(dt_f32(A));
        R3D_NEED(dt_f32(B));
    } else {
        R3D_NEED(dt_bf16(A));
        R3D_NEED(dt_bf16(B));
        R3D_NEED(dt_bf16(Y));
    }
    R3D_NEED(dt_same(A, Y));
    if (!mul) R3D_NEED(dt_same(A, B));

    const long long n = r3d_geti_or(a, "n", r3d_numel(Y));
    if (n <= 0) return R3D_NO(RAD_E_SHAPE);
    const float div = mul ? 1.0f : (float)r3d_getf(a, "div", 1.0);
    if (!(div != 0.0f) || !std::isfinite(div)) return R3D_NO(RAD_E_INVAL);

    long long M = 0, Mb = 0;
    const long long y_ld = r3d_rows_pitch(Y, n, &M);
    const long long a_ld = r3d_rows_pitch(A, n, &Mb);
    if (y_ld < 0 || a_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!row_count_ok(M) || Mb < M) return R3D_NO(RAD_E_SHAPE);

    const bool bcast = r3d_numel(B) == n && M > 1;
    long long b_ld = 0;
    if (!bcast) {
        b_ld = r3d_rows_pitch(B, n, &Mb);
        if (b_ld < 0) return R3D_NO(RAD_E_STRIDE);
        if (Mb < M) return R3D_NO(RAD_E_SHAPE);
    }

    const int rc = mul
        ? (f32 ? r3d_mul_f32(r3d_addr(A), r3d_addr(B), r3d_addr(Y), (int)M, (int)n, (int)a_ld,
                             (int)b_ld, (int)y_ld, bcast ? 1 : 0, (long)(uintptr_t)s)
               : r3d_mul_bf16(r3d_addr(A), r3d_addr(B), r3d_addr(Y), (int)M, (int)n, (int)a_ld,
                              (int)b_ld, (int)y_ld, bcast ? 1 : 0, (long)(uintptr_t)s))
        : (f32 ? r3d_add_f32(r3d_addr(A), r3d_addr(B), r3d_addr(Y), (int)M, (int)n, (int)a_ld,
                             (int)b_ld, (int)y_ld, div, bcast ? 1 : 0, (long)(uintptr_t)s)
               : r3d_add_bf16(r3d_addr(A), r3d_addr(B), r3d_addr(Y), (int)M, (int)n, (int)a_ld,
                              (int)b_ld, (int)y_ld, div, bcast ? 1 : 0, (long)(uintptr_t)s));
    return r3d_launch_ok(rc);
}

extern "C" int r3d_rad_add(const RadArgs* a, RadStream s) { return binary_shim(a, s, 0); }
extern "C" int r3d_rad_mul(const RadArgs* a, RadStream s) { return binary_shim(a, s, 1); }

/* ------------------------------------------------------------------ cast
 * `from` and `to` are the selector's; the tensors' own dtypes are what the kernel moves. The
 * whole op is flat, so one pitch per operand, and the row count is numel(y)/n with n the last
 * dimension of y. */
extern "C" int r3d_rad_cast(const RadArgs* a, RadStream s) {
    const RadTensor* x = r3d_opt(a, 0);
    const RadTensor* y = r3d_opt(a, 1);
    if (!x || !y) return R3D_NO(RAD_E_INVAL);
    const int from_dt = r3d_dt_code(x->dtype), to_dt = r3d_dt_code(y->dtype);
    if (from_dt < 0 || to_dt < 0) return R3D_NO(RAD_E_DTYPE);
    /* f16 has no e5m2 twin in this library, and the f16 <-> integer pairs have no caller: the
     * codes the kernels move are bf16, f16, f32, i32, i64, E4M3 codes and i8. */
    static const int kOk[] = { R3D_DT_BF16, R3D_DT_F16, R3D_DT_F32, R3D_DT_I32, R3D_DT_I64,
                               R3D_DT_U8, R3D_DT_I8 };
    bool ok = false;
    for (unsigned i = 0; i < sizeof kOk / sizeof kOk[0]; ++i)
        ok = ok || (from_dt == kOk[i] && (to_dt == kOk[i] || to_dt == R3D_DT_F32 ||
                                          to_dt == R3D_DT_BF16 || to_dt == R3D_DT_I32 ||
                                          to_dt == R3D_DT_I64 || to_dt == R3D_DT_F16 ||
                                          to_dt == R3D_DT_U8 || to_dt == R3D_DT_I8));
    if (!ok) return R3D_NO(RAD_E_UNSUPPORTED);

    const long long n = r3d_geti_or(a, "n", r3d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r3d_rows_pitch(y, n, &M);
    const long long x_ld = r3d_rows_pitch(x, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!row_count_ok(M) || Mx < M) return R3D_NO(RAD_E_SHAPE);

    return r3d_launch_ok(r3d_cast(r3d_addr(x), r3d_addr(y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                                  from_dt, to_dt, (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ silu / gelu / sigmoid */

static int unary_shim(const RadArgs* a, RadStream s, int act) {
    const RadTensor* x = r3d_opt(a, 0);
    const RadTensor* y = r3d_opt(a, 1);
    if (!x || !y) return R3D_NO(RAD_E_INVAL);
    const bool f32 = r3d_is_f32(x) && r3d_is_f32(y);
    if (!f32) {
        R3D_NEED(dt_bf16(x));
        R3D_NEED(dt_bf16(y));
    }
    /* Only silu has an f32 leg in r3d.h (the vocoder's f32 branch is the one that asked for it);
     * an f32 gelu or sigmoid is refused, not served by the bf16 kernel. */
    if (f32 && act != R3D_ACT_SILU) return R3D_NO(RAD_E_UNSUPPORTED);

    const long long n = r3d_geti_or(a, "n", r3d_numel(y));
    if (n <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r3d_rows_pitch(y, n, &M);
    const long long x_ld = r3d_rows_pitch(x, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!row_count_ok(M) || Mx < M) return R3D_NO(RAD_E_SHAPE);

    const int rc = f32
        ? r3d_silu_f32(r3d_addr(x), r3d_addr(y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                       (long)(uintptr_t)s)
        : (act == R3D_ACT_SILU
               ? r3d_silu_bf16(r3d_addr(x), r3d_addr(y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                               (long)(uintptr_t)s)
               : act == R3D_ACT_GELU
               ? r3d_gelu_bf16(r3d_addr(x), r3d_addr(y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                               (long)(uintptr_t)s)
               : r3d_sigmoid_bf16(r3d_addr(x), r3d_addr(y), (int)M, (int)n, (int)x_ld, (int)y_ld,
                                  (long)(uintptr_t)s));
    return r3d_launch_ok(rc);
}

extern "C" int r3d_rad_silu(const RadArgs* a, RadStream s) { return unary_shim(a, s, R3D_ACT_SILU); }
extern "C" int r3d_rad_gelu(const RadArgs* a, RadStream s) { return unary_shim(a, s, R3D_ACT_GELU); }
extern "C" int r3d_rad_sigmoid(const RadArgs* a, RadStream s) {
    return unary_shim(a, s, R3D_ACT_SIGMOID);
}

/* ------------------------------------------------------------------ act_mul
 * libref's kActs order is silu, gelu, gelu_tanh, sigmoid, quick_gelu; libr3d's rows serve silu,
 * gelu and sigmoid, and the other two are refused with RAD_E_UNSUPPORTED so the row's constraints
 * state which gated MLPs reach an RDNA3 card. */
extern "C" int r3d_rad_act_mul(const RadArgs* a, RadStream s) {
    const RadTensor* A = r3d_opt(a, 0);
    const RadTensor* B = r3d_opt(a, 1);
    const RadTensor* Y = r3d_opt(a, 2);
    if (!A || !B || !Y) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(A));
    R3D_NEED(dt_bf16(B));
    R3D_NEED(dt_bf16(Y));

    int act = -1;
    if (const char* n = r3d_gets(a, "act")) {
        if (!std::strcmp(n, "silu") || !std::strcmp(n, "swish")) act = R3D_ACT_SILU;
        else if (!std::strcmp(n, "gelu")) act = R3D_ACT_GELU;
        else if (!std::strcmp(n, "sigmoid")) act = R3D_ACT_SIGMOID;
        else if (!std::strcmp(n, "gelu_tanh") || !std::strcmp(n, "quick_gelu"))
            return R3D_NO(RAD_E_UNSUPPORTED);
        else return R3D_NO(RAD_E_INVAL);
    } else {
        act = R3D_ACT_SILU;   // libref's p_enum default: the gated MLP's activation
    }

    const long long n = r3d_geti_or(a, "n", r3d_numel(Y));
    if (n <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0, Mg = 0, Mu = 0;
    const long long y_ld = r3d_rows_pitch(Y, n, &M);
    const long long g_ld = r3d_rows_pitch(A, n, &Mg);
    const long long u_ld = r3d_rows_pitch(B, n, &Mu);
    if (y_ld < 0 || g_ld < 0 || u_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!row_count_ok(M) || Mg < M || Mu < M) return R3D_NO(RAD_E_SHAPE);

    return r3d_launch_ok(r3d_act_mul_bf16(r3d_addr(A), r3d_addr(B), r3d_addr(Y), (int)M, (int)n,
                                          (int)g_ld, (int)u_ld, (int)y_ld, act,
                                          (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ silu_mul
 * y[i] = silu(gate_up[i]) * gate_up[n + i]: libref's {gate_up, y} operands and its {M, n, dtype}
 * schema. A distinct op from act_mul -- the llama MLP emits it, and act_mul takes two operands --
 * so a model asking for silu_mul is not served by the act_mul_silu row.
 */
extern "C" int r3d_rad_silu_mul(const RadArgs* a, RadStream s) {
    const RadTensor* gu = r3d_opt(a, 0);
    const RadTensor* y = r3d_opt(a, 1);
    if (!gu || !y) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(gu));
    R3D_NEED(dt_bf16(y));

    const long long n = r3d_geti_or(a, "n", r3d_numel(y));
    if (n <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0, Mgu = 0;
    const long long y_ld = r3d_rows_pitch(y, n, &M);
    const long long gu_ld = r3d_rows_pitch(gu, 2 * n, &Mgu);   /* the FUSED row is 2n wide */
    if (y_ld < 0 || gu_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!row_count_ok(M) || Mgu < M) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(gu) < M * 2 * n) return R3D_NO(RAD_E_SHAPE);   // ref's own bound on the plane

    return r3d_launch_ok(r3d_silu_mul_bf16(r3d_addr(gu), r3d_addr(y), (int)M, (int)n, (int)gu_ld,
                                           (int)y_ld, (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ softmax
 * f32, as ref spells it: the reference casts to f32, computes, and casts back, and the intermediate
 * `e` is read back from an f32 plane. A bf16 y would round that intermediate and the kernel cannot
 * match ref with ref's own arithmetic, so it is refused rather than served at a tolerance. */
extern "C" int r3d_rad_softmax(const RadArgs* a, RadStream s) {
    const RadTensor* x = r3d_opt(a, 0);
    const RadTensor* y = r3d_opt(a, 1);
    if (!x || !y) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_f32(x));
    R3D_NEED(dt_f32(y));

    const long long n = r3d_geti_or(a, "n", r3d_numel(y));
    if (n <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r3d_rows_pitch(y, n, &M);
    const long long x_ld = r3d_rows_pitch(x, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!row_count_ok(M) || Mx < M) return R3D_NO(RAD_E_SHAPE);

    return r3d_launch_ok(r3d_softmax_f32(r3d_addr(x), r3d_addr(y), (int)M, (int)n, (int)x_ld,
                                         (int)y_ld, (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ scale
 * y = x * s. `s` is the op's [1] weight plane: the shim checks its dtype and that it has an
 * element, and the kernel reads the value. The launcher cannot take the number by value, because
 * reading a device plane on the host is a copy and a sync, and a row is neither. */
extern "C" int r3d_rad_scale(const RadArgs* a, RadStream s) {
    const RadTensor* x = r3d_opt(a, 0);
    const RadTensor* sc = r3d_opt(a, 1);
    const RadTensor* y = r3d_opt(a, 2);
    if (!x || !sc || !y) return R3D_NO(RAD_E_INVAL);
    const bool f32 = r3d_is_f32(x) && r3d_is_f32(y);
    if (!f32) {
        R3D_NEED(dt_bf16(x));
        R3D_NEED(dt_bf16(y));
    }
    const int s_dt = r3d_is_f32(sc) ? R3D_DT_F32 : r3d_is_bf16(sc) ? R3D_DT_BF16 : -1;
    if (s_dt < 0) return R3D_NO(RAD_E_DTYPE);
    if (r3d_numel(sc) < 1) return R3D_NO(RAD_E_SHAPE);

    const long long n = r3d_geti_or(a, "n", r3d_numel(y));
    if (n <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r3d_rows_pitch(y, n, &M);
    const long long x_ld = r3d_rows_pitch(x, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!row_count_ok(M) || Mx < M) return R3D_NO(RAD_E_SHAPE);

    const int rc = f32
        ? r3d_scale_f32(r3d_addr(x), r3d_addr(sc), r3d_addr(y), (int)M, (int)n, s_dt, (int)x_ld,
                        (int)y_ld, (long)(uintptr_t)s)
        : r3d_scale_bf16(r3d_addr(x), r3d_addr(sc), r3d_addr(y), (int)M, (int)n, s_dt, (int)x_ld,
                         (int)y_ld, (long)(uintptr_t)s);
    return r3d_launch_ok(rc);
}

/* ------------------------------------------------------------------ fill */

extern "C" int r3d_rad_fill(const RadArgs* a, RadStream s) {
    const RadTensor* y = r3d_opt(a, 0);
    if (!y) return R3D_NO(RAD_E_INVAL);
    const int dt = r3d_dt_code(y->dtype);
    if (dt < 0) return R3D_NO(RAD_E_DTYPE);
    if (dt == R3D_DT_E5M2) return R3D_NO(RAD_E_UNSUPPORTED);   // no e5m2 codec on this path

    const long long n = r3d_geti_or(a, "n", r3d_numel(y));
    if (n <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long y_ld = r3d_rows_pitch(y, n, &M);
    if (y_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!row_count_ok(M)) return R3D_NO(RAD_E_SHAPE);

    const float v = (float)r3d_getf(a, "value", 0.0);
    return r3d_launch_ok(r3d_fill(r3d_addr(y), (int)M, (int)n, (int)y_ld, dt, v,
                                  (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ rope
 * In place, so operand 0 is the row's INOUT: the shim checks it is writable (the role says it is;
 * a null data is the row's bug, and is RAD_E_INVAL) and that head_dim and rot are even.
 *
 * What the shim refuses, and why: the multi-component modes (mrope, imrope, axial) need a
 * [C, M] position plane and per-section ladders; the proportional ladder (`ladder=head`) pairs
 * across the whole head; YaRN and Llama 3 rescale the ladder. Each changes the angles, not the
 * arithmetic, and the kernel has the plain rotary -- so each is RAD_E_UNSUPPORTED, the row's
 * constraints say so, and a model that needs one is not routed to this card's row.
 */
extern "C" int r3d_rad_rope(const RadArgs* a, RadStream s) {
    const RadTensor* qkv = r3d_opt(a, 0);
    const RadTensor* pos = r3d_opt(a, 1);
    if (!qkv || !pos) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(qkv));
    R3D_NEED(dt_i32(pos));
    if (!r3d_contig(pos)) return R3D_NO(RAD_E_STRIDE);
    if (pos->rank >= 2) return R3D_NO(RAD_E_UNSUPPORTED);   // multi-component positions

    const long long head_dim = r3d_geti_or(a, "head_dim", 0);
    const long long n_head = r3d_geti_or(a, "n_head", 0);
    const long long n_head_kv = r3d_geti_or(a, "n_head_kv", 0);
    if (head_dim <= 0 || n_head <= 0 || n_head_kv < 0) return R3D_NO(RAD_E_SHAPE);
    long long rot = r3d_geti_or(a, "rotary_dim", head_dim);
    if (rot <= 0) rot = head_dim;
    if (rot > head_dim) return R3D_NO(RAD_E_SHAPE);
    if ((head_dim & 1) || (rot & 1)) return R3D_NO(RAD_E_SHAPE);   /* odd halves: a shape the kernel does not serve */

    static const char* kModes[] = { "neox", "gptj", "mrope", "imrope", "axial" };
    int mode = -1;
    if (const char* nm = r3d_gets(a, "mode")) {
        for (int i = 0; i < 5; ++i)
            if (!std::strcmp(nm, kModes[i])) { mode = i; break; }
    }
    if (mode < 0) return R3D_NO(RAD_E_UNSUPPORTED);   // absent, or one of the three multi modes
    if (mode > R3D_ROPE_GPTJ) return R3D_NO(RAD_E_UNSUPPORTED);
    if (r3d_gets(a, "sections")) return R3D_NO(RAD_E_UNSUPPORTED);
    if (r3d_getf(a, "yarn_factor", 0.0) != 0.0) return R3D_NO(RAD_E_UNSUPPORTED);
    if (r3d_getf(a, "llama3_factor", 0.0) != 0.0) return R3D_NO(RAD_E_UNSUPPORTED);
    if (const char* lad = r3d_gets(a, "ladder")) {
        /* Absent and "rotary" are the ladder the kernel has: the base frequency divides by rot. */
        if (std::strcmp(lad, "rotary")) return R3D_NO(RAD_E_UNSUPPORTED);
    }

    const float theta = (float)r3d_getf(a, "theta", 10000.0);
    const float scale = (float)r3d_getf(a, "scale", 1.0);

    const long long width = (n_head + 2 * n_head_kv) * head_dim;
    long long Mq = 0;
    const long long qkv_ld = r3d_rows_pitch(qkv, width, &Mq);
    if (qkv_ld < 0) return R3D_NO(RAD_E_STRIDE);
    const long long Mp = r3d_numel(pos);
    if (Mp <= 0 || Mq < Mp) return R3D_NO(RAD_E_SHAPE);
    const long long M = Mp;   // the positions drive the launch; a row past them is left alone

    return r3d_launch_ok(r3d_rope_bf16(r3d_addr(qkv), r3d_addr(pos), (int)M, (int)head_dim,
                                        (int)n_head, (int)n_head_kv, theta, scale, mode, (int)rot,
                                        (int)qkv_ld, (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ kv_store
 * The cache is INOUT: only the mapped slots are touched and every other slot must survive. Its
 * strides are read off the operand (the reference's kv_view): a rank-4 cache carries them; the
 * flat [blocks, kv_heads, block_size, 2*head_dim] form derives them.
 *
 * ref also refuses a slot naming a block past the cache (RAD_E_SHAPE), which the shim cannot do
 * before the launch: the slots are in device memory. The bound travels as `slot_max`, the slot span
 * of the cache the shapes hook sizes, and the kernel drops a slot at or past it -- a slot lost,
 * not a step lost.
 */
extern "C" int r3d_rad_kv_store(const RadArgs* a, RadStream s) {
    const RadTensor* k = r3d_opt(a, 0);
    const RadTensor* v = r3d_opt(a, 1);
    const RadTensor* sm = r3d_opt(a, 2);
    const RadTensor* kv = r3d_opt(a, 3);
    if (!k || !v || !sm || !kv) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(k));
    R3D_NEED(dt_bf16(v));
    R3D_NEED(dt_i32(sm));
    const int kv_dt = r3d_dt_code(kv->dtype);
    if (kv_dt != R3D_DT_BF16 && kv_dt != R3D_DT_F16 && kv_dt != R3D_DT_U8)
        return R3D_NO(RAD_E_DTYPE);

    const long long head_dim = r3d_geti_or(a, "head_dim", 0);
    const long long n_head_kv = r3d_geti_or(a, "n_head_kv", 0);
    const long long block_size = r3d_geti_or(a, "block_size", kv->rank == 4 ? kv->shape[2] : 0);
    if (head_dim <= 0 || n_head_kv <= 0 || block_size <= 0) return R3D_NO(RAD_E_SHAPE);
    const float k_scale = (float)r3d_getf(a, "k_scale", 1.0);
    const float v_scale = (float)r3d_getf(a, "v_scale", 1.0);
    if (kv_dt == R3D_DT_U8 && !(k_scale > 0.0f && v_scale > 0.0f)) return R3D_NO(RAD_E_INVAL);

    long long sb = 0, sh = 0, so = 0, sd = 1;
    if (kv->rank == 4) {
        sb = r3d_stride(kv, 0);
        sh = r3d_stride(kv, 1);
        so = r3d_stride(kv, 2);
        sd = r3d_stride(kv, 3);
        if (sd != 1) return R3D_NO(RAD_E_UNSUPPORTED);   // a strided cache element
        if (r3d_dim(kv, 1) != n_head_kv || r3d_dim(kv, 2) != block_size) return R3D_NO(RAD_E_SHAPE);
        if (r3d_dim(kv, 3) < 2 * head_dim) return R3D_NO(RAD_E_SHAPE);
    } else {
        so = 2 * head_dim;
        sh = block_size * so;
        sb = n_head_kv * sh;
    }

    const long long width = n_head_kv * head_dim;
    long long Mk = 0, Mv = 0;
    const long long k_ld = r3d_rows_pitch(k, width, &Mk);
    const long long v_ld = r3d_rows_pitch(v, width, &Mv);
    if (k_ld < 0 || v_ld < 0) return R3D_NO(RAD_E_STRIDE);
    const long long T = r3d_numel(sm);
    if (T <= 0 || Mk < T || Mv < T) return R3D_NO(RAD_E_SHAPE);
    if (!row_count_ok(T)) return R3D_NO(RAD_E_SHAPE);

    /* The cache's span, in slots: the pool's own plane, NOT this step's. A step that adopts a
     * prefix writes its tail into the blocks the table names, and those sit at the allocator's
     * GLOBAL indices -- a 4-token tail placed at block 3 is slots 48..51, past any bound built
     * from T. A bound built from the step's own span is no bound for the tail's slots, so the
     * pool's plane is the true bound: a slot at or past it names memory the cache does not have,
     * and the kernel drops it there. */
    long long slot_max;
    if (kv->rank == 4) {
        slot_max = (long long)r3d_dim(kv, 0) * block_size;
    } else {
        long long el = 1;
        for (int i = 0; i < kv->rank; ++i) el *= r3d_dim(kv, i);
        slot_max = el / (2 * head_dim);
    }

    if (std::getenv("RAD_R3D_TRACE"))   /* the write side, spoken: where this step's KV lands. A
                                     * tail that names slots of an adopted block writes over the
                                     * shared prefix, and this line is the one that says so. */
    {
        (void)hipDeviceSynchronize();
        int slots[8];
        if (hipMemcpy(slots, (const void*)(uintptr_t)r3d_addr(sm), 8 * sizeof(int),
                      hipMemcpyDeviceToHost) == hipSuccess) {
            std::fprintf(stderr, "[r3d] kvstore: T=%lld kvbase=%p slots=", (long long)T,
                         (const void*)(uintptr_t)r3d_addr(kv));
            for (int c = 0; c < 8 && (long long)c < T; ++c)
                std::fprintf(stderr, "%s%d", c ? " " : "", slots[c]);
            std::fprintf(stderr, "\n");
        }
    }

    return r3d_launch_ok(r3d_kv_store(r3d_addr(k), r3d_addr(v), r3d_addr(sm), r3d_addr(kv), (int)T,
                                      (int)head_dim, (int)n_head_kv, (int)block_size, kv_dt,
                                      k_scale, v_scale, (int)k_ld, (int)v_ld, slot_max, sb, sh, so,
                                      sd, (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ embed_lookup
 * tokens are i32 (the engine's token ids), the table's dtype is the launcher's `dt`, and x is
 * bf16 -- a table in any other float dtype, or an E4M3 table whose scale arrives as a tensor
 * (embed_lookup_q's operand), is refused: one scale per gathered row is a device read the step
 * path cannot do, which is why the op's schema carries `wscale` as a parameter. */
extern "C" int r3d_rad_embed_lookup(const RadArgs* a, RadStream s) {
    const RadTensor* tok = r3d_opt(a, 0);
    const RadTensor* wte = r3d_opt(a, 1);
    const RadTensor* x = r3d_opt(a, 2);
    if (!tok || !wte || !x) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_i32(tok));
    R3D_NEED(dt_bf16(x));
    if (!r3d_contig(tok) || !r3d_contig(wte)) return R3D_NO(RAD_E_STRIDE);
    const int dt = r3d_dt_code(wte->dtype);
    if (dt != R3D_DT_BF16 && dt != R3D_DT_F16 && dt != R3D_DT_F32 && dt != R3D_DT_U8)
        return R3D_NO(RAD_E_DTYPE);
    if (dt != R3D_DT_BF16 && r3d_getf(a, "wscale", 1.0) == 1.0 && dt == R3D_DT_U8)
        return R3D_NO(RAD_E_INVAL);   // an E4M3 table with no scale stated is a bug, not a copy

    const long long n_embd = r3d_geti_or(a, "n_embd", r3d_dim(wte, (int)wte->rank - 1));
    const long long n_vocab = r3d_geti_or(a, "n_vocab", r3d_dim(wte, (int)wte->rank - 2));
    const long long voff = r3d_geti_or(a, "vocab_offset", 0);
    if (n_embd <= 0 || n_vocab <= 0 || voff < 0) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(wte) < n_vocab * n_embd) return R3D_NO(RAD_E_SHAPE);

    long long Mx = 0;
    const long long x_ld = r3d_rows_pitch(x, n_embd, &Mx);
    if (x_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!row_count_ok(Mx) || r3d_numel(tok) < Mx) return R3D_NO(RAD_E_SHAPE);
    const long long M = Mx;

    const float mult = (float)r3d_getf(a, "mult", 1.0);
    const float wscale = (float)r3d_getf(a, "wscale", 1.0);
    const long long w_ld = wte->rank >= 2 ? r3d_stride(wte, (int)wte->rank - 2) : n_embd;

    return r3d_launch_ok(r3d_embed_lookup(r3d_addr(tok), r3d_addr(wte), r3d_addr(x), (int)M,
                                          (int)n_embd, (int)n_vocab, (int)voff, wscale, mult,
                                          (int)x_ld, (int)w_ld, dt, (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ gather_rows / scatter_rows */

extern "C" int r3d_rad_gather_rows(const RadArgs* a, RadStream s) {
    const RadTensor* x = r3d_opt(a, 0);
    const RadTensor* idx = r3d_opt(a, 1);
    const RadTensor* y = r3d_opt(a, 2);
    if (!x || !idx || !y) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_i32(idx));
    const int dt = r3d_dt_code(x->dtype);
    if (dt < 0) return R3D_NO(RAD_E_DTYPE);
    R3D_NEED(dt_same(x, y));
    if (dt == R3D_DT_E5M2) return R3D_NO(RAD_E_UNSUPPORTED);

    const long long n = r3d_geti_or(a, "n", r3d_dim(y, (int)y->rank - 1));
    if (n <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long y_ld = r3d_rows_pitch(y, n, &M);
    const long long x_ld = r3d_rows_pitch(x, n, &Mx);
    if (y_ld < 0 || x_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!row_count_ok(M) || Mx <= 0 || r3d_numel(idx) < M) return R3D_NO(RAD_E_SHAPE);

    /* The source plane's row count, which the shim sees in the descriptor and the kernel cannot:
     * an index at or past it is libref's refused step and takes the zero row here. */
    return r3d_launch_ok(r3d_gather_rows(r3d_addr(x), r3d_addr(idx), r3d_addr(y), (int)M, (int)n,
                                         (int)x_ld, (int)y_ld, dt, Mx, (long)(uintptr_t)s));
}

extern "C" int r3d_rad_scatter_rows(const RadArgs* a, RadStream s) {
    const RadTensor* v = r3d_opt(a, 0);
    const RadTensor* idx = r3d_opt(a, 1);
    const RadTensor* x = r3d_opt(a, 2);
    if (!v || !idx || !x) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_i32(idx));
    const int dt = r3d_dt_code(v->dtype);
    if (dt < 0) return R3D_NO(RAD_E_DTYPE);
    R3D_NEED(dt_same(v, x));
    if (dt == R3D_DT_E5M2) return R3D_NO(RAD_E_UNSUPPORTED);

    const long long n = r3d_geti_or(a, "n", r3d_dim(v, (int)v->rank - 1));
    if (n <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long v_ld = r3d_rows_pitch(v, n, &M);
    const long long x_ld = r3d_rows_pitch(x, n, &Mx);
    if (v_ld < 0 || x_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!row_count_ok(M) || Mx <= 0 || r3d_numel(idx) < M) return R3D_NO(RAD_E_SHAPE);

    /* The destination plane's row count, from the descriptor: an index at or past it writes past
     * the plane, which is the one thing a scatter must not do, so the kernel drops the row. */
    return r3d_launch_ok(r3d_scatter_rows(r3d_addr(v), r3d_addr(idx), r3d_addr(x), (int)M, (int)n,
                                          (int)v_ld, (int)x_ld, dt, Mx, (long)(uintptr_t)s));
}
