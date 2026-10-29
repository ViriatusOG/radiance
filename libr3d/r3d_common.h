/* r3d_common.h -- the RDNA3 seams a kernel needs: the wave's lane maths, the loads, the block
 * reduce, the occupancy and LDS conventions.
 *
 * What RDNA3 gives, measured on the ROCm this tree builds against (AMD clang 24; `__has_builtin`
 * off --offload-device-only, and llvm-mc -show-encoding per target). The codecs are in r3d_codecs.h;
 * this header is what is left.
 *
 *   v_dot2acc_f32_f16       yes, every gfx11 target, from __builtin_amdgcn_fdot2, and RDNA3 pairs
 *                           it as v_dual_dot2acc_f32_f16 where the scheduler can. The matrix work.
 *   v_dot2 with bf16        no, and the builtin's bf16 selector is accepted and ignored -- the f16
 *                           instruction is emitted -- so bf16 is widened, never asserted.
 *   WMMA                    the mnemonic is unreachable -- every v_wmma_* is rejected by the
 *                           assembler on gfx1100, gfx1102, gfx1151 and gfx1201 alike, and inline
 *                           asm dies in amdgcn-link -- but the builtin is not:
 *                           __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 compiles for the three gfx11
 *                           targets and emits a real tile instruction. Measured here against dot2 at
 *                           four occupancies, 1.99 against 1.97 TFLOP/s: RDNA3's AI accelerator is a
 *                           2x f16-rate feature and a paired dot2 is already at that rate, so the
 *                           tile buys a fragment layout and a block-alignment requirement and no
 *                           throughput. No fragment layouts here, so no repacked weights, so no
 *                           layout hook: a dot2 GEMM reads a weight in the order it is stored.
 *   fp8 convert             no (needs the fp8 feature); the builtins are declared and die at
 *                           instruction selection, which is why the codecs are software.
 *   v_cvt_pkrtz_f16_f32     yes. Used only where it is exact.
 *   v_cvt_pk_bf16_f32       no on gfx1100/1102/1151 in this build, so bf16 packs are software.
 *   DPP16, v_permlanex16    yes, RDNA1 onward. The lane maths below is libr4d's, unchanged: the
 *                           encodings are the same silicon, and they were verified on hardware.
 *   ballot_w32, readlane,
 *   readfirstlane, ds_bpermute,
 *   v_perm_b32, exp2f, rcpf,
 *   s_barrier, fence,
 *   sched_group_barrier     yes.
 *   transposed loads        no: neither global_load_tr_b128 (RDNA4) nor ds_read_b64_tr_b16 (dropped
 *                           by RDNA4, kept here as a fact about gfx11 -- this ROCm's assembler
 *                           rejects it too). A transpose is done in registers.
 *
 * Wave32 is gfx11's default and every kernel here is written for it; the guard below fires if a
 * build asks for wave64, because that computes garbage rather than running slower.
 *
 * The register file is the one number that is NOT the same as RDNA4's: a gfx11 SIMD has 1024 VGPRs
 * for a wave32, allocated in groups of 8, against gfx1201's 1536. R3D_WAVES is stated per kernel,
 * and a kernel's budget is computed against 1024.
 */
#ifndef R3D_COMMON_H
#define R3D_COMMON_H

#include "r3d_codecs.h"

#include <hip/hip_runtime.h>
#include <cstdint>

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN_WAVEFRONT_SIZE)
#if __AMDGCN_WAVEFRONT_SIZE != 32
#error "libr3d kernels are written for wave32, gfx11's default; a wave64 build computes garbage"
#endif
#endif

enum : int { kWave = 32 };

/* A 16-byte staging register as a native vector. HIP's uint4 is a class template wrapping a union,
 * and an array of them carried across a loop is kept in scratch by LLVM 22 and 24; an
 * ext_vector_type array is promoted to registers. Loaded and stored whole only. */
typedef unsigned int r3d_u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int r3d_u32x2 __attribute__((ext_vector_type(2)));

/* ================================================================== the occupancy a kernel is
 * built for
 *
 * Fixes the waves per SIMD the compiler schedules for. Left unstated, the scheduler picks the
 * target itself and the choice moves between compiler releases. gfx11's budget is 1024 VGPRs a
 * SIMD at wave32, in granules of 8: a kernel asking for 5 waves has 204 registers to spend. */
#define R3D_WAVES(n) __attribute__((amdgpu_waves_per_eu((n), (n))))

/* ================================================================== exp2
 *
 * The hardware transcendental. libr4d's rule applies: exp2f, powf and sincosf are the accurate
 * forms, not the __ approximations -- rope positions reach 262144, past __sincosf's argument
 * reduction, and libref uses libm. exp2f itself is the one place the hardware op is exactly what
 * the reference computes, because both sides round the same result once. */
__device__ __forceinline__ float ex2(float x) {
    return __builtin_amdgcn_exp2f(x);
}

__device__ __forceinline__ float rcp(float x) { return __builtin_amdgcn_rcpf(x); }

/* ================================================================== one element, one register
 *
 * A 16-bit element of global memory, zero-extended into a whole register (global_load_u16). Read as
 * `uint16_t` it lands in a register half (global_load_d16_b16) whose other half the waitcnt pass
 * then waits for; through a generic pointer it is also a flat load. The pointer must address global
 * memory, which every caller's operand is.
 *
 * libr4d notes that LLVM 24 gives gfx12 real 16-bit registers and the packability that follows;
 * gfx11 does not have them, so keeping bf16 in whole registers from load to store is the plain
 * reading as well as the portable one. */
typedef const uint16_t __attribute__((address_space(1))) *r3d_gu16;
__device__ __forceinline__ uint32_t ldg_u16(const uint16_t* p) { return *(r3d_gu16)p; }
__device__ __forceinline__ uint32_t ldg_u32(const uint32_t* p) { return *(const __attribute__((address_space(1))) uint32_t*)p; }
__device__ __forceinline__ float ldg_f32(const float* p) {
    return *(const __attribute__((address_space(1))) float*)p;
}

/* ================================================================== lane maths
 *
 * A wave is 32 lanes, two rows of sixteen. Reductions are XOR butterflies, in the order a
 * __shfl_xor loop runs in, so the result is bit-identical to that loop -- which is what libref
 * sums, since the reference sums a row in ascending index order and a wave of lanes maps to
 * consecutive elements.
 *
 * `__shfl_xor` lowers to ds_bpermute_b32 plus s_wait_dscnt, an LDS round trip that dominates a
 * dependency chain. DPP16's ROW_XMASK (control 0x160 | N) makes the partner read a modifier on the
 * consuming VALU op, and N = 16 crosses the two rows of a wave32 with v_permlanex16, still one VALU
 * op. Row and bank masks are all-ones because every lane participates; bound_ctrl is moot because
 * every lane has a partner. These are libr4d's encodings, verified on hardware. */
template <int N>
__device__ __forceinline__ uint32_t r3d_xor_lane_u32(uint32_t v) {
    static_assert(N >= 1 && N <= 16 && (N & (N - 1)) == 0, "a lane xor within the wave");
    if constexpr (N < 16)
        return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, 0x160 | N, 0xf, 0xf, true);
    else
        return (uint32_t)__builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210u, 0xFEDCBA98u,
                                                      false, false);
}

template <int N>
__device__ __forceinline__ float r3d_xor_lane(float v) {
    return __uint_as_float(r3d_xor_lane_u32<N>(__float_as_uint(v)));
}

/* A 64-bit value's partner: the halves exchanged separately, as a 64-bit shuffle does. */
template <int N>
__device__ __forceinline__ unsigned long long r3d_xor_lane_u64(unsigned long long v) {
    const uint32_t lo = r3d_xor_lane_u32<N>((uint32_t)v);
    const uint32_t hi = r3d_xor_lane_u32<N>((uint32_t)(v >> 32));
    return ((unsigned long long)hi << 32) | lo;
}

/* v_permlanex16 swaps only with identity selectors (each 4-bit field names the source lane in the
 * other half; sel0 covers lanes 0-7, sel1 lanes 8-15); zero selectors silently broadcast lane 0. */
__device__ __forceinline__ float swap16(float x) { return r3d_xor_lane<16>(x); }

/* v + the partner's v, the partner read by DPP so the add carries it. */
template <int CTRL>
__device__ __forceinline__ float r3d_dpp_xor_add(float v) {
    return v + __uint_as_float(__builtin_amdgcn_update_dpp(0, __float_as_uint(v), CTRL, 0xF, 0xF,
                                                           true));
}

__device__ __forceinline__ float wave_sum(float v) {
    v += swap16(v);                        /* xor 16 -- crosses the two rows of sixteen */
    v = r3d_dpp_xor_add<0x168>(v);         /* xor  8 -- row_xmask from here down */
    v = r3d_dpp_xor_add<0x164>(v);         /* xor  4 */
    v = r3d_dpp_xor_add<0x162>(v);         /* xor  2 */
    v = r3d_dpp_xor_add<0x161>(v);         /* xor  1 */
    return v;
}

__device__ __forceinline__ float wave_max(float v) {
    v = fmaxf(v, r3d_xor_lane<16>(v));
    v = fmaxf(v, r3d_xor_lane<8>(v));
    v = fmaxf(v, r3d_xor_lane<4>(v));
    v = fmaxf(v, r3d_xor_lane<2>(v));
    v = fmaxf(v, r3d_xor_lane<1>(v));
    return v;
}

/* The same butterflies over a group of W lanes (a power of two up to the wave), in the order
 * `for (o = W/2; o; o >>= 1) v = op(v, __shfl_xor(v, o))` runs them, so bit-identical to it. A row
 * of sixteen lanes -- the sixteen columns a lane owns in a bf16 pair layout -- uses W = 16. */
template <int W>
__device__ __forceinline__ float r3d_xor_sum(float v) {
    static_assert(W >= 1 && W <= 32 && (W & (W - 1)) == 0, "a power-of-two group within the wave");
    if constexpr (W >= 32) v += r3d_xor_lane<16>(v);
    if constexpr (W >= 16) v += r3d_xor_lane<8>(v);
    if constexpr (W >= 8)  v += r3d_xor_lane<4>(v);
    if constexpr (W >= 4)  v += r3d_xor_lane<2>(v);
    if constexpr (W >= 2)  v += r3d_xor_lane<1>(v);
    return v;
}

template <int W>
__device__ __forceinline__ float r3d_xor_max(float v) {
    static_assert(W >= 1 && W <= 32 && (W & (W - 1)) == 0, "a power-of-two group within the wave");
    if constexpr (W >= 32) v = fmaxf(v, r3d_xor_lane<16>(v));
    if constexpr (W >= 16) v = fmaxf(v, r3d_xor_lane<8>(v));
    if constexpr (W >= 8)  v = fmaxf(v, r3d_xor_lane<4>(v));
    if constexpr (W >= 4)  v = fmaxf(v, r3d_xor_lane<2>(v));
    if constexpr (W >= 2)  v = fmaxf(v, r3d_xor_lane<1>(v));
    return v;
}

/* A wave's inclusive prefix sum in registers: row_shr 1, 2, 4 and 8 within each row of sixteen,
 * then the other row's total from one readlane. Lane L ends with the sum over lanes 0..L. Replaces
 * an LDS step-doubling scan, which costs two barriers a round. Integer, so combine order is free.
 * Every lane must be active. */
__device__ __forceinline__ uint32_t wave_prefix_u32(uint32_t x, int lane) {
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x111, 0xF, 0xF, true);   /* row_shr:1 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x112, 0xF, 0xF, true);   /* row_shr:2 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x114, 0xF, 0xF, true);   /* row_shr:4 */
    x += (uint32_t)__builtin_amdgcn_update_dpp(0, (int)x, 0x118, 0xF, 0xF, true);   /* row_shr:8 */
    const uint32_t lo = (uint32_t)__builtin_amdgcn_readlane((int)x, 15);
    return lane >= 16 ? x + lo : x;
}

/* ================================================================== a block's sum
 *
 * Wave sum, one LDS round trip, then every thread adds the wave partials in wave order. The order
 * is the contract: libref sums a row in ascending index order, the waves cover consecutive
 * elements, and adding the partials in wave order is the same bracketing at any thread count.
 * `lds` holds NWAVES floats. */
template <int NTHREADS>
__device__ __forceinline__ float block_sum(float v, float* lds) {
    constexpr int NWAVES = NTHREADS / kWave;
    const uint32_t lane = threadIdx.x & (kWave - 1);
    const uint32_t w = threadIdx.x / kWave;
    v = wave_sum(v);
    if (lane == 0) lds[w] = v;
    __syncthreads();
    if constexpr (NWAVES == 1) return lds[0];
    float t = lds[0];
#pragma unroll
    for (int i = 1; i < NWAVES; ++i) t += lds[i];
    return t;
}

/* ================================================================== scheduling
 *
 * LLVM single-buffers LDS and drains before every dependent read unless told otherwise; on gfx11
 * the matrix unit is absent so there is nothing to overlap a load with, and the barrier the
 * kernels need is the plain one. */
__device__ __forceinline__ void sched_barrier() { __builtin_amdgcn_sched_barrier(0); }
__device__ __forceinline__ void group_barrier() { __syncthreads(); }

/* ================================================================== the geometry constants
 *
 * A gfx11 GPU's wave slots: a CU has 4 SIMDs, a SIMD holds 16 wave32s, so 64 slots a CU. HIP's
 * multiProcessorCount reports CUs on gfx11 (it reports WGPs on gfx12, which is where libr4d's 2048
 * comes from), and a launch whose grid passes 64 * multiProcessorCount waves has more work than
 * slots: the tail waves start after another wave retires. A row-per-block kernel with one wave a
 * block is exactly that cliff, and its row count is where the cliff is.
 *
 * A launcher's question, so a host function: this HIP declares no device-side hipGetDevice or
 * hipDeviceGetAttribute, and a __device__ body does not compile at all. */
static inline int wave_slot_count() {
    int dev = 0, cu = 0;
    (void)hipGetDevice(&dev);
    (void)hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, dev);
    return 64 * cu;
}

/* The LDS a workgroup gets is 64 KiB, the same figure RDNA4 gives, and no kernel in libr3d needs
 * more. There is no hipFuncSetAttribute opt-in: 64 KiB of dynamic shared memory is available as
 * soon as it is asked for. */
enum : int { kLdsCap = 64 * 1024 };

#endif  /* R3D_COMMON_H */
