/* Derived from libr4d/r4d_gemm_mxfp4a8_gemv.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_gemm_mxfp4a8_gated.hip - gemm_mxfp4a8_gated_gemv: the m64 kernel's WMMA form over a gate_up
// weight with gated_quant_fp8's pass in its epilogue, gfx1151. The row's constraint set is M <= 8,
// K a multiple of 128, N a multiple of 256. A block is one 16-column tile of the gated output:
// sixteen waves, eight K waves over the gate tile (weight rows 16 j ..) and eight over the up tile
// (rows n + 16 j ..), each tile's partials met in LDS in the m64 kernel's order.
//
// The block forms silu(gate) * up in bf16 for its sixteen columns and each row's amax over them.
// The quantiser's scale is per 128 columns (one activation scale group of the down projection's
// input), so the eight blocks of a group leave their products and amaxes in the op's scratch (a
// bf16 product plane [M, n] and amax tiles [n/16][M]) and the last of a group's eight blocks to
// arrive (an agent-scope counter per group, keyed on the output as the decode row's) takes the
// group's amax and writes its codes and scales.
//
// Byte-identical to the m64 kernel (halo_gemm_mxfp4a8_nt_m64.hip, the row `gemm_mxfp4a8_gemv`
// resolves to at every M) followed by gated_quant_fp8 on gfx1151, because the m64 kernel's
// accumulation order is shape-deterministic and M-independent:
//   - A and W both dequantise to exact bf16 and take halo::mma (the bf16 WMMA, exact products),
//     the e2m1 fold per 32-block against the per-row reference, exactly m64's fragment arithmetic;
//   - a chunk is a whole 32-block pair (the m64 k-loop step), one (row, 128) activation scale per
//     chunk on the block-scaled grid, fmaf'd into the running total exactly as m64's fold; the
//     per-32 fold d = ref - e8 reads N and K only (never M), so wave v of a tile runs chunk
//     v, v+16, ... of the same block's chunk order, and the LDS partials are summed v = 0, 1, ...
//     from zero, as the m64 kernel's epilogue adds its waves' rows of red[] in split order;
//   - on the per-row grid the m64 kernel scales the total once at the store; here the scale
//     multiplies into the bf16 round before silu, which is what the pair (m64 then
//     gated_quant_fp8) does too;
//   - from there the arithmetic is halo_fused_quant_fp8.hip's gated_quant_fp8_kernel: silu(gate)
//     * up rounded to bf16, the group's amax (a max, so the blocks' order does not matter), scale
//     amax / 448, codes through q8_pack4. ACT 1 is the GeGLU form: gelu_mul's tanh GELU with its
//     constants (halo_model_bf16.hip), the product rounded to bf16 as gelu_mul stores it, then
//     quant_act_fp8's quantiser, the same arithmetic as gated_quant_fp8's.
//
// gfx1151 differences from the reference form: both fragments are the gfx11 layout (a lane holds
// its one row's whole 16-wide k run of exact bf16), and the LDS partial is written at the gfx11
// accumulator mapping sP[half][kv][2e + (lane>>4)][lane & 15] -- for each (m, c) exactly one
// (lane, slot) writes, so the m64-order reduction over kv is unchanged. The A fragment is the
// row's full sixteen bytes (gfx12's khalf split of the load is gone with the half-split fragment).
#include "halo.h"
#include "halo_gemm_args.h"
#include "halo_common.h"
#include "halo_gdn_wmma.h"
#include "halo_fp8_wmma.h"
#include "halo_wmma.h"
#include "halo_mxfp4.h"
#include "halo_q8.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>

#define HALO_MXG_MAX_M 8           // the row's band (gemv's: M <= 8)
#define HALO_MXGG_MAX_GROUPS 1024  // the gated fold's counter block: a gated width up to 131072

typedef unsigned mxg_u2  __attribute__((ext_vector_type(2)));
typedef unsigned mxg_u4  __attribute__((ext_vector_type(4)));
typedef float    mxg_v8f __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float mxg_bf2f(uint16_t h) { return __uint_as_float((uint32_t)h << 16); }
__device__ __forceinline__ uint16_t mxg_f2bf(float f) {   // as halo_fused_quant_fp8.hip's f2bf
  const uint32_t u = __float_as_uint(f);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* One e2m1 code and its block's (exp - ref) as a bf16 bit pattern (m64's arithmetic). */
__device__ __forceinline__ unsigned mxg_e2m1_bf16(unsigned code, int ediff) {
  const unsigned mag = code & 7u;
  if (!mag) return (code & 8u) << 12;
  unsigned E = mag >> 1, m = mag & 1u;
  if (mag == 1) m = 0;   /* the subnormal 0.5: (1 + 0) * 2^-1, not (1 + 1/2) * 2^-1 */
  const unsigned ef = E + (unsigned)(ediff + 126);
  return ((code & 8u) << 12) | (ef << 7) | (m << 6);
}

/* Eight e2m1 codes at one (exp - ref) to four bf16-pair dwords, four codes a pass:
 * halo_gemm_mxfp4a8_tiled.hip's mxt_frag8 (that file says how), bit-identical to the scalar
 * form for every code and every ediff with 0 <= E + ediff + 126 <= 255. */
typedef unsigned short mxg_us2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ void mxg_frag8(unsigned dw, int ediff, uint32_t* out) {
    const uint32_t ed = (uint32_t)(ediff << 7) & 0xFFFFu;
    const mxg_us2 ED2 = __builtin_bit_cast(mxg_us2, ed | (ed << 16));
    const uint32_t z = (0u - ed) & 0xFFFFu;
    const uint32_t T_LO = 0xC0800000u | (z & 0xFFu), T_HI = 0x3F3F3F00u | (z >> 8);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t x = __builtin_amdgcn_perm(dw, dw, h ? 0x03030202u : 0x01010000u);
        const uint32_t mag = (x & 0x00070007u) | ((x >> 4) & 0x07000700u);
        const uint32_t sg = ((x << 4) & 0x00800080u) | (x & 0x80008000u);
        const uint32_t lo = __builtin_amdgcn_perm(0xC0804000u, T_LO, mag);
        const uint32_t hi = __builtin_amdgcn_perm(0x40404040u, T_HI, mag);
        const mxg_us2 p0 = __builtin_bit_cast(mxg_us2, __builtin_amdgcn_perm(hi, lo, 0x05010400u)) + ED2;
        const mxg_us2 p1 = __builtin_bit_cast(mxg_us2, __builtin_amdgcn_perm(hi, lo, 0x07030602u)) + ED2;
        out[2 * h]     = __builtin_bit_cast(uint32_t, p0) | __builtin_amdgcn_perm(0u, sg, 0x010C000Cu);
        out[2 * h + 1] = __builtin_bit_cast(uint32_t, p1) | __builtin_amdgcn_perm(0u, sg, 0x030C020Cu);
    }
}

/* The B fragment from this lane's own slot dword (row l & 15, codes k + 8 * (l >> 4) .. + 8) to
 * sixteen exact bf16 in consecutive-K order (m64's mx_frag16x): each half unpacks its dword and
 * the halves trade results by v_permlanex16, so a slot is one load and each code one unpack. */
__device__ __forceinline__ v16bf mxg_frag16x(unsigned dw, int exp, int ref, int lane) {
    uint32_t own[4], oth[4];
    mxg_frag8(dw, exp - ref, own);
#pragma unroll
    for (int q = 0; q < 4; ++q)
        oth[q] = __builtin_amdgcn_permlanex16(own[q], own[q], 0x76543210u, 0xfedcba98u, false, false);
    const bool up = lane >= 16;
    union { uint32_t u[8]; v16bf f; } r;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        r.u[q] = up ? oth[q] : own[q];
        r.u[4 + q] = up ? own[q] : oth[q];
    }
    return r.f;
}

/* The A fragment: sixteen e4m3 bytes to the exact bf16 fragment (m64's mx_deq16_exact). */
__device__ __forceinline__ v16bf mxg_af16(const uint32_t* __restrict__ w4) {
  union { uint32_t u[8]; v16bf f; } q;
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    const v2f p0 = halo_fp8::e4m3_pk<false>(w4[d]);
    const v2f p1 = halo_fp8::e4m3_pk<true>(w4[d]);
    q.u[d * 2]     = halo::pk_bf16(p0.x, p0.y);
    q.u[d * 2 + 1] = halo::pk_bf16(p1.x, p1.y);
  }
  return q.f;
}

/* The activation plane: mxg_af16 over the whole [M, K) activation, once per launch, so the
 * blocks read their fragments instead of each re-decoding the same codes (n / 16 blocks of SK
 * waves). Values are mxg_af16's own; products through the plane are bit-identical. */
__global__ __launch_bounds__(256) void mxg_adq_kernel(const unsigned char* __restrict__ A,
                                                      uint16_t* __restrict__ Adq, int M, int K,
                                                      int lda) {
  const int idx = blockIdx.x * (int)blockDim.x + threadIdx.x;
  if (idx >= M * (K / 16)) return;
  const int r = idx / (K / 16), c = (idx % (K / 16)) * 16;
  const v16bf f = mxg_af16((const uint32_t*)(A + (size_t)r * lda + c));
  uint4* dst = (uint4*)(Adq + (size_t)r * K + c);
  const uint4* src = (const uint4*)&f;
  dst[0] = src[0];
  dst[1] = src[1];
}

template <bool ABLK, int ACT, bool APLANE>
__global__ void halo_gemm_mxfp4a8_gated_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, uint8_t* __restrict__ Q, float* __restrict__ S,
    uint16_t* __restrict__ Y, uint16_t* __restrict__ H, unsigned* __restrict__ cnt, int M, int K,
    int n, int SK, int as_rs, int lda, int ldws, int q_ld, int s_ld, int y_ld,
    const uint16_t* __restrict__ Adq) {
  // One block per 16-column tile of the gated output: SK waves, each a contiguous K split, as the
  // m64 kernel's geometry (halo_gemm_mxfp4a8_nt_m64.hip), whose accumulation order this row must
  // reproduce bit for bit. red[] holds each split's 8 accumulators per half: [2][SK][256].
  extern __shared__ float red[];
  __shared__ unsigned s_last;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int nwaves = __builtin_amdgcn_readfirstlane(blockDim.x >> 5);
  const int ksplit = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int j = blockIdx.x;                     // the gated output's 16-column tile
  const int nlane = lane & 15;
  const int ntg = j, ntu = (n >> 4) + j;        // the gate and up weight tiles
  const int wrowg = ntg * 16 + nlane, wrowu = ntu * 16 + nlane;
  const float rfg = __int_as_float((int)Wref[wrowg] << 23);
  const float rfu = __int_as_float((int)Wref[wrowu] << 23);
  const int Ks = K / SK, kbase = ksplit * Ks;
  const int ksteps = K >> 4;
  const unsigned* __restrict__ Wp = (const unsigned*)W;
  const int r = nlane < M - 1 ? nlane : M - 1;   // m64's A-row clamp, discarded rows included
  mxg_v8f accg = (mxg_v8f)(0.f), accu = (mxg_v8f)(0.f);
  for (int k = kbase; k < kbase + Ks; k += 32) {
    mxg_v8f tg = (mxg_v8f)(0.f), tu = (mxg_v8f)(0.f);
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int kg = k + u * 16, kstep = kg >> 4;
      v16bf af;
      if constexpr (APLANE) {
        union { uint4 q[2]; v16bf f; } x;
        x.q[0] = *(const uint4*)(Adq + (size_t)r * K + kg);
        x.q[1] = *(const uint4*)(Adq + (size_t)r * K + kg + 8);
        af = x.f;
      } else {
        af = mxg_af16((const uint32_t*)(A + (size_t)r * lda + kg));
      }
      const unsigned gw = __builtin_nontemporal_load(&Wp[((size_t)ntg * ksteps + kstep) * 32 + lane]);
      tg = halo::mma(af, mxg_frag16x(gw, (int)Ws[(size_t)(k >> 5) * ldws + wrowg],
                                     (int)Wref[wrowg], lane),
                     tg);
      const unsigned uw = __builtin_nontemporal_load(&Wp[((size_t)ntu * ksteps + kstep) * 32 + lane]);
      tu = halo::mma(af, mxg_frag16x(uw, (int)Ws[(size_t)(k >> 5) * ldws + wrowu],
                                     (int)Wref[wrowu], lane),
                     tu);
    }
    // The 32-block's fold, exactly as m64's: f = rf, and on the block-scaled grid times the
    // element row's (row, 128) activation scale. The element row is 2e + (lane >> 4).
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int re = 2 * e + (lane >> 4);
      const int rc = re < M - 1 ? re : M - 1;
      float fg = rfg, fu = rfu;
      if constexpr (ABLK) {
        const float am = As[(size_t)rc * as_rs + (k >> 7)];
        fg *= am;
        fu *= am;
      }
      accg[e] = fmaf(tg[e], fg, accg[e]);
      accu[e] = fmaf(tu[e], fu, accu[e]);
    }
  }
  // Each split's partials in LDS, reduced in ascending split order as m64's epilogue sums its red[].
  float* __restrict__ sg_ = red + (size_t)ksplit * 256;
  float* __restrict__ su_ = red + (size_t)(SK + ksplit) * 256;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    sg_[e * 32 + lane] = accg[e];
    su_[e * 32 + lane] = accu[e];
  }
  __syncthreads();
  const int g = j >> 3;                          // the 128-column group
  if (tid < 256) {
    const int e = tid >> 5, l = tid & 31;
    const int m = 2 * e + (l >> 4), c = l & 15;
    float vg = 0.f, vu = 0.f;
    for (int s = 0; s < SK; ++s) {
      vg += red[(size_t)s * 256 + tid];
      vu += red[(size_t)(SK + s) * 256 + tid];
    }
    // m64's store: the per-row grid multiplies by As[m] once, the block-scaled grid does not.
    if constexpr (!ABLK) {
      const float am = As[(size_t)m * as_rs];
      vg *= am;
      vu *= am;
    }
    const uint16_t hg = mxg_f2bf(vg), hu = mxg_f2bf(vu);
    const float gv = mxg_bf2f(hg);
    // silu (SwiGLU) or the tanh GELU gelu_mul uses (GeGLU), both in float then rounded once.
    float a_;
    if constexpr (ACT == 0)
      a_ = gv / (1.0f + expf(-gv));
    else {
      const float k0 = 0.7978845608028654f, k1 = 0.044715f;
      a_ = 0.5f * gv * (1.0f + tanhf(k0 * (gv + k1 * gv * gv * gv)));
    }
    const uint16_t h = mxg_f2bf(a_ * mxg_bf2f(hu));
    const int cg = j * 16 + c;
    if (m < M) {
      if (Y) Y[(size_t)m * y_ld + cg] = h;
      H[(size_t)m * n + cg] = h;
    }
  }
  // Every thread's stores are ordered before the counter; then the group's last arriver quantises.
  __threadfence();
  __syncthreads();
  if (tid == 0)
    s_last = __hip_atomic_fetch_add(&cnt[g], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
  __syncthreads();
  if (s_last != 7u) return;                      // the group's eight blocks all landed
  if (tid == 0) cnt[g] = 0u;                     // reset for the next launch
  __threadfence();                               // acquire, before reading the other blocks'
  // gated_quant_fp8's quantiser: one wave a row's group, a lane four consecutive columns.
  for (int m = __builtin_amdgcn_readfirstlane(tid >> 5); m < M; m += nwaves) {
    const int c0 = g * 128 + lane * 4;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = mxg_bf2f(H[(size_t)m * n + c0 + i]);
    float mx = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o));
    const float sc = mx > 0.0f ? mx * (1.0f / q8_top<false>()) : 1.0f;
    if (lane == 0) S[(size_t)m * s_ld + g] = sc;
    const float inv = 1.0f / sc;
    *(uint32_t*)(Q + (size_t)m * q_ld + c0) = q8_pack4<false>(v[0], v[1], v[2], v[3], inv);
  }
}

namespace {
// The arrival counters, one block per output buffer, as the decode row's (mxd_out_ctr): not in the
// scratch, which other ops overwrite, since they must be zero at entry. One allocation per (device,
// output), kept zero by the reducer and never freed (a recorded pass replays the pointer).
unsigned* mxgg_out_ctr(const void* out) {
  static std::mutex mu;
  static std::map<std::pair<int, const void*>, unsigned*> blocks;
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) return nullptr;
  std::lock_guard<std::mutex> lk(mu);
  const auto key = std::make_pair(dev, out);
  if (auto it = blocks.find(key); it != blocks.end()) return it->second;
  unsigned* p = nullptr;
  if (hipMalloc((void**)&p, HALO_MXGG_MAX_GROUPS * sizeof(unsigned)) != hipSuccess) return nullptr;
  // Zero before any launch reads it: the memset is on the null stream, which the engine's streams
  // need not wait for, so wait here (as mxd_out_ctr and the MoE router counters do).
  if (hipMemset(p, 0, HALO_MXGG_MAX_GROUPS * sizeof(unsigned)) != hipSuccess ||
      hipDeviceSynchronize() != hipSuccess) {
    (void)hipFree(p);
    return nullptr;
  }
  blocks.emplace(key, p);
  return p;
}

// The scratch: the gated product [M, n] in bf16. The quantiser reads it back after the group's
// last block arrives (it needs all 128 columns' values for the amax), so it cannot live in LDS.
// Behind it, the decoded activation plane [M, K] bf16 when the hook sized it (else the blocks
// decode in-kernel).
long long mxgg_h_bytes(long long M, long long n) { return (M * n * 2 + 255) / 256 * 256; }
long long mxgg_scratch_bytes(long long M, long long n) { return mxgg_h_bytes(M, n); }
long long mxgg_plane_bytes(long long M, long long K) { return M * K * 2; }

void mxgg_launch(bool ablk, int act, const void* a, const void* as, const void* w, const void* ws,
                 const void* wr, void* q, void* s, void* y, void* scratch, long long scratch_bytes,
                 unsigned* cnt, int M, int K, int n, int SK, int as_rs, int lda, int ldws, int q_ld,
                 int s_ld, int y_ld, RadStream stream) {
  const dim3 grid((uint32_t)(n / 16)), block((uint32_t)(SK * 32));
  const size_t smem = (size_t)2 * SK * 256 * sizeof(float);
  hipStream_t st = (hipStream_t)stream;
  uint16_t* h = (uint16_t*)scratch;
  const long long hb = mxgg_h_bytes(M, n);
  uint16_t* adq = scratch_bytes >= hb + mxgg_plane_bytes(M, K) ? (uint16_t*)((char*)scratch + hb)
                                                                : nullptr;
  if (adq) {
    const int total = M * (K / 16);
    hipLaunchKernelGGL(mxg_adq_kernel, dim3((total + 255) / 256), dim3(256), 0, st,
                       (const unsigned char*)a, adq, M, K, lda);
  }
#define MXGG_G(AB_, ACT_, PL_)                                                                     \
  hipLaunchKernelGGL((halo_gemm_mxfp4a8_gated_kernel<AB_, ACT_, PL_>), grid, block, smem, st,      \
                     (const unsigned char*)a, (const float*)as, (const unsigned char*)w,           \
                     (const unsigned char*)ws, (const unsigned char*)wr, (uint8_t*)q, (float*)s,   \
                     (uint16_t*)y, h, cnt, M, K, n, SK, as_rs, lda, ldws, q_ld, s_ld, y_ld, adq)
#define MXGG_L(AB_, ACT_) do { if (adq) MXGG_G(AB_, ACT_, true); else MXGG_G(AB_, ACT_, false); } while (0)
  if (act == 0) { if (ablk) MXGG_L(true, 0); else MXGG_L(false, 0); }
  else          { if (ablk) MXGG_L(true, 1); else MXGG_L(false, 1); }
#undef MXGG_L
#undef MXGG_G
}
}  // namespace


/* The gated GEMM's shim. Operand indices are `gemm_nt_q_gated_mx`'s (oGemmNtQGatedMx in
 * halo_rows.cpp): gemm_nt_q_gated's with the per-row reference `b_ref` after `b_scale`. The weight
 * is one [N, K] plane, gate rows then up rows, N = 2 * the gated width; no parts. */
enum { MXGG_A = 0, MXGG_AS = 1, MXGG_B = 2, MXGG_BS = 3, MXGG_BR = 4, MXGG_Q = 5, MXGG_S = 6,
       MXGG_Y = 7 };

extern "C" int halo_rad_gemm_mxfp4a8_gated(const RadArgs* a, RadStream s) {
    const RadTensor* x  = halo_opt(a, MXGG_A);
    const RadTensor* xs = halo_opt(a, MXGG_AS);
    const RadTensor* w  = halo_opt(a, MXGG_B);
    const RadTensor* ws = halo_opt(a, MXGG_BS);
    const RadTensor* wr = halo_opt(a, MXGG_BR);
    const RadTensor* q  = halo_opt(a, MXGG_Q);
    const RadTensor* sc = halo_opt(a, MXGG_S);
    const RadTensor* y  = halo_opt(a, MXGG_Y);
    if (!x || !xs || !w || !ws || !wr || !q || !sc) return HALO_NO(RAD_E_INVAL);
    if (x->dtype != RAD_F8E4M3 || xs->dtype != RAD_F32 || q->dtype != RAD_F8E4M3 ||
        sc->dtype != RAD_F32 || (y && y->dtype != RAD_BF16))
        return HALO_NO(RAD_E_DTYPE);
    if (q->rank != 2 || sc->rank != 2 || x->rank != 2) return HALO_NO(RAD_E_SHAPE);
    if (y && y->rank != 2) return HALO_NO(RAD_E_SHAPE);
    if (halo_geti_or(a, "group", 32) != 32) return HALO_NO(RAD_E_SHAPE);
    /* silu (the SwiGLU MLP) or gelu_tanh (GeGLU, gelu_mul's). The GeGLU chain's bf16 plane is
     * gelu_mul's output, which the fold's replaces-hook cannot hand to the optional `out_bf16`, so
     * that form writes none. */
    const char* act = halo_gets(a, "act");
    const int actm = !act ? -1 : !std::strcmp(act, "silu") ? 0 : !std::strcmp(act, "gelu_tanh") ? 1 : -1;
    if (actm < 0) return HALO_NO(RAD_E_UNSUPPORTED);
    if (actm == 1 && y) return HALO_NO(RAD_E_UNSUPPORTED);

    /* The gated width from `q`, the buffer written; N and K from the op's parameters, since the
     * weight operands are packed bytes. */
    const long long M = q->shape[0], nout = q->shape[1];
    const long long N = halo_geti_or(a, "N", 0), K = halo_geti_or(a, "K", 0);
    if (M < 1 || M > HALO_MXG_MAX_M || nout < 128 || nout % 128 || nout / 128 > HALO_MXGG_MAX_GROUPS)
        return HALO_NO(RAD_E_SHAPE);
    if (N != 2 * nout || K < 128 || K % 128) return HALO_NO(RAD_E_SHAPE);
    if (x->shape[0] != M || halo_numel(x) != M * K) return HALO_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != nout / 128) return HALO_NO(RAD_E_SHAPE);
    if (y && (y->shape[0] != M || y->shape[1] != nout)) return HALO_NO(RAD_E_SHAPE);
    /* The activation scale grid as the gemm_nt_q rows read it (mx_ascale_grid), so the fold and the
     * GEMM it replaces take every scale plane the same way. */
    GemmOps go{};
    go.ascale = xs; go.M = M; go.K = K;
    int as_group = 0, as_rs_i = 1;
    if (mx_ascale_grid(go, &as_group, &as_rs_i) != RAD_OK) return HALO_NO(RAD_E_SHAPE);
    const bool ablk = as_group == 128;
    const long long as_rs = as_rs_i;
    /* Row pitches; halo_row_pitch is -1 for a tensor not contiguous past its leading axis. A is read
     * a dword at a time. */
    const long long lda = halo_row_pitch(x), qp = halo_row_pitch(q), sp = halo_row_pitch(sc);
    const long long yp = y ? halo_row_pitch(y) : 0;
    if (lda < K || lda % 4 || qp < nout || sp < nout / 128 || (y && yp < nout))
        return HALO_NO(RAD_E_STRIDE);
    if (!halo_aligned(x->data, 16) || !halo_aligned(w->data, 16) || !halo_aligned(q->data, 4) ||
        !halo_aligned(ws->data, 4) || !halo_aligned(wr->data, 4) || qp % 4)
        return HALO_NO(RAD_E_ALIGN);
    /* The split law. This row must reproduce gemv's accumulation order bit for bit, and gemv
     * resolves to gemm_mxfp4a8_nt_m64 with cfg_for(K, M, 32, BR_FLAT) (halo_gemm_mxfp4a8_gemv.hip).
     * Asking the same law the same question gets the same SK, hence the same contiguous splits in
     * the same order. A tuned cfg that widens WV or NPW would change m64's geometry, so it is
     * refused here rather than silently differing. */
    Cfg cfg{};
    if (cfg_for(a, K, M, HALO_MXFP4_GROUP, BR_FLAT, &cfg, 0, 1, 16) != RAD_OK) return HALO_NO(RAD_E_SHAPE);
    if (cfg.wv != 1 || cfg.npw != 1) return HALO_NO(RAD_E_UNSUPPORTED);
    const int SK = cfg.sk;
    if ((size_t)2 * SK * 256 * sizeof(float) > 60 * 1024) return HALO_NO(RAD_E_UNSUPPORTED);
    /* The cross-block product plane is the op's scratch (sized by the hook below); a shorter one is
     * refused. */
    if (!a->scratch || a->scratch_bytes < mxgg_scratch_bytes(M, nout)) return HALO_NO(RAD_E_SCRATCH);
    unsigned* cnt = mxgg_out_ctr(q->data);
    if (!cnt) return HALO_NO(RAD_E_NOMEM);

    HALO_TRY("gemm_mxfp4a8_gated",
        mxgg_launch(ablk, actm, x->data, xs->data, w->data, ws->data, wr->data, q->data, sc->data,
                    y ? y->data : nullptr, a->scratch, a->scratch_bytes, cnt, (int)M, (int)K, (int)nout, SK,
                    (int)as_rs, (int)lda, (int)N, (int)qp, (int)sp, (int)yp, s));
    return RAD_OK;
}

/* The op's scratch: the product and amaxes at the band's widest M, from N (twice the gated
 * width), and the decoded activation plane behind them. */
extern "C" int64_t halo_rad_gemm_mxfp4a8_gated_scratch(const RadArgs* a) {
    const long long M = halo_geti_or(a, "M", 0), N = halo_geti_or(a, "N", 0);
    const long long K = halo_geti_or(a, "K", 0);
    if (M < 1 || M > HALO_MXG_MAX_M || N < 256 || N % 256) return 0;
    return (int64_t)(mxgg_scratch_bytes(M, N / 2) + (K > 0 && K % 16 == 0 ? mxgg_plane_bytes(M, K) : 0));
}
