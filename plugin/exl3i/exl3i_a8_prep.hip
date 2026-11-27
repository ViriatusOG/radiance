// exl3i_a8_prep.hip -- the A8 path's two producer passes (exl3i_a8.h): the rows' rotation and int8 quantisation
// (actq), and the weight's decode into radiance's int8 fragment plane (wdec8). Both use the shared device
// code as is: rotate128_math (so a row's rotated values are the f16 path's, bit for bit, before they are quantised)
// and the rate policies' decode (exl3i_rate.hip.h: RateInt / RateFrac, codebook mad), whose values
// the plane holds as their int8 grid index. Kernel names carry exl3i_ (exl3i_nt.hip.h).
#include <hip/hip_runtime.h>
#include <stdint.h>

#include <cstdlib>
#include <mutex>

#include "exl3r4d_common.hip.h"
#include "exl3i_rate.hip.h"
#include "exl3i_a8.h"
#include "exl3i_a8_dev.h"
#include "exl3i_abi.h"   // EXL3I_AQ_PRE_*
#include "exl3i_8a8.hip.h"
#include "exl3i_a8q.hip.h"
#include "exl3i_pin.h"

// ============================================================================= actq
// One wave a (mat, row, group) task: the group's 128-feature blocks rotated as the f16 path's prep kernel rotates
// them (rotate128_math: fp16(x * suh), the transform in f32, x 1/sqrt(128), fp16), the group's absmax over the wave,
// then each value quantised (exl3i_a8_q) and the scale written once a 128-K block (repeated over the group). NB: the
// group's blocks held in registers (1, 2, 4, 8 = a8g128 .. a8g1024); 0: any count, rotated twice (absmax, then the
// values: a8t and the wider groups).
template <int NB>
__global__ void __launch_bounds__(256) exl3i_a8_actq_kernel(Exl3iA8Args a) {
  const int lane = (int)threadIdx.x % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane((int)threadIdx.x / WAVE32);
  const int G = a.K / a.group, nbg = a.group / 128, nkb = a.K / 128;
  const long long task = (long long)blockIdx.x * 8 + wave, per = (long long)a.M * G;
  if (task >= per * a.mats) return;
  const int mat = (int)(task / per), row = (int)(task % per / G), g = (int)(task % G);
  const unsigned short* const xr = (const unsigned short*)a.x + (size_t)row * a.x_pitch;
  const _Float16* const su = (const _Float16*)(mat ? a.suh1 : a.suh);
  const bool f16 = a.x_bf16 == 0;
  const int kb0 = g * nbg;
  unsigned* const qo = (unsigned*)(a.aq + ((size_t)mat * a.M + row) * a.K) + kb0 * 32 + lane;
  float* const so = a.sa + ((size_t)mat * a.M + row) * nkb + kb0;
  auto rot = [&](int b) {
    const v2u x = *(const v2u*)(xr + (size_t)(kb0 + b) * 128 + 4 * lane);
    const v2u s = *(const v2u*)(su + (size_t)(kb0 + b) * 128 + 4 * lane);
    return rotate128_math(x, s, f16, lane);
  };
  if constexpr (NB > 0) {
    v2u v[NB];
    float m = 0.0f;
#pragma unroll
    for (int b = 0; b < NB; b++) { v[b] = rot(b); m = fmaxf(m, amax4(v[b])); }
    m = wave_max(m);
    const float inv = exl3i_a8_inv(m), sc = exl3i_a8_scale(m);
#pragma unroll
    for (int b = 0; b < NB; b++) {
      qo[b * 32] = quant4(v[b], inv);
      if (lane == 0) so[b] = sc;
    }
  } else {
    float m = 0.0f;
    for (int b = 0; b < nbg; b++) m = fmaxf(m, amax4(rot(b)));
    m = wave_max(m);
    const float inv = exl3i_a8_inv(m), sc = exl3i_a8_scale(m);
    for (int b = 0; b < nbg; b++) {
      qo[b * 32] = quant4(rot(b), inv);
      if (lane == 0) so[b] = sc;
    }
  }
}

extern "C" int exl3i_a8_actq(const Exl3iA8Args* a, void* stream) {
  if (!a || a->M < 1 || a->K % 128 || a->group < 128 || a->group % 128 || a->K % a->group) return RAD_E_SHAPE;
  if (a->mats == 2 && !a->suh1) return RAD_E_INVAL;
  const long long tasks = (long long)a->mats * a->M * (a->K / a->group);
  if ((tasks + 7) / 8 > 0x7FFFFFFFLL) return RAD_E_SHAPE;
  const void* fn;
  switch (a->group / 128) {
    case 1: fn = (const void*)exl3i_a8_actq_kernel<1>; break;
    case 2: fn = (const void*)exl3i_a8_actq_kernel<2>; break;
    case 4: fn = (const void*)exl3i_a8_actq_kernel<4>; break;
    case 8: fn = (const void*)exl3i_a8_actq_kernel<8>; break;
    default: fn = (const void*)exl3i_a8_actq_kernel<0>; break;
  }
  Exl3iA8Args q = *a;
  void* args[] = {&q};
  return hipLaunchKernel(fn, dim3((unsigned)((tasks + 7) / 8)), dim3(256), args, 0, (hipStream_t)stream) == hipSuccess
             ? RAD_OK : RAD_E_DEVICE;
}

// ============================================================================= AQ: exl3i_actq (exl3i_abi.h version 3)
// For each consumer i the plane exactly as exl3i_a8_actq_kernel<1> makes it at a8g128 (rotate128_math with s_i, the
// wave's absmax, exl3i_a8_scale / _inv, quant4): planes byte for byte the dense op's own. Packed rows: [planes][K] int8,
// then [planes][K / 128] f32. The 128 values a wave quantises are x's, or the producer stage's (pre), computed as
// radiance's unfused op computes them and rounded to bf16 as it stores them (libr4d r4d_model_bf16.hip:
// rmsnorm_body, silu_mul_kernel, binary_kernel; the arithmetic below is theirs, -ffp-contract=off on both sides).

// f32_to_bf16 / bf16_to_f32 (round to nearest even in software; a shift back)
static __device__ __forceinline__ unsigned short aq_f2bf(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static __device__ __forceinline__ float aq_bf2f(unsigned short h) { return __builtin_bit_cast(float, (unsigned)h << 16); }
static __device__ __forceinline__ v2u aq_pack4(const unsigned short* h) {
  v2u v;
  v[0] = (unsigned)h[0] | (unsigned)h[1] << 16;
  v[1] = (unsigned)h[2] | (unsigned)h[3] << 16;
  return v;
}

// One wave's 128 values (lane: features 4 lane .. 4 lane + 3 of block kb) into every consumer's plane of row `row`.
// Form xb (version 4, exl3i_abi.h): the same q as x 0x01010101 int32, the scale, and the block's two -126 x bias sums
// (lane l holds k % 16 = 4 (l % 4) .. + 3 of k step l / 4: half l & 1) -- exl3i_mad.hip.h's own quantiser's bytes.
static __device__ __forceinline__ void aq_planes(const Exl3iAqArgs& a, v2u x, int row, int kb, int lane, bool f16) {
  const int nkb = a.K / 128;
  uint8_t* const o = a.out + (size_t)row * a.out_ld;
  float* const so = (float*)(o + (size_t)a.planes * a.K);
  const size_t xbp = (size_t)EXL3I_AQ_XB_PLANE_BYTES(a.K);
#pragma unroll 1
  for (int i = 0; i < a.planes; i++) {
    const v2u sv = *(const v2u*)((const _Float16*)a.s[i] + (size_t)kb * 128 + 4 * lane);
    const v2u v = rotate128_math(x, sv, f16, lane);
    float m = 0.0f;   // exl3i_a8_actq_kernel<1>'s fold, step for step (a NaN block included)
    m = fmaxf(m, amax4(v));
    m = wave_max(m);
    const float inv = exl3i_a8_inv(m), sc = exl3i_a8_scale(m);
    const unsigned q4 = quant4(v, inv);
    if (a.form == EXL3I_AQ_FORM_XB) {
      uint8_t* const pb = o + (size_t)i * xbp;
      v4u xb;
#pragma unroll
      for (int e = 0; e < 4; e++) xb[e] = ((q4 >> (8 * e)) & 0xFFu) * 0x01010101u;
      *(v4u*)(pb + (size_t)(kb * 128 + 4 * lane) * 4) = xb;
      int s4 = __builtin_amdgcn_sudot4(true, (int)q4, true, 0x01010101, 0, false);
      s4 += __shfl_xor(s4, 2, WAVE32);
      s4 += __shfl_xor(s4, 4, WAVE32);
      s4 += __shfl_xor(s4, 8, WAVE32);
      s4 += __shfl_xor(s4, 16, WAVE32);
      if (lane == 0) ((float*)(pb + (size_t)4 * a.K))[kb] = sc;
      if (lane < 2) ((int*)(pb + (size_t)4 * a.K + 4 * nkb))[2 * kb + lane] = 126 * s4;
    } else {
      *((unsigned*)(o + (size_t)i * a.K + (size_t)kb * 128) + lane) = q4;
      if (lane == 0) so[i * nkb + kb] = sc;
    }
  }
}

// none / silu_mul / mul: one wave a (row, 128-feature block), the block read once for every consumer.
template <int PRE>
__global__ void __launch_bounds__(256) exl3i_aq_actq_kernel(Exl3iAqArgs a) {
  const int lane = (int)threadIdx.x % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane((int)threadIdx.x / WAVE32);
  const int nkb = a.K / 128;
  const long long task = (long long)blockIdx.x * 8 + wave;
  if (task >= (long long)a.M * nkb) return;
  const int row = (int)(task / nkb), kb = (int)(task % nkb);
  const size_t c = (size_t)kb * 128 + 4 * lane;
  const unsigned short* const xr = (const unsigned short*)a.x + (size_t)row * a.x_pitch;
  v2u x;
  if constexpr (PRE == EXL3I_AQ_PRE_NONE) {
    x = *(const v2u*)(xr + c);
  } else {
    const v2u g = *(const v2u*)(xr + c);
    const v2u u = PRE == EXL3I_AQ_PRE_SILU_MUL ? *(const v2u*)(xr + a.K + c)
                                               : *(const v2u*)((const unsigned short*)a.x2 + (size_t)row * a.x2_pitch + c);
    unsigned short hv[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float gf = aq_bf2f((unsigned short)(g[j >> 1] >> (16 * (j & 1))));
      const float uf = aq_bf2f((unsigned short)(u[j >> 1] >> (16 * (j & 1))));
      // silu_mul_kernel: (g / (1 + expf(-g))) * u; binary_kernel (mul): u * v
      hv[j] = PRE == EXL3I_AQ_PRE_SILU_MUL ? aq_f2bf((gf / (1.0f + expf(-gf))) * uf) : aq_f2bf(gf * uf);
    }
    x = aq_pack4(hv);
    if (a.h) *(v2u*)((unsigned short*)a.h + (size_t)row * a.h_pitch + c) = x;
  }
  aq_planes(a, x, row, kb, lane, PRE == EXL3I_AQ_PRE_NONE && a.x_bf16 == 0);
}

// x + res (version 5): add_rmsnorm_body's sum, bf16(f32(x) + f32(res)) -- what the norm reads and what xo receives
static __device__ __forceinline__ unsigned short aq_add(unsigned short x, unsigned short r) {
  return aq_f2bf(aq_bf2f(x) + aq_bf2f(r));
}

// rmsnorm: a workgroup of 256 a row. rmsnorm_body's sum of squares (r4d_lanes_sumsq: lane v adds columns v, v + 256, ...
// in order; the wave's butterfly xor 16, 8, 4, 2, 1 -- r4d_wave_sum's exchanges and order, lane 0's sum kept; the wave
// sums added from the first), its scale rsqrtf(ss / n + eps), its store bf16(x s (g + wadd)); then each wave its blocks.
// A few rows (the decode band: gridDim.y > 1) spread each row over gridDim.y workgroups, each summing the row's squares
// itself (the same order, so the same scale) and taking blocks blockIdx.y 8 + wave, + 8 gridDim.y, ...: one workgroup
// a row left the decode step's norm one workgroup long (14 us a site against rmsnorm's 9).
// With res (version 5) the row is x + res (aq_add) throughout, written to xo; xo may be x and h may be res (in place:
// exl3i_abi.h). One workgroup a row writes xo and h block by block, its reads of the sum of squares behind the barrier;
// over several, each reads the whole row, so the row's last workgroup to finish (rctr[row], reset by it) writes xo and h
// whole once every one of them has read x and res.
__global__ void __launch_bounds__(256) exl3i_aq_rmsnorm_kernel(Exl3iAqArgs a) {
  __shared__ float lds[8];
  __shared__ unsigned s_arrived;
  const int lane = (int)threadIdx.x % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane((int)threadIdx.x / WAVE32);
  const int row = (int)blockIdx.x;
  const unsigned short* const xr = (const unsigned short*)a.x + (size_t)row * a.x_pitch;
  const unsigned short* const rr = a.res ? (const unsigned short*)a.res + (size_t)row * a.res_pitch : nullptr;
  float ss = 0.0f;
  {
    unsigned i = threadIdx.x;
    const unsigned n = (unsigned)a.K;
    if (rr) {
      for (; i + 7 * 256 < n; i += 8 * 256) {
        float v[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = aq_bf2f(aq_add(xr[i + k * 256], rr[i + k * 256]));
#pragma unroll
        for (int k = 0; k < 8; ++k) ss += v[k] * v[k];
      }
      for (; i < n; i += 256) { const float v = aq_bf2f(aq_add(xr[i], rr[i])); ss += v * v; }
    } else {
      for (; i + 7 * 256 < n; i += 8 * 256) {   // r4d_lanes_sumsq's eight loads before their adds; the adds in order
        float v[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) v[k] = aq_bf2f(xr[i + k * 256]);
#pragma unroll
        for (int k = 0; k < 8; ++k) ss += v[k] * v[k];
      }
      for (; i < n; i += 256) { const float v = aq_bf2f(xr[i]); ss += v * v; }
    }
  }
  ss += __shfl_xor(ss, 16, WAVE32);
  ss += __shfl_xor(ss, 8, WAVE32);
  ss += __shfl_xor(ss, 4, WAVE32);
  ss += __shfl_xor(ss, 2, WAVE32);
  ss += __shfl_xor(ss, 1, WAVE32);
  if (lane == 0) lds[wave] = ss;
  __syncthreads();
  float tot = 0.0f;
#pragma unroll
  for (int w = 0; w < 8; ++w) tot += lds[w];
  const float sc = rsqrtf(tot / (float)a.K + a.eps);
  const int nkb = a.K / 128;
  const bool late = rr && gridDim.y > 1;   // xo and h left to the row's last workgroup
#pragma unroll 1
  for (int kb = (int)blockIdx.y * 8 + wave; kb < nkb; kb += 8 * (int)gridDim.y) {
    const size_t c = (size_t)kb * 128 + 4 * lane;
    v2u xv = *(const v2u*)(xr + c);
    if (rr) {
      const v2u rv = *(const v2u*)(rr + c);
      unsigned short sv[4];
#pragma unroll
      for (int j = 0; j < 4; j++)
        sv[j] = aq_add((unsigned short)(xv[j >> 1] >> (16 * (j & 1))), (unsigned short)(rv[j >> 1] >> (16 * (j & 1))));
      xv = aq_pack4(sv);
      if (!late) *(v2u*)((unsigned short*)a.xo + (size_t)row * a.xo_pitch + c) = xv;
    }
    unsigned short hv[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float g = a.g_f32 ? ((const float*)a.g)[c + j] : aq_bf2f(((const unsigned short*)a.g)[c + j]);
      hv[j] = aq_f2bf(aq_bf2f((unsigned short)(xv[j >> 1] >> (16 * (j & 1)))) * sc * (g + a.wadd));
    }
    const v2u x = aq_pack4(hv);
    if (a.h && !late) *(v2u*)((unsigned short*)a.h + (size_t)row * a.h_pitch + c) = x;
    aq_planes(a, x, row, kb, lane, false);
  }
  if (!late) return;
  // KS's last-arriver scheme (exl3i_nt.hip.h): the arrivals after each workgroup's last read of x and res
  __syncthreads();
  if (threadIdx.x == 0)
    s_arrived = __hip_atomic_fetch_add(&a.rctr[row], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
  __syncthreads();
  if (s_arrived != gridDim.y - 1u) return;   // workgroup-uniform
  if (threadIdx.x == 0) __hip_atomic_store(&a.rctr[row], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  // Each thread its 4-element groups q = threadIdx.x + 256 j, a batch of eight loaded before any of their stores: an
  // element is read and written by one thread only, so xo over x and h over res stay right, and the loads are not left
  // behind each store (the compiler must assume xo / h may alias the next group's x / res).
  unsigned short* const xo = (unsigned short*)a.xo + (size_t)row * a.xo_pitch;
  unsigned short* const hr = a.h ? (unsigned short*)a.h + (size_t)row * a.h_pitch : nullptr;
  const int ng = a.K / 4;
#pragma unroll 1
  for (int q0 = (int)threadIdx.x; q0 < ng; q0 += 8 * 256) {
    v2u xv[8], rv[8];
    float gv[8][4];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int q = q0 + 256 * j;
      if (q >= ng) break;
      xv[j] = *(const v2u*)(xr + 4 * q);
      rv[j] = *(const v2u*)(rr + 4 * q);
      if (hr)
#pragma unroll
        for (int e = 0; e < 4; e++)
          gv[j][e] = a.g_f32 ? ((const float*)a.g)[4 * q + e] : aq_bf2f(((const unsigned short*)a.g)[4 * q + e]);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int q = q0 + 256 * j;
      if (q >= ng) break;
      unsigned short sv[4];
#pragma unroll
      for (int e = 0; e < 4; e++)
        sv[e] = aq_add((unsigned short)(xv[j][e >> 1] >> (16 * (e & 1))), (unsigned short)(rv[j][e >> 1] >> (16 * (e & 1))));
      if (hr) {
        unsigned short hv[4];
#pragma unroll
        for (int e = 0; e < 4; e++) hv[e] = aq_f2bf(aq_bf2f(sv[e]) * sc * (gv[j][e] + a.wadd));
        *(v2u*)(hr + 4 * q) = aq_pack4(hv);
      }
      *(v2u*)(xo + 4 * q) = aq_pack4(sv);
    }
  }
}

// The decode band's residual arrival counters (version 5): EXL3I_AQ_XB_MAX_ROWS a (device, stream, xo), zero from module
// load, reset by each row's last workgroup. Keyed by xo too: the engine may run two launches at once only where their
// operands do not overlap (rad_kernel_concurrent), so two such launches write two xo.
#define EXL3I_AQ_RCTR_SLOTS 64
static __device__ unsigned exl3i_aq_rctr[EXL3I_AQ_RCTR_SLOTS * EXL3I_AQ_XB_MAX_ROWS];
static unsigned* exl3i_aq_rctr_for(void* stream, const void* key) {
  static std::mutex mu;
  static struct { int dev; void* s; const void* k; unsigned* p; } tab[1024];
  static int n = 0, used[64];
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) return nullptr;
  std::lock_guard<std::mutex> g(mu);
  for (int i = 0; i < n; i++)
    if (tab[i].dev == dev && tab[i].s == stream && tab[i].k == key) return tab[i].p;
  if (n == 1024 || used[dev] == EXL3I_AQ_RCTR_SLOTS) return nullptr;
  void* base = nullptr;
  if (hipGetSymbolAddress(&base, HIP_SYMBOL(exl3i_aq_rctr)) != hipSuccess || !base) return nullptr;
  tab[n] = {dev, stream, key, (unsigned*)base + (size_t)used[dev]++ * EXL3I_AQ_XB_MAX_ROWS};
  return tab[n++].p;
}

extern "C" int exl3i_aq_actq(const Exl3iAqArgs* a, void* stream) {
  if (!a || a->M < 1 || a->K < 128 || a->K % 128 || a->planes < 1 || a->planes > 4) return RAD_E_SHAPE;
  for (int i = 0; i < a->planes; i++)
    if (!a->s[i]) return RAD_E_INVAL;
  if (a->form != EXL3I_AQ_FORM_I8 && a->form != EXL3I_AQ_FORM_XB) return RAD_E_INVAL;
  if (a->out_ld < (a->form == EXL3I_AQ_FORM_XB ? exl3i_aq_xb_row_bytes(a->K, a->planes) : exl3i_aq_row_bytes(a->K, a->planes)) ||
      a->out_ld % (a->form == EXL3I_AQ_FORM_XB ? 256 : 4))
    return RAD_E_SHAPE;
  if (a->pre != EXL3I_AQ_PRE_NONE && !a->x_bf16) return RAD_E_DTYPE;
  if ((a->pre == EXL3I_AQ_PRE_RMSNORM && !a->g) || (a->pre == EXL3I_AQ_PRE_MUL && !a->x2)) return RAD_E_INVAL;
  if (a->res && (a->pre != EXL3I_AQ_PRE_RMSNORM || !a->xo)) return RAD_E_INVAL;
  const long long tasks = (long long)a->M * (a->K / 128);
  if ((tasks + 7) / 8 > 0x7FFFFFFFLL) return RAD_E_SHAPE;
  Exl3iAqArgs q = *a;
  q.rctr = nullptr;
  void* args[] = {&q};
  const void* fn;
  dim3 grid((unsigned)((tasks + 7) / 8));
  switch (a->pre) {
    case EXL3I_AQ_PRE_NONE: fn = (const void*)exl3i_aq_actq_kernel<EXL3I_AQ_PRE_NONE>; break;
    case EXL3I_AQ_PRE_SILU_MUL: fn = (const void*)exl3i_aq_actq_kernel<EXL3I_AQ_PRE_SILU_MUL>; break;
    case EXL3I_AQ_PRE_MUL: fn = (const void*)exl3i_aq_actq_kernel<EXL3I_AQ_PRE_MUL>; break;
    case EXL3I_AQ_PRE_RMSNORM:   // a row's blocks over several workgroups in the decode band (few rows)
      fn = (const void*)exl3i_aq_rmsnorm_kernel;
      grid = dim3((unsigned)a->M, a->M <= EXL3I_AQ_XB_MAX_ROWS ? (unsigned)((a->K / 128 + 7) / 8) : 1u);
      if (a->res && grid.y > 1) {   // no counters (or the test pin EXL3I_PIN_AQ_RES_SPLIT_OFF): one workgroup a row
        if (exl3i_pins[EXL3I_PIN_AQ_RES_SPLIT_OFF] || !(q.rctr = exl3i_aq_rctr_for(stream, a->xo))) grid.y = 1;
      }
      break;
    default: return RAD_E_INVAL;
  }
  return hipLaunchKernel(fn, grid, dim3(256), args, 0, (hipStream_t)stream) == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

// ============================================================================= wdec8
// One wave a (16-row tile, 64-k supertile) of the chunk, row tiles fastest (a block's eight waves read eight adjacent
// tiles of the k-major record): the four 16 x 16 tiles decoded by the rate policy straight to their int8 indices (mad's q)
// (decode8, the fragment's element order: exl3i_rate.hip.h's closed form),
// one dword exchanged with lane ^ 16 a tile (exl3i_a8.h exl3i_wdec8_send / _frag), and the lane's 32 bytes
// stored whole: a wave writes its 1 KiB supertile contiguously.
// One wave's (16-row tile rt of the chunk, 64-k supertile ks): wdec8's unit, also the fused GEMM's (below).
template <class R>
static __device__ __forceinline__ void wdec8_super(const Exl3iA8Args& a, const R& plan, int rt, int ks, int lane) {
  constexpr int NW = R::NW;
  const int WORDS = R::words(a.m);
  const int ntiles = a.N / 16, ksuper = a.K / 64;
  const gu8 codes = (gu8)(const unsigned char*)a.codes;
  const size_t tn = (size_t)a.c0 / 16 + rt;
  unsigned o[8];
#pragma unroll
  for (int t = 0; t < 4; t++) {
    const gu8 tb = codes + (((size_t)(4 * ks + t)) * ntiles + tn) * WORDS * 4;
    unsigned wd[2][NW];
#pragma unroll
    for (int r = 0; r < 2; r++)
#pragma unroll
      for (int x = 0; x < NW; x++) wd[r][x] = *(gu32)(tb + 4u * (unsigned)plan.woff[r][x]);
    unsigned d[2];
    plan.decode8(wd, d);
    const unsigned recv = (unsigned)__shfl_xor((int)exl3i_wdec8_send(lane, d[0], d[1]), 16, WAVE32);
    unsigned lo, hi;
    exl3i_wdec8_frag(lane, d[0], d[1], recv, &lo, &hi);
    o[2 * t] = lo;
    o[2 * t + 1] = hi;
  }
  unsigned char* const dst = (unsigned char*)a.wq + ((size_t)rt * ksuper + ks) * 1024 + lane * 32;
  *(v4u*)dst = (v4u){o[0], o[1], o[2], o[3]};
  *(v4u*)(dst + 16) = (v4u){o[4], o[5], o[6], o[7]};
}

template <class R>
__global__ void __launch_bounds__(256) exl3i_a8_wdec8_kernel(Exl3iA8Args a) {
  const int lane = (int)threadIdx.x % WAVE32;
  const int ntc = a.nc / 16, ksuper = a.K / 64;
  const size_t st = (size_t)blockIdx.x * 8 + threadIdx.x / WAVE32;
  if (st >= (size_t)ntc * ksuper) return;   // wave-uniform
  const int rt = (int)(st % ntc), ks = (int)(st / ntc);
  const R plan(lane, a.m);
  wdec8_super<R>(a, plan, rt, ks, lane);
}

template <class R>
static int wdec8_go(const Exl3iA8Args* a, void* stream) {
  const size_t st = (size_t)(a->nc / 16) * (a->K / 64);
  Exl3iA8Args q = *a;
  void* args[] = {&q};
  return hipLaunchKernel((const void*)exl3i_a8_wdec8_kernel<R>, dim3((unsigned)((st + 7) / 8)), dim3(256), args, 0,
                         (hipStream_t)stream) == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

// The rate's policy, as exl3i_launch_cb picks it (codebook mad).
template <int CB>
static int wdec8_cb(const Exl3iA8Args* a, void* stream) {
  const int m = a->m;
  if (m % 16 == 0) switch (m / 16) {
    case 1: return wdec8_go<RateInt<1, CB>>(a, stream);
    case 2: return wdec8_go<RateInt<2, CB>>(a, stream);
    case 3: return wdec8_go<RateInt<3, CB>>(a, stream);
    case 4: return wdec8_go<RateInt<4, CB>>(a, stream);
    case 5: return wdec8_go<RateInt<5, CB>>(a, stream);
    case 6: return wdec8_go<RateInt<6, CB>>(a, stream);
    case 7: return wdec8_go<RateInt<7, CB>>(a, stream);
    default: return wdec8_go<RateInt<8, CB>>(a, stream);
  }
  if (m < 16) return wdec8_go<RateFrac<CB, 1>>(a, stream);
  if (m < 80) return wdec8_go<RateFrac<CB, 0>>(a, stream);
  return wdec8_go<RateFrac<CB, 2>>(a, stream);
}

extern "C" int exl3i_a8_wdec8(const Exl3iA8Args* a, void* stream) {
  if (!a || a->K % 64 || a->N % 128 || a->c0 % 128 || a->nc % 128 || a->nc < 128 || a->c0 + a->nc > a->N)
    return RAD_E_SHAPE;
  const int m = a->m;
  if (m < 2 || m > 128 || m % 2) return RAD_E_DTYPE;
  if ((size_t)(a->nc / 16) * (a->K / 64) / 8 + 1 > 0x7FFFFFFFu) return RAD_E_SHAPE;
  if (a->cb != EXL3I_DEV_CB_MAD) return RAD_E_UNSUPPORTED;   /* the A8 plane is mad's */
  return wdec8_cb<EXL3I_DEV_CB_MAD>(a, stream);
}

// ============================================================================= the fused A8 GEMM (FUSE, mad)
// The GEMM (exl3i_8a8.hip.h, exl3i_a8_gemm.hip's tile) decodes its own weight: a block first claims 16-row strips of
// its 128-row weight tile bn (counters ctl[3 bn]: claimed, decoded, finished) and decodes each into the plane as
// wdec8 does (its eight waves over the strip's 64-k supertiles), until none is left; then it waits for the tile's
// eight strips, and runs the K loop. Blocks of one weight tile arrive together (the raster's activation-tile groups),
// so the decode is spread over them and overlapped with other tiles' K loops; later groups find the plane ready. A
// block never waits before claiming everything it can, so a strip being waited for is being decoded by a running
// block. The last of the tile's gridDim.x blocks resets its counters (zero for the next launch on the stream). The
// plane's bytes are wdec8's and the K loop is the tile's, so y is the separate passes' byte for byte.
// Every branch in the claim loop is wave-uniform: wave 0 claims and releases with all its lanes (lane 0 adds 1, the
// others 0; readfirstlane takes lane 0's old value) and st is read with readfirstlane. With `if (tid == 0)` there,
// AMD clang 23 split thread 0 off into an outer loop (its release, then its next claim, behind the loop's barriers)
// while the block's other threads looped through the barriers on the same stale s_v, and the launch hung; uniform
// branches leave the compiler no thread-0 path to split off.
template <class R>
__global__ __launch_bounds__(exl3i_8a8::kPWaves * exl3i_8a8::kWave) __attribute__((amdgpu_waves_per_eu(6, 6)))
void exl3i_a8_gemmd_kernel(exl3i_8a8::Tile p, Exl3iA8Args a, unsigned* ctl) {
  uint32_t bm, bn;
  exl3i_8a8::raster(bm, bn);
  unsigned* const c = ctl + 3u * bn;
  __shared__ unsigned s_v;
  const int tid = (int)threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32), ksuper = a.K / 64;
  const R plan(lane, a.m);
  for (;;) {
    if (wave == 0) {
      unsigned v = __builtin_amdgcn_readfirstlane(__hip_atomic_load(c + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
      if (v < 8u)
        v = __builtin_amdgcn_readfirstlane(
            __hip_atomic_fetch_add(c, lane == 0 ? 1u : 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
      s_v = v < 8u ? v : 8u;   // every lane the same value
    }
    __syncthreads();
    const unsigned st = __builtin_amdgcn_readfirstlane(s_v);
    __syncthreads();
    if (st >= 8u) break;
    const int rt = (int)(bn * 8u + st);
    for (int ks = wave; ks < ksuper; ks += 8) wdec8_super<R>(a, plan, rt, ks, lane);
    __threadfence();
    __syncthreads();
    if (wave == 0) __hip_atomic_fetch_add(c + 1, lane == 0 ? 1u : 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
  if (tid == 0)
    while (__hip_atomic_load(c + 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < 8u) __builtin_amdgcn_s_sleep(2);
  __syncthreads();
  __threadfence();   // every wave's acquire: the tile's plane as the decoders wrote it
  exl3i_8a8::exl3_8a8_body_at(p, bm, bn);
  __syncthreads();
  if (tid == 0 && __hip_atomic_fetch_add(c + 2, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == gridDim.x - 1u) {
    __hip_atomic_store(c, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(c + 1, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(c + 2, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}

// The fused GEMM's counters: a block of EXL3I_A8D_CTL a (device, stream), zero from module load (KS's scheme,
// exl3i_nt.hip.h).
#define EXL3I_A8D_SLOTS 16
#define EXL3I_A8D_CTL (3 * 4096)
static __device__ unsigned exl3i_a8d_ctl[EXL3I_A8D_SLOTS * EXL3I_A8D_CTL];
static unsigned* exl3i_a8d_ctl_for(void* stream) {
  static std::mutex mu;
  static struct { int dev; void* s; unsigned* p; } tab[64];
  static int n = 0, used[64];
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) return nullptr;
  std::lock_guard<std::mutex> g(mu);
  for (int i = 0; i < n; i++)
    if (tab[i].dev == dev && tab[i].s == stream) return tab[i].p;
  if (n == 64 || used[dev] == EXL3I_A8D_SLOTS) return nullptr;
  void* base = nullptr;
  if (hipGetSymbolAddress(&base, HIP_SYMBOL(exl3i_a8d_ctl)) != hipSuccess || !base) return nullptr;
  tab[n] = {dev, stream, (unsigned*)base + (size_t)used[dev]++ * EXL3I_A8D_CTL};
  return tab[n++].p;
}

template <class R>
static int gemmd_go(const Exl3iA8Args* a, unsigned* ctl, void* stream) {
  using namespace exl3i_8a8;
  Tile p;   // as exl3i_a8_gemm (exl3i_a8_gemm.hip) fills it; mad's step
  p.A = (const uint8_t*)a->aq;
  p.Sa = a->sa;
  p.W = (const uint8_t*)a->wq;
  p.svh = (const uint16_t*)a->svh;
  p.y = a->y;
  p.M = (uint32_t)a->M;
  p.K = (uint32_t)a->K;
  p.N = (uint32_t)a->N;
  p.c0 = (uint32_t)a->c0;
  p.mats = (uint32_t)a->mats;
  p.y_pitch = (uint32_t)a->y_pitch;
  p.y_dt = (uint32_t)a->y_dt;
  p.C = EXL3I_A8_WS * 0.08838834764831845f;
  // the activation planes: the arena's [mats][M][K] / [mats][M][K / 128], or AQ's packed rows (a_ld set)
  p.a_ld = a->a_ld ? (uint64_t)a->a_ld : (uint64_t)a->K;
  p.a_ms = a->a_ld ? (uint64_t)a->a_ms : (uint64_t)a->M * a->K;
  p.sa_ld = a->a_ld ? (uint64_t)a->sa_ld : (uint64_t)(a->K / 128);
  p.sa_ms = a->a_ld ? (uint64_t)a->sa_ms : (uint64_t)a->M * (a->K / 128);
  const dim3 grid((unsigned)((a->M + kPBM - 1) / kPBM), (unsigned)(a->nc / kPBN));
  if (grid.y > 65535u) return RAD_E_SHAPE;
  Exl3iA8Args q = *a;
  void* args[] = {&p, &q, &ctl};
  return hipLaunchKernel((const void*)exl3i_a8_gemmd_kernel<R>, grid, dim3(kPWaves * kWave), args, 0, (hipStream_t)stream) ==
                 hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

// wdec8 + GEMM of chunk [c0, c0 + nc) as one launch (mad); RAD_E_UNSUPPORTED when it does not apply (another
// codebook, no counter block, K > 8192 at M <= 512) or is pinned off (EXL3I_PIN_A8_FUSE_OFF, tests): the caller runs
// the two passes. The claim loop's branches are wave-uniform (above): a compiler that split thread 0's claim and
// release off into a loop of its own hung the first launch.
template <int CB>
static int gemmd_cb(const Exl3iA8Args* a, unsigned* ctl, void* stream);
extern "C" int exl3i_a8_gemmd(const Exl3iA8Args* a, void* stream) {
  if (exl3i_pins[EXL3I_PIN_A8_FUSE_OFF] || !a || a->cb != EXL3I_DEV_CB_MAD) return RAD_E_UNSUPPORTED;
  if (a->M < 1 || a->K % 128 || a->N % 128 || a->c0 % 128 || a->nc % 128 || a->nc < 128 || a->c0 + a->nc > a->N ||
      (a->mats == 2 && a->N % 256) || (a->y_pitch * (a->y_dt == 1 ? 4 : 2)) % 16 || a->y_dt < 0 || a->y_dt > 2)
    return RAD_E_SHAPE;
  if (a->nc / 128 > 4096) return RAD_E_UNSUPPORTED;
  // a long K at four activation tiles or fewer: each block decodes two or more long strips before its K loop and too
  // few other blocks hide it (down at M 512: x1.05 fused; from M 768 x0.92)
  if (a->K > 8192 && a->M <= 4 * (int)exl3i_8a8::kPBM) return RAD_E_UNSUPPORTED;
  unsigned* const ctl = exl3i_a8d_ctl_for(stream);
  if (!ctl) return RAD_E_UNSUPPORTED;
  return gemmd_cb<EXL3I_DEV_CB_MAD>(a, ctl, stream);
}

template <int CB>
static int gemmd_cb(const Exl3iA8Args* a, unsigned* ctl, void* stream) {
  const int m = a->m;
  if (m < 2 || m > 128 || m % 2) return RAD_E_DTYPE;
  if (m % 16 == 0) switch (m / 16) {
    case 1: return gemmd_go<RateInt<1, CB>>(a, ctl, stream);
    case 2: return gemmd_go<RateInt<2, CB>>(a, ctl, stream);
    case 3: return gemmd_go<RateInt<3, CB>>(a, ctl, stream);
    case 4: return gemmd_go<RateInt<4, CB>>(a, ctl, stream);
    case 5: return gemmd_go<RateInt<5, CB>>(a, ctl, stream);
    case 6: return gemmd_go<RateInt<6, CB>>(a, ctl, stream);
    case 7: return gemmd_go<RateInt<7, CB>>(a, ctl, stream);
    default: return gemmd_go<RateInt<8, CB>>(a, ctl, stream);
  }
  if (m < 16) return gemmd_go<RateFrac<CB, 1>>(a, ctl, stream);
  if (m < 80) return gemmd_go<RateFrac<CB, 0>>(a, ctl, stream);
  return gemmd_go<RateFrac<CB, 2>>(a, ctl, stream);
}
