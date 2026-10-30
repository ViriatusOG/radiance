/* Derived from libr4d/r4d_gdn_gated_norm_had_quant_i8.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_gdn_gated_norm_had_quant_i8.hip -- the GDN block's gated RMS norm, block Hadamard and
// per-row int8 quantiser in one pass: out_proj's input on the 48 linear-attention layers (the
// fourth rotated site in a W4A8 layer).
//
//     out = x * rsqrt(mean_head(x^2) + eps) * w * silu(z)
//
// Two extents, which is why halo_gdn_gated_rmsnorm_h128_bf16 cannot be reused: the RMS is per head
// (head_v_dim = 128), while the rotation and int8 scale are per token over the flattened H*128.
// The reduction is segmented: a head is 128/VPL consecutive lanes, and xor masks below that stay
// inside one head.
//
//     HAD=128  VPL=4   32 lanes/head   one head per rotation block
//     HAD=256  VPL=8   16 lanes/head   two heads per block
//     HAD=512  VPL=16   8 lanes/head   four heads per block
//
// At width 128 the rotation mixes only within a head; wider widths mix across heads. Activation
// error is 0.00679 at width 128 vs 0.00521 at 512.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_fwht.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>

#define HALO_GNHQ_HEAD    128

template <int HAD, int THR>
__global__ __launch_bounds__(THR) void halo_gdn_gated_norm_had_quant_i8_kernel(
    const __hip_bfloat16* __restrict__ X, const __hip_bfloat16* __restrict__ Z,
    const float* __restrict__ Wg, signed char* __restrict__ Q, float* __restrict__ S,
    int K, long xrow, long zrow, int zgrp, int zblk, float eps, float rnorm) {
  constexpr int VPL = HAD / 32, NW = THR / 32;
  constexpr int LPH = HALO_GNHQ_HEAD / VPL;          // lanes spanned by one head
  extern __shared__ float lds[];
  float* red = lds + K;
  const int row = blockIdx.x;
  const __hip_bfloat16* x = X + (size_t)row * xrow;
  const __hip_bfloat16* z = Z + (size_t)row * zrow;
  // zgrp != 0: Z is read in place from the interleaved qkvz projection (avoiding vLLM's
  // `z_out[:] = z` copy per layer): element i is at (i / zblk) * zgrp + (i % zblk). zblk is a
  // multiple of HAD, so a rotation block never straddles a group.
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;

  float m = 0.f;
  for (int blk = wave; blk < K / HAD; blk += NW) {
    const int base = blk * HAD + lane * VPL;
    const int zbase = zgrp ? (base / zblk) * zgrp + (base % zblk) : base;
    float v[VPL];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < VPL; j++) {
      v[j] = __bfloat162float(x[base + j]);
      ss += v[j] * v[j];
    }
    // segmented: masks below LPH never cross a head
    ss = halo_xor_sum<LPH>(ss);
    const float rms = __frsqrt_rn(ss * (1.f / (float)HALO_GNHQ_HEAD) + eps);
#pragma unroll
    for (int j = 0; j < VPL; j++) {
      const float g = __bfloat162float(z[zbase + j]);
      v[j] = v[j] * rms * Wg[(base + j) & (HALO_GNHQ_HEAD - 1)] * (g / (1.f + __expf(-g)));
    }
    halo_fwht_wave<HAD>(v, lane);
#pragma unroll
    for (int j = 0; j < VPL; j++) {
      lds[base + j] = v[j];
      m = fmaxf(m, fabsf(v[j]));
    }
  }

  m = halo_wave_max(m);
  if (lane == 0) red[wave] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < NW; i++) m = fmaxf(m, red[i]);
  m = fmaxf(m, 1e-8f);
  const float inv = 127.f / m;
  if (threadIdx.x == 0) S[row] = m * rnorm / 127.f;

  signed char* q = Q + (size_t)row * K;
  for (int i = threadIdx.x; i < K; i += THR) {
    const int st = i >> 4, j = i & 15;
    const int sr = (st << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
    const int t = __float2int_rn(lds[sr] * inv);
    q[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
  }
}

// The plugin ABI launch (RadLaunchFn); see the note at the head of halo_quant_act_i8.hip.
enum { GG_X = 0, GG_Z, GG_W, GG_Q, GG_S };

extern "C" int halo_rad_gdn_gated_norm_hq_i8(const RadArgs* args, RadStream stream) {
  const RadTensor* x  = halo_opt(args, GG_X);
  const RadTensor* z  = halo_opt(args, GG_Z);
  const RadTensor* w  = halo_opt(args, GG_W);
  const RadTensor* q  = halo_opt(args, GG_Q);
  const RadTensor* sc = halo_opt(args, GG_S);
  if (!x || !z || !w || !q || !sc) return HALO_NO(RAD_E_INVAL);
  if (!halo_contig(q) || !halo_contig(sc) || !halo_contig(w)) return HALO_NO(RAD_E_STRIDE);

  long long M = 0, K = 0;
  int rc = halo_rows_of(x, &M, &K);
  if (rc != RAD_OK) return HALO_NO(rc);
  const long long head = halo_geti_or(args, "head_dim", HALO_GNHQ_HEAD);
  if (head != HALO_GNHQ_HEAD) return HALO_NO(RAD_E_SHAPE);
  if (K % HALO_GNHQ_HEAD) return HALO_NO(RAD_E_SHAPE);
  if (halo_numel(w) != head) return HALO_NO(RAD_E_SHAPE);   // wgain is 128 floats, not K

  const long long had = halo_geti_or(args, "group", 0);
  if (had != 128 && had != 256 && had != 512) return HALO_NO(RAD_E_SHAPE);
  // A rotation block cannot straddle a tensor-parallel shard boundary.
  if (K % had) return HALO_NO(RAD_E_SHAPE);

  const long long xrow = halo_row_pitch(x);
  const long long zrow = halo_row_pitch(z);
  if (xrow < 0 || zrow < 0) return HALO_NO(RAD_E_STRIDE);

  // Interleaved-gate layout (see the kernel); zgrp 0 means contiguous.
  const long long zgrp = halo_geti_or(args, "zgrp", 0);
  const long long zblk = halo_geti_or(args, "zblk", 0);
  if (zgrp && (zblk % had || zblk < had || zgrp < zblk)) return HALO_NO(RAD_E_INVAL);

  const float eps = (float)halo_getf(args, "eps", 1e-6);
  const int thr = halo_act_threads((int)M, (int)K);
  const size_t shmem = (size_t)K * sizeof(float) + (thr / 32) * sizeof(float);
  if (shmem > 65536) return HALO_NO(RAD_E_SCRATCH);
  const float rn = 1.f / sqrtf((float)had);
  const long xr = (long)(xrow ? xrow : K), zr = (long)(zrow ? zrow : K);
#define HALO_GNHQ_LAUNCH(H, T) hipLaunchKernelGGL((halo_gdn_gated_norm_had_quant_i8_kernel<H, T>), \
    dim3((int)M), dim3(T), shmem, (hipStream_t)stream, (const __hip_bfloat16*)halo_p(x), \
    (const __hip_bfloat16*)halo_p(z), (const float*)halo_p(w), (signed char*)halo_wp(q), \
    (float*)halo_wp(sc), (int)K, xr, zr, (int)zgrp, (int)zblk, eps, rn)
#define HALO_GNHQ_T(H) switch (thr) { \
    case 256: HALO_GNHQ_LAUNCH(H, 256); break; \
    case 512: HALO_GNHQ_LAUNCH(H, 512); break; \
    case 1024: HALO_GNHQ_LAUNCH(H, 1024); break; \
    default: return HALO_NO(RAD_E_INVAL); }
  switch (had) {
    case 128: HALO_GNHQ_T(128); break;
    case 256: HALO_GNHQ_T(256); break;
    default:  HALO_GNHQ_T(512); break;
  }
#undef HALO_GNHQ_T
#undef HALO_GNHQ_LAUNCH
  return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}
