/* Derived from libr4d/r4d_gated_had_quant_i8.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_gated_had_quant_i8.hip - halo_gated_had_quant_i8: an elementwise gated product, then the block
// Hadamard, then the per-row int8 quantiser, in one pass.
//
// Two of the four rotated sites have this shape and differ by one multiply:
//
//   mode 0, down_proj's input      silu(a) * b  =  a * sigmoid(a) * b     (vLLM's silu_and_mul)
//   mode 1, o_proj's input         sigmoid(a) * b                        (Qwen3_5Attention does
//                                                                         attn_output * sigmoid(gate))
//
// so one kernel serves the MLP of all 64 layers and the attention output of the 16 full-attention
// ones. The 48 linear-attention layers reach out_proj through a gated RMSNorm instead.
//
// Mode 0 is the widest site and sets the rotation width. An intermediate of 17408 at TP=2 leaves
// 8704 = 2^9 * 17 per rank, so 512 is the largest power of two that tiles it (a rotation block
// cannot straddle a shard boundary). Between 128 and 512 the activation's quantisation error moves
// far less than the dominant weight error, so the width is chosen on cost.
//
// Gate and up arrive as two tensors, or as two halves of vLLM's fused gate_up with a row pitch of
// 2*K (`up` aliasing `gate + K`); the kernel takes pointers and strides either way.
//
// The transform is in registers (halo_fwht.h), so this is one pass: read gate and up, multiply,
// rotate inside the wave, stage once for the row absmax.
//
// silu is computed in fp32: the product feeds a rotation that mixes 128 values, so a bf16 rounding
// error in one lands on all of them.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_fwht.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>

template <int HAD, int MODE, int THR>
__global__ __launch_bounds__(THR) void halo_gated_had_quant_i8_kernel(
    const __hip_bfloat16* __restrict__ G, const __hip_bfloat16* __restrict__ U,
    signed char* __restrict__ Q, float* __restrict__ S,
    int K, long gstride, long ustride, float rnorm) {
  constexpr int VPL = HAD / 32, NW = THR / 32;
  extern __shared__ float lds[];
  float* red = lds + K;
  const int row = blockIdx.x;
  const __hip_bfloat16* g = G + (size_t)row * gstride;
  const __hip_bfloat16* u = U + (size_t)row * ustride;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;

  float m = 0.f;
  for (int blk = wave; blk < K / HAD; blk += NW) {
    const int base = blk * HAD + lane * VPL;
    float v[VPL];
#pragma unroll
    for (int j = 0; j < VPL; j++) {
      const float xg = __bfloat162float(g[base + j]);
      const float sg = 1.f / (1.f + __expf(-xg));
      v[j] = (MODE == 0 ? xg * sg : sg) * __bfloat162float(u[base + j]);
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
    const int s = i >> 4, j = i & 15;
    const int sr = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
    const int t = __float2int_rn(lds[sr] * inv);
    q[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
  }
}

// The launch, as the plugin ABI's RadLaunchFn. See the note at the head of halo_quant_act_i8.hip.
//
// A missing second operand is the packed gate_up form: one [M, 2n] tensor with `b` = `a + n`
// (vLLM's layout and what the MLP produces), so both row pitches are 2n. The row's dtype
// constraint guarantees bf16.
enum { GH_A = 0, GH_B, GH_Q, GH_S };

extern "C" int halo_rad_gated_had_quant_i8(const RadArgs* args, RadStream stream) {
  const RadTensor* ta = halo_opt(args, GH_A);
  const RadTensor* tb = halo_opt(args, GH_B);
  const RadTensor* q  = halo_opt(args, GH_Q);
  const RadTensor* sc = halo_opt(args, GH_S);
  if (!ta || !q || !sc) return HALO_NO(RAD_E_INVAL);
  if (!halo_contig(q) || !halo_contig(sc)) return HALO_NO(RAD_E_STRIDE);

  const long long K = halo_geti_or(args, "n", 0);   // the GATED width, half of a packed row
  if (K < 1) return HALO_NO(RAD_E_INVAL);

  long long M = 0, width = 0;
  int rc = halo_rows_of(ta, &M, &width);
  if (rc != RAD_OK) return HALO_NO(rc);

  const __hip_bfloat16* pa = (const __hip_bfloat16*)halo_p(ta);
  const __hip_bfloat16* pb = nullptr;
  long long a_stride = 0, b_stride = 0;

  if (tb) {
    if (width != K) return HALO_NO(RAD_E_SHAPE);
    a_stride = halo_row_pitch(ta);
    b_stride = halo_row_pitch(tb);
    if (a_stride < 0 || b_stride < 0) return HALO_NO(RAD_E_STRIDE);
    pb = (const __hip_bfloat16*)halo_p(tb);
  } else {
    if (width != 2 * K) return HALO_NO(RAD_E_SHAPE);
    a_stride = halo_row_pitch(ta);
    if (a_stride < 0) return HALO_NO(RAD_E_STRIDE);
    b_stride = a_stride;
    pb = pa + K;
  }

  const long long had = halo_geti_or(args, "group", 0);
  if (had != 128 && had != 256 && had != 512) return HALO_NO(RAD_E_SHAPE);
  // A rotation block cannot straddle a tensor-parallel shard; down_proj's input is 8704 per rank,
  // so 512 is the widest available.
  if (K % had) return HALO_NO(RAD_E_SHAPE);
  if (K % 16) return HALO_NO(RAD_E_SHAPE);

  // `act` names the activation ("silu" or "sigmoid"); `mode` is this kernel's integer for it.
  const int mode = halo_act_code(args, "act", "mode", 0);
  if (mode != 0 && mode != 1) return HALO_NO(RAD_E_INVAL);
  if (halo_numel(q) != M * K) return HALO_NO(RAD_E_SHAPE);

  const int thr = halo_act_threads((int)M, (int)K);
  const size_t shmem = (size_t)K * sizeof(float) + (thr / 32) * sizeof(float);
  if (shmem > 65536) return HALO_NO(RAD_E_SCRATCH);
  const long gs = (long)(a_stride ? a_stride : K), us = (long)(b_stride ? b_stride : K);
  const float rn = 1.f / sqrtf((float)had);
#define HALO_GHQ_LAUNCH(H, MD, T) hipLaunchKernelGGL((halo_gated_had_quant_i8_kernel<H, MD, T>), \
    dim3((int)M), dim3(T), shmem, (hipStream_t)stream, pa, pb, (signed char*)halo_wp(q), \
    (float*)halo_wp(sc), (int)K, gs, us, rn)
#define HALO_GHQ_T(H, MD) switch (thr) { \
    case 256: HALO_GHQ_LAUNCH(H, MD, 256); break; \
    case 512: HALO_GHQ_LAUNCH(H, MD, 512); break; \
    case 1024: HALO_GHQ_LAUNCH(H, MD, 1024); break; \
    default: return HALO_NO(RAD_E_INVAL); }
#define HALO_GHQ_MODE(MD) switch (had) { \
    case 128: HALO_GHQ_T(128, MD); break; \
    case 256: HALO_GHQ_T(256, MD); break; \
    default:  HALO_GHQ_T(512, MD); break; }
  if (mode == 0) { HALO_GHQ_MODE(0); } else { HALO_GHQ_MODE(1); }
#undef HALO_GHQ_MODE
#undef HALO_GHQ_T
#undef HALO_GHQ_LAUNCH
  return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}
