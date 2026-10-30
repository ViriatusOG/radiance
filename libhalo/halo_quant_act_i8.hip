/* Derived from libr4d/r4d_quant_act_i8.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_quant_act_i8.hip - halo_quant_act_i8: per-row symmetric int8 quantisation of a bf16 activation,
// written in the order halo_gemm_w4a8_nt_m64's A fragment wants.
//
// One row per workgroup: absmax over K, then a second pass writing the bytes.
//
// Byte order: fragment element e of lane l is A[row][16s + 8*(e>>2) + 4*(l>>4) + (e&3)], two dwords
// eight bytes apart in a row-major activation. Each k step's sixteen bytes are written as
// [half 0's eight][half 1's eight], so the GEMM reads a fragment with one global_load_b64. Rows are
// not permuted; splitting A into two planes is slower (see the note in the GEMM).
#include "halo.h"
#include "halo_args.h"
#include "halo_emit.h"
#include "halo_common.h"
#include "halo_chain.h"
#include <hip/hip_bf16.h>
#include <cstdint>

/* CHAIN: the chained form's wait and add (halo_chain.h) around the same body; no block returns
 * early. */
template <int THR, bool CHAIN = false>
__device__ __forceinline__ void halo_quant_act_i8_body(
    const __hip_bfloat16* __restrict__ A, signed char* __restrict__ Q, float* __restrict__ S,
    int K) {
  if (CHAIN) halo_chain_wait();
  const int row = blockIdx.x;
  const __hip_bfloat16* a = A + (size_t)row * K;
  signed char* q = Q + (size_t)row * K;

  float m = 0.f;
  for (int i = threadIdx.x; i < K; i += THR) {
    const float v = __bfloat162float(a[i]);
    m = fmaxf(m, fabsf(v));
  }
  __shared__ float red[THR / 32];
  m = halo_wave_max(m);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < THR / 32; i++) m = fmaxf(m, red[i]);
  const float scale = fmaxf(m, 1e-8f) / 127.f;
  const float inv = 1.f / scale;
  if (threadIdx.x == 0) S[row] = scale;

  // Byte j of a k step comes from lane-half j>>3 and element j&7, i.e. source 8*((j&7)>>2) +
  // 4*(j>>3) + (j&3) inside the step.
  for (int i = threadIdx.x; i < K; i += THR) {
    const int s = i >> 4, j = i & 15;
    const int src = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
    const float v = __bfloat162float(a[src]) * inv;
    const int t = __float2int_rn(v);
    q[i] = (signed char)(t < -128 ? -128 : (t > 127 ? 127 : t));
  }
  if (CHAIN) halo_chain_done();
}

template <int THR>
__global__ __launch_bounds__(THR) void halo_quant_act_i8_kernel(
    const __hip_bfloat16* __restrict__ A, signed char* __restrict__ Q, float* __restrict__ S,
    int K) {
  halo_quant_act_i8_body<THR, false>(A, Q, S, K);
}
template <int THR>
__global__ __launch_bounds__(THR) void halo_r4dchain_quant_act_i8(
    const __hip_bfloat16* __restrict__ A, signed char* __restrict__ Q, float* __restrict__ S,
    int K) {
  halo_quant_act_i8_body<THR, true>(A, Q, S, K);
}

// ---------------------------------------------------------------------------------------------
// Same quantisation, plus the per-row per-group sum of the int8 codes for an asymmetric weight
// grid: sum_k a_k (nibble_k - z_g) = sum_k a_k nibble_k - z_g * sum_k a_k, and the second term is
// computed once here instead of N times in the GEMM. Output is [K/group][Mpad], group-major with
// Mpad = ceil(M/16)*16, because a lane's eight accumulator elements are eight contiguous rows at
// one group, read as two aligned v4f. The caller zeroes it; pad rows are never written.
//
// The fragment permutation stays within a 16-element step and a group is whole steps, so group
// membership is unchanged.
#define HALO_QAI8_MAXG 256

template <int THR>
__global__ __launch_bounds__(THR) void halo_quant_act_i8_asum_kernel(
    const __hip_bfloat16* __restrict__ A, signed char* __restrict__ Q, float* __restrict__ S,
    float* __restrict__ Asum, int K, int group, int Mpad) {
  const int row = blockIdx.x;
  const __hip_bfloat16* a = A + (size_t)row * K;
  signed char* q = Q + (size_t)row * K;
  const int G = K / group;

  __shared__ int gs[HALO_QAI8_MAXG];
  for (int g = threadIdx.x; g < G; g += THR) gs[g] = 0;

  float m = 0.f;
  for (int i = threadIdx.x; i < K; i += THR) {
    const float v = __bfloat162float(a[i]);
    m = fmaxf(m, fabsf(v));
  }
  __shared__ float red[THR / 32];
  m = halo_wave_max(m);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  m = red[0];
#pragma unroll
  for (int i = 1; i < THR / 32; i++) m = fmaxf(m, red[i]);
  const float scale = fmaxf(m, 1e-8f) / 127.f;
  const float inv = 1.f / scale;
  if (threadIdx.x == 0) S[row] = scale;

  for (int i = threadIdx.x; i < K; i += THR) {
    const int st = i >> 4, j = i & 15;
    const int src = (st << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
    const int t = __float2int_rn(__bfloat162float(a[src]) * inv);
    const int c = t < -128 ? -128 : (t > 127 ? 127 : t);
    q[i] = (signed char)c;
    atomicAdd(&gs[src / group], c);
  }
  __syncthreads();
  for (int g = threadIdx.x; g < G; g += THR)
    Asum[(size_t)g * Mpad + row] = (float)gs[g];
}

// The plugin ABI entry point (RadLaunchFn); a refused shape is a status code, not a throw. See
// libhalo/halo_args.h.
//
// The optional fourth operand (asum) selects the form: the row sums are a property of the call,
// so this is one row with a branch rather than two rows a constraint cannot tell apart.
enum { QA_X = 0, QA_Q, QA_S, QA_ASUM };

extern "C" int halo_rad_quant_act_i8(const RadArgs* args, RadStream stream) {
  int emit_rc = RAD_OK;
  const RadTensor* x  = halo_opt(args, QA_X);
  const RadTensor* q  = halo_opt(args, QA_Q);
  const RadTensor* sc = halo_opt(args, QA_S);
  const RadTensor* as = halo_opt(args, QA_ASUM);
  if (!x || !q || !sc) return HALO_NO(RAD_E_INVAL);
  if (!halo_contig(x) || !halo_contig(q) || !halo_contig(sc)) return HALO_NO(RAD_E_STRIDE);

  long long M = 0, K = 0;
  const int rc = halo_rows_of(x, &M, &K);
  if (rc != RAD_OK) return HALO_NO(rc);
  if (K % 16) return HALO_NO(RAD_E_SHAPE);          // a k step is sixteen bytes of one row
  if (halo_numel(q) != M * K) return HALO_NO(RAD_E_SHAPE);

  const int thr = halo_act_threads((int)M, (int)K);   // the workgroup width, from M

  if (!as) {
    /* The chained form where the row resolved with `chain`, at decode heights. */
    const bool chain = halo_geti_or(args, "chain", 0) != 0 && M <= 64;
#define HALO_QAI8_LAUNCH(T) emit_rc = chain                                                       \
    ? HALO_EMIT(stream, (halo_r4dchain_quant_act_i8<T>), dim3((int)M), dim3(T), 0,                     \
               (const __hip_bfloat16*)halo_p(x), (signed char*)halo_wp(q), (float*)halo_wp(sc),       \
               (int)K)                                                                           \
    : HALO_EMIT(stream, (halo_quant_act_i8_kernel<T>), dim3((int)M), dim3(T), 0,                   \
               (const __hip_bfloat16*)halo_p(x), (signed char*)halo_wp(q), (float*)halo_wp(sc),       \
               (int)K)
    switch (thr) {
      case 128:  HALO_QAI8_LAUNCH(128);  break;
      case 256:  HALO_QAI8_LAUNCH(256);  break;
      case 512:  HALO_QAI8_LAUNCH(512);  break;
      case 1024: HALO_QAI8_LAUNCH(1024); break;
      default:   return HALO_NO(RAD_E_INVAL);       // halo_act_threads answers only these four
    }
#undef HALO_QAI8_LAUNCH
    return emit_rc;
  }

  if (!halo_contig(as)) return HALO_NO(RAD_E_STRIDE);
  const long long group = halo_geti_or(args, "group", HALO_W4_GROUP);
  if (group < 1 || K % group) return HALO_NO(RAD_E_SHAPE);
  if (K / group > HALO_QAI8_MAXG) return HALO_NO(RAD_E_SHAPE);   // the LDS accumulator's bound
  // asum is [K/group][Mpad], Mpad = ceil(M/16)*16, because the asymmetric GEMM reads whole
  // 16-row tiles. Sizing it from M alone overruns.
  const long long mpad = ((M + 15) / 16) * 16;
  if (halo_numel(as) < (K / group) * mpad) return HALO_NO(RAD_E_SCRATCH);

#define HALO_QAI8A_LAUNCH(T) emit_rc = HALO_EMIT(stream, (halo_quant_act_i8_asum_kernel<T>),        \
    dim3((int)M), dim3(T), 0, (const __hip_bfloat16*)halo_p(x), (signed char*)halo_wp(q),          \
    (float*)halo_wp(sc), (float*)halo_wp(as), (int)K, (int)group, (int)mpad)
  switch (thr) {
    case 128:  HALO_QAI8A_LAUNCH(128);  break;
    case 256:  HALO_QAI8A_LAUNCH(256);  break;
    case 512:  HALO_QAI8A_LAUNCH(512);  break;
    case 1024: HALO_QAI8A_LAUNCH(1024); break;
    default:   return HALO_NO(RAD_E_INVAL);
  }
#undef HALO_QAI8A_LAUNCH
  return emit_rc;
}
