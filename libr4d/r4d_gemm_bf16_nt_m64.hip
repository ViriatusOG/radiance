// r4d_gemm_bf16_nt_m64.hip - r4d_gemm_bf16_nt_m64: skinny bf16 GEMM at any M, gfx1201.
//
// C[M,N] = A[M,K] @ W[N,K]^T, bf16 in and out, fp32 accumulate.
//
// General in M: MT row tiles per block, bounded grids with rebased row launches. The 64 is
// the registry band, not a kernel limit; the M <= 16 row forwards here too.
//
// A scalar kernel (one wave per output column, M accumulators) costs one weight load and M
// activation loads per 16 K, and cannot run above M=16. One wmma_f32_16x16x16 produces a whole
// 16x16 tile, so a 16-K step costs ceil(M/16) activation fragments and one weight fragment, with
// 8 VGPRs per lane of accumulator per tile.
//
// A block is WV column tiles x SK k-splits waves; each wave owns one 16-wide N tile over one
// contiguous K slice. Splits are reduced through LDS in wave order, so the result does not depend
// on scheduling. SK fills the machine when N is small (N=48 is three tiles).
//
// Fragment rows are clamped, not masked: rows past M or N re-read the last valid row and their
// results are never stored.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_bf16_gemm_rows.h"
#include "r4d_gdn_wmma.h"
#include "r4d_chain.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#define WAVE32 32

// A fragment from a base pointer plus a compile-time k step. With fragGo's 32-bit element offset,
// LLVM re-derives a 64-bit address per fragment per step (a b64 shift and a carry-in add pair); a
// constant step instead folds into the 13-bit immediate of global_load_b64 (32 bytes per k step,
// so eight fit).
//
// The pointer is explicitly global: a plain pointer held in an array across iterations becomes
// generic, so loads become flat_load_b64, which on gfx12 also increments dscnt and makes later LDS
// waits wait on HBM. The element is a scalar because HIP's vector structs cannot be assigned from
// an address-space-qualified source.
typedef __attribute__((address_space(1))) const unsigned long long* r4d_gu2;

__device__ __forceinline__ v8bf fragU(r4d_gu2 p, int u) {
  union { unsigned long long w[2]; v8bf f; } x;
  x.w[0] = p[4 * u];
  x.w[1] = p[4 * u + 2];
  return x.f;
}

// MT = row tiles per block, held as MT v8f accumulators (8 VGPRs each); grid.y covers the rest.
// More amortises each weight fragment, fewer puts more blocks on the machine, so MT is tunable.
// KU is the k-step unroll; see the loop.
constexpr int KU = 8;
// CHAIN brackets the body with the chained launch's wait and add (r4d_chain.h); no block returns
// early. r4d_gemm_bf16_nt_m64_kernel and r4dchain_gemm_bf16_nt_m64 below wrap it.
template <int MT, bool CHAIN, typename CT = __hip_bfloat16>
__device__ __forceinline__ void r4d_gemm_bf16_nt_m64_body(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    CT* __restrict__ C, int M, int K, int N, int WV, int SK, int LDA, int LDC) {
  if (CHAIN) r4d_chain_wait_vec();
  extern __shared__ float red[];              // WV*SK*256 floats: one 16x16 tile per (column, split)
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt = blockIdx.x * WV + col_local;
  const int n0 = nt * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK, ksteps = Ks / 16, kbase = ksplit * Ks;

  v8f acc[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) acc[i] = (v8f)(0.f);

  // Row offsets as 32-bit element offsets, turned into pointers once below; as pointers, LLVM
  // would not keep the 64-bit stride math and the row clamp out of the loop.
  unsigned woff = rowGo((unsigned)K, n0, N - 1, lane) + (unsigned)kbase;
  unsigned aoff[MT];
#pragma unroll
  for (int i = 0; i < MT; i++)
    aoff[i] = rowGo((unsigned)LDA, (mt0 + i) * 16, M - 1, lane) + (unsigned)kbase;

  // Unrolled so the k step is a load immediate (the 13-bit field of global_load_b64) and the base
  // pointers advance once per KU steps; rolled, LLVM rebuilds a 64-bit address every step.
  if (n0 < N) {
    r4d_gu2 pw = (r4d_gu2)(const unsigned long long*)(W + woff);
    r4d_gu2 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) pa[i] = (r4d_gu2)(const unsigned long long*)(A + aoff[i]);
    int s = 0;
    for (; s + KU <= ksteps; s += KU) {
#pragma unroll
      for (int u = 0; u < KU; u++) {
        const v8bf b = fragU(pw, u);
#pragma unroll
        for (int i = 0; i < MT; i++) acc[i] = mma(fragU(pa[i], u), b, acc[i]);
      }
      pw += KU * 4;
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += KU * 4;
    }
    for (; s < ksteps; s++) {
      const v8bf b = fragU(pw, 0);
      pw += 4;
#pragma unroll
      for (int i = 0; i < MT; i++) {
        acc[i] = mma(fragU(pa[i], 0), b, acc[i]);
        pa[i] += 4;
      }
    }
  }

  // Reduce one row tile at a time, so LDS is WV*SK KiB regardless of MT. Every thread in the block
  // shares the sum (one element per thread at SK=8); the order over splits is fixed.
  const int slot = (col_local * SK + ksplit) * 256;
  const int nthreads = WV * SK * WAVE32;
#pragma unroll
  for (int i = 0; i < MT; i++) {
    __syncthreads();
#pragma unroll
    for (int e = 0; e < 8; e++) red[slot + e * WAVE32 + lane] = acc[i][e];   // conflict-free
    __syncthreads();
    for (int idx = tid; idx < WV * 256; idx += nthreads) {
      const int col = idx >> 8, off = idx & 255;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      const int l = off & 31;
      const int n = (blockIdx.x * WV + col) * 16 + (l & 15);
      const int m = (mt0 + i) * 16 + 8 * (l >> 4) + (off >> 5);
      // The language's float-to-bf16 conversion on purpose; the header's f2bf rounds half away
      // from zero.
      if (n < N && m < M) {
        if constexpr (std::is_same<CT, float>::value) C[(size_t)m * LDC + n] = v;   // f32 logits
        else C[(size_t)m * LDC + n] = (__hip_bfloat16)v;
      }
    }
  }
  if (CHAIN) r4d_chain_done();
}

template <int MT>
__global__ __launch_bounds__(1024) void r4d_gemm_bf16_nt_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK, int LDA, int LDC) {
  r4d_gemm_bf16_nt_m64_body<MT, false>(A, W, C, M, K, N, WV, SK, LDA, LDC);
}
template <int MT>
__global__ __launch_bounds__(1024) void r4dchain_gemm_bf16_nt_m64(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK, int LDA, int LDC) {
  r4d_gemm_bf16_nt_m64_body<MT, true>(A, W, C, M, K, N, WV, SK, LDA, LDC);
}

// grid = ceil(ceil(N/16)/WV) x ceil(ceil(M/16)/MB) blocks; block = WV*SK waves. MT = MB is
// compile-time so the accumulators are sized exactly.
static void gemm_bf16_m64_launch(long a, long w, long c, int M, int K, int N, int WV, int SK,
                                 int MB, long stream, bool chain, long long lda, long long ldc);
void r4d_gemm_bf16_nt_m64(long a, long w, long c, int M, int K, int N, int WV, int SK, int MB,
                          long stream) {
  gemm_bf16_m64_launch(a, w, c, M, K, N, WV, SK, MB, stream, false, K, N);
}
/* `chain`: the chained form (r4d_rad_gemm_bf16_m64 passes the row's parameter; M <= 64 always). */
/* `lda` / `ldc`: the rows' pitches in elements (K and N for contiguous planes). A's fragments are
 * 8-byte loads, so its pitch is a multiple of 4; its offsets are 32-bit from each launch's base,
 * so a launch takes at most 2^32 / lda rows (r4d_bf16_gemm_chunk_rows_lda). */
static void gemm_bf16_m64_launch(long a, long w, long c, int M, int K, int N, int WV, int SK,
                                 int MB, long stream, bool chain, long long lda, long long ldc) {
  if (M < 1) throw std::runtime_error("r4d gemm_bf16_nt_m64: M out of range");
  if (lda < K || ldc < N || lda % 4 || ldc > INT32_MAX)
    throw std::runtime_error("r4d gemm_bf16_nt_m64: row pitch unsupported");
  if (WV < 1 || SK < 1 || (int64_t)WV * SK > 1024 / WAVE32)
    throw std::runtime_error("r4d gemm_bf16_nt_m64: invalid threads/block");
  if (K % ((int64_t)SK * 16))
    throw std::runtime_error("r4d gemm_bf16_nt_m64: K must be divisible by SK*16");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_bf16_nt_m64: MB must be 1..4");
  const int chunk = r4d_bf16_gemm_chunk_rows_lda(M, N, K, lda, MB * 16, WV * 16);
  if (!chunk) throw std::runtime_error("r4d gemm_bf16_nt_m64: invalid dimensions or weight span");
  // A chained op must publish one completion, not one per rebased launch. Existing chained
  // decode shapes (M <= 64) fit one launch; do not silently change their synchronization law.
  if (chain && chunk < M)
    throw std::runtime_error("r4d gemm_bf16_nt_m64: chained rows need one launch");
  const int ntiles = (N + 15) / 16;
  dim3 block(WV * SK * WAVE32);
  size_t smem = (size_t)WV * SK * 256 * sizeof(float);
  auto Wp = (const unsigned short*)w;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MT)                                                                                 \
  hipLaunchKernelGGL((chain ? r4dchain_gemm_bf16_nt_m64<MT> : r4d_gemm_bf16_nt_m64_kernel<MT>),    \
                     grid, block, smem, st, Ap, Wp, Cp, mm, K, N, WV, SK, (int)lda, (int)ldc)
  for (int64_t m0 = 0; m0 < M; m0 += chunk) {
    const int mm = (int)std::min<int64_t>(chunk, M - m0), mtiles = (mm + 15) / 16;
    dim3 grid((ntiles + WV - 1) / WV, (mtiles + MB - 1) / MB);
    auto Ap = (const unsigned short*)a + m0 * lda;
    auto Cp = (__hip_bfloat16*)c + m0 * ldc;
    switch (MB) {
    case 1:
      LAUNCH(1);
      break;
    case 2:
      LAUNCH(2);
      break;
    case 3:
      LAUNCH(3);
      break;
    case 4:
      LAUNCH(4);
      break;
    default:
      throw std::runtime_error("r4d gemm_bf16_nt_m64: MB unsupported");
    }
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
  }
#undef LAUNCH
}


// ---- f32-output form for the lm_head at verify/draft row counts (17 <= M <= 64) -----------------
// logits_gemm_bf16 runs one workgroup a vocabulary column at any M (r4d_model_bf16.hip), which at
// M = 56..64 reads the head once per row group and runs far from its bandwidth bound. This is the
// same WMMA body with an f32 store: a wave owns a 16-column tile, the weight fragment is reused over
// the MT row tiles, SK = 1 so there is no split reduction. The summation order differs from the
// scalar kernel (not byte-identical), so the row is off by default (see R4D_PRIO_LOGITS_WMMA).
template <int MT>
__global__ __launch_bounds__(1024) void r4d_logits_wmma_kernel(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    float* __restrict__ C, int M, int K, int N, int WV, int SK) {
  r4d_gemm_bf16_nt_m64_body<MT, false, float>(A, W, C, M, K, N, WV, SK, K, N);
}
extern "C" int r4d_logits_gemm_bf16_wmma(long a, long w, long c, int M, int K, int N, int a_ld,
                                         int c_ld, long stream) {
  if (!a || !w || !c) return -1;
  if (M < 17 || M > 64 || K <= 0 || N <= 0) return -2;
  if (K % 16) return -3;
  if ((long)N * K > 0xFFFFFFFFL) return -3;      // the weight offset is 32-bit unsigned
  if ((a_ld && a_ld != K) || (c_ld && c_ld != N)) return -3;   // dense rows only
  const int WV = 8, SK = 1, MT = (M + 15) / 16;
  const int ntiles = (N + 15) / 16;
  dim3 grid((ntiles + WV - 1) / WV, 1), block(WV * SK * WAVE32);
  const size_t smem = (size_t)WV * SK * 256 * sizeof(float);
  auto Ap = (const unsigned short*)a;
  auto Wp = (const unsigned short*)w;
  auto Cp = (float*)c;
  hipStream_t st = (hipStream_t)stream;
  switch (MT) {
  case 2: hipLaunchKernelGGL(r4d_logits_wmma_kernel<2>, grid, block, smem, st, Ap, Wp, Cp, M, K, N, WV, SK); break;
  case 3: hipLaunchKernelGGL(r4d_logits_wmma_kernel<3>, grid, block, smem, st, Ap, Wp, Cp, M, K, N, WV, SK); break;
  default: hipLaunchKernelGGL(r4d_logits_wmma_kernel<4>, grid, block, smem, st, Ap, Wp, Cp, M, K, N, WV, SK); break;
  }
  return hipGetLastError() == hipSuccess ? 0 : -4;
}

int r4d_gemm_bf16_nt_m64_max_m(void) {
  return INT_MAX;
}

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h; see libr4d/r4d_args.h.
extern "C" int r4d_rad_gemm_bf16_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 0, 0, /*pitched=*/1);
    if (rc != RAD_OK) return rc;
    if (!r4d_bf16_gemm_chunk_rows(g.M, g.N, g.K, 16, 16)) return R4D_NO(RAD_E_SHAPE);
    if (g.lda % 4 || g.ldc > INT32_MAX) return R4D_NO(RAD_E_STRIDE);
    Cfg c;
    rc = cfg_for(a, g.K, g.M, 16, BR_FLAT, &c, (g.N + 15) / 16);  /* "K divisible by SK*16" */
    if (rc != RAD_OK) return rc;
    const bool chain = r4d_geti_or(a, "chain", 0) != 0;
    if (chain && r4d_bf16_gemm_chunk_rows_lda(g.M, g.N, g.K, g.lda, c.mb * 16, c.wv * 16) < g.M)
      return R4D_NO(RAD_E_UNSUPPORTED);
    R4D_TRY("gemm_bf16_nt_m64",
        gemm_bf16_m64_launch(r4d_addr(g.a), r4d_addr(g.b), r4d_addr(g.y),
                             (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb,
                             (long)(uintptr_t)s, chain, g.lda, g.ldc));
    return RAD_OK;
}


// logits_gemm_bf16_wmma: the operand checks of r4d_rad_logits_gemm (r4d_model_bf16.hip), the band
// 17..64 and dense rows, then the f32-output WMMA form.
extern "C" int r4d_rad_logits_gemm_wmma(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, 0);
    const RadTensor* w = r4d_opt(a, 1);
    const RadTensor* y = r4d_opt(a, 2);
    if (!x || !w || !y) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (y->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(w)) return R4D_NO(RAD_E_STRIDE);
    const long long n_vocab = r4d_geti_or(a, "n_vocab", r4d_dim(w, 0));
    const long long n_embd  = r4d_geti_or(a, "n_embd", r4d_dim(w, 1));
    if (n_vocab <= 0 || n_embd <= 0 || n_embd % 16) return R4D_NO(RAD_E_SHAPE);
    if (w->rank != 2 || w->shape[0] != n_vocab || w->shape[1] != n_embd) return R4D_NO(RAD_E_SHAPE);
    /* The launcher's weight offset is 32-bit unsigned: refuse here so the row falls through. */
    if (n_vocab * n_embd > 0xFFFFFFFFLL) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = r4d_rows_pitch(x, n_embd, &M);
    const long long y_ld = r4d_rows_pitch(y, n_vocab, &My);
    if (x_ld < 0 || y_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M < 17 || M > 64 || My < M) return R4D_NO(RAD_E_SHAPE);
    if (x_ld != n_embd || y_ld != n_vocab) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(w->data, 16)) return R4D_NO(RAD_E_ALIGN);
    return r4d_logits_gemm_bf16_wmma(r4d_addr(x), r4d_addr(w), r4d_addr(y), (int)M, (int)n_embd,
                                     (int)n_vocab, 0, 0, (long)(uintptr_t)s) == 0 ? RAD_OK
                                                                                  : RAD_E_DEVICE;
}
