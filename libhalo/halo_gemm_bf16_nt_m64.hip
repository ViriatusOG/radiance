/* Derived from libr4d/r4d_gemm_bf16_nt_m64.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_gemm_bf16_nt_m64.hip - halo_gemm_bf16_nt_m64: skinny bf16 GEMM at any M, gfx1151.
//
// C[M,N] = A[M,K] @ W[N,K]^T, bf16 in and out, fp32 accumulate.
//
// General in M up to HALO_GEMM64_MAX_M: MT row tiles per block, grid.y covers the rest. The 64 is
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
#include "halo.h"
#include "halo_launch.h"   // HALO_DYN_LDS: tests/halo_emu runs this unit on the host
#include "halo_gemm_args.h"
#include "halo_gdn_wmma.h"
#include "halo_chain.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#define HALO_GEMM64_MAX_M 65536
#define WAVE32 32

// A fragment from a base pointer plus a compile-time k step. With fragGo's 32-bit element offset,
// LLVM re-derives a 64-bit address per fragment per step (a b64 shift and a carry-in add pair); a
// constant step instead folds into the 13-bit immediate of global_load_b64 (32 bytes per k step,
// so eight fit).
//
// The pointer is explicitly global: a plain pointer held in an array across iterations becomes
// generic, so loads become flat_load_b64, which on RDNA also increments dscnt and makes later LDS
// waits wait on global memory. The element is a scalar because HIP's vector structs cannot be
// assigned from an address-space-qualified source.
//
// gfx11 form: the fragment is the row's whole 16-wide k run, 32 contiguous bytes (halo_wmma.h),
// where libr4d's gfx12 form is two 8-byte halves sixteen bytes apart. The pointer arithmetic is
// unchanged -- a k step is 32 bytes either way -- only what one step loads differs.
typedef unsigned halo_u4v __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) const halo_u4v* halo_gu4;

__device__ __forceinline__ v16bf fragU(halo_gu4 p, int u) {
  union { halo_u4v w[2]; v16bf f; } x;
  x.w[0] = p[2 * u];
  x.w[1] = p[2 * u + 1];
  return x.f;
}

// MT = row tiles per block, held as MT v8f accumulators (8 VGPRs each); grid.y covers the rest.
// More amortises each weight fragment, fewer puts more blocks on the machine, so MT is tunable.
// CHAIN brackets the body with the chained launch's wait and add (halo_chain.h); no block returns
// early. halo_gemm_bf16_nt_m64_kernel and halo_r4dchain_gemm_bf16_nt_m64 below wrap it.
template <int MT, bool CHAIN>
__device__ __forceinline__ void halo_gemm_bf16_nt_m64_body(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK) {
  if (CHAIN) halo_chain_wait();
  HALO_DYN_LDS(float, red);              // WV*SK*256 floats: one 16x16 tile per (column, split)
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
    aoff[i] = rowGo((unsigned)K, (mt0 + i) * 16, M - 1, lane) + (unsigned)kbase;

  // The k walk is a ring of D steps' fragments in registers: step s + D's loads issue as soon as
  // step s's WMMAs have consumed its slot, so D steps of loads are in flight under the WMMAs. With
  // only the next step in flight (the plain KU-unrolled loop) every step waited out a memory round
  // trip -- 270 us at N 2048, K 4096, M 17 against a 75 us weight stream. Each accumulator still
  // takes its WMMAs in ascending k, so the bits are the plain loop's. A k step is a load immediate
  // (the 13-bit field of global_load_b64) and the base pointers advance once per D steps; rolled,
  // LLVM rebuilds a 64-bit address every step.
  // D at the VGPR bound: a fragment is 8 VGPRs, (MT + 1) of them a step. MT 1 spills from D 6 on
  // (the compiler double-buffers that ring), so it stays at 4.
  constexpr int D = MT == 1 ? 4 : (MT == 2 ? 6 : (MT == 3 ? 4 : 3));
  if (n0 < N) {
    halo_gu4 pw = (halo_gu4)(const halo_u4v*)(W + woff);
    halo_gu4 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) pa[i] = (halo_gu4)(const halo_u4v*)(A + aoff[i]);
    if (ksteps < D) {
      for (int s = 0; s < ksteps; s++) {
        const v16bf b = fragU(pw, 0);
        pw += 2;
#pragma unroll
        for (int i = 0; i < MT; i++) {
          acc[i] = mma(fragU(pa[i], 0), b, acc[i]);
          pa[i] += 2;
        }
      }
    } else {
      v16bf rb[D], ra[D][MT];
#pragma unroll
      for (int d = 0; d < D; d++) {
        rb[d] = fragU(pw, d);
#pragma unroll
        for (int i = 0; i < MT; i++) ra[d][i] = fragU(pa[i], d);
      }
      int s = 0;
      for (; s + 2 * D <= ksteps; s += D) {
#pragma unroll
        for (int d = 0; d < D; d++) {
#pragma unroll
          for (int i = 0; i < MT; i++) acc[i] = mma(ra[d][i], rb[d], acc[i]);
          rb[d] = fragU(pw, D + d);
#pragma unroll
          for (int i = 0; i < MT; i++) ra[d][i] = fragU(pa[i], D + d);
        }
        pw += D * 2;
#pragma unroll
        for (int i = 0; i < MT; i++) pa[i] += D * 2;
      }
      // The ring holds steps s .. s+D-1; r = ksteps - s - D more (0 <= r < D) refill its first r
      // slots and run last, so the order stays ascending.
      const int r = ksteps - s - D;
#pragma unroll
      for (int d = 0; d < D; d++) {
#pragma unroll
        for (int i = 0; i < MT; i++) acc[i] = mma(ra[d][i], rb[d], acc[i]);
        if (d < r) {
          rb[d] = fragU(pw, D + d);
#pragma unroll
          for (int i = 0; i < MT; i++) ra[d][i] = fragU(pa[i], D + d);
        }
      }
#pragma unroll
      for (int d = 0; d < D - 1; d++) {
        if (d >= r) break;
#pragma unroll
        for (int i = 0; i < MT; i++) acc[i] = mma(ra[d][i], rb[d], acc[i]);
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
      const int m = (mt0 + i) * 16 + accRow(l, off >> 5);   // gfx11 D mapping (halo_gdn_wmma.h)
      // The language's float-to-bf16 conversion on purpose; the header's f2bf rounds half away
      // from zero.
      if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)v;
    }
  }
  if (CHAIN) halo_chain_done();
}

template <int MT>
__global__ __launch_bounds__(1024) void halo_gemm_bf16_nt_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK) {
  halo_gemm_bf16_nt_m64_body<MT, false>(A, W, C, M, K, N, WV, SK);
}
template <int MT>
__global__ __launch_bounds__(1024) void halo_r4dchain_gemm_bf16_nt_m64(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK) {
  halo_gemm_bf16_nt_m64_body<MT, true>(A, W, C, M, K, N, WV, SK);
}

// grid = ceil(ceil(N/16)/WV) x ceil(ceil(M/16)/MB) blocks; block = WV*SK waves. MT = MB is
// compile-time so the accumulators are sized exactly.
static void gemm_bf16_m64_launch(long a, long w, long c, int M, int K, int N, int WV, int SK,
                                 int MB, long stream, bool chain);
void halo_gemm_bf16_nt_m64(long a, long w, long c, int M, int K, int N, int WV, int SK, int MB,
                          long stream) {
  gemm_bf16_m64_launch(a, w, c, M, K, N, WV, SK, MB, stream, false);
}
/* `chain`: the chained form (halo_rad_gemm_bf16_m64 passes the row's parameter; M <= 64 always). */
static void gemm_bf16_m64_launch(long a, long w, long c, int M, int K, int N, int WV, int SK,
                                 int MB, long stream, bool chain) {
  if (M < 1 || M > HALO_GEMM64_MAX_M) throw std::runtime_error("r4d gemm_bf16_nt_m64: M out of range");
  if (K % (SK * 16)) throw std::runtime_error("r4d gemm_bf16_nt_m64: K must be divisible by SK*16");
  if (WV * SK * WAVE32 > 1024) throw std::runtime_error("r4d gemm_bf16_nt_m64: threads/block > 1024");
  const int ntiles = (N + 15) / 16, mtiles = (M + 15) / 16;
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_bf16_nt_m64: MB must be 1..4");
  dim3 grid((ntiles + WV - 1) / WV, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)WV * SK * 256 * sizeof(float);
  auto Ap = (const unsigned short*)a; auto Wp = (const unsigned short*)w;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MT) hipLaunchKernelGGL((chain ? halo_r4dchain_gemm_bf16_nt_m64<MT>                  \
                                             : halo_gemm_bf16_nt_m64_kernel<MT>),              \
                                      grid, block, smem, st, Ap, Wp, Cp, M, K, N, WV, SK)
  switch (MB) {
    case 1: LAUNCH(1); break;
    case 2: LAUNCH(2); break;
    case 3: LAUNCH(3); break;
    case 4: LAUNCH(4); break;
    default: throw std::runtime_error("r4d gemm_bf16_nt_m64: MB unsupported");
  }
#undef LAUNCH
}

int halo_gemm_bf16_nt_m64_max_m(void) { return HALO_GEMM64_MAX_M; }

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in halo_gemm_args.h; see libhalo/halo_args.h.
extern "C" int halo_rad_gemm_bf16_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 0);
    if (rc != RAD_OK) return rc;
    if (g.M > halo_gemm_bf16_nt_m64_max_m()) return HALO_NO(RAD_E_SHAPE);
    Cfg c;
    rc = cfg_for(a, g.K, g.M, 16, BR_FLAT, &c, (g.N + 15) / 16);  /* "K divisible by SK*16" */
    if (rc != RAD_OK) return rc;
    /* cfg_for's decode band gives MB = 1 (one row tile a block). With the k ring that stays
     * right for a narrow weight, but from 64 column tiles on every row-tile block re-streams the
     * weight, and one block over all of them reads it once: on gfx1151 at N 2048, K 4096, M 64,
     * 350 -> 194 us; 4608 x 4608 455 -> 285; at N <= 256 MB = 1 stays 20-90% faster. BR_FLAT
     * keeps the split, so the bits do not move; a tuned mb is kept. */
    if (halo_geti_or(a, "mb", 0) <= 0 && g.M <= 64 && (g.N + 15) / 16 >= 64)
        c.mb = (int)((g.M + 15) / 16);
    HALO_TRY("gemm_bf16_nt_m64",
        gemm_bf16_m64_launch(halo_addr(g.a), halo_addr(g.b), halo_addr(g.y),
                             (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb,
                             (long)(uintptr_t)s, halo_geti_or(a, "chain", 0) != 0));
    return RAD_OK;
}

