// r4d_gemm_bf16x12_nt_m64.hip - gemm_bf16x12_nt_m16, _m64 and _tiled: the bf16 GEMMs over a bf16x12
// weight, gfx1201.
//
// C[M,N] = A[M,K] @ W[N,K]^T, bf16 in and out, fp32 accumulate, with W in libquant's bf16x12 (a
// bf16 weight in 12.25 bits a value, losslessly; libquant/lq_bf16x12.cpp) as the rows' layout hook
// stores it (r4d_layout.cpp): four regions of one buffer (r4d_args.h, r4d_x12_regions), in blocks
// of 32 along K, per row:
//   E:  1 byte a block, its largest exponent field. 0xFF marks a raw block.
//   SM: 1 byte an element, sign << 7 | mantissa.
//   D:  4 bits an element, E minus its exponent field (0..15).
// A block whose exponent fields span more than 15, or hold an inf or a NaN, is raw: its 32 bf16 sit
// in the escape table, 64 bytes a row, and its 16 D bytes hold its row four times. Qwen3.8-27B's
// trunk weights have no zeros, subnormals or infinities, and 0.133% of their blocks are raw.
//
// The order inside a block is the WMMA B fragment's (r4d_gdn_wmma.h: lane l holds row l % 16 and
// k = 8*(e>>2) + 4*(l>>4) + (e&3) of a 16-k step). For lane half h, k step t (0, 1) and fragment
// element e (0..7), the element at k = 16t + 8*(e>>2) + 4h + (e&3) is:
//   - SM byte 16h + 8t + e of the block's 32, so a lane half reads 16 contiguous bytes a block
//   - in D dword (h, t), byte 8h + 4t of the block's 16: element 2j's nibble at bit 4j and element
//     2j+1's at bit 16 + 4j (j = 0..3), so fragment dword j's two offsets are one shift and a mask
// The escape table holds a raw block's 32 bf16 in the same (h, t, e) order. Rows are K, K/2 and
// K/32 bytes in SM, D and E.
//
// The rest is r4d_gemm_bf16_nt_m64_kernel's: WV column tiles x SK k-splits waves, the same WMMA
// sequence in the same k order, the same LDS reduction in wave order and the same output rounding.
// The B fragments are bit for bit the bf16 weights, so C is byte-identical to gemm_bf16_nt_m64's at
// the same WV, SK and MB. One constraint more: K % (SK * 32) == 0, so a split holds whole blocks.
//
// On Qwen3.8-27B at TP2's decode shapes (M = 8, weights read from DRAM; r4d_selftest --perfx12):
// 12.27 bits a weight, and 0.785-0.829x gemm_bf16_nt_m64's time with each kernel at its fastest
// point.
//
// The prefill row, gemm_bf16x12_nt_tiled, rebuilds the weight in its scratch as the [N][K] bf16
// matrix and runs r4d_gemm_bf16_nt_tiled over it, at the split this file's rows take. So its C is
// gemm_bf16_nt_tiled's over the decoded weight at that split, and gemm_bf16x12_nt_m64's at any K.
#include "r4d.h"
#include "r4d_chain.h"
#include "r4d_gemm_args.h"
#include "r4d_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#define WAVE32 32
#define R4D_X12_MAX_M 65536

// Global-address-space pointers, as r4d_gemm_bf16_nt_m64.hip holds them (a generic pointer makes
// the loads flat_load, which on gfx12 also increments dscnt). The 16-byte element is a clang vector,
// not HIP's uint4 struct, which cannot be assigned from an address-space-qualified source.
typedef unsigned int x12_u4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) const x12_u4* x12_g16;
typedef __attribute__((address_space(1))) const unsigned long long* x12_g8;
typedef __attribute__((address_space(1))) const unsigned char* x12_g1;

// r4d_gemm_bf16_nt_m64.hip's fragU, for the activation: k step `u` of a K-contiguous row.
__device__ __forceinline__ v8bf x12_fragA(x12_g8 p, int u) {
  union { unsigned long long w[2]; v8bf f; } x;
  x.w[0] = p[4 * u];
  x.w[1] = p[4 * u + 2];
  return x.f;
}

// Fragment dword J (elements 2J and 2J+1) from the SM dword holding them (elements 0-3 or 4-7 of
// the k step), the step's D dword and E2 = E << 7 | E << 23. The perm writes each SM byte twice,
// so one mask keeps a mantissa (bits 0-6) and a sign (the copy's bit 7, bit 15). The exponent
// fields are E minus the offsets, and E >= each offset, so no borrow crosses the halves.
template <int J>
__device__ __forceinline__ unsigned x12_dword(unsigned s, unsigned d, unsigned e2) {
  constexpr unsigned b = (2 * J) & 3;
  constexpr unsigned sel = b | b << 8 | (b + 1) << 16 | (b + 1) << 24;
  const unsigned t = __builtin_amdgcn_perm(0u, s, sel);
  const unsigned off =
      (J == 0 ? d << 7 : J == 1 ? d << 3 : J == 2 ? d >> 1 : d >> 5) & 0x07800780u;
  return (t & 0x807F807Fu) | (e2 - off);
}

__device__ __forceinline__ v8bf x12_frag(unsigned s0, unsigned s1, unsigned d, unsigned e2) {
  union { unsigned w[4]; v8bf f; } x;
  x.w[0] = x12_dword<0>(s0, d, e2);
  x.w[1] = x12_dword<1>(s0, d, e2);
  x.w[2] = x12_dword<2>(s1, d, e2);
  x.w[3] = x12_dword<3>(s1, d, e2);
  return x.f;
}

// One block's two B fragments (k steps 0 and 1) for this lane: `sm` its 16 SM bytes, `dd` its two
// D dwords, `e` the block's E. A raw block reads lane half h's 32 bytes of its escape entry
// instead; on Qwen3.8-27B about 2% of a wave's blocks have a raw row.
__device__ __forceinline__ void x12_block(x12_u4 sm, unsigned long long dd, unsigned e,
                                          const unsigned char* esc, int h, v8bf& b0, v8bf& b1) {
  const unsigned d0 = (unsigned)dd, d1 = (unsigned)(dd >> 32), e2 = e * 0x00800080u;
  b0 = x12_frag(sm.x, sm.y, d0, e2);
  b1 = x12_frag(sm.z, sm.w, d1, e2);
  if (e == 0xFFu) {
    x12_g16 p = (x12_g16)(const x12_u4*)(esc + (size_t)d0 * 64 + 32 * h);
    union { x12_u4 u; v8bf f; } r0, r1;
    r0.u = p[0];
    r1.u = p[1];
    b0 = r0.f;
    b1 = r1.f;
  }
}

// KU is r4d_gemm_bf16_nt_m64's k-step unroll; an iteration covers KB blocks.
constexpr int KU = 8;
constexpr int KB = KU / 2;

// PRE loads the iteration's activation fragments with its plane loads. Without it LLVM issues each
// block's activation loads after that block's raw-block branch, so an iteration waits on the cache
// four times; with it they cost MT * 32 VGPRs, so it is built for MT <= 2 only (the decode band).
// CHAIN brackets the body with the chained launch's wait and add (r4d_chain.h), as in
// r4d_gemm_bf16_nt_m64_body; no block returns early. r4d_gemm_bf16x12_nt_m64_kernel and
// r4dchain_gemm_bf16x12_nt_m64 below wrap it.
template <int MT, bool PRE, bool CHAIN>
__device__ __forceinline__ void r4d_gemm_bf16x12_nt_m64_body(
    const unsigned short* __restrict__ A, const unsigned char* __restrict__ SM,
    const unsigned char* __restrict__ D, const unsigned char* __restrict__ E,
    const unsigned char* __restrict__ ESC, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int WV, int SK) {
  if (CHAIN) r4d_chain_wait();
  extern __shared__ float red[];              // WV*SK*256 floats: one 16x16 tile per (column, split)
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt = blockIdx.x * WV + col_local;
  const int n0 = nt * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK, nblk = Ks / 32, kbase = ksplit * Ks;
  const int h = lane >> 4;

  v8f acc[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) acc[i] = (v8f)(0.f);

  // 32-bit byte offsets for the planes and element offsets for the activation, turned into
  // pointers once, as in r4d_gemm_bf16_nt_m64. Rows past N re-read the last row (never stored).
  int r = n0 + (lane & 15);
  r = r < N - 1 ? r : N - 1;
  const unsigned smoff = (unsigned)r * (unsigned)K + (unsigned)kbase + 16u * (unsigned)h;
  const unsigned doff = (unsigned)r * (unsigned)(K / 2) + (unsigned)(kbase / 2) + 8u * (unsigned)h;
  const unsigned eoff = (unsigned)r * (unsigned)(K / 32) + (unsigned)(kbase / 32);
  unsigned aoff[MT];
#pragma unroll
  for (int i = 0; i < MT; i++)
    aoff[i] = rowGo((unsigned)K, (mt0 + i) * 16, M - 1, lane) + (unsigned)kbase;

  if (n0 < N) {
    x12_g16 psm = (x12_g16)(const x12_u4*)(SM + smoff);
    x12_g8 pd = (x12_g8)(const unsigned long long*)(D + doff);
    x12_g1 pe = (x12_g1)(E + eoff);
    x12_g8 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) pa[i] = (x12_g8)(const unsigned long long*)(A + aoff[i]);
    int b = 0;
    for (; b + KB <= nblk; b += KB) {
      // The iteration's plane loads first, so a raw block's branch cannot hold back the next
      // block's loads.
      x12_u4 sm[KB];
      unsigned long long dd[KB];
      unsigned ex[KB];
      v8bf av[PRE ? MT : 1][PRE ? KU : 1];
#pragma unroll
      for (int j = 0; j < KB; j++) {
        sm[j] = psm[2 * j];
        dd[j] = pd[2 * j];
        ex[j] = pe[j];
      }
      if constexpr (PRE) {
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int u = 0; u < KU; u++) av[i][u] = x12_fragA(pa[i], u);
      }
#pragma unroll
      for (int j = 0; j < KB; j++) {
        v8bf b0, b1;
        x12_block(sm[j], dd[j], ex[j], ESC, h, b0, b1);
#pragma unroll
        for (int i = 0; i < MT; i++)
          acc[i] = mma(PRE ? av[i][2 * j] : x12_fragA(pa[i], 2 * j), b0, acc[i]);
#pragma unroll
        for (int i = 0; i < MT; i++)
          acc[i] = mma(PRE ? av[i][2 * j + 1] : x12_fragA(pa[i], 2 * j + 1), b1, acc[i]);
      }
      psm += 2 * KB;
      pd += 2 * KB;
      pe += KB;
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += KU * 4;
    }
    for (; b < nblk; b++) {
      v8bf b0, b1;
      x12_block(psm[0], pd[0], pe[0], ESC, h, b0, b1);
#pragma unroll
      for (int i = 0; i < MT; i++) acc[i] = mma(x12_fragA(pa[i], 0), b0, acc[i]);
#pragma unroll
      for (int i = 0; i < MT; i++) acc[i] = mma(x12_fragA(pa[i], 1), b1, acc[i]);
      psm += 2;
      pd += 2;
      pe += 1;
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += 8;
    }
  }

  // r4d_gemm_bf16_nt_m64_kernel's reduction and store, unchanged.
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
      if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)v;
    }
  }
  if (CHAIN) r4d_chain_done();
}

template <int MT, bool PRE>
__global__ __launch_bounds__(1024) void r4d_gemm_bf16x12_nt_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned char* __restrict__ SM,
    const unsigned char* __restrict__ D, const unsigned char* __restrict__ E,
    const unsigned char* __restrict__ ESC, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int WV, int SK) {
  r4d_gemm_bf16x12_nt_m64_body<MT, PRE, false>(A, SM, D, E, ESC, C, M, K, N, WV, SK);
}
template <int MT, bool PRE>
__global__ __launch_bounds__(1024) void r4dchain_gemm_bf16x12_nt_m64(
    const unsigned short* __restrict__ A, const unsigned char* __restrict__ SM,
    const unsigned char* __restrict__ D, const unsigned char* __restrict__ E,
    const unsigned char* __restrict__ ESC, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int WV, int SK) {
  r4d_gemm_bf16x12_nt_m64_body<MT, PRE, true>(A, SM, D, E, ESC, C, M, K, N, WV, SK);
}

// grid = ceil(ceil(N/16)/WV) x ceil(ceil(M/16)/MB) blocks; block = WV*SK waves, as in
// r4d_gemm_bf16_nt_m64. `pre` picks the PRE form, which exists at MB <= 2. `chain` the chained
// form: the grid covers every row in one launch, so a chained op publishes one completion.
static void x12_launch(long a, long sm, long d, long e, long esc, long c, int M, int K, int N,
                       int WV, int SK, int MB, bool pre, bool chain, long stream) {
  if (M < 1 || M > R4D_X12_MAX_M)
    throw std::runtime_error("r4d gemm_bf16x12_nt_m64: M out of range");
  if (SK < 1 || K % (SK * 32))
    throw std::runtime_error("r4d gemm_bf16x12_nt_m64: K must be divisible by SK*32");
  if (WV < 1 || WV * SK * WAVE32 > 1024)
    throw std::runtime_error("r4d gemm_bf16x12_nt_m64: threads/block > 1024");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_bf16x12_nt_m64: MB must be 1..4");
  if (pre && MB > 2) throw std::runtime_error("r4d gemm_bf16x12_nt_m64: the PRE form is MB <= 2");
  const int ntiles = (N + 15) / 16, mtiles = (M + 15) / 16;
  dim3 grid((ntiles + WV - 1) / WV, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)WV * SK * 256 * sizeof(float);
  auto Ap = (const unsigned short*)a;
  auto Sp = (const unsigned char*)sm;
  auto Dp = (const unsigned char*)d;
  auto Ep = (const unsigned char*)e;
  auto Xp = (const unsigned char*)esc;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MT, PRE)                                                                            \
  hipLaunchKernelGGL((chain ? r4dchain_gemm_bf16x12_nt_m64<MT, PRE>                                \
                            : r4d_gemm_bf16x12_nt_m64_kernel<MT, PRE>),                            \
                     grid, block, smem, st, Ap, Sp, Dp, Ep, Xp, Cp, M, K, N, WV, SK)
  switch (MB * 2 + (pre ? 1 : 0)) {
    case 2: LAUNCH(1, false); break;
    case 3: LAUNCH(1, true); break;
    case 4: LAUNCH(2, false); break;
    case 5: LAUNCH(2, true); break;
    case 6: LAUNCH(3, false); break;
    case 8: LAUNCH(4, false); break;
    default: throw std::runtime_error("r4d gemm_bf16x12_nt_m64: MB unsupported");
  }
#undef LAUNCH
}

// ============================================================== the prefill row's unpack
//
// The weight as the row-major [N][K] bf16 matrix gemm_bf16_nt_tiled reads, one thread a block. The
// regions are row-major with rows of K, K/2 and K/32 bytes and no block crosses a row, so block i's
// planes sit at bytes 32i, 16i and i, and its values at bf16 32i. Its four B fragments are
// x12_block's, both lane halves of both k steps. Fragment (h, t) holds k = 16t + 4h + (0..3) in its
// dwords 0-1 and k = 16t + 8 + 4h + (0..3) in dwords 2-3, so output chunk 2t + q (k = 16t + 8q ..
// 16t + 8q + 7) is dwords 2q and 2q+1 of fragment (0, t), then of (1, t). Three 16-byte loads and a
// byte in, four 16-byte stores out.
typedef __attribute__((address_space(1))) x12_u4* x12_o16;

__global__ __launch_bounds__(256) void r4d_bf16x12_unpack_kernel(
    const unsigned char* __restrict__ SM, const unsigned char* __restrict__ D,
    const unsigned char* __restrict__ E, const unsigned char* __restrict__ ESC,
    x12_u4* __restrict__ W, unsigned nblk) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nblk) return;
  x12_g16 psm = (x12_g16)(const x12_u4*)(SM + (size_t)i * 32);
  const x12_u4 sm0 = psm[0], sm1 = psm[1];
  const x12_u4 dd = ((x12_g16)(const x12_u4*)(D + (size_t)i * 16))[0];
  const unsigned e = ((x12_g1)(E + i))[0];
  union { v8bf f; x12_u4 u; } f[2][2];   // [h][t]
  x12_block(sm0, (unsigned long long)dd.y << 32 | dd.x, e, ESC, 0, f[0][0].f, f[0][1].f);
  x12_block(sm1, (unsigned long long)dd.w << 32 | dd.z, e, ESC, 1, f[1][0].f, f[1][1].f);
  x12_o16 o = (x12_o16)(W + (size_t)i * 4);
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    o[2 * t] = x12_u4{ f[0][t].u.x, f[0][t].u.y, f[1][t].u.x, f[1][t].u.y };
    o[2 * t + 1] = x12_u4{ f[0][t].u.z, f[0][t].u.w, f[1][t].u.z, f[1][t].u.w };
  }
}

// grid = ceil(nblk / 256) blocks of 256 threads. The launch's HIP error.
static int x12_unpack(long sm, long d, long e, long esc, long w, long long nblk, long stream) {
  hipLaunchKernelGGL(r4d_bf16x12_unpack_kernel, dim3((unsigned)((nblk + 255) / 256)), dim3(256), 0,
                     (hipStream_t)stream, (const unsigned char*)sm, (const unsigned char*)d,
                     (const unsigned char*)e, (const unsigned char*)esc, (x12_u4*)w,
                     (unsigned)nblk);
  return (int)hipGetLastError();
}

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are r4d_gemm_args.h's, at the unit of a 32-block. At
// K = 5120, 8704 and 3072 its legal splits are the same at 16 and at 32, so an untuned row runs
// gemm_bf16_nt_m64's untuned splits there and gives its bytes.
extern "C" int r4d_rad_gemm_bf16x12_m64(const RadArgs* a, RadStream s) {
  GemmOps g;
  int rc = gemm_ops(a, &g, 0);
  if (rc != RAD_OK) return rc;
  // Any other weight would be read as planes it does not hold.
  if (g.b->dtype != R4D_DT_BF16X12) return R4D_NO(RAD_E_DTYPE);
  // The kernel's plane offsets are 32-bit: the last weight row has to be addressable.
  if (g.M > R4D_X12_MAX_M || g.K % R4D_X12_BLOCK || g.N * g.K >= (1ll << 32))
    return R4D_NO(RAD_E_SHAPE);
  Cfg c;
  rc = cfg_for(a, g.K, g.M, R4D_X12_BLOCK, BR_FLAT, &c, (g.N + 15) / 16);  // "K divisible by SK*32"
  if (rc != RAD_OK) return rc;
  bool pre = false;
  switch (r4d_cfg_bf16x12_form(a)) {
    case 0: pre = c.mb <= 2; break;
    case 1: break;
    case 2:
      // Refused rather than run plain, so a tuner never records a form that did not run.
      if (c.mb > 2) return R4D_NO(RAD_E_UNSUPPORTED);
      pre = true;
      break;
    default: return R4D_NO(RAD_E_INVAL);
  }
  // The row's chained form where the core chains it (kChained, r4d_rows.cpp), at decode heights
  // only, as gemm_bf16_nt_m64 (M <= 64): a wider grid's waiting blocks would hold room its
  // predecessor needs. A plain launch on a chained row is safe, since the backend chains a packet
  // by its kernel's name. A launch from outside the engine carries no record, so the chained form
  // runs as the plain one there.
  const bool chain = r4d_geti_or(a, "chain", 0) != 0 && g.M <= 64;
  const R4dX12Regions rg = r4d_x12_regions(g.N, g.K, 0);
  const long w = r4d_addr(g.b);
  R4D_TRY("gemm_bf16x12_nt_m64",
      x12_launch(r4d_addr(g.a), w + rg.sm, w + rg.d, w + rg.e, w + rg.esc, r4d_addr(g.y),
                 (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, pre, chain,
                 (long)(uintptr_t)s));
  return RAD_OK;
}

// The M <= 16 band's row, as gemm_bf16_nt_m16 is gemm_bf16_nt_m64's: the same launch under its own
// tuning entry.
extern "C" int r4d_rad_gemm_bf16x12_m16(const RadArgs* a, RadStream s) {
  return r4d_rad_gemm_bf16x12_m64(a, s);
}

// The prefill band's row: the weight unpacked into the op's scratch, then gemm_bf16_nt_tiled over
// it at the split gemm_bf16x12_nt_m64 takes untuned (the law at the unit of a 32-block), so the two
// packed rows agree at any K. At K = 5120, 8704 and 3072 that is gemm_bf16_nt_tiled's own split,
// so an untuned prefill gives the bf16 row's bytes. Whatever the tiled launcher would refuse is
// refused before the unpack runs.
extern "C" int r4d_rad_gemm_bf16x12_tiled(const RadArgs* a, RadStream s) {
  GemmOps g;
  int rc = gemm_ops(a, &g, 0);
  if (rc != RAD_OK) return rc;
  if (g.b->dtype != R4D_DT_BF16X12) return R4D_NO(RAD_E_DTYPE);
  // The tiled kernel's 64-deep K tile and 32-bit element offsets, and the unpack's block index.
  if (g.K < 64 || g.K % 64 || g.M * g.K >= (1ll << 32) || g.N * g.K >= (1ll << 32))
    return R4D_NO(RAD_E_SHAPE);
  if (!a->scratch || a->scratch_bytes < 2 * g.N * g.K) return R4D_NO(RAD_E_SCRATCH);
  if (!r4d_aligned(a->scratch, 16)) return R4D_NO(RAD_E_ALIGN);
  Cfg c;
  rc = cfg_for(a, g.K, g.M, R4D_X12_BLOCK, BR_FLAT, &c, (g.N + 15) / 16);
  if (rc != RAD_OK) return rc;
  int tile = (int)r4d_geti_or(a, "tile", 0);
  if (tile == 0) tile = r4d_gemm_bf16_nt_tiled_arm((int)g.M, (int)g.N);
  if (tile != 1 && tile != 2) return R4D_NO(RAD_E_SHAPE);
  const R4dX12Regions rg = r4d_x12_regions(g.N, g.K, 0);
  const long w = r4d_addr(g.b), wb = (long)(uintptr_t)a->scratch, st = (long)(uintptr_t)s;
  if (x12_unpack(w + rg.sm, w + rg.d, w + rg.e, w + rg.esc, wb, g.N * (g.K / R4D_X12_BLOCK), st))
    return R4D_NO(RAD_E_DEVICE);
  const int e = r4d_gemm_bf16_nt_tiled(r4d_addr(g.a), wb, r4d_addr(g.y), (int)g.M, (int)g.K,
                                       (int)g.N, c.sk, tile, st);
  return e == 0 ? RAD_OK : R4D_NO(e < 0 ? RAD_E_SHAPE : RAD_E_DEVICE);
}

// The op's scratch: the weight in bf16, N x K x 2 bytes, at any row count. Qwen3.8-27B's gate|up at
// TP2 (17408 x 5120) takes the most, 170 MiB.
extern "C" int64_t r4d_rad_gemm_bf16x12_tiled_scratch(const RadArgs* a) {
  const long long N = r4d_geti_or(a, "N", 0), K = r4d_geti_or(a, "K", 0);
  return N > 0 && K > 0 ? (int64_t)(2 * N * K) : 0;
}
