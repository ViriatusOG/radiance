// exl3i_nt.hip.h -- exl3i's dense linear (exl3i_dense) on gfx1201's f16 matrix cores, with the trellis decode a template
// parameter R, a rate policy (exl3i_rate.hip.h): Plan at integer K, the run-time plan at fractional and sub-1 rates,
// the codebook mad on either. Everything else -- the GSCR rotation, MT / ALDS prefill form,
// T2's two-pass prefill, the epilogue -- is shared with the other kernels (exl3r4d_common.hip.h), unchanged. Included by
// exl3i_nt_mad.hip, which sets the codebook (mad) and exports its launch.
//
// Kernel names carry exl3i_ (two plugins in one process must not export the same kernel symbols).
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <cstdlib>
#include <mutex>

#include "exl3r4d_common.hip.h"
#include "exl3i_rate.hip.h"
#include "exl3i_dev.h"
#include "exl3i_pin.h"

#ifndef EXL3R4D_NT          // nontemporal weight loads (decode band)
#define EXL3R4D_NT 1
#endif
#ifndef EXL3R4D_BUF         // buffer loads for the weight / activation fragments
#define EXL3R4D_BUF 1
#endif
#ifndef EXL3R4D_EPIC        // the epilogue's paired tile sums (adopted; T2 uses the same order)
#define EXL3R4D_EPIC 1
#endif

// ============================================================================= the dense linear (exl3i_gemm_nt)
//
// One weight [N, K], no routing: token-ordered rows in the GSCR form (each workgroup rotates its rows into the
// rotated-rows region, the K loop reads buffer loads from it), K 1..8, mad, any M (one 16-row tile a workgroup at
// M <= 16, EXL3R4D_NT_MT tiles past it -- below), split-K 1 only (the fit logic that would pick a split is not in this form).
//
// THE SEAM: when the weight has plane 3 (suh.1 -- the operand's RadTensor.data is null when it does not),
// the output rows [0, N/2) are scaled by suh and [N/2, N) by suh.1 (the stacked gate_up's rule; the
// reference's launch_nt makes the same two-half split, its 128-point rotation demanding N/2 % 128 -- the
// row enforces N % 256 in that case). Every 128-column workgroup lies wholly on one side, so it rotates
// with one scale vector and reads one half of the region. The region is [T][2K] f16 (one shared copy, not
// P1c's per-column-block copies: same-side workgroups write the SAME bytes, and a workgroup reads only the
// rows it rotated itself, its own barrier ordering them). Without plane 3, planes_of sets suh1 = suh and
// the selection is inactive (both halves rotate identically; the read takes the workgroup's own half).
//
// THE ARITHMETIC, word for word the decode band's: a[m] x suh (or suh.1) rotated by rotate128, the W_hat
// trellis decoded in registers (Plan), f16 WMMAs, the split-K LDS reduction (sk > 1), the 128-point output
// rotation x 1/sqrt(128) x svh, bf16 / f16 / f32 out (the epilogue_block MATS 1 arm, EPIC on, EPIB off). One dispatch, no prep kernel, no counters, no scratch past the region.
#if EXL3R4D_NCLS == 2 && !EXL3R4D_GD4
// Defaults of the decode form: EXL3R4D_NT_PF 2 load stages, buffer loads, EPIC on, EPIB off.
#ifndef EXL3R4D_NT_PF
#define EXL3R4D_NT_PF 2
#endif
// The decode form's (MT 1, no PREP) mad decode: 1 the grid indices into the WMMAs and the step after the sums (DQ in
// nt_pass), 0 the codebook's fp16 values (the prefill forms' decode).
#ifndef EXL3I_DEC_Q
#define EXL3I_DEC_Q 1
#endif
// The decode form's (MT 1, no PREP) load stages: k steps of weight words and activation fragments in flight a wave.
// 3 (the prefill forms use 2): at T 1 on the R9700 the 48 : 16 layer x0.94-0.996 at every K from 2 to 7 and
// the LM head x0.92-0.96 at fractional K from 4.125 (K4.125 531 -> 577 GB/s), x1.05 at K2,
// +-1.5% at K2.5 / K4 / K6: a fractional tile's decode (more VALU, a fourth word) no longer hides behind two stages of
// loads. T 4 / 8 +-1.5% (K6.5 x0.93-0.97). Same bits.
#ifndef EXL3I_DEC_PF
#define EXL3I_DEC_PF 3
#endif
#ifndef EXL3R4D_NT_KX_BUF
#define EXL3R4D_NT_KX_BUF 1
#endif
// The three-word runs (integer K7, fractional K5.125-7.875) by buffer loads too: they took the global form,
// a 64-bit address add a word (12 a K step at NPW 2), and 140 VGPRs against 88-94 at the two-word rates.
#ifndef EXL3I_NW3_BUF
#define EXL3I_NW3_BUF 1
#endif
#ifndef EXL3R4D_NT_EPIC
#define EXL3R4D_NT_EPIC 1
#endif

#ifndef EXL3R4D_NT_WPE
#define EXL3R4D_NT_WPE 0
#endif
#if EXL3R4D_NT_WPE > 0
#define EXL3R4D_NT_WPE_ATTR __attribute__((amdgpu_waves_per_eu(EXL3R4D_NT_WPE)))
#else
#define EXL3R4D_NT_WPE_ATTR
#endif

// MT (the prefill form): a workgroup owns MT consecutive 16-row tiles, so one decoded weight fragment feeds MT
// WMMAs (one weight, token-ordered rows, no run map). MT 1 is the decode
// band's arithmetic exactly; every tile's sums run in the same order whatever MT is, so the bits do not depend on it
// (measured: the 2k-token KLD record identical at MT 1, 2, 4, 8). Measured on gfx1201, Qwen3.8-27B 4.00 bpw, 2143-token
// prefill, dense op total: MT 2 2805 ms, MT 4 2454 ms, MT 8 4038 ms (spills) -- 4 adopted.
#ifndef EXL3R4D_NT_MT
#define EXL3R4D_NT_MT 4
#endif
// ALDS (prefill form): the activation slab [16 MT rows][64 features] staged once a workgroup in LDS (coalesced 16-B
// loads, double-buffered, one barrier a slab) instead of each wave loading the same fragments from global -- the 4
// waves of a workgroup read identical A, and a 16-row fragment load uses a quarter of each line. Same values, same
// sums in the same order.
// Launch bound: at most 8 waves a workgroup.
#ifndef EXL3R4D_NT_LB
#define EXL3R4D_NT_LB 256
#endif
// RG (prefill form, ALDS): row groups a workgroup -- RG sets of 8 / NPW waves over the same 128 columns, each its own
// 16 MT rows: one staged slab and one column block's weight stream serve RG x the rows, each wave's registers as at RG 1.
#ifndef EXL3R4D_NT_RG
#define EXL3R4D_NT_RG 1
#endif
#ifndef EXL3R4D_NT_ALDS
#define EXL3R4D_NT_ALDS 1
#endif
#define EXL3R4D_NT_KC 4                       // k steps a slab (64 features)
#define EXL3R4D_NT_AP (16 * EXL3R4D_NT_KC + 8)  // the slab's LDS row pitch, halfs (+16 B against bank conflicts)

// KS (decode, MT 1, no PREP): this workgroup is K slice blockIdx.z of ksn -- it rotates and multiplies only its slice's
// 128-feature blocks [nkb z / ksn, nkb (z + 1) / ksn), writes its raw sums to kpart, and the last of the column
// block's ksn workgroups to arrive (kctr[blockIdx.x]) sums every slice's in slice order (so the bits do not depend on
// who arrives last) into the epilogue. The sums differ from KS off's (one f32 order a slice, then across slices).
// DOT (the decode form at one row, T == 1): each tile's sums by four v_dot2_f32_f16 a lane (its eight k against row 0's)
// instead of a WMMA whose other 15 rows are padding; after the K loop the lane pair (h 0 / 1) adds and row 0's 16 columns
// move into the WMMA accumulator layout for the epilogue. Sums in another order than the WMMA's.
template <class R, int NPW, int MT, bool PREP, int RG, bool KS = false, bool DOT = false>
static __device__ __forceinline__ void nt_pass(const Exl3r4dGemm& p, const Planes& w, int cb, int rc, int r1, float* lds,
                                               int m, int ksn = 1, float* kpart = nullptr,
                                               unsigned* kctr = nullptr, int r16 = 0) {
  constexpr int NW = R::NW;
  const int WORDS = R::words(m);
  constexpr int PF = MT == 1 && !PREP ? EXL3I_DEC_PF : EXL3R4D_NT_PF;
  const int SK = p.sk;
  const int tid = (int)threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  static_assert(RG == 1 || (PREP && EXL3R4D_NT_ALDS), "row groups are the staged-slab form's");
  static_assert(!KS || (MT == 1 && !PREP && RG == 1), "the K split across workgroups is the decode form's");
  static_assert(!DOT || (MT == 1 && !PREP && RG == 1), "DOT is the decode form's");
  constexpr int WVC = 8 / NPW;           // waves a row group (the 128 columns)
  const int rg = RG > 1 ? wave / WVC : 0, wc = RG > 1 ? wave % WVC : wave;
  const int col_local = wc / SK, ksplit = wc % SK;
  const int h = lane >> 4, nl = lane & 15;
  const int width = w.rows;              // [N, K]: one matrix, no gate/up split
  const int ntiles = width / 16, ksteps = w.cols / 16;
  // The seam (above): this workgroup's half of the region, and its scale vector.
  const int mat_a = (w.suh1 != w.suh && cb * 128 >= width / 2) ? 1 : 0;
  const R plan(lane, m);
  unsigned wvo[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wvo[r][x] = 4u * (unsigned)plan.woff[r][x];

  // The record: one [N, K] matrix (no mat offset -- the stacked form's `mat * ksteps * ntiles * WORDS`
  // is 0 here).
  gu8 wt[NPW];
  const unsigned* wexp = w.codes;
  // KS: this slice's 128-feature blocks [kb0, kb1)
  [[maybe_unused]] int kb0 = 0, kb1 = w.cols / 128;
  if constexpr (KS) {
    const int nkb = w.cols / 128, z = (int)blockIdx.z;
    kb0 = nkb * z / ksn;
    kb1 = nkb * (z + 1) / ksn;
  }
  const size_t kps = KS ? (size_t)(kb1 - kb0) * 8 : ksteps / SK;
  const size_t ks0 = KS ? (size_t)kb0 * 8 : (size_t)ksplit * kps;
  const int vl = col_local * NPW;
  const int nt0 = cb * 8 + vl % 8;
  const size_t tile0 = ks0 * ntiles + nt0;
#pragma unroll
  for (int j = 0; j < NPW; j++) wt[j] = (gu8)(const unsigned char*)(wexp + (tile0 + j) * WORDS);
  const size_t kbytes = (size_t)ntiles * WORDS * 4;

  // GSCR: rotate this workgroup's rows into the shared region [T][2K], this workgroup's half, a batch of 4 with
  // every load of the batch first (the down's P1c loop, the sorted indirection gone -- dense row m is token m).
  // PREP: exl3i_nt_prep_kernel rotated every row once before this dispatch (the same function, the same bits);
  // in-workgroup, every column block would rotate the same rows again (272 times over at gate_up).
  if constexpr (!PREP) {
    const int nrows = min(EXL3_BLOCK * MT, r1 - rc);
    const int blocks = KS ? kb1 - kb0 : w.cols / 128;   // KS: the slice's blocks only (bk below is relative to kb0)
    const int tasks = nrows * blocks;
    const int nwaves = (int)EXL3R4D_BDIM / WAVE32;
    constexpr int NB = 4;
    typedef const __attribute__((address_space(1))) v2u* gv2;
    const gu8 xb = (gu8)(const unsigned char*)p.x;
    const gu8 sx = (gu8)(const unsigned char*)(mat_a ? w.suh1 : w.suh);
    _Float16* xo = (_Float16*)p.xr + (size_t)mat_a * w.cols;
    for (int t0 = wave; t0 < tasks; t0 += NB * nwaves) {
      int bi[NB], bk[NB];
      v2u bx[NB], bs[NB];
#pragma unroll
      for (int b = 0; b < NB; b++) {
        const int t1 = t0 + b * nwaves, tsk = t1 < tasks ? t1 : t0;
        bi[b] = tsk / blocks;
        bk[b] = tsk - bi[b] * blocks + kb0;
      }
#pragma unroll
      for (int b = 0; b < NB; b++) {
        bx[b] = *(gv2)(xb + ((size_t)(rc + bi[b]) * p.x_pitch + bk[b] * 128 + 4 * lane) * 2);
        bs[b] = *(gv2)(sx + (size_t)(bk[b] * 128 + 4 * lane) * 2);
      }
#pragma unroll
      for (int b = 0; b < NB; b++)
        if (t0 + b * nwaves < tasks)
          *(v2u*)(xo + (size_t)(rc + bi[b]) * p.xr_pitch + bk[b] * 128 + 4 * lane) =
              rotate128_math(bx[b], bs[b], p.x_bf16 == 0, lane);
    }
    __syncthreads();   // the workgroup's release/acquire on global: its rows are visible to its reads
  }

  // Tile t's rows: rc + 16 t + nl; a row past the run reads the run's first row (valid memory) and is not stored.
  int row[MT];
  bool in_run[MT];
  unsigned avo[MT];
#pragma unroll
  for (int t = 0; t < MT; t++) {
    row[t] = rc + EXL3_BLOCK * (MT * rg + t) + nl;
    in_run[t] = EXL3_BLOCK * (MT * rg + t) + nl < r1 - rc;
    avo[t] = (unsigned)((size_t)(in_run[t] ? row[t] : rc) * p.xr_pitch + 4 * h) * 2;
  }
  gu8 abase = (gu8)(const unsigned char*)((const _Float16*)p.xr + (size_t)mat_a * w.cols + (size_t)ks0 * 16);

  v8f acc[MT][NPW];
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int j = 0; j < NPW; j++) acc[t][j] = (v8f)(0.f);
  [[maybe_unused]] float dacc[NPW];                     /* DOT: this lane's column (n = lane % 16) over its k half */
#pragma unroll
  for (int j = 0; j < NPW; j++) dacc[j] = 0.f;
  // DQ (the decode form, mad, EXL3I_DEC_Q): the WMMAs take the grid indices q, the sums are scaled by the step once
  // after the K loop (exl3i_fin16q).
  constexpr bool DQ = EXL3I_DEC_Q && MT == 1 && !PREP && R::CBV == EXL3I_DEV_CB_MAD;

  // The K loop: the buffer form with EXL3R4D_NT_PF stages; K = 4 takes three words a run, every other K four.
  [[maybe_unused]] const __amdgpu_buffer_rsrc_t wrs =
      __builtin_amdgcn_make_buffer_rsrc((void*)wexp, (short)0, 0x7fffffff, 0x31004000);
  [[maybe_unused]] const __amdgpu_buffer_rsrc_t ars =
      __builtin_amdgcn_make_buffer_rsrc((void*)abase, (short)0, 0x7fffffff, 0x31004000);
  [[maybe_unused]] const unsigned bso = (unsigned)(tile0 * WORDS * 4);
  [[maybe_unused]] const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  constexpr bool kBuf = EXL3R4D_BUF && (NW == 2 || EXL3I_NW3_BUF) && (R::K4 || EXL3R4D_NT_KX_BUF);
  constexpr int kAux = EXL3R4D_NT ? 1 : 0;

#define NT_LOAD(ks, W_, A_)                                                        \
  do {                                                                               \
    if constexpr (kBuf) {                                                            \
      const unsigned so_ = bso + (unsigned)(ks) * (unsigned)kbytes;                  \
      _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                \
        if constexpr (R::K4) {                                                       \
          unsigned m_, z_, q_;                                                       \
          m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * 128, so_, kAux);  \
          z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * 128, so_, kAux);  \
          q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * 128, so_, kAux);  \
          W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;   \
        } else {                                                                     \
          _Pragma("unroll") for (int r = 0; r < 2; r++)                               \
          _Pragma("unroll") for (int x = 0; x < NW; x++)                             \
            W_[j][r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x] + j * WORDS * 4, so_, kAux); \
        }                                                                            \
      }                                                                              \
      _Pragma("unroll") for (int t = 0; t < MT; t++) {                                 \
        union { v2u w[2]; v8h f; } x_;                                               \
        x_.w[0] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo[t], (unsigned)(ks) * 32u, 0);      \
        x_.w[1] = __builtin_amdgcn_raw_buffer_load_b64(ars, avo[t] + 16u, (unsigned)(ks) * 32u, 0); \
        A_[t] = x_.f;                                                                \
      }                                                                              \
    } else {                                                                         \
      _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                \
        const gu8 tb_ = wt[j] + (size_t)(ks) * kbytes;                                 \
        _Pragma("unroll") for (int r = 0; r < 2; r++)                                \
        _Pragma("unroll") for (int x = 0; x < NW; x++)                               \
          W_[j][r][x] = EXL3R4D_NT ? __builtin_nontemporal_load((gu32)(tb_ + wvo[r][x])) \
                                   : *(gu32)(tb_ + wvo[r][x]);                        \
      }                                                                              \
      _Pragma("unroll") for (int t = 0; t < MT; t++) {                                 \
        union { v2u w[2]; v8h f; } x_;                                               \
        const unsigned char* ap_ = (const unsigned char*)abase + avo[t] + (unsigned)(ks) * 32u; \
        x_.w[0] = *(const v2u*)ap_;                                                  \
        x_.w[1] = *(const v2u*)(ap_ + 16u);                                          \
        A_[t] = x_.f;                                                                \
      }                                                                              \
    }                                                                                \
  } while (0)

#define NT_COMPUTE(W_, A_)                                                         \
  do {                                                                               \
    _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                \
      const v8h wf = plan.template decode<DQ>(W_[j]);                                \
      if constexpr (DOT) {                                                           \
        const v8h af_ = A_[0];                                                       \
        float d_ = dacc[j];                                                          \
        _Pragma("unroll") for (int e = 0; e < 8; e += 2)                             \
          d_ = __builtin_amdgcn_fdot2((v2h){wf[e], wf[e + 1]}, (v2h){af_[e], af_[e + 1]}, d_, false); \
        dacc[j] = d_;                                                                \
      } else {                                                                       \
        _Pragma("unroll") for (int t = 0; t < MT; t++) acc[t][j] = mma(wf, A_[t], acc[t][j]); \
      }                                                                              \
    }                                                                                \
  } while (0)

  if constexpr (PREP && EXL3R4D_NT_ALDS) {
    constexpr int KC = EXL3R4D_NT_KC, AP = EXL3R4D_NT_AP, ROWS = 16 * MT * RG, SLAB = ROWS * AP;
    static_assert(KC % PF == 0, "the weight stages index by the slab's step");
    _Float16* const abuf = (_Float16*)lds;          // [2][16 MT][AP] (the epilogue reuses it after the loop)
    constexpr int nth = WVC * RG * WAVE32;
    constexpr int NLD = (ROWS * KC * 2 + nth - 1) / nth;   // 16-B loads a thread
    const _Float16* const xrow = (const _Float16*)p.xr + (size_t)mat_a * w.cols;
    v4u stg[NLD];
    auto slab_load = [&](int c) {
#pragma unroll
      for (int i = 0; i < NLD; i++) {
        const int idx = tid + nth * i, r = idx / (2 * KC), q = idx % (2 * KC);
        const int rr = rc + r < r1 ? rc + r : rc;   // a row past the run reads the run's first (not stored)
        stg[i] = idx < ROWS * 2 * KC ? *(const v4u*)(xrow + (size_t)rr * p.xr_pitch + c * KC * 16 + q * 8) : (v4u)(0u);
      }
    };
    auto slab_store = [&](int b) {
#pragma unroll
      for (int i = 0; i < NLD; i++) {
        const int idx = tid + nth * i, r = idx / (2 * KC), q = idx % (2 * KC);
        if (idx < ROWS * 2 * KC) *(v4u*)(abuf + b * SLAB + r * AP + q * 8) = stg[i];
      }
    };
#define NT_LOADW(ks, W_)                                                           \
    do {                                                                             \
      const unsigned so_ = bso + (unsigned)(ks) * (unsigned)kbytes;                  \
      _Pragma("unroll") for (int j = 0; j < NPW; j++) {                                \
        if constexpr (kBuf && R::K4) {                                               \
          unsigned m_, z_, q_;                                                       \
          m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * 128, so_, kAux);  \
          z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * 128, so_, kAux);  \
          q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * 128, so_, kAux);  \
          W_[j][0][0] = m_; W_[j][0][1] = z_; W_[j][1][0] = z_; W_[j][1][1] = q_;   \
        } else if constexpr (kBuf) {                                                 \
          _Pragma("unroll") for (int r = 0; r < 2; r++)                               \
          _Pragma("unroll") for (int x = 0; x < NW; x++)                             \
            W_[j][r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x] + j * WORDS * 4, so_, kAux); \
        } else {                                                                     \
          const gu8 tb_ = wt[j] + (size_t)(ks) * kbytes;                               \
          _Pragma("unroll") for (int r = 0; r < 2; r++)                              \
          _Pragma("unroll") for (int x = 0; x < NW; x++)                             \
            W_[j][r][x] = *(gu32)(tb_ + wvo[r][x]);                                  \
        }                                                                            \
      }                                                                              \
    } while (0)
    unsigned wb[PF][NPW][2][NW];
#pragma unroll
    for (int u = 0; u < PF - 1; u++) NT_LOADW(u < kps ? u : kps - 1, wb[u]);
    const int nchunks = (int)kps / KC;
    slab_load(0);
    slab_store(0);
    lds_barrier();   // LDS only: the rows came from an earlier dispatch
    for (int c = 0; c < nchunks; c++) {
      if (c + 1 < nchunks) slab_load(c + 1);
      const _Float16* const ab_ = abuf + (c & 1) * SLAB;
#pragma unroll
      for (int u = 0; u < KC; u++) {
        const int ks = c * KC + u;
        const int ls = ks + PF - 1 < (int)kps ? ks + PF - 1 : (int)kps - 1;
        NT_LOADW(ls, wb[(u + PF - 1) % PF]);
        v8h af[MT];
#pragma unroll
        for (int t = 0; t < MT; t++) af[t] = frag_lds(ab_ + (16 * (MT * rg + t) + nl) * AP + u * 16 + 4 * h);
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const v8h wf = plan.decode(wb[u % PF][j]);
#pragma unroll
          for (int t = 0; t < MT; t++) acc[t][j] = mma(wf, af[t], acc[t][j]);
        }
      }
      if (c + 1 < nchunks) slab_store((c + 1) & 1);
      lds_barrier();   // LDS only: the rows came from an earlier dispatch
    }
#undef NT_LOADW
  } else {
  // The loads are unconditional (the step index clamps to the last one) -- a branch around the prefetch
  // serialised it with the math.
  unsigned wb[PF][NPW][2][NW];
  v8h ab[PF][MT];
#pragma unroll
  for (int u = 0; u < PF - 1; u++)
    NT_LOAD(u < kps ? u : kps - 1, wb[u], ab[u]);
  int ks = 0;
  for (; ks + PF <= kps; ks += PF) {
#pragma unroll
    for (int u = 0; u < PF; u++) {
      const int ls = ks + u + PF - 1 < kps ? ks + u + PF - 1 : kps - 1;
      NT_LOAD(ls, wb[(u + PF - 1) % PF], ab[(u + PF - 1) % PF]);
      NT_COMPUTE(wb[u], ab[u]);
    }
  }
#pragma unroll
  for (int u = 0; u < PF - 1; u++)
    if (ks + u < kps) NT_COMPUTE(wb[u], ab[u]);
  }
#undef NT_LOAD
#undef NT_COMPUTE

  if constexpr (DOT) {
    // column n's sum = lane n's + lane n + 16's; row 0's columns 8 h + e go to lane 16 h's element e (other rows: zero,
    // never stored)
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const float c = dacc[j] + __shfl_xor(dacc[j], 16, WAVE32);
#pragma unroll
      for (int e = 0; e < 8; e++) acc[0][j][e] = __shfl(c, 8 * h + e, WAVE32);
    }
  }
  if constexpr (DQ) {
    const float step = (float)__builtin_bit_cast(_Float16, (unsigned short)EXL3I_I8_STEP);
#pragma unroll
    for (int j = 0; j < NPW; j++) acc[0][j] *= step;
  }
  if constexpr (KS) {
    // this slice's raw sums: lane (row nl, columns 16 tile + 8 h + e of the block), only the run's rows
    const int nrun = r1 - rc, N = w.rows;
    float* const pz = kpart + (size_t)blockIdx.z * nrun * N;
#pragma unroll
    for (int j = 0; j < NPW; j++)
      if (nl < nrun) {
        float* const d = pz + (size_t)nl * N + cb * 128 + 16 * (col_local * NPW + j) + 8 * h;
        *(v4f*)d = (v4f){acc[0][j][0], acc[0][j][1], acc[0][j][2], acc[0][j][3]};
        *(v4f*)(d + 4) = (v4f){acc[0][j][4], acc[0][j][5], acc[0][j][6], acc[0][j][7]};
      }
    __shared__ unsigned s_arrived;
    __threadfence();
    __syncthreads();
    if (tid == 0)
      s_arrived = __hip_atomic_fetch_add(&kctr[cb], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (s_arrived != (unsigned)ksn - 1u) return;   // workgroup-uniform
    if (tid == 0) __hip_atomic_store(&kctr[cb], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __threadfence();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      v8f s = (v8f)(0.f);
      if (nl < nrun)
        for (int z = 0; z < ksn; z++) {
          const float* const d = kpart + ((size_t)z * nrun + nl) * N + cb * 128 + 16 * (col_local * NPW + j) + 8 * h;
          const v4f a = *(const v4f*)d, b = *(const v4f*)(d + 4);
          s += (v8f){a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]};
        }
      acc[0][j] = s;
    }
  }
#pragma unroll
  for (int t = 0; t < MT; t++) {
    if (SK > 1) splitk_reduce<NPW>(acc[t], lds, col_local, ksplit, SK, lane);
    epilogue_block<NPW, 1, false, false, EXL3R4D_NT_EPIC>(p, w, acc[t], col_local, ksplit, lane, cb, row[t],
                                                        in_run[t], true, lds + rg * 4096, nullptr, r16);   // 16 KB a group
  }
}

// One workgroup = (128-column block blockIdx.x, MT 16-row tiles from blockIdx.y * 16 MT): one pass, its own rows.
// Column blocks fastest: the resident workgroups share one row block's activations. REFUTED, the swap (row blocks
// fastest, to share a column block's weights): 2k prefill 2454 -> 3590 ms, ffn_down (K 17408, a 37 MB
// activation) 4.4 -> 8.1 ms a call.
template <class R, int NPW, int MT, bool PREP, int RG, bool KS = false, bool DOT = false>
__global__ void __launch_bounds__(EXL3R4D_NT_LB) EXL3R4D_NT_WPE_ATTR exl3i_gemm_nt_kernel(Exl3iGemm q) {
  EXL3R4D_KARG_PREFETCH();
  extern __shared__ float lds[];
  const Exl3r4dGemm& p = q.g;
  const Planes w = planes_of(p, 0);   // the one weight
  const int rc = (int)blockIdx.y * EXL3_BLOCK * MT * RG;
  if (rc < p.T)
    nt_pass<R, NPW, MT, PREP, RG, KS, DOT>(p, w, (int)blockIdx.x, rc, min(rc + EXL3_BLOCK * MT * RG, (int)p.T), lds, q.m, q.ksn,
                                      q.part, q.kctr, q.r16);
}

// The prefill form's input rotation, once a row: one wave a (row, 128-feature block, half) task -- whole waves
// (rotate128_math's DPP stages need every lane), the region [T][2K] as the GEMM reads it (half 1 only with suh.1).
// Templated on the codebook only so the codebook's kernels are distinct (the body does not read it).
template <int CB>
__global__ void __launch_bounds__(256) exl3i_nt_prep_kernel(Exl3r4dGemm p) {
  const Planes w = planes_of(p, 0);
  const int lane = (int)threadIdx.x % WAVE32;
  const int task = (int)blockIdx.x * (256 / WAVE32) + (int)threadIdx.x / WAVE32;
  const int blocks = w.cols / 128, mats = w.suh1 != w.suh ? 2 : 1;
  if (task >= p.T * blocks * mats) return;
  const int mat = task % mats, rest = task / mats, bk = rest % blocks, row = rest / blocks;
  typedef const __attribute__((address_space(1))) v2u* gv2;
  const gu8 xb = (gu8)(const unsigned char*)p.x;
  const gu8 sx = (gu8)(const unsigned char*)(mat ? w.suh1 : w.suh);
  const v2u bx = *(gv2)(xb + ((size_t)row * p.x_pitch + bk * 128 + 4 * lane) * 2);
  const v2u bs = *(gv2)(sx + (size_t)(bk * 128 + 4 * lane) * 2);
  *(v2u*)((_Float16*)p.xr + (size_t)row * p.xr_pitch + (size_t)mat * w.cols + bk * 128 + 4 * lane) =
      rotate128_math(bx, bs, p.x_bf16 == 0, lane);
}

// ============================================================================= T2: the two-pass prefill form
// Pass 1 (wdec): every 16 x 16 tile of a column chunk of the weight decoded ONCE into its WMMA fragment order,
// [ksteps][chunk tiles][32 lanes] v8h (the fragment's bits), in the op's scratch. Pass 2 (t2 GEMM): a workgroup is
// 128 rows x 256 columns, 8 waves (2 along M x 4 along N), each 64 rows x 64 columns (MT 4 x NPW 4 WMMA tiles):
// the rotated activation rows (the prep kernel's region) staged through LDS in fragment order (one 16-B read a
// fragment), the decoded weight fragments read straight from global a chunk (KC steps) ahead into registers, one
// barrier a chunk; row blocks grouped (GROUP) so the workgroups in flight share a column block's weight in L2.
// Per output: one WMMA a 16 k in ascending order from zero, the same fragments as nt_pass; the epilogue is
// epilogue_block with nt_pass's EPIC sum order, so the output is BYTE-IDENTICAL to nt_pass (checked: ntbench
// out-hash at 600..2048 rows, every K, gate_up/down/GDN/head shapes; KLD scores byte-identical at TP1 and TP2, and on a 64k-token
// document). Below EXL3R4D_T2_MIN rows nt_pass is faster (the whole weight's decode is a fixed
// cost; crossover ~450-512 rows). EXL3R4D_T2_OFF=1 (env) forces nt_pass.
// gfx1201: gate_up 2048x34816x5120 k4 8.08 -> 6.74 ms, down 4.33 -> 3.65, head 248320 rows 65.9 -> 48.1; the GEMM core
// alone ~130-137 TFLOPS (85% of the card's sustained f16 WMMA peak, 158). Standalone core: kldtool/bench/tlgemm.hip.
#ifndef EXL3R4D_T2
#define EXL3R4D_T2 1
#endif
#define T2_MT 4
#define T2_NPW 4
#define T2_WM 2
#define T2_WN 4
#define T2_KC 2
#define T2_GROUP 8
#define T2_BM (16 * T2_MT * T2_WM)
#define T2_BN (16 * T2_NPW * T2_WN)
#define T2_CHUNK_BYTES EXL3R4D_T2_CHUNK_BYTES   // the decoded weight's scratch: a column chunk of at most this
#if EXL3R4D_NCLS == 2 && !EXL3R4D_GD4
// p.raw: the decoded chunk; p.n_expert (unused by the dense op): the chunk's first column; p.top_k: its columns.
template <class R>
__global__ void __launch_bounds__(256) exl3i_wdec_kernel(Exl3iGemm q) {
  const Exl3r4dGemm& p = q.g;
  const Planes w = planes_of(p, 0);
  constexpr int NW = R::NW;
  const int WORDS = R::words(q.m);
  const int lane = (int)threadIdx.x % WAVE32;
  const int n0 = p.n_expert, ntc = p.top_k / 16, ntiles = w.rows / 16;
  const size_t tile = (size_t)blockIdx.x * 8 + threadIdx.x / WAVE32;
  if (tile >= (size_t)ntc * (w.cols / 16)) return;
  const size_t ks = tile / ntc, nt = tile % ntc;
  const R plan(lane, q.m);
  const gu8 tb = (gu8)(const unsigned char*)w.codes + (ks * ntiles + n0 / 16 + nt) * WORDS * 4;
  unsigned wd[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wd[r][x] = *(gu32)(tb + 4u * (unsigned)plan.woff[r][x]);
  ((v8h*)p.raw)[tile * WAVE32 + lane] = plan.decode(wd);
}

template <int CB>   // as exl3i_nt_prep_kernel: one instance a translation unit
__global__ void __launch_bounds__(256) exl3i_gemm_t2_kernel(Exl3iGemm q) {
  const Exl3r4dGemm& p = q.g;
  extern __shared__ float lds[];
  const Planes w = planes_of(p, 0);
  constexpr int MT = T2_MT, NPW = T2_NPW, WN = T2_WN, KC = T2_KC, BM = T2_BM, BN = T2_BN, NTH = 256;
  constexpr int RT = BM / 16, ASLAB = RT * KC * WAVE32 * 8;   // halfs, [RT][KC][32] v8h
  constexpr int NLA = BM * KC * 2 / NTH;
  static_assert(BM * KC * 2 % NTH == 0, "every thread loads the same number of activation chunks");
  const int tid = (int)threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  const int wm = wave / WN, wn = wave % WN;
  const int M = p.T, n0 = p.n_expert, NC = p.top_k, ntc = NC / 16;
  const int nmb = (M + BM - 1) / BM, nnb = NC / BN, lin = (int)blockIdx.x;
  const int per = T2_GROUP * nnb, first = lin / per * T2_GROUP, r0 = lin % per;
  const int gm = min(nmb - first, T2_GROUP);
  const int mb = first + r0 % gm, nb = r0 / gm;
  const int m0 = mb * BM, ntile0 = nb * (BN / 16), ksteps = w.cols / 16;
  const int width = w.rows;
  const int col0 = n0 + nb * BN;                       // this workgroup's first output column
  const int mat_a = (w.suh1 != w.suh && col0 >= width / 2) ? 1 : 0;
  const _Float16* const xrow = (const _Float16*)p.xr + (size_t)mat_a * w.cols;
  const v8h* const wdec = (const v8h*)p.raw;
  _Float16* const abuf = (_Float16*)lds;
  v4u sa[NLA];
  auto load = [&](int c) {
#pragma unroll
    for (int i = 0; i < NLA; i++) {
      const int idx = tid + NTH * i, r = idx / (2 * KC), q = idx % (2 * KC);
      const int gr = m0 + r < M ? m0 + r : M - 1;
      sa[i] = *(const v4u*)(xrow + (size_t)gr * p.xr_pitch + c * KC * 16 + q * 8);
    }
  };
  auto store = [&](int b) {
#pragma unroll
    for (int i = 0; i < NLA; i++) {
      const int idx = tid + NTH * i, r = idx / (2 * KC), q = idx % (2 * KC);
      {
        const int rt = r / 16, n = r % 16, u = q / 2, qh = q % 2;
        _Float16* const t0 = abuf + b * ASLAB + (rt * KC + u) * WAVE32 * 8 + qh * 4;
        *(v2u*)(t0 + n * 8) = (v2u){sa[i][0], sa[i][1]};
        *(v2u*)(t0 + (n + 16) * 8) = (v2u){sa[i][2], sa[i][3]};
      }
    }
  };
  v8h wq[KC][NPW];
  auto wfetch = [&](int ks, v8h (&o)[NPW]) {
#pragma unroll
    for (int j = 0; j < NPW; j++) o[j] = wdec[((size_t)ks * ntc + ntile0 + wn * NPW + j) * WAVE32 + lane];
  };
  v8f acc[MT][NPW];
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int j = 0; j < NPW; j++) acc[t][j] = (v8f)(0.f);
#pragma unroll
  for (int u = 0; u < KC; u++) wfetch(u, wq[u]);
  const int nch = ksteps / KC;
  // chunk c + 2's rows are loaded at the end of iteration c and stored at the end of c + 1: every wait in the loop
  // targets loads a whole iteration old (gfx12's load counter retires in order). tlgemm.hip: 137 vs 125 TFLOPS.
  load(0);
  store(0);
  load(min(1, nch - 1));
  lds_barrier();
  for (int c = 0; c < nch; c++) {
    const v8h* const ab = (const v8h*)(abuf + (c & 1) * ASLAB);
#pragma unroll
    for (int u = 0; u < KC; u++) {
      v8h af[MT];
#pragma unroll
      for (int t = 0; t < MT; t++) af[t] = ab[((wm * MT + t) * KC + u) * WAVE32 + lane];
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int t = 0; t < MT; t++) acc[t][j] = mma(wq[u][j], af[t], acc[t][j]);
      wfetch(min((c + 1) * KC + u, ksteps - 1), wq[u]);
    }
    __builtin_amdgcn_sched_barrier(0);   // the store stays after the WMMAs (the compiler hoisted it: a full stall)
    store((c + 1) & 1);
    load(min(c + 2, nch - 1));
    lds_barrier();
  }
  // The epilogue: each 128-column block of the workgroup (waves wn 0..1 and 2..3) rotated as nt_pass's, per row group
  const int nl = lane & 15;
#pragma unroll
  for (int t = 0; t < MT; t++) {
    const int row = m0 + 16 * (wm * MT + t) + nl;
    epilogue_block<NPW, 1, false, false, EXL3R4D_NT_EPIC>(p, w, acc[t], wn % 2, 0, lane, col0 / 128 + wn / 2, row, row < M, true,
                                                lds + (wm * 2 + wn / 2) * 4096, nullptr, q.r16);
  }
}
template <class R>
static const void* wdec_fn() { return (const void*)exl3i_wdec_kernel<R>; }
#endif

// KS's arrival counters: a block of EXL3I_KCTR_N a (device, stream, output), zero from module load, each launch's last
// arrivers reset theirs -- so launches in one stream writing one output reuse a block, and two launches that may run at
// once never share one: the engine overlaps launches only where their operands do not overlap (rad_kernel_concurrent),
// so two such launches write two outputs. (Keyed by the stream alone, a GEMV started beside the one before it -- z beside
// qkv, k / v beside qg on mad's xb rows -- counted its arrivals into the other's counters.)
#define EXL3I_KCTR_SLOTS 64
#define EXL3I_KCTR_N 4096
static __device__ unsigned exl3i_kctr[EXL3I_KCTR_SLOTS * EXL3I_KCTR_N];
static unsigned* exl3i_kctr_for(void* stream, const void* key) {
  static std::mutex mu;
  static struct { int dev; void* s; const void* k; unsigned* p; } tab[1024];
  static int n = 0, used[64];   // used[dev]: slots handed out on that device
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) return nullptr;
  std::lock_guard<std::mutex> g(mu);
  for (int i = 0; i < n; i++)
    if (tab[i].dev == dev && tab[i].s == stream && tab[i].k == key) return tab[i].p;
  if (n == 1024 || used[dev] == EXL3I_KCTR_SLOTS) return nullptr;
  void* base = nullptr;
  if (hipGetSymbolAddress(&base, HIP_SYMBOL(exl3i_kctr)) != hipSuccess || !base) return nullptr;
  tab[n] = {dev, stream, key, (unsigned*)base + (size_t)used[dev]++ * EXL3I_KCTR_N};
  return tab[n++].p;
}

template <class R>
static const void* nt_fn(int npw, int mt, bool ks = false, bool dot = false) {
  if (mt == EXL3R4D_NT_MT && EXL3R4D_NT_MT != 1)
    return npw == 2 && !ks && !dot ? (const void*)exl3i_gemm_nt_kernel<R, 2, EXL3R4D_NT_MT, true, EXL3R4D_NT_RG> : nullptr;
  if (mt != 1) return nullptr;
  if (dot) {   // the shipped arm's NPW only
    if (npw != 2) return nullptr;
    return ks ? (const void*)exl3i_gemm_nt_kernel<R, 2, 1, false, 1, true, true>
              : (const void*)exl3i_gemm_nt_kernel<R, 2, 1, false, 1, false, true>;
  }
  if (ks) switch (npw) {
    case 1: return (const void*)exl3i_gemm_nt_kernel<R, 1, 1, false, 1, true>;
    case 2: return (const void*)exl3i_gemm_nt_kernel<R, 2, 1, false, 1, true>;
    case 4: return (const void*)exl3i_gemm_nt_kernel<R, 4, 1, false, 1, true>;
    default: return nullptr;
  }
  switch (npw) {
    case 1: return (const void*)exl3i_gemm_nt_kernel<R, 1, 1, false, 1>;
    case 2: return (const void*)exl3i_gemm_nt_kernel<R, 2, 1, false, 1>;
    case 4: return (const void*)exl3i_gemm_nt_kernel<R, 4, 1, false, 1>;
    default: return nullptr;
  }
}

// The dense linear over one weight at rate policy R (any M, MT tiles past 16 rows, T2 from
// EXL3R4D_T2_MIN rows when the scratch holds the decoded chunk). The nt / wdec kernels take the Exl3iGemm (the rate),
// the prep and T2 GEMM kernels share its block. The test pin EXL3I_PIN_T2_OFF forces nt_pass. RAD_OK or a negative RAD_E_*.
template <class R>
static int exl3i_launch_t(const Exl3iGemm* qa, int npw, void* stream) {
  const Exl3r4dGemm* a = &qa->g;
  if (npw != 1 && npw != 2 && npw != 4) return RAD_E_SHAPE;
  const int sk = a->sk > 0 && a->sk <= 8 ? a->sk : 1;
  if (sk != 1) return RAD_E_UNSUPPORTED;   // first pass: the split-K 1 form (the fit logic is the tuning arm)
  const int wv = 8 / npw;
  const int K = a->cls[0].cols;
  int64_t smem = 8 * 256 * 4;   // the epilogue's tiles (one matrix, split-K 1)
  // MT: one 16-row tile a workgroup at decode, EXL3R4D_NT_MT of them past 16 rows (NPW 2, the shipped arm)
  const int mt = a->T > 16 && npw == 2 ? EXL3R4D_NT_MT : 1;
  const int rg = mt > 1 ? EXL3R4D_NT_RG : 1;
  const int rows_wg = 16 * mt * rg;
  if (a->T < 1 || (a->T + rows_wg - 1) / rows_wg > 65535) return RAD_E_UNSUPPORTED;
  if (K % 128 || a->y_cols % 128) return RAD_E_SHAPE;
  if (wv * sk * WAVE32 * (mt > 1 ? EXL3R4D_NT_RG : 1) > (int)EXL3R4D_NT_LB) return RAD_E_SHAPE;
  if (!a->xr) return RAD_E_INVAL;   // the GSCR region (the row's scratch hook)
  Exl3iGemm q = *qa;
  q.g.sk = sk;
  Exl3r4dGemm& p = q.g;
  void* args[] = {&p};    // the prep and T2 GEMM kernels: the same block
  void* qargs[] = {&q};   // the nt and wdec kernels: the block and the rate
  // KS: the decode form's K split across workgroups (nt.c chose ksn and placed the partial sums)
  bool ks = mt == 1 && a->T <= 16 && qa->ksn > 1 && qa->part && a->y_cols / 128 <= EXL3I_KCTR_N;
  if (ks && !(q.kctr = exl3i_kctr_for(stream, a->y))) ks = false;   // no counter block: the unsplit form
  // DOT: one row (the test pin EXL3I_PIN_DEC_DOT_OFF: the WMMA form)
  const bool dot = !exl3i_pins[EXL3I_PIN_DEC_DOT_OFF] && mt == 1 && a->T == 1 && npw == 2;
  const void* fn = nt_fn<R>(npw, mt, ks, dot);
  if (!fn) return RAD_E_SHAPE;
#if EXL3R4D_T2
  // The two-pass form: needs the decoded chunk's scratch past the GSCR region (exl3i_nt_scratch sizes both).
  // T2 picks a workgroup's activation half once per T2_BN columns, so a stacked weight takes it only with its seam on a
  // block boundary (N/2 % 256; every trunk gate_up has it); otherwise nt_pass, whose seam is per 128 columns (at
  // N 256 stacked T2 read the gate half for the up rows)
  if (a->T >= EXL3R4D_T2_MIN && a->raw && !exl3i_pins[EXL3I_PIN_T2_OFF] && a->y_cols % T2_BN == 0 && K % (16 * T2_KC) == 0 &&
      (!a->cls[0].suh1 || (a->y_cols / 2) % T2_BN == 0)) {
    const int mats = a->cls[0].suh1 ? 2 : 1;
    const long long tasks = (long long)a->T * (K / 128) * mats;
    if (hipLaunchKernel((const void*)exl3i_nt_prep_kernel<R::CBV>, dim3((unsigned)((tasks + 7) / 8)), dim3(256), args, 0,
                        (hipStream_t)stream) != hipSuccess)
      return RAD_E_DEVICE;
    const void* df = wdec_fn<R>();
    int nc = (int)(T2_CHUNK_BYTES / ((long long)K * 2)) / T2_BN * T2_BN;
    if (nc < T2_BN) nc = T2_BN;
    p.raw = a->raw;
    for (int c0 = 0; c0 < a->y_cols; c0 += nc) {
      const int ncols = min(nc, a->y_cols - c0);
      p.n_expert = c0;
      p.top_k = ncols;
      const size_t tiles = (size_t)(ncols / 16) * (K / 16);
      if (hipLaunchKernel(df, dim3((unsigned)((tiles + 7) / 8)), dim3(256), qargs, 0, (hipStream_t)stream) != hipSuccess)
        return RAD_E_DEVICE;
      const unsigned grid = (unsigned)(((a->T + T2_BM - 1) / T2_BM) * (ncols / T2_BN));
      if (hipLaunchKernel((const void*)exl3i_gemm_t2_kernel<R::CBV>, dim3(grid), dim3(256), qargs, (size_t)65536,
                          (hipStream_t)stream) != hipSuccess)
        return RAD_E_DEVICE;
    }
    return RAD_OK;
  }
#endif
  if (mt > 1 && EXL3R4D_NT_ALDS) {   // the slab's two buffers (the epilogue's tiles reuse them)
    const int64_t slab = 2LL * 16 * mt * rg * EXL3R4D_NT_AP * 2;
    if (smem < slab) smem = slab;
    if (smem < 16384LL * rg) smem = 16384LL * rg;   // each row group's epilogue tiles
  }
  if (mt > 1) {   // the prefill form: the rows rotated once, by their own dispatch (stream-ordered before the GEMM)
    const int mats = a->cls[0].suh1 ? 2 : 1;
    const long long tasks = (long long)a->T * (K / 128) * mats;
    if (hipLaunchKernel((const void*)exl3i_nt_prep_kernel<R::CBV>, dim3((unsigned)((tasks + 7) / 8)), dim3(256), args, 0,
                        (hipStream_t)stream) != hipSuccess)
      return RAD_E_DEVICE;
  }
  dim3 grid((unsigned)(a->y_cols / 128), (unsigned)((a->T + rows_wg - 1) / rows_wg), ks ? (unsigned)qa->ksn : 1u);
  dim3 block((unsigned)(wv * sk * WAVE32 * rg));
  if (hipLaunchKernel(fn, grid, block, qargs, (size_t)smem, (hipStream_t)stream) != hipSuccess)
    return RAD_E_DEVICE;
  return RAD_OK;
}
#endif  // EXL3R4D_NCLS == 2 && !EXL3R4D_GD4

// Every rate of codebook CB: integer K through Plan<BITS>, the rest through the plan table.
template <int CB>
static int exl3i_launch_cb(const Exl3iGemm* a, int npw, void* stream) {
  const int m = a->m;
  if (m < 2 || m > 128 || m % 2) return RAD_E_DTYPE;
  if (m % 16 == 0) switch (m / 16) {
    case 1: return exl3i_launch_t<RateInt<1, CB>>(a, npw, stream);
    case 2: return exl3i_launch_t<RateInt<2, CB>>(a, npw, stream);
    case 3: return exl3i_launch_t<RateInt<3, CB>>(a, npw, stream);
    case 4: return exl3i_launch_t<RateInt<4, CB>>(a, npw, stream);
    case 5: return exl3i_launch_t<RateInt<5, CB>>(a, npw, stream);
    case 6: return exl3i_launch_t<RateInt<6, CB>>(a, npw, stream);
    case 7: return exl3i_launch_t<RateInt<7, CB>>(a, npw, stream);
    default: return exl3i_launch_t<RateInt<8, CB>>(a, npw, stream);
  }
  if (m < 16) return exl3i_launch_t<RateFrac<CB, 1>>(a, npw, stream);
  if (m < 80) return exl3i_launch_t<RateFrac<CB, 0>>(a, npw, stream);
  return exl3i_launch_t<RateFrac<CB, 2>>(a, npw, stream);
}
