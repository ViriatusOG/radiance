// exl3i_mad.hip.h -- the byte-sum GEMV: exl3i_dense on a mad weight with an int8 activation mode
// (a8g128) at 1 .. 8 rows, the decode band of a step. mad's value is q = bytesum(h) - 126 with h = (s A + B) &
// 0x3F3F3F3F the state's masked hash (exl3i_decode.h), so a weight times an int8 activation x is
//     sudot4(h, x * 0x01010101) - 126 x
// -- one instruction a weight a row, no value unpacked, no WMMA (IQR's form: libr4d r4d_gemm_iqr.hip iqr_gemv_kernel,
// its UNS path). The -126 x terms are the rows' own: their sum over a lane's columns of a 128-feature block is one
// integer a (row, block, lane half) the quantiser writes once, so the K loop never sees it.
//
// THE ROWS, quantised as exl3i_a8_actq does at a8g128 (rotate128_math with suh / suh.1, the wave's absmax, exl3i_a8_q:
// the A8 path's int8 bytes and f32 scales, bit for bit) once a step, by the producer (AQ, exl3i_actq form xb,
// exl3i_abi.h version 4) or by this launch's pre-pass (exl3i_mad_rows_kernel) into the op's scratch; per row and matrix
// (EXL3I_MAD_ROW_BYTES(K), 256-B aligned):
//     xb   int32 [K]          x_c * 0x01010101 (the byte in every lane of the dword: sudot4's broadcast operand)
//     sa   f32   [K / 128]    the block's scale
//     bias int32 [K / 128][2] 126 x (sum of the block's columns of lane half h: k % 16 in 4h .. 4h + 3, 8 + 4h ..)
// THE WEIGHT. exl3i's stored tiles and nt_pass's lane plan (exl3i_rate.hip.h madh: lane (n = lane % 16, h) decodes the
// eight positions of the decode fragment, column n, k = exl3i_frag16_k(lane, e)), the same buffer loads; a wave
// NPW tiles (16 NPW output columns) of the workgroup's 128, the K loop over its slice's 128-feature blocks (KS: the K
// split across workgroups, nt_pass's scheme and counters). Per block and row: an exact int32 sum a lane (its 64
// columns), minus its bias, folded into f32 with the row's scale; the lane pair added, x the grid step (4 K_INV),
// then nt_pass's epilogue (the 128-point output rotation, 1 / sqrt(128), svh; R16 rows) on the WMMA layout the sums
// are moved into. The int sums are the A8 tile's integers; the f32 fold is a lane half at a time, so y is the A8
// path's within f32 rounding, not its bytes.
#pragma once
#include "exl3i_nt.hip.h"
#include "exl3i_a8q.hip.h"

#ifndef EXL3I_MAD_PF   // weight load stages in flight (a divisor of 8: a block's eight k steps). 2: at 4 the registers cost
#define EXL3I_MAD_PF 2  // more than the depth gains above K2
#endif
#ifndef EXL3I_MAD_NPW  // column tiles a wave
#define EXL3I_MAD_NPW 2
#endif
static_assert(8 % EXL3I_MAD_PF == 0, "EXL3I_MAD_PF divides a block's 8 k steps");

template <class R, int NPW, int TT, bool KS>
__global__ void __launch_bounds__(256) exl3i_mad_gemv_kernel(Exl3iGemm q) {
  static_assert(R::CBV == EXL3I_DEV_CB_MAD, "the byte-sum GEMV is mad's");
  constexpr int NW = R::NW, WV = 8 / NPW, PF = EXL3I_MAD_PF;
#ifdef EXL3I_MAD_PFX
  constexpr int PFX = EXL3I_MAD_PFX;
#else
  constexpr int PFX = TT <= 2 ? PF : 1;   // activation stages: with the weights' at 1 / 2 rows, else at use
#endif
  extern __shared__ float lds[];
  const Exl3r4dGemm& p = q.g;
  const Planes w = planes_of(p, 0);
  const int WORDS = R::words(q.m);
  const int tid = (int)threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  const int h = lane >> 4, nl = lane & 15;
  const int cb = (int)blockIdx.x, T = p.T;
  const int N = w.rows, K = w.cols, ntiles = N / 16, nkb = K / 128;
  const int mat_a = (w.suh1 != w.suh && cb * 128 >= N / 2) ? 1 : 0;
  int kb0 = 0, kb1 = nkb;
  if constexpr (KS) {
    const int z = (int)blockIdx.z;
    kb0 = nkb * z / q.ksn;
    kb1 = nkb * (z + 1) / q.ksn;
  }
  // The rows' int8 form, row t at xq + t xq_ld, this matrix's plane xq_plane + mat: the producer's (AQ, exl3i_actq form
  // xb) or the launch's pre-pass (exl3i_mad_rows_kernel). Not quantised in the workgroup: compiled in, that prologue held
  // the K loop's registers for work every workgroup repeated.
  const size_t RB = (size_t)EXL3I_MAD_ROW_BYTES(K);
  unsigned char* const reg = (unsigned char*)q.xq + (size_t)(q.xq_plane + mat_a) * RB;
  const size_t xld = (size_t)q.xq_ld;

  // ---- the weight's plan and its first load stages (they do not wait for the rows)
  const R plan(lane, q.m);
  unsigned wvo[2][NW];
#pragma unroll
  for (int r = 0; r < 2; r++)
#pragma unroll
    for (int x = 0; x < NW; x++) wvo[r][x] = 4u * (unsigned)plan.woff[r][x];
  const int nt0 = cb * 8 + wave * NPW;
  const size_t tile0 = (size_t)kb0 * 8 * ntiles + nt0;
  const __amdgpu_buffer_rsrc_t wrs = __builtin_amdgcn_make_buffer_rsrc((void*)w.codes, (short)0, 0x7fffffff, 0x31004000);
  const __amdgpu_buffer_rsrc_t xrs = __builtin_amdgcn_make_buffer_rsrc((void*)reg, (short)0, 0x7fffffff, 0x31004000);
  const unsigned bso = (unsigned)(tile0 * WORDS * 4), kbytes = (unsigned)ntiles * WORDS * 4, jbytes = WORDS * 4;
  [[maybe_unused]] const unsigned bwm = wvo[0][0], bw0 = wvo[0][1], bwp = wvo[1][NW - 1];
  constexpr int kAux = EXL3R4D_NT ? 1 : 0;
  const int kps = (kb1 - kb0) * 8;
  auto loadw = [&](int ks, unsigned (&W)[NPW][2][NW]) {
    const unsigned so = bso + (unsigned)ks * kbytes;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      if constexpr (R::K4) {
        const unsigned m_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwm + j * jbytes, so, kAux);
        const unsigned z_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bw0 + j * jbytes, so, kAux);
        const unsigned q_ = __builtin_amdgcn_raw_buffer_load_b32(wrs, bwp + j * jbytes, so, kAux);
        W[j][0][0] = m_; W[j][0][1] = z_; W[j][1][0] = z_; W[j][1][1] = q_;
      } else {
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
          for (int x = 0; x < NW; x++)
            W[j][r][x] = __builtin_amdgcn_raw_buffer_load_b32(wrs, wvo[r][x] + j * jbytes, so, kAux);
      }
    }
  };
  unsigned wb[PF][NPW][2][NW];
#pragma unroll
  for (int u = 0; u < PF - 1; u++) loadw(u < kps ? u : kps - 1, wb[u]);

  // ---- the K loop
  // the rows' offsets (a row past T reads row 0: valid memory, never stored)
  unsigned xo[TT];
#pragma unroll
  for (int t = 0; t < TT; t++) xo[t] = (unsigned)((t < T ? t : 0) * xld + (size_t)(kb0 * 128 + 4 * h) * 4);

  auto loadx = [&](int ks, int (&X)[TT][8]) {
#pragma unroll
    for (int t = 0; t < TT; t++) {
      const v4u a = __builtin_bit_cast(v4u, __builtin_amdgcn_raw_buffer_load_b128(xrs, xo[t], (unsigned)ks * 64u, 0));
      const v4u b = __builtin_bit_cast(v4u, __builtin_amdgcn_raw_buffer_load_b128(xrs, xo[t] + 32u, (unsigned)ks * 64u, 0));
#pragma unroll
      for (int e = 0; e < 4; e++) { X[t][e] = (int)a[e]; X[t][4 + e] = (int)b[e]; }
    }
  };

  int xr[PFX][TT][8];
#pragma unroll
  for (int u = 0; u < PFX - 1; u++) loadx(u < kps ? u : kps - 1, xr[u]);
  float accf[NPW][TT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int t = 0; t < TT; t++) accf[j][t] = 0.f;
  const float* const sa0 = (const float*)(reg + (size_t)4 * K);
  const int* const bs0 = (const int*)(reg + (size_t)4 * K + 4 * nkb);
  for (int g = 0; g < kb1 - kb0; g++) {
    float sa[TT];
    int bs[TT];
#pragma unroll
    for (int t = 0; t < TT; t++) {
      const size_t ro = (size_t)(t < T ? t : 0) * xld / 4;
      sa[t] = sa0[ro + kb0 + g];
      bs[t] = bs0[ro + 2 * (kb0 + g) + h];
    }
    int ai[NPW][TT];
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
      for (int t = 0; t < TT; t++) ai[j][t] = 0;
#pragma unroll
    for (int u = 0; u < 8; u++) {
      const int ks = g * 8 + u;
      loadw(ks + PF - 1 < kps ? ks + PF - 1 : kps - 1, wb[(u + PF - 1) % PF]);
      if constexpr (PFX > 1) loadx(ks + PFX - 1 < kps ? ks + PFX - 1 : kps - 1, xr[(u + PFX - 1) % PFX]);
      else loadx(ks, xr[0]);
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        unsigned hh[8];
        plan.madh(wb[u % PF][j], hh);
#pragma unroll
        for (int t = 0; t < TT; t++)
#pragma unroll
          for (int e = 0; e < 8; e++)
            ai[j][t] = __builtin_amdgcn_sudot4(false, (int)hh[e], true, xr[u % PFX][t][e], ai[j][t], false);
      }
    }
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
      for (int t = 0; t < TT; t++) accf[j][t] = fmaf((float)(ai[j][t] - bs[t]), sa[t], accf[j][t]);
  }

  // ---- the lane pair's halves, x the step, into the WMMA accumulator layout (lane: row l % 16, columns 8 (l / 16) + e)
  const float step = (float)__builtin_bit_cast(_Float16, (unsigned short)EXL3I_I8_STEP);
  float* const tr = lds + 2048 + wave * (NPW * TT * 16);   // [NPW][TT][16] past the epilogue's tiles
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int t = 0; t < TT; t++) {
      const float c = (accf[j][t] + __shfl_xor(accf[j][t], 16, WAVE32)) * step;
      if (h == 0) tr[(j * TT + t) * 16 + nl] = c;
    }
  __syncthreads();
  v8f acc[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int e = 0; e < 8; e++) acc[j][e] = nl < TT ? tr[(j * TT + (nl < TT ? nl : 0)) * 16 + 8 * h + e] : 0.f;
  __syncthreads();

  if constexpr (KS) {   // nt_pass's: this slice's sums, the last arriver of the column block sums the slices in order
    const int nrun = T;
    float* const pz = q.part + (size_t)blockIdx.z * nrun * N;
#pragma unroll
    for (int j = 0; j < NPW; j++)
      if (nl < nrun) {
        float* const d = pz + (size_t)nl * N + cb * 128 + 16 * (wave * NPW + j) + 8 * h;
        *(v4f*)d = (v4f){acc[j][0], acc[j][1], acc[j][2], acc[j][3]};
        *(v4f*)(d + 4) = (v4f){acc[j][4], acc[j][5], acc[j][6], acc[j][7]};
      }
    __shared__ unsigned s_arrived;
    __threadfence();
    __syncthreads();
    if (tid == 0)
      s_arrived = __hip_atomic_fetch_add(&q.kctr[cb], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (s_arrived != (unsigned)q.ksn - 1u) return;   // workgroup-uniform
    if (tid == 0) __hip_atomic_store(&q.kctr[cb], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __threadfence();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      v8f s = (v8f)(0.f);
      if (nl < nrun)
        for (int z = 0; z < q.ksn; z++) {
          const float* const d = q.part + ((size_t)z * nrun + nl) * N + cb * 128 + 16 * (wave * NPW + j) + 8 * h;
          const v4f a = *(const v4f*)d, b = *(const v4f*)(d + 4);
          s += (v8f){a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]};
        }
      acc[j] = s;
    }
  }
  [[maybe_unused]] v8h svp[epi_opw<NPW, 1>()][1];
  epilogue_block<NPW, 1, false, false, EXL3R4D_NT_EPIC>(p, w, acc, wave, 0, lane, cb, nl, nl < T, true, lds, svp, q.r16);
}

// The rows quantised once, before the GEMV when no producer wrote them (AQ form xb): one wave a (matrix,
// row, 128-feature block), exl3i_a8_actq's arithmetic, into the scratch laid out as AQ's form xb rows
// (row t at xr + t mats RB, matrix i's plane at i RB), which the GEMV then reads as q.xq.
__global__ void __launch_bounds__(256) exl3i_mad_rows_kernel(Exl3iGemm q) {
  const Exl3r4dGemm& p = q.g;
  const Planes w = planes_of(p, 0);
  const int lane = (int)threadIdx.x % WAVE32;
  const int K = w.cols, nkb = K / 128, mats = w.suh1 != w.suh ? 2 : 1;
  const long long task = (long long)blockIdx.x * 8 + (int)threadIdx.x / WAVE32;
  if (task >= (long long)mats * p.T * nkb) return;   // wave-uniform
  const int kb = (int)(task % nkb), t = (int)(task / nkb % p.T), mat = (int)(task / nkb / p.T);
  typedef const __attribute__((address_space(1))) v2u* gv2;
  const v2u xv = *(gv2)((gu8)(const unsigned char*)p.x + ((size_t)t * p.x_pitch + kb * 128 + 4 * lane) * 2);
  const v2u sv = *(gv2)((gu8)(const unsigned char*)(mat ? w.suh1 : w.suh) + (size_t)(kb * 128 + 4 * lane) * 2);
  const v2u v = rotate128_math(xv, sv, p.x_bf16 == 0, lane);
  float m = 0.0f;
  m = fmaxf(m, amax4(v));
  m = wave_max(m);
  const float inv = exl3i_a8_inv(m), sc = exl3i_a8_scale(m);
  const unsigned q4 = quant4(v, inv);
  const size_t RB = (size_t)EXL3I_MAD_ROW_BYTES(K);
  unsigned char* const rb = (unsigned char*)p.xr + ((size_t)t * mats + mat) * RB;
  v4u o;
#pragma unroll
  for (int e = 0; e < 4; e++) o[e] = ((q4 >> (8 * e)) & 0xFFu) * 0x01010101u;
  *(v4u*)(rb + (size_t)(kb * 128 + 4 * lane) * 4) = o;
  int s4 = __builtin_amdgcn_sudot4(true, (int)q4, true, 0x01010101, 0, false);
  s4 += __shfl_xor(s4, 2, WAVE32);
  s4 += __shfl_xor(s4, 4, WAVE32);
  s4 += __shfl_xor(s4, 8, WAVE32);
  s4 += __shfl_xor(s4, 16, WAVE32);
  if (lane == 0) ((float*)(rb + (size_t)4 * K))[kb] = sc;
  if (lane < 2) ((int*)(rb + (size_t)4 * K + 4 * nkb))[2 * kb + lane] = 126 * s4;
}


template <class R, int TT>
static const void* mad_fn(int npw, bool ks) {
  if (npw != EXL3I_MAD_NPW) return nullptr;
  return ks ? (const void*)exl3i_mad_gemv_kernel<R, EXL3I_MAD_NPW, TT, true>
            : (const void*)exl3i_mad_gemv_kernel<R, EXL3I_MAD_NPW, TT, false>;
}

template <class R>
static int exl3i_mad_launch_t(const Exl3iGemm* qa, int npw, void* stream) {
  const Exl3r4dGemm* a = &qa->g;
  const int T = a->T;
  if (T < 1 || T > 8) return RAD_E_SHAPE;
  if (a->cls[0].cols % 128 || a->y_cols % 128 || !a->xr) return RAD_E_SHAPE;
  Exl3iGemm q = *qa;
  bool ks = qa->ksn > 1 && qa->part && a->y_cols / 128 <= EXL3I_KCTR_N;
  if (ks && !(q.kctr = exl3i_kctr_for(stream, a->y))) ks = false;
  if (!ks) q.ksn = 1;
  const int TT = T <= 1 ? 1 : T <= 2 ? 2 : T <= 4 ? 4 : 8;
  const void* fn = TT == 1 ? mad_fn<R, 1>(npw, ks) : TT == 2 ? mad_fn<R, 2>(npw, ks)
                 : TT == 4 ? mad_fn<R, 4>(npw, ks) : mad_fn<R, 8>(npw, ks);
  if (!fn) return RAD_E_SHAPE;
  if (!q.xq) {   // no producer wrote the rows: the pre-pass quantises them once into the scratch
    {
      const int mats = a->cls[0].suh1 ? 2 : 1, nkb = a->cls[0].cols / 128;
      const long long tasks = (long long)mats * T * nkb;
      void* pargs[] = {&q};
      if (hipLaunchKernel((const void*)exl3i_mad_rows_kernel, dim3((unsigned)((tasks + 7) / 8)), dim3(256), pargs, 0,
                          (hipStream_t)stream) != hipSuccess)
        return RAD_E_DEVICE;
      q.xq = (const unsigned char*)a->xr;
      q.xq_ld = (int64_t)mats * EXL3I_MAD_ROW_BYTES(a->cls[0].cols);
      q.xq_plane = 0;
    }
  }
  void* qargs[] = {&q};
  dim3 grid((unsigned)(a->y_cols / 128), 1u, ks ? (unsigned)q.ksn : 1u);
  dim3 block((unsigned)(8 / npw * WAVE32));
  return hipLaunchKernel(fn, grid, block, qargs, (size_t)16384, (hipStream_t)stream) == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

// Every rate of mad: integer K through Plan<BITS>, the rest through the plan table (exl3i_launch_cb's split).
static int exl3i_mad_launch(const Exl3iGemm* a, int npw, void* stream) {
  constexpr int CB = EXL3I_DEV_CB_MAD;
  const int m = a->m;
  if (m < 2 || m > 128 || m % 2) return RAD_E_DTYPE;
  if (m % 16 == 0) switch (m / 16) {
    case 1: return exl3i_mad_launch_t<RateInt<1, CB>>(a, npw, stream);
    case 2: return exl3i_mad_launch_t<RateInt<2, CB>>(a, npw, stream);
    case 3: return exl3i_mad_launch_t<RateInt<3, CB>>(a, npw, stream);
    case 4: return exl3i_mad_launch_t<RateInt<4, CB>>(a, npw, stream);
    case 5: return exl3i_mad_launch_t<RateInt<5, CB>>(a, npw, stream);
    case 6: return exl3i_mad_launch_t<RateInt<6, CB>>(a, npw, stream);
    case 7: return exl3i_mad_launch_t<RateInt<7, CB>>(a, npw, stream);
    default: return exl3i_mad_launch_t<RateInt<8, CB>>(a, npw, stream);
  }
  if (m < 16) return exl3i_mad_launch_t<RateFrac<CB, 1>>(a, npw, stream);
  if (m < 80) return exl3i_mad_launch_t<RateFrac<CB, 0>>(a, npw, stream);
  return exl3i_mad_launch_t<RateFrac<CB, 2>>(a, npw, stream);
}
