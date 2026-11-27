// exl3i_rerank.hip -- exl3i_rerank's device row (include/exl3i_rerank.h): the drafter's candidates scored against the
// exl3i head where exl3i_dense's layout put it (k-major tiles), then re-sorted. Two launches:
//
//   rr_score  a workgroup (256 threads) a (candidate r, row m). The row's activation rotated into LDS as f16 (each wave
//             a 128-feature block at a time, four features a lane: rotate128's arithmetic, its stages by shuffles);
//             then the candidate's 128-row output-Hadamard block: per 128-feature chunk its 8 strips x 8 tiles staged
//             in LDS (an odd tile stride), thread (row i, half h) decoding the 16 k positions of its row in 4 of the 8
//             tiles from the ring (exl3i_decode.h's windows) and fma-ing them against xr; the two halves summed, x the
//             grid step for mad (its values are the grid indices q, as nt_pass's DQ); one thread folds the
//             128 sums by the cone of element j and writes ((. x 1/sqrt(128)) x svh) to val[m, r]
//   rr_sort   a workgroup a row: every slot's rank (larger value first, ties to the lower id, then the earlier slot),
//             read whole before anything is written (idx may be idx_in), pads last
//
// The arithmetic and order are exl3i_rerank.h's; rerank_ref.c computes the same bits on the host. The one codebook (mad):
// q = bytesum((state A + B) & 0x3F3F3F3F) - 126, the grid index, x the grid step after the sum.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

#include "exl3i_rerank_dev.h"
#include "exl3i_rerank.h"
#include "exl3i_decode.h"
#include "rad_abi.h"

namespace {

constexpr float kRS = 0.08838834764831845f;   // 1 / sqrt(128)

__device__ __forceinline__ float h2f(uint16_t b) { return (float)__builtin_bit_cast(_Float16, b); }
__device__ __forceinline__ uint16_t f2h(float f) { return __builtin_bit_cast(uint16_t, (_Float16)f); }

// exl3_tile_to_pos: tile row r (k axis), column c (n axis) -> ring position
__device__ __forceinline__ int tile_pos(int r, int c) {
  const int lane = 4 * (c % 8) + (r % 8) / 2;
  const int j = 4 * (c / 8) + ((r & 1) | ((r >> 3) << 1));
  return 8 * lane + j;
}

// A state's value: mad's grid index q (the int8 grid; the step is applied after the sum).
template <int CB>
__device__ __forceinline__ float value_of(unsigned state) {
  static_assert(CB == EXL3I_CB_MAD, "the rerank row serves mad");
  const unsigned h = (state * EXL3I_MAD_A + EXL3I_MAD_B) & 0x3F3F3F3Fu;
  return (float)((int)__builtin_amdgcn_sad_u8(h, 0u, 0u) - 126);
}

__device__ __forceinline__ bool rr_valid(int64_t g, const Exl3iRerankCall& c) {
  const int64_t l = g - c.voff;
  return g >= 0 && l >= 0 && l < c.N;
}

// Row mi's rotated activation into xr (f16 bits, K): wave w takes blocks w, w + 8, ..; lane holds features 4 lane ..
// 4 lane + 3 (rotate128_math's arithmetic: f16(x suh), the stages h = 1, 2 in the lane, 4 .. 64 by shuffles, f16(v RS)).
__device__ __forceinline__ void rr_rotate(const Exl3iRerankCall& c, int64_t mi, uint16_t* xr, int lane, int wave) {
  const unsigned short* xb = (const unsigned short*)c.x + mi * c.x_pitch;
  for (int64_t blk = wave; blk < c.K / 128; blk += 8) {
    float v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int64_t k = blk * 128 + 4 * lane + q;
      const unsigned short xv16 = xb[k];
      const float xv = c.x_f16 ? h2f(xv16) : __builtin_bit_cast(float, (unsigned)xv16 << 16);
      v[q] = h2f(f2h(__fmul_rn(xv, h2f(c.suh[k]))));
    }
    {
      const float a = v[0], b = v[1], cc = v[2], d = v[3];
      v[0] = __fadd_rn(a, b); v[1] = __fsub_rn(a, b); v[2] = __fadd_rn(cc, d); v[3] = __fsub_rn(cc, d);
      const float e0 = v[0], e1 = v[1];
      v[0] = __fadd_rn(e0, v[2]); v[1] = __fadd_rn(e1, v[3]); v[2] = __fsub_rn(e0, v[2]); v[3] = __fsub_rn(e1, v[3]);
    }
#pragma unroll
    for (int sb = 1; sb < 32; sb <<= 1) {   // h = 4 sb: the partner lane ^ sb; the upper takes partner - mine
      const bool hi = (lane & sb) != 0;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float o = __shfl_xor(v[q], sb, 32);
        v[q] = hi ? __fsub_rn(o, v[q]) : __fadd_rn(v[q], o);
      }
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) xr[blk * 128 + 4 * lane + q] = f2h(__fmul_rn(v[q], kRS));
  }
}

// The rotation alone, row blockIdx.x into out [M][K] (tests: device == host rotation).
__global__ __launch_bounds__(256) void rr_rotate_only(Exl3iRerankCall c, uint16_t* out) {
  const int t = (int)threadIdx.x;
  rr_rotate(c, blockIdx.x, out + (int64_t)blockIdx.x * c.K, t & 31, t >> 5);
}

template <int CB>
__global__ __launch_bounds__(256) void rr_score(Exl3iRerankCall c) {
  extern __shared__ unsigned char lds_raw[];
  const int t = (int)threadIdx.x, lane = t & 31, wave = t >> 5;
  const int64_t r = blockIdx.x, mi = blockIdx.y;
  const int64_t g = c.idx_in[mi * c.in_pitch + r];
  if (!rr_valid(g, c)) {
    if (t == 0) c.val[mi * c.val_pitch + r] = -INFINITY;
    return;
  }
  const int W = c.m >> 1, TW = W + 1;
  const int64_t K = c.K, nt = c.N / 16;
  // LDS: xr f16 [K] | words u32 [64 TW] | part f32 [128]
  uint16_t* xr = (uint16_t*)lds_raw;
  unsigned* words = (unsigned*)(lds_raw + ((K * 2 + 15) / 16) * 16);
  float* part = (float*)(words + 64 * TW);

  rr_rotate(c, mi, xr, lane, wave);

  const int64_t s = exl3i_rerank_row(g - c.voff, c.N, c.r16), b = s >> 7;
  const int j = (int)(s & 127);
  const int i = t & 127, hf = t >> 7, st = i >> 4, ct = i & 15;
  // this thread's 16 ring windows (tile rows rt = 0 .. 15 at column ct): exl3i_decode_tile's S, start, salt
  const int ka = c.m >> 4, nb = c.m & 15, RB = 16 * c.m;
  // the window as one funnel shift: state = ({word wa, word wb} >> sh) & 0xFFFF with sh = 48 - start % 32 in 17 .. 48;
  // from 32 the window lies in word wa alone (the high word then does not reach the low 16 bits), so (wa, wa) and sh - 32
  int wa[16], wb[16];
  unsigned sh[16], salt[16];
#pragma unroll
  for (int rt = 0; rt < 16; ++rt) {
    const int p = tile_pos(rt, ct), q = p & 15;
    const int S = ka * (p + 1) + (p >> 4) * nb + (((q + 1) * nb) >> 4);
    int start = S - 16;
    if (start < 0) start += RB;
    const int a = start >> 5;
    const unsigned s48 = 48u - (unsigned)(start & 31);
    wa[rt] = a;
    wb[rt] = s48 >= 32u ? a : (a + 1 == W ? 0 : a + 1);
    sh[rt] = s48 >= 32u ? s48 - 32u : s48;
    const int d = ka + ((((q + 1) * nb) >> 4) - ((q * nb) >> 4));
    salt[rt] = d ? 0u : (0x9E37u * (unsigned)(q + 1)) & 0xFFFFu;
  }
  float acc = 0.0f;
  const int sw = 8 * W;   // one k step's 8 strips, contiguous in the k-major form
  for (int64_t cc = 0; cc < K / 128; ++cc) {
    __syncthreads();   // the previous chunk's words read (and, the first time, xr written)
    for (int e = t; e < 8 * sw; e += 256) {   // (W is loop-invariant: the divisions' reciprocals are hoisted)
      const int tkl = e / sw, o = e - tkl * sw, ss = o / W;
      words[(tkl * 8 + ss) * TW + (o - ss * W)] = c.tiles[((cc * 8 + tkl) * nt + b * 8) * W + o];
    }
    __syncthreads();
#pragma unroll
    for (int tq = 0; tq < 4; ++tq) {
      const int tkl = 4 * hf + tq;
      const unsigned* tw = words + (tkl * 8 + st) * TW;
      const uint16_t* xk = xr + cc * 128 + 16 * tkl;
#pragma unroll
      for (int rt = 0; rt < 16; ++rt) {
        const unsigned state = (__builtin_amdgcn_alignbit(tw[wa[rt]], tw[wb[rt]], sh[rt]) & 0xFFFFu) ^ salt[rt];
        acc = fmaf(value_of<CB>(state), h2f(xk[rt]), acc);
      }
    }
  }
  if (hf) part[i] = acc;
  __syncthreads();
  if (!hf) {
    const float a = __fadd_rn(acc, part[i]);
    part[i] = __fmul_rn(a, h2f(0x26EE));   // the grid step (mad's DQ form); own slot: read above by this thread only
  }
  __syncthreads();
  if (t == 0) {
    float u[64];
#pragma unroll
    for (int q = 0; q < 64; ++q) u[q] = (j & 1) ? __fsub_rn(part[2 * q], part[2 * q + 1]) : __fadd_rn(part[2 * q], part[2 * q + 1]);
#pragma unroll
    for (int w = 32, h = 2; w >= 1; w >>= 1, h <<= 1)
#pragma unroll
      for (int q = 0; q < w; ++q) u[q] = (j & h) ? __fsub_rn(u[2 * q], u[2 * q + 1]) : __fadd_rn(u[2 * q], u[2 * q + 1]);
    c.val[mi * c.val_pitch + r] = __fmul_rn(__fmul_rn(u[0], kRS), h2f(c.svh[s]));
  }
}

__device__ __forceinline__ bool better(float va, int ia, int ca, float vb, int ib, int cb) {
  if (va != va) va = -INFINITY;
  if (vb != vb) vb = -INFINITY;
  return va > vb || (va == vb && (ia < ib || (ia == ib && ca < cb)));
}

__global__ __launch_bounds__(256) void rr_sort(Exl3iRerankCall c) {
  __shared__ float v[EXL3I_RERANK_MAX_R];
  __shared__ int id[EXL3I_RERANK_MAX_R], ok[EXL3I_RERANK_MAX_R];
  __shared__ int nv;
  const int t = (int)threadIdx.x, R = (int)c.R;
  const int64_t mi = blockIdx.x;
  if (t == 0) nv = 0;
  if (t < R) {
    const int64_t gi = c.idx_in[mi * c.in_pitch + t];
    id[t] = (int)gi;
    ok[t] = rr_valid(gi, c);
    v[t] = c.val[mi * c.val_pitch + t];
  }
  __syncthreads();
  int rank = -1;
  if (t < R && ok[t]) {
    rank = 0;
    for (int d = 0; d < R; ++d)
      if (d != t && ok[d] && better(v[d], id[d], d, v[t], id[t], t)) ++rank;
    atomicAdd(&nv, 1);
  }
  __syncthreads();   // everything read before anything is written
  if (rank >= 0) {
    c.idx[mi * c.idx_pitch + rank] = id[t];
    c.val[mi * c.val_pitch + rank] = v[t];
    if (c.pairs) {
      float* p = c.pairs + (mi * c.R + rank) * 2;
      p[0] = __builtin_bit_cast(float, id[t]);
      p[1] = v[t];
    }
  }
  if (t >= nv && t < R) {
    c.idx[mi * c.idx_pitch + t] = -1;
    c.val[mi * c.val_pitch + t] = -INFINITY;
    if (c.pairs) {
      float* p = c.pairs + (mi * c.R + t) * 2;
      p[0] = __builtin_bit_cast(float, -1);
      p[1] = -INFINITY;
    }
  }
}

}  // namespace

extern "C" int exl3i_rerank_rotate_rows(const Exl3iRerankCall* c, uint16_t* out, void* stream) {
  if (!c || c->M < 1 || c->K % 128) return RAD_E_SHAPE;
  hipLaunchKernelGGL(rr_rotate_only, dim3((unsigned)c->M), dim3(256), 0, (hipStream_t)stream, *c, out);
  return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int exl3i_launch_rerank(const Exl3iRerankCall* c, void* stream) {
  if (!c || c->M < 0 || c->R < 1 || c->R > EXL3I_RERANK_MAX_R || c->K % 128 || c->N % 128 || !exl3i_rate_ok(c->m))
    return RAD_E_SHAPE;
  if (c->M == 0) return RAD_OK;
  if (c->M > 65535) return RAD_E_SHAPE;
  const hipStream_t st = (hipStream_t)stream;
  const int TW = c->m / 2 + 1;
  const size_t lds = (size_t)((c->K * 2 + 15) / 16) * 16 + (size_t)64 * TW * 4 + 128 * 4;
  if (lds > 65536) return RAD_E_SHAPE;
  const dim3 grid((unsigned)c->R, (unsigned)c->M);
  switch (c->cb) {
  case EXL3I_CB_MAD:    hipLaunchKernelGGL(rr_score<EXL3I_CB_MAD>, grid, dim3(256), lds, st, *c); break;
  default: return RAD_E_INVAL;
  }
  hipLaunchKernelGGL(rr_sort, dim3((unsigned)c->M), dim3(256), 0, st, *c);
  return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
