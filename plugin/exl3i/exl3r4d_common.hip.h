// exl3r4d_common.hip.h -- the device code every kernel of this library shares: types, the trellis decode in
// registers, the 128-point rotations, WTAB addressing and run finding, the input rotation of one prep task, the LDS
// split-K reduction and the per-16-row-block epilogue. Included by the kernel files (exl3i_nt.hip.h, exl3i_a8_prep.hip,
// exl3i_rate.hip.h, ...). Device code only; nothing here launches. A change here moves every kernel's bits, so the
// device suite (tests/exl3i_gpu_test.hip) re-verifies it.
//
// Provenance: the wmma form of the MoE library's decode, and the libr4d skeleton (Dylan, tcclaviger).
// Everything is `static __device__ __forceinline__`: each kernel file gets its own copy, so one translation unit's
// code object is the same whether or not the other includes this.
#ifndef EXL3R4D_COMMON_HIP_H
#define EXL3R4D_COMMON_HIP_H

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "rad_types.h"
#include "exl3r4d_dev.h"

// The class count this translation unit is built for: 2 (one weight, two class records in the argument block; the dense
// op uses class 0). The four-class form, with its own kernel files, is not part of this tree.
#define EXL3R4D_NCLS 2
#define EXL3R4D_GD4 0
#define EXL3R4D_BDIM blockDim.x

#define WAVE32 32
#define EXL3_BLOCK 16

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef _Float16 v2h __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));
typedef unsigned v2u __attribute__((ext_vector_type(2)));

// Cache-policy hints (EXL3R4D_STH_XR / _H / _Y) of the routed ops' row stores, per buffer: XR the prep's rotated rows
// (dead after the GEMM), H gate_up's h (read by down), Y down's y (read by the gather). Each is the th field's code
// (llvm-mc, gfx1201: 0 RT (plain), 1 NT, 2 HT, 3 WB, 4 NT_RT, 5 RT_NT, 6 NT_HT, 7 NT_WB). Same bytes stored; only the
// cache policy changes. EXL3R4D_STH sets all three.
#ifndef EXL3R4D_STH
#define EXL3R4D_STH 0
#endif
#ifndef EXL3R4D_STH_XR
#define EXL3R4D_STH_XR EXL3R4D_STH
#endif
#ifndef EXL3R4D_STH_H
#define EXL3R4D_STH_H EXL3R4D_STH
#endif
#ifndef EXL3R4D_STH_Y
#define EXL3R4D_STH_Y EXL3R4D_STH
#endif
enum { kStXr = 0, kStH = 1, kStY = 2 };   // st_row's buffer
#define EXL3R4D_ST_ASM(th)                                                                                    \
  do {                                                                                                     \
    if constexpr (sizeof(V) == 16)                                                                         \
      asm volatile("global_store_b128 %0, %1, off th:" th ::"v"(dst), "v"(v) : "memory");                  \
    else                                                                                                   \
      asm volatile("global_store_b64 %0, %1, off th:" th ::"v"(dst), "v"(v) : "memory");                   \
  } while (0)
// A row store with gfx12 store th code TH (the codes above).
template <int TH, class V>
static __device__ __forceinline__ void st_th(void* dst, V v) {
  static_assert(sizeof(V) == 8 || sizeof(V) == 16, "b64 / b128 row stores");
  static_assert(TH >= 0 && TH <= 7, "a gfx12 store th code");
  // No builtin carries th codes other than NT on a global store; the asm stores the same register pair / quad.
  if constexpr (TH == 0) *(V*)dst = v;
  else if constexpr (TH == 1) __builtin_nontemporal_store(v, (V*)dst);
  else if constexpr (TH == 2) EXL3R4D_ST_ASM("TH_STORE_HT");
  else if constexpr (TH == 3) EXL3R4D_ST_ASM("TH_STORE_WB");
  else if constexpr (TH == 4) EXL3R4D_ST_ASM("TH_STORE_NT_RT");
  else if constexpr (TH == 5) EXL3R4D_ST_ASM("TH_STORE_RT_NT");
  else if constexpr (TH == 6) EXL3R4D_ST_ASM("TH_STORE_NT_HT");
  else EXL3R4D_ST_ASM("TH_STORE_NT_WB");
}
template <int BUF, class V>
static __device__ __forceinline__ void st_row(void* dst, V v) {
  st_th<BUF == kStXr ? EXL3R4D_STH_XR : BUF == kStH ? EXL3R4D_STH_H : EXL3R4D_STH_Y>(dst, v);
}

// Global-qualified so loads stay global_load (a flat load also bumps dscnt on gfx12).
typedef __attribute__((address_space(1))) const unsigned* gu32;
typedef __attribute__((address_space(1))) const unsigned long long* gu64;
typedef __attribute__((address_space(1))) const unsigned char* gu8;

static __device__ __forceinline__ void lds_barrier() {
  // Order LDS only, so the barrier does not invalidate the vector cache.
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// lane ^ 16 for the epilogue's 16-point stage (__shfl_xor: ds_bpermute_b32, an LDS-crossbar round trip per value).
static __device__ __forceinline__ float swap16(float x) {
  return __shfl_xor(x, 16, 32);
}

static __device__ __forceinline__ unsigned short f32_to_bf16(float f) {  // round to nearest even
  const unsigned u = __builtin_bit_cast(unsigned, f);
  return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

template <int CTRL>
static __device__ __forceinline__ float dpp_f(float v) {  // row_xmask: lane ^ mask within each row of 16
  return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), CTRL, 0xF, 0xF, false));
}

static __device__ __forceinline__ float x16_f(float v) {  // lane ^ 16
  return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, v), __builtin_bit_cast(int, v),
                                                                0x76543210, (int)0xFEDCBA98, false, false));
}

template <int CTRL>
static __device__ __forceinline__ void fwht_lane_stage(float (&v)[4], float sgn) {
#pragma unroll
  for (int q = 0; q < 4; q++) v[q] = fmaf(v[q], sgn, dpp_f<CTRL>(v[q]));
}

// One 128-feature block of one row's input rotation, by one wave: had128(fp16(x * suh)) * 1/sqrt(128) as fp16,
// four consecutive features per lane (rounding points: x * suh rounded to fp16, the transform in fp32, the result
// rounded to fp16 -- exl3_abi.h's f16(had(f16(x * suh)))). The stage order is exl3_had128's (h = 1, 2, .. 64) and
// each stage is exact-sign a +- b. NOT bit-identical to the reference's prologue: the final x 1/sqrt(128) and fp32
// rounding inside the transform differ in the last place on ~4e-5 of elements (1 f16 ulp); every device path (prep
// kernel, in-GEMM rotation, gate_up's h_rot epilogue) uses this same function, so the device's h_rot pair is
// bit-identical to its h_rot = 0 pair. Feature bits 0, 1 in the lane, 2..5 through DPP row_xmask, 6 through
// v_permlanex16_b32.
// rotate128 in two halves for a pipelined caller (the staged K loop): the lane's loads, then the math on
// what they returned. rotate128 itself is the two back to back.
static __device__ __forceinline__ v2u rotate128_math(v2u xr, v2u sr, bool x_f16, int lane);
static __device__ __forceinline__ v2u rotate128(const unsigned short* x_blk, bool x_f16, const _Float16* suh_blk,
                                                int lane) {
  const v2u xr = *(const v2u*)(x_blk + 4 * lane);
  const v2u sr = *(const v2u*)(suh_blk + 4 * lane);
  return rotate128_math(xr, sr, x_f16, lane);
}
static __device__ __forceinline__ v2u rotate128_math(v2u xr, v2u sr, bool x_f16, int lane) {
  float v[4];
  const unsigned xs[2] = {xr[0], xr[1]}, ss[2] = {sr[0], sr[1]};
#pragma unroll
  for (int q = 0; q < 4; q++) {
    const unsigned short xb = (unsigned short)(xs[q >> 1] >> (16 * (q & 1)));
    const unsigned short sb = (unsigned short)(ss[q >> 1] >> (16 * (q & 1)));
    const float xv = x_f16 ? (float)__builtin_bit_cast(_Float16, xb) : __builtin_bit_cast(float, (unsigned)xb << 16);
    v[q] = (float)(_Float16)(xv * (float)__builtin_bit_cast(_Float16, sb));
  }
  { const float a = v[0], b = v[1], c = v[2], d = v[3];
    v[0] = a + b; v[1] = a - b; v[2] = c + d; v[3] = c - d;
    const float e0 = v[0], e1 = v[1];
    v[0] = e0 + v[2]; v[1] = e1 + v[3]; v[2] = e0 - v[2]; v[3] = e1 - v[3]; }
  fwht_lane_stage<0x161>(v, (lane & 1) ? -1.f : 1.f);
  fwht_lane_stage<0x162>(v, (lane & 2) ? -1.f : 1.f);
  fwht_lane_stage<0x164>(v, (lane & 4) ? -1.f : 1.f);
  fwht_lane_stage<0x168>(v, (lane & 8) ? -1.f : 1.f);
  {
    const float sgn = (lane & 16) ? -1.f : 1.f;
#pragma unroll
    for (int q = 0; q < 4; q++) v[q] = fmaf(v[q], sgn, x16_f(v[q]));
  }
  constexpr float RS = 0.08838834764831845f;
  union { _Float16 h[4]; v2u u; } o;
#pragma unroll
  for (int q = 0; q < 4; q++) o.h[q] = (_Float16)(v[q] * RS);
  return o.u;
}

// ----------------------------------------------------------------------------- trellis decode
// The lane plan (Plan<BITS>) and the byte sums; the codebook's hash lives in exl3i_rate.hip.h (mad).

static __device__ __forceinline__ unsigned sad_lo(unsigned x, unsigned acc) {  // acc + byte sum of x
  return __builtin_amdgcn_sad_u8(x, 0u, acc);
}

static __device__ __forceinline__ unsigned sad_hi(unsigned x, unsigned acc) {  // acc + (byte sum of x) << 16
  return __builtin_amdgcn_sad_hi_u8(x, 0u, acc);
}

template <int BITS>
struct Plan {
  static constexpr int WORDS = 8 * BITS;       // u32 per tile
  static constexpr int NW = BITS == 7 ? 3 : 2;  // words a run of four positions touches
  int woff[2][NW];                             // word index in the tile, per run
  unsigned shift[2];                           // funnel shift that ends the run's bits at bit 0

  __device__ __forceinline__ explicit Plan(int lane) {
    const int n = lane & 15, h = lane >> 4;
    const int base = 32 * (n & 7) + 4 * (n >> 3) + 16 * h;
#pragma unroll
    for (int r = 0; r < 2; r++) {
      const int e3 = (base + 8 * r + 4) * BITS;  // end bit of the run's last state
      const int de = (e3 - 1) >> 5;              // word holding that bit
      shift[r] = (unsigned)(32 * (de + 1) - e3);
#pragma unroll
      for (int x = 0; x < NW; x++) woff[r][x] = (de - (NW - 1) + x + WORDS) % WORDS;
    }
  }
};

static __device__ __forceinline__ v8h frag_lds(const _Float16* p) {  // fragA from LDS (halfs 0..3 and 8..11)
  union { unsigned long long w[2]; v8h f; } x;
  x.w[0] = *(const unsigned long long*)p;
  x.w[1] = *(const unsigned long long*)(p + 8);
  return x.f;
}

static __device__ __forceinline__ v8f mma(v8h wf, v8h af, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(wf, af, c);
}

// ----------------------------------------------------------------------------- epilogue helpers

static __device__ __forceinline__ void had8(float (&v)[8]) {  // unnormalized natural-order 8-point Hadamard
#pragma unroll
  for (int s = 1; s < 8; s <<= 1)
#pragma unroll
    for (int e = 0; e < 8; e++)
      if (!(e & s)) {
        const float a = v[e], b = v[e + s];
        v[e] = a + b;
        v[e + s] = a - b;
      }
}

// In-lane over e, then over h: after this, v is the tile's 16-point transform (bits 0..3 of n). Over h the lane
// pair holds a (h = 0) and b (h = 1) and wants a + b and a - b: lane h = 0 computes a * 1 + b, lane h = 1
// computes b * -1 + a.
static __device__ __forceinline__ void had16_lane(float (&v)[8], float sgn_h) {
  had8(v);
#pragma unroll
  for (int e = 0; e < 8; e++) v[e] = fmaf(v[e], sgn_h, swap16(v[e]));
}

static __device__ __forceinline__ void lds_put(float* dst, const float (&v)[8]) {
  *(v4f*)dst = (v4f){v[0], v[1], v[2], v[3]};
  *(v4f*)(dst + 4) = (v4f){v[4], v[5], v[6], v[7]};
}

// The 8-point transform over t: out = sum over t2 of (-1)^popcount(t & t2) * tile t2 (this lane's 8 values).
static __device__ __forceinline__ void had_tiles(const float* xs, int t, int lane, float (&out)[8]) {
#pragma unroll
  for (int e = 0; e < 8; e++) out[e] = 0.f;
#pragma unroll
  for (int t2 = 0; t2 < 8; t2++) {
    const float sg = (__builtin_popcount(t & t2) & 1) ? -1.f : 1.f;
    const v4f a = *(const v4f*)(xs + t2 * 256 + lane * 8);
    const v4f b = *(const v4f*)(xs + t2 * 256 + lane * 8 + 4);
#pragma unroll
    for (int e = 0; e < 4; e++) {
      out[e] = fmaf(sg, a[e], out[e]);
      out[4 + e] = fmaf(sg, b[e], out[4 + e]);
    }
  }
}

static __device__ __forceinline__ void load_h8(const _Float16* p, float (&v)[8]) {
  const v8h x = *(const v8h*)p;
#pragma unroll
  for (int e = 0; e < 8; e++) v[e] = (float)x[e];
}

// ----------------------------------------------------------------------------- a weight's planes

static __device__ __forceinline__ const unsigned char* entry_at(const void* tab, long long s0, int j) {
  if (s0 == 0) return (const unsigned char*)((const void* const*)tab)[j];
  return (const unsigned char*)tab + (size_t)j * (size_t)s0;
}

struct Planes {
  const unsigned* codes;
  const _Float16 *suh, *suh1, *svh;
  const _Float16* dn_suh;  // gate_up with h_rot (v3 EPI 1): down's suh of this expert; null otherwise
  int rows, cols;
};

#define EXL3R4D_KARG_PREFETCH() ((void)0)

static __device__ __forceinline__ Planes planes_of(const Exl3r4dGemm& p, int e) {
  // A class picked field by field: indexing the argument block by a runtime index would copy it to the stack.
  Planes q;
  const int c = e % p.n_class, j = e / p.n_class;
  const Exl3r4dCls& C0 = p.cls[0];
  const Exl3r4dCls& C1 = p.cls[1];
  q.codes = (const unsigned*)entry_at(c ? C1.codes : C0.codes, c ? C1.codes_s0 : C0.codes_s0, j);
  q.suh = (const _Float16*)entry_at(c ? C1.suh : C0.suh, c ? C1.suh_s0 : C0.suh_s0, j);
  const void* s1 = c ? C1.suh1 : C0.suh1;
  q.suh1 = s1 ? (const _Float16*)entry_at(s1, c ? C1.suh1_s0 : C0.suh1_s0, j) : q.suh;
  q.svh = (const _Float16*)entry_at(c ? C1.svh : C0.svh, c ? C1.svh_s0 : C0.svh_s0, j);
  const void* dn = c ? C1.dn_suh : C0.dn_suh;
  q.dn_suh = p.h_rot && dn ? (const _Float16*)entry_at(dn, c ? C1.dn_suh_s0 : C0.dn_suh_s0, j) : nullptr;
  q.rows = c ? C1.rows : C0.rows;
  q.cols = c ? C1.cols : C0.cols;
  return q;
}

// LDS split-K: the partials of waves ksplit > 0 through LDS into the ksplit-0 waves, summed in wave order
// (deterministic). Workgroup-wide (two barriers); every wave calls it. Needs (SK - 1) * NPW * 256 floats a column
// group of LDS at `lds`.
template <int NPW>
static __device__ __forceinline__ void splitk_reduce(v8f (&acc)[NPW], float* lds, int col_local, int ksplit, int SK,
                                                     int lane) {
  if (ksplit) {
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      float* dst = lds + ((col_local * NPW + j) * (SK - 1) + (ksplit - 1)) * 256 + lane * 8;
#pragma unroll
      for (int e = 0; e < 8; e++) dst[e] = acc[j][e];
    }
  }
  lds_barrier();
  if (!ksplit) {
#pragma unroll
    for (int j = 0; j < NPW; j++)
      for (int s = 0; s < SK - 1; s++) {
        const float* src = lds + ((col_local * NPW + j) * (SK - 1) + s) * 256 + lane * 8;
#pragma unroll
        for (int e = 0; e < 8; e++) acc[j][e] += src[e];
      }
  }
  lds_barrier();
}

// The epilogue of one 16-row block (after the split-K reduction), the workgroup owning 128 output columns cb of every
// matrix: WV = 8 MATS / NPW waves of NPW tiles, wave = col_local * SK + ksplit; this lane's row `row` (in the run or
// not; live = sorted_tok >= 0). The 16-point transform in-lane and over the lane pair, the 8-point over tiles through
// LDS (8 MATS tiles of 256 floats at `lds`), x 1/sqrt(128), x svh, then
//   MATS 2 (EPI 1): h = f16(silu(gate) * up), gate rows [0, width), up rows [width, 2 width), f16 [T][y_pitch];
//   MATS 1 (EPI 2): y = bf16(...), bf16 [T][y_pitch];
// a dropped row in the run is written as zeros, rows past the run are not written. Workgroup-wide (two barriers).
// (epilogue_block follows svh_early below.)
//
// v3 EPI 1: gate_up's h block staged at this LDS byte offset (16 rows x 128 f16 = 4 KB, past the epilogue's 16 KB
// of tiles; the launcher allocates kHrotStage + 4096 at least).
constexpr int kHrotStage = 16384;

// Output tiles t (0..7 of a 128-column block) a wave writes in the epilogue: 8 / WV, or one for the first 8 waves when
// WV = 8 MATS / NPW is 16.
template <int NPW, int MATS>
static __host__ __device__ constexpr int epi_opw() { return 8 * MATS / NPW >= 8 ? 1 : 8 / (8 * MATS / NPW); }

// EXL3R4D_SVH_EARLY: the epilogue's svh loads issued in the prologue, before the K loop, and carried
// as raw f16 (svp[o][m]: output tile o of the wave, matrix m), instead of a global round trip after the reduction
// barrier on every workgroup's tail. A wave past the block's 8 tiles (WV 16) loads tile 7's: never used.
template <int NPW, int MATS>
static __device__ __forceinline__ void svh_early(const Planes& w, int col_local, int lane, int cb,
                                                 v8h (&svp)[epi_opw<NPW, MATS>()][MATS]) {
  typedef const __attribute__((address_space(1))) v8h* gv8h;
  constexpr int OPW = epi_opw<NPW, MATS>();
  const int width = MATS == 2 ? w.rows / 2 : w.rows, h = lane >> 4;
#pragma unroll
  for (int o = 0; o < OPW; o++) {
    const int t = min(col_local * OPW + o, 7), n0 = cb * 128 + 16 * t + 8 * h;
#pragma unroll
    for (int m = 0; m < MATS; m++) svp[o][m] = *(gv8h)(w.svh + m * width + n0);
  }
}

// EPIB (opt-in, `EPIB`; down's decode GEMM at NPW 2 / split-K 1, the caller's choice): the 8-point transform over
// the block's tiles as a BUTTERFLY instead of had_tiles' 8-term sums -- a wave holds tiles 2c and 2c + 1, so the
// first stage (tile bit 0) is in registers and the other two are one LDS exchange each with the partner wave c ^ 1,
// then c ^ 2 (write its own 16 floats a lane, barrier, read the partner's; the half with the bit set takes partner
// - mine). 48 adds and 2 x 32 floats of LDS a lane instead of 128 FMAs and 16 + 128; each wave stores its own tiles.
// The sums are in the butterfly order (exllamav3's, the transcription's), not had_tiles' sequential one: bits move
// inside the tolerance (a numerics decision). LDS: two [8][32][8] f32 exchange buffers, 16 KB.
// Callers that do not ask keep had_tiles.
// EPIC (opt-in, `EPIC`; down, two output tiles a wave): the wave's tiles 2c and 2c + 1 differ only in tile bit 0, so
// both come from ONE pass over the 8 staged tiles -- E = the even tiles' signed sum, O = the odd tiles', out(2c) =
// E + O, out(2c + 1) = E - O: 16 LDS loads and 64 FMAs + 16 adds a lane instead of had_tiles twice (32 and 128).
// The sum is paired (E, O) instead of sequential over all 8: bits move inside the tolerance (a numerics decision).
// yr16 (exl3i, EPI 2 without EPIB): the weight's rows are in R16 order -- output s = n0 + e is stored at column
// ((s >> 4) & 7) width / 8 + (s >> 7) 16 + (s & 15) = t width / 8 + cb 16 + 8 h + e (exl3i_encoding.h); svh stays
// indexed by s. 0: n0.
template <int NPW, int MATS, bool PRE = false, bool EPIB = false, bool EPIC = false>
static __device__ __forceinline__ void epilogue_block(const Exl3r4dGemm& p, const Planes& w, const v8f (&acc)[NPW],
                                                      int col_local, int ksplit, int lane, int cb, int row,
                                                      bool in_run, bool live, float* lds,
                                                      const v8h (*svp)[MATS] = nullptr, int yr16 = 0) {
  constexpr int WV = 8 * MATS / NPW;
  constexpr float RS = 0.08838834764831845f;  // 1 / sqrt(128)
  constexpr int OPW = epi_opw<NPW, MATS>();
  static_assert(WV >= 1 && WV <= 16 && (8 % WV == 0 || WV == 16), "EPI 1 / 2: 8 * MATS / NPW waves");
  const int h = lane >> 4;
  const int width = MATS == 2 ? w.rows / 2 : w.rows;  // this class's output columns
  const bool writer = col_local * OPW < 8;
  float* xs = lds;  // [8 MATS tiles][32 lanes][8]
  const float sgn_h = h ? -1.f : 1.f;
  if constexpr (EPIB && MATS == 1 && NPW == 2 && !PRE) {
    if (p.sk == 1) {  // workgroup-uniform: every wave holds two whole tiles
      float v[2][8];
#pragma unroll
      for (int j = 0; j < 2; j++) {
#pragma unroll
        for (int e = 0; e < 8; e++) v[j][e] = acc[j][e];
        had16_lane(v[j], sgn_h);
      }
#pragma unroll
      for (int e = 0; e < 8; e++) {  // tile bit 0: this wave's own pair
        const float a = v[0][e], b = v[1][e];
        v[0][e] = a + b;
        v[1][e] = a - b;
      }
#pragma unroll
      for (int s = 0; s < 2; s++) {  // tile bits 1 and 2: the partner waves c ^ 1, c ^ 2
        float* xb = lds + s * 2048;
        const int pc = col_local ^ (1 << s);
        const bool hi = (col_local >> s) & 1;
#pragma unroll
        for (int j = 0; j < 2; j++) lds_put(xb + (col_local * 2 + j) * 256 + lane * 8, v[j]);
        lds_barrier();
#pragma unroll
        for (int j = 0; j < 2; j++) {
          const v4f a = *(const v4f*)(xb + (pc * 2 + j) * 256 + lane * 8);
          const v4f b = *(const v4f*)(xb + (pc * 2 + j) * 256 + lane * 8 + 4);
#pragma unroll
          for (int e = 0; e < 4; e++) {
            v[j][e] = hi ? a[e] - v[j][e] : v[j][e] + a[e];
            v[j][4 + e] = hi ? b[e] - v[j][4 + e] : v[j][4 + e] + b[e];
          }
        }
      }
      if (in_run) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
          const int n0 = cb * 128 + 16 * (col_local * 2 + j) + 8 * h;
          float sv[8];
          load_h8(w.svh + n0, sv);
          v4u out;
#pragma unroll
          for (int q = 0; q < 4; q++)
            out[q] = live ? (unsigned)f32_to_bf16((v[j][2 * q] * RS) * sv[2 * q]) |
                                ((unsigned)f32_to_bf16((v[j][2 * q + 1] * RS) * sv[2 * q + 1]) << 16)
                          : 0u;
          st_row<kStY>((unsigned short*)p.y + (size_t)row * p.y_pitch + n0, out);
        }
      }
      lds_barrier();  // the exchange buffers are rewritten next pass
      return;
    }
  }
  if (!ksplit) {
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      float v[8];
#pragma unroll
      for (int e = 0; e < 8; e++) v[e] = acc[j][e];
      had16_lane(v, sgn_h);
      lds_put(xs + (col_local * NPW + j) * 256 + lane * 8, v);
    }
  }
  lds_barrier();
  [[maybe_unused]] float epc_e[8], epc_o[8];  // EPIC: the even / odd tiles' sums
  if (!ksplit && writer && in_run) {
#pragma unroll
    for (int o = 0; o < OPW; o++) {
      const int t = col_local * OPW + o;
      const int n0 = cb * 128 + 16 * t + 8 * h;
      if constexpr (MATS == 2) {  // EPI 1: h = f16(silu(gate) * up), gate rows [0, width), up rows [width, 2 width)
        float g[8], u[8], sg[8], su[8];
        had_tiles(xs, t, lane, g);
        had_tiles(xs + 8 * 256, t, lane, u);
        if constexpr (PRE) {
#pragma unroll
          for (int e = 0; e < 8; e++) { sg[e] = (float)svp[o][0][e]; su[e] = (float)svp[o][1][e]; }
        } else {
          load_h8(w.svh + n0, sg);
          load_h8(w.svh + width + n0, su);
        }
        union { _Float16 hv[8]; v4u u; } out;
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const float gv = (g[e] * RS) * sg[e], uv = (u[e] * RS) * su[e];
          out.hv[e] = live ? (_Float16)((gv * (1.0f / (1.0f + expf(-gv)))) * uv) : (_Float16)0.f;
        }
        if (p.h_rot)  // v3 EPI 1: staged, rotated below with down's suh before it is stored
          *(v4u*)((_Float16*)((unsigned char*)lds + kHrotStage) + (lane & 15) * 128 + 16 * t + 8 * h) = out.u;
        else
          st_row<kStH>((_Float16*)p.y + (size_t)row * p.y_pitch + n0, out.u);
      } else {  // EPI 2: y = bf16((had * 1/sqrt(128)) * svh)
        float r[8], sv[8];
        if constexpr (EPIC && OPW % 2 == 0) {  // each pair of the wave's tiles from one pass (even o computes, odd o reuses)
          if (o % 2 == 0) {
#pragma unroll
            for (int e = 0; e < 8; e++) { epc_e[e] = 0.f; epc_o[e] = 0.f; }
#pragma unroll
            for (int t2 = 0; t2 < 8; t2++) {
              const float sg = (__builtin_popcount(t & t2) & 1) ? -1.f : 1.f;
              const v4f a = *(const v4f*)(xs + t2 * 256 + lane * 8);
              const v4f b = *(const v4f*)(xs + t2 * 256 + lane * 8 + 4);
              float* acc8 = (t2 & 1) ? epc_o : epc_e;
#pragma unroll
              for (int e = 0; e < 4; e++) {
                acc8[e] = fmaf(sg, a[e], acc8[e]);
                acc8[4 + e] = fmaf(sg, b[e], acc8[4 + e]);
              }
            }
          }
#pragma unroll
          for (int e = 0; e < 8; e++) r[e] = o % 2 == 0 ? epc_e[e] + epc_o[e] : epc_e[e] - epc_o[e];
        } else {
          had_tiles(xs, t, lane, r);
        }
        if constexpr (PRE) {
#pragma unroll
          for (int e = 0; e < 8; e++) sv[e] = (float)svp[o][0][e];
        } else {
          load_h8(w.svh + n0, sv);
        }
        const int ny = yr16 ? t * (width / 8) + cb * 16 + 8 * h : n0;   // the store's column (R16: the vocabulary row)
        if (p.y_dt == 2) {  // the dense GEMM's f16 y
          union { _Float16 hv[8]; v4u u; } oh;
#pragma unroll
          for (int e = 0; e < 8; e++) oh.hv[e] = live ? (_Float16)((r[e] * RS) * sv[e]) : (_Float16)0.f;
          *(v4u*)((_Float16*)p.y + (size_t)row * p.y_pitch + ny) = oh.u;
        } else if (p.y_dt == 1) {  // the dense GEMM's f32 y (the logits): the same values, unrounded
          v4f o0, o1;
#pragma unroll
          for (int e = 0; e < 4; e++) {
            o0[e] = live ? (r[e] * RS) * sv[e] : 0.f;
            o1[e] = live ? (r[4 + e] * RS) * sv[4 + e] : 0.f;
          }
          float* yf = (float*)p.y + (size_t)row * p.y_pitch + ny;
          *(v4f*)yf = o0;
          *(v4f*)(yf + 4) = o1;
        } else {
          v4u out;
#pragma unroll
          for (int q = 0; q < 4; q++)
            out[q] = live ? (unsigned)f32_to_bf16((r[2 * q] * RS) * sv[2 * q]) |
                                ((unsigned)f32_to_bf16((r[2 * q + 1] * RS) * sv[2 * q + 1]) << 16)
                          : 0u;
          st_row<kStY>((unsigned short*)p.y + (size_t)row * p.y_pitch + ny, out);
        }
      }
    }
  }
  if (MATS == 2 && p.h_rot) {  // (MATS is a template constant: down compiles none of this)
    // v3 EPI 1 (h_rot): the block's h rows, staged above, rotated with DOWN's suh -- exactly down's step 1
    // (prep_rotate for down: x = h f16, suh = down's, rotate128's rounding), so the pair is bit-identical to the
    // h_rot = 0 pair. One wave a row of the pass; a row not in the run is not written, a dropped row is zeros.
    lds_barrier();
    typedef const __attribute__((address_space(1))) v2u* gv2;
    const _Float16* hst = (const _Float16*)((const unsigned char*)lds + kHrotStage);
    const int nwv = (int)EXL3R4D_BDIM / WAVE32, wv = __builtin_amdgcn_readfirstlane((int)(threadIdx.x / WAVE32));
    const int rc = row - (lane & 15);
    const v2u sr = *(gv2)((const unsigned char*)w.dn_suh + (size_t)(cb * 128 + 4 * lane) * 2);
    for (int i = wv; i < 16; i += nwv) {
      if (!__builtin_amdgcn_readlane((int)in_run, i)) continue;  // wave-uniform
      const bool li = __builtin_amdgcn_readlane((int)live, i);
      const v2u xr = *(const v2u*)(hst + i * 128 + 4 * lane);
      const v2u o = li ? rotate128_math(xr, sr, true, lane) : (v2u)(0u);
      st_row<kStH>((_Float16*)p.y + (size_t)(rc + i) * p.y_pitch + cb * 128 + 4 * lane, o);
    }
  }
  lds_barrier();  // xs (and, next pass, the rotated rows) are rewritten
}

#endif  // EXL3R4D_COMMON_HIP_H
