/* Derived from libr4d/r4d_hc_bf16.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_hc_bf16.hip -- the gated residual (hyper-connections): hc_enter, hc_read,
// hc_write and hc_project.
//
// Qwen4-Exp carries a residual stream `hc` times the model width, with no pre-norm and no residual
// add: every block reads a data-dependent mix of the hc sub-streams and writes its output back into
// all of them with hc data-dependent gains. Schema: docs/OPS.md; composition:
// arch/common/rad_block_hc.h.
//
// ============================== hc_read is two kernels ==============================
//
// The read is
//
//     N   = grouped_rmsnorm(h)                     [HN]   HN = hc * n
//     d   = silu(W_down @ N / hc)                  [LR]   LR = lowrank
//     x   = mean_c( sigmoid(W_up @ d)[c] * N[c] )  [n]
//     inj = 2 * sigmoid(W_inj @ N / hc)            [hc]
//
// `x` depends on all of `d`, so no workgroup can produce a channel of `x` before every column of
// `d` exists. The kernel boundary serves as the grid-wide barrier, without cooperative launch.
//
// Weight traffic, not arithmetic, decides the decomposition. At Qwen3.8-Flash-Next's shape
// (n 2560, hc 4, LR 320) each mixing matrix is 6.55 MB and there are 96 connections: 1.27 GB per
// step, a 2.1 ms floor at ~600 GB/s, against a negligible 6.6M MACs per token. So each weight byte
// must be read once per step:
//
//   kernel A   grid (ceil(LR/64), T). A workgroup owns 64 columns of W_down and reads all of h
//              for the norm, so W_down is read once and h ceil(LR/64) times per row (100 KB
//              against 1.3 MB of W_down per workgroup).
//   kernel B   grid (n/128, T). A workgroup owns 128 channels (hc*128 rows of W_up), so W_up is
//              read once, and `u` is never materialised because the workgroup holds all hc rows of
//              a channel and does the mean itself.
//
// 128 channels are exactly one fp8 scale group, so kernel B also quantises `x` to fp8 with no
// extra pass or launch (a separate quant_act_fp8 at 96 connections would cost ~0.43 ms of launch
// overhead).
//
// Tiled path: above ~8 rows the per-row kernels would re-read both matrices per row, so the
// products run tiled. A tile over the flattened channel index (64 consecutive `j`) covers one
// sub-stream, putting a channel's hc terms in different workgroups; `u` must then go through
// `ubuf` and a third kernel (`hc_mix`), costing a launch and a 1.28 MB round trip per connection.
//
// `hc_up_mix_tiled_kernel` instead tiles over (sub-stream, channel) pairs: the same W_up bytes in
// hc runs of NT/hc rows, with all hc terms of a channel in one workgroup. A workgroup then owns
// NT/hc = 16 channels of a 128-wide scale group, so an arrival counter lets the last of the eight
// workgroups covering a group quantise it (1-2% of the step at multi-sequence decode, same
// output). At the 128-row prefill tile the same pairing is `hc_up_mix_wide_kernel`, which keeps a
// channel's sub-streams in one lane's accumulators instead of LDS.
//
// ============================== matching the reference ==============================
//
// libref/ref_hc.cpp is the oracle. torch's grouped norm returns `.type_as(x)`, so the reference
// model reads a bf16 N in both matrix products and the final multiply. This kernel narrows N, `d`
// and sigmoid(u) in the same places. Sums of squares are f32 here and f64 there, within bf16
// tolerance.
#include <hip/hip_runtime.h>

#include <atomic>
#include <type_traits>
#include <cstdlib>
#include <mutex>
#include "halo.h"
#include "halo_emit.h"
#include "halo_gdn_args.h"
#include "halo_common.h"
#include "halo_gdn_wmma.h"
#include "halo_fwht.h"
#include "halo_hc_fp8.h"
#include "halo_q8.h"

namespace {

constexpr int kWave    = 32;
constexpr int kThreads = 256;
constexpr int kWaves   = kThreads / kWave;
constexpr int kHcMax   = 8;      // streams; the reference refuses hc <= 1 and nothing needs > 8
constexpr int kDownTAMax = 64;   // columns of W_down a workgroup owns, at most
constexpr int kUpRows  = 64;     // rows of W_up a workgroup owns
constexpr int kWantWG  = 64;     // workgroups worth launching, when the CU count cannot be read

/* Returns WGPs, not CUs: `hipDeviceProp_t::multiProcessorCount` reports WGPs on RDNA, two CUs
 * each (the R9700 has 64 CUs in 32 WGPs). Left as is on purpose: it is a floor on workgroups worth
 * launching, and here grids of 40, 80 and 160 perform the same. Halving `ta` only doubles redundant
 * work, since each workgroup re-reads all of `h` and the norm weight (60 KiB here) before its own
 * W_down columns (4.8 MB of re-read at ta=4, 2.4 MB at ta=8). A kernel short of blocks should
 * double the number at its own call site. */
static inline int want_wg() {
    static const int v = [] {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) return kWantWG;
        hipDeviceProp_t pr{};
        if (hipGetDeviceProperties(&pr, dev) != hipSuccess) return kWantWG;
        return pr.multiProcessorCount > 0 ? pr.multiProcessorCount : kWantWG;
    }();
    return v;
}

constexpr int kUpTB    = 128;    // CHANNELS a workgroup owns -- and one fp8 scale group
constexpr int kFp8Grp  = 128;
constexpr int kDotU    = 4;      // 512-byte loads a wave keeps in flight over a long weight row

__device__ __forceinline__ float sigmoidf_(float v) { return 1.0f / (1.0f + __expf(-v)); }
__device__ __forceinline__ float siluf_(float v)    { return v * sigmoidf_(v); }
__device__ __forceinline__ float nbf(float v)       { return bf2f(f2bf(v)); }

/* bf16 in whole registers. gfx12's compiler holds a 16-bit value in a register half: shorts written
 * through a union cost a half move each before the pack, and a lone d16 load leaves the other half
 * live, waiting on that register's last write. These helpers keep values 32 bits wide (pairs packed
 * with f2bf2, same rounding as f2bf; halves read by shift and mask; lone bf16 loaded zero-extended)
 * and give the same bits. */
__device__ __forceinline__ uint4 f2bf8(const float (&f)[8]) {
    return make_uint4(f2bf2(f[0], f[1]), f2bf2(f[2], f[3]), f2bf2(f[4], f[5]), f2bf2(f[6], f[7]));
}
__device__ __forceinline__ void bf8_f(uint4 v, float (&f)[8]) {
    f[0] = bf16w_lo(v.x); f[1] = bf16w_hi(v.x); f[2] = bf16w_lo(v.y); f[3] = bf16w_hi(v.y);
    f[4] = bf16w_lo(v.z); f[5] = bf16w_hi(v.z); f[6] = bf16w_lo(v.w); f[7] = bf16w_hi(v.w);
}
__device__ __forceinline__ void bf8_f(halo_u32x4 v, float (&f)[8]) {
    f[0] = bf16w_lo(v.x); f[1] = bf16w_hi(v.x); f[2] = bf16w_lo(v.y); f[3] = bf16w_hi(v.y);
    f[4] = bf16w_lo(v.z); f[5] = bf16w_hi(v.z); f[6] = bf16w_lo(v.w); f[7] = bf16w_hi(v.w);
}
/* Read through a global pointer; through a generic one the load is still flat_load_d16_b16. */
typedef __attribute__((address_space(1))) const unsigned short* hc_gu16;
__device__ __forceinline__ uint32_t bfw(const unsigned short* p) { return *(hc_gu16)p; }
__device__ __forceinline__ float    bfw_f(uint32_t w)            { return __uint_as_float(w << 16); }

/* ---- E4M3 mixing matrices (halo_hc_fp8.h) ------------------------------------------------
 *
 * Every kernel reading mix_down or mix_up has an F8 form, selected by the stored dtype: a row is
 * its codes then its f32 group scales, `ld` bytes apart. Per-row kernels multiply codes in f32 and
 * scale each group's partial sum; tiled kernels decode into their bf16 staging tile with the scale
 * applied and keep the bf16 WMMA. The bf16 form compiles from the same text with F8 false. */

/* Eight E4M3 codes, two dwords of them, to f32 in order: v_cvt_pk_f32_fp8, two a time. */
__device__ __forceinline__ void e4m3x8(uint32_t lo, uint32_t hi, float* f) {
    const v2f a = __builtin_amdgcn_cvt_pk_f32_fp8((int)lo, false);
    const v2f b = __builtin_amdgcn_cvt_pk_f32_fp8((int)lo, true);
    const v2f c = __builtin_amdgcn_cvt_pk_f32_fp8((int)hi, false);
    const v2f d = __builtin_amdgcn_cvt_pk_f32_fp8((int)hi, true);
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
    f[4] = c.x; f[5] = c.y; f[6] = d.x; f[7] = d.y;
}

/* Eight bf16 of a uint4 to f32, in order. */
__device__ __forceinline__ void bf16x8(const uint4& v, float* f) {
    const uint32_t w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i]     = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

/* Eight codes of an E4M3 row, decoded and scaled into the bf16 a tiled kernel stages. */
__device__ __forceinline__ uint4 e4m3x8_bf16(const unsigned char* __restrict__ codes, float sc) {
    const uint2 c = *(const uint2*)codes;
    float f[8];
    e4m3x8(c.x, c.y, f);
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] *= sc;
    return f2bf8(f);
}

/* A wave's dot of the natural-order LDS vector with one E4M3 row of mix_down. A lane takes sixteen
 * consecutive codes of every 512-element step (one 16-byte load, so a wave requests 512 bytes like
 * the bf16 dots). The sixteen share one 128-group, whose scale multiplies the lane's partial sum
 * for the step. kU steps are loaded before the first is summed. K is a multiple of 512. */
constexpr long kHc8Step = kWave * 16;

__device__ __forceinline__ float hc8_dot16(const unsigned short* __restrict__ sv, uint4 c) {
    float wf[16], nf[16];
    e4m3x8(c.x, c.y, wf);
    e4m3x8(c.z, c.w, wf + 8);
    bf16x8(*(const uint4*)sv, nf);
    bf16x8(*(const uint4*)(sv + 8), nf + 8);
    float p = 0.0f;
#pragma unroll
    for (int t = 0; t < 16; ++t) p = fmaf(wf[t], nf[t], p);
    return p;
}

template <int kU>
__device__ __forceinline__ float hc8_dot_down(const unsigned short* __restrict__ sv,
                                              const unsigned char* __restrict__ wr, long K,
                                              int lane) {
    const float* __restrict__ sr = (const float*)(wr + K);
    float acc = 0.0f;
    long  j = 0;
    for (; j + kHc8Step * kU <= K; j += kHc8Step * kU) {
        uint4 c[kU];
        float s[kU];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const long e = j + (long)u * kHc8Step + lane * 16;
            c[u] = *(const uint4*)(wr + e);
            s[u] = sr[e / halo_hc8::kDownGroup];
        }
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const long e = j + (long)u * kHc8Step + lane * 16;
            acc = fmaf(hc8_dot16(sv + e, c[u]), s[u], acc);
        }
    }
    for (; j < K; j += kHc8Step) {
        const long e = j + lane * 16;
        acc = fmaf(hc8_dot16(sv + e, *(const uint4*)(wr + e)), sr[e / halo_hc8::kDownGroup], acc);
    }
    return acc;
}

/* One thread's quarter of an E4M3 row of mix_up against RG rows of `d` in LDS. The quarter is the
 * row's scale group (halo_hc8::up_group), so its partial sums are complete before scaling, and four
 * threads' sums make the row. All of a quarter's sixteen-code loads issue before the first is used
 * (five at lowrank 320); kUpMaxQ bounds lowrank at 512. */
constexpr int kUpMaxQ = 8;
template <int RG>
__device__ __forceinline__ void hc8_dot_up(const float* __restrict__ sd, long lr,
                                           const unsigned char* __restrict__ wr, int q4,
                                           float* acc) {
    const long G = lr / 4, k0 = (long)q4 * G;
    const int  nq = (int)(G / 16);
    uint4 c[kUpMaxQ];
#pragma unroll
    for (int i = 0; i < kUpMaxQ; ++i)
        if (i < nq) c[i] = *(const uint4*)(wr + k0 + 16 * i);
#pragma unroll
    for (int r = 0; r < RG; ++r) acc[r] = 0.0f;
#pragma unroll
    for (int i = 0; i < kUpMaxQ; ++i) {
        if (i >= nq) break;
        float wf[16];
        e4m3x8(c[i].x, c[i].y, wf);
        e4m3x8(c[i].z, c[i].w, wf + 8);
#pragma unroll
        for (int r = 0; r < RG; ++r) {
            const float* dr = sd + (long)r * lr + k0 + 16 * i;
#pragma unroll
            for (int t = 0; t < 16; ++t) acc[r] = fmaf(wf[t], dr[t], acc[r]);
        }
    }
    const float sc = ((const float*)(wr + lr))[q4];
#pragma unroll
    for (int r = 0; r < RG; ++r) {
        float a = acc[r] * sc;
        a += __shfl_xor(a, 1);
        a += __shfl_xor(a, 2);
        acc[r] = a;
    }
}

/* On the VALU, not the LDS crossbar (halo_common.h). The norm prologue reduces `hc + 1`
 * accumulators back to back with a runtime `hc`, so the exchanges form one long dependency chain.
 * Bit-identical: same lanes, same order. */
__device__ __forceinline__ float wave_sum(float v) { return halo_wave_sum(v); }

/* ============================================================== hc_enter
 *
 * h[c][i] = x[i] for every stream. One launch per step; its own op because the embedding lookup's
 * weight is [vocab, n] and a wide output would need a different weight shape. */
__global__ __launch_bounds__(kThreads)
void hc_enter_kernel(const unsigned short* __restrict__ x, unsigned short* __restrict__ h,
                     long rows, long n, int hc, long xrow, long hrow) {
    const long r = blockIdx.y;
    if (r >= rows) return;
    const unsigned short* xp = x + r * xrow;
    unsigned short*       hp = h + r * hrow;
    for (long i = (long)blockIdx.x * kThreads + threadIdx.x; i < n; i += (long)gridDim.x * kThreads) {
        const unsigned short v = xp[i];
        for (int c = 0; c < hc; ++c) hp[(long)c * n + i] = v;
    }
}

/* ============================================================== hc_write
 *
 * h[c][i] += inj[c] * y[i], in place on the unnormed stream: the read's normalisation feeds the
 * block input and gates only, never the residual. */
__global__ __launch_bounds__(kThreads)
void hc_write_kernel(const unsigned short* __restrict__ y, const unsigned short* __restrict__ inj,
                     unsigned short* __restrict__ h,
                     long rows, long n, int hc, long yrow, long irow, long hrow) {
    const long r = blockIdx.y;
    if (r >= rows) return;
    /* Gains go through LDS, not a register array: with a runtime `hc`, `g[c]` would be a dynamic
     * private index, which the compiler spills to scratch. */
    __shared__ float s_g[kHcMax];
    if ((int)threadIdx.x < hc) s_g[threadIdx.x] = bf2f(inj[r * irow + threadIdx.x]);
    __syncthreads();
    const unsigned short* yp = y + r * yrow;
    unsigned short*       hp = h + r * hrow;
    for (long i = (long)blockIdx.x * kThreads + threadIdx.x; i < n; i += (long)gridDim.x * kThreads) {
        const float v = bf2f(yp[i]);
        for (int c = 0; c < hc; ++c) {
            unsigned short* p = hp + (long)c * n + i;
            *p = f2bf(bf2f(*p) + s_g[c] * v);
        }
    }
}

/* ============================================================== hc_project
 *
 * The control vector (core/cvec.h): h[s] -= scale * (h[s] . v) * v for each of the hc streams of a
 * row whose `on` flag is 1 -- llama.cpp's `--cvec-mode project`, issued after a layer's write into
 * the stream. ONE direction for every stream: the dot is per stream, against the same v. One
 * workgroup per (row, stream), and a zero flag is not read and not written, so an unprojected
 * request's stream is the stock model's. */
__global__ __launch_bounds__(kThreads)
void hc_project_kernel(unsigned short* __restrict__ h, const float* __restrict__ v,
                       const int32_t* __restrict__ on, long rows, long n, long hrow, float scale) {
    const long r = blockIdx.y;
    if (r >= rows || on[r] == 0) return;   /* a row's switch: the whole workgroup takes it */
    unsigned short* hp = h + r * hrow + (long)blockIdx.x * n;

    float acc = 0.0f;
    for (long i = threadIdx.x; i < n; i += kThreads) acc += bf2f(hp[i]) * v[i];
    /* A wave sums itself and the waves meet in LDS, at kWave 32: the shuttle is the wave's own
     * width here, not the 64 a 64-wide __shfl_down would assume but gfx12 does not run. */
    __shared__ float s_g[kWaves];
    const float w = halo_wave_sum(acc);
    if ((threadIdx.x % kWave) == 0) s_g[threadIdx.x / kWave] = w;
    __syncthreads();
    if (threadIdx.x == 0) {
        float g = 0.0f;
        for (int k = 0; k < kWaves; ++k) g += s_g[k];
        s_g[0] = g * scale;
    }
    __syncthreads();

    const float g = s_g[0];
    for (long i = threadIdx.x; i < n; i += kThreads) hp[i] = f2bf(bf2f(hp[i]) - g * v[i]);
}

/* A wave's dot product of the LDS vector with a contiguous bf16 row, using 512-byte loads but

/* hc_read, its scratch sizing and mtp_enter are the rest of libr4d's r4d_hc_bf16.hip:
 * the split-K down product, the tiled up product, the fused mix/quantise finisher and the
 * fp8 trunk (kHcReadAStaticLds' 64 KiB LDS budget is a gfx12 figure). They are not ported
 * yet, so their entries (and mtp_enter's) stay in halo_stubs.cpp. */

}  // namespace

enum { HCE_X = 0, HCE_H };
enum { HCW_Y = 0, HCW_INJ, HCW_H };
enum { HCR_H = 0, HCR_W, HCR_DOWN, HCR_UP, HCR_INJW, HCR_X, HCR_INJ, HCR_Q, HCR_S, HCR_RQ, HCR_RS };


static int hc_common(const RadArgs* a, const RadTensor* h, long long* rows, long long* n,
                     long long* hc, long long* hrow) {
    *n  = halo_geti_or(a, "n", 0);
    *hc = halo_geti_or(a, "hc", 0);
    if (*n <= 0 || *hc <= 1 || *hc > kHcMax) return HALO_NO(RAD_E_SHAPE);
    const long long HN = *n * *hc;
    *hrow = halo_rows_pitch(h, HN, rows);
    if (*hrow < 0 || *rows <= 0) return HALO_NO(RAD_E_STRIDE);
    return RAD_OK;
}

extern "C" int halo_rad_hc_enter(const RadArgs* a, RadStream s) {
    const RadTensor* x = halo_opt(a, HCE_X);
    const RadTensor* h = halo_opt(a, HCE_H);
    if (!x || !h) return HALO_NO(RAD_E_INVAL);
    HALO_DT(x, RAD_BF16); HALO_DT(h, RAD_BF16);
    long long rows = 0, n = 0, hc = 0, hrow = 0;
    const int rc = hc_common(a, h, &rows, &n, &hc, &hrow);
    if (rc != RAD_OK) return rc;
    long long xrows = 0;
    const long long xrow = halo_rows_pitch(x, n, &xrows);
    if (xrow < 0 || xrows < rows) return HALO_NO(RAD_E_STRIDE);

    const unsigned blocks = (unsigned)((n + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(hc_enter_kernel, dim3(blocks, (unsigned)rows), dim3(kThreads), 0,
                       (hipStream_t)s, (const unsigned short*)x->data, (unsigned short*)h->data,
                       (long)rows, (long)n, (int)hc, (long)xrow, (long)hrow);
    return (int)hipGetLastError();
}

extern "C" int halo_rad_hc_write(const RadArgs* a, RadStream stream) {
    const RadTensor* y   = halo_opt(a, HCW_Y);
    const RadTensor* inj = halo_opt(a, HCW_INJ);
    const RadTensor* h   = halo_opt(a, HCW_H);
    if (!y || !inj || !h) return HALO_NO(RAD_E_INVAL);
    HALO_DT(y, RAD_BF16); HALO_DT(inj, RAD_BF16); HALO_DT(h, RAD_BF16);
    long long rows = 0, n = 0, hc = 0, hrow = 0;
    const int rc = hc_common(a, h, &rows, &n, &hc, &hrow);
    if (rc != RAD_OK) return rc;
    long long yrows = 0, irows = 0;
    const long long yrow = halo_rows_pitch(y, n, &yrows);
    const long long irow = halo_rows_pitch(inj, hc, &irows);
    if (yrow < 0 || irow < 0 || yrows < rows || irows < rows) return HALO_NO(RAD_E_STRIDE);

    const unsigned blocks = (unsigned)((n + kThreads - 1) / kThreads);
    return HALO_EMIT(stream, hc_write_kernel, dim3(blocks, (unsigned)rows), dim3(kThreads), 0,
                    (const unsigned short*)y->data,
                    (const unsigned short*)inj->data, (unsigned short*)h->data,
                    (long)rows, (long)n, (int)hc, (long)yrow, (long)irow, (long)hrow);
}

enum { HCP_H = 0, HCP_V, HCP_ON };

/* The control vector's projection. `v` is one layer's unit direction, [n], used for EVERY stream
 * (the dot is per stream against the same direction): a [hc, n] table is refused here rather than
 * read as one direction a stream, which is what a per-stream offset would silently do. */
extern "C" int halo_rad_hc_project(const RadArgs* a, RadStream stream) {
    const RadTensor* h  = halo_opt(a, HCP_H);
    const RadTensor* v  = halo_opt(a, HCP_V);
    const RadTensor* on = halo_opt(a, HCP_ON);
    if (!h || !v || !on) return HALO_NO(RAD_E_INVAL);
    HALO_DT(h, RAD_BF16); HALO_DT(v, RAD_F32); HALO_DT(on, RAD_I32);
    long long rows = 0, n = 0, hc = 0, hrow = 0;
    const int rc = hc_common(a, h, &rows, &n, &hc, &hrow);
    if (rc != RAD_OK) return rc;
    long long vrows = 0, onrows = 0;
    if (halo_rows_pitch(v, n, &vrows) < 0 || vrows != 1) return HALO_NO(RAD_E_STRIDE);
    if (halo_rows_pitch(on, 1, &onrows) < 0 || onrows < rows) return HALO_NO(RAD_E_STRIDE);

    return HALO_EMIT(stream, hc_project_kernel, dim3((unsigned)hc, (unsigned)rows), dim3(kThreads),
                    0, (unsigned short*)h->data, (const float*)v->data,
                    (const int32_t*)on->data, (long)rows, (long)n, (long)hrow,
                    (float)halo_getf(a, "scale", 1.0));
}
