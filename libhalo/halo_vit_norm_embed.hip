/* Derived from libr4d/r4d_vit_bf16.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_vit_norm_embed.hip -- the vision tower's non-fragment ops, gfx1151: layer norm with bias,
// the learned-grid position embedding and its antialiased form.
//
// These three are plain block-reduction and per-row kernels in libr4d as well: no WMMA fragment is
// involved, so the port is the rename plus the shared helpers (bf16_to_f32, f32_to_bf16,
// halo_wave_sum).
// Only the header changed. The tower's GEMM (libr4d's gemm_nt_bias / vit_gemm) is NOT here: it is a
// gfx12 fragment body and belongs to its own unit.
#include "halo.h"
#include "halo_args.h"
#include "halo_emit.h"
#include "halo_common.h"
#include "halo_gdn_wmma.h"

#include <cstdint>

namespace {

constexpr int kThr = 256;

template <uint32_t NW>
__device__ __forceinline__ float block_sum(float v, float* lds) {
    v = halo_wave_sum(v);
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) s += lds[w];
    __syncthreads();
    return s;
}

template <bool F32>
__device__ __forceinline__ float ldw(const void* p, uint32_t i) {
    return F32 ? ((const float*)p)[i] : bf16_to_f32(((const unsigned short*)p)[i]);
}

// ---------------------------------------------------------------------------- layernorm
// ref_layernorm: y = (x - mean) * rsqrt(var + eps) * w + b, biased variance, two passes (mean, then
// squared deviations); one-pass E[x^2] - E[x]^2 cancels badly when the mean is large relative to
// the spread. One block per row. `TX` is the row's element: bf16, or f32 for a residual stream kept
// in f32 (a video decoder's), read as it is; the output is bf16 either way.
__device__ __forceinline__ float ldx(const unsigned short* p, uint32_t i) { return bf16_to_f32(p[i]); }
__device__ __forceinline__ float ldx(const float* p, uint32_t i) { return p[i]; }

template <bool WF32, typename TX>
__global__ __launch_bounds__(kThr) void layernorm_kernel(
    const TX* __restrict__ x, const void* __restrict__ w, const void* __restrict__ b,
    unsigned short* __restrict__ y, uint32_t n, float eps, uint32_t x_ld, uint32_t y_ld) {
    __shared__ float lds[kThr / 32];
    const uint32_t m = blockIdx.x;
    const TX* __restrict__ xr = x + (size_t)m * x_ld;
    unsigned short* __restrict__ yr = y + (size_t)m * y_ld;
    float s = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += kThr) s += ldx(xr, i);
    const float mean = block_sum<kThr / 32>(s, lds) / (float)n;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        const float d = ldx(xr, i) - mean;
        ss += d * d;
    }
    const float r = rsqrtf(block_sum<kThr / 32>(ss, lds) / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        float v = (ldx(xr, i) - mean) * r * ldw<WF32>(w, i);
        if (b) v += ldw<WF32>(b, i);
        yr[i] = f32_to_bf16(v);
    }
}

// ---------------------------------------------------------------------------- grid_embed
// ref_grid_embed: a side x side table resampled bilinearly to row m's grid (corners aligned, edge
// taps clamped), four taps summed in f32 in libref's order, rounded, added to x and rounded again.
// `crd` is [4, planes]: row, column, grid height, grid width.
template <bool TF32>
__global__ __launch_bounds__(kThr) void grid_embed_kernel(
    unsigned short* __restrict__ x, const void* __restrict__ tab, const int32_t* __restrict__ crd,
    uint32_t n, int side, uint32_t x_ld, uint32_t plane) {
    const uint32_t m = blockIdx.x;
    const int g[4] = { crd[m], crd[plane + m], crd[2 * plane + m], crd[3 * plane + m] };
    int th[2], tw[2];
    float wh[2], ww[2];
    auto axis = [&](int idx, int size, int* t, float* wt) {
        const float src = (float)idx * (float)(side - 1) / (float)(size - 1 > 1 ? size - 1 : 1);
        const float fl = floorf(src);
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int tap = (int)fl + k;
            t[k] = tap < 0 ? 0 : (tap > side - 1 ? side - 1 : tap);
            const float d = fabsf(src - fl - (float)k);
            wt[k] = 1.0f - d > 0.0f ? 1.0f - d : 0.0f;
        }
    };
    axis(g[0], g[2], th, wh);
    axis(g[1], g[3], tw, ww);
    unsigned short* __restrict__ xr = x + (size_t)m * x_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        float acc = 0.0f;
#pragma unroll
        for (int ki = 0; ki < 2; ++ki)
#pragma unroll
            for (int kj = 0; kj < 2; ++kj)
                acc += ldw<TF32>(tab, (uint32_t)(th[ki] * side + tw[kj]) * n + i) *
                       (wh[ki] * ww[kj]);
        const float pe = bf16_to_f32(f32_to_bf16(acc));
        xr[i] = f32_to_bf16(bf16_to_f32(xr[i]) + pe);
    }
}

// ---------------------------------------------------------------------------- grid_embed_aa
// ref_grid_embed_aa: torch's antialiased half-pixel bilinear resample (SigLIP2's position table).
// Thread 0 builds the row's two tap lists (at most `side` each) in LDS; threads then take channels,
// summing taps in libref's order (columns within a row, then rows), rounded once before the add.
__device__ __forceinline__ int aa_taps(int at, int size, int side, float* w, int* lo) {
    size = size < 1 ? 1 : size;
    at = at < 0 ? 0 : (at >= size ? size - 1 : at);
    const float scale = (float)side / (float)size;
    const float support = scale > 1.0f ? scale : 1.0f, inv = 1.0f / support;
    const float center = scale * ((float)at + 0.5f);
    int a = (int)(center - support + 0.5f), b = (int)(center + support + 0.5f);
    a = a < 0 ? 0 : a;
    b = b > side ? side : b;
    float total = 0.0f;
    for (int j = a; j < b; ++j) {
        const float d = fabsf(((float)j - center + 0.5f) * inv);
        w[j - a] = d < 1.0f ? 1.0f - d : 0.0f;
        total += w[j - a];
    }
    for (int j = a; j < b; ++j) w[j - a] = total != 0.0f ? w[j - a] / total : 0.0f;
    *lo = a;
    return b > a ? b - a : 0;
}

template <bool TF32>
__global__ __launch_bounds__(kThr) void grid_embed_aa_kernel(
    unsigned short* __restrict__ x, const void* __restrict__ tab, const int32_t* __restrict__ crd,
    uint32_t n, int side, uint32_t x_ld, uint32_t plane) {
    extern __shared__ float aa_w[];           // [2 * side]: the row taps, then the column taps
    __shared__ int aa_span[4];                // h0, nh, w0, nw
    const uint32_t m = blockIdx.x;
    if (threadIdx.x == 0) {
        aa_span[1] = aa_taps(crd[m], crd[2 * plane + m], side, aa_w, &aa_span[0]);
        aa_span[3] = aa_taps(crd[plane + m], crd[3 * plane + m], side, aa_w + side, &aa_span[2]);
    }
    __syncthreads();
    const int h0 = aa_span[0], nh = aa_span[1], w0 = aa_span[2], nw = aa_span[3];
    unsigned short* __restrict__ xr = x + (size_t)m * x_ld;
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        float acc = 0.0f;
        for (int y = 0; y < nh; ++y) {
            float row = 0.0f;
            for (int c = 0; c < nw; ++c)
                row += ldw<TF32>(tab, (uint32_t)((h0 + y) * side + w0 + c) * n + i) * aa_w[side + c];
            acc += row * aa_w[y];
        }
        const float pe = bf16_to_f32(f32_to_bf16(acc));
        xr[i] = f32_to_bf16(bf16_to_f32(xr[i]) + pe);
    }
}


}  // namespace

// ============================================================== the plugin ABI entry points

/* ------------------------------------------------------------------ layernorm */
enum { LN_X = 0, LN_W, LN_B, LN_Y };

extern "C" int halo_rad_layernorm(const RadArgs* a, RadStream stream) {
    const RadTensor* x = halo_opt(a, LN_X);
    const RadTensor* w = halo_opt(a, LN_W);
    const RadTensor* b = halo_opt(a, LN_B);
    const RadTensor* y = halo_opt(a, LN_Y);
    if (!x || !w || !y) return HALO_NO(RAD_E_INVAL);
    /* The row bf16 or f32, the output bf16. */
    if ((x->dtype != RAD_BF16 && x->dtype != RAD_F32) || y->dtype != RAD_BF16)
        return HALO_NO(RAD_E_DTYPE);
    /* Gain and bias share one dtype, f32 or bf16. */
    if (w->dtype != RAD_F32 && w->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    if (b && b->dtype != w->dtype) return HALO_NO(RAD_E_DTYPE);
    if (!halo_contig(w) || (b && !halo_contig(b))) return HALO_NO(RAD_E_STRIDE);

    const long long n = halo_geti_or(a, "n", halo_dim(w, (int)w->rank - 1));
    if (n <= 0 || halo_numel(w) < n || (b && halo_numel(b) < n)) return HALO_NO(RAD_E_SHAPE);
    long long M = 0, My = 0;
    const long long x_ld = halo_rows_pitch(x, n, &M);
    const long long y_ld = halo_rows_pitch(y, n, &My);
    if (x_ld < 0 || y_ld < 0) return HALO_NO(RAD_E_STRIDE);
    if (M <= 0 || My < M) return HALO_NO(RAD_E_SHAPE);
    const float eps = (float)halo_getf(a, "eps", 1e-5);
#define HALO_LN(WF, TX)                                                                              \
    return HALO_EMIT(stream, (layernorm_kernel<WF, TX>), dim3((uint32_t)M), dim3(kThr), 0,          \
                    (const TX*)halo_p(x), halo_p(w), halo_p(b), (unsigned short*)halo_wp(y),           \
                    (uint32_t)n, eps, (uint32_t)x_ld, (uint32_t)y_ld)
    if (x->dtype == RAD_F32) {
        if (w->dtype == RAD_F32) HALO_LN(true, float);
        HALO_LN(false, float);
    }
    if (w->dtype == RAD_F32) HALO_LN(true, unsigned short);
    HALO_LN(false, unsigned short);
#undef HALO_LN
}

/* ------------------------------------------------------------------ grid_embed */
enum { GE_X = 0, GE_TAB, GE_CRD };

extern "C" int halo_rad_grid_embed(const RadArgs* a, RadStream stream) {
    const RadTensor* x   = halo_opt(a, GE_X);
    const RadTensor* tab = halo_opt(a, GE_TAB);
    const RadTensor* crd = halo_opt(a, GE_CRD);
    if (!x || !tab || !crd) return HALO_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || crd->dtype != RAD_I32) return HALO_NO(RAD_E_DTYPE);
    if (tab->dtype != RAD_BF16 && tab->dtype != RAD_F32) return HALO_NO(RAD_E_DTYPE);
    if (!halo_contig(tab)) return HALO_NO(RAD_E_STRIDE);
    const long long n = halo_geti_or(a, "n", halo_dim(tab, (int)tab->rank - 1));
    const long long side = halo_geti_or(a, "side", 0);
    if (n <= 0 || side <= 0 || halo_numel(tab) < side * side * n) return HALO_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long x_ld = halo_rows_pitch(x, n, &M);
    if (x_ld < 0) return HALO_NO(RAD_E_STRIDE);
    /* Four planes of coordinates, dense: [4, planes] with planes >= M. */
    if (crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < M || !halo_contig(crd))
        return HALO_NO(RAD_E_SHAPE);
    if (M <= 0) return HALO_NO(RAD_E_SHAPE);
    const uint32_t plane = (uint32_t)crd->shape[1];
    if (tab->dtype == RAD_F32)
        return HALO_EMIT(stream, grid_embed_kernel<true>, dim3((uint32_t)M), dim3(kThr), 0,
                        (unsigned short*)halo_wp(x), halo_p(tab), (const int32_t*)halo_p(crd),
                        (uint32_t)n, (int)side, (uint32_t)x_ld, plane);
    return HALO_EMIT(stream, grid_embed_kernel<false>, dim3((uint32_t)M), dim3(kThr), 0,
                    (unsigned short*)halo_wp(x), halo_p(tab), (const int32_t*)halo_p(crd),
                    (uint32_t)n, (int)side, (uint32_t)x_ld, plane);
}

/* grid_embed_aa: grid_embed's operands and checks, the antialiased resample. */
extern "C" int halo_rad_grid_embed_aa(const RadArgs* a, RadStream stream) {
    const RadTensor* x   = halo_opt(a, GE_X);
    const RadTensor* tab = halo_opt(a, GE_TAB);
    const RadTensor* crd = halo_opt(a, GE_CRD);
    if (!x || !tab || !crd) return HALO_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || crd->dtype != RAD_I32) return HALO_NO(RAD_E_DTYPE);
    if (tab->dtype != RAD_BF16 && tab->dtype != RAD_F32) return HALO_NO(RAD_E_DTYPE);
    if (!halo_contig(tab)) return HALO_NO(RAD_E_STRIDE);
    const long long n = halo_geti_or(a, "n", halo_dim(tab, (int)tab->rank - 1));
    const long long side = halo_geti_or(a, "side", 0);
    if (n <= 0 || side <= 0 || side > 1024 || halo_numel(tab) < side * side * n)
        return HALO_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long x_ld = halo_rows_pitch(x, n, &M);
    if (x_ld < 0) return HALO_NO(RAD_E_STRIDE);
    if (crd->rank != 2 || crd->shape[0] < 4 || crd->shape[1] < M || !halo_contig(crd))
        return HALO_NO(RAD_E_SHAPE);
    if (M <= 0) return HALO_NO(RAD_E_SHAPE);
    const uint32_t plane = (uint32_t)crd->shape[1];
    const size_t lds = (size_t)(2 * side) * sizeof(float);
    if (tab->dtype == RAD_F32)
        return HALO_EMIT(stream, grid_embed_aa_kernel<true>, dim3((uint32_t)M), dim3(kThr), lds,
                        (unsigned short*)halo_wp(x), halo_p(tab), (const int32_t*)halo_p(crd),
                        (uint32_t)n, (int)side, (uint32_t)x_ld, plane);
    return HALO_EMIT(stream, grid_embed_aa_kernel<false>, dim3((uint32_t)M), dim3(kThr), lds,
                    (unsigned short*)halo_wp(x), halo_p(tab), (const int32_t*)halo_p(crd),
                    (uint32_t)n, (int)side, (uint32_t)x_ld, plane);
}
