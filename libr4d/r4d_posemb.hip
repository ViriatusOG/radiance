// r4d_posemb.hip -- posemb_resample: a learned position table resampled bilinearly to a T_h x T_w
// grid and written out as a table of its own.
//
// Arithmetic matches libref (ref_elementwise.cpp) operation for operation: per axis the coordinate
// S * (i + 0.5) / T - 0.5 in f32, clamped into [0, S - 1], the taps floor and the next one clamped
// at S - 1, the four products summed in libref's order and stored once. One workgroup per output
// row, its threads striding the channels, so a grid of thousands of cells is thousands of
// workgroups and any n is one loop. The table and the output each take bf16 or f32; the taps are
// read in f32 whatever the table's storage. Not tuned: the table is read once per cell, which a
// grid of a few hundred cells does not need more than.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"   // bf2f / f32_to_bf16

#include <cstdint>

namespace {

constexpr int kThr = 256;

// The coordinate's taps for target index i of T over a source of size S: lo and hi, and the weight
// of hi (lo carries 1 - frac). Clamped so that the coordinate stays on the table.
__device__ __forceinline__ void posemb_axis(int i, int S, int T, int* lo, int* hi, float* frac) {
    const float c = (float)S * ((float)i + 0.5f) / (float)T - 0.5f;
    const float cc = c < 0.0f ? 0.0f : (c > (float)(S - 1) ? (float)(S - 1) : c);
    const int f = (int)floorf(cc);
    *lo = f;
    *hi = f + 1 < S ? f + 1 : S - 1;
    *frac = cc - (float)f;
}

template <bool TF32>
__device__ __forceinline__ float ld_tab(const void* p, uint32_t i) {
    return TF32 ? ((const float*)p)[i] : bf2f(((const unsigned short*)p)[i]);
}

// One workgroup per output row r = i_h * T_w + i_w.
template <bool TF32, bool YF32>
__global__ __launch_bounds__(kThr) void posemb_resample_kernel(
    void* __restrict__ y, const void* __restrict__ tab, int Sh, int Sw, int Th, int Tw,
    uint32_t n) {
    const uint32_t r = blockIdx.x;
    int h[2], w[2];
    float fh, fw;
    posemb_axis((int)(r / (uint32_t)Tw), Sh, Th, &h[0], &h[1], &fh);
    posemb_axis((int)(r % (uint32_t)Tw), Sw, Tw, &w[0], &w[1], &fw);
    const float wh[2] = { 1.0f - fh, fh }, ww[2] = { 1.0f - fw, fw };
    for (uint32_t i = threadIdx.x; i < n; i += kThr) {
        float acc = 0.0f;
#pragma unroll
        for (int p = 0; p < 2; ++p)
#pragma unroll
            for (int q = 0; q < 2; ++q)
                acc += ld_tab<TF32>(tab, (uint32_t)(h[p] * Sw + w[q]) * n + i) * (wh[p] * ww[q]);
        if (YF32)
            ((float*)y)[(size_t)r * n + i] = acc;
        else
            ((unsigned short*)y)[(size_t)r * n + i] = f32_to_bf16(acc);
    }
}

}  // namespace

/* ------------------------------------------------------------------ posemb_resample */
enum { PR_TAB = 0, PR_Y };

template <bool TF32, bool YF32>
static int launch_posemb(RadStream stream, void* y, const void* tab, int Sh, int Sw, int Th, int Tw,
                         uint32_t n) {
    auto k = posemb_resample_kernel<TF32, YF32>;
    return R4D_EMIT(stream, k, dim3((uint32_t)(Th * Tw)), dim3(kThr), 0, y, tab, Sh, Sw, Th, Tw, n);
}

extern "C" int r4d_rad_posemb_resample(const RadArgs* a, RadStream stream) {
    const RadTensor* tab = r4d_opt(a, PR_TAB);
    const RadTensor* y   = r4d_opt(a, PR_Y);
    if (!tab || !y) return R4D_NO(RAD_E_INVAL);
    if (tab->dtype != RAD_BF16 && tab->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (y->dtype != RAD_BF16 && y->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(tab) || !r4d_contig(y)) return R4D_NO(RAD_E_STRIDE);
    const long long Sh = r4d_geti_or(a, "S_h", 0), Sw = r4d_geti_or(a, "S_w", 0);
    const long long Th = r4d_geti_or(a, "T_h", 0), Tw = r4d_geti_or(a, "T_w", 0);
    const long long n = r4d_geti_or(a, "n", 0);
    if (Sh <= 0 || Sw <= 0 || Th < 1 || Tw < 1 || n <= 0) return R4D_NO(RAD_E_SHAPE);
    if (tab->rank != 2 || tab->shape[0] != Sh * Sw || tab->shape[1] != n) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(y) < Th * Tw * n) return R4D_NO(RAD_E_SHAPE);
    const bool tf = tab->dtype == RAD_F32, yf = y->dtype == RAD_F32;
    void* yp = r4d_wp(y);
    const void* tp = r4d_p(tab);
    const int sh = (int)Sh, sw = (int)Sw, th = (int)Th, tw = (int)Tw;
    const uint32_t nn = (uint32_t)n;
    if (tf && yf) return launch_posemb<true, true>(stream, yp, tp, sh, sw, th, tw, nn);
    if (tf)       return launch_posemb<true, false>(stream, yp, tp, sh, sw, th, tw, nn);
    if (yf)       return launch_posemb<false, true>(stream, yp, tp, sh, sw, th, tw, nn);
    return launch_posemb<false, false>(stream, yp, tp, sh, sw, th, tw, nn);
}
