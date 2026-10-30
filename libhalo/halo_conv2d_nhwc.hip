/* Derived from libr4d/r4d_gen.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_conv2d_nhwc.hip -- the image decoder's convolution, gfx1151: an implicit GEMM over
// (tap, channel) with the weight stored tap-major, plus the direct form for an input whose channel
// count is not a multiple of eight.
//
// The implicit GEMM is libr4d's, with the two gfx11 fragment changes halo_gdn_wmma.h states:
//
//   * the fragment pointer loses `4 * (lane >> 4)`: on RDNA 3.5 a lane holds the WHOLE 16-wide k
//     run (32 bytes), on RDNA4 only its half's 8 values 16 bytes apart. `fragP` is therefore one
//     dwordx4 pair at the un-split k offset.
//   * the accumulator's element e is D[2e + (lane>>4)][lane & 15], not D[8*(lane>>4) + e][lane&15].
//     The blocked store below is unchanged in SHAPE because `accPack` exchanges the two halves'
//     rows first (four v_permlanex16), putting the tile's sixteen consecutive channels back in one
//     lane -- so the fail-safe is 16 bytes a lane, as libr4d writes it.
//
// Arithmetic is untouched: f32 accumulation, one WMMA per 16 K in ascending order, bias added
// after the product, RTNE bf16 stores (libref/ref_gen.cpp's conv2d_nhwc).
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_gdn_wmma.h"

#include <cstdint>
#include <cstring>

namespace {

constexpr int kCvT = 256;

inline int cv_st_of(hipError_t e) { return e == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE); }
inline unsigned cv_blocks_for(int64_t n) { return (unsigned)((n + kCvT - 1) / kCvT); }
inline bool cv_is_bf16(const RadTensor* t) { return t && t->dtype == RAD_BF16; }

// A dense [H, W * c] image operand; `pitched` allows a row pitch wider than the row.
inline bool cv_image_of(const RadTensor* t, long long c, long long* H, long long* W,
                        bool pitched = false) {
    if (!t || t->rank != 2 || c <= 0 || t->shape[1] % c || t->stride[1] != 1 ||
        (pitched ? t->stride[0] < t->shape[1] : t->stride[0] != t->shape[1]))
        return false;
    *H = t->shape[0];
    *W = t->shape[1] / c;
    return true;
}

// ---------------------------------------------------------------------------- conv2d_nhwc
// An implicit GEMM: y[pixel][cout] = sum over (tap, cin) of x_virtual[pixel's tap][cin] *
// w[cout][tap][cin]. The weight is stored tap-major (halo_rows' relayout turns PyTorch's
// [cout][cin][k][k] into it at load), so K runs over taps then channels and a 64-wide K tile of the
// image is eight 16-byte reads of NHWC rows; no im2col buffer.
//
// The block is halo_gemm_bf16_nt_tiled's: eight waves, an activation tile of BP pixels and a weight
// tile of BC output channels staged through LDS 64 K at a time, with the next tile's loads in
// flight under the current WMMA. The weight is the WMMA's A operand and the pixels its B, so a
// lane's accumulator is one pixel and sixteen output channels of it.
//
// An eight-channel input chunk never straddles a tap because cin is a multiple of eight (the shim
// sends anything else to the direct kernel); a chunk past the image, K or the last pixel is zeros.
//
// Kernel size and upsampling are template parameters (KS, UP; 0 reads the argument): at runtime
// they cost two integer divisions per staged chunk, at 3 and 2 a multiply-shift and a shift.
#define CV_BK 64
#define CV_PITCH (CV_BK + 8)

template <int TP, int TC, int KS, int UP>
__global__ __launch_bounds__(256) void conv_igemm_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ bias, uint16_t* __restrict__ y, int Hv, int Wv, int cin, int Wo,
    int cout, int k, int stride, int up, int pt, int pl, int x_ld, int y_ld, int Mpx, int K) {
    constexpr int BP = 2 * TP * 16, BC = 4 * TC * 16;
    constexpr int CPR = CV_BK / 8;
    constexpr int AL = BP * CPR / 256, WL = BC * CPR / 256;
    static_assert(AL >= 1 && WL >= 1, "every thread stages whole chunks");
    __shared__ __attribute__((aligned(16))) uint16_t sA[BP * CV_PITCH];
    __shared__ __attribute__((aligned(16))) uint16_t sW[BC * CV_PITCH];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wp = wave / 4, wc = wave % 4;
    const int p0 = blockIdx.x * BP, c0 = blockIdx.y * BC;
    const int kq = KS ? KS : k, kk2 = kq * kq;

    int a_oy[AL], a_ox[AL], a_t[AL], a_ci[AL];
    bool a_ok[AL];
    uint16_t* adst[AL];
    #pragma unroll
    for (int i = 0; i < AL; ++i) {
        const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
        const int m = p0 + r;
        a_ok[i] = m < Mpx;
        const int mm = a_ok[i] ? m : 0;
        a_oy[i] = mm / Wo;
        a_ox[i] = mm % Wo;
        a_t[i] = e / cin;
        a_ci[i] = e % cin;
        adst[i] = sA + r * CV_PITCH + e;
    }
    size_t w_off[WL];
    int w_e[WL];
    bool w_ok[WL];
    uint16_t* wdst[WL];
    #pragma unroll
    for (int i = 0; i < WL; ++i) {
        const int c = tid + 256 * i, r = c / CPR, e = (c % CPR) * 8;
        const int co = c0 + r;
        w_ok[i] = co < cout;
        w_off[i] = (size_t)(w_ok[i] ? co : 0) * K + e;
        w_e[i] = e;
        wdst[i] = sW + r * CV_PITCH + e;
    }
    const halo_u32x4 zero = {0u, 0u, 0u, 0u};
    auto loadA = [&](int i) -> halo_u32x4 {
        if (!a_ok[i] || a_t[i] >= kk2) return zero;
        const int kh = a_t[i] / kq, kw = a_t[i] - kh * kq;
        const int iy = a_oy[i] * stride + kh - pt, ix = a_ox[i] * stride + kw - pl;
        if (iy < 0 || iy >= Hv || ix < 0 || ix >= Wv) return zero;
        /* Both are non-negative here, so a halving is a shift. */
        const int sy = UP == 1 ? iy : UP == 2 ? (iy >> 1) : iy / up;
        const int sx = UP == 1 ? ix : UP == 2 ? (ix >> 1) : ix / up;
        return *(const halo_u32x4*)(x + (size_t)sy * x_ld + (size_t)sx * cin + a_ci[i]);
    };
    auto loadW = [&](int i, int kt) -> halo_u32x4 {
        if (!w_ok[i] || kt * CV_BK + w_e[i] >= K) return zero;
        return *(const halo_u32x4*)(w + w_off[i] + (size_t)kt * CV_BK);
    };

    /* This lane's fragment row of each tile, at the un-split k offset: gfx11 has no half-wave k
     * term (halo_gdn_wmma.h's rowP). */
    const uint16_t* fa = sA + (wp * TP * 16 + (lane & 15)) * CV_PITCH;
    const uint16_t* fw = sW + (wc * TC * 16 + (lane & 15)) * CV_PITCH;

    v8f acc[TC][TP];
    #pragma unroll
    for (int j = 0; j < TC; ++j)
        #pragma unroll
        for (int i = 0; i < TP; ++i) acc[j][i] = (v8f)(0.f);

    halo_u32x4 ra[AL], rw[WL];
    #pragma unroll
    for (int i = 0; i < AL; ++i) ra[i] = loadA(i);
    #pragma unroll
    for (int i = 0; i < WL; ++i) rw[i] = loadW(i, 0);

    const int nk = (K + CV_BK - 1) / CV_BK;
    for (int kt = 0; kt < nk; ++kt) {
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < AL; ++i) *(halo_u32x4*)adst[i] = ra[i];
        #pragma unroll
        for (int i = 0; i < WL; ++i) *(halo_u32x4*)wdst[i] = rw[i];
        __syncthreads();
        if (kt + 1 < nk) {
            #pragma unroll
            for (int i = 0; i < AL; ++i) {
                a_ci[i] += CV_BK;
                while (a_ci[i] >= cin) { a_ci[i] -= cin; ++a_t[i]; }
                ra[i] = loadA(i);
            }
            #pragma unroll
            for (int i = 0; i < WL; ++i) rw[i] = loadW(i, kt + 1);
        }
        #pragma unroll
        for (int s = 0; s < CV_BK / 16; ++s) {
            v16bf af[TP], wf[TC];
            #pragma unroll
            for (int i = 0; i < TP; ++i) af[i] = fragP(fa + i * 16 * CV_PITCH + s * 16);
            #pragma unroll
            for (int j = 0; j < TC; ++j) wf[j] = fragP(fw + j * 16 * CV_PITCH + s * 16);
            #pragma unroll
            for (int j = 0; j < TC; ++j)
                #pragma unroll
                for (int i = 0; i < TP; ++i) acc[j][i] = mma(wf[j], af[i], acc[j][i]);
        }
    }

    /* The store. Lane l's element e is channel cb + accRow(l, e) of pixel accCol(l), so the bias is
     * added per element and accPack then hands each lane its own eight consecutive channels back. */
    #pragma unroll
    for (int j = 0; j < TC; ++j)
        #pragma unroll
        for (int i = 0; i < TP; ++i) {
            const int px = p0 + (wp * TP + i) * 16 + accCol(lane);
            if (px >= Mpx) continue;
            const int cb = c0 + (wc * TC + j) * 16;
            const int oy = px / Wo, ox = px % Wo;
            uint16_t* dst = y + (size_t)oy * y_ld + (size_t)ox * cout;
            if (cb + 16 <= cout && (cout & 7) == 0) {
                float f[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    f[e] = acc[j][i][e] + bf16_to_f32(bias[cb + accRow(lane, e)]);
                *(uint4*)(dst + cb + accPackRow(lane)) = accPack(f, lane);
            } else {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int co = cb + accRow(lane, e);
                    if (co < cout) dst[co] = f32_to_bf16(acc[j][i][e] + bf16_to_f32(bias[co]));
                }
            }
        }
}

// The direct form, for an input whose channel count is not a multiple of eight (an RGBA image into
// the first layer): a thread an output element, f32 over (tap, cin), the same tap-major weight.
__global__ __launch_bounds__(kCvT) void conv_direct_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ bias, uint16_t* __restrict__ y, int Hv, int Wv, int cin, int Wo,
    int cout, int k, int stride, int up, int pt, int pl, int x_ld, int y_ld, int64_t total) {
    const int64_t t = (int64_t)blockIdx.x * kCvT + threadIdx.x;
    if (t >= total) return;
    const int co = (int)(t % cout);
    const int64_t px = t / cout;
    const int oy = (int)(px / Wo), ox = (int)(px % Wo);
    const int K = k * k * cin;
    float acc = 0.0f;
    for (int kh = 0; kh < k; ++kh) {
        const int iy = oy * stride + kh - pt;
        if (iy < 0 || iy >= Hv) continue;
        for (int kw = 0; kw < k; ++kw) {
            const int ix = ox * stride + kw - pl;
            if (ix < 0 || ix >= Wv) continue;
            const uint16_t* xp = x + (size_t)(iy / up) * x_ld + (size_t)(ix / up) * cin;
            const uint16_t* wp = w + (size_t)co * K + (size_t)(kh * k + kw) * cin;
            for (int ci = 0; ci < cin; ++ci) acc += bf16_to_f32(wp[ci]) * bf16_to_f32(xp[ci]);
        }
    }
    y[(size_t)oy * y_ld + (size_t)ox * cout + co] = f32_to_bf16(acc + bf16_to_f32(bias[co]));
}

}  // namespace

// ============================================================== the plugin ABI entry point
extern "C" int halo_rad_conv2d_nhwc(const RadArgs* a, RadStream s) {
    const RadTensor *x = halo_opt(a, 0), *w = halo_opt(a, 1), *b = halo_opt(a, 2),
                    *y = halo_opt(a, 3);
    if (!x || !w || !b || !y) return HALO_NO(RAD_E_INVAL);
    if (!cv_is_bf16(x) || !cv_is_bf16(w) || !cv_is_bf16(b) || !cv_is_bf16(y))
        return HALO_NO(RAD_E_DTYPE);
    const long long cin = halo_geti_or(a, "cin", 0), cout = halo_geti_or(a, "cout", 0);
    const long long k = halo_geti_or(a, "k", 0), stride = halo_geti_or(a, "stride", 1);
    const long long up = halo_geti_or(a, "up", 1);
    const char* pad = halo_gets(a, "pad");
    const bool down = pad && !std::strcmp(pad, "down");
    const bool causal = pad && !std::strcmp(pad, "causal");
    if (pad && !down && !causal && std::strcmp(pad, "same")) return HALO_NO(RAD_E_INVAL);
    if (cin <= 0 || cout <= 0 || k <= 0 || stride <= 0 || up <= 0) return HALO_NO(RAD_E_SHAPE);
    long long Hin = 0, Win = 0, Ho = 0, Wo = 0;
    /* y may be a column slice of wider rows (a channel written into its own columns): both kernels
     * store at its row pitch. */
    if (!cv_image_of(x, cin, &Hin, &Win) || !cv_image_of(y, cout, &Ho, &Wo, true))
        return HALO_NO(RAD_E_SHAPE);
    const long long Hv = Hin * up, Wv = Win * up;
    /* Rows padded above `pt` and below `pb`, columns `pl` left and `pr` right. */
    const long long pt = down ? 0 : causal ? k - 1 : k / 2, pb = down ? 1 : causal ? 0 : k / 2;
    const long long pl = down ? 0 : k / 2, pr = down ? 1 : k / 2;
    if ((Hv + pt + pb - k) / stride + 1 != Ho || (Wv + pl + pr - k) / stride + 1 != Wo)
        return HALO_NO(RAD_E_SHAPE);
    const long long K = k * k * cin;
    if (halo_numel(w) < cout * K || halo_numel(b) < cout) return HALO_NO(RAD_E_SHAPE);
    const long long Mpx = Ho * Wo;
    if (Mpx <= 0 || Mpx >= (1ll << 31) || (long long)Ho * Wo * cout >= (1ll << 40))
        return HALO_NO(RAD_E_SHAPE);
    hipStream_t st = (hipStream_t)s;
    if (cin % 8 || ((uintptr_t)x->data & 15) || ((uintptr_t)w->data & 15)) {
        const long long total = Mpx * cout;
        hipLaunchKernelGGL(conv_direct_kernel, dim3(cv_blocks_for(total)), dim3(kCvT), 0, st,
                           (const uint16_t*)x->data, (const uint16_t*)w->data,
                           (const uint16_t*)b->data, (uint16_t*)y->data, (int)Hv, (int)Wv,
                           (int)cin, (int)Wo, (int)cout, (int)k, (int)stride, (int)up, (int)pt,
                           (int)pl, (int)x->stride[0], (int)y->stride[0], (int64_t)total);
        return cv_st_of(hipGetLastError());
    }
    // 128 pixels a block, and the widest channel tile (192, 128, then 64) that leaves under a third
    // of its last block idle. A wider tile does two or three times the WMMA per K step for the same
    // two barriers, which pays even with the last block partly empty. The rule is libr4d's; on
    // gfx1151 a lane's fragment is twice as wide, so a wave's WMMA count per K step is unchanged
    // (TP x TC accumulators) and the choice was not re-measured here.
    auto fits = [cout](long long bc) {
        const long long padded = (cout + bc - 1) / bc * bc;
        return (padded - cout) * 3 < padded;
    };
#define HALO_CONV(TP, TC, KS, UP)                                                                 \
    hipLaunchKernelGGL((conv_igemm_kernel<TP, TC, KS, UP>),                                      \
                       dim3((unsigned)((Mpx + 2 * TP * 16 - 1) / (2 * TP * 16)),                  \
                            (unsigned)((cout + 4 * TC * 16 - 1) / (4 * TC * 16))),               \
                       dim3(256), 0, st, (const uint16_t*)x->data, (const uint16_t*)w->data,     \
                       (const uint16_t*)b->data, (uint16_t*)y->data, (int)Hv, (int)Wv, (int)cin, \
                       (int)Wo, (int)cout, (int)k, (int)stride, (int)up, (int)pt, (int)pl,       \
                       (int)x->stride[0], (int)y->stride[0], (int)Mpx, (int)K)
    /* The decoder's two kernel sizes at its two sampling rates, and the generic form for the rest. */
#define HALO_CONV_K(TP, TC)                                                                      \
    if (k == 3 && up == 1)      HALO_CONV(TP, TC, 3, 1);                                         \
    else if (k == 3 && up == 2) HALO_CONV(TP, TC, 3, 2);                                         \
    else if (k == 1 && up == 1) HALO_CONV(TP, TC, 1, 1);                                         \
    else                        HALO_CONV(TP, TC, 0, 0)
    if (fits(192))      { HALO_CONV_K(4, 3); }
    else if (fits(128)) { HALO_CONV_K(4, 2); }
    else                { HALO_CONV_K(4, 1); }
#undef HALO_CONV_K
#undef HALO_CONV
    return cv_st_of(hipGetLastError());
}
