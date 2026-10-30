// halo_gemm_fp8a16_m1.hip - gemm_fp8a16_nt_m1: block-scaled FP8 matvec against the
// checkpoint's own weight bytes. C[M, N] bf16 = X[M, K] bf16 @ dequant(W)[N, K]^T with W
// E4M3 row-major and a bf16 scale per 128x128 tile at [ceil(N/128)][ceil(K/128)].
//
// One wave owns `rb` weight rows (the row's one tunable; 0 is libhalo's rule, 4). Within a
// row the lanes walk 16-byte chunks of K, decode E4M3 in pairs, and fold the block scale of
// the chunk's 128-group into the partial before the xor butterfly closes the row. The weight
// stream is the whole cost: at M rows the activation row is re-read from L2 per row, which
// is why one weight read serves all M <= 8 rows inside the lane loop.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_fp8_wmma.h"

#include <cstdint>

namespace {

__global__ void __launch_bounds__(128) fp8a16_mv_kernel(const uint16_t* __restrict__ x,
                                                        const uint8_t* __restrict__ w,
                                                        const uint16_t* __restrict__ ws,
                                                        uint16_t* __restrict__ y,
                                                        int M, int N, long K, int sk, int rb,
                                                        long a_ld, long w_ld, long sw_ld,
                                                        long c_ld) {
    const int lane = (int)threadIdx.x & 31;
    const int wave = (int)threadIdx.x >> 5;
    const int n0 = ((int)blockIdx.x * 4 + wave) * rb;
    const long nchunks = K >> 4;                    /* 16 codes a chunk */
    for (int r = 0; r < rb; ++r) {
        const int n = n0 + r;
        if (n >= N) return;
        const uint8_t* wr = w + (long)n * w_ld;
        const uint16_t* sr = ws + (long)(n >> 7) * sw_ld;
        float acc[8];
#pragma unroll
        for (int m = 0; m < 8; ++m) acc[m] = 0.0f;
        for (long c = lane; c < nchunks; c += 32) {
            const long k0 = c << 4;
            const float wsc = bf16_to_f32(sr[k0 >> 7]);
            const uint4 wb = *(const uint4*)(wr + k0);
            const uint32_t qw[4] = { wb.x, wb.y, wb.z, wb.w };
            for (int m = 0; m < M; ++m) {
                const uint16_t* xr = x + (long)m * a_ld + k0;
                const uint4 xb0 = *(const uint4*)xr;          /* elements 0..7  */
                const uint4 xb1 = *(const uint4*)(xr + 8);    /* elements 8..15 */
                const uint32_t xp[8] = { xb0.x, xb0.y, xb0.z, xb0.w, xb1.x, xb1.y, xb1.z, xb1.w };
                float part = 0.0f;
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    /* qw[d]'s bytes are elements 4d..4d+3; xp[2d], xp[2d+1] pack their bf16
                     * halves in the same order. */
                    const v2f wlo = halo_fp8::e4m3_pk<false>(qw[d]);
                    const v2f whi = halo_fp8::e4m3_pk<true>(qw[d]);
                    part = fmaf(wlo.x, bf16_to_f32((uint16_t)(xp[2 * d] & 0xffffu)), part);
                    part = fmaf(wlo.y, bf16_to_f32((uint16_t)(xp[2 * d] >> 16)), part);
                    part = fmaf(whi.x, bf16_to_f32((uint16_t)(xp[2 * d + 1] & 0xffffu)), part);
                    part = fmaf(whi.y, bf16_to_f32((uint16_t)(xp[2 * d + 1] >> 16)), part);
                }
                acc[m] = fmaf(wsc, part, acc[m]);
            }
        }
        for (int m = 0; m < M; ++m) {
            float v = acc[m];
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) v += __shfl_xor(v, mask);
            if (lane == 0) y[(long)m * c_ld + n] = f32_to_bf16(v);
        }
    }
}

}  /* namespace */

extern "C" int halo_rad_gemm_fp8a16_m1(const RadArgs* a, RadStream stream) {
    const RadTensor* X = halo_opt(a, 0);
    const RadTensor* W = halo_opt(a, 2);
    const RadTensor* S = halo_opt(a, 3);
    const RadTensor* Y = halo_opt(a, 4);
    if (!X || !W || !S || !Y) return HALO_NO(RAD_E_INVAL);
    if (X->dtype != RAD_BF16 || W->dtype != RAD_F8E4M3 || S->dtype != RAD_BF16 ||
        Y->dtype != RAD_BF16)
        return HALO_NO(RAD_E_DTYPE);
    const char* dt = halo_gets(a, "dtype");
    if (!dt || std::strcmp(dt, "fp8a16")) return HALO_NO(RAD_E_DTYPE);
    if (halo_geti_or(a, "group", 128) != 128) return HALO_NO(RAD_E_SHAPE);

    const long long K = halo_geti_or(a, "K", halo_dim(W, (int)W->rank - 1));
    const long long N = halo_geti_or(a, "N", K > 0 ? halo_numel(W) / K : 0);
    /* The rows are X's; `M` is the band (at most 8 here), which a step may undershoot. */
    if (K <= 0 || (K % 128) || N <= 0) return HALO_NO(RAD_E_SHAPE);
    const long long M = halo_numel(X) / K;
    if (M < 1 || M > 8 || M > halo_geti_or(a, "M", M)) return HALO_NO(RAD_E_SHAPE);
    if (halo_numel(X) < M * K || halo_numel(W) < N * K || halo_numel(Y) < M * N)
        return HALO_NO(RAD_E_SHAPE);
    const long long sk = (K + 127) / 128;
    if (halo_numel(S) < ((N + 127) / 128) * sk) return HALO_NO(RAD_E_SHAPE);
    if (!halo_contig(X) || !halo_contig(W) || !halo_contig(S) || !halo_contig(Y))
        return HALO_NO(RAD_E_STRIDE);

    HaloMvCfg mv;
    halo_cfg_mv(a, &mv);
    const int rb = mv.rb > 0 ? mv.rb : 4;
    const int waves = (int)((N + rb - 1) / rb);
    fp8a16_mv_kernel<<<dim3((waves + 3) / 4), 128, 0, (hipStream_t)stream>>>(
        (const uint16_t*)halo_p(X), (const uint8_t*)halo_p(W), (const uint16_t*)halo_p(S),
        (uint16_t*)halo_p(Y), (int)M, (int)N, K, (int)sk, rb, K, K, sk, N);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

/* The raw-pointer entries libr4d exports beside the row (halo.h): same kernel, the row's
 * strides spelled out. Negative codes, as libr4d's: -1 a null pointer, -2 a non-positive
 * extent, -3 M past the band, -4 K not a whole 128-block, -5 an rb the kernel has no arm
 * for, -6 the launch failed. */
extern "C" int halo_gemm_fp8a16_nt_m1_block(void) { return 128; }
extern "C" int halo_gemm_fp8a16_nt_m1_max_m(void) { return 8; }

static int fp8a16_mv_launch(long a, long wq, long ws, long c, int M, int K, int N,
                            int a_ld, int c_ld, int w_ld, int sw_ld, int rb, long stream) {
    if (!a || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    if (M > 8) return -3;
    if (K % 128) return -4;
    if (rb != 0 && rb != 1 && rb != 2 && rb != 4 && rb != 8) return -5;
    if (!a_ld) a_ld = K;
    if (!c_ld) c_ld = N;
    if (!w_ld) w_ld = K;
    if (!sw_ld) sw_ld = K / 128;
    if (!rb) rb = 4;
    const int waves = (N + rb - 1) / rb;
    fp8a16_mv_kernel<<<dim3((waves + 3) / 4), 128, 0, (hipStream_t)stream>>>(
        (const uint16_t*)(uintptr_t)a, (const uint8_t*)(uintptr_t)wq,
        (const uint16_t*)(uintptr_t)ws, (uint16_t*)(uintptr_t)c, M, N, K, K / 128, rb,
        a_ld, w_ld, sw_ld, c_ld);
    return hipGetLastError() == hipSuccess ? 0 : -6;
}

extern "C" int halo_gemm_fp8a16_nt_m1(long a, long wq, long ws, long c, int M, int K, int N,
                                      int w_ld, int sw_ld, int rb, long stream) {
    if (!a || !wq || !ws || !c) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -2;
    if (M > 1) return -3;
    return fp8a16_mv_launch(a, wq, ws, c, M, K, N, K, N, w_ld, sw_ld, rb, stream);
}

extern "C" int halo_gemm_fp8a16_nt_mt(long a, long wq, long ws, long c, int M, int K, int N,
                                      int a_ld, int c_ld, int w_ld, int sw_ld, long stream) {
    return fp8a16_mv_launch(a, wq, ws, c, M, K, N, a_ld, c_ld, w_ld, sw_ld, 0, stream);
}
