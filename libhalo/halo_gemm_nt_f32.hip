// halo_gemm_nt_f32.hip - gemm_nt_f32_bf16: the F32 routing and index gate projection.
// C[M, N] f32 = X[M, K] bf16 @ W[N, K] f32^T, pitched rows on both sides.
//
// The accumulation order is part of the row's contract (the selftest compares byte for byte
// with the host arithmetic): lane l folds k = l, l + 32, ... with fmaf, then an xor butterfly
// over 16, 8, 4, 2, 1 closes the wave. Any tiling must keep that order, so the kernel is one
// wave per output element; the shapes (N <= 512, K <= 16384, small M) make that the right
// size anyway.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"

#include <cstdint>

namespace {

__global__ void __launch_bounds__(32) gemm_nt_f32_kernel(const uint16_t* __restrict__ x,
                                                         const float* __restrict__ w,
                                                         float* __restrict__ y,
                                                         int K, long ldx, long ldw, long ldy) {
    const int m = (int)blockIdx.y;
    const int n = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    const uint16_t* xr = x + (long)m * ldx;
    const float* wr = w + (long)n * ldw;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32)
        acc = fmaf(bf16_to_f32(xr[k]), wr[k], acc);
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) acc += __shfl_xor(acc, mask);
    if (lane == 0) y[(long)m * ldy + n] = acc;
}

}  /* namespace */

extern "C" int halo_rad_gemm_nt_f32(const RadArgs* a, RadStream stream) {
    const RadTensor* X = halo_opt(a, 0);
    const RadTensor* W = halo_opt(a, 1);
    const RadTensor* Y = halo_opt(a, 2);
    if (!X || !W || !Y) return HALO_NO(RAD_E_INVAL);
    if (X->dtype != RAD_BF16 || W->dtype != RAD_F32 || Y->dtype != RAD_F32)
        return HALO_NO(RAD_E_DTYPE);
    const char* dt = halo_gets(a, "dtype");
    if (dt && *dt && std::strcmp(dt, "bf16")) return HALO_NO(RAD_E_DTYPE);

    const long long K = halo_geti_or(a, "K", halo_dim(W, (int)W->rank - 1));
    const long long N = halo_geti_or(a, "N", K > 0 ? halo_numel(W) / K : 0);
    if (K <= 0 || N <= 0 || halo_numel(W) < N * K) return HALO_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long ldy = halo_rows_pitch(Y, N, &M);
    const long long ldx = halo_rows_pitch(X, K, &Mx);
    long long Mw = 0;
    const long long ldw = halo_rows_pitch(W, K, &Mw);
    if (ldy < 0 || ldx < 0 || ldw < 0) return HALO_NO(RAD_E_STRIDE);
    if (M <= 0 || Mx < M || Mw < N) return HALO_NO(RAD_E_SHAPE);
    if (M > 65535 || N > 0x7fffffffll) return HALO_NO(RAD_E_SHAPE);

    dim3 grid((unsigned)N, (unsigned)M);
    hipLaunchKernelGGL((gemm_nt_f32_kernel), grid, 32, 0, (hipStream_t)stream, 
        (const uint16_t*)halo_p(X), (const float*)halo_p(W), (float*)halo_p(Y),
        (int)K, ldx, ldw, ldy);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
