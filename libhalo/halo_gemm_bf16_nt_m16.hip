/* The kernel is carried over from the strix-halo-vllm project's libhalo
 * (halo_gemm_bf16_nt.hip), itself a port of libr4d's earlier dot2 m16 kernel,
 * measured on this part in production; see libhalo/README.md. The shim and
 * split law are written for radiance. */
// halo_gemm_bf16_nt_m16.hip - halo_gemm_bf16_nt_m16: skinny bf16 GEMM at M <= 16, gfx1151.
//
// C[M,N] = A[M,K] @ W[N,K]^T, bf16 in and out, fp32 accumulate. A wave owns one output column
// and one contiguous K slice (SK splits, reduced through LDS in split order, so the result does
// not depend on scheduling); each lane holds eight consecutive K of the slice and dots them with
// v_dot2_f32_bf16. No WMMA: at these heights the weight read is the whole cost and this streams
// it faster than the fragment form (halo_gemm_bf16_nt_m64.hip) -- the opposite of the R9700,
// where libr4d retired this kernel in favour of forwarding to its WMMA. The reduction tree
// differs from the WMMA row's, so the two rows' outputs differ in low bits; each is
// deterministic. Weight loads for the whole slice are issued up front (eight 256-element tiles
// in registers) so the latency overlaps the dots.
//
// A masked tail tile means K/SK need only be a multiple of 8 (VEC), not of 256: masking is at
// lane granularity and no lane ever holds a partial 128-bit load.
#include "halo.h"
#include "halo_gemm_args.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#define WAVE 32
#define VEC  8                       // bf16 per 128-bit (dwordx4) load
#define HALO_GEMM16_MAX_M 16

typedef int    int4v  __attribute__((ext_vector_type(4)));
typedef __bf16 bf162  __attribute__((ext_vector_type(2)));   // v_dot2_f32_bf16 operand

template <int MT, int TEMPORAL>
__global__ __launch_bounds__(1024) void halo_gemm_bf16_nt_m16_kernel(
        const __hip_bfloat16* __restrict__ A, const __hip_bfloat16* __restrict__ W,
        __hip_bfloat16* __restrict__ C, int K, int N, int WV, int SK) {
    extern __shared__ float red[];                      // WV*SK*MT split-K scratch
    const int tid = threadIdx.x, wave = tid / WAVE, lane = tid % WAVE;
    const int col_local = wave / SK, ksplit = wave % SK;
    const int j = blockIdx.x * WV + col_local;
    const int Ks = K / SK, kbase = ksplit * Ks;

    float acc[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) acc[i] = 0.f;

    if (j < N) {
        const __hip_bfloat16* Wj = W + (size_t)j * K + kbase + lane * VEC;
        const int niter = Ks / (WAVE * VEC);
        const int rem = Ks - niter * (WAVE * VEC);
        const int tail_lanes = rem / VEC;
        const bool has_tail = rem > 0 && lane < tail_lanes;
        int4v wbuf[8];
#pragma unroll
        for (int t = 0; t < 8; ++t)                     // all weight loads up-front
            if (t < niter) wbuf[t] = TEMPORAL ? *(const int4v*)(Wj + t * WAVE * VEC)
                                              : __builtin_nontemporal_load((const int4v*)(Wj + t * WAVE * VEC));
        int4v wtail;
        if (has_tail) {
            const __hip_bfloat16* Wt = Wj + niter * WAVE * VEC;
            wtail = TEMPORAL ? *(const int4v*)Wt
                             : __builtin_nontemporal_load((const int4v*)Wt);
        }
        for (int t = 0; t < niter; ++t) {
            const bf162* wp = (const bf162*)&wbuf[t];
            const int kk = kbase + lane * VEC + t * WAVE * VEC;
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                int4 av = *(const int4*)(A + (size_t)i * K + kk);
                const bf162* ap = (const bf162*)&av;
#pragma unroll
                for (int p = 0; p < VEC / 2; ++p)
                    acc[i] = __builtin_amdgcn_fdot2_f32_bf16(ap[p], wp[p], acc[i], false);
            }
        }
        if (has_tail) {
            const bf162* wp = (const bf162*)&wtail;
            const int kk = kbase + lane * VEC + niter * WAVE * VEC;
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                int4 av = *(const int4*)(A + (size_t)i * K + kk);
                const bf162* ap = (const bf162*)&av;
#pragma unroll
                for (int p = 0; p < VEC / 2; ++p)
                    acc[i] = __builtin_amdgcn_fdot2_f32_bf16(ap[p], wp[p], acc[i], false);
            }
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            float v = acc[i];
            for (int off = WAVE / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
            if (lane == 0) red[(col_local * SK + ksplit) * MT + i] = v;
        }
    }
    __syncthreads();
    if (ksplit == 0 && j < N && lane < MT) {            // split-K reduce, one column of C
        float v = 0.f;
        for (int s = 0; s < SK; ++s) v += red[(col_local * SK + s) * MT + lane];
        C[(size_t)lane * N + j] = (__hip_bfloat16)v;
    }
}

static int gemm_bf16_m16_launch(long a, long w, long c, int M, int K, int N, int WV, int SK,
                                int nt, long stream) {
    if (M < 1 || M > HALO_GEMM16_MAX_M) return -1;
    if (K % SK != 0 || (K / SK) % VEC != 0) return -2;
    if (WV * SK * WAVE > 1024) return -3;
    if ((K / SK) / (WAVE * VEC) > 8) return -4;          // wbuf holds 8 tiles
    dim3 grid((N + WV - 1) / WV), block(WV * SK * WAVE);
    size_t smem = (size_t)WV * SK * M * sizeof(float);
    auto A = (const __hip_bfloat16*)a; auto Wp = (const __hip_bfloat16*)w;
    auto Cp = (__hip_bfloat16*)c; auto st = (hipStream_t)stream;
#define LAUNCH(MT)                                                                    \
    do { if (nt) hipLaunchKernelGGL((halo_gemm_bf16_nt_m16_kernel<MT, 0>), grid, block, smem, st, A, Wp, Cp, K, N, WV, SK); \
         else    hipLaunchKernelGGL((halo_gemm_bf16_nt_m16_kernel<MT, 1>), grid, block, smem, st, A, Wp, Cp, K, N, WV, SK); } while (0)
    switch (M) {
      case 1: LAUNCH(1); break;   case 2: LAUNCH(2); break;   case 3: LAUNCH(3); break;
      case 4: LAUNCH(4); break;   case 5: LAUNCH(5); break;   case 6: LAUNCH(6); break;
      case 7: LAUNCH(7); break;   case 8: LAUNCH(8); break;   case 9: LAUNCH(9); break;
      case 10: LAUNCH(10); break; case 11: LAUNCH(11); break; case 12: LAUNCH(12); break;
      case 13: LAUNCH(13); break; case 14: LAUNCH(14); break; case 15: LAUNCH(15); break;
      case 16: LAUNCH(16); break; default: return -1;
    }
#undef LAUNCH
    return hipGetLastError() == hipSuccess ? 0 : -5;
}

int halo_gemm_bf16_nt_m16(long a, long w, long c, int M, int K, int N, int WV, int SK, int nt,
                          long stream) {
    return gemm_bf16_m16_launch(a, w, c, M, K, N, WV, SK, nt, stream);
}

int halo_gemm_bf16_nt_m16_max_m(void) { return HALO_GEMM16_MAX_M; }

// ============================================================== the plugin ABI entry point
//
// The operand reader is halo_gemm_args.h. The split law is this kernel's own: cfg_for's unit
// law would insist on K % (SK*256) == 0 and can pick SK=1 at large N, which this kernel's
// eight-tile register file refuses at large K. A tuned (wv, sk, nt) is honoured when legal;
// a shape no legal split serves goes to the WMMA row's shim, which owns the rest of the band.
extern "C" int halo_rad_gemm_bf16_m64(const RadArgs*, RadStream);

static bool m16_legal(int K, int WV, int SK) {
    return SK >= 1 && WV >= 1 && WV * SK * WAVE <= 1024 && K % SK == 0 &&
           ((K / SK) % VEC) == 0 && (K / SK) / (WAVE * VEC) <= 8;
}

extern "C" int halo_rad_gemm_bf16_m16(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 0);
    if (rc != RAD_OK) return rc;
    if (g.M < 1 || g.M > HALO_GEMM16_MAX_M) return HALO_NO(RAD_E_SHAPE);
    HaloGemmCfg cv;
    halo_cfg_gemm(a, &cv);
    const int wv = cv.wv > 0 ? cv.wv : 1;
    int sk = cv.sk;
    if (sk <= 0 || !m16_legal((int)g.K, wv, sk)) {
        /* The smallest legal split that puts one full card of waves (1280) in flight; the
         * largest legal split when even that is not reachable. */
        static const int ladder[] = { 1, 2, 4, 5, 8, 10, 16, 20, 32 };
        sk = 0;
        for (int cand : ladder) {
            if (!m16_legal((int)g.K, wv, cand)) continue;
            sk = cand;
            if (g.N * cand >= 1280) break;
        }
        if (!sk) return halo_rad_gemm_bf16_m64(a, s);
    }
    rc = gemm_bf16_m16_launch(halo_addr(g.a), halo_addr(g.b), halo_addr(g.y),
                              (int)g.M, (int)g.K, (int)g.N, wv, sk, cv.nt,
                              (long)(uintptr_t)s);
    if (rc != 0) return halo_rad_gemm_bf16_m64(a, s);
    return RAD_OK;
}
