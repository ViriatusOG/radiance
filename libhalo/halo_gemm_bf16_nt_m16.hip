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
#include "halo_chain.h"
#include "halo.h"
#include "halo_gemm_args.h"
#include "halo_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#define WAVE 32
#define VEC  8                       // bf16 per 128-bit (dwordx4) load
#define HALO_GEMM16_MAX_M 16

typedef int    int4v  __attribute__((ext_vector_type(4)));
typedef __bf16 bf162  __attribute__((ext_vector_type(2)));   // v_dot2_f32_bf16 operand

template <int MT, int TEMPORAL>
__device__ __forceinline__ void halo_gemm_bf16_nt_m16_kernel_body(
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

/* halo_gemm_bf16_nt_m16_kernel's chained form (halo_chain.h): the body above, bracketed so every block waits once and adds once. */
template <int MT, int TEMPORAL, class CH = HaloUnchained>
__global__ __launch_bounds__(1024)
void halo_gemm_bf16_nt_m16_kernel(
        const __hip_bfloat16* __restrict__ A, const __hip_bfloat16* __restrict__ W,
        __hip_bfloat16* __restrict__ C, int K, int N, int WV, int SK) {
    if (CH::on) halo_chain_wait();
    halo_gemm_bf16_nt_m16_kernel_body<MT, TEMPORAL>(A, W, C, K, N, WV, SK);
    if (CH::on) halo_chain_done();
}

// ============================================================== the streaming form
//
// The kernel above gives each column of a split its own block of SK waves (32 to 64 threads at
// decode shapes) and lets it go once its slice is summed: at decode heights such blocks are mostly
// launch and reduction, and no configuration passes ~120 GB/s on gfx1151, where a plain streaming
// read reaches ~220. Here a block holds G column groups of SK waves (eight waves where SK divides
// eight); a wave owns CPW consecutive columns and a K slice of at most KT 256-wide tiles, and
// issues every weight load of the slice before its first dot. Activation rows come from global
// memory (a few kilobytes, cache-resident); rows past M re-read row M - 1 and are never stored, so
// the row loop is fully unrolled over MT with no branch (a runtime bound there put the
// accumulators in scratch).
//
// Up to 16 rows it is the kernel above, bit for bit: the same split law (the caller's SK), each
// lane's eight K a tile in the same order with v_dot2_f32_bf16, a lane past its slice adding
// nothing, the same xor butterfly, the splits summed in order from 0.f and the language's
// narrowing. halo_router_gemm_scatter.hip reproduces that reduction for its fused router and
// promises the launches' bytes, so this form keeps it; only the block shape and the load schedule
// change. 17..32 rows (Breeze-TTS-2's depth decoder issues 32 through this row) are column pairs
// over at most two splits, a reduction of its own.
//
// Measured on the 8060S against the kernel above, each at the row's split law: M = 2, N 12288
// K 2048 447 -> 221 us (228 GB/s), N 1024 K 8192 219 -> 85 us, N 2048 K 2048 102 -> 45 us; M = 16,
// N 16384 K 1024 280 -> 161 us; M = 32, N 16384 K 1024, 462 us on the WMMA row -> 275 us.
template <int MT, int CPW, int KT>
__device__ __forceinline__ void halo_gemv_bf16_body(
        const uint16_t* __restrict__ A, const uint16_t* __restrict__ W,
        __hip_bfloat16* __restrict__ C, int M, int K, int N, int SK, int G) {
    __shared__ float red[512];                          /* G * SK * MT * CPW <= 512 */
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int g = wave / SK, s = wave % SK;
    const int j0 = (blockIdx.x * G + g) * CPW;
    const int Ks = K / SK, kb = s * Ks;
    float acc[MT][CPW];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int c = 0; c < CPW; ++c) acc[i][c] = 0.f;
    int4v wb[KT][CPW];
#pragma unroll
    for (int t = 0; t < KT; ++t)
#pragma unroll
        for (int c = 0; c < CPW; ++c) {
            const int jc = j0 + c < N ? j0 + c : N - 1;
            const int kk = t * WAVE * VEC + lane * VEC;
            wb[t][c] = kk < Ks
                ? __builtin_nontemporal_load((const int4v*)(W + (size_t)jc * K + kb + kk))
                : int4v{ 0, 0, 0, 0 };
        }
#pragma unroll
    for (int t = 0; t < KT; ++t) {
        const int kk = t * WAVE * VEC + lane * VEC;
        if (kk < Ks) {                                  /* a lane past its slice adds nothing */
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int ir = i < M ? i : M - 1;
                const int4v av = *(const int4v*)(A + (size_t)ir * K + kb + kk);
                const bf162* a2 = (const bf162*)&av;
#pragma unroll
                for (int c = 0; c < CPW; ++c) {
                    const bf162* w2 = (const bf162*)&wb[t][c];
#pragma unroll
                    for (int p = 0; p < VEC / 2; ++p)
                        acc[i][c] = __builtin_amdgcn_fdot2_f32_bf16(a2[p], w2[p], acc[i][c], false);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int c = 0; c < CPW; ++c) {
            float v = acc[i][c];
            for (int off = WAVE / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
            acc[i][c] = v;
        }
    if (SK == 1) {                                      /* uniform: no wave waits below */
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int c = 0; c < CPW; ++c)
                if (lane == ((i * CPW + c) & (WAVE - 1)) && i < M && j0 + c < N) {
                    float v = 0.f;                      /* the split sum's fold, as above */
                    v += acc[i][c];
                    C[(size_t)i * N + j0 + c] = (__hip_bfloat16)v;
                }
        return;
    }
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int c = 0; c < CPW; ++c) red[(wave * MT + i) * CPW + c] = acc[i][c];
    }
    __syncthreads();
    if (s == 0)
        for (int e = lane; e < MT * CPW; e += WAVE) {
            const int i = e / CPW, c = e % CPW;
            float v = 0.f;
            for (int q = 0; q < SK; ++q) v += red[((g * SK + q) * MT + i) * CPW + c];
            if (i < M && j0 + c < N) C[(size_t)i * N + j0 + c] = (__hip_bfloat16)v;
        }
}

template <int MT, int CPW, int KT, int NT, class CH = HaloUnchained>
__global__ __launch_bounds__(NT)
void halo_gemv_bf16_kernel(const uint16_t* __restrict__ A, const uint16_t* __restrict__ W,
                           __hip_bfloat16* __restrict__ C, int M, int K, int N, int SK, int G) {
    if (CH::on) halo_chain_wait();
    halo_gemv_bf16_body<MT, CPW, KT>(A, W, C, M, K, N, SK, G);
    if (CH::on) halo_chain_done();
}

/* The streaming form. Up to 16 rows `sk` is the kernel above's split (its law, the caller's);
 * 17..32 rows choose their own (sk 0): at most 1024 K a split, two splits at most. A block is
 * eight waves where the split divides eight, else one column group of up to 32 waves. -1: a shape
 * it does not take (a slice past eight tiles), which the caller serves as before. */
static int gemv_bf16_launch(long a, long w, long c, int M, int K, int N, int sk, long stream,
                            bool chain) {
    if (M < 1 || M > 32 || N < 1 || K < VEC || K % VEC) return -1;
    int cpw = 1;
    if (M <= 16) {
        if (sk < 1 || sk > 32 || K % sk || (K / sk) % VEC) return -1;
    } else {
        cpw = 2;
        sk = 1;
        while (K / sk > 4 * WAVE * VEC && sk < 2 && K % (sk * 2 * VEC) == 0) sk *= 2;
        if (K % (sk * VEC)) return -1;
    }
    const int tiles = (K / sk + WAVE * VEC - 1) / (WAVE * VEC);
    int kt = 0;
    for (int c2 : { 1, 2, 4, 8 })
        if (tiles <= c2) { kt = c2; break; }
    if (!kt || (M > 16 && kt > 4)) return -1;
    const int G = sk <= 8 && 8 % sk == 0 ? 8 / sk : 1;
    const bool big = G * sk > 8;
    const int mt = M <= 2 ? 2 : M <= 4 ? 4 : M <= 8 ? 8 : M <= 16 ? 16 : 32;
    const int per = G * cpw;
    dim3 grid((unsigned)((N + per - 1) / per)), block((unsigned)(G * sk * WAVE));
    auto A = (const uint16_t*)a; auto Wp = (const uint16_t*)w; auto Cp = (__hip_bfloat16*)c;
    auto st = (hipStream_t)stream;
#define GV1(MT, CPW, KT, NT)                                                                       \
    do { if (chain) hipLaunchKernelGGL((halo_gemv_bf16_kernel<MT, CPW, KT, NT, r4dchain>), grid,   \
                                       block, 0, st, A, Wp, Cp, M, K, N, sk, G);                   \
         else       hipLaunchKernelGGL((halo_gemv_bf16_kernel<MT, CPW, KT, NT>), grid, block, 0,   \
                                       st, A, Wp, Cp, M, K, N, sk, G); } while (0)
#define GV(MT, CPW, KT) do { if (big) GV1(MT, CPW, KT, 1024); else GV1(MT, CPW, KT, 256); } while (0)
#define GV_KT(MT)                                                                                  \
    switch (kt) { case 1: GV(MT, 1, 1); break; case 2: GV(MT, 1, 2); break;                        \
                  case 4: GV(MT, 1, 4); break; default: GV(MT, 1, 8); break; }
    switch (mt) {
      case 2: GV_KT(2); break;  case 4: GV_KT(4); break;
      case 8: GV_KT(8); break;  case 16: GV_KT(16); break;
      default:
        switch (kt) { case 1: GV1(32, 2, 1, 256); break; case 2: GV1(32, 2, 2, 256); break;
                      default: GV1(32, 2, 4, 256); break; }
    }
#undef GV_KT
#undef GV
#undef GV1
    return hipGetLastError() == hipSuccess ? 0 : -5;
}

static int gemm_bf16_m16_launch(long a, long w, long c, int M, int K, int N, int WV, int SK,
                                int nt, long stream, bool chain = false) {
    if (M < 1 || M > HALO_GEMM16_MAX_M) return -1;
    if (K % SK != 0 || (K / SK) % VEC != 0) return -2;
    if (WV * SK * WAVE > 1024) return -3;
    if ((K / SK) / (WAVE * VEC) > 8) return -4;          // wbuf holds 8 tiles
    dim3 grid((N + WV - 1) / WV), block(WV * SK * WAVE);
    size_t smem = (size_t)WV * SK * M * sizeof(float);
    auto A = (const __hip_bfloat16*)a; auto Wp = (const __hip_bfloat16*)w;
    auto Cp = (__hip_bfloat16*)c; auto st = (hipStream_t)stream;
#define LAUNCH(MT)                                                                    \
    do { if (chain) {                                                                              \
             if (nt) hipLaunchKernelGGL((halo_gemm_bf16_nt_m16_kernel<MT, 0, r4dchain>), grid, block, smem, st, A, Wp, Cp, K, N, WV, SK); \
             else    hipLaunchKernelGGL((halo_gemm_bf16_nt_m16_kernel<MT, 1, r4dchain>), grid, block, smem, st, A, Wp, Cp, K, N, WV, SK); \
         } else {                                                                                   \
             if (nt) hipLaunchKernelGGL((halo_gemm_bf16_nt_m16_kernel<MT, 0>), grid, block, smem, st, A, Wp, Cp, K, N, WV, SK); \
             else    hipLaunchKernelGGL((halo_gemm_bf16_nt_m16_kernel<MT, 1>), grid, block, smem, st, A, Wp, Cp, K, N, WV, SK); \
         } } while (0)
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
    if (g.M < 1) return HALO_NO(RAD_E_SHAPE);
    const bool chain = halo_geti_or(a, "chain", 0) != 0;
    /* More rows than the kernel below takes: the streaming form's column pairs where it takes the
     * shape (Breeze-TTS-2's depth decoder issues 32 through a row its step's M placed in this
     * band), else the WMMA row, as libr4d's m16 row hands every M on. */
    if (g.M > HALO_GEMM16_MAX_M) {
        if (gemv_bf16_launch(halo_addr(g.a), halo_addr(g.b), halo_addr(g.y), (int)g.M, (int)g.K,
                             (int)g.N, 0, (long)(uintptr_t)s, chain) == 0)
            return RAD_OK;
        return halo_rad_gemm_bf16_m64(a, s);
    }
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
        /* Then more splits while a wave's slice passes four 256-wide tiles: the streaming form's
         * eight-wave blocks keep too little in flight past that (N 1024 K 8192: four splits 124 us,
         * eight 84). halo_router_gemm_scatter.hip's rg_m16_sk is this law and moves with it. */
        for (int cand : ladder)
            if (cand > sk && (int)g.K / sk > 4 * WAVE * VEC && m16_legal((int)g.K, wv, cand))
                sk = cand;
    }
    /* The streaming form at this split, bit for bit this kernel's (above); the kernel itself for a
     * slice past eight tiles. */
    if (gemv_bf16_launch(halo_addr(g.a), halo_addr(g.b), halo_addr(g.y), (int)g.M, (int)g.K,
                         (int)g.N, sk, (long)(uintptr_t)s, chain) == 0)
        return RAD_OK;
    rc = gemm_bf16_m16_launch(halo_addr(g.a), halo_addr(g.b), halo_addr(g.y),
                              (int)g.M, (int)g.K, (int)g.N, wv, sk, cv.nt,
                              (long)(uintptr_t)s, chain);
    if (rc != 0) return halo_rad_gemm_bf16_m64(a, s);
    return RAD_OK;
}
