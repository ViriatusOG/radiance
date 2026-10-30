/* Derived from libr4d/r4d_hc_bf16.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_hc_read.hip -- hc_read_bf16 / hc_read_e4m3: the gated residual's read.
//
//     N   = grouped_rmsnorm(h)        per sub-stream: bf16(h * rsqrt(ss/n + eps) * (gain + wadd))
//     d   = silu(W_down @ N / hc)     [lowrank], narrowed to bf16
//     x   = mean_c( sigmoid(W_up @ d)[c] * N[c] )
//     inj = 2 * sigmoid(W_inj @ N / hc)
//
// plus the fp8 (or int8) twin of x at one scale per (row, 128) and, with `rotate`, the
// rotated E4M3 twin (unnormalised FWHT over the group, the scale absorbing the norm, as
// libr4d's hc_rot_group quantises it).
//
// libr4d splits this over per-row kernels, decode tiles and two prefill forms with arrival
// counters, all bit-identical by construction. This port runs one path at every M -- a norm
// kernel, two tiled WMMA products (the second over the transposed product into a scratch
// `u`), and a mix kernel that owns one 128-channel group a row so the twins close in-block.
// Every pinned form therefore runs the same launches and the bytes agree trivially; the
// pin hook is accepted and ignored. The E4M3 mixing matrices are decoded to bf16 scratch
// planes first (their stored rows carry f32 group scales in the tail, halo_hc_fp8.h).
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_gdn_wmma.h"
#include "halo_fp8_wmma.h"
#include "halo_hc_fp8.h"
#include "halo_hc_f16.h"
#include "halo_wmma.h"
#include <hip/hip_bf16.h>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>

namespace {

__device__ __forceinline__ float hc_sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }

/* ---- the grouped norm: one block a (row, sub-stream) ---------------------------------- */
__global__ __launch_bounds__(256) void hc_norm_kernel(const uint16_t* __restrict__ h,
                                                      const float* __restrict__ w,
                                                      uint16_t* __restrict__ N,
                                                      long n, long HN, float eps, float wadd) {
    __shared__ float red[8];
    const long m = blockIdx.y, c = blockIdx.x;
    const uint16_t* hr = h + m * HN + c * n;
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (long i = tid; i < n; i += 256) {
        const float v = bf16_to_f32(hr[i]);
        ss += v * v;
    }
    const int lane = tid & 31, wave = tid >> 5;
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) ss += __shfl_xor(ss, mask);
    if (lane == 0) red[wave] = ss;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) t += red[i];
        red[0] = 1.0f / sqrtf(t / (float)n + eps);
    }
    __syncthreads();
    const float sc = red[0];
    uint16_t* Nr = N + m * HN + c * n;
    for (long i = tid; i < n; i += 256)
        Nr[i] = f32_to_bf16(bf16_to_f32(hr[i]) * sc * (w[c * n + i] + wadd));
}

/* ---- decode one stored E4M3 plane to bf16: out[r][k] = e4m3(code) * scale[k / G] -------- */
__global__ __launch_bounds__(256) void hc_decode_f8_kernel(const uint8_t* __restrict__ src,
                                                           uint16_t* __restrict__ dst,
                                                           long rows, long K, long G, long ld) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * (K / 16)) return;
    const long r = idx / (K / 16), chunk = idx % (K / 16);
    const uint8_t* rowp = src + r * ld;
    const float* scales = (const float*)(rowp + K);
    const uint4 raw = *(const uint4*)(rowp + chunk * 16);
    const uint32_t d[4] = { raw.x, raw.y, raw.z, raw.w };
    uint16_t* out = dst + r * K + chunk * 16;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        const long k = chunk * 16 + e;
        const unsigned code = (d[e >> 2] >> (8 * (e & 3))) & 0xffu;
        const float v = halo_fp8::e4m3_pk<false>((uint32_t)code).x;
        out[e] = f32_to_bf16(v * scales[k / G]);
    }
}

/* ---- the inject gains: one wave a (row, stream) ---------------------------------------- */
__global__ __launch_bounds__(128) void hc_inj_kernel(const uint16_t* __restrict__ N,
                                                     const uint16_t* __restrict__ injw,
                                                     uint16_t* __restrict__ inj,
                                                     long HN, int hc, long irow) {
    const long m = blockIdx.y;
    const int c = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const uint16_t* wr = injw + (long)c * HN;
    const uint16_t* Nr = N + m * HN;
    float acc = 0.0f;
    for (long i = lane; i < HN; i += 32)
        acc += bf16_to_f32(Nr[i]) * bf16_to_f32(wr[i]);
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) acc += __shfl_xor(acc, mask);
    if (lane == 0)
        inj[m * irow + c] = f32_to_bf16(2.0f * hc_sigmoid(acc * (1.0f / (float)hc)));
}

/* ---- the mix and the twins: one block a (row, 128-channel group) ------------------------ */
__global__ __launch_bounds__(128) void hc_mix_kernel(const uint16_t* __restrict__ N,
                                                     const uint16_t* __restrict__ u,
                                                     uint16_t* __restrict__ x,
                                                     uint8_t* __restrict__ q, float* __restrict__ qs,
                                                     uint8_t* __restrict__ rq, float* __restrict__ rs,
                                                     long n, long HN, int hc, long xrow, long qrow,
                                                     long srow, int i8) {
    __shared__ float lds[128];
    const long m = blockIdx.y;
    const long base = (long)blockIdx.x * 128;
    const int j = threadIdx.x;
    float acc = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const float uv = bf16_to_f32(u[(m * HN) + c * n + base + j]);
        const float sig = bf16_to_f32(f32_to_bf16(hc_sigmoid(uv)));
        acc += sig * bf16_to_f32(N[(m * HN) + c * n + base + j]);
    }
    const float xv = acc * (1.0f / (float)hc);
    const uint16_t xb = f32_to_bf16(xv);
    x[m * xrow + base + j] = xb;
    const float xf = bf16_to_f32(xb);
    lds[j] = fabsf(xf);
    __syncthreads();
    /* amax over the group */
    for (int stride = 64; stride; stride >>= 1) {
        if (j < stride) lds[j] = fmaxf(lds[j], lds[j + stride]);
        __syncthreads();
    }
    const float amax = lds[0];
    if (q) {
        if (i8) {
            const float sc = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
            if (j == 0) qs[m * srow + blockIdx.x] = sc;
            const float inv = 1.0f / sc;
            q[m * qrow + base + j] =
                (uint8_t)(int8_t)(int)rintf(fminf(fmaxf(xf * inv, -127.0f), 127.0f));
        } else {
            const float sc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
            if (j == 0) qs[m * srow + blockIdx.x] = sc;
            const float inv = 1.0f / sc;
            q[m * qrow + base + j] = (uint8_t)halo_fp8::f32_to_e4m3(
                fminf(fmaxf(xf * inv, -448.0f), 448.0f));
        }
    }
    if (rq) {
        /* The FWHT over the group in LDS: H[i][j] = (-1)^popcount(i & j), unnormalised. */
        lds[j] = xf;
        __syncthreads();
#pragma unroll
        for (int hstep = 1; hstep < 128; hstep <<= 1) {
            if ((j & hstep) == 0) {
                const float a = lds[j], b = lds[j + hstep];
                lds[j] = a + b;
                lds[j + hstep] = a - b;
            }
            __syncthreads();
        }
        const float rv = lds[j];
        float am2 = fabsf(rv);
        lds[j] = am2;
        __syncthreads();
        for (int stride = 64; stride; stride >>= 1) {
            if (j < stride) lds[j] = fmaxf(lds[j], lds[j + stride]);
            __syncthreads();
        }
        const float ramax = lds[0];
        const float rsc = ramax > 0.0f ? ramax * (1.0f / 448.0f) : 1.0f;
        if (j == 0) rs[m * srow + blockIdx.x] = rsc;
        const float inv = 1.0f / rsc;
        rq[m * qrow + base + j] = (uint8_t)halo_fp8::f32_to_e4m3(
            fminf(fmaxf(rv * inv, -448.0f), 448.0f));
    }
}

/* ---- the tiled product (clone of halo_gemm_nt_bias's tile, plain or silu(/hc) epilogue) - */
#define HC_BK 64
#define HC_PITCH (HC_BK + 8)
#define HC_WM 2
#define HC_WN 4

template <int TM, int TN>
__global__ __launch_bounds__(256) void hc_gemm_kernel(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    __hip_bfloat16* __restrict__ C, int M, int K, int N,
    int lda, int ldw, int ldc, int epi_silu, float inv_hc) {
    constexpr int BM = HC_WM * TM * 16, BN = HC_WN * TN * 16;
    constexpr int CPR = HC_BK / 8;
    constexpr int AL = BM * CPR / 256, WL = BN * CPR / 256;
    __shared__ __attribute__((aligned(16))) unsigned short sA[BM * HC_PITCH];
    __shared__ __attribute__((aligned(16))) unsigned short sW[BN * HC_PITCH];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave / HC_WN, wn = wave % HC_WN;
    const int m0 = (int)blockIdx.y * BM, n0 = (int)blockIdx.x * BN;

    unsigned aoff[AL], woff[WL];
    unsigned short* adst[AL];
    unsigned short* wdst[WL];
#pragma unroll
    for (int i = 0; i < AL; ++i) {
        const int c_ = tid + 256 * i, rr = c_ / CPR, e = (c_ % CPR) * 8;
        const int gr = m0 + rr < M ? m0 + rr : M - 1;
        aoff[i] = (unsigned)gr * (unsigned)lda + (unsigned)e;
        adst[i] = sA + rr * HC_PITCH + e;
    }
#pragma unroll
    for (int i = 0; i < WL; ++i) {
        const int c_ = tid + 256 * i, rr = c_ / CPR, e = (c_ % CPR) * 8;
        const int gr = n0 + rr < N ? n0 + rr : N - 1;
        woff[i] = (unsigned)gr * (unsigned)ldw + (unsigned)e;
        wdst[i] = sW + rr * HC_PITCH + e;
    }

    const unsigned short* fa = sA + (wm * TM * 16 + (lane & 15)) * HC_PITCH;
    const unsigned short* fw = sW + (wn * TN * 16 + (lane & 15)) * HC_PITCH;

    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (v8f)(0.f);

    const int nk = (K + HC_BK - 1) / HC_BK;
    auto load_tile = [&](int kt, halo_u32x4* ra, halo_u32x4* rw) {
        const unsigned k0 = (unsigned)kt * HC_BK;
#pragma unroll
        for (int i = 0; i < AL; ++i) {
            const int c_ = tid + 256 * i, e = (c_ % CPR) * 8;
            ra[i] = (k0 + (unsigned)e < (unsigned)K)
                        ? *(const halo_u32x4*)(A + aoff[i] + k0) : (halo_u32x4)0u;
        }
#pragma unroll
        for (int i = 0; i < WL; ++i) {
            const int c_ = tid + 256 * i, e = (c_ % CPR) * 8;
            rw[i] = (k0 + (unsigned)e < (unsigned)K)
                        ? *(const halo_u32x4*)(W + woff[i] + k0) : (halo_u32x4)0u;
        }
    };

    halo_u32x4 ra[AL], rw[WL];
    load_tile(0, ra, rw);

    for (int kt = 0; kt < nk; ++kt) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < AL; ++i) *(halo_u32x4*)adst[i] = ra[i];
#pragma unroll
        for (int i = 0; i < WL; ++i) *(halo_u32x4*)wdst[i] = rw[i];
        __syncthreads();
        if (kt + 1 < nk) load_tile(kt + 1, ra, rw);
#pragma unroll
        for (int s = 0; s < HC_BK / 16; ++s) {
            v16bf af[TM], bf[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) af[i] = fragP(fa + i * 16 * HC_PITCH + s * 16);
#pragma unroll
            for (int j = 0; j < TN; ++j) bf[j] = fragP(fw + j * 16 * HC_PITCH + s * 16);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = mma(af[i], bf[j], acc[i][j]);
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int n_ = n0 + (wn * TN + j) * 16 + (lane & 15);
            const int mb = m0 + (wm * TM + i) * 16;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = mb + accRow(lane, e);
                if (n_ < N && m < M) {
                    float v = acc[i][j][e];
                    if (epi_silu) {
                        v *= inv_hc;
                        v = v * hc_sigmoid(v);
                    }
                    C[(size_t)m * ldc + n_] = (__hip_bfloat16)v;
                }
            }
        }
}

int hc_gemm_launch(const unsigned short* A, const unsigned short* W, __hip_bfloat16* C,
                   int M, int K, int N, int lda, int ldw, int ldc, int epi_silu, float inv_hc,
                   hipStream_t st) {
    const long long g = (long long)((N + 127) / 128) * ((M + 127) / 128);
    const int bt = g >= 32 ? 128 : 64;
    dim3 grid((N + bt - 1) / bt, (M + bt - 1) / bt), block(256);
#define HC_LAUNCH(TM, TN) hipLaunchKernelGGL((hc_gemm_kernel<TM, TN>), grid, block, 0, st, \
    A, W, C, M, K, N, lda, ldw, ldc, epi_silu, inv_hc)
    if (bt == 128) HC_LAUNCH(4, 2); else HC_LAUNCH(2, 1);
#undef HC_LAUNCH
    return (int)hipGetLastError();
}

/* ==== the E4M3 fast path ===================================================================
 *
 * Three launches at every M: the per (row, stream) norm scales; the down product, as split-K
 * partials of W_down @ N plus the inject dots; and the up product, with the mix and both twins
 * closing in the block that owns a 128-channel group. N is never stored: both products rebuild
 * it from h exactly as hc_norm_kernel writes it, bf16(h * sc * (gain + wadd)).
 *
 * The products run on the f16 WMMA over the codes as they are stored (halo_hc_f16.h); N and d are
 * bf16 by contract and f16 holds them exactly in the range they live in (|N| is sqrt(n) times the
 * gain at most). The inject dots read bf16 weights and run on the VALU in f32 over the same
 * rounded N.
 *
 * Decode (rows <= kHcfDecRows): the down product is split kHcfDecS ways along K, a block of four
 * waves a (16-column tile, slice). Prefill: a block a (64 rows, stream), the stream its K slice.
 * Either writes partials P[s][m][0..lr+hc) that the up kernel sums in slice order, so a call is
 * deterministic; the two forms sum in different orders, which the engine allows (spec §17). */

constexpr int kHcfMaxHc   = 4;     /* the scale table's row stride */
constexpr int kHcfDecRows = 64;    /* the decode form's row bound */
constexpr int kHcfDecS    = 16;    /* its K slices */
constexpr int kHcfUpLr    = 320;   /* lowrank bound: a quarter row of W_up, and W_down's k tile, fit LDS */

__device__ __forceinline__ void hcf_codes16(uint4 r, uint32_t* d) { halo_hcf::codes16(r, d); }
__device__ __forceinline__ v16hf hcf_frag(const uint32_t* d) { return halo_hcf::frag(d); }
__device__ __forceinline__ uint32_t hcf_pk16(float a, float b) { return halo_hcf::pk16(a, b); }
/* sigmoid on v_exp and v_rcp: within a few ulp of hc_sigmoid, and every use rounds it to bf16. */
__device__ __forceinline__ float hcf_sigmoid(float v) {
    return __builtin_amdgcn_rcpf(1.0f + __expf(-v));
}
/* One element of N, rounded as hc_norm_kernel rounds it. */
__device__ __forceinline__ float hcf_n(uint32_t hbits, float s, float g) {
    return bf16_to_f32(f32_to_bf16(bf16_to_f32((uint16_t)hbits) * s * g));
}

/* ---- the norm scales: one block a (row, stream), sc[m][c] = rsqrt(ss/n + eps) ------------ */
__global__ __launch_bounds__(256) void hcf_stats_kernel(const uint16_t* __restrict__ h, long hrow,
                                                        float* __restrict__ sc, int n, float eps) {
    __shared__ float red[8];
    const int m = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
    const uint4* hr = (const uint4*)(h + (long)m * hrow + (long)c * n);
    float ss = 0.0f;
    for (int i = tid; i < n / 8; i += 256) {
        const uint4 v = hr[i];
        const uint32_t d[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float a = bf16_to_f32((uint16_t)d[e]), b = bf16_to_f32((uint16_t)(d[e] >> 16));
            ss += a * a;
            ss += b * b;
        }
    }
    const int lane = tid & 31, wave = tid >> 5;
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) ss += __shfl_xor(ss, mask);
    if (lane == 0) red[wave] = ss;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) t += red[i];
        sc[m * kHcfMaxHc + c] = 1.0f / sqrtf(t / (float)n + eps);
    }
}

/* ---- the down product, decode form: a block a (16-column tile, K slice) -------------------
 * The slice is walked a scale group (128 k) at a time, one k step a wave. Every wave issues the
 * codes and scales of all its steps first, so the tile's weight bytes are in flight at once; the
 * block stages each group's N for its rows in LDS (double-buffered, one barrier a group). The
 * eight waves' sums close in LDS in wave order. Column tile 0 also carries the inject dots: the
 * sixteen threads staging a row hold its terms and close them with shuffles. */
#ifndef HCF_EXP_NOH
#define HCF_EXP_NOH 0
#endif
constexpr int kHcfSub = 128;       /* k a staged group: one scale group, a k step a wave */
constexpr int kHcfSubMax = 8;      /* groups a slice at most */

template <int RT>
__global__ __launch_bounds__(256) void hcf_down_dec_kernel(
    const uint16_t* __restrict__ h, long hrow, const float* __restrict__ gain, float wadd,
    const float* __restrict__ sc, const uint8_t* __restrict__ down, long dld,
    const uint16_t* __restrict__ injw, float* __restrict__ P, long pslice, int pw,
    int rows, int n, int hc, int HN, int lr) {
    constexpr int NP = kHcfSub + 8, NB = RT * 16 * NP;
    constexpr int LB = (2 * NB * 2 > 8 * RT * 8 * 32 * 4) ? 2 * NB * 2 : 8 * RT * 8 * 32 * 4;
    __shared__ __attribute__((aligned(16))) unsigned char lds[LB];
    uint16_t* Nb = (uint16_t*)lds;
    const int ct = blockIdx.x, s = blockIdx.y;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, col = lane & 15;
    const int sps = HN / kHcfSub / (int)gridDim.y, sub0 = s * sps;
    const uint8_t* wrow = down + (long)(ct * 16 + col) * dld;
    const float* wsc = (const float*)(wrow + HN);

    uint4 cw[kHcfSubMax];
    float cs[kHcfSubMax];
#pragma unroll
    for (int i = 0; i < kHcfSubMax; ++i)
        if (i < sps) {
            cw[i] = *(const uint4*)(wrow + (long)(sub0 + i) * kHcfSub + wave * 16);
            cs[i] = wsc[sub0 + i] * 256.0f;
        }

    const int sr = tid >> 4, sk = (tid & 15) * 8;
    const bool doinj = injw != nullptr && ct == 0;
    float ip[RT][kHcfMaxHc];
    v8f acc[RT];
#pragma unroll
    for (int p = 0; p < RT; ++p) {
        acc[p] = (v8f)(0.0f);
#pragma unroll
        for (int cc = 0; cc < kHcfMaxHc; ++cc) ip[p][cc] = 0.0f;
    }

#pragma unroll
    for (int i = 0; i < kHcfSubMax; ++i) {
        if (i >= sps) break;
        uint16_t* nb = Nb + (i & 1) * NB;
        const long k0 = (long)(sub0 + i) * kHcfSub + sk;
        const int c = (int)(k0 / n);
        const float4 g0 = *(const float4*)(gain + k0), g1 = *(const float4*)(gain + k0 + 4);
        const float gg[8] = { g0.x + wadd, g0.y + wadd, g0.z + wadd, g0.w + wadd,
                              g1.x + wadd, g1.y + wadd, g1.z + wadd, g1.w + wadd };
#pragma unroll
        for (int p = 0; p < RT; ++p) {
            const int m = sr + 16 * p;
            uint32_t ad[4] = { 0u, 0u, 0u, 0u };
            if (HCF_EXP_NOH == 0 && m < rows) {
                const float sm = sc[m * kHcfMaxHc + c];
                const uint4 hv = *(const uint4*)(h + (long)m * hrow + k0);
                const uint32_t hw[4] = { hv.x, hv.y, hv.z, hv.w };
                float nv[8];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    nv[2 * e]     = hcf_n(hw[e] & 0xffffu, sm, gg[2 * e]);
                    nv[2 * e + 1] = hcf_n(hw[e] >> 16, sm, gg[2 * e + 1]);
                    ad[e] = hcf_pk16(nv[2 * e], nv[2 * e + 1]);
                }
                if (doinj)
                    for (int cc = 0; cc < hc; ++cc) {
                        const uint4 iv = *(const uint4*)(injw + (long)cc * HN + k0);
                        const uint32_t iw[4] = { iv.x, iv.y, iv.z, iv.w };
                        float t = ip[p][cc];
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            t += nv[2 * e] * bf16_to_f32((uint16_t)iw[e]);
                            t += nv[2 * e + 1] * bf16_to_f32((uint16_t)(iw[e] >> 16));
                        }
                        ip[p][cc] = t;
                    }
            }
            *(uint4*)(nb + m * NP + sk) = make_uint4(ad[0], ad[1], ad[2], ad[3]);
        }
        __syncthreads();
        const float csi = cs[i];
        uint32_t bd[8];
        hcf_codes16(cw[i], bd);
        const v16hf b = hcf_frag(bd);
#pragma unroll
        for (int p = 0; p < RT; ++p) {
            const uint16_t* ap = nb + (p * 16 + col) * NP + wave * 16;
            union { uint4 w[2]; v16hf f; } a;
            a.w[0] = ((const uint4*)ap)[0];
            a.w[1] = ((const uint4*)ap)[1];
            const v8f t = halo::mma(a.f, b, (v8f)(0.0f));
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[p][e] = fmaf(csi, t[e], acc[p][e]);
        }
    }

    float* Ps = P + (long)s * pslice;
    if (doinj)
#pragma unroll
        for (int p = 0; p < RT; ++p)
            for (int cc = 0; cc < hc; ++cc) {
                float v = ip[p][cc];
                v += __shfl_xor(v, 8);
                v += __shfl_xor(v, 4);
                v += __shfl_xor(v, 2);
                v += __shfl_xor(v, 1);
                const int m = sr + 16 * p;
                if ((tid & 15) == 0 && m < rows) Ps[(long)m * pw + lr + cc] = v;
            }
    __syncthreads();
    float* red = (float*)lds;                      /* [8][RT][8][32], over the N buffers */
#pragma unroll
    for (int p = 0; p < RT; ++p)
#pragma unroll
        for (int e = 0; e < 8; ++e) red[((wave * RT + p) * 8 + e) * 32 + lane] = acc[p][e];
    __syncthreads();
    /* Wave w closes row tile w % RT's elements e with e % (8 / RT')... simply: each of the
     * RT*8*32 outputs is summed over the eight waves in order by one thread. */
    for (int o = tid; o < RT * 8 * 32; o += 256) {
        const int p = o / 256, e = (o / 32) & 7, ln = o & 31;
        float v = red[((0 * RT + p) * 8 + e) * 32 + ln];
#pragma unroll
        for (int w = 1; w < 8; ++w) v += red[((w * RT + p) * 8 + e) * 32 + ln];
        const int m = p * 16 + accRow(ln, e);
        if (m < rows) Ps[(long)m * pw + ct * 16 + (ln & 15)] = v;
    }
}

/* ---- decode at up to sixteen rows: N once, then a down product with no staging ------------
 * hcf_normdec_kernel is hcf_stats_kernel (the same sums in the same order, so the same scale)
 * that goes on to write the row's N for its stream, rounded as hcf_n rounds it and held as f16
 * (exact: bf16 values in f16's range), and the stream's inject dots over that N, closed over the
 * block in wave order. hcf_down_dec1_kernel is hcf_down_dec_kernel at one row tile with its A
 * fragments read straight from that N, every group's issued up front with the codes: the twenty
 * column-tile blocks of a slice no longer each normalise the slice behind a barrier a group, which
 * was a third of the kernel. The WMMA chain, the fold and the wave-order close are
 * hcf_down_dec_kernel's, so the partials are the same bits; the inject terms are summed in a
 * different order (per stream, then over the streams, in hcf_normdec_kernel). SPS is the slice's
 * groups (five at Flash-Next's 10240 / 128 over sixteen slices), fixed so the preloads take no
 * registers past it. */
/* A row a block, its HC streams side by side: stream c is threads [256c, 256c + 256), each running
 * hcf_stats_kernel's thread over the stream (the same chunks, sums and wave order). Every load --
 * the row's chunks, their gains and inject weights -- is issued before the first barrier. The
 * inject dots are closed here, over the streams in order, and written as the gate. */
template <int HC>
__global__ __launch_bounds__(HC * 256) void hcf_normdec_kernel(
    const uint16_t* __restrict__ h, long hrow, const float* __restrict__ gain, float wadd,
    const uint16_t* __restrict__ injw, float* __restrict__ sc, uint16_t* __restrict__ Nf,
    uint16_t* __restrict__ inj, long irow, int n, float eps) {
    constexpr int HN_CH = 2;                       /* chunks a thread: n / 8 <= 512 */
    __shared__ float red[HC][8];
    __shared__ float ired[HC][HC][8];
    const int m = blockIdx.x, tid = threadIdx.x, c = tid >> 8, v = tid & 255;
    const int lane = tid & 31, wv = v >> 5;
    const int HN = HC * n, nch = n / 8;
    const uint16_t* hr = h + (long)m * hrow + (long)c * n;
    uint4 hv[HN_CH];
    float4 g0[HN_CH], g1[HN_CH];
    uint4 iv[HN_CH][HC];
#pragma unroll
    for (int j = 0; j < HN_CH; ++j) {
        const int i = v + 256 * j;
        const int ic = i < nch ? i : 0;
        const long k0 = (long)c * n + ic * 8;
        hv[j] = *(const uint4*)(hr + ic * 8);
        g0[j] = *(const float4*)(gain + k0);
        g1[j] = *(const float4*)(gain + k0 + 4);
        if (injw)
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) iv[j][cc] = *(const uint4*)(injw + (long)cc * HN + k0);
    }
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < HN_CH; ++j) {
        if (v + 256 * j >= nch) break;
        const uint32_t d[4] = { hv[j].x, hv[j].y, hv[j].z, hv[j].w };
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float a = bf16_to_f32((uint16_t)d[e]), b = bf16_to_f32((uint16_t)(d[e] >> 16));
            ss += a * a;
            ss += b * b;
        }
    }
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) ss += __shfl_xor(ss, mask);
    if (lane == 0) red[c][wv] = ss;
    __syncthreads();
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) t += red[c][i];
    const float sm = 1.0f / sqrtf(t / (float)n + eps);
    if (v == 0) sc[m * kHcfMaxHc + c] = sm;
    float ip[HC];
#pragma unroll
    for (int cc = 0; cc < HC; ++cc) ip[cc] = 0.0f;
#pragma unroll
    for (int j = 0; j < HN_CH; ++j) {
        const int i = v + 256 * j;
        if (i >= nch) break;
        const long k0 = (long)c * n + i * 8;
        const uint32_t hw[4] = { hv[j].x, hv[j].y, hv[j].z, hv[j].w };
        const float gg[8] = { g0[j].x + wadd, g0[j].y + wadd, g0[j].z + wadd, g0[j].w + wadd,
                              g1[j].x + wadd, g1[j].y + wadd, g1[j].z + wadd, g1[j].w + wadd };
        float nv[8];
        uint32_t ad[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            nv[2 * e]     = hcf_n(hw[e] & 0xffffu, sm, gg[2 * e]);
            nv[2 * e + 1] = hcf_n(hw[e] >> 16, sm, gg[2 * e + 1]);
            ad[e] = hcf_pk16(nv[2 * e], nv[2 * e + 1]);
        }
        *(uint4*)(Nf + (long)m * HN + k0) = make_uint4(ad[0], ad[1], ad[2], ad[3]);
        if (injw)
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                const uint32_t iw[4] = { iv[j][cc].x, iv[j][cc].y, iv[j][cc].z, iv[j][cc].w };
                float u = ip[cc];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    u += nv[2 * e] * bf16_to_f32((uint16_t)iw[e]);
                    u += nv[2 * e + 1] * bf16_to_f32((uint16_t)(iw[e] >> 16));
                }
                ip[cc] = u;
            }
    }
    if (!inj) return;
#pragma unroll
    for (int cc = 0; cc < HC; ++cc) {
        float u = ip[cc];
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) u += __shfl_xor(u, mask);
        if (lane == 0) ired[cc][c][wv] = u;
    }
    __syncthreads();
    if (tid < HC) {
        float u = 0.0f;
        for (int cs = 0; cs < HC; ++cs)
#pragma unroll
            for (int i = 0; i < 8; ++i) u += ired[tid][cs][i];
        inj[(long)m * irow + tid] = f32_to_bf16(2.0f * hcf_sigmoid(u * (1.0f / (float)HC)));
    }
}

template <int SPS>
__global__ __launch_bounds__(256) void hcf_down_dec1_kernel(
    const uint16_t* __restrict__ Nf, const uint8_t* __restrict__ down, long dld,
    float* __restrict__ P, long pslice, int pw, int rows, int HN) {
    __shared__ __attribute__((aligned(16))) float red[8 * 8 * 32];
    const int ct = blockIdx.x, s = blockIdx.y;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, col = lane & 15;
    const int sub0 = s * SPS;
    const uint8_t* wrow = down + (long)(ct * 16 + col) * dld;
    const float* wsc = (const float*)(wrow + HN);
    const bool live = col < rows;
    const uint16_t* np = Nf + (long)(live ? col : 0) * HN + wave * 16;

    uint4 cw[SPS], a0[SPS], a1[SPS];
    float cs[SPS];
#pragma unroll
    for (int i = 0; i < SPS; ++i) {
        const long k = (long)(sub0 + i) * kHcfSub;
        cw[i] = *(const uint4*)(wrow + k + wave * 16);
        cs[i] = wsc[sub0 + i] * 256.0f;
        a0[i] = *(const uint4*)(np + k);
        a1[i] = *(const uint4*)(np + k + 8);
    }
    v8f acc = (v8f)(0.0f);
#pragma unroll
    for (int i = 0; i < SPS; ++i) {
        uint32_t bd[8];
        hcf_codes16(cw[i], bd);
        const v16hf b = hcf_frag(bd);
        union { uint4 w[2]; v16hf f; } a;
        a.w[0] = live ? a0[i] : make_uint4(0u, 0u, 0u, 0u);
        a.w[1] = live ? a1[i] : make_uint4(0u, 0u, 0u, 0u);
        const v8f t = halo::mma(a.f, b, (v8f)(0.0f));
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = fmaf(cs[i], t[e], acc[e]);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) red[(wave * 8 + e) * 32 + lane] = acc[e];
    __syncthreads();
    float* Ps = P + (long)s * pslice;
    {
        const int e = (tid / 32) & 7, ln = tid & 31;
        float v = red[(0 * 8 + e) * 32 + ln];
#pragma unroll
        for (int w = 1; w < 8; ++w) v += red[(w * 8 + e) * 32 + ln];
        const int m = accRow(ln, e);
        if (m < rows) Ps[(long)m * pw + ct * 16 + (ln & 15)] = v;
    }
}

/* ---- the down product, prefill form: a block a (64 rows, stream), the stream its K slice ---
 * A 64-wide k tile at a time, half a scale group: a thread stages N for one row (sixteen k, as
 * f16) and its inject terms, and decodes five sixteen-code chunks of the tile's codes to f16 with
 * their column's scale (times 2^8) folded in, as W_up's plane holds its weights; the next tile is
 * loaded while this one multiplies. Wave w owns row tiles 2(w%2) and 2(w%2)+1 and five column
 * tiles from (w/2)*5. Decoding a code once at staging, rather than in both waves that build a
 * fragment from it, halves the kernel's VALU work and outweighs the occupancy the larger tile
 * costs (two blocks a WGP instead of three): 300 against 350 us a 1024-row slice. Folding the
 * norm scales in as well (each wave first reducing eight of the block's rows, in
 * hcf_stats_kernel's order) measured slower: nothing else runs while those DRAM reads land. A
 * wave-a-row form of hcf_stats_kernel itself measured no faster (both run near DRAM speed). */
template <int LR>
__global__ __launch_bounds__(256) void hcf_down_pf_kernel(
    const uint16_t* __restrict__ h, long hrow, const float* __restrict__ gain, float wadd,
    const float* __restrict__ sc, const uint8_t* __restrict__ down, long dld,
    const uint16_t* __restrict__ injw, float* __restrict__ P, long pslice, int pw,
    int rows, int n, int hc, int HN, int lr) {
    constexpr int KT = 64, AP = KT + 8, BP = KT + 8;       /* k tile; LDS pitches (halves) */
    constexpr int TPW = LR / 64;                           /* column tiles a wave */
    constexpr int NBC = (LR * (KT / 16) + 255) / 256;      /* code chunks a thread stages */
    __shared__ __attribute__((aligned(16))) uint16_t sA[64 * AP];
    __shared__ __attribute__((aligned(16))) uint16_t sB[LR * BP];
    const int c = blockIdx.y, m0 = blockIdx.x * 64;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, col = lane & 15;
    const int sw = __builtin_amdgcn_readfirstlane(wave);
    const int rp = sw & 1, cb = sw >> 1;                   /* row tiles 2rp, 2rp+1; col tiles cb*TPW.. */
    const long kc = (long)c * n;

    /* Staging: wave w takes sixteen k (w%4) of rows (w/4)*32 + lane, so its gains and inject
     * weights are wave-uniform and come through the scalar cache. */
    const int ar = (sw >> 2) * 32 + lane, ak = (sw & 3) * 16, am = m0 + ar;
    const bool arow = am < rows;
    const uint16_t* ahp = h + (long)(arow ? am : 0) * hrow + kc + ak;
    const float* agp = gain + kc + ak;
    const int last = LR * (KT / 16) - 1;

    halo_u32x4 rh0, rh1, rb[NBC];
    float rbs[NBC];
#define HCF_PFLOAD(KT_)                                                                            \
    do {                                                                                           \
        const int k0 = (KT_) * KT;                                                                 \
        rh0 = arow ? *(const halo_u32x4*)(ahp + k0) : (halo_u32x4)0u;                              \
        rh1 = arow ? *(const halo_u32x4*)(ahp + k0 + 8) : (halo_u32x4)0u;                          \
        for (int i = 0; i < NBC; ++i) {                                                            \
            const int qd = min(tid + 256 * i, last);                                               \
            const uint8_t* rowp = down + (long)(qd >> 2) * dld;                                    \
            rb[i] = *(const halo_u32x4*)(rowp + kc + k0 + (qd & 3) * 16);                          \
            rbs[i] = ((const float*)(rowp + HN))[(kc + k0) >> 7];                                  \
        }                                                                                          \
    } while (0)

    const float as = arow ? sc[am * kHcfMaxHc + c] : 0.0f;
    HCF_PFLOAD(0);

    v8f acc[2][TPW];
#pragma unroll
    for (int r2 = 0; r2 < 2; ++r2)
#pragma unroll
        for (int i = 0; i < TPW; ++i) acc[r2][i] = (v8f)(0.0f);
    float ip[kHcfMaxHc] = { 0.0f, 0.0f, 0.0f, 0.0f };

    const int nk = n / KT;
    for (int kt = 0; kt < nk; ++kt) {
        __syncthreads();
        {
            const uint32_t hw[8] = { rh0.x, rh0.y, rh0.z, rh0.w, rh1.x, rh1.y, rh1.z, rh1.w };
            float g[16];
#pragma unroll
            for (int e = 0; e < 16; ++e) g[e] = agp[kt * KT + e];
            float nv[16];
            uint32_t ad[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                nv[2 * e]     = hcf_n(hw[e] & 0xffffu, as, g[2 * e] + wadd);
                nv[2 * e + 1] = hcf_n(hw[e] >> 16, as, g[2 * e + 1] + wadd);
                ad[e] = hcf_pk16(nv[2 * e], nv[2 * e + 1]);
            }
            *(halo_u32x4*)(sA + ar * AP + ak) = (halo_u32x4){ ad[0], ad[1], ad[2], ad[3] };
            *(halo_u32x4*)(sA + ar * AP + ak + 8) = (halo_u32x4){ ad[4], ad[5], ad[6], ad[7] };
            if (injw)
#pragma unroll
                for (int cc = 0; cc < kHcfMaxHc; ++cc)
                    if (cc < hc) {
                        /* injw is small and stays in L2; loaded here rather than prefetched. */
                        const uint16_t* ipp = injw + (long)cc * HN + kc + kt * KT + ak;
                        const halo_u32x4 i0 = ((const halo_u32x4*)ipp)[0], i1 = ((const halo_u32x4*)ipp)[1];
                        const uint32_t iw[8] = { i0.x, i0.y, i0.z, i0.w, i1.x, i1.y, i1.z, i1.w };
                        float t = ip[cc];
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            t += nv[2 * e] * bf16_to_f32((uint16_t)iw[e]);
                            t += nv[2 * e + 1] * bf16_to_f32((uint16_t)(iw[e] >> 16));
                        }
                        ip[cc] = t;
                    }
            typedef _Float16 h2 __attribute__((ext_vector_type(2)));
#pragma unroll
            for (int i = 0; i < NBC; ++i) {
                const int qd = min(tid + 256 * i, last);
                const _Float16 wsh = (_Float16)(rbs[i] * 256.0f);
                const h2 ws2 = { wsh, wsh };
                uint32_t bd[8];
                hcf_codes16(make_uint4(rb[i].x, rb[i].y, rb[i].z, rb[i].w), bd);
#pragma unroll
                for (int q = 0; q < 8; ++q)
                    bd[q] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2, bd[q]) * ws2);
                uint16_t* dst = sB + (qd >> 2) * BP + (qd & 3) * 16;
                ((halo_u32x4*)dst)[0] = (halo_u32x4){ bd[0], bd[1], bd[2], bd[3] };
                ((halo_u32x4*)dst)[1] = (halo_u32x4){ bd[4], bd[5], bd[6], bd[7] };
            }
        }
        __syncthreads();
        if (kt + 1 < nk) HCF_PFLOAD(kt + 1);
#pragma unroll
        for (int i = 0; i < TPW; ++i) {
            const uint16_t* bp = sB + ((cb * TPW + i) * 16 + col) * BP;
#pragma unroll
            for (int ks = 0; ks < KT / 16; ++ks) {
                union { halo_u32x4 w[2]; v16hf f; } b, a0, a1;
                b.w[0] = ((const halo_u32x4*)(bp + ks * 16))[0];
                b.w[1] = ((const halo_u32x4*)(bp + ks * 16))[1];
                const uint16_t* ap0 = sA + ((2 * rp) * 16 + col) * AP + ks * 16;
                const uint16_t* ap1 = ap0 + 16 * AP;
                a0.w[0] = ((const halo_u32x4*)ap0)[0];
                a0.w[1] = ((const halo_u32x4*)ap0)[1];
                a1.w[0] = ((const halo_u32x4*)ap1)[0];
                a1.w[1] = ((const halo_u32x4*)ap1)[1];
                acc[0][i] = halo::mma(a0.f, b.f, acc[0][i]);
                acc[1][i] = halo::mma(a1.f, b.f, acc[1][i]);
            }
        }
    }
#undef HCF_PFLOAD

    float* Ps = P + (long)c * pslice;
#pragma unroll
    for (int r2 = 0; r2 < 2; ++r2)
#pragma unroll
        for (int i = 0; i < TPW; ++i) {
            const int ct = cb * TPW + i;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + (2 * rp + r2) * 16 + accRow(lane, e);
                if (m < rows) Ps[(long)m * pw + ct * 16 + col] = acc[r2][i][e];
            }
        }
    /* A row's inject terms are spread over the four waves of its k quarters: close them in LDS
     * in quarter order. */
    if (injw) {
        __syncthreads();
        float* ired = (float*)sB;                  /* [4 quarters][64 rows][hc] */
        for (int cc = 0; cc < hc; ++cc) ired[((sw & 3) * 64 + ar) * kHcfMaxHc + cc] = ip[cc];
        __syncthreads();
        if (tid < 64 && m0 + tid < rows)
            for (int cc = 0; cc < hc; ++cc)
                Ps[(long)(m0 + tid) * pw + lr + cc] =
                    ((ired[(0 * 64 + tid) * kHcfMaxHc + cc] + ired[(1 * 64 + tid) * kHcfMaxHc + cc])
                     + ired[(2 * 64 + tid) * kHcfMaxHc + cc]) + ired[(3 * 64 + tid) * kHcfMaxHc + cc];
    }
}

/* ---- the prefill plane: a mixing matrix decoded once a call to scaled f16 -----------------
 * out[r][k] = f16(e4m3(code) * scale[k / G]), the row's E4M3 rows as stored. At prefill every
 * block reads each weight many times, so decoding it once beats decoding it at every fragment;
 * f16 keeps eleven bits of the product (the general path's planes kept bf16's eight). */
__global__ __launch_bounds__(256) void hcf_plane_kernel(const uint8_t* __restrict__ src, long ld,
                                                        uint16_t* __restrict__ dst, int rows,
                                                        int K, int G) {
    const long idx = (long)blockIdx.x * 256 + threadIdx.x;
    if (idx >= (long)rows * (K / 16)) return;
    const long r = idx / (K / 16);
    const int k0 = (int)(idx - r * (K / 16)) * 16;
    const uint8_t* row = src + r * ld;
    const float s = ((const float*)(row + K))[k0 / G] * 256.0f;
    uint32_t d[8];
    halo_hcf::codes16(*(const uint4*)(row + k0), d);
    uint32_t o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        typedef _Float16 h2 __attribute__((ext_vector_type(2)));
        const h2 v = __builtin_bit_cast(h2, d[i]);
        o[i] = __builtin_bit_cast(uint32_t, (h2){ (_Float16)((float)v[0] * s), (_Float16)((float)v[1] * s) });
    }
    uint4* out = (uint4*)(dst + r * K + k0);
    out[0] = make_uint4(o[0], o[1], o[2], o[3]);
    out[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

/* ---- the partials' sum: d = bf16(silu(sum_s P[s][m][j] / hc)) as f16, and the inject gains -
 * One thread an output, the ns partials loaded together (unrolled to kHcfDecS) and summed in
 * slice order, once for every block of the up kernel. */
__global__ __launch_bounds__(256) void hcf_dsum_kernel(const float* __restrict__ P, long pslice,
                                                       int pw, int ns, uint16_t* __restrict__ D,
                                                       uint16_t* __restrict__ inj, long irow,
                                                       int rows, int hc, int lr) {
    const int width = lr + (inj ? hc : 0);
    const long o = (long)blockIdx.x * 256 + threadIdx.x;
    if (o >= (long)rows * width) return;
    const int m = (int)(o / width), j = (int)(o - (long)m * width);
    const float* pp = P + (long)m * pw + j;
    float t[kHcfDecS];
#pragma unroll
    for (int s = 0; s < kHcfDecS; ++s) t[s] = s < ns ? pp[(long)s * pslice] : 0.0f;
    float v = 0.0f;
#pragma unroll
    for (int s = 0; s < kHcfDecS; ++s)
        if (s < ns) v += t[s];
    const float inv_hc = 1.0f / (float)hc;
    if (j < lr) {
        v *= inv_hc;
        v = v * hcf_sigmoid(v);
        D[(long)m * lr + j] = (uint16_t)hcf_pk16(bf16_to_f32(f32_to_bf16(v)), 0.0f);
    } else {
        inj[(long)m * irow + (j - lr)] = f32_to_bf16(2.0f * hcf_sigmoid(v * inv_hc));
    }
}

/* One quarter of a group's W_up rows into registers: thread t takes sixteen-code chunks t,
 * t+256, ... (an index past the last is clamped, so every load is issued) and the scales of
 * staged rows t and t+256. Staged row R is W_up row (R/128)*n + g*128 + R%128. */
template <int LPT, int HC, int NTH = 256>
__device__ __forceinline__ void hcf_up_load(halo_u32x4 (&rc)[LPT], float (&rsv)[(HC * 128 + NTH - 1) / NTH],
                                            const uint8_t* __restrict__ up, long uld, int n, int g,
                                            int tid, int nur, int qch, int QB, int lr, int qq) {
    const int last = nur * qch - 1;
#pragma unroll
    for (int i = 0; i < LPT; ++i) {
        const int qd = min(tid + NTH * i, last), R = qd / qch;
        rc[i] = *(const halo_u32x4*)(up + ((long)(R >> 7) * n + (long)g * 128 + (R & 127)) * uld +
                                qq * QB + (qd - R * qch) * 16);
    }
#pragma unroll
    for (int i = 0; i < (HC * 128 + NTH - 1) / NTH; ++i) {
        const int R = min(tid + NTH * i, nur - 1);
        rsv[i] = ((const float*)(up + ((long)(R >> 7) * n + (long)g * 128 + (R & 127)) * uld + lr))[qq];
    }
}

template <int RT, int HC, int LR, bool PLANE, int NW = 8, bool DSUM = false>
__global__ __launch_bounds__(NW * 32) void hcf_upmix_kernel(
    const uint16_t* __restrict__ D, const float* __restrict__ P, long pslice, int pw, int ns,
    const uint8_t* __restrict__ up,
    long uld, const uint16_t* __restrict__ Up, const uint16_t* __restrict__ h, long hrow, const float* __restrict__ gain,
    float wadd, const float* __restrict__ sc, uint16_t* __restrict__ x, long xrow,
    uint8_t* __restrict__ q, float* __restrict__ qs,
    long qrow, long srow, uint8_t* __restrict__ rq, float* __restrict__ rs, long rqrow,
    long rsrow, int rows, int n, int i8) {
    constexpr int BM = RT * 16, hc = HC, lr = LR, NTH = NW * 32;
    constexpr int QCH = LR / 4 / 16;                 /* sixteen-code chunks a quarter */
    constexpr int LPT = (HC * 128 * QCH + NTH - 1) / NTH; /* chunks a thread stages */
    constexpr int SPT = (HC * 128 + NTH - 1) / NTH;  /* scales a thread stages */
    constexpr int TPW = HC * 8 / NW;                 /* column tiles a wave (of HC * 8) */
    constexpr int PFD = PLANE ? 1 : (NW > 8 ? 2 : 1); /* quarters in flight */
    constexpr int QU = PFD > 1 ? 4 : 1;              /* the quarter loop unrolled when it alternates */
    static_assert(!(PLANE && (NW != 8 || DSUM)), "the prefill form is eight waves over a stored d");
    extern __shared__ __attribute__((aligned(16))) unsigned char hcf_lds[];
    const int m0 = blockIdx.x * BM, g = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, col = lane & 15;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    constexpr int dp = LR + 8, QB = LR / 4, qch = QCH, nur = HC * 128;
    uint16_t* dA = (uint16_t*)hcf_lds;
    uint8_t* cq = hcf_lds + (size_t)BM * dp * 2;          /* [nur][QB] codes of a quarter */
    float* sq = (float*)(cq + (size_t)nur * QB);         /* [nur] its scales, times 2^8 */

    halo_u32x4 rc[PFD][LPT];
    float rsv[PFD][SPT];
    if constexpr (!PLANE) {
#pragma unroll
        for (int f = 0; f < PFD; ++f)
            hcf_up_load<LPT, HC, NTH>(rc[f], rsv[f], up, uld, n, g, tid, nur, qch, QB, lr, f);
    }

    if constexpr (DSUM) {
        /* d from the down product's partials, as hcf_dsum_kernel forms it (the slices summed in
         * order, then silu of the mean over the streams, rounded to bf16, held as f16): every
         * block sums them for its rows rather than a launch before it. Four columns a thread. */
        const float inv_hc = 1.0f / (float)HC;
        for (int idx = tid; idx < BM * (lr / 4); idx += NTH) {
            const int ml = idx / (lr / 4), jj = (idx - ml * (lr / 4)) * 4, m = m0 + ml;
            uint32_t w0 = 0u, w1 = 0u;
            if (m < rows) {
                const float* pp = P + (long)m * pw + jj;
                float4 t[kHcfDecS];
#pragma unroll
                for (int s = 0; s < kHcfDecS; ++s)
                    t[s] = s < ns ? *(const float4*)(pp + (long)s * pslice) : make_float4(0.f, 0.f, 0.f, 0.f);
                float v[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
                for (int s = 0; s < kHcfDecS; ++s)
                    if (s < ns) { v[0] += t[s].x; v[1] += t[s].y; v[2] += t[s].z; v[3] += t[s].w; }
                uint32_t dv[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float u = v[e] * inv_hc;
                    u = u * hcf_sigmoid(u);
                    dv[e] = hcf_pk16(bf16_to_f32(f32_to_bf16(u)), 0.0f) & 0xffffu;
                }
                w0 = dv[0] | (dv[1] << 16);
                w1 = dv[2] | (dv[3] << 16);
            }
            *(uint2*)(dA + ml * dp + jj) = make_uint2(w0, w1);
        }
    } else {
        /* d for the block's rows: eight values a chunk, rows past the last zero. */
        for (int idx = tid; idx < BM * (lr / 8); idx += NTH) {
            const int ml = idx / (lr / 8), jj = (idx - ml * (lr / 8)) * 8, m = m0 + ml;
            const uint4 v = m < rows ? *(const uint4*)(D + (long)m * lr + jj) : make_uint4(0u, 0u, 0u, 0u);
            *(uint4*)(dA + ml * dp + jj) = v;
        }
    }

    v8f acc[RT][TPW];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int i = 0; i < TPW; ++i) acc[rt][i] = (v8f)(0.0f);
    if constexpr (PLANE) {
        /* The weights' fragments straight from the f16 plane (hcf_plane_kernel's), in L2: rows
         * of the group's tiles, sixteen k a load pair, the WMMAs chained with no scale. */
        __syncthreads();
        const uint16_t* urow[TPW];
#pragma unroll
        for (int i = 0; i < TPW; ++i) {
            const int R = (wave * TPW + i) * 16 + col;
            urow[i] = Up + ((long)(R >> 7) * n + (long)g * 128 + (R & 127)) * LR;
        }
#pragma unroll 4
        for (int kk = 0; kk < LR / 16; ++kk) {
            v16hf a[RT];
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) {
                const uint16_t* ap = dA + (rt * 16 + col) * dp + kk * 16;
                union { uint4 w[2]; v16hf f; } u;
                u.w[0] = ((const uint4*)ap)[0];
                u.w[1] = ((const uint4*)ap)[1];
                a[rt] = u.f;
            }
#pragma unroll
            for (int i = 0; i < TPW; ++i) {
                union { halo_u32x4 w[2]; v16hf f; } b;
                const halo_u32x4* bp = (const halo_u32x4*)(urow[i] + kk * 16);
                b.w[0] = bp[0];
                b.w[1] = bp[1];
#pragma unroll
                for (int rt = 0; rt < RT; ++rt) acc[rt][i] = halo::mma(a[rt], b.f, acc[rt][i]);
            }
        }
    } else
#pragma unroll QU
    for (int qq = 0; qq < 4; ++qq) {
        /* PFD quarters in flight: quarter qq is staged from rc[qq % PFD], which then takes
         * quarter qq + PFD. */
        __syncthreads();                                   /* the previous quarter is consumed */
#pragma unroll
        for (int i = 0; i < LPT; ++i) {
            const int qd = min(tid + NTH * i, nur * qch - 1), R = qd / qch;
            *(halo_u32x4*)(cq + R * QB + (qd - R * qch) * 16) = rc[qq % PFD][i];
        }
#pragma unroll
        for (int i = 0; i < SPT; ++i) sq[min(tid + NTH * i, nur - 1)] = rsv[qq % PFD][i] * 256.0f;
        __syncthreads();
        if (qq + PFD < 4)
            hcf_up_load<LPT, HC, NTH>(rc[qq % PFD], rsv[qq % PFD], up, uld, n, g, tid, nur, qch, QB, lr,
                                      qq + PFD);
        for (int kk = 0; kk < qch; ++kk) {
            const int k = qq * QB + kk * 16;
            v16hf a[RT];
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) {
                const uint16_t* ap = dA + (rt * 16 + col) * dp + k;
                union { uint4 w[2]; v16hf f; } u;
                u.w[0] = ((const uint4*)ap)[0];
                u.w[1] = ((const uint4*)ap)[1];
                a[rt] = u.f;
            }
#pragma unroll
            for (int i = 0; i < TPW; ++i) {
                {
                    const int R = (wave * TPW + i) * 16 + col;
                    uint32_t bd[8];
                    hcf_codes16(*(const uint4*)(cq + R * QB + kk * 16), bd);
                    const v16hf b = hcf_frag(bd);
                    const float ws = sq[R];
#pragma unroll
                    for (int rt = 0; rt < RT; ++rt) {
                        const v8f t = halo::mma(a[rt], b, (v8f)(0.0f));
#pragma unroll
                        for (int e = 0; e < 8; ++e) acc[rt][i][e] = fmaf(ws, t[e], acc[rt][i][e]);
                    }
                }
            }
        }
    }
    __syncthreads();                               /* every wave is done with d and the codes */

    constexpr int HC128 = HC * 128;
    constexpr float inv_hc = 1.0f / (float)HC;
    /* The gates in bf16 through LDS, RTE row tiles a pass: at prefill one tile a pass, so the
     * buffer (16 x hc*128 halves) is smaller than d's and more blocks fit a WGP. */
    constexpr int RTE = PLANE ? 1 : RT;
    uint16_t* sg = (uint16_t*)hcf_lds;             /* [RTE*16][hc*128] bf16 gates */
    const int i0 = lane * 4;
    const long gc = (long)g * 128 + i0;
#pragma unroll
    for (int rp = 0; rp < RT; rp += RTE) {
        if (rp) __syncthreads();                       /* the previous pass's gates are consumed */
#pragma unroll
        for (int i = 0; i < TPW; ++i) {
            const int t = wave * TPW + i, cbase = (t >> 3) * 128 + (t & 7) * 16 + col;
#pragma unroll
            for (int rt = 0; rt < RTE; ++rt)
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    sg[(rt * 16 + accRow(lane, e)) * HC128 + cbase] =
                        f32_to_bf16(hcf_sigmoid(acc[rp + rt][i][e]));
        }
        __syncthreads();

        for (int ml = rp * 16 + wave; ml < (rp + RTE) * 16; ml += NW) {
            const int m = m0 + ml;
            if (m >= rows) break;
            float ax[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            for (int c = 0; c < hc; ++c) {
                const float sm = sc[m * kHcfMaxHc + c];
                const long kb = (long)c * n + gc;
                const uint2 hv = *(const uint2*)(h + (long)m * hrow + kb);
                const float4 gv = *(const float4*)(gain + kb);
                const uint2 sv = *(const uint2*)(sg + (ml - rp * 16) * HC128 + c * 128 + i0);
                const uint32_t hw[2] = { hv.x, hv.y }, sw[2] = { sv.x, sv.y };
                const float gg[4] = { gv.x + wadd, gv.y + wadd, gv.z + wadd, gv.w + wadd };
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const uint32_t hb = (e & 1) ? hw[e >> 1] >> 16 : hw[e >> 1] & 0xffffu;
                    const uint32_t sb = (e & 1) ? sw[e >> 1] >> 16 : sw[e >> 1] & 0xffffu;
                    ax[e] += bf16_to_f32((uint16_t)sb) * hcf_n(hb, sm, gg[e]);
                }
            }
            float xf[4];
            uint16_t xb[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                xb[e] = f32_to_bf16(ax[e] * inv_hc);
                xf[e] = bf16_to_f32(xb[e]);
            }
            *(uint2*)(x + (long)m * xrow + gc) =
                make_uint2((uint32_t)xb[0] | ((uint32_t)xb[1] << 16),
                           (uint32_t)xb[2] | ((uint32_t)xb[3] << 16));
            if (!q) continue;
            float am = fmaxf(fmaxf(fabsf(xf[0]), fabsf(xf[1])), fmaxf(fabsf(xf[2]), fabsf(xf[3])));
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) am = fmaxf(am, __shfl_xor(am, mask));
            uint32_t qw = 0u;
            if (i8) {
                const float s8 = am > 0.0f ? am * (1.0f / 127.0f) : 1.0f;
                if (lane == 0) qs[(long)m * srow + g] = s8;
                const float inv = 1.0f / s8;
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    qw |= (uint32_t)(uint8_t)(int8_t)(int)rintf(fminf(fmaxf(xf[e] * inv, -127.0f), 127.0f))
                          << (8 * e);
            } else {
                const float s8 = am > 0.0f ? am * (1.0f / 448.0f) : 1.0f;
                if (lane == 0) qs[(long)m * srow + g] = s8;
                const float inv = 1.0f / s8;
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    qw |= halo_fp8::f32_to_e4m3(fminf(fmaxf(xf[e] * inv, -448.0f), 448.0f)) << (8 * e);
            }
            *(uint32_t*)(q + (long)m * qrow + gc) = qw;
            if (!rq) continue;
            float r[4] = { xf[0], xf[1], xf[2], xf[3] };
            {
                const float a0 = r[0], b0 = r[1], a1 = r[2], b1 = r[3];
                r[0] = a0 + b0; r[1] = a0 - b0; r[2] = a1 + b1; r[3] = a1 - b1;
            }
            {
                const float a0 = r[0], b0 = r[2], a1 = r[1], b1 = r[3];
                r[0] = a0 + b0; r[2] = a0 - b0; r[1] = a1 + b1; r[3] = a1 - b1;
            }
#pragma unroll
            for (int mask = 1; mask < 32; mask <<= 1)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float o = __shfl_xor(r[e], mask);
                    r[e] = (lane & mask) ? o - r[e] : r[e] + o;
                }
            float ra = fmaxf(fmaxf(fabsf(r[0]), fabsf(r[1])), fmaxf(fabsf(r[2]), fabsf(r[3])));
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) ra = fmaxf(ra, __shfl_xor(ra, mask));
            const float rsc = ra > 0.0f ? ra * (1.0f / 448.0f) : 1.0f;
            if (lane == 0) rs[(long)m * rsrow + g] = rsc;
            const float rinv = 1.0f / rsc;
            uint32_t rw = 0u;
#pragma unroll
            for (int e = 0; e < 4; ++e)
                rw |= halo_fp8::f32_to_e4m3(fminf(fmaxf(r[e] * rinv, -448.0f), 448.0f)) << (8 * e);
            *(uint32_t*)(rq + (long)m * rqrow + gc) = rw;
        }
    }
}

/* Scratch of the fast path at band M: the scale table, then the partials of whichever form a
 * launch at up to M rows takes (16 slices up to kHcfDecRows rows, hc slices past it). */
long long hcf_pw(long long lr) { return lr + kHcfMaxHc; }
long long hcf_plane_off(long long M, long long hc, long long lr) {
    const long long slices_rows = std::max((long long)kHcfDecS * std::min(M, (long long)kHcfDecRows),
                                           hc * M);
    return (((M * kHcfMaxHc * 4 + 255) & ~255LL) + slices_rows * hcf_pw(lr) * 4 + M * lr * 2 + 255) &
           ~255LL;
}
/* The one-tile decode form's N (sixteen rows of hc*n halves), in the plane's place: a launch takes
 * one form or the other. */
constexpr long long kHcfDec1Rows = 16;
long long hcf_dec1_bytes(long long hc, long long n) { return kHcfDec1Rows * hc * n * 2; }
/* ... and past kHcfDecRows W_up's f16 plane (hc*n x lr halves). */
long long hcf_bytes(long long M, long long hc, long long lr, long long n) {
    return hcf_plane_off(M, hc, lr) +
           std::max(M > kHcfDecRows ? 2LL * lr * hc * n : 0LL, hcf_dec1_bytes(hc, n));
}
enum { HCR_H = 0, HCR_W, HCR_DOWN, HCR_UP, HCR_INJW, HCR_X, HCR_INJ, HCR_Q, HCR_S, HCR_RQ, HCR_RS };

struct HcPlan {
    long long rows, n, hc, lr, HN;
    bool f8;
    size_t off_d, off_u, off_wd, bytes;
};

int hc_plan(const RadArgs* a, const RadTensor* down, HcPlan* p) {
    p->rows = halo_geti_or(a, "M", 0);
    p->n = halo_geti_or(a, "n", 0);
    p->hc = halo_geti_or(a, "hc", 0);
    p->lr = halo_geti_or(a, "lowrank", 0);
    if (p->rows <= 0 || p->n <= 0 || p->hc <= 0 || p->lr <= 0) return RAD_E_SHAPE;
    p->HN = p->hc * p->n;
    /* The builder sizes scratch with no tensors, so the row's `mix` parameter names the format
     * when the operand is absent. */
    const char* mix = halo_gets(a, "mix");
    const bool known = down && (down->dtype == RAD_F8E4M3 || down->dtype == RAD_BF16);
    p->f8 = known ? down->dtype == RAD_F8E4M3 : (mix && !std::strcmp(mix, "e4m3"));
    const size_t align = 256;
    auto up = [&](size_t v) { return (v + align - 1) & ~(align - 1); };
    size_t off = 0;
    off = up(off + (size_t)p->rows * p->HN * 2);          /* N */
    p->off_d = off; off = up(off + (size_t)p->rows * p->lr * 2);
    p->off_u = off; off = up(off + (size_t)p->rows * p->HN * 2);
    p->off_wd = off;
    if (p->f8) off = up(off + (size_t)2 * p->lr * p->HN * 2);
    p->bytes = off;
    return RAD_OK;
}

}  // namespace

extern "C" int64_t halo_rad_hc_read_scratch(const RadArgs* a) {
    if (!a) return 0;
    const RadTensor* down = a->n_t > HCR_DOWN ? &a->t[HCR_DOWN] : nullptr;
    HcPlan p;
    if (hc_plan(a, down, &p) != RAD_OK) return 0;
    /* Either path may take a launch, so the larger of the two. */
    return (int64_t)std::max((long long)p.bytes, hcf_bytes(p.rows, p.hc, p.lr, p.n));
}

/* The form pin: every form runs this same path, so the pin changes nothing. */
static int g_hc_prefill_form = 0;
extern "C" void halo_hc_read_pin_prefill(int form) { g_hc_prefill_form = form; }

/* The E4M3 fast path's launcher: 1 when the call is not its shape or alignment (the general path
 * takes it), else the launches' result. */
static int hcf_launch(const RadArgs* a, RadStream s, const HcPlan& p, float eps, float wadd) {
    const RadTensor* h    = halo_opt(a, HCR_H);
    const RadTensor* w    = halo_opt(a, HCR_W);
    const RadTensor* down = halo_opt(a, HCR_DOWN);
    const RadTensor* up   = halo_opt(a, HCR_UP);
    const RadTensor* injw = halo_opt(a, HCR_INJW);
    const RadTensor* x    = halo_opt(a, HCR_X);
    const RadTensor* inj  = halo_opt(a, HCR_INJ);
    const RadTensor* q    = halo_opt(a, HCR_Q);
    const RadTensor* qs   = halo_opt(a, HCR_S);
    const RadTensor* rq   = halo_opt(a, HCR_RQ);
    const RadTensor* rs   = halo_opt(a, HCR_RS);
    const long long n = p.n, hc = p.hc, lr = p.lr, HN = p.HN, band = p.rows;
    /* The instantiated shape: Qwen3.8-Flash-Next's (four streams, lowrank 320). */
    if (hc != 4 || lr != 320 || n % 128) return 1;
    /* The decode form's K slices: as many as kHcfDecS, each a whole number of staged groups and
     * at most kHcfSubMax of them. */
    const long long nsub = HN / kHcfSub;
    int S = 0;
    for (int c = kHcfDecS; c >= 1 && !S; --c)
        if (nsub % c == 0 && nsub / c <= kHcfSubMax) S = c;
    if (!S) return 1;
    if (down->rank != 2 || down->shape[0] != lr || down->shape[1] != HN || up->rank != 2 ||
        up->shape[0] != HN || up->shape[1] != lr)
        return 1;
    long long rows = 0;
    const long long hrow = halo_rows_pitch(h, HN, &rows);
    if (hrow < 0 || rows <= 0 || rows > band || hrow % 8) return 1;
    auto al = [](const void* ptr, uintptr_t b) { return ((uintptr_t)ptr & (b - 1)) == 0; };
    if (!al(h->data, 16) || !al(w->data, 16) || !al(down->data, 16) || !al(up->data, 16) ||
        !halo_contig(w))
        return 1;
    if (injw && (!al(injw->data, 16) || halo_numel(injw) < hc * HN)) return 1;
    long long xr = 0, xrow = halo_rows_pitch(x, n, &xr);
    if (xrow < 0 || xr < rows || xrow % 4 || !al(x->data, 8)) return 1;
    long long irow = hc;
    if (inj) {
        long long ir = 0;
        irow = halo_rows_pitch(inj, hc, &ir);
        if (irow < 0 || ir < rows) return 1;
    }
    long long qrow = 0, srow = 0, rqrow = 0, rsrow = 0;
    if (q) {
        if (halo_geti_or(a, "group", 128) != 128) return 1;
        long long r1 = 0, r2 = 0;
        qrow = halo_rows_pitch(q, n, &r1);
        srow = halo_rows_pitch(qs, n / 128, &r2);
        if (qrow < 0 || srow < 0 || r1 < rows || r2 < rows || qrow % 4 || !al(q->data, 4)) return 1;
    }
    if (rq) {
        long long r1 = 0, r2 = 0;
        rqrow = halo_rows_pitch(rq, n, &r1);
        rsrow = halo_rows_pitch(rs, n / 128, &r2);
        if (rqrow < 0 || rsrow < 0 || r1 < rows || r2 < rows || rqrow % 4 || !al(rq->data, 4))
            return 1;
    }
    if (!a->scratch || a->scratch_bytes < hcf_bytes(band, hc, lr, n)) return 1;

    hipStream_t st = (hipStream_t)s;
    float* scs = (float*)a->scratch;
    float* P = (float*)((unsigned char*)a->scratch + ((band * kHcfMaxHc * 4 + 255) & ~255LL));
    const int pw = (int)hcf_pw(lr);
    const long dld = halo_hc8::row_bytes(HN, halo_hc8::down_group(HN));
    const long uld = halo_hc8::row_bytes(lr, halo_hc8::up_group(lr));
    const uint16_t* hp = (const uint16_t*)halo_p(h);
    const float* wp = (const float*)halo_p(w);
    const uint8_t* dp = (const uint8_t*)halo_p(down);
    const uint8_t* upp = (const uint8_t*)halo_p(up);
    const uint16_t* ip = injw ? (const uint16_t*)halo_p(injw) : nullptr;
    const bool dec = rows <= kHcfDecRows;
    const long pslice = (long)(dec ? std::min(band, (long long)kHcfDecRows) : band) * pw;
    /* The prefill form's f16 planes sit after D (hcf_bytes). */
    uint16_t* Wup = (uint16_t*)((unsigned char*)a->scratch + hcf_plane_off(band, hc, lr));

    uint16_t* xp = (uint16_t*)halo_p(x);
    uint16_t* injp = inj ? (uint16_t*)halo_p(inj) : nullptr;
    uint8_t* qp = q ? (uint8_t*)halo_p(q) : nullptr;
    float* qsp = q ? (float*)halo_p(qs) : nullptr;
    uint8_t* rqp = rq ? (uint8_t*)halo_p(rq) : nullptr;
    float* rsp = rq ? (float*)halo_p(rs) : nullptr;
    const int i8 = (q && q->dtype == RAD_I8) ? 1 : 0;
    uint16_t* Dd = (uint16_t*)(P + (long)(dec ? S : hc) * pslice);

    if (!dec) {
        /* W_up decoded once to a scaled f16 plane for the up kernel. W_down stays coded: its
         * fragments decoded from LDS measured faster than f16 ones read from the plane in L2
         * (1.15 against 2.5 ms at 2048 rows).
         *
         * The rows run in slices of kHcfSlice: each slice's h (20 MiB at 1024 rows of Flash-Next)
         * stays in the 32 MiB last-level cache between the norm pass that streams it and the two
         * products that read it back in short runs; a whole 2048-row chunk does not, and those
         * runs then go to DRAM (2.8 against 1.6 ms a chunk). */
        constexpr long long kHcfSlice = 1024;
        const long pu = (long)HN * (lr / 16);
        hcf_plane_kernel<<<dim3((unsigned)((pu + 255) / 256)), 256, 0, st>>>(upp, uld, Wup, (int)HN,
                                                                            (int)lr, (int)halo_hc8::up_group(lr));
        /* d's 32 rows, or one row tile's gates (hcf_upmix_kernel's epilogue): six blocks a WGP.
         * 32 rows a block measured best (48: 306, 64: 390, 16: 358 against 280 us a slice). */
        const size_t smem = (size_t)std::max(32 * (lr + 8) * 2, 16 * hc * 128 * 2);
        for (long long r0 = 0; r0 < rows; r0 += kHcfSlice) {
            const int sr = (int)std::min(kHcfSlice, rows - r0);
            const uint16_t* hs = hp + r0 * hrow;
            hcf_stats_kernel<<<dim3((unsigned)hc, (unsigned)sr), 256, 0, st>>>(hs, hrow, scs, (int)n, eps);
            hcf_down_pf_kernel<320><<<dim3((unsigned)((sr + 63) / 64), (unsigned)hc), 256, 0, st>>>(
                hs, hrow, wp, wadd, scs, dp, dld, ip, P, pslice, pw, sr, (int)n, (int)hc, (int)HN,
                (int)lr);
            const long nout = (long)sr * (lr + (injp ? hc : 0));
            hcf_dsum_kernel<<<dim3((unsigned)((nout + 255) / 256)), 256, 0, st>>>(
                P, pslice, pw, (int)hc, Dd, injp ? injp + r0 * irow : nullptr, irow, sr, (int)hc, (int)lr);
            hcf_upmix_kernel<2, 4, 320, true><<<dim3((unsigned)((sr + 31) / 32), (unsigned)(n / 128)), 256,
                                                 smem, st>>>(
                Dd, nullptr, 0, 0, 0, upp, uld, Wup, hs, hrow, wp, wadd, scs, xp + r0 * xrow, xrow,
                qp ? qp + r0 * qrow : nullptr, qsp ? qsp + r0 * srow : nullptr, qrow, srow,
                rqp ? rqp + r0 * rqrow : nullptr, rsp ? rsp + r0 * rsrow : nullptr, rqrow, rsrow,
                sr, (int)n, i8);
        }
        return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
    }

    /* Up to sixteen rows: N once (hcf_normdec_kernel), then the staging-free down product. */
    const bool dec1 = rows <= kHcfDec1Rows && HN / kHcfSub / S == 5 && n / 8 <= 512 &&
                      (ip != nullptr) == (injp != nullptr);
    uint16_t* Nf = (uint16_t*)Wup;
    if (dec1) {
        hcf_normdec_kernel<4><<<dim3((unsigned)rows), 4 * 256, 0, st>>>(
            hp, hrow, wp, wadd, ip, scs, Nf, injp, irow, (int)n, eps);
        hcf_down_dec1_kernel<5><<<dim3((unsigned)(lr / 16), (unsigned)S), 256, 0, st>>>(
            Nf, dp, dld, P, pslice, pw, (int)rows, (int)HN);
    } else {
    hcf_stats_kernel<<<dim3((unsigned)hc, (unsigned)rows), 256, 0, st>>>(hp, hrow, scs, (int)n, eps);
    {
        const dim3 grid((unsigned)(lr / 16), (unsigned)S);
#define HCF_DEC(RT) hcf_down_dec_kernel<RT><<<grid, 256, 0, st>>>(hp, hrow, wp, wadd, scs, dp, dld, \
            ip, P, pslice, pw, (int)rows, (int)n, (int)hc, (int)HN, (int)lr)
        switch ((rows + 15) / 16) {
            case 1: HCF_DEC(1); break;
            case 2: HCF_DEC(2); break;
            case 3: HCF_DEC(3); break;
            default: HCF_DEC(4); break;
        }
#undef HCF_DEC
    }
    }
    if (hipGetLastError() != hipSuccess) return HALO_NO(RAD_E_DEVICE);

    const int RTu = rows <= 16 ? 1 : 2, BM = 16 * RTu;
    const size_t stage = (size_t)BM * (lr + 8) * 2 + (size_t)hc * 128 * (lr / 4 + 4);
    const size_t smem = std::max(stage, (size_t)(BM * hc * 128 * 2));
    const dim3 ugrid((unsigned)((rows + BM - 1) / BM), (unsigned)(n / 128));
    if (dec1) {
        /* Sixteen waves, two quarters of W_up in flight, d summed from the partials in the
         * kernel: twenty blocks a call, one a WGP, so each block's loads in flight set the pace. */
        hcf_upmix_kernel<1, 4, 320, false, 16, true><<<ugrid, 512, smem, st>>>(
            nullptr, P, pslice, pw, S, upp, uld, Wup, hp, hrow, wp, wadd, scs, xp, xrow, qp, qsp,
            qrow, srow, rqp, rsp, rqrow, rsrow, (int)rows, (int)n, i8);
        return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
    }
    const long nout = rows * (lr + (injp ? hc : 0));
    hcf_dsum_kernel<<<dim3((unsigned)((nout + 255) / 256)), 256, 0, st>>>(P, pslice, pw, S, Dd, injp,
                                                                         irow, (int)rows, (int)hc,
                                                                         (int)lr);
#define HCF_UP(RT) hcf_upmix_kernel<RT, 4, 320, false><<<ugrid, 256, smem, st>>>(Dd, nullptr, 0, 0, 0, \
        upp, uld, Wup, hp, hrow, wp, wadd, scs, xp, xrow, qp, qsp, qrow, srow, rqp, rsp, rqrow, \
        rsrow, (int)rows, (int)n, i8)
    if (RTu == 1) HCF_UP(1); else HCF_UP(2);
#undef HCF_UP
    return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}

extern "C" int halo_rad_hc_read(const RadArgs* a, RadStream s) {
    (void)g_hc_prefill_form;
    const RadTensor* h    = halo_opt(a, HCR_H);
    const RadTensor* w    = halo_opt(a, HCR_W);
    const RadTensor* down = halo_opt(a, HCR_DOWN);
    const RadTensor* up   = halo_opt(a, HCR_UP);
    const RadTensor* injw = halo_opt(a, HCR_INJW);
    const RadTensor* x    = halo_opt(a, HCR_X);
    const RadTensor* inj  = halo_opt(a, HCR_INJ);
    const RadTensor* q    = halo_opt(a, HCR_Q);
    const RadTensor* qs   = halo_opt(a, HCR_S);
    const RadTensor* rq   = halo_opt(a, HCR_RQ);
    const RadTensor* rs   = halo_opt(a, HCR_RS);
    if (!h || !w || !down || !up || !x) return HALO_NO(RAD_E_INVAL);
    if (h->dtype != RAD_BF16 || w->dtype != RAD_F32 || x->dtype != RAD_BF16)
        return HALO_NO(RAD_E_DTYPE);
    const bool f8 = down->dtype == RAD_F8E4M3;
    if (f8) { if (up->dtype != RAD_F8E4M3) return HALO_NO(RAD_E_DTYPE); }
    else if (down->dtype != RAD_BF16 || up->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);
    if ((injw != nullptr) != (inj != nullptr)) return HALO_NO(RAD_E_INVAL);
    if (injw && (injw->dtype != RAD_BF16 || inj->dtype != RAD_BF16)) return HALO_NO(RAD_E_DTYPE);
    if ((q != nullptr) != (qs != nullptr)) return HALO_NO(RAD_E_INVAL);
    if (q && q->dtype != RAD_F8E4M3 && q->dtype != RAD_I8) return HALO_NO(RAD_E_DTYPE);
    if (q && qs->dtype != RAD_F32) return HALO_NO(RAD_E_DTYPE);
    if ((rq != nullptr) != (rs != nullptr) || (rq && !q)) return HALO_NO(RAD_E_INVAL);
    if (rq && (rq->dtype != RAD_F8E4M3 || rs->dtype != RAD_F32)) return HALO_NO(RAD_E_DTYPE);

    HcPlan p;
    int rc = hc_plan(a, down, &p);
    if (rc != RAD_OK) return HALO_NO(rc);
    const long long n = p.n, hc = p.hc, lr = p.lr, HN = p.HN;
    if (hc < 1 || hc > 8) return HALO_NO(RAD_E_SHAPE);
    if (n % 128) return HALO_NO(RAD_E_SHAPE);
    /* The rows are h's; `M` is the band the scratch was sized for (as libr4d's hc_common reads
     * them). A step at fewer rows than its band is the engine's normal case. */
    long long rows = 0;
    const long long hrow = halo_rows_pitch(h, HN, &rows);
    if (hrow < 0 || rows <= 0) return HALO_NO(RAD_E_STRIDE);
    if (rows > p.rows || halo_numel(w) < HN) return HALO_NO(RAD_E_SHAPE);

    const float eps  = (float)halo_getf(a, "eps", 1e-6);
    const float wadd = (float)halo_getf(a, "wadd", 0.0);
    const char* dt = halo_gets(a, "dtype");
    if (dt && *dt && std::strcmp(dt, "bf16")) return HALO_NO(RAD_E_DTYPE);

    if (f8) {
        const int frc = hcf_launch(a, s, p, eps, wadd);
        if (frc != 1) return frc;
    }
    /* The general path below reads h as contiguous rows, in its own scratch layout. */
    if (hrow != HN) return HALO_NO(RAD_E_STRIDE);
    if (!a->scratch || (size_t)a->scratch_bytes < p.bytes) return HALO_NO(RAD_E_SCRATCH);

    hipStream_t st = (hipStream_t)s;
    unsigned char* scr = (unsigned char*)a->scratch;
    uint16_t* Nbuf = (uint16_t*)(scr);
    uint16_t* dbuf = (uint16_t*)(scr + p.off_d);
    uint16_t* ubuf = (uint16_t*)(scr + p.off_u);
    const uint16_t* Wd = (const uint16_t*)halo_p(down);
    const uint16_t* Wu = (const uint16_t*)halo_p(up);
    if (f8) {
        uint16_t* Wdb = (uint16_t*)(scr + p.off_wd);
        uint16_t* Wub = Wdb + (size_t)lr * HN;
        const long dld = halo_hc8::row_bytes(HN, halo_hc8::down_group(HN));
        const long uld = halo_hc8::row_bytes(lr, halo_hc8::up_group(lr));
        auto dec = [&](const uint8_t* srcp, uint16_t* dstp, long rws, long K, long G, long ld) {
            const long chunks = rws * (K / 16);
            hc_decode_f8_kernel<<<dim3((unsigned)((chunks + 255) / 256)), 256, 0, st>>>(
                srcp, dstp, rws, K, G, ld);
        };
        dec((const uint8_t*)halo_p(down), Wdb, lr, HN, halo_hc8::down_group(HN), dld);
        dec((const uint8_t*)halo_p(up), Wub, HN, lr, halo_hc8::up_group(lr), uld);
        if (hipGetLastError() != hipSuccess) return HALO_NO(RAD_E_DEVICE);
        Wd = Wdb;
        Wu = Wub;
    }

    hc_norm_kernel<<<dim3((unsigned)hc, (unsigned)rows), 256, 0, st>>>(
        (const uint16_t*)halo_p(h), (const float*)halo_p(w), Nbuf, n, HN, eps, wadd);
    if (hipGetLastError() != hipSuccess) return HALO_NO(RAD_E_DEVICE);

    const float inv_hc = 1.0f / (float)hc;
    if (hc_gemm_launch(Nbuf, Wd, (__hip_bfloat16*)dbuf, (int)rows, (int)HN, (int)lr,
                       (int)HN, (int)HN, (int)lr, 1, inv_hc, st) != 0)
        return HALO_NO(RAD_E_DEVICE);
    if (hc_gemm_launch(dbuf, Wu, (__hip_bfloat16*)ubuf, (int)rows, (int)lr, (int)HN,
                       (int)lr, (int)lr, (int)HN, 0, inv_hc, st) != 0)
        return HALO_NO(RAD_E_DEVICE);

    if (injw) {
        long long irows_ = 0;
        const long long irow = halo_rows_pitch(inj, hc, &irows_);
        if (irow < 0) return HALO_NO(RAD_E_STRIDE);
        hc_inj_kernel<<<dim3((unsigned)hc, (unsigned)rows), 128, 0, st>>>(
            Nbuf, (const uint16_t*)halo_p(injw), (uint16_t*)halo_p(inj), HN, (int)hc, irow);
        if (hipGetLastError() != hipSuccess) return HALO_NO(RAD_E_DEVICE);
    }

    long long xrows_ = 0, qrows_ = 0, srows_ = 0;
    const long long xrow = halo_rows_pitch(x, n, &xrows_);
    if (xrow < 0 || xrows_ < rows) return HALO_NO(RAD_E_STRIDE);
    const long long qrow = q ? halo_rows_pitch(q, n, &qrows_) : n;
    const long long srow = q ? halo_rows_pitch(qs, n / 128, &srows_) : n / 128;
    if (q && (qrow < 0 || srow < 0 || qrows_ < rows || srows_ < rows)) return HALO_NO(RAD_E_STRIDE);
    hc_mix_kernel<<<dim3((unsigned)(n / 128), (unsigned)rows), 128, 0, st>>>(
        Nbuf, ubuf, (uint16_t*)halo_p(x),
        q ? (uint8_t*)halo_p(q) : nullptr, q ? (float*)halo_p(qs) : nullptr,
        rq ? (uint8_t*)halo_p(rq) : nullptr, rs ? (float*)halo_p(rs) : nullptr,
        n, HN, (int)hc, xrow, qrow, srow, (q && q->dtype == RAD_I8) ? 1 : 0);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
