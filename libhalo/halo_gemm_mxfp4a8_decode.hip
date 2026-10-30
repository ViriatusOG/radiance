/* Derived from libr4d/r4d_gemm_mxfp4a8_decode.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_gemm_mxfp4a8_decode.hip - gemm_mxfp4a8_decode and gemm_mxfp4a8_decode_shortk (the
// same entry point): the MXFP4 GEMM at M up to 128 with K split across workgroups.
//
// Split partials land in the op's scratch, one f32 [M][N] plane a split, and a second kernel
// sums the planes in split order (times the per-row activation scale on the per-row grid).
// The summation order is the split index order at every M, so a row's bits do not depend on
// how many rows ride with it; with ks == 1 the GEMM kernel writes the output itself and no
// scratch is touched. libr4d fuses the reduction into the last workgroup to arrive (arrival
// counters keyed on the output); the separate reduction kernel is the same arithmetic with
// no shared state at all, which is what the row's concurrent-stream contract wants: two
// launches on unordered streams with disjoint scratch slices cannot interfere, and there
// are no counters to leave behind.
//
// The per-element accumulation is the skinny kernel's: bf16 WMMA over 16-wide steps, the
// weight element e2m1 * 2^(exp - ref) exact in bf16, folds at each 32-block close (and at
// the split's end) with 2^(ref - 127) and the (row, 128) activation scale.
#include "halo.h"
#include "halo_gemm_args.h"
#include "halo_plugin.h"   // halo_rad_gemm_mxfp4a8_tiled: the band past 64 rows
#include "halo_common.h"
#include "halo_gdn_wmma.h"
#include "halo_fp8_wmma.h"
#include "halo_wmma.h"
#include <hip/hip_bf16.h>
#include <cstdint>

#define WAVE32 32

typedef int      int4v  __attribute__((ext_vector_type(4)));

__device__ __forceinline__ v16bf mxd_deq16_exact(const uint32_t* __restrict__ w4) {
    union { uint32_t u[8]; v16bf f; } q;
#pragma unroll
    for (int d = 0; d < 4; ++d) {
        const v2f p0 = halo_fp8::e4m3_pk<false>(w4[d]);
        const v2f p1 = halo_fp8::e4m3_pk<true>(w4[d]);
        q.u[d * 2]     = halo::pk_bf16(p0.x, p0.y);
        q.u[d * 2 + 1] = halo::pk_bf16(p1.x, p1.y);
    }
    return q.f;
}

__device__ __forceinline__ unsigned mxd_e2m1_bf16(unsigned code, int ediff) {
    const unsigned mag = code & 7u;
    if (!mag) return (code & 8u) << 12;
    unsigned E = mag >> 1, m = mag & 1u;
    if (mag == 1) m = 0;   /* the subnormal 0.5 */
    const unsigned ef = E + (unsigned)(ediff + 126);
    return ((code & 8u) << 12) | (ef << 7) | (m << 6);
}

/* Eight e2m1 codes at one (exp - ref) to four bf16-pair dwords, four codes a pass:
 * halo_gemm_mxfp4a8_tiled.hip's mxt_frag8 (that file says how), bit-identical to the scalar
 * form for every code and every ediff with 0 <= E + ediff + 126 <= 255. */
typedef unsigned short mxd_us2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ void mxd_frag8(unsigned dw, int ediff, uint32_t* out) {
    const uint32_t ed = (uint32_t)(ediff << 7) & 0xFFFFu;
    const mxd_us2 ED2 = __builtin_bit_cast(mxd_us2, ed | (ed << 16));
    const uint32_t z = (0u - ed) & 0xFFFFu;
    const uint32_t T_LO = 0xC0800000u | (z & 0xFFu), T_HI = 0x3F3F3F00u | (z >> 8);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t x = __builtin_amdgcn_perm(dw, dw, h ? 0x03030202u : 0x01010000u);
        const uint32_t mag = (x & 0x00070007u) | ((x >> 4) & 0x07000700u);
        const uint32_t sg = ((x << 4) & 0x00800080u) | (x & 0x80008000u);
        const uint32_t lo = __builtin_amdgcn_perm(0xC0804000u, T_LO, mag);
        const uint32_t hi = __builtin_amdgcn_perm(0x40404040u, T_HI, mag);
        const mxd_us2 p0 = __builtin_bit_cast(mxd_us2, __builtin_amdgcn_perm(hi, lo, 0x05010400u)) + ED2;
        const mxd_us2 p1 = __builtin_bit_cast(mxd_us2, __builtin_amdgcn_perm(hi, lo, 0x07030602u)) + ED2;
        out[2 * h]     = __builtin_bit_cast(uint32_t, p0) | __builtin_amdgcn_perm(0u, sg, 0x010C000Cu);
        out[2 * h + 1] = __builtin_bit_cast(uint32_t, p1) | __builtin_amdgcn_perm(0u, sg, 0x030C020Cu);
    }
}

__device__ __forceinline__ v16bf mxd_frag16(unsigned dw0, unsigned dw1, int exp, int ref) {
    union { uint32_t u[8]; v16bf f; } q;
    mxd_frag8(dw0, exp - ref, q.u);
    mxd_frag8(dw1, exp - ref, q.u + 4);
    return q.f;
}

/* One split's partial: rows [0, M), its K range [kbase, kbase + Ks). Writes f32 partials
 * (PARTIAL) or the bf16 output directly (ks == 1). WPW waves share a block; a wave's tile
 * and its K order are unchanged by the block's width, so the output bits do not depend
 * on WPW -- it only buys latency hiding on a part whose grid is otherwise a wave a CU. */
template <int MT, int NPW, bool ABLK, bool PARTIAL, int WPW = 8>
__global__ __launch_bounds__(WPW * WAVE32) void mxd_partial_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, void* __restrict__ Out,
    int M, int K, int N, int lda, int ldws, int ldc, int kbase, int Ks, int as_rs,
    const uint16_t* __restrict__ Adq = nullptr) {
    const int lane = threadIdx.x & (WAVE32 - 1), wave = threadIdx.x / WAVE32;
    const int nt0 = ((int)blockIdx.x * WPW + wave) * NPW;
    const int mt0 = (int)blockIdx.y * MT;
    const int nlane = lane & 15;
    const unsigned int* __restrict__ Wp = (const unsigned int*)W;
    const int ksteps = K / 16;
    const int kend = kbase + Ks;

    const int ntmax = N / 16 - 1;
    int wtile[NPW], wrow[NPW];
    float rf[NPW];
#pragma unroll
    for (int j = 0; j < NPW; j++) {
        const int t = nt0 + j;
        wtile[j] = t < ntmax ? t : ntmax;
        wrow[j] = wtile[j] * 16 + nlane;
        rf[j] = __int_as_float((int)Wref[wrow[j]] << 23);
    }

    v8f acc[NPW][MT];
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

    v8f tmp[NPW][MT];
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int i = 0; i < MT; i++) tmp[j][i] = (v8f)(0.f);

    for (int kg = kbase; kg < kend; kg += 16) {
        const int kstep = kg / 16;
        v16bf af[MT];
#pragma unroll
        for (int i = 0; i < MT; i++) {
            int r = (mt0 + i) * 16 + nlane;
            r = r < M - 1 ? r : M - 1;
            if (Adq) {
                /* The launch's activation plane (mxd_deq16_exact's outputs, [M, K] bf16). */
                const uint16_t* src = Adq + (size_t)r * K + kg;
                uint32_t* dst = (uint32_t*)&af[i];
                *(uint4*)dst = *(const uint4*)src;
                *(uint4*)(dst + 4) = *(const uint4*)(src + 8);
            } else {
                const int4v av = *(const int4v*)(A + (size_t)r * lda + kg);
                af[i] = mxd_deq16_exact((const uint32_t*)&av);
            }
        }
#pragma unroll
        for (int j = 0; j < NPW; j++) {
            const int n = wrow[j];
            const unsigned dw0 = Wp[((size_t)wtile[j] * ksteps + kstep) * WAVE32 + nlane];
            const unsigned dw1 = Wp[((size_t)wtile[j] * ksteps + kstep) * WAVE32 + 16 + nlane];
            const v16bf bf = mxd_frag16(dw0, dw1,
                                        (int)Ws[(size_t)(kg / 32) * ldws + n], (int)Wref[n]);
#pragma unroll
            for (int i = 0; i < MT; i++) tmp[j][i] = halo::mma(af[i], bf, tmp[j][i]);
        }
        if (((kg + 16) % 32 == 0) || (kg + 16 == kend)) {
            /* The 32-block (or the split tail) closed: fold it. kg/128 is its scale group. */
#pragma unroll
            for (int j = 0; j < NPW; j++)
#pragma unroll
                for (int i = 0; i < MT; i++)
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        float f = rf[j];
                        if constexpr (ABLK) {
                            int r = (mt0 + i) * 16 + 2 * e + (lane >> 4);
                            r = r < M - 1 ? r : M - 1;
                            f *= As[(size_t)r * as_rs + (kg >> 7)];
                        }
                        acc[j][i][e] = fmaf(tmp[j][i][e], f, acc[j][i][e]);
                        tmp[j][i][e] = 0.f;
                    }
        }
    }

#pragma unroll
    for (int j = 0; j < NPW; j++) {
        const int n = nt0 * 16 + j * 16 + nlane;
        if (n >= N) continue;
#pragma unroll
        for (int i = 0; i < MT; i++) {
            const int mb = (mt0 + i) * 16;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = mb + accRow(lane, e);
                if (m >= M) continue;
                if constexpr (PARTIAL) {
                    ((float*)Out)[(size_t)m * N + n] = acc[j][i][e];
                } else {
                    ((__hip_bfloat16*)Out)[(size_t)m * ldc + n] =
                        (__hip_bfloat16)(ABLK ? acc[j][i][e] : acc[j][i][e] * As[m]);
                }
            }
        }
    }
}

/* The split reduction. `N` is the part's width (the partial planes' stride) and `ldc` the output's
 * row pitch: a part is a window of the real [M, N_full] output at column `n0`, so with a partitioned
 * weight N != ldc and writing at idx would land the part's columns in the wrong place (and leave
 * the rest of the row at its initial NaN). */
template <bool ABLK>
__global__ void mxd_reduce_kernel(const float* __restrict__ P, const float* __restrict__ As,
                                  __hip_bfloat16* __restrict__ C, int M, int N, int ldc, int ks) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)M * N) return;
    const int m = (int)(idx / N), n = (int)(idx % N);
    float v = 0.f;
    for (int s = 0; s < ks; ++s) v += P[(size_t)s * M * N + idx];
    if constexpr (!ABLK) v *= As[m];
    C[(size_t)m * ldc + n] = (__hip_bfloat16)v;
}

/* The activation plane: mxd_deq16_exact over the whole [M, K) activation, once per
 * launch; the partial kernel's waves read fragments instead of each
 * re-dequantising the same codes for its column. Bit-identical products. */
__global__ __launch_bounds__(256) void mxd_adq_kernel(const unsigned char* __restrict__ A,
                                                      uint16_t* __restrict__ Adq,
                                                      int M, int K, int lda) {
    const int idx = blockIdx.x * (int)blockDim.x + threadIdx.x;
    if (idx >= M * (K / 16)) return;
    const int r = idx / (K / 16), c = (idx % (K / 16)) * 16;
    const int4v av = *(const int4v*)(A + (size_t)r * lda + c);
    const v16bf f = mxd_deq16_exact((const uint32_t*)&av);
    uint4* dst = (uint4*)(Adq + (size_t)r * K + c);
    const uint4* src = (const uint4*)&f;
    dst[0] = src[0];
    dst[1] = src[1];
}

/* The fill rule for the split count: enough workgroups to occupy the card, from (N, K)
 * alone so a row's bits never depend on M. Legal splits divide K/32. The smallest split
 * that fills the card wins: a wider split is more launches and more partial traffic
 * for the same waves. (An earlier form took the largest, which at this geometry split
 * a full-width part sixteen ways for no occupancy gain.) */
static int mxd_ks_law(long long N, long long K) {
    const long long tiles = N / 16;
    const long long colblocks = (tiles + 7) / 8;      /* WPW 8 waves x NPW 1 tiles a block */
    for (int s : { 1, 2, 4, 8, 16 }) {
        if (((K / 32) % s) == 0 && colblocks * s >= 40) return s;
    }
    return 1;
}

template <bool ABLK>
static int mxd_run(const RadArgs* a, RadStream s, const GemmOps& g, const MxParts& mp,
                   int as_rs) {
    const long long M = g.M, K = g.K;
    const int ks_t = (int)halo_geti_or(a, "ks", 0);
    (void)halo_geti_or(a, "bk", 0);                   /* the slab: folded per 32 either way */
    (void)halo_geti_or(a, "nt", 0);                   /* the load hint */
    int ks = ks_t > 0 ? ks_t : mxd_ks_law(g.N, g.K);
    if (ks > 1 && ((K / 32) % ks)) ks = 1;
    const long long plane = M * g.N * 4;
    /* A split needs one f32 plane per split in the op's scratch, and a shorter one is refused
     * rather than clamped: the row's bits depend on the split count, so silently reducing it would
     * answer a shape with a different kernel than the one the hook sized the scratch for (and a
     * caller that under-sized it hears about it here, as libr4d's row does). */
    if (ks > 1) {
        while (ks > 1 && ((K / 32) % ks)) --ks;
        if (!a->scratch || a->scratch_bytes < (long long)ks * plane)
            return HALO_NO(RAD_E_SCRATCH);
    }
    const long long kb = (K + 127) / 128;
    for (int p = 0; p < mp.n; ++p) {
        const long long n0 = mp.n0[p], np = mp.n1[p] - mp.n0[p];
        const int lda = (int)(mp.lda ? mp.lda : K);
        const float* as_base = (const float*)(halo_addr(g.ascale) + p * kb * 4);
        /* The activation plane sits behind this part's split partials in the
         * scratch (at offset 0 when the direct path writes none). Callers that
         * sized no room for it keep the in-kernel decode. */
        const long long partial_bytes = ks > 1 ? (long long)ks * M * np * 4 : 0;
        const uint16_t* adq = nullptr;
        if (a->scratch && a->scratch_bytes >= partial_bytes + (long long)M * K * 2) {
            adq = (const uint16_t*)((const char*)a->scratch + partial_bytes);
            const int total = (int)(M * (K / 16));
            hipLaunchKernelGGL((mxd_adq_kernel), dim3((total + 255) / 256), 256, 0, (hipStream_t)s, 
                (const unsigned char*)halo_addr(g.a) + p * K, (uint16_t*)adq, (int)M, (int)K, lda);
        }
        /* One tile a wave (NPW 1) and, at M <= 16, one M tile: this part has no
         * in-block split, so the wave count is its parallelism -- NPW 2 halves it and
         * measured ~2x slower here, unlike the m64 kernel where SK supplies waves.
         * Per-output accumulation order is unchanged either way. */
        auto launch_partial = [&](int split, int kbase, int Ks, void* out, bool partial) {
            const int mt = M <= 16 ? 1 : 2;
            const dim3 grid((unsigned)((np / 16 + 7) / 8), (unsigned)((M + mt * 16 - 1) / (mt * 16)), 1);
            auto emit = [&](auto kern) {
                hipLaunchKernelGGL((kern), grid, 256, 0, (hipStream_t)s, 
                    (const unsigned char*)halo_addr(g.a) + p * K, as_base,
                    (const unsigned char*)halo_addr(g.b) + n0 * K / 2,
                    (const unsigned char*)halo_addr(g.bscale) + n0,
                    (const unsigned char*)halo_addr(g.bref) + n0, out,
                    (int)M, (int)K, (int)np, lda, (int)g.N, (int)g.N, kbase, Ks, as_rs, adq);
            };
            if (partial) {
                if (ABLK) { if (mt == 1) emit(mxd_partial_kernel<1, 1, true, true>); else emit(mxd_partial_kernel<2, 1, true, true>); }
                else      { if (mt == 1) emit(mxd_partial_kernel<1, 1, false, true>); else emit(mxd_partial_kernel<2, 1, false, true>); }
            } else {
                if (ABLK) { if (mt == 1) emit(mxd_partial_kernel<1, 1, true, false>); else emit(mxd_partial_kernel<2, 1, true, false>); }
                else      { if (mt == 1) emit(mxd_partial_kernel<1, 1, false, false>); else emit(mxd_partial_kernel<2, 1, false, false>); }
            }
        };
        if (ks == 1) {
            launch_partial(0, 0, (int)K,
                           (void*)((const char*)halo_p(g.y) + (size_t)n0 * 2), false);
        } else {
            const int Ks = (int)(K / ks);
            for (int sp = 0; sp < ks; ++sp)
                launch_partial(sp, sp * Ks, Ks,
                               (char*)a->scratch + (size_t)sp * M * np * 4, true);
            const long elems = M * np;
            const int threads = 256;
            const long blocks = (elems + threads - 1) / threads;
            if (ABLK)
                hipLaunchKernelGGL((mxd_reduce_kernel<true>), (unsigned)blocks, threads, 0, (hipStream_t)s, 
                    (const float*)a->scratch, as_base,
                    (__hip_bfloat16*)((const char*)halo_p(g.y) + (size_t)n0 * 2),
                    (int)M, (int)np, (int)g.N, ks);
            else
                hipLaunchKernelGGL((mxd_reduce_kernel<false>), (unsigned)blocks, threads, 0, (hipStream_t)s, 
                    (const float*)a->scratch, as_base,
                    (__hip_bfloat16*)((const char*)halo_p(g.y) + (size_t)n0 * 2),
                    (int)M, (int)np, (int)g.N, ks);
        }
        if (hipGetLastError() != hipSuccess) return HALO_NO(RAD_E_DEVICE);
    }
    return RAD_OK;
}

extern "C" int64_t halo_rad_gemm_mxfp4a8_decode_scratch(const RadArgs* a) {
    const long long M = halo_geti_or(a, "M", 0);
    const long long N = halo_geti_or(a, "N", 0);
    const long long K = halo_geti_or(a, "K", 0);
    if (M <= 0 || N <= 0 || K <= 0) return 0;
    int ks = (int)halo_geti_or(a, "ks", 0);
    if (ks <= 0) ks = mxd_ks_law(N, K);
    if (ks > 1 && ((K / 32) % ks)) ks = 1;
    return (int64_t)ks * M * N * 4 + M * K * 2;   /* partials + the activation plane */
}

extern "C" int halo_rad_gemm_mxfp4a8_decode(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1, /*parts=*/1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return HALO_NO(RAD_E_SHAPE);
    if (g.M < 1 || g.M > 128) return HALO_NO(RAD_E_SHAPE);
    if (!g.ascale) return HALO_NO(RAD_E_INVAL);
    if (!g.bref) return HALO_NO(RAD_E_INVAL);
    /* Past 64 rows the tiled row's kernel (byte-identical to nt_m64 at one split): this
     * kernel's M unroll costs more than the tile there (3.2 ms against 1.3 ms at M 128, N 5120,
     * K 25600; 2.1 against 1.3 at M 65). The row's bits stay M-invariant through 64 rows, the
     * band the verify steps and the gemv rows share; at 64 -> 65 the selection changes rows
     * anyway. The scratch the hook sized holds the tiled kernel's activation plane. */
    if (g.M > 64) return halo_rad_gemm_mxfp4a8_tiled(a, s);
    MxParts mp;
    rc = mx_parts(a, &g, &mp);
    if (rc != RAD_OK) return rc;
    int as_group = 0, as_rs = 1;
    rc = mx_ascale_grid(g, &as_group, &as_rs, mp.n);
    if (rc != RAD_OK) return rc;
    /* Past one row tile at one split (the law's answer from 640 column tiles up, so every
     * decode_shortk shape), the LDS-staged tile at one split: the same bytes (one chain of folds
     * in K order, stored as is), the weight read once for all the rows rather than once a 32-row
     * block (19456 x 2560 at 64 rows: 520 -> ~320 us on gfx1151). The plane sits where mxd_run
     * puts it, at the scratch's start when there are no partials. */
    {
        const int ks_t = (int)halo_geti_or(a, "ks", 0);
        int ks = ks_t > 0 ? ks_t : mxd_ks_law(g.N, g.K);
        if (ks > 1 && ((g.K / 32) % ks)) ks = 1;
        if (ks == 1 && g.M > 16) {
            rc = halo_mxfp4a8_tiled_split(a, s, g, mp, as_group, as_rs, 1);
            if (rc != RAD_E_UNSUPPORTED) return rc;
        }
    }
    return as_group == 128 ? mxd_run<true>(a, s, g, mp, as_rs)
                           : mxd_run<false>(a, s, g, mp, as_rs);
}

/* The raw-pointer query libr4d exports beside the row (halo.h): the decode band's
 * widest M, 16 rows a tile times the 8-deep M unroll. */
extern "C" int halo_gemm_mxfp4a8_decode_max_m(void) { return 128; }
