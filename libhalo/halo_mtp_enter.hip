/* Derived from libr4d/r4d_hc_bf16.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_mtp_enter.hip -- mtp_enter_bf16 / mtp_enter_e4m3: the MTP head's entry.
//
//     ne[j]    = e[j] * rsqrt(mean(e^2) + eps) * (w_e[j] + wadd)
//     fe[i]    = fc_e[i] . ne
//     nh[k][j] = h[k*n+j] * rs[k] * (w_h[k*n+j] + wadd)     rs[k] per sub-stream, or one
//                                                           flat rsqrt over the wide row
//                                                           when `ngroup` == 1
//     x[k*n+i] = fc_h[i] . nh[k] + fe[i]
//
// Up to kMtfRows stream rows (a decode step) with E4M3 matrices: the norm passes, both
// products as one split-K launch over the stored codes (halo_hc_f16.h), and a finisher that
// adds the halves in f32. Otherwise: a norm-scale pass, a norm-apply pass, two tiled WMMA
// products over the norms (the fc_e product first, so its result can broadcast-add in
// the finisher), and the broadcast add, the E4M3 matrices decoded to bf16 scratch planes
// first (stored rows carry f32 group scales in the tail, halo_hc_fp8.h).
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

/* ---- per-(row, stream) sums of squares; stream hc is the embedding ----------------------- */
__global__ __launch_bounds__(256) void mtp_ss_kernel(const uint16_t* __restrict__ h,
                                                     const uint16_t* __restrict__ e,
                                                     float* __restrict__ ss,
                                                     long n, int hc, long hrow, long erow) {
    __shared__ float red[8];
    const long r = blockIdx.y;
    const int c = blockIdx.x;
    const uint16_t* src = c < hc ? h + r * hrow + (long)c * n : e + r * erow;
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (long i = tid; i < n; i += 256) {
        const float v = bf16_to_f32(src[i]);
        acc += v * v;
    }
    const int lane = tid & 31, wave = tid >> 5;
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) acc += __shfl_xor(acc, mask);
    if (lane == 0) red[wave] = acc;
    __syncthreads();
    if (tid == 0) {
        float t = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) t += red[i];
        ss[r * (hc + 1) + c] = t;
    }
}

/* ---- apply the norm: one block a (row, stream) ------------------------------------------ */
__global__ __launch_bounds__(256) void mtp_norm_kernel(const uint16_t* __restrict__ h,
                                                       const uint16_t* __restrict__ e,
                                                       const float* __restrict__ wh,
                                                       const float* __restrict__ we,
                                                       const float* __restrict__ ss,
                                                       uint16_t* __restrict__ nh,
                                                       uint16_t* __restrict__ ne,
                                                       long n, int hc, int flat,
                                                       long hrow, long erow, float eps,
                                                       float wadd) {
    __shared__ float sc_sh;
    const long r = blockIdx.y;
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid == 0) {
        float sc;
        if (c == hc) {
            sc = 1.0f / sqrtf(ss[r * (hc + 1) + hc] / (float)n + eps);
        } else if (flat) {
            float tot = 0.0f;
            for (int k = 0; k < hc; ++k) tot += ss[r * (hc + 1) + k];
            sc = 1.0f / sqrtf(tot / (float)(n * hc) + eps);
        } else {
            sc = 1.0f / sqrtf(ss[r * (hc + 1) + c] / (float)n + eps);
        }
        sc_sh = sc;
    }
    __syncthreads();
    const float sc = sc_sh;
    if (c == hc) {
        const uint16_t* src = e + r * erow;
        uint16_t* dst = ne + r * n;
        for (long i = tid; i < n; i += 256)
            dst[i] = f32_to_bf16(bf16_to_f32(src[i]) * sc * (we[i] + wadd));
    } else {
        const uint16_t* src = h + r * hrow + (long)c * n;
        uint16_t* dst = nh + (r * hc + c) * n;
        for (long i = tid; i < n; i += 256)
            dst[i] = f32_to_bf16(bf16_to_f32(src[i]) * sc * (wh[c * n + i] + wadd));
    }
}

/* ---- decode one stored E4M3 plane to bf16: out[r][k] = e4m3(code) * scale[k / G] -------- */
__global__ __launch_bounds__(256) void mtp_decode_f8_kernel(const uint8_t* __restrict__ src,
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

/* ---- x[r][k*n+i] = tmp[r][k*n+i] + fe[r][i] ---------------------------------------------- */
__global__ __launch_bounds__(256) void mtp_add_kernel(const uint16_t* __restrict__ tmp,
                                                      const uint16_t* __restrict__ fe,
                                                      uint16_t* __restrict__ x,
                                                      long rows, long n, int hc, long xrow) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * hc * n) return;
    const long r = idx / (hc * n), rem = idx % (hc * n), i = rem % n;
    x[r * xrow + rem] = f32_to_bf16(bf16_to_f32(tmp[idx]) + bf16_to_f32(fe[r * n + i]));
}

/* ==== the decode path (at most kMtfRows stream rows) =========================================
 * Both products as one split-K launch over the E4M3 rows as stored (halo_hc_f16.h): a block a
 * (16-column tile, K slice, matrix), the slice walked a scale group at a time, one k step a wave,
 * every wave's codes issued first, the group's normalised rows (mtp_norm_kernel's, bf16) staged
 * in LDS as f16. The finisher sums the slices in order and adds the embedding half in f32,
 * rounding x once as the reference does. */
constexpr int kMtfRows = 64;       /* stream rows (rows * hc) the decode path takes */
constexpr int kMtfS = 4;           /* its K slices */
constexpr int kMtfSubMax = 8;      /* scale groups a slice at most */

template <int RT>
__global__ __launch_bounds__(256) void mtf_gemv_kernel(const uint16_t* __restrict__ ne,
                                                       const uint16_t* __restrict__ nh,
                                                       const uint8_t* __restrict__ fce,
                                                       const uint8_t* __restrict__ fch, long ld,
                                                       float* __restrict__ P, long pslice,
                                                       int re, int rh, int n) {
    constexpr int NP = 128 + 8, NB = RT * 16 * NP;
    constexpr int LB = (2 * NB * 2 > 8 * RT * 8 * 32 * 4) ? 2 * NB * 2 : 8 * RT * 8 * 32 * 4;
    __shared__ __attribute__((aligned(16))) unsigned char lds[LB];
    uint16_t* Nb = (uint16_t*)lds;
    const int ct = blockIdx.x, s = blockIdx.y, mat = blockIdx.z;
    const uint16_t* A = mat ? nh : ne;
    const int R = mat ? rh : re;
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, col = lane & 15;
    const int sps = n / 128 / (int)gridDim.y, sub0 = s * sps;
    const uint8_t* wrow = (mat ? fch : fce) + (long)(ct * 16 + col) * ld;
    const float* wsc = (const float*)(wrow + n);

    uint4 cw[kMtfSubMax];
    float cs[kMtfSubMax];
#pragma unroll
    for (int i = 0; i < kMtfSubMax; ++i)
        if (i < sps) {
            cw[i] = *(const uint4*)(wrow + (long)(sub0 + i) * 128 + wave * 16);
            cs[i] = wsc[sub0 + i] * 256.0f;
        }
    v8f acc[RT];
#pragma unroll
    for (int p = 0; p < RT; ++p) acc[p] = (v8f)(0.0f);

    const int sr = tid >> 4, sk = (tid & 15) * 8;
#pragma unroll
    for (int i = 0; i < kMtfSubMax; ++i) {
        if (i >= sps) break;
        uint16_t* nb = Nb + (i & 1) * NB;
        const long k0 = (long)(sub0 + i) * 128 + sk;
#pragma unroll
        for (int p = 0; p < RT; ++p) {
            const int r = sr + 16 * p;
            uint32_t ad[4] = { 0u, 0u, 0u, 0u };
            if (r < R) {
                const uint4 v = *(const uint4*)(A + (long)r * n + k0);
                const uint32_t w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    ad[e] = halo_hcf::pk16(bf16_to_f32((uint16_t)w[e]), bf16_to_f32((uint16_t)(w[e] >> 16)));
            }
            *(uint4*)(nb + r * NP + sk) = make_uint4(ad[0], ad[1], ad[2], ad[3]);
        }
        __syncthreads();
        uint32_t bd[8];
        halo_hcf::codes16(cw[i], bd);
        const v16hf b = halo_hcf::frag(bd);
#pragma unroll
        for (int p = 0; p < RT; ++p) {
            if (p * 16 >= R) break;
            const uint16_t* ap = nb + (p * 16 + col) * NP + wave * 16;
            union { uint4 w[2]; v16hf f; } a;
            a.w[0] = ((const uint4*)ap)[0];
            a.w[1] = ((const uint4*)ap)[1];
            const v8f t = halo::mma(a.f, b, (v8f)(0.0f));
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[p][e] = fmaf(cs[i], t[e], acc[p][e]);
        }
    }

    __syncthreads();
    float* red = (float*)lds;                      /* [8][RT][8][32], over the staged rows */
#pragma unroll
    for (int p = 0; p < RT; ++p)
#pragma unroll
        for (int e = 0; e < 8; ++e) red[((wave * RT + p) * 8 + e) * 32 + lane] = acc[p][e];
    __syncthreads();
    float* Ps = P + (long)(mat * gridDim.y + s) * pslice;
    for (int o = tid; o < RT * 8 * 32; o += 256) {
        const int p = o / 256, e = (o / 32) & 7, ln = o & 31;
        float v = red[((0 * RT + p) * 8 + e) * 32 + ln];
#pragma unroll
        for (int w = 1; w < 8; ++w) v += red[((w * RT + p) * 8 + e) * 32 + ln];
        const int r = p * 16 + 2 * e + (ln >> 4);
        if (r < R) Ps[(long)r * n + ct * 16 + (ln & 15)] = v;
    }
}

/* x[r][k*n+i] = fc_h . nh[r][k] + fc_e . ne[r], each a sum of its slices in order. */
__global__ __launch_bounds__(256) void mtf_finish_kernel(const float* __restrict__ P, long pslice,
                                                         int S, uint16_t* __restrict__ x, long xrow,
                                                         int rows, int n, int hc) {
    const long idx = (long)blockIdx.x * 256 + threadIdx.x;
    if (idx >= (long)rows * hc * n) return;
    const long r = idx / ((long)hc * n), rem = idx - r * hc * n, k = rem / n, i = rem - k * n;
    float vh = 0.0f, ve = 0.0f;
    for (int s = 0; s < S; ++s) {
        vh += P[(long)(S + s) * pslice + (r * hc + k) * n + i];
        ve += P[(long)s * pslice + r * n + i];
    }
    x[r * xrow + rem] = f32_to_bf16(vh + ve);
}

/* ---- the tiled product (clone of halo_hc_read's tile, plain epilogue) -------------------- */
#define MTP_BK 64
#define MTP_PITCH (MTP_BK + 8)
#define MTP_WM 2
#define MTP_WN 4

template <int TM, int TN>
__global__ __launch_bounds__(256) void mtp_gemm_kernel(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    __hip_bfloat16* __restrict__ C, int M, int K, int N,
    int lda, int ldw, int ldc) {
    constexpr int BM = MTP_WM * TM * 16, BN = MTP_WN * TN * 16;
    constexpr int CPR = MTP_BK / 8;
    constexpr int AL = BM * CPR / 256, WL = BN * CPR / 256;
    __shared__ __attribute__((aligned(16))) unsigned short sA[BM * MTP_PITCH];
    __shared__ __attribute__((aligned(16))) unsigned short sW[BN * MTP_PITCH];

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave / MTP_WN, wn = wave % MTP_WN;
    const int m0 = (int)blockIdx.y * BM, n0 = (int)blockIdx.x * BN;

    unsigned aoff[AL], woff[WL];
    unsigned short* adst[AL];
    unsigned short* wdst[WL];
#pragma unroll
    for (int i = 0; i < AL; ++i) {
        const int c_ = tid + 256 * i, rr = c_ / CPR, e = (c_ % CPR) * 8;
        const int gr = m0 + rr < M ? m0 + rr : M - 1;
        aoff[i] = (unsigned)gr * (unsigned)lda + (unsigned)e;
        adst[i] = sA + rr * MTP_PITCH + e;
    }
#pragma unroll
    for (int i = 0; i < WL; ++i) {
        const int c_ = tid + 256 * i, rr = c_ / CPR, e = (c_ % CPR) * 8;
        const int gr = n0 + rr < N ? n0 + rr : N - 1;
        woff[i] = (unsigned)gr * (unsigned)ldw + (unsigned)e;
        wdst[i] = sW + rr * MTP_PITCH + e;
    }

    const unsigned short* fa = sA + (wm * TM * 16 + (lane & 15)) * MTP_PITCH;
    const unsigned short* fw = sW + (wn * TN * 16 + (lane & 15)) * MTP_PITCH;

    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (v8f)(0.f);

    const int nk = (K + MTP_BK - 1) / MTP_BK;
    auto load_tile = [&](int kt, halo_u32x4* ra, halo_u32x4* rw) {
        const unsigned k0 = (unsigned)kt * MTP_BK;
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
        for (int s = 0; s < MTP_BK / 16; ++s) {
            v16bf af[TM], bf[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) af[i] = fragP(fa + i * 16 * MTP_PITCH + s * 16);
#pragma unroll
            for (int j = 0; j < TN; ++j) bf[j] = fragP(fw + j * 16 * MTP_PITCH + s * 16);
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
                if (n_ < N && m < M)
                    C[(size_t)m * ldc + n_] = (__hip_bfloat16)acc[i][j][e];
            }
        }
}

int mtp_gemm_launch(const unsigned short* A, const unsigned short* W, __hip_bfloat16* C,
                    int M, int K, int N, int lda, int ldw, int ldc, hipStream_t st) {
    const long long g = (long long)((N + 127) / 128) * ((M + 127) / 128);
    const int bt = g >= 32 ? 128 : 64;
    dim3 grid((N + bt - 1) / bt, (M + bt - 1) / bt), block(256);
#define MTP_LAUNCH(TM, TN) hipLaunchKernelGGL((mtp_gemm_kernel<TM, TN>), grid, block, 0, st, \
    A, W, C, M, K, N, lda, ldw, ldc)
    if (bt == 128) MTP_LAUNCH(4, 2); else MTP_LAUNCH(2, 1);
#undef MTP_LAUNCH
    return (int)hipGetLastError();
}

enum { MTE_H = 0, MTE_E, MTE_WH, MTE_WE, MTE_FCH, MTE_FCE, MTE_X };

struct MtpPlan {
    long long rows, n, hc, HN;
    bool f8;
    size_t off_nh, off_ne, off_fe, off_tmp, off_w, off_p, bytes;
};

int mtp_plan(const RadArgs* a, const RadTensor* h, const RadTensor* fch, MtpPlan* p) {
    p->n = halo_geti_or(a, "n", 0);
    p->hc = halo_geti_or(a, "hc", 0);
    if (p->n <= 0 || p->hc <= 1 || p->hc > 8) return RAD_E_SHAPE;
    p->HN = p->hc * p->n;
    p->rows = 0;
    if (h && p->HN > 0) p->rows = halo_numel(h) / p->HN;
    const long long band = halo_geti_or(a, "M", 0);
    if (band > p->rows) p->rows = band;
    if (p->rows <= 0) return RAD_E_SHAPE;
    /* The builder sizes scratch with no tensors: the row's `fc` parameter names the format then. */
    const char* fc = halo_gets(a, "fc");
    const bool known = fch && (fch->dtype == RAD_F8E4M3 || fch->dtype == RAD_BF16);
    p->f8 = known ? fch->dtype == RAD_F8E4M3 : (fc && !std::strcmp(fc, "e4m3"));
    const size_t align = 256;
    auto up = [&](size_t v) { return (v + align - 1) & ~(align - 1); };
    size_t off = 0;
    off = up(off + (size_t)p->rows * (p->hc + 1) * 4);            /* ss */
    p->off_nh = off; off = up(off + (size_t)p->rows * p->HN * 2);
    p->off_ne = off; off = up(off + (size_t)p->rows * p->n * 2);
    p->off_fe = off; off = up(off + (size_t)p->rows * p->n * 2);
    p->off_tmp = off; off = up(off + (size_t)p->rows * p->HN * 2);
    p->off_w = off;
    if (p->f8) off = up(off + (size_t)2 * p->n * p->n * 2);
    p->off_p = off;
    if (p->f8) off = up(off + (size_t)2 * kMtfS * std::min<long long>(p->rows * p->hc, kMtfRows) * p->n * 4);
    p->bytes = off;
    return RAD_OK;
}

}  // namespace

extern "C" int64_t halo_rad_mtp_enter_scratch(const RadArgs* a) {
    if (!a) return 0;
    const RadTensor* h = a->n_t > MTE_H ? &a->t[MTE_H] : nullptr;
    const RadTensor* fch = a->n_t > MTE_FCH ? &a->t[MTE_FCH] : nullptr;
    MtpPlan p;
    if (mtp_plan(a, h, fch, &p) != RAD_OK) return 0;
    return (int64_t)p.bytes;
}

extern "C" int halo_rad_mtp_enter(const RadArgs* a, RadStream s) {
    const RadTensor* h   = halo_opt(a, MTE_H);
    const RadTensor* e   = halo_opt(a, MTE_E);
    const RadTensor* wh  = halo_opt(a, MTE_WH);
    const RadTensor* we  = halo_opt(a, MTE_WE);
    const RadTensor* fch = halo_opt(a, MTE_FCH);
    const RadTensor* fce = halo_opt(a, MTE_FCE);
    const RadTensor* x   = halo_opt(a, MTE_X);
    if (!h || !e || !wh || !we || !fch || !fce || !x) return HALO_NO(RAD_E_INVAL);
    if (h->dtype != RAD_BF16 || e->dtype != RAD_BF16 || x->dtype != RAD_BF16)
        return HALO_NO(RAD_E_DTYPE);
    if (wh->dtype != RAD_F32 || we->dtype != RAD_F32) return HALO_NO(RAD_E_DTYPE);
    const bool f8 = fch->dtype == RAD_F8E4M3;
    if (f8) { if (fce->dtype != RAD_F8E4M3) return HALO_NO(RAD_E_DTYPE); }
    else if (fch->dtype != RAD_BF16 || fce->dtype != RAD_BF16) return HALO_NO(RAD_E_DTYPE);

    MtpPlan p;
    int rc = mtp_plan(a, h, fch, &p);
    if (rc != RAD_OK) return HALO_NO(rc);
    const long long n = p.n, hc = p.hc, HN = p.HN;

    long long hrows_ = 0, erows_ = 0, xrows_ = 0;
    const long long hrow = halo_rows_pitch(h, HN, &hrows_);
    const long long erow = halo_rows_pitch(e, n, &erows_);
    const long long xrow = halo_rows_pitch(x, HN, &xrows_);
    if (hrow < 0 || erow < 0 || xrow < 0) return HALO_NO(RAD_E_STRIDE);
    const long long rows = hrows_;
    if (erows_ < rows || xrows_ < rows) return HALO_NO(RAD_E_STRIDE);
    if (halo_numel(wh) < HN || halo_numel(we) < n) return HALO_NO(RAD_E_SHAPE);
    if (f8) {
        if (!halo_hc8::group_ok(n, halo_hc8::kDownGroup)) return HALO_NO(RAD_E_SHAPE);
        if (fch->rank != 2 || fch->shape[0] != n || fch->shape[1] != n ||
            fce->rank != 2 || fce->shape[0] != n || fce->shape[1] != n)
            return HALO_NO(RAD_E_SHAPE);
    } else if (halo_numel(fch) < n * n || halo_numel(fce) < n * n) {
        return HALO_NO(RAD_E_SHAPE);
    }
    if (!a->scratch || (size_t)a->scratch_bytes < p.bytes) return HALO_NO(RAD_E_SCRATCH);

    const long long ngroup = halo_geti_or(a, "ngroup", hc);
    if (ngroup != 1 && ngroup != hc) return HALO_NO(RAD_E_SHAPE);
    const float eps  = (float)halo_getf(a, "eps", 1e-6);
    const float wadd = (float)halo_getf(a, "wadd", 0.0);

    hipStream_t st = (hipStream_t)s;
    unsigned char* scr = (unsigned char*)a->scratch;
    float* ssbuf = (float*)(scr);
    uint16_t* nh = (uint16_t*)(scr + p.off_nh);
    uint16_t* ne = (uint16_t*)(scr + p.off_ne);
    uint16_t* fe = (uint16_t*)(scr + p.off_fe);
    uint16_t* tmp = (uint16_t*)(scr + p.off_tmp);
    const uint16_t* Wh = (const uint16_t*)halo_p(fch);
    const uint16_t* We = (const uint16_t*)halo_p(fce);
    const bool dec = f8 && rows * hc <= kMtfRows && (n / 128) % kMtfS == 0 &&
                     n / 128 / kMtfS <= kMtfSubMax &&
                     ((uintptr_t)halo_p(fch) & 15u) == 0 && ((uintptr_t)halo_p(fce) & 15u) == 0;
    if (dec) {
        mtp_ss_kernel<<<dim3((unsigned)(hc + 1), (unsigned)rows), 256, 0, st>>>(
            (const uint16_t*)halo_p(h), (const uint16_t*)halo_p(e), ssbuf, n, (int)hc, hrow, erow);
        mtp_norm_kernel<<<dim3((unsigned)(hc + 1), (unsigned)rows), 256, 0, st>>>(
            (const uint16_t*)halo_p(h), (const uint16_t*)halo_p(e),
            (const float*)halo_p(wh), (const float*)halo_p(we), ssbuf, nh, ne,
            n, (int)hc, ngroup == 1 ? 1 : 0, hrow, erow, eps, wadd);
        float* P = (float*)(scr + p.off_p);
        const long pslice = (long)std::min<long long>(p.rows * hc, kMtfRows) * n;
        const long ld = halo_hc8::row_bytes(n, halo_hc8::kDownGroup);
        const dim3 grid((unsigned)(n / 16), (unsigned)kMtfS, 2u);
#define MTF(RT) mtf_gemv_kernel<RT><<<grid, 256, 0, st>>>(ne, nh, (const uint8_t*)halo_p(fce), \
            (const uint8_t*)halo_p(fch), ld, P, pslice, (int)rows, (int)(rows * hc), (int)n)
        switch ((rows * hc + 15) / 16) {
            case 1: MTF(1); break;
            case 2: MTF(2); break;
            case 3: MTF(3); break;
            default: MTF(4); break;
        }
#undef MTF
        const long tot = rows * HN;
        mtf_finish_kernel<<<dim3((unsigned)((tot + 255) / 256)), 256, 0, st>>>(
            P, pslice, kMtfS, (uint16_t*)halo_p(x), xrow, (int)rows, (int)n, (int)hc);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    if (f8) {
        uint16_t* Whb = (uint16_t*)(scr + p.off_w);
        uint16_t* Web = Whb + (size_t)n * n;
        const long ld = halo_hc8::row_bytes(n, halo_hc8::kDownGroup);
        auto dec = [&](const uint8_t* srcp, uint16_t* dstp) {
            const long chunks = n * (n / 16);
            mtp_decode_f8_kernel<<<dim3((unsigned)((chunks + 255) / 256)), 256, 0, st>>>(
                srcp, dstp, n, n, halo_hc8::kDownGroup, ld);
        };
        dec((const uint8_t*)halo_p(fch), Whb);
        dec((const uint8_t*)halo_p(fce), Web);
        if (hipGetLastError() != hipSuccess) return HALO_NO(RAD_E_DEVICE);
        Wh = Whb;
        We = Web;
    }

    mtp_ss_kernel<<<dim3((unsigned)(hc + 1), (unsigned)rows), 256, 0, st>>>(
        (const uint16_t*)halo_p(h), (const uint16_t*)halo_p(e), ssbuf, n, (int)hc, hrow, erow);
    if (hipGetLastError() != hipSuccess) return HALO_NO(RAD_E_DEVICE);
    mtp_norm_kernel<<<dim3((unsigned)(hc + 1), (unsigned)rows), 256, 0, st>>>(
        (const uint16_t*)halo_p(h), (const uint16_t*)halo_p(e),
        (const float*)halo_p(wh), (const float*)halo_p(we), ssbuf, nh, ne,
        n, (int)hc, ngroup == 1 ? 1 : 0, hrow, erow, eps, wadd);
    if (hipGetLastError() != hipSuccess) return HALO_NO(RAD_E_DEVICE);

    if (mtp_gemm_launch(ne, We, (__hip_bfloat16*)fe, (int)rows, (int)n, (int)n,
                        (int)n, (int)n, (int)n, st) != 0)
        return HALO_NO(RAD_E_DEVICE);
    if (mtp_gemm_launch(nh, Wh, (__hip_bfloat16*)tmp, (int)(rows * hc), (int)n, (int)n,
                        (int)n, (int)n, (int)n, st) != 0)
        return HALO_NO(RAD_E_DEVICE);

    const long tot = rows * HN;
    mtp_add_kernel<<<dim3((unsigned)((tot + 255) / 256)), 256, 0, st>>>(
        tmp, fe, (uint16_t*)halo_p(x), rows, n, (int)hc, xrow);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
