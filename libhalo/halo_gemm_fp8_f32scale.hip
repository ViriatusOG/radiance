/* Derived from libr4d/r4d_fp8_f32scale.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// Canonical E4M3 weights with exact F32 block scales: C[M,N] = A[M,K] @ (W[N,K] * ws)^T, one f32
// scale per (row, 128 of K) on the activation and one per (128 of N, 128 of K) on the weight.
//
// gfx1201 runs this as an fp8 WMMA with the block scale folded on the accumulator. gfx1151 has no
// fp8 matrix instruction, so both operands dequantise to EXACT bf16 (e4m3 has 3 mantissa bits,
// bf16 7, so the widening loses nothing) and the product runs on halo_wmma.h's bf16 WMMA -- the
// recipe the block-scaled MXFP4 GEMMs already use (halo_gemm_mxfp4a8_tiled.hip). A bf16 WMMA with
// an f32 accumulator does what the fp8 one did: each 128-K block is a fresh chain of eight 16-wide
// WMMAs from zero, and the block's sum folds into the running accumulator once, as
// `acc += raw * (a_scale * w_scale)` -- fp8_f32scale's own fold. So the products are the fp8
// kernel's exact products and the unit's own loss is only the WMMA's f32 accumulation order (1 ulp
// on a 16-term dot), where the output is bf16.
//
// One dense form serves every shape. libr4d splits that op into a decode form, two row-tile forms
// and a legacy 16x64 tile, byte-identical and chosen for speed; here the dequantise is the same
// work in each, so the unit ships the row-tile form alone at any M, and the legacy activation pitch
// (K + 8, off a 16-byte multiple) runs through the same kernel -- hence the dword operand reads,
// which keep 4-byte alignment enough. The grouped op is expert-major: a block takes one expert's
// run of the sorted rows, 16 rows at a time, so an expert's weights are read once per run however
// its rows fall in the sort.
//
// Meta and the operand reading are libr4d's, including the two-parity weight tables (even experts
// in `w`/`ws`, odd in the second pair) and their pointer-table form (stride[0] == 0: the operand is
// an array of expert pointers).
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_fp8_wmma.h"
#include "halo_wmma.h"

#include <climits>
#include <cstdint>
#include <cstring>

#define F8S_CHK() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

namespace {

struct Meta {
    long ar, asr, yr, wr, sr, wstep[2], sstep[2];
    int T, N, K, experts, topk, sorted_a;
};

// Expert `e`'s bytes: a stepped plane, or an entry of a pointer table (step 0).
__device__ __forceinline__ const uint8_t* f8_entry(const void* p, long step, int e) {
    return step ? (const uint8_t*)p + step * e : ((const uint8_t* const*)p)[e];
}

/* Sixteen e4m3 bytes to the exact bf16 fragment (halo_gemm_mxfp4a8_nt_m64's mx_deq16_exact): the
 * hardware's fp8 -> f32 conversion, then bf16, which holds every e4m3 value. NOT halo_wmma.h's
 * deq16_e4m3, whose shift-only form comes out low by 2^-120. */
__device__ __forceinline__ v16bf f8s_deq16(const uint32_t* __restrict__ w4) {
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

// The same from sixteen bytes at any 4-byte alignment (the activation's pitch may be K + 8).
__device__ __forceinline__ v16bf f8s_deq16p(const uint8_t* __restrict__ p) {
    uint32_t w4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) w4[i] = ((const uint32_t*)p)[i];
    return f8s_deq16(w4);
}

// ================================================================ the dense form
//
// A block is 4 waves along M, each owning TM = 2 sixteen-row fragments, over TN = 2 sixteen-column
// fragments -- 128 rows by 32 columns. N is a multiple of 128, so a block's columns all lie in one
// weight-scale group and the block's weight scale is a scalar a K block. The weight slab of 128 K
// is dequantised once into LDS and read by all four waves; the activation fragments are read
// straight from global, one 16-byte run a lane a step, and dequantised in registers.
#define F8S_NW   4
#define F8S_TM   2
#define F8S_TN   2
#define F8S_KB   128                                     // K per scale block
#define F8S_STR  (F8S_KB + 8)                            // LDS weight pitch, bf16: 16-byte rows
#define F8S_NTHR (F8S_NW * 32)
#define F8S_BMF  (F8S_NW * F8S_TM * 16)                  // rows a block covers
#define F8S_BNF  (F8S_TN * 16)                           // columns a block covers
#define F8S_NS   (F8S_KB / 16)                           // 16-wide k steps in a scale block
#define F8S_WPT  (F8S_BNF * (F8S_KB / 16) / F8S_NTHR)    // 16-byte weight runs a thread stages

__global__ __launch_bounds__(F8S_NTHR) void halo_fp8_f32scale_dense(
    const uint8_t* __restrict__ a, const float* __restrict__ as, const uint8_t* __restrict__ w,
    const float* __restrict__ ws, uint16_t* __restrict__ y, Meta m) {
    constexpr int BMF = F8S_BMF, BNF = F8S_BNF, NS = F8S_NS, STR = F8S_STR, WPT = F8S_WPT;
    __shared__ __align__(16) uint16_t sW[BNF * STR];
    __shared__ float sR[BMF];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF;
    const int col = halo::b_col(lane);
    // How many of this wave's fragments hold a row below M. Few rows are the decode shapes, and a
    // fragment past M costs its WMMAs for nothing; the test is wave-uniform (halo_gemm_mxfp4a8_tiled
    // skips padding fragments the same way), so the loads and WMMAs of the rest are never issued.
    const int live = min(F8S_TM, max(0, ((m.T + 15) >> 4) - ((m0 >> 4) + wave * F8S_TM)));

    v8f acc[F8S_TM][F8S_TN];
#pragma unroll
    for (int i = 0; i < F8S_TM; ++i)
#pragma unroll
        for (int j = 0; j < F8S_TN; ++j) acc[i][j] = (v8f)(0.f);

    for (int k0 = 0; k0 < m.K; k0 += F8S_KB) {
        // The weight slab, dequantised to exact bf16 on the way into LDS.
#pragma unroll
        for (int t = 0; t < WPT; ++t) {
            const int u = tid + t * F8S_NTHR;
            const int r = u / (F8S_KB / 16), c = (u % (F8S_KB / 16)) * 16;
            union { uint32_t u[8]; v16bf f; } s;
            s.f = f8s_deq16p(w + (size_t)(n0 + r) * m.wr + k0 + c);
            uint16_t* dst = &sW[r * STR + c];
            *(uint2*)(dst)      = make_uint2(s.u[0], s.u[1]);
            *(uint2*)(dst + 4)  = make_uint2(s.u[2], s.u[3]);
            *(uint2*)(dst + 8)  = make_uint2(s.u[4], s.u[5]);
            *(uint2*)(dst + 12) = make_uint2(s.u[6], s.u[7]);
        }
        // Every row's activation scale for this block, then the block's one weight scale.
        const int trow = m0 + tid;
        sR[tid] = trow < m.T ? as[(size_t)trow * m.asr + k0 / F8S_KB] : 0.f;
        const float wsc = ws[(size_t)(n0 / F8S_KB) * m.sr + k0 / F8S_KB];
        lds_barrier();

        // The wave's activation fragments: lane holds row (lane & 15) of the fragment, k = e.
        const uint8_t* arow[F8S_TM];
#pragma unroll
        for (int i = 0; i < F8S_TM; ++i) {
            int r = m0 + wave * F8S_TM * 16 + i * 16 + halo::a_row(lane);
            r = r < m.T ? r : m.T - 1;
            arow[i] = a + (size_t)r * m.ar + k0;
        }
        if (live <= 0) { lds_barrier(); continue; }
        v8f raw[F8S_TM][F8S_TN];
#pragma unroll
        for (int i = 0; i < F8S_TM; ++i)
#pragma unroll
            for (int j = 0; j < F8S_TN; ++j) raw[i][j] = (v8f)(0.f);
#pragma unroll
        for (int step = 0; step < NS; ++step) {
            v16bf af[F8S_TM], wf[F8S_TN];
#pragma unroll
            for (int i = 0; i < F8S_TM; ++i) {
                if (i >= live) break;
                af[i] = f8s_deq16p(arow[i] + step * 16);
            }
#pragma unroll
            for (int j = 0; j < F8S_TN; ++j) {
                const uint16_t* p = &sW[(j * 16 + col) * STR + step * 16];
                union { uint4 w[2]; v16bf f; } u;
                u.w[0] = *(const uint4*)p;
                u.w[1] = *(const uint4*)(p + 8);
                wf[j] = u.f;
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int i = 0; i < F8S_TM; ++i) {
                if (i >= live) break;   // wave-uniform: padding fragments past M
#pragma unroll
                for (int j = 0; j < F8S_TN; ++j) raw[i][j] = halo::mma(af[i], wf[j], raw[i][j]);
            }
        }
        // fp8_f32scale's fold: one rounding of the block sum times the scale pair, one add.
#pragma unroll
        for (int i = 0; i < F8S_TM; ++i) {
            if (i >= live) break;
#pragma unroll
            for (int j = 0; j < F8S_TN; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    acc[i][j][e] +=
                        raw[i][j][e] * (sR[wave * F8S_TM * 16 + i * 16 + halo::d_row(lane, e)] * wsc);
        }
        lds_barrier();
    }

#pragma unroll
    for (int j = 0; j < F8S_TN; ++j) {
        const int nb = n0 + j * 16 + col;
#pragma unroll
        for (int i = 0; i < F8S_TM; ++i)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int r = m0 + wave * F8S_TM * 16 + i * 16 + halo::d_row(lane, e);
                if (r < m.T) y[(size_t)r * m.yr + nb] = halo::f32_to_bf16(acc[i][j][e]);
            }
    }
}

// ================================================================ the grouped form
//
// Expert-major, 4 waves of 16 columns (64 a block), 16 rows of the sorted order a tile: a block
// walks its expert's run [offsets[e], offsets[e+1]) and every tile is one expert's weights, so the
// rows of a tile never disagree about the weight. A row whose `sorted` entry is negative (dropped by
// capacity) is written as zeros, as libr4d writes it, and so are the rows past offsets[experts]
// (the block of row `experts`).
#define F8G_NW  4
#define F8G_BNF (F8G_NW * 16)
#define F8G_NTHR (F8G_NW * 32)

__global__ __launch_bounds__(F8G_NTHR) void halo_fp8_f32scale_experts(
    const uint8_t* __restrict__ a, const float* __restrict__ as, const void* w0, const void* s0,
    const void* w1, const void* s1, const int* __restrict__ sorted, const int* __restrict__ off,
    uint16_t* __restrict__ y, Meta m) {
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int e = blockIdx.y, n0 = blockIdx.x * F8G_BNF;
    if (e == m.experts) {
        // The rows past the last expert's run: zero, never stale arena bytes.
        const int r0 = off[m.experts];
        for (long i = tid; i < (long)(m.T - r0) * F8G_BNF; i += F8G_NTHR) {
            const int c = (int)(i % F8G_BNF);
            if (n0 + c < m.N) y[(size_t)(r0 + i / F8G_BNF) * m.yr + n0 + c] = 0;
        }
        return;
    }
    const int lo = off[e], hi = off[e + 1];
    if (lo >= hi) return;
    const int parity = e & 1;
    const uint8_t* w = f8_entry(parity ? w1 : w0, m.wstep[parity], e / 2);
    const float* ws = (const float*)f8_entry(parity ? s1 : s0, m.sstep[parity], e / 2);
    const int col = n0 + wave * 16 + halo::b_col(lane);
    // A class's columns past its width hold zeros (equal parity widths today, kept for the shape).
    const bool live_col = col < m.N;
    const int cc = live_col ? col : m.N - 1;
    const uint8_t* wrow = w + (size_t)cc * m.wr;
    const float* wsr = ws + (size_t)(cc / F8S_KB) * m.sr;
    const int rowbase = halo::a_row(lane);

    for (int r0 = lo; r0 < hi; r0 += 16) {
        // Each slot e of the accumulator is output row 2e + (lane >> 4) of the tile, and its
        // activation row -- and so its activation scale -- is that row's, not the fragment row's
        // this lane happens to hold.
        int src[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = r0 + halo::d_row(lane, j);
            int s = -1;
            if (r < hi) {
                const int flat = sorted[r];
                s = flat >= 0 ? (m.sorted_a ? r : flat / m.topk) : -1;
                if (s >= m.T) s = -1;
            }
            src[j] = s;
        }
        const int rowA = r0 + rowbase;
        const int flatA = rowA < hi ? sorted[rowA] : -1;
        int srcA = flatA >= 0 ? (m.sorted_a ? rowA : flatA / m.topk) : -1;
        if (srcA >= m.T) srcA = -1;
        const uint8_t* arow = a + (size_t)(srcA >= 0 ? srcA : 0) * m.ar;
        float acc[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = 0.f;
        for (int k0 = 0; k0 < m.K; k0 += F8S_KB) {
            v8f raw = (v8f)(0.f);
#pragma unroll
            for (int step = 0; step < F8S_NS; ++step) {
                const v16bf af = f8s_deq16p(arow + k0 + step * 16);
                const v16bf wf = f8s_deq16p(wrow + k0 + step * 16);
                if (step == 0) __builtin_amdgcn_sched_barrier(0);
                raw = halo::mma(af, wf, raw);
            }
            const int kb = k0 / F8S_KB;
            const float sw = live_col ? wsr[kb] : 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float sa = src[j] >= 0 ? as[(size_t)src[j] * m.asr + kb] : 0.f;
                acc[j] += raw[j] * (sa * sw);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = r0 + halo::d_row(lane, j);
            if (r >= hi) continue;
            const bool keep = src[j] >= 0 && live_col;
            y[(size_t)r * m.yr + col] = halo::f32_to_bf16(keep ? acc[j] : 0.f);
        }
    }
}

// ================================================================ the entry points
static int body(const RadArgs* a, RadStream stream, bool grouped) {
    if (!a || !a->t || a->n_t < (grouped ? 9 : 5)) return RAD_E_INVAL;
    const auto &x = a->t[0], &xs = a->t[1], &w = a->t[2], &ws = a->t[3], &y = a->t[grouped ? 8 : 4];
    const int64_t N = halo_geti_or(a, "N", 0), K = halo_geti_or(a, "K", 0);
    if (N < 128 || N % 128 || K < 128 || K % 128 || N > INT_MAX || K > INT_MAX ||
        halo_geti_or(a, "group", 0) != 128 || halo_geti_or(a, "pb1", 0) > 0)
        return HALO_NO(RAD_E_SHAPE);
    if (x.dtype != RAD_F8E4M3 || xs.dtype != RAD_F32 || w.dtype != RAD_F8E4M3 ||
        ws.dtype != RAD_F32 || y.dtype != RAD_BF16)
        return HALO_NO(RAD_E_DTYPE);
    if (!x.data || !xs.data || !w.data || !ws.data || !y.data || x.rank != 2 || xs.rank != 2 ||
        y.rank != 2 || x.shape[1] != K || xs.shape[0] != x.shape[0] || xs.shape[1] != K / 128 ||
        y.shape[1] != N || x.stride[1] != 1 || xs.stride[1] != 1 || y.stride[1] != 1 ||
        y.shape[0] < 1 || y.shape[0] > INT_MAX)
        return HALO_NO(RAD_E_SHAPE);
    const uint32_t rank = grouped ? 3 : 2;
    if (w.rank != rank || ws.rank != rank || w.shape[rank - 2] != N || w.shape[rank - 1] != K ||
        ws.shape[rank - 2] != N / 128 || ws.shape[rank - 1] != K / 128 || w.stride[rank - 1] != 1 ||
        ws.stride[rank - 1] != 1)
        return HALO_NO(RAD_E_SHAPE);
    Meta m{};
    m.ar = x.stride[0];
    m.asr = xs.stride[0];
    m.yr = y.stride[0];
    m.wr = w.stride[rank - 2];
    m.sr = ws.stride[rank - 2];
    m.T = (int)y.shape[0];
    m.N = (int)N;
    m.K = (int)K;
    if (!grouped) {
        if (x.shape[0] != m.T) return HALO_NO(RAD_E_SHAPE);
        const dim3 grid((unsigned)(N / F8S_BNF), (unsigned)((m.T + F8S_BMF - 1) / F8S_BMF));
        hipLaunchKernelGGL(halo_fp8_f32scale_dense, grid, dim3(F8S_NTHR), 0, (hipStream_t)stream,
                           (const uint8_t*)x.data, (const float*)xs.data, (const uint8_t*)w.data,
                           (const float*)ws.data, (uint16_t*)y.data, m);
        F8S_CHK();
        return RAD_OK;
    }
    m.experts = (int)halo_geti_or(a, "n_expert", 0);
    m.topk = (int)halo_geti_or(a, "top_k", 0);
    const char* order = halo_gets(a, "a_order");
    m.sorted_a = order && !std::strcmp(order, "sorted");
    const auto &wo = a->t[6], &so = a->t[7], &st = a->t[4], &off = a->t[5];
    if (m.experts < 2 || m.topk < 1 || halo_geti_or(a, "N_odd", N) != N ||
        halo_geti_or(a, "K_odd", K) != K || wo.dtype != RAD_F8E4M3 || so.dtype != RAD_F32 ||
        wo.rank != 3 || so.rank != 3 || wo.shape[1] != N || wo.shape[2] != K ||
        so.shape[1] != N / 128 || so.shape[2] != K / 128 || wo.stride[1] != m.wr ||
        so.stride[1] != m.sr || wo.stride[2] != 1 || so.stride[2] != 1 || st.dtype != RAD_I32 ||
        off.dtype != RAD_I32 || st.rank != 1 || off.rank != 1 || st.stride[0] != 1 ||
        off.stride[0] != 1 || st.shape[0] < m.T || off.shape[0] < m.experts + 1 || !wo.data ||
        !so.data || !st.data || !off.data)
        return HALO_NO(RAD_E_SHAPE);
    m.wstep[0] = w.stride[0];
    m.wstep[1] = wo.stride[0];
    m.sstep[0] = ws.stride[0] * 4;
    m.sstep[1] = so.stride[0] * 4;
    if (w.shape[0] < (m.experts + 1) / 2 || wo.shape[0] < m.experts / 2) return HALO_NO(RAD_E_SHAPE);
    const dim3 grid((unsigned)((N + F8G_BNF - 1) / F8G_BNF), (unsigned)m.experts + 1);
    hipLaunchKernelGGL(halo_fp8_f32scale_experts, grid, dim3(F8G_NTHR), 0, (hipStream_t)stream,
                       (const uint8_t*)x.data, (const float*)xs.data, w.data, ws.data, wo.data,
                       so.data, (const int*)st.data, (const int*)off.data, (uint16_t*)y.data, m);
    F8S_CHK();
    return RAD_OK;
}

}  // namespace

extern "C" int halo_rad_fp8_f32scale(const RadArgs* a, RadStream stream) {
    return body(a, stream, false);
}
extern "C" int halo_rad_moe_fp8_f32scale(const RadArgs* a, RadStream stream) {
    return body(a, stream, true);
}

/* Both canonical planes (E4M3 codes, F32 scales per 128x128) are read where they lie, so there is
 * no stored layout to arrange: like libr4d's hook, this validates the encoding and answers
 * RAD_E_UNSUPPORTED, which the caller takes as "the canonical planes, bytes unchanged". */
extern "C" int halo_layout_fp8_f32scale(const RadParam*, int, int operand, const RadEncoding* e,
                                        const int* sel, const RadTensor*, int n, RadLayout*) {
    if (operand != 2 && operand != 3 && operand != 6 && operand != 7) return RAD_E_UNSUPPORTED;
    if (!e || !sel || n != 1 || !rad_enc_is(e, "affine") || e->n_planes != 2 ||
        e->plane[0].dtype != RAD_F8E4M3 || e->plane[1].dtype != RAD_F32 ||
        e->plane[1].block[0] != 128 || e->plane[1].block[1] != 128 ||
        sel[0] != (operand == 3 || operand == 7 ? 1 : 0))
        return RAD_E_DTYPE;
    return RAD_E_UNSUPPORTED;  // Both canonical planes are read without a relayout.
}
