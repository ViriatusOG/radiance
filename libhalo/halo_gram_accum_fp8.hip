/* Derived from libr4d/r4d_gram_accum_fp8.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// H += X^T X, in f32, over an E4M3 activation with a per-(row, group) scale, or over a plain bf16
// one.
//
// Purpose: the device half of GPTQ's Gram matrix `H = sum_t x_t x_t^T` over a calibration corpus,
// for the 4-bit expert format (docs/MOE-W4.md).
//
// Not a GEMM: `gemm_nt` contracts the last axis of both operands, this contracts the first.
// Expressing it as a GEMM would need a transposed copy of X (a 26 MB scratch and a transpose).
//
// E4M3 input: the taps are the MoE block's own quantiser outputs (`had_quant_act_fp8` for gate_up,
// `gated_had_quant_fp8` for down), the bytes the expert GEMM actually contracts, and the only form
// the rotated path stores. A bf16 instantiation reads the plain activation of a bf16-served model;
// everything past the staging is shared.
//
// The gate_up tap is the unsorted activation. Every token is routed to exactly top_k experts, so
// the expert GEMM's input distribution gives H_sorted = top_k * H_token, and GPTQ is invariant to a
// positive scale on H (damping is a fraction of the trace, LDLQ uses ratios). Taking M token rows
// instead of M*top_k gathered ones saves a factor of top_k.
//
// Accumulation is f32, not f64: a block keeps its tile in registers across the M loop and adds to
// global once per launch. The consumer damps H by a percent of its trace, far above this rounding;
// f64 would double 1.3 GB of state.
//
// Returns a negative code rather than throwing, for the reason given in halo_gemm_fp8a8.hip.

#include "halo.h"
#include "halo_args.h"
#include "halo_fp8_wmma.h"

namespace {

// A 64-wide C tile, 256 threads, 4x4 accumulators per thread (16x16 threads x 16 = 64*64). A
// 64-aligned tile lies inside one scale group, so a row's scale is one load per tile; the launcher
// refuses a group the tile does not divide.
constexpr int kTile    = 64;
constexpr int kThreads = 256;
constexpr int kMB      = 32;    // M rows staged a pass; 2 * 32 * 64 * 4 = 16 KiB of LDS
constexpr int kTT      = 4;

/* BF16 reads a plain bf16 activation with no scale: a staging slot is four elements in two dwords.
 * Everything after the staging is shared with the fp8 form. */
template <bool BF16>
__global__ __launch_bounds__(kThreads) void gram_accum_kernel(
    const void* __restrict__ qv, const float* __restrict__ sc, float* __restrict__ h,
    uint32_t M, uint32_t K, uint32_t q_ld, uint32_t s_ld, uint32_t h_ld, uint32_t grp) {

    __shared__ float sa[kMB][kTile];
    __shared__ float sb[kMB][kTile];

    const uint32_t k1_0 = blockIdx.x * (uint32_t)kTile;
    const uint32_t k2_0 = blockIdx.y * (uint32_t)kTile;
    const uint32_t tx = threadIdx.x & 15u, ty = threadIdx.x >> 4;   // 16 x 16
    // Both tiles sit inside one scale group each -- see kTile's note. bf16 has no groups.
    const uint32_t g1 = BF16 ? 0u : k1_0 / grp, g2 = BF16 ? 0u : k2_0 / grp;

    float acc[kTT][kTT];
#pragma unroll
    for (int i = 0; i < kTT; ++i)
#pragma unroll
        for (int j = 0; j < kTT; ++j) acc[i][j] = 0.0f;

    for (uint32_t m0 = 0; m0 < M; m0 += (uint32_t)kMB) {
        __syncthreads();
        // 32 rows x 16 slots = 512 per tile, two per thread. An fp8 slot is a dword of four E4M3
        // codes, decoded by two v_cvt_pk_f32_fp8.
        for (uint32_t idx = threadIdx.x; idx < (uint32_t)(kMB * kTile / 4);
             idx += (uint32_t)kThreads) {
            const uint32_t mi = idx >> 4, cq = (idx & 15u) * 4u;
            const uint32_t m = m0 + mi;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
            if (m < M) {
                const size_t qrow = (size_t)m * q_ld;
                const uint32_t c1 = k1_0 + cq, c2 = k2_0 + cq;
                if constexpr (BF16) {
                    // K is a multiple of four, so a slot never runs past the row.
                    if (c1 < K) {
                        const uint2 w = *(const uint2*)((const uint16_t*)qv + qrow + c1);
                        a0 = bf16_to_f32((uint16_t)w.x); a1 = bf16_to_f32((uint16_t)(w.x >> 16));
                        a2 = bf16_to_f32((uint16_t)w.y); a3 = bf16_to_f32((uint16_t)(w.y >> 16));
                    }
                    if (c2 < K) {
                        const uint2 w = *(const uint2*)((const uint16_t*)qv + qrow + c2);
                        b0 = bf16_to_f32((uint16_t)w.x); b1 = bf16_to_f32((uint16_t)(w.x >> 16));
                        b2 = bf16_to_f32((uint16_t)w.y); b3 = bf16_to_f32((uint16_t)(w.y >> 16));
                    }
                } else {
                    if (c1 < K) {
                        // The row is owned out to K rounded up to four bytes (quant_act_fp8's
                        // contract), so a dword load at c1 < K stays inside the allocation.
                        const uint32_t w = *(const uint32_t*)((const uint8_t*)qv + qrow + c1);
                        const float s = sc[(size_t)m * s_ld + g1];
                        const v2f lo = halo_fp8::e4m3_pk<false>(w), hi = halo_fp8::e4m3_pk<true>(w);
                        a0 = lo.x * s; a1 = lo.y * s; a2 = hi.x * s; a3 = hi.y * s;
                    }
                    if (c2 < K) {
                        const uint32_t w = *(const uint32_t*)((const uint8_t*)qv + qrow + c2);
                        const float s = sc[(size_t)m * s_ld + g2];
                        const v2f lo = halo_fp8::e4m3_pk<false>(w), hi = halo_fp8::e4m3_pk<true>(w);
                        b0 = lo.x * s; b1 = lo.y * s; b2 = hi.x * s; b3 = hi.y * s;
                    }
                }
            }
            // Columns past K stage as zeros and contribute nothing; the store masks them anyway.
            sa[mi][cq + 0] = a0; sa[mi][cq + 1] = a1; sa[mi][cq + 2] = a2; sa[mi][cq + 3] = a3;
            sb[mi][cq + 0] = b0; sb[mi][cq + 1] = b1; sb[mi][cq + 2] = b2; sb[mi][cq + 3] = b3;
        }
        __syncthreads();

        const uint32_t rows = M - m0 < (uint32_t)kMB ? M - m0 : (uint32_t)kMB;
        for (uint32_t r = 0; r < rows; ++r) {
            float u[kTT], v[kTT];
#pragma unroll
            for (int i = 0; i < kTT; ++i) u[i] = sa[r][ty * kTT + i];
#pragma unroll
            for (int j = 0; j < kTT; ++j) v[j] = sb[r][tx * kTT + j];
#pragma unroll
            for (int i = 0; i < kTT; ++i)
#pragma unroll
                for (int j = 0; j < kTT; ++j) acc[i][j] = fmaf(u[i], v[j], acc[i][j]);
        }
    }

    // One block owns one C tile, so the read-modify-write is race-free. The caller zeroes H once
    // and accumulates the whole corpus into it.
#pragma unroll
    for (int i = 0; i < kTT; ++i) {
        const uint32_t k1 = k1_0 + ty * kTT + (uint32_t)i;
        if (k1 >= K) continue;
#pragma unroll
        for (int j = 0; j < kTT; ++j) {
            const uint32_t k2 = k2_0 + tx * kTT + (uint32_t)j;
            if (k2 < K) h[(size_t)k1 * h_ld + k2] += acc[i][j];
        }
    }
}

}  // namespace

extern "C" int halo_gram_accum_fp8_tile(void) { return kTile; }

//   q  [M][K]                E4M3, `q_ld` bytes a row (0 = K), owned out to a multiple of 4
//   s  [M][ceil(K/grp)]      f32,  `s_ld` elements a row (0 = ceil(K/grp))
//   h  [K][K]                f32,  `h_ld` elements a row (0 = K), read and written
extern "C" int halo_gram_accum_fp8(long q, long s, long h, int M, int K, int grp,
                                  int q_ld, int s_ld, int h_ld, long stream) {
    if (!q || !s || !h) return -1;
    if (M <= 0 || K <= 0) return -2;
    if (K % 4) return -3;                       // a staging slot is one dword of four codes
    if (grp <= 0 || (grp % kTile)) return -5;   // a tile must lie inside one scale group
    if (!q_ld) q_ld = K;
    if (q_ld % 4) return -6;                    // ...and each row's dwords must stay aligned
    if (!s_ld) s_ld = (K + grp - 1) / grp;
    if (!h_ld) h_ld = K;
    const uint32_t nt = (uint32_t)((K + kTile - 1) / kTile);
    hipLaunchKernelGGL(gram_accum_kernel<false>, dim3(nt, nt), dim3(kThreads), 0,
                       (hipStream_t)stream, (const void*)q, (const float*)s, (float*)h,
                       (uint32_t)M, (uint32_t)K, (uint32_t)q_ld, (uint32_t)s_ld, (uint32_t)h_ld,
                       (uint32_t)grp);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

//   a  [M][K]                bf16, `a_ld` elements a row (0 = K)
//   h  [K][K]                f32,  `h_ld` elements a row (0 = K), read and written
extern "C" int halo_gram_accum_bf16(long a, long h, int M, int K, int a_ld, int h_ld, long stream) {
    if (!a || !h) return -1;
    if (M <= 0 || K <= 0) return -2;
    if (K % 4) return -3;                       // a staging slot is four elements of one row
    if (!a_ld) a_ld = K;
    if (a_ld % 4 || (a % 8)) return -6;         // ...loaded as two aligned dwords
    if (!h_ld) h_ld = K;
    const uint32_t nt = (uint32_t)((K + kTile - 1) / kTile);
    hipLaunchKernelGGL(gram_accum_kernel<true>, dim3(nt, nt), dim3(kThreads), 0,
                       (hipStream_t)stream, (const void*)a, (const float*)nullptr, (float*)h,
                       (uint32_t)M, (uint32_t)K, (uint32_t)a_ld, 0u, (uint32_t)h_ld, 1u);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// ============================================================== the plugin ABI entry point

enum { GA_Q = 0, GA_S = 1, GA_H = 2 };

extern "C" int halo_rad_gram_accum(const RadArgs* a, RadStream s) {
    const RadTensor* q = halo_opt(a, GA_Q);
    const RadTensor* sc = halo_opt(a, GA_S);
    const RadTensor* h = halo_opt(a, GA_H);
    if (!q || !sc || !h) return HALO_NO(RAD_E_INVAL);
    if (q->dtype != RAD_F8E4M3 || sc->dtype != RAD_F32 || h->dtype != RAD_F32)
        return HALO_NO(RAD_E_DTYPE);
    /* Rows are dim 0 and everything after is the row, as in halo_rad_quant_act_fp8; `sc` and `h`
     * are rank 2 by definition. */
    if (q->rank < 2 || sc->rank != 2 || h->rank != 2) return HALO_NO(RAD_E_SHAPE);
    const long long M = q->shape[0];
    if (M < 1 || halo_numel(q) % M) return HALO_NO(RAD_E_SHAPE);
    const long long K = halo_numel(q) / M;
    const long long grp = halo_geti_or(a, "group", HALO_FP8_BLOCK);
    const long long nkb = (K + grp - 1) / grp;
    if (sc->shape[0] != M || sc->shape[1] != nkb) return HALO_NO(RAD_E_SHAPE);
    if (h->shape[0] != K || h->shape[1] != K) return HALO_NO(RAD_E_SHAPE);
    const long long qp = halo_row_pitch(q), sp = halo_row_pitch(sc), hp = halo_row_pitch(h);
    if (qp < 0 || sp < 0 || hp < 0) return HALO_NO(RAD_E_STRIDE);
    const int e = halo_gram_accum_fp8(halo_addr(q), halo_addr(sc), halo_addr(h), (int)M, (int)K,
                                     (int)grp, (int)qp, (int)sp, (int)hp, (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

extern "C" int halo_rad_gram_accum_bf16(const RadArgs* a, RadStream s) {
    const RadTensor* x = halo_opt(a, GA_Q);
    const RadTensor* h = halo_opt(a, GA_H);
    if (!x || !h) return HALO_NO(RAD_E_INVAL);
    /* A scale operand means the caller believes the input is quantised; refuse it. */
    if (halo_opt(a, GA_S)) return HALO_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || h->dtype != RAD_F32) return HALO_NO(RAD_E_DTYPE);
    if (x->rank < 2 || h->rank != 2) return HALO_NO(RAD_E_SHAPE);
    const long long M = x->shape[0];
    if (M < 1 || halo_numel(x) % M) return HALO_NO(RAD_E_SHAPE);
    const long long K = halo_numel(x) / M;
    if (h->shape[0] != K || h->shape[1] != K) return HALO_NO(RAD_E_SHAPE);
    const long long xp = halo_row_pitch(x), hp = halo_row_pitch(h);
    if (xp < 0 || hp < 0) return HALO_NO(RAD_E_STRIDE);
    const int e = halo_gram_accum_bf16(halo_addr(x), halo_addr(h), (int)M, (int)K, (int)xp, (int)hp,
                                      (long)(uintptr_t)s);
    return e == 0 ? RAD_OK : HALO_NO(RAD_E_SHAPE);
}
