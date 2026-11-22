/* moe_gemm_w4a8i8's single-row-run decode form for the w4nl64a8i8h plane: a GEMV.
 *
 * At decode every expert's run is a row or a few (a row a token, T <= 4 top_k here), so the
 * (run tile, column group) WMMA form of halo_moe_gemm_q.hip stages a 16-row tile for one live
 * row. Here a block is (128 output columns, first sorted row of an expert's run): eight waves,
 * a wave a 16-column tile walking K a 128-wide block at a time with the next block's codes and
 * scales in flight. Lane l holds K 4l..4l+3 of each of the tile's sixteen rows: its nibbles
 * become f16 levels by a byte permute of the w4nl table and dot the run's activation (staged in
 * LDS once a block) by v_dot2_f32_f16; the sixteen columns' sums close by halving across the
 * lanes, a 64-wide scale group's apart from the other's.
 *
 * Every level is a multiple of 1/8 and every code |a| <= 127, so a scale group's sum is exact in
 * f32 whatever the order. The WMMA form stages the operands at 2^-8 each (kMoeF16Fix); here the
 * product is taken at 1 and scaled by 2^-16, and folded as that form folds (raw * ws per group,
 * summed, times the activation's scale, added in K order). On a Flash-Next layer at one token
 * the two agree in all but one of 23 040 outputs, which differs by one bf16 ulp. */
#include "halo.h"
#include "halo_args.h"
#include "halo_chain.h"
#include "halo_emit.h"
#include "halo_fp8_wmma.h"
#include "halo_wmma.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>

namespace {

constexpr uint32_t kWave = 32, kWaves = 8, kRows = 4, kMaxK = 4096;

struct Route {
    const int8_t* A;
    const float* As;
    const int32_t* sorted;
    const int32_t* off;
    uint16_t* Y;
    uint32_t T, N, n_expert, top_k, a_ld, as_ld, y_ld, a_group, a_sorted;
};
/* One class of experts: entry e of the weight and scale tables (a pointer array where the
 * stride is 0, else a stacked plane), `w_ld` bytes a weight row, `s_ld` an E4M3 scale a row and
 * 64 of K. */
struct Cls {
    const void* const* w;
    const void* const* s;
    uint64_t w_s0, s_s0;
    uint32_t n, k, w_ld, s_ld;
};
/* The w4nl magnitudes as f16: high bytes in v[0..1], low bytes in v[2..3], a nibble's low three
 * bits the index and its bit 3 the sign. */
struct Lv { uint32_t v[4]; };

__device__ __forceinline__ const uint8_t* tab_at(const void* const* t, uint64_t s0, uint32_t i) {
    return s0 == 0 ? (const uint8_t*)t[i] : (const uint8_t*)t + (size_t)i * s0;
}
/* Field by field: indexing a Cls[2] by the expert's parity would put both in scratch. */
__device__ __forceinline__ Cls pick(const Cls& c0, const Cls& c1, uint32_t ex) {
    const bool o = (ex & 1u) != 0u;
    Cls c;
    c.w = o ? c1.w : c0.w;
    c.s = o ? c1.s : c0.s;
    c.w_s0 = o ? c1.w_s0 : c0.w_s0;
    c.s_s0 = o ? c1.s_s0 : c0.s_s0;
    c.n = o ? c1.n : c0.n;
    c.k = o ? c1.k : c0.k;
    c.w_ld = o ? c1.w_ld : c0.w_ld;
    c.s_ld = o ? c1.s_ld : c0.s_ld;
    return c;
}
__device__ __forceinline__ uint32_t expert_of(const int32_t* off, uint32_t n_expert, uint32_t i) {
    uint32_t lo = 0, hi = n_expert;
    while (hi - lo > 1u) {
        const uint32_t mid = (lo + hi) / 2u;
        if ((uint32_t)off[mid] <= i) lo = mid; else hi = mid;
    }
    return lo;
}

/* The lane-xor exchange in registers: DPP within a row of 16, permlanex16 across the rows. */
template <uint32_t M>
__device__ __forceinline__ float xor_lane(float x) {
    const int i = __float_as_int(x);
    if constexpr (M == 16u)
        return __int_as_float(__builtin_amdgcn_permlanex16(i, i, 0x76543210u, 0xFEDCBA98u, false, false));
    else
        return __int_as_float(__builtin_amdgcn_update_dpp(i, i, 0x160 | (int)M, 0xF, 0xF, false));
}
/* Halve the lanes' N values by xor mask M: each lane keeps the half its bit M selects. */
template <uint32_t N, uint32_t M>
__device__ __forceinline__ void halve(float (&v)[16], uint32_t lane) {
    const bool hi = (lane & M) != 0u;
#pragma unroll
    for (uint32_t i = 0; i < N / 2u; ++i) {
        const float a = v[i], b = v[i + N / 2u];
        v[i] = (hi ? b : a) + xor_lane<M>(hi ? a : b);
    }
}

typedef _Float16 h2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ void levels(uint32_t nib, const Lv& t, h2& l01, h2& l23) {
    const uint32_t m = nib & 0x07070707u;
    const uint32_t hb = __builtin_amdgcn_perm(t.v[1], t.v[0], m) | ((nib << 4) & 0x80808080u);
    const uint32_t lb = __builtin_amdgcn_perm(t.v[3], t.v[2], m);
    l01 = __builtin_bit_cast(h2, __builtin_amdgcn_perm(hb, lb, 0x05010400u));
    l23 = __builtin_bit_cast(h2, __builtin_amdgcn_perm(hb, lb, 0x07030602u));
}

/* Block (x, y): columns 128x.., sorted row y, which serves only where it opens its expert's
 * run; the block at y == live zeroes the rows past the live ones. */
__device__ __forceinline__ void w4nl_gemv_body(const Route& R, const Cls& c0, const Cls& c1,
                                               const Lv& lv) {
    __shared__ __attribute__((aligned(16))) int8_t sa[kRows * kMaxK];
    __shared__ float ss[kRows * (kMaxK / 128)];
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1u);
    const uint32_t y = blockIdx.y, n0 = blockIdx.x * (16u * kWaves) + (tid / kWave) * 16u;
    const uint32_t live = (uint32_t)R.off[R.n_expert];
    if (y >= live) {
        if (y == live && n0 < R.N)
            for (uint32_t i = live + (lane >> 4); i < R.T; i += 2u) R.Y[(size_t)i * R.y_ld + n0 + (lane & 15u)] = 0;
        return;
    }
    const uint32_t ex = expert_of(R.off, R.n_expert, y);
    if ((uint32_t)R.off[ex] != y) return;
    const uint32_t r1 = (uint32_t)R.off[ex + 1u], cnt = r1 - y;
    const Cls c = pick(c0, c1, ex);
    const uint32_t K = c.k, nsc = (K + R.a_group - 1u) / R.a_group;
    for (uint32_t i = tid; i < cnt * (K / 16u); i += kWaves * kWave) {
        const uint32_t r = i / (K / 16u), o = i % (K / 16u);
        const int32_t f = R.sorted[y + r];
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (f >= 0)
            v = *(const uint4*)(R.A + (size_t)(R.a_sorted ? y + r : (uint32_t)f / R.top_k) * R.a_ld + o * 16u);
        *(uint4*)(sa + r * K + o * 16u) = v;
    }
    for (uint32_t i = tid; i < cnt * nsc; i += kWaves * kWave) {
        const uint32_t r = i / nsc, j = i % nsc;
        const int32_t f = R.sorted[y + r];
        ss[i] = f >= 0 ? R.As[(size_t)(R.a_sorted ? y + r : (uint32_t)f / R.top_k) * R.as_ld + j] : 0.f;
    }
    __syncthreads();
    if (n0 >= R.N) return;
    if (n0 >= c.n) {   /* a narrower class: its columns past n are zero */
        for (uint32_t i = y + (lane >> 4); i < r1; i += 2u) R.Y[(size_t)i * R.y_ld + n0 + (lane & 15u)] = 0;
        return;
    }
    bool vr[kRows];
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) vr[r] = r < cnt && R.sorted[y + r] >= 0;
    /* After four halving steps lanes 0-15 hold the first scale group's sums and lane l, l ^ 16
     * one column's pair. */
    const uint32_t col = ((lane >> 3) & 1u) * 8u + ((lane >> 2) & 1u) * 4u + ((lane >> 1) & 1u) * 2u + (lane & 1u);
    const uint8_t* w = tab_at(c.w, c.w_s0, ex >> 1);
    const uint8_t* s = tab_at(c.s, c.s_s0, ex >> 1);
    const uint8_t* wl = w + (size_t)n0 * c.w_ld + (lane >> 1) * 4u;
    const uint8_t* sl = s + (size_t)(n0 + col) * c.s_ld;
    const uint32_t nkb = K / 128u;
    auto fetch = [&](uint32_t kb, uint32_t (&d)[16]) {
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q) d[q] = *(const uint32_t*)(wl + (size_t)q * c.w_ld + kb * 64u);
    };
    /* the scales' E4M3 times kW4G64Scale * kMoeF16Fix (2^-13 * 2^16), as the WMMA form folds */
    auto scale = [&](uint32_t kb, float& w0, float& w1) {
        w0 = halo_fp8::e4m3(sl[2u * kb]) * 8.0f;
        w1 = halo_fp8::e4m3(sl[2u * kb + 1u]) * 8.0f;
    };
    float acc[kRows];
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) acc[r] = 0.f;
    uint32_t cur[16];
    fetch(0, cur);
    float ws0, ws1;
    scale(0, ws0, ws1);
    for (uint32_t kb = 0; kb < nkb; ++kb) {
        const uint32_t kn = kb + 1u < nkb ? kb + 1u : kb;
        uint32_t nxt[16];
        fetch(kn, nxt);
        float wn0, wn1;
        scale(kn, wn0, wn1);
        h2 l01[16], l23[16];
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q)
            levels((lane & 1u) ? (cur[q] >> 4) & 0x0F0F0F0Fu : cur[q] & 0x0F0F0F0Fu, lv, l01[q], l23[q]);
#pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
            if (!vr[r]) continue;
            const int32_t av = *(const int32_t*)(sa + r * K + kb * 128u + 4u * lane);
            const h2 a01 = { (_Float16)(float)((av << 24) >> 24), (_Float16)(float)((av << 16) >> 24) };
            const h2 a23 = { (_Float16)(float)((av << 8) >> 24), (_Float16)(float)(av >> 24) };
            float v[16];
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q)
                v[q] = __builtin_amdgcn_fdot2(l23[q], a23, __builtin_amdgcn_fdot2(l01[q], a01, 0.0f, false), false);
            halve<16, 8>(v, lane); halve<8, 4>(v, lane); halve<4, 2>(v, lane); halve<2, 1>(v, lane);
            const float o = xor_lane<16>(v[0]);
            const float raw0 = ((lane & 16u) ? o : v[0]) * 0x1p-16f, raw1 = ((lane & 16u) ? v[0] : o) * 0x1p-16f;
            float t = raw0 * ws0;
            t += raw1 * ws1;
            acc[r] += t * ss[r * nsc + kb * 128u / R.a_group];
        }
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q) cur[q] = nxt[q];
        ws0 = wn0;
        ws1 = wn1;
    }
    if (lane >= 16u) return;
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r)
        if (r < cnt) R.Y[(size_t)(y + r) * R.y_ld + n0 + col] = halo::f32_to_bf16(acc[r]);
}

template <class CH>
__global__ __launch_bounds__(kWaves * kWave) void moe_w4nl_gemv_kernel(Route R, Cls c0, Cls c1, Lv lv) {
    if (CH::on) halo_chain_wait();
    w4nl_gemv_body(R, c0, c1, lv);
    if (CH::on) halo_chain_done();
}

/* A weight-table operand: rank 3 [entries, rows, cols]; `bits` the element width. */
int tab_of(const RadTensor* t, uint32_t dtype, int bits, long long cnt, long long rows, long long cols,
           const void* const** tp, uint64_t* s0, long long* ld) {
    if (!t || t->rank != 3) return RAD_E_UNSUPPORTED;
    if (cnt > 0 && !t->data) return RAD_E_UNSUPPORTED;
    if (t->dtype != dtype) return RAD_E_UNSUPPORTED;
    if (t->shape[0] < cnt || t->shape[1] < rows || t->shape[2] < cols) return RAD_E_UNSUPPORTED;
    if (t->stride[2] != 1) return RAD_E_UNSUPPORTED;
    *tp = (const void* const*)t->data;
    *s0 = (uint64_t)((t->stride[0] * (int64_t)bits) / 8);
    *ld = t->stride[1];
    return RAD_OK;
}

/* The f16 magnitudes of the w4nl table (halo_args.h's kW4nlTable, its second half the first
 * negated), each a multiple of 1/8 so that the sums stay exact. */
bool lv_of(Lv* out) {
    Lv t{};
    for (int i = 0; i < 8; ++i) {
        const float m = kW4nlTable[i];
        if (kW4nlTable[i + 8] != -m || !(m > 0.0f) || m * 8.0f != (float)(int)(m * 8.0f)) return false;
        const _Float16 h = (_Float16)m;
        if ((float)h != m) return false;
        const uint16_t b = __builtin_bit_cast(uint16_t, h);
        t.v[i / 4] |= (uint32_t)(b >> 8) << (8 * (i % 4));
        t.v[2 + i / 4] |= (uint32_t)(b & 0xFFu) << (8 * (i % 4));
    }
    *out = t;
    return true;
}

}  // namespace

/* The GEMV where it serves the call (w4nl64a8i8h, two classes, one part, every expert's run at
 * most four rows, K within the staging), else RAD_E_UNSUPPORTED with nothing launched. Operands
 * as moe_gemm_q's: [0] a [1] a_scale [2] w [3] w_scale [4] sorted [5] off [6] w_odd
 * [7] w_odd_scale [8] y. */
int halo_moe_w4nl_gemv(const RadArgs* a, RadStream stream, bool chain) {
    const char* wdt = halo_gets(a, "dtype");
    if (!wdt || std::strcmp(wdt, "w4nl64a8i8h")) return RAD_E_UNSUPPORTED;
    const RadTensor *A = halo_opt(a, 0), *As = halo_opt(a, 1), *sorted = halo_opt(a, 4),
                    *off = halo_opt(a, 5), *Y = halo_opt(a, 8);
    const RadTensor* Wc[2] = { halo_opt(a, 2), halo_opt(a, 6) };
    const RadTensor* Wsc[2] = { halo_opt(a, 3), halo_opt(a, 7) };
    if (!A || !As || !sorted || !off || !Y || !Wc[0] || !Wsc[0] || !Wc[1] || !Wsc[1]) return RAD_E_UNSUPPORTED;
    const long long N0 = halo_geti_or(a, "N", 0), K0 = halo_geti_or(a, "K", 0);
    const long long Nc[2] = { N0, halo_geti_or(a, "N_odd", N0) };
    const long long Kc[2] = { K0, halo_geti_or(a, "K_odd", K0) };
    const long long n_expert = halo_geti_or(a, "n_expert", 0), top_k = halo_geti_or(a, "top_k", 0);
    const long long a_group = halo_geti_or(a, "a_group", 128);
    if (halo_geti_or(a, "parts", 1) != 1 || halo_geti_or(a, "group", 128) != 128) return RAD_E_UNSUPPORTED;
    if (n_expert <= 0 || top_k <= 0 || (a_group != 128 && a_group != 256)) return RAD_E_UNSUPPORTED;
    long long N = 0, K = 0;
    for (int q = 0; q < 2; ++q) {
        if (Nc[q] <= 0 || Kc[q] <= 0 || Nc[q] % 16 || Kc[q] % 128 || Kc[q] % a_group) return RAD_E_UNSUPPORTED;
        N = Nc[q] > N ? Nc[q] : N;
        K = Kc[q] > K ? Kc[q] : K;
    }
    if (K > (long long)kMaxK) return RAD_E_UNSUPPORTED;
    if (A->dtype != RAD_I8 || As->dtype != RAD_F32 || Y->dtype != RAD_BF16 || sorted->dtype != RAD_I32 ||
        off->dtype != RAD_I32)
        return RAD_E_UNSUPPORTED;
    long long T = 0, M = 0, Ms = 0;
    const long long y_ld = halo_rows_pitch(Y, N, &T);
    const long long a_ld = halo_rows_pitch(A, K, &M);
    const long long as_ld = halo_rows_pitch(As, K / a_group, &Ms);
    if (y_ld < 0 || T <= 0 || a_ld < 0 || a_ld % 16 || M <= 0 || as_ld < 0 || Ms < M) return RAD_E_UNSUPPORTED;
    /* every expert's run at most kRows rows: one a token */
    if (T > (long long)kRows * top_k) return RAD_E_UNSUPPORTED;
    if (halo_numel(sorted) < T || halo_numel(off) < n_expert + 1) return RAD_E_UNSUPPORTED;
    const char* ord = halo_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return RAD_E_UNSUPPORTED;
    if (M < (a_sorted ? T : (T + top_k - 1) / top_k) || !halo_aligned(A->data, 16)) return RAD_E_UNSUPPORTED;
    Lv lv;
    if (!lv_of(&lv)) return RAD_E_UNSUPPORTED;
    Cls cl[2]{};
    for (int q = 0; q < 2; ++q) {
        const long long cnt = (n_expert + 1 - q) / 2;
        long long wld = 0, sld = 0;
        if (tab_of(Wc[q], (uint32_t)HALO_DT_MOEW4, 4, cnt, Nc[q], Kc[q], &cl[q].w, &cl[q].w_s0, &wld) != RAD_OK ||
            tab_of(Wsc[q], RAD_F8E4M3, 8, cnt, Nc[q], Kc[q] / 64, &cl[q].s, &cl[q].s_s0, &sld) != RAD_OK ||
            wld % 2)
            return RAD_E_UNSUPPORTED;
        cl[q].w_ld = (uint32_t)(wld / 2);
        cl[q].s_ld = (uint32_t)(Kc[q] / 64);
        cl[q].n = (uint32_t)Nc[q];
        cl[q].k = (uint32_t)Kc[q];
    }
    Route R{};
    R.A = (const int8_t*)A->data;
    R.As = (const float*)As->data;
    R.sorted = (const int32_t*)sorted->data;
    R.off = (const int32_t*)off->data;
    R.Y = (uint16_t*)Y->data;
    R.T = (uint32_t)T;
    R.N = (uint32_t)N;
    R.n_expert = (uint32_t)n_expert;
    R.top_k = (uint32_t)top_k;
    R.a_ld = (uint32_t)a_ld;
    R.as_ld = (uint32_t)as_ld;
    R.y_ld = (uint32_t)y_ld;
    R.a_group = (uint32_t)a_group;
    R.a_sorted = a_sorted ? 1u : 0u;
    const dim3 grid((unsigned)((N + 16 * kWaves - 1) / (16 * kWaves)), (unsigned)T), block(kWaves * kWave);
    const int rc = chain ? HALO_EMIT(stream, moe_w4nl_gemv_kernel<r4dchain>, grid, block, 0, R, cl[0], cl[1], lv)
                         : HALO_EMIT(stream, moe_w4nl_gemv_kernel<HaloUnchained>, grid, block, 0, R, cl[0], cl[1], lv);
    if (rc != RAD_OK) return rc;
    return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}
