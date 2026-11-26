// r4d_ar_mhc_pcn.hip -- ar_mhc_post_collapse_rmsnorm: four-rank exact all-reduce fused with
// DeepSeek-V4.1's mhc_post_collapse_rmsnorm (`mhc_post -> mhc_collapse -> rmsnorm`) in one launch.
//
// Why this exists: DeepSeek-V4.1 runs `all_reduce(x)` immediately before `mhc_post_collapse_rmsnorm`
// 84 times per forward pass (after attention and after MoE in every layer). `out` ([M, hc*n]) is
// 4x wider than `x` ([M, n]), so all-reduce-first (one-shot at M<=4 / two-shot tail at M>=5)
// carries far fewer interconnect bytes than split-reduce + all-gather while keeping the reduced sum
// in `x` byte-for-byte identical to the standalone pair.
//
// In both modes:
//   * The reduction folds the 4 ranks in ascending rank order in fp32 from 0.0f (`r4d_ar_to_f` ->
//     `r4d_ar_from_f`), storing the exact standalone `all_reduce` bf16 word into `x`.
//   * One-shot wide (`ar_mhc_pcn_wide_kernel`, nb blocks of r4d_act_threads(M, n) threads): each block
//     owns whole rows `[r0, r1)`, pushes via P2P, takes one `r4d_ar_handshake<4>`, calls
//     `r4d_ar_link_landed`, folds each row into `x` AND an LDS row buffer (avoiding any VRAM
//     re-read of `x` and needing only `lds_barrier()`), then runs the exact `post_collapse_norm`
//     arithmetic (`f32_to_bf16` for `out` and `y`, NaN-quieted RNE `pcn_bf16` for `xc`, and
//     `pcn_block_sum` wave-by-wave reduction).
//   * Two-shot tail (`ar_mhc_pcn_tail_kernel`, nbt transport blocks + M epilogue blocks of 1024
//     threads for M in [5, 32]): blocks `[0, nbt)` run the exact two-shot over `x`, release the
//     link gate, write all shards to `x`, and release `tail[0]`; blocks `[nbt, nbt + M)` wait for
//     `tail[0] >= nbt`, acquire, and run `post_collapse_norm` on row `b - nbt`.
//
// Byte-identical to `all_reduce` (exact bf16) + `mhc_post_collapse_rmsnorm` across all outputs.
// Also provides wire 6 (ti8) 2-shot kernels for `ar_mhc_post_collapse_rmsnorm` and `ar_mhc_post`.
#include "r4d.h"
#include "r4d_common.h"
#include "r4d_ar_seq.h"
#include "r4d_ar_wide.h"
#include "r4d_ar_ti8.h"
#include "r4d_chain.h"
#include <cstdint>

namespace {

constexpr uint32_t kPcnWave = 32;

// rad_f32_to_bf16 / r4d_ds41_elem.hip ds_bf16: RNE with quieted NaNs.
__device__ __forceinline__ uint16_t pcn_bf16(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// Identical to r4d_ds41_elem.hip fz_block_sum: wave sum butterfly, then waves 0..NW-1 in order.
template <uint32_t NW>
__device__ __forceinline__ float pcn_block_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & (kPcnWave - 1), wave = threadIdx.x / kPcnWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) s += lds[w];
    return s;
}

template <int WS>
__device__ __forceinline__ int4 pcn_fold_word8(const int4* const (&src)[WS], size_t idx) {
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.0f;
#pragma unroll
    for (int r = 0; r < WS; ++r) {
        const int4 v = src[r][idx];
        const __hip_bfloat16* pv = reinterpret_cast<const __hip_bfloat16*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] += r4d_ar_to_f(pv[j]);
    }
    int4 o;
    __hip_bfloat16* po = reinterpret_cast<__hip_bfloat16*>(&o);
#pragma unroll
    for (int j = 0; j < 8; ++j) r4d_ar_from_f(acc[j], po[j]);
    return o;
}

template <bool WF32, uint32_t THR, int HCM>
__device__ __forceinline__ void ar_mhc_pcn_row(
    const uint16_t* __restrict__ xir, const uint16_t* __restrict__ hr,
    const float* __restrict__ po_r, const float* __restrict__ cb_r, uint16_t* __restrict__ orow,
    const float* __restrict__ pr_r, long pst, uint16_t* __restrict__ xr, const void* __restrict__ w,
    uint16_t* __restrict__ yr, uint32_t n, int hc, float eps, float wadd, float mult,
    float* lds_sum) {
    // At HCM <= 4, kCols = 5 fits DeepSeek-V4.1's n = 5120 cleanly in one register pass (5 * 1024),
    // eliminating the reload of `xc` and `w` and keeping column addition order bit-identical.
    constexpr int kCols = (HCM <= 4) ? 5 : 4;
    float pj[HCM], pk[HCM], cb[HCM][HCM];
#pragma unroll
    for (int k = 0; k < HCM; ++k) {
        const int kk = k < hc ? k : hc - 1;
        pj[k] = pr_r[(long)kk * pst];
        pk[k] = po_r[kk];
#pragma unroll
        for (int j = 0; j < HCM; ++j) cb[j][k] = cb_r[(j < hc ? j : hc - 1) * hc + kk];
    }
    const bool one = n <= (uint32_t)kCols * THR;
    float xv[kCols], gv[kCols];
    float ss = 0.0f;
#pragma unroll 1
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += (uint32_t)kCols * THR) {
        uint32_t hv[kCols][HCM], xi[kCols];
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            const uint32_t i = min(i0 + (uint32_t)c * THR, n - 1);
#pragma unroll
            for (int j = 0; j < HCM; ++j) hv[c][j] = hr[(long)(j < hc ? j : hc - 1) * n + i];
            xi[c] = xir[i];
            gv[c] = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        }
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            const uint32_t i = i0 + (uint32_t)c * THR;
            float v = 0.0f;
#pragma unroll
            for (int k = 0; k < HCM; ++k) {
                if (k >= hc) continue;
                float u = 0.0f;
#pragma unroll
                for (int j = 0; j < HCM; ++j)
                    if (j < hc) u += cb[j][k] * bf16_to_f32((uint16_t)hv[c][j]);
                u += pk[k] * bf16_to_f32((uint16_t)xi[c]);
                const uint16_t o = f32_to_bf16(u);
                if (i < n) orow[(long)k * n + i] = o;
                v += pj[k] * bf16_to_f32(o);
            }
            const uint16_t b = pcn_bf16(v);
            xv[c] = bf16_to_f32(b);
            if (i < n) {
                xr[i] = b;
                ss += xv[c] * xv[c];
            }
        }
    }
    ss = pcn_block_sum<THR / kPcnWave>(ss, lds_sum);
    const float s = rsqrtf(ss / (float)n + eps);
    if (one) {
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            const uint32_t i = threadIdx.x + (uint32_t)c * THR;
            if (i < n) yr[i] = f32_to_bf16(xv[c] * s * (gv[c] + wadd) * mult);
        }
        return;
    }
#pragma unroll 1
    for (uint32_t i0 = threadIdx.x; i0 < n; i0 += (uint32_t)kCols * THR) {
        uint16_t xb[kCols];
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            const uint32_t i = min(i0 + (uint32_t)c * THR, n - 1);
            xb[c] = xr[i];
            gv[c] = WF32 ? ((const float*)w)[i] : bf16_to_f32(((const uint16_t*)w)[i]);
        }
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
            const uint32_t i = i0 + (uint32_t)c * THR;
            if (i < n) yr[i] = f32_to_bf16(bf16_to_f32(xb[c]) * s * (gv[c] + wadd) * mult);
        }
    }
}

#define R4D_ARMHC_ARGS_DECL                                                                        \
    uint16_t *__restrict__ x, const uint16_t *__restrict__ h, long hp,                             \
    const float *__restrict__ post, long ptp, const float *__restrict__ comb, long cbp,            \
    uint16_t *__restrict__ out, long op, const float *__restrict__ pre, long pp, long pst,         \
    uint16_t *__restrict__ xc, uint32_t xc_ld, const void *__restrict__ w,                         \
    uint16_t *__restrict__ y, uint32_t y_ld, R4DArWidePeers<int4, WS - 1> peers,                   \
    const int4 *__restrict__ my_base, unsigned int *__restrict__ my_flags,                         \
    unsigned int *__restrict__ seq_ctrs, int M, uint32_t n, int hc, float eps, float wadd,         \
    float mult, int slot_stride16, int rank, int drain, int acq, int pub, long long *gate,         \
    unsigned int *gate_ctr

#define R4D_ARMHC_ARGS_PASS                                                                        \
    x, h, hp, post, ptp, comb, cbp, out, op, pre, pp, pst, xc, xc_ld, w, y, y_ld, peers, my_base,  \
    my_flags, seq_ctrs, M, n, hc, eps, wadd, mult, slot_stride16, rank, drain, acq, pub, gate,     \
    gate_ctr

template <bool WF32, uint32_t THR, int HCM, int WS>
__device__ __forceinline__ void ar_mhc_pcn_wide_body(R4D_ARMHC_ARGS_DECL) {
    constexpr int NPEERS = WS - 1;
    extern __shared__ __attribute__((aligned(16))) uint16_t x_lds[];
    __shared__ float lds_sum[THR / kPcnWave];
    __shared__ unsigned int s_seq, s_slot;
    const int b = (int)blockIdx.x, nb = (int)gridDim.x;
    const uint32_t tid = threadIdx.x;
    const int r0 = (int)(((long)b * M) / nb), r1 = (int)(((long)(b + 1) * M) / nb);
    const long long gate_word = r4d_ar_link_word(gate, tid);

    r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
    const unsigned int s = s_seq;
    const int parity = (int)s_slot;

    int4* ps[NPEERS];
#pragma unroll
    for (int p = 0; p < NPEERS; ++p)
        ps[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;
    const int wpr = (int)(n >> 3);
    const int4* const in4 = (const int4*)x;
    for (int k = r0 * wpr + (int)tid; k < r1 * wpr; k += (int)THR) {
        const int4 v = in4[k];
#pragma unroll
        for (int p = 0; p < NPEERS; ++p) ps[p][k] = v;
    }

    r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
    r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, tid);

    const int4* src4[WS];
#pragma unroll
    for (int r = 0; r < WS; ++r)
        src4[r] = r == rank ? (const int4*)x
                            : my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16;
    int4* const lds4 = (int4*)x_lds;
    for (int row = r0; row < r1; ++row) {
        int4* const xr4 = (int4*)(x + (size_t)row * n);
        for (int k = (int)tid; k < wpr; k += (int)THR) {
            const int4 o = pcn_fold_word8<WS>(src4, (size_t)row * wpr + k);
            xr4[k] = o;
            lds4[k] = o;
        }
        lds_barrier();
        ar_mhc_pcn_row<WF32, THR, HCM>(
            x_lds, h + (long)row * hp, post + (long)row * ptp, comb + (long)row * cbp,
            out + (long)row * op, pre + (long)row * pp, pst, xc + (size_t)row * xc_ld, w,
            y + (size_t)row * y_ld, n, hc, eps, wadd, mult, lds_sum);
        if (row + 1 < r1) __syncthreads();
    }
}

// Sequence/parity take across all `nbt + ep` blocks of tail/ti8 launches: every block advances its
// shared parity words in lockstep with r4d_ar_seq_take while only collective blocks b < nbt advance
// handshake counters ctrs[b].
__device__ __forceinline__ void post_seq_take_nbt(unsigned int* __restrict__ ctrs,
                                                  unsigned int nbw, int nbt,
                                                  unsigned int& s_seq, unsigned int& s_slot) {
    const unsigned int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
    unsigned int* const par = ctrs + nbw;
    const unsigned int pt = nt > 32u ? 32u : 0u;
    if (b < (unsigned int)nbt && tid == 0) s_seq = atomicAdd(&ctrs[b], 1u) + 1u;
    if (tid == pt) s_slot = (atomicAdd(&par[b], 1u) + 1u) & 1u;
    for (unsigned int j = b + (tid + 1u) * nb; j < nbw; j += nt * nb) atomicAdd(&par[j], 1u);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Shared two-shot tiered-int8 all-reduce body for collective blocks b < nbt.
template <uint32_t THR, int WS>
__device__ __forceinline__ void mhc_ti8_collect(
    uint16_t* __restrict__ x, R4DArWidePeers<int4, WS - 1> peers, const int4* __restrict__ my_base,
    unsigned int* __restrict__ my_flags, unsigned int* __restrict__ tail, int tid, int b, int nbt,
    unsigned int s, int parity, int rank, int n_groups, long slot_stride_bytes, long b_off_bytes,
    int drain, int acq, int pub, long long* gate, unsigned int* gate_ctr, long long gate_word) {
    constexpr int NPEERS = WS - 1;
    constexpr int nt = (int)THR;
    const int gbase = n_groups / WS, grem = n_groups % WS;
    const int sg = gbase + (grem ? 1 : 0);
    const int rec_off = TI8_REC_OFF * sg, sc_off = TI8_SC_OFF * sg;
    const int gpb = (sg + nbt - 1) / nbt;
    const int gA = b * gpb, gB = (gA + gpb < sg) ? (gA + gpb) : sg;
    const char* const my_cbase = (const char*)my_base;

    char* pslot[NPEERS];
#pragma unroll
    for (int p = 0; p < NPEERS; ++p)
        pslot[p] = (char*)peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride_bytes;

    for (int p = 0; p < NPEERS; ++p) {
        const int owner = peers.rank[p];
        const int glen = gbase + (owner < grem ? 1 : 0);
        const int gE = gB < glen ? gB : glen;
        const __hip_bfloat16* const src =
            (const __hip_bfloat16*)x +
            (size_t)(owner * gbase + (owner < grem ? owner : grem)) * TI8_GROUP;
        for (int k = 4 * gA + tid; k < 4 * gE; k += nt) {
            const int lig = k & 3;
            const int g = k >> 2;
            uint2 codes;
            unsigned int rec;
            __half scale;
            ti8_encode_group<__hip_bfloat16>(src + (size_t)k * TI8_EPL, lig, &codes, &rec, &scale);
            ((uint2*)pslot[p])[k] = codes;
            if (lig == 0) {
                ((unsigned int*)(pslot[p] + rec_off))[g] = rec;
                ((__half*)(pslot[p] + sc_off))[g] = scale;
            }
        }
    }
    r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s - 1u, drain, acq, pub);

    const size_t my_off = (size_t)(rank * gbase + (rank < grem ? rank : grem)) * TI8_GROUP;
    const __hip_bfloat16* const my_in = (const __hip_bfloat16*)x + my_off;
    __hip_bfloat16* const my_out = (__hip_bfloat16*)x + my_off;
    const int my_glen = gbase + (rank < grem ? 1 : 0);
    const int gM = gB < my_glen ? gB : my_glen;
    for (int k = 4 * gA + tid; k < 4 * gM; k += nt) {
        const int lig = k & 3;
        const int g = k >> 2;
        float acc[TI8_EPL];
#pragma unroll
        for (int i = 0; i < TI8_EPL; ++i) acc[i] = 0.0f;
        for (int r = 0; r < WS; ++r) {
            float v[TI8_EPL];
            if (r == rank) {
                ti8_load8(my_in + (size_t)k * TI8_EPL, v);
            } else {
                const char* const base =
                    my_cbase + ((size_t)parity * WS + r) * (size_t)slot_stride_bytes;
                const uint2 c = ((const uint2*)base)[k];
                const unsigned int rc = ((const unsigned int*)(base + rec_off))[g];
                const float scl = __half2float(((const __half*)(base + sc_off))[g]);
                ti8_decode_group<__hip_bfloat16>(c, rc, scl, lig, v);
            }
#pragma unroll
            for (int i = 0; i < TI8_EPL; ++i) acc[i] += v[i];
        }
        uint2 codes;
        unsigned int rc2;
        __half scl2;
        ti8_encode_group_v<__hip_bfloat16>(acc, lig, &codes, &rc2, &scl2);
#pragma unroll
        for (int p = 0; p < NPEERS; ++p) {
            char* const bB = pslot[p] + b_off_bytes;
            ((uint2*)bB)[k] = codes;
            if (lig == 0) {
                ((unsigned int*)(bB + rec_off))[g] = rc2;
                ((__half*)(bB + sc_off))[g] = scl2;
            }
        }
        float v2[TI8_EPL];
        ti8_decode_group<__hip_bfloat16>(codes, rc2, __half2float(scl2), lig, v2);
        ti8_store8(my_out + (size_t)k * TI8_EPL, v2);
    }
    r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
    r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nbt, (unsigned int)tid);

    for (int p = 0; p < NPEERS; ++p) {
        const int owner = peers.rank[p];
        const char* const baseB =
            my_cbase + ((size_t)parity * WS + owner) * (size_t)slot_stride_bytes + b_off_bytes;
        __hip_bfloat16* const dst =
            (__hip_bfloat16*)x +
            (size_t)(owner * gbase + (owner < grem ? owner : grem)) * TI8_GROUP;
        const int glen = gbase + (owner < grem ? 1 : 0);
        const int gE = gB < glen ? gB : glen;
        for (int k = 4 * gA + tid; k < 4 * gE; k += nt) {
            const int lig = k & 3;
            const int g = k >> 2;
            const uint2 c = ((const uint2*)baseB)[k];
            const unsigned int rc2 = ((const unsigned int*)(baseB + rec_off))[g];
            const float scl = __half2float(((const __half*)(baseB + sc_off))[g]);
            float v[TI8_EPL];
            ti8_decode_group<__hip_bfloat16>(c, rc2, scl, lig, v);
            ti8_store8(dst + (size_t)k * TI8_EPL, v);
        }
    }
    __syncthreads();
    if (tid == 0) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __hip_atomic_fetch_add(&tail[0], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
}

template <bool WF32, uint32_t THR, int HCM, int WS>
__global__ __launch_bounds__(THR) void ar_mhc_pcn_wide_kernel(R4D_ARMHC_ARGS_DECL) {
    ar_mhc_pcn_wide_body<WF32, THR, HCM, WS>(R4D_ARMHC_ARGS_PASS);
}

template <bool WF32, uint32_t THR, int HCM, int WS>
__global__ __launch_bounds__(THR) void r4dchain_ar_mhc_pcn_wide(R4D_ARMHC_ARGS_DECL) {
    r4d_chain_wait();
    ar_mhc_pcn_wide_body<WF32, THR, HCM, WS>(R4D_ARMHC_ARGS_PASS);
    r4d_chain_done();
}

template <bool WF32, int HCM, int WS>
__device__ __forceinline__ void ar_mhc_pcn_tail_body(
    R4D_ARMHC_ARGS_DECL, unsigned int* __restrict__ tail, int nbt) {
    constexpr uint32_t THR = 1024;
    constexpr int NPEERS = WS - 1;
    __shared__ float lds_sum[THR / kPcnWave];
    __shared__ unsigned int s_seq, s_slot;
    const int b = (int)blockIdx.x, ep = (int)gridDim.x - nbt, tid = (int)threadIdx.x, nt = (int)THR;

    post_seq_take_nbt(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, nbt, s_seq, s_slot);

    if (b < nbt) {
        const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
        const unsigned int s = s_seq;
        const int parity = (int)s_slot;
        int4* const x4 = (int4*)x;
        const int n16 = (int)(((long)M * n) >> 3);
        const int base = n16 / WS, rem = n16 % WS;
        const int sh_max = base + (rem ? 1 : 0);
#define R4D_MPTT_OFF(o) ((o) * base + ((o) < rem ? (o) : rem))
#define R4D_MPTT_LEN(o) (base + ((o) < rem ? 1 : 0))
        const int wd = (sh_max + nbt - 1) / nbt;
        const int w0 = b * wd;
        int w1 = w0 + wd;
        if (w1 > sh_max) w1 = sh_max;
        int4* pa[NPEERS];
#pragma unroll
        for (int p = 0; p < NPEERS; ++p)
            pa[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;
        for (int p = 0; p < NPEERS; ++p) {
            const int owner = peers.rank[p];
            const int4* src = x4 + (size_t)R4D_MPTT_OFF(owner);
            int4* dst = pa[p];
            const int lim = w1 < R4D_MPTT_LEN(owner) ? w1 : R4D_MPTT_LEN(owner);
            for (int k = w0 + tid; k < lim; k += nt) dst[k] = src[k];
        }
        r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s - 1u, drain, acq, pub);
        const int4* src[WS];
#pragma unroll
        for (int r = 0; r < WS; ++r)
            src[r] = (r == rank) ? (x4 + (size_t)R4D_MPTT_OFF(rank))
                                 : (my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16);
        int4* my_out = x4 + (size_t)R4D_MPTT_OFF(rank);
        const int my_lim = w1 < R4D_MPTT_LEN(rank) ? w1 : R4D_MPTT_LEN(rank);
        for (int k = w0 + tid; k < my_lim; k += nt) {
            const int4 o = pcn_fold_word8<WS>(src, (size_t)k);
            my_out[k] = o;
#pragma unroll
            for (int p = 0; p < NPEERS; ++p) pa[p][sh_max + k] = o;
        }
        r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
        r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nbt, (unsigned int)tid);
        for (int p = 0; p < NPEERS; ++p) {
            const int owner = peers.rank[p];
            const int4* srcB = my_base + ((size_t)parity * WS + owner) * (size_t)slot_stride16 + sh_max;
            int4* dst = x4 + (size_t)R4D_MPTT_OFF(owner);
            const int lim = w1 < R4D_MPTT_LEN(owner) ? w1 : R4D_MPTT_LEN(owner);
            for (int k = w0 + tid; k < lim; k += nt) dst[k] = srcB[k];
        }
#undef R4D_MPTT_OFF
#undef R4D_MPTT_LEN
        __syncthreads();
        if (tid == 0) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            __hip_atomic_fetch_add(&tail[0], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    } else {
        if (tid == 0) {
            unsigned long long z = 0;
            while (__hip_atomic_load(&tail[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) <
                   (unsigned int)nbt) {
                __builtin_amdgcn_s_sleep(1);
                if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
            }
        }
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        for (int row = b - nbt; row < M; row += ep) {
            ar_mhc_pcn_row<WF32, THR, HCM>(
                x + (size_t)row * n, h + (long)row * hp, post + (long)row * ptp,
                comb + (long)row * cbp, out + (long)row * op, pre + (long)row * pp, pst,
                xc + (size_t)row * xc_ld, w, y + (size_t)row * y_ld, n, hc, eps, wadd, mult,
                lds_sum);
            if (row + ep < M) __syncthreads();
        }
        __syncthreads();
        if (tid == 0) {
            const unsigned int prev =
                __hip_atomic_fetch_add(&tail[1], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            if (prev + 1u == (unsigned int)ep) {
                __hip_atomic_store(&tail[0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_store(&tail[1], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
        }
    }
}

template <bool WF32, int HCM, int WS>
__global__ __launch_bounds__(1024) void ar_mhc_pcn_tail_kernel(
    R4D_ARMHC_ARGS_DECL, unsigned int* __restrict__ tail, int nbt) {
    ar_mhc_pcn_tail_body<WF32, HCM, WS>(R4D_ARMHC_ARGS_PASS, tail, nbt);
}

template <bool WF32, int HCM, int WS>
__global__ __launch_bounds__(1024) void r4dchain_ar_mhc_pcn_tail(
    R4D_ARMHC_ARGS_DECL, unsigned int* __restrict__ tail, int nbt) {
    r4d_chain_wait();
    ar_mhc_pcn_tail_body<WF32, HCM, WS>(R4D_ARMHC_ARGS_PASS, tail, nbt);
    r4d_chain_done();
}

template <bool WF32, int HCM, int WS>
__device__ __forceinline__ void ar_mhc_pcn_tail_ti8_body(
    R4D_ARMHC_ARGS_DECL, unsigned int* __restrict__ tail, int nbt, int n_groups,
    long slot_stride_bytes, long b_off_bytes) {
    constexpr uint32_t THR = 512;
    __shared__ float lds_sum[THR / kPcnWave];
    __shared__ unsigned int s_seq, s_slot;
    const int b = (int)blockIdx.x, ep = (int)gridDim.x - nbt, tid = (int)threadIdx.x;
    (void)slot_stride16;

    post_seq_take_nbt(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, nbt, s_seq, s_slot);

    if (b < nbt) {
        const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
        mhc_ti8_collect<THR, WS>(
            x, peers, my_base, my_flags, tail, tid, b, nbt, s_seq, (int)s_slot, rank,
            n_groups, slot_stride_bytes, b_off_bytes, drain, acq, pub, gate, gate_ctr, gate_word);
    } else {
        if (tid == 0) {
            unsigned long long z = 0;
            while (__hip_atomic_load(&tail[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) <
                   (unsigned int)nbt) {
                __builtin_amdgcn_s_sleep(1);
                if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
            }
        }
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        for (int row = b - nbt; row < M; row += ep) {
            ar_mhc_pcn_row<WF32, THR, HCM>(
                x + (size_t)row * n, h + (long)row * hp, post + (long)row * ptp,
                comb + (long)row * cbp, out + (long)row * op, pre + (long)row * pp, pst,
                xc + (size_t)row * xc_ld, w, y + (size_t)row * y_ld, n, hc, eps, wadd, mult,
                lds_sum);
            if (row + ep < M) __syncthreads();
        }
        __syncthreads();
        if (tid == 0) {
            const unsigned int prev =
                __hip_atomic_fetch_add(&tail[1], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            if (prev + 1u == (unsigned int)ep) {
                __hip_atomic_store(&tail[0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_store(&tail[1], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
        }
    }
}

template <bool WF32, int HCM, int WS>
__global__ __launch_bounds__(512) void ar_mhc_pcn_ti8_kernel(
    R4D_ARMHC_ARGS_DECL, unsigned int* __restrict__ tail, int nbt, int n_groups,
    long slot_stride_bytes, long b_off_bytes) {
    ar_mhc_pcn_tail_ti8_body<WF32, HCM, WS>(
        R4D_ARMHC_ARGS_PASS, tail, nbt, n_groups, slot_stride_bytes, b_off_bytes);
}

template <bool WF32, int HCM, int WS>
__global__ __launch_bounds__(512) void r4dchain_ar_mhc_pcn_ti8(
    R4D_ARMHC_ARGS_DECL, unsigned int* __restrict__ tail, int nbt, int n_groups,
    long slot_stride_bytes, long b_off_bytes) {
    r4d_chain_wait();
    ar_mhc_pcn_tail_ti8_body<WF32, HCM, WS>(
        R4D_ARMHC_ARGS_PASS, tail, nbt, n_groups, slot_stride_bytes, b_off_bytes);
    r4d_chain_done();
}

#undef R4D_ARMHC_ARGS_PASS
#undef R4D_ARMHC_ARGS_DECL

}  // namespace

extern "C" int r4d_ar_mhc_pcn_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                    long peer_flags0, long peer_flags1, long peer_flags2,
                                    long my_scratch, long my_flags, long seq_ctrs,
                                    long slot_stride16, long x, long h, long hp, long post,
                                    long ptp, long comb, long cbp, long out, long op, long pre,
                                    long pp, long pst, long xc, long xc_ld, long w, int w_f32,
                                    long y, long y_ld, int M, int n, int hc, float eps, float wadd,
                                    float mult, int rank, long stream, int nblocks, int drain,
                                    int acq, int pub, long gate, long gate_ctr, long tail_ctr,
                                    int tail_nbt, int chain) {
    constexpr int WS = 4, NPEERS = WS - 1;
    const long ps[NPEERS] = { peer_scratch0, peer_scratch1, peer_scratch2 };
    const long pf[NPEERS] = { peer_flags0, peer_flags1, peer_flags2 };
    if (!x || !h || !post || !comb || !out || !pre || !xc || !w || !y) return -1;
    if (!my_scratch || !my_flags || !seq_ctrs) return -1;
    if (gate && !gate_ctr) return -1;
    if (rank < 0 || rank >= WS) return -1;
    if (M <= 0 || M > 32 || n <= 0 || n > 16384 || hc < 2 || hc > 8) return -2;
    if (n % 16) return -3;
    if ((long)M * n / 8 > slot_stride16) return -7;
    if (!xc_ld) xc_ld = n;
    if (!y_ld) y_ld = n;

    R4DArWidePeers<int4, NPEERS> peers{};
    for (int i = 0; i < NPEERS; ++i) {
        if (!ps[i] || !pf[i]) return -1;
        peers.scratch[i] = (int4*)ps[i];
        peers.flags[i] = (unsigned int*)pf[i];
        peers.rank[i] = i < rank ? i : i + 1;
    }

    int nb = nblocks < 1 ? 1 : nblocks;
    if (nb > M) nb = M;
    if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
    try {
        const long slot_bytes = 2L * WS * slot_stride16 * 16;
        const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
        r4d_ar_check_extent(my_scratch, slot_bytes, "ar_mhc_post_collapse_rmsnorm", "my_scratch");
        r4d_ar_check_extent(my_flags, flag_bytes, "ar_mhc_post_collapse_rmsnorm", "my_flags");
        r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                            "ar_mhc_post_collapse_rmsnorm", "seq_ctrs");
        for (int i = 0; i < NPEERS; ++i) {
            r4d_ar_check_extent(ps[i], slot_bytes, "ar_mhc_post_collapse_rmsnorm", "peer scratch");
            r4d_ar_check_extent(pf[i], flag_bytes, "ar_mhc_post_collapse_rmsnorm", "peer flags");
        }
    } catch (...) {
        return -8;
    }

    const bool chain_ok = chain != 0 && r4d_chain_rows_ok(M);
    const int thr = r4d_act_threads(M, n);
    const long n16 = ((long)M * n) >> 3;
    const long sh_max = n16 / WS + (n16 % WS ? 1 : 0);
    if (tail_nbt > 0 && tail_ctr && thr == 1024 && M <= 32 &&
        tail_nbt + M <= R4D_AR_WIDE_MAX_BLOCKS && 2L * sh_max <= slot_stride16) {
        try {
            r4d_ar_check_extent(my_flags, (long)tail_nbt * WS * (long)sizeof(unsigned int),
                                "ar_mhc_post_collapse_rmsnorm", "my_flags");
            r4d_ar_check_extent(tail_ctr, 2L * (long)sizeof(unsigned int),
                                "ar_mhc_post_collapse_rmsnorm", "tail counters");
            for (int i = 0; i < NPEERS; ++i)
                r4d_ar_check_extent(pf[i], (long)tail_nbt * WS * (long)sizeof(unsigned int),
                                    "ar_mhc_post_collapse_rmsnorm", "peer flags");
        } catch (...) {
            return -8;
        }
#define R4D_AMPT_L(FN)                                                                             \
        hipLaunchKernelGGL(FN,                                                                     \
                           dim3((uint32_t)(tail_nbt + M)), dim3(1024), 0, (hipStream_t)stream,     \
                           (uint16_t*)x, (const uint16_t*)h, hp, (const float*)post, ptp,          \
                           (const float*)comb, cbp, (uint16_t*)out, op, (const float*)pre, pp,     \
                           pst, (uint16_t*)xc, (uint32_t)xc_ld, (const void*)w, (uint16_t*)y,      \
                           (uint32_t)y_ld, peers, (const int4*)my_scratch, (unsigned int*)my_flags,\
                           (unsigned int*)seq_ctrs, M, (uint32_t)n, hc, eps, wadd, mult,           \
                           (int)slot_stride16, rank, drain, acq, pub, (long long*)gate,            \
                           (unsigned int*)gate_ctr, (unsigned int*)tail_ctr, tail_nbt)
#define R4D_AMPT_K(WF, HM)                                                                         \
        do { if (chain_ok) R4D_AMPT_L((r4dchain_ar_mhc_pcn_tail<WF, HM, WS>));                     \
             else          R4D_AMPT_L((ar_mhc_pcn_tail_kernel<WF, HM, WS>)); } while (0)
#define R4D_AMPT_W(WF)                                                                             \
        do { if (hc <= 4) R4D_AMPT_K(WF, 4); else R4D_AMPT_K(WF, 8); } while (0)
        if (w_f32) R4D_AMPT_W(true);
        else       R4D_AMPT_W(false);
#undef R4D_AMPT_W
#undef R4D_AMPT_K
#undef R4D_AMPT_L
        return hipGetLastError() == hipSuccess ? 0 : -4;
    }

    const size_t lds = (size_t)n * sizeof(uint16_t);
#define R4D_AMPW_L(FN, T)                                                                          \
    hipLaunchKernelGGL(FN,                                                                         \
                       dim3((uint32_t)nb), dim3(T), lds, (hipStream_t)stream, (uint16_t*)x,        \
                       (const uint16_t*)h, hp, (const float*)post, ptp, (const float*)comb, cbp,   \
                       (uint16_t*)out, op, (const float*)pre, pp, pst, (uint16_t*)xc,              \
                       (uint32_t)xc_ld, (const void*)w, (uint16_t*)y, (uint32_t)y_ld, peers,       \
                       (const int4*)my_scratch, (unsigned int*)my_flags, (unsigned int*)seq_ctrs,  \
                       M, (uint32_t)n, hc, eps, wadd, mult, (int)slot_stride16, rank, drain, acq,  \
                       pub, (long long*)gate, (unsigned int*)gate_ctr)
#define R4D_AMPW_K(WF, T, HM)                                                                      \
    do { if (chain_ok) R4D_AMPW_L((r4dchain_ar_mhc_pcn_wide<WF, T, HM, WS>), T);                   \
         else          R4D_AMPW_L((ar_mhc_pcn_wide_kernel<WF, T, HM, WS>), T); } while (0)
#define R4D_AMPW_T(WF, HM)                                                                         \
    do { if (thr == 1024)     R4D_AMPW_K(WF, 1024, HM);                                            \
         else if (thr == 512) R4D_AMPW_K(WF, 512, HM);                                             \
         else                 R4D_AMPW_K(WF, 256, HM); } while (0)
#define R4D_AMPW_W(WF)                                                                             \
    do { if (hc <= 4) R4D_AMPW_T(WF, 4); else R4D_AMPW_T(WF, 8); } while (0)
    if (w_f32) R4D_AMPW_W(true);
    else       R4D_AMPW_W(false);
#undef R4D_AMPW_W
#undef R4D_AMPW_T
#undef R4D_AMPW_K
#undef R4D_AMPW_L
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_ar_mhc_pcn_4rank_ti8(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                        long peer_flags0, long peer_flags1, long peer_flags2,
                                        long my_scratch, long my_flags, long seq_ctrs,
                                        long slot_bytes, long x, long h, long hp, long post,
                                        long ptp, long comb, long cbp, long out, long op,
                                        long pre, long pp, long pst, long xc, long xc_ld,
                                        long w, int w_f32, long y, long y_ld, int M, int n,
                                        int hc, float eps, float wadd, float mult, int rank,
                                        long stream, int nbt, int drain, int acq, int pub,
                                        long gate, long gate_ctr, long tail_ctr, int chain) {
    constexpr int WS = 4, NPEERS = WS - 1;
    const long ps[NPEERS] = { peer_scratch0, peer_scratch1, peer_scratch2 };
    const long pf[NPEERS] = { peer_flags0, peer_flags1, peer_flags2 };
    if (!x || !h || !post || !comb || !out || !pre || !xc || !w || !y) return -1;
    if (!my_scratch || !my_flags || !seq_ctrs || !tail_ctr) return -1;
    if (gate && !gate_ctr) return -1;
    if (rank < 0 || rank >= WS) return -1;
    if (M <= 0 || M > 32 || n < 128 || n > 16384 || hc < 2 || hc > 8) return -2;
    if ((n % 32) != 0 || (((long)M * n) % 128) != 0) return -3;
    if (!xc_ld) xc_ld = n;
    if (!y_ld) y_ld = n;

    R4DArWidePeers<int4, NPEERS> peers{};
    for (int i = 0; i < NPEERS; ++i) {
        if (!ps[i] || !pf[i]) return -1;
        peers.scratch[i] = (int4*)ps[i];
        peers.flags[i] = (unsigned int*)pf[i];
        peers.rank[i] = i < rank ? i : i + 1;
    }

    const int n_groups = (int)(((long)M * n) / TI8_GROUP);
    const int gbase = n_groups / WS, grem = n_groups % WS;
    const int sg = gbase + (grem ? 1 : 0);
    const long b_off = ((long)TI8_GROUP_BYTES * sg + 15L) & ~15L;
    if (2L * b_off > slot_bytes) return -7;

    int nb = nbt < 1 ? 1 : nbt;
    if (sg > 0 && nb > sg) nb = sg;
    if (nb >= R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS - 1;
    const int ep = (nb + M <= R4D_AR_WIDE_MAX_BLOCKS) ? M : (R4D_AR_WIDE_MAX_BLOCKS - nb);
    if (ep < 1) return -2;

    try {
        const long total_slot_bytes = 2L * WS * slot_bytes;
        const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
        r4d_ar_check_extent(my_scratch, total_slot_bytes, "ar_mhc_post_collapse_rmsnorm", "my_scratch");
        r4d_ar_check_extent(my_flags, flag_bytes, "ar_mhc_post_collapse_rmsnorm", "my_flags");
        r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                            "ar_mhc_post_collapse_rmsnorm", "seq_ctrs");
        r4d_ar_check_extent(tail_ctr, 2L * (long)sizeof(unsigned int),
                            "ar_mhc_post_collapse_rmsnorm", "tail counters");
        for (int i = 0; i < NPEERS; ++i) {
            r4d_ar_check_extent(ps[i], total_slot_bytes, "ar_mhc_post_collapse_rmsnorm", "peer scratch");
            r4d_ar_check_extent(pf[i], flag_bytes, "ar_mhc_post_collapse_rmsnorm", "peer flags");
        }
    } catch (...) {
        return -8;
    }

    const bool chain_ok = chain != 0 && r4d_chain_rows_ok(M);
#define R4D_AMP6_L(FN)                                                                             \
    hipLaunchKernelGGL(FN,                                                                         \
                       dim3((uint32_t)(nb + ep)), dim3(512u), 0, (hipStream_t)stream,              \
                       (uint16_t*)x, (const uint16_t*)h, hp, (const float*)post, ptp,              \
                       (const float*)comb, cbp, (uint16_t*)out, op, (const float*)pre, pp, pst,    \
                       (uint16_t*)xc, (uint32_t)xc_ld, (const void*)w, (uint16_t*)y,               \
                       (uint32_t)y_ld, peers, (const int4*)my_scratch, (unsigned int*)my_flags,    \
                       (unsigned int*)seq_ctrs, M, (uint32_t)n, hc, eps, wadd, mult, 0, rank,      \
                       drain, acq, pub, (long long*)gate, (unsigned int*)gate_ctr,                 \
                       (unsigned int*)tail_ctr, nb, n_groups, slot_bytes, b_off)
#define R4D_AMP6_K(WF, HM)                                                                         \
    do { if (chain_ok) R4D_AMP6_L((r4dchain_ar_mhc_pcn_ti8<WF, HM, WS>));                          \
         else          R4D_AMP6_L((ar_mhc_pcn_ti8_kernel<WF, HM, WS>)); } while (0)
#define R4D_AMP6_W(WF)                                                                             \
    do { if (hc <= 4) R4D_AMP6_K(WF, 4); else R4D_AMP6_K(WF, 8); } while (0)
    if (w_f32) R4D_AMP6_W(true);
    else       R4D_AMP6_W(false);
#undef R4D_AMP6_W
#undef R4D_AMP6_K
#undef R4D_AMP6_L
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// =================================================================================================
// ar_mhc_post: four-rank all_reduce(x) (wire=0 exact or wire=6 ti8) fused with mhc_post.
// Trunk junctions before Engram layers (blk.0 and blk.13 in DeepSeek-V4.1); init/host_init/fini
// and RadArgs entry points live in r4d_ar_entry.hip on the shared ArInst.
// =================================================================================================

namespace {

template <int HCM>
__device__ __forceinline__ void post_load_gates(const float* __restrict__ po_r,
                                                const float* __restrict__ cb_r, int hc,
                                                float (&pk)[HCM], float (&cb)[HCM][HCM]) {
#pragma unroll
    for (int k = 0; k < HCM; ++k) {
        const int kk = k < hc ? k : hc - 1;
        pk[k] = po_r[kk];
#pragma unroll
        for (int j = 0; j < HCM; ++j)
            cb[j][k] = cb_r[(j < hc ? j : hc - 1) * hc + kk];
    }
}

template <int HCM>
__device__ __forceinline__ void post_store_word8(const int4 xv, const uint16_t* __restrict__ hr,
                                                 uint16_t* __restrict__ orow, uint32_t n, int hc,
                                                 const float (&pk)[HCM],
                                                 const float (&cb)[HCM][HCM], int k,
                                                 bool aligned16) {
    union {
        int4 v;
        uint16_t u[8];
    } xu;
    xu.v = xv;
    if (aligned16) {
        const int wpr = (int)(n >> 3);
        union {
            int4 v;
            uint16_t u[8];
        } hu[HCM];
#pragma unroll
        for (int j = 0; j < HCM; ++j)
            if (j < hc) hu[j].v = ((const int4*)hr)[(size_t)j * wpr + k];
#pragma unroll
        for (int d = 0; d < HCM; ++d) {
            if (d >= hc) continue;
            union {
                int4 v;
                uint16_t u[8];
            } ou;
            const float pkd = pk[d];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float u = 0.0f;
#pragma unroll
                for (int j = 0; j < HCM; ++j)
                    if (j < hc) u += cb[j][d] * bf16_to_f32(hu[j].u[c]);
                u += pkd * bf16_to_f32(xu.u[c]);
                ou.u[c] = f32_to_bf16(u);
            }
            ((int4*)orow)[(size_t)d * wpr + k] = ou.v;
        }
        return;
    }
    const uint32_t i0 = (uint32_t)k << 3;
    uint16_t hv[HCM][8];
#pragma unroll
    for (int j = 0; j < HCM; ++j)
        if (j < hc)
#pragma unroll
            for (int c = 0; c < 8; ++c) hv[j][c] = hr[(size_t)j * n + i0 + c];
#pragma unroll
    for (int d = 0; d < HCM; ++d) {
        if (d >= hc) continue;
        const float pkd = pk[d];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            float u = 0.0f;
#pragma unroll
            for (int j = 0; j < HCM; ++j)
                if (j < hc) u += cb[j][d] * bf16_to_f32(hv[j][c]);
            u += pkd * bf16_to_f32(xu.u[c]);
            orow[(size_t)d * n + i0 + c] = f32_to_bf16(u);
        }
    }
}

template <int HCM, uint32_t THR>
__device__ __forceinline__ void post_epilogue_wait_and_write(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ h, long hp,
    const float* __restrict__ post, long ptp, const float* __restrict__ comb, long cbp,
    uint16_t* __restrict__ out, long op, unsigned int* __restrict__ tail, int M, uint32_t n,
    int hc, int nbt) {
    const int b = (int)blockIdx.x, ep = (int)gridDim.x - nbt, tid = (int)threadIdx.x;
    if (tid == 0) {
        unsigned long long z = 0;
        while (__hip_atomic_load(&tail[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) <
               (unsigned int)nbt) {
            __builtin_amdgcn_s_sleep(1);
            if (++z > R4D_AR_SPIN_MAX) __builtin_trap();
        }
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

    const int wpr = (int)(n >> 3);
    for (int row = b - nbt; row < M; row += ep) {
        float pk[HCM], cb[HCM][HCM];
        post_load_gates<HCM>(post + (long)row * ptp, comb + (long)row * cbp, hc, pk, cb);
        const uint16_t* const hr = h + (long)row * hp;
        uint16_t* const orow = out + (long)row * op;
        const bool aligned16 = (((uintptr_t)hr | (uintptr_t)orow) & 15u) == 0;
        const int4* const xr4 = (const int4*)(x + (size_t)row * n);
        for (int k = tid; k < wpr; k += (int)THR)
            post_store_word8<HCM>(xr4[k], hr, orow, n, hc, pk, cb, k, aligned16);
    }
    __syncthreads();
    if (tid == 0) {
        const unsigned int prev =
            __hip_atomic_fetch_add(&tail[1], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (prev + 1u == (unsigned int)ep) {
            __hip_atomic_store(&tail[0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&tail[1], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

#define R4D_AMP_POST_DECL                                                                          \
    uint16_t *__restrict__ x, const uint16_t *__restrict__ h, long hp,                             \
    const float *__restrict__ post, long ptp, const float *__restrict__ comb, long cbp,            \
    uint16_t *__restrict__ out, long op, R4DArWidePeers<int4, WS - 1> peers,                       \
    const int4 *__restrict__ my_base, unsigned int *__restrict__ my_flags,                         \
    unsigned int *__restrict__ seq_ctrs, int M, uint32_t n, int hc, int slot_stride16, int rank,   \
    int drain, int acq, int pub, long long *gate, unsigned int *gate_ctr

#define R4D_AMP_POST_PASS                                                                          \
    x, h, hp, post, ptp, comb, cbp, out, op, peers, my_base, my_flags, seq_ctrs, M, n, hc,         \
    slot_stride16, rank, drain, acq, pub, gate, gate_ctr

template <uint32_t THR, int HCM, int WS>
__device__ __forceinline__ void ar_mhc_post_wide_body(R4D_AMP_POST_DECL) {
    constexpr int NPEERS = WS - 1;
    __shared__ unsigned int s_seq, s_slot;
    const int b = (int)blockIdx.x, nb = (int)gridDim.x;
    const uint32_t tid = threadIdx.x;
    const int r0 = (int)(((long)b * M) / nb), r1 = (int)(((long)(b + 1) * M) / nb);
    const long long gate_word = r4d_ar_link_word(gate, tid);

    r4d_ar_seq_take(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, s_seq, s_slot);
    const unsigned int s = s_seq;
    const int parity = (int)s_slot;

    int4* ps[NPEERS];
#pragma unroll
    for (int p = 0; p < NPEERS; ++p)
        ps[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;
    const int wpr = (int)(n >> 3);
    const int4* const in4 = (const int4*)x;
    for (int k = r0 * wpr + (int)tid; k < r1 * wpr; k += (int)THR) {
        const int4 v = in4[k];
#pragma unroll
        for (int p = 0; p < NPEERS; ++p) ps[p][k] = v;
    }

    r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
    r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nb, tid);

    const int4* src4[WS];
#pragma unroll
    for (int r = 0; r < WS; ++r)
        src4[r] = r == rank ? (const int4*)x
                            : my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16;
    for (int row = r0; row < r1; ++row) {
        float pk[HCM], cb[HCM][HCM];
        post_load_gates<HCM>(post + (long)row * ptp, comb + (long)row * cbp, hc, pk, cb);
        int4* const xr4 = (int4*)(x + (size_t)row * n);
        const uint16_t* const hr = h + (long)row * hp;
        uint16_t* const orow = out + (long)row * op;
        const bool aligned16 = (((uintptr_t)hr | (uintptr_t)orow) & 15u) == 0;
        for (int k = (int)tid; k < wpr; k += (int)THR) {
            const int4 o = pcn_fold_word8<WS>(src4, (size_t)row * wpr + k);
            xr4[k] = o;
            post_store_word8<HCM>(o, hr, orow, n, hc, pk, cb, k, aligned16);
        }
    }
}

template <uint32_t THR, int HCM, int WS>
__global__ __launch_bounds__(THR) void ar_mhc_post_4rank_kernel(R4D_AMP_POST_DECL) {
    ar_mhc_post_wide_body<THR, HCM, WS>(R4D_AMP_POST_PASS);
}

template <uint32_t THR, int HCM, int WS>
__global__ __launch_bounds__(THR) void r4dchain_ar_mhc_post_4rank_wide(R4D_AMP_POST_DECL) {
    r4d_chain_wait();
    ar_mhc_post_wide_body<THR, HCM, WS>(R4D_AMP_POST_PASS);
    r4d_chain_done();
}

template <int HCM, int WS>
__device__ __forceinline__ void ar_mhc_post_tail_exact_body(
    R4D_AMP_POST_DECL, unsigned int* __restrict__ tail, int nbt) {
    constexpr uint32_t THR = 1024;
    constexpr int NPEERS = WS - 1;
    __shared__ unsigned int s_seq, s_slot;
    const int b = (int)blockIdx.x, tid = (int)threadIdx.x, nt = (int)THR;
    (void)slot_stride16;

    post_seq_take_nbt(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, nbt, s_seq, s_slot);

    if (b < nbt) {
        const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
        const unsigned int s = s_seq;
        const int parity = (int)s_slot;
        int4* const x4 = (int4*)x;
        const int n16 = (int)(((long)M * n) >> 3);
        const int base = n16 / WS, rem = n16 % WS;
        const int sh_max = base + (rem ? 1 : 0);
#define R4D_AMPE_OFF(o) ((o) * base + ((o) < rem ? (o) : rem))
#define R4D_AMPE_LEN(o) (base + ((o) < rem ? 1 : 0))
        const int wd = (sh_max + nbt - 1) / nbt;
        const int w0 = b * wd;
        int w1 = w0 + wd;
        if (w1 > sh_max) w1 = sh_max;
        int4* pa[NPEERS];
#pragma unroll
        for (int p = 0; p < NPEERS; ++p)
            pa[p] = peers.scratch[p] + ((size_t)parity * WS + rank) * (size_t)slot_stride16;
        for (int p = 0; p < NPEERS; ++p) {
            const int owner = peers.rank[p];
            const int4* src = x4 + (size_t)R4D_AMPE_OFF(owner);
            int4* dst = pa[p];
            const int lim = w1 < R4D_AMPE_LEN(owner) ? w1 : R4D_AMPE_LEN(owner);
            for (int k = w0 + tid; k < lim; k += nt) dst[k] = src[k];
        }
        r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s - 1u, drain, acq, pub);
        const int4* src[WS];
#pragma unroll
        for (int r = 0; r < WS; ++r)
            src[r] = (r == rank) ? (x4 + (size_t)R4D_AMPE_OFF(rank))
                                 : (my_base + ((size_t)parity * WS + r) * (size_t)slot_stride16);
        int4* const my_out = x4 + (size_t)R4D_AMPE_OFF(rank);
        const int my_lim = w1 < R4D_AMPE_LEN(rank) ? w1 : R4D_AMPE_LEN(rank);
        for (int k = w0 + tid; k < my_lim; k += nt) {
            const int4 o = pcn_fold_word8<WS>(src, (size_t)k);
            my_out[k] = o;
#pragma unroll
            for (int p = 0; p < NPEERS; ++p) pa[p][sh_max + k] = o;
        }
        r4d_ar_handshake<WS>(peers, my_flags, b, rank, 2u * s, drain, acq, pub);
        r4d_ar_link_landed(gate, gate_word, gate_ctr, (unsigned int)nbt, (unsigned int)tid);
        for (int p = 0; p < NPEERS; ++p) {
            const int owner = peers.rank[p];
            const int4* srcB = my_base + ((size_t)parity * WS + owner) * (size_t)slot_stride16 + sh_max;
            int4* dst = x4 + (size_t)R4D_AMPE_OFF(owner);
            const int lim = w1 < R4D_AMPE_LEN(owner) ? w1 : R4D_AMPE_LEN(owner);
            for (int k = w0 + tid; k < lim; k += nt) dst[k] = srcB[k];
        }
#undef R4D_AMPE_OFF
#undef R4D_AMPE_LEN
        __syncthreads();
        if (tid == 0) {
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            __hip_atomic_fetch_add(&tail[0], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    } else {
        post_epilogue_wait_and_write<HCM, THR>(
            x, h, hp, post, ptp, comb, cbp, out, op, tail, M, n, hc, nbt);
    }
}

template <int HCM, int WS>
__global__ __launch_bounds__(1024) void ar_mhc_post_4rank_tail_kernel(
    R4D_AMP_POST_DECL, unsigned int* __restrict__ tail, int nbt) {
    ar_mhc_post_tail_exact_body<HCM, WS>(R4D_AMP_POST_PASS, tail, nbt);
}

template <int HCM, int WS>
__global__ __launch_bounds__(1024) void r4dchain_ar_mhc_post_4rank_tail(
    R4D_AMP_POST_DECL, unsigned int* __restrict__ tail, int nbt) {
    r4d_chain_wait();
    ar_mhc_post_tail_exact_body<HCM, WS>(R4D_AMP_POST_PASS, tail, nbt);
    r4d_chain_done();
}

template <int HCM, int WS>
__device__ __forceinline__ void ar_mhc_post_tail_ti8_body(
    R4D_AMP_POST_DECL, unsigned int* __restrict__ tail, int nbt, int n_groups,
    long slot_stride_bytes, long b_off_bytes) {
    constexpr uint32_t THR = 512;
    __shared__ unsigned int s_seq, s_slot;
    const int b = (int)blockIdx.x, tid = (int)threadIdx.x;
    (void)slot_stride16;

    post_seq_take_nbt(seq_ctrs, R4D_AR_WIDE_MAX_BLOCKS, nbt, s_seq, s_slot);

    if (b < nbt) {
        const long long gate_word = r4d_ar_link_word(gate, (unsigned int)tid);
        mhc_ti8_collect<THR, WS>(
            x, peers, my_base, my_flags, tail, tid, b, nbt, s_seq, (int)s_slot, rank,
            n_groups, slot_stride_bytes, b_off_bytes, drain, acq, pub, gate, gate_ctr, gate_word);
    } else {
        post_epilogue_wait_and_write<HCM, THR>(
            x, h, hp, post, ptp, comb, cbp, out, op, tail, M, n, hc, nbt);
    }
}

template <int HCM, int WS>
__global__ __launch_bounds__(512) void ar_mhc_post_4rank_ti8_kernel(
    R4D_AMP_POST_DECL, unsigned int* __restrict__ tail, int nbt, int n_groups,
    long slot_stride_bytes, long b_off_bytes) {
    ar_mhc_post_tail_ti8_body<HCM, WS>(R4D_AMP_POST_PASS, tail, nbt, n_groups,
                                       slot_stride_bytes, b_off_bytes);
}

template <int HCM, int WS>
__global__ __launch_bounds__(512) void r4dchain_ar_mhc_post_4rank_ti8(
    R4D_AMP_POST_DECL, unsigned int* __restrict__ tail, int nbt, int n_groups,
    long slot_stride_bytes, long b_off_bytes) {
    r4d_chain_wait();
    ar_mhc_post_tail_ti8_body<HCM, WS>(R4D_AMP_POST_PASS, tail, nbt, n_groups,
                                       slot_stride_bytes, b_off_bytes);
    r4d_chain_done();
}

#undef R4D_AMP_POST_PASS
#undef R4D_AMP_POST_DECL

}  // namespace

extern "C" int r4d_ar_mhc_post_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                     long peer_flags0, long peer_flags1, long peer_flags2,
                                     long my_scratch, long my_flags, long seq_ctrs,
                                     long slot_stride16, long x, long h, long hp, long post,
                                     long ptp, long comb, long cbp, long out, long op, int M,
                                     int n, int hc, int rank, long stream, int nblocks,
                                     int drain, int acq, int pub, long gate, long gate_ctr,
                                     long tail_ctr, int tail_nbt, int chain) {
    constexpr int WS = 4, NPEERS = WS - 1;
    const long ps[NPEERS] = { peer_scratch0, peer_scratch1, peer_scratch2 };
    const long pf[NPEERS] = { peer_flags0, peer_flags1, peer_flags2 };
    if (!x || !h || !post || !comb || !out) return -1;
    if (!my_scratch || !my_flags || !seq_ctrs) return -1;
    if (gate && !gate_ctr) return -1;
    if (rank < 0 || rank >= WS) return -1;
    if (M <= 0 || M > 32 || n <= 0 || n > 16384 || (n % 16) != 0 || hc < 2 || hc > 8) return -2;
    if ((long)M * n / 8 > slot_stride16) return -7;

    R4DArWidePeers<int4, NPEERS> peers{};
    for (int i = 0; i < NPEERS; ++i) {
        if (!ps[i] || !pf[i]) return -1;
        peers.scratch[i] = (int4*)ps[i];
        peers.flags[i] = (unsigned int*)pf[i];
        peers.rank[i] = i < rank ? i : i + 1;
    }

    int nb = nblocks < 1 ? 1 : nblocks;
    if (nb > M) nb = M;
    if (nb > R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS;
    try {
        const long slot_bytes = 2L * WS * slot_stride16 * 16;
        const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
        r4d_ar_check_extent(my_scratch, slot_bytes, "ar_mhc_post", "my_scratch");
        r4d_ar_check_extent(my_flags, flag_bytes, "ar_mhc_post", "my_flags");
        r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                            "ar_mhc_post", "seq_ctrs");
        for (int i = 0; i < NPEERS; ++i) {
            r4d_ar_check_extent(ps[i], slot_bytes, "ar_mhc_post", "peer_scratch");
            r4d_ar_check_extent(pf[i], flag_bytes, "ar_mhc_post", "peer_flags");
        }
    } catch (...) {
        return -8;
    }

    const bool chain_ok = chain != 0 && r4d_chain_rows_ok(M);
    const long words16 = ((long)M * n) >> 3;
    const long sh_max = words16 / WS + (words16 % WS ? 1L : 0L);
    if (tail_nbt > 0 && tail_ctr && M >= 5 && M <= 32 && tail_nbt < R4D_AR_WIDE_MAX_BLOCKS &&
        2L * sh_max <= slot_stride16) {
        const int ep = (tail_nbt + M <= R4D_AR_WIDE_MAX_BLOCKS) ? M
                                                                 : (R4D_AR_WIDE_MAX_BLOCKS - tail_nbt);
        if (ep < 1) return -2;
        try {
            r4d_ar_check_extent(my_flags, (long)tail_nbt * WS * (long)sizeof(unsigned int),
                                "ar_mhc_post", "my_flags");
            r4d_ar_check_extent(tail_ctr, 2L * (long)sizeof(unsigned int),
                                "ar_mhc_post", "tail counters");
            for (int i = 0; i < NPEERS; ++i)
                r4d_ar_check_extent(pf[i], (long)tail_nbt * WS * (long)sizeof(unsigned int),
                                    "ar_mhc_post", "peer_flags");
        } catch (...) {
            return -8;
        }
#define R4D_POSTT_L(FN)                                                                            \
        hipLaunchKernelGGL(FN,                                                                     \
                           dim3((uint32_t)(tail_nbt + ep)), dim3(1024u), 0, (hipStream_t)stream,   \
                           (uint16_t*)x, (const uint16_t*)h, hp, (const float*)post, ptp,          \
                           (const float*)comb, cbp, (uint16_t*)out, op, peers,                     \
                           (const int4*)my_scratch, (unsigned int*)my_flags,                       \
                           (unsigned int*)seq_ctrs, M, (uint32_t)n, hc, (int)slot_stride16, rank,  \
                           drain, acq, pub, (long long*)gate, (unsigned int*)gate_ctr,             \
                           (unsigned int*)tail_ctr, tail_nbt)
#define R4D_POSTT_K(HM)                                                                            \
        do { if (chain_ok) R4D_POSTT_L((r4dchain_ar_mhc_post_4rank_tail<HM, WS>));                 \
             else          R4D_POSTT_L((ar_mhc_post_4rank_tail_kernel<HM, WS>)); } while (0)
        if (hc <= 4) R4D_POSTT_K(4);
        else         R4D_POSTT_K(8);
#undef R4D_POSTT_K
#undef R4D_POSTT_L
        return hipGetLastError() == hipSuccess ? 0 : -4;
    }

    const int thr = r4d_act_threads(M, n);
#define R4D_POSTW_L(FN, T)                                                                         \
    hipLaunchKernelGGL(FN,                                                                         \
                       dim3((uint32_t)nb), dim3(T), 0, (hipStream_t)stream, (uint16_t*)x,          \
                       (const uint16_t*)h, hp, (const float*)post, ptp, (const float*)comb, cbp,   \
                       (uint16_t*)out, op, peers, (const int4*)my_scratch,                         \
                       (unsigned int*)my_flags, (unsigned int*)seq_ctrs, M, (uint32_t)n, hc,       \
                       (int)slot_stride16, rank, drain, acq, pub, (long long*)gate,                \
                       (unsigned int*)gate_ctr)
#define R4D_POSTW_K(T, HM)                                                                         \
    do { if (chain_ok) R4D_POSTW_L((r4dchain_ar_mhc_post_4rank_wide<T, HM, WS>), T);               \
         else          R4D_POSTW_L((ar_mhc_post_4rank_kernel<T, HM, WS>), T); } while (0)
#define R4D_POSTW_T(HM)                                                                            \
    do { if (thr == 1024)     R4D_POSTW_K(1024, HM);                                               \
         else if (thr == 512) R4D_POSTW_K(512, HM);                                                \
         else                 R4D_POSTW_K(256, HM); } while (0)
    if (hc <= 4) R4D_POSTW_T(4);
    else         R4D_POSTW_T(8);
#undef R4D_POSTW_T
#undef R4D_POSTW_K
#undef R4D_POSTW_L
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r4d_ar_mhc_post_4rank_ti8(long peer_scratch0, long peer_scratch1,
                                         long peer_scratch2, long peer_flags0,
                                         long peer_flags1, long peer_flags2, long my_scratch,
                                         long my_flags, long seq_ctrs, long slot_bytes, long x,
                                         long h, long hp, long post, long ptp, long comb,
                                         long cbp, long out, long op, int M, int n, int hc,
                                         int rank, long stream, int nbt, int drain, int acq,
                                         int pub, long gate, long gate_ctr, long tail_ctr,
                                         int chain) {
    constexpr int WS = 4, NPEERS = WS - 1;
    const long ps[NPEERS] = { peer_scratch0, peer_scratch1, peer_scratch2 };
    const long pf[NPEERS] = { peer_flags0, peer_flags1, peer_flags2 };
    if (!x || !h || !post || !comb || !out) return -1;
    if (!my_scratch || !my_flags || !seq_ctrs || !tail_ctr) return -1;
    if (gate && !gate_ctr) return -1;
    if (rank < 0 || rank >= WS) return -1;
    if (M <= 0 || M > 32 || n < 128 || n > 16384 || (n % 32) != 0 ||
        (((long)M * n) % 128) != 0 || hc < 2 || hc > 8)
        return -2;

    R4DArWidePeers<int4, NPEERS> peers{};
    for (int i = 0; i < NPEERS; ++i) {
        if (!ps[i] || !pf[i]) return -1;
        peers.scratch[i] = (int4*)ps[i];
        peers.flags[i] = (unsigned int*)pf[i];
        peers.rank[i] = i < rank ? i : i + 1;
    }

    const int n_groups = (int)(((long)M * n) / TI8_GROUP);
    const int gbase = n_groups / WS, grem = n_groups % WS;
    const int sg = gbase + (grem ? 1 : 0);
    const long b_off = ((long)TI8_GROUP_BYTES * sg + 15L) & ~15L;
    if (2L * b_off > slot_bytes) return -7;

    int nb = nbt < 1 ? 1 : nbt;
    if (sg > 0 && nb > sg) nb = sg;
    if (nb >= R4D_AR_WIDE_MAX_BLOCKS) nb = R4D_AR_WIDE_MAX_BLOCKS - 1;
    const int ep = (nb + M <= R4D_AR_WIDE_MAX_BLOCKS) ? M : (R4D_AR_WIDE_MAX_BLOCKS - nb);
    if (ep < 1) return -2;

    try {
        const long total_slot_bytes = 2L * WS * slot_bytes;
        const long flag_bytes = (long)nb * WS * (long)sizeof(unsigned int);
        r4d_ar_check_extent(my_scratch, total_slot_bytes, "ar_mhc_post", "my_scratch");
        r4d_ar_check_extent(my_flags, flag_bytes, "ar_mhc_post", "my_flags");
        r4d_ar_check_extent(seq_ctrs, 2L * R4D_AR_WIDE_MAX_BLOCKS * (long)sizeof(unsigned int),
                            "ar_mhc_post", "seq_ctrs");
        r4d_ar_check_extent(tail_ctr, 2L * (long)sizeof(unsigned int),
                            "ar_mhc_post", "tail counters");
        for (int i = 0; i < NPEERS; ++i) {
            r4d_ar_check_extent(ps[i], total_slot_bytes, "ar_mhc_post", "peer_scratch");
            r4d_ar_check_extent(pf[i], flag_bytes, "ar_mhc_post", "peer_flags");
        }
    } catch (...) {
        return -8;
    }

    const bool chain_ok = chain != 0 && r4d_chain_rows_ok(M);
#define R4D_POST6_L(FN)                                                                            \
    hipLaunchKernelGGL(FN,                                                                         \
                       dim3((uint32_t)(nb + ep)), dim3(512u), 0, (hipStream_t)stream,              \
                       (uint16_t*)x, (const uint16_t*)h, hp, (const float*)post, ptp,              \
                       (const float*)comb, cbp, (uint16_t*)out, op, peers,                         \
                       (const int4*)my_scratch, (unsigned int*)my_flags, (unsigned int*)seq_ctrs,  \
                       M, (uint32_t)n, hc, 0, rank, drain, acq, pub, (long long*)gate,             \
                       (unsigned int*)gate_ctr, (unsigned int*)tail_ctr, nb, n_groups, slot_bytes, \
                       b_off)
#define R4D_POST6_K(HM)                                                                            \
    do { if (chain_ok) R4D_POST6_L((r4dchain_ar_mhc_post_4rank_ti8<HM, WS>));                      \
         else          R4D_POST6_L((ar_mhc_post_4rank_ti8_kernel<HM, WS>)); } while (0)
    if (hc <= 4) R4D_POST6_K(4);
    else         R4D_POST6_K(8);
#undef R4D_POST6_K
#undef R4D_POST6_L
    return hipGetLastError() == hipSuccess ? 0 : -4;
}
