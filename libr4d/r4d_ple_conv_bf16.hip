// r4d_ple_conv_bf16.hip -- Per-Layer Embedding's dilated depthwise convolution, its silu, and the
// residual add that finishes the layer; and, as `causal_conv1d`, the same convolution with no
// activation, no residual and no dilation, which is LFM2's short convolution.
//
//   y[t][c] = resid[t][c] + silu( sum_j w[c][j] * x[t - (width-1-j)*dilation][c] )
//
// libref/ref_ple.cpp is the oracle and docs/OPS.md the contract. The taps are `dilation` apart, so
// at width 4 and dilation 3 they sit 9, 6, 3 and 0 timesteps back and a sequence carries a history
// of (width-1)*dilation = 9, not 3. An off-by-one in a convolution cache yields fluent output from
// a slightly wrong context, so the indexing mirrors ref's line for line: state slot `i` holds
// relative position `i - hist`, a read at `src < 0` becomes `cs[slot][c][off + src + hist]`, and
// `has_init` 0 means that window is zeros, not the slot's stale contents.
//
// One thread per channel: a depthwise convolution never mixes channels, so only time must stay
// together. A thread walks a tile of tokens; consecutive threads own consecutive channels, so a
// wave loads one contiguous 512-byte slice of a token row.
//
// The grid is (channel blocks, token tiles, sequences). Token tiles are counted from the batch's
// total token count, an upper bound on any one sequence's; a short sequence's extra blocks find
// `t0 >= slen` and return. A per-sequence count would mean reading `cu` on the host.
//
// The taps are read from memory, not kept in a shift register: the window would need `hist + 1`
// registers indexed by a runtime value, which compiles to a select tree (see CU_STATIC in
// r4d_gdn_conv_w4_h128_bf16.hip), and a compile-time window would need a specialisation per
// (width, dilation). The re-reads hit a tile working set of (BT + hist) * 512 bytes (21 KB at
// BT 32, L0-resident), and the op is bound by the three arrays it must touch anyway.
//
// The window write-back is in the same launch; see the note at the end of the kernel. It relies
// on `hist < kPcBT`, asserted below.
#include "r4d.h"
#include "r4d_args.h"
#include "r4d_chain.h"
#include "r4d_common.h"

#include <cstdint>

namespace {

enum {
    kPcThreads = 256,
    kPcW4Thr   = 64,   /* 2 waves, 8 channels/thread -> 512 ch = 1 blk, 2304 ch = 5 blks */
    kPcBT      = 32,   /* tokens a thread walks; the tile's working set is (BT + hist) * 512 B */
    kPcMaxW    = 8,    /* taps held in registers, so the weight loop unrolls to a static bound */
    kPcMaxHist = 16,   /* (width-1)*dilation, the write-back's register window; see cPleConv */
};

/* The write-back lives in tile 0, which is sound only while the deepest tap is inside the first
 * tile. At hist >= BT a second block would read history and the write-back would overwrite bytes
 * it is still reading, silently corrupting a sequence's first tokens. The constraint table caps
 * hist <= 16; the tile size is the other half of that bound. */
static_assert(kPcMaxHist < kPcBT, "the conv history must fit inside tile 0 for the fused write-back");

/* expf rather than __expf: the op is bound by memory, not the transcendental, and the oracle uses
 * std::exp. The instruction-bound GDN conv makes the opposite trade. */
__device__ __forceinline__ float pc_silu(float z) { return z / (1.0f + expf(-z)); }

/* SILU is ple_conv's activation; causal_conv1d (LFM2's short convolution) is the same window with
 * no activation, no residual and dilation one. */
/* sconv_res (SCONV) reads `dyn` [M, heads*k] at row pitch `dld`, a channel's head being c / hch,
 * and takes its residual from x rather than `resid`, scaled by `sdiv` = sqrt(m*k + 1). */
#define PLE_CONV_PARAMS                                                                         \
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,                             \
    uint16_t* __restrict__ cs, const int32_t* __restrict__ cu,                                  \
    const uint16_t* __restrict__ resid, const int32_t* __restrict__ sidx,                       \
    const int32_t* __restrict__ hini, const int32_t* __restrict__ nacc,                         \
    uint16_t* __restrict__ y, const uint16_t* __restrict__ bias, uint32_t n, uint32_t wid,      \
    uint32_t dil, uint32_t hist, uint32_t xld, uint32_t rld, uint32_t yld, uint32_t wld,        \
    long long cs_slot, long long cs_ch, long long cs_tok, long long state_len,                  \
    const uint16_t* __restrict__ dyn, uint32_t dld, uint32_t hch, float sdiv
template <bool SILU, bool SCONV>
__device__ __forceinline__ void ple_conv_body(PLE_CONV_PARAMS) {

    const uint32_t c = blockIdx.x * kPcThreads + threadIdx.x;
    if (c >= n) return;
    const uint32_t s = blockIdx.z;
    const int32_t bos = cu[s];
    const int32_t slen = cu[s + 1] - bos;
    if (slen <= 0) return;
    const int32_t t0 = (int32_t)(blockIdx.y * kPcBT);
    if (t0 >= slen) return;
    const int32_t t1 = min(t0 + (int32_t)kPcBT, slen);

    const int32_t slot = sidx ? sidx[s] : (int32_t)s;
    if (slot < 0) return;
    /* A cold slot's window is zeros: a fresh sequence on a recycled slot would otherwise convolve
     * the previous sequence's tail into its first `hist` tokens. */
    const bool warm = cs && (!hini || hini[s] != 0);
    const long long off = nacc ? (long long)nacc[s] - 1 : 0;

    /* This channel's taps, read once for the whole tile. */
    float wt[kPcMaxW];
#pragma unroll
    for (uint32_t j = 0; j < kPcMaxW; ++j) wt[j] = j < wid ? bf16_to_f32(w[(size_t)c * wld + j]) : 0.0f;

    const uint16_t* __restrict__ xc = x + (size_t)bos * xld + c;
    const uint16_t* __restrict__ rc = resid ? resid + (size_t)bos * rld + c : nullptr;
    uint16_t* __restrict__ dst = cs ? cs + (size_t)slot * cs_slot + (size_t)c * cs_ch : nullptr;
    const uint16_t* __restrict__ hc = warm ? dst + off * cs_tok : nullptr;
    uint16_t* __restrict__ yc = y + (size_t)bos * yld + c;

    const float b0 = bias ? bf16_to_f32(bias[c]) : 0.0f;
    /* sconv_res's taps: this channel's head's row of dyn, where column head*k + lag is lag's tap. */
    const uint16_t* __restrict__ dc = SCONV && dyn ? dyn + (size_t)bos * dld + (c / hch) * wid : nullptr;
    for (int32_t t = t0; t < t1; ++t) {
        float acc = b0;
#pragma unroll
        for (uint32_t j = 0; j < kPcMaxW; ++j) {
            if (j < wid) {
                const int32_t src = t - (int32_t)hist + (int32_t)(j * dil);
                const float v = src >= 0 ? bf16_to_f32(xc[(size_t)src * xld])
                                         : (hc ? bf16_to_f32(hc[(src + (int32_t)hist) * cs_tok]) : 0.0f);
                float tap = wt[j];
                if (SCONV && dc) tap += bf16_to_f32(dc[(size_t)t * dld + (wid - 1 - j)]);
                acc += v * tap;
            }
        }
        if (SCONV) {
            const float xr = bf16_to_f32(xc[(size_t)t * xld]);
            const uint16_t zb = f32_to_bf16((xr + acc) / sdiv);
            if (rc)
                yc[(size_t)t * yld] = f32_to_bf16(bf16_to_f32(rc[(size_t)t * rld]) + bf16_to_f32(zb));
            else
                yc[(size_t)t * yld] = zb;
            continue;
        }
        const float r = rc ? bf16_to_f32(rc[(size_t)t * rld]) : 0.0f;
        yc[(size_t)t * yld] = f32_to_bf16(r + (SILU ? pc_silu(acc) : acc));
    }

    /* The window this step leaves, in the same launch with no barrier. History is read only where
     * `t - hist + j*dilation < 0`, i.e. by tokens below `hist` (9, against a 32-token tile), so
     * tile 0 is the only block touching state[slot][c]. One thread owns the whole channel there, so
     * its reads precede its writes in program order. At decode the op's cost is the dispatch itself
     * (flat from M=1 to M=8), so a second launch would roughly double it.
     *
     * The layout is libref/ref_ple.cpp's, chosen by `num_accepted`. Present, this is a decode step
     * whose drafts a verify may reject: the history read shifts down by one and every input of the
     * step follows it, slot i holding position i + 1 of (window read ++ step), so a next step that
     * kept k tokens reads at k - 1. Absent, every token commits and the last `hist` inputs land at
     * offset zero. The two agree at one token. Slots past `state_len` are not written; the manager
     * sizes the window hist + n_spec, enough for a step of 1 + n_spec tokens.
     *
     * Values are copied as bf16 bits, the same tensor's bytes on both sides. The step's rows come
     * from the read-only `x`, so other tiles reading them is no hazard. Every history value is
     * loaded before the first store. */
    if (!dst || blockIdx.y != 0) return;
    uint16_t keep[kPcMaxHist];
    if (nacc) {
#pragma unroll
        for (uint32_t i = 0; i + 1 < kPcMaxHist; ++i)
            if (i + 1 < hist) keep[i] = hc ? hc[(size_t)(i + 1) * cs_tok] : (uint16_t)0;
#pragma unroll
        for (uint32_t i = 0; i + 1 < kPcMaxHist; ++i)
            if (i + 1 < hist) dst[(size_t)i * cs_tok] = keep[i];
        for (int32_t t = 0; t < slen && (long long)hist - 1 + t < state_len; ++t)
            dst[(size_t)(hist - 1 + (uint32_t)t) * cs_tok] = xc[(size_t)t * xld];
        return;
    }
#pragma unroll
    for (uint32_t i = 0; i < kPcMaxHist; ++i) {
        if (i < hist) {
            const int32_t src = slen - (int32_t)hist + (int32_t)i;
            keep[i] = src >= 0 ? xc[(size_t)src * xld]
                               : (hc ? hc[(src + (int32_t)hist) * cs_tok] : (uint16_t)0);
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < kPcMaxHist; ++i)
        if (i < hist) dst[(size_t)i * cs_tok] = keep[i];
}

/* Vectorized width=4 dilation=1 sconv_res / sconv_res_add: 8 contiguous channels per thread,
 * keeping a 4-slot bf16 shift register in registers so each token reads 1 uint4 from x and writes
 * 1 uint4 to y (plus 1 uint4 read from resid if HAS_RESID), bit-identical to ple_conv_body. */
template <bool HAS_DYN, bool HAS_RESID>
__device__ __forceinline__ void sconv_res_w4_vec8_body(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    uint16_t* __restrict__ cs, const int32_t* __restrict__ cu,
    const uint16_t* __restrict__ resid, const int32_t* __restrict__ sidx,
    const int32_t* __restrict__ hini, const int32_t* __restrict__ nacc,
    uint16_t* __restrict__ y, uint32_t n,
    uint32_t xld, uint32_t rld, uint32_t yld, uint32_t wld,
    long long cs_slot, long long cs_ch, long long cs_tok, long long state_len,
    const uint16_t* __restrict__ dyn, uint32_t dld, uint32_t hch, float sdiv) {
    const uint32_t c8 = (blockIdx.x * kPcW4Thr + threadIdx.x) * 8u;
    if (c8 >= n) return;
    const uint32_t s = blockIdx.z;
    const int32_t bos = cu[s];
    const int32_t slen = cu[s + 1] - bos;
    if (slen <= 0) return;
    const int32_t t0 = (int32_t)(blockIdx.y * kPcBT);
    if (t0 >= slen) return;
    const int32_t t1 = min(t0 + (int32_t)kPcBT, slen);

    const int32_t slot = sidx ? sidx[s] : (int32_t)s;
    if (slot < 0) return;
    const bool warm = cs && (!hini || hini[s] != 0);
    const long long off = nacc ? (long long)nacc[s] - 1 : 0;

    float wt[8][4];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint2 wv = *(const uint2*)(w + (size_t)(c8 + k) * wld);
        wt[k][0] = bf16_to_f32((uint16_t)(wv.x & 0xffffu));
        wt[k][1] = bf16_to_f32((uint16_t)(wv.x >> 16));
        wt[k][2] = bf16_to_f32((uint16_t)(wv.y & 0xffffu));
        wt[k][3] = bf16_to_f32((uint16_t)(wv.y >> 16));
    }

    const uint16_t* __restrict__ xc = x + (size_t)bos * xld + c8;
    const uint16_t* __restrict__ rc = HAS_RESID ? resid + (size_t)bos * rld + c8 : nullptr;
    uint16_t* __restrict__ dst = cs ? cs + (size_t)slot * cs_slot + (size_t)c8 * cs_ch : nullptr;
    const uint16_t* __restrict__ hc = warm ? dst + off * cs_tok : nullptr;
    uint16_t* __restrict__ yc = y + (size_t)bos * yld + c8;
    const uint16_t* __restrict__ dc = HAS_DYN ? dyn + (size_t)bos * dld + (c8 / hch) * 4u : nullptr;

    auto ld_u4 = [](const void* p) -> uint4 {
#ifdef __HIP__
        return *(const uint4*)p;
#else
        uint4 v;
        __builtin_memcpy(&v, p, sizeof v);
        return v;
#endif
    };
    auto st_u4 = [](void* p, uint4 v) {
#ifdef __HIP__
        *(uint4*)p = v;
#else
        __builtin_memcpy(p, &v, sizeof v);
#endif
    };
    auto ld_hist = [&](int32_t src) -> uint4 {
        if (src >= 0) return ld_u4(xc + (size_t)src * xld);
        if (!hc) return make_uint4(0u, 0u, 0u, 0u);
        uint16_t h[8];
        const size_t ho = (size_t)(src + 3) * (size_t)cs_tok;
#pragma unroll
        for (int k = 0; k < 8; ++k) h[k] = hc[(size_t)k * (size_t)cs_ch + ho];
        return make_uint4((uint32_t)h[0] | ((uint32_t)h[1] << 16),
                          (uint32_t)h[2] | ((uint32_t)h[3] << 16),
                          (uint32_t)h[4] | ((uint32_t)h[5] << 16),
                          (uint32_t)h[6] | ((uint32_t)h[7] << 16));
    };

    uint4 s0 = ld_hist(t0 - 3);
    uint4 s1 = ld_hist(t0 - 2);
    uint4 s2 = ld_hist(t0 - 1);
    const float inv_sdiv = 1.0f / sdiv;
    (void)inv_sdiv;

    for (int32_t t = t0; t < t1; ++t) {
        const uint4 s3 = ld_u4(xc + (size_t)t * xld);
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        if constexpr (HAS_DYN) {
            const uint2 dv = *(const uint2*)(dc + (size_t)t * dld);
            d0 = bf16_to_f32((uint16_t)(dv.x & 0xffffu));
            d1 = bf16_to_f32((uint16_t)(dv.x >> 16));
            d2 = bf16_to_f32((uint16_t)(dv.y & 0xffffu));
            d3 = bf16_to_f32((uint16_t)(dv.y >> 16));
        }
        const uint32_t p0[4] = { s0.x, s0.y, s0.z, s0.w };
        const uint32_t p1[4] = { s1.x, s1.y, s1.z, s1.w };
        const uint32_t p2[4] = { s2.x, s2.y, s2.z, s2.w };
        const uint32_t p3[4] = { s3.x, s3.y, s3.z, s3.w };
        uint32_t rv_arr[4];
        if constexpr (HAS_RESID) {
            const uint4 rv = ld_u4(rc + (size_t)t * rld);
            rv_arr[0] = rv.x; rv_arr[1] = rv.y; rv_arr[2] = rv.z; rv_arr[3] = rv.w;
        }
        uint32_t out[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int k0 = 2 * p, k1 = 2 * p + 1;
            const float x0_a = bf16w_lo(p0[p]), x1_a = bf16w_lo(p1[p]),
                        x2_a = bf16w_lo(p2[p]), x3_a = bf16w_lo(p3[p]);
            const float x0_b = bf16w_hi(p0[p]), x1_b = bf16w_hi(p1[p]),
                        x2_b = bf16w_hi(p2[p]), x3_b = bf16w_hi(p3[p]);
            float acc_a = 0.0f, acc_b = 0.0f;
            if constexpr (HAS_DYN) {
                acc_a += x0_a * (wt[k0][0] + d3) + x1_a * (wt[k0][1] + d2) +
                         x2_a * (wt[k0][2] + d1) + x3_a * (wt[k0][3] + d0);
                acc_b += x0_b * (wt[k1][0] + d3) + x1_b * (wt[k1][1] + d2) +
                         x2_b * (wt[k1][2] + d1) + x3_b * (wt[k1][3] + d0);
            } else {
                acc_a += x0_a * wt[k0][0] + x1_a * wt[k0][1] +
                         x2_a * wt[k0][2] + x3_a * wt[k0][3];
                acc_b += x0_b * wt[k1][0] + x1_b * wt[k1][1] +
                         x2_b * wt[k1][2] + x3_b * wt[k1][3];
            }
            const uint16_t za = f32_to_bf16((x3_a + acc_a) / sdiv);
            const uint16_t zb = f32_to_bf16((x3_b + acc_b) / sdiv);
            if constexpr (HAS_RESID) {
                const uint16_t ya = f32_to_bf16(bf16w_lo(rv_arr[p]) + bf16_to_f32(za));
                const uint16_t yb = f32_to_bf16(bf16w_hi(rv_arr[p]) + bf16_to_f32(zb));
                out[p] = (uint32_t)ya | ((uint32_t)yb << 16);
            } else {
                out[p] = (uint32_t)za | ((uint32_t)zb << 16);
            }
        }
        st_u4(yc + (size_t)t * yld, make_uint4(out[0], out[1], out[2], out[3]));
        s0 = s1; s1 = s2; s2 = s3;
    }

    if (!dst || blockIdx.y != 0) return;
    if (nacc) {
        uint16_t keep[8][2];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            keep[k][0] = hc ? hc[(size_t)k * (size_t)cs_ch + 1u * (size_t)cs_tok] : (uint16_t)0;
            keep[k][1] = hc ? hc[(size_t)k * (size_t)cs_ch + 2u * (size_t)cs_tok] : (uint16_t)0;
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            dst[(size_t)k * (size_t)cs_ch + 0u]              = keep[k][0];
            dst[(size_t)k * (size_t)cs_ch + (size_t)cs_tok]  = keep[k][1];
        }
        for (int32_t t = 0; t < slen && 2LL + t < state_len; ++t) {
            const uint4 xv = ld_u4(xc + (size_t)t * xld);
            const uint32_t xw[4] = { xv.x, xv.y, xv.z, xv.w };
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                dst[(size_t)(2 * p)     * (size_t)cs_ch + (size_t)(2 + t) * (size_t)cs_tok] = (uint16_t)(xw[p] & 0xffffu);
                dst[(size_t)(2 * p + 1) * (size_t)cs_ch + (size_t)(2 + t) * (size_t)cs_tok] = (uint16_t)(xw[p] >> 16);
            }
        }
        return;
    }
    uint4 k0v = ld_hist(slen - 3);
    uint4 k1v = ld_hist(slen - 2);
    uint4 k2v = ld_hist(slen - 1);
    const uint32_t kv0[4] = { k0v.x, k0v.y, k0v.z, k0v.w };
    const uint32_t kv1[4] = { k1v.x, k1v.y, k1v.z, k1v.w };
    const uint32_t kv2[4] = { k2v.x, k2v.y, k2v.z, k2v.w };
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        dst[(size_t)(2 * p)     * (size_t)cs_ch + 0u * (size_t)cs_tok] = (uint16_t)(kv0[p] & 0xffffu);
        dst[(size_t)(2 * p + 1) * (size_t)cs_ch + 0u * (size_t)cs_tok] = (uint16_t)(kv0[p] >> 16);
        dst[(size_t)(2 * p)     * (size_t)cs_ch + 1u * (size_t)cs_tok] = (uint16_t)(kv1[p] & 0xffffu);
        dst[(size_t)(2 * p + 1) * (size_t)cs_ch + 1u * (size_t)cs_tok] = (uint16_t)(kv1[p] >> 16);
        dst[(size_t)(2 * p)     * (size_t)cs_ch + 2u * (size_t)cs_tok] = (uint16_t)(kv2[p] & 0xffffu);
        dst[(size_t)(2 * p + 1) * (size_t)cs_ch + 2u * (size_t)cs_tok] = (uint16_t)(kv2[p] >> 16);
    }
}

template <bool HAS_DYN, bool HAS_RESID, class CH>
__global__ __launch_bounds__(kPcW4Thr)
void sconv_res_w4_vec8_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    uint16_t* __restrict__ cs, const int32_t* __restrict__ cu,
    const uint16_t* __restrict__ resid, const int32_t* __restrict__ sidx,
    const int32_t* __restrict__ hini, const int32_t* __restrict__ nacc,
    uint16_t* __restrict__ y, uint32_t n,
    uint32_t xld, uint32_t rld, uint32_t yld, uint32_t wld,
    long long cs_slot, long long cs_ch, long long cs_tok, long long state_len,
    const uint16_t* __restrict__ dyn, uint32_t dld, uint32_t hch, float sdiv) {
    if (CH::on) r4d_chain_wait();
    sconv_res_w4_vec8_body<HAS_DYN, HAS_RESID>(
        x, w, cs, cu, resid, sidx, hini, nacc, y, n,
        xld, rld, yld, wld, cs_slot, cs_ch, cs_tok, state_len, dyn, dld, hch, sdiv);
    if (CH::on) r4d_chain_done();
}

/* The kernel around the body, whose blocks return where their work ends: CH `r4dchain` is the
 * chained form (r4d_chain.h), taken at decode heights. */
template <bool SILU, bool SCONV, class CH>
__global__ __launch_bounds__(kPcThreads)
void ple_conv_kernel(PLE_CONV_PARAMS) {
    if (CH::on) r4d_chain_wait();
    ple_conv_body<SILU, SCONV>(x, w, cs, cu, resid, sidx, hini, nacc, y, bias, n, wid, dil, hist,
                               xld, rld, yld, wld, cs_slot, cs_ch, cs_tok, state_len, dyn, dld,
                               hch, sdiv);
    if (CH::on) r4d_chain_done();
}
#undef PLE_CONV_PARAMS

enum { PC_X = 0, PC_W, PC_CS, PC_CU, PC_RESID, PC_CIDX, PC_HINIT, PC_NACC, PC_Y };
/* ple_conv_bias: ple_conv's operands with `b`, the per-channel bias inside the silu (Mamba-2's
 * conv1d), before `y`. */
enum { PB_B = PC_Y, PB_Y };
/* causal_conv1d's operands: ple_conv's without `resid`. */
enum { CC_X = 0, CC_W, CC_CS, CC_CU, CC_CIDX, CC_HINIT, CC_NACC, CC_Y };
/* sconv_res's operands: causal_conv1d's with `dyn` before the optionals. */
enum { SC_X = 0, SC_W, SC_CS, SC_CU, SC_DYN, SC_CIDX, SC_HINIT, SC_NACC, SC_Y };

int conv_launch(const RadArgs* a, RadStream s, bool silu, const RadTensor* x, const RadTensor* w,
                const RadTensor* cs, const RadTensor* cu, const RadTensor* resid,
                const RadTensor* cidx, const RadTensor* hini, const RadTensor* nacc,
                const RadTensor* y, const RadTensor* b = nullptr,
                const RadTensor* dyn = nullptr, long long heads = 0, bool sconv = false) {
    if (!x || !w || !cu || !y) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || w->dtype != RAD_BF16 || y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (resid && resid->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (b && b->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (cs && cs->data && cs->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (cu->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (cidx && cidx->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    /* `has_init` must be int32 (RadBatch::ctx_lens, like every derived buffer): read as bytes, a
     * ctx_len of 512 has a zero low byte and the history would be dropped. Anything else is
     * refused. */
    if (hini && hini->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (nacc && nacc->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    const long long n   = r4d_geti_or(a, "n", 0);
    const long long wid = r4d_geti_or(a, "width", 0);
    const long long dil = r4d_geti_or(a, "dilation", 1);
    if (n <= 0 || wid <= 0 || dil <= 0) return R4D_NO(RAD_E_SHAPE);
    if (wid > kPcMaxW) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long hist = (wid - 1) * dil;
    if (hist > kPcMaxHist) return R4D_NO(RAD_E_UNSUPPORTED);

    long long T = 0, ty = 0, tr = 0, wrows = 0;
    const long long xld = r4d_rows_pitch(x, n, &T);
    const long long yld = r4d_rows_pitch(y, n, &ty);
    const long long rld = resid ? r4d_rows_pitch(resid, n, &tr) : 0;
    const long long wld = r4d_rows_pitch(w, wid, &wrows);
    if (xld < 0 || yld < 0 || wld < 0 || (resid && rld < 0)) return R4D_NO(RAD_E_STRIDE);
    /* The row count comes from the operands, and they must agree: `M` is ranged, so the arg holds
     * the band's upper bound and a grid sized by it would run off the arena. */
    if (T <= 0 || ty != T || (resid && tr != T)) return R4D_NO(RAD_E_SHAPE);
    if (wrows < n) return R4D_NO(RAD_E_SHAPE);
    if (b && (r4d_numel(b) != n || !r4d_contig(b))) return R4D_NO(RAD_E_SHAPE);

    /* sconv_res's dyn: rows of heads*k, the heads dividing the channels, bf16 like the rest. */
    long long dld = 0;
    if (sconv && dyn) {
        if (dyn->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
        if (heads <= 0 || n % heads != 0) return R4D_NO(RAD_E_SHAPE);
        long long tdy = 0;
        dld = r4d_rows_pitch(dyn, heads * wid, &tdy);
        if (dld < 0) return R4D_NO(RAD_E_STRIDE);
        if (tdy != T) return R4D_NO(RAD_E_SHAPE);
    }
    const uint32_t hch = dyn ? (uint32_t)(n / heads) : 1;
    const float sdiv = sqrtf((float)((dyn ? 2 : 1) * wid + 1));

    const long long N = r4d_numel(cu) - 1;
    if (N <= 0) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(cu)) return R4D_NO(RAD_E_STRIDE);

    /* [n_slots, n, state_len]. A state too shallow for the deepest tap is refused here; the
     * rollback offset cannot be checked without reading `num_accepted` on the host, which is why
     * the manager sizes the window hist + n_spec. */
    long long cs_slot = 0, cs_ch = 0, cs_tok = 0, state_len = 0;
    const bool have_cs = cs && cs->data && hist > 0;
    if (have_cs) {
        if (cs->rank != 3) return R4D_NO(RAD_E_SHAPE);
        if (r4d_dim(cs, 1) < n || r4d_dim(cs, 2) < hist) return R4D_NO(RAD_E_SHAPE);
        cs_slot   = r4d_stride(cs, 0);
        cs_ch     = r4d_stride(cs, 1);
        cs_tok    = r4d_stride(cs, 2);
        state_len = r4d_dim(cs, 2);
    }

    const uint32_t cb = (uint32_t)((n + kPcThreads - 1) / kPcThreads);
    const uint32_t tb = (uint32_t)((T + kPcBT - 1) / kPcBT);
    const bool chain = r4d_geti_or(a, "chain", 0) != 0 && T <= 64;
    const bool sconv_w4_vec8 = sconv && wid == 4 && dil == 1 && !b && (n % 8) == 0 &&
                               (!dyn || ((hch % 8) == 0 && rad_ptr_aligned(dyn->data, 8) && (dld % 4) == 0)) &&
                               rad_ptr_aligned(x->data, 16) && (xld % 8) == 0 &&
                               rad_ptr_aligned(y->data, 16) && (yld % 8) == 0 &&
                               (!resid || (rad_ptr_aligned(resid->data, 16) && (rld % 8) == 0));
    if (sconv_w4_vec8) {
        const uint32_t wcb = (uint32_t)(((n / 8) + kPcW4Thr - 1) / kPcW4Thr);
#define R4D_SC_V8(HD_, HR_, CH_)                                                                \
        hipLaunchKernelGGL((sconv_res_w4_vec8_kernel<HD_, HR_, CH_>), dim3(wcb, tb, (uint32_t)N), \
                           dim3(kPcW4Thr), 0, (hipStream_t)s,                                   \
                           (const uint16_t*)r4d_addr(x), (const uint16_t*)r4d_addr(w),           \
                           have_cs ? (uint16_t*)r4d_addr(cs) : nullptr,                          \
                           (const int32_t*)r4d_addr(cu),                                         \
                           resid ? (const uint16_t*)r4d_addr(resid) : nullptr,                   \
                           cidx ? (const int32_t*)r4d_addr(cidx) : nullptr,                      \
                           hini ? (const int32_t*)r4d_addr(hini) : nullptr,                      \
                           nacc ? (const int32_t*)r4d_addr(nacc) : nullptr,                      \
                           (uint16_t*)r4d_addr(y), (uint32_t)n,                                  \
                           (uint32_t)xld, (uint32_t)rld, (uint32_t)yld, (uint32_t)wld,           \
                           cs_slot, cs_ch, cs_tok, state_len,                                    \
                           dyn ? (const uint16_t*)r4d_addr(dyn) : nullptr, (uint32_t)dld, hch, sdiv)
        if (dyn) {
            if (resid) { if (chain) R4D_SC_V8(true, true, r4dchain); else R4D_SC_V8(true, true, R4dUnchained); }
            else       { if (chain) R4D_SC_V8(true, false, r4dchain); else R4D_SC_V8(true, false, R4dUnchained); }
        } else {
            if (resid) { if (chain) R4D_SC_V8(false, true, r4dchain); else R4D_SC_V8(false, true, R4dUnchained); }
            else       { if (chain) R4D_SC_V8(false, false, r4dchain); else R4D_SC_V8(false, false, R4dUnchained); }
        }
#undef R4D_SC_V8
        return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
    }
#define R4D_PC(S_, SC_, CH_)                                                                    \
    hipLaunchKernelGGL((ple_conv_kernel<S_, SC_, CH_>), dim3(cb, tb, (uint32_t)N),              \
                       dim3(kPcThreads), 0, (hipStream_t)s,                                      \
                       (const uint16_t*)r4d_addr(x), (const uint16_t*)r4d_addr(w),               \
                       have_cs ? (uint16_t*)r4d_addr(cs) : nullptr,                              \
                       (const int32_t*)r4d_addr(cu),                                             \
                       resid ? (const uint16_t*)r4d_addr(resid) : nullptr,                       \
                       cidx ? (const int32_t*)r4d_addr(cidx) : nullptr,                          \
                       hini ? (const int32_t*)r4d_addr(hini) : nullptr,                          \
                       nacc ? (const int32_t*)r4d_addr(nacc) : nullptr,                          \
                       (uint16_t*)r4d_addr(y), b ? (const uint16_t*)r4d_addr(b) : nullptr,       \
                       (uint32_t)n, (uint32_t)wid, (uint32_t)dil,                                \
                       (uint32_t)hist, (uint32_t)xld, (uint32_t)rld, (uint32_t)yld, (uint32_t)wld, \
                       cs_slot, cs_ch, cs_tok, state_len,                                        \
                       dyn ? (const uint16_t*)r4d_addr(dyn) : nullptr, (uint32_t)dld, hch, sdiv)
    if (sconv) { if (chain) R4D_PC(false, true, r4dchain); else R4D_PC(false, true, R4dUnchained); }
    else if (silu) { if (chain) R4D_PC(true, false, r4dchain); else R4D_PC(true, false, R4dUnchained); }
    else           { if (chain) R4D_PC(false, false, r4dchain); else R4D_PC(false, false, R4dUnchained); }
#undef R4D_PC
    return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
}

}  // namespace

extern "C" int r4d_rad_ple_conv(const RadArgs* a, RadStream s) {
    return conv_launch(a, s, true, r4d_opt(a, PC_X), r4d_opt(a, PC_W), r4d_opt(a, PC_CS),
                       r4d_opt(a, PC_CU), r4d_opt(a, PC_RESID), r4d_opt(a, PC_CIDX),
                       r4d_opt(a, PC_HINIT), r4d_opt(a, PC_NACC), r4d_opt(a, PC_Y));
}

extern "C" int r4d_rad_ple_conv_bias(const RadArgs* a, RadStream s) {
    const RadTensor* b = r4d_opt(a, PB_B);
    if (!b) return R4D_NO(RAD_E_INVAL);
    return conv_launch(a, s, true, r4d_opt(a, PC_X), r4d_opt(a, PC_W), r4d_opt(a, PC_CS),
                       r4d_opt(a, PC_CU), r4d_opt(a, PC_RESID), r4d_opt(a, PC_CIDX),
                       r4d_opt(a, PC_HINIT), r4d_opt(a, PC_NACC), r4d_opt(a, PB_Y), b);
}

extern "C" int r4d_rad_causal_conv1d(const RadArgs* a, RadStream s) {
    if (r4d_geti_or(a, "dilation", 1) != 1) return R4D_NO(RAD_E_INVAL);
    return conv_launch(a, s, false, r4d_opt(a, CC_X), r4d_opt(a, CC_W), r4d_opt(a, CC_CS),
                       r4d_opt(a, CC_CU), nullptr, r4d_opt(a, CC_CIDX), r4d_opt(a, CC_HINIT),
                       r4d_opt(a, CC_NACC), r4d_opt(a, CC_Y));
}

/* Inkling 2's short convolution with its residual. Its dyn and heads must agree (libref refuses
 * the rest too); the window is causal_conv1d's, and the dilation is one by construction. */
extern "C" int r4d_rad_sconv_res(const RadArgs* a, RadStream s) {
    const long long heads = r4d_geti_or(a, "heads", 0);
    /* heads 0 means no dyn: the operand is ignored even when it is filled. */
    const RadTensor* dyn = heads > 0 ? r4d_opt(a, SC_DYN) : nullptr;
    if (heads > 0 && dyn == nullptr) return R4D_NO(RAD_E_INVAL);
    return conv_launch(a, s, false, r4d_opt(a, SC_X), r4d_opt(a, SC_W), r4d_opt(a, SC_CS),
                       r4d_opt(a, SC_CU), nullptr, r4d_opt(a, SC_CIDX), r4d_opt(a, SC_HINIT),
                       r4d_opt(a, SC_NACC), r4d_opt(a, SC_Y), nullptr, dyn, heads, true);
}

/* Device-only: sconv_res with a stream residual add (`y = add_resid + sconv_res(x)`). */
enum { SCA_ADD_RESID = 9 };
extern "C" int r4d_rad_sconv_res_add(const RadArgs* a, RadStream s) {
    const long long heads = r4d_geti_or(a, "heads", 0);
    const RadTensor* dyn = heads > 0 ? r4d_opt(a, SC_DYN) : nullptr;
    if (heads > 0 && dyn == nullptr) return R4D_NO(RAD_E_INVAL);
    const RadTensor* add_r = r4d_opt(a, SCA_ADD_RESID);
    if (!add_r) return R4D_NO(RAD_E_INVAL);
    return conv_launch(a, s, false, r4d_opt(a, SC_X), r4d_opt(a, SC_W), r4d_opt(a, SC_CS),
                       r4d_opt(a, SC_CU), add_r, r4d_opt(a, SC_CIDX), r4d_opt(a, SC_HINIT),
                       r4d_opt(a, SC_NACC), r4d_opt(a, SC_Y), nullptr, dyn, heads, true);
}
