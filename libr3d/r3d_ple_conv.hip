/* r3d_ple_conv.hip -- ple_conv: Per-Layer Embedding's dilated depthwise convolution, its silu, and
 * the residual add that finishes the layer.
 *
 *   y[t][c] = resid[t][c] + silu( sum_j w[c][j] * x[t - (width-1-j)*dilation][c] )
 *
 *   x [M, n] bf16, w [n, width] bf16, conv_state [n_slots, n, state_len] bf16, cu [N+1] i32,
 *   resid [M, n] bf16 (optional), cache_idx [N] i32 (optional), has_init [N] i32 (optional),
 *   num_accepted [N] i32 (optional), y [M, n] bf16 out.
 *
 * libref/ref_ple.cpp is the oracle and docs/OPS.md the contract. The taps are `dilation` apart, so
 * at width 4 and dilation 3 they sit 9, 6, 3 and 0 timesteps back and a sequence carries a history
 * of (width-1)*dilation = 9, not 3. An off-by-one in a convolution cache yields fluent output from
 * a slightly wrong context, so the indexing mirrors ref's line for line: state slot `i` holds
 * relative position `i - hist`, a read at `src < 0` becomes `cs[slot][c][off + src + hist]`, and
 * `has_init` 0 means that window is zeros, not the slot's stale contents.
 *
 * One thread per channel: a depthwise convolution never mixes channels, so only time must stay
 * together. A thread walks a tile of tokens; consecutive threads own consecutive channels, so a
 * wave loads one contiguous 64-byte slice of a token row and a workgroup a contiguous 512-byte one.
 *
 * The grid is (channel blocks, token tiles, sequences). Token tiles are counted from the batch's
 * total token count, an upper bound on any one sequence's; a short sequence's extra blocks find
 * `t0 >= slen` and return. A per-sequence count would mean reading `cu` on the host.
 *
 * The taps are read from memory, not kept in a shift register: the window would need `hist + 1`
 * registers indexed by a runtime value, which compiles to a select tree, and a compile-time window
 * would need a specialisation per (width, dilation). The re-reads hit a tile working set of
 * (BT + hist) * 512 bytes (21 KB at BT 32), and the op is bound by the three arrays it must touch
 * anyway. Nothing is staged in LDS: there is no reduction here, so the launcher's 60 KiB stage
 * check has no stage to size, and no wave barrier is needed for the write-back (see below).
 *
 * The window write-back is in the same launch; see the note at the end of the kernel. It relies
 * on `hist < kPcBT`, asserted below.
 *
 * RDNA3 notes. The kernel is the libr4d one (r4d_ple_conv_bf16.hip) with three adaptations:
 * every 16-bit read goes through ldg_u16, so it is a global load through the device address space
 * (measured on the box: the two builds emit the same 50 global_load_u16 / 50 global_load_d16_b16 /
 * 46 global_load_d16_hi_b16 / 2 global_load_b32, the d16 pairs being the write-back packing two
 * bf16 into one register); the kernel is annotated R3D_WAVES(16), the wave-slot cap of a gfx11
 * SIMD, not R3D_WAVES(8) for its own eight waves -- the op is bandwidth-bound and wants the most
 * resident waves it can get, and the 8 annotation costs registers (113 VGPRs against 28 at 16,
 * both spill-free, 28 being what the RDNA4 build of the same source allocates); and an optional
 * operand whose descriptor is present but unbound is read as absent (r3d_has), which is how the
 * kernel treats it. The compute is plain f32: the op moves bytes, and RDNA3 has no bf16 dot2 and
 * nothing here is a 16x16x16 tile, so the wave matrix would add a fragment layout for no
 * throughput (r3d_common.h).
 */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <algorithm>
#include <cstdint>

namespace {

enum {
    kPcThreads = 256,
    kPcWaves   = kPcThreads / kWave,   /* 8 waves of 32 a workgroup */
    kPcWpS     = 16,   /* waves a SIMD: gfx11's wave32 slot count, and the register budget (64) */
    kPcBT      = 32,   /* tokens a thread walks; the tile's working set is (BT + hist) * 512 B */
    kPcMaxW    = 8,    /* taps held in registers, so the weight loop unrolls to a static bound */
    kPcMaxHist = 16,   /* (width-1)*dilation, the write-back's register window; see cPleConv */
};

/* The write-back lives in tile 0, which is sound only while the deepest tap is inside the first
 * tile. At hist >= BT a second block would read history and the write-back would overwrite bytes
 * it is still reading, silently corrupting a sequence's first tokens. The constraint table caps
 * hist <= 16; the tile size is the other half of that bound. */
static_assert(kPcMaxHist < kPcBT, "the conv history must fit inside tile 0 for the fused write-back");

/* A bf16 of global memory widened in one instruction; see ldg_u16 in r3d_common.h. */
__device__ __forceinline__ float pc_ld(const uint16_t* p, size_t i) {
    return bf16_to_f32((uint16_t)ldg_u16(p + i));
}

/* expf rather than __expf: the op is bound by memory, not the transcendental, and the oracle uses
 * std::exp (ref_common.h's act_silu). The instruction-bound GDN conv makes the opposite trade. */
__device__ __forceinline__ float pc_silu(float z) { return z / (1.0f + expf(-z)); }

/* SILU is ple_conv's activation; causal_conv1d (LFM2's short convolution) is the same window with
 * no activation, no residual and dilation one, and shares this kernel. */
template <bool SILU>
__global__ __launch_bounds__(kPcThreads) R3D_WAVES(kPcWpS) void r3d_ple_conv_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w,
    uint16_t* __restrict__ cs, const int32_t* __restrict__ cu,
    const uint16_t* __restrict__ resid, const int32_t* __restrict__ sidx,
    const int32_t* __restrict__ hini, const int32_t* __restrict__ nacc,
    uint16_t* __restrict__ y, uint32_t n, uint32_t wid, uint32_t dil, uint32_t hist,
    uint32_t xld, uint32_t rld, uint32_t yld, uint32_t wld,
    long long cs_slot, long long cs_ch, long long cs_tok, long long state_len) {

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
    for (uint32_t j = 0; j < kPcMaxW; ++j) wt[j] = j < wid ? pc_ld(w, (size_t)c * wld + j) : 0.0f;

    const uint16_t* __restrict__ xc = x + (size_t)bos * xld + c;
    const uint16_t* __restrict__ rc = resid ? resid + (size_t)bos * rld + c : nullptr;
    uint16_t* __restrict__ dst = cs ? cs + (size_t)slot * cs_slot + (size_t)c * cs_ch : nullptr;
    const uint16_t* __restrict__ hc = warm ? dst + off * cs_tok : nullptr;
    uint16_t* __restrict__ yc = y + (size_t)bos * yld + c;

    for (int32_t t = t0; t < t1; ++t) {
        float acc = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < kPcMaxW; ++j) {
            if (j < wid) {
                const int32_t src = t - (int32_t)hist + (int32_t)(j * dil);
                const float v = src >= 0 ? pc_ld(xc, (size_t)src * xld)
                                         : (hc ? pc_ld(hc, (size_t)(src + (int32_t)hist) * cs_tok)
                                               : 0.0f);
                acc += v * wt[j];
            }
        }
        const float r = rc ? pc_ld(rc, (size_t)t * rld) : 0.0f;
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
            if (i + 1 < hist)
                keep[i] = hc ? (uint16_t)ldg_u16(hc + (size_t)(i + 1) * cs_tok) : (uint16_t)0;
#pragma unroll
        for (uint32_t i = 0; i + 1 < kPcMaxHist; ++i)
            if (i + 1 < hist) dst[(size_t)i * cs_tok] = keep[i];
        for (int32_t t = 0; t < slen && (long long)hist - 1 + t < state_len; ++t)
            dst[(size_t)(hist - 1 + (uint32_t)t) * cs_tok] = (uint16_t)ldg_u16(xc + (size_t)t * xld);
        return;
    }
#pragma unroll
    for (uint32_t i = 0; i < kPcMaxHist; ++i) {
        if (i < hist) {
            const int32_t src = slen - (int32_t)hist + (int32_t)i;
            keep[i] = src >= 0 ? (uint16_t)ldg_u16(xc + (size_t)src * xld)
                               : (hc ? (uint16_t)ldg_u16(hc + (size_t)(src + (int32_t)hist) * cs_tok)
                                     : (uint16_t)0);
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < kPcMaxHist; ++i)
        if (i < hist) dst[(size_t)i * cs_tok] = keep[i];
}

/* The operand slots, in ple_conv's schema order. causal_conv1d is this kernel's SILU=false arm
 * with `resid` out of the schema and `dilation` fixed at one; its slots are named beside its
 * entry point at the end of this unit. */
enum { PC_X = 0, PC_W, PC_CS, PC_CU, PC_RESID, PC_CIDX, PC_HINIT, PC_NACC, PC_Y };

/* The launcher: the shape checks libr4d's r4d_rad_ple_conv makes, in the same order with the same
 * refusals. `silu` selects the activation. */
int r3d_conv_launch(const RadArgs* a, RadStream s, bool silu, const RadTensor* x, const RadTensor* w,
                    const RadTensor* cs, const RadTensor* cu, const RadTensor* resid,
                    const RadTensor* cidx, const RadTensor* hini, const RadTensor* nacc,
                    const RadTensor* y) {
    if (!r3d_p(x) || !r3d_p(w) || !r3d_p(cu) || !r3d_wp(y)) return R3D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || w->dtype != RAD_BF16 || y->dtype != RAD_BF16)
        return R3D_NO(RAD_E_DTYPE);
    /* An optional operand the engine bound to nothing is absent, so it is tested with r3d_has:
     * a descriptor with a null `data` carries the schema's dtype, not a tensor's. */
    if (r3d_has(resid) && resid->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(cs) && cs->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (cu->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(cidx) && cidx->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);
    /* `has_init` must be int32 (RadBatch::ctx_lens, like every derived buffer): read as bytes, a
     * ctx_len of 512 has a zero low byte and the history would be dropped. Anything else is
     * refused. */
    if (r3d_has(hini) && hini->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(nacc) && nacc->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);

    const long long n   = r3d_geti_or(a, "n", 0);
    const long long wid = r3d_geti_or(a, "width", 0);
    const long long dil = r3d_geti_or(a, "dilation", 1);
    if (n <= 0 || wid <= 0 || dil <= 0) return R3D_NO(RAD_E_SHAPE);
    if (wid > kPcMaxW) return R3D_NO(RAD_E_UNSUPPORTED);
    const long long hist = (wid - 1) * dil;
    if (hist > kPcMaxHist) return R3D_NO(RAD_E_UNSUPPORTED);

    long long T = 0, ty = 0, tr = 0, wrows = 0;
    const long long xld = r3d_rows_pitch(x, n, &T);
    const long long yld = r3d_rows_pitch(y, n, &ty);
    const bool have_r = r3d_has(resid);
    const long long rld = have_r ? r3d_rows_pitch(resid, n, &tr) : 0;
    const long long wld = r3d_rows_pitch(w, wid, &wrows);
    if (xld < 0 || yld < 0 || wld < 0 || (have_r && rld < 0)) return R3D_NO(RAD_E_STRIDE);
    /* The row count comes from the operands, and they must agree: `M` is ranged, so the arg holds
     * the band's upper bound and a grid sized by it would run off the arena. */
    if (T <= 0 || ty != T || (have_r && tr != T)) return R3D_NO(RAD_E_SHAPE);
    if (wrows < n) return R3D_NO(RAD_E_SHAPE);

    const long long N = r3d_numel(cu) - 1;
    if (N <= 0) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(cu)) return R3D_NO(RAD_E_STRIDE);

    /* [n_slots, n, state_len]. A state too shallow for the deepest tap is refused here; the
     * rollback offset cannot be checked without reading `num_accepted` on the host, which is why
     * the manager sizes the window hist + n_spec. */
    long long cs_slot = 0, cs_ch = 0, cs_tok = 0, state_len = 0;
    const bool have_cs = r3d_has(cs) && hist > 0;
    if (have_cs) {
        if (cs->rank != 3) return R3D_NO(RAD_E_SHAPE);
        if (r3d_dim(cs, 1) < n || r3d_dim(cs, 2) < hist) return R3D_NO(RAD_E_SHAPE);
        cs_slot   = r3d_stride(cs, 0);
        cs_ch     = r3d_stride(cs, 1);
        cs_tok    = r3d_stride(cs, 2);
        state_len = r3d_dim(cs, 2);
    }

    const uint32_t cb = (uint32_t)((n + kPcThreads - 1) / kPcThreads);
    const uint32_t tb = (uint32_t)((T + kPcBT - 1) / kPcBT);
    const auto kernel = silu ? r3d_ple_conv_kernel<true> : r3d_ple_conv_kernel<false>;
    R3D_KLAUNCH(R3D_KNAME(kernel), dim3(cb, tb, (uint32_t)N), dim3(kPcThreads), (hipStream_t)s,
                (const uint16_t*)r3d_p(x), (const uint16_t*)r3d_p(w),
                have_cs ? (uint16_t*)r3d_wp(cs) : nullptr,
                (const int32_t*)r3d_p(cu),
                have_r ? (const uint16_t*)r3d_p(resid) : nullptr,
                r3d_has(cidx) ? (const int32_t*)r3d_p(cidx) : nullptr,
                r3d_has(hini) ? (const int32_t*)r3d_p(hini) : nullptr,
                r3d_has(nacc) ? (const int32_t*)r3d_p(nacc) : nullptr,
                (uint16_t*)r3d_wp(y), (uint32_t)n, (uint32_t)wid, (uint32_t)dil,
                (uint32_t)hist, (uint32_t)xld, (uint32_t)rld, (uint32_t)yld, (uint32_t)wld,
                cs_slot, cs_ch, cs_tok, state_len);
    return hipGetLastError() == hipSuccess ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

}  // namespace

extern "C" int r3d_rad_ple_conv(const RadArgs* a, RadStream s) {
    return r3d_conv_launch(a, s, true, r3d_opt(a, PC_X), r3d_opt(a, PC_W), r3d_opt(a, PC_CS),
                           r3d_opt(a, PC_CU), r3d_opt(a, PC_RESID), r3d_opt(a, PC_CIDX),
                           r3d_opt(a, PC_HINIT), r3d_opt(a, PC_NACC), r3d_opt(a, PC_Y));
}

/* causal_conv1d (LFM2's and LFM3's short convolution): the SILU=false arm, no residual, and
 * dilation one, which the schema has no parameter for, so the launcher's default holds. The
 * operand slots are libref's causal_conv1d schema order: ple_conv's without `resid`. */
enum { CC_X = 0, CC_W, CC_CS, CC_CU, CC_CIDX, CC_HINIT, CC_NACC, CC_Y };

extern "C" int r3d_rad_causal_conv1d(const RadArgs* a, RadStream s) {
    if (r3d_geti_or(a, "dilation", 1) != 1) return R3D_NO(RAD_E_INVAL);
    return r3d_conv_launch(a, s, false, r3d_opt(a, CC_X), r3d_opt(a, CC_W), r3d_opt(a, CC_CS),
                           r3d_opt(a, CC_CU), nullptr, r3d_opt(a, CC_CIDX), r3d_opt(a, CC_HINIT),
                           r3d_opt(a, CC_NACC), r3d_opt(a, CC_Y));
}
