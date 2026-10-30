/* Derived from libr4d/r4d_moe.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_moe.hip -- the MoE routing machinery: the stable counting sort that turns per-token expert
// ids into the expert-sorted slot list (moe_scatter, moe_scatter_inv), and the weighted gather back
// to token order (moe_gather, moe_gather_inv).
//
// Ported from libr4d/r4d_moe.hip's scatter and gather sections. NONE of it is WMMA code: the sort
// is a one-workgroup counting sort over LDS and the gather is a per-element weighted sum, so both
// are architecture-neutral and port as they stand. What differs from libr4d is only the plumbing
// this library spells differently (halo_emit instead of hipLaunchKernelGGL, halo_opt/halo_rows_pitch
// instead of r4d_opt/r4d_rows_pitch) and the two shared headers the kernels live in
// (halo_moe_sort.h, halo_moe_gather.h), which are this library's copy of libr4d's.
//
// Determinism (the reason the sort is not an atomic cursor): a slot's position is
// `expert_offset[e] + blocks_before[b][e] + rank_within_block`, all computed by reduction, so
// sorted_tok -- and every row of the grouped GEMM downstream -- is independent of scheduling.
//
// ---- the two forms of each op -------------------------------------------------------------
//
// moe_scatter: one workgroup for T <= kScatterSmall (1024 slots: every decode and verify step), and
// a three-pass form (per-tile histogram, scan, placement) above it, which needs the caller's
// scratch. Both are stable, so both give the same bytes.
//
// moe_gather: a column a thread (gather_kernel), a wave per token with 16 B a lane
// (gather_wide_kernel) past the self-scan bound, and the inverse permutation either built in the
// kernel itself (T <= kGatherSelfT), by two prepass kernels in scratch, or handed in
// (moe_gather_inv). Every form sums a token's slots in ascending slot order with the same f32
// arithmetic, so all of them are bit-identical; halo_selftest checks them against each other.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "halo_plugin.h"
#include "halo_launch.h"
#include "halo_emit.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_moe_sort.h"
#include "halo_moe_gather.h"

namespace halo_moe {

using halo_moesort::kMaxExpert;
using halo_moesort::Protect;
using halo_moesort::sort_place;
using halo_moesort::protect_of;
using halo_moesort::kScatterSmall;
using halo_moesort::ScatterLds;
using halo_moesort::wave_incl_scan_i32;
using halo_moesort::scatter_small_place;
using halo_moesort::kScatWaves;
using halo_moesort::scatter_small_kernel;
using halo_moegather::kGatherSlots;
using halo_moegather::kGatherSelfT;
using halo_moegather::gather_kernel;

constexpr uint32_t kWave = 32;

#define HALO_MOE_CHK() do { hipError_t e_ = hipGetLastError();                       \
                            if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

/* ================================================================== the general scatter
 *
 * Three passes so the sort stays stable without an atomic deciding order. `bcnt` is
 * [n_tile, n_expert] in the caller's scratch.
 *
 * Ported verbatim: the histogram, the per-expert scan across tiles (a thread's column of `bcnt` is
 * loaded kScanBatch at a time, since one at a time it would be a dependent round trip per tile) and
 * the placement (rank within the tile from a serial LDS scan, the price of stability). */
constexpr uint32_t kScatterTile = 256;

__global__ __launch_bounds__(kScatterTile) void scatter_hist_kernel(
    const int32_t* __restrict__ ids, int32_t* __restrict__ bcnt, uint32_t T, uint32_t n_expert,
    int32_t base, Protect prot) {
    extern __shared__ int32_t s_h[];
    const uint32_t tid = threadIdx.x, b = blockIdx.x;
    for (uint32_t e = tid; e < n_expert; e += blockDim.x) s_h[e] = 0;
    __syncthreads();
    const uint32_t i = b * kScatterTile + tid;
    if (i < T) {
        const int32_t e = ids[i] - base;
        if (e >= 0 && e < (int32_t)n_expert) atomicAdd(&s_h[sort_place(e, n_expert, prot)], 1);
    }
    __syncthreads();
    for (uint32_t e = tid; e < n_expert; e += blockDim.x)
        bcnt[(size_t)b * n_expert + e] = s_h[e];
}

constexpr uint32_t kScanBatch = 16;

__global__ __launch_bounds__(kMaxExpert) void scatter_scan_kernel(
    int32_t* __restrict__ bcnt, int32_t* __restrict__ off, int32_t* __restrict__ cnt,
    uint32_t n_tile, uint32_t n_expert) {
    constexpr uint32_t NW = kMaxExpert / kWave;
    __shared__ int32_t s_wt[NW];
    const uint32_t e = threadIdx.x, lane = e % kWave, wave = e / kWave;
    int32_t run = 0;
    if (e < n_expert) {
        for (uint32_t b0 = 0; b0 < n_tile; b0 += kScanBatch) {
            /* Every load in bounds (past the end re-reads the last row, discarded), so no load sits
             * in a branch. */
            int32_t v[kScanBatch];
#pragma unroll
            for (uint32_t k = 0; k < kScanBatch; ++k) {
                const uint32_t b = b0 + k < n_tile ? b0 + k : n_tile - 1u;
                v[k] = bcnt[(size_t)b * n_expert + e];
            }
#pragma unroll
            for (uint32_t k = 0; k < kScanBatch; ++k) {
                if (b0 + k >= n_tile) break;
                bcnt[(size_t)(b0 + k) * n_expert + e] = run;
                run += v[k];
            }
        }
        cnt[e] = run;
    }
    const int32_t x = wave_incl_scan_i32(run, lane);
    if (lane == kWave - 1) s_wt[wave] = x;
    lds_barrier();
    int32_t below = 0, tot = 0;
#pragma unroll
    for (uint32_t w = 0; w < NW; ++w) {
        const int32_t c = s_wt[w];
        below += (w < wave) ? c : 0;
        tot   += c;
    }
    if (e < n_expert) off[e] = below + x - run;     /* inclusive -> exclusive */
    if (e == 0) off[n_expert] = tot;
}

/* Placement and tail. Each routed slot goes to its expert's offset plus the expert's count in
 * earlier tiles plus its rank in this tile. Sorted rows past the live count are -1; no placement
 * lands there (all placed slots are below `off[n_expert]`), so each thread writes its own index
 * without races. INV also writes each slot's row, -1 when dropped. */
template <bool INV>
__global__ __launch_bounds__(kScatterTile) void scatter_place_kernel(
    const int32_t* __restrict__ ids, const int32_t* __restrict__ bcnt,
    const int32_t* __restrict__ off, int32_t* __restrict__ sorted,
    uint32_t T, uint32_t n_expert, int32_t base, Protect prot, int32_t* __restrict__ inv) {
    __shared__ int32_t s_id[kScatterTile];
    const uint32_t tid = threadIdx.x, b = blockIdx.x;
    const uint32_t i = b * kScatterTile + tid;
    /* Everything works in sort places (local index, protected experts last), the indices the
     * offsets and weight tables use. */
    const int32_t  l = (i < T) ? ids[i] - base : -1;
    const int32_t  e = (l >= 0 && l < (int32_t)n_expert) ? sort_place(l, n_expert, prot) : -1;
    s_id[tid] = e;
    const uint32_t live = (uint32_t)off[n_expert];
    if (i >= live && i < T) sorted[i] = -1;
    __syncthreads();
    if constexpr (INV) if (i < T && e < 0) inv[i] = -1;
    if (i >= T || e < 0) return;
    int32_t rank = 0;
    for (uint32_t j = 0; j < tid; ++j) rank += (s_id[j] == e);
    const int32_t at = off[e] + bcnt[(size_t)b * n_expert + e] + rank;
    sorted[at] = (int32_t)i;
    if constexpr (INV) inv[i] = at;
}

/* ================================================================== the gather's prepass
 *
 * `sorted` permutes the live slots, so `inv[sorted[i]] = i` is one scattered store per row. An
 * unclaimed (dropped) slot keeps -1 and contributes nothing. Two passes: the clear cannot be fused
 * into the scatter because no thread knows which entries nobody will write. */
__global__ void gather_inv_kernel(int32_t* __restrict__ inv, uint32_t T) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T) inv[i] = -1;
}

__global__ void gather_scatter_kernel(const int32_t* __restrict__ sorted,
                                      int32_t* __restrict__ inv, uint32_t T) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T) return;
    const int32_t f = sorted[i];
    if (f >= 0 && f < (int32_t)T) inv[f] = (int32_t)i;
}

/* ---- prefill gather: a wave per token, eight columns per lane ---------------------------------
 *
 * At prefill this op reads `ye` (105 MB per layer at a 2048 chunk of Qwen3.8-Flash-Next on four
 * ranks), and gather_kernel moves two bytes per lane with slot loads serialised on the slot table.
 * Here a wave owns one token and 256 columns, eight per lane: the slot table is read first (scalar
 * loads), then every slot's sixteen bytes are issued before any is used. Arithmetic is
 * gather_kernel's (ascending slot order, dropped slots skipped, f32 `acc += w * x`, the shared arm's
 * two narrows), so the bits match.
 *
 * Architecture-neutral, and it ports as it stands: the slot broadcast is readlane (available on
 * both parts), not a WMMA fragment or an LDS tile. */
constexpr uint32_t kGatherWideSlots = 16;      /* top_k at most this: the slots live in registers */
constexpr uint32_t kGatherWideCols  = 8;       /* columns a lane, one sixteen-byte load */
static_assert(kGatherWideSlots <= kWave, "slot j is read off lane j");

template <uint32_t NW>
__global__ __launch_bounds__(NW * kWave) void gather_wide_kernel(
    const uint16_t* __restrict__ ye, const uint16_t* __restrict__ ew,
    const int32_t* __restrict__ inv, uint16_t* __restrict__ y,
    uint32_t M, uint32_t n, uint32_t top_k, uint32_t ye_ld, uint32_t ew_ld, uint32_t y_ld,
    const uint16_t* __restrict__ sh, const uint16_t* __restrict__ sg,
    uint32_t sh_ld, uint32_t sg_ld, int sig) {
    const uint32_t lane = threadIdx.x % kWave;
    const uint32_t wave = (uint32_t)__builtin_amdgcn_readfirstlane((int)(threadIdx.x / kWave));
    const uint32_t t = blockIdx.x * NW + wave;
    const uint32_t c = (blockIdx.y * kWave + lane) * kGatherWideCols;
    /* No barrier below, so a wave past the last token simply leaves. */
    if (t >= M) return;

    /* Every lane takes part (a lane past the last column leaves only afterwards), since values are
     * read off any lane. */
    const uint32_t jl = lane < top_k ? lane : 0u;
    const int32_t  il = inv[(size_t)t * top_k + jl];
    const int32_t  wl = (int32_t)ew[(size_t)t * ew_ld + jl];
    int32_t si[kGatherWideSlots];
    float   sw[kGatherWideSlots];
#pragma unroll
    for (uint32_t j = 0; j < kGatherWideSlots; ++j) {
        si[j] = __builtin_amdgcn_readlane(il, (int)j);
        sw[j] = bf16_to_f32((uint16_t)__builtin_amdgcn_readlane(wl, (int)j));
    }
    if (c >= n) return;
    /* No break or continue in these loops: an early exit keeps the slot loop rolled and `yv` goes
     * to scratch. Uniform branches over a constant trip count keep every slot in a register. */
    bool  lv[kGatherWideSlots];
    uint4 yv[kGatherWideSlots];
#pragma unroll
    for (uint32_t j = 0; j < kGatherWideSlots; ++j) {
        lv[j] = j < top_k && si[j] >= 0;
        yv[j] = make_uint4(0, 0, 0, 0);
        if (lv[j]) yv[j] = *(const uint4*)(ye + (size_t)si[j] * ye_ld + c);
    }
    /* The shared arm after the slots, so its loads do not delay theirs. */
    float gate = 0.0f;
    uint4 shv  = make_uint4(0, 0, 0, 0);
    if (sh) {
        const float v0 = bf16_to_f32(sg[(size_t)t * sg_ld]);
        gate = sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
        shv  = *(const uint4*)(sh + (size_t)t * sh_ld + c);
    }

    float acc[kGatherWideCols];
#pragma unroll
    for (uint32_t q = 0; q < kGatherWideCols; ++q) acc[q] = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kGatherWideSlots; ++j) {
        if (lv[j]) {
            const float    wj   = sw[j];
            const uint32_t d[4] = { yv[j].x, yv[j].y, yv[j].z, yv[j].w };
#pragma unroll
            for (uint32_t q = 0; q < 4; ++q) {
                acc[2 * q]     += wj * bf16_to_f32((uint16_t)(d[q] & 0xFFFFu));
                acc[2 * q + 1] += wj * bf16_to_f32((uint16_t)(d[q] >> 16));
            }
        }
    }

    const uint32_t s[4] = { shv.x, shv.y, shv.z, shv.w };
    uint32_t o[4];
#pragma unroll
    for (uint32_t q = 0; q < 4; ++q) {
        uint16_t lo = f32_to_bf16(acc[2 * q]), hi = f32_to_bf16(acc[2 * q + 1]);
        if (sh) {
            const uint16_t plo = f32_to_bf16(bf16_to_f32((uint16_t)(s[q] & 0xFFFFu)) * gate);
            const uint16_t phi = f32_to_bf16(bf16_to_f32((uint16_t)(s[q] >> 16)) * gate);
            lo = f32_to_bf16(bf16_to_f32(plo) + bf16_to_f32(lo));
            hi = f32_to_bf16(bf16_to_f32(phi) + bf16_to_f32(hi));
        }
        o[q] = (uint32_t)lo | ((uint32_t)hi << 16);
    }
    *(uint4*)(y + (size_t)t * y_ld + c) = make_uint4(o[0], o[1], o[2], o[3]);
}

/* ================================================================== the plugin entries */

/* moe_gather's form, pinnable for tests: 0 the launcher's rule, 1 gather_kernel (a column per
 * thread), 2 gather_wide_kernel (a wave per token). The rule takes 2 past the self-scan bound where
 * the shape allows; an impossible pin is refused. Both are bit-identical (halo_selftest). */
static int g_moe_gather_pin = 0;

}  /* namespace halo_moe */

extern "C" void halo_moe_gather_pin_form(int form) { halo_moe::g_moe_gather_pin = form; }

namespace halo_moe {

/* Scratch for the general scatter: the per-tile histogram plane (none for the small form). Sized
 * for the shape asked about, since the resolved band sizes the arena. */
static int scatter_dims(const RadArgs* a, long long* T, long long* n_expert) {
    const RadTensor* ids = halo_opt(a, 0);
    *n_expert = halo_geti_or(a, "n_expert", 0);
    const long long top_k = halo_geti_or(a, "top_k", 1);
    if (*n_expert <= 0 || *n_expert > (long long)kMaxExpert || top_k <= 0)
        return HALO_NO(RAD_E_SHAPE);
    *T = ids && ids->data ? halo_numel(ids) : halo_geti_or(a, "M", 0) * top_k;
    if (*T <= 0) return HALO_NO(RAD_E_SHAPE);
    return RAD_OK;
}

/* moe_scatter, and with `inv` (moe_scatter_inv's fifth operand) the inverse beside it. */
static int moe_scatter_body(const RadArgs* a, RadStream s, const RadTensor* inv) {
    const RadTensor* ids    = halo_opt(a, 0);
    const RadTensor* sorted = halo_opt(a, 1);
    const RadTensor* off    = halo_opt(a, 2);
    const RadTensor* cnt    = halo_opt(a, 3);
    if (!ids || !sorted || !off || !cnt) return HALO_NO(RAD_E_INVAL);
    if (!ids->data || !sorted->data || !off->data || !cnt->data) return HALO_NO(RAD_E_INVAL);
    if (ids->dtype != RAD_I32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32) return HALO_NO(RAD_E_DTYPE);
    if (!halo_contig(ids) || !halo_contig(sorted) || !halo_contig(off) || !halo_contig(cnt))
        return HALO_NO(RAD_E_STRIDE);

    long long T = 0, n_expert = 0;
    const int rc = scatter_dims(a, &T, &n_expert);
    if (rc != RAD_OK) return rc;
    if (halo_numel(sorted) < T) return HALO_NO(RAD_E_SHAPE);
    if (halo_numel(off) < n_expert + 1 || halo_numel(cnt) < n_expert) return HALO_NO(RAD_E_SHAPE);
    /* This rank's first expert, subtracted from every routed id so offsets, counts and the weight
     * table are indexed locally. 0 without expert parallelism. */
    const long long base = halo_geti_or(a, "expert_base", 0);
    if (base < 0) return HALO_NO(RAD_E_SHAPE);
    Protect prot;
    const int prc = protect_of(a, n_expert, &prot);
    if (prc != RAD_OK) return prc;

    if (inv && (inv->dtype != RAD_I32 || !halo_contig(inv) || halo_numel(inv) < T))
        return HALO_NO(RAD_E_SHAPE);

    const int32_t* idp = (const int32_t*)ids->data;
    int32_t* sp = (int32_t*)sorted->data;
    int32_t* op = (int32_t*)off->data;
    int32_t* cp = (int32_t*)cnt->data;
    int32_t* ip = inv ? (int32_t*)inv->data : nullptr;

    if (T <= (long long)kScatterSmall) {
        const int rc2 = ip
            ? HALO_EMIT(s, (scatter_small_kernel<true>), dim3(1), dim3(kScatWaves * kWave), 0,
                        idp, sp, op, cp, (uint32_t)T, (uint32_t)n_expert, (int32_t)base, prot, ip)
            : HALO_EMIT(s, (scatter_small_kernel<false>), dim3(1), dim3(kScatWaves * kWave), 0,
                        idp, sp, op, cp, (uint32_t)T, (uint32_t)n_expert, (int32_t)base, prot, ip);
        if (rc2 != RAD_OK) return rc2;
        HALO_MOE_CHK();
        return RAD_OK;
    }

    const long long n_tile = (T + kScatterTile - 1) / kScatterTile;
    const int64_t need = n_tile * n_expert * (long long)sizeof(int32_t);
    if (!a->scratch || a->scratch_bytes < need) return HALO_NO(RAD_E_SCRATCH);
    int32_t* bcnt = (int32_t*)a->scratch;

    int rc2 = HALO_EMIT(s, scatter_hist_kernel, dim3((unsigned)n_tile), dim3(kScatterTile),
                        (unsigned)(n_expert * (long long)sizeof(int32_t)),
                        idp, bcnt, (uint32_t)T, (uint32_t)n_expert, (int32_t)base, prot);
    if (rc2 != RAD_OK) return rc2;
    HALO_MOE_CHK();
    rc2 = HALO_EMIT(s, scatter_scan_kernel, dim3(1), dim3(kMaxExpert), 0,
                    bcnt, op, cp, (uint32_t)n_tile, (uint32_t)n_expert);
    if (rc2 != RAD_OK) return rc2;
    HALO_MOE_CHK();
    rc2 = ip
        ? HALO_EMIT(s, (scatter_place_kernel<true>), dim3((unsigned)n_tile), dim3(kScatterTile), 0,
                    idp, bcnt, op, sp, (uint32_t)T, (uint32_t)n_expert, (int32_t)base, prot, ip)
        : HALO_EMIT(s, (scatter_place_kernel<false>), dim3((unsigned)n_tile), dim3(kScatterTile), 0,
                    idp, bcnt, op, sp, (uint32_t)T, (uint32_t)n_expert, (int32_t)base, prot, ip);
    if (rc2 != RAD_OK) return rc2;
    HALO_MOE_CHK();
    return RAD_OK;
}

/* moe_gather_scratch: the inverse permutation, unless the gather builds or is given one. */
static int64_t moe_gather_scratch(const RadArgs* a) {
    const long long M = halo_geti_or(a, "M", 0);
    const long long top_k = halo_geti_or(a, "top_k", 1);
    if (M <= 0 || top_k <= 0) return HALO_NO(RAD_E_SHAPE);
    return M * top_k * (long long)sizeof(int32_t);
}

/* moe_gather, or with `given` moe_gather_inv, whose third operand is the inverse moe_scatter_inv
 * wrote: no prepass and no self-scan, and the same kernel at the same T, so the same sums. */
static int moe_gather_body(const RadArgs* a, RadStream s, bool given) {
    const RadTensor* ye     = halo_opt(a, 0);
    const RadTensor* ew     = halo_opt(a, 1);
    const RadTensor* sorted = halo_opt(a, 2);
    const RadTensor* sh     = halo_opt(a, 3);
    const RadTensor* sg     = halo_opt(a, 4);
    const RadTensor* Y      = halo_opt(a, 5);
    if (!ye || !ew || !sorted || !Y) return HALO_NO(RAD_E_INVAL);
    if (!ye->data || !ew->data || !sorted->data || !Y->data) return HALO_NO(RAD_E_INVAL);
    const long long n = halo_geti_or(a, "n", 0);
    const long long top_k = halo_geti_or(a, "top_k", 0);
    if (n <= 0 || top_k <= 0) return HALO_NO(RAD_E_SHAPE);
    if (ye->dtype != RAD_BF16 || (ew->dtype != RAD_BF16 && ew->dtype != RAD_F32) ||
        Y->dtype != RAD_BF16)
        return HALO_NO(RAD_E_DTYPE);
    if (sorted->dtype != RAD_I32 || !halo_contig(sorted)) return HALO_NO(RAD_E_STRIDE);
    /* The shared arm is a pair: the plane and its per-row gate, or neither. */
    if ((sh != nullptr) != (sg != nullptr)) return HALO_NO(RAD_E_INVAL);
    if (sh && (sh->dtype != RAD_BF16 || sg->dtype != RAD_BF16)) return HALO_NO(RAD_E_DTYPE);

    long long M = 0;
    const long long y_ld = halo_rows_pitch(Y, n, &M);
    if (y_ld < 0 || M <= 0) return HALO_NO(RAD_E_STRIDE);
    long long Te = 0;
    const long long ye_ld = halo_rows_pitch(ye, n, &Te);
    if (ye_ld < 0) return HALO_NO(RAD_E_STRIDE);
    long long Mw = 0;
    const long long ew_ld = halo_rows_pitch(ew, top_k, &Mw);
    if (ew_ld < 0 || Mw < M) return HALO_NO(RAD_E_STRIDE);
    const long long T = M * top_k;
    if (Te < T || halo_numel(sorted) < T) return HALO_NO(RAD_E_SHAPE);

    /* SELF: small enough for the gather to build its own slot table, skipping the two prepass
     * kernels. Bounded on T because each of the M*gy blocks repeats the scan; at the cap it is under
     * half a megabyte of L2 traffic. Prefill takes the prepass. A given table replaces both, and
     * `small` keeps the kernel choice the self-scan would make. */
    const bool small = T <= (long long)kGatherSelfT && top_k <= (long long)kGatherSlots;
    const bool self = small && !given;

    const int64_t need = self || given ? 0 : T * (long long)sizeof(int32_t);
    int32_t* inv = given ? (int32_t*)sorted->data : nullptr;
    if (!self && !given) {
        if (!a->scratch || a->scratch_bytes < need) return HALO_NO(RAD_E_SCRATCH);
        inv = (int32_t*)a->scratch;
        const unsigned blocks = (unsigned)((T + 255) / 256);
        int rc2 = HALO_EMIT(s, gather_inv_kernel, dim3(blocks), dim3(256), 0, inv, (uint32_t)T);
        if (rc2 != RAD_OK) return rc2;
        HALO_MOE_CHK();
        rc2 = HALO_EMIT(s, gather_scatter_kernel, dim3(blocks), dim3(256), 0,
                        (const int32_t*)sorted->data, inv, (uint32_t)T);
        if (rc2 != RAD_OK) return rc2;
        HALO_MOE_CHK();
    }
    long long shrows = 0, sgrows = 0, sh_ld = 0, sg_ld = 0;
    int sig = 0;
    if (sh) {
        sh_ld = halo_rows_pitch(sh, n, &shrows);
        sg_ld = halo_rows_pitch(sg, 1, &sgrows);
        if (sh_ld < 0 || sg_ld < 0 || shrows < M || sgrows < M) return HALO_NO(RAD_E_STRIDE);
        /* Read as a name, not through halo_act_code, so "none" is distinguishable from a typo. */
        const char* const act = halo_gets(a, "act");
        if (act && !std::strcmp(act, "sigmoid"))   sig = 1;
        else if (act && std::strcmp(act, "none"))  return HALO_NO(RAD_E_UNSUPPORTED);
    }

    /* The prefill form (gather_wide_kernel, a wave per token, sixteen bytes per lane) past the
     * self-scan bound, where top_k fits kGatherWideSlots and every row is 16-byte aligned. Same sums
     * as gather_kernel, which serves everything else. */
    auto al16 = [](const void* p, long long ld) {
        return ((uintptr_t)p % 16) == 0 && ld % (long long)kGatherWideCols == 0;
    };
    const bool wide_ok = ew->dtype == RAD_BF16 && !small &&
                         top_k <= (long long)kGatherWideSlots &&
                         n % (long long)kGatherWideCols == 0 && al16(ye->data, ye_ld) &&
                         al16(Y->data, y_ld) && (!sh || al16(sh->data, sh_ld));
    const int gpin = g_moe_gather_pin;
    if (gpin < 0 || gpin > 2) return HALO_NO(RAD_E_INVAL);
    if (gpin >= 2 && !wide_ok) return HALO_NO(RAD_E_UNSUPPORTED);
    if ((gpin == 0 && wide_ok) || gpin >= 2) {
        constexpr uint32_t NW = 8;
        const unsigned gw =
            (unsigned)((n + kWave * kGatherWideCols - 1) / (kWave * kGatherWideCols));
        const int rc2 = HALO_EMIT(s, (gather_wide_kernel<NW>),
                                  dim3((unsigned)((M + NW - 1) / NW), gw), dim3(NW * kWave), 0,
                                  (const uint16_t*)ye->data, (const uint16_t*)ew->data, inv,
                                  (uint16_t*)Y->data, (uint32_t)M, (uint32_t)n, (uint32_t)top_k,
                                  (uint32_t)ye_ld, (uint32_t)ew_ld, (uint32_t)y_ld,
                                  sh ? (const uint16_t*)sh->data : nullptr,
                                  sh ? (const uint16_t*)sg->data : nullptr,
                                  (uint32_t)sh_ld, (uint32_t)sg_ld, sig);
        if (rc2 != RAD_OK) return rc2;
        HALO_MOE_CHK();
        return RAD_OK;
    }

    const unsigned gy = (unsigned)((n + 255) / 256);
    int rc2;
    if (ew->dtype == RAD_F32)
        rc2 = HALO_EMIT(s, (gather_kernel<256, true>), dim3((unsigned)M, gy), dim3(256), 0,
                        (const uint16_t*)ye->data, ew->data, inv, (uint16_t*)Y->data,
                        (uint32_t)M, (uint32_t)n, (uint32_t)top_k, (uint32_t)ye_ld,
                        (uint32_t)ew_ld, (uint32_t)y_ld,
                        self ? (const int32_t*)sorted->data : nullptr, (uint32_t)T,
                        sh ? (const uint16_t*)sh->data : nullptr,
                        sh ? (const uint16_t*)sg->data : nullptr,
                        (uint32_t)sh_ld, (uint32_t)sg_ld, sig);
    else
        rc2 = HALO_EMIT(s, (gather_kernel<256>), dim3((unsigned)M, gy), dim3(256), 0,
                        (const uint16_t*)ye->data, (const uint16_t*)ew->data, inv,
                        (uint16_t*)Y->data, (uint32_t)M, (uint32_t)n, (uint32_t)top_k,
                        (uint32_t)ye_ld, (uint32_t)ew_ld, (uint32_t)y_ld,
                        self ? (const int32_t*)sorted->data : nullptr, (uint32_t)T,
                        sh ? (const uint16_t*)sh->data : nullptr,
                        sh ? (const uint16_t*)sg->data : nullptr,
                        (uint32_t)sh_ld, (uint32_t)sg_ld, sig);
    if (rc2 != RAD_OK) return rc2;
    HALO_MOE_CHK();
    return RAD_OK;
}

}  /* namespace halo_moe */

extern "C" int halo_rad_moe_scatter(const RadArgs* a, RadStream s) {
    return halo_moe::moe_scatter_body(a, s, nullptr);
}

/* moe_scatter that also writes, for every slot, its row in `sorted_tok` (-1 when dropped): the table
 * moe_gather_inv reads instead of rebuilding it from `sorted_tok`. Same sort, same scratch. */
extern "C" int halo_rad_moe_scatter_inv(const RadArgs* a, RadStream s) {
    const RadTensor* inv = halo_opt(a, 4);
    if (!inv) return HALO_NO(RAD_E_INVAL);
    return halo_moe::moe_scatter_body(a, s, inv);
}

extern "C" int64_t halo_rad_moe_scatter_scratch(const RadArgs* a) {
    long long T = 0, n_expert = 0;
    const int rc = halo_moe::scatter_dims(a, &T, &n_expert);
    if (rc != RAD_OK) return rc;
    if (T <= (long long)halo_moesort::kScatterSmall) return 0;
    const long long n_tile = (T + halo_moe::kScatterTile - 1) / halo_moe::kScatterTile;
    return n_tile * n_expert * (long long)sizeof(int32_t);
}

extern "C" int64_t halo_rad_moe_gather_scratch(const RadArgs* a) {
    return halo_moe::moe_gather_scratch(a);
}

extern "C" int halo_rad_moe_gather(const RadArgs* a, RadStream s) {
    return halo_moe::moe_gather_body(a, s, false);
}

extern "C" int halo_rad_moe_gather_inv(const RadArgs* a, RadStream s) {
    return halo_moe::moe_gather_body(a, s, true);
}
