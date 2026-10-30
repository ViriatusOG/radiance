/* Derived from libr4d/r4d_sam_attn_bf16.hip by the libhalo rename transform (2026-10-26).
 * Ported to gfx1151: no change was needed. This body is scalar/SIMT -- each thread owns one query
 * row in f32 registers and the dots are elementwise -- so it touches no WMMA fragment layout, no
 * 16-bit operand width and no (lane & 15) mapping. It ports as it stands; see libhalo/README.md. */
// halo_sam_attn_bf16 -- Unlimited-OCR's DeepEncoder (SAM ViT-B/16) attention with decomposed relative
// position (`sam_attn`; arch/common/rad_block_sam.h, libref/ref_sam_attn.cpp is the oracle).
//
// The logits of a query row i against a key j of the same window tile are
//
//     scale * q.k  +  q.Rh[i_h, j_h]  +  q.Rw[i_w, j_w]
//
// over the head's whole head_dim, the tables indexed by the offset inside the tile. With `window`
// > 0 the grid is padded with ZERO rows to a multiple of the window and cut into window x window
// tiles (deepencoder's window_partition, after norm1): the pad rows are keys and values of their
// tile and take part in the softmax, their v being zero. `window` 0 is one tile over the segment.
//
// Shape. One block per (segment, tile, head), so the tile's keys are shared by the block's query
// rows and the k/v rows a wave reads at each step are one address (an L1/L2 hit, broadcast across
// the wave). Each thread owns one query row: q and the output accumulator are HD floats in
// registers, the softmax is the online form (running max and sum), and the rel_pos dot of a tile
// row is computed once per (query row, tile row) rather than once per key.
//
// Head widths. HD is a template parameter and only 64 is instantiated (SAM ViT-B's 12 x 64); the
// shim refuses any other width rather than reading past the registers. A table whose row count is
// not 2*tile-1 is resampled on read (get_rel_pos's F.interpolate(size=want, mode="linear",
// align_corners=False) in f32): the 127-row global tables give 79 rows at a 40 grid, and every
// other geometry the tower declares already has exactly the rows it wants.
//
// Exactness. f32 throughout, one bf16 rounding of the output, exp2 (the hardware transcendental)
// with the log2(e) factor; the selftest holds it to the house tolerance against a host port of the
// reference, not to bits.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_dt16.h"

#include <cmath>

namespace {

constexpr uint32_t kThreads = 128;
constexpr float    kLog2e   = 1.4426950408889634f;

/* Row loads stay scalar uint16: the operand contract allows any row pitch (the tower's q/k/v are
 * column slices of one fused projection, and a tower may pad rows), so a 16-byte bf16x8 read
 * would be misaligned for a pitch that is not a multiple of 8 elements. The register diet is the
 * w-dot hoist below, not the load width. */

/* q . row(d) of a rel_pos table as the reference resized it to `want` rows: linear interpolation
 * between the two source rows at align_corners=False's coordinate, clamped at the edges. With
 * rows == want the source is row d itself (f == 0), so the direct read is unchanged bit for bit. */
template <int HD>
__device__ inline float tab_dot(const uint16_t* __restrict__ tab, int rows, int want, int d,
                                const float* qf) {
    if (rows == want) {
        const uint16_t* rp = tab + (size_t)d * HD;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HD; ++c) s += qf[c] * bf16_to_f32(rp[c]);
        return s;
    }
    double src = ((double)d + 0.5) * (double)rows / (double)want - 0.5;
    if (src < 0.0) src = 0.0;
    if (src > (double)(rows - 1)) src = (double)(rows - 1);
    const int i0 = (int)src, i1 = i0 + 1 < rows ? i0 + 1 : i0;
    const float f = (float)(src - (double)i0);
    const uint16_t* a = tab + (size_t)i0 * HD;
    const uint16_t* b = tab + (size_t)i1 * HD;
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HD; ++c) {
        const float av = bf16_to_f32(a[c]);
        s += qf[c] * (av + (bf16_to_f32(b[c]) - av) * f);
    }
    return s;
}

/* Eight bf16 of a q/k/v/out row as one 16-byte uint4. VEC is only instantiated when the caller
 * proved every involved pointer is 16-byte aligned and every row pitch a multiple of 8 elements
 * (launch_sam_attn); otherwise the scalar uint16 path below runs, because the operand contract
 * allows any pitch (the tower's q/k/v are column slices and may carry a filler column). The
 * element order is the scalar order, so both paths produce bit-identical dots. */
__device__ inline void bf16x8_to_f32(uint4 u, float* f) {
    const unsigned* w = (const unsigned*)&u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i]     = bf16_to_f32((uint16_t)(w[i] & 0xffffu));
        f[2 * i + 1] = bf16_to_f32((uint16_t)(w[i] >> 16));
    }
}

template <bool VEC, int HD>
__device__ inline void ld_row(const uint16_t* p, float* f) {
    if (VEC) {
        const uint4* u = (const uint4*)p;
#pragma unroll
        for (int c8 = 0; c8 < HD / 8; ++c8) bf16x8_to_f32(u[c8], f + c8 * 8);
    } else {
#pragma unroll
        for (int c = 0; c < HD; ++c) f[c] = bf16_to_f32(p[c]);
    }
}

template <bool VEC, int HD>
__device__ inline void st_row(uint16_t* p, const float* f) {
    if (VEC) {
        uint4 u[HD / 8];
        uint16_t* b = (uint16_t*)u;
#pragma unroll
        for (int c = 0; c < HD; ++c) b[c] = f32_to_bf16(f[c]);
        uint4* o = (uint4*)p;
#pragma unroll
        for (int c8 = 0; c8 < HD / 8; ++c8) o[c8] = u[c8];
    } else {
#pragma unroll
        for (int c = 0; c < HD; ++c) p[c] = f32_to_bf16(f[c]);
    }
}

template <bool VEC, int HD>
__device__ inline float dot_row(const uint16_t* p, const float* qf) {
    float s = 0.0f;
    if (VEC) {
        const uint4* u = (const uint4*)p;
#pragma unroll
        for (int c8 = 0; c8 < HD / 8; ++c8) {
            const unsigned* w = (const unsigned*)&u[c8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s += qf[c8 * 8 + 2 * i] * bf16_to_f32((uint16_t)(w[i] & 0xffffu));
                s += qf[c8 * 8 + 2 * i + 1] * bf16_to_f32((uint16_t)(w[i] >> 16));
            }
        }
    } else {
#pragma unroll
        for (int c = 0; c < HD; ++c) s += qf[c] * bf16_to_f32(p[c]);
    }
    return s;
}

template <bool VEC, int HD>
__device__ inline void axpy_row(float* acc, const uint16_t* p, float pv) {
    if (VEC) {
        const uint4* u = (const uint4*)p;
#pragma unroll
        for (int c8 = 0; c8 < HD / 8; ++c8) {
            float f8[8];
            bf16x8_to_f32(u[c8], f8);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[c8 * 8 + i] += pv * f8[i];
        }
    } else {
#pragma unroll
        for (int c = 0; c < HD; ++c) acc[c] += pv * bf16_to_f32(p[c]);
    }
}

template <int HD, bool VEC>
__device__ inline void sam_attn_body(
    const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
    const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
    const int* __restrict__ cu, const uint16_t* __restrict__ rh,
    const uint16_t* __restrict__ rw, int rh_rows, int rw_rows,
    int q_ld, int k_ld, int v_ld, int o_ld, int nhead, int gh, int gw, int th, int tw,
    int seq, int head, int ty, int tx, int row_lo, int row_hi, float scale) {
    // The caller supplies the (segment, tile, head) block coordinates: the windowed kernel reads
    // them from blockIdx.x's window-tile layout, the global one from its query-tile layout.
    const int s0 = cu[seq];
    if (cu[seq + 1] - s0 != gh * gw) {
        // A segment that is not one grid is a tower bug, not a shape: the shim pre-checks the
        // row count against the segment count, and this is the device-side guard. Write NaN over
        // the segment's rows (the vit kernel's poison convention) so the corruption reaches the
        // consumer -- a silent return would pass a garbage plane on as success.
        const int n = cu[seq + 1] - s0;
        for (int r = (int)threadIdx.x; r < n; r += (int)kThreads) {
            uint16_t* op = go + (size_t)(s0 + r) * (size_t)o_ld + (size_t)head * HD;
            for (int c = 0; c < HD; ++c) op[c] = (uint16_t)0x7fc0u;   // bf16 NaN, as vit poisons
        }
        return;
    }

    for (int r = row_lo + (int)threadIdx.x; r < row_hi; r += (int)kThreads) {
        const int lh = r / tw, lw = r % tw;
        const int ih = ty * th + lh, iw = tx * tw + lw;
        if (ih >= gh || iw >= gw) continue;    // a pad row: no output

        const size_t qrow = (size_t)(s0 + ih * gw + iw);
        const uint16_t* qp = gq + qrow * (size_t)q_ld + (size_t)head * HD;
        float qf[HD];
        ld_row<VEC, HD>(qp, qf);

        float acc[HD];
#pragma unroll
        for (int c = 0; c < HD; ++c) acc[c] = 0.0f;
        float m = -INFINITY, l = 0.0f;

        /* The w rel-pos dots depend on (lw, jw) only: computed once for the query row when the
         * tile is narrow (the windowed tiles are 14 wide), recomputed per key otherwise. For a
         * whole-grid tile that is gw dots a query row instead of gw per key row, i.e. the same
         * count as the q.k dots rather than gh times it; the array is spilled to local memory
         * when the compiler runs out of registers, which is still far cheaper than the dot. */
        float wr[16];
        float wr_all[64];
        const bool wr_hoist = tw <= 16;
        const bool wr_wide  = !wr_hoist && tw <= 64;
        if (wr_hoist)
            for (int jw = 0; jw < tw; ++jw)
                wr[jw] = tab_dot<HD>(rw, rw_rows, 2 * tw - 1, lw - jw + tw - 1, qf);
        else if (wr_wide)
            for (int jw = 0; jw < tw; ++jw)
                wr_all[jw] = tab_dot<HD>(rw, rw_rows, 2 * tw - 1, lw - jw + tw - 1, qf);

        for (int jh = 0; jh < th; ++jh) {
            const float hr = tab_dot<HD>(rh, rh_rows, 2 * th - 1, lh - jh + th - 1, qf);

            const int kh = ty * th + jh;
            const bool hreal = kh < gh;
            for (int jw = 0; jw < tw; ++jw) {
                float s = hr + (wr_hoist ? wr[jw]
                                         : (wr_wide ? wr_all[jw]
                                                    : tab_dot<HD>(rw, rw_rows, 2 * tw - 1,
                                                                  lw - jw + tw - 1, qf)));

                const int kw = tx * tw + jw;
                const bool real = hreal && kw < gw;
                const uint16_t* vp = nullptr;
                if (real) {
                    const size_t krow = (size_t)(s0 + kh * gw + kw);
                    const uint16_t* kp = gk + krow * (size_t)k_ld + (size_t)head * HD;
                    const float d = dot_row<VEC, HD>(kp, qf);
                    s += scale * d;
                    vp = gv + krow * (size_t)v_ld + (size_t)head * HD;
                }

                // Online softmax: a new maximum rescales what has been accumulated.
                if (s > m) {
                    const float corr = ex2((m - s) * kLog2e);
#pragma unroll
                    for (int c = 0; c < HD; ++c) acc[c] *= corr;
                    l *= corr;
                    m = s;
                }
                const float p = ex2((s - m) * kLog2e);
                l += p;
                if (real) axpy_row<VEC, HD>(acc, vp, p);
            }
        }

        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        uint16_t* op = go + qrow * (size_t)o_ld + (size_t)head * HD;
#pragma unroll
        for (int c = 0; c < HD; ++c) acc[c] *= inv;
        st_row<VEC, HD>(op, acc);
    }
}

template <int HD>
__global__ __launch_bounds__(kThreads) void sam_attn_kernel(
    const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
    const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
    const int* __restrict__ cu, const uint16_t* __restrict__ rh,
    const uint16_t* __restrict__ rw, int rh_rows, int rw_rows,
    int q_ld, int k_ld, int v_ld, int o_ld, int nhead, int gh, int gw, int th, int tw,
    int nth, int ntw, int vec_rows, float scale) {
    // blockIdx.x = ((seq * nth + ty) * ntw + tx) * nhead + head: heads adjacent, so the tile's k/v
    // rows stay hot across them.
    int b = blockIdx.x;
    const int head = b % nhead; b /= nhead;
    const int tx   = b % ntw;   b /= ntw;
    const int ty   = b % nth;   b /= nth;
    const int seq  = b;
    if (vec_rows)
        sam_attn_body<HD, true>(gq, gk, gv, go, cu, rh, rw, rh_rows, rw_rows, q_ld, k_ld, v_ld,
                                o_ld, nhead, gh, gw, th, tw, seq, head, ty, tx, 0, th * tw, scale);
    else
        sam_attn_body<HD, false>(gq, gk, gv, go, cu, rh, rw, rh_rows, rw_rows, q_ld, k_ld, v_ld,
                                 o_ld, nhead, gh, gw, th, tw, seq, head, ty, tx, 0, th * tw, scale);
}

/* The global path (window 0) is one tile over the whole segment, so the kernel above would launch
 * nseq*nhead blocks with each thread walking rows/N-threads query rows in series (12-32 on a 40 or
 * 64 grid): 60 blocks for a 5-tile page, far too few to fill the card. This variant keeps the same
 * per-row math and key order but tiles the query rows, one block per (segment, query tile, head).
 * Grid positions are the segment's row-major cells, so a tile is a contiguous row range. */
template <int HD>
__global__ __launch_bounds__(kThreads) void sam_attn_qtile_kernel(
    const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
    const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
    const int* __restrict__ cu, const uint16_t* __restrict__ rh,
    const uint16_t* __restrict__ rw, int rh_rows, int rw_rows,
    int q_ld, int k_ld, int v_ld, int o_ld, int nhead, int gh, int gw,
    int nqb, int qrows, int vec_rows, float scale) {
    int b = blockIdx.x;
    const int head = b % nhead; b /= nhead;
    const int qblk = b % nqb;   b /= nqb;
    const int seq  = b;
    const int keys = gh * gw;
    const int lo   = qblk * qrows;
    const int hi   = keys < lo + qrows ? keys : lo + qrows;
    if (vec_rows)
        sam_attn_body<HD, true>(gq, gk, gv, go, cu, rh, rw, rh_rows, rw_rows, q_ld, k_ld, v_ld,
                                o_ld, nhead, gh, gw, gh, gw, seq, head, 0, 0, lo, hi, scale);
    else
        sam_attn_body<HD, false>(gq, gk, gv, go, cu, rh, rw, rh_rows, rw_rows, q_ld, k_ld, v_ld,
                                 o_ld, nhead, gh, gw, gh, gw, seq, head, 0, 0, lo, hi, scale);
}

/* The global path on the matrix units. The SIMT kernels above give each thread one query row and
 * walk the keys one at a time: at the 64 grid of a 1024-pixel view that is 4096 keys of
 * elementwise dots a row, ~28 ms a layer on gfx1151 (~2 TFLOP/s). This form is
 * halo_attn_dense_bf16.hip's core (S^T = K.Q^T, so a lane owns one query column and its softmax
 * state; f16 operands, f32 accumulate; the C fragment of S^T is the B fragment of P.V) with the
 * decomposed rel-pos term added to each score:
 *
 *   s(i, j) = scale * q_i.k_j + q_i.Rh[lh_i - jh + gh - 1] + q_i.Rw[lw_i - jw + gw - 1]
 *
 * The two dots depend on (query, jh) and (query, jw) only, so a block computes them once for its
 * BQ query rows into LDS (f32, from the bf16 q row, through tab_dot's resampling) before the key
 * loop: gh + gw dots a row rather than one per key. They are stored in log2 units, as the scores
 * are (q is scaled by scale * log2(e) once).
 *
 * Exactness: not the SIMT kernel's bits. q.k runs through f16 WMMA, P is rounded to f16 for P.V
 * (halo_dt16.h), and the scores are summed in another order; the selftest holds both forms to
 * the same tolerance against the host port of the reference. Taken when the rows are 16-byte
 * aligned (the 16-byte K/V fetches) and gh + gw fits the LDS table, else the SIMT form runs. */
constexpr int kSamBStr = 129;          // a query row's gh + gw <= 128 rel dots, +1 off the banks

template <int NQB>
__global__ __launch_bounds__(NQB * 32) void sam_attn_wmma_kernel(
    const uint16_t* __restrict__ gq, const uint16_t* __restrict__ gk,
    const uint16_t* __restrict__ gv, uint16_t* __restrict__ go,
    const int* __restrict__ cu, const uint16_t* __restrict__ rh,
    const uint16_t* __restrict__ rw, int rh_rows, int rw_rows,
    int q_ld, int k_ld, int v_ld, int o_ld, int gh, int gw, float scale) {
    typedef DT16<1> D;
    typedef D::frag frag16;
    constexpr int HD     = 64;
    constexpr int TILE   = 64;
    constexpr int NTHR   = NQB * 32;
    constexpr int BQ     = NQB * 16;
    constexpr int NKS    = HD / 16;
    constexpr int MT     = TILE / 16;
    constexpr int KSTR   = HD + 8;
    constexpr int VSTR   = TILE + 8;
    constexpr int DG     = HD / 8;
    constexpr int KGRP   = TILE / 8;
    constexpr int KITEMS = TILE * DG;
    constexpr int KP     = (KITEMS + NTHR - 1) / NTHR;
    constexpr int NTR    = KGRP * DG;
    constexpr int LGRP   = NTHR / 8;
    constexpr int VP     = (NTR + LGRP - 1) / LGRP;
    constexpr float GROW = D::GROW;
    static_assert(NTHR == 2 * BQ, "one thread a (query row, table) for the rel-pos pass");

    __shared__ __align__(16) uint16_t sK[TILE * KSTR];
    __shared__ __align__(16) uint16_t sV[HD * VSTR];
    __shared__ float sB[BQ * kSamBStr];

    const int tid = threadIdx.x, lane = tid & 31, qbi = tid >> 5;
    const int c = lane & 15, h = lane >> 4;
    const int head = blockIdx.y, seq = blockIdx.z;
    const int s0 = cu[seq], slen = gh * gw;
    const int q0 = blockIdx.x * BQ;
    if (cu[seq + 1] - s0 != slen) {
        // The SIMT body's guard: poison this block's rows of the malformed segment.
        const int n = cu[seq + 1] - s0;
        for (int r = q0 + tid; r < n && r < q0 + BQ; r += NTHR)
            for (int e = 0; e < HD; ++e)
                go[(size_t)(s0 + r) * (size_t)o_ld + (size_t)head * HD + e] = (uint16_t)0x7fc0u;
        return;
    }
    if (q0 >= slen) return;

    // ---- the rel-pos dots of the block's query rows: thread t < BQ the h table, else the w ----
    {
        const int qi = tid % BQ, which = tid / BQ;
        const int qpos = q0 + qi;
        float* brow = &sB[qi * kSamBStr];
        if (qpos < slen) {
            float qf[HD];
            ld_row<true, HD>(gq + (size_t)(s0 + qpos) * (size_t)q_ld + (size_t)head * HD, qf);
            const int lh = qpos / gw, lw = qpos - lh * gw;
            if (which == 0)
                for (int jh = 0; jh < gh; ++jh)
                    brow[jh] = kLog2e * tab_dot<HD>(rh, rh_rows, 2 * gh - 1, lh - jh + gh - 1, qf);
            else
                for (int jw = 0; jw < gw; ++jw)
                    brow[gh + jw] =
                        kLog2e * tab_dot<HD>(rw, rw_rows, 2 * gw - 1, lw - jw + gw - 1, qf);
        }
    }

    const float mul = scale * kLog2e;
    const int  qpos = q0 + qbi * 16 + c;
    const bool live = qpos < slen;

    frag16 qf[NKS];
    {
        const int qrow = live ? qpos : slen - 1;
        const uint16_t* qp = gq + (size_t)(s0 + qrow) * (size_t)q_ld + (size_t)head * HD;
#pragma unroll
        for (int t = 0; t < NKS; ++t) {
            const uint4 u0 = *(const uint4*)(qp + 16 * t);
            const uint4 u1 = *(const uint4*)(qp + 16 * t + 8);
            auto cv = [&](uint32_t w) {
                return D::pk(__builtin_bit_cast(float, w << 16) * mul,
                             __builtin_bit_cast(float, w & 0xffff0000u) * mul);
            };
            const uint32_t w8[8] = { cv(u0.x), cv(u0.y), cv(u0.z), cv(u0.w),
                                     cv(u1.x), cv(u1.y), cv(u1.z), cv(u1.w) };
            qf[t] = D::mk(w8);
        }
    }
    const float* brow = &sB[(qbi * 16 + c) * kSamBStr];

    float m_ref = -3.0e38f, soff = 3.0e38f, l_i = 0.0f;
    v8f acc[NKS];
#pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};

    const int vj = tid & 7, lgi = tid >> 3;
    const uint16_t* kbase = gk + (size_t)s0 * (size_t)k_ld + (size_t)head * HD;
    const uint16_t* vbase = gv + (size_t)s0 * (size_t)v_ld + (size_t)head * HD;

    const int ntiles = (slen + TILE - 1) / TILE;
    for (int ti = 0; ti < ntiles; ++ti) {
        const int k0 = ti * TILE;
        const bool mask = k0 + TILE > slen;

        uint4 kr[KP], vr[VP];
#pragma unroll
        for (int p = 0; p < KP; ++p) {
            const int i = tid + p * NTHR;
            if (KITEMS % NTHR == 0 || i < KITEMS) {
                const int key = i / DG, gg = i % DG;
                const int kk = (k0 + key < slen) ? k0 + key : slen - 1;
                kr[p] = *(const uint4*)(kbase + (size_t)kk * (size_t)k_ld + gg * 8);
            }
        }
#pragma unroll
        for (int r = 0; r < VP; ++r) {
            const int u = lgi + r * LGRP;
            if (NTR % LGRP == 0 || u < NTR) {
                const int kg = u % KGRP, dg = u / KGRP;
                const int kk0 = k0 + kg * 8 + vj;
                const int kk = (kk0 < slen) ? kk0 : slen - 1;
                vr[r] = __builtin_bit_cast(uint4, load_tr_b128(vbase + (size_t)kk * (size_t)v_ld +
                                                               dg * 8));
            }
        }

        lds_barrier();          // the previous tile's readers are done (and, first, the rel dots)
#pragma unroll
        for (int r = 0; r < VP; ++r) {
            const int u = lgi + r * LGRP;
            if (NTR % LGRP == 0 || u < NTR) {
                const int kg = u % KGRP, dg = u / KGRP;
                const uint4 w = vr[r];
                *(uint4*)&sV[(dg * 8 + vj) * VSTR + kg * 8] =
                    make_uint4(D::from_bf16w(w.x), D::from_bf16w(w.y),
                               D::from_bf16w(w.z), D::from_bf16w(w.w));
            }
        }
#pragma unroll
        for (int p = 0; p < KP; ++p) {
            const int i = tid + p * NTHR;
            if (KITEMS % NTHR == 0 || i < KITEMS) {
                const uint4 w = kr[p];
                *(uint4*)&sK[(i / DG) * KSTR + (i % DG) * 8] =
                    make_uint4(D::from_bf16w(w.x), D::from_bf16w(w.y),
                               D::from_bf16w(w.z), D::from_bf16w(w.w));
            }
        }
        lds_barrier();

#pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            v8f s = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
            {
                const uint16_t* krow = &sK[(mt * 16 + c) * KSTR];
                frag16 kf[NKS];
#pragma unroll
                for (int j = 0; j < NKS; ++j) {
                    const uint4 u0 = *(const uint4*)(krow + 16 * j);
                    const uint4 u1 = *(const uint4*)(krow + 16 * j + 8);
                    const uint32_t w8[8] = { u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w };
                    kf[j] = D::mk(w8);
                }
                sched_barrier();
#pragma unroll
                for (int j = 0; j < NKS; ++j) s = D::wmma(kf[j], qf[j], s);
            }

            // Element e is key kq + 2e + h: walk its (jh, jw) by two, wrapping at the row end.
            const int key0 = k0 + mt * 16 + h;
            int kh = key0 / gw, kw = key0 - kh * gw;
            float sc[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const bool ok = !mask || key0 + 2 * e < slen;
                const int khc = kh < gh ? kh : gh - 1;
                sc[e] = ok ? s[e] + brow[khc] + brow[gh + kw] : -INFINITY;
                kw += 2;
                if (kw >= gw) { kw -= gw; ++kh; }
            }

            float smax = sc[0];
#pragma unroll
            for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);
            if (wave_any(smax > m_ref + GROW)) {
                const float rmax  = fmaxf(smax, swap16(smax));
                const float mnew  = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
                const float alpha = ex2(m_ref - mnew);
#pragma unroll
                for (int t = 0; t < NKS; ++t)
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[t][e] *= alpha;
                l_i *= alpha;
                m_ref = mnew;
                soff  = -mnew;
            }

            uint32_t pw[4];
#pragma unroll
            for (int e = 0; e < 8; e += 2)
                pw[e >> 1] = D::pk(ex2(sc[e] + soff), ex2(sc[e + 1] + soff));
            const v2hf_dt one = {(__fp16)1.0f, (__fp16)1.0f};
#pragma unroll
            for (int e = 0; e < 4; ++e)
                l_i = __builtin_amdgcn_fdot2(__builtin_bit_cast(v2hf_dt, pw[e]), one, l_i, false);
            uint32_t pw16[8];
            p16w<D>(pw, pw16);
            const frag16 p16 = D::mk(pw16);

            {
                const uint16_t* vrow = &sV[c * VSTR + mt * 16];
                frag16 vf[NKS];
#pragma unroll
                for (int j = 0; j < NKS; ++j) {
                    const uint4 u0 = *(const uint4*)(vrow + j * 16 * VSTR);
                    const uint4 u1 = *(const uint4*)(vrow + j * 16 * VSTR + 8);
                    const uint32_t w8[8] = { u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w };
                    vf[j] = D::mk(w8);
                }
                sched_barrier();
#pragma unroll
                for (int j = 0; j < NKS; ++j) acc[j] = D::wmma(vf[j], p16, acc[j]);
            }
        }
    }

    if (!live) return;
    const float ls = l_i + swap16(l_i);
    const float inv = (ls > 0.0f) ? (1.0f / ls) : 0.0f;
    uint16_t* op = go + (size_t)(s0 + qpos) * (size_t)o_ld + (size_t)head * HD;
    // Element e of the accumulator is head dim 2*e + (lane>>4): a scatter of eight 16-bit values.
#pragma unroll
    for (int t = 0; t < NKS; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e)
            op[t * 16 + 2 * e + h] = f32_to_bf16(acc[t][e] * inv);
}

int launch_sam_attn(const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* o,
                    const int* cu, const uint16_t* rh, const uint16_t* rw,
                    int rh_rows, int rw_rows,
                    int q_ld, int k_ld, int v_ld, int o_ld, int nseq, int nhead, int gh, int gw,
                    int win, float scale, long stream) {
    const int th = win > 0 ? win : gh, tw = win > 0 ? win : gw;
    const int nth = (gh + th - 1) / th, ntw = (gw + tw - 1) / tw;
    /* The 16-byte row loads are legal only when every q/k/v/out row starts on a 16-byte
     * boundary: all four bases aligned, every row pitch a multiple of 8 bf16. The selftest's
     * padded pitch and a slice with a filler column both fail this and take the scalar path. */
    const uintptr_t addr_bits = (uintptr_t)q | (uintptr_t)k | (uintptr_t)v | (uintptr_t)o;
    const bool vec_rows = (addr_bits & 15u) == 0 &&
                          ((q_ld | k_ld | v_ld | o_ld) & 7) == 0;
    if (win == 0 && vec_rows && gh + gw < kSamBStr) {
        constexpr int NQB = 4;
        const dim3 grid((uint32_t)((gh * gw + NQB * 16 - 1) / (NQB * 16)), (uint32_t)nhead,
                        (uint32_t)nseq);
        hipLaunchKernelGGL((sam_attn_wmma_kernel<NQB>), grid, dim3(NQB * 32), 0,
                           (hipStream_t)stream, q, k, v, o, cu, rh, rw, rh_rows, rw_rows,
                           q_ld, k_ld, v_ld, o_ld, gh, gw, scale);
        return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
    }
    if (win == 0) {
        /* One block per (segment, 128-row query tile, head): the whole-segment tile above would
         * give each thread 12-32 query rows in series and launch only nseq*nhead blocks. */
        const int qrows = 128;
        const int keys  = gh * gw;
        const int nqb   = (keys + qrows - 1) / qrows;
        const size_t qblocks = (size_t)nseq * nqb * nhead;
        if (qblocks == 0 || qblocks > 0x7fffffff) return HALO_NO(RAD_E_SHAPE);
        hipLaunchKernelGGL((sam_attn_qtile_kernel<64>), dim3((uint32_t)qblocks),
                           dim3(kThreads), 0, (hipStream_t)stream, q, k, v, o, cu, rh, rw,
                           rh_rows, rw_rows, q_ld, k_ld, v_ld, o_ld, nhead, gh, gw, nqb, qrows,
                           vec_rows ? 1 : 0, scale);
        return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
    }
    const size_t blocks = (size_t)nseq * nth * ntw * nhead;
    if (blocks == 0 || blocks > 0x7fffffff) return HALO_NO(RAD_E_SHAPE);
    hipLaunchKernelGGL((sam_attn_kernel<64>), dim3((uint32_t)blocks), dim3(kThreads), 0,
                       (hipStream_t)stream, q, k, v, o, cu, rh, rw, rh_rows, rw_rows,
                       q_ld, k_ld, v_ld, o_ld, nhead, gh, gw, th, tw, nth, ntw,
                       vec_rows ? 1 : 0, scale);
    return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}

}  // namespace

// ================================================================================ entry point
//
// The shim checks what the kernel assumes: bf16 operands, rank 2 with dense columns and any row
// pitch (the tower's q/k/v are the three column slices of one fused projection), an i32 cu with one
// entry a segment plus the terminator, tables of exactly 2*tile-1 rows, head_dim 64. The shim
// checks the row count against the segment count host-side; per-segment length is the kernel's to
// check, and a malformed segment writes NaN over its rows (the vit kernel's poison convention)
// so the corruption reaches the consumer instead of passing as success.

#define HALO_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return HALO_NO(_s); } while (0)

static inline int bf16_rows(const RadTensor* t, int64_t cols, int64_t* rows, int64_t* ld) {
    if (!t || !t->data) return RAD_E_INVAL;
    if (t->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (t->rank != 2) return RAD_E_SHAPE;
    if (t->shape[1] != cols || t->stride[1] != 1) return RAD_E_SHAPE;
    *rows = t->shape[0];
    *ld   = t->stride[0];
    return RAD_OK;
}

extern "C" int halo_rad_sam_attn(const RadArgs* a, RadStream s) {
    enum { V_Q = 0, V_K, V_V, V_CU, V_OUT, V_RH, V_RW };
    const RadTensor* q  = halo_opt(a, V_Q);
    const RadTensor* k  = halo_opt(a, V_K);
    const RadTensor* v  = halo_opt(a, V_V);
    const RadTensor* cu = halo_opt(a, V_CU);
    const RadTensor* o  = halo_opt(a, V_OUT);
    const RadTensor* rh = halo_opt(a, V_RH);
    const RadTensor* rw = halo_opt(a, V_RW);
    if (!q || !k || !v || !cu || !o || !rh || !rw) return HALO_NO(RAD_E_INVAL);

    const long long hd  = halo_geti_or(a, "head_dim", 0);
    const long long nh  = halo_geti_or(a, "n_head", 0);
    const long long gh  = halo_geti_or(a, "grid_h", 0);
    const long long gw  = halo_geti_or(a, "grid_w", 0);
    const long long win = halo_geti_or(a, "window", 0);
    if (hd != 64 || nh <= 0 || gh <= 0 || gw <= 0 || win < 0) return HALO_NO(RAD_E_UNSUPPORTED);
    const long long W = nh * hd;

    int64_t qrows = 0, qld = 0, krows = 0, kld = 0, vrows = 0, vld = 0, orows = 0, old = 0;
    HALO_NEED(bf16_rows(q, W, &qrows, &qld));
    HALO_NEED(bf16_rows(k, W, &krows, &kld));
    HALO_NEED(bf16_rows(v, W, &vrows, &vld));
    HALO_NEED(bf16_rows(o, W, &orows, &old));
    if (qrows <= 0 || krows < qrows || vrows < qrows || orows < qrows) return HALO_NO(RAD_E_SHAPE);

    if (!cu->data) return HALO_NO(RAD_E_INVAL);
    if (cu->dtype != RAD_I32 || !halo_contig(cu)) return HALO_NO(RAD_E_DTYPE);
    const long long nseq = halo_numel(cu) - 1;
    if (nseq < 1) return HALO_NO(RAD_E_SHAPE);
    /* The tower declares cu as one segment per grid, so the host-side row count pins the
     * segment count. A mismatch is a malformed segment and fails at issue, not on the device
     * (the kernel's per-segment guard poisons the visible rows for what this cannot see). */
    if (qrows != nseq * gh * gw) return HALO_NO(RAD_E_SHAPE);

    // The tables: [2*tile-1, head_dim] when the geometry has its own, else the checkpoint's rows
    // (the 127-row global table at a 40 grid), resampled on read.
    int64_t rhrows = 0, rhld = 0, rwrows = 0, rwld = 0;
    HALO_NEED(bf16_rows(rh, hd, &rhrows, &rhld));
    HALO_NEED(bf16_rows(rw, hd, &rwrows, &rwld));
    if (rhrows < 1 || rwrows < 1) return HALO_NO(RAD_E_UNSUPPORTED);

    const float scale = (float)halo_getf(a, "scale", 1.0 / std::sqrt((double)hd));

    return launch_sam_attn((const uint16_t*)halo_addr(q), (const uint16_t*)halo_addr(k),
                           (const uint16_t*)halo_addr(v), (uint16_t*)halo_addr(o),
                           (const int*)halo_addr(cu), (const uint16_t*)halo_addr(rh),
                           (const uint16_t*)halo_addr(rw), (int)rhrows, (int)rwrows,
                           (int)qld, (int)kld, (int)vld, (int)old,
                           (int)nseq, (int)nh, (int)gh, (int)gw, (int)win, scale,
                           (long)(uintptr_t)s);
}
