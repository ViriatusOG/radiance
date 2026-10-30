/* Derived from libr4d/r4d_ngram_ids.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_ngram_ids.hip -- the hashed row ids of Qwen4-Exp's Per-Layer Embedding.
//
//   mixed(k) = t0*mult[0] ^ t1*mult[1] ^ ... ^ tk*mult[k]        int64, wrapping
//   ids[h]   = mixed(b+1) % vocab_sizes[h] + offsets[h]          h in block b
//
// `heads` is laid out as `ngram - 1` equal blocks, block b using the (b+2)-gram, and `t_s` is the
// token `s` positions back, or the EOS id if the shift would cross one. libref/ref_ngram.cpp
// derives that EOS rule (a cummax over segment starts reduces to "once a shift falls back, every
// deeper one does"); docs/OPS.md holds the operand contract.
//
// One thread a token, looping its heads. A token's heads share the shifted tokens, the EOS walk
// and the `ngram - 1` hash mixes and differ only in the modulus, so a thread per head would redo
// that work per head, and at decode (one token) would be partial waves either way. At prefill the
// grid is thousands of tokens; at decode the kernel is negligible beside the gather that follows.
//
// `mixed % vocab_sizes[h]` is a 64-bit division by a runtime value (~70 cycles on this part),
// sixteen per token. Magic-number division would need the architecture to compute a multiplier
// and shift per head and pass them as a second weight; not worth it at this head count, but the
// change to make if a model hashes at prefill width with many more heads.
#include "halo.h"
#include "halo_args.h"
#include "halo_emit.h"

#include <cstdint>

namespace {

/* The same two operations libref spells out: signed overflow is UB, and torch's `remainder` floors
 * where C truncates. On a real checkpoint neither fires (the multipliers keep a real token's
 * product non-negative), but kernel and oracle must also agree on test inputs. */
__device__ __forceinline__ long long ng_wrap_mul(long long a, long long b) {
    return (long long)((unsigned long long)a * (unsigned long long)b);
}
__device__ __forceinline__ long long ng_floor_mod(long long a, long long b) {
    const long long r = a % b;
    return r < 0 ? r + b : r;
}

enum { kNgMaxShift = 8, kNgThreads = 256 };

/* One workgroup per (token tile, sequence), with the window write-back in the same launch.
 *
 * `s` is blockIdx.y. The fold is legal because only a token with `t - bos < ctx` reads past its
 * sequence's first row into the window, and those are the first `ctx` tokens, all in tile 0. So
 * tile 0's block is the only reader of state[slot]; its threads are ordered by __syncthreads(), and
 * thread 0 writes the new window after it. No other block touches those bytes.
 *
 * At decode this op's cost is the dispatch itself (flat from M=1 to M=512), so avoiding a second
 * launch for the write-back is most of the saving. */
__global__ __launch_bounds__(kNgThreads) void ngram_ids_kernel(
    const int32_t* __restrict__ tok, int32_t* __restrict__ state,
    const int32_t* __restrict__ cu, const long long* __restrict__ mult,
    const long long* __restrict__ vsz, const long long* __restrict__ off,
    const int32_t* __restrict__ sidx, const int32_t* __restrict__ hini,
    const int32_t* __restrict__ nacc, int32_t* __restrict__ ids,
    uint32_t heads, uint32_t ngram, uint32_t per, uint32_t W,
    long long eos, uint32_t ids_ld, int32_t* __restrict__ ahead, uint32_t n_ahead,
    uint32_t ahead_ld) {

    const uint32_t s   = blockIdx.y;
    const uint32_t bos = (uint32_t)cu[s];
    const uint32_t eot = (uint32_t)cu[s + 1];
    const uint32_t ctx = ngram - 1u;
    /* Only block-uniform exits before the barrier below: an empty sequence and a negative slot are
     * properties of the block, so all its threads leave together. */
    if (eot <= bos) return;
    const int32_t slot = sidx ? sidx[s] : (int32_t)s;
    if (slot < 0) return;

    const uint32_t slen = eot - bos;
    const bool warm = !hini || hini[s] != 0;
    const long long o = nacc ? (long long)nacc[s] - 1 : 0;
    const uint32_t lt = blockIdx.x * kNgThreads + threadIdx.x;   /* this thread's row, in-sequence */

    /* The following prompt continues the last sequence: its tokens sit in `tok` after the step's,
     * so a row past the sequence's end hashes exactly as it will when the next step runs it, and
     * lands in `ahead` instead of `ids`. The window below is the step's own and never sees them. */
    const uint32_t reach = slen + ((ahead && s + 1u == gridDim.y) ? n_ahead : 0u);
    if (lt < reach) {
        const uint32_t t = bos + lt;
        /* The `ctx` committed ids before this sequence's first row. A cold slot's window is EOS,
         * not its stale contents and not zero (a real token id). */
        long long prev[kNgMaxShift];
        for (uint32_t i = 0; i < ctx; ++i)
            prev[i] = warm ? (long long)state[(long long)slot * W + o + i] : eos;

        /* `p` indexes the history `prev ++ this sequence's rows`, so row `lt` sits at `ctx + lt`
         * and every shift is in range without a bounds test. */
        const uint32_t p = ctx + lt;
        long long sh[kNgMaxShift];
        sh[0] = (long long)tok[t];
        bool cut = false;
        for (uint32_t k = 1; k < ngram; ++k) {
            if (cut || sh[k - 1] == eos) { cut = true; sh[k] = eos; continue; }
            const uint32_t q = p - k;
            sh[k] = q < ctx ? prev[q] : (long long)tok[bos + (q - ctx)];
        }

        /* One mix per n-gram order, extended in place: the (b+2)-gram's mix is the (b+1)-gram's
         * XOR one more term, as in the reference's inner loop. */
        long long mixed = 0;
        int blk = -1;
        for (uint32_t k = 0; k < ngram; ++k) {
            const long long term = ng_wrap_mul(sh[k], mult[k]);
            mixed = k == 0 ? term : (mixed ^ term);
            if (k == 0) continue;
            ++blk;
            int32_t* const row = lt < slen ? ids + (long long)t * ids_ld
                                           : ahead + (long long)(lt - slen) * ahead_ld;
            for (uint32_t j = 0; j < per; ++j) {
                const uint32_t h = (uint32_t)blk * per + j;
                row[h] = (int32_t)(ng_floor_mod(mixed, vsz[h]) + off[h]);
            }
        }
    }

    /* The window this step leaves, in libref/ref_ngram.cpp's two layouts.
     *
     * With `num_accepted` this is a decode step whose drafts a verify may reject: the window shifts
     * down by one and every id of the step follows it, so slot i holds position i + 1 of
     * (window read ++ step). A next step that kept k of these ids reads at k - 1 and finds the
     * `ctx` ids ending at the k-th. Slot i is read from o + i + 1, past every slot already written,
     * so the ascending walk is safe in place. Slots past W are not written; the manager sizes the
     * window ctx + n_spec, enough for a step of 1 + n_spec ids.
     *
     * Without it every id commits and the last `ctx` land at offset zero.
     *
     * One thread, not one per slot: a thread per (sequence, k) races within the sequence (at
     * ctx = 2 and slen = 1, thread 0 reads slot o+1 while thread 1 writes slot 1), and at decode
     * the serial loop is only ctx - 1 + slen stores. */
    __syncthreads();
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int32_t* const win = state + (long long)slot * W;
    if (nacc) {
        for (uint32_t k = 0; k + 1 < ctx; ++k)
            win[k] = warm ? win[o + k + 1] : (int32_t)eos;
        for (uint32_t t = 0; t < slen && ctx - 1u + t < W; ++t) win[ctx - 1u + t] = tok[bos + t];
        return;
    }
    long long keep[kNgMaxShift];
    for (uint32_t k = 0; k < ctx; ++k) {
        const long long src = (long long)slen - (long long)ctx + (long long)k;
        keep[k] = src >= 0 ? (long long)tok[bos + src]
                           : (warm ? (long long)win[o + src + ctx] : eos);
    }
    for (uint32_t k = 0; k < ctx; ++k) win[k] = (int32_t)keep[k];
}

}  // namespace

enum { NG_TOK = 0, NG_STATE, NG_CU, NG_MULT, NG_VSZ, NG_OFF, NG_CIDX, NG_HINIT, NG_NACC, NG_IDS,
       NG_AHEAD };

extern "C" int halo_rad_ngram_ids(const RadArgs* a, RadStream stream) {
    const RadTensor* tok  = halo_opt(a, NG_TOK);
    const RadTensor* st   = halo_opt(a, NG_STATE);
    const RadTensor* cu   = halo_opt(a, NG_CU);
    const RadTensor* mult = halo_opt(a, NG_MULT);
    const RadTensor* vsz  = halo_opt(a, NG_VSZ);
    const RadTensor* off  = halo_opt(a, NG_OFF);
    const RadTensor* cidx = halo_opt(a, NG_CIDX);
    const RadTensor* hini = halo_opt(a, NG_HINIT);
    const RadTensor* nacc = halo_opt(a, NG_NACC);
    const RadTensor* ids  = halo_opt(a, NG_IDS);
    const RadTensor* ahd  = halo_opt(a, NG_AHEAD);
    if (!tok || !st || !cu || !mult || !vsz || !off || !ids) return HALO_NO(RAD_E_INVAL);
    if (tok->dtype != RAD_I32 || st->dtype != RAD_I32 || ids->dtype != RAD_I32)
        return HALO_NO(RAD_E_DTYPE);
    if (cu->dtype != RAD_I32) return HALO_NO(RAD_E_DTYPE);
    if (mult->dtype != RAD_I64 || vsz->dtype != RAD_I64 || off->dtype != RAD_I64)
        return HALO_NO(RAD_E_DTYPE);
    if (!halo_contig(tok) || !halo_contig(st) || !halo_contig(cu)) return HALO_NO(RAD_E_STRIDE);

    const long long M     = halo_geti_or(a, "M", 0);
    const long long heads = halo_geti_or(a, "heads", 0);
    const long long ngram = halo_geti_or(a, "ngram", 0);
    const long long eos   = halo_geti_or(a, "eos", -1);
    if (M <= 0 || heads <= 0 || ngram < 2 || eos < 0) return HALO_NO(RAD_E_SHAPE);
    if (ngram > kNgMaxShift) return HALO_NO(RAD_E_UNSUPPORTED);
    if (heads % (ngram - 1)) return HALO_NO(RAD_E_SHAPE);
    const long long per = heads / (ngram - 1), ctx = ngram - 1;

    long long rows = 0;
    const long long ids_ld = halo_rows_pitch(ids, heads, &rows);
    if (ids_ld < 0 || rows <= 0) return HALO_NO(RAD_E_STRIDE);
    /* The row count comes from the operand: `M` is ranged, so the arg holds the band's upper bound
     * and a grid sized by it would run off the arena. */
    const long long T = rows;
    /* The prompt that follows: `ahead`'s rows, whose tokens `tok` holds after the step's. */
    long long A = 0, ahead_ld = heads;
    if (ahd) {
        if (ahd->dtype != RAD_I32) return HALO_NO(RAD_E_DTYPE);
        ahead_ld = halo_rows_pitch(ahd, heads, &A);
        if (ahead_ld < 0) return HALO_NO(RAD_E_STRIDE);
    }
    if (halo_numel(tok) < T + A) return HALO_NO(RAD_E_SHAPE);
    if (halo_numel(mult) < ngram || halo_numel(vsz) < heads || halo_numel(off) < heads)
        return HALO_NO(RAD_E_SHAPE);

    const long long N = halo_numel(cu) - 1;
    if (N <= 0) return HALO_NO(RAD_E_SHAPE);
    const long long W = st->rank >= 2 ? halo_dim(st, (int)st->rank - 1) : ctx;
    if (W < ctx) return HALO_NO(RAD_E_SHAPE);

    /* (token tiles, sequences). Tiles are sized by the batch's total token count, an upper bound on
     * any one sequence's; a short sequence's extra blocks find their rows past its end. A
     * per-sequence count would mean reading `cu` on the host. */
    const uint32_t tiles = (uint32_t)((T + A + kNgThreads - 1) / kNgThreads);
    return HALO_EMIT(stream, ngram_ids_kernel, dim3(tiles, (uint32_t)N), dim3(kNgThreads), 0,
                       (const int32_t*)halo_addr(tok), (int32_t*)halo_addr(st),
                       (const int32_t*)halo_addr(cu), (const long long*)halo_addr(mult),
                       (const long long*)halo_addr(vsz), (const long long*)halo_addr(off),
                       cidx ? (const int32_t*)halo_addr(cidx) : nullptr,
                       hini ? (const int32_t*)halo_addr(hini) : nullptr,
                       nacc ? (const int32_t*)halo_addr(nacc) : nullptr,
                       (int32_t*)halo_addr(ids), (uint32_t)heads, (uint32_t)ngram,
                       (uint32_t)per, (uint32_t)W, eos, (uint32_t)ids_ld,
                       A > 0 ? (int32_t*)halo_addr(ahd) : nullptr, (uint32_t)A,
                       (uint32_t)ahead_ld);
}
