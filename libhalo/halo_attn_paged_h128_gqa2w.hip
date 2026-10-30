/* Derived from libr4d/r4d_attn_paged_h128_gqa2w.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// Halo attention dispatch for Gemma 3's sliding-window layers (arch/gemma3, TranslateGemma 27B):
// head_dim 128, 2 queries per KV head (32 over 16, 16 over 8 at TP2), paged block 16, window 1024
// (52 of 62 layers). Instantiations of the h128 prefill kernel template with WIN = 1, as
// halo_attn_paged_h256_gqa2.hip does at head_dim 256 for Gemma 4.
//
// Both bands: decode (q_len * gqa <= 64, so q_len <= 32) and the query-tiled prefill kernel. The
// windowless h128/gqa2 rows (halo_attn_paged_h128_gqa2.hip) keep window == 0; these rows take
// window >= 1 only, so the global layers stay on the tuned full-attention legs.
//
// gfx1151: both bands. The split-KV decode kernel is a body of its own
// (halo_attn_decode_h256_gqa6.hip); this unit instantiates it at the end, window bit on.
#include "halo.h"
#include "halo_args.h"
#include "halo_attn_args.h"
#include "halo_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "halo_attn_prefill_h256_gqa6.hip"

#define CHK4() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

#define B_HEAD_DIM   128
#define B_GQA          2
#define B_BLOCK_SIZE  16

// ---- prefill ---------------------------------------------------------------------------------
// The windowless h128 prefill option set (halo_attn_paged_gqa8.hip, G8_P_OPT_128: no O_KWIDE, which
// is a head_dim 256 fetch, K prefetch 4) with WIN on. At gqa 2 a block covers NWARPS * 16 / 2 = 192
// query positions. window <= 0 would give full attention, but the rows never send one here.
#define P_NWARPS 24
#define P_TILE   48
#define P_OPT    3430923

template <int KVP>
static int prefill_launch_h128_gqa2w(const HaloArgs* a, hipStream_t s) {
    if (a->head_dim != B_HEAD_DIM || a->block_size != B_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != B_GQA) return -2;
    constexpr int BQ = (P_NWARPS * 16) / B_GQA;
    dim3 grid((a->q_len + BQ - 1) / BQ, a->kv_heads, a->num_seqs);
    hipLaunchKernelGGL((halo_attn_prefill_kernel<P_NWARPS, P_TILE, B_HEAD_DIM, B_GQA,
                                                 B_BLOCK_SIZE, KVP, P_OPT, 1>),
                       grid, dim3(P_NWARPS * 32), 0, s, *a);
    CHK4();
    return 0;
}

extern "C" int halo_attn_prefill_h128_gqa2w_fp8kv(const HaloArgs* a, hipStream_t s) {
    return prefill_launch_h128_gqa2w<0>(a, s);
}
extern "C" int halo_attn_prefill_h128_gqa2w_bf16kv(const HaloArgs* a, hipStream_t s) {
    return prefill_launch_h128_gqa2w<1>(a, s);
}

/* Prefill RadArgs wrappers. Absent descales mean 1.0 (fill_paged leaves them null). */
extern "C" int halo_rad_attn_prefill_h128_gqa2w_fp8(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = fill_paged(a, &r, B_HEAD_DIM);
    if (rc != RAD_OK) return rc;
    return halo_attn_prefill_h128_gqa2w_fp8kv(&r, (hipStream_t)s);
}
extern "C" int halo_rad_attn_prefill_h128_gqa2w_bf16(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = fill_paged(a, &r, B_HEAD_DIM);
    if (rc != RAD_OK) return rc;
    return halo_attn_prefill_h128_gqa2w_bf16kv(&r, (hipStream_t)s);
}

// ============================================================== the split-KV decode leg
//
// Same geometry, the decode band (q_len * gqa <= 64, so q_len <= 32), with the window bit on: the
// body halo_attn_decode_h256_gqa6.hip instantiated at h128/gqa2 with WIN = 1. The rows carry
// window >= 1, so a global (windowless) layer stays on the tuned full-attention legs.
#include "halo_attn_decode_h256_gqa6.hip"

#define B_TILE HALO_D_TILE
// The h256/gqa6 decode option set. libr4d's h128 windowed set adds O_KRAW, a gfx12 shape this port
// does not instantiate; the set's GPREV turns itself off at head_dim 128 (a V prefetch no longer
// fits in three registers).
#define B_OPT  HALO_D_OPT

extern "C" long halo_attn_decode_h128_gqa2w_scratch_bytes(const HaloArgs* a) {
    return decode_scratch_bytes<B_OPT, 1>(a);
}

extern "C" int halo_attn_decode_h128_gqa2w_fp8kv(const HaloArgs* a, hipStream_t s) {
    return decode_launch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 0, 1>(a, s);
}
extern "C" int halo_attn_decode_h128_gqa2w_bf16kv(const HaloArgs* a, hipStream_t s) {
    return decode_launch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1, 1>(a, s);
}

/* ---- the plugin ABI entry points of the decode band */

extern "C" int64_t halo_rad_attn_decode_h128_gqa2w_scratch(const RadArgs* a) {
    return decode_scratch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a);
}
extern "C" int halo_rad_attn_decode_h128_gqa2w_fp8(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = decode_common<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, &r);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h128_gqa2w_fp8kv(&r, (hipStream_t)s);
}
extern "C" int halo_rad_attn_decode_h128_gqa2w_bf16(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = decode_common<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, &r);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h128_gqa2w_bf16kv(&r, (hipStream_t)s);
}
