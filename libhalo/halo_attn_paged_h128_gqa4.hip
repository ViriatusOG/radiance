/* Derived from libr4d/r4d_attn_paged_h128_gqa4.hip by the libhalo rename transform (2026-10-27),
 * then ported to gfx1151; see libhalo/README.md. */
// Halo attention dispatch for the DFlash2 drafter: head_dim 128, 4 queries per KV head, paged
// block 16, sliding window. Instantiations of the split-KV decode body
// (halo_attn_decode_h256_gqa6.hip) with WIN = 1, as the Gemma 4 and Gemma 3 legs do at h256/gqa2
// and h128/gqa2w.
//
// Decode only. Queries long enough for the query-tiled prefill kernel occur on a drafter only in
// prefill steps; the decode rows declare the decode band (q_len * gqa <= 64), so those steps
// resolve to the prefill kernel of another geometry's unit.
//
// Rationale: a general unified-attention kernel (vLLM's) launches (q_blocks, kv_heads, segments)
// with q_blocks padded for capture, so most workgroups return immediately; on this shape it is
// 8-9x slower than the window-capped DRAM bound. halo keeps the query inside the workgroup (up to
// 64 rows of q_len * gqa), so its grid is (splits, kv_heads, num_seqs) with no dead workgroups.
//
// The same geometry serves attn_paged_qrow: the gqa-4 kernel with each kv head mapped onto two
// query groups (HaloArgs::kv_rep = 2) is the 31B drafter's gqa 8 over a cache that holds each kv
// head once, and q read at its own row pitch (HaloArgs::q_ld). The workgroup then holds q_len * 4
// rows, so a draft block of 16 fits where gqa 8's own decode (q_len <= 8) does not.
#include "halo.h"
#include "halo_args.h"
#include "halo_attn_args.h"
#include "halo_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "halo_attn_decode_h256_gqa6.hip"

#define B_HEAD_DIM   128
#define B_GQA          4
#define B_BLOCK_SIZE  16

// The h256/gqa6 decode option set (HALO_D_OPT). libr4d's h128 set adds O_KRAW, one-round raw K,
// which is a gfx12 shape the port does not instantiate (halo_attn_decode_h256_gqa6.hip asserts it
// off), and the set's GPREV turns itself off here: at head_dim 128 a V prefetch is 4 registers more
// than the three it fits in.
#define B_OPT  HALO_D_OPT

extern "C" long halo_attn_decode_h128_gqa4_scratch_bytes(const HaloArgs* a) {
    return decode_scratch_bytes<B_OPT, 1>(a);
}

extern "C" int halo_attn_decode_h128_gqa4_fp8kv(const HaloArgs* a, hipStream_t s) {
    return decode_launch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 0, 1>(a, s);
}
extern "C" int halo_attn_decode_h128_gqa4_bf16kv(const HaloArgs* a, hipStream_t s) {
    return decode_launch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1, 1>(a, s);
}
// The same over a cache holding each kv head once for kv_rep query groups (O_KVREP), q at
// HaloArgs::q_ld (O_QLD): gqa 8 on this gqa-4 kernel, under attn_paged_qrow.
extern "C" int halo_attn_decode_h128_gqa4_rep_fp8kv(const HaloArgs* a, hipStream_t s) {
    return decode_launch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT | O_QLD | O_KVREP, 0, 1>(a, s);
}
extern "C" int halo_attn_decode_h128_gqa4_rep_bf16kv(const HaloArgs* a, hipStream_t s) {
    return decode_launch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT | O_QLD | O_KVREP, 1, 1>(a, s);
}

// ============================================================== the plugin ABI entry points

extern "C" int64_t halo_rad_attn_decode_h128_scratch(const RadArgs* a) {
    return decode_scratch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a);
}
extern "C" int halo_rad_attn_decode_h128_fp8(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = decode_common<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, &r);
    if (rc != RAD_OK) return rc;
    /* Absent descales mean 1.0; see the prefill twins. */
    return halo_attn_decode_h128_gqa4_fp8kv(&r, (hipStream_t)s);
}
extern "C" int halo_rad_attn_decode_h128_bf16(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = decode_common<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, &r);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h128_gqa4_bf16kv(&r, (hipStream_t)s);
}

/* Head_dim 128 at 8 queries per kv head under attn_paged_qrow: the gqa-4 kernel, each kv head
 * mapped onto two query groups (kv_rep 2), q at its own row pitch. */
extern "C" int64_t halo_rad_attn_qrow_h128_gqa8_scratch(const RadArgs* a) {
    return decode_scratch<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, 0, 2, true);
}
extern "C" int halo_rad_attn_qrow_h128_gqa8_fp8(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = decode_common<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, &r, 0, 2, true);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h128_gqa4_rep_fp8kv(&r, (hipStream_t)s);
}
extern "C" int halo_rad_attn_qrow_h128_gqa8_bf16(const RadArgs* a, RadStream s) {
    HaloArgs r;
    int rc = decode_common<B_HEAD_DIM, B_GQA, B_BLOCK_SIZE, B_OPT, 1>(a, &r, 0, 2, true);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h128_gqa4_rep_bf16kv(&r, (hipStream_t)s);
}
