/* Derived from libr4d/r4d_attn_paged_h256_gqa6.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// Halo paged attention dispatch for head_dim 256, 6 queries per KV head, paged block 16. The
// KV-dtype variants are template instantiations of one device code. The geometry is a hard gate:
// anything else is rejected.
//
// The query is bf16 in every exact variant. Two families differ in KV dtype: bf16 and 8-bit (fp8,
// the production configuration, and the fp8_v4 and k6_v4 caches). The e4m3-product rows (QK8, PV8)
// run the same body with their operands rounded to e4m3 first: gfx1151 has no fp8 matrix
// instruction, so the legs are emulated on the f16 WMMA (see halo_attn_prefill_h256_gqa6.hip).
//
// This unit is the gfx1151 counterpart of libr4d's and ships both bands. libhalo keeps the split-KV
// decode kernel in a body of its own (halo_attn_decode_h256_gqa6.hip, text shared by every paged
// geometry); this unit includes it at the end and defines this geometry's decode entry points.
//
// `opt` is a bitmask of optimisation steps, defined in halo_attn_prefill_h256_gqa6.hip:
//     prefill 3954187 = f16 operands | mask skip | folded query scale | contiguous-k
//                     fragments | prefetch 2 + scheduling barrier | dot2 row sum | block table in
//                     SGPRs | wide K fetch | V prefetch across the m-tile loop | LDS-scoped
//                     barrier. Not O_KLATE (8-bit K fetched a tile ahead): on gfx1151 its live
//                     K registers took the fp8 and k6_v4 kernels from ~90 to ~300 spilled VGPRs,
//                     and those two ran 2.3x slower than fp8_v4 and bf16 (q_len 512 at 16K:
//                     11.8 -> 4.3 ms without it).
#include "halo.h"
#include "halo_args.h"
#include "halo_attn_args.h"
#include "halo_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "halo_attn_prefill_h256_gqa6.hip"

#define CHK() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

// The geometry of every kernel in this unit. BLOCK_SIZE is not in the entry-point names because
// it is 16 for all of them.
#define A_HEAD_DIM   256
#define A_GQA          6
#define A_BLOCK_SIZE  16
#define A_MAX_DECODE_ROWS 64        // q_len * gqa a single decode workgroup can hold

extern "C" void halo_attn_dims(int* head_dim, int* gqa, int* block_size, int* max_decode_rows) {
    if (head_dim)        *head_dim        = A_HEAD_DIM;
    if (gqa)             *gqa             = A_GQA;
    if (block_size)      *block_size      = A_BLOCK_SIZE;
    if (max_decode_rows) *max_decode_rows = A_MAX_DECODE_ROWS;
}

// ---- prefill ---------------------------------------------------------------------------------
// LDS at head_dim 256, 16-bit K and V: sK = TILE*264*2, sV = 256*(TILE+8)*2. TILE 64 would need
// 68096 B > 65536, so TILE is 48.
#define P_NWARPS 24
#define P_TILE   48
// K prefetch 2, not 4: four fragments in flight make the m-tile loop spill at six waves per SIMD.
#define P_OPT    3954187

// KVP: 0 = an 8-bit cache (XOPT names fp8_v4 or k6_v4), 1 = bf16 cache.
template <int KVP, int XOPT = 0>
static int prefill_launch(const HaloArgs* a, hipStream_t s) {
    if (a->head_dim != A_HEAD_DIM || a->block_size != A_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != A_GQA) return -2;
    // Split-KV when the caller set a->splits > 1 with scratch for the partials
    // (halo_prefill_split_setup); otherwise the query-tiled grid as before.
    return halo_prefill_launch<P_NWARPS, P_TILE, A_HEAD_DIM, A_GQA, A_BLOCK_SIZE, KVP,
                               P_OPT | XOPT>(a, s);
}

/* The split-KV partials (halo_prefill_scratch in the body), one hook per cache row width. */
extern "C" int64_t halo_rad_attn_prefill_h256_scratch(const RadArgs* a) {
    return halo_prefill_scratch<P_NWARPS, P_TILE, A_HEAD_DIM, A_GQA>(a);
}
extern "C" int64_t halo_rad_attn_prefill_h256_v4_scratch(const RadArgs* a) {
    return halo_prefill_scratch<P_NWARPS, P_TILE, A_HEAD_DIM, A_GQA>(a, 400);
}
extern "C" int64_t halo_rad_attn_prefill_h256_k6_scratch(const RadArgs* a) {
    return halo_prefill_scratch<P_NWARPS, P_TILE, A_HEAD_DIM, A_GQA>(a, 352);
}
#define P_SPLIT(a, r) halo_prefill_split_setup<P_NWARPS, P_TILE, A_HEAD_DIM, A_GQA>((a), (r))

extern "C" int halo_attn_prefill_h256_gqa6_fp8kv(const HaloArgs* a, hipStream_t s) {
    return prefill_launch<0>(a, s);
}
extern "C" int halo_attn_prefill_h256_gqa6_bf16kv(const HaloArgs* a, hipStream_t s) {
    return prefill_launch<1>(a, s);
}
extern "C" int halo_attn_prefill_h256_gqa6_fp8v4kv(const HaloArgs* a, hipStream_t s) {
    return prefill_launch<0, O_V4>(a, s);
}
extern "C" int halo_attn_prefill_h256_gqa6_k6v4kv(const HaloArgs* a, hipStream_t s) {
    return prefill_launch<0, O_V4 | O_K6>(a, s);
}

// The e4m3 legs over an 8-bit cache: `legs` 1 Q.K^T, 2 P.V, 3 both. XOPT names the cache.
template <int XOPT>
static int prefill_legs_launch(const HaloArgs* a, hipStream_t s, int legs) {
    if (legs == 1) return prefill_launch<0, XOPT | O_QK8>(a, s);
    if (legs == 2) return prefill_launch<0, XOPT | O_PV8>(a, s);
    return prefill_launch<0, XOPT | O_QK8 | O_PV8>(a, s);
}
extern "C" int halo_attn_prefill_h256_gqa6_fp8kv_qk8(const HaloArgs* a, hipStream_t s) {
    return prefill_legs_launch<0>(a, s, 1);
}
extern "C" int halo_attn_prefill_h256_gqa6_fp8kv_pv8(const HaloArgs* a, hipStream_t s) {
    return prefill_legs_launch<0>(a, s, 2);
}
extern "C" int halo_attn_prefill_h256_gqa6_fp8kv_qk8pv8(const HaloArgs* a, hipStream_t s) {
    return prefill_legs_launch<0>(a, s, 3);
}
extern "C" int halo_attn_prefill_h256_gqa6_fp8v4kv_qk8(const HaloArgs* a, hipStream_t s) {
    return prefill_legs_launch<O_V4>(a, s, 1);
}
extern "C" int halo_attn_prefill_h256_gqa6_fp8v4kv_pv8(const HaloArgs* a, hipStream_t s) {
    return prefill_legs_launch<O_V4>(a, s, 2);
}
extern "C" int halo_attn_prefill_h256_gqa6_fp8v4kv_qk8pv8(const HaloArgs* a, hipStream_t s) {
    return prefill_legs_launch<O_V4>(a, s, 3);
}
extern "C" int halo_attn_prefill_h256_gqa6_k6v4kv_qk8(const HaloArgs* a, hipStream_t s) {
    return prefill_legs_launch<O_V4 | O_K6>(a, s, 1);
}
extern "C" int halo_attn_prefill_h256_gqa6_k6v4kv_pv8(const HaloArgs* a, hipStream_t s) {
    return prefill_legs_launch<O_V4 | O_K6>(a, s, 2);
}
extern "C" int halo_attn_prefill_h256_gqa6_k6v4kv_qk8pv8(const HaloArgs* a, hipStream_t s) {
    return prefill_legs_launch<O_V4 | O_K6>(a, s, 3);
}

// ============================================================== the plugin ABI entry points
//
// The operand reader is halo_attn_args.h, shared by every paged geometry.
/* ------------------------------------------------------------------ paged prefill */

extern "C" int halo_rad_attn_prefill_h256_fp8(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return HALO_NO(RAD_E_SHAPE);
    HaloArgs r;
    int rc = fill_paged(a, &r, 256);
    if (rc != RAD_OK) return rc;
    P_SPLIT(a, &r);
    /* Absent descales mean 1.0, not an error: the kernels read them that way, vLLM defaults both
     * scales to 1.0, and kv_store_fp8 at its default scale needs no descale plane. */
    return halo_attn_prefill_h256_gqa6_fp8kv(&r, (hipStream_t)s);
}

extern "C" int halo_rad_attn_prefill_h256_bf16(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return HALO_NO(RAD_E_SHAPE);
    HaloArgs r;
    int rc = fill_paged(a, &r, 256);
    if (rc != RAD_OK) return rc;
    P_SPLIT(a, &r);
    return halo_attn_prefill_h256_gqa6_bf16kv(&r, (hipStream_t)s);
}

/* The fp8_v4 cache: a token's row is 25 * 256 / 16 = 400 bytes (E4M3 K, int4 V, f16 scales). */
extern "C" int halo_rad_attn_prefill_h256_fp8v4(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return HALO_NO(RAD_E_SHAPE);
    HaloArgs r;
    int rc = fill_paged_row(a, &r, 256, 400);
    if (rc != RAD_OK) return rc;
    P_SPLIT(a, &r);
    return halo_attn_prefill_h256_gqa6_fp8v4kv(&r, (hipStream_t)s);
}

/* The k6_v4 cache: a token's row is 22 * 256 / 16 = 352 bytes (int6 K, int4 V, f16 scales). */
extern "C" int halo_rad_attn_prefill_h256_k6v4(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return HALO_NO(RAD_E_SHAPE);
    HaloArgs r;
    int rc = fill_paged_row(a, &r, 256, 352);
    if (rc != RAD_OK) return rc;
    P_SPLIT(a, &r);
    return halo_attn_prefill_h256_gqa6_k6v4kv(&r, (hipStream_t)s);
}

/* The 8-bit caches with e4m3 products. `legs` is the row's `prefill_fp8`, which its constraint has
 * matched: 1 Q.K^T, 2 P.V, 3 both. `row` is a token's bytes in a (block, head) page: 512 for fp8,
 * 400 for fp8_v4, 352 for k6_v4. */
template <int XOPT>
static int prefill_h256_legs(const RadArgs* a, RadStream s, long long row, int legs) {
    if (check_dims_once() != RAD_OK) return HALO_NO(RAD_E_SHAPE);
    HaloArgs r;
    int rc = fill_paged_row(a, &r, 256, row);
    if (rc != RAD_OK) return rc;
    P_SPLIT(a, &r);
    return prefill_legs_launch<XOPT>(&r, (hipStream_t)s, legs);
}
extern "C" int halo_rad_attn_prefill_h256_fp8_qk8(const RadArgs* a, RadStream s) {
    return prefill_h256_legs<0>(a, s, 512, 1);
}
extern "C" int halo_rad_attn_prefill_h256_fp8_pv8(const RadArgs* a, RadStream s) {
    return prefill_h256_legs<0>(a, s, 512, 2);
}
extern "C" int halo_rad_attn_prefill_h256_fp8_qk8pv8(const RadArgs* a, RadStream s) {
    return prefill_h256_legs<0>(a, s, 512, 3);
}
extern "C" int halo_rad_attn_prefill_h256_fp8v4_qk8(const RadArgs* a, RadStream s) {
    return prefill_h256_legs<O_V4>(a, s, 400, 1);
}
extern "C" int halo_rad_attn_prefill_h256_fp8v4_pv8(const RadArgs* a, RadStream s) {
    return prefill_h256_legs<O_V4>(a, s, 400, 2);
}
extern "C" int halo_rad_attn_prefill_h256_fp8v4_qk8pv8(const RadArgs* a, RadStream s) {
    return prefill_h256_legs<O_V4>(a, s, 400, 3);
}
extern "C" int halo_rad_attn_prefill_h256_k6v4_qk8(const RadArgs* a, RadStream s) {
    return prefill_h256_legs<O_V4 | O_K6>(a, s, 352, 1);
}
extern "C" int halo_rad_attn_prefill_h256_k6v4_pv8(const RadArgs* a, RadStream s) {
    return prefill_h256_legs<O_V4 | O_K6>(a, s, 352, 2);
}
extern "C" int halo_rad_attn_prefill_h256_k6v4_qk8pv8(const RadArgs* a, RadStream s) {
    return prefill_h256_legs<O_V4 | O_K6>(a, s, 352, 3);
}

// ============================================================== the split-KV decode leg
//
// Same geometry, the decode band: the body (halo_attn_decode_h256_gqa6.hip) is text shared by every
// paged geometry, so this unit instantiates it at the geometry it owns and defines that geometry's
// entry points -- the internal `<geom>_*kv` forms the engine dlsyms and the plugin rows the table
// points at. An entry point added to the body itself would be defined by every unit that includes
// it, which does not link.
#include "halo_attn_decode_h256_gqa6.hip"

extern "C" long halo_attn_decode_h256_gqa6_scratch_bytes(const HaloArgs* a) {
    return decode_scratch_bytes<HALO_D_OPT>(a);
}

extern "C" int halo_attn_decode_h256_gqa6_fp8kv(const HaloArgs* a, hipStream_t stream) {
    return decode_launch<A_HEAD_DIM, A_GQA, A_BLOCK_SIZE, HALO_D_OPT, 0>(a, stream);
}
extern "C" int halo_attn_decode_h256_gqa6_bf16kv(const HaloArgs* a, hipStream_t stream) {
    return decode_launch<A_HEAD_DIM, A_GQA, A_BLOCK_SIZE, HALO_D_OPT, 1>(a, stream);
}
extern "C" int halo_attn_decode_h256_gqa6_fp8v4kv(const HaloArgs* a, hipStream_t stream) {
    return decode_launch<A_HEAD_DIM, A_GQA, A_BLOCK_SIZE, HALO_D_OPT | O_V4, 0>(a, stream);
}
extern "C" int halo_attn_decode_h256_gqa6_k6v4kv(const HaloArgs* a, hipStream_t stream) {
    return decode_launch<A_HEAD_DIM, A_GQA, A_BLOCK_SIZE, HALO_D_OPT | O_V4 | O_K6, 0>(a, stream);
}

/* ---- the plugin ABI entry points of the decode band */

extern "C" int64_t halo_rad_attn_decode_h256_scratch(const RadArgs* a) {
    return decode_scratch<A_HEAD_DIM, A_GQA, A_BLOCK_SIZE, HALO_D_OPT>(a);
}
/* The fp8_v4 cache: a token's row is 25 * 256 / 16 = 400 bytes (E4M3 K, int4 V, f16 scales). */
extern "C" int64_t halo_rad_attn_decode_h256_v4_scratch(const RadArgs* a) {
    return decode_scratch<A_HEAD_DIM, A_GQA, A_BLOCK_SIZE, HALO_D_OPT>(a, 400);
}
/* The k6_v4 cache: a token's row is 22 * 256 / 16 = 352 bytes (int6 K, int4 V, f16 scales). */
extern "C" int64_t halo_rad_attn_decode_h256_k6_scratch(const RadArgs* a) {
    return decode_scratch<A_HEAD_DIM, A_GQA, A_BLOCK_SIZE, HALO_D_OPT>(a, 352);
}

extern "C" int halo_rad_attn_decode_h256_fp8(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return HALO_NO(RAD_E_SHAPE);
    HaloArgs r;
    int rc = decode_common<A_HEAD_DIM, A_GQA, A_BLOCK_SIZE, HALO_D_OPT>(a, &r);
    if (rc != RAD_OK) return rc;
    /* Absent descales mean 1.0, not an error: the kernels read them that way and kv_store_fp8 at
     * its default scale needs no descale plane. */
    return halo_attn_decode_h256_gqa6_fp8kv(&r, (hipStream_t)s);
}

extern "C" int halo_rad_attn_decode_h256_bf16(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return HALO_NO(RAD_E_SHAPE);
    HaloArgs r;
    int rc = decode_common<A_HEAD_DIM, A_GQA, A_BLOCK_SIZE, HALO_D_OPT>(a, &r);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h256_gqa6_bf16kv(&r, (hipStream_t)s);
}

extern "C" int halo_rad_attn_decode_h256_fp8v4(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return HALO_NO(RAD_E_SHAPE);
    HaloArgs r;
    int rc = decode_common<A_HEAD_DIM, A_GQA, A_BLOCK_SIZE, HALO_D_OPT>(a, &r, 400);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h256_gqa6_fp8v4kv(&r, (hipStream_t)s);
}

extern "C" int halo_rad_attn_decode_h256_k6v4(const RadArgs* a, RadStream s) {
    if (check_dims_once() != RAD_OK) return HALO_NO(RAD_E_SHAPE);
    HaloArgs r;
    int rc = decode_common<A_HEAD_DIM, A_GQA, A_BLOCK_SIZE, HALO_D_OPT>(a, &r, 352);
    if (rc != RAD_OK) return rc;
    return halo_attn_decode_h256_gqa6_k6v4kv(&r, (hipStream_t)s);
}
