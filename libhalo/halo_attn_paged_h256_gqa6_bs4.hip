/* Derived from libr4d/r4d_attn_paged_h256_gqa6_bs4.hip by the libhalo rename transform
 * (2026-10-26); see libhalo/README.md. */
// Halo attention at six queries per KV head, head_dim 256, KV page of FOUR tokens: the TP4 leg of
// Qwen3.8-Flash-Next's sparse (QSA) attention, where a four-rank world gives a rank the gqa12
// geometry's ratio halved (12 query heads over 2 KV heads). Geometry only;
// halo_attn_paged_gqa8.hip is the body. Everything but the page size is the gqa12 unit's, and the
// rows state block_size 4.
//
// gfx1151: both bands -- the split-KV decode leg comes from the generic body too.
#define HALO_ATTN_GQA  6
#define HALO_ATTN_H128 0
#define G8_BLOCK_SIZE  4
#define HALO_ATTN_V4   1
// The fused gate+quantise rows beside its unfused pair (attn_paged_gate_quant).
#define HALO_ATTN_GQ   1

#include "halo_attn_paged_gqa8.hip"
