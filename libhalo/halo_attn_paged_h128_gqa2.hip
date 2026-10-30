/* Derived from libr4d/r4d_attn_paged_h128_gqa2.hip by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. */
/* halo_attn_paged_h128_gqa2.hip -- head_dim 128, two queries per KV head, paged block 16, no
 * window: the windowless h128/gqa2 rows. Gemma 3's sliding layers take their own unit
 * (halo_attn_paged_h128_gqa2w.hip) with the window bit on, so the global layers stay on the tuned
 * full-attention legs.
 *
 * The device code is the generic body (halo_attn_paged_gqa8.hip) at ratio 2, h128 option set, both
 * bands: the split-KV decode leg comes from the same body. */
#define HALO_ATTN_GQA  2
#define HALO_ATTN_H128 1
#define HALO_ATTN_H256 0

#include "halo_attn_paged_gqa8.hip"
