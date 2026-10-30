/* Derived from libr4d/r4d_attn_paged_h128_gqa1.hip by the libhalo rename transform (2026-10-26);
 * see libhalo/README.md. */
/* halo_attn_paged_h128_gqa1.hip -- head_dim 128 with ONE query head per KV head: plain MHA.
 *
 * Everything else in this library pages attention through a grouped ratio (2, 4, 6, 8 or 12
 * queries per KV head). The Unlimited-OCR decoder has 10 query heads over 10 KV heads at 128 wide,
 * so the ratio is 1 and none of the existing legs matches it; the engine reports every band of that
 * geometry as having no kernel rather than serving it on the host.
 *
 * The device code is the generic body (halo_attn_paged_gqa8.hip, which is parameterised by
 * HALO_ATTN_GQA rather than fixed at 8) instantiated at ratio 1: the same templated prefill
 * kernel, with GQA=1 substituted. No new kernel, no new masking path.
 *
 * The prefill option set is the h128 set (G8_P_OPT_128), which is the one compiled for this head
 * width. gfx1151: both bands -- the split-KV decode leg is the generic body's, included below. */
#define HALO_ATTN_GQA  1
#define HALO_ATTN_H128 1
#define HALO_ATTN_H256 0

#include "halo_attn_paged_gqa8.hip"
