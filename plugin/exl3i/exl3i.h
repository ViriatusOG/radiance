/* exl3i.h -- what exl3i's host files share. The exl3i kernel library serves `exl3i_dense` (include/exl3i_abi.h):
 * EXL3-V3's weights at any rate in 1/8 steps, mad, on gfx1201 (the dense GEMM kernel with the decode a rate
 * policy, exl3i_nt.hip.h). Containers of the older `exl3` format have no importer in this tree, so they never reach
 * it. */
#ifndef EXL3I_H
#define EXL3I_H

#include "exl3i_abi.h"

#if EXL3I_IFACE_VERSION != 5
#error "exl3i is written against EXL3I_IFACE_VERSION 5 (include/exl3i_abi.h: aq, exl3i_actq, codebook mad, actq's res)"
#endif
#if EXL3I_DECODE_API_VERSION != 4 || EXL3I_ENCODING_VERSION != 4
#error "exl3i is written against EXL3I_DECODE_API_VERSION 4 and EXL3I_ENCODING_VERSION 4"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* The stored record: tiles k-major (nt.c), [rows/16, row words] as the container's plane. Plugin-private dtypes:
 * EXL3I_DT_TILES_R16 the same bytes of a weight whose rows are in R16 order (the encoding's /r16 row order), so the
 * launch writes output s at row exl3i_r16_row(s, N). */
enum { EXL3I_DT_TILES = RAD_DT_PLUGIN_BASE + 0, EXL3I_DT_TILES_R16 = RAD_DT_PLUGIN_BASE + 1 };

int     exl3i_nt_launch(const RadArgs* a, RadStream s);
int     exl3i_aq_launch(const RadArgs* a, RadStream s);                    /* exl3i_actq (AQ) */
int     exl3i_aq_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int     exl3i_aq_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                        const RadTensor* planes, int n_planes, RadLayout* out);
int     exl3i_aq_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                          const RadTensor* planes, int n_planes, void* dst, int64_t dst_bytes);
int     exl3i_aq_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                            const void* src, int64_t src_bytes, const RadTensor* planes, int n_planes);
int64_t exl3i_nt_scratch(const RadArgs* a);
int     exl3i_nt_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                        const RadTensor* planes, int n_planes, RadLayout* out);
int     exl3i_nt_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                          const RadTensor* planes, int n_planes, void* dst, int64_t dst_bytes);
int     exl3i_nt_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                            const void* src, int64_t src_bytes, const RadTensor* planes, int n_planes);
int     exl3i_rerank_launch(const RadArgs* a, RadStream s);                 /* exl3i_rerank (rerank.c) */
int     exl3i_rerank_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
void    exl3i_rows_to_tiles(const uint32_t* rec, int64_t rows, int64_t cols, int m, uint32_t* out);
void    exl3i_tiles_to_rows(const uint32_t* tiles, int64_t rows, int64_t cols, int m, uint32_t* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* EXL3I_H */
