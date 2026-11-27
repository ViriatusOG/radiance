/* r4d_ds41_elem.h -- launches and shape hooks of DeepSeek-V4.1's elem ops; included by r4d_plugin.h
 * inside its extern "C". */
#ifndef R4D_DS41_ELEM_H
#define R4D_DS41_ELEM_H

/* Launches (r4d_ds41_elem.hip). */
int r4d_rad_token_map(const RadArgs* a, RadStream stream);
int r4d_rad_stream_gate_add(const RadArgs* a, RadStream stream);
int r4d_rad_ds41_rope_rows(const RadArgs* a, RadStream stream);
int r4d_rad_quant_roundtrip(const RadArgs* a, RadStream stream);
int r4d_rad_mhc_mix(const RadArgs* a, RadStream stream);
int64_t r4d_rad_mhc_mix_scratch(const RadArgs* a);
int r4d_rad_mhc_collapse(const RadArgs* a, RadStream stream);
int r4d_rad_softmax_pool(const RadArgs* a, RadStream stream);
int r4d_rad_entry_store(const RadArgs* a, RadStream stream);
int r4d_rad_router_scored_topk(const RadArgs* a, RadStream stream);
int r4d_rad_router_scored_topk_vl(const RadArgs* a, RadStream stream);
int r4d_rad_ds41_vit_rope2d(const RadArgs* a, RadStream stream);
int r4d_rad_ds41_pack3x3(const RadArgs* a, RadStream stream);
int r4d_rad_ds41_span_rows(const RadArgs* a, RadStream stream);
int r4d_rad_embed_lookup_planes(const RadArgs* a, RadStream stream);
/* The fused chains (one launch each, byte-identical to the chain r4d_fuse.cpp declares). */
int r4d_rad_rmsnorm_rope_roundtrip(const RadArgs* a, RadStream stream);
int r4d_rad_rope_rows_roundtrip(const RadArgs* a, RadStream stream);
int r4d_rad_mhc_collapse_rmsnorm(const RadArgs* a, RadStream stream);
int r4d_rad_mhc_post_collapse_rmsnorm(const RadArgs* a, RadStream stream);

/* Shapes (r4d_ds41_elem_shapes.inc), libref's ref_<op>_shape operand for operand. */
int r4d_shape_token_map(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_stream_gate_add(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_rope_rows(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_quant_roundtrip(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_mhc_mix(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_mhc_collapse(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_softmax_pool(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_entry_store(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_router_scored_topk(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_router_scored_topk_vl(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_ds41_vit_rope2d(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_ds41_pack3x3(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_ds41_span_rows(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_embed_lookup_planes(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_rmsnorm_rope_roundtrip(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_rope_rows_roundtrip(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_mhc_collapse_rmsnorm(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_mhc_post_collapse_rmsnorm(const RadParam* p, int n_p, int operand, RadOpdDesc* out);

/* The chains the fused rows replace (r4d_fuse.cpp). */
int r4d_fuse_rmsnorm_rope_roundtrip(const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_rope_rows_roundtrip(const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_mhc_collapse_rmsnorm(const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_mhc_post_collapse_rmsnorm(const RadParam*, int, uint64_t, int, RadFuseStep*);

#endif
