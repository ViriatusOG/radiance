/* r4d_ds41_ggml.h -- launches and shape hooks of the GEMMs over llama.cpp's native blocks (the ggml ops); included by r4d_plugin.h
 * inside its extern "C".
 *
 * gemm_nt_ggml, moe_gemm_ggml, logits_gemm_ggml and embed_lookup_ggml over llama.cpp's own GGUF
 * blocks (r4d_ds41_ggml.hip); the weight layout hooks that relayout a GGUF import's planes into the
 * blocks at load are host code (r4d_ds41_ggml_layout.cpp, built only with libquant's sources,
 * R4D_HAVE_GGML). */
#ifndef R4D_DS41_GGML_H
#define R4D_DS41_GGML_H

/* -- launches (r4d_ds41_ggml.hip): `_gemv` the weight-streaming decode path (M <= 16 rows, or
 * tokens for the grouped GEMM), `_tiled` the 128 x 128 tile path at any M. */
int r4d_rad_gemm_nt_ggml_gemv(const RadArgs*, RadStream);
int r4d_rad_gemm_nt_ggml_tiled(const RadArgs*, RadStream);
int r4d_rad_moe_gemm_ggml_gemv(const RadArgs*, RadStream);
int r4d_rad_moe_gemm_ggml_tiled(const RadArgs*, RadStream);
int r4d_rad_logits_gemm_ggml_gemv(const RadArgs*, RadStream);
int r4d_rad_logits_gemm_ggml_tiled(const RadArgs*, RadStream);
int r4d_rad_embed_lookup_ggml(const RadArgs*, RadStream);

/* -- the weight layout (r4d_ds41_ggml_layout.cpp): libref's ref_ggml_layout / relayout /
 * unrelayout, byte for byte and with the same tag, so either library reads the other's bytes. */
int r4d_ds41_ggml_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                         const int* sel, const RadTensor* planes, int n_planes, RadLayout* out);
int r4d_ds41_ggml_relayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                           const int* sel, const RadTensor* planes, int n_planes, void* dst,
                           int64_t dst_bytes);
int r4d_ds41_ggml_unrelayout(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                             const int* sel, const void* src, int64_t src_bytes,
                             const RadTensor* planes, int n_planes);

/* -- operand descriptions (r4d_ds41_ggml_shapes.inc), libref's: the weight in its float logical
 * form. */
int r4d_shape_gemm_nt_ggml(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_moe_gemm_ggml(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_logits_gemm_ggml(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r4d_shape_embed_lookup_ggml(const RadParam* p, int n_p, int operand, RadOpdDesc* out);

#endif
