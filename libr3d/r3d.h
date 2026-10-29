/* r3d.h -- the entry points libr3d exports, and the launchers behind them.
 *
 * Two layers, in every unit, as in libr4d:
 *
 *   The launcher (`r3d_<op>(...)`, this header) takes pointers as longs, a geometry as ints and a
 *   stream as a long, checks what a `__global__` cannot (nulls, non-positive extents, a pitch that
 *   does not describe the operand), computes the grid, and issues one kernel with
 *   hipLaunchKernelGGL on the stream it was handed. It returns 0, or a negative launcher code:
 *   -1 a null pointer, -2 a bad extent, -3 a bad pitch, -4 the HIP launch failed. It allocates
 *   nothing and synchronises on nothing.
 *
 *   The shim (`r3d_rad_<op>(const RadArgs*, RadStream)`, declared in the unit) is the RadLaunchFn
 *   the row table names. It reads the operands with r3d_args.h, takes the dtype from the tensor and
 *   not from the `dtype` parameter, takes the row count from the operand and not from `M` (which is
 *   the band's upper bound), checks strides, translates the ABI dtypes to the codes below, and
 *   returns a RAD status.
 *
 * A launcher is declared here and defined beside its kernel, in the same translation unit as the
 * shim that calls it: there is no adapter, and an out-of-tree author reading one unit sees the whole
 * path from RadArgs to the grid.
 *
 * Streams: a RadStream is the engine's queue, usable with hipLaunchKernel (and so <<<>>>),
 * hipMemcpyAsync, hipMemcpy2DAsync and hipMemsetAsync and with nothing else in HIP
 * (docs/PLUGIN.md §5). libr3d passes it to the launch and nowhere else.
 */
#ifndef RAD_R3D_H
#define RAD_R3D_H

#include "r3d_plugin.h"

extern "C" {

/* ================================================================== r3d_model_bf16.hip
 *
 * One workgroup per row. Every launcher takes a row pitch in elements for each operand that can be
 * a slice (a column slice of a fused projection is the usual case), 0 meaning "dense, so n".
 */

/* y = x * rsqrt(mean(x^2) + eps) * (w + wadd) * mult. `wdt` is R3D_DT_BF16 or R3D_DT_F32 for a
 * gain, R3D_DT_N for none (the gain is then 1, and wadd is refused by the shim, as libref does). */
int r3d_rmsnorm_bf16(long x, long w, long y, int M, int n, float eps, float wadd, float mult,
                     int x_ld, int y_ld, int wdt, long stream);

/* The same at f32 activations, for an f32 checkpoint's residual stream: the row is the op's, the
 * store narrows nothing. `wdt` is the gain's dtype code, R3D_DT_N for none. */
int r3d_rmsnorm_f32(long x, long w, long y, int M, int n, float eps, float wadd, float mult,
                    int x_ld, int y_ld, int wdt, long stream);

/* y = (x - mean) / sqrt(var + eps) * w + b, the mean and variance both taken in the reference's
 * two passes. `b` may be 0 (no bias). */
int r3d_layernorm_bf16(long x, long w, long b, long y, int M, int n, float eps,
                       int x_ld, int y_ld, long stream);

/* y = a + b, and y = a * b. `bcast` says b carries n elements and rides every row. `div` is the
 * `add` op's optional divisor, 0 meaning none: with it, y = rd(rd(a + b) / div). */
int r3d_add_bf16(long a, long b, long y, int M, int n, int a_ld, int b_ld, int y_ld, float div,
                 int bcast, long stream);
int r3d_add_f32(long a, long b, long y, int M, int n, int a_ld, int b_ld, int y_ld, float div,
                int bcast, long stream);
int r3d_mul_bf16(long a, long b, long y, int M, int n, int a_ld, int b_ld, int y_ld, int bcast,
                 long stream);
int r3d_mul_f32(long a, long b, long y, int M, int n, int a_ld, int b_ld, int y_ld, int bcast,
                long stream);

/* y = x, one dtype to another, over M rows of n at the given pitches. The dtypes are R3D_DT_*;
 * an integer pair goes straight through (ids above 2^24 survive an i64 -> i32 copy), everything
 * else widens to f32 and narrows once on the way out. */
int r3d_cast(long x, long y, int M, int n, int x_ld, int y_ld, int from_dt, int to_dt, long stream);

/* The unary activations, f32 in, narrowed once by the store: y = x*sigmoid(x), 0.5x(1+erf(x/sqrt2))
 * and 1/(1+exp(-x)) respectively. */
int r3d_silu_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream);
int r3d_gelu_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream);
int r3d_sigmoid_bf16(long x, long y, int M, int n, int x_ld, int y_ld, long stream);
int r3d_silu_f32(long x, long y, int M, int n, int x_ld, int y_ld, long stream);

/* y = act(gate) * up, `act` R3D_ACT_SILU, R3D_ACT_GELU (the exact erf form) or R3D_ACT_SIGMOID.
 * The two inputs are the halves of a fused [M, 2n] projection, so their pitches are separate. */
int r3d_act_mul_bf16(long gate, long up, long y, int M, int n, int g_ld, int u_ld, int y_ld,
                     int act, long stream);

/* y[i] = silu(gate_up[i]) * gate_up[n + i]: the llama MLP's gated product over one fused plane,
 * the gate the FIRST half, which is the order rad-convert fuses gate and up in. The activation is
 * not narrowed before the product: that is `act_mul`'s order, over two operands, and it stands for
 * the unfused pair that stored the activation. `gu_ld` is the FUSED row's pitch (>= 2n), `y_ld` the
 * output's; 0 takes each one's dense width. 0, -1, -2, -3, -4. */
int r3d_silu_mul_bf16(long gate_up, long y, int M, int n, int gu_ld, int y_ld, long stream);

/* Row-wise softmax over f32, maximum subtracted, the denominator summed in ascending order. */
int r3d_softmax_f32(long x, long y, int M, int n, int x_ld, int y_ld, long stream);

/* y = x * s, where `s` is the op's [1] weight plane, in device memory, of dtype `s_dt`
 * (R3D_DT_BF16 or R3D_DT_F32). The scalar is read by the kernel, not by the launcher: a launch may
 * not copy device memory to the host, and the reference reads the same one value on the host because
 * its operands are host planes. Every lane loads the same address, which is one uniform load per
 * block; `s` is 0 only for a call the shim has already refused.
 *
 * y = v (fill) is the other scalar-carried op, and its value is a parameter, so it is by value. */
int r3d_scale_bf16(long x, long s, long y, int M, int n, int s_dt, int x_ld, int y_ld, long stream);
int r3d_scale_f32(long x, long s, long y, int M, int n, int s_dt, int x_ld, int y_ld, long stream);
int r3d_fill(long y, int M, int n, int y_ld, int dt, float v, long stream);

/* In-place rotary on a fused qkv row: q, then k, then v untouched. `mode` is R3D_ROPE_NEOX or
 * R3D_ROPE_GPTJ; `rot` the rotary width (<= head_dim, even). `scale` is the position divisor: the
 * angle of pair i is (pos / scale) * theta^-i, which is libref's proportional ladder at a factor
 * other than 1 (ref_rope's `scale` divides the position, and YaRN's magnitude is carried by
 * `yarn_factor`, which this kernel does not implement). `pos` is M i32 positions, one a row. */
int r3d_rope_bf16(long qkv, long pos, int M, int head_dim, int n_head, int n_head_kv, float theta,
                  float scale, int mode, int rot, int qkv_ld, long stream);

/* K and V into a paged cache, at slot_mapping's slots. The cache is [blocks, kv_heads, block_size,
 * 2*head_dim] with the strides the shim read off the operand and passes as sb/sh/so/sd; a negative
 * slot is skipped, and so is one at or past `slot_max` = n_blocks * block_size, which the launcher
 * computes from M and block_size exactly as `r3d_shape_kv_store` sizes the cache. The reference
 * *refuses* such a slot on the host, where it can see the block count; a launch cannot refuse a
 * slot it has not read, and writing past the cache is the one failure mode worse than a value
 * dropped, so the bound is the kernel's and the divergence is a slot lost rather than a step lost. `kv_dt` R3D_DT_BF16 stores the values, R3D_DT_U8 encodes them as E4M3 with the
 * reciprocal of k_scale / v_scale, R3D_DT_F16 as f16. */
int r3d_kv_store(long k, long v, long sm, long kv, int M, int head_dim, int n_head_kv,
                 int block_size, int kv_dt, float k_scale, float v_scale, int k_ld, int v_ld,
                 long long slot_max, long sb, long sh, long so, long sd, long stream);

/* x[r] = wte[ids[r] - vocab_offset] * wscale, narrowed once. A row whose id is out of the table's
 * range is zeroed, as libref does. */
int r3d_embed_lookup(long ids, long wte, long x, int M, int n_embd, int n_vocab, int vocab_offset,
                     float wscale, float mult, int x_ld, int w_ld, int dt, long stream);

/* y[r] = x[idx[r]] and x[idx[r]] = v[r], rows of n elements.
 *
 * `src_rows` and `dst_rows` are the plane's own row count. Only the shim can supply it -- it has the
 * operand's numel, the launcher has a pointer -- and the kernel needs it, because the reference
 * reads the index plane on the host and refuses an index outside the row count, while a launch
 * cannot refuse an index it has not read. So the bound travels to the kernel and the kernel acts on
 * it: a gather at idx >= src_rows takes the padding path (the zero row libref's padding form
 * writes), a scatter at idx >= dst_rows drops the row, which is the same refusal seen from inside a
 * launch. The worst failure here is a read or a write past the plane, and a dropped row is what
 * stands between this card and it. 0 means unchecked: the shim's bug to avoid, not a caller's to
 * supply. */
int r3d_gather_rows(long x, long idx, long y, int M, int n, int x_ld, int y_ld, int dt,
                    long long src_rows, long stream);
int r3d_scatter_rows(long v, long idx, long x, int M, int n, int v_ld, int x_ld, int dt,
                     long long dst_rows, long stream);

/* ================================================================== r3d_quant_fp8.hip
 *
 * All three quantisers write a plain row-major code plane and one f32 scale per (row, group); no
 * fragment permutation, because a dot2 GEMM reads codes in order.
 */

/* Per-(row, group) symmetric quantisation. scale = amax * (1/top) with top 448 for E4M3 and 127
 * for int8, the codes v * (1/scale) clamped to +-top and rounded by the codec. An all-zero group
 * gets scale 1. */
int r3d_quant_act_fp8(long x, long q, long s, int M, int n, int group, int x_ld, int s_ld,
                      long stream);
int r3d_quant_act_i8(long x, long q, long s, long asum, int M, int n, int group, int x_ld, int s_ld,
                     long stream);

/* quant_act_i8g: the same code at `group` granularity -- amax / 127 per (row, group), 1 for an
 * all-zero group -- stored row-major. This is the plane libr3d's i8a8 GEMM reads; `quant_act_i8`
 * above is the per-row plane the WMMA-ordered ops read, and the two are not interchangeable.
 * 0, -1, -2, -3, -4. */
int r3d_quant_act_i8g(const void* x, void* q, void* s, int M, int n, int group,
                      long long x_ld, long long s_ld, long stream);

/* The gated form: gate_up is [M, 2n] (gate in the first n, up in the second), and what is
 * quantised is act(gate) * up. `q`/`s` carry that product's codes and scales; `out_bf16`, when
 * present, also receives the product in bf16. */
int r3d_gated_quant_fp8(long gate_up, long q, long s, long out_bf16, int M, int n, int group,
                        int act, long stream);

/* The residual add, the RMS norm and the E4M3 quantiser in one pass, so the normalised row never
 * goes back to VRAM. `res` is the residual plane, INOUT, and 0 when the op has no residual: with it
 * the sum is STORED, narrowed to bf16, BEFORE the variance reads it, so the norm is over the bf16
 * sum -- the `add` + `rmsnorm` pair, not `rmsnorm_add`, which normalises the unrounded sum. `ob`,
 * when present, receives exactly the bf16 values the codes were taken over. Pitches are in elements
 * of their own operand, `s_ld` in scales, so a dense scale plane is n/group. `group` must divide n:
 * a partial group is the standalone quantiser's zero-padded store, which this pass does not have.
 * 0, -1, -2, -3, -4. */
int r3d_rmsnorm_quant_fp8(long x, long res, long w, long q, long s, long ob, int M, int n,
                          int group, float eps, float wadd, int x_ld, int res_ld, int ob_ld,
                          int q_ld, int s_ld, int wdt, long stream);

/* y = code * scale, the read of a container's own quantised weight: group-wide f32 scales, E4M3 or
 * int8 codes. */
int r3d_dequant(long q, long s, long y, int M, int n, int group, int q_dt, int s_ld, int y_ld,
                long stream);

/* ================================================================== r3d_gemm_*.hip
 *
 * y[M,N] = a[M,K] @ b[N,K]^T. One output element a lane, k ascending, f32 accumulate, the sum
 * narrowed once by the store. Split-K (sk > 0) writes partials to `scratch` and reduces them in a
 * second kernel on the same stream; the row's scratch hook sizes the arena for the largest sk the
 * band allows.
 */

/* bf16 (and f16) operands. `nt` 1 streams the weight with non-temporal loads. */
int r3d_gemm_nt_bf16_m16(long a, long b, long y, int M, int N, int K, int a_ld, int y_ld, int sk,
                         int wv, int nt, long scratch, long stream);
int r3d_gemm_nt_bf16_tiled(long a, long b, long y, int M, int N, int K, int a_ld, int y_ld, int sk,
                           int kb, int nt, long scratch, long stream);

/* A block-scaled fp8 weight: b is E4M3 codes, b_scale one scale per (pb1 rows of N, pb2 of K)
 * block, pb1/pb2 the block sizes the container states (absent, both default to `group`). The
 * scale plane is **bf16** in the container's quantised family -- `gemm_q_shape` writes
 * `o.t(f32s ? RAD_F32 : RAD_BF16, {nb, kb})` -- and f32 only under the separate dtype `fp8a8f32`.
 * a is bf16 (fp8a16, no a_scale) or E4M3 codes with per-(row, group) f32 a_scale (fp8a8). */
int r3d_gemm_nt_q_fp8(long a, long a_scale, long b, long b_scale, long y, int M, int N, int K,
                      int group, int pb1, int pb2, int a_ld, int y_ld, int sk, int nt,
                      long scratch, long stream);

/* The same kernel with the scale plane's dtype named, which the 17-argument entry above cannot
 * state: it forwards RAD_F32, and the container's quantised weights carry bf16 block scales. Land
 * new callers on this one; the older entry is kept for callers written against its RAD_F32
 * behaviour. bs_dt is the plane's RadDataType: RAD_BF16 reads bf16, anything else -- 0 included --
 * reads f32. */
int r3d_gemm_nt_q_fp8_ex(long a, long a_scale, long b, long b_scale, long y, int M, int N, int K,
                         int group, int pb1, int pb2, int a_ld, int y_ld, int sk, int nt,
                         long scratch, int bs_dt, long stream);

/* The int8 weight twin, with the optional a_sum / b_ref row sums an int8 GEMM with a zero point
 * needs (libquant's symmetric form leaves them null). */
int r3d_gemm_nt_q_i8(long a, long a_scale, long b, long b_scale, long a_sum, long b_ref, long y,
                     int M, int N, int K, int group, int a_ld, int y_ld, int sk, int nt,
                     long scratch, long stream);

/* ================================================================== r3d_attn_paged_h128.hip
 *
 * Paged causal attention. `kv` is the cache the shim read its strides from; `bt` a
 * [num_seqs, blocks] block table; `seqused` per-sequence lengths; cu_seqlens, when present, the
 * ragged query layout; k_descale / v_descale the per-(seq, head) fp8 cache descales.
 */
int r3d_attn_decode_h128(long q, long kv, long bt, long seqused, long k_desc, long v_desc,
                         long cu, long out, int num_seqs, int q_len, int head_dim, int gqa,
                         int n_head, int block_size, int bt_cols, int bt_rs, int bt_cs,
                         int max_ctx, int window, int causal,
                         float scale, int kv_dt, long q_ld, long out_ld, int splits, long scratch,
                         long stream);
int r3d_attn_prefill_h128(long q, long kv, long bt, long seqused, long k_desc, long v_desc,
                          long cu, long out, int num_seqs, int q_len, int head_dim, int gqa,
                          int n_head, int block_size, int bt_cols, int bt_rs, int bt_cs,
                          int max_ctx, int window, int causal,
                          float scale, int kv_dt, long q_ld, long out_ld, long scratch,
                          long stream);

/* ================================================================== r3d_gemm_fp8a8_tiled.hip
 *
 * The quantised GEMM at prefill heights. b_row/bs_row are the container's row strides, 0 meaning
 * dense; `group` is the scale block, 128 on every band this checkpoint declares. The scales arrive
 * as the container's two planes behind one operand, which is what the row's layout hook presents;
 * the plain two-tensor form is the same entry with bs_row dense.
 */
int r3d_gemm_nt_q_fp8_tiled(long a, long a_scale, long b_codes, long b_scale, long y, int M, int N,
                            int K, int group, int a_ld, long long b_row, int as_ld, long long bs_row,
                            int y_ld, int nt, long stream);

/* ================================================================== r3d_gemm_mxfp4a8.hip
 *
 * The quantised GEMM at dtype=mxfp4a8: an OCP-MXFP4 weight against the fp8 activation. `b` is the
 * e2m1 codes, two per byte, [N, K/2] u8; `b_scale` the e8m0 scale, one per 32 of K, [N, K/32] u8;
 * `a` E4M3 with `a_scale` f32 per (row, 128 of K); `y` bf16. b_row/e8_row are the row strides in
 * bytes, 0 meaning dense. There is no zero point: the scale is symmetric, and a reserved e8 (0,
 * 255) is refused. M <= 16 takes the m16 launcher, M >= 17 the tiled one.
 */
int r3d_gemm_nt_q_mxfp4a8_m16(long a, long a_scale, long b, long b_scale, long y, int M, int N,
                              int K, int group, int a_ld, int y_ld, long long e8_row, long stream);
int r3d_gemm_nt_q_mxfp4a8_tiled(long a, long a_scale, long b_codes, long b_scale, long y, int M,
                                int N, int K, int group, int a_ld, long long b_row, int as_ld,
                                long long e8_row, int y_ld, int nt, long stream);

/* ================================================================== r3d_gemm_inta8.hip
 *
 * The quantised GEMM at dtype=w4a8 and w8a8: the symmetric int4 and int8 weight grids at decode
 * heights. `b` is the signed codes (I4, two a byte, or I8, one a byte), [N, K]; `b_scale` the f16
 * scale, one per (row, 128 of K), [N, K/128]; `a` int8 in the step order with `a_scale` f32 per
 * row; `y` bf16. b_rb and sb_rb are the row strides (bytes / elements), 0 meaning dense.
 * There is no zero point:
 * the dequant is code times scale. M <= 16 only; the prefill band is the tiled rows' and the
 * 8-bit grid at prefill heights is declared out of this unit.
 */
int r3d_gemm_nt_q_w4a8_m16(long a, long a_scale, long b, long b_scale, long y, int M, int N,
                           int K, int a_ld, int y_ld, long long b_rb, long long sb_rb, long stream);
int r3d_gemm_nt_q_w8a8_m16(long a, long a_scale, long b, long b_scale, long y, int M, int N,
                           int K, int a_ld, int y_ld, long long b_rb, long long sb_rb, long stream);

/* ================================================================== r3d_vocab_bf16.hip
 *
 * The vocab-sharded logits and the top-k that follows them.
 */

/* logits[M, n_vocab] = x[M, n_embd] @ lm_head[n_vocab, n_embd]^T: the decode heights (M <= 16) on
 * the decode arm, a prefill arm above them, over a vocabulary-wide N. */
int r3d_logits_gemm_bf16(long x, long lm, long y, int M, int n_vocab, int n_embd, int x_ld,
                         int y_ld, int sk, int nt, long scratch, long stream);

/* Per-row top-R of a [M, N] logits row: idx (i32) and val (f32), descending, ties to the lower
 * index. `pairs`, when present, also receives the (index, value) dwords a cross-rank merge reads. */
int r3d_row_topk_bf16(long x, long idx, long val, long pairs, int M, int N, int R, int vocab_off,
                      int x_ld, int rc, int nch, long scratch, long stream);

/* The cross-rank top-1: one id and one value a row, out of [M, n] candidate pairs. */
int r3d_pair_argmax(long pairs, long idx, long val, int M, int n, long stream);

/* ================================================================== r3d_sample_f32.hip
 * The sampler, and the vocabulary row it starts from.
 *
 * The shapes are libref's, operand for operand, and so is the `params` plane:
 * one `RadSampleParams` a row, 96 bytes of f32 words, which every stage reads by field. `logits` is
 * f32 and the candidate planes are word-typed -- `cand_idx` i32, `cand_val` f32, the pair planes
 * i32 holding an id's bits beside a logit -- because that is what the reference writes.
 *
 * The stages over the full vocabulary (argmax, temp, topk, logprobs, penalties, dry, ngram_ban) are
 * one block a row and are bound by the plane: at the band MiniCPM5-2B asks for, 2048 rows of 130560
 * f32, the logits are 1.02 GiB and one pass is the floor. The stages over the candidate set (the
 * four narrowings and pick) are one block a row of `n_cand` and are bound by the arithmetic, which
 * is why `pick` keeps the reference's double `exp`: the law is `(float)std::exp((double)x)`, and a
 * hardware `v_exp_f32` two ulp away moves a draw whenever a true probability lands within an ulp of
 * the `u < cum` boundary.
 *
 * M comes off the operand, never off the band: the band's `M` is a range's bound and a step's row
 * count is what the planes hold. 0, -1 null pointer, -2 bad extent, -3 pitch, -4 launch failed.
 */
int r3d_row_topk(long x, long idx, long val, long pairs, int M, int N, int R, int x_dt,
                 long long x_ld, long long idx_ld, long long val_ld, long long pairs_ld,
                 int vocab_off, long stream);
int r3d_sample_argmax(long lg, long tok, int M, int n_vocab, long long lg_ld, long stream);
int r3d_sample_temp(long lg, long params, int M, int n_vocab, long long lg_ld, long stream);
int r3d_sample_topk(long lg, long params, long ci, long cv, long pairs, int M, int n_vocab,
                    int n_cand, long long lg_ld, long long ci_ld, long long cv_ld,
                    long long pairs_ld, int vocab_off, long stream);
int r3d_sample_pick(long ci, long cv, long params, long tok, int M, int n_cand,
                    long long ci_ld, long long cv_ld, long stream);

/* The four narrowing stages: one shape, `cand_idx` and `cand_val` in place, one predicate each --
 * top-p a prefix rule in candidate order, min-p a floor under the best candidate's probability,
 * typical the band around the set's entropy, XTC the exclusion-the-model-would-not-choose cut. */
int r3d_sample_topp(long ci, long cv, long params, int M, int n_cand, long long ci_ld,
                    long long cv_ld, long stream);
int r3d_sample_minp(long ci, long cv, long params, int M, int n_cand, long long ci_ld,
                    long long cv_ld, long stream);
int r3d_sample_typical(long ci, long cv, long params, int M, int n_cand, long long ci_ld,
                       long long cv_ld, long stream);
int r3d_sample_xtc(long ci, long cv, long params, int M, int n_cand, long long ci_ld,
                   long long cv_ld, long stream);

/* The pre-filters over the logits row. `hist` is the row's history of global token ids, i32
 * [M, max_hist]; its window, and the rule that a token is penalised at its first occurrence in the
 * window, are libref's (`hist_window`), and the window is placed inside the params plane by the
 * descriptor's `fill_chunk`. `vocab_off` shifts the ids a stage reports, for a split vocabulary. */
int r3d_sample_penalties(long lg, long params, long hist, int M, int n_vocab, long long lg_ld,
                         long long hist_ld, int vocab_off, long stream);
int r3d_sample_dry(long lg, long params, long hist, long breakers, int M, int n_vocab,
                   long long lg_ld, long long hist_ld, int vocab_off, long stream);
int r3d_sample_ngram_ban(long lg, long params, long hist, long ngram, long whitelist, int M,
                         int n_vocab, long long lg_ld, long long hist_ld, int vocab_off,
                         long stream);
int r3d_sample_mask(long lg, long params, long bitmask, int M, int n_vocab, long long lg_ld,
                    int mask_rows, long stream);
/* The gated MLP band: b is E4M3 codes of N = 2n rows (gate rows [0,n), up rows [n,N)), b_scale one
 * scale per (group, group) block, bf16 as the container writes it. The row is narrowed to bf16 as
 * the unfused pair leaves it, gated (act: R3D_ACT_SILU | R3D_ACT_SIGMOID), narrowed again, and coded
 * per group of `group` columns: q is the codes [M,n], scale the per-(row, group) amax scales, and
 * out_bf16 the narrowed gated tensor when the caller wants it (y_ld is its row pitch, n when the
 * plane is absent). M <= 64, sk == 1; no scratch, because the row never leaves the registers. */
int r3d_gemm_nt_q_gated(long a, long a_scale, long b, long b_scale, long q, long scale,
                        long out_bf16, int M, int N, int K, int group, int act,
                        int a_ld, int q_ld, int y_ld, int sk, int nt, long scratch, long stream);

/* The same kernel naming the scale plane's dtype. The entry above forwards RAD_BF16, which is the
 * container's form for this op; f32 is accepted and equal bit for bit. */
int r3d_gemm_nt_q_gated_ex(long a, long a_scale, long b, long b_scale, long q, long scale,
                           long out_bf16, int M, int N, int K, int group, int act,
                           int a_ld, int q_ld, int y_ld, int sk, int nt, long scratch,
                           int bs_dt, long stream);

/* DFlash2's greedy walk over the candidate lattice. `hp` may be 0: the plane of ones the reference
 * computes when the operand is absent. The rows of cand, unary and tokens are (sequence, step),
 * M * steps of them; `anchor` is a row of M * anchor_stride ids. */
int r3d_dflash_select(long cand, long unary, long hp, long anchor, long pred, long succ,
                      long tokens, int M, int steps, int top_k, int rank, int n_vocab,
                      int anchor_stride, long long cand_ld, long long un_ld, long long hp_ld,
                      long long pred_ld, long long succ_ld, long long tok_ld, long stream);
int r3d_sample_logprobs(long lg, long params, long stats, int M, int n_vocab, int k,
                        long long lg_ld, long long stats_ld, int vocab_off, long stream);

}  /* extern "C" */

#endif  /* RAD_R3D_H */
