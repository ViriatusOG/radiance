/* r3d_plugin.h -- what libr3d's translation units share.
 *
 * The row table (r3d_rows.cpp) and the operand descriptions (r3d_shapes.cpp) are pure host C++ and
 * include only this file and the ABI; the launch shims include r3d.h and so <hip/hip_runtime.h>.
 * The split lets the constraint table compile and be checked without ROCm, and is why every shim is
 * declared here rather than beside its definition.
 *
 * Nothing in this header may include r3d.h.
 */
#ifndef RAD_R3D_PLUGIN_H
#define RAD_R3D_PLUGIN_H

#include "r3d_args.h"

/* ================================================================== the priority ladder
 *
 * The selector ranks candidate rows by an integer priority (spec §2.1); where two rows of one op
 * can match a geometry, it decides which serves. Each group states the reason for its order.
 * Values are spaced by ten so a row can be slipped between two without renumbering.
 *
 * libr3d ships one row per (op, dtype) pair far more often than libr4d does: the matrix work runs
 * on the dot2 path, whose lane map takes a dtype as a template argument, so the kernel that serves
 * bf16 serves f16 as is, and there is nothing to specialise a second row for. Where two rows of
 * one op do exist, the reason is the geometry, and it is written next to the pair.
 */
enum {
    /* The common case: this op, these dtypes, one row. Nothing competes with it. */
    R3D_PRIO_ONLY = 50,

    /* `attn_paged` is banded on q_len, and the decode row's bound (q_len <= 16) is a bucket
     * boundary, not a partition: a speculative verify at 8 query tokens matches the prefill row
     * too, and the decode kernel -- split-KV, one head group a workgroup, no query tile -- is the
     * faster one there. Decode therefore outranks prefill, as it does in libr4d. */
    R3D_PRIO_ATTN_DECODE  = 70,
    R3D_PRIO_ATTN_PREFILL = 60,

    /* The skinny bf16 GEMM's band is M <= 16 and the tiled one's starts at M >= 16, so the two
     * meet at exactly 16 rows and the skinny kernel wins there (it keeps the whole weight stream
     * in flight with one wave a row and has no LDS staging to pay for). The overlap is deliberate:
     * a caller that wants the tiled kernel at 16 rows asks with M=17, the same convention libr4d
     * uses at its own m64/tiled boundary. */
    R3D_PRIO_GEMM_M16   = 60,
    R3D_PRIO_GEMM_TILED = 40,

    /* The vocab is a GEMM whose N is the vocabulary: one row, M <= 16, split-K across the whole
     * card. The tiled GEMM has no row for these N, so this priority only decides against a future
     * tiled vocab row. */
    R3D_PRIO_VOCAB_M1 = 60,

    /* An fp8 weight row and an int8 weight row never compete: no dtype string reaches both, and a
     * container's encoding picks which is legal. */
    R3D_PRIO_GEMM_FP8 = 60,

    /* The sampler stages are one row each, on one dtype (f32 logits), so there is nothing to rank.
     * The name is here so a second row for a stage -- an i8 logits form, say -- has a ladder to
     * join. */
    R3D_PRIO_SAMPLE = 50
};

/* ================================================================== the dtype codes a C
 * launcher takes
 *
 * The launchers in r3d.h take a dtype as an int, because a `__global__`'s dtype has to be a runtime
 * value for the ops that are one kernel over several (cast, fill, dequant) and a compile-time one
 * everywhere else. These are the codes; they are libr3d's own, not rad_types.h's RAD_DT_* (a
 * launcher's argument list is a private contract between a shim and its kernel, and the kernel
 * switch reads better over a dense 0..n than over the ABI's sparse enum). Every shim translates. */
enum {
    R3D_DT_BF16 = 0,
    R3D_DT_F16  = 1,
    R3D_DT_F32  = 2,
    R3D_DT_I32  = 3,
    R3D_DT_I64  = 4,
    R3D_DT_U8   = 5,   /* byte codes: fp8 e4m3, read with the codec */
    R3D_DT_I8   = 6,
    R3D_DT_E5M2 = 7,
    R3D_DT_N    = 8
};

/* A RadTensor's dtype to the code above, or -1 for one libr3d has no kernel for. */
static inline int r3d_dt_code(uint32_t dt) {
    switch (dt) {
        case RAD_BF16:   return R3D_DT_BF16;
        case RAD_F16:    return R3D_DT_F16;
        case RAD_F32:    return R3D_DT_F32;
        case RAD_I32:
        case RAD_U32:    return R3D_DT_I32;
        case RAD_I64:    return R3D_DT_I64;
        case RAD_F8E4M3: return R3D_DT_U8;
        case RAD_F8E5M2: return R3D_DT_E5M2;
        case RAD_I8:     return R3D_DT_I8;
        case RAD_U8:     return R3D_DT_U8;
        default:         return -1;
    }
}

/* The bytes an element of that code occupies. */
static inline int r3d_dt_size(int code) {
    static const int kSize[R3D_DT_N] = { 2, 2, 4, 4, 8, 1, 1, 1 };
    return (code >= 0 && code < R3D_DT_N) ? kSize[code] : 0;
}

/* The activation an `act` parameter names, as the kernels' template argument. */
enum { R3D_ACT_SILU = 0, R3D_ACT_SIGMOID = 1, R3D_ACT_GELU = 2 };

/* rope's `mode`, same numbering as libref's kModes. libr3d's kernel implements neox and gptj; the
 * rows' constraints say so, so a model that needs mrope is not routed here. */
enum { R3D_ROPE_NEOX = 0, R3D_ROPE_GPTJ = 1 };

/* ================================================================== the row table's entry points
 *
 * Declared here so r3d_rows.cpp, which is host code built without ROCm, can name them; each is
 * defined beside its kernel, in the unit named below, and is documented there. A shim is a
 * `RadLaunchFn`: operands read with r3d_args.h, dtype from the tensor, row count from the operand.
 * A shape hook is a `RadShapeFn`: the extent and content domain of one operand (r3d_shapes.cpp).
 */
extern "C" {

/* --- r3d_model_bf16.hip --- */
int r3d_rad_rmsnorm       (const RadArgs* a, RadStream s);
int r3d_rad_layernorm     (const RadArgs* a, RadStream s);
int r3d_rad_add           (const RadArgs* a, RadStream s);
int r3d_rad_mul           (const RadArgs* a, RadStream s);
int r3d_rad_cast          (const RadArgs* a, RadStream s);
int r3d_rad_silu          (const RadArgs* a, RadStream s);
int r3d_rad_gelu          (const RadArgs* a, RadStream s);
int r3d_rad_sigmoid       (const RadArgs* a, RadStream s);
int r3d_rad_act_mul       (const RadArgs* a, RadStream s);
int r3d_rad_softmax       (const RadArgs* a, RadStream s);
int r3d_rad_scale         (const RadArgs* a, RadStream s);
int r3d_rad_fill          (const RadArgs* a, RadStream s);
int r3d_rad_rope          (const RadArgs* a, RadStream s);
int r3d_rad_kv_store      (const RadArgs* a, RadStream s);
int r3d_rad_embed_lookup  (const RadArgs* a, RadStream s);
int r3d_rad_gather_rows   (const RadArgs* a, RadStream s);
int r3d_rad_scatter_rows  (const RadArgs* a, RadStream s);

/* --- r3d_quant_fp8.hip --- */
int r3d_rad_quant_act_fp8   (const RadArgs* a, RadStream s);
int r3d_rad_quant_act_i8    (const RadArgs* a, RadStream s);
int r3d_rad_quant_act_i8g   (const RadArgs* a, RadStream s);
int r3d_rad_gated_quant_fp8 (const RadArgs* a, RadStream s);
int r3d_rad_dequant         (const RadArgs* a, RadStream s);

/* --- r3d_gemm_bf16_nt_m16.hip, r3d_gemm_fp8a16_nt_m1.hip --- */
int r3d_rad_gemm_nt_bf16_m16 (const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_fp8    (const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_i8     (const RadArgs* a, RadStream s);

/* --- r3d_shapes.cpp --- */
int r3d_shape_rmsnorm         (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_layernorm       (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_add             (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_mul             (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_act_mul         (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_cast            (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_silu            (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gelu            (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_sigmoid         (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_softmax         (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_scale           (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_fill            (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_rope            (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_kv_store        (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_embed_lookup    (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gather_rows     (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_scatter_rows    (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_quant_act_fp8   (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_quant_act_i8    (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_quant_act_i8g   (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gated_quant_fp8 (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_dequant         (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gemm_nt         (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gemm_nt_q       (const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gemm_nt_q_mxfp4a8(const RadParam* p, int n_p, int operand, RadOpdDesc* out);

/* --- r3d_attn_paged_h128.hip, r3d_vocab_bf16.hip, r3d_sample_f32.hip --- */
extern "C" int r3d_shape_silu_mul          (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_attn_paged        (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_gemm_nt_q_gated (const RadParam*, int, int, RadOpdDesc*);
/* The vocab projection at w2a8, decode. The layout hook is this library's scoped exception to
 * "no layout hooks" (the other two are the fp8 weight's and the logits head's, below): it repacks
 * nothing, it presents the container's `scale` and `zero` planes as the single joined `b_scale`
 * operand the way rad_planes stores them, transcribed from
 * ref_layout_gemm_q. Without it the plan has a placement with no reader. */
extern "C" int r3d_layout_gemm_nt_q_w2a8(const RadParam*, int, int, const RadEncoding*,
                                         const int* sel, const RadTensor* planes, int n_planes,
                                         RadLayout*);
extern "C" int r3d_shape_dflash_select     (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_row_topk          (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_argmax     (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_temp       (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_topk       (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_topp       (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_minp       (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_typical    (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_xtc        (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_pick       (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_penalties  (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_dry        (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_ngram_ban  (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_mask       (const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_sample_logprobs   (const RadParam*, int, int, RadOpdDesc*);

/* --- r3d_sample_f32.hip: the draw spine (row_topk, argmax, temp, topk, pick, the four narrowings)
     and r3d_model_bf16.hip's pre-filters (penalties, dry, ngram_ban, mask, logprobs) --- */
int r3d_rad_gemm_nt_q_w2a8     (const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_gated  (const RadArgs* a, RadStream s);
int r3d_rad_dflash_select      (const RadArgs* a, RadStream s);
int r3d_rad_row_topk           (const RadArgs* a, RadStream s);
int r3d_rad_sample_argmax      (const RadArgs* a, RadStream s);
int r3d_rad_sample_temp        (const RadArgs* a, RadStream s);
int r3d_rad_sample_topk        (const RadArgs* a, RadStream s);
int r3d_rad_sample_pick        (const RadArgs* a, RadStream s);
int r3d_rad_sample_topp        (const RadArgs* a, RadStream s);
int r3d_rad_sample_minp        (const RadArgs* a, RadStream s);
int r3d_rad_sample_typical     (const RadArgs* a, RadStream s);
int r3d_rad_sample_xtc         (const RadArgs* a, RadStream s);
int r3d_rad_sample_penalties   (const RadArgs* a, RadStream s);
int r3d_rad_sample_dry         (const RadArgs* a, RadStream s);
int r3d_rad_sample_ngram_ban   (const RadArgs* a, RadStream s);
int r3d_rad_sample_mask        (const RadArgs* a, RadStream s);
int r3d_rad_sample_logprobs    (const RadArgs* a, RadStream s);

/* --- r3d_model_bf16.hip: the fused MLP activation and the fused norm-quantiser --- */
/* The paged attention's two entry points, banded on q_len: the decode kernel is split-KV over the
 * cache with one head group a workgroup, the prefill kernel stages a 16-token query tile. */
int r3d_rad_attn_decode_h128       (const RadArgs* a, RadStream s);
int r3d_rad_attn_prefill_h128      (const RadArgs* a, RadStream s);

/* --- r3d_vocab_bf16.hip: the vocabulary projection, both heights. The shape hook is a
 * transcription of ref_logits_gemm_shape -- three plain tensors, nothing joined -- so the band the
 * engine declares is the band kbench fills, which is the opposite of the w2a8 row's null hook: there
 * the describable byte count is the trap, here it is the point. */
/* --- r3d_gemm_fp8a8_tiled.hip: the quantised GEMM above the decode heights, and the reader for
 * the container's fp8 weight: codes and per-128x128 scale as two planes behind `b_scale`. The
 * reference's own gemm_nt_q hook answers nothing for this dtype -- ref_layout_gemm_q reads zp_grid,
 * and zp_grid has no fp8a8 entry -- which is why the row carries its own. */
int r3d_rad_gemm_nt_q_fp8_tiled  (const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_w2a8_tiled  (const RadArgs* a, RadStream s);
extern "C" int r3d_layout_gemm_q_fp8_weight(const RadParam*, int, int, const RadEncoding*,
                                            const int*, const RadTensor*, int, RadLayout*);

/* --- r3d_gemm_mxfp4a8.hip: the OCP-MXFP4 weight against the fp8 activation, both heights. The
 * weight is two plain u8 planes -- e2m1 codes [N, K/2] and the e8m0 scale [N, K/32] -- so the row
 * reads them in place and carries no layout hook; the shape hook is its own because the weight's
 * two planes are not the fp8 grid's E4M3 codes and block scale. */
int r3d_rad_gemm_nt_q_mxfp4a8_m16  (const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_mxfp4a8_tiled(const RadArgs* a, RadStream s);

/* --- r3d_gemm_inta8.hip: the symmetric int4 and int8 weight grids at decode heights. The weight
 * is signed codes (I4, two a byte, or I8, one a byte) against a plain f16 scale per (row, 128 of
 * K) -- no zero point and no a_sum, so both planes are read in place and the rows carry no layout
 * hook. */
int r3d_rad_gemm_nt_q_w4a8_m16(const RadArgs* a, RadStream s);
int r3d_rad_gemm_nt_q_w8a8_m16(const RadArgs* a, RadStream s);
int r3d_shape_gemm_nt_q_w4a8(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int r3d_shape_gemm_nt_q_w8a8(const RadParam* p, int n_p, int operand, RadOpdDesc* out);

/* The three writers that go with the three plane hooks, one per row, each delegating to
 * rad_planes_relayout: the loader converts a quantised weight into the joined operand the plan names
 * at load, and only the row that reads it can say how. See abi/rad_abi.h. */
extern "C" int r3d_relayout_logits_head(const RadParam*, int, int, const RadEncoding*, const int*,
                                        const RadTensor*, int, void*, int64_t);
extern "C" int r3d_relayout_gemm_q_fp8_weight(const RadParam*, int, int, const RadEncoding*,
                                              const int*, const RadTensor*, int, void*, int64_t);
extern "C" int r3d_relayout_gemm_nt_q_w2a8(const RadParam*, int, int, const RadEncoding*,
                                           const int*, const RadTensor*, int, void*, int64_t);

int r3d_rad_logits_gemm_bf16     (const RadArgs* a, RadStream s);
/* The head's reader: `output.weight` at a llama_dense_fp8 checkpoint is E4M3 codes and a bf16 scale
 * per 128x128 block, stored as two container planes behind the one `lm_head` operand, and the mapped
 * tier will not place a weight a loaded library cannot read in place. Transcribed from
 * ref_layout_head (libref's own logits row carries that hook), with the statics it uses inlined
 * beside it. It moves no byte: a round-trip through its relayout writer (above) reproduces the
 * planes byte for byte, and that writer exists for the round-trip, not for a move. */
extern "C" int r3d_layout_logits_head(const RadParam*, int, int, const RadEncoding*,
                                      const int*, const RadTensor*, int, RadLayout*);
int r3d_rad_gemm_nt_bf16_tiled   (const RadArgs* a, RadStream s);
int r3d_shape_logits_gemm_bf16  (const RadParam*, int, int, RadOpdDesc*);

int r3d_rad_silu_mul             (const RadArgs* a, RadStream s);
int r3d_rad_rmsnorm_quant_fp8    (const RadArgs* a, RadStream s);
int r3d_shape_rmsnorm_quant_fp8  (const RadParam*, int, int, RadOpdDesc*);

}  /* extern "C" */

#endif  /* RAD_R3D_PLUGIN_H */
