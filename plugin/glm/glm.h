/* plugin/glm/glm.h -- GLM-5.3-Flash's own kernels, as a kernel plugin beside libr4d.
 *
 * One op so far, `swiglu_clamped_had_quant_i8g`: GLM's clamped SwiGLU (`swiglu_clamped`) and the
 * rotated int8 quantiser an i8a8h consumer reads (`had_quant_act_i8g`) in one pass. glm5next issues
 * it only where a loaded library declares it (GateQuantFP8::fuse_clamped, rad_op_declared), and no
 * other architecture issues it, so loading this plugin changes no other model's graph. Its bytes are
 * the chain's: each element is swiglu_clamped's expression rounded to bf16, then the 128-wide
 * Hadamard and the codes and scale had_quant_act_i8g writes over that bf16 row.
 *
 * The schema, the row, the argument reader, the operand shapes and the declared chain are here and
 * in glm.cpp; the device launch is glm.hip and the host reference ref_glm.cpp.
 */
#pragma once
#include "rad_abi.h"
#include "rad_plugin.h"

#include <cstdint>

/* RAD_GLM_N_OPS: the ops this plugin defines (their schemas). RAD_GLM_N_DEV: its device rows, which
 * also serve ops others define (mla_attn, whose schema is libref's and libr4d's); RAD_GLM_N_REF: its
 * host reference rows, for its own ops only (libref is mla_attn's reference). */
enum { RAD_GLM_N_OPS = 1, RAD_GLM_N_DEV = 2, RAD_GLM_N_REF = 1 };

/* swiglu_clamped_had_quant_i8g's resolved call. Strides in elements. */
struct RadGlmSwigluQuantCall {
    const uint16_t* gu;     /* [M, 2n] bf16: gate, then up */
    int8_t*         q;      /* [M, n] int8 codes over the rotated product */
    float*          s;      /* [M, n / group] f32 scales */
    uint16_t*       y;      /* [M, n] bf16 unrotated product, or null */
    int64_t M, n, group;
    int64_t gu_ld, q_ld, s_ld, y_ld;
    float   limit;
};

int rad_glm_schema_count();
const RadOpSchema* rad_glm_schema_at(int i);
RadKernelInfo rad_glm_row(int i, RadLaunchFn launch);

/* Reads and checks the operands: RAD_E_SHAPE / RAD_E_DTYPE / RAD_E_INVAL as the launches return. */
int rad_glm_swiglu_quant_args(const RadArgs* a, RadGlmSwigluQuantCall* out);

int rad_glm_swiglu_quant_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int rad_glm_swiglu_quant_replaces(const RadParam* p, int n_p, uint64_t present, int step,
                                  RadFuseStep* out);

/* mla_attn at kv_dtype i8 (glm_mla.hip): the row for an op libref and libr4d define. */
RadKernelInfo rad_glm_mla_i8_row(RadLaunchFn launch, RadScratchFn scratch);

int rad_ref_glm_swiglu_quant(const RadArgs* a, RadStream s);
#ifdef RAD_GLM_WITH_DEVICE
extern "C" int rad_glm_swiglu_quant_launch(const RadArgs* a, RadStream s);
extern "C" int rad_glm_mla_i8_launch(const RadArgs* a, RadStream s);
extern "C" int64_t rad_glm_mla_i8_scratch(const RadArgs* a);
#endif
