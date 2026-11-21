/* plugin/glm/glm.cpp -- the GLM plugin's schema, row, argument reader, operand shapes and declared
 * chain. Pure host code: a card-less machine loads it to check the schema and run the reference. */
#include "glm.h"

#include <cmath>
#include <cstring>

namespace {

#define PI(k)  { (k), RAD_P_INT, RAD_REQUIRED, 0 }
#define PF(k)  { (k), RAD_P_F64, RAD_REQUIRED, 0 }
#define PS(k)  { (k), RAD_P_STR, RAD_REQUIRED, 0 }
const RadParamSpec kSwigluQuantParams[] = { PI("M"), PI("n"), PF("limit"), PI("group"), PS("dtype") };
#undef PI
#undef PF
#undef PS

/* gate_up then the codes, the scales and the optional unrotated bf16 product: swiglu_clamped's
 * input, had_quant_act_i8g's outputs, and swiglu_clamped's output where a bf16 consumer wants it. */
const RadOperandSpec kSwigluQuantOperands[] = {
    { "gate_up",  RAD_OPD_IN,  0 },
    { "q",        RAD_OPD_OUT, 0 },
    { "scale",    RAD_OPD_OUT, 0 },
    { "out_bf16", RAD_OPD_OUT, 1 },
};

const RadOpSchema kSchemas[RAD_GLM_N_OPS] = {
    { "swiglu_clamped_had_quant_i8g", kSwigluQuantParams, 5, kSwigluQuantOperands, 4,
      "swiglu_clamped then had_quant_act_i8g over its bf16 product, in one pass: min(g, limit) "
      "silu'd times up clamped to [-limit, limit], rounded to bf16; then each `group` of columns "
      "(128, or IQ8R's 256 as two 128-wide halves under one scale) rotated by the unnormalised "
      "128-wide Hadamard and coded int8 at absmax / 127. gate_up is [M, 2n], gate then up; q [M, n] "
      "int8; scale [M, n / group] f32; out_bf16, if passed, [M, n], the unrotated product. n a "
      "multiple of group. The bytes are the two ops'." },
};

/* n a multiple of 128 so a rotation block never straddles a row; group 128 or 256. */
const RadConstraint kSwigluQuantC[] = {
    RAD_CDIV("n", 128), RAD_CDIV("group", 128), RAD_CGE("group", 128), RAD_CLE("group", 256),
    RAD_CIN("dtype", "bf16"),
};

/* mla_attn's band at an int8 latent cache: libr4d's MLA geometry (latent 512, at most 64 heads,
 * blocks of 4) with kv_dtype i8, which no libr4d row reads. */
const RadConstraint kMlaI8C[] = {
    RAD_CGE("M", 1),       RAD_CLE("M", 65535),   RAD_CEQ("latent_dim", 512),
    RAD_CGE("heads", 1),   RAD_CLE("heads", 64),  RAD_CGE("blocks", 1),
    RAD_CEQ("block_size", 4), RAD_CIN("dtype", "bf16"), RAD_CIN("kv_dtype", "i8"),
};

RadOpdDesc desc(uint32_t dt, int64_t rows, int64_t cols) {
    RadOpdDesc d;
    std::memset(&d, 0, sizeof d);
    d.dtype = dt;
    d.rank = 2;
    d.shape[0] = rows;
    d.shape[1] = cols;
    d.fill = RAD_FILL_NORMAL;
    d.idx_const = -1;
    return d;
}

bool mat(const RadTensor* t, uint32_t dt, int64_t rows, int64_t cols) {
    return t && t->dtype == dt && t->rank == 2 && t->shape[0] == rows && t->shape[1] == cols &&
           t->stride[1] == 1 && t->stride[0] >= cols;
}

}  /* namespace */

int rad_glm_schema_count() { return RAD_GLM_N_OPS; }
const RadOpSchema* rad_glm_schema_at(int i) {
    return i >= 0 && i < RAD_GLM_N_OPS ? &kSchemas[i] : nullptr;
}

RadKernelInfo rad_glm_row(int i, RadLaunchFn launch) {
    RadKernelInfo r{};
    if (i != 0) return r;
    r.name = "glm_swiglu_clamped_had_quant_i8g";
    r.op = "swiglu_clamped_had_quant_i8g";
    r.family = "quant";
    r.computes = "GLM's clamped SwiGLU, the 128-wide Hadamard and the per-(row, group) int8 "
                 "quantiser in one pass: a wave a 128-column block, four columns a lane";
    r.shape = "n a multiple of 128, group 128 or 256";
    r.dtypes = "bf16 in, int8 codes and f32 scales out, optional bf16 out";
    r.domain = RAD_DOMAIN_DEVICE;
    r.priority = 100;
    r.constraints = kSwigluQuantC;
    r.n_constraints = (int)(sizeof kSwigluQuantC / sizeof kSwigluQuantC[0]);
    r.launch = launch;
    r.opd_shape = rad_glm_swiglu_quant_shape;
    r.replaces = rad_glm_swiglu_quant_replaces;
    return r;
}

RadKernelInfo rad_glm_mla_i8_row(RadLaunchFn launch, RadScratchFn scratch) {
    RadKernelInfo r{};
    r.name = "glm_mla_attn_i8kv";
    r.op = "mla_attn";
    r.family = "attn";
    r.computes = "absorbed MLA over an int8 latent cache: a wave a (row, head), each key decoded "
                 "to f32 by its half's f16 scale and used for the score and the value, split "
                 "history merged from f32 partials";
    r.shape = "latent_dim=512, heads<=64, block_size=4, cache rows of 516 bytes";
    r.dtypes = "bf16 Q/output, i8 cache (codes and two f16 half scales a row), f32 softmax";
    r.domain = RAD_DOMAIN_DEVICE;
    r.priority = 100;
    r.constraints = kMlaI8C;
    r.n_constraints = (int)(sizeof kMlaI8C / sizeof kMlaI8C[0]);
    r.launch = launch;
    r.scratch = scratch;
    return r;
}

int rad_glm_swiglu_quant_args(const RadArgs* a, RadGlmSwigluQuantCall* out) {
    if (!a || !out || a->n_t < 3) return RAD_E_INVAL;
    const RadTensor* gu = rad_arg_in(a, 0);
    const RadTensor* q  = rad_arg_in(a, 1);
    const RadTensor* s  = rad_arg_in(a, 2);
    const RadTensor* y  = rad_arg_in(a, 3);
    if (!gu || !q || !s) return RAD_E_INVAL;
    const int64_t n = rad_args_geti_or(a, "n", 0), group = rad_args_geti_or(a, "group", 128);
    const double limit = rad_args_getf_or(a, "limit", 0.0);
    if (gu->dtype != RAD_BF16 || q->dtype != RAD_I8 || s->dtype != RAD_F32 ||
        (y && y->dtype != RAD_BF16))
        return RAD_E_DTYPE;
    if ((group != 128 && group != 256) || n <= 0 || n % group || !std::isfinite(limit) ||
        limit <= 0)
        return RAD_E_SHAPE;
    const int64_t M = gu->rank == 2 ? gu->shape[0] : 0;
    if (M < 1 || M > INT32_MAX || !mat(gu, RAD_BF16, M, 2 * n) || !mat(q, RAD_I8, M, n) ||
        !mat(s, RAD_F32, M, n / group) || (y && !mat(y, RAD_BF16, M, n)))
        return RAD_E_SHAPE;
    RadGlmSwigluQuantCall c{};
    c.gu = (const uint16_t*)gu->data;
    c.q = (int8_t*)q->data;
    c.s = (float*)s->data;
    c.y = y ? (uint16_t*)y->data : nullptr;
    c.M = M;
    c.n = n;
    c.group = group;
    c.gu_ld = gu->stride[0];
    c.q_ld = q->stride[0];
    c.s_ld = s->stride[0];
    c.y_ld = y ? y->stride[0] : 0;
    c.limit = (float)limit;
    *out = c;
    return RAD_OK;
}

int rad_glm_swiglu_quant_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    if (!out) return RAD_E_INVAL;
    const int64_t M = rad_param_getdim(p, n_p, "M", 0), n = rad_param_getdim(p, n_p, "n", 0);
    const int64_t g = rad_param_getdim(p, n_p, "group", 128);
    if (M <= 0 || n <= 0 || (g != 128 && g != 256) || n % g) return RAD_E_UNSUPPORTED;
    switch (operand) {
    case 0: *out = desc(RAD_BF16, M, 2 * n); return RAD_OK;
    case 1: *out = desc(RAD_I8, M, n); return RAD_OK;
    case 2: *out = desc(RAD_F32, M, n / g); return RAD_OK;
    case 3: *out = desc(RAD_BF16, M, n); return RAD_OK;
    default: return RAD_E_NOTFOUND;
    }
}

/* swiglu_clamped into the bf16 product (the caller's out_bf16 when passed, else a temporary), then
 * had_quant_act_i8g over it into the codes and scales. */
int rad_glm_swiglu_quant_replaces(const RadParam* p, int n_p, uint64_t present, int step,
                                  RadFuseStep* out) {
    if (!out) return RAD_E_INVAL;
    const RadParam* M = rad_param_find(p, n_p, "M");
    const RadParam* n = rad_param_find(p, n_p, "n");
    const RadParam* lim = rad_param_find(p, n_p, "limit");
    const RadParam* grp = rad_param_find(p, n_p, "group");
    const RadParam* dt = rad_param_find(p, n_p, "dtype");
    if (!M || !n || !lim || !grp || !dt) return RAD_E_UNSUPPORTED;
    const int16_t mid = RAD_FUSE_HAS(present, 3) ? (int16_t)3 : RAD_FUSE_TMP(0);
    std::memset(out, 0, sizeof *out);
    switch (step) {
    case 0:
        out->op = "swiglu_clamped";
        out->p[0] = *M; out->p[1] = *n; out->p[2] = *lim; out->p[3] = *dt;
        out->n_p = 4;
        out->from[0] = 0;
        out->from[1] = mid;
        out->n_from = 2;
        return RAD_OK;
    case 1:
        out->op = "had_quant_act_i8g";
        out->p[0] = *M; out->p[1] = *n; out->p[2] = *grp; out->p[3] = *dt;
        out->n_p = 4;
        out->from[0] = mid;
        out->from[1] = 1;
        out->from[2] = 2;
        out->n_from = 3;
        return RAD_OK;
    default:
        return RAD_E_NOTFOUND;
    }
}
