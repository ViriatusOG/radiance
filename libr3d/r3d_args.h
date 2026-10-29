/* r3d_args.h -- how a libr3d entry point reads its arguments.
 *
 * There is no adapter layer. An entry point takes `(const RadArgs*, RadStream)` and returns a
 * status, the same RadLaunchFn contract every other plugin implements; an out-of-tree author can
 * write the same kernel without an adapter they could not reach.
 *
 * This header holds the shared helpers: reading a parameter, finding an operand, testing
 * contiguity. Most are one line over rad_abi.h.
 *
 * Nothing here includes r3d.h or HIP: host translation units (the row table, the shape hooks)
 * include it too, so those tables compile and can be checked without ROCm.
 */
#ifndef R3D_ARGS_H
#define R3D_ARGS_H

#include "rad_abi.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ================================================================== parameter access
 *
 * Forwards to rad_abi.h's static inline helpers, the ones a third-party plugin gets, keeping
 * libr3d's spelling at its call sites. */
static inline int r3d_geti(const RadArgs* a, const char* k, long long* out) {
    return rad_args_geti(a, k, out);
}

static inline long long r3d_geti_or(const RadArgs* a, const char* k, long long dflt) {
    return rad_args_geti_or(a, k, dflt);
}

static inline const char* r3d_gets(const RadArgs* a, const char* k) {
    return rad_args_gets(a, k);
}

/* A float parameter is RAD_P_F64 (rad_types.h, docs/OPS.md "The four rules"), so `dval` arrives
 * and the step path runs no strtod. Deliberately laxer than rad_args_getf_or: string and integer
 * spellings are still read, because a tool or selftest may call a launch directly with no declare
 * to enforce the type. */
static inline double r3d_getf(const RadArgs* a, const char* k, double dflt) {
    if (!a || !k) return dflt;
    for (int i = 0; i < a->n_p; ++i) {
        if (!a->p[i].key || std::strcmp(a->p[i].key, k) != 0) continue;
        if (a->p[i].kind == RAD_P_F64) return a->p[i].dval;
        if (a->p[i].kind == RAD_P_INT) return (double)a->p[i].ival;
        if (a->p[i].kind == RAD_P_STR && a->p[i].sval && a->p[i].sval[0]) {
            char* end = nullptr;
            const double v = std::strtod(a->p[i].sval, &end);
            return (end && end != a->p[i].sval) ? v : dflt;
        }
        return dflt;
    }
    return dflt;
}

/* An enumerated parameter: `act` is "silu" or "sigmoid", not 0 or 1, so a --debug-graph dump says
 * what runs; the mapping to libr3d's integer lives in the shim. Returns -1 for an unknown name,
 * which every caller turns into RAD_E_INVAL. `alt` is libr3d's own integer spelling of the same
 * key, consulted only when the name is absent. */
static inline int r3d_act_code(const RadArgs* a, const char* key, const char* alt, int dflt) {
    if (const char* s = r3d_gets(a, key)) {
        if (!std::strcmp(s, "silu") || !std::strcmp(s, "swish")) return 0;
        if (!std::strcmp(s, "sigmoid")) return 1;
        return -1;
    }
    long long v = 0;
    if (alt && r3d_geti(a, alt, &v)) return (v == 0 || v == 1) ? (int)v : -1;
    return dflt;
}

/* Same, off a bare RadParam list (init() and the layout hooks); forwards to rad_plugin.h's
 * rad_param_*. */
static inline long long r3d_pgeti_or(const RadParam* p, int n_p, const char* k, long long dflt) {
    return rad_param_geti(p, n_p, k, dflt);
}

static inline const char* r3d_pgets(const RadParam* p, int n_p, const char* k) {
    return rad_param_gets(p, n_p, k, nullptr);
}

/* ================================================================== operand access
 *
 * An absent optional operand is a null RadTensor.data (rad_abi.h). r3d_opt() returns null both
 * past the end and for a present-but-null operand, so an optional test is one pointer test. For a
 * required operand the caller turns null into RAD_E_INVAL: it is a bug in the issue, not a shape
 * the kernel declines.
 */
static inline const RadTensor* r3d_opt(const RadArgs* a, int i) { return rad_arg_in(a, i); }

/* An optional operand the plan carries a slot for, with no bytes behind it, is ABSENT -- it is not a
 * dtype problem. The engine hands the slots of an op's optional operands as descriptors with a null
 * `data` when the graph did not bind one, and a shim that dtype-checks such a descriptor would refuse
 * a band the op never reads -- `rmsnorm_quant_fp8` answering E_DTYPE to a band whose dtype is fine,
 * for an empty `out_bf16` slot. Every test of an optional operand goes through here; a bare `if (t)`
 * is the bug. */
static inline bool r3d_has(const RadTensor* t) { return t && t->data; }

static inline const void* r3d_p(const RadTensor* t) { return t ? t->data : nullptr; }

/* The same for an operand a kernel writes, so an output cast reads as one rather than as a pointer
 * laundered through `long`. */
static inline void* r3d_wp(const RadTensor* t) { return t ? t->data : nullptr; }
static inline long r3d_addr(const RadTensor* t) { return t ? (long)(uintptr_t)t->data : 0L; }

static inline long long r3d_dim(const RadTensor* t, int i) {
    if (!t || i < 0 || (uint32_t)i >= t->rank) return 0;
    return t->shape[i];
}

static inline long long r3d_stride(const RadTensor* t, int i) {
    if (!t || i < 0 || (uint32_t)i >= t->rank) return 0;
    return t->stride[i];
}

static inline long long r3d_numel(const RadTensor* t) { return rad_tensor_numel(t); }

/* [rows..., K] flattened to (M, K). Every per-row kernel in libr3d owns whole rows (one workgroup
 * per row), so the leading axes are only a row count. The GEMM makes the same fold explicitly. */
static inline int r3d_rows_of(const RadTensor* t, long long* M, long long* K) {
    if (!t || t->rank < 1) return RAD_E_SHAPE;
    *K = t->shape[t->rank - 1];
    *M = 1;
    for (uint32_t i = 0; i + 1 < t->rank; ++i) *M *= t->shape[i];
    return (*M >= 1 && *K >= 1) ? RAD_OK : RAD_E_SHAPE;
}

/* Dense row-major over the whole tensor. libr3d's entry points take at most one pitch, so
 * everything else must be tight, and the shim checks it. This is the "necessary condition only"
 * boundary of spec §2.1: the constraint table settles geometry, contiguity is a property of the
 * call. */
static inline int r3d_contig(const RadTensor* t) { return rad_tensor_is_contiguous(t); }

/* Contiguous in every axis but the leading one, whose stride is the row pitch libr3d takes as an
 * explicit argument (`x_ld`, `a_stride`, ...). Returns the pitch in elements, or -1. */
static inline long long r3d_row_pitch(const RadTensor* t) {
    if (!t || t->rank == 0) return -1;
    if (t->rank == 1) return 1;
    long long acc = 1;
    for (int i = (int)t->rank - 1; i >= 1; --i) {
        if (t->shape[i] != 1 && t->stride[i] != acc) return -1;
        acc *= t->shape[i];
    }
    return t->stride[0];
}

/* [rows..., n] read as (rows, n): the leading axes are a row count and only the last is
 * structural. Returns the pitch in elements, or -1.
 *
 * A column slice must refuse. k out of a fused [tok, 2*kv_dim] projection, read as
 * [tok*n_head_kv, head_dim] for a per-head norm, has rows `n` apart within a token and a buffer row
 * apart between tokens; no single pitch describes that, and the leading stride would make the
 * consumer read most heads from garbage. So the trailing axes must cover the whole row. */
static inline long long r3d_rows_pitch(const RadTensor* t, long long n, long long* rows) {
    if (!t || t->rank < 1 || n <= 0) return -1;
    const long long nel = rad_tensor_numel(t);
    if (nel % n) return -1;
    *rows = nel / n;
    if (t->rank == 1) return n;
    if (t->shape[t->rank - 1] == n) {
        if (t->stride[t->rank - 1] != 1) return -1;
        return t->rank >= 2 ? t->stride[t->rank - 2] : n;
    }
    if (rad_tensor_is_contiguous(t)) return n;
    long long trail = 1;
    for (int i = 1; i < (int)t->rank; ++i) trail *= t->shape[i];
    return trail == n ? r3d_row_pitch(t) : -1;
}

/* An attention operand, [tokens, heads * head_dim] over its trailing axes: the token count, head
 * count and row pitch in elements. False when the trailing axes are not whole heads or no single
 * pitch describes the rows. */
static inline bool r3d_head_rows(const RadTensor* t, long long head_dim, long long* n_tok,
                                 long long* n_head, long long* ld) {
    long long width = 1;
    for (uint32_t i = 1; i < t->rank; ++i) width *= t->shape[i];
    if (t->rank < 2 || width % head_dim) return false;
    *n_head = width / head_dim;
    *ld = r3d_rows_pitch(t, width, n_tok);
    return *ld >= 0;
}

/* 16-byte alignment, needed by every kernel that loads a dword pair per lane. The kernels do not
 * check it, so the shim does. */
static inline int r3d_aligned(const void* p, unsigned bytes) {
    return p ? (((uintptr_t)p & (uintptr_t)(bytes - 1)) == 0) : 1;
}

/* ================================================================== which check refused
 *
 * A shim refuses in many places with the same few codes, so the engine's report names the kernel
 * and geometry but not the failing condition. R3D_NO() wraps a refusal and prints its own line when
 * RAD_R3D_TRACE is set; otherwise it is the bare return plus one cached-flag load on a path that is
 * failing anyway. */
static inline int r3d_trace_on(void) {
    static int on = -1;
    if (on < 0) { const char* e = std::getenv("RAD_R3D_TRACE"); on = (e && *e && *e != '0'); }
    return on;
}
static inline int r3d_refused(int code, const char* file, int line) {
    if (r3d_trace_on()) std::fprintf(stderr, "[r3d] %s:%d refused -> %d\n", file, line, code);
    return code;
}
#define R3D_NO(code) r3d_refused((code), __FILE__, __LINE__)

/* The descriptor we were handed, printed when we decline one. A stride refusal that names no stride
 * is a refusal the reader has to reproduce, and reproducing it costs a probe cycle: the engine binds
 * rank, shape and strides in ELEMENTS, and the shape it binds is the step's, not the band's. */
static inline void r3d_trace_desc(const char* what, const RadTensor* t) {
    if (!r3d_trace_on()) return;
    if (!t) { std::fprintf(stderr, "[r3d] %s: (absent)\n", what); return; }
    std::fprintf(stderr, "[r3d] %s: dtype %u rank %u shape [", what, t->dtype, t->rank);
    for (int i = 0; i < (int)t->rank; ++i) std::fprintf(stderr, "%s%lld", i ? "," : "", (long long)t->shape[i]);
    std::fprintf(stderr, "] stride [");
    for (int i = 0; i < (int)t->rank; ++i) std::fprintf(stderr, "%s%lld", i ? "," : "", (long long)t->stride[i]);
    std::fprintf(stderr, "] data %p\n", t->data);
}
#define R3D_STR(why, a1, v1, a2, v2)                                                  \
    do {                                                                              \
        if (r3d_trace_on())                                                           \
            std::fprintf(stderr, "[r3d] %s:%d stride refusal: %s, %s %lld, %s %lld\n",\
                         __FILE__, __LINE__, (why), (a1), (long long)(v1), (a2),      \
                         (long long)(v2));                                            \
        return R3D_NO(RAD_E_STRIDE);                                                  \
    } while (0)

/* ================================================================== dtype helpers
 *
 * A row's `dtype` constraint is a necessary condition on the geometry, not a promise about the
 * tensors: the shim reads the dtype off the operand, and these are the tests it reads it with. */
static inline int r3d_is_bf16(const RadTensor* t) { return t && t->dtype == RAD_BF16; }
static inline int r3d_is_f16(const RadTensor* t) { return t && t->dtype == RAD_F16; }
static inline int r3d_is_f32(const RadTensor* t) { return t && t->dtype == RAD_F32; }
static inline int r3d_is_i32(const RadTensor* t) { return t && t->dtype == RAD_I32; }
static inline int r3d_is_i64(const RadTensor* t) { return t && t->dtype == RAD_I64; }
static inline int r3d_is_fp8(const RadTensor* t) {
    return t && (t->dtype == RAD_F8E4M3 || t->dtype == RAD_F8E5M2);
}
static inline int r3d_is_e4m3(const RadTensor* t) { return t && t->dtype == RAD_F8E4M3; }
static inline int r3d_is_i8(const RadTensor* t) { return t && t->dtype == RAD_I8; }
static inline int r3d_is_u8(const RadTensor* t) { return t && t->dtype == RAD_U8; }

/* The dtype name a row's constraint table carries, for a shim that must match a string parameter
 * against an operand. */
static inline int r3d_dt_named(uint32_t dt, const char* name) {
    if (!name) return 0;
    if (!std::strcmp(name, "bf16")) return dt == RAD_BF16;
    if (!std::strcmp(name, "f16") || !std::strcmp(name, "fp16")) return dt == RAD_F16;
    if (!std::strcmp(name, "f32") || !std::strcmp(name, "fp32")) return dt == RAD_F32;
    if (!std::strcmp(name, "fp8_e4m3") || !std::strcmp(name, "e4m3")) return dt == RAD_F8E4M3;
    if (!std::strcmp(name, "fp8_e5m2") || !std::strcmp(name, "e5m2")) return dt == RAD_F8E5M2;
    if (!std::strcmp(name, "i8") || !std::strcmp(name, "int8")) return dt == RAD_I8;
    return 0;
}

/* ================================================================== libr3d's constants
 *
 * Restated here because r3d_rows.cpp and r3d_shapes.cpp are host code built without ROCm and cannot
 * include r3d.h, and because a .rad container records them in its layout tag. */
enum {
    R3D_WAVE = 32,
    /* The activation quantiser's group: the width a scale covers along n. 128 is the WEIGHT
     * scale's block width and therefore a property of the checkpoint, not a knob. */
    R3D_ACT_GROUP = 128,
    /* Block-scaled fp8, on both axes: the checkpoint's `quantization_config.weight_block_size`,
     * which the bytes on disk follow. The entry points reject a mismatch. */
    R3D_FP8_BLOCK = 128,
    /* The int8 weight group, the same 128 the checkpoint's scales use. */
    R3D_W8_GROUP = 128,
    /* The largest N a dot2 GEMM's per-lane accumulator grid spans in one wave. */
    R3D_NTILE = 16,
    /* The operand count of `gemm_nt_q` (a, a_scale, b, b_scale, y, a_sum, b_ref). The shape hook
     * builds the whole list at once and needs a name for its length; the schema in r3d_rows.cpp is
     * the authority, and rad-kbench checks the two against each other. */
    R3D_GEMM_Q_NOPERAND = 7
};

/* ================================================================== libr3d's tunables
 *
 * The knobs the row table declares, exposed so `rad-tune` can walk them and `--debug-graph` can
 * name the pick (spec §15). The chosen point comes back appended to RadArgs::p, so a launch reads
 * it with r3d_geti_or like any other parameter; these structs are how a launch reads several at
 * once, with the row's defaults where a caller supplied none.
 *
 * sk == 0 means "choose the largest legal split count at issue". Split-K requires
 * K % (sk * group) == 0, so no fixed default is legal at every shape, and an illegal default would
 * make an untuned install fail rather than run slowly. The sentinel is resolved in one place, in
 * the GEMM unit.
 */
struct R3dGemmCfg { int wv, sk, kb, nt; };   /* waves a block, split-K, K slab, non-temporal loads */
struct R3dSplitCfg { int splits; };          /* split-KV segments; 0 is the kernel's own law */
struct R3dTopkCfg { int rc, nch; };          /* candidates a chunk keeps, chunk width in columns */

extern "C" {
/* -- the tuned configuration (r3d_rows.cpp) --
 *
 * The core picked a point in the space the row declared and appended it to RadArgs::p. These read
 * it back, falling to the axis's default (defined once, in the RadTunable table) when a caller
 * supplied none, so a checker with no tuned key runs what an untuned install runs. They cannot
 * fail: the loader refuses a plugin whose default is not one of the axis's values. */
void r3d_cfg_gemm (const RadArgs*, struct R3dGemmCfg*);
void r3d_cfg_split(const RadArgs*, struct R3dSplitCfg*);
void r3d_cfg_topk (const RadArgs*, struct R3dTopkCfg*);

}  /* extern "C" */

#endif  /* R3D_ARGS_H */
