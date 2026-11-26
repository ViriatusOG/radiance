/* exl3i_abi.h -- the interface of EXL3-V3's dense linear, `exl3i_dense`: its schema, its operand positions, the
 * weight planes it reads (include/exl3i_encoding.h), and one parse of its geometry. Header-only C.
 *
 * THE OP. The EXL3 dense GEMM over an exl3i weight: per row m, a[m] scaled by suh (suh.1 for
 * the second half when the optional suh.1 operand is present) and rotated, times W_hat (the trellis decoded at rate
 * m / 16 in the codebook `codebook`), rotated back and scaled by svh; bf16 / f16 / f32 out.
 *
 * Parameters: M (rows), N, K, rate (exl3i_rate_str's canonical text: "2.25", "0.5", "4"), codebook ("mad"; anything
 * else is refused, RAD_E_DTYPE), act (include/exl3_act.h; absent = "f16"), aq_plane / aq_planes (version 3, optional: where the
 * weight's planes sit in the aq operand's rows). Operands: a, record, suh, svh, suh1 (optional), y, aq (optional,
 * version 3). Version 3: the producer-quantised activation.
 * Version 4: codebook mad (exl3i_decode.h: IQR's computed code on the int8 grid), and exl3i_actq's `form` xb
 * for it: the decode band's rows (1 .. EXL3I_AQ_XB_MAX_ROWS) in the byte-sum GEMV's form, read by a mad a8g128 dense op
 * at that many rows (below). Version 5: exl3i_actq's residual add (res / xo, below).
 *
 * AQ form xb (version 4). exl3i_actq with form = "xb" writes, per row and consumer plane i (exl3i_aq_xb_plane_bytes(K)
 * each, packed per row as [planes][that]): xb int32 [K] (the int8 value q of the a8g128 plane above, times
 * 0x01010101), the f32 scales [K / 128], and int32 [K / 128][2] = 126 x the sum of q over the block's columns with k % 16
 * in {4h .. 4h + 3, 8 + 4h .. 8 + 4h + 3} for h = 0, 1. A mad dense op at act a8g128 launched at 1 ..
 * EXL3I_AQ_XB_MAX_ROWS rows reads its planes there (aq_plane / aq_planes as above); the A8 threshold's rule is the
 * int8 form's. The values are the int8 form's, so the output is the same as without aq.
 *
 * AQ (version 3). `exl3i_actq` quantises one activation x [M, K] for up to four consumers, each with its own input scale
 * vector s_i: per row the consumers' int8 planes as the A8 path's own quantiser makes them (x scaled by s_i, rotated,
 * one f32 scale a (row, 128 inputs), a8g128), packed per row as [planes][K] int8 then [planes][K / 128] f32
 * (exl3i_aq_row_bytes). A dense op given aq (and aq_plane = p, aq_planes = n of the actq that wrote it) reads its planes
 * p (and p + 1 for a stacked gate_up's suh.1) instead of quantising `a` itself. The bytes are the A8 path's own, so the
 * output is the same. aq is read at any launch it is given at (a8g128 only), and a launch below the A8 path's threshold
 * (EXL3I_AQ_MIN_ROWS, the dense op's own EXL3I_A8_MIN_ROWS) with it is refused: a producer that wrote planes
 * instead of `a` (`pre` below) has left `a` stale, so the caller issues AQ only from that many rows.
 *
 * exl3i_actq parameters: M (rows), K, planes (1..4), pre (optional, "none"), eps / wadd (f64, pre rmsnorm); operands: x
 * (f16 | bf16 [M, K]; [M, 2K] for silu_mul), s0 (weight, f16 [K]), s1 .. s3 (weights, optional: planes > 1), out (u8
 * [M, exl3i_aq_row_bytes(K, planes)]), x2 (optional, bf16 [M, K]: mul's second factor), g (optional weight, f32 | bf16
 * [K]: rmsnorm's gain), h (optional out, bf16 [M, K]: the pre stage's value, written when given).
 * `pre` folds the producer into the quantiser, the value quantised being radiance's unfused op's bf16 output byte for
 * byte (libr4d r4d_model_bf16.hip, its arithmetic copied): "rmsnorm" bf16(x rsqrt(mean(x^2) + eps)
 * (g + wadd)), the sum of squares in rmsnorm_kernel's kR4dSumLanes partition and wave order; "silu_mul" bf16(silu(x[:,
 * :K]) x[:, K:]); "mul" bf16(x x2). x is bf16 for every pre but "none".
 *
 * Residual (version 5, pre "rmsnorm" only): param res = 1 and operands res (bf16 [M, K]) and xo (out, bf16 [M, K]): the
 * row first becomes bf16(x + res), which is written to xo and is what the norm takes -- libr4d's add_rmsnorm (x = x + a,
 * the norm over the bf16 sum it stores), byte for byte. xo may be x itself and h may be res itself (the residual stream
 * updated in place, the norm written over the addend): every read of a row's x and res comes before any write of xo or
 * h to it.
 *
 * Both sides check EXL3I_IFACE_VERSION; bump it on any change below. */
#ifndef EXL3I_ABI_H
#define EXL3I_ABI_H

#include "rad_abi.h"
#include "rad_plugin.h"
#include "exl3i_encoding.h"
#include "exl3_act.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef EXL3I_IFACE_VERSION
#define EXL3I_IFACE_VERSION 5
#endif

#define EXL3I_OP_NT "exl3i_dense"
#define EXL3I_OP_ACTQ "exl3i_actq"

/* operand positions; a launch from before version 3 carries the first EXL3I_N_OPD_MIN */
enum { EXL3I_OPD_A = 0, EXL3I_OPD_RECORD = 1, EXL3I_OPD_SUH = 2, EXL3I_OPD_SVH = 3, EXL3I_OPD_SUH1 = 4,
       EXL3I_OPD_Y = 5, EXL3I_OPD_AQ = 6, EXL3I_N_OPD = 7, EXL3I_N_OPD_MIN = 6 };
/* exl3i_actq's operands */
enum { EXL3I_AQ_X = 0, EXL3I_AQ_S0 = 1, EXL3I_AQ_OUT = 5, EXL3I_AQ_X2 = 6, EXL3I_AQ_G = 7, EXL3I_AQ_H = 8,
       EXL3I_AQ_RES = 9, EXL3I_AQ_XO = 10,   /* version 5: the residual add (param res = 1) */
       EXL3I_AQ_N_OPD = 11, EXL3I_AQ_N_OPD_MIN = 6, EXL3I_AQ_MAX_PLANES = 4 };
/* exl3i_actq's producer stage (`pre`) */
enum { EXL3I_AQ_PRE_NONE = 0, EXL3I_AQ_PRE_RMSNORM = 1, EXL3I_AQ_PRE_SILU_MUL = 2, EXL3I_AQ_PRE_MUL = 3 };
/* the fewest real rows a dense launch reads aq at: the A8 path's threshold (exl3i_a8.h EXL3I_A8_MIN_ROWS) */
#define EXL3I_AQ_MIN_ROWS 512

/* AQ's row: [planes][K] int8, then [planes][K / 128] f32 */
static inline int64_t exl3i_aq_row_bytes(int64_t K, int planes) { return (int64_t)planes * (K + 4 * (K / 128)); }
/* form xb (version 4): a plane is xb int32 [K], scales f32 [K / 128], bias int32 [K / 128][2], 256-B aligned; the row
 * [planes][that]. At most EXL3I_AQ_XB_MAX_ROWS rows a launch (the decode band). */
#define EXL3I_AQ_XB_MAX_ROWS 8
/* The rows at which a producer feeding mad consumers writes form xb instead of its bf16 value (qwen35_exl3): the dense op
 * takes the GEMV with the producer's rows at any T <= EXL3I_AQ_XB_MAX_ROWS, but by itself only at 1 (2 below K4) row --
 * past that its nt_pass needs the bf16 rows. */
#define EXL3I_AQ_XB_ISSUE_ROWS 1
enum { EXL3I_AQ_FORM_I8 = 0, EXL3I_AQ_FORM_XB = 1 };
#define EXL3I_AQ_XB_PLANE_BYTES(K) ((4 * (int64_t)(K) + 12 * ((int64_t)(K) / 128) + 255) / 256 * 256)   /* device code too */
static inline int64_t exl3i_aq_xb_plane_bytes(int64_t K) { return EXL3I_AQ_XB_PLANE_BYTES(K); }
static inline int64_t exl3i_aq_xb_row_bytes(int64_t K, int planes) { return (int64_t)planes * exl3i_aq_xb_plane_bytes(K); }
static inline int exl3i_aq_form_parse(const char* s) {
    if (!s || !strcmp(s, "int8")) return EXL3I_AQ_FORM_I8;
    if (!strcmp(s, "xb")) return EXL3I_AQ_FORM_XB;
    return -1;
}

static inline int exl3i_schema_count(void) { return 2; }

static inline const RadOpSchema* exl3i_schema_at(int i) {
    static const RadParamSpec p_nt[] = {
        { "M",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "N",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "K",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "rate",     RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE },
        { "codebook", RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE },
        { "act",      RAD_P_STR, RAD_OPTIONAL, RAD_PROLE_NONE },
        { "aq_plane", RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE },
        { "aq_planes", RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE },
    };
    static const RadOperandSpec nt_opd[EXL3I_N_OPD] = {
        { "a",      RAD_OPD_IN,     0 },
        { "record", RAD_OPD_WEIGHT, 0 },
        { "suh",    RAD_OPD_WEIGHT, 0 },
        { "svh",    RAD_OPD_WEIGHT, 0 },
        { "suh1",   RAD_OPD_WEIGHT, 1 },
        { "y",      RAD_OPD_OUT,    0 },
        { "aq",     RAD_OPD_IN,     1 },
    };
    static const RadParamSpec p_aq[] = {
        { "M",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "K",        RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "planes",   RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
        { "pre",      RAD_P_STR, RAD_OPTIONAL, RAD_PROLE_NONE },
        { "eps",      RAD_P_F64, RAD_OPTIONAL, RAD_PROLE_NONE },
        { "wadd",     RAD_P_F64, RAD_OPTIONAL, RAD_PROLE_NONE },
        { "form",     RAD_P_STR, RAD_OPTIONAL, RAD_PROLE_NONE },
        { "res",      RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE },
    };
    static const RadOperandSpec aq_opd[EXL3I_AQ_N_OPD] = {
        { "x",   RAD_OPD_IN,     0 },
        { "s0",  RAD_OPD_WEIGHT, 0 },
        { "s1",  RAD_OPD_WEIGHT, 1 },
        { "s2",  RAD_OPD_WEIGHT, 1 },
        { "s3",  RAD_OPD_WEIGHT, 1 },
        { "out", RAD_OPD_OUT,    0 },
        { "x2",  RAD_OPD_IN,     1 },
        { "g",   RAD_OPD_WEIGHT, 1 },
        { "h",   RAD_OPD_OUT,    1 },
        { "res", RAD_OPD_IN,     1 },
        { "xo",  RAD_OPD_OUT,    1 },
    };
    static const RadOpSchema s[2] = {
        { EXL3I_OP_NT, p_nt, (int)(sizeof p_nt / sizeof p_nt[0]), nt_opd, EXL3I_N_OPD,
          "EXL3-V3 dense linears (exl3i weights: any rate in 1/8 steps, mad): per row, a[m] scaled by suh "
          "(suh.1 for the second half when present) and rotated, times W_hat, rotated back and scaled by svh; "
          "with aq (version 3) the A8 path reads the producer-quantised planes. "
          "include/exl3i_abi.h." },
        { EXL3I_OP_ACTQ, p_aq, (int)(sizeof p_aq / sizeof p_aq[0]), aq_opd, EXL3I_AQ_N_OPD,
          "EXL3-V3's activation quantiser for the A8 path, one x for up to four consumers: per row and consumer i, x scaled "
          "by s_i, rotated (128 points), one f32 scale a 128 inputs, int8 -- the dense op's own a8g128 planes, packed per "
          "row as [planes][K] int8 then [planes][K / 128] f32 (form xb, version 4: the mad GEMV's broadcast form); `pre` "
          "folds the producer (rmsnorm, silu_mul, mul) in; with res (version 5) rmsnorm's residual add x = x + res too. "
          "include/exl3i_abi.h." },
    };
    return i >= 0 && i < 2 ? &s[i] : 0;
}

/* exl3i_actq's producer stage: EXL3I_AQ_PRE_*, -1 for a name not known */
static inline int exl3i_aq_pre_parse(const char* s) {
    if (!s || !strcmp(s, "none")) return EXL3I_AQ_PRE_NONE;
    if (!strcmp(s, "rmsnorm")) return EXL3I_AQ_PRE_RMSNORM;
    if (!strcmp(s, "silu_mul")) return EXL3I_AQ_PRE_SILU_MUL;
    if (!strcmp(s, "mul")) return EXL3I_AQ_PRE_MUL;
    return -1;
}

/* exl3i_actq's residual add (version 5): 1 with res / xo, 0 without, -1 for a value not 0 or 1 */
static inline int exl3i_aq_res(const RadParam* p, int n_p) {
    const int64_t r = rad_param_geti(p, n_p, "res", 0);
    return r == 0 || r == 1 ? (int)r : -1;
}

/* exl3i_actq's geometry: M, K (a positive multiple of 128), planes 1 .. EXL3I_AQ_MAX_PLANES, pre (may be 0); res only
 * with pre rmsnorm. */
static inline int exl3i_aq_parse(const RadParam* p, int n_p, int64_t* M, int64_t* K, int* planes, int* pre) {
    if (!rad_param_find(p, n_p, "M") || !rad_param_find(p, n_p, "K") || !rad_param_find(p, n_p, "planes"))
        return RAD_E_SCHEMA;
    *M = rad_param_getdim(p, n_p, "M", 0);
    *K = rad_param_geti(p, n_p, "K", 0);
    *planes = (int)rad_param_geti(p, n_p, "planes", 0);
    if (*M < 1 || *K < 128 || *K % 128 || *planes < 1 || *planes > EXL3I_AQ_MAX_PLANES) return RAD_E_SHAPE;
    const int pr = exl3i_aq_pre_parse(rad_param_gets(p, n_p, "pre", 0));
    if (pr < 0) return RAD_E_SCHEMA;
    if (exl3i_aq_form_parse(rad_param_gets(p, n_p, "form", 0)) < 0) return RAD_E_SCHEMA;
    if (exl3i_aq_res(p, n_p) < 0 || (exl3i_aq_res(p, n_p) && pr != EXL3I_AQ_PRE_RMSNORM)) return RAD_E_SCHEMA;
    if (pre) *pre = pr;
    return RAD_OK;
}

typedef struct Exl3iGeom {
    int64_t M, N, K;
    int     m, cb;     /* rate m / 16, EXL3I_CB_* */
    Exl3Act act;
} Exl3iGeom;

/* RAD_OK, or: RAD_E_SCHEMA a required parameter absent; RAD_E_DTYPE a rate, codebook or act this interface has no
 * name for; RAD_E_SHAPE K or N not a positive multiple of 128, or M < 1. Every implementation parses through this. */
static inline int exl3i_geom_parse(const RadParam* p, int n_p, Exl3iGeom* g) {
    static const char* const req[] = { "M", "N", "K", "rate", "codebook" };
    const char* rt = rad_param_gets(p, n_p, "rate", 0);
    const char* cb = rad_param_gets(p, n_p, "codebook", 0);
    const char* ac = rad_param_gets(p, n_p, "act", 0);
    int c;
    memset(g, 0, sizeof *g);
    for (c = 0; c < (int)(sizeof req / sizeof req[0]); ++c)
        if (!rad_param_find(p, n_p, req[c])) return RAD_E_SCHEMA;
    if (!rt || !cb) return RAD_E_SCHEMA;
    g->m = exl3i_rate_parse(rt);
    g->cb = exl3i_codebook_from_name(cb);
    if (g->m < 0 || g->cb < 0) return RAD_E_DTYPE;
    if (exl3_act_parse(ac ? ac : "f16", &g->act) != 0) return RAD_E_DTYPE;
    g->N = rad_param_geti(p, n_p, "N", 0);
    g->K = rad_param_geti(p, n_p, "K", 0);
    if (g->K < 1 || g->K % 128 || g->N < 1 || g->N % 128) return RAD_E_SHAPE;
    g->M = rad_param_getdim(p, n_p, "M", 0);
    if (g->M < 1) return RAD_E_SHAPE;
    return RAD_OK;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* EXL3I_ABI_H */
