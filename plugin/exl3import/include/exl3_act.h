/* exl3_act.h -- the per-weight activation mode of the dense EXL3 linear: how a weight's GEMM input is fed to the
 * kernel. Header-only C; owner Q.
 *
 * It is a property of the deployment, not of the stored bytes, so it lives in the container's metadata (one key a
 * weight), never in the weight's encoding: a mix changes it without touching a weight, and one stored weight serves
 * every mode its kernels offer.
 *
 *   key    "exl3.act.<weight name>"           e.g. exl3.act.blk.12.ffn_gate_up.weight
 *   value  "f16"     the activation as is (f16 / bf16), the f16 WMMA rows       (also: the key absent)
 *          "a8t"     int8, one scale a token row
 *          "a8g<N>"  int8, one scale a token row and N inputs, N a multiple of 128 (a8g128: the measured one)
 *
 * (The int4 modes are not served: any other spelling is refused like any unknown value.)
 *
 * The architecture plugin reads the key per weight (exl3_act_of) and hands the mode to the op as its parameter
 * "act" (include/exl3_abi.h), every declaration, f16 included: a kernel row constrains "act" to the modes it
 * computes, so a weight is never served in a mode it was not chosen for -- with no row for its mode the load fails
 * by name. A value this header does not parse is refused (RAD_E_FORMAT), never read as f16. */
#ifndef EXL3_ACT_H
#define EXL3_ACT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EXL3_ACT_KEY_PREFIX "exl3.act."
enum { EXL3_ACT_STR = 16 };   /* the longest canonical value, "a8g" + digits, with room */

enum { EXL3_ACT_F16 = 0, EXL3_ACT_A8T = 1, EXL3_ACT_A8G = 2 };

typedef struct Exl3Act {
    int kind;    /* EXL3_ACT_* */
    int group;   /* a8g: inputs a scale (a multiple of 128); 0 otherwise */
} Exl3Act;

/* Parse a value. 0 on success; -1 for anything that is not exactly a canonical spelling ("a8g0128" and "A8T" are
 * refused, so a key has one spelling and a string compare between containers means what it says). */
static inline int exl3_act_parse(const char* s, Exl3Act* out) {
    Exl3Act a = { EXL3_ACT_F16, 0 };
    if (!s) return -1;
    if (!strcmp(s, "f16")) { if (out) *out = a; return 0; }
    if (!strcmp(s, "a8t")) { a.kind = EXL3_ACT_A8T; if (out) *out = a; return 0; }
    if (!strncmp(s, "a8g", 3) && s[3] >= '1' && s[3] <= '9') {
        char* end = 0;
        const long g = strtol(s + 3, &end, 10), unit = 128;
        if (*end || g < unit || g % unit || g > (1L << 20)) return -1;
        a.kind = EXL3_ACT_A8G;
        a.group = (int)g;
        if (out) *out = a;
        return 0;
    }
    return -1;
}

/* The canonical spelling of a mode (buf of EXL3_ACT_STR). */
static inline const char* exl3_act_str(const Exl3Act* a, char* buf) {
    if (a->kind == EXL3_ACT_A8T) snprintf(buf, EXL3_ACT_STR, "a8t");
    else if (a->kind == EXL3_ACT_A8G) snprintf(buf, EXL3_ACT_STR, "a8g%d", a->group);
    else snprintf(buf, EXL3_ACT_STR, "f16");
    return buf;
}

/* The metadata key of a weight (buf of n bytes). 0, or -1 when it does not fit. */
static inline int exl3_act_key(const char* weight, char* buf, size_t n) {
    const int k = snprintf(buf, n, EXL3_ACT_KEY_PREFIX "%s", weight ? weight : "");
    return k > 0 && (size_t)k < n ? 0 : -1;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* EXL3_ACT_H */
