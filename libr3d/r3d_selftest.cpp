/* r3d_selftest -- dump the libr3d row table, cross-check it, and prove every row links and runs.
 *
 * A transcription error in a constraint table stays invisible until a model silently picks the
 * wrong kernel, so the table is printed and checked against itself:
 *
 *   - every row's `op` has a schema declared by this plugin;
 *   - every constraint key is a declared parameter of that op (a key renamed in one place only
 *     yields a row that matches nothing);
 *   - names are unique, launches exist, an init has a fini, a tunable's default is one of its
 *     values (the loader refuses a plugin otherwise), and layout hooks appear only on the
 *     joined-weight rows, always with their relayout writer;
 *   - every `opd_shape` describes as many operands as its schema declares, with positive extents.
 *
 * The codecs are checked against the ABI's host versions in `rad_plugin.h`: bf16 and f16 over all
 * 65536 patterns, E4M3 over all 256 codes, the E4M3 encoder over saturating, subnormal,
 * signed-zero and NaN cases, int8 against `rad_round_even_sat` over every tie in its band.
 * rad-kbench compares whole ops, where a codec error appears only as a diffuse percentage.
 * libr3d's codecs are software (its bf16 packing is `v_perm`, not `v_cvt_pk_bf16_f32`), so they
 * are the plugin's own arithmetic.
 *
 * On a device, every launch is called with a minimal valid input built from the row's own
 * `opd_shape`, and every output plane is read back and checked to be finite. That catches what a
 * table check cannot: an untested null operand, a stride from the wrong axis, an operand index off
 * by one, an undefined symbol, a stray write.
 *
 * The numerical oracle is `rad-kbench` (`kernels.rkb` against libref); this program checks that
 * the rows are reachable on this device, from this .so, at the geometry their constraints state.
 * The plugin is dlopened rather than linked, so an undefined symbol in the .so is caught too.
 *
 * One child process per row: a GPU page fault makes ROCr print a line and abort the process, so
 * the parent runs the static phase and then re-execs itself per row with `--case <name>`. A child
 * killed by a signal is a FAIL naming the signal. posix_spawn rather than fork, because the
 * parent's HIP runtime threads are already up and only exec resets them.
 *
 * `--case <name>` also reproduces one row alone in the foreground.
 */

#include "rad_abi.h"
#include "rad_plugin.h"
#include "r3d_codecs.h"
#include "rad_planes.h"   /* the joined scale+zero plane, for the row no shape hook can describe */
#include "r3d_plugin.h"   /* the entry points */

#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdarg>
#include <cerrno>
#include <cmath>
#include <csetjmp>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* posix_spawn's environment; declared here rather than relying on unistd.h's _GNU_SOURCE guard. */
extern char** environ;

#ifdef RAD_HAVE_HIP
#include <hip/hip_runtime.h>
#endif

#define NELEM(a) ((int)(sizeof(a) / sizeof((a)[0])))
static int g_fail = 0;
static void* g_h = nullptr;   /* the plugin, opened by main */
static int g_skip = 0;
static const char* g_case = nullptr;
static const char* g_plugin = nullptr;

/* The plugin's table, through the symbols dlsym'd in main (the program does not link libr3d). */
static int (*P_count)(void) = nullptr;
static const RadKernelInfo* (*P_at)(int) = nullptr;
static int (*P_scount)(void) = nullptr;
static const RadOpSchema* (*P_sat)(int) = nullptr;

static int rows() { return P_count ? P_count() : 0; }
static const RadKernelInfo* row_at(int i) { return P_at ? P_at(i) : nullptr; }
static int schemas() { return P_scount ? P_scount() : 0; }
static const RadOpSchema* schema_at(int i) { return P_sat ? P_sat(i) : nullptr; }

/* printf into a string, as the loader builds its messages. */
static std::string fmt(const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    char buf[512];
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return std::string(buf);
}

static void say(const char* tag, const char* fmt, va_list ap) {
    std::printf("  %s ", tag);
    std::vprintf(fmt, ap);
    std::printf("\n");
}
static void ok(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); say("ok  ", fmt, ap); va_end(ap);
}
static void fail(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); say("FAIL", fmt, ap); va_end(ap); ++g_fail;
}
static void skip(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); say("skip", fmt, ap); va_end(ap); ++g_skip;
}

static const RadOpSchema* find_schema(const char* op) {
    if (!op) return nullptr;
    for (int i = 0; i < schemas(); ++i) {
        const RadOpSchema* s = schema_at(i);
        if (s && s->op && !std::strcmp(s->op, op)) return s;
    }
    return nullptr;
}

static const RadParamSpec* find_param(const RadOpSchema* s, const char* key) {
    if (!s || !key) return nullptr;
    for (int i = 0; i < s->n_params; ++i)
        if (s->params[i].key && !std::strcmp(s->params[i].key, key)) return &s->params[i];
    return nullptr;
}

/* ================================================================== the table's own checks
 *
 * Failure modes: a constraint on an undeclared key (the row silently matches nothing), a tunable
 * default outside its axis (the loader refuses the whole plugin), a row whose `op` names another
 * op's schema, a description whose operand count disagrees with its schema (every case skipped).
 */
static void check_table() {
    const int n = rows();
    std::printf("== rows: %d, schemas: %d\n", n, schemas());

    for (int i = 0; i < n; ++i) {
        const RadKernelInfo* k = row_at(i);
        if (!k) { fail("row %d is null", i); continue; }
        std::printf("%-26s %-18s %-6s %s\n", k->name ? k->name : "(null)",
                    k->op ? k->op : "(null)", k->family ? k->family : "-",
                    k->priority >= 0 ? "" : "no priority");

        if (!k->name || !*k->name) fail("row %d has no name", i);
        else
            for (int j = i + 1; j < n; ++j) {
                const RadKernelInfo* o = row_at(j);
                if (o && o->name && !std::strcmp(o->name, k->name))
                    fail("row '%s' is declared twice", k->name);
            }
        if (!k->launch) fail("row '%s' has no launch", k->name);
        if (k->init && !k->fini) fail("row '%s' inits without a fini", k->name);
        if (k->domain != RAD_DOMAIN_DEVICE) fail("row '%s' is not a device row", k->name);
        /* No row repacks a weight; kernels read the container's stored order. The exception is the
         * container's quantised weight (codes and scales as planes behind one operand), which the
         * mapped tier places only if a loaded library can read it in place. The rows below cover
         * w2a8's {1,group} grid with a zero, and the fp8a8 GEMM's and lm_head's {128,128} grid
         * without, each transcribed from the reference's hook. Any other name is a rule change and
         * should fail here. */
        const bool joined = 0 == std::strcmp(k->name, "gemm_nt_q_w2a8_m16")
                         || 0 == std::strcmp(k->name, "logits_gemm_bf16")
                         || 0 == std::strcmp(k->name, "gemm_nt_q_fp8a8_tiled")
                         || 0 == std::strcmp(k->name, "gemm_nt_q_w2a8_tiled")
                         || 0 == std::strcmp(k->name, "gemm_nt_q_w4a16asym_m64")
                         || 0 == std::strcmp(k->name, "gemm_nt_q_w5a16_m64");
        /* The query and the writer come as a pair, and the pair is the rule: a row with a hook must
         * supply the relayout that produces the form the hook names (abi/rad_abi.h -- a query
         * answering RAD_OK is a promise the loader keeps on the row's word), and no row may carry an
         * unrelayout, because nothing in this library invents a format a tool would need to invert.
         * A hook with a null writer is the failure the engine reports at load -- `could not produce
         * 'output.weight' (rad.planes) from fp8_e4m3*bf16[128x128]: unsupported` -- and a relayout
         * without a hook is a row repacking a weight nobody asked to have repacked.
                         || 0 == std::strcmp(k->name, "gemm_nt_q_w2a8_tiled");
        /* A layout hook and its relayout writer come as a pair (abi/rad_abi.h: a query answering
         * RAD_OK promises the loader the writer produces that form; a hook without one fails at
         * load). No row may carry an unrelayout, since this library invents no format a tool
         * would need to invert. */
        if ((k->layout && !joined) || (k->layout && !k->relayout) || (k->relayout && !joined)
            || (k->relayout && !k->layout) || k->unrelayout)
            fail("row '%s' declares a layout hook; libr3d stores weights in order", k->name);
        if (!k->computes || !*k->computes || !k->shape || !*k->shape || !k->dtypes ||
            !*k->dtypes)
            fail("row '%s' is missing prose", k->name);

        const RadOpSchema* s = find_schema(k->op);
        if (!s) { fail("row '%s' implements '%s', which no schema declares", k->name, k->op); }
        else {
            for (int c = 0; c < k->n_constraints; ++c) {
                const RadConstraint& x = k->constraints[c];
                const RadParamSpec* p = find_param(s, x.key);
                if (!p) {
                    fail("row '%s' constrains '%s', not a parameter of '%s'", k->name,
                         x.key ? x.key : "(null)", k->op);
                    continue;
                }
                /* String constraints match string parameters and integer ones integers; a mismatch
                 * matches nothing. */
                if (x.op == RAD_C_IN && p->type != RAD_P_STR)
                    fail("row '%s' matches '%s' with a string list on a non-string parameter",
                         k->name, x.key);
                if (x.op == RAD_C_EQ && p->type == RAD_P_STR && !x.sval)
                    fail("row '%s' equals the string parameter '%s' with no string", k->name,
                         x.key);
                if (x.op == RAD_C_IN && p->type == RAD_P_STR && x.sval) {
                    /* The band must name at least one value, and no value may be empty. */
                    int words = 0;
                    for (const char* q = x.sval; *q; ++q)
                        if (*q == ' ') ++words;
                    if (!words && !*x.sval) fail("row '%s' constrains '%s' to nothing", k->name,
                                                 x.key);
                }
                if (x.op == RAD_C_DIV && x.ival <= 0)
                    fail("row '%s' constrains '%s' to divide by %lld", k->name, x.key,
                         (long long)x.ival);
            }
        }

        for (int t = 0; t < k->n_tunables; ++t) {
            const RadTunable& a = k->tunables[t];
            if (!a.key || a.n_values <= 0 || !a.values) {
                fail("row '%s' axis %d is empty", k->name, t);
                continue;
            }
            bool has_default = false;
            for (int v = 0; v < a.n_values; ++v) {
                if (v && a.values[v] <= a.values[v - 1])
                    fail("row '%s' axis '%s' values are not ascending", k->name, a.key);
                if (a.values[v] == a.deflt) has_default = true;
            }
            if (!has_default)
                fail("row '%s' axis '%s' defaults to %lld, which is not one of its values",
                     k->name, a.key, (long long)a.deflt);
            if (!a.doc || !*a.doc) fail("row '%s' axis '%s' has no description", k->name, a.key);
        }
    }
}

/* ================================================================== the reference library
 *
 * The loader refuses the whole .so when its schema for an op differs from the first declarer's
 * (core/plugin/loader.cpp, `schema_same`). The comparison is positional: parameter count, each
 * parameter's key, type and requiredness; operand count, each operand's role and optionality.
 * Names and doc strings are free. This repeats that comparison against libref, naming the op and
 * the slot, since a mismatch otherwise shows only as one log line at engine load.
 */
static void check_reference_schemas() {
    std::printf("== schemas against the reference\n");

    /* libref.so beside the plugin under test, as the engine's home holds them. */
    std::string path(g_plugin);
    const size_t slash = path.find_last_of('/');
    path = (slash == std::string::npos ? std::string() : path.substr(0, slash + 1)) + "libref.so";
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { skip("no reference library at %s (%s)", path.c_str(), dlerror()); return; }
    auto rc = (int (*)(void))dlsym(h, "rad_kernel_schema_count");
    auto ra = (const RadOpSchema* (*)(int))dlsym(h, "rad_kernel_schema_at");
    if (!rc || !ra) { skip("%s declares no schemas", path.c_str()); return; }

    static const char* kRole[] = { "in", "out", "inout", "weight", "wtab" };
    auto role_name = [](int r) { return (r >= 0 && r <= 4) ? kRole[r] : "?"; };
    int mismatches = 0, compared = 0;
    for (int i = 0; i < schemas(); ++i) {
        const RadOpSchema* mine = schema_at(i);
        if (!mine || !mine->op) continue;
        const RadOpSchema* theirs = nullptr;
        for (int j = 0; j < rc() && !theirs; ++j) {
            const RadOpSchema* o = ra(j);
            if (o && o->op && !std::strcmp(o->op, mine->op)) theirs = o;
        }
        if (!theirs) {
            /* Not a refusal: an op the reference does not declare has nothing to agree with. */
            skip("'%s' has no schema in the reference, so nothing to compare against", mine->op);
            continue;
        }
        ++compared;
        std::string diff;
        if (mine->n_params != theirs->n_params) {
            diff = fmt("%d parameters vs %d", mine->n_params, theirs->n_params);
        } else {
            for (int q = 0; q < mine->n_params && diff.empty(); ++q) {
                const RadParamSpec& x = mine->params[q];
                const RadParamSpec& y = theirs->params[q];
                if (!x.key || std::strcmp(x.key, y.key ? y.key : ""))
                    diff = fmt("parameter %d is '%s' vs '%s'", q, x.key ? x.key : "(null)",
                               y.key ? y.key : "(null)");
                else if (x.type != y.type)
                    diff = fmt("parameter '%s' is %s vs %s", x.key,
                               x.type == RAD_P_STR ? "a string" : "an int",
                               y.type == RAD_P_STR ? "a string" : "an int");
                else if (x.required != y.required)
                    diff = fmt("parameter '%s' is %s in the reference", x.key,
                               y.required ? "required" : "optional");
                else if (x.role != y.role)
                    diff = fmt("parameter '%s' has core role %d vs %d", x.key, x.role, y.role);
            }
            if (diff.empty() && mine->n_operands != theirs->n_operands)
                diff = fmt("%d operands vs %d", mine->n_operands, theirs->n_operands);
            for (int q = 0; q < mine->n_operands && diff.empty(); ++q) {
                const RadOperandSpec& x = mine->operands[q];
                const RadOperandSpec& y = theirs->operands[q];
                /* The name is free; the role and the optionality are the contract. */
                if (x.role != y.role)
                    diff = fmt("operand %d ('%s' / '%s') has a different role: %s vs %s", q,
                               x.name ? x.name : "?", y.name ? y.name : "?",
                               role_name(x.role), role_name(y.role));
                else if (x.optional != y.optional)
                    diff = fmt("operand %d ('%s') is %s in the reference", q,
                               x.name ? x.name : "?", y.optional ? "optional" : "required");
            }
        }
        if (!diff.empty()) {
            fail("'%s' has two schemas: the reference fixed it, libr3d disagrees (%s)", mine->op,
                 diff.c_str());
            ++mismatches;
        }
    }
    if (!mismatches)
        ok("%d of %d schemas match the reference, slot for slot%s", compared, schemas(),
           schemas() == compared ? "" : " (the rest the reference does not declare)");
    dlclose(h);
}

/* ================================================================== band reachability
 *
 * A constraint on a key the band's geometry lacks does not hold (core/util/rad_util.cpp,
 * constraint_holds), and bands carry only the parameters their graph set. So a row that constrains
 * an optional parameter serves only graphs that state it. kbench cannot show this, because its
 * fixtures name every parameter. Such a constraint is legitimate but must be deliberate, so each one
 * is listed here.
 */
static void check_band_reach() {
    std::printf("== band reachability\n");
    std::string path(g_plugin);
    const size_t slash = path.find_last_of('/');
    path = (slash == std::string::npos ? std::string() : path.substr(0, slash + 1)) + "libref.so";
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { skip("no reference library at %s", path.c_str()); return; }
    auto rc = (int (*)(void))dlsym(h, "rad_kernel_schema_count");
    auto ra = (const RadOpSchema* (*)(int))dlsym(h, "rad_kernel_schema_at");
    if (!rc || !ra) { skip("%s declares no schemas", path.c_str()); return; }

    int narrowed = 0, checked = 0;
    for (int i = 0; i < rows(); ++i) {
        const RadKernelInfo* k = row_at(i);
        if (!k || !k->op) continue;
        const RadOpSchema* s = nullptr;
        for (int j = 0; j < rc() && !s; ++j) {
            const RadOpSchema* o = ra(j);
            if (o && o->op && !std::strcmp(o->op, k->op)) s = o;
        }
        if (!s) continue;                       /* not in the reference: nothing to compare */
        for (int ci = 0; ci < k->n_constraints; ++ci) {
            const RadConstraint& c = k->constraints[ci];
            if (!c.key) continue;
            ++checked;
            const RadParamSpec* sp = nullptr;
            for (int q = 0; q < s->n_params && !sp; ++q)
                if (s->params[q].key && !std::strcmp(s->params[q].key, c.key)) sp = &s->params[q];
            if (sp && !sp->required) {
                ok("row '%s' constrains '%s', optional on '%s': a band that leaves it out does not "
                   "resolve to this row", k->name, c.key, k->op);
                ++narrowed;
            }
        }
    }
    if (!narrowed) ok("%d constraint(s) over %d rows, every one on a parameter a band always "
                      "carries", checked, rows());
    dlclose(h);
}

/* ================================================================== the operand descriptions
 *
 * Against the schemas, for one geometry per op at real model widths (4096 hidden, 128 head, 11008
 * intermediate, 128256 vocabulary). A wrong operand count, rank or zero extent is a bug in the
 * description or the schema; rad-kbench would only report it as skipped cases.
 */
static void check_shapes() {
    /* RadParam is { key, kind, ival, ihi, sval, dval }. */
#define PI(k, v) { (k), RAD_P_INT, (long long)(v), 0, nullptr, 0.0 }
#define PF(k, v) { (k), RAD_P_F64, 0, 0, nullptr, (double)(v) }
#define PS(k, v) { (k), RAD_P_STR, 0, 0, (v), 0.0 }
    struct Geom { const char* op; std::vector<RadParam> p; };
    const std::vector<Geom> g = {
        { "rmsnorm", { PI("M", 8), PI("n", 4096), PF("eps", 1e-5), PS("dtype", "bf16") } },
        { "layernorm", { PI("M", 8), PI("n", 4096), PF("eps", 1e-5), PS("dtype", "bf16") } },
        { "add", { PI("M", 8), PI("n", 4096), PS("dtype", "bf16") } },
        { "mul", { PI("M", 8), PI("n", 4096), PS("dtype", "bf16") } },
        { "act_mul", { PI("M", 8), PI("n", 11008), PS("act", "silu"), PS("dtype", "bf16") } },
        { "silu", { PI("M", 8), PI("n", 11008), PS("dtype", "bf16") } },
        { "gelu", { PI("M", 8), PI("n", 11008), PS("dtype", "bf16") } },
        { "sigmoid", { PI("M", 8), PI("n", 4096), PS("dtype", "bf16") } },
        { "softmax", { PI("M", 8), PI("n", 4096), PS("dtype", "f32") } },
        { "cast", { PI("M", 8), PI("n", 4096), PS("from", "bf16"), PS("to", "f32") } },
        { "scale", { PI("M", 8), PI("n", 4096), PS("dtype", "bf16") } },
        { "fill", { PI("M", 8), PI("n", 4096), PS("dtype", "bf16"), PF("value", 0.0) } },
        { "rope", { PI("M", 8), PI("head_dim", 128), PI("n_head", 32), PI("n_head_kv", 8),
                    PF("theta", 500000.0), PF("scale", 1.0), PS("mode", "neox") } },
        /* The row forms as LFM2 and LFM3 declare them: the q/k head norms over rows of head_dim
         * (64 and 128), and q rotated as a q-only row (n_head_kv 0); LFM3's full layers turn 64 of
         * 128 channels on the proportional ladder. */
        { "rmsnorm_rows", { PI("M", 8 * 32), PI("n", 64), PF("eps", 1e-5), PS("dtype", "bf16") } },
        { "rmsnorm_rows", { PI("M", 8 * 16), PI("n", 128), PF("eps", 1e-5), PS("dtype", "bf16") } },
        { "rope_rows", { PI("M", 8), PI("head_dim", 64), PI("n_head", 32), PI("n_head_kv", 0),
                         PF("theta", 1000000.0), PF("scale", 1.0), PS("mode", "neox") } },
        { "rope_rows", { PI("M", 8), PI("head_dim", 128), PI("n_head", 16), PI("n_head_kv", 0),
                         PF("theta", 1000000.0), PF("scale", 1.0), PS("mode", "neox"),
                         PI("rotary_dim", 64), PS("ladder", "head") } },
        { "kv_store", { PI("M", 8), PI("head_dim", 128), PI("n_head_kv", 8), PI("block_size", 32),
                        PS("kv_dtype", "bf16") } },
        { "embed_lookup", { PI("M", 8), PI("n_embd", 4096), PI("n_vocab", 128256),
                            PS("dtype", "bf16") } },
        { "gather_rows", { PI("M", 8), PI("n", 4096), PS("dtype", "bf16") } },
        { "scatter_rows", { PI("M", 8), PI("n", 4096), PS("dtype", "bf16") } },
        { "quant_act_fp8", { PI("M", 8), PI("n", 4096), PI("group", 128), PS("dtype", "bf16") } },
        { "quant_act_i8", { PI("M", 8), PI("n", 4096), PI("group", 128), PS("dtype", "bf16") } },
        { "quant_act_i8g", { PI("M", 8), PI("n", 4096), PI("group", 128), PS("dtype", "bf16") } },
        { "had_quant_act_i8", { PI("M", 8), PI("n", 1024), PI("group", 128), PS("dtype", "bf16") } },
        { "gelu_mul_quant_i8", { PI("M", 8), PI("n", 1024), PS("dtype", "bf16") } },
        { "kv_len_trim", { PI("trim", 4), PI("max_seqs", 8) } },
        { "qsa_tail_store", { PI("M", 16), PI("n", 128), PI("ratio", 4), PI("ring", 8),
                              PI("seqs", 2), PS("dtype", "bf16") } },
        { "gated_had_quant_i8", { PI("M", 8), PI("n", 128), PI("group", 128), PS("dtype", "bf16") } },
        { "had_quant_act_fp8", { PI("M", 8), PI("n", 128), PI("group", 128), PS("dtype", "bf16") } },
        { "gated_had_quant_fp8", { PI("M", 8), PI("n", 128), PI("group", 128), PS("dtype", "bf16") } },
        { "quant_cols_fp8", { PI("M", 32), PI("n", 128), PS("dtype", "bf16") } },
        { "had_quant_act_i8g", { PI("M", 8), PI("n", 128), PI("group", 128), PS("dtype", "bf16") } },
        { "had_quant_act_i8g", { PI("M", 8), PI("n", 7680), PI("group", 128), PS("dtype", "bf16") } },
        { "gated_had_quant_i8g", { PI("M", 8), PI("n", 128), PI("group", 128), PS("dtype", "bf16") } },
        { "rmsnorm_quant_i8", { PI("M", 8), PI("n", 128), PF("eps", 1e-6), PS("dtype", "bf16") } },
        { "gated_quant_fp8", { PI("M", 8), PI("n", 11008), PI("group", 128), PS("dtype", "bf16"),
                               PS("act", "silu") } },
        { "dequant", { PI("M", 11008), PI("n", 4096), PI("group", 128),
                       PS("dtype", "fp8_e4m3") } },
        { "rmsnorm_quant_fp8", { PI("M", 1792), PI("n", 5120), PF("eps", 1e-5), PI("group", 128),
                                 PS("dtype", "bf16") } },
        { "gemm_nt", { PI("M", 8), PI("N", 11008), PI("K", 4096), PS("dtype", "bf16") } },
        { "gemm_nt", { PI("M", 8), PI("N", 11008), PI("K", 4096), PS("dtype", "f16") } },
        { "gemm_nt_q", { PI("M", 8), PI("N", 11008), PI("K", 4096), PI("group", 128),
                         PS("dtype", "fp8a16"), PI("pb1", 128), PI("pb2", 128) } },
        /* The rest are MiniCPM5-2B's bands: vocab 130560, MLP 6144 wide, head 128 with 16 heads
         * over 2 KV heads at 16-token blocks. */
        { "silu_mul", { PI("M", 1792), PI("n", 6144), PS("dtype", "bf16") } },
        { "attn_paged", { PI("q_len", 2048), PI("head_dim", 128), PI("gqa", 8), PI("block_size", 16),
                          PI("causal", 1), PI("window", 0), PS("q_dtype", "bf16"),
                          PS("kv_dtype", "bf16"), PI("n_head", 16), PI("max_ctx", 4096),
                          PI("max_seqs", 256) } },
        { "attn_paged", { PI("q_len", 7), PI("head_dim", 128), PI("gqa", 8), PI("block_size", 16),
                          PI("causal", 0), PI("window", 0), PS("q_dtype", "bf16"),
                          PS("kv_dtype", "fp8_e4m3"), PI("n_head", 16), PI("max_ctx", 4096),
                          PI("max_seqs", 256) } },
        /* A context deeper than the query chunk, so the walk covers keys before the chunk; then a
         * window that starts mid-block over an fp8 cache, which exercises tile sharing. */
        { "attn_paged", { PI("q_len", 7), PI("head_dim", 128), PI("gqa", 8), PI("block_size", 16),
                          PI("causal", 0), PI("window", 0), PS("q_dtype", "bf16"),
                          PS("kv_dtype", "bf16"), PI("n_head", 16), PI("ctx", 4096),
                          PI("max_ctx", 4096) } },
        { "attn_paged", { PI("q_len", 7), PI("head_dim", 128), PI("gqa", 8), PI("block_size", 16),
                          PI("causal", 1), PI("window", 37), PS("q_dtype", "bf16"),
                          PS("kv_dtype", "fp8_e4m3"), PI("n_head", 16), PI("ctx", 512),
                          PI("max_ctx", 4096) } },
        /* A 27B model's sliding-window band as the fixture draws it: the descales are absent, so an
         * fp8 cache has no scale plane and the reference reads 1.0. */
        { "attn_paged", { PI("q_len", 1), PI("head_dim", 128), PI("gqa", 4), PI("block_size", 16),
                          PI("causal", 0), PI("window", 2048), PS("q_dtype", "bf16"),
                          PS("kv_dtype", "fp8_e4m3"), PI("n_head", 32), PI("max_seqs", 8),
                          PI("max_ctx", 262144) } },
        { "row_topk", { PI("M", 1792), PI("N", 130560), PI("R", 16), PS("dtype", "bf16") } },
        { "sample_argmax", { PI("M", 2048), PI("n_vocab", 130560) } },
        { "sample_temp", { PI("M", 2048), PI("n_vocab", 130560) } },
        { "sample_topk", { PI("M", 2048), PI("n_vocab", 130560) } },
        { "sample_topp", { PI("M", 2048), PI("n_cand", 1024) } },
        { "sample_minp", { PI("M", 2048), PI("n_cand", 1024) } },
        { "sample_typical", { PI("M", 2048), PI("n_cand", 1024) } },
        { "sample_xtc", { PI("M", 2048), PI("n_cand", 1024) } },
        { "sample_pick", { PI("M", 2048), PI("n_cand", 1024) } },
        { "sample_penalties", { PI("M", 2048), PI("n_vocab", 130560) } },
        { "sample_dry", { PI("M", 2048), PI("n_vocab", 130560) } },
        { "sample_ngram_ban", { PI("M", 2048), PI("n_vocab", 130560) } },
        { "sample_mask", { PI("M", 2048), PI("n_vocab", 130560) } },
        { "sample_logprobs", { PI("M", 2048), PI("n_vocab", 130560), PI("k", 21) } },
        /* MiniCPM5-2B's speculative decoder band: a 7-token draft block, 16 candidates a step, a
         * rank-256 codebook over a 130560-token vocabulary (63.8 MiB each), anchors 7 apart. */
        { "dflash_select", { PI("M", 256), PI("steps", 7), PI("top_k", 16), PI("rank", 256),
                             PI("n_vocab", 130560), PI("anchor_stride", 7), PS("dtype", "bf16") } },
    };
#undef PI
#undef PF
#undef PS

    /* Shape hooks for ops with no row in the table, found by symbol name in the plugin, so their
     * descriptions are checked even without a kernel. */
    static const struct { const char* op; const char* sym; } kHooks[] = {
        { "silu_mul", "r3d_shape_silu_mul" },           { "attn_paged", "r3d_shape_attn_paged" },
        { "row_topk", "r3d_shape_row_topk" },           { "sample_argmax", "r3d_shape_sample_argmax" },
        { "sample_temp", "r3d_shape_sample_temp" },     { "sample_topk", "r3d_shape_sample_topk" },
        { "sample_topp", "r3d_shape_sample_topp" },     { "sample_minp", "r3d_shape_sample_minp" },
        { "sample_typical", "r3d_shape_sample_typical" },
        { "sample_xtc", "r3d_shape_sample_xtc" },       { "sample_pick", "r3d_shape_sample_pick" },
        { "sample_penalties", "r3d_shape_sample_penalties" },
        { "sample_dry", "r3d_shape_sample_dry" },       { "sample_ngram_ban", "r3d_shape_sample_ngram_ban" },
        { "sample_mask", "r3d_shape_sample_mask" },     { "sample_logprobs", "r3d_shape_sample_logprobs" },
    };

    for (const Geom& gm : g) {
        const RadOpSchema* s = find_schema(gm.op);
        if (!s) { fail("the geometry for '%s' has no schema", gm.op); continue; }
        const RadKernelInfo* row = nullptr;
        RadShapeFn hook = nullptr;
        for (int i = 0; i < rows() && !row; ++i) {
            const RadKernelInfo* k = row_at(i);
            if (k && k->op && !std::strcmp(k->op, gm.op) && k->opd_shape) row = k;
        }
        if (row) hook = row->opd_shape;
        for (size_t q = 0; q < NELEM(kHooks) && !hook; ++q)
            if (!std::strcmp(kHooks[q].op, gm.op))
                hook = (int (*)(const RadParam*, int, int, RadOpdDesc*))dlsym(g_h, kHooks[q].sym);
        if (!hook) { skip("no operand description for '%s' -- no row and no hook", gm.op);
                     continue; }
        if (!row) std::printf("  --   '%s': operand description from the shape hook (no row)\n", gm.op);

        bool bad = false;
        for (int o = 0; o < s->n_operands && !bad; ++o) {
            RadOpdDesc d{};
            const int st = hook(gm.p.data(), (int)gm.p.size(), o, &d);
            if (st == RAD_E_UNSUPPORTED) { skip("'%s': the description declines this geometry",
                                                gm.op); bad = true; break; }
            if (st != RAD_OK) { fail("'%s' operand %d: the description returns %d", gm.op, o, st);
                                bad = true; break; }
            if (d.flags & RAD_OPD_F_ABSENT) {
                if (!s->operands[o].optional)
                    { fail("'%s' operand %d is required and described absent", gm.op, o); bad = true; }
                continue;
            }
            if (d.rank == 0 || d.rank > RAD_MAX_RANK)
                { fail("'%s' operand %d has rank %u", gm.op, o, d.rank); bad = true; break; }
            for (uint32_t r = 0; r < d.rank && !bad; ++r)
                if (d.shape[r] <= 0)
                    { fail("'%s' operand %d extent %u is %lld", gm.op, o, r,
                           (long long)d.shape[r]); bad = true; }
            if (rad_dtype_bits(d.dtype) <= 0)
                { fail("'%s' operand %d has an unparseable dtype", gm.op, o); bad = true; }
        }
        if (!bad) ok("'%s' described over %d operands", gm.op, s->n_operands);
    }
}

/* Two floats are the same when their bits are: `a == b` is false for two NaNs, and a codec that
 * quiets a NaN the way the ABI's does agrees. */
static bool same_float(float a, float b) {
    return bits_of(a) == bits_of(b);
}

/* ================================================================== the codecs
 *
 * Against the ABI's host implementations, bit for bit: the widenings are exact (seven mantissa bits
 * into ten; four exponent and three mantissa bits into five and ten) and the encoders mirror
 * `rad_plugin.h`.
 */
static void check_codecs() {
    std::printf("== codecs\n");

    /* bf16: every one of the 65536 patterns, both directions. */
    int bad = 0;
    for (uint32_t u = 0; u < 0x10000 && !bad; ++u) {
        const uint16_t b = (uint16_t)u;
        if (!same_float(bf16_to_f32(b), rad_bf16_to_f32(b))) { ++bad; break; }
        /* Encoder against encoder: both quiet a NaN's payload, so a round trip to the original
         * pattern is not the claim. */
        if (f32_to_bf16(bf16_to_f32(b)) != rad_f32_to_bf16(rad_bf16_to_f32(b))) { ++bad; break; }
    }
    if (bad) fail("bf16: the codecs disagree with rad_plugin.h");
    else ok("bf16: 65536 patterns, decode and encode, match rad_plugin.h bit for bit");

    /* f16: the same, subnormals included (a naive shift-and-round loses them). */
    bad = 0;
    for (uint32_t u = 0; u < 0x10000 && !bad; ++u) {
        const uint16_t h = (uint16_t)u;
        const float f = f16_to_f32(h);
        if (!same_float(f, rad_f16_to_f32(h))) { ++bad; break; }
        if (f32_to_f16(f) != rad_f32_to_f16(rad_f16_to_f32(h))) { ++bad; break; }
    }
    if (bad) fail("f16: the codecs disagree with rad_plugin.h");
    else ok("f16: 65536 patterns, both directions, match rad_plugin.h (subnormals included)");

    /* A bf16 pair widened to an f16 pair in one register, which the dot2 path relies on. Exact over
     * f16's normal range [2^-14, 65504] (r3d_codecs.h), which is what is swept. Outside it the
     * widening saturates above 65504 and truncates toward zero below 2^-14, where f16's subnormal
     * grid is coarser than bf16's; those patterns are counted in the message. */
    bad = 0;
    int inside = 0, outside = 0;
    for (uint32_t u = 0; u < 0x10000 && !bad; ++u) {
        const uint16_t b = (uint16_t)u;
        const float v = bf16_to_f32(b);
        const float av = v < 0.0f ? -v : v;
        if (!(av >= 6.103515625e-05f && av <= 65504.0f)) { ++outside; continue; }  /* f16 normals */
        ++inside;
        const uint32_t w = ((uint32_t)(uint16_t)(b ^ 0x8000u) << 16) | b;
        const uint32_t p = bf16x2_to_f16x2(w);
        if (!same_float(f16_to_f32((uint16_t)(p & 0xFFFFu)), v) ||
            !same_float(f16_to_f32((uint16_t)(p >> 16)), -v)) { ++bad; break; }
    }
    if (bad) fail("bf16x2 -> f16x2 is not exact over f16's normal range");
    else ok("bf16 -> f16 exact over the %d patterns in f16's normal range (the other %d saturate "
            "or truncate as the format says they must)", inside, outside);

    /* E4M3, both directions, over every code, and the exact widening to f16 (448 is well inside
     * f16's range). */
    bad = 0;
    for (uint32_t c = 0; c < 256 && !bad; ++c) {
        if (!same_float(e4m3_to_f32((uint8_t)c), rad_fp8e4m3_to_f32((uint8_t)c))) { ++bad; break; }
        if ((c & 0x7Fu) == 0x7Fu) continue;                       /* the code's one NaN */
        if (f32_to_e4m3(e4m3_to_f32((uint8_t)c)) != (uint8_t)c) { ++bad; break; }
        const uint32_t p = e4m3x2_to_f16x2((c << 8) | c);
        const float want = e4m3_to_f32((uint8_t)c);
        if (!same_float(f16_to_f32((uint16_t)(p & 0xFFFFu)), want) ||
            !same_float(f16_to_f32((uint16_t)(p >> 16)), want)) { ++bad; break; }
    }
    if (bad) fail("e4m3: the codecs disagree with rad_plugin.h, or the widening is not exact");
    else ok("e4m3: 256 codes both ways, and the f16 widening exact, match rad_plugin.h");

    /* The encoder over the cases that break a naive one: saturation at 448, subnormals below 2^-6,
     * the sign of zero, NaN, and typical scale values. */
    bad = 0;
    const float sweep[] = { 0.0f, -0.0f, 1.0f / 512, 1.0f / 1024, 1.0f / 6, 0.33f, 1.0f, 1.7f,
                            2.0f, 3.0f, 447.0f, 448.0f, 449.0f, 1024.0f, 1e30f, -1e30f,
                            1.0f / 0.0f, -1.0f / 0.0f, 0.0f / 0.0f, 1.0f / 448, 5.0f / 448,
                            448.0f * 1.0000001f, 447.99f, -0.0f - 0.0f };
    int sbad = 0;
    for (float v : sweep)
        if (f32_to_e4m3(v) != rad_f32_to_fp8e4m3(v)) ++sbad;
    if (sbad) fail("e4m3: %d of %d sweep values disagree with rad_plugin.h", sbad,
                   (int)(sizeof(sweep) / sizeof(sweep[0])));
    else ok("e4m3 encoder: saturating, subnormal, signed-zero and NaN cases match rad_plugin.h");
    (void)bad;

    /* int8: round to nearest, ties to even, saturating, over every tie and past the band. */
    bad = 0;
    for (int i = -600; i <= 600 && !bad; ++i)
        for (int f = 0; f < 4; ++f) {
            const float v = (float)i + 0.25f * (float)f;
            if (f32_to_i8_rtne(v) != (int8_t)rad_round_even_sat(v, -128.0, 127.0)) { ++bad; break; }
        }
    if (bad) fail("int8: the encoder disagrees with rad_round_even_sat");
    else ok("int8 encoder: 4804 values, every tie in the band, match rad_round_even_sat");

    /* dot2's operand order only: the device instruction folds the products with guard bits, so
     * this checks that pairs are read low-then-high; accumulation order is rad-kbench's to check. */
    float acc = 0.0f;
    const uint32_t a = f16x2_pack(1.5f, -2.5f), b = f16x2_pack(4.0f, 4.0f);
    acc = dot2_f16x2(a, b, acc);
    if (acc != 1.5f * 4.0f + -2.5f * 4.0f) fail("dot2: the pair is not read low-then-high");
    else ok("dot2: the pair is read low-then-high, into the accumulator it is handed");

    /* The epilogue's bf16 store: two values, one dword, round to nearest even, the high value in
     * the high half. A `v_perm` picking the wrong bytes shows here; in a kernel's output a swapped
     * pair looks plausible. */
    bad = 0;
    for (int i = -500; i < 500 && !bad; ++i) {
        const float lo = (float)i * 0.001f, hi = (float)i * -0.002f;
        const uint32_t w = bf16x2_rne(lo, hi);
        if (!same_float(bf16w_lo(w), rad_bf16_to_f32(rad_f32_to_bf16(lo))) ||
            !same_float(bf16w_hi(w), rad_bf16_to_f32(rad_f32_to_bf16(hi)))) { ++bad; break; }
    }
    if (bad) fail("bf16x2_rne: the pair is not (low, high) as rad_f32_to_bf16 rounds it");
    else ok("bf16x2_rne: a pair packs as rad_f32_to_bf16 rounds it, low half low");
}

/* ================================================================== a row, run
 *
 * Buffers from the row's own `opd_shape`, filled per the fill code the description carries, the
 * launch called, the outputs read back and checked to be finite.
 */
enum { MAX_OPD = 24 };

struct Case {
    RadTensor t[MAX_OPD];
    RadParam p[32];
    std::vector<void*> dev;
    std::vector<void*> host;       /* pinned operands, read by both host and kernel */
    std::vector<std::string> words;   /* the strings out.sval points into; reserved in make_params */
    int n_p = 0;
};

/* ================================================================== the lane helpers, on the card
 *
 * Every wave-sum in the library is five butterfly rounds over lane shuffles, and an error there is
 * invisible to a codec sweep: `d += (double)f((unsigned long long)d)` moves a value where a bit
 * pattern is wanted, and a wave of fractions sums to an integer. The device runs the butterflies
 * through `r3d_lanes_check` (r3d_sample_pre_f32.hip) and every lane of every wave is compared with
 * a host emulation of the same five rounds in the same order.
 */
static void check_lanes() {
    std::printf("== lane helpers\n");
    auto run = (int (*)(long, long))dlsym(g_h, "r3d_lanes_check");
    if (!run) { skip("r3d_lanes_check is not in this plugin"); return; }
#ifndef RAD_HAVE_HIP
    skip("built without HIP: the lane helpers are not checked");
    return;
#else
    enum { NF = 4, NT = 256, NW = 32 };
    auto seed = [](int l) {
        return l == 6 ? 0.08945 : l == 9 ? 1.0 : l == 19 ? 0.5971 : l == 25 ? 0.9768 : 0.0;
    };
    /* The butterfly the kernel runs, emulated: five rounds, lane l meeting lane l^N. */
    double wd[NW];
    float wf[NW];
    for (int l = 0; l < NW; ++l) { wd[l] = seed(l); wf[l] = (float)seed(l); }
    for (int N = NW / 2; N >= 1; N >>= 1) {
        double t[NW];
        float u[NW];
        for (int l = 0; l < NW; ++l) { t[l] = wd[l ^ N]; u[l] = wf[l ^ N]; }
        for (int l = 0; l < NW; ++l) { wd[l] += t[l]; wf[l] += u[l]; }
    }
    static const char* names[NF] = {
        "f32 butterfly, the shape every wave_sum is",
        "f64 halves (r3d_xor_lane_u64), lane-derived",
        "f64 halves (r3d_xor_lane_u64), loaded from LDS",
        "f64 one 64-bit __shfl_xor_sync, loaded from LDS",
    };
    double* d = nullptr;
    const size_t bytes = (size_t)NF * NT * sizeof(double);
    if (hipMalloc(&d, bytes) != hipSuccess) { skip("no device for the lane helpers"); return; }
    if (hipMemset(d, 0, bytes) != hipSuccess) { fail("could not clear the lane helper's plane");
                                                (void)hipFree(d); return; }
    if (run((long)(uintptr_t)d, 0) != 0) {
        fail("r3d_lanes_check refused to launch");
        (void)hipFree(d);
        return;
    }
    if (hipDeviceSynchronize() != hipSuccess) {
        fail("the lane helper kernel faulted");
        (void)hipFree(d);
        return;
    }
    std::vector<double> out((size_t)NF * NT, 0.0);
    if (hipMemcpy(out.data(), d, bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        fail("reading the lane helper's plane back failed");
        (void)hipFree(d);
        return;
    }
    (void)hipFree(d);
    for (int f = 0; f < NF; ++f) {
        int bad = 0, first = -1;
        for (int i = 0; i < NT; ++i) {
            const double w = (f == 0) ? (double)wf[i % NW] : wd[i % NW];
            uint64_t a, b;
            std::memcpy(&a, &out[(size_t)f * NT + i], 8);
            std::memcpy(&b, &w, 8);
            if (a != b) { ++bad; if (first < 0) first = i; }
        }
        if (!bad) {
            ok("%s: %d lanes over %d waves, every one exact (wave sums to %.6f)", names[f], NT,
               NT / NW, wd[0]);
            continue;
        }
        const double w = (f == 0) ? (double)wf[first % NW] : wd[first % NW];
        fail("%s: %d of %d lanes disagree; first is lane %d of wave %d: %.17g, want %.17g",
             names[f], bad, NT, first % NW, first / NW, out[(size_t)f * NT + first], w);
    }
#endif
}

static size_t dt_bytes(uint32_t dt) {
    const int bits = rad_dtype_bits(dt);
    return bits > 0 ? (size_t)((bits + 7) / 8) : 1;
}

static uint32_t xorshift(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

/* Fill a host plane per its description. "Normals" are a sum of four uniforms: any non-special
 * value exercises the arithmetic, and special cases are rad-kbench's. */
static void fill_plane(const RadOpdDesc& d, uint8_t* p, size_t nel, uint32_t seed) {
    uint32_t s = seed ? seed : 1u;
    const size_t nbyte = d.dtype == RAD_U2 ? nel / 4
                       : (d.dtype == RAD_U4 || d.dtype == RAD_I4 || d.dtype == RAD_FP4E2M1) ? nel / 2
                       : nel * dt_bytes(d.dtype);
    std::memset(p, 0, nbyte);
    if (d.fill == RAD_FILL_INDEX) {
        const long long mx = d.idx_const >= 0 ? d.idx_const
                           : (d.idx_max > 0 ? d.idx_max : (long long)nel);
        for (size_t i = 0; i < nel; ++i) {
            const long long v = d.idx_const >= 0
                                    ? d.idx_const
                                    : (long long)(xorshift(s) % (uint32_t)(mx > 0 ? mx : 1));
            if (dt_bytes(d.dtype) >= 8) *(int64_t*)(p + i * dt_bytes(d.dtype)) = v;
            else if (dt_bytes(d.dtype) == 4) *(int32_t*)(p + i * 4) = (int32_t)v;
            else p[i] = (uint8_t)v;
        }
        return;
    }
    if (d.dtype == RAD_I4) {
        /* Two signed codes a byte, low nibble first, drawn in -7..7 and stored as the low four bits
         * of two's complement. */
        for (size_t j = 0; j < nel / 2; ++j) {
            double v0 = ((double)xorshift(s) / 4294967296.0) * 2.0 - 1.0;
            double v1 = ((double)xorshift(s) / 4294967296.0) * 2.0 - 1.0;
            const int c0 = (int)(v0 * 7.0);
            const int c1 = (int)(v1 * 7.0);
            p[j] = (uint8_t)(((uint32_t)c0 & 0xFu) | (((uint32_t)c1 & 0xFu) << 4));
        }
        return;
    }
    if (d.dtype == RAD_FP4E2M1) {
        /* Two e2m1 codes a byte over all 16 values, exercising the unpack's magnitude and sign
         * paths. */
        for (size_t j = 0; j < nel / 2; ++j) {
            const uint8_t lo = (uint8_t)(xorshift(s) & 0xFu);
            const uint8_t hi = (uint8_t)(xorshift(s) & 0xFu);
            p[j] = (uint8_t)(lo | (hi << 4));
        }
        return;
    }
    for (size_t i = 0; i < nel; ++i) {
        double v = 0.0;
        for (int k = 0; k < 4; ++k) v += (double)xorshift(s) / 4294967296.0;
        v = v / 4.0 * 2.0 - 1.0;                       /* [-1, 1) */
        if (d.fill == RAD_FILL_SIGMOID) v = 0.05 + 0.9 * (v * 0.5 + 0.5);
        switch (d.dtype) {
            case RAD_F32:    *(float*)(p + i * 4) = (float)v; break;
            case RAD_BF16:   *(uint16_t*)(p + i * 2) = f32_to_bf16((float)v); break;
            case RAD_F16:    *(uint16_t*)(p + i * 2) = f32_to_f16((float)v); break;
            case RAD_F8E4M3: p[i] = f32_to_e4m3((float)v); break;
            case RAD_I8:     *(int8_t*)(p + i) = (int8_t)(v * 100.0); break;
            case RAD_U8:     p[i] = (uint8_t)(v * 100.0 + 128.0); break;
            case RAD_I32:
            case RAD_U32:    *(int32_t*)(p + i * 4) = (int32_t)(v * 1000.0); break;
            case RAD_I64:    *(int64_t*)(p + i * 8) = (int64_t)(v * 1000.0); break;
            default: break;
        }
    }
}

/* The geometry a row runs at: the row's constraint values where it has them, real model widths
 * elsewhere, so the constraint table itself is what is tested. */
static void make_params(Case& c, const RadKernelInfo* k, const RadOpSchema* s) {
    c.words.reserve(32);   /* more parameters than any op declares, so no string ever moves */
    for (int q = 0; q < s->n_params && c.n_p < 32; ++q) {
        const RadParamSpec& sp = s->params[q];
        /* Only required and constrained parameters, as the engine passes them: an optional
         * parameter the graph did not set is absent, and the kernel uses libref's `p_*` default.
         * Filling every optional slot would test geometries no graph produces (for rope, the
         * `sections` string would switch to a mode the row's constraints exclude). */
        bool constrained = false;
        for (int ci = 0; ci < k->n_constraints && !constrained; ++ci)
            if (k->constraints[ci].key && sp.key && !std::strcmp(k->constraints[ci].key, sp.key))
                constrained = true;
        if (!sp.required && !constrained) continue;
        RadParam out{};
        out.key = sp.key;
        out.kind = sp.type;
        out.ival = 8;
        out.dval = 1.0;
        out.sval = "bf16";
        if (sp.key && !std::strcmp(sp.key, "eps")) out.dval = 1e-5;
        if (sp.key && !std::strcmp(sp.key, "theta")) out.dval = 500000.0;
        if (sp.key && !std::strcmp(sp.key, "head_dim")) out.ival = 128;
        if (sp.key && !std::strcmp(sp.key, "n_head")) out.ival = 8;
        if (sp.key && !std::strcmp(sp.key, "n_head_kv")) out.ival = 8;
        if (sp.key && !std::strcmp(sp.key, "block_size")) out.ival = 32;
        if (sp.key && !std::strcmp(sp.key, "gqa")) out.ival = 1;
        if (sp.key && !std::strcmp(sp.key, "q_len")) out.ival = 1;
        if (sp.key && !std::strcmp(sp.key, "causal")) out.ival = 1;
        if (sp.key && !std::strcmp(sp.key, "window")) out.ival = -1;
        if (sp.key && !std::strcmp(sp.key, "n")) out.ival = 4096;
        if (sp.key && !std::strcmp(sp.key, "N")) out.ival = 11008;
        if (sp.key && !std::strcmp(sp.key, "K")) out.ival = 4096;
        if (sp.key && !std::strcmp(sp.key, "n_embd")) out.ival = 4096;
        if (sp.key && !std::strcmp(sp.key, "n_vocab")) out.ival = 128256;
        if (sp.key && !std::strcmp(sp.key, "group")) out.ival = 128;
        if (sp.key && !std::strcmp(sp.key, "pb1")) out.ival = 128;
        if (sp.key && !std::strcmp(sp.key, "pb2")) out.ival = 128;
        if (sp.key && !std::strcmp(sp.key, "from")) out.sval = "bf16";
        if (sp.key && !std::strcmp(sp.key, "to")) out.sval = "f32";
        if (sp.key && !std::strcmp(sp.key, "act")) out.sval = "silu";
        /* A constraint's own value wins; a C_IN list takes its first word. */
        for (int ci = 0; ci < k->n_constraints; ++ci) {
            const RadConstraint& x = k->constraints[ci];
            if (!x.key || !sp.key || std::strcmp(x.key, sp.key)) continue;
            if (x.op == RAD_C_EQ && sp.type == RAD_P_STR && x.sval) out.sval = x.sval;
            if ((x.op == RAD_C_EQ || x.op == RAD_C_GE) && sp.type == RAD_P_INT) out.ival = x.ival;
            if (x.op == RAD_C_IN && sp.type == RAD_P_STR && x.sval && x.sval[0]) {
                /* out.sval must outlive the loop, so the words live in c.words, reserved once so
                 * no reallocation moves them. */
                c.words.emplace_back(x.sval);
                std::string& w = c.words.back();
                const size_t sp0 = w.find(' ');
                if (sp0 != std::string::npos) w.resize(sp0);
                out.sval = w.c_str();
            }
        }
        c.p[c.n_p++] = out;
    }
}

/* ------------------------------------------------------------------ a row with no shape hook
 *
 * `gemm_q_shape` declines w2a8: the container stores scale and zero as two planes behind the one
 * `b_scale` operand, and a RadOpdDesc names one dtype and extent set. A byte-count-only hook would
 * fill a container header with random bytes. So the row's `opd_shape` is null and this builds its
 * case by hand at MiniCPM5-2B's band, with the joined plane built by rad_planes_relayout as
 * r3d_gemm_nt_q_w2a8_driver.cpp builds it. */
static int build_w2a8(Case& c, const RadOpSchema* s) {
#define W_PI(k, v) { (k), RAD_P_INT, (long long)(v), 0, nullptr, 0.0 }
#define W_PS(k, v) { (k), RAD_P_STR, 0, 0, (v), 0.0 }
    const long long M = 16, N = 130560, K = 2048, group = 128, G = K / group;
    if (s->n_operands != 7) { fail("w2a8: the schema has %d operands, not 7", s->n_operands); return 1; }
    c.n_p = 0;                     /* the model's band, not the constraint minimums */
    const RadParam pp[] = { W_PI("M", M), W_PI("N", N), W_PI("K", K), W_PI("group", group),
                            W_PS("dtype", "w2a8") };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    RadEncoding e;
    std::memset(&e, 0, sizeof e);
    rad_enc_copy_str(e.scheme, "affine");
    e.n_planes = 3;
    rad_enc_copy_str(e.plane[0].role, "codes");
    e.plane[0].dtype = RAD_U2; e.plane[0].kind = RAD_PLANE_TILED;
    e.plane[0].block[0] = 1; e.plane[0].block[1] = 1;
    rad_enc_copy_str(e.plane[1].role, "scale");
    e.plane[1].dtype = RAD_F16; e.plane[1].kind = RAD_PLANE_TILED;
    e.plane[1].block[0] = 1; e.plane[1].block[1] = group;
    rad_enc_copy_str(e.plane[2].role, "zero");
    e.plane[2].dtype = RAD_U8; e.plane[2].kind = RAD_PLANE_TILED;
    e.plane[2].block[0] = 1; e.plane[2].block[1] = group;
    const int sel[2] = { 1, 2 };

    std::vector<uint16_t> sc((size_t)N * G), ze((size_t)N * G);
    uint32_t st = 0x1234567u;
    for (size_t i = 0; i < sc.size(); ++i) {
        st = st * 1664525u + 1013904223u;
        sc[i] = f32_to_f16(0.001f + 0.002f * (float)((st >> 8) & 0xffffu) / 65536.0f);
        ze[i] = (uint8_t)((st >> 16) & 3u);
    }
    RadTensor ps[2] = {};      /* zero-initialised: rad_planes_plan reads every field */
    const long long pd[2][2] = { { N, G }, { N, G } };
    const uint32_t pdt[2] = { RAD_F16, RAD_U8 };
    void* pdat[2] = { sc.data(), ze.data() };
    for (int i = 0; i < 2; ++i) {
        ps[i].data = pdat[i]; ps[i].dtype = pdt[i]; ps[i].rank = 2;
        ps[i].shape[0] = pd[i][0]; ps[i].shape[1] = pd[i][1];
        ps[i].stride[1] = 1; ps[i].stride[0] = pd[i][1];
    }
    RadPlanesEntry en[RAD_ENC_MAX_PLANES];
    const long long jb = rad_planes_plan(&e, sel, ps, 2, en);
    if (jb <= 0) { fail("w2a8: rad_planes_plan refused the joined scale+zero"); return 1; }
    std::vector<uint8_t> jbytes((size_t)jb, 0);
    if (rad_planes_relayout(&e, sel, ps, 2, jbytes.data(), jb) != RAD_OK) {
        fail("w2a8: rad_planes_relayout refused"); return 1;
    }

    /* Schema order: a, a_scale, b, b_scale (the joined blob), y, a_sum, b_ref (absent: a null
     * tensor). y is pre-filled with a bf16 NaN, since a shim that returns RAD_OK without launching
     * would otherwise leave finite bytes behind. */
    struct Slot { long long rows, cols; uint32_t dt; int nan; } q[7] = {
        { M, K, RAD_I8, 0 }, { M, 1, RAD_F32, 0 }, { N, K, RAD_U2, 0 },
        { jb, 1, RAD_U8, 0 }, { M, N, RAD_BF16, 1 }, { G, 16, RAD_F32, 0 }, { 0, 0, RAD_F32, 0 } };
    for (int o = 0; o < 7; ++o) {
        if (o == 6) { c.t[o] = RadTensor{}; continue; }
        size_t bytes = o == 3 ? (size_t)jb
                     : q[o].dt == RAD_U2 ? (size_t)(q[o].rows * q[o].cols) / 4
                     : (size_t)(q[o].rows * q[o].cols) * dt_bytes(q[o].dt);
        std::vector<uint8_t> h(bytes, 0);
        if (o == 3) std::memcpy(h.data(), jbytes.data(), bytes);   /* the blob, not a random fill */
        else for (size_t i = 0; i < bytes; ++i) {
            st = st * 1664525u + 1013904223u;
            h[i] = (uint8_t)(st >> 13);
        }
        if (o == 1 || o == 5) {                              /* scales and sums: small and positive */
            float* f = (float*)h.data();
            const size_t n = bytes / 4;
            for (size_t i = 0; i < n; ++i) f[i] = 0.001f + 0.01f * (float)((st = st * 1664525u +
                1013904223u) >> 8 & 0xffffu) / 65536.0f;
        }
        if (q[o].nan) {
            uint16_t* b16 = (uint16_t*)h.data();
            for (size_t i = 0; i < bytes / 2; ++i) b16[i] = 0x7fc0u;
        }
        void* d = nullptr;
        if (o == 3) {
            /* The joined operand is pinned host memory, as the engine's mapped tier holds a joined
             * weight: the shim reads its header on the host and the kernel reads the same bytes.
             * Plain malloc is not device-addressable. */
            if (hipHostMalloc(&d, bytes, 0) != hipSuccess) { fail("w2a8: hipHostMalloc(%zu) failed", bytes); return 1; }
            c.dev.push_back(d);
            c.host.push_back(d);
            std::memcpy(d, h.data(), bytes);
        } else {
            if (hipMalloc(&d, bytes) != hipSuccess) { fail("w2a8: hipMalloc(%zu) failed", bytes); return 1; }
            c.dev.push_back(d);
            if (hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
                fail("w2a8: hipMemcpy in failed"); return 1;
            }
        }
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = q[o].dt;
        t.rank = (o == 3) ? 1 : 2;
        t.shape[0] = q[o].rows; t.shape[1] = q[o].cols;
        t.stride[1] = 1; t.stride[0] = q[o].cols;
        if (o == 3) { t.shape[0] = jb; t.shape[1] = 0; t.stride[0] = 1; }
    }
    return 0;
#undef W_PI
#undef W_PS
}

/* ------------------------------------------------------------------ the a16 band
 *
 * The 4-bit weight against an f16 activation. The coverage is the band a w4a16 checkpoint asks
 * for once one runs: the codes are read where the container stores them, and the (scale, zero)
 * pair reaches the row joined by rad_planes_relayout exactly as the loader will do. y is the
 * only output and it is pre-filled with a bf16 NaN, so a kernel that never launches leaves its
 * status RAD_OK behind and the NaN says the row did not run. */
static int build_w4a16asym(Case& c, const RadOpSchema* s) {
#define W_PI(k, v) { (k), RAD_P_INT, (long long)(v), 0, nullptr, 0.0 }
#define W_PS(k, v) { (k), RAD_P_STR, 0, 0, (v), 0.0 }
    const long long M = 16, N = 11008, K = 4096, group = 128, G = K / group;
    if (s->n_operands != 7) { fail("w4a16asym: the schema has %d operands, not 7", s->n_operands);
                              return 1; }
    c.n_p = 0;
    const RadParam pp[] = { W_PI("M", M), W_PI("N", N), W_PI("K", K), W_PI("group", group),
                            W_PS("dtype", "w4a16_asym") };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    RadEncoding e;
    std::memset(&e, 0, sizeof e);
    rad_enc_copy_str(e.scheme, "affine");
    e.n_planes = 3;
    rad_enc_copy_str(e.plane[0].role, "codes");
    e.plane[0].dtype = RAD_U4; e.plane[0].kind = RAD_PLANE_TILED;
    e.plane[0].block[0] = 1; e.plane[0].block[1] = 1;
    rad_enc_copy_str(e.plane[1].role, "scale");
    e.plane[1].dtype = RAD_F16; e.plane[1].kind = RAD_PLANE_TILED;
    e.plane[1].block[0] = 1; e.plane[1].block[1] = group;
    rad_enc_copy_str(e.plane[2].role, "zero");
    e.plane[2].dtype = RAD_U8; e.plane[2].kind = RAD_PLANE_TILED;
    e.plane[2].block[0] = 1; e.plane[2].block[1] = group;
    const int sel_sz[2] = { 1, 2 };

    /* The canonical planes, in the quantiser's own order: the codes the kernel reads where they
     * lie, the scale and zero the row joins. */
    std::vector<uint8_t> cd((size_t)N * K / 2);
    std::vector<uint16_t> sc((size_t)N * G);
    std::vector<uint8_t> ze((size_t)N * G);
    uint32_t st = 0x41A15339u;
    for (size_t i = 0; i < cd.size(); ++i) {
        st = st * 1664525u + 1013904223u;
        cd[i] = (uint8_t)((st & 0xF) | ((st >> 12) & 0xF0));
    }
    for (size_t i = 0; i < sc.size(); ++i) {
        st = st * 1664525u + 1013904223u;
        sc[i] = f32_to_f16(0.001f + 0.01f * (float)((st >> 8) & 0xffffu) / 65536.0f);
        ze[i] = (uint8_t)((st >> 16) & 0xFu);      /* a zero that fits a 4-bit code */
    }
    RadTensor ps[2] = {};
    const uint32_t pdt[2] = { RAD_F16, RAD_U8 };
    void* pdat[2] = { sc.data(), ze.data() };
    for (int i = 0; i < 2; ++i) {
        ps[i].data = pdat[i]; ps[i].dtype = pdt[i]; ps[i].rank = 2;
        ps[i].shape[0] = N; ps[i].shape[1] = G;
        ps[i].stride[1] = 1; ps[i].stride[0] = G;
    }
    RadPlanesEntry en[RAD_ENC_MAX_PLANES];
    const long long jb = rad_planes_plan(&e, sel_sz, ps, 2, en);
    if (jb <= 0) { fail("w4a16asym: rad_planes_plan refused the joined scale+zero"); return 1; }
    std::vector<uint8_t> jbytes((size_t)jb, 0);
    if (rad_planes_relayout(&e, sel_sz, ps, 2, jbytes.data(), jb) != RAD_OK) {
        fail("w4a16asym: rad_planes_relayout refused"); return 1;
    }

    /* Schema order: a, a_scale (absent), b (the codes, in order), b_scale (the joined pair),
     * y, a_sum (absent), b_ref (absent). */
    struct Slot { long long rows, cols; uint32_t dt; int nan; } q[7] = {
        { M, K, RAD_F16, 0 }, { 0, 0, RAD_F32, 0 }, { N, K, RAD_U4, 0 },
        { jb, 1, RAD_F16, 0 }, { M, N, RAD_BF16, 1 }, { 0, 0, RAD_F32, 0 },
        { 0, 0, RAD_F32, 0 } };
    for (int o = 0; o < 7; ++o) {
        if (!q[o].rows) { c.t[o] = RadTensor{}; continue; }
        size_t bytes = o == 2 ? (size_t)N * K / 2      /* codes count elements, not bytes */
                     : o == 3 ? (size_t)jb
                     : (size_t)q[o].rows * q[o].cols * dt_bytes(q[o].dt);
        std::vector<uint8_t> h(bytes, 0);
        if (o == 2) std::memcpy(h.data(), cd.data(), bytes);
        else if (o == 3) std::memcpy(h.data(), jbytes.data(), bytes);
        else if (o == 0) {
            uint16_t* hf = (uint16_t*)h.data();
            for (size_t i = 0; i < bytes / 2; ++i) {
                st = st * 1664525u + 1013904223u;
                hf[i] = f32_to_f16(((float)((st >> 8) & 0xffffu) / 65536.0f - 0.5f) * 4.0f);
            }
        }
        if (q[o].nan) {
            uint16_t* b16 = (uint16_t*)h.data();
            for (size_t i = 0; i < bytes / 2; ++i) b16[i] = 0x7fc0u;
        }
        /* The joined operand stays pinned, the mapped tier's shape: the shim reads its header
         * where it lies and the kernel reads the same bytes over PCIe. */
        void* d = nullptr;
        if (o == 3) {
            if (hipHostMalloc(&d, bytes, 0) != hipSuccess) {
                fail("w4a16asym: hipHostMalloc(%zu) failed", bytes); return 1; }
            c.dev.push_back(d); c.host.push_back(d);
            std::memcpy(d, h.data(), bytes);
        } else {
            if (hipMalloc(&d, bytes) != hipSuccess) { fail("w4a16asym: hipMalloc(%zu) failed", bytes);
                                                      return 1; }
            c.dev.push_back(d);
            if (hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
                fail("w4a16asym: hipMemcpy in failed"); return 1; }
        }
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = q[o].dt;
        t.rank = (o == 3) ? 1 : 2;
        t.shape[0] = q[o].rows; t.shape[1] = q[o].cols;
        t.stride[1] = 1; t.stride[0] = q[o].cols;
        if (o == 3) { t.shape[1] = 0; t.stride[0] = 1; }
    }
    return 0;
#undef W_PI
#undef W_PS
}

/* The 5-bit row: the 4-bit case with the fifth-bit plane, [N/16][K/2] u32. */
static int build_w5a16(Case& c, const RadOpSchema* s) {
#define W_PI(k, v) { (k), RAD_P_INT, (long long)(v), 0, nullptr, 0.0 }
#define W_PS(k, v) { (k), RAD_P_STR, 0, 0, (v), 0.0 }
    const long long M = 16, N = 11008, K = 4096, group = 128, G = K / group;
    if (s->n_operands != 7) { fail("w5a16: the schema has %d operands, not 7", s->n_operands);
                              return 1; }
    c.n_p = 0;
    const RadParam pp[] = { W_PI("M", M), W_PI("N", N), W_PI("K", K), W_PI("group", group),
                            W_PS("dtype", "w5a16") };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    RadEncoding e;
    std::memset(&e, 0, sizeof e);
    rad_enc_copy_str(e.scheme, "affine");
    e.n_planes = 4;
    rad_enc_copy_str(e.plane[0].role, "codes");
    e.plane[0].dtype = RAD_U4; e.plane[0].kind = RAD_PLANE_TILED;
    e.plane[0].block[0] = 1; e.plane[0].block[1] = 1;
    rad_enc_copy_str(e.plane[1].role, "scale");
    e.plane[1].dtype = RAD_F16; e.plane[1].kind = RAD_PLANE_TILED;
    e.plane[1].block[0] = 1; e.plane[1].block[1] = group;
    rad_enc_copy_str(e.plane[2].role, "zero");
    e.plane[2].dtype = RAD_U8; e.plane[2].kind = RAD_PLANE_TILED;
    e.plane[2].block[0] = 1; e.plane[2].block[1] = group;
    rad_enc_copy_str(e.plane[3].role, "hi");
    e.plane[3].dtype = RAD_U32; e.plane[3].kind = RAD_PLANE_TILED;
    e.plane[3].block[0] = 1; e.plane[3].block[1] = group / 2;
    const int sel_sz[3] = { 1, 2, 3 };

    std::vector<uint8_t> cd((size_t)N * K / 2);
    std::vector<uint16_t> sc((size_t)N * G);
    std::vector<uint8_t> ze((size_t)N * G);
    std::vector<uint32_t> hi((size_t)(N / 16) * (K / 2));
    uint32_t st = 0x5A11CE57u;
    for (size_t i = 0; i < cd.size(); ++i) {
        st = st * 1664525u + 1013904223u;
        cd[i] = (uint8_t)((st & 0xF) | ((st >> 12) & 0xF0));
    }
    for (size_t i = 0; i < sc.size(); ++i) {
        st = st * 1664525u + 1013904223u;
        sc[i] = f32_to_f16(0.001f + 0.01f * (float)((st >> 8) & 0xffffu) / 65536.0f);
        ze[i] = (uint8_t)((st >> 16) & 0x1Fu);     /* a zero that fits a 5-bit code */
    }
    for (size_t i = 0; i < hi.size(); ++i) {
        st = st * 1664525u + 1013904223u;
        hi[i] = st;
    }
    RadTensor ps[3] = {};
    const uint32_t pdt[3] = { RAD_F16, RAD_U8, RAD_U32 };
    void* pdat[3] = { sc.data(), ze.data(), hi.data() };
    for (int i = 0; i < 3; ++i) {
        ps[i].data = pdat[i]; ps[i].dtype = pdt[i]; ps[i].rank = 2;
        ps[i].shape[0] = (i == 2) ? N / 16 : N;
        ps[i].shape[1] = (i == 2) ? K / 2 : G;
        ps[i].stride[1] = 1; ps[i].stride[0] = ps[i].shape[1];
    }
    RadPlanesEntry en[RAD_ENC_MAX_PLANES];
    const long long jb = rad_planes_plan(&e, sel_sz, ps, 3, en);
    if (jb <= 0) { fail("w5a16: rad_planes_plan refused the joined scale+zero+hi"); return 1; }
    std::vector<uint8_t> jbytes((size_t)jb, 0);
    if (rad_planes_relayout(&e, sel_sz, ps, 3, jbytes.data(), jb) != RAD_OK) {
        fail("w5a16: rad_planes_relayout refused"); return 1;
    }

    struct Slot { long long rows, cols; uint32_t dt; int nan; } q[7] = {
        { M, K, RAD_F16, 0 }, { 0, 0, RAD_F32, 0 }, { N, K, RAD_U4, 0 },
        { jb, 1, RAD_F16, 0 }, { M, N, RAD_BF16, 1 }, { 0, 0, RAD_F32, 0 },
        { 0, 0, RAD_F32, 0 } };
    for (int o = 0; o < 7; ++o) {
        if (!q[o].rows) { c.t[o] = RadTensor{}; continue; }
        size_t bytes = o == 2 ? (size_t)N * K / 2
                      : o == 3 ? (size_t)jb
                      : (size_t)q[o].rows * q[o].cols * dt_bytes(q[o].dt);
        std::vector<uint8_t> h(bytes, 0);
        if (o == 2) std::memcpy(h.data(), cd.data(), bytes);
        else if (o == 3) std::memcpy(h.data(), jbytes.data(), bytes);
        else if (o == 0) {
            uint16_t* hf = (uint16_t*)h.data();
            for (size_t i = 0; i < bytes / 2; ++i) {
                st = st * 1664525u + 1013904223u;
                hf[i] = f32_to_f16(((float)((st >> 8) & 0xffffu) / 65536.0f - 0.5f) * 4.0f);
            }
        }
        if (q[o].nan) {
            uint16_t* b16 = (uint16_t*)h.data();
            for (size_t i = 0; i < bytes / 2; ++i) b16[i] = 0x7fc0u;
        }
        void* d = nullptr;
        if (o == 3) {
            if (hipHostMalloc(&d, bytes, 0) != hipSuccess) {
                fail("w5a16: hipHostMalloc(%zu) failed", bytes); return 1; }
            c.dev.push_back(d); c.host.push_back(d);
            std::memcpy(d, h.data(), bytes);
        } else {
            if (hipMalloc(&d, bytes) != hipSuccess) { fail("w5a16: hipMalloc(%zu) failed", bytes);
                                                     return 1; }
            c.dev.push_back(d);
            if (hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
                fail("w5a16: hipMemcpy in failed"); return 1; }
        }
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = q[o].dt;
        t.rank = (o == 3) ? 1 : 2;
        t.shape[0] = q[o].rows; t.shape[1] = q[o].cols;
        t.stride[1] = 1; t.stride[0] = q[o].cols;
        if (o == 3) { t.shape[1] = 0; t.stride[0] = 1; }
    }
    return 0;
#undef W_PI
#undef W_PS
}

/* The gated-delta-net preamble: two sequences of 65 tokens each (two chunks apiece, the second
 * partial), four k heads and two v heads each, head_k 128, chunk 64. The gate decreases within a
 * chunk as the kernel assumes; the output starts as NaN, so a row the solve does not write is a
 * failure. */
static int build_gdn_kkt(Case& c, const RadOpSchema* s) {
#define GK_PI(k, v) { (k), RAD_P_INT, (long long)(v), 0, nullptr, 0.0 }
    const long long M = 130, Hg = 4, H = 8, K = 128, C = 64;
    if (s->n_operands != 5) { fail("gdn_kkt: the schema has %d operands, not 5", s->n_operands);
                              return 1; }
    c.n_p = 0;
    const RadParam pp[] = { GK_PI("M", M), GK_PI("head_k", K), GK_PI("chunk", C) };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x5A11CE57u;
    std::vector<uint16_t> hk((size_t)M * Hg * K);
    for (auto& v : hk) {
        st = st * 1664525u + 1013904223u;
        v = f32_to_bf16(((float)((st >> 8) & 0xffffu) / 65536.0f - 0.5f) * 2.0f);
    }
    std::vector<float> bt((size_t)M * H), gt((size_t)M * H);
    for (size_t m = 0; m < M; ++m)
        for (size_t h = 0; h < H; ++h) {
            st = st * 1664525u + 1013904223u;
            bt[m * H + h] = (float)((st >> 8) & 0xffffu) / 65536.0f;   // 0..1
            gt[m * H + h] = -0.01f * (int)(m % 64);                   // decreasing in the chunk
        }
    const int cuv[3] = { 0, 65, (int)M };
    std::vector<uint16_t> ha((size_t)M * H * C, 0x7fc0u);             // NaN: an unwritten row shows

    struct { uint32_t dt; long long r0, r1; } q[5] = {
        { RAD_BF16, M, Hg * K }, { RAD_F32, M, H }, { RAD_F32, M, H },
        { RAD_I32, 3, 1 }, { RAD_BF16, M, H * C } };
    for (int o = 0; o < 5; ++o) {
        const size_t bytes = (size_t)q[o].r0 * q[o].r1 * dt_bytes(q[o].dt);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("gdn_kkt: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        if (o == 0) std::memcpy(d, hk.data(), bytes);
        else if (o == 1) std::memcpy(d, bt.data(), bytes);
        else if (o == 2) std::memcpy(d, gt.data(), bytes);
        else if (o == 3) std::memcpy(d, cuv, sizeof cuv);
        else std::memcpy(d, ha.data(), bytes);
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = q[o].dt;
        t.rank = (o == 3) ? 1 : (o == 0) ? 3 : 2;
        if (o == 3) { t.shape[0] = 3; t.stride[0] = 1; }
        else if (o == 0) { t.shape[0] = M; t.shape[1] = Hg; t.shape[2] = K;
                           t.stride[2] = 1; t.stride[1] = K; t.stride[0] = Hg * K; }
        else { t.shape[0] = q[o].r0; t.shape[1] = q[o].r1;
               t.stride[1] = 1; t.stride[0] = q[o].r1; }
    }
    return 0;
#undef GK_PI
}

/* The chunked scan: the same geometry as the kkt case, with v and the state cache. The state
 * starts random (an h0 the scan must carry through), ht must come back finite, and the output
 * starts as NaN so an unwritten row is a failure. No state slot index: the cache is row
 * indexed. */
static int build_gdn_chunk_scan(Case& c, const RadOpSchema* s) {
#define GC_PI(k, v) { (k), RAD_P_INT, (long long)(v), 0, nullptr, 0.0 }
    const long long M = 130, Hg = 4, H = 8, K = 128, V = 128, C = 64;
    if (s->n_operands != 11) { fail("gdn_chunk_scan: the schema has %d operands, not 11",
                                     s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { GC_PI("M", M), GC_PI("head_k", K), GC_PI("head_v", V),
                            GC_PI("chunk", C) };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x51B2CE01u;
    auto rnd16 = [&]() { st = st * 1664525u + 1013904223u; return (st >> 8) & 0xffffu; };
    auto rndf  = [&]() { st = st * 1664525u + 1013904223u;
                         return (float)((st >> 8) & 0xffffu) / 65536.0f; };

    std::vector<uint16_t> hq((size_t)M * Hg * K), hk((size_t)M * Hg * K),
                          hv((size_t)M * H * V), hA((size_t)M * H * C);
    for (auto& x : hq) x = f32_to_bf16((rndf() - 0.5f) * 0.5f);
    for (auto& x : hk) x = f32_to_bf16((rndf() - 0.5f) * 0.5f);
    for (auto& x : hv) x = f32_to_bf16((rndf() - 0.5f) * 0.5f);
    // A: the kkt_solve output's shape -- a unit lower triangular with a small random fill.
    for (size_t m = 0; m < M; ++m)
        for (size_t h = 0; h < H; ++h)
            for (int j = 0; j < C; ++j) {
                const int i = m % C;
                hA[m * H * C + h * C + j] =
                    (j == i) ? f32_to_bf16(1.0f)
                    : (j < i) ? f32_to_bf16((rndf() - 0.5f) * 0.1f)
                              : 0x0000u;
            }
    std::vector<float> gt((size_t)M * H), btt((size_t)M * H);
    for (size_t m = 0; m < M; ++m)
        for (size_t h = 0; h < H; ++h) {
            btt[m * H + h] = rndf();                          // 0..1
            gt[m * H + h] = -0.01f * (int)(m % 64);           // decreasing in the chunk
        }
    std::vector<float> h0((size_t)M * H * V * K);
    for (auto& x : h0) x = (rndf() - 0.5f) * 0.1f;
    const int cuv[3] = { 0, 65, (int)M };
    std::vector<uint16_t> ho((size_t)M * H * V, 0x7fc0u);     // NaN: an unwritten row shows

    struct { uint32_t dt; long long r0, r1, r2; } qd[11] = {
        { RAD_BF16, M, Hg * K, K },   // q
        { RAD_BF16, M, Hg * K, K },   // k
        { RAD_BF16, M, H * V, V },    // v
        { RAD_BF16, M, H * C, C },    // A
        { RAD_F32, M, H, 1 },         // g
        { RAD_F32, M, H, 1 },         // beta
        { RAD_F32, M * H, V * K, V * K },  // h0 [M, H, V*K] dense
        { RAD_I32, 3, 1, 1 },         // cu
        { RAD_BF16, M, H * V, V },    // o
        { RAD_F32, M * H, V * K, V * K },  // ht
        { RAD_I32, 0, 0, 0 }          // state_idx: absent
    };
    for (int o = 0; o < 10; ++o) {
        const long long nel = (o == 6 || o == 9) ? M * H * V * K
                         : (o == 7) ? 3
                         : qd[o].r0 * qd[o].r1 / qd[o].r2 * qd[o].r2;
        const size_t bytes = (size_t)nel * dt_bytes(qd[o].dt);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("gdn_chunk_scan: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        if (o == 0) std::memcpy(d, hq.data(), bytes);
        else if (o == 1) std::memcpy(d, hk.data(), bytes);
        else if (o == 2) std::memcpy(d, hv.data(), bytes);
        else if (o == 3) std::memcpy(d, hA.data(), bytes);
        else if (o == 4) std::memcpy(d, gt.data(), bytes);
        else if (o == 5) std::memcpy(d, btt.data(), bytes);
        else if (o == 6) std::memcpy(d, h0.data(), bytes);
        else if (o == 7) std::memcpy(d, cuv, sizeof cuv);
        else if (o == 8) std::memcpy(d, ho.data(), bytes);
        else std::memcpy(d, h0.data(), bytes);
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt;
        if (o == 7) { t.rank = 1; t.shape[0] = 3; t.stride[0] = 1; }
        else if (o == 6 || o == 9) {
            t.rank = 3; t.shape[0] = M; t.shape[1] = H; t.shape[2] = V * K;
            t.stride[2] = 1; t.stride[1] = V * K; t.stride[0] = H * V * K;
        } else if (o == 0 || o == 1 || o == 2 || o == 3 || o == 8) {
            t.rank = 3; t.shape[0] = M; t.shape[1] = qd[o].r1 / qd[o].r2; t.shape[2] = qd[o].r2;
            t.stride[2] = 1; t.stride[1] = qd[o].r2; t.stride[0] = qd[o].r1;
        } else {
            t.rank = 2; t.shape[0] = M; t.shape[1] = H;
            t.stride[1] = 1; t.stride[0] = H;
        }
    }
    c.t[10] = RadTensor{};   // state_idx absent
    return 0;
#undef GC_PI
}

static int build_gdn_gated_rmsnorm(Case& c, const RadOpSchema* s) {
#define GR_PI(k, v) { (k), RAD_P_INT, (long long)(v), 0, nullptr, 0.0 }
    const long long M = 130, H = 10, C = 128;
    if (s->n_operands != 4) { fail("gdn_gated_rmsnorm: the schema has %d operands, not 4",
                                    s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { GR_PI("M", M), GR_PI("channels", C) };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x9E3779B9u;
    auto rndf = [&]() { st = st * 1664525u + 1013904223u;
                        return (float)((st >> 8) & 0xffffu) / 65536.0f; };

    std::vector<uint16_t> hx((size_t)M * C), hz((size_t)M * C);
    for (auto& x : hx) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    for (auto& x : hz) x = f32_to_bf16((rndf() - 0.5f) * 4.0f);
    std::vector<float> hw(C);
    for (auto& x : hw) x = (rndf() - 0.5f) * 2.0f;
    std::vector<uint16_t> ho((size_t)M * C, 0x7fc0u);   // NaN: an unwritten row shows

    struct { uint32_t dt; long long r0, r1, r2; } qd[4] = {
        { RAD_BF16, M, C, 1 },   // x [M, C]
        { RAD_BF16, M, C, 1 },   // z [M, C]
        { RAD_F32,  C, 1, 1 },   // w [C]
        { RAD_BF16, M, C, 1 }    // o [M, C]
    };
    for (int o = 0; o < 4; ++o) {
        const long long nel = qd[o].r0 * qd[o].r1;
        const size_t bytes = (size_t)nel * dt_bytes(qd[o].dt);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("gdn_gated_rmsnorm: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        if (o == 0) std::memcpy(d, hx.data(), bytes);
        else if (o == 1) std::memcpy(d, hz.data(), bytes);
        else if (o == 2) std::memcpy(d, hw.data(), bytes);
        else std::memcpy(d, ho.data(), bytes);
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt;
        if (o == 2) { t.rank = 1; t.shape[0] = C; t.stride[0] = 1; }
        else { t.rank = 2; t.shape[0] = M; t.shape[1] = C; t.stride[1] = 1; t.stride[0] = C; }
    }
    return 0;
#undef GR_PI
}

static int build_gdn_gated_norm_hq(Case& c, const RadOpSchema* s) {
#define GH_PI(k, v) { (k), RAD_P_INT, (long long)(v), 0, nullptr, 0.0 }
    const long long M = 64, H = 8, C = 128, K = H * C;
    if (s->n_operands != 5) { fail("gdn_gated_norm_had_quant_i8: the schema has %d operands, not 5",
                                    s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { GH_PI("M", M), GH_PI("n", K), GH_PI("head_dim", C),
                            GH_PI("group", 128) };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0xC2B2AE35u;
    auto rndf = [&]() { st = st * 1664525u + 1013904223u;
                        return (float)((st >> 8) & 0xffffu) / 65536.0f; };

    std::vector<uint16_t> hx((size_t)M * K), hz((size_t)M * K);
    for (auto& x : hx) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    for (auto& x : hz) x = f32_to_bf16((rndf() - 0.5f) * 4.0f);
    std::vector<float> hw(C);
    for (auto& x : hw) x = (rndf() - 0.5f) * 2.0f;
    std::vector<int8_t> hq((size_t)M * K, 0);
    std::vector<float> hs(M, 0.0f);

    struct { uint32_t dt; long long r0, r1; } qd[5] = {
        { RAD_BF16, M, K },   // x [M, K]
        { RAD_BF16, M, K },   // z [M, K]
        { RAD_F32,  C, 1 },   // w [C]
        { RAD_I8,   M, K },   // q [M, K]
        { RAD_F32,  M, 1 }    // scale [M]
    };
    for (int o = 0; o < 5; ++o) {
        const long long nel = qd[o].r0 * qd[o].r1;
        const size_t bytes = (size_t)nel * dt_bytes(qd[o].dt);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("gdn_gated_norm_hq: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        if (o == 0) std::memcpy(d, hx.data(), bytes);
        else if (o == 1) std::memcpy(d, hz.data(), bytes);
        else if (o == 2) std::memcpy(d, hw.data(), bytes);
        else if (o == 3) std::memcpy(d, hq.data(), bytes);
        else std::memcpy(d, hs.data(), bytes);
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt;
        if (o == 2) { t.rank = 1; t.shape[0] = C; t.stride[0] = 1; }
        else if (o == 4) { t.rank = 1; t.shape[0] = M; t.stride[0] = 1; }
        else { t.rank = 2; t.shape[0] = M; t.shape[1] = K; t.stride[1] = 1; t.stride[0] = K; }
    }
    return 0;
#undef GH_PI
}

static int build_gdn_conv_prep(Case& c, const RadOpSchema* s) {
    const long long T = 64, Hg = 2, H = 4, K = 128, V = 128, W = 4;
    const long long units = 2 * Hg + H, XD = units * K;   // 8 * 128 = 1024
    if (s->n_operands != 16) { fail("gdn_conv_prep: the schema has %d operands, not 16",
                                     s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "M", RAD_P_INT, T, 0, nullptr, 0.0 },
                            { "head_k", RAD_P_INT, K, 0, nullptr, 0.0 },
                            { "head_v", RAD_P_INT, V, 0, nullptr, 0.0 },
                            { "chunk", RAD_P_INT, 64, 0, nullptr, 0.0 },
                            { "conv_width", RAD_P_INT, W, 0, nullptr, 0.0 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x7E57BD71u;
    auto rndf = [&]() { st = st * 1664525u + 1013904223u;
                        return (float)((st >> 8) & 0xffffu) / 65536.0f; };

    std::vector<uint16_t> hx(T * XD), hw(XD * W), hb(XD);
    for (auto& x : hx) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    for (auto& x : hw) x = f32_to_bf16((rndf() - 0.5f) * 0.5f);
    for (auto& x : hb) x = f32_to_bf16((rndf() - 0.5f) * 0.2f);
    std::vector<uint16_t> hcs(XD * 3, 0), ha(T * H), hbg(T * H);
    for (auto& x : hcs) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    for (auto& x : ha)  x = f32_to_bf16((rndf() - 0.5f) * 4.0f);
    for (auto& x : hbg) x = f32_to_bf16((rndf() - 0.5f) * 4.0f);
    std::vector<float> halog(H), hdtb(H);
    for (auto& x : halog) x = logf(0.1f + rndf());
    for (auto& x : hdtb)  x = (rndf() - 0.5f) * 4.0f;
    const int cuv[2] = { 0, (int)T };
    const int cidxv[1] = { 0 }, hin[1] = { 1 };
    std::vector<uint16_t> hq(T * Hg * K, 0x7fc0u), hk(T * Hg * K, 0x7fc0u), hv(T * H * V, 0x7fc0u);
    std::vector<float> hg(T * H, 0.0f), hbeta(T * H, 0.0f);

    struct { uint32_t dt; long long nel; } qd[16] = {
        { RAD_BF16, T * XD },   // x
        { RAD_BF16, XD * W },   // w
        { RAD_BF16, XD },       // b
        { RAD_F32,  H },        // A_log
        { RAD_F32,  H },        // dt_bias
        { RAD_BF16, XD * 3 },   // conv_state [XD, 1, 3]
        { RAD_I32,  2 },        // cu
        { RAD_BF16, T * H },    // a
        { RAD_BF16, T * H },    // b_gate
        { RAD_I32,  1 },        // cache_idx
        { RAD_I32,  1 },        // has_init
        { RAD_BF16, T * Hg * K },  // q
        { RAD_BF16, T * Hg * K },  // k
        { RAD_BF16, T * H * V },   // v
        { RAD_F32,  T * H },       // g
        { RAD_F32,  T * H }        // beta
    };
    for (int o = 0; o < 16; ++o) {
        const size_t bytes = (size_t)qd[o].nel * dt_bytes(qd[o].dt);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("gdn_conv_prep: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        switch (o) {
            case 0: std::memcpy(d, hx.data(), bytes); break;
            case 1: std::memcpy(d, hw.data(), bytes); break;
            case 2: std::memcpy(d, hb.data(), bytes); break;
            case 3: std::memcpy(d, halog.data(), bytes); break;
            case 4: std::memcpy(d, hdtb.data(), bytes); break;
            case 5: std::memcpy(d, hcs.data(), bytes); break;
            case 6: std::memcpy(d, cuv, sizeof cuv); break;
            case 7: std::memcpy(d, ha.data(), bytes); break;
            case 8: std::memcpy(d, hbg.data(), bytes); break;
            case 9: std::memcpy(d, cidxv, sizeof cidxv); break;
            case 10: std::memcpy(d, hin, sizeof hin); break;
            case 11: std::memcpy(d, hq.data(), bytes); break;
            case 12: std::memcpy(d, hk.data(), bytes); break;
            case 13: std::memcpy(d, hv.data(), bytes); break;
            case 14: std::memcpy(d, hg.data(), bytes); break;
            default: std::memcpy(d, hbeta.data(), bytes); break;
        }
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt;
        switch (o) {
            case 0: t.rank = 2; t.shape[0] = T; t.shape[1] = XD; t.stride[1] = 1; t.stride[0] = XD; break;
            case 1: t.rank = 2; t.shape[0] = XD; t.shape[1] = W; t.stride[1] = 1; t.stride[0] = W; break;
            case 2: t.rank = 1; t.shape[0] = XD; t.stride[0] = 1; break;
            case 3: case 4: t.rank = 1; t.shape[0] = H; t.stride[0] = 1; break;
            case 5: t.rank = 3; t.shape[0] = XD; t.shape[1] = 1; t.shape[2] = 3;
                    t.stride[2] = 1; t.stride[1] = 3; t.stride[0] = 3; break;
            case 6: t.rank = 1; t.shape[0] = 2; t.stride[0] = 1; break;
            case 7: case 8: t.rank = 2; t.shape[0] = T; t.shape[1] = H; t.stride[1] = 1; t.stride[0] = H; break;
            case 9: case 10: t.rank = 1; t.shape[0] = 1; t.stride[0] = 1; break;
            case 11: case 12: t.rank = 3; t.shape[0] = T; t.shape[1] = Hg; t.shape[2] = K;
                    t.stride[2] = 1; t.stride[1] = K; t.stride[0] = Hg * K; break;
            case 13: t.rank = 3; t.shape[0] = T; t.shape[1] = H; t.shape[2] = V;
                    t.stride[2] = 1; t.stride[1] = V; t.stride[0] = H * V; break;
            case 14: case 15: t.rank = 2; t.shape[0] = T; t.shape[1] = H; t.stride[1] = 1; t.stride[0] = H; break;
        }
    }
    return 0;
}

static int build_gdn_conv_update(Case& c, const RadOpSchema* s) {
    const long long T = 1, Hg = 2, H = 4, K = 128, V = 128, W = 4, SL = 4;
    const long long units = 2 * Hg + H, XD = units * K;
    if (s->n_operands != 10) { fail("gdn_conv_update: the schema has %d operands, not 10",
                                     s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "q_len", RAD_P_INT, T, 0, nullptr, 0.0 },
                            { "head_k", RAD_P_INT, K, 0, nullptr, 0.0 },
                            { "head_v", RAD_P_INT, V, 0, nullptr, 0.0 },
                            { "conv_width", RAD_P_INT, W, 0, nullptr, 0.0 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x35A7F12Bu;
    auto rndf = [&]() { st = st * 1664525u + 1013904223u;
                        return (float)((st >> 8) & 0xffffu) / 65536.0f; };

    std::vector<uint16_t> hx(T * XD), hw(XD * W), hb(XD);
    for (auto& x : hx) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    for (auto& x : hw) x = f32_to_bf16((rndf() - 0.5f) * 0.5f);
    for (auto& x : hb) x = f32_to_bf16((rndf() - 0.5f) * 0.2f);
    std::vector<uint16_t> hcs(XD * SL, 0);
    for (auto& x : hcs) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    const int cuv[2] = { 0, (int)T };
    const int sidxv[1] = { 0 }, nav[1] = { 0 };
    std::vector<uint16_t> hq(T * Hg * K, 0x7fc0u), hk(T * Hg * K, 0x7fc0u), hv(T * H * V, 0x7fc0u);

    struct { uint32_t dt; long long nel; } qd[10] = {
        { RAD_BF16, T * XD },   // x
        { RAD_BF16, XD * W },   // w
        { RAD_BF16, XD },       // b
        { RAD_BF16, XD * SL },  // conv_state [XD, 1, SL]
        { RAD_I32,  1 },        // state_idx
        { RAD_I32,  1 },        // num_accepted
        { RAD_I32,  2 },        // cu
        { RAD_BF16, T * Hg * K },  // q
        { RAD_BF16, T * Hg * K },  // k
        { RAD_BF16, T * H * V }    // v
    };
    for (int o = 0; o < 10; ++o) {
        const size_t bytes = (size_t)qd[o].nel * dt_bytes(qd[o].dt);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("gdn_conv_update: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        switch (o) {
            case 0: std::memcpy(d, hx.data(), bytes); break;
            case 1: std::memcpy(d, hw.data(), bytes); break;
            case 2: std::memcpy(d, hb.data(), bytes); break;
            case 3: std::memcpy(d, hcs.data(), bytes); break;
            case 4: std::memcpy(d, sidxv, sizeof sidxv); break;
            case 5: std::memcpy(d, nav, sizeof nav); break;
            case 6: std::memcpy(d, cuv, sizeof cuv); break;
            case 7: std::memcpy(d, hq.data(), bytes); break;
            case 8: std::memcpy(d, hk.data(), bytes); break;
            default: std::memcpy(d, hv.data(), bytes); break;
        }
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt;
        switch (o) {
            case 0: t.rank = 2; t.shape[0] = T; t.shape[1] = XD; t.stride[1] = 1; t.stride[0] = XD; break;
            case 1: t.rank = 2; t.shape[0] = XD; t.shape[1] = W; t.stride[1] = 1; t.stride[0] = W; break;
            case 2: t.rank = 1; t.shape[0] = XD; t.stride[0] = 1; break;
            case 3: t.rank = 3; t.shape[0] = XD; t.shape[1] = 1; t.shape[2] = SL;
                    t.stride[2] = 1; t.stride[1] = SL; t.stride[0] = SL; break;
            case 4: case 5: t.rank = 1; t.shape[0] = 1; t.stride[0] = 1; break;
            case 6: t.rank = 1; t.shape[0] = 2; t.stride[0] = 1; break;
            case 7: case 8: t.rank = 3; t.shape[0] = T; t.shape[1] = Hg; t.shape[2] = K;
                    t.stride[2] = 1; t.stride[1] = K; t.stride[0] = Hg * K; break;
            default: t.rank = 3; t.shape[0] = T; t.shape[1] = H; t.shape[2] = V;
                     t.stride[2] = 1; t.stride[1] = V; t.stride[0] = H * V; break;
        }
    }
    return 0;
}

/* causal_conv1d at LFM2's band: two sequences in one prefill chunk (cu 0, 11, 19), width 3 over
 * 2048 channels, each sequence on its own state slot with a history to read (has_init 1), and no
 * num_accepted, so every token commits and the window is rewritten. `cu` and the slots are
 * offsets the kernel follows, so they are built here rather than drawn. */
static int build_causal_conv1d(Case& c, const RadOpSchema* s) {
    const long long T = 19, N = 2, n = 2048, W = 3, SL = W - 1, SLOTS = 2;
    if (s->n_operands != 8) { fail("causal_conv1d: the schema has %d operands, not 8",
                                   s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "M", RAD_P_INT, T, 0, nullptr, 0.0 },
                            { "n", RAD_P_INT, n, 0, nullptr, 0.0 },
                            { "width", RAD_P_INT, W, 0, nullptr, 0.0 },
                            { "dtype", RAD_P_STR, 0, 0, "bf16", 0.0 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x2C0F1D35u;
    auto rndf = [&]() { st = st * 1664525u + 1013904223u;
                        return (float)((st >> 8) & 0xffffu) / 65536.0f; };
    std::vector<uint16_t> hx(T * n), hw(n * W), hcs(SLOTS * n * SL), hy(T * n, 0x7fc0u);
    for (auto& x : hx) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    for (auto& x : hw) x = f32_to_bf16((rndf() - 0.5f) * 0.5f);
    for (auto& x : hcs) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    const int cuv[N + 1] = { 0, 11, (int)T };
    const int sidxv[N] = { 1, 0 }, hiv[N] = { 1, 1 };

    struct { uint32_t dt; long long nel; const void* src; } qd[8] = {
        { RAD_BF16, T * n, hx.data() },              // x
        { RAD_BF16, n * W, hw.data() },              // w
        { RAD_BF16, SLOTS * n * SL, hcs.data() },    // conv_state [SLOTS, n, SL]
        { RAD_I32,  N + 1, cuv },                    // cu
        { RAD_I32,  N, sidxv },                      // cache_idx
        { RAD_I32,  N, hiv },                        // has_init
        { RAD_I32,  0, nullptr },                    // num_accepted, absent: a prefill
        { RAD_BF16, T * n, hy.data() }               // y, NaN until written
    };
    for (int o = 0; o < 8; ++o) {
        RadTensor& t = c.t[o];
        t = RadTensor{};
        if (!qd[o].nel) continue;
        const size_t bytes = (size_t)qd[o].nel * dt_bytes(qd[o].dt);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("causal_conv1d: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        if (hipMemcpy(d, qd[o].src, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            fail("causal_conv1d: hipMemcpy in failed"); return 1;
        }
        t.data = d; t.dtype = qd[o].dt;
        switch (o) {
            case 0: case 7: t.rank = 2; t.shape[0] = T; t.shape[1] = n;
                            t.stride[1] = 1; t.stride[0] = n; break;
            case 1: t.rank = 2; t.shape[0] = n; t.shape[1] = W; t.stride[1] = 1; t.stride[0] = W; break;
            case 2: t.rank = 3; t.shape[0] = SLOTS; t.shape[1] = n; t.shape[2] = SL;
                    t.stride[2] = 1; t.stride[1] = SL; t.stride[0] = n * SL; break;
            default: t.rank = 1; t.shape[0] = qd[o].nel; t.stride[0] = 1; break;
        }
    }
    return 0;
}

static int build_expert_scale(Case& c, const RadOpSchema* s) {
    const long long M = 3, K = 4, NE = 8;
    if (s->n_operands != 3) { fail("expert_scale: the schema has %d operands, not 3",
                                    s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "M", RAD_P_INT, M, 0, nullptr, 0.0 },
                            { "n_expert", RAD_P_INT, NE, 0, nullptr, 0.0 },
                            { "top_k", RAD_P_INT, K, 0, nullptr, 0.0 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0xA11CEu;
    auto rndf = [&]() { st = st * 1664525u + 1013904223u;
                        return (float)((st >> 8) & 0xffffu) / 65536.0f; };

    std::vector<int32_t> hids(M * K);
    for (auto& x : hids) x = (int32_t)(rndf() * (NE + 2)) - 1;   // includes an out-of-table id
    std::vector<uint16_t> hsc(NE), hew(M * K);
    for (auto& x : hsc) x = f32_to_bf16(0.1f + rndf());
    for (auto& x : hew) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);

    struct { uint32_t dt; long long nel; } qd[3] = {
        { RAD_I32,  M * K },   // ids
        { RAD_BF16, NE },      // w_e
        { RAD_BF16, M * K }    // ew
    };
    for (int o = 0; o < 3; ++o) {
        const size_t bytes = (size_t)qd[o].nel * dt_bytes(qd[o].dt);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("expert_scale: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        switch (o) {
            case 0: std::memcpy(d, hids.data(), bytes); break;
            case 1: std::memcpy(d, hsc.data(), bytes); break;
            default: std::memcpy(d, hew.data(), bytes); break;
        }
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt;
        if (o == 1) { t.rank = 1; t.shape[0] = NE; t.stride[0] = 1; }
        else { t.rank = 2; t.shape[0] = M; t.shape[1] = K; t.stride[1] = 1; t.stride[0] = K; }
    }
    return 0;
}

static int build_moe_router_topk(Case& c, const RadOpSchema* s) {
    const long long M = 6, NE = 48, TK = 8;
    if (s->n_operands != 3) { fail("router_topk: the schema has %d operands, not 3",
                                    s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "M", RAD_P_INT, M, 0, nullptr, 0.0 },
                            { "n_expert", RAD_P_INT, NE, 0, nullptr, 0.0 },
                            { "top_k", RAD_P_INT, TK, 0, nullptr, 0.0 },
                            { "norm", RAD_P_INT, 1, 0, nullptr, 0.0 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x7017u;
    auto rndf = [&]() { st = st * 1664525u + 1013904223u;
                        return (float)((st >> 8) & 0xffffu) / 65536.0f; };
    std::vector<uint16_t> hlg(M * NE), hw(M * TK, 0);
    for (auto& x : hlg) x = f32_to_bf16((rndf() - 0.5f) * 4.0f);

    struct { uint32_t dt; long long nel; } qd[3] = {
        { RAD_BF16, M * NE },   // logits
        { RAD_I32,  M * TK },   // expert_ids
        { RAD_BF16, M * TK }    // expert_w
    };
    for (int o = 0; o < 3; ++o) {
        const size_t bytes = (size_t)qd[o].nel * dt_bytes(qd[o].dt);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("router_topk: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        std::memcpy(d, o == 0 ? hlg.data() : hw.data(), bytes);
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt;
        if (o == 0) { t.rank = 2; t.shape[0] = M; t.shape[1] = NE; t.stride[1] = 1; t.stride[0] = NE; }
        else { t.rank = 2; t.shape[0] = M; t.shape[1] = TK; t.stride[1] = 1; t.stride[0] = TK; }
    }
    return 0;
}

static int build_moe_scatter(Case& c, const RadOpSchema* s) {
    const long long M = 16, TK = 8, NE = 32;   // T = 128
    const long long T = M * TK;
    if (s->n_operands != 4) { fail("moe_scatter: the schema has %d operands, not 4",
                                    s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "M", RAD_P_INT, M, 0, nullptr, 0.0 },
                            { "n_expert", RAD_P_INT, NE, 0, nullptr, 0.0 },
                            { "top_k", RAD_P_INT, TK, 0, nullptr, 0.0 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x5CA7u;
    auto rnd = [&]() { st = st * 1664525u + 1013904223u; return (int32_t)((st >> 8) & 0xffffu); };
    std::vector<int32_t> hids(T), hsorted(T, -1), hoff(NE + 1, -1), hcnt(NE, -1);
    for (auto& x : hids) x = rnd() % NE;

    struct { uint32_t dt; long long nel; } qd[4] = {
        { RAD_I32, T },        // expert_ids
        { RAD_I32, T },        // sorted_tok
        { RAD_I32, NE + 1 },   // expert_offset
        { RAD_I32, NE }        // expert_count
    };
    for (int o = 0; o < 4; ++o) {
        const size_t bytes = (size_t)qd[o].nel * 4;
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("moe_scatter: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        std::memcpy(d, o == 0 ? hids.data() : (o == 1 ? hsorted.data()
                      : (o == 2 ? hoff.data() : hcnt.data())), bytes);
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = RAD_I32; t.rank = 1;
        t.shape[0] = qd[o].nel; t.stride[0] = 1;
    }
    return 0;
}

/* moe_gemm_q at fp8a8f32 (the e4m3 row's dtype): E4M3 activation and experts, an f32 scale per
 * (row, 128 of K) and per 128x128 tile of each expert, the even/odd weight tables stacked, y bf16. */
static int build_moe_gemm_q(Case& c, const RadOpSchema* s) {
    constexpr int NOPD = 9;
    const long long M = 8, N = 128, K = 128, NE = 8, TK = 2;   // T = 16; N % 128
    const long long T = M * TK, NE0 = (NE + 1) / 2, NE1 = NE / 2, NB = (N + 127) / 128;
    if (s->n_operands != NOPD + 1) { fail("moe_gemm_q: the schema has %d operands, not 10",
                                          s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "M", RAD_P_INT, T, 0, nullptr, 0.0 },
                            { "N", RAD_P_INT, N, 0, nullptr, 0.0 },
                            { "K", RAD_P_INT, K, 0, nullptr, 0.0 },
                            { "n_expert", RAD_P_INT, NE, 0, nullptr, 0.0 },
                            { "top_k", RAD_P_INT, TK, 0, nullptr, 0.0 },
                            { "group", RAD_P_INT, 128, 0, nullptr, 0.0 },
                            { "a_order", RAD_P_STR, 0, 0, (const char*)"token", 0.0 },
                            { "dtype", RAD_P_STR, 0, 0, (const char*)"fp8a8f32", 0.0 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x0E0E5A1Cu;
    auto rnd = [&]() { st = st * 1664525u + 1013904223u; return (st >> 8) & 0xffu; };
    auto rndf = [&]() { st = st * 1664525u + 1013904223u; return 0.5f + ((st >> 8) & 0xff) / 255.0f; };
    // E4M3 codes in the normal range (exp 1..14), a valid bit pattern.
    auto re4 = [&]() { return (uint8_t)(((rnd() & 0x7) | 1) << 3 | (rnd() & 0x7) | ((rnd() & 1) << 7)); };
    std::vector<uint8_t> ha((size_t)M * K), hw0((size_t)NE0 * N * K), hw1((size_t)NE1 * N * K);
    for (auto& x : ha) x = re4();
    for (auto& x : hw0) x = re4();
    for (auto& x : hw1) x = re4();
    std::vector<float> hA((size_t)M * (K / 128)), hW0((size_t)NE0 * NB * (K / 128)),
        hW1((size_t)NE1 * NB * (K / 128));
    for (auto& x : hA) x = rndf();
    for (auto& x : hW0) x = rndf();
    for (auto& x : hW1) x = rndf();
    std::vector<int32_t> hs(T), ho(NE + 1);
    for (long long i = 0; i < T; ++i) hs[i] = (int32_t)i;
    for (long long e = 0; e < NE; ++e) ho[e] = (int32_t)(e * (T / NE));
    ho[NE] = (int32_t)T;

    // The contract's order: a, a_scale, w, w_scale, sorted_tok, expert_offset, w_odd, w_odd_scale, y.
    struct { uint32_t dt; int rank; long long sh[3]; long long nel; } qd[NOPD] = {
        { RAD_F8E4M3, 2, { M, K, 1 }, M * K },
        { RAD_F32, 2, { M, K / 128, 1 }, M * (K / 128) },
        { RAD_F8E4M3, 3, { NE0, N, K }, NE0 * N * K },
        { RAD_F32, 3, { NE0, NB, K / 128 }, NE0 * NB * (K / 128) },
        { RAD_I32, 1, { T, 1, 1 }, T },
        { RAD_I32, 1, { NE + 1, 1, 1 }, NE + 1 },
        { RAD_F8E4M3, 3, { NE1, N, K }, NE1 * N * K },
        { RAD_F32, 3, { NE1, NB, K / 128 }, NE1 * NB * (K / 128) },
        { RAD_BF16, 2, { T, N, 1 }, T * N } };
    const uint32_t bits_of[NOPD] = { 8, 32, 8, 32, 32, 32, 8, 32, 16 };
    const void* src[NOPD] = { ha.data(), hA.data(), hw0.data(), hW0.data(), hs.data(), ho.data(),
                              hw1.data(), hW1.data(), nullptr };
    for (int o = 0; o < NOPD; ++o) {
        const size_t bytes = (size_t)((qd[o].nel * bits_of[o] + 7) / 8);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("moe_gemm_q: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        if (src[o]) std::memcpy(d, src[o], bytes);
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt; t.rank = qd[o].rank;
        for (int r = 0; r < qd[o].rank; ++r) { t.shape[r] = qd[o].sh[r]; t.stride[r] = 1; }
        rad_tensor_pack(&t);   /* dense row-major: the ascending form computes stride[0] before stride[1] */
    }
    c.t[NOPD] = RadTensor{};   // split absent
    return 0;
}

static int build_moe_gemm_bf16(Case& c, const RadOpSchema* s) {
    constexpr int NOPD = 5;
    const long long M = 8, N = 128, K = 64, NE = 8, TK = 2;   // T = 16; N % 128, K % 64
    const long long T = M * TK;
    if (s->n_operands != NOPD) { fail("moe_gemm_bf16: the schema has %d operands, not 5",
                                      s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "M", RAD_P_INT, T, 0, nullptr, 0.0 },
                            { "N", RAD_P_INT, N, 0, nullptr, 0.0 },
                            { "K", RAD_P_INT, K, 0, nullptr, 0.0 },
                            { "n_expert", RAD_P_INT, NE, 0, nullptr, 0.0 },
                            { "top_k", RAD_P_INT, TK, 0, nullptr, 0.0 },
                            { "a_order", RAD_P_STR, 0, 0, (const char*)"token", 0.0 },
                            { "dtype", RAD_P_STR, 0, 0, (const char*)"bf16", 0.0 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x1A2B3C4Du;
    auto rndf = [&]() { st = st * 1664525u + 1013904223u;
                        return (float)((st >> 8) & 0xffffu) / 65536.0f; };
    std::vector<uint16_t> ha((size_t)M * K), hw((size_t)NE * N * K);
    for (auto& x : ha) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    for (auto& x : hw) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    std::vector<int32_t> hs(T), ho(NE + 1);
    for (long long i = 0; i < T; ++i) hs[i] = (int32_t)i;
    for (long long e = 0; e < NE; ++e) ho[e] = (int32_t)(e * (T / NE));
    ho[NE] = (int32_t)T;

    // The contract's order: a, w (one stacked table), sorted_tok, expert_offset, y.
    struct { uint32_t dt; int rank; long long sh[3]; long long nel; } qd[NOPD] = {
        { RAD_BF16, 2, { M, K, 1 }, M * K },
        { RAD_BF16, 3, { NE, N, K }, NE * N * K },
        { RAD_I32, 1, { T, 1, 1 }, T },
        { RAD_I32, 1, { NE + 1, 1, 1 }, NE + 1 },
        { RAD_BF16, 2, { T, N, 1 }, T * N } };
    const uint32_t bits_of[NOPD] = { 16, 16, 32, 32, 16 };
    const void* src[NOPD] = { ha.data(), hw.data(), hs.data(), ho.data(), nullptr };
    for (int o = 0; o < NOPD; ++o) {
        const size_t bytes = (size_t)((qd[o].nel * bits_of[o] + 7) / 8);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("moe_gemm_bf16: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        if (src[o]) std::memcpy(d, src[o], bytes);
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt; t.rank = qd[o].rank;
        for (int r = 0; r < qd[o].rank; ++r) { t.shape[r] = qd[o].sh[r]; t.stride[r] = 1; }
        rad_tensor_pack(&t);   /* dense row-major: the ascending form computes stride[0] before stride[1] */
    }
    return 0;
}

/* moe_gemm_q at mxfp4a8: E4M3 activation with an f32 scale per (row, 128 of K); E2M1 experts (two
 * codes a byte, shaped in codes) with an E8M0 exponent per (row, 32 of K), typed raw u8; y bf16. */
static int build_moe_gemm_mxfp4a8(Case& c, const RadOpSchema* s) {
    constexpr int NOPD = 9;
    const long long M = 8, N = 128, K = 128, NE = 8, TK = 2;   // T = 16; N % 128
    const long long T = M * TK, NE0 = (NE + 1) / 2, NE1 = NE / 2;
    if (s->n_operands != NOPD + 1) { fail("moe_gemm_mxfp4a8: the schema has %d operands, not 10",
                                          s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "M", RAD_P_INT, T, 0, nullptr, 0.0 },
                            { "N", RAD_P_INT, N, 0, nullptr, 0.0 },
                            { "K", RAD_P_INT, K, 0, nullptr, 0.0 },
                            { "n_expert", RAD_P_INT, NE, 0, nullptr, 0.0 },
                            { "top_k", RAD_P_INT, TK, 0, nullptr, 0.0 },
                            { "group", RAD_P_INT, 128, 0, nullptr, 0.0 },
                            { "a_order", RAD_P_STR, 0, 0, (const char*)"token", 0.0 },
                            { "dtype", RAD_P_STR, 0, 0, (const char*)"mxfp4a8", 0.0 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x7A6B5C4Du;
    auto rnd = [&]() { st = st * 1664525u + 1013904223u; return (st >> 8) & 0xffu; };
    auto rndf = [&]() { st = st * 1664525u + 1013904223u; return 0.5f + ((st >> 8) & 0xff) / 255.0f; };
    auto re4 = [&]() { return (uint8_t)(((rnd() & 0x7) | 1) << 3 | (rnd() & 0x7) | ((rnd() & 1) << 7)); };
    auto re2x2 = [&]() { return (uint8_t)(rnd() & 0xFF); };   // two E2M1 codes
    auto re8 = [&]() { return (uint8_t)(120 + (rnd() & 14)); };  // E8M0 near 1.0 (exp 120..133)
    std::vector<uint8_t> ha((size_t)M * K), hw0((size_t)NE0 * N * (K / 2)), hw1((size_t)NE1 * N * (K / 2));
    for (auto& x : ha) x = re4();
    for (auto& x : hw0) x = re2x2();
    for (auto& x : hw1) x = re2x2();
    std::vector<float> hA((size_t)M * (K / 128));
    std::vector<uint8_t> hW0((size_t)NE0 * N * (K / 32)), hW1((size_t)NE1 * N * (K / 32));
    for (auto& x : hA) x = rndf();
    for (auto& x : hW0) x = re8();
    for (auto& x : hW1) x = re8();
    std::vector<int32_t> hs(T), ho(NE + 1);
    for (long long i = 0; i < T; ++i) hs[i] = (int32_t)i;
    for (long long e = 0; e < NE; ++e) ho[e] = (int32_t)(e * (T / NE));
    ho[NE] = (int32_t)T;

    // The contract's order: a, a_scale, w, w_scale, sorted_tok, expert_offset, w_odd, w_odd_scale, y.
    struct { uint32_t dt; int rank; long long sh[3]; long long nel; } qd[NOPD] = {
        { RAD_F8E4M3, 2, { M, K, 1 }, M * K },
        { RAD_F32, 2, { M, K / 128, 1 }, M * (K / 128) },
        { RAD_FP4E2M1, 3, { NE0, N, K }, NE0 * N * K },
        { RAD_U8, 3, { NE0, N, K / 32 }, NE0 * N * (K / 32) },
        { RAD_I32, 1, { T, 1, 1 }, T },
        { RAD_I32, 1, { NE + 1, 1, 1 }, NE + 1 },
        { RAD_FP4E2M1, 3, { NE1, N, K }, NE1 * N * K },
        { RAD_U8, 3, { NE1, N, K / 32 }, NE1 * N * (K / 32) },
        { RAD_BF16, 2, { T, N, 1 }, T * N } };
    const uint32_t bits_of[NOPD] = { 8, 32, 4, 8, 32, 32, 4, 8, 16 };
    const void* src[NOPD] = { ha.data(), hA.data(), hw0.data(), hW0.data(), hs.data(), ho.data(),
                              hw1.data(), hW1.data(), nullptr };
    for (int o = 0; o < NOPD; ++o) {
        const size_t bytes = (size_t)((qd[o].nel * bits_of[o] + 7) / 8);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("moe_gemm_mxfp4a8: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        if (src[o]) std::memcpy(d, src[o], bytes);
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt; t.rank = qd[o].rank;
        for (int r = 0; r < qd[o].rank; ++r) { t.shape[r] = qd[o].sh[r]; t.stride[r] = 1; }
        rad_tensor_pack(&t);   /* dense row-major: the ascending form computes stride[0] before stride[1] */
    }
    c.t[NOPD] = RadTensor{};   // split absent
    return 0;
}

/* moe_gemm_q dtype w4a8: an int8 activation (an f32 scale per (row, 128 of K)) and two's-complement
 * int4 experts (a bf16 scale per (row, 128 of K)), two stacked weight tables by parity (the even in
 * `w` / `w_scale`, the odd in `w_odd` / `w_odd_scale`), token-order a, y bf16. Every sorted row is
 * owned (no dropped slots), so the kernel writes all of y; the NaN prefill catches a launch that
 * skips rows. */
static int build_moe_gemm_w4a8(Case& c, const RadOpSchema* s) {
    constexpr int NOPD = 9;
    const long long M = 6, N = 96, K = 256, NE = 4, TK = 4;   // T = 24, six rows per expert
    const long long T = M * TK;
    const long long ne0 = (NE + 1) / 2, ne1 = NE / 2;
    if (s->n_operands != NOPD + 1) { fail("moe_gemm_w4a8: the schema has %d operands, not 10",
                                           s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "M", RAD_P_INT, M, 0, nullptr, 0.0 },
                            { "N", RAD_P_INT, N, 0, nullptr, 0.0 },
                            { "K", RAD_P_INT, K, 0, nullptr, 0.0 },
                            { "n_expert", RAD_P_INT, NE, 0, nullptr, 0.0 },
                            { "top_k", RAD_P_INT, TK, 0, nullptr, 0.0 },
                            { "group", RAD_P_INT, 128, 0, nullptr, 0.0 },
                            { "a_order", RAD_P_STR, 0, 0, (const char*)"token", 0.0 },
                            { "dtype", RAD_P_STR, 0, 0, (const char*)"w4a8", 0.0 },
                            { "N_odd", RAD_P_INT, N, 0, nullptr, 0.0 },
                            { "K_odd", RAD_P_INT, K, 0, nullptr, 0.0 },
                            { "parts", RAD_P_INT, 1, 0, nullptr, 0.0 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x74A8C0Deu;
    auto rnd = [&]() { st = st * 1664525u + 1013904223u; return (st >> 8) & 0xffu; };
    auto rndf = [&]() { st = st * 1664525u + 1013904223u; return 0.5f + ((st >> 8) & 0xff) / 255.0f; };
    auto to_bf16 = [](float f) { return (uint16_t)(__builtin_bit_cast(uint32_t, f) >> 16); };
    std::vector<int8_t> ha((size_t)M * K);
    for (auto& x : ha) x = (int8_t)((rnd() & 7) - 4);                 // -4..4
    std::vector<float> hA((size_t)M * (K / 128));
    for (auto& x : hA) x = rndf();
    // The weight codes and bf16 scales for all NE experts, then split by parity.
    std::vector<uint8_t> hw((size_t)NE * N * (K / 2));
    for (auto& x : hw) x = (uint8_t)(rnd() & 0xFFu);
    std::vector<uint16_t> hsce((size_t)NE * N * (K / 128));
    for (auto& x : hsce) x = to_bf16(rndf());
    std::vector<int32_t> hs(T), ho(NE + 1);
    for (long long i = 0; i < T; ++i) hs[i] = (int32_t)i;
    for (long long e = 0; e < NE; ++e) ho[e] = (int32_t)(e * (T / NE));
    ho[NE] = (int32_t)T;
    std::vector<uint16_t> hy((size_t)T * N);
    for (auto& x : hy) x = 0x7fc0u;   // bf16 NaN prefill
    std::vector<uint8_t> hwe((size_t)ne0 * N * (K / 2)), hwo((size_t)ne1 * N * (K / 2));
    std::vector<uint16_t> hse((size_t)ne0 * N * (K / 128)), hso((size_t)ne1 * N * (K / 128));
    for (long long e = 0; e < NE; ++e) {
        const bool odd = e & 1;
        const long long ix = e >> 1;
        std::memcpy((odd ? hwo.data() : hwe.data()) + (size_t)ix * N * (K / 2),
                    hw.data() + (size_t)e * N * (K / 2), (size_t)N * (K / 2));
        std::memcpy((odd ? hso.data() : hse.data()) + (size_t)ix * N * (K / 128),
                    hsce.data() + (size_t)e * N * (K / 128), (size_t)N * (K / 128) * sizeof(uint16_t));
    }

    // The contract's order: a, a_scale, w, w_scale, sorted_tok, expert_offset, w_odd, w_odd_scale, y.
    struct { uint32_t dt; int rank; long long sh[3]; long long nel; } qd[NOPD] = {
        { RAD_I8, 2, { M, K, 1 }, M * K },
        { RAD_F32, 2, { M, K / 128, 1 }, M * (K / 128) },
        { RAD_I4, 3, { ne0, N, K }, ne0 * N * K },
        { RAD_BF16, 3, { ne0, N, K / 128 }, ne0 * N * (K / 128) },
        { RAD_I32, 1, { T, 1, 1 }, T },
        { RAD_I32, 1, { NE + 1, 1, 1 }, NE + 1 },
        { RAD_I4, 3, { ne1, N, K }, ne1 * N * K },
        { RAD_BF16, 3, { ne1, N, K / 128 }, ne1 * N * (K / 128) },
        { RAD_BF16, 2, { T, N, 1 }, T * N } };
    const uint32_t bits_of[NOPD] = { 8, 32, 4, 16, 32, 32, 4, 16, 16 };
    const void* src[NOPD] = { ha.data(), hA.data(), hwe.data(), hse.data(), hs.data(), ho.data(),
                              hwo.data(), hso.data(), hy.data() };
    for (int o = 0; o < NOPD; ++o) {
        const size_t bytes = (size_t)((qd[o].nel * bits_of[o] + 7) / 8);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("moe_gemm_w4a8: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        if (src[o]) std::memcpy(d, src[o], bytes);
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt; t.rank = qd[o].rank;
        for (int r = 0; r < qd[o].rank; ++r) { t.shape[r] = qd[o].sh[r]; t.stride[r] = 1; }
        rad_tensor_pack(&t);
    }
    c.t[NOPD] = RadTensor{};   // split absent
    return 0;
}

static int build_moe_router_scatter(Case& c, const RadOpSchema* s) {
    const long long M = 6, NE = 48, TK = 8;
    const long long T = M * TK;
    if (s->n_operands != 6) { fail("router_topk_scatter: the schema has %d operands, not 6",
                                    s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "M", RAD_P_INT, M, 0, nullptr, 0.0 },
                            { "n_expert", RAD_P_INT, NE, 0, nullptr, 0.0 },
                            { "top_k", RAD_P_INT, TK, 0, nullptr, 0.0 },
                            { "n_local", RAD_P_INT, NE, 0, nullptr, 0.0 },
                            { "norm", RAD_P_INT, 1, 0, nullptr, 0.0 },
                            { "dtype", RAD_P_STR, 0, 0, (const char*)"bf16", 0.0 },
                            { "expert_base", RAD_P_INT, 0, 0, nullptr, 0.0 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x8070u;
    auto rndf = [&]() { st = st * 1664525u + 1013904223u;
                        return (float)((st >> 8) & 0xffffu) / 65536.0f; };
    std::vector<uint16_t> hlg(M * NE);
    for (auto& x : hlg) x = f32_to_bf16((rndf() - 0.5f) * 4.0f);

    struct { uint32_t dt; long long nel; } qd[6] = {
        { RAD_BF16, M * NE },   // logits
        { RAD_I32,  M * TK },   // expert_ids
        { RAD_BF16, M * TK },   // expert_w
        { RAD_I32,  T },        // sorted_tok
        { RAD_I32,  NE + 1 },   // expert_offset
        { RAD_I32,  NE }        // expert_count
    };
    for (int o = 0; o < 6; ++o) {
        const size_t bytes = (size_t)qd[o].nel * dt_bytes(qd[o].dt);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("router_topk_scatter: hipMalloc(%zu) failed",
                                                       bytes); return 1; }
        c.dev.push_back(d);
        if (o == 0) std::memcpy(d, hlg.data(), bytes);
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt;
        if (o <= 2) { t.rank = 2; t.shape[0] = M; t.shape[1] = o == 0 ? NE : TK; t.stride[1] = 1;
                      t.stride[0] = o == 0 ? NE : TK; }
        else { t.rank = 1; t.shape[0] = qd[o].nel; t.stride[0] = 1; }
    }
    return 0;
}

static int build_moe_router_sigmoid(Case& c, const RadOpSchema* s) {
    const long long M = 6, NE = 48, TK = 8;
    if (s->n_operands != 4) { fail("router_sigmoid_topk: the schema has %d operands, not 4",
                                   s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "M", RAD_P_INT, M, 0, nullptr, 0.0 },
                            { "n_expert", RAD_P_INT, NE, 0, nullptr, 0.0 },
                            { "top_k", RAD_P_INT, TK, 0, nullptr, 0.0 },
                            { "norm", RAD_P_INT, 1, 0, nullptr, 0.0 },
                            { "route_scale", RAD_P_F64, 0, 0, nullptr, 1.0 },
                            { "dtype", RAD_P_STR, 0, 0, (const char*)"bf16", 0.0 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x51A2u;
    auto rndf = [&]() { st = st * 1664525u + 1013904223u;
                        return (float)((st >> 8) & 0xffffu) / 65536.0f; };
    std::vector<uint16_t> hlg(M * NE);
    std::vector<float> hbi(NE);
    for (auto& x : hlg) x = f32_to_bf16((rndf() - 0.5f) * 4.0f);
    for (auto& x : hbi) x = (rndf() - 0.5f) * 0.1f;

    struct { uint32_t dt; long long nel; } qd[4] = {
        { RAD_BF16, M * NE },   // logits
        { RAD_F32,  NE },       // bias
        { RAD_I32,  M * TK },   // expert_ids
        { RAD_BF16, M * TK }    // expert_w
    };
    const void* src[] = { hlg.data(), hbi.data(), nullptr, nullptr };
    for (int o = 0; o < 4; ++o) {
        const size_t bytes = (size_t)qd[o].nel * dt_bytes(qd[o].dt);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("router_sigmoid_topk: hipMalloc(%zu) failed",
                                                       bytes); return 1; }
        c.dev.push_back(d);
        if (src[o]) std::memcpy(d, src[o], bytes);
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt;
        if (o == 0) { t.rank = 2; t.shape[0] = M; t.shape[1] = NE; t.stride[1] = 1; t.stride[0] = NE; }
        else if (o == 1) { t.rank = 1; t.shape[0] = NE; t.stride[0] = 1; }
        else { t.rank = 2; t.shape[0] = M; t.shape[1] = TK; t.stride[1] = 1; t.stride[0] = TK; }
    }
    return 0;
}


static int build_moe_gather(Case& c, const RadOpSchema* s, int inv) {
    const long long M = 8, n = 128, TK = 2;   // T = 16
    const long long T = M * TK;
    if (s->n_operands != 6) { fail("moe_gather%s: the schema has %d operands, not 6",
                                   inv ? "_inv" : "", s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "M", RAD_P_INT, M, 0, nullptr, 0.0 },
                            { "n", RAD_P_INT, n, 0, nullptr, 0.0 },
                            { "top_k", RAD_P_INT, TK, 0, nullptr, 0.0 },
                            { "dtype", RAD_P_STR, 0, 0, (const char*)"bf16", 0.0 },
                            { "act", RAD_P_STR, 0, 0, (const char*)"silu", 0.0 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x0B0C0Du;
    auto rndf = [&]() { st = st * 1664525u + 1013904223u;
                        return (float)((st >> 8) & 0xffffu) / 65536.0f; };
    std::vector<uint16_t> hye((size_t)T * n), hew(M * TK), hsh((size_t)M * n), hsg(M);
    for (auto& x : hye) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    for (auto& x : hew) x = f32_to_bf16(rndf());
    for (auto& x : hsh) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    for (auto& x : hsg) x = f32_to_bf16(rndf() * 4.0f);
    std::vector<int32_t> hslot(T);
    for (long long i = 0; i < T; ++i) hslot[i] = (int32_t)(i % M);

    struct { uint32_t dt; int rank; long long sh[2]; long long nel; } qd[6] = {
        { RAD_BF16, 2, { T, n }, T * n },   // y_expert
        { RAD_BF16, 2, { M, TK }, M * TK }, // expert_w
        { RAD_I32,  1, { T, 1 }, T },       // sorted_tok / inverse
        { RAD_BF16, 2, { M, n }, M * n },   // shared
        { RAD_BF16, 1, { M, 1 }, M },       // shared_gate
        { RAD_BF16, 2, { M, n }, M * n }    // y
    };
    const uint32_t bytes_of[] = { 2, 2, 4, 2, 2, 2 };
    const void* src[] = { hye.data(), hew.data(), hslot.data(), hsh.data(), hsg.data(), nullptr };
    for (int o = 0; o < 6; ++o) {
        const size_t bytes = (size_t)qd[o].nel * bytes_of[o];
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("moe_gather: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        if (src[o]) std::memcpy(d, src[o], bytes);
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt; t.rank = qd[o].rank;
        for (int r = 0; r < qd[o].rank; ++r) { t.shape[r] = qd[o].sh[r]; t.stride[r] = 1; }
        rad_tensor_pack(&t);   /* dense row-major: the ascending form computes stride[0] before stride[1] */
    }
    return 0;
}


static int build_gdn_recurrent(Case& c, const RadOpSchema* s) {
    const long long T = 4, Hg = 2, H = 4, K = 128, V = 128, SLOTS = 2;
    if (s->n_operands != 16) { fail("gdn_recurrent_update: the schema has %d operands, not 16",
                                     s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "q_len", RAD_P_INT, T, 0, nullptr, 0.0 },
                            { "head_k", RAD_P_INT, K, 0, nullptr, 0.0 },
                            { "head_v", RAD_P_INT, V, 0, nullptr, 0.0 },
                            { "eps", RAD_P_F64, 0, 0, nullptr, 1e-6 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x4A7B2C9Du;
    auto rndf = [&]() { st = st * 1664525u + 1013904223u;
                        return (float)((st >> 8) & 0xffffu) / 65536.0f; };

    std::vector<uint16_t> hq(T*Hg*K), hk(T*Hg*K), hv(T*H*V), ha(T*H), hb(T*H), hz(T*H*V);
    for (auto& x : hq) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    for (auto& x : hk) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    for (auto& x : hv) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    for (auto& x : ha) x = f32_to_bf16((rndf() - 0.5f) * 4.0f);
    for (auto& x : hb) x = f32_to_bf16((rndf() - 0.5f) * 4.0f);
    for (auto& x : hz) x = f32_to_bf16((rndf() - 0.5f) * 4.0f);
    std::vector<float> halog(H), hdtb(H), hstate(SLOTS*H*V*K), hnw(V);
    for (auto& x : halog) x = logf(0.1f + rndf());
    for (auto& x : hdtb)  x = (rndf() - 0.5f) * 4.0f;
    for (auto& x : hstate) x = (rndf() - 0.5f) * 0.1f;
    for (auto& x : hnw)    x = (rndf() - 0.5f) * 2.0f;
    std::vector<uint16_t> ho(T*H*V, 0x7fc0u);
    const int sidxv[T] = { 0, 0, 1, 1 };
    /* num_accepted counts from 1 (docs/OPS.md): the window's history is read at column
     * num_accepted - 1, so 1 is the handover out of a prefill. */
    const int nav[1] = { 1 };
    const int cuv[2] = { 0, (int)T };

    struct { uint32_t dt; long long nel; } qd[16] = {
        { RAD_BF16, T*Hg*K },   // q
        { RAD_BF16, T*Hg*K },   // k
        { RAD_BF16, T*H*V },    // v
        { RAD_BF16, T*H },      // a
        { RAD_BF16, T*H },      // b
        { RAD_F32,  H },        // A_log
        { RAD_F32,  H },        // dt_bias
        { RAD_F32,  SLOTS*H*V*K },  // state
        { RAD_I32,  T },        // state_idx [1, T]
        { RAD_I32,  1 },        // num_accepted
        { RAD_I32,  2 },        // cu
        { RAD_BF16, T*H*V },    // z
        { RAD_F32,  V },        // norm_w
        { RAD_BF16, T*H*V },    // o
        { RAD_I8,   0 },        // o_q: absent
        { RAD_F32,  0 }         // o_scale: absent
    };
    for (int o = 0; o < 14; ++o) {
        const size_t bytes = (size_t)qd[o].nel * dt_bytes(qd[o].dt);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("gdn_recurrent: hipMalloc(%zu) failed", bytes);
                                                  return 1; }
        c.dev.push_back(d);
        switch (o) {
            case 0: std::memcpy(d, hq.data(), bytes); break;
            case 1: std::memcpy(d, hk.data(), bytes); break;
            case 2: std::memcpy(d, hv.data(), bytes); break;
            case 3: std::memcpy(d, ha.data(), bytes); break;
            case 4: std::memcpy(d, hb.data(), bytes); break;
            case 5: std::memcpy(d, halog.data(), bytes); break;
            case 6: std::memcpy(d, hdtb.data(), bytes); break;
            case 7: std::memcpy(d, hstate.data(), bytes); break;
            case 8: std::memcpy(d, sidxv, sizeof sidxv); break;
            case 9: std::memcpy(d, nav, sizeof nav); break;
            case 10: std::memcpy(d, cuv, sizeof cuv); break;
            case 11: std::memcpy(d, hz.data(), bytes); break;
            case 12: std::memcpy(d, hnw.data(), bytes); break;
            default: std::memcpy(d, ho.data(), bytes); break;
        }
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt;
        switch (o) {
            case 0: case 1: t.rank = 3; t.shape[0]=T; t.shape[1]=Hg; t.shape[2]=K;
                    t.stride[2]=1; t.stride[1]=K; t.stride[0]=Hg*K; break;
            case 2: t.rank = 3; t.shape[0]=T; t.shape[1]=H; t.shape[2]=V;
                    t.stride[2]=1; t.stride[1]=V; t.stride[0]=H*V; break;
            case 3: case 4: t.rank = 2; t.shape[0]=T; t.shape[1]=H; t.stride[1]=1; t.stride[0]=H; break;
            case 5: case 6: t.rank = 1; t.shape[0]=H; t.stride[0]=1; break;
            case 7: t.rank = 4; t.shape[0]=SLOTS; t.shape[1]=H; t.shape[2]=V; t.shape[3]=K;
                    t.stride[3]=1; t.stride[2]=K; t.stride[1]=V*K; t.stride[0]=H*V*K; break;
            case 8: t.rank = 2; t.shape[0]=1; t.shape[1]=T; t.stride[1]=1; t.stride[0]=T; break;
            case 9: t.rank = 1; t.shape[0]=1; t.stride[0]=1; break;
            case 10: t.rank = 1; t.shape[0]=2; t.stride[0]=1; break;
            case 11: t.rank = 3; t.shape[0]=T; t.shape[1]=H; t.shape[2]=V;
                    t.stride[2]=1; t.stride[1]=V; t.stride[0]=H*V; break;
            case 12: t.rank = 1; t.shape[0]=V; t.stride[0]=1; break;
            default: t.rank = 3; t.shape[0]=T; t.shape[1]=H; t.shape[2]=V;
                     t.stride[2]=1; t.stride[1]=V; t.stride[0]=H*V; break;
        }
    }
    c.t[14] = RadTensor{};   // o_q absent
    c.t[15] = RadTensor{};   // o_scale absent
    return 0;
}

/* gdn_conv_recurrent_update: the conv's inputs (as build_gdn_conv_update) then the recurrence's
 * minus q, k and v (as build_gdn_recurrent), one geometry across both halves. */
static int build_gdn_conv_recurrent(Case& c, const RadOpSchema* s) {
    const long long T = 4, Hg = 2, H = 4, K = 128, V = 128, W = 4, SL = 4, SLOTS = 2;
    const long long units = 2 * Hg + H, XD = units * K;
    if (s->n_operands != 18) { fail("gdn_conv_recurrent_update: the schema has %d operands, not 18",
                                     s->n_operands); return 1; }
    c.n_p = 0;
    const RadParam pp[] = { { "q_len", RAD_P_INT, T, 0, nullptr, 0.0 },
                            { "head_k", RAD_P_INT, K, 0, nullptr, 0.0 },
                            { "head_v", RAD_P_INT, V, 0, nullptr, 0.0 },
                            { "conv_width", RAD_P_INT, W, 0, nullptr, 0.0 },
                            { "eps", RAD_P_F64, 0, 0, nullptr, 1e-6 } };
    for (const RadParam& q : pp) c.p[c.n_p++] = q;

    uint32_t st = 0x5C1D8E3Fu;
    auto rndf = [&]() { st = st * 1664525u + 1013904223u;
                        return (float)((st >> 8) & 0xffffu) / 65536.0f; };

    std::vector<uint16_t> hx(T * XD), hw(XD * W), hb(XD);
    for (auto& x : hx) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    for (auto& x : hw) x = f32_to_bf16((rndf() - 0.5f) * 0.5f);
    for (auto& x : hb) x = f32_to_bf16((rndf() - 0.5f) * 0.2f);
    std::vector<uint16_t> hcs(XD * SL, 0);
    for (auto& x : hcs) x = f32_to_bf16((rndf() - 0.5f) * 2.0f);
    std::vector<uint16_t> ha(T * H), hbb(T * H), hz(T * H * V);
    for (auto& x : ha) x = f32_to_bf16((rndf() - 0.5f) * 4.0f);
    for (auto& x : hbb) x = f32_to_bf16((rndf() - 0.5f) * 4.0f);
    for (auto& x : hz) x = f32_to_bf16((rndf() - 0.5f) * 4.0f);
    std::vector<float> halog(H), hdtb(H), hstate(SLOTS * H * V * K), hnw(V);
    for (auto& x : halog) x = logf(0.1f + rndf());
    for (auto& x : hdtb)  x = (rndf() - 0.5f) * 4.0f;
    for (auto& x : hstate) x = (rndf() - 0.5f) * 0.1f;
    for (auto& x : hnw)    x = (rndf() - 0.5f) * 2.0f;
    std::vector<uint16_t> ho(T * H * V, 0x7fc0u);
    const int cidxv[1] = { 0 }, sidxv[T] = { 0, 0, 1, 1 };
    /* num_accepted counts from 1 (docs/OPS.md): the window's history is read at column
     * num_accepted - 1, so 1 is the handover out of a prefill. */
    const int nav[1] = { 1 };
    const int cuv[2] = { 0, (int)T };

    struct { uint32_t dt; long long nel; } qd[18] = {
        { RAD_BF16, T * XD },        // x
        { RAD_BF16, XD * W },        // conv_w
        { RAD_BF16, XD },            // conv_b
        { RAD_BF16, XD * SL },       // conv_state [XD, 1, SL]
        { RAD_I32,  1 },             // conv_idx
        { RAD_BF16, T * H },         // a
        { RAD_BF16, T * H },         // b
        { RAD_F32,  H },             // A_log
        { RAD_F32,  H },             // dt_bias
        { RAD_F32,  SLOTS * H * V * K },  // state
        { RAD_I32,  T },             // state_idx [1, T]
        { RAD_I32,  1 },             // num_accepted
        { RAD_I32,  2 },             // cu
        { RAD_BF16, T * H * V },     // z
        { RAD_F32,  V },             // norm_w
        { RAD_BF16, T * H * V },     // o
        { RAD_I8,   0 },             // o_q: absent
        { RAD_F32,  0 }              // o_scale: absent
    };
    for (int o = 0; o < 16; ++o) {
        const size_t bytes = (size_t)qd[o].nel * dt_bytes(qd[o].dt);
        void* d = nullptr;
        if (hipMalloc(&d, bytes) != hipSuccess) { fail("gdn_conv_recurrent: hipMalloc(%zu) failed",
                                                       bytes); return 1; }
        c.dev.push_back(d);
        switch (o) {
            case 0: std::memcpy(d, hx.data(), bytes); break;
            case 1: std::memcpy(d, hw.data(), bytes); break;
            case 2: std::memcpy(d, hb.data(), bytes); break;
            case 3: std::memcpy(d, hcs.data(), bytes); break;
            case 4: std::memcpy(d, cidxv, sizeof cidxv); break;
            case 5: std::memcpy(d, ha.data(), bytes); break;
            case 6: std::memcpy(d, hbb.data(), bytes); break;
            case 7: std::memcpy(d, halog.data(), bytes); break;
            case 8: std::memcpy(d, hdtb.data(), bytes); break;
            case 9: std::memcpy(d, hstate.data(), bytes); break;
            case 10: std::memcpy(d, sidxv, sizeof sidxv); break;
            case 11: std::memcpy(d, nav, sizeof nav); break;
            case 12: std::memcpy(d, cuv, sizeof cuv); break;
            case 13: std::memcpy(d, hz.data(), bytes); break;
            case 14: std::memcpy(d, hnw.data(), bytes); break;
            default: std::memcpy(d, ho.data(), bytes); break;
        }
        RadTensor& t = c.t[o];
        t = RadTensor{};
        t.data = d; t.dtype = qd[o].dt;
        switch (o) {
            case 0: t.rank = 2; t.shape[0] = T; t.shape[1] = XD; t.stride[1] = 1; t.stride[0] = XD; break;
            case 1: t.rank = 2; t.shape[0] = XD; t.shape[1] = W; t.stride[1] = 1; t.stride[0] = W; break;
            case 2: t.rank = 1; t.shape[0] = XD; t.stride[0] = 1; break;
            case 3: t.rank = 3; t.shape[0] = XD; t.shape[1] = 1; t.shape[2] = SL;
                    t.stride[2] = 1; t.stride[1] = SL; t.stride[0] = SL; break;
            case 4: t.rank = 1; t.shape[0] = 1; t.stride[0] = 1; break;
            case 5: case 6: t.rank = 2; t.shape[0] = T; t.shape[1] = H; t.stride[1] = 1; t.stride[0] = H; break;
            case 7: case 8: t.rank = 1; t.shape[0] = H; t.stride[0] = 1; break;
            case 9: t.rank = 4; t.shape[0] = SLOTS; t.shape[1] = H; t.shape[2] = V; t.shape[3] = K;
                    t.stride[3] = 1; t.stride[2] = K; t.stride[1] = V * K; t.stride[0] = H * V * K; break;
            case 10: t.rank = 2; t.shape[0] = 1; t.shape[1] = T; t.stride[1] = 1; t.stride[0] = T; break;
            case 11: t.rank = 1; t.shape[0] = 1; t.stride[0] = 1; break;
            case 12: t.rank = 1; t.shape[0] = 2; t.stride[0] = 1; break;
            case 13: t.rank = 3; t.shape[0] = T; t.shape[1] = H; t.shape[2] = V;
                     t.stride[2] = 1; t.stride[1] = V; t.stride[0] = H * V; break;
            case 14: t.rank = 1; t.shape[0] = V; t.stride[0] = 1; break;
            default: t.rank = 3; t.shape[0] = T; t.shape[1] = H; t.shape[2] = V;
                     t.stride[2] = 1; t.stride[1] = V; t.stride[0] = H * V; break;
        }
    }
    c.t[16] = RadTensor{};   // o_q absent
    c.t[17] = RadTensor{};   // o_scale absent
    return 0;
}

static int run_row(int i) {
    const RadKernelInfo* k = row_at(i);
    if (!k) return 1;
    std::printf("== %s (%s)\n", k->name, k->op);
#ifndef RAD_HAVE_HIP
    (void)i;
    skip("built without the HIP runtime: no device run");
    return 0;
#else
    /* No device is a skip, not a failure: build machines and CI run the static phase alone. */
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
        skip("no device visible");
        return 0;
    }
    const int hand = k->opd_shape ? 0
        : (0 == std::strcmp(k->name, "gemm_nt_q_w2a8_m16")) ? 1
        : (0 == std::strcmp(k->name, "gemm_nt_q_w4a16asym_m64")) ? 3
        : (0 == std::strcmp(k->name, "gemm_nt_q_w5a16_m64")) ? 4
        : (0 == std::strcmp(k->name, "gdn_kkt_solve_c64")) ? 5
        : (0 == std::strcmp(k->name, "gdn_chunk_scan_k128_v128_c64")) ? 6
        : (0 == std::strcmp(k->name, "gdn_gated_rmsnorm_h128")) ? 7
        : (0 == std::strcmp(k->name, "gdn_gated_norm_had_quant_i8")) ? 8
        : (0 == std::strcmp(k->name, "gdn_conv_prep_w4_h128")) ? 9
        : (0 == std::strcmp(k->name, "gdn_conv_update_w4_h128")) ? 10
        : (0 == std::strcmp(k->name, "causal_conv1d_bf16")) ? 24
        : (0 == std::strcmp(k->name, "gdn_recurrent_update_k128_v128")) ? 11
        : (0 == std::strcmp(k->name, "gdn_conv_recurrent_update_k128_v128")) ? 23
        : (0 == std::strcmp(k->name, "expert_scale_bf16")) ? 12
        : (0 == std::strcmp(k->name, "moe_router_topk_bf16")) ? 13
        : (0 == std::strcmp(k->name, "moe_scatter_small")) ? 14
        : (0 == std::strcmp(k->name, "moe_gemm_q_e4m3")) ? 15
        : (0 == std::strcmp(k->name, "moe_gemm_bf16")) ? 16
        : (0 == std::strcmp(k->name, "moe_gemm_mxfp4a8")) ? 17
        : (0 == std::strcmp(k->name, "moe_router_topk_scatter")) ? 18
        : (0 == std::strcmp(k->name, "moe_router_sigmoid_topk")) ? 19
        : (0 == std::strcmp(k->name, "moe_gather")) ? 21
        : (0 == std::strcmp(k->name, "moe_gemm_w4a8")) ? 22 : 2;
    if (hand == 2) { skip("no operand description, so no buffers to build"); return 0; }
    const RadOpSchema* s = find_schema(k->op);
    if (!s) { fail("no schema for '%s'", k->op); return 1; }

    Case c;
    make_params(c, k, s);
    RadArgs a{};
    a.p = c.p;
    a.n_p = c.n_p;

    if (hand) {
        const int rc = hand == 1 ? build_w2a8(c, s)
                        : hand == 3 ? build_w4a16asym(c, s)
                        : hand == 4 ? build_w5a16(c, s)
                        : hand == 5 ? build_gdn_kkt(c, s)
                        : hand == 7 ? build_gdn_gated_rmsnorm(c, s)
                        : hand == 8 ? build_gdn_gated_norm_hq(c, s)
                        : hand == 9 ? build_gdn_conv_prep(c, s)
                        : hand == 10 ? build_gdn_conv_update(c, s)
                        : hand == 24 ? build_causal_conv1d(c, s)
                        : hand == 11 ? build_gdn_recurrent(c, s)
                        : hand == 23 ? build_gdn_conv_recurrent(c, s)
                        : hand == 12 ? build_expert_scale(c, s)
                        : hand == 13 ? build_moe_router_topk(c, s)
                        : hand == 14 ? build_moe_scatter(c, s)
                        : hand == 15 ? build_moe_gemm_q(c, s)
                        : hand == 16 ? build_moe_gemm_bf16(c, s)
                        : hand == 17 ? build_moe_gemm_mxfp4a8(c, s)
                        : hand == 18 ? build_moe_router_scatter(c, s)
                        : hand == 19 ? build_moe_router_sigmoid(c, s)
                        : hand == 21 ? build_moe_gather(c, s, 0)
                        : hand == 22 ? build_moe_gemm_w4a8(c, s)
                        : build_gdn_chunk_scan(c, s);
        if (rc) return 1;
        goto launched;
    }
    for (int o = 0; o < s->n_operands && o < MAX_OPD; ++o) {
        RadOpdDesc d{};
        const int st = k->opd_shape(c.p, c.n_p, o, &d);
        if (st != RAD_OK) { skip("operand %d: the description declines this geometry (%d)", o, st);
                            return 0; }
        RadTensor t{};
        if (d.flags & RAD_OPD_F_ABSENT) { c.t[o] = t; continue; }
        long long nel = 1;
        for (uint32_t r = 0; r < d.rank; ++r) nel *= d.shape[r];
        t.dtype = d.dtype;
        t.rank = d.rank;
        for (uint32_t r = 0; r < d.rank; ++r) t.shape[r] = d.shape[r];
        /* Sub-byte dtypes count elements, not bytes: fp4_e2m1 packs two a byte. */
        const size_t bytes = d.dtype == RAD_U2 ? (size_t)(nel / 4)
                         : (d.dtype == RAD_U4 || d.dtype == RAD_I4 || d.dtype == RAD_FP4E2M1)
                             ? (size_t)(nel / 2)
                         : (size_t)nel * dt_bytes(d.dtype);
        std::vector<uint8_t> h(bytes ? bytes : 1);
        fill_plane(d, h.data(), (size_t)nel, 0x9E3779B9u + (uint32_t)o * 7919u);
        /* TODO: pre-fill output planes with the dtype's NaN, to catch a shim that returns RAD_OK
         * without launching. Doing so makes `quant_act_i8_bf16` and `sample_logprobs_f32` die on
         * signal 11 in the spawned child (both pass with `--case`); both write planes the
         * reference pads. build_w2a8 already NaN-fills its output. */
        void* p = nullptr;
        if (hipMalloc(&p, bytes) != hipSuccess) { fail("hipMalloc(%zu) failed", bytes); return 1; }
        c.dev.push_back(p);
        if (hipMemcpy(p, h.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) {
            fail("hipMemcpy in failed"); return 1;
        }
        t.data = p;
        long long acc = 1;
        for (int r = (int)t.rank - 1; r >= 0; --r) { t.stride[r] = acc; acc *= t.shape[r]; }
        c.t[o] = t;
    }
launched:
    a.t = c.t;
    a.n_t = s->n_operands;

    /* The scratch the row asked for, at this geometry, if it asks for any. */
    void* scratch = nullptr;
    if (k->scratch) {
        const int64_t bytes = k->scratch(&a);
        if (bytes > 0 && hipMalloc(&scratch, (size_t)bytes) == hipSuccess) {
            (void)hipMemset(scratch, 0, (size_t)bytes);
            a.scratch = scratch;
            a.scratch_bytes = bytes;
        } else if (bytes > 0) {
            fail("the row wants %lld scratch bytes and none could be allocated",
                 (long long)bytes);
            return 1;
        }
    }

    const int rc = k->launch(&a, nullptr);
    const hipError_t le = hipGetLastError();
    const int sync = (int)hipDeviceSynchronize();
    int ret = 0;
    if (rc == RAD_E_UNSUPPORTED || rc == RAD_E_INVAL || rc == RAD_E_SHAPE ||
        rc == RAD_E_DTYPE) {
        skip("the shim declines this geometry (%d)", rc);
    } else if (rc != RAD_OK) {
        fail("launch returned %d", rc);
        /* Print the geometry, since the return code alone does not say what the shim was given. */
        for (int i = 0; i < c.n_p; ++i)
            if (c.p[i].kind == RAD_P_STR)
                std::printf("   param %s = %s\n", c.p[i].key, c.p[i].sval ? c.p[i].sval : "(null)");
            else
                std::printf("   param %s = %lld\n", c.p[i].key, c.p[i].ival);
        for (int o = 0; o < s->n_operands; ++o) {
            const RadTensor& t = c.t[o];
            std::printf("   operand %d '%s' dt=%u rank=%u data=%p", o, s->operands[o].name,
                        t.dtype, t.rank, t.data);
            for (uint32_t r = 0; r < t.rank; ++r) std::printf(" %lld/%lld", (long long)t.shape[r], (long long)t.stride[r]);
            std::printf("\n");
        }
        ret = 1;
    } else if (le != hipSuccess || sync != hipSuccess) {
        fail("the kernel faulted (last error %d, sync %d)", (int)le, sync);
        ret = 1;
    } else {
        int outs = 0;
        for (int o = 0; o < s->n_operands && !ret; ++o) {
            if (s->operands[o].role == RAD_OPD_IN) continue;
            const RadTensor& t = c.t[o];
            if (!t.data) continue;
            /* A joined plane is pinned host memory holding the container's bytes, not the row's
             * output: nothing to read back. */
            if (std::find(c.host.begin(), c.host.end(), t.data) != c.host.end()) continue;
            ++outs;
            long long nel = 1;
            for (uint32_t r = 0; r < t.rank; ++r) nel *= t.shape[r];
            /* Sub-byte planes are counted in codes, not bytes (a U2 code is a quarter byte). */
            const size_t bytes = t.dtype == RAD_U2 ? (size_t)(nel / 4)
                             : (t.dtype == RAD_U4 || t.dtype == RAD_I4 || t.dtype == RAD_FP4E2M1)
                                 ? (size_t)(nel / 2)
                             : (size_t)nel * dt_bytes(t.dtype);
            std::vector<uint8_t> h(bytes);
            if (hipMemcpy(h.data(), t.data, bytes, hipMemcpyDeviceToHost) != hipSuccess) {
                fail("hipMemcpy out failed: operand %d ('%s'), %zu bytes", o, s->operands[o].name,
                     bytes); ret = 1; break;
            }
            int nan = 0, big = 0;
            for (long long e = 0; e < nel; ++e) {
                float v;
                switch (t.dtype) {
                    case RAD_F32:   v = *(const float*)(h.data() + e * 4); break;
                    case RAD_BF16:  v = bf16_to_f32(*(const uint16_t*)(h.data() + e * 2)); break;
                    case RAD_F16:   v = f16_to_f32(*(const uint16_t*)(h.data() + e * 2)); break;
                    case RAD_F8E4M3: v = e4m3_to_f32(h.data()[e]); break;
                    default: continue;
                }
                if (v != v) ++nan;
                else if (!(v > -1e31f && v < 1e31f)) ++big;
            }
            /* -infinity is the sampler's contract (a dropped or unfilled candidate, a masked or
             * banned logit, an unfilled row_topk slot); a NaN is always a failure. */
            const bool sentinel = (k->family && 0 == std::strcmp(k->family, "sample")) ||
                                  (k->op && 0 == std::strcmp(k->op, "row_topk"));
            /* Two planes are padded to the reference's grid: `asum`, the int8 activation's row sums on
             * a 16-row grid, and `stats`. The row leaves the pad unwritten, so a NaN there is not
             * the row's work. Named planes only; NaN anywhere else is a failure. */
            const bool padded = s->operands[o].name &&
                (0 == std::strcmp(s->operands[o].name, "asum") ||
                 0 == std::strcmp(s->operands[o].name, "stats"));
            if (nan && padded) {
                ok("ran, %d output plane%s read back with %d pad NaN of %lld on '%s' (the "
                   "reference's grid, not the row's work)", outs, outs == 1 ? "" : "s", nan,
                   nel, s->operands[o].name);
            } else if (nan) { fail("operand %d ('%s'): %d NaN of %lld", o, s->operands[o].name, nan,
                            nel); ret = 1; }
            else if (big && !sentinel) {
                fail("operand %d ('%s'): %d non-finite of %lld", o, s->operands[o].name, big, nel);
                ret = 1;
            } else if (big) {
                ok("ran, %d output plane%s read back with %d sentinel%s of %lld on '%s' (the "
                   "contract's -inf)", outs, outs == 1 ? "" : "s", big, big == 1 ? "" : "s", nel,
                   s->operands[o].name);
            }
        }
        if (!ret) ok("ran, %d output plane%s read back finite", outs, outs == 1 ? "" : "s");

        /* The implementation arms. A row that declared an `impl` axis has a second kernel the
         * default run never touches, and the engine's own way of reaching it is to append the
         * chosen point to RadArgs::p -- which is what this pass does: the same buffers, the same
         * fill, the arm's launch, the same finiteness read-back. An arm that faults, declines a
         * geometry the dot2 arm serves, or leaves a NaN is a failure here, not at a tune run. */
        for (int t = 0; t < k->n_tunables && !ret; ++t) {
            if (!k->tunables[t].key || std::strcmp(k->tunables[t].key, "impl")) continue;
            // impl=1, and sk=1 where the row has a split-K axis: the WMMA arm's ladder would
            // pick a wide sk that needs a scratch buffer, which this probe does not carry.
            // sk=1 is a valid point on every axis that has one. The original parameters are
            // kept: a shim that reads N / K from them must see the same geometry.
            RadParam extra = { "impl", RAD_P_INT, 1, 0, nullptr, 0.0 };
            RadArgs a2 = a;
            static RadParam extra_p[64];
            int np = 0;
            for (int u = 0; u < a.n_p && np < 60; ++u) extra_p[np++] = a.p[u];
            extra_p[np++] = extra;
            for (int u = 0; u < k->n_tunables; ++u)
                if (k->tunables[u].key && !std::strcmp(k->tunables[u].key, "sk"))
                    extra_p[np++] = (RadParam){ "sk", RAD_P_INT, 1, 0, nullptr, 0.0 };
            a2.p = extra_p;
            a2.n_p = np;
            const int rc2 = k->launch(&a2, nullptr);
            if (rc2 != RAD_OK) {
                fail("the impl=1 arm declined the geometry the default serves (%d)", rc2);
                ret = 1;
            } else {
                const hipError_t le2 = hipGetLastError();
                const int sync2 = (int)hipDeviceSynchronize();
                if (le2 != hipSuccess || sync2 != hipSuccess) {
                    fail("the impl=1 arm faulted (last error %d, sync %d)", (int)le2, (int)sync2);
                    ret = 1;
                } else {
                    int nan = 0, big = 0, outs2 = 0;
                    for (int o = 0; o < s->n_operands && !ret; ++o) {
                        if (s->operands[o].role == RAD_OPD_IN) continue;
                        const RadTensor& t2 = c.t[o];
                        if (!t2.data) continue;
                        if (std::find(c.host.begin(), c.host.end(), t2.data) != c.host.end())
                            continue;
                        ++outs2;
                        long long nel = 1;
                        for (uint32_t r = 0; r < t2.rank; ++r) nel *= t2.shape[r];
                        const size_t bytes = t2.dtype == RAD_U2 ? (size_t)(nel / 4)
                                       : (t2.dtype == RAD_U4 || t2.dtype == RAD_I4 ||
                                          t2.dtype == RAD_FP4E2M1)
                                           ? (size_t)(nel / 2)
                                           : (size_t)nel * dt_bytes(t2.dtype);
                        std::vector<uint8_t> h(bytes);
                        if (hipMemcpy(h.data(), t2.data, bytes, hipMemcpyDeviceToHost) !=
                            hipSuccess) {
                            fail("impl=1 read-back failed on operand %d", o);
                            ret = 1;
                            break;
                        }
                        for (long long e = 0; e < nel; ++e) {
                            float v = 0.f;
                            switch (t2.dtype) {
                                case RAD_F32:   v = *(const float*)(h.data() + e * 4); break;
                                case RAD_BF16:  v = bf16_to_f32(*(const uint16_t*)(h.data() + e * 2)); break;
                                case RAD_F16:   v = f16_to_f32(*(const uint16_t*)(h.data() + e * 2)); break;
                                case RAD_F8E4M3: v = e4m3_to_f32(h.data()[e]); break;
                                default: continue;
                            }
                            if (v != v) ++nan;
                            else if (!(v > -1e31f && v < 1e31f)) ++big;
                        }
                        if (nan) { fail("impl=1 arm, operand %d: %d NaN of %lld", o, nan, nel);
                                   ret = 1; }
                    }
                    if (!ret)
                        ok("impl=1 arm ran, %d output plane%s read back finite", outs2,
                           outs2 == 1 ? "" : "s");
                }
            }
        }
    }

    if (scratch) (void)hipFree(scratch);
    for (void* p : c.dev) {
        if (std::find(c.host.begin(), c.host.end(), p) != c.host.end()) (void)hipHostFree(p);
        else (void)hipFree(p);
    }
    return ret;
#endif
}

/* ================================================================== main */
/* ================================================================== the grouped GEMMs, by value
 *
 * The row cases above check that each MoE GEMM runs and writes finite planes; this checks what it
 * writes. A routed batch (random ids, some outside the local slice and so dropped, stably sorted
 * by expert as moe_scatter sorts them) runs through moe_gemm at bf16 and moe_gemm_q at fp8a8f32
 * (every impl arm) and mxfp4a8, in both a_orders, with the experts as the engine hands them: a
 * device table of one pointer per expert (stride[0] 0), even and odd experts in separate classes.
 * N is not a multiple of the 128-column block, so the tail block is checked too. The reference is
 * libref's arithmetic on the host (the products exact, the sums in double); a dropped row must be
 * zero. Agreement is the relative L2 error over y, as rad-kbench measures it.
 */
#ifdef RAD_HAVE_HIP
namespace moenum {

struct Dev {
    std::vector<void*> p;
    ~Dev() { for (void* x : p) hipFree(x); }
    void* put(const void* h, size_t n) {
        void* d = nullptr;
        if (hipMalloc(&d, n ? n : 4) != hipSuccess) return nullptr;
        p.push_back(d);
        if (h && n) hipMemcpy(d, h, n, hipMemcpyHostToDevice);
        else hipMemset(d, 0, n ? n : 4);
        return d;
    }
};

RadTensor dense(void* d, uint32_t dt, long long r, long long c) {
    RadTensor t{};
    t.data = d; t.dtype = dt; t.rank = 2;
    t.shape[0] = r; t.shape[1] = c; t.stride[0] = c; t.stride[1] = 1;
    return t;
}
RadTensor vec(void* d, uint32_t dt, long long n) {
    RadTensor t{};
    t.data = d; t.dtype = dt; t.rank = 1; t.shape[0] = n; t.stride[0] = 1;
    return t;
}
/* A weight table of `n` entries, each [rows, cols] (cols in elements): the device array of
 * pointers the core builds for a run of separately placed experts. */
RadTensor table(Dev& dv, const std::vector<std::vector<uint8_t>>& ent, uint32_t dt, long long rows,
                long long cols) {
    std::vector<void*> ptr;
    for (const auto& e : ent) ptr.push_back(dv.put(e.data(), e.size()));
    if (ptr.empty()) ptr.push_back(dv.put(nullptr, 64));
    RadTensor t{};
    t.data = dv.put(ptr.data(), ptr.size() * sizeof(void*));
    t.dtype = dt; t.rank = 3;
    t.shape[0] = (long long)ent.size(); t.shape[1] = rows; t.shape[2] = cols;
    t.stride[0] = 0; t.stride[1] = cols; t.stride[2] = 1;
    return t;
}

struct Route {
    std::vector<int32_t> sorted, off;
    long long T = 0;
};
/* Each token picks top_k distinct experts out of ne + 1, the last meaning "not on this rank":
 * those slots are dropped, sorted to the tail as -1. */
Route route(long long M, long long top_k, long long ne, uint32_t& st) {
    auto rnd = [&]() { st = st * 1664525u + 1013904223u; return st >> 8; };
    Route r;
    r.T = M * top_k;
    std::vector<int32_t> ids(r.T);
    for (long long t = 0; t < M; ++t)
        for (long long j = 0; j < top_k; ++j) {
            int32_t e;
            bool dup;
            do {
                e = (int32_t)(rnd() % (uint32_t)(ne + 1));
                dup = false;
                for (long long q = 0; q < j; ++q) dup |= ids[t * top_k + q] == e;
            } while (dup);
            ids[t * top_k + j] = e;
        }
    r.off.assign(ne + 1, 0);
    for (int32_t e = 0; e < ne; ++e) {
        r.off[e] = (int32_t)r.sorted.size();
        for (long long i = 0; i < r.T; ++i) if (ids[i] == e) r.sorted.push_back((int32_t)i);
    }
    r.off[ne] = (int32_t)r.sorted.size();
    while ((long long)r.sorted.size() < r.T) r.sorted.push_back(-1);
    return r;
}

int expert_of(const Route& r, long long i, long long ne) {
    for (long long e = 0; e < ne; ++e) if (r.off[e] <= i && i < r.off[e + 1]) return (int)e;
    return -1;
}

void compare(const char* what, const std::vector<double>& ref, const std::vector<uint16_t>& y,
             double tol) {
    double num = 0, den = 0, worst_zero = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double v = bf16_to_f32(y[i]);
        num += (v - ref[i]) * (v - ref[i]);
        den += ref[i] * ref[i];
        if (ref[i] == 0.0) worst_zero = std::max(worst_zero, std::fabs(v));
    }
    const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
    if (rel <= tol && worst_zero == 0.0) ok("%s: rel_l2 %.2e", what, rel);
    else fail("%s: rel_l2 %.2e (tolerance %.0e), a dropped row's largest |y| %.3g", what, rel, tol,
              worst_zero);
}

RadParam pi(const char* k, long long v) { return RadParam{ k, RAD_P_INT, v, 0, nullptr, 0.0 }; }
RadParam ps(const char* k, const char* v) { return RadParam{ k, RAD_P_STR, 0, 0, v, 0.0 }; }

typedef int (*Launch)(const RadArgs*, RadStream);

}  // namespace moenum
#endif

static void check_moe_numerics() {
    std::printf("== the grouped GEMMs, by value\n");
#ifndef RAD_HAVE_HIP
    skip("built without HIP: the grouped GEMMs are not run");
#else
    using namespace moenum;
    auto bf16_launch = (Launch)dlsym(g_h, "r3d_rad_moe_gemm_bf16");
    auto q_launch    = (Launch)dlsym(g_h, "r3d_rad_moe_gemm_q");
    auto mx_launch   = (Launch)dlsym(g_h, "r3d_rad_moe_gemm_mxfp4a8");
    if (!bf16_launch || !q_launch || !mx_launch) {
        fail("a grouped GEMM entry point is missing");
        return;
    }

    const long long M = 12, TK = 2, NE = 6, N = 192;
    for (int sorted_a = 0; sorted_a < 2; ++sorted_a) {
        const char* order = sorted_a ? "sorted" : "token";
        uint32_t st = 0x5EED0000u + (uint32_t)sorted_a;
        const Route r = route(M, TK, NE, st);
        const long long T = r.T, AR = sorted_a ? T : M;
        auto rndf = [&]() {
            st = st * 1664525u + 1013904223u;
            return ((st >> 8) & 0xffffu) / 65536.0f - 0.5f;
        };
        auto rnd8 = [&]() { st = st * 1664525u + 1013904223u; return (uint8_t)(st >> 16); };
        /* An E4M3 code in the normal range (exponent 1..14), either sign. */
        auto e4 = [&]() {
            return (uint8_t)(((rnd8() & 0x7) | 1) << 3 | (rnd8() & 0x7) | ((rnd8() & 1) << 7));
        };
        auto tok_of = [&](long long i) { return sorted_a ? i : r.sorted[i] / TK; };

        /* ---- moe_gemm, bf16 experts in one class. */
        {
            const long long K = 128;
            Dev dv;
            std::vector<uint16_t> ha(AR * K);
            for (auto& x : ha) x = f32_to_bf16(rndf());
            std::vector<std::vector<uint8_t>> we(NE, std::vector<uint8_t>(N * K * 2));
            for (auto& e : we) for (long long q = 0; q < N * K; ++q) {
                const uint16_t b = f32_to_bf16(rndf());
                std::memcpy(&e[q * 2], &b, 2);
            }
            RadTensor t[5] = { dense(dv.put(ha.data(), ha.size() * 2), RAD_BF16, AR, K),
                               table(dv, we, RAD_BF16, N, K),
                               vec(dv.put(r.sorted.data(), T * 4), RAD_I32, T),
                               vec(dv.put(r.off.data(), (NE + 1) * 4), RAD_I32, NE + 1),
                               dense(dv.put(nullptr, T * N * 2), RAD_BF16, T, N) };
            const RadParam p[] = { pi("M", T), pi("N", N), pi("K", K), pi("n_expert", NE),
                                   pi("top_k", TK), ps("a_order", order), ps("dtype", "bf16") };
            RadArgs a{ t, 5, p, (int)(sizeof(p) / sizeof(p[0])), nullptr, 0, nullptr, 0, 1 };
            const int rc = bf16_launch(&a, nullptr);
            std::vector<uint16_t> y(T * N);
            hipDeviceSynchronize();
            hipMemcpy(y.data(), t[4].data, y.size() * 2, hipMemcpyDeviceToHost);
            std::vector<double> ref(T * N, 0.0);
            for (long long i = 0; i < T; ++i) {
                const int e = expert_of(r, i, NE);
                if (e < 0) continue;
                for (long long n = 0; n < N; ++n) {
                    double acc = 0;
                    for (long long k = 0; k < K; ++k) {
                        uint16_t w; std::memcpy(&w, &we[e][(n * K + k) * 2], 2);
                        acc += (double)bf16_to_f32(ha[tok_of(i) * K + k]) * bf16_to_f32(w);
                    }
                    ref[i * N + n] = acc;
                }
            }
            std::string what = std::string("moe_gemm bf16, a_order ") + order;
            if (rc != 0) fail("%s: the launch returned %d", what.c_str(), rc);
            else compare(what.c_str(), ref, y, 1e-2);
        }

        /* ---- moe_gemm_q at fp8a8f32 (every impl arm) and mxfp4a8: the even and odd classes. */
        for (int mx = 0; mx < 2; ++mx) {
            const long long K = 256, NB = (N + 127) / 128;
            Dev dv;
            std::vector<uint8_t> ha(AR * K);
            std::vector<float> has(AR * (K / 128));
            for (auto& x : ha) x = e4();
            for (auto& x : has) x = 0.25f + std::fabs(rndf());
            /* Expert e's codes and scales, and their values for the reference. */
            const long long wbytes = mx ? N * K / 2 : N * K;
            const long long scols = mx ? K / 32 : K / 128, srows = mx ? N : NB;
            std::vector<std::vector<uint8_t>> wc(NE, std::vector<uint8_t>(wbytes)),
                ws(NE, std::vector<uint8_t>(srows * scols * (mx ? 1 : 4)));
            for (long long e = 0; e < NE; ++e) {
                for (auto& x : wc[e]) x = mx ? rnd8() : e4();
                for (long long q = 0; q < srows * scols; ++q) {
                    if (mx) {
                        ws[e][q] = (uint8_t)(120 + (rnd8() & 15));
                    } else {
                        const float f = 0.25f + std::fabs(rndf());
                        std::memcpy(&ws[e][q * 4], &f, 4);
                    }
                }
            }
            auto wval = [&](int e, long long n, long long k) -> double {
                if (!mx) {
                    float sc; std::memcpy(&sc, &ws[e][((n / 128) * scols + k / 128) * 4], 4);
                    return (double)e4m3_to_f32(wc[e][n * K + k]) * sc;
                }
                static const float kMag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
                const uint8_t b = wc[e][(n * K + k) / 2];
                const uint8_t nib = (k & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0xf);
                const double v = (nib & 8) ? -kMag[nib & 7] : kMag[nib & 7];
                return v * std::ldexp(1.0, (int)ws[e][n * scols + k / 32] - 127);
            };
            std::vector<double> ref(T * N, 0.0);
            for (long long i = 0; i < T; ++i) {
                const int e = expert_of(r, i, NE);
                if (e < 0) continue;
                const long long tk = tok_of(i);
                for (long long n = 0; n < N; ++n) {
                    double acc = 0;
                    for (long long k = 0; k < K; ++k)
                        acc += (double)e4m3_to_f32(ha[tk * K + k]) * has[tk * (K / 128) + k / 128] *
                               wval(e, n, k);
                    ref[i * N + n] = acc;
                }
            }
            std::vector<std::vector<uint8_t>> wc0, wc1, ws0, ws1;
            for (long long e = 0; e < NE; ++e) {
                (e % 2 ? wc1 : wc0).push_back(wc[e]);
                (e % 2 ? ws1 : ws0).push_back(ws[e]);
            }
            const uint32_t wdt = mx ? (uint32_t)RAD_FP4E2M1 : (uint32_t)RAD_F8E4M3;
            const uint32_t sdt = mx ? (uint32_t)RAD_U8 : (uint32_t)RAD_F32;
            RadTensor t[9] = { dense(dv.put(ha.data(), ha.size()), RAD_F8E4M3, AR, K),
                               dense(dv.put(has.data(), has.size() * 4), RAD_F32, AR, K / 128),
                               table(dv, wc0, wdt, N, K), table(dv, ws0, sdt, srows, scols),
                               vec(dv.put(r.sorted.data(), T * 4), RAD_I32, T),
                               vec(dv.put(r.off.data(), (NE + 1) * 4), RAD_I32, NE + 1),
                               table(dv, wc1, wdt, N, K), table(dv, ws1, sdt, srows, scols),
                               dense(dv.put(nullptr, T * N * 2), RAD_BF16, T, N) };
            for (int impl = 0; impl < (mx ? 1 : 3); ++impl) {
                /* Stale bytes, which a dropped row must not keep. */
                hipMemset(t[8].data, 0xff, T * N * 2);
                const RadParam p[] = { pi("M", T), pi("N", N), pi("K", K), pi("n_expert", NE),
                                       pi("top_k", TK), pi("group", 128), ps("a_order", order),
                                       ps("dtype", mx ? "mxfp4a8" : "fp8a8f32"), pi("impl", impl),
                                       pi("vgpr", 128), pi("acc", 0) };
                RadArgs a{ t, 9, p, (int)(sizeof(p) / sizeof(p[0])), nullptr, 0, nullptr, 0, 1 };
                const int rc = (mx ? mx_launch : q_launch)(&a, nullptr);
                std::vector<uint16_t> y(T * N);
                hipDeviceSynchronize();
                hipMemcpy(y.data(), t[8].data, y.size() * 2, hipMemcpyDeviceToHost);
                char what[96];
                std::snprintf(what, sizeof what, "moe_gemm_q %s%s, a_order %s",
                              mx ? "mxfp4a8" : "fp8a8f32",
                              mx ? "" : (impl == 0 ? " impl 0" : impl == 1 ? " impl 1" : " impl 2"),
                              order);
                if (rc != 0) fail("%s: the launch returned %d", what, rc);
                else compare(what, ref, y, 1e-2);
            }
        }

    }
#endif
}

/* ================================================================== causal_conv1d, by value
 *
 * LFM2's and LFM3's short convolution against a host mirror of libref/ref_ple.cpp's depthwise
 * causal convolution (ported from libhalo's causal_conv1d_cases): the taps summed in f64 in ref's
 * order over the same bf16 bytes, the store narrowed to bf16, and the window rewritten by ref's
 * rule -- `num_accepted` present shifts the read down by one and puts the step's inputs behind it,
 * absent keeps the last width-1 inputs at offset zero. y is compared to a relative L2 (the kernel
 * accumulates in f32); the window, a copy of input bytes, byte for byte; and the rows of a
 * sequence that writes nothing (empty, or on slot -1) must keep their NaN fill.
 */
namespace convnum {
using namespace moenum;

struct ConvCase {
    long long n = 0, wid = 0, slots = 1, state_len = 0;
    std::vector<int32_t> cu, sidx, hini, nacc;           /* sidx/hini/nacc empty: absent */
    std::vector<uint16_t> x, w, st;                       /* x [T, n], w [n, wid], st [slots, n, state_len] */
    std::vector<double> y;                                /* the answer, [T, n] */
    std::vector<char> live;
    long long T() const { return cu.empty() ? 0 : cu.back(); }
};

void model(ConvCase& c) {
    const long long hist = c.wid - 1, N = (long long)c.cu.size() - 1, T = c.T();
    c.y.assign((size_t)(T * c.n), 0.0);
    c.live.assign((size_t)T, 0);
    for (long long s = 0; s < N; ++s) {
        const long long bos = c.cu[s], slen = c.cu[s + 1] - bos;
        if (slen <= 0) continue;
        const long long slot = c.sidx.empty() ? s : c.sidx[s];
        if (slot < 0) continue;
        const bool warm = c.hini.empty() || c.hini[s] != 0;
        const long long off = c.nacc.empty() ? 0 : (long long)c.nacc[s] - 1;
        for (long long ch = 0; ch < c.n; ++ch) {
            const long long base = (slot * c.n + ch) * c.state_len;
            float h[8] = {};
            for (long long i = 0; i < hist; ++i) h[i] = warm ? bf16_to_f32(c.st[base + off + i]) : 0.0f;
            for (long long t = 0; t < slen; ++t) {
                c.live[bos + t] = 1;
                double acc = 0.0;
                for (long long j = 0; j < c.wid; ++j) {
                    const long long src = t - hist + j;
                    const double v = src >= 0 ? (double)bf16_to_f32(c.x[(bos + src) * c.n + ch])
                                              : (double)h[src + hist];
                    acc += v * (double)bf16_to_f32(c.w[ch * c.wid + j]);
                }
                c.y[(bos + t) * c.n + ch] = acc;
            }
            if (!c.nacc.empty()) {
                uint16_t keep[8] = {};
                for (long long i = 0; i + 1 < hist; ++i) keep[i] = warm ? c.st[base + off + i + 1] : 0;
                for (long long i = 0; i + 1 < hist; ++i) c.st[base + i] = keep[i];
                for (long long t = 0; t < slen && hist - 1 + t < c.state_len; ++t)
                    c.st[base + hist - 1 + t] = c.x[(bos + t) * c.n + ch];
            } else {
                uint16_t keep[8] = {};
                for (long long i = 0; i < hist; ++i) {
                    const long long src = slen - hist + i;
                    keep[i] = src >= 0 ? c.x[(bos + src) * c.n + ch]
                                       : (warm ? c.st[base + off + src + hist] : (uint16_t)0);
                }
                for (long long i = 0; i < hist; ++i) c.st[base + i] = keep[i];
            }
        }
    }
}

/* Upload, launch, model, compare. `dil` > 0 adds a `dilation` parameter (a refusal case). */
int run(Launch fn, ConvCase& c, const char* what, long long dil = 0, bool report = true) {
    const long long T = c.T();
    Dev dv;
    std::vector<uint16_t> ynan((size_t)(T * c.n), 0x7fc0u);
    RadTensor cs{};
    cs.data = dv.put(c.st.data(), c.st.size() * 2); cs.dtype = RAD_BF16; cs.rank = 3;
    cs.shape[0] = c.slots; cs.shape[1] = c.n; cs.shape[2] = c.state_len;
    cs.stride[2] = 1; cs.stride[1] = c.state_len; cs.stride[0] = c.n * c.state_len;
    auto opt = [&](const std::vector<int32_t>& v) {
        return v.empty() ? RadTensor{} : vec(dv.put(v.data(), v.size() * 4), RAD_I32, (long long)v.size());
    };
    RadTensor t[8] = { dense(dv.put(c.x.data(), c.x.size() * 2), RAD_BF16, T, c.n),
                       dense(dv.put(c.w.data(), c.w.size() * 2), RAD_BF16, c.n, c.wid),
                       cs,
                       vec(dv.put(c.cu.data(), c.cu.size() * 4), RAD_I32, (long long)c.cu.size()),
                       opt(c.sidx), opt(c.hini), opt(c.nacc),
                       dense(dv.put(ynan.data(), ynan.size() * 2), RAD_BF16, T, c.n) };
    std::vector<RadParam> p = { pi("M", T), pi("n", c.n), pi("width", c.wid), ps("dtype", "bf16") };
    if (dil) p.push_back(pi("dilation", dil));
    RadArgs a{ t, 8, p.data(), (int)p.size(), nullptr, 0, nullptr, 0, 1 };
    const int rc = fn(&a, nullptr);
    hipDeviceSynchronize();
    if (!report) return rc;
    if (rc != 0) { fail("%s: the launch returned %d", what, rc); return rc; }
    std::vector<uint16_t> y((size_t)(T * c.n)), st(c.st.size());
    hipMemcpy(y.data(), t[7].data, y.size() * 2, hipMemcpyDeviceToHost);
    hipMemcpy(st.data(), cs.data, st.size() * 2, hipMemcpyDeviceToHost);
    model(c);
    std::vector<double> ref;
    std::vector<uint16_t> got;
    long long dead_moved = 0;
    for (long long r = 0; r < T; ++r)
        for (long long ch = 0; ch < c.n; ++ch) {
            const uint16_t g = y[r * c.n + ch];
            if (c.live[r]) { ref.push_back(c.y[r * c.n + ch]); got.push_back(g); }
            else dead_moved += g != 0x7fc0u;
        }
    compare(what, ref, got, 2e-3);
    long long sdiff = 0;
    for (size_t i = 0; i < st.size(); ++i) sdiff += st[i] != c.st[i];
    if (sdiff) fail("%s: the window after the step: %lld of %zu values differ", what, sdiff, st.size());
    else ok("%s: the window after the step, byte for byte", what);
    if (dead_moved) fail("%s: %lld values written in rows no sequence commits", what, dead_moved);
    return 0;
}

}  // namespace convnum

static void check_conv_numerics() {
    std::printf("== causal_conv1d, by value\n");
#ifndef RAD_HAVE_HIP
    skip("built without HIP: causal_conv1d is not run");
#else
    using namespace convnum;
    auto fn = (Launch)dlsym(g_h, "r3d_rad_causal_conv1d");
    if (!fn) { fail("r3d_rad_causal_conv1d is missing"); return; }
    /* No device is a skip, as in run_row: a build machine runs the static phase alone. */
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) { skip("no device visible"); return; }
    uint32_t st = 0xC0417D1u;
    auto rnd = [&](float sc) { st = st * 1664525u + 1013904223u;
                               return f32_to_bf16((((st >> 8) & 0xffffu) / 65536.0f - 0.5f) * sc); };
    auto fill = [&](ConvCase& c, int with_state) {
        const long long T = c.T();
        c.x.resize((size_t)(T * c.n)); c.w.resize((size_t)(c.n * c.wid));
        c.st.resize((size_t)(c.slots * c.n * c.state_len));
        for (auto& v : c.x) v = rnd(2.0f);
        for (auto& v : c.w) v = rnd(1.0f);
        for (auto& v : c.st) v = with_state ? rnd(2.0f) : 0;
    };
    {   /* LFM2's band at prefill: an empty sequence, a cold window over stale bytes, a slot of -1. */
        ConvCase c; c.n = 2048; c.wid = 3; c.slots = 3; c.state_len = 2;
        c.cu = { 0, 0, 3, 12, 15 }; c.sidx = { 0, 2, 0, -1 }; c.hini = { 1, 0, 1, 1 };
        fill(c, 1);
        run(fn, c, "causal_conv1d prefill, width 3: empty, cold, warm and slot -1 sequences");
    }
    {   /* A decode step with drafts (num_accepted) over a window of width - 1 + 3. */
        ConvCase c; c.n = 1024; c.wid = 3; c.slots = 2; c.state_len = 2 + 3;
        c.cu = { 0, 4, 8 }; c.sidx = { 1, 0 }; c.hini = { 1, 1 }; c.nacc = { 2, 4 };
        fill(c, 1);
        run(fn, c, "causal_conv1d decode, width 3, num_accepted 2 and 4, a window of 5");
    }
    {   /* The widest and narrowest the row takes. */
        ConvCase c; c.n = 512; c.wid = 8; c.slots = 1; c.state_len = 7; c.cu = { 0, 10 };
        fill(c, 1);
        run(fn, c, "causal_conv1d prefill, width 8, 7 values of history");
        ConvCase d; d.n = 512; d.wid = 2; d.slots = 1; d.state_len = 1; d.cu = { 0, 5 };
        fill(d, 1);
        run(fn, d, "causal_conv1d prefill, width 2, one value of history");
    }
    {   /* The refusals: a dilation is not this op's; a window shallower than width - 1. */
        ConvCase c; c.n = 256; c.wid = 3; c.slots = 1; c.state_len = 2; c.cu = { 0, 4 };
        fill(c, 1);
        const int r1 = run(fn, c, "", 2, false);
        if (r1 == RAD_E_INVAL) ok("causal_conv1d refuses a dilation of 2");
        else fail("causal_conv1d: a dilation of 2 returned %d, not RAD_E_INVAL", r1);
        ConvCase d = c; d.state_len = 1; fill(d, 1);
        const int r2 = run(fn, d, "", 0, false);
        if (r2 == RAD_E_SHAPE) ok("causal_conv1d refuses a window shallower than width - 1");
        else fail("causal_conv1d: a window of 1 at width 3 returned %d, not RAD_E_SHAPE", r2);
    }
#endif
}

/* ---- rope, by value -----------------------------------------------------------------------------
 * run_row launches the rope rows with their required params only and checks that the output is
 * finite, so neither `ladder` nor a partial `rotary_dim` is exercised there. This case is libhalo's
 * (rope_ladder_case): an f32 mirror of libref's rotation at the three ladders. `head` is the
 * proportional ladder: pair i is (i, i + head_dim/2), the exponent's divisor is head_dim, and the
 * pairs from rotary_dim/2 on do not turn. Q and K heads rotate; the V head must come back byte
 * for byte. */
static void check_rope_numerics() {
    std::printf("== rope, by value\n");
#ifndef RAD_HAVE_HIP
    skip("built without HIP: rope is not run");
#else
    using namespace moenum;
    auto fn = (Launch)dlsym(g_h, "r3d_rad_rope");
    if (!fn) { fail("r3d_rad_rope is missing"); return; }
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) { skip("no device visible"); return; }
    const long long M = 5, hd = 64, rot = 32, n_head = 2, nkv = 1, n_rope = n_head + nkv;
    const long long row = (n_head + 2 * nkv) * hd;
    const float theta = 10000.0f;
    const std::vector<int32_t> pos = { 0, 1, 7, 128, 4096 };
    for (const char* lad : { (const char*)nullptr, "rotary", "head" }) {
        const bool head = lad && !std::strcmp(lad, "head");
        const long long den = head ? hd : rot, off = head ? hd / 2 : rot / 2;
        char what[96];
        std::snprintf(what, sizeof(what), "rope neox, head_dim 64, rotary_dim 32, ladder %s",
                      lad ? lad : "absent");
        std::vector<uint16_t> x((size_t)(M * row));
        for (long long i = 0; i < M * row; ++i) {
            float v = (float)(((i * 5) % 17) - 8) / 9.0f;
            if (i % 13 == 0) v *= 4.0f;                        /* a few big channels */
            x[(size_t)i] = f32_to_bf16(v);
        }
        Dev dv;
        RadTensor t[2] = { dense(dv.put(x.data(), x.size() * 2), RAD_BF16, M, row),
                           vec(dv.put(pos.data(), pos.size() * 4), RAD_I32, M) };
        std::vector<RadParam> p = { pi("M", M), pi("head_dim", hd), pi("n_head", n_head),
                                    pi("n_head_kv", nkv),
                                    RadParam{ "theta", RAD_P_F64, 0, 0, nullptr, (double)theta },
                                    RadParam{ "scale", RAD_P_F64, 0, 0, nullptr, 1.0 },
                                    ps("mode", "neox"), pi("rotary_dim", rot) };
        if (lad) p.push_back(ps("ladder", lad));
        RadArgs a{ t, 2, p.data(), (int)p.size(), nullptr, 0, nullptr, 0, 1 };
        const int rc = fn(&a, nullptr);
        hipDeviceSynchronize();
        if (rc != 0) { fail("%s: the launch returned %d", what, rc); continue; }
        std::vector<uint16_t> got(x.size());
        hipMemcpy(got.data(), t[0].data, got.size() * 2, hipMemcpyDeviceToHost);
        std::vector<double> ref;
        std::vector<uint16_t> y;
        long long v_moved = 0;
        for (long long m = 0; m < M; ++m) {
            const size_t base = (size_t)(m * row);
            for (long long h = 0; h < n_rope; ++h) {
                std::vector<double> w((size_t)hd);
                for (long long e = 0; e < hd; ++e) w[(size_t)e] = bf16_to_f32(x[base + h * hd + e]);
                for (long long i = 0; i < rot / 2; ++i) {
                    const float x0 = bf16_to_f32(x[base + h * hd + i]);
                    const float x1 = bf16_to_f32(x[base + h * hd + i + off]);
                    const float inv = 1.0f / powf(theta, (float)(2 * i) / (float)den);
                    const float sn = sinf((float)pos[(size_t)m] * inv);
                    const float cs = cosf((float)pos[(size_t)m] * inv);
                    w[(size_t)i] = x0 * cs - x1 * sn;
                    w[(size_t)(i + off)] = x0 * sn + x1 * cs;
                }
                for (long long e = 0; e < hd; ++e) {
                    ref.push_back(w[(size_t)e]); y.push_back(got[base + h * hd + e]);
                }
            }
            for (long long e = 0; e < hd; ++e)
                v_moved += got[base + n_rope * hd + e] != x[base + n_rope * hd + e];
        }
        compare(what, ref, y, 2e-3);
        if (v_moved) fail("%s: %lld values of the V head moved", what, v_moved);
        else ok("%s: the V head byte for byte", what);
    }
#endif
}

/* ---- attn_paged decode at depth, by value --------------------------------------------------------
 * One query row over a 4,096-key context, the shape a decode step has and rad-kbench's attention
 * shape (context = q_len) does not. The row's scratch hook is called the way the builder calls it,
 * with the params and no tensors, and must size the split-KV partials: a hook that returns 0 makes
 * the shim fall back to one slice, which is correct and several times slower, so nothing else
 * notices. The launch is then checked against a host mirror of the reference (scores and sums in
 * double), with the partials and without (one slice), full causal and under a window.
 * The window is chosen so that its first three keys are the last three of a slice, at any slice
 * count from 2 to 32: the real row sees three keys of that slice and the idle row slots, which sit
 * at later positions, see none, the case in which every lane must still stage the tile. */
static void check_attn_depth() {
    std::printf("== attn_paged decode at depth, by value\n");
#ifndef RAD_HAVE_HIP
    skip("built without HIP: attention is not run");
#else
    using namespace moenum;
    const RadKernelInfo* k = nullptr;
    for (int i = 0; i < rows(); ++i)
        if (row_at(i) && !std::strcmp(row_at(i)->name, "attn_paged_bf16_decode")) k = row_at(i);
    if (!k || !k->launch) { fail("attn_paged_bf16_decode is missing"); return; }
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) { skip("no device visible"); return; }
    const long long hd = 128, nh = 16, gqa = 4, kvh = nh / gqa, bs = 16, ctx = 4096 - 9;
    const long long nb = (ctx + bs - 1) / bs;
    const float scale = 1.0f / std::sqrt((float)hd);
    std::vector<uint16_t> q((size_t)(nh * hd)), kv((size_t)(nb * kvh * bs * 2 * hd));
    uint32_t seed = 0x1234567u;
    auto draw = [&](float amp) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        return amp * ((float)(seed & 0xffff) / 32768.0f - 1.0f);
    };
    for (auto& v : q) v = f32_to_bf16(draw(1.5f));
    std::vector<int32_t> bt((size_t)nb), su = { (int32_t)ctx };
    for (long long b = 0; b < nb; ++b) bt[(size_t)b] = (int32_t)((b * 37 + 5) % nb);   /* a permutation */
    for (long long i = 0; i < (long long)kv.size(); ++i)
        kv[(size_t)i] = f32_to_bf16(draw((i / hd) % 2 ? 2.0f : 1.5f));                 /* K then V a slot */
    const long long edge = (nb / 2) * bs - 3;          /* three keys under a slice boundary */
    for (long long window : { 0LL, ctx - edge }) {
        std::vector<RadParam> p = { pi("q_len", 1), pi("head_dim", hd), pi("gqa", gqa),
                                    pi("block_size", bs), pi("causal", 1), pi("window", window),
                                    pi("n_head", nh), pi("max_seqs", 1), pi("max_ctx", nb * bs),
                                    ps("q_dtype", "bf16"), ps("kv_dtype", "bf16"),
                                    RadParam{ "scale", RAD_P_F64, 0, 0, nullptr, (double)scale } };
        /* The builder's call: params, no tensors. */
        RadArgs pa{ nullptr, 0, p.data(), (int)p.size(), nullptr, 0, nullptr, 0, 1 };
        const long long bytes = k->scratch ? (long long)k->scratch(&pa) : 0;
        const long long need = nh * 8 * 132 * 4;     /* eight slices of one row's partials */
        if (bytes < need)
            fail("attn_paged decode, window %lld: the scratch hook gives %lld bytes from the params "
                 "alone, eight slices need %lld", window, bytes, need);
        else ok("attn_paged decode, window %lld: the scratch hook gives %lld bytes from the params "
                "alone", window, bytes);
        /* The host mirror. */
        std::vector<double> ref((size_t)(nh * hd));
        const long long pos = ctx - 1, j0 = window > 0 && pos - window + 1 > 0 ? pos - window + 1 : 0;
        for (long long h = 0; h < nh; ++h) {
            const long long kh = h / gqa;
            std::vector<double> sc((size_t)ctx, 0.0);
            double mx = -1e300;
            for (long long j = j0; j < ctx; ++j) {
                const uint16_t* slot = kv.data() + ((bt[(size_t)(j / bs)] * kvh + kh) * bs + j % bs) * 2 * hd;
                double d = 0;
                for (long long e = 0; e < hd; ++e)
                    d += (double)bf16_to_f32(q[(size_t)(h * hd + e)]) * (double)bf16_to_f32(slot[e]);
                sc[(size_t)j] = d * (double)scale;
                mx = std::max(mx, sc[(size_t)j]);
            }
            double sum = 0;
            std::vector<double> acc((size_t)hd, 0.0);
            for (long long j = j0; j < ctx; ++j) {
                const uint16_t* slot = kv.data() + ((bt[(size_t)(j / bs)] * kvh + kh) * bs + j % bs) * 2 * hd;
                const double w = std::exp(sc[(size_t)j] - mx);
                sum += w;
                for (long long e = 0; e < hd; ++e) acc[(size_t)e] += w * (double)bf16_to_f32(slot[hd + e]);
            }
            for (long long e = 0; e < hd; ++e) ref[(size_t)(h * hd + e)] = acc[(size_t)e] / sum;
        }
        std::vector<uint16_t> first;
        for (int split : { 1, 0 }) {
            Dev dv;
            RadTensor t[8] = {};
            t[0].data = dv.put(q.data(), q.size() * 2); t[0].dtype = RAD_BF16; t[0].rank = 3;
            t[0].shape[0] = 1; t[0].shape[1] = nh; t[0].shape[2] = hd;
            t[0].stride[0] = nh * hd; t[0].stride[1] = hd; t[0].stride[2] = 1;
            t[7] = t[0]; t[7].data = dv.put(nullptr, q.size() * 2);
            t[1].data = dv.put(kv.data(), kv.size() * 2); t[1].dtype = RAD_BF16; t[1].rank = 4;
            t[1].shape[0] = nb; t[1].shape[1] = kvh; t[1].shape[2] = bs; t[1].shape[3] = 2 * hd;
            t[1].stride[0] = kvh * bs * 2 * hd; t[1].stride[1] = bs * 2 * hd; t[1].stride[2] = 2 * hd;
            t[1].stride[3] = 1;
            t[2] = dense(dv.put(bt.data(), bt.size() * 4), RAD_I32, 1, nb);
            t[3] = vec(dv.put(su.data(), 4), RAD_I32, 1);
            RadArgs a{ t, 8, p.data(), (int)p.size(), nullptr, 0, nullptr, 0, 1 };
            if (split && bytes > 0) {
                a.scratch = dv.put(nullptr, (size_t)bytes);
                a.scratch_bytes = bytes;
            }
            char what[112];
            std::snprintf(what, sizeof(what), "attn_paged decode, context %lld, window %lld, %s", ctx,
                          window, split ? "with the partials" : "one slice (no scratch)");
            const int rc = k->launch(&a, nullptr);
            hipDeviceSynchronize();
            if (rc != 0) { fail("%s: the launch returned %d", what, rc); continue; }
            std::vector<uint16_t> got(q.size());
            hipMemcpy(got.data(), t[7].data, got.size() * 2, hipMemcpyDeviceToHost);
            compare(what, ref, got, 2e-3);
            if (split) first = got;
            else if (!first.empty()) {
                long long far = 0;
                for (size_t i = 0; i < got.size(); ++i) {
                    const double a0 = bf16_to_f32(first[i]), a1 = bf16_to_f32(got[i]);
                    if (!(std::fabs(a0 - a1) <= 0.02 * std::fabs(a1) + 1e-4)) ++far;
                }
                if (far) fail("attn_paged decode, window %lld: %lld values far apart between the split "
                              "launch and one slice", window, far);
                else ok("attn_paged decode, window %lld: the split launch and one slice agree", window);
            }
        }
    }
#endif
}

/* ---- the int8 trunk, by value -------------------------------------------------------------------
 * had_quant_act_i8g and gemm_nt_q at i8a8 against host mirrors of libref's arithmetic
 * (had_quant_act_q8 / q8_group, quant_dot). The quantiser's scale must be amax / 127 of the
 * unnormalised transform and 1 for an all-zero group; a code may sit one step from the mirror's
 * where the device's reciprocal differs in its last place. The GEMM's group term is
 * (sub * a_scale[m, g]) * b_scale[n, g], each product rounded, groups ascending, one bf16 store,
 * and the device's output is compared bit for bit. */
static void check_i8a8_numerics() {
    std::printf("== the int8 trunk (had_quant_act_i8g, gemm_nt_q i8a8), by value\n");
#ifndef RAD_HAVE_HIP
    skip("built without HIP: not run");
#else
    using namespace moenum;
    auto qfn = (Launch)dlsym(g_h, "r3d_rad_had_quant_act_i8g");
    auto gfn = (Launch)dlsym(g_h, "r3d_rad_gemm_nt_q_i8a8");
    if (!qfn || !gfn) { fail("an int8 trunk entry point is missing"); return; }
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) { skip("no device visible"); return; }
    uint32_t st = 0x18A8u;
    auto rnd = [&]() { st = st * 1664525u + 1013904223u; return ((st >> 8) & 0xffffu) / 65536.0f - 0.5f; };

    for (long long n : { 128LL, 2048LL, 7680LL }) {
        const long long M = 3, G = n / 128;
        char what[96];
        std::snprintf(what, sizeof(what), "had_quant_act_i8g, n %lld, group 128, 3 rows", n);
        std::vector<uint16_t> x((size_t)(M * n));
        for (size_t i = 0; i < x.size(); ++i) x[i] = f32_to_bf16(rnd() * ((i % 97) == 0 ? 40.0f : 3.0f));
        if (G > 1) for (long long i = 0; i < 128; ++i) x[(size_t)(n + 128 + i)] = 0;   /* row 1, group 1: all zero */
        std::vector<int8_t> q0((size_t)(M * n), (int8_t)0x55);
        std::vector<float> s0((size_t)(M * G), -1.0f);
        Dev dv;
        RadTensor t[3] = { dense(dv.put(x.data(), x.size() * 2), RAD_BF16, M, n),
                           dense(dv.put(q0.data(), q0.size()), RAD_I8, M, n),
                           dense(dv.put(s0.data(), s0.size() * 4), RAD_F32, M, G) };
        std::vector<RadParam> p = { pi("M", M), pi("n", n), pi("group", 128), ps("dtype", "bf16") };
        RadArgs a{ t, 3, p.data(), (int)p.size(), nullptr, 0, nullptr, 0, 1 };
        const int rc = qfn(&a, nullptr);
        hipDeviceSynchronize();
        if (rc != 0) { fail("%s: the launch returned %d", what, rc); continue; }
        std::vector<int8_t> q(q0.size());
        std::vector<float> sc(s0.size());
        hipMemcpy(q.data(), t[1].data, q.size(), hipMemcpyDeviceToHost);
        hipMemcpy(sc.data(), t[2].data, sc.size() * 4, hipMemcpyDeviceToHost);
        long long off1 = 0, worse = 0, sbad = 0;
        for (long long r = 0; r < M; ++r)
            for (long long g = 0; g < G; ++g) {
                float v[128];
                for (int i = 0; i < 128; ++i) v[i] = bf16_to_f32(x[(size_t)(r * n + g * 128 + i)]);
                for (int h = 1; h < 128; h <<= 1)
                    for (int i = 0; i < 128; i += 2 * h)
                        for (int j = i; j < i + h; ++j) {
                            const float u = v[j], w = v[j + h];
                            v[j] = u + w; v[j + h] = u - w;
                        }
                float amax = 0.0f;
                for (int i = 0; i < 128; ++i) amax = std::max(amax, std::fabs(v[i]));
                const float want = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f, inv = 1.0f / want;
                if (sc[(size_t)(r * G + g)] != want) ++sbad;
                for (int i = 0; i < 128; ++i) {
                    float c = v[i] * inv;
                    c = c > 127.0f ? 127.0f : (c < -127.0f ? -127.0f : c);
                    const long long d = std::llabs((long long)std::nearbyint(c) -
                                                   (long long)q[(size_t)(r * n + g * 128 + i)]);
                    if (d == 1) ++off1; else if (d > 1) ++worse;
                }
            }
        if (sbad == 0 && worse == 0 && off1 * 1000 <= M * n)
            ok("%s: every scale exact, %lld of %lld codes one step off", what, off1, M * n);
        else
            fail("%s: %lld scales differ, %lld codes one step off, %lld further", what, sbad, off1, worse);
    }

    /* gemm_nt_q i8a8, bit for bit. Four host mirrors of an output: the reference's (two rounded
     * multiplies, then the add, groups ascending: quant_dot), a fused one (the add takes the
     * unrounded product, what a v_fmac does), the other scale order ((sub * b_scale) * a_scale)
     * and the reference's terms summed from the last group down. The device must equal the first
     * on every output. Each case counts the outputs the other three would move. Random codes
     * seldom land on a rounding boundary (a handful of outputs in a few hundred thousand), so the
     * rounding and the order are carried by three constructed cells, each run on every kernel
     * (M <= 16, 17 to 127, the tile from 128) and each required to be moved on every output by
     * the mirror it is for: a pass of a cell says the case could have failed. The volatile
     * temporaries keep the host's own compiler from fusing the mirror. */
    auto term = [](float acc, int32_t sub, float sa, float sb, int mode) -> float {
        if (mode == 2) { volatile float u = (float)sub * sb; volatile float q = u * sa; return acc + q; }
        volatile float u = (float)sub * sa;
        if (mode == 1) return std::fma((float)u, sb, acc);
        volatile float q = u * sb;
        return acc + q;
    };
    /* The order cell's scales: group 0 leaves exactly -16129, group 1 adds about +16129, so the
     * sum is the last places of group 1's product. Searched, not typed: a pair is kept only when
     * the fused mirror and the swapped mirror both store a bf16 other than the reference's. */
    float o_sa = 0.0f; uint16_t o_sb = 0; bool o_found = false;
    for (int i = 0; i < 200000 && !o_found; ++i) {
        const uint16_t sbw = f32_to_bf16(0.75f + (rnd() + 0.5f) * 0.75f);
        const float sb = bf16_to_f32(sbw);
        float sa = 1.0f / sb;
        const int steps = (int)std::lrint(rnd() * 16.0f);
        for (int j = 0; j < std::abs(steps); ++j) sa = std::nextafter(sa, steps > 0 ? 4.0f : 0.0f);
        uint16_t r[3];
        for (int mode = 0; mode < 3; ++mode)
            r[mode] = f32_to_bf16(term(term(0.0f, -16129, 1.0f, 1.0f, mode), 16129, sa, sb, mode));
        if (r[0] != r[1] && r[0] != r[2]) { o_sa = sa; o_sb = sbw; o_found = true; }
    }
    if (!o_found) fail("gemm_nt_q i8a8: no scale pair separates the three mirrors (the order cell cannot be built)");

    /* cell: 0 random, 1 fused-add, 2 scale order, 3 group order. nt 0 launches the kernels'
     * plain-load instantiation (the row has no tune axis, so an engine launch is always nt 1). */
    struct G8 { long long M, N, K; int pad; int cell; int nt = 1; };
    for (const G8& c : { G8{ 1, 1, 128, 0, 0 },     G8{ 1, 143, 256, 1, 0 },   G8{ 5, 512, 2048, 0, 0 },
                         G8{ 16, 17, 384, 1, 0, 0 }, G8{ 16, 128, 7680, 0, 0 }, G8{ 17, 127, 256, 0, 0, 0 },
                         G8{ 33, 129, 2048, 1, 0 }, G8{ 127, 257, 128, 0, 0 }, G8{ 128, 143, 384, 1, 0 },
                         G8{ 129, 17, 2048, 0, 0 }, G8{ 257, 129, 256, 1, 0 }, G8{ 300, 144, 384, 0, 0 },
                         G8{ 128, 1, 128, 0, 0 },   G8{ 255, 130, 256, 1, 0, 0 }, G8{ 256, 128, 7680, 0, 0 },
                         G8{ 130, 131, 7680, 1, 0 }, G8{ 513, 255, 384, 0, 0 },
                         G8{ 1, 17, 256, 0, 1 },    G8{ 33, 143, 256, 1, 1 },  G8{ 257, 143, 256, 0, 1 },
                         G8{ 16, 17, 256, 1, 2 },   G8{ 33, 17, 256, 1, 2 },   G8{ 129, 17, 256, 1, 2 },
                         G8{ 5, 17, 384, 0, 3 },    G8{ 33, 17, 384, 1, 3 },   G8{ 129, 143, 384, 1, 3 } }) {
        if (c.cell == 2 && !o_found) continue;
        const long long G = c.K / 128;
        /* a's padded pitch is 12 mod 16: the rows start at every 4-byte phase of a 16-byte load. */
        const long long a_ld = c.K + (c.pad ? 12 : 0), as_ld = G + (c.pad ? 3 : 0);
        const long long bs_ld = G + (c.pad ? 5 : 0), y_ld = c.N + (c.pad ? 7 : 0);
        char what[144];
        std::snprintf(what, sizeof(what), "gemm_nt_q i8a8, M %lld N %lld K %lld%s%s%s", c.M, c.N, c.K,
                      c.pad ? ", padded pitches" : "", c.nt ? "" : ", nt 0",
                      c.cell == 1 ? ", the fused-add cell" : c.cell == 2 ? ", the order cell"
                      : c.cell == 3 ? ", the group-order cell" : "");
        std::vector<int8_t> a8((size_t)(c.M * a_ld), (int8_t)0x55), b8((size_t)(c.N * c.K));
        std::vector<float> as((size_t)(c.M * as_ld), std::nanf(""));
        /* y has a guard row before its first row and one after its last: a store guard off by a
         * row writes there, where the pitch canaries of the M rows do not look. y's canary is a
         * finite word, not the NaN the scale padding holds: a stray store of a value computed from
         * that padding would be the same NaN word and go unseen. */
        const uint16_t kYCanary = 0x5a5au;
        std::vector<uint16_t> bs((size_t)(c.N * bs_ld), 0x7fc0u), y0((size_t)((c.M + 2) * y_ld), kYCanary);
        auto A = [&](long long m, long long k) -> int8_t& { return a8[(size_t)(m * a_ld + k)]; };
        auto B = [&](long long n, long long k) -> int8_t& { return b8[(size_t)(n * c.K + k)]; };
        auto AS = [&](long long m, long long g) -> float& { return as[(size_t)(m * as_ld + g)]; };
        auto BS = [&](long long n, long long g) -> uint16_t& { return bs[(size_t)(n * bs_ld + g)]; };
        if (c.cell == 0) {
            /* Random codes over the whole int8 range, a saturated group on one row and one
             * column (sub = 127 * -128 * 128), an all-zero group, and scales that step 2^6
             * between neighbouring groups, so a scale read from the group beside is far off. */
            auto code = [&]() { const float u = rnd(); return (st & 0x3f00u) == 0 ? (int8_t)-128 : (int8_t)std::lrint(u * 254.0f); };
            for (long long m = 0; m < c.M; ++m) for (long long k = 0; k < c.K; ++k) A(m, k) = code();
            for (long long n = 0; n < c.N; ++n) for (long long k = 0; k < c.K; ++k) B(n, k) = code();
            if (G > 1) for (long long k = 128; k < 256; ++k) { A(0, k) = 127; B(0, k) = -128; }
            if (c.M > 2) for (long long k = 0; k < 128; ++k) A(2, k) = 0;
            for (long long m = 0; m < c.M; ++m) for (long long g = 0; g < G; ++g)
                AS(m, g) = (0.002f + (rnd() + 0.5f) * 0.02f) * ((g & 1) ? 64.0f : 1.0f);
            for (long long n = 0; n < c.N; ++n) for (long long g = 0; g < G; ++g)
                BS(n, g) = f32_to_bf16((0.0005f + (rnd() + 0.5f) * 0.004f) * ((g & 1) ? 1.0f / 64.0f : 1.0f));
        } else if (c.cell == 3) {
            /* Three groups: 2^25, then -2^25, then 1 (64 * 64 = 2^12 with an activation scale of
             * 2^13; the weight's scale 1). Ascending, the first two cancel and 1 is stored (0x3f80);
             * from the last group down, 1 - 2^25 rounds to -2^25 and 0 is stored. */
            for (long long m = 0; m < c.M; ++m) for (long long k = 0; k < c.K; ++k) A(m, k) = 0;
            for (long long m = 0; m < c.M; ++m) {
                A(m, 0) = 64; A(m, 128) = 64; A(m, 256) = 1;
                AS(m, 0) = 8192.0f; AS(m, 1) = 8192.0f; AS(m, 2) = 1.0f;
            }
            for (long long n = 0; n < c.N; ++n) {
                B(n, 0) = 64; B(n, 128) = -64; B(n, 256) = 1;
                for (long long g = 0; g < G; ++g) BS(n, g) = 0x3f80u;
            }
        } else {
            /* One code pair a group. The fused-add cell: 1 from group 0, then 6710989 * 2^-31 * 1.25,
             * whose rounded product is 2^-8 + 2^-24: the sum ties to 1 + 2^-8 and stores 0x3f80, while
             * the unrounded product sits above the tie and stores 0x3f81. The order cell: see o_sa. */
            for (long long m = 0; m < c.M; ++m) for (long long k = 0; k < c.K; ++k) A(m, k) = 0;
            const int8_t a0 = c.cell == 1 ? 1 : 127, b0 = c.cell == 1 ? 1 : -127, b1 = c.cell == 1 ? 1 : 127;
            for (long long m = 0; m < c.M; ++m) {
                A(m, 0) = a0; A(m, 128) = a0;
                AS(m, 0) = 1.0f; AS(m, 1) = c.cell == 1 ? std::ldexp(6710989.0f, -31) : o_sa;
            }
            for (long long n = 0; n < c.N; ++n) {
                B(n, 0) = b0; B(n, 128) = b1;
                BS(n, 0) = 0x3f80u; BS(n, 1) = c.cell == 1 ? (uint16_t)0x3fa0u : o_sb;
            }
        }
        Dev dv;
        auto pitched = [](void* d, uint32_t dt, long long r, long long cols, long long ld) {
            RadTensor t = dense(d, dt, r, cols); t.stride[0] = ld; return t; };
        void* yd = dv.put(y0.data(), y0.size() * 2);
        RadTensor t[7] = { pitched(dv.put(a8.data(), a8.size()), RAD_I8, c.M, c.K, a_ld),
                           pitched(dv.put(as.data(), as.size() * 4), RAD_F32, c.M, G, as_ld),
                           dense(dv.put(b8.data(), b8.size()), RAD_I8, c.N, c.K),
                           pitched(dv.put(bs.data(), bs.size() * 2), RAD_BF16, c.N, G, bs_ld),
                           pitched(yd ? (char*)yd + y_ld * 2 : nullptr, RAD_BF16, c.M, c.N, y_ld),
                           RadTensor{}, RadTensor{} };
        if (!yd) { fail("%s: no device memory for y", what); continue; }
        std::vector<RadParam> p = { pi("M", c.M), pi("N", c.N), pi("K", c.K), pi("group", 128),
                                    ps("dtype", "i8a8") };
        if (!c.nt) p.push_back(pi("nt", 0));
        RadArgs a{ t, 7, p.data(), (int)p.size(), nullptr, 0, nullptr, 0, 1 };
        const int rc = gfn(&a, nullptr);
        hipDeviceSynchronize();
        if (rc != 0) { fail("%s: the launch returned %d", what, rc); continue; }
        std::vector<uint16_t> yg(y0.size());
        hipMemcpy(yg.data(), (char*)t[4].data - y_ld * 2, yg.size() * 2, hipMemcpyDeviceToHost);
        const uint16_t* y = yg.data() + y_ld;
        long long bad = 0, fused = 0, swapped = 0, reversed = 0, canary = 0, guard = 0, bm = -1, bn = -1;
        for (long long i = 0; i < y_ld; ++i) guard += (yg[(size_t)i] != kYCanary) + (y[(size_t)(c.M * y_ld + i)] != kYCanary);
        std::vector<int32_t> subs((size_t)G);
        uint16_t bgot = 0, bwant = 0;
        for (long long m = 0; m < c.M; ++m) {
            for (long long n = 0; n < c.N; ++n) {
                float acc[3] = { 0.0f, 0.0f, 0.0f };
                for (long long g = 0; g < G; ++g) {
                    int32_t sub = 0;
                    for (long long k = g * 128; k < (g + 1) * 128; ++k) sub += (int32_t)A(m, k) * (int32_t)B(n, k);
                    subs[(size_t)g] = sub;
                    for (int mode = 0; mode < 3; ++mode)
                        acc[mode] = term(acc[mode], sub, AS(m, g), bf16_to_f32(BS(n, g)), mode);
                }
                float rev = 0.0f;
                for (long long g = G - 1; g >= 0; --g) rev = term(rev, subs[(size_t)g], AS(m, g), bf16_to_f32(BS(n, g)), 0);
                const uint16_t want = f32_to_bf16(acc[0]), got = y[(size_t)(m * y_ld + n)];
                if (f32_to_bf16(acc[1]) != want) ++fused;
                if (f32_to_bf16(acc[2]) != want) ++swapped;
                if (f32_to_bf16(rev) != want) ++reversed;
                if (got != want) { if (!bad) { bm = m; bn = n; bgot = got; bwant = want; } ++bad; }
            }
            for (long long n = c.N; n < y_ld; ++n) if (y[(size_t)(m * y_ld + n)] != kYCanary) ++canary;
        }
        const long long nout = c.M * c.N;
        if ((c.cell == 1 && fused != nout) || (c.cell == 2 && (fused != nout || swapped != nout)) ||
            (c.cell == 3 && reversed != nout))
            fail("%s: the cell does not discriminate (a fused add moves %lld of %lld outputs, the other "
                 "scale order %lld, the other group order %lld)", what, fused, nout, swapped, reversed);
        else if (bad || canary || guard)
            fail("%s: %lld of %lld outputs differ from the reference (first at m %lld n %lld: 0x%04x, "
                 "want 0x%04x), %lld written past a row's end, %lld in the guard rows", what, bad, nout, bm, bn,
                 bgot, bwant, canary, guard);
        else
            ok("%s: %lld outputs bit-identical to the reference; a fused add would move %lld, the other "
               "scale order %lld, the other group order %lld", what, nout, fused, swapped, reversed);
    }
#endif
}

/* ---- a plain weight the host cannot read -------------------------------------------------------
 * The engine maps weights through its own pool (core/device/aql.cpp), and on an integrated GPU (a
 * Radeon 860M) the host cannot read those pages, where hipMalloc's can be. Until 2026-11-18
 * rad_gemm_run looked for a joined-planes header through every weight's address on the host, so
 * the first prefill GEMM of a bf16 model faulted there, while this selftest, whose weights come
 * from hipMalloc, passed. This case maps the weight with HIP's virtual memory calls and grants
 * access to the device only; the host's read is caught (SIGSEGV) and reported as a FAIL. */
#ifdef RAD_HAVE_HIP
namespace devw {

sigjmp_buf g_jb;
volatile sig_atomic_t g_armed = 0;
void on_segv(int) { if (g_armed) siglongjmp(g_jb, 1); signal(SIGSEGV, SIG_DFL); raise(SIGSEGV); }

/* fn() with SIGSEGV caught: 1 when it faulted, else 0 and *rc its return. */
template <class F> int guarded(F fn, int* rc) {
    struct sigaction sa{}, old{};
    sa.sa_handler = on_segv; sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &old);
    int faulted = 0;
    if (sigsetjmp(g_jb, 1) == 0) { g_armed = 1; *rc = fn(); }
    else faulted = 1;
    g_armed = 0;
    sigaction(SIGSEGV, &old, nullptr);
    return faulted;
}

struct DeviceOnly {
    void* p = nullptr; size_t size = 0; hipMemGenericAllocationHandle_t h{}; int stage = 0;
    ~DeviceOnly() {
        if (stage >= 3) (void)hipMemUnmap(p, size);
        if (stage >= 2) (void)hipMemAddressFree(p, size);
        if (stage >= 1) (void)hipMemRelease(h);
    }
    /* 0, or the failing call's status */
    int map(size_t bytes) {
        hipMemAllocationProp prop{};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice; prop.location.id = 0;
        size_t g = 0;
        if (int e = hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum)) return e;
        if (!g) return -1;
        size = (bytes + g - 1) / g * g;
        if (int e = hipMemCreate(&h, size, &prop, 0)) return e;
        stage = 1;
        if (int e = hipMemAddressReserve(&p, size, 0, nullptr, 0)) return e;
        stage = 2;
        if (int e = hipMemMap(p, size, 0, h, 0)) return e;
        stage = 3;
        hipMemAccessDesc ad{};
        ad.location.type = hipMemLocationTypeDevice; ad.location.id = 0;
        ad.flags = hipMemAccessFlagsProtReadWrite;
        if (int e = hipMemSetAccess(p, size, &ad, 1)) return e;
        return 0;
    }
};

}  // namespace devw
#endif

static void check_gemm_device_weight() {
    std::printf("== gemm_nt_bf16_tiled, a plain weight the host cannot read\n");
#ifndef RAD_HAVE_HIP
    skip("built without HIP: not run");
#else
    using namespace moenum;
    using namespace devw;
    auto fn = (int (*)(const RadArgs*, RadStream))dlsym(g_h, "r3d_rad_gemm_nt_bf16_tiled");
    if (!fn) { fail("r3d_rad_gemm_nt_bf16_tiled is missing"); return; }
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) { skip("no device visible"); return; }
    const long long M = 32, N = 64, K = 128;   /* M >= 17: the tiled row's band */
    uint32_t st = 0x6E3D17u;
    auto rnd = [&](float sc) { st = st * 1664525u + 1013904223u;
                               return f32_to_bf16((((st >> 8) & 0xffffu) / 65536.0f - 0.5f) * sc); };
    std::vector<uint16_t> x((size_t)(M * K)), w((size_t)(N * K)), ynan((size_t)(M * N), 0x7fc0u);
    for (auto& v : x) v = rnd(2.0f);
    for (auto& v : w) v = rnd(1.0f);
    DeviceOnly dw;
    if (int e = dw.map(w.size() * 2)) { skip("HIP's virtual memory calls are not available here (status %d)", e); return; }
    if (hipMemcpy(dw.p, w.data(), w.size() * 2, hipMemcpyHostToDevice) != hipSuccess) {
        fail("copying the weight into device-only memory failed"); return; }
    int probe = 0;
    if (!guarded([&] { return (int)*(volatile const uint8_t*)dw.p; }, &probe)) {
        skip("the host can read device-only memory on this machine, so the case proves nothing here"); return; }
    Dev dv;
    RadTensor t[3] = { dense(dv.put(x.data(), x.size() * 2), RAD_BF16, M, K),
                       dense(dw.p, RAD_BF16, N, K),
                       dense(dv.put(ynan.data(), ynan.size() * 2), RAD_BF16, M, N) };
    std::vector<RadParam> p = { pi("M", M), pi("N", N), pi("K", K), ps("dtype", "bf16") };
    RadArgs a{ t, 3, p.data(), (int)p.size(), nullptr, 0, nullptr, 0, 1 };
    int rc = 0;
    if (guarded([&] { return fn(&a, nullptr); }, &rc)) {
        fail("the shim read the weight's bytes on the host (SIGSEGV)"); return; }
    (void)hipDeviceSynchronize();
    if (rc != 0) { fail("the launch returned %d", rc); return; }
    std::vector<uint16_t> y((size_t)(M * N));
    if (hipMemcpy(y.data(), t[2].data, y.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) { fail("copying y back failed"); return; }
    std::vector<double> ref((size_t)(M * N));
    for (long long m = 0; m < M; ++m)
        for (long long n = 0; n < N; ++n) {
            double s = 0;
            for (long long k = 0; k < K; ++k) s += (double)bf16_to_f32(x[m * K + k]) * bf16_to_f32(w[n * K + k]);
            ref[m * N + n] = s;
        }
    compare("gemm_nt_bf16_tiled, M 32 N 64 K 128, the weight device-only", ref, y, 2e-3);
#endif
}

int main(int argc, char** argv) {
    /* Unbuffered stdout, so when a row dies the last line printed is the last thing that worked,
     * even when output goes to a file or pipe. */
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--case") && i + 1 < argc) g_case = argv[++i];
        else if (!std::strcmp(argv[i], "--help")) {
            std::printf("usage: r3d_selftest <path>/libr3d.so [--case <row>]\n");
            return 0;
        } else if (!path) path = argv[i];
    }
    if (!path) { std::fprintf(stderr, "r3d_selftest: no plugin given\n"); return 2; }
    g_plugin = path;

    /* dlopen, not link, so an undefined symbol in the .so is reported here. */
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "r3d_selftest: %s\n", dlerror()); return 2; }
    g_h = h;
    P_count  = (int (*)(void))dlsym(h, "rad_kernel_count");
    P_at     = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    P_scount = (int (*)(void))dlsym(h, "rad_kernel_schema_count");
    P_sat    = (const RadOpSchema* (*)(int))dlsym(h, "rad_kernel_schema_at");
    if (!P_count || !P_at || !P_scount || !P_sat) {
        std::fprintf(stderr, "r3d_selftest: %s is not a kernel plugin\n", path);
        return 2;
    }

    if (!g_case) {
        check_codecs();
        check_lanes();
        check_table();
        check_reference_schemas();
        check_band_reach();
        check_shapes();
        check_moe_numerics();
        check_conv_numerics();
        check_rope_numerics();
    check_attn_depth();
        check_i8a8_numerics();
        check_gemm_device_weight();

        const int n = rows();
        int passed = 0, failed = 0;
        for (int i = 0; i < n; ++i) {
            const RadKernelInfo* k = row_at(i);
            if (!k || !k->name) continue;
            std::vector<char*> argv2 = { argv[0], const_cast<char*>(path),
                                         const_cast<char*>("--case"),
                                         const_cast<char*>(k->name), nullptr };
            pid_t pid = 0;
            if (posix_spawn(&pid, argv[0], nullptr, nullptr, argv2.data(), environ) != 0) {
                fail("could not spawn a child for '%s': %s", k->name, std::strerror(errno));
                ++failed;
                continue;
            }
            int status = 0;
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
            if (WIFEXITED(status)) {
                if (WEXITSTATUS(status) == 0) ++passed; else ++failed;
            } else if (WIFSIGNALED(status)) {
                std::printf("  FAIL %s died on signal %d\n", k->name, WTERMSIG(status));
                ++failed;
            }
        }
        std::printf("== %d rows ran clean, %d failed, %d skipped by the shims\n", passed, failed,
                    g_skip);
        dlclose(h);
        return (failed || g_fail) ? 1 : 0;
    }

    for (int i = 0; i < rows(); ++i) {
        const RadKernelInfo* k = row_at(i);
        if (k && k->name && !std::strcmp(k->name, g_case)) return run_row(i);
    }
    std::printf("  FAIL no row named '%s'\n", g_case);
    return 1;
}
