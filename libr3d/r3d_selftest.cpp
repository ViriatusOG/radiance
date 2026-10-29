/* r3d_selftest -- dump the libr3d row table, cross-check it, and prove every row links and runs.
 *
 * A transcription error in a constraint table stays invisible until a model silently picks the
 * wrong kernel, and a codec one bit out on a subnormal shows up as a percentage of a perplexity. So
 * the table is printed in full and checked against itself:
 *
 *   - every row's `op` has a schema declared by this plugin, or nothing can select it;
 *   - every constraint key is a declared parameter of that op. libr3d's constraints and libref's
 *     schemas are one vocabulary, and a key renamed in only one place yields a row that matches
 *     nothing;
 *   - names are unique, launches exist, an init has a fini, a tunable's default is one of its
 *     values (the loader refuses a plugin whose default it cannot launch, and it is better to hear
 *     it here), and no row carries a `layout`, `relayout` or `unrelayout` hook: libr3d repacks
 *     nothing (r3d_rows.cpp explains why, and a hook appearing means a container format only this
 *     plugin can read has crept in);
 *   - every `opd_shape` describes as many operands as its schema declares, for a spread of
 *     geometries, with positive extents.
 *
 * The codecs are checked against the ABI's own host versions in `rad_plugin.h` over every bit
 * pattern where the domain is small and a sweep where it is not: bf16 and f16 over all 65536, E4M3
 * over all 256 codes and the encoder over the saturating, subnormal, signed-zero and NaN cases,
 * int8 against `rad_round_even_sat` over every tie in its band. This is the check `rad-kbench`
 * cannot do: it compares whole ops, so a codec's disagreement appears as a diffuse percentage.
 * libr3d's codecs are software -- the lane map, not a missing instruction, is the reason, and its
 * bf16 packing is `v_perm` and not `v_cvt_pk_bf16_f32` -- so they are the plugin's own
 * arithmetic and belong to the tests.
 *
 * Then, on a device, every launch is called with a minimal valid input built from the row's own
 * `opd_shape`, and every output plane is read back and checked to be finite. That catches the class
 * of bug a table check cannot: a null operand the shim does not test, a stride read from the wrong
 * axis, an operand index off by one, an undefined symbol, a kernel that writes where it should not.
 *
 * The numerical oracle is `rad-kbench`, which replays `kernels.rkb` against libref; this program
 * does not duplicate it. What it adds is that the rows are reachable at all -- on this card, from
 * this .so, with the geometry their constraints state.
 *
 * The plugin is dlopened rather than linked, so an undefined symbol in the .so is caught too.
 *
 * One child process per row. A bad pointer handed to a kernel is a GPU page fault, which ROCr
 * reports by printing a line and aborting the process. In one process the first bad row would end
 * the run, and with stdout block-buffered down a pipe the fault line can land next to the wrong
 * row. So the parent runs the static phase and then re-execs itself once per row with
 * `--case <name>`; a child killed by a signal is a FAIL naming the signal, and the run continues.
 * posix_spawn rather than fork, because the parent already has the HIP runtime's threads up and
 * only exec resets them.
 *
 * `--case <name>` is also the reproducer: it runs one row, alone, in the foreground.
 */

#include "rad_abi.h"
#include "rad_plugin.h"
#include "r3d_codecs.h"
#include "rad_planes.h"   /* the joined scale+zero plane, for the row no shape hook can describe */
#include "r3d_plugin.h"   /* the entry points, for the shape hooks that precede their rows */

#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdarg>
#include <cerrno>
#include <cmath>
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

/* The plugin's table, reached through the symbols dlopened in main. The program is not linked
 * against libr3d: an undefined symbol in the .so must be this program's failure, not its own
 * failure to build. */
static int (*P_count)(void) = nullptr;
static const RadKernelInfo* (*P_at)(int) = nullptr;
static int (*P_scount)(void) = nullptr;
static const RadOpSchema* (*P_sat)(int) = nullptr;

static int rows() { return P_count ? P_count() : 0; }
static const RadKernelInfo* row_at(int i) { return P_at ? P_at(i) : nullptr; }
static int schemas() { return P_scount ? P_scount() : 0; }
static const RadOpSchema* schema_at(int i) { return P_sat ? P_sat(i) : nullptr; }

/* The loader's messages are built with a printf into a string; so are these, so a failure reads
 * like the line the engine would have printed. */
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
 * Every check here has a failure mode behind it: a constraint on a key the schema does not declare
 * (the row matches nothing, silently, and the model falls back to a reference kernel), a tunable
 * default outside its axis (the loader refuses the whole plugin), a row whose `op` is spelled for
 * another op's schema (one call meaning two things), a description whose operand count disagrees
 * with its schema (every case for that op skipped, forever).
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
        /* No row repacks a weight: the kernel reads the container's stored order, which is the rule that
 * stops a plugin quietly demanding its own format. The exception is a class with a name list, and
 * the class is the container's quantised weight: codes and scales stored as planes behind one
 * operand, which the mapped tier will not place unless a loaded library can read them in place.
 * Three rows, one per grid the container uses -- w2a8's {1,group} with a zero, the fp8a8 GEMM's
 * {128,128} without, and the lm_head's same form -- each transcribed from the reference's own hook,
 * including the case the reference answers nothing for (ref_layout_gemm_q reads zp_grid, and zp_grid
 * has no fp8a8 entry). What carries the rule is the pair: every one of these names has a `layout`
 * **and** a `relayout`, no row anywhere has an `unrelayout`, and a hook that names a joined layout
 * without the writer that produces it is the failure the engine prints at load. A fifth name is a
 * rule change, and it should fail here. */
        const bool joined = 0 == std::strcmp(k->name, "gemm_nt_q_w2a8_m16")
                         || 0 == std::strcmp(k->name, "logits_gemm_bf16")
                         || 0 == std::strcmp(k->name, "gemm_nt_q_fp8a8_tiled")
                         || 0 == std::strcmp(k->name, "gemm_nt_q_w2a8_tiled");
        /* The query and the writer come as a pair, and the pair is the rule: a row with a hook must
         * supply the relayout that produces the form the hook names (abi/rad_abi.h -- a query
         * answering RAD_OK is a promise the loader keeps on the row's word), and no row may carry an
         * unrelayout, because nothing in this library invents a format a tool would need to invert.
         * A hook with a null writer is the failure the engine reports at load -- `could not produce
         * 'output.weight' (rad.planes) from fp8_e4m3*bf16[128x128]: unsupported` -- and a relayout
         * without a hook is a row repacking a weight nobody asked to have repacked. */
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
                /* The selector compares a string constraint against a string parameter and an
                 * integer one against an integer. A mismatch matches nothing. */
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
 * The loader refuses a plugin whose schema for an op differs from the one the first declarer fixed
 * (core/plugin/loader.cpp, `schema_same`), and it refuses the *whole* .so, every row with it. The
 * comparison is positional and narrow: parameter count, each parameter's key, type and
 * requiredness; operand count, each operand's role and optionality. Operand names and doc strings
 * are free, because arguments are positional and names are for diagnostics.
 *
 * So a schema copied from libref with one operand's role mistyped is a plugin that loads in
 * r3d_selftest -- which does not open libref -- and disappears from the engine, with one line in a
 * log most people never read. This check is the loader's own comparison, run here, naming the op
 * and the slot.
 */
static void check_reference_schemas() {
    std::printf("== schemas against the reference\n");

    /* The reference library beside the plugin under test, the way the engine's home holds them. */
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
            /* Not a refusal: an op the reference does not serve is an op with nothing to agree
             * with, and it is the reason libr3d's rows for a chain libref has no oracle for are
             * checked by the selftest's own arithmetic instead. */
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
 * A row is chosen by matching its constraints against a band's geometry, and the engine's rule
 * (core/util/rad_util.cpp, constraint_holds) is that a constraint on a key the geometry lacks does
 * not hold, in its own words "a row constraining an optional parameter serves only callers that
 * state it, and others get a row without it". The bands an architecture emits carry the parameters
 * its graph set, and nothing else, so a row that constrains an optional parameter is a row narrowed
 * to the graphs that spell that parameter out.
 *
 * Nothing in kbench shows this: a fixture case names every parameter the op declares, so the
 * constraint is tested against a geometry that has it. A server shows all of it, at startup, as
 * "N device band(s) resolved to no kernel" and a model that will not start. So the check is here:
 * a constraint on an optional parameter is not an error -- it is how a row says "I serve the form
 * that states the parameter" -- but it is a decision, and a row whose author did not make it is a
 * row that will silently not serve.
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
        if (!s) continue;                       /* the table phase already failed this row */
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
 * Against the schemas, for one geometry per op taken from the widths a model really has: 4096
 * hidden, 128 head, 11008 intermediate, 128256 vocabulary. A description that returns a count the
 * schema disagrees with, a rank of zero, or an extent of zero is a bug in the description or in the
 * schema; rad-kbench reports it as a skipped case for every geometry, which is how a broken hook
 * hides for years.
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
        { "kv_store", { PI("M", 8), PI("head_dim", 128), PI("n_head_kv", 8), PI("block_size", 32),
                        PS("kv_dtype", "bf16") } },
        { "embed_lookup", { PI("M", 8), PI("n_embd", 4096), PI("n_vocab", 128256),
                            PS("dtype", "bf16") } },
        { "gather_rows", { PI("M", 8), PI("n", 4096), PS("dtype", "bf16") } },
        { "scatter_rows", { PI("M", 8), PI("n", 4096), PS("dtype", "bf16") } },
        { "quant_act_fp8", { PI("M", 8), PI("n", 4096), PI("group", 128), PS("dtype", "bf16") } },
        { "quant_act_i8", { PI("M", 8), PI("n", 4096), PI("group", 128), PS("dtype", "bf16") } },
        { "quant_act_i8g", { PI("M", 8), PI("n", 4096), PI("group", 128), PS("dtype", "bf16") } },
        { "gated_quant_fp8", { PI("M", 8), PI("n", 11008), PI("group", 128), PS("dtype", "bf16"),
                               PS("act", "silu") } },
        { "dequant", { PI("M", 11008), PI("n", 4096), PI("group", 128),
                       PS("dtype", "fp8_e4m3") } },
        { "rmsnorm_quant_fp8", { PI("M", 1792), PI("n", 5120), PI("eps", 1e-5), PI("group", 128),
                                 PS("dtype", "bf16") } },
        { "gemm_nt", { PI("M", 8), PI("N", 11008), PI("K", 4096), PS("dtype", "bf16") } },
        { "gemm_nt_q", { PI("M", 8), PI("N", 11008), PI("K", 4096), PI("group", 128),
                         PS("dtype", "fp8a16"), PI("pb1", 128), PI("pb2", 128) } },
        /* The rest are the bands MiniCPM5-2B asks for, in /home/sysadmin/libr3d/bands.txt: the
         * model's own widths, not invented ones. The vocab is 130560, the MLP is 6144 wide, the
         * head is 128 with 16 heads over 2 KV heads at 16-token blocks. */
        { "silu_mul", { PI("M", 1792), PI("n", 6144), PS("dtype", "bf16") } },
        { "attn_paged", { PI("q_len", 2048), PI("head_dim", 128), PI("gqa", 8), PI("block_size", 16),
                          PI("causal", 1), PI("window", 0), PS("q_dtype", "bf16"),
                          PS("kv_dtype", "bf16"), PI("n_head", 16), PI("max_ctx", 4096),
                          PI("max_seqs", 256) } },
        { "attn_paged", { PI("q_len", 7), PI("head_dim", 128), PI("gqa", 8), PI("block_size", 16),
                          PI("causal", 0), PI("window", 0), PS("q_dtype", "bf16"),
                          PS("kv_dtype", "fp8_e4m3"), PI("n_head", 16), PI("max_ctx", 4096),
                          PI("max_seqs", 256) } },
        /* The decode band with the context deeper than the query chunk: the walk has to cover keys
         * that precede the chunk, which is the only thing in the op a q_len-sized cache cannot
         * show. The second is the geometry that found the tile-sharing bug -- a window that starts
         * mid-block, over an fp8 cache. */
        { "attn_paged", { PI("q_len", 7), PI("head_dim", 128), PI("gqa", 8), PI("block_size", 16),
                          PI("causal", 0), PI("window", 0), PS("q_dtype", "bf16"),
                          PS("kv_dtype", "bf16"), PI("n_head", 16), PI("ctx", 4096),
                          PI("max_ctx", 4096) } },
        { "attn_paged", { PI("q_len", 7), PI("head_dim", 128), PI("gqa", 8), PI("block_size", 16),
                          PI("causal", 1), PI("window", 37), PS("q_dtype", "bf16"),
                          PS("kv_dtype", "fp8_e4m3"), PI("n_head", 16), PI("ctx", 512),
                          PI("max_ctx", 4096) } },
        /* The 27B's sliding-window band, in the shape the fixture draws it: the shape hook describes
         * the descales as ABSENT, so this is the case where an fp8 cache has no scale plane and the
         * reference reads 1.0. It is the geometry kbench caught a refusal on. */
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
        /* The speculative decoder's band, as MiniCPM5-2B declares it: a 7-token draft block, 16
         * candidates a step, a rank-256 codebook over a 130560-token vocabulary, anchors strided 7
         * apart. The two codebooks are 63.8 MiB each, which is what the op really is. */
        { "dflash_select", { PI("M", 256), PI("steps", 7), PI("top_k", 16), PI("rank", 256),
                             PI("n_vocab", 130560), PI("anchor_stride", 7), PS("dtype", "bf16") } },
    };
#undef PI
#undef PF
#undef PS

    /* The descriptions of the ops whose kernels have not landed: the schemas and the hooks are in
     * the library, checked the moment it loads, so a transcription typo is found here and not on the
     * day the row lands and kbench skips its cases -- "describes an output differently" is one line
     * in a report, and a row nobody exercises is a row nobody tested. */
    /* By symbol name, dlsym'd from the plugin: this program links nothing from the library under
     * test -- that is the point of dlopening it -- so a hook is a name and a pointer out of the
     * handle, exactly as the rows are. */
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
        if (!row) std::printf("  --   '%s': the hook, ahead of its kernel\n", gm.op);

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

/* ================================================================== the codecs
 *
 * Against the ABI's host implementations. Exactness is the claim, and it is the claim because the
 * widenings are exact (seven mantissa bits into ten, four exponent and three mantissa bits into
 * five and ten) and the encoders are mirrors of `rad_plugin.h`, line for line.
 */
/* Two floats are the same when their bits are: `a == b` is false for two NaNs, and a codec that
 * quiet a NaN the same way the ABI's does is a codec that agrees. */
static bool same_float(float a, float b) {
    return bits_of(a) == bits_of(b);
}

/* ================================================================== the codecs
 *
 * Against the ABI's host implementations. Exactness is the claim, and it is the claim because the
 * widenings are exact (seven mantissa bits into ten, four exponent and three mantissa bits into
 * five and ten) and the encoders are mirrors of `rad_plugin.h`, line for line.
 */
static void check_codecs() {
    std::printf("== codecs\n");

    /* bf16: every one of the 65536 patterns, both directions. */
    int bad = 0;
    for (uint32_t u = 0; u < 0x10000 && !bad; ++u) {
        const uint16_t b = (uint16_t)u;
        if (!same_float(bf16_to_f32(b), rad_bf16_to_f32(b))) { ++bad; break; }
        /* Encoder against encoder: a NaN's payload is quieted on the way out by both, so a
         * round-trip to the original pattern is not the claim. */
        if (f32_to_bf16(bf16_to_f32(b)) != rad_f32_to_bf16(rad_bf16_to_f32(b))) { ++bad; break; }
    }
    if (bad) fail("bf16: the codecs disagree with rad_plugin.h");
    else ok("bf16: 65536 patterns, decode and encode, match rad_plugin.h bit for bit");

    /* f16: the same, subnormals included -- the narrowing keeps them, which the ABI's codec does
     * and a naive shift-and-round does not. */
    bad = 0;
    for (uint32_t u = 0; u < 0x10000 && !bad; ++u) {
        const uint16_t h = (uint16_t)u;
        const float f = f16_to_f32(h);
        if (!same_float(f, rad_f16_to_f32(h))) { ++bad; break; }
        if (f32_to_f16(f) != rad_f32_to_f16(rad_f16_to_f32(h))) { ++bad; break; }
    }
    if (bad) fail("f16: the codecs disagree with rad_plugin.h");
    else ok("f16: 65536 patterns, both directions, match rad_plugin.h (subnormals included)");

    /* A bf16 pair widened to an f16 pair, in one register, exactly -- the reason the dot2 path is
     * sound. Exact over f16's normal range [2^-14, 65504], as r3d_codecs.h states, and the sweep is
     * over the patterns in it. Outside, the widening is not exact and is not claimed to be: above
     * 65504 it saturates, and below 2^-14 an f16 subnormal's grid is coarser than bf16's, so the
     * packing instruction truncates toward zero. Those patterns are counted and named in the
     * message, because the number of them -- and the fact that they are the format's range, not a
     * bug -- is the thing a reader needs. */
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

    /* E4M3, both directions, over every code. The widening to f16 is exact for every code: e4m3 has
     * four exponent and three mantissa bits, f16 five and ten, and 448 is well inside f16. */
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

    /* The encoder over the cases that break a naive one: saturation at 448, the subnormals below
     * 2^-6, the sign of zero, NaN, and the values a scale lands on. */
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

    /* dot2's order. The device instruction folds the products with guard bits, so the host path is
     * the reference's arithmetic and this checks only that the pairs are read low-then-high: the
     * GEMMs' accumulation order is checked by rad-kbench against libref. */
    float acc = 0.0f;
    const uint32_t a = f16x2_pack(1.5f, -2.5f), b = f16x2_pack(4.0f, 4.0f);
    acc = dot2_f16x2(a, b, acc);
    if (acc != 1.5f * 4.0f + -2.5f * 4.0f) fail("dot2: the pair is not read low-then-high");
    else ok("dot2: the pair is read low-then-high, into the accumulator it is handed");

    /* The bf16 store a kernel's epilogue uses: two values, one dword, round to nearest even, the
     * high half in the high half. A `v_perm` picking the wrong bytes shows up here and not in a
     * kernel's output, where a swapped pair looks like a plausible tensor. */
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
    std::vector<void*> host;       /* pinned operands: read on the host, read by the kernel too */
    std::vector<std::string> words;   /* the strings out.sval points into; reserved in make_params */
    int n_p = 0;
};

/* ================================================================== the lane helpers, on the card
 *
 * Every wave-sum in the library is five rounds of a butterfly over lane shuffles, and the
 * arithmetic that rides on a shuffle is invisible to a codec sweep: `d += (double)f((unsigned long
 * long)d)` moves a *value* where a bit pattern is wanted, and a wave of fractions then sums to 1 --
 * which is a bug this file's own predecessor shipped, in the logprobs sum, and which showed up as
 * 40% of a row's probability mass missing at one shape. So the card runs the butterflies through
 * `r3d_lanes_check` (r3d_sample_pre_f32.hip) and every lane of every wave is compared with a host
 * emulation of the same five rounds, in the same order.
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

/* Fill a host plane per its description. "Normals" is a sum of four uniforms: a kernel's
 * arithmetic is exercised by any value that is not a special case, and the special cases are what
 * the hand-written cases and rad-kbench's sweeps are for. */
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
        /* Two signed codes a byte, the low nibble first: both grids' -7..7 draw, stored in
         * two's complement, which is the low four bits. */
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
        /* Two e2m1 codes a byte, drawn independently: all 16 values, so the unpack's magnitude
         * and sign paths are both exercised. */
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

/* The geometry a row is run at: the band's bounds where the row constrains them, the widths a
 * model has elsewhere. Reading the row's own constraints is what makes this a test of the
 * constraint table and not of a geometry the table happens to admit. */
static void make_params(Case& c, const RadKernelInfo* k, const RadOpSchema* s) {
    c.words.reserve(32);   /* more parameters than any op declares, so no string ever moves */
    for (int q = 0; q < s->n_params && c.n_p < 32; ++q) {
        const RadParamSpec& sp = s->params[q];
        /* Only the parameters the geometry has to carry. The engine passes the graph's parameters
         * with their ranges collapsed, and an optional parameter the graph did not set is simply
         * not in the list: the kernel's own default is then the one libref's `p_*` default is. A
         * selftest that filled every optional slot with a value would be testing a geometry no
         * graph produces -- and for rope, whose optional `sections` string switches the position
         * plane to one row per component, it would be testing a mode the row's constraints
         * exclude. Constrained parameters are always passed: the constraint states the row's band,
         * so its value is the row's own claim. */
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
        /* A constraint's own value wins: it is the band the row is compiled for, and C_IN spells
         * the whole list. A list takes its first word. */
        for (int ci = 0; ci < k->n_constraints; ++ci) {
            const RadConstraint& x = k->constraints[ci];
            if (!x.key || !sp.key || std::strcmp(x.key, sp.key)) continue;
            if (x.op == RAD_C_EQ && sp.type == RAD_P_STR && x.sval) out.sval = x.sval;
            if ((x.op == RAD_C_EQ || x.op == RAD_C_GE) && sp.type == RAD_P_INT) out.ival = x.ival;
            if (x.op == RAD_C_IN && sp.type == RAD_P_STR && x.sval && x.sval[0]) {
                /* The band's first word. The words outlive the parameter list, so they live in a
                 * vector that is reserved once: a string built per constraint and pointed at from
                 * out.sval is a pointer to storage the next constraint reuses, and a row with two
                 * string constraints reads the second one's value into the first's key. */
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
 * `gemm_q_shape` declines w2a8 for a good reason: the container stores the weight's scale and zero
 * as two planes behind the one `b_scale` operand, and a RadOpdDesc names one dtype and one extent
 * set, so it cannot describe that. A hook that named only the byte count would have the tool fill a
 * container header with random bytes, and the kernel's correct refusal would be reported as a bug.
 * So the row's `opd_shape` is null, kbench names the skip, and this is the row's coverage: the band
 * MiniCPM5-2B asks for, with the joined plane built by rad_planes_relayout exactly as
 * r3d_gemm_nt_q_w2a8_driver.cpp builds it, so the card test and the driver cannot drift apart. */
static int build_w2a8(Case& c, const RadOpSchema* s) {
#define W_PI(k, v) { (k), RAD_P_INT, (long long)(v), 0, nullptr, 0.0 }
#define W_PS(k, v) { (k), RAD_P_STR, 0, 0, (v), 0.0 }
    const long long M = 16, N = 130560, K = 2048, group = 128, G = K / group;
    if (s->n_operands != 7) { fail("w2a8: the schema has %d operands, not 7", s->n_operands); return 1; }
    c.n_p = 0;                     /* the band, not the constraint table's minimums */
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

    /* Slots in the schema's order: a, a_scale, b, b_scale (the joined blob), y, a_sum. y is the
     * output and it is pre-filled with a bf16 NaN: a kernel that never launches leaves its status
     * RAD_OK behind and the caller's bytes in the plane, and a plane of the caller's own zeros is
     * finite, so only a NaN fill says the row ran. */
    /* Schema order: a, a_scale (optional), b, b_scale, y, a_sum (optional), b_ref (an absent
     * weight: a null tensor is what an omitted optional means to the shim). */
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
        /* The joined operand stays on the host: the shim reads its header where it lies, which is
         * the mapped tier the engine puts a joined weight on, and a device pointer there is a null
         * header and a refused geometry. The driver hands it over the same way. The allocation is
         * left standing -- this process is short-lived and one blob per run. */
        void* d = nullptr;
        if (o == 3) {
            /* Pinned host memory, the mapped tier's shape: the shim reads this operand's header on
             * the host, the kernel reads the same bytes as a device pointer, and plain malloc is
             * not device-addressable -- it faults with 700. hipHostMalloc is valid on both sides,
             * which is what the engine's container mapping gives a joined weight. */
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

static int run_row(int i) {
    const RadKernelInfo* k = row_at(i);
    if (!k) return 1;
    std::printf("== %s (%s)\n", k->name, k->op);
#ifndef RAD_HAVE_HIP
    (void)i;
    skip("built without the HIP runtime: no device run");
    return 0;
#else
    /* No card is not a failure: the static phase above is a third of what this program checks, and
     * a build machine, a container and the CI job that builds the plugin all run it without one. */
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
        skip("no device visible");
        return 0;
    }
    const int hand = k->opd_shape ? 0
        : (0 == std::strcmp(k->name, "gemm_nt_q_w2a8_m16")) ? 1 : 2;
    if (hand == 2) { skip("no operand description, so no buffers to build"); return 0; }
    const RadOpSchema* s = find_schema(k->op);
    if (!s) { fail("no schema for '%s'", k->op); return 1; }

    Case c;
    make_params(c, k, s);
    RadArgs a{};
    a.p = c.p;
    a.n_p = c.n_p;

    if (hand) { if (build_w2a8(c, s)) return 1; goto launched; }
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
        /* Standing to-do, measured and not shipped: filling every OUT plane with the dtype's NaN --
         * which is what says a row that returns RAD_OK without launching has left the caller's bytes
         * in the plane, the failure the attention unit's third bug is -- works for 66 of the 68 rows
         * and, in the forked child, kills `quant_act_i8_bf16` and `sample_logprobs_f32` on signal 11
         * while both run clean in-process (`--case`), and both write planes the reference pads. It
         * is the right check, it needs the padded-plane story finished, and the hand-built geometry
         * below already does it for the row whose coverage is this file. */
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
        /* The band, spelled out: a shim's mapped code says what kind of answer it gave, not what
         * it was given, and the difference between a launch that failed and a geometry nobody
         * intended is in the numbers. */
        for (int i = 0; i < c.n_p; ++i)
            if (c.p[i].kind == RAD_P_STR)
                std::printf("   param %s = %s\n", c.p[i].key, c.p[i].sval ? c.p[i].sval : "(null)");
            else
                std::printf("   param %s = %lld\n", c.p[i].key, c.p[i].ival);
        for (int o = 0; o < s->n_operands; ++o) {
            const RadTensor& t = c.t[o];
            std::printf("   operand %d '%s' dt=%u rank=%u data=%p", o, s->operands[o].name,
                        t.dtype, t.rank, t.data);
            for (uint32_t r = 0; r < t.rank; ++r) std::printf(" %lld/%lld", t.shape[r], t.stride[r]);
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
            /* A joined plane lives where the mapped tier puts it: pinned host bytes the shim reads
             * on the host and the kernel reads over PCIe. It is not a device buffer, so there is
             * nothing to copy back, and its contents are the container's, not the row's work. */
            if (std::find(c.host.begin(), c.host.end(), t.data) != c.host.end()) continue;
            ++outs;
            long long nel = 1;
            for (uint32_t r = 0; r < t.rank; ++r) nel *= t.shape[r];
            /* Sub-byte planes are counted in codes, not bytes: a U2 tensor of 267M codes is
             * 66.8 MB on the card, and asking the runtime for 267 MB of it is an out-of-range
             * copy, not a large one. */
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
            /* An infinity is the sampler's contract, not a broken number: a dropped candidate is
             * `cand_idx` -1 and `cand_val` -inf, a narrowing stage drops by writing -inf, the mask
             * and the ban write -inf into the logits, and a slot row_topk cannot fill is -inf. A
             * NaN is nobody's contract, so it stays a failure everywhere. */
            const bool sentinel = (k->family && 0 == std::strcmp(k->family, "sample")) ||
                                  (k->op && 0 == std::strcmp(k->op, "row_topk"));
            /* Two planes carry the reference's grid rather than the row's work: `asum`, the int8
             * activation's row sums padded to a 16-row grid, and `stats`, the quantiser's per-group
             * statistics. The row writes the rows it was given and leaves the pad, and with outputs
             * pre-filled NaN the pad is NaN by construction. Named planes only -- NaN anywhere else
             * is exactly the silent no-op the fill exists to find. */
            const bool padded = s->operands[o].name &&
                (0 == std::strcmp(s->operands[o].name, "asum") ||
                 0 == std::strcmp(s->operands[o].name, "stats"));
            if (nan && padded) {
                ok("ran, %d output plane%s read back with %d pad NaN of %lld on '%s' (the "
                   "reference's grid, not the row's work)", outs, outs == 1 ? "" : "s", nan, nan == 1
                   ? "" : "s", nel, s->operands[o].name);
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
    }

    if (scratch) (void)hipFree(scratch);
    for (void* p : c.dev) (void)hipFree(p);
    return ret;
#endif
}

/* ================================================================== main */
int main(int argc, char** argv) {
    /* A row that dies takes the line it was printing with it, and knowing which row died is the
     * whole point of the census: stdout is unbuffered, so the last line is the last thing that
     * worked. Redirecting the output to a file is exactly when the buffering would hide it. */
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

    /* dlopen, not link: an undefined symbol in the .so must be this program's failure. */
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
