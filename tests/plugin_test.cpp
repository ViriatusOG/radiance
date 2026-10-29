/* plugin_test.cpp -- the plugin layer: loading, the hierarchy, op schemas, selection, bucket
 * tables, the selection table and per-instance init (spec.md §2 and §16).
 *
 * The test plugins in core/plugin/testplugin/ are real .so files, so the loader is tested through
 * dlopen, and their rows make the spec's claims assertable (§2.2's worked example band for band).
 * A passing run prints error and warning lines: they are the diagnostics under test.
 */
#include "rad_test.h"

#include "plugin/registry.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"

#include <nlohmann/json.hpp>

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <vector>

using namespace rad;

/* ------------------------------------------------------------------ fixtures */
namespace {

std::string kernels_dir() { return home() + "/testkernels"; }
std::string bad_dir()     { return home() + "/testkernels/bad"; }
std::string order_dir()   { return home() + "/testkernels/order"; }
std::string arch_dir()    { return home() + "/testarch"; }

/* The hierarchy every test but the hierarchy tests uses: the override first, the base behind it. */
int load(Registry& r, std::vector<std::string> hierarchy = { "radtest_over", "radtest_base" }) {
    const int st = r.load_dir(kernels_dir(), hierarchy);
    if (st == RAD_E_IO)
        fprintf(stderr, "    (no test plugins in %s -- build the radtest_* targets)\n",
                kernels_dir().c_str());
    return st;
}

std::string errs(const Registry& r) {
    std::string s;
    for (const auto& e : r.load_errors()) { s += e; s += '\n'; }
    return s;
}

Geometry gemm_geom(long long M, long long K = 4096, const char* dtype = "bf16") {
    Geometry g;
    g.set_i("M", M);
    g.set_i("N", 4096);
    g.set_i("K", K);
    g.set_s("dtype", dtype);
    return g;
}

/* A t_attn geometry every constraint of t_attn_paged holds for, without the derived block_size,
 * as a real caller asks (block_size is what it wants to find out). The operator tests break one
 * key at a time, so the first unmet constraint is the one under test. */
Geometry attn_geom() {
    Geometry g;
    g.set_i("q_len", 8);
    g.set_i("head_dim", 128);
    g.set_i("window", 0);
    g.set_i("causal", 1);
    g.set_i("n_head", 8);
    g.set_s("kv_dtype", "bf16");
    return g;
}

/* The same question with the derived key pinned by the caller. */
Geometry attn_geom_pinned(long long block_size) {
    Geometry g = attn_geom();
    g.set_i("block_size", block_size);
    return g;
}

std::string picked(const KernelRow* k) { return k ? k->info->name : std::string("<none>"); }

/* Each band as "hi:row " with its device row, for comparing a whole band table at once. */
std::string band_names(const std::vector<Band>& b) {
    std::string s;
    for (const Band& x : b)
        s += fmt("%lld:%s ", (long long)x.hi, picked(x.dom[RAD_DOMAIN_DEVICE].row).c_str());
    return s;
}

/* The shipped libr4d.so, or "" for the case to leave. A build without HIP makes none, which skips;
 * a HIP build always makes it, so there its absence is a missing in-tree artifact. */
std::string libr4d() {
    const std::string so = home() + "/kernels/libr4d.so";
    if (access(so.c_str(), R_OK) == 0) return so;
#ifdef RAD_HAVE_HIP
    radtest::artifact_missing(__FILE__, __LINE__, so);
#else
    radtest::skip("no %s (a build without HIP)", so.c_str());
#endif
    return "";
}

/* The reason select() gives for a refusal, which is the miss column of §16's table. */
std::string refusal(Registry& r, const char* op, const Geometry& g, int domain = RAD_DOMAIN_DEVICE) {
    SelectionTrace t;
    const KernelRow* k = r.select(op, g, domain, &t);
    return k ? std::string("<resolved to ") + k->info->name + ">" : t.reason;
}

RadParam pint(const char* k, long long v)   { return RadParam{ k, RAD_P_INT, v, 0, nullptr, 0.0 }; }
RadParam pstr(const char* k, const char* v) { return RadParam{ k, RAD_P_STR, 0, 0, v, 0.0 }; }
RadParam pf64(const char* k, double v)      { return RadParam{ k, RAD_P_F64, 0, 0, nullptr, v }; }
RadParam prange(const char* k, long long lo, long long hi) {
    return RadParam{ k, RAD_P_RANGE, lo, hi, nullptr, 0.0 };
}

}  /* namespace */

/* ================================================================== loading and the hierarchy */

TEST(loading_reads_identity_and_calls_open) {
    Registry r;
    CHECK_OK(load(r));
    CHECK_EQ(r.plugins().size(), (size_t)2);

    const Plugin* base = r.plugin("radtest_base");
    CHECK(base != nullptr);
    if (!base) return;
    CHECK_EQ((int)base->kind, RAD_PLUGIN_KERNEL);
    CHECK_EQ(base->version, "0.1");
    CHECK_EQ(base->n_kernels, 17);
    CHECK_EQ(base->n_schemas, 10);

    /* The optional open hook runs once, after load and before any selection. */
    auto open_calls = reinterpret_cast<int (*)(void)>(dlsym(base->handle, "radtest_open_calls"));
    CHECK(open_calls != nullptr);
    if (open_calls) CHECK(open_calls() >= 1);
}

/* The hierarchy is configured, not discovered: a named plugin takes its position, and a kernel
 * library on disk but absent from the list is not loaded. With no list, everything loads. */
TEST(hierarchy_order_is_configured) {
    Registry r;
    CHECK_OK(r.load_dir(kernels_dir(), { "radtest_base" }));
    const Plugin* base = r.plugin("radtest_base");
    CHECK(base != nullptr);
    if (!base) return;
    CHECK_EQ(base->order, 0);
    CHECK(r.plugin("radtest_over") == nullptr);   /* unnamed, so not loaded */
    for (const KernelRow& row : r.rows()) CHECK(row.plugin != "radtest_over");

    Registry all;
    CHECK_OK(all.load_dir(kernels_dir(), {}));
    CHECK(all.plugin("radtest_base") && all.plugin("radtest_over"));

    Registry r2;
    CHECK_OK(r2.load_dir(kernels_dir(), { "radtest_over", "radtest_base" }));
    CHECK_EQ(r2.plugin("radtest_over")->order, 0);
    CHECK_EQ(r2.plugin("radtest_base")->order, 1);
}

/* With nothing named, the reference library still sorts last: it declares every op, so anything
 * ranked behind it could never be selected and the model would silently run the naive kernels.
 * Alphabetical order would get this wrong: `radtest_ord_a.so` is the reference and is staged
 * first. Both plugins declare the same schemas, which is not a conflict. */
TEST(the_reference_library_sorts_last_with_no_hierarchy) {
    Registry r;
    const int st = r.load_dir(order_dir(), {});
    if (st == RAD_E_IO) {
        fprintf(stderr, "    (no order fixture in %s -- build the radtest_ord_* targets)\n",
                order_dir().c_str());
        return;
    }
    CHECK_OK(st);
    const Plugin* ref   = r.plugin("radtest_reference");
    const Plugin* other = r.plugin("radtest_base");
    CHECK(ref && other);
    if (!ref || !other) return;
    CHECK(ref->reference);
    CHECK(!other->reference);
    CHECK(other->order < ref->order);

    /* Naming it is still an override: an operator comparing a library against the reference may
     * put the reference first on purpose. */
    Registry r2;
    CHECK_OK(r2.load_dir(order_dir(), { "radtest_reference", "radtest_base" }));
    CHECK_EQ(r2.plugin("radtest_reference")->order, 0);
    CHECK(r2.plugin("radtest_base")->order > 0);

    /* A list that leaves it out leaves it out: the reference is a library like any other. */
    Registry r3;
    CHECK_OK(r3.load_dir(order_dir(), { "radtest_base" }));
    CHECK(r3.plugin("radtest_reference") == nullptr);
}

TEST(abi_mismatch_is_refused_by_name_with_both_versions) {
    Registry r;
    const int st = r.load_plugin(bad_dir() + "/radtest_badabi.so", 0);
    CHECK_EQ(st, RAD_E_ABI);
    CHECK_EQ(r.plugins().size(), (size_t)0);
    const std::string e = errs(r);
    CHECK(has(e, "radtest_badabi.so"));
    CHECK(has(e, fmt("ABI version %u", (unsigned)RAD_ABI_VERSION + 7u)));
    CHECK(has(e, fmt("this engine is version %u", (unsigned)RAD_ABI_VERSION)));
}

/* Two disagreeing schemas would make the same positional call mean different things depending on
 * which plugin won selection. Refused at load, with both plugins named. */
TEST(schema_conflict_is_refused_with_both_plugins_named) {
    Registry r;
    CHECK_OK(load(r));
    const size_t rows_before = r.rows().size();
    const Plugin* owner = r.schema_owner("t_gemm");
    CHECK(owner != nullptr);

    const int st = r.load_plugin(bad_dir() + "/radtest_conflict.so", 9);
    CHECK_EQ(st, RAD_E_DUPLICATE);
    CHECK(r.plugin("radtest_conflict") == nullptr);
    CHECK_EQ(r.rows().size(), rows_before);        /* nothing of a refused plugin is registered */

    const std::string e = errs(r);
    CHECK(has(e, "t_gemm"));
    CHECK(has(e, "radtest_conflict"));
    if (owner) CHECK(has(e, owner->name));

    /* Its priority-1000 row would have won every t_gemm question had it registered. */
    CHECK_EQ(picked(r.select("t_gemm", gemm_geom(8), RAD_DOMAIN_DEVICE)), "t_gemm_m16");
}

/* An architecture plugin is loaded and identified here; the build component drives it. One plugin
 * implements one architecture id and no two may claim the same one, since the id selects the
 * plugin a model is built from. */
TEST(architecture_plugins_are_identified_and_deduplicated) {
    Registry r;
    const int st = r.load_dir(arch_dir(), {});
    CHECK_EQ(st, RAD_E_DUPLICATE);
    CHECK_EQ(r.plugins().size(), (size_t)1);

    const Plugin* a = r.plugin("radtest_arch");
    CHECK(a != nullptr);
    if (!a) return;
    CHECK_EQ((int)a->kind, RAD_PLUGIN_ARCH);
    CHECK_EQ(a->arch_id, "radtest.v1");
    CHECK(a->arch_declare != nullptr);
    CHECK(a->arch_step != nullptr);
    CHECK_EQ(a->n_kernels, 0);     /* it exports no kernel symbols, and that is not an error */

    const std::string e = errs(r);
    CHECK(has(e, "radtest.v1"));
    CHECK(has(e, "radtest_arch2"));
}

/* The reply format is an optional export. Absent, the slot stays null and the plugin loads (as
 * with rad_arch_probe, also not exported here), and the engine derives the format from the chat
 * template. Present, it resolves to a description the reply parser can run. */
TEST(an_architecture_plugin_may_declare_its_reply_format) {
    Registry r;
    (void)r.load_dir(arch_dir(), {});
    const Plugin* a = r.plugin("radtest_arch");
    CHECK(a != nullptr);
    if (!a) return;
    CHECK(a->arch_probe == nullptr);
    REQUIRE(a->arch_chat_format != nullptr);
    const RadChatFormat* d = a->arch_chat_format(nullptr);
    REQUIRE(d != nullptr);
    CHECK_EQ(d->struct_size, (uint32_t)sizeof(RadChatFormat));

    ChatFormat f;
    std::string why;
    CHECK_OK(chat_format_from_abi(d, &f, &why));
    CHECK_OK(chat_format_check(f, &why));
    CHECK_EQ(f.name, std::string("radtest-format"));
    CHECK_EQ(f.call_open, std::string("<call>"));
    CHECK_EQ(f.arg_value_prefix, std::string("="));
    CHECK(f.triggers == std::vector<std::string>{ "<call>" });

    /* And the parser reads a reply in it. */
    std::shared_ptr<const ReplyFormat> rf;
    CHECK_OK(ReplyFormat::compile(f, &rf, &why));
    const nlohmann::ordered_json tools = nlohmann::ordered_json::parse(
        R"([{"type":"function","function":{"name":"go","parameters":{"type":"object",)"
        R"("properties":{"n":{"type":"integer"},"s":{"type":"string"}}}}}])");
    ReplyOptions ro;
    ro.tools = &tools;
    std::shared_ptr<const ReplyParser> rp;
    CHECK_OK(ReplyParser::create(rf, ro, &rp, &why));
    const ReplyMessage m = reply_parse(*rp, "<reason>hm</reason>ok<call><fn go><arg n=3</arg>"
                                            "<arg s=<![CDATA[x<y]]></arg></fn></call>");
    CHECK_EQ(m.reasoning, std::string("hm"));
    CHECK_EQ(m.content, std::string("ok"));
    REQUIRE(m.calls.size() == 1);
    CHECK_EQ(m.calls[0].name, std::string("go"));
    CHECK_EQ(m.calls[0].arguments, std::string("{\"n\":3,\"s\":\"x<y\"}"));
}

TEST(every_row_s_op_has_a_schema) {
    Registry r;
    CHECK_OK(load(r));
    CHECK_OK(r.check_complete());
    const auto ops = r.op_names();
    CHECK_EQ(ops.size(), (size_t)10);
    CHECK_EQ(ops[0], "t_attn");
}

/* ================================================================== op schemas */

TEST(schema_validation_accepts_the_declared_shape) {
    Registry r;
    CHECK_OK(load(r));
    const RadOpSchema* s = r.schema("t_gemm");
    CHECK(s != nullptr);
    if (!s) return;
    CHECK_EQ(s->n_params, 4);

    std::string err;
    RadParam ok[] = { pint("M", 8), pint("N", 4096), pint("K", 4096), pstr("dtype", "bf16") };
    CHECK_OK(r.validate("t_gemm", ok, 4, &err));

    /* A range is an integer that has not been collapsed yet, so it satisfies an integer slot. */
    RadParam ranged[] = { prange("M", 1, 8192), pint("N", 4096), pint("K", 4096),
                          pstr("dtype", "bf16") };
    CHECK_OK(r.validate("t_gemm", ranged, 4, &err));
}

TEST(schema_validation_names_the_near_miss) {
    Registry r;
    CHECK_OK(load(r));
    std::string err;

    RadParam ok[] = { pint("M", 8), pint("N", 4096), pint("K", 4096), pstr("dtype", "bf16") };
    CHECK_EQ(r.validate("t_gemmm", ok, 4, &err), RAD_E_NOSCHEMA);
    CHECK(has(err, "did you mean 't_gemm'"));
    CHECK_EQ(r.near_miss("t_atn"), "t_attn");
    CHECK_EQ(r.near_miss("completely_different"), "");

    RadParam typo[] = { pint("M", 8), pint("N", 4096), pint("KK", 4096), pstr("dtype", "bf16") };
    CHECK_EQ(r.validate("t_gemm", typo, 4, &err), RAD_E_SCHEMA);
    CHECK(has(err, "did you mean 'K'"));
}

TEST(schema_validation_rejects_a_bad_param_set) {
    Registry r;
    CHECK_OK(load(r));
    std::string err;

    RadParam missing[] = { pint("M", 8), pint("N", 4096), pint("K", 4096) };
    CHECK_EQ(r.validate("t_gemm", missing, 3, &err), RAD_E_SCHEMA);
    CHECK(has(err, "dtype"));

    RadParam wrong_kind[] = { pstr("M", "eight"), pint("N", 4096), pint("K", 4096),
                              pstr("dtype", "bf16") };
    CHECK_EQ(r.validate("t_gemm", wrong_kind, 4, &err), RAD_E_SCHEMA);
    CHECK(has(err, "integer"));

    RadParam twice[] = { pint("M", 8), pint("M", 9), pint("N", 4096), pint("K", 4096),
                         pstr("dtype", "bf16") };
    CHECK_EQ(r.validate("t_gemm", twice, 5, &err), RAD_E_SCHEMA);
    CHECK(has(err, "twice"));

    /* At most one ranged parameter per op, so the bucket-table index stays a single integer at
     * issue. */
    RadParam two_ranges[] = { prange("M", 1, 64), pint("N", 4096), prange("K", 1, 64),
                              pstr("dtype", "bf16") };
    CHECK_EQ(r.validate("t_gemm", two_ranges, 4, &err), RAD_E_SCHEMA);
    CHECK(has(err, "at most one ranged parameter"));
}

/* A RAD_DERIVED parameter is not owed by the caller; demanding it would recreate the circularity
 * it exists to break. Supplying it is allowed and pins it. */
TEST(schema_validation_does_not_demand_a_derived_parameter) {
    Registry r;
    CHECK_OK(load(r));
    std::string err;

    RadParam without[] = { pint("q_len", 8), pint("head_dim", 128), pint("window", 0),
                           pint("causal", 1), pint("n_head", 8), pstr("kv_dtype", "bf16") };
    CHECK_OK(r.validate("t_attn", without, 6, &err));

    RadParam with[] = { pint("q_len", 8), pint("head_dim", 128), pint("window", 0),
                        pint("block_size", 16), pint("causal", 1), pint("n_head", 8),
                        pstr("kv_dtype", "bf16") };
    CHECK_OK(r.validate("t_attn", with, 7, &err));
}

/* Constraints never match on a float, so the schema check is all that stops an integer spelling
 * of RAD_F64("eps", 1e-6) from reaching the kernel as zero. */
TEST(schema_validation_separates_a_float_from_an_integer) {
    Registry r;
    CHECK_OK(load(r));
    std::string err;

    RadParam f64[] = { pint("M", 8), pint("n", 4096), pf64("eps", 1e-6), pstr("dtype", "bf16") };
    CHECK_OK(r.validate("t_norm", f64, 4, &err));

    RadParam as_int[] = { pint("M", 8), pint("n", 4096), pint("eps", 0), pstr("dtype", "bf16") };
    CHECK_EQ(r.validate("t_norm", as_int, 4, &err), RAD_E_SCHEMA);
    CHECK(has(err, "declare it with RAD_F64()"));

    RadParam f64_for_int[] = { pf64("M", 8.0), pint("n", 4096), pstr("dtype", "bf16") };
    CHECK_EQ(r.validate("t_norm", f64_for_int, 3, &err), RAD_E_SCHEMA);
    CHECK(has(err, "is a float"));
}

/* ================================================================== selection */

TEST(selection_takes_the_highest_declared_priority) {
    Registry r;
    CHECK_OK(load(r));
    CHECK_EQ(picked(r.select("t_gemm", gemm_geom(16), RAD_DOMAIN_DEVICE)), "t_gemm_m16");
    CHECK_EQ(picked(r.select("t_gemm", gemm_geom(17), RAD_DOMAIN_DEVICE)), "t_gemm_m64");
    CHECK_EQ(picked(r.select("t_gemm", gemm_geom(65), RAD_DOMAIN_DEVICE)), "t_gemm_any");

    /* K = 1000 is divisible by neither 256 nor 16, and w8a8 is in no row's set, so both
     * specialised rows drop out and the unconstrained one serves. */
    CHECK_EQ(picked(r.select("t_gemm", gemm_geom(8, 1000), RAD_DOMAIN_DEVICE)), "t_gemm_any");
    CHECK_EQ(picked(r.select("t_gemm", gemm_geom(8, 4096, "w8a8"), RAD_DOMAIN_DEVICE)),
             "t_gemm_any");

    SelectionTrace t;
    r.select("t_gemm", gemm_geom(8, 1000), RAD_DOMAIN_DEVICE, &t);
    CHECK_EQ(t.candidates.size(), (size_t)3);      /* the host row is not a candidate */
    CHECK(t.reason.empty());
}

/* Specificity is declared, not derived, and a tie goes to the row declared first. */
TEST(selection_breaks_ties_on_declaration_order) {
    Registry r;
    CHECK_OK(load(r));
    Geometry g;
    g.set_i("M", 4);
    g.set_i("n", 1);
    CHECK_EQ(picked(r.select("t_tie", g, RAD_DOMAIN_DEVICE)), "t_tie_first");
}

/* The hierarchy is absolute. radtest_over's t_norm row is unconstrained at priority 1;
 * radtest_base's is EQ n==4096 at priority 100. The plugin named first wins; priority never
 * crosses the plugin boundary. */
TEST(hierarchy_beats_priority_and_specificity) {
    Geometry g;
    g.set_i("M", 1);
    g.set_i("n", 4096);
    g.set_s("dtype", "bf16");

    Registry over_first;
    CHECK_OK(over_first.load_dir(kernels_dir(), { "radtest_over", "radtest_base" }));
    CHECK_EQ(picked(over_first.select("t_norm", g, RAD_DOMAIN_DEVICE)), "t_over_norm");

    Registry base_first;
    CHECK_OK(base_first.load_dir(kernels_dir(), { "radtest_base", "radtest_over" }));
    CHECK_EQ(picked(base_first.select("t_norm", g, RAD_DOMAIN_DEVICE)), "t_norm_specialised");

    /* Domains resolve independently: radtest_over has no host row, so the host question falls
     * through to the next plugin instead of being answered "none". */
    CHECK_EQ(picked(over_first.select("t_norm", g, RAD_DOMAIN_HOST)), "t_norm_host");
}

TEST(a_missing_domain_is_reported_not_substituted) {
    Registry r;
    CHECK_OK(load(r));
    CHECK_EQ(picked(r.select("t_gemm", gemm_geom(8), RAD_DOMAIN_HOST)), "t_gemm_host");

    SelectionTrace t;
    const KernelRow* k = r.select("t_attn", attn_geom(), RAD_DOMAIN_HOST, &t);
    CHECK(k == nullptr);
    CHECK(t.other_domain_has_rows);
    CHECK(has(t.reason, "no host kernel"));
}

TEST(an_unknown_op_names_the_near_miss) {
    Registry r;
    CHECK_OK(load(r));
    const std::string why = refusal(r, "t_gem", gemm_geom(8));
    CHECK(has(why, "no loaded kernel plugin implements op 't_gem'"));
    CHECK(has(why, "did you mean 't_gemm'"));
}

/* Every constraint operator, and the sentence each one produces when it refuses. */
TEST(every_constraint_operator_refuses_with_its_own_reason) {
    Registry r;
    CHECK_OK(load(r));
    CHECK_EQ(picked(r.select("t_attn", attn_geom(), RAD_DOMAIN_DEVICE)), "t_attn_paged");

    Geometry g = attn_geom(); g.set_i("head_dim", 64);
    CHECK(has(refusal(r, "t_attn", g), "head_dim == 128, have 64"));

    g = attn_geom(); g.set_i("window", -1);
    CHECK(has(refusal(r, "t_attn", g), "window >= 0, have -1"));

    /* Pinned by the caller, and therefore matched normally rather than skipped. */
    CHECK(has(refusal(r, "t_attn", attn_geom_pinned(32)), "block_size == 16, have 32"));

    g = attn_geom(); g.set_i("causal", 2);
    CHECK(has(refusal(r, "t_attn", g), "causal in {0 1}, have 2"));

    g = attn_geom(); g.set_i("q_len", 17);
    CHECK(has(refusal(r, "t_attn", g), "q_len <= 16, have 17"));

    g = attn_geom(); g.set_i("n_head", 6);
    CHECK(has(refusal(r, "t_attn", g), "n_head % 4 == 0, have 6"));

    g = attn_geom(); g.set_s("kv_dtype", "f32");
    CHECK(has(refusal(r, "t_attn", g), "kv_dtype in {bf16 fp8_e4m3}, have f32"));

    /* A constraint on a key the geometry lacks does not hold (libr4d's rule): a row that needs a
     * parameter the caller did not state is not a match. */
    Geometry partial;
    partial.set_i("q_len", 8);
    CHECK(has(refusal(r, "t_attn", partial), "did not name head_dim"));
}

/* RAD_DERIVED: a parameter the kernel supplies. Skipped during matching when the caller did not
 * name it, since otherwise spec §7.2 is circular (the block manager reads the block size off the
 * resolved kernel, which cannot resolve without it); written into the resolved geometry once a
 * winner exists. */
TEST(a_derived_parameter_is_skipped_then_supplied) {
    Registry r;
    CHECK_OK(load(r));

    SelectionTrace t;
    const KernelRow* k = r.select("t_attn", attn_geom(), RAD_DOMAIN_DEVICE, &t);
    CHECK_EQ(picked(k), "t_attn_paged");        /* would refuse "did not name block_size" without */
    CHECK_EQ(t.derived.size(), (size_t)1);
    if (t.derived.size() == 1) {
        CHECK_EQ(t.derived[0].key, "block_size");
        CHECK(t.derived[0].known);
        CHECK_EQ(t.derived[0].value, 16);
    }

    /* The resolved geometry carries it, so rad_kv_block_size() and RadArgs.p see one number. */
    Resolved res = r.resolve("t_attn", attn_geom(), RAD_DOMAIN_DEVICE);
    CHECK(res);
    long long bs = 0;
    CHECK(res.geom.get_i("block_size", &bs));
    CHECK_EQ(bs, 16);

    /* A caller that supplies it has pinned it: matched normally, so a mismatch is a refusal by
     * name, and nothing is derived afterwards. */
    SelectionTrace pinned;
    CHECK_EQ(picked(r.select("t_attn", attn_geom_pinned(16), RAD_DOMAIN_DEVICE, &pinned)),
             "t_attn_paged");
    CHECK_EQ(pinned.derived.size(), (size_t)0);
    CHECK(r.select("t_attn", attn_geom_pinned(32), RAD_DOMAIN_DEVICE) == nullptr);
}

/* A winner with no RAD_C_EQ on the derived key (ref's `block_size >= 1`, which admits any value)
 * leaves it unset: the caller must decide rather than be handed a zero that looks like an
 * answer. */
TEST(a_derived_key_with_no_eq_stays_unset) {
    Registry r;
    CHECK_OK(load(r));

    SelectionTrace t;
    CHECK_EQ(picked(r.select("t_attn_ref", attn_geom(), RAD_DOMAIN_DEVICE, &t)), "t_attn_ref_any");
    CHECK_EQ(t.derived.size(), (size_t)1);
    if (t.derived.size() == 1) {
        CHECK_EQ(t.derived[0].key, "block_size");
        CHECK(!t.derived[0].known);
    }
    Resolved res = r.resolve("t_attn_ref", attn_geom(), RAD_DOMAIN_DEVICE);
    CHECK(res);
    CHECK(!res.geom.has("block_size"));
}

TEST(bands_carry_the_derived_value_into_every_band) {
    Registry r;
    CHECK_OK(load(r));
    std::vector<Band> b = r.build_bands("t_attn", "q_len", 1, 64, attn_geom());
    /* q_len <= 16 is the only boundary the rows place on it, so (0,16] resolves and (16,64] does
     * not, and the resolved band carries the kernel's block size. */
    CHECK_EQ(b.size(), (size_t)2);
    if (b.size() != 2) return;
    CHECK_EQ(b[0].hi, 16);
    CHECK_EQ(picked(b[0].dom[RAD_DOMAIN_DEVICE].row), "t_attn_paged");
    long long bs = 0;
    CHECK(b[0].dom[RAD_DOMAIN_DEVICE].geom.get_i("block_size", &bs));
    CHECK_EQ(bs, 16);
    CHECK(!b[1].dom[RAD_DOMAIN_DEVICE]);
}

/* §7.2: the block size comes from the resolved attention kernel's constraints, not a core
 * constant. This is the accessor the block manager reads it with. */
TEST(a_resolved_kernel_s_constraints_are_readable) {
    Registry r;
    CHECK_OK(load(r));
    const KernelRow* k = r.select("t_attn", attn_geom(), RAD_DOMAIN_DEVICE);
    CHECK(k != nullptr);
    if (!k) return;
    long long v = 0;
    CHECK(kernel_constraint(*k, "block_size", RAD_C_EQ, &v));
    CHECK_EQ(v, 16);
    CHECK(kernel_constraint(*k, "q_len", RAD_C_LE, &v));
    CHECK_EQ(v, 16);
    /* Absent is not a default: the kernel states no EQ on window, and the caller must notice
     * rather than be handed a zero. */
    CHECK(!kernel_constraint(*k, "window", RAD_C_EQ, &v));
    CHECK(!kernel_constraint(*k, "nonesuch", RAD_C_EQ, &v));
}

/* ================================================================== bucket tables */

/* spec.md §2.2's worked example, band for band: three kernels constraining M<=16, M<=64 and
 * nothing produce (0,16], (16,64], (64,max] without any configured policy. */
TEST(band_table_is_the_worked_example) {
    Registry r;
    CHECK_OK(load(r));
    Geometry base;
    base.set_i("N", 4096);
    base.set_i("K", 4096);
    base.set_s("dtype", "bf16");

    std::vector<Band> b = r.build_bands("t_gemm", "M", 1, 8192, base);
    CHECK_EQ(b.size(), (size_t)3);
    if (b.size() != 3) return;

    CHECK_EQ(b[0].hi, 16);
    CHECK_EQ(b[1].hi, 64);
    CHECK_EQ(b[2].hi, 8192);
    CHECK_EQ(picked(b[0].dom[RAD_DOMAIN_DEVICE].row), "t_gemm_m16");
    CHECK_EQ(picked(b[1].dom[RAD_DOMAIN_DEVICE].row), "t_gemm_m64");
    CHECK_EQ(picked(b[2].dom[RAD_DOMAIN_DEVICE].row), "t_gemm_any");

    /* Both domains are resolved wherever both exist, so the core can change execution site at
     * runtime without re-resolving. */
    for (const Band& band : b) {
        CHECK_EQ(picked(band.dom[RAD_DOMAIN_HOST].row), "t_gemm_host");
        CHECK(band.miss[RAD_DOMAIN_DEVICE].empty());
    }

    /* Each band is frozen at its upper bound, the worst case in the band. */
    long long m = 0;
    CHECK(b[1].dom[RAD_DOMAIN_DEVICE].geom.get_i("M", &m));
    CHECK_EQ(m, 64);
    CHECK_EQ(b[1].dom[RAD_DOMAIN_DEVICE].tune_source, "axis defaults");
}

/* A range with a hole is reported as that band, not as the whole op. Boundaries are the
 * constraint values themselves: LE 64 gives 64, GE 4096 ends the band below it at 4095. */
TEST(band_boundaries_come_from_the_constraints) {
    Registry r;
    CHECK_OK(load(r));
    Geometry base;
    base.set_i("N", 4096);
    base.set_i("K", 4096);
    base.set_s("dtype", "bf16");

    std::vector<Band> b = r.build_bands("t_skinny", "M", 1, 8192, base);
    CHECK_EQ(b.size(), (size_t)3);
    if (b.size() != 3) return;
    CHECK_EQ(b[0].hi, 64);
    CHECK_EQ(b[1].hi, 4095);
    CHECK_EQ(b[2].hi, 8192);

    CHECK_EQ(picked(b[0].dom[RAD_DOMAIN_DEVICE].row), "t_skinny_m64");
    CHECK(!b[1].dom[RAD_DOMAIN_DEVICE]);
    CHECK(has(b[1].miss[RAD_DOMAIN_DEVICE], "M <= 64, have 4095"));
    CHECK(has(b[1].miss[RAD_DOMAIN_DEVICE], "M >= 4096, have 4095"));
    CHECK_EQ(picked(b[2].dom[RAD_DOMAIN_DEVICE].row), "t_skinny_big");

    /* No host kernel anywhere for this op: every band says which placement option that removed. */
    for (const Band& band : b) {
        CHECK(!band.dom[RAD_DOMAIN_HOST]);
        CHECK(has(band.miss[RAD_DOMAIN_HOST], "site 'host' is unavailable"));
    }
}

TEST(an_eq_constraint_cuts_both_edges) {
    Registry r;
    CHECK_OK(load(r));
    Geometry base;
    base.set_i("n", 1);

    std::vector<Band> b = r.build_bands("t_eq", "M", 1, 64, base);
    CHECK_EQ(b.size(), (size_t)3);
    if (b.size() != 3) return;
    CHECK_EQ(b[0].hi, 31);
    CHECK_EQ(b[1].hi, 32);
    CHECK_EQ(b[2].hi, 64);
    CHECK(!b[0].dom[RAD_DOMAIN_DEVICE]);
    CHECK_EQ(picked(b[1].dom[RAD_DOMAIN_DEVICE].row), "t_eq_m32");
    CHECK(!b[2].dom[RAD_DOMAIN_DEVICE]);
}

TEST(bands_clamp_to_the_declared_range) {
    Registry r;
    CHECK_OK(load(r));
    Geometry base;
    base.set_i("N", 4096);
    base.set_i("K", 4096);
    base.set_s("dtype", "bf16");

    /* 16 is below the range and 64 is at or above its top: neither cuts anything. */
    std::vector<Band> narrow = r.build_bands("t_gemm", "M", 32, 48, base);
    CHECK_EQ(narrow.size(), (size_t)1);
    CHECK_EQ(narrow[0].hi, 48);
    CHECK_EQ(picked(narrow[0].dom[RAD_DOMAIN_DEVICE].row), "t_gemm_m64");

    std::vector<Band> tiny = r.build_bands("t_gemm", "M", 1, 16, base);
    CHECK_EQ(tiny.size(), (size_t)1);
    CHECK_EQ(tiny[0].hi, 16);
    CHECK_EQ(picked(tiny[0].dom[RAD_DOMAIN_DEVICE].row), "t_gemm_m16");
}

TEST(an_op_with_no_range_gets_one_band) {
    Registry r;
    CHECK_OK(load(r));
    std::vector<Band> b = r.build_bands("t_attn", "", 0, 0, attn_geom());
    CHECK_EQ(b.size(), (size_t)1);
    if (b.empty()) return;
    CHECK_EQ(b[0].hi, INT64_MAX);
    CHECK_EQ(picked(b[0].dom[RAD_DOMAIN_DEVICE].row), "t_attn_paged");
    CHECK(!b[0].dom[RAD_DOMAIN_HOST]);
}

/* ================================================================== instances */

TEST(init_runs_once_per_resolved_instance) {
    Registry r;
    CHECK_OK(load(r));
    const Plugin* base = r.plugin("radtest_base");
    CHECK(base != nullptr);
    if (!base) return;
    auto n_init = reinterpret_cast<int (*)(void)>(dlsym(base->handle, "radtest_init_calls"));
    auto n_fini = reinterpret_cast<int (*)(void)>(dlsym(base->handle, "radtest_fini_calls"));
    auto inst_m = reinterpret_cast<long long (*)(void*)>(dlsym(base->handle, "radtest_instance_m"));
    CHECK(n_init && n_fini && inst_m);
    if (!n_init || !n_fini || !inst_m) return;
    const int init0 = n_init(), fini0 = n_fini();

    Geometry g;
    g.set_i("N", 4096);
    g.set_i("K", 4096);
    g.set_s("dtype", "bf16");
    std::vector<Band> b = r.build_bands("t_gemm", "M", 1, 8192, g);
    CHECK_EQ(b.size(), (size_t)3);
    if (b.size() != 3) return;

    for (Band& band : b) CHECK_OK(r.init_instance(band.dom[RAD_DOMAIN_DEVICE], 0, 1));
    CHECK_EQ(n_init() - init0, 3);
    CHECK_EQ(r.n_instances(), (size_t)3);

    /* The instance saw the frozen geometry: its own band's upper bound, not the range. */
    CHECK_EQ(inst_m(b[0].dom[RAD_DOMAIN_DEVICE].instance), 16);
    CHECK_EQ(inst_m(b[2].dom[RAD_DOMAIN_DEVICE].instance), 8192);

    /* The host rows have no init hook: that is RAD_OK and a null instance, not a call. */
    CHECK_OK(r.init_instance(b[0].dom[RAD_DOMAIN_HOST], 0, 1));
    CHECK(b[0].dom[RAD_DOMAIN_HOST].instance == nullptr);
    CHECK_EQ(n_init() - init0, 3);

    r.fini_instances();
    CHECK_EQ(n_fini() - fini0, 3);
    CHECK_EQ(r.n_instances(), (size_t)0);
}

TEST(a_negative_init_is_a_value_not_a_crash) {
    Registry r;
    CHECK_OK(load(r));
    Geometry base;
    base.set_i("n", 1);
    std::vector<Band> b = r.build_bands("t_eq", "M", 32, 32, base);
    CHECK_EQ(b.size(), (size_t)1);
    if (b.empty()) return;
    CHECK_EQ(picked(b[0].dom[RAD_DOMAIN_DEVICE].row), "t_eq_m32");
    CHECK_EQ(r.init_instance(b[0].dom[RAD_DOMAIN_DEVICE], 0, 1), RAD_E_NOMEM);
    CHECK_EQ(r.n_instances(), (size_t)0);
}

/* ================================================================== the selection table (§16) */

TEST(the_selection_table_folds_repeats_and_names_the_refusal) {
    Registry r;
    CHECK_OK(load(r));
    r.clear_selections();

    r.select("t_gemm", gemm_geom(16), RAD_DOMAIN_DEVICE);
    r.select("t_gemm", gemm_geom(16), RAD_DOMAIN_DEVICE);   /* the same question, folded */
    CHECK_EQ(r.n_selections(), (size_t)1);

    r.select("t_gemm", gemm_geom(16), RAD_DOMAIN_HOST);     /* a different domain is a different
                                                             * question */
    CHECK_EQ(r.n_selections(), (size_t)2);

    Geometry g = attn_geom();
    g.set_i("q_len", 17);
    r.select("t_attn", g, RAD_DOMAIN_DEVICE);
    CHECK_EQ(r.n_selections(), (size_t)3);

    r.select("t_attn", attn_geom(), RAD_DOMAIN_DEVICE);
    CHECK_EQ(r.n_selections(), (size_t)4);

    const std::string table = r.selection_table();
    CHECK(has(table, "4 distinct questions, 5 asked"));
    CHECK(has(table, "t_gemm_m16 (radtest_base, prio 30)"));
    CHECK(has(table, "M=16 N=4096 K=4096 dtype=bf16"));
    CHECK(has(table, "-- none: q_len <= 16, have 17"));
    /* A derived key was not part of the question, so it is printed as the kernel's answer, not
     * in the geometry column. */
    CHECK(has(table, "derived block_size:=16"));
    CHECK(!has(table, "block_size=16 "));

    r.clear_selections();
    CHECK_EQ(r.n_selections(), (size_t)0);
}

/* A "device." constraint is read from the machine, not the query: the engine measures peer
 * access at launch. Unmeasured, the kernel is not selected and the refusal says so; measured
 * absent, the plugin's next row takes the op, and an op with no such row is refused naming the
 * fact. The query never carries the key, so no geometry (and no tuning key) changes. */
TEST(a_device_fact_constraint_is_matched_against_the_machine) {
    Registry r;
    CHECK_OK(load(r));
    Geometry g;
    g.set_i("M", 4);
    g.set_i("n", 64);

    CHECK_EQ(picked(r.select("t_peer", g, RAD_DOMAIN_DEVICE)), "t_peer_staged");
    CHECK(has(refusal(r, "t_peer_only", g), "device.p2p is not known"));

    r.set_device_fact(RAD_FACT_P2P, 1);
    CHECK_EQ(picked(r.select("t_peer", g, RAD_DOMAIN_DEVICE)), "t_peer_direct");
    CHECK_EQ(picked(r.select("t_peer_only", g, RAD_DOMAIN_DEVICE)), "t_peer_only_direct");
    CHECK(!g.has(RAD_FACT_P2P));

    r.set_device_fact(RAD_FACT_P2P, 0);
    CHECK_EQ(picked(r.select("t_peer", g, RAD_DOMAIN_DEVICE)), "t_peer_staged");
    const std::string why = refusal(r, "t_peer_only", g);
    CHECK(has(why, "device.p2p == 1, have 0"));
    CHECK(has(why, "measured at launch"));
}

/* An engine with no card leaves a GPU library's device rows out, since on the host backend a
 * resolved device row would fail at its first launch. Only the engine sets this
 * (set_no_device_code); a selecting-only registry keeps every row, as the libr4d tests below need.
 * As the first libr4d test, its dlopen registers the library's fat binaries. */
TEST(an_engine_with_no_card_leaves_a_gpu_librarys_device_rows_out) {
    const std::string so = libr4d();
    if (so.empty()) return;
    Registry r;
    r.set_no_device_code(true);
    CHECK(r.load_plugin(so, 0) >= 0);
    const Plugin* p = r.plugin("libr4d");
    CHECK(p != nullptr);
    if (!p) return;
    CHECK(p->device_code_absent);
    for (const KernelRow& row : r.rows()) CHECK(row.info->domain != RAD_DOMAIN_DEVICE);
}

/* The shipped libr4d picks the all-reduce by message size. At 3 and 4 ranks the exact sum is a
 * one-shot for decode-sized messages and a two-shot for prefill chunks, at 8 ranks a one-shot only
 * up to 8192 elements (its seven-fold send loses early); the two-shot rows start one element above
 * the one-shot bound, which is a band edge of the ranged `numel`. Edges are the union over every
 * candidate row, so the eight-rank edge at 8192 also splits the other widths' bands, each half
 * resolving the same row. The band under the one-shot bound must stay resolvable for every row
 * with a DIV on `numel` (two-rank wht6 (64), three- and four-rank ti8 (32, 128)), or a lossy
 * serve loses its decode band. Only the real plugin can show
 * resolution above two ranks on a two-card machine. Geometry: Qwen3.8-Flash-Next, 2560 wide,
 * 2048-token chunk. */
TEST(libr4d_resolves_the_all_reduce_by_message_size) {
    const std::string so = libr4d();
    if (so.empty()) return;
    Registry r;
    CHECK(r.load_plugin(so, 0) >= 0);
    r.set_device_fact(RAD_FACT_P2P, 1);

    const int64_t width = 2560, chunk = 2048;
    auto bands = [&](int world, int exact, int64_t max_tok) {
        Geometry g;
        g.set_i("world_size", world);
        g.set_s("dtype", "bf16");
        g.set_i("exact", exact);
        g.set_i("min_bytes", 128 * 1024);
        return r.build_bands("all_reduce", "numel", width, max_tok * width, g);
    };

    CHECK_EQ(band_names(bands(3, 1, chunk)),
             "8192:ar_oneshot_3rank_exact 524288:ar_oneshot_3rank_exact "
             "5242880:ar_twoshot_3rank_exact ");
    CHECK_EQ(band_names(bands(4, 1, chunk)),
             "8192:ar_oneshot_4rank_exact 524288:ar_oneshot_4rank_exact "
             "5242880:ar_twoshot_4rank_exact ");
    CHECK_EQ(band_names(bands(8, 1, chunk)),
             "8192:ar_oneshot_8rank_exact 524288:ar_twoshot_8rank_exact "
             "5242880:ar_twoshot_8rank_exact ");
    CHECK_EQ(band_names(bands(3, 0, chunk)),
             "8192:ar_twoshot_3rank_ti8 524288:ar_twoshot_3rank_ti8 5242880:ar_twoshot_3rank_ti8 ");
    CHECK_EQ(band_names(bands(4, 0, chunk)),
             "8192:ar_twoshot_4rank_ti8 524288:ar_twoshot_4rank_ti8 5242880:ar_twoshot_4rank_ti8 ");
    CHECK_EQ(band_names(bands(2, 0, chunk)),
             "8192:ar_oneshot_2rank_wht6 524288:ar_oneshot_2rank_wht6 "
             "5242880:ar_oneshot_2rank_wht6 ");
    CHECK_EQ(band_names(bands(2, 1, chunk)),
             "8192:ar_oneshot_2rank_exact 524288:ar_oneshot_2rank_exact "
             "5242880:ar_oneshot_2rank_exact ");

    /* A 16384-token chunk is two-shot to its end: the two-shot's bound guards its word offsets,
     * it is not a prefill size, and a chunk past 20971520 elements would otherwise fall back to
     * the one-shot's twice-as-heavy send. */
    CHECK_EQ(band_names(bands(4, 1, 16384)),
             "8192:ar_oneshot_4rank_exact 524288:ar_oneshot_4rank_exact "
             "41943040:ar_twoshot_4rank_exact ");
    CHECK_EQ(band_names(bands(3, 1, 16384)),
             "8192:ar_oneshot_3rank_exact 524288:ar_oneshot_3rank_exact "
             "41943040:ar_twoshot_3rank_exact ");

    /* At `overlap` 1 (an all-reduce beside compute) the four-rank two-shot band takes the lane
     * rows, exact and lossy; the one-shot band has no lane row past two ranks. Three and eight ranks
     * have no lane row and keep the plain two-shot. */
    auto lane = [&](int world, int exact) {
        Geometry g;
        g.set_i("world_size", world);
        g.set_s("dtype", "bf16");
        g.set_i("exact", exact);
        g.set_i("min_bytes", 128 * 1024);
        g.set_i("overlap", 1);
        return r.build_bands("all_reduce", "numel", width, chunk * width, g);
    };
    CHECK_EQ(band_names(lane(4, 1)), "8192:ar_oneshot_4rank_exact 524288:ar_oneshot_4rank_exact "
             "5242880:ar_twoshot_4rank_exact_lane ");
    CHECK_EQ(band_names(lane(4, 0)), "8192:ar_twoshot_4rank_ti8_lane 524288:ar_twoshot_4rank_ti8_lane "
             "5242880:ar_twoshot_4rank_ti8_lane ");
    CHECK_EQ(band_names(lane(3, 1)), "8192:ar_oneshot_3rank_exact 524288:ar_oneshot_3rank_exact "
             "5242880:ar_twoshot_3rank_exact ");
    CHECK_EQ(band_names(lane(8, 1)), "8192:ar_oneshot_8rank_exact 524288:ar_twoshot_8rank_exact "
             "5242880:ar_twoshot_8rank_exact ");

    /* Above the two-shot's bound the one-shot takes the message back. */
    CHECK_EQ(band_names(bands(4, 1, 131072)),
             "8192:ar_oneshot_4rank_exact 524288:ar_oneshot_4rank_exact "
             "268435456:ar_twoshot_4rank_exact 335544320:ar_oneshot_4rank_exact ");

    /* A step that never exceeds the bound stays on the one-shot, split only at the eight-rank
     * edge. */
    CHECK_EQ(band_names(bands(4, 1, 128)),
             "8192:ar_oneshot_4rank_exact 327680:ar_oneshot_4rank_exact ");

    /* Eight ranks have no lossy wire: every band is a miss that names the request. */
    const std::vector<Band> none = bands(8, 0, chunk);
    CHECK(!none.empty());
    for (const Band& b : none) {
        CHECK(!b.dom[RAD_DOMAIN_DEVICE]);
        CHECK(has(b.miss[RAD_DOMAIN_DEVICE], "exact"));
    }

    /* A row with a DIV on the message carries it, and the issue path refuses a message it does
     * not divide. Every message here (rows times 2560) is a multiple of 128, so none is refused. */
    for (const Band& b : bands(2, 0, chunk)) CHECK_EQ(b.dom[RAD_DOMAIN_DEVICE].ranged_div, 64ll);
    for (const Band& b : bands(3, 0, chunk)) CHECK_EQ(b.dom[RAD_DOMAIN_DEVICE].ranged_div, 32ll);
    for (const Band& b : bands(4, 0, chunk)) CHECK_EQ(b.dom[RAD_DOMAIN_DEVICE].ranged_div, 128ll);
    const std::vector<Band> exact4 = bands(4, 1, chunk);
    REQUIRE_EQ(exact4.size(), 3u);
    CHECK_EQ(exact4[0].dom[RAD_DOMAIN_DEVICE].ranged_div, 0ll);
    CHECK_EQ(exact4[1].dom[RAD_DOMAIN_DEVICE].ranged_div, 0ll);
    CHECK_EQ(exact4[2].dom[RAD_DOMAIN_DEVICE].ranged_div, 64ll);
    for (const Band& b : bands(3, 1, chunk)) CHECK_EQ(b.dom[RAD_DOMAIN_DEVICE].ranged_div, 0ll);
}

/* The four-rank fused writes serve the exact wire only, and selection says so. `wire` 0 (exact)
 * gets the four-rank rows; the rotated wire at four ranks is a miss naming `wire`, so the caller
 * keeps its unfused pair (a row that matched and then refused its instance would stop declare).
 * At two ranks both wires keep their row. Geometry is Qwen3.8-Flash-Next's: 2560 wide, four
 * streams, top-k 10, over a decode or verify step of 8 sequences at depth 3. */
TEST(libr4d_resolves_the_four_rank_fused_writes_on_the_exact_wire_only) {
    const std::string so = libr4d();
    if (so.empty()) return;
    Registry r;
    CHECK(r.load_plugin(so, 0) >= 0);
    r.set_device_fact(RAD_FACT_P2P, 1);

    const int64_t rows = 8 * (1 + 3);
    auto bands = [&](const std::string& op, int world, int wire) {
        Geometry g;
        g.set_i("n", 2560);
        g.set_i("hc", 4);
        g.set_i("world_size", world);
        g.set_s("dtype", "bf16");
        g.set_i("wire", wire);
        if (op == "ar_gather_hc_write") {
            g.set_i("top_k", 10);
            g.set_s("act", "sigmoid");
        }
        return r.build_bands(op, "M", 1, rows, g);
    };

    CHECK_EQ(band_names(bands("ar_hc_write", 4, 0)), "32:ar_hc_write_4rank ");
    CHECK_EQ(band_names(bands("ar_gather_hc_write", 4, 0)), "32:ar_gather_hc_write_4rank ");
    for (int wire : { 0, 6 }) {
        CHECK_EQ(band_names(bands("ar_hc_write", 2, wire)), "32:ar_hc_write ");
        CHECK_EQ(band_names(bands("ar_gather_hc_write", 2, wire)), "32:ar_gather_hc_write ");
    }

    for (const char* op : { "ar_hc_write", "ar_gather_hc_write" }) {
        const std::vector<Band> none = bands(op, 4, 6);
        CHECK(!none.empty());
        for (const Band& b : none) {
            CHECK(!b.dom[RAD_DOMAIN_DEVICE]);
            CHECK(has(b.miss[RAD_DOMAIN_DEVICE], "wire"));
        }
    }
}

/* The fused all-reduce junction resolves by width: two ranks take the two-rank row, four the
 * wide family's row (or its host-mailbox twin without peer access), and three or eight ranks have
 * no row, so a caller keeps the pair. Geometry is Gemma 4 31B's at decode heights. */
TEST(libr4d_resolves_the_fused_all_reduce_junction_at_two_and_four_ranks) {
    const std::string so = libr4d();
    if (so.empty()) return;
    Registry r;
    CHECK(r.load_plugin(so, 0) >= 0);
    auto bands = [&](int world) {
        Geometry g;
        g.set_i("n", 5376);
        g.set_i("world_size", world);
        g.set_s("dtype", "bf16");
        g.set_f("eps", 1e-6);
        g.set_f("wadd", 0.0);
        return r.build_bands("ar_rmsnorm_quant_i8", "M", 1, 4, g);
    };
    r.set_device_fact(RAD_FACT_P2P, 1);
    CHECK_EQ(band_names(bands(2)), "4:ar_rmsnorm_quant_i8 ");
    CHECK_EQ(band_names(bands(4)), "4:ar_rmsnorm_quant_i8_4rank ");
    for (int world : { 3, 8 })
        for (const Band& b : bands(world)) CHECK(!b.dom[RAD_DOMAIN_DEVICE]);
    r.set_device_fact(RAD_FACT_P2P, 0);
    CHECK_EQ(band_names(bands(2)), "4:ar_rmsnorm_quant_i8_host ");
    CHECK_EQ(band_names(bands(4)), "4:ar_rmsnorm_quant_i8_4rank_host ");
}

/* The vision tower's attention resolves as rad_block_vit.h declares it: head_dim 72, no
 * `window`. `window` is optional in attn_dense's schema and a constraint on an absent key does
 * not hold, so a row constraining it is unreachable for a caller that omits it. rad-kbench cannot
 * catch this: its cases always name the window. */
TEST(libr4d_resolves_the_vision_attention_without_a_window) {
    const std::string so = libr4d();
    if (so.empty()) return;
    Registry r;
    CHECK(r.load_plugin(so, 0) >= 0);
    Geometry g;
    g.set_i("head_dim", 72);
    g.set_i("gqa", 1);
    g.set_i("causal", 0);
    g.set_s("dtype", "bf16");
    g.set_i("max_seqlen", 16384);
    const std::vector<Band> b = r.build_bands("attn_dense", "M", 1, 16384, g);
    CHECK(!b.empty());
    for (const Band& x : b) CHECK_EQ(picked(x.dom[RAD_DOMAIN_DEVICE].row), "attn_vit_h72_bf16");
}

/* GLM-5.3-Flash's MLA at eight heads a rank: the matrix row ranks above the wave row and serves
 * every M up to its bound, a decode step's rows and a prefill chunk alike. At 64 heads the rows
 * row ranks above both. */
TEST(libr4d_resolves_the_mla_attention_by_rows) {
    const std::string so = libr4d();
    if (so.empty()) return;
    Registry r;
    CHECK(r.load_plugin(so, 0) >= 0);
    Geometry g;
    g.set_i("latent_dim", 512);
    g.set_i("heads", 8);
    g.set_i("blocks", 513);
    g.set_i("block_size", 4);
    g.set_s("dtype", "bf16");
    CHECK_EQ(band_names(r.build_bands("mla_attn", "M", 1, 2048, g)), "2048:mla_attn_matrix_bf16 ");
    CHECK_EQ(band_names(r.build_bands("mla_attn", "M", 1, 48, g)), "48:mla_attn_matrix_bf16 ");
    // All 64 heads of a row, as GLM's prefill split by rows runs it: the rows row.
    g.set_i("heads", 64);
    CHECK_EQ(band_names(r.build_bands("mla_attn", "M", 1, 512, g)), "512:mla_attn_rows_bf16 ");
    g.set_i("heads", 16);
    CHECK_EQ(band_names(r.build_bands("mla_attn", "M", 1, 512, g)), "512:mla_attn_matrix_bf16 ");
}

/* GLM-5.3-Flash's MLA split by rows at eight ranks (arch/common/rad_block_glm_mla.h), with the
 * parameters its ops declare against the container's encodings (block fp8 at f32 scales): every
 * band of every op the row form adds resolves on the device, up to a 4096-token step's run of
 * 512 rows. A form libr4d does not serve would stop a TP8 start at "resolved to no kernel". */
TEST(libr4d_resolves_every_band_of_glm_mla_split_by_rows) {
    const std::string so = libr4d();
    if (so.empty()) return;
    Registry r;
    CHECK(r.load_plugin(so, 0) >= 0);
    r.set_device_fact(RAD_FACT_P2P, 1);
    auto all_resolve = [&](const char* op, const char* ranged, int64_t lo, int64_t hi,
                           const Geometry& g) {
        const std::vector<Band> bands = r.build_bands(op, ranged, lo, hi, g);
        bool ok = !bands.empty();
        for (const Band& x : bands) ok = ok && x.dom[RAD_DOMAIN_DEVICE].row;
        if (!ok) std::fprintf(stderr, "    %s: %s\n", op, band_names(bands).c_str());
        return ok;
    };
    const int64_t R = 512;
    for (const auto& [n, k] : { std::pair<int64_t, int64_t>{ 16384, 1536 }, { 4096, 16384 } }) {
        Geometry g;
        g.set_i("N", n);
        g.set_i("K", k);
        g.set_i("group", 128);
        g.set_s("dtype", "fp8a8f32");
        CHECK(all_resolve("gemm_nt_q", "M", 1, R, g));
    }
    for (const char* mode : { "absorb", "expand" }) {
        Geometry g;
        g.set_i("heads", 64);
        g.set_i("head_dim", 256);
        g.set_i("latent_dim", 512);
        g.set_s("mode", mode);
        g.set_s("dtype", "bf16");
        CHECK(all_resolve("mla_project", "M", 1, R, g));
    }
    {
        Geometry g;
        g.set_i("latent_dim", 512);
        g.set_i("heads", 64);
        g.set_i("blocks", 513);
        g.set_i("block_size", 4);
        g.set_s("dtype", "bf16");
        CHECK(all_resolve("mla_attn", "M", 1, R, g));
    }
    {
        Geometry g;
        g.set_i("n", 16384);
        g.set_i("group", 128);
        g.set_s("dtype", "bf16");
        CHECK(all_resolve("quant_act_fp8", "M", 1, 4096, g));
    }
    {
        Geometry g;
        g.set_i("world_size", 8);
        g.set_s("dtype", "bf16");
        CHECK(all_resolve("all_gather", "numel", 4096, 4096 * 4096, g));
    }
}

/* The dense integer gemm_nt_q kernels compile their group in, so their rows take no other: at
 * group 64 each dtype is a miss, at 128 it resolves. At M = 1 every skinny, tiled and prefill row
 * of these dtypes is a candidate. */
TEST(libr4d_resolves_a_dense_integer_gemm_only_at_its_group) {
    const std::string so = libr4d();
    if (so.empty()) return;
    Registry r;
    CHECK(r.load_plugin(so, 0) >= 0);
    for (const char* dtype : { "w4a16", "w5a16", "w4a16_asym", "w4a8i8_asym", "w5a8i8", "w2a8",
                               "w8a16", "w4a8", "w4a8_asym", "w8a8" }) {
        Geometry g;
        g.set_i("M", 1);
        g.set_i("N", 256);
        g.set_i("K", 256);
        g.set_s("dtype", dtype);
        g.set_i("group", 64);
        CHECK_EQ(picked(r.select("gemm_nt_q", g, RAD_DOMAIN_DEVICE)), "<none>");
        g.set_i("group", 128);
        CHECK(r.select("gemm_nt_q", g, RAD_DOMAIN_DEVICE) != nullptr);
    }
}

/* ================================================================== constraints that are not intervals */
namespace {
std::string extra_dir()   { return home() + "/testkernels/extra"; }
std::string decline_dir() { return home() + "/testkernels/decline"; }

/* The base plugin and radtest_extra behind it. */
int load_extra(Registry& r) {
    return r.load_dirs({ kernels_dir(), extra_dir() }, { "radtest_base", "radtest_extra" });
}

/* The band holding `v`, or null. */
const Band* band_at(const std::vector<Band>& b, int64_t v) {
    for (const Band& x : b) if (v <= x.hi) return &x;
    return nullptr;
}

/* An empty directory of its own under the test's temporary directory. */
struct TempDir {
    std::string path;
    TempDir() {
        std::string tmpl = radtest::tmp_path("dir.XXXXXX");
        if (mkdtemp(tmpl.data())) path = tmpl;
    }
    std::string sub(const std::string& rel) const {
        const std::string p = path + "/" + rel;
        mkdir(p.c_str(), 0755);
        return p;
    }
};
}  /* namespace */

/* Bands are resolved at their upper bound, so a row whose constraint on the ranged parameter is not
 * an interval would win a band at its top and be issued at values inside it that it rejects. An IN
 * lists its values, each of which gets a band of its own; a DIV row keeps its bands and they carry
 * its divisor, which the issue path checks. Every value of the range is then either served by its
 * band's row or refused at issue, never launched outside the row's contract. */
TEST(no_band_issues_a_row_at_a_value_its_constraints_reject) {
    Registry r;
    REQUIRE_EQ(load_extra(r), RAD_OK);
    Geometry base;
    base.set_i("n", 1);
    const std::vector<Band> b = r.build_bands("t_div", "M", 1, 64, base);
    for (int64_t m = 1; m <= 64; ++m) {
        const Band* x = band_at(b, m);
        REQUIRE(x != nullptr);
        const Resolved& w = x->dom[RAD_DOMAIN_DEVICE];
        REQUIRE(w.row != nullptr);
        Geometry g = base;
        g.set_i("M", m);
        bool holds = true;
        for (int i = 0; i < w.row->info->n_constraints; ++i)
            holds = holds && constraint_holds(w.row->info->constraints[i], g);
        const bool refused = w.ranged_div > 1 && m % w.ranged_div != 0;
        if (holds == refused)
            fprintf(stderr, "    M=%lld: band <= %lld, %s, ranged_div %lld\n", (long long)m,
                    (long long)x->hi, w.row->info->name, (long long)w.ranged_div);
        CHECK(holds != refused);
    }
    /* The listed values have bands of their own, and only they reach the IN row. */
    CHECK_EQ(picked(band_at(b, 8)->dom[RAD_DOMAIN_DEVICE].row), "t_div_in");
    CHECK_EQ(band_at(b, 8)->hi, 8);
    CHECK_EQ(picked(band_at(b, 24)->dom[RAD_DOMAIN_DEVICE].row), "t_div_in");
    CHECK_EQ(picked(band_at(b, 9)->dom[RAD_DOMAIN_DEVICE].row), "t_div_any");
    CHECK_EQ(picked(band_at(b, 7)->dom[RAD_DOMAIN_DEVICE].row), "t_div_any");
    /* The DIV row still wins the top band at its upper bound, as before, and says what it needs. */
    CHECK_EQ(picked(band_at(b, 64)->dom[RAD_DOMAIN_DEVICE].row), "t_div_m16");
    CHECK_EQ(band_at(b, 64)->dom[RAD_DOMAIN_DEVICE].ranged_div, 16ll);
    CHECK_EQ(band_at(b, 8)->dom[RAD_DOMAIN_DEVICE].ranged_div, 0ll);
    /* An IN or DIV on any other key cuts nothing: t_gemm's DIV on K and IN on dtype leave the
     * worked example's three bands. */
    Geometry gb;
    gb.set_i("N", 4096);
    gb.set_i("K", 4096);
    gb.set_s("dtype", "bf16");
    const std::vector<Band> g = r.build_bands("t_gemm", "M", 1, 8192, gb);
    CHECK_EQ(band_names(g), "16:t_gemm_m16 64:t_gemm_m64 8192:t_gemm_any ");
    for (const Band& x : g) CHECK_EQ(x.dom[RAD_DOMAIN_DEVICE].ranged_div, 0ll);
}

/* ================================================================== the tuning cache */
/* A cache row moves one shape off its axis defaults. It applies to the kernel, plugin and geometry
 * it names, keyed without the choices themselves; a value the axis does not declare is never
 * launched; and a combination the kernel's cross-axis test refuses falls back to every default,
 * which the kernel must always run. */
TEST(a_tuning_cache_row_moves_its_shape_off_the_defaults_and_only_within_the_axis) {
    TempDir home_dir;
    REQUIRE(!home_dir.path.empty());
    const std::string machine = "radtest-machine";
    Geometry wide;
    wide.set_i("M", 64);
    wide.set_i("N", 256);
    wide.set_i("K", 64);
    wide.set_s("dtype", "bf16");
    Geometry narrow = wide;
    narrow.set_i("N", 64);
    std::string wide_key, narrow_key;
    {
        Registry r;
        REQUIRE_EQ(load_extra(r), RAD_OK);
        const Resolved a = r.resolve("t_tune", wide, RAD_DOMAIN_DEVICE);
        REQUIRE(a.row != nullptr);
        CHECK_EQ(a.tune_source, "axis defaults");
        CHECK_EQ(a.choice, "bn=64");
        wide_key = a.geom_key;
        narrow_key = r.resolve("t_tune", narrow, RAD_DOMAIN_DEVICE).geom_key;
        CHECK(!has(wide_key, "bn="));
    }
    TuneCache tc;
    tc.set_machine(machine);
    auto row = [&](const char* choice, const std::string& geom) {
        TuneRow t;
        t.kernel = "t_tune_picky";
        t.plugin = "radtest_extra";
        t.choice = choice;
        t.geom = geom;
        t.us = 1.0;
        t.measured = "2026-10-15T00:00:00Z";
        tc.put(t);
    };
    row("bn=128", wide_key);
    row("bn=128", narrow_key);
    /* Rows a lookup must never take: an undeclared value, and a key that includes its answer. */
    Geometry other = wide;
    other.set_i("N", 512);
    Registry probe;
    REQUIRE_EQ(load_extra(probe), RAD_OK);
    const std::string other_key = probe.resolve("t_tune", other, RAD_DOMAIN_DEVICE).geom_key;
    row("bn=999", other_key);
    Geometry with_answer = wide;
    with_answer.set_i("N", 1024);
    with_answer.set_i("bn", 128);
    row("bn=128", TuneCache::canon_geometry(with_answer));
    REQUIRE_EQ(tc.save(home_dir.path), RAD_OK);

    Registry r;
    REQUIRE_EQ(load_extra(r), RAD_OK);
    CHECK_EQ(r.load_tune(home_dir.path, machine), 4);
    const Resolved a = r.resolve("t_tune", wide, RAD_DOMAIN_DEVICE);
    long long bn = 0;
    CHECK(a.geom.get_i("bn", &bn));
    CHECK_EQ(bn, 128ll);
    CHECK(a.tune_source.rfind("tuned", 0) == 0);
    CHECK_EQ(a.choice, "bn=128");

    const Resolved u = r.resolve("t_tune", other, RAD_DOMAIN_DEVICE);
    CHECK(u.geom.get_i("bn", &bn));
    CHECK_EQ(bn, 64ll);
    CHECK_EQ(u.tune_source, "axis defaults");

    Geometry asked = wide;
    asked.set_i("N", 1024);
    const Resolved k = r.resolve("t_tune", asked, RAD_DOMAIN_DEVICE);
    CHECK_EQ(k.choice, "bn=64");

    Resolved n;
    const std::string warn = stderr_of([&] { n = r.resolve("t_tune", narrow, RAD_DOMAIN_DEVICE); });
    CHECK(n.row != nullptr);
    CHECK_EQ(n.choice, "bn=64");
    CHECK_EQ(n.tune_source, "axis defaults");
    CHECK(has(warn, "falling back to its defaults"));
}

/* ================================================================== the loader's edges */

/* --kernels names a library by its declared name or its file, in any spelling: `radtest_base`,
 * `libradtest_base`, `radtest_base.so` and a path to it are one library. An entry that matches
 * nothing loads nothing in its place and is warned about. */
TEST(a_kernel_library_is_named_by_any_spelling_of_its_file) {
    for (const std::string& h : { std::string("radtest_base"), std::string("libradtest_base"),
                                  std::string("radtest_base.so"),
                                  kernels_dir() + "/radtest_base.so" }) {
        Registry r;
        CHECK_OK(r.load_dir(kernels_dir(), { h }));
        const Plugin* p = r.plugin("radtest_base");
        CHECK(p != nullptr);
        if (p) CHECK_EQ(p->order, 0);
        CHECK(r.plugin("radtest_over") == nullptr);
    }
    Registry r;
    std::string log;
    log = stderr_of([&] { CHECK_OK(r.load_dir(kernels_dir(), { "nope", "radtest_base" })); });
    CHECK(has(log, "'nope'"));
    CHECK_EQ(r.plugins().size(), (size_t)1);
    CHECK(r.plugin("radtest_base") != nullptr);
}

/* A directory earlier on the search path shadows a file of the same name later on it, so an
 * operator's own build of a library wins over the installation's. Loading a second directory
 * continues the order from the first. */
TEST(an_earlier_directory_shadows_a_later_one_and_orders_continue) {
    TempDir own;
    REQUIRE(!own.path.empty());
    const std::string link = own.path + "/radtest_over.so";
    REQUIRE_EQ(symlink((kernels_dir() + "/radtest_over.so").c_str(), link.c_str()), 0);
    Registry r;
    CHECK_OK(r.load_dirs({ own.path, kernels_dir() }, {}));
    const Plugin* over = r.plugin("radtest_over");
    REQUIRE(over != nullptr);
    CHECK_EQ(over->path, link);
    CHECK_EQ(r.plugins().size(), (size_t)2);

    Registry r2;
    CHECK_OK(r2.load_dir(kernels_dir(), {}));
    (void)stderr_of([&] { (void)r2.load_dir(arch_dir(), {}); });
    const Plugin* arch = r2.plugin("radtest_arch");
    REQUIRE(arch != nullptr);
    CHECK(arch->order >= 2);
}

/* A tunable axis whose default is not one of its values, or an axis declared twice, would run an
 * unsupported configuration on every untuned shape; the loader refuses the library by name. */
TEST(a_tunable_axis_the_kernel_cannot_run_by_default_is_refused_at_load) {
    Registry r;
    std::string log = stderr_of([&] {
        CHECK_EQ(r.load_plugin(bad_dir() + "/radtest_badtune_default.so", 0), RAD_E_INVAL);
    });
    CHECK(has(errs(r), "defaults to 96, which is not one of its declared values"));
    CHECK(r.plugins().empty());
    Registry r2;
    log = stderr_of([&] {
        CHECK_EQ(r2.load_plugin(bad_dir() + "/radtest_badtune_twice.so", 0), RAD_E_INVAL);
    });
    CHECK(has(errs(r2), "declares tunable 'bn' twice"));
    CHECK(r2.rows().empty());
}

/* A library that declines this machine at open (built for another card) is skipped, not refused:
 * nothing registers and it takes no position in the hierarchy. */
TEST(a_library_that_declines_the_machine_registers_nothing_and_takes_no_place) {
    Registry r;
    CHECK_OK(r.load_dirs({ decline_dir(), kernels_dir() }, {}));
    CHECK(r.plugin("radtest_decline") == nullptr);
    CHECK_EQ(r.plugins().size(), (size_t)2);
    for (const Plugin& p : r.plugins()) CHECK(p.order < 2);
    for (const KernelRow& k : r.rows()) CHECK(std::string(k.info->name) != "t_decline_gemm");
    Registry r2;
    CHECK_EQ(r2.load_plugin(decline_dir() + "/radtest_decline.so", 0), 1);
    CHECK(r2.plugins().empty());
    CHECK(r2.load_errors().empty());
}

RAD_TEST_MAIN()
