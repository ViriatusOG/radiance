/* shipped_plugins_test.cpp -- the plugins this tree ships, loaded through the engine's loader.
 *
 * rad-avx-check keeps its own copy of the loader's schema comparison, and only rad-kbench and the
 * out-of-tree test load the shipped libraries through the Registry, both slowly. A change to the
 * loader's rules, or to a shipped plugin's schemas or selection key, fails here in milliseconds.
 */
#include "rad_test.h"

#include "plugin/registry.h"
#include "rad_builder.h"

#include <cstring>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace rad;

namespace {

std::string dir_of(const char* sub) { return home() + "/" + sub; }

std::string errors(const Registry& r) {
    std::string s;
    for (const auto& e : r.load_errors()) s += "    " + e + "\n";
    return s;
}

}  /* namespace */

/* The CPU-only engine's kernels: libavx ahead of libref, as `--kernels libavx,libref` orders them.
 * They load with no refusal (the loader refuses two plugins that disagree on an op's schema),
 * libref is the reference and ranks below, and every op libavx serves libref serves too, so a
 * libavx row that declines always has a reference behind it. */
TEST(the_host_kernel_libraries_load_together_through_the_loader) {
    Registry r;
    const int st = r.load_dir(dir_of("kernels"), { "libavx", "libref" });
    if (st == RAD_E_IO && r.plugins().empty())
        ARTIFACT_MISSING("the kernel libraries under " + dir_of("kernels"));
    if (!r.load_errors().empty()) fprintf(stderr, "%s", errors(r).c_str());
    CHECK_OK(st);
    CHECK(r.load_errors().empty());
    const Plugin* avx = r.plugin("libavx");
    const Plugin* ref = r.plugin("libref");
    REQUIRE(avx != nullptr && ref != nullptr);
    CHECK(avx->order < ref->order);
    CHECK(ref->reference);
    CHECK(!avx->reference);
    CHECK(avx->n_kernels > 0 && ref->n_kernels > 0);
    std::set<std::string> avx_ops, ref_ops;
    for (const KernelRow& k : r.rows()) {
        if (k.plugin == avx->name) avx_ops.insert(k.info->op);
        if (k.plugin == ref->name) ref_ops.insert(k.info->op);
    }
    int orphans = 0;
    for (const auto& op : avx_ops) {
        if (!ref_ops.count(op)) {
            fprintf(stderr, "    libavx serves '%s' and libref does not\n", op.c_str());
            ++orphans;
        }
        CHECK(r.schema(op) != nullptr);
    }
    CHECK_EQ(orphans, 0);
    CHECK(avx_ops.size() > 50);
    /* Every op has one schema, fixed by the first library to declare it. */
    for (const auto& op : r.op_names()) {
        const Plugin* owner = r.schema_owner(op);
        REQUIRE(owner != nullptr);
        if (avx_ops.count(op)) CHECK_EQ(owner->name, avx->name);
    }
}

/* Every shipped architecture plugin loads, exports declare and step, and claims its own
 * (architecture id, quantisation descriptor) pair, the key a container selects it by: two
 * plugins on one pair would make the choice arbitrary. */
TEST(the_shipped_architectures_load_with_distinct_selection_keys) {
    Registry r;
    const int st = r.load_dir(dir_of("architectures"), {});
    if (st == RAD_E_IO && r.plugins().empty())
        ARTIFACT_MISSING("the architecture plugins under " + dir_of("architectures"));
    if (!r.load_errors().empty()) fprintf(stderr, "%s", errors(r).c_str());
    CHECK_OK(st);
    std::map<std::pair<std::string, std::string>, std::string> keys;
    for (const Plugin& p : r.plugins()) {
        if (p.kind != RAD_PLUGIN_ARCH) continue;
        CHECK(p.arch_declare != nullptr);
        CHECK(p.arch_step != nullptr);
        CHECK(keys.emplace(std::make_pair(p.arch_id, p.arch_quant), p.name).second);
    }
    const std::set<std::pair<std::string, std::string>> want = {
        { "anima", "" }, { "bailinghybrid", "" }, { "breeze", "" }, { "clef", "" },
        { "bailinghybrid", "gguf_ftype_7" }, { "bailinghybrid", "gguf_ftype_15" },
        { "bailinghybrid", "gguf_ftype_17" }, { "bailinghybrid", "gguf_ftype_18" },
        { "bailinghybrid", "gguf_ftype_25" },
        { "clef", "fp8_e4m3" }, { "embeddinggemma2", "" },
        { "fishomni", "bsgemm-moe" }, { "flint", "" }, { "flux3", "" }, { "flux3distilled", "" },
        { "deepseek41", "" }, { "gemma3", "" }, { "gemma4", "" },
        { "gemma4", "compressed-tensors" }, { "gemma4", "exl3" }, { "glm5next", "" },
        { "hunyuan3d2", "" }, { "hyv3", "" }, { "hyv3", "fp8" }, { "hyv3", "gguf_ftype_15" },
        { "inkling2", "" }, { "laya", "" }, { "lfm2", "" }, { "lfm3", "" }, { "llama", "" }, { "mistral3", "" },
        { "llama", "fp8_e4m3" }, { "nemotronh", "" }, { "nomicbert", "" }, { "qwen35", "" },
        { "museglimmer", "" },
        { "museglimmer", "compressed-tensors" },
        { "qwen35", "exl3" }, { "qwen35", "fp8_e4m3" }, { "qwen35", "w8a8_fwht128" },
        { "qwen35moe", "fp8_e4m3" },
        { "qwen35moe", "w8a8_fwht128" },
        { "qwen3asr", "" }, { "krea2", "" }, { "ltx2", "" }, { "minimaxh3", "" },
        { "minimaxmusic3", "" }, { "qwen4exp", "" }, { "qwenimage21", "" },
        { "spark25", "" }, { "spark25", "w8a8_fwht128" }, { "sdxl", "" },
        { "unlimitedocr", "" }, { "wan", "" } };
    std::set<std::pair<std::string, std::string>> got;
    for (const auto& [k, name] : keys) got.insert(k);
    if (got != want)
        for (const auto& [k, name] : keys)
            fprintf(stderr, "    %s claims (%s, '%s')\n", name.c_str(), k.first.c_str(),
                    k.second.c_str());
    CHECK(got == want);
}

TEST(hyv3_plugins_declare_the_checkpoint_native_tool_format) {
    Registry r;
    const int st = r.load_dir(dir_of("architectures"), {});
    if (st == RAD_E_IO && r.plugins().empty())
        ARTIFACT_MISSING("the architecture plugins under " + dir_of("architectures"));
    CHECK_OK(st);
    for (const char* name : { "hyv3_bf16", "hyv3_fp8", "hyv3_gguf" }) {
        const Plugin* p = r.plugin(name);
        REQUIRE(p != nullptr);
        REQUIRE(p->arch_chat_format != nullptr);
        const RadChatFormat* f = p->arch_chat_format(nullptr);
        REQUIRE(f != nullptr);
        CHECK_EQ(std::string(f->reasoning_open), std::string("<think>"));
        CHECK_EQ(std::string(f->reasoning_close), std::string("</think>"));
        CHECK_EQ(f->reasoning_in_prompt, (uint32_t)RAD_CHAT_THINK_FROM_PROMPT);
        REQUIRE(f->reasoning_breaks != nullptr && f->n_reasoning_breaks == 1);
        CHECK_EQ(std::string(f->reasoning_breaks[0]), std::string("<tool_calls>"));
        CHECK_EQ(std::string(f->section_open), std::string("<tool_calls>"));
        CHECK_EQ(std::string(f->section_close), std::string("</tool_calls>"));
        CHECK_EQ(std::string(f->name_prefix), std::string("<tool_call>"));
        CHECK_EQ(std::string(f->name_suffix), std::string("<tool_sep>"));
        CHECK_EQ(std::string(f->arg_prefix), std::string("<arg_key>"));
        CHECK_EQ(std::string(f->arg_name_suffix), std::string("</arg_key>\n"));
        CHECK_EQ(std::string(f->arg_value_prefix), std::string("<arg_value>"));
        CHECK_EQ(std::string(f->arg_value_suffix), std::string("</arg_value>"));
        REQUIRE(f->n_triggers == 1 && f->triggers != nullptr);
        CHECK_EQ(std::string(f->triggers[0]), std::string("<tool_calls>"));
        CHECK_EQ(f->string_value, (uint32_t)RAD_CHAT_VALUE_RAW);
        CHECK_EQ(f->other_value, (uint32_t)RAD_CHAT_VALUE_JSON);
    }
}

/* A checkpoint whose quantisation was imported weight by weight (Quark, ParoQuant, GPTQ, NVFP4)
 * names a tool no plugin claims, so the family's one quantised plugin serves it, and two are
 * refused (startup.h). So each family has at most one quantised plugin that takes imports:
 * qwen35_w8a8 declines them (rad_arch_serves_imports), leaving Qwen3.8 imports to qwen35_fp8. */
TEST(every_shipped_family_has_at_most_one_plugin_for_an_imported_checkpoint) {
    Registry r;
    const int st = r.load_dir(dir_of("architectures"), {});
    if (st == RAD_E_IO && r.plugins().empty())
        ARTIFACT_MISSING("the architecture plugins under " + dir_of("architectures"));
    CHECK_OK(st);
    std::map<std::string, std::vector<std::string>> takers;
    for (const Plugin& p : r.plugins())
        if (p.kind == RAD_PLUGIN_ARCH && !p.arch_quant.empty() && p.arch_serves_imports)
            takers[p.arch_id].push_back(p.name);
    for (const auto& [id, names] : takers) {
        if (names.size() > 1)
            for (const auto& n : names)
                fprintf(stderr, "    '%s' would take an imported %s checkpoint\n", n.c_str(),
                        id.c_str());
        CHECK_EQ(names.size(), 1u);
    }
    REQUIRE(takers.count("qwen35"));
    CHECK_EQ(takers["qwen35"][0], std::string("qwen35_fp8"));
    const Plugin* w8 = r.plugin("qwen35_w8a8");
    REQUIRE(w8 != nullptr);
    CHECK(!w8->arch_serves_imports);
}

/* drama-3's input as fish-speech's server writes it (chunking.py, generate.py): each speaker's turn
 * tagged and stripped, turns on lines of their own, untagged text speaker 0's; refused when empty,
 * when it holds a token only the prompt builder writes, or when it names a speaker other than 0..15
 * the canonical way. A refusal leaves its reason in the output. */
TEST(fishomni_writes_the_input_as_the_reference_server_does) {
    Registry r;
    const int st = r.load_dir(dir_of("architectures"), {});
    if (st == RAD_E_IO && r.plugins().empty())
        ARTIFACT_MISSING("the architecture plugins under " + dir_of("architectures"));
    CHECK_OK(st);
    const Plugin* p = r.plugin("fishomni");
    REQUIRE(p != nullptr);
    REQUIRE(p->arch_text_prepare != nullptr);
    auto prep = [&](const char* in, std::string* out) {
        char buf[512] = {};
        const int64_t n = p->arch_text_prepare(RAD_TEXT_INPUT, in, buf, sizeof buf);
        *out = buf;
        return n;
    };
    const struct { const char* in; const char* want; } ok[] = {
        { "Hello there.", "<|speaker:0|> Hello there." },
        { "  Hello there. \n", "<|speaker:0|> Hello there." },
        { "\xe3\x80\x80Hello\xc2\xa0", "<|speaker:0|> Hello" },          /* U+3000, U+00A0 */
        { "<|speaker:0|> Are you ready? <|speaker:1|> Always.",
          "<|speaker:0|> Are you ready?\n<|speaker:1|> Always." },
        { "Hi <|speaker:12|>Hey  <|speaker:3|>   ", "<|speaker:0|> Hi\n<|speaker:12|> Hey" },
        { "<|speaker:2|>[laughs] Fine.", "<|speaker:2|> [laughs] Fine." },
    };
    for (const auto& c : ok) {
        std::string got;
        const int64_t n = prep(c.in, &got);
        if (got != c.want) fprintf(stderr, "    '%s' -> '%s'\n", c.in, got.c_str());
        CHECK_EQ(n, (int64_t)std::strlen(c.want));
        CHECK_EQ(got, std::string(c.want));
    }
    const struct { const char* in; const char* why; } refused[] = {
        { "   \t", "empty" },
        { "<|speaker:1|>  ", "empty" },
        { "Say <|im_end|> now", "<|im_end|>" },
        { "<|voice|>", "<|voice|>" },
        { "a <|semantic:12|> b", "<|semantic:" },
        { "<|speaker:16|> hi", "speaker tags" },
        { "<|speaker:01|> hi", "speaker tags" },
        { "<|speaker_change|> hi", "speaker tags" },
    };
    for (const auto& c : refused) {
        std::string why;
        CHECK(prep(c.in, &why) < 0);
        if (why.find(c.why) == std::string::npos) fprintf(stderr, "    '%s': %s\n", c.in, why.c_str());
        CHECK(why.find(c.why) != std::string::npos);
    }
    /* A short buffer still gets the length to retry with, and the instructions (which the model
     * refuses before they get here) pass as written. */
    char tiny[4] = {};
    CHECK_EQ(p->arch_text_prepare(RAD_TEXT_INPUT, "Hello", tiny, sizeof tiny),
             (int64_t)std::strlen("<|speaker:0|> Hello"));
    char buf[64] = {};
    CHECK_EQ(p->arch_text_prepare(RAD_TEXT_INSTRUCTIONS, " calm ", buf, sizeof buf), 6);
    CHECK_EQ(std::string(buf), std::string(" calm "));
}

/* A reference's transcript as generate.py's build_drama3_prompt writes it after "Text: ": speaker
 * 0's tag and a space before untagged text, which is otherwise kept as given; white space before a
 * leading tag moved behind it. The input's rules refuse the same tokens. */
TEST(fishomni_tags_a_reference_transcript_as_the_prompt_builder_does) {
    Registry r;
    const int st = r.load_dir(dir_of("architectures"), {});
    if (st == RAD_E_IO && r.plugins().empty())
        ARTIFACT_MISSING("the architecture plugins under " + dir_of("architectures"));
    CHECK_OK(st);
    const Plugin* p = r.plugin("fishomni");
    REQUIRE(p != nullptr);
    REQUIRE(p->arch_text_prepare != nullptr);
    auto prep = [&](const char* in, std::string* out) {
        char buf[512] = {};
        const int64_t n = p->arch_text_prepare(RAD_TEXT_REFERENCE, in, buf, sizeof buf);
        *out = buf;
        return n;
    };
    const struct { const char* in; const char* want; } ok[] = {
        { "We asked over twenty people.", "<|speaker:0|> We asked over twenty people." },
        { " Two spaces  kept. ", "<|speaker:0|>  Two spaces  kept. " },
        { "  <|speaker:1|>Hi there", "<|speaker:1|>  Hi there" },
        { "<|speaker:3|> Hello. <|speaker:4|> Hi.", "<|speaker:3|> Hello. <|speaker:4|> Hi." },
    };
    for (const auto& c : ok) {
        std::string got;
        const int64_t n = prep(c.in, &got);
        if (got != c.want) fprintf(stderr, "    '%s' -> '%s'\n", c.in, got.c_str());
        CHECK_EQ(n, (int64_t)std::strlen(c.want));
        CHECK_EQ(got, std::string(c.want));
    }
    const struct { const char* in; const char* why; } refused[] = {
        { " \n ", "empty" },
        { "Then <|im_start|> came", "<|im_start|>" },
        { "<|speaker:20|> hi", "speaker tags" },
    };
    for (const auto& c : refused) {
        std::string why;
        CHECK(prep(c.in, &why) < 0);
        if (why.find(c.why) == std::string::npos) fprintf(stderr, "    '%s': %s\n", c.in, why.c_str());
        CHECK(why.find(c.why) != std::string::npos);
    }
}

RAD_TEST_MAIN()

/* libr4d's rows that chain answer 2 to rad_kernel_chain: every chained form of theirs ends its wait
 * in r4d_chain_acquire, which empties the scalar data cache behind the fence, and that is what
 * lets an engine issue their packets without the packet processor's acquire
 * (core/device/aql.cpp, "the acquire of a chained launch"). One row answering 1 keeps the acquire
 * on every chained packet of an engine that loads it, so a row added with a wait of its own shows
 * up here rather than as a slower step. */
TEST(libr4ds_rows_that_chain_acquire_for_themselves) {
#ifndef RAD_TEST_HAS_LIBR4D
    SKIP("this build makes no libr4d");
#endif
    Registry r;
    const int st = r.load_dir(dir_of("kernels"), { "libr4d", "libref" });
    if (!r.plugin("libr4d")) {
        (void)st;
        ARTIFACT_MISSING("libr4d under " + dir_of("kernels"));
    }
    int chained = 0;
    for (const KernelRow& k : r.rows()) {
        if (!k.chain) continue;
        ++chained;
        CHECK(k.chain_fenced);
        if (!k.chain_fenced) fprintf(stderr, "    %s chains and answers 1\n", k.info->name);
    }
    CHECK(chained > 0);
}

/* The EXL3 routed-expert libraries (exl3/) load beside the tree's own kernel libraries with no
 * refusal: same ABI version and layout fingerprint, and no op schema that disagrees with one a
 * library ahead of them declared. exl3ref is a reference and ranks last; it serves every
 * moe_exl3 op exl3r4d serves, so a device row that declines has a reference behind it. */
TEST(the_exl3_kernel_libraries_load_beside_the_trees_own) {
    std::vector<std::string> names = { "libavx", "libref", "exl3ref" };
#ifdef RAD_TEST_HAS_EXL3R4D
    names.insert(names.begin(), "exl3r4d");
#endif
#ifdef RAD_TEST_HAS_LIBR4D
    names.insert(names.begin(), "libr4d");
#endif
    Registry r;
    const int st = r.load_dir(dir_of("kernels"), names);
    if (!r.plugin("exl3ref")) {
        (void)st;
        ARTIFACT_MISSING("exl3ref under " + dir_of("kernels"));
    }
    if (!r.load_errors().empty()) fprintf(stderr, "%s", errors(r).c_str());
    CHECK_OK(st);
    CHECK(r.load_errors().empty());
    const Plugin* ref = r.plugin("exl3ref");
    CHECK(ref->reference);
    std::set<std::string> ref_ops, dev_ops;
    for (const KernelRow& k : r.rows()) {
        if (k.plugin == "exl3ref") ref_ops.insert(k.info->op);
        if (k.plugin == "exl3r4d") dev_ops.insert(k.info->op);
    }
    CHECK(ref_ops.count("moe_exl3_gate_up") == 1 && ref_ops.count("moe_exl3_down") == 1);
    for (const auto& op : ref_ops) CHECK(r.schema(op) != nullptr);
#ifdef RAD_TEST_HAS_EXL3R4D
    const Plugin* dev = r.plugin("exl3r4d");
    REQUIRE(dev != nullptr);
    CHECK(!dev->reference);
    CHECK(dev->order < ref->order);
    CHECK(!dev_ops.empty());
    /* exl3r4d's decode-band _dk ops have no reference row; every other op it serves must. */
    for (const auto& op : dev_ops) {
        const bool dk = op.size() > 3 && op.compare(op.size() - 3, 3, "_dk") == 0;
        if (!dk && !ref_ops.count(op)) {
            fprintf(stderr, "    exl3r4d serves '%s' and exl3ref does not\n", op.c_str());
            CHECK(false);
        }
    }
#endif
}

/* Every shipped quantiser plugin loads, and the `exl3` quantiser (exl3/quantizers/exl3) is offered
 * by exactly one of them: quantiser names are unique across plugins, and a clash refuses the later
 * plugin whole. */
TEST(the_shipped_quantisers_load_and_exl3_is_offered_once) {
    Registry r;
    const int st = r.load_dir(dir_of("quantizers"), {});
    if (st == RAD_E_IO && r.plugins().empty())
        ARTIFACT_MISSING("the quantiser plugins under " + dir_of("quantizers"));
    if (!r.load_errors().empty()) fprintf(stderr, "%s", errors(r).c_str());
    CHECK_OK(st);
    CHECK(r.load_errors().empty());
    const Registry::QuantRow* q = r.quantizer("exl3");
    REQUIRE(q != nullptr);
    CHECK_EQ(q->plugin, std::string("exl3_quant"));
}
