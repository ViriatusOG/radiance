/* rad_gbnf_check.cpp -- the GBNF mask engine's acceptance run, on a real vocabulary.
 *
 *   rad-gbnf-check --tokenizer PATH/tokenizer.json --template PATH/chat_template.jinja
 *                  [--tokenizer-config PATH/tokenizer_config.json]
 *                  [--schema FILE.json]... [--gbnf FILE]...
 *                  [--walks N] [--walk-steps N] [--passes N] [--seed N]
 *                  [--oracle cold|all|off]
 *
 * A tool as well as a test: mask cost, key remainder sizes and key counts depend on the real
 * vocabulary's piece-length and byte distributions. ctest's synthetic vocabulary covers the same
 * code against the same oracle but cannot measure a quarter-million-piece byte-level vocabulary.
 *
 * For each grammar (the chat template's tool-call grammar with parallel tool calls off and on, a
 * set of JSON-schema grammars, and any client GBNF given) it runs:
 *
 *   compile     from a cold program: wall time, plan contents, and resident-set growth;
 *   replay      realistic outputs (four agent tool calls; a document per schema) driven exactly as
 *               Sampler::build_step drives a constrained sequence at depth-3 speculation (push,
 *               accept the drafts, mask every position, roll back, accept), timing each step's
 *               host work cold and then warm;
 *   walks       random walks that draft a window of up to three tokens (mostly admissible), verify
 *               it as the engine does and accept the verified prefix;
 *   oracle      every computed mask (or every mask, with --oracle all) compared bit for bit and by
 *               status with upstream's candidate walk.
 *
 * Timing and oracle are separate passes over separately compiled programs, taking the same walks:
 * the oracle rescans the vocabulary per stack and would evict the caches a timed mask needs.
 *
 * Reports cold (computed) and warm (cached) mask latency as p50/p99/max, with computed masks at a
 * state carrying a split character listed separately (they come from the vocabulary's resume
 * index); plus per-step replay cost, compile time, memory and every mismatch. Exits 1 on any
 * mismatch. With --oracle off it is a latency run only.
 */
#include "gbnf_oracle.h"
#include "iface.h"

#include "rad_internal.h"
#include "sample/gbnf.h"
#include "sample/grammar.h"
#include "sample/vocab_view.h"
#include "text/chat.h"
#include "text/token_view.h"
#include "text/tokenizer.h"

#include "util/json.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

using namespace rad;
using ojson = Json;
using Clock = std::chrono::steady_clock;

namespace {

/* ================================================================== the vocabulary */


double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

double rss_mib() {
    long pages = 0, resident = 0;
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0.0;
    if (std::fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = 0;
    std::fclose(f);
    return (double)resident * (double)sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0);
}

/* ================================================================== what is measured */

struct Dist {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    void merge(const Dist& o) { v.insert(v.end(), o.v.begin(), o.v.end()); }
    double q(double p) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, (size_t)(p * (double)v.size()))];
    }
    double max() { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); }
    double mean() const {
        double s = 0.0;
        for (double x : v) s += x;
        return v.empty() ? 0.0 : s / (double)v.size();
    }
    std::string line() {
        char b[160];
        std::snprintf(b, sizeof b, "n=%zu p50 %.3f p99 %.3f max %.3f ms", v.size(), q(0.50),
                      q(0.99), max());
        return b;
    }
};

enum class OracleMode { Off, Cold, All };

struct Run {
    const TokenView* view = nullptr;
    OracleMode      oracle = OracleMode::Cold;
    int64_t         n_words = 0;
    Dist            cold, cold_split, warm;
    int64_t         compared = 0, mismatches = 0;
    std::vector<int32_t> structural;   /* pieces that carry structure, to steer the walks */

    /* One mask: timed and classified as computed or cached, or on the oracle pass, compared and
     * not timed. */
    int mask(const GbnfGrammar& g, uint32_t* m, const char* where, double* engine_ms = nullptr) {
        const uint64_t w0 = g.masks_walked();
        const bool     split = g.partial().n_remain > 0;
        const auto     t0 = Clock::now();
        const int      st = g.fill_mask(m, n_words);
        const double   ms = ms_since(t0);
        if (engine_ms) *engine_ms = ms;
        const bool     computed = g.masks_walked() != w0;
        if (oracle == OracleMode::Off) (computed ? (split ? cold_split : cold) : warm).add(ms);
        if (oracle == OracleMode::All || (oracle == OracleMode::Cold && computed)) {
            std::vector<uint32_t> ref((size_t)n_words);
            const int rs = gbnf_oracle_mask(g, ref.data(), n_words);
            if (rs != RAD_E_FULL && rs != RAD_E_NOMEM) {
                ++compared;
                const bool same = rs == st &&
                    (st < 0 || std::memcmp(ref.data(), m, (size_t)n_words * 4) == 0);
                if (!same) report(g, m, st, ref.data(), rs, where);
            }
        }
        return st;
    }

    void report(const GbnfGrammar& g, const uint32_t* m, int st, const uint32_t* ref, int rs,
                const char* where) {
        ++mismatches;
        if (mismatches > 8) return;
        int64_t only_e = 0, only_o = 0;
        std::string ex;
        for (int32_t t = 0; t < view->n_tokens(); ++t) {
            const bool a = (m[t >> 5] >> (t & 31)) & 1u, b = (ref[t >> 5] >> (t & 31)) & 1u;
            if (a == b) continue;
            (a ? only_e : only_o)++;
            if (only_e + only_o <= 6) {
                ex += " ";
                ex += a ? "+" : "-";
                ex += std::to_string(t) + "'" + escape(view->token_piece(t)) + "'";
            }
        }
        std::printf("  MISMATCH in %s: status %d, oracle %d; %zu stacks, partial {%u,%d}; "
                    "%lld tokens only the engine admits, %lld only the oracle:%s\n",
                    where, st, rs, g.stacks().size(), g.partial().value, g.partial().n_remain,
                    (long long)only_e, (long long)only_o, ex.c_str());
    }

    static std::string escape(const std::string& s) {
        std::string o;
        for (unsigned char c : s) {
            if (c == '\n')      o += "\\n";
            else if (c == '\t') o += "\\t";
            else if (c < 0x20 || c == 0x7f || c == '\'') {
                char b[8];
                std::snprintf(b, sizeof b, "\\x%02x", c);
                o += b;
            } else {
                o += (char)c;
            }
        }
        return o;
    }
};

/* ================================================================== grammars and texts */

const char* kTools = R"([
 {"type":"function","function":{"name":"bash","description":"Run a shell command.",
  "parameters":{"type":"object","properties":{
   "command":{"type":"string","description":"The command"},
   "timeout":{"type":"integer","description":"Timeout in ms"},
   "description":{"type":"string","description":"What it does"}},"required":["command"]}}},
 {"type":"function","function":{"name":"read_file","description":"Read a file.",
  "parameters":{"type":"object","properties":{
   "path":{"type":"string"},"offset":{"type":"integer"},"limit":{"type":"integer"}},
   "required":["path"]}}},
 {"type":"function","function":{"name":"write_file","description":"Write a file.",
  "parameters":{"type":"object","properties":{
   "path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
 {"type":"function","function":{"name":"edit_file","description":"Replace a string in a file.",
  "parameters":{"type":"object","properties":{
   "path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},
   "replace_all":{"type":"boolean"}},"required":["path","old_string","new_string"]}}},
 {"type":"function","function":{"name":"glob","description":"Find files.",
  "parameters":{"type":"object","properties":{
   "pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]}}},
 {"type":"function","function":{"name":"grep","description":"Search file contents.",
  "parameters":{"type":"object","properties":{
   "pattern":{"type":"string"},"path":{"type":"string"},"include":{"type":"string"},
   "case_sensitive":{"type":"boolean"}},"required":["pattern"]}}},
 {"type":"function","function":{"name":"todo_write","description":"Update the todo list.",
  "parameters":{"type":"object","properties":{
   "todos":{"type":"array","items":{"type":"object","properties":{
     "content":{"type":"string"},
     "status":{"type":"string","enum":["pending","in_progress","completed"]},
     "id":{"type":"string"}},"required":["content","status","id"]}}},"required":["todos"]}}}
])";

std::string go_file() {
    std::string s = "package cache\n\nimport (\n\t\"container/list\"\n\t\"sync\"\n)\n\n";
    for (int i = 0; i < 12; ++i) {
        const std::string n = std::to_string(i);
        s += "// Shard" + n + " holds a fixed-capacity LRU keyed by string.\n"
             "type Shard" + n + " struct {\n\tmu    sync.Mutex\n\tcap   int\n"
             "\tll    *list.List\n\titems map[string]*list.Element\n}\n\n"
             "func (s *Shard" + n + ") Get(key string) (any, bool) {\n"
             "\ts.mu.Lock()\n\tdefer s.mu.Unlock()\n\tif e, ok := s.items[key]; ok {\n"
             "\t\ts.ll.MoveToFront(e)\n\t\treturn e.Value.(*entry).val, true\n\t}\n"
             "\treturn nil, false\n}\n\n"
             "func (s *Shard" + n + ") Put(key string, val any) {\n"
             "\ts.mu.Lock()\n\tdefer s.mu.Unlock()\n\tif e, ok := s.items[key]; ok {\n"
             "\t\te.Value.(*entry).val = val\n\t\ts.ll.MoveToFront(e)\n\t\treturn\n\t}\n"
             "\tif s.ll.Len() >= s.cap {\n\t\tlast := s.ll.Back()\n\t\ts.ll.Remove(last)\n"
             "\t\tdelete(s.items, last.Value.(*entry).key)\n\t}\n"
             "\ts.items[key] = s.ll.PushFront(&entry{key: key, val: val})\n}\n\n";
    }
    return s;
}

/* Four agent tool calls in the XML-parameter syntax. Templates with another syntax are exercised
 * by the walks only. */
std::vector<std::string> tool_call_texts() {
    return {
        "<tool_call>\n<function=bash>\n<parameter=command>\n"
        "cd /repo && go test ./... 2>&1 | head -50\n</parameter>\n"
        "<parameter=description>\nRun the test suite\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=read_file>\n<parameter=path>\n/repo/cache/lru.go\n</parameter>\n"
        "<parameter=limit>\n200\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=write_file>\n<parameter=path>\n/repo/cache/shards.go\n"
        "</parameter>\n"
        "<parameter=content>\n" + go_file() + "\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=edit_file>\n<parameter=path>\n/repo/cache/lru.go\n</parameter>\n"
        "<parameter=old_string>\n\tif s.ll.Len() >= s.cap {\n\t\tlast := s.ll.Back()\n"
        "\t\ts.ll.Remove(last)\n</parameter>\n<parameter=new_string>\n"
        "\tfor s.ll.Len() >= s.cap && s.cap > 0 {\n\t\tlast := s.ll.Back()\n"
        "\t\ts.ll.Remove(last)\n\t\tevictions.Add(1)\n</parameter>\n</function>\n</tool_call>",
    };
}

/* JSON schemas shaped like response_format and tool arguments, each with an admitted document as a
 * model would write it. The converter puts required properties before optional ones and reads an
 * empty schema as any object; the documents follow suit. */
struct SchemaCase { const char* name; const char* schema; std::vector<std::string> docs; };

std::vector<SchemaCase> schema_cases() {
    return {
        { "tool-arguments",
          R"({"type":"object","properties":{"command":{"type":"string"},
              "timeout":{"type":"integer"},
              "env":{"type":"object","additionalProperties":{"type":"string"}},
              "background":{"type":"boolean"}},"required":["command"]})",
          { R"({"command": "grep -rn \"TODO\" src/ | sort | uniq -c | sort -rn | head -20", )"
            R"("timeout": 120000, "env": {"LANG": "C.UTF-8", "PAGER": "cat"}, )"
            R"("background": false})" } },
        { "answer-with-sources",
          R"({"type":"object","properties":{"answer":{"type":"string"},
              "confidence":{"type":"number","minimum":0,"maximum":1},
              "sources":{"type":"array","items":{"type":"object","properties":{
                  "title":{"type":"string"},"url":{"type":"string"},"year":{"type":"integer"}},
                  "required":["title","url"]}},
              "language":{"type":"string","enum":["en","de","ja","zh"]}},
              "required":["answer","confidence","sources"]})",
          { "{\"answer\": \"Die Hauptstadt ist Berlin \xe2\x80\x94 seit 1990 wieder, davor Bonn. "
            "\xe6\x9d\xb1\xe4\xba\xac\xe3\x81\xaf\xe6\x97\xa5\xe6\x9c\xac\xe3\x81\xae\xe9\xa6\x96"
            "\xe9\x83\xbd\xe3\x81\xa7\xe3\x81\x99\xe3\x80\x82 \\\"quoted\\\" \\u00e9\", "
            "\"confidence\": 0.93, "
            "\"sources\": [{\"title\": \"Einigungsvertrag\", \"url\": \"https://example.org/ev\", "
            "\"year\": 1990}, {\"title\": \"Hauptstadtbeschluss\", "
            "\"url\": \"https://example.org/hb\"}], "
            "\"language\": \"de\"}" } },
        { "records",
          R"({"type":"array","items":{"type":"object","properties":{
              "id":{"type":"integer"},"name":{"type":"string","minLength":1,"maxLength":40},
              "tags":{"type":"array","items":{"type":"string"},"maxItems":5},
              "score":{"type":["number","null"]},
              "kind":{"oneOf":[{"const":"user"},{"const":"group"}]}},
              "required":["id","name","kind"]},"minItems":1})",
          { R"([{"id": 1, "name": "alice", "kind": "user", "tags": ["admin", "ops"], )"
            R"("score": 12.5}, {"id": 2, "name": "wheel", "kind": "group", "tags": [], )"
            R"("score": null}])" } },
        { "nested-free-form",
          R"({"type":"object","properties":{"plan":{"type":"array","items":{
              "type":"object","properties":{"step":{"type":"string"},"detail":{}},
              "required":["step"]}}},"required":["plan"]})",
          { R"({"plan": [{"step": "read", "detail": {"files": ["a.go", "b.go"], "depth": 2}}, )"
            R"({"step": "edit", "detail": {"lines": [1, 2.5e3, true, null, {"k": "v"}]}}, )"
            R"({"step": "test"}]})" } },
    };
}

/* A few client grammars of the kinds /v1/completions sees. */
struct GbnfCase { std::string name, text; std::vector<std::string> docs; };

std::vector<GbnfCase> gbnf_cases() {
    return {
        { "arithmetic",
          "root ::= expr\nexpr ::= term ([-+*/] term)*\n"
          "term ::= num | \"(\" ws expr ws \")\"\nnum ::= [0-9]+ (\".\" [0-9]+)?\n"
          "ws ::= [ \\t\\n]*\n",
          { "(12.5+3)*4-(7/(2+1.25))" } },
        { "csv",
          "root ::= header row+\nheader ::= field (\",\" field)* \"\\n\"\n"
          "row ::= field (\",\" field)* \"\\n\"\n"
          "field ::= [^,\\n\"]* | \"\\\"\" ([^\"] | \"\\\"\\\"\")* \"\\\"\"\n",
          { "name,city,note\nAda,London,\"likes \"\"engines\"\"\"\nLinus,Helsinki,\n" } },
        { "list-of-words",
          "root ::= (\"- \" [^\\n]+ \"\\n\"){1,8}\n",
          { "- buy milk\n- r\xc3\xa9sum\xc3\xa9 \xe2\x9c\x93\n"
            "- \xe5\x86\x99\xe4\xbd\x9c\xe6\x96\x87\xe4\xbb\xb6\n" } },
    };
}

/* ================================================================== driving a grammar */

std::vector<int32_t> admitted(const uint32_t* m, int32_t n_tok, bool with_eog, const VocabView& v) {
    std::vector<int32_t> ids;
    for (int32_t t = 0; t < n_tok; ++t)
        if ((m[t >> 5] >> (t & 31)) & 1u)
            if (with_eog || !v.is_eog(t)) ids.push_back(t);
    return ids;
}

bool has(const uint32_t* m, int32_t t) { return (m[t >> 5] >> (t & 31)) & 1u; }

uint64_t xorshift(uint64_t& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

/* A fresh copy of the grammar text: a trailing comment changes the program cache key but not the
 * grammar, so the compile and every mask start cold. */
std::string fresh(const std::string& text) {
    static int n = 0;
    return text + "\n# rad-gbnf-check run " + std::to_string(++n) + "\n";
}

struct Replay {
    std::vector<double> step_ms;
    uint64_t            walks = 0;
    int                 refused = 0;     /* tokens of the text the mask did not admit */
};

/* Sampler::build_step's walk at depth-3 speculation, with the true continuation as the drafts:
 * every draft verifies, so each step masks four positions and accepts three tokens. */
Replay replay(Run& run, GbnfGrammar& g, const std::vector<int32_t>& toks, const char* where) {
    Replay r;
    std::vector<uint32_t> m((size_t)run.n_words);
    const uint64_t w0 = g.masks_walked();
    size_t i = 0;
    while (i < toks.size()) {
        const int n_pos = (int)std::min<size_t>(4, toks.size() - i);
        double ms = 0.0;
        auto timed = [&ms](auto&& f) {
            const auto t0 = Clock::now();
            auto x = f();
            ms += ms_since(t0);
            return x;
        };
        const bool walk = n_pos > 1;
        if (walk) timed([&] { g.push_state(); return 0; });
        bool dead = false;
        for (int k = 0; k < n_pos && !dead; ++k) {
            if (walk && k > 0 && timed([&] { return g.accept_token(toks[i + k - 1]); }) < 0) {
                dead = true;
                break;
            }
            double mask_ms = 0.0;
            const int st = run.mask(g, m.data(), where, &mask_ms);
            ms += mask_ms;
            if (st < 0) { dead = true; break; }
            if (!has(m.data(), toks[i + k])) ++r.refused;
        }
        if (walk) timed([&] { return g.rollback(); });
        const int n_acc = (int)std::min<size_t>(3, toks.size() - i);
        for (int k = 0; k < n_acc; ++k)
            if (timed([&] { return g.accept_token(toks[i + k]); }) < 0) {
                ++r.refused;
                r.walks = g.masks_walked() - w0;
                return r;
            }
        r.step_ms.push_back(ms);
        i += (size_t)n_acc;
    }
    r.walks = g.masks_walked() - w0;
    return r;
}

/* Random walks: each step drafts up to three tokens (mostly admissible), masks every position,
 * verifies as the engine does and accepts the verified prefix plus one. Returns the masks taken. */
int64_t walks(Run& run, const std::string& text, int n_walks, int n_steps, uint64_t seed,
              const char* where) {
    const TokenView& v = *run.view;
    const int32_t n_tok = v.n_tokens();
    int64_t masks = 0;
    for (int w = 0; w < n_walks; ++w) {
        std::unique_ptr<GbnfGrammar> g;
        std::string err;
        if (GbnfGrammar::create(text, "root", &v, false, {}, {}, &g, &err) < 0) return masks;
        uint64_t rng = (seed + 1) * 0x9e3779b97f4a7c15ull + (uint64_t)w * 0x632be59bd9b4e019ull;
        std::vector<std::vector<uint32_t>> m(4, std::vector<uint32_t>((size_t)run.n_words));
        auto pick = [&](const uint32_t* mask, bool admissible) -> int32_t {
            if (!admissible) return (int32_t)(xorshift(rng) % (uint64_t)n_tok);
            if (xorshift(rng) % 100 < 35) {
                std::vector<int32_t> s;
                for (int32_t t : run.structural) if (has(mask, t)) s.push_back(t);
                if (!s.empty()) return s[xorshift(rng) % s.size()];
            }
            const auto ids = admitted(mask, n_tok, false, v);
            if (ids.empty()) return -1;
            return ids[xorshift(rng) % ids.size()];
        };
        for (int step = 0; step < n_steps; ++step) {
            const int n_pos = 1 + (int)(xorshift(rng) % 4);
            std::vector<int32_t> d;
            g->push_state();
            int live = 0;
            for (int k = 0; k < n_pos; ++k) {
                if (k > 0 && g->accept_token(d[(size_t)k - 1]) < 0) break;
                if (run.mask(*g, m[(size_t)k].data(), where) < 0) break;
                ++masks;
                ++live;
                if (k + 1 < n_pos) d.push_back(pick(m[(size_t)k].data(), xorshift(rng) % 100 < 85));
                if (!d.empty() && d.back() < 0) break;
            }
            g->rollback();
            if (live == 0) break;
            size_t j = 0;
            while ((int)j + 1 < live && j < d.size() && d[j] >= 0 && has(m[j].data(), d[j])) ++j;
            bool ok = true;
            for (size_t k = 0; k < j && ok; ++k) ok = g->accept_token(d[k]) >= 0;
            if (!ok) break;
            const int32_t bonus = pick(m[j].data(), true);
            if (bonus < 0) break;                              /* only the end is admissible */
            if (g->accept_token(bonus) < 0) break;
        }
    }
    return masks;
}

/* ================================================================== one grammar */

struct Totals {
    int64_t compared = 0, mismatches = 0;
    Dist    cold, cold_split, warm, first_steps;
};

void run_grammar(Run& base, Totals& tot, const std::string& label, const std::string& text,
                 const std::vector<std::string>& docs, int passes, int n_walks, int n_steps,
                 uint64_t seed) {
    const TokenView& v = *base.view;
    std::printf("\n== %s: %zu bytes of GBNF\n", label.c_str(), text.size());

    /* Compile from nothing. */
    const std::string fresh_text = fresh(text);
    const double rss0 = rss_mib();
    const auto t0 = Clock::now();
    std::shared_ptr<const GbnfProgram> prog;
    std::string err;
    const int cs = gbnf_compile(fresh_text, "root", &v, &prog, &err);
    const double compile_ms = ms_since(t0);
    if (cs < 0) {
        std::printf("  REFUSED in %.1f ms: %s\n", compile_ms, err.c_str());
        return;
    }
    const GbnfProgramInfo info = gbnf_program_info(*prog);
    std::printf("  compile %.1f ms (mask plan %.1f ms, %.0fM units): %zu rules, %zu mask keys "
                "(%zu dense, %zu deep at most), %zu remainder nodes (largest %zu); plan %.2f MiB, "
                "resident set +%.1f MiB\n",
                compile_ms, info.plan_ms, (double)info.plan_work / 1e6, info.rules, info.keys,
                info.dense_masks, info.max_key_depth, info.rem_nodes, info.max_rem_nodes,
                (double)info.bytes / (1024.0 * 1024.0), rss_mib() - rss0);

    /* The replay texts this grammar admits. */
    std::vector<std::vector<int32_t>> texts;
    for (const std::string& d : docs) {
        std::vector<int32_t> toks;
        v.tok().encode(d, toks, false, true);
        std::unique_ptr<GbnfGrammar> probe;
        if (GbnfGrammar::create(fresh_text, "root", &v, false, {}, {}, &probe, &err) < 0) break;
        bool admitted = true;
        for (int32_t t : toks) {
            if (probe->accept_token(t) < 0) { admitted = false; break; }
        }
        if (!admitted) {
            std::printf("  (a replay text is not admitted by this grammar; walks only)\n");
            continue;
        }
        texts.push_back(std::move(toks));
    }

    /* Timing pass on the program just compiled: the first replay pass is cold, later ones warm. */
    Run run;
    run.view = base.view;
    run.oracle = OracleMode::Off;
    run.n_words = base.n_words;
    run.structural = base.structural;
    for (int pass = 0; pass < passes && !texts.empty(); ++pass) {
        Dist steps;
        std::string per;
        uint64_t walks_n = 0;
        int refused = 0;
        for (size_t c = 0; c < texts.size(); ++c) {
            std::unique_ptr<GbnfGrammar> g;
            if (GbnfGrammar::create(fresh_text, "root", &v, false, {}, {}, &g, &err) < 0) break;
            const std::string where = label + " replay " + std::to_string(c);
            Replay r = replay(run, *g, texts[c], where.c_str());
            Dist d;
            for (double x : r.step_ms) { d.add(x); steps.add(x); }
            if (pass == 0) for (double x : r.step_ms) tot.first_steps.add(x);
            char b[200];
            std::snprintf(b, sizeof b, "\n    text %zu: %zu tokens, %zu steps, %.3f ms/step mean, "
                          "max %.3f, %llu masks computed", c, texts[c].size(), r.step_ms.size(),
                          d.mean(), d.max(), (unsigned long long)r.walks);
            per += b;
            walks_n += r.walks;
            refused += r.refused;
        }
        std::printf("  replay pass %d (%s): %.3f ms/step mean, %s, %llu masks computed%s%s\n",
                    pass, pass == 0 ? "cold program" : "warm", steps.mean(), steps.line().c_str(),
                    (unsigned long long)walks_n, refused ? ", TEXT TOKENS REFUSED" : "",
                    per.c_str());
    }
    const std::string walk_label = label + " walk";
    const int64_t n_masks = walks(run, fresh_text, n_walks, n_steps, seed, walk_label.c_str());
    std::printf("  walks: %d x %d steps, %lld masks\n", n_walks, n_steps, (long long)n_masks);
    std::printf("  masks computed:     %s\n", run.cold.line().c_str());
    if (!run.cold_split.v.empty())
        std::printf("  ... at a split character: %s\n", run.cold_split.line().c_str());
    std::printf("  masks from cache:   %s\n", run.warm.line().c_str());
    tot.cold.merge(run.cold);
    tot.cold_split.merge(run.cold_split);
    tot.warm.merge(run.warm);

    /* Oracle pass on a second cold program over the same replays and walks; every mask the
     * timing pass computed is recomputed and compared. */
    if (base.oracle == OracleMode::Off) return;
    const std::string oracle_text = fresh(text);
    Run orc;
    orc.view = base.view;
    orc.oracle = base.oracle;
    orc.n_words = base.n_words;
    orc.structural = base.structural;
    for (size_t c = 0; c < texts.size(); ++c) {
        std::unique_ptr<GbnfGrammar> g;
        if (GbnfGrammar::create(oracle_text, "root", &v, false, {}, {}, &g, &err) < 0) break;
        replay(orc, *g, texts[c], (label + " replay " + std::to_string(c)).c_str());
    }
    walks(orc, oracle_text, n_walks, n_steps, seed, (label + " walk").c_str());
    std::printf("  oracle: %lld states compared, %lld mismatches\n", (long long)orc.compared,
                (long long)orc.mismatches);
    tot.compared += orc.compared;
    tot.mismatches += orc.mismatches;
}

void usage() {
    std::fprintf(stderr,
        "usage: rad-gbnf-check --tokenizer PATH/tokenizer.json "
        "--template PATH/chat_template.jinja\n"
        "                      [--tokenizer-config PATH] [--schema FILE.json]... [--gbnf FILE]...\n"
        "                      [--walks N] [--walk-steps N] [--passes N] [--seed N]\n"
        "                      [--oracle cold|all|off]\n");
}

}  /* namespace */

int main(int argc, char** argv) {
    std::string tok_path, tpl_path, cfg_path;
    std::vector<std::string> schema_files, gbnf_files;
    int n_walks = 6, n_steps = 60, passes = 2;
    uint64_t seed = 1;
    OracleMode oracle = OracleMode::Cold;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        auto num = [&](const char* flag, int64_t lo, int64_t hi) {
            return rad_flag_int("rad-gbnf-check", flag, next().c_str(), lo, hi);
        };
        if (a == "--tokenizer") tok_path = next();
        else if (a == "--template") tpl_path = next();
        else if (a == "--tokenizer-config") cfg_path = next();
        else if (a == "--schema") schema_files.push_back(next());
        else if (a == "--gbnf") gbnf_files.push_back(next());
        else if (a == "--walks") n_walks = (int)num(a.c_str(), 0, INT32_MAX);
        else if (a == "--walk-steps") n_steps = (int)num(a.c_str(), 0, INT32_MAX);
        else if (a == "--passes") passes = (int)num(a.c_str(), 0, INT32_MAX);
        else if (a == "--seed") seed = (uint64_t)num(a.c_str(), 0, INT64_MAX);
        else if (a == "--oracle") {
            const std::string m = next();
            if (m == "all") oracle = OracleMode::All;
            else if (m == "off") oracle = OracleMode::Off;
            else if (m == "cold") oracle = OracleMode::Cold;
            else {
                std::fprintf(stderr, "rad-gbnf-check: --oracle takes cold, all or off, not '%s'\n",
                             m.c_str());
                return 2;
            }
        } else { usage(); return 2; }
    }
    if (tok_path.empty() || tpl_path.empty()) { usage(); return 2; }
    if (cfg_path.empty()) {
        const size_t slash = tok_path.find_last_of('/');
        cfg_path = (slash == std::string::npos ? std::string() : tok_path.substr(0, slash + 1)) +
                   "tokenizer_config.json";
    }

    /* The report prints compiles itself, so the engine's per-compile line is suppressed; stdout is
     * line-buffered so warnings cannot land mid-line. */
    rad::log_set_level(rad::Log::Warn);
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    /* The vocabulary, as the engine builds it. */
    VocabBuild vb;
    if (parse_tokenizer_json(tok_path.c_str(), vb) < 0) {
        std::fprintf(stderr, "rad-gbnf-check: cannot read %s\n", tok_path.c_str());
        return 2;
    }
    auto vocab = std::make_shared<Vocab>();
    if (vocab->load(std::move(vb)) < 0) return 2;
    TokenView view;
    if (view.init(vocab) < 0) return 2;
    const double rss_vocab = rss_mib();

    GbnfVocabInfo vi;
    if (gbnf_prepare_vocab(&view, &vi) < 0) return 2;
    std::printf("vocabulary: %d tokens, %zu end-of-generation; index built in %.0f ms, %.1f MiB "
                "(resident set +%.1f MiB)\n", view.n_tokens(), vocab->eog().size(), vi.build_ms,
                (double)vi.bytes / (1024.0 * 1024.0), rss_mib() - rss_vocab);

    Run base;
    base.view = &view;
    base.oracle = oracle;
    base.n_words = bitmask_words(view.n_tokens());
    for (int32_t t = 0; t < view.n_tokens(); ++t) {
        const std::string& p = view.token_piece(t);
        if (!p.empty() && p.find_first_of("\"<>{}[],:\n") != std::string::npos)
            base.structural.push_back(t);
    }

    /* The chat template's tool-call grammar, with parallel tool calls off and on. */
    std::string tpl;
    if (!read_file_bytes(tpl_path, &tpl)) {
        std::fprintf(stderr, "rad-gbnf-check: cannot read %s\n", tpl_path.c_str());
        return 2;
    }
    std::string bos, eos;
    {
        std::string cfg;
        if (read_file_bytes(cfg_path, &cfg)) {
            try {
                const ojson c = ojson::parse(cfg);
                auto tokstr = [&](const char* k) -> std::string {
                    if (!c.contains(k) || c[k].is_null()) return "";
                    return c[k].is_string() ? c[k].get<std::string>() : c[k].value("content", "");
                };
                bos = tokstr("bos_token");
                eos = tokstr("eos_token");
            } catch (const std::exception&) {}
        }
    }
    ChatTemplate ct;
    if (ct.load(tpl, bos, eos) < 0) {
        std::fprintf(stderr, "rad-gbnf-check: the template does not load\n");
        return 2;
    }

    Totals tot;
    const ojson messages = ojson::array({
        ojson{ { "role", "system" }, { "content", "You are a coding agent." } },
        ojson{ { "role", "user" }, { "content", "Add sharding to the cache." } } });
    for (int par = 0; par < 2; ++par) {
        std::string grammar;
        for (const char* choice : { "auto", "required" }) {
            ChatOptions opt;
            opt.parallel_tool_calls = par != 0;
            opt.tool_choice = choice;
            ChatPrompt cp;
            if (ct.apply(messages, ojson::parse(kTools), opt, &cp) < 0) continue;
            if (cp.grammar.empty()) continue;
            grammar = cp.grammar;
            break;
        }
        const std::string label =
            std::string("tool calls, parallel_tool_calls=") + (par ? "true" : "false");
        if (grammar.empty()) {
            std::printf("\n== %s: the template produces no tool-call grammar\n", label.c_str());
            continue;
        }
        run_grammar(base, tot, label, grammar, tool_call_texts(), passes, n_walks, n_steps, seed);
    }

    /* JSON schemas: the built-in set, then any given. */
    auto run_schema = [&](const std::string& label, const std::string& schema,
                          const std::vector<std::string>& docs) {
        std::string gbnf, err;
        if (grammar_from_json_schema(schema, "root", &gbnf, &err) < 0) {
            std::printf("\n== %s: the schema does not convert: %s\n", label.c_str(), err.c_str());
            return;
        }
        run_grammar(base, tot, label, gbnf, docs, passes, n_walks, n_steps, seed);
    };
    for (const SchemaCase& sc : schema_cases())
        run_schema(std::string("json schema ") + sc.name, sc.schema, sc.docs);
    for (const std::string& f : schema_files) {
        std::string s;
        if (read_file_bytes(f, &s)) run_schema("json schema " + f, s, {});
    }

    for (const GbnfCase& gc : gbnf_cases())
        run_grammar(base, tot, "gbnf " + gc.name, gc.text, gc.docs, passes, n_walks, n_steps, seed);
    for (const std::string& f : gbnf_files) {
        std::string s;
        if (read_file_bytes(f, &s))
            run_grammar(base, tot, "gbnf " + f, s, {}, passes, n_walks, n_steps, seed);
    }

    std::printf("\n== all grammars\n");
    std::printf("  masks computed:     %s\n", tot.cold.line().c_str());
    if (!tot.cold_split.v.empty())
        std::printf("  ... at a split character: %s\n", tot.cold_split.line().c_str());
    std::printf("  masks from cache:   %s\n", tot.warm.line().c_str());
    std::printf("  replay steps, cold program: %.3f ms mean, %s\n", tot.first_steps.mean(),
                tot.first_steps.line().c_str());
    if (oracle != OracleMode::Off)
        std::printf("  oracle: %lld states compared, %lld mismatches\n", (long long)tot.compared,
                    (long long)tot.mismatches);
    std::printf("  resident set %.1f MiB\n", rss_mib());
    return tot.mismatches ? 1 : 0;
}
