/* gbnf_test.cpp -- the GBNF mask engine, held to upstream's candidate walk.
 *
 * fill_mask serves every mask from a plan precomputed at grammar compile time
 * (core/sample/gbnf_mask.cpp); tests/gbnf_oracle.cpp computes the same mask as llama.cpp does, by
 * rescanning the vocabulary. Both compute one pure function of grammar state, so at every state a
 * walk reaches they must agree bit for bit and status for status. Accepting is checked the same
 * way: the engine steps through the mask plan's automaton, the oracle through the stack machinery
 * one code point at a time, and every accepted token must leave both in the same state.
 *
 * States are reached as the sampler reaches them. The walks drive a grammar as
 * Sampler::build_step does under speculation (push the state, advance over a window of drafts,
 * mask every position, roll back, accept the verified prefix plus one token), with drafts mostly
 * but not always admissible, so dead windows, rollbacks over a failed accept and the mask cache
 * are all exercised.
 *
 * The vocabulary is shaped like a real one, since the plan's keys and remainder tries depend on
 * piece lengths and byte distributions. bpe_like() mimics byte-level BPE: every byte as a piece,
 * pieces ending or beginning mid-character, multi-byte characters whole and split, words with and
 * without a leading space, long runs, special and end-of-generation tokens, and textless control
 * tokens. rad-gbnf-check runs the same comparison against a real tokenizer.json.
 *
 * Grammars: hand-written edge cases, JSON-schema conversions, the tool-call grammars of the three
 * chat templates under data/gbnf and of Gemma 4's declared format, and random grammars. Other
 * cases pin the plan's size guarantees and refusals, and the masks' work per call against a
 * per-mask vocabulary walk.
 *
 * The oracle is the cost: it rescans the vocabulary per stack per mask. Within one grammar its
 * answers are memoised by state (OracleMemo), so a state the walks revisit is compared against the
 * oracle's first answer; the engine still computes every mask and every accept.
 */
#include "rad_test.h"
#include "work_counter.h"

#include "gbnf_oracle.h"
#include "sample/gbnf.h"
#include "sample/gbnf_internal.h"
#include "sample/grammar.h"
#include "sample/vocab_view.h"
#include "text/chat.h"
#include "text/chatfmt.h"

#include <arch/rad_gemma4_chat.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace rad;
using ojson = nlohmann::ordered_json;

namespace {

/* ================================================================== randomness */

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull) {
        if (!s) s = 1;
    }
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    size_t below(size_t n) { return n ? (size_t)(next() % n) : 0; }
    bool   chance(int pct) { return (int)(next() % 100) < pct; }
    template <class T> const T& pick(const std::vector<T>& v) { return v[below(v.size())]; }
};

/* ======================================================== a vocabulary like a real one */

const char* kMultibyte[] = {
    "\xc3\xa9", "\xc3\xbc", "\xc3\x9f", "\xc3\xb1", "\xce\xa9", "\xd1\x8f",    /* é ü ß ñ Ω я */
    "\xe6\x97\xa5", "\xe6\x9c\xac", "\xe4\xb8\xad", "\xe6\x96\x87", "\xe5\xad\x97",  /* 日 本 中 文 字 */
    "\xe2\x80\x94", "\xe2\x80\x9c", "\xe2\x82\xac", "\xe2\x9c\x93",                  /* — “ € ✓ */
    "\xf0\x9f\x98\x80", "\xf0\x9f\x99\x8f", "\xf0\x9f\x9a\x80",                /* three emoji */
};

struct BpeVocab {
    SimpleVocab              v;
    std::vector<int32_t>     structural;   /* pieces carrying structure, to steer walks by */
    std::vector<char>        nul;          /* id -> the piece decodes to U+0000 before its end */
    std::unordered_map<std::string, int32_t> by_piece;
    size_t                   longest = 0;
    double                   mean_bytes = 0.0;

    int32_t add(const std::string& p, bool eog = false) {
        auto it = by_piece.find(p);
        if (it != by_piece.end() && !eog) return it->second;
        const int32_t id = v.add(p, eog);
        by_piece.emplace(p, id);
        longest = std::max(longest, p.size());
        return id;
    }

    /* Greedy longest match: enough to tokenise a realistic text for a replay. */
    std::vector<int32_t> tokenize(const std::string& text) const {
        std::vector<int32_t> out;
        size_t pos = 0;
        while (pos < text.size()) {
            size_t n = std::min(longest, text.size() - pos);
            for (; n > 0; --n) {
                auto it = by_piece.find(text.substr(pos, n));
                if (it != by_piece.end() && !v.is_eog(it->second)) {
                    out.push_back(it->second);
                    break;
                }
            }
            if (n == 0) return {};
            pos += n;
        }
        return out;
    }
};

std::unique_ptr<BpeVocab> bpe_like(uint64_t seed, size_t n_target) {
    auto b = std::make_unique<BpeVocab>();
    Rng rng(seed);

    /* Every byte, as in byte-level BPE: lone lead and continuation bytes begin or end
     * mid-character. NUL is a control token. */
    for (int c = 1; c < 256; ++c) b->add(std::string(1, (char)c));

    /* Control tokens with no text, and one that starts with a NUL; special and end-of-generation
     * tokens, whose pieces are rendered. */
    b->v.add("");
    b->v.add("");
    b->v.add(std::string("\0x", 2));
    /* Pieces with an inner NUL, which the decoder treats as their end: after plain text, after a
     * character's head, and after the bytes that finish one. */
    b->add(std::string("a\0b", 3));
    b->add(std::string("\xe6\0z", 3));
    b->add(std::string("\x97\0y", 3));
    b->add(std::string("\x97\xa5\0q", 4));
    /* Overlong sequences: heads carrying no bits, and pieces that finish them as U+0000. */
    for (const char* o : { "\xc0\x80", "\xc0\x80z", "\xe0\x80", "\x80\x80", "\x80\x80q",
                           "\xf0\x80\x80", "\x80z" })
        b->add(o);
    for (const char* s : { "<|im_start|>", "<tool_call>", "</tool_call>", "<think>", "</think>" })
        b->add(s);
    b->add("<|im_end|>", true);
    b->add("<|endoftext|>", true);

    /* The pieces structured output is made of. */
    for (const char* s : {
             "{\"", "\":", "\": \"", "\",", "\", \"", "\"}", "\"]", "},", "],", "{}", "[]", "[{",
             "}]", "}}", "\"", "\\\"", "\\\\", "\\n", "\\t", "\\u", "\\u00", "\n", "\n\n", "\n\n\n",
             "\t", "\t\t", " ", "  ", "   ", "    ", "        ", ">\n", "</", "/>", "<", ">",
             "=", "==", "====", "----", "//", "/*", "*/", "();", "),", "):", "=>", "->", "::",
             "&&", "||", "!=",
             "<=", ">=", "...", "e+", "E-", ".5", "0.", "true", "false", "null", " true", " false",
             " null", "<function=", "<parameter=", "</parameter>", "</function>", "parameter",
             "<|tool_call>", "<tool_call|>", "<|\"|>", "call:",
             "function", "tool", "_call", "\">", "</param>", "<param name=\"", "<function name=\"",
             " {", " }", " [", " ]", "\":\"", "\",\"", ",\"", "\":{", "\":[", ", ", ": " })
        b->add(s);
    for (const char* s : { "ab", "abc", "xy", "the", " the", "dog", "held", "other" }) b->add(s);
    for (int i = 0; i < 10; ++i) b->add(std::to_string(i));
    for (const char* s : { "00", "10", "12", "99", "123", "2024", "-1", "0x", "1.5" }) b->add(s);

    /* Multi-byte characters: whole, after a space, beside ASCII, in pairs, and split at every byte
     * (the head alone ends mid-character, the tail alone begins mid-character). */
    for (const char* m : kMultibyte) {
        const std::string s(m);
        b->add(s);
        b->add(" " + s);
        b->add("a" + s);
        b->add(s + "s");
        for (size_t k = 1; k < s.size(); ++k) {
            b->add(s.substr(0, k));
            b->add(s.substr(k));
            b->add("x" + s.substr(0, k));
            b->add(s.substr(k) + "y");
        }
        for (const char* m2 : kMultibyte) b->add(s + m2);
    }
    b->add("caf\xc3\xa9");
    b->add("r\xc3\xa9sum\xc3\xa9");
    b->add("na\xc3\xafve");

    /* Long pieces: whitespace and rule runs, as in code vocabularies. */
    for (int n : { 12, 16, 24, 32, 48, 64 }) b->add(std::string((size_t)n, ' '));
    for (int n : { 8, 16, 40 }) {
        b->add(std::string((size_t)n, '='));
        b->add(std::string((size_t)n, '-'));
    }
    b->add(std::string(20, '\n'));

    /* Words built from syllables with a realistic subword length spread: with and without a
     * leading space, capitalised, suffixed, followed by punctuation. */
    const std::vector<std::string> onset = { "", "b", "c", "d", "f", "g", "h", "k", "l", "m", "n",
                                             "p", "r", "s", "t", "v", "w", "st", "th", "ch", "sh",
                                             "tr", "pr", "gr", "qu" };
    const std::vector<std::string> vowel = { "a", "e", "i", "o", "u", "ou", "ea", "ai", "io", "y" };
    const std::vector<std::string> coda  = { "", "", "n", "r", "s", "t", "l", "nd", "st", "ng",
                                             "ck", "x" };
    const std::vector<std::string> tail  = { "ing", "ed", "s", "tion", "er", "ly", "_id", "()",
                                             "\",", "\":", ",", ".", ":", "\"" };
    while (b->v.n_tokens() < (int32_t)n_target) {
        std::string w;
        const size_t syl = 1 + rng.below(3);
        for (size_t k = 0; k < syl; ++k) w += rng.pick(onset) + rng.pick(vowel) + rng.pick(coda);
        if (rng.chance(15)) w[0] = (char)std::toupper((unsigned char)w[0]);
        if (rng.chance(12)) w += rng.pick(tail);
        if (rng.chance(50)) w = " " + w;
        if (rng.chance(3)) w += rng.pick(std::vector<std::string>(std::begin(kMultibyte),
                                                                  std::end(kMultibyte)));
        b->add(w);
    }
    /* Two pieces with the same text, as real vocabularies have. */
    b->v.add("dup");
    b->v.add("dup");

    size_t bytes = 0, n = 0;
    b->nul.assign((size_t)b->v.n_tokens(), 0);
    for (int32_t id = 0; id < b->v.n_tokens(); ++id) {
        const std::string& p = b->v.token_piece(id);
        if (p.empty()) continue;
        const auto dec = gbnf::decode_utf8(p, PartialUtf8{});
        for (size_t i = 0; i + 1 < dec.first.size(); ++i)
            if (dec.first[i] == 0) b->nul[(size_t)id] = 1;
        bytes += p.size();
        ++n;
        if (p.find_first_of("\"<>{}[],:\n\\") != std::string::npos) b->structural.push_back(id);
    }
    b->mean_bytes = (double)bytes / (double)n;
    return b;
}

/* ================================================================== the comparison */

struct Tally {
    int64_t masks = 0;       /* compared with the oracle */
    int64_t partial = 0;     /* of those, at a state carrying a split character */
    int64_t dead = 0;        /* of those, where both admitted no token */
    int64_t skipped = 0;     /* the oracle ran out of its work bound */
    int64_t bounded = 0;     /* the engine refused a state past its bounds that the oracle
                              * (one stack at a time) could still answer */
    int64_t grammars = 0;
    int64_t mismatches = 0;
    int64_t accepts = 0;     /* accepted tokens whose new state was compared with the oracle's */
};

std::string show_piece(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c >= 0x20 && c < 0x7f && c != '\'') { o += (char)c; continue; }
        char b[8];
        std::snprintf(b, sizeof b, "\\x%02x", c);
        o += b;
    }
    return o;
}

/* The oracle's work bound per mask: enough for every non-pathological grammar here, and about a
 * tenth of a second of rescanning otherwise. */
constexpr uint64_t kOracleWork = 1ull << 24;

/* The oracle's answers for one grammar text over one vocabulary, by state. A mask is a function of
 * the stack set and the carried partial character; an accept also of the token. Stacks are keyed
 * by element index into the rule arena, not by address, so the key stays valid when a later walk
 * of the same text runs on a recompiled program (compiling is deterministic). Scoped to one text,
 * so no key outlives what it describes. */
struct OracleMemo {
    struct Mask { int status; std::vector<uint32_t> bits; };
    struct Next { int status; std::vector<std::vector<uint32_t>> stacks; PartialUtf8 partial; };
    std::map<std::vector<uint32_t>, Mask> masks;
    std::map<std::vector<uint32_t>, Next> accepts;
};

std::vector<std::vector<uint32_t>> stack_indices(const GbnfGrammar& g, const GbnfStacks& st) {
    const gbnf::GbnfRuleSet& rules = gbnf::program_rules(g.program());
    std::vector<std::vector<uint32_t>> out;
    out.reserve(st.size());
    for (const GbnfStack& s : st) {
        std::vector<uint32_t> v;
        v.reserve(s.size());
        for (const GbnfElement* e : s) v.push_back(rules.index(e));
        out.push_back(std::move(v));
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<uint32_t> state_key(const GbnfGrammar& g) {
    std::vector<uint32_t> k = { g.partial().value, (uint32_t)g.partial().n_remain };
    for (const auto& v : stack_indices(g, g.stacks())) {
        k.push_back((uint32_t)v.size());
        k.insert(k.end(), v.begin(), v.end());
    }
    return k;
}

int oracle_mask(const GbnfGrammar& g, uint32_t* bits, int64_t n_words, OracleMemo* memo) {
    if (!memo) return gbnf_oracle_mask(g, bits, n_words, kOracleWork);
    const std::vector<uint32_t> key = state_key(g);
    auto it = memo->masks.find(key);
    if (it == memo->masks.end()) {
        OracleMemo::Mask m{ 0, std::vector<uint32_t>((size_t)n_words) };
        m.status = gbnf_oracle_mask(g, m.bits.data(), n_words, kOracleWork);
        it = memo->masks.emplace(key, std::move(m)).first;
    }
    std::memcpy(bits, it->second.bits.data(), (size_t)n_words * 4);
    return it->second.status;
}

/* The engine's mask at g's state against the oracle's. Returns the engine's status.
 *
 * One allowed difference: the engine answers through unions of its stacks' successors, and in a
 * sufficiently ambiguous grammar (a nullable star under a repetition forks a stack per split of
 * the text) those unions can exceed the stack-set bound, while the oracle holds one stack at a
 * time. The engine then fails the request, as its bounds promise. Such cases are counted and only
 * random grammars may reach them; realistic grammars must have zero. */
int compare(const GbnfGrammar& g, const VocabView& v, std::vector<uint32_t>& a,
            std::vector<uint32_t>& b, Tally& t, const std::string& what,
            OracleMemo* memo = nullptr) {
    const int64_t n_words = (int64_t)a.size();
    const int sa = g.fill_mask(a.data(), n_words);
    const int sb = oracle_mask(g, b.data(), n_words, memo);
    if (sb == RAD_E_FULL || sb == RAD_E_NOMEM) { ++t.skipped; return sa; }
    if ((sa == RAD_E_FULL || sa == RAD_E_NOMEM) && sb >= 0) {
        ++t.bounded;
        return sa;
    }
    ++t.masks;
    if (g.partial().n_remain > 0) ++t.partial;
    if (sa == RAD_E_STATE && sb == RAD_E_STATE) ++t.dead;
    const bool same = sa == sb && (sa < 0 || std::memcmp(a.data(), b.data(), a.size() * 4) == 0);
    CHECK(same);
    if (!same && ++t.mismatches <= 4) {
        std::string ex;
        int shown = 0;
        for (int32_t id = 0; id < v.n_tokens() && shown < 4; ++id) {
            const bool x = (a[(size_t)id >> 5] >> (id & 31)) & 1u;
            const bool y = (b[(size_t)id >> 5] >> (id & 31)) & 1u;
            if (x == y) continue;
            ex += std::string(" ") + (x ? "+" : "-") + std::to_string(id) + "'" +
                  show_piece(v.token_piece(id)) + "'";
            ++shown;
        }
        std::fprintf(stderr,
                     "    %s: engine status %d, oracle %d, partial {%u,%d}, %zu stacks;%s\n",
                     what.c_str(), sa, sb, g.partial().value, g.partial().n_remain,
                     g.stacks().size(), ex.c_str());
    }
    return sa;
}

bool has(const std::vector<uint32_t>& m, int32_t id) {
    return id >= 0 && ((m[(size_t)id >> 5] >> (id & 31)) & 1u);
}

/* Accept `id` and compare the resulting state with the oracle's: status, stack set and carried
 * partial character. Returns the engine's status. A side past its bounds is not compared (see
 * compare()). */
int accept(GbnfGrammar& g, int32_t id, Tally& t, OracleMemo* memo = nullptr) {
    const bool cmp = !g.awaiting_trigger() && !g.vocab()->is_eog(id);
    OracleMemo::Next ref{ RAD_OK, {}, {} };
    if (cmp) {
        std::vector<uint32_t> key;
        const OracleMemo::Next* hit = nullptr;
        if (memo) {
            key = state_key(g);
            key.push_back((uint32_t)id);
            const auto it = memo->accepts.find(key);
            if (it != memo->accepts.end()) hit = &it->second;
        }
        if (hit) {
            ref = *hit;
        } else {
            GbnfStacks st;
            ref.status = gbnf_oracle_accept(g, id, &st, &ref.partial);
            ref.stacks = stack_indices(g, st);
            if (memo) memo->accepts.emplace(std::move(key), ref);
        }
    }
    const int rs = ref.status;
    const PartialUtf8 rp = ref.partial;
    const int as = g.accept_token(id);
    if (!cmp || rs == RAD_E_FULL || rs == RAD_E_NOMEM || as == RAD_E_FULL || as == RAD_E_NOMEM)
        return as;
    ++t.accepts;
    bool same = (as >= 0) == (rs >= 0);
    if (same && as >= 0) {
        same = stack_indices(g, g.stacks()) == ref.stacks && g.partial().value == rp.value &&
               g.partial().n_remain == rp.n_remain;
    }
    CHECK(same);
    if (!same && ++t.mismatches <= 4)
        std::fprintf(stderr, "    accept %d '%s': engine status %d with %zu stacks, oracle %d "
                     "with %zu\n", id, show_piece(g.vocab()->token_piece(id)).c_str(), as,
                     g.stacks().size(), rs, ref.stacks.size());
    return as;
}

/* Whether the mask may admit a token that accept then refuses. Upstream's mask and accept read a
 * piece differently in two places, and the engine reproduces the mask exactly:
 *
 *   - a token terminal (`<[id]>`) is applied by the mask wherever reached, mid-piece included,
 *     but by accept only at a token's start;
 *   - a piece decoding to U+0000 before its end (an overlong sequence, finished or carried in) is
 *     read by the mask as ending at that 0, but fed whole by accept.
 *
 * Anywhere else, an admitted token must be accepted. */
bool accept_may_refuse(const GbnfGrammar& g, const std::string& gbnf, const VocabView& v,
                       int32_t id) {
    if (gbnf.find("<[") != std::string::npos) return true;
    const auto dec = gbnf::decode_utf8(v.token_piece(id), g.partial());
    for (size_t i = 0; i + 1 < dec.first.size(); ++i)
        if (dec.first[i] == 0) return true;
    return false;
}

/* A token the mask admits: sometimes a structural one, so walks pass a grammar's delimiters
 * instead of circling its widest loop; otherwise any. -1 if only end of generation is admissible.
 * Pieces decoding to U+0000 are admitted almost everywhere but rarely accepted
 * (accept_may_refuse), so they are taken rarely. */
int32_t pick_admitted(const std::vector<uint32_t>& m, const BpeVocab& bv, Rng& rng) {
    if (rng.chance(40)) {
        std::vector<int32_t> s;
        for (int32_t id : bv.structural) if (has(m, id)) s.push_back(id);
        if (!s.empty()) return rng.pick(s);
    }
    const bool take_nul = rng.chance(2);
    std::vector<int32_t> ids;
    for (int32_t id = 0; id < bv.v.n_tokens(); ++id)
        if (has(m, id) && !bv.v.is_eog(id) && (take_nul || !bv.nul[(size_t)id])) ids.push_back(id);
    return ids.empty() ? -1 : rng.pick(ids);
}

/* Walks as the sampler drives them. Each step drafts up to three tokens (mostly admissible,
 * sometimes arbitrary), pushes the state, advances over the drafts masking every position, rolls
 * back, then accepts what verification would: the drafts before the first refused one, plus one
 * token. */
void walk_and_compare(const std::string& gbnf, const BpeVocab& bv, uint64_t seed, int n_walks,
                      int n_steps, Tally& t, const std::string& name) {
    const VocabView& v = bv.v;
    OracleMemo memo;
    const int64_t n_words = bitmask_words(v.n_tokens());
    std::vector<std::vector<uint32_t>> m(4, std::vector<uint32_t>((size_t)n_words));
    std::vector<uint32_t> ref((size_t)n_words);
    for (int w = 0; w < n_walks; ++w) {
        std::unique_ptr<GbnfGrammar> g;
        std::string err;
        REQUIRE(GbnfGrammar::create(gbnf, "root", &v, false, {}, {}, &g, &err) >= 0);
        Rng rng(seed * 1000 + (uint64_t)w);
        for (int step = 0; step < n_steps; ++step) {
            const int n_pos = 1 + (int)rng.below(4);
            std::vector<int32_t> d;
            g->push_state();
            int live = 0;
            for (int k = 0; k < n_pos; ++k) {
                if (k > 0 && accept(*g, d[(size_t)k - 1], t, &memo) < 0) {
                    /* A refused draft leaves no parse; the mask there must agree. */
                    compare(*g, v, m[(size_t)k], ref, t, name + " after a refused draft", &memo);
                    break;
                }
                const int st = compare(*g, v, m[(size_t)k], ref, t,
                                       name + " step " + std::to_string(step) + " pos " +
                                           std::to_string(k), &memo);
                if (st < 0) break;
                ++live;
                if (k + 1 < n_pos) {
                    int32_t x = rng.chance(85) ? pick_admitted(m[(size_t)k], bv, rng)
                                               : (int32_t)rng.below((size_t)v.n_tokens());
                    if (x < 0) break;
                    d.push_back(x);
                }
            }
            CHECK_OK(g->rollback());
            if (live == 0) break;
            size_t j = 0;
            while ((int)j + 1 < live && j < d.size() && has(m[j], d[j])) ++j;
            bool ok = true;
            for (size_t k = 0; k < j && ok; ++k) {
                const bool quirk = accept_may_refuse(*g, gbnf, v, d[k]);
                const int  as = accept(*g, d[k], t, &memo);
                ok = as >= 0;
                CHECK(ok || quirk || as == RAD_E_FULL || as == RAD_E_NOMEM);
            }
            if (!ok) break;
            const int32_t bonus = pick_admitted(m[j], bv, rng);
            if (bonus < 0) break;
            const bool quirk = accept_may_refuse(*g, gbnf, v, bonus);
            const int  as = accept(*g, bonus, t, &memo);
            if (as == RAD_E_FULL || as == RAD_E_NOMEM) break;     /* the parse exceeded its bounds */
            CHECK(as >= 0 || quirk);
            if (as < 0) break;
        }
    }
    ++t.grammars;
}

/* A replay: a realistic tokenised output driven at depth-3 speculation with the true continuation
 * as drafts. Every draft verifies, so each step masks four positions and accepts three tokens,
 * the steady state of a speculating constrained request. */
void replay_and_compare(const std::string& gbnf, const BpeVocab& bv,
                        const std::vector<int32_t>& toks, Tally& t, const std::string& name) {
    const VocabView& v = bv.v;
    const int64_t n_words = bitmask_words(v.n_tokens());
    std::vector<uint32_t> m((size_t)n_words), ref((size_t)n_words);
    OracleMemo memo;
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    REQUIRE(GbnfGrammar::create(gbnf, "root", &v, false, {}, {}, &g, &err) >= 0);
    size_t i = 0;
    while (i < toks.size()) {
        const int n_pos = (int)std::min<size_t>(4, toks.size() - i);
        g->push_state();
        for (int k = 0; k < n_pos; ++k) {
            if (k > 0 && accept(*g, toks[i + (size_t)k - 1], t, &memo) < 0) { CHECK(false); break; }
            const std::string where = name + " token " + std::to_string(i + (size_t)k);
            if (compare(*g, v, m, ref, t, where, &memo) < 0) break;
            CHECK(has(m, toks[i + (size_t)k]));
        }
        CHECK_OK(g->rollback());
        const size_t n_acc = std::min<size_t>(3, toks.size() - i);
        for (size_t k = 0; k < n_acc; ++k) REQUIRE(accept(*g, toks[i + k], t, &memo) >= 0);
        i += n_acc;
    }
    ++t.grammars;
}

void report(const char* what, const Tally& t) {
    std::fprintf(stderr, "    %s: %lld grammars, %lld masks compared (%lld at a split character, "
                 "%lld dead), %lld accepts; %lld beyond the oracle's bound, %lld beyond the "
                 "engine's\n", what,
                 (long long)t.grammars, (long long)t.masks, (long long)t.partial, (long long)t.dead,
                 (long long)t.accepts, (long long)t.skipped, (long long)t.bounded);
}

/* ================================================================== grammars */

std::string slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string fixture(const char* name) {
    return std::string(RAD_GBNF_TEST_DATA) + "/" + name;
}

const char* kTools = R"([
 {"type":"function","function":{"name":"bash","description":"Run a shell command.",
  "parameters":{"type":"object","properties":{"command":{"type":"string"},
   "timeout":{"type":"integer"},"description":{"type":"string"}},"required":["command"]}}},
 {"type":"function","function":{"name":"read_file","description":"Read a file.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},
   "offset":{"type":"integer"},"limit":{"type":"integer"}},"required":["path"]}}},
 {"type":"function","function":{"name":"write_file","description":"Write a file.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},
   "content":{"type":"string"}},"required":["path","content"]}}},
 {"type":"function","function":{"name":"edit_file","description":"Replace a string.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},
   "old_string":{"type":"string"},"new_string":{"type":"string"},
   "replace_all":{"type":"boolean"}},"required":["path","old_string","new_string"]}}},
 {"type":"function","function":{"name":"todo_write","description":"Update the todo list.",
  "parameters":{"type":"object","properties":{"todos":{"type":"array","items":{
   "type":"object","properties":{"content":{"type":"string"},
    "status":{"type":"string","enum":["pending","in_progress","completed"]}},
   "required":["content","status"]}}},"required":["todos"]}}}
])";

struct TemplateCase { const char* file; const char* bos; const char* eos; };
const TemplateCase kTemplates[] = {
    { "qwen3.8.jinja", "", "<|im_end|>" },
    { "qwen3.6.jinja", "", "<|im_end|>" },
    { "minicpm5.jinja", "<s>", "</s>" },
};

/* The tool-call grammar a template generates for kTools, parallel calls off and on, with
 * tool_choice "auto" or, where auto leaves the reply unconstrained, "required". Then Gemma 4's,
 * from the format its architecture declares (object arguments, `<|"|>` string delimiters and bare
 * keys) rather than from a template. */
std::vector<std::pair<std::string, std::string>> template_grammars() {
    std::vector<std::pair<std::string, std::string>> out;
    const ojson messages = ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } });
    for (const TemplateCase& tc : kTemplates) {
        ChatTemplate ct;
        if (ct.load(slurp(fixture(tc.file)), tc.bos, tc.eos) < 0) continue;
        for (int par = 0; par < 2; ++par) {
            for (const char* choice : { "auto", "required" }) {
                ChatOptions opt;
                opt.parallel_tool_calls = par != 0;
                opt.tool_choice = choice;
                ChatPrompt cp;
                if (ct.apply(messages, ojson::parse(kTools), opt, &cp) < 0) continue;
                if (cp.grammar.empty()) continue;
                out.push_back({ std::string(tc.file) + (par ? " parallel " : " single ") + choice,
                                cp.grammar });
                break;
            }
        }
    }
    ChatFormat f;
    std::string why;
    if (chat_format_from_abi(rad::arch::gemma4_chat_format(), &f, &why) >= 0) {
        const ojson tools = ojson::parse(kTools);
        for (int par = 0; par < 2; ++par) {
            ChatGrammarRequest gq;
            gq.tools = &tools;
            gq.parallel_tool_calls = par != 0;
            ChatGrammar g;
            if (chat_format_grammar(f, gq, &g, &why) < 0 || g.text.empty()) continue;
            out.push_back({ std::string("gemma4") + (par ? " parallel" : " single"), g.text });
        }
    }
    return out;
}

const char* kSchemas[] = {
    R"({"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer"},
        "env":{"type":"object","additionalProperties":{"type":"string"}}},"required":["command"]})",
    R"({"type":"object","properties":{"answer":{"type":"string"},
        "confidence":{"type":"number","minimum":0,"maximum":1},
        "sources":{"type":"array","items":{"type":"object","properties":{"title":{"type":"string"},
            "year":{"type":"integer"}},"required":["title"]},"maxItems":4},
        "lang":{"enum":["en","de","ja"]}},"required":["answer","sources"]})",
    R"({"type":"array","items":{"type":"object","properties":{"id":{"type":"integer"},
        "name":{"type":"string","minLength":1,"maxLength":12},"tags":{"type":"array",
        "items":{"type":"string"}},"score":{"type":["number","null"]},
        "kind":{"oneOf":[{"const":"user"},{"const":"group"}]}},"required":["id","name","kind"]}})",
    R"({"anyOf":[{"type":"object","properties":{"name":{"const":"a"},"arguments":{"type":"object",
        "properties":{"x":{"type":"boolean"}}}},"required":["name"]},
        {"type":"string","pattern":"^[a-z]{2,5}-[0-9]+$"}]})",
    R"({"$defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},
        "kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["v"]}},
        "$ref":"#/$defs/node"})",
    R"({"type":"object","properties":{"s":{"type":"string","maxLength":200},
        "code":{"type":"string","minLength":90}},"required":["s","code"]})",
};

/* Hand-written edge-case grammars. `%T`/`%N` are replaced with two vocabulary token ids, for the
 * token syntax. */
const char* kTricky[][2] = {
    { "recursion",       "root ::= \"(\" root \")\" | \"x\"\n" },
    { "stack growth",    "root ::= \"[\" root \"]\" root | \"\"\n" },
    { "nested repeats",  "root ::= (\"a\" (\"b\" \"c\"?)* )+ \"d\"\n" },
    { "nullable star",   "root ::= (\"a\"?)* \"b\"\n" },
    { "ambiguity",       "root ::= \"ab\" \"c\" | \"a\" [b-c] \"c\" | \"a\" \"b\"?\n" },
    { "empty alts",      "root ::= \"a\" x \"b\"\nx ::= | \"c\" | y\ny ::= \"d\" |\n" },
    { "counted",         "root ::= [0-9]{2,4} \"-\" [a-f]{0,3} \"x\"{3} \" \"?\n" },
    { "ranges",          "root ::= [a-zA-Z0-9_]+ ([\\u00C0-\\u00FF] | [\\u4E00-\\u9FFF] | "
                         "\"\xc3\xa9\")* \"!\"\n" },
    { "negated",         "root ::= \"\\\"\" [^\"\\\\\\n]* \"\\\"\"\n" },
    { "negated wide",    "root ::= [^a-z\\u4E00-\\u9FFF]+ \".\"\n" },
    { "any char",        "root ::= . . (\"x\" | .) \"\\n\"\n" },
    { "multibyte",       "root ::= (\"\xe6\x97\xa5\xe6\x9c\xac\" | [\\U0001F600-\\U0001F64F] | "
                         "\"\xc3\xbc\")+ \"s\"?\n" },
    { "above bmp",       "root ::= [\\u0080-\\U0010FFFF]+ \"a\"\n" },
    { "escapes",         "root ::= \"\\x41\\u00e9\" [\\x01-\\x1F]? \"\\n\" \"\\t\"*\n" },
    { "long literal",    "root ::= \"the quick brown fox jumps over the lazy dog\" "
                         "\" \"? \"!\"\n" },
    { "whitespace",      "root ::= \"a\" [ \\t\\n]* \"b\" [ \\t]{0,20} \"c\" ws\nws ::= \" \"*\n" },
    { "deep nesting",    "root ::= ((((((((((((\"a\" | \"b\"))))))))))))+ ((((\"c\"))))\n" },
    { "json by hand",
      "root ::= value\n"
      "value ::= object | array | string | number | \"true\" | \"false\" | \"null\"\n"
      "object ::= \"{\" ws (string ws \":\" ws value (ws \",\" ws string ws \":\" ws value)*)? "
      "ws \"}\"\n"
      "array ::= \"[\" ws (value (ws \",\" ws value)*)? ws \"]\"\n"
      "string ::= \"\\\"\" ([^\"\\\\] | \"\\\\\" [\"\\\\/bfnrt])* \"\\\"\"\n"
      "number ::= \"-\"? [0-9]+ (\".\" [0-9]+)?\nws ::= [ \\t\\n]*\n" },
    { "token syntax",    "root ::= <[%T]> \"x\"* | !<[%N]> \"y\" | \"z\" <[%N]>\n" },
    { "token mid-piece", "root ::= \"a\" <[%T]> | \"a\" !<[%N]> \"b\"\n" },
    { "long bounded",    "root ::= \"\\\"\" [^\"]{0,300} \"\\\"\"\n" },
    { "long required",   "root ::= [a-z]{150} \";\"\n" },
};

std::string with_tokens(std::string g, const BpeVocab& bv) {
    const std::string t = std::to_string(bv.by_piece.at("ab"));
    const std::string n = std::to_string(bv.by_piece.at("x"));
    for (size_t p; (p = g.find("%T")) != std::string::npos;) g.replace(p, 2, t);
    for (size_t p; (p = g.find("%N")) != std::string::npos;) g.replace(p, 2, n);
    return g;
}

/* ================================================================== random grammars
 *
 * Rules of literals, character classes (negated, multi-byte, above the BMP), `.`, groups under
 * every repetition, forward and backward rule references, token terminals and empty alternatives.
 * A reference in first position only points forward, so left recursion is rare; what the compiler
 * refuses anyway is counted, not failed. */
struct GrammarGen {
    Rng& rng;
    int  n_rules;
    const BpeVocab& bv;

    std::string literal() {
        static const char* atoms[] = { "a", "b", "c", "ab", "the", " ", "\\n", "\\\"", "<", "/",
                                       ">", "{", "}", ":", ",", "0", "7", "\xc3\xa9",
                                       "\xe6\x97\xa5", "\xf0\x9f\x98\x80", "x", "y" };
        std::string s = "\"";
        const size_t n = 1 + rng.below(3);
        for (size_t i = 0; i < n; ++i) s += atoms[rng.below(sizeof atoms / sizeof *atoms)];
        return s + "\"";
    }
    std::string cls() {
        static const char* parts[] = { "a-c", "x", "0-9", " ", "\\n", "\"", "\\\\", "<", "/",
                                       "\\u00C0-\\u00FF", "\\u4E00-\\u9FFF",
                                       "\\U0001F600-\\U0001F64F",
                                       "a-z", "{}", ":", "b" };
        std::string s = rng.chance(30) ? "[^" : "[";
        const size_t n = 1 + rng.below(3);
        for (size_t i = 0; i < n; ++i) s += parts[rng.below(sizeof parts / sizeof *parts)];
        return s + "]";
    }
    std::string repeat() {
        switch (rng.below(7)) {
            case 0: return "*";
            case 1: return "+";
            case 2: return "?";
            case 3:
                return "{" + std::to_string(rng.below(3)) + "," +
                       std::to_string(2 + rng.below(4)) + "}";
            case 4: return "{" + std::to_string(1 + rng.below(3)) + "}";
            default: return "";
        }
    }
    std::string item(int self, bool first, int depth) {
        const size_t k = rng.below(depth > 2 ? 8 : 10);
        switch (k) {
            case 0: case 1: case 2: return literal() + repeat();
            case 3: case 4: return cls() + repeat();
            case 5:
                if (rng.chance(50)) return ".";
                return "\"" + std::string(1, (char)('a' + rng.below(3))) + "\"";
            case 6: {
                if (rng.chance(15)) {
                    const int32_t id = rng.pick(bv.structural);
                    return std::string(rng.chance(50) ? "<[" : "!<[") + std::to_string(id) + "]>";
                }
                const int lo = first ? self + 1 : 0;
                if (lo >= n_rules) return literal();
                return "r" + std::to_string(lo + (int)rng.below((size_t)(n_rules - lo)));
            }
            case 7:
                return "r" + std::to_string(std::min(n_rules - 1, self + 1 + (int)rng.below(2)));
            default: return "(" + alts(self, depth + 1) + ")" + repeat();
        }
    }
    /* An empty top-level sequence is written `""`: GBNF skips newlines after `::=` and `|`, so an
     * empty last alternative would run into the next rule. */
    std::string seq(int self, int depth) {
        std::string s;
        const size_t n = rng.chance(10) ? 0 : 1 + rng.below(3);
        for (size_t i = 0; i < n; ++i) s += (i ? " " : "") + item(self, i == 0, depth);
        return s.empty() && depth == 0 ? "\"\"" : s;
    }
    std::string alts(int self, int depth) {
        std::string s = seq(self, depth);
        const size_t n = rng.below(3);
        for (size_t i = 0; i < n; ++i) s += " | " + seq(self, depth);
        return s;
    }
    std::string grammar() {
        std::string g = "root ::= r0 (\"!\" | \"\")\n";
        for (int r = 0; r < n_rules; ++r)
            g += "r" + std::to_string(r) + " ::= " + alts(r, 0) + "\n";
        return g;
    }
};

/* ================================================================== timing */

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string go_file() {
    std::string s = "package cache\n\nimport (\n\t\"container/list\"\n\t\"sync\"\n)\n\n";
    for (int i = 0; i < 6; ++i) {
        const std::string n = std::to_string(i);
        s += "// Shard" + n + " holds a fixed-capacity LRU keyed by string.\n"
             "type Shard" + n + " struct {\n\tmu    sync.Mutex\n\tll    *list.List\n}\n\n"
             "func (s *Shard" + n + ") Get(key string) (any, bool) {\n"
             "\ts.mu.Lock()\n\tdefer s.mu.Unlock()\n\tif e, ok := s.items[key]; ok {\n"
             "\t\treturn e.Value.(*entry).val, true\n\t}\n\treturn nil, false\n}\n\n";
    }
    return s;
}

/* Agent calls in two syntaxes: the Qwen templates' XML-ish tags and Gemma 4's object form. A
 * grammar replays the ones it admits. */
std::vector<std::string> tool_call_texts() {
    const std::string q = "<|\"|>";
    return {
        "<|tool_call>call:bash{command:" + q + "cd /repo && go test ./... 2>&1 | head -50" + q +
        ",description:" + q + "Run the test suite" + q + "}<tool_call|>",
        "<|tool_call>call:edit_file{new_string:" + q + "for s.ll.Len() >= s.cap { // caf\xc3\xa9" +
        q + ",old_string:" + q + "if s.ll.Len() >= s.cap {" + q + ",path:" + q +
        "/repo/\xe6\x97\xa5\xe6\x9c\xac/lru.go" + q + ",replace_all:false}<tool_call|>",
        "<|tool_call>call:read_file{limit:200,path:" + q + "/repo/cache/lru.go" + q +
        "}<tool_call|>",
        "<tool_call>\n<function=bash>\n<parameter=command>\n"
        "cd /repo && go test ./... 2>&1 | head -50\n</parameter>\n"
        "<parameter=description>\nRun the test suite\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=read_file>\n<parameter=path>\n/repo/cache/lru.go\n</parameter>\n"
        "<parameter=limit>\n200\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=write_file>\n<parameter=path>\n/repo/cache/shards.go\n"
        "</parameter>\n"
        "<parameter=content>\n" + go_file() + "\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=edit_file>\n<parameter=path>\n"
        "/repo/\xe6\x97\xa5\xe6\x9c\xac/lru.go\n"
        "</parameter>\n<parameter=old_string>\n\tif s.ll.Len() >= s.cap {\n</parameter>\n"
        "<parameter=new_string>\n\tfor s.ll.Len() >= s.cap && s.cap > 0 { "
        "// caf\xc3\xa9 \xf0\x9f\x98\x80\n"
        "</parameter>\n</function>\n</tool_call>",
    };
}

}  /* namespace */

/* ================================================================== the cases */

TEST(gbnf_the_synthetic_vocabulary_is_shaped_like_a_real_one) {
    auto bv = bpe_like(7, 4000);
    /* The vocabulary properties the plan is sensitive to, pinned so a generator change cannot
     * silently make the other cases easier. */
    CHECK(bv->v.n_tokens() >= 4000);
    CHECK(bv->mean_bytes > 4.0 && bv->mean_bytes < 10.0);
    CHECK(bv->longest >= 40);
    int ends_mid = 0, starts_mid = 0, eog = 0, control = 0;
    for (int32_t id = 0; id < bv->v.n_tokens(); ++id) {
        const std::string& p = bv->v.token_piece(id);
        if (bv->v.is_eog(id)) { ++eog; continue; }
        if (p.empty() || p[0] == 0) { ++control; continue; }
        if (((uint8_t)p[0] >> 6) == 2) ++starts_mid;
        const bool head_last = (uint8_t)p.back() >= 0xC0;
        const bool head_second = p.size() >= 2 && (uint8_t)p[p.size() - 2] >= 0xE0;
        if (head_last || head_second) ++ends_mid;
    }
    CHECK(ends_mid > 50);
    CHECK(starts_mid > 50);
    CHECK_EQ(eog, 2);
    CHECK_EQ(control, 3);
}

TEST(gbnf_hand_written_grammars_agree_with_the_oracle) {
    auto bv = bpe_like(11, 3000);
    Tally t;
    uint64_t seed = 1;
    for (const auto& c : kTricky) {
        const std::string g = with_tokens(c[1], *bv);
        std::unique_ptr<GbnfGrammar> probe;
        std::string err;
        const int st = GbnfGrammar::create(g, "root", &bv->v, false, {}, {}, &probe, &err);
        if (st < 0) std::fprintf(stderr, "    %s: %s\n", c[0], err.c_str());
        CHECK_OK(st);
        if (st < 0) continue;
        walk_and_compare(g, *bv, seed++, 6, 40, t, c[0]);
    }
    report("hand-written", t);
    CHECK_EQ(t.mismatches, 0);
    CHECK_EQ(t.bounded, 0);
    CHECK(t.masks > 2500);
    CHECK(t.partial > 20);
}

TEST(gbnf_json_schema_grammars_agree_with_the_oracle) {
    auto bv = bpe_like(13, 3000);
    Tally t;
    uint64_t seed = 100;
    for (const char* schema : kSchemas) {
        std::string gbnf, err;
        CHECK_OK(grammar_from_json_schema(schema, "root", &gbnf, &err));
        if (gbnf.empty()) continue;
        ++seed;
        walk_and_compare(gbnf, *bv, seed, 6, 50, t, "schema " + std::to_string(seed - 100));
    }
    report("json schema", t);
    CHECK_EQ(t.mismatches, 0);
    CHECK_EQ(t.bounded, 0);
    CHECK(t.masks > 1200);
}

/* A schema no value satisfies is represented, not refused (vendor/patches/0012). oh-my-pi's
 * `task` tool declares `"model?": "never"` in batch form, which Arktype exports as `{"not": {}}`
 * (JSON Schema's `false`); refusing it would fail every request carrying the tool. It becomes a
 * key the object never holds. */
TEST(gbnf_a_property_no_value_satisfies_is_a_key_never_written) {
    auto bv = bpe_like(23, 1500);
    /* A whole value from a fresh grammar: every byte accepted and the parse complete. */
    auto whole = [&](const std::string& gbnf, const std::string& text) {
        std::unique_ptr<GbnfGrammar> g;
        std::string err;
        if (GbnfGrammar::create(gbnf, "root", &bv->v, false, {}, {}, &g, &err) < 0) return false;
        return g->accept_str(text) >= 0 && g->complete();
    };

    for (const char* never : { "{\"not\":{}}", "false" }) {
        const std::string schema = std::string("{\"type\":\"object\",\"properties\":{"
                                               "\"context\":{\"type\":\"string\"},\"model\":") +
                                   never + "},\"required\":[\"context\"]}";
        std::string gbnf, err;
        CHECK_OK(grammar_from_json_schema(schema, "root", &gbnf, &err));
        CHECK(whole(gbnf, "{\"context\": \"x\"}"));
        CHECK(!whole(gbnf, "{\"context\": \"x\", \"model\": \"y\"}"));
        CHECK(!whole(gbnf, "{\"model\": \"y\", \"context\": \"x\"}"));
    }

    /* additionalProperties must not readmit the key. */
    {
        std::string gbnf, err;
        CHECK_OK(grammar_from_json_schema(
            R"({"type":"object","properties":{"model":{"not":{}}},"additionalProperties":true})",
            "root", &gbnf, &err));
        CHECK(whole(gbnf, "{\"other\": 1}"));
        CHECK(!whole(gbnf, "{\"model\": 1}"));
    }

    /* Such a union alternative contributes nothing. */
    {
        std::string gbnf, err;
        CHECK_OK(grammar_from_json_schema(R"({"anyOf":[{"type":"string"},{"not":{}}]})", "root",
                                          &gbnf, &err));
        CHECK(whole(gbnf, "\"a\""));
        CHECK(!whole(gbnf, "1"));
    }

    /* Where a value is required nothing can be written, so it is refused by name. */
    for (const char* schema : {
             R"({"type":"object","properties":{"model":{"not":{}}},"required":["model"]})",
             R"({"not":{}})",
             R"({"anyOf":[{"not":{}},false]})" }) {
        std::string gbnf, err;
        CHECK(grammar_from_json_schema(schema, "root", &gbnf, &err) < 0);
        CHECK(!err.empty());
    }

    /* The tool as oh-my-pi sends it, through each served template's tool-call grammar. */
    const char* tools = R"([{"type":"function","function":{"name":"task",
      "description":"Spawn subagents.","parameters":{"type":"object","properties":{
       "context":{"type":"string"},"model":{"not":{}},
       "tasks":{"type":"array","items":{"type":"object","properties":{
        "name":{"type":"string"},"agent":{"type":"string","default":"task"},
        "task":{"type":"string"},"solutionSpace":{"type":"string"},
        "model":{"anyOf":[{"type":"string"},{"type":"array","items":{"type":"string"}}]},
        "schemaMode":{"enum":["permissive","strict"]},
        "tools":{"type":"array","items":{"type":"string"}}},
        "required":["agent","solutionSpace","task"]}}},
       "required":["context","tasks"]}}}])";
    const ojson messages = ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } });
    /* "auto" (the server refuses "required"), with parallel calls on (the default) and off. A
     * template may leave the reply unconstrained and produce no grammar, but must not refuse. */
    int built = 0;
    for (const TemplateCase& tc : kTemplates) {
        ChatTemplate ct;
        REQUIRE(ct.load(slurp(fixture(tc.file)), tc.bos, tc.eos) >= 0);
        for (int par = 0; par < 2; ++par) {
            ChatOptions opt;
            opt.tool_choice = "auto";
            opt.parallel_tool_calls = par != 0;
            ChatPrompt cp;
            CHECK(ct.apply(messages, ojson::parse(tools), opt, &cp) >= 0);
            if (!cp.grammar.empty()) ++built;
        }
    }
    CHECK(built >= 3);
}

/* The `not` of a schema every value satisfies is also `never`, however the universal schema is
 * spelled; oh-my-pi 18.6.0 sends its `task` tool's optional `model` as `not` of a union of all six
 * JSON types. A `not` of anything narrower is a real complement and stays refused. */
TEST(gbnf_a_not_of_every_json_type_is_never_too) {
    auto bv = bpe_like(29, 1500);
    auto whole = [&](const std::string& gbnf, const std::string& text) {
        std::unique_ptr<GbnfGrammar> g;
        std::string err;
        if (GbnfGrammar::create(gbnf, "root", &bv->v, false, {}, {}, &g, &err) < 0) return false;
        return g->accept_str(text) >= 0 && g->complete();
    };
    auto optional_model = [](const std::string& never) {
        return std::string(R"({"type":"object","properties":{"context":{"type":"string"},"model":)") +
               never + R"(},"required":["context"]})";
    };

    for (const char* never : {
             R"({"not":{"anyOf":[{"type":"string"},{"type":"number"},{"type":"boolean"},)"
             R"({"type":"object"},{"type":"array"},{"type":"null"}]}})",
             R"({"not":{"type":["string","number","boolean","object","array","null"]}})",
             R"({"not":{"anyOf":[{"type":["string","number"]},{"type":["boolean","object"]},)"
             R"({"type":"array","description":"a list"},{"type":"null"}]}})",
             R"({"not":{"anyOf":[{"type":"string","minLength":1},{}]}})",
             R"({"not":true})",
             R"({"not":{"description":"anything"}})" }) {
        std::string gbnf, err;
        CHECK_OK(grammar_from_json_schema(optional_model(never), "root", &gbnf, &err));
        CHECK(whole(gbnf, "{\"context\": \"x\"}"));
        CHECK(!whole(gbnf, "{\"context\": \"x\", \"model\": \"y\"}"));
        CHECK(!whole(gbnf, "{\"context\": \"x\", \"model\": null}"));
    }

    for (const char* complement : {
             /* no null among the six */
             R"({"not":{"anyOf":[{"type":"string"},{"type":"number"},{"type":"boolean"},)"
             R"({"type":"object"},{"type":"array"}]}})",
             /* integer is not number */
             R"({"not":{"type":["string","integer","boolean","object","array","null"]}})",
             /* an alternative that constrains more than its type is not all of that type */
             R"({"not":{"anyOf":[{"type":"string","minLength":1},{"type":"number"},)"
             R"({"type":"boolean"},{"type":"object"},{"type":"array"},{"type":"null"}]}})",
             /* a keyword beside the type narrows it */
             R"({"not":{"type":["string","number","boolean","object","array","null"],"minimum":0}})",
             R"({"not":{"type":"string"}})" }) {
        std::string gbnf, err;
        CHECK(grammar_from_json_schema(optional_model(complement), "root", &gbnf, &err) < 0);
        CHECK(err.find("not") != std::string::npos);
    }
}

/* oh-my-pi 18.6.0's tools as an unmodified client sends them, minus descriptions
 * (tests/data/chat/oh-my-pi-18.6.0-tools.json). Every served template must render them and every
 * grammar it writes must compile, as admission requires. */
TEST(gbnf_oh_my_pi_tools_build_and_compile_under_every_template) {
    auto bv = bpe_like(31, 3000);
    const ojson tools = ojson::parse(slurp(fixture("../chat/oh-my-pi-18.6.0-tools.json")));
    REQUIRE(tools.is_array() && tools.size() == 11);
    const ojson messages = ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } });
    int built = 0;
    for (const TemplateCase& tc : kTemplates) {
        ChatTemplate ct;
        REQUIRE(ct.load(slurp(fixture(tc.file)), tc.bos, tc.eos) >= 0);
        for (int par = 0; par < 2; ++par) {
            ChatOptions opt;
            opt.tool_choice = "auto";
            opt.parallel_tool_calls = par != 0;
            opt.template_kwargs["preserve_thinking"] = "true";
            ChatPrompt cp;
            CHECK(ct.apply(messages, tools, opt, &cp) >= 0);
            if (cp.grammar.empty()) continue;
            ++built;
            std::unique_ptr<GbnfGrammar> g;
            std::string err;
            CHECK(GbnfGrammar::create(cp.grammar, "root", &bv->v, false, {}, {}, &g, &err) >= 0);
            if (!err.empty()) fprintf(stderr, "    %s: %s\n", tc.file, err.c_str());
        }
    }
    CHECK(built >= 2);
}

/* The tool-call grammars of the served templates and of Gemma 4's declared format: walked, and
 * replayed over the agent calls whose syntax matches. One case per format; `min_replays` is how
 * many of tool_call_texts() the format's grammars admit. */
void tool_call_case(const std::string& prefix, int min_grammars, int min_replays,
                    int64_t min_masks) {
    static auto bv = bpe_like(17, 3000);
    Tally t;
    int replays = 0, grammars = 0;
    uint64_t next_seed = 200;
    for (const auto& ng : template_grammars()) {
        const uint64_t seed = next_seed++;
        if (ng.first.compare(0, prefix.size(), prefix) != 0) continue;
        ++grammars;
        walk_and_compare(ng.second, *bv, seed, 4, 40, t, ng.first);
        for (const std::string& text : tool_call_texts()) {
            std::unique_ptr<GbnfGrammar> probe;
            std::string err;
            REQUIRE(GbnfGrammar::create(ng.second, "root", &bv->v, false, {}, {}, &probe,
                                        &err) >= 0);
            if (probe->accept_str(text) < 0) continue;
            const std::vector<int32_t> toks = bv->tokenize(text);
            REQUIRE(!toks.empty());
            replay_and_compare(ng.second, *bv, toks, t, ng.first + " replay");
            ++replays;
        }
    }
    report(prefix.c_str(), t);
    CHECK_EQ(t.mismatches, 0);
    CHECK_EQ(t.bounded, 0);
    CHECK(grammars >= min_grammars);
    CHECK(replays >= min_replays);
    CHECK(t.masks > min_masks);
}

TEST(gbnf_qwen3_8_tool_call_grammars_agree_with_the_oracle) { tool_call_case("qwen3.8", 1, 0, 0); }
TEST(gbnf_qwen3_6_tool_call_grammars_agree_with_the_oracle) { tool_call_case("qwen3.6", 1, 0, 0); }
TEST(gbnf_minicpm5_tool_call_grammars_agree_with_the_oracle) { tool_call_case("minicpm5", 1, 0, 0); }
TEST(gbnf_gemma4_tool_call_grammars_agree_with_the_oracle) { tool_call_case("gemma4", 1, 0, 0); }

/* Random grammars, seeds [first, last]. Most compile; refusals name their reason (mostly left
 * recursion). The 250 seeds are split in fifty, which compile 30 to 39 grammars and compare 1200
 * to 2300 masks each; the floors sit below the smallest share, and the engine may refuse at most
 * 2% of what the oracle could answer (1% over all 250). */
void random_case(uint64_t first, uint64_t last) {
    static auto bv = bpe_like(19, 1500);
    Tally t;
    int compiled = 0, refused = 0;
    std::map<std::string, int> why;
    for (uint64_t seed = first; seed <= last; ++seed) {
        Rng rng(seed);
        GrammarGen gen{ rng, 2 + (int)rng.below(5), *bv };
        const std::string g = gen.grammar();
        std::unique_ptr<GbnfGrammar> probe;
        std::string err;
        if (GbnfGrammar::create(g, "root", &bv->v, false, {}, {}, &probe, &err) < 0) {
            ++refused;
            CHECK(!err.empty());
            ++why[err.substr(0, err.find_first_of(";:"))];
            continue;
        }
        ++compiled;
        walk_and_compare(g, *bv, seed, 2, 24, t, "random grammar " + std::to_string(seed));
        if (t.mismatches) {
            std::fprintf(stderr, "    the grammar:\n%s", g.c_str());
            break;
        }
    }
    report("random", t);
    CHECK(t.bounded * 50 <= t.masks);
    for (const auto& kv : why)
        std::fprintf(stderr, "      refused %d: %s\n", kv.second, kv.first.c_str());
    std::fprintf(stderr, "    %d of %d random grammars compiled\n", compiled, compiled + refused);
    CHECK_EQ(t.mismatches, 0);
    const uint64_t n = last - first + 1;
    CHECK((uint64_t)compiled * 50 > 25 * n);
    CHECK((uint64_t)t.masks * 50 > 1000 * n);
}

TEST(gbnf_random_grammars_1_to_50_agree_with_the_oracle)    { random_case(1, 50); }
TEST(gbnf_random_grammars_51_to_100_agree_with_the_oracle)  { random_case(51, 100); }
TEST(gbnf_random_grammars_101_to_150_agree_with_the_oracle) { random_case(101, 150); }
TEST(gbnf_random_grammars_151_to_200_agree_with_the_oracle) { random_case(151, 200); }
TEST(gbnf_random_grammars_201_to_250_agree_with_the_oracle) { random_case(201, 250); }

/* A state carrying a split character is served by its own index, not the plan: only pieces
 * starting with the bytes that finish the character are admissible. Reached deliberately (accept a
 * character's head, then mask) over grammars whose classes cut through the ranges the carried bits
 * could complete to. */
TEST(gbnf_masks_after_a_split_character_agree_with_the_oracle) {
    auto bv = bpe_like(23, 3000);
    const char* grammars[] = {
        "root ::= [\\u0080-\\U0010FFFF]+\n",
        "root ::= ([\\u4E00-\\u9FFF] | [\\u00C0-\\u00FF] \"s\"?)+ \"!\"\n",
        "root ::= (\"\xe6\x97\xa5\" | \"\xe6\x9c\xac\" [^\\u4E00-\\u4E2F])* [a-z]\n",
        "root ::= [^\\u00E0-\\u00FF]* [\\U0001F600-\\U0001F602] \"x\"\n",
        "root ::= \"\\\"\" [^\"\\\\\\x00-\\x1F]* \"\\\"\"\n",
    };
    const int64_t n_words = bitmask_words(bv->v.n_tokens());
    std::vector<uint32_t> a((size_t)n_words), b((size_t)n_words);
    Tally t;
    for (const char* gt : grammars) {
        OracleMemo memo;
        for (const char* m : kMultibyte) {
            const std::string s(m);
            for (size_t k = 1; k < s.size(); ++k) {
                std::unique_ptr<GbnfGrammar> g;
                std::string err;
                REQUIRE(GbnfGrammar::create(gt, "root", &bv->v, false, {}, {}, &g, &err) >= 0);
                /* The character's head, then its middle a byte at a time. */
                if (g->accept_token(bv->by_piece.at(s.substr(0, k))) < 0) continue;
                compare(*g, bv->v, a, b, t,
                        std::string(gt) + " after " + show_piece(s.substr(0, k)), &memo);
                if (k + 1 < s.size() && g->accept_token(bv->by_piece.at(s.substr(k, 1))) >= 0)
                    compare(*g, bv->v, a, b, t, std::string(gt) + " a byte later", &memo);
            }
        }
        walk_and_compare(gt, *bv, 300, 6, 40, t, gt);
    }
    report("split characters", t);
    CHECK_EQ(t.mismatches, 0);
    CHECK_EQ(t.bounded, 0);
    CHECK(t.partial > 60);
}

/* A character finished as U+0000 ends the piece, as upstream reads it: decoded sequences are
 * NUL-terminated, so a piece completing an overlong head to 0 (or decoding to 0) has no characters
 * whatever follows. The engine serves the mask after such a head by its own path, checked against
 * the oracle state by state. */
TEST(gbnf_a_character_finished_as_nul_agrees_with_the_oracle) {
    auto bv = bpe_like(59, 1500);
    const char* grammars[] = {
        "root ::= [^a]* \"!\"\n",
        "root ::= [\\x00-\\x7F]+\n",
        "root ::= (\"x\" | [\\u0080-\\U0010FFFF])+ \"z\"?\n",
        "root ::= \"q\" | !<[5]> \"y\"\n",
    };
    const int64_t n_words = bitmask_words(bv->v.n_tokens());
    std::vector<uint32_t> a((size_t)n_words), b((size_t)n_words);
    Tally t;
    for (const char* gt : grammars) {
        for (const char* head : { "\xc0", "\xe0", "\xe0\x80", "\xf0", "\xf0\x80\x80" }) {
            std::unique_ptr<GbnfGrammar> g;
            std::string err;
            if (GbnfGrammar::create(gt, "root", &bv->v, false, {}, {}, &g, &err) < 0) continue;
            if (g->accept_token(bv->by_piece.at(head)) < 0) continue;
            compare(*g, bv->v, a, b, t, std::string(gt) + " after " + show_piece(head));
        }
        walk_and_compare(gt, *bv, 700, 6, 30, t, gt);
    }
    report("characters finished as NUL", t);
    CHECK_EQ(t.mismatches, 0);
    CHECK(t.partial > 10);
}

/* A token terminal matches an id, not text, and upstream applies it wherever reached, mid-piece
 * included, admitting that id (or every other token for `!<[id]>`). The engine reproduces this
 * exactly. */
TEST(gbnf_token_terminals_agree_with_the_oracle) {
    auto bv = bpe_like(29, 2000);
    const int32_t ab = bv->by_piece.at("ab"), a = bv->by_piece.at("a"), x = bv->by_piece.at("x");
    const std::string grammars[] = {
        "root ::= <[" + std::to_string(ab) + "]> \"c\" | \"d\"\n",
        "root ::= !<[" + std::to_string(x) + "]> \"y\"*\n",
        "root ::= \"a\" <[" + std::to_string(ab) + "]> | \"a\" !<[" + std::to_string(a) + "]>\n",
        "root ::= (\"x\" !<[" + std::to_string(ab) + "]>)+ \"z\"?\n",
        "root ::= \"<think>\" [^<]* <[" + std::to_string(bv->by_piece.at("</think>")) + "]>\n",
    };
    const int64_t n_words = bitmask_words(bv->v.n_tokens());
    std::vector<uint32_t> ma((size_t)n_words), mb((size_t)n_words);
    Tally t;
    uint64_t seed = 400;
    for (const std::string& g : grammars) {
        std::unique_ptr<GbnfGrammar> gg;
        std::string err;
        REQUIRE(GbnfGrammar::create(g, "root", &bv->v, false, {}, {}, &gg, &err) >= 0);
        compare(*gg, bv->v, ma, mb, t, g);
        walk_and_compare(g, *bv, seed++, 6, 30, t, g);
    }
    report("token terminals", t);
    CHECK_EQ(t.mismatches, 0);
    CHECK_EQ(t.bounded, 0);
}

/* The statuses are part of the contract: no parse left, only the end admissible, and a completed
 * grammar. */
TEST(gbnf_dead_and_complete_states_agree_with_the_oracle) {
    auto bv = bpe_like(31, 1500);
    const int64_t n_words = bitmask_words(bv->v.n_tokens());
    std::vector<uint32_t> a((size_t)n_words), b((size_t)n_words);
    Tally t;
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    REQUIRE(GbnfGrammar::create("root ::= \"ab\" | \"c\"\n", "root", &bv->v, false, {}, {}, &g,
                                &err) >= 0);
    CHECK_OK(compare(*g, bv->v, a, b, t, "start"));
    CHECK_OK(g->accept_token(bv->by_piece.at("ab")));
    CHECK(g->complete());
    CHECK_OK(compare(*g, bv->v, a, b, t, "complete"));       /* only the end is admissible */
    CHECK(g->accept_token(bv->by_piece.at("x")) < 0);         /* no parse left */
    CHECK_EQ(compare(*g, bv->v, a, b, t, "dead"), RAD_E_STATE);
    CHECK_EQ(t.mismatches, 0);
}

/* A repetition longer than any token is a star to the plan. `x{0,n}` is n rules and `x{n}` n
 * copies, each a plan key; past the longest token they all answer alike and share one key. So the
 * plan size is bounded by the longest token whatever n is, and masks deep into the repetition stay
 * exact. */
TEST(gbnf_a_repetition_longer_than_any_token_costs_what_the_token_does) {
    auto bv = bpe_like(37, 1500);
    std::vector<GbnfProgramInfo> info;
    for (int n : { 200, 400, 1600 }) {
        std::shared_ptr<const GbnfProgram> p;
        std::string err;
        const std::string g = "root ::= \"\\\"\" [^\"]{0," + std::to_string(n) +
                              "} \"\\\"\" [a-z]{" +
                              std::to_string(n) + "} \";\"\n";
        REQUIRE(gbnf_compile(g, "root", &bv->v, &p, &err) >= 0);
        info.push_back(gbnf_program_info(*p));
    }
    CHECK_EQ(info[0].keys, info[1].keys);
    CHECK_EQ(info[1].keys, info[2].keys);
    CHECK_EQ(info[0].dense_masks, info[2].dense_masks);

    Tally t;
    walk_and_compare("root ::= \"\\\"\" [^\"]{0,400} \"\\\"\" [a-z]{400} \";\"\n", *bv, 500, 4, 120,
                     t, "long repetitions");
    report("long repetitions", t);
    CHECK_EQ(t.mismatches, 0);
}

/* A length bound past 2000 is enforced exactly (upstream refuses repetition counts past 2000, so
 * `maxLength: 8000` would fail). Expansion is limited by compile cost, not the count: the bound is
 * exact to the character, the plan matches one for a count under 2000, and only a count beyond the
 * compile budget is refused, naming its rule. */
TEST(gbnf_a_length_bound_past_two_thousand_is_held_exactly) {
    auto bv = bpe_like(59, 1500);
    auto whole = [&](const std::string& gbnf, const std::string& text) {
        std::unique_ptr<GbnfGrammar> g;
        std::string err;
        if (GbnfGrammar::create(gbnf, "root", &bv->v, false, {}, {}, &g, &err) < 0) return false;
        return g->accept_str(text) >= 0 && g->complete();
    };
    auto from_schema = [](const std::string& schema) {
        std::string gbnf, err;
        CHECK_OK(grammar_from_json_schema(schema, "root", &gbnf, &err));
        return gbnf;
    };
    auto quoted = [](size_t n) { return "\"" + std::string(n, 'a') + "\""; };

    for (int n : { 2001, 8000, 65536 }) {
        const std::string g =
            from_schema("{\"type\":\"string\",\"maxLength\":" + std::to_string(n) + "}");
        CHECK(whole(g, quoted(0)));
        CHECK(whole(g, quoted((size_t)n)));
        CHECK(!whole(g, quoted((size_t)n + 1)));
    }
    {
        const std::string g = from_schema(R"({"type":"string","minLength":3000,"maxLength":8000})");
        CHECK(!whole(g, quoted(2999)));
        CHECK(whole(g, quoted(3000)));
        CHECK(whole(g, quoted(8000)));
        CHECK(!whole(g, quoted(8001)));
    }
    {
        const std::string g =
            from_schema(R"({"type":"array","items":{"type":"integer"},"maxItems":5000})");
        auto ones = [](int k) {
            std::string s = "[";
            for (int i = 0; i < k; ++i) s += i ? ",1" : "1";
            return s + "]";
        };
        CHECK(whole(g, ones(5000)));
        CHECK(!whole(g, ones(5001)));
    }

    /* The plan does not grow with the bound: past the longest token no token distinguishes the
     * counts. */
    std::vector<GbnfProgramInfo> info;
    for (int n : { 1999, 8000, 65536 }) {
        std::shared_ptr<const GbnfProgram> p;
        std::string err;
        REQUIRE(gbnf_compile(from_schema("{\"type\":\"string\",\"maxLength\":" +
                                         std::to_string(n) + "}"),
                             "root", &bv->v, &p, &err) >= 0);
        info.push_back(gbnf_program_info(*p));
    }
    CHECK_EQ(info[0].keys, info[1].keys);
    CHECK_EQ(info[1].keys, info[2].keys);
    CHECK_EQ(info[0].dense_masks, info[2].dense_masks);

    /* A tool with four 8000-character fields compiles through every served template's tool-call
     * grammar, so its request is served. */
    const char* tools = R"([{"type":"function","function":{"name":"clarify",
      "description":"Ask the user to choose.","parameters":{"type":"object","properties":{
       "question":{"type":"string","maxLength":8000},
       "choices":{"type":"array","maxItems":4,"items":{"type":"string","maxLength":8000}}},
       "required":["question","choices"]}}}])";
    const ojson messages = ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } });
    int compiled = 0, carried = 0;
    for (const TemplateCase& tc : kTemplates) {
        ChatTemplate ct;
        REQUIRE(ct.load(slurp(fixture(tc.file)), tc.bos, tc.eos) >= 0);
        for (int par = 0; par < 2; ++par) {
            for (const char* choice : { "auto", "required" }) {
                ChatOptions opt;
                opt.tool_choice = choice;
                opt.parallel_tool_calls = par != 0;
                ChatPrompt cp;
                CHECK(ct.apply(messages, ojson::parse(tools), opt, &cp) >= 0);
                if (cp.grammar.empty()) continue;
                std::shared_ptr<const GbnfProgram> p;
                std::string err;
                const auto t0 = std::chrono::steady_clock::now();
                CHECK(gbnf_compile(cp.grammar, "root", &bv->v, &p, &err) >= 0);
                if (!err.empty()) std::fprintf(stderr, "    %s: %s\n", tc.file, err.c_str());
                const double ms = ms_since(t0);
                CHECK(ms < 2000.0);
                ++compiled;
                if (cp.grammar.find("8000}") != std::string::npos) {
                    ++carried;
                    std::fprintf(stderr, "    %s, %s, parallel %s: the bound is in the grammar\n",
                                 tc.file, choice, par ? "on" : "off");
                }
            }
        }
    }
    CHECK(compiled >= 6);
    CHECK(carried >= 1);

    /* A group under a repetition references the group's rule rather than copying it, so its cost
     * is the counts added, not multiplied (upstream counts it multiplied). */
    {
        std::shared_ptr<const GbnfProgram> p;
        std::string err;
        CHECK(gbnf_compile("root ::= ([a-z]{3000}){3000} \";\"\n", "root", &bv->v, &p, &err) >= 0);
    }

    /* A schema bound past INT_MAX (Number.MAX_SAFE_INTEGER is a common "no limit") is treated as
     * unbounded, which no served context can distinguish. A negative or non-numeric bound is
     * refused by the converter, by name. */
    for (const char* big : { "9007199254740991", "1e300", "2147483648" }) {
        const std::string g =
            from_schema(std::string("{\"type\":\"string\",\"maxLength\":") + big + "}");
        CHECK(whole(g, quoted(3000)));
        const std::string a = from_schema(std::string("{\"type\":\"array\",\"maxItems\":") + big +
                                          ",\"items\":{\"type\":\"integer\"}}");
        CHECK(whole(a, "[1,2,3]"));
    }
    for (const char* bad : { R"({"type":"string","maxLength":-5})",
                             R"({"type":"string","minLength":"3"})",
                             R"({"type":"array","items":{},"maxItems":-1})" }) {
        std::string gbnf, err;
        CHECK(grammar_from_json_schema(bad, "root", &gbnf, &err) < 0);
        CHECK(err.find("must be a non-negative number") != std::string::npos);
    }

    /* A bound beyond the compile budget is refused naming the property's rule and the count, so
     * a client can find it in its tool list. */
    {
        std::shared_ptr<const GbnfProgram> p;
        std::string err;
        CHECK(gbnf_compile(from_schema(R"({"type":"object","properties":{"content":)"
                                       R"({"type":"string","maxLength":1000000}}})"),
                           "root", &bv->v, &p, &err) < 0);
        CHECK(err.find("rule 'content'") != std::string::npos);
        CHECK(err.find("{0,1000000}") != std::string::npos);
    }

    /* What exceeds the budget is refused by rule, and so is a count past 2^64 - 2; none wraps into
     * a small expansion. A repetition of a repetition copies the expanded run, so `{3000}{3000}`
     * is nine million elements. */
    for (const char* g : { "root ::= \"a\" item\nitem ::= [a-z]{0,100000000}\n",
                           "root ::= \"a\" item\nitem ::= [a-z]{0,18446744073709551614}\n",
                           "root ::= \"a\" item\nitem ::= [a-z]{18446744073709551614}\n",
                           "root ::= \"a\" item\nitem ::= [a-z]{0,99999999999999999999999}\n",
                           "root ::= \"a\" item\nitem ::= [a-z]{3000}{3000}\n" }) {
        std::shared_ptr<const GbnfProgram> p;
        std::string err;
        CHECK(gbnf_compile(g, "root", &bv->v, &p, &err) < 0);
        CHECK(err.find("rule 'item'") != std::string::npos);
    }
}

/* Bounded repetitions of one operand share one chain of links, so a schema's many length bounds
 * cost its longest once rather than their sum. 200 properties at `maxLength: 4000` were 200 chains,
 * past the compile budget (a tool schema like that is refused in full); shared, they compile, and
 * every bound, short or long, string or array, is still held exactly. */
TEST(gbnf_bounds_over_one_operand_share_their_links) {
    auto bv = bpe_like(59, 1500);
    std::string props = R"("a":{"type":"string","maxLength":4000},"b":{"type":"string","maxLength":7},)"
                        R"("c":{"type":"array","items":{"type":"string","maxLength":5},"maxItems":3})";
    for (int i = 0; i < 200; ++i)
        props += ",\"f" + std::to_string(i) + "\":{\"type\":\"string\",\"maxLength\":4000}";
    std::string gbnf, err;
    CHECK_OK(grammar_from_json_schema("{\"type\":\"object\",\"properties\":{" + props +
                                          "},\"required\":[\"a\",\"b\",\"c\"]}",
                                      "root", &gbnf, &err));
    std::unique_ptr<GbnfGrammar> g;
    CHECK_OK(GbnfGrammar::create(gbnf, "root", &bv->v, false, {}, {}, &g, &err));
    auto whole = [&](const std::string& text) {
        std::unique_ptr<GbnfGrammar> h;
        std::string e;
        if (GbnfGrammar::create(gbnf, "root", &bv->v, false, {}, {}, &h, &e) < 0) return false;
        return h->accept_str(text) >= 0 && h->complete();
    };
    auto doc = [](size_t a, size_t b, std::vector<size_t> c, size_t f7) {
        std::string s = "{\"a\":\"" + std::string(a, 'x') + "\",\"b\":\"" + std::string(b, 'y') +
                        "\",\"c\":[";
        for (size_t i = 0; i < c.size(); ++i) s += (i ? ",\"" : "\"") + std::string(c[i], 'z') + "\"";
        s += "]";
        if (f7) s += ",\"f7\":\"" + std::string(f7, 'w') + "\"";
        return s + "}";
    };
    CHECK(whole(doc(4000, 7, { 5, 0, 5 }, 4000)));
    CHECK(whole(doc(0, 0, {}, 0)));
    CHECK(!whole(doc(4001, 7, { 5 }, 0)));
    CHECK(!whole(doc(10, 8, { 5 }, 0)));
    CHECK(!whole(doc(10, 7, { 6 }, 0)));
    CHECK(!whole(doc(10, 7, { 1, 1, 1, 1 }, 0)));
    CHECK(!whole(doc(10, 7, { 1 }, 4001)));
}

/* Identical clauses across tools are one clause to the plan. A tool-calling template writes the
 * same undeclared-parameter clause, free-text value and closing tags as separate rules per tool;
 * the plan hash-conses rules by body, so its size and the compile's vocabulary walks do not grow
 * with the number of tools. */
TEST(gbnf_rule_copies_share_one_plan) {
    auto bv = bpe_like(41, 1500);
    ChatTemplate ct;
    REQUIRE(ct.load(slurp(fixture("qwen3.8.jinja")), "", "<|im_end|>") >= 0);
    std::vector<size_t> dense;
    for (int n_tools : { 3, 30 }) {
        ojson tools = ojson::array();
        for (int i = 0; i < n_tools; ++i)
            tools.push_back({ { "type", "function" }, { "function", {
                { "name", "tool_" + std::to_string(i) }, { "description", "d" },
                { "parameters", { { "type", "object" }, { "properties", {
                    { "path", { { "type", "string" } } }, { "n", { { "type", "integer" } } } } },
                    { "required", { "path" } } } } } } });
        ChatPrompt cp;
        REQUIRE(ct.apply(ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } }), tools,
                         ChatOptions(), &cp) >= 0);
        std::shared_ptr<const GbnfProgram> p;
        std::string err;
        REQUIRE(gbnf_compile(cp.grammar, "root", &bv->v, &p, &err) >= 0);
        dense.push_back(gbnf_program_info(*p).dense_masks);
    }
    CHECK_EQ(dense[0], dense[1]);
}

/* A grammar whose mask needs unbounded context is refused at compile time, by name. Here each
 * recursion level leaves an optional space below it, so the spaces a token may start with depend
 * on parse depth and no bounded stack suffix decides the mask. The engine names the rule it cannot
 * precompute rather than serving it slowly. */
TEST(gbnf_a_grammar_whose_masks_need_unbounded_context_is_refused) {
    auto bv = bpe_like(43, 24000);
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    const int st = GbnfGrammar::create("root ::= [a-z] root s | \"\"\ns ::= [^a-z]?\n", "root",
                                       &bv->v, false, {}, {}, &g, &err);
    CHECK_EQ(st, RAD_E_UNSUPPORTED);
    CHECK(err.find("'root'") != std::string::npos || err.find("'s'") != std::string::npos);
    std::fprintf(stderr, "    refused: %s\n", err.c_str());
}

/* A rule that can never match is refused by name. `r ::= r` passes the left-recursion check (the
 * reference is in tail position) but only leads back into itself, and upstream reads the resulting
 * empty stack set as admitting every token. A grammar that merely never ends still compiles. */
TEST(gbnf_a_rule_that_can_never_match_is_refused) {
    auto bv = bpe_like(61, 800);
    std::unique_ptr<GbnfGrammar> g;
    std::string err;
    CHECK_EQ(GbnfGrammar::create("root ::= \"a\" r | \"b\"\nr ::= r\n", "root", &bv->v, false,
                                 {}, {}, &g, &err), RAD_E_UNSUPPORTED);
    CHECK(err.find("'r'") != std::string::npos);
    err.clear();
    CHECK_EQ(GbnfGrammar::create("root ::= \"a\" s\ns ::= t\nt ::= (\"\")? s\n", "root", &bv->v,
                                 false, {}, {}, &g, &err), RAD_E_UNSUPPORTED);
    CHECK(!err.empty());
    CHECK_OK(GbnfGrammar::create("root ::= \"a\" root\n", "root", &bv->v, false, {}, {}, &g, &err));
    CHECK_OK(GbnfGrammar::create("root ::= [a-z] root | \"\"\n", "root", &bv->v, false, {}, {},
                                 &g, &err));
}

/* The scheduler thread never compiles an admitted grammar. Admission compiles it and the request
 * holds the program; starting the request finds it by text however many grammars were compiled
 * since, because a program stays findable while anything holds it, not only while cached. */
TEST(gbnf_a_held_program_is_found_whatever_was_compiled_since) {
    auto bv = bpe_like(47, 800);
    std::shared_ptr<const GbnfProgram> held;
    std::string err;
    REQUIRE(gbnf_compile("root ::= \"held\" [a-z]*\n", "root", &bv->v, &held, &err) >= 0);
    for (int i = 0; i < 80; ++i) {
        std::shared_ptr<const GbnfProgram> p;
        REQUIRE(gbnf_compile("root ::= \"other" + std::to_string(i) + "\"\n", "root", &bv->v, &p,
                             &err) >= 0);
    }
    std::unique_ptr<GbnfGrammar> g;
    REQUIRE(GbnfGrammar::create("root ::= \"held\" [a-z]*\n", "root", &bv->v, false, {}, {}, &g,
                                &err) >= 0);
    CHECK(&g->program() == held.get());
}

/* A mask costs a small fraction of one pass over the vocabulary, counted in work: a realistic
 * 48,000-piece vocabulary, the Qwen template's tool-call grammar, and four agent calls replayed as
 * the sampler drives them, on a freshly compiled program so no mask comes from a warm cache. The
 * pass that sets the scale only decodes each piece once, the least any per-mask vocabulary walk
 * does, so a regression to one fails here whatever the machine's load. */
TEST(gbnf_a_mask_does_a_fraction_of_a_vocabulary_pass) {
    auto bv = bpe_like(53, 48000);
    ChatTemplate ct;
    REQUIRE(ct.load(slurp(fixture("qwen3.8.jinja")), "", "<|im_end|>") >= 0);
    ChatPrompt cp;
    REQUIRE(ct.apply(ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } }),
                     ojson::parse(kTools), ChatOptions(), &cp) >= 0);
    REQUIRE(!cp.grammar.empty());
    GbnfVocabInfo vi;
    CHECK_OK(gbnf_prepare_vocab(&bv->v, &vi));
    const int64_t n_words = bitmask_words(bv->v.n_tokens());
    std::vector<uint32_t> m((size_t)n_words);
    radtest::Work& w = radtest::work();

    size_t sink = 0;
    const uint64_t pass = w.of([&] {
        for (int32_t id = 0; id < bv->v.n_tokens(); ++id)
            sink += gbnf::decode_utf8(bv->v.token_piece(id), PartialUtf8{}).first.size();
    });
    CHECK(sink > 0);

    const std::string text = cp.grammar + "\n# a program of its own\n";
    std::vector<uint64_t> masks;
    for (const std::string& call : tool_call_texts()) {
        const std::vector<int32_t> toks = bv->tokenize(call);
        std::unique_ptr<GbnfGrammar> g;
        std::string err;
        REQUIRE(GbnfGrammar::create(text, "root", &bv->v, false, {}, {}, &g, &err) >= 0);
        if (g->accept_str(call) < 0) continue;
        REQUIRE(GbnfGrammar::create(text, "root", &bv->v, false, {}, {}, &g, &err) >= 0);
        size_t i = 0;
        while (i < toks.size()) {
            const int n_pos = (int)std::min<size_t>(4, toks.size() - i);
            g->push_state();
            for (int k = 0; k < n_pos; ++k) {
                if (k > 0) REQUIRE(g->accept_token(toks[i + (size_t)k - 1]) >= 0);
                int st = 0;
                masks.push_back(w.of([&] { st = g->fill_mask(m.data(), n_words); }));
                REQUIRE(st >= 0);
            }
            g->rollback();
            const size_t n_acc = std::min<size_t>(3, toks.size() - i);
            for (size_t k = 0; k < n_acc; ++k) REQUIRE(g->accept_token(toks[i + k]) >= 0);
            i += n_acc;
        }
    }
    REQUIRE(masks.size() > 200);
    std::sort(masks.begin(), masks.end());
    const uint64_t p50 = masks[masks.size() / 2], p99 = masks[masks.size() * 99 / 100];
    const uint64_t worst = masks.back();
    std::fprintf(stderr, "    %d-token vocabulary, one decoding pass %llu %s; %zu masks: p50 %llu, "
                 "p99 %llu, max %llu\n", bv->v.n_tokens(), (unsigned long long)pass, w.unit(),
                 masks.size(), (unsigned long long)p50, (unsigned long long)p99,
                 (unsigned long long)worst);
    CHECK(p99 * 100 < pass);
    CHECK(worst * 20 < pass);
}

/* Mask latency budgets in wall time, for the serial performance tier: the same replay, best of
 * three runs per mask. The budgets sit an order of magnitude above the engine's cost and an order
 * below one vocabulary walk. */
TEST(perf_gbnf_mask_latency_stays_within_budget) {
    auto bv = bpe_like(53, 48000);
    ChatTemplate ct;
    REQUIRE(ct.load(slurp(fixture("qwen3.8.jinja")), "", "<|im_end|>") >= 0);
    ChatPrompt cp;
    REQUIRE(ct.apply(ojson::array({ ojson{ { "role", "user" }, { "content", "hi" } } }),
                     ojson::parse(kTools), ChatOptions(), &cp) >= 0);
    REQUIRE(!cp.grammar.empty());
    std::vector<std::vector<int32_t>> calls;
    for (const std::string& text : tool_call_texts()) {
        std::unique_ptr<GbnfGrammar> probe;
        std::string err;
        REQUIRE(GbnfGrammar::create(cp.grammar, "root", &bv->v, false, {}, {}, &probe, &err) >= 0);
        if (probe->accept_str(text) >= 0) calls.push_back(bv->tokenize(text));
    }
    REQUIRE(calls.size() >= 3);

    GbnfVocabInfo vi;
    CHECK_OK(gbnf_prepare_vocab(&bv->v, &vi));
    const int64_t n_words = bitmask_words(bv->v.n_tokens());
    std::vector<uint32_t> m((size_t)n_words);

    std::vector<double> best_mask, best_step;
    double best_compile = 1e30;
    for (int run = 0; run < 3; ++run) {
        /* A trailing comment makes a new program, compiled from scratch with every mask computed. */
        const std::string text = cp.grammar + "\n# run " + std::to_string(run) + "\n";
        std::shared_ptr<const GbnfProgram> prog;
        std::string err;
        const auto tc = std::chrono::steady_clock::now();
        REQUIRE(gbnf_compile(text, "root", &bv->v, &prog, &err) >= 0);
        best_compile = std::min(best_compile, ms_since(tc));

        std::vector<double> masks, steps;
        for (const auto& toks : calls) {
            std::unique_ptr<GbnfGrammar> g;
            REQUIRE(GbnfGrammar::create(text, "root", &bv->v, false, {}, {}, &g, &err) >= 0);
            size_t i = 0;
            while (i < toks.size()) {
                const auto ts = std::chrono::steady_clock::now();
                const int n_pos = (int)std::min<size_t>(4, toks.size() - i);
                g->push_state();
                for (int k = 0; k < n_pos; ++k) {
                    if (k > 0) REQUIRE(g->accept_token(toks[i + (size_t)k - 1]) >= 0);
                    const auto tm = std::chrono::steady_clock::now();
                    REQUIRE(g->fill_mask(m.data(), n_words) >= 0);
                    masks.push_back(ms_since(tm));
                }
                g->rollback();
                const size_t n_acc = std::min<size_t>(3, toks.size() - i);
                for (size_t k = 0; k < n_acc; ++k) REQUIRE(g->accept_token(toks[i + k]) >= 0);
                steps.push_back(ms_since(ts));
                i += n_acc;
            }
        }
        if (best_mask.empty()) { best_mask = masks; best_step = steps; }
        for (size_t k = 0; k < masks.size() && k < best_mask.size(); ++k)
            best_mask[k] = std::min(best_mask[k], masks[k]);
        for (size_t k = 0; k < steps.size() && k < best_step.size(); ++k)
            best_step[k] = std::min(best_step[k], steps[k]);
    }

    std::vector<double> sm = best_mask;
    std::sort(sm.begin(), sm.end());
    const double p50 = sm[sm.size() / 2], p99 = sm[sm.size() * 99 / 100], worst = sm.back();
    double step_sum = 0.0;
    for (double x : best_step) step_sum += x;
    const double step_mean = step_sum / (double)best_step.size();
    std::fprintf(stderr, "    %d-token vocabulary: index %.0f ms; compile %.1f ms; %zu masks on a "
                 "fresh program: p50 %.3f p99 %.3f max %.3f ms; %zu steps, %.3f ms mean\n",
                 bv->v.n_tokens(), vi.build_ms, best_compile, sm.size(), p50, p99, worst,
                 best_step.size(), step_mean);
    CHECK(best_compile < 2000.0);
    CHECK(p99 < 0.25);
    CHECK(worst < 1.0);
    CHECK(step_mean < 0.3);
}

/* ================================================================== JSON Schema semantics */

namespace {

/* Whether `text` is a whole value of the grammar compiled from `schema`. */
bool schema_admits(const VocabView& v, const std::string& schema, const std::string& text) {
    std::string gbnf, err;
    if (grammar_from_json_schema(schema, "root", &gbnf, &err) < 0) return false;
    std::unique_ptr<GbnfGrammar> g;
    if (GbnfGrammar::create(gbnf, "root", &v, false, {}, {}, &g, &err) < 0) return false;
    return g->accept_str(text) >= 0 && g->complete();
}

}  /* namespace */

/* The grammar a JSON Schema compiles to enforces the schema: per keyword, instances that satisfy it
 * are whole values and instances that break it are not. The mask engine is held to upstream's walk
 * elsewhere; this holds the converter to the schema, which is the client's contract (a dropped
 * range or length bound would pass every mask comparison). Instances are written compactly, in
 * the property order the converter emits (declared, required first); a value may be followed by
 * whitespace, which JSON allows. */
TEST(json_schema_grammars_admit_exactly_their_instances) {
    auto bv = bpe_like(61, 800);
    struct Case { const char* schema; std::vector<const char*> yes, no; };
    const Case cases[] = {
        { R"({"enum":["a",1,null]})", { R"("a")", "1", "null", R"("a" )" }, { R"("b")", "2", "true" } },
        { R"({"const":"abc"})", { R"("abc")" }, { R"("abd")", R"("ab")", "1" } },
        { R"({"type":["integer","null"]})", { "12", "-3", "null", "0" }, { "1.5", R"("x")", "true" } },
        { R"({"type":"integer","minimum":-5,"maximum":120})",
          { "-5", "-1", "0", "7", "99", "100", "120" }, { "-6", "-10", "121", "130", "1000", "1.0" } },
        { R"({"type":"integer","exclusiveMinimum":0})", { "1", "57", "1000" }, { "0", "-1" } },
        { R"({"type":"string","minLength":2,"maxLength":4})",
          { R"("ab")", R"("abcd")", R"("éé")" }, { R"("a")", R"("abcde")", R"("")" } },
        { R"({"type":"string","pattern":"^[a-z]{2,5}-[0-9]+$"})",
          { R"("ab-1")", R"("abcde-123")" },
          { R"("a-1")", R"("ab-")", R"("abcdef-1")", R"("AB-1")", R"("ab1")" } },
        /* A pattern without ^ or $ searches, as JSON Schema says: the missing side is any text.
         * Tool schemas send them bare; on array items a tool call's JSON arguments reach this
         * converter, so one bare pattern there failed the whole request. */
        { R"({"type":"string","pattern":"[0-9]+"})",
          { R"("7")", R"("ab12c")" }, { R"("abc")", R"("")" } },
        { R"({"type":"string","pattern":"^ab"})", { R"("ab")", R"("abz")" }, { R"("zab")" } },
        { R"({"type":"string","pattern":"ab$"})", { R"("ab")", R"("zab")" }, { R"("abz")" } },
        { R"({"type":"string","pattern":"a\\$"})", { R"("a$")", R"("za$z")" }, { R"("a")" } },
        { R"({"type":"string","pattern":"x|yz"})", { R"("x")", R"("ayzb")" }, { R"("y")" } },
        { R"({"type":"array","items":{"type":"string","pattern":"[0-9]+"}})",
          { R"(["1","x2"])", R"(["x\"1"])", "[]" }, { R"(["x"])", R"(["1","x"])" } },
        /* \d \w \s and their negations (0018), as tool schemas send them: oh-my-pi's workpool tool
         * carries "\S" thirteen times. A negated class must not cross the closing quote. */
        { R"({"type":"string","pattern":"\\S"})", { R"("a")", R"(" a ")" }, { R"("")", R"("   ")" } },
        { R"({"type":"array","items":{"type":"string","pattern":"\\S"}})",
          { R"(["a"," b"])" }, { R"(["  "])", R"(["a","  "])" } },
        { R"({"type":"string","pattern":"^\\d{4}-\\d{2}-\\d{2}$"})",
          { R"("2026-10-19")" }, { R"("2026-1-19")", R"("20x6-10-19")" } },
        { R"({"type":"string","pattern":"^[\\w-]+$"})", { R"("a_b-1")" }, { R"("a b")", R"("")" } },
        { R"({"type":"string","pattern":"^-?\\d+(?:\\.\\d+)?$"})",
          { R"("3")", R"("-3.14")" }, { R"("3.")", R"("x")" } },
        { R"({"type":"string","pattern":"^a\\sb$"})", { R"("a b")", R"("a\tb")" }, { R"("ab")" } },
        { R"({"type":"string","pattern":"^[\\s\\S]{2}$"})", { R"("ab")", R"("a ")" }, { R"("a")" } },
        { R"({"type":"string","pattern":"^\\D$"})", { R"("x")" }, { R"("7")" } },
        /* identity escapes of punctuation (0019): zod's email pattern sends "\-" inside a class, and
         * JavaScript-written patterns escape "/". An escaped '-' in a class is not a range. */
        { R"({"type":"string","pattern":"^a\\/b$"})", { R"("a/b")" }, { R"("ab")" } },
        { R"({"type":"string","pattern":"^a\\-b$"})", { R"("a-b")" }, { R"("ab")" } },
        { R"({"type":"string","pattern":"^[a\\-z]$"})", { R"("a")", R"("-")", R"("z")" }, { R"("b")" } },
        { R"({"type":"string","pattern":"^[a\\^]$"})", { R"("^")", R"("a")" }, { R"("b")" } },
        { R"({"type":"string","pattern":"^[\\^a]$"})", { R"("^")", R"("a")" }, { R"("b")" } },
        { R"({"type":"string","pattern":"^[\\--z]$"})", { R"("-")", R"("a")" }, { R"("+")" } },
        { R"({"type":"string","pattern":"^[\\.]$"})", { R"(".")" }, { R"("a")" } },
        { R"({"type":"string","pattern":"a\\-b"})", { R"("xa-by")" }, { R"("ab")" } },
        /* control-character escapes (0020): a JSON string holds a tab or a backspace only as an
         * escape, so [\t] [\b] [\f] [\v] [\0] and \t outside a class match the escape, and an
         * escaped space or non-ASCII character is itself. */
        { R"({"type":"string","pattern":"^[\\b]$"})", { R"("\b")" }, { R"("b")", R"("\\")" } },
        { R"({"type":"string","pattern":"^[a\\t]+$"})", { R"("a\ta")", R"("a")" }, { R"("t")", R"("a b")" } },
        { R"({"type":"string","pattern":"^[\\f\\v]$"})", { R"("\f")", R"("\u000b")", R"("\u000B")" }, { R"("f")", R"("v")" } },
        { R"({"type":"string","pattern":"^[\\0]$"})", { R"("\u0000")" }, { R"("0")" } },
        { R"({"type":"string","pattern":"^a\\tb$"})", { R"("a\tb")" }, { R"("atb")" } },
        { R"({"type":"string","pattern":"^[a\\ ]+$"})", { R"("a a")" }, { R"("ab")" } },
        { R"({"type":"string","pattern":"^a\\ b$"})", { R"("a b")" }, { R"("ab")" } },
        { R"({"type":"string","pattern":"^[\\s]$"})", { R"(" ")", R"("\n")" }, { R"("n")" } },
        { R"({"type":"string","pattern":"^[^\\t]$"})", { R"("a")" }, { R"("")" } },
        { R"({"type":"string","pattern":"^a\\t+b$"})", { R"("a\t\tb")" }, { R"("a\tttb")", R"("ab")" } },
        { R"({"type":"string","pattern":"^x\\v{2}$"})", { R"("x\u000b\u000B")" }, { R"("x\u000b")" } },
        { R"({"type":"string","pattern":"^[\\t]{2}$"})", { R"("\t\t")" }, { R"("\t")" } },
        { R"({"type":"string","pattern":"^[^\\n]+$"})", { R"("ab")", R"("a\tb")" }, { R"("a\nb")" } },
        { R"({"type":"string","pattern":"^[^\\s]+$"})", { R"("ab")" }, { R"("\t")", R"("a b")" } },
        { R"({"type":"string","pattern":"^(?:[A-Za-z0-9_'+\\-]+\\.)*[A-Za-z0-9_'+\\-]*[A-Za-z0-9_+-]@(?:[A-Za-z0-9][A-Za-z0-9\\-]*\\.)+[A-Za-z]{2,}$"})",
          { R"("a.b@c.io")", R"("x-y@d-e.org")" }, { R"("a@b")", R"("a b@c.io")" } },
        { R"({"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":3})",
          { "[1]", "[1,2,3]", "[-4,0]" }, { "[]", "[1,2,3,4]", R"(["a"])" } },
        { R"({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"}},)"
          R"("required":["a"]})",
          { R"({"a":1})", R"({"a":1,"b":"x"})" }, { R"({"b":"x"})", "{}", R"({"a":"1"})" } },
        { R"({"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":false})",
          { R"({"a":1})", "{}" }, { R"({"a":1,"c":2})", R"({"c":2})" } },
        { R"({"$defs":{"p":{"type":"integer","minimum":0}},"type":"object",)"
          R"("properties":{"x":{"$ref":"#/$defs/p"}},"required":["x"]})",
          { R"({"x":3})", R"({"x":0})" }, { R"({"x":-1})", "{}" } },
        { R"({"type":"number"})", { "0.5", "-1", "2.25", "1e3" }, { R"("1")", "1.", ".5" } },
        { R"({"type":"boolean"})", { "true", "false" }, { "1", "null", R"("true")" } },
    };
    /* Known gap, not enforced: the converter drops minimum/maximum on a "number" (it bounds only
     * integers), so {"type":"number","minimum":0.5} admits -1. Refusing such schemas would refuse
     * common structured-output requests; it is reported rather than pinned here. */
    if (schema_admits(bv->v, R"({"type":"number","minimum":0.5})", "-1"))
        fprintf(stderr, "    SKIP number bounds: the converter does not enforce minimum/maximum "
                        "on \"number\"\n");
    for (const char* schema : { R"({"type":"string","pattern":"^a|b"})",
                                R"({"type":"string","pattern":"a|b$"})",
                                R"({"type":"string","pattern":"^[a\\S]$"})",
                                R"({"type":"string","pattern":"^[\\p{L}]$"})",
                                R"({"type":"string","pattern":"^[\\cA]$"})",
                                R"({"type":"string","pattern":"^[a\\1]$"})",
                                R"({"type":"string","pattern":"^[\\01]$"})",
                                R"({"type":"string","pattern":"^[\\xZ]$"})",
                                R"({"type":"string","pattern":"^[\\u{41}]$"})" }) {
        std::string gbnf, err;
        CHECK(grammar_from_json_schema(schema, "root", &gbnf, &err) < 0);
    }
    for (const Case& c : cases) {
        for (const char* y : c.yes) {
            const bool ok = schema_admits(bv->v, c.schema, y);
            CHECK(ok);
            if (!ok) fprintf(stderr, "    %s refused %s\n", c.schema, y);
        }
        for (const char* n : c.no) {
            const bool ok = schema_admits(bv->v, c.schema, n);
            CHECK(!ok);
            if (ok) fprintf(stderr, "    %s admitted %s\n", c.schema, n);
        }
    }
}

/* Whatever a walk through the engine's masks writes is valid against the schema: random admitted
 * tokens until the parse is complete, then parsed with a JSON parser and checked against the
 * keywords by hand. The walk leaves out the pieces upstream's decoder reads leniently (an inner
 * NUL ends a piece early; a byte that never occurs in UTF-8 is taken as a lead), whose bytes are
 * not the characters the grammar saw. Seeded; the failing text is printed. */
TEST(json_schema_walks_write_only_valid_instances) {
    auto bv = bpe_like(67, 1500);
    const std::string schema =
        R"({"type":"object","properties":{"n":{"type":"integer","minimum":-20,"maximum":300},)"
        R"("s":{"type":"string","minLength":1,"maxLength":5},)"
        R"("t":{"type":"array","items":{"enum":["x","y"]},"maxItems":2}},"required":["n","s"],)"
        R"("additionalProperties":false})";
    std::string gbnf, err;
    REQUIRE(grammar_from_json_schema(schema, "root", &gbnf, &err) >= 0);
    const int64_t n_words = bitmask_words(bv->v.n_tokens());
    std::vector<uint32_t> m((size_t)n_words);
    int finished = 0;
    for (uint64_t seed = 1; seed <= 120; ++seed) {
        std::unique_ptr<GbnfGrammar> g;
        REQUIRE(GbnfGrammar::create(gbnf, "root", &bv->v, false, {}, {}, &g, &err) >= 0);
        Rng rng(seed);
        std::string text;
        bool done = false;
        for (int step = 0; step < 200 && !done; ++step) {
            if (g->complete() && rng.chance(50)) { done = true; break; }
            if (g->fill_mask(m.data(), n_words) < 0) break;
            int32_t id = -1;
            for (int attempt = 0; attempt < 16; ++attempt) {
                id = pick_admitted(m, *bv, rng);
                if (id < 0 || bv->v.is_eog(id)) break;
                bool lenient = bv->nul[(size_t)id] != 0;
                for (unsigned char ch : bv->v.token_piece(id))
                    lenient = lenient || ch == 0 || ch == 0xC0 || ch == 0xC1 || ch >= 0xF5;
                if (!lenient) break;
                id = -1;
            }
            if (id < 0 || bv->v.is_eog(id)) { done = g->complete(); break; }
            if (g->accept_token(id) < 0) break;
            text += bv->v.token_piece(id);
        }
        if (!done) continue;
        ++finished;
        bool ok = false;
        try {
            const nlohmann::json j = nlohmann::json::parse(text);
            ok = j.is_object() && j.contains("n") && j.contains("s") && j["n"].is_number_integer() &&
                 j["n"].get<int64_t>() >= -20 && j["n"].get<int64_t>() <= 300 && j["s"].is_string();
            if (ok) {
                const std::string s = j["s"].get<std::string>();
                size_t chars = 0;
                for (unsigned char ch : s) chars += (ch & 0xC0) != 0x80;
                ok = chars >= 1 && chars <= 5;
            }
            for (auto it = j.begin(); ok && it != j.end(); ++it)
                ok = it.key() == "n" || it.key() == "s" || it.key() == "t";
            if (ok && j.contains("t")) {
                ok = j["t"].is_array() && j["t"].size() <= 2;
                for (const auto& e : j["t"]) ok = ok && (e == "x" || e == "y");
            }
        } catch (const std::exception&) {
            ok = false;
        }
        CHECK(ok);
        if (!ok) fprintf(stderr, "    seed %llu wrote %s\n", (unsigned long long)seed, show_piece(text).c_str());
    }
    CHECK(finished > 60);
}

RAD_TEST_MAIN()
