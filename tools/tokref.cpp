/* tokref.cpp -- a reference encoder: llama.cpp's splitters, std::regex and a string-keyed merge.
 *
 * A reference, not a fallback; core/ does not link it. A `Split` goes through llama.cpp's
 * unicode_regex_split (hand-written splitters for known patterns, std::regex over a
 * category-collapsed copy of the text otherwise), and merging uses a string-keyed bigram queue.
 * It is independent of core/text/tokenizer_regex.cpp and tokenizer_bpe.cpp. Known divergences are
 * listed in tokref.h.
 */
#include "tokref.h"

#include "text/tokenizer_ucd.h"
#include "text/unicode_norm.h"
#include "llama/src/unicode.h"

#include <algorithm>
#include <cfloat>
#include <queue>
#include <regex>

namespace rad {
namespace tokref {

namespace {

struct ByteLevelAlphabet {
    std::string to_utf8[256];                         /* byte -> its 1-2 byte UTF-8 spelling */
    std::unordered_map<std::string, uint8_t> from;    /* and back */
    uint32_t cpt_of[256];

    ByteLevelAlphabet() {
        bool taken[256] = { false };
        auto give = [&](int b, uint32_t cpt) {
            to_utf8[b] = unicode_cpt_to_utf8(cpt);
            cpt_of[b]  = cpt;
            from[to_utf8[b]] = (uint8_t)b;
            taken[b] = true;
        };
        for (int b = 0x21; b <= 0x7E; ++b) give(b, (uint32_t)b);   /* '!' .. '~'  */
        for (int b = 0xA1; b <= 0xAC; ++b) give(b, (uint32_t)b);   /* U+00A1..AC  */
        for (int b = 0xAE; b <= 0xFF; ++b) give(b, (uint32_t)b);   /* U+00AE..FF  */
        uint32_t n = 0;
        for (int b = 0; b < 256; ++b) if (!taken[b]) give(b, 256 + n++);
    }
};

const ByteLevelAlphabet& alphabet() {
    static const ByteLevelAlphabet a;
    return a;
}

std::string byte_encode(std::string_view raw) {
    const ByteLevelAlphabet& a = alphabet();
    std::string out;
    out.reserve(raw.size() * 2);
    for (unsigned char c : raw) out += a.to_utf8[c];
    return out;
}

constexpr const char* k_metaspace = "\xe2\x96\x81";   /* U+2581 LOWER ONE EIGHTH BLOCK */

void replace_all(std::string& s, std::string_view from, std::string_view to) {
    if (from.empty()) return;
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    for (;;) {
        size_t j = s.find(from, i);
        if (j == std::string::npos) { out.append(s, i, std::string::npos); break; }
        out.append(s, i, j - i);
        out.append(to);
        i = j + from.size();
    }
    s.swap(out);
}

/* ================================================================== the pre-tokeniser regex
 *
 * Structural patterns: vendor/llama/src/unicode.cpp has hand-written splitters for the patterns
 * that ship on models (GPT-2, Llama-3, Qwen2, Qwen3.5, Kimi-K2, the AFMoE digit rule and
 * "[^\n]+|[\n]+"). They are codepoint-class state machines without backtracking, correct on
 * lookaheads such as `\s+(?!\S)`. `unicode_regex_split` picks one by exact pattern string.
 *
 * Everything else falls through to std::regex over a copy of the text where each non-ASCII
 * codepoint is replaced by one byte for its Unicode category, which makes `\p{L}` work. Limits:
 *   - a pattern mixing `\p{...}` with a non-ASCII literal is refused, not approximated;
 *   - possessive quantifiers and atomic groups (not ECMAScript) are unsupported;
 *   - std::regex backtracks, so a pathological pattern is slow rather than wrong.
 * A model that needs more gets a hand-written splitter.
 *
 * One rewrite is applied before dispatch: tokenizer.json's `(?i:'s|'t|'re|'ve|'m|'ll|'d)` becomes
 * explicit classes, since ECMAScript has no inline flag group and llama.cpp keys its structural
 * splitters off the expanded form. Without it Qwen's pattern would miss its structural splitter. */
const char* k_ci_contractions_in  = "(?i:'s|'t|'re|'ve|'m|'ll|'d)";
const char* k_ci_contractions_out = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";

std::string canonicalise_pattern(const std::string& p) {
    std::string out = p;
    replace_all(out, k_ci_contractions_in, k_ci_contractions_out);
    return out;
}

/* Whether `unicode_regex_split` has a structural splitter for this pattern (diagnostics only).
 * Canonicalises first, as apply_split does, or a structurally handled pattern would be reported as
 * a std::regex fallback. */
bool pattern_is_structural(const std::string& raw) {
    const std::string p = canonicalise_pattern(raw);
    static const char* k_structural[] = {
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
        "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|"
        "\\s+(?!\\S)|\\s+",
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
        "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|"
        "\\s+(?!\\S)|\\s+",
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|"
        "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        "\\p{Han}+",
        "\\p{AFMoE_digits}",
        "[^\\n]+|[\\n]+",
        "\\d{1,3}(?=(?:\\d{3})*\\b)",
    };
    for (const char* s : k_structural) if (p == s) return true;
    return false;
}
/* --- normalisers ------------------------------------------------------------------------- */

/* A literal Replace is find-and-replace; a regex Replace uses std::regex, acceptable here because
 * normalisers run once per prompt and real patterns (`\s+` -> ` `, `` -> `_`) are trivial. */
void apply_replace(std::string& text, const VocabStep& s) {
    if (s.flags & 1) {
        try {
            text = std::regex_replace(text, std::regex(s.arg0), s.arg1);
        } catch (const std::regex_error& e) {
            /* Warn and leave the text unchanged. */
            RAD_WARN("tokenizer: Replace pattern '%s' did not compile (%s); step skipped",
                     s.arg0.c_str(), e.what());
        }
    } else {
        replace_all(text, s.arg0, s.arg1);
    }
}

void apply_norm(std::string& text, const VocabStep& s) {
    switch (s.kind) {
        case RAD_NORM_NFC:  text = unorm_utf8(text, false, true);  break;
        case RAD_NORM_NFKC: text = unorm_utf8(text, true,  true);  break;
        case RAD_NORM_NFD:  text = unorm_utf8(text, false, false); break;
        case RAD_NORM_NFKD: text = unorm_utf8(text, true,  false); break;
        case RAD_NORM_LOWERCASE: {
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(text);
            std::string out;
            out.reserve(text.size());
            /* Rust's char::to_lowercase, as HuggingFace runs it: U+0130 is "i" U+0307. */
            for (uint32_t c : cpts) {
                if (c == 0x130) out += "i\xcc\x87";
                else out += unicode_cpt_to_utf8(unicode_tolower(c));
            }
            text.swap(out);
            break;
        }
        case RAD_NORM_STRIP: {
            /* Unicode whitespace, not isspace(): HuggingFace strips e.g. a non-breaking space. */
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(text);
            size_t a = 0, b = cpts.size();
            if (s.flags & 1) while (a < b && unicode_cpt_flags_from_cpt(cpts[a]).is_whitespace) ++a;
            if (s.flags & 2)
                while (b > a && unicode_cpt_flags_from_cpt(cpts[b-1]).is_whitespace) --b;
            std::string out;
            for (size_t i = a; i < b; ++i) out += unicode_cpt_to_utf8(cpts[i]);
            text.swap(out);
            break;
        }
        case RAD_NORM_REPLACE: apply_replace(text, s); break;
        case RAD_NORM_PREPEND: text = s.arg0 + text; break;
        default:
            RAD_WARN("tokenizer: normaliser step %u is not implemented; text passed through",
                     s.kind);
            break;
    }
}

/* --- pre-tokenisers ---------------------------------------------------------------------- */

enum { SPLIT_REMOVED = 0, SPLIT_ISOLATED = 1, SPLIT_MERGED_PREV = 2,
       SPLIT_MERGED_NEXT = 3, SPLIT_CONTIGUOUS = 4 };

void apply_split(std::vector<std::string>& words, const VocabStep& s) {
    const std::string pat = canonicalise_pattern(s.arg0);
    const std::vector<std::string> exprs = { pat };
    std::vector<std::string> out;
    out.reserve(words.size() * 2);
    for (const std::string& w : words) {
        if (w.empty()) continue;
        /* byte_encode=false: ByteLevel is its own declared step. */
        std::vector<std::string> parts;
        try {
            parts = unicode_regex_split(w, exprs, /*byte_encode=*/false);
        } catch (const std::exception& e) {
            RAD_ERR("tokenizer: Split('%s') failed: %s", s.arg0.c_str(), e.what());
            parts = { w };
        }
        /* Every behaviour is treated as Isolated (see tokref.h): behaviour only affects
         * delimiters, and shipped patterns are Isolated and match the pieces themselves. */
        for (std::string& p : parts) if (!p.empty()) out.push_back(std::move(p));
    }
    words.swap(out);
}

void apply_bytelevel(std::vector<std::string>& words, const VocabStep& s) {
    const bool add_prefix_space = (s.flags & 1) != 0;
    const bool use_regex        = (s.flags & 2) != 0;

    if (add_prefix_space && !words.empty() && !words[0].empty() && words[0][0] != ' ') {
        words[0].insert(words[0].begin(), ' ');
    }
    if (use_regex) {
        /* HuggingFace's ByteLevel can carry the GPT-2 pattern inline. rad-convert normally emits
         * it as an explicit Split step, but the inline form must still work. */
        VocabStep split;
        split.kind = RAD_PRE_SPLIT;
        split.arg0 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|"
                     "\\s+(?!\\S)";
        split.iarg = SPLIT_ISOLATED;
        apply_split(words, split);
    }
    for (std::string& w : words) w = byte_encode(w);
}

void apply_metaspace(std::vector<std::string>& words, const VocabStep& s, bool at_start) {
    const std::string rep = s.arg0.empty() ? std::string(k_metaspace) : s.arg0;
    const int64_t prepend = s.iarg;            /* 0 never, 1 first, 2 always */
    const bool    split   = (s.flags & 1) != 0;

    for (size_t i = 0; i < words.size(); ++i) {
        std::string& w = words[i];
        if (prepend == 2 || (prepend == 1 && i == 0 && at_start)) {
            if (w.empty() || w.compare(0, 1, " ") != 0) w.insert(0, " ");
        }
        replace_all(w, " ", rep);
    }
    if (!split) return;
    std::vector<std::string> out;
    for (const std::string& w : words) {
        size_t i = 0;
        while (i < w.size()) {
            size_t j = w.find(rep, i + (w.compare(i, rep.size(), rep) == 0 ? rep.size() : 0));
            if (j == std::string::npos) { out.push_back(w.substr(i)); break; }
            out.push_back(w.substr(i, j - i));
            i = j;
        }
        if (w.empty()) out.push_back(w);
    }
    words.swap(out);
}

/* Membership in one of tokenizer_ucd.h's sorted range tables. */
bool in_ranges(const ucd::CpRange* r, size_t n, uint32_t c) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (r[mid].last < c) lo = mid + 1;
        else if (r[mid].first > c) hi = mid;
        else return true;
    }
    return false;
}

uint32_t gc_of(uint32_t c) {
    size_t lo = 0, hi = ucd::k_gc_runs_n;
    while (hi - lo > 1) {
        const size_t mid = (lo + hi) / 2;
        if (ucd::k_gc_runs[mid].first <= c) lo = mid; else hi = mid;
    }
    return ucd::k_gc_runs[lo].gc;
}

bool is_space(uint32_t c) { return in_ranges(ucd::k_white_space, ucd::k_white_space_n, c); }

/* Rust's char::is_ascii_punctuation || char::is_punctuation (a P* category), the test
 * HuggingFace's Punctuation and BertPreTokenizer split on. */
bool is_hf_punct(uint32_t c) {
    if ((c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) || (c >= 0x5B && c <= 0x60) ||
        (c >= 0x7B && c <= 0x7E))
        return true;
    const uint32_t g = gc_of(c);
    return g >= ucd::Pc && g <= ucd::Po;
}

/* Rust's char::is_numeric, the test Digits splits on: Nd, Nl, No. */
bool is_numeric(uint32_t c) {
    const uint32_t g = gc_of(c);
    return g == ucd::Nd || g == ucd::Nl || g == ucd::No;
}

/* NormalizedString::split with a character predicate: every character it accepts is a delimiter
 * of its own, the runs between them are the other parts, and `behavior` folds the two. */
template <class Pred>
void split_chars(std::vector<std::string>& words, Pred is_delim, int behavior) {
    std::vector<std::string> out;
    for (const std::string& w : words) {
        std::vector<std::pair<std::string, bool>> parts;
        for (uint32_t c : unicode_cpts_from_utf8(w)) {
            const bool d = is_delim(c);
            if (!d && !parts.empty() && !parts.back().second)
                parts.back().first += unicode_cpt_to_utf8(c);
            else parts.push_back({ unicode_cpt_to_utf8(c), d });
        }
        std::vector<std::string> kept;
        bool prev = false;
        switch (behavior) {
            case SPLIT_REMOVED:
                for (auto& p : parts) if (!p.second) kept.push_back(p.first);
                break;
            case SPLIT_MERGED_PREV:
                for (auto& p : parts) {
                    if (p.second && !prev && !kept.empty()) kept.back() += p.first;
                    else kept.push_back(p.first);
                    prev = p.second;
                }
                break;
            case SPLIT_MERGED_NEXT:
                for (size_t k = parts.size(); k-- > 0; ) {
                    if (parts[k].second && !prev && !kept.empty())
                        kept.back() = parts[k].first + kept.back();
                    else kept.push_back(parts[k].first);
                    prev = parts[k].second;
                }
                std::reverse(kept.begin(), kept.end());
                break;
            case SPLIT_CONTIGUOUS:
                for (auto& p : parts) {
                    if (p.second == prev && !kept.empty()) kept.back() += p.first;
                    else kept.push_back(p.first);
                    prev = p.second;
                }
                break;
            default:
                for (auto& p : parts) kept.push_back(p.first);
                break;
        }
        for (std::string& k : kept) if (!k.empty()) out.push_back(std::move(k));
    }
    words.swap(out);
}

/* Rust's regex \w, which HuggingFace's Whitespace uses: Alphabetic (letters, Nl, and the
 * circled and squared Latin letters among the symbols), marks, Nd, Pc and the two joiners. */
bool is_rust_word(uint32_t c) {
    const uint32_t g = gc_of(c);
    return g <= ucd::Lo || g == ucd::Nl || (g >= ucd::Mn && g <= ucd::Me) || g == ucd::Nd ||
           g == ucd::Pc || c == 0x200C || c == 0x200D || (c >= 0x24B6 && c <= 0x24E9) ||
           (c >= 0x1F130 && c <= 0x1F149) || (c >= 0x1F150 && c <= 0x1F169) ||
           (c >= 0x1F170 && c <= 0x1F189);
}

/* HuggingFace's Whitespace: the matches of \w+|[^\w\s]+, everything else removed. Alternation is
 * leftmost-first, so a run of \w is taken first, then a run of what is neither \w nor \s. */
void apply_whitespace_words(std::vector<std::string>& words) {
    std::vector<std::string> out;
    for (const std::string& w : words) {
        const std::vector<uint32_t> cpts = unicode_cpts_from_utf8(w);
        auto word  = [](uint32_t c) { return is_rust_word(c); };
        auto other = [](uint32_t c) { return !is_rust_word(c) && !is_space(c); };
        size_t i = 0;
        while (i < cpts.size()) {
            size_t j = i;
            if (word(cpts[i]))       while (j < cpts.size() && word(cpts[j])) ++j;
            else if (other(cpts[i])) while (j < cpts.size() && other(cpts[j])) ++j;
            if (j == i) { ++i; continue; }
            std::string piece;
            for (size_t k = i; k < j; ++k) piece += unicode_cpt_to_utf8(cpts[k]);
            out.push_back(std::move(piece));
            i = j;
        }
    }
    words.swap(out);
}

void apply_whitespace(std::vector<std::string>& words, const VocabStep& s) {
    if (s.flags == kPreWhitespace) { apply_whitespace_words(words); return; }
    split_chars(words, is_space, SPLIT_REMOVED);
    if (s.flags == kPreBert) split_chars(words, is_hf_punct, SPLIT_ISOLATED);
}

void apply_punctuation(std::vector<std::string>& words, const VocabStep& s) {
    split_chars(words, is_hf_punct, (int)s.iarg);
}

void apply_digits(std::vector<std::string>& words, const VocabStep& s) {
    split_chars(words, is_numeric, (s.flags & 1) ? SPLIT_ISOLATED : SPLIT_CONTIGUOUS);
}

}  /* namespace */

std::vector<std::string> split(const std::string& pattern, const std::string& word) {
    VocabStep s;
    s.kind = RAD_PRE_SPLIT;
    s.arg0 = pattern;
    s.iarg = SPLIT_ISOLATED;
    std::vector<std::string> words = { word };
    apply_split(words, s);
    return words;
}

bool structural(const std::string& pattern) { return pattern_is_structural(pattern); }

namespace {

/* `at_start`: the text is the first fragment, which HuggingFace's Metaspace "first" prepends to. */
int run_pre(const Vocab& v, const std::string& text, std::vector<std::string>& words,
            bool at_start) {
    std::string t = text;
    for (const VocabStep& s : v.steps()) if (step_is_norm(s.kind)) apply_norm(t, s);

    words.assign(1, t);
    for (const VocabStep& s : v.steps()) {
        if (!step_is_pre(s.kind)) continue;
        switch (s.kind) {
            case RAD_PRE_SPLIT:       apply_split(words, s); break;
            case RAD_PRE_BYTELEVEL:   apply_bytelevel(words, s); break;
            case RAD_PRE_METASPACE:   apply_metaspace(words, s, at_start); break;
            case RAD_PRE_WHITESPACE:  apply_whitespace(words, s); break;
            case RAD_PRE_PUNCTUATION: apply_punctuation(words, s); break;
            case RAD_PRE_DIGITS:      apply_digits(words, s); break;
            case RAD_PRE_BYTE_FALLBACK:
                /* Declares how unknown pieces are handled, not a split; bpe() applies it. */
                break;
            default:
                RAD_ERR("tokenizer: pre-tokeniser step %u is not implemented", s.kind);
                return RAD_E_UNSUPPORTED;
        }
    }
    return RAD_OK;
}

/* The byte-fallback token for byte c, spelt "<0xNN>"; -1 when the vocab has none. */
int32_t byte_token(const Vocab& v, unsigned char c) {
    static const char* hex = "0123456789ABCDEF";
    const char buf[7] = { '<', '0', 'x', hex[c >> 4], hex[c & 15], '>', 0 };
    return v.find(buf);
}

/* The BPE merge loop, after llama.cpp's llm_tokenizer_bpe_session (MIT, 06938ac12). A priority
 * queue of candidate bigrams ordered by merge rank: pop the best, splice the two symbols, offer the
 * new neighbours. Stale entries are detected by comparing recorded text with the symbols' current
 * text instead of deleting from the heap. */
void bpe(const Vocab& v, const std::vector<std::string>& words, std::vector<int32_t>& out) {
    struct Symbol { int prev, next; const char* text; size_t n; };
    struct Bigram {
        int left, right, rank;
        size_t size;
        std::string text;
        bool operator<(const Bigram& o) const {
            /* Lower rank wins (inverted for a max-heap); ties go to the leftmost position, so the
             * result does not depend on heap order. */
            return rank > o.rank || (rank == o.rank && left > o.left);
        }
    };

    std::vector<Symbol> symbols, final_syms;
    std::priority_queue<Bigram> work;
    int final_prev = -1;

    /* Byte fallback applies only when the chain declares it; otherwise an unmatched piece is
     * dropped, as in llama.cpp (a vocab with a full byte alphabet never misses). */
    bool have_byte_fallback = false;
    bool have_bytelevel = false;
    for (const VocabStep& s : v.steps()) {
        if (s.kind == RAD_PRE_BYTE_FALLBACK) have_byte_fallback = true;
        if (s.kind == RAD_PRE_BYTELEVEL)     have_bytelevel = true;
    }

    for (const std::string& word : words) {
        if (word.empty()) continue;
        work = std::priority_queue<Bigram>();
        symbols.clear();

        size_t offset = 0;
        int index = 0;
        /* A user-defined token is taken whole: added tokens must round-trip, and the merge table
         * was not fitted over them. Ordinary entries go through the merge loop unless the vocab
         * sets `ignore_merges`, which takes any non-control entry whole (see bpe_word in
         * core/text/tokenizer.cpp for why control entries are excluded). */
        const int32_t whole = v.find(word);
        if (whole >= 0 && (v.type(whole) == RAD_TT_USER_DEFINED ||
                           (v.build().ignore_merges && v.type(whole) != RAD_TT_CONTROL))) {
            symbols.push_back({ -1, -1, word.c_str(), word.size() });
            offset = word.size();
        }
        while (offset < word.size()) {
            Symbol sym;
            const size_t len = std::min(unicode_len_utf8(word[offset]), word.size() - offset);
            sym.text = word.c_str() + offset;
            sym.n    = len;
            offset  += len;
            sym.prev = index - 1;
            sym.next = offset == word.size() ? -1 : index + 1;
            ++index;
            symbols.push_back(sym);
        }

        auto offer = [&](int l, int r) {
            if (l == -1 || r == -1) return;
            const std::string lt(symbols[l].text, symbols[l].n);
            const std::string rt(symbols[r].text, symbols[r].n);
            const int32_t rank = v.merge_rank(lt, rt);
            if (rank < 0) return;
            work.push({ l, r, rank, lt.size() + rt.size(), lt + rt });
        };

        for (int i = 1; i < (int)symbols.size(); ++i) offer(i - 1, i);

        while (!work.empty()) {
            const Bigram b = work.top();
            work.pop();
            Symbol& l = symbols[b.left];
            Symbol& r = symbols[b.right];
            if (l.n == 0 || r.n == 0) continue;
            if (std::string(l.text, l.n) + std::string(r.text, r.n) != b.text) continue;
            l.n += r.n;
            r.n = 0;
            l.next = r.next;
            if (r.next >= 0) symbols[r.next].prev = b.left;
            offer(l.prev, b.left);
            offer(b.left, l.next);
        }

        for (Symbol& s : symbols) {
            if (s.n == 0) continue;
            s.prev = final_prev;
            s.next = -1;
            if (final_prev != -1) final_syms[final_prev].next = (int)final_syms.size();
            final_syms.push_back(s);
            final_prev = (int)final_syms.size() - 1;
        }
    }

    for (int i = final_syms.empty() ? -1 : 0; i != -1; i = final_syms[i].next) {
        const Symbol& s = final_syms[i];
        if (s.n == 0) continue;
        const std::string str(s.text, s.n);
        const int32_t id = v.find(str);
        if (id >= 0) { out.push_back(id); continue; }

        /* Miss: decompose into single-byte tokens. Under ByteLevel each byte is already a mapped
         * codepoint, so split at codepoint boundaries; under byte fallback a byte is "<0xNN>". */
        if (have_bytelevel) {
            for (size_t k = 0; k < str.size(); ) {
                const size_t n = std::min(unicode_len_utf8(str[k]), str.size() - k);
                const int32_t b = v.find(str.substr(k, n));
                if (b >= 0) out.push_back(b);
                else RAD_WARN("tokenizer: no token for byte-level piece '%s'",
                              str.substr(k, n).c_str());
                k += n;
            }
        } else if (have_byte_fallback) {
            for (unsigned char c : str) {
                const int32_t b = byte_token(v, c);
                if (b >= 0) out.push_back(b);
                else if (v.unk() >= 0) out.push_back(v.unk());
            }
        } else if (v.unk() >= 0) {
            out.push_back(v.unk());
        } else {
            RAD_WARN("tokenizer: piece '%s' is not in the vocab and the chain declares neither "
                     "ByteLevel nor ByteFallback; it is dropped", str.c_str());
        }
    }
}

/* Unigram by Viterbi (SentencePiece's algorithm), after llama.cpp's llm_tokenizer_ugm_session.
 * Scores are summed in double, as the reference does; in float the argmax differs on long
 * inputs. */
void unigram(const Vocab& v, const std::string& text, std::vector<int32_t>& out) {
    if (text.empty()) return;
    const size_t n = text.size();

    struct Best { int32_t id; size_t from; double score; };
    std::vector<Best> best(n + 1, { v.unk(), 0, -DBL_MAX });
    best[0] = { v.unk(), 0, 0.0 };

    /* Unknown-token penalty, as in SentencePiece: min score - 10. Derived here, not stored. */
    double unk_penalty = 0.0;
    if (!v.build().scores.empty()) {
        unk_penalty = *std::min_element(v.build().scores.begin(), v.build().scores.end()) - 10.0;
    }

    /* Prefix scan by substring lookup, bounded by the longest entry so it stays linear in the
     * text. A trie would be faster on a large vocab. */
    size_t longest = 0;
    for (const std::string& t : v.build().tokens) longest = std::max(longest, t.size());

    for (size_t i = 0; i < n; ) {
        const size_t cp_len = std::min(unicode_len_utf8(text[i]), n - i);
        bool matched_whole_cp = false;
        const size_t max_len = std::min(n - i, longest);
        for (size_t l = 1; l <= max_len; ++l) {
            const int32_t id = v.find(std::string_view(text.data() + i, l));
            if (id < 0) continue;
            if (l == cp_len) matched_whole_cp = true;
            const double sc = v.type(id) == RAD_TT_USER_DEFINED ? 0.0 : (double)v.score(id);
            const double cand = best[i].score + sc;
            if (cand > best[i + l].score) best[i + l] = { id, i, cand };
        }
        if (!matched_whole_cp) {
            const double cand = best[i].score + unk_penalty;
            if (cand > best[i + cp_len].score) best[i + cp_len] = { v.unk(), i, cand };
        }
        i += cp_len;
    }

    /* Without an unk token, uncovered text still takes the unk transition with id -1, which must
     * never reach the output. Such spans become byte tokens under byte fallback and are dropped
     * otherwise, as in bpe(). The backtrack runs back to front, so bytes are pushed reversed. */
    const int32_t unk = v.unk();
    bool byte_fallback = false;
    if (unk < 0)
        for (const VocabStep& s : v.steps())
            if (s.kind == RAD_PRE_BYTE_FALLBACK) byte_fallback = true;

    const size_t start = out.size();
    size_t dropped = 0;
    bool prev_unk = false;
    for (size_t pos = n; ; ) {
        const Best& b = best[pos];
        const bool is_unk = (b.id == unk);
        if (is_unk && unk < 0) {
            for (size_t k = pos; k-- > b.from; ) {
                const int32_t id = byte_fallback ? byte_token(v, (unsigned char)text[k]) : -1;
                if (id >= 0) out.push_back(id);
                else         ++dropped;
            }
        } else if (!(prev_unk && is_unk)) {
            out.push_back(b.id);
        }
        if (b.from == 0) break;
        prev_unk = is_unk;
        pos = b.from;
    }
    std::reverse(out.begin() + start, out.end());
    if (dropped)
        RAD_WARN("tokenizer: %zu byte(s) of this text have no piece in the vocab, which has no unk "
                 "token and %s; they are dropped", dropped,
                 byte_fallback ? "lacks a byte token for them" : "declares no byte fallback");
}

/* WordPiece (BERT-family models): greedy longest-match-first with a "##" continuation prefix; a
 * word of more than 100 characters is one unknown token, as in HuggingFace. */
void wordpiece(const Vocab& v, const std::vector<std::string>& words, std::vector<int32_t>& out) {
    for (const std::string& w : words) {
        if (w.empty()) continue;
        size_t i = 0;
        std::vector<int32_t> piece;
        bool ok = unicode_cpts_from_utf8(w).size() <= 100;
        if (!ok) i = w.size();
        while (i < w.size()) {
            size_t j = w.size();
            int32_t id = -1;
            for (; j > i; --j) {
                std::string sub = w.substr(i, j - i);
                if (i > 0) sub = "##" + sub;
                id = v.find(sub);
                if (id >= 0) break;
            }
            if (id < 0) { ok = false; break; }
            piece.push_back(id);
            i = j;
        }
        if (ok) out.insert(out.end(), piece.begin(), piece.end());
        else if (v.unk() >= 0) out.push_back(v.unk());
    }
}

}  /* namespace */

int encode(const Vocab& v, std::string_view text, std::vector<int32_t>& out,
           bool add_special, bool parse_special) {
    if (!v.ready()) return RAD_E_STATE;

    if (add_special && v.add_bos() && v.bos() >= 0) out.push_back(v.bos());

    /* Cut the text into (raw, token, raw, ...) around literal special tokens before any other
     * step, so the pre-tokeniser never splits one. With `parse_special == false` a control token
     * in user text is ordinary text; otherwise "<|im_start|>system" in user content would be a
     * real role switch. */
    struct Frag { bool is_token; int32_t id; size_t off, len; };
    std::vector<Frag> frags;
    const std::string raw(text);
    frags.push_back({ false, -1, 0, raw.size() });

    for (int32_t sid : v.special_ids()) {
        const uint8_t t = v.type(sid);
        if (!parse_special && (t == RAD_TT_CONTROL || t == RAD_TT_UNKNOWN)) continue;
        const std::string& st = v.text(sid);
        if (st.empty()) continue;

        std::vector<Frag> next;
        next.reserve(frags.size());
        for (const Frag& f : frags) {
            if (f.is_token) { next.push_back(f); continue; }
            size_t pos = f.off;
            const size_t end = f.off + f.len;
            for (;;) {
                const size_t m = raw.find(st, pos);
                if (m == std::string::npos || m + st.size() > end) break;
                if (m > pos) next.push_back({ false, -1, pos, m - pos });
                next.push_back({ true, sid, m, st.size() });
                pos = m + st.size();
            }
            if (pos < end) next.push_back({ false, -1, pos, end - pos });
        }
        frags.swap(next);
    }

    for (const Frag& f : frags) {
        if (f.is_token) { out.push_back(f.id); continue; }
        if (f.len == 0) continue;
        const std::string chunk = raw.substr(f.off, f.len);

        std::vector<std::string> words;
        RAD_TRY(run_pre(v, chunk, words, f.off == 0));

        switch (v.kind()) {
            case RAD_TOK_BPE:       bpe(v, words, out); break;
            case RAD_TOK_WORDPIECE: wordpiece(v, words, out); break;
            case RAD_TOK_UNIGRAM: {
                /* Unigram runs over the whole fragment; pre-tokenisation only inserted the
                 * metaspace, so rejoin the words. */
                std::string joined;
                for (const std::string& w : words) joined += w;
                unigram(v, joined, out);
                break;
            }
            default:
                RAD_ERR("tokenizer: kind %u has no encoder", v.kind());
                return RAD_E_UNSUPPORTED;
        }
    }

    if (add_special && v.add_eos() && v.eos() >= 0) out.push_back(v.eos());
    return RAD_OK;
}

}  /* namespace tokref */
}  /* namespace rad */
