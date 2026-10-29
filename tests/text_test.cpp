/* text_test.cpp -- the text frontend.
 *
 * The key test is `tokenize_against_llama_cpp`. Its expected ids come from llama.cpp's
 * `llama_tokenize()` (add_special=false, parse_special=true) over two models with different
 * pre-tokenisers (pre=qwen2 and pre=qwen35). They are baked in, so llama.cpp is needed only to
 * regenerate them (build llama-tokenize and run it over the same strings).
 *
 * The same comparison holds per model on random strings over Latin, Cyrillic, CJK, Hangul, kana,
 * emoji, ZWJ sequences, combining marks and arbitrary bytes, and on real C++ source, with one
 * deliberate divergence pinned by `does_not_promote_control_looking_tokens`. Arbitrary-byte
 * strings on which llama.cpp throws out of llama_tokenize are why vendor/patches/0001 exists.
 *
 * The vocabs (151,669 and 248,320 tokens) are too large to bake, so the parity table reads the
 * GGUFs; when one is missing it skips with a message. RAD_TEST_GGUF and RAD_TEST_GGUF_QWEN35 set
 * the paths. The cases about special tokens, detokenisation and the byte-level alphabet run on a
 * byte-level vocabulary trained in memory (tests/tokenizer_fixture.h) and, when the GGUF is
 * there, on the real one too.
 *
 * The tokenizer.json cases compare decoders, pre-tokenisers and a Gemma-shaped encoder with
 * HuggingFace `tokenizers`, whose answers tests/mk_tokenizer_goldens.py baked in below.
 */
#include "rad_test.h"

#include <fstream>

#include "text/gguf.h"
#include "text/tokenizer.h"
#include "text/tokenizer_nfc.h"
#include "text/unicode_norm.h"
#include "norm_vectors.h"
#include "tokenizer_fixture.h"

#include <functional>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <unistd.h>

using namespace rad;

/* ------------------------------------------------------------------ helpers */

/* The table stores strings escaped so a control character or a lone 0xFF byte is readable in
 * source, using the same escapes as the tool that produced the expectations. */
static std::string unescape(const std::string& in) {
    std::string u;
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] != '\\' || i + 1 >= in.size()) { u += in[i]; continue; }
        ++i;
        switch (in[i]) {
            case 'n': u += '\n'; break;
            case 't': u += '\t'; break;
            case 'r': u += '\r'; break;
            case '\\': u += '\\'; break;
            case 'x': {
                auto h = [](int c) {
                    return c >= '0' && c <= '9' ? c - '0'
                         : c >= 'a' && c <= 'f' ? c - 'a' + 10
                         : c >= 'A' && c <= 'F' ? c - 'A' + 10 : 0;
                };
                if (i + 2 < in.size()) { u += (char)(h(in[i + 1]) * 16 + h(in[i + 2])); i += 2; }
                break;
            }
            default: u += '\\'; u += in[i];
        }
    }
    return u;
}

/* Two models, two pre-tokenisers.
 *
 *   qwen3e  Qwen3-Embedding-0.6B, tokenizer.ggml.pre = "qwen2"
 *   qwen35  Qwen3.5-0.8B,         tokenizer.ggml.pre = "qwen35"
 *
 * They differ in the letter-run rule (`[\p{L}\p{M}]+` versus `\p{L}+`), the difference a
 * hash-the-file pre-tokeniser gets wrong on a model it has not seen. Both are pinned.
 *
 * Both are checkpoint files, not in the repo: RAD_TEST_GGUF and RAD_TEST_GGUF_QWEN35 name them.
 * Unset, the cases that need a vocab skip and name the file they wanted. */
static const char* gguf_path_qwen3e() {
    if (const char* p = std::getenv("RAD_TEST_GGUF")) return p;
    return "Qwen3-Embedding-0.6B-f16.gguf (set RAD_TEST_GGUF)";
}
static const char* gguf_path_qwen35() {
    if (const char* p = std::getenv("RAD_TEST_GGUF_QWEN35")) return p;
    return "Qwen3.5-0.8B-BF16.gguf (set RAD_TEST_GGUF_QWEN35)";
}

/* Loaded once each: reading 150K-250K tokens takes ~0.2 s and every test wants the same one. */
static std::shared_ptr<const Vocab> load_vocab(const char* path) {
    auto g = std::make_shared<GgufFile>();
    if (g->open(path) < 0) return nullptr;
    VocabBuild vb;
    if (vocab_from_gguf(*g, vb) < 0) return nullptr;
    auto out = std::make_shared<Vocab>();
    if (out->load(std::move(vb)) < 0) return nullptr;
    return out;
}

static std::shared_ptr<const Vocab> qwen3e_vocab() {
    static std::shared_ptr<const Vocab> v = load_vocab(gguf_path_qwen3e());
    return v;
}
static std::shared_ptr<const Vocab> qwen35_vocab() {
    static std::shared_ptr<const Vocab> v = load_vocab(gguf_path_qwen35());
    return v;
}

static const char* gguf_path() { return gguf_path_qwen3e(); }

/* A byte-level BPE vocabulary trained in memory on text holding the words the cases use, with
 * Qwen's chat markers as control tokens, its reasoning markers as user-defined ones (rendered, as
 * Qwen3 declares them), and "</s>" as a NORMAL token that looks like a control one. Built once. */
static std::shared_ptr<const Vocab> synthetic_vocab() {
    static std::shared_ptr<const Vocab> v = [] {
        tokfix::TextGen gen(5);
        std::string corpus;
        for (int i = 0; i < 24; ++i)
            corpus += "the answer is 42 and this must never be seen. restating it, Q: what is two "
                      "and two</think>The answer is 4. Q: next done </tool_call> Hello, world! " +
                      gen.text(120) + "\n";
        VocabBuild vb = tokfix::train_bytelevel(corpus, tokfix::k_qwen35, 400);
        VocabStep dec;
        dec.kind = RAD_DEC_BYTELEVEL;
        vb.steps = { tokfix::split_step(tokfix::k_qwen35), tokfix::bytelevel_step(), dec };
        tokfix::add_specials(vb, { { "<|im_start|>", RAD_TT_CONTROL },
                                   { "<|im_end|>", RAD_TT_CONTROL },
                                   { "<think>", RAD_TT_USER_DEFINED },
                                   { "</think>", RAD_TT_USER_DEFINED },
                                   { "</s>", RAD_TT_NORMAL } });
        return tokfix::load(std::move(vb));
    }();
    return v;
}

/* The vocabularies the detokeniser and special-token cases run on: the synthetic one always, and
 * Qwen3-Embedding's when RAD_TEST_GGUF names it. */
struct NamedVocab {
    std::string                  name;
    std::shared_ptr<const Vocab> v;
};
static std::vector<NamedVocab> vocabs() {
    std::vector<NamedVocab> out;
    out.push_back({ "synthetic", synthetic_vocab() });
    if (qwen3e_vocab()) out.push_back({ "qwen3e", qwen3e_vocab() });
    else fprintf(stderr, "    (%s: no vocab at %s; the synthetic one only)\n",
                 ::radtest::current(), gguf_path());
    return out;
}

/* ------------------------------------------------------------------ the baked table */

/* `detok` is what the round trip must produce; null means the input itself, which holds for every
 * valid UTF-8 input. The three invalid ones are lossy by design: an invalid byte becomes U+FFFD
 * when decoded to codepoints and cannot come back. llama.cpp's detokenizer does the same, and
 * these expectations are its output. */
struct Case { const char* text; std::vector<int32_t> ids; const char* detok = nullptr; };

static const std::vector<Case>& cases_qwen35() {
    static const std::vector<Case> k = {
    { "Hello, world!",
      { 9419,11,1814,0 } },
    { "The quick brown fox jumps over the lazy dog. 1234567890",
      { 760,3841,13477,37550,33075,888,279,15217,5388,13,220,16,17,18,19,20,21,22,23,24,15 } },
    { "   leading spaces",
      { 256,6187,12258 } },
    { "trailing spaces   \\t\\t",
      { 371,13868,12258,66227 } },
    { "\\n\\n  mixed\\t whitespace \\n",
      { 271,220,9238,197,35169,695 } },
    { "你好，世界！这是一个测试。",
      { 109266,3709,96748,6115,109455,99449,1710 } },
    { "こんにちは世界。テストです。",
      { 85951,96748,1710,181801,36298,1710 } },
    { "안녕하세요 세계",
      { 148924,154982,88005,155308 } },
    { "Здравствуй, мир",
      { 34279,161541,19587,79734,11,160202 } },
    { "🙂🚀✨",
      { 169171,9008,248,222,169379 } },
    { "👨‍👩‍👧‍👦 family zwj",
      { 9008,239,101,373,235,9008,239,102,373,235,9008,239,100,373,235,9008,239,99,2902,23624,73 } },
    { "🏳️‍🌈 flag zwj vs16",
      { 9008,237,111,29545,373,235,9008,234,230,5016,23624,73,5972,16,21 } },
    { "def main():\\n\\tif x == 1:\\n\\t\\tprint(\"hi\")\\n\\treturn 0\\n",
      { 727,1822,4406,198,720,830,606,220,16,25,198,197,6685,437,5834,871,198,827,220,15,198 } },
    { "```cpp\\nint main() {\\n\\tstd::cout << \"x\" << std::endl;\\n}\\n```",
      { 71093,10504,198,390,1822,363,313,198,6524,476,5914,1079,328,87,1,1079,1407,476,5168,26,198,92,198,71093 } },
    { "bad byte: \\xff here",
      { 13458,4763,25,28373,1532 },  "bad byte: \\xef\\xbf\\xbd here" },
    { "lone continuation \\x80\\x81 end",
      { 75,588,39811,220,9678,809 },  "lone continuation \\xef\\xbf\\xbd\\xef\\xbf\\xbd end" },
    { "truncated emoji \\xf0\\x9f\\x98",
      { 371,36751,40958,220,55413 },  "truncated emoji \\xef\\xbf\\xbd\\xef\\xbf\\xbd\\xef\\xbf\\xbd" },
    { "café naïve Ångström",
      { 895,56868,91603,571,76533,938,485,82628 } },
    { "It's the model's output, isn't it?",
      { 2064,579,279,1558,579,2468,11,4290,914,424,30 } },
    { "<|im_start|>system\\nYou are helpful.<|im_end|>\\n",
      { 248045,8678,198,2523,513,10631,13,248046,198 } },
    };
    return k;
}

/* Qwen3-Embedding-0.6B, tokenizer.ggml.pre = "qwen2". Same strings, different vocab and a
 * different letter-run rule, so the two tables disagree on nearly every line: a tokeniser that
 * used one pre-tokeniser for both would pass only one table.
 */
static const std::vector<Case>& cases_qwen3e() {
    static const std::vector<Case> k = {
    { "Hello, world!",
      { 9707,11,1879,0 } },
    { "The quick brown fox jumps over the lazy dog. 1234567890",
      { 785,3974,13876,38835,34208,916,279,15678,5562,13,220,16,17,18,19,20,21,22,23,24,15 } },
    { "   leading spaces",
      { 256,6388,12621 } },
    { "trailing spaces   \\t\\t",
      { 376,14277,12621,68546 } },
    { "\\n\\n  mixed\\t whitespace \\n",
      { 271,220,9519,197,36372,715 } },
    { "你好，世界！这是一个测试。",
      { 108386,3837,99489,6313,105464,81705,1773 } },
    { "こんにちは世界。テストです。",
      { 89015,99489,1773,140592,37541,1773 } },
    { "안녕하세요 세계",
      { 126246,144370,91145,133196 } },
    { "Здравствуй, мир",
      { 35451,6949,26988,20200,82580,11,137144 } },
    { "🙂🚀✨",
      { 145080,145836,144232 } },
    { "👨‍👩‍👧‍👦 family zwj",
      { 145367,378,235,145233,378,235,145665,378,235,145988,2997,24396,73 } },
    { "🏳️‍🌈 flag zwj vs16",
      { 147233,30543,378,235,145354,5181,24396,73,6165,16,21 } },
    { "def main():\\n\\tif x == 1:\\n\\t\\tprint(\"hi\")\\n\\treturn 0\\n",
      { 750,1887,3932,743,856,621,220,16,510,197,6900,445,6023,1138,853,220,15,198 } },
    { "```cpp\\nint main() {\\n\\tstd::cout << \"x\" << std::endl;\\n}\\n```",
      { 73594,10821,198,396,1887,368,341,6736,486,6104,1115,330,87,1,1115,1460,486,5336,280,532,73594 } },
    { "bad byte: \\xff here",
      { 13855,4922,25,29333,1588 },  "bad byte: \\xef\\xbf\\xbd here" },
    { "lone continuation \\x80\\x81 end",
      { 75,603,41171,220,9973,835 },  "lone continuation \\xef\\xbf\\xbd\\xef\\xbf\\xbd end" },
    { "truncated emoji \\xf0\\x9f\\x98",
      { 376,38007,42365,220,57332 },  "truncated emoji \\xef\\xbf\\xbd\\xef\\xbf\\xbd\\xef\\xbf\\xbd" },
    { "café naïve Ångström",
      { 924,58858,94880,586,79252,968,495,85584 } },
    { "It's the model's output, isn't it?",
      { 2132,594,279,1614,594,2550,11,4436,944,432,30 } },
    { "<|im_start|>system\\nYou are helpful.<|im_end|>\\n",
      { 151644,8948,198,2610,525,10950,13,151645,198 } },
    };
    return k;
}

/* ================================================================== llama.cpp parity */

static void check_table(const char* name, std::shared_ptr<const Vocab> v,
                        const std::vector<Case>& cases) {
    if (!v) {
        fprintf(stderr, "  SKIP tokenize_against_llama_cpp[%s]: vocab not available\n", name);
        return;
    }
    Tokenizer t;
    CHECK_OK(t.init(v));
    for (const Case& c : cases) {
        std::vector<int32_t> got;
        CHECK_OK(t.encode(unescape(c.text), got, /*add_special=*/false, /*parse_special=*/true));
        if (got != c.ids) {
            std::string w, g;
            for (int32_t i : c.ids) w += std::to_string(i) + ",";
            for (int32_t i : got)   g += std::to_string(i) + ",";
            ::radtest::fail(__FILE__, __LINE__,
                            std::string(name) + " \"" + c.text +
                            "\"\n      want " + w + "\n      got  " + g);
        }
    }
}

TEST(tokenize_against_llama_cpp) {
    check_table("qwen3e", qwen3e_vocab(), cases_qwen3e());
    check_table("qwen35", qwen35_vocab(), cases_qwen35());
}

/* A deliberate divergence from llama.cpp, pinned so nobody "fixes" it. llama.cpp forces a
 * hardcoded list of control-looking texts ("<|eot_id|>", "<eos>", "</s>", ...) to CONTROL even when
 * the container says NORMAL; in Qwen3-Embedding "</s>" is id 128247, NORMAL, an ordinary merge.
 * HuggingFace isolates only `added_tokens`, so "</s>" splits into three tokens. We follow the
 * vocab as declared, like HuggingFace; on C++ source and fuzzed text this is the only difference
 * from llama.cpp. Matching llama.cpp here would mean a recognise-the-token heuristic had been
 * added, which spec §12 exists to prevent. */
TEST(does_not_promote_control_looking_tokens) {
    for (const NamedVocab& nv : vocabs()) {
        REQUIRE(nv.v != nullptr);
        const int32_t id = nv.v->find("</s>");
        CHECK(id >= 0);
        CHECK_EQ(nv.v->type(id), (uint8_t)RAD_TT_NORMAL);
        Tokenizer t;
        CHECK_OK(t.init(nv.v));
        std::vector<int32_t> got;
        CHECK_OK(t.encode("x </s> y", got, false, /*parse_special=*/true));
        CHECK(std::find(got.begin(), got.end(), id) == got.end());
    }
}

/* A control token typed by a user must not become a role switch. With parse_special off,
 * "<|im_start|>" is ordinary text and tokenises into several ordinary tokens. */
TEST(special_tokens_are_not_parsed_from_user_text) {
    for (const NamedVocab& nv : vocabs()) {
        Tokenizer t;
        CHECK_OK(t.init(nv.v));
        std::vector<int32_t> as_special, as_text;
        CHECK_OK(t.encode("<|im_start|>", as_special, false, true));
        CHECK_OK(t.encode("<|im_start|>", as_text,    false, false));
        CHECK_EQ(as_special.size(), (size_t)1);
        CHECK(as_text.size() > 1);
        CHECK(as_text != as_special);
    }
}

/* ================================================================== detokenisation */

/* Streaming one token at a time, as the server drives it: the concatenation of every push() plus
 * flush() equals the text. On Qwen3-Embedding the ids are llama.cpp's table, including invalid
 * UTF-8 the byte-level alphabet carried through as bytes; on the synthetic vocabulary they are
 * its own encoding of every well-formed text in the table. */
TEST(detokenize_round_trip) {
    for (const NamedVocab& nv : vocabs()) {
        Tokenizer t;
        CHECK_OK(t.init(nv.v));
        int checked = 0;
        for (const Case& c : cases_qwen3e()) {
            std::string want = unescape(c.detok ? c.detok : c.text);
            std::vector<int32_t> ids = c.ids;
            if (nv.name != "qwen3e") {
                want = unescape(c.text);
                if (utf8_valid_prefix(want) != want.size()) continue;
                ids.clear();
                CHECK_OK(t.encode(want, ids, false, true));
            }
            Detokenizer d(nv.v, {}, /*render_special=*/true);
            std::string got;
            for (int32_t id : ids) got += d.push(id);
            got += d.flush();
            CHECK_EQ(got, want);
            ++checked;
        }
        CHECK(checked > 10);
    }
}

/* Every push() must return complete UTF-8. A three-byte codepoint arriving as three byte tokens
 * must not appear until its last byte does. */
TEST(detokenize_holds_partial_utf8) {
    for (const NamedVocab& nv : vocabs()) {
        /* The worst case directly: a three-byte codepoint's byte-level tokens one at a time.
         * Every byte-level vocab has all 256, so this works on any model. U+4EFF is E4 BB BF,
         * all in the 0xAE..0xFF range the GPT-2 alphabet maps to itself, so each token's text
         * is that codepoint in UTF-8. */
        const std::string cjk = "\xe4\xbb\xbf";
        std::vector<int32_t> ids;
        for (unsigned char b : cjk) {
            /* Looked up rather than encoded: a lone continuation byte is invalid UTF-8 input
             * and encode() would replace it with U+FFFD. This tests the decoder. */
            CHECK(b >= 0xAE);
            std::string enc;
            enc += (char)(0xC0 | (b >> 6));
            enc += (char)(0x80 | (b & 0x3F));
            const int32_t id = nv.v->find(enc);
            CHECK(id >= 0);
            ids.push_back(id);
        }
        CHECK_EQ(ids.size(), (size_t)3);

        Detokenizer d(nv.v);
        std::string all;
        bool saw_partial_hold = false;
        for (int32_t id : ids) {
            const std::string out = d.push(id);
            CHECK_EQ(utf8_valid_prefix(out), out.size());
            if (out.empty()) saw_partial_hold = true;
            all += out;
        }
        all += d.flush();
        CHECK_EQ(all, cjk);
        CHECK(saw_partial_hold);   /* at least one push held everything back */
    }
}

/* The straddle: a stop string arriving across several tokens must never be partially emitted. */
TEST(detokenize_stop_string_straddle) {
    for (const NamedVocab& nv : vocabs()) {
        Tokenizer t;
        CHECK_OK(t.init(nv.v));
        const std::string body = "the answer is 42";
        const std::string stop = "</tool_call>";
        std::vector<int32_t> ids;
        CHECK_OK(t.encode(body + stop + " and this must never be seen", ids, false, true));

        Detokenizer d(nv.v, { stop });
        std::string emitted;
        for (int32_t id : ids) {
            emitted += d.push(id);
            /* No emitted suffix may be a prefix of the stop string; that is what makes emitted
             * text safe to forward to a client immediately. */
            for (size_t n = 1; n < stop.size(); ++n)
                if (emitted.size() >= n) CHECK(emitted.compare(emitted.size() - n, n, stop, 0, n) != 0);
        }
        CHECK(d.stopped());
        CHECK_EQ(d.stop_hit(), stop);
        CHECK_EQ(emitted, body);
        /* flush() after a stop yields nothing: the tail was discarded, not buffered. */
        CHECK_EQ(d.flush(), std::string());
    }
}

/* A reply that starts inside its reasoning block: a stop string written while thinking is text,
 * not a stop, and the first one after the closer cuts as usual. */
TEST(detokenize_stops_wait_for_the_reasoning_closer) {
    for (const NamedVocab& nv : vocabs()) {
        Tokenizer t;
        CHECK_OK(t.init(nv.v));
        const std::string think = "restating it, Q: what is two and two</think>";
        std::vector<int32_t> ids;
        CHECK_OK(t.encode(think + "The answer is 4. Q: next", ids, false, true));
        Detokenizer d(nv.v, { "Q:" });
        d.defer_stops_until("</think>");
        std::string s;
        for (int32_t id : ids) {
            s += d.push(id);
            if (d.stopped()) break;
        }
        CHECK(d.stopped());
        CHECK_EQ(d.stop_hit(), std::string("Q:"));
        CHECK_EQ(s, think + "The answer is 4. ");

        /* Without the deferral the same text stops inside the reasoning. */
        Detokenizer plain(nv.v, { "Q:" });
        std::string p;
        for (int32_t id : ids) {
            p += plain.push(id);
            if (plain.stopped()) break;
        }
        CHECK_EQ(p, std::string("restating it, "));
    }
}

/* A stop string that never completes must be released at flush() rather than swallowed. */
TEST(detokenize_incomplete_stop_is_flushed) {
    for (const NamedVocab& nv : vocabs()) {
        Tokenizer t;
        CHECK_OK(t.init(nv.v));
        std::vector<int32_t> ids;
        CHECK_OK(t.encode("done </tool", ids, false, true));
        Detokenizer d(nv.v, { "</tool_call>" });
        std::string s;
        for (int32_t id : ids) s += d.push(id);
        CHECK(!d.stopped());
        s += d.flush();
        CHECK_EQ(s, std::string("done </tool"));
    }
}

/* ================================================================== byte level */

/* The GPT-2 alphabet, spot-checked at every boundary of the three literal ranges plus the
 * remapped run. A wrong entry silently costs about 20% of the prompt length, so it is pinned. */
TEST(bytelevel_alphabet_is_exact) {
    for (const NamedVocab& nv : vocabs()) {
        Tokenizer t;
        CHECK_OK(t.init(nv.v));
        /* Space (0x20) is not in a literal range, so it is remapped, to U+0120 'Ġ'; every
         * byte-level vocab is built on that. */
        const int32_t g = nv.v->find("\xc4\xa0");
        CHECK(g >= 0);
        std::vector<int32_t> ids;
        CHECK_OK(t.encode(" ", ids, false, true));
        CHECK_EQ(ids.size(), (size_t)1);
        CHECK_EQ(ids[0], g);

        /* Newline (0x0A) is the eleventh remapped byte, U+010A 'Ċ'. */
        const int32_t nl = nv.v->find("\xc4\x8a");
        ids.clear();
        CHECK_OK(t.encode("\n", ids, false, true));
        CHECK_EQ(ids.size(), (size_t)1);
        CHECK_EQ(ids[0], nl);
        /* llama.cpp prints it as the LF token, id 198 in Qwen3-Embedding. */
        if (nv.name == "qwen3e") CHECK_EQ(nl, 198);

        /* '!' is 0x21, the first byte of the first literal range, and stands for itself; 0x7E
         * '~' is its last. */
        CHECK(nv.v->find("!") >= 0);
        CHECK(nv.v->find("~") >= 0);
        CHECK_EQ(t.piece(g), std::string(" "));
        CHECK_EQ(t.piece(nl), std::string("\n"));
    }
}

/* ================================================================== normalisation */

/* Pinned vectors from UAX #15. The tables are generated from ICU
 * (core/text/unicode_norm_data.cpp); these cases would break first if they were regenerated
 * wrong. */
TEST(unicode_normalisation) {
    auto cpts = [](std::initializer_list<uint32_t> v) { return std::vector<uint32_t>(v); };
    /* The four forms, as the tokeniser composes them from the two steps. */
    auto form = [](bool compat, bool compose) {
        return [=](std::vector<uint32_t> v) {
            unorm_decompose(v, compat);
            if (compose) unorm_compose(v);
            return v;
        };
    };
    const auto unorm_nfd = form(false, false), unorm_nfkd = form(true, false);
    const auto unorm_nfc = form(false, true), unorm_nfkc = form(true, true);

    /* Composed vs decomposed Angstrom: U+00C5 decomposes to A + ring, and U+212B (ANGSTROM SIGN)
     * is a singleton that NFC folds onto U+00C5, not back onto itself. */
    CHECK(unorm_nfd(cpts({0x00C5})) == cpts({0x0041, 0x030A}));
    CHECK(unorm_nfc(cpts({0x0041, 0x030A})) == cpts({0x00C5}));
    CHECK(unorm_nfc(cpts({0x212B})) == cpts({0x00C5}));

    /* Canonical ordering: two marks of different combining class are sorted, and composition
     * then sees the one it can compose with. */
    CHECK(unorm_nfd(cpts({0x1E69})) == cpts({0x0073, 0x0323, 0x0307}));
    CHECK(unorm_nfc(cpts({0x0073, 0x0307, 0x0323})) == cpts({0x1E69}));

    /* A composition exclusion: U+0344 decomposes but never recomposes. */
    CHECK(unorm_nfd(cpts({0x0344})) == cpts({0x0308, 0x0301}));
    CHECK(unorm_nfc(cpts({0x0308, 0x0301})) == cpts({0x0308, 0x0301}));

    /* Hangul, which is arithmetic rather than tabular. */
    CHECK(unorm_nfd(cpts({0xAC01})) == cpts({0x1100, 0x1161, 0x11A8}));
    CHECK(unorm_nfc(cpts({0x1100, 0x1161, 0x11A8})) == cpts({0xAC01}));
    CHECK(unorm_nfc(cpts({0x1100, 0x1161})) == cpts({0xAC00}));

    /* Compatibility: the ligature and the fullwidth letter move only under the K forms. */
    CHECK(unorm_nfd (cpts({0xFB01})) == cpts({0xFB01}));
    CHECK(unorm_nfkd(cpts({0xFB01})) == cpts({0x0066, 0x0069}));
    CHECK(unorm_nfkc(cpts({0xFF21})) == cpts({0x0041}));
    CHECK(unorm_nfc (cpts({0xFF21})) == cpts({0xFF21}));

    /* The UTF-8 wrapper, including the ASCII fast path. */
    CHECK_EQ(unorm_utf8("plain ascii", true, true), std::string("plain ascii"));
    CHECK_EQ(unorm_utf8("A\xcc\x8a", false, true), std::string("\xc3\x85"));
    CHECK_EQ(unorm_utf8("\xc3\x85", false, false), std::string("A\xcc\x8a"));
}

/* ================================================================== gguf reader */

TEST(gguf_reader) {
    GgufFile g;
    if (g.open(gguf_path()) < 0) {
        fprintf(stderr, "  SKIP gguf_reader: no file at %s\n", gguf_path());
        return;
    }
    CHECK(g.version() >= 2);
    CHECK(!g.kv().empty());
    CHECK(!g.tensors().empty());

    /* The architecture names its own metadata prefix, the one GGUF convention a reader must know:
     * `general.architecture` is "qwen3" and the dimensions are under "qwen3.*". */
    std::string arch;
    CHECK(g.get_str("general.architecture", &arch));
    CHECK(!arch.empty());

    int64_t layers = 0;
    CHECK(g.get_i64(arch + ".block_count", &layers));
    CHECK(layers > 0);

    std::vector<std::string> toks;
    CHECK(g.get_strs("tokenizer.ggml.tokens", &toks));
    CHECK(toks.size() > 100000);

    /* Every tensor's extent must be inside the file, and tensor_data() must agree. */
    for (const GgufTensor& t : g.tensors()) {
        CHECK(gguf_dtype(t.ggml_type) != nullptr);
        CHECK(t.bytes > 0);
        CHECK(g.tensor_data(t) != nullptr);
        CHECK((const uint8_t*)g.tensor_data(t) + t.bytes
              <= (const uint8_t*)g.base() + g.file_bytes());
    }

    /* Lookup by name agrees with the directory. */
    const GgufTensor& first = g.tensors()[0];
    CHECK(g.tensor(first.name) == &first);
    CHECK(g.tensor("this tensor does not exist") == nullptr);

    /* A missing key returns false and leaves the output untouched. */
    int64_t v = 12345;
    CHECK(!g.get_i64("no.such.key", &v));
    CHECK_EQ(v, (int64_t)12345);
}

TEST(gguf_rejects_non_gguf) {
    GgufFile g;
    /* Something that exists and is not a GGUF: this source file. */
    const int s = g.open(__FILE__);
    CHECK(s == RAD_E_FORMAT);
    CHECK(!g.is_open());

    GgufFile h;
    CHECK_EQ(h.open("/definitely/not/a/path/anywhere.gguf"), RAD_E_IO);
}

namespace {

/* A GGUF file byte by byte, for the shapes no producer writes on purpose. */
struct GgufBytes {
    std::vector<uint8_t> b;
    template <class T> void pod(T v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        b.insert(b.end(), p, p + sizeof v);
    }
    void str(const std::string& s) {
        pod<uint64_t>(s.size());
        b.insert(b.end(), s.begin(), s.end());
    }
    void header(uint64_t n_tensors, uint64_t n_kv) {
        for (char c : std::string("GGUF")) b.push_back((uint8_t)c);
        pod<uint32_t>(3);
        pod<uint64_t>(n_tensors);
        pod<uint64_t>(n_kv);
    }
    /* One tensor record, then the data section padded to the default alignment. */
    void tensor(const std::vector<int64_t>& ne, uint32_t type, size_t data_bytes) {
        info("w", ne, type, 0);
        pad(32);
        b.insert(b.end(), data_bytes, 0);
    }
    void kv_u32(const std::string& key, uint32_t v) {
        str(key);
        pod<uint32_t>(GGUF_U32);
        pod<uint32_t>(v);
    }
    /* A tensor record alone, for files with several tensors or a chosen offset. */
    void info(const std::string& name, const std::vector<int64_t>& ne, uint32_t type,
              uint64_t offset) {
        str(name);
        pod<uint32_t>((uint32_t)ne.size());
        for (int64_t n : ne) pod<int64_t>(n);
        pod<uint32_t>(type);
        pod<uint64_t>(offset);
    }
    void pad(size_t align) {
        while (b.size() % align) b.push_back(0);
    }
    std::string path() const { return radtest::tmp_path("bytes.gguf"); }
    bool write(const std::string& p) const {
        FILE* f = fopen(p.c_str(), "wb");
        if (!f) return false;
        const bool ok = fwrite(b.data(), 1, b.size(), f) == b.size();
        return fclose(f) == 0 && ok;
    }
    /* Unlinked once opened: `g` keeps its mapping, and the next write to the path makes a new
     * file rather than truncating one still mapped. */
    int open(GgufFile* g) const {
        const std::string p = path();
        if (!write(p)) return RAD_E_IO;
        const int st = g->open(p.c_str());
        ::unlink(p.c_str());
        return st;
    }
};

}  /* namespace */

/* Counts and dimensions in a GGUF are the file's claims, not facts. A string-array count that
 * sizes a reservation before any string is read would end the converter at 2^60; a product of
 * tensor dimensions can wrap to a zero-byte tensor that passes the bounds check while claiming to
 * be vast. Each is checked against what the file can hold. */
TEST(gguf_refuses_counts_and_sizes_its_file_cannot_hold) {
    {
        GgufBytes f;                    /* a string array of 2^60 */
        f.header(0, 1);
        f.str("tokenizer.ggml.tokens");
        f.pod<uint32_t>(GGUF_ARR); f.pod<uint32_t>(GGUF_STR); f.pod<uint64_t>(1ull << 60);
        f.str("a");
        GgufFile g;
        CHECK_EQ(f.open(&g), RAD_E_FORMAT);
    }
    {
        GgufBytes f;                    /* an array of a type GGUF does not define */
        f.header(0, 1);
        f.str("k");
        f.pod<uint32_t>(GGUF_ARR); f.pod<uint32_t>(99); f.pod<uint64_t>(1ull << 60);
        GgufFile g;
        CHECK_EQ(f.open(&g), RAD_E_FORMAT);
    }
    {
        GgufBytes f;                    /* 2^32 x 2^32: the element count wraps to zero */
        f.header(1, 0);
        f.tensor({ 1ll << 32, 1ll << 32 }, 0, 64);
        GgufFile g;
        CHECK_EQ(f.open(&g), RAD_E_FORMAT);
    }
    {
        GgufBytes f;                    /* 2^62 f32: the count fits, its bytes wrap to zero */
        f.header(1, 0);
        f.tensor({ 1ll << 62 }, 0, 64);
        GgufFile g;
        CHECK_EQ(f.open(&g), RAD_E_FORMAT);
    }
    {
        GgufBytes f;                    /* more KVs than the rest of the file can encode */
        f.header(0, 200);
        f.b.insert(f.b.end(), 100, 0);
        GgufFile g;
        CHECK_EQ(f.open(&g), RAD_E_FORMAT);
    }
    {
        /* A well-formed file with each of those shapes at a sane size opens and reads. */
        GgufBytes f;
        f.header(1, 2);
        f.str("general.name"); f.pod<uint32_t>(GGUF_STR); f.str("t");
        f.str("tokenizer.ggml.tokens");
        f.pod<uint32_t>(GGUF_ARR); f.pod<uint32_t>(GGUF_STR); f.pod<uint64_t>(2);
        f.str("a"); f.str("b");
        f.tensor({ 4, 2 }, 0, 32);
        GgufFile g;
        CHECK_OK(f.open(&g));
        std::vector<std::string> toks;
        CHECK(g.get_strs("tokenizer.ggml.tokens", &toks));
        CHECK_EQ(toks.size(), (size_t)2);
        CHECK_EQ(g.tensors().size(), (size_t)1);
        CHECK_EQ(g.tensors()[0].bytes, (uint64_t)32);
    }
}

namespace {

/* A small well-formed GGUF holding every kind of entry the reader walks: a string, a u32, a
 * bool, a string array and an f32 array as KVs, general.alignment 64, and two tensors whose data
 * (bytes 1, 2, ... in the first, 101, 102, ... in the second) ends the file. */
GgufBytes small_gguf() {
    GgufBytes f;
    f.header(2, 5);
    f.str("general.name"); f.pod<uint32_t>(GGUF_STR); f.str("t");
    f.kv_u32("general.alignment", 64);
    f.str("x.flag"); f.pod<uint32_t>(GGUF_BOOL); f.pod<uint8_t>(1);
    f.str("tokenizer.ggml.tokens");
    f.pod<uint32_t>(GGUF_ARR); f.pod<uint32_t>(GGUF_STR); f.pod<uint64_t>(2);
    f.str("a"); f.str("bc");
    f.str("tokenizer.ggml.scores");
    f.pod<uint32_t>(GGUF_ARR); f.pod<uint32_t>(GGUF_F32); f.pod<uint64_t>(2);
    f.pod<float>(0.5f); f.pod<float>(-1.0f);
    f.info("a", { 4, 2 }, 0, 0);     /* f32, 32 bytes */
    f.info("b", { 8 }, 30, 64);      /* bf16, 16 bytes */
    f.pad(64);
    for (int i = 1; i <= 32; ++i) f.b.push_back((uint8_t)i);
    f.pad(64);
    for (int i = 101; i <= 116; ++i) f.b.push_back((uint8_t)i);
    return f;
}

}  /* namespace */

/* The reader hands out pointers into a mapping of the file, so a file cut short anywhere (a
 * partial download, a full disk) must be refused, never read past its end. Every prefix of a
 * well-formed file is one; under ASan an over-read in any of the reader's walks fails here. */
TEST(every_truncation_of_a_gguf_is_refused) {
    const GgufBytes f = small_gguf();
    const std::string p = f.path();
    REQUIRE(f.write(p));
    {
        GgufFile g;
        CHECK_OK(g.open(p.c_str()));
        CHECK_EQ(g.alignment(), (uint64_t)64);
        CHECK_EQ(g.kv().size(), (size_t)5);
        const GgufTensor* b = g.tensor("b");
        REQUIRE(b != nullptr);
        CHECK_EQ(g.data_off() % 64, (uint64_t)0);
        CHECK_EQ(b->bytes, (uint64_t)16);
        const uint8_t* d = static_cast<const uint8_t*>(g.tensor_data(*b));
        REQUIRE(d != nullptr);
        CHECK_EQ(d[0], 101);
        CHECK_EQ(d[15], 116);
    }
    /* Each refusal logs a line; they are kept out of the test output. */
    std::vector<std::string> wrong;
    size_t opened = 0;
    stderr_of([&] {
        for (size_t n = f.b.size(); n-- > 0;) {
            if (::truncate(p.c_str(), (off_t)n) != 0) {
                wrong.push_back("truncate failed");
                break;
            }
            GgufFile g;
            const int st = g.open(p.c_str());
            ++opened;
            /* An empty file cannot be mapped: an I/O error, not a format error. */
            if (st != (n == 0 ? RAD_E_IO : RAD_E_FORMAT) || g.is_open())
                wrong.push_back("a " + std::to_string(n) + "-byte prefix: status " +
                                std::to_string(st));
        }
    });
    for (const std::string& w : wrong) ::radtest::fail(__FILE__, __LINE__, w);
    CHECK_EQ(opened, f.b.size());
}

/* A tensor must start on the file's alignment, as ggml's reader requires: tensor_data() hands out
 * pointers into the mapping, and a reader with aligned vector loads faults on one that is off it.
 * A misplaced offset is also what a writer that disagrees about the alignment produces, so the
 * tensor would be read from the wrong bytes. */
TEST(a_gguf_tensor_off_the_files_alignment_is_refused) {
    auto two_tensors = [](uint64_t second) {
        GgufBytes f;
        f.header(2, 0);
        f.info("a", { 2 }, 0, 0);
        f.info("b", { 2 }, 0, second);
        f.pad(32);
        f.b.insert(f.b.end(), (size_t)second + 8, 0);
        return f;
    };
    GgufFile g;
    CHECK_OK(two_tensors(32).open(&g));
    GgufFile h;
    CHECK_EQ(two_tensors(8).open(&h), RAD_E_FORMAT);
    GgufFile k;
    CHECK_EQ(two_tensors(16).open(&k), RAD_E_FORMAT);
}

/* general.alignment places the data section. A value ggml would refuse (not a power of two,
 * zero, not an integer) is refused rather than replaced by the default, which would read every
 * tensor from where no writer put it. */
TEST(a_gguf_alignment_that_is_not_a_power_of_two_is_refused) {
    /* One tensor, padded so that it lies inside the file wherever any of these alignments (the
     * default 32 included) places the data: only the declared value is wrong. */
    auto aligned = [](const std::function<void(GgufBytes&)>& alignment_kv) {
        GgufBytes f;
        f.header(1, 1);
        alignment_kv(f);
        f.info("w", { 2 }, 0, 0);
        f.pad(96);
        f.b.insert(f.b.end(), 8, 0);
        return f;
    };
    GgufFile g;
    CHECK_OK(aligned([](GgufBytes& f) { f.kv_u32("general.alignment", 32); }).open(&g));
    CHECK_EQ(g.alignment(), (uint64_t)32);
    for (uint32_t bad : { 48u, 0u, 96u }) {
        GgufFile h;
        CHECK_EQ(aligned([&](GgufBytes& f) { f.kv_u32("general.alignment", bad); }).open(&h),
                 RAD_E_FORMAT);
    }
    GgufFile k;                        /* a string where the alignment belongs */
    CHECK_EQ(aligned([](GgufBytes& f) {
                 f.str("general.alignment"); f.pod<uint32_t>(GGUF_STR); f.str("32");
             }).open(&k), RAD_E_FORMAT);
}

/* ================================================================== the vocab section */

namespace {
/* A minimal StringSink: rad-convert's dedupes, this one need not, since the test is about offsets
 * being read back as written. */
struct TestStrings : StringSink {
    std::string blob{ '\0' };   /* offset 0 is the empty string, per rad_format.h */
    rad_stroff intern(std::string_view s) override {
        if (s.empty()) return 0;
        const rad_stroff at = blob.size();
        blob.append(s);
        blob.push_back('\0');
        return at;
    }
};
}  /* namespace */

namespace {

/* `in` through the .rad's vocab section and back into `out`. */
int round_trip(const VocabBuild& in, VocabBuild* out) {
    TestStrings strings;
    std::vector<uint8_t> section;
    /* Lay the file out as rad-convert does: header, string blob, vocab section. The serialiser
     * writes absolute offsets, so the section must know where it lands before it is written, and
     * it lands 8-aligned, as the writer places it, because the reader takes the header as a typed
     * pointer into the file. */
    const uint64_t str_off = sizeof(RadFileHeader);
    RAD_TRY(vocab_serialize(in, strings, 0, section));   /* pass 1: sizes only */
    const uint64_t sec_off = (str_off + strings.blob.size() + 7) & ~(uint64_t)7;

    TestStrings strings2;
    section.clear();
    RAD_TRY(vocab_serialize(in, strings2, sec_off, section));

    std::vector<uint8_t> file(sec_off + section.size(), 0);
    RadFileHeader* fh = reinterpret_cast<RadFileHeader*>(file.data());
    fh->magic = RAD_MAGIC;
    fh->version = RAD_FORMAT_VER;
    fh->str_off = str_off;
    fh->str_bytes = strings2.blob.size();
    fh->vocab_off = sec_off;
    fh->vocab_bytes = section.size();
    std::memcpy(file.data() + str_off, strings2.blob.data(), strings2.blob.size());
    std::memcpy(file.data() + sec_off, section.data(), section.size());
    return vocab_deserialize(file.data(), file.size(),
                             reinterpret_cast<const RadVocabHeader*>(file.data() + sec_off), *out);
}

}  /* namespace */

TEST(vocab_section_round_trip) {
    VocabBuild in;
    in.kind   = RAD_TOK_BPE;
    in.tokens = { "<unk>", "a", "b", "ab", "\xc4\xa0" };
    in.types  = { RAD_TT_UNKNOWN, RAD_TT_NORMAL, RAD_TT_NORMAL, RAD_TT_NORMAL, RAD_TT_NORMAL };
    in.merges = { { 1, 2 } };
    in.bos = -1; in.eos = 2; in.eot = 2; in.unk = 0; in.pad = -1; in.sep = -1;
    in.add_bos = false;
    in.add_eos = true;
    in.ignore_merges = true;
    in.chat_template = "{{ messages }}";
    VocabStep s0; s0.kind = RAD_NORM_NFC;             in.steps.push_back(s0);
    VocabStep s1; s1.kind = RAD_PRE_SPLIT; s1.arg0 = "\\p{N}"; s1.iarg = 1; in.steps.push_back(s1);
    VocabStep s2; s2.kind = RAD_PRE_BYTELEVEL; s2.flags = 2;  in.steps.push_back(s2);
    VocabStep s3; s3.kind = RAD_DEC_BYTELEVEL;        in.steps.push_back(s3);

    VocabBuild out;
    CHECK_OK(round_trip(in, &out));

    CHECK_EQ(out.kind, in.kind);
    CHECK(out.tokens == in.tokens);
    CHECK(out.types == in.types);
    CHECK(out.merges == in.merges);
    CHECK_EQ(out.eos, in.eos);
    CHECK_EQ(out.unk, in.unk);
    CHECK_EQ(out.add_eos, in.add_eos);
    CHECK_EQ(out.ignore_merges, in.ignore_merges);
    CHECK_EQ(out.chat_template, in.chat_template);
    CHECK_EQ(out.steps.size(), in.steps.size());
    for (size_t i = 0; i < out.steps.size(); ++i) {
        CHECK_EQ(out.steps[i].kind, in.steps[i].kind);
        CHECK_EQ(out.steps[i].flags, in.steps[i].flags);
        CHECK_EQ(out.steps[i].arg0, in.steps[i].arg0);
        CHECK_EQ(out.steps[i].iarg, in.steps[i].iarg);
    }

    /* The round-tripped vocab tokenises. */
    auto v = std::make_shared<Vocab>();
    CHECK_OK(v->load(std::move(out)));
    Tokenizer t;
    CHECK_OK(t.init(v));
    std::vector<int32_t> ids;
    CHECK_OK(t.encode("ab", ids, false, false));
    CHECK_EQ(ids.size(), (size_t)1);
    CHECK_EQ(ids[0], 3);      /* the merge fired */
}

/* ================================================================== tokenizer.json */

TEST(parse_tokenizer_json_chain) {
    /* A miniature Qwen-shaped tokenizer.json: byte-level BPE with an explicit Split. The test is
     * about the chain, not the vocab: the declared steps come out in declared order with the
     * declared arguments. */
    const char* js = R"JSON({
      "added_tokens": [ {"id": 4, "content": "<|end|>", "special": true} ],
      "normalizer": { "type": "Sequence", "normalizers": [ {"type": "NFC"},
                      {"type": "Replace", "pattern": {"String": "  "}, "content": " "} ] },
      "pre_tokenizer": { "type": "Sequence", "pretokenizers": [
          {"type": "Split", "pattern": {"Regex": "\\p{N}"}, "behavior": "Isolated", "invert": false},
          {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false} ] },
      "decoder": { "type": "ByteLevel" },
      "model": { "type": "BPE", "byte_fallback": false,
                 "vocab": {"a": 0, "b": 1, "ab": 2, "Ġ": 3, "<|end|>": 4},
                 "merges": ["a b"] }
    })JSON";

    VocabBuild vb;
    CHECK_OK(parse_tokenizer_json_buf(js, "test.json", vb));
    CHECK_EQ(vb.kind, (uint32_t)RAD_TOK_BPE);
    CHECK_EQ(vb.tokens.size(), (size_t)5);
    CHECK_EQ(vb.merges.size(), (size_t)1);
    CHECK_EQ(vb.types[4], (uint8_t)RAD_TT_CONTROL);

    CHECK_EQ(vb.steps.size(), (size_t)5);
    CHECK_EQ(vb.steps[0].kind, (uint32_t)RAD_NORM_NFC);
    CHECK_EQ(vb.steps[1].kind, (uint32_t)RAD_NORM_REPLACE);
    CHECK_EQ(vb.steps[1].arg0, std::string("  "));
    CHECK_EQ(vb.steps[1].arg1, std::string(" "));
    CHECK_EQ(vb.steps[2].kind, (uint32_t)RAD_PRE_SPLIT);
    CHECK_EQ(vb.steps[2].arg0, std::string("\\p{N}"));
    CHECK_EQ(vb.steps[3].kind, (uint32_t)RAD_PRE_BYTELEVEL);
    CHECK_EQ(vb.steps[3].flags, (uint32_t)0);      /* no prefix space, no inline regex */
    CHECK_EQ(vb.steps[4].kind, (uint32_t)RAD_DEC_BYTELEVEL);
}

/* A component we cannot express must fail, not be dropped; that refusal is why tokenizer.json is
 * interpreted rather than hashed. */
TEST(parse_tokenizer_json_refuses_unknown_components) {
    const char* js = R"JSON({
      "normalizer": { "type": "StripAccents" },
      "pre_tokenizer": { "type": "ByteLevel" },
      "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] }
    })JSON";
    VocabBuild vb;
    CHECK_EQ(parse_tokenizer_json_buf(js, "test.json", vb), RAD_E_UNSUPPORTED);
}

/* A step the tokenizer does not run is refused at load, not passed over on every encode. */
TEST(a_vocab_with_an_unknown_step_does_not_load) {
    for (uint32_t kind : { 10u, 41u, 70u, 200u }) {
        VocabBuild vb;
        vb.kind = RAD_TOK_BPE;
        vb.tokens = { "a" };
        VocabStep s;
        s.kind = kind;
        vb.steps.push_back(s);
        Vocab v;
        CHECK_EQ(v.load(std::move(vb)), RAD_E_UNSUPPORTED);
    }
}

/* A unigram piece is matched at any length, as SentencePiece's trie matches it: a 300-byte piece
 * that scores best is one token, not 300 single bytes. */
TEST(a_unigram_piece_longer_than_256_bytes_is_matched) {
    const std::string longest(300, 'a');
    VocabBuild vb;
    vb.kind = RAD_TOK_UNIGRAM;
    vb.tokens = { "<unk>", "a", longest };
    vb.scores = { 0.0f, -10.0f, -1.0f };
    vb.unk = 0;
    auto v = std::make_shared<Vocab>();
    CHECK_OK(v->load(std::move(vb)));
    Tokenizer t;
    CHECK_OK(t.init(v));
    std::vector<int32_t> ids;
    CHECK_OK(t.encode(longest, ids, /*add_special=*/false, /*parse_special=*/false));
    CHECK_EQ(ids.size(), (size_t)1);
    if (!ids.empty()) CHECK_EQ(ids[0], 2);
}

TEST(parse_tokenizer_json_rejects_bpe_without_pretokenizer) {
    const char* js = R"JSON({
      "model": { "type": "BPE", "vocab": {"a": 0}, "merges": [] }
    })JSON";
    VocabBuild vb;
    CHECK_EQ(parse_tokenizer_json_buf(js, "test.json", vb), RAD_E_FORMAT);
}

/* An id is an index as soon as it is read. The token table is sized from the largest id and
 * written at each one, so a negative id writes outside it, a huge one sizes it to tens of
 * gigabytes, and a value of the wrong JSON type would throw out of the reader and end the
 * converter. Each malformed file must come back as a refusal. */
TEST(parse_tokenizer_json_refuses_ids_and_values_it_cannot_place) {
    const std::string pre =
        R"("pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true})";
    auto bpe = [&](const std::string& vocab, const std::string& extra) {
        return "{" + pre + R"(, "model": {"type": "BPE", "vocab": )" + vocab +
               R"(, "merges": []})" + extra + "}";
    };
    const std::string bad[] = {
        bpe(R"({"a": 0, "b": -1})", ""),                                  /* negative */
        bpe(R"({"a": "x"})", ""),                                         /* not a number */
        bpe(R"({"a": 0, "b": 1.5})", ""),                                 /* not an integer */
        bpe(R"({"a": 2147483647})", ""),                                  /* past the file */
        bpe(R"(["a", "b"])", ""),                                         /* not {text: id} */
        bpe(R"({"a": 0})", R"(, "added_tokens": [{"id": -1, "content": "x"}])"),
        bpe(R"({"a": 0})", R"(, "added_tokens": [{"id": 4000000000, "content": "x"}])"),
        bpe(R"({"a": 0})", R"(, "added_tokens": [{"id": 3, "content": 7}])"),
        R"({"model": {"type": "Unigram", "vocab": [["a", "x"]]}})",
        R"({"model": {"type": "Unigram", "vocab": [[5, 1.0]]}})",
        R"({"model": {"type": "Unigram", "unk_id": 9, "vocab": [["a", -1.0]]}})",
        R"({"model": {"type": "Unigram", "unk_id": -1, "vocab": [["a", -1.0]]}})",
        R"({"model": {"type": "WordPiece", "vocab": {"a": 0, "b": -7}}})",
        R"({"pre_tokenizer": {"type": "Split", "pattern": {"String": 5}}, )"
        R"("model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}})",
    };
    for (const std::string& js : bad) {
        VocabBuild vb;
        const int st = parse_tokenizer_json_buf(js, "test.json", vb);
        CHECK_EQ(st, RAD_E_FORMAT);
        if (st != RAD_E_FORMAT) fprintf(stderr, "    accepted: %s\n", js.c_str());
    }

    /* A merge that is not two strings is dropped like one naming a missing token, not fatal. */
    VocabBuild vb;
    CHECK_OK(parse_tokenizer_json_buf(
        "{" + pre + R"(, "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2},)"
                    R"( "merges": [[1, 2], ["a", "b"]]}})", "test.json", vb));
    CHECK_EQ(vb.merges.size(), (size_t)1);

    /* A sparse but sane table still loads: the gap is empty text. */
    CHECK_OK(parse_tokenizer_json_buf(bpe(R"({"a": 0, "b": 5})", ""), "test.json", vb));
    CHECK_EQ(vb.tokens.size(), (size_t)6);
    CHECK_EQ(vb.tokens[5], std::string("b"));
}

/* Unigram takes the other path through the model reader: an array vocab with scores, and Viterbi
 * rather than merges. */
TEST(unigram_viterbi) {
    const char* js = R"JSON({
      "pre_tokenizer": { "type": "Metaspace", "replacement": "▁", "prepend_scheme": "always" },
      "decoder": { "type": "Metaspace", "replacement": "▁", "prepend_scheme": "always" },
      "model": { "type": "Unigram", "unk_id": 0,
                 "vocab": [["<unk>", 0.0], ["▁", -3.0], ["▁he", -1.0],
                           ["h", -6.0], ["e", -6.0], ["▁h", -5.0], ["llo", -2.0],
                           ["l", -7.0], ["o", -7.0]] }
    })JSON";
    VocabBuild vb;
    CHECK_OK(parse_tokenizer_json_buf(js, "test.json", vb));
    CHECK_EQ(vb.kind, (uint32_t)RAD_TOK_UNIGRAM);

    auto v = std::make_shared<Vocab>();
    CHECK_OK(v->load(std::move(vb)));
    Tokenizer t;
    CHECK_OK(t.init(v));

    std::vector<int32_t> ids;
    CHECK_OK(t.encode("hello", ids, false, false));
    /* "_he" (-1) + "llo" (-2) = -3 beats "_h"(-5)+"e"(-6)+... by a wide margin. */
    CHECK_EQ(ids.size(), (size_t)2);
    CHECK_EQ(ids[0], 2);
    CHECK_EQ(ids[1], 6);
    /* HuggingFace's Metaspace decoder drops the replacement the prepend added on the first
     * token, so decode() and the streaming Detokenizer agree. */
    CHECK_EQ(t.decode(ids), std::string("hello"));
    Detokenizer d(v);
    std::string streamed;
    for (int32_t id : ids) streamed += d.push(id);
    streamed += d.flush();
    CHECK_EQ(streamed, std::string("hello"));
}

/* A Unigram vocabulary with no unk token still routes uncovered text through the unk transition,
 * whose id is then -1, an id no vocabulary has. It must be spelled as byte tokens when the chain
 * declares byte fallback and dropped otherwise, as the BPE encoder does with an unmatched piece. */
TEST(unigram_without_an_unk_token_never_emits_an_invalid_id) {
    auto encode = [](const char* js, const char* text) {
        std::vector<int32_t> ids;
        VocabBuild vb;
        if (parse_tokenizer_json_buf(js, "test.json", vb) < 0) return ids;
        auto v = std::make_shared<Vocab>();
        if (v->load(std::move(vb)) < 0) return ids;
        Tokenizer t;
        if (t.init(v) < 0) return ids;
        t.encode(text, ids, false, false);
        return ids;
    };
    /* "x" and the euro sign are in no piece. */
    const char* plain = R"JSON({
      "pre_tokenizer": { "type": "Metaspace", "replacement": "▁", "prepend_scheme": "always" },
      "model": { "type": "Unigram",
                 "vocab": [["▁", -3.0], ["▁he", -1.0], ["h", -6.0], ["e", -6.0],
                           ["l", -7.0], ["o", -7.0]] } })JSON";
    const std::vector<int32_t> dropped = encode(plain, "hex\xe2\x82\xac" "lo");
    CHECK(dropped == (std::vector<int32_t>{ 1, 4, 5 }));

    const char* bytes = R"JSON({
      "pre_tokenizer": { "type": "Metaspace", "replacement": "▁", "prepend_scheme": "always" },
      "model": { "type": "Unigram", "byte_fallback": true,
                 "vocab": [["▁", -3.0], ["▁he", -1.0], ["h", -6.0], ["e", -6.0],
                           ["l", -7.0], ["o", -7.0], ["<0x78>", -9.0], ["<0xE2>", -9.0],
                           ["<0x82>", -9.0], ["<0xAC>", -9.0]] } })JSON";
    const std::vector<int32_t> spelled = encode(bytes, "hex\xe2\x82\xac" "lo");
    CHECK(spelled == (std::vector<int32_t>{ 1, 6, 7, 8, 9, 4, 5 }));
}

/* A special id is emitted as is, so one naming no token (a GGUF or container can carry one;
 * nothing else checks) is treated as absent. */
TEST(vocab_treats_an_out_of_range_special_id_as_absent) {
    VocabBuild vb;
    vb.kind   = RAD_TOK_BPE;
    vb.tokens = { "a", "b", "</s>" };
    vb.eos = 2; vb.unk = 7; vb.bos = -5; vb.pad = 3;
    vb.add_eos = true;
    auto v = std::make_shared<Vocab>();
    CHECK_OK(v->load(std::move(vb)));
    CHECK_EQ(v->eos(), 2);
    CHECK_EQ(v->unk(), -1);
    CHECK_EQ(v->bos(), -1);
    CHECK_EQ(v->pad(), -1);
}

/* The checkpoint's end-of-generation list (generation_config.json's eos_token_id, carried as
 * `generation.eos_token_id`) ends a generation on every id it names; Gemma 4 names
 * <|tool_response> there, where the model stops after its calls. An id naming no token is
 * ignored. */
TEST(vocab_ends_a_generation_on_every_id_the_checkpoint_names) {
    VocabBuild vb;
    vb.kind   = RAD_TOK_BPE;
    vb.tokens = { "a", "<eos>", "<|tool_response>", "<|tool_call>" };
    vb.types  = { RAD_TT_NORMAL, RAD_TT_CONTROL, RAD_TT_CONTROL, RAD_TT_CONTROL };
    vb.eos = 1;
    vb.eos_ids = { 1, 2, 99, -3 };
    auto v = std::make_shared<Vocab>();
    CHECK_OK(v->load(std::move(vb)));
    CHECK(v->is_eog(1));
    CHECK(v->is_eog(2));
    CHECK(!v->is_eog(3));
    CHECK(!v->is_eog(0));
    CHECK_EQ(v->eog().size(), (size_t)2);
}

/* For a container that declares one eos and no list, the loader completes the set from known
 * names: Gemma 4's turn end and tool-response opener end a generation; `</s>` does not where the
 * vocabulary has its own turn marker, and does where it has none. */
TEST(vocab_completes_the_end_set_from_known_control_names_and_keeps_a_turn_vocabularys_s_tag_as_text) {
    VocabBuild vb;
    vb.kind   = RAD_TOK_BPE;
    vb.tokens = { "a", "<eos>", "<turn|>", "<|tool_response>", "</s>" };
    vb.types  = { RAD_TT_NORMAL, RAD_TT_CONTROL, RAD_TT_CONTROL, RAD_TT_CONTROL, RAD_TT_CONTROL };
    vb.eos = 1;
    auto v = std::make_shared<Vocab>();
    CHECK_OK(v->load(std::move(vb)));
    CHECK(v->is_eog(1));
    CHECK(v->is_eog(2));
    CHECK(v->is_eog(3));
    CHECK(!v->is_eog(4));

    VocabBuild plain;
    plain.kind   = RAD_TOK_BPE;
    plain.tokens = { "a", "<eos>", "</s>" };
    plain.types  = { RAD_TT_NORMAL, RAD_TT_CONTROL, RAD_TT_CONTROL };
    plain.eos = 1;
    auto w = std::make_shared<Vocab>();
    CHECK_OK(w->load(std::move(plain)));
    CHECK(w->is_eog(2));
}

/* ================================================================== chain steps */

TEST(step_chain_units) {
    /* A vocab of single-character tokens, so the test observes the split rather than the
     * merges. Each case installs one step and counts the tokens that come out. */
    auto make = [](std::vector<VocabStep> steps) {
        VocabBuild vb;
        vb.kind = RAD_TOK_BPE;
        for (int c = 0; c < 128; ++c) vb.tokens.push_back(std::string(1, (char)c));
        vb.tokens.push_back("ab");
        vb.merges = { { 'a', 'b' } };    /* so "a" + "b" can reach id 128 */
        vb.steps = std::move(steps);
        auto v = std::make_shared<Vocab>();
        v->load(std::move(vb));
        return v;
    };

    /* Digits, individual: "a12b" -> "a", "1", "2", "b". */
    {
        VocabStep d; d.kind = RAD_PRE_DIGITS; d.flags = 1;
        Tokenizer t;
        CHECK_OK(t.init(make({ d })));
        std::vector<int32_t> ids;
        CHECK_OK(t.encode("a12b", ids, false, false));
        CHECK_EQ(ids.size(), (size_t)4);
    }
    /* Punctuation, isolated: "a,b" -> "a", ",", "b" and the merge of "ab" cannot fire. */
    {
        VocabStep p; p.kind = RAD_PRE_PUNCTUATION; p.iarg = 1;
        Tokenizer t;
        CHECK_OK(t.init(make({ p })));
        std::vector<int32_t> ids;
        CHECK_OK(t.encode("a,b", ids, false, false));
        CHECK_EQ(ids.size(), (size_t)3);
    }
    /* Lowercase normaliser. */
    {
        VocabStep n; n.kind = RAD_NORM_LOWERCASE;
        VocabStep w; w.kind = RAD_PRE_WHITESPACE;
        Tokenizer t;
        CHECK_OK(t.init(make({ n, w })));
        std::vector<int32_t> ids;
        CHECK_OK(t.encode("AB", ids, false, false));
        CHECK_EQ(ids.size(), (size_t)1);      /* "ab" is one token */
        CHECK_EQ(ids[0], 128);
    }
    /* Prepend + Strip. */
    {
        VocabStep pre; pre.kind = RAD_NORM_PREPEND; pre.arg0 = "a";
        VocabStep w;   w.kind = RAD_PRE_WHITESPACE;
        Tokenizer t;
        CHECK_OK(t.init(make({ pre, w })));
        std::vector<int32_t> ids;
        CHECK_OK(t.encode("b", ids, false, false));
        CHECK_EQ(ids.size(), (size_t)1);
        CHECK_EQ(ids[0], 128);                /* "a" + "b" merged into "ab" */
    }
}

/* ================================================================== utf8_sanitize */

/* A byte-fallback vocab can emit a byte no continuation completes, and the completion goes into a
 * JSON string, where one bad byte costs the client the whole response. These are the WHATWG
 * cases: one U+FFFD per maximal subpart, well-formed input untouched. */
TEST(utf8_sanitize_repairs_only_what_is_broken) {
    const std::string fffd = "\xEF\xBF\xBD";

    /* Untouched: ASCII, and every well-formed length. */
    CHECK_EQ(utf8_sanitize("hello"), std::string("hello"));
    CHECK_EQ(utf8_sanitize("caf\xC3\xA9 \xE6\x97\xA5 \xF0\x9F\x98\x80"),
             std::string("caf\xC3\xA9 \xE6\x97\xA5 \xF0\x9F\x98\x80"));
    CHECK_EQ(utf8_sanitize(""), std::string(""));

    /* 0xC1 leads an overlong encoding, so it is never legal however it continues; one in a JSON
     * string is a 500 for the client. */
    CHECK_EQ(utf8_sanitize("\xC1"), fffd);
    CHECK_EQ(utf8_sanitize("a\xC1\x61z"), "a" + fffd + "az");

    /* A lone continuation byte, and a lead byte that is simply not a lead. */
    CHECK_EQ(utf8_sanitize("\x80"), fffd);
    CHECK_EQ(utf8_sanitize("\xFF"), fffd);

    /* Truncated three-byte character: one replacement for the whole subpart, not two. */
    CHECK_EQ(utf8_sanitize("\xE6\x97"), fffd);
    CHECK_EQ(utf8_sanitize("\xE6\x97 "), fffd + " ");

    /* Overlong, surrogate and out-of-range encodings are rejected at the second byte, so the
     * lead is one subpart and the rest decodes on its own. */
    CHECK_EQ(utf8_sanitize("\xE0\x80\xAF"), fffd + fffd + fffd);  /* overlong '/' */
    CHECK_EQ(utf8_sanitize("\xED\xA0\x80"), fffd + fffd + fffd);  /* U+D800 */
    CHECK_EQ(utf8_sanitize("\xF4\x90\x80\x80"), fffd + fffd + fffd + fffd);  /* > U+10FFFF */
}

/* ================================================================== the ggml dtype map
 *
 * The block quants are deliberately absent, and that absence is under test. Reinterpreting a
 * q4_K block as RAD_I4 would give a tensor that loads, computes and is wrong, so rad-convert must
 * be told "no equivalent" and dequantise or refuse.
 */
TEST(the_ggml_dtype_map_carries_only_the_types_that_mean_the_same_thing) {
    CHECK_EQ(gguf_to_rad_dtype(0),  (uint32_t)RAD_F32);
    CHECK_EQ(gguf_to_rad_dtype(1),  (uint32_t)RAD_F16);
    CHECK_EQ(gguf_to_rad_dtype(24), (uint32_t)RAD_I8);
    CHECK_EQ(gguf_to_rad_dtype(25), (uint32_t)RAD_I16);
    CHECK_EQ(gguf_to_rad_dtype(26), (uint32_t)RAD_I32);
    CHECK_EQ(gguf_to_rad_dtype(27), (uint32_t)RAD_I64);
    CHECK_EQ(gguf_to_rad_dtype(30), (uint32_t)RAD_BF16);

    /* q4_0, q8_0, q4_K, q6_K and the iq family are block layouts with their own scales; none has
       a RAD_* equivalent. */
    for (uint32_t t : { 2u, 3u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u,
                        16u, 17u, 18u, 19u, 20u, 21u, 22u, 23u, 28u, 29u })
        CHECK_EQ(gguf_to_rad_dtype(t), (uint32_t)RAD_DT_INVALID);

    /* A type code from a newer ggml than this build knows is invalid, not a guess. */
    CHECK_EQ(gguf_to_rad_dtype(9999), (uint32_t)RAD_DT_INVALID);

    /* The table beside it is bounds-checked: a truncated or hostile GGUF names its own type
       codes, and a loader segfault is not a diagnosis. */
    CHECK(gguf_dtype(0) != nullptr);
    CHECK(gguf_dtype(9999) == nullptr);
    const GgufDtype* f32 = gguf_dtype(0);
    if (f32) {
        CHECK(f32->name && *f32->name);
        CHECK(f32->blck > 0);
        CHECK(f32->block_bytes > 0);
    }
    /* Every type the table carries describes a non-degenerate block, or the element-count
       arithmetic that reads it divides by zero. */
    for (uint32_t t = 0; t < 64; ++t) {
        const GgufDtype* d = gguf_dtype(t);
        if (!d || !d->name || !*d->name) continue;
        CHECK(d->blck > 0);
        CHECK(d->block_bytes > 0);
    }
}

/* ================================================================== the step chain's names
 *
 * What a step prints is the contract rad-convert writes against. A chain this build cannot
 * express is refused by name (spec §12), so a step of unknown kind must say exactly that, with
 * its number, rather than fall through to a neighbouring step's description.
 */
TEST(every_step_kind_describes_itself_and_an_unknown_one_names_its_number) {
    VocabStep s;
    s.kind = RAD_NORM_NFC;
    CHECK_EQ(step_describe(s), std::string("NFC"));
    s.kind = RAD_NORM_NFKD;
    CHECK_EQ(step_describe(s), std::string("NFKD"));

    /* The arguments are in the description because they distinguish two declarations of the
       same step: a Replace chain is several rows differing only in their strings. */
    s.kind = RAD_NORM_REPLACE;
    s.flags = 0;
    s.arg0 = " ";
    s.arg1 = "_";
    const std::string rep = step_describe(s);
    CHECK(rep.find("Replace") != std::string::npos);
    CHECK(rep.find("literal") != std::string::npos);
    CHECK(rep.find("_") != std::string::npos);
    s.flags = 1;
    CHECK(step_describe(s).find("regex") != std::string::npos);

    s = VocabStep{};
    s.kind = RAD_PRE_BYTELEVEL;
    s.flags = 3;
    const std::string bl = step_describe(s);
    CHECK(bl.find("ByteLevel") != std::string::npos);
    CHECK(bl.find("add_prefix_space=1") != std::string::npos);
    CHECK(bl.find("use_regex=1") != std::string::npos);

    /* A decoder step says so. The three chains are one list in declared order, so a reader must
       be able to tell where the list turns over. */
    s = VocabStep{};
    s.kind = RAD_DEC_BYTELEVEL;
    CHECK(step_describe(s).rfind("decode", 0) == 0);
    s.kind = RAD_DEC_FUSE;
    CHECK(step_describe(s).rfind("decode", 0) == 0);

    /* An unexpressible step names its number, which makes the refusal actionable. */
    s = VocabStep{};
    s.kind = 9999;
    const std::string unk = step_describe(s);
    CHECK(unk.find("unknown") != std::string::npos);
    CHECK(unk.find("9999") != std::string::npos);
}

/* The kind range decides which of the three chains a row belongs to; the three are stored as one
 * list in declared order so no reader has to reassemble them. A boundary off by one would put a
 * pre-tokeniser in the normaliser chain, silently changing token boundaries. */
TEST(the_kind_ranges_partition_the_three_chains) {
    CHECK(step_is_norm(RAD_NORM_NFC));
    CHECK(step_is_norm(RAD_NORM_PREPEND));
    CHECK(!step_is_norm(0));                       /* 0 is the unset sentinel, not a normaliser */
    CHECK(!step_is_norm(RAD_PRE_BYTELEVEL));
    CHECK(!step_is_norm(RAD_DEC_BYTELEVEL));

    /* Every kind is in at most one chain. */
    for (uint32_t k : { (uint32_t)RAD_NORM_NFC, (uint32_t)RAD_NORM_LOWERCASE,
                        (uint32_t)RAD_PRE_BYTELEVEL, (uint32_t)RAD_PRE_METASPACE,
                        (uint32_t)RAD_DEC_BYTELEVEL, (uint32_t)RAD_DEC_FUSE }) {
        const int in = (step_is_norm(k) ? 1 : 0) + (step_is_pre(k) ? 1 : 0) +
                       (step_is_dec(k) ? 1 : 0);
        CHECK_EQ(in, 1);
    }
}

/* ================================================================== the normalisation tables
 *
 * The four public forms are covered above; these are the two primitives under them, pinned
 * separately because their tables are generated. A table regenerated from a newer Unicode can
 * change token boundaries for one script only, which no round-trip test over ASCII would notice.
 */
TEST(the_combining_class_table_holds_the_values_canonical_ordering_depends_on) {
    /* Starters are class 0; the canonical ordering algorithm only reorders non-starters. */
    CHECK_EQ(unorm_ccc('a'), 0u);
    CHECK_EQ(unorm_ccc(0x00E9), 0u);               /* precomposed e-acute is itself a starter */
    /* Combining acute is 230 (above), cedilla 202 (below); their order is what makes
       "e + cedilla + acute" and "e + acute + cedilla" normalise to the same string. */
    CHECK_EQ(unorm_ccc(0x0301), 230u);
    CHECK_EQ(unorm_ccc(0x0327), 202u);
    CHECK(unorm_ccc(0x0327) < unorm_ccc(0x0301));
    /* A codepoint past the table's coverage is a starter, not a read past the table. */
    CHECK_EQ(unorm_ccc(0x10FFFF), 0u);
    CHECK_EQ(unorm_ccc(0), 0u);
}

TEST(decompose_and_compose_are_inverses_over_the_canonical_forms) {
    /* e-acute decomposes to e + combining acute and composes back. */
    std::vector<uint32_t> v = { 0x00E9 };
    unorm_decompose(v, /*compat=*/false);
    CHECK_EQ(v.size(), 2u);
    CHECK_EQ(v[0], (uint32_t)'e');
    CHECK_EQ(v[1], 0x0301u);
    unorm_compose(v);
    CHECK_EQ(v.size(), 1u);
    CHECK_EQ(v[0], 0x00E9u);

    /* Canonical ordering is part of decomposition: the two accents come back in ccc order
       whichever order they went in, which is what makes NFD a normal form. */
    std::vector<uint32_t> a = { 'e', 0x0301, 0x0327 };
    std::vector<uint32_t> b = { 'e', 0x0327, 0x0301 };
    unorm_decompose(a, false);
    unorm_decompose(b, false);
    CHECK(a == b);
    CHECK_EQ(a[1], 0x0327u);                       /* 202 before 230 */

    /* Compatibility decomposition is not canonical decomposition; a K form behaving like the plain
       one would silently stop folding the ligatures and full-width forms NFKC promises. */
    std::vector<uint32_t> lig = { 0xFB01 };        /* the fi ligature */
    unorm_decompose(lig, /*compat=*/false);
    CHECK_EQ(lig.size(), 1u);                      /* it has no canonical decomposition */
    std::vector<uint32_t> ligk = { 0xFB01 };
    unorm_decompose(ligk, /*compat=*/true);
    CHECK_EQ(ligk.size(), 2u);
    CHECK_EQ(ligk[0], (uint32_t)'f');
    CHECK_EQ(ligk[1], (uint32_t)'i');
    /* Composition does not put it back: NFKC is lossy by design. */
    unorm_compose(ligk);
    CHECK_EQ(ligk.size(), 2u);

    /* Plain ASCII is untouched by either, the path almost every token takes. */
    std::vector<uint32_t> ascii = { 'h', 'e', 'l', 'l', 'o' };
    const std::vector<uint32_t> before = ascii;
    unorm_decompose(ascii, false);
    unorm_compose(ascii);
    CHECK(ascii == before);

    /* Empty input gives empty output, not a read of element zero. */
    std::vector<uint32_t> empty;
    unorm_decompose(empty, true);
    unorm_compose(empty);
    CHECK_EQ(empty.size(), 0u);

    /* A lone combining mark has no starter to attach to and must survive, not be dropped or
       folded into whatever precedes it in the buffer. */
    std::vector<uint32_t> lone = { 0x0301 };
    unorm_compose(lone);
    CHECK_EQ(lone.size(), 1u);
    CHECK_EQ(lone[0], 0x0301u);
}

namespace {

/* BEGIN generated by mk_tokenizer_goldens.py (tokenizers 0.23.3) */

struct DecodeGolden { const char* fixture; std::vector<int32_t> ids; const char* text; };
struct PretokGolden { const char* name; const char* json; std::vector<int32_t> ids; };
struct EncodeGolden { const char* text; std::vector<int32_t> ids; std::vector<int32_t> with_special; };

const std::vector<std::pair<const char*, const char*>>& decode_fixtures() {
    static const std::vector<std::pair<const char*, const char*>> v = {
        { "llama_like", R"JSON({"added_tokens": [{"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}], "normalizer": null, "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false}, "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 1, "stop": 0}]}, "model": {"type": "BPE", "byte_fallback": true, "unk_token": "<unk>", "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "<0xE2>": 3, "<0x82>": 4, "<0xAC>": 5, "▁": 6, "▁▁": 7, "▁Hello": 8, "▁world": 9, "Hello": 10, "a▁b": 11, "x": 12, "▁x": 13, "!": 14}, "merges": []}})JSON" },
        { "metaspace_always", R"JSON({"added_tokens": [{"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}], "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}, "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}, "model": {"type": "BPE", "byte_fallback": true, "unk_token": "<unk>", "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "<0xE2>": 3, "<0x82>": 4, "<0xAC>": 5, "▁": 6, "▁▁": 7, "▁Hello": 8, "▁world": 9, "Hello": 10, "a▁b": 11, "x": 12, "▁x": 13, "!": 14}, "merges": []}})JSON" },
        { "metaspace_first", R"JSON({"added_tokens": [{"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}], "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": true}, "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": true}, "model": {"type": "BPE", "byte_fallback": true, "unk_token": "<unk>", "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "<0xE2>": 3, "<0x82>": 4, "<0xAC>": 5, "▁": 6, "▁▁": 7, "▁Hello": 8, "▁world": 9, "Hello": 10, "a▁b": 11, "x": 12, "▁x": 13, "!": 14}, "merges": []}})JSON" },
        { "metaspace_never", R"JSON({"added_tokens": [{"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}], "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": true}, "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": true}, "model": {"type": "BPE", "byte_fallback": true, "unk_token": "<unk>", "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "<0xE2>": 3, "<0x82>": 4, "<0xAC>": 5, "▁": 6, "▁▁": 7, "▁Hello": 8, "▁world": 9, "Hello": 10, "a▁b": 11, "x": 12, "▁x": 13, "!": 14}, "merges": []}})JSON" },
        { "wordpiece_cleanup", R"JSON({"added_tokens": [{"id": 0, "content": "[PAD]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 1, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 2, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 3, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}], "pre_tokenizer": {"type": "BertPreTokenizer"}, "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "hello": 4, "##world": 5, "!": 6, "a": 7, "'": 8, "s": 9, "do": 10, "n't": 11, ",": 12, "x": 13, "##y": 14, ".": 15, "##s": 16}}})JSON" },
        { "wordpiece_raw", R"JSON({"added_tokens": [{"id": 0, "content": "[PAD]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 1, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 2, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 3, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}], "pre_tokenizer": {"type": "BertPreTokenizer"}, "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "hello": 4, "##world": 5, "!": 6, "a": 7, "'": 8, "s": 9, "do": 10, "n't": 11, ",": 12, "x": 13, "##y": 14, ".": 15, "##s": 16}}})JSON" },
        { "bpe_decoder", R"JSON({"pre_tokenizer": {"type": "WhitespaceSplit"}, "decoder": {"type": "BPEDecoder", "suffix": "</w>"}, "model": {"type": "BPE", "byte_fallback": false, "unk_token": "<unk>", "vocab": {"hel": 0, "lo</w>": 1, "wor": 2, "ld</w>": 3, "hi</w>": 4, "x": 5, "<unk>": 6}, "merges": []}})JSON" },
        { "bytelevel", R"JSON({"pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}, "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}, "model": {"type": "BPE", "byte_fallback": false, "unk_token": null, "vocab": {"Ġhello": 0, "Ġworld": 1, "Ċ": 2, "x": 3, "Ã": 4, "©": 5}, "merges": []}})JSON" },
    };
    return v;
}

const std::vector<DecodeGolden>& decode_goldens() {
    static const std::vector<DecodeGolden> v = {
        { "llama_like", { 8, 9 }, "Hello world" },
        { "llama_like", { 1, 8, 9 }, "Hello world" },
        { "llama_like", { 6, 8 }, " Hello" },
        { "llama_like", { 7, 8 }, "  Hello" },
        { "llama_like", { 10, 9, 14 }, "Hello world!" },
        { "llama_like", { 3, 4, 5, 13 }, "\xe2\x82\xac x" },
        { "llama_like", { 9, 3, 4, 5 }, "world\xe2\x82\xac" },
        { "llama_like", { 11, 9 }, "a b world" },
        { "llama_like", { 1, 6, 9 }, " world" },
        { "metaspace_always", { 8, 9 }, "Hello world" },
        { "metaspace_always", { 7, 12 }, "x" },
        { "metaspace_always", { 11, 13 }, "ab x" },
        { "metaspace_always", { 10, 9 }, "Hello world" },
        { "metaspace_always", { 1, 8 }, "Hello" },
        { "metaspace_first", { 8, 9 }, "Hello world" },
        { "metaspace_first", { 7, 12 }, "x" },
        { "metaspace_first", { 11, 13 }, "ab x" },
        { "metaspace_first", { 10, 9 }, "Hello world" },
        { "metaspace_never", { 8, 9 }, " Hello world" },
        { "metaspace_never", { 7, 12 }, "  x" },
        { "metaspace_never", { 11, 13 }, "a b x" },
        { "metaspace_never", { 10, 9 }, "Hello world" },
        { "wordpiece_cleanup", { 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, "helloworld! a ' s don't, xy." },
        { "wordpiece_cleanup", { 5, 4 }, "##world hello" },
        { "wordpiece_cleanup", { 2, 4, 16, 3 }, "hellos" },
        { "wordpiece_cleanup", { 7, 8, 9 }, "a ' s" },
        { "wordpiece_raw", { 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, "helloworld ! a ' s do n't , xy ." },
        { "wordpiece_raw", { 5, 4 }, "##world hello" },
        { "bpe_decoder", { 0, 1, 2, 3 }, "hello world" },
        { "bpe_decoder", { 4 }, "hi" },
        { "bpe_decoder", { 4, 4 }, "hi hi" },
        { "bpe_decoder", { 0, 1, 5 }, "hello x" },
        { "bytelevel", { 0, 1, 2 }, " hello world\n" },
        { "bytelevel", { 3, 0 }, "x hello" },
        { "bytelevel", { 4, 5 }, "\xc3\xa9" },
    };
    return v;
}

const char* const kPretokText = "hello, world!! a,,b \xe2\x82\xac" "5 \xe2\x86\x92 x\xcc\x81y \xe2\x80\xbfz 12345 don't \xc2\xa9x_y \xd9\xa3\xd9\xa4 \xe2\x85\xab \xc2\xbd x\xc2\xb2 \xc2\xa0n\xe3\x80\x80m \xc2\xbfq?";

const std::vector<PretokGolden>& pretok_goldens() {
    static const std::vector<PretokGolden> v = {
        /* hello | , | world | !! | a | ,, | b | \u20ac | 5 | \u2192 | x\u0301y | \u203fz | 12345 | don | ' | t | \xa9 | x_y | \u0663\u0664 | \u216b | \xbd | x | \xb2 | n | m | \xbf | q | ? */
        { "Whitespace", R"JSON({"pre_tokenizer": {"type": "Whitespace"}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "hello": 1, ",": 2, "world": 3, "!!": 4, "a": 5, ",,": 6, "b": 7, "€": 8, "5": 9, "→": 10, "x́y": 11, "‿z": 12, "12345": 13, "don": 14, "'": 15, "t": 16, "©": 17, "x_y": 18, "٣٤": 19, "Ⅻ": 20, "½": 21, "x": 22, "²": 23, "n": 24, "m": 25, "¿": 26, "q": 27, "?": 28}}})JSON", { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28 } },
        /* hello, | world!! | a,,b | \u20ac5 | \u2192 | x\u0301y | \u203fz | 12345 | don't | \xa9x_y | \u0663\u0664 | \u216b | \xbd | x\xb2 | n | m | \xbfq? */
        { "WhitespaceSplit", R"JSON({"pre_tokenizer": {"type": "WhitespaceSplit"}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "hello,": 1, "world!!": 2, "a,,b": 3, "€5": 4, "→": 5, "x́y": 6, "‿z": 7, "12345": 8, "don't": 9, "©x_y": 10, "٣٤": 11, "Ⅻ": 12, "½": 13, "x²": 14, "n": 15, "m": 16, "¿q?": 17}}})JSON", { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 } },
        /* hello | , | world | ! | ! | a | , | , | b | \u20ac5 | \u2192 | x\u0301y | \u203f | z | 12345 | don | ' | t | \xa9x | _ | y | \u0663\u0664 | \u216b | \xbd | x\xb2 | n | m | \xbf | q | ? */
        { "BertPreTokenizer", R"JSON({"pre_tokenizer": {"type": "BertPreTokenizer"}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "hello": 1, ",": 2, "world": 3, "!": 4, "a": 5, "b": 6, "€5": 7, "→": 8, "x́y": 9, "‿": 10, "z": 11, "12345": 12, "don": 13, "'": 14, "t": 15, "©x": 16, "_": 17, "y": 18, "٣٤": 19, "Ⅻ": 20, "½": 21, "x²": 22, "n": 23, "m": 24, "¿": 25, "q": 26, "?": 27}}})JSON", { 1, 2, 3, 4, 4, 5, 2, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 } },
        /* hello |  world |  a | b \u20ac5 \u2192 x\u0301y  | z 12345 don | t \xa9x | y \u0663\u0664 \u216b \xbd x\xb2 \xa0n\u3000m  | q */
        { "Punctuation_Removed", R"JSON({"pre_tokenizer": {"type": "Punctuation", "behavior": "Removed"}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "hello": 1, " world": 2, " a": 3, "b €5 → x́y ": 4, "z 12345 don": 5, "t ©x": 6, "y ٣٤ Ⅻ ½ x²  n　m ": 7, "q": 8}}})JSON", { 1, 2, 3, 4, 5, 6, 7, 8 } },
        /* hello | , |  world | ! | ! |  a | , | , | b \u20ac5 \u2192 x\u0301y  | \u203f | z 12345 don | ' | t \xa9x | _ | y \u0663\u0664 \u216b \xbd x\xb2 \xa0n\u3000m  | \xbf | q | ? */
        { "Punctuation_Isolated", R"JSON({"pre_tokenizer": {"type": "Punctuation", "behavior": "Isolated"}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "hello": 1, ",": 2, " world": 3, "!": 4, " a": 5, "b €5 → x́y ": 6, "‿": 7, "z 12345 don": 8, "'": 9, "t ©x": 10, "_": 11, "y ٣٤ Ⅻ ½ x²  n　m ": 12, "¿": 13, "q": 14, "?": 15}}})JSON", { 1, 2, 3, 4, 4, 5, 2, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 } },
        /* hello, |  world! | ! |  a, | , | b \u20ac5 \u2192 x\u0301y \u203f | z 12345 don' | t \xa9x_ | y \u0663\u0664 \u216b \xbd x\xb2 \xa0n\u3000m \xbf | q? */
        { "Punctuation_MergedWithPrevious", R"JSON({"pre_tokenizer": {"type": "Punctuation", "behavior": "MergedWithPrevious"}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "hello,": 1, " world!": 2, "!": 3, " a,": 4, ",": 5, "b €5 → x́y ‿": 6, "z 12345 don'": 7, "t ©x_": 8, "y ٣٤ Ⅻ ½ x²  n　m ¿": 9, "q?": 10}}})JSON", { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 } },
        /* hello | , world | ! | ! a | , | ,b \u20ac5 \u2192 x\u0301y  | \u203fz 12345 don | 't \xa9x | _y \u0663\u0664 \u216b \xbd x\xb2 \xa0n\u3000m  | \xbfq | ? */
        { "Punctuation_MergedWithNext", R"JSON({"pre_tokenizer": {"type": "Punctuation", "behavior": "MergedWithNext"}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "hello": 1, ", world": 2, "!": 3, "! a": 4, ",": 5, ",b €5 → x́y ": 6, "‿z 12345 don": 7, "'t ©x": 8, "_y ٣٤ Ⅻ ½ x²  n　m ": 9, "¿q": 10, "?": 11}}})JSON", { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 } },
        /* hello | , |  world | !! |  a | ,, | b \u20ac5 \u2192 x\u0301y  | \u203f | z 12345 don | ' | t \xa9x | _ | y \u0663\u0664 \u216b \xbd x\xb2 \xa0n\u3000m  | \xbf | q | ? */
        { "Punctuation_Contiguous", R"JSON({"pre_tokenizer": {"type": "Punctuation", "behavior": "Contiguous"}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "hello": 1, ",": 2, " world": 3, "!!": 4, " a": 5, ",,": 6, "b €5 → x́y ": 7, "‿": 8, "z 12345 don": 9, "'": 10, "t ©x": 11, "_": 12, "y ٣٤ Ⅻ ½ x²  n　m ": 13, "¿": 14, "q": 15, "?": 16}}})JSON", { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 } },
        /* hello, world!! a,,b \u20ac | 5 |  \u2192 x\u0301y \u203fz  | 1 | 2 | 3 | 4 | 5 |  don't \xa9x_y  | \u0663 | \u0664 |   | \u216b |   | \xbd |  x | \xb2 |  \xa0n\u3000m \xbfq? */
        { "Digits_individual", R"JSON({"pre_tokenizer": {"type": "Digits", "individual_digits": true}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "hello, world!! a,,b €": 1, "5": 2, " → x́y ‿z ": 3, "1": 4, "2": 5, "3": 6, "4": 7, " don't ©x_y ": 8, "٣": 9, "٤": 10, " ": 11, "Ⅻ": 12, "½": 13, " x": 14, "²": 15, "  n　m ¿q?": 16}}})JSON", { 1, 2, 3, 4, 5, 6, 7, 2, 8, 9, 10, 11, 12, 11, 13, 14, 15, 16 } },
        /* hello, world!! a,,b \u20ac | 5 |  \u2192 x\u0301y \u203fz  | 12345 |  don't \xa9x_y  | \u0663\u0664 |   | \u216b |   | \xbd |  x | \xb2 |  \xa0n\u3000m \xbfq? */
        { "Digits_runs", R"JSON({"pre_tokenizer": {"type": "Digits", "individual_digits": false}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "hello, world!! a,,b €": 1, "5": 2, " → x́y ‿z ": 3, "12345": 4, " don't ©x_y ": 5, "٣٤": 6, " ": 7, "Ⅻ": 8, "½": 9, " x": 10, "²": 11, "  n　m ¿q?": 12}}})JSON", { 1, 2, 3, 4, 5, 6, 7, 8, 7, 9, 10, 11, 12 } },
        /* \u2581hello, | \u2581world!! | \u2581a,,b | \u2581\u20ac5 | \u2581\u2192 | \u2581x\u0301y | \u2581\u203fz | \u258112345 | \u2581don't | \u2581\xa9x_y | \u2581\u0663\u0664 | \u2581\u216b | \u2581\xbd | \u2581x\xb2 | \u2581\xa0n\u3000m | \u2581\xbfq? */
        { "Metaspace_split_always", R"JSON({"pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "▁hello,": 1, "▁world!!": 2, "▁a,,b": 3, "▁€5": 4, "▁→": 5, "▁x́y": 6, "▁‿z": 7, "▁12345": 8, "▁don't": 9, "▁©x_y": 10, "▁٣٤": 11, "▁Ⅻ": 12, "▁½": 13, "▁x²": 14, "▁ n　m": 15, "▁¿q?": 16}}})JSON", { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 } },
        /* hello, | \u2581world!! | \u2581a,,b | \u2581\u20ac5 | \u2581\u2192 | \u2581x\u0301y | \u2581\u203fz | \u258112345 | \u2581don't | \u2581\xa9x_y | \u2581\u0663\u0664 | \u2581\u216b | \u2581\xbd | \u2581x\xb2 | \u2581\xa0n\u3000m | \u2581\xbfq? */
        { "Metaspace_split_never", R"JSON({"pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": true}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "hello,": 1, "▁world!!": 2, "▁a,,b": 3, "▁€5": 4, "▁→": 5, "▁x́y": 6, "▁‿z": 7, "▁12345": 8, "▁don't": 9, "▁©x_y": 10, "▁٣٤": 11, "▁Ⅻ": 12, "▁½": 13, "▁x²": 14, "▁ n　m": 15, "▁¿q?": 16}}})JSON", { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 } },
        /* \u2581hello,\u2581world!!\u2581a,,b\u2581\u20ac5\u2581\u2192\u2581x\u0301y\u2581\u203fz\u258112345\u2581don't\u2581\xa9x_y\u2581\u0663\u0664\u2581\u216b\u2581\xbd\u2581x\xb2\u2581\xa0n\u3000m\u2581\xbfq? */
        { "Metaspace_whole_first", R"JSON({"pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "▁hello,▁world!!▁a,,b▁€5▁→▁x́y▁‿z▁12345▁don't▁©x_y▁٣٤▁Ⅻ▁½▁x²▁ n　m▁¿q?": 1}}})JSON", { 1 } },
        /* hello | , | world | ! | ! | a | , | , | b | \u20ac | 5 | \u2192 | x\u0301y | \u203f | z | 1 | 2 | 3 | 4 | 5 | don | ' | t | \xa9x | _ | y | \u0663 | \u0664 | \u216b | \xbd | x | \xb2 | n | m | \xbf | q | ? */
        { "Sequence_Whitespace_Digits", R"JSON({"pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"}, {"type": "Digits", "individual_digits": true}, {"type": "Punctuation", "behavior": "Isolated"}]}, "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "hello": 1, ",": 2, "world": 3, "!": 4, "a": 5, "b": 6, "€": 7, "5": 8, "→": 9, "x́y": 10, "‿": 11, "z": 12, "1": 13, "2": 14, "3": 15, "4": 16, "don": 17, "'": 18, "t": 19, "©x": 20, "_": 21, "y": 22, "٣": 23, "٤": 24, "Ⅻ": 25, "½": 26, "x": 27, "²": 28, "n": 29, "m": 30, "¿": 31, "q": 32, "?": 33}}})JSON", { 1, 2, 3, 4, 4, 5, 2, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33 } },
    };
    return v;
}

const char* const kGemmaLike = R"JSON({"added_tokens": [{"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 1, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 2, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 3, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 4, "content": "<mask>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}, {"id": 5, "content": "<start_of_turn>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 6, "content": "<end_of_turn>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}], "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}, "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false}, "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<eos>", "type_id": 0}}], "pair": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<eos>", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}, {"SpecialToken": {"id": "<eos>", "type_id": 1}}], "special_tokens": {"<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}, "<eos>": {"id": "<eos>", "ids": [1], "tokens": ["<eos>"]}}}, "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}]}, "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false, "vocab": {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4, "<start_of_turn>": 5, "<end_of_turn>": 6, "<0x00>": 7, "<0x01>": 8, "<0x02>": 9, "<0x03>": 10, "<0x04>": 11, "<0x05>": 12, "<0x06>": 13, "<0x07>": 14, "<0x08>": 15, "<0x09>": 16, "<0x0A>": 17, "<0x0B>": 18, "<0x0C>": 19, "<0x0D>": 20, "<0x0E>": 21, "<0x0F>": 22, "<0x10>": 23, "<0x11>": 24, "<0x12>": 25, "<0x13>": 26, "<0x14>": 27, "<0x15>": 28, "<0x16>": 29, "<0x17>": 30, "<0x18>": 31, "<0x19>": 32, "<0x1A>": 33, "<0x1B>": 34, "<0x1C>": 35, "<0x1D>": 36, "<0x1E>": 37, "<0x1F>": 38, "<0x20>": 39, "<0x21>": 40, "<0x22>": 41, "<0x23>": 42, "<0x24>": 43, "<0x25>": 44, "<0x26>": 45, "<0x27>": 46, "<0x28>": 47, "<0x29>": 48, "<0x2A>": 49, "<0x2B>": 50, "<0x2C>": 51, "<0x2D>": 52, "<0x2E>": 53, "<0x2F>": 54, "<0x30>": 55, "<0x31>": 56, "<0x32>": 57, "<0x33>": 58, "<0x34>": 59, "<0x35>": 60, "<0x36>": 61, "<0x37>": 62, "<0x38>": 63, "<0x39>": 64, "<0x3A>": 65, "<0x3B>": 66, "<0x3C>": 67, "<0x3D>": 68, "<0x3E>": 69, "<0x3F>": 70, "<0x40>": 71, "<0x41>": 72, "<0x42>": 73, "<0x43>": 74, "<0x44>": 75, "<0x45>": 76, "<0x46>": 77, "<0x47>": 78, "<0x48>": 79, "<0x49>": 80, "<0x4A>": 81, "<0x4B>": 82, "<0x4C>": 83, "<0x4D>": 84, "<0x4E>": 85, "<0x4F>": 86, "<0x50>": 87, "<0x51>": 88, "<0x52>": 89, "<0x53>": 90, "<0x54>": 91, "<0x55>": 92, "<0x56>": 93, "<0x57>": 94, "<0x58>": 95, "<0x59>": 96, "<0x5A>": 97, "<0x5B>": 98, "<0x5C>": 99, "<0x5D>": 100, "<0x5E>": 101, "<0x5F>": 102, "<0x60>": 103, "<0x61>": 104, "<0x62>": 105, "<0x63>": 106, "<0x64>": 107, "<0x65>": 108, "<0x66>": 109, "<0x67>": 110, "<0x68>": 111, "<0x69>": 112, "<0x6A>": 113, "<0x6B>": 114, "<0x6C>": 115, "<0x6D>": 116, "<0x6E>": 117, "<0x6F>": 118, "<0x70>": 119, "<0x71>": 120, "<0x72>": 121, "<0x73>": 122, "<0x74>": 123, "<0x75>": 124, "<0x76>": 125, "<0x77>": 126, "<0x78>": 127, "<0x79>": 128, "<0x7A>": 129, "<0x7B>": 130, "<0x7C>": 131, "<0x7D>": 132, "<0x7E>": 133, "<0x7F>": 134, "<0x80>": 135, "<0x81>": 136, "<0x82>": 137, "<0x83>": 138, "<0x84>": 139, "<0x85>": 140, "<0x86>": 141, "<0x87>": 142, "<0x88>": 143, "<0x89>": 144, "<0x8A>": 145, "<0x8B>": 146, "<0x8C>": 147, "<0x8D>": 148, "<0x8E>": 149, "<0x8F>": 150, "<0x90>": 151, "<0x91>": 152, "<0x92>": 153, "<0x93>": 154, "<0x94>": 155, "<0x95>": 156, "<0x96>": 157, "<0x97>": 158, "<0x98>": 159, "<0x99>": 160, "<0x9A>": 161, "<0x9B>": 162, "<0x9C>": 163, "<0x9D>": 164, "<0x9E>": 165, "<0x9F>": 166, "<0xA0>": 167, "<0xA1>": 168, "<0xA2>": 169, "<0xA3>": 170, "<0xA4>": 171, "<0xA5>": 172, "<0xA6>": 173, "<0xA7>": 174, "<0xA8>": 175, "<0xA9>": 176, "<0xAA>": 177, "<0xAB>": 178, "<0xAC>": 179, "<0xAD>": 180, "<0xAE>": 181, "<0xAF>": 182, "<0xB0>": 183, "<0xB1>": 184, "<0xB2>": 185, "<0xB3>": 186, "<0xB4>": 187, "<0xB5>": 188, "<0xB6>": 189, "<0xB7>": 190, "<0xB8>": 191, "<0xB9>": 192, "<0xBA>": 193, "<0xBB>": 194, "<0xBC>": 195, "<0xBD>": 196, "<0xBE>": 197, "<0xBF>": 198, "<0xC0>": 199, "<0xC1>": 200, "<0xC2>": 201, "<0xC3>": 202, "<0xC4>": 203, "<0xC5>": 204, "<0xC6>": 205, "<0xC7>": 206, "<0xC8>": 207, "<0xC9>": 208, "<0xCA>": 209, "<0xCB>": 210, "<0xCC>": 211, "<0xCD>": 212, "<0xCE>": 213, "<0xCF>": 214, "<0xD0>": 215, "<0xD1>": 216, "<0xD2>": 217, "<0xD3>": 218, "<0xD4>": 219, "<0xD5>": 220, "<0xD6>": 221, "<0xD7>": 222, "<0xD8>": 223, "<0xD9>": 224, "<0xDA>": 225, "<0xDB>": 226, "<0xDC>": 227, "<0xDD>": 228, "<0xDE>": 229, "<0xDF>": 230, "<0xE0>": 231, "<0xE1>": 232, "<0xE2>": 233, "<0xE3>": 234, "<0xE4>": 235, "<0xE5>": 236, "<0xE6>": 237, "<0xE7>": 238, "<0xE8>": 239, "<0xE9>": 240, "<0xEA>": 241, "<0xEB>": 242, "<0xEC>": 243, "<0xED>": 244, "<0xEE>": 245, "<0xEF>": 246, "<0xF0>": 247, "<0xF1>": 248, "<0xF2>": 249, "<0xF3>": 250, "<0xF4>": 251, "<0xF5>": 252, "<0xF6>": 253, "<0xF7>": 254, "<0xF8>": 255, "<0xF9>": 256, "<0xFA>": 257, "<0xFB>": 258, "<0xFC>": 259, "<0xFD>": 260, "<0xFE>": 261, "<0xFF>": 262, "a": 263, "b": 264, "c": 265, "d": 266, "e": 267, "f": 268, "g": 269, "h": 270, "i": 271, "j": 272, "k": 273, "l": 274, "m": 275, "n": 276, "o": 277, "p": 278, "q": 279, "r": 280, "s": 281, "t": 282, "u": 283, "v": 284, "w": 285, "x": 286, "y": 287, "z": 288, "▁": 289, "\n": 290, "\t": 291, "H": 292, "T": 293, ".": 294, ",": 295, "!": 296, "?": 297, "é": 298, "'": 299, "he": 300, "ll": 301, "hell": 302, "hello": 303, "▁w": 304, "or": 305, "▁wor": 306, "ld": 307, "▁world": 308, "▁t": 309, "▁th": 310, "▁the": 311, "▁c": 312, "▁ca": 313, "▁cat": 314, "▁▁": 315, "▁▁▁▁": 316, "\n\n": 317, "th": 318, "the": 319, "in": 320, "ing": 321, "ed": 322, "▁a": 323, "▁hello": 324, "He": 325, "Hell": 326, "Hello": 327, "ém": 328, "oj": 329, "oji": 330}, "merges": [["h", "e"], ["l", "l"], ["he", "ll"], ["hell", "o"], ["▁", "w"], ["o", "r"], ["▁w", "or"], ["l", "d"], ["▁wor", "ld"], ["▁", "t"], ["▁t", "h"], ["▁th", "e"], ["▁", "c"], ["▁c", "a"], ["▁ca", "t"], ["▁", "▁"], ["▁▁", "▁▁"], ["\n", "\n"], ["▁", "the"], ["t", "h"], ["th", "e"], ["i", "n"], ["in", "g"], ["e", "d"], ["▁", "a"], ["▁", "hello"], ["H", "e"], ["He", "ll"], ["Hell", "o"], ["é", "m"], ["o", "j"], ["oj", "i"]]}})JSON";

const std::vector<EncodeGolden>& gemma_goldens() {
    static const std::vector<EncodeGolden> v = {
        { "hello", { 303 }, { 2, 303, 1 } },
        { " hello", { 324 }, { 2, 324, 1 } },
        { "hello world", { 303, 308 }, { 2, 303, 308, 1 } },
        { "hello  world", { 303, 289, 308 }, { 2, 303, 289, 308, 1 } },
        { "the cat", { 282, 300, 314 }, { 2, 282, 300, 314, 1 } },
        { "\nthe\n\ncat", { 290, 282, 300, 317, 265, 263, 282 }, { 2, 290, 282, 300, 317, 265, 263, 282, 1 } },
        { "  leading", { 315, 274, 267, 263, 266, 321 }, { 2, 315, 274, 267, 263, 266, 321, 1 } },
        { "trailing  ", { 282, 280, 263, 271, 274, 321, 315 }, { 2, 282, 280, 263, 271, 274, 321, 315, 1 } },
        { "a\tb", { 263, 291, 264 }, { 2, 263, 291, 264, 1 } },
        { "\xc3\xa9moji \xf0\x9f\x98\x80", { 328, 330, 289, 247, 166, 159, 135 }, { 2, 328, 330, 289, 247, 166, 159, 135, 1 } },
        { "x<eos>y", { 286, 1, 287 }, { 2, 286, 1, 287, 1 } },
        { "<start_of_turn>user\nhello<end_of_turn>", { 5, 283, 281, 267, 280, 290, 303, 6 }, { 2, 5, 283, 281, 267, 280, 290, 303, 6, 1 } },
        { "Hello, world!", { 327, 295, 308, 296 }, { 2, 327, 295, 308, 296, 1 } },
        { "   ", { 315, 289 }, { 2, 315, 289, 1 } },
        { "th the thee", { 318, 309, 300, 309, 300, 267 }, { 2, 318, 309, 300, 309, 300, 267, 1 } },
        { "\xe6\x97\xa5\xe6\x9c\xac", { 237, 158, 172, 237, 163, 179 }, { 2, 237, 158, 172, 237, 163, 179, 1 } },
        { "zz  zz   zz", { 288, 288, 315, 288, 288, 315, 289, 288, 288 }, { 2, 288, 288, 315, 288, 288, 315, 289, 288, 288, 1 } },
        { "<mask> the", { 4, 309, 300 }, { 2, 4, 309, 300, 1 } },
        { "<bos>hello", { 2, 303 }, { 2, 2, 303, 1 } },
        { "", {  }, { 2, 1 } },
    };
    return v;
}

/* The decode of every Gemma golden with special tokens skipped, as HF reports it. */
const std::vector<const char*>& gemma_decoded() {
    static const std::vector<const char*> v = {
        "hello",
        " hello",
        "hello world",
        "hello  world",
        "the cat",
        "\nthe\n\ncat",
        "  leading",
        "trailing  ",
        "a\tb",
        "\xc3\xa9moji \xf0\x9f\x98\x80",
        "xy",
        "user\nhello",
        "Hello, world!",
        "   ",
        "th the thee",
        "\xe6\x97\xa5\xe6\x9c\xac",
        "zz  zz   zz",
        "<mask> the",
        "hello",
        "",
    };
    return v;
}

const char* const kMmbertLike = R"JSON({"added_tokens": [{"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 1, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 2, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 3, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 4, "content": "<mask>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}, {"id": 5, "content": "<start_of_turn>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 6, "content": "<end_of_turn>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}], "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}, "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}, "post_processor": {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<eos>", "type_id": 0}}], "pair": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<eos>", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}, {"SpecialToken": {"id": "<eos>", "type_id": 1}}], "special_tokens": {"<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}, "<eos>": {"id": "<eos>", "ids": [1], "tokens": ["<eos>"]}}}, "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}]}, "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": true, "byte_fallback": true, "ignore_merges": false, "vocab": {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4, "<start_of_turn>": 5, "<end_of_turn>": 6, "<0x00>": 7, "<0x01>": 8, "<0x02>": 9, "<0x03>": 10, "<0x04>": 11, "<0x05>": 12, "<0x06>": 13, "<0x07>": 14, "<0x08>": 15, "<0x09>": 16, "<0x0A>": 17, "<0x0B>": 18, "<0x0C>": 19, "<0x0D>": 20, "<0x0E>": 21, "<0x0F>": 22, "<0x10>": 23, "<0x11>": 24, "<0x12>": 25, "<0x13>": 26, "<0x14>": 27, "<0x15>": 28, "<0x16>": 29, "<0x17>": 30, "<0x18>": 31, "<0x19>": 32, "<0x1A>": 33, "<0x1B>": 34, "<0x1C>": 35, "<0x1D>": 36, "<0x1E>": 37, "<0x1F>": 38, "<0x20>": 39, "<0x21>": 40, "<0x22>": 41, "<0x23>": 42, "<0x24>": 43, "<0x25>": 44, "<0x26>": 45, "<0x27>": 46, "<0x28>": 47, "<0x29>": 48, "<0x2A>": 49, "<0x2B>": 50, "<0x2C>": 51, "<0x2D>": 52, "<0x2E>": 53, "<0x2F>": 54, "<0x30>": 55, "<0x31>": 56, "<0x32>": 57, "<0x33>": 58, "<0x34>": 59, "<0x35>": 60, "<0x36>": 61, "<0x37>": 62, "<0x38>": 63, "<0x39>": 64, "<0x3A>": 65, "<0x3B>": 66, "<0x3C>": 67, "<0x3D>": 68, "<0x3E>": 69, "<0x3F>": 70, "<0x40>": 71, "<0x41>": 72, "<0x42>": 73, "<0x43>": 74, "<0x44>": 75, "<0x45>": 76, "<0x46>": 77, "<0x47>": 78, "<0x48>": 79, "<0x49>": 80, "<0x4A>": 81, "<0x4B>": 82, "<0x4C>": 83, "<0x4D>": 84, "<0x4E>": 85, "<0x4F>": 86, "<0x50>": 87, "<0x51>": 88, "<0x52>": 89, "<0x53>": 90, "<0x54>": 91, "<0x55>": 92, "<0x56>": 93, "<0x57>": 94, "<0x58>": 95, "<0x59>": 96, "<0x5A>": 97, "<0x5B>": 98, "<0x5C>": 99, "<0x5D>": 100, "<0x5E>": 101, "<0x5F>": 102, "<0x60>": 103, "<0x61>": 104, "<0x62>": 105, "<0x63>": 106, "<0x64>": 107, "<0x65>": 108, "<0x66>": 109, "<0x67>": 110, "<0x68>": 111, "<0x69>": 112, "<0x6A>": 113, "<0x6B>": 114, "<0x6C>": 115, "<0x6D>": 116, "<0x6E>": 117, "<0x6F>": 118, "<0x70>": 119, "<0x71>": 120, "<0x72>": 121, "<0x73>": 122, "<0x74>": 123, "<0x75>": 124, "<0x76>": 125, "<0x77>": 126, "<0x78>": 127, "<0x79>": 128, "<0x7A>": 129, "<0x7B>": 130, "<0x7C>": 131, "<0x7D>": 132, "<0x7E>": 133, "<0x7F>": 134, "<0x80>": 135, "<0x81>": 136, "<0x82>": 137, "<0x83>": 138, "<0x84>": 139, "<0x85>": 140, "<0x86>": 141, "<0x87>": 142, "<0x88>": 143, "<0x89>": 144, "<0x8A>": 145, "<0x8B>": 146, "<0x8C>": 147, "<0x8D>": 148, "<0x8E>": 149, "<0x8F>": 150, "<0x90>": 151, "<0x91>": 152, "<0x92>": 153, "<0x93>": 154, "<0x94>": 155, "<0x95>": 156, "<0x96>": 157, "<0x97>": 158, "<0x98>": 159, "<0x99>": 160, "<0x9A>": 161, "<0x9B>": 162, "<0x9C>": 163, "<0x9D>": 164, "<0x9E>": 165, "<0x9F>": 166, "<0xA0>": 167, "<0xA1>": 168, "<0xA2>": 169, "<0xA3>": 170, "<0xA4>": 171, "<0xA5>": 172, "<0xA6>": 173, "<0xA7>": 174, "<0xA8>": 175, "<0xA9>": 176, "<0xAA>": 177, "<0xAB>": 178, "<0xAC>": 179, "<0xAD>": 180, "<0xAE>": 181, "<0xAF>": 182, "<0xB0>": 183, "<0xB1>": 184, "<0xB2>": 185, "<0xB3>": 186, "<0xB4>": 187, "<0xB5>": 188, "<0xB6>": 189, "<0xB7>": 190, "<0xB8>": 191, "<0xB9>": 192, "<0xBA>": 193, "<0xBB>": 194, "<0xBC>": 195, "<0xBD>": 196, "<0xBE>": 197, "<0xBF>": 198, "<0xC0>": 199, "<0xC1>": 200, "<0xC2>": 201, "<0xC3>": 202, "<0xC4>": 203, "<0xC5>": 204, "<0xC6>": 205, "<0xC7>": 206, "<0xC8>": 207, "<0xC9>": 208, "<0xCA>": 209, "<0xCB>": 210, "<0xCC>": 211, "<0xCD>": 212, "<0xCE>": 213, "<0xCF>": 214, "<0xD0>": 215, "<0xD1>": 216, "<0xD2>": 217, "<0xD3>": 218, "<0xD4>": 219, "<0xD5>": 220, "<0xD6>": 221, "<0xD7>": 222, "<0xD8>": 223, "<0xD9>": 224, "<0xDA>": 225, "<0xDB>": 226, "<0xDC>": 227, "<0xDD>": 228, "<0xDE>": 229, "<0xDF>": 230, "<0xE0>": 231, "<0xE1>": 232, "<0xE2>": 233, "<0xE3>": 234, "<0xE4>": 235, "<0xE5>": 236, "<0xE6>": 237, "<0xE7>": 238, "<0xE8>": 239, "<0xE9>": 240, "<0xEA>": 241, "<0xEB>": 242, "<0xEC>": 243, "<0xED>": 244, "<0xEE>": 245, "<0xEF>": 246, "<0xF0>": 247, "<0xF1>": 248, "<0xF2>": 249, "<0xF3>": 250, "<0xF4>": 251, "<0xF5>": 252, "<0xF6>": 253, "<0xF7>": 254, "<0xF8>": 255, "<0xF9>": 256, "<0xFA>": 257, "<0xFB>": 258, "<0xFC>": 259, "<0xFD>": 260, "<0xFE>": 261, "<0xFF>": 262, "a": 263, "b": 264, "c": 265, "d": 266, "e": 267, "f": 268, "g": 269, "h": 270, "i": 271, "j": 272, "k": 273, "l": 274, "m": 275, "n": 276, "o": 277, "p": 278, "q": 279, "r": 280, "s": 281, "t": 282, "u": 283, "v": 284, "w": 285, "x": 286, "y": 287, "z": 288, "▁": 289, "\n": 290, "\t": 291, "H": 292, "T": 293, ".": 294, ",": 295, "!": 296, "?": 297, "é": 298, "'": 299, "he": 300, "ll": 301, "hell": 302, "hello": 303, "▁w": 304, "or": 305, "▁wor": 306, "ld": 307, "▁world": 308, "▁t": 309, "▁th": 310, "▁the": 311, "▁c": 312, "▁ca": 313, "▁cat": 314, "▁▁": 315, "▁▁▁▁": 316, "\n\n": 317, "th": 318, "the": 319, "in": 320, "ing": 321, "ed": 322, "▁a": 323, "▁hello": 324, "He": 325, "Hell": 326, "Hello": 327, "ém": 328, "oj": 329, "oji": 330}, "merges": [["h", "e"], ["l", "l"], ["he", "ll"], ["hell", "o"], ["▁", "w"], ["o", "r"], ["▁w", "or"], ["l", "d"], ["▁wor", "ld"], ["▁", "t"], ["▁t", "h"], ["▁th", "e"], ["▁", "c"], ["▁c", "a"], ["▁ca", "t"], ["▁", "▁"], ["▁▁", "▁▁"], ["\n", "\n"], ["▁", "the"], ["t", "h"], ["th", "e"], ["i", "n"], ["in", "g"], ["e", "d"], ["▁", "a"], ["▁", "hello"], ["H", "e"], ["He", "ll"], ["Hell", "o"], ["é", "m"], ["o", "j"], ["oj", "i"]]}})JSON";

const std::vector<EncodeGolden>& mmbert_goldens() {
    static const std::vector<EncodeGolden> v = {
        { "hello", { 324 }, { 2, 324, 1 } },
        { " hello", { 324 }, { 2, 324, 1 } },
        { "hello world", { 324, 308 }, { 2, 324, 308, 1 } },
        { "hello  world", { 324, 289, 308 }, { 2, 324, 289, 308, 1 } },
        { "the cat", { 309, 300, 314 }, { 2, 309, 300, 314, 1 } },
        { "\nthe\n\ncat", { 289, 290, 282, 300, 317, 265, 263, 282 }, { 2, 289, 290, 282, 300, 317, 265, 263, 282, 1 } },
        { "  leading", { 289, 289, 274, 267, 263, 266, 321 }, { 2, 289, 289, 274, 267, 263, 266, 321, 1 } },
        { "trailing  ", { 309, 280, 263, 271, 274, 321, 289, 289 }, { 2, 309, 280, 263, 271, 274, 321, 289, 289, 1 } },
        { "a\tb", { 323, 291, 264 }, { 2, 323, 291, 264, 1 } },
        { "\xc3\xa9moji \xf0\x9f\x98\x80", { 289, 328, 330, 289, 247, 166, 159, 135 }, { 2, 289, 328, 330, 289, 247, 166, 159, 135, 1 } },
        { "x<eos>y", { 289, 286, 1, 289, 287 }, { 2, 289, 286, 1, 289, 287, 1 } },
        { "<start_of_turn>user\nhello<end_of_turn>", { 5, 289, 283, 281, 267, 280, 290, 303, 6 }, { 2, 5, 289, 283, 281, 267, 280, 290, 303, 6, 1 } },
        { "Hello, world!", { 289, 327, 295, 308, 296 }, { 2, 289, 327, 295, 308, 296, 1 } },
        { "   ", { 289, 289, 289 }, { 2, 289, 289, 289, 1 } },
        { "th the thee", { 310, 309, 300, 309, 300, 267 }, { 2, 310, 309, 300, 309, 300, 267, 1 } },
        { "\xe6\x97\xa5\xe6\x9c\xac", { 289, 237, 158, 172, 237, 163, 179 }, { 2, 289, 237, 158, 172, 237, 163, 179, 1 } },
        { "zz  zz   zz", { 289, 288, 288, 289, 289, 288, 288, 289, 289, 289, 288, 288 }, { 2, 289, 288, 288, 289, 289, 288, 288, 289, 289, 289, 288, 288, 1 } },
        { "<mask> the", { 4, 309, 300 }, { 2, 4, 309, 300, 1 } },
        { "<bos>hello", { 2, 324 }, { 2, 2, 324, 1 } },
        { "", {  }, { 2, 1 } },
        { " the cat", { 309, 300, 314 }, { 2, 309, 300, 314, 1 } },
        { "  two  leading", { 289, 309, 285, 277, 289, 289, 274, 267, 263, 266, 321 }, { 2, 289, 309, 285, 277, 289, 289, 274, 267, 263, 266, 321, 1 } },
        { " ", { 289 }, { 2, 289, 1 } },
        { "x\n the", { 289, 286, 290, 309, 300 }, { 2, 289, 286, 290, 309, 300, 1 } },
    };
    return v;
}

const std::vector<std::pair<const char*, const char*>>& metaspace_special_fixtures() {
    static const std::vector<std::pair<const char*, const char*>> v = {
        { "first", R"JSON({"added_tokens": [{"id": 0, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 1, "content": "[INST]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}], "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false}, "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false, "vocab": {"<s>": 0, "[INST]": 1, "▁": 2, "H": 3, "i": 4, "▁H": 5, "Hi": 6, "▁Hi": 7, "<unk>": 8}, "merges": [["▁", "H"], ["H", "i"], ["▁H", "i"]]}})JSON" },
        { "always", R"JSON({"added_tokens": [{"id": 0, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}, {"id": 1, "content": "[INST]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}], "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": false}, "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false, "vocab": {"<s>": 0, "[INST]": 1, "▁": 2, "H": 3, "i": 4, "▁H": 5, "Hi": 6, "▁Hi": 7, "<unk>": 8}, "merges": [["▁", "H"], ["H", "i"], ["▁H", "i"]]}})JSON" },
    };
    return v;
}

const std::vector<std::pair<const char*, EncodeGolden>>& metaspace_special_goldens() {
    static const std::vector<std::pair<const char*, EncodeGolden>> v = {
        { "first", { "Hi", { 7 }, {} } },
        { "first", { "[INST]Hi", { 1, 6 }, {} } },
        { "first", { "[INST] Hi", { 1, 7 }, {} } },
        { "first", { "Hi[INST]Hi", { 7, 1, 6 }, {} } },
        { "first", { "<s>Hi", { 0, 6 }, {} } },
        { "first", { "Hi Hi[INST]Hi", { 7, 7, 1, 6 }, {} } },
        { "always", { "Hi", { 7 }, {} } },
        { "always", { "[INST]Hi", { 1, 7 }, {} } },
        { "always", { "[INST] Hi", { 1, 7 }, {} } },
        { "always", { "Hi[INST]Hi", { 7, 1, 7 }, {} } },
        { "always", { "<s>Hi", { 0, 7 }, {} } },
        { "always", { "Hi Hi[INST]Hi", { 7, 7, 1, 7 }, {} } },
    };
    return v;
}

const char* const kClipLike = R"JSON({"added_tokens": [{"id": 620, "content": "<|startoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": true, "special": true}, {"id": 621, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}], "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFC"}, {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "}, {"type": "Lowercase"}]}, "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"}, "behavior": "Removed", "invert": true}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true}]}, "post_processor": {"type": "RobertaProcessing", "sep": ["<|endoftext|>", 621], "cls": ["<|startoftext|>", 620], "trim_offsets": false, "add_prefix_space": false}, "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true}, "model": {"type": "BPE", "dropout": null, "unk_token": "<|endoftext|>", "continuing_subword_prefix": "", "end_of_word_suffix": "</w>", "fuse_unk": false, "vocab": {"!": 0, "\"": 1, "#": 2, "$": 3, "%": 4, "&": 5, "'": 6, "(": 7, ")": 8, "*": 9, "+": 10, ",": 11, "-": 12, ".": 13, "/": 14, "0": 15, "1": 16, "2": 17, "3": 18, "4": 19, "5": 20, "6": 21, "7": 22, "8": 23, "9": 24, ":": 25, ";": 26, "<": 27, "=": 28, ">": 29, "?": 30, "@": 31, "A": 32, "B": 33, "C": 34, "D": 35, "E": 36, "F": 37, "G": 38, "H": 39, "I": 40, "J": 41, "K": 42, "L": 43, "M": 44, "N": 45, "O": 46, "P": 47, "Q": 48, "R": 49, "S": 50, "T": 51, "U": 52, "V": 53, "W": 54, "X": 55, "Y": 56, "Z": 57, "[": 58, "\\": 59, "]": 60, "^": 61, "_": 62, "`": 63, "a": 64, "b": 65, "c": 66, "d": 67, "e": 68, "f": 69, "g": 70, "h": 71, "i": 72, "j": 73, "k": 74, "l": 75, "m": 76, "n": 77, "o": 78, "p": 79, "q": 80, "r": 81, "s": 82, "t": 83, "u": 84, "v": 85, "w": 86, "x": 87, "y": 88, "z": 89, "{": 90, "|": 91, "}": 92, "~": 93, "¡": 94, "¢": 95, "£": 96, "¤": 97, "¥": 98, "¦": 99, "§": 100, "¨": 101, "©": 102, "ª": 103, "«": 104, "¬": 105, "®": 106, "¯": 107, "°": 108, "±": 109, "²": 110, "³": 111, "´": 112, "µ": 113, "¶": 114, "·": 115, "¸": 116, "¹": 117, "º": 118, "»": 119, "¼": 120, "½": 121, "¾": 122, "¿": 123, "À": 124, "Á": 125, "Â": 126, "Ã": 127, "Ä": 128, "Å": 129, "Æ": 130, "Ç": 131, "È": 132, "É": 133, "Ê": 134, "Ë": 135, "Ì": 136, "Í": 137, "Î": 138, "Ï": 139, "Ð": 140, "Ñ": 141, "Ò": 142, "Ó": 143, "Ô": 144, "Õ": 145, "Ö": 146, "×": 147, "Ø": 148, "Ù": 149, "Ú": 150, "Û": 151, "Ü": 152, "Ý": 153, "Þ": 154, "ß": 155, "à": 156, "á": 157, "â": 158, "ã": 159, "ä": 160, "å": 161, "æ": 162, "ç": 163, "è": 164, "é": 165, "ê": 166, "ë": 167, "ì": 168, "í": 169, "î": 170, "ï": 171, "ð": 172, "ñ": 173, "ò": 174, "ó": 175, "ô": 176, "õ": 177, "ö": 178, "÷": 179, "ø": 180, "ù": 181, "ú": 182, "û": 183, "ü": 184, "ý": 185, "þ": 186, "ÿ": 187, "Ā": 188, "ā": 189, "Ă": 190, "ă": 191, "Ą": 192, "ą": 193, "Ć": 194, "ć": 195, "Ĉ": 196, "ĉ": 197, "Ċ": 198, "ċ": 199, "Č": 200, "č": 201, "Ď": 202, "ď": 203, "Đ": 204, "đ": 205, "Ē": 206, "ē": 207, "Ĕ": 208, "ĕ": 209, "Ė": 210, "ė": 211, "Ę": 212, "ę": 213, "Ě": 214, "ě": 215, "Ĝ": 216, "ĝ": 217, "Ğ": 218, "ğ": 219, "Ġ": 220, "ġ": 221, "Ģ": 222, "ģ": 223, "Ĥ": 224, "ĥ": 225, "Ħ": 226, "ħ": 227, "Ĩ": 228, "ĩ": 229, "Ī": 230, "ī": 231, "Ĭ": 232, "ĭ": 233, "Į": 234, "į": 235, "İ": 236, "ı": 237, "Ĳ": 238, "ĳ": 239, "Ĵ": 240, "ĵ": 241, "Ķ": 242, "ķ": 243, "ĸ": 244, "Ĺ": 245, "ĺ": 246, "Ļ": 247, "ļ": 248, "Ľ": 249, "ľ": 250, "Ŀ": 251, "ŀ": 252, "Ł": 253, "ł": 254, "Ń": 255, "!</w>": 256, "\"</w>": 257, "#</w>": 258, "$</w>": 259, "%</w>": 260, "&</w>": 261, "'</w>": 262, "(</w>": 263, ")</w>": 264, "*</w>": 265, "+</w>": 266, ",</w>": 267, "-</w>": 268, ".</w>": 269, "/</w>": 270, "0</w>": 271, "1</w>": 272, "2</w>": 273, "3</w>": 274, "4</w>": 275, "5</w>": 276, "6</w>": 277, "7</w>": 278, "8</w>": 279, "9</w>": 280, ":</w>": 281, ";</w>": 282, "<</w>": 283, "=</w>": 284, "></w>": 285, "?</w>": 286, "@</w>": 287, "A</w>": 288, "B</w>": 289, "C</w>": 290, "D</w>": 291, "E</w>": 292, "F</w>": 293, "G</w>": 294, "H</w>": 295, "I</w>": 296, "J</w>": 297, "K</w>": 298, "L</w>": 299, "M</w>": 300, "N</w>": 301, "O</w>": 302, "P</w>": 303, "Q</w>": 304, "R</w>": 305, "S</w>": 306, "T</w>": 307, "U</w>": 308, "V</w>": 309, "W</w>": 310, "X</w>": 311, "Y</w>": 312, "Z</w>": 313, "[</w>": 314, "\\</w>": 315, "]</w>": 316, "^</w>": 317, "_</w>": 318, "`</w>": 319, "a</w>": 320, "b</w>": 321, "c</w>": 322, "d</w>": 323, "e</w>": 324, "f</w>": 325, "g</w>": 326, "h</w>": 327, "i</w>": 328, "j</w>": 329, "k</w>": 330, "l</w>": 331, "m</w>": 332, "n</w>": 333, "o</w>": 334, "p</w>": 335, "q</w>": 336, "r</w>": 337, "s</w>": 338, "t</w>": 339, "u</w>": 340, "v</w>": 341, "w</w>": 342, "x</w>": 343, "y</w>": 344, "z</w>": 345, "{</w>": 346, "|</w>": 347, "}</w>": 348, "~</w>": 349, "¡</w>": 350, "¢</w>": 351, "£</w>": 352, "¤</w>": 353, "¥</w>": 354, "¦</w>": 355, "§</w>": 356, "¨</w>": 357, "©</w>": 358, "ª</w>": 359, "«</w>": 360, "¬</w>": 361, "®</w>": 362, "¯</w>": 363, "°</w>": 364, "±</w>": 365, "²</w>": 366, "³</w>": 367, "´</w>": 368, "µ</w>": 369, "¶</w>": 370, "·</w>": 371, "¸</w>": 372, "¹</w>": 373, "º</w>": 374, "»</w>": 375, "¼</w>": 376, "½</w>": 377, "¾</w>": 378, "¿</w>": 379, "À</w>": 380, "Á</w>": 381, "Â</w>": 382, "Ã</w>": 383, "Ä</w>": 384, "Å</w>": 385, "Æ</w>": 386, "Ç</w>": 387, "È</w>": 388, "É</w>": 389, "Ê</w>": 390, "Ë</w>": 391, "Ì</w>": 392, "Í</w>": 393, "Î</w>": 394, "Ï</w>": 395, "Ð</w>": 396, "Ñ</w>": 397, "Ò</w>": 398, "Ó</w>": 399, "Ô</w>": 400, "Õ</w>": 401, "Ö</w>": 402, "×</w>": 403, "Ø</w>": 404, "Ù</w>": 405, "Ú</w>": 406, "Û</w>": 407, "Ü</w>": 408, "Ý</w>": 409, "Þ</w>": 410, "ß</w>": 411, "à</w>": 412, "á</w>": 413, "â</w>": 414, "ã</w>": 415, "ä</w>": 416, "å</w>": 417, "æ</w>": 418, "ç</w>": 419, "è</w>": 420, "é</w>": 421, "ê</w>": 422, "ë</w>": 423, "ì</w>": 424, "í</w>": 425, "î</w>": 426, "ï</w>": 427, "ð</w>": 428, "ñ</w>": 429, "ò</w>": 430, "ó</w>": 431, "ô</w>": 432, "õ</w>": 433, "ö</w>": 434, "÷</w>": 435, "ø</w>": 436, "ù</w>": 437, "ú</w>": 438, "û</w>": 439, "ü</w>": 440, "ý</w>": 441, "þ</w>": 442, "ÿ</w>": 443, "Ā</w>": 444, "ā</w>": 445, "Ă</w>": 446, "ă</w>": 447, "Ą</w>": 448, "ą</w>": 449, "Ć</w>": 450, "ć</w>": 451, "Ĉ</w>": 452, "ĉ</w>": 453, "Ċ</w>": 454, "ċ</w>": 455, "Č</w>": 456, "č</w>": 457, "Ď</w>": 458, "ď</w>": 459, "Đ</w>": 460, "đ</w>": 461, "Ē</w>": 462, "ē</w>": 463, "Ĕ</w>": 464, "ĕ</w>": 465, "Ė</w>": 466, "ė</w>": 467, "Ę</w>": 468, "ę</w>": 469, "Ě</w>": 470, "ě</w>": 471, "Ĝ</w>": 472, "ĝ</w>": 473, "Ğ</w>": 474, "ğ</w>": 475, "Ġ</w>": 476, "ġ</w>": 477, "Ģ</w>": 478, "ģ</w>": 479, "Ĥ</w>": 480, "ĥ</w>": 481, "Ħ</w>": 482, "ħ</w>": 483, "Ĩ</w>": 484, "ĩ</w>": 485, "Ī</w>": 486, "ī</w>": 487, "Ĭ</w>": 488, "ĭ</w>": 489, "Į</w>": 490, "į</w>": 491, "İ</w>": 492, "ı</w>": 493, "Ĳ</w>": 494, "ĳ</w>": 495, "Ĵ</w>": 496, "ĵ</w>": 497, "Ķ</w>": 498, "ķ</w>": 499, "ĸ</w>": 500, "Ĺ</w>": 501, "ĺ</w>": 502, "Ļ</w>": 503, "ļ</w>": 504, "Ľ</w>": 505, "ľ</w>": 506, "Ŀ</w>": 507, "ŀ</w>": 508, "Ł</w>": 509, "ł</w>": 510, "Ń</w>": 511, "in": 512, "of</w>": 513, "at</w>": 514, "ho": 515, "ing</w>": 516, "st": 517, "ta": 518, "ast": 519, "do": 520, "gi": 521, "hot": 522, "ie": 523, "it</w>": 524, "phot": 525, "ra": 526, "un": 527, "'s</w>": 528, "'t</w>": 529, "ain": 530, "al": 531, "an</w>": 532, "ar": 533, "au": 534, "aint": 535, "ainting</w>": 536, "ali": 537, "alit": 538, "ality</w>": 539, "art</w>": 540, "aste": 541, "astr": 542, "aster": 543, "asterp": 544, "asterpie": 545, "asterpiec": 546, "asterpiece</w>": 547, "astro": 548, "astron": 549, "astronau": 550, "astronaut</w>": 551, "be": 552, "bes": 553, "best</w>": 554, "cat</w>": 555, "de": 556, "di": 557, "ding</w>": 558, "deta": 559, "detai": 560, "detail": 561, "detaile": 562, "detailed</w>": 563, "digi": 564, "digita": 565, "digital</w>": 566, "dog</w>": 567, "don</w>": 568, "er</w>": 569, "et</w>": 570, "gh": 571, "gra": 572, "ghl": 573, "ghly</w>": 574, "gir": 575, "girl</w>": 576, "grap": 577, "graph</w>": 578, "he</w>": 579, "hi": 580, "highly</w>": 581, "hor": 582, "hors": 583, "horse</w>": 584, "iding</w>": 585, "iew": 586, "iewer</w>": 587, "ins</w>": 588, "king</w>": 589, "lo": 590, "lo</w>": 591, "loo": 592, "looking</w>": 593, "masterpiece</w>": 594, "mo": 595, "moun": 596, "mounta": 597, "mountains</w>": 598, "ograph</w>": 599, "olo</w>": 600, "op</w>": 601, "or": 602, "ort": 603, "ortra": 604, "ortrait</w>": 605, "painting</w>": 606, "portrait</w>": 607, "photo</w>": 608, "photograph</w>": 609, "qu": 610, "quality</w>": 611, "riding</w>": 612, "set</w>": 613, "solo</w>": 614, "sun": 615, "stop</w>": 616, "sunset</w>": 617, "the</w>": 618, "viewer</w>": 619, "<|startoftext|>": 620, "<|endoftext|>": 621}, "merges": ["i n", "o f</w>", "a t</w>", "h o", "in g</w>", "s t", "t a", "a st", "d o", "g i", "ho t", "i e", "i t</w>", "p hot", "r a", "u n", "' s</w>", "' t</w>", "a in", "a l", "a n</w>", "a r", "a u", "ain t", "aint ing</w>", "al i", "ali t", "alit y</w>", "ar t</w>", "ast e", "ast r", "aste r", "aster p", "asterp ie", "asterpie c", "asterpiec e</w>", "astr o", "astro n", "astron au", "astronau t</w>", "b e", "be s", "bes t</w>", "c at</w>", "d e", "d i", "d ing</w>", "de ta", "deta i", "detai l", "detail e", "detaile d</w>", "di gi", "digi ta", "digita l</w>", "do g</w>", "do n</w>", "e r</w>", "e t</w>", "g h", "g ra", "gh l", "ghl y</w>", "gi r", "gir l</w>", "gra p", "grap h</w>", "h e</w>", "h i", "hi ghly</w>", "ho r", "hor s", "hors e</w>", "i ding</w>", "ie w", "iew er</w>", "in s</w>", "k ing</w>", "l o", "l o</w>", "lo o", "loo king</w>", "m asterpiece</w>", "m o", "mo un", "moun ta", "mounta ins</w>", "o graph</w>", "o lo</w>", "o p</w>", "o r", "or t", "ort ra", "ortra it</w>", "p ainting</w>", "p ortrait</w>", "phot o</w>", "phot ograph</w>", "q u", "qu ality</w>", "r iding</w>", "s et</w>", "s olo</w>", "s un", "st op</w>", "sun set</w>", "t he</w>", "v iewer</w>"]}})JSON";

const std::vector<EncodeGolden>& clip_goldens() {
    static const std::vector<EncodeGolden> v = {
        { "a photo of a cat", { 320, 608, 513, 320, 555 }, { 620, 320, 608, 513, 320, 555, 621 } },
        { "1girl, solo, masterpiece, best quality, (absurdres:1.2), looking_at_viewer, hatsune_miku", { 272, 576, 267, 614, 267, 594, 267, 554, 611, 267, 263, 64, 65, 82, 84, 81, 67, 81, 68, 338, 281, 272, 269, 273, 8, 267, 593, 318, 514, 318, 619, 267, 71, 64, 83, 615, 324, 318, 76, 72, 74, 340 }, { 620, 272, 576, 267, 614, 267, 594, 267, 554, 611, 267, 263, 64, 65, 82, 84, 81, 67, 81, 68, 338, 281, 272, 269, 273, 8, 267, 593, 318, 514, 318, 619, 267, 71, 64, 83, 615, 324, 318, 76, 72, 74, 340, 621 } },
        { "A Photograph of the Mountains, it's DON'T stop", { 320, 609, 513, 618, 598, 267, 524, 528, 568, 529, 616 }, { 620, 320, 609, 513, 618, 598, 267, 524, 528, 568, 529, 616, 621 } },
        { "1024x1024 2.5 v1.0", { 272, 271, 273, 275, 343, 272, 271, 273, 275, 273, 269, 276, 341, 272, 269, 271 }, { 620, 272, 271, 273, 275, 343, 272, 271, 273, 275, 273, 269, 276, 341, 272, 269, 271, 621 } },
        { "Caf\xc3\xa9 na\xc3\xafve \xe6\x9d\xb1\xe4\xba\xac\xe3\x82\xbf\xe3\x83\xaf\xe3\x83\xbc \xec\x84\x9c\xec\x9a\xb8 \xf0\x9f\x8d\xa3\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x91\xa7 \xc4\xb0stanbul \xce\xa3\xce\x8a\xce\xa3\xce\xa5\xce\xa6\xce\x9f\xce\xa3", { 66, 64, 69, 127, 358, 77, 64, 127, 107, 85, 324, 162, 251, 109, 160, 118, 105, 159, 224, 123, 159, 225, 107, 159, 225, 376, 168, 226, 250, 168, 248, 372, 172, 253, 235, 96, 172, 253, 239, 102, 158, 222, 235, 172, 253, 239, 102, 158, 222, 235, 172, 253, 239, 356, 328, 136, 485, 517, 64, 77, 65, 84, 331, 139, 225, 138, 107, 139, 225, 139, 227, 139, 228, 138, 123, 139, 481 }, { 620, 66, 64, 69, 127, 358, 77, 64, 127, 107, 85, 324, 162, 251, 109, 160, 118, 105, 159, 224, 123, 159, 225, 107, 159, 225, 376, 168, 226, 250, 168, 248, 372, 172, 253, 235, 96, 172, 253, 239, 102, 158, 222, 235, 172, 253, 239, 102, 158, 222, 235, 172, 253, 239, 356, 328, 136, 485, 517, 64, 77, 65, 84, 331, 139, 225, 138, 107, 139, 225, 139, 227, 139, 228, 138, 123, 139, 481, 621 } },
        { "  many   spaces\t\ttabs\n\nnewlines  ", { 76, 64, 77, 344, 82, 79, 64, 66, 68, 338, 518, 65, 338, 77, 68, 86, 75, 512, 68, 338 }, { 620, 76, 64, 77, 344, 82, 79, 64, 66, 68, 338, 518, 65, 338, 77, 68, 86, 75, 512, 68, 338, 621 } },
        { "...!!!??? ((()))[[]]", { 13, 13, 13, 0, 0, 0, 30, 30, 286, 7, 7, 7, 8, 8, 8, 58, 58, 60, 316 }, { 620, 13, 13, 13, 0, 0, 0, 30, 30, 286, 7, 7, 7, 8, 8, 8, 58, 58, 60, 316, 621 } },
        { "", {  }, { 620, 621 } },
        { "a", { 320 }, { 620, 320, 621 } },
        { "<|startoftext|>a cat<|endoftext|>", { 620, 320, 555, 621 }, { 620, 620, 320, 555, 621, 621 } },
        { "x<|StartOfText|>y", { 343, 620, 344 }, { 620, 343, 620, 344, 621 } },
        { "a</w>b catcat", { 320, 27, 270, 342, 285, 321, 66, 64, 83, 555 }, { 620, 320, 27, 270, 342, 285, 321, 66, 64, 83, 555, 621 } },
        { "photographphotograph", { 525, 78, 577, 71, 609 }, { 620, 525, 78, 577, 71, 609, 621 } },
        { "it'sit's'll", { 524, 528, 524, 528, 6, 75, 331 }, { 620, 524, 528, 524, 528, 6, 75, 331, 621 } },
    };
    return v;
}

/* The decode of every CLIP golden with special tokens skipped, as HF reports it. */
const std::vector<const char*>& clip_decoded() {
    static const std::vector<const char*> v = {
        "a</w>photo</w>of</w>a</w>cat</w>",
        "1</w>girl</w>,</w>solo</w>,</w>masterpiece</w>,</w>best</w>quality</w>,</w>(</w>absurdres</w>:</w>1</w>.</w>2</w>),</w>looking</w>_</w>at</w>_</w>viewer</w>,</w>hatsune</w>_</w>miku</w>",
        "a</w>photograph</w>of</w>the</w>mountains</w>,</w>it</w>'s</w>don</w>'t</w>stop</w>",
        "1</w>0</w>2</w>4</w>x</w>1</w>0</w>2</w>4</w>2</w>.</w>5</w>v</w>1</w>.</w>0</w>",
        "caf\xc3\xa9</w>na\xc3\xafve</w>\xe6\x9d\xb1\xe4\xba\xac\xe3\x82\xbf\xe3\x83\xaf\xe3\x83\xbc</w>\xec\x84\x9c\xec\x9a\xb8</w>\xf0\x9f\x8d\xa3\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x91\xa7</w>i</w>\xcc\x87</w>stanbul</w>\xcf\x83\xce\xaf\xcf\x83\xcf\x85\xcf\x86\xce\xbf\xcf\x83</w>",
        "many</w>spaces</w>tabs</w>newlines</w>",
        "...!!!???</w>((()))[[]]</w>",
        "",
        "a</w>",
        "a</w>cat</w>",
        "x</w>y</w>",
        "a</w></</w>w</w>></w>b</w>catcat</w>",
        "photographphotograph</w>",
        "it</w>'s</w>it</w>'s</w>'ll</w>",
    };
    return v;
}

/* END generated by mk_tokenizer_goldens.py */

}  /* namespace */

/* ================================================================== parity with HuggingFace */

namespace {

std::shared_ptr<const Vocab> vocab_from_json(const char* js, const char* origin) {
    VocabBuild vb;
    if (parse_tokenizer_json_buf(js, origin, vb) < 0) return nullptr;
    auto v = std::make_shared<Vocab>();
    if (v->load(std::move(vb)) < 0) return nullptr;
    return v;
}

std::string streamed(const std::shared_ptr<const Vocab>& v, const std::vector<int32_t>& ids) {
    Detokenizer d(v);
    std::string s;
    for (int32_t id : ids) s += d.push(id);
    return s + d.flush();
}

}  /* namespace */

/* Each decoder shape a tokenizer.json declares, against what HuggingFace's decode() returns for
 * the same ids with special tokens skipped: a Llama-style Sequence ending in Strip after Fuse
 * (one leading space off the whole text, not off every token), Metaspace under each prepend
 * scheme (the first token's replacement characters dropped, not turned into spaces), WordPiece
 * with and without cleanup (words joined with a space, the first left as it is), BPEDecoder (no
 * space after the last word) and ByteLevel. decode() and the streaming Detokenizer must both give
 * it. */
TEST(decoders_match_huggingface) {
    for (const auto& fx : decode_fixtures()) {
        auto v = vocab_from_json(fx.second, fx.first);
        REQUIRE(v != nullptr);
        Tokenizer t;
        CHECK_OK(t.init(v));
        int n = 0;
        for (const DecodeGolden& g : decode_goldens()) {
            if (std::string(g.fixture) != fx.first) continue;
            ++n;
            const std::string got = t.decode(g.ids), stream = streamed(v, g.ids);
            CHECK_EQ(got, std::string(g.text));
            CHECK_EQ(stream, std::string(g.text));
            if (got != g.text || stream != g.text)
                fprintf(stderr, "    %s: decode \"%s\", stream \"%s\", HF \"%s\"\n", fx.first,
                        got.c_str(), stream.c_str(), g.text);
        }
        CHECK(n >= 2);
    }
}

/* Each pre-tokenizer variant over one line that exercises every class boundary (ASCII and
 * Unicode punctuation, symbols that are not punctuation, combining marks, a connector punctuation,
 * letter and other numbers, superscripts, non-ASCII whitespace). The vocabulary is a WordPiece one
 * made of exactly the pieces HuggingFace cuts the line into, so the ids name the pieces one for
 * one and a different cut shows as another id or [UNK]. */
TEST(pre_tokenizers_split_as_huggingface) {
    for (const PretokGolden& g : pretok_goldens()) {
        auto v = vocab_from_json(g.json, g.name);
        REQUIRE(v != nullptr);
        Tokenizer t;
        CHECK_OK(t.init(v));
        t.set_segment_cache(false);
        std::vector<int32_t> ids;
        CHECK_OK(t.encode(kPretokText, ids, false, false));
        CHECK(ids == g.ids);
        if (ids != g.ids) {
            std::string got;
            for (int32_t id : ids) got += "|" + v->text(id);
            fprintf(stderr, "    %s: %s\n", g.name, got.c_str());
        }
    }
}

/* The SentencePiece-converted shape Gemma ships: a Replace normaliser (space to U+2581), a Split
 * on " " that is not Isolated, BPE with byte fallback, a Replace/ByteFallback/Fuse decoder and a
 * TemplateProcessing post-processor `<bos> $A <eos>`, which is the model's add-BOS/EOS rule
 * (EmbeddingGemma pools over that <eos>). Encoded with and without special tokens and decoded,
 * against HuggingFace on spaces, newlines, tabs, byte fallback (an emoji, CJK), special tokens in
 * the text and the empty string. */
/* mmBERT's shape (Laya's multilingual checkpoint): the Gemma normaliser turns spaces into U+2581,
 * then Metaspace(always, split) prepends U+2581 only to text that does not already start with it.
 * Prepending it again gave " hello" a lone U+2581 token ahead of "U+2581hello". */
TEST(a_metaspace_after_a_space_replacing_normaliser_encodes_as_huggingface) {
    VocabBuild vb;
    REQUIRE(parse_tokenizer_json_buf(kMmbertLike, "mmbert_like", vb) >= 0);
    auto v = std::make_shared<Vocab>();
    vb.bos = 2;
    vb.eos = 1;
    REQUIRE(v->load(std::move(vb)) >= 0);
    Tokenizer t;
    CHECK_OK(t.init(v));
    for (const EncodeGolden& g : mmbert_goldens()) {
        std::vector<int32_t> plain;
        CHECK_OK(t.encode(g.text, plain, /*add_special=*/false, /*parse_special=*/true));
        CHECK(plain == g.ids);
        if (plain != g.ids) {
            std::string got;
            for (int32_t id : plain) got += std::to_string(id) + ",";
            fprintf(stderr, "    \"%s\": %s\n", g.text, got.c_str());
        }
    }
}

TEST(the_gemma_shaped_tokenizer_encodes_as_huggingface) {
    VocabBuild vb;
    REQUIRE(parse_tokenizer_json_buf(kGemmaLike, "gemma_like", vb) >= 0);
    CHECK(vb.add_bos);
    CHECK(vb.add_eos);
    auto v = std::make_shared<Vocab>();
    vb.bos = 2;
    vb.eos = 1;
    REQUIRE(v->load(std::move(vb)) >= 0);
    Tokenizer t;
    CHECK_OK(t.init(v));
    size_t k = 0;
    for (const EncodeGolden& g : gemma_goldens()) {
        std::vector<int32_t> plain, special;
        CHECK_OK(t.encode(g.text, plain, /*add_special=*/false, /*parse_special=*/true));
        CHECK_OK(t.encode(g.text, special, /*add_special=*/true, /*parse_special=*/true));
        CHECK(plain == g.ids);
        /* Deliberate: a text that opens with <bos> gets no second one (Tokenizer::encode);
         * HuggingFace adds it again. */
        std::vector<int32_t> want = g.with_special;
        if (std::string(g.text).rfind("<bos>", 0) == 0) want.erase(want.begin());
        CHECK(special == want);
        CHECK_EQ(std::count(special.begin(), special.end(), 1) -
                     std::count(plain.begin(), plain.end(), 1), (ptrdiff_t)1);
        if (plain != g.ids) {
            std::string got;
            for (int32_t id : plain) got += std::to_string(id) + ",";
            fprintf(stderr, "    \"%s\": %s\n", g.text, got.c_str());
        }
        CHECK_EQ(t.decode(g.with_special), std::string(gemma_decoded()[k]));
        CHECK_EQ(streamed(v, g.with_special), std::string(gemma_decoded()[k]));
        ++k;
    }
    CHECK_EQ(k, gemma_decoded().size());
}

/* CLIP's shape (Stable Diffusion's text encoders): an NFC / whitespace-collapse / Lowercase
 * normaliser, a Split that keeps letter runs, single digits and punctuation runs, ByteLevel, BPE
 * whose word-final symbols carry the "</w>" suffix, RobertaProcessing `<|startoftext|> $A
 * <|endoftext|>`, and `<|startoftext|>` matched in the normalised text (so `<|StartOfText|>` is it
 * too). Encoded with and without the framing and decoded, against HuggingFace. */
TEST(the_clip_shaped_tokenizer_encodes_as_huggingface) {
    VocabBuild vb;
    REQUIRE(parse_tokenizer_json_buf(kClipLike, "clip_like", vb) >= 0);
    CHECK(vb.add_bos);
    CHECK(vb.add_eos);
    CHECK_EQ(vb.bos, 620);
    CHECK_EQ(vb.eos, 621);
    /* Served as a container holds it: the suffix and the normalised added token are steps. */
    VocabBuild baked;
    REQUIRE(round_trip(vb, &baked) == RAD_OK);
    auto v = std::make_shared<Vocab>();
    REQUIRE(v->load(std::move(baked)) >= 0);
    Tokenizer t;
    CHECK_OK(t.init(v));
    size_t k = 0;
    for (const EncodeGolden& g : clip_goldens()) {
        std::vector<int32_t> plain, special;
        CHECK_OK(t.encode(g.text, plain, /*add_special=*/false, /*parse_special=*/true));
        CHECK_OK(t.encode(g.text, special, /*add_special=*/true, /*parse_special=*/true));
        CHECK(plain == g.ids);
        /* A text opening with the BOS token gets a second one, as in HuggingFace: CLIP's BOS
         * is matched in the normalised text, after Tokenizer::encode decides on adding one. */
        CHECK(special == g.with_special);
        if (plain != g.ids) {
            std::string got;
            for (int32_t id : plain) got += std::to_string(id) + ",";
            fprintf(stderr, "    \"%s\": %s\n", g.text, got.c_str());
        }
        CHECK_EQ(t.decode(g.with_special), std::string(clip_decoded()[k]));
        CHECK_EQ(streamed(v, g.with_special), std::string(clip_decoded()[k]));
        ++k;
    }
    CHECK_EQ(k, clip_decoded().size());

    /* A token matched in the normalised text follows parse_special as the raw ones do, through
     * the segment cache as well: the same long text with it off and then on. */
    std::string text;
    for (int i = 0; i < 30; ++i) text += "a photo of a cat <|StartOfText|> ";
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<int32_t> off, on;
        CHECK_OK(t.encode(text, off, false, /*parse_special=*/false));
        CHECK_OK(t.encode(text, on, false, /*parse_special=*/true));
        CHECK_EQ(std::count(off.begin(), off.end(), 620), (ptrdiff_t)0);
        CHECK_EQ(std::count(on.begin(), on.end(), 620), (ptrdiff_t)30);
    }
}

/* Metaspace's "first" scheme prepends the replacement only where the text starts: a segment after
 * a special token ("[INST]Hi", "<s>Hi") has a non-zero original offset in HuggingFace and gets
 * none, where "always" gives every segment one. Encoded through the segment cache both cold and
 * warm, since the first segment's ids depend on its place and not only its bytes. */
TEST(a_first_prepend_applies_only_where_the_text_starts) {
    for (const auto& fx : metaspace_special_fixtures()) {
        auto v = vocab_from_json(fx.second, fx.first);
        REQUIRE(v != nullptr);
        Tokenizer t;
        CHECK_OK(t.init(v));
        for (int pass = 0; pass < 2; ++pass)
            for (const auto& g : metaspace_special_goldens()) {
                if (std::string(g.first) != fx.first) continue;
                /* Long enough to reach the segment cache. */
                std::string text = g.second.text;
                std::vector<int32_t> want = g.second.ids;
                for (int k = 0; k < 40; ++k) {
                    text += " Hi";
                    want.push_back(7);
                }
                std::vector<int32_t> ids;
                CHECK_OK(t.encode(text, ids, false, true));
                CHECK(ids == want);
                if (ids != want) fprintf(stderr, "    %s \"%s\"\n", fx.first, g.second.text);
            }
    }
}

/* ================================================================== refusals */

/* A value the tokenizer has no encoding for is refused, never guessed: an unknown Split or
 * Punctuation behaviour, an unknown Metaspace prepend scheme, a Strip after Fuse that strips the
 * end of a streamed text, WordPiece or BPE model settings the encoder does not implement, and
 * AddedToken flags at a value that changes what matches. A flag at its neutral value, which most
 * files spell out, loads; lstrip/rstrip load with a warning naming the tokens, since refusing them
 * would refuse released models and the container cannot carry them. */
TEST(parse_tokenizer_json_refuses_values_it_cannot_honour) {
    auto bpe = [](const std::string& pre, const std::string& extra, const std::string& model_extra = "") {
        return "{\"pre_tokenizer\": " + pre + extra +
               ", \"model\": {\"type\": \"BPE\", \"vocab\": {\"a\": 0, \"b\": 1, \"<unk>\": 2}" +
               model_extra + ", \"merges\": []}}";
    };
    const std::string ws = R"({"type": "WhitespaceSplit"})";
    auto added = [](const char* flags) {
        return std::string(R"(, "added_tokens": [{"id": 3, "content": "X", "special": true, )") +
               flags + "}]";
    };
    const std::string refused[] = {
        bpe(R"({"type": "Split", "pattern": {"String": " "}, "behavior": "Bogus"})", ""),
        bpe(R"({"type": "Punctuation", "behavior": "Bogus"})", ""),
        bpe(R"({"type": "Metaspace", "replacement": "_", "prepend_scheme": "sometimes"})", ""),
        bpe(ws, R"(, "decoder": {"type": "Metaspace", "replacement": "_", "prepend_scheme": "sometimes"})"),
        bpe(ws, R"(, "decoder": {"type": "Sequence", "decoders": [{"type": "Fuse"},)"
                R"( {"type": "Strip", "content": " ", "start": 0, "stop": 1}]})"),
        bpe(ws, R"(, "decoder": {"type": "Sequence", "decoders": [{"type": "Fuse"},)"
                R"( {"type": "Strip", "content": " ", "start": 1, "stop": 0},)"
                R"( {"type": "Strip", "content": " ", "start": 1, "stop": 0}]})"),
        bpe(ws, "", R"(, "continuing_subword_prefix": "##")"),
        bpe(ws, "", R"(, "dropout": 0.1)"),
        bpe(ws, added(R"("single_word": true, "lstrip": false, "rstrip": false, "normalized": false)")),
        bpe(ws, R"(, "normalizer": {"type": "Lowercase"})" +
                added(R"("single_word": false, "lstrip": false, "rstrip": false, "normalized": true)")),
        R"({"pre_tokenizer": {"type": "WhitespaceSplit"}, "model": {"type": "WordPiece",)"
        R"( "continuing_subword_prefix": "@@", "vocab": {"a": 0}}})",
        R"({"pre_tokenizer": {"type": "WhitespaceSplit"}, "model": {"type": "WordPiece",)"
        R"( "max_input_chars_per_word": 50, "vocab": {"a": 0}}})",
    };
    for (const std::string& js : refused) {
        VocabBuild vb;
        const std::string err = stderr_of([&] {
            CHECK_EQ(parse_tokenizer_json_buf(js, "test.json", vb), RAD_E_UNSUPPORTED);
        });
        CHECK(!err.empty());
    }

    const std::string accepted[] = {
        bpe(ws, added(R"("single_word": false, "lstrip": false, "rstrip": false, "normalized": false)")),
        bpe(ws, added(R"("single_word": false, "lstrip": false, "rstrip": false, "normalized": true)")),
        bpe(ws, R"(, "normalizer": {"type": "Lowercase"})" +
                added(R"("single_word": false, "lstrip": false, "rstrip": false, "normalized": false)")),
        bpe(ws, R"(, "decoder": {"type": "Sequence", "decoders": [{"type": "Fuse"},)"
                R"( {"type": "Strip", "content": " ", "start": 1, "stop": 0}]})"),
        bpe(ws, "", R"(, "dropout": null, "continuing_subword_prefix": null, "end_of_word_suffix": null)"),
        bpe(ws, "", R"(, "continuing_subword_prefix": "", "end_of_word_suffix": "</w>")"),
        /* Lower case already, so the Lowercase normaliser leaves its spelling alone. */
        bpe(ws, R"(, "normalizer": {"type": "Lowercase"}, "added_tokens": [{"id": 3, "content": "x",)"
                R"( "special": true, "single_word": false, "lstrip": false, "rstrip": false,)"
                R"( "normalized": true}])"),
    };
    for (const std::string& js : accepted) {
        VocabBuild vb;
        CHECK_OK(parse_tokenizer_json_buf(js, "test.json", vb));
    }

    VocabBuild vb;
    const std::string warned = stderr_of([&] {
        CHECK_OK(parse_tokenizer_json_buf(
            bpe(ws, added(R"("single_word": false, "lstrip": true, "rstrip": false, "normalized": false)")),
            "test.json", vb));
    });
    CHECK(warned.find("lstrip") != std::string::npos);
    CHECK(warned.find("X") != std::string::npos);

    /* The model's unk_token names the unknown token, as tokenizer_config.json would. */
    CHECK_OK(parse_tokenizer_json_buf(bpe(ws, "", R"(, "unk_token": "<unk>")"), "test.json", vb));
    CHECK_EQ(vb.unk, 2);
}

/* ================================================================== end of generation */

/* A Qwen-shaped vocabulary: eos is <|im_end|> and <|endoftext|> sits in pad, a control token the
 * header does not call terminal; a turn ending on it would otherwise run on. A control token with
 * a near name (<|end_of_thinking|>) and a known name typed NORMAL (text a reply may write) never
 * end a generation. */
TEST(vocab_ends_a_qwen_turn_on_both_markers_and_nothing_else) {
    VocabBuild vb;
    vb.kind   = RAD_TOK_BPE;
    vb.tokens = { "a", "<|im_end|>", "<|endoftext|>", "<|end_of_thinking|>", "<|eot_id|>" };
    vb.types  = { RAD_TT_NORMAL, RAD_TT_CONTROL, RAD_TT_CONTROL, RAD_TT_CONTROL, RAD_TT_NORMAL };
    vb.eos = 1;
    vb.pad = 2;
    auto v = std::make_shared<Vocab>();
    CHECK_OK(v->load(std::move(vb)));
    CHECK(v->is_eog(1));
    CHECK(v->is_eog(2));
    CHECK(!v->is_eog(3));
    CHECK(!v->is_eog(4));
    CHECK(!v->is_eog(0));
}

/* ================================================================== GGUF vocabularies */

namespace {

/* A GGUF holding only tokenizer metadata, for vocab_from_gguf. Each record writes itself once the
 * count is known. */
struct TokGguf {
    std::vector<std::function<void(GgufBytes&)>> kvs;
    void str(const char* k, const std::string& v) {
        kvs.push_back([k, v](GgufBytes& f) { f.str(k); f.pod<uint32_t>(GGUF_STR); f.str(v); });
    }
    void strs(const char* k, const std::vector<std::string>& v) {
        kvs.push_back([k, v](GgufBytes& f) {
            f.str(k); f.pod<uint32_t>(GGUF_ARR); f.pod<uint32_t>(GGUF_STR); f.pod<uint64_t>(v.size());
            for (const std::string& s : v) f.str(s);
        });
    }
    void i32s(const char* k, const std::vector<int32_t>& v) {
        kvs.push_back([k, v](GgufBytes& f) {
            f.str(k); f.pod<uint32_t>(GGUF_ARR); f.pod<uint32_t>(GGUF_I32); f.pod<uint64_t>(v.size());
            for (int32_t x : v) f.pod<int32_t>(x);
        });
    }
    void f32s(const char* k, const std::vector<float>& v) {
        kvs.push_back([k, v](GgufBytes& f) {
            f.str(k); f.pod<uint32_t>(GGUF_ARR); f.pod<uint32_t>(GGUF_F32); f.pod<uint64_t>(v.size());
            for (float x : v) f.pod<float>(x);
        });
    }
    void u32(const char* k, uint32_t v) {
        kvs.push_back([k, v](GgufBytes& f) { f.str(k); f.pod<uint32_t>(GGUF_U32); f.pod<uint32_t>(v); });
    }
    int build(VocabBuild* out) const {
        GgufBytes f;
        f.header(0, kvs.size());
        for (const auto& kv : kvs) kv(f);
        while (f.b.size() % 32) f.b.push_back(0);   /* the (empty) data section's alignment */
        GgufFile g;
        const int st = f.open(&g);
        if (st < 0) return st;
        return vocab_from_gguf(g, *out);
    }
};

TokGguf tok_gguf(const char* model, const char* pre) {
    TokGguf t;
    t.str("tokenizer.ggml.model", model);
    if (pre) t.str("tokenizer.ggml.pre", pre);
    t.strs("tokenizer.ggml.tokens", { "a", "b", "ab", "<s>", "</s>" });
    t.i32s("tokenizer.ggml.token_type", { 1, 1, 1, 3, 3 });
    t.u32("tokenizer.ggml.bos_token_id", 3);
    t.u32("tokenizer.ggml.eos_token_id", 4);
    return t;
}

/* Every piece the reconstructed chain's Split cuts `text` into, over the text's own bytes. */
std::vector<std::string> gguf_split(const VocabBuild& vb, const std::string& text) {
    std::vector<std::string> pieces = { text };
    for (const VocabStep& s : vb.steps) {
        if (s.kind != RAD_PRE_SPLIT) continue;
        std::unique_ptr<TokRegex> re;
        std::string why;
        if (TokRegex::compile(s.arg0, &re, &why) < 0) return {};
        std::vector<std::string> next;
        for (const std::string& p : pieces) {
            std::vector<TokSpan> out, scratch;
            tok_split(*re, p, (int)s.iarg, false, out, scratch);
            for (const TokSpan& sp : out) next.push_back(p.substr(sp.begin, sp.end - sp.begin));
        }
        pieces.swap(next);
    }
    return pieces;
}

}  /* namespace */

/* GGUF carries no chain, only `tokenizer.ggml.pre`, so the chain is rebuilt from a table of names:
 * byte-level BPE with the named pattern, SentencePiece-shaped BPE, unigram and WordPiece, with the
 * header's ids and merges. A name not in the table is refused rather than given a default split.
 * The qwen2 and llama-bpe entries must split as the patterns those models' tokenizer.json files
 * carry, apart from llama.cpp's rewrite of the case-insensitive contractions, which only "'"
 * before U+017F tells apart. */
TEST(gguf_vocabularies_are_rebuilt_from_the_pre_name) {
    {
        TokGguf t = tok_gguf("gpt2", "qwen2");
        t.strs("tokenizer.ggml.merges", { "a b", "x y" });
        VocabBuild vb;
        CHECK_OK(t.build(&vb));
        CHECK_EQ(vb.kind, (uint32_t)RAD_TOK_BPE);
        CHECK_EQ(vb.tokens.size(), (size_t)5);
        CHECK_EQ(vb.merges.size(), (size_t)1);       /* "x y" names no token */
        CHECK_EQ(vb.bos, 3);
        CHECK_EQ(vb.eos, 4);
        CHECK_EQ(vb.eot, 4);
        REQUIRE_EQ(vb.steps.size(), (size_t)3);
        CHECK_EQ(vb.steps[0].kind, (uint32_t)RAD_PRE_SPLIT);
        CHECK_EQ(vb.steps[1].kind, (uint32_t)RAD_PRE_BYTELEVEL);
        CHECK_EQ(vb.steps[2].kind, (uint32_t)RAD_DEC_BYTELEVEL);
        const char* hf_qwen2 =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+"
            "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
        tokfix::TextGen gen(17);
        for (int i = 0; i < 40; ++i) {
            std::string text = gen.text(60, false, [](uint32_t c) { return c == 0x17F; });
            for (size_t q; (q = text.find("'\xc5\xbf")) != std::string::npos; ) text.replace(q, 1, "_");
            VocabBuild hf;
            hf.steps = { tokfix::split_step(hf_qwen2) };
            const auto a = gguf_split(vb, text), b = gguf_split(hf, text);
            CHECK(a == b);
            if (a != b) {
                std::string x, y;
                for (const auto& p : a) x += "|" + p;
                for (const auto& p : b) y += "|" + p;
                fprintf(stderr, "    gguf %s\n    json %s\n", x.c_str(), y.c_str());
            }
        }
    }
    {
        TokGguf t = tok_gguf("gpt2", "llama-bpe");
        VocabBuild vb;
        CHECK_OK(t.build(&vb));
        tokfix::TextGen gen(19);
        for (int i = 0; i < 40; ++i) {
            std::string text = gen.text(60, false, [](uint32_t c) { return c == 0x17F; });
            for (size_t q; (q = text.find("'\xc5\xbf")) != std::string::npos; ) text.replace(q, 1, "_");
            VocabBuild hf;
            hf.steps = { tokfix::split_step(tokfix::k_llama3) };
            CHECK(gguf_split(vb, text) == gguf_split(hf, text));
        }
    }
    {
        TokGguf t = tok_gguf("gpt2", "no-such-pre");
        VocabBuild vb;
        const std::string err = stderr_of([&] { CHECK_EQ(t.build(&vb), RAD_E_UNSUPPORTED); });
        CHECK(err.find("no-such-pre") != std::string::npos);
    }
    {
        /* SentencePiece as BPE: Metaspace in and out, byte fallback. */
        TokGguf t = tok_gguf("llama", nullptr);
        VocabBuild vb;
        CHECK_OK(t.build(&vb));
        REQUIRE_EQ(vb.steps.size(), (size_t)3);
        CHECK_EQ(vb.steps[0].kind, (uint32_t)RAD_PRE_METASPACE);
        CHECK_EQ(vb.steps[1].kind, (uint32_t)RAD_PRE_BYTE_FALLBACK);
        CHECK_EQ(vb.steps[2].kind, (uint32_t)RAD_DEC_METASPACE);
        CHECK_EQ(vb.steps[0].arg0, std::string("\xe2\x96\x81"));
        CHECK_EQ(vb.steps[0].iarg, (int64_t)2);
    }
    {
        TokGguf t = tok_gguf("t5", nullptr);
        t.f32s("tokenizer.ggml.scores", { -1.0f, -2.0f, -0.5f, 0.0f, 0.0f });
        VocabBuild vb;
        CHECK_OK(t.build(&vb));
        CHECK_EQ(vb.kind, (uint32_t)RAD_TOK_UNIGRAM);
        CHECK_EQ(vb.scores.size(), (size_t)5);
        auto v = std::make_shared<Vocab>();
        CHECK_OK(v->load(std::move(vb)));
    }
    {
        /* BERT: lowercase, BertPreTokenizer, the WordPiece decoder. */
        TokGguf t = tok_gguf("bert", nullptr);
        VocabBuild vb;
        CHECK_OK(t.build(&vb));
        CHECK_EQ(vb.kind, (uint32_t)RAD_TOK_WORDPIECE);
        REQUIRE_EQ(vb.steps.size(), (size_t)3);
        CHECK_EQ(vb.steps[0].kind, (uint32_t)RAD_NORM_LOWERCASE);
        CHECK_EQ(vb.steps[1].kind, (uint32_t)RAD_PRE_WHITESPACE);
        CHECK_EQ(vb.steps[1].flags, (uint32_t)kPreBert);
        CHECK_EQ(vb.steps[2].kind, (uint32_t)RAD_DEC_REPLACE);
        CHECK((vb.steps[2].flags & kDecWordPiece) != 0);
    }
}

/* ================================================================== normalisation, independently */

/* The four normal forms against CPython's unicodedata, an implementation independent of the ICU
 * the tables were generated from (tests/mk_norm_vectors.py): a third of the codepoints with a
 * decomposition, every pair that composes, Hangul, and marks in shuffled canonical classes after
 * a base. A regenerated table that is wrong but self-consistent passes the inverse checks above
 * and fails here. The tokeniser's NFC shortcut must give the same text. */
TEST(the_normal_forms_agree_with_an_independent_implementation) {
    auto utf8_of = [](const std::string& hex) {
        std::string out;
        size_t i = 0;
        while (i < hex.size()) {
            size_t j = hex.find(' ', i);
            if (j == std::string::npos) j = hex.size();
            out += tokfix::cp_utf8((uint32_t)std::stoul(hex.substr(i, j - i), nullptr, 16));
            i = j + 1;
        }
        return out;
    };
    int rows = 0, bad = 0;
    for (const char* row : kNormVectors) {
        std::vector<std::string> f;
        std::string r = row;
        for (size_t i = 0, j; i <= r.size(); i = j + 1) {
            j = r.find(';', i);
            if (j == std::string::npos) j = r.size();
            f.push_back(utf8_of(r.substr(i, j - i)));
        }
        REQUIRE_EQ(f.size(), (size_t)5);
        std::string nfc = f[0];
        tok_nfc(nfc);
        const bool ok = unorm_utf8(f[0], false, true) == f[1] && unorm_utf8(f[0], false, false) == f[2] &&
                        unorm_utf8(f[0], true, true) == f[3] && unorm_utf8(f[0], true, false) == f[4] &&
                        nfc == f[1];
        CHECK(ok);
        if (!ok && ++bad <= 5) fprintf(stderr, "    %s\n", row);
        ++rows;
    }
    CHECK(rows > 2500);
}

/* --chat-template serves a release that ships no chat template (DeepSeek's): the file is read,
 * trailing newlines stripped exactly as the checkpoint's own file is, and an unreadable or empty
 * file is refused rather than skipped. */
TEST(read_chat_template_path_strips_newlines_and_refuses_empty) {
    VocabBuild vb;
    const std::string p = radtest::tmp_path("chat_template.jinja");
    {
        std::ofstream f(p);
        f << "{%- raw %}{{ messages }}{%- endraw %}\n\n";
    }
    CHECK_OK(read_chat_template_path(p.c_str(), vb));
    CHECK_EQ(vb.chat_template, std::string("{%- raw %}{{ messages }}{%- endraw %}"));

    vb.chat_template.clear();
    {
        std::ofstream f(p);
        f << "\n\r\n";
    }
    CHECK_EQ(read_chat_template_path(p.c_str(), vb), (int)RAD_E_IO);
    CHECK(vb.chat_template.empty());

    CHECK_EQ(read_chat_template_path(radtest::tmp_path("absent.jinja").c_str(), vb), (int)RAD_E_IO);
    std::remove(p.c_str());
}

RAD_TEST_MAIN()
