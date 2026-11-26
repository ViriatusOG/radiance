/* chatparse_gemma4_test.cpp -- Gemma 4's reply format: a reasoning channel, and calls written as a
 * name and one object in a JSON dialect.
 *
 * llama.cpp reads Gemma 4's template with a hand-written handler, so template analysis finds no
 * format and the plugins declare one (arch/common/rad_gemma4_chat.h):
 *
 *   <|channel>thought\n ... <channel|>                         reasoning
 *   <|tool_call>call:NAME{key:<|"|>text<|"|>,n:3}<tool_call|>   a call, repeated for more
 *
 * Each case reads a reply under a request rendered through the real template
 * (tests/data/chat/gemma4). Where the reply is within the format, the result must match llama.cpp's
 * Gemma 4 parser byte for byte (tests/legacy, the `old` side). Every reply is also streamed at
 * every cut (chatdiff.h check_stream): each delta a prefix of the final value, the deltas summing
 * to it, held bytes within the format's bound, and a closed call's arguments valid JSON.
 *
 * The object family in general (JSON behind a name or function tag) is chatparse_test's.
 */
#include "rad_test.h"

#include "chatdiff.h"
#include "rad_builder.h"
#include "sample/gbnf.h"
#include "sample/vocab_view.h"
#include "server/adapters.h"
#include "text/chat.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"
#include "text/tokenizer.h"

#include <arch/rad_gemma4_chat.h>

#include "util/json.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace rad;
using namespace rad::chatdiff;
using json = Json;

/* v with every object's members in key order, so two values compare whatever order their keys
 * were written in. */
static json keys_sorted(const json& v) {
    if (v.is_array()) {
        json out = json::array();
        for (const json& e : v) out.push_back(keys_sorted(e));
        return out;
    }
    if (!v.is_object()) return v;
    std::vector<std::pair<std::string, json>> m;
    for (auto it = v.begin(); it != v.end(); ++it) m.emplace_back(it.key(), keys_sorted(it.value()));
    std::sort(m.begin(), m.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    return json(json::object_t(std::move(m)));
}

/* ================================================================== fixtures */

static std::string fixture() {
    std::ifstream f(std::string(RAD_CHAT_FIXTURES) + "/gemma4/chat_template.jinja");
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/* Turn terminators, per generation_config.json: <eos>, <turn|> and <|tool_response>. */
static const Model& gemma() {
    static const Model m = { "gemma4", "gemma4", "<bos>", "<eos>",
                             { "<eos>", "<turn|>", "<|tool_response>" },
                             rad::arch::gemma4_chat_format() };
    return m;
}

static Bench* bench(const ChatRequest& rq, const json& tools = agent_tools()) {
    static std::map<std::string, std::unique_ptr<Bench>> cache;
    const std::string key = describe(rq) + "|" + tools.dump();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second.get();
    auto b = std::make_unique<Bench>();
    std::string why;
    if (b->load(gemma(), fixture(), &why) < 0 || b->bind(rq, tools, &why) < 0) {
        ::radtest::fail(__FILE__, __LINE__, "gemma4 (" + describe(rq) + "): " + why);
        return nullptr;
    }
    return (cache[key] = std::move(b)).get();
}

static ChatRequest thinking(bool on) {
    ChatRequest r;
    r.thinking = on;
    return r;
}

static std::string quote(const std::string& s) {
    return json(s).dump(-1, ' ', false, json::error_handler_t::replace);
}

/* One call as the model writes it; `args` is the object's body. */
static std::string G(const std::string& name, const std::string& args) {
    return "<|tool_call>call:" + name + "{" + args + "}<tool_call|>";
}
static const std::string Q = "<|\"|>";
static std::string S(const std::string& text) { return Q + text + Q; }

/* ================================================================== the streaming checks */

static std::vector<size_t> every(size_t n, size_t step) {
    std::vector<size_t> c;
    for (size_t i = step; i < n; i += step) c.push_back(i);
    return c;
}

/* Every cut of the reply against its one-shot parse: whole, byte by byte, token-sized pieces, and
 * every two-way split. */
static std::string stream_all(const ReplyParser& p, const std::string& text) {
    const ReplyMessage expect = reply_parse(p, text);
    std::string err = check_stream(p, text, {}, expect);
    if (err.empty()) err = check_stream(p, text, every(text.size(), 1), expect);
    std::mt19937 rng(11);
    for (int k = 0; err.empty() && k < 3; ++k) {
        std::vector<size_t> c;
        for (size_t at = 1 + rng() % 5; at < text.size(); at += 1 + rng() % 7) c.push_back(at);
        err = check_stream(p, text, c, expect);
    }
    for (size_t k = 1; err.empty() && k < text.size() && text.size() <= 2048; ++k)
        err = check_stream(p, text, { k }, expect);
    return err;
}

static void streams(const ReplyParser& p, const std::string& text, const std::string& what) {
    const std::string err = stream_all(p, text);
    if (!err.empty()) ::radtest::fail(__FILE__, __LINE__, what + ": " + err + "\n  reply: " + quote(text));
    ++::radtest::checks();
}

static ReplyMessage parse(const ReplyParser& p, const std::string& text, bool* strict = nullptr) {
    return reply_parse(p, text, strict);
}

static std::shared_ptr<const ReplyParser> parser(const std::string& gp, bool tools = true,
                                                 bool reasoning = true, bool parallel = true) {
    static std::shared_ptr<const ReplyFormat> rf = [] {
        ChatFormat f;
        std::string why;
        std::shared_ptr<const ReplyFormat> r;
        if (chat_format_from_abi(rad::arch::gemma4_chat_format(), &f, &why) < 0 ||
            ReplyFormat::compile(f, &r, &why) < 0)
            ::radtest::fail(__FILE__, __LINE__, "gemma4 format: " + why);
        return r;
    }();
    ReplyOptions ro;
    ro.generation_prompt   = gp;
    ro.extract_reasoning   = reasoning;
    ro.tools               = tools ? &agent_tools() : nullptr;
    ro.parallel_tool_calls = parallel;
    ro.terminators         = gemma().terminators;
    std::shared_ptr<const ReplyParser> p;
    std::string why;
    if (ReplyParser::create(rf, ro, &p, &why) < 0)
        ::radtest::fail(__FILE__, __LINE__, "gemma4 parser: " + why);
    return p;
}

/* The three generation prompts: thinking off (the template closes an empty channel), thinking on
 * (the model opens its own), and the continuation after a tool response with thinking on (the
 * template has opened the channel). With thinking off a continuation has no generation prompt. */
static const std::string kOff   = "<|turn>model\n<|channel>thought\n<channel|>";
static const std::string kOn    = "<|turn>model\n";
static const std::string kAfter = "<|channel>thought\n";

/* ================================================================== the declaration */

TEST(the_declaration_is_a_format_the_parser_runs) {
    ChatFormat f;
    std::string why;
    CHECK_OK(chat_format_from_abi(rad::arch::gemma4_chat_format(), &f, &why));
    CHECK_OK(chat_format_check(f, &why));
    CHECK(f.object_args());
    CHECK(f.reasoning_anywhere);
    CHECK_EQ(f.object_string_open, Q);
    CHECK_EQ(chat_format_describe(f),
             std::string("plugin:gemma4: reasoning \"<|channel>thought\\n\"..\"<channel|>\" "
                         "anywhere; calls \"<|tool_call>\"\"call:\"NAME\"\" {OBJECT, strings "
                         "\"<|\\\"|>\"..\"<|\\\"|>\", bare keys} \"\"\"<tool_call|>\"; grammar on "
                         "\"<|tool_call>\""));
    /* The decoder must render every delimiter as text, including the string delimiter, or string
     * arguments lose their quotes. */
    const std::vector<std::string> m = chat_format_markers(f);
    for (const char* want : { "<|channel>thought", "<channel|>", "<|tool_call>", "<tool_call|>",
                              "call:", "<|\"|>" })
        CHECK(std::find(m.begin(), m.end(), std::string(want)) != m.end());
}

/* A plugin built against the header before the object family reads the new fields as zero (the
 * tagged family, no dialect, reasoning at the start only); Gemma's declaration read that way is
 * refused by name. */
TEST(an_older_struct_reads_the_object_fields_as_zero) {
    RadChatFormat c = *rad::arch::gemma4_chat_format();
    c.struct_size = (uint32_t)offsetof(RadChatFormat, args_form);
    ChatFormat f;
    std::string why;
    CHECK_OK(chat_format_from_abi(&c, &f, &why));
    CHECK(!f.object_args());
    CHECK(f.object_string_open.empty());
    CHECK(!f.object_bare_keys);
    CHECK(!f.reasoning_anywhere);
    CHECK(chat_format_check(f, &why) == RAD_E_UNSUPPORTED);
    CHECK(why.find("call_end") != std::string::npos);
}

TEST(an_object_format_the_parser_cannot_read_is_refused_by_name) {
    auto refused = [](void (*edit)(RadChatFormat&), const char* word) {
        RadChatFormat c = *rad::arch::gemma4_chat_format();
        edit(c);
        ChatFormat f;
        std::string why;
        if (chat_format_from_abi(&c, &f, &why) < 0) return false;
        const bool no = chat_format_check(f, &why) == RAD_E_UNSUPPORTED;
        if (!no || why.find(word) == std::string::npos)
            ::radtest::fail(__FILE__, __LINE__, std::string("expected a refusal naming '") +
                            word + "', got: " + (no ? why : "accepted"));
        return no;
    };
    CHECK(refused([](RadChatFormat& c) { c.arg_prefix = "<arg>"; }, "arg_*"));
    CHECK(refused([](RadChatFormat& c) { c.object_string_close = nullptr; }, "together"));
    CHECK(refused([](RadChatFormat& c) { c.object_string_open = c.object_string_close = " <q>"; },
                  "whitespace"));
    CHECK(refused([](RadChatFormat& c) { c.object_string_open = "q"; }, "bare word"));
    CHECK(refused([](RadChatFormat& c) {
        c.object_string_open = c.object_string_close = "<q>"; c.call_close = "<q>"; }, "closes the call"));
    CHECK(refused([](RadChatFormat& c) { c.call_close = "<tc,>"; }, "punctuation"));
    CHECK(refused([](RadChatFormat& c) { c.call_close = "<|"; }, "told apart"));
    CHECK(refused([](RadChatFormat& c) { c.args_form = 7; }, "RAD_CHAT_ARGS"));
    CHECK(refused([](RadChatFormat& c) { c.reasoning_open = ""; }, "reasoning_anywhere"));
    CHECK(refused([](RadChatFormat& c) { c.call_open = "<|channel>thought"; c.reasoning_breaks = nullptr;
                                         c.n_reasoning_breaks = 0; }, "content"));
    /* In the tagged family a dialect field is refused, not ignored. */
    CHECK(refused([](RadChatFormat& c) {
        c.args_form = RAD_CHAT_ARGS_TAGGED;
        c.name_suffix = ">"; c.call_end = "</f>"; c.arg_prefix = "<a="; c.arg_name_suffix = ">";
        c.arg_value_suffix = "</a>"; }, "object_*"));
}

/* ================================================================== the template's own replies
 *
 * An assistant message rendered through the template is what the model is trained to write. With
 * thinking on, the reply is what the template writes after the generation prompt; with it off,
 * a past turn omits the empty channel the generation prompt holds, so the hand-written cases below
 * cover those. */
TEST(rendered_replies_read_back_as_the_template_wrote_them) {
    Bench* b = bench(thinking(true));
    REQUIRE(b != nullptr);
    int read = 0;
    for (const json& msg : rendered_corpus()) {
        const std::string reply = b->render_reply(msg);
        REQUIRE(!reply.empty());
        bool strict = false;
        const ReplyMessage m = b->new_parse(reply, &strict);
        const legacy::Answer o = b->old_parse(reply);

        /* The calls come back as in the message, arguments as JSON values. */
        const size_t ncalls = msg.contains("tool_calls") ? msg["tool_calls"].size() : 0;
        REQUIRE_EQ(m.calls.size(), ncalls);
        for (size_t i = 0; i < ncalls; ++i) {
            const json& want = msg["tool_calls"][i]["function"];
            CHECK_EQ(m.calls[i].name, want["name"].get<std::string>());
            CHECK(m.calls[i].closed);
            /* Compared as values: the template writes keys in sorted order. */
            CHECK(keys_sorted(Json::parse(m.calls[i].arguments)) ==
                  keys_sorted(Json::parse(want["arguments"].get<std::string>())));
        }
        const std::string r = msg.contains("reasoning_content") ? msg["reasoning_content"].get<std::string>() : "";
        CHECK_EQ(m.reasoning, r.empty() ? r : r + "\n");
        CHECK_EQ(m.content, msg["content"].get<std::string>());

        /* llama.cpp's parser reads the same calls and reasoning byte for byte. Where the template
         * writes content after the calls (its past-turn order, not the model's), the old parser
         * drops it and the new one keeps it. */
        REQUIRE_EQ(o.calls.size(), m.calls.size());
        for (size_t i = 0; i < std::min(o.calls.size(), m.calls.size()); ++i) {
            CHECK_EQ(o.calls[i].name, m.calls[i].name);
            CHECK_EQ(o.calls[i].arguments, m.calls[i].arguments);
        }
        CHECK_EQ(o.reasoning, m.reasoning);
        const bool after = ncalls > 0 && !msg["content"].get<std::string>().empty();
        CHECK_EQ(strict, !after);
        if (!after) CHECK_EQ(o.content, m.content);
        streams(b->parser(), reply, "rendered reply");
        ++read;
    }
    CHECK(read > 10);
}

/* Every prefix of a rendered reply (the decoder stopping anywhere) streams to its own one-shot
 * parse, and a call cut off mid-way reads back as its text. */
TEST(every_truncation_of_a_rendered_reply_keeps_the_contract) {
    Bench* b = bench(thinking(true));
    REQUIRE(b != nullptr);
    for (const json& msg : rendered_corpus()) {
        const std::string reply = b->render_reply(msg);
        for (size_t n = 1; n <= reply.size(); n += 1 + n / 64) {
            const std::string cut = sanitize(reply.substr(0, n));
            const std::string err = check_stream(b->parser(), cut, every(cut.size(), 3),
                                                 reply_parse(b->parser(), cut));
            if (!err.empty())
                ::radtest::fail(__FILE__, __LINE__, err + "\n  reply: " + quote(cut));
            std::unique_ptr<ReplyStream> s = b->parser().open();
            s->feed(cut, nullptr);
            s->finish(nullptr);
            const ReplyMessage& m = s->message();
            if (!m.calls.empty() && !m.calls.back().closed) {
                const std::string text = s->content_with_open_call(cut);
                CHECK(text.find("<|tool_call>call:") != std::string::npos);
            }
        }
        ++::radtest::checks();
    }
}

/* ================================================================== the decided shapes
 *
 * Replies as the model writes them, including ways it leaves its format. Each reads as stated,
 * streams under the contract, and where `oracle` is set matches llama.cpp's parser byte for byte. */

struct Call { std::string name, args; bool closed = true; };
struct Shape {
    const char* what;
    std::string gp;
    std::string reply;
    std::string content, reasoning;
    std::vector<Call> calls;
    bool strict;
    bool oracle;          /* llama.cpp's parser reads it the same */
};

static std::vector<Shape> shapes() {
    const std::string ls = G("bash", "command:" + S("ls"));
    const std::string LS = "{\"command\":\"ls\"}";
    std::vector<Shape> v = {
        { "an answer", kOff, "The answer is 4.", "The answer is 4.", "", {}, true, true },
        { "a call", kOff, ls, "", "", { { "bash", LS } }, true, true },
        { "a sentence, then a call", kOff, "Let me look.\n" + ls, "Let me look.", "",
          { { "bash", LS } }, true, true },
        { "two calls", kOff, ls + G("read", "limit:40,path:" + S("/a")), "", "",
          { { "bash", LS }, { "read", "{\"limit\":40,\"path\":\"/a\"}" } }, true, true },
        { "two calls on two lines", kOff, ls + "\n" + G("read", "path:" + S("/a")), "", "",
          { { "bash", LS }, { "read", "{\"path\":\"/a\"}" } }, true, false },
        { "every kind of value", kOff,
          G("configure", "extra:{a:[1,2,[]],b:{c:null},d:{}},label:" + S("") +
                         ",level:-1.5e3,mode:" + S("fast") + ",tags:[" + S("x") + "," + S("y") +
                         "],verbose:false"),
          "", "", { { "configure", "{\"extra\":{\"a\":[1,2,[]],\"b\":{\"c\":null},\"d\":{}},"
                                   "\"label\":\"\",\"level\":-1.5e3,\"mode\":\"fast\","
                                   "\"tags\":[\"x\",\"y\"],\"verbose\":false}" } }, true, true },
        { "a call that takes no arguments", kOff, G("ping", ""), "", "", { { "ping", "{}" } },
          true, true },
        { "a string holding what JSON escapes", kOff,
          G("write", "content:" + S("a \"q\" \\ b\n\tc\x01 é 🚀") + ",path:" + S("/x")), "", "",
          { { "write", "{\"content\":\"a \\\"q\\\" \\\\ b\\n\\tc\\u0001 é 🚀\",\"path\":\"/x\"}" } },
          true, true },
        { "a string holding the format's own punctuation", kOff,
          G("write", "content:" + S("{a:1,} [<|tool_call>call:x()<tool_call|>] <channel|> <|\" .") +
                     ",path:" + S("/x")), "", "",
          { { "write", "{\"content\":\"{a:1,} [<|tool_call>call:x()<tool_call|>] <channel|> <|\\\" .\","
                       "\"path\":\"/x\"}" } }, true, true },
        /* Text ending in the start of the delimiter reads as written (no grammar of this shape,
         * llama.cpp's included, can produce it). */
        { "a string ending in the start of its own delimiter", kOff,
          G("write", "content:" + S("a <|\"") + ",path:" + S("/x")), "", "",
          { { "write", "{\"content\":\"a <|\\\"\",\"path\":\"/x\"}" } }, true, true },
        /* llama.cpp's key rule keeps the delimiters, yielding the key `<|"|>path<|"|>`. */
        { "a key written as a string", kOff, G("read", S("path") + ":" + S("/a")), "", "",
          { { "read", "{\"path\":\"/a\"}" } }, true, false },
        /* It also keeps the blank after a number, `"limit":3 }`, which is not JSON. */
        { "whitespace inside the object", kOff, G("read", " path: " + S("/a") + ",\n limit: 3 "),
          "", "", { { "read", "{\"path\":\"/a\",\"limit\":3}" } }, true, false },

        /* reasoning */
        { "thinking on: a thought, then an answer", kOn,
          "<|channel>thought\nIt is two plus two.\n<channel|>The answer is 4.", "The answer is 4.",
          "It is two plus two.\n", {}, true, true },
        { "thinking on: no thought at all", kOn, "The answer is 4.", "The answer is 4.", "", {},
          true, true },
        { "thinking on: a thought, then a call", kOn,
          "<|channel>thought\nList it.\n<channel|>" + ls, "", "List it.\n", { { "bash", LS } }, true,
          true },
        { "a thought never closed before a call", kOn, "<|channel>thought\nList it.\n" + ls, "",
          "List it.\n", { { "bash", LS } }, true, false },
        { "after a tool response, thinking on", kAfter,
          "The tool says 18.\n<channel|>It is 18 degrees.", "It is 18 degrees.",
          "The tool says 18.\n", {}, true, false },
        { "after a tool response, thinking off", "", "It is 18 degrees.", "It is 18 degrees.", "",
          {}, true, false },
        { "an empty thought after the answer", kOff, "The answer is 4.<|channel>thought\n<channel|>",
          "The answer is 4.", "", {}, true, true },
        { "a thought in the middle of the answer", kOff,
          "First.\n<|channel>thought\nhm\n<channel|>Second.", "First.\nSecond.", "hm\n", {}, true,
          true },
        { "a close with no thought open", kOff, "The answer is 4.<channel|>", "The answer is 4.", "",
          {}, false, false },
        { "the turn's terminator left on", kOff, "The answer is 4.<turn|>", "The answer is 4.", "",
          {}, true, false },

        /* leaving the object */
        { "a JSON string", kOff, G("bash", "command:\"ls -la\""), "", "",
          { { "bash", "{\"command\":\"ls -la\"}" } }, false, false },
        /* Gemma's plain-quoted values escape at most the quotes (the model writes
         * `code: "const s = \"a\nb\";"` for the JS `const s = "a\nb";`): an escaped quote is a
         * quote, every other backslash is text. */
        { "a plain-quoted value with its quotes escaped", kOff,
          G("bash", "command:\"a\\\"b\\\\c\\u00e9\\n\\q\""), "", "",
          { { "bash", "{\"command\":\"a\\\"b\\\\\\\\c\\\\u00e9\\\\n\\\\q\"}" } }, false,
          false },
        { "a plain-quoted value written verbatim", kOff,
          G("write", "content:\"const s = \"a\\nb\"; f(\"x\", {y: 1});\",path:" + S("/x")), "", "",
          { { "write", "{\"content\":\"const s = \\\"a\\\\nb\\\"; f(\\\"x\\\", {y: 1});\",\"path\":\"/x\"}" } },
          false, false },
        /* For nested arguments the model writes Python-style quotes. */
        { "single-quoted values, as served", kOff,
          G("configure", "label: 'prod \"eu\"', extra: {note: 'x: {y}', tags: ['a,b', 'c}']}, mode: 'it\\'s'"),
          "", "", { { "configure", "{\"label\":\"prod \\\"eu\\\"\",\"extra\":{\"note\":\"x: {y}\",\"tags\":[\"a,b\",\"c}\"]},"
                                 "\"mode\":\"it's\"}" } }, false, false },
        { "a plain quote closed by the dialect's delimiter", kOff, G("bash", "command:\"ls" + Q), "",
          "", { { "bash", "{\"command\":\"ls\"}" } }, false, false },
        /* The model may leave a string unclosed before the object's `}` and the call's close
         * (observed: close delimiter at p=0.86, `}` at p=0.08). The close ends the string and the
         * call; the closers before it are the object's. */
        { "a string the model left open, as served", kOff,
          "<|tool_call>call:execute{code:" + Q + "const s = \"line1\\nline2\"; console.log(...);}"
          "<tool_call|><tool_call|><|channel>thought\n<channel|><channel|>"
          "<|tool_call>call:execute{code:" + Q + "const s = \"line1\\nline2\"; console.log(...);" + Q +
          "}<tool_call|>", "", "",
          { { "execute", "{\"code\":\"const s = \\\"line1\\\\nline2\\\"; console.log(...);\"}" },
            { "execute", "{\"code\":\"const s = \\\"line1\\\\nline2\\\"; console.log(...);\"}" } },
          false, false },
        { "a string left open whose code ends in a brace", kOff,
          "<|tool_call>call:bash{command:" + Q + "f() { return 1; }}<tool_call|>", "", "",
          { { "bash", "{\"command\":\"f() { return 1; }\"}" } }, false, false },
        { "a nested string left open", kOff,
          "<|tool_call>call:configure{extra:{a:[" + Q + "x]}\n}<tool_call|>" + G("bash", "command:" + S("ls")),
          "", "", { { "configure", "{\"extra\":{\"a\":[\"x\"]}}" }, { "bash", "{\"command\":\"ls\"}" } },
          false, false },
        /* Observed with thinking on: the close spelled as a plain quote, and text after the call. */
        { "a string closed by a stray quote, text after the call", kOn,
          "<|tool_call>call:bash{command:" + Q + "f(\"a\");\n\"}<tool_call|>end", "end", "",
          { { "bash", "{\"command\":\"f(\\\"a\\\");\\n\"}" } }, false, false },
        { "a string left open that ends in its own quote", kOff,
          "<|tool_call>call:bash{command:" + Q + "say \"hi\"}<tool_call|>", "", "",
          { { "bash", "{\"command\":\"say \\\"hi\\\"\"}" } }, false, false },
        { "a string left open at the end of the reply", kOff,
          "<|tool_call>call:bash{command:" + Q + "ls}<tool_call|>", "", "",
          { { "bash", "{\"command\":\"ls\"}" } }, false, false },
        { "code with quotes, backticks, a regex and braces", kOff,
          G("bash", "command:" + S("const r = /\\d+\"}/g; `${a}` <b> '{'") + ",x:1"), "", "",
          { { "bash", "{\"command\":\"const r = /\\\\d+\\\"}/g; `${a}` <b> '{'\",\"x\":1}" } }, true, true },
        { "Python literals", kOff, G("configure", "verbose:True,nothing:None"), "", "",
          { { "configure", "{\"verbose\":true,\"nothing\":null}" } }, false, false },
        { "a bare word", kOff, G("configure", "mode:fast,level:01"), "", "",
          { { "configure", "{\"mode\":\"fast\",\"level\":\"01\"}" } }, false, false },
        { "a trailing comma", kOff, G("read", "path:" + S("/a") + ","), "", "",
          { { "read", "{\"path\":\"/a\"}" } }, false, false },
        { "a missing comma", kOff, G("read", "path:" + S("/a") + " limit:3"), "", "",
          { { "read", "{\"path\":\"/a\",\"limit\":3}" } }, false, false },
        { "a key with no value", kOff, G("read", "path:,limit"), "", "",
          { { "read", "{\"path\":null,\"limit\":null}" } }, false, false },
        { "the wrong closing bracket", kOff, G("configure", "tags:[" + S("x") + "}"), "", "",
          { { "configure", "{\"tags\":[\"x\"]}" } }, false, false },
        { "an object the call's close cuts off", kOff,
          "<|tool_call>call:configure{extra:{a:[1,{b:" + S("c") + "<tool_call|>" + ls, "", "",
          { { "configure", "{\"extra\":{\"a\":[1,{\"b\":\"c\"}]}}" }, { "bash", LS } }, false, false },
        /* llama.cpp's grammar has no arm for it and reads no call. */
        { "a call to a function that is not a tool", kOff, G("frobnicate", "x:1"), "", "",
          { { "frobnicate", "{\"x\":1}" } }, false, false },
        { "a call written without its wrapper", kOff, "call:bash{command:" + S("ls") + "}", "", "",
          { { "bash", LS } }, false, false },
        { "the name prefix in prose", kOff, "This is what I call: a plan, and call:me{ maybe.",
          "This is what I call: a plan, and call:me{ maybe.", "", {}, true, true },
        { "a reply that ends inside a call", kOff,
          "<|tool_call>call:write{path:" + S("/a") + ",content:" + Q + "half", "", "",
          { { "write", "{\"path\":\"/a\",\"content\":\"half", false } }, true, false },
    };
    return v;
}

TEST(what_gemma_writes_reads_as_decided) {
    for (const Shape& s : shapes()) {
        std::shared_ptr<const ReplyParser> p = parser(s.gp);
        bool strict = false;
        const ReplyMessage m = parse(*p, s.reply, &strict);
        const std::string tag = std::string(s.what) + ": " + quote(s.reply) + "\n  got " + show(m);
        if (m.content != s.content || m.reasoning != s.reasoning || m.calls.size() != s.calls.size() ||
            strict != s.strict)
            ::radtest::fail(__FILE__, __LINE__, tag + (strict ? " (strict)" : " (lenient)"));
        for (size_t i = 0; i < std::min(m.calls.size(), s.calls.size()); ++i)
            if (m.calls[i].name != s.calls[i].name || m.calls[i].arguments != s.calls[i].args ||
                m.calls[i].closed != s.calls[i].closed)
                ::radtest::fail(__FILE__, __LINE__, tag);
        for (const ReplyCall& c : m.calls)
            if (c.closed && !json::accept(c.arguments))
                ::radtest::fail(__FILE__, __LINE__, tag + "\n  arguments are no JSON");
        streams(*p, s.reply, s.what);
    }
}

/* Where the reply is within the format, llama.cpp's Gemma 4 parser (run on generation prompt and
 * reply, as its handler does) reads exactly what this one reads. */
TEST(the_shapes_in_the_format_read_as_llama_cpp_reads_them) {
    int compared = 0;
    for (const Shape& s : shapes()) {
        if (!s.oracle) continue;
        const bool on = s.gp == kOn;
        Bench* b = bench(thinking(on));
        REQUIRE(b != nullptr);
        REQUIRE_EQ(b->prompt().generation_prompt, s.gp);
        const legacy::Answer o = b->old_parse(s.reply);
        const ReplyMessage m = b->new_parse(s.reply);
        bool same = !o.strict_failed && o.content == m.content && o.reasoning == m.reasoning &&
                    o.calls.size() == m.calls.size();
        for (size_t i = 0; same && i < o.calls.size(); ++i)
            same = o.calls[i].name == m.calls[i].name && o.calls[i].arguments == m.calls[i].arguments;
        if (!same)
            ::radtest::fail(__FILE__, __LINE__, std::string(s.what) + ": " + quote(s.reply) +
                            "\n    old: " + show(o) + "\n    new: " + show(m));
        ++compared;
    }
    CHECK(compared >= 15);
}

/* ================================================================== the request switches */

TEST(thinking_off_starts_the_reply_after_the_closed_channel) {
    Bench* b = bench(thinking(false));
    REQUIRE(b != nullptr);
    CHECK_EQ(b->prompt().generation_prompt, kOff);
    const ReasoningMarkers rm = b->parser().reasoning();
    CHECK_EQ(rm.open, std::string("<|channel>thought"));
    CHECK_EQ(rm.close, std::string("<channel|>"));
    CHECK(!rm.starts_open);

    Bench* on = bench(thinking(true));
    REQUIRE(on != nullptr);
    CHECK_EQ(on->prompt().generation_prompt, kOn);
    CHECK(!on->parser().reasoning().starts_open);
    /* The system turn carries the thinking switch. */
    CHECK(on->prompt().prompt.find("<|turn>system\n<|think|>\n") != std::string::npos);
    CHECK(b->prompt().prompt.find("<|think|>") == std::string::npos);

    CHECK(parser(kAfter)->reasoning().starts_open);
    CHECK(!parser("")->reasoning().starts_open);
}

/* reasoning_format "none": the channel stays in the content as written, and a call still ends
 * the text before it. */
TEST(reasoning_format_none_keeps_the_channel_in_the_content) {
    std::shared_ptr<const ReplyParser> p = parser(kOn, true, /*reasoning=*/false);
    const std::string reply = "<|channel>thought\nhm\n<channel|>Yes." +
                              G("bash", "command:" + S("ls"));
    const ReplyMessage m = parse(*p, reply);
    CHECK_EQ(m.content, std::string("<|channel>thought\nhm\n<channel|>Yes."));
    CHECK(m.reasoning.empty());
    REQUIRE_EQ(m.calls.size(), (size_t)1);
    streams(*p, reply, "reasoning none");
}

/* A request without tools reads call syntax as text, but the thought channel is still reasoning
 * wherever it opens. */
TEST(without_tools_a_call_is_text_and_a_late_channel_is_still_reasoning) {
    std::shared_ptr<const ReplyParser> p = parser(kOff, /*tools=*/false);
    const std::string reply = "Use <|tool_call>call:bash{} to run it.<|channel>thought\nx\n<channel|>";
    const ReplyMessage m = parse(*p, reply);
    CHECK_EQ(m.content, std::string("Use <|tool_call>call:bash{} to run it."));
    CHECK_EQ(m.reasoning, std::string("x\n"));
    CHECK(m.calls.empty());
    streams(*p, reply, "no tools");
}

TEST(parallel_tool_calls_false_still_reads_a_second_call) {
    std::shared_ptr<const ReplyParser> p = parser(kOff, true, true, /*parallel=*/false);
    bool strict = true;
    const ReplyMessage m = parse(*p, G("bash", "command:" + S("ls")) + G("bash", "command:" + S("pwd")), &strict);
    CHECK_EQ(m.calls.size(), (size_t)2);
    CHECK(!strict);
}

/* ================================================================== a conversation
 *
 * The turn after a call: the assistant message carries the call and a `role: tool` message its
 * result. The template writes both into the model's turn (the call in the model's spelling, the
 * result as a response block) and leaves the turn open. */
static json conversation(bool two_calls) {
    json calls = json::array();
    calls.push_back({ { "id", "c1" }, { "type", "function" },
                      { "function", { { "name", "read" }, { "arguments", "{\"path\":\"/a\",\"limit\":3}" } } } });
    if (two_calls)
        calls.push_back({ { "id", "c2" }, { "type", "function" },
                          { "function", { { "name", "bash" }, { "arguments", "{\"command\":\"ls\"}" } } } });
    json msgs = json::array({ { { "role", "user" }, { "content", "Read /a." } },
                              { { "role", "assistant" }, { "content", "" }, { "tool_calls", calls } },
                              { { "role", "tool" }, { "tool_call_id", "c1" }, { "content", "line one" } } });
    if (two_calls)
        msgs.push_back({ { "role", "tool" }, { "tool_call_id", "c2" }, { "content", "a.txt" } });
    return msgs;
}

TEST(a_tool_result_renders_into_the_open_turn_and_the_reply_goes_on) {
    for (bool on : { false, true }) {
        for (bool two : { false, true }) {
            ChatTemplate t;
            CHECK_OK(t.load(fixture(), "<bos>", "<eos>"));
            ChatOptions opt;
            opt.enable_thinking = on;
            ChatPrompt p;
            REQUIRE_EQ(t.apply(conversation(two), agent_tools(), opt, &p), RAD_OK);
            const std::string want =
                "<|turn>model\n<|tool_call>call:read{limit:3,path:<|\"|>/a<|\"|>}<tool_call|>" +
                std::string(two ? "<|tool_call>call:bash{command:<|\"|>ls<|\"|>}<tool_call|>" : "") +
                "<|tool_response>response:read{value:<|\"|>line one<|\"|>}<tool_response|>" +
                std::string(two ? "<|tool_response>response:bash{value:<|\"|>a.txt<|\"|>}<tool_response|>"
                                : "");
            if (p.prompt.find(want) == std::string::npos)
                ::radtest::fail(__FILE__, __LINE__, "the turn is not rendered as " + quote(want) +
                                "\n  prompt: " + quote(p.prompt));
            /* The model continues the same turn: with thinking off nothing is written for it;
             * with thinking on, inside a channel the template opened. */
            CHECK_EQ(p.generation_prompt, std::string(on ? kAfter : ""));
            CHECK(p.prompt.size() >= want.size() &&
                  p.prompt.compare(p.prompt.size() - p.generation_prompt.size() - 0,
                                   p.generation_prompt.size(), p.generation_prompt) == 0);

            std::shared_ptr<const ReplyParser> rp = parser(p.generation_prompt);
            const std::string reply = on ? "The file has one line.\n<channel|>It reads \"line one\"."
                                         : "It reads \"line one\".";
            const ReplyMessage m = parse(*rp, reply);
            CHECK_EQ(m.content, std::string("It reads \"line one\"."));
            CHECK_EQ(m.reasoning, std::string(on ? "The file has one line.\n" : ""));
            streams(*rp, reply, "after a tool result");
        }
    }
}

/* ================================================================== fuzz */

static std::vector<std::string> fragments() {
    return { "<|tool_call>", "<tool_call|>", "call:", "call:bash", "call:read", "{", "}", "[", "]",
             ",", ":", Q, "<|\"", "<|", "|>", "<|channel>thought\n", "<channel|>", "<|channel>",
             "\"", "\\", "\\u", "\\ud83d", "true", "None", "-1.5e3", "path", "command", "x", "\n",
             " ", "中", "🚀", "<turn|>", "<eos>", "<|tool_response>" };
}

TEST(fuzzed_replies_keep_the_contract) {
    std::mt19937 rng(3);
    const std::vector<std::string> frag = fragments();
    const std::vector<std::string> seeds = {
        G("configure", "extra:{a:[1,{b:" + S("c") + "}]},tags:[" + S("x") + "]"),
        "Sure.\n" + G("bash", "command:" + S("ls -la")) + G("read", "path:" + S("/a") + ",limit:3"),
        "<|channel>thought\nPlan.\n<channel|>Done.<|channel>thought\n<channel|>",
        G("write", "content:\"a\\nb\\u00e9\",path:" + S("/x")),
    };
    int n = 0;
    for (const std::string& gp : { kOff, kOn, kAfter }) {
        std::shared_ptr<const ReplyParser> p = parser(gp);
        for (const std::string& seed : seeds)
            for (int k = 0; k < 60; ++k) {
                const std::string text = sanitize(mutate(rng, seed, frag));
                const std::string err = check_stream(*p, text, every(text.size(), 1),
                                                     reply_parse(*p, text));
                if (!err.empty()) ::radtest::fail(__FILE__, __LINE__, err + "\n  reply: " + quote(text));
                ++n;
            }
    }
    CHECK(n > 0);
}

/* A long reply holds a bounded number of bytes: a file written into a string streams as it
 * arrives. */
TEST(a_long_string_streams_and_holds_a_bounded_number_of_bytes) {
    std::string file;
    while (file.size() < 200000) file += "    x = compute(a, b)  # <tag> & \"quoted\" <| ]] \\n\n";
    const std::string reply = "<|channel>thought\nWriting it.\n<channel|>" +
                              G("write", "content:" + S(file) + ",path:" + S("/big.py"));
    std::shared_ptr<const ReplyParser> p = parser(kOn);
    const std::string err = check_stream(*p, reply, every(reply.size(), 7), reply_parse(*p, reply));
    if (!err.empty()) ::radtest::fail(__FILE__, __LINE__, err);
    const ReplyMessage m = parse(*p, reply);
    REQUIRE_EQ(m.calls.size(), (size_t)1);
    CHECK(json::parse(m.calls[0].arguments)["content"] == file);
}

/* ================================================================== the grammar */

static const SimpleVocab& printable() {
    static const SimpleVocab v = [] {
        SimpleVocab s;
        for (int c = 32; c < 127; ++c) s.add(std::string(1, (char)c));
        s.add("\n");
        s.add("\t");
        return s;
    }();
    return v;
}

/* 1 complete, 0 admitted but unfinished or refused, -1 does not compile. */
static int admits(const std::string& gbnf, const std::string& prefix, const std::string& text) {
    std::unique_ptr<GbnfGrammar> gr;
    std::string err;
    if (GbnfGrammar::create(gbnf, "root", &printable(), false, {}, {}, &gr, &err) < 0) return -1;
    if (gr->accept_str(prefix) < 0 || gr->accept_str(text) < 0) return 0;
    return gr->complete() ? 1 : 0;
}

static std::string grammar_for(const ChatGrammarRequest& gq) {
    ChatFormat f;
    std::string why;
    CHECK_OK(chat_format_from_abi(rad::arch::gemma4_chat_format(), &f, &why));
    ChatGrammar g;
    CHECK_OK(chat_format_grammar(f, gq, &g, &why));
    return g.text;
}

/* The lazy call grammar admits every call the template writes and every strict call above, as
 * llama.cpp's Gemma 4 grammar does, and refuses what leaves the dialect. */
TEST(the_call_grammar_admits_what_the_model_writes_and_refuses_the_rest) {
    Bench* b = bench(thinking(true));
    REQUIRE(b != nullptr);
    ChatGrammarRequest gq;
    gq.tools = &agent_tools();
    gq.generation_prompt = b->prompt().generation_prompt;
    ChatFormat f;
    std::string why;
    CHECK_OK(chat_format_from_abi(rad::arch::gemma4_chat_format(), &f, &why));
    ChatGrammar g;
    CHECK_OK(chat_format_grammar(f, gq, &g, &why));
    CHECK(g.lazy);
    CHECK(g.trigger_words == std::vector<std::string>{ "<|tool_call>" });
    const std::string& vendored = b->prompt().grammar;

    int seen = 0;
    for (const json& msg : rendered_corpus()) {
        std::string reply = b->render_reply(msg);
        const size_t at = reply.find("<|tool_call>");
        if (at == std::string::npos) continue;
        reply = reply.substr(at, reply.rfind("<tool_call|>") + 12 - at);
        if (reply.find_first_of("\x01") != std::string::npos) continue;
        CHECK_EQ(admits(g.text, "", reply), 1);
        CHECK_EQ(admits(vendored, "", reply), 1);
        ++seen;
    }
    CHECK(seen > 8);
    for (const Shape& s : shapes()) {
        if (!s.strict || s.calls.empty() || s.reply.find("\x01") != std::string::npos) continue;
        const size_t at = s.reply.find("<|tool_call>");
        if (at == std::string::npos || !s.calls.back().closed) continue;
        const std::string calls = s.reply.substr(at, s.reply.rfind("<tool_call|>") + 12 - at);
        if (s.reply.find("\n<|tool_call>") != std::string::npos) continue;   /* a line between calls */
        if (std::string(s.what) == "whitespace inside the object" ||
            std::string(s.what) == "a string ending in the start of its own delimiter")
            continue;
        if (admits(g.text, "", calls) != 1)
            ::radtest::fail(__FILE__, __LINE__, std::string("refused: ") + s.what + " " + quote(calls));
    }
    /* Python's quotes, as the model writes arguments it was handed as JSON. */
    CHECK_EQ(admits(g.text, "", G("bash", "command:'echo \"hi\" \\' {x}'")), 1);
    CHECK_EQ(admits(g.text, "", G("configure", "extra:{tags:['a,b', 'c}']}")), 1);
    for (const std::string& bad : {
             G("bash", "command:ls"),                              /* a bare word */
             G("bash", "command:\"ls\""),                          /* a JSON string */
             G("bash", "command:" + S("ls") + ","),                 /* a trailing comma */
             "<|tool_call>call:bash{command:" + S("ls") + "<tool_call|>",   /* the object unclosed */
             G("frobnicate", "x:1"),                               /* not a tool */
             "<|tool_call>call:bash {command:" + S("ls") + "}<tool_call|>",  /* a space after the name */
             G("bash", "command:True"),
         })
        CHECK_EQ(admits(g.text, "", bad), 0);
}

/* tool_choice "required" constrains the whole reply: an optional thought, then calls and nothing
 * after. A named function admits only that one; parallel_tool_calls false admits one call. */
TEST(a_required_call_is_the_whole_reply) {
    for (const std::string& gp : { kOff, kOn }) {
        ChatGrammarRequest gq;
        gq.tools = &agent_tools();
        gq.generation_prompt = gp;
        gq.require_call = true;
        const std::string any = grammar_for(gq);
        gq.parallel_tool_calls = false;
        const std::string one = grammar_for(gq);
        gq.parallel_tool_calls = true;
        gq.only_tool = "read";
        const std::string rd = grammar_for(gq);
        const std::string ls = G("bash", "command:" + S("ls")), ra = G("read", "path:" + S("/a"));
        const std::string T = gp == kOn ? "<|channel>thought\nPlan.\n<channel|>" : "";
        CHECK_EQ(admits(any, gp, T + ls), 1);
        CHECK_EQ(admits(any, gp, T + ls + ra), 1);
        CHECK_EQ(admits(any, gp, T + "The answer."), 0);
        CHECK_EQ(admits(any, gp, T + ls + "Done."), 0);
        CHECK_EQ(admits(one, gp, T + ls), 1);
        CHECK_EQ(admits(one, gp, T + ls + ra), 0);
        CHECK_EQ(admits(rd, gp, T + ra), 1);
        CHECK_EQ(admits(rd, gp, T + ls), 0);
    }
}

TEST(a_response_format_is_the_whole_reply) {
    const json schema = json::parse(R"({"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]})");
    ChatGrammarRequest gq;
    gq.json_schema = &schema;
    gq.generation_prompt = kOff;
    const std::string g = grammar_for(gq);
    CHECK_EQ(admits(g, kOff, "{\"n\": 3}"), 1);
    CHECK_EQ(admits(g, kOff, "```json\n{\"n\": 3}\n```"), 1);
    CHECK_EQ(admits(g, kOff, "{\"n\": \"x\"}"), 0);
}

/* ================================================================== the server's view
 *
 * The chat bridge serves the model from the plugin's declaration: tool requests are accepted, the
 * grammar comes from the format, and the decoder renders the dialect's string delimiter (a control
 * token like the format's tags) so the parser sees it. */
static std::shared_ptr<Vocab> gemma_vocab() {
    VocabBuild vb;
    vb.tokens = { "<pad>", "<eos>", "<bos>", "a", "<|tool_call>", "<tool_call|>", "<|tool_response>",
                  "<tool_response|>", "<|\"|>", "<|channel>", "<channel|>", "<|turn>", "<turn|>",
                  "thought", "call:" };
    vb.types.assign(vb.tokens.size(), RAD_TT_CONTROL);
    vb.types[3] = vb.types[13] = vb.types[14] = RAD_TT_NORMAL;
    /* The letters the markers are spelled with, so a marker tokenises whole. */
    for (const char* c : { "t", "h", "o", "u", "g", "c", "l", ":" }) {
        vb.tokens.push_back(c);
        vb.types.push_back(RAD_TT_NORMAL);
    }
    vb.bos = 2;
    vb.eos = 1;
    vb.chat_template = fixture();
    /* generation_config.json: eos_token_id [1, 106, 50] -- here <eos>, <turn|>, <|tool_response>. */
    vb.eos_ids = { 1, 12, 6 };
    auto v = std::make_shared<Vocab>();
    return v->load(std::move(vb)) >= 0 ? v : nullptr;
}

TEST(the_chat_bridge_serves_gemma_from_its_declaration) {
    std::shared_ptr<Vocab> v = gemma_vocab();
    REQUIRE(v != nullptr);
    CHECK(v->is_eog(6));     /* <|tool_response>: after its calls the model stops there */
    server::ChatTemplateBridge br;
    CHECK_OK(br.init(*v, rad::arch::gemma4_chat_format(), "gemma4_bf16"));
    CHECK(br.supports_tools());
    CHECK(br.tool_refusal().empty());

    server::ChatRenderOptions opt;
    opt.enable_thinking = false;
    server::ChatRender r;
    std::string why;
    const json msgs = json::array({ { { "role", "user" }, { "content", "List /work." } } });
    REQUIRE_EQ(br.apply(msgs, agent_tools(), opt, &r,
                        &why),
               RAD_OK);
    CHECK_EQ(r.generation_prompt, kOff);
    CHECK(r.grammar_lazy);
    REQUIRE_EQ(r.grammar_triggers.size(), (size_t)1);
    CHECK_EQ(r.grammar_triggers[0].value, std::string("<|tool_call>"));
    CHECK(std::find(r.preserved_tokens.begin(), r.preserved_tokens.end(), Q) != r.preserved_tokens.end());
    CHECK(!r.think_starts_open);
    REQUIRE(r.parser != nullptr);

    /* The decoder renders every control token the markers are made of, the string's among them;
     * the turn's terminators it does not. */
    Detokenizer d(v, {}, false, r.preserved_tokens);
    std::string text;
    for (int32_t id : { 9, 13, 3, 10, 4, 14, 3, 8, 3, 8, 5 }) text += d.push(id);
    text += d.flush();
    CHECK_EQ(text, std::string("<|channel>thoughta<channel|><|tool_call>call:a<|\"|>a<|\"|><tool_call|>"));

    std::unique_ptr<server::IReplyReader> rd = r.parser->open();
    std::vector<server::ReplyDelta> deltas;
    const std::string reply = "On it." + G("bash", "command:" + S("ls"));
    for (char c : reply) rd->feed(std::string(1, c), &deltas);
    rd->finish(&deltas);
    const server::ParsedMessage pm = rd->message();
    CHECK_EQ(pm.content, std::string("On it."));
    REQUIRE_EQ(pm.tool_calls.size(), (size_t)1);
    CHECK_EQ(pm.tool_calls[0].name, std::string("bash"));
    CHECK_EQ(pm.tool_calls[0].arguments, std::string("{\"command\":\"ls\"}"));
    CHECK(!pm.open_call);
    std::string args;
    int begins = 0, ends = 0;
    for (const server::ReplyDelta& dl : deltas) {
        if (dl.kind == server::ReplyDelta::CallBegin) ++begins;
        if (dl.kind == server::ReplyDelta::CallArgs) args += dl.text;
        if (dl.kind == server::ReplyDelta::CallEnd) ++ends;
    }
    CHECK_EQ(begins, 1);
    CHECK_EQ(ends, 1);
    CHECK_EQ(args, pm.tool_calls[0].arguments);

    /* A tool whose name the object's opening would end is refused at admission, by name. */
    const json bad = json::parse(R"([{"type":"function","function":{"name":"a{b"}}])");
    server::ChatRender rb;
    CHECK(br.apply(msgs, bad, opt, &rb, &why) < 0);
    CHECK(why.find("a{b") != std::string::npos);
}

RAD_TEST_MAIN()
