/* chatparse_inkling2_test.cpp -- Inkling 2's reply format: every message opens with a header, and
 * a call's arguments sit in a JSON envelope that names the call again.
 *
 * The plugin declares the format (arch/common/rad_inkling2_chat.h):
 *
 *   <|message_model|><|content_thinking|>REASONING<|end_message|>      reasoning
 *   <|message_model|><|content_text|>ANSWER<|end_message|>             the answer
 *   <|message_model|>NAME<|content_invoke_tool_json|>
 *   {"name":"NAME","args":{...}}<|end_message|>                        a call, one per message
 *
 * The replies are the model's own, captured at temperature 0 from the real server with every
 * special token kept; the eos (<|content_model_end_sampling|>) ends generation and is not part of
 * what the parser reads. Control tokens the format does not name render as nothing, so each
 * reply is fed as the decoder would render it for this format; this format names every token in
 * these replies but the eos. Every reply is also streamed at every cut (chatdiff.h check_stream).
 */
#include "rad_test.h"

#include "chatdiff.h"
#include "rad_builder.h"
#include "sample/gbnf.h"
#include "sample/vocab_view.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"

#include <arch/rad_inkling2_chat.h>

#include "util/json.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace rad;
using namespace rad::chatdiff;
using json = Json;

/* ================================================================== fixtures */

static const char* const kOneCall =
    "<|message_model|><|content_thinking|>The user asks weather in Lehi, Utah. Need use "
    "get_weather. Need decide units? Not specified. Use c? Could use f since US city. Let's call "
    "get_weather city Lehi, Utah units f.<|end_message|><|message_model|>get_weather"
    "<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Lehi, Utah\","
    "\"units\":\"f\"}}<|end_message|>";

static const char* const kTwoCalls =
    "<|message_model|><|content_thinking|>The user wants weather in Lehi, Utah in Fahrenheit and "
    "time in America/Denver. Need call both tools. Weather tool takes city and units. Time tool "
    "takes tz. Do both in parallel.<|end_message|><|message_model|>get_weather"
    "<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Lehi, Utah\","
    "\"units\":\"f\"}}<|end_message|><|message_model|>get_time<|content_invoke_tool_json|>"
    "{\"name\":\"get_time\",\"args\":{\"tz\":\"America/Denver\"}}<|end_message|>";

static const char* const kNoCall =
    "<|message_model|><|content_thinking|>The user explicitly says do not use any tools. Need "
    "answer one short sentence. No need for weather/time. Final should be just greeting. Ensure "
    "one short sentence.<|end_message|><|message_model|><|content_text|>Hello! Nice to meet "
    "you.<|end_message|>";

static const json& tools() {
    static const json t = json::parse(R"([
      {"type":"function","function":{"name":"get_weather","description":"Weather in a city",
        "parameters":{"type":"object","properties":{"city":{"type":"string"},
          "units":{"type":"string","enum":["c","f"]}},"required":["city"]}}},
      {"type":"function","function":{"name":"get_time","description":"Time in a zone",
        "parameters":{"type":"object","properties":{"tz":{"type":"string"}},"required":["tz"]}}}
    ])");
    return t;
}

/* The format exactly as the plugin exports it, through the ABI. */
static ChatFormat inkling() {
    ChatFormat f;
    std::string why;
    if (chat_format_from_abi(inkling2::inkling2_chat_format(), &f, &why) < 0)
        ::radtest::fail(__FILE__, __LINE__, why);
    return f;
}

static std::shared_ptr<const ReplyParser> parser(const ChatFormat& f, const json* t = &tools(),
                                                 bool reasoning = true) {
    std::string why;
    std::shared_ptr<const ReplyFormat> rf;
    std::shared_ptr<const ReplyParser> rp;
    if (ReplyFormat::compile(f, &rf, &why) < 0) {
        ::radtest::fail(__FILE__, __LINE__, why);
        return nullptr;
    }
    ReplyOptions ro;
    ro.tools = t;
    ro.extract_reasoning = reasoning;
    ro.terminators = { "<|content_model_end_sampling|>" };
    if (ReplyParser::create(rf, ro, &rp, &why) < 0) ::radtest::fail(__FILE__, __LINE__, why);
    return rp;
}

static std::vector<size_t> every(size_t n) {
    std::vector<size_t> c;
    for (size_t i = 1; i < n; ++i) c.push_back(i);
    return c;
}

static std::vector<size_t> random_cuts(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<size_t> c;
    for (size_t at = 1 + rng() % 9; at < n; at += 1 + rng() % 9) c.push_back(at);
    return c;
}

/* The reply whole, byte by byte, at every two-way cut and in random pieces: each against the
 * one-shot parse, with the streaming contract checked on every piece. */
static void streams(const ReplyParser& p, const std::string& text) {
    const ReplyMessage expect = reply_parse(p, text);
    std::string err = check_stream(p, text, {}, expect);
    if (err.empty()) err = check_stream(p, text, every(text.size()), expect);
    for (size_t cut = 1; err.empty() && cut < text.size(); ++cut)
        err = check_stream(p, text, { cut }, expect);
    for (uint32_t seed = 1; err.empty() && seed <= 20; ++seed)
        err = check_stream(p, text, random_cuts(text.size(), seed), expect);
    if (!err.empty()) ::radtest::fail(__FILE__, __LINE__, err + " in " + json(text).dump());
}

static json args(const ReplyCall& c) {
    try {
        return json::parse(c.arguments);
    } catch (...) {
        return json("unparsed: " + c.arguments);
    }
}

/* ================================================================== the model's replies */

TEST(a_call_reads_as_the_call_the_model_wrote) {
    const auto p = parser(inkling());
    REQUIRE(p);
    bool strict = false;
    const ReplyMessage m = reply_parse(*p, kOneCall, &strict);
    CHECK_EQ(m.reasoning, std::string("The user asks weather in Lehi, Utah. Need use get_weather. "
                                      "Need decide units? Not specified. Use c? Could use f since "
                                      "US city. Let's call get_weather city Lehi, Utah units f."));
    CHECK_EQ(m.content, std::string());
    REQUIRE(m.calls.size() == 1u);
    CHECK_EQ(m.calls[0].name, std::string("get_weather"));
    CHECK(args(m.calls[0]) == json::parse(R"({"city":"Lehi, Utah","units":"f"})"));
    CHECK(m.calls[0].closed);
    CHECK(strict);
    streams(*p, kOneCall);
}

TEST(two_calls_read_in_order) {
    const auto p = parser(inkling());
    REQUIRE(p);
    bool strict = false;
    const ReplyMessage m = reply_parse(*p, kTwoCalls, &strict);
    CHECK(m.reasoning.starts_with("The user wants weather"));
    CHECK(m.reasoning.ends_with("Do both in parallel."));
    CHECK_EQ(m.content, std::string());
    REQUIRE(m.calls.size() == 2u);
    CHECK_EQ(m.calls[0].name, std::string("get_weather"));
    CHECK(args(m.calls[0]) == json::parse(R"({"city":"Lehi, Utah","units":"f"})"));
    CHECK_EQ(m.calls[1].name, std::string("get_time"));
    CHECK(args(m.calls[1]) == json::parse(R"({"tz":"America/Denver"})"));
    CHECK(m.calls[0].closed && m.calls[1].closed);
    CHECK(strict);
    streams(*p, kTwoCalls);
}

TEST(an_answer_reads_as_reasoning_and_content_with_no_markup) {
    const std::string want_r = "The user explicitly says do not use any tools. Need answer one "
                               "short sentence. No need for weather/time. Final should be just "
                               "greeting. Ensure one short sentence.";
    /* with tools offered, and with none */
    for (const json* t : { &tools(), (const json*)nullptr }) {
        const auto p = parser(inkling(), t);
        REQUIRE(p);
        const ReplyMessage m = reply_parse(*p, kNoCall);
        CHECK_EQ(m.reasoning, want_r);
        CHECK_EQ(m.content, std::string("Hello! Nice to meet you."));
        CHECK(m.calls.empty());
        streams(*p, kNoCall);
    }
    /* an answer with no reasoning before it */
    const auto p = parser(inkling());
    REQUIRE(p);
    const ReplyMessage m =
        reply_parse(*p, "<|message_model|><|content_text|>Hi there.<|end_message|>");
    CHECK_EQ(m.reasoning, std::string());
    CHECK_EQ(m.content, std::string("Hi there."));
    CHECK(m.calls.empty());
}

TEST(the_answer_reads_as_before_the_header_was_named) {
    /* The format without its message header and calls is what the plugin declared before: there
     * the headers and <|content_text|> rendered as nothing. Both read the answer the same. */
    ChatFormat old = inkling();
    old.message_open.clear();
    old.content_open.clear();
    old.name_prefix.clear();
    old.name_suffix.clear();
    old.call_end.clear();
    old.args_form = ChatArgs::Tagged;
    old.object_args_key.clear();
    const auto po = parser(old, nullptr);
    const auto pn = parser(inkling(), nullptr);
    REQUIRE(po && pn);
    const std::string rendered_before =
        "<|content_thinking|>Think.<|end_message|>Hello! Nice to meet you.<|end_message|>";
    const ReplyMessage a = reply_parse(*po, rendered_before);
    const ReplyMessage b = reply_parse(*pn, kNoCall);
    const ReplyMessage c = reply_parse(*pn, rendered_before);
    CHECK_EQ(a.content, b.content);
    CHECK_EQ(a.content, c.content);
    CHECK_EQ(a.reasoning, c.reasoning);
}

TEST(a_call_with_reasoning_left_inline_is_still_a_call) {
    const auto p = parser(inkling(), &tools(), /*reasoning=*/false);
    REQUIRE(p);
    const ReplyMessage m = reply_parse(*p, kOneCall);
    REQUIRE(m.calls.size() == 1u);
    CHECK(args(m.calls[0]) == json::parse(R"({"city":"Lehi, Utah","units":"f"})"));
    CHECK(m.content.starts_with("<|content_thinking|>The user asks"));
    streams(*p, kOneCall);
}

/* ================================================================== the envelope */

static std::string call_of(const std::string& name, const std::string& object) {
    return "<|message_model|>" + name + "<|content_invoke_tool_json|>" + object + "<|end_message|>";
}

TEST(the_envelope_carries_any_json_arguments) {
    const auto p = parser(inkling());
    REQUIRE(p);
    const std::vector<std::string> argsets = {
        R"({"city":"Say \"hi\" \\ there\n","units":"c"})",
        R"({"city":"{\"name\":\"x\",\"args\":}","units":"f"})",
        R"({"city":"Lehi","nested":{"a":[1,2,{"b":[true,false,null]}],"c":{}},"list":[[],[[]]]})",
        R"({"city":"\u00e9t\u00e9 \ud83d\ude00","n":-1.5e3})",
        R"({})",
    };
    for (const std::string& a : argsets) {
        const std::string reply = call_of("get_weather", "{\"name\":\"get_weather\",\"args\":" + a + "}");
        bool strict = false;
        const ReplyMessage m = reply_parse(*p, reply, &strict);
        REQUIRE(m.calls.size() == 1u);
        CHECK(args(m.calls[0]) == json::parse(a));
        CHECK(m.content.empty());
        CHECK(strict);
        streams(*p, reply);
    }
}

TEST(an_envelope_the_template_would_not_write_is_still_read) {
    const auto p = parser(inkling());
    REQUIRE(p);
    struct Case { const char* what; std::string object; const char* want; };
    const std::vector<Case> cases = {
        { "names another function",
          R"({"name":"get_time","args":{"city":"Lehi"}})", R"({"city":"Lehi"})" },
        { "keys reversed", R"({"args":{"city":"Lehi"},"name":"get_weather"})", R"({"city":"Lehi"})" },
        { "no name", R"({"args":{"city":"Lehi"}})", R"({"city":"Lehi"})" },
        { "no envelope", R"({"city":"Lehi","units":"f"})", R"({"city":"Lehi","units":"f"})" },
        { "a name and no args", R"({"name":"get_weather"})", R"({})" },
        { "a name then the arguments' own keys",
          R"({"name":"get_weather","city":"Lehi"})", R"({"city":"Lehi"})" },
        { "spaced", R"( { "name" : "get_weather" , "args" : { "city" : "Lehi" } } )",
          R"({"city":"Lehi"})" },
    };
    for (const Case& c : cases) {
        const std::string reply = call_of("get_weather", c.object);
        bool strict = true;
        const ReplyMessage m = reply_parse(*p, reply, &strict);
        if (m.calls.size() != 1u) {
            ::radtest::fail(__FILE__, __LINE__, std::string(c.what) + ": " + show(m));
            continue;
        }
        CHECK_EQ(m.calls[0].name, std::string("get_weather"));
        if (!(args(m.calls[0]) == json::parse(c.want)))
            ::radtest::fail(__FILE__, __LINE__, std::string(c.what) + ": " + m.calls[0].arguments);
        if (std::string(c.what) != "spaced" && strict)
            ::radtest::fail(__FILE__, __LINE__, std::string(c.what) + ": read as strict");
        CHECK(m.content.empty());
        streams(*p, reply);
    }
}

TEST(a_reply_cut_inside_a_call_reads_back_as_its_text) {
    const auto p = parser(inkling());
    REQUIRE(p);
    const std::string full = kOneCall;
    const size_t at = full.find("<|message_model|>get_weather");
    for (size_t cut : { at + 5, at + 30, full.find("\"args\"") + 3, full.find("Lehi, Utah\",") }) {
        const std::string part = full.substr(0, cut);
        std::unique_ptr<ReplyStream> s = p->open();
        s->feed(part, nullptr);
        s->finish(nullptr);
        const std::string text = s->content_with_open_call(part);
        if (!s->message().calls.empty())
            CHECK_EQ(s->message().calls.back().at, at);
        /* the call's own bytes, as the reply spelled them, and no reasoning */
        CHECK(text.find("The user asks") == std::string::npos);
        CHECK(full.compare(at, text.size(), text) == 0 || text.empty());
    }
    /* a later call's position is still the reply's, after the headers dropped before it */
    const ReplyMessage m = reply_parse(*p, kTwoCalls);
    REQUIRE(m.calls.size() == 2u);
    const std::string two = kTwoCalls;
    CHECK_EQ(m.calls[0].at, two.find("<|message_model|>get_weather"));
    CHECK_EQ(m.calls[1].at, two.find("<|message_model|>get_time"));
}

TEST(mangled_replies_never_break_the_contract) {
    /* Pieces of the three replies cut and spliced at random: whatever the frame holds or drops,
     * every cut reads as the whole reply does. */
    const auto p = parser(inkling());
    REQUIRE(p);
    const std::vector<std::string> src = { kOneCall, kTwoCalls, kNoCall };
    std::mt19937 rng(7);
    for (int n = 0; n < 300; ++n) {
        std::string r;
        const int parts = 1 + (int)(rng() % 4);
        for (int k = 0; k < parts; ++k) {
            const std::string& s = src[rng() % src.size()];
            const size_t a = rng() % s.size(), b = a + rng() % (s.size() - a + 1);
            r += s.substr(a, b - a);
        }
        const ReplyMessage expect = reply_parse(*p, r);
        std::string err = check_stream(*p, r, every(r.size()), expect);
        if (err.empty()) err = check_stream(*p, r, random_cuts(r.size(), (uint32_t)n), expect);
        if (!err.empty()) {
            ::radtest::fail(__FILE__, __LINE__, err + " in " + json(r).dump());
            break;
        }
    }
}

/* ================================================================== the declaration */

TEST(the_format_is_accepted_and_described) {
    const ChatFormat f = inkling();
    std::string why;
    CHECK_OK(chat_format_check(f, &why));
    CHECK_EQ(f.object_args_key, std::string("args"));
    CHECK_EQ(f.message_open, std::string("<|message_model|>"));
    const std::string d = chat_format_describe(f);
    CHECK(d.find("envelope {\"name\":NAME,\"args\":OBJECT}") != std::string::npos);
    CHECK(d.find("messages open with \"<|message_model|>\"") != std::string::npos);
    CHECK(d.find("no tool calls") == std::string::npos);
    /* every token the replies carry but the eos is rendered for the parser */
    const std::vector<std::string> mk = chat_format_markers(f);
    for (const char* t : { "<|message_model|>", "<|content_thinking|>", "<|content_text|>",
                           "<|content_invoke_tool_json|>", "<|end_message|>" })
        CHECK(std::find(mk.begin(), mk.end(), t) != mk.end());
}

TEST(what_the_envelope_and_header_cannot_read_is_refused) {
    auto refused = [](void (*edit)(ChatFormat&), const char* word) {
        ChatFormat f = inkling();
        edit(f);
        std::string why;
        const bool no = chat_format_check(f, &why) == RAD_E_UNSUPPORTED;
        if (!no || why.find(word) == std::string::npos)
            ::radtest::fail(__FILE__, __LINE__, std::string("expected a refusal naming ") + word +
                                                    ", got \"" + why + "\"");
    };
    refused([](ChatFormat& f) { f.args_form = ChatArgs::Tagged; f.arg_prefix = "<a>";
                                f.arg_name_suffix = "</a>"; f.arg_value_suffix = "</v>"; },
            "object_*");
    refused([](ChatFormat& f) { f.object_string_open = "<|\"|>"; f.object_string_close = "<|\"|>"; },
            "dialect");
    refused([](ChatFormat& f) { f.object_bare_keys = true; }, "dialect");
    refused([](ChatFormat& f) { f.object_args_key = "a\"b"; }, "quote");
    refused([](ChatFormat& f) { f.object_args_key = "a\\b"; }, "backslash");
    refused([](ChatFormat& f) { f.name_suffix.clear(); }, "name_suffix");
    refused([](ChatFormat& f) {
                f.name_prefix.clear(); f.name_suffix.clear(); f.call_end.clear();
                f.args_form = ChatArgs::Tagged; },
            "object_args_key");
    refused([](ChatFormat& f) { f.name_prefix = "<|call|>"; }, "name_prefix");
    refused([](ChatFormat& f) { f.call_open = "<tool>"; f.call_close = "</tool>"; }, "call_open");
    refused([](ChatFormat& f) { f.message_open = " <|message_model|>"; f.name_prefix = f.message_open; },
            "whitespace");
    refused([](ChatFormat& f) { f.content_open = "<|content_thinking|>x"; }, "cannot be told");
    refused([](ChatFormat& f) { f.message_open = "<|content_text|>"; f.name_prefix = f.message_open; },
            "cannot be told");
    refused([](ChatFormat& f) {
                f.reasoning_open.clear(); f.reasoning_close.clear(); f.reasoning_anywhere = false;
                f.content_open.clear(); },
            "message_open needs");
}

TEST(a_plugin_compiled_before_the_fields_reads_them_as_empty) {
    RadChatFormat c = *inkling2::inkling2_chat_format();
    ChatFormat f;
    std::string why;
    c.struct_size = (uint32_t)offsetof(RadChatFormat, object_args_key);
    c.name_prefix = c.name_suffix = c.call_end = nullptr;
    c.args_form = 0;
    CHECK_OK(chat_format_from_abi(&c, &f, &why));
    CHECK(f.object_args_key.empty());
    CHECK(f.message_open.empty());
    CHECK_EQ(f.content_open, std::string("<|content_text|>"));
    c.struct_size = (uint32_t)offsetof(RadChatFormat, message_open);
    CHECK_OK(chat_format_from_abi(&c, &f, &why));
    CHECK_EQ(f.object_args_key, std::string("args"));
    CHECK(f.message_open.empty());
}

/* ================================================================== the grammar */

static int admits(const std::string& gbnf, const std::string& text) {
    static const SimpleVocab vocab = [] {
        SimpleVocab v;
        for (int c = 32; c < 127; ++c) v.add(std::string(1, (char)c));
        v.add("\n");
        return v;
    }();
    std::unique_ptr<GbnfGrammar> gr;
    std::string err;
    if (GbnfGrammar::create(gbnf, "root", &vocab, false, {}, {}, &gr, &err) < 0) return -1;
    if (gr->accept_str(text) < 0) return 0;
    return gr->complete() ? 1 : 0;
}

TEST(the_grammar_writes_the_envelope) {
    /* The plugin declares no trigger, so a lazy request is unconstrained. */
    ChatGrammar g;
    std::string why;
    ChatGrammarRequest rq;
    rq.tools = &tools();
    CHECK_OK(chat_format_grammar(inkling(), rq, &g, &why));
    CHECK(g.text.empty());

    /* With a trigger the call rule writes the envelope compactly, the template's way. */
    ChatFormat f = inkling();
    f.triggers = { "<|message_model|>" };
    CHECK_OK(chat_format_grammar(f, rq, &g, &why));
    REQUIRE(!g.text.empty());
    CHECK(g.text.find("\"{\\\"name\\\":\\\"get_weather\\\",\\\"args\\\":\"") != std::string::npos);
    const std::string one = call_of("get_weather",
                                    R"({"name":"get_weather","args":{"city":"Lehi, Utah","units":"f"}})");
    CHECK_EQ(admits(g.text, one), 1);
    CHECK_EQ(admits(g.text, call_of("get_weather", R"({"city":"Lehi, Utah"})")), 0);
    CHECK_EQ(admits(g.text, call_of("get_weather", R"({"name":"get_time","args":{}})")), 0);

    /* A required call: the reasoning message's header, then the calls. */
    rq.require_call = true;
    CHECK_OK(chat_format_grammar(f, rq, &g, &why));
    const std::string reply = "<|message_model|><|content_thinking|>Plan.<|end_message|>" + one;
    CHECK_EQ(admits(g.text, reply), 1);
}

RAD_TEST_MAIN()
