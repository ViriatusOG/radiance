#include "chatdiff.h"
#include "text/tokenizer.h"

#include <cstdio>
#include <random>

namespace rad {
namespace chatdiff {

const std::vector<Model>& models() {
    static const std::vector<Model> m = {
        { "qwen3.8",  "qwen3.8",  "",    "<|im_end|>", { "<|im_end|>", "<|endoftext|>" } },
        { "qwen3.6",  "qwen3.6",  "",    "<|im_end|>", { "<|im_end|>", "<|endoftext|>" } },
        { "minicpm5", "minicpm5", "<s>", "</s>",       { "</s>", "<|im_end|>" } },
    };
    return m;
}

std::string describe(const ChatRequest& r) {
    std::string s = r.tools ? "tools" : "no-tools";
    if (!r.parallel) s += ",single";
    if (!r.thinking) s += ",no-think";
    if (!r.reasoning) s += ",reasoning-none";
    if (!r.schema.empty()) s += ",response-format";
    return s;
}

const ojson& agent_tools() {
    static const ojson t = ojson::parse(R"([
 {"type":"function","function":{"name":"bash","description":"Run a shell command.",
  "parameters":{"type":"object","properties":{"command":{"type":"string"},
   "timeout":{"type":"integer"},"description":{"type":"string"}},"required":["command"]}}},
 {"type":"function","function":{"name":"read","description":"Read a file.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},
   "offset":{"type":"integer"},"limit":{"type":"integer"}},"required":["path"]}}},
 {"type":"function","function":{"name":"write","description":"Write a file.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},
   "content":{"type":"string"}},"required":["path","content"]}}},
 {"type":"function","function":{"name":"edit","description":"Apply edits to a file.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},
   "edits":{"type":"array","items":{"type":"object","properties":{
     "oldText":{"type":"string"},"newText":{"type":"string"}},
     "required":["oldText","newText"]}}},"required":["path","edits"]}}},
 {"type":"function","function":{"name":"todo_write","description":"Update the todo list.",
  "parameters":{"type":"object","properties":{"todos":{"type":"array","items":{
   "type":"object","properties":{"content":{"type":"string"},
   "status":{"type":"string","enum":["pending","in_progress","completed"]}},
   "required":["content","status"]}}},"required":["todos"]}}},
 {"type":"function","function":{"name":"configure","description":"Set options.",
  "parameters":{"type":"object","properties":{"verbose":{"type":"boolean"},
   "level":{"type":"number"},"mode":{"enum":["fast","slow"]},"extra":{"type":"object"},
   "label":{"type":["string","null"]},"tags":{"type":"array","items":{"type":"string"}},
   "nothing":{"type":"null"}}}}},
 {"type":"function","function":{"name":"ping","description":"Takes no arguments.",
  "parameters":{"type":"object","properties":{}}}}
])");
    return t;
}

int Bench::load(const Model& m, const std::string& jinja, std::string* why) {
    model_ = m;
    jinja_ = jinja;
    tmpl_ = std::make_shared<ChatTemplate>();
    if (tmpl_->load(jinja, m.bos, m.eos) < 0) {
        *why = "the template did not compile";
        return RAD_E_FORMAT;
    }
    if (m.declared) {
        const int st = chat_format_from_abi(m.declared, &fmt_, why);
        if (st < 0) return st;
        return ReplyFormat::compile(fmt_, &rf_, why);
    }
    std::string refused;
    int st = chat_format_from_template(jinja, m.bos, m.eos, &fmt_, &refused, why);
    if (st < 0) return st;
    if (!refused.empty()) {
        *why = "tool calls refused: " + refused;
        return RAD_E_UNSUPPORTED;
    }
    return ReplyFormat::compile(fmt_, &rf_, why);
}

static ChatOptions options_for(const ChatRequest& rq) {
    ChatOptions opt;
    opt.add_generation_prompt = true;
    opt.enable_thinking       = rq.thinking;
    opt.parallel_tool_calls   = rq.parallel;
    opt.reasoning_format      = rq.reasoning ? "auto" : "none";
    opt.json_schema           = rq.schema;
    return opt;
}

static ojson user_turn() {
    return ojson::array({ ojson{ { "role", "user" }, { "content", "Fix the failing test in /work." } } });
}

int Bench::bind(const ChatRequest& rq, const ojson& tools, std::string* why) {
    req_ = rq;
    tools_ = tools;
    prompt_ = ChatPrompt{};
    replies_.clear();
    const ChatOptions opt = options_for(rq);
    if (tmpl_->apply(user_turn(), rq.tools ? tools_ : ojson(), opt, &prompt_) < 0) {
        *why = "render failed: " + prompt_.error;
        return RAD_E_INVAL;
    }
    old_ = std::make_unique<legacy::ToolParser>();
    if (old_->init(prompt_, opt.reasoning_format, rq.tools) < 0) {
        *why = "the old parser did not load";
        return RAD_E_FORMAT;
    }
    ReplyOptions ro;
    ro.generation_prompt   = prompt_.generation_prompt;
    ro.extract_reasoning   = rq.reasoning;
    ro.tools               = rq.tools ? &tools_ : nullptr;
    ro.parallel_tool_calls = rq.parallel;
    ro.response_format     = !rq.schema.empty();
    ro.terminators         = model_.terminators;
    return ReplyParser::create(rf_, ro, &parser_, why);
}

legacy::Answer Bench::old_parse(const std::string& text, bool partial) const {
    return legacy::legacy_parse(*old_, text, partial, req_.tools, model_.terminators);
}

ReplyMessage Bench::new_parse(const std::string& text, bool* strict) const {
    return reply_parse(*parser_, text, strict);
}

std::string Bench::render_reply(const ojson& assistant) const {
    /* A render is a function of the template, the request and the message, and the first two are
     * fixed for a bench; the corpora render the same messages many times. */
    std::string key = assistant.dump();
    auto it = replies_.find(key);
    if (it != replies_.end()) return it->second;
    return replies_[std::move(key)] = render_reply_uncached(assistant);
}

std::string Bench::render_reply_uncached(const ojson& assistant) const {
    /* The head is bind()'s render: the same user turn, tools and options. */
    const std::string& head = prompt_.prompt;
    ChatOptions opt = options_for(req_);
    ChatPrompt whole;
    ojson msgs = user_turn();
    msgs.push_back(assistant);
    opt.add_generation_prompt = false;
    if (tmpl_->apply(msgs, req_.tools ? tools_ : ojson(), opt, &whole) < 0) return "";
    if (whole.prompt.size() < head.size() || whole.prompt.compare(0, head.size(), head) != 0)
        return "";
    std::string reply = whole.prompt.substr(head.size());
    while (!reply.empty() && reply.back() == '\n') reply.pop_back();
    for (const std::string& t : model_.terminators)
        if (reply.size() >= t.size() && reply.compare(reply.size() - t.size(), t.size(), t) == 0) {
            reply.erase(reply.size() - t.size());
            break;
        }
    return reply;
}

static std::string q(const std::string& s) {
    return Json(s).dump(-1, ' ', false, Json::error_handler_t::replace);
}

std::string show(const ReplyMessage& m) {
    std::string s = "content=" + q(m.content) + " reasoning=" + q(m.reasoning);
    for (const ReplyCall& c : m.calls)
        s += "\n      call " + q(c.name) + " " + q(c.arguments) + (c.closed ? "" : " (open)");
    return s;
}

std::string show(const legacy::Answer& a) {
    std::string s = "content=" + q(a.content) + " reasoning=" + q(a.reasoning);
    if (a.strict_failed) s += " [strict parse failed]";
    if (a.lifted) s += " [" + std::to_string(a.lifted) + " lifted]";
    for (const legacy::ToolCall& c : a.calls) s += "\n      call " + q(c.name) + " " + q(c.arguments);
    return s;
}

bool same(const ReplyMessage& a, const ReplyMessage& b) {
    if (a.content != b.content || a.reasoning != b.reasoning || a.calls.size() != b.calls.size())
        return false;
    for (size_t i = 0; i < a.calls.size(); ++i)
        if (a.calls[i].name != b.calls[i].name || a.calls[i].arguments != b.calls[i].arguments ||
            a.calls[i].closed != b.calls[i].closed)
            return false;
    return true;
}

Verdict compare(const Bench& b, const std::string& text) {
    Verdict v;
    const legacy::Answer o = b.old_parse(text);
    const ReplyMessage m = b.new_parse(text, &v.strict);
    v.old_clean = !o.strict_failed && o.lifted == 0;
    v.bare_lift = !o.strict_failed && o.lifted > 0;
    const ChatFormat& f = b.format();
    if (!f.call_open.empty()) {
        size_t a = 0, e = f.call_open.size();
        while (a < e && (unsigned char)f.call_open[a] <= ' ') ++a;
        while (e > a && (unsigned char)f.call_open[e - 1] <= ' ') --e;
        v.old_call_syntax = o.content.find(f.name_prefix) != std::string::npos ||
                            o.content.find(f.call_open.substr(a, e - a)) != std::string::npos;
    }
    v.same = o.content == m.content && o.reasoning == m.reasoning && o.calls.size() == m.calls.size();
    /* Arguments are compared only where the old object is JSON, and then byte for byte. The old
     * reader can pass a typed value's text through bare (`"verbose":True`, `"extra":}`), while the
     * new one makes every complete value JSON (chatparse.cpp, ValueOut); the decided shapes and
     * streaming checks cover the new reader there. */
    auto args_agree = [](const std::string& old_a, const std::string& new_a) {
        if (old_a == new_a || !Json::accept(old_a)) return true;
        /* One allowed difference: a leading newline the old reader stripped from a value after
         * a tag that already ends with one (Qwen's `<parameter=NAME>\n`), which the template
         * writes when the value itself starts with a newline. Every other byte must agree. */
        const ojson o = ojson::parse(old_a, nullptr, false);
        const ojson n = ojson::parse(new_a, nullptr, false);
        if (!o.is_object() || !n.is_object() || o.size() != n.size()) return false;
        for (auto it = o.begin(); it != o.end(); ++it) {
            if (!n.contains(it.key())) return false;
            const ojson& nv = n.at(it.key());
            if (nv == it.value()) continue;
            if (!nv.is_string() || !it.value().is_string()) return false;
            const std::string& ns = nv.get_ref<const std::string&>();
            if (ns.empty() || ns[0] != '\n' || ns.compare(1, std::string::npos,
                                                         it.value().get_ref<const std::string&>()) != 0)
                return false;
        }
        return true;
    };
    for (size_t i = 0; v.same && i < o.calls.size(); ++i)
        v.same = o.calls[i].name == m.calls[i].name &&
                 args_agree(o.calls[i].arguments, m.calls[i].arguments);
    for (const legacy::ToolCall& c : o.calls)
        for (char ch : c.arguments)
            if ((unsigned char)ch < 0x20) v.old_raw_control = true;
    if (!v.same) v.detail = "    old: " + show(o) + "\n    new: " + show(m);
    return v;
}

std::string judge(const Verdict& v) {
    if (v.strict) {
        if (v.old_clean && !v.same && !v.old_raw_control)
            return "both read the reply within the grammar and disagree";
        if (!v.old_clean && !v.bare_lift)
            return "the new parser read within the grammar a reply the old grammar rejected";
        return std::string();
    }
    if (v.old_clean && !v.old_call_syntax && !v.old_raw_control)
        return "the new parser left the grammar on a reply the old grammar accepted";
    return std::string();
}

std::string check_stream(const ReplyParser& p, const std::string& text,
                         const std::vector<size_t>& cuts, const ReplyMessage& expect) {
    std::unique_ptr<ReplyStream> s = p.open();
    ReplyMessage got;
    std::vector<ReplyEvent> ev;
    const size_t bound = p.hold_bound();
    auto grew = [](const std::string& now, size_t before, const std::string& want) {
        return now.size() <= want.size() &&
               now.compare(before, std::string::npos, want, before, now.size() - before) == 0;
    };
    auto take = [&](const char* where) -> std::string {
        for (const ReplyEvent& e : ev) {
            switch (e.kind) {
            case ReplyEvent::Reasoning: {
                const size_t b = got.reasoning.size();
                got.reasoning += e.text;
                if (!grew(got.reasoning, b, expect.reasoning))
                    return std::string(where) + ": reasoning delta is not a prefix of the answer";
                break;
            }
            case ReplyEvent::Content: {
                const size_t b = got.content.size();
                got.content += e.text;
                if (!grew(got.content, b, expect.content))
                    return std::string(where) + ": content delta is not a prefix of the answer";
                break;
            }
            case ReplyEvent::CallBegin:
                if (e.call != got.calls.size() || e.call >= expect.calls.size() ||
                    expect.calls[e.call].name != e.text)
                    return std::string(where) + ": a call began that the answer does not have";
                got.calls.push_back({ e.text, std::string(), false });
                break;
            case ReplyEvent::CallArgs: {
                if (e.call >= got.calls.size()) return std::string(where) + ": arguments for no call";
                std::string& a = got.calls[e.call].arguments;
                const size_t b = a.size();
                a += e.text;
                if (!grew(a, b, expect.calls[e.call].arguments))
                    return std::string(where) + ": arguments delta is not a prefix of the answer";
                break;
            }
            case ReplyEvent::CallEnd:
                if (e.call >= got.calls.size() || !expect.calls[e.call].closed)
                    return std::string(where) + ": a call closed that the answer leaves open";
                got.calls[e.call].closed = true;
                break;
            }
        }
        ev.clear();
        return std::string();
    };
    size_t at = 0;
    for (size_t c : cuts) {
        if (c <= at || c > text.size()) continue;
        s->feed(std::string_view(text).substr(at, c - at), &ev);
        at = c;
        std::string err = take("after a piece");
        if (!err.empty()) return err + " (at byte " + std::to_string(at) + ")";
        const ReplyHeld h = s->held();
        if (h.delimiter > bound)
            return "held " + std::to_string(h.delimiter) + " bytes of text, over the bound of " +
                   std::to_string(bound) + " (at byte " + std::to_string(at) + ")";
    }
    if (at < text.size()) s->feed(std::string_view(text).substr(at), &ev);
    s->finish(&ev);
    std::string err = take("at the end");
    if (!err.empty()) return err;
    if (!same(got, expect)) return "the deltas add up to\n      " + show(got);
    if (!same(s->message(), expect)) return "the message is\n      " + show(s->message());
    /* Whether the reply stayed in the format is the reply's, not the cut's: the server logs a
     * reply that left it, so a cut that changes the verdict changes what is logged. */
    bool whole = false;
    reply_parse(p, text, &whole);
    if (s->strict() != whole)
        return std::string("read ") + (s->strict() ? "strict" : "lenient") + " cut this way, " +
               (whole ? "strict" : "lenient") + " read whole";
    /* Where each call begins (an open call's text is read back from there) depends on the reply,
     * not on how it was cut. */
    for (size_t i = 0; i < expect.calls.size(); ++i) {
        const ReplyCall& a = s->message().calls[i];
        const ReplyCall& b = expect.calls[i];
        if (a.at != b.at || a.content_at != b.content_at || a.content_held != b.content_held)
            return "call " + std::to_string(i) + " begins at " + std::to_string(a.at) + " cut this way, " +
                   std::to_string(b.at) + " read whole";
        if (b.at >= text.size() || (i > 0 && b.at <= expect.calls[i - 1].at))
            return "call " + std::to_string(i) + " begins at " + std::to_string(b.at) + ", outside the reply";
    }
    /* A closed call is handed to a client to run, so its arguments are JSON whatever the model
     * wrote, for any valid UTF-8 reply (the decoder sanitises every reply). */
    if (sanitize(text) == text)
        for (const ReplyCall& c : expect.calls)
            if (c.closed && !Json::accept(c.arguments))
                return "a closed call's arguments are no JSON: " + q(c.arguments);
    return std::string();
}

std::string sanitize(const std::string& s) { return utf8_sanitize(s); }

/* ================================================================== the corpora */

bool is_qwen(const Model& m) { return m.name != "minicpm5"; }

ojson call(const std::string& name, const ojson& args) {
    return { { "type", "function" },
             { "function", { { "name", name }, { "arguments", args.dump() } } } };
}

ojson message(const std::string& reasoning, const std::string& content,
              const std::vector<ojson>& calls) {
    ojson m = { { "role", "assistant" }, { "content", content } };
    if (!reasoning.empty()) m["reasoning_content"] = reasoning;
    if (!calls.empty()) {
        m["tool_calls"] = ojson::array();
        for (size_t i = 0; i < calls.size(); ++i) {
            ojson c = calls[i];
            c["id"] = "call_" + std::to_string(i);
            m["tool_calls"].push_back(c);
        }
    }
    return m;
}

static const std::string kFile =
    "def add(a, b):\n\n\t# sum\n    return a + b  \n\nclass X:\n    s = \"<b>&amp;</b>\"\n";

std::vector<ojson> rendered_corpus() {
    std::vector<ojson> v;
    v.push_back(message("", "The tests pass now.", {}));
    v.push_back(message("The user wants a summary.\nIt is short.", "It prints one.", {}));
    v.push_back(message("I should look first.", "",
                        { call("bash", { { "command", "ls -la /work" }, { "timeout", 30 } }) }));
    v.push_back(message("", "Let me check the directory.",
                        { call("bash", { { "command", "ls" } }) }));
    v.push_back(message("Two things.", "", { call("read", { { "path", "/work/a.py" } }),
                                             call("read", { { "path", "/work/b.py" },
                                                            { "offset", 10 },
                                                            { "limit", 40 } }) }));
    {
        std::vector<ojson> five;
        for (int i = 0; i < 5; ++i)
            five.push_back(call("bash", { { "command", "echo " + std::to_string(i) } }));
        v.push_back(message("Five commands.", "Running them all.", five));
    }
    v.push_back(message("", "", { call("write", { { "path", "/work/a.py" }, { "content", kFile } }) }));
    v.push_back(message("Editing.", "", { call("edit", { { "path", "/work/a.py" },
        { "edits", ojson::array({ { { "oldText", "    return a + b" },
                                   { "newText", "    return a + b + 1" } } }) } }) }));
    v.push_back(message("", "", { call("todo_write", { { "todos", ojson::array({
        { { "content", "write the test" }, { "status", "completed" } },
        { { "content", "fix it" }, { "status", "in_progress" } } }) } }) }));
    v.push_back(message("", "", { call("configure", { { "level", 1.5 },
                                                      { "extra", { { "a", ojson::array({ 1, 2 }) },
                                                                   { "b", { { "c", nullptr } } } } },
                                                      { "tags", ojson::array({ "x", "y" }) },
                                                      { "mode", "fast" },
                                                      { "label", "l" } }) }));
    v.push_back(message("Unicode.", "Größe, 大小, 🚀.",
                        { call("write", { { "path", "/work/中文.txt" },
                                          { "content", "こんにちは 🌍\nÀ la carte — «ok»\n" } }) }));
    v.push_back(message("", "", { call("write", { { "path", "/work/x.xml" },
        { "content", "<param name=\"k\">v</param>\n</param\n</parameter\n</function>x" } }) }));
    v.push_back(message("", "", { call("write", { { "path", "" }, { "content", "" } }) }));
    v.push_back(message("", "", { call("write", { { "path", "/a" }, { "content", "\n\nx\n\n" } }) }));
    v.push_back(message("", "", { call("ping", ojson::object()) }));
    return v;
}

std::string qcall(const std::string& name,
                         const std::vector<std::pair<std::string, std::string>>& args) {
    std::string s = "<tool_call>\n<function=" + name + ">\n";
    for (const auto& a : args) s += "<parameter=" + a.first + ">\n" + a.second + "\n</parameter>\n";
    return s + "</function>\n</tool_call>";
}

std::string mcall(const std::string& name,
                         const std::vector<std::pair<std::string, std::string>>& args) {
    std::string s = "<function name=\"" + name + "\">";
    for (const auto& a : args) s += "<param name=\"" + a.first + "\">" + a.second + "</param>";
    return s + "</function>";
}

std::string C(const Model& m, const std::string& name,
                     const std::vector<std::pair<std::string, std::string>>& args) {
    return is_qwen(m) ? qcall(name, args) : mcall(name, args);
}

std::vector<Adversarial> adversarial(const Model& m) {
    const std::string T = "Thinking it over.\n</think>\n\n";
    std::vector<Adversarial> v;
    auto add = [&](const std::string& n, const std::string& t) { v.push_back({ n, t }); };

    add("reasoning only, never closed", "Thinking and thinking");
    add("reasoning only, closed", "Thinking.\n</think>\n\n");
    add("content only", T + "The answer is 42.");
    add("reasoning then a call", T + C(m, "bash", { { "command", "ls" } }));
    add("content then a call", T + "Checking.\n" + C(m, "bash", { { "command", "ls" } }));
    add("calls then trailing text", T + C(m, "bash", { { "command", "ls" } }) + "\nDone now.");
    add("two calls", T + C(m, "bash", { { "command", "ls" } }) + "\n" +
                     C(m, "read", { { "path", "/a" } }));
    add("two calls with a sentence between", T + C(m, "bash", { { "command", "ls" } }) +
                                             "\nNow the file.\n" + C(m, "read", { { "path", "/a" } }));
    {
        std::string five = T;
        for (int i = 0; i < 5; ++i) five += C(m, "bash", { { "command", "echo " + std::to_string(i) } }) + "\n";
        add("five calls", five);
    }
    add("an integer", T + C(m, "read", { { "path", "/a" }, { "limit", "40" } }));
    add("a number", T + C(m, "configure", { { "level", "-1.5e3" } }));
    add("a boolean", T + C(m, "configure", { { "verbose", "true" } }));
    add("a null", T + C(m, "configure", { { "nothing", "null" } }));
    add("an object", T + C(m, "configure", { { "extra", "{\"a\": {\"b\": [1, {\"c\": \"d\"}]}}" } }));
    add("an array", T + C(m, "configure", { { "tags", "[\"x\", \"y\"]" } }));
    add("a Python-quoted array", T + C(m, "edit", { { "path", "/a" },
                                                    { "edits", "[{'oldText': 'a', 'newText': 'b\\'s'}]" } }));
    add("an integer with units", T + C(m, "bash", { { "command", "sleep" }, { "timeout", "30 seconds" } }));
    add("JSON then a stray character", T + C(m, "configure", { { "extra", "{\"a\": 1}x" } }));
    add("an unterminated JSON string", T + C(m, "configure", { { "extra", "\"abc" } }));
    add("a number that is not one", T + C(m, "read", { { "path", "/a" }, { "limit", "01" } }));
    add("Python literals", T + C(m, "configure", { { "verbose", "True" }, { "nothing", "None" } }));
    add("a Python dict with literals", T + C(m, "configure",
                                             { { "extra", "{'a': True, 'b': None, 'c': 'False'}" } }));
    add("an empty typed value", T + C(m, "configure", { { "level", "" } }));
    add("a typed value that is no JSON", T + C(m, "configure", { { "verbose", "yes please" } }));
    add("JSON strings holding a raw line break and tab", T + C(m, "configure",
        { { "extra", "{\"a\": \"line1\nline2\",\n \"b\": \"\tx\"}" } }));
    add("Python containers that are no JSON", T + C(m, "configure",
        { { "extra", "{'a': 1, 'b': [2, 3],}" }, { "tags", "['x', nan]" } }));
    add("a container missing its close", T + C(m, "configure", { { "extra", "{'a': [1, 2}" } }));
    add("JSON a client cannot read", T + C(m, "configure",
        { { "level", "1e400" }, { "extra", R"({"a": "\ud800"})" } }));
    add("an escaped quote in a double-quoted string", T + C(m, "configure",
        { { "tags", "  [\"it\\'s\", 'x']" } }));
    add("unicode", T + "Größe 大小 🚀 " + C(m, "write", { { "path", "/中.txt" }, { "content", "こんにちは 🌍" } }));
    add("CDATA with brackets and tags", T + C(m, "write", { { "path", "/a" },
        { "content", "<![CDATA[if (a[b[0]]) { x = \"]]\"; }\n  <tag>\n    indented\n]]>" } }));
    add("CDATA holding almost a closer", T + C(m, "write", { { "path", "/a" }, { "content", "<![CDATA[a]]b]] >]]>" } }));
    add("a value holding part of its closer", T + C(m, "write", { { "path", "/a" },
        { "content", "x </param y </parameter z </para" } }));
    add("empty values", T + C(m, "write", { { "path", "" }, { "content", "" } }));
    add("an undeclared argument", T + C(m, "read", { { "path", "/a" }, { "mode", "fast" } }));
    add("only an undeclared argument", T + C(m, "read", { { "action", "read" } }));
    add("arguments out of order", T + C(m, "write", { { "content", "x" }, { "path", "/a" } }));
    add("a required argument missing", T + C(m, "write", { { "path", "/a" } }));
    add("no arguments for a tool that needs one", T + C(m, "bash", {}));
    add("an argument for a tool that takes none", T + C(m, "ping", { { "x", "1" } }));
    add("a function that is not a tool", T + C(m, "which gcc cc clang make", {}));
    add("unclosed think before a call", "I will list it.\n" + C(m, "bash", { { "command", "ls" } }));
    add("a truncated reply inside a call", T + "Writing.\n" + C(m, "write", { { "path", "/a" }, { "content", "partial fi" } })
                                                      .substr(0, 40));
    add("a truncated reply inside an argument name", T + (is_qwen(m) ? "<tool_call>\n<function=read>\n<parameter=pa"
                                                                     : "<function name=\"read\"><param name=\"pa"));
    add("a truncated reply inside a function name", T + (is_qwen(m) ? "<tool_call>\n<function=wri" : "<function name=\"wri"));
    add("delimiter-like text in content", T + "Use `</think>`, `<tool_call>`, `<function=f>` and "
                                              "`<function name=\"f\">` to call. </parameter> </param>");
    add("a closing tag mentioned in reasoning", "The tag </thin is not </think>\n\nfine");
    add("the call opener mentioned in reasoning",
        std::string("I could write ") + (is_qwen(m) ? "<tool_call>\n" : "<function name=\"") +
        " here.\n</think>\n\nNo.");
    add("a second </think>", T + "Answer.\n</think>\nmore");
    add("a turn terminator at the end", T + "Answer." + m.terminators[0]);
    add("trailing spaces and newlines", T + "Answer. \n \n\n");
    add("a call inside a sentence", T + "I will " + C(m, "bash", { { "command", "ls" } }) + " now.");
    if (is_qwen(m)) {
        add("the template's instructions echoed",
            T + "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
                "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for "
                "the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n"
                "</tool_call>");
        add("a call without the newline after its opener",
            T + "<tool_call><function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>");
        add("a bare function without its wrapper",
            T + "<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>");
        add("a function name without its newline",
            T + "<tool_call>\n<function=bash><parameter=command>\nls\n</parameter>\n</function>\n</tool_call>");
        add("a close without its newline",
            T + "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function></tool_call>");
        add("two calls in one wrapper",
            T + "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n"
                "<function=bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>");
        add("a call with no closing wrapper",
            T + "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n"
                "<tool_call>\n<function=read>\n<parameter=path>\n/a\n</parameter>\n</function>\n</tool_call>");
        add("a call closed by its wrapper alone",
            T + "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</tool_call>\n" +
                C(m, "read", { { "path", "/a" } }));
        add("an argument with no newline after its name",
            T + "<tool_call>\n<function=bash>\n<parameter=command>ls</parameter>\n</function>\n</tool_call>");
    } else {
        add("the template's instructions echoed",
            T + "<function name=\"function-name\"><param name=\"param-name\">param-value</param></function>");
        add("the instruction's CDATA example echoed",
            T + "<function name=\"write\"><param name=\"param-name\"><![CDATA[...multi-line value...]]></param></function>");
        add("a value closed with the wrong tag",
            T + "<function name=\"bash\"><param name=\"command\">ls\n</command></function>");
        add("junk inside a call", T + "<function name=\"bash\"><param name=\"command\">ls</param>, "
                                      "then<param name=\"timeout\">3</param></function>");
        add("JSON arguments inside the tag", T + "<function name=\"bash\">{\"command\": \"ls\"}</function>");
    }
    return v;
}

std::vector<std::string> fragments(const Model& m) {
    std::vector<std::string> f = {
        "</think>\n\n", "<think>\n", "ls -la", "30", "-1.5e3", "[1,2]", "{\"a\":true}", "{'a': 1}",
        "hello", "\n", " ", "<![CDATA[", "]]>", "]", "<", "</", "\"", "'", "\\", "{", "}", "true",
        "null", "30 seconds", "中", "🚀", "x", "<|im_end|>", "</s>", "```json\n", "```",
    };
    const std::vector<std::string> q = {
        "<tool_call>\n", "</tool_call>", "<function=bash>\n", "<function=read>\n",
        "<function=ping>\n", "<function=configure>\n", "<function=zzz>\n", "</function>\n",
        "<parameter=command>\n", "<parameter=timeout>\n", "<parameter=path>\n",
        "<parameter=extra>\n", "<parameter=level>\n", "<parameter=xx>\n", "</parameter>\n",
        "</parameter", "<tool_call>", "<function=", ">\n",
    };
    const std::vector<std::string> mc = {
        "<function name=\"bash\">", "<function name=\"read\">", "<function name=\"ping\">",
        "<function name=\"configure\">", "<function name=\"zz\">", "</function>",
        "<param name=\"command\">", "<param name=\"timeout\">", "<param name=\"path\">",
        "<param name=\"extra\">", "<param name=\"xx\">", "</param>", "</param", "<function name=\"",
        "\">",
    };
    for (const std::string& s : is_qwen(m) ? q : mc) f.push_back(s);
    return f;
}

std::string mutate(std::mt19937& rng, std::string s, const std::vector<std::string>& frag) {
    const int n = 1 + (int)(rng() % 4);
    for (int k = 0; k < n; ++k) {
        const size_t at = s.empty() ? 0 : rng() % (s.size() + 1);
        switch (rng() % 4) {
        case 0: s.insert(at, frag[rng() % frag.size()]); break;
        case 1: if (at < s.size()) s.erase(at, 1 + rng() % 8); break;
        case 2: if (at < s.size()) s[at] = (char)(rng() & 0xFF); break;
        default: s.insert(at, 1, (char)(rng() & 0xFF)); break;
        }
    }
    return s;
}

std::string long_reply(const Model& m, size_t target) {
    std::string think, file, prose;
    while (think.size() < target / 4) think += "Considering the failing assertion in test_add again. ";
    while (file.size() < target / 2) file += "    x = compute(a, b)  # <tag> & \"quoted\" ]] \n";
    while (prose.size() < target / 4) prose += "The change keeps the interface and fixes the sum. ";
    return think + "\n</think>\n\n" + prose + "\n" +
           C(m, "write", { { "path", "/work/big.py" }, { "content", file } });
}

}  /* namespace chatdiff */
}  /* namespace rad */
