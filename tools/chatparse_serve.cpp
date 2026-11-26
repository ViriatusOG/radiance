/* chatparse_serve.cpp -- rad-chatparse's request mode: one chat request rendered and its replies
 * parsed exactly as the server does it, offline, against a container's own vocabulary and the
 * architecture plugin's declared reply format.
 *
 *   rad-chatparse --request REQ.json --rad MODEL.rad [--override-chat-template T.jinja]
 *                 [--chat-template-kwargs JSON] [--reasoning-format auto|none] [--home DIRS]
 *                 [--no-plugin] [--reply FILE ...]
 *   rad-chatparse --render-raw T.jinja --context CTX.json
 *
 * --request goes through parse_chat_request with the server's own adapters (TokenizerBridge,
 * ChatTemplateBridge, GrammarBridge), so the prompt is the one the server would tokenise, the
 * grammar the one it would compile and the parser the one it would read the reply with. The
 * template is the container's, or --override-chat-template's (trailing newlines stripped, as the
 * engine does). The reply format is the architecture plugin's declaration when it makes one
 * (rad_arch_chat_format, loaded from $RADIANCE_HOME / --home as the engine does), else the
 * template's analysis; --no-plugin forces the latter.
 *
 * Each --reply is a file holding text the model generates after the prompt. It is tokenised with
 * the container's tokenizer (control tokens as themselves), stopped at the first end-of-generation
 * token, decoded token by token through the server's Detokenizer with the request's preserved
 * tokens, and then read twice: whole, as the non-streaming answer (chat_completion_body), and
 * piece by piece through ChatDeltaStream, as the SSE stream, whose chunks are reassembled the way
 * an OpenAI client does. Both messages, both finish reasons and whether they agree are printed,
 * and the request's tool-call grammar is run over the reply's tokens as the sampler masks them
 * (lazy until its trigger; a whole-turn grammar first takes the generation prompt).
 *
 * --render-raw renders a template with an arbitrary JSON context (messages, tools, any variable)
 * through the vendored jinja engine alone, for message shapes the OpenAI API cannot carry.
 *
 * Output is one JSON object on stdout. Exit status: 0, or 1 when the request is refused, a reply
 * cannot be read, the streamed and non-streamed answers disagree, or the grammar refuses a
 * token. */
#include "build/startup.h"
#include "format/radfile.h"
#include "iface.h"
#include "sample/gbnf.h"
#include "sample/grammar.h"
#include "server/adapters.h"
#include "server/oai.h"
#include "text/chatfmt.h"
#include "text/token_view.h"
#include "text/tokenizer.h"

#include "llama/common/chat.h"
#include "llama/common/jinja/runtime.h"

#include "util/json.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace rad;
using ojson = Json;

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage: rad-chatparse --request REQ.json --rad MODEL.rad "
                 "[--override-chat-template T]\n"
                 "                     [--chat-template-kwargs JSON] "
                 "[--reasoning-format auto|none]\n"
                 "                     [--home DIRS] [--no-plugin] [--reply FILE ...]\n"
                 "       rad-chatparse --render-raw T.jinja --context CTX.json\n");
    return 2;
}

/* The SSE chunks of one choice, reassembled as an OpenAI client does: content and reasoning
 * concatenated, each tool call's name and arguments joined by its index. */
ojson reassemble(const std::string& sse, std::string* finish, int* chunks) {
    ojson msg = { { "role", "assistant" }, { "content", "" } };
    std::string reasoning;
    std::vector<ojson> calls;
    bool any_content = false;
    size_t at = 0;
    while (at < sse.size()) {
        size_t nl = sse.find('\n', at);
        if (nl == std::string::npos) nl = sse.size();
        const std::string line = sse.substr(at, nl - at);
        at = nl + 1;
        if (!line.starts_with("data: ")) continue;
        const std::string body = line.substr(6);
        if (body == "[DONE]") continue;
        const ojson j = ojson::parse(body, nullptr, false);
        if (!j.is_object() || !j.contains("choices")) continue;
        ++*chunks;
        for (const ojson& c : j.at("choices")) {
            if (c.contains("finish_reason") && c.at("finish_reason").is_string())
                *finish = c.at("finish_reason").get<std::string>();
            if (!c.contains("delta")) continue;
            const ojson& d = c.at("delta");
            if (d.contains("content") && d.at("content").is_string()) {
                msg["content"] =
                    msg["content"].get<std::string>() + d.at("content").get<std::string>();
                any_content = true;
            }
            if (d.contains("reasoning_content") && d.at("reasoning_content").is_string())
                reasoning += d.at("reasoning_content").get<std::string>();
            if (d.contains("tool_calls"))
                for (const ojson& tc : d.at("tool_calls")) {
                    const size_t i = tc.value("index", (size_t)0);
                    while (calls.size() <= i)
                        calls.push_back(
                            { { "type", "function" },
                              { "function", { { "name", "" }, { "arguments", "" } } } });
                    if (tc.contains("id")) calls[i]["id"] = tc.at("id");
                    if (tc.contains("function")) {
                        const ojson& f = tc.at("function");
                        if (f.contains("name") && f.at("name").is_string())
                            calls[i]["function"]["name"] =
                                calls[i]["function"]["name"].get<std::string>() +
                                f.at("name").get<std::string>();
                        if (f.contains("arguments") && f.at("arguments").is_string())
                            calls[i]["function"]["arguments"] =
                                calls[i]["function"]["arguments"].get<std::string>() +
                                f.at("arguments").get<std::string>();
                    }
                }
        }
    }
    if (!any_content) msg["content"] = nullptr;
    if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
    if (!calls.empty()) msg["tool_calls"] = calls;
    return msg;
}

/* What the two answers must agree on: content (null and "" alike), reasoning, and each call's
 * name and arguments. Call ids are random per answer. */
ojson comparable(const ojson& m) {
    ojson o;
    o["content"] = m.contains("content") && m.at("content").is_string() ? m.at("content")
                                                                        : ojson("");
    o["reasoning_content"] = m.value("reasoning_content", std::string());
    ojson calls = ojson::array();
    if (m.contains("tool_calls"))
        for (const ojson& tc : m.at("tool_calls"))
            calls.push_back({ tc.at("function").at("name"), tc.at("function").at("arguments") });
    o["tool_calls"] = calls;
    return o;
}

int render_raw(const std::string& tpath, const std::string& cpath) {
    std::string src, ctxs;
    if (!read_file_bytes(tpath, &src) || !read_file_bytes(cpath, &ctxs)) {
        std::fprintf(stderr, "rad-chatparse: cannot read %s or %s\n", tpath.c_str(), cpath.c_str());
        return 2;
    }
    while (!src.empty() && (src.back() == '\n' || src.back() == '\r')) src.pop_back();
    const ojson ctx = ojson::parse(ctxs, nullptr, false);
    if (!ctx.is_object()) {
        std::fprintf(stderr, "rad-chatparse: %s is not a JSON object\n", cpath.c_str());
        return 2;
    }
    ojson out;
    try {
        common_chat_template tmpl(src, ctx.value("bos_token", std::string()),
                                  ctx.value("eos_token", std::string()));
        jinja::context jc(tmpl.source());
        jinja::global_from_json(jc, ctx, false);
        jinja::runtime rt(jc);
        const jinja::value res = rt.execute(tmpl.prog);
        out["prompt"] = jinja::runtime::gather_string_parts(res)->as_string().str();
    } catch (const std::exception& e) {
        out["error"] = e.what();
    }
    std::printf("%s\n", out.dump(-1, ' ', false, ojson::error_handler_t::replace).c_str());
    return out.contains("error") ? 1 : 0;
}

}  /* namespace */

int serve_main(int argc, char** argv) {
    std::string req_path, rad_path, tmpl_path, kwargs, rfmt = "auto", home, raw_tmpl, raw_ctx;
    std::vector<std::string> replies;
    bool plugin = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](std::string* dst) {
            if (i + 1 >= argc) return false;
            *dst = argv[++i];
            return true;
        };
        bool ok = true;
        if (a == "--request") ok = val(&req_path);
        else if (a == "--rad") ok = val(&rad_path);
        else if (a == "--override-chat-template") ok = val(&tmpl_path);
        else if (a == "--chat-template-kwargs") ok = val(&kwargs);
        else if (a == "--reasoning-format") ok = val(&rfmt);
        else if (a == "--home") ok = val(&home);
        else if (a == "--no-plugin") plugin = false;
        else if (a == "--reply") { std::string r; ok = val(&r); replies.push_back(r); }
        else if (a == "--render-raw") ok = val(&raw_tmpl);
        else if (a == "--context") ok = val(&raw_ctx);
        else ok = false;
        if (!ok) return usage();
    }
    if (!raw_tmpl.empty()) return raw_ctx.empty() ? usage() : render_raw(raw_tmpl, raw_ctx);
    if (req_path.empty() || rad_path.empty()) return usage();

    /* ---------------------------------------------------------------- the vocabulary, as
     * Engine::load_vocab reads it out of a container */
    RadFile f;
    if (f.open(rad_path.c_str()) < 0) return 1;
    VocabSection vs = f.vocab();
    if (!vs) {
        std::fprintf(stderr, "rad-chatparse: %s has no vocabulary\n", rad_path.c_str());
        return 1;
    }
    VocabBuild vb;
    if (vocab_deserialize(f.base(), (uint64_t)f.size(), vs.h, vb) < 0) return 1;
    if (!tmpl_path.empty()) {
        std::string t;
        if (!read_file_bytes(tmpl_path, &t)) {
            std::fprintf(stderr, "rad-chatparse: cannot read %s\n", tmpl_path.c_str());
            return 1;
        }
        while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
        vb.chat_template = t;
    }
    if (const char* ids = f.get_s("generation.eos_token_id", nullptr)) {
        const char* q = ids;
        while (*q) {
            char* end = nullptr;
            const long v = std::strtol(q, &end, 10);
            if (end == q) break;
            if (v >= 0 && v <= INT32_MAX) vb.eos_ids.push_back((int32_t)v);
            q = end;
        }
    }
    auto vocab = std::make_shared<Vocab>();
    if (vocab->load(std::move(vb)) < 0) return 1;

    /* ---------------------------------------------------------------- the reply format */
    const RadChatFormat* declared = nullptr;
    LoadedPlugins lp;
    if (plugin) {
        const RadFileHeader& h = f.header();
        if (rad_tools_load_plugins(home.empty() ? rad_home() : home, rad_hierarchy_from_env(),
                                   f.str(h.arch_id), f.str(h.quant), &lp) < 0) {
            std::fprintf(stderr, "rad-chatparse: the architecture plugin did not load\n");
            return 1;
        }
        RadModelMeta meta{};
        meta.arch_id = f.str(h.arch_id);
        declared = rad_tools_chat_format(lp, meta);
    }
    server::TokenizerBridge tok(vocab);
    server::ChatTemplateBridge chat;
    if (chat.init(*vocab, declared, lp.arch_plugin) < 0) return 1;
    TokenView view;
    if (view.init(vocab) < 0) return 1;
    server::GrammarBridge grammar(&view);

    server::OaiDeps d;
    d.tok = &tok;
    d.chat = &chat;
    d.grammar = &grammar;
    server::OaiLimits lim;
    lim.reasoning_format = rfmt;
    if (!kwargs.empty()) {
        const ojson kw = ojson::parse(kwargs, nullptr, false);
        if (!kw.is_object()) {
            std::fprintf(stderr, "rad-chatparse: --chat-template-kwargs is not a JSON object\n");
            return 2;
        }
        for (const auto& [k, v] : kw.items()) lim.default_template_kwargs[k] = v.dump();
    }

    /* What the template's analysis alone reads, beside whatever the plugin declares. */
    ojson derived;
    {
        ChatFormat tf;
        std::string refused, why;
        const std::string bos = vocab->bos() >= 0 ? vocab->text(vocab->bos()) : std::string();
        const std::string eos = vocab->eos() >= 0 ? vocab->text(vocab->eos()) : std::string();
        if (chat_format_from_template(vocab->chat_template(), bos, eos, &tf, &refused, &why) < 0) {
            derived["error"] = why;
        } else {
            derived["describe"] = chat_format_describe(tf);
            derived["tools_refused"] = refused;
            derived["fields"] = { { "section_open", tf.section_open },
                                  { "section_close", tf.section_close },
                                  { "call_open", tf.call_open }, { "call_close", tf.call_close },
                                  { "name_prefix", tf.name_prefix },
                                  { "name_suffix", tf.name_suffix },
                                  { "call_end", tf.call_end }, { "arg_prefix", tf.arg_prefix },
                                  { "arg_name_suffix", tf.arg_name_suffix },
                                  { "arg_value_prefix", tf.arg_value_prefix },
                                  { "arg_value_suffix", tf.arg_value_suffix },
                                  { "reasoning_open", tf.reasoning_open },
                                  { "reasoning_close", tf.reasoning_close } };
        }
    }

    std::string body;
    if (!read_file_bytes(req_path, &body)) {
        std::fprintf(stderr, "rad-chatparse: cannot read %s\n", req_path.c_str());
        return 1;
    }
    server::OaiRequest r;
    server::ApiError err;
    ojson out;
    out["plugin"] = lp.arch_plugin;
    out["declared_format"] = declared != nullptr;
    out["tool_refusal"] = chat.tool_refusal();
    out["template_analysis"] = derived;
    if (server::parse_chat_request(body, d, lim, &r, &err) < 0) {
        out["error"] = { { "status", err.status }, { "message", err.message },
                         { "param", err.param } };
        std::printf("%s\n", out.dump(-1, ' ', false, ojson::error_handler_t::replace).c_str());
        return 1;
    }
    r.id = "chatcmpl-check";
    r.created = 1761500000;
    out["prompt"] = r.prompts.empty() ? std::string() : r.prompts[0].text;
    out["prompt_tokens"] = r.prompts.empty() ? 0 : r.prompts[0].tokens.size();
    out["parse_tools"] = r.parse_tools;
    out["preserved_tokens"] = r.preserved_tokens;
    out["grammar_lazy"] = r.sp.grammar_lazy;
    out["grammar_bytes"] = r.sp.grammar.size();
    ojson trig = ojson::array();
    for (const auto& t : r.sp.grammar_triggers) trig.push_back(t.value);
    out["grammar_triggers"] = trig;

    /* ---------------------------------------------------------------- the replies */
    int bad = 0;
    out["replies"] = ojson::array();
    for (const std::string& rp : replies) {
        ojson one;
        one["file"] = rp;
        std::string text;
        if (!read_file_bytes(rp, &text)) {
            one["error"] = "cannot read the file";
            out["replies"].push_back(one);
            ++bad;
            continue;
        }
        /* The model's tokens, up to the first end-of-generation one, which stops the request and
         * is not decoded. */
        std::vector<int32_t> ids = tok.encode(text, false, true);
        size_t n = 0;
        while (n < ids.size() && !vocab->is_eog(ids[n])) ++n;
        one["tokens"] = n;
        one["stopped_at_eog"] = n < ids.size();

        /* The request's grammar, as the sampler runs it: every token of the reply must be in the
         * mask when it is sampled, and the end of the turn admissible after the last. */
        if (!r.sp.grammar.empty()) {
            std::vector<std::string> words;
            std::vector<int32_t> trig_tokens;
            for (const auto& t : r.sp.grammar_triggers) {
                std::vector<int32_t> tt;
                view.tokenize_special(t.value, &tt);
                if (tt.size() == 1) trig_tokens.push_back(tt[0]);
                else words.push_back(t.value);
            }
            std::unique_ptr<GbnfGrammar> g;
            std::string gerr;
            ojson gj;
            if (GbnfGrammar::create(r.sp.grammar, "root", &view, r.sp.grammar_lazy, words,
                                    trig_tokens, &g, &gerr) < 0) {
                gj["error"] = gerr;
                ++bad;
            } else if (!r.sp.grammar_lazy && !r.sp.grammar_prefix.empty() &&
                       g->accept_str(r.sp.grammar_prefix) < 0) {
                /* A grammar that is the whole turn starts at the generation prompt, which the
                 * sampler replays into it first (grammar.cpp, GrammarSpec::prefix). */
                gj["error"] = "the generation prompt is not admitted";
                ++bad;
            } else {
                std::vector<uint32_t> mask((size_t)grammar_mask_words(view.n_tokens()));
                int64_t refused = -1;
                for (size_t i = 0; i < n && refused < 0; ++i) {
                    if (g->fill_mask(mask.data(), (int64_t)mask.size()) < 0 ||
                        !(mask[(size_t)ids[i] / 32] >> ((size_t)ids[i] % 32) & 1u) ||
                        g->accept_token(ids[i]) < 0)
                        refused = (int64_t)i;
                }
                gj["admitted"] = refused < 0;
                if (refused >= 0) {
                    gj["refused_token"] = refused;
                    gj["refused_piece"] = vocab->text(ids[(size_t)refused]);
                    ++bad;
                }
                /* A lazy grammar that never triggered constrains nothing, the end included. */
                gj["triggered"] = !g->awaiting_trigger();
                gj["end_admissible"] = refused < 0 && (g->awaiting_trigger() || g->complete());
            }
            one["grammar"] = gj;
        }
        Detokenizer dt(vocab, {}, /*render_special=*/false, r.preserved_tokens);
        std::vector<std::string> pieces;
        std::string all;
        for (size_t i = 0; i < n; ++i) {
            std::string p = dt.push(ids[i]);
            if (!p.empty()) { all += p; pieces.push_back(std::move(p)); }
        }
        std::string tail = dt.flush();
        if (!tail.empty()) { all += tail; pieces.push_back(std::move(tail)); }
        one["decoded"] = all;

        /* Non-streaming: the server's assembly (server.cpp, the chat branch). */
        server::ChoiceOut c;
        c.finish = server::Finish::Stop;
        if (r.reply_parser) {
            std::unique_ptr<server::IReplyReader> rd = r.reply_parser->open();
            rd->feed(all, nullptr);
            rd->finish(nullptr);
            c.msg = rd->message();
            const bool complete = r.parse_tools && server::calls_complete(c.msg);
            if (complete) c.finish = server::Finish::ToolCalls;
            if (c.msg.open_call) {
                c.msg.content = rd->content_with_open_call(all);
                c.msg.tool_calls.pop_back();
                c.msg.open_call = false;
            }
        } else {
            c.msg.content = all;
        }
        int k = 0;
        for (auto& tc : c.msg.tool_calls) tc.id = "call_" + std::to_string(k++);
        const ojson resp = ojson::parse(server::chat_completion_body(r, { c }, server::UsageOut()),
                                        nullptr, false);
        const ojson& ch = resp.at("choices").at(0);
        one["message"] = ch.at("message");
        one["finish_reason"] = ch.at("finish_reason");

        /* Streaming: one push per decoded piece, as the server pushes per read. */
        server::ChatDeltaStream st(r, 0, r.reply_parser.get());
        std::string sse = st.first_chunk() + st.started();
        for (const std::string& p : pieces) sse += st.push(p, false);
        sse += st.final_chunk(server::Finish::Stop, server::TimingsOut(), server::UsageOut());
        std::string sfin;
        int chunks = 0;
        const ojson smsg = reassemble(sse, &sfin, &chunks);
        one["stream"] = { { "message", smsg }, { "finish_reason", sfin }, { "chunks", chunks },
                          { "pieces", pieces.size() } };
        const bool agree = comparable(smsg) == comparable(one["message"]) &&
                           sfin == one["finish_reason"].get<std::string>();
        one["stream_agrees"] = agree;
        if (!agree) ++bad;
        out["replies"].push_back(one);
    }
    std::printf("%s\n", out.dump(-1, ' ', false, ojson::error_handler_t::replace).c_str());
    return bad ? 1 : 0;
}
