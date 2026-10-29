#include "text/chat.h"
#include "text/tokenizer.h"

/* Lifted headers: only this file, chatfmt.cpp and chatparse.cpp include them (see chat.h). */
#include "llama/common/chat.h"
/* For jinja::enable_debug (RADIANCE_DEBUG_JINJA). */
#include "llama/common/jinja/runtime.h"
#include "shim/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstdlib>

namespace rad {

using ojson = nlohmann::ordered_json;

struct ChatTemplate::Impl {
    common_chat_templates_ptr tmpls;
    ChatCaps                  caps;
    std::string               src;
    bool                      add_bos = false, add_eos = false;
};

/* The template's own words: the lifted autoparser wraps whatever the render threw in a sentence
 * about parser generation, which the render's failure is not. */
static std::string template_words(const char* what) {
    static const char kWrap[] = "Unable to generate parser for this template. Automatic parser "
                                "generation failed: ";
    std::string w = what ? what : "";
    if (w.compare(0, sizeof kWrap - 1, kWrap) == 0) w.erase(0, sizeof kWrap - 1);
    return w;
}

ChatTemplate::ChatTemplate() : impl_(std::make_unique<Impl>()) {}
ChatTemplate::~ChatTemplate() = default;

const char* ChatTemplate::chatml_source() {
    /* Byte-identical to CHATML_TEMPLATE_SRC in the lift, copied so the lifted header stays
     * unedited. Keep in sync on a vendor refresh. */
    return
        "{%- for message in messages -%}\n"
        "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n"
        "{%- endfor -%}\n"
        "{%- if add_generation_prompt -%}\n"
        "  {{- '<|im_start|>assistant\n' -}}\n"
        "{%- endif -%}";
}

int ChatTemplate::load(const std::string& jinja_source, const std::string& bos,
                       const std::string& eos, bool add_bos, bool add_eos) {
    /* Gates the vendored LOG_DBG to stderr. Set here, where the template is first touched,
     * because the auto-parser analyses it in apply() and again in chatfmt.cpp; any later is too
     * late to trace a format the analyzer could not handle. */
    rad_vendor_log::verbose() = (int)log_level() >= (int)Log::Debug;

    /* The template engine's own trace. The capability probe in common_chat_templates_init
     * swallows every exception, so a template without tool support and one that throws on the
     * probe both report tools=0; this trace tells them apart. Very verbose: it logs every
     * statement of every render, serving ones included. */
    if (const char* dbg = std::getenv("RADIANCE_DEBUG_JINJA"); dbg && *dbg && *dbg != '0')
        jinja::enable_debug(true);

    const bool fallback = jinja_source.empty();
    if (fallback) {
        /* ChatML is only a guess at the prompt format, so warn once. */
        RAD_WARN("chat: the model declares no chat template; falling back to ChatML. "
                 "If the model was not trained on ChatML its replies will be subtly wrong.");
    }
    const std::string src = fallback ? std::string(chatml_source()) : jinja_source;

    try {
        /* model = nullptr: what would be read off a llama_model is passed in as overrides
         * (vendor/shim/llama.h). */
        impl_->tmpls = common_chat_templates_init(nullptr, src, bos, eos);
    } catch (const std::exception& e) {
        RAD_ERR("chat: template did not compile: %s", e.what());
        impl_->tmpls.reset();
        return RAD_E_FORMAT;
    }
    if (!impl_->tmpls) return RAD_E_FORMAT;

    impl_->src = src;
    impl_->add_bos = add_bos;
    impl_->add_eos = add_eos;

    const std::map<std::string, bool> m = common_chat_templates_get_caps(impl_->tmpls.get());
    auto get = [&](const char* k, bool dflt) {
        auto it = m.find(k);
        return it == m.end() ? dflt : it->second;
    };
    impl_->caps.tools               = get("supports_tools", true);
    impl_->caps.tool_calls          = get("supports_tool_calls", true);
    impl_->caps.system_role         = get("supports_system_role", true);
    impl_->caps.parallel_tool_calls = get("supports_parallel_tool_calls", true);
    impl_->caps.preserve_reasoning  = get("supports_preserve_reasoning", false);
    impl_->caps.string_content      = get("supports_string_content", true);
    impl_->caps.typed_content       = get("supports_typed_content", false);
    impl_->caps.object_arguments    = get("supports_object_arguments", false);

    /* The thinking probe is the first full render, through the reply-format analysis every render
     * runs. A template that compiles but cannot render a one-message conversation there (it calls
     * a function this engine lacks, or the analysis cannot generate its parser) would fail every
     * request the same way; it is refused here as a value, since the throw would end the
     * process. */
    bool thinking = false;
    try {
        thinking = common_chat_templates_support_enable_thinking(impl_->tmpls.get());
    } catch (const std::exception& e) {
        RAD_ERR("chat: the template compiles but does not render a conversation: %s",
                template_words(e.what()).c_str());
        impl_->tmpls.reset();
        return RAD_E_FORMAT;
    }
    RAD_INFO("chat: template loaded (%zu bytes)%s; tools=%d tool_calls=%d system=%d thinking=%d",
             src.size(), fallback ? " [ChatML fallback]" : "",
             (int)impl_->caps.tools, (int)impl_->caps.tool_calls,
             (int)impl_->caps.system_role, (int)thinking);
    return RAD_OK;
}

int ChatTemplate::load_from_vocab(const Vocab& v) {
    /* Templates reference bos_token/eos_token by text; Vocab::text() returns a control token's
     * literal spelling. */
    const std::string bos = v.bos() >= 0 ? v.text(v.bos()) : std::string();
    const std::string eos = v.eos() >= 0 ? v.text(v.eos()) : std::string();
    return load(v.chat_template(), bos, eos, v.add_bos(), v.add_eos());
}

bool ChatTemplate::ready() const { return impl_ && impl_->tmpls != nullptr; }
const ChatCaps& ChatTemplate::caps() const { return impl_->caps; }
const std::string& ChatTemplate::source() const { return impl_->src; }

/* Maximum nesting (objects and arrays) of a client's tool or response schema. The JSON-schema
 * converter recurses once per level and names each rule after its full path, so its output grows
 * with the square of the depth; a few thousand levels exhaust memory or the rendering thread's
 * stack. 512 still converts in tens of milliseconds. grammar.cpp bounds the same converter on the
 * path that bypasses the template. */
static constexpr size_t kMaxSchemaDepth = 512;

/* Iterative, because the values it must catch are too deep to recurse over. */
static size_t json_depth(const ojson& root) {
    std::vector<std::pair<const ojson*, size_t>> todo{ { &root, 1 } };
    size_t deepest = 0;
    while (!todo.empty()) {
        const auto [j, d] = todo.back();
        todo.pop_back();
        if (!j->is_structured()) continue;
        deepest = std::max(deepest, d);
        for (const auto& c : *j) todo.push_back({ &c, d + 1 });
    }
    return deepest;
}

/* The conversation with every system message merged, in order, into one at its head, or false
 * when that is the conversation already (none, or one leading) or a system message carries more
 * than text. The Chat Completions API places no constraint on system messages, while many
 * templates raise unless there is exactly one, first. */
static bool lead_system(const ojson& messages, ojson* out) {
    if (!messages.is_array()) return false;
    std::string text;
    size_t n_sys = 0;
    bool first = false;
    for (size_t i = 0; i < messages.size(); ++i) {
        const ojson& m = messages[i];
        if (!m.is_object() || m.value("role", "") != "system") continue;
        if (i == 0) first = true;
        std::string t;
        const auto c = m.find("content");
        if (c != m.end() && c->is_string()) {
            t = c->get<std::string>();
        } else if (c != m.end() && c->is_array()) {
            for (const ojson& part : *c) {
                if (!part.is_object() || part.value("type", "") != "text" ||
                    !part.contains("text") || !part.at("text").is_string())
                    return false;
                t += part.at("text").get<std::string>();
            }
        } else if (c != m.end() && !c->is_null()) {
            return false;
        }
        if (n_sys++ && !text.empty() && !t.empty()) text += "\n\n";
        text += t;
    }
    if (n_sys == 0 || (n_sys == 1 && first)) return false;
    *out = ojson::array();
    out->push_back(ojson{ { "role", "system" }, { "content", text } });
    for (const ojson& m : messages)
        if (!m.is_object() || m.value("role", "") != "system") out->push_back(m);
    return true;
}

int ChatTemplate::apply(const ojson& messages, const ojson& tools, const ChatOptions& opt,
                        ChatPrompt* out) const {
    if (!ready()) {
        RAD_ERR("chat: apply() before load()");
        return RAD_E_STATE;
    }
    if (!out) return RAD_E_INVAL;

    /* Checked before either schema reaches the converter. A response format that does not parse
     * is left for the template to refuse. */
    {
        const ojson schema = opt.json_schema.empty()
                           ? ojson()
                           : ojson::parse(opt.json_schema, nullptr, /*allow_exceptions=*/false);
        if (json_depth(tools) > kMaxSchemaDepth || json_depth(schema) > kMaxSchemaDepth) {
            out->error = "a tool or response schema nests more than " +
                         std::to_string(kMaxSchemaDepth) + " levels deep";
            RAD_WARN("chat: %s", out->error.c_str());
            return RAD_E_INVAL;
        }
    }

    common_chat_templates_inputs in;
    try {
        in.messages = common_chat_msgs_parse_oaicompat(messages);
        if (tools.is_array() && !tools.empty())
            in.tools = common_chat_tools_parse_oaicompat(tools);
        in.tool_choice = common_chat_tool_choice_parse_oaicompat(opt.tool_choice);
    } catch (const std::exception& e) {
        /* A malformed request: return a 400, never throw through the request loop. */
        RAD_WARN("chat: request did not parse: %s", e.what());
        out->error = e.what();
        return RAD_E_INVAL;
    }
    if (!in.tools.empty() && !impl_->caps.tools) {
        RAD_WARN("chat: the model's template declares no tool support; request refused");
        out->error = "this model's chat template declares no tool support";
        return RAD_E_UNSUPPORTED;
    }

    in.grammar               = opt.grammar;
    in.json_schema           = opt.json_schema;
    in.add_generation_prompt = opt.add_generation_prompt;
    in.use_jinja             = true;
    in.parallel_tool_calls   = opt.parallel_tool_calls;
    in.enable_thinking       = opt.enable_thinking;
    in.chat_template_kwargs  = opt.template_kwargs;
    in.add_bos               = opt.add_bos || impl_->add_bos;
    in.add_eos               = opt.add_eos || impl_->add_eos;
    in.reasoning_format      = common_reasoning_format_from_name(opt.reasoning_format);
    in.continue_final_message =
        opt.continue_final.empty()          ? COMMON_CHAT_CONTINUATION_NONE
        : opt.continue_final == "content"   ? COMMON_CHAT_CONTINUATION_CONTENT
        : opt.continue_final == "reasoning_content" ? COMMON_CHAT_CONTINUATION_REASONING
                                            : COMMON_CHAT_CONTINUATION_AUTO;

    common_chat_params p;
    try {
        p = common_chat_templates_apply(impl_->tmpls.get(), in);
    } catch (const std::exception& e) {
        /* A template that takes one system message, first, refuses a conversation that has
         * another or has it later, which the API allows: it is rendered once more with them
         * merged into one at the head. Anything else the template refuses is the request's. */
        ojson merged;
        bool served = false;
        if (lead_system(messages, &merged)) {
            try {
                in.messages = common_chat_msgs_parse_oaicompat(merged);
                p = common_chat_templates_apply(impl_->tmpls.get(), in);
                served = true;
                static std::atomic<bool> said{ false };
                if (!opt.quiet && !said.exchange(true))
                    RAD_WARN("chat: the model's chat template refuses system messages after the "
                             "first (%s); such a conversation is rendered with its system "
                             "messages merged, in order, into one at the head. Said once.",
                             template_words(e.what()).c_str());
            } catch (const std::exception&) {
            }
        }
        if (!served) {
            const std::string w = template_words(e.what());
            if (!opt.quiet) RAD_WARN("chat: the model's chat template refused the request: %s",
                                     w.c_str());
            out->error = "the model's chat template refused this conversation: " + w;
            return RAD_E_INVAL;
        }
    }

    out->prompt            = std::move(p.prompt);
    out->generation_prompt = std::move(p.generation_prompt);
    out->grammar           = std::move(p.grammar);
    out->grammar_lazy      = p.grammar_lazy;
    out->preserved_tokens  = std::move(p.preserved_tokens);
    out->additional_stops  = std::move(p.additional_stops);
    out->supports_thinking = p.supports_thinking;
    out->thinking_start_tag = std::move(p.thinking_start_tag);
    out->thinking_end_tag   = std::move(p.thinking_end_tag);
    out->format            = (int)p.format;
    out->parser            = std::move(p.parser);
    out->grammar_triggers.clear();
    for (const common_grammar_trigger& t : p.grammar_triggers)
        out->grammar_triggers.push_back({ (int)t.type, t.value });

    RAD_DEBUG("chat: rendered %zu bytes, format %s, grammar %zu bytes%s, %zu extra stop(s)",
              out->prompt.size(), common_chat_format_name(p.format),
              out->grammar.size(), out->grammar_lazy ? " (lazy)" : "",
              out->additional_stops.size());
    if ((int)log_level() >= (int)Log::Trace) {
        /* The rendered prompt as raw bytes, control tokens included, for checking the template's
         * output format. Trace only, because it is user content. */
        RAD_TRACE("chat: prompt <<<%s>>>", out->prompt.c_str());
        if (!out->grammar.empty()) RAD_TRACE("chat: grammar <<<%s>>>", out->grammar.c_str());
        for (const auto& s : out->additional_stops) RAD_TRACE("chat: extra stop <<<%s>>>", s.c_str());
    }
    return RAD_OK;
}

}  /* namespace rad */
