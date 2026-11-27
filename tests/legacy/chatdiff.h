/* chatdiff.h -- the streaming reply parser beside the legacy parser, on the same request.
 *
 * Shared by tests/chatparse_test.cpp and tools/rad_chatparse.cpp. A Bench holds one model's chat
 * template, its reply format and, for one rendered request, both parsers: the legacy path
 * (toolparse_legacy.h, `old`) and the streaming parser (core/text/chatparse.h, `new`). compare()
 * says whether they agree on a reply and, if not, which side left its grammar; check_stream()
 * feeds a reply in pieces and checks the streaming contract on every piece.
 */
#pragma once
#include "text/chat.h"
#include "text/chatfmt.h"
#include "text/chatparse.h"
#include "toolparse_legacy.h"

struct RadChatFormat;

#include "util/json.h"

#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace rad {
namespace chatdiff {

using ojson = Json;

/* A model: where its template is and how its vocabulary spells the tokens a template reads. */
struct Model {
    std::string name;
    std::string dir;                          /* under the fixture root */
    std::string bos, eos;
    std::vector<std::string> terminators;     /* its end-of-generation spellings */
    /* The architecture plugin's reply format, when it declares one; null derives the format from
     * the template, as the engine does. */
    const RadChatFormat* declared = nullptr;
};
const std::vector<Model>& models();

/* How a request is rendered. */
struct ChatRequest {
    bool tools = true;
    bool parallel = true;
    bool thinking = true;
    bool reasoning = true;                    /* reasoning_format "auto"; false is "none" */
    std::string schema;                       /* a response format's schema, "" for none */
};
std::string describe(const ChatRequest& r);

/* An agent-shaped tool set, with every argument type the formats distinguish. */
const ojson& agent_tools();

class Bench {
public:
    /* RAD_OK, or a status with *why. `jinja` is the template's text. */
    int load(const Model& m, const std::string& jinja, std::string* why);
    int bind(const ChatRequest& rq, const ojson& tools, std::string* why);

    const Model& model() const { return model_; }
    const ChatFormat& format() const { return fmt_; }
    const ChatPrompt& prompt() const { return prompt_; }
    const ReplyParser& parser() const { return *parser_; }
    std::shared_ptr<const ReplyParser> parser_ptr() const { return parser_; }
    const legacy::ToolParser& old_parser() const { return *old_; }
    const ChatRequest& request() const { return req_; }
    const ChatTemplate& tmpl() const { return *tmpl_; }

    legacy::Answer old_parse(const std::string& text, bool partial = false) const;
    ReplyMessage new_parse(const std::string& text, bool* strict = nullptr) const;

    /* The reply as the template writes an assistant turn (what the model is trained to produce),
     * without the turn terminator the decoder would not render. "" when the template does not
     * render the turn as a continuation of the generation prompt. */
    std::string render_reply(const ojson& assistant) const;

private:
    std::string render_reply_uncached(const ojson& assistant) const;

    Model model_;
    std::string jinja_;
    std::shared_ptr<ChatTemplate> tmpl_;
    ChatFormat fmt_;
    std::shared_ptr<const ReplyFormat> rf_;
    ChatRequest req_;
    ojson tools_;
    ChatPrompt prompt_;
    std::unique_ptr<legacy::ToolParser> old_;
    std::shared_ptr<const ReplyParser> parser_;
    /* render_reply's results, by the message's JSON. */
    mutable std::map<std::string, std::string> replies_;
};

struct Verdict {
    bool same = false;          /* the two answers are identical */
    bool strict = false;        /* the streaming parser read the reply within the grammar */
    bool old_clean = false;     /* the legacy strict parse matched and nothing was lifted */
    bool bare_lift = false;     /* the legacy path lifted a call written without the call opener */
    /* The legacy content still carries a wrapped format's call syntax: a call written without
     * its wrapper, which the legacy lifter takes only when nothing else parses as a call and the
     * streaming parser always reads as one. */
    bool old_call_syntax = false;
    /* A legacy call has a raw control character in its arguments: either its JSON reader took a
     * string holding one (the schema grammar refuses it, continues in its text arm and ends the
     * value at its suffix, where the legacy reader kept reading), or it passed a typed value's
     * text through bare. Neither is JSON, so the streaming parser is not held to it. */
    bool old_raw_control = false;
    std::string detail;         /* both answers, when they differ */
};
Verdict compare(const Bench& b, const std::string& text);

/* Whether a verdict is one the streaming parser is allowed: identical where both read the reply
 * within the grammar, and a documented difference otherwise. "" or the reason it is not. */
std::string judge(const Verdict& v);

/* Feed `text` to a fresh stream cut at `cuts` (ascending offsets). On every piece, the emitted
 * text must be a prefix of `expect` and the held bytes within the bound; at the end, the
 * concatenated deltas and the message must equal `expect`, and the stream must agree with the
 * whole reply read at once on whether it stayed in the format (ReplyStream::strict). Returns ""
 * or the first violation. */
std::string check_stream(const ReplyParser& p, const std::string& text,
                         const std::vector<size_t>& cuts, const ReplyMessage& expect);

std::string show(const ReplyMessage& m);
std::string show(const legacy::Answer& a);
bool same(const ReplyMessage& a, const ReplyMessage& b);

/* Replace every byte that is not part of well-formed UTF-8 with U+FFFD, as the decoder does
 * before any reply reaches a parser. */
std::string sanitize(const std::string& s);

/* ================================================================== the corpora */

bool is_qwen(const Model& m);

/* A call, and an assistant message carrying calls, OpenAI-shaped. */
ojson call(const std::string& name, const ojson& args);
ojson message(const std::string& reasoning, const std::string& content,
              const std::vector<ojson>& calls);

/* Assistant messages that, rendered through a model's template, are replies the model is trained
 * to write: content only, reasoning, one to five calls, every argument type, unicode, and values
 * holding the format's own punctuation. */
std::vector<ojson> rendered_corpus();

/* Hand-written replies in a model's own spelling: shapes models write when they leave their
 * format, and those a streaming parser most easily gets wrong. */
struct Adversarial { std::string name, text; };
std::vector<Adversarial> adversarial(const Model& m);

/* One call in a model's spelling. */
std::string C(const Model& m, const std::string& name,
              const std::vector<std::pair<std::string, std::string>>& args);

/* The pieces random replies are assembled from, and a random edit of a reply. */
std::vector<std::string> fragments(const Model& m);
std::string mutate(std::mt19937& rng, std::string s, const std::vector<std::string>& frag);

/* About `target` bytes of reasoning, prose and one call writing a file. */
std::string long_reply(const Model& m, size_t target);

}  /* namespace chatdiff */
}  /* namespace rad */
