/* clef.cpp -- clef's prompt, spans and answer (clef.h).
 *
 * Every string below is the release's own, from joint_schema_model.py: the system prompt, the
 * state and schema framing, the field and option lines, and the noul defaults. The prompt is
 * tokenised piece by piece and the ids concatenated, as encode_record does, because each span is
 * the ids of one piece: BPE over the joined text could merge across a span's edge.
 */
#include "clef.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace rad {
namespace server {

namespace {

const char* const kSystemPrompt =
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options.";

bool only_keys(const json& o, std::initializer_list<const char*> allowed, const std::string& path,
               ApiError* err) {
    for (const auto& kv : o.items()) {
        bool ok = false;
        for (const char* k : allowed) ok = ok || kv.key() == k;
        if (!ok) {
            *err = err_unsupported(path + kv.key(), "not a field of a System One request");
            return false;
        }
    }
    return true;
}

/* question_options: each allowed option and its description (null for none). */
int question_options(const std::string& path, const std::string& type, const json& criteria,
                     ClefQuestion* q, std::vector<json>* descs, ApiError* err) {
    if (type == "noul") {
        json c = { { "true", "The proposition is true or the answer is yes." },
                   { "false", "The proposition is false or the answer is no." } };
        if (!criteria.is_null()) {
            if (!criteria.is_object()) {
                *err = err_bad(path + ".criteria", "must be an object with `true` and `false`, or absent");
                return RAD_E_INVAL;
            }
            if (!only_keys(criteria, { "true", "false" }, path + ".criteria.", err)) return RAD_E_INVAL;
            for (const auto& kv : criteria.items()) c[kv.key()] = kv.value();
        }
        q->options = { "true", "false" };
        *descs = { c["true"], c["false"] };
        q->response_order = q->options;
        return RAD_OK;
    }
    if (type == "choice") {
        if (!criteria.is_object() || criteria.empty()) {
            *err = err_bad(path + ".criteria", "must be a non-empty object of option names to descriptions");
            return RAD_E_INVAL;
        }
        std::vector<std::pair<std::string, json>> o;
        for (const auto& kv : criteria.items()) {
            o.emplace_back(kv.key(), kv.value());
            q->response_order.push_back(kv.key());
        }
        /* Python sorts str keys by code point, which UTF-8 byte order preserves. */
        std::sort(o.begin(), o.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (auto& p : o) {
            q->options.push_back(p.first);
            descs->push_back(std::move(p.second));
        }
        return RAD_OK;
    }
    if (!criteria.is_array() || criteria.empty()) {
        *err = err_bad(path + ".criteria", "must be a non-empty array of level descriptions, lowest first");
        return RAD_E_INVAL;
    }
    for (size_t i = 0; i < criteria.size(); ++i) {
        q->options.push_back(std::to_string(i));
        descs->push_back(criteria[i]);
    }
    q->response_order = q->options;
    for (const json& c : criteria) q->legend.push_back(c.dump());
    return RAD_OK;
}

double round4(double v) { return std::round(v * 10000.0) / 10000.0; }

}  /* namespace */

std::string clef_render(const json& v) {
    if (v.is_string()) return v.get<std::string>();
    /* json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False): nlohmann's unordered
     * json keeps its keys sorted by bytes, which is code point order for UTF-8. */
    return nlohmann::json::parse(v.dump()).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

int parse_clef_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                       const std::string& model_id, ClefRequest* out, ApiError* err) {
    *out = ClefRequest{};
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    if (!only_keys(b, { "model", "state", "questions", "images" }, "", err)) return RAD_E_INVAL;
    if (b.contains("model") && (!b["model"].is_string() || b["model"] != model_id)) {
        *err = err_bad("model", "must name the loaded model, " + model_id);
        return RAD_E_INVAL;
    }
    if (!b.contains("state") || (b["state"].is_string() && b["state"].get<std::string>().empty())) {
        *err = err_bad("state", "required: a non-empty string or any JSON value");
        return RAD_E_INVAL;
    }
    if (b.contains("images") && !(b["images"].is_array() && b["images"].empty())) {
        *err = err_not_implemented("images: this server reads clef's state as text only");
        return RAD_E_UNSUPPORTED;
    }
    const json& qs = b.contains("questions") ? b["questions"] : json();
    if (!qs.is_object() || qs.empty()) {
        *err = err_bad("questions", "must be a non-empty object of named questions");
        return RAD_E_INVAL;
    }

    const ITokenizer& tok = *d.tok;
    auto ids = [&](const std::string& text) { return tok.encode(text, false, true); };
    auto append = [](std::vector<int32_t>* to, const std::vector<int32_t>& v) {
        to->insert(to->end(), v.begin(), v.end());
    };

    /* encode_record: the schema first, with spans relative to its own start. */
    std::vector<int32_t> schema = ids("\n\nSCHEMA FIELDS:\n");
    size_t index = 0;
    for (const auto& kv : qs.items()) {
        const std::string path = "questions." + kv.key();
        const json& src = kv.value();
        if (!src.is_object()) { *err = err_bad(path, "must be an object"); return RAD_E_INVAL; }
        if (!only_keys(src, { "type", "instructions", "criteria" }, path + ".", err)) return RAD_E_INVAL;
        const auto ty = src.find("type");
        if (ty == src.end() || !ty->is_string() ||
            (*ty != "noul" && *ty != "choice" && *ty != "score")) {
            *err = err_bad(path + ".type", "must be noul, choice or score");
            return RAD_E_INVAL;
        }
        ClefQuestion q;
        q.name = kv.key();
        q.type = ty->get<std::string>();
        q.type_id = q.type == "noul" ? 0 : q.type == "choice" ? 1 : 2;
        std::vector<json> descs;
        RAD_TRY(question_options(path, q.type, src.contains("criteria") ? src.at("criteria") : json(),
                                 &q, &descs, err));

        append(&schema, ids("\nFIELD " + std::to_string(++index) + "\nID: " + q.name + "\nTYPE: " +
                            q.type + "\nINSTRUCTION: "));
        json instructions = src.contains("instructions") ? src.at("instructions") : json();
        if (instructions.is_null() || instructions == "") instructions = q.name;
        q.question.first = (int64_t)schema.size();
        append(&schema, ids(clef_render(instructions)));
        q.question.second = (int64_t)schema.size();
        append(&schema, ids("\nALLOWED OPTIONS:\n"));
        for (size_t i = 0; i < q.options.size(); ++i) {
            append(&schema, ids("OPTION " + std::to_string(i + 1) + ": "));
            json semantics = { { "option_id", q.options[i] } };
            if (!descs[i].is_null()) semantics["description"] = descs[i];
            const int64_t start = (int64_t)schema.size();
            append(&schema, ids(clef_render(semantics)));
            q.option_spans.emplace_back(start, (int64_t)schema.size());
            append(&schema, ids("\n"));
        }
        append(&schema, ids("END FIELD\n"));
        /* A span the tokenizer left empty would make the head average nothing. */
        if (q.question.first == q.question.second) {
            *err = err_bad(path + ".instructions", "renders to no tokens");
            return RAD_E_INVAL;
        }
        out->questions.push_back(std::move(q));
    }

    const std::vector<int32_t> prefix = ids(std::string("<|im_start|>system\n") + kSystemPrompt +
                                            "<|im_end|>\n<|im_start|>user\nSTATE:\n");
    const std::vector<int32_t> state = ids(clef_render(b["state"]));
    const std::vector<int32_t> suffix =
        ids("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");
    const int64_t n = (int64_t)(prefix.size() + state.size() + schema.size() + suffix.size());
    if (lim.max_ctx > 0 && n > lim.max_ctx) {
        ApiError e = err_bad("state", "the prompt is " + std::to_string(n) + " tokens; this "
                             "deployment's context holds " + std::to_string(lim.max_ctx));
        e.code = "context_length_exceeded";
        e.n_prompt_tokens = n;
        e.n_ctx = lim.max_ctx;
        *err = e;
        return RAD_E_INVAL;
    }
    const int64_t offset = (int64_t)(prefix.size() + state.size());
    for (ClefQuestion& q : out->questions) {
        q.question.first += offset;
        q.question.second += offset;
        for (TokenSpan& s : q.option_spans) { s.first += offset; s.second += offset; }
    }
    out->tokens = prefix;
    append(&out->tokens, state);
    append(&out->tokens, schema);
    append(&out->tokens, suffix);
    return RAD_OK;
}

int clef_answer(const ClefQuestion& q, const std::vector<float>& logits, json* out, ApiError* err) {
    if (logits.size() != q.options.size()) {
        *err = err_internal("the head returned no logits for question " + q.name);
        return RAD_E_STATE;
    }
    double peak = -std::numeric_limits<double>::infinity();
    for (float l : logits) {
        if (!std::isfinite(l)) {
            *err = err_internal("question " + q.name + " scored a non-finite logit");
            return RAD_E_STATE;
        }
        peak = std::max(peak, (double)l);
    }
    std::vector<double> p;
    double total = 0.0;
    for (float l : logits) total += p.emplace_back(std::exp((double)l - peak));
    for (double& v : p) v /= total;
    auto prob = [&](const std::string& option) {
        return p[(size_t)(std::find(q.options.begin(), q.options.end(), option) - q.options.begin())];
    };

    /* systemone_answer. */
    json a = json::object();
    a["type"] = q.type;
    if (q.type == "noul") {
        a["noul"] = round4(prob("true"));
        *out = std::move(a);
        return RAD_OK;
    }
    json probs = json::object();
    if (q.type == "choice") {
        std::string best = q.response_order[0];
        for (const std::string& o : q.response_order) if (prob(o) > prob(best)) best = o;
        a["choice"] = best;
        a["confidence"] = round4(prob(best));
        for (const std::string& o : q.response_order) probs[o] = round4(prob(o));
        a["probabilities"] = std::move(probs);
    } else {
        double expected = 0.0, top = 0.0;
        json legend = json::object();
        for (size_t i = 0; i < q.options.size(); ++i) {
            expected += (double)i * p[i];
            top = std::max(top, p[i]);
            legend[q.options[i]] = json::parse(q.legend[i]);
            probs[q.options[i]] = round4(p[i]);
        }
        a["score"] = round4(expected);
        a["confidence"] = round4(top);
        a["legend"] = std::move(legend);
        a["probabilities"] = std::move(probs);
    }
    *out = std::move(a);
    return RAD_OK;
}

}  /* namespace server */
}  /* namespace rad */
