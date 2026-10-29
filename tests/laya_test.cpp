/* laya_test.cpp -- Laya's sequences, words and answers (core/server/laya.h).
 *
 * A byte tokenizer makes every piece of a sequence decode back to its exact text, so the layout
 * build_head and build_sequence produce is checked as text: [CLS], "<type> question: ...", [SEP],
 * each option behind its [MASK], [SEP], the state, [SEP]. The state cut, the option budget, the
 * refusals and the answer arithmetic (temperature bucket and clamp, softmax, confidence) follow
 * laya's own common.py and agent.py; tests/laya_decision_test.cpp checks the whole path against
 * the package's Agent.predict.
 */
#include "rad_test.h"
#include "server/laya.h"
#include "text/tokenizer.h"

#include "rad_laya.h"

#include <nlohmann/json.hpp>

#include <cmath>

using namespace rad;
using namespace rad::server;

namespace {

enum : int32_t { kCls = 1000, kSep = 1001, kMask = 1002 };

/* One id a byte; the specials above the bytes. */
struct ByteTokenizer : ITokenizer {
    std::vector<int32_t> encode(const std::string& s, bool, bool) const override {
        std::vector<int32_t> v;
        for (unsigned char c : s) v.push_back(c);
        return v;
    }
    std::string token_bytes(int32_t id) const override { return std::string(1, (char)id); }
    std::string decode(const std::vector<int32_t>& t) const override {
        std::string s;
        for (int32_t x : t) s += x == kCls ? "<CLS>" : x == kSep ? "<SEP>" : x == kMask ? "<MASK>" : std::string(1, (char)x);
        return s;
    }
    int32_t eos() const override { return 0; }
    int32_t bos() const override { return -1; }
    std::unique_ptr<IDetokenizer> detokenizer(const std::vector<std::string>&,
                                              const std::vector<std::string>&) const override {
        return nullptr;
    }
};

LayaSettings settings(int64_t max_len = 512, int64_t head_max_len = 192) {
    LayaSettings s;
    s.max_len = max_len;
    s.head_max_len = head_max_len;
    s.cls = kCls;
    s.sep = kSep;
    s.mask = kMask;
    s.mask_token = "[MASK]";
    return s;
}

int parse(const std::string& body, LayaRequest* r, ApiError* e, const LayaSettings& s = settings()) {
    static ByteTokenizer tok;
    OaiDeps d;
    d.tok = &tok;
    OaiLimits l;
    l.model_id = "laya";
    return parse_laya_request(body, d, l, "laya", s, r, e);
}

std::string text(const LayaQuestion& q) { return ByteTokenizer().decode(q.tokens); }

}  /* namespace */

TEST(laya_render_is_python_json_dumps_with_the_request_order_kept) {
    const json v = json::parse(R"({"b": [1, 2.5, "x", null], "a": {"é": true, "q": "say \"hi\"\n"}})");
    CHECK_EQ(laya_render(v), std::string(R"({"b": [1, 2.5, "x", null], "a": {"é": true, "q": "say \"hi\"\n"}})"));
    CHECK_EQ(laya_render(json::parse("[]")), std::string("[]"));
    CHECK_EQ(laya_render(json::parse("{}")), std::string("{}"));
    CHECK_EQ(laya_render(json::parse("1.0")), std::string("1.0"));
}

TEST(each_question_is_its_own_sequence_with_the_state_after_its_options) {
    LayaRequest r;
    ApiError e;
    CHECK_EQ(parse(R"({"state": "Charged [MASK] twice.", "questions": {
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": "Charges", "tech": "", "other": null}},
        "level": {"type": "score", "instructions": {"ask": "How urgent?"}, "criteria": ["Low", {"x": 1}]},
        "refund": {"type": "noul", "instructions": "Refund?", "criteria": {"True": "asks for money"}}}})",
                   &r, &e), RAD_OK);
    CHECK_EQ(r.questions.size(), (size_t)3);
    CHECK_EQ(r.state_tokens, (int64_t)std::string("Charged   twice.").size());
    const LayaQuestion& t = r.questions[0];
    CHECK_EQ(text(t), std::string("<CLS>choice question: Which team?<SEP><MASK> billing: Charges"
                                  "<MASK> tech<MASK> other<SEP>Charged   twice.<SEP>"));
    CHECK_EQ(t.type_id, (int)RAD_LAYA_CHOICE);
    CHECK_EQ(t.markers.size(), (size_t)3);
    for (int32_t m : t.markers) CHECK_EQ(t.tokens[(size_t)m], (int32_t)kMask);
    CHECK(t.keys == std::vector<std::string>({ "billing", "tech", "other" }));
    const LayaQuestion& l = r.questions[1];
    CHECK_EQ(text(l), std::string("<CLS>score question: {\"ask\": \"How urgent?\"}<SEP><MASK> level 0: Low"
                                  "<MASK> level 1: {\"x\": 1}<SEP>Charged   twice.<SEP>"));
    CHECK(l.legend == std::vector<std::string>({ "Low", "{\"x\": 1}" }));
    const LayaQuestion& n = r.questions[2];
    CHECK_EQ(text(n), std::string("<CLS>noul question: Refund?<SEP><MASK> false: no, the statement does "
                                  "not hold<MASK> true: asks for money<SEP>Charged   twice.<SEP>"));
    CHECK(n.keys == std::vector<std::string>({ "false", "true" }));

    const std::vector<int32_t> w = laya_words(t);
    CHECK_EQ(w.size(), (size_t)RAD_LAYA_W_MARKERS + 3);
    CHECK_EQ(w[RAD_LAYA_W_N], (int32_t)t.tokens.size());
    CHECK_EQ(w[RAD_LAYA_W_TYPE], (int32_t)RAD_LAYA_CHOICE);
    CHECK_EQ(w[RAD_LAYA_W_K], 3);
    CHECK_EQ(w[RAD_LAYA_W_MARKERS + 2], t.markers[2]);

    const json u = laya_usage(r);
    CHECK_EQ(u["input_tokens"].get<int64_t>(),
             (int64_t)(t.tokens.size() + l.tokens.size() + n.tokens.size()));
    CHECK(u["truncated"] == false);
    CHECK_EQ(u["state_tokens_dropped"].get<int64_t>(), (int64_t)0);
    CHECK(u["truncated_questions"].empty());
}

TEST(a_long_state_is_cut_to_the_room_its_question_leaves_and_usage_says_so) {
    LayaRequest r;
    ApiError e;
    std::string state(200, 'a');
    state += "END";
    const std::string q = R"(, "questions": {"q": {"type": "noul", "instructions": "Ok?"}}})";
    CHECK_EQ(parse("{\"state\": \"" + state + "\"" + q, &r, &e, settings(128, 64)), RAD_OK);
    const LayaQuestion& a = r.questions[0];
    CHECK_EQ(a.tokens.size(), (size_t)128);
    CHECK_EQ(a.tokens.back(), (int32_t)kSep);
    CHECK_EQ(a.tokens[a.tokens.size() - 2], (int32_t)'a');          /* the start is kept */
    const json u = laya_usage(r);
    CHECK(u["truncated"] == true);
    CHECK_EQ(u["state_tokens"].get<int64_t>(), (int64_t)state.size());
    CHECK_EQ(u["state_tokens_dropped"].get<int64_t>(), (int64_t)state.size() - a.state_used);
    CHECK(u["truncated_questions"] == json::array({ "q" }));

    /* A list of turns keeps its end. */
    CHECK_EQ(parse("{\"state\": [\"" + state + "\"]" + q, &r, &e, settings(128, 64)), RAD_OK);
    const std::string s = text(r.questions[0]);
    CHECK(s.size() > 11 && s.substr(s.size() - 11) == "aEND\"]<SEP>");
}

TEST(many_options_share_the_head_budget_and_too_many_are_refused) {
    LayaRequest r;
    ApiError e;
    /* Fifteen options of ~44 bytes leave fewer than 16 of 64: each is cut to
     * max(4, (64 - 16) / 15) = 4, and the instruction keeps max(8, 64 - 60) = 8 tokens. */
    json c = json::object();
    for (int i = 0; i < 15; ++i) c["option number " + std::to_string(i)] = "a fairly long description";
    json body = { { "state", "s" }, { "questions", { { "q", { { "type", "choice" }, { "instructions", "Which?" },
                                                              { "criteria", c } } } } } };
    CHECK_EQ(parse(body.dump(), &r, &e, settings(512, 64)), RAD_OK);
    const LayaQuestion& q = r.questions[0];
    for (size_t i = 0; i + 1 < q.markers.size(); ++i) CHECK_EQ(q.markers[i + 1] - q.markers[i], 4);
    CHECK_EQ(text(q).substr(0, 18), std::string("<CLS>choice q<SEP>"));
    /* Options past the sequence are refused, not answered with fewer options. */
    CHECK(parse(body.dump(), &r, &e, settings(24, 64)) < 0);
    CHECK_EQ(e.status, 400);
}

TEST(what_the_package_refuses_is_a_400_and_images_are_a_501) {
    LayaRequest r;
    ApiError e;
    const char* bad[] = {
        R"({"questions": {"q": {"type": "noul", "instructions": "x"}}})",
        R"({"state": "", "questions": {"q": {"type": "noul", "instructions": "x"}}})",
        R"({"state": "s", "questions": {}})",
        R"({"state": "s", "questions": {"q": {"type": "rank", "instructions": "x"}}})",
        R"({"state": "s", "questions": {"q": {"type": "noul", "instructions": "  "}}})",
        R"({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": []}}})",
        R"({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": [1, 1.0]}}})",
        R"({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": ["a", null]}}})",
        R"({"state": "s", "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["a", null]}}})",
        R"({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}}}})",
        R"({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x", "labels": {}}}})",
        R"({"state": "s", "questions": {" ": {"type": "noul", "instructions": "x"}}})",
    };
    for (const char* b : bad) {
        e = ApiError{};
        CHECK(parse(b, &r, &e) < 0);
        CHECK_EQ(e.status, 400);
    }
    CHECK(parse(R"({"state": "s", "images": ["data:image/png;base64,AA"], "questions": {"q": {"type": "noul", "instructions": "x"}}})",
                &r, &e) < 0);
    CHECK_EQ(e.status, 501);
}

TEST(the_answer_is_the_packages_softmax_rounding_and_confidence) {
    LayaRequest r;
    ApiError e;
    CHECK_EQ(parse(R"({"state": "s", "questions": {
        "c": {"type": "choice", "instructions": "x", "criteria": [1, 2.5, false]},
        "s": {"type": "score", "instructions": "x", "criteria": ["lo", "mid", "hi"]},
        "n": {"type": "noul", "instructions": "x"}}})", &r, &e), RAD_OK);
    LayaSettings s = settings();
    s.temperature[RAD_LAYA_CHOICE] = 9.0;                  /* clamped to 5 */
    s.temperature_by_options["score:3-5"] = 0.1;          /* clamped to 0.5 */
    s.temperature[RAD_LAYA_NOUL] = 2.0;
    auto softmax = [](std::vector<double> z) {
        double m = *std::max_element(z.begin(), z.end()), t = 0;
        for (double& v : z) t += (v = std::exp(v - m));
        for (double& v : z) v /= t;
        return z;
    };
    auto conf = [](const std::vector<double>& p) {
        double h = 0;
        for (double v : p) h -= v * std::log(v);
        return 1.0 - h / std::log((double)p.size());
    };
    auto near = [](double a, double b) { return std::fabs(a - b) < 1.5e-4; };

    json a;
    CHECK_EQ(laya_answer(r.questions[0], { 3.0f, 1.0f, 0.0f, -1e4f }, s, &a, &e), RAD_OK);
    const std::vector<double> pc = softmax({ 3.0 / 5, 1.0 / 5, 0.0 });
    CHECK(a["choice"] == 1);
    CHECK(near(a["probabilities"]["1"].get<double>(), pc[0]));
    CHECK(near(a["probabilities"]["2.5"].get<double>(), pc[1]));
    CHECK(near(a["probabilities"]["false"].get<double>(), pc[2]));
    CHECK(near(a["confidence"].get<double>(), conf(pc)));
    CHECK(text(r.questions[0]).find("<MASK> 2.5<MASK> False<SEP>") != std::string::npos);

    CHECK_EQ(laya_answer(r.questions[1], { 0.0f, 0.5f, 0.25f }, s, &a, &e), RAD_OK);
    const std::vector<double> ps = softmax({ 0.0, 1.0, 0.5 });
    CHECK(near(a["score"].get<double>(), ps[1] + 2 * ps[2]));
    CHECK(near(a["confidence"].get<double>(), conf(ps)));
    CHECK(a["legend"] == json::parse(R"({"0": "lo", "1": "mid", "2": "hi"})"));

    CHECK_EQ(laya_answer(r.questions[2], { -1.0f, 1.0f }, s, &a, &e), RAD_OK);
    CHECK(near(a["noul"].get<double>(), softmax({ -0.5, 0.5 })[1]));
    CHECK(!a.contains("confidence"));

    CHECK(laya_answer(r.questions[2], { NAN, 1.0f }, s, &a, &e) < 0);
}

RAD_TEST_MAIN()
