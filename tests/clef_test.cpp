/* clef_test.cpp -- clef's prompt, spans and answer (core/server/clef.h).
 *
 * The head reads spans of the prompt, so the spans are checked as text: the ids of a question's
 * span decode to its rendered instructions, an option's to its rendered semantics. With a
 * tokenizer.json from a clef release in RAD_CLEF_DIR, the ids and spans are also compared with
 * the release's own encode_record over tests/data/golden/clef_encode.json (clef_encode_gen.py).
 */
#include "rad_test.h"

#include "server/adapters.h"
#include "server/clef.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>

using namespace rad;
using namespace rad::server;

namespace {

/* One id a byte, so a prompt decodes back to its exact text and a span to its exact piece. */
struct ByteTokenizer : ITokenizer {
    std::vector<int32_t> encode(const std::string& s, bool, bool) const override {
        std::vector<int32_t> v;
        for (unsigned char c : s) v.push_back(c);
        return v;
    }
    std::string token_bytes(int32_t id) const override { return std::string(1, (char)id); }
    std::string decode(const std::vector<int32_t>& t) const override {
        std::string s;
        for (int32_t x : t) s += (char)x;
        return s;
    }
    int32_t eos() const override { return 0; }
    int32_t bos() const override { return -1; }
    std::unique_ptr<IDetokenizer> detokenizer(const std::vector<std::string>&,
                                              const std::vector<std::string>&) const override {
        return nullptr;
    }
};

int parse(const ITokenizer& tok, const std::string& body, ClefRequest* r, ApiError* e,
          int64_t max_ctx = 0) {
    OaiDeps d;
    d.tok = const_cast<ITokenizer*>(&tok);
    OaiLimits l;
    l.model_id = "clef-flash";
    l.max_ctx = max_ctx;
    return parse_clef_request(body, d, l, "clef-flash", r, e);
}

std::string span(const ITokenizer& tok, const ClefRequest& r, TokenSpan s) {
    return tok.decode(std::vector<int32_t>(r.tokens.begin() + s.first, r.tokens.begin() + s.second));
}

const char* const kBody = R"({"state": "I was charged twice.",
    "questions": {
      "refund": {"type": "noul", "instructions": "Refund?"},
      "team": {"type": "choice", "instructions": "Which team?",
               "criteria": {"technical": "Faults", "billing": "Charges"}},
      "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Later", "Now"]}}})";

}  /* namespace */

TEST(the_prompt_is_the_releases_text) {
    ByteTokenizer tok;
    ClefRequest r;
    ApiError e;
    CHECK_OK(parse(tok, kBody, &r, &e));
    CHECK_EQ(tok.decode(r.tokens),
             "<|im_start|>system\nRead the complete state and schema. Decide every field jointly. "
             "Each answer must be exactly one of that field's allowed options.<|im_end|>\n"
             "<|im_start|>user\nSTATE:\nI was charged twice."
             "\n\nSCHEMA FIELDS:\n"
             "\nFIELD 1\nID: refund\nTYPE: noul\nINSTRUCTION: Refund?\nALLOWED OPTIONS:\n"
             "OPTION 1: {\"description\":\"The proposition is true or the answer is yes.\",\"option_id\":\"true\"}\n"
             "OPTION 2: {\"description\":\"The proposition is false or the answer is no.\",\"option_id\":\"false\"}\n"
             "END FIELD\n"
             "\nFIELD 2\nID: team\nTYPE: choice\nINSTRUCTION: Which team?\nALLOWED OPTIONS:\n"
             "OPTION 1: {\"description\":\"Charges\",\"option_id\":\"billing\"}\n"
             "OPTION 2: {\"description\":\"Faults\",\"option_id\":\"technical\"}\n"
             "END FIELD\n"
             "\nFIELD 3\nID: urgency\nTYPE: score\nINSTRUCTION: How urgent?\nALLOWED OPTIONS:\n"
             "OPTION 1: {\"description\":\"Later\",\"option_id\":\"0\"}\n"
             "OPTION 2: {\"description\":\"Now\",\"option_id\":\"1\"}\n"
             "END FIELD\n"
             "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");
}

TEST(spans_cover_the_instructions_and_each_options_semantics) {
    ByteTokenizer tok;
    ClefRequest r;
    ApiError e;
    CHECK_OK(parse(tok, kBody, &r, &e));
    REQUIRE(r.questions.size() == 3);
    CHECK_EQ(span(tok, r, r.questions[0].question), "Refund?");
    CHECK_EQ(r.questions[1].type_id, 1);
    /* A choice's options are scored sorted; the response keeps the request's order. */
    CHECK(r.questions[1].options == (std::vector<std::string>{ "billing", "technical" }));
    CHECK(r.questions[1].response_order == (std::vector<std::string>{ "technical", "billing" }));
    CHECK_EQ(span(tok, r, r.questions[1].option_spans[0]),
             "{\"description\":\"Charges\",\"option_id\":\"billing\"}");
    CHECK_EQ(span(tok, r, r.questions[2].option_spans[1]), "{\"description\":\"Now\",\"option_id\":\"1\"}");
}

TEST(render_sorts_keys_and_keeps_unicode) {
    CHECK_EQ(clef_render(json::parse(R"({"b": [1, 2.5, null], "a": "ü\n"})")),
             "{\"a\":\"ü\\n\",\"b\":[1,2.5,null]}");
    CHECK_EQ(clef_render(json("as is")), "as is");
}

TEST(bad_requests_are_400s) {
    ByteTokenizer tok;
    ClefRequest r;
    ApiError e;
    CHECK(parse(tok, R"({"questions": {"q": {"type": "noul"}}})", &r, &e) < 0);
    CHECK_EQ(e.param, "state");
    CHECK(parse(tok, R"({"state": "", "questions": {"q": {"type": "noul"}}})", &r, &e) < 0);
    CHECK(parse(tok, R"({"state": "s", "questions": {}})", &r, &e) < 0);
    CHECK_EQ(e.param, "questions");
    CHECK(parse(tok, R"({"state": "s", "questions": {"q": {"instructions": "x"}}})", &r, &e) < 0);
    CHECK_EQ(e.param, "questions.q.type");
    CHECK(parse(tok, R"({"state": "s", "questions": {"q": {"type": "choice", "criteria": {}}}})", &r, &e) < 0);
    CHECK_EQ(e.param, "questions.q.criteria");
    CHECK(parse(tok, R"({"state": "s", "questions": {"q": {"type": "noul", "criteria": {"maybe": "x"}}}})", &r, &e) < 0);
    CHECK(parse(tok, R"({"state": "s", "questions": {"q": {"type": "noul"}}, "extra": 1})", &r, &e) < 0);
    CHECK(parse(tok, R"({"model": "other", "state": "s", "questions": {"q": {"type": "noul"}}})", &r, &e) < 0);
    CHECK_EQ(e.param, "model");
    /* The whole prompt must fit; nothing is cut. */
    CHECK(parse(tok, kBody, &r, &e, 100) < 0);
    CHECK_EQ(e.code, "context_length_exceeded");
}

TEST(answers_are_softmaxes_in_the_releases_shape) {
    ByteTokenizer tok;
    ClefRequest r;
    ApiError e;
    CHECK_OK(parse(tok, kBody, &r, &e));
    json a;
    CHECK_OK(clef_answer(r.questions[0], { 2.0f, 0.0f }, &a, &e));
    CHECK_EQ(a["noul"].get<double>(), std::round(1.0 / (1.0 + std::exp(-2.0)) * 1e4) / 1e4);
    /* Options in head order (billing, technical); the response in request order. */
    CHECK_OK(clef_answer(r.questions[1], { 0.0f, 1.0f }, &a, &e));
    CHECK_EQ(a["choice"].get<std::string>(), "technical");
    CHECK_EQ(a["probabilities"].begin().key(), "technical");
    double total = 0.0;
    for (const auto& kv : a["probabilities"].items()) total += kv.value().get<double>();
    CHECK(std::fabs(total - 1.0) < 1e-3);
    CHECK_OK(clef_answer(r.questions[2], { 0.0f, 0.0f }, &a, &e));
    CHECK_EQ(a["score"].get<double>(), 0.5);
    CHECK_EQ(a["legend"]["1"].get<std::string>(), "Now");
    CHECK(clef_answer(r.questions[2], { 0.0f, NAN }, &a, &e) < 0);
    CHECK(clef_answer(r.questions[2], { 0.0f }, &a, &e) < 0);
}

TEST(ids_and_spans_match_the_releases_encode_record) {
    const char* dir = std::getenv("RAD_CLEF_DIR");
    if (!dir) SKIP("RAD_CLEF_DIR names no clef release with a tokenizer.json");
    VocabBuild vb;
    REQUIRE(parse_tokenizer_json((std::string(dir) + "/tokenizer.json").c_str(), vb) == RAD_OK);
    auto vocab = std::make_shared<Vocab>();
    REQUIRE(vocab->load(std::move(vb)) == RAD_OK);
    TokenizerBridge tok(vocab);
    std::ifstream f(std::string(RAD_GOLDEN_FIXTURES) + "/clef_encode.json");
    REQUIRE(f.is_open());
    for (const auto& g : json::parse(f)) {   /* ordered: the questions keep their order */
        ClefRequest r;
        ApiError e;
        CHECK_OK(parse(tok, g["request"].dump(), &r, &e));
        CHECK(r.tokens == g["input_ids"].get<std::vector<int32_t>>());
        REQUIRE(r.questions.size() == g["questions"].size());
        for (size_t i = 0; i < r.questions.size(); ++i) {
            const auto& gq = g["questions"][i];
            const ClefQuestion& q = r.questions[i];
            CHECK_EQ(q.name, gq["id"].get<std::string>());
            CHECK_EQ(q.type_id, gq["type"].get<int>());
            CHECK_EQ(q.question.first, gq["span"][0].get<int64_t>());
            CHECK_EQ(q.question.second, gq["span"][1].get<int64_t>());
            CHECK(q.options == gq["option_ids"].get<std::vector<std::string>>());
            REQUIRE(q.option_spans.size() == gq["option_spans"].size());
            for (size_t k = 0; k < q.option_spans.size(); ++k) {
                CHECK_EQ(q.option_spans[k].first, gq["option_spans"][k][0].get<int64_t>());
                CHECK_EQ(q.option_spans[k].second, gq["option_spans"][k][1].get<int64_t>());
            }
        }
    }
}

RAD_TEST_MAIN()
