/* systemone_test.cpp -- /v1/systemone's adapter (core/server/systemone.h) and its handler.
 *
 * d1 reads exact text: a reworded instruction or a different option code is a different question
 * to a model trained on its own prompts. So rendering is checked as text against strings copied
 * from d1's prompt.py, and the readout against hand-computed softmaxes. End-to-end parity with d1
 * itself needs a serve against the model (docs/SYSTEMONE.md).
 */
#include "rad_test.h"

#include "server/server.h"
#include "server/sink.h"
#include "server/systemone.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <thread>

using namespace rad;
using namespace rad::server;

namespace {

/* A vocabulary in which d1's answer spellings are single tokens (yes/no in three cases, the
 * letters, " A".." Z", the digits) and every other byte is one id from 1000 up, so a prompt
 * decodes back to its exact text. "Q" and "07" are not single tokens, which exercises the
 * fallback codebook. Every string passed to encode is logged, to observe the tokenisation split. */
struct DictTokenizer : ITokenizer {
    std::map<std::string, int32_t> whole;
    mutable std::vector<std::string> log;
    mutable std::mutex mu;
    DictTokenizer() {
        int32_t id = 1;
        for (const char* s : { "yes", "Yes", "YES", "no", "No", "NO", "<|startoftext|>" })
            whole[s] = id++;
        for (char c = 'A'; c <= 'Z'; ++c) {
            if (c == 'Q') continue;
            whole[std::string(1, c)] = id++;
            whole[" " + std::string(1, c)] = id++;
        }
        for (char c = '0'; c <= '9'; ++c) whole[std::string(1, c)] = id++;
        whole["00"] = id++;
    }
    std::vector<int32_t> encode(const std::string& s, bool, bool) const override {
        { std::lock_guard<std::mutex> lk(mu); log.push_back(s); }
        auto it = whole.find(s);
        if (it != whole.end()) return { it->second };
        if (s == "Q") return { 1000 + 'Q', 3000 };     /* two ids, the second spelled "" */
        std::vector<int32_t> v;
        for (unsigned char c : s) v.push_back(1000 + c);
        return v;
    }
    std::string token_bytes(int32_t id) const override {
        if (id == 3000) return "";
        if (id >= 1000) return std::string(1, (char)(id - 1000));
        for (const auto& kv : whole) if (kv.second == id) return kv.first;
        return "";
    }
    std::string decode(const std::vector<int32_t>& t) const override {
        std::string s;
        for (int32_t x : t) s += token_bytes(x);
        return s;
    }
    int32_t eos() const override { return 0; }
    int32_t bos() const override { return whole.at("<|startoftext|>"); }
    std::unique_ptr<IDetokenizer> detokenizer(const std::vector<std::string>&,
                                              const std::vector<std::string>&) const override {
        return nullptr;
    }
    int32_t id(const std::string& s) const { return whole.at(s); }
};

OaiLimits limits(int64_t max_ctx = 0) {
    OaiLimits l;
    l.model_id = "d1";
    l.max_ctx = max_ctx;
    return l;
}

int parse(DictTokenizer& tok, const std::string& body, DecisionRequest* r, ApiError* e,
          int64_t max_ctx = 0) {
    OaiDeps d;
    d.tok = &tok;
    return parse_decision_request(body, d, limits(max_ctx), "d1", r, e);
}

}  /* namespace */

TEST(a_single_question_is_d1s_prompt_as_one_text) {
    DictTokenizer tok;
    DecisionRequest r;
    ApiError e;
    CHECK_OK(parse(tok, R"({"state": "I was charged twice.",
        "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}}})",
                   &r, &e));
    REQUIRE(r.questions.size() == 1);
    CHECK_EQ(tok.decode(r.questions[0].tokens),
             "<|startoftext|><|im_start|>user\nI was charged twice.\n\n\nQUESTION:\n"
             "Is a refund requested?\n\nReply with yes or no only.<|im_end|>\n"
             "<|im_start|>assistant\n");
    CHECK_EQ(r.trunk, 0);
    CHECK_EQ(r.input_tokens, (int64_t)r.questions[0].tokens.size());
    /* One encode of the whole prompt, after the candidates' single-token probes. */
    CHECK_EQ(tok.log.back(), tok.decode(r.questions[0].tokens));
    /* yes/Yes/YES then no/No/NO, max-pooled per answer. */
    CHECK(r.questions[0].labels == (std::vector<std::string>{ "true", "false" }));
    CHECK_EQ(r.questions[0].candidates.size(), (size_t)6);
    REQUIRE(r.questions[0].groups.size() == 2);
    CHECK_EQ(r.questions[0].groups[0].size(), (size_t)3);
}

TEST(several_questions_share_a_head_tokenised_on_its_own) {
    DictTokenizer tok;
    DecisionRequest r;
    ApiError e;
    CHECK_OK(parse(tok, R"({"state": {"b": 1, "a": [true, null]},
        "questions": {
          "z": {"type": "score", "instructions": "How urgent?", "criteria": ["Low", "High"]},
          "a": {"type": "noul", "instructions": "Late?",
                "criteria": {"true": "Not on time", "false": null}}}})", &r, &e));
    REQUIRE(r.questions.size() == 2);
    /* Insertion order, for the questions and for the state's keys (indented JSON). */
    CHECK_EQ(r.questions[0].name, "z");
    const std::string head = "<|startoftext|><|im_start|>user\n{\n  \"b\": 1,\n  \"a\": [\n    true,\n"
                             "    null\n  ]\n}\n\n\nQUESTION:\n";
    CHECK_EQ(r.trunk, (int32_t)head.size());        /* one id a byte in this vocabulary */
    CHECK_EQ(tok.decode(r.questions[0].tokens),
             head + "How urgent?\n\n0 Low\n1 High\n\nReply with a single digit 0-1 only."
                    "<|im_end|>\n<|im_start|>assistant\n");
    CHECK_EQ(tok.decode(r.questions[1].tokens),
             head + "Late?\nYes: Not on time\nNo: None\n\nReply with yes or no only."
                    "<|im_end|>\n<|im_start|>assistant\n");
    /* The head was encoded once, as itself; usage counts it once. */
    int heads = 0;
    for (const std::string& s : tok.log) heads += s == head;
    CHECK_EQ(heads, 1);
    CHECK_EQ(r.input_tokens, (int64_t)(r.questions[0].tokens.size() + r.questions[1].tokens.size()) -
                             r.trunk);
}

TEST(choice_codes_follow_d1s_codebook) {
    DictTokenizer tok;
    {   /* Names that are not letters: A, B, C, and a description falling back to the name. */
        DecisionRequest r;
        ApiError e;
        CHECK_OK(parse(tok, R"({"state": null, "questions": {"team": {"instructions": "Who?",
            "criteria": {"billing": "Charges", "tech_support": null, "fraud": ""}}}})", &r, &e));
        const DecisionQuestion& q = r.questions[0];
        CHECK_EQ(q.type, "choice");                    /* d1's default type */
        CHECK_EQ(tok.decode(q.tokens), "<|startoftext|><|im_start|>user\nWho?\n\nOptions:\nA Charges\n"
                 "B tech support\nC fraud\n\nReply with the option code only.<|im_end|>\n"
                 "<|im_start|>assistant\n");
        REQUIRE(q.groups.size() == 3);
        /* Each option reads its code and " code". */
        CHECK_EQ(q.candidates[q.groups[1][0]], tok.id("B"));
        CHECK_EQ(q.candidates[q.groups[1][1]], tok.id(" B"));
    }
    {   /* Letters are their own codes, except one that is not a token: Q takes the first free
         * fallback, A, which is still unused because codes are assigned in option order. */
        DecisionRequest r;
        ApiError e;
        CHECK_OK(parse(tok, R"({"state": "s", "questions": {"q": {"type": "choice",
            "instructions": "Pick.", "criteria": {"Q": "first", " B ": "second"}}}})", &r, &e));
        const std::string text = tok.decode(r.questions[0].tokens);
        CHECK(text.find("Options:\nA first\nB second\n") != std::string::npos);
        CHECK(r.questions[0].labels == (std::vector<std::string>{ "Q", " B " }));
    }
    {   /* Over 26 options: 00, 01, ... and every two-digit code but 00 falls back. */
        std::string crit;
        for (int i = 0; i < 27; ++i) crit += (i ? "," : "") + std::string("\"o") + std::to_string(i) + "\": null";
        DecisionRequest r;
        ApiError e;
        CHECK_OK(parse(tok, R"({"state": "s", "questions": {"q": {"type": "choice",
            "instructions": "Pick.", "criteria": {)" + crit + "}}}}", &r, &e));
        const std::string text = tok.decode(r.questions[0].tokens);
        CHECK(text.find("Options:\n00 o0\nA o1\nB o2\n") != std::string::npos);
        CHECK_EQ(r.questions[0].groups.size(), (size_t)27);
    }
}

TEST(the_readout_is_a_softmax_over_each_answers_best_spelling) {
    DecisionQuestion q;
    q.name = "n"; q.type = "noul"; q.labels = { "true", "false" };
    q.candidates = { 10, 11, 20, 21 };
    q.groups = { { 0, 1 }, { 2, 3 } };
    json a;
    ApiError e;
    CHECK_OK(decision_answer(q, { 1.0f, 3.0f, 2.0f, -5.0f }, &a, &e));
    CHECK_NEAR(a["noul"].get<double>(), 1.0 / (1.0 + std::exp(-1.0)), 1e-12);   /* 3 vs 2 */

    DecisionQuestion s;
    s.name = "s"; s.type = "score"; s.labels = { "0", "1", "2" }; s.legend = { "a", "b", "c" };
    s.candidates = { 1, 2, 3 };
    s.groups = { { 0 }, { 1 }, { 2 } };
    CHECK_OK(decision_answer(s, { 0.0f, 0.0f, std::log(2.0f) }, &a, &e));
    CHECK_NEAR(a["score"].get<double>(), (0.25 + 2 * 0.5), 1e-6);
    CHECK_NEAR(a["confidence"].get<double>(), 0.5, 1e-6);
    CHECK_EQ(a["legend"]["2"].get<std::string>(), "c");
    CHECK_EQ(a.dump().substr(0, 26), "{\"type\":\"score\",\"score\":1.");

    CHECK(decision_answer(s, { 0.0f, NAN, 1.0f }, &a, &e) < 0);
    CHECK_EQ(e.status, 500);
    CHECK(decision_answer(s, { 0.0f }, &a, &e) < 0);
}

TEST(requests_d1_cannot_answer_are_refused_by_field) {
    DictTokenizer tok;
    struct Case { const char* body; const char* param; };
    const Case cases[] = {
        { R"({"questions": {"a": {"type": "noul", "instructions": "x"}}})", "state" },
        { R"({"state": "s", "questions": {}})", "questions" },
        { R"({"state": "s", "questions": {"a": {"type": "noul", "instructions": "x"}}, "stream": true})",
          "stream" },
        { R"({"state": "s", "model": "other", "questions": {"a": {"type": "noul", "instructions": "x"}}})",
          "model" },
        { R"({"state": "s", "questions": {"a": {"type": "vote", "instructions": "x"}}})", "questions.a.type" },
        { R"({"state": "s", "questions": {"a": {"type": "score", "instructions": "x", "criteria": ["one"]}}})",
          "questions.a.criteria" },
        { R"({"state": "s", "questions": {"a": {"type": "choice", "instructions": "x", "criteria": {}}}})",
          "questions.a.criteria" },
        { R"({"state": "s", "questions": {"a": {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}}}})",
          "questions.a.criteria.maybe" },
        { R"({"state": "s", "questions": {"a": {"type": "noul"}}})", "questions.a.instructions" },
    };
    for (const Case& c : cases) {
        DecisionRequest r;
        ApiError e;
        CHECK(parse(tok, c.body, &r, &e) < 0);
        CHECK_EQ(e.param, c.param);
    }
    /* Pictures without an encoder are the deployment's refusal, not the request's. */
    DecisionRequest r;
    ApiError e;
    CHECK(parse(tok, R"({"state": null, "images": ["data:image/png;base64,AAAA"],
        "questions": {"a": {"type": "noul", "instructions": "x"}}})", &r, &e) < 0);
    CHECK_EQ(e.status, 501);
    /* A prompt must leave one position for the token the request samples. */
    CHECK(parse(tok, R"({"state": "s", "questions": {"a": {"type": "noul", "instructions": "x"}}})",
                &r, &e, 40) < 0);
    CHECK_EQ(e.code, "context_length_exceeded");
}

/* ------------------------------------------------------------------ the handler */
namespace {

/* Answers every scored request with logits that make the first spelling of the first answer win
 * by ln 3 (P = 0.75 with two answers), and records each request and each batch it was handed.
 * `fail_first` fails the first request with an engine error; `refuse` answers as SchedulerBridge
 * refuses (id 0, the sink failed). Reaps are checked as FakeScheduler checks them. */
struct ScoringScheduler : IScheduler {
    std::mutex mu;
    std::vector<Request> seen;
    std::vector<size_t> batches;
    std::vector<std::thread> workers;
    std::map<uint64_t, Sink*> sinks;
    std::vector<uint64_t> reaped;
    int errors = 0;
    bool fail_first = false, refuse = false;
    std::atomic<uint64_t> next{ 1 };
    ~ScoringScheduler() override { join(); }
    void join() { for (auto& t : workers) if (t.joinable()) t.join(); }
    void submit_all(std::vector<Request>& rs, std::vector<uint64_t>* ids) override {
        {
            std::lock_guard<std::mutex> lk(mu);
            batches.push_back(rs.size());
        }
        ids->clear();
        for (Request& r : rs) ids->push_back(submit(std::move(r)));
    }
    uint64_t submit(Request&& r) override {
        std::lock_guard<std::mutex> lk(mu);
        auto ts = r.token_scores;
        void* sink = r.sink;
        const bool fail = fail_first && seen.empty();
        seen.push_back(std::move(r));
        if (refuse) {
            sink_fail(sink, RAD_E_INVAL, "refused");
            return 0;
        }
        const uint64_t id = next++;
        sinks[id] = static_cast<Sink*>(sink);
        workers.emplace_back([ts, sink, fail] {
            sink_admitted(sink);
            if (fail) { sink_fail(sink, RAD_E_IO, "the step failed"); return; }
            if (ts) {
                ts->logits.assign(ts->tokens.size(), 0.0f);
                ts->logits[0] = std::log(3.0f);
            }
            sink_push(sink, 'x', nullptr);
            sink_finish(sink, Finish::Length);
        });
        return id;
    }
    void cancel(uint64_t) override {}
    void reap(uint64_t id) override {
        std::lock_guard<std::mutex> lk(mu);
        auto it = sinks.find(id);
        if (it == sinks.end() || !it->second->done() ||
            std::find(reaped.begin(), reaped.end(), id) != reaped.end())
            ++errors;
        reaped.push_back(id);
    }
    SchedMetrics metrics() const override { return {}; }
};

}  /* namespace */

TEST(the_endpoint_answers_only_for_a_declared_d1) {
    for (const char* type : { "", "d1" }) {
        ScoringScheduler sched;
        DictTokenizer tok;
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        ServerOptions o;
        o.model_id = "d1";
        o.decision_type = type;
        Server srv(d, o);
        HttpResponse res = srv.router().call("POST", "/v1/systemone",
            R"({"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"},
                                            "c": {"type": "choice", "instructions": "y",
                                                  "criteria": {"one": null, "two": null}}}})");
        if (!*type) { CHECK_EQ(res.status, 501); continue; }
        REQUIRE(res.status == 200);
        const json j = json::parse(res.body);
        CHECK_EQ(j["model"].get<std::string>(), "d1");
        CHECK_NEAR(j["answers"]["q"]["noul"].get<double>(), 0.75, 1e-6);
        CHECK_EQ(j["answers"]["c"]["choice"].get<std::string>(), "one");
        CHECK_EQ(j["usage"]["output_tokens"].get<int>(), 0);
        /* One request a question, scored, one token long, greedy. */
        std::lock_guard<std::mutex> lk(sched.mu);
        REQUIRE(sched.seen.size() == 2);
        for (const Request& rq : sched.seen) {
            CHECK(rq.token_scores != nullptr);
            CHECK_EQ(rq.max_tokens, 1);
            CHECK_EQ(rq.sp.temp, 0.0f);
        }
    }
}

namespace {

const char* const kThreeQuestions = R"({"state": "s", "questions": {
    "a": {"type": "noul", "instructions": "x"},
    "b": {"type": "noul", "instructions": "y"},
    "c": {"type": "noul", "instructions": "z"}}})";

ServerOptions d1_options() {
    ServerOptions o;
    o.model_id = "d1";
    o.decision_type = "d1";
    return o;
}

}  /* namespace */

/* The questions share the state they ask about, and it is computed once: the first question runs
 * alone with a checkpoint at the end of the shared head, so the head stays in the prefix cache,
 * and the rest follow as one batch that reads it. Submitting them together would answer the same
 * and compute the head once per question. A single question has no head to share. */
TEST(the_first_question_runs_alone_and_leaves_the_shared_head_cached) {
    DictTokenizer tok;
    DecisionRequest want;
    ApiError e;
    CHECK_OK(parse(tok, kThreeQuestions, &want, &e));
    REQUIRE(want.trunk > 0);

    ScoringScheduler sched;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    {
        Server srv(d, d1_options());
        CHECK_EQ(srv.router().call("POST", "/v1/systemone", kThreeQuestions).status, 200);
        CHECK_EQ(srv.in_flight(), (int64_t)0);
        CHECK_EQ(srv.router().call("POST", "/v1/systemone",
                                   R"({"state": "s", "questions": {"a": {"type": "noul", "instructions": "x"}}})")
                     .status, 200);
        sched.join();
    }
    std::lock_guard<std::mutex> lk(sched.mu);
    CHECK(sched.batches == std::vector<size_t>({ 1, 2, 1 }));
    REQUIRE(sched.seen.size() == 4);
    CHECK_EQ(sched.seen[0].checkpoint_at, (int64_t)want.trunk);
    CHECK_EQ(sched.seen[1].checkpoint_at, (int64_t)0);
    CHECK_EQ(sched.seen[2].checkpoint_at, (int64_t)0);
    CHECK_EQ(sched.seen[3].checkpoint_at, (int64_t)0);
    CHECK_EQ(sched.errors, 0);
    CHECK_EQ(sched.reaped.size(), (size_t)4);
}

/* A first question the engine fails, or the scheduler refuses, fails the call with a 500: the
 * rest would only repeat it, so nothing else is submitted, and the admission places come back.
 * What was submitted is reaped once; a refused question, never issued, is not. */
TEST(a_failed_first_question_fails_the_call_and_submits_nothing_more) {
    for (bool refuse : { false, true }) {
        DictTokenizer tok;
        ScoringScheduler sched;
        sched.fail_first = !refuse;
        sched.refuse = refuse;
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        {
            Server srv(d, d1_options());
            HttpResponse r = srv.router().call("POST", "/v1/systemone", kThreeQuestions);
            CHECK_EQ(r.status, 500);
            CHECK(has(r.body, refuse ? "refused question a" : "the step failed"));
            CHECK_EQ(srv.in_flight(), (int64_t)0);
            sched.join();
        }
        std::lock_guard<std::mutex> lk(sched.mu);
        CHECK(sched.batches == std::vector<size_t>({ 1 }));
        CHECK_EQ(sched.errors, 0);
        CHECK_EQ(sched.reaped.size(), (size_t)(refuse ? 0 : 1));
    }
}

RAD_TEST_MAIN()
