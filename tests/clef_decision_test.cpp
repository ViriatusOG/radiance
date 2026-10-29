/* clef_decision_test.cpp -- /v1/systemone on a clef container, end to end on the CPU.
 *
 * A tiny qwen35 (tests/tiny_model.h) with a tiny joint schema head of its own, converted with the
 * head added, `--arch clef` (the clef plugins) and `decision.type=clef`, served by the radiance
 * binary on the host backend. The
 * answers are checked against tests/data/golden/clef_tiny.json, the release's own head
 * (joint_schema_model.py) over transformers' forward pass in float32, made by
 * tests/mkclefgolden.py from the checkpoint this test writes (RAD_E2E_KEEP=1 keeps it). The
 * golden carries the weights' hashes, so a golden for other weights is refused by name.
 *
 * What one run covers: the prompt and spans from the request, the words to the plugin, the
 * head's cache group filled over two prefill chunks, the gather of every row back out of it, the
 * head's own ops on libref, the three values an option through the logits row, the scalars from
 * the container header; then the same request again, a whole prefix hit, so the head reads every
 * row from the cache alone; and the bounds and the image refusal.
 */
#include "decision_e2e.h"

#include "rad_clef.h"

#include <sys/stat.h>

#include <cmath>
#include <fstream>

using namespace decision_e2e;

namespace {

/* The tiny clef: tiny_model.h's qwen35 (hidden 64) and a head of width 64 with one head of 64
 * (libr4d's attention width, so the same declaration serves the card), feedforward 128, one
 * routing and one decoder layer; LayerNorm gains near one, every linear at unit gain, the three
 * scalars fixed. The release's tensor names, so rad-convert and the plugin read them as they
 * read clef-flash's. */
bool write_tiny_clef(const std::string& dir, uint64_t seed) {
    tiny::Qwen35 s;
    s.seed = seed;
    /* The head writes three stretches of RAD_CLEF_MAX_OPTIONS into a logits row, so the
     * vocabulary must hold them; the byte tokenizer uses the first 259 ids. */
    s.vocab = 2048;
    if (!tiny::write_qwen35(dir, s)) return false;
    const int64_t E = s.hidden, W = 64, F = 128;
    tiny::Rng r(seed + 500);
    std::vector<tiny::Tensor> ts;
    auto ln = [&](const std::string& p, int64_t n) {
        ts.push_back(tiny::noise(p + ".weight", { n }, r, 0.1f, 1.0f));
        ts.push_back(tiny::noise(p + ".bias", { n }, r, 0.1f));
    };
    auto lin = [&](const std::string& p, int64_t n_out, int64_t n_in, bool bias) {
        ts.push_back(tiny::linear(p + ".weight", n_out, n_in, r));
        if (bias) ts.push_back(tiny::noise(p + ".bias", { n_out }, r, 0.1f));
    };
    /* torch.nn.MultiheadAttention packs q, k and v as in_proj_weight and in_proj_bias. */
    auto in_proj = [&](const std::string& p) {
        ts.push_back(tiny::linear(p + ".in_proj_weight", 3 * W, W, r));
        ts.push_back(tiny::noise(p + ".in_proj_bias", { 3 * W }, r, 0.1f));
    };
    ln("hidden_norm", E);
    for (const char* p : { "memory_projection", "question_projection", "option_question_projection",
                           "global_projection", "option_context_projection",
                           "option_lexical_projection" })
        lin(p, W, E, false);
    ts.push_back(tiny::noise("type_embedding.weight", { 3, W }, r, 0.5f));
    for (int i = 0; i < 1; ++i) {
        const std::string p = "evidence_layers." + std::to_string(i) + ".";
        ln(p + "query_norm", W);
        ln(p + "memory_norm", W);
        in_proj(p + "attention");
        lin(p + "attention.out_proj", W, W, true);
        ln(p + "feedforward_norm", W);
        lin(p + "feedforward.0", F, W, true);
        lin(p + "feedforward.3", W, F, true);
    }
    ln("option_summary_norm", W);
    for (int i = 0; i < 1; ++i) {
        const std::string p = "layers." + std::to_string(i) + ".";
        in_proj(p + "self_attn");
        lin(p + "self_attn.out_proj", W, W, true);
        in_proj(p + "multihead_attn");
        lin(p + "multihead_attn.out_proj", W, W, true);
        lin(p + "linear1", F, W, true);
        lin(p + "linear2", W, F, true);
        ln(p + "norm1", W);
        ln(p + "norm2", W);
        ln(p + "norm3", W);
    }
    ln("field_norm", W);
    ln("option_norm", W);
    lin("residual_scorer.0", W, 4 * W, true);
    lin("residual_scorer.3", 1, W, true);
    ts.push_back(tiny::noise("prior_logit_scale", {}, r, 0.0f, 0.5f));
    ts.push_back(tiny::noise("joint_logit_scale", {}, r, 0.0f, 0.5f));
    ts.push_back(tiny::noise("residual_gate", {}, r, 0.0f, -1.0f));
    if (!tiny::write_safetensors(dir + "/joint_head.safetensors", ts)) return false;
    nlohmann::ordered_json c;
    c["hidden_size"] = E;
    c["width"] = W;
    c["routing_layers"] = 1;
    c["layers"] = 1;
    c["heads"] = 1;
    c["feedforward"] = F;
    return tiny::write_text(dir + "/joint_head_config.json", c.dump(2));
}

double prob(const json& answers, const char* q, const char* field, const char* option) {
    const json& a = answers.at(q);
    if (option) return a.at("probabilities").at(option).get<double>();
    return a.at(field).get<double>();
}

}  /* namespace */

TEST(a_clef_container_decides_every_question_on_the_card_and_matches_the_release) {
    const bool keep = std::getenv("RAD_E2E_KEEP") != nullptr;
    tiny::TempDir dir{ "clef", keep };
    REQUIRE(dir.ok());
    const std::string ck = dir / "clef-tiny";
    REQUIRE_EQ(mkdir(ck.c_str(), 0755), 0);
    REQUIRE(write_tiny_clef(ck, 7));
    if (keep) fprintf(stderr, "    the tiny clef is kept at %s (tests/mkclefgolden.py)\n", ck.c_str());

    /* Converted with the head added; the head's geometry and scalars land in the header. */
    const std::string rad = dir / "clef-tiny.rad";
    std::string log;
    const int st = tiny::run({ RAD_E2E_CONVERT, ck, "-o", rad, "--arch", "clef", "--add",
                               ck + "/joint_head.safetensors", "--set", "decision.type=clef" },
                             engine_env(), &log, 300);
    if (st != 0) fprintf(stderr, "    rad-convert exited %d:\n%s\n", st, tail(log).c_str());
    REQUIRE_EQ(st, 0);
    CHECK(log.find("clef     head 64 x 1") != std::string::npos);

    std::ifstream gf(RAD_CLEF_GOLDEN);
    if (!gf.is_open())
        fprintf(stderr, "    no golden at %s: run this test with RAD_E2E_KEEP=1, then "
                        "tests/mkclefgolden.py on the kept directory\n", RAD_CLEF_GOLDEN);
    REQUIRE(gf.is_open());
    const json gold = json::parse(gf);
    const std::string wf = hex16(fnv1a(ck + "/model.safetensors"));
    const std::string hf = hex16(fnv1a(ck + "/joint_head.safetensors"));
    if (gold["weights_fnv"] != wf || gold["head_fnv"] != hf)
        fprintf(stderr, "    the golden was made from other weights (%s / %s here): regenerate "
                        "it with tests/mkclefgolden.py\n", wf.c_str(), hf.c_str());
    REQUIRE(gold["weights_fnv"] == wf && gold["head_fnv"] == hf);

    /* A small step, so the prompt prefills in more than one chunk and the head's cache is
     * filled across them. */
    Server s;
    REQUIRE(s.start(rad, { "--max-num-seqs", "2", "--max-model-len", "2048",
                           "--max-num-batched-tokens", "512" }));
    int status = 0;
    const json models = call(s.port, "GET", "/v1/models", json(), &status);
    CHECK_EQ(status, 200);
    CHECK(models["data"][0]["capabilities"]["decision"] == "clef");

    const json a = call(s.port, "POST", "/v1/systemone", gold["request"], &status);
    if (status != 200) fprintf(stderr, "    %s\n%s\n", a.dump().c_str(), tail(s.output()).c_str());
    REQUIRE_EQ(status, 200);
    CHECK_EQ(a["usage"]["input_tokens"].get<int64_t>(), gold["n_prompt_tokens"].get<int64_t>());
    const json& ans = a["answers"];
    REQUIRE(ans.contains("refund") && ans.contains("team") && ans.contains("tone"));
    CHECK(ans["refund"]["type"] == "noul");
    CHECK(ans["team"]["type"] == "choice");
    CHECK(ans["tone"]["type"] == "score");
    double sum = 0.0;
    for (const auto& kv : ans["team"]["probabilities"].items()) sum += kv.value().get<double>();
    CHECK(std::fabs(sum - 1.0) < 2e-3);

    /* Against the release's head in float32 over the same tiny weights. bf16 through the trunk
     * and the head moves a probability by a few hundredths at most. */
    const json& g = gold["answers"];
    double worst = 0.0;
    auto diff = [&](double got, double want) { worst = std::max(worst, std::fabs(got - want)); };
    diff(prob(ans, "refund", "noul", nullptr), g["refund"]["true"].get<double>());
    for (const char* o : { "billing", "product", "shipping" })
        diff(prob(ans, "team", nullptr, o), g["team"][o].get<double>());
    for (const char* l : { "0", "1", "2" })
        diff(prob(ans, "tone", nullptr, l), g["tone"][l].get<double>());
    fprintf(stderr, "    largest probability difference from the release's head: %.4f\n", worst);
    CHECK(worst < 0.03);
    CHECK(ans["team"]["choice"] == [&] {
        std::string best;
        double bp = -1.0;
        for (const auto& kv : g["team"].items())
            if (kv.value().get<double>() > bp) { bp = kv.value().get<double>(); best = kv.key(); }
        return best;
    }());

    /* The same request again: the whole prompt is a prefix hit, so the trunk recomputes one
     * token and the head reads every row from its cache. The answer must not move. Then the
     * LiteLLM spelling of the endpoint. */
    const json b = call(s.port, "POST", "/v1/systemone", gold["request"], &status);
    CHECK_EQ(status, 200);
    CHECK(b["answers"] == ans);
    const json d = call(s.port, "POST", "/v1/decisions", gold["request"], &status);
    CHECK_EQ(status, 200);
    CHECK(d["answers"] == ans);

    /* The bounds and the refusals. */
    json many = gold["request"];
    many["questions"] = json::object();
    for (int i = 0; i <= RAD_CLEF_MAX_QUESTIONS; ++i)
        many["questions"]["q" + std::to_string(i)] = { { "type", "noul" } };
    call(s.port, "POST", "/v1/systemone", many, &status);
    CHECK_EQ(status, 400);
    json img = gold["request"];
    img["images"] = json::array({ "data:image/png;base64,AAAA" });
    call(s.port, "POST", "/v1/systemone", img, &status);
    CHECK_EQ(status, 501);
    json wrong = gold["request"];
    wrong["model"] = "other";
    call(s.port, "POST", "/v1/systemone", wrong, &status);
    CHECK_EQ(status, 400);

    CHECK_EQ(s.stop(), 0);
}

/* A container converted without the head's metadata is refused at convert, not served half. */
TEST(a_clef_convert_without_the_head_is_refused) {
    tiny::TempDir dir{ "clef" };
    REQUIRE(dir.ok());
    const std::string ck = dir / "clef-tiny";
    REQUIRE_EQ(mkdir(ck.c_str(), 0755), 0);
    REQUIRE(write_tiny_clef(ck, 7));
    std::string log;
    const int st = tiny::run({ RAD_E2E_CONVERT, ck, "-o", dir / "no-head.rad", "--arch", "clef",
                               "--set", "decision.type=clef" }, engine_env(), &log, 300);
    CHECK(st != 0);
    CHECK(log.find("joint_head.safetensors") != std::string::npos);
    /* And one that keeps the qwen35 id: the head would never run, so the convert refuses it. */
    log.clear();
    const int st2 = tiny::run({ RAD_E2E_CONVERT, ck, "-o", dir / "no-arch.rad", "--add",
                                ck + "/joint_head.safetensors", "--set", "decision.type=clef" },
                              engine_env(), &log, 300);
    CHECK(st2 != 0);
    CHECK(log.find("--arch clef") != std::string::npos);
}

RAD_TEST_MAIN()
