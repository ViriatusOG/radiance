/* laya_decision_test.cpp -- /v1/systemone on a Laya container, end to end on the CPU.
 *
 * A tiny Laya in the release's own layout (config.json saying model_type laya, encoder/config.json
 * a three-layer ModernBERT with one sliding layer pair, rl_agent_config.json, tokenizer/, and
 * model.safetensors with the release's tensor names), converted with `--set decision.type=laya`
 * and served by the radiance binary on the host backend. The answers and usage are checked against
 * tests/data/golden/laya_tiny.json, laya's own Agent.predict on the same checkpoint in float32,
 * made by tests/mklayagolden.py from the directory this test writes (RAD_E2E_KEEP=1 keeps it). The
 * golden carries the weights' hash, so a golden for other weights is refused by name.
 *
 * What one run covers: rad-convert reading the release's folders; each question's sequence, the
 * state cut and its report; one engine request a question on the embedding path; the plugin's
 * encoder (full and sliding layers), head and scorer on libref; the option logits through the
 * pooled row; the fitted temperatures (one clamped) and the package's confidence; then the same
 * request again, the capability report, and the refusals.
 */
#include "decision_e2e.h"

#include "rad_laya.h"

#include <sys/stat.h>

#include <cmath>
#include <fstream>

using namespace decision_e2e;

namespace {

/* The tiny Laya: hidden 64 in one head of 64 (libr4d's attention width, so the same declaration
 * serves the card), feed-forward 96, three encoder layers (full, sliding, sliding over
 * local_attention 8), two head layers; LayerNorm gains near one, every linear at unit gain but the
 * scorer's last, which spreads the option logits. The byte tokenizer's specials after its own three
 * are [CLS] 259, [SEP] 260, [MASK] 261, [PAD] 262, [UNK] 263; the vocabulary pads to 320. */
bool write_tiny_laya(const std::string& dir, uint64_t seed) {
    const int64_t E = 64, F = 96, L = 3, V = 320, HL = 2, A = 256;
    tiny::Rng r(seed);
    std::vector<tiny::Tensor> ts;
    auto gain = [&](const std::string& n, int64_t k) { ts.push_back(tiny::noise(n, { k }, r, 0.1f, 1.0f)); };
    auto bias = [&](const std::string& n, int64_t k) { ts.push_back(tiny::noise(n, { k }, r, 0.1f)); };
    auto lin = [&](const std::string& n, int64_t o, int64_t i, float g = 1.0f) {
        ts.push_back(tiny::linear(n, o, i, r, g));
    };
    ts.push_back(tiny::noise("encoder.embeddings.tok_embeddings.weight", { V, E }, r, 1.0f));
    gain("encoder.embeddings.norm.weight", E);
    for (int64_t l = 0; l < L; ++l) {
        const std::string p = "encoder.layers." + std::to_string(l) + ".";
        if (l > 0) gain(p + "attn_norm.weight", E);
        lin(p + "attn.Wqkv.weight", 3 * E, E);
        lin(p + "attn.Wo.weight", E, E);
        gain(p + "mlp_norm.weight", E);
        lin(p + "mlp.Wi.weight", 2 * F, E);
        lin(p + "mlp.Wo.weight", E, F);
    }
    gain("encoder.final_norm.weight", E);
    for (int64_t j = 0; j < HL; ++j) {
        const std::string p = "head.layers." + std::to_string(j) + ".";
        lin(p + "self_attn.in_proj_weight", 3 * E, E);
        bias(p + "self_attn.in_proj_bias", 3 * E);
        lin(p + "self_attn.out_proj.weight", E, E);
        bias(p + "self_attn.out_proj.bias", E);
        lin(p + "linear1.weight", 4 * E, E);
        bias(p + "linear1.bias", 4 * E);
        lin(p + "linear2.weight", E, 4 * E);
        bias(p + "linear2.bias", E);
        gain(p + "norm1.weight", E);
        bias(p + "norm1.bias", E);
        gain(p + "norm2.weight", E);
        bias(p + "norm2.bias", E);
    }
    ts.push_back(tiny::noise("type_emb.weight", { 3, E }, r, 0.5f));
    gain("scorer.0.weight", E);
    bias("scorer.0.bias", E);
    lin("scorer.1.weight", E, E);
    bias("scorer.1.bias", E);
    lin("scorer.3.weight", 1, E, 3.0f);
    bias("scorer.3.bias", 1);
    /* The escalation head and the stored temperatures: in the release's state dict (its loader
     * is strict), not declared by the plugin. */
    lin("act_head.0.weight", A, E + 4);
    bias("act_head.0.bias", A);
    lin("act_head.2.weight", 2, A);
    bias("act_head.2.bias", 2);
    ts.push_back(tiny::noise("temperature", { 3 }, r, 0.0f, 1.0f, "F32"));
    if (!tiny::write_safetensors(dir + "/model.safetensors", ts)) return false;

    nlohmann::ordered_json root;
    root["model_type"] = "laya";
    root["architectures"] = { "LayaTypedDecisions" };
    nlohmann::ordered_json enc;
    enc["architectures"] = { "ModernBertForMaskedLM" };
    enc["model_type"] = "modernbert";
    enc["hidden_size"] = E;
    enc["intermediate_size"] = F;
    enc["num_hidden_layers"] = L;
    enc["num_attention_heads"] = 1;
    enc["layer_types"] = { "full_attention", "sliding_attention", "sliding_attention" };
    enc["global_attn_every_n_layers"] = 3;
    enc["local_attention"] = 8;
    enc["rope_parameters"] = { { "full_attention", { { "rope_theta", 160000.0 }, { "rope_type", "default" } } },
                               { "sliding_attention", { { "rope_theta", 10000.0 }, { "rope_type", "default" } } } };
    enc["max_position_embeddings"] = 8192;
    enc["vocab_size"] = V;
    enc["norm_eps"] = 1e-5;
    enc["norm_bias"] = false;
    enc["attention_bias"] = false;
    enc["mlp_bias"] = false;
    enc["hidden_activation"] = "gelu";
    enc["embedding_dropout"] = 0.0;
    enc["attention_dropout"] = 0.0;
    enc["mlp_dropout"] = 0.0;
    enc["pad_token_id"] = 262;
    enc["cls_token_id"] = 259;
    enc["sep_token_id"] = 260;
    enc["bos_token_id"] = 259;
    enc["eos_token_id"] = 260;
    enc["tie_word_embeddings"] = true;
    nlohmann::ordered_json rl;
    rl["encoder"] = "answerdotai/ModernBERT-large";
    rl["head_layers"] = HL;
    rl["max_len"] = 128;
    rl["head_max_len"] = 48;
    rl["max_prefixes"] = 6;
    rl["act_costs"] = { { "escalate", 0.5 } };
    rl["amp_dtype"] = "bf16";
    rl["model_name"] = "laya-tiny";
    rl["temperature"] = { 1.3, 0.9, 1.7 };
    rl["temperature_by_options"] = { { "choice:3-5", 1.6 }, { "noul:2", 2.2 }, { "choice:11+", 0.1 } };
    nlohmann::ordered_json tc;
    tc["cls_token"] = "[CLS]";
    tc["mask_token"] = "[MASK]";
    tc["pad_token"] = "[PAD]";
    tc["sep_token"] = "[SEP]";
    tc["unk_token"] = "[UNK]";
    tc["model_max_length"] = 8192;
    tc["tokenizer_class"] = "PreTrainedTokenizerFast";
    return mkdir((dir + "/encoder").c_str(), 0755) == 0 &&
           mkdir((dir + "/tokenizer").c_str(), 0755) == 0 &&
           tiny::write_text(dir + "/config.json", root.dump(2)) &&
           tiny::write_text(dir + "/encoder/config.json", enc.dump(2)) &&
           tiny::write_text(dir + "/rl_agent_config.json", rl.dump(2)) &&
           tiny::write_text(dir + "/tokenizer/tokenizer_config.json", tc.dump(2)) &&
           tiny::write_text(dir + "/tokenizer/tokenizer.json",
                            tiny::tokenizer_json({ "[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]" }));
}

/* The largest difference between radiance's answers and the package's: every probability, every
 * confidence, every noul and score; and whether each choice agrees where the package's top two are
 * apart. */
double compare(const json& got, const json& want, bool* choices_agree) {
    double worst = 0.0;
    auto d = [&](const json& a, const json& b) {
        worst = std::max(worst, std::fabs(a.get<double>() - b.get<double>()));
    };
    for (const auto& kv : want.items()) {
        const json& w = kv.value();
        if (!got.contains(kv.key())) { *choices_agree = false; return 1.0; }
        const json& g = got[kv.key()];
        if (g["type"] != w["type"]) { *choices_agree = false; return 1.0; }
        if (w["type"] == "noul") { d(g["noul"], w["noul"]); continue; }
        d(g["confidence"], w["confidence"]);
        if (w["type"] == "score") d(g["score"], w["score"]);
        std::vector<double> top;
        for (const auto& p : w["probabilities"].items()) {
            if (!g["probabilities"].contains(p.key())) { *choices_agree = false; return 1.0; }
            d(g["probabilities"][p.key()], p.value());
            top.push_back(p.value().get<double>());
        }
        if (w["type"] == "choice") {
            std::sort(top.rbegin(), top.rend());
            if (top.size() < 2 || top[0] - top[1] > 0.02) *choices_agree = *choices_agree && g["choice"] == w["choice"];
            CHECK(!g.contains("legend"));
        } else {
            CHECK(g["legend"] == w["legend"]);
        }
    }
    return worst;
}

}  /* namespace */

TEST(a_laya_container_answers_every_question_and_matches_the_package) {
    const bool keep = std::getenv("RAD_E2E_KEEP") != nullptr;
    tiny::TempDir dir{ "laya", keep };
    REQUIRE(dir.ok());
    const std::string ck = dir / "laya-tiny";
    REQUIRE_EQ(mkdir(ck.c_str(), 0755), 0);
    REQUIRE(write_tiny_laya(ck, 11));
    if (keep) fprintf(stderr, "    the tiny Laya is kept at %s (tests/mklayagolden.py)\n", ck.c_str());

    const std::string rad = dir / "laya-tiny.rad";
    std::string log;
    const int st = tiny::run({ RAD_E2E_CONVERT, ck, "-o", rad, "--set", "decision.type=laya" },
                             engine_env(), &log, 300);
    if (st != 0) fprintf(stderr, "    rad-convert exited %d:\n%s\n", st, tail(log).c_str());
    REQUIRE_EQ(st, 0);
    CHECK(log.find("encoder 3 x 64, head 2 layers, max_len 128, head_max_len 48") != std::string::npos);

    std::ifstream gf(RAD_LAYA_GOLDEN);
    if (!gf.is_open())
        fprintf(stderr, "    no golden at %s: run this test with RAD_E2E_KEEP=1, then "
                        "tests/mklayagolden.py on the kept directory\n", RAD_LAYA_GOLDEN);
    REQUIRE(gf.is_open());
    const json gold = json::parse(gf);
    const std::string wf = hex16(fnv1a(ck + "/model.safetensors"));
    if (gold["weights_fnv"] != wf)
        fprintf(stderr, "    the golden was made from other weights (%s here): regenerate it with "
                        "tests/mklayagolden.py\n", wf.c_str());
    REQUIRE(gold["weights_fnv"] == wf);

    Server s;
    REQUIRE(s.start(rad, { "--max-num-seqs", "4", "--max-num-batched-tokens", "512" }));
    int status = 0;
    const json models = call(s.port, "GET", "/v1/models", json(), &status);
    CHECK_EQ(status, 200);
    CHECK(models["data"][0]["capabilities"]["decision"] == "laya");
    CHECK(models["data"][0]["capabilities"]["embedding"] == false);

    double worst = 0.0;
    for (size_t i = 0; i < gold["requests"].size(); ++i) {
        const json& req = gold["requests"][i];
        const json& want = gold["results"][i];
        const json a = call(s.port, "POST", "/v1/systemone", req, &status);
        if (status != 200) fprintf(stderr, "    %s\n%s\n", a.dump().c_str(), tail(s.output()).c_str());
        REQUIRE_EQ(status, 200);
        /* The sequences and the cut are the package's to the token. */
        for (const char* k : { "input_tokens", "output_tokens", "state_tokens", "state_tokens_dropped",
                               "truncated", "truncated_questions" }) {
            if (a["usage"][k] != want["usage"][k])
                fprintf(stderr, "    request %zu: usage.%s %s, the package's %s\n", i, k,
                        a["usage"][k].dump().c_str(), want["usage"][k].dump().c_str());
            CHECK(a["usage"][k] == want["usage"][k]);
        }
        bool agree = true;
        const double w = compare(a["answers"], want["answers"], &agree);
        fprintf(stderr, "    request %zu: largest difference from the package %.4f\n", i, w);
        CHECK(agree);
        worst = std::max(worst, w);
        /* The same request again: no cache, the same step, the same answer. */
        const json b = call(s.port, "POST", "/v1/decisions", req, &status);
        CHECK_EQ(status, 200);
        CHECK(b["answers"] == a["answers"]);
    }
    CHECK(worst < 0.03);

    /* The refusals. */
    json base = gold["requests"][0];
    json many = base;
    many["questions"] = { { "q", { { "type", "choice" }, { "instructions", "x" }, { "criteria", json::array() } } } };
    for (int i = 0; i <= RAD_LAYA_MAX_OPTIONS; ++i) many["questions"]["q"]["criteria"].push_back("o" + std::to_string(i));
    call(s.port, "POST", "/v1/systemone", many, &status);
    CHECK_EQ(status, 400);
    json empty = base;
    empty["state"] = "";
    call(s.port, "POST", "/v1/systemone", empty, &status);
    CHECK_EQ(status, 400);
    json img = base;
    img["images"] = json::array({ "data:image/png;base64,AAAA" });
    call(s.port, "POST", "/v1/systemone", img, &status);
    CHECK_EQ(status, 501);
    call(s.port, "POST", "/v1/embeddings", json{ { "input", "hello" } }, &status);
    CHECK_EQ(status, 501);

    CHECK_EQ(s.stop(), 0);
}

/* The checkpoint names its own type, so --set is optional; a checkpoint that is not Laya's is
 * refused by name. */
TEST(a_laya_checkpoint_converts_without_flags_and_another_is_refused) {
    tiny::TempDir dir{ "laya" };
    REQUIRE(dir.ok());
    const std::string ck = dir / "laya-tiny";
    REQUIRE_EQ(mkdir(ck.c_str(), 0755), 0);
    REQUIRE(write_tiny_laya(ck, 11));
    std::string log;
    CHECK_EQ(tiny::run({ RAD_E2E_CONVERT, ck, "-o", dir / "plain.rad" }, engine_env(), &log, 300), 0);
    CHECK(log.find("laya     ") != std::string::npos);

    const std::string other = dir / "qwen";
    REQUIRE_EQ(mkdir(other.c_str(), 0755), 0);
    REQUIRE(tiny::write_qwen35(other, tiny::Qwen35{}));
    log.clear();
    CHECK(tiny::run({ RAD_E2E_CONVERT, other, "-o", dir / "no.rad", "--set", "decision.type=laya" },
                    engine_env(), &log, 300) != 0);
    CHECK(log.find("model_type laya") != std::string::npos);
}

RAD_TEST_MAIN()
