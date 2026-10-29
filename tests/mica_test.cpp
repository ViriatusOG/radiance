/* mica_test.cpp -- Mica's rows, prompts and answers (core/server/mica.h), against the contract
 * fixtures Mica's own repository ships (tests/data/golden/mica_contract_cases.json, from
 * github.com/akivet/Mica's tests/fixtures/contract_cases.json; its `source` names the commit). Each
 * case holds a request, the prompts Mica's native.py renders for it, a distribution a row and the
 * response Mica's contract.py packages from them; each invalid case a request Mica refuses. So
 * every prompt is checked byte for byte and every response value for value, with no tokenizer and
 * no weights.
 */
#include "rad_test.h"
#include "server/mica.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>

using namespace rad;
using namespace rad::server;

namespace {

const json& fixtures() {
    static const json f = [] {
        std::ifstream in(std::string(RAD_GOLDEN_FIXTURES) + "/mica_contract_cases.json");
        return json::parse(in);
    }();
    return f;
}

int parse(const json& request, MicaRequest* r, ApiError* e) {
    OaiLimits l;
    l.model_id = "mica";
    return parse_mica_request(request.dump(), l, "mica", r, e);
}

}  /* namespace */

TEST(every_fixture_prompt_is_micas_byte_for_byte) {
    const json& f = fixtures();
    REQUIRE(f["cases"].size() >= 40);
    for (const json& c : f["cases"]) {
        MicaRequest r;
        ApiError e;
        const int rc = parse(c["request"], &r, &e);
        if (rc != RAD_OK) fprintf(stderr, "    %s: refused: %s\n", c["name"].get<std::string>().c_str(), e.message.c_str());
        CHECK_EQ(rc, RAD_OK);
        if (rc != RAD_OK) continue;
        CHECK_EQ(r.rows.size(), c["prompts"].size());
        for (size_t i = 0; i < r.rows.size() && i < c["prompts"].size(); ++i) {
            const std::string got = mica_prompt(r.rows[i], r.state);
            if (got != c["prompts"][i].get<std::string>())
                fprintf(stderr, "    %s row %zu differs:\n%s\n", c["name"].get<std::string>().c_str(), i,
                        got.substr(0, 600).c_str());
            CHECK(got == c["prompts"][i].get<std::string>());
            CHECK_EQ(r.rows[i].id, c["rows"][i]["id"].get<std::string>());
        }
    }
}

TEST(every_fixture_response_is_micas_from_the_same_probabilities) {
    for (const json& c : fixtures()["cases"]) {
        MicaRequest r;
        ApiError e;
        REQUIRE_EQ(parse(c["request"], &r, &e), RAD_OK);
        std::vector<std::vector<double>> dist;
        for (const json& p : c["probabilities"]) dist.push_back(p.get<std::vector<double>>());
        json answers;
        CHECK_EQ(mica_answers(r, dist, &answers, &e), RAD_OK);
        const json& want = c["response"]["answers"];
        if (answers != want)
            fprintf(stderr, "    %s:\n      got  %s\n      want %s\n", c["name"].get<std::string>().c_str(),
                    answers.dump().substr(0, 400).c_str(), want.dump().substr(0, 400).c_str());
        CHECK(answers == want);
    }
}

TEST(every_request_mica_refuses_is_a_400) {
    for (const json& c : fixtures()["invalid"]) {
        MicaRequest r;
        ApiError e;
        const int rc = parse(c["request"], &r, &e);
        if (rc == RAD_OK) fprintf(stderr, "    %s was accepted\n", c["name"].get<std::string>().c_str());
        CHECK(rc != RAD_OK);
        CHECK_EQ(e.status, 400);
    }
}

TEST(the_labels_are_micas_codebook) {
    MicaRow noul;
    noul.kind = "noul";
    CHECK(mica_labels(noul) == std::vector<std::string>({ "No", "Yes" }));
    MicaRow choice;
    choice.kind = "choice";
    choice.candidates.resize(255);
    const std::vector<std::string> l = mica_labels(choice);
    CHECK_EQ(l.size(), (size_t)255);
    CHECK_EQ(l[0], std::string("A"));
    CHECK_EQ(l[51], std::string("z"));
    CHECK_EQ(l[52], std::string("AA"));
    CHECK_EQ(l[254], std::string("IM"));
}

TEST(the_distribution_is_micas_calibration) {
    MicaRow noul;
    noul.kind = "noul";
    MicaSettings s;
    s.temperature = 2.0;
    s.noul_logit_bias = 0.5;
    const std::vector<double> p = mica_distribution(noul, { 1.0f, 2.0f }, s);
    const double z0 = 0.5, z1 = 1.0 + 0.5, m = std::max(z0, z1);
    const double w0 = std::exp(z0 - m), w1 = std::exp(z1 - m);
    CHECK(std::fabs(p[0] - w0 / (w0 + w1)) < 1e-15);
    CHECK(std::fabs(p[1] - w1 / (w0 + w1)) < 1e-15);
    MicaRow choice;
    choice.kind = "choice";
    const std::vector<double> q = mica_distribution(choice, { 1.0f, 2.0f, 3.0f }, s);   /* no bias */
    CHECK(std::fabs(q[2] / q[1] - std::exp(0.5)) < 1e-12);
}

RAD_TEST_MAIN()
