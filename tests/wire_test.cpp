/* wire_test.cpp -- the pieces of the HTTP and OpenAI surface that are not a route.
 *
 * server_test drives handlers end to end. This file covers the small functions underneath that
 * it cannot reach directly: the target splitter, percent decoder, base64 pair, the UTF-8 back-off
 * that keeps a streamed delta from splitting a code point, the bucket ladders Prometheus
 * quantiles are computed against, and the body builders. Each is reachable from one route, so a
 * fault shows up as one endpoint being subtly wrong (a query parameter that does not arrive, an
 * undecodable stream, histogram quantiles that move between versions).
 *
 * Each has a non-obvious case: base64 has two padding lengths, the decoder must reject a bad
 * character rather than skip it, `+` means different things in the two halves of a target, and
 * the easy implementation of `utf8_trunc` is wrong.
 */
#include "rad_test.h"

#include "util/json.h"

#include "server/http.h"
#include "server/json_partial.h"
#include "server/metrics.h"
#include "server/oai.h"
#include "server/sink.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

using namespace rad;
using namespace rad::server;

/* ================================================================== the target */

TEST(a_target_splits_into_a_path_and_decoded_query_parameters) {
    std::string path;
    std::map<std::string, std::string> q;
    parse_target("/v1/models/foo?x=1&y=two", &path, &q);
    CHECK_EQ(path, std::string("/v1/models/foo"));
    REQUIRE_EQ(q.size(), 2u);
    CHECK_EQ(q["x"], std::string("1"));
    CHECK_EQ(q["y"], std::string("two"));

    /* No query at all leaves the map untouched rather than inventing an empty entry. */
    std::map<std::string, std::string> none;
    parse_target("/health", &path, &none);
    CHECK_EQ(path, std::string("/health"));
    CHECK_EQ(none.size(), 0u);

    /* A key with no value is present and empty: "?stream" is a flag, and a caller must be able
     * to tell it from a key that was never sent. */
    std::map<std::string, std::string> flag;
    parse_target("/x?stream", &path, &flag);
    REQUIRE_EQ(flag.size(), 1u);
    CHECK(flag.count("stream") == 1);
    CHECK_EQ(flag["stream"], std::string(""));

    /* An empty query after the '?' contributes nothing. */
    std::map<std::string, std::string> empty;
    parse_target("/x?", &path, &empty);
    CHECK_EQ(path, std::string("/x"));
    CHECK_EQ(empty.size(), 0u);

    /* Both outputs are optional: the router asks for the path alone on the hot path. */
    std::string only;
    parse_target("/x?a=b", &only, nullptr);
    CHECK_EQ(only, std::string("/x"));
    std::map<std::string, std::string> qonly;
    parse_target("/x?a=b", nullptr, &qonly);
    CHECK_EQ(qonly["a"], std::string("b"));
}

TEST(query_values_are_percent_decoded_and_survive_an_embedded_equals) {
    std::string path;
    std::map<std::string, std::string> q;
    parse_target("/x?name=a%20b&eq=1%3D2&plus=a+b", &path, &q);
    CHECK_EQ(q["name"], std::string("a b"));
    /* Split on the first '=' only, or a base64 value (which ends in '=' padding) loses its
     * tail. */
    CHECK_EQ(q["eq"], std::string("1=2"));
    CHECK_EQ(q["plus"], std::string("a b"));

    std::map<std::string, std::string> raw;
    parse_target("/x?v=a=b=c", &path, &raw);
    CHECK_EQ(raw["v"], std::string("a=b=c"));
}

/* `+` is a space in a query string and a plus sign in a path. Form encoding applies to the query
 * string, not the path, so /v1/models/gpt+4 must not become a request for "gpt 4". */
TEST(a_plus_in_the_path_stays_a_plus) {
    std::string path;
    std::map<std::string, std::string> q;
    parse_target("/v1/models/gpt+4?q=a+b", &path, &q);
    CHECK_EQ(path, std::string("/v1/models/gpt+4"));
    CHECK_EQ(q["q"], std::string("a b"));

    CHECK_EQ(url_decode("a+b"), std::string("a b"));
    CHECK_EQ(url_decode("a+b", false), std::string("a+b"));
    /* Percent-encoding works the same in both. */
    CHECK_EQ(url_decode("a%2Bb", false), std::string("a+b"));
}

TEST(url_decode_leaves_a_truncated_or_invalid_escape_alone) {
    CHECK_EQ(url_decode("plain"), std::string("plain"));
    CHECK_EQ(url_decode(""), std::string(""));
    CHECK_EQ(url_decode("%41%42"), std::string("AB"));
    CHECK_EQ(url_decode("%4a"), std::string("J"));        /* lower-case hex */
    CHECK_EQ(url_decode("%4A"), std::string("J"));
    /* A bad escape is left verbatim; consuming it would eat the bytes after it and silently
     * shorten the value. */
    CHECK_EQ(url_decode("%zz"), std::string("%zz"));
    CHECK_EQ(url_decode("%4"), std::string("%4"));
    CHECK_EQ(url_decode("%"), std::string("%"));
    CHECK_EQ(url_decode("100%"), std::string("100%"));
    /* A percent-encoded percent. */
    CHECK_EQ(url_decode("%2541"), std::string("%41"));
}

/* ================================================================== base64 */

TEST(base64_round_trips_at_every_padding_length) {
    /* The three residues are all of the encoder's branching and what the padding length encodes;
     * a decoder that got the two-byte tail wrong would return a trailing zero. */
    const char* cases[] = { "", "f", "fo", "foo", "foob", "fooba", "foobar" };
    const char* want[]  = { "", "Zg==", "Zm8=", "Zm9v", "Zm9vYg==", "Zm9vYmE=", "Zm9vYmFy" };
    for (int i = 0; i < 7; ++i) {
        const std::string in = cases[i];
        const std::string enc = base64_encode(in.data(), in.size());
        CHECK_EQ(enc, std::string(want[i]));
        std::vector<uint8_t> back;
        CHECK(base64_decode(enc, &back));
        CHECK_EQ(back.size(), in.size());
        CHECK_EQ(std::string(back.begin(), back.end()), in);
    }
}

TEST(base64_carries_every_byte_value_including_zero) {
    std::vector<uint8_t> all(256);
    for (int i = 0; i < 256; ++i) all[(size_t)i] = (uint8_t)i;
    const std::string enc = base64_encode(all.data(), all.size());
    /* 256 bytes is not a multiple of three, so this exercises a padded tail on real data. */
    CHECK_EQ(enc.size(), (size_t)344);
    std::vector<uint8_t> back;
    CHECK(base64_decode(enc, &back));
    REQUIRE_EQ(back.size(), (size_t)256);
    for (int i = 0; i < 256; ++i) CHECK_EQ((int)back[(size_t)i], i);

    /* The standard alphabet, not the URL-safe one: a decoder chooses by the characters it sees,
     * and '+' and '/' are what it must see. */
    bool plus = false, slash = false;
    for (char c : enc) { if (c == '+') plus = true; if (c == '/') slash = true; }
    CHECK(plus);
    CHECK(slash);
}

/* Embeddings arrive as base64 from clients, so the decoder parses untrusted input: a character
 * outside the alphabet must be an error, not skipped, or a corrupted payload decodes to a shorter
 * vector that the shape check then judges for the wrong reason. */
TEST(base64_decode_rejects_a_character_outside_the_alphabet) {
    std::vector<uint8_t> out;
    CHECK(!base64_decode("Zm9v!", &out));
    CHECK(!base64_decode("****", &out));
    CHECK(!base64_decode("Zm-9v", &out));        /* the URL-safe alphabet is not accepted */
    CHECK(!base64_decode("Zm_9v", &out));

    /* Whitespace and padding are skipped, since a wrapped MIME payload is still the value. */
    CHECK(base64_decode("Zm9v\nYmFy", &out));
    CHECK_EQ(std::string(out.begin(), out.end()), std::string("foobar"));
    CHECK(base64_decode("Zm9v YmFy\t\r\n", &out));
    CHECK_EQ(std::string(out.begin(), out.end()), std::string("foobar"));
    CHECK(base64_decode("", &out));
    CHECK_EQ(out.size(), 0u);

    /* The output is cleared first, so a failed decode never leaves a previous payload behind. */
    CHECK(base64_decode("Zm9v", &out));
    CHECK_EQ(out.size(), 3u);
    CHECK(!base64_decode("!", &out));
    CHECK_EQ(out.size(), 0u);
}

/* Every length mod 3 round-trips; padding and whitespace are skipped wherever they stand, even
 * inside a four-digit block; anything else is refused. */
TEST(base64_round_trips_and_refuses_stray_bytes) {
    std::vector<uint8_t> bytes(1000);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (uint8_t)((i * 2654435761u) >> 13);
    for (size_t n : { 0, 1, 2, 3, 4, 5, 999, 1000 }) {
        std::vector<uint8_t> got;
        CHECK(base64_decode(base64_encode(bytes.data(), n), &got));
        CHECK(got == std::vector<uint8_t>(bytes.begin(), bytes.begin() + (long)n));
    }
    std::string e = base64_encode(bytes.data(), 100), spaced;
    for (size_t i = 0; i < e.size(); ++i) { spaced += e[i]; if (i % 7 == 3) spaced += (i % 2 ? "\n" : " "); }
    std::vector<uint8_t> got;
    CHECK(base64_decode(spaced, &got));
    CHECK(got == std::vector<uint8_t>(bytes.begin(), bytes.begin() + 100));
    CHECK(!base64_decode("QUJD*REVG", &got));
    CHECK(!base64_decode("QUJDREVG\x80", &got));
}

/* ================================================================== ids and errors */

TEST(a_random_id_carries_its_prefix_and_does_not_repeat) {
    const std::string a = random_id("chatcmpl-");
    const std::string b = random_id("chatcmpl-");
    CHECK(a.rfind("chatcmpl-", 0) == 0);
    CHECK(b.rfind("chatcmpl-", 0) == 0);
    CHECK(a.size() > std::string("chatcmpl-").size());
    CHECK(a != b);                 /* ids collide in a client's logs, not in the server */
    CHECK_EQ(a.size(), b.size());  /* fixed width, so it sorts and pads predictably */

    const std::string c = random_id("cmpl-");
    CHECK(c.rfind("cmpl-", 0) == 0);
    /* The body is printable: an id lands in URLs and log lines. */
    for (size_t i = std::string("cmpl-").size(); i < c.size(); ++i)
        CHECK(c[i] > 0x20 && c[i] < 0x7f);
}

TEST(the_error_envelope_is_the_shape_a_client_parses) {
    const ApiError bad = err_bad("temperature", "must be in [0, 2]");
    CHECK_EQ(bad.status, 400);
    CHECK_EQ(bad.param, std::string("temperature"));
    const std::string d = bad.dump();
    /* OpenAI's envelope is {"error":{"message":...}} and clients read message and param by name;
     * anything flatter is reported by client libraries as "unknown". */
    CHECK(d.find("\"error\"") != std::string::npos);
    CHECK(d.find("\"message\"") != std::string::npos);
    CHECK(d.find("temperature") != std::string::npos);
    CHECK(d.find("must be in") != std::string::npos);
    CHECK(json_complete(d));

    const ApiError uns = err_unsupported("logprobs", "not implemented");
    CHECK_EQ(uns.status, 400);
    CHECK_EQ(uns.param, std::string("logprobs"));
    CHECK(json_complete(uns.dump()));

    /* A capability this build lacks is 501, not 400: 400 tells the caller to fix a request that
     * has nothing to fix. */
    const ApiError ni = err_not_implemented("embeddings");
    CHECK_EQ(ni.status, 501);
    CHECK(ni.dump().find("embeddings") != std::string::npos);

    /* A full queue is 429, which carries Retry-After: the request is fine and the server is
     * momentarily busy, so the client should back off and resend. 503 would say the service is
     * down, which a client may treat as fatal. */
    const ApiError ov = err_overloaded("admission queue is full");
    CHECK_EQ(ov.status, 429);
    CHECK(json_complete(ov.dump()));

    /* An error about no particular field carries no param rather than an empty one. */
    CHECK(err_overloaded("busy").param.empty());
}

/* ================================================================== server-sent events */

TEST(an_sse_event_is_one_data_line_and_a_blank_line) {
    const std::string e = sse_event("{\"a\":1}");
    CHECK_EQ(e, std::string("data: {\"a\":1}\n\n"));
    /* The blank line is the frame delimiter; without it a client buffers the event forever and
     * the stream looks hung. */
    CHECK(e.size() >= 2 && e.compare(e.size() - 2, 2, "\n\n") == 0);

    CHECK_EQ(std::string(SSE_DONE), std::string("data: [DONE]\n\n"));

    const std::string err = sse_error(err_bad("n", "too large"));
    CHECK(err.rfind("data: ", 0) == 0);
    CHECK(err.find("\"error\"") != std::string::npos);
    CHECK(err.find("too large") != std::string::npos);
    CHECK(err.compare(err.size() - 2, 2, "\n\n") == 0);
    /* Between the prefix and the delimiter is one complete JSON value, as a client's per-event
     * parse requires. */
    CHECK(json_complete(err.substr(6, err.size() - 8)));
}

/* ================================================================== finish reasons */

TEST(the_finish_reason_matches_vllms_spelling) {
    CHECK_EQ(std::string(finish_reason_oai(Finish::Stop)), std::string("stop"));
    CHECK_EQ(std::string(finish_reason_oai(Finish::Length)), std::string("length"));
    CHECK_EQ(std::string(finish_reason_oai(Finish::ToolCalls)), std::string("tool_calls"));
    /* Cancelled has no OpenAI spelling; vLLM reports "abort" and dashboards key off it. */
    CHECK_EQ(std::string(finish_reason_oai(Finish::Cancelled)), std::string("abort"));
    /* Every reason is non-empty: a missing finish_reason is null in the choice object, which
     * clients read as "still streaming". */
    for (Finish f : { Finish::Stop, Finish::Length, Finish::ToolCalls, Finish::Cancelled,
                      Finish::Error })
        CHECK(finish_reason_oai(f) && *finish_reason_oai(f));
}

/* ================================================================== the UTF-8 back-off */

TEST(utf8_trunc_backs_out_of_a_split_code_point_and_nothing_else) {
    /* Complete input is returned whole: the common case, taken on every delta. */
    CHECK_EQ(utf8_trunc("hello"), (size_t)5);
    CHECK_EQ(utf8_trunc(""), (size_t)0);
    CHECK_EQ(utf8_trunc("\xC3\xA9"), (size_t)2);              /* e-acute, complete */
    CHECK_EQ(utf8_trunc("\xE2\x82\xAC"), (size_t)3);          /* euro sign, complete */
    CHECK_EQ(utf8_trunc("\xF0\x9F\x98\x80"), (size_t)4);      /* an emoji, complete */

    /* A trailing partial sequence is cut. A delta ending mid-code-point is not decodable, and a
     * client that concatenates deltas before decoding would show a permanent replacement
     * character. */
    CHECK_EQ(utf8_trunc("ab\xC3"), (size_t)2);
    CHECK_EQ(utf8_trunc("ab\xE2\x82"), (size_t)2);
    CHECK_EQ(utf8_trunc("ab\xE2"), (size_t)2);
    CHECK_EQ(utf8_trunc("ab\xF0\x9F\x98"), (size_t)2);
    CHECK_EQ(utf8_trunc("\xF0"), (size_t)0);

    /* A complete sequence followed by a partial one keeps the complete one. */
    CHECK_EQ(utf8_trunc("\xE2\x82\xAC\xC3"), (size_t)3);
    /* A partial sequence in the middle is left alone: this backs out of a truncation, it does not
     * validate, so only the tail is reconsidered. */
    CHECK_EQ(utf8_trunc("\xC3 ok"), (size_t)4);
    CHECK_EQ(utf8_trunc("\xC3 ok\xE2\x82"), (size_t)4);   /* ... and the real tail still cuts */
}

/* ================================================================== histogram ladders */

/* vLLM's layouts, verbatim. Dashboards compute quantiles by interpolating between bucket bounds,
 * so changing a bound silently shifts every historical quantile across a version boundary. */
TEST(the_bucket_ladders_are_sorted_positive_and_the_ones_vllm_publishes) {
    struct { const char* name; std::vector<double> b; size_t n; double first, last; } cases[] = {
        { "ttft",    buckets_ttft(),    22, 0.001, 2560.0 },
        { "tpot",    buckets_tpot(),    19, 0.01,  80.0 },
        { "latency", buckets_latency(), 21, 0.3,   7680.0 },
    };
    for (auto& c : cases) {
        REQUIRE_EQ(c.b.size(), c.n);
        CHECK_NEAR(c.b.front(), c.first, 1e-12);
        CHECK_NEAR(c.b.back(), c.last, 1e-9);
        for (size_t i = 0; i < c.b.size(); ++i) {
            CHECK(c.b[i] > 0.0);
            /* Strictly increasing: a repeated bound gives a zero-width bucket, which quantile
             * interpolation divides by. */
            if (i) CHECK(c.b[i] > c.b[i - 1]);
        }
    }
}

TEST(the_token_ladder_is_mantissas_1_2_5_stopping_past_the_context) {
    /* vLLM's build_1_2_5_buckets: a 32K-context model gets 1..20000 and no more. */
    const std::vector<double> b = buckets_1_2_5(32768);
    REQUIRE(!b.empty());
    CHECK_NEAR(b.front(), 1.0, 1e-12);
    CHECK_NEAR(b.back(), 20000.0, 1e-9);
    REQUIRE_EQ(b.size(), 14u);
    for (size_t i = 1; i < b.size(); ++i) CHECK(b[i] > b[i - 1]);
    /* Every entry is a mantissa of 1, 2 or 5 at some decade. */
    for (double v : b) {
        double m = v;
        while (m >= 10.0) m /= 10.0;
        CHECK(std::fabs(m - 1.0) < 1e-9 || std::fabs(m - 2.0) < 1e-9 || std::fabs(m - 5.0) < 1e-9);
    }
    /* It stops before the first value past the bound: nothing above the context can be
     * observed, so a bucket there would never be reached. */
    for (double v : b) CHECK(v <= 32768.0);

    const std::vector<double> big = buckets_1_2_5(200000);
    REQUIRE(big.size() > b.size());
    CHECK_NEAR(big.back(), 200000.0, 1e-9);        /* the bound itself is kept when it is on the
                                                    * ladder, since it can be observed */

    /* A degenerate bound still gives a usable histogram: zero or negative is clamped to one, not
     * an empty ladder that would put every observation in the overflow bucket. */
    CHECK_EQ(buckets_1_2_5(1).size(), 1u);
    CHECK_EQ(buckets_1_2_5(0).size(), 1u);
    CHECK_EQ(buckets_1_2_5(-5).size(), 1u);
}

namespace {

/* Every histogram in a Prometheus body, checked as a scraper reads it: within a family the
 * buckets never decrease, and the +Inf bucket equals `_count`. Returns the violations. */
int histogram_violations(const std::string& body) {
    int bad = 0;
    std::string family;
    double last = -1.0, inf = -1.0;
    size_t at = 0;
    while (at < body.size()) {
        size_t nl = body.find('\n', at);
        if (nl == std::string::npos) nl = body.size();
        const std::string line = body.substr(at, nl - at);
        at = nl + 1;
        const size_t b = line.find("_bucket{");
        const size_t sp = line.rfind(' ');
        if (line.empty() || line[0] == '#' || sp == std::string::npos) continue;
        const double v = std::stod(line.substr(sp + 1));
        if (b != std::string::npos) {
            const std::string fam = line.substr(0, b);
            if (fam != family) { family = fam; last = -1.0; }
            if (v < last) ++bad;
            last = v;
            if (line.find("le=\"+Inf\"") != std::string::npos) inf = v;
        } else if (!family.empty() && line.rfind(family + "_count{", 0) == 0) {
            if (v != inf) ++bad;
        }
    }
    return bad;
}

}  /* namespace */

/* A scrape that races observations still renders histograms a scraper accepts: buckets that never
 * decrease and a +Inf bucket equal to `_count`. An observation counted in one bucket and not yet
 * in a higher one, or in a bucket and not yet in the count, makes the quantile interpolation
 * divide a negative difference. Four threads observe during five hundred scrapes. */
TEST(a_histogram_scraped_during_observations_stays_cumulative) {
    Metrics m("m", 1024);
    std::atomic<bool> stop{false};
    std::vector<std::thread> obs;
    for (int t = 0; t < 4; ++t)
        obs.emplace_back([&m, &stop, t] {
            const double v = 0.0001 * (t + 1);   /* below every bucket's bound */
            while (!stop.load(std::memory_order_relaxed))
                m.observe_request(1, 2, v, v, v, v, v, 1, 8, SuccessReason::Stop);
        });
    int bad = 0;
    for (int i = 0; i < 500 && bad == 0; ++i) bad += histogram_violations(m.render());
    stop.store(true);
    for (auto& t : obs) t.join();
    CHECK_EQ(bad, 0);
    CHECK_EQ(histogram_violations(m.render()), 0);
}

/* The queue time, the prefix cache as counters and the cache settings, under vLLM's names. A
 * request that was never admitted leaves the queue histogram alone, and the hits are the tokens
 * requests did not compute again. */
TEST(queue_time_and_prefix_cache_counters_render_under_vllm_names) {
    Metrics m("m", 1024);
    m.set_cache_config("fp8", 16, true, 4096);
    m.observe_request(10, 2, 0.5, 0.01, 1.0, 0.5, 0.25, 1, 8, SuccessReason::Stop);
    m.observe_request(10, 0, -1.0, -1.0, 0.1, -1.0, -1.0, 1, 8, SuccessReason::Abort);
    SchedMetrics sm;
    sm.prefix_queries = 100;
    sm.linear_cached_tokens = 60;
    m.sample(sm);
    const std::string out = m.render();
    const std::string L = "{model_name=\"m\",engine=\"0\"} ";
    CHECK(out.find("vllm:prefix_cache_queries_total" + L + "100\n") != std::string::npos);
    CHECK(out.find("vllm:prefix_cache_hits_total" + L + "60\n") != std::string::npos);
    CHECK(out.find("vllm:request_queue_time_seconds_count" + L + "1\n") != std::string::npos);
    CHECK(out.find("vllm:request_queue_time_seconds_sum" + L + "0.25\n") != std::string::npos);
    CHECK(out.find("vllm:cache_config_info{model_name=\"m\",engine=\"0\",cache_dtype=\"fp8\","
                   "block_size=\"16\",enable_prefix_caching=\"True\",prefix_cache_mib=\"4096\"} 1\n")
          != std::string::npos);
    CHECK_EQ(histogram_violations(out), 0);

    /* Not set: no info line. */
    Metrics bare("m", 1024);
    CHECK(bare.render().find("vllm:cache_config_info") == std::string::npos);
}

/* ================================================================== the response bodies
 *
 * The routes cover content end to end; these cases check shape. Client libraries deserialise into
 * generated types, so a field of the wrong type (a null where a string was declared, a missing
 * usage block, a number for an array) is a hard parse failure. These builders are also the only
 * place `usage` is assembled, which callers bill against.
 */

namespace {

OaiRequest a_request(const char* id, const char* model) {
    OaiRequest r;
    r.id = id;
    r.model = model;
    r.created = 1700000000;
    return r;
}

/* Assertions parse the body as a client would rather than searching it for substrings. */
json parsed(const std::string& body) {
    CHECK(json_complete(body));
    return json::parse(body, nullptr, false);
}

ModelCard card(const char* id, int64_t created, int64_t max_model_len = 0) {
    ModelCard c;
    c.id = id;
    c.created = created;
    c.max_model_len = max_model_len;
    return c;
}

}  /* namespace */

TEST(the_model_list_is_a_list_of_one_with_the_fields_clients_read) {
    const json o = parsed(models_body(card("q38-flashnext", 1700000000)));
    CHECK_EQ(o.value("object", std::string()), std::string("list"));
    CHECK(o["data"].is_array());
    CHECK_EQ(o["data"].size(), 1u);
    CHECK_EQ(o["data"][0].value("id", std::string()), std::string("q38-flashnext"));
    CHECK_EQ(o["data"][0].value("object", std::string()), std::string("model"));
    CHECK_EQ(o["data"][0].value("created", (int64_t)0), (int64_t)1700000000);
    /* `owned_by` is in the schema and generated clients deserialise it. */
    CHECK(o["data"][0].contains("owned_by"));
    /* No length known, none reported: a client falls back to its own default. */
    CHECK(!o["data"][0].contains("max_model_len"));
    CHECK(!o["data"][0].contains("root"));
    CHECK(!o["data"][0].contains("capabilities"));
}

TEST(the_model_entry_carries_the_capabilities_it_was_given) {
    ModelCard c = card("q38-flashnext", 1700000000);
    c.capabilities = R"({"tools":true,"image":false})";
    const json m = parsed(model_body(c));
    CHECK(m["capabilities"].value("tools", false));
    CHECK(!m["capabilities"].value("image", true));
}

TEST(the_model_entry_names_the_file_behind_the_served_name) {
    ModelCard c = card("q38-flashnext", 1700000000);
    c.root = "/models/qwen3.8-next-flash-fp8-iq4r-moe.rad";
    CHECK_EQ(parsed(model_body(c)).value("root", std::string()), c.root);
}

TEST(the_model_list_reports_the_context_the_server_runs_with) {
    const json o = parsed(models_body(card("q38-flashnext", 1700000000, 262144)));
    CHECK_EQ(o["data"][0].value("max_model_len", (int64_t)0), (int64_t)262144);
}

TEST(one_model_is_the_entry_itself_not_a_list_of_one) {
    const json m = parsed(model_body(card("q38-flashnext", 1700000000, 262144)));
    CHECK_EQ(m.value("object", std::string()), std::string("model"));
    CHECK_EQ(m.value("id", std::string()), std::string("q38-flashnext"));
    CHECK_EQ(m.value("max_model_len", (int64_t)0), (int64_t)262144);
    CHECK(!m.contains("data"));
}

TEST(a_chat_body_carries_usage_and_one_choice_per_completion) {
    ChoiceOut a;
    a.index = 0;
    a.msg.content = "hello";
    a.finish = Finish::Stop;
    ChoiceOut b;
    b.index = 1;
    b.msg.content = "world";
    b.finish = Finish::Length;

    UsageOut u;
    u.prompt = 10;
    u.completion = 4;
    u.cached = 6;

    const json o = parsed(chat_completion_body(a_request("chatcmpl-1", "m"), { a, b }, u));
    CHECK_EQ(o.value("object", std::string()), std::string("chat.completion"));
    CHECK_EQ(o.value("id", std::string()), std::string("chatcmpl-1"));
    CHECK_EQ(o["choices"].size(), 2u);
    CHECK_EQ(o["choices"][0]["message"].value("role", std::string()), std::string("assistant"));
    CHECK_EQ(o["choices"][0]["message"].value("content", std::string()), std::string("hello"));
    CHECK_EQ(o["choices"][0].value("finish_reason", std::string()), std::string("stop"));
    CHECK_EQ(o["choices"][1].value("finish_reason", std::string()), std::string("length"));
    CHECK_EQ(o["choices"][1].value("index", -1), 1);

    /* total is the sum, not a third counter, so a bill reconciles against the two halves. */
    CHECK_EQ(o["usage"].value("prompt_tokens", (int64_t)0), (int64_t)10);
    CHECK_EQ(o["usage"].value("completion_tokens", (int64_t)0), (int64_t)4);
    CHECK_EQ(o["usage"].value("total_tokens", (int64_t)0), (int64_t)14);
    /* The cached count sits where vLLM puts it, not at the top level. */
    CHECK_EQ(o["usage"]["prompt_tokens_details"].value("cached_tokens", (int64_t)0), (int64_t)6);

    /* logprobs is present and null rather than absent: a generated client distinguishes "not
     * asked for" from "the key does not exist". */
    CHECK(o["choices"][0].contains("logprobs"));
    CHECK(o["choices"][0]["logprobs"].is_null());
}

/* A tool call makes content null, not empty. The schema says a message with tool_calls has no
 * content, and clients that branch on `content == null` read an empty string as an ordinary text
 * reply and ignore the calls. */
TEST(a_chat_body_with_a_tool_call_sends_a_null_content) {
    ChoiceOut c;
    c.msg.tool_calls.push_back(ToolCall{ "call_1", "get_weather", "{\"city\":\"Berlin\"}" });
    c.finish = Finish::ToolCalls;

    const json o = parsed(chat_completion_body(a_request("chatcmpl-2", "m"), { c }, UsageOut{}));
    const json& m = o["choices"][0]["message"];
    CHECK(m["content"].is_null());
    CHECK(m["tool_calls"].is_array());
    CHECK_EQ(m["tool_calls"].size(), 1u);
    CHECK_EQ(o["choices"][0].value("finish_reason", std::string()), std::string("tool_calls"));

    /* Text beside a call keeps the text: content is only null when there is none. */
    ChoiceOut both = c;
    both.msg.content = "checking";
    const json p = parsed(chat_completion_body(a_request("chatcmpl-3", "m"), { both }, UsageOut{}));
    CHECK_EQ(p["choices"][0]["message"].value("content", std::string()), std::string("checking"));

    /* Reasoning goes under both spellings, and under neither when empty. */
    CHECK(!p["choices"][0]["message"].contains("reasoning_content"));
    CHECK(!p["choices"][0]["message"].contains("reasoning"));
    ChoiceOut think;
    think.msg.reasoning = "the user wants weather";
    think.msg.content = "it is cold";
    const json q = parsed(chat_completion_body(a_request("chatcmpl-4", "m"), { think }, UsageOut{}));
    CHECK_EQ(q["choices"][0]["message"].value("reasoning_content", std::string()),
             std::string("the user wants weather"));
    CHECK_EQ(q["choices"][0]["message"].value("reasoning", std::string()),
             std::string("the user wants weather"));
}

TEST(a_text_completion_body_is_the_other_shape_entirely) {
    ChoiceOut c;
    c.text = "once upon a time";
    c.finish = Finish::Stop;
    UsageOut u;
    u.prompt = 3;
    u.completion = 4;

    const json o = parsed(text_completion_body(a_request("cmpl-1", "m"), { c }, u));
    CHECK_EQ(o.value("object", std::string()), std::string("text_completion"));
    /* A completion choice carries `text` and no `message`; mixing them makes one endpoint's
     * response undeserialisable by the other's client. */
    CHECK_EQ(o["choices"][0].value("text", std::string()), std::string("once upon a time"));
    CHECK(!o["choices"][0].contains("message"));
    CHECK_EQ(o["usage"].value("total_tokens", (int64_t)0), (int64_t)7);
}

/* The logprob shapes differ by endpoint, both fixed by the schema: chat nests an array of
 * per-token objects under `content`, completions carries four parallel arrays. text_offset is
 * where each token's text starts in `text`, i.e. where the previous token ended. */
TEST(logprobs_take_each_endpoints_own_shape) {
    ChoiceOut c;
    c.text = "abcd";
    auto entry = [](int32_t id, const char* t, float lp, size_t end) {
        LogprobEntry e;
        e.token = id;
        e.text = e.bytes = t;
        e.logprob = lp;
        e.end = end;
        return e;
    };
    c.logprobs = { entry(1, "ab", -0.5f, 2), entry(2, "cd", -1.5f, 4) };
    OaiRequest r = a_request("x", "m");
    r.want_logprobs = true;

    const json chat = parsed(chat_completion_body(r, { c }, UsageOut{}));
    const json& lp = chat["choices"][0]["logprobs"];
    CHECK(lp["content"].is_array());
    CHECK_EQ(lp["content"].size(), 2u);
    CHECK_EQ(lp["content"][0].value("token", std::string()), std::string("ab"));
    CHECK_NEAR(lp["content"][0].value("logprob", 0.0), -0.5, 1e-6);
    CHECK(lp["content"][0]["bytes"].is_array());
    CHECK_EQ(lp["content"][0]["bytes"].size(), 2u);

    const json txt = parsed(text_completion_body(r, { c }, UsageOut{}));
    const json& tl = txt["choices"][0]["logprobs"];
    CHECK_EQ(tl["tokens"].size(), 2u);
    CHECK_EQ(tl["token_logprobs"].size(), 2u);
    CHECK_EQ(tl["text_offset"].size(), 2u);
    CHECK_EQ(tl["text_offset"][0].get<int64_t>(), (int64_t)0);
    CHECK_EQ(tl["text_offset"][1].get<int64_t>(), (int64_t)2);
}

/* stream_options.include_usage asks for a final SSE event with only usage. The empty choices
 * array must be present, not omitted, or a client waits for a chunk that never arrives. */
TEST(the_completion_usage_chunk_is_an_event_with_an_empty_choices_array) {
    UsageOut u;
    u.prompt = 5;
    u.completion = 2;
    const std::string ev = text_usage_chunk(a_request("cmpl-9", "m"), u);
    CHECK(ev.rfind("data: ", 0) == 0);
    CHECK(ev.compare(ev.size() - 2, 2, "\n\n") == 0);

    const json o = json::parse(ev.substr(6, ev.size() - 8), nullptr, false);
    CHECK(!o.is_discarded());
    CHECK_EQ(o.value("object", std::string()), std::string("text_completion"));
    CHECK(o["choices"].is_array());
    CHECK_EQ(o["choices"].size(), 0u);
    CHECK_EQ(o["usage"].value("total_tokens", (int64_t)0), (int64_t)7);
    CHECK_EQ(o.value("id", std::string()), std::string("cmpl-9"));
}

/* An embedding value is spelled with the shortest digits that read back as the float: the
 * engine's float32, not the extra digits widening it to a double would add. */
TEST(an_embeddings_value_is_spelled_as_its_float) {
    const std::vector<std::vector<float>> vecs = { { 0.1f, 1.0f, -2.5e-7f, 3.4028235e38f, -0.0f,
                                                     0.0123f } };
    UsageOut u;
    const std::string body = embeddings_body(a_request("emb-3", "m"), vecs, u);
    CHECK(body.find("\"embedding\":[0.1,1.0,-2.5e-07,3.4028235e+38,-0.0,0.0123]") !=
          std::string::npos);
    const json o = parsed(body);
    for (size_t j = 0; j < vecs[0].size(); ++j)
        CHECK_EQ(o["data"][0]["embedding"][j].get<float>(), vecs[0][j]);
}

/* Base64 encoding is per request and is the raw float buffer, which every client that asks for it
 * unpacks. Sending the JSON array under that flag, or base64 of a formatted array, would
 * deserialise to nonsense instead of failing. */
TEST(an_embeddings_body_honours_the_encoding_the_request_asked_for) {
    const std::vector<std::vector<float>> vecs = { { 1.0f, 2.0f }, { -0.5f, 0.25f } };
    UsageOut u;
    u.prompt = 8;

    OaiRequest plain = a_request("emb-1", "m");
    const json o = parsed(embeddings_body(plain, vecs, u));
    CHECK_EQ(o.value("object", std::string()), std::string("list"));
    CHECK_EQ(o["data"].size(), 2u);
    CHECK_EQ(o["data"][0].value("object", std::string()), std::string("embedding"));
    CHECK_EQ(o["data"][0].value("index", (int64_t)-1), (int64_t)0);
    CHECK_EQ(o["data"][1].value("index", (int64_t)-1), (int64_t)1);
    CHECK(o["data"][0]["embedding"].is_array());
    CHECK_NEAR(o["data"][0]["embedding"][1].get<double>(), 2.0, 1e-6);
    /* An embedding has no completion half, so total is the prompt. */
    CHECK_EQ(o["usage"].value("prompt_tokens", (int64_t)0), (int64_t)8);
    CHECK_EQ(o["usage"].value("total_tokens", (int64_t)0), (int64_t)8);

    OaiRequest b64 = a_request("emb-2", "m");
    b64.base64_embeddings = true;
    const json e = parsed(embeddings_body(b64, vecs, u));
    CHECK(e["data"][0]["embedding"].is_string());
    std::vector<uint8_t> raw;
    CHECK(base64_decode(e["data"][0]["embedding"].get<std::string>(), &raw));
    CHECK_EQ(raw.size(), 2 * sizeof(float));
    float back[2];
    std::memcpy(back, raw.data(), sizeof back);
    CHECK_NEAR(back[0], 1.0f, 1e-6);
    CHECK_NEAR(back[1], 2.0f, 1e-6);

    /* No inputs gives an empty list, not null or an error, in the envelope the client expects. */
    const json z = parsed(embeddings_body(plain, {}, UsageOut{}));
    CHECK(z["data"].is_array());
    CHECK_EQ(z["data"].size(), 0u);
}

/* ================================================================== the body reader
 *
 * parse_json_bounded is the DOM's bounded read (util/json.cpp). Its verdicts and typing are
 * the recorded contract the request parser has always been written against: a non-negative
 * integer reads as unsigned, an integer past uint64 as a double, and the value-budget and
 * depth boundaries refuse the same bodies. These cases are what a yyjson refresh is re-run
 * against (vendor/UPSTREAM). */

TEST(the_bounded_body_reader_agrees_with_the_reference_reader) {
    const char* cases[] = {
        "{\"a\":1e-999}", "{\"a\":1e-400}", "{\"a\":0e999}", "{\"a\":1e309}", "{\"a\":-1e309}",
        "{\"a\":1.7976931348623157e308}", "{\"a\":1.7976931348623159e308}",
        "{\"a\":2.2250738585072011e-308}", "{\"a\":5e-324}", "{\"a\":4e-324}",
        "{\"a\":0.1}", "{\"a\":123456789012345678901234567890}",
        "{\"a\":9223372036854775807}", "{\"a\":9223372036854775808}",
        "{\"a\":-9223372036854775808}", "{\"a\":-9223372036854775809}",
        "{\"a\":18446744073709551615}", "{\"a\":18446744073709551616}",
        "{\"a\":-0}", "{\"a\":-0.0}", "{\"a\":0.0}", "{\"a\":1E7}", "{\"a\":1e+7}",
        "{\"a\":0x10}", "{\"a\":.5}", "{\"a\":1.}", "{\"a\":1e}", "{\"a\":01}", "{\"a\":+1}",
        "{\"a\":NaN}", "{\"a\":Infinity}",
        "{\"a\":\"\\u0000\"}", "{\"a\":\"\\ud83d\\ude00\"}", "{\"a\":\"\\ude00\"}",
        "{\"a\":\"\\ud800\"}", "{\"a\":\"\\/\"}", "{\"a\":\"\\x\"}", "{\"a\":\"\\u12g4\"}",
        "{\"a\":\"caf\\u00e9\"}", "{\"a\":\"\x7f\"}", "{\"a\":\"\"}",
        "{\"a\":1,\"b\":0,\"a\":2,\"b\":3}", "{\"\\u0061\":1}",
        "[]", "{}", "null", "true", "42", "-17", "\"hi\"", "  {\"a\":1}  ",
        "[1,2,]", "{,}", "{\"a\"}", "{\"a\":}", "[}", "nul", "{\"a\":\"unterminated}",
        "", "   ", "{\"a\":1} x", "{\"a\":1} {\"b\":2}",
    };
    for (const char* c : cases) {
        const std::string s = c;
        bool over = false, deep = false;
        const json got = parse_json_bounded(s, 1 << 20, &over, &deep);
        const json want = json::parse(s, nullptr, false);
        CHECK_EQ(got.is_discarded(), want.is_discarded());
        CHECK(!over && !deep);
        if (!want.is_discarded()) {
            CHECK_EQ(got.dump(), want.dump());
            CHECK_EQ(got.type(), want.type());
        }
    }
    /* A BOM is skipped, as the reference reader skips it; invalid UTF-8 and a CESU-8
     * surrogate are refused. */
    {
        bool over = false, deep = false;
        const json j = parse_json_bounded(std::string("\xef\xbb\xbf") + "{\"a\":1}", 0, &over, &deep);
        CHECK(!j.is_discarded());
        CHECK_EQ(j["a"].get<uint64_t>(), 1u);
        CHECK(parse_json_bounded("{\"a\":\"\xff\"}", 0, &over, &deep).is_discarded());
        CHECK(parse_json_bounded("{\"a\":\"\xc3\x28\"}", 0, &over, &deep).is_discarded());
        CHECK(parse_json_bounded("{\"a\":\"\xed\xa0\x80\"}", 0, &over, &deep).is_discarded());
        CHECK(!parse_json_bounded("{\"a\":\"\xc3\xa9\"}", 0, &over, &deep).is_discarded());
    }
    /* Number typing is pinned, because the request parser's checks read it. */
    {
        bool over = false, deep = false;
        const json j = parse_json_bounded(
            "{\"u\":5,\"n\":-5,\"big\":9223372036854775808,\"huge\":18446744073709551616}",
            0, &over, &deep);
        CHECK(j["u"].is_number_unsigned());
        CHECK(!j["n"].is_number_unsigned() && j["n"].is_number_integer());
        CHECK(j["big"].is_number_unsigned());
        CHECK(j["huge"].is_number_float());
        /* Duplicate keys: the last value at the first key's position. */
        const json d = parse_json_bounded("{\"a\":1,\"b\":0,\"a\":2}", 0, &over, &deep);
        CHECK_EQ(d.dump(), std::string("{\"a\":2,\"b\":0}"));
    }
}

TEST(the_bounded_body_reader_keeps_the_value_and_depth_boundaries) {
    auto ask = [](const std::string& s, bool* over, bool* deep, int64_t ctx = 0) {
        *over = *deep = false;
        return parse_json_bounded(s, ctx, over, deep);
    };
    auto nested = [](int n, bool scalar) {
        std::string s((size_t)n, '[');
        if (scalar) s += '1';
        return s + std::string((size_t)n, ']');
    };
    bool over, deep;
    /* A value nested kJsonMaxDepth below the root is the deepest accepted. The innermost
     * empty container sits one level above its would-be elements, hence the off-by-one
     * between the scalar and empty shapes: both are the reference reader's boundaries. */
    ask(nested(kJsonMaxDepth, true), &over, &deep);
    CHECK(!deep && !over);
    ask(nested(kJsonMaxDepth + 1, true), &over, &deep);
    CHECK(deep && !over);
    ask(nested(kJsonMaxDepth + 1, false), &over, &deep);
    CHECK(!deep && !over);
    ask(nested(kJsonMaxDepth + 2, false), &over, &deep);
    CHECK(deep && !over);

    /* Every value counts once -- containers included, keys excluded. The budget at ctx 0 is
     * the floor. One repeated key in the object shape: a million distinct keys would pay
     * the ordered map's quadratic insert for no extra information. */
    const int64_t budget = json_value_budget(0);
    auto flat = [](int64_t n, bool object) {
        std::string s = object ? "{" : "[";
        for (int64_t i = 0; i < n; ++i) {
            if (i) s += ',';
            s += object ? "\"k\":1" : "1";
        }
        return s + (object ? '}' : ']');
    };
    ask(flat(budget - 1, false), &over, &deep);   /* root + elements == budget */
    CHECK(!over && !deep);
    ask(flat(budget, false), &over, &deep);
    CHECK(over && !deep);
    ask(flat(budget - 1, true), &over, &deep);
    CHECK(!over && !deep);
    ask(flat(budget, true), &over, &deep);
    CHECK(over && !deep);

    /* An over-deep or over-budget body still returns a usable (null) value rather than a
     * discarded one, matching the reference reader: a caller tests discarded for "not JSON"
     * before it tests the flags. */
    CHECK(!ask(nested(kJsonMaxDepth + 5, true), &over, &deep).is_discarded());
    CHECK(deep);
}

RAD_TEST_MAIN()
