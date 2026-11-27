/* tests/images_test.cpp -- OpenAI's images API over an image generator (core/server/images.cpp).
 *
 * The generator is faked (server_fakes.h: FakeImageGen finishes every request on submit), so
 * these pin what the server owns: which requests are refused and under which field, the request
 * it hands the generator (sizes rounded to the model's multiple, steps from `quality`, a seed a
 * JavaScript client can read back), the usage split between text and pictures, the background an
 * `auto` resolves to, failures mapped to 400 or 500 in both the body and a stream, multipart
 * uploads, and the stream's events. Cases that encode or decode a picture need the media decoder
 * and skip without it.
 */
#include "server_fakes.h"
#include "server/front.h"

#include "mm/media.h"

#include <future>

namespace {

struct Images {
    FakeImageGen img;
    std::unique_ptr<Fixture> f;
    Images() {
        f.reset(new Fixture(Fixture::Tweak{ [this](ServerOptions& o, Deps& d) {
            d.images = &img;
            o.image_model = true;
        } }));
    }
    HttpResponse post(const char* path, const json& body) { return f->post(path, body.dump()); }
    json gen(const json& body, int status = 200) {
        HttpResponse r = post("/v1/images/generations", body);
        CHECK_EQ(r.status, status);
        return json::parse(r.body, nullptr, false);
    }
    const gen::Request& last() const { return *img.got.back(); }
};

/* A PNG of w x h with every pixel `rgba`. */
std::string png(int32_t w, int32_t h, uint8_t alpha_left, uint8_t alpha_right) {
    std::vector<uint8_t> px((size_t)w * h * 4, 200);
    for (int32_t y = 0; y < h; ++y)
        for (int32_t x = 0; x < w; ++x) px[((size_t)y * w + x) * 4 + 3] = x < w / 2 ? alpha_left : alpha_right;
    std::vector<uint8_t> file;
    std::string why;
    if (mm::encode_image(px.data(), w, h, 4, "png", 0, &file, &why) < 0) return std::string();
    return std::string(file.begin(), file.end());
}

std::string data_url(const std::string& bytes) {
    return "data:image/png;base64," + base64_encode(bytes.data(), bytes.size());
}

}  /* namespace */

/* ================================================================== refusals */

/* A request this deployment cannot serve is a 400 naming the field it is about, before anything
 * is queued: unknown fields, a missing or non-string prompt, n outside 1..lanes, url output (no
 * picture is stored), a transparent jpeg, previews without a stream, a step count on a model with
 * its own schedule, half a size (naming the half that is missing), a malformed or oversized size,
 * guidance without a negative prompt, and pictures sent to the endpoint that takes none. */
TEST(an_images_request_that_cannot_be_served_names_its_field) {
    Images s;
    s.img.inf.fixed_schedule = true;
    s.img.inf.fixed_steps = 8;
    struct Case { json body; const char* param; };
    const Case cases[] = {
        { { { "prompt", "x" }, { "wobble", 1 } }, "wobble" },
        { json::object(), "prompt" },
        { { { "prompt", 5 } }, "prompt" },
        { { { "prompt", "x" }, { "n", 0 } }, "n" },
        { { { "prompt", "x" }, { "n", 5 } }, "n" },
        { { { "prompt", "x" }, { "response_format", "url" } }, "response_format" },
        { { { "prompt", "x" }, { "output_format", "jpeg" }, { "background", "transparent" } }, "background" },
        { { { "prompt", "x" }, { "partial_images", 2 } }, "partial_images" },
        { { { "prompt", "x" }, { "num_inference_steps", 20 } }, "num_inference_steps" },
        { { { "prompt", "x" }, { "width", 256 } }, "height" },
        { { { "prompt", "x" }, { "height", 256 } }, "width" },
        { { { "prompt", "x" }, { "size", "12x" } }, "size" },
        { { { "prompt", "x" }, { "size", "2048x2048" } }, "size" },
        { { { "prompt", "x" }, { "true_cfg_scale", 4.0 } }, "true_cfg_scale" },
        { { { "prompt", "x" }, { "image", "data:image/png;base64,AAAA" } }, "image" },
    };
    for (const Case& c : cases) {
        const json e = s.gen(c.body, 400);
        CHECK_EQ(str_at(e["error"], "param"), std::string(c.param));
        if (str_at(e["error"], "param") != c.param) fprintf(stderr, "    %s\n", c.body.dump().c_str());
    }
    CHECK(s.gen({ { "prompt", "x" }, { "response_format", "url" } }, 400)["error"]["supported"] ==
          json::array({ "b64_json" }));
    CHECK(s.img.got.empty());
    /* An edit needs a picture; a model that reads none takes no edit at all. */
    CHECK_EQ(str_at(json::parse(s.post("/v1/images/edits", { { "prompt", "x" } }).body)["error"], "param"),
             std::string("image"));
    s.img.inf.max_refs = 0;
    CHECK_EQ(s.post("/v1/images/edits", { { "prompt", "x" } }).status, 501);
}

/* An image model generates no text, so the text endpoints point at the images endpoints, and its
 * model entry says what it does: image generation and editing, no grammar, no video. */
TEST(an_image_model_names_the_images_endpoints_and_its_capabilities) {
    Images s;
    HttpResponse r = s.f->post("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}]})");
    CHECK_EQ(r.status, 501);
    CHECK(has(r.body, "/v1/images/generations"));
    CHECK_EQ(s.f->post("/v1/completions", R"({"prompt":"x"})").status, 501);
    const json caps = json::parse(s.f->get("/v1/models").body)["data"][0]["capabilities"];
    CHECK(caps.value("image_generation", false));
    CHECK(caps.value("image_edit", false));
    CHECK(!caps.value("grammar", true));
    CHECK(!caps.value("video", true));
}

/* ================================================================== the request handed over */

/* What reaches the generator: sides rounded down to the model's multiple, steps from `quality`
 * (half the default for low, three quarters for medium) unless named, the seed given or a fresh
 * one under 2^53 (returned for reuse, and JavaScript reads JSON numbers as doubles), and each
 * image's seed counting up from it. */
TEST(the_generator_gets_rounded_sizes_quality_steps_and_a_readable_seed) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    Images s;
    const json a = s.gen({ { "prompt", "x" }, { "size", "300x200" }, { "quality", "low" },
                           { "seed", 5 }, { "n", 2 } });
    CHECK_EQ(s.last().width, 288);
    CHECK_EQ(s.last().height, 192);
    CHECK_EQ(s.last().steps, 10);
    CHECK_EQ(s.last().seed, (uint64_t)5);
    CHECK_EQ(a.value("size", ""), std::string("288x192"));
    REQUIRE(a["data"].size() == 2);
    CHECK_EQ(a["data"][1]["seed"].get<uint64_t>(), (uint64_t)6);
    std::vector<uint8_t> file;
    REQUIRE(base64_decode(a["data"][0]["b64_json"].get<std::string>(), &file));
    CHECK(file.size() > 8 && file[1] == 'P' && file[2] == 'N' && file[3] == 'G');

    s.gen({ { "prompt", "x" }, { "quality", "medium" } });
    CHECK_EQ(s.last().steps, 15);
    CHECK_EQ(s.last().width, 256);
    CHECK_EQ(s.last().height, 128);
    s.gen({ { "prompt", "x" }, { "quality", "low" }, { "num_inference_steps", 7 } });
    CHECK_EQ(s.last().steps, 7);
    for (const json& seed : { json(nullptr), json(-1) }) {
        json body = { { "prompt", "x" } };
        if (!seed.is_null()) body["seed"] = seed;
        const json r = s.gen(body);
        CHECK(s.last().seed < ((uint64_t)1 << 53));
        CHECK(r["data"][0]["seed"].get<uint64_t>() < ((uint64_t)1 << 53));
    }
}

/* Usage counts the pictures' token runs as image tokens, once in each branch that carries them:
 * twice with guidance (the negative prompt is a second conditioning). An edit's pictures are
 * resized to the model's reference area at their own aspect before the encoder sees them. */
TEST(usage_counts_each_pictures_tokens_in_every_branch) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    Images s;
    const std::string pic = png(40, 20, 255, 255);
    REQUIRE(!pic.empty());
    const json plain = json::parse(s.post("/v1/images/edits",
        { { "prompt", "x" }, { "image", data_url(pic) } }).body);
    const json& u = plain["usage"];
    CHECK_EQ(u["input_tokens_details"].value("image_tokens", (int64_t)-1), (int64_t)3);
    CHECK_EQ(u["input_tokens_details"].value("text_tokens", (int64_t)-1),
             u.value("input_tokens", (int64_t)0) - 3);
    REQUIRE(!s.img.frames.empty());
    CHECK(s.img.frames.back() == std::make_pair(96, 32));
    CHECK_EQ(plain.value("size", ""), std::string("256x128"));
    const json guided = json::parse(s.post("/v1/images/edits",
        { { "prompt", "x" }, { "image", data_url(pic) }, { "negative_prompt", "" },
          { "true_cfg_scale", 4.0 } }).body);
    CHECK_EQ(guided["usage"]["input_tokens_details"].value("image_tokens", (int64_t)-1), (int64_t)6);
    CHECK(s.last().has_neg);
}

/* A model that reads its prompt through a second tokenizer gets the user's prompt as that tokenizer
 * encodes it, in each branch, and a prompt longer than it reads is refused under `prompt`. A
 * prompt shorter than the model's minimum after the fixed prefix is padded with its padding token
 * (an empty prompt is one, and is not turned into " " first). */
TEST(a_second_tokenizer_and_a_minimum_prompt_reach_the_generator) {
    Images s;
    s.img.inf.aux_tokens = true;
    s.img.inf.aux_max_tokens = 4;
    s.img.inf.min_tokens = 2;
    s.img.inf.pad_token = 9;
    s.gen({ { "prompt", "ab" }, { "negative_prompt", "c" }, { "true_cfg_scale", 4.0 } });
    REQUIRE(!s.img.got.empty());
    CHECK(s.last().pos.aux == std::vector<int32_t>({ 100 + 'a', 100 + 'b', 1 }));
    CHECK(s.last().neg.aux == std::vector<int32_t>({ 100 + 'c', 1 }));
    /* "c" is one token after the prefix, padded to two. */
    const gen::Branch& n = s.last().neg;
    REQUIRE((int64_t)n.tokens.size() - n.n_drop == 2);
    CHECK_EQ(n.tokens[(size_t)n.n_drop], (int32_t)'c');
    CHECK_EQ(n.tokens.back(), (int32_t)9);
    s.gen({ { "prompt", "" } });
    const gen::Branch& p = s.last().pos;
    REQUIRE((int64_t)p.tokens.size() - p.n_drop == 2);
    CHECK_EQ(p.tokens[(size_t)p.n_drop], (int32_t)9);
    CHECK(p.aux == std::vector<int32_t>({ 1 }));
    /* An empty negative prompt stays empty for such a model: all padding. */
    s.gen({ { "prompt", "x" }, { "negative_prompt", "" }, { "true_cfg_scale", 4.0 } });
    const gen::Branch& e = s.last().neg;
    REQUIRE((int64_t)e.tokens.size() - e.n_drop == 2);
    CHECK_EQ(e.tokens[(size_t)e.n_drop], (int32_t)9);
    CHECK(e.aux == std::vector<int32_t>({ 1 }));
    const json longp = s.gen({ { "prompt", "abcd" } }, 400);
    CHECK_EQ(str_at(longp["error"], "param"), std::string("prompt"));
}

/* A discrete-time model takes `sampler` and `scheduler` by ComfyUI's names, defaulting to its
 * own, and refuses one it does not serve; a flow-matching model refuses both. A prompt read in
 * windows reaches the generator cut into runs between their markers, each padded to the window,
 * after the template's fixed prefix, and a prompt that fills its last window exactly takes no
 * empty one after it. */
TEST(a_discrete_model_takes_its_sampler_and_reads_the_prompt_in_windows) {
    Images s;
    const json flow = s.gen({ { "prompt", "x" }, { "sampler", "euler" } }, 400);
    CHECK_EQ(str_at(flow["error"], "param"), std::string("sampler"));
    s.img.inf.samplers = (1u << RAD_GEN_SAMPLER_EULER) | (1u << RAD_GEN_SAMPLER_DPMPP_2M);
    s.img.inf.default_sampler = RAD_GEN_SAMPLER_EULER;
    s.img.inf.default_scheduler = RAD_GEN_SCHED_NORMAL;
    s.img.inf.chunk_tokens = 4;
    s.img.inf.chunk_bos = 2;
    s.img.inf.chunk_eos = 3;
    s.img.inf.chunk_pad = 9;
    s.gen({ { "prompt", "abcdefghij" } });
    REQUIRE(!s.img.got.empty());
    CHECK_EQ(s.last().sampler, (int32_t)RAD_GEN_SAMPLER_EULER);
    CHECK_EQ(s.last().scheduler, (int32_t)RAD_GEN_SCHED_NORMAL);
    const std::vector<int32_t> want = { '<', 's', '>', 2, 'a', 'b', 'c', 'd', 3,
                                        2, 'e', 'f', 'g', 'h', 3, 2, 'i', 'j', 3, 9, 9 };
    CHECK(s.last().pos.tokens == want);
    CHECK_EQ(s.last().pos.n_drop, (int32_t)3);
    s.gen({ { "prompt", "x" }, { "sampler", "dpmpp_2m" }, { "scheduler", "karras" } });
    CHECK_EQ(s.last().sampler, (int32_t)RAD_GEN_SAMPLER_DPMPP_2M);
    CHECK_EQ(s.last().scheduler, (int32_t)RAD_GEN_SCHED_KARRAS);
    /* A prompt that fills a window takes no second one. */
    s.gen({ { "prompt", "abcd" } });
    const std::vector<int32_t> one = { '<', 's', '>', 2, 'a', 'b', 'c', 'd', 3 };
    CHECK(s.last().pos.tokens == one);
    const json bad = s.gen({ { "prompt", "x" }, { "sampler", "euler_ancestral" } }, 400);
    CHECK_EQ(str_at(bad["error"], "param"), std::string("sampler"));
    CHECK(bad["error"]["supported"] == json::array({ "euler", "dpmpp_2m" }));
    const json badk = s.gen({ { "prompt", "x" }, { "scheduler", "ddim_uniform" } }, 400);
    CHECK_EQ(str_at(badk["error"], "param"), std::string("scheduler"));
}

/* `rebalance` is a boolean, JSON or form ("true", "1", "false", "0"). `true` reaches the generator
 * as RAD_GEN_COND_REBALANCE only on a model that declares it, and is a 400 naming the field on
 * one that does not; `false` is accepted by every model. */
TEST(rebalance_reaches_a_model_that_declares_it_and_is_refused_by_one_that_does_not) {
    Images s;
    const json no = s.gen({ { "prompt", "x" }, { "rebalance", true } }, 400);
    CHECK_EQ(str_at(no["error"], "param"), std::string("rebalance"));
    CHECK(s.img.got.empty());
    s.gen({ { "prompt", "x" }, { "rebalance", false } });
    REQUIRE(!s.img.got.empty());
    CHECK_EQ(s.last().cond_opts, (int32_t)0);
    s.img.inf.cond_opts = RAD_GEN_COND_REBALANCE;
    for (const json& bad : { json("yes"), json(1), json::array() }) {
        const json e = s.gen({ { "prompt", "x" }, { "rebalance", bad } }, 400);
        CHECK_EQ(str_at(e["error"], "param"), std::string("rebalance"));
    }
    s.gen({ { "prompt", "x" }, { "rebalance", true } });
    CHECK_EQ(s.last().cond_opts, (int32_t)RAD_GEN_COND_REBALANCE);
    s.gen({ { "prompt", "x" } });
    CHECK_EQ(s.last().cond_opts, (int32_t)0);
    s.gen({ { "prompt", "x" }, { "rebalance", nullptr } });
    CHECK_EQ(s.last().cond_opts, (int32_t)0);
    /* The form's text values. */
    HttpRequest q;
    q.method = "POST";
    q.path = "/v1/images/generations";
    q.form = { FormPart{ "prompt", "", "", "a cat" }, FormPart{ "rebalance", "", "", "1" } };
    CHECK_EQ(s.f->srv->router().dispatch(q).status, 200);
    CHECK_EQ(s.last().cond_opts, (int32_t)RAD_GEN_COND_REBALANCE);
    q.form[1].data = "false";
    CHECK_EQ(s.f->srv->router().dispatch(q).status, 200);
    CHECK_EQ(s.last().cond_opts, (int32_t)0);
    q.form[1].data = "on";
    CHECK_EQ(s.f->srv->router().dispatch(q).status, 400);
}

/* `rebalance` also names a preset: "legacy" is `true`, "balanced", "detail" and "subtle" set
 * RAD_GEN_COND_REBALANCE with the preset's row in RAD_GEN_COND_REBALANCE_PRESET, in JSON and in a
 * form. A model whose rebalance declares no presets takes `true` and "legacy" and refuses the
 * others with a 400 naming the field; an unknown name is a 400 everywhere. */
TEST(rebalance_presets_reach_a_model_that_declares_them) {
    Images s;
    const int32_t R = RAD_GEN_COND_REBALANCE, sh = RAD_GEN_COND_REBALANCE_PRESET_SHIFT;
    s.img.inf.cond_opts = RAD_GEN_COND_REBALANCE;
    s.gen({ { "prompt", "x" }, { "rebalance", "legacy" } });
    CHECK_EQ(s.last().cond_opts, R);
    for (const char* p : { "balanced", "detail", "subtle" }) {
        const json e = s.gen({ { "prompt", "x" }, { "rebalance", p } }, 400);
        CHECK_EQ(str_at(e["error"], "param"), std::string("rebalance"));
    }
    s.img.inf.cond_opts = RAD_GEN_COND_REBALANCE | RAD_GEN_COND_REBALANCE_PRESET;
    const struct { const char* name; int32_t opts; } kWant[] = {
        { "legacy", R }, { "balanced", R | 1 << sh }, { "detail", R | 2 << sh }, { "subtle", R | 3 << sh },
    };
    for (const auto& w : kWant) {
        s.gen({ { "prompt", "x" }, { "rebalance", w.name } });
        CHECK_EQ(s.last().cond_opts, w.opts);
    }
    s.gen({ { "prompt", "x" }, { "rebalance", true } });
    CHECK_EQ(s.last().cond_opts, R);
    for (const json& bad : { json("Balanced"), json("strong"), json("") }) {
        const json e = s.gen({ { "prompt", "x" }, { "rebalance", bad } }, 400);
        CHECK_EQ(str_at(e["error"], "param"), std::string("rebalance"));
    }
    HttpRequest q;
    q.method = "POST";
    q.path = "/v1/images/generations";
    q.form = { FormPart{ "prompt", "", "", "a cat" }, FormPart{ "rebalance", "", "", "detail" } };
    CHECK_EQ(s.f->srv->router().dispatch(q).status, 200);
    CHECK_EQ(s.last().cond_opts, R | 2 << sh);
    q.form[1].data = "nope";
    CHECK_EQ(s.f->srv->router().dispatch(q).status, 400);
}

/* `background: auto` keeps an edited picture's transparency and is opaque otherwise; a mask must
 * be the first picture's size. */
/* A model whose card states guidance as cond + s (cond - uncond) and guides by default (Krea 2):
 * a request naming nothing is guided at the card's scale with the model's own empty negative
 * prompt, handed on as s + 1; `guidance_scale` is the same field as `true_cfg_scale`, 0 turns it
 * off, and naming both is refused. An RGB model refuses a transparent background. */
TEST(guidance_follows_the_model_cards_terms_and_its_default_negative) {
    Images s;
    s.img.inf.cfg_original = true;
    s.img.inf.default_cfg = 4.5;
    s.img.inf.has_default_negative = true;
    s.img.inf.default_negative = "";
    s.img.inf.pixel_channels = 3;
    s.gen({ { "prompt", "a fox" } });
    REQUIRE(!s.img.got.empty());
    CHECK(s.last().has_neg);
    CHECK_EQ(s.last().cfg, 5.5f);
    s.gen({ { "prompt", "a fox" }, { "guidance_scale", 2.0 } });
    CHECK(s.last().has_neg);
    CHECK_EQ(s.last().cfg, 3.0f);
    s.gen({ { "prompt", "a fox" }, { "guidance_scale", 0.0 } });
    CHECK(!s.last().has_neg);
    const json both = s.gen({ { "prompt", "x" }, { "guidance_scale", 2.0 }, { "true_cfg_scale", 2.0 } },
                            400);
    CHECK_EQ(str_at(both["error"], "param"), std::string("guidance_scale"));
    const json alpha = s.gen({ { "prompt", "x" }, { "background", "transparent" } }, 400);
    CHECK_EQ(str_at(alpha["error"], "param"), std::string("background"));
}

TEST(an_auto_background_follows_the_pictures_alpha) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    Images s;
    const std::string opaque = png(40, 20, 255, 255), half = png(40, 20, 0, 255);
    REQUIRE(!opaque.empty() && !half.empty());
    auto edit = [&](const std::string& pic) {
        return json::parse(s.post("/v1/images/edits", { { "prompt", "x" }, { "image", data_url(pic) } }).body);
    };
    CHECK_EQ(edit(opaque).value("background", ""), std::string("opaque"));
    CHECK_EQ(edit(half).value("background", ""), std::string("transparent"));
    CHECK_EQ(s.gen({ { "prompt", "x" } }).value("background", ""), std::string("opaque"));
    const json m = json::parse(s.post("/v1/images/edits",
        { { "prompt", "x" }, { "image", data_url(opaque) }, { "mask", data_url(png(10, 10, 0, 0)) } }).body);
    CHECK_EQ(str_at(m["error"], "param"), std::string("mask"));
}

/* ================================================================== failures */

/* The generator's refusal at submit is the client's 400 when it says the request can never run
 * (RAD_E_INVAL) and a 503 otherwise; a request that failed while running is a 500 carrying the
 * generator's reason, or a 400 for its admission refusal. A stream ends in an error event of the
 * same type. A prompt the model's template does not open with its fixed turn is the server's
 * failure, a 500 typed as one. */
TEST(a_generators_failure_is_a_400_or_a_500_streamed_or_not) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    Images s;
    s.img.submit_status = RAD_E_INVAL;
    s.img.submit_why = "too long for the cache";
    CHECK(has(s.gen({ { "prompt", "x" } }, 400)["error"].value("message", ""), "too long"));
    s.img.submit_status = RAD_E_IO;
    s.gen({ { "prompt", "x" } }, 503);
    s.img.submit_status = RAD_OK;
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);

    for (int st : { RAD_E_INVAL, RAD_E_IO }) {
        s.img.finish_status = st;
        s.img.finish_why = "the decoder ran out of memory";
        const json e = s.gen({ { "prompt", "x" } }, st == RAD_E_INVAL ? 400 : 500)["error"];
        CHECK_EQ(e.value("message", ""), s.img.finish_why);
        CHECK_EQ(e.value("type", ""), std::string(st == RAD_E_INVAL ? "invalid_request_error"
                                                                    : "server_error"));
        HttpResponse r = s.post("/v1/images/generations", { { "prompt", "x" }, { "stream", true } });
        CHECK_EQ(r.status, 200);
        const std::vector<json> ev = sse_events(drain(r));
        REQUIRE(!ev.empty());
        CHECK_EQ(ev.back().value("type", ""), std::string("error"));
        CHECK_EQ(ev.back()["error"].value("type", ""), e.value("type", ""));
    }
    s.img.finish_status = RAD_OK;
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);

    s.img.inf.drop_prefix = "<not the template's turn>";
    const json e = s.gen({ { "prompt", "x" } }, 500)["error"];
    CHECK_EQ(e.value("type", ""), std::string("internal_server_error"));
}

/* ================================================================== the wire forms */

/* An upload as multipart form-data: fields arrive as text (n as "2") and pictures as `image[]`
 * parts. */
TEST(a_multipart_edit_reads_text_fields_and_picture_parts) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    Images s;
    const std::string pic = png(40, 20, 255, 255);
    HttpRequest q;
    q.method = "POST";
    q.path = "/v1/images/edits";
    q.form = { FormPart{ "prompt", "", "", "a cat" }, FormPart{ "n", "", "", "2" },
               FormPart{ "image[]", "a.png", "image/png", pic },
               FormPart{ "image[]", "b.png", "image/png", pic } };
    HttpResponse r = s.f->srv->router().dispatch(q);
    CHECK_EQ(r.status, 200);
    CHECK_EQ(json::parse(r.body)["data"].size(), (size_t)2);
    CHECK_EQ(s.last().n, 2);
    CHECK_EQ(s.last().refs.size(), (size_t)2);
    q.form[1].data = "two";
    CHECK_EQ(s.f->srv->router().dispatch(q).status, 400);
}

/* A stream sends each preview as a partial_image event numbered from 0, then the image as
 * `completed` with its usage, per image. */
TEST(a_streamed_generation_sends_previews_then_the_image) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    Images s;
    HttpResponse r = s.post("/v1/images/generations",
                            { { "prompt", "x" }, { "stream", true }, { "partial_images", 2 } });
    CHECK_EQ(r.status, 200);
    const std::vector<json> ev = sse_events(drain(r));
    REQUIRE(ev.size() == 3);
    CHECK_EQ(ev[0].value("type", ""), std::string("image_generation.partial_image"));
    CHECK_EQ(ev[0].value("partial_image_index", -1), 0);
    CHECK_EQ(ev[1].value("partial_image_index", -1), 1);
    CHECK_EQ(ev[2].value("type", ""), std::string("image_generation.completed"));
    CHECK(ev[2].contains("usage"));
    CHECK(!ev[2].contains("partial_image_index"));
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);
}

TEST(auto_edit_size_respects_a_tight_pixel_limit_after_rounding) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    Images s;
    s.img.inf.default_width = s.img.inf.default_height = 1024;
    s.img.inf.max_tokens = 4096;
    const std::string pic = png(128, 96, 255, 255);
    REQUIRE(!pic.empty());
    const HttpResponse res = s.post("/v1/images/edits",
        { { "prompt", "x" }, { "image", data_url(pic) } });
    REQUIRE_EQ(res.status, 200);
    CHECK_EQ(s.last().width, 1152);
    CHECK_EQ(s.last().height, 896);
}


/* A model whose references fit the output (Krea's instruction edit): `ref_boost` is a finite
 * number in 0.0001..16 on an edit only, masks are refused, a reference is resized to the output's
 * grid (near-matched aspects fill it, others fit inside it), and the grounding picture is cut
 * to 768 px on its long side. */
TEST(a_reference_fitting_edit_bounds_ref_boost_refuses_masks_and_resizes_references) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    Images s;
    s.img.inf.refs_fit_output = true;
    const std::string pic = png(100, 100, 255, 255);
    REQUIRE(!pic.empty());
    auto edit = [&](const json& extra, int status) {
        json b = { { "prompt", "x" }, { "image", data_url(pic) }, { "size", "256x128" } };
        for (auto it = extra.begin(); it != extra.end(); ++it) b[it.key()] = it.value();
        HttpResponse r = s.post("/v1/images/edits", b);
        CHECK_EQ(r.status, status);
        return json::parse(r.body, nullptr, false);
    };
    /* No boost leaves the bias at 0; in range it is the natural log. */
    edit(json::object(), 200);
    CHECK_EQ(s.last().ref_log_bias, 0.0f);
    edit({ { "ref_boost", 1.0 } }, 200);
    CHECK_EQ(s.last().ref_log_bias, 0.0f);
    edit({ { "ref_boost", 2.0 } }, 200);
    CHECK(std::fabs(s.last().ref_log_bias - std::log(2.0f)) < 1e-6f);
    edit({ { "ref_boost", 16 } }, 200);
    CHECK(std::fabs(s.last().ref_log_bias - std::log(16.0f)) < 1e-6f);
    edit({ { "ref_boost", 0.0001 } }, 200);
    CHECK(std::fabs(s.last().ref_log_bias - std::log(0.0001f)) < 1e-5f);
    const size_t accepted = s.img.got.size();
    /* Out of range, not a number, or not finite: refused under the field. */
    const json bad[] = { 0.00009, 0.0, -1.0, 16.1, 1e300, "2", true, nullptr, json::array() };
    for (const json& v : bad) {
        const json e = edit({ { "ref_boost", v } }, 400);
        CHECK_EQ(str_at(e["error"], "param"), std::string("ref_boost"));
    }
    /* A form field arrives as text: "nan" and "inf" parse as doubles and are still refused. */
    for (const char* v : { "nan", "inf", "-inf", "16.5", "0" }) {
        HttpRequest q;
        q.method = "POST";
        q.path = "/v1/images/edits";
        q.form = { FormPart{ "prompt", "", "", "x" }, FormPart{ "ref_boost", "", "", v },
                   FormPart{ "image", "a.png", "image/png", pic } };
        CHECK_EQ(s.f->srv->router().dispatch(q).status, 400);
    }
    {
        HttpRequest q;
        q.method = "POST";
        q.path = "/v1/images/edits";
        q.form = { FormPart{ "prompt", "", "", "x" }, FormPart{ "ref_boost", "", "", "3" },
                   FormPart{ "image", "a.png", "image/png", pic } };
        CHECK_EQ(s.f->srv->router().dispatch(q).status, 200);
        CHECK(std::fabs(s.last().ref_log_bias - std::log(3.0f)) < 1e-6f);
    }
    CHECK_EQ(s.img.got.size(), accepted + 1);   /* nothing refused reached the generator */
    /* Only an edit on a reference-fitting model takes it. */
    CHECK_EQ(str_at(s.gen({ { "prompt", "x" }, { "ref_boost", 2.0 } }, 400)["error"], "param"),
             std::string("ref_boost"));
    s.img.inf.refs_fit_output = false;
    CHECK_EQ(str_at(edit({ { "ref_boost", 2.0 } }, 400)["error"], "param"), std::string("ref_boost"));
    s.img.inf.refs_fit_output = true;

    /* A mask is refused (the first picture is not repainted under it). */
    const json m = edit({ { "mask", data_url(png(100, 100, 0, 255)) } }, 400);
    CHECK_EQ(str_at(m["error"], "param"), std::string("mask"));
    CHECK(has(m["error"].value("message", ""), "instruction edits"));

    /* The reference: a near-matched aspect fills the output grid, another fits inside it on the
     * model's multiple, never above the output. Output is 256 x 128 here. */
    edit(json::object(), 200);
    REQUIRE(s.last().refs.size() == 1);
    CHECK_EQ(s.last().refs[0].w, 128);
    CHECK_EQ(s.last().refs[0].h, 128);
    CHECK_EQ(s.last().refs[0].bf16.size(), (size_t)128 * 128 * s.last().refs[0].ch);
    const std::string wide = png(512, 256, 255, 255);
    edit({ { "image", data_url(wide) } }, 200);
    CHECK_EQ(s.last().refs[0].w, 256);
    CHECK_EQ(s.last().refs[0].h, 128);
    const std::string tall = png(64, 256, 255, 255);
    edit({ { "image", data_url(tall) } }, 200);
    CHECK_EQ(s.last().refs[0].w, 32);
    CHECK_EQ(s.last().refs[0].h, 128);
    /* The grounding picture keeps the original aspect, cut to 768 on its long side. */
    s.img.frames.clear();
    edit({ { "image", data_url(png(1536, 768, 255, 255)) } }, 200);
    REQUIRE(s.img.frames.size() == 1);
    CHECK_EQ(s.img.frames[0].first, 768);
    CHECK_EQ(s.img.frames[0].second, 384);
    s.img.frames.clear();
    edit({ { "image", data_url(wide) } }, 200);
    REQUIRE(s.img.frames.size() == 1);
    CHECK_EQ(s.img.frames[0].first, 512);
    CHECK_EQ(s.img.frames[0].second, 256);
}


TEST(an_admitted_image_waits_for_the_previous_adapter_slot) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    FakeImageGen image;
    int attempts = 0;
    Fixture f(Fixture::Tweak{[&](ServerOptions& o, Deps& d) {
        d.images = &image;
        o.image_model = true;
        o.gen_lora_acquire = [&](const auto&, int32_t* slot, std::shared_ptr<void>* hold,
                                 std::string*, std::string*) {
            if (++attempts == 1) return RAD_E_FULL;
            *slot = 0;
            *hold = std::make_shared<int>(0);
            return RAD_OK;
        };
    }});
    const auto result = f.post("/v1/images/generations",
        R"({"prompt":"a cup","size":"64x64","loras":["turbo"]})");
    CHECK_EQ(result.status, 200);
    CHECK_EQ(attempts, 2);
    CHECK_EQ(f.srv->in_flight(), 0);
}

TEST(disconnecting_the_head_image_waiter_releases_its_turn_and_admission) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    FakeImageGen image;
    std::promise<void> attempted;
    std::atomic<bool> tried{false}, disconnected{false};
    Fixture f(Fixture::Tweak{[&](ServerOptions& o, Deps& d) {
        d.images = &image;
        o.image_model = true;
        o.gen_lora_acquire = [&](const auto& choices, int32_t* slot, std::shared_ptr<void>* hold,
                                 std::string*, std::string*) {
            if (choices.front().first == "busy") {
                if (!tried.exchange(true)) attempted.set_value();
                return RAD_E_FULL;
            }
            *slot = 0;
            *hold = std::make_shared<int>(0);
            return RAD_OK;
        };
    }});
    HttpRequest q;
    q.method = "POST";
    q.path = "/v1/images/generations";
    q.body = R"({"prompt":"a cup","size":"64x64","loras":["busy"]})";
    q.closed = [&] { return disconnected.load(); };
    auto first = std::async(std::launch::async, [&] { return f.srv->router().dispatch(q).status; });
    const auto started = attempted.get_future().wait_for(std::chrono::seconds(3));
    auto second = std::async(std::launch::async, [&] {
        return f.post("/v1/images/generations",
                      R"({"prompt":"a cup","size":"64x64","loras":["next"]})").status;
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (f.srv->in_flight() < 2 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto admitted = f.srv->in_flight();
    disconnected.store(true);
    CHECK(started == std::future_status::ready);
    CHECK_EQ(admitted, 2);
    CHECK_EQ(first.get(), 499);
    CHECK_EQ(second.get(), 200);
    CHECK_EQ(f.srv->in_flight(), 0);
}


/* Admission callbacks are serialized, but the fake generator must also tolerate
 * overlapping HTTP threads after they leave the adapter queue. */
namespace {
struct ConcurrentImageGen : FakeImageGen {
    std::mutex mutex;
    int submit(std::unique_ptr<gen::Request> r, std::string* why) override {
        std::lock_guard lock(mutex);
        return FakeImageGen::submit(std::move(r), why);
    }
};
}

TEST(image_adapter_waiters_acquire_in_fifo_order_even_for_a_cached_choice) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    ConcurrentImageGen image;
    std::atomic<bool> release_head{false}, attempted{false};
    std::atomic<bool> middle_queued{false}, last_queued{false};
    std::promise<void> head_seen, middle_seen, last_seen;
    std::vector<std::string> acquired;
    Fixture f(Fixture::Tweak{[&](ServerOptions& o, Deps& d) {
        d.images = &image;
        o.image_model = true;
        o.gen_lora_acquire = [&](const auto& choices, int32_t* slot, std::shared_ptr<void>* hold,
                                 std::string*, std::string*) {
            const auto& name = choices.front().first;
            if (name == "busy" && !release_head.load()) {
                if (!attempted.exchange(true)) head_seen.set_value();
                return RAD_E_FULL;
            }
            acquired.push_back(name);
            *slot = 0;
            *hold = std::make_shared<int>(0);
            return RAD_OK;
        };
    }});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    auto request = [&](const char* name, std::atomic<bool>* queued, std::promise<void>* seen) {
        HttpRequest q;
        q.method = "POST";
        q.path = "/v1/images/generations";
        q.body = json({{"prompt", "a cup"}, {"size", "64x64"}, {"loras", {name}}}).dump();
        // peer_gone() is checked after insertion, so this is a queue-order barrier.
        q.closed = [&, queued, seen] {
            if (queued && !queued->exchange(true)) seen->set_value();
            return std::chrono::steady_clock::now() >= deadline;
        };
        return f.srv->router().dispatch(q).status;
    };
    auto first = std::async(std::launch::async, [&] { return request("busy", nullptr, nullptr); });
    CHECK(head_seen.get_future().wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    auto middle = std::async(std::launch::async, [&] {
        return request("middle", &middle_queued, &middle_seen);
    });
    CHECK(middle_seen.get_future().wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    auto last = std::async(std::launch::async, [&] {
        return request("cached", &last_queued, &last_seen);
    });
    CHECK(last_seen.get_future().wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    release_head.store(true);
    CHECK_EQ(first.get(), 200);
    CHECK_EQ(middle.get(), 200);
    CHECK_EQ(last.get(), 200);
    CHECK(acquired == std::vector<std::string>({"busy", "middle", "cached"}));
    CHECK_EQ(f.srv->in_flight(), 0);
}

TEST(stopping_the_server_releases_all_image_adapter_waiters) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    ConcurrentImageGen image;
    std::promise<void> attempted;
    std::atomic<bool> tried{false}, second_queued{false}, cancel{false};
    std::promise<void> second_seen;
    Fixture f(Fixture::Tweak{[&](ServerOptions& o, Deps& d) {
        d.images = &image;
        o.image_model = true;
        o.gen_lora_acquire = [&](const auto&, int32_t*, std::shared_ptr<void>*,
                                 std::string*, std::string*) {
            if (!tried.exchange(true)) attempted.set_value();
            return RAD_E_FULL;
        };
    }});
    auto request = [&](bool second) {
        HttpRequest q;
        q.method = "POST";
        q.path = "/v1/images/generations";
        q.body = R"({"prompt":"a cup","size":"64x64","loras":["busy"]})";
        q.closed = [&, second] {
            if (second && !second_queued.exchange(true)) second_seen.set_value();
            return cancel.load();
        };
        return f.srv->router().dispatch(q).status;
    };
    auto first = std::async(std::launch::async, [&] { return request(false); });
    CHECK(attempted.get_future().wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    auto second = std::async(std::launch::async, [&] { return request(true); });
    CHECK(second_seen.get_future().wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    f.srv->stop();
    const auto a = first.wait_for(std::chrono::seconds(2));
    const auto b = second.wait_for(std::chrono::seconds(2));
    cancel.store(true); // Bound cleanup even if a shutdown regression is introduced.
    CHECK(a == std::future_status::ready);
    CHECK(b == std::future_status::ready);
    CHECK_EQ(first.get(), 429);
    CHECK_EQ(second.get(), 429);
    CHECK_EQ(f.srv->in_flight(), 0);
}
TEST(adapter_validation_uses_the_selected_model_without_generating_or_holding_a_slot) {
    int validations = 0;
    Fixture f(Fixture::Tweak{[&](ServerOptions& o, Deps&) {
        o.api_key = "fixture";
        o.gen_lora_validate = [&](const auto& choices, std::string* why) {
            ++validations;
            if (choices.front().first != "turbo") {
                *why = "unmatched adapter";
                return RAD_E_NOTFOUND;
            }
            CHECK_EQ(choices.front().second, 0.6);
            return RAD_OK;
        };
    }});
    auto validate = [&](const char* model, const json& adapters) {
        return f.post("/v1/loras/validate", json{{"model", model}, {"loras", adapters}}.dump());
    };
    CHECK_EQ(validate("missing", {"turbo"}).status, 404);
    CHECK_EQ(validate("test-model", json::array()).status, 400);
    CHECK_EQ(validations, 0);
    const auto answer = validate("test-model", {{"turbo", 0.6}});
    REQUIRE_EQ(answer.status, 200);
    CHECK_EQ(json::parse(answer.body)["compatibility"].get<std::string>(), std::string("compatible"));
    CHECK_EQ(validate("test-model", {"unknown"}).status, 404);
    CHECK_EQ(f.srv->in_flight(), 0);
    CHECK_EQ(validations, 2);
    CHECK_EQ(f.srv->in_flight(), 0);
}

TEST(fleet_validation_targets_the_parsed_model_or_alias_without_waking_a_cold_model) {
    Front front(TransportOptions{}, 0);
    Fixture image(Fixture::Tweak{[&](ServerOptions& o, Deps& d) {
        o.model_id = "image";
        o.api_key = "fixture";
        d.front = &front;
        o.gen_lora_validate = [](const auto&, std::string*) { return RAD_OK; };
    }});
    front.add("cold", {"default"}, 0, false);
    front.attach("image", {"edit"}, image.srv.get());
    auto validate = [&](std::string body) {
        HttpRequest q;
        q.method = "POST";
        q.path = "/v1/loras/validate";
        q.body = std::move(body);
        return front.forward(std::move(q)).status;
    };
    CHECK_EQ(validate(R"({"mod\u0065l":"edit","loras":["style"]})"), 200);
    CHECK_EQ(validate(R"({"model":"cold","model":"image","loras":["style"]})"), 200);
    CHECK_EQ(validate(R"({"model":"cold","loras":["style"]})"), 409);
    CHECK_EQ(validate(R"({"model":"image:missing","loras":["style"]})"), 404);
    CHECK_EQ(front.activity("image").uses, 0);
}


TEST(krea_outfill_scales_the_long_edge_and_preserves_the_source_in_the_canvas) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    Images s;
    s.img.inf.refs_fit_output = true;
    s.img.inf.size_multiple = 16;
    const json request = { {"prompt", "A portrait in a studio"}, {"size", "1024x1024"},
                           {"image", data_url(png(48, 64, 255, 255))},
                           {"outfill", { {"horizontal", "right"} }} };
    const HttpResponse response = s.post("/v1/images/edits", request);
    REQUIRE_EQ(response.status, 200);
    CHECK_EQ(s.last().outfill.x, 256);
    CHECK_EQ(s.last().outfill.y, 0);
    CHECK_EQ(s.last().outfill.w, 768);
    CHECK_EQ(s.last().outfill.h, 1024);
    REQUIRE_EQ(s.last().refs.size(), (size_t)1);
    CHECK_EQ(s.last().refs[0].w, 336);
    CHECK_EQ(s.last().refs[0].h, 448);
    CHECK(s.img.frames.empty());
    const json answer = json::parse(response.body);
    std::vector<uint8_t> png_bytes;
    REQUIRE(base64_decode(answer["data"][0]["b64_json"].get<std::string>(), &png_bytes));
    mm::FrameRGBA result;
    std::string why;
    REQUIRE_EQ(mm::decode_image_rgba(png_bytes.data(), png_bytes.size(), 1024 * 1024, &result, &why),
               RAD_OK);
    const auto red = [&](int x, int y) { return (int)result.rgba[((size_t)y * 1024 + x) * 4]; };
    CHECK_EQ(red(10, 100), 0);
    CHECK_EQ(red(256, 100), 0);
    CHECK_EQ(red(272, 100), 100);
    CHECK_EQ(red(288, 100), 200);
    json other = request;
    other["size"] = "768x768";
    CHECK_EQ(s.post("/v1/images/edits", other).status, 400);
    other = request;
    other["ref_boost"] = 4.0;
    CHECK_EQ(s.post("/v1/images/edits", other).status, 400);
}

/* Only sides that border generated area fade into the model's output; a source side on the canvas
 * boundary keeps its pixels bit-identical to the scaled source. */
TEST(krea_outfill_keeps_source_pixels_on_canvas_edge_sides) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    Images s;
    s.img.inf.refs_fit_output = true;
    s.img.inf.size_multiple = 16;
    auto run = [&](const json& place, std::vector<uint8_t>* rgba) {
        const HttpResponse r = s.post("/v1/images/edits",
            { {"prompt", "x"}, {"size", "1024x1024"}, {"image", data_url(png(48, 64, 255, 255))},
              {"outfill", place} });
        REQUIRE_EQ(r.status, 200);
        std::vector<uint8_t> file;
        REQUIRE(base64_decode(json::parse(r.body)["data"][0]["b64_json"].get<std::string>(), &file));
        mm::FrameRGBA out;
        std::string why;
        REQUIRE_EQ(mm::decode_image_rgba(file.data(), file.size(), 1024 * 1024, &out, &why), RAD_OK);
        *rgba = out.rgba;
    };
    auto red = [](const std::vector<uint8_t>& p, int x, int y) { return (int)p[((size_t)y * 1024 + x) * 4]; };
    std::vector<uint8_t> p;
    /* Placed right (x = 256, 768 x 1024): left is the only generated side. The source's right,
     * top and bottom borders are the source's 200 right up to the canvas edge. */
    run({ {"horizontal", "right"} }, &p);
    for (int y : { 0, 1, 31, 500, 1022, 1023 })
        for (int x : { 1023, 1022, 1000, 992 }) CHECK_EQ(red(p, x, y), 200);
    for (int x : { 288, 300, 600, 1000 })
        for (int y : { 0, 1, 31, 1023 }) CHECK_EQ(red(p, x, y), 200);
    CHECK_EQ(red(p, 256, 500), 0);    /* the left seam still fades from the generated side */
    CHECK_EQ(red(p, 272, 500), 100);
    CHECK_EQ(red(p, 288, 500), 200);
    CHECK_EQ(red(p, 255, 500), 0);    /* outside the source: generated */
    /* Centered: both side seams fade, the top and bottom (canvas edge) rows do not. */
    run({ {"horizontal", "center"} }, &p);
    CHECK_EQ(red(p, 512, 0), 200);
    CHECK_EQ(red(p, 512, 1023), 200);
    CHECK_EQ(red(p, 128, 500), 0);
    CHECK_EQ(red(p, 128 + 767, 500), 0);
    CHECK_EQ(red(p, 128 + 16, 500), 100);
    CHECK_EQ(red(p, 128 + 767 - 16, 500), 100);
    CHECK_EQ(red(p, 128 + 16, 0), 100);   /* the left ramp only: no top ramp on the canvas edge */
}

/* Malformed outfill input is a 400 under the field at fault, before anything is queued. */
TEST(krea_outfill_refuses_malformed_input_before_the_generator) {
    if (!mm::decoder_available()) SKIP("image codec unavailable");
    Images s;
    s.img.inf.refs_fit_output = true;
    s.img.inf.size_multiple = 16;
    const std::string portrait = data_url(png(48, 64, 255, 255));
    auto body = [&](const json& outfill) {
        return json{ {"prompt", "x"}, {"size", "1024x1024"}, {"image", portrait}, {"outfill", outfill} };
    };
    auto refused = [&](json b, const char* param) {
        const HttpResponse r = s.post("/v1/images/edits", b);
        CHECK_EQ(r.status, 400);
        CHECK_EQ(str_at(json::parse(r.body, nullptr, false)["error"], "param"), std::string(param));
    };
    refused(body("left"), "outfill");
    refused(body(json::array({ "left" })), "outfill");
    refused(body(5), "outfill");
    refused(body(nullptr), "outfill");
    refused(body({ {"horizontal", "left"}, {"diagonal", "up"} }), "outfill");
    refused(body({ {"x", 10} }), "outfill");
    refused(body({ {"horizontal", "middle"} }), "outfill");
    refused(body({ {"horizontal", "top"} }), "outfill");
    refused(body({ {"vertical", "left"} }), "outfill");
    refused(body({ {"horizontal", 1} }), "outfill");
    refused(body({ {"horizontal", "RIGHT"} }), "outfill");
    json b = body(json::object());
    b["size"] = "auto";
    refused(b, "size");
    b = body(json::object());
    b["image"] = json::array({ portrait, portrait });
    refused(b, "outfill");
    b = body(json::object());
    b["mask"] = data_url(png(48, 64, 0, 255));
    refused(b, "mask");
    b = body(json::object());
    b["image"] = data_url(png(64, 64, 255, 255));   /* scales to fill the 1024 canvas */
    refused(b, "image");
    b = body(json::object());
    b["ref_boost"] = 2.0;
    refused(b, "outfill");
    /* A plain edit endpoint model with no reference fitting has no outfill at all. */
    s.img.inf.refs_fit_output = false;
    refused(body(json::object()), "outfill");
    /* Well-formed placements of every word are accepted (the generations endpoint takes none). */
    s.img.inf.refs_fit_output = true;
    CHECK(s.img.got.empty());
    for (const char* h : { "left", "center", "right" })
        for (const char* v : { "top", "center", "bottom" })
            CHECK_EQ(s.post("/v1/images/edits", body({ {"horizontal", h}, {"vertical", v} })).status, 200);
    CHECK_EQ(s.img.got.size(), (size_t)9);
    CHECK_EQ(s.post("/v1/images/edits", body(json::object())).status, 200);
    CHECK_EQ(s.last().outfill.x, 0);
    CHECK_EQ(s.last().outfill.y, 0);
    CHECK_EQ(str_at(s.gen({ {"prompt", "x"}, {"outfill", json::object()} }, 400)["error"], "param"),
             std::string("outfill"));
}

RAD_TEST_MAIN()
