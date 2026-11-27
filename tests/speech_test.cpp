/* tests/speech_test.cpp -- OpenAI's speech API over an audio generator (core/server/audio_gen.cpp).
 *
 * The generator is faked (server_fakes.h: FakeAudioGen finishes every request on submit, its
 * windows out of order), so these pin what the server owns: which requests are refused and under
 * which field, the request it hands the generator (the prompt rendered from the template, the
 * length in frames, a seed a JavaScript client can read back), the samples as 16-bit PCM in a WAV
 * or raw, the windows put back in order, failures mapped to 400 or 500, and the two streamed forms
 * carrying the same samples as the whole response.
 */
#include "server_fakes.h"

#include <cstring>

namespace {

struct Speech {
    FakeAudioGen gen;
    std::unique_ptr<Fixture> f;
    Speech() {
        f.reset(new Fixture(Fixture::Tweak{ [this](ServerOptions& o, Deps& d) {
            d.audio = &gen;
            o.audio_model = true;
        } }));
    }
    static json body(json extra = json::object()) {
        json b = { { "input", "la la" }, { "instructions", "pop" } };
        for (auto it = extra.begin(); it != extra.end(); ++it) b[it.key()] = it.value();
        return b;
    }
    HttpResponse post(const json& b) { return f->post("/v1/audio/speech", b.dump()); }
    /* A refusal's status and the field it names. */
    std::string refused(const json& b, int status = 400) {
        HttpResponse r = post(b);
        CHECK_EQ(r.status, status);
        return str_at(json::parse(r.body, nullptr, false)["error"], "param");
    }
    const gen::AudioRequest& last() const { return *gen.got.back(); }
};

std::string header(const HttpResponse& r, const char* k) {
    for (const auto& [n, v] : r.headers)
        if (n == k) return v;
    return std::string();
}

uint32_t le32(const std::string& s, size_t at) {
    uint32_t v = 0;
    std::memcpy(&v, s.data() + at, 4);
    return v;
}

/* The PCM the fake's windows make, in window order: each sample clipped to [-1, 1], times 32767,
 * rounded to the nearest. */
std::string expected_pcm(const FakeAudioGen& g) {
    std::string out;
    for (int32_t k = 0; k < (int32_t)g.order.size(); ++k)
        for (int32_t j = 0; j < g.per_window; ++j) {
            const float v = std::min(1.0f, std::max(-1.0f, FakeAudioGen::sample(k, j)));
            const int16_t q = (int16_t)std::lrint(v * 32767.0f);
            out.append((const char*)&q, 2);
        }
    return out;
}

}  /* namespace */

/* ================================================================== refusals */

/* A request this deployment cannot serve is a 400 naming its field before anything is queued: an
 * unknown field, missing or blank lyrics or description, a format other than wav or pcm, a speed,
 * malformed stream fields, the two stream forms at once, events in a WAV, both lengths at once or
 * either out of range, steps or guidance out of range, a non-integer seed, a prompt over the
 * model's limit, and text the plugin refuses. */
TEST(a_speech_request_that_cannot_be_served_names_its_field) {
    Speech s;
    CHECK_EQ(s.refused(Speech::body({ { "temperature", 1 } })), std::string("temperature"));
    CHECK_EQ(s.refused({ { "instructions", "pop" } }), std::string("input"));
    CHECK_EQ(s.refused(Speech::body({ { "input", "  \n" } })), std::string("input"));
    CHECK_EQ(s.refused({ { "input", "la" } }), std::string("instructions"));
    CHECK_EQ(s.refused(Speech::body({ { "response_format", "mp3" } })),
             std::string("response_format"));
    CHECK_EQ(s.refused(Speech::body({ { "speed", 2.0 } })), std::string("speed"));
    CHECK_EQ(s.refused(Speech::body({ { "stream", "yes" } })), std::string("stream"));
    CHECK_EQ(s.refused(Speech::body({ { "stream_format", "ogg" } })), std::string("stream_format"));
    CHECK_EQ(s.refused(Speech::body({ { "stream", true }, { "stream_format", "audio" } })),
             std::string("stream"));
    CHECK_EQ(s.refused(Speech::body({ { "stream_format", "sse" }, { "response_format", "wav" } })),
             std::string("response_format"));
    CHECK_EQ(s.refused(Speech::body({ { "max_new_tokens", 10 }, { "duration", 1.0 } })),
             std::string("duration"));
    CHECK_EQ(s.refused(Speech::body({ { "max_new_tokens", 101 } })), std::string("max_new_tokens"));
    CHECK_EQ(s.refused(Speech::body({ { "max_new_tokens", 0 } })), std::string("max_new_tokens"));
    CHECK_EQ(s.refused(Speech::body({ { "duration", 4.5 } })), std::string("duration"));
    CHECK_EQ(s.refused(Speech::body({ { "num_inference_steps", 0 } })),
             std::string("num_inference_steps"));
    CHECK_EQ(s.refused(Speech::body({ { "guidance_scale", 0.5 } })), std::string("guidance_scale"));
    CHECK_EQ(s.refused(Speech::body({ { "seed", "7" } })), std::string("seed"));
    CHECK_EQ(s.refused(Speech::body({ { "input", std::string(80, 'a') } })), std::string("input"));
    CHECK_EQ(s.refused(Speech::body({ { "input", "REFUSE" } })), std::string("input"));
    CHECK_EQ(s.refused(Speech::body({ { "instructions", "REFUSE" } })),
             std::string("instructions"));
    CHECK(s.gen.got.empty());
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);
}

/* A model that makes no audio answers the route with a 501. */
TEST(a_text_model_answers_the_speech_route_with_501) {
    Fixture f;
    CHECK_EQ(f.post("/v1/audio/speech", Speech::body().dump()).status, 501);
}

/* ================================================================== the request */

/* The prompt is the template with each slot's text in it; the length is max_new_tokens frames, or
 * a duration rounded up to whole frames, or the model's longest; an absent or -1 seed is chosen
 * below 2^53 and read back in X-Seed. */
TEST(the_generator_gets_the_rendered_prompt_the_length_and_a_readable_seed) {
    Speech s;
    HttpResponse r = s.post(Speech::body({ { "max_new_tokens", 40 }, { "seed", 9 },
                                           { "num_inference_steps", 12 },
                                           { "guidance_scale", 2.5 } }));
    CHECK_EQ(r.status, 200);
    const std::string want = "<s>pop|la la";
    REQUIRE(s.last().tokens.size() == want.size());
    for (size_t i = 0; i < want.size(); ++i)
        CHECK_EQ(s.last().tokens[i], (int32_t)(uint8_t)want[i]);
    CHECK_EQ(s.last().max_frames, (int64_t)40);
    CHECK_EQ(s.last().steps, 12);
    CHECK(std::fabs(s.last().cfg - 2.5f) < 1e-6f);
    CHECK_EQ(s.last().seed, (uint64_t)9);
    CHECK_EQ(header(r, "X-Seed"), std::string("9"));

    CHECK_EQ(s.post(Speech::body({ { "duration", 1.01 } })).status, 200);
    CHECK_EQ(s.last().max_frames, (int64_t)26);
    CHECK_EQ(s.last().steps, 0);
    CHECK_EQ(s.post(Speech::body()).status, 200);
    CHECK_EQ(s.last().max_frames, (int64_t)100);
    for (const json& seed : { json(nullptr), json(-1) }) {
        HttpResponse q = s.post(Speech::body({ { "seed", seed } }));
        CHECK_EQ(q.status, 200);
        CHECK(s.last().seed < ((uint64_t)1 << 53));
        CHECK_EQ(header(q, "X-Seed"), std::to_string(s.last().seed));
    }
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);
}

/* A speech model declares a prompt without instructions and the unconditional branch's own
 * template: instructions become optional and pick the template, the unconditional prompt is
 * rendered from the same texts, and the denoiser's step count is refused (it has none). */
TEST(a_speech_model_takes_optional_instructions_and_renders_its_unconditional_prompt) {
    Speech s;
    s.gen.inf.tmpl_plain = "<s>{input}";
    s.gen.inf.neg_tmpl = "<n>{input}";
    s.gen.inf.default_steps = 0;
    auto text = [](const std::vector<int32_t>& t) {
        std::string r;
        for (int32_t c : t) r += (char)c;
        return r;
    };
    CHECK_EQ(s.post({ { "input", "hi" } }).status, 200);
    CHECK_EQ(text(s.last().tokens), std::string("<s>hi"));
    CHECK_EQ(text(s.last().neg_tokens), std::string("<n>hi"));
    CHECK_EQ(s.post({ { "input", "hi" }, { "instructions", " " } }).status, 200);
    CHECK_EQ(text(s.last().tokens), std::string("<s>hi"));
    CHECK_EQ(s.post({ { "input", "hi" }, { "instructions", "calm" } }).status, 200);
    CHECK_EQ(text(s.last().tokens), std::string("<s>calm|hi"));
    CHECK_EQ(text(s.last().neg_tokens), std::string("<n>hi"));
    CHECK_EQ(s.refused({ { "input", "hi" }, { "num_inference_steps", 4 } }),
             std::string("num_inference_steps"));
    CHECK_EQ(s.refused({ { "input", "hi" }, { "instructions", 3 } }), std::string("instructions"));
    /* A music model's prompt has no unconditional template of its own. */
    Speech m;
    CHECK_EQ(m.post(Speech::body()).status, 200);
    CHECK(m.last().neg_tokens.empty());
}

/* A model that declares input_special (its speaker tags are special tokens) has its input
 * tokenised with them, and one of one branch refuses any guidance but 1 under its field; other
 * models' input stays plain text. */
TEST(a_model_that_asks_has_its_input_read_with_special_tokens_and_one_branch_has_no_guidance) {
    Speech s;
    s.gen.inf.tmpl_plain = "<s>{input}";
    s.gen.inf.default_steps = 0;
    s.gen.inf.input_special = true;
    s.gen.inf.branches = 1;
    s.f->tok.last_parse_special = false;
    CHECK_EQ(s.post({ { "input", "hi" } }).status, 200);
    CHECK(s.f->tok.last_parse_special);
    CHECK_EQ(s.refused({ { "input", "hi" }, { "guidance_scale", 2.0 } }),
             std::string("guidance_scale"));
    CHECK_EQ(s.post({ { "input", "hi" }, { "guidance_scale", 1.0 } }).status, 200);
    /* A model that declares no template with instructions refuses them by name. */
    s.gen.inf.tmpl.clear();
    CHECK_EQ(s.refused({ { "input", "hi" }, { "instructions", "calm" } }),
             std::string("instructions"));
    CHECK_EQ(s.post({ { "input", "hi" } }).status, 200);
    Speech m;
    m.gen.inf.tmpl_plain = "<s>{input}";
    m.gen.inf.default_steps = 0;
    CHECK_EQ(m.post({ { "input", "hi" } }).status, 200);
    CHECK(!m.f->tok.last_parse_special);
}

/* A 16-bit mono WAV of the given samples at `rate`. */
static std::string wav_of(const std::vector<int16_t>& x, uint32_t rate) {
    std::string d;
    for (int16_t v : x) {
        d.push_back((char)(v & 0xff));
        d.push_back((char)((v >> 8) & 0xff));
    }
    auto u32 = [](uint32_t v) {
        std::string b;
        for (int i = 0; i < 4; ++i) b.push_back((char)((v >> (8 * i)) & 0xff));
        return b;
    };
    auto u16 = [](uint16_t v) { return std::string{ (char)(v & 0xff), (char)(v >> 8) }; };
    return "RIFF" + u32((uint32_t)(36 + d.size())) + "WAVEfmt " + u32(16) + u16(1) + u16(1) +
           u32(rate) + u32(rate * 2) + u16(2) + u16(16) + "data" + u32((uint32_t)d.size()) + d;
}

/* A model that clones a voice takes references: each clip decoded at its rate and padded to
 * whole frames, the prompt its references' template with each one's header, transcript and a row
 * a frame (token 0, the clips' frames numbered in order), the rest of the rows -1. A data: URL and
 * bare base64 both carry a clip. Refused by name: references on a model without them, too many,
 * a reference without its text or clip or with an unknown field, a clip past the budget, and
 * instructions beside them. */
TEST(a_cloning_model_takes_references_as_clips_and_transcripts_in_its_prompt) {
    if (!mm::decoder_available()) SKIP("audio decoder unavailable");
    Speech s;
    s.gen.inf.tmpl_plain = "<s>{input}";
    s.gen.inf.tmpl_ref = "<s>R{references}|{input}";
    s.gen.inf.ref_item = "[{index}:{text}]{audio}.";
    s.gen.inf.ref_rate = 8000;
    s.gen.inf.ref_hop = 4;
    s.gen.inf.ref_max_samples = 16;
    s.gen.inf.ref_max = 2;
    s.gen.inf.default_steps = 0;
    s.gen.inf.branches = 1;
    const std::string w1 = wav_of({ 1000, -2000, 3000, 4000, 5000 }, 8000);
    const std::string w2 = wav_of({ -16384, 16384 }, 8000);
    const std::string b1 = base64_encode((const uint8_t*)w1.data(), w1.size());
    const std::string b2 = base64_encode((const uint8_t*)w2.data(), w2.size());
    json refs = json::array({ json{ { "audio", b1 }, { "text", "one" } },
                              json{ { "audio", "data:audio/wav;base64," + b2 },
                                    { "text", "two" } } });
    CHECK_EQ(s.post({ { "input", "hi" }, { "references", refs } }).status, 200);
    const gen::AudioRequest& q = s.last();
    /* "<s>R", "[1:one]", 2 frames, ".", "[2:two]", 1 frame, ".", "|hi". */
    const std::string head = "<s>R[1:one]";
    std::vector<int32_t> want_tok, want_frame;
    auto text = [&](const std::string& t) {
        for (unsigned char c : t) { want_tok.push_back(c); want_frame.push_back(-1); }
    };
    auto frames = [&](int32_t f0, int32_t n) {
        for (int32_t f = 0; f < n; ++f) { want_tok.push_back(0); want_frame.push_back(f0 + f); }
    };
    text(head);
    frames(0, 2);
    text(".[2:two]");
    frames(2, 1);
    text(".|hi");
    CHECK(q.tokens == want_tok);
    CHECK(q.ref_frame == want_frame);
    REQUIRE(q.ref_cu.size() == 3);
    CHECK_EQ(q.ref_cu[1], 8);
    CHECK_EQ(q.ref_cu[2], 12);
    REQUIRE(q.ref_audio.size() == 12);
    CHECK(std::fabs(q.ref_audio[0] - 1000.0f / 32768.0f) < 1e-6f);
    CHECK(std::fabs(q.ref_audio[1] + 2000.0f / 32768.0f) < 1e-6f);
    CHECK_EQ(q.ref_audio[5], 0.0f);
    CHECK_EQ(q.ref_audio[7], 0.0f);
    CHECK(std::fabs(q.ref_audio[8] + 0.5f) < 1e-6f);
    CHECK_EQ(q.ref_audio[11], 0.0f);
    /* Without references the plain prompt, and no clips. */
    CHECK_EQ(s.post({ { "input", "hi" }, { "references", json::array() } }).status, 200);
    CHECK(s.last().ref_cu.empty() && s.last().ref_frame.empty() && s.last().ref_audio.empty());

    const json one = json{ { "audio", b1 }, { "text", "one" } };
    CHECK_EQ(s.refused({ { "input", "hi" }, { "references", json::array({ one, one, one }) } }),
             std::string("references"));
    CHECK_EQ(s.refused({ { "input", "hi" }, { "references", one } }), std::string("references"));
    CHECK_EQ(s.refused({ { "input", "hi" },
                         { "references", json::array({ json{ { "audio", b1 } } }) } }),
             std::string("references[0].text"));
    CHECK_EQ(s.refused({ { "input", "hi" },
                         { "references", json::array({ json{ { "text", "x" } } }) } }),
             std::string("references[0].audio"));
    CHECK_EQ(s.refused({ { "input", "hi" },
                         { "references", json::array({ json{ { "audio", b1 }, { "text", "x" },
                                                             { "speaker", 1 } } }) } }),
             std::string("references[0].speaker"));
    CHECK_EQ(s.refused({ { "input", "hi" },
                         { "references", json::array({ json{ { "audio", "@@@" },
                                                             { "text", "x" } } }) } }),
             std::string("references[0].audio"));
    /* 8 and 8 samples fit the 16; a third frame past them does not. */
    const std::string w3 = wav_of(std::vector<int16_t>(9, 7), 8000);
    const std::string b3 = base64_encode((const uint8_t*)w3.data(), w3.size());
    CHECK_EQ(s.refused({ { "input", "hi" },
                         { "references", json::array({ one, json{ { "audio", b3 },
                                                                  { "text", "x" } } }) } }),
             std::string("references[1].audio"));
    CHECK_EQ(s.refused({ { "input", "hi" }, { "instructions", "calm" }, { "references", refs } }),
             std::string("instructions"));
    CHECK_EQ(s.refused({ { "input", "hi" },
                         { "references", json::array({ json{ { "audio", b1 },
                                                             { "text", "REFUSE" } } }) } }),
             std::string("references[0].text"));
    Speech m;
    CHECK_EQ(m.refused(Speech::body({ { "references", refs } })), std::string("references"));
}

/* ================================================================== the response */
TEST(breeze_cloning_uses_one_reference_and_direction_is_opt_in) {
    if (!mm::decoder_available()) SKIP("audio decoder unavailable");
    Speech s;
    auto& info = s.gen.inf;
    info.max_prompt = 512;
    info.tmpl_plain = "<bos>[S0]{input}";
    info.tmpl = "<bos>[S0]<ins_bos>{instructions}<ins_eos>{input}";
    info.neg_tmpl = info.tmpl_plain;
    info.tmpl_ref = "{references}<bos>[S0]{input}";
    info.ref_item = "<bos>[S0]{text}{audio}<|audio_eos|>";
    info.ref_rate = 24000;
    info.ref_hop = 1920;
    info.ref_max_samples = 240000;
    info.ref_max = 1;
    info.branches = 2;
    info.default_steps = 0;
    const std::string wav = wav_of(std::vector<int16_t>(2000, 1000), 24000);
    const std::string audio = base64_encode((const uint8_t*)wav.data(), wav.size());
    const json ref = {{"audio", audio}, {"text", "reference words"}};
    json body = {{"input", "new words"}, {"references", json::array({ref})}, {"guidance_scale", 1}};
    CHECK_EQ(s.post(body).status, 200);
    const auto& q = s.last();
    CHECK_EQ(q.ref_audio.size(), (size_t)3840);
    CHECK_EQ(q.ref_cu.size(), (size_t)2);
    CHECK_EQ(std::count_if(q.ref_frame.begin(), q.ref_frame.end(), [](int v) { return v >= 0; }), 2);
    CHECK_EQ(q.ref_frame.size(), q.tokens.size());
    CHECK_EQ(q.cfg, 1.0f);
    body["guidance_scale"] = 4;
    CHECK_EQ(s.refused(body), std::string("guidance_scale"));
    body["guidance_scale"] = 1;
    body["instructions"] = "a different voice";
    CHECK_EQ(s.refused(body), std::string("instructions"));
    info.tmpl_ref_instruct = "{references}<bos>[S0]<ins_bos>{instructions}<ins_eos>{input}";
    info.neg_tmpl_ref = info.tmpl_ref;
    body["guidance_scale"] = 4;
    CHECK_EQ(s.post(body).status, 200);
    const auto& directed = s.last();
    CHECK_EQ(directed.cfg, 4.0f);
    CHECK_EQ(directed.neg_ref_frame.size(), directed.neg_tokens.size());
    CHECK_EQ(std::count(directed.ref_frame.begin(), directed.ref_frame.end(), 0), 1);
    CHECK_EQ(std::count(directed.neg_ref_frame.begin(), directed.neg_ref_frame.end(), 0), 1);
    CHECK(directed.tokens.size() > directed.neg_tokens.size());
    const auto text_of = [](const std::vector<int32_t>& ids) {
        std::string text; for (int32_t i : ids) if (i) text.push_back((char)i); return text;
    };
    CHECK(has(text_of(directed.tokens), "<ins_bos>a different voice<ins_eos>new words"));
    CHECK(!has(text_of(directed.neg_tokens), "a different voice"));
    CHECK(has(text_of(directed.neg_tokens), "reference words"));
    body.erase("instructions");
    body["references"].push_back(ref);
    CHECK_EQ(s.refused(body), std::string("references"));
    body["references"] = json::array({{{"audio", audio}, {"text", " "}}});
    CHECK_EQ(s.refused(body), std::string("references[0].text"));
}

/* The whole piece: the windows in order whatever order they were played in, as a WAV whose header
 * states the rate, channels and length, or as the same samples raw. */
TEST(the_whole_piece_is_a_wav_or_raw_pcm_in_window_order) {
    Speech s;
    const std::string pcm = expected_pcm(s.gen);
    HttpResponse r = s.post(Speech::body());
    CHECK_EQ(r.status, 200);
    CHECK_EQ(r.content_type, std::string("audio/wav"));
    REQUIRE(r.body.size() == 44 + pcm.size());
    CHECK_EQ(r.body.substr(0, 4), std::string("RIFF"));
    CHECK_EQ(le32(r.body, 4), (uint32_t)(36 + pcm.size()));
    CHECK_EQ(le32(r.body, 24), (uint32_t)8000);
    CHECK_EQ((int)(uint8_t)r.body[22], 2);
    CHECK_EQ(le32(r.body, 40), (uint32_t)pcm.size());
    CHECK(r.body.substr(44) == pcm);
    CHECK_EQ(header(r, "X-Audio-Sample-Rate"), std::string("8000"));
    CHECK_EQ(header(r, "X-Audio-Channels"), std::string("2"));

    HttpResponse p = s.post(Speech::body({ { "response_format", "pcm" } }));
    CHECK_EQ(p.content_type, std::string("audio/pcm"));
    CHECK(p.body == pcm);
}

/* stream_format audio sends a WAV header that states no length, then the samples; the bytes after
 * it are the whole response's. */
TEST(a_raw_stream_carries_the_whole_responses_samples) {
    Speech s;
    const std::string pcm = expected_pcm(s.gen);
    HttpResponse r = s.post(Speech::body({ { "stream_format", "audio" } }));
    CHECK_EQ(r.status, 200);
    REQUIRE(r.streaming());
    CHECK_EQ(r.content_type, std::string("audio/wav"));
    const std::string all = drain(r);
    REQUIRE(all.size() == 44 + pcm.size());
    CHECK_EQ(le32(all, 4), 0xffffffffu);
    CHECK_EQ(le32(all, 40), 0xffffffffu);
    CHECK(all.substr(44) == pcm);
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);

    HttpResponse p = s.post(Speech::body({ { "stream_format", "audio" },
                                           { "response_format", "pcm" } }));
    CHECK(drain(p) == pcm);
}

/* stream true (or stream_format sse) sends speech.audio.delta events whose PCM, joined, is the
 * whole response's, then speech.audio.done with the prompt tokens in and the frames out. */
TEST(an_event_stream_carries_the_samples_then_the_usage) {
    Speech s;
    const std::string pcm = expected_pcm(s.gen);
    for (const json& form : { json{ { "stream", true } }, json{ { "stream_format", "sse" } } }) {
        json b = Speech::body(form);
        b["max_new_tokens"] = 30;
        HttpResponse r = s.post(b);
        CHECK_EQ(r.status, 200);
        CHECK_EQ(r.content_type, std::string("text/event-stream"));
        const std::vector<json> ev = sse_events(drain(r));
        REQUIRE(ev.size() >= 2);
        std::string got;
        for (size_t i = 0; i + 1 < ev.size(); ++i) {
            CHECK_EQ(ev[i].value("type", ""), std::string("speech.audio.delta"));
            CHECK_EQ(ev[i].value("response_format", ""), std::string("pcm"));
            CHECK_EQ(ev[i].value("sample_rate", 0), 8000);
            CHECK_EQ(ev[i].value("channels", 0), 2);
            std::vector<uint8_t> b;
            CHECK(base64_decode(ev[i].value("audio", ""), &b));
            got.append(b.begin(), b.end());
        }
        CHECK(got == pcm);
        CHECK_EQ(ev.back().value("type", ""), std::string("speech.audio.done"));
        const json& u = ev.back()["usage"];
        CHECK_EQ(u.value("input_tokens", 0), (int)std::strlen("<s>pop|la la"));
        CHECK_EQ(u.value("output_tokens", 0), 30);
        CHECK_EQ(u.value("total_tokens", 0), (int)std::strlen("<s>pop|la la") + 30);
    }
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);
}

/* A refusal at submit is a 400 (the request can never run) or a 503; a failure while running is a
 * 400 or a 500 with the generator's reason, and on an event stream a speech.audio.error of the
 * same type. */
TEST(a_generators_failure_is_a_400_or_a_500_streamed_or_not) {
    Speech s;
    s.gen.submit_status = RAD_E_INVAL;
    s.gen.submit_why = "too long for the cache";
    HttpResponse r = s.post(Speech::body());
    CHECK_EQ(r.status, 400);
    CHECK(has(r.body, "too long"));
    s.gen.submit_status = RAD_E_IO;
    CHECK_EQ(s.post(Speech::body()).status, 503);
    s.gen.submit_status = RAD_OK;
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);

    for (int st : { RAD_E_INVAL, RAD_E_IO }) {
        s.gen.finish_status = st;
        s.gen.finish_why = "the vocoder ran out of memory";
        HttpResponse q = s.post(Speech::body());
        CHECK_EQ(q.status, st == RAD_E_INVAL ? 400 : 500);
        const json e = json::parse(q.body, nullptr, false)["error"];
        CHECK_EQ(e.value("message", ""), s.gen.finish_why);
        const std::string type = st == RAD_E_INVAL ? "invalid_request_error" : "server_error";
        CHECK_EQ(e.value("type", ""), type);
        HttpResponse t = s.post(Speech::body({ { "stream", true } }));
        CHECK_EQ(t.status, 200);
        const std::vector<json> ev = sse_events(drain(t));
        REQUIRE(ev.size() == 1);
        CHECK_EQ(ev[0].value("type", ""), std::string("speech.audio.error"));
        CHECK_EQ(ev[0]["error"].value("type", ""), type);
        CHECK_EQ(ev[0]["error"].value("message", ""), s.gen.finish_why);
    }
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);
}

RAD_TEST_MAIN()
