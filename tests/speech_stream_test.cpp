/* tests/speech_stream_test.cpp -- vLLM-Omni's realtime speech session over a WebSocket
 * (core/server/speech_stream.cpp), against server_fakes.h's audio generator.
 *
 * These pin what the server owns: that text sent in pieces is spoken a sentence at a time in
 * order, each sentence's events around its audio, the audio the same bytes the POST route gives
 * for the same request, sent whole or as it is made and as PCM or WAV, the utterance's end, the
 * refusals that leave the session going, and a close that cancels the sentence being spoken.
 */
#include "server_fakes.h"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <thread>

namespace {

/* A WebSocket the test plays the client of: what the session sends, text and binary, in order. */
struct Conn : WsConn {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> in;
    bool hung_up = false;
    bool closed = false;
    int  close_code = 0;
    std::vector<std::pair<std::string, bool>> out;   /* (message, binary) */

    bool read(std::string* msg, bool* binary) override {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !in.empty() || hung_up || closed; });
        if (in.empty()) return false;
        *msg = std::move(in.front());
        *binary = false;
        in.pop_front();
        return true;
    }
    bool send(const std::string& t) override { return put(t, false); }
    bool send_binary(const std::string& b) override { return put(b, true); }
    bool put(const std::string& t, bool binary) {
        std::lock_guard<std::mutex> lk(m);
        out.emplace_back(t, binary);
        cv.notify_all();
        return !closed;
    }
    void close(int code, const std::string&) override {
        std::lock_guard<std::mutex> lk(m);
        closed = true;
        close_code = code;
        cv.notify_all();
    }
    void client(const json& j) {
        std::lock_guard<std::mutex> lk(m);
        in.push_back(j.dump());
        cv.notify_all();
    }
    void hangup() {
        std::lock_guard<std::mutex> lk(m);
        hung_up = true;
        cv.notify_all();
    }
    /* The `nth` event of `type` (from 0), waiting for it; null when it does not come. */
    json wait_for(const std::string& type, size_t nth = 0, int timeout_ms = 10000) {
        std::unique_lock<std::mutex> lk(m);
        const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            size_t seen = 0;
            for (const auto& o : out) {
                if (o.second) continue;
                json j = json::parse(o.first);
                if (j.value("type", "") == type && seen++ == nth) return j;
            }
            if (cv.wait_until(lk, until) == std::cv_status::timeout) return json();
        }
    }
    bool wait_closed(int timeout_ms = 10000) {
        std::unique_lock<std::mutex> lk(m);
        return cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return closed; });
    }
    /* Every message so far: an event's type, or "#<bytes>" for a binary one. */
    std::vector<std::string> trace() {
        std::lock_guard<std::mutex> lk(m);
        std::vector<std::string> t;
        for (const auto& o : out)
            t.push_back(o.second ? "#" + std::to_string(o.first.size())
                                 : json::parse(o.first).value("type", ""));
        return t;
    }
    /* The binary messages between sentence `k`'s audio.start and audio.done, in order. */
    std::vector<std::string> frames(int64_t k) {
        std::lock_guard<std::mutex> lk(m);
        std::vector<std::string> f;
        bool in_k = false;
        for (const auto& o : out) {
            if (o.second) {
                if (in_k) f.push_back(o.first);
                continue;
            }
            const json j = json::parse(o.first);
            if (j.value("type", "") == "audio.start") in_k = j.value("sentence_index", -1) == k;
            if (j.value("type", "") == "audio.done") in_k = false;
        }
        return f;
    }
};

/* A server over the fake generator with one session connected to it. */
struct Stream {
    FakeAudioGen gen;
    std::unique_ptr<Fixture> f;
    Conn conn;
    std::thread route;
    Stream() {
        f.reset(new Fixture(Fixture::Tweak{ [this](ServerOptions& o, Deps& d) {
            d.audio = &gen;
            o.audio_model = true;
        } }));
        const SocketHandler* h = f->srv->router().socket_for("/v1/audio/speech/stream");
        REQUIRE(h != nullptr);
        route = std::thread([this, h] {
            HttpRequest q;
            q.method = "GET";
            q.path = "/v1/audio/speech/stream";
            (*h)(q, conn);
        });
    }
    ~Stream() {
        conn.hangup();
        if (route.joinable()) route.join();
    }
    void config(json extra = json::object()) {
        json c = { { "type", "session.config" }, { "instructions", "calm" } };
        for (auto it = extra.begin(); it != extra.end(); ++it) c[it.key()] = it.value();
        conn.client(c);
    }
    void text(const std::string& t) { conn.client({ { "type", "input.text" }, { "text", t } }); }
    void done() { conn.client({ { "type", "input.done" } }); }
    /* The POST route's audio for the same request: the fake's samples, whatever the text. */
    std::string post_pcm() {
        json b = { { "input", "x" }, { "instructions", "calm" }, { "response_format", "pcm" } };
        HttpResponse r = f->post("/v1/audio/speech", b.dump());
        CHECK_EQ(r.status, 200);
        return r.body;
    }
};

std::string joined(const std::vector<std::string>& v) {
    std::string s;
    for (const auto& x : v) s += x;
    return s;
}

uint32_t le32(const std::string& s, size_t at) {
    uint32_t v = 0;
    std::memcpy(&v, s.data() + at, 4);
    return v;
}

}  /* namespace */

/* Text sent in pieces is spoken a sentence at a time, in order: each sentence an audio.start
 * naming it, its audio in one binary message (the bytes the POST route gives for the same
 * request), and an audio.done counting them; the utterance ends with session.done once its last
 * sentence, the text input.done flushed, is spoken. Each sentence is its own request, rendered
 * from the session's instructions. The session then takes the next utterance. */
TEST(text_in_pieces_is_spoken_a_sentence_at_a_time) {
    Stream s;
    const std::string pcm = s.post_pcm();
    s.config();
    s.text("Hello wor");
    s.text("ld. How are ");
    s.text("you? Fine");
    s.done();
    const json end = s.conn.wait_for("session.done");
    REQUIRE(!end.is_null());
    CHECK_EQ(end.value("utterance_index", -1), 0);
    CHECK_EQ(end.value("total_sentences", -1), 3);
    const std::string n = "#" + std::to_string(pcm.size());
    CHECK(s.conn.trace() == std::vector<std::string>({ "audio.start", n, "audio.done",
                                                       "audio.start", n, "audio.done",
                                                       "audio.start", n, "audio.done",
                                                       "session.done" }));
    const char* want[] = { "Hello world.", "How are you?", "Fine" };
    for (int k = 0; k < 3; ++k) {
        const json st = s.conn.wait_for("audio.start", (size_t)k);
        CHECK_EQ(st.value("sentence_index", -1), k);
        CHECK_EQ(st.value("sentence_text", ""), std::string(want[k]));
        CHECK_EQ(st.value("format", ""), std::string("pcm"));
        CHECK_EQ(st.value("sample_rate", 0), 8000);
        CHECK_EQ(st.value("channels", 0), 2);
        const json dn = s.conn.wait_for("audio.done", (size_t)k);
        CHECK_EQ(dn.value("total_bytes", -1), (int)pcm.size());
        CHECK_EQ(dn.value("error", true), false);
        CHECK(joined(s.conn.frames(k)) == pcm);
    }
    /* The POST request was the first; the sentences are the three after it. */
    REQUIRE_EQ(s.gen.n_got(), (size_t)4);
    CHECK_EQ(s.gen.prompt_of(1), std::string("<s>calm|Hello world."));
    CHECK_EQ(s.gen.prompt_of(2), std::string("<s>calm|How are you?"));
    CHECK_EQ(s.gen.prompt_of(3), std::string("<s>calm|Fine"));

    s.text("Again.");
    s.done();
    const json next = s.conn.wait_for("session.done", 1);
    REQUIRE(!next.is_null());
    CHECK_EQ(next.value("utterance_index", -1), 1);
    CHECK_EQ(next.value("total_sentences", -1), 1);
    CHECK_EQ(s.conn.wait_for("audio.start", 3).value("sentence_index", -1), 0);
    CHECK(!s.conn.closed);
}

/* With stream_audio the audio goes out as the generator makes it, in order however its pieces
 * arrive: here windows 1, 0, 2 become two messages, the first two windows and the third, which
 * together are the whole response's bytes. A WAV says its length in one message, and while it
 * streams opens with a header that states none. */
TEST(audio_is_sent_whole_or_as_it_is_made_as_pcm_or_wav) {
    {
        Stream s;
        const std::string pcm = s.post_pcm();
        s.config({ { "stream_audio", true } });
        s.text("One.");
        s.done();
        REQUIRE(!s.conn.wait_for("session.done").is_null());
        const std::vector<std::string> f = s.conn.frames(0);
        CHECK_EQ(f.size(), (size_t)2);
        CHECK(joined(f) == pcm);
        CHECK_EQ(s.conn.wait_for("audio.done").value("total_bytes", -1), (int)pcm.size());
    }
    {
        Stream s;
        const std::string pcm = s.post_pcm();
        s.config({ { "response_format", "wav" } });
        s.text("One.");
        s.done();
        REQUIRE(!s.conn.wait_for("session.done").is_null());
        CHECK_EQ(s.conn.wait_for("audio.start").value("format", ""), std::string("wav"));
        const std::vector<std::string> f = s.conn.frames(0);
        REQUIRE_EQ(f.size(), (size_t)1);
        REQUIRE(f[0].size() == 44 + pcm.size());
        CHECK_EQ(f[0].substr(0, 4), std::string("RIFF"));
        CHECK_EQ(le32(f[0], 40), (uint32_t)pcm.size());
        CHECK(f[0].substr(44) == pcm);
    }
    {
        Stream s;
        const std::string pcm = s.post_pcm();
        s.config({ { "response_format", "wav" }, { "stream_audio", true } });
        s.text("One.");
        s.done();
        REQUIRE(!s.conn.wait_for("session.done").is_null());
        const std::vector<std::string> f = s.conn.frames(0);
        REQUIRE_EQ(f.size(), (size_t)3);
        CHECK_EQ(f[0].size(), (size_t)44);
        CHECK_EQ(le32(f[0], 40), 0xffffffffu);
        CHECK(f[1] + f[2] == pcm);
        CHECK_EQ(s.conn.wait_for("audio.done").value("total_bytes", -1), (int)(44 + pcm.size()));
    }
}

/* A message the session cannot take is answered with an error naming it, and the session goes on:
 * text before any configuration, an unknown message type, the commitment text mode, a field the
 * configuration does not have (or the input in it), a bad granularity or format, a request field
 * the model refuses. A configuration refused is not applied; a later one is, and the text after it
 * is spoken. */
TEST(a_speech_session_refuses_what_it_cannot_take_and_goes_on) {
    Stream s;
    size_t n = 0;
    auto refused = [&](const json& msg, const char* param, const char* says) {
        s.conn.client(msg);
        const json e = s.conn.wait_for("error", n++);
        REQUIRE(!e.is_null());
        if (param) CHECK_EQ(e.value("param", std::string()), std::string(param));
        const bool named = has(e.value("message", ""), says);
        CHECK(named);
        if (!named) std::fprintf(stderr, "    error said: %s\n", e.value("message", "").c_str());
    };
    refused({ { "type", "input.text" }, { "text", "Hi." } }, nullptr, "session.config");
    refused({ { "type", "response.create" } }, "type", "response.create");
    refused({ { "type", "session.config" }, { "text_input_mode", "commitment" } },
            "text_input_mode", "commitment");
    refused({ { "type", "session.config" }, { "temperature", 0.5 } }, "temperature",
            "temperature");
    refused({ { "type", "session.config" }, { "input", "Hi." } }, "input", "input.text");
    refused({ { "type", "session.config" }, { "split_granularity", "word" } },
            "split_granularity", "clause");
    refused({ { "type", "session.config" }, { "response_format", "mp3" } }, "response_format",
            "pcm or wav");
    refused({ { "type", "session.config" }, { "instructions", "calm" }, { "guidance_scale", 0.5 } },
            "guidance_scale", "[1, 30]");
    refused({ { "type", "session.config" } }, "instructions", "instructions");
    /* None of it applied: text still waits for a configuration. */
    refused({ { "type", "input.done" } }, nullptr, "session.config");
    s.config({ { "split_granularity", "clause" } });
    s.text("Yes, no");
    s.done();
    REQUIRE(!s.conn.wait_for("session.done").is_null());
    CHECK_EQ(s.conn.wait_for("audio.start", 0).value("sentence_text", ""), std::string("Yes,"));
    CHECK_EQ(s.conn.wait_for("audio.start", 1).value("sentence_text", ""), std::string("no"));
    CHECK(!s.conn.closed);
}

/* A frame that is brackets alone, nested tens of thousands deep, is still valid JSON: parsing it
 * into a tree and tearing that tree down both recurse once per level, so the session parses it
 * bounded -- the tree never outgrows the stack -- answers its invalid-message error, and goes on.
 */
TEST(a_deeply_nested_frame_is_refused_not_fatal) {
    Stream s;
    const std::string deep = std::string(20000, '[') + std::string(20000, ']');
    {
        std::lock_guard<std::mutex> lk(s.conn.m);
        s.conn.in.push_back(deep);
        s.conn.cv.notify_all();
    }
    CHECK(!s.conn.wait_for("error").is_null());
    CHECK(!s.conn.closed);
    s.config();
    s.text("Still here.");
    s.done();
    REQUIRE(!s.conn.wait_for("session.done").is_null());
}

/* A sentence the generator refuses or fails is an error and an audio.done saying so; the next
 * sentence is spoken as usual. */
TEST(a_failed_sentence_says_so_and_the_next_is_spoken) {
    Stream s;
    s.config();
    s.gen.finish_status = RAD_E_IO;
    s.gen.finish_why = "the vocoder ran out of memory";
    s.text("One. ");
    const json e = s.conn.wait_for("error");
    REQUIRE(!e.is_null());
    CHECK_EQ(e.value("message", ""), s.gen.finish_why);
    const json d = s.conn.wait_for("audio.done");
    REQUIRE(!d.is_null());
    CHECK_EQ(d.value("error", false), true);
    CHECK_EQ(d.value("total_bytes", -1), 0);
    s.gen.finish_status = RAD_OK;
    s.text("Two.");
    s.done();
    REQUIRE(!s.conn.wait_for("session.done").is_null());
    CHECK_EQ(s.conn.wait_for("audio.done", 1).value("error", true), false);
    /* A sentence too long for the model's prompt is refused by name before it runs. */
    s.text(std::string(80, 'a') + ".");
    s.done();
    const json e2 = s.conn.wait_for("error", 1);
    REQUIRE(!e2.is_null());
    CHECK_EQ(e2.value("param", std::string()), std::string("input"));
    CHECK(!s.conn.wait_for("session.done", 1).is_null());
}

/* session.close ends the session at once: the sentence being spoken is cancelled and the socket
 * is closed normally. The admission place the session held comes back. */
TEST(session_close_cancels_the_sentence_being_spoken) {
    Stream s;
    s.gen.hold = true;
    s.config();
    s.text("Hello there. More");
    REQUIRE(!s.conn.wait_for("audio.start").is_null());
    for (int i = 0; i < 1000 && s.gen.n_got() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    REQUIRE_EQ(s.gen.n_got(), (size_t)1);
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)1);
    s.conn.client({ { "type", "session.close" } });
    REQUIRE(s.conn.wait_closed());
    CHECK_EQ(s.conn.close_code, 1000);
    s.route.join();
    CHECK(s.gen.ticket_of(0)->cancel.load());
    CHECK_EQ(s.f->srv->in_flight(), (int64_t)0);
    CHECK(s.conn.wait_for("audio.done", 0, 100).is_null());
}

/* A plain GET of the path is told it is a socket; a model that makes no audio refuses the session
 * and closes it. */
TEST(the_speech_socket_wants_an_upgrade_and_an_audio_model) {
    Fixture text_model;
    CHECK_EQ(text_model.srv->router().call("GET", "/v1/audio/speech/stream").status, 426);
    Conn c;
    HttpRequest q;
    q.path = "/v1/audio/speech/stream";
    (*text_model.srv->router().socket_for("/v1/audio/speech/stream"))(q, c);
    CHECK(c.closed);
    CHECK_EQ(c.close_code, 1008);
    REQUIRE(!c.out.empty());
    CHECK(has(c.out.front().first, "audio generation model"));
}

RAD_TEST_MAIN()
