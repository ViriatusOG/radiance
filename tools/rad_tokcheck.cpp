/* rad-tokcheck -- checks the tokeniser against its references over real text, and times it.
 *
 * Per corpus file:
 *   1. Compare ids with the reference encoder (tools/tokref: llama.cpp's splitters, std::regex, a
 *      string-keyed merge). A difference is printed with the surrounding ids and text.
 *   2. Compare ids with HuggingFace tokenizers, read from pre-made files
 *      (`tokenizer.encode(text, add_special_tokens=False).ids`, little-endian int32, one file per
 *      corpus file), so no Python is needed here.
 *   3. Time both encoders (MB/s), and re-tokenising a growing conversation with and without the
 *      segment cache, as when an agent re-sends its whole history every turn.
 *
 * Exits non-zero on any difference, so it can gate a change.
 */
#include "text/tokenizer.h"
#include "tokref.h"
#include "iface.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace rad;

namespace {

void usage() {
    std::printf(
        "rad-tokcheck -- compare the tokeniser against its reference encoder and HuggingFace\n"
        "\n"
        "usage: rad-tokcheck [options] <tokenizer.json> <corpus-file>...\n"
        "\n"
        "  --hf-dir DIR    compare against HuggingFace ids in DIR/<corpus basename>.ids\n"
        "                  (little-endian int32, encode(text, add_special_tokens=False))\n"
        "  --no-ref        skip the reference encoder, which is slow\n"
        "  --user          encode with parse_special off: a control token's spelling is text\n"
        "  --turns N       turns in the multi-turn measurement (default 40, 0 skips it)\n"
        "  --repeat N      timing repeats, best taken (default 3)\n"
        "  --context N     ids shown either side of a difference (default 6)\n"
        "  -h, --help\n");
}

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

std::string basename_of(const std::string& p) {
    const size_t s = p.find_last_of('/');
    return s == std::string::npos ? p : p.substr(s + 1);
}

std::string printable(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20 || c == 0x7F) o += fmt("\\x%02x", c);
        else o += (char)c;
    }
    return o;
}

/* The first index where `a` and `b` differ, printed with the ids either side and the text they
 * spell. Returns false when they are identical. */
bool report_diff(const Tokenizer& tok, const char* what, const std::vector<int32_t>& a,
                 const std::vector<int32_t>& b, const std::string& text, int ctx) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
    if (i == a.size() && i == b.size()) return false;
    size_t byte = 0;
    for (size_t k = 0; k < i; ++k) byte += tok.piece(a[k], true).size();
    std::printf("    DIFFERENT from %s at id %zu (of %zu vs %zu), about byte %zu\n", what, i,
                a.size(), b.size(), byte);
    const size_t lo = i > (size_t)ctx ? i - ctx : 0;
    auto show = [&](const char* name, const std::vector<int32_t>& v) {
        std::printf("      %-9s", name);
        for (size_t k = lo; k < std::min(v.size(), i + ctx); ++k)
            std::printf("%s%d'%s'", k == i ? " >>" : " ", v[k],
                        printable(tok.piece(v[k], true)).c_str());
        std::printf("\n");
    };
    show("tokenizer", a);
    show(what, b);
    /* The pieces' bytes overshoot the text where a piece spells more than it covers (an
     * end-of-word suffix), so the window is clamped to the text. */
    const size_t t0 = std::min(text.size(), byte > 60 ? byte - 60 : 0);
    std::printf("      text      '%s'\n",
                printable(text.substr(t0, std::min<size_t>(120, text.size() - t0))).c_str());
    return true;
}

/* The corpus cut into `n` turns at line boundaries, and each turn's prompt: every message so far,
 * alternating user and assistant, framed the way a ChatML template frames them when the vocab has
 * its markers. */
std::vector<std::string> conversation(const Vocab& v, const std::string& text, int n) {
    std::vector<std::string> msgs;
    const size_t per = text.size() / (size_t)n + 1;
    size_t pos = 0;
    while (pos < text.size() && (int)msgs.size() < n) {
        size_t end = std::min(text.size(), pos + per);
        const size_t nl = text.find('\n', end);
        end = nl == std::string::npos ? text.size() : nl + 1;
        msgs.push_back(text.substr(pos, end - pos));
        pos = end;
    }
    const bool chatml = v.find("<|im_start|>") >= 0 && v.find("<|im_end|>") >= 0;
    std::vector<std::string> prompts;
    std::string conv;
    for (size_t k = 0; k < msgs.size(); ++k) {
        const char* role = k % 2 ? "assistant" : "user";
        conv += chatml ? std::string("<|im_start|>") + role + "\n" + msgs[k] + "<|im_end|>\n"
                       : msgs[k];
        prompts.push_back(conv + (chatml ? "<|im_start|>assistant\n" : ""));
    }
    return prompts;
}

}  /* namespace */

int main(int argc, char** argv) {
    std::string hf_dir;
    bool ref = true, parse_special = true;
    int turns = 40, repeat = 3, ctx = 6;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "rad-tokcheck: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--hf-dir")  hf_dir = next();
        else if (a == "--no-ref")  ref = false;
        else if (a == "--user")    parse_special = false;
        else if (a == "--turns")
            turns = (int)rad_flag_int("rad-tokcheck", "--turns", next(), 0, INT32_MAX);
        else if (a == "--repeat")
            repeat = (int)rad_flag_int("rad-tokcheck", "--repeat", next(), 1, INT32_MAX);
        else if (a == "--context")
            ctx = (int)rad_flag_int("rad-tokcheck", "--context", next(), 0, INT32_MAX);
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "rad-tokcheck: unknown option %s\n", a.c_str());
            return 2;
        }
        else args.push_back(a);
    }
    if (args.size() < 2) { usage(); return 2; }

    log_set_level(Log::Warn);
    VocabBuild vb;
    if (parse_tokenizer_json(args[0].c_str(), vb) < 0) return 1;
    auto vocab = std::make_shared<Vocab>();
    const auto l0 = Clock::now();
    if (vocab->load(std::move(vb)) < 0) return 1;
    const double load_s = secs(l0);
    Tokenizer tok;
    if (tok.init(vocab) < 0) return 1;
    tok.set_segment_cache(false);
    std::printf("%s: %d tokens, vocab load %.1f ms\n", args[0].c_str(), vocab->n_tokens(),
                load_s * 1e3);

    int bad = 0;
    size_t total_bytes = 0;
    double total_new = 0, total_ref = 0;
    size_t total_ids = 0, hf_checked = 0;
    for (size_t f = 1; f < args.size(); ++f) {
        std::string text;
        if (!read_file_bytes(args[f], &text)) {
            std::printf("  %s: cannot read\n", args[f].c_str());
            ++bad;
            continue;
        }
        std::printf("  %s: %zu bytes\n", args[f].c_str(), text.size());

        /* The first encode is reported separately from the best repeat: the vocab caches which
         * tokens merge back to themselves, so repeats are faster. */
        std::vector<int32_t> ids;
        double best = 1e30, first = 0;
        for (int r = 0; r < repeat; ++r) {
            ids.clear();
            const auto t0 = Clock::now();
            if (tok.encode(text, ids, false, parse_special) < 0) {
                std::printf("    encode failed\n");
                ++bad;
                break;
            }
            const double t = secs(t0);
            if (r == 0) first = t;
            best = std::min(best, t);
        }
        std::printf("    tokenizer  %8zu ids  %8.2f ms  %7.1f MB/s  (first encode %.2f ms, "
                    "%.1f MB/s)\n",
                    ids.size(), best * 1e3, text.size() / best / 1e6, first * 1e3,
                    text.size() / first / 1e6);
        total_bytes += text.size();
        total_new += best;
        total_ids += ids.size();

        if (ref) {
            std::vector<int32_t> rids;
            const auto t0 = Clock::now();
            const int st = tokref::encode(*vocab, text, rids, false, parse_special);
            const double rs = secs(t0);
            if (st < 0) {
                std::printf("    reference  failed (%d)\n", st);
            } else {
                std::printf("    reference  %8zu ids  %8.2f ms  %7.1f MB/s\n", rids.size(),
                            rs * 1e3, text.size() / rs / 1e6);
                total_ref += rs;
                if (report_diff(tok, "reference", ids, rids, text, ctx)) ++bad;
                else std::printf("    reference  identical\n");
            }
        }

        if (!hf_dir.empty()) {
            std::string raw;
            const std::string path = hf_dir + "/" + basename_of(args[f]) + ".ids";
            if (!read_file_bytes(path, &raw)) {
                std::printf("    huggingface: no %s\n", path.c_str());
            } else {
                std::vector<int32_t> hids(raw.size() / 4);
                std::memcpy(hids.data(), raw.data(), hids.size() * 4);
                ++hf_checked;
                if (report_diff(tok, "hf", ids, hids, text, ctx)) ++bad;
                else std::printf("    huggingface identical (%zu ids)\n", hids.size());
            }
        }

        if (turns > 0) {
            const std::vector<std::string> prompts = conversation(*vocab, text, turns);
            Tokenizer cached;
            if (cached.init(vocab) < 0) {
                std::printf("    multi-turn: init failed\n");
                ++bad;
                continue;
            }
            cached.clear_cache();
            double cold = 0, warm = 0, cold_last = 0, warm_last = 0, rtot = 0, rlast = 0;
            size_t prompt_bytes = 0;
            bool same = true;
            for (size_t k = 0; k < prompts.size(); ++k) {
                std::vector<int32_t> a, b, r;
                auto t0 = Clock::now();
                const int sa = tok.encode(prompts[k], a, false, true);
                const double c = secs(t0);
                t0 = Clock::now();
                const int sb = cached.encode(prompts[k], b, false, true);
                const double w = secs(t0);
                if (sa < 0 || sb < 0) {
                    std::printf("    turn %zu: encode failed\n", k);
                    same = false;
                    continue;
                }
                cold += c;
                warm += w;
                cold_last = c;
                warm_last = w;
                prompt_bytes += prompts[k].size();
                if (a != b) {
                    same = false;
                    report_diff(tok, "cached", a, b, prompts[k], ctx);
                }
                if (ref) {
                    t0 = Clock::now();
                    tokref::encode(*vocab, prompts[k], r, false, true);
                    rlast = secs(t0);
                    rtot += rlast;
                }
            }
            const Tokenizer::CacheStats cs = cached.cache_stats();
            std::printf("    %zu turns, %.1f MB re-sent (last prompt %zu bytes): %s\n",
                        prompts.size(), prompt_bytes / 1e6,
                        prompts.empty() ? 0 : prompts.back().size(),
                        same ? "cached ids identical to uncached on every turn"
                             : "cached ids differ");
            std::printf("      uncached   %9.1f ms total, last turn %8.2f ms\n", cold * 1e3,
                        cold_last * 1e3);
            std::printf("      cached     %9.1f ms total, last turn %8.2f ms  "
                        "(%llu hits, %zu entries, %.1f MB)\n", warm * 1e3, warm_last * 1e3,
                        (unsigned long long)cs.hits, cs.entries, cs.bytes / 1e6);
            if (ref)
                std::printf("      reference  %9.1f ms total, last turn %8.2f ms\n", rtot * 1e3,
                            rlast * 1e3);
            if (!same) ++bad;
        }
    }
    std::printf("total: %.1f MB, %zu ids; tokenizer %.1f MB/s", total_bytes / 1e6, total_ids,
                total_bytes / total_new / 1e6);
    if (ref && total_ref > 0) std::printf(", reference %.1f MB/s", total_bytes / total_ref / 1e6);
    if (!hf_dir.empty()) std::printf("; %zu file(s) checked against huggingface", hf_checked);
    std::printf("; %d difference(s)\n", bad);
    return bad ? 1 : 0;
}
