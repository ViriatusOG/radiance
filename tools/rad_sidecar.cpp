/* rad-sidecar -- a weight sidecar: some of a container's plain weights, quantised, in a small
 * container of their own.
 *
 * usage: rad-sidecar <model.rad> -o <sidecar.rad> (--recipe FILE | --quant RULE)...
 *                    [--home DIR] [--plan-only] [-v]
 *
 * Every entry of <model.rad> the recipe's first matching rule decides (rad-convert's rules, the
 * same spellings, core/format/recipe.h) is read as f32 from its plain planes (bf16, f16 or f32)
 * and quantised by that rule's quantiser through the loop rad-convert writes containers by
 * (core/format/quantise.h), so the sidecar holds the numbers rad-convert would have written from
 * the same values. The engine serves it under `weight-sidecar: FILE`: each sidecar entry stands
 * in for the container's entry of the same name and logical shape (core/engine_bringup.cpp,
 * Engine::open_weight_sidecar). The container is only read.
 *
 * Refused: a matched entry that is a routed expert's (the container's expert directory addresses
 * those at a fixed stride), or one whose planes are not plain floats (a quantised weight is not
 * quantised again here). A rule its quantiser declines (a norm under a matrix glob) leaves that
 * entry in the container, as rad-convert keeps it. The sidecar records the container's size and
 * entry count (sidecar.container_bytes / sidecar.container_entries), which the engine checks,
 * and the recipe as written.
 */
#include "iface.h"
#include "format/encoding.h"
#include "format/quantise.h"
#include "format/recipe.h"
#include "plugin/registry.h"
#include "rad_quant.h"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <string>
#include <tuple>
#include <vector>

using namespace rad;

namespace {

void usage() {
    std::printf(
        "rad-sidecar -- some of a container's plain weights, quantised, as a sidecar the engine "
        "serves in their place\n"
        "\n"
        "usage: rad-sidecar <model.rad> -o <sidecar.rad> (--recipe FILE | --quant RULE)... "
        "[options]\n"
        "\n"
        "  -o, --out FILE     the sidecar to write\n"
        "      --recipe FILE  rules as rad-convert reads them: PATTERN  QUANTISER  key=value ...;\n"
        "                     the first rule whose glob matches an entry's name decides it; an\n"
        "                     entry no rule matches is left out\n"
        "      --quant RULE   'PATTERN=QUANTISER:key=value,...', ahead of the recipe file's\n"
        "                     rules. Repeatable\n"
        "      --home DIR     $RADIANCE_HOME (its quantizers/)\n"
        "      --plan-only    print what each matched entry becomes, write nothing\n"
        "  -v, --verbose\n"
        "  -h, --help\n"
        "\n"
        "Serve it with weight-sidecar: <sidecar.rad> in the model's entry beside model: "
        "<model.rad>.\n");
}

std::string basename_of(const std::string& p) {
    const size_t s = p.find_last_of('/');
    return s == std::string::npos ? p : p.substr(s + 1);
}

std::string enc_text(const RadEncoding& e) {
    char b[200];
    rad_enc_format(&e, b, sizeof b);
    return b;
}

struct Job {
    const RadFileEntry* e = nullptr;
    std::string         name;
    const CompiledRule* rule = nullptr;
    RadQuantWeight      qw{};
    RadEncoding         enc{};
    int64_t             idx = -1;
    int64_t             bytes = 0;
    double              err = -1;
};

}  /* namespace */

int main(int argc, char** argv) {
    std::string in, out, home = rad_home();
    std::vector<std::string> recipe_files, quant_flags;
    bool plan_only = false, verbose = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) {
            return rad_flag_value("rad-sidecar", argc, argv, &i, what);
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "-o" || a == "--out") out = next("--out");
        else if (a == "--recipe") recipe_files.push_back(next("--recipe"));
        else if (a == "--quant") quant_flags.push_back(next("--quant"));
        else if (a == "--home") home = next("--home");
        else if (a == "--plan-only") plan_only = true;
        else if (a == "-v" || a == "--verbose") { verbose = true; log_set_level(Log::Debug); }
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "rad-sidecar: unrecognised argument '%s'. Try --help.\n",
                         a.c_str());
            return 2;
        } else if (in.empty()) in = a;
        else { std::fprintf(stderr, "rad-sidecar: more than one input given\n"); return 2; }
    }
    if (in.empty() || (out.empty() && !plan_only) ||
        (recipe_files.empty() && quant_flags.empty())) {
        usage();
        return 2;
    }

    /* ---- the rules, typed by the quantisers' own option tables */
    Recipe recipe;
    for (const std::string& q : quant_flags) {
        std::string why;
        if (recipe.add_flag(q, &why) < 0) {
            RAD_ERR("--quant %s: %s", q.c_str(), why.c_str());
            return 2;
        }
    }
    for (const std::string& f : recipe_files) {
        std::string text, why;
        if (!read_file_bytes(f, &text)) {
            RAD_ERR("--recipe %s cannot be read", f.c_str());
            return 2;
        }
        if (recipe.add_text(text, f, &why) < 0) { RAD_ERR("%s", why.c_str()); return 2; }
    }
    Registry& reg = rad_tools_registry();
    if (rad_load_quantizers(reg, home) < 0) return 1;
    std::deque<CompiledRule> compiled;
    std::map<const RecipeRule*, const CompiledRule*> by_rule;
    for (const RecipeRule& r : recipe.rules()) {
        compiled.emplace_back();
        std::string why;
        if (compiled.back().build(r, reg, &why) < 0) { RAD_ERR("%s", why.c_str()); return 2; }
        by_rule[&r] = &compiled.back();
    }

    /* ---- the plan: every matched entry, its encoding as the quantiser answers it */
    RadFile f;
    RAD_TRY(f.open(in.c_str()));
    std::vector<Job> jobs;
    int64_t src_bytes = 0, declined = 0;
    for (int64_t i = 0; i < f.entry_count(); ++i) {
        const RadFileEntry& e = f.entry(i);
        const std::string name = f.str(e.name);
        const RecipeRule* r = recipe.match(name);
        if (!r) continue;
        const RadEncoding& se = f.encoding(e);
        if (e.expert >= 0) {
            RAD_ERR("%s (%s) matches '%s', a routed expert's weight: the container's expert "
                    "directory addresses those at a fixed stride, so a sidecar does not replace "
                    "them", r->origin.c_str(), r->pattern.c_str(), name.c_str());
            return 2;
        }
        const uint32_t dt = se.n_planes == 1 ? se.plane[0].dtype : (uint32_t)RAD_DT_INVALID;
        if (!rad_enc_is(&se, "plain") || se.n_planes != 1 ||
            (dt != RAD_BF16 && dt != RAD_F16 && dt != RAD_F32)) {
            RAD_ERR("%s (%s) matches '%s', which the container holds as %s: only plain bf16 / "
                    "f16 / f32 weights are quantised here, not a quantised one again",
                    r->origin.c_str(), r->pattern.c_str(), name.c_str(), enc_text(se).c_str());
            return 2;
        }
        Job j;
        j.e = &e;
        j.name = name;
        j.rule = by_rule[r];
        j.qw.name = nullptr;
        j.qw.rank = e.rank;
        for (uint32_t k = 0; k < RAD_MAX_RANK; ++k) j.qw.shape[k] = e.shape[k];
        const int64_t cols = e.rank ? e.shape[e.rank - 1] : 1;
        int64_t rows = 1;
        for (uint32_t k = 0; k + 1 < e.rank; ++k) rows *= e.shape[k];
        j.qw.rows = rows;
        j.qw.cols = cols;
        j.qw.layer = e.layer;
        j.qw.expert = -1;
        j.qw.importance = nullptr;
        jobs.push_back(std::move(j));
    }
    /* names point into the jobs' own strings once the vector stops moving */
    for (Job& j : jobs) j.qw.name = j.name.c_str();
    std::vector<Job> keep;
    for (Job& j : jobs) {
        const CompiledRule& c = *j.rule;
        const int st = c.q->encoding(c.params.data(), (int)c.params.size(), &j.qw, &j.enc);
        if (st == RAD_E_UNSUPPORTED) {
            ++declined;
            if (verbose)
                RAD_DEBUG("  %-52s declined by %s: stays in the container", j.name.c_str(),
                          c.q->name);
            continue;
        }
        if (st != RAD_OK || !rad_enc_valid(&j.enc)) {
            RAD_ERR("%s (%s) refuses '%s': %s", c.rule->origin.c_str(), c.rule->quantizer.c_str(),
                    j.name.c_str(), st != RAD_OK ? rad_strerror(st) : "a malformed encoding");
            return 2;
        }
        keep.push_back(std::move(j));
    }
    jobs.swap(keep);
    if (jobs.empty()) {
        RAD_ERR("no entry of %s is matched by a rule its quantiser takes (%lld declined): "
                "nothing to write", in.c_str(), (long long)declined);
        return 2;
    }
    /* movement-unit order, as rad-convert writes: model-level weights, then each layer's */
    std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
        return std::make_tuple(a.e->layer < 0 ? -1 : a.e->layer, a.name) <
               std::make_tuple(b.e->layer < 0 ? -1 : b.e->layer, b.name);
    });
    /* the sort moved the names; point each at its own string again */
    for (Job& j : jobs) j.qw.name = j.name.c_str();
    for (const Job& j : jobs) src_bytes += (int64_t)j.e->bytes;
    if (plan_only) {
        int64_t nb = 0;
        for (const Job& j : jobs) {
            int64_t b = 0;
            for (int k = 0; k < j.enc.n_planes; ++k)
                b += rad_enc_plane_bytes(&j.enc.plane[k], j.qw.rows, j.qw.cols);
            nb += b;
            std::printf("%-52s %-24s -> %-48s %12lld -> %12lld bytes (%s)\n", j.name.c_str(),
                        enc_text(f.encoding(*j.e)).c_str(), enc_text(j.enc).c_str(),
                        (long long)j.e->bytes, (long long)b, j.rule->q->name);
        }
        std::printf("plan: %zu entries, %lld -> %lld bytes (%.3f); %lld declined\n", jobs.size(),
                    (long long)src_bytes, (long long)nb,
                    src_bytes ? (double)nb / (double)src_bytes : 0.0, (long long)declined);
        return 0;
    }

    /* ---- the sidecar */
    RadWriter w;
    RAD_TRY(w.begin(out.c_str()));
    w.set_arch(f.str(f.header().arch_id));
    w.set_model_name(std::string(f.str(f.header().model_name)) + " (weight sidecar)");
    w.set_quant("sidecar");
    w.set_recipe(recipe.text());
    w.set_created_by("rad-sidecar");
    w.meta_i("sidecar.container_bytes", f.size());
    w.meta_i("sidecar.container_entries", f.entry_count());
    w.meta_s("sidecar.container", basename_of(in));
    for (Job& j : jobs) {
        RadWriter::Weight ww;
        ww.name = j.name;
        ww.quantizer = j.rule->rule->quantizer;
        ww.options = j.rule->rule->options_text();
        ww.enc = j.enc;
        ww.rank = j.e->rank;
        for (uint32_t k = 0; k < RAD_MAX_RANK; ++k) ww.shape[k] = j.e->shape[k];
        ww.layer = j.e->layer;
        ww.expert = -1;
        ww.slot = 0;
        j.idx = w.add_weight(ww);
        if (j.idx < 0) { w.abort(); return 1; }
    }
    if (w.seal() < 0) { w.abort(); return 1; }
    int64_t written = 0;
    for (Job& j : jobs) {
        const CompiledRule& c = *j.rule;
        const RadEncoding& se = f.encoding(*j.e);
        const RadFilePlane& sp = f.plane(*j.e, 0);
        const void* sd = f.data(sp);
        EncodedView sv;
        enc_view(se, j.qw.rows, j.qw.cols, &sd, &sv);
        const int64_t C = j.qw.cols;
        double num = 0, den = 0;
        bool decodes = true;
        std::vector<float> dec;
        QuantIo io;
        io.read = [&](int64_t r0, int64_t n, float* o, std::string* why) {
            return enc_decode_rows(sv, r0, n, o, why);
        };
        io.write = [&](int k, int64_t off, const void* data, int64_t bytes) {
            return w.write_plane(j.idx, k, off, data, bytes);
        };
        io.band = [&](int64_t, int64_t n, const float* ref, const void* const* pd) {
            if (!decodes) return;
            EncodedView v;
            std::string why;
            enc_view(j.enc, n, C, pd, &v);
            dec.resize((size_t)(n * C));
            if (enc_decode_rows(v, 0, n, dec.data(), &why) != RAD_OK) {
                RAD_WARN("'%s': nothing decodes what %s wrote (%s); its error is not reported",
                         j.name.c_str(), c.q->name, why.c_str());
                decodes = false;
                return;
            }
            for (int64_t i = 0; i < n * C; ++i) {
                const double d = (double)ref[(size_t)i] - dec[(size_t)i], x = ref[(size_t)i];
                num += d * d;
                den += x * x;
            }
        };
        std::string why;
        const int st = quantise_weight(*c.q, c.params.data(), (int)c.params.size(), j.qw, j.enc,
                                       kQuantBandBytes, io, &why);
        if (st != RAD_OK) {
            RAD_ERR("'%s': %s", j.name.c_str(), why.c_str());
            w.abort();
            return 1;
        }
        j.err = decodes && den > 0 ? std::sqrt(num / den) : -1;
        j.bytes = 0;
        for (int k = 0; k < j.enc.n_planes; ++k) j.bytes += w.plane_bytes(j.idx, k);
        written += j.bytes;
        std::printf("  %-52s %-22s -> %-40s %11lld -> %11lld bytes  rel %.3e  (%s)\n",
                    j.name.c_str(), enc_text(se).c_str(), enc_text(j.enc).c_str(),
                    (long long)j.e->bytes, (long long)j.bytes, j.err, c.q->name);
    }
    if (w.finish() < 0) return 1;
    double worst = 0;
    for (const Job& j : jobs) worst = std::max(worst, j.err);
    std::printf("rad-sidecar: %s: %zu entries of %s, %lld -> %lld bytes of planes (%.3f), worst "
                "rel error %.3e; %lld declined (left in the container)\n", out.c_str(),
                jobs.size(), basename_of(in).c_str(), (long long)src_bytes, (long long)written,
                src_bytes ? (double)written / (double)src_bytes : 0.0, worst, (long long)declined);
    return 0;
}
