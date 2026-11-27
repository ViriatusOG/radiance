/* rad-info -- print what a .rad contains.
 *
 * Output is for reading: aligned columns, human byte units, and every count labelled with what it
 * counts. The container is opened read-only with no plugins and no declare phase, so it works on
 * any container regardless of what is installed.
 *
 * `--plugins` instead loads the plugins from $RADIANCE_HOME as the engine does and prints each as
 * it registered, with the file it came from.
 */
#include "iface.h"
#include "format/encoding.h"
#include "format/recipe.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

using namespace rad;

namespace {

void usage() {
    std::printf(
        "rad-info -- print what a .rad container holds\n"
        "\n"
        "usage: rad-info [options] <model.rad>\n"
        "       rad-info --plugins [--home DIRS]\n"
        "\n"
        "  -v, --verbose      list every tensor, not just the per-unit summary\n"
        "      --meta         print the metadata table and stop\n"
        "      --vocab        print the vocab summary and its normaliser chain, and stop\n"
        "      --profile N    print the top N entries of the expert profile (default 20, 0 = all)\n"
        "      --recipe       print the recipe the container was converted with, and stop\n"
        "      --unreferenced [N]  print the byte ranges of at least N bytes (default 1048576)\n"
        "                     that no table and no plane names, one `offset bytes` a line, and\n"
        "                     stop: what `rad-convert --in-place --replace` left behind. Nothing\n"
        "                     reads them, so `fallocate --punch-hole` gives their disk back.\n"
        "      --planes GLOB  print the planes of every weight whose name the glob matches, one\n"
        "                     `offset bytes name` a line, and stop\n"
        "      --plugins      print every plugin the $RADIANCE_HOME search path offers, as the\n"
        "                     engine loads it, and stop\n"
        "      --home DIRS    the search path for --plugins (default $RADIANCE_HOME)\n"
        "  -h, --help\n");
}

std::string shape_str(const int64_t* shape, uint32_t rank) {
    if (rank == 0) return "-";
    std::string s;
    for (uint32_t i = 0; i < rank; ++i) {
        if (i) s += 'x';
        s += fmt("%lld", (long long)shape[i]);
    }
    return s;
}

const char* step_name(uint32_t k) {
    switch (k) {
        case RAD_NORM_NFC:        return "norm.NFC";
        case RAD_NORM_NFKC:       return "norm.NFKC";
        case RAD_NORM_NFD:        return "norm.NFD";
        case RAD_NORM_NFKD:       return "norm.NFKD";
        case RAD_NORM_LOWERCASE:  return "norm.Lowercase";
        case RAD_NORM_STRIP:      return "norm.Strip";
        case RAD_NORM_REPLACE:    return "norm.Replace";
        case RAD_NORM_PREPEND:    return "norm.Prepend";
        case RAD_PRE_BYTELEVEL:   return "pre.ByteLevel";
        case RAD_PRE_SPLIT:       return "pre.Split";
        case RAD_PRE_METASPACE:   return "pre.Metaspace";
        case RAD_PRE_WHITESPACE:  return "pre.Whitespace";
        case RAD_PRE_PUNCTUATION: return "pre.Punctuation";
        case RAD_PRE_DIGITS:      return "pre.Digits";
        case RAD_PRE_BYTE_FALLBACK: return "pre.ByteFallback";
        case RAD_PRE_END_OF_WORD:   return "pre.EndOfWordSuffix";
        case RAD_PRE_ADDED_NORMALIZED: return "pre.AddedNormalized";
        case RAD_DEC_BYTELEVEL:   return "dec.ByteLevel";
        case RAD_DEC_METASPACE:   return "dec.Metaspace";
        case RAD_DEC_REPLACE:     return "dec.Replace";
        case RAD_DEC_STRIP:       return "dec.Strip";
        case RAD_DEC_FUSE:        return "dec.Fuse";
        case RAD_DEC_BYTE_FALLBACK: return "dec.ByteFallback";
        default:                  return "?";
    }
}

const char* tok_kind(uint32_t k) {
    switch (k) {
        case RAD_TOK_BPE:       return "bpe";
        case RAD_TOK_UNIGRAM:   return "unigram";
        case RAD_TOK_WORDPIECE: return "wordpiece";
        case RAD_TOK_RWKV:      return "rwkv";
        default:                return "?";
    }
}

/* Escape a token's text for one report line (control characters, newlines), truncated at `cap`. */
std::string vis(const char* s, size_t cap = 40) {
    std::string out;
    for (const char* p = s; *p && out.size() < cap; ++p) {
        unsigned char c = (unsigned char)*p;
        if (c == '\n') out += "\\n";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) out += fmt("\\x%02x", c);
        else out.push_back((char)c);
    }
    if (out.size() >= cap) out += "...";
    return out;
}

/* ------------------------------------------------------------------ sections */

void print_header(const RadFile& f) {
    const RadFileHeader& h = f.header();
    std::printf("%s\n", f.path());
    std::printf("  format        RAD%u, %s, little-endian\n", h.version,
                humanb((int64_t)h.file_bytes).c_str());
    std::printf("  architecture  %s\n", f.str(h.arch_id));
    std::printf("  model         %s\n", f.str(h.model_name));
    std::printf("  source quant  %s\n", f.str(h.quant)[0] ? f.str(h.quant) : "(none)");
    {
        const char* r = f.str(h.recipe);
        int lines = 0;
        for (const char* p = r; *p; ++p) lines += *p == '\n';
        if (r[0] && r[std::strlen(r) - 1] != '\n') ++lines;
        std::printf("  recipe        %s\n",
                    r[0] ? fmt("%d line(s) -- --recipe prints it", lines).c_str()
                         : "(none: every weight is the checkpoint's own)");
    }
    std::printf("  created by    %s\n", f.str(h.created_by));

    std::printf("\n  section          offset        bytes    count\n");
    auto row = [&](const char* n, uint64_t off, uint64_t bytes, const char* count) {
        std::printf("  %-12s %9llu %12s %8s\n", n, (unsigned long long)off,
                    humanb((int64_t)bytes).c_str(), count);
    };
    row("strings",  h.str_off,   h.str_bytes, "-");
    row("metadata", h.meta_off,  h.meta_count * sizeof(RadFileKV),
        fmt("%llu", (unsigned long long)h.meta_count).c_str());
    row("directory",h.dir_off,   h.dir_count * sizeof(RadFileEntry),
        fmt("%llu", (unsigned long long)h.dir_count).c_str());
    row("planes",   h.plane_off, h.plane_count * sizeof(RadFilePlane),
        fmt("%llu", (unsigned long long)h.plane_count).c_str());
    row("encodings",h.enc_off,   h.enc_count * sizeof(RadEncoding),
        fmt("%llu", (unsigned long long)h.enc_count).c_str());
    row("vocab",    h.vocab_off, h.vocab_bytes, "-");
    row("profile",  h.prof_off,  h.prof_count * sizeof(RadFileProfile),
        fmt("%llu", (unsigned long long)h.prof_count).c_str());
    row("data",     h.data_off,  h.data_bytes,
        fmt("%llu", (unsigned long long)h.dir_count).c_str());
}

void print_meta(const RadFile& f) {
    if (f.meta_count() == 0) { std::printf("\nmetadata: none\n"); return; }
    std::printf("\nmetadata (%lld)\n", (long long)f.meta_count());

    size_t w = 4;
    for (int64_t i = 0; i < f.meta_count(); ++i)
        w = std::max(w, std::strlen(f.str(f.meta_at(i).key)));
    w = std::min<size_t>(w, 44);

    for (int64_t i = 0; i < f.meta_count(); ++i) {
        const RadFileKV& kv = f.meta_at(i);
        std::printf("  %-*s  ", (int)w, f.str(kv.key));
        switch (kv.type) {
            case RAD_P_INT:  std::printf("%lld\n", (long long)kv.v.i); break;
            case RAD_KV_F64: std::printf("%g\n", kv.v.f); break;
            case RAD_P_STR:  std::printf("%s\n", vis(f.str(kv.v.s), 120).c_str()); break;
            default:         std::printf("(type %u)\n", kv.type); break;
        }
    }
}

/* Where a weight's numbers came from: the checkpoint as it was, or a quantiser and its options. */
std::string provenance(const RadFile& f, const RadFileEntry& e) {
    const char* q = f.str(e.quantizer);
    if (!q[0]) return "checkpoint";
    const char* o = f.str(e.options);
    return o[0] ? fmt("%s %s", q, o) : std::string(q);
}

void print_tensor_row(const RadFile& f, const RadFileEntry& e, const char* indent) {
    std::printf("%s%-44s %-30s %-16s %12s  @%-12llu %s\n", indent, f.str(e.name),
                enc_name(f.encoding(e)).c_str(), shape_str(e.shape, e.rank).c_str(),
                humanb((int64_t)e.bytes).c_str(), (unsigned long long)e.offset,
                provenance(f, e).c_str());
    const RadEncoding& enc = f.encoding(e);
    if (e.n_planes > 1)
        for (uint32_t k = 0; k < e.n_planes; ++k) {
            const RadFilePlane& p = f.plane(e, (int)k);
            std::printf("%s    plane %-12s %-9s %12s  @%llu\n", indent, enc.plane[k].role,
                        rad_dtype_name(enc.plane[k].dtype), humanb((int64_t)p.bytes).c_str(),
                        (unsigned long long)p.offset);
        }
}

/* The directory, grouped by layer and then by movement unit (what the mover transfers). Unit sizes
 * are printed even without --verbose. */
void print_directory(const RadFile& f, bool verbose) {
    std::printf("\nweight directory (%lld entries, grouped by layer and movement unit)\n",
                (long long)f.entry_count());
    if (f.entry_count() == 0) return;

    int64_t i = 0;
    int64_t total_payload = 0;
    /* Totals by encoding and by provenance (which quantiser made the weights). */
    std::map<std::string, std::pair<int64_t, int64_t>> by_enc;   /* entries, bytes */
    std::map<std::string, std::pair<int64_t, int64_t>> by_prov;

    for (int64_t k = 0; k < f.entry_count(); ++k) {
        const RadFileEntry& e = f.entry(k);
        for (uint32_t p = 0; p < e.n_planes; ++p)
            total_payload += (int64_t)f.plane(e, (int)p).bytes;
        auto& a = by_enc[enc_name(f.encoding(e))];
        ++a.first; a.second += (int64_t)e.bytes;
        auto& b = by_prov[provenance(f, e)];
        ++b.first; b.second += (int64_t)e.bytes;
    }

    while (i < f.entry_count()) {
        const int32_t layer = f.entry(i).layer;

        /* Non-expert entries of this layer. */
        int64_t j = i;
        int64_t unit_bytes = 0, n = 0;
        while (j < f.entry_count() && f.entry(j).layer == layer && f.entry(j).expert < 0) {
            unit_bytes += (int64_t)f.entry(j).bytes;
            ++j; ++n;
        }
        if (n) {
            std::printf("\n  %-20s %3lld tensor(s) %14s   unit: one tensor\n",
                        rad_unit_name(layer, -1).c_str(), (long long)n,
                        humanb(unit_bytes).c_str());
            if (verbose) for (int64_t k = i; k < j; ++k) print_tensor_row(f, f.entry(k), "      ");
        }
        i = j;

        if (i >= f.entry_count() || f.entry(i).layer != layer || f.entry(i).expert < 0) continue;

        const RadFile::ExpertGroup* g = f.group_for_layer(layer);
        if (!g) { ++i; continue; }   /* validate() rules this out */

        std::printf("\n  layer %-14d %3d expert(s) x %d slot(s) %11s   unit: one expert\n",
                    layer, g->n_expert, g->n_slot,
                    humanb((int64_t)g->stride * g->n_expert).c_str());
        /* Experts are addressed as base + id * stride. */
        std::printf("      addressing  base %llu + id * %llu  (payload %s, %s of it padding)\n",
                    (unsigned long long)g->base, (unsigned long long)g->stride,
                    humanb((int64_t)g->unit_bytes).c_str(),
                    humanb((int64_t)(g->stride - g->unit_bytes)).c_str());
        for (int32_t s = 0; s < g->n_slot; ++s) {
            const RadFileEntry& e = f.expert_entry(*g, 0, s);
            std::printf("      slot %d  +%-10llu %-40s %-30s %-16s %10s\n", s,
                        (unsigned long long)g->slot_off[s], f.str(e.name),
                        enc_name(f.encoding(e)).c_str(), shape_str(e.shape, e.rank).c_str(),
                        humanb((int64_t)e.bytes).c_str());
        }
        if (verbose)
            for (int32_t e_id = 0; e_id < g->n_expert; ++e_id)
                for (int32_t s = 0; s < g->n_slot; ++s)
                    print_tensor_row(f, f.expert_entry(*g, e_id, s), "      ");

        i = g->first_index + (int64_t)g->n_expert * g->n_slot;
    }

    const int64_t blob = (int64_t)f.header().data_bytes;
    std::printf("\n  pool totals\n");
    std::printf("      planes         %14s\n", humanb(total_payload).c_str());
    std::printf("      alignment pad  %14s   (%u B per entry, %u B per plane; the movement unit\n"
                "                                      never straddles a page needlessly)\n",
                humanb(blob - total_payload).c_str(), RAD_ALIGN_UNIT, RAD_ALIGN_SUB);
    std::printf("      data blob      %14s\n", humanb(blob).c_str());

    auto table = [&](const char* title,
                     const std::map<std::string, std::pair<int64_t, int64_t>>& m) {
        std::printf("\n  %s\n", title);
        std::vector<std::pair<std::string, std::pair<int64_t, int64_t>>> v(m.begin(), m.end());
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
            return a.second.second > b.second.second;
        });
        for (auto& [k, c] : v)
            std::printf("      %-40s %7lld %14s   %5.1f%%\n", k.c_str(), (long long)c.first,
                        humanb(c.second).c_str(),
                        blob ? 100.0 * (double)c.second / (double)blob : 0.0);
    };
    table("by encoding", by_enc);
    table("by what made it", by_prov);
}

void print_vocab(const RadFile& f) {
    VocabSection v = f.vocab();
    if (!v) { std::printf("\nvocab: none baked into this container\n"); return; }

    std::printf("\nvocab\n");
    std::printf("  kind          %s\n", tok_kind(v.h->kind));
    std::printf("  tokens        %u\n", v.h->n_tokens);
    std::printf("  merges        %u\n", v.h->n_merges);
    std::printf("  specials      bos=%d eos=%d eot=%d pad=%d unk=%d sep=%d  add_bos=%u "
                "add_eos=%u\n",
                v.h->bos, v.h->eos, v.h->eot, v.h->pad_id, v.h->unk, v.h->sep,
                v.h->add_bos, v.h->add_eos);
    const char* ct = f.str(v.h->chat_template);
    std::printf("  chat template %s\n", ct[0] ? fmt("%zu bytes", std::strlen(ct)).c_str() : "none");

    if (v.h->n_steps) {
        /* The chain in declared order, as rad-convert interpreted tokenizer.json (spec §12). */
        std::printf("  chain (%u steps, in order)\n", v.h->n_steps);
        for (uint32_t i = 0; i < v.h->n_steps; ++i) {
            const RadVocabStep& s = v.steps[i];
            std::printf("      %2u  %-20s flags=0x%x", i, step_name(s.kind), s.flags);
            if (f.str(s.arg0)[0]) std::printf("  arg0=\"%s\"", vis(f.str(s.arg0), 60).c_str());
            if (f.str(s.arg1)[0]) std::printf("  arg1=\"%s\"", vis(f.str(s.arg1), 60).c_str());
            if (s.iarg) std::printf("  i=%lld", (long long)s.iarg);
            std::printf("\n");
        }
    } else {
        std::printf("  chain         EMPTY -- nothing will split the input. If this came from a\n"
                    "                tokenizer.json the converter did not interpret it.\n");
    }

    if (v.tok_type) {
        int64_t n[8] = {0};
        for (uint32_t i = 0; i < v.h->n_tokens; ++i)
            n[v.tok_type[i] < 8 ? v.tok_type[i] : 0]++;
        std::printf("  token types   normal=%lld unknown=%lld control=%lld user=%lld byte=%lld\n",
                    (long long)n[RAD_TT_NORMAL], (long long)n[RAD_TT_UNKNOWN],
                    (long long)n[RAD_TT_CONTROL], (long long)n[RAD_TT_USER_DEFINED],
                    (long long)n[RAD_TT_BYTE]);
    }

    /* Every control and user-defined token with its id: these decide where a turn ends. A model
     * that ends its turn on a token not declared eos or eot runs on into a hallucinated next
     * turn. */
    if (v.tok_type && v.tok_text) {
        std::printf("  control/user  (the tokens a turn can end on; eos/eot marked)\n");
        for (uint32_t i = 0; i < v.h->n_tokens; ++i) {
            if (v.tok_type[i] != RAD_TT_CONTROL && v.tok_type[i] != RAD_TT_USER_DEFINED) continue;
            const char* mark = (int32_t)i == v.h->eos ? " <- eos"
                             : (int32_t)i == v.h->eot ? " <- eot" : "";
            std::printf("      %-8u %-8s \"%s\"%s\n", i,
                        v.tok_type[i] == RAD_TT_CONTROL ? "control" : "user",
                        vis(f.str(v.tok_text[i]), 40).c_str(), mark);
        }
    }

    const uint32_t show = std::min<uint32_t>(v.h->n_tokens, 8);
    if (show) {
        std::printf("  first %u       ", show);
        for (uint32_t i = 0; i < show; ++i)
            std::printf("%s\"%s\"", i ? " " : "", vis(f.str(v.tok_text[i]), 16).c_str());
        std::printf("\n");
    }
}

void print_profile(const RadFile& f, int top) {
    if (f.profile_count() == 0) {
        std::printf("\nexpert profile: none. The placement planner will start cold and the heat\n"
                    "engine will rank from real routing within a few dispatches (spec §5.3).\n");
        return;
    }
    std::printf("\nexpert profile (%lld entries, from the imatrix's activation counts)\n",
                (long long)f.profile_count());
    std::printf("  This is a WARM START, not a policy: imatrix popularity is calibration-set\n"
                "  popularity, not decode popularity, and the heat engine replaces it within a\n"
                "  few hundred steps. What it buys is the first few hundred.\n");

    std::vector<const RadFileProfile*> v;
    v.reserve((size_t)f.profile_count());
    for (int64_t i = 0; i < f.profile_count(); ++i) v.push_back(&f.profile()[i]);
    std::sort(v.begin(), v.end(),
              [](const RadFileProfile* a, const RadFileProfile* b) { return a->share > b->share; });

    const int n = (top <= 0) ? (int)v.size() : std::min<int>(top, (int)v.size());
    std::printf("\n  rank  layer  expert     share\n");
    for (int i = 0; i < n; ++i)
        std::printf("  %4d  %5d  %6d  %7.4f%%\n", i + 1, v[i]->layer, v[i]->expert,
                    100.0 * v[i]->share);
    if (n < (int)v.size())
        std::printf("  ... %zu more (--profile 0 for all)\n", v.size() - (size_t)n);
}

/* Every plugin on the search path in registration order: kind, name, what it declares, and its
 * file. */
int print_plugins(const std::string& home) {
    LoadedPlugins lp;
    if (rad_tools_load_plugins(home, rad_hierarchy_from_env(), "", "", &lp) < 0) return 1;
    const Registry& reg = rad_tools_registry();
    std::printf("plugins on %s\n", home.c_str());
    for (const Plugin& p : reg.plugins()) {
        std::string what;
        if (p.kind == RAD_PLUGIN_KERNEL) {
            what = fmt("kernels       %4d kernel(s), %3d schema(s), built for %s%s%s", p.n_kernels,
                       p.n_schemas, p.build_target.empty() ? "anything" : p.build_target.c_str(),
                       p.reference ? ", a reference implementation" : "",
                       p.device_code_absent ? ", no device code for this card" : "");
        } else if (p.kind == RAD_PLUGIN_ARCH) {
            what = fmt("architecture  %s @ %s", p.arch_id.c_str(),
                       p.arch_quant.empty() ? "(unquantised)" : p.arch_quant.c_str());
        } else {
            std::string names;
            for (const auto& q : reg.quantizers())
                if (q.plugin == p.name && q.info)
                    names += (names.empty() ? "" : " ") + std::string(q.info->name);
            what = "quantizers    " + names;
        }
        std::printf("  %2d %-16s %-8s %s\n     %s\n", p.order, p.name.c_str(), p.version.c_str(),
                    what.c_str(), p.path.c_str());
    }
    return 0;
}

/* ------------------------------------------------------------------ unreferenced bytes */

/* Every range of at least `min_bytes` that neither the header, a table it places, nor a plane of
 * the directory covers, as `offset bytes` lines; a summary on stderr. A container written in one
 * pass has only alignment padding between its planes; one extended in place also keeps the tables
 * it had before and the planes of every weight it replaced. */
int print_unreferenced(const RadFile& f, int64_t min_bytes) {
    const RadFileHeader& h = f.header();
    std::vector<std::pair<uint64_t, uint64_t>> live;   /* [first, last) */
    auto add = [&](uint64_t off, uint64_t bytes) {
        if (bytes) live.emplace_back(off, off + bytes);
    };
    add(0, sizeof(RadFileHeader));
    add(h.str_off,   h.str_bytes);
    add(h.meta_off,  h.meta_count * sizeof(RadFileKV));
    add(h.dir_off,   h.dir_count * sizeof(RadFileEntry));
    add(h.plane_off, h.plane_count * sizeof(RadFilePlane));
    add(h.enc_off,   h.enc_count * sizeof(RadEncoding));
    add(h.vocab_off, h.vocab_bytes);
    add(h.prof_off,  h.prof_count * sizeof(RadFileProfile));
    add(h.reserved[RAD_HDR_AUX_VOCAB_OFF], h.reserved[RAD_HDR_AUX_VOCAB_BYTES]);
    for (int64_t i = 0; i < f.entry_count(); ++i) {
        const RadFileEntry& e = f.entry(i);
        for (uint32_t k = 0; k < e.n_planes; ++k) {
            const RadFilePlane& p = f.plane(e, (int)k);
            add(p.offset, p.bytes);
        }
    }
    std::sort(live.begin(), live.end());
    const uint64_t end = (uint64_t)f.size();
    uint64_t at = 0, total = 0, listed = 0;
    int64_t n = 0;
    auto gap = [&](uint64_t a, uint64_t b) {
        if (b <= a) return;
        total += b - a;
        if ((int64_t)(b - a) < min_bytes) return;
        std::printf("%llu %llu\n", (unsigned long long)a, (unsigned long long)(b - a));
        listed += b - a;
        ++n;
    };
    for (const auto& r : live) {
        gap(at, r.first < end ? r.first : end);
        if (r.second > at) at = r.second;
    }
    gap(at, end);
    std::fprintf(stderr, "rad-info: %s of %s is in no table and no plane; %lld range(s) of at "
                 "least %s hold %s\n", humanb((int64_t)total).c_str(),
                 humanb((int64_t)end).c_str(), (long long)n, humanb(min_bytes).c_str(),
                 humanb((int64_t)listed).c_str());
    return 0;
}

/* ------------------------------------------------------------------ a weight's planes */

/* Where the planes of the weights `glob` names sit in the file: before an in-place run replaces
 * or moves them, the list of bytes that run will leave behind. */
int print_planes(const RadFile& f, const std::string& glob) {
    for (int64_t i = 0; i < f.entry_count(); ++i) {
        const RadFileEntry& e = f.entry(i);
        const char* name = f.str(e.name);
        if (!rad::recipe_glob(glob.c_str(), name)) continue;
        for (uint32_t k = 0; k < e.n_planes; ++k) {
            const RadFilePlane& p = f.plane(e, (int)k);
            std::printf("%llu %llu %s\n", (unsigned long long)p.offset,
                        (unsigned long long)p.bytes, name);
        }
    }
    return 0;
}

}  /* namespace */

int main(int argc, char** argv) {
    std::string path;
    bool verbose = false, only_meta = false, only_vocab = false, only_recipe = false;
    bool plugins = false;
    std::string home = rad_home();
    int  top = 20;
    int64_t unref_min = -1;
    std::string planes_glob;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) {
            return rad_flag_value("rad-info", argc, argv, &i, what);
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "-v" || a == "--verbose") verbose = true;
        else if (a == "--meta")  only_meta = true;
        else if (a == "--vocab") only_vocab = true;
        else if (a == "--profile")
            top = (int)rad_flag_int("rad-info", "--profile", next("--profile"), 0, INT32_MAX);
        else if (a == "--recipe") only_recipe = true;
        else if (a == "--unreferenced") {
            unref_min = 1 << 20;
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')
                unref_min = rad_flag_int("rad-info", "--unreferenced", argv[++i], 1, INT64_MAX);
        }
        else if (a == "--planes") planes_glob = next("--planes");
        else if (a == "--plugins") plugins = true;
        else if (a == "--home") home = next("--home");
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "rad-info: unrecognised argument '%s'. Try --help.\n", a.c_str());
            return 2;
        } else if (path.empty()) path = a;
        else { std::fprintf(stderr, "rad-info: more than one model given\n"); return 2; }
    }
    if (plugins) return print_plugins(home);
    if (path.empty()) { usage(); return 2; }

    RadFile f;
    int s = f.open(path.c_str());
    if (s < 0) return 1;

    if (only_recipe) {
        const char* r = f.str(f.header().recipe);
        std::fputs(r[0] ? r : "# no recipe: every weight is the checkpoint's own\n", stdout);
        return 0;
    }
    if (unref_min >= 0) return print_unreferenced(f, unref_min);
    if (!planes_glob.empty()) return print_planes(f, planes_glob);
    if (only_meta)  { print_meta(f);  return 0; }
    if (only_vocab) { print_vocab(f); return 0; }

    print_header(f);
    print_meta(f);
    print_directory(f, verbose);
    print_vocab(f);
    print_profile(f, top);
    return 0;
}
