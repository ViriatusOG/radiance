/* iface.cpp -- CLI plumbing shared by the tools.
 *
 * The plugin loader and declare phase stay in core/build/startup.h, shared with the engine: a
 * second copy could disagree on hierarchy order, which decides the layout baked into a container.
 */
#include "iface.h"
#include "util/kernel_list.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

namespace rad {

/* Same fallback as the engine (core/config.cpp): RAD_DEFAULT_HOME. Do not derive it from the
 * install prefix; the home is a directory below it, so the tools would find a different plugin
 * tree. */
std::string rad_home() {
    if (const char* e = std::getenv("RADIANCE_HOME")) if (*e) return e;
#ifdef RAD_DEFAULT_HOME
    return RAD_DEFAULT_HOME;
#else
    return ".";
#endif
}

std::vector<std::string> rad_split_list(const std::string& s) { return rad_split_kernel_list(s); }

std::vector<std::string> rad_hierarchy_from_env() {
    const char* e = std::getenv("RADIANCE_KERNELS");
    return (e && *e) ? rad_split_list(e) : std::vector<std::string>{};
}

bool rad_parse_int(const char* text, int64_t lo, int64_t hi, int64_t* out, std::string* why) {
    char* end = nullptr;
    errno = 0;
    const long long n = text ? std::strtoll(text, &end, 10) : 0;
    if (!text || !*text || !end || *end || errno == ERANGE) {
        if (why) *why = fmt("takes a whole number, not '%s'", text ? text : "");
        return false;
    }
    if (n < lo || n > hi) {
        if (why) *why = hi == INT64_MAX ? fmt("must be >= %lld, not %lld", (long long)lo, n)
                                        : fmt("must be in [%lld, %lld], not %lld", (long long)lo,
                                              (long long)hi, n);
        return false;
    }
    *out = n;
    return true;
}

int rad_load_quantizers(Registry& reg, const std::string& home) {
    std::vector<std::string> qdirs;
    for (const std::string& d : rad_home_dirs(home)) {
        struct stat st {};
        if (stat((d + "/quantizers").c_str(), &st) == 0 && S_ISDIR(st.st_mode))
            qdirs.push_back(d + "/quantizers");
    }
    if (qdirs.empty()) qdirs.push_back(rad_home_dirs(home).front() + "/quantizers");
    return reg.load_dirs(qdirs, {});
}

ContainerMeta::ContainerMeta(const RadFile& f) {
    meta.arch_id = f.str(f.header().arch_id);
    meta.name    = f.str(f.header().model_name);
    meta.quant   = f.str(f.header().quant);
    auto mi = [&](const char* k) { int64_t v = 0; f.get_i(k, &v); return v; };
    auto mf = [&](const char* k, double d) { double v = d; f.get_f(k, &v); return (float)v; };
    meta.n_layers = mi("n_layers");  meta.n_embd = mi("n_embd");
    meta.n_head   = mi("n_head");    meta.n_head_kv = mi("n_head_kv");
    meta.head_dim = mi("head_dim");  meta.n_ff = mi("n_ff");
    meta.n_vocab  = mi("n_vocab");   meta.n_ctx_train = mi("n_ctx_train");
    meta.n_expert = mi("n_expert");  meta.n_expert_used = mi("n_expert_used");
    meta.n_expert_shared = mi("n_expert_shared");
    meta.rms_eps = mf("rms_eps", 1e-6); meta.rope_theta = mf("rope_theta", 10000.f);
    meta.rope_scale = 1.f;   /* the rotary's scaling is the free-form rope keys (rad_builder.h) */

    for (int64_t i = 0; i < f.meta_count(); ++i) {
        const RadFileKV& e = f.meta_at(i);
        if (e.type != RAD_P_STR) continue;
        keys_.push_back(f.str(e.key));
        vals_.push_back(f.str(e.v.s));
    }
    for (const std::string& s : keys_) key_ptrs_.push_back(s.c_str());
    for (const std::string& s : vals_) val_ptrs_.push_back(s.c_str());
    meta.n_kv = (int)keys_.size();
    meta.kv_key = key_ptrs_.data();
    meta.kv_val = val_ptrs_.data();
}

const char* rad_flag_value(const char* tool, int argc, char** argv, int* i, const char* flag) {
    if (*i + 1 >= argc) {
        std::fprintf(stderr, "%s: %s needs a value\n", tool, flag);
        std::exit(2);
    }
    return argv[++*i];
}

int64_t rad_flag_int(const char* tool, const char* flag, const char* text, int64_t lo, int64_t hi) {
    int64_t v = 0;
    std::string why;
    if (!rad_parse_int(text, lo, hi, &v, &why)) {
        std::fprintf(stderr, "%s: %s %s\n", tool, flag, why.c_str());
        std::exit(2);
    }
    return v;
}

}  /* namespace rad */
