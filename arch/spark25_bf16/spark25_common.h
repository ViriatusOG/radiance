/* What Spark's two trunk plugins (spark25_bf16, spark25_w8a8) share: the drafter check, the layer
 * schedule and the tensor-parallel split of a ragged GQA / FFN world. */
#pragma once
#include <arch/rad_arch.h>

#include <sstream>
#include <string>
#include <vector>

namespace spark25_common {
using namespace rad::arch;

/* Whether the container carries a drafter at all (`--draft-model` merged one under `draft.`), and
 * whether the plugin implements it. A drafter it cannot run is refused at declare rather
 * than left unused: the container would serve at the speed of a model with no drafter, and the
 * user would have no way to find out that the flag they passed did nothing. */
inline int drafter_kind(const RadModelMeta* meta, bool* present) {
    *present = false;
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    if (!da || !*da) {
        /* A drafter's config.json flattened, or a GGUF import, either of which names it. */
        if (rad_meta_geti(meta, "draft.num_hidden_layers", 0) <= 0) return RAD_OK;
        da = rad_meta_gets(meta, "draft.model_type", "");
    }
    if (std::strstr(da, "DSpark") || std::strcmp(da, "dspark") == 0) {
        *present = true;
        return RAD_OK;
    }
    *present = true;
    fprintf(stderr, "radiance: spark25: this container carries a '%s' drafter. Spark's drafts are "
                    "DSpark, with a rank-N Markov head over the target's own table; this "
                    "architecture implements that and nothing else\n", da);
    return RAD_E_UNSUPPORTED;
}

/* The HF config names every layer. Do not infer a periodic schedule: it changes the mask and
 * rotary of a layer and would silently produce a different model. */
inline int schedule(const RadModelMeta* meta, int64_t n, std::vector<bool>& local) {
    const char* s = rad_meta_gets(meta, "layer_types", nullptr);
    if (!s) {
        fprintf(stderr, "radiance: spark25: layer_types is required\n");
        return RAD_E_INVAL;
    }
    std::istringstream in(s);
    std::string word;
    while (in >> word) {
        if (word != "sliding_attention" && word != "full_attention") {
            fprintf(stderr, "radiance: spark25: unknown layer type '%s'\n", word.c_str());
            return RAD_E_UNSUPPORTED;
        }
        local.push_back(word == "sliding_attention");
    }
    if ((int64_t)local.size() != n) {
        fprintf(stderr, "radiance: spark25: layer_types has %zu entries for %lld layers\n",
                local.size(), (long long)n);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

struct Shards {
    Span q, kv, ff;  /* q and kv in heads; ff in channels */
};

/* A rank's attention heads must describe whole GQA groups, or a slice of a single group with
 * that group's KV head replicated. This also covers non-divisible worlds such as 3 ranks over
 * the checkpoint's 4 KV heads. The returned spans give the loader those exact cuts. */
inline int shard_geometry(Geom& g, const RadBuildCtx* ctx, Shards& s) {
    g.world = ctx->world_size > 0 ? ctx->world_size : 1;
    g.rank = ctx->rank;
    if (g.rank >= g.world || g.world > g.n_head_all || g.n_head_kv_all <= 0 ||
        g.n_head_all % g.n_head_kv_all || g.world > g.n_ff_all) {
        fprintf(stderr, "radiance: spark25: cannot shard %lld query heads, %lld KV heads and "
                        "%lld FFN channels over %d ranks\n", (long long)g.n_head_all,
                (long long)g.n_head_kv_all, (long long)g.n_ff_all, g.world);
        return RAD_E_INVAL;
    }
    const int64_t group = g.n_head_all / g.n_head_kv_all;
    if (g.world <= g.n_head_kv_all) {
        s.kv = split_span(g.n_head_kv_all, g.world, g.rank);
        s.q = {s.kv.first * group, s.kv.count * group};
    } else {
        int first_rank = 0;
        for (int64_t k = 0; k < g.n_head_kv_all; ++k) {
            const int n = (int)split_span(g.world, (int)g.n_head_kv_all, (int)k).count;
            if (g.rank < first_rank + n) {
                const Span q = split_span(group, n, g.rank - first_rank);
                s.kv = {k, 1};
                s.q = {k * group + q.first, q.count};
                break;
            }
            first_rank += n;
        }
    }
    /* Prefer whole 128-channel blocks for a ragged split, keeping device GEMMs aligned. */
    const int64_t unit = g.n_ff_all % 128 == 0 && g.n_ff_all / 128 >= g.world ? 128 : 1;
    s.ff = split_span(g.n_ff_all / unit, g.world, g.rank);
    s.ff.first *= unit;
    s.ff.count *= unit;
    g.n_head = s.q.count;
    g.n_head_kv = s.kv.count;
    g.n_ff = s.ff.count;
    g.n_vocab = shard_extent(g.n_vocab_all, g.world, g.rank);
    g.vocab_off = shard_begin(g.n_vocab_all, g.world, g.rank);
    return g.n_head > 0 && g.n_head_kv > 0 && g.n_ff > 0 && g.n_vocab > 0 ? RAD_OK : RAD_E_INVAL;
}

inline bool needs_span(const Span& s, int64_t total, const Geom& g) {
    return total % g.world != 0 || s.first != g.rank * (total / g.world) ||
           s.count != total / g.world;
}

}  // namespace spark25_common
