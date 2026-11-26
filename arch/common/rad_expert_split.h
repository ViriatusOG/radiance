/* rad_expert_split.h -- how many routed experts each rank holds, for a model that gives each rank
 * whole experts (expert parallelism: MoeQ::Config::n_expert and expert_base).
 *
 * By default every rank holds n_expert / world. `list` (RadBuildCtx::expert_split, the
 * `expert-split` setting) gives every rank's count instead, comma-separated in rank order
 * ("108,84,108,84"): whole experts, in steps of `step` (2 where the block's quantised GEMM takes two
 * tables by parity, which is what MoeQ does), summing to the model's. The router is replicated
 * and each rank sorts only the slots in its own range, so the ranks need not hold equal runs.
 *
 * It is for a machine whose cards are not on equal links: a rank given few enough experts holds
 * all of them in VRAM, so none is streamed to it over its link, and the ranks on the faster links
 * take the rest. What the counts should be depends on the links and on what a card holds, so the
 * deployment states them.
 *
 * A list that does not fit the model or the world is refused, not ignored: the function returns
 * false with *given true, and the caller stops the start, naming the list. A rank at or past
 * `world` (a shape probe) takes rank 0's run. */
#pragma once
#include <cstdint>
#include <cstdlib>
#include <vector>

inline bool rad_expert_split(int64_t n_expert, int world, int rank, int step, const char* list,
                             int64_t* n, int64_t* base, bool* given) {
    *given = false;
    if (step < 1) step = 1;
    if (rank < 0 || rank >= world) rank = 0;
    if (!list || !*list || world <= 1) {
        *n = world > 0 ? n_expert / world : n_expert;
        *base = rank * *n;
        return world <= 1 || (n_expert % world == 0 && *n % step == 0);
    }
    *given = true;
    std::vector<int64_t> v;
    for (const char* p = list; *p;) {
        char* end = nullptr;
        const long long x = std::strtoll(p, &end, 10);
        if (end == p) return false;
        v.push_back(x);
        if (*end && *end != ',') return false;
        p = *end == ',' ? end + 1 : end;
    }
    int64_t sum = 0;
    for (int64_t x : v) {
        if (x < step || x % step) return false;
        sum += x;
    }
    if ((int)v.size() != world || sum != n_expert) return false;
    *n = v[(size_t)rank];
    *base = 0;
    for (int r = 0; r < rank; ++r) *base += v[(size_t)r];
    return true;
}
