/* avx_subset_plugin.cpp -- a host kernel library made of part of libref, for avx_check_general.
 *
 * rad-avx-check holds libavx to its own contract and checks any other host library as one: the
 * ops it lacks are listed rather than failed, each case runs on the row whose constraints admit it,
 * and the ISA sweep, the threaded rerun and the file-backed gather are left out. This plugin is
 * such a library. It loads libref (the path in RAD_SUBSET_OF) and re-exports its schemas and its
 * rows, minus three ops, with rmsnorm limited to f32 and bf16 by a constraint. With
 * RAD_SUBSET_DECLINE=<op> that op's row returns RAD_E_UNSUPPORTED for every launch, which the
 * checker must fail in any library. */
#include "rad_abi.h"
#include "rad_abi_layout.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

const char* const kDropped[] = { "softmax", "sample_merge_topk", "dflash_select" };
const RadConstraint kRmsnormDtypes[] = { { "dtype", RAD_C_IN, 0, "f32 bf16" } };

struct Subset {
    int (*schema_count)() = nullptr;
    const RadOpSchema* (*schema_at)(int) = nullptr;
    std::vector<RadKernelInfo> rows;
};

int declined(const RadArgs*, RadStream) { return RAD_E_UNSUPPORTED; }

const Subset& subset() {
    static const Subset s = [] {
        Subset r;
        const char* path = std::getenv("RAD_SUBSET_OF");
        void* h = path ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : nullptr;
        if (!h) {
            std::fprintf(stderr, "avx_subset_plugin: set RAD_SUBSET_OF to libref.so\n");
            return r;
        }
        auto count = (int (*)())dlsym(h, "rad_kernel_count");
        auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        r.schema_count = (int (*)())dlsym(h, "rad_kernel_schema_count");
        r.schema_at = (const RadOpSchema* (*)(int))dlsym(h, "rad_kernel_schema_at");
        if (!count || !at || !r.schema_count || !r.schema_at) return r;
        const char* decline = std::getenv("RAD_SUBSET_DECLINE");
        for (int i = 0, n = count(); i < n; ++i) {
            RadKernelInfo k = *at(i);
            bool drop = false;
            for (const char* d : kDropped) drop = drop || !std::strcmp(k.op, d);
            if (drop) continue;
            if (!std::strcmp(k.op, "rmsnorm")) {
                k.constraints = kRmsnormDtypes;
                k.n_constraints = 1;
            }
            if (decline && !std::strcmp(k.op, decline)) k.launch = declined;
            r.rows.push_back(k);
        }
        return r;
    }();
    return s;
}

}  // namespace

extern "C" {

uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT

const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = { RAD_PLUGIN_KERNEL, "subset", "0.1.0",
                                        "part of libref, for avx_check_general", "host" };
    return &info;
}

int rad_kernel_schema_count(void) {
    return subset().schema_count ? subset().schema_count() : 0;
}
const RadOpSchema* rad_kernel_schema_at(int i) {
    return subset().schema_at ? subset().schema_at(i) : nullptr;
}
int rad_kernel_count(void) { return (int)subset().rows.size(); }
const RadKernelInfo* rad_kernel_at(int i) {
    const std::vector<RadKernelInfo>& r = subset().rows;
    return i >= 0 && i < (int)r.size() ? &r[(size_t)i] : nullptr;
}

}  // extern "C"
