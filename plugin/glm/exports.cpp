/* plugin/glm/exports.cpp -- the GLM plugin's export block.
 *
 * Reference status is per row, as in plugin/spechub: the device rows are production kernels
 * (rad_kernel_reference 0) and the host rows beside them their oracle (1), so rad-kbench checks one
 * against the other and the reference gate keeps the host rows out of serving. On a machine with no
 * card only the host rows exist.
 *
 * Its own op, swiglu_clamped_had_quant_i8g, is issued only where a loaded library declares it, and
 * its mla_attn rows serve only `kv_dtype` `i8`, which no other row serves, so putting this library
 * first in `server.kernels` changes no other model's selection. The plugin declines a machine whose
 * cards are all another family's (rad_plugin_open, below).
 */
#include "glm.h"
#include "rad_abi_layout.h"

#include <dirent.h>
#include <cstdio>
#include <cstring>
#include <string>

#ifdef RAD_GLM_WITH_DEVICE
#include "../../libr4d/r4d_emit.h"   /* only for the three globals it declares */

/* r4d_emit.h's dispatch recorder, defined here because plugins load RTLD_LOCAL and cannot resolve
 * libr4d's copy. Nothing in this plugin sets the log, so the recorder stays inert. */
const void** r4d_dispatch_log = nullptr;
int          r4d_dispatch_cap = 0;
int          r4d_dispatch_n   = 0;
#endif

namespace {
const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "glm", "0.1.0",
    "GLM-5.3-Flash's own kernels beside libr4d: the clamped SwiGLU and its rotated int8 quantiser "
    "in one pass, and mla_attn over an int8 latent cache",
    "gfx1201",
};

RadKernelInfo ref_row(int op, RadLaunchFn launch, const char* name) {
    RadKernelInfo x = rad_glm_row(op, launch);
    x.name = name;
    x.domain = RAD_DOMAIN_HOST;
    x.priority = 0;
    return x;
}
}  /* namespace */

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
extern "C" int rad_plugin_reference(void) { return 0; }

/* The plugin declines a machine whose cards are all another family's, as libr4d does
 * (r4d_rows.cpp, rad_plugin_open). Its device rows are gfx12 code; where no code object fits, the
 * loader leaves them out, but the schema would stay declared, and GateQuant asks
 * rad_op_declared before it issues swiglu_clamped_had_quant_i8g: on such a card a model would
 * pick the fused form and fail at startup with the op unserved (the host row is a reference row,
 * kept out of serving). The test reads the KFD topology (gfx_target_version 120000..129999 is the
 * gfx12 family) rather than asking a runtime, which open must not initialise. A machine where no
 * card is listed (no topology, or a container given no /dev/kfd) keeps the plugin: rad-convert and
 * the host tests load it there. `root` is the topology's nodes directory, a parameter so a test
 * can hand it a fake tree. */
extern "C" int rad_glm_machine_check(const char* root) {
    DIR* d = opendir(root);
    if (!d) return RAD_OK;
    bool card = false, gfx12 = false;
    while (const dirent* e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        const std::string path = std::string(root) + "/" + e->d_name + "/properties";
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) continue;
        char key[64];
        unsigned long long v = 0;
        while (std::fscanf(f, "%63s %llu", key, &v) == 2) {
            if (std::strcmp(key, "gfx_target_version") || v == 0) continue;   /* 0: a CPU node */
            card = true;
            if (v >= 120000ull && v < 130000ull) gfx12 = true;
        }
        std::fclose(f);
    }
    closedir(d);
    return card && !gfx12 ? RAD_E_UNSUPPORTED : RAD_OK;
}

extern "C" int rad_plugin_open(void) {
    return rad_glm_machine_check("/sys/class/kfd/kfd/topology/nodes");
}

extern "C" int rad_kernel_reference(int i) {
#ifdef RAD_GLM_WITH_DEVICE
    return (i >= 0 && i < RAD_GLM_N_DEV) ? 0 : 1;
#else
    (void)i;
    return 1;
#endif
}

/* Which rows chain (RadKernelChainFn): the device swiglu quantiser, whose chained form waits in
 * r4d_chain_wait and so acquires for its scalar loads itself (2). */
extern "C" int rad_kernel_chain(int i) {
#ifdef RAD_GLM_WITH_DEVICE
    return i == 0 ? 2 : 0;
#else
    (void)i;
    return 0;
#endif
}

extern "C" int rad_kernel_schema_count(void) { return rad_glm_schema_count(); }
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) { return rad_glm_schema_at(i); }

extern "C" int rad_kernel_count(void) {
#ifdef RAD_GLM_WITH_DEVICE
    return RAD_GLM_N_DEV + RAD_GLM_N_REF;
#else
    return RAD_GLM_N_REF;
#endif
}

extern "C" const RadKernelInfo* rad_kernel_at(int i) {
#ifdef RAD_GLM_WITH_DEVICE
    static const RadKernelInfo dev[RAD_GLM_N_DEV] = {
        rad_glm_row(0, rad_glm_swiglu_quant_launch),
        rad_glm_mla_i8_row(rad_glm_mla_i8_launch, rad_glm_mla_i8_scratch),
    };
    if (i >= 0 && i < RAD_GLM_N_DEV) return &dev[i];
    i -= RAD_GLM_N_DEV;
#endif
    static const RadKernelInfo ref[RAD_GLM_N_REF] = {
        ref_row(0, rad_ref_glm_swiglu_quant, "ref_glm_swiglu_clamped_had_quant_i8g"),
    };
    if (i < 0 || i >= RAD_GLM_N_REF) return nullptr;
    return &ref[i];
}
