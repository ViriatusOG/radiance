/* plugin/glm/exports.cpp -- the GLM plugin's export block.
 *
 * Reference status is per row, as in plugin/spechub: the device rows are production kernels
 * (rad_kernel_reference 0) and the host rows beside them their oracle (1), so rad-kbench checks one
 * against the other and the reference gate keeps the host rows out of serving. On a machine with no
 * card only the host rows exist.
 *
 * Every op here is one only glm5next issues, and only where a loaded library declares it, so putting
 * this library first in `server.kernels` changes no other model's selection.
 */
#include "glm.h"
#include "rad_abi_layout.h"

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
    "in one pass",
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

extern "C" int rad_kernel_reference(int i) {
#ifdef RAD_GLM_WITH_DEVICE
    return (i >= 0 && i < RAD_GLM_N_OPS) ? 0 : 1;
#else
    (void)i;
    return 1;
#endif
}

extern "C" int rad_kernel_schema_count(void) { return rad_glm_schema_count(); }
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) { return rad_glm_schema_at(i); }

extern "C" int rad_kernel_count(void) {
#ifdef RAD_GLM_WITH_DEVICE
    return 2 * RAD_GLM_N_OPS;
#else
    return RAD_GLM_N_OPS;
#endif
}

extern "C" const RadKernelInfo* rad_kernel_at(int i) {
#ifdef RAD_GLM_WITH_DEVICE
    static const RadKernelInfo dev[RAD_GLM_N_OPS] = {
        rad_glm_row(0, rad_glm_swiglu_quant_launch),
    };
    if (i >= 0 && i < RAD_GLM_N_OPS) return &dev[i];
    i -= RAD_GLM_N_OPS;
#endif
    static const RadKernelInfo ref[RAD_GLM_N_OPS] = {
        ref_row(0, rad_ref_glm_swiglu_quant, "ref_glm_swiglu_clamped_had_quant_i8g"),
    };
    if (i < 0 || i >= RAD_GLM_N_OPS) return nullptr;
    return &ref[i];
}
