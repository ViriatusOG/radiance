#ifdef R3D_IQR_SHIPPED_CATALOGUE
extern "C" int rad_plugin_open() { return 0; }
#else
#include "r3d_i8_gemm_rows.h"
#include "rad_abi_layout.h"

extern "C" {
int r3d_rad_gemm_nt_q_i8a8(const RadArgs*, RadStream) { return RAD_E_UNSUPPORTED; }
int r3d_rad_gemm_iqr(const RadArgs*, RadStream) { return RAD_E_UNSUPPORTED; }

uint32_t rad_plugin_abi_version() { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
const RadPluginInfo* rad_plugin_info() {
    static const RadPluginInfo info{RAD_PLUGIN_KERNEL, "r3d_iqr_fixture", "1", "dense declaration only", "host"};
    return &info;
}
int rad_kernel_schema_count() { return 1; }
const RadOpSchema* rad_kernel_schema_at(int index) {
    using namespace r3d_i8_gemm;
    static const RadOpSchema schema{"gemm_nt_q", params, std::size(params), operands, std::size(operands), ""};
    return index == 0 ? &schema : nullptr;
}
int rad_kernel_count() { return 2; }
const RadKernelInfo* rad_kernel_at(int index) {
    return index == 0 ? &r3d_i8_gemm::packed : index == 1 ? &r3d_i8_gemm::native : nullptr;
}
}
#endif
