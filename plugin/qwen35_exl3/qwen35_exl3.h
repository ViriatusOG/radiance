/* qwen35_exl3.h -- declarations and static structures for qwen35_exl3 architecture plugin.
 * EXL3 weights dequantized on-the-fly per rank scratch -> BF16 GEMM.
 */
#ifndef QWEN35_EXL3_H
#define QWEN35_EXL3_H

#include "rad_abi.h"
#include "rad_builder.h"
#include "rad_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

uint32_t             rad_plugin_abi_version(void);
const RadPluginInfo* rad_plugin_info(void);
const char*          rad_arch_id(void);
const char*          rad_arch_quant(void);
int                  rad_arch_declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx);
void                 rad_arch_step(RadCtx* c, const RadBatch* batch);

#ifdef __cplusplus
}
#endif

#endif /* QWEN35_EXL3_H */
