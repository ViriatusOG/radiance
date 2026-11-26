/* qwen35_w8a8 -- Qwen3.5 / 3.6 / 3.8 dense with an int8 trunk: i8 weights with a bf16 scale a 128
 * of K, rotated by a 128-wide Hadamard, against int8 activations quantised under the same rotation.
 *
 * This is arch/qwen35_fp8/qwen35_fp8.cpp built again, as qwen35moe_fp8 is: the same gated
 * attention, gated delta net, MLP, layer schedule, rope, vocabulary edges, drafters and vision
 * tower. A linear's weight encoding makes it int8 (arch/common/rad_linear.h: LinearQ):
 * `i8*bf16[1x128]/fwht128` is gemm_nt_q at i8a8h behind had_quant_act_i8g, each linear quantising
 * its own input from the bf16 plane its producer already writes. The places this build differs are
 * behind `#if QWEN35_W8A8` in that file.
 *
 * A plugin of its own rather than qwen35_fp8 serving any encoding it is handed, because the
 * quantisation descriptor is half of the selection key (spec §2.4): a container converted with
 * data/recipes/q38-27b-w8a8-df2.recipe says `w8a8_fwht128`.
 */
#define QWEN35_W8A8    1
#define QWEN35_NS      qwen35_w8a8
#define QWEN35_ARCH_ID "qwen35"

#include "../qwen35_fp8/qwen35_fp8.cpp"
