/* qwen35moe_w8a8 -- Qwen3.5 / 3.6 MoE (35B-A3B) with an int8 trunk and int8 experts.
 *
 * arch/qwen35_fp8/qwen35_fp8.cpp built a fourth time, with both of its switches: the routed
 * feed-forward of qwen35moe_fp8 and the rotated int8 trunk of qwen35_w8a8. Every linear of the
 * trunk is IQ8R (`i8` codes rotated by a 128-wide Hadamard, read at i8a8h against int8
 * activations), and the routed experts are rotated int8 at a bf16 scale a (row, 128 of K) or one
 * of the rotated four-bit forms, read by the grouped GEMM's int8 rows from the codes the trunk's
 * own quantisers write (arch/common/rad_block_moe_q.h, MoeQ::a8).
 *
 * A plugin of its own for the reason qwen35_w8a8 is: the quantisation descriptor is half of the
 * selection key (spec §2.4), and a container converted with data/recipes/q36-35b-a3b-iq8r.recipe
 * says `qwen35moe` and `w8a8_fwht128`.
 */
#define QWEN35_MOE     1
#define QWEN35_W8A8    1
#define QWEN35_NS      qwen35moe_w8a8
#define QWEN35_ARCH_ID "qwen35moe"

#include "../qwen35_fp8/qwen35_fp8.cpp"
