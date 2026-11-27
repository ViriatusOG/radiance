/* r4d_ds41_ggml_mr64.hip -- the multi-row decode-band kernels (r4d_ds41_ggml_mr.hip) compiled wave64
 * (CMakeLists: -mwavefrontsize64). A 64-lane wave runs two of the source's 32-lane waves: lane = threadIdx & 31,
 * wave = threadIdx >> 5 as written, and every cross-lane step stays inside a half (DPP row_xmask inside a row of
 * 16, v_permlanex16 between the two rows of a half), so each output's operations are the wave32 kernel's. The
 * point is the VALU: RDNA4 can issue a wave64 f32 mul / add over its dual ALUs in one pass, where wave32 needs
 * VOPD pairs the compiler finds for about a quarter of the slots. Reached only through r4d_gg_mr_pin_w64 (bench). */
#define R4D_GG_MR_W64 1
#include "r4d_ds41_ggml_mr.hip"
