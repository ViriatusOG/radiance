#pragma once

#include "r3d_plugin.h"
#include <iterator>

namespace r3d_i8_gemm {

inline constexpr RadParamSpec params[]{
    {"M", RAD_P_INT, RAD_REQUIRED}, {"N", RAD_P_INT, RAD_REQUIRED}, {"K", RAD_P_INT, RAD_REQUIRED},
    {"group", RAD_P_INT, RAD_REQUIRED}, {"dtype", RAD_P_STR, RAD_REQUIRED}, {"pb1", RAD_P_INT, RAD_OPTIONAL},
    {"pb2", RAD_P_INT, RAD_OPTIONAL}, {"split_n", RAD_P_INT, RAD_OPTIONAL}, {"w_rb", RAD_P_INT, RAD_OPTIONAL}};
inline constexpr RadOperandSpec operands[]{
    {"a", RAD_OPD_IN, 0}, {"a_scale", RAD_OPD_IN, 1}, {"b", RAD_OPD_WEIGHT, 0}, {"b_scale", RAD_OPD_WEIGHT, 0},
    {"y", RAD_OPD_OUT, 0}, {"a_sum", RAD_OPD_IN, 1}, {"b_ref", RAD_OPD_WEIGHT, 1}};

inline constexpr RadConstraint native_constraints[]{
    {"dtype", RAD_C_IN, 0, "i8a8 i8a8h"}, {"group", RAD_C_EQ, R3D_W8_GROUP}, {"M", RAD_C_GE, 1},
    {"K", RAD_C_DIV, 128}, {"N", RAD_C_GE, 1}};
inline constexpr RadKernelInfo native{
    "gemm_nt_q_i8a8", "gemm_nt_q", "gemm",
    "the int8 trunk grid (i8a8, and i8a8h, its rotated twin): signed int8 codes against row-major "
    "int8 activations, the group's integer sub-sum scaled by the activation's (row, group) scale "
    "and the weight's (row, group) scale",
    "group 128, K a multiple of it, every M; below M = 128 a wave is a 16-column tile over one "
    "16-row tile (M <= 16) or two, the weight read from global at each step; from M = 128 a block "
    "is a 128x128 output tile with a group of both operands staged in LDS, so a prefill chunk "
    "reads the weight once per 128 rows",
    "I8 a row-major with an f32 scale per (row, 128 of K) -- quant_act_i8g's output, or "
    "had_quant_act_i8g's at i8a8h; I8 codes b with a bf16 scale per (row, 128 of K), bf16 y, fp32 "
    "accumulation in ascending group order; the sub-sums are exact integers under 2^24, and the "
    "store is the only narrowing; each group term rounds as quant_dot's (two multiplies, an add, "
    "none fused), so the row is bit-identical to the reference; no zero point, so both sums are absent",
    RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8, native_constraints, std::size(native_constraints), nullptr, 0, nullptr,
    nullptr, nullptr, r3d_rad_gemm_nt_q_i8a8, nullptr, nullptr, nullptr, r3d_shape_gemm_nt_q_i8a8};

#ifdef R3D_DENSE_IQR
extern "C" int r3d_rad_gemm_iqr(const RadArgs*, RadStream);
extern "C" int r3d_shape_gemm_iqr(const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_rad_layout_iqr(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int, RadLayout*);
extern "C" int r3d_rad_relayout_iqr(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int, void*, int64_t);
extern "C" int r3d_rad_unrelayout_iqr(const RadParam*, int, int, const RadEncoding*, const int*, const void*, int64_t, const RadTensor*, int);
inline constexpr RadConstraint packed_constraints[]{
    {"dtype", RAD_C_IN, 0, "i8a8h"}, {"group", RAD_C_EQ, 128}, {"M", RAD_C_GE, 1}, {"M", RAD_C_LE, 2048},
    {"N", RAD_C_GE, 128}, {"N", RAD_C_DIV, 128}, {"K", RAD_C_GE, 128}, {"K", RAD_C_DIV, 128}};
inline constexpr RadKernelInfo packed{
    "gemm_nt_q_iqr", "gemm_nt_q", "gemm", "Computed-trellis IQR with exact integer group sums",
    "u3/u4/u5, L16, run 128 or 256, BF16 group scales or factored F32 scales",
    "rotated int8 activations against packed weights, bf16 output",
    RAD_DOMAIN_DEVICE, 1, packed_constraints, std::size(packed_constraints), nullptr, 0, nullptr,
    nullptr, nullptr, r3d_rad_gemm_iqr, nullptr, r3d_rad_layout_iqr, r3d_rad_relayout_iqr,
    r3d_shape_gemm_iqr, r3d_rad_unrelayout_iqr};
#endif
}
