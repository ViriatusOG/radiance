/* Derived from libr4d/r4d_givens_rotate.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// ParoQuant's rotation to f16: the channel scale and Givens stages applied to a bf16 activation,
// for the 16-bit-activation GEMMs (w5a16 / w4a16_asym, rows up to 64, i.e. decode). The weight is
// stored as v = w . D . G_1 ... G_S (abi/rad_encoding.h, "givens"), so with
// x' = x . D^-1 . G_1 ... G_S, <v, x'> = <w, x>: the activation is rotated once a token and the
// GEMM reads the weight unchanged. The quantising form, for prefill-sized rows and MXFP4, is
// halo_givens_dense.hip.
//
// A group at a time: ParoQuant's pairs never leave a group of 128 columns (the checkpoint is
// trained that way, and the converter and loader, halo_rad_relayout_givens_pairs, refuse one that
// does), so a group's 128 values are rotated by S stages of 64 disjoint pairs. A wave owns a group,
// held in LDS (128 floats a wave). No block barrier is needed: a wave's LDS accesses are in order
// and a stage's pairs are disjoint, so no two lanes touch one address. Each lane takes two pairs
// per stage.
//
// Parts: a fused projection (gate|up, q|k|v) is rotated per piece, so a row yields `parts` rotated
// copies side by side: y is [M, parts * K]. A (part, group) is one wave's work (a row spans several
// blocks, grid.y). The tables stack the parts as the "givens" encoding does: `stages` rows each of
// pairs and angle, one row of channel scales.
//
// The channel scale is a true IEEE divide: t.scale is the factor D the stored row was multiplied
// by, and libref also divides in f32, so the two agree exactly on this step. Cos and sin are
// computed here in f32 from the stored angle (sincosf, an ulp or two, as libref's libm).
//
// Returns a negative code instead of throwing, for the reason given in halo_gemm_fp8a8.hip.

#include "halo.h"
#include "halo_emit.h"
#include "halo_fp8_args.h"
#include <hip/hip_fp16.h>

namespace {

constexpr uint32_t kWave = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr uint32_t kGroup = 128;
constexpr uint32_t kPairsPerLane = kGroup / 2 / kWave;      // 2

// kPre stages' tables are loaded and their sines and cosines taken before the first is applied, so
// each stage's dependent chain (table load, LDS read, rotate, LDS write) overlaps its neighbours'
// loads. At decode it is latency bound: a row is 40 groups of 128, each eight serial stages.
constexpr uint32_t kPre = 8;

__device__ __forceinline__ void wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// One wave rotates one (part, group) of one row: grid.x is the row and grid.y spreads the row's
// `parts * nkb` groups over blocks of kWaves waves, so a decode step of a few rows still fills the
// part. A wave with no group returns at once; there is no block barrier. The rotated row is
// written as f16 into `y`, y_ld elements a row.
__global__ __launch_bounds__(kThreads) void givens_rotate_kernel(
    const uint16_t* __restrict__ x, const int32_t* __restrict__ pairs,
    const float* __restrict__ angle, const float* __restrict__ rscale, uint16_t* __restrict__ y,
    uint32_t K, uint32_t nkb, uint32_t parts, uint32_t stages, uint32_t x_ld, uint32_t y_ld) {
    __shared__ float sbuf[kWaves][kGroup];
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const uint32_t item = blockIdx.y * kWaves + wave;
    if (item >= parts * nkb) return;
    const uint32_t part = item / nkb, g = item % nkb;
    const uint32_t c0 = g * kGroup;
    const size_t xrow = (size_t)row * x_ld;
    float* buf = sbuf[wave];

    // channel scale: v = x / D, four consecutive columns a lane
    {
        const float4 d = *(const float4*)(rscale + (size_t)part * K + c0 + lane * 4u);
        const uint32_t xc = c0 + lane * 4u;
        const float v0 = __uint_as_float((uint32_t)x[xrow + xc + 0] << 16) / d.x;
        const float v1 = __uint_as_float((uint32_t)x[xrow + xc + 1] << 16) / d.y;
        const float v2 = __uint_as_float((uint32_t)x[xrow + xc + 2] << 16) / d.z;
        const float v3 = __uint_as_float((uint32_t)x[xrow + xc + 3] << 16) / d.w;
        *(float4*)(buf + lane * 4u) = make_float4(v0, v1, v2, v3);
    }

    for (uint32_t st0 = 0; st0 < stages; st0 += kPre) {
        const uint32_t nst = (stages - st0) < kPre ? (stages - st0) : kPre;
        int2  ab[kPre][kPairsPerLane];
        float th[kPre][kPairsPerLane], cs[kPre][kPairsPerLane], sn[kPre][kPairsPerLane];
#pragma unroll
        for (uint32_t k = 0; k < kPre; ++k)
            if (k < nst) {
                const size_t trow = (size_t)part * stages + st0 + k;
                const int32_t* pr = pairs + trow * K + c0;
                const float* an = angle + trow * (K / 2u) + g * (kGroup / 2u);
#pragma unroll
                for (uint32_t u = 0; u < kPairsPerLane; ++u) {
                    const uint32_t j = lane + u * kWave;
                    ab[k][u] = *(const int2*)(pr + 2u * j);
                    th[k][u] = an[j];
                }
            }
#pragma unroll
        for (uint32_t k = 0; k < kPre; ++k)
            if (k < nst)
#pragma unroll
                for (uint32_t u = 0; u < kPairsPerLane; ++u) sincosf(th[k][u], &sn[k][u], &cs[k][u]);
        wave_sync();            // the scaled row (or the previous chunk's last stage) is in LDS
#pragma unroll
        for (uint32_t k = 0; k < kPre; ++k)
            if (k < nst) {
#pragma unroll
                for (uint32_t u = 0; u < kPairsPerLane; ++u) {
                    const int ia = ab[k][u].x - (int)c0, ib = ab[k][u].y - (int)c0;
                    const float va = buf[ia], vb = buf[ib];
                    buf[ia] = fmaf(cs[k][u], va, sn[k][u] * vb);
                    buf[ib] = fmaf(cs[k][u], vb, -sn[k][u] * va);
                }
                wave_sync();
            }
    }

    const float4 r = *(const float4*)(buf + lane * 4u);
    const float v[4] = { r.x, r.y, r.z, r.w };
    // f16 holds +-65504; a rotated post-norm row is far below that, and the clamp keeps an outlier
    // finite.
    union { __half h[4]; uint2 u; } o;
#pragma unroll
    for (int j = 0; j < 4; ++j) o.h[j] = __float2half(fminf(fmaxf(v[j], -65504.0f), 65504.0f));
    *(uint2*)(y + (size_t)row * y_ld + (size_t)part * K + c0 + lane * 4u) = o.u;
}

}  // namespace

//   x       [M][K]               bf16, `x_ld` elements a row (0 = K)
//   pairs   [parts*stages][K]    int32 column indices; (pairs[2j], pairs[2j+1]) is pair j
//   angle   [parts*stages][K/2]  f32
//   rscale  [parts][K]           f32, the factor the stored row was multiplied by
//   y       [M][parts*K]         f16
enum { GV_X = 0, GV_PAIRS = 1, GV_ANGLE = 2, GV_RSCALE = 3, GV_Y = 4 };

extern "C" int halo_rad_givens_rotate(const RadArgs* a, RadStream stream) {
    const RadTensor* x  = halo_opt(a, GV_X);
    const RadTensor* pr = halo_opt(a, GV_PAIRS);
    const RadTensor* an = halo_opt(a, GV_ANGLE);
    const RadTensor* rs = halo_opt(a, GV_RSCALE);
    const RadTensor* y  = halo_opt(a, GV_Y);
    if (!x || !pr || !an || !rs || !y) return HALO_NO(RAD_E_INVAL);
    if (x->rank < 2 || y->rank != 2 || pr->rank != 2 || an->rank != 2 || rs->rank != 2)
        return HALO_NO(RAD_E_SHAPE);
    if (y->dtype != RAD_F16) return HALO_NO(RAD_E_DTYPE);
    if (pr->dtype != RAD_I32 || an->dtype != RAD_F32 || rs->dtype != RAD_F32)
        return HALO_NO(RAD_E_DTYPE);
    // rows are dim 0 and everything after it is the row
    const long long M = x->shape[0], K = halo_numel(x) / (x->shape[0] > 0 ? x->shape[0] : 1);
    const long long grp = (long long)kGroup, nkb = K / grp;
    const long long parts = halo_geti_or(a, "parts", 1), stages = halo_geti_or(a, "stages", 0);
    if (M <= 0 || halo_geti_or(a, "group", grp) != grp || K % grp || parts < 1 || stages < 1)
        return HALO_NO(RAD_E_SHAPE);
    if (y->shape[0] != M || y->shape[1] != parts * K) return HALO_NO(RAD_E_SHAPE);
    // the tables are the weight's, contiguous: parts * stages rows of K pairs and K/2 angles
    if (pr->shape[0] != parts * stages || pr->shape[1] != K) return HALO_NO(RAD_E_SHAPE);
    if (an->shape[0] != parts * stages || an->shape[1] != K / 2) return HALO_NO(RAD_E_SHAPE);
    if (rs->shape[0] != parts || rs->shape[1] != K) return HALO_NO(RAD_E_SHAPE);
    // A lane stores four f16 (8 bytes) at a time, so y's base and row pitch keep 8-byte alignment.
    if (!halo_aligned(rs->data, 16) || !halo_aligned(pr->data, 8) || !halo_aligned(y->data, 8))
        return HALO_NO(RAD_E_ALIGN);
    const long long xp = halo_row_pitch(x), yp = halo_row_pitch(y);
    if (xp < 0 || yp < 0) return HALO_NO(RAD_E_STRIDE);
    if (yp % 4) return HALO_NO(RAD_E_ALIGN);
    const uint32_t gy = ((uint32_t)(parts * nkb) + kWaves - 1) / kWaves;
    return HALO_EMIT(stream, givens_rotate_kernel, dim3((uint32_t)M, gy), dim3(kThreads), 0,
                    (const uint16_t*)x->data, (const int32_t*)pr->data, (const float*)an->data,
                    (const float*)rs->data, (uint16_t*)y->data, (uint32_t)K, (uint32_t)nkb,
                    (uint32_t)parts, (uint32_t)stages, (uint32_t)(xp ? xp : K),
                    (uint32_t)(yp ? yp : parts * K));
}
