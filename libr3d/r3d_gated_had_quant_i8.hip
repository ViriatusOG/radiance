/* r3d_gated_had_quant_i8.hip -- gated_had_quant_i8: an elementwise gated product, then the block
 * Hadamard, then the per-row symmetric int8 quantiser, in one pass.
 *
 * Two of the four rotated sites have this shape and differ by one multiply:
 *   mode 0, down_proj's input   silu(a) * b  =  a * sigmoid(a) * b   (vLLM's silu_and_mul)
 *   mode 1, o_proj's input      sigmoid(a) * b                       (attn_output * sigmoid(gate))
 *
 * `a` is the gate and `b` the up; when `b` is absent it aliases `a + n` in a fused gate_up of row
 * pitch 2*n. The transform is the wave-local FWHT of r3d_fwht.h; normalisation is folded into the
 * scale. The codes are written in the int8 WMMA step order the w4a8/w8a8/w2a8 GEMMs read.
 *
 * silu is computed in fp32: the product feeds a rotation that mixes n values, so a bf16 rounding
 * error in one lands on all of them.
 *
 * gfx1102: the FWHT's cross-lane stages are DPP, and a DPP exchange in a loop body miscompiles
 * here (a non-first wave drops out -- the same bug as the GDN recurrent kernel, the MoE router,
 * and had_quant_act_i8). This serves the single-block form, n == group: one wave runs the whole
 * FWHT with no block loop. A wider n is declined.
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3d_fwht.h"
#include <cmath>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr float kTopI8 = 127.0f;
constexpr float kI8Floor = 1e-8f;

__device__ __forceinline__ float ld_bf(const uint16_t* p, uint32_t i) {
    return bf16_to_f32((uint16_t)ldg_u16(p + i));
}
__device__ __forceinline__ float clamp_top(float c, float top) {
    if (c > top) c = top;
    if (c < -top) c = -top;
    return c;
}
__device__ __forceinline__ uint32_t i8_codes4(const float* v, float inv) {
    const uint32_t b0 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[0] * inv, kTopI8));
    const uint32_t b1 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[1] * inv, kTopI8));
    const uint32_t b2 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[2] * inv, kTopI8));
    const uint32_t b3 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[3] * inv, kTopI8));
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}
__device__ __forceinline__ uint32_t i8_step_pos(uint32_t c) {
    return (c & ~12u) | ((c & 4u) << 1) | ((c & 8u) >> 1);
}
__device__ __forceinline__ float block_max(float v, float* lds) {
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    v = wave_max(v);
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float t = lds[0];
#pragma unroll
    for (uint32_t w = 1; w < kWaves; ++w) t = fmaxf(t, lds[w]);
    return t;
}

template <int HAD, int MODE>
__global__ __launch_bounds__(kThreads) R3D_WAVES(kWaves) void gated_had_quant_i8_kernel(
    const uint16_t* __restrict__ g, const uint16_t* __restrict__ u, uint8_t* __restrict__ q,
    float* __restrict__ s, uint32_t n, uint32_t g_ld, uint32_t u_ld, float rnorm) {
    constexpr int VPL = HAD / 32;
    __shared__ float lds[1024];
    __shared__ float red[kWaves];
    const uint32_t row = blockIdx.x;
    const uint16_t* gp = g + (size_t)row * g_ld;
    const uint16_t* up = u + (size_t)row * u_ld;
    uint8_t* qr = q + (size_t)row * n;
    const uint32_t lane = threadIdx.x & 31, wave = threadIdx.x >> 5;

    float m = 0.0f;
    if (wave == 0) {   // one wave does the whole single-block FWHT (no block loop, gfx1102)
        const int base = lane * VPL;
        float v[VPL];
#pragma unroll
        for (int j = 0; j < VPL; ++j) {
            const float xg = ld_bf(gp, base + j);
            const float sg = 1.0f / (1.0f + expf(-xg));   /* full exp: the reference's sigmoid */
            v[j] = (MODE == 0 ? xg * sg : sg) * ld_bf(up, base + j);
        }
        r3d_fwht_wave<HAD>(v, lane);
#pragma unroll
        for (int j = 0; j < VPL; ++j) { lds[base + j] = v[j]; m = fmaxf(m, fabsf(v[j])); }
    }
    m = block_max(m, red);
    m = fmaxf(m, kI8Floor);
    const float inv = kTopI8 / m;
    if (threadIdx.x == 0) s[row] = m * rnorm / kTopI8;

    for (uint32_t c = threadIdx.x * 4; c < n; c += kThreads * 4) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = (c + (uint32_t)j < n) ? lds[c + (uint32_t)j] : 0.0f;
        *(uint32_t*)(qr + i8_step_pos(c)) = i8_codes4(v, inv);
    }
}

}  // namespace

extern "C" {

int r3d_rad_gated_had_quant_i8(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* gate = r3d_opt(a, 0), *up = r3d_opt(a, 1), *q = r3d_opt(a, 2),
                    *sc = r3d_opt(a, 3);
    if (!r3d_p(gate) || !r3d_wp(q) || !r3d_wp(sc)) return R3D_NO(RAD_E_INVAL);
    if (gate->dtype != RAD_BF16 || q->dtype != RAD_I8 || sc->dtype != RAD_F32)
        return R3D_NO(RAD_E_DTYPE);

    long long M = 0, n = 0;
    if (r3d_rows_of(q, &M, &n) != 0) return R3D_NO(RAD_E_INVAL);
    if (n <= 0 || n % 16) return R3D_NO(RAD_E_SHAPE);

    // `b` (up) is absent only in the fused form, where it aliases gate + n in a [M, 2n] gate_up.
    const uint32_t g_ld = (uint32_t)(gate->rank >= 2 ? gate->shape[1] : gate->stride[0]);
    uint32_t u_ld = g_ld;
    const uint16_t* up_p;
    if (up) {
        if (up->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
        u_ld = (uint32_t)(up->rank >= 2 ? up->shape[1] : up->stride[0]);
        up_p = (const uint16_t*)r3d_p(up);
    } else {
        up_p = (const uint16_t*)r3d_p(gate) + (size_t)n;
    }
    if (up) {
        if (r3d_numel(gate) < M * n || r3d_numel(up) < M * n) return R3D_NO(RAD_E_SHAPE);
    } else if (r3d_numel(gate) < M * 2 * n) {
        return R3D_NO(RAD_E_SHAPE);
    }

    const long long had = r3d_geti_or(a, "group", 128);
    if (had != 128 && had != 256 && had != 512) return R3D_NO(RAD_E_SHAPE);
    if (n != had) return R3D_NO(RAD_E_SHAPE);   // single-block form (gfx1102, see the header)

    // `act` names the gate ("silu" or "sigmoid"); `mode` is its integer spelling, read when `act`
    // is absent. A name this kernel does not compute (SiTU) is refused rather than run as silu.
    const int mode = r3d_act_code(a, "act", "mode", 0);
    if (mode != 0 && mode != 1) return R3D_NO(RAD_E_INVAL);
    // The caps belong to SiTU; with silu or sigmoid they would be silently ignored.
    if (r3d_getf(a, "gate_cap", 0.0) != 0.0 || r3d_getf(a, "up_cap", 0.0) != 0.0)
        return R3D_NO(RAD_E_UNSUPPORTED);
    const float rnorm = 1.0f / sqrtf((float)had);
    dim3 grid((unsigned)M), blk(kThreads);
    const uint32_t nn = (uint32_t)n;
#define GHQ(H, MD)                                                                                      \
    R3D_KLAUNCH(R3D_KNAME(gated_had_quant_i8_kernel<H, MD>), grid, blk, (hipStream_t)s,                 \
                (const uint16_t*)r3d_p(gate), up_p, (uint8_t*)r3d_wp(q), (float*)r3d_wp(sc), nn, g_ld,   \
                u_ld, rnorm)
    if (mode == 0) {
        if (had == 256) GHQ(256, 0); else if (had == 512) GHQ(512, 0); else GHQ(128, 0);
    } else {
        if (had == 256) GHQ(256, 1); else if (had == 512) GHQ(512, 1); else GHQ(128, 1);
    }
#undef GHQ
    return (int)hipGetLastError();
}

}  // extern "C"
