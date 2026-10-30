/* Derived from libr4d/r4d_givens_dense.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
// halo_givens_dense.hip -- ParoQuant's rotation as a dense per-group matrix on the matrix
// cores: out[m][j] = sum_i T[j][i] * (x[m][i] / D[i]) per 128-column group, then the
// sequential producer's quantiser (int8 at amax / 127, E4M3 at amax / 448).
//
// The gfx1151 fragment layout is the attention kernels' (halo_dt16.h): a lane holds all
// sixteen k of a run, the accumulator's column is lane & 15 (here the output row m) and
// its element e is output column j = 2e + (lane>>4). So where the gfx12 kernel reads eight
// T values and eight activation values a step, this one reads sixteen of each: the A
// fragment is T row (lane & 15) at the step's sixteen i, the B fragment the activation
// row's sixteen i divided by D and rounded to f16. A row's 128 outputs sit in lanes col
// and col + 16 (the two j parities), so the amax closes with the same xor-16 shuffle.
//
// One block per (part, group) x row chunk. The group's T (32 KiB of f16) is staged in LDS
// once and each of the four waves takes 16-row tiles.
#include "halo.h"
#include "halo_args.h"
#include "halo_common.h"
#include "halo_wmma.h"
#include "halo_fp8_wmma.h"
#include "halo_q8.h"
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>

namespace {

constexpr uint32_t kGroup = 128;
constexpr uint32_t kThreads = 128;
constexpr uint32_t kWaves = kThreads / 32;
constexpr uint32_t kTStride = kGroup + 8;   // T row pitch in f16: 16 B pad for conflict-free b128
static_assert(kThreads == kGroup, "a thread stages one channel scale of the group (invd[tid])");

template <bool I8>
__global__ __launch_bounds__(kThreads) void givens_dense_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ tdense, const float* __restrict__ rscale,
    uint8_t* __restrict__ q, float* __restrict__ s, uint32_t M, uint32_t K, uint32_t nkb, uint32_t x_ld,
    uint32_t q_ld, uint32_t s_ld) {
    __shared__ uint16_t Tl[kGroup * kTStride];
    __shared__ float invd[kGroup];
    const uint32_t gp = blockIdx.x, part = gp / nkb, g = gp % nkb;
    const uint32_t tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;

    // all sixteen loads in flight before the first LDS store: the staging is latency, not bandwidth
    const uint4* tg = (const uint4*)(tdense + (size_t)gp * kGroup * kGroup);
    constexpr uint32_t kStage = kGroup * kGroup / 8 / kThreads;
    uint4 tv[kStage];
#pragma unroll
    for (uint32_t k = 0; k < kStage; ++k) tv[k] = tg[tid + k * kThreads];
#pragma unroll
    for (uint32_t k = 0; k < kStage; ++k) {
        const uint32_t i = tid + k * kThreads, j = i / (kGroup / 8), c = i % (kGroup / 8);
        *(uint4*)&Tl[j * kTStride + c * 8] = tv[k];
    }
    invd[tid] = 1.0f / rscale[(size_t)part * K + g * kGroup + tid];
    __syncthreads();

    const uint32_t h = lane >> 4, col = lane & 15;
    const uint32_t Mt = (M + 15) >> 4;
    const uint32_t step = gridDim.y * kWaves;
    for (uint32_t tile = blockIdx.y * kWaves + wave; tile < Mt; tile += step) {
        uint32_t m = tile * 16 + col;
        const bool live = m < M;
        if (!live) m = M - 1;
        const uint16_t* xr = x + (size_t)m * x_ld + g * kGroup;
        v8f acc[8];
#pragma unroll
        for (int jt = 0; jt < 8; ++jt) acc[jt] = (v8f)(0.f);
#pragma unroll
        for (uint32_t ks = 0; ks < 8; ++ks) {
            const uint32_t c0 = ks * 16;
            // The B fragment: this row's sixteen activation values at the step, / D, to f16.
            const uint4 r0 = *(const uint4*)(xr + c0);
            const uint4 r1 = *(const uint4*)(xr + c0 + 8);
            const uint32_t w[4] = { r0.x, r0.y, r0.z, r0.w };
            const uint32_t w2[4] = { r1.x, r1.y, r1.z, r1.w };
            union { uint32_t u[8]; v16hf f; } bfq;
#pragma unroll
            for (int e = 0; e < 16; ++e) {
                const uint32_t wd = e < 8 ? w[e >> 1] : w2[(e >> 1) & 3];
                const uint32_t b = (wd >> (16 * (e & 1))) & 0xFFFFu;
                const float v = __uint_as_float(b << 16) * invd[c0 + e];
                const _Float16 hv = (_Float16)fminf(fmaxf(v, -65504.0f), 65504.0f);
                uint16_t hb;
                __builtin_memcpy(&hb, &hv, 2);
                if ((e & 1) == 0) bfq.u[e >> 1] = hb;
                else bfq.u[e >> 1] |= (uint32_t)hb << 16;
            }
#pragma unroll
            for (int jt = 0; jt < 8; ++jt) {
                const v16hf af = *(const v16hf*)&Tl[(jt * 16 + col) * kTStride + c0];
                acc[jt] = halo::mma(af, bfq.f, acc[jt]);
            }
        }
        float amax = 0.0f;
#pragma unroll
        for (int jt = 0; jt < 8; ++jt)
#pragma unroll
            for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(acc[jt][e]));
        amax = fmaxf(amax, __shfl_xor(amax, 16));
        const float sc = amax > 0.0f ? amax * (1.0f / q8_top<I8>()) : 1.0f;
        const float inv = 1.0f / sc;
        if (!live) continue;
        if (h == 0) s[(size_t)m * s_ld + (size_t)part * nkb + g] = sc;
        uint8_t* qr = q + (size_t)m * q_ld + (size_t)part * K + g * kGroup;
        // Element e of tile jt is output column jt*16 + 2e + h: four consecutive e of one
        // parity pack as q8_pack4's four values, stored at the strided columns.
#pragma unroll
        for (int jt = 0; jt < 8; ++jt) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float v = acc[jt][e];
                const uint32_t code = I8
                    ? (uint32_t)(uint8_t)(int8_t)(int)rintf(fminf(fmaxf(v * inv, -127.0f), 127.0f))
                    : halo_fp8::f32_to_e4m3(fminf(fmaxf(v * inv, -448.0f), 448.0f));
                qr[jt * 16 + 2 * e + h] = (uint8_t)code;
            }
        }
    }
}

}  // namespace

//   x       [M][K]                       bf16, `x_ld` elements a row (0 = K)
//   tdense  [parts * K/128][128][128]    f16, T[j][i] of each (part, group)
//   rscale  [parts][K]                   f32, the factor the stored row was multiplied by
//   q       [M][parts * K]               int8 or E4M3, `q_ld` elements a row (0 = parts * K)
//   s       [M][parts * K/128]           f32, `s_ld` elements a row (0 = parts * K/128)
enum { GD_X = 0, GD_T = 1, GD_RSCALE = 2, GD_Q = 3, GD_S = 4 };

extern "C" int halo_rad_givens_dense_quant(const RadArgs* a, RadStream stream) {
    const RadTensor* x  = halo_opt(a, GD_X);
    const RadTensor* t  = halo_opt(a, GD_T);
    const RadTensor* rs = halo_opt(a, GD_RSCALE);
    const RadTensor* q  = halo_opt(a, GD_Q);
    const RadTensor* sc = halo_opt(a, GD_S);
    if (!x || !t || !rs || !q || !sc) return HALO_NO(RAD_E_INVAL);
    if (x->rank < 2 || q->rank != 2 || sc->rank != 2 || t->rank != 2 || rs->rank != 2)
        return HALO_NO(RAD_E_SHAPE);
    if (t->dtype != RAD_F16 || rs->dtype != RAD_F32) return HALO_NO(RAD_E_DTYPE);
    // `out` names the quantiser and q's dtype must be it: a mismatch is refused, not read from q
    const char* out = halo_gets(a, "out");
    const bool i8 = q->dtype == RAD_I8;
    if (!i8 && q->dtype != RAD_F8E4M3) return HALO_NO(RAD_E_DTYPE);
    if (!out || std::strcmp(out, i8 ? "i8" : "fp8") != 0) return HALO_NO(RAD_E_DTYPE);
    const long long M = x->shape[0], K = halo_numel(x) / (x->shape[0] > 0 ? x->shape[0] : 1);
    const long long parts = halo_geti_or(a, "parts", 1), nkb = K / kGroup;
    if (halo_geti_or(a, "group", kGroup) != (long long)kGroup || K % kGroup || parts < 1)
        return HALO_NO(RAD_E_SHAPE);
    if (q->shape[0] != M || q->shape[1] != parts * K) return HALO_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != parts * nkb) return HALO_NO(RAD_E_SHAPE);
    // tdense is a packed blob: its byte count is the layout hook's ([parts * K, 128] f16), not a shape to check
    if (rs->shape[0] != parts || rs->shape[1] != K) return HALO_NO(RAD_E_SHAPE);
    if (!halo_aligned(t->data, 16) || !halo_aligned(x->data, 16) || !halo_aligned(q->data, 8))
        return HALO_NO(RAD_E_ALIGN);
    const long long xp = halo_row_pitch(x), qp = halo_row_pitch(q), sp = halo_row_pitch(sc);
    if (xp < 0 || qp < 0 || sp < 0 || (xp * 2) % 16) return HALO_NO(RAD_E_STRIDE);
    const long long mt = (M + 15) / 16;
    const uint32_t gy = (uint32_t)std::max<long long>(1, (mt + 4 * kWaves - 1) / (4 * kWaves));
    const dim3 grid((uint32_t)(parts * nkb), gy);
    if (i8)
        givens_dense_kernel<true><<<grid, dim3(kThreads), 0, (hipStream_t)stream>>>(
            (const uint16_t*)x->data, (const uint16_t*)t->data, (const float*)rs->data,
            (uint8_t*)q->data, (float*)sc->data, (uint32_t)M, (uint32_t)K, (uint32_t)nkb,
            (uint32_t)xp, (uint32_t)qp, (uint32_t)sp);
    else
        givens_dense_kernel<false><<<grid, dim3(kThreads), 0, (hipStream_t)stream>>>(
            (const uint16_t*)x->data, (const uint16_t*)t->data, (const float*)rs->data,
            (uint8_t*)q->data, (float*)sc->data, (uint32_t)M, (uint32_t)K, (uint32_t)nkb,
            (uint32_t)xp, (uint32_t)qp, (uint32_t)sp);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
