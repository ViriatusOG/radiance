// r4d_ggml_wmma.hip -- gemm_nt_ggml / logits_gemm_ggml at decode heights (M <= 48) over IQ2_XXS and
// Q8_0 blocks with the bf16 WMMA, gfx1201. The weight is llama.cpp's blocks row after row, as
// r4d_ds41_ggml_layout stores it; x is bf16, y bf16 (gemm) or f32 (logits).
//
// Shared by both types: one block is one 16-column tile (16 weight rows n) and 8 waves that split K
// (wave, wave + 8, ... over a type's unit of work). The weight is the WMMA B operand: lane
// (n = lane % 16, half h = lane / 16) holds 8 k of column n, and in the D fragment the 8 values of a
// lane share that column, so one block scale a lane folds into the f32 accumulator with an FMA after
// the WMMAs of the scale's 32 k. The weight values are exact in bf16 (IQ2_XXS: +-grid 8, 25, 43;
// Q8_0: int8), so each product is exact and only the f32 sums round, in an order that depends on K
// alone: a row's bits are the same at every M, as the decode kernel's are.
//
// IQ2_XXS: a 66-byte block a 256 k. Half h loads units 4h..4h+3 (32 bytes), permlanex16 hands over
// the other four. A group's 8 bf16 are the grid row (LDS table, 256 x 8 bf16) XOR its sign mask (LDS,
// 128 masks with the parity bit), db = d (0.5 + scale) 0.25 as llama.cpp, exact in f32.
// Q8_0: a 34-byte block a 32 k; a wave owns quads of 4 blocks (136 bytes of a row), half h loads
// blocks 2h, 2h+1 (68 bytes) and hands the other half its k-half words. An int8 becomes bf16 by a
// magic-number float (2^23 + v + 128, minus 2^23 + 128), exact.
#include "r4d.h"
#include "r4d_emit.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {
typedef float v4f __attribute__((ext_vector_type(4)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned u4 __attribute__((ext_vector_type(4)));
typedef unsigned u2 __attribute__((ext_vector_type(2)));
typedef unsigned u4a4 __attribute__((ext_vector_type(4), aligned(4)));
typedef u2 __attribute__((aligned(2))) u2a2;
typedef __attribute__((address_space(1))) const u2a2* gu2a2;
typedef uint16_t __attribute__((aligned(2))) u16a2;
constexpr int KW = 8;

constexpr uint64_t kGwIq2xxsGrid[256] = {
    0x0808080808080808ull, 0x080808080808082bull, 0x0808080808081919ull, 0x0808080808082b08ull,
    0x0808080808082b2bull, 0x0808080808190819ull, 0x0808080808191908ull, 0x08080808082b0808ull,
    0x08080808082b082bull, 0x08080808082b2b08ull, 0x08080808082b2b2bull, 0x0808080819080819ull,
    0x0808080819081908ull, 0x0808080819190808ull, 0x0808080819192b08ull, 0x08080808192b0819ull,
    0x08080808192b1908ull, 0x080808082b080808ull, 0x080808082b08082bull, 0x080808082b082b2bull,
    0x080808082b2b082bull, 0x0808081908080819ull, 0x0808081908081908ull, 0x0808081908190808ull,
    0x0808081908191919ull, 0x0808081919080808ull, 0x080808192b081908ull, 0x080808192b192b08ull,
    0x0808082b08080808ull, 0x0808082b0808082bull, 0x0808082b082b082bull, 0x0808082b2b08082bull,
    0x0808190808080819ull, 0x0808190808081908ull, 0x0808190808190808ull, 0x08081908082b0819ull,
    0x08081908082b1908ull, 0x0808190819080808ull, 0x080819081908082bull, 0x0808190819082b08ull,
    0x08081908192b0808ull, 0x080819082b080819ull, 0x080819082b081908ull, 0x080819082b190808ull,
    0x080819082b2b1908ull, 0x0808191908080808ull, 0x080819190808082bull, 0x0808191908082b08ull,
    0x08081919082b0808ull, 0x080819191908192bull, 0x08081919192b2b19ull, 0x080819192b080808ull,
    0x080819192b190819ull, 0x0808192b08082b19ull, 0x0808192b08190808ull, 0x0808192b19080808ull,
    0x0808192b2b081908ull, 0x0808192b2b2b1908ull, 0x08082b0808080808ull, 0x08082b0808081919ull,
    0x08082b0808082b08ull, 0x08082b0808191908ull, 0x08082b08082b2b08ull, 0x08082b0819080819ull,
    0x08082b0819081908ull, 0x08082b0819190808ull, 0x08082b081919082bull, 0x08082b082b082b08ull,
    0x08082b1908081908ull, 0x08082b1919080808ull, 0x08082b2b0808082bull, 0x08082b2b08191908ull,
    0x0819080808080819ull, 0x0819080808081908ull, 0x0819080808190808ull, 0x08190808082b0819ull,
    0x0819080819080808ull, 0x08190808192b0808ull, 0x081908082b081908ull, 0x081908082b190808ull,
    0x081908082b191919ull, 0x0819081908080808ull, 0x0819081908082b08ull, 0x08190819082b0808ull,
    0x0819081919190808ull, 0x0819081919192b2bull, 0x081908192b080808ull, 0x0819082b082b1908ull,
    0x0819082b19081919ull, 0x0819190808080808ull, 0x0819190808082b08ull, 0x08191908082b0808ull,
    0x08191908082b1919ull, 0x0819190819082b19ull, 0x081919082b080808ull, 0x0819191908192b08ull,
    0x08191919192b082bull, 0x0819192b08080808ull, 0x0819192b0819192bull, 0x08192b0808080819ull,
    0x08192b0808081908ull, 0x08192b0808190808ull, 0x08192b0819080808ull, 0x08192b082b080819ull,
    0x08192b1908080808ull, 0x08192b1908081919ull, 0x08192b192b2b0808ull, 0x08192b2b19190819ull,
    0x082b080808080808ull, 0x082b08080808082bull, 0x082b080808082b2bull, 0x082b080819081908ull,
    0x082b0808192b0819ull, 0x082b08082b080808ull, 0x082b08082b08082bull, 0x082b0819082b2b19ull,
    0x082b081919082b08ull, 0x082b082b08080808ull, 0x082b082b0808082bull, 0x082b190808080819ull,
    0x082b190808081908ull, 0x082b190808190808ull, 0x082b190819080808ull, 0x082b19081919192bull,
    0x082b191908080808ull, 0x082b191919080819ull, 0x082b1919192b1908ull, 0x082b192b2b190808ull,
    0x082b2b0808082b08ull, 0x082b2b08082b0808ull, 0x082b2b082b191908ull, 0x082b2b2b19081908ull,
    0x1908080808080819ull, 0x1908080808081908ull, 0x1908080808190808ull, 0x1908080808192b08ull,
    0x19080808082b0819ull, 0x19080808082b1908ull, 0x1908080819080808ull, 0x1908080819082b08ull,
    0x190808081919192bull, 0x19080808192b0808ull, 0x190808082b080819ull, 0x190808082b081908ull,
    0x190808082b190808ull, 0x1908081908080808ull, 0x19080819082b0808ull, 0x19080819192b0819ull,
    0x190808192b080808ull, 0x190808192b081919ull, 0x1908082b08080819ull, 0x1908082b08190808ull,
    0x1908082b19082b08ull, 0x1908082b1919192bull, 0x1908082b192b2b08ull, 0x1908190808080808ull,
    0x1908190808082b08ull, 0x19081908082b0808ull, 0x190819082b080808ull, 0x190819082b192b19ull,
    0x190819190819082bull, 0x19081919082b1908ull, 0x1908192b08080808ull, 0x19082b0808080819ull,
    0x19082b0808081908ull, 0x19082b0808190808ull, 0x19082b0819080808ull, 0x19082b0819081919ull,
    0x19082b1908080808ull, 0x19082b1919192b08ull, 0x19082b19192b0819ull, 0x19082b192b08082bull,
    0x19082b2b19081919ull, 0x19082b2b2b190808ull, 0x1919080808080808ull, 0x1919080808082b08ull,
    0x1919080808190819ull, 0x1919080808192b19ull, 0x19190808082b0808ull, 0x191908082b080808ull,
    0x191908082b082b08ull, 0x1919081908081908ull, 0x191908191908082bull, 0x191908192b2b1908ull,
    0x1919082b2b190819ull, 0x191919082b190808ull, 0x191919082b19082bull, 0x1919191908082b2bull,
    0x1919192b08080819ull, 0x1919192b19191908ull, 0x19192b0808080808ull, 0x19192b0808190819ull,
    0x19192b0808192b19ull, 0x19192b08192b1908ull, 0x19192b1919080808ull, 0x19192b2b08082b08ull,
    0x192b080808081908ull, 0x192b080808190808ull, 0x192b080819080808ull, 0x192b0808192b2b08ull,
    0x192b081908080808ull, 0x192b081919191919ull, 0x192b082b08192b08ull, 0x192b082b192b0808ull,
    0x192b190808080808ull, 0x192b190808081919ull, 0x192b191908190808ull, 0x192b19190819082bull,
    0x192b19192b081908ull, 0x192b2b081908082bull, 0x2b08080808080808ull, 0x2b0808080808082bull,
    0x2b08080808082b2bull, 0x2b08080819080819ull, 0x2b0808082b08082bull, 0x2b08081908081908ull,
    0x2b08081908192b08ull, 0x2b08081919080808ull, 0x2b08082b08190819ull, 0x2b08190808080819ull,
    0x2b08190808081908ull, 0x2b08190808190808ull, 0x2b08190808191919ull, 0x2b08190819080808ull,
    0x2b081908192b0808ull, 0x2b08191908080808ull, 0x2b0819191908192bull, 0x2b0819192b191908ull,
    0x2b08192b08082b19ull, 0x2b08192b19080808ull, 0x2b08192b192b0808ull, 0x2b082b080808082bull,
    0x2b082b1908081908ull, 0x2b082b2b08190819ull, 0x2b19080808081908ull, 0x2b19080808190808ull,
    0x2b190808082b1908ull, 0x2b19080819080808ull, 0x2b1908082b2b0819ull, 0x2b1908190819192bull,
    0x2b1908192b080808ull, 0x2b19082b19081919ull, 0x2b19190808080808ull, 0x2b191908082b082bull,
    0x2b19190819081908ull, 0x2b19191919190819ull, 0x2b192b082b080819ull, 0x2b192b19082b0808ull,
    0x2b2b08080808082bull, 0x2b2b080819190808ull, 0x2b2b08082b081919ull, 0x2b2b081908082b19ull,
    0x2b2b082b08080808ull, 0x2b2b190808192b08ull, 0x2b2b2b0819190808ull, 0x2b2b2b1908081908ull,
};

__device__ __forceinline__ unsigned swap16(unsigned v) {
    return (unsigned)__builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210, (int)0xfedcba98, false, false);
}
/* IEEE half -> float exactly (subnormals, infinities, NaN payloads), as r4d_ds41_ggml's gg_h2f. */
__device__ __forceinline__ float h2f(uint32_t h) {
    const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    const float sub = __uint_as_float(__float_as_uint((float)m * 0x1p-24f) | s);
    const uint32_t nb = e == 0x1fu ? (s | 0x7f800000u | (m << 13)) : (s | ((e + 112u) << 23) | (m << 13));
    return e == 0u ? sub : __uint_as_float(nb);
}
__device__ __forceinline__ unsigned bfi(int v) { return __float_as_uint((float)v) >> 16; }
/* The two tables, built at compile time: 256 grid rows as 8 bf16 (8, 25, 43 are 0x4100, 0x41C8,
 * 0x422C), then 128 sign masks (bit 7 the parity of the 7 sign bits, as ksigns_iq2xs). */
struct GwTab { uint32_t v[384][4]; };
constexpr uint32_t gw_bf(uint32_t b) { return b == 0x08u ? 0x4100u : b == 0x19u ? 0x41C8u : 0x422Cu; }
constexpr GwTab gw_make() {
    GwTab t{};
    for (int i = 0; i < 256; ++i)
        for (int j = 0; j < 4; ++j) {
            const uint64_t e = kGwIq2xxsGrid[i];
            t.v[i][j] = gw_bf((uint32_t)(e >> (16 * j)) & 0xffu) | (gw_bf((uint32_t)(e >> (16 * j + 8)) & 0xffu) << 16);
        }
    for (int i = 0; i < 128; ++i) {
        uint32_t pc = 0;
        for (int b = 0; b < 7; ++b) pc += (uint32_t)(i >> b) & 1u;
        const uint32_t sg = (uint32_t)i | ((pc & 1u) << 7);
        for (int j = 0; j < 4; ++j) t.v[256 + i][j] = (((sg >> (2 * j)) & 1u) << 15) | (((sg >> (2 * j + 1)) & 1u) << 31);
    }
    return t;
}
__constant__ __attribute__((aligned(16))) GwTab kGwTab = gw_make();
/* One wave's share of a 16-column tile: weight row `row` (column n of the lane), activation row
 * `xr` of the lane's token (null: a row of zeros), blocks wave, wave + KW, ... The gemm and the moe
 * kernel both run this, so an expert's output row has the bits of the gemm's for that row. */
template <int MR>
__device__ __forceinline__ v8f iq2w_tile(const uint8_t* __restrict__ row,
                                         const __hip_bfloat16* __restrict__ xr, int K, int wave, int kw, int h) {
    v8f acc;
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.f;
    const v8f zero = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int nb = K >> 8;
    const u4* gt = (const u4*)(const void*)&kGwTab;   /* constant memory, read through the vector cache */
    u2 wv[4] = {};
    uint32_t dv = 0;
    auto load = [&](int b) {
        const uint8_t* blk = row + (size_t)b * 66;
        dv = *(const u16a2*)blk;
#pragma unroll
        for (int i = 0; i < 4; ++i) wv[i] = *(gu2a2)(const void*)(blk + 2 + 8 * (4 * h + i));
    };
    if (wave < nb) load(wave);
    for (int b = wave; b < nb; b += kw) {
        u2 own[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) own[i] = wv[i];
        const float d = h2f(dv);
        if (b + kw < nb) load(b + kw);
        u2 rcv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { rcv[i].x = swap16(own[i].x); rcv[i].y = swap16(own[i].y); }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const u2 au = ((u >> 2) == h) ? own[u & 3] : rcv[u & 3];
            const float db = d * (0.5f + (float)(au.y >> 28)) * 0.25f;
            v8f dd;
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int G = 2 * s + h;
                const unsigned gi = (au.x >> (8 * G)) & 255u, sx = (au.y >> (7 * G)) & 127u;
                const u4 bw = gt[gi] ^ gt[256 + sx];
                const v4f bf = __builtin_bit_cast(v4f, bw);
                v4f a = {0.f, 0.f, 0.f, 0.f};
                if (xr) a = *(const v4f*)(xr + b * 256 + u * 32 + s * 16 + h * 8);
                dd = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, bf, s ? dd : zero);
            }
#pragma unroll
            for (int j = 0; j < MR; ++j) acc[j] = __builtin_fmaf(dd[j], db, acc[j]);
        }
    }
    return acc;
}

/* Waves a tile: 8, or the 256-k blocks of a row where fewer (a short K leaves no wave idle: the block
 * then holds 8 / kw tiles). Depends on K alone, so the sum order does too. */
__device__ __forceinline__ int iq2w_kw(int K) {
    const int nb = K >> 8;
    return nb >= 8 ? 8 : nb >= 4 ? 4 : nb >= 2 ? 2 : 1;
}
/* The tile's kw partials summed in a fixed order, row r < M written to yrow(r)[n0 + column]. */
template <typename OT, typename F>
__device__ __forceinline__ void tile_out(float (*sP)[16][16], const v8f& acc, int wave, int kw, int c, int h,
                                         int M, F&& yrow, int n0, bool live) {
#pragma unroll
    for (int j = 0; j < 8; ++j) sP[wave][h * 8 + j][c] = acc[j];
    __syncthreads();
    const int w0 = wave - wave % kw, ln = (int)threadIdx.x - w0 * 32;   /* this tile's waves, my index */
    for (int e = ln; e < 256 && live; e += kw * 32) {
        const int r = e >> 4, cc = e & 15;
        if (r >= M) continue;
        float sum = 0.f;
        for (int w = 0; w < kw; ++w) sum += sP[w0 + w][r][cc];
        OT* yp = yrow(r) + n0 + cc;
        if constexpr (sizeof(OT) == 4) *yp = sum;
        else *yp = __float2bfloat16(sum);
    }
}

template <int MT, int MR, typename OT>
__global__ __launch_bounds__(KW * 32) void iq2w_kernel(const uint8_t* __restrict__ W,
        const __hip_bfloat16* __restrict__ x, long x_ld, OT* __restrict__ y, long y_ld, int M, int N, int K) {
    static_assert(MT == 1, "one 16-row token tile a launch");
    __shared__ float sP[KW][16][16];
    const int lane = threadIdx.x & 31;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int c = lane & 15, h = lane >> 4;
    const int kw = iq2w_kw(K), tile = blockIdx.x * (KW / kw) + wave / kw;
    const bool live = tile * 16 < N;
    const int n0 = (live ? tile : 0) * 16;
    v8f acc = iq2w_tile<MR>(W + (size_t)(n0 + c) * (K >> 8) * 66,
                            c < M ? x + (size_t)c * x_ld : nullptr, K, wave % kw, kw, h);
    tile_out<OT>(sP, acc, wave, kw, c, h, M, [&](int r) { return y + (size_t)r * y_ld; }, n0, live);
}

/* moe_gemm_ggml over IQ2_XXS experts: block (tile, slot) takes the slot-th expert that has rows, in
 * passes of 16 of its sorted rows. a row reads a[row(i)] (a_order), a dropped slot a row of zeros. */
__global__ __launch_bounds__(KW * 32) void iq2w_moe_kernel(const uint8_t* __restrict__ W,
        const uint8_t* const* __restrict__ Wtab, long long w_es, const __hip_bfloat16* __restrict__ x,
        long x_ld, const int32_t* __restrict__ sorted, const int32_t* __restrict__ off, int T, int ne,
        int top_k, int a_sorted, int AR, __hip_bfloat16* __restrict__ y, long y_ld, int N, int K) {
    __shared__ float sP[KW][16][16];
    __shared__ int s_e;
    const int lane = threadIdx.x & 31;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    if (wave == 0) {   /* the blockIdx.y-th expert with rows */
        int cnt = 0, found = -1;
        const int target = (int)blockIdx.y;
        for (int base = 0; base < ne && found < 0; base += 32) {
            const int e = base + lane;
            bool live = false;
            if (e < ne) {
                int a = off[e], b = off[e + 1];
                a = a < 0 ? 0 : (a > T ? T : a);
                b = b < a ? a : (b > T ? T : b);
                live = b > a;
            }
            const unsigned bal = (unsigned)__ballot(live);
            const int rank = __builtin_popcount(bal & ((1u << lane) - 1u));
            const unsigned fb = (unsigned)__ballot(live && cnt + rank == target);
            if (fb) found = base + __ffs(fb) - 1;
            cnt += __builtin_popcount(bal);
        }
        if (lane == 0) s_e = found;
    }
    __syncthreads();
    const int e = s_e;
    if (e < 0) return;
    int r0 = off[e], r1 = off[e + 1];
    r0 = r0 < 0 ? 0 : (r0 > T ? T : r0);
    r1 = r1 < r0 ? r0 : (r1 > T ? T : r1);
    const int c = lane & 15, h = lane >> 4;
    const int kw = iq2w_kw(K), tile = blockIdx.x * (KW / kw) + wave / kw;
    const bool live = tile * 16 < N;
    const int n0 = (live ? tile : 0) * 16;
    const uint8_t* we = Wtab ? Wtab[e] : W + (size_t)e * w_es;
    const uint8_t* row = we + (size_t)(n0 + c) * (K >> 8) * 66;
    for (int p0 = r0; p0 < r1; p0 += 16) {
        const int Mp = r1 - p0 < 16 ? r1 - p0 : 16;
        const __hip_bfloat16* xr = nullptr;
        if (c < Mp) {
            const int i = p0 + c, flat = sorted[i];
            const int tok = flat < 0 ? -1 : (a_sorted ? i : flat / top_k);
            if (tok >= 0 && tok < AR) xr = x + (size_t)tok * x_ld;
        }
        const v8f acc = iq2w_tile<8>(row, xr, K, wave % kw, kw, h);
        tile_out<__hip_bfloat16>(sP, acc, wave, kw, c, h, Mp, [&](int r) { return y + (size_t)(p0 + r) * y_ld; }, n0, live);
        __syncthreads();
    }
}


__device__ __forceinline__ unsigned bf_pair(unsigned u, int i) {   // bytes i, i+1 of u (v + 128) -> 2 bf16
    const float f0 = __uint_as_float(__builtin_amdgcn_perm(0x4B4B4B4Bu, u, 0x040C0C00u + i)) - 8388736.0f;
    const float f1 = __uint_as_float(__builtin_amdgcn_perm(0x4B4B4B4Bu, u, 0x040C0C00u + i + 1)) - 8388736.0f;
    return __builtin_amdgcn_perm(__float_as_uint(f1), __float_as_uint(f0), 0x07060302u);
}
__device__ __forceinline__ v4f dec8_q8(unsigned lo, unsigned hi) {   // 8 int8 (k order) -> 8 bf16
    lo ^= 0x80808080u; hi ^= 0x80808080u;
    u4 o; o.x = bf_pair(lo, 0); o.y = bf_pair(lo, 2); o.z = bf_pair(hi, 0); o.w = bf_pair(hi, 2);
    return __builtin_bit_cast(v4f, o);
}
template <int MT, int MR, typename OT>
__global__ __launch_bounds__(KW * 32) void q8w_kernel(const uint8_t* __restrict__ W,
        const __hip_bfloat16* __restrict__ x, long x_ld, OT* __restrict__ y, long y_ld, int M, int N, int K) {
    __shared__ float sP[KW][MT][16][16];
    const int lane = threadIdx.x & 31;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int c = lane & 15, h = lane >> 4;
    const int n = blockIdx.x * 16 + c;
    const int nq = K >> 7;
    const uint8_t* __restrict__ row = W + (size_t)n * (size_t)(K >> 5) * 34 + 68 * h;
    v8f acc[MT];
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[t][j] = 0.f;
    const v8f zero = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    u4a4 D0 = {}, D1 = {}, D2 = {}, D3 = {};
    unsigned D4 = 0;
#define Q8W_LOAD(q) do { const uint8_t* p_ = row + (size_t)(q) * 136; \
        D0 = *(const u4a4*)(p_); D1 = *(const u4a4*)(p_ + 16); D2 = *(const u4a4*)(p_ + 32); \
        D3 = *(const u4a4*)(p_ + 48); D4 = *(const unsigned*)(p_ + 64); } while (0)
    if (wave < nq) Q8W_LOAD(wave);
    for (int q = wave; q < nq; q += KW) {
        const unsigned C[17] = {D0.x, D0.y, D0.z, D0.w, D1.x, D1.y, D1.z, D1.w, D2.x, D2.y, D2.z, D2.w,
                                D3.x, D3.y, D3.z, D3.w, D4};
        if (q + KW < nq) Q8W_LOAD(q + KW);
        // my two blocks (2h, 2h+1): block A bytes 0..33, block B bytes 34..67 of my 68.
        // qs dwords (aligned): A: bytes 2+4i -> alignbyte(C[i+1], C[i], 2); B: C[9+i].
        unsigned qa[8], qb[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) { qa[i] = __builtin_amdgcn_alignbyte(C[i + 1], C[i], 2); qb[i] = C[9 + i]; }
        const unsigned dA = C[0] & 0xffffu, dB = C[8] >> 16;
        // the partner (other half) needs, of each of my blocks, its own k-half of both steps:
        // step s, half hh: qs dwords 4s + 2hh, 4s + 2hh + 1.  I send hh = 1 - h, receive mine.
        unsigned ra[4], rb[4];
#pragma unroll
        for (int s = 0; s < 2; ++s)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                ra[2*s+e] = swap16(h ? qa[4*s + e] : qa[4*s + 2 + e]);
                rb[2*s+e] = swap16(h ? qb[4*s + e] : qb[4*s + 2 + e]);
            }
        const unsigned rdA = swap16(dA), rdB = swap16(dB);
#pragma unroll
        for (int j = 0; j < 4; ++j) {   // block j of the quad: owner half j >> 1
            const bool mine = (j >> 1) == h;
            const bool isB = j & 1;
            const float db = h2f(mine ? (isB ? dB : dA) : (isB ? rdB : rdA));
            v8f dd[MT];
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                unsigned lo, hi;
                if (mine) {
                    const unsigned a0 = isB ? qb[4*s] : qa[4*s], a1 = isB ? qb[4*s + 1] : qa[4*s + 1];
                    const unsigned a2 = isB ? qb[4*s + 2] : qa[4*s + 2], a3 = isB ? qb[4*s + 3] : qa[4*s + 3];
                    lo = h ? a2 : a0; hi = h ? a3 : a1;
                }
                else      { lo = isB ? rb[2*s] : ra[2*s]; hi = isB ? rb[2*s + 1] : ra[2*s + 1]; }
                const v4f bf = dec8_q8(lo, hi);
                const int k0 = q * 128 + j * 32 + s * 16 + h * 8;
#pragma unroll
                for (int t = 0; t < MT; ++t) {
                    const int m = t * 16 + c;
                    v4f a = {0.f, 0.f, 0.f, 0.f};
                    if (m < M) a = *(const v4f*)(x + (size_t)m * x_ld + k0);
                    dd[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, bf, s ? dd[t] : zero);
                }
            }
#pragma unroll
            for (int t = 0; t < MT; ++t)
#pragma unroll
                for (int jj = 0; jj < MR; ++jj) acc[t][jj] = __builtin_fmaf(dd[t][jj], db, acc[t][jj]);
        }
    }
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
        for (int j = 0; j < 8; ++j) sP[wave][t][h * 8 + j][c] = acc[t][j];
    __syncthreads();
    for (int e = threadIdx.x; e < MT * 256; e += KW * 32) {
        const int t = e >> 8, r = (e >> 4) & 15, cc = e & 15;
        const int m = t * 16 + r;
        if (m >= M) continue;
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < KW; ++w) sum += sP[w][t][r][cc];
        if constexpr (sizeof(OT) == 4) y[(size_t)m * y_ld + blockIdx.x * 16 + cc] = sum;
        else y[(size_t)m * y_ld + blockIdx.x * 16 + cc] = __float2bfloat16(sum);
    }
}

/* Rows in passes of 16 (one 16-row token tile a launch): the weight is read once a pass. */
template <typename OT>
int launch(int iq2, const uint8_t* w, const __hip_bfloat16* x, long x_ld, OT* y, long y_ld, int M,
           int N, int K, RadStream s) {
    const int tpb = iq2 ? 8 / (K / 256 >= 8 ? 8 : K / 256 >= 4 ? 4 : K / 256 >= 2 ? 2 : 1) : 1;
    const dim3 g((unsigned)((N / 16 + tpb - 1) / tpb)), b(KW * 32);
    for (int m0 = 0; m0 < M; m0 += 16) {
        const int Mp = M - m0 < 16 ? M - m0 : 16;
        const __hip_bfloat16* xp = x + (size_t)m0 * x_ld;
        OT* yp = y + (size_t)m0 * y_ld;
        int rc;
#define GW_L(KER, MR) R4D_EMIT(s, (KER<1, MR, OT>), g, b, 0, w, xp, x_ld, yp, y_ld, Mp, N, K)
        if (iq2) rc = Mp == 1 ? GW_L(iq2w_kernel, 1) : Mp <= 2 ? GW_L(iq2w_kernel, 2) : Mp <= 4 ? GW_L(iq2w_kernel, 4) : GW_L(iq2w_kernel, 8);
        else     rc = Mp == 1 ? GW_L(q8w_kernel, 1) : Mp <= 2 ? GW_L(q8w_kernel, 2) : Mp <= 4 ? GW_L(q8w_kernel, 4) : GW_L(q8w_kernel, 8);
#undef GW_L
        if (rc != RAD_OK) return rc;
    }
    return RAD_OK;
}
}  // namespace

/* 1 when the shape is not served here (the caller runs its own kernel), else RAD_OK or an error.
 * w_ld: the weight's row bytes, which must be the blocks' (K / qk * block bytes). */
extern "C" int r4d_ggml_wmma_gemm(int iq2, const void* w, long long w_ld, const void* x, long long x_ld,
                                  void* y, long long y_ld, int f32out, int M, int N, int K, RadStream s) {
    if (M < 1 || N < 16 || N % 16 || K < 256) return 1;
    if (iq2 ? (K % 256 || w_ld != (long long)(K / 256) * 66) : (K % 128 || w_ld != (long long)(K / 32) * 34)) return 1;
    if (((uintptr_t)w & 3u) || ((uintptr_t)x & 15u) || (x_ld & 7)) return 1;
    const uint8_t* wb = (const uint8_t*)w;
    const __hip_bfloat16* xb = (const __hip_bfloat16*)x;
    if (f32out) return launch(iq2, wb, xb, (long)x_ld, (float*)y, (long)y_ld, M, N, K, s);
    return launch(iq2, wb, xb, (long)x_ld, (__hip_bfloat16*)y, (long)y_ld, M, N, K, s);
}

/* moe_gemm_ggml over IQ2_XXS experts in the decode band; 1 = not served. W: experts back to back
 * (w_es bytes apart) or, with Wtab, a table of expert pointers. */
extern "C" int r4d_ggml_wmma_moe(const void* W, const void* const* Wtab, long long w_ld, long long w_es,
                                 const void* x, long long x_ld, const int32_t* sorted, const int32_t* off,
                                 int T, int ne, int top_k, int a_sorted, int AR, void* y, long long y_ld,
                                 int N, int K, RadStream s) {
    if (T < 1 || ne < 1 || N < 16 || N % 16 || K < 256 || K % 256) return 1;
    if (w_ld != (long long)(K / 256) * 66) return 1;
    if (((uintptr_t)x & 15u) || (x_ld & 7)) return 1;
    const int L = ne < T ? ne : T;
    const int nb = K / 256, tpb = 8 / (nb >= 8 ? 8 : nb >= 4 ? 4 : nb >= 2 ? 2 : 1);
    const dim3 g((unsigned)((N / 16 + tpb - 1) / tpb), (unsigned)L), b(KW * 32);
    return R4D_EMIT(s, iq2w_moe_kernel, g, b, 0, (const uint8_t*)W, (const uint8_t* const*)Wtab, w_es,
                    (const __hip_bfloat16*)x, (long)x_ld, sorted, off, T, ne, top_k, a_sorted, AR,
                    (__hip_bfloat16*)y, (long)y_ld, N, K);
}
