// exl3i_8a8.hip.h -- the i8a8 prefill tile for exl3i's A8 path (exl3i_a8.h): a radiance tile copied here (Apache-2.0,
// the repository's LICENSE), because a plugin links only the ABI headers. The tile's measurements and layout notes are in
// the radiance source and are not repeated. What is the same, line for line: the block (128 weight rows x 128 activation rows, eight waves as 4
// weight x 2 activation, a wave 2 x 4 tiles of 16 x 16), the weight read from the fragment-order plane straight into
// registers, the activation staged through LDS (two threads a row, double-buffered, padded pitch, one barrier a
// staged tile), the raster grouping (kGroupM), the int8 WMMA, and the RB 32 fold (each 128-K block's int32 sums
// start at 0x4B400000 and fold with one FMA a slot, the bias removed every kRbFlush blocks and at the end).
//
// What is EXL3's: (1) the weight has no scale in the K loop (the int8 grid's step EXL3I_A8_WS is one constant), so
// the fold's wave-uniform scale is 1 and only the activation's (row, block) scale multiplies; (2) a stacked
// gate_up's blocks read the activation plane of their half (suh or suh.1); (3) the store is EXL3's: the block's 128
// weight rows are one 128-point output Hadamard block, so before the store they are rotated back (bits 0-2 of the
// row in the lane's registers, bit 4 across the wave's two tiles, bit 3 with lane ^ 16, bits 5-6 across the four
// weight waves through the staging LDS, one activation tile at a time), then x WS / sqrt(128) x svh, and stored as
// bf16 / f16 / f32 (one 16- or 32-byte write a lane a tile).
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

namespace exl3i_8a8 {

// ---- the device helpers the tile uses ---------------------------------------------------------------------------
typedef int   i2v __attribute__((ext_vector_type(2)));
typedef float f8v __attribute__((ext_vector_type(8)));
typedef int   i8v __attribute__((ext_vector_type(8)));
typedef float f4v __attribute__((ext_vector_type(4)));

constexpr uint32_t kWave = 32, kBlock = 128, kK = 16;
enum { kSuperK = 64, kTileBytes = 1024, kLaneBytes = 32 };

__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ __forceinline__ f8v mma8(i2v a, i2v b, f8v c) {   // v_wmma_i32_16x16x16_iu8, both signed
    return __builtin_bit_cast(f8v, __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                                       true, a, true, b, __builtin_bit_cast(i8v, c), false));
}

__device__ __forceinline__ uint32_t frag_c(uint32_t lane) { return lane & 15u; }
__device__ __forceinline__ uint32_t frag_k0(uint32_t lane) { return (lane >> 4) * 8u; }
__device__ __forceinline__ uint32_t acc_row(uint32_t lane, uint32_t j) { return (lane >> 4) * 8u + j; }

__device__ __forceinline__ f8v acc_zero() {
    f8v z;
#pragma unroll
    for (int j = 0; j < 8; ++j) z[j] = 0.0f;
    return z;
}

template <uint32_t FK>
__device__ __forceinline__ constexpr uint32_t lds_pitch() { return FK + 8u; }

template <uint32_t FK>
__device__ __forceinline__ i2v frag(const uint8_t* __restrict__ lds, uint32_t row0, uint32_t k0, uint32_t lane) {
    const uint32_t row = row0 + frag_c(lane);
    const uint32_t k = k0 + frag_k0(lane);
    const uint2 v = *(const uint2*)(lds + (size_t)row * lds_pitch<FK>() + k);
    i2v f;
    f[0] = (int)v.x;
    f[1] = (int)v.y;
    return f;
}

struct Stage8 { uint4 v, w; };

__device__ __forceinline__ i2v frag_of(const Stage8& s, uint32_t t) {
    const uint2 g[4] = { make_uint2(s.v.x, s.v.y), make_uint2(s.v.z, s.v.w),
                         make_uint2(s.w.x, s.w.y), make_uint2(s.w.z, s.w.w) };
    i2v f;
    f[0] = (int)g[t].x;
    f[1] = (int)g[t].y;
    return f;
}

__device__ __forceinline__ Stage8 stage8_load(const uint8_t* __restrict__ src, bool live) {
    Stage8 r;
    r.v = make_uint4(0, 0, 0, 0);
    r.w = make_uint4(0, 0, 0, 0);
    if (live) {
        r.v = *(const uint4*)src;
        r.w = *(const uint4*)(src + 16);
    }
    return r;
}

template <uint32_t FK>
__device__ __forceinline__ void stage8_store(uint8_t* __restrict__ lds, uint32_t row, uint32_t k0, Stage8 r) {
    const uint4 v = r.v, w = r.w;
    const uint2 gr[4] = { make_uint2(v.x, v.y), make_uint2(v.z, v.w), make_uint2(w.x, w.y), make_uint2(w.z, w.w) };
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t k = k0 + i * 8;
        *(uint2*)(lds + (size_t)row * lds_pitch<FK>() + k) = gr[i];
    }
}

__device__ __forceinline__ uint32_t bf16_hi_rne(float f) {
    const uint32_t u = __float_as_uint(f);
    return u + 0x7fffu + ((u >> 16) & 1u);
}
__device__ __forceinline__ uint32_t bf16x2_rne(float lo, float hi) {
    return __builtin_amdgcn_perm(bf16_hi_rne(hi), bf16_hi_rne(lo), 0x07060302u);
}

__device__ __forceinline__ float x16(float v) {   // lane ^ 16 (v_permlanex16_b32, as x16_f in exl3r4d_common.hip.h)
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, v), __builtin_bit_cast(int, v),
                                                                  0x76543210, (int)0xFEDCBA98, false, false));
}

// ---- the prefill tile's geometry --------------------------------------------------------
constexpr uint32_t kPTW = 2, kPTA = 4, kPWW = 4, kPWA = 2;
constexpr uint32_t kPBN = kPWW * kPTW * 16, kPBM = kPWA * kPTA * 16;
constexpr uint32_t kPWaves = kPWW * kPWA;
constexpr uint32_t kPFK = 64;
static_assert(kPBN == kBlock, "a block's weight rows are one 128-point output Hadamard block");
constexpr uint32_t kGroupM = 8;
constexpr int kRbBias = 0x4B400000;
constexpr float kRbBiasF = 12582912.0f;
constexpr uint32_t kRbFlush = 8;

// What one launch covers: one column chunk of the weight (its int8 plane), every row.
struct Tile {
    const uint8_t* A;    // int8 [mats][M][K]
    const float* Sa;     // f32 [mats][M][K / 128]
    const uint8_t* W;    // the chunk's plane
    const uint16_t* svh; // f16 [N]
    void* y;             // [M][y_pitch]
    uint32_t M, K, N;    // N: the whole weight's (the seam, svh)
    uint32_t c0;         // the chunk's first column
    uint32_t mats;
    uint32_t y_pitch;    // elements
    uint32_t y_dt;       // 0 bf16, 1 f32, 2 f16
    float C;             // WS / sqrt(128)
    // the activation planes' addressing: row pitches (A in bytes, Sa in floats) and the second plane's offset
    // (exl3i_a8_dev.h a_ld .. sa_ms: the arena's [mats][M][K] or the producer's packed rows)
    uint64_t a_ld, a_ms, sa_ld, sa_ms;
};

template <class TileT>   // Tile (the A8 tile): the store's fields
__device__ __forceinline__ void exl3_store(const TileT& p, uint32_t m, uint32_t n, const float (&r)[8]) {
    float sv[8];
    {
        const uint4 s = *(const uint4*)(p.svh + n);
        const uint32_t w[4] = { s.x, s.y, s.z, s.w };
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            sv[2 * q] = (float)__builtin_bit_cast(_Float16, (uint16_t)(w[q] & 0xFFFFu));
            sv[2 * q + 1] = (float)__builtin_bit_cast(_Float16, (uint16_t)(w[q] >> 16));
        }
    }
    float o[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) o[t] = (r[t] * p.C) * sv[t];
    if (p.y_dt == 1) {
        float* const d = (float*)p.y + (size_t)m * p.y_pitch + n;
        *(f4v*)d = (f4v){ o[0], o[1], o[2], o[3] };
        *(f4v*)(d + 4) = (f4v){ o[4], o[5], o[6], o[7] };
    } else if (p.y_dt == 2) {
        uint32_t w[4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
            w[q] = (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)o[2 * q]) |
                   (uint32_t)__builtin_bit_cast(uint16_t, (_Float16)o[2 * q + 1]) << 16;
        *(uint4*)((uint16_t*)p.y + (size_t)m * p.y_pitch + n) = make_uint4(w[0], w[1], w[2], w[3]);
    } else {
        *(uint4*)((uint16_t*)p.y + (size_t)m * p.y_pitch + n) =
            make_uint4(bf16x2_rne(o[0], o[1]), bf16x2_rne(o[2], o[3]), bf16x2_rne(o[4], o[5]), bf16x2_rne(o[6], o[7]));
    }
}

// The block's (activation tile bm, weight tile bn) under the raster grouping (kGroupM activation tiles a group).
__device__ __forceinline__ void raster(uint32_t& bm, uint32_t& bn) {
    bm = blockIdx.x;
    bn = blockIdx.y;
    if (gridDim.x > kGroupM) {
        const uint32_t per = kGroupM * gridDim.y;
        const uint32_t lin = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t first = (lin / per) * kGroupM, r = lin % per;
        const uint32_t gm = min(gridDim.x - first, kGroupM);
        bm = first + r % gm;
        bn = r / gm;
    }
}

// The tile: gemm_8a8_prefill_body<I8 = true, RB = 32> with the weight scale 1 and EXL3's store (above), at the block
// (bm, bn) (exl3_8a8_body: raster's).
__device__ __forceinline__ void exl3_8a8_body_at(const Tile& p, uint32_t bm, uint32_t bn) {
    constexpr uint32_t P = lds_pitch<kPFK>();
    static_assert(kPBM * 2u == kPWaves * kWave, "two threads stage an activation row");
    __shared__ __attribute__((aligned(16))) uint8_t s_a[2][kPBM * P];
    __shared__ float s_as[2][kPBM];
    static_assert(kPWaves * kPTW * kWave * 8u * 4u <= sizeof(s_a), "the epilogue's exchange in the staging LDS");

    const uint32_t tid = threadIdx.x, lane = tid % kWave;
    const uint32_t wave = __builtin_amdgcn_readfirstlane(tid / kWave);
    const uint32_t ww = wave % kPWW, wa = wave / kPWW;
    const uint32_t M = p.M, K = p.K;
    const uint32_t m0 = bm * kPBM, n0 = bn * kPBN, nabs = p.c0 + n0;
    const uint32_t ksuper = K / kSuperK, nkb = K / kBlock;
    // the stacked gate_up's seam: a block's 128 rows lie on one side (N % 256), and read that side's activation
    const uint32_t mat = (p.mats == 2u && nabs >= p.N / 2u) ? 1u : 0u;
    const uint8_t* const A = p.A + (size_t)mat * p.a_ms;
    const float* const Sa = p.Sa + (size_t)mat * p.sa_ms;
    const size_t a_ld = p.a_ld, sa_ld = p.sa_ld;

    const uint32_t sr = tid / 2u, sc = (tid % 2u) * 32u;
    const uint8_t* const asrc = A + (size_t)min(m0 + sr, M - 1u) * a_ld + sc;
    const float* const ssrc = Sa + (size_t)min(m0 + tid % kPBM, M - 1u) * sa_ld;
    const uint32_t wt0 = n0 / 16u + ww * kPTW;
    const uint8_t* const wsrc = p.W + (size_t)wt0 * ksuper * kTileBytes + lane * kLaneBytes;
    const uint32_t arow = wa * kPTA * 16u;

    f8v raw[kPTW][kPTA], out[kPTW][kPTA];
    float corr[kPTA];
#pragma unroll
    for (uint32_t i = 0; i < kPTW; ++i)
#pragma unroll
        for (uint32_t j = 0; j < kPTA; ++j) {
            out[i][j] = acc_zero();
            raw[i][j] = acc_zero();
        }
#pragma unroll
    for (uint32_t j = 0; j < kPTA; ++j) corr[j] = 0.0f;

    Stage8 pa = stage8_load(asrc, true);
    Stage8 w0[kPTW], w1[kPTW];
#pragma unroll
    for (uint32_t i = 0; i < kPTW; ++i) w0[i] = stage8_load(wsrc + (size_t)i * ksuper * kTileBytes, true);
    s_as[0][tid % kPBM] = ssrc[0];
    float scn = ssrc[min(1u, nkb - 1u)];

    auto tile = [&](uint32_t kb, uint32_t half, Stage8 (&cur)[kPTW], Stage8 (&nxt)[kPTW]) {
        stage8_store<kPFK>(s_a[half], sr, sc, pa);
        if (half == 1u) s_as[(kb + 1u) % 2u][tid % kPBM] = scn;
        lds_barrier();
        const uint32_t c1 = min(kb * kBlock + (half + 1u) * kPFK, K - kPFK);
        pa = stage8_load(asrc + c1, true);
#pragma unroll
        for (uint32_t i = 0; i < kPTW; ++i)
            nxt[i] = stage8_load(wsrc + ((size_t)i * ksuper + c1 / kSuperK) * kTileBytes, true);
        if (half == 1u) scn = ssrc[min(kb + 2u, nkb - 1u)];
#pragma unroll
        for (uint32_t ks = 0; ks < kPFK / kK; ++ks) {
            i2v bf[kPTA];
#pragma unroll
            for (uint32_t j = 0; j < kPTA; ++j) bf[j] = frag<kPFK>(s_a[half], arow + j * 16u, ks * kK, lane);
#pragma unroll
            for (uint32_t i = 0; i < kPTW; ++i) {
                const i2v af = frag_of(cur[i], ks);
#pragma unroll
                for (uint32_t j = 0; j < kPTA; ++j)
                    raw[i][j] = mma8(af, bf[j], (half == 0u && ks == 0u) ? __builtin_bit_cast(f8v, (i8v)(kRbBias))
                                                                         : raw[i][j]);
            }
        }
    };

    for (uint32_t kb = 0; kb < nkb; ++kb) {
        tile(kb, 0u, w0, w1);
        tile(kb, 1u, w1, w0);
        if (kb % kRbFlush == 0u && kb) {   // the last kRbFlush blocks' bias, before this fold
#pragma unroll
            for (uint32_t i = 0; i < kPTW; ++i)
#pragma unroll
                for (uint32_t j = 0; j < kPTA; ++j)
#pragma unroll
                    for (uint32_t t = 0; t < 8; ++t) out[i][j][t] = __builtin_fmaf(-kRbBiasF, corr[j], out[i][j][t]);
#pragma unroll
            for (uint32_t j = 0; j < kPTA; ++j) corr[j] = 0.0f;
        }
#pragma unroll
        for (uint32_t j = 0; j < kPTA; ++j) {
            const float as = s_as[kb % 2u][arow + j * 16u + frag_c(lane)];
            corr[j] += as;
#pragma unroll
            for (uint32_t i = 0; i < kPTW; ++i)
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) out[i][j][t] = __builtin_fmaf(raw[i][j][t], as, out[i][j][t]);
        }
    }

    // ---- EXL3's store. Slot t of tile i of this wave is weight row n = n0 + 32 ww + 16 i + 8 h + t (h = lane >> 4)
    // of activation row m = m0 + arow + 16 j + (lane & 15): the bias out, then the natural-order Hadamard over the 7
    // bits of n - n0 (stage on bit s: low + high, low - high), the 4 weight waves' bits through LDS.
    const float sgn_h = (lane & 16u) ? -1.0f : 1.0f;
    float* const xs = (float*)&s_a[0][0];   // [kPWA][kPWW][kPTW][32 lanes][8]
    lds_barrier();                          // every wave is past its last staged-tile read
#pragma unroll
    for (uint32_t j = 0; j < kPTA; ++j) {   // one activation tile at a time, in place in its accumulators
#pragma unroll
        for (uint32_t i = 0; i < kPTW; ++i) {
            f8v& v = out[i][j];
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) v[t] = __builtin_fmaf(-kRbBiasF, corr[j], v[t]);
#pragma unroll
            for (uint32_t s = 1; s < 8; s <<= 1)   // bits 0-2: the slot
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t)
                    if (!(t & s)) {
                        const float a = v[t], b = v[t + s];
                        v[t] = a + b;
                        v[t + s] = a - b;
                    }
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) v[t] = __builtin_fmaf(v[t], sgn_h, x16(v[t]));   // bit 3: lane ^ 16
        }
#pragma unroll
        for (uint32_t t = 0; t < 8; ++t) {   // bit 4: the wave's two tiles
            const float a = out[0][j][t], b = out[1][j][t];
            out[0][j][t] = a + b;
            out[1][j][t] = a - b;
        }
#pragma unroll
        for (uint32_t i = 0; i < kPTW; ++i) {
            float* const d = xs + (((wa * kPWW + ww) * kPTW + i) * kWave + lane) * 8u;
            *(f4v*)d = (f4v){ out[i][j][0], out[i][j][1], out[i][j][2], out[i][j][3] };
            *(f4v*)(d + 4) = (f4v){ out[i][j][4], out[i][j][5], out[i][j][6], out[i][j][7] };
        }
        lds_barrier();
        const uint32_t m = m0 + arow + j * 16u + frag_c(lane);
#pragma unroll
        for (uint32_t i = 0; i < kPTW; ++i) {
            float r[8];
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t) r[t] = 0.0f;
#pragma unroll
            for (uint32_t w2 = 0; w2 < kPWW; ++w2) {   // bits 5-6: the four weight waves, summed in wave order
                const float sg = (__builtin_popcount(ww & w2) & 1) ? -1.0f : 1.0f;
                const float* const s = xs + (((wa * kPWW + w2) * kPTW + i) * kWave + lane) * 8u;
                const f4v a = *(const f4v*)s, b = *(const f4v*)(s + 4);
#pragma unroll
                for (uint32_t t = 0; t < 4; ++t) {
                    r[t] = __builtin_fmaf(sg, a[t], r[t]);
                    r[4 + t] = __builtin_fmaf(sg, b[t], r[4 + t]);
                }
            }
            if (m < M) exl3_store(p, m, nabs + (ww * kPTW + i) * 16u + acc_row(lane, 0), r);
        }
        lds_barrier();   // the exchange is rewritten for the next j
    }
}

__device__ __forceinline__ void exl3_8a8_body(const Tile& p) {
    uint32_t bm, bn;
    raster(bm, bn);
    exl3_8a8_body_at(p, bm, bn);
}

}  // namespace exl3i_8a8
