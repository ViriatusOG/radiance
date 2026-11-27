#ifndef RAD_KV_FP8_H
#define RAD_KV_FP8_H

/* Block-scaled E4M3 cache entries (docs/OPS.md "Block-scaled E4M3 cache entries"): one compressed
 * or index entry of n values as n E4M3 codes, then n / block scale bytes, then zeros to a multiple
 * of 16 bytes. A value is code x scale in f32: two E4M3 significands (4 bits each) multiply to at
 * most 8 significant bits, so the product is exact in f32 and in bf16, and a reader widening an
 * entry gets the very bf16 value an exact entry was packed from.
 *
 *   e4m3_e4m3_16    an E4M3 scale a block of 16 (quant_roundtrip fp4_e2m1 / e4m3 / 16 output)
 *   e4m3_ue8m0_32   a power-of-two scale a block of 32: byte e is 2^(e - 127), 0 is zero
 *                   (quant_roundtrip fp4_e2m1 or fp8_e4m3 / ue8m0 / 32 output)
 *
 * entry_store packs (rad_kvf_pack_block); attn_sparse_sink (latent_format) and index_score
 * (key_format) widen (rad_kvf_value). The packing is exact for every block the named
 * quant_roundtrip produced:
 *   - e4m3_e4m3_16: the block's own scale s is amax / c for its largest code c in {6, 4, 3, 2,
 *     1.5, 1, 0.5}, so searching those for an E4M3 scale under which every value comes back bit
 *     for bit finds one (re-deriving s by quant_roundtrip's rule is not exact where E4M3 is
 *     subnormal). A block no scale reproduces (not quant_roundtrip output) takes the rule, lossy.
 *   - e4m3_ue8m0_32: the smallest power of two s' with amax / s' <= 448 (at least 2^-126) is at
 *     most the block's own scale, so every value / s' is its code times 2^j, j >= 0, at most 448:
 *     an E4M3 value. Other blocks round to nearest, lossy.
 * The sign of zero is kept (code 0x80).
 *
 * Plain C (abi_c_test), and host-and-device under HIP, so libref, libavx and libr4d's entry_store
 * run this very code; the E4M3 conversions are rad_plugin.h's, restated here for the device. */
#include <stdint.h>
#include <string.h>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define RAD_KVF_HD __host__ __device__
#else
#define RAD_KVF_HD
#endif

enum {
    RAD_KVF_BF16 = 0,          /* bf16 values, 2 bytes each (the default) */
    RAD_KVF_E4M3_E4M3_16 = 1,
    RAD_KVF_E4M3_UE8M0_32 = 2
};

/* A format name to its RAD_KVF_*; null or "bf16" is RAD_KVF_BF16, anything unknown -1. Host. */
static inline int rad_kvf_parse(const char* s) {
    if (!s || !strcmp(s, "bf16")) return RAD_KVF_BF16;
    if (!strcmp(s, "e4m3_e4m3_16")) return RAD_KVF_E4M3_E4M3_16;
    if (!strcmp(s, "e4m3_ue8m0_32")) return RAD_KVF_E4M3_UE8M0_32;
    return -1;
}

/* Values a scale covers; 0 for bf16. */
static inline RAD_KVF_HD int64_t rad_kvf_block(int f) {
    return f == RAD_KVF_E4M3_E4M3_16 ? 16 : f == RAD_KVF_E4M3_UE8M0_32 ? 32 : 0;
}

/* Bytes of one entry of n values (n a multiple of the block); 0 when n does not fit the format. */
static inline RAD_KVF_HD int64_t rad_kvf_bytes(int f, int64_t n) {
    const int64_t blk = rad_kvf_block(f);
    if (f == RAD_KVF_BF16) return 2 * n;
    if (blk < 1 || n < 1 || n % blk) return 0;
    return (n + n / blk + 15) / 16 * 16;
}

typedef union { float f; uint32_t u; } RadKvfBits;
static inline RAD_KVF_HD uint32_t rad_kvf_bits(float x) { RadKvfBits b; b.f = x; return b.u; }
static inline RAD_KVF_HD float rad_kvf_float(uint32_t u) { RadKvfBits b; b.u = u; return b.f; }

/* rad_plugin.h's rad_fp8e4m3_to_f32: OCP E4M3 (e4m3fn), S1111111 the only NaN. */
static inline RAD_KVF_HD float rad_kvf_e4m3(uint8_t b) {
    const uint32_t s = (uint32_t)(b >> 7) << 31;
    const uint32_t e = (uint32_t)(b >> 3) & 0xfu;
    uint32_t m = (uint32_t)b & 0x7u;
    if (e == 0u) {
        int sh = 0;
        if (m == 0u) return rad_kvf_float(s);
        while (!(m & 0x8u)) { m <<= 1; ++sh; }
        m &= 0x7u;
        return rad_kvf_float(s | ((uint32_t)(127 - 6 - sh) << 23) | (m << 20));
    }
    if (e == 0xfu && m == 0x7u) return rad_kvf_float(s | 0x7fc00000u);
    return rad_kvf_float(s | ((e - 7u + 127u) << 23) | (m << 20));
}

/* rad_plugin.h's rad_f32_to_fp8e4m3: nearest even, saturating at +-448, NaN to S1111111. */
static inline RAD_KVF_HD uint8_t rad_kvf_to_e4m3(float f) {
    const uint32_t u = rad_kvf_bits(f);
    const uint8_t s = (uint8_t)((u >> 24) & 0x80u);
    const uint32_t a = u & 0x7fffffffu;
    int e;
    uint32_t m;
    if (a > 0x7f800000u) return (uint8_t)(s | 0x7fu);
    if (a >= 0x43e00000u) return (uint8_t)(s | 0x7eu);
    e = (int)(a >> 23) - 127;
    m = a & 0x7fffffu;
    if (e < -10) return s;
    if (e < -6) {
        const int sh = -6 - e + 20;
        uint32_t lo, q, half;
        m |= 0x800000u;
        lo = m & ((1u << sh) - 1u);
        q = m >> sh;
        half = 1u << (sh - 1);
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return (uint8_t)(s | q);
    }
    {
        uint32_t q = m >> 20;
        const uint32_t lo = m & 0xfffffu;
        int ee;
        if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
        ee = e + 7;
        if (q == 0x8u) { q = 0; ++ee; }
        if (ee > 15 || (ee == 15 && q >= 7u)) return (uint8_t)(s | 0x7eu);
        return (uint8_t)(s | ((uint32_t)ee << 3) | q);
    }
}

/* A scale byte's f32. */
static inline RAD_KVF_HD float rad_kvf_scale(int f, uint8_t b) {
    return f == RAD_KVF_E4M3_E4M3_16 ? rad_kvf_e4m3(b) : rad_kvf_float((uint32_t)b << 23);
}

/* Value d of an entry `e` of n values. */
static inline RAD_KVF_HD float rad_kvf_value(int f, const uint8_t* e, int64_t n, int64_t d) {
    return rad_kvf_e4m3(e[d]) * rad_kvf_scale(f, e[n + d / rad_kvf_block(f)]);
}

/* One block's codes under scale `sc` (> 0), each the E4M3 nearest x / sc; returns 1 when every
 * code times sc gives x back bit for bit. */
static inline RAD_KVF_HD int rad_kvf_codes(const float* x, int nb, float sc, uint8_t* codes) {
    int exact = 1, i;
    for (i = 0; i < nb; ++i) {
        const uint8_t c = rad_kvf_to_e4m3(x[i] / sc);
        codes[i] = c;
        if (rad_kvf_bits(rad_kvf_e4m3(c) * sc) != rad_kvf_bits(x[i])) exact = 0;
    }
    return exact;
}

/* One block of rad_kvf_block(f) values `x` (bf16 values as f32) packed: its codes into `codes`,
 * its scale byte returned. */
static inline RAD_KVF_HD uint8_t rad_kvf_pack_block(int f, const float* x, uint8_t* codes) {
    const int nb = (int)rad_kvf_block(f);
    float am = 0.0f;
    int i;
    for (i = 0; i < nb; ++i) {
        const float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > am) am = a;
    }
    if (!(am > 0.0f)) {                  /* zeros (and NaN, which no cache value is): signs only */
        for (i = 0; i < nb; ++i) codes[i] = (uint8_t)((rad_kvf_bits(x[i]) >> 24) & 0x80u);
        return 0;
    }
    if (f == RAD_KVF_E4M3_E4M3_16) {
        uint8_t sb;
        int c;
        for (c = 0; c < 7; ++c) {
            /* amax / c, c = 6, 4, 3, 2, 1.5, 1, 0.5: the largest e2m1 code */
            const float cand = c == 0 ? 6.0f : c == 1 ? 4.0f : c == 2 ? 3.0f : c == 3 ? 2.0f
                             : c == 4 ? 1.5f : c == 5 ? 1.0f : 0.5f;
            const float sc = am / cand;
            sb = rad_kvf_to_e4m3(sc);
            if (sb == 0 || rad_kvf_bits(rad_kvf_e4m3(sb)) != rad_kvf_bits(sc)) continue;
            if (rad_kvf_codes(x, nb, sc, codes)) return sb;
        }
        /* quant_roundtrip's rule, fp8(max(amax, 6 * 2^-9) / 6): lossy */
        sb = rad_kvf_to_e4m3((am > 6.0f * 0.001953125f ? am : 6.0f * 0.001953125f) / 6.0f);
        (void)rad_kvf_codes(x, nb, rad_kvf_e4m3(sb), codes);
        return sb;
    }
    {
        /* 2^e with am / 2^e <= 448, the smallest: am in [2^E, 2^(E+1)) gives E - 8 or E - 7 */
        int e = (int)((rad_kvf_bits(am) >> 23) & 0xffu) - 127 - 8;
        if (e < -126) e = -126;
        if (am > 448.0f * rad_kvf_float((uint32_t)(e + 127) << 23)) ++e;
        (void)rad_kvf_codes(x, nb, rad_kvf_float((uint32_t)(e + 127) << 23), codes);
        return (uint8_t)(e + 127);
    }
}

/* An entry of n values `x` packed into `e` (rad_kvf_bytes(f, n) bytes, the padding zeroed). */
static inline RAD_KVF_HD void rad_kvf_pack(int f, const float* x, int64_t n, uint8_t* e) {
    const int64_t nb = rad_kvf_block(f), bytes = rad_kvf_bytes(f, n);
    int64_t b;
    for (b = 0; b < n / nb; ++b) e[n + b] = rad_kvf_pack_block(f, x + b * nb, e + b * nb);
    for (b = n + n / nb; b < bytes; ++b) e[b] = 0;
}

#endif /* RAD_KV_FP8_H */
