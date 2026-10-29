/* r3d_codecs.h -- the numeric codecs, shared by every libr3d unit and the host.
 *
 * rad_plugin.h's bf16 / f16 / fp8 / int8 rounders, restated so a device kernel can call them
 * (rad_plugin.h's are `static inline` host functions a `__global__` cannot reach). libref, the
 * oracle every libr3d row is checked against at kbench, is built from those, so these must match
 * bit for bit. r3d_selftest.cpp compares the two over the whole fp8 and bf16 code spaces and 2^20
 * random f32s, so a one-ulp drift fails there.
 *
 * The contract, from ref_common.h and rad_plugin.h: f32 accumulation, IEEE round-to-nearest-ties-
 * to-even narrowing, no fused multiply-add, no reassociation, subnormals kept. Widening is exact.
 * `-ffp-contract=off` is a correctness flag; rad_hipify puts it on every HIP source.
 *
 * R3D_HD is `__host__ __device__` in a HIP translation unit and nothing in a host one, so the same
 * header serves kernels and r3d_selftest.cpp. Code that uses a GPU instruction sits behind
 * __HIP_DEVICE_COMPILE__ with a plain-C++ host equivalent.
 */
#ifndef R3D_CODECS_H
#define R3D_CODECS_H

#include <cstdint>
#include <cstring>

/* The HIP runtime comes first, before R3D_HD: `__forceinline__` is a hip_runtime.h macro, and
 * r3d_common.h includes the codecs before the runtime, so without this it would parse as a type
 * name. Host translation units (the row table, the operand descriptions, r3d_selftest.cpp) pull in
 * no ROCm header. */
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#if defined(__HIPCC__)
#define R3D_HD __device__ __host__ __forceinline__
#else
#define R3D_HD inline
#endif

/* The bit cast. Device code cannot call `memcpy` (the HIP headers do not declare a
 * `__host__ __device__` one on every release), so the device path uses the compiler's bit cast.
 * The host path keeps memcpy, which folds to a move, because GCC has `__builtin_bit_cast` only in
 * C++20 mode and r3d_selftest.cpp builds at C++17. */
R3D_HD uint32_t bits_of(float f) {
#if defined(__HIPCC__)
    return __builtin_bit_cast(uint32_t, f);
#else
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
#endif
}

R3D_HD float float_of(uint32_t u) {
#if defined(__HIPCC__)
    return __builtin_bit_cast(float, u);
#else
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
#endif
}


/* ================================================================== bf16
 *
 * bf16 is f32's top half: widening is a shift, narrowing one add and a shift.
 *
 * `(u >> 16) & 1` is the tie-break: it adds one more than the halfway point when the surviving bit
 * is odd, so ties go to even. Overflow carries into the bf16 infinity. A NaN is forced quiet so it
 * cannot round into an infinity, as in rad_f32_to_bf16 and libref. */
R3D_HD float bf16_to_f32(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    return float_of(u);
}

R3D_HD uint16_t f32_to_bf16(float f) {
    const uint32_t u = bits_of(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* The same rounding without the final shift: the rounded value in a word's high half, the form a
 * packed store wants. gfx11 in this ROCm has no v_cvt_pk_bf16_f32 (llvm-mc rejects it on gfx1100,
 * gfx1102 and gfx1151), so a bf16 pair is two of these and one byte permute. */
R3D_HD uint32_t bf16_hi_rne(float f) {
    const uint32_t u = bits_of(f);
    return u + 0x7fffu + ((u >> 16) & 1u);
}

/* Two such words as one bf16 pair, `lo`'s value in the low half: one v_perm_b32. */
R3D_HD uint32_t bf16_pack_hi(uint32_t lo, uint32_t hi) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_perm(hi, lo, 0x07060302u);
#else
    return ((lo >> 16) & 0xffffu) | (hi & 0xffff0000u);
#endif
}

R3D_HD uint32_t bf16x2_rne(float lo, float hi) {
    return bf16_pack_hi(bf16_hi_rne(lo), bf16_hi_rne(hi));
}

/* A bf16 pair word's two values as floats: a shift or a mask, no conversion. */
R3D_HD float bf16w_lo(uint32_t w) {
    const uint32_t u = w << 16;
    return float_of(u);
}
R3D_HD float bf16w_hi(uint32_t w) {
    const uint32_t u = w & 0xffff0000u;
    return float_of(u);
}

/* ================================================================== f16
 *
 * RTNE both ways with subnormals kept, as rad_f16_to_f32 and rad_f32_to_f16 do. The hardware
 * convert is not used: its rounding mode is caller-settable and it flushes subnormal halves. */
R3D_HD float f16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t)(h >> 15) << 31;
    const uint32_t e = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t m = (uint32_t)h & 0x3ffu;
    uint32_t u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;                       /* subnormal: renormalise into f32's range */
            while (!(m & 0x400u)) { m <<= 1; ++sh; }
            m &= 0x3ffu;
            u = s | ((uint32_t)(127 - 15 - sh + 1) << 23) | (m << 13);
        }
    } else if (e == 0x1fu) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e - 15u + 127u) << 23) | (m << 13);
    }
    return float_of(u);
}

R3D_HD uint16_t f32_to_f16(float f) {
    const uint32_t u = bits_of(f);
    const uint32_t s = (u >> 16) & 0x8000u;
    const uint32_t a = u & 0x7fffffffu;

    if (a >= 0x7f800000u)                                      /* inf or nan */
        return (uint16_t)(s | 0x7c00u | (a > 0x7f800000u ? ((a >> 13) & 0x3ffu) | 0x200u : 0u));
    if (a >= 0x47800000u) return (uint16_t)(s | 0x7c00u);      /* overflow -> inf */
    if (a <  0x33000000u) return (uint16_t)s;                  /* underflow -> zero */

    const int e = (int)(a >> 23) - 127;
    uint32_t m = a & 0x7fffffu;
    if (e < -14) {                                             /* subnormal half */
        const int sh = -14 - e + 13;
        m |= 0x800000u;
        const uint32_t lo = m & ((1u << sh) - 1u);
        uint32_t q = m >> sh;
        const uint32_t half = 1u << (sh - 1);
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return (uint16_t)(s | q);
    }
    uint32_t q = m >> 13;
    const uint32_t lo = m & 0x1fffu;
    uint32_t he = (uint32_t)(e + 15);
    if (lo > 0x1000u || (lo == 0x1000u && (q & 1u))) ++q;
    if (q == 0x400u) { q = 0; ++he; }                          /* rounding carried into the exp */
    if (he >= 0x1fu) return (uint16_t)(s | 0x7c00u);
    return (uint16_t)(s | (he << 10) | q);
}

/* Two f32 as one dword of two f16, `lo` in the low half.
 *
 * v_cvt_pkrtz_f16_f32 rounds toward zero, not RTNE, so this is only called with values f16
 * represents exactly: a widened bf16 (seven mantissa bits into ten) or an E4M3 code (three into
 * ten). For those both paths give the same bits. */
R3D_HD uint32_t f16x2_pack(float lo, float hi) {
#if defined(__HIP_DEVICE_COMPILE__)
    /* The builtin returns a pair of __fp16; its bits are the dword the dot2 reads. */
    return __builtin_bit_cast(uint32_t, __builtin_amdgcn_cvt_pkrtz(lo, hi));
#else
    return (uint32_t)f32_to_f16(lo) | ((uint32_t)f32_to_f16(hi) << 16);
#endif
}

/* A dword of two bf16 as a dword of two f16.
 *
 * Exact over f16's normal range, which makes the dot2 path sound: bf16 has seven mantissa bits and
 * f16 ten, so within [2^-14, 65504] widening costs no rounding and a dot over widened bf16 forms
 * the sums libref forms over its bf16 operands.
 *
 * Outside that range: above 65504 a magnitude becomes an infinity. Below 2^-14 the operand is an
 * f16 subnormal on a coarser grid than bf16's, and v_cvt_pkrtz_f16_f32 truncates toward zero;
 * below 2^-24 it flushes. The low edge does not affect a served model: an activation below 2^-14
 * times a weight of 0.01 is 1e-18 against an accumulator of order one, and the quantised rows keep
 * the scale on the block, not the element, so scale-reduced operands are scaled back before use. No
 * checkpoint's weights live below 2^-14. Rows whose band can reach 65504 say so in their `shape`
 * text. */
R3D_HD uint32_t bf16x2_to_f16x2(uint32_t w) {
    return f16x2_pack(bf16w_lo(w), bf16w_hi(w));
}

/* ================================================================== fp8 e4m3 ("e4m3fn")
 *
 * 1-4-3, bias 7, no infinity, S1111111 the only NaN, 448 the largest finite. The KV-cache and
 * block-scaled weight format, so the attention oracle and every fp8 row depend on its rounding.
 *
 * RDNA3 has no fp8 instructions: v_cvt_pk_fp8_f32 and v_cvt_pk_f32_fp8 need the fp8 feature. The
 * builtins are declared for gfx11, so a `__has_builtin` guard (as in libr4d) passes, but they are
 * rejected at instruction selection. Both directions are bit maths, about eleven VALU per encode
 * and six per decode. */
R3D_HD float e4m3_to_f32(uint8_t b) {
    const uint32_t s = (uint32_t)(b >> 7) << 31;
    const uint32_t e = (uint32_t)(b >> 3) & 0xfu;
    uint32_t m = (uint32_t)b & 0x7u;
    uint32_t u;
    if (e == 0u) {
        if (m == 0u) {
            u = s;
        } else {
            int sh = 0;                       /* subnormal: 2^-9 * m/8 */
            while (!(m & 0x8u)) { m <<= 1; ++sh; }
            m &= 0x7u;
            u = s | ((uint32_t)(127 - 6 - sh) << 23) | (m << 20);
        }
    } else if (e == 0xfu && m == 0x7u) {
        u = s | 0x7fc00000u;                  /* the single NaN encoding */
    } else {
        u = s | ((e - 7u + 127u) << 23) | (m << 20);
    }
    return float_of(u);
}

/* Saturates at 448 rather than overflowing, because a NaN from one large activation spreads
 * through the whole row; every fp8 cast in this project clamps. The clamp is inside the codec
 * because there is no hardware convert to clamp in front of. */
R3D_HD uint8_t f32_to_e4m3(float f) {
    const uint32_t u = bits_of(f);
    const uint8_t s = (uint8_t)((u >> 24) & 0x80u);
    const uint32_t a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return (uint8_t)(s | 0x7fu);          /* nan */
    if (a >= 0x43e00000u) return (uint8_t)(s | 0x7eu);         /* >= 448: the largest finite */
    const int e = (int)(a >> 23) - 127;
    uint32_t m = a & 0x7fffffu;
    /* The smallest subnormal is 2^-9, so everything below half of it (exponent -11 and below)
     * rounds to zero. Exponent -10 is [2^-10, 2^-9): its tie point 2^-10 rounds to the even zero
     * and the rest up to 2^-9, which the subnormal branch does with a 24-bit shift. */
    if (e < -10) return s;
    if (e < -6) {                                              /* subnormal fp8 */
        const int sh = -6 - e + 20;
        m |= 0x800000u;
        const uint32_t lo = m & ((1u << sh) - 1u);
        uint32_t q = m >> sh;
        const uint32_t half = 1u << (sh - 1);
        if (lo > half || (lo == half && (q & 1u))) ++q;
        return (uint8_t)(s | q);
    }
    uint32_t q = m >> 20;
    const uint32_t lo = m & 0xfffffu;
    int ee = e + 7;
    if (lo > 0x80000u || (lo == 0x80000u && (q & 1u))) ++q;
    if (q == 0x8u) { q = 0; ++ee; }
    if (ee > 15 || (ee == 15 && q >= 7u)) return (uint8_t)(s | 0x7eu);
    return (uint8_t)(s | ((uint32_t)ee << 3) | q);
}

/* Two codes in one dword, the low element in the low byte: one lane's store. */
R3D_HD uint32_t e4m3x2_pack(float lo, float hi) {
    return (uint32_t)f32_to_e4m3(lo) | ((uint32_t)f32_to_e4m3(hi) << 8);
}

/* A dword of two E4M3 codes as a dword of two f16.
 *
 * Exact (e4m3 has four exponent and three mantissa bits, f16 five and ten). This is the fp8 matrix
 * path: a code becomes its f16 value, dot2 multiplies it, and the block scale is applied once per
 * block. Without fp8 hardware the decode is VALU work, done where the weight bytes are read. */
R3D_HD uint32_t e4m3x2_to_f16x2(uint32_t w) {
    return f16x2_pack(e4m3_to_f32((uint8_t)(w & 0xffu)), e4m3_to_f32((uint8_t)(w >> 8)));
}

/* ================================================================== int8
 *
 * Symmetric int8: amax/127, RTNE, saturating. f16 holds every int8 exactly, so widening into a
 * dot product is exact and the scale is the only float in the path. */
R3D_HD int8_t f32_to_i8_rtne(float f) {
    if (!(f == f)) return 0;
    /* Round to nearest, ties to even, in integer arithmetic, then clamp. The clamp follows the
     * rounding, not an early return: 127.5 rounds to the even 128 and clamps to 127, as
     * rad_round_even_sat does. */
    const float fl = __builtin_floorf(f);
    const float frac = f - fl;
    int32_t q = (int32_t)fl;
    if (frac > 0.5f) ++q;
    else if (frac == 0.5f && (q & 1)) ++q;
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

R3D_HD uint32_t i8x2_pack(float lo, float hi) {
    return (uint32_t)(uint8_t)f32_to_i8_rtne(lo) | ((uint32_t)(uint8_t)f32_to_i8_rtne(hi) << 8);
}

R3D_HD uint32_t i8x2_to_f16x2(uint32_t w) {
    return f16x2_pack((float)(int8_t)(uint8_t)(w & 0xffu), (float)(int8_t)(uint8_t)(w >> 8));
}

/* ================================================================== the dot product
 *
 * Two f16 products into an f32 accumulator in one instruction: v_dot2acc_f32_f16, which
 * __builtin_amdgcn_fdot2 lowers to on every gfx11 target (verified in the emitted asm on gfx1100,
 * gfx1102 and gfx1151). RDNA3 pairs it as v_dual_dot2acc_f32_f16 where the scheduler can.
 *
 * Invariant: the last argument is always false. RDNA3 has no bf16 dot2, yet this compiler accepts
 * the bf16 selector and emits the f16 instruction, so a bf16 pair passed with sel=true is silently
 * read as two f16s. bf16 operands are widened exactly with bf16x2_to_f16x2.
 *
 * The two products go into `acc` in the order written, so a kernel walking k ascending follows
 * libref's summand order (`s += a[k]*b[k]` with -ffp-contract=off). The result is not bit-exact:
 * v_dot2acc_f32_f16 folds both products and the accumulator with guard bits and rounds once, where
 * the reference rounds each product and sum. The difference is under an f32 ulp per pair, a random
 * walk of about sqrt(K/2) ulps over a K-long dot: about a thousandth of the bf16 the store narrows
 * to, and inside what `rad-kbench` measures (as with libr4d's WMMA accumulation). What every loop
 * here must keep is the reference's summand order: k ascending, one accumulator per output element,
 * no reassociation across lanes.
 *
 * The host path is the same arithmetic with each product rounded first, for tests; it is not the
 * hardware's rounding. */
R3D_HD float dot2_f16x2(uint32_t a, uint32_t b, float acc) {
#if defined(__HIP_DEVICE_COMPILE__)
    typedef __fp16 v2h __attribute__((ext_vector_type(2)));
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(v2h, a), __builtin_bit_cast(v2h, b), acc,
                                  false);
#else
    const uint32_t alo = a & 0xffffu, ahi = a >> 16;
    const uint32_t blo = b & 0xffffu, bhi = b >> 16;
    return acc + f16_to_f32((uint16_t)alo) * f16_to_f32((uint16_t)blo)
               + f16_to_f32((uint16_t)ahi) * f16_to_f32((uint16_t)bhi);
#endif
}

/* Four f16 products into one accumulator, the step every libr3d matmul runs. The two dot2s form a
 * dependency chain of two; a kernel keeps four such chains in flight. */
R3D_HD float dot4_f16x4(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1, float acc) {
    return dot2_f16x2(a1, b1, dot2_f16x2(a0, b0, acc));
}

#endif  /* R3D_CODECS_H */
