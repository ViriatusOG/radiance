/* exl3_decode.c -- the shared EXL3 numerics. See exl3_decode.h for the conventions.
 *
 * Bit-level and scalar on purpose: the exl3i decoders (exl3i_decode.c) and the device kernels are written
 * against these functions line by line. Compile with -ffp-contract=off (an FMA would change exl3_had128's
 * f32 roundings).
 */
#include "exl3_decode.h"

#include <math.h>

/* ------------------------------------------------------------------ fp16 */

float exl3_half_to_float(uint16_t h) {
    const int e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float v;
    if (e == 0) v = ldexpf((float)m, -24);                 /* zero and subnormals */
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexpf((float)(m | 0x400), e - 25);           /* (1024 + m) * 2^(e - 25) */
    return (h & 0x8000) ? -v : v;
}

uint16_t exl3_half_from_double(double x) {
    const uint16_t sign = signbit(x) ? 0x8000 : 0;
    const double a = fabs(x);
    double m;
    int e;
    if (isnan(x)) return 0x7E00;
    if (a >= 65520.0) return sign | 0x7C00;                /* past max + half an ulp: inf (RNE) */
    if (a < 0x1p-14) {                                     /* subnormal: units of 2^-24 */
        return sign | (uint16_t)nearbyint(a * 0x1p24);     /* 1024 lands on the smallest normal */
    }
    m = nearbyint(ldexp(frexp(a, &e), 11));                /* a = m * 2^(e - 11), m in [1024, 2048] */
    if (m == 2048.0) { m = 1024.0; ++e; }
    if (e + 14 >= 31) return sign | 0x7C00;
    return sign | (uint16_t)((e + 14) << 10) | (uint16_t)(m - 1024.0);
}

/* ------------------------------------------------------------------ the tile */

void exl3_pos_to_tile(int p, int* r, int* c) {
    static const int off[4] = { 0, 1, 8, 9 };
    const int lane = p / 8, j = p % 8;
    *r = 2 * (lane % 4) + off[j % 4];
    *c = lane / 4 + 8 * (j / 4);
}

int exl3_tile_to_pos(int r, int c) {
    const int lane = 4 * (c % 8) + (r % 8) / 2;
    const int j = 4 * (c / 8) + ((r & 1) | ((r >> 3) << 1));
    return 8 * lane + j;
}

/* ------------------------------------------------------------------ the rotation */

int exl3_had128(float* v, int64_t n) {
    const float scale = 0.08838834764831844055f;        /* 1/sqrt(128) */
    int64_t base, i, j;
    int h;
    if (!v || n < 0 || n % EXL3_HAD) return EXL3_E_SHAPE;
    for (base = 0; base < n; base += EXL3_HAD) {
        float* x = v + base;
        for (h = 1; h < EXL3_HAD; h <<= 1)
            for (i = 0; i < EXL3_HAD; i += 2 * h)
                for (j = i; j < i + h; ++j) {
                    const float a = x[j], b = x[j + h];
                    x[j] = a + b;
                    x[j + h] = a - b;
                }
        for (i = 0; i < EXL3_HAD; ++i) x[i] *= scale;
    }
    return EXL3_OK;
}
