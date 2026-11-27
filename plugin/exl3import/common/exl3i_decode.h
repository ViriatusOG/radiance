/* exl3i_decode.h -- THE ORACLE for "exl3i" weights (include/exl3i_encoding.h): plain C99, slow and readable on purpose.
 * Every exl3i decoder (the quantiser's decode hook, the device rows' tests) is checked against this file.
 *
 * A 256-weight tile at rate K = m / 16 (m even, 2..128; KA = m / 16, MASK = exl3i_mask(m % 16)) is m / 2 u32 words
 * (exllamav3's 16 K int16, read as little-endian u32), one MSB-first bit ring of R = 16 m bits (word w holds ring
 * bits [32 w, 32 w + 32), bit 31 first). Ring position i (0..255) takes D(i) = KA + bit (i mod 16) of MASK new bits;
 * S(i) = D(0) + ... + D(i) is where its 16-bit window ends, its STATE ring bits [S(i) - 16, S(i)), circular. Its
 * VALUE is codebook(state ^ salt(i)) when D(i) = 0 (a sub-1 rate's 0-bit step keeps the state and reads a position-
 * salted index, salt(i) = 0x9E37 (i mod 16 + 1) mod 2^16), codebook(state) otherwise. Ring position i sits at tile row
 * r (k axis) and column c (n axis) of exl3_pos_to_tile(i) -- exllamav3's tensor_core_perm. Integer K (m % 16 = 0) is
 * exllamav3's own trellis exactly.
 *
 * Codebook: mad (the one codebook), q = bytesum(((state * MAD_A + MAD_B) mod 2^32) & 0x3F3F3F3F) - 126 in [-126, 126],
 * value fp16(q 4 K_INV) with K_INV = fp16 0x1EEE (the int8 grid, step 4 K_INV; the constants A 0xD0474B, B 0x2545F491,
 * no output map). A transform that names any other codebook is refused at load (exl3i_encoding_read).
 *
 * The weight is W = diag(suh) . Hb . W_hat . Hb . diag(svh) as for "exl3" (exl3_decode.h), in radiance's [rows = n,
 * cols = k] view. Build with -ffp-contract=off. Reentrant. */
#ifndef EXL3I_DECODE_H
#define EXL3I_DECODE_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __cplusplus
extern "C" {
#endif

#define EXL3I_DECODE_API_VERSION 4   /* 4: the codebook enum is mad alone */

enum { EXL3I_CB_MAD = 0 };   /* the one codebook */
/* mad's constants: the hash's multiplier (< 2^24) and addend */
#define EXL3I_MAD_A 0xD0474Bu
#define EXL3I_MAD_B 0x2545F491u
enum { EXL3I_M_MIN = 2, EXL3I_M_MAX = 128 };   /* rates 1/8 .. 8 */

static inline const char* exl3i_codebook_name(int cb) {
    return cb == EXL3I_CB_MAD ? "mad" : 0;
}
static inline int exl3i_codebook_from_name(const char* s) {
    return s && !strcmp(s, "mad") ? EXL3I_CB_MAD : -1;   /* mad is the one codebook */
}
/* A rate this format has: m even, 2 <= m <= 128 (at m = 128 no extra bits, K8). */
static inline int exl3i_rate_ok(int m) { return m >= EXL3I_M_MIN && m <= EXL3I_M_MAX && m % 2 == 0; }

/* The canonical text of rate m / 16 ("%g": 0.125, 2.25, 4) into buf (>= 16 bytes); the parse of one, -1 unless it
 * is exactly a canonical spelling of a rate that exists. Inline: include/exl3i_encoding.h is header-only. */
static inline const char* exl3i_rate_str(int m, char* buf) {
    snprintf(buf, 16, "%g", m / 16.0);
    return buf;
}
static inline int exl3i_rate_parse(const char* s) {
    char want[16];
    char* end = 0;
    double x;
    int m;
    if (!s || !*s) return -1;
    x = strtod(s, &end);
    if (*end || !(x > 0) || x > 8) return -1;
    m = (int)(x * 16.0 + 0.5);
    if (!exl3i_rate_ok(m) || strcmp(exl3i_rate_str(m, want), s) != 0) return -1;   /* one spelling a rate */
    return m;
}

/* The even step pattern with n extra bits a 16 positions (exllamav3's frac_mask): bit i set iff
 * floor((i + 1) n / 16) - floor(i n / 16) = 1. n = 0: integer K. */
int exl3i_mask(int n);

/* A 16-bit state's value in codebook cb, as fp16 bits (mad; 0 for any other cb). */
uint16_t exl3i_codebook_value(uint32_t state, int cb);
/* mad: S, the byte sum of a state's masked hash (q = S - 126); the int8 grid index q of a state's value (mad: value
 * fp16(q 4 K_INV)), what the integer kernels multiply; 0 for any other cb. */
uint32_t exl3i_mad_sum(uint32_t state);
int exl3i_codebook_q(uint32_t state, int cb);

/* One tile (m / 2 u32 words) -> its 256 values (fp16 bits) in RING order (exl3_pos_to_tile places them); the states
 * too when `states` is not null. -1 for a bad rate or codebook. */
int exl3i_decode_tile(const uint32_t* words, int m, int cb, uint16_t values[256], uint16_t* states);

/* One weight, or a 16-row-aligned row slice of one, as the planes hold it. */
typedef struct Exl3iView {
    int64_t         rows, cols;   /* n, k */
    int             m, cb;        /* rate m / 16, EXL3I_CB_* */
    const uint32_t* record;       /* table rows rows / 16, each exl3i_row_words(cols, m) words: tiles in k order */
    const uint16_t* suh;          /* f16 [cols] */
    const uint16_t* suh1;         /* f16 [cols], the up half of a stacked weight, or null */
    const uint16_t* svh;          /* f16 [rows] */
} Exl3iView;

/* W_hat (the values a kernel multiplies) of rows [row0, row0 + n) as f32 [n][cols]; rows, cols multiples of 16.
 * 0, or -1 (bad arguments) / -2 (shape) / -3 (no memory). */
int exl3i_decode_what_rows(const Exl3iView* v, int64_t row0, int64_t n, float* out);

/* The weight, rows [row0, row0 + n) as f32 [n][cols] (exl3_decode_rows' arithmetic and order). rows, cols multiples
 * of 128 (of 256 rows with suh1). */
int exl3i_decode_rows(const Exl3iView* v, int64_t row0, int64_t n, float* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif
#endif  /* EXL3I_DECODE_H */
