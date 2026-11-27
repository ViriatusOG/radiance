/* exl3i_encoding.h -- EXL3-V3's weight encoding, "exl3i": the EXL3 trellis at any rate K = m / 16 (m even, 2..128: the
 * fractional rates, 1/8 steps, sub-1 K), in its one codebook, mad. A transform that names any other codebook is
 * refused. Header-only; owner Q.
 *
 * Why not a TILED plane of the trellis: the plain `exl3` encoding keeps one K-bit symbol a weight in a uK plane, which
 * needs a whole number of bits a weight. At a fractional rate, position i of a tile takes D(i) = KA or KA + 1 bits,
 * and under the 1/8-step patterns the 16 positions a weight row owns in a tile do not even sum to the same count on
 * every row, so no core dtype and block makes a TILED plane of it. The record does: exllamav3's own tile stream.
 *
 *   scheme     "exl3i"
 *   transform  "<codebook>/h128/k<rate>"      "mad/h128/k2.625", "mad/h128/k0.5": the codebook (mad: IQR's computed
 *                                              code on the int8 grid; version 4: a reader older than that refuses
 *                                              it), the rotation (Hadamard block 128, both sides), the rate in
 *                                              canonical %g form
 *   plane 0    codes   u1  TILED [1 x cols]   one bit a row, zero: the core's encoding rule wants a "codes" plane
 *                                              first; this scheme's data is the record (plane 1)
 *   plane 1    record  u32 TABLE [rows/16 x (cols/16) m/2]
 *                                              the trellis: table row t = weight rows [16 t, 16 t + 16), its cols/16
 *                                              tiles in k order, each 8 K = m/2 little-endian u32 words, one
 *                                              MSB-first ring (common/exl3i_decode.h); a stacked gate_up's up rows
 *                                              follow its gate rows
 *   plane 2    suh     f16 TILED [0 x 1]      input scales
 *   plane 3    svh     f16 TILED [1 x 0]      output scales
 *   plane 4    suh.1   f16 TILED [0 x 1]      STACKED gate_up only: the up half's input scales
 *
 * Row order (version 3): a transform ending "/r16" ("mad/h128/k4/r16") says the weight's rows are stored in R16
 * order (the output head): stored row s holds vocabulary row ((s >> 4) & 7) N / 8 + (s >> 7) 16 + (s & 15), i.e.
 * each 128-row Hadamard block takes one 16-row run from each eighth of the rows (rows a multiple of 128, not stacked).
 * The codes were searched in that basis, so the order stays; the kernel writes output s at that row. A reader older
 * than version 3 refuses the transform (rather than serve the rows permuted).
 *
 * The step pattern is not stored: rate m / 16 has one, the even spread exl3i_mask(m % 16) (exllamav3's frac_mask),
 * and a rate below 1 reads its 0-bit steps salted (exl3i_decode.h). The file is the format; a reader that does not
 * know a rate refuses the encoding.
 *
 * Note: a TABLE plane of a weight declared over several planes goes whole to every slice; the dense op declares each
 * exl3i plane as its own single-plane weight, which the loader cuts in the plane's own elements (core/format/share.cpp)
 * -- the record by 16-row groups, suh / suh.1 along K, svh along the rows -- so a tensor-parallel split is the rank's
 * planes, and the kernel's layout hook (plugin/exl3i/nt.c) checks and transposes them at the op's per-rank geometry. */
#ifndef EXL3I_ENCODING_H
#define EXL3I_ENCODING_H

#include "rad_encoding.h"
#include "../common/exl3i_decode.h"

#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EXL3I_ENCODING_VERSION 4   /* 3: the R16 row order (transform suffix /r16); 4: codebook mad */
#define EXL3I_ROWS_R16_SUFFIX "/r16"
#define EXL3I_SCHEME "exl3i"

enum { EXL3I_PLANE_CODES = 0, EXL3I_PLANE_RECORD = 1, EXL3I_PLANE_SUH = 2, EXL3I_PLANE_SVH = 3,
       EXL3I_PLANE_SUH1 = 4 };

/* Words of the record a 16-row table row holds for `cols` input features at rate m / 16. */
static inline int64_t exl3i_row_words(int64_t cols, int m) { return cols / 16 * (m / 2); }

/* The encoding of a [rows, cols] weight at rate m / 16 in codebook cb (EXL3I_CB_*); `stacked` adds suh.1.
 * RAD_OK, or RAD_E_INVAL for a rate exl3i_rate_ok refuses, an unknown codebook or a shape not in 16s. */
static inline int exl3i_encoding_make(int m, int cb, int stacked, int64_t rows, int64_t cols, RadEncoding* out) {
    char t[RAD_ENC_STR], r[16];
    const char* cbn = exl3i_codebook_name(cb);
    if (!out || !cbn || !exl3i_rate_ok(m) || rows <= 0 || cols <= 0 || rows % 16 || cols % 16) return RAD_E_INVAL;
    rad_enc_clear(out);
    rad_enc_copy_str(out->scheme, EXL3I_SCHEME);
    snprintf(t, sizeof t, "%s/h128/k%.5s", cbn, exl3i_rate_str(m, r));   /* rates are at most 5 characters */
    rad_enc_copy_str(out->transform, t);
    rad_enc_add_plane(out, "codes", RAD_U1, 1, cols);
    rad_enc_add_table(out, "record", RAD_U32, rows / 16, exl3i_row_words(cols, m));
    rad_enc_add_plane(out, "suh", RAD_F16, 0, 1);
    rad_enc_add_plane(out, "svh", RAD_F16, 1, 0);
    if (stacked) rad_enc_add_plane(out, "suh.1", RAD_F16, 0, 1);
    return RAD_OK;
}

/* R16 row order (version 3): output row s of the weight is row exl3i_r16_row(s, rows) of the natural order. */
static inline int64_t exl3i_r16_row(int64_t s, int64_t rows) { return ((s >> 4) & 7) * (rows / 8) + (s >> 7) * 16 + (s & 15); }
/* The same encoding with the R16 row order: RAD_OK, or RAD_E_INVAL when the rows are not a multiple of 128, the weight
 * is stacked (suh.1) or the transform has no room. */
static inline int exl3i_encoding_make_r16(int m, int cb, int stacked, int64_t rows, int64_t cols, RadEncoding* out) {
    char t[RAD_ENC_STR];
    const int rc = exl3i_encoding_make(m, cb, stacked, rows, cols, out);
    if (rc != RAD_OK) return rc;
    if (stacked || rows % 128) return RAD_E_INVAL;
    if (strlen(out->transform) + strlen(EXL3I_ROWS_R16_SUFFIX) >= RAD_ENC_STR) return RAD_E_INVAL;
    snprintf(t, sizeof t, "%s%s", out->transform, EXL3I_ROWS_R16_SUFFIX);
    rad_enc_copy_str(out->transform, t);
    return RAD_OK;
}
/* Read one back: RAD_E_UNSUPPORTED when the scheme is not exl3i (a decoder answers that and the core asks the next);
 * RAD_E_FORMAT when it says exl3i but is not exactly what exl3i_encoding_make writes for some rate, codebook,
 * stacking and shape; RAD_OK with the fields filled (any pointer may be null). */
static inline int exl3i_encoding_read_r16(const RadEncoding* e, int* m, int* cb, int* stacked, int64_t* rows,
                                          int64_t* cols, int* r16) {
    RadEncoding want;
    char buf[RAD_ENC_STR];
    char* k;
    int mm, c, s, o;
    int64_t r, cl;
    size_t n;
    const size_t ns = strlen(EXL3I_ROWS_R16_SUFFIX);
    if (!e || !rad_enc_is(e, EXL3I_SCHEME)) return RAD_E_UNSUPPORTED;
    if (e->n_planes < 4 || e->n_planes > 5) return RAD_E_FORMAT;
    memcpy(buf, e->transform, RAD_ENC_STR);
    buf[RAD_ENC_STR - 1] = 0;
    n = strlen(buf);
    o = n > ns && !strcmp(buf + n - ns, EXL3I_ROWS_R16_SUFFIX);
    if (o) buf[n - ns] = 0;
    k = strstr(buf, "/h128/k");
    if (!k) return RAD_E_FORMAT;
    *k = 0;
    c = exl3i_codebook_from_name(buf);
    mm = exl3i_rate_parse(k + 7);
    s = e->n_planes == 5;
    cl = e->plane[EXL3I_PLANE_CODES].block[1];
    r = e->plane[EXL3I_PLANE_RECORD].extent[0] * 16;
    if (c < 0 || mm < 0 || (o ? exl3i_encoding_make_r16(mm, c, s, r, cl, &want) : exl3i_encoding_make(mm, c, s, r, cl, &want)) != RAD_OK ||
        !rad_enc_equal(e, &want))
        return RAD_E_FORMAT;
    if (m) *m = mm;
    if (cb) *cb = c;
    if (stacked) *stacked = s;
    if (rows) *rows = r;
    if (cols) *cols = cl;
    if (r16) *r16 = o;
    return RAD_OK;
}
/* exl3i_encoding_read_r16 without the row order (an R16 weight reads too; ask exl3i_encoding_read_r16 for it). */
static inline int exl3i_encoding_read(const RadEncoding* e, int* m, int* cb, int* stacked, int64_t* rows,
                                      int64_t* cols) {
    return exl3i_encoding_read_r16(e, m, cb, stacked, rows, cols, 0);
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* EXL3I_ENCODING_H */
