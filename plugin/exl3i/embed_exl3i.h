/* embed_exl3i.h -- `embed_exl3i`, the token-embedding gather from an exl3i (EXL3-V3 trellis) table, served from the
 * stored form: the record, suh and svh exactly as the container holds them (exl3import/include/exl3i_encoding.h), no
 * load-time expansion.
 *
 * The table is what the exl3i importer writes for
 *
 *   token_embd.weight  exl3i src=<ourq level of the embedding> codebook=mad [rows=r16]
 *
 * W = diag(svh) . Hb_n . (W_hat . Hb_k) . diag(suh) in radiance's [rows = n_vocab, cols = n_embd] view, W_hat the
 * trellis (16 x 16 tiles, one 16-row strip a record row, its tiles in k order). The output Hadamard runs over 128
 * vocabulary rows, so one token's row needs the 128 rows of its block. The op's row is EXACTLY the oracle's
 * (exl3import/common/exl3i_decode.c exl3i_decode_rows: what a load-time expansion would hold), rounded RNE to bf16:
 *
 *   n      = the stored row of vocabulary id tokens[m] (r16: (id % (V / 8)) / 16 * 128 + id / (V / 8) * 16 + id % 16,
 *            else id); b = n / 128, j = n % 128
 *   per 128 columns: the block's 128 rows of W_hat, each rotated by exl3_had128 (stages h = 1 .. 64 in f32, then
 *            x (float)(1/sqrt(128))) and x suh; the 128-point exl3_had128 down each column, its element j; x svh
 *   x[m,k] = bf16 of that f32 (RNE); an id outside [0, n_vocab) writes zeros
 *
 * The device row computes only the cone of element j of the column butterfly (the same f32 additions, in the same
 * order); the host row runs the oracle's own functions. Both give the oracle's bits (tests/embed_exl3i_test.cpp: host
 * row == exl3i_decode_rows + RNE bf16 in every bit; tests/embed_exl3i_gpu_test.hip: device row == host row).
 *
 *   params    M (tokens), n_embd (% 128), n_vocab (% 128), m (the rate 16 K: even, 2 .. 128), codebook (mad),
 *             r16 (0 / 1: the encoding's /r16 row order)
 *   operands  tokens (in i32 [M]), record (w u32 [n_vocab / 16, n_embd / 16 * m / 2]), suh (w f16 [1, n_embd]),
 *             svh (w f16 [n_vocab, 1]), x (out bf16 [M, n_embd])
 */
#ifndef EMBED_EXL3I_H
#define EMBED_EXL3I_H

#include "rad_abi.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EMBED_EXL3I_OP "embed_exl3i"

enum { EMBED_EXL3I_TOKENS = 0, EMBED_EXL3I_RECORD, EMBED_EXL3I_SUH, EMBED_EXL3I_SVH, EMBED_EXL3I_X, EMBED_EXL3I_N_OPD };

typedef struct EmbedExl3iCall {
    const int32_t*  tokens;
    const uint32_t* record;            /* [n_vocab / 16][row_words] */
    int64_t         row_words;         /* n_embd / 16 * m / 2 */
    const uint16_t* suh;               /* f16 [n_embd] */
    const uint16_t* svh;               /* f16 [n_vocab] */
    uint16_t*       x;                 /* bf16 [M][x_pitch] */
    int64_t         x_pitch;
    int64_t         M, n_embd, n_vocab;
    int32_t         m;                 /* rate m / 16 */
    int32_t         cb;                /* EXL3I_CB_* */
    int32_t         r16;
    int32_t         pad;               /* explicit, zeroed: replay compares the launch's argument bytes */
} EmbedExl3iCall;

const RadOpSchema* embed_exl3i_schema(void);
int embed_exl3i_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int embed_exl3i_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                       const RadTensor* planes, int n_planes, RadLayout* out);
int embed_exl3i_args(const RadArgs* a, EmbedExl3iCall* out);
int embed_exl3i_host_launch(const RadArgs* a, RadStream s);
int embed_exl3i_device_launch(const RadArgs* a, RadStream s);

#if defined(__HIPCC__)
#define EMBED_EXL3I_HD __host__ __device__
#else
#define EMBED_EXL3I_HD
#endif

/* The stored row of vocabulary id v (r16 or natural) in a table of n_vocab rows. */
EMBED_EXL3I_HD static inline int64_t embed_exl3i_row(int64_t v, int64_t n_vocab, int r16) {
    const int64_t e8 = n_vocab / 8;
    return r16 ? (v % e8) / 16 * 128 + v / e8 * 16 + v % 16 : v;
}

#ifdef __cplusplus
}
#endif

#endif /* EMBED_EXL3I_H */
