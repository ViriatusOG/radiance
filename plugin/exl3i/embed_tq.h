/* embed_tq.h -- `embed_tq`, the token-embedding gather from a table-quantised, Hadamard-rotated table.
 *
 * The table is what radiance's rtn quantiser writes for
 *
 *   token_embd.weight  rtn  table=<2^B values> group=128 scale=bf16 transform=fwht128 rule=search
 *
 * the affine scheme with a codebook: codes uB [n_vocab, n_embd] (a bit stream along each row, lowest bits
 * first, each row on a byte), a bf16 scale per (row, 128 columns), an f32 table of 2^B values, and the fwht128
 * transform. A row decodes exactly as the core's enc_decode_rows does it (core/format/encoding.cpp):
 *
 *   q[k] = table[code[k]] * scale[k / 128]        f32
 *   w    = H128 q per 128 columns                 the unnormalised Sylvester butterfly, (a + b) low, (a - b) high,
 *                                                 stages h = 1, 2, .. 64 (the 1/128 lives in the stored scale)
 *   x[m] = bf16(w) of row tokens[m]               round to nearest even; an id outside [0, n_vocab) writes zeros
 *
 * so the host row and the device row give the same bits, and both the values the converter's own decode does.
 *
 *   params    M (tokens), n_embd (% 128), n_vocab, bits (1..8)
 *   operands  tokens (in i32 [M]), codes (w uB [n_vocab, n_embd]), scale (w bf16 [n_vocab, n_embd / 128]),
 *             table (w f32 [1, 2^bits]), x (out bf16 [M, n_embd])
 */
#ifndef EMBED_TQ_H
#define EMBED_TQ_H

#include "rad_abi.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EMBED_TQ_OP "embed_tq"

enum { EMBED_TQ_TOKENS = 0, EMBED_TQ_CODES, EMBED_TQ_SCALE, EMBED_TQ_TABLE, EMBED_TQ_X, EMBED_TQ_N_OPD };

typedef struct EmbedTqCall {
    const int32_t*  tokens;
    const uint8_t*  codes;
    int64_t         codes_row_bytes;   /* ceil(n_embd * bits / 8) */
    const uint16_t* scale;             /* bf16 [n_vocab][n_embd / 128] */
    const float*    table;             /* [2^bits] */
    uint16_t*       x;                 /* bf16 [M][x_pitch] */
    int64_t         x_pitch;
    int64_t         M, n_embd, n_vocab;
    int             bits;
    int             pad;               /* explicit, zeroed: replay compares the launch's argument bytes */
} EmbedTqCall;

const RadOpSchema* embed_tq_schema(void);
int embed_tq_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out);
int embed_tq_layout(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                    const RadTensor* planes, int n_planes, RadLayout* out);
int embed_tq_args(const RadArgs* a, EmbedTqCall* out);
int embed_tq_host_launch(const RadArgs* a, RadStream s);
int embed_tq_device_launch(const RadArgs* a, RadStream s);

#ifdef __cplusplus
}
#endif

#endif /* EMBED_TQ_H */
