/* lq_bf16x12.cpp -- `bf16x12`: a bf16 weight in 12 bits and a quarter a value, losslessly.
 *
 * A trained weight's exponents sit in a narrow window. Along a row, 32 values at a time, each
 * value's exponent field is kept as its offset below the block's largest (four bits) and its sign
 * and mantissa as they are (eight): 12 bits a value and 8 a block, against bf16's 16. Every bf16
 * decodes to itself, bit for bit, so a kernel that rebuilds the bf16 and runs a bf16 kernel's
 * arithmetic gives that kernel's bytes.
 *
 * A block whose exponent fields span more than 15, or whose largest is 0xFF (an infinity or a
 * NaN), is raw: its 32 bf16 go whole to an escape table. So the format is lossless for any bf16.
 * Zeros and subnormals (exponent field 0) pack like any value in a block whose largest field is
 * 15 or less, and make any other block raw. Qwen3.8-27B's trunk has no zeros, subnormals or
 * infinities, and 0.133% of its blocks are raw; with the table at its default size the trunk is
 * 12.31 bits a value.
 *
 * ===================================================================== The planes
 *
 * A scheme of its own, "bf16x12": the core does not decode it, this file's hook does
 * (abi/rad_quant.h). In natural k order; a kernel's fragment order is its layout, applied at load.
 *
 *   codes  u8   [1x1]       sign << 7 | mantissa; 0 in a raw block
 *   d      u4   [1x1]       E minus the value's exponent field. A raw block's 16 bytes hold its
 *                           index into esc instead, a u32 in bytes 0-3, then zeros
 *   e      u8   [1x32]      E, the block's largest exponent field; 0xFF marks a raw block
 *   esc    bf16 {cap x 32}  the raw blocks, 32 bf16 each in k order, numbered in row-major order
 *
 * A packed value is  sign << 15 | (E - d) << 7 | mantissa.
 *
 * The table's size cannot depend on the data: rad-convert fixes every encoding before it reads a
 * weight, and refuses to write another. So cap = max(64, ceil(blocks * esc)), at most the block
 * count, from the shape and the option `esc`, and a weight with more raw blocks than that is
 * refused, by name. Every slice of a weight carries its whole table, and an index counts from the
 * weight's first row, so a rank's share decodes alone.
 */
#include "lq_quant.h"
#include "lq_common.h"
#include "rad_plugin.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace lq {

namespace {

const char kScheme[] = "bf16x12";
constexpr int64_t kB = 32;               /* values a block */
constexpr int64_t kMinCap = 64;          /* table rows at least */
/* At least twice the share of raw blocks in any of Qwen3.8-27B's trunk weights (0.184%, its
 * blk.59.attn_k), rounded up. */
constexpr double kEscDefault = 0.004;

enum { kCodes, kD, kE, kEsc };           /* the planes, in order */

const RadQuantOption kOptions[] = {
    { "esc", RAD_P_F64, "0.004",
      "the escape table's size, a fraction of the weight's 32-value blocks: max(64, ceil(blocks "
      "x esc)) raw blocks, at most every block. A weight with more raw blocks is refused, by "
      "name" },
};

RadEncoding encoding_of(int64_t cap) {
    RadEncoding e;
    rad_enc_clear(&e);
    rad_enc_copy_str(e.scheme, kScheme);
    rad_enc_add_plane(&e, "codes", RAD_U8, 1, 1);
    rad_enc_add_plane(&e, "d", RAD_U4, 1, 1);
    rad_enc_add_plane(&e, "e", RAD_U8, 1, kB);
    rad_enc_add_table(&e, "esc", RAD_BF16, cap, kB);
    return e;
}

double esc_of(const RadParam* o, int n_o) { return rad_param_getf(o, n_o, "esc", kEscDefault); }

/* The table's rows for a weight of `blocks` blocks. */
int64_t capacity(int64_t blocks, double esc) {
    const int64_t want = std::max<int64_t>(kMinCap, (int64_t)std::ceil((double)blocks * esc));
    return std::min(want, blocks);
}

/* A rank-1 weight, or one whose rows are not whole blocks, is declined and kept as it is. */
int x12_encoding(const RadParam* o, int n_o, const RadQuantWeight* w, RadEncoding* out) {
    if (declines(w)) return RAD_E_UNSUPPORTED;
    if (w->rows <= 0 || w->cols <= 0) return RAD_E_SHAPE;
    if (w->cols % kB) return RAD_E_UNSUPPORTED;
    const double esc = esc_of(o, n_o);
    if (!(esc >= 0.0 && esc <= 1.0))
        return refuse(w, "esc= is a fraction of the weight's blocks, from 0 to 1");
    const int64_t cap = capacity(w->rows * (w->cols / kB), esc);
    if (cap > (int64_t)UINT32_MAX)
        return refuse(w, "the escape table would hold more blocks than a u32 index numbers");
    *out = encoding_of(cap);
    return RAD_OK;
}

/* The whole weight in one call: the table and its numbering span the rows. */
int64_t x12_row_block(const RadParam*, int, const RadQuantWeight*) { return 0; }

int x12_quantize(const RadParam* o, int n_o, const RadQuantWeight* w, float* src, int64_t row0,
                 int64_t rows, void* const* planes) {
    RadEncoding enc;
    const int st = x12_encoding(o, n_o, w, &enc);
    if (st != RAD_OK) return st;
    if (row0 != 0 || rows != w->rows) return RAD_E_SHAPE;
    const int64_t N = w->rows, K = w->cols, nb = K / kB, cap = enc.plane[kEsc].extent[0];
    uint8_t* const codes = (uint8_t*)planes[kCodes];
    uint8_t* const dp = (uint8_t*)planes[kD];
    uint8_t* const ep = (uint8_t*)planes[kE];
    uint8_t* const esc = (uint8_t*)planes[kEsc];
    std::memset(esc, 0, (size_t)(cap * kB * 2));

    /* Each row's blocks, packed or marked raw and counted, and the row's first value that is not
     * a bf16, to name it. Rows are independent. */
    std::vector<int64_t> raw((size_t)N, 0), bad((size_t)N, -1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t b = 0; b < nb; ++b) {
            const float* x = src + r * K + b * kB;
            uint16_t h[kB];
            unsigned hi = 0, lo = 0xFFu;
            for (int i = 0; i < kB; ++i) {
                uint32_t u;
                std::memcpy(&u, x + i, 4);
                if ((u & 0xFFFFu) && bad[(size_t)r] < 0) bad[(size_t)r] = b * kB + i;
                h[i] = (uint16_t)(u >> 16);
                const unsigned ex = (h[i] >> 7) & 0xFFu;
                hi = std::max(hi, ex);
                lo = std::min(lo, ex);
            }
            uint8_t* c = codes + r * K + b * kB;
            uint8_t* d = dp + r * (K / 2) + b * (kB / 2);
            if (hi == 0xFFu || hi - lo > 15) {
                ep[r * nb + b] = 0xFF;
                std::memset(c, 0, (size_t)kB);
                std::memset(d, 0, (size_t)(kB / 2));    /* the index, once every block is counted */
                ++raw[(size_t)r];
                continue;
            }
            ep[r * nb + b] = (uint8_t)hi;
            for (int i = 0; i < kB; ++i) c[i] = (uint8_t)((h[i] >> 15) << 7 | (h[i] & 0x7Fu));
            for (int i = 0; i < kB; i += 2)
                d[i / 2] = (uint8_t)((hi - ((h[i] >> 7) & 0xFFu)) |
                                     (hi - ((h[i + 1] >> 7) & 0xFFu)) << 4);
        }
    }
    for (int64_t r = 0; r < N; ++r)
        if (bad[(size_t)r] >= 0) {
            char why[192];
            std::snprintf(why, sizeof why, "the value at row %lld, column %lld is not a bf16 (its "
                          "f32 has low bits set), and bf16x12 is lossless only for bf16",
                          (long long)r, (long long)bad[(size_t)r]);
            return refuse(w, why);
        }

    /* Raw blocks numbered in row-major order, so a weight converts to the same bytes on any
     * thread count. */
    std::vector<int64_t> first((size_t)N);
    int64_t total = 0;
    for (int64_t r = 0; r < N; ++r) {
        first[(size_t)r] = total;
        total += raw[(size_t)r];
    }
    if (total > cap) {
        char why[256];
        std::snprintf(why, sizeof why, "%lld of its %lld blocks are raw (exponents spanning more "
                      "than 15, or an inf or NaN), and esc=%g gives its escape table %lld; a rule "
                      "with a larger esc= for this weight converts it", (long long)total,
                      (long long)(N * nb), esc_of(o, n_o), (long long)cap);
        (void)refuse(w, why);
        return RAD_E_FULL;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t r = 0; r < N; ++r) {
        if (!raw[(size_t)r]) continue;
        uint32_t idx = (uint32_t)first[(size_t)r];
        for (int64_t b = 0; b < nb; ++b) {
            if (ep[r * nb + b] != 0xFFu) continue;
            std::memcpy(dp + r * (K / 2) + b * (kB / 2), &idx, 4);
            uint8_t* t = esc + (size_t)idx * kB * 2;
            for (int i = 0; i < kB; ++i) {
                uint32_t u;
                std::memcpy(&u, src + r * K + b * kB + i, 4);
                const uint16_t h = (uint16_t)(u >> 16);
                std::memcpy(t + 2 * i, &h, 2);
            }
            ++idx;
        }
    }
    return RAD_OK;
}

/* Exactly the encodings this file writes. A raw block's index past the table, or an offset above
 * its block's E, is RAD_E_FORMAT, not a read past the table or an exponent that wrapped. */
int x12_decode(const RadEncoding* e, const RadQuantWeight* w, const void* const* planes,
               int64_t row0, int64_t rows, float* out) {
    if (!rad_enc_is(e, kScheme)) return RAD_E_UNSUPPORTED;
    const int64_t cap = e->plane[kEsc].extent[0];
    const RadEncoding want = encoding_of(cap);
    if (!rad_enc_equal(e, &want)) return RAD_E_FORMAT;
    if (!w || w->cols <= 0 || w->cols % kB || row0 < 0 || rows < 0 || row0 + rows > w->rows)
        return RAD_E_SHAPE;
    const int64_t K = w->cols, nb = K / kB;
    const uint8_t* const codes = (const uint8_t*)planes[kCodes];
    const uint8_t* const dp = (const uint8_t*)planes[kD];
    const uint8_t* const ep = (const uint8_t*)planes[kE];
    const uint8_t* const esc = (const uint8_t*)planes[kEsc];
    int bad = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(| : bad)
#endif
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t r = row0 + i;
        for (int64_t b = 0; b < nb; ++b) {
            const unsigned hi = ep[r * nb + b];
            const uint8_t* d = dp + r * (K / 2) + b * (kB / 2);
            float* y = out + i * K + b * kB;
            if (hi == 0xFFu) {
                uint32_t idx;
                std::memcpy(&idx, d, 4);
                if ((int64_t)idx >= cap) {
                    bad |= 1;
                    continue;
                }
                const uint8_t* t = esc + (size_t)idx * kB * 2;
                for (int k = 0; k < kB; ++k) {
                    uint16_t h;
                    std::memcpy(&h, t + 2 * k, 2);
                    y[k] = bf16_to_f32(h);
                }
                continue;
            }
            const uint8_t* c = codes + r * K + b * kB;
            for (int k = 0; k < kB; ++k) {
                const unsigned off = (d[k / 2] >> (4 * (k & 1))) & 0xFu;
                bad |= off > hi;
                y[k] = bf16_to_f32((uint16_t)((c[k] >> 7) << 15 | ((hi - off) & 0xFFu) << 7 |
                                              (c[k] & 0x7Fu)));
            }
        }
    }
    return bad ? RAD_E_FORMAT : RAD_OK;
}

}  /* namespace */

const RadQuantizerInfo& bf16x12_info() {
    static const RadQuantizerInfo q = {
        "bf16x12",
        "a bf16 weight packed losslessly in 12 bits and a quarter a value: the sign and mantissa "
        "as they are, the exponent as a 4-bit offset below its 32-value block's largest, and a "
        "block whose exponents span more than 15 whole in an escape table. A scheme of its own, "
        "decoded by this quantiser's hook",
        kOptions, (int)(sizeof kOptions / sizeof kOptions[0]),
        x12_encoding, x12_row_block, x12_quantize, x12_decode,
    };
    return q;
}

}  /* namespace lq */
