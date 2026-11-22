/* museglimmer_ct.cpp -- Meta's Muse Glimmer 30B under compressed-tensors quantisation.
 *
 * The same graph arch/museglimmer.cpp declares at the unquantised release, with every trunk
 * linear encoding-aware (arch/gemma3's G3Linear shape, gemma4_bf16's G4Linear grids): the weight's
 * container encoding picks the op. Plain bf16 rides `gemm_nt`; the symmetric int4/int8 grid the
 * museglimmer-30b-w4a8 recipe writes (`codes=i4/i8 group=128 scale=f16`) rides quant_act_i8 +
 * gemm_nt_q at w4a8 / w8a8; asymmetric int4/int5 affine per
 * [1 x 128] (`codes=u4/u5 zero=u8 group=128 scale=f16`) rides quant_act_i8g + gemm_nt_q at
 * w4a8i8_asym / w5a8i8, with the a16 low-latency decode form w4a16_asym / w5a16 under 64 rows;
 * int8 a 128 (i8g) rides gemm_nt_q at i8a8. Anything else is refused by name. A vision linear is
 * BF16, or int8 with a bf16 scale per (row, 128 of K) read by gemm_i8a8_row; block FP8 is retired.
 * The stem, position table, norms and biases stay BF16 and are refused by name if quantised.
 *
 * Rotated i8a8h trunks are whole-trunk forms, detected from layer 0's q projection before any
 * activation is declared. IQ8R uses a 256-column scale group; computed-trellis IQR (including
 * IQ4R-V4) uses 128. Both reuse LinearFP8's encoding-aware IQR views and GEMMs. The two centered
 * pre-norms produce rotated int8 pairs, the gated product produces the down-projection pair, and
 * o_proj quantises its gated attention input. q, gate, k and v share one pair. Every trunk linear
 * must use the same rotated family; mixed encodings fail at declaration with the offending name.
 *
 * The lm_head is declared at whatever the model holds (weight_codes_dtype): bf16, or a recipe-made
 * int8 head (`codes=i8 group=128 scale=bf16`), which `logits_gemm` then reads with the int8 head
 * kernel. The drafter reads the same declared head.
 */
#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>
#include <arch/rad_block_museglimmer_vision.h>
#include <arch/rad_block_muse_dflash.h>
#include <arch/rad_calib_full.h>
#include <arch/rad_museglimmer_chat.h>
#include "../museglimmer/dflash2.h"

#include <cmath>
#include <cstdlib>
#include <vector>

namespace museglimmer_ct {

using namespace rad::arch;

/* The family's shape, as the bf16 plugin states it: 52 layers, hidden 6656, ffn 19968, vocab
 * 202048; sliding_window 2048 with the sliding layers owning the rotary; q and the output gate
 * separate checkpoint tensors; four centered norms a layer; a weightless q/k RMS norm with
 * qk_scale_factor on the query; a weightless embedding norm; a plain-gain final norm carrying
 * output_multiplier before the tanh softcap.
 */

/* ---- the checkpoint's config. A multimodal model keeps the trunk under `text_config.`; accept
 * the bare spelling too, so a text-only GGUF container of the same family reads the same way. */
namespace {

}  // namespace

/* ---- one trunk linear, encoding-aware. The gemma3 G3Linear shape with gemma4_bf16's grids: the
 * weight's container encoding picks the op.
 *
 *   plain bf16          -> decl_w + gemm_nt
 *   symmetric affine    -> decl_view codes i4/i8 + scale f16 per [1 x 128] (the
 *                          museglimmer-30b-w4a8 recipe's grid), quant_act_i8 over the bf16
 *                          activation, gemm_nt_q at w4a8 / w8a8
 *   asymmetric affine   -> decl_view codes u4/u5 + scale,zero f16 per [1 x 128], quant_act_i8g
 *                          over the bf16 activation, gemm_nt_q at w4a8i8_asym / w5a8i8; up to
 *                          kA4Split rows the input is cast to f16 for w4a16_asym / w5a16 (the
 *                          decode band: w4a8i8_asym has only a prefill kernel, which at one row
 *                          does 16 times the matrix work and reads its weight a quarter as fast)
 *   int8 a 128 (i8g)    -> decl_view codes i8 + scale bf16 per [1 x 128], quant_act_i8g,
 *                          gemm_nt_q at i8a8
 *   anything else       -> refused by name (RAD_E_DTYPE, the formatted encoding)
 *
 * The name map is made before the weight is queried (Builder::finish() needs the map entry before
 * the declaration). A builder with no model behind it (a test, a synthetic tool) answers
 * RAD_E_NOTFOUND from rad_weight_encoding, and the weight is read as plain bf16.
 * q, gate, k and v quantise the same norm output on the same per-row grid, so with `in_from` set
 * the later linear reads that linear's int8 pair and declares no quantiser of its own: one
 * dispatch instead of four. An a4 linear writes no pair below its decode band, so an a4 q
 * provides none and each of the four keeps its own quantiser. */
struct MlLinear {
    rad_weight w = 0, ws = 0;   /* the codes + scale (quantised), or the single bf16 weight */
    rad_op     op = 0, op_q = 0;   /* the GEMM and, for a quantised form, its activation quantiser */
    rad_buf    q8 = 0, s8 = 0;     /* the int8 pair (i8 codes + a scale row) the quantiser writes */
    int64_t    N = 0, K = 0;
    /* true: symmetric i4/i8 codes with an f16 scale per [1 x 128] (the museglimmer-30b-w4a8
     * recipe's grid), gemm_nt_q at w4a8 / w8a8 over a per-row int8 activation. */
    bool       quant = false;
    bool       i8g = false;     /* true: int8 codes, bf16 scale per [1 x 128], gemm_nt_q at i8a8 */
    /* true: asymmetric int4/int5 (libquant `codes=u4/u5 zero=u8 group=128 scale=f16`), gemm_nt_q
     * at w4a8i8_asym / w5a8i8 over the same per-(row, 128) int8 activation as i8g. */
    bool       a4 = false;
    bool       w5 = false;      /* true with a4: the same grid at five bits */
    bool       grouped() const { return i8g || a4; }
    /* Block-scaled fp8 (fp8_e4m3 codes, a bf16 scale per [128 x 128]) or OCP-MXFP4 (e2m1 codes, an
     * e8m0 exponent per [1 x 32]), read by gemm_nt_q at fp8a8 / mxfp4a8 over an E4M3 activation
     * pair. These are what NVIDIA's ModelOpt NVFP4 release leaves behind once the recipe imports
     * its FP8 projections and makes its NVFP4 MLPs MXFP4. */
    bool       fp8b = false;
    bool       mx = false;
    bool       fp8_f32 = false; /* block fp8 with an f32 rather than a bf16 weight scale */
    rad_weight wref = 0;        /* mx: the e8m0 plane again as the per-row reference operand */
    rad_buf    fq = 0, fs = 0;  /* the E4M3 activation pair the quantiser writes */
    /* the pair's scales a row: one, or one per 128 columns */
    int64_t    scale_w() const { return grouped() ? K / RAD_FP8_BLOCK : 1; }
    rad_buf    in = 0;            /* the bf16 input row `a` given at declare */
    static constexpr int64_t kA4Split = 64;
    int64_t    a4_split = 0;      /* rows up to which op_dec serves; 0: no decode band */
    rad_buf    a16 = 0;
    rad_op     op_a16 = 0, op_dec = 0;
    bool       shared_in = false;

    int  declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                 std::initializer_list<const char*> src,
                 int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                 rad_buf a, rad_buf y,
                 std::initializer_list<int64_t> row_parts = {},
                 const MlLinear* in_from = nullptr);
    void step(RadCtx* c, rad_buf a, rad_buf y, int64_t T) const;
    /* the GEMM alone, for a caller that wrote the int8 pair itself (a fused norm + quantiser) */
    void gemm(RadCtx* c, rad_buf y, int64_t T) const;
};

inline int MlLinear::declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                            std::initializer_list<const char*> src,
                            int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                            rad_buf a, rad_buf y,
                            std::initializer_list<int64_t> row_parts,
                            const MlLinear* in_from) {
    N = n_out; K = n_in; in = a;
    if (N <= 0 || K <= 0) return RAD_E_INVAL;
    const char* wn = nm.f("%s.weight", base);
    if (src.size() == 1)     RAD_ARCH_TRY(map_copy(b, wn, *src.begin()));
    else if (src.size() > 1) RAD_ARCH_TRY(map_concat(b, wn, src, 0));
    RadEncoding enc{};
    const bool known = weight_enc(b, wn, &enc);
    const bool plain = !known || (rad_enc_is(&enc, "plain") && enc.n_planes == 1 &&
                                  enc.plane[0].dtype == RAD_BF16);
    int64_t parts[RAD_MAX_ROW_PARTS] = {0};
    int np = 0;
    for (int64_t p : row_parts) if (np < RAD_MAX_ROW_PARTS) parts[np++] = p;

    if (plain) {
        w = decl_w(b, wn, RAD_BF16, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp, 0, row_parts);
        if (!w) return RAD_E_INVAL;
        op = rw(b, RAD_OP(b, "gemm_nt",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                               RAD_STR("dtype", g.dtype)),
                    RAD_WEIGHTS(w)),
                { a }, { y });
        return op ? RAD_OK : RAD_E_INVAL;
    }

    const RadEncPlane* sc = rad_enc_plane(&enc, "scale");
    const RadEncPlane* zp = rad_enc_plane(&enc, "zero");
    const uint32_t cdt = known && enc.n_planes > 0 ? enc.plane[0].dtype : 0;
    quant = known && rad_enc_is(&enc, "affine") && enc.n_planes == 2 && !enc.transform[0] &&
            (cdt == RAD_I4 || cdt == RAD_I8) && enc.plane[0].block[1] == 1 &&
            sc && sc->dtype == RAD_F16 && sc->block[0] == 1 && sc->block[1] == 128;
    i8g = known && linear_is_w8(enc);
    a4 = known && rad_enc_is(&enc, "affine") && enc.n_planes == 3 && !enc.transform[0] &&
         (cdt == RAD_U4 || cdt == RAD_U5) && enc.plane[0].block[1] == 1 &&
         sc && sc->dtype == RAD_F16 && sc->block[0] == 1 && sc->block[1] == RAD_FP8_BLOCK && zp &&
         zp->dtype == RAD_U8 && zp->block[0] == 1 && zp->block[1] == RAD_FP8_BLOCK;
    w5 = a4 && cdt == RAD_U5;
    fp8b = known && rad_enc_is(&enc, "affine") && enc.n_planes == 2 && !enc.transform[0] &&
           cdt == RAD_F8E4M3 && enc.plane[0].block[1] == 1 &&
           sc && (sc->dtype == RAD_BF16 || sc->dtype == RAD_F32) &&
           sc->block[0] == RAD_FP8_BLOCK && sc->block[1] == RAD_FP8_BLOCK;
    fp8_f32 = fp8b && sc && sc->dtype == RAD_F32;
    mx = known && linear_is_mxfp4(enc);
    /* Block fp8 and MXFP4 are forms this build does not serve, in any plugin. */
    if (known)
        if (const char* gone = linear_retired(enc)) return linear_refuse_retired(base, gone);

    if (fp8b || mx) {
        if (N % 16 || K % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; a %s weight takes rows in 16s "
                            "and K in whole %d-column activation scale groups\n", base,
                    (long long)N, (long long)K, mx ? "MXFP4" : "block fp8", RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        const char* sn = nm.f("%s.scale", base);
        if (mx) {
            w    = decl_view(b, wn, nullptr, "codes", RAD_FP4E2M1, { N, K }, RAD_ACCESS_PER_TOKEN,
                             shard, grp, parts, np);
            ws   = decl_view(b, sn, wn, "scale", RAD_E8M0, { N, K / RAD_MX_GROUP },
                             RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
            wref = decl_view(b, nm.f("%s.ref", base), wn, "scale", RAD_E8M0,
                             { N, K / RAD_MX_GROUP }, RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
            if (!w || !ws || !wref) return RAD_E_INVAL;
        } else {
            w  = decl_view(b, wn, nullptr, "codes", RAD_F8E4M3, { N, K }, RAD_ACCESS_PER_TOKEN,
                           shard, grp, parts, np);
            ws = decl_view(b, sn, wn, "scale", fp8_f32 ? RAD_F32 : RAD_BF16,
                           { fp8_blocks(N), fp8_blocks(K) },
                           RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
            if (!w || !ws) return RAD_E_INVAL;
        }
        if (in_from && in_from->fp8b == fp8b && in_from->mx == mx && in_from->fq &&
            in_from->K == K) {
            fq = in_from->fq;
            fs = in_from->fs;
            shared_in = true;
        } else {
            fq = decl_b(b, nm.f("%s.q8", base), RAD_F8E4M3, { g.max_tok, K });
            fs = decl_b(b, nm.f("%s.s8", base), RAD_F32, { g.max_tok, K / RAD_FP8_BLOCK });
            if (!fq || !fs) return RAD_E_INVAL;
            /* quant_act_fp8: the E4M3 codes and an f32 scale per (row, 128), which the fp8a8 and
             * mxfp4a8 GEMMs read. */
            op_q = rw(b, RAD_OP(b, "quant_act_fp8",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", K),
                                   RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
                    { a }, { fq, fs });
            if (!op_q) return RAD_E_INVAL;
        }
        op = mx ? rw(b, RAD_OP(b, "gemm_nt_q",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                                   RAD_INT("group", RAD_MX_GROUP), RAD_STR("dtype", "mxfp4a8")),
                        RAD_WEIGHTS(w, ws, wref)),
                    { fq, fs }, { y })
                : rw(b, RAD_OP(b, "gemm_nt_q",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                                   RAD_INT("group", RAD_FP8_BLOCK),
                                   RAD_STR("dtype", fp8_f32 ? "fp8a8f32" : "fp8a8")),
                        RAD_WEIGHTS(w, ws)),
                    { fq, fs }, { y });
        return op ? RAD_OK : RAD_E_INVAL;
    }

    if (quant || i8g || a4) {
        if (N % 16 || K % (quant ? 128 : RAD_FP8_BLOCK)) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; it takes rows in 16s and K in "
                            "whole %d-column activation scale groups\n", base, (long long)N,
                    (long long)K, quant ? 128 : RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        if (quant) {
            w  = decl_view(b, wn, nullptr, "codes", cdt, { N, K }, RAD_ACCESS_PER_TOKEN, shard,
                           grp, parts, np);
            ws = decl_view(b, nm.f("%s.scale", base), wn, "scale", RAD_F16, { N, K / 128 },
                           RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        } else if (a4) {
            w  = decl_view(b, wn, nullptr, "codes", w5 ? RAD_U5 : RAD_U4, { N, K },
                           RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
            ws = decl_view(b, nm.f("%s.scale", base), wn, "scale,zero", RAD_F16, { N, K },
                           RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        } else {
            w  = decl_view(b, wn, nullptr, "codes", RAD_I8, { N, K }, RAD_ACCESS_PER_TOKEN, shard,
                           grp, parts, np);
            ws = decl_view(b, nm.f("%s.scale", base), wn, "scale", RAD_BF16, { N, K / RAD_FP8_BLOCK },
                           RAD_ACCESS_PER_TOKEN, shard, grp, parts, np);
        }
        if (!w || !ws) return RAD_E_INVAL;
        if (in_from && ((quant && in_from->quant) ||
                        (!quant && in_from->grouped() && !in_from->a4 && !a4 &&
                         in_from->i8g == i8g)) &&
            in_from->q8 && in_from->K == K) {
            q8 = in_from->q8;
            s8 = in_from->s8;
            shared_in = true;
        } else {
            q8 = decl_b(b, nm.f("%s.q8", base), RAD_I8, { g.max_tok, K });
            s8 = decl_b(b, nm.f("%s.s8", base), RAD_F32,
                        { g.max_tok, quant ? 1 : K / RAD_FP8_BLOCK });
            if (!q8 || !s8) return RAD_E_INVAL;
            /* quant_act_i8 for the symmetric grid: row-major codes with ONE f32 scale a row,
             * which the w4a8 / w8a8 GEMMs read (docs/OPS.md). quant_act_i8g for the grouped
             * grids (a scale per (row, 128)); paired the other way round they produce NaN
             * logits. `group` only sizes the optional code sums, which the symmetric grid does
             * not take. */
            op_q = rw(b, RAD_OP(b, quant ? "quant_act_i8" : "quant_act_i8g",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", K),
                                   RAD_INT("group", quant ? 128 : RAD_FP8_BLOCK),
                                   RAD_STR("dtype", "bf16")),
                        RAD_NOWEIGHTS),
                    { a }, { q8, s8 });
            if (!op_q) return RAD_E_INVAL;
        }
        op = rw(b, RAD_OP(b, "gemm_nt_q",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", N), RAD_INT("K", K),
                               RAD_INT("group", quant ? 128 : RAD_FP8_BLOCK),
                               RAD_STR("dtype", quant ? (cdt == RAD_I8 ? "w8a8" : "w4a8")
                                       : w5 ? "w5a8i8" : a4 ? "w4a8i8_asym" : "i8a8")),
                    RAD_WEIGHTS(w, ws)),
                { q8, s8 }, { y });
        if (!op) return RAD_E_INVAL;
        if (a4) {
            a4_split = std::min<int64_t>(kA4Split, g.max_tok);
            a16 = decl_b(b, nm.f("%s.a16", base), RAD_F16, { a4_split, K });
            if (!a16) return RAD_E_INVAL;
            op_a16 = rw(b, RAD_OP(b, "cast",
                           RAD_PARAMS(RAD_RANGE("M", 1, a4_split), RAD_INT("n", K),
                                      RAD_STR("from", g.dtype), RAD_STR("to", "f16")),
                           RAD_NOWEIGHTS),
                        { a }, { a16 });
            op_dec = rw(b, RAD_OP(b, "gemm_nt_q",
                            RAD_PARAMS(RAD_RANGE("M", 1, a4_split), RAD_INT("N", N), RAD_INT("K", K),
                                       RAD_INT("group", RAD_FP8_BLOCK),
                                       RAD_STR("dtype", w5 ? "w5a16" : "w4a16_asym")),
                            RAD_WEIGHTS(w, ws)),
                        { a16 }, { y });
            if (!op_a16 || !op_dec) return RAD_E_INVAL;
        }
        return RAD_OK;
    }

    char en[256];
    rad_enc_format(&enc, en, sizeof en);
    /* Rotated i8a8h encodings are served only as a whole trunk, decided by layer 0's q. */
    if (known && linear_is_w8h(enc)) {
        fprintf(stderr, "radiance: '%s' is a rotated i8a8h weight (%s), but layer 0's attn_q "
                        "does not select a matching whole-trunk path\n", wn, en);
        return RAD_E_DTYPE;
    }
    fprintf(stderr, "radiance: '%s' is %s, and this model's linears are plain bf16, symmetric "
                    "int4 or int8 a 128 (w4a8 / w8a8), asymmetric int4 or int5 a 128 "
                    "(w4a8i8_asym / w5a8i8), int8 a 128 (i8a8), block fp8 e4m3 with a bf16 "
                    "scale a 128x128 (fp8a8) or MXFP4 e2m1 with an e8m0 exponent a 32 "
                    "(mxfp4a8) -- none matches\n", wn, en);
    return RAD_E_DTYPE;
}

inline void MlLinear::step(RadCtx* c, rad_buf a, rad_buf y, int64_t T) const {
    if (quant) {
        /* quant_act_i8 takes x, q, scale, asum; `asum` (the per-group row sums asymmetric grids
         * subtract) is absent on this symmetric grid and passed as a null tensor, since operands
         * are positional. */
        if (!shared_in)
            RAD_ISSUE_N(c, op_q, T, brows(a, T), brows(q8, T), brows(s8, T), RAD_NONE);
        gemm(c, y, T);
    } else if (fp8b || mx) {
        /* quant_act_fp8 takes x, q, scale: an E4M3 pair at group 128. The mx form reads the e8m0
         * plane again as its per-row reference, so the GEMM carries a third weight operand. */
        if (!shared_in)
            RAD_ISSUE_N(c, op_q, T, brows(a, T), brows(fq, T), brows(fs, T));
        gemm(c, y, T);
    } else if (a4 && T <= a4_split) {
        /* the decode band: f16 copy of the input, the skinny kernel; operands as LinearFP8's w4 */
        RAD_ISSUE_N(c, op_a16, T, brows(a, T), brows(a16, T));
        RAD_ISSUE_N(c, op_dec, T, brows(a16, T), RAD_NONE, RAD_W(w), RAD_W(ws), brows(y, T),
                    RAD_NONE, RAD_NONE);
    } else if (grouped()) {
        /* quant_act_i8g takes x, q, scale: no optional `asum`. */
        if (!shared_in)
            RAD_ISSUE_N(c, op_q, T, brows(a, T), brows(q8, T), brows(s8, T));
        gemm(c, y, T);
    } else
        RAD_ISSUE(c, op, brows(a, T), RAD_W(w), brows(y, T));
}

inline void MlLinear::gemm(RadCtx* c, rad_buf y, int64_t T) const {
    if (fp8b || mx) {
        /* the caller wrote the E4M3 pair itself (a fused norm + quantiser): issue the GEMM only */
        if (mx)
            RAD_ISSUE_N(c, op, T, brows(fq, T), brows(fs, T), RAD_W(w), RAD_W(ws), brows(y, T),
                        RAD_NONE, RAD_W(wref));
        else
            RAD_ISSUE_N(c, op, T, brows(fq, T), brows(fs, T), RAD_W(w), RAD_W(ws), brows(y, T),
                        RAD_NONE, RAD_NONE);
        return;
    }
    if (a4 && T <= a4_split) {
        RAD_ISSUE_N(c, op_a16, T, brows(in, T), brows(a16, T));
        RAD_ISSUE_N(c, op_dec, T, brows(a16, T), RAD_NONE, RAD_W(w), RAD_W(ws), brows(y, T),
                    RAD_NONE, RAD_NONE);
        return;
    }
    RAD_ISSUE_N(c, op, T, brows(q8, T), brows(s8, T), RAD_W(w), RAD_W(ws),
                brows(y, T), RAD_NONE, RAD_NONE);
}

/* One trunk layer. The rope ops exist only where layer_rope_theta says so; the branch is fixed
 * per layer by the model. */
struct Layer {
    int full = 0;      /* full attention (no window, no rope in this family) */
    int rope = 0;      /* carries a rotary (the sliding layers) */

    rad_weight w_norm = 0, w_post = 0;
    rad_weight w_fnorm = 0, w_fpost = 0;
    MlLinear lq, lgate, lk, lv, lo, lgu, ldown;

    /* The seven rotated linears in an IQ8R or IQR trunk, at gemm_nt_q i8a8h. The pre-norms
     * write their int8 pair as rmsnorm_had_quant_i8g when available; the gated product writes the
     * down-projection pair as gated_had_quant_i8g. Otherwise the first consumer quantises with
     * had_quant_act_i8g. `rot_group` is 256 for IQ8R, 128 for computed-trellis IQR. */
    LinearFP8    rq, rgate, rk, rv, ro, rgu, rdown;
    NormQuantFP8 nq_attn, nq_ffn;
    GateQuantFP8 gq_ffn;

    rad_op op_norm = 0, op_qn = 0, op_kn = 0, op_rq = 0, op_rk = 0;
    rad_op op_kv = 0, op_attn = 0, op_sig = 0, op_mul = 0, op_ar = 0;
    rad_op op_post = 0, op_add = 0;
    rad_op op_fnorm = 0, op_act = 0, op_far = 0, op_fpost = 0, op_fadd = 0;

    /* The layer's calibration inputs (rad_calib_full.h): -1 where the run does not record it. */
    int cs_in = -1, cs_o = -1, cs_mlp = -1, cs_down = -1;
};

struct Model {
    Names nm{""};
    Geom  g{};        /* the trunk geometry, the rank's shard of it */
    Geom  g_rope{};   /* the same, with the sliding layers' rotary base (decl_rope reads theta) */
    float qk_mult  = 0;   /* qk_scale_factor */
    float post_eps = 0;   /* post_norm_eps */
    float out_mult = 0;   /* output_multiplier */
    float softcap  = 0;   /* final_logit_softcapping; 0: none */
    int64_t window = 0;   /* sliding_window */

    rad_kvgroup kv_win = 0, kv_full = 0;

    /* the residual stream and the two vocabulary edges */
    rad_buf b_x = 0, b_h = 0, b_hout = 0, b_logits = 0;
    rad_buf b_q = 0, b_gate = 0, b_k = 0, b_v = 0, b_attn = 0;
    rad_buf b_gu = 0, b_ffn = 0;

    /* A whole-trunk rotated-int8 mode selected from layer 0's q projection. `iq8r` keeps the
     * legacy wide-group case explicit; IQR-V4 is computed trellis/MAD at group 128. */
    bool   rotated_i8 = false, iq8r = false, iqr_v4 = false;
    int64_t rot_group = RAD_FP8_BLOCK;
    ActFP8 a_h{}, a_attn{}, a_ffn{};

    rad_weight w_tok = 0, w_out_norm = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_embed_norm = 0;
    rad_op op_final_norm = 0, op_gather = 0, op_logits = 0, op_cap = 0;

    /* ---- media (arch/common/rad_block_museglimmer_vision.h): the vision tower and the scatter of
     * its rows, declared only when the container carries a tower and the deployment asked for
     * media. The tower is declared last, so its activations share the arena with every buffer
     * above. */
    bool      media = false;
    rad_op    op_media_rows = 0;
    MuseTower vit{};

    /* ---- Meta's DFlash or incoai's DFlash2, selected by draft.architectures. */
    bool        have_df = false, have_df2 = false;
    MuseDflash  df{};
    MuseDflash::Wire dfw{};
    Dflash2     df2{};
    Dflash2::Wire df2w{};
    MuseTaps    taps{};
    rad_kvgroup kv_draft = 0;

    std::vector<Layer> layers;

    /* --calib-dir: the whole Hessian of every linear's input over a --calib-corpus run. */
    CalibFull calib;
    int       cs_head = -1;
};

/* Declare runs once per rank of an instance, on the rank's own thread (spec §1, §9); every
 * instance's state is its slot's (rad_build_slot / rad_slot). */
static Model g_model[MAX_SLOTS];

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;

    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));

    /* A routed model is not this family. */
    if (meta->n_expert > 0) {
        fprintf(stderr, "radiance: museglimmer refuses a checkpoint with %lld experts\n",
                (long long)meta->n_expert);
        return RAD_E_UNSUPPORTED;
    }

    Geom&  g  = m.g;
    Names& nm = m.nm;

    /* ---- the family's own config. Everything that changes the mathematics is required or read
     * from the family defaults the reference config class states; a wrong default yields fluent
     * wrong text, so nothing here falls back to a "reasonable" value silently. */
    m.qk_mult  = (float)meta_text_getf(meta, "qk_scale_factor", 3.87);
    m.post_eps = (float)meta_text_getf(meta, "post_norm_eps", 1e-8);
    m.out_mult = (float)meta_text_getf(meta, "output_multiplier", 0.19611613513818404);
    m.softcap  = (float)meta_text_getf(meta, "final_logit_softcapping", 20.0);
    m.window   = meta_text_geti(meta, "sliding_window", 2048);
    const double theta_rope = meta_text_getf(meta, "rope_parameters.rope_theta",
                                    meta_text_getf(meta, "rope_theta", 0.0));
    if (!(m.qk_mult > 0.0f) || !(m.out_mult > 0.0f) || !(m.post_eps > 0.0f) || m.window <= 0 ||
        !(theta_rope > 0.0)) {
        fprintf(stderr, "radiance: museglimmer: qk_scale_factor %g output_multiplier %g "
                        "post_norm_eps %g sliding_window %lld rope theta %g are all required\n",
                (double)m.qk_mult, (double)m.out_mult, (double)m.post_eps,
                (long long)m.window, theta_rope);
        return RAD_E_INVAL;
    }
    if (g.rope_mc) {
        fprintf(stderr, "radiance: museglimmer: the trunk has one-component rotary; a container "
                        "with mrope_section is not this family\n");
        return RAD_E_UNSUPPORTED;
    }

    /* The rotary width is the full head: rope_parameters has no partial_rotary_factor and the
     * reference rotates all of head_dim. */
    const int64_t n_rot = g.head_dim;

    /* ---- layer kinds. The config ships `layer_types`; where it does not (a GGUF container), the
     * interval rule stands in. The two must agree where both exist: the schedule decides which
     * layers keep a long context, and a wrong layer still generates fluent text. */
    std::vector<int> full;
    const char* kinds = meta_text_gets(meta, "layer_types", nullptr);
    if (kinds && *kinds)
        RAD_ARCH_TRY(parse_layer_kinds("museglimmer", kinds, {"full_attention"},
                                       "sliding_attention", &full));
    const int64_t interval = 4;
    if ((int64_t)full.size() < g.n_layer) {
        full.assign((size_t)g.n_layer, 0);
        for (int64_t l = 0; l < g.n_layer; ++l) full[(size_t)l] = is_full_attention(l, interval);
        if (kinds && *kinds)
            rad_note(b, "the container's layer_types named %zu layers and the model has %lld; "
                        "building the full-attention layers by the interval %lld",
                    full.size() - (size_t)g.n_layer + (size_t)g.n_layer, (long long)g.n_layer,
                    (long long)interval);
    } else {
        for (int64_t l = 0; l < g.n_layer; ++l) {
            if ((full[(size_t)l] != 0) != is_full_attention(l, interval)) {
                fprintf(stderr, "radiance: museglimmer: layer_types makes layer %lld %s and "
                                "full_attention_interval %lld makes it %s\n",
                        (long long)l, full[(size_t)l] ? "full" : "sliding", (long long)interval,
                        is_full_attention(l, interval) ? "full" : "sliding");
                return RAD_E_UNSUPPORTED;
            }
        }
    }

    /* Per-layer rotary: the shipped `layer_rope_theta`, else the family pattern (theta on the
     * sliding layers, 0 -- NoPE -- on the full ones). */
    std::vector<float> theta((size_t)g.n_layer, 0.0f);
    const char* thetas = meta_text_gets(meta, "layer_rope_theta", nullptr);
    if (thetas && *thetas) {
        size_t i = 0;
        for_each_token(thetas, [&](const char* t, size_t n) {
            if (i >= theta.size()) return false;
            char* e = nullptr;
            const double v = std::strtod(std::string(t, n).c_str(), &e);
            theta[i++] = e == t ? -1.0f : (float)v;
            return true;
        });
        if (i < theta.size()) {
            fprintf(stderr, "radiance: museglimmer: layer_rope_theta names %zu layers and the "
                            "model has %lld\n", i, (long long)g.n_layer);
            return RAD_E_INVAL;
        }
    } else {
        for (int64_t l = 0; l < g.n_layer; ++l)
            theta[(size_t)l] = full[(size_t)l] ? 0.0f : (float)theta_rope;
    }
    for (int64_t l = 0; l < g.n_layer; ++l) {
        if (!(theta[(size_t)l] >= 0.0f)) {
            fprintf(stderr, "radiance: museglimmer: layer_rope_theta[%lld] is not a number\n",
                    (long long)l);
            return RAD_E_INVAL;
        }
    }

    /* The rotary scaling is read into the rope Geom, against the sliding layers' own base: the
     * extension reaches their rope and leaves the NoPE full layers untouched (they have no rotary
     * to scale); the mechanical extension past the trained positions is the no-keys default. */
    m.g_rope          = g;
    m.g_rope.theta    = (float)theta_rope;
    RAD_ARCH_TRY(yarn_from(m.g_rope, meta, "", g.head_dim));

    const int64_t q_dim  = g.q_dim();
    const int64_t kv_dim = g.kv_dim();

    /* ---- two KV groups (spec §7.2): the sliding layers' pool recycles blocks below the window,
     * the full layers' does not. A group with no bound layers is a declare error. */
    const uint32_t kvdt = rad_kv_cache_dtype(ctx, RAD_BF16);
    RadKVGroupDecl sd{};
    sd.kind      = RAD_KV_WINDOW;
    sd.dtype     = kvdt;
    sd.window    = m.window;
    sd.n_head_kv = g.n_head_kv;
    sd.head_dim  = g.head_dim;
    m.kv_win = rad_decl_kv_group(b, nm.f("kv_sliding"), &sd);
    RadKVGroupDecl gd{};
    gd.kind      = RAD_KV_FULL;
    gd.dtype     = kvdt;
    gd.n_head_kv = g.n_head_kv;
    gd.head_dim  = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_full"), &gd);
    g.kv_dtype = rad_dtype_name(kvdt);
    if (m.kv_win == 0 || m.kv_full == 0) {
        fprintf(stderr, "radiance: museglimmer: a KV group failed to declare\n");
        return RAD_E_INVAL;
    }

    /* ---- the trunk's form, read from layer 0's q projection before any activation is declared:
     * IQ8R and computed-trellis IQR activations carry an int8 code pair. The probe maps layer 0's
     * q weight, so neither linear maps it again. */
    const char* const pre = "model.language_model.layers";
    m.layers.resize((size_t)g.n_layer);
    RAD_ARCH_TRY(m.layers[0].rq.probe(b, nm, g, nm.f("blk.0.attn_q"),
                                      { nm.ckpt("%s.0.self_attn.q_proj.weight", pre) }));
    {
        RadEncoding e0{};
        const bool known = weight_enc(b, nm.f("blk.0.attn_q.weight"), &e0);
        const W8hForm form = known ? linear_w8h_form(e0) : W8hForm::None;
        m.iq8r = known && linear_is_iq8r(e0);
        m.iqr_v4 = known && !m.iq8r && form == W8hForm::Trellis && linear_w8h_mad(e0);
        m.rotated_i8 = m.iq8r || m.iqr_v4;
        if (m.rotated_i8) m.rot_group = linear_w8h_group(e0);
    }
    if (m.rotated_i8 && (g.n_embd % m.rot_group || q_dim % m.rot_group ||
                         g.n_ff % m.rot_group)) {
        fprintf(stderr, "radiance: museglimmer: the rotated trunk's %lld-column activation group "
                        "does not divide this rank's widths (hidden %lld, attention %lld, "
                        "feed-forward %lld)\n", (long long)m.rot_group, (long long)g.n_embd,
                (long long)q_dim, (long long)g.n_ff);
        return RAD_E_UNSUPPORTED;
    }

    /* ---- activation buffers, sized at max_tok and shared by every layer; liveness over the
     * declared read and write sets fits them in one arena (spec §3.1). */
    m.b_x    = decl_b(b, nm.f("x"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.b_h    = decl_b(b, nm.f("h"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.b_q    = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, q_dim});
    m.b_gate = decl_b(b, nm.f("attn_gate"),g.act_dtype, {g.max_tok, q_dim});
    m.b_k    = decl_b(b, nm.f("attn_k"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_v    = decl_b(b, nm.f("attn_v"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_attn = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, q_dim});
    m.b_gu   = decl_b(b, nm.f("gate_up"),  g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.b_ffn  = decl_b(b, nm.f("ffn"),      g.act_dtype, {g.max_tok, g.n_ff});
    /* Rotated int8 pairs, one scale per row and `rot_group` columns. The two centered pre-norms
     * can fuse normalization and quantisation; the gated product can fuse its quantiser. The
     * attention gate/multiply stays BF16, so o_proj quantises that product separately. */
    if (m.rotated_i8) {
        m.a_h.x = m.b_h;
        RAD_ARCH_TRY(m.a_h.declare_qs(b, nm, g, "h", g.n_embd, 0, true, false, m.rot_group));
        m.a_h.q8_fed = m.a_h.q8_rot = rad_op_declared(b, "rmsnorm_had_quant_i8g") != 0;
        m.a_attn.x = m.b_attn;
        RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "attn_out", q_dim, 0, true, false,
                                         m.rot_group));
        m.a_ffn.x = m.b_ffn;
        RAD_ARCH_TRY(m.a_ffn.declare_qs(b, nm, g, "ffn", g.n_ff, 0, true, false, m.rot_group));
        m.a_ffn.q8_fed = m.a_ffn.q8_rot = rad_op_declared(b, "gated_had_quant_i8g") != 0;
    }
    /* The lm_head runs over the sampled rows only (RadBatch::out_ids), not the whole step. */
    m.b_hout   = decl_b(b, nm.f("h_out"),   g.act_dtype, {g.max_logit_rows, g.n_embd});
    m.b_logits = decl_b(b, nm.f("logits"),  RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the drafter's taps, declared before the trunk's layers: the buffer planner takes
     * lifetimes in declaration order (see the bf16 plugin's note). */
    if (ctx->max_spec > 0 && m.taps.read(meta, g.n_layer) > 0) {
        RAD_ARCH_TRY(m.taps.declare(b, nm, g, g.max_tok, m.b_x, g.act_dtype));
        if (m.taps.n <= 0) {
            fprintf(stderr, "radiance: museglimmer: the container carries a drafter whose "
                            "draft.target_layer_ids name no layer of this %lld-layer trunk\n",
                    (long long)g.n_layer);
            return RAD_E_INVAL;
        }
    }


    /* ---- vocabulary edges, both row-sharded. The 30B does not tie its head; the second source
     * serves family members that do, first present wins. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"),
                          nm.ckpt("model.language_model.norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.language_model.embed_tokens.weight")));

    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    /* A plain RMSNorm with a gain: declared at f32, cast at load, read with wadd 0. */
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.w_lm_head  = decl_w(b, nm.f("output.weight"),
                          weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                          {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                           {m.b_x}, {m.b_x});
    /* The embedding norm, after the vocabulary shard's reduction: a weightless RMS norm at
     * rms_norm_eps, in place. Weightless because MuseGlimmerTextNormedEmbedding has no gain. */
    m.op_embed_norm = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_STR("dtype", g.dtype)),
                             RAD_NOWEIGHTS),
                         {m.b_x}, {m.b_x});

    /* The encoder's rows overwrite the lookup before its reduction (as rad_block_vit.h's
     * MediaRows): declared only when the tower is served, issued on the passes that carry rows,
     * so the reduction counts each row once. */
    m.media = MuseTower::present(meta) && ctx->max_enc_patches > 0;
    if (m.media)
        m.op_media_rows = rw(b, RAD_OP(b, "scatter_rows",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                         RAD_STR("dtype", g.dtype)),
                              RAD_NOWEIGHTS),
                          {m.b_x}, {m.b_x});

    /* ---- calibration: the widest split input bounds the Hessian's width. */
    RAD_ARCH_TRY(m.calib.begin(b, nm, g, ctx, g.n_layer,
                               std::max({g.n_embd, q_dim * g.world, g.n_ff * g.world}),
                               std::max({g.n_embd, q_dim * g.world, g.n_ff * g.world})));

    /* ---- the layers. */
    const int64_t attn_scale_num = 1;   /* attention scale stays 1/sqrt(head_dim): the query's
                                         * qk_scale_factor rides the q norm's `mult`. */
    (void)attn_scale_num;

    /* One rotated linear: IQ8R must stay IQ8R; IQR-V4 must be a computed MAD trellis. LinearFP8
     * selects the stored code/trellis/scale views and gemm_nt_q i8a8h at the proper scale group. */
    auto rot_linear = [&](int64_t l, LinearFP8& L, const char* base,
                          std::initializer_list<const char*> src, int64_t n_out, int64_t n_in,
                          int shard, const ActFP8& a, rad_buf y,
                          std::initializer_list<int64_t> parts) -> int {
        if (!L.mapped) RAD_ARCH_TRY(L.probe(b, nm, g, base, src));
        const char* wn = nm.f("%s.weight", base);
        RadEncoding e{};
        const bool known = weight_enc(b, wn, &e);
        const W8hForm form = known ? linear_w8h_form(e) : W8hForm::None;
        const bool valid = known && (m.iq8r ? linear_is_iq8r(e)
                                             : m.iqr_v4 && form == W8hForm::Trellis &&
                                                   linear_w8h_mad(e));
        if (!valid || linear_w8h_group(e) != m.rot_group) {
            char en[256] = "plain bf16";
            if (known) rad_enc_format(&e, en, sizeof en);
            fprintf(stderr, "radiance: '%s' is %s, which does not match this container's whole "
                            "rotated trunk (layer 0 attn_q selects %s)\n", wn, en,
                    m.iq8r ? "IQ8R" : "computed-trellis IQR");
            return RAD_E_DTYPE;
        }
        return L.declare(b, nm, g, base, n_out, n_in, shard, grp_layer((int)l), src, a, y, 0,
                         parts);
    };
    /* The pre-norms' geometry for NormQuantFP8, which reads `wadd` from it: this family's input
     * and pre-feed-forward norms are centered, x_hat * (1 + w), at rms_norm_eps. */
    Geom gn = g;
    gn.wadd = 1.0f;

    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer&   lay  = m.layers[(size_t)l];
        lay.full     = full[(size_t)l];
        lay.rope     = theta[(size_t)l] > 0.0f;
        const bool   glob = lay.full;
        rad_kvgroup  kg   = glob ? m.kv_full : m.kv_win;
        const Geom&  gr   = lay.rope ? m.g_rope : m.g;

        /* -- weights. Shapes are [out, in]; gemm_nt reads B transposed. The linears are declared
         * encoding-aware; the norm gains stay f32 whatever the trunk quantises. q is declared
         * first: gate, k and v quantise its same norm output and, on the grouped grids, read its
         * int8 pair (MlLinear::in_from). */
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l),
                              nm.ckpt("%s.%lld.input_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_post.weight", l),
                              nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_norm.weight", l),
                              nm.ckpt("%s.%lld.pre_feedforward_layernorm.weight", pre, (long long)l)));
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ffn_post.weight", l),
                              nm.ckpt("%s.%lld.post_feedforward_layernorm.weight", pre, (long long)l)));

        lay.w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        lay.w_post = decl_w(b, nm.f("blk.%d.attn_post.weight", l), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        lay.w_fnorm  = decl_w(b, nm.f("blk.%d.ffn_norm.weight", l), RAD_F32, {g.n_embd},
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        lay.w_fpost  = decl_w(b, nm.f("blk.%d.ffn_post.weight", l), RAD_F32, {g.n_embd},
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

        /* -- ops, in run order except attn_paged, declared before kv_store like every family: the
         * two touch disjoint buffers and the block size comes from the resolved kernel. */
        /* A rotated trunk can fuse the centered norm and its int8 pair into one pass. */
        if (m.rotated_i8 && m.a_h.q8_fed)
            RAD_ARCH_TRY(lay.nq_attn.declare(b, gn, lay.w_norm, m.b_x, m.a_h, g.n_embd));
        else
            lay.op_norm = rw(b, RAD_OP(b, "rmsnorm",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                           RAD_F64("eps", g.eps), RAD_F64("wadd", 1.0),
                                           RAD_STR("dtype", g.dtype)),
                                RAD_WEIGHTS(lay.w_norm)),
                            {m.b_x}, {m.b_h});
        const char* const src_q = nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l);
        const char* const src_gate =
            nm.ckpt("%s.%lld.self_attn.gate_proj.weight", pre, (long long)l);
        const char* const src_k = nm.ckpt("%s.%lld.self_attn.k_proj.weight", pre, (long long)l);
        const char* const src_v = nm.ckpt("%s.%lld.self_attn.v_proj.weight", pre, (long long)l);
        if (m.rotated_i8) {
            /* q, gate, k and v read one int8 pair of the norm output: the norm's, or q's own
             * quantiser's where the norm writes none. The three after q read it as fed. */
            ActFP8 h_shared = m.a_h;
            h_shared.q8_fed = h_shared.q8_rot = true;
            RAD_ARCH_TRY(rot_linear(l, lay.rq, nm.f("blk.%d.attn_q", (int)l), {src_q}, q_dim,
                                    g.n_embd, RAD_SHARD_ROW, m.a_h, m.b_q, {}));
            RAD_ARCH_TRY(rot_linear(l, lay.rgate, nm.f("blk.%d.attn_gate", (int)l), {src_gate},
                                    q_dim, g.n_embd, RAD_SHARD_ROW, h_shared, m.b_gate, {}));
            RAD_ARCH_TRY(rot_linear(l, lay.rk, nm.f("blk.%d.attn_k", (int)l), {src_k}, kv_dim,
                                    g.n_embd, RAD_SHARD_ROW, h_shared, m.b_k, {}));
            RAD_ARCH_TRY(rot_linear(l, lay.rv, nm.f("blk.%d.attn_v", (int)l), {src_v}, kv_dim,
                                    g.n_embd, RAD_SHARD_ROW, h_shared, m.b_v, {}));
        } else {
            /* The linears, declared in run order after the norm that feeds them: q first, so
             * gate, k and v quantise its same norm output and, on the grouped grids, read its
             * int8 pair (MlLinear::in_from). o and down are cut along K, in whole scale groups.
             * Layer 0's q was mapped by the trunk probe. */
            if (l == 0)
                RAD_ARCH_TRY(lay.lq.declare(b, nm, g, nm.f("blk.%d.attn_q", (int)l), {}, q_dim,
                                            g.n_embd, RAD_SHARD_ROW, grp_layer((int)l), m.b_h,
                                            m.b_q));
            else
                RAD_ARCH_TRY(lay.lq.declare(b, nm, g, nm.f("blk.%d.attn_q", (int)l), {src_q},
                                            q_dim, g.n_embd, RAD_SHARD_ROW, grp_layer((int)l),
                                            m.b_h, m.b_q));
            RAD_ARCH_TRY(lay.lgate.declare(b, nm, g, nm.f("blk.%d.attn_gate", (int)l), {src_gate},
                              q_dim, g.n_embd, RAD_SHARD_ROW, grp_layer((int)l), m.b_h, m.b_gate,
                              {}, &lay.lq));
            RAD_ARCH_TRY(lay.lk.declare(b, nm, g, nm.f("blk.%d.attn_k", (int)l), {src_k},
                              kv_dim, g.n_embd, RAD_SHARD_ROW, grp_layer((int)l), m.b_h, m.b_k,
                              {}, &lay.lq));
            RAD_ARCH_TRY(lay.lv.declare(b, nm, g, nm.f("blk.%d.attn_v", (int)l), {src_v},
                              kv_dim, g.n_embd, RAD_SHARD_ROW, grp_layer((int)l), m.b_h, m.b_v,
                              {}, &lay.lq));
        }
        /* the q/k norms are weightless (no gain operand), the query carrying qk_scale_factor on
         * `mult`. Both run in place: a row's scale is computed before the row is written. */
        lay.op_qn = rw(b, RAD_OP(b, "rmsnorm",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                     RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                     RAD_F64("mult", (double)m.qk_mult), RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {m.b_q}, {m.b_q});
        lay.op_kn = rw(b, RAD_OP(b, "rmsnorm",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head_kv),
                                     RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {m.b_k}, {m.b_k});

        if (lay.rope) {
            lay.op_rq = rw(b, decl_rope(b, gr, g.max_tok, g.head_dim, g.n_head, 0, n_rot, "neox"),
                           {m.b_q}, {m.b_q});
            lay.op_rk = rw(b, decl_rope(b, gr, g.max_tok, g.head_dim, g.n_head_kv, 0, n_rot, "neox"),
                           {m.b_k}, {m.b_k});
        }

        lay.op_attn = rw(b, rad_decl_op(b, "attn_paged",
                             RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok),
                                        RAD_INT("head_dim", g.head_dim), RAD_INT("gqa", g.gqa()),
                                        RAD_INT("causal", 1),
                                        RAD_INT("window", glob ? 0 : m.window),
                                        RAD_INT("n_head", g.n_head),
                                        RAD_INT("block_size", RAD_KV_BLOCK),
                                        RAD_INT("max_seqs", g.max_seqs),
                                        RAD_INT("max_ctx", g.max_ctx),
                                        RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                             RAD_NOWEIGHTS),
                         {m.b_q}, {m.b_attn});

        lay.op_kv = rw(b, RAD_OP(b, "kv_store",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                      RAD_INT("n_head_kv", g.n_head_kv),
                                      RAD_INT("block_size", rad_kv_block_size(b, kg)),
                                      RAD_STR("kv_dtype", g.kv_dtype)),
                           RAD_NOWEIGHTS),
                       {m.b_k, m.b_v}, {});

        /* out = attn * sigmoid(gate): two ops, as in the gated-attention block. */
        lay.op_sig = rw(b, RAD_OP(b, "sigmoid",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                       RAD_INT("n", g.head_dim), RAD_STR("dtype", g.dtype)),
                            RAD_NOWEIGHTS),
                        {m.b_gate}, {m.b_gate});
        lay.op_mul = rw(b, RAD_OP(b, "mul",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_NOWEIGHTS),
                        {m.b_attn, m.b_gate}, {m.b_attn});
        const char* const src_o = nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l);
        /* The rotated path quantises o_proj's gated product into its int8 pair. */
        if (m.rotated_i8)
            RAD_ARCH_TRY(rot_linear(l, lay.ro, nm.f("blk.%d.attn_output", (int)l), {src_o},
                                    g.n_embd, q_dim, RAD_SHARD_COL, m.a_attn, m.b_h, {}));
        else
            RAD_ARCH_TRY(lay.lo.declare(b, nm, g, nm.f("blk.%d.attn_output", l), {src_o},
                              g.n_embd, q_dim, RAD_SHARD_COL, grp_layer(l), m.b_attn, m.b_h));

        if (g.world > 1)
            lay.op_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                          RAD_INT("min_bytes", g.wire_min_bytes)),
                               RAD_NOWEIGHTS),
                           {m.b_h}, {m.b_h});
        /* The branch's own norm, at post_norm_eps, before the residual add. */
        lay.op_post = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", m.post_eps), RAD_F64("wadd", 1.0),
                                        RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(lay.w_post)),
                         {m.b_h}, {m.b_h});
        lay.op_add = rw(b, RAD_OP(b, "add",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_NOWEIGHTS),
                        {m.b_x, m.b_h}, {m.b_x});

        /* ---- the feed-forward, the same double norm. A rotated trunk can fuse the norm and pair. */
        if (m.rotated_i8 && m.a_h.q8_fed)
            RAD_ARCH_TRY(lay.nq_ffn.declare(b, gn, lay.w_fnorm, m.b_x, m.a_h, g.n_embd));
        else
            lay.op_fnorm = rw(b, RAD_OP(b, "rmsnorm",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                             RAD_F64("eps", g.eps), RAD_F64("wadd", 1.0),
                                             RAD_STR("dtype", g.dtype)),
                              RAD_WEIGHTS(lay.w_fnorm)),
                          {m.b_x}, {m.b_h});
        const char* const src_gate_ff = nm.ckpt("%s.%lld.mlp.gate_proj.weight", pre, (long long)l);
        const char* const src_up = nm.ckpt("%s.%lld.mlp.up_proj.weight", pre, (long long)l);
        const char* const src_down = nm.ckpt("%s.%lld.mlp.down_proj.weight", pre, (long long)l);
        if (m.rotated_i8) {
            /* gate|up in the checkpoint's row order, two row parts; then silu(gate) * up. Where
             * `a_ffn` is fed, gated_had_quant_i8g writes down_proj's int8 pair, and the bf16
             * product only when the calibration taps read it (down_proj reads codes alone).
             * Where it is not, silu_mul writes the product and down_proj quantises it. */
            RAD_ARCH_TRY(rot_linear(l, lay.rgu, nm.f("blk.%d.ffn_gate_up", (int)l),
                                    {src_gate_ff, src_up}, 2 * g.n_ff, g.n_embd, RAD_SHARD_ROW,
                                    m.a_h, m.b_gu, {g.n_ff, g.n_ff}));
            if (m.a_ffn.q8_fed)
                RAD_ARCH_TRY(lay.gq_ffn.declare(b, g, m.b_gu, m.a_ffn, g.n_ff, "silu", 0,
                                                m.calib.records(l)));
            else
                lay.op_act = rw(b, RAD_OP(b, "silu_mul",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff),
                                             RAD_STR("dtype", g.dtype)),
                                  RAD_NOWEIGHTS),
                              {m.b_gu}, {m.b_ffn});
            RAD_ARCH_TRY(rot_linear(l, lay.rdown, nm.f("blk.%d.ffn_down", (int)l), {src_down},
                                    g.n_embd, g.n_ff, RAD_SHARD_COL, m.a_ffn, m.b_h, {}));
        } else {
            RAD_ARCH_TRY(lay.lgu.declare(b, nm, g, nm.f("blk.%d.ffn_gate_up", l),
                              {src_gate_ff, src_up},
                              2 * g.n_ff, g.n_embd, RAD_SHARD_ROW, grp_layer(l), m.b_h, m.b_gu,
                              {g.n_ff, g.n_ff}));
            lay.op_act = rw(b, RAD_OP(b, "silu_mul",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_ff),
                                         RAD_STR("dtype", g.dtype)),
                              RAD_NOWEIGHTS),
                          {m.b_gu}, {m.b_ffn});
            RAD_ARCH_TRY(lay.ldown.declare(b, nm, g, nm.f("blk.%d.ffn_down", l), {src_down},
                              g.n_embd, g.n_ff, RAD_SHARD_COL, grp_layer(l), m.b_ffn, m.b_h));
        }
        if (g.world > 1)
            lay.op_far = rw(b, RAD_OP(b, "all_reduce",
                                RAD_PARAMS(RAD_INT("world_size", g.world),
                                           RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                           RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                           RAD_INT("min_bytes", g.wire_min_bytes)),
                                RAD_NOWEIGHTS),
                            {m.b_h}, {m.b_h});
        lay.op_fpost = rw(b, RAD_OP(b, "rmsnorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                         RAD_F64("eps", m.post_eps), RAD_F64("wadd", 1.0),
                                         RAD_STR("dtype", g.dtype)),
                              RAD_WEIGHTS(lay.w_fpost)),
                          {m.b_h}, {m.b_h});
        lay.op_fadd = rw(b, RAD_OP(b, "add",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                         RAD_STR("dtype", g.dtype)),
                              RAD_NOWEIGHTS),
                          {m.b_x, m.b_h}, {m.b_x});

        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, kg));

        /* -- calibration taps, the same four a dense family records. */
        if (m.calib.records(l)) {
            const std::string L = std::string(pre) + "." + std::to_string(l);
            lay.cs_in = (int)m.calib.sites.size();
            RAD_ARCH_TRY(m.calib.site(b, g, l, L + ".input",
                                      L + ".self_attn.q_proj," + L + ".self_attn.k_proj," +
                                      L + ".self_attn.v_proj," + L + ".self_attn.gate_proj",
                                      m.b_h, g.n_embd, g.n_embd));
            lay.cs_o = (int)m.calib.sites.size();
            RAD_ARCH_TRY(m.calib.site(b, g, l, L + ".self_attn.o_proj.input", L + ".self_attn.o_proj",
                                      m.b_attn, q_dim, q_dim * g.world));
            lay.cs_mlp = (int)m.calib.sites.size();
            RAD_ARCH_TRY(m.calib.site(b, g, l, L + ".pre_feedforward.input",
                                      L + ".mlp.gate_proj," + L + ".mlp.up_proj",
                                      m.b_h, g.n_embd, g.n_embd));
            lay.cs_down = (int)m.calib.sites.size();
            RAD_ARCH_TRY(m.calib.site(b, g, l, L + ".mlp.down_proj.input", L + ".mlp.down_proj",
                                      m.b_ffn, g.n_ff, g.n_ff * g.world));
        }
    }

    /* ---- the head. The final norm is a plain RMSNorm (wadd 0) that also carries the logits'
     * pre-scale: its output feeds only the head, so mult here is logits * output_multiplier
     * exactly. */
    m.op_final_norm = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_F64("wadd", 0.0),
                                        RAD_F64("mult", (double)m.out_mult), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(m.w_out_norm)),
                         {m.b_x}, {m.b_h});
    if (m.calib.on && m.calib.hi == g.n_layer) {
        m.cs_head = (int)m.calib.sites.size();
        RAD_ARCH_TRY(m.calib.site(b, g, -1, "lm_head.input", "lm_head", m.b_h, g.n_embd, g.n_embd));
    }

    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {m.b_h}, {m.b_hout});
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", g.n_embd),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(m.w_lm_head)),
                     {m.b_hout}, {m.b_logits});
    /* cap * tanh(x / cap) over the f32 logits, after the pre-scale. */
    if (m.softcap > 0.0f) {
        m.op_cap = rw(b, RAD_OP(b, "logit_softcap",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                     RAD_INT("n", g.n_vocab), RAD_F64("cap", (double)m.softcap),
                                     RAD_STR("dtype", "f32")),
                          RAD_NOWEIGHTS),
                      {m.b_logits}, {m.b_logits});
        if (!m.op_cap)
            rad_note(b, "warning: no kernel serves `logit_softcap` (load libr4d or libavx), so "
                        "the logits skip %g * tanh(x / %g): greedy output is unchanged, sampled "
                        "output draws from a sharper distribution", (double)m.softcap,
                    (double)m.softcap);
    }

    /* ---- media: the vision tower, declared last so its activations share the arena with every
     * buffer above (the trunk's weights were mapped and declared first). A tower linear is BF16
     * or int8 with a bf16 scale per (row, 128 of K); block FP8 is retired. The stem, norms and
     * biases stay BF16, which the tower refuses by name if a recipe quantised them. */
    if (m.media) {
        const char* tower_srcs[] = {
            nm.ckpt("model.vision_tower.patch_embedder.patch_embedding.weight"),
            nm.ckpt("model.vision_tower.layers.0.attn.q_proj.weight"),
            nm.ckpt("model.vision_tower.layers.0.mlp.fc1.weight"),
            nm.ckpt("model.vision_adapter.fc1.weight"),
            nm.ckpt("model.vision_projection.weight"),
        };
        for (const char* src : tower_srcs) {
            RadEncoding te{};
            if (!weight_enc(b, src, &te)) continue;
            const bool plain = rad_enc_is(&te, "plain") && te.n_planes == 1 &&
                               te.plane[0].dtype == RAD_BF16;
            if (plain || linear_is_w8(te)) continue;
            if (const char* gone = linear_retired(te))
                return linear_refuse_retired(src, gone);
            char ten[256];
            rad_enc_format(&te, ten, sizeof ten);
            fprintf(stderr, "radiance: '%s' is %s; a vision linear is BF16 or int8 with a bf16 "
                            "scale per (row, 128 of K)\n", src, ten);
            return RAD_E_DTYPE;
        }
        RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer));
    }

    /* ---- the drafter is declared last so its buffers cannot overlap trunk temporaries. */
    if (m.taps.n > 0) {
        if (muse_draft_is_dflash2(meta)) {
            RAD_ARCH_TRY(declare_muse_dflash2(&m.df2, &m.df2w, &m.kv_draft, b, nm, meta, ctx, g,
                                              m.taps, (int)g.n_layer, m.w_tok, m.w_lm_head));
            m.have_df2 = true;
        } else {
            RAD_ARCH_TRY(declare_muse_dflash(&m.df, &m.dfw, &m.kv_draft, b, nm, meta, ctx, g,
                                             m.taps, (int)g.n_layer, m.w_tok, m.w_lm_head));
            m.have_df = true;
        }
    }

    /* ---- notes for --debug-graph. */
    int64_t nfull = 0, nrope = 0;
    for (int64_t l = 0; l < g.n_layer; ++l) {
        nfull += full[(size_t)l];
        nrope += theta[(size_t)l] > 0.0f;
    }
    rad_note(b, "rank %d/%d: %lld layers (%lld full attention, %lld sliding over a %lld-token "
                "window; %lld with rotary at theta %g), n_head=%lld n_head_kv=%lld head_dim=%lld "
                "n_ff=%lld vocab=[%lld,%lld)",
             g.rank, g.world, (long long)g.n_layer, (long long)nfull,
             (long long)(g.n_layer - nfull), (long long)m.window, (long long)nrope,
             theta_rope, (long long)g.n_head, (long long)g.n_head_kv, (long long)g.head_dim,
             (long long)g.n_ff, (long long)g.vocab_off, (long long)(g.vocab_off + g.n_vocab));
    rad_note(b, "centered norms (1 + w): input/pre at eps %g, post at eps %g; the final norm is a "
                "plain gain; the embedding and the q/k norms are weightless", (double)g.eps,
            (double)m.post_eps);
    rad_note(b, "attention scale 1/sqrt(%lld) with the query pre-scaled by qk_scale_factor %g; "
                "logits pre-scaled by output_multiplier %g%s%g",
             (long long)g.head_dim, (double)m.qk_mult, (double)m.out_mult,
             m.softcap > 0.0f ? " and soft-capped at " : "", (double)m.softcap);
    if (m.g_rope.theta != g.theta)
        rad_note(b, "rotary runs on the sliding layers only (theta %g); the full-attention layers "
                    "are NoPE", (double)m.g_rope.theta);
    if (m.rotated_i8)
        rad_note(b, "%s trunk: gemm_nt_q i8a8h over 128-wide Hadamard codes, one activation scale "
                    "per %lld K columns; the pre-norms %s, the gated product %s, o_proj quantises "
                    "its own input; q, gate, k and v read one pair",
                 m.iq8r ? "IQ8R" : "computed-trellis IQR", (long long)m.rot_group,
                 m.a_h.q8_fed ? "write the codes (rmsnorm_had_quant_i8g)"
                              : "are plain (q and gate/up quantise)",
                 m.a_ffn.q8_fed ? "writes down_proj's codes (gated_had_quant_i8g)"
                                : "is plain (down_proj quantises)");
    rad_note(b, "logits for at most %lld sampled rows a step (of %lld tokens): gather_rows "
                "selects RadBatch::out_ids before the lm_head", (long long)g.max_logit_rows,
            (long long)g.max_tok);

    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_slot(c)];

    /* An encoder pass runs only the vision tower (RadBatch::enc). */
    if (batch->enc) {
        m.vit.step(c, batch);
        return;
    }

    /* A draft pass runs only the drafter: >0 the query pass on the staged block, <0 the context
     * pass that projects and stores K/V for the tokens the trunk just committed. */
    if (batch->draft_pass != 0) {
        if (m.have_df2) {
            if (batch->draft_pass > 0) m.df2.step(c, batch);
            else                       m.df2.context(c, batch);
            return;
        }
        if (!m.have_df) {
            rad_step_fail(c, "this serve stages a draft pass but the container carries no drafter");
            return;
        }
        if (batch->draft_pass > 0) m.df.step(c, batch);
        else                       m.df.context(c, batch);
        return;
    }

    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
              brows(m.b_x, batch->n_tok));
    /* The encoder's rows overwrite the lookup before the vocabulary shard's reduction, so the
     * reduction counts each row once. */
    if (m.op_media_rows && batch->n_mm_rows > 0)
        RAD_ISSUE_N(c, m.op_media_rows, batch->n_mm_rows,
                    praw2(batch->mm_embd, RAD_BF16, batch->n_mm_rows, m.g.n_embd),
                    praw(batch->mm_rows, RAD_I32, batch->n_mm_rows),
                    brows(m.b_x, batch->n_tok));
    /* The vocab shard's reduction; declared only when world > 1, so tp1 is unchanged. */
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);
    /* The embedding norm, after the reduction: it normalises whole rows. */
    RAD_ISSUE(c, m.op_embed_norm, brows(m.b_x, batch->n_tok), RAD_NONE, brows(m.b_x, batch->n_tok));

    const int64_t T = batch->n_tok;
    for (size_t li = 0; li < m.layers.size(); ++li) {
        const Layer& l = m.layers[li];
        const rad_kvgroup kg = l.full ? m.kv_full : m.kv_win;
        const RadKVGroupBatch* kvb = kv_batch(batch, kg);
        const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
        const int32_t* table = kvb ? kvb->block_table  : nullptr;
        const int32_t* used  = kvb ? kvb->seqused      : nullptr;

        /* -- attention. An IQ8R layer issues LinearFP8's ops over the int8 pairs; a linear that
         * declared a quantiser (op_q) issues it first, over the pair's bf16 plane. */
        if (l.nq_attn.op) l.nq_attn.step(c, T);
        else RAD_ISSUE(c, l.op_norm, brows(m.b_x, T), RAD_W(l.w_norm), brows(m.b_h, T));
        if (l.cs_in >= 0) m.calib.tap(c, (size_t)l.cs_in, T);
        if (m.rotated_i8) {
            l.rq.step(c, m.a_h, m.b_q, T);
            l.rgate.step(c, m.a_h, m.b_gate, T);
            l.rk.step(c, m.a_h, m.b_k, T);
            l.rv.step(c, m.a_h, m.b_v, T);
        } else {
            l.lq.step(c, m.b_h, m.b_q, T);
            l.lgate.step(c, m.b_h, m.b_gate, T);
            l.lk.step(c, m.b_h, m.b_k, T);
            l.lv.step(c, m.b_h, m.b_v, T);
        }
        RAD_ISSUE_N(c, l.op_qn, T * m.g.n_head,      brows(m.b_q, T), RAD_NONE, brows(m.b_q, T));
        RAD_ISSUE_N(c, l.op_kn, T * m.g.n_head_kv,   brows(m.b_k, T), RAD_NONE, brows(m.b_k, T));
        if (l.rope) {
            RAD_ISSUE(c, l.op_rq, brows(m.b_q, T), rope_posmc(m.g, batch, T));
            RAD_ISSUE(c, l.op_rk, brows(m.b_k, T), rope_posmc(m.g, batch, T));
        }
        RAD_ISSUE(c, l.op_kv,
                  brows(m.b_k, T), brows(m.b_v, T), praw(slot, RAD_I32, T),
                  kv_cache(kg, (int)li));
        /* The longest query length, not the token count, selects the attention band. */
        RAD_ISSUE_N(c, l.op_attn, batch->max_q_len,
                    brows(m.b_q, T), kv_cache(kg, (int)li),
                    praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(m.b_attn, T));
        RAD_ISSUE_N(c, l.op_sig, T * m.g.n_head, brows(m.b_gate, T), brows(m.b_gate, T));
        RAD_ISSUE(c, l.op_mul, brows(m.b_attn, T), brows(m.b_gate, T), brows(m.b_attn, T));
        if (l.cs_o >= 0) m.calib.tap(c, (size_t)l.cs_o, T);
        if (m.rotated_i8) l.ro.step(c, m.a_attn, m.b_h, T);
        else        l.lo.step(c, m.b_attn, m.b_h, T);
        if (l.op_ar) RAD_ISSUE_N(c, l.op_ar, T * m.g.n_embd, brows(m.b_h, T), RAD_NONE);
        RAD_ISSUE(c, l.op_post, brows(m.b_h, T), RAD_W(l.w_post), brows(m.b_h, T));
        RAD_ISSUE(c, l.op_add,  brows(m.b_x, T), brows(m.b_h, T), brows(m.b_x, T));

        /* -- feed-forward. */
        if (l.nq_ffn.op) l.nq_ffn.step(c, T);
        else RAD_ISSUE(c, l.op_fnorm, brows(m.b_x, T), RAD_W(l.w_fnorm), brows(m.b_h, T));
        if (l.cs_mlp >= 0) m.calib.tap(c, (size_t)l.cs_mlp, T);
        if (m.rotated_i8) {
            l.rgu.step(c, m.a_h, m.b_gu, T);
            if (l.gq_ffn.op) l.gq_ffn.step(c, m.b_gu, T);
            else RAD_ISSUE(c, l.op_act, brows(m.b_gu, T), brows(m.b_ffn, T));
            l.rdown.step(c, m.a_ffn, m.b_h, T);
        } else {
            l.lgu.step(c, m.b_h, m.b_gu, T);
            RAD_ISSUE(c, l.op_act,   brows(m.b_gu, T), brows(m.b_ffn, T));
            l.ldown.step(c, m.b_ffn, m.b_h, T);
        }
        if (l.cs_down >= 0) m.calib.tap(c, (size_t)l.cs_down, T);
        if (l.op_far) RAD_ISSUE_N(c, l.op_far, T * m.g.n_embd, brows(m.b_h, T), RAD_NONE);
        RAD_ISSUE(c, l.op_fpost, brows(m.b_h, T), RAD_W(l.w_fpost), brows(m.b_h, T));
        RAD_ISSUE(c, l.op_fadd,  brows(m.b_x, T), brows(m.b_h, T), brows(m.b_x, T));

        /* The drafter reads this layer's completed residual, if it taps it. */
        m.taps.step(c, m.b_x, T, (int64_t)li, m.g.n_embd);
    }

    RAD_ISSUE(c, m.op_final_norm,
              brows(m.b_x, batch->n_tok), RAD_W(m.w_out_norm), brows(m.b_h, batch->n_tok));
    if (m.cs_head >= 0) m.calib.tap(c, (size_t)m.cs_head, T);
    m.calib.after_step(c, T);

    /* The lm_head runs only on the rows the sampler asked for; a mid-prompt prefill chunk has
     * n_out zero and skips it. */
    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.b_h, batch->n_tok),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                    brows(m.b_logits, batch->n_out));
        if (m.op_cap)
            RAD_ISSUE_N(c, m.op_cap, batch->n_out, brows(m.b_logits, batch->n_out),
                        brows(m.b_logits, batch->n_out));
    }
}

/* What the plugin says before it declares anything: this family's tower merges 4 patches into a
 * row, so an automatic --mm-max-patches rounds to whole merged rows. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    if (MuseTower::present(meta)) out->enc_merge = 4;
    /* The drafter denoises `block_size` positions, the anchor row proposes nothing. */
    if (muse_draft_is_dflash2(meta)) {
        const long long bs = rad_meta_geti(meta, "draft.dflash_config.block_size", 16);
        out->draft_depth = bs > 1 ? (int)(bs - 1) : 15;
        out->draft_depth_fixed = 1;
        return RAD_OK;
    }
    const long long bs = rad_meta_geti(meta, "draft.block_size", 0);
    if (bs > 1) {
        out->draft_depth = (int)(bs - 1);
        out->draft_depth_fixed = 1;
    }
    return RAD_OK;
}

static const RadChatFormat* chat_format(const RadModelMeta*) {
    return museglimmer::muse_chat_format();
}

}  /* namespace museglimmer_ct */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(museglimmer_ct)
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(museglimmer_ct, "museglimmer", "compressed-tensors", "0.1.0",
                "Meta Muse Glimmer 30B dense under compressed-tensors: encoding-aware trunk "
                "linears (bf16, asymmetric/symmetric integer, IQ8R, or fwht128 computed-MAD "
                "IQR-V4 at i8a8h), gated attention alternating sliding-window (rotary) and NoPE "
                "full-attention layers, Gemma-2 style double norms, bf16 vision tower, and DFlash "
                "or DFlash2 drafter")
RAD_ARCH_CHAT_FORMAT(museglimmer_ct)
#endif
