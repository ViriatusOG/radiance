/* rad_block_flux2_vae.h -- FLUX.2's VAE decoder as a block. arch/mossimage declares its decoder
 * through it; arch/flux3 declares the same steps inline (flux3.cpp), and could move onto it.
 *
 * `AutoencoderKLFlux2` (diffusers) is not a Wan-style VAE with a residual shortcut: its latent is
 * normalised by a BatchNorm's running statistics over the *packed* width (4 x latent_channels),
 * its decoder unpacks 2x2 to `latent_channels` and then runs ordinary GroupNorm residual blocks
 * with nearest-neighbour upsampling, and an optional 1x1 `post_quant_conv` sits in front of
 * `decoder.conv_in`. Both models in this tree that make pictures with it run the same four steps:
 *
 *   1. chan_stats `denorm`   x = rd(rd(x std) + mean), std = sqrt(var + eps), over the packed rows
 *   2. fill + d2s_add        the 2x2 unpack: latent channel 4*co + 2i + j -> pixel (2h+i, 2w+j)
 *   3. the block graph       post_quant_conv, conv_in, the mid block (two resnets around an
 *                            attention), four up blocks of `layers_per_block + 1` resnets with an
 *                            upsampler between them, conv_norm_out + silu, conv_out
 *   4. to_rgba8              -1..1 bf16 -> 0..255 u8 rows
 *
 * The declared liveness is the caller's: it hands in its own state/preview buffers and gets the
 * ops back, so the Vae's own op list holds only what the decoder issues.
 */
#ifndef RAD_BLOCK_FLUX2_VAE_H
#define RAD_BLOCK_FLUX2_VAE_H

#include "rad_arch.h"
#include "rad_vae2d.h"

namespace rad {
namespace arch {

/* The denormalisation and the unpack, declared against the caller's latent state buffers. `NT` is
 * the largest latent token count, `C` the packed width (4 x latent_channels), `zc` the VAE's own
 * latent channel count, and `cu` a 2-row i32 that `cu_repeat` fills with {0, 4 * pixels}. */
struct Flux2VaeIn {
    rad_weight w_mean = 0, w_std = 0;
    rad_op     op_bn = 0, op_zero = 0, op_unpack = 0, op_cu4 = 0;
};

inline int flux2_vae_declare_in(RadBuilder* b, Names& nm, Flux2VaeIn* in, int64_t NT, int64_t C,
                                int64_t zc, double bn_eps, rad_buf state, rad_buf preview,
                                rad_buf* b_dl, rad_buf* b_din, rad_buf* b_cu) {
    *b_dl  = decl_b(b, nm.f("vdec.lat"), RAD_BF16, { NT, C });
    *b_din = decl_b(b, nm.f("vdec.in"), RAD_BF16, { 4 * NT, zc });
    *b_cu  = decl_b(b, nm.f("vdec.cu"), RAD_I32, { 2 });
    if (map_copy(b, nm.f("vae.bn.mean"), nm.ckpt("vae.bn.running_mean")) < 0) return RAD_E_INVAL;
    in->w_mean = decl_w(b, nm.f("vae.bn.mean"), RAD_BF16, { C }, RAD_ACCESS_PER_REQUEST,
                        RAD_SHARD_NONE, grp_layer(-1));
    if (map_copy(b, nm.f("vae.bn.var"), nm.ckpt("vae.bn.running_var")) < 0) return RAD_E_INVAL;
    in->w_std = decl_w(b, nm.f("vae.bn.var"), RAD_BF16, { C }, RAD_ACCESS_PER_REQUEST,
                       RAD_SHARD_NONE, grp_layer(-1));
    in->op_bn = rw(b, RAD_OP(b, "chan_stats",
                             RAD_PARAMS(RAD_RANGE("M", 1, NT), RAD_INT("n", C),
                                        RAD_STR("mode", "denorm"), RAD_INT("var", 1),
                                        RAD_F64("eps", bn_eps), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(in->w_mean, in->w_std)),
                   { state, preview }, { *b_dl });
    in->op_zero = rw(b, RAD_OP(b, "fill",
                               RAD_PARAMS(RAD_RANGE("M", 1, 4 * NT), RAD_INT("n", zc),
                                          RAD_STR("dtype", "bf16"), RAD_F64("value", 0.0)),
                               RAD_NOWEIGHTS),
                     {}, { *b_din });
    in->op_unpack = rw(b, RAD_OP(b, "d2s_add",
                                 RAD_PARAMS(RAD_RANGE("M", 1, 4 * NT), RAD_INT("cin", C),
                                            RAD_INT("cout", zc), RAD_INT("ft", 1),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       { *b_dl, *b_din }, { *b_din });
    in->op_cu4 = rw(b, RAD_OP(b, "cu_repeat", RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("mult", 4)),
                              RAD_NOWEIGHTS),
                    {}, { *b_cu });
    return in->op_bn && in->op_zero && in->op_unpack && in->op_cu4 ? RAD_OK : RAD_E_INVAL;
}

/* Steps 1-2 and `cu_repeat`; the caller issues `to_rgba8` itself. `lane_row` is the lane's
 * first latent row in `src` (the state or the preview), `h`/`w` the latent grid, `cu` the
 * batch's {0, pixels}, which the caller issues as `cu_repeat` rather than writing. */
inline void flux2_vae_issue_in(RadCtx* c, const Flux2VaeIn& in, rad_buf dl, rad_buf din,
                              rad_buf cu4, rad_buf src, int64_t lane_row, int64_t h, int64_t w,
                              int64_t C, int64_t zc, const RadOperand& pixel_cu) {
    const int64_t n = h * w, H0 = 2 * h, W0 = 2 * w;
    RAD_ISSUE_N(c, in.op_bn, n, brow_slice(src, lane_row, n, C), RAD_W(in.w_mean),
                RAD_W(in.w_std), brows(dl, n));
    RAD_ISSUE_N(c, in.op_zero, H0 * W0, brows(din, H0 * W0));
    RAD_ISSUE_N(c, in.op_unpack, H0 * W0, RAD_P_T2(rad_buf_ptr(c, dl), RAD_BF16, h, w * C),
                RAD_P_T2(rad_buf_ptr(c, din), RAD_BF16, H0, W0 * zc));
    RAD_ISSUE_N(c, in.op_cu4, 2, pixel_cu, brows(cu4, 2));
}

/* Declares the block graph from `vae.post_quant_conv` on, into `v`, and returns the output image
 * (3 channels at level 0). `ch` is `vae.decoder_block_out_channels` (reversed for the up blocks),
 * `nres` `vae.layers_per_block`, `groups` `vae.norm_num_groups`, `zc` `vae.latent_channels`. */
struct Flux2VaeDec {
    VImg       out{};
    rad_op     op_rgb = 0;
    rad_buf    b_img = 0;
};

inline int flux2_vae_declare_dec(RadBuilder* b, RadModelMeta const* meta, Names& nm, Vae& v,
                                 Flux2VaeDec* d, int64_t px, int64_t NT, int64_t zc,
                                 int64_t pixel_channels, const rad_buf* din, rad_buf cu,
                                 int first_group) {
    v.nm = &nm;
    v.b = b;
    v.scope = "vdec";
    v.down = false;
    v.first_group = first_group;
    v.base_px = NT * 4;
    std::vector<double> ch;
    {
        const char* s = rad_meta_gets(meta, "vae.decoder_block_out_channels", nullptr);
        while (s && *s) {
            char* e = nullptr;
            const double x = std::strtod(s, &e);
            if (e == s) { ++s; continue; }
            ch.push_back(x);
            s = e;
        }
    }
    if (ch.empty()) ch = { 128, 256, 512, 512 };
    const int64_t nres = rad_meta_geti(meta, "vae.layers_per_block", 2);
    const int64_t groups = rad_meta_geti(meta, "vae.norm_num_groups", 32);
    const double eps = 1e-6;
    const std::vector<double> rch(ch.rbegin(), ch.rend());
    VImg x{ *din, zc, 0 };
    if (rad_meta_geti(meta, "vae.use_post_quant_conv", 1))
        x = v.conv(x, zc, 1, 1, 1, "same", "vae.post_quant_conv");
    x = v.conv(x, (int64_t)rch[0], 3, 1, 1, "same", "vae.decoder.conv_in");
    x = v.resblock_gn(x, (int64_t)rch[0], "vae.decoder.mid_block.resnets.0", groups, eps);
    x = v.attention_gn(x, "vae.decoder.mid_block.attentions.0", groups, eps);
    x = v.resblock_gn(x, (int64_t)rch[0], "vae.decoder.mid_block.resnets.1", groups, eps);
    for (size_t i = 0; i < rch.size(); ++i) {
        for (int64_t r = 0; r < nres + 1; ++r)
            x = v.resblock_gn(x, (int64_t)rch[i],
                              nm.f("vae.decoder.up_blocks.%d.resnets.%d", (int)i, (int)r), groups,
                              eps);
        if (i + 1 < rch.size())
            x = v.conv(x, x.ch, 3, 1, 2, "same",
                       nm.f("vae.decoder.up_blocks.%d.upsamplers.0.conv", (int)i));
    }
    x = v.gnorm(x, true, "vae.decoder.conv_norm_out", groups, eps);
    d->out = v.conv(x, pixel_channels, 3, 1, 1, "same", "vae.decoder.conv_out");
    if (!d->out.buf) return RAD_E_INVAL;
    RAD_ARCH_TRY(v.attn_reads(cu));
    d->b_img = decl_b(b, nm.f("image_out"), RAD_U8, { px * pixel_channels }, RAD_BUF_PERSIST);
    d->op_rgb = rw(b, RAD_OP(b, "to_rgba8",
                             RAD_PARAMS(RAD_RANGE("M", 1, px), RAD_INT("n", pixel_channels),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { d->out.buf }, { d->b_img });
    return d->op_rgb ? RAD_OK : RAD_E_INVAL;
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_FLUX2_VAE_H */
