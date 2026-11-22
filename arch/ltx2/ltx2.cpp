/* ltx2 -- LTX-2.5: a video and its stereo soundtrack from text, or from text and a first frame.
 *
 * ============================== THE MODEL ==============================
 *
 * diffusers' LTX2Pipeline (LTX-2.5's components, Lightricks/LTX-2.5-Diffusers):
 *
 *   Text encoder: Gemma 4's language model over the prompt, left-padded to 1024 tokens (no
 *   template; the tokenizer's <bos> and <eos> around it). Every layer's output is kept -- the
 *   embeddings, each layer's but the last, and the final norm of the last -- and each token's
 *   stack of them is RMS-normed per layer and flattened channel-major.
 *
 *   Connectors: per modality a projection of that stack to the DiT's width (video 4096, audio
 *   2048), then 1024 rows -- the prompt's tokens, the rest filled from 128 learnable registers --
 *   through eight bidirectional transformer blocks with a split rotary over the row index.
 *
 *   Denoiser: a dual-stream DiT, 48 blocks over the video's latents (1 x 1 patches of 128 channels,
 *   32 pixels a latent, 8 frames a latent frame) and the soundtrack's (25 a second, 8 channels x 16
 *   mel bins). A block's video stream runs self-attention, cross-attention to its text, attention
 *   to the soundtrack, and a feed-forward, each behind an adaptive norm; the soundtrack's the same
 *   against the video. Every attention is gated per head (2 sigmoid of a logit) and normalises q
 *   and k across all heads.
 *
 *   Guidance: up to four DiT evaluations a step -- the prompt, the negative prompt, the prompt with
 *   block 28's self-attentions bypassed (spatio-temporal guidance) and the prompt without the
 *   cross-modal attentions -- combined in the clean-estimate space with a rescale
 *   (abi/rad_builder.h, RadVideoGenDecl::guidance; the op ltx_guide).
 *
 *   Video VAE: a causal 3D-conv decoder (four up blocks of residual blocks, depth-to-space
 *   resampling, PerChannelRMSNorm), 4 x 4 pixel patches out; its encoder takes a first frame in.
 *
 *   Audio: a causal 2D-conv decoder to a stereo log-mel spectrogram at 16 kHz, a BigVGAN vocoder
 *   to 16 kHz, and a bandwidth extender (its own vocoder over the waveform's log-mel spectrogram
 *   plus a sinc-resampled skip) to 48 kHz.
 *
 *   Latent upsampler (latent_upsampler/): two-times spatial upscaling of the video latent for the
 *   two-stage recipe (a distilled release's stage-2 rungs, RadVideoGenDecl::refine_rungs).
 *
 *   Duration head (duration_head/): the length the prompt implies, pooled by one learned query
 *   from both connectors' output (RadVideoGenDecl::duration_out).
 *
 *   Diffusion decoder (diffusion_decoder/): the release's second decoder, on request
 *   (RadVideoGenDecl::decoders): neighbourhood-attention stages over the latent, then one denoising
 *   step over pixel patches, tile by tile (declare_ddec says how).
 *
 * The passes are the core's video generator (core/gen/video_driver.h):
 *
 *   VCOND    the text encoder over both prompts, the connectors into the lane's text rows, the
 *            duration head over the prompt's, and with a first frame the VAE encoder into the
 *            lane's clean first latent frame.
 *   VSTEP    the step's modulation tables, then each guidance evaluation the request needs, then
 *            one ltx_guide a modality; with a first frame, its latent frame restored.
 *   ADECODE  the soundtrack: the audio VAE, the vocoder, the bandwidth extender.
 *   VDECODE  chunk 0 decodes the whole video into the lane's tile store, a tile at a time for a
 *            frame larger than the default canvas; every chunk then stitches its frames to bytes.
 *   VUPSCALE the latent upsampler, then both latents re-noised for the second stage.
 *
 * ============================== PRECISION ==============================
 *
 * Every network computes in bf16, as the reference loads it; the latents, the guidance and the
 * sampler are f32. The text encoder's, the connectors' and the DiT's large linears are bf16 or
 * rotated int8 as the container holds them (data/recipes/ltx-2.5-bf16.recipe,
 * ltx-2.5-iq8r.recipe), one kind a container; the vocoder runs in
 * f32 (its anti-aliased activation's kernels are f32), its log-mel input rounded where the
 * reference's bf16 one is.
 *
 * ============================== LIMITS ==============================
 *
 * One rank (TP 1): the DiT's q and k norms span every head, which a head split would cut. A step
 * carries 121 frames at 1920 x 1088 at most; the diffusion decoder decodes the default canvas's
 * area at most. The prompt enhancer and the temporal upsampler (the DFR pipeline's) are not run.
 */
#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ltx2 {

using namespace rad::arch;

/* ================================================================== the model's constants */
/* Connector rows a prompt: the text encoder's padded length. */
enum { kTextRows = 1024 };
/* Requests holding state at once: one denoising, one conditioned and waiting. */
enum { kMaxLanes = 2 };
/* Evaluations a request may take, and the base model's default. */
enum { kMaxSteps = 64, kDefSteps = 30 };
/* Time: 24 frames a second; a video of n chunks is 8 n + 1 frames over n + 1 latent frames. A
 * decode pass is a chunk of frames; chunk 0 decodes the whole video, a tile at a time. */
enum { kFps = 24, kFpc = 8, kFtail = 1, kLpc = 1, kLtail = 1, kDecLat = 2 };
enum { kMinChunks = 2, kMaxChunks = 40, kDefChunks = 15 };
/* Space: 32 pixels a latent; sides multiples of 32 from 256, 3:1 at most; the release's 960 x 544
 * canvas by default, and four times its area at most -- the two-stage recipe's 1920 x 1088. A
 * step holds at most the video rows of 121 frames at that size. */
enum { kScale = 32, kMult = 32, kMinSide = 256, kMaxAspect = 3 };
enum { kDefW = 960, kDefH = 544, kMaxArea = 4 * kDefW * kDefH };
enum { kMaxVideoRows = 16 * (2 * kDefW / kScale) * (2 * kDefH / kScale) };
/* The VAE decodes a frame of the default canvas's latents or fewer whole, as the pipeline does
 * without tiling; a larger one in diffusers' tiles (enable_tiling: 512 pixels every 448, 16 latents
 * every 14), blended where they overlap (tile_stitch). */
enum { kVaeLat = kDefW * kDefH / (kScale * kScale), kTile = 16, kTileStride = 14 };
/* The video latent's channels (a 1 x 1 patch), the VAE's pixel patch and its channels. */
enum { kC = 128, kPix = 4, kPixC = 3 * kPix * kPix };
/* The soundtrack: 25 latents a second of 8 channels x 16 mel bins, packed channel-major; its
 * spectrogram 64 bins at 16 kHz with 160-sample hops; 48 kHz stereo out. */
enum { kAudioRate = 25, kAudioCh = 2, kAudioC = 128, kLatCh = 8, kLatMel = 16, kMel = 64 };
enum { kRate = 48000, kVocRate = 16000, kMelHop = 160 };
/* Guidance terms (RadVideoGenBatch::guidance): the video's then the soundtrack's four. */
enum { kGsV = 0, kStgV = 1, kMsV = 2, kGsA = 4, kStgA = 5, kMsA = 6 };
/* The block whose self-attentions spatio-temporal guidance bypasses. */
enum { kStgBlock = 28 };
/* An evaluation of the DiT within a step: the prompt, the negative prompt, the prompt with STG,
 * the prompt without the cross-modal attentions. */
enum { kPassCond = 0, kPassUncond = 1, kPassStg = 2, kPassIso = 3, kPasses = 4 };

/* ================================================================== geometry, on the host */
static int64_t max_lat_frames() { return (int64_t)kLpc * kMaxChunks + kLtail; }
static int64_t max_patches() { return (int64_t)kMaxArea / (kScale * kScale); }
static int64_t max_video_rows() { return kMaxVideoRows; }
static int64_t max_frames() { return (int64_t)kFpc * kMaxChunks + kFtail; }
static int64_t max_audio() { return (max_frames() * kAudioRate + kFps / 2) / kFps + 1; }
static int64_t max_step_rows() { return kTextRows + max_audio() + max_video_rows(); }
/* The spectrogram's frames for n audio latents (the audio VAE's causal upsampling). */
static int64_t mel_frames(int64_t n) { return 4 * n - 3; }

/* The VAE's tiles of an h x w latent grid, as tile_grid cuts them: one when the frame is the
 * untiled decoder's, else a tile every kTileStride latents, kTile long or to the edge. */
struct VaeTile { int64_t h, w; };
static bool vae_tiled(int64_t h, int64_t w) { return h * w > kVaeLat; }
static std::vector<VaeTile> vae_tiles(int64_t h, int64_t w) {
    if (!vae_tiled(h, w)) return { { h, w } };
    std::vector<VaeTile> t;
    for (int64_t y0 = 0; y0 < h; y0 += kTileStride)
        for (int64_t x0 = 0; x0 < w; x0 += kTileStride)
            t.push_back({ std::min<int64_t>(kTile, h - y0), std::min<int64_t>(kTile, w - x0) });
    return t;
}

/* The step's rotary coordinates, flat (the core hands them through as RadVideoGenBatch::pos): the
 * video's (t, y, x) midpoints as [3, S] component-major, then the soundtrack's time midpoints,
 * f32 as the reference computes them (LTX2AudioVideoRotaryPosEmbed's prepare_*_coords at 24
 * frames a second). A latent frame f spans pixel frames [max(8 f - 7, 0), 8 f + 1) over the fps;
 * a latent (h, w) pixels [32 h, 32 h + 32); an audio latent mel frames [max(4 f - 3, 0), 4 f + 1)
 * at 160 / 16000 s a frame. */
static int64_t positions(const RadVideoGeom* g, float* pos, int64_t cap) {
    if (!g || g->n_text < 1 || g->lat_frames < 1 || g->lat_h < 1 || g->lat_w < 1 || g->n_audio < 1)
        return RAD_E_INVAL;
    const int64_t F = g->lat_frames, H = g->lat_h, W = g->lat_w, S = F * H * W, A = g->n_audio;
    const int64_t rows = g->n_text + A + S;
    if (S > max_video_rows()) return RAD_E_INVAL;
    if (!pos) return rows;
    if (cap < rows) return RAD_E_INVAL;
    float* pt = pos;
    float* py = pos + S;
    float* px = pos + 2 * S;
    float* pa = pos + 3 * S;
    int64_t r = 0;
    for (int64_t f = 0; f < F; ++f) {
        const float t0 = std::max(8.0f * (float)f + 1.0f - 8.0f, 0.0f) / (float)kFps;
        const float t1 = std::max(8.0f * (float)(f + 1) + 1.0f - 8.0f, 0.0f) / (float)kFps;
        const float tm = (t0 + t1) / 2.0f;
        for (int64_t h = 0; h < H; ++h)
            for (int64_t w = 0; w < W; ++w, ++r) {
                pt[r] = tm;
                py[r] = (32.0f * (float)h + 32.0f * (float)(h + 1)) / 2.0f;
                px[r] = (32.0f * (float)w + 32.0f * (float)(w + 1)) / 2.0f;
            }
    }
    for (int64_t f = 0; f < A; ++f) {
        const float m0 = std::max(4.0f * (float)f + 1.0f - 4.0f, 0.0f);
        const float m1 = std::max(4.0f * (float)(f + 1) + 1.0f - 4.0f, 0.0f);
        const float s0 = m0 * (float)kMelHop / (float)kVocRate;
        const float s1 = m1 * (float)kMelHop / (float)kVocRate;
        pa[f] = (s0 + s1) / 2.0f;
    }
    return rows;
}

/* ================================================================== linears */
/* An activation a linear reads: its bf16 plane and, for block-fp8 or rotated int8 linears, its
 * codes, which a quantiser makes from the plane (QuantFP8: E4M3, or had_quant_act_i8g's int8 at
 * the weights' scale group). Rows narrower than that group carry no codes: a bf16 linear reads
 * them (the latents' projections in). Where no library serves the rotating quantiser, each int8
 * linear quantises its own input and this declares none. */
struct Act {
    ActFP8   a{};
    QuantFP8 q{};
    int64_t  n = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& g, const char* name, int64_t rows,
                int64_t width) {
        n = width;
        a.x = decl_b(b, nm.f("%s", name), RAD_BF16, { rows, width });
        if (!a.x) return RAD_E_INVAL;
        Geom gg = g;
        gg.max_tok = rows;
        if (g.w_i8h && width % g.w_i8_group) return RAD_OK;
        RAD_ARCH_TRY(a.declare_pair(b, nm, gg, name, width, rows, g.w_i8h, g.w_i8_group,
                                    "had_quant_act_i8g"));
        if (g.w_i8h && !a.q8_fed) return RAD_OK;
        return q.declare(b, gg, a, width);
    }
    /* The codes of rows [r0, r0 + rows), at those rows' count. */
    void step(RadCtx* c, int64_t rows, int64_t r0 = 0) const {
        if (!q.op || rows <= 0) return;
        RAD_ISSUE_N(c, q.op, rows, brow_slice(a.x, r0, rows, n), brow_slice(a.cq(), r0, rows, n),
                    brow_slice(a.cs(), r0, rows, n / a.cg()));
    }
};

/* A linear, y = x W^T (+ b): `gemm_nt_bias` over the bf16 plane for a bf16 weight (the bias in the
 * GEMM's one rounding, as torch's), or `gemm_nt_q` over the codes for a block-fp8 or rotated int8
 * one with the bias added after (`table_add`, one more rounding). `small` keeps a weight bf16 in
 * any container (a gate's logits: too few rows for 128-row fp8 blocks). The input and output rows
 * may sit at different offsets of their buffers. */
struct Lin {
    LinearFP8  l{};
    rad_weight w = 0, bias = 0;
    rad_op     op = 0, op_b = 0;
    bool       fp8 = false;      /* read through its codes: block fp8, or rotated int8 */
    bool       mapped = false;   /* the weight's name map was made already (a probe's) */
    int64_t    N = 0, K = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                std::initializer_list<const char*> wsrc, std::initializer_list<const char*> bsrc,
                std::initializer_list<int64_t> parts, int64_t n_out, int64_t n_in, int access,
                int group, const Act& in, rad_buf y, int64_t max_rows, bool small = false) {
        N = n_out;
        K = n_in;
        const char* wn = nm.f("%s.weight", base);
        const char* bn = nm.f("%s.bias", base);
        if (bsrc.size()) {
            if (bsrc.size() == 1) RAD_ARCH_TRY(map_copy(b, bn, *bsrc.begin()));
            else RAD_ARCH_TRY(map_concat(b, bn, bsrc));
            bias = decl_w(b, bn, RAD_BF16, { N }, access, RAD_SHARD_NONE, grp_layer(group));
            if (!bias) return RAD_E_INVAL;
        }
        fp8 = !(g.w_bf16 || small);
        if (!fp8) {
            if (mapped) {}
            else if (wsrc.size() == 1) RAD_ARCH_TRY(map_copy(b, wn, *wsrc.begin()));
            else RAD_ARCH_TRY(map_concat(b, wn, wsrc));
            w = decl_w(b, wn, RAD_BF16, { N, K }, access, RAD_SHARD_NONE, grp_layer(group));
            if (!w) return RAD_E_INVAL;
            op = bias ? rw(b, RAD_OP(b, "gemm_nt_bias",
                                     RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("N", N),
                                                RAD_INT("K", K), RAD_STR("dtype", "bf16")),
                                     RAD_WEIGHTS(w, bias)),
                           { in.a.x }, { y })
                      : rw(b, RAD_OP(b, "gemm_nt",
                                     RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("N", N),
                                                RAD_INT("K", K), RAD_STR("dtype", "bf16")),
                                     RAD_WEIGHTS(w)),
                           { in.a.x }, { y });
            return op ? RAD_OK : RAD_E_INVAL;
        }
        l.access = access;
        l.mapped = mapped;
        RAD_ARCH_TRY(l.declare(b, nm, g, base, N, K, RAD_SHARD_NONE, grp_layer(group), wsrc, in.a, y,
                               max_rows, parts));
        const bool other = l.bf16 || l.mx || l.w4 || l.paro || l.paro5 || l.iqr;
        if (other || (g.w_i8h ? !(l.i8 && l.i8_rot) : l.i8)) {
            fprintf(stderr, "radiance: ltx2: '%s' is not %s as the first DiT projection is; this "
                            "plugin reads bf16, block-fp8 or rotated int8 linears, one kind a "
                            "container\n", base, g.w_i8h ? "rotated int8" : "block fp8");
            return RAD_E_UNSUPPORTED;
        }
        op = l.op;
        if (bias)
            op_b = rw(b, RAD_OP(b, "table_add",
                                RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("n", N),
                                           RAD_INT("L", 1), RAD_STR("dtype", "bf16")),
                                RAD_WEIGHTS(bias)),
                      { y }, { y });
        return op && (!bias || op_b) ? RAD_OK : RAD_E_INVAL;
    }

    /* Input rows [ri, ri + rows) to output rows [ro, ro + rows). */
    void step(RadCtx* c, const Act& in, rad_buf y, int64_t rows, int64_t ri = 0,
              int64_t ro = 0) const {
        if (rows <= 0) return;
        const RadOperand out = brow_slice(y, ro, rows, N);
        if (!fp8) {
            if (bias)
                RAD_ISSUE_N(c, op, rows, brow_slice(in.a.x, ri, rows, K), RAD_W(w), RAD_W(bias),
                            RAD_NONE, out);
            else
                RAD_ISSUE_N(c, op, rows, brow_slice(in.a.x, ri, rows, K), RAD_W(w), out);
            return;
        }
        if (l.i8)
            l.step(c, in.a, y, ri + rows, ri, rows, ro);
        else
            RAD_ISSUE_N(c, op, rows, brow_slice(in.a.q, ri, rows, K),
                        brow_slice(in.a.s, ri, rows, K / RAD_FP8_BLOCK), RAD_W(l.w), RAD_W(l.ws),
                        out, RAD_NONE, RAD_NONE);
        if (op_b) RAD_ISSUE_N(c, op_b, rows, out, RAD_W(bias), out);
    }
};

/* A weight copied as the checkpoint holds it, read whole. */
static rad_weight wcopy(RadBuilder* b, const char* dn, const char* src, uint32_t dtype,
                        std::initializer_list<int64_t> shape, int access, int group) {
    if (map_copy(b, dn, src) < 0) return 0;
    return decl_w(b, dn, dtype, shape, access, RAD_SHARD_NONE, grp_layer(group));
}

static rad_op unary(RadBuilder* b, const char* op, int64_t rows, int64_t n, const char* dtype,
                    rad_buf in, rad_buf out) {
    return rw(b, RAD_OP(b, op, RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                          RAD_STR("dtype", dtype)),
                        RAD_NOWEIGHTS),
              { in }, { out });
}

static rad_op add_op(RadBuilder* b, int64_t rows, int64_t n, const char* dtype, rad_buf a, rad_buf c,
                     rad_buf y) {
    return rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", dtype)),
                        RAD_NOWEIGHTS),
              { a, c }, { y });
}

static rad_op cast_op(RadBuilder* b, int64_t rows, int64_t n, const char* from, const char* to,
                      rad_buf x, rad_buf y) {
    return rw(b, RAD_OP(b, "cast",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("from", from),
                                   RAD_STR("to", to)),
                        RAD_NOWEIGHTS),
              { x }, { y });
}

/* An RMS norm over n with an optional gain (none: a gain of one), bf16. */
static rad_op rms_op(RadBuilder* b, int64_t rows, int64_t n, double eps, rad_weight w, rad_buf x,
                     rad_buf y) {
    return rw(b, w ? RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w))
                   : RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
              { x }, { y });
}

/* `norm(x) (1 + scale) + shift`, the shift and scale one row of the table `st` per entry of
 * `cu_seqlens`; with `gated` the branch before it gated in first (x += g y, g a row of `gt`).
 * `norm` rms, layer or none, without a gain. */
static rad_op mod_op(RadBuilder* b, int64_t rows, int64_t n, double eps, bool gated,
                     const char* norm, rad_buf st, rad_buf gt, rad_buf x, rad_buf y, rad_buf z) {
    rad_op op = RAD_OP(b, "rms_mod",
                       RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                  RAD_INT("gated", gated ? 1 : 0), RAD_STR("dtype", "bf16"),
                                  RAD_STR("norm", norm)),
                       RAD_NOWEIGHTS);
    if (!op) return 0;
    rad_buf rd[4] = { st, x, gt, y };
    rad_buf wr[2] = { x, z };
    rad_op_reads(b, op, rd, gated ? 4 : 2);
    rad_op_writes(b, op, wr, 2);
    return op;
}

/* x += g y, the gate one row per entry of `cu_seqlens`, no activation. */
static rad_op gate_op(RadBuilder* b, int64_t rows, int64_t n, rad_buf table, rad_buf y, rad_buf x) {
    return rw(b, RAD_OP(b, "gate_add",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                   RAD_STR("dtype", "bf16"), RAD_STR("act", "none")),
                        RAD_NOWEIGHTS),
              { y, table, x }, { x });
}

/* A space-separated list off the container (rad-convert flattens arrays that way). */
static std::vector<double> meta_list(const RadModelMeta* m, const char* key) {
    std::vector<double> v;
    const char* s = rad_meta_gets(m, key, nullptr);
    while (s && *s) {
        char* e = nullptr;
        const double x = std::strtod(s, &e);
        if (e == s) { ++s; continue; }
        v.push_back(x);
        s = e;
    }
    return v;
}

static std::string ints(std::initializer_list<long long> v) {
    std::string s;
    for (long long x : v) {
        char t[24];
        std::snprintf(t, sizeof t, "%s%lld", s.empty() ? "" : " ", x);
        s += t;
    }
    return s;
}

/* ================================================================== the model's parts */
/* A text encoder layer: sliding or full attention, each class its own head geometry and rotary. */
struct TeLayer {
    bool       full = false, v_from_k = false;
    int64_t    nh = 0, nkv = 0, hd = 0, rot = 0;
    double     theta = 0.0;
    rad_weight w_in = 0, w_pa = 0, w_pf = 0, w_ff = 0, w_qn = 0, w_kn = 0, w_sc = 0;
    Lin        q{}, k{}, v{}, o{}, gu{}, down{};
    rad_op     op_in = 0, op_vn = 0, op_qn = 0, op_kn = 0, op_rq = 0, op_rk = 0, op_attn = 0;
    rad_op     op_pa = 0, op_add1 = 0, op_pf = 0, op_act = 0, op_ff = 0, op_add2 = 0, op_sc = 0;
    rad_op     op_keep = 0;
};
/* An attention class's rows: q, k, v and the attention's output. */
struct TeClass {
    rad_buf q = 0, k = 0, v = 0;
    Act     o{};
};

/* An attention: q (k and v from the same rows for a self-attention), the per-head gate's logits, q
 * and k RMS-normed across every head, the output projection. A cross-attention's k and v come from
 * the context's rows. */
struct Attn {
    int64_t    H = 0, hd = 0, inner = 0;
    rad_weight w_qn = 0, w_kn = 0;
    Lin        qkv{}, kv{}, gate{}, out{};
    rad_op     op_qn = 0, op_kn = 0, op_rq = 0, op_rk = 0, op_attn = 0, op_hg = 0, op_v = 0;
};

struct ConnBlock {
    Attn   at{};
    Lin    up{}, down{};
    rad_op op_n1 = 0, op_add1 = 0, op_n2 = 0, op_act = 0, op_add2 = 0;
};
/* A modality's connector: the stack's projection, the registers and the blocks. */
struct Conn {
    int64_t    D = 0, H = 0, hd = 0, ff = 0;
    double     mult = 1.0;          /* sqrt(D / text width), the normed stack's scale */
    rad_weight w_reg = 0;           /* the registers, tiled to kTextRows rows */
    Lin        proj{};
    rad_op     op_norm = 0, op_reg = 0, op_nout = 0;
    Act        a_in{}, a_h{}, a_attn{}, a_ffn{};
    rad_buf    b_x = 0, b_qkv = 0, b_g = 0, b_o = 0, b_up = 0;
    rad_buf    b_text = 0;          /* PERSIST [lanes * 2 * kTextRows, D]: a lane's two prompts */
    std::vector<ConnBlock> blk;
};

/* A DiT block. The modulation tables are each block's learned rows plus the step's shared ones,
 * made once a step into the step's tables (DitStep). */
struct DitBlock {
    Attn       sv{}, sa{}, tv{}, ta{}, a2v{}, v2a{};
    Lin        fv_up{}, fv_down{}, fa_up{}, fa_down{};
    rad_weight t_v = 0, t_a = 0, t_p = 0, t_pa = 0, t_cv = 0, t_cvg = 0, t_ca = 0, t_cag = 0;
    rad_op     op_tv = 0, op_ta = 0, op_tp = 0, op_tpa = 0, op_tcv = 0, op_tcvg = 0, op_tca = 0;
    rad_op     op_tcag = 0;
    /* Video: the self-attention's norm (gated by the previous block's feed-forward), the text
     * query's (gated by the self-attention), the text's modulation, the text gate, the two
     * cross-modal modulations, the cross-modal gate, the feed-forward's norm ungated and gated by
     * the text (a pass without the cross-modal attentions), the activation. Audio the same. */
    rad_op     m1v = 0, m2v = 0, mtv = 0, g2v = 0, mav = 0, mvv = 0, gcv = 0, m3v = 0, m3vg = 0;
    rad_op     actv = 0;
    rad_op     m1a = 0, m2a = 0, mta = 0, g2a = 0, maa = 0, mva = 0, gca = 0, m3a = 0, m3ag = 0;
    rad_op     acta = 0;
};

/* One AdaLayerNormSingle: the timestep's MLP to `emb`, then SiLU and a linear to `temb`. */
struct Ada {
    int64_t    d = 0, k = 0;
    rad_weight w1 = 0, b1 = 0, w2 = 0, b2 = 0, wl = 0, bl = 0;
    rad_op     op1 = 0, ops1 = 0, op2 = 0, ops2 = 0, opl = 0;
    rad_buf    h = 0, hs = 0, emb = 0, es = 0, temb = 0;
};

/* A 3D residual block of the video VAE: PerChannelRMSNorm and SiLU, a convolution, again, the
 * residual added. */
struct VRes {
    int64_t    C = 0;
    rad_weight w1 = 0, b1 = 0, w2 = 0, b2 = 0;
    rad_op     op_n1 = 0, op_c1 = 0, op_n2 = 0, op_c2 = 0;
};
/* An up block: the upsampler's convolution and depth to space, then its residual blocks. */
struct VUp {
    int64_t    cin = 0, cout = 0, st = 1, sh = 1, sw = 1, conv_out = 0;
    rad_weight w = 0, b = 0;
    rad_op     op_conv = 0, op_d2s = 0, op_geom = 0;
    rad_buf    geom = 0;            /* its output volume, i32 [3] */
    std::vector<VRes> res;
};
/* A down block: its residual blocks, then the downsampler's convolution, space to depth, and the
 * input's group mean added. */
struct VDown {
    int64_t    cin = 0, cout = 0, st = 1, sh = 1, sw = 1, conv_out = 0, group = 1;
    std::vector<VRes> res;
    rad_weight w = 0, b = 0;
    rad_op     op_conv = 0, op_mean = 0, op_s2d = 0, op_geom = 0, op_geom_in = 0, op_next = 0;
    rad_buf    geom = 0, geom_in = 0;   /* its convolution's volume, and its input's */
};

/* An audio VAE residual block (2D, causal along time): PixelNorm and SiLU, conv, again, the
 * input (or its 1x1 shortcut) added. */
struct ARes {
    int64_t    cin = 0, cout = 0;
    rad_weight w1 = 0, b1 = 0, w2 = 0, b2 = 0, ws = 0, bs = 0;
    rad_op     op_n1 = 0, op_c1 = 0, op_n2 = 0, op_c2 = 0, op_sc = 0, op_add = 0;
};

/* A vocoder convolution and an anti-aliased SnakeBeta (f32). */
struct VConv {
    rad_weight w = 0, b = 0;
    rad_op     op = 0;
};
struct VAct {
    rad_weight alpha = 0, beta = 0, up = 0, down = 0;
    rad_op     op = 0;
};
struct VUnit {
    VAct  a1{}, a2{};
    VConv c1{}, c2{};
};
struct VStage {
    int64_t ch = 0, rate = 0, k = 0;
    VConv   up{};
    VUnit   u[3][3];               /* resnet j, dilation d */
    rad_op  op_cu = 0, op_sum = 0, op_avg = 0;
};
/* A vocoder (the base one or the bandwidth extender's). */
struct Voc {
    int64_t   in = 0, hidden = 0, out = 0, mult = 1;
    std::vector<int64_t> rates, ks, rk;
    VConv     cin{}, cout{};
    VAct      act{};
    std::vector<VStage> st;
    rad_op    op_cu0 = 0;
    rad_buf   in_buf = 0, wav = 0;  /* its input [rows, 128] and output [rows, 2], f32 */
    std::vector<rad_buf> cu, h, avg, t1, t2, amp[3];
};

/* A latent upsampler residual block: conv, GroupNorm, SiLU, conv, GroupNorm, the residual added,
 * SiLU. */
struct URes {
    rad_weight c1w = 0, c1b = 0, n1w = 0, n1b = 0, c2w = 0, c2b = 0, n2w = 0, n2b = 0;
    rad_op     op_c1 = 0, op_n1 = 0, op_c2 = 0, op_n2 = 0, op_add = 0, op_act = 0;
};

/* The diffusion decoder's neighbourhood-attention block: a deterministic stage's (pre-norm) or a
 * diffusion block's (the latent context added, its norms modulated by the shared table plus its
 * own). */
struct DdBlock {
    int64_t    C = 0, ff = 0, kt = 0, kh = 0, kw = 0, chunk = 0;
    int        slot = 0;          /* the q|k|v and gate|up buffers of its width */
    rad_weight w_n1 = 0, w_n2 = 0, w_nq = 0, w_nk = 0, w_o = 0, b_o = 0, w_gu = 0, w_down = 0;
    rad_weight w_ctx = 0, b_ctx = 0, w_sst = 0;
    rad_op     op_n1 = 0, op_rope = 0, op_attn = 0, op_o = 0;
    rad_op     op_n2 = 0, op_gu = 0, op_act = 0, op_down = 0, op_add2 = 0;
    rad_op     op_ctx = 0, op_mod = 0;
    /* q|k|v of a group of heads as one weight (the three projections' rows of the group), and its
     * gemm into the q|k|v buffer: every head in one group, or in the diffusion stage one a group. */
    std::vector<rad_weight> w_qkvh, b_qkvh;
    std::vector<rad_op>     op_qkvh;
};
/* A deterministic stage: its blocks, then its pixel-shuffle upsample (a linear, then depth to
 * space; the last stage's keeps the duplicate leading frame on a tile that does not start the
 * video). */
struct DdStage {
    int64_t    C = 0, cout = 0, st = 1, sh = 1, sw = 1;
    std::vector<DdBlock> blk;
    rad_weight w_up = 0, b_up = 0;
    rad_op     op_proj = 0, op_d2s = 0, op_d2s_keep = 0, op_geom = 0, op_pos = 0;
    rad_buf    geom = 0;          /* PERSIST: the stage's volume (its input grid) */
};

/* Weight groups (movement units) by network. */
enum : int {
    kGrpConn  = 1000,
    kGrpTemb  = 1100,
    kGrpDit   = 2000,
    kGrpVae   = 3000,
    kGrpVenc  = 3100,
    kGrpAudio = 3200,
    kGrpVoc   = 3300,
    kGrpUps   = 3400,
    kGrpDur   = 1200,
    kGrpDdec  = 3500,
};

struct Model {
    Names nm{""};
    Geom  gt{}, gd{};                /* text encoder and connectors (prompt rows), DiT (step rows) */
    int64_t maxS = 0, maxA = 0, maxPpf = 0, maxVox = 0;
    std::vector<double> rungs, refine;

    /* ---- the text encoder: Gemma 4's language model, every layer's output kept */
    int64_t tC = 0, tL = 0, tFF = 0, tRows = 0;
    double  teps = 1e-6;
    rad_weight w_tok = 0, w_tnorm = 0;
    rad_op  op_embed = 0, op_keep0 = 0, op_tnorm = 0;
    rad_buf b_tx = 0, b_th2 = 0, b_tgu = 0, b_stack = 0;
    Act     a_th{}, a_tffn{};
    TeClass cls[2];                  /* sliding, full */
    std::vector<TeLayer> te;

    /* ---- the connectors */
    Conn    cv{}, ca{};
    rad_buf b_rows = 0;              /* i32 {0, kTextRows, 2 kTextRows} */
    rad_buf b_rpos = 0;              /* i32 [2 kTextRows]: a connector row's index in its prompt */
    rad_buf b_zero = 0;              /* bf16 zeros, one row of the widest connector */
    rad_op  op_rows = 0, op_rpos = 0, op_zero = 0;

    /* ---- the duration head: the length the prompt implies, off both connectors' output */
    bool    dur = false;
    int64_t dP = 0;                  /* the pooler's width */
    rad_weight dw_vp = 0, dw_vpb = 0, dw_vemb = 0, dw_ap = 0, dw_apb = 0, dw_aemb = 0, dw_qt = 0;
    rad_weight dw_q = 0, dw_qb = 0, dw_k = 0, dw_kb = 0, dw_v = 0, dw_vb = 0, dw_o = 0, dw_ob = 0;
    rad_weight dw_h = 0, dw_hb = 0, dw_out = 0, dw_outb = 0;
    rad_op  op_dvp = 0, op_dap = 0, op_dvemb = 0, op_daemb = 0, op_dq0 = 0, op_dq = 0, op_dk = 0;
    rad_op  op_dv = 0, op_dcu = 0, op_dcuk = 0, op_dattn = 0, op_do = 0, op_dh = 0, op_dout = 0;
    rad_op  op_dlog = 0;
    rad_buf b_dtok = 0, b_dk = 0, b_dv = 0, b_dq0 = 0, b_dq = 0, b_dat = 0, b_do = 0, b_dh = 0;
    rad_buf b_dl = 0, b_dcu = 0, b_dcuk = 0, b_dur = 0;

    /* ---- the DiT */
    int64_t D = 0, Da = 0, H = 0, Ha = 0, hd = 0, hda = 0, Dc = 0, ffv = 0, ffa = 0, nb = 0;
    double  eps = 1e-6;
    /* the step's timestep inputs {0, sigma}, their sinusoids, the eight modulation MLPs */
    rad_buf b_tin = 0, b_tsin = 0;
    rad_op  op_tin0 = 0, op_tin1 = 0, op_tsin = 0;
    Ada     ad_v{}, ad_a{}, ad_p{}, ad_pa{}, ad_cv{}, ad_cvg{}, ad_ca{}, ad_cag{};
    /* the step's tables (PERSIST): every block's rows, then the output norms' */
    rad_buf b_mv = 0, b_ma = 0, b_mp = 0, b_mpa = 0, b_mcv = 0, b_mcvg = 0, b_mca = 0, b_mcag = 0;
    rad_buf b_fv = 0, b_fa = 0;
    rad_weight w_fv = 0, w_fa = 0;
    rad_op  op_fv = 0, op_fa = 0;
    /* the lanes (PERSIST): both latents and the clean first latent frame */
    rad_buf b_vstate = 0, b_astate = 0, b_first = 0;
    rad_op  op_rnd_v = 0, op_rnd_a = 0, op_keep_first = 0;
    /* an evaluation */
    Act     a_lv{}, a_la{};
    Lin     pin_v{}, pin_a{}, pout_v{}, pout_a{};
    rad_op  op_lv = 0, op_la = 0;
    rad_buf b_hv = 0, b_ha = 0, b_qkvv = 0, b_qkva = 0, b_gv = 0, b_ga = 0, b_ov = 0, b_oa = 0;
    rad_buf b_qv = 0, b_qa = 0, b_kvtv = 0, b_kvta = 0;
    rad_buf b_qxv = 0, b_qxa = 0, b_kvxv = 0, b_kvxa = 0, b_gxv = 0, b_gxa = 0;
    rad_buf b_upv = 0, b_upa = 0;
    Act     a_hv{}, a_ha{}, a_ctxv{}, a_ctxa{}, a_nv{}, a_nv2{}, a_na{}, a_na2{};
    Act     a_aov{}, a_aoa{}, a_xv{}, a_xa{}, a_ffv{}, a_ffa{}, a_fv{}, a_fa{};
    std::vector<DitBlock> dit;
    rad_op  op_fnv = 0, op_fna = 0;
    rad_buf b_vel_v[kPasses] = {}, b_vel_a[kPasses] = {};   /* PERSIST: each evaluation's */
    rad_buf b_gst = 0;
    rad_op  op_gv = 0, op_ga = 0;

    /* ---- the video VAE's decoder: the whole video, a tile at a time, into the lane's tile
     * store; each chunk's frames stitched to bytes. A latent frame f is pixel frames
     * vAt f + vBt; a latent vAhw patches a side. */
    int64_t vC0 = 0, vAt = 1, vBt = 0, vAhw = 1, maxTiles = 0, maxStore = 0, maxSide = 0;
    rad_weight vd_mean = 0, vd_std = 0, w_vcin = 0, b_vcin = 0, w_vcout = 0, b_vcout = 0;
    rad_op  op_vlat = 0, op_vden = 0, op_vcin = 0, op_vnout = 0, op_vcout = 0;
    rad_op  op_vgrid1 = 0, op_vgrid = 0, op_vcrop = 0, op_vf0 = 0, op_vst1 = 0, op_vst = 0;
    std::vector<VRes> vmid;
    std::vector<VUp>  vup;
    rad_buf b_vlat = 0, b_vtlat = 0, b_va = 0, b_vb = 0, b_vc = 0, b_vtiles = 0, b_frames = 0;
    rad_buf b_vgrid = 0, b_vf0 = 0;

    /* ---- its encoder: a first frame into the lane's clean first latent frame */
    bool    venc = false;
    int64_t eC0 = 0, eCout = 0;
    rad_weight w_ecin = 0, b_ecin = 0, w_ecout = 0, b_ecout = 0;
    rad_op  op_epatch = 0, op_ecin = 0, op_enout = 0, op_ecout = 0, op_enorm = 0, op_ef32 = 0;
    rad_op  op_egeomL = 0;
    std::vector<VDown> vdown;
    std::vector<VRes>  emid;
    rad_buf b_egeomL = 0, b_epatch = 0, b_ea = 0, b_eb = 0, b_ec = 0, b_eout = 0, b_elat = 0;

    /* ---- the audio VAE's decoder and the vocoders */
    int64_t aC0 = 0;
    std::string a_mean, a_std;
    rad_op  op_aden = 0, op_acast = 0, op_aunpack = 0, op_acin = 0, op_anout = 0;
    rad_op  op_acout[2] = {}, op_amel = 0;
    rad_weight w_acin = 0, b_acin = 0, w_acout[2] = {}, b_acout[2] = {}, w_amean = 0, w_astd = 0;
    ARes    amid[2];
    std::vector<std::vector<ARes>> aup;          /* level 2, 1, 0 */
    std::vector<VConv> aups;                     /* the upsamples after levels 2 and 1 */
    std::vector<rad_op> aups_cp;                 /* each upsample's rows past the first, back */
    std::vector<int64_t> alev_c;
    rad_buf b_aden = 0, b_alat = 0, b_aimg = 0, b_aa = 0, b_ab = 0, b_ac = 0, b_as = 0;
    rad_buf b_amel16 = 0, b_amel = 0;
    Voc     voc{}, bwe{};
    rad_weight w_basis = 0, w_melb = 0;
    rad_op  op_stft = 0, op_bwe = 0, op_bmel = 0, op_tail = 0;
    rad_buf b_bmel16 = 0, b_bmel = 0, b_audio = 0;

    /* ---- the latent upsampler */
    bool    ups = false;
    int64_t uC = 0;
    rad_weight u_cinw = 0, u_cinb = 0, u_ninw = 0, u_ninb = 0, u_upw = 0, u_upb = 0;
    rad_weight u_coutw = 0, u_coutb = 0;
    rad_op  op_uden = 0, op_ucast = 0, op_ucin = 0, op_unin = 0, op_uup = 0, op_ud2s = 0;
    rad_op  op_ucout = 0, op_unorm = 0, op_uf32 = 0, op_ukeep = 0, op_ufirst = 0;
    rad_op  op_urnv = 0, op_urna = 0;
    std::vector<URes> u_pre, u_post;
    rad_buf b_ua = 0, b_ub = 0, b_uc = 0, b_ust = 0, b_uin = 0, b_uout = 0, b_uden = 0;

    /* ---- the diffusion decoder (diffusion_decoder/): deterministic stages 0..2 over the whole
     * latent into the feature volume, then per tile stage 3 and the one-step diffusion stage into
     * the tile store */
    bool    ddec = false;
    std::vector<DdStage> dst;        /* the four deterministic stages */
    std::vector<DdBlock> dblk;       /* the diffusion blocks */
    int64_t dV0 = 0, dV1 = 0, dV2 = 0, dV3 = 0, dV4 = 0, dV5 = 0, dTiles = 0;
    rad_weight dw_cin = 0, db_cin = 0, dw_xin = 0, db_xin = 0, dw_nout = 0, dw_cout = 0, db_cout = 0;
    rad_weight dw_t1 = 0, db_t1 = 0, dw_t2 = 0, db_t2 = 0, dw_ad = 0, db_ad = 0;
    rad_op  op_dpad = 0, op_dcin = 0, op_dtin = 0, op_dtsin = 0, op_dt1 = 0, op_dts = 0, op_dt2 = 0;
    rad_op  op_das = 0, op_dad = 0, op_ddcu = 0, op_dsig = 0, op_dtgeom = 0, op_dgrid = 0;
    rad_op  op_dcrop = 0, op_dpos4 = 0, op_dg5[4] = {}, op_dpos5 = 0, op_dz = 0, op_dxcast = 0;
    rad_op  op_dxin = 0, op_dnout = 0, op_ddout = 0, op_dst = 0;
    std::vector<rad_op> op_dnoise;   /* each tile's noise: its own stream */
    rad_buf b_dz = 0, b_dx = 0, b_dfeat = 0, b_d4 = 0, b_ddh = 0, b_dctx = 0, b_dn = 0;
    rad_buf b_dqkv[4] = {}, b_dg[4] = {};   /* per width 2048, 1024, 512, 256: q|k|v, gate|up */
    rad_buf b_dao = 0;                      /* the diffusion stage's attention, its heads side by side */
    rad_buf b_da = 0, b_dmo = 0, b_dpos = 0, b_dxt = 0, b_dxb = 0, b_dtin = 0;
    rad_buf b_dtsin = 0, b_dth = 0, b_dths = 0, b_dtemb = 0, b_dtes = 0, b_dmod = 0, b_dbmod = 0;
    rad_buf b_ddcu = 0, b_dsig = 0, b_dtg = 0, b_dgrid = 0, b_dg5 = 0;
};

static Model g_model[MAX_SLOTS];

/* ================================================================== check_format */
static long long tgeti(const RadModelMeta* m, const char* key, long long def) {
    char k[160];
    std::snprintf(k, sizeof k, "text_config.%s", key);
    return rad_meta_geti(m, k, rad_meta_geti(m, key, def));
}
static double tgetf(const RadModelMeta* m, const char* key, double def) {
    char k[160];
    std::snprintf(k, sizeof k, "text_config.%s", key);
    return rad_meta_getf(m, k, rad_meta_getf(m, key, def));
}
static const char* tgets(const RadModelMeta* m, const char* key, const char* def) {
    char k[160];
    std::snprintf(k, sizeof k, "text_config.%s", key);
    return rad_meta_gets(m, k, rad_meta_gets(m, key, def));
}

/* What this plugin runs: LTX-2.5's components (LTX-2.3's are the same graph). Anything else is
 * refused by name rather than run wrong. */
static int check_format(const RadModelMeta* meta) {
    const char* cls = rad_meta_gets(meta, "pipeline._class_name", "");
    if (std::strcmp(cls, "LTX2Pipeline") != 0 && std::strcmp(cls, "LTX2ImageToVideoPipeline") != 0) {
        fprintf(stderr, "radiance: ltx2: the container's pipeline is '%s', not LTX2Pipeline\n", cls);
        return RAD_E_UNSUPPORTED;
    }
    auto is = [&](const char* k, long long v) { return rad_meta_geti(meta, k, -9999) == v; };
    auto str = [&](const char* k, const char* v) {
        return !std::strcmp(rad_meta_gets(meta, k, ""), v);
    };
    struct Need { bool ok; const char* what; };
    const Need need[] = {
        { is("transformer.patch_size", 1) && is("transformer.patch_size_t", 1) &&
          is("transformer.in_channels", kC) && is("transformer.out_channels", kC) &&
          is("transformer.audio_in_channels", kAudioC) && is("transformer.audio_patch_size", 1),
          "1 x 1 x 1 patches of 128-channel video and audio latents" },
        { str("transformer.rope_type", "split") && is("transformer.gated_attn", 1) &&
          is("transformer.audio_gated_attn", 1) && is("transformer.cross_attn_mod", 1) &&
          is("transformer.audio_cross_attn_mod", 1) && is("transformer.perturbed_attn", 1) &&
          is("transformer.use_prompt_embeddings", 0) && is("transformer.use_prompt_adaln_single", 1),
          "the 2.3 / 2.5 DiT (split rotary, gated attention, prompt modulation)" },
        { is("transformer.causal_offset", 1) && is("transformer.pos_embed_max_pos", 20) &&
          is("transformer.audio_pos_embed_max_pos", 20) && is("transformer.base_height", 2048) &&
          is("transformer.base_width", 2048) && is("transformer.audio_scale_factor", 4) &&
          is("transformer.audio_hop_length", kMelHop) &&
          is("transformer.audio_sampling_rate", kVocRate) &&
          str("transformer.vae_scale_factors", "8 32 32") &&
          is("transformer.timestep_scale_multiplier", 1000) &&
          is("transformer.cross_attn_timestep_scale_multiplier", 1000),
          "the released rotary coordinates and timestep scales" },
        { is("connectors.per_modality_projections", 1) && str("connectors.rope_type", "split") &&
          is("connectors.video_gated_attn", 1) && is("connectors.audio_gated_attn", 1) &&
          is("connectors.connector_rope_base_seq_len", 4096),
          "per-modality connectors with gated attention and a split rotary" },
        { is("vae.patch_size", kPix) && is("vae.patch_size_t", 1) && is("vae.latent_channels", kC) &&
          is("vae.timestep_conditioning", 0) && is("vae.decoder_causal", 0) &&
          is("vae.encoder_causal", 1) && str("vae.upsample_residual", "0 0 0 0") &&
          str("vae.decoder_inject_noise", "0 0 0 0 0") &&
          str("vae.decoder_spatial_padding_mode", "zeros") &&
          rad_meta_getf(meta, "vae.scaling_factor", 1.0) == 1.0,
          "the 2.3 / 2.5 video VAE (4 x 4 patches, no timestep conditioning)" },
        { str("audio_vae.norm_type", "pixel") && str("audio_vae.causality_axis", "height") &&
          is("audio_vae.latent_channels", kLatCh) && is("audio_vae.mel_bins", kMel) &&
          is("audio_vae.mid_block_add_attention", 0) && is("audio_vae.output_channels", 2) &&
          is("audio_vae.base_channels", kAudioC) &&
          !rad_meta_gets(meta, "audio_vae.attn_resolutions", nullptr),
          "the audio VAE (pixel norms, causal in time, 8 x 16 latents, no attention)" },
        { str("vocoder.act_fn", "snakebeta") && is("vocoder.antialias", 1) &&
          is("vocoder.antialias_ratio", 2) && is("vocoder.antialias_kernel_size", 12) &&
          is("vocoder.final_bias", 0) && !rad_meta_gets(meta, "vocoder.final_act_fn", nullptr) &&
          str("vocoder.bwe_act_fn", "snakebeta") && is("vocoder.bwe_antialias", 1) &&
          is("vocoder.bwe_final_bias", 0) && !rad_meta_gets(meta, "vocoder.bwe_final_act_fn", nullptr) &&
          is("vocoder.filter_length", 512) && is("vocoder.hop_length", 80) &&
          is("vocoder.window_length", 512) && is("vocoder.num_mel_channels", kMel) &&
          is("vocoder.input_sampling_rate", kVocRate) &&
          is("vocoder.output_sampling_rate", kRate) && is("vocoder.in_channels", 2 * kMel) &&
          is("vocoder.out_channels", 2) && is("vocoder.bwe_in_channels", 2 * kMel) &&
          is("vocoder.bwe_out_channels", 2),
          "the vocoder with bandwidth extension (SnakeBeta, 16 to 48 kHz)" },
    };
    for (const Need& n : need)
        if (!n.ok) {
            fprintf(stderr, "radiance: ltx2: this plugin runs %s; the container's config differs\n",
                    n.what);
            return RAD_E_UNSUPPORTED;
        }
    return RAD_OK;
}

/* ================================================================== the text encoder */
/* Gemma 4's language model over both prompts at once (one sequence each, causal), at positions
 * that end each prompt where a 1024-token left-padded sequence does (RadVideoGenDecl::
 * text_left_pad). Every output the connectors read lands in the stack, [T, C (L + 1)] read as
 * [T (L + 1), C]: the embeddings, each layer's output but the last, and the last's final norm. */
static int declare_te(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gt;
    Names& nm = m.nm;
    const int64_t T = m.tRows, C = m.tC, L = m.tL;
    const int acc = RAD_ACCESS_PER_REQUEST;
    const char* pre = "model.language_model";
    const double eps = m.teps;

    std::vector<int> full;
    RAD_ARCH_TRY(parse_layer_kinds("ltx2", tgets(meta, "layer_types", ""), { "full_attention" },
                                   "sliding_attention", &full));
    if ((int64_t)full.size() != L) {
        fprintf(stderr, "radiance: ltx2: the text encoder names %zu layer types for %lld layers\n",
                full.size(), (long long)L);
        return RAD_E_INVAL;
    }
    const int64_t nh = tgeti(meta, "num_attention_heads", 0);
    const int64_t hd_s = tgeti(meta, "head_dim", 0);
    const int64_t kv_s = tgeti(meta, "num_key_value_heads", 0);
    const int64_t hd_g0 = tgeti(meta, "global_head_dim", hd_s);
    const int64_t kv_g0 = tgeti(meta, "num_global_key_value_heads", kv_s);
    const double th_s = tgetf(meta, "rope_parameters.sliding_attention.rope_theta", 0.0);
    const double th_g = tgetf(meta, "rope_parameters.full_attention.rope_theta", 0.0);
    double prf = tgetf(meta, "rope_parameters.full_attention.partial_rotary_factor", 1.0);
    const char* gtype = tgets(meta, "rope_parameters.full_attention.rope_type", "default");
    const int64_t win = tgeti(meta, "sliding_window", 0);
    const long long k_eq_v = tgeti(meta, "attention_k_eq_v", 0);
    if (nh <= 0 || hd_s <= 0 || kv_s <= 0 || !(th_s > 0.0) || !(th_g > 0.0) ||
        tgeti(meta, "num_kv_shared_layers", 0) != 0 || tgeti(meta, "attention_bias", 0) != 0 ||
        std::strcmp(tgets(meta, "hidden_activation", ""), "gelu_pytorch_tanh") != 0 ||
        tgeti(meta, "use_double_wide_mlp", 0) != 0 ||
        (std::strcmp(gtype, "proportional") && std::strcmp(gtype, "default"))) {
        fprintf(stderr, "radiance: ltx2: the text encoder is not Gemma 4's dense language model "
                        "(no KV sharing, no biases, a tanh GELU, default or proportional rotary)\n");
        return RAD_E_UNSUPPORTED;
    }
    if (win > 0 && win < kTextRows) {
        fprintf(stderr, "radiance: ltx2: a %lld-token sliding window shorter than the %d-token "
                        "prompt; this plugin runs the window as full attention\n", (long long)win,
                (int)kTextRows);
        return RAD_E_UNSUPPORTED;
    }

    RAD_ARCH_TRY(map_copy(b, nm.f("te.token_embd.weight"), nm.ckpt("%s.embed_tokens.weight", pre)));
    m.w_tok = decl_w(b, nm.f("te.token_embd.weight"), RAD_BF16, { g.n_vocab_all, C },
                     RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    m.w_tnorm = wcopy(b, nm.f("te.norm.weight"), nm.ckpt("%s.norm.weight", pre), RAD_BF16, { C }, acc,
                      (int)L);
    m.b_tx = decl_b(b, nm.f("te.x"), RAD_BF16, { T, C });
    m.b_th2 = decl_b(b, nm.f("te.h2"), RAD_BF16, { T, C });
    m.b_tgu = decl_b(b, nm.f("te.gate_up"), RAD_BF16, { T, 2 * m.tFF });
    m.b_stack = decl_b(b, nm.f("te.stack"), RAD_BF16, { T, C * (L + 1) });
    RAD_ARCH_TRY(m.a_th.declare(b, nm, g, "te.h", T, C));
    RAD_ARCH_TRY(m.a_tffn.declare(b, nm, g, "te.ffn", T, m.tFF));
    if (!m.w_tok || !m.w_tnorm) return RAD_E_INVAL;
    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", C),
                                         RAD_INT("n_vocab", g.n_vocab_all), RAD_STR("dtype", "bf16"),
                                         RAD_INT("vocab_offset", 0),
                                         RAD_F64("mult", std::sqrt((double)C))),
                              RAD_WEIGHTS(m.w_tok)),
                    {}, { m.b_tx });
    m.op_keep0 = cast_op(b, T, C, "bf16", "bf16", m.b_tx, m.b_stack);

    /* Each class's rows at its own head geometry. */
    for (int k = 0; k < 2; ++k) {
        int64_t qd = 0, kvd = 0;
        for (int64_t l = 0; l < L; ++l) {
            if ((full[(size_t)l] != 0) != (k == 1)) continue;
            const int64_t hd = tgeti(meta, nm.f("per_layer_config.%lld.head_dim", (long long)l),
                                     k ? hd_g0 : hd_s);
            const int64_t kv = tgeti(meta, nm.f("per_layer_config.%lld.num_key_value_heads",
                                                (long long)l), k ? kv_g0 : kv_s);
            qd = std::max(qd, nh * hd);
            kvd = std::max(kvd, kv * hd);
        }
        if (!qd) continue;
        TeClass& K = m.cls[k];
        K.q = decl_b(b, nm.f("te.q%d", k), RAD_BF16, { T, qd });
        K.k = decl_b(b, nm.f("te.k%d", k), RAD_BF16, { T, kvd });
        K.v = decl_b(b, nm.f("te.v%d", k), RAD_BF16, { T, kvd });
        RAD_ARCH_TRY(K.o.declare(b, nm, g, nm.f("te.attn%d", k), T, qd));
    }

    m.te.resize((size_t)L);
    for (int64_t l = 0; l < L; ++l) {
        TeLayer& Y = m.te[(size_t)l];
        const int li = (int)l;
        Y.full = full[(size_t)l] != 0;
        Y.nh = nh;
        Y.hd = tgeti(meta, nm.f("per_layer_config.%d.head_dim", li), Y.full ? hd_g0 : hd_s);
        Y.nkv = tgeti(meta, nm.f("per_layer_config.%d.num_key_value_heads", li),
                      Y.full ? kv_g0 : kv_s);
        Y.theta = Y.full ? th_g : th_s;
        Y.rot = Y.full ? (int64_t)(Y.hd * prf + 0.5) : Y.hd;
        Y.v_from_k = Y.full && k_eq_v != 0;
        if (Y.nkv <= 0 || nh % Y.nkv || Y.rot <= 0 || Y.rot > Y.hd || (Y.rot & 1)) return RAD_E_INVAL;
        const TeClass& K = m.cls[Y.full ? 1 : 0];
        const int64_t qd = nh * Y.hd, kvd = Y.nkv * Y.hd;
        const char* lp = nm.ckpt("%s.layers.%d", pre, li);
        auto norm_w = [&](const char* what) {
            return wcopy(b, nm.f("te.blk.%d.%s.weight", li, what), nm.ckpt("%s.%s.weight", lp, what),
                         RAD_BF16, { C }, acc, li);
        };
        Y.w_in = norm_w("input_layernorm");
        Y.w_pa = norm_w("post_attention_layernorm");
        Y.w_pf = norm_w("pre_feedforward_layernorm");
        Y.w_ff = norm_w("post_feedforward_layernorm");
        Y.w_qn = wcopy(b, nm.f("te.blk.%d.q_norm.weight", li), nm.ckpt("%s.self_attn.q_norm.weight", lp),
                       RAD_BF16, { Y.hd }, acc, li);
        Y.w_kn = wcopy(b, nm.f("te.blk.%d.k_norm.weight", li), nm.ckpt("%s.self_attn.k_norm.weight", lp),
                       RAD_BF16, { Y.hd }, acc, li);
        Y.w_sc = wcopy(b, nm.f("te.blk.%d.layer_scalar", li), nm.ckpt("%s.layer_scalar", lp), RAD_BF16,
                       { 1 }, acc, li);
        if (!Y.w_in || !Y.w_pa || !Y.w_pf || !Y.w_ff || !Y.w_qn || !Y.w_kn || !Y.w_sc)
            return RAD_E_INVAL;
        Y.op_in = rms_op(b, T, C, eps, Y.w_in, m.b_tx, m.a_th.a.x);
        auto lin = [&](Lin& Z, const char* name, const char* src, int64_t N, int64_t Kd, const Act& in,
                       rad_buf y) {
            return Z.declare(b, nm, g, nm.f("te.blk.%d.%s", li, name),
                             { nm.ckpt("%s.%s.weight", lp, src) }, {}, {}, N, Kd, acc, li, in, y, T);
        };
        RAD_ARCH_TRY(lin(Y.q, "attn_q", "self_attn.q_proj", qd, C, m.a_th, K.q));
        RAD_ARCH_TRY(lin(Y.k, "attn_k", "self_attn.k_proj", kvd, C, m.a_th, K.k));
        if (!Y.v_from_k) RAD_ARCH_TRY(lin(Y.v, "attn_v", "self_attn.v_proj", kvd, C, m.a_th, K.v));
        /* v's weightless RMS norm, from k's raw rows when the class reads v from k (before k's
         * own norm rewrites them); q and k per head with their gains; then the rotaries. */
        Y.op_vn = rw(b, RAD_OP(b, "rmsnorm_p",
                               RAD_PARAMS(RAD_RANGE("M", 1, T * Y.nkv), RAD_INT("n", Y.hd),
                                          RAD_F64("eps", eps), RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { Y.v_from_k ? K.k : K.v }, { K.v });
        Y.op_qn = rms_op(b, T * nh, Y.hd, eps, Y.w_qn, K.q, K.q);
        Y.op_kn = rms_op(b, T * Y.nkv, Y.hd, eps, Y.w_kn, K.k, K.k);
        auto rope = [&](int64_t heads, rad_buf x) {
            return rw(b, RAD_OP(b, "rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", Y.hd),
                                           RAD_INT("n_head", heads), RAD_INT("n_head_kv", 0),
                                           RAD_F64("theta", Y.theta), RAD_F64("scale", 1.0),
                                           RAD_STR("mode", "neox"), RAD_INT("rotary_dim", Y.rot),
                                           RAD_STR("ladder",
                                                   Y.full && !std::strcmp(gtype, "proportional")
                                                                 ? "head" : "rotary")),
                                RAD_NOWEIGHTS),
                      { x }, { x });
        };
        Y.op_rq = rope(nh, K.q);
        Y.op_rk = rope(Y.nkv, K.k);
        /* Gemma 4 scores q.k without 1 / sqrt(head_dim): the q and k norms' gains carry it. */
        Y.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", Y.hd),
                                            RAD_INT("gqa", nh / Y.nkv), RAD_INT("causal", 1),
                                            RAD_STR("dtype", "bf16"), RAD_F64("scale", 1.0),
                                            RAD_INT("max_seqlen", kTextRows)),
                                 RAD_NOWEIGHTS),
                       { K.q, K.k, K.v }, { K.o.a.x });
        RAD_ARCH_TRY(lin(Y.o, "attn_output", "self_attn.o_proj", C, qd, K.o, m.b_th2));
        Y.op_pa = rms_op(b, T, C, eps, Y.w_pa, m.b_th2, m.b_th2);
        Y.op_add1 = add_op(b, T, C, "bf16", m.b_tx, m.b_th2, m.b_tx);
        Y.op_pf = rms_op(b, T, C, eps, Y.w_pf, m.b_tx, m.a_th.a.x);
        {
            const char* gw = nm.f("te.blk.%d.ffn_gate_up", li);
            RAD_ARCH_TRY(Y.gu.declare(b, nm, g, gw, { nm.ckpt("%s.mlp.gate_proj.weight", lp),
                                                      nm.ckpt("%s.mlp.up_proj.weight", lp) },
                                      {}, { m.tFF, m.tFF }, 2 * m.tFF, C, acc, li, m.a_th, m.b_tgu, T));
        }
        Y.op_act = rw(b, RAD_OP(b, "gelu_mul",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", m.tFF),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_tgu }, { m.a_tffn.a.x });
        RAD_ARCH_TRY(lin(Y.down, "ffn_down", "mlp.down_proj", C, m.tFF, m.a_tffn, m.b_th2));
        Y.op_ff = rms_op(b, T, C, eps, Y.w_ff, m.b_th2, m.b_th2);
        Y.op_add2 = add_op(b, T, C, "bf16", m.b_tx, m.b_th2, m.b_tx);
        Y.op_sc = rw(b, RAD_OP(b, "scale",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", C),
                                          RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(Y.w_sc)),
                     { m.b_tx }, { m.b_tx });
        /* Into the stack: a layer's output but the last's, which is normed first. */
        if (l + 1 < L) Y.op_keep = cast_op(b, T, C, "bf16", "bf16", m.b_tx, m.b_stack);
        if (!Y.op_in || !Y.op_vn || !Y.op_qn || !Y.op_kn || !Y.op_rq || !Y.op_rk || !Y.op_attn ||
            !Y.op_pa || !Y.op_add1 || !Y.op_pf || !Y.op_act || !Y.op_ff || !Y.op_add2 || !Y.op_sc ||
            (l + 1 < L && !Y.op_keep))
            return RAD_E_INVAL;
    }
    m.op_tnorm = rms_op(b, T, C, eps, m.w_tnorm, m.b_tx, m.b_stack);
    return m.op_embed && m.op_keep0 && m.op_tnorm ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== attention */
/* The rotary an attention's q and k take: LTX-2's split rotary over `n_dims` coordinates of a
 * `dim`-wide row, each coordinate over its `max`; none when n_dims is 0. */
struct Rope {
    int64_t     n_dims = 0, dim = 0;
    const char* max = "";
};

/* An attention's weights and ops (LTX2Attention): for a self-attention q|k|v from `xin` into `bq`
 * [rows, 3 inner]; for a cross-attention q from `xin` into `bq` [rows_q, inner] and k|v from
 * `kvin` into `bkv` [rows_k, 2 inner]. The gate's logits into `bg`, the attention into `ao`, the
 * output projection into `y` [rows_q, odim]. Every linear has a bias. A self-attention also gets
 * the value copy spatio-temporal guidance bypasses it with. */
static int attn_declare(RadBuilder* b, Model& m, const Geom& g, Attn& A, const char* dn,
                        const char* src, int64_t H, int64_t hd, int64_t qdim, int64_t kvdim,
                        int64_t odim, bool self, const Act& xin, const Act* kvin, rad_buf bq,
                        rad_buf bkv, rad_buf bg, const Act& ao, rad_buf y, int64_t rows_q,
                        int64_t rows_k, int acc, int group, const Rope& rope, double eps) {
    Names& nm = m.nm;
    A.H = H;
    A.hd = hd;
    A.inner = H * hd;
    const int64_t in = A.inner;
    auto w = [&](const char* name) { return nm.ckpt("%s.%s.weight", src, name); };
    auto bs = [&](const char* name) { return nm.ckpt("%s.%s.bias", src, name); };
    if (self)
        RAD_ARCH_TRY(A.qkv.declare(b, nm, g, nm.f("%s.qkv", dn), { w("to_q"), w("to_k"), w("to_v") },
                                   { bs("to_q"), bs("to_k"), bs("to_v") }, { in, in, in }, 3 * in,
                                   qdim, acc, group, xin, bq, rows_q));
    else {
        RAD_ARCH_TRY(A.qkv.declare(b, nm, g, nm.f("%s.q", dn), { w("to_q") }, { bs("to_q") }, {}, in,
                                   qdim, acc, group, xin, bq, rows_q));
        RAD_ARCH_TRY(A.kv.declare(b, nm, g, nm.f("%s.kv", dn), { w("to_k"), w("to_v") },
                                  { bs("to_k"), bs("to_v") }, { in, in }, 2 * in, kvdim, acc, group,
                                  *kvin, bkv, rows_k));
    }
    RAD_ARCH_TRY(A.gate.declare(b, nm, g, nm.f("%s.gate", dn), { w("to_gate_logits") },
                                { bs("to_gate_logits") }, {}, H, qdim, acc, group, xin, bg, rows_q,
                                true));
    RAD_ARCH_TRY(A.out.declare(b, nm, g, nm.f("%s.out", dn), { w("to_out.0") }, { bs("to_out.0") }, {},
                               odim, in, acc, group, ao, y, rows_q));
    A.w_qn = wcopy(b, nm.f("%s.q_norm.weight", dn), w("norm_q"), RAD_BF16, { in }, acc, group);
    A.w_kn = wcopy(b, nm.f("%s.k_norm.weight", dn), w("norm_k"), RAD_BF16, { in }, acc, group);
    if (!A.w_qn || !A.w_kn) return RAD_E_INVAL;
    const rad_buf kb = self ? bq : bkv;
    A.op_qn = rms_op(b, rows_q, in, eps, A.w_qn, bq, bq);
    A.op_kn = rms_op(b, rows_k, in, eps, A.w_kn, kb, kb);
    if (rope.n_dims) {
        auto rop = [&](int64_t rows, rad_buf x) {
            return rw(b, RAD_OP(b, "rope_split",
                                RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n_head", H),
                                           RAD_INT("head_dim", hd), RAD_INT("dim", rope.dim),
                                           RAD_INT("n_dims", rope.n_dims), RAD_F64("theta", 10000.0),
                                           RAD_STR("max", rope.max), RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { x }, { x });
        };
        A.op_rq = rop(rows_q, bq);
        A.op_rk = rop(rows_k, kb);
        if (!A.op_rq || !A.op_rk) return RAD_E_INVAL;
    }
    const double sc = 1.0 / std::sqrt((double)hd);
    if (self)
        A.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, rows_q), RAD_INT("head_dim", hd),
                                            RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                            RAD_STR("dtype", "bf16"), RAD_F64("scale", sc)),
                                 RAD_NOWEIGHTS),
                       { bq }, { ao.a.x });
    else
        A.op_attn = rw(b, RAD_OP(b, "attn_cross",
                                 RAD_PARAMS(RAD_RANGE("M", 1, rows_q), RAD_INT("head_dim", hd),
                                            RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                            RAD_STR("dtype", "bf16"), RAD_F64("scale", sc)),
                                 RAD_NOWEIGHTS),
                       { bq, bkv }, { ao.a.x });
    A.op_hg = rw(b, RAD_OP(b, "head_gate",
                           RAD_PARAMS(RAD_RANGE("M", 1, rows_q), RAD_INT("n_head", H),
                                      RAD_INT("head_dim", hd), RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS),
                 { bg, ao.a.x }, { ao.a.x });
    if (self) A.op_v = cast_op(b, rows_q, in, "bf16", "bf16", bq, ao.a.x);
    return A.op_qn && A.op_kn && A.op_attn && A.op_hg && (!self || A.op_v) ? RAD_OK : RAD_E_INVAL;
}

/* q and k normed and turned, the attention, the gate: the attention part both kinds share once
 * their projections ran. `cu` and (cross) `cu_k` bound the query and key rows; `pq` and `pk` the
 * rotary coordinates. */
static void attn_core(RadCtx* c, const Attn& A, bool self, rad_buf bq, rad_buf bkv, rad_buf bg,
                      const Act& ao, int64_t nq, int64_t nk, const RadOperand& cu,
                      const RadOperand& cu_k, const RadOperand& pq, const RadOperand& pk) {
    const int64_t in = A.inner;
    const RadOperand q = bcol(bq, 0, in, nq);
    const RadOperand k = self ? bcol(bq, in, in, nk) : bcol(bkv, 0, in, nk);
    const RadOperand v = self ? bcol(bq, 2 * in, in, nk) : bcol(bkv, in, in, nk);
    RAD_ISSUE_N(c, A.op_qn, nq, q, RAD_W(A.w_qn), q);
    RAD_ISSUE_N(c, A.op_kn, nk, k, RAD_W(A.w_kn), k);
    if (A.op_rq) {
        RAD_ISSUE_N(c, A.op_rq, nq, q, pq);
        RAD_ISSUE_N(c, A.op_rk, nk, k, pk);
    }
    if (self) RAD_ISSUE_N(c, A.op_attn, nq, q, k, v, cu, brows(ao.a.x, nq));
    else      RAD_ISSUE_N(c, A.op_attn, nq, q, k, v, cu, cu_k, brows(ao.a.x, nq));
    RAD_ISSUE_N(c, A.op_hg, nq, brows(bg, nq), brows(ao.a.x, nq));
}

/* ================================================================== the connectors */
/* A modality's connector: the normed stack scaled to its width and projected, the registers in
 * the rows past the prompt's tokens, eight blocks over 1024 rows a prompt, the final norm into the
 * lane's text rows. */
static int declare_conn(RadBuilder* b, const RadModelMeta* meta, Model& m, Conn& K, const char* md) {
    Names& nm = m.nm;
    const Geom& g = m.gt;
    const int64_t T = m.tRows, C = m.tC, Lp = m.tL + 1, R = 2 * (int64_t)kTextRows;
    const int acc = RAD_ACCESS_PER_REQUEST;
    const int gx = kGrpConn + (md[0] == 'v' ? 0 : 50);
    const double eps = 1e-6;
    K.H = rad_meta_geti(meta, nm.f("connectors.%s_connector_num_attention_heads", md), 0);
    K.hd = rad_meta_geti(meta, nm.f("connectors.%s_connector_attention_head_dim", md), 0);
    K.D = rad_meta_geti(meta, nm.f("connectors.%s_hidden_dim", md), 0);
    K.ff = 4 * K.D;
    const int64_t layers = rad_meta_geti(meta, nm.f("connectors.%s_connector_num_layers", md), 0);
    const int64_t regs = rad_meta_geti(meta, nm.f("connectors.%s_connector_num_learnable_registers", md),
                                       0);
    if (K.H <= 0 || K.hd <= 0 || K.D != K.H * K.hd || layers <= 0 || regs <= 0 ||
        kTextRows % regs || kTextRows / regs > 8 ||
        rad_meta_geti(meta, "connectors.caption_channels", 0) != C ||
        rad_meta_geti(meta, "connectors.text_proj_in_factor", 0) != Lp) {
        fprintf(stderr, "radiance: ltx2: the %s connector's geometry does not fit the text encoder "
                        "(%lld channels, %lld layers kept)\n", md, (long long)C, (long long)Lp);
        return RAD_E_UNSUPPORTED;
    }
    K.mult = std::sqrt((double)K.D / (double)C);
    const char* pre = nm.ckpt("connectors.%s_connector", md);

    /* The stack, per-layer RMS-normed over its channels and scaled, then the projection. */
    RAD_ARCH_TRY(K.a_in.declare(b, nm, g, nm.f("conn.%s.in", md), T, C * Lp));
    K.op_norm = rw(b, RAD_OP(b, "pixel_norm",
                             RAD_PARAMS(RAD_RANGE("M", 1, T * Lp), RAD_INT("n", C), RAD_F64("eps", eps),
                                        RAD_STR("form", "rsqrt"), RAD_INT("ileave", Lp),
                                        RAD_STR("dtype", "bf16"), RAD_F64("mult", K.mult)),
                             RAD_NOWEIGHTS),
                   { m.b_stack }, { K.a_in.a.x });
    K.b_x = decl_b(b, nm.f("conn.%s.x", md), RAD_BF16, { R, K.D });
    RAD_ARCH_TRY(K.proj.declare(b, nm, g, nm.f("conn.%s.proj", md),
                                { nm.ckpt("connectors.%s_text_proj_in.weight", md) },
                                { nm.ckpt("connectors.%s_text_proj_in.bias", md) }, {}, K.D, C * Lp,
                                acc, gx, K.a_in, K.b_x, T));
    {
        const char* wn = nm.f("conn.%s.registers", md);
        const char* rs = nm.ckpt("%s.learnable_registers", pre);
        const int reps = (int)(kTextRows / regs);
        RadNameMap mp = map_base(wn, RAD_MAP_ROWS);
        for (int i = 0; i < reps; ++i) {
            mp.src[mp.n_src] = rs;
            mp.src_index[mp.n_src] = 0;
            mp.src_rows[mp.n_src] = regs;
            ++mp.n_src;
        }
        RAD_ARCH_TRY(rad_decl_name_map(b, &mp));
        K.w_reg = decl_w(b, wn, RAD_BF16, { (int64_t)kTextRows, K.D }, acc, RAD_SHARD_NONE,
                         grp_layer(gx));
        if (!K.w_reg) return RAD_E_INVAL;
        K.op_reg = rw(b, RAD_OP(b, "table_add",
                                RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", K.D),
                                           RAD_INT("L", (int64_t)kTextRows), RAD_STR("dtype", "bf16")),
                                RAD_WEIGHTS(K.w_reg)),
                      { m.b_zero }, { K.b_x });
    }

    /* The blocks. */
    RAD_ARCH_TRY(K.a_h.declare(b, nm, g, nm.f("conn.%s.h", md), R, K.D));
    RAD_ARCH_TRY(K.a_attn.declare(b, nm, g, nm.f("conn.%s.attn", md), R, K.D));
    RAD_ARCH_TRY(K.a_ffn.declare(b, nm, g, nm.f("conn.%s.ffn", md), R, K.ff));
    K.b_qkv = decl_b(b, nm.f("conn.%s.qkv", md), RAD_BF16, { R, 3 * K.D });
    K.b_g = decl_b(b, nm.f("conn.%s.g", md), RAD_BF16, { R, K.H });
    K.b_o = decl_b(b, nm.f("conn.%s.o", md), RAD_BF16, { R, K.D });
    K.b_up = decl_b(b, nm.f("conn.%s.up", md), RAD_BF16, { R, K.ff });
    Rope rope{ 1, K.D, "4096" };
    K.blk.resize((size_t)layers);
    for (int64_t l = 0; l < layers; ++l) {
        ConnBlock& B = K.blk[(size_t)l];
        const char* bp = nm.ckpt("%s.transformer_blocks.%lld", pre, (long long)l);
        const char* dn = nm.f("conn.%s.blk.%lld", md, (long long)l);
        B.op_n1 = rms_op(b, R, K.D, eps, 0, K.b_x, K.a_h.a.x);
        RAD_ARCH_TRY(attn_declare(b, m, g, B.at, dn, nm.ckpt("%s.attn1", bp), K.H, K.hd, K.D, K.D, K.D,
                                  true, K.a_h, nullptr, K.b_qkv, 0, K.b_g, K.a_attn, K.b_o, R, R, acc,
                                  gx, rope, eps));
        B.op_add1 = add_op(b, R, K.D, "bf16", K.b_x, K.b_o, K.b_x);
        B.op_n2 = rms_op(b, R, K.D, eps, 0, K.b_x, K.a_h.a.x);
        RAD_ARCH_TRY(B.up.declare(b, nm, g, nm.f("%s.ff_up", dn),
                                  { nm.ckpt("%s.ff.net.0.proj.weight", bp) },
                                  { nm.ckpt("%s.ff.net.0.proj.bias", bp) }, {}, K.ff, K.D, acc, gx,
                                  K.a_h, K.b_up, R));
        B.op_act = unary(b, "gelu_tanh", R, K.ff, "bf16", K.b_up, K.a_ffn.a.x);
        RAD_ARCH_TRY(B.down.declare(b, nm, g, nm.f("%s.ff_down", dn),
                                    { nm.ckpt("%s.ff.net.2.weight", bp) },
                                    { nm.ckpt("%s.ff.net.2.bias", bp) }, {}, K.D, K.ff, acc, gx,
                                    K.a_ffn, K.b_o, R));
        B.op_add2 = add_op(b, R, K.D, "bf16", K.b_x, K.b_o, K.b_x);
        if (!B.op_n1 || !B.op_add1 || !B.op_n2 || !B.op_act || !B.op_add2) return RAD_E_INVAL;
    }
    K.b_text = decl_b(b, nm.f("lane.%s_text", md), RAD_BF16, { (int64_t)kMaxLanes * R, K.D },
                      RAD_BUF_PERSIST);
    K.op_nout = rms_op(b, R, K.D, eps, 0, K.b_x, K.b_text);
    return K.op_norm && K.op_reg && K.b_text && K.op_nout ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the DiT */
/* One AdaLayerNormSingle (`mod`): the [cos | sin] sinusoids of the step's timesteps through
 * linear_1, SiLU and linear_2 to `emb`, then SiLU and `linear` to `temb`, bf16 throughout. `perm`
 * reorders linear's output rows in blocks of d (LTX-2's cross-modal tables hold scale before
 * shift; the norms take shift first). */
static int ada_declare(RadBuilder* b, Model& m, Ada& A, const char* dn, const char* mod, int64_t d,
                       int64_t k, std::initializer_list<int> perm = {}) {
    Names& nm = m.nm;
    const int acc = RAD_ACCESS_PER_TOKEN, gx = kGrpTemb;
    A.d = d;
    A.k = k;
    const char* src = nm.ckpt("transformer.%s", mod);
    A.w1 = wcopy(b, nm.f("%s.l1.weight", dn), nm.ckpt("%s.emb.timestep_embedder.linear_1.weight", src),
                 RAD_BF16, { d, 256 }, acc, gx);
    A.b1 = wcopy(b, nm.f("%s.l1.bias", dn), nm.ckpt("%s.emb.timestep_embedder.linear_1.bias", src),
                 RAD_BF16, { d }, acc, gx);
    A.w2 = wcopy(b, nm.f("%s.l2.weight", dn), nm.ckpt("%s.emb.timestep_embedder.linear_2.weight", src),
                 RAD_BF16, { d, d }, acc, gx);
    A.b2 = wcopy(b, nm.f("%s.l2.bias", dn), nm.ckpt("%s.emb.timestep_embedder.linear_2.bias", src),
                 RAD_BF16, { d }, acc, gx);
    const char* lw = nm.f("%s.lin.weight", dn);
    const char* lb = nm.f("%s.lin.bias", dn);
    if (perm.size()) {
        std::vector<RowPiece> pw, pb;
        const char* sw = nm.ckpt("%s.linear.weight", src);
        const char* sb = nm.ckpt("%s.linear.bias", src);
        RadNameMap mw = map_base(lw, RAD_MAP_ROWS), mb = map_base(lb, RAD_MAP_ROWS);
        for (int p : perm) {
            mw.src[mw.n_src] = sw;
            mw.src_index[mw.n_src] = (int)(p * d);
            mw.src_rows[mw.n_src++] = d;
            mb.src[mb.n_src] = sb;
            mb.src_index[mb.n_src] = (int)(p * d);
            mb.src_rows[mb.n_src++] = d;
        }
        RAD_ARCH_TRY(rad_decl_name_map(b, &mw));
        RAD_ARCH_TRY(rad_decl_name_map(b, &mb));
        A.wl = decl_w(b, lw, RAD_BF16, { k * d, d }, acc, RAD_SHARD_NONE, grp_layer(gx));
        A.bl = decl_w(b, lb, RAD_BF16, { k * d }, acc, RAD_SHARD_NONE, grp_layer(gx));
    } else {
        A.wl = wcopy(b, lw, nm.ckpt("%s.linear.weight", src), RAD_BF16, { k * d, d }, acc, gx);
        A.bl = wcopy(b, lb, nm.ckpt("%s.linear.bias", src), RAD_BF16, { k * d }, acc, gx);
    }
    if (!A.w1 || !A.b1 || !A.w2 || !A.b2 || !A.wl || !A.bl) return RAD_E_INVAL;
    A.h = decl_b(b, nm.f("%s.h", dn), RAD_BF16, { 2, d });
    A.hs = decl_b(b, nm.f("%s.hs", dn), RAD_BF16, { 2, d });
    A.emb = decl_b(b, nm.f("%s.emb", dn), RAD_BF16, { 2, d }, RAD_BUF_PERSIST);
    A.es = decl_b(b, nm.f("%s.es", dn), RAD_BF16, { 2, d });
    A.temb = decl_b(b, nm.f("%s.temb", dn), RAD_BF16, { 2, k * d });
    auto lin = [&](rad_weight w, rad_weight bb, int64_t N, int64_t K, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "gemm_nt_bias",
                            RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("N", N), RAD_INT("K", K),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w, bb)),
                  { x }, { y });
    };
    A.op1 = lin(A.w1, A.b1, d, 256, m.b_tsin, A.h);
    A.ops1 = unary(b, "silu", 2, d, "bf16", A.h, A.hs);
    A.op2 = lin(A.w2, A.b2, d, d, A.hs, A.emb);
    A.ops2 = unary(b, "silu", 2, d, "bf16", A.emb, A.es);
    A.opl = lin(A.wl, A.bl, k * d, d, A.es, A.temb);
    return A.op1 && A.ops1 && A.op2 && A.ops2 && A.opl ? RAD_OK : RAD_E_INVAL;
}

/* The module over rows [r0, r0 + n) of the step's timesteps' sinusoids `tsin`, into its own rows
 * [r0, r0 + n). */
static void ada_step(RadCtx* c, const Ada& A, rad_buf tsin, int64_t r0, int64_t n) {
    const int64_t d = A.d, kd = A.k * A.d;
    auto rows = [&](rad_buf h, int64_t w) { return brow_slice(h, r0, n, w); };
    RAD_ISSUE_N(c, A.op1, n, rows(tsin, 256), RAD_W(A.w1), RAD_W(A.b1), RAD_NONE, rows(A.h, d));
    RAD_ISSUE_N(c, A.ops1, n, rows(A.h, d), rows(A.hs, d));
    RAD_ISSUE_N(c, A.op2, n, rows(A.hs, d), RAD_W(A.w2), RAD_W(A.b2), RAD_NONE, rows(A.emb, d));
    RAD_ISSUE_N(c, A.ops2, n, rows(A.emb, d), rows(A.es, d));
    RAD_ISSUE_N(c, A.opl, n, rows(A.es, d), RAD_W(A.wl), RAD_W(A.bl), RAD_NONE, rows(A.temb, kd));
}

static int declare_dit(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    const Geom& g = m.gd;
    const int64_t S = m.maxS, A = m.maxA, R = kTextRows, L = m.nb;
    const int64_t D = m.D, Da = m.Da, Dc = m.Dc;
    const int acc = RAD_ACCESS_PER_TOKEN;
    const double eps = m.eps;
    const int64_t La = kMaxLanes;

    /* ---- the step's timesteps {0, sigma} (the first frame's and everything else's), their
     * sinusoids, and the eight modulation MLPs */
    m.b_tin = decl_b(b, nm.f("dit.tin"), RAD_F32, { 2 });
    m.b_tsin = decl_b(b, nm.f("dit.tsin"), RAD_BF16, { 2, 256 });
    m.op_tin0 = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 2), RAD_STR("dtype", "f32"),
                                        RAD_F64("value", 0.0)),
                             RAD_NOWEIGHTS),
                   {}, { m.b_tin });
    m.op_tin1 = cast_op(b, 1, 1, "f32", "f32", m.b_tin, m.b_tin);
    /* t' = f32(1000 sigma), the timestep diffusers' scheduler hands the transformer. */
    m.op_tsin = rw(b, RAD_OP(b, "timestep_embed",
                             RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("n", 256),
                                        RAD_F64("scale", 1000.0), RAD_F64("max_period", 10000.0),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_tin }, { m.b_tsin });
    if (!m.op_tin0 || !m.op_tin1 || !m.op_tsin) return RAD_E_INVAL;
    RAD_ARCH_TRY(ada_declare(b, m, m.ad_v, "dit.ada_v", "time_embed", D, 9));
    RAD_ARCH_TRY(ada_declare(b, m, m.ad_a, "dit.ada_a", "audio_time_embed", Da, 9));
    RAD_ARCH_TRY(ada_declare(b, m, m.ad_p, "dit.ada_p", "prompt_adaln", D, 2));
    RAD_ARCH_TRY(ada_declare(b, m, m.ad_pa, "dit.ada_pa", "audio_prompt_adaln", Da, 2));
    RAD_ARCH_TRY(ada_declare(b, m, m.ad_cv, "dit.ada_cv", "av_cross_attn_video_scale_shift", D, 4,
                             { 1, 0, 3, 2 }));
    RAD_ARCH_TRY(ada_declare(b, m, m.ad_cvg, "dit.ada_cvg", "av_cross_attn_video_a2v_gate", D, 1));
    RAD_ARCH_TRY(ada_declare(b, m, m.ad_ca, "dit.ada_ca", "av_cross_attn_audio_scale_shift", Da, 4,
                             { 1, 0, 3, 2 }));
    RAD_ARCH_TRY(ada_declare(b, m, m.ad_cag, "dit.ada_cag", "av_cross_attn_audio_v2a_gate", Da, 1));

    /* ---- the step's tables (PERSIST): block l's rows at column l of each */
    m.b_mv = decl_b(b, nm.f("dit.mod_v"), RAD_BF16, { 2, L * 9 * D }, RAD_BUF_PERSIST);
    m.b_ma = decl_b(b, nm.f("dit.mod_a"), RAD_BF16, { 1, L * 9 * Da }, RAD_BUF_PERSIST);
    m.b_mp = decl_b(b, nm.f("dit.mod_p"), RAD_BF16, { 1, L * 2 * D }, RAD_BUF_PERSIST);
    m.b_mpa = decl_b(b, nm.f("dit.mod_pa"), RAD_BF16, { 1, L * 2 * Da }, RAD_BUF_PERSIST);
    m.b_mcv = decl_b(b, nm.f("dit.mod_cv"), RAD_BF16, { 1, L * 4 * D }, RAD_BUF_PERSIST);
    m.b_mcvg = decl_b(b, nm.f("dit.mod_cvg"), RAD_BF16, { 1, L * D }, RAD_BUF_PERSIST);
    m.b_mca = decl_b(b, nm.f("dit.mod_ca"), RAD_BF16, { 1, L * 4 * Da }, RAD_BUF_PERSIST);
    m.b_mcag = decl_b(b, nm.f("dit.mod_cag"), RAD_BF16, { 1, L * Da }, RAD_BUF_PERSIST);
    m.b_fv = decl_b(b, nm.f("dit.mod_fv"), RAD_BF16, { 2, 2 * D }, RAD_BUF_PERSIST);
    m.b_fa = decl_b(b, nm.f("dit.mod_fa"), RAD_BF16, { 1, 2 * Da }, RAD_BUF_PERSIST);
    auto table_op = [&](rad_weight t, int64_t rows, int64_t n, int64_t Lr, rad_buf x, rad_buf y) {
        return rw(b, RAD_OP(b, "table_add",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_INT("L", Lr),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(t)),
                  { x }, { y });
    };
    m.w_fv = wcopy(b, nm.f("dit.out_table"), nm.ckpt("transformer.scale_shift_table"), RAD_BF16,
                   { 2, D }, acc, kGrpTemb);
    m.w_fa = wcopy(b, nm.f("dit.audio_out_table"), nm.ckpt("transformer.audio_scale_shift_table"),
                   RAD_BF16, { 2, Da }, acc, kGrpTemb);
    if (!m.w_fv || !m.w_fa) return RAD_E_INVAL;
    m.op_fv = table_op(m.w_fv, 2, D, 2, m.ad_v.emb, m.b_fv);
    m.op_fa = table_op(m.w_fa, 1, Da, 2, m.ad_a.emb, m.b_fa);

    /* ---- the lanes (PERSIST): both latents f32, the clean first latent frame */
    m.b_vstate = decl_b(b, nm.f("lane.video"), RAD_F32, { La * S, kC }, RAD_BUF_PERSIST);
    m.b_astate = decl_b(b, nm.f("lane.audio"), RAD_F32, { La * A, kAudioC }, RAD_BUF_PERSIST);
    m.b_first = decl_b(b, nm.f("lane.first"), RAD_F32, { La * m.maxPpf, kC }, RAD_BUF_PERSIST);
    auto randn = [&](int64_t rows, int64_t stride, rad_buf x) {
        return rw(b, RAD_OP(b, "randn_lanes",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", kC),
                                       RAD_INT("stride", stride), RAD_STR("dtype", "f32")),
                            RAD_NOWEIGHTS),
                  { x }, { x });
    };
    m.op_rnd_v = randn(S, S, m.b_vstate);
    m.op_rnd_a = randn(A, A, m.b_astate);
    m.op_keep_first = cast_op(b, m.maxPpf, kC, "f32", "f32", m.b_first, m.b_vstate);

    /* ---- an evaluation's rows */
    RAD_ARCH_TRY(m.a_lv.declare(b, nm, g, "dit.lat_v", S, kC));
    RAD_ARCH_TRY(m.a_la.declare(b, nm, g, "dit.lat_a", A, kAudioC));
    m.op_lv = cast_op(b, S, kC, "f32", "bf16", m.b_vstate, m.a_lv.a.x);
    m.op_la = cast_op(b, A, kAudioC, "f32", "bf16", m.b_astate, m.a_la.a.x);
    m.b_hv = decl_b(b, nm.f("dit.hv"), RAD_BF16, { S, D });
    m.b_ha = decl_b(b, nm.f("dit.ha"), RAD_BF16, { A, Da });
    RAD_ARCH_TRY(m.pin_v.declare(b, nm, g, "dit.proj_in", { nm.ckpt("transformer.proj_in.weight") },
                                 { nm.ckpt("transformer.proj_in.bias") }, {}, D, kC, acc, kGrpDit,
                                 m.a_lv, m.b_hv, S, true));
    RAD_ARCH_TRY(m.pin_a.declare(b, nm, g, "dit.audio_proj_in",
                                 { nm.ckpt("transformer.audio_proj_in.weight") },
                                 { nm.ckpt("transformer.audio_proj_in.bias") }, {}, Da, kAudioC, acc,
                                 kGrpDit, m.a_la, m.b_ha, A, true));
    RAD_ARCH_TRY(m.a_hv.declare(b, nm, g, "dit.hnv", S, D));
    RAD_ARCH_TRY(m.a_ha.declare(b, nm, g, "dit.hna", A, Da));
    RAD_ARCH_TRY(m.a_ctxv.declare(b, nm, g, "dit.ctx_v", R, D));
    RAD_ARCH_TRY(m.a_ctxa.declare(b, nm, g, "dit.ctx_a", R, Da));
    RAD_ARCH_TRY(m.a_nv.declare(b, nm, g, "dit.nv_a2v", S, D));
    RAD_ARCH_TRY(m.a_nv2.declare(b, nm, g, "dit.nv_v2a", S, D));
    RAD_ARCH_TRY(m.a_na.declare(b, nm, g, "dit.na_a2v", A, Da));
    RAD_ARCH_TRY(m.a_na2.declare(b, nm, g, "dit.na_v2a", A, Da));
    RAD_ARCH_TRY(m.a_aov.declare(b, nm, g, "dit.attn_v", S, D));
    RAD_ARCH_TRY(m.a_aoa.declare(b, nm, g, "dit.attn_a", A, Da));
    RAD_ARCH_TRY(m.a_xv.declare(b, nm, g, "dit.attn_a2v", S, Dc));
    RAD_ARCH_TRY(m.a_xa.declare(b, nm, g, "dit.attn_v2a", A, Dc));
    RAD_ARCH_TRY(m.a_ffv.declare(b, nm, g, "dit.ffn_v", S, m.ffv));
    RAD_ARCH_TRY(m.a_ffa.declare(b, nm, g, "dit.ffn_a", A, m.ffa));
    RAD_ARCH_TRY(m.a_fv.declare(b, nm, g, "dit.out_v", S, D));
    RAD_ARCH_TRY(m.a_fa.declare(b, nm, g, "dit.out_a", A, Da));
    m.b_qkvv = decl_b(b, nm.f("dit.qkv_v"), RAD_BF16, { S, 3 * D });
    m.b_qkva = decl_b(b, nm.f("dit.qkv_a"), RAD_BF16, { A, 3 * Da });
    m.b_gv = decl_b(b, nm.f("dit.gate_v"), RAD_BF16, { S, m.H });
    m.b_ga = decl_b(b, nm.f("dit.gate_a"), RAD_BF16, { A, m.Ha });
    m.b_ov = decl_b(b, nm.f("dit.o_v"), RAD_BF16, { S, D });
    m.b_oa = decl_b(b, nm.f("dit.o_a"), RAD_BF16, { A, Da });
    m.b_qv = decl_b(b, nm.f("dit.q_v"), RAD_BF16, { S, D });
    m.b_qa = decl_b(b, nm.f("dit.q_a"), RAD_BF16, { A, Da });
    m.b_kvtv = decl_b(b, nm.f("dit.kv_tv"), RAD_BF16, { R, 2 * D });
    m.b_kvta = decl_b(b, nm.f("dit.kv_ta"), RAD_BF16, { R, 2 * Da });
    m.b_qxv = decl_b(b, nm.f("dit.q_a2v"), RAD_BF16, { S, Dc });
    m.b_gxv = decl_b(b, nm.f("dit.gate_a2v"), RAD_BF16, { S, m.Ha });
    m.b_kvxa = decl_b(b, nm.f("dit.kv_a2v"), RAD_BF16, { A, 2 * Dc });
    m.b_qxa = decl_b(b, nm.f("dit.q_v2a"), RAD_BF16, { A, Dc });
    m.b_gxa = decl_b(b, nm.f("dit.gate_v2a"), RAD_BF16, { A, m.Ha });
    m.b_kvxv = decl_b(b, nm.f("dit.kv_v2a"), RAD_BF16, { S, 2 * Dc });
    m.b_upv = decl_b(b, nm.f("dit.up_v"), RAD_BF16, { S, m.ffv });
    m.b_upa = decl_b(b, nm.f("dit.up_a"), RAD_BF16, { A, m.ffa });

    /* ---- the blocks */
    const Rope rv{ 3, D, "20 2048 2048" }, ra{ 1, Da, "20" }, rx{ 1, Dc, "20" }, none{};
    m.dit.resize((size_t)L);
    for (int64_t l = 0; l < L; ++l) {
        DitBlock& B = m.dit[(size_t)l];
        const int li = (int)l, gx = kGrpDit + 1 + li;
        const char* bp = nm.ckpt("transformer.transformer_blocks.%d", li);
        const char* dn = nm.f("dit.blk.%d", li);
        auto attn = [&](Attn& At, const char* name, const char* src, int64_t H, int64_t hd,
                        int64_t qdim, int64_t kvdim, int64_t odim, bool self, const Act& xin,
                        const Act* kvin, rad_buf bq, rad_buf bkv, rad_buf bg, const Act& ao, rad_buf y,
                        int64_t rq, int64_t rk, const Rope& rope) {
            return attn_declare(b, m, g, At, nm.f("%s.%s", dn, name), nm.ckpt("%s.%s", bp, src), H, hd,
                                qdim, kvdim, odim, self, xin, kvin, bq, bkv, bg, ao, y, rq, rk, acc, gx,
                                rope, eps);
        };
        /* The step's tables: the block's learned rows plus the step's shared ones. */
        auto table = [&](rad_weight* w, const char* dname, const char* name,
                         std::initializer_list<int> rows, int64_t d) {
            const char* wn = nm.f("%s.%s", dn, dname);
            bool natural = true;
            int i = 0;
            for (int r : rows) natural = natural && r == i++;
            if (natural && (int64_t)rows.size() > 1) {
                if (map_copy(b, wn, nm.ckpt("%s.%s", bp, name)) < 0) return RAD_E_INVAL;
            } else {
                if (rows.size() > 8) return RAD_E_INVAL;
                RadNameMap mp = map_base(wn, RAD_MAP_ROWS);
                for (int r : rows) {
                    mp.src[mp.n_src] = nm.ckpt("%s.%s", bp, name);
                    mp.src_index[mp.n_src] = r;
                    mp.src_rows[mp.n_src++] = 1;
                }
                if (rad_decl_name_map(b, &mp) < 0) return RAD_E_INVAL;
            }
            *w = decl_w(b, wn, RAD_BF16, { (int64_t)rows.size(), d }, acc, RAD_SHARD_NONE,
                        grp_layer(gx));
            return *w ? RAD_OK : RAD_E_INVAL;
        };
        RAD_ARCH_TRY(table(&B.t_v, "table_v", "scale_shift_table", { 0, 1, 2, 3, 4, 5, 6, 7, 8 }, D));
        RAD_ARCH_TRY(table(&B.t_a, "table_a", "audio_scale_shift_table", { 0, 1, 2, 3, 4, 5, 6, 7, 8 },
                           Da));
        RAD_ARCH_TRY(table(&B.t_p, "table_p", "prompt_scale_shift_table", { 0, 1 }, D));
        RAD_ARCH_TRY(table(&B.t_pa, "table_pa", "audio_prompt_scale_shift_table", { 0, 1 }, Da));
        RAD_ARCH_TRY(table(&B.t_cv, "table_cv", "video_a2v_cross_attn_scale_shift_table", { 1, 0, 3, 2 },
                           D));
        RAD_ARCH_TRY(table(&B.t_cvg, "table_cvg", "video_a2v_cross_attn_scale_shift_table", { 4 }, D));
        RAD_ARCH_TRY(table(&B.t_ca, "table_ca", "audio_a2v_cross_attn_scale_shift_table", { 1, 0, 3, 2 },
                           Da));
        RAD_ARCH_TRY(table(&B.t_cag, "table_cag", "audio_a2v_cross_attn_scale_shift_table", { 4 }, Da));
        B.op_tv = table_op(B.t_v, 2, 9 * D, 1, m.ad_v.temb, m.b_mv);
        B.op_ta = table_op(B.t_a, 1, 9 * Da, 1, m.ad_a.temb, m.b_ma);
        B.op_tp = table_op(B.t_p, 1, 2 * D, 1, m.ad_p.temb, m.b_mp);
        B.op_tpa = table_op(B.t_pa, 1, 2 * Da, 1, m.ad_pa.temb, m.b_mpa);
        B.op_tcv = table_op(B.t_cv, 1, 4 * D, 1, m.ad_cv.temb, m.b_mcv);
        B.op_tcvg = table_op(B.t_cvg, 1, D, 1, m.ad_cvg.temb, m.b_mcvg);
        B.op_tca = table_op(B.t_ca, 1, 4 * Da, 1, m.ad_ca.temb, m.b_mca);
        B.op_tcag = table_op(B.t_cag, 1, Da, 1, m.ad_cag.temb, m.b_mcag);

        /* Self-attention (block 0's q|k|v was mapped by declare's format probe). */
        if (l == 0) B.sv.qkv.mapped = true;
        B.m1v = mod_op(b, S, D, eps, l > 0, "rms", m.b_mv, m.b_mv, m.b_hv, m.b_ov, m.a_hv.a.x);
        B.m1a = mod_op(b, A, Da, eps, l > 0, "rms", m.b_ma, m.b_ma, m.b_ha, m.b_oa, m.a_ha.a.x);
        RAD_ARCH_TRY(attn(B.sv, "attn1", "attn1", m.H, m.hd, D, D, D, true, m.a_hv, nullptr, m.b_qkvv, 0,
                          m.b_gv, m.a_aov, m.b_ov, S, S, rv));
        RAD_ARCH_TRY(attn(B.sa, "audio_attn1", "audio_attn1", m.Ha, m.hda, Da, Da, Da, true, m.a_ha,
                          nullptr, m.b_qkva, 0, m.b_ga, m.a_aoa, m.b_oa, A, A, ra));
        /* Text cross-attention: the query's norm gated by the self-attention, the text keys and
         * values modulated as they are. */
        B.m2v = mod_op(b, S, D, eps, true, "rms", m.b_mv, m.b_mv, m.b_hv, m.b_ov, m.a_hv.a.x);
        B.m2a = mod_op(b, A, Da, eps, true, "rms", m.b_ma, m.b_ma, m.b_ha, m.b_oa, m.a_ha.a.x);
        B.mtv = mod_op(b, R, D, eps, false, "none", m.b_mp, m.b_mp, m.cv.b_text, 0, m.a_ctxv.a.x);
        B.mta = mod_op(b, R, Da, eps, false, "none", m.b_mpa, m.b_mpa, m.ca.b_text, 0, m.a_ctxa.a.x);
        RAD_ARCH_TRY(attn(B.tv, "attn2", "attn2", m.H, m.hd, D, D, D, false, m.a_hv, &m.a_ctxv, m.b_qv,
                          m.b_kvtv, m.b_gv, m.a_aov, m.b_ov, S, R, none));
        RAD_ARCH_TRY(attn(B.ta, "audio_attn2", "audio_attn2", m.Ha, m.hda, Da, Da, Da, false, m.a_ha,
                          &m.a_ctxa, m.b_qa, m.b_kvta, m.b_ga, m.a_aoa, m.b_oa, A, R, none));
        /* The cross-modal attentions: both streams gated by their text attention, each normed
         * twice (as queries and as keys), the attentions, their gates. */
        B.g2v = gate_op(b, S, D, m.b_mv, m.b_ov, m.b_hv);
        B.g2a = gate_op(b, A, Da, m.b_ma, m.b_oa, m.b_ha);
        B.mav = mod_op(b, S, D, eps, false, "rms", m.b_mcv, m.b_mcv, m.b_hv, 0, m.a_nv.a.x);
        B.mvv = mod_op(b, S, D, eps, false, "rms", m.b_mcv, m.b_mcv, m.b_hv, 0, m.a_nv2.a.x);
        B.maa = mod_op(b, A, Da, eps, false, "rms", m.b_mca, m.b_mca, m.b_ha, 0, m.a_na.a.x);
        B.mva = mod_op(b, A, Da, eps, false, "rms", m.b_mca, m.b_mca, m.b_ha, 0, m.a_na2.a.x);
        RAD_ARCH_TRY(attn(B.a2v, "a2v", "audio_to_video_attn", m.Ha, m.hda, D, Da, D, false, m.a_nv,
                          &m.a_na, m.b_qxv, m.b_kvxa, m.b_gxv, m.a_xv, m.b_ov, S, A, rx));
        RAD_ARCH_TRY(attn(B.v2a, "v2a", "video_to_audio_attn", m.Ha, m.hda, Da, D, Da, false, m.a_na2,
                          &m.a_nv2, m.b_qxa, m.b_kvxv, m.b_gxa, m.a_xa, m.b_oa, A, S, rx));
        B.gcv = gate_op(b, S, D, m.b_mcvg, m.b_ov, m.b_hv);
        B.gca = gate_op(b, A, Da, m.b_mcag, m.b_oa, m.b_ha);
        /* The feed-forward (the video's without biases). */
        B.m3v = mod_op(b, S, D, eps, false, "rms", m.b_mv, m.b_mv, m.b_hv, 0, m.a_hv.a.x);
        B.m3a = mod_op(b, A, Da, eps, false, "rms", m.b_ma, m.b_ma, m.b_ha, 0, m.a_ha.a.x);
        B.m3vg = mod_op(b, S, D, eps, true, "rms", m.b_mv, m.b_mv, m.b_hv, m.b_ov, m.a_hv.a.x);
        B.m3ag = mod_op(b, A, Da, eps, true, "rms", m.b_ma, m.b_ma, m.b_ha, m.b_oa, m.a_ha.a.x);
        RAD_ARCH_TRY(B.fv_up.declare(b, nm, g, nm.f("%s.ff_up", dn),
                                     { nm.ckpt("%s.ff.net.0.proj.weight", bp) },
                                     {}, {}, m.ffv, D, acc, gx, m.a_hv, m.b_upv, S));
        B.actv = unary(b, "gelu_tanh", S, m.ffv, "bf16", m.b_upv, m.a_ffv.a.x);
        RAD_ARCH_TRY(B.fv_down.declare(b, nm, g, nm.f("%s.ff_down", dn),
                                       { nm.ckpt("%s.ff.net.2.weight", bp) },
                                       {}, {}, D, m.ffv, acc, gx, m.a_ffv, m.b_ov, S));
        RAD_ARCH_TRY(B.fa_up.declare(b, nm, g, nm.f("%s.audio_ff_up", dn),
                                     { nm.ckpt("%s.audio_ff.net.0.proj.weight", bp) },
                                     { nm.ckpt("%s.audio_ff.net.0.proj.bias", bp) }, {}, m.ffa,
                                     Da, acc, gx, m.a_ha, m.b_upa, A));
        B.acta = unary(b, "gelu_tanh", A, m.ffa, "bf16", m.b_upa, m.a_ffa.a.x);
        RAD_ARCH_TRY(B.fa_down.declare(b, nm, g, nm.f("%s.audio_ff_down", dn),
                                       { nm.ckpt("%s.audio_ff.net.2.weight", bp) },
                                       { nm.ckpt("%s.audio_ff.net.2.bias", bp) }, {}, Da, m.ffa, acc, gx,
                                       m.a_ffa, m.b_oa, A));
        const rad_op all[] = { B.op_tv, B.op_ta, B.op_tp, B.op_tpa, B.op_tcv, B.op_tcvg, B.op_tca,
                               B.op_tcag, B.m1v, B.m1a, B.m2v, B.m2a, B.mtv, B.mta, B.g2v, B.g2a,
                               B.mav, B.mvv, B.maa, B.mva, B.gcv, B.gca, B.m3v, B.m3a, B.m3vg,
                               B.m3ag, B.actv, B.acta };
        for (rad_op o : all)
            if (!o) return RAD_E_INVAL;
    }

    /* ---- the output: the last feed-forward gated in, LayerNorm modulated by the output tables,
     * proj_out; each evaluation's velocities (PERSIST, the guidance reads them all) */
    m.op_fnv = mod_op(b, S, D, eps, true, "layer", m.b_fv, m.b_mv, m.b_hv, m.b_ov, m.a_fv.a.x);
    m.op_fna = mod_op(b, A, Da, eps, true, "layer", m.b_fa, m.b_ma, m.b_ha, m.b_oa, m.a_fa.a.x);
    for (int p = 0; p < kPasses; ++p) {
        m.b_vel_v[p] = decl_b(b, nm.f("dit.vel_v%d", p), RAD_BF16, { S, kC }, RAD_BUF_PERSIST);
        m.b_vel_a[p] = decl_b(b, nm.f("dit.vel_a%d", p), RAD_BF16, { A, kAudioC }, RAD_BUF_PERSIST);
    }
    RAD_ARCH_TRY(m.pout_v.declare(b, nm, g, "dit.proj_out", { nm.ckpt("transformer.proj_out.weight") },
                                  { nm.ckpt("transformer.proj_out.bias") }, {}, kC, D, acc, kGrpDit,
                                  m.a_fv, m.b_vel_v[0], S, true));
    RAD_ARCH_TRY(m.pout_a.declare(b, nm, g, "dit.audio_proj_out",
                                  { nm.ckpt("transformer.audio_proj_out.weight") },
                                  { nm.ckpt("transformer.audio_proj_out.bias") }, {}, kAudioC, Da, acc,
                                  kGrpDit, m.a_fa, m.b_vel_a[0], A, true));
    /* The other evaluations' velocities are written by the same heads. */
    for (int p = 1; p < kPasses; ++p) {
        RAD_ARCH_TRY(rad_op_writes(b, m.pout_v.op, &m.b_vel_v[p], 1));
        RAD_ARCH_TRY(rad_op_writes(b, m.pout_a.op, &m.b_vel_a[p], 1));
    }

    /* ---- the guided step: each modality's latent from its evaluations */
    m.b_gst = decl_b(b, nm.f("dit.guide_stats"), RAD_F32, { 8200 });
    auto guide = [&](int64_t rows, rad_buf* vel, rad_buf x) {
        rad_op op = RAD_OP(b, "ltx_guide",
                           RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", kC),
                                      RAD_STR("dtype", "bf16")),
                           RAD_NOWEIGHTS);
        if (!op) return op;
        rad_buf rd[6] = { vel[0], vel[1], vel[2], vel[3], m.b_gst, x };
        rad_buf wr[2] = { m.b_gst, x };
        rad_op_reads(b, op, rd, 6);
        rad_op_writes(b, op, wr, 2);
        return op;
    };
    m.op_gv = guide(S, m.b_vel_v, m.b_vstate);
    m.op_ga = guide(A, m.b_vel_a, m.b_astate);
    return m.op_fnv && m.op_fna && m.op_gv && m.op_ga && m.op_fv && m.op_fa && m.op_rnd_v &&
                   m.op_rnd_a && m.op_keep_first && m.op_lv && m.op_la
               ? RAD_OK
               : RAD_E_INVAL;
}

/* ================================================================== the video VAE */
/* A residual block's ops at C channels over voxel rows (`rows` at most): PerChannelRMSNorm with its
 * SiLU, conv1, again, conv2 with the input added. `tpad` is the frames' padding (the decoder's
 * replicate, the encoder's causal). */
static int vres_declare(RadBuilder* b, Model& m, VRes& Rb, const char* dn, const char* src, int64_t C,
                        int64_t rows, const char* tpad, int group, rad_buf x, rad_buf t, rad_buf h) {
    Names& nm = m.nm;
    const int acc = RAD_ACCESS_PER_TOKEN;
    Rb.C = C;
    Rb.w1 = wcopy(b, nm.f("%s.conv1.weight", dn), nm.ckpt("%s.conv1.conv.weight", src), RAD_BF16,
                  { C, C, 3, 3, 3 }, acc, group);
    Rb.b1 = wcopy(b, nm.f("%s.conv1.bias", dn), nm.ckpt("%s.conv1.conv.bias", src), RAD_BF16, { C },
                  acc, group);
    Rb.w2 = wcopy(b, nm.f("%s.conv2.weight", dn), nm.ckpt("%s.conv2.conv.weight", src), RAD_BF16,
                  { C, C, 3, 3, 3 }, acc, group);
    Rb.b2 = wcopy(b, nm.f("%s.conv2.bias", dn), nm.ckpt("%s.conv2.conv.bias", src), RAD_BF16, { C },
                  acc, group);
    if (!Rb.w1 || !Rb.b1 || !Rb.w2 || !Rb.b2) return RAD_E_INVAL;
    auto norm = [&](rad_buf in, rad_buf out) {
        return rw(b, RAD_OP(b, "pixel_norm",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", C), RAD_F64("eps", 1e-8),
                                       RAD_STR("form", "div"), RAD_STR("act", "silu"),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { in }, { out });
    };
    auto conv = [&](rad_weight w, rad_weight bb, rad_buf in, rad_buf res, rad_buf out) {
        const rad_op op = RAD_OP(b, "conv3d_ndhwc",
                                 RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("cin", C),
                                            RAD_INT("cout", C), RAD_INT("k", 3), RAD_INT("kt", 3),
                                            RAD_STR("dtype", "bf16"), RAD_STR("tpad", tpad)),
                                 RAD_WEIGHTS(w, bb));
        return res ? rw(b, op, { in, res }, { out }) : rw(b, op, { in }, { out });
    };
    Rb.op_n1 = norm(x, t);
    Rb.op_c1 = conv(Rb.w1, Rb.b1, t, 0, h);
    Rb.op_n2 = norm(h, t);
    Rb.op_c2 = conv(Rb.w2, Rb.b2, t, x, x);
    return Rb.op_n1 && Rb.op_c1 && Rb.op_n2 && Rb.op_c2 ? RAD_OK : RAD_E_INVAL;
}

/* A dense [rows, cols] view of an arena buffer, at element offset `off` (the VAE's buffers hold
 * a different width at each scale). */
static RadOperand view(RadCtx* c, rad_buf h, uint32_t dt, int64_t rows, int64_t cols, int64_t off = 0) {
    const int64_t es = dt == RAD_F32 ? 4 : dt == RAD_U8 ? 1 : 2;
    return RAD_P_T2((const char*)rad_buf_ptr(c, h) + off * es, dt, rows, cols);
}

static void vres_step(RadCtx* c, const VRes& Rb, int64_t vox, const RadOperand& geom, rad_buf x,
                      rad_buf t, rad_buf h) {
    const int64_t C = Rb.C;
    const RadOperand X = view(c, x, RAD_BF16, vox, C), Tt = view(c, t, RAD_BF16, vox, C);
    const RadOperand Hh = view(c, h, RAD_BF16, vox, C);
    RAD_ISSUE_N(c, Rb.op_n1, vox, X, Tt);
    RAD_ISSUE_N(c, Rb.op_c1, vox, Tt, RAD_W(Rb.w1), RAD_W(Rb.b1), geom, RAD_NONE, Hh);
    RAD_ISSUE_N(c, Rb.op_n2, vox, Hh, Tt);
    RAD_ISSUE_N(c, Rb.op_c2, vox, Tt, RAD_W(Rb.w2), RAD_W(Rb.b2), geom, X, X);
}

/* The decoder (LTX2VideoDecoder3d, non-causal): conv_in, the mid block, four up blocks (the
 * upsampler's convolution and depth to space, then residual blocks), PerChannelRMSNorm with SiLU,
 * conv_out to 4 x 4 pixel patches. The whole video at once, a tile at a time (vae_tiles: one for a
 * frame of kVaeLat latents or fewer), into the lane's tile store; each chunk's frames stitched to
 * bytes from it. */
static int declare_vae(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    const int acc = RAD_ACCESS_PER_TOKEN, gx = kGrpVae;
    std::vector<double> boc = meta_list(meta, "vae.decoder_block_out_channels");
    std::vector<double> lpb = meta_list(meta, "vae.decoder_layers_per_block");
    std::vector<double> uf = meta_list(meta, "vae.upsample_factor");
    const char* types = rad_meta_gets(meta, "vae.upsample_type", "");
    std::vector<double> sts = meta_list(meta, "vae.decoder_spatio_temporal_scaling");
    std::vector<std::string> ut;
    for (const char* s = types; s && *s;) {
        while (*s == ' ') ++s;
        const char* e = s;
        while (*e && *e != ' ') ++e;
        if (e > s) ut.emplace_back(s, e);
        s = e;
    }
    const size_t nup = boc.size();
    if (nup == 0 || lpb.size() != nup + 1 || uf.size() != nup || ut.size() != nup ||
        sts.size() != nup) {
        fprintf(stderr, "radiance: ltx2: the VAE decoder's block lists disagree in length\n");
        return RAD_E_UNSUPPORTED;
    }
    std::reverse(boc.begin(), boc.end());
    std::reverse(lpb.begin(), lpb.end());
    std::reverse(uf.begin(), uf.end());
    std::reverse(sts.begin(), sts.end());

    /* The volumes: latent (F, h, w) to (At F + Bt, Ah h, Aw w) after each up block; the scratch
     * sized to the largest volume times channels any step of the network holds. */
    const int64_t F = max_lat_frames(), HW = kVaeLat;
    int64_t At = 1, Bt = 0, Ahw = 1, C = (int64_t)boc[0], big = F * HW * C;
    m.vC0 = C;
    m.vup.resize(nup);
    for (size_t i = 0; i < nup; ++i) {
        VUp& U = m.vup[i];
        U.cout = (int64_t)boc[i] / (int64_t)uf[i];
        U.cin = C;
        if (!sts[i]) {
            fprintf(stderr, "radiance: ltx2: an up block without its upsampler\n");
            return RAD_E_UNSUPPORTED;
        }
        if (ut[i] == "spatiotemporal") { U.st = 2; U.sh = 2; U.sw = 2; }
        else if (ut[i] == "temporal") { U.st = 2; U.sh = 1; U.sw = 1; }
        else if (ut[i] == "spatial") { U.st = 1; U.sh = 2; U.sw = 2; }
        else return RAD_E_UNSUPPORTED;
        if (U.cout * (int64_t)uf[i] != C) {
            fprintf(stderr, "radiance: ltx2: up block %zu would take %lld channels and has %lld; a "
                            "block that changes width first is not run\n", i,
                    (long long)(U.cout * (int64_t)uf[i]), (long long)C);
            return RAD_E_UNSUPPORTED;
        }
        U.conv_out = C * U.st * U.sh * U.sw / (int64_t)uf[i];
        big = std::max(big, (At * F + Bt) * Ahw * HW * U.conv_out);
        At *= U.st;
        Bt = Bt * U.st - (U.st - 1);
        Ahw *= U.sh * U.sw;
        C = U.cout;
        big = std::max(big, (At * F + Bt) * Ahw * HW * C);
    }
    const int64_t vox_out = (At * F + Bt) * Ahw * HW;
    {
        int64_t side = 1;
        while (side * side < Ahw) ++side;
        if (side * side != Ahw) return RAD_E_UNSUPPORTED;
        /* The causal mapping tile_stitch reads: the first latent frame one frame. */
        if (Bt != 1 - At) return RAD_E_UNSUPPORTED;
        m.vAt = At;
        m.vBt = Bt;
        m.vAhw = side;
    }
    /* The tile store holds the largest video's tiles: every size a request may ask for, at the
     * most frames a step carries at that size. */
    for (int64_t Wd = kMinSide; Wd * kMinSide <= kMaxArea; Wd += kMult)
        for (int64_t Hd = kMinSide; Wd * Hd <= kMaxArea; Hd += kMult) {
            if (std::max(Wd, Hd) > kMaxAspect * std::min(Wd, Hd)) continue;
            const int64_t h = Hd / kScale, w = Wd / kScale;
            const int64_t Fl = std::min(max_lat_frames(), max_video_rows() / (h * w));
            if (Fl < (int64_t)kLpc * kMinChunks + kLtail) continue;
            const std::vector<VaeTile> tl = vae_tiles(h, w);
            int64_t area = 0;
            for (const VaeTile& t : tl) area += t.h * t.w;
            m.maxStore = std::max(m.maxStore, (At * Fl + Bt) * Ahw * area);
            m.maxTiles = std::max(m.maxTiles, (int64_t)tl.size());
            m.maxSide = std::max(m.maxSide, std::max(Wd, Hd));
        }
    /* Every tile's crop reads the latent again after earlier tiles' decodes, which the planner's
     * declaration order cannot see: kept. */
    m.b_vlat = decl_b(b, nm.f("vae.lat"), RAD_BF16, { m.maxS, kC }, RAD_BUF_PERSIST);
    m.b_vtlat = decl_b(b, nm.f("vae.tile_lat"), RAD_BF16, { F * HW, kC });
    m.b_va = decl_b(b, nm.f("vae.a"), RAD_BF16, { big });
    m.b_vb = decl_b(b, nm.f("vae.b"), RAD_BF16, { big });
    m.b_vc = decl_b(b, nm.f("vae.c"), RAD_BF16, { big });
    m.b_vtiles = decl_b(b, nm.f("lane.tiles"), RAD_BF16, { m.maxStore, kPixC }, RAD_BUF_PERSIST);
    /* The boxes are read through raw pointers the planner does not see: kept. */
    m.b_vgrid = decl_b(b, nm.f("vae.grid"), RAD_I32, { m.maxTiles, 8 }, RAD_BUF_PERSIST);
    m.b_vf0 = decl_b(b, nm.f("vae.first_frame"), RAD_I32, { 1 });
    m.vd_mean = wcopy(b, nm.f("vae.latents_mean"), nm.ckpt("vae.latents_mean"), RAD_BF16, { kC },
                      acc, gx);
    m.vd_std = wcopy(b, nm.f("vae.latents_std"), nm.ckpt("vae.latents_std"), RAD_BF16, { kC }, acc, gx);
    if (!m.vd_mean || !m.vd_std) return RAD_E_INVAL;
    /* The latent to bf16 first, then denormalised in bf16, as the pipeline casts it. */
    m.op_vlat = cast_op(b, m.maxS, kC, "f32", "bf16", m.b_vstate, m.b_vlat);
    m.op_vden = rw(b, RAD_OP(b, "chan_stats",
                             RAD_PARAMS(RAD_RANGE("M", 1, m.maxS), RAD_INT("n", kC),
                                        RAD_STR("mode", "denorm"), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(m.vd_mean, m.vd_std)),
                   { m.b_vlat }, { m.b_vlat });
    auto conv = [&](rad_weight* w, rad_weight* bb, const char* dn, const char* src, int64_t cin,
                    int64_t cout, int64_t rows, rad_buf in, rad_buf out) -> rad_op {
        *w = wcopy(b, nm.f("%s.weight", dn), nm.ckpt("%s.conv.weight", src), RAD_BF16,
                   { cout, cin, 3, 3, 3 }, acc, gx);
        *bb = wcopy(b, nm.f("%s.bias", dn), nm.ckpt("%s.conv.bias", src), RAD_BF16, { cout }, acc, gx);
        if (!*w || !*bb) return 0;
        return rw(b, RAD_OP(b, "conv3d_ndhwc",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("cin", cin),
                                       RAD_INT("cout", cout), RAD_INT("k", 3), RAD_INT("kt", 3),
                                       RAD_STR("dtype", "bf16"), RAD_STR("tpad", "replicate")),
                            RAD_WEIGHTS(*w, *bb)),
                  { in }, { out });
    };
    auto grid = [&](int64_t tile, int64_t stride) {
        return rw(b, RAD_OP(b, "tile_grid",
                            RAD_PARAMS(RAD_RANGE("M", 1, m.maxTiles), RAD_INT("tile", tile),
                                       RAD_INT("stride", stride)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_vgrid });
    };
    /* One tile: a tile_grid row of the whole frame. */
    m.op_vgrid1 = grid(1 << 20, 1 << 20);
    m.op_vgrid = grid(kTile, kTileStride);
    m.op_vcrop = rw(b, RAD_OP(b, "tile_crop",
                              RAD_PARAMS(RAD_RANGE("M", 1, F * HW), RAD_INT("C", kC),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_vlat }, { m.b_vtlat });
    m.op_vcin = conv(&m.w_vcin, &m.b_vcin, "vae.dec.conv_in", "vae.decoder.conv_in", kC, m.vC0, F * HW,
                     m.b_vtlat, m.b_va);
    const int64_t nmid = (int64_t)lpb[0];
    m.vmid.resize((size_t)nmid);
    for (int64_t r = 0; r < nmid; ++r)
        RAD_ARCH_TRY(vres_declare(b, m, m.vmid[(size_t)r], nm.f("vae.dec.mid.%lld", (long long)r),
                                  nm.ckpt("vae.decoder.mid_block.resnets.%lld", (long long)r), m.vC0,
                                  m.maxS, "replicate", gx, m.b_va, m.b_vb, m.b_vc));
    At = 1;
    Bt = 0;
    int64_t Ah = 1, Aw = 1;
    for (size_t i = 0; i < nup; ++i) {
        VUp& U = m.vup[i];
        const int64_t vin = (At * F + Bt) * Ah * Aw * HW;
        At *= U.st;
        Bt = Bt * U.st - (U.st - 1);
        Ah *= U.sh;
        Aw *= U.sw;
        const int64_t vout = (At * F + Bt) * Ah * Aw * HW;
        const char* dn = nm.f("vae.dec.up.%zu", i);
        const char* src = nm.ckpt("vae.decoder.up_blocks.%zu", i);
        U.op_conv = conv(&U.w, &U.b, nm.f("%s.conv", dn), nm.ckpt("%s.upsamplers.0.conv", src), U.cin,
                         U.conv_out, vin, m.b_va, m.b_vb);
        U.op_d2s = rw(b, RAD_OP(b, "d2s3d",
                                RAD_PARAMS(RAD_RANGE("M", 1, vin), RAD_INT("C", U.cout),
                                           RAD_INT("st", U.st), RAD_INT("sh", U.sh), RAD_INT("sw", U.sw),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_vb }, { m.b_va });
        /* Its output volume on the device, off the latent's (kept, so every reader finds it). */
        U.geom = decl_b(b, nm.f("%s.geom", dn), RAD_I32, { 3 }, RAD_BUF_PERSIST);
        U.op_geom = rw(b, RAD_OP(b, "int_affine",
                                 RAD_PARAMS(RAD_RANGE("M", 1, 3), RAD_STR("src", "0 1 2"),
                                            RAD_STR("mul", ints({ At, Ah, Aw }).c_str()),
                                            RAD_STR("add", ints({ Bt, 0, 0 }).c_str())),
                                 RAD_NOWEIGHTS),
                       {}, { U.geom });
        if (!U.op_conv || !U.op_d2s || !U.op_geom) return RAD_E_INVAL;
        const int64_t nres = (int64_t)lpb[i + 1];
        U.res.resize((size_t)nres);
        for (int64_t r = 0; r < nres; ++r)
            RAD_ARCH_TRY(vres_declare(b, m, U.res[(size_t)r], nm.f("%s.res.%lld", dn, (long long)r),
                                      nm.ckpt("%s.resnets.%lld", src, (long long)r), U.cout, vout,
                                      "replicate", gx, m.b_va, m.b_vb, m.b_vc));
    }
    m.op_vnout = rw(b, RAD_OP(b, "pixel_norm",
                              RAD_PARAMS(RAD_RANGE("M", 1, vox_out), RAD_INT("n", C),
                                         RAD_F64("eps", 1e-8),
                                         RAD_STR("form", "div"), RAD_STR("act", "silu"),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_va }, { m.b_vb });
    m.op_vcout = conv(&m.w_vcout, &m.b_vcout, "vae.dec.conv_out", "vae.decoder.conv_out", C, kPixC,
                      vox_out, m.b_vb, m.b_vtiles);
    /* A chunk's frames to bytes: 4 x 4 patches, the column offset outer (LTX-2's patchifier), the
     * tiles blended where they overlap; the chunk's first frame off the pass's geometry. */
    m.b_frames = decl_b(b, nm.f("frames_out"), RAD_U8, { (int64_t)(kFpc + kFtail) * kMaxArea, 3 },
                        RAD_BUF_PERSIST);
    m.op_vf0 = rw(b, RAD_OP(b, "int_affine",
                            RAD_PARAMS(RAD_RANGE("M", 1, 1),
                                       RAD_STR("src", ints({ RAD_VGEOM_CHUNK_LAT0 }).c_str()),
                                       RAD_STR("mul", ints({ kFpc / kLpc }).c_str()),
                                       RAD_STR("add", "0")),
                            RAD_NOWEIGHTS),
                  {}, { m.b_vf0 });
    auto stitch = [&](int64_t tile, int64_t stride) {
        return rw(b, RAD_OP(b, "tile_stitch",
                            RAD_PARAMS(RAD_RANGE("M", 1, (int64_t)(kFpc + kFtail) * m.maxSide),
                                       RAD_INT("tile", tile), RAD_INT("stride", stride),
                                       RAD_INT("scale", m.vAhw), RAD_INT("scale_t", m.vAt),
                                       RAD_INT("p", kPix), RAD_INT("C", 3), RAD_STR("order", "wh"),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { m.b_vtiles, m.b_vf0 }, { m.b_frames });
    };
    m.op_vst1 = stitch(1 << 20, 1 << 20);
    m.op_vst = stitch(kTile, kTileStride);
    return m.op_vlat && m.op_vden && m.op_vcin && m.op_vnout && m.op_vcout && m.op_vgrid1 &&
                   m.op_vgrid && m.op_vcrop && m.op_vf0 && m.op_vst1 && m.op_vst
               ? RAD_OK
               : RAD_E_INVAL;
}

/* The encoder (LTX2VideoEncoder3d, causal) over one frame: the image in 4 x 4 patches, conv_in,
 * four down blocks (residual blocks, then the downsampler: its convolution over the volume with
 * its first st - 1 frames prepended, space to depth, the input's group mean added), the mid block,
 * PerChannelRMSNorm with SiLU, conv_out; the mean's 128 channels normalised into the lane's clean
 * first latent frame. A frame stays one frame through every temporal downsample. */
static int declare_venc(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    const int acc = RAD_ACCESS_PER_REQUEST, gx = kGrpVenc;
    const std::vector<double> boc = meta_list(meta, "vae.block_out_channels");
    const std::vector<double> lpb = meta_list(meta, "vae.layers_per_block");
    const std::vector<double> sts = meta_list(meta, "vae.spatio_temporal_scaling");
    const char* types = rad_meta_gets(meta, "vae.downsample_type", "");
    std::vector<std::string> dt;
    for (const char* s = types; s && *s;) {
        while (*s == ' ') ++s;
        const char* e = s;
        while (*e && *e != ' ') ++e;
        if (e > s) dt.emplace_back(s, e);
        s = e;
    }
    const size_t nd = boc.size();
    if (nd == 0 || lpb.size() != nd + 1 || dt.size() != nd || sts.size() != nd) {
        fprintf(stderr, "radiance: ltx2: the VAE encoder's block lists disagree in length\n");
        return RAD_E_UNSUPPORTED;
    }
    m.venc = true;
    const int64_t HW = m.maxPpf;                /* latents a frame at most */
    int64_t Ahw = 64, C = kC, big = Ahw * HW * kPixC;
    m.eC0 = C;
    m.vdown.resize(nd);
    /* The volumes: input (1, 8 h, 8 w) in latent units; the conv's (1 + tpre) frames. */
    for (size_t i = 0; i < nd; ++i) {
        VDown& Dn = m.vdown[i];
        Dn.cin = C;
        Dn.cout = (int64_t)boc[i];
        if (!sts[i]) return RAD_E_UNSUPPORTED;
        if (dt[i] == "spatiotemporal") { Dn.st = 2; Dn.sh = 2; Dn.sw = 2; }
        else if (dt[i] == "temporal") { Dn.st = 2; Dn.sh = 1; Dn.sw = 1; }
        else if (dt[i] == "spatial") { Dn.st = 1; Dn.sh = 2; Dn.sw = 2; }
        else return RAD_E_UNSUPPORTED;
        const int64_t P = Dn.st * Dn.sh * Dn.sw;
        Dn.conv_out = Dn.cout / P;
        Dn.group = Dn.cin * P / Dn.cout;
        if (Dn.conv_out * P != Dn.cout || Dn.group * Dn.cout != Dn.cin * P) return RAD_E_UNSUPPORTED;
        big = std::max(big, Ahw * HW * C);
        big = std::max(big, Dn.st * Ahw * HW * Dn.conv_out);
        Ahw /= Dn.sh * Dn.sw;
        C = Dn.cout;
        big = std::max(big, Ahw * HW * C);
    }
    if (Ahw != 1) {
        fprintf(stderr, "radiance: ltx2: the VAE encoder does not reach the latent's 32-pixel scale\n");
        return RAD_E_UNSUPPORTED;
    }
    m.eCout = C;
    big = std::max(big, HW * (kC + 1));
    m.b_epatch = decl_b(b, nm.f("venc.patch"), RAD_BF16, { 64 * HW, kPixC });
    m.b_ea = decl_b(b, nm.f("venc.a"), RAD_BF16, { big });
    m.b_eb = decl_b(b, nm.f("venc.b"), RAD_BF16, { big });
    m.b_ec = decl_b(b, nm.f("venc.c"), RAD_BF16, { big });
    m.b_eout = decl_b(b, nm.f("venc.out"), RAD_BF16, { HW, kC + 1 });
    m.b_elat = decl_b(b, nm.f("venc.lat"), RAD_BF16, { HW, kC });
    m.op_epatch = rw(b, RAD_OP(b, "pix_patch",
                               RAD_PARAMS(RAD_RANGE("M", 1, 8 * HW), RAD_INT("p", kPix), RAD_INT("C", 3),
                                          RAD_STR("order", "wh"), RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     {}, { m.b_epatch });
    auto geom = [&](rad_buf* g, rad_op* op, const char* name, int64_t frames, int64_t mul) {
        *g = decl_b(b, nm.f("venc.%s", name), RAD_I32, { 3 }, RAD_BUF_PERSIST);
        *op = rw(b, RAD_OP(b, "int_affine",
                           RAD_PARAMS(RAD_RANGE("M", 1, 3), RAD_STR("src", "-1 1 2"),
                                      RAD_STR("mul", ints({ 0, mul, mul }).c_str()),
                                      RAD_STR("add", ints({ frames, 0, 0 }).c_str())),
                           RAD_NOWEIGHTS),
                 {}, { *g });
        return *g && *op ? RAD_OK : RAD_E_INVAL;
    };
    RAD_ARCH_TRY(geom(&m.b_egeomL, &m.op_egeomL, "geom_mid", 1, 1));
    auto conv = [&](rad_weight* w, rad_weight* bb, const char* dn, const char* src, int64_t cin,
                    int64_t cout, int64_t rows, int64_t tpre, rad_buf in, rad_buf out) -> rad_op {
        *w = wcopy(b, nm.f("%s.weight", dn), nm.ckpt("%s.conv.weight", src), RAD_BF16,
                   { cout, cin, 3, 3, 3 }, acc, gx);
        *bb = wcopy(b, nm.f("%s.bias", dn), nm.ckpt("%s.conv.bias", src), RAD_BF16, { cout }, acc, gx);
        if (!*w || !*bb) return 0;
        return rw(b, RAD_OP(b, "conv3d_ndhwc",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("cin", cin),
                                       RAD_INT("cout", cout), RAD_INT("k", 3), RAD_INT("kt", 3),
                                       RAD_STR("dtype", "bf16"), RAD_STR("tpad", "causal"),
                                       RAD_INT("tpre", tpre)),
                            RAD_WEIGHTS(*w, *bb)),
                  { in }, { out });
    };
    m.op_ecin = conv(&m.w_ecin, &m.b_ecin, "venc.conv_in", "vae.encoder.conv_in", kPixC, kC, 64 * HW, 0,
                     m.b_epatch, m.b_ea);
    if (!m.op_ecin) return RAD_E_INVAL;
    Ahw = 64;
    for (size_t i = 0; i < nd; ++i) {
        VDown& Dn = m.vdown[i];
        const char* dn = nm.f("venc.down.%zu", i);
        const char* src = nm.ckpt("vae.encoder.down_blocks.%zu", i);
        const int64_t vin = Ahw * HW, nres = (int64_t)lpb[i];
        Dn.res.resize((size_t)nres);
        for (int64_t r = 0; r < nres; ++r)
            RAD_ARCH_TRY(vres_declare(b, m, Dn.res[(size_t)r], nm.f("%s.res.%lld", dn, (long long)r),
                                      nm.ckpt("%s.resnets.%lld", src, (long long)r), Dn.cin, vin,
                                      "causal", gx, m.b_ea, m.b_eb, m.b_ec));
        /* The residual's group mean into b, the convolution into c, its space to depth added. */
        const int64_t tpre = Dn.st - 1, sm = (int64_t)std::lround(std::sqrt((double)Ahw));
        Dn.op_mean = rw(b, RAD_OP(b, "s2d3d",
                                  RAD_PARAMS(RAD_RANGE("M", 1, vin), RAD_INT("C", Dn.cin),
                                             RAD_INT("st", Dn.st), RAD_INT("sh", Dn.sh),
                                             RAD_INT("sw", Dn.sw), RAD_INT("tpre", tpre),
                                             RAD_INT("group", Dn.group), RAD_STR("mode", "mean"),
                                             RAD_INT("add", 0), RAD_STR("dtype", "bf16")),
                                  RAD_NOWEIGHTS),
                        { m.b_ea }, { m.b_eb });
        Dn.op_conv = conv(&Dn.w, &Dn.b, nm.f("%s.conv", dn), nm.ckpt("%s.downsamplers.0.conv", src),
                          Dn.cin, Dn.conv_out, vin, tpre, m.b_ea, m.b_ec);
        Dn.op_s2d = rw(b, RAD_OP(b, "s2d3d",
                                 RAD_PARAMS(RAD_RANGE("M", 1, Dn.st * vin), RAD_INT("C", Dn.conv_out),
                                            RAD_INT("st", Dn.st), RAD_INT("sh", Dn.sh),
                                            RAD_INT("sw", Dn.sw), RAD_STR("mode", "copy"),
                                            RAD_INT("add", 1), RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       { m.b_ec, m.b_eb }, { m.b_eb });
        /* The block's input volume and its convolution's ((1 + tpre) frames at this scale). */
        RAD_ARCH_TRY(geom(&Dn.geom_in, &Dn.op_geom_in, nm.f("geom_in%zu", i), 1, sm));
        RAD_ARCH_TRY(geom(&Dn.geom, &Dn.op_geom, nm.f("geom_conv%zu", i), 1 + tpre, sm));
        Ahw /= Dn.sh * Dn.sw;
        /* The block's output (in b) back into the stream (a). */
        Dn.op_next = cast_op(b, Ahw * HW, Dn.cout, "bf16", "bf16", m.b_eb, m.b_ea);
        if (!Dn.op_mean || !Dn.op_conv || !Dn.op_s2d || !Dn.op_next) return RAD_E_INVAL;
    }
    const int64_t nmid = (int64_t)lpb[nd];
    m.emid.resize((size_t)nmid);
    for (int64_t r = 0; r < nmid; ++r)
        RAD_ARCH_TRY(vres_declare(b, m, m.emid[(size_t)r], nm.f("venc.mid.%lld", (long long)r),
                                  nm.ckpt("vae.encoder.mid_block.resnets.%lld", (long long)r), C, HW,
                                  "causal", gx, m.b_ea, m.b_eb, m.b_ec));
    m.op_enout = rw(b, RAD_OP(b, "pixel_norm",
                              RAD_PARAMS(RAD_RANGE("M", 1, HW), RAD_INT("n", C), RAD_F64("eps", 1e-8),
                                         RAD_STR("form", "div"), RAD_STR("act", "silu"),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_ea }, { m.b_eb });
    m.op_ecout = conv(&m.w_ecout, &m.b_ecout, "venc.conv_out", "vae.encoder.conv_out", C, kC + 1, HW, 0,
                      m.b_eb, m.b_eout);
    m.op_enorm = rw(b, RAD_OP(b, "chan_stats",
                              RAD_PARAMS(RAD_RANGE("M", 1, HW), RAD_INT("n", kC),
                                         RAD_STR("mode", "norm"),
                                         RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(m.vd_mean, m.vd_std)),
                    { m.b_eout }, { m.b_elat });
    m.op_ef32 = cast_op(b, HW, kC, "bf16", "f32", m.b_elat, m.b_first);
    return m.op_epatch && m.op_enout && m.op_ecout && m.op_enorm && m.op_ef32 && m.op_egeomL
               ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the soundtrack */
/* An audio VAE residual block over an image of (time) rows x `w` mel columns: PixelNorm with SiLU,
 * conv1 (causal along time), again, conv2, the input or its 1 x 1 shortcut added. */
static int ares_declare(RadBuilder* b, Model& m, ARes& Rb, const char* dn, const char* src,
                        int64_t cin, int64_t cout, int64_t px, rad_buf x, rad_buf t, rad_buf h,
                        rad_buf s) {
    Names& nm = m.nm;
    const int acc = RAD_ACCESS_PER_TOKEN, gx = kGrpAudio;
    Rb.cin = cin;
    Rb.cout = cout;
    auto conv = [&](rad_weight* w, rad_weight* bb, const char* name, int64_t ci, int64_t co, int64_t k,
                    rad_buf in, rad_buf out) -> rad_op {
        *w = wcopy(b, nm.f("%s.%s.weight", dn, name), nm.ckpt("%s.%s.conv.weight", src, name), RAD_BF16,
                   { co, ci, k, k }, acc, gx);
        *bb = wcopy(b, nm.f("%s.%s.bias", dn, name), nm.ckpt("%s.%s.conv.bias", src, name), RAD_BF16,
                    { co }, acc, gx);
        if (!*w || !*bb) return 0;
        return rw(b, RAD_OP(b, "conv2d_nhwc",
                            RAD_PARAMS(RAD_RANGE("M", 1, px), RAD_INT("cin", ci), RAD_INT("cout", co),
                                       RAD_INT("k", k), RAD_INT("stride", 1),
                                       RAD_STR("pad", k > 1 ? "causal" : "same"), RAD_INT("up", 1),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(*w, *bb)),
                  { in }, { out });
    };
    auto norm = [&](int64_t n, rad_buf in, rad_buf out) {
        return rw(b, RAD_OP(b, "pixel_norm",
                            RAD_PARAMS(RAD_RANGE("M", 1, px), RAD_INT("n", n), RAD_F64("eps", 1e-6),
                                       RAD_STR("form", "div"), RAD_STR("act", "silu"),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { in }, { out });
    };
    Rb.op_n1 = norm(cin, x, t);
    Rb.op_c1 = conv(&Rb.w1, &Rb.b1, "conv1", cin, cout, 3, t, h);
    Rb.op_n2 = norm(cout, h, t);
    Rb.op_c2 = conv(&Rb.w2, &Rb.b2, "conv2", cout, cout, 3, t, h);
    if (cin != cout) Rb.op_sc = conv(&Rb.ws, &Rb.bs, "nin_shortcut", cin, cout, 1, x, s);
    Rb.op_add = add_op(b, px, cout, "bf16", cin != cout ? s : x, h, x);
    return Rb.op_n1 && Rb.op_c1 && Rb.op_n2 && Rb.op_c2 && (cin == cout || Rb.op_sc) && Rb.op_add
               ? RAD_OK : RAD_E_INVAL;
}

static void ares_step(RadCtx* c, const ARes& Rb, int64_t H, int64_t W, rad_buf x, rad_buf t,
                      rad_buf h, rad_buf s) {
    const int64_t px = H * W, ci = Rb.cin, co = Rb.cout;
    RAD_ISSUE_N(c, Rb.op_n1, px, view(c, x, RAD_BF16, px, ci), view(c, t, RAD_BF16, px, ci));
    RAD_ISSUE_N(c, Rb.op_c1, px, view(c, t, RAD_BF16, H, W * ci), RAD_W(Rb.w1), RAD_W(Rb.b1),
                view(c, h, RAD_BF16, H, W * co));
    RAD_ISSUE_N(c, Rb.op_n2, px, view(c, h, RAD_BF16, px, co), view(c, t, RAD_BF16, px, co));
    RAD_ISSUE_N(c, Rb.op_c2, px, view(c, t, RAD_BF16, H, W * co), RAD_W(Rb.w2), RAD_W(Rb.b2),
                view(c, h, RAD_BF16, H, W * co));
    if (Rb.op_sc) {
        RAD_ISSUE_N(c, Rb.op_sc, px, view(c, x, RAD_BF16, H, W * ci), RAD_W(Rb.ws), RAD_W(Rb.bs),
                    view(c, s, RAD_BF16, H, W * co));
        RAD_ISSUE_N(c, Rb.op_add, px, view(c, s, RAD_BF16, px, co), view(c, h, RAD_BF16, px, co),
                    view(c, x, RAD_BF16, px, co));
    } else {
        RAD_ISSUE_N(c, Rb.op_add, px, view(c, x, RAD_BF16, px, co), view(c, h, RAD_BF16, px, co),
                    view(c, x, RAD_BF16, px, co));
    }
}

/* A vocoder (LTX2Vocoder, f32) from `in` [rows, 128] to `V.wav` [rows x 160 or 240, 2]: conv_in,
 * each stage an upsampling transposed convolution and the mean of three resnets (kernel 3, 7 and
 * 11), each three dilations of anti-aliased SnakeBeta, dilated conv, again, conv and the residual;
 * the last SnakeBeta and conv_out without a bias. `mult`: its input rows a mel frame of the audio
 * VAE's output (the extender's input is at twice the rate). */
static int voc_declare(RadBuilder* b, const RadModelMeta* meta, Model& m, Voc& V, const char* dn,
                       const char* src, const char* cfg, int64_t mult, rad_buf in) {
    Names& nm = m.nm;
    const int acc = RAD_ACCESS_PER_TOKEN, gx = kGrpVoc;
    static const int64_t kDil[3] = { 1, 3, 5 };
    auto key = [&](const char* k) { return nm.f("vocoder.%s%s", cfg, k); };
    V.in = 2 * kMel;
    V.out = 2;
    V.mult = mult;
    V.in_buf = in;
    V.hidden = rad_meta_geti(meta, key("hidden_channels"), 0);
    for (double x : meta_list(meta, key("upsample_factors"))) V.rates.push_back((int64_t)x);
    for (double x : meta_list(meta, key("upsample_kernel_sizes"))) V.ks.push_back((int64_t)x);
    for (double x : meta_list(meta, key("resnet_kernel_sizes"))) V.rk.push_back((int64_t)x);
    const std::vector<double> dil = meta_list(meta, key("resnet_dilations"));
    const size_t ns = V.rates.size();
    bool ok = V.hidden > 0 && ns > 0 && V.ks.size() == ns && V.rk.size() == 3 && dil.size() == 9 &&
              (V.hidden >> ns) >= 1;
    for (size_t i = 0; ok && i < 9; ++i) ok = (int64_t)dil[i] == kDil[i % 3];
    if (!ok) {
        fprintf(stderr, "radiance: ltx2: the %svocoder is not a BigVGAN of three resnets a stage "
                        "at dilations 1, 3, 5 (hidden %lld, %zu stages, %zu kernels, %zu resnet "
                        "kernels, %zu dilations)\n", cfg, (long long)V.hidden, ns, V.ks.size(),
                V.rk.size(), dil.size());
        return RAD_E_UNSUPPORTED;
    }
    /* Rows a stage: (4 n - 3) mel frames times the upsampling so far. */
    std::vector<int64_t> rows(ns + 1);
    rows[0] = mel_frames(m.maxA) * mult;
    for (size_t s = 0; s < ns; ++s) rows[s + 1] = rows[s] * V.rates[s];
    auto cu = [&](int64_t f) {
        rad_buf y = decl_b(b, nm.f("%s.cu%lld", dn, (long long)f), RAD_I32, { 2 }, RAD_BUF_PERSIST);
        V.cu.push_back(y);
        return rw(b, RAD_OP(b, "int_affine",
                            RAD_PARAMS(RAD_RANGE("M", 1, 2),
                                       RAD_STR("src", ints({ -1, RAD_VGEOM_N_AUDIO }).c_str()),
                                       RAD_STR("mul", ints({ 0, 4 * f }).c_str()),
                                       RAD_STR("add", ints({ 0, -3 * f }).c_str())),
                            RAD_NOWEIGHTS),
                  {}, { y });
    };
    auto wt = [&](const char* name, std::initializer_list<int64_t> shape) {
        return wcopy(b, nm.f("%s.%s", dn, name), nm.ckpt("%s.%s", src, name), RAD_F32, shape, acc, gx);
    };
    auto conv1 = [&](VConv& K, const char* name, int64_t ci, int64_t co, int64_t k, int64_t d,
                     bool bias, int64_t R, rad_buf x, rad_buf cub, rad_buf res, rad_buf y) {
        K.w = wt(nm.f("%s.weight", name), { co, ci, k });
        if (bias) K.b = wt(nm.f("%s.bias", name), { co });
        if (!K.w || (bias && !K.b)) return RAD_E_INVAL;
        rad_op op = bias ? RAD_OP(b, "conv1d",
                                  RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("cin", ci),
                                             RAD_INT("cout", co),
                                             RAD_INT("k", k), RAD_INT("dil", d), RAD_STR("act", "none"),
                                             RAD_STR("dtype", "f32")),
                                  RAD_WEIGHTS(K.w, K.b))
                         : RAD_OP(b, "conv1d",
                                  RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("cin", ci),
                                             RAD_INT("cout", co),
                                             RAD_INT("k", k), RAD_INT("dil", d), RAD_STR("act", "none"),
                                             RAD_STR("dtype", "f32")),
                                  RAD_WEIGHTS(K.w));
        K.op = res ? rw(b, op, { x, cub, res }, { y }) : rw(b, op, { x, cub }, { y });
        return K.op ? RAD_OK : RAD_E_INVAL;
    };
    auto act = [&](VAct& a, const char* name, int64_t ch, int64_t R, rad_buf x, rad_buf cub, rad_buf y) {
        a.alpha = wt(nm.f("%s.act.alpha", name), { ch });
        a.beta = wt(nm.f("%s.act.beta", name), { ch });
        a.up = wt(nm.f("%s.upsample.filter", name), { 1, 1, 12 });
        a.down = wt(nm.f("%s.downsample.filter", name), { 1, 1, 12 });
        if (!a.alpha || !a.beta || !a.up || !a.down) return RAD_E_INVAL;
        a.op = rw(b, RAD_OP(b, "snake_aa",
                            RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", ch), RAD_INT("ratio", 2),
                                       RAD_INT("k", 12), RAD_INT("log", 1), RAD_STR("dtype", "f32")),
                            RAD_WEIGHTS(a.alpha, a.beta, a.up, a.down)),
                  { x, cub }, { y });
        return a.op ? RAD_OK : RAD_E_INVAL;
    };
    V.op_cu0 = cu(mult);
    V.h.resize(ns + 1);
    V.avg.resize(ns + 1);
    V.t1.resize(ns + 1);
    V.t2.resize(ns + 1);
    for (int j = 0; j < 3; ++j) V.amp[j].resize(ns + 1);
    V.avg[0] = decl_b(b, nm.f("%s.h0", dn), RAD_F32, { rows[0], V.hidden });
    RAD_ARCH_TRY(conv1(V.cin, "conv_in", V.in, V.hidden, 7, 1, true, rows[0], in, V.cu[0], 0, V.avg[0]));
    int64_t f = mult;
    V.st.resize(ns);
    for (size_t s = 0; s < ns; ++s) {
        VStage& St = V.st[s];
        const int64_t ci = V.hidden >> s, co = V.hidden >> (s + 1), R = rows[s + 1];
        St.ch = co;
        St.rate = V.rates[s];
        St.k = V.ks[s];
        f *= St.rate;
        St.op_cu = cu(f);
        V.h[s + 1] = decl_b(b, nm.f("%s.s%zu.in", dn, s), RAD_F32, { R, co });
        V.t1[s + 1] = decl_b(b, nm.f("%s.s%zu.t1", dn, s), RAD_F32, { R, co });
        V.t2[s + 1] = decl_b(b, nm.f("%s.s%zu.t2", dn, s), RAD_F32, { R, co });
        for (int j = 0; j < 3; ++j)
            V.amp[j][s + 1] = decl_b(b, nm.f("%s.s%zu.amp%d", dn, s, j), RAD_F32, { R, co });
        St.up.w = wt(nm.f("upsamplers.%zu.weight", s), { ci, co, St.k });
        St.up.b = wt(nm.f("upsamplers.%zu.bias", s), { co });
        if (!St.up.w || !St.up.b || !St.op_cu) return RAD_E_INVAL;
        St.up.op = rw(b, RAD_OP(b, "conv_t1d",
                                RAD_PARAMS(RAD_RANGE("M", 1, rows[s]), RAD_INT("cin", ci),
                                           RAD_INT("cout", co), RAD_INT("stride", St.rate),
                                           RAD_INT("k", St.k), RAD_INT("pad", (St.k - St.rate) / 2),
                                           RAD_STR("dtype", "f32")),
                                RAD_WEIGHTS(St.up.w, St.up.b)),
                      { V.avg[s], V.cu[s] }, { V.h[s + 1] });
        if (!St.up.op) return RAD_E_INVAL;
        for (int j = 0; j < 3; ++j) {
            const int ri = (int)(s * 3) + j;
            for (int d = 0; d < 3; ++d) {
                VUnit& U = St.u[j][d];
                const rad_buf x = d == 0 ? V.h[s + 1] : V.amp[j][s + 1];
                RAD_ARCH_TRY(act(U.a1, nm.f("resnets.%d.acts1.%d", ri, d), co, R, x, V.cu[s + 1],
                                 V.t1[s + 1]));
                RAD_ARCH_TRY(conv1(U.c1, nm.f("resnets.%d.convs1.%d", ri, d), co, co, V.rk[(size_t)j],
                                   kDil[d], true, R, V.t1[s + 1], V.cu[s + 1], 0, V.t2[s + 1]));
                RAD_ARCH_TRY(act(U.a2, nm.f("resnets.%d.acts2.%d", ri, d), co, R, V.t2[s + 1],
                                 V.cu[s + 1], V.t1[s + 1]));
                RAD_ARCH_TRY(conv1(U.c2, nm.f("resnets.%d.convs2.%d", ri, d), co, co, V.rk[(size_t)j], 1,
                                   true, R, V.t1[s + 1], V.cu[s + 1], x, V.amp[j][s + 1]));
            }
        }
        V.avg[s + 1] = decl_b(b, nm.f("%s.s%zu.out", dn, s), RAD_F32, { R, co });
        St.op_sum = add_op(b, R, co, "f32", V.amp[0][s + 1], V.amp[1][s + 1], V.avg[s + 1]);
        St.op_avg = rw(b, RAD_OP(b, "add",
                                 RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", co),
                                            RAD_STR("dtype", "f32"),
                                            RAD_F64("div", 3.0)),
                                 RAD_NOWEIGHTS),
                       { V.avg[s + 1], V.amp[2][s + 1] }, { V.avg[s + 1] });
        if (!St.op_sum || !St.op_avg) return RAD_E_INVAL;
    }
    const int64_t Rn = rows[ns], cn = V.hidden >> ns;
    V.wav = decl_b(b, nm.f("%s.wav", dn), RAD_F32, { Rn, V.out });
    RAD_ARCH_TRY(act(V.act, "act_out", cn, Rn, V.avg[ns], V.cu[ns], V.t1[ns]));
    return conv1(V.cout, "conv_out", cn, V.out, 7, 1, false, Rn, V.t1[ns], V.cu[ns], 0, V.wav);
}

/* The soundtrack: the latent denormalised (f32) and unpacked to an image of (time) x 16 mel bins
 * of 8 channels, the audio VAE's decoder (causal along time) to a stereo log-mel spectrogram,
 * channel-major [frames, 2 x 64]; the vocoder to 16 kHz; the bandwidth extender's log-mel of that,
 * its vocoder, and the sinc-resampled skip, to 48 kHz. */
static int declare_audio(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    const int acc = RAD_ACCESS_PER_TOKEN, gx = kGrpAudio;
    const int64_t A = m.maxA, Tm = mel_frames(A);
    const std::vector<double> mult = meta_list(meta, "audio_vae.ch_mult");
    const int64_t base = rad_meta_geti(meta, "audio_vae.base_channels", 0);
    const int64_t nrb = rad_meta_geti(meta, "audio_vae.num_res_blocks", 0);
    if (mult.size() != 3 || nrb <= 0 || base <= 0) {
        fprintf(stderr, "radiance: ltx2: the audio VAE is not three levels of residual blocks\n");
        return RAD_E_UNSUPPORTED;
    }
    m.aC0 = base * (int64_t)mult.back();
    m.b_aden = decl_b(b, nm.f("aud.den"), RAD_F32, { A, kAudioC });
    m.b_alat = decl_b(b, nm.f("aud.lat"), RAD_BF16, { A, kAudioC });
    m.b_aimg = decl_b(b, nm.f("aud.img"), RAD_BF16, { A * kLatMel, kLatCh });
    {
        rad_weight& mw = m.w_amean;
        rad_weight& sw = m.w_astd;
        mw = wcopy(b, nm.f("aud.latents_mean"), nm.ckpt("audio_vae.latents_mean"), RAD_F32, { kAudioC },
                   acc, gx);
        sw = wcopy(b, nm.f("aud.latents_std"), nm.ckpt("audio_vae.latents_std"), RAD_F32, { kAudioC },
                   acc, gx);
        if (!mw || !sw) return RAD_E_INVAL;
        m.op_aden = rw(b, RAD_OP(b, "chan_stats",
                                 RAD_PARAMS(RAD_RANGE("M", 1, A), RAD_INT("n", kAudioC),
                                            RAD_STR("mode", "denorm"), RAD_STR("dtype", "f32")),
                                 RAD_WEIGHTS(mw, sw)),
                       { m.b_astate }, { m.b_aden });
    }
    m.op_acast = cast_op(b, A, kAudioC, "f32", "bf16", m.b_aden, m.b_alat);
    m.op_aunpack = rw(b, RAD_OP(b, "d2s3d",
                                RAD_PARAMS(RAD_RANGE("M", 1, A), RAD_INT("C", kLatCh), RAD_INT("st", 1),
                                           RAD_INT("sh", 1), RAD_INT("sw", kLatMel),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_alat }, { m.b_aimg });
    /* The decoder's images (time x mel x channels), the largest the last upsample's output:
     * (4 A - 2) x 64 of level 1's channels. */
    const int64_t big = std::max<int64_t>(4 * A * kMel * base * (int64_t)mult[1],
                                          A * kLatMel * m.aC0 * 4);
    m.b_aa = decl_b(b, nm.f("aud.a"), RAD_BF16, { big });
    m.b_ab = decl_b(b, nm.f("aud.b"), RAD_BF16, { big });
    m.b_ac = decl_b(b, nm.f("aud.c"), RAD_BF16, { big });
    m.b_as = decl_b(b, nm.f("aud.s"), RAD_BF16, { big });
    m.b_amel16 = decl_b(b, nm.f("aud.mel16"), RAD_BF16, { Tm, 2 * kMel });
    m.b_amel = decl_b(b, nm.f("aud.mel"), RAD_F32, { Tm, 2 * kMel });
    const int64_t pxmax = 4 * A * kMel;
    auto conv = [&](rad_weight* w, rad_weight* bb, const char* dn, const char* src, int64_t ci,
                    int64_t co, int up, rad_buf in, rad_buf out, int row = -1) -> rad_op {
        const char* wn = nm.f("%s.weight", dn);
        const char* bn = nm.f("%s.bias", dn);
        if (row >= 0) {
            if (map_rows(b, wn, { { nm.ckpt("%s.conv.weight", src), row, 1 } }) < 0 ||
                map_rows(b, bn, { { nm.ckpt("%s.conv.bias", src), row, 1 } }) < 0)
                return 0;
            *w = decl_w(b, wn, RAD_BF16, { co, ci, 3, 3 }, acc, RAD_SHARD_NONE, grp_layer(gx));
            *bb = decl_w(b, bn, RAD_BF16, { co }, acc, RAD_SHARD_NONE, grp_layer(gx));
        } else {
            *w = wcopy(b, wn, nm.ckpt("%s.conv.weight", src), RAD_BF16, { co, ci, 3, 3 }, acc, gx);
            *bb = wcopy(b, bn, nm.ckpt("%s.conv.bias", src), RAD_BF16, { co }, acc, gx);
        }
        if (!*w || !*bb) return 0;
        return rw(b, RAD_OP(b, "conv2d_nhwc",
                            RAD_PARAMS(RAD_RANGE("M", 1, pxmax), RAD_INT("cin", ci), RAD_INT("cout", co),
                                       RAD_INT("k", 3), RAD_INT("stride", 1), RAD_STR("pad", "causal"),
                                       RAD_INT("up", up), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(*w, *bb)),
                  { in }, { out });
    };
    const char* dec = "audio_vae.decoder";
    m.op_acin = conv(&m.w_acin, &m.b_acin, "aud.conv_in", nm.ckpt("%s.conv_in", dec), kLatCh, m.aC0, 1,
                     m.b_aimg, m.b_aa);
    if (!m.op_acin) return RAD_E_INVAL;
    for (int i = 0; i < 2; ++i)
        RAD_ARCH_TRY(ares_declare(b, m, m.amid[i], nm.f("aud.mid.%d", i),
                                  nm.ckpt("%s.mid.block_%d", dec, i + 1), m.aC0, m.aC0, pxmax, m.b_aa,
                                  m.b_ab, m.b_ac, m.b_as));
    /* Levels 2, 1, 0: residual blocks, then (but after level 0) the nearest upsample, its causal
     * conv, and the rows past the first back into the stream. */
    int64_t cin = m.aC0;
    m.aup.resize(3);
    m.aups.resize(2);
    m.aups_cp.resize(2);
    m.alev_c.resize(3);
    for (int lev = 2; lev >= 0; --lev) {
        const int64_t co = base * (int64_t)mult[(size_t)lev];
        m.alev_c[(size_t)lev] = co;
        std::vector<ARes>& blk = m.aup[(size_t)(2 - lev)];
        blk.resize((size_t)(nrb + 1));
        for (int64_t i = 0; i <= nrb; ++i) {
            RAD_ARCH_TRY(ares_declare(b, m, blk[(size_t)i], nm.f("aud.up%d.%lld", lev, (long long)i),
                                      nm.ckpt("%s.up.%d.block.%lld", dec, lev, (long long)i), cin, co,
                                      pxmax, m.b_aa, m.b_ab, m.b_ac, m.b_as));
            cin = co;
        }
        if (lev > 0) {
            VConv& U = m.aups[(size_t)(2 - lev)];
            U.op = conv(&U.w, &U.b, nm.f("aud.up%d.upsample", lev),
                        nm.ckpt("%s.up.%d.upsample.conv", dec, lev),
                        co, co, 2, m.b_aa, m.b_ab);
            m.aups_cp[(size_t)(2 - lev)] = cast_op(b, pxmax, co, "bf16", "bf16", m.b_ab, m.b_aa);
            if (!U.op || !m.aups_cp[(size_t)(2 - lev)]) return RAD_E_INVAL;
        }
    }
    m.op_anout = rw(b, RAD_OP(b, "pixel_norm",
                              RAD_PARAMS(RAD_RANGE("M", 1, pxmax), RAD_INT("n", cin),
                                         RAD_F64("eps", 1e-6),
                                         RAD_STR("form", "div"), RAD_STR("act", "silu"),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_aa }, { m.b_ab });
    /* conv_out one output channel at a time, each into its 64 columns of the spectrogram: the
     * vocoder's channel-major layout (channel 64 + mel). */
    for (int ch = 0; ch < 2; ++ch) {
        m.op_acout[ch] = conv(&m.w_acout[ch], &m.b_acout[ch], nm.f("aud.conv_out%d", ch),
                              nm.ckpt("%s.conv_out", dec), cin, 1, 1, m.b_ab, m.b_amel16, ch);
        if (!m.op_acout[ch]) return RAD_E_INVAL;
    }
    m.op_amel = cast_op(b, Tm, 2 * kMel, "bf16", "f32", m.b_amel16, m.b_amel);

    /* The vocoder; the extender's log-mel of its waveform (rounded as the reference's bf16 one),
     * its vocoder, the skip and the clamp into the soundtrack. */
    RAD_ARCH_TRY(voc_declare(b, meta, m, m.voc, "voc", "vocoder.vocoder", "", 1, m.b_amel));
    const int64_t Sw = Tm * kMelHop;
    m.w_basis = wcopy(b, nm.f("voc.stft_basis"), nm.ckpt("vocoder.mel_stft.stft_fn.forward_basis"),
                      RAD_BF16, { 514, 1, 512 }, acc, kGrpVoc);
    m.w_melb = wcopy(b, nm.f("voc.mel_basis"), nm.ckpt("vocoder.mel_stft.mel_basis"), RAD_BF16,
                     { kMel, 257 }, acc, kGrpVoc);
    if (!m.w_basis || !m.w_melb) return RAD_E_INVAL;
    m.b_bmel16 = decl_b(b, nm.f("bwe.mel16"), RAD_BF16, { Sw / 80, 2 * kMel });
    m.b_bmel = decl_b(b, nm.f("bwe.mel"), RAD_F32, { Sw / 80, 2 * kMel });
    m.op_stft = rw(b, RAD_OP(b, "stft_logmel",
                             RAD_PARAMS(RAD_RANGE("M", 1, Sw / 80), RAD_INT("hop", 80),
                                        RAD_INT("nfft", 512), RAD_INT("n_mel", kMel),
                                        RAD_INT("ch", 2), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(m.w_basis, m.w_melb)),
                   { m.voc.wav }, { m.b_bmel16 });
    m.op_bmel = cast_op(b, Sw / 80, 2 * kMel, "bf16", "f32", m.b_bmel16, m.b_bmel);
    RAD_ARCH_TRY(voc_declare(b, meta, m, m.bwe, "bwe", "vocoder.bwe_generator", "bwe_", 2, m.b_bmel));
    /* The vocoders' f32 convolutions read f32 weights; a pipeline saved in bf16 holds them bf16,
     * and the recipes cast them back (`voc.*` and `bwe.*`, cast dtype=f32). */
    for (const char* w : { "voc.conv_in.weight", "bwe.conv_in.weight" }) {
        if (weight_codes_dtype(b, w, RAD_F32) != RAD_F32) {
            fprintf(stderr, "radiance: ltx2: '%s' is stored %s; the vocoders run in f32 -- convert "
                            "with one of data/recipes/ltx-2.5-*.recipe, which "
                            "cast them\n", w, rad_dtype_name(weight_codes_dtype(b, w, RAD_F32)));
            return RAD_E_UNSUPPORTED;
        }
    }
    m.b_audio = decl_b(b, nm.f("audio_out"), RAD_F32, { A * ((int64_t)kRate / kAudioRate), kAudioCh },
                       RAD_BUF_PERSIST);
    m.op_bwe = rw(b, RAD_OP(b, "bwe_up",
                            RAD_PARAMS(RAD_RANGE("M", 1, 3 * Sw), RAD_INT("ratio", 3), RAD_INT("ch", 2),
                                       RAD_STR("dtype", "f32")),
                            RAD_NOWEIGHTS),
                  { m.voc.wav, m.bwe.wav }, { m.b_audio });
    /* The samples past the decoder's (its causal frames are 3 short of 4 a latent): silence. */
    m.op_tail = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", 1, 3 * kMelHop * 3), RAD_INT("n", kAudioCh),
                                        RAD_STR("dtype", "f32"), RAD_F64("value", 0.0)),
                             RAD_NOWEIGHTS),
                   {}, { m.b_audio });
    return m.op_aden && m.op_acast && m.op_aunpack && m.op_anout && m.op_amel && m.op_stft &&
                   m.op_bmel && m.op_bwe && m.op_tail
               ? RAD_OK
               : RAD_E_INVAL;
}

/* ================================================================== the latent upsampler */
/* LTX2LatentUpsamplerModel (spatial x2, no rational resampler): the clean small latent
 * denormalised (f32, then bf16), initial_conv, GroupNorm and SiLU, residual blocks, each frame's
 * Conv2d to four times the channels and the pixel shuffle, residual blocks at the full size,
 * final_conv; normalised again (bf16) into the lane. */
static int declare_ups(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    if (!rad_meta_gets(meta, "latent_upsampler._class_name", nullptr)) return RAD_OK;
    Names& nm = m.nm;
    const int acc = RAD_ACCESS_PER_TOKEN, gx = kGrpUps;
    const int64_t C = rad_meta_geti(meta, "latent_upsampler.mid_channels", 0);
    const int64_t nb = rad_meta_geti(meta, "latent_upsampler.num_blocks_per_stage", 0);
    if (std::strcmp(rad_meta_gets(meta, "latent_upsampler._class_name", ""),
                    "LTX2LatentUpsamplerModel") ||
        rad_meta_geti(meta, "latent_upsampler.in_channels", 0) != kC ||
        rad_meta_geti(meta, "latent_upsampler.dims", 0) != 3 ||
        rad_meta_geti(meta, "latent_upsampler.spatial_upsample", 0) != 1 ||
        rad_meta_geti(meta, "latent_upsampler.temporal_upsample", 1) != 0 ||
        rad_meta_geti(meta, "latent_upsampler.use_rational_resampler", 1) != 0 || C <= 0 || C % 32 ||
        nb <= 0) {
        fprintf(stderr, "radiance: ltx2: latent_upsampler/ is not the spatial x2 upsampler this plugin "
                        "runs (3D, no rational resampler, GroupNorm over 32 groups)\n");
        return RAD_E_UNSUPPORTED;
    }
    m.ups = true;
    m.uC = C;
    /* the full size's voxels; the small size's are a quarter */
    const int64_t V = m.maxS;
    auto wt = [&](const char* name, std::initializer_list<int64_t> shape) {
        return wcopy(b, nm.f("ups.%s", name), nm.ckpt("latent_upsampler.%s", name), RAD_BF16, shape, acc,
                     gx);
    };
    auto conv = [&](rad_weight w, rad_weight bb, int64_t ci, int64_t co, int64_t rows, rad_buf x,
                    rad_buf y) -> rad_op {
        if (!w || !bb) return 0;
        return rw(b, RAD_OP(b, "conv3d_ndhwc",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("cin", ci), RAD_INT("cout", co),
                                       RAD_INT("k", 3), RAD_INT("kt", 3), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w, bb)),
                  { x }, { y });
    };
    m.b_ust = decl_b(b, nm.f("ups.gn_stats"), RAD_F32, { 32 * (2 + 3 * 1024) });
    auto gnorm = [&](rad_weight w, rad_weight bb, bool silu, int64_t rows, rad_buf x,
                     rad_buf y) -> rad_op {
        if (!w || !bb) return 0;
        return rw(b, RAD_OP(b, "group_norm",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", C), RAD_INT("groups", 32),
                                       RAD_F64("eps", 1e-5), RAD_STR("act", silu ? "silu" : "none"),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w, bb)),
                  { x, m.b_ust }, { y, m.b_ust });
    };
    m.b_uin = decl_b(b, nm.f("ups.in"), RAD_BF16, { V, kC });
    m.b_ua = decl_b(b, nm.f("ups.a"), RAD_BF16, { V, C });
    m.b_ub = decl_b(b, nm.f("ups.b"), RAD_BF16, { V, C });
    m.b_uc = decl_b(b, nm.f("ups.c"), RAD_BF16, { V, C });
    m.b_uout = decl_b(b, nm.f("ups.out"), RAD_BF16, { V, kC });
    m.b_uden = decl_b(b, nm.f("ups.den"), RAD_F32, { V, kC });
    m.op_uden = rw(b, RAD_OP(b, "chan_stats",
                             RAD_PARAMS(RAD_RANGE("M", 1, V), RAD_INT("n", kC),
                                        RAD_STR("mode", "denorm"),
                                        RAD_STR("dtype", "f32")),
                             RAD_WEIGHTS(m.vd_mean, m.vd_std)),
                   { m.b_vstate }, { m.b_uden });
    m.op_ucast = cast_op(b, V, kC, "f32", "bf16", m.b_uden, m.b_uin);
    m.u_cinw = wt("initial_conv.weight", { C, kC, 3, 3, 3 });
    m.u_cinb = wt("initial_conv.bias", { C });
    m.u_ninw = wt("initial_norm.weight", { C });
    m.u_ninb = wt("initial_norm.bias", { C });
    m.op_ucin = conv(m.u_cinw, m.u_cinb, kC, C, V, m.b_uin, m.b_ua);
    m.op_unin = gnorm(m.u_ninw, m.u_ninb, true, V, m.b_ua, m.b_ua);
    auto blocks = [&](const char* side, std::vector<URes>& list) {
        list.resize((size_t)nb);
        for (int64_t i = 0; i < nb; ++i) {
            URes& U = list[(size_t)i];
            const char* p = nm.f("%s.%lld", side, (long long)i);
            U.c1w = wt(nm.f("%s.conv1.weight", p), { C, C, 3, 3, 3 });
            U.c1b = wt(nm.f("%s.conv1.bias", p), { C });
            U.n1w = wt(nm.f("%s.norm1.weight", p), { C });
            U.n1b = wt(nm.f("%s.norm1.bias", p), { C });
            U.c2w = wt(nm.f("%s.conv2.weight", p), { C, C, 3, 3, 3 });
            U.c2b = wt(nm.f("%s.conv2.bias", p), { C });
            U.n2w = wt(nm.f("%s.norm2.weight", p), { C });
            U.n2b = wt(nm.f("%s.norm2.bias", p), { C });
            U.op_c1 = conv(U.c1w, U.c1b, C, C, V, m.b_ua, m.b_ub);
            U.op_n1 = gnorm(U.n1w, U.n1b, true, V, m.b_ub, m.b_ub);
            U.op_c2 = conv(U.c2w, U.c2b, C, C, V, m.b_ub, m.b_uc);
            U.op_n2 = gnorm(U.n2w, U.n2b, false, V, m.b_uc, m.b_uc);
            U.op_add = add_op(b, V, C, "bf16", m.b_uc, m.b_ua, m.b_ua);
            U.op_act = unary(b, "silu", V, C, "bf16", m.b_ua, m.b_ua);
            if (!U.op_c1 || !U.op_n1 || !U.op_c2 || !U.op_n2 || !U.op_add || !U.op_act)
                return RAD_E_INVAL;
        }
        return RAD_OK;
    };
    RAD_ARCH_TRY(blocks("res_blocks", m.u_pre));
    m.u_upw = wt("upsampler.0.weight", { 4 * C, C, 3, 3 });
    m.u_upb = wt("upsampler.0.bias", { 4 * C });
    if (!m.u_upw || !m.u_upb) return RAD_E_INVAL;
    /* Each frame through Conv2d (issued a frame at a time), then the pixel shuffle over the volume
     * (channel c 4 + 2 i_h + i_w to (2 h + i_h, 2 w + i_w)). */
    m.op_uup = rw(b, RAD_OP(b, "conv2d_nhwc",
                            RAD_PARAMS(RAD_RANGE("M", 1, V / 4), RAD_INT("cin", C),
                                       RAD_INT("cout", 4 * C),
                                       RAD_INT("k", 3), RAD_INT("stride", 1), RAD_STR("pad", "same"),
                                       RAD_INT("up", 1), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(m.u_upw, m.u_upb)),
                  { m.b_ua }, { m.b_ub });
    m.op_ud2s = rw(b, RAD_OP(b, "d2s3d",
                             RAD_PARAMS(RAD_RANGE("M", 1, V / 4), RAD_INT("C", C), RAD_INT("st", 1),
                                        RAD_INT("sh", 2), RAD_INT("sw", 2), RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_ub }, { m.b_ua });
    RAD_ARCH_TRY(blocks("post_upsample_res_blocks", m.u_post));
    m.u_coutw = wt("final_conv.weight", { kC, C, 3, 3, 3 });
    m.u_coutb = wt("final_conv.bias", { kC });
    m.op_ucout = conv(m.u_coutw, m.u_coutb, C, kC, V, m.b_ua, m.b_uout);
    m.op_unorm = rw(b, RAD_OP(b, "chan_stats",
                              RAD_PARAMS(RAD_RANGE("M", 1, V), RAD_INT("n", kC), RAD_STR("mode", "norm"),
                                         RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(m.vd_mean, m.vd_std)),
                    { m.b_uout }, { m.b_uout });
    m.op_uf32 = cast_op(b, V, kC, "bf16", "f32", m.b_uout, m.b_vstate);
    /* With a first frame, its upscaled latent frame is the new clean one: kept aside before the
     * re-noise and put back after. */
    m.op_ufirst = cast_op(b, m.maxPpf, kC, "f32", "f32", m.b_vstate, m.b_first);
    m.op_ukeep = cast_op(b, m.maxPpf, kC, "f32", "f32", m.b_first, m.b_vstate);
    auto renoise = [&](int64_t rows, rad_buf x) {
        return rw(b, RAD_OP(b, "flow_renoise",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", kC), RAD_INT("stream", 1),
                                       RAD_STR("dtype", "f32")),
                            RAD_NOWEIGHTS),
                  { x }, { x });
    };
    m.op_urnv = renoise(V, m.b_vstate);
    m.op_urna = renoise(m.maxA, m.b_astate);
    return m.op_uden && m.op_ucast && m.op_ucin && m.op_unin && m.op_uup && m.op_ud2s && m.op_ucout &&
                   m.op_unorm && m.op_uf32 && m.op_ufirst && m.op_ukeep && m.op_urnv && m.op_urna
               ? RAD_OK
               : RAD_E_INVAL;
}

/* ================================================================== the duration head */
/* LTX2DurationHead: each connector's output for the prompt (not the negative) projected to the
 * pooler's width with its modality's embedding added, the two runs concatenated, one learned query
 * attending over them (four heads, no mask), the pooled row through a tanh-GELU MLP to the log of
 * the length in seconds, which the core reads back (RadVideoGenDecl::duration_out). */
static int declare_dur(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    if (!rad_meta_gets(meta, "duration_head._class_name", nullptr)) return RAD_OK;
    Names& nm = m.nm;
    const int acc = RAD_ACCESS_PER_TOKEN, gx = kGrpDur;
    const int64_t P = rad_meta_geti(meta, "duration_head.pooler_hidden_dim", 0);
    const int64_t Hm = rad_meta_geti(meta, "duration_head.mlp_hidden_dim", 0);
    const int64_t nh = rad_meta_geti(meta, "duration_head.num_pooler_heads", 0);
    if (std::strcmp(rad_meta_gets(meta, "duration_head._class_name", ""), "LTX2DurationHead") ||
        rad_meta_geti(meta, "duration_head.video_cross_attention_dim", 0) != m.cv.D ||
        rad_meta_geti(meta, "duration_head.audio_cross_attention_dim", 0) != m.ca.D ||
        rad_meta_geti(meta, "duration_head.num_queries", 0) != 1 || P <= 0 || Hm <= 0 ||
        nh <= 0 || P % nh || P / nh != 64 || P > std::max(m.cv.D, m.ca.D)) {
        fprintf(stderr, "radiance: ltx2: duration_head/ is not the LTX2DurationHead this plugin "
                        "runs (the connectors' widths in, one query, heads of 64)\n");
        return RAD_E_UNSUPPORTED;
    }
    m.dur = true;
    m.dP = P;
    const int64_t R = kTextRows;
    auto wt = [&](const char* name, std::initializer_list<int64_t> shape) {
        return wcopy(b, nm.f("dur.%s", name), nm.ckpt("duration_head.%s", name), RAD_BF16, shape,
                     acc, gx);
    };
    m.dw_vp = wt("video_input_proj.weight", { P, m.cv.D });
    m.dw_vpb = wt("video_input_proj.bias", { P });
    m.dw_vemb = wt("video_modality_emb", { 1, P });
    m.dw_ap = wt("audio_input_proj.weight", { P, m.ca.D });
    m.dw_apb = wt("audio_input_proj.bias", { P });
    m.dw_aemb = wt("audio_modality_emb", { 1, P });
    m.dw_qt = wt("attention_pooler.query_tokens", { 1, P });
    m.dw_q = wt("attention_pooler.to_q.weight", { P, P });
    m.dw_qb = wt("attention_pooler.to_q.bias", { P });
    m.dw_k = wt("attention_pooler.to_k.weight", { P, P });
    m.dw_kb = wt("attention_pooler.to_k.bias", { P });
    m.dw_v = wt("attention_pooler.to_v.weight", { P, P });
    m.dw_vb = wt("attention_pooler.to_v.bias", { P });
    m.dw_o = wt("attention_pooler.to_out.weight", { P, P });
    m.dw_ob = wt("attention_pooler.to_out.bias", { P });
    m.dw_h = wt("mlp_hidden.weight", { Hm, P });
    m.dw_hb = wt("mlp_hidden.bias", { Hm });
    m.dw_out = wt("mlp_out.weight", { 1, Hm });
    m.dw_outb = wt("mlp_out.bias", { 1 });
    m.b_dtok = decl_b(b, nm.f("dur.tok"), RAD_BF16, { 2 * R, P });
    m.b_dk = decl_b(b, nm.f("dur.k"), RAD_BF16, { 2 * R, P });
    m.b_dv = decl_b(b, nm.f("dur.v"), RAD_BF16, { 2 * R, P });
    m.b_dq0 = decl_b(b, nm.f("dur.query"), RAD_BF16, { 1, P });
    m.b_dq = decl_b(b, nm.f("dur.q"), RAD_BF16, { 1, P });
    m.b_dat = decl_b(b, nm.f("dur.attn"), RAD_BF16, { 1, P });
    m.b_do = decl_b(b, nm.f("dur.pooled"), RAD_BF16, { 1, P });
    m.b_dh = decl_b(b, nm.f("dur.hidden"), RAD_BF16, { 1, Hm });
    m.b_dl = decl_b(b, nm.f("dur.log"), RAD_BF16, { 1, 1 });
    m.b_dcu = decl_b(b, nm.f("dur.cu"), RAD_I32, { 2 });
    m.b_dcuk = decl_b(b, nm.f("dur.cu_k"), RAD_I32, { 2 });
    m.b_dur = decl_b(b, nm.f("dur.out"), RAD_F32, { 1 }, RAD_BUF_PERSIST);
    auto gemm = [&](rad_weight w, rad_weight bb, int64_t rows, int64_t N, int64_t K, rad_buf x,
                    rad_buf y, const char* act) -> rad_op {
        if (!w || !bb || !x || !y) return 0;
        return rw(b, RAD_OP(b, "gemm_nt_bias",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                       RAD_STR("dtype", "bf16"), RAD_STR("act", act)),
                            RAD_WEIGHTS(w, bb)),
                  { x }, { y });
    };
    auto emb = [&](rad_weight w, int64_t rows, rad_buf x, rad_buf y) -> rad_op {
        if (!w) return 0;
        return rw(b, RAD_OP(b, "table_add",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", P), RAD_INT("L", 1),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w)),
                  { x }, { y });
    };
    auto iota = [&](int64_t step, rad_buf y) {
        return rw(b, RAD_OP(b, "iota",
                            RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("start", 0),
                                       RAD_INT("step", step)),
                            RAD_NOWEIGHTS),
                  {}, { y });
    };
    m.op_dvp = gemm(m.dw_vp, m.dw_vpb, R, P, m.cv.D, m.cv.b_text, m.b_dtok, "none");
    m.op_dap = gemm(m.dw_ap, m.dw_apb, R, P, m.ca.D, m.ca.b_text, m.b_dtok, "none");
    m.op_dvemb = emb(m.dw_vemb, R, m.b_dtok, m.b_dtok);
    m.op_daemb = emb(m.dw_aemb, R, m.b_dtok, m.b_dtok);
    m.op_dq0 = emb(m.dw_qt, 1, m.b_zero, m.b_dq0);
    m.op_dq = gemm(m.dw_q, m.dw_qb, 1, P, P, m.b_dq0, m.b_dq, "none");
    m.op_dk = gemm(m.dw_k, m.dw_kb, 2 * R, P, P, m.b_dtok, m.b_dk, "none");
    m.op_dv = gemm(m.dw_v, m.dw_vb, 2 * R, P, P, m.b_dtok, m.b_dv, "none");
    m.op_dcu = iota(1, m.b_dcu);
    m.op_dcuk = iota(2 * R, m.b_dcuk);
    m.op_dattn = rw(b, RAD_OP(b, "attn_cross",
                              RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("head_dim", P / nh),
                                         RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_dq, m.b_dk, m.b_dv, m.b_dcu, m.b_dcuk }, { m.b_dat });
    m.op_do = gemm(m.dw_o, m.dw_ob, 1, P, P, m.b_dat, m.b_do, "none");
    m.op_dh = gemm(m.dw_h, m.dw_hb, 1, Hm, P, m.b_do, m.b_dh, "gelu_tanh");
    m.op_dout = gemm(m.dw_out, m.dw_outb, 1, 1, Hm, m.b_dh, m.b_dl, "none");
    m.op_dlog = cast_op(b, 1, 1, "bf16", "f32", m.b_dl, m.b_dur);
    return m.op_dvp && m.op_dap && m.op_dvemb && m.op_daemb && m.op_dq0 && m.op_dq && m.op_dk &&
                   m.op_dv && m.op_dcu && m.op_dcuk && m.op_dattn && m.op_do && m.op_dh &&
                   m.op_dout && m.op_dlog
               ? RAD_OK
               : RAD_E_INVAL;
}

/* ================================================================== the diffusion decoder */
/* LTX2VideoDiffusionDecoder3d (LTX-2.5's diffusion_decoder/; diffusers' ltx2_diffusion_decoder.py),
 * run as the README's diffusion decode runs it, its tiling on (enable_tiling). The latent, its last
 * frame repeated twice past its end (the ghost frames NATTEN's inward-shifted border needs), goes
 * through four deterministic stages of neighbourhood-attention blocks (pre-norm, a 3-D rotary over
 * the voxel coordinates, a SwiGLU feed-forward), each followed by a pixel-shuffle upsample; the
 * first three over the whole video into the feature volume. That volume is cut as _tile_intervals
 * cuts it (768 pixels every 704, 80 frames every 56: on the feature grid 96 cells every 88 and 40
 * every 28, 6 at least), and each tile runs the fourth stage (the tile starting the video drops
 * the upsample's duplicate leading frame; the one ending it its ghost frames) and one denoising
 * step from noise -- x0 predicted at timestep 1 -- over 4 x 4 pixel patches: eight diffusion blocks
 * conditioned on the fourth stage's output, their norms modulated by the timestep's shared table
 * plus each block's own. The tiles' pixels go to the tile store and blend as tile_stitch does. */
enum { kDdTile = 96, kDdStride = 88, kDdTileT = 40, kDdStrideT = 28, kDdLeast = 6, kDdGhost = 2,
       kDdGhostCells = 8, kDdMaxTiles = 16 };
/* A feed-forward launch's hidden rows: gate and up at most this many elements. */
static const int64_t kDdFfElems = 64ll << 20;

/* The decoder's cut of an axis (tile_grid's `intervals` rule) and the tiles of a feature grid. */
struct DdTile { int64_t t, t0, h, w; bool origin, trailing; };
static int64_t dd_count(int64_t L, int64_t tile, int64_t stride) {
    if (L <= tile) return 1;
    int64_t n = (L + stride - 1) / stride;
    while (n > 1 && L - (n - 1) * stride < kDdLeast) --n;
    return n;
}
static int64_t dd_len(int64_t L, int64_t k, int64_t tile, int64_t stride) {
    return k == dd_count(L, tile, stride) - 1 ? L - k * stride : std::min(tile, L - k * stride);
}
static std::vector<DdTile> dd_tiles(int64_t T, int64_t H, int64_t W) {
    std::vector<DdTile> v;
    const int64_t nt = dd_count(T, kDdTileT, kDdStrideT), ni = dd_count(H, kDdTile, kDdStride);
    const int64_t nj = dd_count(W, kDdTile, kDdStride);
    for (int64_t kt = 0; kt < nt; ++kt)
        for (int64_t i = 0; i < ni; ++i)
            for (int64_t j = 0; j < nj; ++j)
                v.push_back({ dd_len(T, kt, kDdTileT, kDdStrideT) + (kt == nt - 1 ? kDdGhostCells : 0),
                              kt * kDdStrideT, dd_len(H, i, kDdTile, kDdStride),
                              dd_len(W, j, kDdTile, kDdStride), kt == 0, kt == nt - 1 });
    return v;
}

static rad_op dd_gemm(RadBuilder* b, rad_weight w, rad_weight bb, int64_t rows, int64_t N, int64_t K,
                      std::initializer_list<rad_buf> rd, rad_buf y) {
    if (!w) return 0;
    rad_op op = bb ? RAD_OP(b, "gemm_nt_bias",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w, bb))
                   : RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w));
    return rw(b, op, rd, { y });
}

/* One block's weights and ops over at most V voxels of residual `x`. */
static int dd_block_declare(RadBuilder* b, Model& m, DdBlock& B, const char* dn, const char* src,
                            int64_t C, int slot, const int64_t* kernel, int64_t V, bool diff, rad_buf x) {
    Names& nm = m.nm;
    const int acc = RAD_ACCESS_PER_REQUEST, gx = kGrpDdec;
    B.C = C;
    B.ff = (C * 4 + 15) / 16 * 16;
    B.kt = kernel[0];
    B.kh = kernel[1];
    B.kw = kernel[2];
    B.slot = slot;
    B.chunk = std::min<int64_t>(V, kDdFfElems / (2 * B.ff));
    auto wt = [&](const char* name, std::initializer_list<int64_t> shape) {
        return wcopy(b, nm.f("%s.%s", dn, name), nm.ckpt("%s.%s", src, name), RAD_BF16, shape, acc, gx);
    };
    B.w_n1 = wt("norm1.weight", { C });
    /* The diffusion stage's q|k|v of every voxel would be three times its residual (3.7 GiB at the
     * largest tile), so it projects and attends a head at a time into b_dao; the others take
     * every head at once. */
    const int64_t gw = diff ? 64 : C;
    for (int64_t g = 0; g < C / gw; ++g) {
        const char* wn = nm.f("%s.attn.qkv%lld.weight", dn, (long long)g);
        const char* bn = nm.f("%s.attn.qkv%lld.bias", dn, (long long)g);
        RadNameMap mw = map_base(wn, RAD_MAP_ROWS), mb = map_base(bn, RAD_MAP_ROWS);
        for (const char* pr : { "to_q", "to_k", "to_v" }) {
            mw.src[mw.n_src] = nm.ckpt("%s.attn.%s.weight", src, pr);
            mw.src_index[mw.n_src] = (int)(g * gw);
            mw.src_rows[mw.n_src++] = gw;
            mb.src[mb.n_src] = nm.ckpt("%s.attn.%s.bias", src, pr);
            mb.src_index[mb.n_src] = (int)(g * gw);
            mb.src_rows[mb.n_src++] = gw;
        }
        RAD_ARCH_TRY(rad_decl_name_map(b, &mw));
        RAD_ARCH_TRY(rad_decl_name_map(b, &mb));
        B.w_qkvh.push_back(decl_w(b, wn, RAD_BF16, { 3 * gw, C }, acc, RAD_SHARD_NONE, grp_layer(gx)));
        B.b_qkvh.push_back(decl_w(b, bn, RAD_BF16, { 3 * gw }, acc, RAD_SHARD_NONE, grp_layer(gx)));
        if (!B.w_qkvh.back() || !B.b_qkvh.back()) return RAD_E_INVAL;
    }
    B.w_nq = wt("attn.norm_q.weight", { 64 });
    B.w_nk = wt("attn.norm_k.weight", { 64 });
    B.w_o = wt("attn.to_out.0.weight", { C, C });
    B.b_o = wt("attn.to_out.0.bias", { C });
    B.w_n2 = wt("norm2.weight", { C });
    /* gate and up as one weight: one gemm writes whole rows of the gate|up buffer */
    const char* gun = nm.f("%s.mlp.w_gate_up.weight", dn);
    RAD_ARCH_TRY(map_concat(b, gun, { nm.ckpt("%s.mlp.w_gate.weight", src),
                                      nm.ckpt("%s.mlp.w_up.weight", src) }));
    B.w_gu = decl_w(b, gun, RAD_BF16, { 2 * B.ff, C }, acc, RAD_SHARD_NONE, grp_layer(gx));
    B.w_down = wt("mlp.w_down.weight", { C, B.ff });
    if (!B.w_n1 || !B.w_nq || !B.w_nk || !B.w_o || !B.b_o || !B.w_n2 || !B.w_gu || !B.w_down)
        return RAD_E_INVAL;
    const rad_buf qkv = m.b_dqkv[slot], gu = m.b_dg[slot];
    auto modded = [&](rad_weight w) {
        const rad_op op = RAD_OP(b, "rms_mod",
                                 RAD_PARAMS(RAD_RANGE("M", 1, V), RAD_INT("n", C), RAD_F64("eps", 1e-6),
                                            RAD_INT("gated", 0), RAD_STR("dtype", "bf16"),
                                            RAD_INT("norm_once", 1)),
                                 RAD_WEIGHTS(w));
        if (!op) return (rad_op)0;
        const rad_buf rd[3] = { m.b_dbmod, x, m.b_ddcu };
        const rad_buf wr[2] = { x, m.b_dn };
        rad_op_reads(b, op, rd, 3);
        rad_op_writes(b, op, wr, 2);
        return op;
    };
    if (diff) {
        B.w_ctx = wt("context_proj.weight", { C, C });
        B.b_ctx = wt("context_proj.bias", { C });
        /* The block's table rows (shift, scale) of the attention's norm, then of the MLP's: the
         * reference's chunks 1, 0, 4 and 3, the order rms_mod reads them in. */
        const char* tn = nm.f("%s.mod_table", dn);
        RadNameMap mp = map_base(tn, RAD_MAP_ROWS);
        for (int r : { 1, 0, 4, 3 }) {
            mp.src[mp.n_src] = nm.ckpt("%s.scale_shift_table", src);
            mp.src_index[mp.n_src] = r;
            mp.src_rows[mp.n_src++] = 1;
        }
        RAD_ARCH_TRY(rad_decl_name_map(b, &mp));
        B.w_sst = decl_w(b, tn, RAD_BF16, { 1, 4 * C }, acc, RAD_SHARD_NONE, grp_layer(gx));
        if (!B.w_ctx || !B.b_ctx || !B.w_sst) return RAD_E_INVAL;
        B.op_ctx = dd_gemm(b, B.w_ctx, B.b_ctx, V, C, C, { m.b_dctx, x }, x);
        B.op_mod = rw(b, RAD_OP(b, "table_add",
                                RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 4 * C), RAD_INT("L", 1),
                                           RAD_STR("dtype", "bf16")),
                                RAD_WEIGHTS(B.w_sst)),
                      { m.b_dmod }, { m.b_dbmod });
        B.op_n1 = modded(B.w_n1);
        B.op_n2 = modded(B.w_n2);
        if (!B.op_ctx || !B.op_mod) return RAD_E_INVAL;
    } else {
        B.op_n1 = rms_op(b, V, C, 1e-6, B.w_n1, x, m.b_dn);
        B.op_n2 = rms_op(b, V, C, 1e-6, B.w_n2, x, m.b_dn);
    }
    const int64_t nh = gw / 64;
    const rad_buf ao = diff ? m.b_dao : m.b_dn;
    for (size_t g = 0; g < B.w_qkvh.size(); ++g) {
        B.op_qkvh.push_back(dd_gemm(b, B.w_qkvh[g], B.b_qkvh[g], V, 3 * gw, C, { m.b_dn }, qkv));
        if (!B.op_qkvh.back()) return RAD_E_INVAL;
    }
    B.op_rope = RAD_OP(b, "qk_norm_rope",
                       RAD_PARAMS(RAD_RANGE("M", 1, V), RAD_INT("head_dim", 64),
                                  RAD_INT("n_head", nh), RAD_INT("n_head_kv", nh),
                                  RAD_F64("theta", 10000.0), RAD_F64("eps", 1e-6),
                                  RAD_STR("mode", "gptj_blocks"), RAD_INT("rotary_dim", 64),
                                  RAD_STR("sections", "8 12 12")),
                       RAD_WEIGHTS(B.w_nq, B.w_nk));
    if (B.op_rope) {
        const rad_buf rd[2] = { qkv, m.b_dpos };
        rad_op_reads(b, B.op_rope, rd, 2);
        rad_op_writes(b, B.op_rope, &qkv, 1);
    }
    B.op_attn = rw(b, RAD_OP(b, "attn_na3d",
                             RAD_PARAMS(RAD_RANGE("M", 1, V), RAD_INT("head_dim", 64),
                                        RAD_INT("kt", B.kt), RAD_INT("kh", B.kh), RAD_INT("kw", B.kw),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { qkv }, { ao });
    B.op_o = dd_gemm(b, B.w_o, B.b_o, V, C, C, { ao, x }, x);
    B.op_gu = dd_gemm(b, B.w_gu, 0, B.chunk, 2 * B.ff, C, { m.b_dn }, gu);
    B.op_act = rw(b, RAD_OP(b, "silu_mul",
                            RAD_PARAMS(RAD_RANGE("M", 1, B.chunk), RAD_INT("n", B.ff),
                                       RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { gu }, { m.b_da });
    B.op_down = dd_gemm(b, B.w_down, 0, B.chunk, C, B.ff, { m.b_da }, m.b_dmo);
    B.op_add2 = add_op(b, B.chunk, C, "bf16", x, m.b_dmo, x);
    return B.op_n1 && B.op_n2 && B.op_rope && B.op_attn && B.op_o && B.op_gu && B.op_act &&
                   B.op_down && B.op_add2
               ? RAD_OK
               : RAD_E_INVAL;
}

static int declare_ddec(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    if (!rad_meta_gets(meta, "diffusion_decoder._class_name", nullptr)) return RAD_OK;
    Names& nm = m.nm;
    const int acc = RAD_ACCESS_PER_REQUEST, gx = kGrpDdec;
    auto list = [&](const char* k) { return meta_list(meta, nm.f("diffusion_decoder.%s", k)); };
    const std::vector<double> ch = list("decoder_stage_channels"), dep = list("decoder_stage_depths");
    const std::vector<double> ker = list("decoder_stage_kernels"), str = list("decoder_upsample_strides");
    const std::vector<double> red = list("decoder_upsample_channel_reductions");
    const std::vector<double> k5 = list("decoder_stage5_kernel");
    const std::vector<double> want_ch = { 2048, 1024, 512, 512, 256 }, want_dep = { 4, 6, 4, 2, 8 };
    const std::vector<double> want_str = { 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2 };
    if (std::strcmp(rad_meta_gets(meta, "diffusion_decoder._class_name", ""),
                    "LTX2VideoDiffusionDecoderModel") ||
        rad_meta_geti(meta, "diffusion_decoder.decoder_head_dim", 0) != 64 ||
        rad_meta_geti(meta, "diffusion_decoder.latent_channels", 0) != kC ||
        rad_meta_geti(meta, "diffusion_decoder.patch_size", 0) != kPix ||
        rad_meta_geti(meta, "diffusion_decoder.out_channels", 0) != 3 ||
        rad_meta_geti(meta, "diffusion_decoder.decoder_t_emb_dim", 0) != 384 ||
        rad_meta_geti(meta, "diffusion_decoder.decoder_num_inference_steps", 0) != 1 ||
        std::strcmp(rad_meta_gets(meta, "diffusion_decoder.decoder_model_output_type", ""), "x0") ||
        rad_meta_getf(meta, "diffusion_decoder.decoder_timestep_scale_multiplier", 0.0) != 1000.0 ||
        ch != want_ch || dep != want_dep || str != want_str || ker.size() != 12 || k5.size() != 3 ||
        red != std::vector<double>({ 2, 2, 1, 2 })) {
        fprintf(stderr, "radiance: ltx2: diffusion_decoder/ is not the LTX-2.5 diffusion decoder this "
                        "plugin runs (stages 2048..256 of depths 4 6 4 2 8, one x0 step, heads of "
                        "64)\n");
        return RAD_E_UNSUPPORTED;
    }
    m.ddec = true;
    /* The largest volumes: the default canvas's latents a frame (the decoder's own limit), the
     * longest video, the feature grid four times finer, a tile's cells and the diffusion stage's
     * voxels (two a cell a side, ghost frames included in the fourth stage's output). */
    const int64_t F = max_lat_frames(), A = kVaeLat, T0 = F + kDdGhost;
    m.dV0 = T0 * A;
    m.dV1 = 4 * m.dV0;
    m.dV2 = (2 * T0 - 1) * 4 * A;
    m.dV3 = (4 * T0 - 3) * 16 * A;
    const int64_t cells = std::min<int64_t>((int64_t)kDdTile * kDdTile, 16 * A);
    m.dV4 = ((int64_t)kDdTileT + kDdGhostCells) * cells;
    const int64_t ctx_rows = 2 * ((int64_t)kDdTileT + kDdGhostCells) * 4 * cells;
    m.dV5 = 2 * (int64_t)kDdTileT * 4 * cells;
    m.dTiles = kDdMaxTiles;
    const int64_t maxV = std::max({ m.dV0, m.dV1, m.dV2, m.dV4, m.dV5 });
    m.b_dz = decl_b(b, nm.f("ddec.lat"), RAD_BF16, { m.dV0, kC });
    m.b_dx = decl_b(b, nm.f("ddec.x"), RAD_BF16, { std::max({ m.dV0 * 2048, m.dV1 * 1024, m.dV2 * 512 }) });
    m.b_dfeat = decl_b(b, nm.f("ddec.features"), RAD_BF16, { m.dV3 * 512 });
    m.b_d4 = decl_b(b, nm.f("ddec.x4"), RAD_BF16, { m.dV4 * 512 });
    m.b_ddh = decl_b(b, nm.f("ddec.h"), RAD_BF16, { m.dV5 * 256 });
    m.b_dctx = decl_b(b, nm.f("ddec.ctx"), RAD_BF16, { ctx_rows * 256 });
    m.b_dn = decl_b(b, nm.f("ddec.n"), RAD_BF16,
                    { std::max({ m.dV0 * 2048, m.dV1 * 1024, m.dV2 * 512, m.dV4 * 512, m.dV5 * 256 }) });
    const int64_t widths[4] = { 2048, 1024, 512, 256 };
    const int64_t qrows[4] = { m.dV0, m.dV1, std::max(m.dV2, m.dV4), m.dV5 };
    for (int s = 0; s < 4; ++s) {
        /* A width's q|k|v rows, which its stage's upsample also takes as scratch; the diffusion
         * stage's holds one head's. */
        int64_t rows = qrows[s];
        if (s == 2) rows = std::max({ rows, (m.dV2 * 4096 + 1535) / 1536, (m.dV4 * 2048 + 1535) / 1536 });
        m.b_dqkv[s] = decl_b(b, nm.f("ddec.qkv%lld", (long long)widths[s]), RAD_BF16,
                             { rows, s == 3 ? 192 : 3 * widths[s] });
        m.b_dg[s] = decl_b(b, nm.f("ddec.gu%lld", (long long)widths[s]), RAD_BF16,
                           { kDdFfElems / (8 * widths[s]), 8 * widths[s] });
    }
    m.b_dao = decl_b(b, nm.f("ddec.attn"), RAD_BF16, { m.dV5, 256 });
    m.b_da = decl_b(b, nm.f("ddec.act"), RAD_BF16, { kDdFfElems / 2 });
    m.b_dmo = decl_b(b, nm.f("ddec.ffo"), RAD_BF16, { std::max<int64_t>(kDdFfElems / 8, 1) });
    m.b_dpos = decl_b(b, nm.f("ddec.pos"), RAD_I32, { 3, maxV });
    m.b_dxt = decl_b(b, nm.f("ddec.noise"), RAD_F32, { m.dV5, (int64_t)kPixC });
    m.b_dxb = decl_b(b, nm.f("ddec.noise16"), RAD_BF16, { m.dV5, (int64_t)kPixC });
    m.b_dtin = decl_b(b, nm.f("ddec.t"), RAD_F32, { 1 });
    m.b_dtsin = decl_b(b, nm.f("ddec.tsin"), RAD_BF16, { 1, 256 });
    m.b_dth = decl_b(b, nm.f("ddec.th"), RAD_BF16, { 1, 384 });
    m.b_dths = decl_b(b, nm.f("ddec.ths"), RAD_BF16, { 1, 384 });
    m.b_dtemb = decl_b(b, nm.f("ddec.temb"), RAD_BF16, { 1, 384 });
    m.b_dtes = decl_b(b, nm.f("ddec.tes"), RAD_BF16, { 1, 384 });
    m.b_dmod = decl_b(b, nm.f("ddec.mod"), RAD_BF16, { 1, 1024 }, RAD_BUF_PERSIST);
    m.b_dbmod = decl_b(b, nm.f("ddec.block_mod"), RAD_BF16, { 1, 1024 });
    m.b_ddcu = decl_b(b, nm.f("ddec.cu"), RAD_I32, { 2 }, RAD_BUF_PERSIST);
    m.b_dsig = decl_b(b, nm.f("ddec.sigma"), RAD_F32, { 1 }, RAD_BUF_PERSIST);
    m.b_dtg = decl_b(b, nm.f("ddec.tile_geom"), RAD_I32, { 3 }, RAD_BUF_PERSIST);
    m.b_dgrid = decl_b(b, nm.f("ddec.grid"), RAD_I32, { (int64_t)kDdMaxTiles, 8 }, RAD_BUF_PERSIST);
    m.b_dg5 = decl_b(b, nm.f("ddec.geom5"), RAD_I32, { 3 }, RAD_BUF_PERSIST);
    auto wt = [&](const char* name, std::initializer_list<int64_t> shape) {
        return wcopy(b, nm.f("ddec.%s", name), nm.ckpt("diffusion_decoder.decoder.%s", name), RAD_BF16,
                     shape, acc, gx);
    };
    m.dw_cin = wt("conv_in.weight", { 2048, kC });
    m.db_cin = wt("conv_in.bias", { 2048 });
    m.dw_t1 = wt("t_embedder.timestep_embedder.linear_1.weight", { 384, 256 });
    m.db_t1 = wt("t_embedder.timestep_embedder.linear_1.bias", { 384 });
    m.dw_t2 = wt("t_embedder.timestep_embedder.linear_2.weight", { 384, 384 });
    m.db_t2 = wt("t_embedder.timestep_embedder.linear_2.bias", { 384 });
    {
        /* The shared table's chunks 1, 0, 4 and 3 (rms_mod's shift then scale, the attention's then
         * the MLP's); the context gate and the residual gates are not read. */
        const char* wn = nm.f("ddec.adaln.weight");
        const char* bn = nm.f("ddec.adaln.bias");
        RadNameMap mw = map_base(wn, RAD_MAP_ROWS), mb = map_base(bn, RAD_MAP_ROWS);
        for (int p : { 1, 0, 4, 3 }) {
            mw.src[mw.n_src] = nm.ckpt("diffusion_decoder.decoder.shared_adaln.proj.weight");
            mw.src_index[mw.n_src] = p * 256;
            mw.src_rows[mw.n_src++] = 256;
            mb.src[mb.n_src] = nm.ckpt("diffusion_decoder.decoder.shared_adaln.proj.bias");
            mb.src_index[mb.n_src] = p * 256;
            mb.src_rows[mb.n_src++] = 256;
        }
        RAD_ARCH_TRY(rad_decl_name_map(b, &mw));
        RAD_ARCH_TRY(rad_decl_name_map(b, &mb));
        m.dw_ad = decl_w(b, wn, RAD_BF16, { 1024, 384 }, acc, RAD_SHARD_NONE, grp_layer(gx));
        m.db_ad = decl_w(b, bn, RAD_BF16, { 1024 }, acc, RAD_SHARD_NONE, grp_layer(gx));
    }
    m.dw_xin = wt("conv_in_x_t.weight", { 256, (int64_t)kPixC });
    m.db_xin = wt("conv_in_x_t.bias", { 256 });
    m.dw_nout = wt("norm_out.weight", { 256 });
    m.dw_cout = wt("conv_out.weight", { (int64_t)kPixC, 256 });
    m.db_cout = wt("conv_out.bias", { (int64_t)kPixC });
    if (!m.dw_cin || !m.db_cin || !m.dw_t1 || !m.db_t1 || !m.dw_t2 || !m.db_t2 || !m.dw_ad || !m.db_ad ||
        !m.dw_xin || !m.db_xin || !m.dw_nout || !m.dw_cout || !m.db_cout)
        return RAD_E_INVAL;

    /* ---- once a decode: the ghost frames, the shared modulation, the constants, the grids */
    m.op_dpad = cast_op(b, m.dV0, kC, "bf16", "bf16", m.b_vlat, m.b_dz);
    m.op_dtin = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 1), RAD_STR("dtype", "f32"),
                                        RAD_F64("value", 1.0)),
                             RAD_NOWEIGHTS),
                   {}, { m.b_dtin });
    m.op_dtsin = rw(b, RAD_OP(b, "timestep_embed",
                              RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 256),
                                         RAD_F64("scale", 1000.0), RAD_F64("max_period", 10000.0),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_dtin }, { m.b_dtsin });
    m.op_dt1 = dd_gemm(b, m.dw_t1, m.db_t1, 1, 384, 256, { m.b_dtsin }, m.b_dth);
    m.op_dts = unary(b, "silu", 1, 384, "bf16", m.b_dth, m.b_dths);
    m.op_dt2 = dd_gemm(b, m.dw_t2, m.db_t2, 1, 384, 384, { m.b_dths }, m.b_dtemb);
    m.op_das = unary(b, "silu", 1, 384, "bf16", m.b_dtemb, m.b_dtes);
    m.op_dad = dd_gemm(b, m.dw_ad, m.db_ad, 1, 1024, 384, { m.b_dtes }, m.b_dmod);
    m.op_ddcu = rw(b, RAD_OP(b, "iota",
                            RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("start", 0),
                                       RAD_INT("step", 1 << 30)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_ddcu });
    m.op_dsig = rw(b, RAD_OP(b, "fill",
                             RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 1), RAD_STR("dtype", "f32"),
                                        RAD_F64("value", 1.0)),
                             RAD_NOWEIGHTS),
                   {}, { m.b_dsig });
    auto affine = [&](const char* src, std::initializer_list<long long> mul,
                      std::initializer_list<long long> add, rad_buf y) {
        return rw(b, RAD_OP(b, "int_affine",
                            RAD_PARAMS(RAD_RANGE("M", 1, 3), RAD_STR("src", src),
                                       RAD_STR("mul", ints(mul).c_str()),
                                       RAD_STR("add", ints(add).c_str())),
                            RAD_NOWEIGHTS),
                  {}, { y });
    };
    /* The tiles cut the feature grid without its ghost cells: 4 F - 3 frames. */
    m.op_dtgeom = affine("0 1 2", { 4, 4, 4 }, { -3, 0, 0 }, m.b_dtg);
    m.op_dgrid = rw(b, RAD_OP(b, "tile_grid",
                              RAD_PARAMS(RAD_RANGE("M", 1, kDdMaxTiles), RAD_INT("tile", kDdTile),
                                         RAD_INT("stride", kDdStride), RAD_STR("rule", "intervals"),
                                         RAD_INT("min", kDdLeast), RAD_INT("tile_t", kDdTileT),
                                         RAD_INT("stride_t", kDdStrideT), RAD_INT("min_t", kDdLeast),
                                         RAD_INT("tail", kDdGhostCells)),
                              RAD_NOWEIGHTS),
                    { m.b_dtg }, { m.b_dgrid });
    m.op_dcin = dd_gemm(b, m.dw_cin, m.db_cin, m.dV0, 2048, kC, { m.b_dz }, m.b_dx);

    /* ---- the deterministic stages: the first three over the whole latent, the fourth per tile */
    m.dst.resize(4);
    const int64_t svol[4] = { m.dV0, m.dV1, m.dV2, m.dV4 };
    /* each stage's volume off the latent's (F, h, w): (F + 2, h, w), (F + 2, 2h, 2w),
     * (2 F + 3, 2h, 2w), and the feature grid (4 F + 5, 4h, 4w) the tiles are cut from */
    const long long gmul[4][3] = { { 1, 1, 1 }, { 1, 2, 2 }, { 2, 2, 2 }, { 4, 4, 4 } };
    const long long gadd[4] = { 2, 2, 3, 5 };
    for (int s = 0; s < 4; ++s) {
        DdStage& S = m.dst[(size_t)s];
        S.C = (int64_t)ch[(size_t)s];
        S.cout = (int64_t)ch[(size_t)s + 1];
        S.st = (int64_t)str[(size_t)(3 * s)];
        S.sh = (int64_t)str[(size_t)(3 * s + 1)];
        S.sw = (int64_t)str[(size_t)(3 * s + 2)];
        S.geom = decl_b(b, nm.f("ddec.s%d.geom", s), RAD_I32, { 3 }, RAD_BUF_PERSIST);
        S.op_geom = affine("0 1 2", { gmul[s][0], gmul[s][1], gmul[s][2] }, { gadd[s], 0, 0 }, S.geom);
        const rad_buf x = s < 3 ? m.b_dx : m.b_d4;
        if (s == 3)
            m.op_dcrop = rw(b, RAD_OP(b, "tile_crop",
                                      RAD_PARAMS(RAD_RANGE("M", 1, m.dV4), RAD_INT("C", 512),
                                                 RAD_STR("dtype", "bf16")),
                                      RAD_NOWEIGHTS),
                            { m.b_dfeat }, { m.b_d4 });
        S.op_pos = rw(b, RAD_OP(b, "vol_index", RAD_PARAMS(RAD_RANGE("M", 1, maxV)), RAD_NOWEIGHTS), {},
                      { m.b_dpos });
        const int slot = s == 0 ? 0 : s == 1 ? 1 : 2;
        int64_t kk[3] = { (int64_t)ker[(size_t)(3 * s)], (int64_t)ker[(size_t)(3 * s + 1)],
                          (int64_t)ker[(size_t)(3 * s + 2)] };
        S.blk.resize((size_t)dep[(size_t)s]);
        for (size_t l = 0; l < S.blk.size(); ++l)
            RAD_ARCH_TRY(dd_block_declare(b, m, S.blk[l], nm.f("ddec.s%d.%zu", s, l),
                                          nm.ckpt("diffusion_decoder.decoder.det_stages.%d.%zu", s, l),
                                          S.C, slot, kk, svol[s], false, x));
        const int64_t P = S.st * S.sh * S.sw * S.C / (int64_t)red[(size_t)s];
        S.w_up = wcopy(b, nm.f("ddec.up%d.weight", s),
                       nm.ckpt("diffusion_decoder.decoder.upsamples.%d.proj.weight", s), RAD_BF16,
                       { P, S.C }, acc, gx);
        S.b_up = wcopy(b, nm.f("ddec.up%d.bias", s),
                       nm.ckpt("diffusion_decoder.decoder.upsamples.%d.proj.bias", s), RAD_BF16, { P },
                       acc, gx);
        /* The upsample's linear into its width's q|k|v buffer; the fourth stage's into the 512 one too,
         * so that the diffusion stage's is first touched by its blocks and its noise can share it. */
        const rad_buf scratch = m.b_dqkv[slot];
        const rad_buf dest = s < 2 ? m.b_dx : s == 2 ? m.b_dfeat : m.b_dctx;
        S.op_proj = dd_gemm(b, S.w_up, S.b_up, svol[s], P, S.C, { x }, scratch);
        auto d2s = [&](int64_t drop) {
            return rw(b, RAD_OP(b, "d2s3d",
                                RAD_PARAMS(RAD_RANGE("M", 1, svol[s]), RAD_INT("C", S.cout),
                                           RAD_INT("st", S.st), RAD_INT("sh", S.sh), RAD_INT("sw", S.sw),
                                           RAD_INT("drop", drop), RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { scratch }, { dest });
        };
        S.op_d2s = d2s(S.st - 1);
        S.op_d2s_keep = s == 3 ? d2s(0) : 0;
        if (!S.op_geom || !S.op_pos || !S.w_up || !S.b_up || !S.op_proj || !S.op_d2s ||
            (s == 3 && (!S.op_d2s_keep || !m.op_dcrop)))
            return RAD_E_INVAL;
    }
    /* ---- the diffusion stage, per tile: its volume, positions, noise and blocks */
    {
        /* (2 t - drop - ghost, 2 h, 2 w) off the tile's row: the tile starting the video drops a
         * frame, the one ending it 16 ghost frames. */
        int v = 0;
        for (int origin = 0; origin < 2; ++origin)
            for (int trailing = 0; trailing < 2; ++trailing)
                m.op_dg5[v++] = affine("0 1 2", { 2, 2, 2 },
                                       { -(origin ? 1 : 0) - (trailing ? 2 * kDdGhostCells : 0), 0, 0 },
                                       m.b_dg5);
    }
    m.op_dpos5 = rw(b, RAD_OP(b, "vol_index", RAD_PARAMS(RAD_RANGE("M", 1, maxV)), RAD_NOWEIGHTS),
                    { m.b_dg5 }, { m.b_dpos });
    m.op_dz = rw(b, RAD_OP(b, "fill",
                           RAD_PARAMS(RAD_RANGE("M", 1, m.dV5), RAD_INT("n", kPixC),
                                      RAD_STR("dtype", "f32")),
                           RAD_NOWEIGHTS),
                 {}, { m.b_dxt });
    m.op_dnoise.resize(kDdMaxTiles);
    for (int k = 0; k < kDdMaxTiles; ++k)
        m.op_dnoise[(size_t)k] = rw(b, RAD_OP(b, "flow_renoise",
                                              RAD_PARAMS(RAD_RANGE("M", 1, m.dV5), RAD_INT("n", kPixC),
                                                         RAD_INT("stream", 2 + k), RAD_STR("dtype", "f32")),
                                              RAD_NOWEIGHTS),
                                    { m.b_dsig, m.b_dxt }, { m.b_dxt });
    m.op_dxcast = cast_op(b, m.dV5, kPixC, "f32", "bf16", m.b_dxt, m.b_dxb);
    m.op_dxin = dd_gemm(b, m.dw_xin, m.db_xin, m.dV5, 256, kPixC, { m.b_dxb }, m.b_ddh);
    m.dblk.resize((size_t)dep[4]);
    const int64_t kk5[3] = { (int64_t)k5[0], (int64_t)k5[1], (int64_t)k5[2] };
    for (size_t l = 0; l < m.dblk.size(); ++l)
        RAD_ARCH_TRY(dd_block_declare(b, m, m.dblk[l], nm.f("ddec.d.%zu", l),
                                      nm.ckpt("diffusion_decoder.decoder.diff_blocks.%zu", l), 256, 3,
                                      kk5, m.dV5, true, m.b_ddh));
    m.op_dnout = rms_op(b, m.dV5, 256, 1e-6, m.dw_nout, m.b_ddh, m.b_dn);
    /* The tile's pixels to the store; it is the tile loop's last op, so the feature volume, which
     * every tile's crop reads, stays live through the loop. */
    m.op_ddout = dd_gemm(b, m.dw_cout, m.db_cout, m.dV5, kPixC, 256, { m.b_dn, m.b_dfeat }, m.b_vtiles);
    m.op_dst = rw(b, RAD_OP(b, "tile_stitch",
                            RAD_PARAMS(RAD_RANGE("M", 1, (int64_t)(kFpc + kFtail) * m.maxSide),
                                       RAD_INT("tile", kDdTile), RAD_INT("stride", kDdStride),
                                       RAD_STR("rule", "intervals"), RAD_INT("min", kDdLeast),
                                       RAD_INT("tile_t", kDdTileT), RAD_INT("stride_t", kDdStrideT),
                                       RAD_INT("min_t", kDdLeast), RAD_INT("scale", 2),
                                       RAD_INT("scale_t", 2), RAD_INT("p", kPix), RAD_INT("C", 3),
                                       RAD_STR("order", "wh"), RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                  { m.b_vtiles, m.b_dtg, m.b_vf0 }, { m.b_frames });
    /* The store holds the largest video's tiles at the decoder's own largest canvas. */
    for (int64_t Wd = kMinSide; Wd * kMinSide <= kDefW * kDefH; Wd += kMult)
        for (int64_t Hd = kMinSide; Wd * Hd <= kDefW * kDefH; Hd += kMult) {
            if (std::max(Wd, Hd) > kMaxAspect * std::min(Wd, Hd)) continue;
            int64_t rows = 0;
            for (const DdTile& t : dd_tiles(4 * F - 3, 4 * (Hd / kScale), 4 * (Wd / kScale)))
                rows += (2 * (t.t - (t.trailing ? kDdGhostCells : 0)) - (t.origin ? 1 : 0)) * 4 * t.h * t.w;
            if (rows > m.maxStore || (int64_t)dd_tiles(4 * F - 3, 4 * (Hd / kScale), 4 * (Wd / kScale)).size() > kDdMaxTiles) {
                fprintf(stderr, "radiance: ltx2: the diffusion decoder's tiles of %lldx%lld (%lld rows) "
                                "outgrow the tile store (%lld rows) or its %d tiles\n",
                        (long long)Wd, (long long)Hd, (long long)rows, (long long)m.maxStore,
                        (int)kDdMaxTiles);
                return RAD_E_UNSUPPORTED;
            }
        }
    return m.op_dpad && m.op_dtin && m.op_dtsin && m.op_dt1 && m.op_dts && m.op_dt2 && m.op_das &&
                   m.op_dad && m.op_ddcu && m.op_dsig && m.op_dtgeom && m.op_dgrid && m.op_dcin &&
                   m.op_dpos5 && m.op_dz && m.op_dxcast && m.op_dxin && m.op_dnout && m.op_ddout &&
                   m.op_dst
               ? RAD_OK
               : RAD_E_INVAL;
}

/* ================================================================== declare */
/* The distilled release's trained sigmas, and its second stage's after the latent upsampler
 * (diffusers' pipelines/ltx2/utils.py, DISTILLED_SIGMA_VALUES and
 * STAGE_2_DISTILLED_SIGMA_VALUES). */
static const double kDistilled[] = { 1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875 };
static const double kStage2[] = { 0.909375, 0.725, 0.421875 };

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    if (ctx->is_draft) {
        fprintf(stderr, "radiance: ltx2: this plugin has no draft scope\n");
        return RAD_E_UNSUPPORTED;
    }
    if (ctx->world_size > 1) {
        fprintf(stderr, "radiance: ltx2: runs on one rank (--tp 1): the DiT's q and k norms span "
                        "every head\n");
        return RAD_E_UNSUPPORTED;
    }
    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    Names& nm = m.nm;
    RAD_ARCH_TRY(check_format(meta));

    /* ---- the text encoder's geometry, off its config */
    m.tC = tgeti(meta, "hidden_size", 0);
    m.tL = tgeti(meta, "num_hidden_layers", 0);
    m.tFF = tgeti(meta, "intermediate_size", 0);
    m.teps = tgetf(meta, "rms_norm_eps", 1e-6);
    const int64_t vocab = tgeti(meta, "vocab_size", 0);
    if (m.tC <= 0 || m.tL <= 0 || m.tFF <= 0 || vocab <= 0) {
        fprintf(stderr, "radiance: ltx2: the text encoder's config names no width, depth or "
                        "vocabulary\n");
        return RAD_E_INVAL;
    }
    m.tRows = 2 * (int64_t)kTextRows;
    Geom base{};
    base.rank = ctx->rank;
    base.world = 1;
    base.max_tok = ctx->max_tok;
    base.max_seqs = ctx->max_seqs;
    base.max_ctx = ctx->max_ctx;
    base.n_vocab = base.n_vocab_all = vocab;
    base.eps = (float)m.teps;
    base.dtype = "bf16";
    base.act_dtype = RAD_BF16;
    m.gt = base;
    m.gt.max_tok = m.tRows;
    m.gt.n_embd = m.tC;
    m.gt.n_layer = m.tL;
    m.gd = base;

    /* ---- the DiT's */
    m.H = rad_meta_geti(meta, "transformer.num_attention_heads", 0);
    m.hd = rad_meta_geti(meta, "transformer.attention_head_dim", 0);
    m.Ha = rad_meta_geti(meta, "transformer.audio_num_attention_heads", 0);
    m.hda = rad_meta_geti(meta, "transformer.audio_attention_head_dim", 0);
    m.nb = rad_meta_geti(meta, "transformer.num_layers", 0);
    m.D = m.H * m.hd;
    m.Da = m.Ha * m.hda;
    m.Dc = m.Ha * m.hda;
    m.ffv = 4 * m.D;
    m.ffa = 4 * m.Da;
    m.eps = rad_meta_getf(meta, "transformer.norm_eps", 1e-6);
    if (m.D <= 0 || m.Da <= 0 || m.nb <= kStgBlock ||
        rad_meta_geti(meta, "transformer.cross_attention_dim", 0) != m.D ||
        rad_meta_geti(meta, "transformer.audio_cross_attention_dim", 0) != m.Da ||
        std::strcmp(rad_meta_gets(meta, "transformer.activation_fn", ""), "gelu-approximate") ||
        rad_meta_geti(meta, "transformer.ff_bias", 1) != 0 ||
        rad_meta_geti(meta, "transformer.audio_ff_bias", 0) != 1) {
        fprintf(stderr, "radiance: ltx2: the DiT is not LTX-2.5's (%lld blocks of %lld + %lld wide, "
                        "block %d for spatio-temporal guidance, text at its width, a tanh GELU "
                        "feed-forward without biases on the video)\n",
                (long long)m.nb, (long long)m.D, (long long)m.Da, (int)kStgBlock);
        return RAD_E_UNSUPPORTED;
    }
    m.maxS = max_video_rows();
    m.maxA = max_audio();
    m.maxPpf = max_patches();
    if (ctx->max_tok < max_step_rows() || ctx->max_tok < m.tRows) {
        fprintf(stderr, "radiance: ltx2: a pass carries %lld rows; the longest video's step needs "
                        "%lld\n", (long long)ctx->max_tok, (long long)max_step_rows());
        return RAD_E_INVAL;
    }

    /* ---- bf16, block fp8 or rotated int8: what the container holds for the first DiT projection
     * decides every large linear (rad_arch.h, Geom::w_bf16 and w_i8h) */
    {
        LinearFP8 probe{};
        RAD_ARCH_TRY(probe.probe(b, nm, m.gd, "dit.blk.0.attn1.qkv",
                                 { "transformer.transformer_blocks.0.attn1.to_q.weight",
                                   "transformer.transformer_blocks.0.attn1.to_k.weight",
                                   "transformer.transformer_blocks.0.attn1.to_v.weight" }));
        RadEncoding e{};
        const bool known = weight_enc(b, nm.f("dit.blk.0.attn1.qkv.weight"), &e);
        m.gd.w_bf16 = m.gt.w_bf16 = known && rad_enc_is(&e, "plain");
        m.gd.w_i8h = m.gt.w_i8h = known && linear_is_w8h(e);
        if (m.gd.w_i8h) m.gd.w_i8_group = m.gt.w_i8_group = linear_w8h_group(e);
    }

    /* ---- the scheduler: the base model's shifted linspace, or the distilled release's sigmas */
    const bool distilled = rad_meta_geti(meta, "scheduler.use_dynamic_shifting", 1) == 0;
    if (distilled) {
        m.rungs.assign(std::begin(kDistilled), std::end(kDistilled));
        m.refine.assign(std::begin(kStage2), std::end(kStage2));
    }

    /* ---- in issue order: the conditioning pass, the step, the frames, the soundtrack, the
     * upscale */
    /* LoRA files for LTX-2.5 name the DiT's modules as the checkpoint does (`transformer_blocks.N.
     * attn1.to_q`, `ff.net.0.proj`), behind ComfyUI's or kohya's prefix. */
    RAD_ARCH_TRY(lora_names(b, "transformer.", kLoraDiffusionSpellings, {}));
    RAD_ARCH_TRY(declare_te(b, meta, m));
    {
        const int64_t wz = std::max<int64_t>(rad_meta_geti(meta, "connectors.video_hidden_dim", 0),
                                             rad_meta_geti(meta, "connectors.audio_hidden_dim", 0));
        m.b_zero = decl_b(b, nm.f("conn.zero"), RAD_BF16, { 1, wz }, RAD_BUF_PERSIST);
        m.b_rows = decl_b(b, nm.f("conn.rows"), RAD_I32, { 3 }, RAD_BUF_PERSIST);
        m.b_rpos = decl_b(b, nm.f("conn.pos"), RAD_I32, { 2 * (int64_t)kTextRows }, RAD_BUF_PERSIST);
        m.op_zero = rw(b, RAD_OP(b, "fill",
                                 RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", wz),
                                            RAD_STR("dtype", "bf16"), RAD_F64("value", 0.0)),
                                 RAD_NOWEIGHTS),
                       {}, { m.b_zero });
        m.op_rows = rw(b, RAD_OP(b, "iota",
                                 RAD_PARAMS(RAD_RANGE("M", 1, 3), RAD_INT("start", 0),
                                            RAD_INT("step", (int64_t)kTextRows)),
                                 RAD_NOWEIGHTS),
                       {}, { m.b_rows });
        m.op_rpos = rw(b, RAD_OP(b, "iota",
                                 RAD_PARAMS(RAD_RANGE("M", 1, 2 * (int64_t)kTextRows),
                                            RAD_INT("start", 0),
                                            RAD_INT("step", 1), RAD_INT("period", (int64_t)kTextRows)),
                                 RAD_NOWEIGHTS),
                       {}, { m.b_rpos });
        if (!m.b_zero || !m.op_zero || !m.op_rows || !m.op_rpos) return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(declare_conn(b, meta, m, m.cv, "video"));
    RAD_ARCH_TRY(declare_conn(b, meta, m, m.ca, "audio"));
    if (m.cv.D != m.D || m.ca.D != m.Da) {
        fprintf(stderr, "radiance: ltx2: the connectors' widths (%lld, %lld) are not the DiT's\n",
                (long long)m.cv.D, (long long)m.ca.D);
        return RAD_E_UNSUPPORTED;
    }
    RAD_ARCH_TRY(declare_dur(b, meta, m));
    RAD_ARCH_TRY(declare_dit(b, meta, m));
    RAD_ARCH_TRY(declare_vae(b, meta, m));
    RAD_ARCH_TRY(declare_venc(b, meta, m));
    RAD_ARCH_TRY(declare_audio(b, meta, m));
    RAD_ARCH_TRY(declare_ups(b, meta, m));
    RAD_ARCH_TRY(declare_ddec(b, meta, m));

    if (!ctx->shape_probe) {
        RadVideoGenDecl d{};
        d.name = distilled ? "ltx-2.5-distilled" : "ltx-2.5";
        /* Gemma's tokenizer wraps the prompt in <bos> and <eos> (its TemplateProcessing). */
        d.tmpl = "<bos>{prompt}<eos>";
        d.max_text = kTextRows;
        d.fps_num = kFps;
        d.fps_den = 1;
        d.frames_per_chunk = kFpc;
        d.frames_tail = kFtail;
        d.latents_per_chunk = kLpc;
        d.latents_tail = kLtail;
        d.dec_latents = kDecLat;
        d.min_chunks = kMinChunks;
        d.max_chunks = kMaxChunks;
        d.latent_scale = kScale;
        d.latent_channels = kC;
        d.patch = 1;
        d.size_multiple = kMult;
        d.min_side = kMinSide;
        d.max_area = kMaxArea;
        d.max_aspect = kMaxAspect;
        d.default_width = kDefW;
        d.default_height = kDefH;
        d.default_chunks = kDefChunks;
        d.sample_rate = kRate;
        d.audio_rate = kAudioRate;
        d.audio_channels = kAudioCh;
        d.audio_latent_channels = kAudioC;
        d.audio_streams = 1;
        d.max_lanes = kMaxLanes;
        d.frames_out = m.b_frames;
        d.audio_out = m.b_audio;
        d.sched_kind = 1;
        d.base_shift = rad_meta_getf(meta, "scheduler.base_shift", 0.95);
        d.max_shift = rad_meta_getf(meta, "scheduler.max_shift", 2.05);
        d.base_seq = rad_meta_geti(meta, "scheduler.base_image_seq_len", 1024);
        d.max_seq = rad_meta_geti(meta, "scheduler.max_image_seq_len", 4096);
        if (distilled) {
            d.n_rungs = (int32_t)m.rungs.size();
            d.rungs = m.rungs.data();
            d.default_steps = d.max_steps = d.n_rungs;
        } else {
            d.dyn_shift = 1;
            d.shift_terminal = rad_meta_getf(meta, "scheduler.shift_terminal", 0.1);
            d.default_steps = kDefSteps;
            d.max_steps = kMaxSteps;
        }
        d.step_text_rows = kTextRows;
        /* The base model is guided four ways a step by default; the distilled release unguided
         * (one evaluation a step). */
        d.guidance = 1;
        static const double kGuided[8] = { 3.0, 1.0, 3.0, 0.7, 7.0, 1.0, 3.0, 0.7 };
        static const double kPlain[8] = { 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 };
        std::memcpy(d.guidance_default, distilled ? kPlain : kGuided, sizeof d.guidance_default);
        d.image_cond = m.venc ? 1 : 0;
        d.image_crf = 18;
        d.text_left_pad = kTextRows;
        /* The two-stage recipe: the distilled release at half size, the upsampler, its second
         * stage's sigmas at the full size. */
        if (m.ups && distilled) {
            d.upscale_max = 2.0;
            d.upscale_exact = 1;
            d.n_refine_rungs = (int32_t)m.refine.size();
            d.refine_rungs = m.refine.data();
            d.upscale_refine = d.n_refine_rungs;
        }
        /* The VAE decodes by default; the diffusion decoder, when the container has it, on request,
         * up to the default canvas. */
        static const int64_t kDecArea[2] = { 0, (int64_t)kDefW * kDefH };
        if (m.ddec) {
            d.decoders = "vae diffusion";
            d.decoder_max_area = kDecArea;
        }
        /* The duration head's bounds are the pipeline's defaults (LTX2AutoDurationStep). */
        if (m.dur) {
            d.duration_out = m.b_dur;
            d.duration_min = 1.0;
            d.duration_max = 20.0;
            d.duration_dtype = RAD_BF16;
        }
        RAD_ARCH_TRY(rad_declare_video_generator(b, &d));
    }
    rad_note(b, "LTX-2.5%s: Gemma 4 text encoder %lld layers of %lld, connectors %zu + %zu blocks, "
                "DiT %lld blocks (video %lld, audio %lld), %s linears; VAE %s, audio VAE + vocoder + "
                "bandwidth extender%s%s",
             distilled ? " distilled" : "", (long long)m.tL, (long long)m.tC, m.cv.blk.size(),
             m.ca.blk.size(), (long long)m.nb, (long long)m.D, (long long)m.Da,
             m.gd.w_bf16 ? "bf16" : m.gd.w_i8h ? "rotated int8" : "block-fp8",
             m.venc ? "decoder + encoder" : "decoder",
             m.ups ? ", latent upsampler x2" : "", m.dur ? ", duration head" : "");
    return RAD_OK;
}

/* ================================================================== step */
/* A column slice of a table's rows [r0, r0 + n): `first` columns into a row of `pitch`. */
static RadOperand tab(rad_buf t, int64_t r0, int64_t n, int64_t pitch, int64_t first, int64_t width) {
    return bcol_at(t, r0, pitch, first, width, n);
}

/* The conditioning pass: the text encoder over both prompts into the stack, each connector into
 * the lane's text rows, and with a first frame the VAE encoder into the lane's clean frame. */
static void step_cond(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    const int64_t T = batch->n_tok, Tn = V->n_neg, Tp = T - Tn, lane = V->lane;
    const int64_t C = m.tC, L = m.tL, Lp = L + 1, np = Tn > 0 ? 2 : 1;
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, np + 1);

    /* ---- the text encoder */
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_tx, T));
    RAD_ISSUE_N(c, m.op_keep0, T, brows(m.b_tx, T), bcol(m.b_stack, 0, C, T));
    for (int64_t l = 0; l < L; ++l) {
        const TeLayer& Y = m.te[(size_t)l];
        const TeClass& K = m.cls[Y.full ? 1 : 0];
        const int64_t qd = Y.nh * Y.hd, kvd = Y.nkv * Y.hd;
        RAD_ISSUE_N(c, Y.op_in, T, brows(m.b_tx, T), RAD_W(Y.w_in), brows(m.a_th.a.x, T));
        m.a_th.step(c, T);
        Y.q.step(c, m.a_th, K.q, T);
        Y.k.step(c, m.a_th, K.k, T);
        if (!Y.v_from_k) Y.v.step(c, m.a_th, K.v, T);
        RAD_ISSUE_N(c, Y.op_vn, T * Y.nkv, brows(Y.v_from_k ? K.k : K.v, T), brows(K.v, T));
        RAD_ISSUE_N(c, Y.op_qn, T * Y.nh, brows(K.q, T), RAD_W(Y.w_qn), brows(K.q, T));
        RAD_ISSUE_N(c, Y.op_kn, T * Y.nkv, brows(K.k, T), RAD_W(Y.w_kn), brows(K.k, T));
        RAD_ISSUE_N(c, Y.op_rq, T, bcol(K.q, 0, qd, T), pos);
        RAD_ISSUE_N(c, Y.op_rk, T, bcol(K.k, 0, kvd, T), pos);
        RAD_ISSUE_N(c, Y.op_attn, T, bcol(K.q, 0, qd, T), bcol(K.k, 0, kvd, T), bcol(K.v, 0, kvd, T), cu,
                    bcol(K.o.a.x, 0, qd, T));
        K.o.step(c, T);
        Y.o.step(c, K.o, m.b_th2, T);
        RAD_ISSUE_N(c, Y.op_pa, T, brows(m.b_th2, T), RAD_W(Y.w_pa), brows(m.b_th2, T));
        RAD_ISSUE_N(c, Y.op_add1, T, brows(m.b_tx, T), brows(m.b_th2, T), brows(m.b_tx, T));
        RAD_ISSUE_N(c, Y.op_pf, T, brows(m.b_tx, T), RAD_W(Y.w_pf), brows(m.a_th.a.x, T));
        m.a_th.step(c, T);
        Y.gu.step(c, m.a_th, m.b_tgu, T);
        RAD_ISSUE_N(c, Y.op_act, T, brows(m.b_tgu, T), brows(m.a_tffn.a.x, T));
        m.a_tffn.step(c, T);
        Y.down.step(c, m.a_tffn, m.b_th2, T);
        RAD_ISSUE_N(c, Y.op_ff, T, brows(m.b_th2, T), RAD_W(Y.w_ff), brows(m.b_th2, T));
        RAD_ISSUE_N(c, Y.op_add2, T, brows(m.b_tx, T), brows(m.b_th2, T), brows(m.b_tx, T));
        RAD_ISSUE_N(c, Y.op_sc, T, brows(m.b_tx, T), RAD_W(Y.w_sc), brows(m.b_tx, T));
        if (Y.op_keep)
            RAD_ISSUE_N(c, Y.op_keep, T, brows(m.b_tx, T), bcol(m.b_stack, (l + 1) * C, C, T));
    }
    RAD_ISSUE_N(c, m.op_tnorm, T, brows(m.b_tx, T), RAD_W(m.w_tnorm), bcol(m.b_stack, L * C, C, T));

    /* ---- the connectors */
    RAD_ISSUE_N(c, m.op_zero, 1, brows(m.b_zero, 1));
    RAD_ISSUE_N(c, m.op_rows, 3, brows(m.b_rows, 3));
    RAD_ISSUE_N(c, m.op_rpos, 2 * kTextRows, brows(m.b_rpos, 2 * kTextRows));
    const int64_t R = np * kTextRows;
    const RadOperand ccu = brows(m.b_rows, np + 1);
    const RadOperand rpos = brows(m.b_rpos, R);
    for (const Conn* Kp : { &m.cv, &m.ca }) {
        const Conn& K = *Kp;
        const int64_t D = K.D;
        RAD_ISSUE_N(c, K.op_norm, T * Lp, view(c, m.b_stack, RAD_BF16, T * Lp, C),
                    brows(K.a_in.a.x, T));
        K.a_in.step(c, T);
        /* Every row the registers first, then the prompts' tokens over the front of each. */
        for (int64_t p = 0; p < np; ++p)
            RAD_ISSUE_N(c, K.op_reg, 1, bcol(m.b_zero, 0, D, 1), RAD_W(K.w_reg),
                        view(c, K.b_x, RAD_BF16, 1, (int64_t)kTextRows * D, p * kTextRows * D));
        K.proj.step(c, K.a_in, K.b_x, Tp, 0, 0);
        if (Tn > 0) K.proj.step(c, K.a_in, K.b_x, Tn, Tp, kTextRows);
        for (const ConnBlock& B : K.blk) {
            RAD_ISSUE_N(c, B.op_n1, R, brows(K.b_x, R), RAD_NONE, brows(K.a_h.a.x, R));
            K.a_h.step(c, R);
            B.at.qkv.step(c, K.a_h, K.b_qkv, R);
            B.at.gate.step(c, K.a_h, K.b_g, R);
            attn_core(c, B.at, true, K.b_qkv, 0, K.b_g, K.a_attn, R, R, ccu, RAD_NONE, rpos, rpos);
            K.a_attn.step(c, R);
            B.at.out.step(c, K.a_attn, K.b_o, R);
            RAD_ISSUE_N(c, B.op_add1, R, brows(K.b_x, R), brows(K.b_o, R), brows(K.b_x, R));
            RAD_ISSUE_N(c, B.op_n2, R, brows(K.b_x, R), RAD_NONE, brows(K.a_h.a.x, R));
            K.a_h.step(c, R);
            B.up.step(c, K.a_h, K.b_up, R);
            RAD_ISSUE_N(c, B.op_act, R, brows(K.b_up, R), brows(K.a_ffn.a.x, R));
            K.a_ffn.step(c, R);
            B.down.step(c, K.a_ffn, K.b_o, R);
            RAD_ISSUE_N(c, B.op_add2, R, brows(K.b_x, R), brows(K.b_o, R), brows(K.b_x, R));
        }
        RAD_ISSUE_N(c, K.op_nout, R, brows(K.b_x, R), RAD_NONE,
                    brow_slice(K.b_text, lane * 2 * kTextRows, R, D));
    }

    /* ---- the duration head, over the prompt's rows of each connector */
    if (m.dur) {
        const int64_t r0 = lane * 2 * kTextRows, P = m.dP;
        RAD_ISSUE_N(c, m.op_dvp, kTextRows, brow_slice(m.cv.b_text, r0, kTextRows, m.cv.D),
                    RAD_W(m.dw_vp), RAD_W(m.dw_vpb), RAD_NONE, brows(m.b_dtok, kTextRows));
        RAD_ISSUE_N(c, m.op_dap, kTextRows, brow_slice(m.ca.b_text, r0, kTextRows, m.ca.D),
                    RAD_W(m.dw_ap), RAD_W(m.dw_apb), RAD_NONE,
                    brow_slice(m.b_dtok, kTextRows, kTextRows, P));
        RAD_ISSUE_N(c, m.op_dvemb, kTextRows, brows(m.b_dtok, kTextRows), RAD_W(m.dw_vemb),
                    brows(m.b_dtok, kTextRows));
        RAD_ISSUE_N(c, m.op_daemb, kTextRows, brow_slice(m.b_dtok, kTextRows, kTextRows, P),
                    RAD_W(m.dw_aemb), brow_slice(m.b_dtok, kTextRows, kTextRows, P));
        RAD_ISSUE_N(c, m.op_dq0, 1, bcol(m.b_zero, 0, P, 1), RAD_W(m.dw_qt), brows(m.b_dq0, 1));
        RAD_ISSUE_N(c, m.op_dq, 1, brows(m.b_dq0, 1), RAD_W(m.dw_q), RAD_W(m.dw_qb), RAD_NONE,
                    brows(m.b_dq, 1));
        RAD_ISSUE_N(c, m.op_dk, 2 * kTextRows, brows(m.b_dtok, 2 * kTextRows), RAD_W(m.dw_k),
                    RAD_W(m.dw_kb), RAD_NONE, brows(m.b_dk, 2 * kTextRows));
        RAD_ISSUE_N(c, m.op_dv, 2 * kTextRows, brows(m.b_dtok, 2 * kTextRows), RAD_W(m.dw_v),
                    RAD_W(m.dw_vb), RAD_NONE, brows(m.b_dv, 2 * kTextRows));
        RAD_ISSUE_N(c, m.op_dcu, 2, brows(m.b_dcu, 2));
        RAD_ISSUE_N(c, m.op_dcuk, 2, brows(m.b_dcuk, 2));
        RAD_ISSUE_N(c, m.op_dattn, 1, brows(m.b_dq, 1), brows(m.b_dk, 2 * kTextRows),
                    brows(m.b_dv, 2 * kTextRows), brows(m.b_dcu, 2), brows(m.b_dcuk, 2),
                    brows(m.b_dat, 1));
        RAD_ISSUE_N(c, m.op_do, 1, brows(m.b_dat, 1), RAD_W(m.dw_o), RAD_W(m.dw_ob), RAD_NONE,
                    brows(m.b_do, 1));
        RAD_ISSUE_N(c, m.op_dh, 1, brows(m.b_do, 1), RAD_W(m.dw_h), RAD_W(m.dw_hb), RAD_NONE,
                    brows(m.b_dh, 1));
        RAD_ISSUE_N(c, m.op_dout, 1, brows(m.b_dh, 1), RAD_W(m.dw_out), RAD_W(m.dw_outb), RAD_NONE,
                    brows(m.b_dl, 1));
        RAD_ISSUE_N(c, m.op_dlog, 1, brows(m.b_dl, 1), brows(m.b_dur, 1));
    }

    /* ---- a first frame through the VAE's encoder */
    if (!V->image || !m.venc) return;
    const int64_t h = V->geom.lat_h, w = V->geom.lat_w, ppf = h * w;
    const int64_t Hp = 8 * h, Wp = 8 * w, vin0 = Hp * Wp;
    const RadOperand gd = praw(V->geom_d, RAD_I32, RAD_VGEOM_N);
    RAD_ISSUE_N(c, m.op_epatch, Hp, RAD_P_T2(V->image, RAD_BF16, 32 * h, 32 * w * 3),
                view(c, m.b_epatch, RAD_BF16, Hp, Wp * kPixC));
    RAD_ISSUE_N(c, m.op_egeomL, 3, gd, brows(m.b_egeomL, 3));
    for (const VDown& Dn : m.vdown) {
        RAD_ISSUE_N(c, Dn.op_geom_in, 3, gd, brows(Dn.geom_in, 3));
        RAD_ISSUE_N(c, Dn.op_geom, 3, gd, brows(Dn.geom, 3));
    }
    RAD_ISSUE_N(c, m.op_ecin, vin0, view(c, m.b_epatch, RAD_BF16, vin0, kPixC), RAD_W(m.w_ecin),
                RAD_W(m.b_ecin), brows(m.vdown[0].geom_in, 3), RAD_NONE,
                view(c, m.b_ea, RAD_BF16, vin0, m.eC0));
    int64_t vin = vin0;
    for (const VDown& Dn : m.vdown) {
        const RadOperand gin = brows(Dn.geom_in, 3);
        for (const VRes& Rb : Dn.res) vres_step(c, Rb, vin, gin, m.b_ea, m.b_eb, m.b_ec);
        const int64_t vout = vin / (Dn.sh * Dn.sw), tv = Dn.st * vin;
        RAD_ISSUE_N(c, Dn.op_mean, vin, view(c, m.b_ea, RAD_BF16, vin, Dn.cin), gin,
                    view(c, m.b_eb, RAD_BF16, vout, Dn.cout));
        RAD_ISSUE_N(c, Dn.op_conv, vin, view(c, m.b_ea, RAD_BF16, vin, Dn.cin), RAD_W(Dn.w), RAD_W(Dn.b),
                    gin, RAD_NONE, view(c, m.b_ec, RAD_BF16, tv, Dn.conv_out));
        RAD_ISSUE_N(c, Dn.op_s2d, tv, view(c, m.b_ec, RAD_BF16, tv, Dn.conv_out), brows(Dn.geom, 3),
                    view(c, m.b_eb, RAD_BF16, vout, Dn.cout));
        RAD_ISSUE_N(c, Dn.op_next, vout, view(c, m.b_eb, RAD_BF16, vout, Dn.cout),
                    view(c, m.b_ea, RAD_BF16, vout, Dn.cout));
        vin = vout;
    }
    const RadOperand gl = brows(m.b_egeomL, 3);
    for (const VRes& Rb : m.emid) vres_step(c, Rb, ppf, gl, m.b_ea, m.b_eb, m.b_ec);
    RAD_ISSUE_N(c, m.op_enout, ppf, view(c, m.b_ea, RAD_BF16, ppf, m.eCout),
                view(c, m.b_eb, RAD_BF16, ppf, m.eCout));
    RAD_ISSUE_N(c, m.op_ecout, ppf, view(c, m.b_eb, RAD_BF16, ppf, m.eCout), RAD_W(m.w_ecout),
                RAD_W(m.b_ecout), gl, RAD_NONE, brows(m.b_eout, ppf));
    RAD_ISSUE_N(c, m.op_enorm, ppf, bcol(m.b_eout, 0, kC, ppf), RAD_W(m.vd_mean), RAD_W(m.vd_std),
                brows(m.b_elat, ppf));
    RAD_ISSUE_N(c, m.op_ef32, ppf, brows(m.b_elat, ppf),
                brow_slice(m.b_first, lane * m.maxPpf, ppf, kC));
}

/* One DiT evaluation into evaluation `p`'s velocities: the text of prompt `branch` (0 the prompt, 1
 * the negative), `stg` bypassing block 28's self-attentions, `iso` without the cross-modal
 * attentions. */
static void dit_eval(RadCtx* c, const RadVideoGenBatch* V, const Model& m, int p, int branch, bool stg,
                     bool iso) {
    const RadVideoGeom& gm = V->geom;
    const int64_t lane = V->lane, ppf = (int64_t)gm.lat_h * gm.lat_w;
    const int64_t S = (int64_t)gm.lat_frames * ppf, A = gm.n_audio, R = kTextRows, L = m.nb;
    const int64_t D = m.D, Da = m.Da;
    const bool first = V->has_image != 0;
    /* The video's rows take the first frame's modulation (timestep 0) or everything's: two table
     * rows over the first latent frame and the rest, or the second row over all of them. */
    const int64_t vr0 = first ? 0 : 1, vn = first ? 2 : 1;
    const RadOperand cuv = first ? praw(V->geom_d + RAD_VGEOM_FIRST_CU, RAD_I32, 3)
                                 : praw(V->lat_cu, RAD_I32, 2);
    const RadOperand cu1 = praw(V->lat_cu, RAD_I32, 2), cua = praw(V->lat_cu + 2, RAD_I32, 2);
    const RadOperand cut = brows(m.b_rows, 2);
    const RadOperand pv = praw2(V->pos, RAD_F32, 3, S), pvt = praw2(V->pos, RAD_F32, 1, S);
    const RadOperand pa = praw2(V->pos + 3 * S, RAD_F32, 1, A);
    auto mv = [&](int64_t l, int k, int n) {
        return tab(m.b_mv, vr0, vn, L * 9 * D, l * 9 * D + k * D, n * D);
    };
    auto ma = [&](int64_t l, int k, int n) {
        return tab(m.b_ma, 0, 1, L * 9 * Da, l * 9 * Da + k * Da, n * Da);
    };

    RAD_ISSUE_N(c, m.op_lv, S, brow_slice(m.b_vstate, lane * m.maxS, S, kC), brows(m.a_lv.a.x, S));
    RAD_ISSUE_N(c, m.op_la, A, brow_slice(m.b_astate, lane * m.maxA, A, kAudioC), brows(m.a_la.a.x, A));
    m.pin_v.step(c, m.a_lv, m.b_hv, S);
    m.pin_a.step(c, m.a_la, m.b_ha, A);
    const RadOperand tv = brow_slice(m.cv.b_text, (lane * 2 + branch) * R, R, D);
    const RadOperand ta = brow_slice(m.ca.b_text, (lane * 2 + branch) * R, R, Da);

    for (int64_t l = 0; l < L; ++l) {
        const DitBlock& B = m.dit[(size_t)l];
        const bool g0 = l > 0;
        /* ---- self-attention, gated after by gate_msa; spatio-temporal guidance takes v as is */
        RAD_ISSUE_N(c, B.m1v, S, g0 ? brows(m.b_ov, S) : RAD_NONE, g0 ? mv(l - 1, 5, 1) : RAD_NONE,
                    mv(l, 0, 2), RAD_NONE, cuv, brows(m.b_hv, S), brows(m.a_hv.a.x, S));
        RAD_ISSUE_N(c, B.m1a, A, g0 ? brows(m.b_oa, A) : RAD_NONE, g0 ? ma(l - 1, 5, 1) : RAD_NONE,
                    ma(l, 0, 2), RAD_NONE, cua, brows(m.b_ha, A), brows(m.a_ha.a.x, A));
        const bool bypass = stg && l == kStgBlock;
        for (int s = 0; s < 2; ++s) {
            const bool vid = s == 0;
            const Attn& At = vid ? B.sv : B.sa;
            const Act& in = vid ? m.a_hv : m.a_ha;
            const Act& ao = vid ? m.a_aov : m.a_aoa;
            const rad_buf bq = vid ? m.b_qkvv : m.b_qkva, bg = vid ? m.b_gv : m.b_ga;
            const int64_t n = vid ? S : A;
            in.step(c, n);
            At.qkv.step(c, in, bq, n);
            At.gate.step(c, in, bg, n);
            if (bypass) {
                RAD_ISSUE_N(c, At.op_v, n, bcol(bq, 2 * At.inner, At.inner, n), brows(ao.a.x, n));
                RAD_ISSUE_N(c, At.op_hg, n, brows(bg, n), brows(ao.a.x, n));
            } else {
                attn_core(c, At, true, bq, 0, bg, ao, n, n, vid ? cu1 : cua, RAD_NONE, vid ? pv : pa,
                          vid ? pv : pa);
            }
            ao.step(c, n);
            At.out.step(c, ao, vid ? m.b_ov : m.b_oa, n);
        }
        /* ---- text cross-attention: the query's norm gated by gate_msa, the text modulated as
         * it is */
        RAD_ISSUE_N(c, B.m2v, S, brows(m.b_ov, S), mv(l, 2, 1), mv(l, 6, 2), RAD_NONE, cuv,
                    brows(m.b_hv, S), brows(m.a_hv.a.x, S));
        RAD_ISSUE_N(c, B.m2a, A, brows(m.b_oa, A), ma(l, 2, 1), ma(l, 6, 2), RAD_NONE, cua,
                    brows(m.b_ha, A), brows(m.a_ha.a.x, A));
        RAD_ISSUE_N(c, B.mtv, R, RAD_NONE, RAD_NONE, tab(m.b_mp, 0, 1, L * 2 * D, l * 2 * D, 2 * D),
                    RAD_NONE, cut, tv, brows(m.a_ctxv.a.x, R));
        RAD_ISSUE_N(c, B.mta, R, RAD_NONE, RAD_NONE, tab(m.b_mpa, 0, 1, L * 2 * Da, l * 2 * Da, 2 * Da),
                    RAD_NONE, cut, ta, brows(m.a_ctxa.a.x, R));
        for (int s = 0; s < 2; ++s) {
            const bool vid = s == 0;
            const Attn& At = vid ? B.tv : B.ta;
            const Act& in = vid ? m.a_hv : m.a_ha;
            const Act& ctx = vid ? m.a_ctxv : m.a_ctxa;
            const Act& ao = vid ? m.a_aov : m.a_aoa;
            const rad_buf bq = vid ? m.b_qv : m.b_qa, bkv = vid ? m.b_kvtv : m.b_kvta;
            const rad_buf bg = vid ? m.b_gv : m.b_ga;
            const int64_t n = vid ? S : A;
            in.step(c, n);
            ctx.step(c, R);
            At.qkv.step(c, in, bq, n);
            At.gate.step(c, in, bg, n);
            At.kv.step(c, ctx, bkv, R);
            attn_core(c, At, false, bq, bkv, bg, ao, n, R, vid ? cu1 : cua, cut, RAD_NONE, RAD_NONE);
            ao.step(c, n);
            At.out.step(c, ao, vid ? m.b_ov : m.b_oa, n);
        }
        if (!iso) {
            /* ---- the cross-modal attentions: each stream gated by its text attention, normed for
             * a2v and for v2a (both before either update), the two attentions, their gates */
            RAD_ISSUE_N(c, B.g2v, S, brows(m.b_ov, S), mv(l, 8, 1), cuv, brows(m.b_hv, S));
            RAD_ISSUE_N(c, B.g2a, A, brows(m.b_oa, A), ma(l, 8, 1), cua, brows(m.b_ha, A));
            RAD_ISSUE_N(c, B.mav, S, RAD_NONE, RAD_NONE, tab(m.b_mcv, 0, 1, L * 4 * D, l * 4 * D, 2 * D),
                        RAD_NONE, cu1, brows(m.b_hv, S), brows(m.a_nv.a.x, S));
            RAD_ISSUE_N(c, B.mvv, S, RAD_NONE, RAD_NONE,
                        tab(m.b_mcv, 0, 1, L * 4 * D, l * 4 * D + 2 * D, 2 * D), RAD_NONE, cu1,
                        brows(m.b_hv, S), brows(m.a_nv2.a.x, S));
            RAD_ISSUE_N(c, B.maa, A, RAD_NONE, RAD_NONE,
                        tab(m.b_mca, 0, 1, L * 4 * Da, l * 4 * Da, 2 * Da), RAD_NONE, cua,
                        brows(m.b_ha, A), brows(m.a_na.a.x, A));
            RAD_ISSUE_N(c, B.mva, A, RAD_NONE, RAD_NONE,
                        tab(m.b_mca, 0, 1, L * 4 * Da, l * 4 * Da + 2 * Da, 2 * Da), RAD_NONE, cua,
                        brows(m.b_ha, A), brows(m.a_na2.a.x, A));
            m.a_nv.step(c, S);
            m.a_nv2.step(c, S);
            m.a_na.step(c, A);
            m.a_na2.step(c, A);
            /* a2v: video queries over the soundtrack's keys, each on its time axis. */
            B.a2v.qkv.step(c, m.a_nv, m.b_qxv, S);
            B.a2v.gate.step(c, m.a_nv, m.b_gxv, S);
            B.a2v.kv.step(c, m.a_na, m.b_kvxa, A);
            attn_core(c, B.a2v, false, m.b_qxv, m.b_kvxa, m.b_gxv, m.a_xv, S, A, cu1, cua, pvt, pa);
            m.a_xv.step(c, S);
            B.a2v.out.step(c, m.a_xv, m.b_ov, S);
            /* v2a: the soundtrack's queries over the video's keys. */
            B.v2a.qkv.step(c, m.a_na2, m.b_qxa, A);
            B.v2a.gate.step(c, m.a_na2, m.b_gxa, A);
            B.v2a.kv.step(c, m.a_nv2, m.b_kvxv, S);
            attn_core(c, B.v2a, false, m.b_qxa, m.b_kvxv, m.b_gxa, m.a_xa, A, S, cua, cu1, pa, pvt);
            m.a_xa.step(c, A);
            B.v2a.out.step(c, m.a_xa, m.b_oa, A);
            RAD_ISSUE_N(c, B.gcv, S, brows(m.b_ov, S), tab(m.b_mcvg, 0, 1, L * D, l * D, D), cu1,
                        brows(m.b_hv, S));
            RAD_ISSUE_N(c, B.gca, A, brows(m.b_oa, A), tab(m.b_mcag, 0, 1, L * Da, l * Da, Da), cua,
                        brows(m.b_ha, A));
            RAD_ISSUE_N(c, B.m3v, S, RAD_NONE, RAD_NONE, mv(l, 3, 2), RAD_NONE, cuv, brows(m.b_hv, S),
                        brows(m.a_hv.a.x, S));
            RAD_ISSUE_N(c, B.m3a, A, RAD_NONE, RAD_NONE, ma(l, 3, 2), RAD_NONE, cua, brows(m.b_ha, A),
                        brows(m.a_ha.a.x, A));
        } else {
            RAD_ISSUE_N(c, B.m3vg, S, brows(m.b_ov, S), mv(l, 8, 1), mv(l, 3, 2), RAD_NONE, cuv,
                        brows(m.b_hv, S), brows(m.a_hv.a.x, S));
            RAD_ISSUE_N(c, B.m3ag, A, brows(m.b_oa, A), ma(l, 8, 1), ma(l, 3, 2), RAD_NONE, cua,
                        brows(m.b_ha, A), brows(m.a_ha.a.x, A));
        }
        /* ---- the feed-forwards, gated by gate_mlp in the next block's first norm */
        m.a_hv.step(c, S);
        B.fv_up.step(c, m.a_hv, m.b_upv, S);
        RAD_ISSUE_N(c, B.actv, S, brows(m.b_upv, S), brows(m.a_ffv.a.x, S));
        m.a_ffv.step(c, S);
        B.fv_down.step(c, m.a_ffv, m.b_ov, S);
        m.a_ha.step(c, A);
        B.fa_up.step(c, m.a_ha, m.b_upa, A);
        RAD_ISSUE_N(c, B.acta, A, brows(m.b_upa, A), brows(m.a_ffa.a.x, A));
        m.a_ffa.step(c, A);
        B.fa_down.step(c, m.a_ffa, m.b_oa, A);
    }
    RAD_ISSUE_N(c, m.op_fnv, S, brows(m.b_ov, S), mv(L - 1, 5, 1), tab(m.b_fv, vr0, vn, 2 * D, 0, 2 * D),
                RAD_NONE, cuv, brows(m.b_hv, S), brows(m.a_fv.a.x, S));
    RAD_ISSUE_N(c, m.op_fna, A, brows(m.b_oa, A), ma(L - 1, 5, 1), tab(m.b_fa, 0, 1, 2 * Da, 0, 2 * Da),
                RAD_NONE, cua, brows(m.b_ha, A), brows(m.a_fa.a.x, A));
    m.pout_v.step(c, m.a_fv, m.b_vel_v[p], S);
    m.pout_a.step(c, m.a_fa, m.b_vel_a[p], A);
}

/* One denoising step: the first step's noise (and first frame), the step's tables, each
 * evaluation the request's guidance needs, each modality's guided Euler step, the first frame
 * restored. */
static void step_dit(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    const RadVideoGeom& gm = V->geom;
    const int64_t lane = V->lane, ppf = (int64_t)gm.lat_h * gm.lat_w;
    const int64_t S = (int64_t)gm.lat_frames * ppf, A = gm.n_audio, L = m.nb, La = kMaxLanes;
    const bool first = V->has_image != 0;
    const RadOperand lane_d = praw(V->lane_d, RAD_I32, 1);
    const RadOperand keep = brow_slice(m.b_vstate, lane * m.maxS, ppf, kC);
    const RadOperand clean = brow_slice(m.b_first, lane * m.maxPpf, ppf, kC);
    if (V->step == 0) {
        RAD_ISSUE_N(c, m.op_rnd_v, S, praw(V->first_d, RAD_I32, 1), praw(V->seed, RAD_U32, 2), lane_d,
                    praw(V->lat_cu, RAD_I32, 2), brows(m.b_vstate, La * m.maxS));
        RAD_ISSUE_N(c, m.op_rnd_a, A, praw(V->first_d, RAD_I32, 1), praw(V->seed + 2, RAD_U32, 2),
                    lane_d, praw(V->lat_cu + 2, RAD_I32, 2), brows(m.b_astate, La * m.maxA));
        if (first) RAD_ISSUE_N(c, m.op_keep_first, ppf, clean, keep);
    }

    /* ---- the step's tables: {0, sigma} through the eight modulation MLPs (the video's both
     * rows, the rest sigma's), then each block's rows */
    RAD_ISSUE_N(c, m.op_tin0, 1, view(c, m.b_tin, RAD_F32, 1, 2));
    RAD_ISSUE_N(c, m.op_tin1, 1, praw2(V->sched + 1, RAD_F32, 1, 1), view(c, m.b_tin, RAD_F32, 1, 1, 1));
    RAD_ISSUE_N(c, m.op_tsin, 2, praw(rad_buf_ptr(c, m.b_tin), RAD_F32, 2), brows(m.b_tsin, 2));
    ada_step(c, m.ad_v, m.b_tsin, 0, 2);
    for (const Ada* a : { &m.ad_a, &m.ad_p, &m.ad_pa, &m.ad_cv, &m.ad_cvg, &m.ad_ca, &m.ad_cag })
        ada_step(c, *a, m.b_tsin, 1, 1);
    const int64_t D = m.D, Da = m.Da;
    auto temb = [&](const Ada& a, int64_t r0, int64_t n) {
        return brow_slice(a.temb, r0, n, a.k * a.d);
    };
    for (int64_t l = 0; l < L; ++l) {
        const DitBlock& B = m.dit[(size_t)l];
        RAD_ISSUE_N(c, B.op_tv, 2, temb(m.ad_v, 0, 2), RAD_W(B.t_v),
                    tab(m.b_mv, 0, 2, L * 9 * D, l * 9 * D, 9 * D));
        RAD_ISSUE_N(c, B.op_ta, 1, temb(m.ad_a, 1, 1), RAD_W(B.t_a),
                    tab(m.b_ma, 0, 1, L * 9 * Da, l * 9 * Da, 9 * Da));
        RAD_ISSUE_N(c, B.op_tp, 1, temb(m.ad_p, 1, 1), RAD_W(B.t_p),
                    tab(m.b_mp, 0, 1, L * 2 * D, l * 2 * D, 2 * D));
        RAD_ISSUE_N(c, B.op_tpa, 1, temb(m.ad_pa, 1, 1), RAD_W(B.t_pa),
                    tab(m.b_mpa, 0, 1, L * 2 * Da, l * 2 * Da, 2 * Da));
        RAD_ISSUE_N(c, B.op_tcv, 1, temb(m.ad_cv, 1, 1), RAD_W(B.t_cv),
                    tab(m.b_mcv, 0, 1, L * 4 * D, l * 4 * D, 4 * D));
        RAD_ISSUE_N(c, B.op_tcvg, 1, temb(m.ad_cvg, 1, 1), RAD_W(B.t_cvg),
                    tab(m.b_mcvg, 0, 1, L * D, l * D, D));
        RAD_ISSUE_N(c, B.op_tca, 1, temb(m.ad_ca, 1, 1), RAD_W(B.t_ca),
                    tab(m.b_mca, 0, 1, L * 4 * Da, l * 4 * Da, 4 * Da));
        RAD_ISSUE_N(c, B.op_tcag, 1, temb(m.ad_cag, 1, 1), RAD_W(B.t_cag),
                    tab(m.b_mcag, 0, 1, L * Da, l * Da, Da));
    }
    RAD_ISSUE_N(c, m.op_fv, 2, brows(m.ad_v.emb, 2), RAD_W(m.w_fv), brows(m.b_fv, 2));
    RAD_ISSUE_N(c, m.op_fa, 1, brow_slice(m.ad_a.emb, 1, 1, Da), RAD_W(m.w_fa), brows(m.b_fa, 1));

    /* ---- the evaluations: a term is on for either modality, each modality weighs its own */
    const float* gs = V->guidance_h;
    const bool cfg = gs[kGsV] != 1.0f || gs[kGsA] != 1.0f;
    const bool stg = gs[kStgV] != 0.0f || gs[kStgA] != 0.0f;
    const bool iso = gs[kMsV] != 1.0f || gs[kMsA] != 1.0f;
    dit_eval(c, V, m, kPassCond, 0, false, false);
    if (cfg) dit_eval(c, V, m, kPassUncond, 1, false, false);
    if (stg) dit_eval(c, V, m, kPassStg, 0, true, false);
    if (iso) dit_eval(c, V, m, kPassIso, 0, false, true);
    auto vel = [&](const rad_buf* v, int p, bool on, int64_t n) {
        return on ? brows(v[p], n) : RAD_NONE;
    };
    RAD_ISSUE_N(c, m.op_gv, S, vel(m.b_vel_v, kPassCond, true, S), vel(m.b_vel_v, kPassUncond, cfg, S),
                vel(m.b_vel_v, kPassStg, stg, S), vel(m.b_vel_v, kPassIso, iso, S),
                praw(V->guidance, RAD_F32, 4), praw(V->sched + 1, RAD_F32, 2), RAD_B(m.b_gst),
                brow_slice(m.b_vstate, lane * m.maxS, S, kC));
    RAD_ISSUE_N(c, m.op_ga, A, vel(m.b_vel_a, kPassCond, true, A), vel(m.b_vel_a, kPassUncond, cfg, A),
                vel(m.b_vel_a, kPassStg, stg, A), vel(m.b_vel_a, kPassIso, iso, A),
                praw(V->guidance + 4, RAD_F32, 4), praw(V->sched + 4, RAD_F32, 2), RAD_B(m.b_gst),
                brow_slice(m.b_astate, lane * m.maxA, A, kAudioC));
    if (first) RAD_ISSUE_N(c, m.op_keep_first, ppf, clean, keep);
}

/* A vocoder over `rows0` input rows: each stage's sequence bounds, conv_in, the stages, the last
 * activation and conv_out. */
static void voc_step(RadCtx* c, const Voc& V, const RadVideoGenBatch* B, int64_t rows0) {
    const RadOperand gd = praw(B->geom_d, RAD_I32, RAD_VGEOM_N);
    auto conv = [&](const VConv& k, rad_buf x, rad_buf res, rad_buf y, int64_t R, rad_buf cu) {
        RAD_ISSUE_N(c, k.op, R, brows(x, R), RAD_W(k.w), brows(cu, 2), res ? brows(res, R) : RAD_NONE,
                    brows(y, R), k.b ? RAD_W(k.b) : RAD_NONE);
    };
    auto act = [&](const VAct& a, rad_buf x, rad_buf y, int64_t R, rad_buf cu) {
        RAD_ISSUE_N(c, a.op, R, brows(x, R), RAD_W(a.alpha), RAD_W(a.beta), RAD_W(a.up), RAD_W(a.down),
                    brows(cu, 2), brows(y, R));
    };
    RAD_ISSUE_N(c, V.op_cu0, 2, gd, brows(V.cu[0], 2));
    conv(V.cin, V.in_buf, 0, V.avg[0], rows0, V.cu[0]);
    int64_t rows = rows0;
    for (size_t s = 0; s < V.st.size(); ++s) {
        const VStage& St = V.st[s];
        const int64_t R = rows * St.rate;
        RAD_ISSUE_N(c, St.op_cu, 2, gd, brows(V.cu[s + 1], 2));
        RAD_ISSUE_N(c, St.up.op, rows, brows(V.avg[s], rows), RAD_W(St.up.w), RAD_W(St.up.b),
                    brows(V.cu[s], 2), brows(V.h[s + 1], R));
        for (int j = 0; j < 3; ++j)
            for (int d = 0; d < 3; ++d) {
                const VUnit& U = St.u[j][d];
                const rad_buf x = d == 0 ? V.h[s + 1] : V.amp[j][s + 1];
                act(U.a1, x, V.t1[s + 1], R, V.cu[s + 1]);
                conv(U.c1, V.t1[s + 1], 0, V.t2[s + 1], R, V.cu[s + 1]);
                act(U.a2, V.t2[s + 1], V.t1[s + 1], R, V.cu[s + 1]);
                conv(U.c2, V.t1[s + 1], x, V.amp[j][s + 1], R, V.cu[s + 1]);
            }
        RAD_ISSUE_N(c, St.op_sum, R, brows(V.amp[0][s + 1], R), brows(V.amp[1][s + 1], R),
                    brows(V.avg[s + 1], R));
        RAD_ISSUE_N(c, St.op_avg, R, brows(V.avg[s + 1], R), brows(V.amp[2][s + 1], R),
                    brows(V.avg[s + 1], R));
        rows = R;
    }
    const size_t ns = V.st.size();
    act(V.act, V.avg[ns], V.t1[ns], rows, V.cu[ns]);
    conv(V.cout, V.t1[ns], 0, V.wav, rows, V.cu[ns]);
}

/* The soundtrack, once after the last step. */
static void step_audio(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    const int64_t A = V->geom.n_audio, lane = V->lane, Tm = mel_frames(A);
    RAD_ISSUE_N(c, m.op_aden, A, brow_slice(m.b_astate, lane * m.maxA, A, kAudioC), RAD_W(m.w_amean),
                RAD_W(m.w_astd), brows(m.b_aden, A));
    RAD_ISSUE_N(c, m.op_acast, A, brows(m.b_aden, A), brows(m.b_alat, A));
    RAD_ISSUE_N(c, m.op_aunpack, A, brows(m.b_alat, A),
                praw(V->geom_d + RAD_VGEOM_AUDIO_VOL, RAD_I32, 3),
                brows(m.b_aimg, A * kLatMel));
    int64_t H = A, W = kLatMel;
    RAD_ISSUE_N(c, m.op_acin, H * W, view(c, m.b_aimg, RAD_BF16, H, W * kLatCh), RAD_W(m.w_acin),
                RAD_W(m.b_acin), view(c, m.b_aa, RAD_BF16, H, W * m.aC0));
    for (const ARes& Rb : m.amid) ares_step(c, Rb, H, W, m.b_aa, m.b_ab, m.b_ac, m.b_as);
    for (int lev = 2; lev >= 0; --lev) {
        for (const ARes& Rb : m.aup[(size_t)(2 - lev)])
            ares_step(c, Rb, H, W, m.b_aa, m.b_ab, m.b_ac, m.b_as);
        if (lev == 0) break;
        const int64_t co = m.alev_c[(size_t)lev];
        const VConv& U = m.aups[(size_t)(2 - lev)];
        /* The nearest x2 upsample read by the conv, then the first time row dropped. */
        RAD_ISSUE_N(c, U.op, 4 * H * W, view(c, m.b_aa, RAD_BF16, H, W * co), RAD_W(U.w), RAD_W(U.b),
                    view(c, m.b_ab, RAD_BF16, 2 * H, 2 * W * co));
        const int64_t px = (2 * H - 1) * 2 * W;
        RAD_ISSUE_N(c, m.aups_cp[(size_t)(2 - lev)], px,
                    view(c, m.b_ab, RAD_BF16, px, co, 2 * W * co), view(c, m.b_aa, RAD_BF16, px, co));
        H = 2 * H - 1;
        W *= 2;
    }
    const int64_t C0 = m.alev_c[0];
    RAD_ISSUE_N(c, m.op_anout, H * W, view(c, m.b_aa, RAD_BF16, H * W, C0),
                view(c, m.b_ab, RAD_BF16, H * W, C0));
    for (int ch = 0; ch < 2; ++ch)
        RAD_ISSUE_N(c, m.op_acout[ch], H * W, view(c, m.b_ab, RAD_BF16, H, W * C0), RAD_W(m.w_acout[ch]),
                    RAD_W(m.b_acout[ch]), bcol(m.b_amel16, ch * kMel, kMel, Tm));
    RAD_ISSUE_N(c, m.op_amel, Tm, brows(m.b_amel16, Tm), brows(m.b_amel, Tm));
    voc_step(c, m.voc, V, Tm);
    const int64_t Sw = Tm * kMelHop;
    RAD_ISSUE_N(c, m.op_stft, Sw / 80, brows(m.voc.wav, Sw), RAD_W(m.w_basis), RAD_W(m.w_melb),
                brows(m.b_bmel16, Sw / 80));
    RAD_ISSUE_N(c, m.op_bmel, Sw / 80, brows(m.b_bmel16, Sw / 80), brows(m.b_bmel, Sw / 80));
    voc_step(c, m.bwe, V, Sw / 80);
    RAD_ISSUE_N(c, m.op_bwe, 3 * Sw, brows(m.voc.wav, Sw), brows(m.bwe.wav, 3 * Sw),
                brows(m.b_audio, 3 * Sw));
    const int64_t tail = V->n_samples - 3 * Sw;
    if (tail > 0)
        RAD_ISSUE_N(c, m.op_tail, tail, brow_slice(m.b_audio, 3 * Sw, tail, kAudioCh));
}

/* One chunk of frames: chunk 0 decodes the whole video into the lane's frame store first; every
 * chunk turns its frames into bytes. */
/* The decoder over one tile's latent (b_vtlat: F x h x w voxels, `tg` that volume on the device)
 * to its pixel patches at `out`. */
static void vae_decode(RadCtx* c, const Model& m, int64_t F, int64_t h, int64_t w,
                       const RadOperand& tg, const RadOperand& out) {
    const int64_t S = F * h * w;
    RAD_ISSUE_N(c, m.op_vcin, S, brows(m.b_vtlat, S), RAD_W(m.w_vcin), RAD_W(m.b_vcin), tg, RAD_NONE,
                view(c, m.b_va, RAD_BF16, S, m.vC0));
    for (const VRes& Rb : m.vmid) vres_step(c, Rb, S, tg, m.b_va, m.b_vb, m.b_vc);
    int64_t T = F, Hh = h, Ww = w;
    RadOperand gin = tg;
    for (const VUp& U : m.vup) {
        const int64_t vin = T * Hh * Ww;
        RAD_ISSUE_N(c, U.op_geom, 3, tg, brows(U.geom, 3));
        RAD_ISSUE_N(c, U.op_conv, vin, view(c, m.b_va, RAD_BF16, vin, U.cin), RAD_W(U.w), RAD_W(U.b),
                    gin, RAD_NONE, view(c, m.b_vb, RAD_BF16, vin, U.conv_out));
        T = T * U.st - (U.st - 1);
        Hh *= U.sh;
        Ww *= U.sw;
        const int64_t vout = T * Hh * Ww;
        RAD_ISSUE_N(c, U.op_d2s, vin, view(c, m.b_vb, RAD_BF16, vin, U.conv_out), gin,
                    view(c, m.b_va, RAD_BF16, vout, U.cout));
        gin = brows(U.geom, 3);
        for (const VRes& Rb : U.res) vres_step(c, Rb, vout, gin, m.b_va, m.b_vb, m.b_vc);
    }
    const int64_t vout = T * Hh * Ww, C = m.vup.back().cout;
    RAD_ISSUE_N(c, m.op_vnout, vout, view(c, m.b_va, RAD_BF16, vout, C),
                view(c, m.b_vb, RAD_BF16, vout, C));
    RAD_ISSUE_N(c, m.op_vcout, vout, view(c, m.b_vb, RAD_BF16, vout, C), RAD_W(m.w_vcout),
                RAD_W(m.b_vcout), gin, RAD_NONE, out);
}

/* One diffusion-decoder block over V voxels of residual `xb` (a C-wide view), the volume `geom`
 * on the device and its positions already in b_dpos. */
static void dd_block_step(RadCtx* c, const Model& m, const DdBlock& B, int64_t V, rad_buf xb,
                          const RadOperand& geom, bool diff) {
    const int64_t C = B.C;
    const RadOperand X = view(c, xb, RAD_BF16, V, C), N = view(c, m.b_dn, RAD_BF16, V, C);
    const rad_buf qkv = m.b_dqkv[B.slot], gu = m.b_dg[B.slot];
    if (diff) {
        RAD_ISSUE_N(c, B.op_ctx, V, view(c, m.b_dctx, RAD_BF16, V, C), RAD_W(B.w_ctx), RAD_W(B.b_ctx),
                    X, X);
        RAD_ISSUE_N(c, B.op_mod, 1, brows(m.b_dmod, 1), RAD_W(B.w_sst), brows(m.b_dbmod, 1));
        RAD_ISSUE_N(c, B.op_n1, V, RAD_NONE, RAD_NONE, bcol(m.b_dbmod, 0, 2 * C, 1), RAD_W(B.w_n1),
                    brows(m.b_ddcu, 2), X, N);
    } else {
        RAD_ISSUE_N(c, B.op_n1, V, X, RAD_W(B.w_n1), N);
    }
    const int64_t gw = C / (int64_t)B.op_qkvh.size();
    for (size_t g = 0; g < B.op_qkvh.size(); ++g) {
        RAD_ISSUE_N(c, B.op_qkvh[g], V, N, RAD_W(B.w_qkvh[g]), RAD_W(B.b_qkvh[g]), RAD_NONE,
                    brows(qkv, V));
        RAD_ISSUE_N(c, B.op_rope, V, brows(qkv, V), bcol(m.b_dpos, 0, V, 3), RAD_W(B.w_nq),
                    RAD_W(B.w_nk));
        RAD_ISSUE_N(c, B.op_attn, V, bcol(qkv, 0, gw, V), bcol(qkv, gw, gw, V), bcol(qkv, 2 * gw, gw, V),
                    geom, diff ? bcol(m.b_dao, (int64_t)g * gw, gw, V) : N);
    }
    RAD_ISSUE_N(c, B.op_o, V, diff ? view(c, m.b_dao, RAD_BF16, V, C) : N, RAD_W(B.w_o), RAD_W(B.b_o),
                X, X);
    if (diff)
        RAD_ISSUE_N(c, B.op_n2, V, RAD_NONE, RAD_NONE, bcol(m.b_dbmod, 2 * C, 2 * C, 1), RAD_W(B.w_n2),
                    brows(m.b_ddcu, 2), X, N);
    else
        RAD_ISSUE_N(c, B.op_n2, V, X, RAD_W(B.w_n2), N);
    for (int64_t r0 = 0; r0 < V; r0 += B.chunk) {
        const int64_t n = std::min(B.chunk, V - r0);
        const RadOperand Nr = view(c, m.b_dn, RAD_BF16, n, C, r0 * C);
        RAD_ISSUE_N(c, B.op_gu, n, Nr, RAD_W(B.w_gu), brows(gu, n));
        RAD_ISSUE_N(c, B.op_act, n, brows(gu, n), view(c, m.b_da, RAD_BF16, n, B.ff));
        RAD_ISSUE_N(c, B.op_down, n, view(c, m.b_da, RAD_BF16, n, B.ff), RAD_W(B.w_down),
                    view(c, m.b_dmo, RAD_BF16, n, C));
        const RadOperand Xr = view(c, xb, RAD_BF16, n, C, r0 * C);
        RAD_ISSUE_N(c, B.op_add2, n, Xr, view(c, m.b_dmo, RAD_BF16, n, C), Xr);
    }
}

/* The diffusion decoder over the lane's latent (b_vlat, cast and denormalised) into the tile store
 * (chunk 0 of a request that named it); the chunks then stitch from the store with op_dst. */
static void step_ddec(RadCtx* c, const RadVideoGenBatch* V, const Model& m) {
    const RadVideoGeom& gm = V->geom;
    const int64_t h = gm.lat_h, w = gm.lat_w, F = gm.lat_frames, A = h * w, S = F * A;
    const RadOperand gd = praw(V->geom_d, RAD_I32, RAD_VGEOM_N);
    /* The latent and its last frame twice more. */
    RAD_ISSUE_N(c, m.op_dpad, S, brows(m.b_vlat, S), view(c, m.b_dz, RAD_BF16, S, kC));
    for (int64_t g = 0; g < kDdGhost; ++g)
        RAD_ISSUE_N(c, m.op_dpad, A, view(c, m.b_vlat, RAD_BF16, A, kC, (F - 1) * A * kC),
                    view(c, m.b_dz, RAD_BF16, A, kC, (S + g * A) * kC));
    for (const DdStage& St : m.dst) RAD_ISSUE_N(c, St.op_geom, 3, gd, brows(St.geom, 3));
    RAD_ISSUE_N(c, m.op_dtgeom, 3, gd, brows(m.b_dtg, 3));
    RAD_ISSUE_N(c, m.op_dgrid, kDdMaxTiles, brows(m.b_dtg, 3), brows(m.b_dgrid, kDdMaxTiles));
    RAD_ISSUE_N(c, m.op_dtin, 1, brows(m.b_dtin, 1));
    RAD_ISSUE_N(c, m.op_dtsin, 1, brows(m.b_dtin, 1), brows(m.b_dtsin, 1));
    RAD_ISSUE_N(c, m.op_dt1, 1, brows(m.b_dtsin, 1), RAD_W(m.dw_t1), RAD_W(m.db_t1), RAD_NONE,
                brows(m.b_dth, 1));
    RAD_ISSUE_N(c, m.op_dts, 1, brows(m.b_dth, 1), brows(m.b_dths, 1));
    RAD_ISSUE_N(c, m.op_dt2, 1, brows(m.b_dths, 1), RAD_W(m.dw_t2), RAD_W(m.db_t2), RAD_NONE,
                brows(m.b_dtemb, 1));
    RAD_ISSUE_N(c, m.op_das, 1, brows(m.b_dtemb, 1), brows(m.b_dtes, 1));
    RAD_ISSUE_N(c, m.op_dad, 1, brows(m.b_dtes, 1), RAD_W(m.dw_ad), RAD_W(m.db_ad), RAD_NONE,
                brows(m.b_dmod, 1));
    RAD_ISSUE_N(c, m.op_ddcu, 2, brows(m.b_ddcu, 2));
    RAD_ISSUE_N(c, m.op_dsig, 1, brows(m.b_dsig, 1));

    /* ---- stages 0..2 over the whole latent into the feature volume */
    int64_t T = F + kDdGhost, Hh = h, Ww = w;
    const int64_t V0 = T * Hh * Ww;
    RAD_ISSUE_N(c, m.op_dcin, V0, view(c, m.b_dz, RAD_BF16, V0, kC), RAD_W(m.dw_cin), RAD_W(m.db_cin),
                RAD_NONE, view(c, m.b_dx, RAD_BF16, V0, 2048));
    for (int s = 0; s < 3; ++s) {
        const DdStage& St = m.dst[(size_t)s];
        const int64_t Vs = T * Hh * Ww;
        const RadOperand g = brows(St.geom, 3);
        RAD_ISSUE_N(c, St.op_pos, Vs, g, bcol(m.b_dpos, 0, Vs, 3));
        for (const DdBlock& B : St.blk) dd_block_step(c, m, B, Vs, m.b_dx, g, false);
        const int64_t P = St.st * St.sh * St.sw * St.cout;
        const rad_buf scratch = m.b_dqkv[s];
        RAD_ISSUE_N(c, St.op_proj, Vs, view(c, m.b_dx, RAD_BF16, Vs, St.C), RAD_W(St.w_up),
                    RAD_W(St.b_up), RAD_NONE, view(c, scratch, RAD_BF16, Vs, P));
        T = T * St.st - (St.st - 1);
        Hh *= St.sh;
        Ww *= St.sw;
        const rad_buf dest = s < 2 ? m.b_dx : m.b_dfeat;
        RAD_ISSUE_N(c, St.op_d2s, Vs, view(c, scratch, RAD_BF16, Vs, P), g,
                    view(c, dest, RAD_BF16, T * Hh * Ww, St.cout));
    }
    const int64_t V3 = T * Hh * Ww;

    /* ---- per tile: stage 3, then the diffusion step into the store */
    const DdStage& S3 = m.dst[3];
    const std::vector<DdTile> tl = dd_tiles(4 * F - 3, Hh, Ww);
    const int32_t* rows = (const int32_t*)rad_buf_ptr(c, m.b_dgrid);
    int64_t off = 0;
    for (size_t k = 0; k < tl.size() && k < (size_t)kDdMaxTiles; ++k) {
        const DdTile& t = tl[k];
        const int64_t V4 = t.t * t.h * t.w;
        const RadOperand tg = praw(rows + 8 * k, RAD_I32, 3);
        RAD_ISSUE_N(c, m.op_dcrop, V4, view(c, m.b_dfeat, RAD_BF16, V3, 512), brows(S3.geom, 3),
                    praw(rows + 8 * k, RAD_I32, 6), view(c, m.b_d4, RAD_BF16, V4, 512));
        RAD_ISSUE_N(c, S3.op_pos, V4, tg, bcol(m.b_dpos, 0, V4, 3));
        for (const DdBlock& B : S3.blk) dd_block_step(c, m, B, V4, m.b_d4, tg, false);
        const int64_t P = S3.st * S3.sh * S3.sw * S3.cout;
        RAD_ISSUE_N(c, S3.op_proj, V4, view(c, m.b_d4, RAD_BF16, V4, 512), RAD_W(S3.w_up),
                    RAD_W(S3.b_up), RAD_NONE, view(c, m.b_dqkv[2], RAD_BF16, V4, P));
        const int64_t drop = t.origin ? 1 : 0, Fc = 2 * t.t - drop;
        RAD_ISSUE_N(c, t.origin ? S3.op_d2s : S3.op_d2s_keep, V4,
                    view(c, m.b_dqkv[2], RAD_BF16, V4, P), tg,
                    view(c, m.b_dctx, RAD_BF16, Fc * 4 * t.h * t.w, 256));
        const int64_t F5 = Fc - (t.trailing ? 2 * kDdGhostCells : 0), V5 = F5 * 4 * t.h * t.w;
        RAD_ISSUE_N(c, m.op_dg5[(t.origin ? 2 : 0) + (t.trailing ? 1 : 0)], 3, tg, brows(m.b_dg5, 3));
        const RadOperand g5 = brows(m.b_dg5, 3);
        RAD_ISSUE_N(c, m.op_dpos5, V5, g5, bcol(m.b_dpos, 0, V5, 3));
        RAD_ISSUE_N(c, m.op_dz, V5, brows(m.b_dxt, V5));
        RAD_ISSUE_N(c, m.op_dnoise[k], V5, brows(m.b_dsig, 1), praw(V->seed, RAD_U32, 2),
                    brows(m.b_dxt, V5));
        RAD_ISSUE_N(c, m.op_dxcast, V5, brows(m.b_dxt, V5), brows(m.b_dxb, V5));
        RAD_ISSUE_N(c, m.op_dxin, V5, brows(m.b_dxb, V5), RAD_W(m.dw_xin), RAD_W(m.db_xin), RAD_NONE,
                    view(c, m.b_ddh, RAD_BF16, V5, 256));
        for (const DdBlock& B : m.dblk) dd_block_step(c, m, B, V5, m.b_ddh, g5, true);
        RAD_ISSUE_N(c, m.op_dnout, V5, view(c, m.b_ddh, RAD_BF16, V5, 256), RAD_W(m.dw_nout),
                    view(c, m.b_dn, RAD_BF16, V5, 256));
        RAD_ISSUE_N(c, m.op_ddout, V5, view(c, m.b_dn, RAD_BF16, V5, 256), RAD_W(m.dw_cout),
                    RAD_W(m.db_cout), RAD_NONE, view(c, m.b_vtiles, RAD_BF16, V5, kPixC, off * kPixC));
        off += V5;
    }
}

/* Chunk 0 decodes the whole video into the lane's tile store, a tile at a time (the boxes made on
 * the device by tile_grid, their sizes and store offsets known here); every chunk stitches its
 * frames to bytes. */
static void step_video(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    const RadVideoGeom& gm = V->geom;
    const int64_t lane = V->lane, h = gm.lat_h, w = gm.lat_w, F = gm.lat_frames, S = F * h * w;
    const int64_t Fo = m.vAt * F + m.vBt, sc = m.vAhw;
    const bool tiled = vae_tiled(h, w);
    const std::vector<VaeTile> tl = vae_tiles(h, w);
    int64_t store = 0;
    for (const VaeTile& t : tl) store += Fo * t.h * sc * t.w * sc;
    const RadOperand gd = praw(V->geom_d, RAD_I32, RAD_VGEOM_N);
    const bool diffusion = V->decoder == 1 && m.ddec;
    if (V->chunk == 0) {
        RAD_ISSUE_N(c, m.op_vlat, S, brow_slice(m.b_vstate, lane * m.maxS, S, kC), brows(m.b_vlat, S));
        RAD_ISSUE_N(c, m.op_vden, S, brows(m.b_vlat, S), RAD_W(m.vd_mean), RAD_W(m.vd_std),
                    brows(m.b_vlat, S));
    }
    if (V->chunk == 0 && diffusion) step_ddec(c, V, m);
    if (V->chunk == 0 && !diffusion) {
        const RadOperand g0 = praw(V->geom_d, RAD_I32, 3);
        RAD_ISSUE_N(c, tiled ? m.op_vgrid : m.op_vgrid1, m.maxTiles, g0, brows(m.b_vgrid, m.maxTiles));
        const int32_t* rows = (const int32_t*)rad_buf_ptr(c, m.b_vgrid);
        int64_t off = 0;
        for (size_t k = 0; k < tl.size(); ++k) {
            const int64_t tv = F * tl[k].h * tl[k].w, n = Fo * tl[k].h * sc * tl[k].w * sc;
            const RadOperand tg = praw(rows + 8 * k, RAD_I32, 3);
            RAD_ISSUE_N(c, m.op_vcrop, tv, brows(m.b_vlat, S), g0, praw(rows + 8 * k, RAD_I32, 6),
                        brows(m.b_vtlat, tv));
            vae_decode(c, m, F, tl[k].h, tl[k].w, tg,
                       view(c, m.b_vtiles, RAD_BF16, n, kPixC, off * kPixC));
            off += n;
        }
    }
    const int64_t R = (int64_t)V->n_frames_out * h * kScale, Wpx = w * kScale;
    RAD_ISSUE_N(c, m.op_vf0, 1, gd, brows(m.b_vf0, 1));
    if (diffusion) {
        int64_t drows = 0;
        for (const DdTile& t : dd_tiles(4 * F - 3, 4 * h, 4 * w))
            drows += (2 * (t.t - (t.trailing ? kDdGhostCells : 0)) - (t.origin ? 1 : 0)) * 4 * t.h * t.w;
        RAD_ISSUE_N(c, m.op_dtgeom, 3, gd, brows(m.b_dtg, 3));
        RAD_ISSUE_N(c, m.op_dst, R, brows(m.b_vtiles, drows), brows(m.b_dtg, 3), brows(m.b_vf0, 1),
                    view(c, m.b_frames, RAD_U8, R, Wpx * 3));
        return;
    }
    RAD_ISSUE_N(c, tiled ? m.op_vst : m.op_vst1, R, brows(m.b_vtiles, store),
                praw(V->geom_d, RAD_I32, 3), brows(m.b_vf0, 1), view(c, m.b_frames, RAD_U8, R, Wpx * 3));
}

/* The latent upscale: the clean small latent through the upsampler to the full size, both latents
 * re-noised to the second stage's first sigma with streams of their own, a first frame kept
 * clean. */
static void step_upscale(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    const RadVideoGeom& s = V->geom_from;
    const RadVideoGeom& d = V->geom;
    const int64_t lane = V->lane, F = d.lat_frames, C = m.uC, hs = s.lat_h, ws = s.lat_w;
    const int64_t ppf = (int64_t)d.lat_h * d.lat_w, vs = F * hs * ws, vd = (int64_t)F * ppf;
    const RadOperand gsrc = praw(V->geom_d + RAD_VGEOM_UP_SRC, RAD_I32, 3);
    const RadOperand gdst = praw(V->geom_d + RAD_VGEOM_UP_DST, RAD_I32, 3);
    const RadOperand st = RAD_B(m.b_ust);
    const RadOperand lat = brow_slice(m.b_vstate, lane * m.maxS, vs, kC);
    RAD_ISSUE_N(c, m.op_uden, vs, lat, RAD_W(m.vd_mean), RAD_W(m.vd_std), brows(m.b_uden, vs));
    RAD_ISSUE_N(c, m.op_ucast, vs, brows(m.b_uden, vs), brows(m.b_uin, vs));
    auto x = [&](rad_buf bf, int64_t v) { return view(c, bf, RAD_BF16, v, C); };
    RAD_ISSUE_N(c, m.op_ucin, vs, brows(m.b_uin, vs), RAD_W(m.u_cinw), RAD_W(m.u_cinb), gsrc, RAD_NONE,
                x(m.b_ua, vs));
    RAD_ISSUE_N(c, m.op_unin, vs, x(m.b_ua, vs), RAD_W(m.u_ninw), RAD_W(m.u_ninb), RAD_NONE, st,
                x(m.b_ua, vs));
    auto blocks = [&](const std::vector<URes>& list, int64_t v, const RadOperand& g) {
        for (const URes& U : list) {
            RAD_ISSUE_N(c, U.op_c1, v, x(m.b_ua, v), RAD_W(U.c1w), RAD_W(U.c1b), g, RAD_NONE,
                        x(m.b_ub, v));
            RAD_ISSUE_N(c, U.op_n1, v, x(m.b_ub, v), RAD_W(U.n1w), RAD_W(U.n1b), RAD_NONE, st,
                        x(m.b_ub, v));
            RAD_ISSUE_N(c, U.op_c2, v, x(m.b_ub, v), RAD_W(U.c2w), RAD_W(U.c2b), g, RAD_NONE,
                        x(m.b_uc, v));
            RAD_ISSUE_N(c, U.op_n2, v, x(m.b_uc, v), RAD_W(U.n2w), RAD_W(U.n2b), RAD_NONE, st,
                        x(m.b_uc, v));
            RAD_ISSUE_N(c, U.op_add, v, x(m.b_uc, v), x(m.b_ua, v), x(m.b_ua, v));
            RAD_ISSUE_N(c, U.op_act, v, x(m.b_ua, v), x(m.b_ua, v));
        }
    };
    blocks(m.u_pre, vs, gsrc);
    /* Each frame's Conv2d to 4 C channels, then the pixel shuffle over the volume. */
    const int64_t fr = hs * ws;
    for (int64_t f = 0; f < F; ++f)
        RAD_ISSUE_N(c, m.op_uup, fr, view(c, m.b_ua, RAD_BF16, hs, ws * C, f * fr * C), RAD_W(m.u_upw),
                    RAD_W(m.u_upb), view(c, m.b_ub, RAD_BF16, hs, ws * 4 * C, f * fr * 4 * C));
    RAD_ISSUE_N(c, m.op_ud2s, vs, view(c, m.b_ub, RAD_BF16, vs, 4 * C), gsrc, x(m.b_ua, vd));
    blocks(m.u_post, vd, gdst);
    RAD_ISSUE_N(c, m.op_ucout, vd, x(m.b_ua, vd), RAD_W(m.u_coutw), RAD_W(m.u_coutb), gdst, RAD_NONE,
                brows(m.b_uout, vd));
    RAD_ISSUE_N(c, m.op_unorm, vd, brows(m.b_uout, vd), RAD_W(m.vd_mean), RAD_W(m.vd_std),
                brows(m.b_uout, vd));
    const RadOperand full = brow_slice(m.b_vstate, lane * m.maxS, vd, kC);
    RAD_ISSUE_N(c, m.op_uf32, vd, brows(m.b_uout, vd), full);
    const bool first = V->has_image != 0;
    const RadOperand keep = brow_slice(m.b_vstate, lane * m.maxS, ppf, kC);
    const RadOperand clean = brow_slice(m.b_first, lane * m.maxPpf, ppf, kC);
    if (first) RAD_ISSUE_N(c, m.op_ufirst, ppf, keep, clean);
    RAD_ISSUE_N(c, m.op_urnv, vd, praw(V->sched + 1, RAD_F32, 1), praw(V->seed, RAD_U32, 2), full);
    RAD_ISSUE_N(c, m.op_urna, d.n_audio, praw(V->sched + 4, RAD_F32, 1), praw(V->seed + 2, RAD_U32, 2),
                brow_slice(m.b_astate, lane * m.maxA, d.n_audio, kAudioC));
    if (first) RAD_ISSUE_N(c, m.op_ukeep, ppf, clean, keep);
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_slot(c)];
    if (!batch->gen || !batch->vgen_b) {
        rad_step_fail(c, "ltx2 is a video generator and runs its generator passes only");
        return;
    }
    switch (batch->gen) {
    case RAD_GEN_PASS_VCOND:   step_cond(c, batch, m); return;
    case RAD_GEN_PASS_VSTEP:   step_dit(c, batch, m); return;
    case RAD_GEN_PASS_ADECODE: step_audio(c, batch, m); return;
    case RAD_GEN_PASS_VDECODE: step_video(c, batch, m); return;
    case RAD_GEN_PASS_VUPSCALE:
        if (!m.ups) {
            rad_step_fail(c, "ltx2: an upscale pass, and this container has no latent upsampler");
            return;
        }
        step_upscale(c, batch, m);
        return;
    default:
        rad_step_fail(c, "ltx2: a generator pass of a kind this plugin does not know");
    }
}

/* ================================================================== the probe */
/* The rows of the largest pass: a step of the longest video at the largest size. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->gen_rows = std::max<int64_t>(max_step_rows(), 2 * (int64_t)kTextRows);
    out->gen_video = 1;
    return RAD_OK;
}

}  /* namespace ltx2 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(ltx2)
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(ltx2, "ltx2", "", "0.1.0",
                "LTX-2.5 (t2va, i2va): a Gemma 4 text encoder and connectors, the duration head, a "
                "22B dual-stream audio-video diffusion transformer with four-way guidance, the "
                "causal 3D video VAE (tiled past the default canvas) and the diffusion decoder, the "
                "audio VAE with a BigVGAN vocoder and bandwidth extension, the x2 latent upsampler; "
                "bf16, block-fp8 or rotated int8 linears")

extern "C" int64_t rad_arch_video_positions(int which, const RadVideoGeom* g, float* pos,
                                            int64_t cap) {
    if (!g || g->n_text != ltx2::kTextRows) return RAD_E_INVAL;
    if (which == RAD_VPOS_STEP) return ltx2::positions(g, pos, cap);
    if (which == RAD_VPOS_DECODE) return 1;
    return RAD_E_INVAL;
}
#endif
