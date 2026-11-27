/* minimaxh3 -- MiniMax-H3: a video and its stereo soundtrack from text, between keyframes or from
 * reference media.
 *
 * ============================== THE MODEL ==============================
 *
 * Four networks behind one prompt (diffusers' MiniMaxH3ModularPipeline, workflow t2va):
 *
 *   Text encoder: Qwen3-VL-32B's language model over the prompt's tokens (no template, no special
 *   tokens). The conditioning is the hidden state after its 50th layer, before any norm, so only
 *   the first 50 layers exist here.
 *
 *   Denoiser: a 33B single-stream diffusion transformer over ONE packed sequence
 *   [text | soundtrack | video] -- the conditioning through `context_embedder` and a two-block
 *   token refiner, the soundtrack's latents (40 a second, two channels one after the other, 32
 *   values each) through `audio_proj_in`, the video's latents in 1 x 2 x 2 patches through
 *   `proj_in` -- with full attention and a 3-axis rotary over (time, row, column). 50 blocks of
 *   `h += gate * attn(rms(h) w1 (1 + scale) + shift); h += gate' * mlp(...)`, each block's six
 *   modulation vectors chosen per row by its modality (video, text, audio) at its timestep. It is
 *   guidance-distilled: one forward a step. The video and the soundtrack step down their own
 *   schedules together.
 *
 *   Video decoder: a 36-layer ViT over the latents, decoded in 16 x 16-latent tiles of 7 latent
 *   frames (a 256-pixel tile, 28 pixel frames), blended where tiles and chunks overlap; frames
 *   come out at 24 a second.
 *
 *   Audio decoder: a BigVGAN with anti-aliased SnakeBeta activations, 800 samples a latent at
 *   32 kHz, each stereo channel a sequence of its own.
 *
 * The passes, their batches and the lanes a request's state lives in are the core's video
 * generator (abi/rad_builder.h, rad_declare_video_generator; core/gen/video_driver.h).
 *
 * ============================== KEYFRAMES AND REFERENCES ==============================
 *
 * The workflows i2va, l2va, fl2va and ref2va (ComfyUI's MiniMax-H3 nodes) add rows to the packed
 * sequence: a keyframe's latents, pinned at its frame of the video, or a reference picture's,
 * video's or soundtrack's in its own grid (rad_ref2va_pack.h). The VAEs' encoders make them, one
 * Vencode or Aencode pass a media into the lane before the prompt pass. Their rows take a
 * near-clean timestep, so the modulation table has a cell for every run of segments that share
 * one. A reference picture is also shown to the text encoder, through Qwen3-VL's vision tower.
 * A song guide (ComfyUI's MiniMaxH3AddGuide) is a keyframe of the soundtrack alone: its latents
 * pinned from its frame on, after any first and last frame, beside references or keyframes.
 *
 * ============================== A DISTILLED RELEASE ==============================
 *
 * FastVideo's FastH3 is the same denoiser distilled to eight evaluations on fixed timesteps with
 * video sparse attention, and says so in its inference settings (fastvideo_inference.json): the
 * timesteps on a 1000-step clock, the shifts (10 for the video), the attention and its sparsity.
 * Its attention cuts the packed sequence into tiles -- the prompt's and the soundtrack's rows in
 * 64-row chunks, the video's patches in 4 x 4 x 4 boxes of (time, row, column) -- and a video
 * query tile attends to every chunk and to the video tiles of best mean-query . mean-key score,
 * (1 - sparsity) of them; a chunk's queries attend to everything. A fourth projection beside q, k
 * and v (`to_gate_compress`) gates in an attention over the tiles' means. Everything else is the
 * base model's.
 *
 * ============================== LATENT UPSCALING ==============================
 *
 * A container may also carry LBH-123-AI's latent upscaler (LatentResizer3D, the pipeline
 * directory's latent_upscaler/ folder): a 3D-convolution network that enlarges a video latent's
 * frames by a factor from 1 to 4, told the factor through a small embedding. With it a request can
 * be denoised at a smaller size for its first steps (abi/rad_builder.h, "Latent upscaling"): the
 * upscale pass unfolds the clean small latent into voxels, normalises it with the upscaler's own
 * statistics, runs `conv_in`, 18 blocks at the small size (12 residual blocks conditioned on the
 * factor, a temporal block after every second), a bilinear resize of each frame, 18 blocks at the
 * full size, GroupNorm, SiLU and `conv_out`; then denormalises, folds the voxels back into patch
 * rows and re-noises both latents for the remaining steps. Every rank runs it whole.
 *
 * ============================== ONCE A REQUEST ==============================
 *
 * The conditioning pass runs the text encoder and the refiner, and also every step's modulation:
 * a block's six vectors depend only on the step's timesteps, which the request's schedule fixes,
 * so the 13B parameters of the modulation projections are read once a request rather than once a
 * step. They are a table in the lane, [step][block][modality][this rank's share], gathered across
 * ranks one step at a time. Both the text encoder and the modulation projections are therefore
 * read once a request (RAD_ACCESS_PER_REQUEST), which lets placement keep them in host memory
 * (--vision-tower-placement ram).
 *
 * ============================== WHAT IS QUANTISED ==============================
 *
 * Every linear of the text encoder, of the refiner and of the DiT's blocks, and the modulation
 * projections, are block fp8, or the DiT's int8 at int8 activations (LinearQ / DenseQ read what
 * the container holds). A curve-form checkpoint reads its timestep embeddings off a table it
 * ships, and its modulation projections, eight wide, are bf16. The small
 * linears around them keep the checkpoint's precision: the timestep MLP and the four patch
 * projections in f32, `context_embedder` and `norm_out` in bf16. The DiT's dense attention takes
 * its scores in int8 (keys per row and head, the query inside the op; P.V in bf16), which runs the
 * score half at twice the rate. The video decoder's linears are bf16 with f32 norms and layer
 * scales; the audio decoder is f32 throughout.
 *
 * ============================== TENSOR PARALLEL ==============================
 *
 * The text encoder, the refiner and the DiT's blocks split by heads and feed-forward width, an
 * all-reduce after each output projection. A modulation projection splits by its output rows
 * (6 / world of the six vectors a rank) and the table is gathered a step at a time. The video
 * decoder splits its tiles between ranks with whole weights on each, and gathers the decoded
 * tiles; the audio decoder runs whole on every rank.
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_vit.h>
#include <arch/rad_linear.h>

#include "rad_ref2va_pack.h"
#include "rad_vae_geom.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace minimaxh3 {

using namespace rad::arch;

/* ================================================================== the model's constants */
/* The text encoder's layer the conditioning is read after. */
enum { kTextLayers = 50 };
/* The longest prompt, in tokens. A structured prompt in the model's own style is about 1500; one
 * that shows reference pictures carries their vision tokens too, about 3000 for a few. */
enum { kMaxText = 8192 };
/* Requests holding state at once: one denoising, one conditioned and waiting. */
enum { kMaxLanes = 2 };
/* Model evaluations a request may take (kMaxSteps), and the default sigma grid points (the
 * terminal 0 included, so one more than the evaluations). */
enum { kMaxSteps = 64, kDefSteps = 50 };
/* Time: 24 frames a second; a video of n chunks is 17 n + 5 frames over 5 n + 2 latent frames,
 * and a decode pass reads 7 latent frames from 5 c. 5 to 15 seconds: 7 to 20 chunks. */
enum { kFps = 24, kFpc = 17, kFtail = 5, kLpc = 5, kLtail = 2, kDecLat = 7 };
enum { kMinChunks = 7, kMaxChunks = 20, kDefChunks = 7 };
/* Space: 16 pixels a latent, 2 x 2 latents a patch; sides multiples of 32 from 256, at most the
 * 1344 x 768 canvas's area, 4:1 at most. */
enum { kScale = 16, kPatch = 2, kMult = 32, kMinSide = 256, kMaxAspect = 4 };
enum { kDefW = 1344, kDefH = 768, kMaxArea = kDefW * kDefH };
/* The soundtrack. */
enum { kRate = 32000, kAudioRate = 40, kAudioCh = 2, kAudioC = 32 };
/* The video decoder's tiles, in latents: 16 square, 4 apart at least, 4 register tokens and one
 * zero token after a tile's 7 x 16 x 16. */
enum { kTile = 16, kOverlap = 4, kRegs = 4, kTileTok = kDecLat * kTile * kTile,
       kPeriod = kTileTok + kRegs + 1 };
/* Rows the DiT's heads run a part at a time: the f32 copy of the normed rows they read. */
enum { kHeadRows = 16384 };
/* Rows a DiT all-reduce carries. The library's peer scratch is sized by the largest message an
 * all-reduce declares, two slots of it and outside the VRAM budget, so a step's whole 106K x 5376
 * residual would hold over 4 GiB a card; 16K rows is 176 MB a message, past where the link
 * saturates. */
enum { kArRows = 16384 };

/* The latent channels and the patch's values. */
enum { kC = 24, kPatchVals = kC * kPatch * kPatch };

/* Video sparse attention's tiles, in patches (time, row, column), for a checkpoint distilled with
 * it: 64 rows a tile, the dense segments cut into 64-row chunks of their own. */
enum { kVsaT = 4, kVsaH = 4, kVsaW = 4, kVsaRows = kVsaT * kVsaH * kVsaW };
/* The device selector's tile limit and the video driver's segment boundary capacity. */
enum { kVsaTileLimit = 3072, kVsaPrefixSegments = 64 };

/* ImageNet's normalisation, which the video decoder's pixels are in. */
static const char* kPixMean = "0.485 0.456 0.406";
static const char* kPixStd = "0.229 0.224 0.225";

/* The packed sequence's segments, in order, and each one's modulation table row: text rows take
 * the video's timestep with modality 1, the soundtrack its own with 2, the video its own with 0. */
enum { kSegText = 0, kSegAudio = 1, kSegVideo = 2, kSegs = 3 };
static const int kSegMod[kSegs] = { 1, 2, 0 };

/* The modulation groups a step's sequence may change through (rad_ref2va_pack.h): the plain
 * sequence's three, or with references at most the prompt, keyframes or pictures, each video's
 * soundtrack and video, standalone soundtracks and the two latents in their runs. */
enum { kMaxSegs = 8 };

/* ================================================================== geometry, on the host */
/* diffusers' AutoencoderKLMiniMaxH3._split_tiles in latents: the first latent of each tile across
 * a side of `L` latents. The overlaps start at the minimum and the slack goes to them round-robin,
 * a latent at a time. A side no longer than a tile is one tile (min_side keeps it exactly one). */
static std::vector<int64_t> tile_starts(int64_t L) {
    std::vector<int64_t> st{ 0 };
    if (kTile >= L) return st;
    int64_t n = (L + kTile - 1) / kTile;
    while (kTile * n - kOverlap * (n - 1) - L < 0) ++n;
    std::vector<int64_t> ov((size_t)(n - 1), kOverlap);
    const int64_t rem = kTile * n - kOverlap * (n - 1) - L;
    for (int64_t i = 0; i < rem; ++i) ov[(size_t)(i % (n - 1))] += 1;
    for (int64_t i = 0; i + 1 < n; ++i) st.push_back(st.back() + kTile - ov[(size_t)i]);
    return st;
}

static int64_t tiles_of(int64_t lat_h, int64_t lat_w) {
    return (int64_t)tile_starts(lat_h).size() * (int64_t)tile_starts(lat_w).size();
}

/* Whether a frame of w x h pixels is one this model makes. */
static int64_t video_area(int64_t requested) {
    return requested > 0 ? std::min<int64_t>(requested, kMaxArea) : (int64_t)kMaxArea;
}

static bool size_ok(int64_t w, int64_t h, int64_t area) {
    return w % kMult == 0 && h % kMult == 0 && w >= kMinSide && h >= kMinSide &&
           w * h <= area && std::max(w, h) <= kMaxAspect * std::min(w, h);
}

/* The most tiles a decode pass carries, over every size. */
static int64_t max_tiles(int64_t area) {
    int64_t best = 0;
    for (int64_t w = kMinSide; w <= area / kMinSide; w += kMult)
        for (int64_t h = kMinSide; h * w <= area; h += kMult)
            if (size_ok(w, h, area)) best = std::max(best, tiles_of(h / kScale, w / kScale));
    return best;
}

static int64_t max_lat_frames(int64_t chunks = kMaxChunks) { return kLpc * chunks + kLtail; }
static int64_t max_video_rows(int64_t area, int64_t chunks = kMaxChunks) {
    return max_lat_frames(chunks) * (area / (kMult * kMult));
}
static int64_t max_audio(int64_t chunks = kMaxChunks) {
    const int64_t frames = (int64_t)kFpc * chunks + kFtail;
    return (frames * kAudioRate + kFps / 2) / kFps + 1;
}
static int64_t max_step_rows(int64_t area, int64_t chunks = kMaxChunks) {
    return kMaxText + kAudioCh * max_audio(chunks) + max_video_rows(area, chunks);
}

/* Video sparse attention's video tiles over a grid of F x ph x pw patches, and the ones a video
 * query keeps at `sparsity`: ceil((1 - sparsity) V) in f64, at least one. */
static int64_t vsa_video_tiles(int64_t F, int64_t ph, int64_t pw) {
    return ((F + kVsaT - 1) / kVsaT) * ((ph + kVsaH - 1) / kVsaH) * ((pw + kVsaW - 1) / kVsaW);
}
static int64_t vsa_keep(double sparsity, int64_t V) {
    const double keep = std::ceil((1.0 - sparsity) * (double)V);
    return std::max<int64_t>(1, std::min<int64_t>((int64_t)keep, V));
}
static int64_t vsa_prefix_tiles(int64_t rows) { return rows / kVsaRows + (rows % kVsaRows != 0); }

/* A packed step has at most max_rows. Each conditioning segment can add a padding tile, and
 * the video can add more when a grid edge is not a multiple of four. */
static void vsa_bounds(double sparsity, int64_t area, int64_t max_rows, int64_t* tiles,
                       int64_t* keep) {
    int64_t most_video_tiles = 0, most_video_padding = 0;
    // An upscale pass can start below the output's minimum side or round its aspect ratio.
    for (int64_t w = kMult; w * w <= kMaxAspect * area; w += kMult)
        for (int64_t h = kMult; h * h <= kMaxAspect * area && h * w <= area; h += kMult) {
            const int64_t ph = h / kMult, pw = w / kMult;
            for (int64_t chunks = kMinChunks; chunks <= kMaxChunks; ++chunks) {
                const int64_t frames = max_lat_frames(chunks);
                const int64_t rows = frames * ph * pw;
                if (rows >= max_rows) break;
                const int64_t video_tiles = vsa_video_tiles(frames, ph, pw);
                most_video_tiles = std::max(most_video_tiles, video_tiles);
                most_video_padding = std::max(most_video_padding, video_tiles - rows / kVsaRows);
            }
        }
    *tiles = std::min<int64_t>(kVsaTileLimit, vsa_prefix_tiles(max_rows) + kVsaPrefixSegments +
                                           most_video_padding);
    *keep = vsa_keep(sparsity, std::min(*tiles, most_video_tiles));
}

/* The packed sequence's rotary positions (diffusers' MiniMaxH3PrepareLayoutStep), the pack's
 * with no references (rad_ref2va_pack.h). */
static int64_t positions(const RadVideoGeom* g, float* pos, int64_t cap) {
    if (!g || g->n_text < 1 || g->lat_frames < 1 || g->lat_h < kPatch || g->lat_w < kPatch ||
        g->n_audio < 1)
        return RAD_E_INVAL;
    const Ref2VAGeom pg{ (int)g->lat_frames, (int)g->lat_h, (int)g->lat_w, (int)g->n_audio };
    Ref2VAPack p;
    if (ref2va_build((int)g->n_text, pg, {}, {}, &p, pos != nullptr) != 0) return RAD_E_INVAL;
    if (!pos) return p.rows;
    if (cap < p.rows) return RAD_E_INVAL;
    std::copy(p.pos.begin(), p.pos.end(), pos);
    return p.rows;
}

/* A decode pass's rows: every tile of the frame, each kPeriod rows. */
static int64_t decode_rows(const RadVideoGeom* g) {
    if (!g || g->lat_h < kTile || g->lat_w < kTile) return RAD_E_INVAL;
    return tiles_of(g->lat_h, g->lat_w) * kPeriod;
}

/* ================================================================== the model's parts */
struct TeLayer {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    LinearQ  qkv{}, o{}, gate_up{}, down{};
    QuantAct   q_h{}, q_attn{}, q_h2{};
    GateQuant gq{};
    rad_op     op_n1 = 0, op_qkr = 0, op_attn = 0, op_ar1 = 0, op_add1 = 0;
    rad_op     op_n2 = 0, op_ar2 = 0, op_add2 = 0;
    rad_op     op_qkr_m = 0;  /* `op_qkr` at interleaved M-RoPE, for a prompt with pictures */
};

/* A token refiner block: a plain pre-norm transformer block, no rotary, no modulation. */
struct RefBlock {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    LinearQ  qkv{}, o{}, gate_up{}, down{};
    QuantAct   q_h{}, q_attn{}, q_h2{};
    GateQuant gq{};
    rad_op     op_n1 = 0, op_qkn = 0, op_attn = 0, op_ar1 = 0, op_add1 = 0;
    rad_op     op_n2 = 0, op_ar2 = 0, op_add2 = 0;
};

/* A DiT block. Its two modulated norms carry the gated residual in front of them: the first the
 * previous block's MLP branch (none for block 0), the second this block's attention branch. */
struct DitBlock {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    LinearQ  qkv{}, o{}, gate_up{}, down{};
    QuantAct   q_attn{};
    GateQuant gq{};
    rad_op     op_mq1 = 0, op_mq2 = 0, op_qkr = 0, op_attn = 0, op_ar1 = 0, op_ar2 = 0;
    rad_op     op_sel = 0, op_vsa = 0;
    rad_op     op_kq = 0;          /* the keys to int8 codes for the attention's scores */
    rad_op     op_vq = 0;          /* the values to E4M3 codes for the attention's P.V */
    /* The three modulation projections, one a modality, this rank's share of their rows; a
     * curve-form checkpoint's are bf16 (ada_w). A rotated int8 one adds its bias after the GEMM
     * (op_ada_b). */
    DenseQ   ada[3];
    rad_weight ada_w[3] = {}, ada_b[3] = {};
    rad_op     op_ada[3] = {}, op_ada_b[3] = {};
};

/* A video decoder block. */
struct VitBlock {
    rad_weight w_n1 = 0, w_n2 = 0, w_qkv = 0, b_qkv = 0, w_o = 0, b_o = 0;
    rad_weight w_up = 0, b_up = 0, w_down = 0, b_down = 0, s1 = 0, s2 = 0;
    rad_op     op_n1 = 0, op_qkv = 0, op_qkr = 0, op_attn = 0, op_o = 0, op_s1 = 0, op_r1 = 0;
    rad_op     op_n2 = 0, op_up = 0, op_act = 0, op_down = 0, op_s2 = 0, op_r2 = 0;
};

/* An audio decoder convolution (weight-normalised in the checkpoint, folded at convert). */
struct AConv {
    rad_weight w = 0, b = 0;
    rad_op     op = 0;
};
/* An anti-aliased SnakeBeta. */
struct AAct {
    rad_weight alpha = 0, beta = 0, up = 0, down = 0;
    rad_op     op = 0;
};
/* One dilation of an AMP block: act, dilated conv, act, conv with the residual. */
struct AmpUnit {
    AAct  a1{}, a2{};
    AConv c1{}, c2{};
};
struct AmpBlock {
    AmpUnit u[3];
};
struct AStage {
    AConv    up{};
    rad_op   op_cu = 0;
    AmpBlock amp[3];
    rad_op   op_sum = 0, op_avg = 0;
    int64_t  ch = 0, rate = 0, k = 0;
};

/* A latent upscaler block: a residual block (GroupNorm, SiLU, conv; GroupNorm modulated by the
 * factor's embedding, SiLU, conv; the residual added) or a temporal block (GroupNorm, SiLU, a
 * depthwise convolution over frames, a pointwise one with the residual). */
struct UpsBlock {
    bool       temporal = false;
    int64_t    emb = -1;           /* a residual block's row of the factor's embeddings */
    rad_weight n1w = 0, n1b = 0, c1w = 0, c1b = 0, n2w = 0, n2b = 0, c2w = 0, c2b = 0;
    rad_weight ew = 0, eb = 0;     /* the embedding's projection to (scale, shift) */
    rad_op     op_e = 0, op_n1 = 0, op_c1 = 0, op_n2 = 0, op_c2 = 0;
};

/* Weight groups (movement units) by network: the text encoder's layers 0.., the refiner, the
 * timestep MLP with norm_out's projection, the DiT's model-level weights and its blocks after
 * them, each block's modulation projections, the decoders. A unit takes the strongest access class
 * of its weights, so what is read once a request (the modulation, the timestep MLP) is never
 * grouped with what a step reads: that would keep it off the tower placement. */
enum : int {
    kGrpRefiner = 1000,
    kGrpTemb    = 1010,
    kGrpDit     = 2000,
    kGrpAda     = 2200,
    kGrpVae     = 3000,
    kGrpAudio   = 3100,
    kGrpUps     = 3200,
    kGrpTower   = 3300,
    kGrpVenc    = 3400,
    kGrpAenc    = 3500,
    /* The encoders' weights only a host kernel reads (conv_in, the k=1 convs, the snake alphas):
     * placement keeps a group with a host-only reader in host memory whole, so they are a group
     * of their own. */
    kGrpVencH   = 3450,
    kGrpAencH   = 3550,
};

/* ---- the VAE encoders. A reference frame is at most 256 x 256 (ComfyUI's tile), a video at
 * most 289 frames (17 clips), a soundtrack 500 000 samples (625 latents). The lane keeps a slot a
 * media: a video's 82 latent frames at 256 x 256 are 82 x 8 x 8 patch rows, a soundtrack's two
 * channels 2 x 625 rows. */
enum { kEncMaxSide = 256, kEncMaxFrames = 289, kEncMaxSamples = 500000 };
/* A keyframe is encoded at the video's size, up to the whole canvas, as one frame, which the
 * temporal downsamplers do not shrink: every level holds at least this many frames at 256 x 256,
 * the canvas's area. */
enum { kEncOneFrames = ((int64_t)kDefW * kDefH + (int64_t)kEncMaxSide * kEncMaxSide - 1) /
                       ((int64_t)kEncMaxSide * kEncMaxSide) };
enum { kEncSlotsV = 12, kEncSlotsA = 7 };
enum { kEncSlotRowsV = 82 * (kEncMaxSide / 32) * (kEncMaxSide / 32),
       kEncSlotRowsA = kAudioCh * (kEncMaxSamples / kAudioHop) };

/* The evaluations a request runs at the full size after a latent upscale, by default: of a
 * distilled release's eight, and of the base model's 49. */
enum { kUpsRefineDistilled = 3, kUpsRefineBase = 12 };

struct Model {
    Names nm{""};
    Geom  gt{}, gd{}, gr{};      /* text encoder, DiT (step rows), refiner (prompt rows) */
    int64_t D = 0, S = kMaxSteps;
    int64_t W = 1, Ws = 0;       /* ranks, and a rank's share of a modality's six vectors */
    int64_t maxV = 0, maxA = 0;  /* a lane's video and soundtrack latent rows */
    int64_t max_area = kMaxArea;
    int64_t tiles = 0, tpr = 0;  /* the most tiles a decode carries, and a rank's share of them */
    int64_t enc_tiles = 0;      /* reference images have their own bounds */

    /* ---- the text encoder */
    rad_weight w_tok = 0;
    rad_op  op_embed = 0;
    rad_buf b_tx = 0, b_tkv = 0, b_th2 = 0, b_tgu = 0;
    ActQ  a_th{}, a_tattn{}, a_tffn{};
    std::vector<TeLayer> te;

    /* ---- the text encoder's Qwen3-VL vision tower, for the pictures a prompt shows, when the
     * container carries it (every config names one). Its rows replace the pictures' placeholder
     * tokens and its deepstack taps join the first layers. */
    VisionTower tv{};
    bool   media = false;
    rad_op op_mm = 0, op_ds[VisionTower::MAX_DS] = {};

    /* ---- context_embedder and the refiner, over the prompt's rows */
    rad_weight w_ce = 0, b_ce = 0, w_fn = 0;
    rad_op  op_ce = 0, op_fn = 0;
    rad_buf b_rh = 0, b_rqkv = 0, b_rao = 0, b_rgu = 0;
    ActQ  a_rh{}, a_rattn{}, a_rffn{};
    RefBlock rb[2];

    /* ---- the timestep MLP and the per-step tables. A curve-form checkpoint has a table of
     * embeddings in place of the sinusoid and the MLP: `curve` rows of it (0 without), and
     * `temb_w` is the embedding's width either way. */
    int64_t curve = 0, temb_w = 2688;
    rad_weight w_t1 = 0, b_t1 = 0, w_t2 = 0, b_t2 = 0, w_no = 0, b_no = 0, w_ttable = 0;
    rad_op  op_temb = 0, op_temb_pv = 0, op_temb_pa = 0, op_t1 = 0, op_ts = 0, op_t2 = 0,
            op_ts2 = 0, op_no = 0;
    QuantAct q_temb{};
    rad_buf b_tproj = 0, b_te1 = 0, b_te1s = 0, b_temb = 0;
    ActQ  a_temb{};

    /* ---- the lanes (PERSIST) */
    rad_buf b_text = 0;          /* [lanes * kMaxText, D] bf16: the refined prompt */
    rad_buf b_ada = 0;           /* [lanes * S, 150 * Ws] bf16: the modulation table */
    rad_buf b_adat = 0;          /* [S, Ws] bf16: a rotated int8 projection's rows before its bias */
    rad_buf b_nout = 0;          /* [lanes * S, 3 * 2D] bf16: norm_out's, by segment */
    rad_buf b_vstate = 0;        /* [lanes * maxV, 96] f32 */
    rad_buf b_astate = 0;        /* [lanes * maxA, 32] f32 */
    rad_buf b_vx0 = 0;           /* [lanes * maxV, 96] f32: res_multistep's last estimate */
    rad_buf b_ax0 = 0;           /* [lanes * maxA, 32] f32 */
    rad_buf b_prev = 0;          /* [lanes * 5 * area, 3] bf16: the decoder's frames across chunks */

    /* ---- the DiT */
    rad_op  op_rnd_v = 0, op_rnd_a = 0, op_modg = 0, op_text = 0, op_ain = 0, op_vin = 0;
    rad_weight w_ain = 0, b_ain = 0, w_vin = 0, b_vin = 0;
    rad_buf b_modg = 0;          /* [W * 150, Ws] bf16: this step's table, every rank's share */
    /* Video sparse attention (a checkpoint distilled with it): the dropped share of a video
     * query's video tiles, the most tiles a step has and the most a query keeps, and the per-layer
     * selection and pooled-tile attention, [heads, tiles, k] i32 and [heads, tiles, head_dim]. */
    bool    vsa = false;
    bool    vsa_gated = false;
    double  vsa_sparsity = 0.0;
    int64_t vsa_tiles = 0, vsa_k = 0;
    int64_t vsa_min_rows = 0, vsa_dense_layers = 0;
    rad_buf b_sel = 0, b_coarse = 0;
    rad_buf b_k8 = 0, b_ks = 0;  /* the attention's keys at int8 codes and their scales */
    rad_buf b_v8 = 0, b_vs = 0;  /* its values at E4M3 codes and their column scales */
    /* The trained schedule's base times (a few-step checkpoint's), or none. */
    std::vector<double> rungs;
    rad_buf b_h = 0, b_qkv = 0, b_ao = 0, b_gu = 0, b_hn = 0, b_hf = 0, b_vel = 0,
            b_avel = 0;
    ActQ  a_hm{}, a_attn{}, a_ffn{};
    /* The DiT's linears are int8 (W8A8): their activations are int8 codes as well. */
    bool    dit_i8 = false;
    /* The text encoder's, the refiner's and the modulation projections' linears are rotated int8
     * (IQ8R, or the 128-group form where K is not whole 256s), each read off its first weight:
     * their activations then carry int8 pairs at that weight's scale group in place of the E4M3
     * ones (ActQ::declare_pair). */
    bool    te_i8 = false, ref_i8 = false, ada_i8 = false;
    int64_t te_g8 = RAD_SCALE_GROUP, ref_g8 = RAD_SCALE_GROUP, ada_g8 = RAD_SCALE_GROUP;
    std::vector<DitBlock> dit;
    rad_weight w_no_n = 0, w_vout = 0, b_vout = 0, w_aout = 0, b_aout = 0;
    rad_op  op_nout = 0, op_hcast = 0, op_vout = 0, op_aout = 0, op_ev = 0, op_ea = 0;
    rad_op  op_rmv = 0, op_rma = 0;   /* res_multistep, video and soundtrack */

    /* ---- the audio decoder */
    std::string a_mean, a_std;
    rad_op  op_adenorm = 0, op_acu0 = 0, op_vmix = 0;
    AConv   a_in{}, a_pre{}, a_post{};
    AAct    a_actpost{};
    AStage  ast[7];
    int64_t a_rows[8] = {};
    rad_buf b_aden = 0, b_ain0 = 0, b_acu[8] = {}, b_ah[8] = {}, b_aavg[8] = {}, b_aamp[3][8] = {};
    rad_buf b_at1[8] = {}, b_at2[8] = {}, b_aout0 = 0, b_audio = 0;

    /* ---- the video decoder */
    std::string v_mean, v_std;
    rad_op  op_unfold = 0, op_pqc = 0, op_vpin = 0, op_tiles = 0, op_dpos = 0, op_dcu = 0;
    rad_op  op_vnorm = 0, op_vproj = 0, op_vgather = 0, op_stitch = 0;
    rad_weight w_pqc = 0, b_pqc = 0, w_vpin2 = 0, b_vpin2 = 0, w_reg = 0;
    rad_weight w_vno = 0, b_vno = 0, w_vpo = 0, b_vpo = 0;
    rad_buf b_vox = 0, b_vox2 = 0, b_vpin = 0, b_vx = 0, b_dpos = 0, b_dcu = 0, b_vn = 0;
    rad_buf b_vqkv = 0, b_vattn = 0, b_vo = 0, b_vt = 0, b_vup = 0, b_vact = 0;
    rad_buf b_vpix = 0, b_vpix_all = 0, b_frames = 0;
    std::vector<VitBlock> vit;

    /* ---- the latent upscaler, when the container carries one */
    bool    ups = false;
    double  ups_max = 0.0;       /* the largest factor it was trained on */
    int64_t uC = 0, uE = 0;      /* its width and its embedding's */
    std::string u_inv_mul, u_inv_add, u_std, u_mean;   /* its normalisation, both ways */
    rad_weight u_e0w = 0, u_e0b = 0, u_e2w = 0, u_e2b = 0;
    rad_weight u_cinw = 0, u_cinb = 0, u_now = 0, u_nob = 0, u_coutw = 0, u_coutb = 0;
    rad_op  op_uunf = 0, op_ucast = 0, op_ufac = 0, op_ue0 = 0, op_ues0 = 0, op_ue2 = 0;
    rad_op  op_ues2 = 0, op_ucin = 0, op_ursz = 0, op_unout = 0, op_ucout = 0, op_ufold = 0;
    rad_op  op_urnv = 0, op_urna = 0;
    /* The voxels in and out, three activations of the network's width (each the residual stream
     * or a temporary in turn), GroupNorm's statistics, and the factor's embedding: the factor less
     * one, the MLP's two layers, then every residual block's (scale, shift) row. */
    rad_buf b_uin32 = 0, b_uin = 0, b_ua = 0, b_ub = 0, b_uc = 0, b_uout = 0, b_ust = 0;
    rad_buf b_ufac = 0, b_ue0 = 0, b_ue0s = 0, b_ue1 = 0, b_ue1s = 0, b_uemb = 0;
    std::vector<UpsBlock> u_in, u_out;

    /* ---- the video VAE's encoder, when the container carries it: bf16 voxel rows, causal
     * reflect-padded convolutions, six levels of two residual blocks and four strided
     * downsamplers. A level's stream is `a`, `b` and `c` are a block's temporaries, and `in` the
     * input of a level whose first block changes the width (it has a 1x1x1 shortcut). */
    struct VEncBlock {
        rad_weight n1w = 0, n1b = 0, c1w = 0, c1b = 0, n2w = 0, n2b = 0, c2w = 0, c2b = 0;
        rad_op     op_n1 = 0, op_c1 = 0, op_n2 = 0, op_c2 = 0;
    };
    struct VEncLevel {
        int64_t    cin = 0, cout = 0, rows = 0, pf = 0, frames = 0;   /* pf: rows a frame */
        rad_buf    in = 0, a = 0, b = 0, c = 0;
        rad_weight scw = 0, scb = 0, dnw = 0, dnb = 0;
        rad_op     op_sc = 0, op_dn = 0;
        VEncBlock  blk[2];
    };
    std::vector<VEncLevel> v_lv;
    bool    venc = false;
    rad_weight ve_cinw = 0, ve_cinb = 0, ve_now = 0, ve_nob = 0, ve_coutw = 0, ve_coutb = 0;
    rad_weight ve_qw = 0, ve_qb = 0;
    rad_op  op_ve_pix = 0, op_ve_pcast = 0, op_ve_cin = 0, op_ve_nout = 0, op_ve_cout = 0;
    rad_op  op_ve_q = 0, op_ve_fold = 0, op_eaug = 0;
    std::string ve_pix_mul, ve_pix_add, ve_fold_mul, ve_fold_add;   /* pixel and latent norms */
    rad_buf b_epix32 = 0, b_epix = 0, b_eq48 = 0, b_eq2 = 0, b_egn = 0;
    rad_buf b_evc = 0;               /* [kEncSlotsV * kEncSlotRowsV, kPatchVals] f32 PERSIST */
    /* The tiled encode of a picture too big for one volume (step_venc_tiled): the tiles' geometry
     * rows (read back through raw pointers, so PERSIST), the pixels and the moments tile-major. */
    rad_buf b_tplan = 0, b_vtile = 0, b_eqt = 0;
    rad_op  op_vt_plan = 0, op_vt_crop = 0, op_vt_join = 0;

    /* ---- the audio VAE's encoder, when the container carries it: f32 like its decoder, a
     * channel a sequence, conv_in and five strided blocks of three units [snake, conv k7, snake,
     * conv k1 + residual]. The strided convolutions and the attention block run in bf16, the
     * only dtype those ops take for every operand. Every convolution but mean_proj is
     * weight-normalised. */
    struct AEncUnit {
        rad_weight a1 = 0, c1w = 0, c1b = 0, a2 = 0, c2w = 0, c2b = 0;
        rad_op     op_a1 = 0, op_c1 = 0, op_a2 = 0, op_c2 = 0;
    };
    struct AEncBlk {
        AEncUnit u[3];
        rad_weight al = 0, dnw = 0, dnb = 0;      /* the block's last snake, its downsampler */
        rad_op   op_al = 0, op_dn = 0;
        /* The strided convolution's bf16 input and output. */
        rad_buf  bi = 0, bo = 0;
        rad_op   op_ci = 0, op_co = 0;
    };
    bool    aenc = false;
    AEncBlk ae_blk[5];
    rad_weight ae_cinw = 0, ae_cinb = 0, ae_al6 = 0, ae_finw = 0, ae_finb = 0;
    rad_op  op_ae_cin = 0, op_ae_snk6 = 0, op_ae_fin = 0;
    /* The attention block (causal attention, then a GeGLU MLP at 32 channels), bf16. */
    rad_weight p_ln1w = 0, p_ln1b = 0, p_qkvw = 0, p_qkvb = 0, p_apw = 0, p_apb = 0;
    rad_weight p_ln3w = 0, p_ln3b = 0, p_prw = 0, p_prb = 0, p_ln2w = 0, p_ln2b = 0;
    rad_weight p_mlw = 0, p_mlb = 0, p_guw = 0, p_gub = 0, p_w2w = 0, p_w2b = 0;
    rad_op  p_op_ln3 = 0, p_op_qkv = 0, p_op_attn = 0, p_op_pool = 0, p_op_aproj = 0;
    rad_op  p_op_ln1 = 0, p_op_proj = 0, p_op_hadd = 0, p_op_ln2 = 0, p_op_mln = 0;
    rad_op  p_op_gu = 0, p_op_act = 0, p_op_w2 = 0, p_op_xadd = 0;
    rad_op  p_op_hs = 0, p_op_hsd = 0;   /* the heads' mean: six adds, then an add over eight */
    rad_weight ae_mpw = 0, ae_mpb = 0;            /* mean_proj, f32 */
    rad_op  op_ae_mp = 0, op_ae_norm = 0, op_ae_cast6 = 0, op_ae_castx = 0;
    std::string ae_norm_mul, ae_norm_add;         /* the latents' normalisation */
    rad_buf b_ax[6] = {}, b_et1[5] = {}, b_et2[5] = {}, b_a6t = 0, b_ax6 = 0;
    rad_buf b_apb = 0, b_aln3 = 0, b_aqkv = 0, b_aatt = 0, b_ahsum = 0, b_apool = 0;
    rad_buf b_eh = 0, b_a32a = 0, b_a32b = 0, b_a32c = 0, b_a32d = 0, b_agu = 0, b_agu64 = 0;
    rad_buf b_ax2 = 0, b_amb = 0, b_amb2 = 0;
    rad_buf b_enca = 0;              /* [kEncSlotsA * kEncSlotRowsA, kAudioC] f32 PERSIST */
};

static Model g_model[MAX_SLOTS];

static std::string join_list(const std::vector<double>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        char t[40];
        std::snprintf(t, sizeof t, "%s%.9g", i ? " " : "", v[i]);
        s += t;
    }
    return s;
}

/* A linear with a bias, `gemm_nt_bias` at `dtype` (bf16 or f32), read whole on every rank. */
static rad_op dense_bias(RadBuilder* b, Names& nm, const char* dn, const char* src, uint32_t wdt,
                         const char* dtype, int64_t N, int64_t K, int64_t rows, rad_buf in,
                         rad_buf out, rad_weight* w, rad_weight* bias, int access, int group) {
    *w = wcopy(b, nm.f("%s.weight", dn), nm.ckpt("%s.weight", src), wdt, { N, K }, access, group);
    *bias = wcopy(b, nm.f("%s.bias", dn), nm.ckpt("%s.bias", src), wdt, { N }, access, group);
    if (!*w || !*bias) return 0;
    return rw(b, RAD_OP(b, "gemm_nt_bias",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                   RAD_STR("dtype", dtype)),
                        RAD_WEIGHTS(*w, *bias)),
              { in }, { out });
}

/* (a + b) / div, each step rounded to y's dtype. */
static rad_op add_div_op(RadBuilder* b, int64_t rows, int64_t n, const char* dtype, rad_buf a,
                         rad_buf c, rad_buf y, double div) {
    return rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", dtype),
                                   RAD_F64("div", div)),
                        RAD_NOWEIGHTS),
              { a, c }, { y });
}

static rad_op rmsnorm_op(RadBuilder* b, int64_t rows, int64_t n, double eps, rad_weight w, rad_buf x,
                         rad_buf y) {
    return rw(b, RAD_OP(b, "rmsnorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                   RAD_STR("dtype", "bf16")),
                        RAD_WEIGHTS(w)),
              { x }, { y });
}

/* `overlap`: issued on the second lane beside compute, which picks a narrow launch. */
static rad_op all_reduce(RadBuilder* b, const Geom& g, rad_buf x, int64_t rows, int64_t n,
                         const char* dtype = "bf16", bool overlap = false) {
    if (g.world <= 1) return 0;
    return decl_all_reduce_overlap(b, g, x, n, rows, dtype, overlap ? 1 : 0);
}

/* ================================================================== check_format */
static int check_format(const RadModelMeta* meta) {
    const char* cls = rad_meta_gets(meta, "pipeline._class_name", "");
    if (std::strcmp(cls, "MiniMaxH3ModularPipeline") != 0 &&
        std::strcmp(cls, "MiniMaxH3Pipeline") != 0) {
        fprintf(stderr, "radiance: minimaxh3: the container's pipeline is '%s', not "
                        "MiniMaxH3ModularPipeline\n", cls);
        return RAD_E_UNSUPPORTED;
    }
    const int64_t ps[3] = { rad_meta_geti(meta, "transformer.patch_size.0", 1),
                            rad_meta_geti(meta, "transformer.patch_size.1", 2),
                            rad_meta_geti(meta, "transformer.patch_size.2", 2) };
    const std::vector<double> pl = meta_list(meta, "transformer.patch_size");
    const double pp = (double)kPatch;
    const bool patch_ok = pl.size() == 3 ? pl[0] == 1.0 && pl[1] == pp && pl[2] == pp
                                         : ps[0] == 1 && ps[1] == kPatch && ps[2] == kPatch;
    if (!patch_ok || rad_meta_geti(meta, "transformer.in_channels", 0) != kC ||
        rad_meta_geti(meta, "transformer.audio_in_channels", 0) != kAudioC ||
        rad_meta_geti(meta, "transformer.rope_freq_dim", 0) != 16 ||
        rad_meta_geti(meta, "vae.latent_channels", 0) != kC ||
        rad_meta_geti(meta, "vae.clip_length", 0) != kFpc ||
        rad_meta_geti(meta, "vae.token_drop", 0) != 3 ||
        rad_meta_geti(meta, "audio_vae.latent_channels", 0) != kAudioC ||
        rad_meta_geti(meta, "audio_vae.sampling_rate", 0) != kRate) {
        fprintf(stderr, "radiance: minimaxh3: the transformer, video VAE or audio VAE is not the "
                        "released geometry (patch 1x2x2, 24 + 32 latent channels, 17-frame "
                        "chunks, 32 kHz)\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* ================================================================== the text encoder */
/* Qwen3-VL's language model, its first kTextLayers layers: plain RMSNorm gains, q/k head norms,
 * SwiGLU. Its interleaved M-RoPE turns each frequency by one of three position components; a text
 * token's three are one position, which makes it NeoX's rotary over the whole head, declared as
 * such. Read once a request. */
static int declare_te(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gt;
    Names& nm = m.nm;
    const int64_t T = g.max_tok;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    const int acc = RAD_ACCESS_PER_REQUEST;

    RAD_ARCH_TRY(map_copy(b, nm.f("te.token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    m.w_tok = decl_w(b, nm.f("te.token_embd.weight"), RAD_BF16, { g.n_vocab_all, g.n_embd },
                     RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    /* The linears' format, from layer 0's projection, before the activations' codes are declared:
     * rotated int8 ones read int8 pairs. */
    m.te.resize((size_t)g.n_layer);
    {
        const char* pre = "model.language_model.layers";
        RAD_ARCH_TRY(m.te[0].qkv.probe(b, nm, g, "te.blk.0.attn_qkv",
                                       { nm.ckpt("%s.0.self_attn.q_proj.weight", pre),
                                         nm.ckpt("%s.0.self_attn.k_proj.weight", pre),
                                         nm.ckpt("%s.0.self_attn.v_proj.weight", pre) }));
        RadEncoding e{};
        if (weight_enc(b, nm.f("te.blk.0.attn_qkv.weight"), &e) &&
            (linear_is_w8h(e) || linear_is_wnh(e))) {
            m.te_i8 = true;
            m.te_g8 = linear_w8h_group(e);
        }
    }
    m.b_tx   = decl_b(b, nm.f("te.x"), RAD_BF16, { T, g.n_embd });
    m.a_th.x = decl_b(b, nm.f("te.h"), RAD_BF16, { T, g.n_embd });
    RAD_ARCH_TRY(m.a_th.declare_pair(b, nm, g, "te.h", g.n_embd, 0, m.te_i8, m.te_g8,
                                     "had_quant_act_i8g"));
    m.b_tkv  = decl_b(b, nm.f("te.qkv"), RAD_BF16, { T, q_dim + 2 * kv_dim });
    m.a_tattn.x = decl_b(b, nm.f("te.attn"), RAD_BF16, { T, q_dim });
    RAD_ARCH_TRY(m.a_tattn.declare_pair(b, nm, g, "te.attn", q_dim, 0, m.te_i8, m.te_g8,
                                        "had_quant_act_i8g"));
    m.b_th2  = decl_b(b, nm.f("te.h2"), RAD_BF16, { T, g.n_embd });
    m.b_tgu  = decl_b(b, nm.f("te.gate_up"), RAD_BF16, { T, 2 * g.n_ff });
    m.a_tffn.x = decl_b(b, nm.f("te.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(m.a_tffn.declare_pair(b, nm, g, "te.ffn", g.n_ff, 0, m.te_i8, m.te_g8,
                                       "gated_had_quant_i8g"));

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", g.n_embd),
                                         RAD_INT("n_vocab", g.n_vocab_all), RAD_STR("dtype", "bf16"),
                                         RAD_INT("vocab_offset", 0)),
                              RAD_WEIGHTS(m.w_tok)),
                    {}, { m.b_tx });
    if (m.media) {
        m.op_mm = rw(b, RAD_OP(b, "scatter_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.tv.b_out, m.b_tx }, { m.b_tx });
        for (int t = 0; t < m.tv.n_ds; ++t)
            m.op_ds[t] = rw(b, RAD_OP(b, "scatter_add_rows",
                                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", g.n_embd),
                                                 RAD_STR("dtype", "bf16")),
                                      RAD_NOWEIGHTS),
                            { m.tv.b_ds[t], m.b_tx }, { m.b_tx });
    }

    /* With the tower every layer also has its rotary at interleaved M-RoPE, Qwen3-VL's sections,
     * for a prompt with pictures; text alone is the same rotation at NeoX. */
    const std::vector<double> sec = meta_list(meta, "text_config.rope_scaling.mrope_section");
    char sections[48];
    if (sec.size() == 3)
        std::snprintf(sections, sizeof sections, "%d %d %d", (int)sec[0], (int)sec[1], (int)sec[2]);
    else
        std::snprintf(sections, sizeof sections, "24 20 20");

    m.te.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        TeLayer& L = m.te[(size_t)l];
        const int li = (int)l;
        const char* pre = "model.language_model.layers";
        L.w_n1 = wcopy(b, nm.f("te.blk.%d.attn_norm.weight", li),
                       nm.ckpt("%s.%d.input_layernorm.weight", pre, li), RAD_F32, { g.n_embd }, acc, li);
        L.w_n2 = wcopy(b, nm.f("te.blk.%d.ffn_norm.weight", li),
                       nm.ckpt("%s.%d.post_attention_layernorm.weight", pre, li), RAD_F32,
                       { g.n_embd }, acc, li);
        L.w_qn = wcopy(b, nm.f("te.blk.%d.q_norm.weight", li),
                       nm.ckpt("%s.%d.self_attn.q_norm.weight", pre, li), RAD_BF16, { g.head_dim },
                       acc, li);
        L.w_kn = wcopy(b, nm.f("te.blk.%d.k_norm.weight", li),
                       nm.ckpt("%s.%d.self_attn.k_norm.weight", pre, li), RAD_BF16, { g.head_dim },
                       acc, li);
        if (!L.w_n1 || !L.w_n2 || !L.w_qn || !L.w_kn) return RAD_E_INVAL;
        L.op_n1 = rmsnorm_op(b, T, g.n_embd, g.eps, L.w_n1, m.b_tx, m.a_th.x);
        RAD_ARCH_TRY(L.q_h.declare(b, g, m.a_th, g.n_embd));
        L.qkv.access = acc;
        RAD_ARCH_TRY(L.qkv.declare(b, nm, g, nm.f("te.blk.%d.attn_qkv", li), q_dim + 2 * kv_dim,
                                   g.n_embd, RAD_SHARD_ROW, grp_layer(li),
                                   { nm.ckpt("%s.%d.self_attn.q_proj.weight", pre, li),
                                     nm.ckpt("%s.%d.self_attn.k_proj.weight", pre, li),
                                     nm.ckpt("%s.%d.self_attn.v_proj.weight", pre, li) },
                                   m.a_th, m.b_tkv, 0, { q_dim, kv_dim, kv_dim }));
        L.op_qkr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", g.head_dim),
                                           RAD_INT("n_head", g.n_head),
                                           RAD_INT("n_head_kv", g.n_head_kv),
                                           RAD_F64("theta", g.theta), RAD_F64("eps", g.eps),
                                           RAD_STR("mode", "neox"),
                                           RAD_INT("rotary_dim", g.head_dim)),
                                RAD_WEIGHTS(L.w_qn, L.w_kn)),
                      { m.b_tkv }, { m.b_tkv });
        if (m.media)
            L.op_qkr_m = rw(b, RAD_OP(b, "qk_norm_rope",
                                      RAD_PARAMS(RAD_RANGE("M", 1, T),
                                                 RAD_INT("head_dim", g.head_dim),
                                                 RAD_INT("n_head", g.n_head),
                                                 RAD_INT("n_head_kv", g.n_head_kv),
                                                 RAD_F64("theta", g.theta), RAD_F64("eps", g.eps),
                                                 RAD_STR("mode", "imrope"),
                                                 RAD_INT("rotary_dim", g.head_dim),
                                                 RAD_STR("sections", nm.f("%s", sections))),
                                      RAD_WEIGHTS(L.w_qn, L.w_kn)),
                            { m.b_tkv }, { m.b_tkv });
        L.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", g.head_dim),
                                            RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)g.head_dim)),
                                            RAD_INT("max_seqlen", T)),
                                 RAD_NOWEIGHTS),
                       { m.b_tkv }, { m.a_tattn.x });
        RAD_ARCH_TRY(L.q_attn.declare(b, g, m.a_tattn, q_dim));
        L.o.access = acc;
        RAD_ARCH_TRY(L.o.declare(b, nm, g, nm.f("te.blk.%d.attn_output", li), g.n_embd, q_dim,
                                 RAD_SHARD_COL, grp_layer(li),
                                 { nm.ckpt("%s.%d.self_attn.o_proj.weight", pre, li) },
                                 m.a_tattn, m.b_th2));
        L.op_ar1 = all_reduce(b, g, m.b_th2, T, g.n_embd);
        L.op_add1 = add_op(b, T, g.n_embd, "bf16", m.b_tx, m.b_th2, m.b_tx);
        L.op_n2 = rmsnorm_op(b, T, g.n_embd, g.eps, L.w_n2, m.b_tx, m.a_th.x);
        RAD_ARCH_TRY(L.q_h2.declare(b, g, m.a_th, g.n_embd));
        L.gate_up.access = acc;
        RAD_ARCH_TRY(L.gate_up.declare(b, nm, g, nm.f("te.blk.%d.ffn_gate_up", li), 2 * g.n_ff,
                                       g.n_embd, RAD_SHARD_ROW, grp_layer(li),
                                       { nm.ckpt("%s.%d.mlp.gate_proj.weight", pre, li),
                                         nm.ckpt("%s.%d.mlp.up_proj.weight", pre, li) },
                                       m.a_th, m.b_tgu, 0, { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(L.gq.declare(b, g, m.b_tgu, m.a_tffn, g.n_ff, "silu", 0, true));
        L.down.access = acc;
        RAD_ARCH_TRY(L.down.declare(b, nm, g, nm.f("te.blk.%d.ffn_down", li), g.n_embd, g.n_ff,
                                    RAD_SHARD_COL, grp_layer(li),
                                    { nm.ckpt("%s.%d.mlp.down_proj.weight", pre, li) },
                                    m.a_tffn, m.b_th2));
        L.op_ar2 = all_reduce(b, g, m.b_th2, T, g.n_embd);
        L.op_add2 = add_op(b, T, g.n_embd, "bf16", m.b_tx, m.b_th2, m.b_tx);
        if (!L.op_n1 || !L.op_qkr || !L.op_attn || !L.op_add1 || !L.op_n2 || !L.op_add2)
            return RAD_E_INVAL;
    }
    return m.op_embed ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== adapter key names */
/* LoRA files for H3 name the DiT by ComfyUI's modules (comfy/ldm/minimax): one fused
 * `attn.qkv_proj` (q, k, v), `attn.out_proj`, and `mlp.fc1` with the SwiGLU's gate half first,
 * where the checkpoint keeps the value half first in `ff.net.0.proj`. */
static int declare_lora_names(RadBuilder* b, const Model& m) {
    std::vector<LoraName> n;
    const int64_t Q = m.gd.n_head_all * m.gd.head_dim, KV = m.gd.n_head_kv_all * m.gd.head_dim;
    const int64_t F = m.gd.n_ff_all;
    static const char* const kBlocks[][2] = {
        { "blocks.#.", "transformer_blocks.#." },
        { "token_refiner.blocks.#.", "token_refiner.refiner_blocks.#." },
    };
    for (const auto& bl : kBlocks) {
        const std::string f = bl[0], t = bl[1];
        n.push_back({ f + "attn.qkv_proj", t + "attn.to_q", 0, Q, 0 });
        n.push_back({ f + "attn.qkv_proj", t + "attn.to_k", Q, Q + KV, 0 });
        n.push_back({ f + "attn.qkv_proj", t + "attn.to_v", Q + KV, Q + 2 * KV, 0 });
        n.push_back({ f + "attn.out_proj", t + "attn.to_out.0" });
        n.push_back({ f + "attn.to_gate_compress", t + "attn.to_gate_compress" });
        n.push_back({ f + "mlp.fc1", t + "ff.net.0.proj", 0, F, F });
        n.push_back({ f + "mlp.fc1", t + "ff.net.0.proj", F, 2 * F, 0 });
        n.push_back({ f + "mlp.fc2", t + "ff.net.2" });
    }
    n.push_back({ "blocks.#.adaln_proj.linear", "transformer_blocks.#.adaln_proj.linear" });
    n.push_back({ "video_patch_proj", "proj_in" });
    n.push_back({ "audio_patch_proj", "audio_proj_in" });
    n.push_back({ "condition_proj", "context_embedder" });
    n.push_back({ "time_embedder.proj_in", "time_embedder.linear_1" });
    n.push_back({ "time_embedder.proj_out", "time_embedder.linear_2" });
    n.push_back({ "final_layer.video_out", "proj_out" });
    n.push_back({ "final_layer.audio_out", "audio_proj_out" });
    n.push_back({ "final_layer.adaln_proj.linear", "norm_out.linear" });
    return lora_names(b, "transformer.", kLoraDiffusionSpellings, n);
}

/* ================================================================== the token refiner */
/* `context_embedder` (5120 -> 5376, bf16, replicated), two plain blocks with full attention over
 * the prompt and no rotary, and the final RMSNorm into the lane's prompt rows. The feed-forward is
 * diffusers' SwiGLU, `value * silu(gate)` with the value first in the checkpoint; the gate half is
 * moved first at convert, which is the order `silu_mul` reads. */
static int declare_refiner(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gr;
    Names& nm = m.nm;
    const int64_t T = g.max_tok, D = g.n_embd, qd = g.q_dim(), hd = g.head_dim;
    const int acc = RAD_ACCESS_PER_REQUEST;
    const int gx = kGrpRefiner;
    const double eps = rad_meta_getf(meta, "transformer.norm_eps", 1e-5);
    const double qk_eps = rad_meta_getf(meta, "transformer.qk_norm_eps", 1e-5);
    const int64_t tdim = rad_meta_geti(meta, "transformer.text_dim", 5120);
    if (tdim != m.gt.n_embd) {
        fprintf(stderr, "radiance: minimaxh3: the DiT reads %lld-wide conditioning and the text "
                        "encoder writes %lld\n", (long long)tdim, (long long)m.gt.n_embd);
        return RAD_E_INVAL;
    }

    m.b_rh = decl_b(b, nm.f("ref.h"), RAD_BF16, { T, D });
    m.op_ce = dense_bias(b, nm, "ref.ctx", "transformer.context_embedder", RAD_BF16, "bf16", D,
                         tdim, T, m.b_tx, m.b_rh, &m.w_ce, &m.b_ce, acc, gx);
    /* The refiner's linears' format, from block 0's projection (declare_te's note). */
    {
        const char* pre = nm.ckpt("transformer.token_refiner.refiner_blocks.0");
        RAD_ARCH_TRY(m.rb[0].qkv.probe(b, nm, g, "ref.blk.0.attn_qkv",
                                       { nm.ckpt("%s.attn.to_q.weight", pre),
                                         nm.ckpt("%s.attn.to_k.weight", pre),
                                         nm.ckpt("%s.attn.to_v.weight", pre) }));
        RadEncoding e{};
        if (weight_enc(b, nm.f("ref.blk.0.attn_qkv.weight"), &e) &&
            (linear_is_w8h(e) || linear_is_wnh(e))) {
            m.ref_i8 = true;
            m.ref_g8 = linear_w8h_group(e);
        }
    }
    m.a_rh.x = decl_b(b, nm.f("ref.hn"), RAD_BF16, { T, D });
    RAD_ARCH_TRY(m.a_rh.declare_pair(b, nm, g, "ref.hn", D, 0, m.ref_i8, m.ref_g8,
                                     "had_quant_act_i8g"));
    m.b_rqkv = decl_b(b, nm.f("ref.qkv"), RAD_BF16, { T, 3 * qd });
    m.a_rattn.x = decl_b(b, nm.f("ref.attn"), RAD_BF16, { T, qd });
    RAD_ARCH_TRY(m.a_rattn.declare_pair(b, nm, g, "ref.attn", qd, 0, m.ref_i8, m.ref_g8,
                                        "had_quant_act_i8g"));
    m.b_rao = decl_b(b, nm.f("ref.ao"), RAD_BF16, { T, D });
    m.b_rgu = decl_b(b, nm.f("ref.gate_up"), RAD_BF16, { T, 2 * g.n_ff });
    m.a_rffn.x = decl_b(b, nm.f("ref.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(m.a_rffn.declare_pair(b, nm, g, "ref.ffn", g.n_ff, 0, m.ref_i8, m.ref_g8,
                                       "gated_had_quant_i8g"));
    const int64_t ff_all = g.n_ff_all;

    for (int l = 0; l < 2; ++l) {
        RefBlock& R = m.rb[l];
        const char* pre = nm.ckpt("transformer.token_refiner.refiner_blocks.%d", l);
        const char* dn = nm.f("ref.blk.%d", l);
        R.w_n1 = wcopy(b, nm.f("%s.norm1.weight", dn), nm.ckpt("%s.norm1.weight", pre), RAD_BF16,
                       { D }, acc, gx);
        R.w_n2 = wcopy(b, nm.f("%s.norm2.weight", dn), nm.ckpt("%s.norm2.weight", pre), RAD_BF16,
                       { D }, acc, gx);
        R.w_qn = wcopy(b, nm.f("%s.q_norm.weight", dn), nm.ckpt("%s.attn.norm_q.weight", pre),
                       RAD_BF16, { hd }, acc, gx);
        R.w_kn = wcopy(b, nm.f("%s.k_norm.weight", dn), nm.ckpt("%s.attn.norm_k.weight", pre),
                       RAD_BF16, { hd }, acc, gx);
        if (!R.w_n1 || !R.w_n2 || !R.w_qn || !R.w_kn) return RAD_E_INVAL;
        R.op_n1 = rmsnorm_op(b, T, D, eps, R.w_n1, m.b_rh, m.a_rh.x);
        RAD_ARCH_TRY(R.q_h.declare(b, g, m.a_rh, D));
        R.qkv.access = acc;
        RAD_ARCH_TRY(R.qkv.declare(b, nm, g, nm.f("%s.attn_qkv", dn), 3 * qd, D, RAD_SHARD_ROW,
                                   grp_layer(gx),
                                   { nm.ckpt("%s.attn.to_q.weight", pre),
                                     nm.ckpt("%s.attn.to_k.weight", pre),
                                     nm.ckpt("%s.attn.to_v.weight", pre) },
                                   m.a_rh, m.b_rqkv, 0, { qd, qd, qd }));
        R.op_qkn = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                           RAD_INT("n_head", g.n_head),
                                           RAD_INT("n_head_kv", g.n_head),
                                           RAD_F64("theta", 10000.0), RAD_F64("eps", qk_eps),
                                           RAD_INT("rotary_dim", 0)),
                                RAD_WEIGHTS(R.w_qn, R.w_kn)),
                      { m.b_rqkv }, { m.b_rqkv });
        R.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                            RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                            RAD_INT("max_seqlen", T)),
                                 RAD_NOWEIGHTS),
                       { m.b_rqkv }, { m.a_rattn.x });
        RAD_ARCH_TRY(R.q_attn.declare(b, g, m.a_rattn, qd));
        R.o.access = acc;
        RAD_ARCH_TRY(R.o.declare(b, nm, g, nm.f("%s.attn_output", dn), D, qd, RAD_SHARD_COL,
                                 grp_layer(gx), { nm.ckpt("%s.attn.to_out.0.weight", pre) },
                                 m.a_rattn, m.b_rao));
        R.op_ar1 = all_reduce(b, g, m.b_rao, T, D);
        R.op_add1 = add_op(b, T, D, "bf16", m.b_rh, m.b_rao, m.b_rh);
        R.op_n2 = rmsnorm_op(b, T, D, eps, R.w_n2, m.b_rh, m.a_rh.x);
        RAD_ARCH_TRY(R.q_h2.declare(b, g, m.a_rh, D));
        const char* fw = nm.f("%s.ffn_gate_up.weight", dn);
        const char* fsrc = nm.ckpt("%s.ff.net.0.proj.weight", pre);
        RAD_ARCH_TRY(map_rows(b, fw, { { fsrc, (int)ff_all, ff_all }, { fsrc, 0, ff_all } }));
        R.gate_up.mapped = true;
        R.gate_up.access = acc;
        RAD_ARCH_TRY(R.gate_up.declare(b, nm, g, nm.f("%s.ffn_gate_up", dn), 2 * g.n_ff, D,
                                       RAD_SHARD_ROW, grp_layer(gx), { fsrc }, m.a_rh, m.b_rgu, 0,
                                       { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(R.gq.declare(b, g, m.b_rgu, m.a_rffn, g.n_ff, "silu", 0, true));
        R.down.access = acc;
        RAD_ARCH_TRY(R.down.declare(b, nm, g, nm.f("%s.ffn_down", dn), D, g.n_ff, RAD_SHARD_COL,
                                    grp_layer(gx), { nm.ckpt("%s.ff.net.2.weight", pre) },
                                    m.a_rffn, m.b_rao));
        R.op_ar2 = all_reduce(b, g, m.b_rao, T, D);
        R.op_add2 = add_op(b, T, D, "bf16", m.b_rh, m.b_rao, m.b_rh);
        if (!R.op_n1 || !R.op_qkn || !R.op_attn || !R.op_add1 || !R.op_n2 || !R.op_add2)
            return RAD_E_INVAL;
    }
    m.w_fn = wcopy(b, nm.f("ref.final_norm.weight"),
                   nm.ckpt("transformer.token_refiner.final_norm.weight"), RAD_BF16, { D }, acc, gx);
    if (!m.w_fn) return RAD_E_INVAL;
    m.op_fn = rmsnorm_op(b, T, D, rad_meta_getf(meta, "transformer.final_norm_eps", 1e-5), m.w_fn,
                         m.b_rh, m.b_text);
    return m.op_ce && m.op_fn ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the per-step tables */
/* The timestep embedding of every step's two timesteps -- rows [0, S) the video's, [S, 2S) the
 * soundtrack's -- through the f32 timestep MLP, then silu, rounded to bf16 as the bf16 projections
 * read it, and quantised for the modulation projections. Each block's modulation for segment e is
 * its modality's projection of its timestep's row, into the lane's table at column block
 * (l, e); norm_out's shift and scale are the same of norm_out's own projection, by timestep. */
static int declare_tables(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    const Geom& g = m.gd;
    const int64_t S = m.S, D = m.D, L = g.n_layer;
    const int64_t fd = rad_meta_geti(meta, "transformer.freq_dim", 256);
    const int64_t hd1 = rad_meta_geti(meta, "transformer.time_embed_hidden_dim", 5376);
    const int64_t td = rad_meta_geti(meta, "transformer.time_embed_dim", 2688);
    const int64_t G = rad_meta_geti(meta, "transformer.adaln_curve_grid", 0);
    m.curve = G;
    m.temb_w = td;
    const int acc = RAD_ACCESS_PER_REQUEST;
    const int gx = kGrpTemb;
    if (!G && td % RAD_SCALE_GROUP) {
        fprintf(stderr, "radiance: minimaxh3: the timestep embedding is %lld wide, not whole "
                        "%d-column blocks\n", (long long)td, RAD_SCALE_GROUP);
        return RAD_E_INVAL;
    }
    /* Every step's embeddings in four blocks of S: the video's schedule, the soundtrack's, then
     * each again at max(t, pin) for the keyframes and references (`clamp_min`), which a request
     * without them does not read. */
    const int64_t TT = 4 * S;
    if (G > 0) {
        /* A curve-form checkpoint reads each embedding off its table, between the two rows
         * around the timestep, in bf16: that is the projections' input itself, with no MLP, SiLU
         * or codes. */
        m.w_ttable = wcopy(b, nm.f("tmb.table"), nm.ckpt("transformer.adaln_t_table"),
                           RAD_F32, { G, td }, acc, gx);
        m.b_temb = decl_b(b, nm.f("tmb.emb"), RAD_BF16, { TT, td });
        m.a_temb.x = m.b_temb;
        if (!m.w_ttable || !m.b_temb) return RAD_E_INVAL;
        /* `scale` and `max_period` are required and unused with a curve. */
        m.op_temb = rw(b, RAD_OP(b, "timestep_embed",
                                 RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", td),
                                            RAD_F64("scale", 1.0), RAD_F64("max_period", 10000.0),
                                            RAD_INT("curve", G), RAD_STR("dtype", "bf16")),
                                 RAD_WEIGHTS(m.w_ttable)),
                       {}, { m.b_temb });
        m.op_temb_pv = rw(b, RAD_OP(b, "timestep_embed",
                                    RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", td),
                                               RAD_F64("scale", 1.0),
                                               RAD_F64("max_period", 10000.0),
                                               RAD_INT("curve", G),
                                               RAD_F64("clamp_min", kR2VVisualPin),
                                               RAD_STR("dtype", "bf16")),
                                    RAD_WEIGHTS(m.w_ttable)),
                          {}, { m.b_temb });
        m.op_temb_pa = rw(b, RAD_OP(b, "timestep_embed",
                                    RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", td),
                                               RAD_F64("scale", 1.0),
                                               RAD_F64("max_period", 10000.0),
                                               RAD_INT("curve", G),
                                               RAD_F64("clamp_min", kR2VAudioPin),
                                               RAD_STR("dtype", "bf16")),
                                    RAD_WEIGHTS(m.w_ttable)),
                          {}, { m.b_temb });
        if (!m.op_temb || !m.op_temb_pv || !m.op_temb_pa) return RAD_E_INVAL;
    } else {
        m.b_tproj = decl_b(b, nm.f("tmb.proj"), RAD_F32, { TT, fd });
        m.b_te1 = decl_b(b, nm.f("tmb.h1"), RAD_F32, { TT, hd1 });
        m.b_te1s = decl_b(b, nm.f("tmb.h1s"), RAD_F32, { TT, hd1 });
        m.b_temb = decl_b(b, nm.f("tmb.emb"), RAD_F32, { TT, td });
        m.op_temb = rw(b, RAD_OP(b, "timestep_embed",
                                 RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", fd),
                                            RAD_F64("scale", 1.0), RAD_F64("max_period", 10000.0),
                                            RAD_STR("dtype", "f32")),
                                 RAD_NOWEIGHTS),
                       {}, { m.b_tproj });
        m.op_temb_pv = rw(b, RAD_OP(b, "timestep_embed",
                                    RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", fd),
                                               RAD_F64("scale", 1.0), RAD_F64("max_period", 10000.0),
                                               RAD_F64("clamp_min", kR2VVisualPin),
                                               RAD_STR("dtype", "f32")),
                                    RAD_NOWEIGHTS),
                          {}, { m.b_tproj });
        m.op_temb_pa = rw(b, RAD_OP(b, "timestep_embed",
                                    RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", fd),
                                               RAD_F64("scale", 1.0), RAD_F64("max_period", 10000.0),
                                               RAD_F64("clamp_min", kR2VAudioPin),
                                               RAD_STR("dtype", "f32")),
                                    RAD_NOWEIGHTS),
                          {}, { m.b_tproj });
        m.op_t1 = dense_bias(b, nm, "tmb.l1", "transformer.time_embedder.linear_1", RAD_F32, "f32",
                             hd1, fd, TT, m.b_tproj, m.b_te1, &m.w_t1, &m.b_t1, acc, gx);
        m.op_ts = unary(b, "silu", TT, hd1, "f32", m.b_te1, m.b_te1s);
        m.op_t2 = dense_bias(b, nm, "tmb.l2", "transformer.time_embedder.linear_2", RAD_F32, "f32",
                             td, hd1, TT, m.b_te1s, m.b_temb, &m.w_t2, &m.b_t2, acc, gx);
        /* silu(temb) in f32, rounded once to the bf16 the projections take. Its codes are
         * declared below, once the first projection's encoding says which they are. */
        m.a_temb.x = decl_b(b, nm.f("tmb.act"), RAD_BF16, { TT, td });
        m.op_ts2 = unary(b, "silu", TT, td, "f32", m.b_temb, m.a_temb.x);
        if (!m.op_temb || !m.op_temb_pv || !m.op_temb_pa || !m.op_t1 || !m.op_ts || !m.op_t2
            || !m.op_ts2) return RAD_E_INVAL;
    }

    /* The lane's tables, a cell a modulation group, kMaxSegs of them. */
    m.b_ada = decl_b(b, nm.f("lane.ada"), RAD_BF16, { (int64_t)kMaxLanes * S, L * kMaxSegs * m.Ws },
                     RAD_BUF_PERSIST);
    m.b_nout = decl_b(b, nm.f("lane.nout"), RAD_BF16, { (int64_t)kMaxLanes * S, kMaxSegs * 2 * D },
                      RAD_BUF_PERSIST);
    const int64_t ada_rows = 6 * D;   /* one modality's six vectors */
    m.dit.resize((size_t)L);
    for (int64_t l = 0; l < L; ++l) {
        DitBlock& B = m.dit[(size_t)l];
        const int li = (int)l;
        const int grp = kGrpAda + li;
        const char* src_w = nm.ckpt("transformer.transformer_blocks.%d.adaln_proj.linear.weight", li);
        const char* src_b = nm.ckpt("transformer.transformer_blocks.%d.adaln_proj.linear.bias", li);
        for (int md = 0; md < 3; ++md) {
            const char* base = nm.f("dit.blk.%d.ada%d", li, md);
            const char* wn = nm.f("%s.weight", base);
            const char* bn = nm.f("%s.bias", base);
            RAD_ARCH_TRY(map_rows(b, wn, { { src_w, (int)(md * ada_rows), ada_rows } }));
            RAD_ARCH_TRY(map_rows(b, bn, { { src_b, (int)(md * ada_rows), ada_rows } }));
            B.ada_b[md] = decl_w(b, bn, RAD_BF16, { m.Ws }, acc, RAD_SHARD_ROW, grp_layer(grp));
            if (!B.ada_b[md]) return RAD_E_INVAL;
            if (G > 0) {
                /* A curve's embedding is 8 wide, too narrow for E4M3 blocks: bf16 throughout. */
                B.ada_w[md] = decl_w(b, wn, RAD_BF16, { m.Ws, td }, acc, RAD_SHARD_ROW,
                                     grp_layer(grp));
                if (!B.ada_w[md]) return RAD_E_INVAL;
                B.op_ada[md] = rw(b, RAD_OP(b, "gemm_nt_bias",
                                            RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("N", m.Ws),
                                                       RAD_INT("K", td), RAD_STR("dtype", "bf16")),
                                            RAD_WEIGHTS(B.ada_w[md], B.ada_b[md])),
                                  { m.a_temb.x }, { m.b_ada });
            } else {
                /* The embedding's codes, from the first projection's encoding: E4M3, or for
                 * rotated int8 projections the int8 pair at their scale group. */
                if (l == 0 && md == 0) {
                    RadEncoding e{};
                    if (weight_enc(b, wn, &e) && linear_is_w8h(e)) {
                        m.ada_i8 = true;
                        m.ada_g8 = linear_w8h_group(e);
                    }
                    Geom gs = g;
                    gs.max_tok = TT;
                    RAD_ARCH_TRY(m.a_temb.declare_pair(b, nm, gs, "tmb.act", td, TT, m.ada_i8,
                                                       m.ada_g8, "had_quant_act_i8g"));
                    if (m.ada_i8 && !m.a_temb.q8_fed) {
                        fprintf(stderr, "radiance: minimaxh3: rotated int8 modulation projections "
                                        "need had_quant_act_i8g, which no loaded library serves\n");
                        return RAD_E_UNSUPPORTED;
                    }
                    RAD_ARCH_TRY(m.q_temb.declare(b, gs, m.a_temb, td));
                }
                /* The row slice is mapped above (DenseQ takes that as its argument). */
                B.ada[md].w8h_ok = m.ada_i8;
                RAD_ARCH_TRY(B.ada[md].declare(b, nm, base, m.Ws, td, acc, RAD_SHARD_ROW,
                                               grp_layer(grp), { src_w }, {}, true));
                if (B.ada[md].w8h != m.ada_i8) {
                    fprintf(stderr, "radiance: minimaxh3: '%s' is not encoded as the first "
                                    "modulation projection is\n", wn);
                    return RAD_E_DTYPE;
                }
                if (m.ada_i8) {
                    /* i8a8h has no biased row and writes whole rows: the GEMM goes to its own
                     * rows, and the bias is added on the way into the table's cell. */
                    if (!m.b_adat) m.b_adat = decl_b(b, nm.f("tmb.ada"), RAD_BF16, { S, m.Ws });
                    if (!m.b_adat) return RAD_E_INVAL;
                    B.op_ada[md] = B.ada[md].gemm(b, S, m.Ws, td,
                                                  { m.a_temb.cq(), m.a_temb.cs() }, { m.b_adat });
                    B.op_ada_b[md] = rw(b, RAD_OP(b, "table_add",
                                                  RAD_PARAMS(RAD_RANGE("M", 1, S),
                                                             RAD_INT("n", m.Ws), RAD_INT("L", 1),
                                                             RAD_STR("dtype", "bf16")),
                                                  RAD_WEIGHTS(B.ada_b[md])),
                                        { m.b_adat }, { m.b_ada });
                    if (!B.op_ada_b[md]) return RAD_E_INVAL;
                } else {
                    B.op_ada[md] = rw(b, RAD_OP(b, "gemm_nt_q_bias",
                                                RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("N", m.Ws),
                                                           RAD_INT("K", td),
                                                           RAD_INT("group", RAD_SCALE_GROUP),
                                                           RAD_STR("dtype", "fp8a8")),
                                                RAD_WEIGHTS(B.ada[md].w, B.ada[md].s,
                                                            B.ada_b[md])),
                                      { m.a_temb.q, m.a_temb.s }, { m.b_ada });
                }
            }
            if (!B.op_ada[md]) return RAD_E_INVAL;
        }
    }
    m.op_no = dense_bias(b, nm, "dit.nout.lin", "transformer.norm_out.linear", RAD_BF16, "bf16",
                         2 * D, td, S, m.a_temb.x, m.b_nout, &m.w_no, &m.b_no, acc, gx);
    return m.op_no ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the DiT */
static int declare_dit(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.gd;
    Names& nm = m.nm;
    const int64_t T = g.max_tok, D = g.n_embd, qd = g.q_dim(), hd = g.head_dim, L = g.n_layer;
    const double eps = rad_meta_getf(meta, "transformer.norm_eps", 1e-5);
    const double qk_eps = rad_meta_getf(meta, "transformer.qk_norm_eps", 1e-5);
    const double theta = rad_meta_getf(meta, "transformer.rope_theta", 10000.0);
    const int64_t rf = rad_meta_geti(meta, "transformer.rope_freq_dim", 16);
    const int gx = kGrpDit;
    const int64_t La = (int64_t)kMaxLanes;

    /* ---- the lanes' latents, and the first step's noise */
    m.b_vstate = decl_b(b, nm.f("lane.video"), RAD_F32, { La * m.maxV, kPatchVals }, RAD_BUF_PERSIST);
    m.b_astate = decl_b(b, nm.f("lane.audio"), RAD_F32, { La * m.maxA, kAudioC }, RAD_BUF_PERSIST);
    m.b_vx0 = decl_b(b, nm.f("lane.video.x0"), RAD_F32, { La * m.maxV, kPatchVals }, RAD_BUF_PERSIST);
    m.b_ax0 = decl_b(b, nm.f("lane.audio.x0"), RAD_F32, { La * m.maxA, kAudioC }, RAD_BUF_PERSIST);
    m.op_rnd_v = rw(b, RAD_OP(b, "randn_lanes",
                              RAD_PARAMS(RAD_RANGE("M", 1, m.maxV), RAD_INT("n", kPatchVals),
                                         RAD_INT("stride", m.maxV), RAD_STR("dtype", "f32")),
                              RAD_NOWEIGHTS),
                    { m.b_vstate }, { m.b_vstate });
    m.op_rnd_a = rw(b, RAD_OP(b, "randn_lanes",
                              RAD_PARAMS(RAD_RANGE("M", 1, m.maxA), RAD_INT("n", kAudioC),
                                         RAD_INT("stride", m.maxA), RAD_STR("dtype", "f32")),
                              RAD_NOWEIGHTS),
                    { m.b_astate }, { m.b_astate });

    /* ---- this step's modulation, every rank's share of it */
    const int64_t row = L * kMaxSegs * m.Ws;
    m.b_modg = decl_b(b, nm.f("dit.mod"), RAD_BF16, { (int64_t)g.world * L * kMaxSegs, m.Ws });
    if (g.world > 1)
        m.op_modg = rw(b, RAD_OP(b, "all_gather",
                                 RAD_PARAMS(RAD_INT("world_size", g.world),
                                            RAD_RANGE("numel", 1, row), RAD_STR("dtype", "bf16")),
                                 RAD_NOWEIGHTS),
                       { m.b_ada }, { m.b_modg });
    else
        m.op_modg = cast_op(b, 1, row, "bf16", "bf16", m.b_ada, m.b_modg);

    /* ---- the packed sequence: the lane's prompt rows, the soundtrack's and the video's latents
     * through their f32 projections, rounded to bf16 as the reference packs them */
    m.b_h = decl_b(b, nm.f("dit.h"), RAD_BF16, { T, D });
    /* The linears' format, from block 0's down projection (one source, so declare() does not map
     * it again), before the activations' codes are declared. An int8 DiT takes int8 activations,
     * from the same producers as the E4M3 one. */
    RAD_ARCH_TRY(m.dit[0].down.probe(b, nm, g, nm.f("dit.blk.0.ffn_down"),
                                     { nm.ckpt("transformer.transformer_blocks.0.ff.net.2.weight") }));
    m.dit_i8 = m.dit[0].down.probed_i8;
    m.op_text = rw(b, RAD_OP(b, "lane_gather",
                             RAD_PARAMS(RAD_RANGE("M", 1, (int64_t)kMaxText), RAD_INT("n", D),
                                        RAD_INT("stride", (int64_t)kMaxText),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_text }, { m.b_h });
    m.op_ain = dense_bias(b, nm, "dit.audio_in", "transformer.audio_proj_in", RAD_F32, "f32", D,
                          kAudioC, m.maxA, m.b_astate, m.b_h, &m.w_ain, &m.b_ain,
                          RAD_ACCESS_PER_TOKEN, gx);
    m.op_vin = dense_bias(b, nm, "dit.video_in", "transformer.proj_in", RAD_F32, "f32", D,
                          kPatchVals, m.maxV, m.b_vstate, m.b_h, &m.w_vin, &m.b_vin,
                          RAD_ACCESS_PER_TOKEN, gx);
    if (!m.op_rnd_v || !m.op_rnd_a || !m.op_modg || !m.op_text || !m.op_ain || !m.op_vin)
        return RAD_E_INVAL;

    /* ---- the blocks */
    RAD_ARCH_TRY(m.a_hm.declare_qs(b, nm, g, "dit.hm", D, 0, m.dit_i8, !m.dit_i8));
    /* rms_mod_quant_i8 writes a_hm's int8 codes, as rms_mod_quant_fp8 writes its E4M3 ones. */
    m.a_hm.q8_fed = m.dit_i8;
    /* With video sparse attention a fourth projection beside q, k and v gates the pooled-tile
     * attention into each row's output. */
    const int64_t nparts = m.vsa_gated ? 4 : 3;
    m.b_qkv = decl_b(b, nm.f("dit.qkv"), RAD_BF16, { T, nparts * qd });
    if (m.vsa) {
        m.b_sel = decl_b(b, nm.f("dit.vsa_sel"), RAD_I32, { g.n_head, m.vsa_tiles, m.vsa_k });
        if (m.vsa_gated)
            m.b_coarse = decl_b(b, nm.f("dit.vsa_coarse"), RAD_BF16,
                                { g.n_head, m.vsa_tiles, hd });
        if (!m.b_sel || (m.vsa_gated && !m.b_coarse)) return RAD_E_INVAL;
    }
    m.b_k8 = decl_b(b, nm.f("dit.k8"), RAD_I8, { T, qd });
    m.b_ks = decl_b(b, nm.f("dit.ks"), RAD_F32, { T, qd / hd });
    m.b_v8 = decl_b(b, nm.f("dit.v8"), RAD_F8E4M3, { T, qd });
    m.b_vs = decl_b(b, nm.f("dit.vs"), RAD_F32, { qd });
    m.a_attn.x = decl_b(b, nm.f("dit.attn"), RAD_BF16, { T, qd });
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "dit.attn", qd, 0, m.dit_i8, !m.dit_i8));
    /* q_attn writes the int8 codes the output projection reads. */
    m.a_attn.q8_fed = m.dit_i8;
    m.b_ao = decl_b(b, nm.f("dit.ao"), RAD_BF16, { T, D });
    /* More than one rank: the all-reduces of b_ao's halves run on the second lane (step_dit). */
    if (g.world > 1) RAD_ARCH_TRY(rad_buf_concurrent(b, m.b_ao));
    m.b_gu = decl_b(b, nm.f("dit.gate_up"), RAD_BF16, { T, 2 * g.n_ff });
    m.a_ffn.x = decl_b(b, nm.f("dit.ffn"), RAD_BF16, { T, g.n_ff });
    RAD_ARCH_TRY(m.a_ffn.declare_qs(b, nm, g, "dit.ffn", g.n_ff, 0, m.dit_i8, !m.dit_i8));
    const char* sections = nm.f("%lld %lld %lld", (long long)rf, (long long)rf, (long long)rf);
    const int64_t ff_all = g.n_ff_all;

    /* A block's modulated norm: the previous block's delta added to `b_h`, the norm, and `a_hm`'s
     * codes, int8 or E4M3, in one op. */
    auto mod_q = [&](rad_weight w, bool gated) {
        rad_op op = RAD_OP(b, m.dit_i8 ? "rms_mod_quant_i8" : "rms_mod_quant_fp8",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D), RAD_F64("eps", eps),
                                      RAD_INT("group", RAD_SCALE_GROUP),
                                      RAD_INT("gated", gated ? 1 : 0), RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(w));
        if (!op) return op;
        rad_buf rd[3] = { m.b_modg, m.b_h, m.b_ao };
        rad_op_reads(b, op, rd, gated ? 3 : 2);
        rad_buf wr[3] = { m.a_hm.cq(), m.a_hm.cs(), m.b_h };
        rad_op_writes(b, op, wr, 3);
        return op;
    };

    for (int64_t l = 0; l < L; ++l) {
        DitBlock& B = m.dit[(size_t)l];
        const int li = (int)l;
        const int grp = kGrpDit + 1 + li;
        const char* pre = nm.ckpt("transformer.transformer_blocks.%d", li);
        const char* dn = nm.f("dit.blk.%d", li);
        const int acc = RAD_ACCESS_PER_TOKEN;
        B.w_n1 = wcopy(b, nm.f("%s.norm1.weight", dn), nm.ckpt("%s.norm1.weight", pre), RAD_BF16,
                       { D }, acc, grp);
        B.w_n2 = wcopy(b, nm.f("%s.norm2.weight", dn), nm.ckpt("%s.norm2.weight", pre), RAD_BF16,
                       { D }, acc, grp);
        B.w_qn = wcopy(b, nm.f("%s.q_norm.weight", dn), nm.ckpt("%s.attn.norm_q.weight", pre),
                       RAD_BF16, { hd }, acc, grp);
        B.w_kn = wcopy(b, nm.f("%s.k_norm.weight", dn), nm.ckpt("%s.attn.norm_k.weight", pre),
                       RAD_BF16, { hd }, acc, grp);
        if (!B.w_n1 || !B.w_n2 || !B.w_qn || !B.w_kn) return RAD_E_INVAL;
        B.op_mq1 = mod_q(B.w_n1, l > 0);
        if (m.vsa_gated)
            RAD_ARCH_TRY(B.qkv.declare(b, nm, g, nm.f("%s.attn_qkv", dn), 4 * qd, D, RAD_SHARD_ROW,
                                       grp_layer(grp),
                                       { nm.ckpt("%s.attn.to_q.weight", pre),
                                         nm.ckpt("%s.attn.to_k.weight", pre),
                                         nm.ckpt("%s.attn.to_v.weight", pre),
                                         nm.ckpt("%s.attn.to_gate_compress.weight", pre) },
                                       m.a_hm, m.b_qkv, 0, { qd, qd, qd, qd }));
        else
            RAD_ARCH_TRY(B.qkv.declare(b, nm, g, nm.f("%s.attn_qkv", dn), 3 * qd, D, RAD_SHARD_ROW,
                                       grp_layer(grp),
                                       { nm.ckpt("%s.attn.to_q.weight", pre),
                                         nm.ckpt("%s.attn.to_k.weight", pre),
                                         nm.ckpt("%s.attn.to_v.weight", pre) },
                                       m.a_hm, m.b_qkv, 0, { qd, qd, qd }));
        B.op_qkr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                           RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", g.n_head),
                                           RAD_F64("theta", theta), RAD_F64("eps", qk_eps),
                                           RAD_STR("mode", "axial"), RAD_INT("rotary_dim", 6 * rf),
                                           RAD_STR("sections", sections)),
                                RAD_WEIGHTS(B.w_qn, B.w_kn)),
                      { m.b_qkv }, { m.b_qkv });
        /* The keys to int8 codes and scales per (row, head) for the attention's scores. */
        B.op_kq = decl_quant_act(b, "quant_act_i8g", T, qd, hd, m.b_qkv, m.b_k8, m.b_ks);
        /* The values to E4M3 with a scale per column (one dimension of one head), for P.V on
         * the fp8 matrix instruction. */
        B.op_vq = rw(b, RAD_OP(b, "quant_cols_fp8",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", qd),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_qkv }, { m.b_v8, m.b_vs });
        if (m.vsa && l >= m.vsa_dense_layers) {
            /* Tiles of the packed sequence: each query tile's best video tiles by pooled score
             * and the softmax over every tile's pooled keys, then attention over the kept tiles
             * with that pooled term gated in. */
            B.op_sel = RAD_OP(b, "vsa_select",
                                    RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                               RAD_INT("tile_t", kVsaT), RAD_INT("tile_h", kVsaH),
                                               RAD_INT("tile_w", kVsaW), RAD_INT("patch", kPatch),
                                               RAD_F64("sparsity", m.vsa_sparsity),
                                               RAD_F64("scale", 1.0 / std::sqrt((double)hd))),
                                    RAD_NOWEIGHTS);
            /* The kept tiles' scores in int8, as the dense form below takes them. */
            B.op_vsa = RAD_OP(b, "attn_vsa",
                                     RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                                RAD_INT("tile_t", kVsaT), RAD_INT("tile_h", kVsaH),
                                                RAD_INT("tile_w", kVsaW), RAD_INT("patch", kPatch),
                                                RAD_F64("sparsity", m.vsa_sparsity),
                                                RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                                RAD_STR("qk", "i8"), RAD_STR("pv", "fp8")),
                                     RAD_NOWEIGHTS);
            if (!B.op_sel || !B.op_vsa) return RAD_E_INVAL;
            if (m.vsa_gated) {
                rw(b, B.op_sel, { m.b_qkv }, { m.b_sel, m.b_coarse });
                rw(b, B.op_vsa,
                   { m.b_qkv, m.b_sel, m.b_coarse, m.b_k8, m.b_ks, m.b_v8, m.b_vs },
                   { m.a_attn.x });
            } else {
                rw(b, B.op_sel, { m.b_qkv }, { m.b_sel });
                rw(b, B.op_vsa, { m.b_qkv, m.b_sel, m.b_k8, m.b_ks, m.b_v8, m.b_vs },
                   { m.a_attn.x });
            }
        }
        if (!m.vsa_gated) {
            /* Dense attention with int8 scores: the keys quantised per (row, head) by
             * quant_act_i8g, the query inside the op, the score an exact integer product times
             * both scales, P.V in bf16. The prefix-free form of attn_prefix_self, the op that
             * carries the int8 form. */
            B.op_attn = rw(b, RAD_OP(b, "attn_prefix_self",
                                     RAD_PARAMS(RAD_INT("q_len", T), RAD_INT("head_dim", hd),
                                                RAD_INT("gqa", 1), RAD_INT("block_size", 16),
                                                RAD_INT("causal", 0), RAD_STR("q_dtype", "bf16"),
                                                RAD_STR("kv_dtype", "bf16"),
                                                RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                                RAD_INT("max_ctx", 0), RAD_INT("n_head", g.n_head),
                                                RAD_INT("max_seqs", 1), RAD_STR("qk", "i8"),
                                                RAD_STR("pv", "fp8")),
                                     RAD_NOWEIGHTS),
                           { m.b_qkv, m.b_k8, m.b_ks, m.b_v8, m.b_vs }, { m.a_attn.x });
        }
        if (!B.op_kq || !B.op_vq || (!m.vsa_gated && !B.op_attn)) return RAD_E_INVAL;
        RAD_ARCH_TRY(B.q_attn.declare(b, g, m.a_attn, qd));
        RAD_ARCH_TRY(B.o.declare(b, nm, g, nm.f("%s.attn_output", dn), D, qd, RAD_SHARD_COL,
                                 grp_layer(grp), { nm.ckpt("%s.attn.to_out.0.weight", pre) },
                                 m.a_attn, m.b_ao));
        B.op_ar1 = all_reduce(b, g, m.b_ao, std::min<int64_t>(T, kArRows), D, "bf16", true);
        B.op_mq2 = mod_q(B.w_n2, true);
        const char* fw = nm.f("%s.ffn_gate_up.weight", dn);
        const char* fsrc = nm.ckpt("%s.ff.net.0.proj.weight", pre);
        /* The gate half first, as in the refiner. diffusers' SwiGLU (`ff.net.0.proj`) takes
         * `value, gate = chunk(2)` and computes value * silu(gate), so the checkpoint stores the
         * gate second, while silu_mul and gated_quant_fp8 gate with the first half. Left in
         * checkpoint order, every frame renders as scrambled colour tiles. The int8 path's group
         * scales are quantised from the mapped rows, so they follow the swap. */
        RAD_ARCH_TRY(map_rows(b, fw, { { fsrc, (int)ff_all, ff_all }, { fsrc, 0, ff_all } }));
        B.gate_up.mapped = true;
        RAD_ARCH_TRY(B.gate_up.declare(b, nm, g, nm.f("%s.ffn_gate_up", dn), 2 * g.n_ff, D,
                                       RAD_SHARD_ROW, grp_layer(grp), { fsrc }, m.a_hm, m.b_gu, 0,
                                       { g.n_ff, g.n_ff }));
        B.gq.silu_only = m.dit_i8;
        RAD_ARCH_TRY(B.gq.declare(b, g, m.b_gu, m.a_ffn, g.n_ff, "silu", 0, false));
        RAD_ARCH_TRY(B.down.declare(b, nm, g, nm.f("%s.ffn_down", dn), D, g.n_ff, RAD_SHARD_COL,
                                    grp_layer(grp), { nm.ckpt("%s.ff.net.2.weight", pre) },
                                    m.a_ffn, m.b_ao));
        B.op_ar2 = all_reduce(b, g, m.b_ao, std::min<int64_t>(T, kArRows), D, "bf16", true);
        if (!B.op_mq1 || !B.op_mq2 || !B.op_qkr) return RAD_E_INVAL;
    }

    /* ---- the output: norm_out's modulated norm (the last block's branch gated in), the heads in
     * f32 over parts of the rows, and the two Euler steps into the lane */
    m.b_hn = decl_b(b, nm.f("dit.hn"), RAD_BF16, { T, D });
    m.w_no_n = wcopy(b, nm.f("dit.nout.norm.weight"), nm.ckpt("transformer.norm_out.norm.weight"),
                     RAD_BF16, { D }, RAD_ACCESS_PER_TOKEN, gx);
    if (!m.w_no_n) return RAD_E_INVAL;
    {
        rad_op op = RAD_OP(b, "rms_mod",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                      RAD_F64("eps", rad_meta_getf(meta, "transformer.final_norm_eps",
                                                                   1e-5)),
                                      RAD_INT("gated", 1), RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(m.w_no_n));
        if (!op) return RAD_E_INVAL;
        rad_buf rd[4] = { m.b_ao, m.b_modg, m.b_nout, m.b_h };
        rad_buf wr[2] = { m.b_hn, m.b_h };
        rad_op_reads(b, op, rd, 4);
        rad_op_writes(b, op, wr, 2);
        m.op_nout = op;
    }
    m.b_hf = decl_b(b, nm.f("dit.hf"), RAD_F32, { (int64_t)kHeadRows, D });
    m.op_hcast = cast_op(b, (int64_t)kHeadRows, D, "bf16", "f32", m.b_hn, m.b_hf);
    m.b_vel = decl_b(b, nm.f("dit.vel"), RAD_F32, { m.maxV, kPatchVals });
    m.b_avel = decl_b(b, nm.f("dit.avel"), RAD_F32, { m.maxA, kAudioC });
    m.op_vout = dense_bias(b, nm, "dit.video_out", "transformer.proj_out", RAD_F32, "f32",
                           kPatchVals, D, kHeadRows, m.b_hf, m.b_vel, &m.w_vout, &m.b_vout,
                           RAD_ACCESS_PER_TOKEN, gx);
    m.op_aout = dense_bias(b, nm, "dit.audio_out", "transformer.audio_proj_out", RAD_F32, "f32",
                           kAudioC, D, kHeadRows, m.b_hf, m.b_avel, &m.w_aout, &m.b_aout,
                           RAD_ACCESS_PER_TOKEN, gx);
    /* The cast and the two heads are issued in a loop over parts of the rows, the cast again after
     * a head has run, so the normed rows outlive both heads: declared as read by them, their
     * storage is never the velocities'. */
    if (m.op_vout && m.op_aout) {
        RAD_ARCH_TRY(rad_op_reads(b, m.op_vout, &m.b_hn, 1));
        RAD_ARCH_TRY(rad_op_reads(b, m.op_aout, &m.b_hn, 1));
    }
    m.op_ev = rw(b, RAD_OP(b, "flow_x0_step",
                           RAD_PARAMS(RAD_RANGE("M", 1, m.maxV), RAD_INT("n", kPatchVals),
                                      RAD_STR("dtype", "f32")),
                           RAD_NOWEIGHTS),
                 { m.b_vel, m.b_vstate }, { m.b_vstate });
    m.op_ea = rw(b, RAD_OP(b, "flow_x0_step",
                           RAD_PARAMS(RAD_RANGE("M", 1, m.maxA), RAD_INT("n", kAudioC),
                                      RAD_STR("dtype", "f32")),
                           RAD_NOWEIGHTS),
                 { m.b_avel, m.b_astate }, { m.b_astate });
    /* res_multistep reads and rewrites the last estimate as well as the latent. */
    m.op_rmv = rw(b, RAD_OP(b, "flow_res_multistep",
                            RAD_PARAMS(RAD_RANGE("M", 1, m.maxV), RAD_INT("n", kPatchVals),
                                       RAD_STR("dtype", "f32")),
                            RAD_NOWEIGHTS),
                  { m.b_vel, m.b_vx0, m.b_vstate }, { m.b_vx0, m.b_vstate });
    m.op_rma = rw(b, RAD_OP(b, "flow_res_multistep",
                            RAD_PARAMS(RAD_RANGE("M", 1, m.maxA), RAD_INT("n", kAudioC),
                                       RAD_STR("dtype", "f32")),
                            RAD_NOWEIGHTS),
                  { m.b_avel, m.b_ax0, m.b_astate }, { m.b_ax0, m.b_astate });
    return m.op_hcast && m.op_vout && m.op_aout && m.op_ev && m.op_ea && m.op_rmv && m.op_rma
               ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the audio decoder */
/* A weight-normalised convolution's folded weight and its bias (none for conv_post). */
static int aconv(RadBuilder* b, Names& nm, AConv& c, const char* dn, const char* src, int64_t cin,
                 int64_t cout, int64_t k, int64_t dil, bool wnorm, bool bias, int64_t rows, rad_buf x,
                 rad_buf cu, rad_buf res, rad_buf y, const char* act) {
    const char* wn = nm.f("%s.weight", dn);
    if (wnorm) RAD_ARCH_TRY(map_wnorm(b, wn, nm.ckpt("%s.weight_g", src), nm.ckpt("%s.weight_v", src)));
    else RAD_ARCH_TRY(map_copy(b, wn, nm.ckpt("%s.weight", src)));
    c.w = decl_w(b, wn, RAD_F32, { cout, cin, k }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                 grp_layer(kGrpAudio));
    if (bias)
        c.b = wcopy(b, nm.f("%s.bias", dn), nm.ckpt("%s.bias", src), RAD_F32, { cout },
                    RAD_ACCESS_PER_TOKEN, kGrpAudio);
    if (!c.w || (bias && !c.b)) return RAD_E_INVAL;
    rad_op op = bias ? RAD_OP(b, "conv1d",
                              RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("cin", cin),
                                         RAD_INT("cout", cout), RAD_INT("k", k), RAD_INT("dil", dil),
                                         RAD_STR("act", act), RAD_STR("dtype", "f32")),
                              RAD_WEIGHTS(c.w, c.b))
                     : RAD_OP(b, "conv1d",
                              RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("cin", cin),
                                         RAD_INT("cout", cout), RAD_INT("k", k), RAD_INT("dil", dil),
                                         RAD_STR("act", act), RAD_STR("dtype", "f32")),
                              RAD_WEIGHTS(c.w));
    c.op = res ? rw(b, op, { x, cu, res }, { y }) : rw(b, op, { x, cu }, { y });
    return c.op ? RAD_OK : RAD_E_INVAL;
}

/* An anti-aliased SnakeBeta: its log-space alpha and beta, and the two Kaiser filters. */
static int aact(RadBuilder* b, Names& nm, AAct& a, const char* dn, const char* src, int64_t ch,
                int64_t rows, rad_buf x, rad_buf cu, rad_buf y) {
    a.alpha = wcopy(b, nm.f("%s.alpha", dn), nm.ckpt("%s.act.alpha", src), RAD_F32, { ch },
                    RAD_ACCESS_PER_TOKEN, kGrpAudio);
    a.beta = wcopy(b, nm.f("%s.beta", dn), nm.ckpt("%s.act.beta", src), RAD_F32, { ch },
                   RAD_ACCESS_PER_TOKEN, kGrpAudio);
    a.up = wcopy(b, nm.f("%s.up", dn), nm.ckpt("%s.upsample.filter", src), RAD_F32, { 1, 1, 12 },
                 RAD_ACCESS_PER_TOKEN, kGrpAudio);
    a.down = wcopy(b, nm.f("%s.down", dn), nm.ckpt("%s.downsample.lowpass.filter", src), RAD_F32,
                   { 1, 1, 12 }, RAD_ACCESS_PER_TOKEN, kGrpAudio);
    if (!a.alpha || !a.beta || !a.up || !a.down) return RAD_E_INVAL;
    a.op = rw(b, RAD_OP(b, "snake_aa",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", ch), RAD_INT("ratio", 2),
                                   RAD_INT("k", 12), RAD_INT("log", 1), RAD_STR("dtype", "f32")),
                        RAD_WEIGHTS(a.alpha, a.beta, a.up, a.down)),
              { x, cu }, { y });
    return a.op ? RAD_OK : RAD_E_INVAL;
}

static rad_op cu_op(RadBuilder* b, int64_t mult, rad_buf y) {
    return rw(b, RAD_OP(b, "cu_repeat",
                        RAD_PARAMS(RAD_RANGE("M", 1, 3), RAD_INT("mult", mult)),
                        RAD_NOWEIGHTS),
              {}, { y });
}

/* dec_in_proj and BigVGAN over the soundtrack's two channels, each a sequence (`cu`); every stage
 * an upsampling transposed convolution and the mean of three AMP blocks (kernel 3, 7 and 11), each
 * three dilations of act, dilated conv, act, conv and the residual. */
static int declare_audio(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    const std::vector<double> rates = meta_list(meta, "audio_vae.decoder_rates");
    const std::vector<double> ks = meta_list(meta, "audio_vae.decoder_kernel_sizes");
    const std::vector<double> rk = meta_list(meta, "audio_vae.resblock_kernel_sizes");
    const int64_t dd = rad_meta_geti(meta, "audio_vae.decoder_dim", 1024);
    const int64_t ld = rad_meta_geti(meta, "audio_vae.latent_dim", 2048);
    const std::vector<double> mean = meta_list(meta, "audio_vae.latents_mean");
    const std::vector<double> sd = meta_list(meta, "audio_vae.latents_std");
    int64_t hop = 1;
    for (double r : rates) hop *= (int64_t)r;
    if (rates.size() != 7 || ks.size() != 7 || rk.size() != 3 || hop != kRate / kAudioRate ||
        (int64_t)mean.size() != kAudioC || (int64_t)sd.size() != kAudioC) {
        fprintf(stderr, "radiance: minimaxh3: the audio VAE is not the released BigVGAN (7 stages "
                        "to 800 samples a latent, three AMP kernels)\n");
        return RAD_E_UNSUPPORTED;
    }
    static const int64_t kDil[3] = { 1, 3, 5 };
    m.a_mean = join_list(mean);
    m.a_std = join_list(sd);
    m.a_rows[0] = m.maxA;
    for (int s = 0; s < 7; ++s) m.a_rows[s + 1] = m.a_rows[s] * (int64_t)rates[(size_t)s];

    m.b_aden = decl_b(b, nm.f("aud.den"), RAD_F32, { m.maxA, kAudioC });
    m.op_adenorm = rw(b, RAD_OP(b, "chan_affine",
                                RAD_PARAMS(RAD_RANGE("M", 1, m.maxA), RAD_INT("n", kAudioC),
                                           RAD_STR("mul", m.a_std.c_str()),
                                           RAD_STR("add", m.a_mean.c_str()), RAD_STR("mode", "fma"),
                                           RAD_INT("round", 0), RAD_STR("dtype", "f32")),
                                RAD_NOWEIGHTS),
                      { m.b_astate }, { m.b_aden });
    for (int s = 0; s < 8; ++s) m.b_acu[s] = decl_b(b, nm.f("aud.cu%d", s), RAD_I32, { 3 });
    m.op_acu0 = cu_op(b, 1, m.b_acu[0]);
    m.b_ain0 = decl_b(b, nm.f("aud.in"), RAD_F32, { m.maxA, ld });
    RAD_ARCH_TRY(aconv(b, nm, m.a_in, "aud.in_proj", "audio_vae.dec_in_proj", kAudioC, ld, 1, 1,
                       false, true, m.maxA, m.b_aden, m.b_acu[0], 0, m.b_ain0, "none"));
    m.b_aavg[0] = decl_b(b, nm.f("aud.h0"), RAD_F32, { m.maxA, dd });
    RAD_ARCH_TRY(aconv(b, nm, m.a_pre, "aud.conv_pre", "audio_vae.decoder.conv_pre", ld, dd, 7, 1,
                       true, true, m.maxA, m.b_ain0, m.b_acu[0], 0, m.b_aavg[0], "none"));
    int64_t mult = 1;
    for (int s = 0; s < 7; ++s) {
        AStage& St = m.ast[s];
        const int64_t ci = dd >> s, co = dd >> (s + 1), R = m.a_rows[s + 1];
        St.ch = co;
        St.rate = (int64_t)rates[(size_t)s];
        St.k = (int64_t)ks[(size_t)s];
        mult *= St.rate;
        St.op_cu = cu_op(b, mult, m.b_acu[s + 1]);
        m.b_ah[s + 1] = decl_b(b, nm.f("aud.s%d.in", s), RAD_F32, { R, co });
        const char* wn = nm.f("aud.s%d.up.weight", s);
        const char* src = nm.ckpt("audio_vae.decoder.ups.%d.0", s);
        RAD_ARCH_TRY(map_wnorm(b, wn, nm.ckpt("%s.weight_g", src), nm.ckpt("%s.weight_v", src)));
        St.up.w = decl_w(b, wn, RAD_F32, { ci, co, St.k }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                         grp_layer(kGrpAudio));
        St.up.b = wcopy(b, nm.f("aud.s%d.up.bias", s), nm.ckpt("%s.bias", src), RAD_F32, { co },
                        RAD_ACCESS_PER_TOKEN, kGrpAudio);
        if (!St.up.w || !St.up.b) return RAD_E_INVAL;
        St.up.op = rw(b, RAD_OP(b, "conv_t1d",
                                RAD_PARAMS(RAD_RANGE("M", 1, m.a_rows[s]), RAD_INT("cin", ci),
                                           RAD_INT("cout", co), RAD_INT("stride", St.rate),
                                           RAD_INT("k", St.k), RAD_INT("pad", (St.k - St.rate) / 2),
                                           RAD_STR("dtype", "f32")),
                                RAD_WEIGHTS(St.up.w, St.up.b)),
                      { m.b_aavg[s], m.b_acu[s] }, { m.b_ah[s + 1] });
        if (!St.up.op || !St.op_cu) return RAD_E_INVAL;
        m.b_at1[s + 1] = decl_b(b, nm.f("aud.s%d.t1", s), RAD_F32, { R, co });
        m.b_at2[s + 1] = decl_b(b, nm.f("aud.s%d.t2", s), RAD_F32, { R, co });
        for (int j = 0; j < 3; ++j) m.b_aamp[j][s + 1] = decl_b(b, nm.f("aud.s%d.amp%d", s, j), RAD_F32,
                                                               { R, co });
        for (int j = 0; j < 3; ++j) {
            AmpBlock& A = St.amp[j];
            const int64_t k = (int64_t)rk[(size_t)j];
            const int ri = s * 3 + j;
            for (int d = 0; d < 3; ++d) {
                AmpUnit& U = A.u[d];
                const char* bn = nm.f("aud.s%d.amp%d.d%d", s, j, d);
                const char* bs = nm.ckpt("audio_vae.decoder.resblocks.%d", ri);
                /* The unit reads the stage's input on its first dilation, its own running sum
                 * after. */
                const rad_buf x = d == 0 ? m.b_ah[s + 1] : m.b_aamp[j][s + 1];
                RAD_ARCH_TRY(aact(b, nm, U.a1, nm.f("%s.act1", bn),
                                  nm.ckpt("%s.activations.%d", bs, 2 * d), co, R, x, m.b_acu[s + 1],
                                  m.b_at1[s + 1]));
                RAD_ARCH_TRY(aconv(b, nm, U.c1, nm.f("%s.conv1", bn),
                                   nm.ckpt("%s.convs1.%d", bs, d), co, co, k, kDil[d], true, true, R,
                                   m.b_at1[s + 1], m.b_acu[s + 1], 0, m.b_at2[s + 1], "none"));
                RAD_ARCH_TRY(aact(b, nm, U.a2, nm.f("%s.act2", bn),
                                  nm.ckpt("%s.activations.%d", bs, 2 * d + 1), co, R, m.b_at2[s + 1],
                                  m.b_acu[s + 1], m.b_at1[s + 1]));
                RAD_ARCH_TRY(aconv(b, nm, U.c2, nm.f("%s.conv2", bn),
                                   nm.ckpt("%s.convs2.%d", bs, d), co, co, k, 1, true, true, R,
                                   m.b_at1[s + 1], m.b_acu[s + 1], x, m.b_aamp[j][s + 1], "none"));
            }
        }
        m.b_aavg[s + 1] = decl_b(b, nm.f("aud.s%d.out", s), RAD_F32, { R, co });
        St.op_sum = add_op(b, R, co, "f32", m.b_aamp[0][s + 1], m.b_aamp[1][s + 1], m.b_aavg[s + 1]);
        St.op_avg = add_div_op(b, R, co, "f32", m.b_aavg[s + 1], m.b_aamp[2][s + 1], m.b_aavg[s + 1],
                               3.0);
        if (!St.op_sum || !St.op_avg) return RAD_E_INVAL;
    }
    const int64_t R7 = m.a_rows[7], c7 = dd >> 7;
    RAD_ARCH_TRY(aact(b, nm, m.a_actpost, "aud.act_post", nm.ckpt("audio_vae.decoder.activation_post"),
                      c7, R7, m.b_aavg[7], m.b_acu[7], m.b_at1[7]));
    m.b_aout0 = decl_b(b, nm.f("aud.mono"), RAD_F32, { R7, 1 });
    RAD_ARCH_TRY(aconv(b, nm, m.a_post, "aud.conv_post", "audio_vae.decoder.conv_post", c7, 1, 7, 1,
                       true, false, R7, m.b_at1[7], m.b_acu[7], 0, m.b_aout0, "clamp"));
    m.b_audio = decl_b(b, nm.f("audio_out"), RAD_F32, { R7 / kAudioCh + 1, kAudioCh },
                       RAD_BUF_PERSIST);
    m.op_vmix = rw(b, RAD_OP(b, "interleave2",
                             RAD_PARAMS(RAD_RANGE("M", 1, R7 / kAudioCh), RAD_STR("dtype", "f32")),
                             RAD_NOWEIGHTS),
                   { m.b_aout0 }, { m.b_audio });
    return m.op_adenorm && m.op_acu0 && m.op_vmix ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the video decoder */
/* A decode pass is one chunk: latent frames [5 c, 5 c + 7) unpatchified out of the lane and
 * denormalised, post_quant_conv and proj_in (f32) over every latent, then this rank's share of the
 * tiles (7 x 16 x 16 latents, its registers and a zero token) through the ViT, its pixels gathered
 * across ranks, and the tiles stitched into the chunk's frames. */
static int declare_video(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    const int64_t dim = rad_meta_geti(meta, "vae.decoder_num_attention_heads", 32) *
                        rad_meta_geti(meta, "vae.decoder_attention_head_dim", 64);
    const int64_t hd = rad_meta_geti(meta, "vae.decoder_attention_head_dim", 64);
    const int64_t nl = rad_meta_geti(meta, "vae.decoder_num_layers", 36);
    const int64_t ff = dim * rad_meta_geti(meta, "vae.decoder_ffn_mult", 4);
    const double eps = rad_meta_getf(meta, "vae.decoder_norm_eps", 1e-5);
    const double theta = rad_meta_getf(meta, "vae.decoder_rope_theta", 100.0);
    const double ratio = rad_meta_getf(meta, "vae.decoder_rope_dim_ratio", 0.75);
    const std::vector<double> mean = meta_list(meta, "vae.latents_mean");
    const std::vector<double> sd = meta_list(meta, "vae.latents_std");
    const int64_t rot = (int64_t)(hd * ratio);
    const int64_t px = 3 * 4 * kScale * kScale;    /* a token's pixels: 3 colours x 4 frames x 16^2 */
    if ((int64_t)mean.size() != kC || (int64_t)sd.size() != kC || rot % 6 ||
        rad_meta_geti(meta, "vae.decoder_num_register_tokens", 0) != kRegs) {
        fprintf(stderr, "radiance: minimaxh3: the video VAE decoder is not the released ViT\n");
        return RAD_E_UNSUPPORTED;
    }
    m.v_mean = join_list(mean);
    m.v_std = join_list(sd);
    const int64_t maxhw = m.max_area / (kScale * kScale);
    const int64_t vox = (int64_t)kDecLat * maxhw;
    const int64_t R = m.tpr * kPeriod;               /* this rank's tile rows */
    const int64_t RA = (int64_t)m.W * R;             /* every rank's, gathered */
    const int gv = kGrpVae;
    const int acc = RAD_ACCESS_PER_TOKEN;

    m.b_vox = decl_b(b, nm.f("vae.vox"), RAD_F32, { vox, kC });
    m.op_unfold = rw(b, RAD_OP(b, "patch_unfold",
                               RAD_PARAMS(RAD_RANGE("M", 1, vox), RAD_INT("C", kC), RAD_INT("pt", 1),
                                          RAD_INT("ph", kPatch), RAD_INT("pw", kPatch),
                                          RAD_STR("mul", m.v_std.c_str()),
                                          RAD_STR("add", m.v_mean.c_str()), RAD_STR("dtype", "f32")),
                               RAD_NOWEIGHTS),
                     { m.b_vstate }, { m.b_vox });
    m.b_vox2 = decl_b(b, nm.f("vae.vox2"), RAD_F32, { vox, kC });
    {
        RAD_ARCH_TRY(map_copy(b, nm.f("vae.pqc.weight"), nm.ckpt("vae.post_quant_conv.weight")));
        m.w_pqc = decl_w(b, nm.f("vae.pqc.weight"), RAD_F32, { kC, kC, 1, 1, 1 }, acc,
                         RAD_SHARD_NONE, grp_layer(gv));
        m.b_pqc = wcopy(b, nm.f("vae.pqc.bias"), nm.ckpt("vae.post_quant_conv.bias"), RAD_F32,
                        { kC }, acc, gv);
        if (!m.w_pqc || !m.b_pqc) return RAD_E_INVAL;
        m.op_pqc = rw(b, RAD_OP(b, "gemm_nt_bias",
                                RAD_PARAMS(RAD_RANGE("M", 1, vox), RAD_INT("N", kC), RAD_INT("K", kC),
                                           RAD_STR("dtype", "f32")),
                                RAD_WEIGHTS(m.w_pqc, m.b_pqc)),
                      { m.b_vox }, { m.b_vox2 });
    }
    m.b_vpin = decl_b(b, nm.f("vae.pin"), RAD_F32, { vox, dim });
    m.op_vpin = dense_bias(b, nm, "vae.proj_in", "vae.decoder.proj_in", RAD_F32, "f32", dim, kC, vox,
                           m.b_vox2, m.b_vpin, &m.w_vpin2, &m.b_vpin2, acc, gv);
    m.b_vx = decl_b(b, nm.f("vae.x"), RAD_F32, { (int64_t)m.tiles * kPeriod, dim });
    m.w_reg = wcopy(b, nm.f("vae.registers"), nm.ckpt("vae.decoder.register_tokens"), RAD_F32,
                    { 1, kRegs, dim }, acc, gv);
    if (!m.w_reg || !m.op_vpin || !m.op_pqc) return RAD_E_INVAL;
    m.op_tiles = rw(b, RAD_OP(b, "vtile_gather",
                              RAD_PARAMS(RAD_RANGE("M", 1, (int64_t)m.tiles * kPeriod),
                                         RAD_INT("n", dim), RAD_INT("tile", kTile),
                                         RAD_INT("overlap", kOverlap), RAD_INT("frames", kDecLat),
                                         RAD_INT("period", kPeriod), RAD_INT("n_reg", kRegs),
                                         RAD_STR("dtype", "f32")),
                              RAD_WEIGHTS(m.w_reg)),
                    { m.b_vpin }, { m.b_vx });
    m.b_dpos = decl_b(b, nm.f("vae.pos"), RAD_F32, { 3, R });
    m.op_dpos = rw(b, RAD_OP(b, "grid_pos3",
                             RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("T", kDecLat),
                                        RAD_INT("H", kTile), RAD_INT("W", kTile),
                                        RAD_INT("period", kPeriod), RAD_F64("scale", 2.0 * M_PI)),
                             RAD_NOWEIGHTS),
                   {}, { m.b_dpos });
    m.b_dcu = decl_b(b, nm.f("vae.cu"), RAD_I32, { m.tpr + 1 });
    m.op_dcu = rw(b, RAD_OP(b, "iota",
                            RAD_PARAMS(RAD_RANGE("M", 1, m.tpr + 1), RAD_INT("start", 0),
                                       RAD_INT("step", kPeriod)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_dcu });
    if (!m.op_tiles || !m.op_dpos || !m.op_dcu) return RAD_E_INVAL;

    m.b_vn = decl_b(b, nm.f("vae.hn"), RAD_BF16, { R, dim });
    m.b_vqkv = decl_b(b, nm.f("vae.qkv"), RAD_BF16, { R, 3 * dim });
    m.b_vattn = decl_b(b, nm.f("vae.attn"), RAD_BF16, { R, dim });
    m.b_vo = decl_b(b, nm.f("vae.o"), RAD_BF16, { R, dim });
    m.b_vt = decl_b(b, nm.f("vae.t"), RAD_F32, { R, dim });
    m.b_vup = decl_b(b, nm.f("vae.up"), RAD_BF16, { R, 2 * ff });
    m.b_vact = decl_b(b, nm.f("vae.act"), RAD_BF16, { R, ff });
    const char* sections = nm.f("%lld %lld %lld", (long long)(rot / 6), (long long)(rot / 6),
                                (long long)(rot / 6));
    m.vit.resize((size_t)nl);
    for (int64_t l = 0; l < nl; ++l) {
        VitBlock& V = m.vit[(size_t)l];
        const int li = (int)l;
        const char* pre = nm.ckpt("vae.decoder.transformer_blocks.%d", li);
        const char* dn = nm.f("vae.blk.%d", li);
        V.w_n1 = wcopy(b, nm.f("%s.norm1.weight", dn), nm.ckpt("%s.norm1.weight", pre), RAD_F32,
                       { dim }, acc, gv);
        V.w_n2 = wcopy(b, nm.f("%s.norm2.weight", dn), nm.ckpt("%s.norm2.weight", pre), RAD_F32,
                       { dim }, acc, gv);
        V.s1 = wcopy(b, nm.f("%s.scale1", dn), nm.ckpt("%s.scale1", pre), RAD_F32, { dim }, acc, gv);
        V.s2 = wcopy(b, nm.f("%s.scale2", dn), nm.ckpt("%s.scale2", pre), RAD_F32, { dim }, acc, gv);
        const char* qn = nm.f("%s.attn_qkv.weight", dn);
        const char* qb = nm.f("%s.attn_qkv.bias", dn);
        RAD_ARCH_TRY(map_concat(b, qn, { nm.ckpt("%s.attn.to_q.weight", pre),
                                         nm.ckpt("%s.attn.to_k.weight", pre),
                                         nm.ckpt("%s.attn.to_v.weight", pre) }));
        RAD_ARCH_TRY(map_concat(b, qb, { nm.ckpt("%s.attn.to_q.bias", pre),
                                         nm.ckpt("%s.attn.to_k.bias", pre),
                                         nm.ckpt("%s.attn.to_v.bias", pre) }));
        V.w_qkv = decl_w(b, qn, RAD_BF16, { 3 * dim, dim }, acc, RAD_SHARD_NONE, grp_layer(gv));
        V.b_qkv = decl_w(b, qb, RAD_F32, { 3 * dim }, acc, RAD_SHARD_NONE, grp_layer(gv));
        V.w_o = wcopy(b, nm.f("%s.attn_out.weight", dn), nm.ckpt("%s.attn.to_out.0.weight", pre),
                      RAD_BF16, { dim, dim }, acc, gv);
        V.b_o = wcopy(b, nm.f("%s.attn_out.bias", dn), nm.ckpt("%s.attn.to_out.0.bias", pre),
                      RAD_F32, { dim }, acc, gv);
        const char* un = nm.f("%s.ffn_up.weight", dn);
        const char* ub = nm.f("%s.ffn_up.bias", dn);
        const char* usrc = nm.ckpt("%s.ff.net.0.proj.weight", pre);
        const char* ubsrc = nm.ckpt("%s.ff.net.0.proj.bias", pre);
        RAD_ARCH_TRY(map_rows(b, un, { { usrc, (int)ff, ff }, { usrc, 0, ff } }));
        RAD_ARCH_TRY(map_rows(b, ub, { { ubsrc, (int)ff, ff }, { ubsrc, 0, ff } }));
        V.w_up = decl_w(b, un, RAD_BF16, { 2 * ff, dim }, acc, RAD_SHARD_NONE, grp_layer(gv));
        V.b_up = decl_w(b, ub, RAD_F32, { 2 * ff }, acc, RAD_SHARD_NONE, grp_layer(gv));
        V.w_down = wcopy(b, nm.f("%s.ffn_down.weight", dn), nm.ckpt("%s.ff.net.2.weight", pre),
                         RAD_BF16, { dim, ff }, acc, gv);
        V.b_down = wcopy(b, nm.f("%s.ffn_down.bias", dn), nm.ckpt("%s.ff.net.2.bias", pre), RAD_F32,
                         { dim }, acc, gv);
        if (!V.w_n1 || !V.w_n2 || !V.s1 || !V.s2 || !V.w_qkv || !V.b_qkv || !V.w_o || !V.b_o ||
            !V.w_up || !V.b_up || !V.w_down || !V.b_down)
            return RAD_E_INVAL;
        auto gemm = [&](rad_weight w, rad_weight bias, int64_t N, int64_t K, rad_buf x, rad_buf y) {
            return rw(b, RAD_OP(b, "gemm_nt_bias",
                                RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("N", N), RAD_INT("K", K),
                                           RAD_STR("dtype", "bf16")),
                                RAD_WEIGHTS(w, bias)),
                      { x }, { y });
        };
        V.op_n1 = rmsnorm_op(b, R, dim, eps, V.w_n1, m.b_vx, m.b_vn);
        V.op_qkv = gemm(V.w_qkv, V.b_qkv, 3 * dim, dim, m.b_vn, m.b_vqkv);
        V.op_qkr = rw(b, RAD_OP(b, "qk_norm_rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("head_dim", hd),
                                           RAD_INT("n_head", dim / hd), RAD_INT("n_head_kv", dim / hd),
                                           RAD_F64("theta", theta), RAD_F64("eps", eps),
                                           RAD_STR("mode", "axial"), RAD_INT("rotary_dim", rot),
                                           RAD_STR("sections", sections)),
                                RAD_NOWEIGHTS),
                      { m.b_vqkv, m.b_dpos }, { m.b_vqkv });
        V.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("head_dim", hd),
                                            RAD_INT("gqa", 1), RAD_INT("causal", 0),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                            RAD_INT("max_seqlen", kPeriod)),
                                 RAD_NOWEIGHTS),
                       { m.b_vqkv, m.b_dcu }, { m.b_vattn });
        V.op_o = gemm(V.w_o, V.b_o, dim, dim, m.b_vattn, m.b_vo);
        V.op_s1 = rw(b, RAD_OP(b, "mul_weight",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", dim),
                                          RAD_STR("dtype", "f32")),
                               RAD_WEIGHTS(V.s1)),
                     { m.b_vo }, { m.b_vt });
        V.op_r1 = add_op(b, R, dim, "f32", m.b_vx, m.b_vt, m.b_vx);
        V.op_n2 = rmsnorm_op(b, R, dim, eps, V.w_n2, m.b_vx, m.b_vn);
        V.op_up = gemm(V.w_up, V.b_up, 2 * ff, dim, m.b_vn, m.b_vup);
        V.op_act = rw(b, RAD_OP(b, "silu_mul",
                                RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", ff),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      { m.b_vup }, { m.b_vact });
        V.op_down = gemm(V.w_down, V.b_down, dim, ff, m.b_vact, m.b_vo);
        V.op_s2 = rw(b, RAD_OP(b, "mul_weight",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", dim),
                                          RAD_STR("dtype", "f32")),
                               RAD_WEIGHTS(V.s2)),
                     { m.b_vo }, { m.b_vt });
        V.op_r2 = add_op(b, R, dim, "f32", m.b_vx, m.b_vt, m.b_vx);
        if (!V.op_n1 || !V.op_qkv || !V.op_qkr || !V.op_attn || !V.op_o || !V.op_s1 || !V.op_r1 ||
            !V.op_n2 || !V.op_up || !V.op_act || !V.op_down || !V.op_s2 || !V.op_r2)
            return RAD_E_INVAL;
    }
    m.w_vno = wcopy(b, nm.f("vae.norm_out.weight"), nm.ckpt("vae.decoder.norm_out.weight"), RAD_F32,
                    { dim }, acc, gv);
    m.b_vno = wcopy(b, nm.f("vae.norm_out.bias"), nm.ckpt("vae.decoder.norm_out.bias"), RAD_F32,
                    { dim }, acc, gv);
    if (!m.w_vno || !m.b_vno) return RAD_E_INVAL;
    m.op_vnorm = rw(b, RAD_OP(b, "layernorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", dim), RAD_F64("eps", eps),
                                         RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(m.w_vno, m.b_vno)),
                    { m.b_vx }, { m.b_vn });
    m.b_vpix = decl_b(b, nm.f("vae.pix"), RAD_BF16, { R, px });
    m.op_vproj = dense_bias(b, nm, "vae.proj_out", "vae.decoder.proj_out", RAD_BF16, "bf16", px, dim,
                            R, m.b_vn, m.b_vpix, &m.w_vpo, &m.b_vpo, acc, gv);
    if (!m.op_vnorm || !m.op_vproj) return RAD_E_INVAL;
    if (m.W > 1) {
        m.b_vpix_all = decl_b(b, nm.f("vae.pix_all"), RAD_BF16, { RA, px });
        m.op_vgather = rw(b, RAD_OP(b, "all_gather",
                                    RAD_PARAMS(RAD_INT("world_size", m.W),
                                               RAD_RANGE("numel", 1, R * px), RAD_STR("dtype", "bf16")),
                                    RAD_NOWEIGHTS),
                          { m.b_vpix }, { m.b_vpix_all });
        if (!m.op_vgather) return RAD_E_INVAL;
    }
    const int64_t fmax = (int64_t)kFpc + kFtail;
    m.b_prev = decl_b(b, nm.f("lane.prev"), RAD_BF16, { (int64_t)kMaxLanes * kFtail * m.max_area, 3 },
                      RAD_BUF_PERSIST);
    m.b_frames = decl_b(b, nm.f("frames_out"), RAD_U8, { fmax * m.max_area, 3 }, RAD_BUF_PERSIST);
    m.op_stitch = rw(b, RAD_OP(b, "vtile_stitch",
                               RAD_PARAMS(RAD_RANGE("M", 1, fmax * m.max_area), RAD_INT("tile", kTile),
                                          RAD_INT("overlap", kOverlap), RAD_INT("scale", kScale),
                                          RAD_INT("pt", 4), RAD_INT("skip", 3), RAD_INT("keep", kFpc),
                                          RAD_INT("blend", kFtail), RAD_INT("period", kPeriod),
                                          RAD_STR("mean", kPixMean), RAD_STR("std", kPixStd)),
                               RAD_NOWEIGHTS),
                     { m.W > 1 ? m.b_vpix_all : m.b_vpix, m.b_prev }, { m.b_prev, m.b_frames });
    return m.op_stitch ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the latent upscaler */
/* LatentResizer3D (the companion folder latent_upscaler/, its config.json and weights), declared
 * when the container carries it. Its architecture is the ComfyUI node's, which infers it from the
 * weights; the config states the same numbers and they are checked here. Weights are bf16 but the
 * factor's embedding MLP, f32 (the recipe casts it) so that its first layer, one input wide, is an
 * f32 GEMM. Activations are bf16 voxel rows of the network's width, statistics and accumulation
 * f32. */
static int declare_ups(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    const char* cls = rad_meta_gets(meta, "latent_upscaler._class_name", nullptr);
    if (!cls) return RAD_OK;
    Names& nm = m.nm;
    auto geti = [&](const char* k, long long def) {
        return (int64_t)rad_meta_geti(meta, nm.ckpt("latent_upscaler.%s", k), def);
    };
    const int64_t C = geti("channels", 512), E = geti("embed_dim", 64), G = geti("norm_num_groups", 32);
    const int64_t nin = geti("in_blocks", 12), nout = geti("out_blocks", 12);
    const int64_t every = geti("temporal_every", 2), tk = geti("temporal_kernel", 5);
    const int64_t K = geti("conv_kernel_size", 3);
    const std::vector<double> mean = meta_list(meta, "latent_upscaler.latent_normalization.mean");
    const std::vector<double> sd = meta_list(meta, "latent_upscaler.latent_normalization.std");
    const std::vector<double> range = meta_list(meta, "latent_upscaler.scale_range");
    const char* mode = rad_meta_gets(meta, "latent_upscaler.upsample_mode", "trilinear");
    if (std::strcmp(cls, "LatentResizer3D") || geti("in_channels", kC) != kC ||
        geti("out_channels", kC) != kC || K != 3 || geti("attn", 0) != 0 || C % G || C % 8 ||
        every < 0 || tk < 1 || tk % 2 == 0 || std::strcmp(mode, "trilinear") ||
        geti("upsample_align_corners", 0) != 0 || (int64_t)mean.size() != kC ||
        (int64_t)sd.size() != kC || range.size() != 2 || !(range[1] > 1.0)) {
        fprintf(stderr, "radiance: minimaxh3: latent_upscaler/ is a '%s' this plugin does not run: "
                        "it runs LatentResizer3D over %d channels, 3x3x3 convolutions, odd "
                        "temporal kernels, no attention, a trilinear resize without aligned "
                        "corners and a scale range\n", cls, (int)kC);
        return RAD_E_UNSUPPORTED;
    }
    m.ups = true;
    m.ups_max = range[1];
    m.uC = C;
    m.uE = E;
    /* patch_unfold normalises as x * mul + add: (x - mean) / std is x / std - mean / std. */
    std::vector<double> im(kC), ia(kC);
    for (int64_t c = 0; c < kC; ++c) {
        im[(size_t)c] = 1.0 / sd[(size_t)c];
        ia[(size_t)c] = -mean[(size_t)c] / sd[(size_t)c];
    }
    m.u_inv_mul = join_list(im);
    m.u_inv_add = join_list(ia);
    m.u_std = join_list(sd);
    m.u_mean = join_list(mean);

    /* The most voxels a frame of latents and a video hold, before or after: the small size is
     * never larger than the full one. */
    const int64_t V = max_lat_frames() * (m.max_area / (kScale * kScale));
    const int64_t P = m.maxV;
    const int acc = RAD_ACCESS_PER_TOKEN;
    auto wt = [&](const char* name, uint32_t dt, std::initializer_list<int64_t> shape) {
        return wcopy(b, nm.f("ups.%s", name), nm.ckpt("latent_upscaler.%s", name), dt, shape, acc,
                     kGrpUps);
    };
    auto conv = [&](rad_weight w, rad_weight bias, int64_t cin, int64_t cout, rad_buf x, rad_buf res,
                    rad_buf y) -> rad_op {
        if (!w || !bias) return 0;
        const rad_op op = RAD_OP(b, "conv3d_ndhwc",
                                 RAD_PARAMS(RAD_RANGE("M", 1, V), RAD_INT("cin", cin),
                                            RAD_INT("cout", cout), RAD_INT("k", K), RAD_INT("kt", K),
                                            RAD_STR("dtype", "bf16")),
                                 RAD_WEIGHTS(w, bias));
        return res ? rw(b, op, { x, res }, { y }) : rw(b, op, { x }, { y });
    };
    auto gnorm = [&](rad_weight w, rad_weight bias, bool mod, rad_buf x, rad_buf y) -> rad_op {
        if (!w || !bias) return 0;
        const rad_op op = RAD_OP(b, "group_norm",
                                 RAD_PARAMS(RAD_RANGE("M", 1, V), RAD_INT("n", C),
                                            RAD_INT("groups", G), RAD_F64("eps", 1e-5),
                                            RAD_STR("act", "silu"), RAD_STR("dtype", "bf16")),
                                 RAD_WEIGHTS(w, bias));
        return mod ? rw(b, op, { x, m.b_uemb, m.b_ust }, { y, m.b_ust })
                   : rw(b, op, { x, m.b_ust }, { y, m.b_ust });
    };
    auto gemm32 = [&](rad_weight w, rad_weight bias, int64_t N, int64_t Kd, rad_buf x,
                      rad_buf y) -> rad_op {
        if (!w || !bias) return 0;
        return rw(b, RAD_OP(b, "gemm_nt_bias",
                            RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("N", N), RAD_INT("K", Kd),
                                       RAD_STR("dtype", "f32")),
                            RAD_WEIGHTS(w, bias)),
                  { x }, { y });
    };

    /* ---- the latent in: unfolded and normalised in f32, then bf16 */
    m.b_uin32 = decl_b(b, nm.f("ups.in32"), RAD_F32, { V, kC });
    m.op_uunf = rw(b, RAD_OP(b, "patch_unfold",
                             RAD_PARAMS(RAD_RANGE("M", 1, V), RAD_INT("C", kC), RAD_INT("pt", 1),
                                        RAD_INT("ph", kPatch), RAD_INT("pw", kPatch),
                                        RAD_STR("mul", m.u_inv_mul.c_str()),
                                        RAD_STR("add", m.u_inv_add.c_str()),
                                        RAD_STR("dtype", "f32")),
                             RAD_NOWEIGHTS),
                   { m.b_vstate }, { m.b_uin32 });
    m.b_uin = decl_b(b, nm.f("ups.in"), RAD_BF16, { V, kC });
    m.op_ucast = cast_op(b, V, kC, "f32", "bf16", m.b_uin32, m.b_uin);

    /* ---- the factor's embedding: Linear(1, E), SiLU, Linear(E, E), then every residual block's
     * SiLU and Linear(E, 2C), each into its own row */
    m.b_ufac = decl_b(b, nm.f("ups.fac"), RAD_F32, { 1, 1 });
    m.op_ufac = rw(b, RAD_OP(b, "chan_affine",
                             RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", 1), RAD_STR("mul", "1"),
                                        RAD_STR("add", "-1"), RAD_STR("mode", "fma"),
                                        RAD_INT("round", 0), RAD_STR("dtype", "f32")),
                             RAD_NOWEIGHTS),
                   {}, { m.b_ufac });
    m.b_ue0 = decl_b(b, nm.f("ups.e0"), RAD_F32, { 1, E });
    m.b_ue0s = decl_b(b, nm.f("ups.e0s"), RAD_F32, { 1, E });
    m.b_ue1 = decl_b(b, nm.f("ups.e1"), RAD_F32, { 1, E });
    m.b_ue1s = decl_b(b, nm.f("ups.e1s"), RAD_F32, { 1, E });
    m.u_e0w = wt("embed.0.weight", RAD_F32, { E, 1 });
    m.u_e0b = wt("embed.0.bias", RAD_F32, { E });
    m.u_e2w = wt("embed.2.weight", RAD_F32, { E, E });
    m.u_e2b = wt("embed.2.bias", RAD_F32, { E });
    m.op_ue0 = gemm32(m.u_e0w, m.u_e0b, E, 1, m.b_ufac, m.b_ue0);
    m.op_ues0 = unary(b, "silu", 1, E, "f32", m.b_ue0, m.b_ue0s);
    m.op_ue2 = gemm32(m.u_e2w, m.u_e2b, E, E, m.b_ue0s, m.b_ue1);
    m.op_ues2 = unary(b, "silu", 1, E, "f32", m.b_ue1, m.b_ue1s);
    if (!m.op_uunf || !m.op_ucast || !m.op_ufac || !m.op_ue0 || !m.op_ues0 || !m.op_ue2 ||
        !m.op_ues2)
        return RAD_E_INVAL;

    /* The checkpoint's block lists: a residual block per index, a temporal block after the b-th
     * when b % every == 0, numbered in one sequence. */
    auto layout = [&](int64_t n, std::vector<UpsBlock>* out) {
        for (int64_t i = 0; i < n; ++i) {
            out->emplace_back();
            if (every > 0 && i % every == 0) {
                out->emplace_back();
                out->back().temporal = true;
            }
        }
    };
    layout(nin, &m.u_in);
    layout(nout, &m.u_out);
    int64_t n_res = 0;
    for (auto* list : { &m.u_in, &m.u_out })
        for (UpsBlock& B : *list) if (!B.temporal) B.emb = n_res++;
    m.b_uemb = decl_b(b, nm.f("ups.emb"), RAD_F32, { n_res, 2 * C });
    for (const char* side : { "in_blocks", "out_blocks" }) {
        std::vector<UpsBlock>& list = !std::strcmp(side, "in_blocks") ? m.u_in : m.u_out;
        for (size_t i = 0; i < list.size(); ++i) {
            UpsBlock& B = list[i];
            if (B.temporal) continue;
            B.ew = wt(nm.ckpt("%s.%zu.emb_layers.1.weight", side, i), RAD_F32, { 2 * C, E });
            B.eb = wt(nm.ckpt("%s.%zu.emb_layers.1.bias", side, i), RAD_F32, { 2 * C });
            if (!B.ew || !B.eb) return RAD_E_INVAL;
            B.op_e = rw(b, RAD_OP(b, "gemm_nt_bias",
                                  RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("N", 2 * C),
                                             RAD_INT("K", E), RAD_STR("dtype", "f32")),
                                  RAD_WEIGHTS(B.ew, B.eb)),
                        { m.b_ue1s }, { m.b_uemb });
            if (!B.op_e) return RAD_E_INVAL;
        }
    }

    /* ---- conv_in, the blocks before the resize, the resize, the blocks after it: the residual
     * stream is A before the resize and B after it, the other two the blocks' temporaries */
    m.b_ua = decl_b(b, nm.f("ups.a"), RAD_BF16, { V, C });
    m.b_ub = decl_b(b, nm.f("ups.b"), RAD_BF16, { V, C });
    m.b_uc = decl_b(b, nm.f("ups.c"), RAD_BF16, { V, C });
    m.b_ust = decl_b(b, nm.f("ups.gn_stats"), RAD_F32, { G * (2 + 3 * 1024) });
    m.u_cinw = wt("conv_in.weight", RAD_BF16, { C, kC, K, K, K });
    m.u_cinb = wt("conv_in.bias", RAD_BF16, { C });
    m.op_ucin = conv(m.u_cinw, m.u_cinb, kC, C, m.b_uin, 0, m.b_ua);
    if (!m.op_ucin) return RAD_E_INVAL;
    auto blocks = [&](const char* side, std::vector<UpsBlock>& list, rad_buf X, rad_buf T,
                      rad_buf H) -> int {
        for (size_t i = 0; i < list.size(); ++i) {
            UpsBlock& B = list[i];
            if (B.temporal) {
                B.n1w = wt(nm.ckpt("%s.%zu.norm.weight", side, i), RAD_BF16, { C });
                B.n1b = wt(nm.ckpt("%s.%zu.norm.bias", side, i), RAD_BF16, { C });
                B.c1w = wt(nm.ckpt("%s.%zu.dwconv.weight", side, i), RAD_BF16, { C, 1, tk, 1, 1 });
                B.c1b = wt(nm.ckpt("%s.%zu.dwconv.bias", side, i), RAD_BF16, { C });
                B.c2w = wt(nm.ckpt("%s.%zu.pwconv.weight", side, i), RAD_BF16, { C, C, 1, 1, 1 });
                B.c2b = wt(nm.ckpt("%s.%zu.pwconv.bias", side, i), RAD_BF16, { C });
                if (!B.c1w || !B.c1b || !B.c2w || !B.c2b) return RAD_E_INVAL;
                B.op_n1 = gnorm(B.n1w, B.n1b, false, X, T);
                B.op_c1 = rw(b, RAD_OP(b, "tconv_dw",
                                       RAD_PARAMS(RAD_RANGE("M", 1, V), RAD_INT("n", C),
                                                  RAD_INT("k", tk), RAD_STR("dtype", "bf16")),
                                       RAD_WEIGHTS(B.c1w, B.c1b)),
                             { T }, { H });
                B.op_c2 = rw(b, RAD_OP(b, "gemm_nt_bias",
                                       RAD_PARAMS(RAD_RANGE("M", 1, V), RAD_INT("N", C),
                                                  RAD_INT("K", C), RAD_STR("dtype", "bf16")),
                                       RAD_WEIGHTS(B.c2w, B.c2b)),
                             { H, X }, { X });
                if (!B.op_n1 || !B.op_c1 || !B.op_c2) return RAD_E_INVAL;
            } else {
                B.n1w = wt(nm.ckpt("%s.%zu.in_layers.0.weight", side, i), RAD_BF16, { C });
                B.n1b = wt(nm.ckpt("%s.%zu.in_layers.0.bias", side, i), RAD_BF16, { C });
                B.c1w = wt(nm.ckpt("%s.%zu.in_layers.2.weight", side, i), RAD_BF16, { C, C, K, K, K });
                B.c1b = wt(nm.ckpt("%s.%zu.in_layers.2.bias", side, i), RAD_BF16, { C });
                B.n2w = wt(nm.ckpt("%s.%zu.out_norm.weight", side, i), RAD_BF16, { C });
                B.n2b = wt(nm.ckpt("%s.%zu.out_norm.bias", side, i), RAD_BF16, { C });
                B.c2w = wt(nm.ckpt("%s.%zu.out_layers.2.weight", side, i), RAD_BF16, { C, C, K, K, K });
                B.c2b = wt(nm.ckpt("%s.%zu.out_layers.2.bias", side, i), RAD_BF16, { C });
                B.op_n1 = gnorm(B.n1w, B.n1b, false, X, T);
                B.op_c1 = conv(B.c1w, B.c1b, C, C, T, 0, H);
                B.op_n2 = gnorm(B.n2w, B.n2b, true, H, T);
                B.op_c2 = conv(B.c2w, B.c2b, C, C, T, X, X);
                if (!B.op_n1 || !B.op_c1 || !B.op_n2 || !B.op_c2) return RAD_E_INVAL;
            }
        }
        return RAD_OK;
    };
    RAD_ARCH_TRY(blocks("in_blocks", m.u_in, m.b_ua, m.b_ub, m.b_uc));
    m.op_ursz = rw(b, RAD_OP(b, "resize_bilinear",
                             RAD_PARAMS(RAD_RANGE("M", 1, V), RAD_INT("n", C),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { m.b_ua }, { m.b_ub });
    if (!m.op_ursz) return RAD_E_INVAL;
    RAD_ARCH_TRY(blocks("out_blocks", m.u_out, m.b_ub, m.b_ua, m.b_uc));

    /* ---- GroupNorm, SiLU, conv_out, then back to the lane: denormalised, folded into patch rows
     * and re-noised, the soundtrack with it */
    m.u_now = wt("norm_out.weight", RAD_BF16, { C });
    m.u_nob = wt("norm_out.bias", RAD_BF16, { C });
    m.op_unout = gnorm(m.u_now, m.u_nob, false, m.b_ub, m.b_ua);
    m.b_uout = decl_b(b, nm.f("ups.out"), RAD_BF16, { V, kC });
    m.u_coutw = wt("conv_out.weight", RAD_BF16, { kC, C, K, K, K });
    m.u_coutb = wt("conv_out.bias", RAD_BF16, { kC });
    m.op_ucout = conv(m.u_coutw, m.u_coutb, C, kC, m.b_ua, 0, m.b_uout);
    m.op_ufold = rw(b, RAD_OP(b, "patch_fold",
                              RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("C", kC), RAD_INT("pt", 1),
                                         RAD_INT("ph", kPatch), RAD_INT("pw", kPatch),
                                         RAD_STR("mul", m.u_std.c_str()),
                                         RAD_STR("add", m.u_mean.c_str()), RAD_STR("dtype", "f32")),
                              RAD_NOWEIGHTS),
                    { m.b_uout }, { m.b_vstate });
    auto renoise = [&](int64_t rows, int64_t n, rad_buf x) {
        return rw(b, RAD_OP(b, "flow_renoise",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                       RAD_INT("stream", 1), RAD_STR("dtype", "f32")),
                            RAD_NOWEIGHTS),
                  { x }, { x });
    };
    m.op_urnv = renoise(P, kPatchVals, m.b_vstate);
    m.op_urna = renoise(m.maxA, kAudioC, m.b_astate);
    if (!m.op_unout || !m.op_ucout || !m.op_ufold || !m.op_urnv || !m.op_urna) return RAD_E_INVAL;
    return RAD_OK;
}

/* ================================================================== the VAE encoders */
/* An encode pass's geometry (rad_arch_encode_geom). Vencode, 25 int32s: the five volumes (T, H, W)
 * of the clip and after each downsampler, the fold's (frames, h, w) for a full and for the last
 * clip, then the clip's frames, the clips, the padded frames and the latent frames. Aencode, 13:
 * the padded samples, then each of the six stages' length as a {0, n} cu. */
const int kVEncGeomN = 25, kAEncGeomN = 13;

static bool venc_geom(int64_t frames, int64_t h, int64_t w, int32_t* out) {
    if (frames < 1 || frames > kEncMaxFrames || h < kVaeRatio || w < kVaeRatio ||
        h % kVaeRatio || w % kVaeRatio)
        return false;
    const int64_t clip = frames == 1 ? 1 : kVaeClipLength;
    const int64_t n_clips = frames == 1 ? 1 : (frames + kVaeClipLength - 1) / kVaeClipLength;
    const int64_t t1 = (clip + 1) / 2, t2 = (t1 + 1) / 2;
    /* Every level's volume fits its buffers (declare_venc's R): a video's clip at most 256 x 256,
     * a single frame up to the canvas. The spatial divisors cancel, leaving the frames. */
    {
        const int64_t tv[5] = { clip, clip, t1, t2, t2 };
        const int64_t tf5[5] = { kVaeClipLength, kVaeClipLength, (kVaeClipLength + 1) / 2,
                                 ((kVaeClipLength + 1) / 2 + 1) / 2,
                                 ((kVaeClipLength + 1) / 2 + 1) / 2 };
        for (int i = 0; i < 5; ++i)
            if (tv[i] * h * w > std::max(tf5[i], (int64_t)kEncOneFrames) *
                                    (int64_t)kEncMaxSide * kEncMaxSide)
                return false;
    }
    const int64_t lat = vae_latent_frames(frames);
    const int64_t full = vae_clip_latents(clip);
    const int64_t last = lat - full * (n_clips - 1);
    if (lat < 1 || full < 1 || last < 1 || last > full) return false;
    if (!out) return true;
    const int64_t g[5][3] = { { clip, h, w }, { clip, h / 2, w / 2 }, { t1, h / 4, w / 4 },
                              { t2, h / 8, w / 8 }, { t2, h / 16, w / 16 } };
    int32_t* o = out;
    for (const auto& v : g)
        for (int64_t x : v) *o++ = (int32_t)x;
    const int64_t f[2][3] = { { full, h / (kVaeRatio * kPatch), w / (kVaeRatio * kPatch) },
                              { last, h / (kVaeRatio * kPatch), w / (kVaeRatio * kPatch) } };
    for (const auto& v : f)
        for (int64_t x : v) *o++ = (int32_t)x;
    *o++ = (int32_t)clip;
    *o++ = (int32_t)n_clips;
    *o++ = (int32_t)(n_clips * clip);
    *o++ = (int32_t)lat;
    return o - out == kVEncGeomN;
}

static bool aenc_geom(int64_t samples, int64_t ch, int32_t* out) {
    if (samples < 1 || samples > kEncMaxSamples || ch < 1 || ch > kAudioCh) return false;
    const int64_t n0 = vae_audio_latents(samples) * kAudioHop;
    if (n0 > kEncMaxSamples) return false;
    if (!out) return true;
    static const int64_t kStr[5] = { 2, 4, 4, 5, 5 };
    out[0] = (int32_t)n0;
    int64_t n = n0;
    for (int i = 0; i <= 5; ++i) {
        out[1 + 2 * i] = 0;
        out[2 + 2 * i] = (int32_t)n;
        if (i < 5) n /= kStr[i];
    }
    return true;
}

/* A media's first row in b_evc / b_enca: nine pictures then three videos, three videos'
 * soundtracks then three standalone ones and the song guide. A request with picture keyframes
 * carries no references, so its two keyframes take the first two pictures' rows. -1 for no
 * slot. */
static int64_t venc_slot_row(int32_t kind, int32_t slot) {
    int64_t idx = -1;
    if (kind == RAD_ENC_IMAGE && slot >= 1 && slot <= 9) idx = slot - 1;
    else if (kind == RAD_ENC_KEYFRAME && slot >= 1 && slot <= 2) idx = slot - 1;
    else if (kind == RAD_ENC_VIDEO && slot >= 1 && slot <= 3) idx = 9 + slot - 1;
    return idx < 0 ? -1 : idx * (int64_t)kEncSlotRowsV;
}
static int64_t aenc_slot_row(int32_t kind, int32_t slot) {
    int64_t idx = -1;
    if (kind == RAD_ENC_VAUDIO && slot >= 1 && slot <= 3) idx = slot - 1;
    else if (kind == RAD_ENC_AUDIO && slot >= 1 && slot <= 3) idx = 3 + slot - 1;
    else if (kind == RAD_ENC_GUIDE && slot == 1) idx = 6;
    return idx < 0 ? -1 : idx * (int64_t)kEncSlotRowsA;
}

/* The encoder's CausalConv3d: causal in time, reflect-padded in space, on bf16 rows. */
static rad_op venc_conv(RadBuilder* b, int64_t M, int64_t cin, int64_t cout, int64_t k,
                        rad_weight w, rad_weight bias, rad_buf x, rad_buf res, rad_buf y) {
    if (!w || !bias) return 0;
    const rad_op op = RAD_OP(b, "conv3d_ndhwc",
                             RAD_PARAMS(RAD_RANGE("M", 1, M), RAD_INT("cin", cin),
                                        RAD_INT("cout", cout), RAD_INT("k", k), RAD_INT("kt", k),
                                        RAD_INT("causal", 1), RAD_INT("reflect", 1),
                                        RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(w, bias));
    return res ? rw(b, op, { x, res }, { y }) : rw(b, op, { x }, { y });
}

/* GroupNorm over a level's whole clip volume, then SiLU. */
static rad_op venc_gn(RadBuilder* b, int64_t M, int64_t n, int64_t G, double eps, rad_weight w,
                      rad_weight bias, rad_buf x, rad_buf y, rad_buf st) {
    if (!w || !bias) return 0;
    const rad_op op = RAD_OP(b, "group_norm",
                             RAD_PARAMS(RAD_RANGE("M", 1, M), RAD_INT("n", n),
                                        RAD_INT("groups", G), RAD_F64("eps", eps),
                                        RAD_STR("act", "silu"), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(w, bias));
    return rw(b, op, { x, st }, { y, st });
}

/* The video VAE's encoder (AutoencoderKLMiniMaxH3), when the container carries it: pixels in
 * 0..255 normalised to bf16, then the levels a clip at a time, at the geometry `enc_geom_d` gives
 * each pass. */
static int declare_venc(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    /* Whether the container has it, as the vision tower asks (rad_block_vit.h). */
    if (map_copy(b, nm.f("venc.cin.weight"), nm.ckpt("vae.encoder.conv_in.weight")) < 0)
        return RAD_E_INVAL;
    {
        RadEncoding enc;
        if (rad_weight_encoding(b, nm.f("venc.cin.weight"), &enc, nullptr, nullptr) != RAD_OK) {
            fprintf(stderr, "radiance: minimaxh3: the container ships no video VAE encoder; "
                            "declaring without it\n");
            return RAD_OK;
        }
    }
    const std::vector<double> dims = meta_list(meta, "vae.block_out_channels");
    const std::vector<double> sds = meta_list(meta, "vae.spatial_downsample_factors");
    const std::vector<double> tds = meta_list(meta, "vae.temporal_downsample_factors");
    const double eps = rad_meta_getf(meta, "vae.norm_eps", 1e-6);
    const int64_t G = rad_meta_geti(meta, "vae.norm_num_groups", 32);
    if (dims.size() != 6 || sds.size() != 6 || tds.size() != 6 ||
        rad_meta_geti(meta, "vae.layers_per_block", 2) != 2 ||
        rad_meta_geti(meta, "vae.clip_length", kVaeClipLength) != kVaeClipLength ||
        rad_meta_geti(meta, "vae.in_channels", 3) != 3 ||
        rad_meta_geti(meta, "vae.latent_channels", kC) != kC ||
        std::strcmp(rad_meta_gets(meta, "vae.spatial_padding_mode", "reflect"), "reflect") ||
        G < 1 || eps <= 0.0) {
        fprintf(stderr, "radiance: minimaxh3: the video VAE is not the released encoder (six "
                        "levels of two blocks, 17-frame clips, reflect padding)\n");
        return RAD_E_UNSUPPORTED;
    }
    for (size_t i = 0; i < 6; ++i)
        if ((int64_t)dims[i] % G || ((int64_t)sds[i] != 1 && (int64_t)sds[i] != 2) ||
            ((int64_t)tds[i] != 1 && (int64_t)tds[i] != 2))
            return RAD_E_UNSUPPORTED;
    m.venc = true;
    /* Read once a media, so they live in pinned host memory like the text encoder. */
    const int acc = RAD_ACCESS_PER_REQUEST;
    const int gv = kGrpVenc;
    auto wt = [&](const char* dn, const char* src, std::initializer_list<int64_t> shape) {
        return wcopy(b, dn, src, RAD_F32, shape, acc, gv);
    };
    /* The 3x3x3 convolutions' weights, bf16 as the implicit-GEMM kernel stores them. */
    auto wb = [&](const char* dn, const char* src, std::initializer_list<int64_t> shape) {
        return wcopy(b, dn, src, RAD_BF16, shape, acc, gv);
    };
    /* conv_in's and the k=1 convolutions', f32, read by a host kernel (kGrpVencH). */
    auto wh = [&](const char* dn, const char* src, std::initializer_list<int64_t> shape) {
        return wcopy(b, dn, src, RAD_F32, shape, acc, kGrpVencH);
    };

    /* Each level's frames, rows and rows a frame: a full clip at 256 x 256, or a frame of the
     * canvas, whichever is more. */
    int64_t tf[6], sd[6] = { 1, 1, 1, 1, 1, 1 }, R[6], PF[6];
    tf[0] = kVaeClipLength;
    for (int i = 1; i < 6; ++i) {
        tf[i] = (tf[i - 1] + (int64_t)tds[i - 1] - 1) / (int64_t)tds[i - 1];
        sd[i] = sd[i - 1] * (int64_t)sds[i - 1];
    }
    const int64_t HW = kEncMaxSide * kEncMaxSide;
    for (int i = 0; i < 6; ++i) {
        PF[i] = HW / (sd[i] * sd[i]);
        R[i] = std::max(tf[i], (int64_t)kEncOneFrames) * PF[i];
    }
    const int64_t Rmax = R[0];

    /* Pixels: f32 normalised, then bf16; the statistics every GroupNorm shares. */
    std::vector<double> pm(3), pa(3);
    for (int c = 0; c < 3; ++c) {
        pm[(size_t)c] = 1.0 / (255.0 * kVaePixelStd[c]);
        pa[(size_t)c] = -kVaePixelMean[c] / kVaePixelStd[c];
    }
    m.ve_pix_mul = join_list(pm);
    m.ve_pix_add = join_list(pa);
    std::vector<double> fm(kC), fa(kC);
    for (int64_t c = 0; c < kC; ++c) {
        fm[(size_t)c] = 1.0 / kVaeLatentsStd[c];
        fa[(size_t)c] = -kVaeLatentsMean[c] / kVaeLatentsStd[c];
    }
    m.ve_fold_mul = join_list(fm);
    m.ve_fold_add = join_list(fa);
    m.b_epix32 = decl_b(b, nm.f("venc.pix32"), RAD_F32, { Rmax, 3 });
    /* conv_in and the k=1 projections run on the host. Their activation operands must be
     * pinned host buffers too: joining a device queue does not make its VRAM CPU-readable.
     * Device convolutions can read/write these boundary buffers over the link in place. */
    m.b_epix = decl_b(b, nm.f("venc.pix"), RAD_BF16, { Rmax, 3 },
                      RAD_BUF_TRANSIENT, RAD_DOMAIN_HOST);
    m.b_egn = decl_b(b, nm.f("venc.gn"), RAD_F32, { G * (2 + 3 * 1024) });
    const rad_buf st = m.b_egn;
    /* The tiled encode's two planes and the quant_conv are declared ahead of the trunk, and the
     * tiled ops last. The arena packs by declared lifetime (core/build/rad_bufplan.cpp), and the
     * tiled pass reruns the whole trunk once a tile while `b_vtile` holds every tile's pixels and
     * `b_eqt` the finished tiles' moments, so both lifetimes must span the trunk's ops (from
     * op_ve_pix's and op_ve_q's use to op_vt_crop and op_vt_join). */
    m.enc_tiles = max_tiles(kMaxArea);
    m.b_tplan = decl_b(b, nm.f("venc.tplan"), RAD_I32, { m.enc_tiles, 22 },
                       RAD_BUF_PERSIST, RAD_DOMAIN_HOST);
    m.b_vtile = decl_b(b, nm.f("venc.vtile"), RAD_F32,
                       { m.enc_tiles * (int64_t)kEncMaxSide * kEncMaxSide, 3 });
    m.b_eq48 = decl_b(b, nm.f("venc.q48"), RAD_BF16, { R[5], 48 },
                      RAD_BUF_TRANSIENT, RAD_DOMAIN_HOST);
    m.b_eq2 = decl_b(b, nm.f("venc.q"), RAD_BF16, { R[5], 48 },
                     RAD_BUF_TRANSIENT, RAD_DOMAIN_HOST);
    m.b_eqt = decl_b(b, nm.f("venc.qt"), RAD_BF16, { m.enc_tiles * (int64_t)kTile * kTile, 48 },
                     RAD_BUF_TRANSIENT, RAD_DOMAIN_HOST);
    m.ve_qw = wh(nm.f("venc.q.weight"), nm.ckpt("vae.quant_conv.weight"), { 48, 48, 1, 1, 1 });
    m.ve_qb = wh(nm.f("venc.q.bias"), nm.ckpt("vae.quant_conv.bias"), { 48 });
    /* The whole-canvas pass writes b_eq2; the tiled pass writes a tile's slice of b_eqt, which
     * op_vt_join blends into b_eq2. */
    m.op_ve_q = rw(b, RAD_OP(b, "conv3d_ndhwc",
                             RAD_PARAMS(RAD_RANGE("M", 1, R[5]), RAD_INT("cin", 48),
                                        RAD_INT("cout", 48), RAD_INT("k", 1), RAD_INT("kt", 1),
                                        RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(m.ve_qw, m.ve_qb)),
                   { m.b_eq48 }, { m.b_eq2, m.b_eqt });
    if (!m.op_ve_q) return RAD_E_INVAL;
    m.op_ve_pix = rw(b, RAD_OP(b, "chan_affine",
                               RAD_PARAMS(RAD_RANGE("M", 1, Rmax), RAD_INT("n", 3),
                                          RAD_STR("mul", m.ve_pix_mul.c_str()),
                                          RAD_STR("add", m.ve_pix_add.c_str()),
                                          RAD_STR("mode", "fma"), RAD_INT("round", 0),
                                          RAD_STR("dtype", "f32")),
                               RAD_NOWEIGHTS),
                     { m.b_vtile }, { m.b_epix32 });
    m.op_ve_pcast = cast_op(b, Rmax, 3, "f32", "bf16", m.b_epix32, m.b_epix);
    /* Mapped by the presence check above. */
    m.ve_cinw = decl_w(b, nm.f("venc.cin.weight"), RAD_F32, { (int64_t)dims[0], 3, 3, 3, 3 },
                       acc, RAD_SHARD_NONE, grp_layer(kGrpVencH));
    m.ve_cinb = wh(nm.f("venc.cin.bias"), nm.ckpt("vae.encoder.conv_in.bias"), { (int64_t)dims[0] });
    if (!m.op_ve_pix || !m.op_ve_pcast) return RAD_E_INVAL;
    m.v_lv.reserve(6);
    for (int L = 0; L < 6; ++L) {
        m.v_lv.emplace_back();
        Model::VEncLevel& E = m.v_lv.back();
        const int64_t cout = (int64_t)dims[L], cin = L == 0 ? cout : (int64_t)dims[L - 1];
        const bool sc = L > 0 && cout != cin;
        E.cin = cin, E.cout = cout, E.rows = R[L], E.pf = PF[L], E.frames = tf[L];
        E.a = decl_b(b, nm.f("venc.l%d.a", L), RAD_BF16, { R[L], cout },
                     RAD_BUF_TRANSIENT, RAD_DOMAIN_HOST);
        E.b = decl_b(b, nm.f("venc.l%d.b", L), RAD_BF16, { R[L], cout });
        E.c = decl_b(b, nm.f("venc.l%d.c", L), RAD_BF16, { R[L], cout });
        /* A level that changes the width takes its input in a buffer of that width, which the
         * downsampler before it writes, or which is the level before's stream where there is
         * none. */
        if (sc) {
            const bool dn_in = (int64_t)sds[L - 1] == 2 || (int64_t)tds[L - 1] == 2;
            E.in = dn_in ? decl_b(b, nm.f("venc.l%d.in", L), RAD_BF16, { R[L], cin },
                                 RAD_BUF_TRANSIENT, RAD_DOMAIN_HOST)
                         : m.v_lv[(size_t)L - 1].a;
            const char* sc_src = nm.ckpt("vae.encoder.down_blocks.%d.resnets.0.conv_shortcut", L);
            E.scw = wh(nm.f("venc.l%d.sc.weight", L), nm.ckpt("%s.weight", sc_src),
                       { cout, cin, 1, 1, 1 });
            E.scb = wh(nm.f("venc.l%d.sc.bias", L), nm.ckpt("%s.bias", sc_src), { cout });
            if (!E.scw || !E.scb) return RAD_E_INVAL;
            E.op_sc = rw(b, RAD_OP(b, "conv3d_ndhwc",
                                   RAD_PARAMS(RAD_RANGE("M", 1, R[L]), RAD_INT("cin", cin),
                                              RAD_INT("cout", cout), RAD_INT("k", 1),
                                              RAD_INT("kt", 1), RAD_STR("dtype", "bf16")),
                                   RAD_WEIGHTS(E.scw, E.scb)),
                         { E.in }, { E.a });
            if (!E.op_sc) return RAD_E_INVAL;
        }
        for (int blk = 0; blk < 2; ++blk) {
            Model::VEncBlock& B = E.blk[blk];
            const char* bs = nm.ckpt("vae.encoder.down_blocks.%d.resnets.%d", L, blk);
            /* norm1 at the block's input width: a widening level's first block normalises `in`
             * in place. */
            const int64_t w1 = blk == 0 && sc ? cin : cout;
            const rad_buf xa = blk == 0 && sc ? E.in : E.a;
            const rad_buf nb = blk == 0 && sc ? E.in : E.b;
            B.n1w = wt(nm.f("venc.l%d.b%d.n1w", L, blk), nm.ckpt("%s.norm1.weight", bs), { w1 });
            B.n1b = wt(nm.f("venc.l%d.b%d.n1b", L, blk), nm.ckpt("%s.norm1.bias", bs), { w1 });
            B.c1w = wb(nm.f("venc.l%d.b%d.c1w", L, blk), nm.ckpt("%s.conv1.weight", bs), { cout, w1, 3, 3, 3 });
            B.c1b = wt(nm.f("venc.l%d.b%d.c1b", L, blk), nm.ckpt("%s.conv1.bias", bs), { cout });
            B.n2w = wt(nm.f("venc.l%d.b%d.n2w", L, blk), nm.ckpt("%s.norm2.weight", bs), { cout });
            B.n2b = wt(nm.f("venc.l%d.b%d.n2b", L, blk), nm.ckpt("%s.norm2.bias", bs), { cout });
            B.c2w = wb(nm.f("venc.l%d.b%d.c2w", L, blk), nm.ckpt("%s.conv2.weight", bs), { cout, cout, 3, 3, 3 });
            B.c2b = wt(nm.f("venc.l%d.b%d.c2b", L, blk), nm.ckpt("%s.conv2.bias", bs), { cout });
            if (!B.n1w || !B.n1b || !B.c1w || !B.c1b || !B.n2w || !B.n2b || !B.c2w || !B.c2b)
                return RAD_E_INVAL;
            B.op_n1 = venc_gn(b, R[L], w1, G, eps, B.n1w, B.n1b, xa, nb, st);
            B.op_c1 = venc_conv(b, R[L], w1, cout, 3, B.c1w, B.c1b, nb, 0, E.c);
            B.op_n2 = venc_gn(b, R[L], cout, G, eps, B.n2w, B.n2b, E.c, E.b, st);
            B.op_c2 = venc_conv(b, R[L], cout, cout, 3, B.c2w, B.c2b, E.b, E.a, E.a);
            if (!B.op_n1 || !B.op_c1 || !B.op_n2 || !B.op_c2) return RAD_E_INVAL;
        }
    }
    /* conv_in, into level 0. */
    m.op_ve_cin = venc_conv(b, Rmax, 3, (int64_t)dims[0], 3, m.ve_cinw, m.ve_cinb, m.b_epix, 0,
                            m.v_lv[0].a);
    if (!m.op_ve_cin) return RAD_E_INVAL;

    /* The downsamplers after the first four levels (Downsample3D): padded below and right,
     * strided in space and, where the config says, in time. */
    for (int L = 0; L < 4; ++L) {
        Model::VEncLevel& E = m.v_lv[(size_t)L];
        Model::VEncLevel& X = m.v_lv[(size_t)L + 1];
        const char* bs = nm.ckpt("vae.encoder.down_blocks.%d.downsamplers.0.conv", L);
        E.dnw = wb(nm.f("venc.l%d.dnw", L), nm.ckpt("%s.weight", bs), { E.cout, E.cout, 3, 3, 3 });
        E.dnb = wt(nm.f("venc.l%d.dnb", L), nm.ckpt("%s.bias", bs), { E.cout });
        if (!E.dnw || !E.dnb) return RAD_E_INVAL;
        const rad_op op = RAD_OP(b, "conv3d_ndhwc",
                                 RAD_PARAMS(RAD_RANGE("M", 1, E.rows), RAD_INT("cin", E.cout),
                                            RAD_INT("cout", E.cout), RAD_INT("k", 3),
                                            RAD_INT("kt", 3), RAD_INT("stride", (int64_t)sds[L]),
                                            RAD_INT("stride_t", (int64_t)tds[L]),
                                            RAD_INT("causal", 1), RAD_STR("pad", "down"),
                                            RAD_INT("reflect", 1), RAD_STR("dtype", "bf16")),
                                 RAD_WEIGHTS(E.dnw, E.dnb));
        rad_buf y = X.op_sc ? X.in : X.a;
        E.op_dn = rw(b, op, { E.a }, { y });
        if (!E.op_dn) return RAD_E_INVAL;
    }

    /* norm_out, conv_out and quant_conv, then the fold of the means, the first 24 of the 48
     * columns, normalised into patch rows of 96. */
    m.ve_now = wt(nm.f("venc.nout.weight"), nm.ckpt("vae.encoder.norm_out.weight"),
                  { (int64_t)dims[5] });
    m.ve_nob = wt(nm.f("venc.nout.bias"), nm.ckpt("vae.encoder.norm_out.bias"), { (int64_t)dims[5] });
    m.op_ve_nout = venc_gn(b, R[5], (int64_t)dims[5], G, eps, m.ve_now, m.ve_nob, m.v_lv[5].a,
                           m.v_lv[5].b, st);
    m.ve_coutw = wb(nm.f("venc.cout.weight"), nm.ckpt("vae.encoder.conv_out.weight"),
                    { 48, (int64_t)dims[5], 3, 3, 3 });
    m.ve_coutb = wt(nm.f("venc.cout.bias"), nm.ckpt("vae.encoder.conv_out.bias"), { 48 });
    m.op_ve_cout = venc_conv(b, R[5], (int64_t)dims[5], 48, 3, m.ve_coutw, m.ve_coutb,
                             m.v_lv[5].b, 0, m.b_eq48);
    m.b_evc = decl_b(b, nm.f("lane.evc"), RAD_F32,
                     { (int64_t)kEncSlotsV * kEncSlotRowsV, kPatchVals }, RAD_BUF_PERSIST);
    m.op_ve_fold = rw(b, RAD_OP(b, "patch_fold",
                                RAD_PARAMS(RAD_RANGE("M", 1, kEncSlotRowsV), RAD_INT("C", kC),
                                           RAD_INT("pt", 1), RAD_INT("ph", kPatch),
                                           RAD_INT("pw", kPatch),
                                           RAD_STR("mul", m.ve_fold_mul.c_str()),
                                           RAD_STR("add", m.ve_fold_add.c_str()),
                                           RAD_STR("dtype", "f32")),
                                RAD_NOWEIGHTS),
                      { m.b_eq2 }, { m.b_evc });
    /* ComfyUI blends every visual condition's latents with noise, a r + (1 - a) z at a = 0.999
     * (_cond_video_rows); its audio conditions take none. z is stream 2 of the request's seed,
     * restarted at each item, as ComfyUI reseeds each. */
    m.op_eaug = rw(b, RAD_OP(b, "cond_aug",
                             RAD_PARAMS(RAD_RANGE("M", 1, kEncSlotRowsV), RAD_INT("n", kPatchVals),
                                        RAD_INT("stream", 2), RAD_F64("aug", 0.999),
                                        RAD_STR("dtype", "f32")),
                             RAD_NOWEIGHTS),
                   { m.b_evc }, { m.b_evc });
    if (!m.op_ve_nout || !m.op_ve_cout || !m.op_ve_q || !m.op_ve_fold || !m.op_eaug)
        return RAD_E_INVAL;
    /* The tiled encode's ops, declared last: b_vtile and b_eqt then live across the whole trunk,
     * which the tiled pass runs once a tile, so no trunk buffer shares their bytes. */
    m.op_vt_plan = rw(b, RAD_OP(b, "vtile_plan",
                                RAD_PARAMS(RAD_RANGE("M", 1, m.enc_tiles), RAD_INT("tile", kTile),
                                           RAD_INT("overlap", kOverlap),
                                           RAD_INT("ratio", kVaeRatio)),
                                RAD_NOWEIGHTS),
                     {}, { m.b_tplan });
    m.op_vt_crop = rw(b, RAD_OP(b, "vtile_crop",
                                RAD_PARAMS(RAD_RANGE("M", 1, m.enc_tiles * (int64_t)kEncMaxSide * kEncMaxSide),
                                           RAD_INT("tile", kTile), RAD_INT("overlap", kOverlap),
                                           RAD_INT("ratio", kVaeRatio), RAD_STR("dtype", "f32")),
                                RAD_NOWEIGHTS),
                     {}, { m.b_vtile });
    m.op_vt_join = rw(b, RAD_OP(b, "vtile_join",
                                RAD_PARAMS(RAD_RANGE("M", 1, R[5]), RAD_INT("tile", kTile),
                                           RAD_INT("overlap", kOverlap), RAD_INT("n", 48),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                     { m.b_eqt }, { m.b_eq2 });
    if (!m.op_vt_plan || !m.op_vt_crop || !m.op_vt_join) return RAD_E_INVAL;
    return RAD_OK;
}

/* The audio VAE's encoder (audio_vae.py's Encoder and AttnProjection), when the container
 * carries it, a channel at a time; the weight norms are folded at convert. */
static int declare_aenc(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    if (map_copy(b, nm.f("aenc.mp.weight"), nm.ckpt("audio_vae.mean_proj.weight")) < 0)
        return RAD_E_INVAL;
    {
        RadEncoding enc;
        if (rad_weight_encoding(b, nm.f("aenc.mp.weight"), &enc, nullptr, nullptr) != RAD_OK) {
            fprintf(stderr, "radiance: minimaxh3: the container ships no audio VAE encoder; "
                            "declaring without it\n");
            return RAD_OK;
        }
    }
    const std::vector<double> rates = meta_list(meta, "audio_vae.encoder_rates");
    const int64_t ld = rad_meta_geti(meta, "audio_vae.latent_dim", 2048);
    const std::vector<double> mean = meta_list(meta, "audio_vae.latents_mean");
    const std::vector<double> sd = meta_list(meta, "audio_vae.latents_std");
    const int64_t heads = rad_meta_geti(meta, "audio_vae.num_attention_heads", 8);
    if (rates.size() != 5 || ld != 2048 || (int64_t)mean.size() != kAudioC ||
        (int64_t)sd.size() != kAudioC || heads != 8 ||
        rad_meta_geti(meta, "audio_vae.encoder_dim", 64) != 64 ||
        rad_meta_geti(meta, "audio_vae.latent_channels", kAudioC) != kAudioC) {
        fprintf(stderr, "radiance: minimaxh3: the audio VAE is not the released encoder (five "
                        "strides to 800 samples a latent, eight attention heads)\n");
        return RAD_E_UNSUPPORTED;
    }
    m.aenc = true;
    /* Read once a media, as the video encoder's. */
    const int acc = RAD_ACCESS_PER_REQUEST;
    const int ga = kGrpAenc;
    /* The snake alphas: `snake` in f32 is a host kernel (kGrpVencH). */
    const int gh = kGrpAencH;
    static const int64_t kStr[5] = { 2, 4, 4, 5, 5 };
    for (int i = 0; i < 5; ++i)
        if ((int64_t)rates[(size_t)i] != kStr[i]) return RAD_E_UNSUPPORTED;
    int64_t nn[6], uu[5];
    nn[0] = kEncMaxSamples;
    for (int i = 0; i < 5; ++i) nn[i + 1] = nn[i] / kStr[i];
    for (int i = 0; i < 5; ++i) uu[i] = 64 << i;
    const int64_t hd = ld / 8;                 /* a head's width */

    /* The latents' normalisation, the chain's last op. */
    std::vector<double> nm_(kAudioC), na(kAudioC);
    for (int64_t c = 0; c < kAudioC; ++c) {
        nm_[(size_t)c] = 1.0 / sd[(size_t)c];
        na[(size_t)c] = -mean[(size_t)c] / sd[(size_t)c];
    }
    m.ae_norm_mul = join_list(nm_);
    m.ae_norm_add = join_list(na);

    /* conv_in: a sample a row. */
    const char* bs = nm.ckpt("audio_vae.encoder.block.0");
    RAD_ARCH_TRY(map_wnorm(b, nm.f("aenc.cin.weight"), nm.ckpt("%s.weight_g", bs),
                           nm.ckpt("%s.weight_v", bs)));
    m.ae_cinw = decl_w(b, nm.f("aenc.cin.weight"), RAD_F32, { 64, 1, 7 }, acc, RAD_SHARD_NONE,
                       grp_layer(ga));
    m.ae_cinb = wcopy(b, nm.f("aenc.cin.bias"), nm.ckpt("%s.bias", bs), RAD_F32, { 64 }, acc, ga);
    m.b_ax[0] = decl_b(b, nm.f("aenc.x0"), RAD_F32, { nn[0], 64 });
    if (!m.ae_cinw || !m.ae_cinb || !m.b_ax[0]) return RAD_E_INVAL;
    m.op_ae_cin = rw(b, RAD_OP(b, "conv1d",
                               RAD_PARAMS(RAD_RANGE("M", 1, nn[0]), RAD_INT("cin", 1),
                                          RAD_INT("cout", 64), RAD_INT("k", 7), RAD_INT("dil", 1),
                                          RAD_STR("act", "none"), RAD_STR("dtype", "f32")),
                               RAD_WEIGHTS(m.ae_cinw, m.ae_cinb)),
                     {}, { m.b_ax[0] });
    if (!m.op_ae_cin) return RAD_E_INVAL;

    for (int blk = 0; blk < 5; ++blk) {
        Model::AEncBlk& A = m.ae_blk[blk];
        const int64_t u = uu[blk], d = u * 2, np = nn[blk], no = nn[blk + 1];
        const int64_t s = kStr[blk];
        m.b_et1[blk] = decl_b(b, nm.f("aenc.t1_%d", blk), RAD_F32, { np, u });
        m.b_et2[blk] = decl_b(b, nm.f("aenc.t2_%d", blk), RAD_F32, { np, u });
        m.b_ax[blk + 1] = decl_b(b, nm.f("aenc.x%d", blk + 1), RAD_F32, { no, d });
        A.bi = decl_b(b, nm.f("aenc.bf1_%d", blk), RAD_BF16, { np, u });
        A.bo = decl_b(b, nm.f("aenc.bf2_%d", blk), RAD_BF16, { no, d });
        if (!m.b_et1[blk] || !m.b_et2[blk] || !m.b_ax[blk + 1] || !A.bi || !A.bo)
            return RAD_E_INVAL;
        const int64_t R = np;
        /* Three units: snake, conv k7, snake, conv k1 with the residual. */
        for (int r = 0; r < 3; ++r) {
            Model::AEncUnit& U = A.u[r];
            const char* us = nm.ckpt("audio_vae.encoder.block.%d.block.%d.block", blk + 1, r);
            U.a1 = wcopy(b, nm.f("aenc.b%d.u%d.a1", blk, r), nm.ckpt("%s.0.alpha", us), RAD_F32,
                         { u }, acc, gh);
            if (!U.a1) return RAD_E_INVAL;
            U.op_a1 = rw(b, RAD_OP(b, "snake",
                                   RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", u),
                                              RAD_STR("dtype", "f32")),
                                   RAD_WEIGHTS(U.a1)),
                         { m.b_ax[blk] }, { m.b_et1[blk] });
            RAD_ARCH_TRY(map_wnorm(b, nm.f("aenc.b%d.u%d.c1.weight", blk, r),
                                   nm.ckpt("%s.1.weight_g", us), nm.ckpt("%s.1.weight_v", us)));
            U.c1w = decl_w(b, nm.f("aenc.b%d.u%d.c1.weight", blk, r), RAD_F32, { u, u, 7 }, acc,
                           RAD_SHARD_NONE, grp_layer(ga));
            U.c1b = wcopy(b, nm.f("aenc.b%d.u%d.c1.bias", blk, r), nm.ckpt("%s.1.bias", us),
                          RAD_F32, { u }, acc, ga);
            if (!U.c1w || !U.c1b) return RAD_E_INVAL;
            U.op_c1 = rw(b, RAD_OP(b, "conv1d",
                                   RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("cin", u),
                                              RAD_INT("cout", u), RAD_INT("k", 7),
                                              RAD_INT("dil", 2 * r + 1), RAD_STR("act", "none"),
                                              RAD_STR("dtype", "f32")),
                                   RAD_WEIGHTS(U.c1w, U.c1b)),
                         { m.b_et1[blk] }, { m.b_et2[blk] });
            U.a2 = wcopy(b, nm.f("aenc.b%d.u%d.a2", blk, r), nm.ckpt("%s.2.alpha", us), RAD_F32,
                         { u }, acc, gh);
            if (!U.a2) return RAD_E_INVAL;
            U.op_a2 = rw(b, RAD_OP(b, "snake",
                                   RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", u),
                                              RAD_STR("dtype", "f32")),
                                   RAD_WEIGHTS(U.a2)),
                         { m.b_et2[blk] }, { m.b_et1[blk] });
            RAD_ARCH_TRY(map_wnorm(b, nm.f("aenc.b%d.u%d.c2.weight", blk, r),
                                   nm.ckpt("%s.3.weight_g", us), nm.ckpt("%s.3.weight_v", us)));
            U.c2w = decl_w(b, nm.f("aenc.b%d.u%d.c2.weight", blk, r), RAD_F32, { u, u, 1 }, acc,
                           RAD_SHARD_NONE, grp_layer(ga));
            U.c2b = wcopy(b, nm.f("aenc.b%d.u%d.c2.bias", blk, r), nm.ckpt("%s.3.bias", us),
                          RAD_F32, { u }, acc, ga);
            if (!U.c2w || !U.c2b) return RAD_E_INVAL;
            U.op_c2 = rw(b, RAD_OP(b, "conv1d",
                                   RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("cin", u),
                                              RAD_INT("cout", u), RAD_INT("k", 1),
                                              RAD_INT("dil", 1), RAD_STR("act", "none"),
                                              RAD_STR("dtype", "f32")),
                                   RAD_WEIGHTS(U.c2w, U.c2b)),
                         { m.b_et1[blk], m.b_ax[blk] }, { m.b_ax[blk] });
            if (!U.op_a1 || !U.op_c1 || !U.op_a2 || !U.op_c2) return RAD_E_INVAL;
        }
        /* The block's last snake, then the strided convolution in bf16. */
        const char* ds = nm.ckpt("audio_vae.encoder.block.%d.block", blk + 1);
        A.al = wcopy(b, nm.f("aenc.b%d.al", blk), nm.ckpt("%s.3.alpha", ds), RAD_F32, { u }, acc,
                     gh);
        if (!A.al) return RAD_E_INVAL;
        A.op_al = rw(b, RAD_OP(b, "snake",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", u),
                                          RAD_STR("dtype", "f32")),
                               RAD_WEIGHTS(A.al)),
                     { m.b_ax[blk] }, { m.b_et1[blk] });
        A.op_ci = cast_op(b, R, u, "f32", "bf16", m.b_et1[blk], A.bi);
        RAD_ARCH_TRY(map_wnorm(b, nm.f("aenc.d%d.w", blk), nm.ckpt("%s.4.weight_g", ds),
                               nm.ckpt("%s.4.weight_v", ds)));
        A.dnw = decl_w(b, nm.f("aenc.d%d.w", blk), RAD_BF16, { d, u, 2 * s }, acc, RAD_SHARD_NONE,
                       grp_layer(ga));
        RAD_ARCH_TRY(map_copy(b, nm.f("aenc.d%d.b", blk), nm.ckpt("%s.4.bias", ds)));
        A.dnb = decl_w(b, nm.f("aenc.d%d.b", blk), RAD_BF16, { d }, acc, RAD_SHARD_NONE,
                       grp_layer(ga));
        if (!A.dnw || !A.dnb) return RAD_E_INVAL;
        A.op_dn = rw(b, RAD_OP(b, "conv1d",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("cin", u),
                                          RAD_INT("cout", d), RAD_INT("k", 2 * s),
                                          RAD_INT("dil", 1), RAD_INT("stride", s),
                                          RAD_INT("pad", (s + 1) / 2), RAD_STR("act", "none"),
                                          RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(A.dnw, A.dnb)),
                     { A.bi }, { A.bo });
        A.op_co = cast_op(b, no, d, "bf16", "f32", A.bo, m.b_ax[blk + 1]);
        if (!A.op_al || !A.op_ci || !A.op_dn || !A.op_co) return RAD_E_INVAL;
    }

    /* The last snake and convolution, then the attention block. */
    const int64_t T = nn[5];
    const char* f6 = nm.ckpt("audio_vae.encoder.block.6");
    const char* f7 = nm.ckpt("audio_vae.encoder.block.7");
    m.ae_al6 = wcopy(b, nm.f("aenc.al6"), nm.ckpt("%s.alpha", f6), RAD_F32, { ld }, acc, gh);
    RAD_ARCH_TRY(map_wnorm(b, nm.f("aenc.fin.weight"), nm.ckpt("%s.weight_g", f7),
                           nm.ckpt("%s.weight_v", f7)));
    m.ae_finw = decl_w(b, nm.f("aenc.fin.weight"), RAD_F32, { ld, ld, 3 }, acc, RAD_SHARD_NONE,
                       grp_layer(ga));
    m.ae_finb = wcopy(b, nm.f("aenc.fin.bias"), nm.ckpt("%s.bias", f7), RAD_F32, { ld }, acc, ga);
    m.b_a6t = decl_b(b, nm.f("aenc.snk6"), RAD_F32, { T, ld });
    m.b_ax6 = decl_b(b, nm.f("aenc.x6"), RAD_F32, { T, ld });
    if (!m.ae_al6 || !m.ae_finw || !m.ae_finb || !m.b_a6t || !m.b_ax6) return RAD_E_INVAL;
    m.op_ae_snk6 = rw(b, RAD_OP(b, "snake",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", ld),
                                           RAD_STR("dtype", "f32")),
                                RAD_WEIGHTS(m.ae_al6)),
                      { m.b_ax[5] }, { m.b_a6t });
    m.op_ae_fin = rw(b, RAD_OP(b, "conv1d",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("cin", ld),
                                          RAD_INT("cout", ld), RAD_INT("k", 3), RAD_INT("dil", 1),
                                          RAD_STR("act", "none"), RAD_STR("dtype", "f32")),
                               RAD_WEIGHTS(m.ae_finw, m.ae_finb)),
                     { m.b_a6t }, { m.b_ax6 });
    if (!m.op_ae_snk6 || !m.op_ae_fin) return RAD_E_INVAL;

    /* The attention block, bf16: LayerNorms over the 2048-wide stream, qkv (the checkpoint's q,
     * zero k and v biases joined), eight causal heads averaged, pooled to 32 and projected; then
     * the gated MLP. */
    const char* pb = nm.ckpt("audio_vae.pre_block");
    auto pw_ = [&](const char* dn, const char* src, uint32_t dt,
                   std::initializer_list<int64_t> shape) {
        return wcopy(b, nm.f("aenc.pb.%s", dn), nm.ckpt("%s.%s", pb, src), dt, shape, acc, ga);
    };
    m.p_ln1w = pw_("ln1w", "norm1.weight", RAD_BF16, { ld });
    m.p_ln1b = pw_("ln1b", "norm1.bias", RAD_BF16, { ld });
    m.p_ln3w = pw_("ln3w", "norm3.weight", RAD_BF16, { ld });
    m.p_ln3b = pw_("ln3b", "norm3.bias", RAD_BF16, { ld });
    m.p_ln2w = pw_("ln2w", "norm2.weight", RAD_BF16, { kAudioC });
    m.p_ln2b = pw_("ln2b", "norm2.bias", RAD_BF16, { kAudioC });
    m.p_mlw = pw_("mlnw", "mlp.norm.weight", RAD_BF16, { kAudioC });
    m.p_mlb = pw_("mlnb", "mlp.norm.bias", RAD_BF16, { kAudioC });
    m.p_prw = pw_("prw", "proj.weight", RAD_BF16, { kAudioC, ld });
    m.p_prb = pw_("prb", "proj.bias", RAD_BF16, { kAudioC });
    m.p_apw = pw_("apw", "attn.proj.weight", RAD_BF16, { kAudioC, kAudioC });
    m.p_apb = pw_("apb", "attn.proj.bias", RAD_BF16, { kAudioC });
    m.p_w2w = pw_("w2w", "mlp.w2.weight", RAD_BF16, { kAudioC, 2 * kAudioC });
    m.p_w2b = pw_("w2b", "mlp.w2.bias", RAD_BF16, { kAudioC });
    if (!m.p_ln1w || !m.p_ln1b || !m.p_ln3w || !m.p_ln3b || !m.p_ln2w || !m.p_ln2b ||
        !m.p_mlw || !m.p_mlb || !m.p_prw || !m.p_prb || !m.p_apw || !m.p_apb || !m.p_w2w ||
        !m.p_w2b)
        return RAD_E_INVAL;
    RAD_ARCH_TRY(map_copy(b, nm.f("aenc.pb.qkvw"), nm.ckpt("%s.attn.qkv.weight", pb)));
    m.p_qkvw = decl_w(b, nm.f("aenc.pb.qkvw"), RAD_BF16, { 3 * ld, ld }, acc, RAD_SHARD_NONE,
                      grp_layer(ga));
    RAD_ARCH_TRY(map_concat(b, nm.f("aenc.pb.qkvb"),
                            { nm.ckpt("%s.attn.q_bias", pb), nm.ckpt("%s.attn.zero_k_bias", pb),
                              nm.ckpt("%s.attn.v_bias", pb) }));
    m.p_qkvb = decl_w(b, nm.f("aenc.pb.qkvb"), RAD_BF16, { 3 * ld }, acc, RAD_SHARD_NONE,
                      grp_layer(ga));
    /* w0 (the gate) then w1, as gelu_mul splits them. */
    RAD_ARCH_TRY(map_concat(b, nm.f("aenc.pb.guw"),
                            { nm.ckpt("%s.mlp.w0.weight", pb), nm.ckpt("%s.mlp.w1.weight", pb) }));
    m.p_guw = decl_w(b, nm.f("aenc.pb.guw"), RAD_BF16, { 2 * kAudioC * 2, kAudioC }, acc,
                     RAD_SHARD_NONE, grp_layer(ga));
    RAD_ARCH_TRY(map_concat(b, nm.f("aenc.pb.gub"),
                            { nm.ckpt("%s.mlp.w0.bias", pb), nm.ckpt("%s.mlp.w1.bias", pb) }));
    m.p_gub = decl_w(b, nm.f("aenc.pb.gub"), RAD_BF16, { 2 * kAudioC * 2 }, acc, RAD_SHARD_NONE,
                     grp_layer(ga));
    if (!m.p_qkvw || !m.p_qkvb || !m.p_guw || !m.p_gub) return RAD_E_INVAL;

    m.b_apb = decl_b(b, nm.f("aenc.pb"), RAD_BF16, { T, ld });
    m.b_aln3 = decl_b(b, nm.f("aenc.ln3"), RAD_BF16, { T, ld });
    m.b_aqkv = decl_b(b, nm.f("aenc.qkv"), RAD_BF16, { T, 3 * ld });
    m.b_aatt = decl_b(b, nm.f("aenc.att"), RAD_BF16, { T, hd });
    m.b_ahsum = decl_b(b, nm.f("aenc.hsum"), RAD_BF16, { T, hd });
    m.b_apool = decl_b(b, nm.f("aenc.pool"), RAD_BF16, { T, kAudioC });
    m.b_eh = decl_b(b, nm.f("aenc.h"), RAD_BF16, { T, kAudioC });
    m.b_a32a = decl_b(b, nm.f("aenc.r32a"), RAD_BF16, { T, kAudioC });
    m.b_a32b = decl_b(b, nm.f("aenc.r32b"), RAD_BF16, { T, kAudioC });
    m.b_a32c = decl_b(b, nm.f("aenc.r32c"), RAD_BF16, { T, kAudioC });
    m.b_a32d = decl_b(b, nm.f("aenc.r32d"), RAD_BF16, { T, kAudioC });
    m.b_agu = decl_b(b, nm.f("aenc.gu"), RAD_BF16, { T, 4 * kAudioC });
    m.b_agu64 = decl_b(b, nm.f("aenc.gu64"), RAD_BF16, { T, 2 * kAudioC });
    m.b_ax2 = decl_b(b, nm.f("aenc.xfin"), RAD_BF16, { T, kAudioC });
    m.b_amb = decl_b(b, nm.f("aenc.mono"), RAD_F32, { T, kAudioC });
    m.b_amb2 = decl_b(b, nm.f("aenc.mp"), RAD_F32, { T, kAudioC });
    m.op_ae_cast6 = cast_op(b, T, ld, "f32", "bf16", m.b_ax6, m.b_apb);
    if (!m.op_ae_cast6) return RAD_E_INVAL;
    auto ln = [&](rad_weight w, rad_weight b2, int64_t n, rad_buf x, rad_buf y) -> rad_op {
        if (!w || !b2) return 0;
        return rw(b, RAD_OP(b, "layernorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n),
                                       RAD_F64("eps", 1e-5), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w, b2)),
                  { x }, { y });
    };
    auto gemm = [&](rad_weight w, rad_weight b2, int64_t N, int64_t K, rad_buf x,
                    rad_buf y) -> rad_op {
        if (!w || !b2) return 0;
        return rw(b, RAD_OP(b, "gemm_nt_bias",
                            RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", N), RAD_INT("K", K),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(w, b2)),
                  { x }, { y });
    };
    m.p_op_ln3 = ln(m.p_ln3w, m.p_ln3b, ld, m.b_apb, m.b_aln3);
    m.p_op_qkv = gemm(m.p_qkvw, m.p_qkvb, 3 * ld, ld, m.b_aln3, m.b_aqkv);
    m.p_op_attn = rw(b, RAD_OP(b, "attn_dense",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                          RAD_INT("gqa", 1), RAD_INT("causal", 1),
                                          RAD_STR("dtype", "bf16"),
                                          RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                          RAD_INT("max_seqlen", T)),
                               RAD_NOWEIGHTS),
                     /* The first head writes b_ahsum, the others b_aatt: both are outputs. */
                     { m.b_aqkv }, { m.b_aatt, m.b_ahsum });
    m.p_op_pool = rw(b, RAD_OP(b, "pool_cols",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", hd),
                                          RAD_INT("k", hd / kAudioC), RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_ahsum }, { m.b_apool });
    m.p_op_aproj = gemm(m.p_apw, m.p_apb, kAudioC, kAudioC, m.b_apool, m.b_a32a);
    m.p_op_ln1 = ln(m.p_ln1w, m.p_ln1b, ld, m.b_apb, m.b_aln3);
    m.p_op_proj = gemm(m.p_prw, m.p_prb, kAudioC, ld, m.b_aln3, m.b_eh);
    m.p_op_hadd = add_op(b, T, kAudioC, "bf16", m.b_eh, m.b_a32a, m.b_a32b);
    m.p_op_ln2 = ln(m.p_ln2w, m.p_ln2b, kAudioC, m.b_a32b, m.b_a32c);
    m.p_op_mln = ln(m.p_mlw, m.p_mlb, kAudioC, m.b_a32c, m.b_a32d);
    m.p_op_gu = gemm(m.p_guw, m.p_gub, 4 * kAudioC, kAudioC, m.b_a32d, m.b_agu);
    m.p_op_act = rw(b, RAD_OP(b, "gelu_mul",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", 2 * kAudioC),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_agu }, { m.b_agu64 });
    m.p_op_w2 = gemm(m.p_w2w, m.p_w2b, kAudioC, 2 * kAudioC, m.b_agu64, m.b_a32c);
    m.p_op_xadd = add_op(b, T, kAudioC, "bf16", m.b_a32b, m.b_a32c, m.b_ax2);
    /* The heads' mean: six adds into b_ahsum, and the seventh divides by eight. */
    m.p_op_hs = add_op(b, T, hd, "bf16", m.b_ahsum, m.b_aatt, m.b_ahsum);
    m.p_op_hsd = add_div_op(b, T, hd, "bf16", m.b_ahsum, m.b_aatt, m.b_ahsum, 8.0);
    if (!m.p_op_ln3 || !m.p_op_qkv || !m.p_op_attn || !m.p_op_pool || !m.p_op_aproj ||
        !m.p_op_ln1 || !m.p_op_proj || !m.p_op_hadd || !m.p_op_ln2 || !m.p_op_mln || !m.p_op_gu ||
        !m.p_op_act || !m.p_op_w2 || !m.p_op_xadd || !m.p_op_hs || !m.p_op_hsd)
        return RAD_E_INVAL;
    m.op_ae_castx = cast_op(b, T, kAudioC, "bf16", "f32", m.b_ax2, m.b_amb);
    if (!m.op_ae_castx) return RAD_E_INVAL;
    /* Mapped by the presence check above. */
    m.ae_mpw = decl_w(b, nm.f("aenc.mp.weight"), RAD_F32, { kAudioC, kAudioC, 1 }, acc,
                      RAD_SHARD_NONE, grp_layer(ga));
    m.ae_mpb = wcopy(b, nm.f("aenc.mp.bias"), nm.ckpt("audio_vae.mean_proj.bias"), RAD_F32,
                     { kAudioC }, acc, ga);
    if (!m.ae_mpw || !m.ae_mpb) return RAD_E_INVAL;
    m.op_ae_mp = rw(b, RAD_OP(b, "conv1d",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("cin", kAudioC),
                                         RAD_INT("cout", kAudioC), RAD_INT("k", 1),
                                         RAD_INT("dil", 1), RAD_STR("act", "none"),
                                         RAD_STR("dtype", "f32")),
                              RAD_WEIGHTS(m.ae_mpw, m.ae_mpb)),
                    { m.b_amb }, { m.b_amb2 });
    m.b_enca = decl_b(b, nm.f("lane.enca"), RAD_F32,
                      { (int64_t)kEncSlotsA * kEncSlotRowsA, kAudioC }, RAD_BUF_PERSIST);
    m.op_ae_norm = rw(b, RAD_OP(b, "chan_affine",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", kAudioC),
                                           RAD_STR("mul", m.ae_norm_mul.c_str()),
                                           RAD_STR("add", m.ae_norm_add.c_str()),
                                           RAD_STR("mode", "fma"), RAD_INT("round", 0),
                                           RAD_STR("dtype", "f32")),
                                RAD_NOWEIGHTS),
                      { m.b_amb2 }, { m.b_enca });
    if (!m.op_ae_mp || !m.op_ae_norm || !m.b_enca) return RAD_E_INVAL;
    return RAD_OK;
}

/* ================================================================== declare */
static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.max_area = video_area(ctx->gen_max_pixels);
    if (m.max_area < kMinSide * kMinSide) {
        fprintf(stderr, "radiance: minimaxh3: the video pixel limit must allow a %dx%d frame\n",
                kMinSide, kMinSide);
        return RAD_E_INVAL;
    }
    m.nm = Names(ctx->scope);
    Names& nm = m.nm;
    RAD_ARCH_TRY(check_format(meta));
    RAD_ARCH_TRY(geom_from(m.gt, meta, ctx, "bf16", RAD_BF16));
    if (m.gt.n_layer <= kTextLayers) {
        fprintf(stderr, "radiance: minimaxh3: the conditioning is the text encoder's state after "
                        "layer %d, and it has %lld layers\n", kTextLayers, (long long)m.gt.n_layer);
        return RAD_E_INVAL;
    }
    m.gt.n_layer = kTextLayers;
    m.gt.max_tok = kMaxText;
    m.W = m.gt.world;

    /* ---- the DiT's geometry, off the transformer's own config */
    Geom& gd = m.gd;
    gd = m.gt;
    gd.max_tok = ctx->max_tok;
    gd.n_layer    = rad_meta_geti(meta, "transformer.num_layers", 50);
    gd.head_dim   = rad_meta_geti(meta, "transformer.attention_head_dim", 128);
    gd.n_head_all = rad_meta_geti(meta, "transformer.num_attention_heads", 56);
    gd.n_head_kv_all = gd.n_head_all;
    gd.n_embd     = rad_meta_geti(meta, "transformer.hidden_size", 5376);
    gd.n_ff_all   = rad_meta_geti(meta, "transformer.ffn_dim", 14336);
    gd.eps        = (float)rad_meta_getf(meta, "transformer.norm_eps", 1e-5);
    gd.rope_mc    = false;
    RAD_ARCH_TRY(divide_or_fail(gd.n_head_all, gd.world, "transformer.num_attention_heads",
                                &gd.n_head));
    gd.n_head_kv = gd.n_head;
    gd.kv_off = 0;
    RAD_ARCH_TRY(divide_or_fail(gd.n_ff_all, gd.world, "the DiT's feed-forward width", &gd.n_ff));
    m.D = gd.n_embd;
    /* A rank holds 6 / world of a modality's six vectors, which keeps each [shift | scale] pair and
     * its gate on one rank. */
    if (6 % m.W || (6 * m.D / m.W) % RAD_SCALE_GROUP) {
        fprintf(stderr, "radiance: minimaxh3: %d ranks do not split the six modulation vectors "
                        "into whole fp8 blocks; use 1 or 2\n", (int)m.W);
        return RAD_E_UNSUPPORTED;
    }
    m.Ws = 6 * m.D / m.W;

    /* A checkpoint distilled for few steps with video sparse attention says so in its inference
     * settings (fastvideo_inference.json): the attention, its sparsity and tile, the timesteps it
     * was trained on (a 1000-step clock) and the shifts it was trained with. */
    const char* backend = rad_meta_gets(meta, "fastvideo_inference.attention_backend", "");
    if (!std::strcmp(backend, "VIDEO_SPARSE_ATTN_H3")) {
        m.vsa = true;
        m.vsa_gated = rad_meta_geti(meta, "fastvideo_inference.vsa_gate_compress", 1) != 0;
        m.vsa_min_rows = rad_meta_geti(meta, "fastvideo_inference.vsa_min_tokens", 0);
        m.vsa_dense_layers = rad_meta_geti(meta, "fastvideo_inference.vsa_dense_prefix_layers", 0);
        if (m.vsa_min_rows < 0 || m.vsa_dense_layers < 0 || m.vsa_dense_layers >= gd.n_layer ||
            (m.vsa_gated && (m.vsa_min_rows || m.vsa_dense_layers))) {
            fprintf(stderr, "radiance: minimaxh3: invalid video sparse attention cutoffs\n");
            return RAD_E_UNSUPPORTED;
        }
        m.vsa_sparsity = rad_meta_getf(meta, "fastvideo_inference.vsa_sparsity", -1.0);
        const long long tile = rad_meta_geti(meta, "fastvideo_inference.vsa_tile_size", 0);
        if (tile != kVsaRows || !(m.vsa_sparsity >= 0.0 && m.vsa_sparsity < 1.0)) {
            fprintf(stderr, "radiance: minimaxh3: video sparse attention at %lld-row tiles and "
                            "sparsity %g; this plugin runs %d-row (4 x 4 x 4) tiles at a sparsity "
                            "in [0, 1)\n", tile, m.vsa_sparsity, (int)kVsaRows);
            return RAD_E_UNSUPPORTED;
        }
        vsa_bounds(m.vsa_sparsity, m.max_area, gd.max_tok, &m.vsa_tiles, &m.vsa_k);
    } else if (*backend && std::strcmp(backend, "FLASH_ATTN")) {
        fprintf(stderr, "radiance: minimaxh3: the checkpoint was made for the attention '%s'; this "
                        "plugin runs dense attention and VIDEO_SPARSE_ATTN_H3\n", backend);
        return RAD_E_UNSUPPORTED;
    }
    for (double t : meta_list(meta, "fastvideo_inference.dmd_denoising_steps"))
        m.rungs.push_back(t / 1000.0);
    if (!m.rungs.empty()) {
        const long long fw = rad_meta_geti(meta, "fastvideo_inference.transformer_forwards",
                                           (long long)m.rungs.size());
        if (fw != (long long)m.rungs.size() || (int64_t)m.rungs.size() > kMaxSteps) {
            fprintf(stderr, "radiance: minimaxh3: a trained schedule of %zu timesteps for %lld "
                            "evaluations; at most %d, one a timestep\n", m.rungs.size(), fw,
                    (int)kMaxSteps);
            return RAD_E_UNSUPPORTED;
        }
        m.S = (int64_t)m.rungs.size();
    }
    for (const char* k : { "video", "audio" }) {
        const double want = rad_meta_getf(meta, m.nm.f("fastvideo_inference.%s_scheduler_shift", k),
                                          0.0);
        const double have = rad_meta_getf(meta, !std::strcmp(k, "video") ? "scheduler.shift"
                                                                          : "audio_scheduler.shift",
                                          !std::strcmp(k, "video") ? 12.0 : 3.0);
        if (want > 0.0 && want != have) {
            fprintf(stderr, "radiance: minimaxh3: the checkpoint was trained with a %s shift of %g "
                            "and its scheduler says %g\n", k, want, have);
            return RAD_E_INVAL;
        }
    }
    m.gr = gd;
    m.gr.max_tok = kMaxText;
    m.maxV = max_video_rows(m.max_area);
    m.maxA = kAudioCh * max_audio();
    m.tiles = max_tiles(m.max_area);
    m.tpr = (m.tiles + m.W - 1) / m.W;
    if (gd.n_layer >= kGrpAda - kGrpDit - 1 || gd.n_layer > kGrpVae - kGrpAda ||
        m.gt.n_layer >= kGrpRefiner) {
        fprintf(stderr, "radiance: minimaxh3: the networks are too deep for the weight groups\n");
        return RAD_E_UNSUPPORTED;
    }
    /* The driver checks the full decode row count, even though each rank issues only its share. */
    const int64_t decode_max_rows = m.tiles * kPeriod;
    const int64_t min_step_rows = max_step_rows(m.max_area, kMinChunks);
    if (ctx->max_tok < min_step_rows || ctx->max_tok < decode_max_rows) {
        fprintf(stderr, "radiance: minimaxh3: a pass carries %lld rows; the shortest video's step "
                        "needs up to %lld and a decode up to %lld\n", (long long)ctx->max_tok,
                (long long)min_step_rows, (long long)decode_max_rows);
        return RAD_E_INVAL;
    }
    /* An explicit row cap can shorten the video while keeping enough room for every canvas.
     * Keep the original bounds for codecs and reference buffers. References consume the
     * remaining step rows and the driver checks the complete packed sequence per request. */
    int64_t max_chunks = kMaxChunks;
    while (max_step_rows(m.max_area, max_chunks) > ctx->max_tok) --max_chunks;

    /* The vision tower, when the checkpoint or container has its weights: every MiniMax-H3 config
     * names one. */
    m.media = VisionTower::present(meta) && ctx->max_enc_patches > 0;
    if (m.media) {
        if (m.tv.declare(b, meta, ctx, nm, m.gt.n_embd, kGrpTower, true, true) != RAD_OK)
            return RAD_E_UNSUPPORTED;
        m.media = m.tv.on;
    }

    /* ---- in issue order: the conditioning pass, the step, the soundtrack, the frames */
    m.b_text = decl_b(b, nm.f("lane.text"), RAD_BF16, { (int64_t)kMaxLanes * kMaxText, m.D },
                      RAD_BUF_PERSIST);
    RAD_ARCH_TRY(declare_lora_names(b, m));
    RAD_ARCH_TRY(declare_te(b, meta, m));
    RAD_ARCH_TRY(declare_refiner(b, meta, m));
    RAD_ARCH_TRY(declare_tables(b, meta, m));
    RAD_ARCH_TRY(declare_dit(b, meta, m));
    if (m.dit[0].qkv.bf16) {
        fprintf(stderr, "radiance: minimaxh3: the DiT's linears are bf16; this plugin serves the "
                        "int8 container (data/recipes/minimax-h3-i8.recipe)\n");
        return RAD_E_UNSUPPORTED;
    }
    RAD_ARCH_TRY(declare_ups(b, meta, m));
    RAD_ARCH_TRY(declare_audio(b, meta, m));
    RAD_ARCH_TRY(declare_video(b, meta, m));
    RAD_ARCH_TRY(declare_venc(b, meta, m));
    RAD_ARCH_TRY(declare_aenc(b, meta, m));

    if (!ctx->shape_probe) {
        RadVideoGenDecl d{};
        /* A distilled release is served under its own id's last part, lower case. */
        std::string name = "minimax-h3";
        if (const char* id = rad_meta_gets(meta, "fastvideo_inference.model_id", nullptr)) {
            const char* last = std::strrchr(id, '/');
            name = last ? last + 1 : id;
            for (char& ch : name) ch = (char)std::tolower((unsigned char)ch);
        }
        d.name = name.c_str();
        d.tmpl = "{prompt}";
        d.max_text = kMaxText;
        d.fps_num = kFps;
        d.fps_den = 1;
        d.frames_per_chunk = kFpc;
        d.frames_tail = kFtail;
        d.latents_per_chunk = kLpc;
        d.latents_tail = kLtail;
        d.dec_latents = kDecLat;
        d.min_chunks = kMinChunks;
        d.max_chunks = max_chunks;
        d.latent_scale = kScale;
        d.latent_channels = kC;
        d.patch = kPatch;
        d.size_multiple = kMult;
        d.min_side = kMinSide;
        d.max_area = m.max_area;
        d.max_aspect = kMaxAspect;
        const double shrink = std::sqrt(static_cast<double>(m.max_area) / static_cast<double>(kMaxArea));
        const int64_t scaled_h = static_cast<int64_t>(static_cast<double>(kDefH) * shrink);
        const int64_t scaled_w = static_cast<int64_t>(static_cast<double>(kDefW) * shrink);
        d.default_height = std::max<int64_t>(kMinSide, (scaled_h / kMult) * kMult);
        const int64_t fit_w = std::min(scaled_w, m.max_area / d.default_height);
        d.default_width = std::max<int64_t>(kMinSide, (fit_w / kMult) * kMult);
        d.default_chunks = kDefChunks;
        d.sample_rate = kRate;
        d.audio_rate = kAudioRate;
        d.audio_channels = kAudioCh;
        d.audio_latent_channels = kAudioC;
        d.default_steps = kDefSteps;
        d.max_steps = kMaxSteps + 1;
        if (!m.rungs.empty()) {
            d.n_rungs = (int32_t)m.rungs.size();
            d.rungs = m.rungs.data();
            d.default_steps = d.max_steps = d.n_rungs + 1;
        }
        d.shift_video = rad_meta_getf(meta, "scheduler.shift", 12.0);
        d.shift_audio = rad_meta_getf(meta, "audio_scheduler.shift", 3.0);
        d.max_lanes = kMaxLanes;
        d.frames_out = m.b_frames;
        d.audio_out = m.b_audio;
        if (m.ups) {
            d.upscale_max = m.ups_max;
            d.upscale_refine = m.rungs.empty() ? kUpsRefineBase : kUpsRefineDistilled;
        }
        /* Keyframes and reference media as far as the container has the encoders, and the
         * vision tower for pictures. The counts are ComfyUI's ref2va workflow's. */
        if (m.venc) {
            d.max_ref_images = m.media ? 9 : 0;
            d.max_ref_videos = 3;
            d.max_enc_frames = kEncMaxFrames;
            d.max_enc_side = kEncMaxSide;
            d.enc_ratio = kVaeRatio;
        }
        if (m.aenc) {
            d.max_ref_audios = 3;
            d.max_enc_samples = kEncMaxSamples;
        }
        if (m.venc || m.aenc) d.max_mod_cells = kMaxSegs;
        RAD_ARCH_TRY(rad_declare_video_generator(b, &d));
    }

    rad_note(b, "rank %d/%d: MiniMax-H3 t2va -- text encoder %lld layers (%lld heads a rank), "
                "refiner 2 and DiT %lld blocks (%lld heads, ff %lld a rank), modulation %lld of "
                "6 x %lld a rank, video decoder %lld tiles at most (%lld a rank)",
             m.gt.rank, m.gt.world, (long long)m.gt.n_layer, (long long)m.gt.n_head,
             (long long)gd.n_layer, (long long)gd.n_head, (long long)gd.n_ff, (long long)m.Ws,
             (long long)m.D, (long long)m.tiles, (long long)m.tpr);
    if (m.ups)
        rad_note(b, "rank %d/%d: latent upscaler LatentResizer3D -- %zu + %zu blocks of %lld "
                    "channels, factors up to %g, whole on every rank", m.gt.rank, m.gt.world,
                 m.u_in.size(), m.u_out.size(), (long long)m.uC, m.ups_max);
    return RAD_OK;
}

/* ================================================================== step */
/* RadVideoGenBatch::ref_tab's reference kinds are the pack's. */
static_assert((int)RAD_VREF_PICTURE == (int)kR2VRefImage &&
              (int)RAD_VREF_AUDIO == (int)kR2VRefAudioOnly &&
              (int)RAD_VREF_VIDEO == (int)kR2VRefVideo, "ref_tab kinds");

/* The packed sequence of a request with references (RadVideoGenBatch::ref_tab), as
 * rad_arch_video_pack laid it out for the driver; false for a table the pack refuses. */
static bool ref_pack_of(const RadVideoGeom* g, const int32_t* ref_tab, Ref2VAPack* p,
                        bool positions) {
    const int32_t nr = ref_tab[0], nk = ref_tab[1];
    if (nr < 0 || nk < 0 || nr > 64 || nk > 3) return false;
    std::vector<Ref2VARef> refs;
    std::vector<Ref2VAKeyframe> keys;
    const int32_t* w = ref_tab + 2;
    for (int32_t i = 0; i < nr; ++i, w += 6) {
        if (w[0] < RAD_VREF_PICTURE || w[0] > RAD_VREF_VIDEO || w[1] < 1) return false;
        refs.push_back(Ref2VARef{ w[0], w[2], w[3], w[4], w[5] });
    }
    /* The keyframes' slots: two pictures, the first and last frames', and one soundtrack, the
     * song guide's. */
    int32_t pics = 0, songs = 0;
    for (int32_t i = 0; i < nk; ++i, w += 3) {
        pics += w[1] > 0;
        songs += w[2] > 0;
        if (pics > 2 || songs > 1) return false;
        keys.push_back(Ref2VAKeyframe{ w[0], w[1], w[2] });
    }
    const Ref2VAGeom pg{ (int)g->lat_frames, (int)g->lat_h, (int)g->lat_w, (int)g->n_audio };
    return ref2va_build((int)g->n_text, pg, keys, refs, p, positions) == 0;
}

/* The prompt's vision pads, the table's tail after the keyframes: a count, then {start, length}
 * of each picture's tokens. They read the video's projection, as ComfyUI tags them. */
static std::vector<Ref2VATagRun> ref_pads_of(const int32_t* ref_tab) {
    const int32_t* w = ref_tab + 2 + 6 * (int64_t)ref_tab[0] + 3 * (int64_t)ref_tab[1];
    std::vector<Ref2VATagRun> runs;
    for (int32_t i = 0, n = w[0]; i < n; ++i)
        runs.push_back({ w[1 + 2 * i], (int64_t)w[1 + 2 * i] + w[2 + 2 * i], 0 });
    return runs;
}

/* Where each reference block's latents are, in the order the pack lays the blocks down: the
 * keyframes, then the references, a video's soundtrack before the video. `audio` picks b_enca
 * over b_evc; `row` is -1 for a slot no encode pass wrote. */
struct RefPart { int audio; int64_t row; };
static void ref_parts(const int32_t* ref_tab, std::vector<RefPart>* out) {
    const int32_t nr = ref_tab[0], nk = ref_tab[1];
    const int32_t* k = ref_tab + 2 + 6 * (int64_t)nr;
    for (int32_t i = 0, pic = 0; i < nk; ++i, k += 3) {
        if (k[1] > 0) out->push_back({ 0, venc_slot_row(RAD_ENC_KEYFRAME, ++pic) });
        if (k[2] > 0) out->push_back({ 1, aenc_slot_row(RAD_ENC_GUIDE, 1) });
    }
    const int32_t* w = ref_tab + 2;
    for (int32_t i = 0; i < nr; ++i, w += 6) {
        const int32_t kind = w[0], slot = w[1];
        if (kind == RAD_VREF_AUDIO) {
            out->push_back({ 1, aenc_slot_row(RAD_ENC_AUDIO, slot) });
        } else if (kind == RAD_VREF_PICTURE) {
            out->push_back({ 0, venc_slot_row(RAD_ENC_IMAGE, slot) });
        } else {
            if (w[5] > 0) out->push_back({ 1, aenc_slot_row(RAD_ENC_VAUDIO, slot) });
            out->push_back({ 0, venc_slot_row(RAD_ENC_VIDEO, slot) });
        }
    }
}

/* The conditioning pass: the text encoder, context_embedder and the refiner into the lane's prompt
 * rows, then every step's modulation into the lane's tables. */
static void step_cond(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    const Geom& g = m.gt;
    const int64_t T = batch->n_tok, lane = V->lane, S = V->n_steps;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    const int64_t nmm = batch->n_mm_rows;
    if (m.media && batch->enc_n_patch > 0) m.tv.step(c, batch);
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.b_tx, T));
    if (m.media && nmm > 0)
        RAD_ISSUE_N(c, m.op_mm, nmm, brows(m.tv.b_out, nmm), praw(batch->mm_rows, RAD_I32, nmm),
                    brows(m.b_tx, T));
    /* A prompt with pictures has M-RoPE positions. */
    const RadOperand pos = batch->rope_pos ? praw2(batch->rope_pos, RAD_I32, 3, T)
                                           : praw(batch->positions, RAD_I32, T);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, 2);
    for (size_t li = 0; li < m.te.size(); ++li) {
        const TeLayer& L = m.te[li];
        RAD_ISSUE(c, L.op_n1, brows(m.b_tx, T), RAD_W(L.w_n1), brows(m.a_th.x, T));
        L.q_h.step(c, T);
        L.qkv.step(c, m.a_th, m.b_tkv, T);
        RAD_ISSUE(c, batch->rope_pos ? L.op_qkr_m : L.op_qkr, brows(m.b_tkv, T), pos,
                  RAD_W(L.w_qn), RAD_W(L.w_kn));
        RAD_ISSUE_N(c, L.op_attn, T, bcol(m.b_tkv, 0, q_dim, T), bcol(m.b_tkv, q_dim, kv_dim, T),
                    bcol(m.b_tkv, q_dim + kv_dim, kv_dim, T), cu, brows(m.a_tattn.x, T));
        L.q_attn.step(c, T);
        L.o.step(c, m.a_tattn, m.b_th2, T);
        if (L.op_ar1) RAD_ISSUE_N(c, L.op_ar1, T * g.n_embd, brows(m.b_th2, T), RAD_NONE);
        RAD_ISSUE(c, L.op_add1, brows(m.b_tx, T), brows(m.b_th2, T), brows(m.b_tx, T));
        RAD_ISSUE(c, L.op_n2, brows(m.b_tx, T), RAD_W(L.w_n2), brows(m.a_th.x, T));
        L.q_h2.step(c, T);
        L.gate_up.step(c, m.a_th, m.b_tgu, T);
        L.gq.step(c, m.b_tgu, T);
        L.down.step(c, m.a_tffn, m.b_th2, T);
        if (L.op_ar2) RAD_ISSUE_N(c, L.op_ar2, T * g.n_embd, brows(m.b_th2, T), RAD_NONE);
        RAD_ISSUE(c, L.op_add2, brows(m.b_tx, T), brows(m.b_th2, T), brows(m.b_tx, T));
        /* Deepstack: tap l joins the pictures' rows after layer l. */
        if (m.media && nmm > 0 && (int)li < m.tv.n_ds)
            RAD_ISSUE_N(c, m.op_ds[li], nmm, brows(m.tv.b_ds[li], nmm),
                        praw(batch->mm_rows, RAD_I32, nmm), brows(m.b_tx, T));
    }

    const Geom& r = m.gr;
    const int64_t D = m.D, qd = r.q_dim();
    RAD_ISSUE_N(c, m.op_ce, T, brows(m.b_tx, T), RAD_W(m.w_ce), RAD_W(m.b_ce), RAD_NONE,
                brows(m.b_rh, T));
    for (const RefBlock& R : m.rb) {
        RAD_ISSUE_N(c, R.op_n1, T, brows(m.b_rh, T), RAD_W(R.w_n1), brows(m.a_rh.x, T));
        R.q_h.step(c, T);
        R.qkv.step(c, m.a_rh, m.b_rqkv, T);
        RAD_ISSUE_N(c, R.op_qkn, T, brows(m.b_rqkv, T), RAD_NONE, RAD_W(R.w_qn), RAD_W(R.w_kn));
        RAD_ISSUE_N(c, R.op_attn, T, bcol(m.b_rqkv, 0, qd, T), bcol(m.b_rqkv, qd, qd, T),
                    bcol(m.b_rqkv, 2 * qd, qd, T), cu, brows(m.a_rattn.x, T));
        R.q_attn.step(c, T);
        R.o.step(c, m.a_rattn, m.b_rao, T);
        if (R.op_ar1) RAD_ISSUE_N(c, R.op_ar1, T * D, brows(m.b_rao, T), RAD_NONE);
        RAD_ISSUE_N(c, R.op_add1, T, brows(m.b_rh, T), brows(m.b_rao, T), brows(m.b_rh, T));
        RAD_ISSUE_N(c, R.op_n2, T, brows(m.b_rh, T), RAD_W(R.w_n2), brows(m.a_rh.x, T));
        R.q_h2.step(c, T);
        R.gate_up.step(c, m.a_rh, m.b_rgu, T);
        R.gq.step(c, m.b_rgu, T);
        R.down.step(c, m.a_rffn, m.b_rao, T);
        if (R.op_ar2) RAD_ISSUE_N(c, R.op_ar2, T * D, brows(m.b_rao, T), RAD_NONE);
        RAD_ISSUE_N(c, R.op_add2, T, brows(m.b_rh, T), brows(m.b_rao, T), brows(m.b_rh, T));
    }
    RAD_ISSUE_N(c, m.op_fn, T, brows(m.b_rh, T), RAD_W(m.w_fn),
                brow_slice(m.b_text, lane * kMaxText, T, D));

    /* ---- every step's timestep embedding, the four blocks of declare_tables. At its own rows:
     * QuantAct::step bands by the rows it is issued for. */
    const int64_t fd = 256, td = m.temb_w, TT = 4 * S;
    if (m.curve > 0) {
        /* The table is operand 2, after the output. */
        RAD_ISSUE_N(c, m.op_temb, S, praw(V->t_video, RAD_F32, S),
                    brow_slice(m.b_temb, 0, S, td), RAD_W(m.w_ttable));
        RAD_ISSUE_N(c, m.op_temb, S, praw(V->t_audio, RAD_F32, S),
                    brow_slice(m.b_temb, S, S, td), RAD_W(m.w_ttable));
        RAD_ISSUE_N(c, m.op_temb_pv, S, praw(V->t_video, RAD_F32, S),
                    brow_slice(m.b_temb, 2 * S, S, td), RAD_W(m.w_ttable));
        RAD_ISSUE_N(c, m.op_temb_pa, S, praw(V->t_audio, RAD_F32, S),
                    brow_slice(m.b_temb, 3 * S, S, td), RAD_W(m.w_ttable));
    } else {
        RAD_ISSUE_N(c, m.op_temb, S, praw(V->t_video, RAD_F32, S), brow_slice(m.b_tproj, 0, S, fd));
        RAD_ISSUE_N(c, m.op_temb, S, praw(V->t_audio, RAD_F32, S), brow_slice(m.b_tproj, S, S, fd));
        RAD_ISSUE_N(c, m.op_temb_pv, S, praw(V->t_video, RAD_F32, S),
                    brow_slice(m.b_tproj, 2 * S, S, fd));
        RAD_ISSUE_N(c, m.op_temb_pa, S, praw(V->t_audio, RAD_F32, S),
                    brow_slice(m.b_tproj, 3 * S, S, fd));
        RAD_ISSUE_N(c, m.op_t1, TT, brows(m.b_tproj, TT), RAD_W(m.w_t1), RAD_W(m.b_t1), RAD_NONE,
                    brows(m.b_te1, TT));
        RAD_ISSUE_N(c, m.op_ts, TT, brows(m.b_te1, TT), brows(m.b_te1s, TT));
        RAD_ISSUE_N(c, m.op_t2, TT, brows(m.b_te1s, TT), RAD_W(m.w_t2), RAD_W(m.b_t2), RAD_NONE,
                    brows(m.b_temb, TT));
        RAD_ISSUE_N(c, m.op_ts2, TT, brows(m.b_temb, TT), brows(m.a_temb.x, TT));
        RAD_ISSUE_N(c, m.q_temb.op, TT, brows(m.a_temb.x, TT), brows(m.a_temb.cq(), TT),
                    brows(m.a_temb.cs(), TT));
    }

    /* ---- the modulation table: block l, cell e from its modality's projection of its
     * timestep's rows, at the table's column block (l, e). A request with references has a cell
     * for each distinct modulation its groups read (rad_ref2va_pack.h); every step's row is laid
     * at the table's full width, kMaxSegs cells a block. */
    const int64_t L = m.gd.n_layer, base = lane * m.S;
    std::vector<std::pair<int, int64_t>> cell;
    if (V->ref_tab) {
        Ref2VAPack p;
        Ref2VAModGroups grp;
        if (!ref_pack_of(&V->geom, V->ref_tab, &p, false)) {
            rad_step_fail(c, "minimaxh3: a reference table the pack refuses");
            return;
        }
        ref2va_mod_groups(p, S, ref_pads_of(V->ref_tab), &grp);
        for (size_t i = 0; i < grp.md.size(); ++i) cell.emplace_back(grp.md[i], grp.r0[i]);
    } else {
        for (int e = 0; e < kSegs; ++e)
            cell.emplace_back(kSegMod[e], e == kSegAudio ? S : 0);
    }
    const int64_t ncell = (int64_t)cell.size();
    if (ncell > kMaxSegs) {
        rad_step_fail(c, "minimaxh3: the references read more modulations than the table holds");
        return;
    }
    for (int64_t l = 0; l < L; ++l) {
        const DitBlock& B = m.dit[(size_t)l];
        for (int64_t e = 0; e < ncell; ++e) {
            const int md = cell[(size_t)e].first;
            const int64_t r0 = cell[(size_t)e].second;
            const RadOperand out =
                bcol_at(m.b_ada, base, L * kMaxSegs * m.Ws, (l * kMaxSegs + e) * m.Ws, m.Ws, S);
            if (m.curve > 0) {
                RAD_ISSUE_N(c, B.op_ada[md], S, brow_slice(m.a_temb.x, r0, S, td),
                            RAD_W(B.ada_w[md]), RAD_W(B.ada_b[md]), RAD_NONE, out);
            } else if (m.ada_i8) {
                RAD_ISSUE_N(c, B.op_ada[md], S, brow_slice(m.a_temb.q8, r0, S, td),
                            brow_slice(m.a_temb.s8, r0, S, td / m.ada_g8), RAD_W(B.ada[md].w),
                            RAD_W(B.ada[md].s), brows(m.b_adat, S), RAD_NONE, RAD_NONE);
                RAD_ISSUE_N(c, B.op_ada_b[md], S, brows(m.b_adat, S), RAD_W(B.ada_b[md]), out);
            } else
                RAD_ISSUE_N(c, B.op_ada[md], S, brow_slice(m.a_temb.q, r0, S, td),
                            brow_slice(m.a_temb.s, r0, S, td / RAD_SCALE_GROUP), RAD_W(B.ada[md].w),
                            RAD_W(B.ada[md].s), RAD_W(B.ada_b[md]), out);
        }
    }
    for (int64_t e = 0; e < ncell; ++e) {
        const int64_t r0 = cell[(size_t)e].second;
        RAD_ISSUE_N(c, m.op_no, S, brow_slice(m.a_temb.x, r0, S, td), RAD_W(m.w_no), RAD_W(m.b_no),
                    RAD_NONE, bcol_at(m.b_nout, base, (int64_t)kMaxSegs * 2 * D, e * 2 * D, 2 * D, S));
    }
}

/* One denoising step of the lane: the packed sequence, the blocks, the heads and both Euler
 * steps. */
static void step_dit(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    const Geom& g = m.gd;
    const int64_t R = batch->n_tok, lane = V->lane, nt = V->geom.n_text;
    const int64_t arows = (int64_t)kAudioCh * V->geom.n_audio;
    const int64_t D = m.D, qd = g.q_dim(), L = g.n_layer, La = kMaxLanes;
    const RadOperand lane_d = praw(V->lane_d, RAD_I32, 1);

    /* With references the sequence is the pack's: the reference blocks between the prompt and
     * the two latents, which stay last, and the modulation in groups (rad_ref2va_pack.h). */
    Ref2VAPack pk;
    Ref2VAModGroups grp;
    int64_t ng = kSegs, ncell = kSegs, nfine = kSegs;
    int64_t vrows = R - nt - arows, aseg = nt, vseg = nt + arows;
    if (V->ref_tab) {
        if (!ref_pack_of(&V->geom, V->ref_tab, &pk, false) || pk.rows != R) {
            rad_step_fail(c, "minimaxh3: a step whose reference table is not its sequence");
            return;
        }
        ref2va_mod_groups(pk, m.S, ref_pads_of(V->ref_tab), &grp);
        ng = (int64_t)grp.col.size();
        ncell = (int64_t)grp.md.size();
        if (ncell > kMaxSegs) {
            rad_step_fail(c, "minimaxh3: the references read more modulations than the table holds");
            return;
        }
        nfine = (int64_t)pk.segs.size();
        const Ref2VASeg& vid = pk.segs.back();
        vrows = vid.end - vid.start;
        aseg = pk.segs[pk.segs.size() - 2].start;
        vseg = vid.start;
    }
    const bool use_vsa = m.vsa && R >= m.vsa_min_rows;
    if (use_vsa) {
        int64_t prefix = 0;
        if (V->ref_tab) {
            for (const Ref2VASeg& s : pk.segs)
                if (s.start < vseg) prefix += vsa_prefix_tiles(s.end - s.start);
        } else {
            prefix = vsa_prefix_tiles(nt) + vsa_prefix_tiles(arows);
        }
        const int64_t video = vsa_video_tiles(V->geom.lat_frames, V->geom.lat_h / kPatch,
                                              V->geom.lat_w / kPatch);
        if (prefix + video > m.vsa_tiles || vsa_keep(m.vsa_sparsity, video) > m.vsa_k) {
            rad_step_fail(c, "minimaxh3: packed sequence exceeds sparse attention tile capacity");
            return;
        }
    }

    if (V->step == 0) {
        RAD_ISSUE_N(c, m.op_rnd_v, vrows, praw(V->first_d, RAD_I32, 1), praw(V->seed, RAD_U32, 2),
                    lane_d, praw(V->lat_cu, RAD_I32, 2), brows(m.b_vstate, La * m.maxV));
        RAD_ISSUE_N(c, m.op_rnd_a, arows, praw(V->first_d, RAD_I32, 1),
                    praw(V->seed + 2, RAD_U32, 2), lane_d, praw(V->lat_cu + 2, RAD_I32, 2),
                    brows(m.b_astate, La * m.maxA));
    }
    /* The whole row whatever the groups: the gather's kernel is specialised to the width it was
     * declared at. */
    const int64_t row = L * kMaxSegs * m.Ws, ti = lane * m.S + V->step;
    if (g.world > 1)
        RAD_ISSUE_N(c, m.op_modg, row, brow_slice(m.b_ada, ti, 1, row),
                    brows(m.b_modg, (int64_t)g.world * L * kMaxSegs));
    else
        RAD_ISSUE_N(c, m.op_modg, 1, brow_slice(m.b_ada, ti, 1, row),
                    RAD_P_T2(rad_buf_ptr(c, m.b_modg), RAD_BF16, 1, row));

    RAD_ISSUE_N(c, m.op_text, nt, brows(m.b_text, La * kMaxText), lane_d,
                praw(V->seg_cu, RAD_I32, 2), brows(m.b_h, nt));
    /* The two latents, through their input projections. */
    RAD_ISSUE_N(c, m.op_ain, arows, brow_slice(m.b_astate, lane * m.maxA, arows, kAudioC),
                RAD_W(m.w_ain), RAD_W(m.b_ain), RAD_NONE, brow_slice(m.b_h, aseg, arows, D));
    RAD_ISSUE_N(c, m.op_vin, vrows, brow_slice(m.b_vstate, lane * m.maxV, vrows, kPatchVals),
                RAD_W(m.w_vin), RAD_W(m.b_vin), RAD_NONE, brow_slice(m.b_h, vseg, vrows, D));
    /* Then each keyframe's and reference's block, through the same projection from the rows its
     * encode pass wrote. */
    if (V->ref_tab) {
        std::vector<RefPart> parts;
        ref_parts(V->ref_tab, &parts);
        size_t pi = 0;
        for (const Ref2VASeg& s : pk.segs) {
            if (s.update || s.kind == kR2VText) continue;
            const bool aud = ref2va_is_audio(s.kind) != 0;
            const int64_t n = s.end - s.start;
            if (pi >= parts.size() || parts[pi].row < 0 || (parts[pi].audio == 1) != aud) {
                rad_step_fail(c, "minimaxh3: a reference block with no encoded latents");
                return;
            }
            const int64_t row = parts[pi++].row;
            /* A soundtrack's slot holds a channel every kEncSlotRowsA / kAudioCh rows (step_aenc)
             * and its segment the channels one after the other, n / 2 latents each: a song
             * guide's are the encoded song's first, cut where the video ends. */
            if (aud)
                for (int64_t ch = 0, t = n / kAudioCh; ch < kAudioCh; ++ch)
                    RAD_ISSUE_N(c, m.op_ain, t,
                                brow_slice(m.b_enca,
                                           row + ch * ((int64_t)kEncSlotRowsA / kAudioCh), t,
                                           kAudioC),
                                RAD_W(m.w_ain), RAD_W(m.b_ain), RAD_NONE,
                                brow_slice(m.b_h, s.start + ch * t, t, D));
            else
                RAD_ISSUE_N(c, m.op_vin, n, brow_slice(m.b_evc, row, n, kPatchVals),
                            RAD_W(m.w_vin), RAD_W(m.b_vin), RAD_NONE,
                            brow_slice(m.b_h, s.start, n, D));
        }
    }

    const RadOperand segs = praw(V->seg_cu, RAD_I32, nfine + 1);
    /* The attention takes the segments, the modulated norms the modulation groups and, with
     * references, the cell each group reads (mod_cu's tail). */
    const RadOperand mod_segs = praw(V->mod_cu, RAD_I32, ng + 1);
    const RadOperand mod_col = V->ref_tab ? praw(V->mod_cu + ng + 1, RAD_I32, ng) : RAD_NONE;
    const int64_t vcell = V->ref_tab ? grp.col.back() : kSegs - 1;
    const RadOperand geom = praw(V->geom_d, RAD_I32, RAD_VGEOM_N);
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, 2);
    const RadOperand pos = praw2(V->pos, RAD_F32, 3, R);
    /* Chunk k of block l's six, for every group: on rank k / (6 / world), at its column. */
    const int64_t per = 6 / g.world, nD = D / RAD_SCALE_GROUP;
    auto mod = [&](int64_t l, int k, int chunks) {
        const int64_t r = k / per;
        return bcol_at(m.b_modg, r * L * kMaxSegs + l * kMaxSegs, m.Ws, (k % per) * D, chunks * D,
                       ncell);
    };
    /* Two halves of the rows with more than one rank. Everything between two attentions is
     * row-local, so a half computes exactly the bits the whole would, and each half's all-reduces
     * run on lane 1 while lane 0 computes the other half; the attention stays whole. The second
     * half starts at the video's middle row, so it is video alone: the batch's one-sequence
     * cu_seqlens and the video's modulation (its cell alone) stand for the tables. Lane 0 waits
     * for a half's all-reduce before its first reader, issued before the next all-reduce goes to
     * lane 1 (a join orders a lane behind everything issued on the other). */
    const bool split = g.world > 1;
    const int64_t cut = split ? vseg + vrows / 2 : R;
    const int64_t hr0[2] = { 0, cut }, hn[2] = { cut, R - cut };
    const int nh = split ? 2 : 1;
    const RadOperand segh[2] = { mod_segs, cu }, colh[2] = { mod_col, RAD_NONE };
    auto modh = [&](int h, int64_t l, int k, int chunks) {
        if (h == 0) return mod(l, k, chunks);
        const int64_t r = k / per;
        return bcol_at(m.b_modg, r * L * kMaxSegs + l * kMaxSegs + vcell, m.Ws, (k % per) * D,
                       chunks * D, 1);
    };
    /* A half's partial sums across the ranks, kArRows rows a message, on lane 1 when split. */
    auto all_reduce_half = [&](rad_op op, int64_t r0, int64_t n) {
        if (!op) return;
        if (split) {
            rad_lane_join(c, 0, 1);
            rad_lane(c, 1);
        }
        for (int64_t q0 = 0; q0 < n; q0 += kArRows) {
            const int64_t k = std::min<int64_t>(kArRows, n - q0);
            RAD_ISSUE_N(c, op, k * D, brow_slice(m.b_ao, r0 + q0, k, D), RAD_NONE);
        }
        if (split) rad_lane(c, 0);
    };
    auto wait_ar = [&]() { if (split) rad_lane_join(c, 1, 0); };
    for (int64_t l = 0; l < L; ++l) {
        const DitBlock& B = m.dit[(size_t)l];
        /* The first half's norm reads the all-reduce the previous block waited for last; the
         * second half's waits here, so its reduction ran under the first half's norm and qkv. */
        for (int h = 0; h < nh; ++h) {
            const int64_t r0 = hr0[h], n = hn[h];
            if (h > 0 && l > 0) wait_ar();
            RAD_ISSUE_N(c, B.op_mq1, n, l > 0 ? brow_slice(m.b_ao, r0, n, D) : RAD_NONE,
                        l > 0 ? modh(h, l - 1, 5, 1) : RAD_NONE, modh(h, l, 0, 2),
                        RAD_W(B.w_n1), segh[h], brow_slice(m.b_h, r0, n, D),
                        brow_slice(m.a_hm.cq(), r0, n, D), brow_slice(m.a_hm.cs(), r0, n, nD),
                        RAD_NONE, colh[h]);
            B.qkv.step(c, m.a_hm, m.b_qkv, R, r0, n);
        }
        RAD_ISSUE_N(c, B.op_qkr, R, bcol(m.b_qkv, 0, 3 * qd, R), pos, RAD_W(B.w_qn),
                    RAD_W(B.w_kn));
        RAD_ISSUE_N(c, B.op_kq, R, bcol(m.b_qkv, qd, qd, R), brows(m.b_k8, R), brows(m.b_ks, R));
        RAD_ISSUE_N(c, B.op_vq, R, bcol(m.b_qkv, 2 * qd, qd, R), brows(m.b_v8, R), RAD_B(m.b_vs));
        if (use_vsa && B.op_vsa) {
            const RadOperand pooled_values = m.vsa_gated ? bcol(m.b_qkv, 2 * qd, qd, R) : RAD_NONE;
            const RadOperand coarse = m.vsa_gated ? RAD_B(m.b_coarse) : RAD_NONE;
            const RadOperand gate = m.vsa_gated ? bcol(m.b_qkv, 3 * qd, qd, R) : RAD_NONE;
            RAD_ISSUE_N(c, B.op_sel, R, bcol(m.b_qkv, 0, qd, R), bcol(m.b_qkv, qd, qd, R),
                        pooled_values, segs, geom, RAD_B(m.b_sel), coarse);
            RAD_ISSUE_N(c, B.op_vsa, R, bcol(m.b_qkv, 0, qd, R), bcol(m.b_qkv, qd, qd, R),
                        brows(m.b_v8, R), segs, geom, RAD_B(m.b_sel), coarse,
                        gate, brows(m.a_attn.x, R), brows(m.b_k8, R),
                        brows(m.b_ks, R), RAD_B(m.b_vs));
        } else {
            RAD_ISSUE_N(c, B.op_attn, R, bcol(m.b_qkv, 0, qd, R), bcol(m.b_qkv, qd, qd, R),
                        brows(m.b_v8, R), RAD_NONE, RAD_NONE, RAD_NONE, cu, RAD_NONE, RAD_NONE,
                        brows(m.b_k8, R), brows(m.b_ks, R), brows(m.a_attn.x, R), RAD_B(m.b_vs));
        }
        /* Each wait below is for the all-reduce lane 1 holds last, which is the one lane 0 reads
         * next: the second half's projection runs under the first half's reduction, the first
         * half's feed-forward under the second's, and so on. */
        for (int h = 0; h < nh; ++h) {
            B.q_attn.step(c, R, hr0[h], hn[h]);
            B.o.step(c, m.a_attn, m.b_ao, R, hr0[h], hn[h]);
            if (h > 0) wait_ar();
            all_reduce_half(B.op_ar1, hr0[h], hn[h]);
        }
        for (int h = 0; h < nh; ++h) {
            const int64_t r0 = hr0[h], n = hn[h];
            RAD_ISSUE_N(c, B.op_mq2, n, brow_slice(m.b_ao, r0, n, D), modh(h, l, 2, 1),
                        modh(h, l, 3, 2), RAD_W(B.w_n2), segh[h], brow_slice(m.b_h, r0, n, D),
                        brow_slice(m.a_hm.cq(), r0, n, D), brow_slice(m.a_hm.cs(), r0, n, nD),
                        RAD_NONE, colh[h]);
            B.gate_up.step(c, m.a_hm, m.b_gu, R, r0, n);
            B.gq.step(c, m.b_gu, R, r0, n);
            B.down.step(c, m.a_ffn, m.b_ao, R, r0, n);
            wait_ar();
            all_reduce_half(B.op_ar2, r0, n);
        }
    }
    wait_ar();
    /* Row (lane, step) of the table, kMaxSegs cells of 2 D. */
    const char* nout = (const char*)rad_buf_ptr(c, m.b_nout) +
                       (lane * m.S + V->step) * (int64_t)kMaxSegs * 2 * D * (int64_t)sizeof(uint16_t);
    RAD_ISSUE_N(c, m.op_nout, R, brows(m.b_ao, R), mod(L - 1, 5, 1),
                RAD_P_T2(nout, RAD_BF16, ncell, 2 * D), RAD_W(m.w_no_n), mod_segs, brows(m.b_h, R),
                brows(m.b_hn, R), mod_col);

    /* The heads, in f32 over parts of the rows. */
    auto heads = [&](rad_op op, rad_weight w, rad_weight bias, int64_t first, int64_t n, int64_t nout_c,
                     rad_buf y) {
        for (int64_t r0 = 0; r0 < n; r0 += kHeadRows) {
            const int64_t k = std::min<int64_t>(kHeadRows, n - r0);
            RAD_ISSUE_N(c, m.op_hcast, k, brow_slice(m.b_hn, first + r0, k, D), brows(m.b_hf, k));
            RAD_ISSUE_N(c, op, k, brows(m.b_hf, k), RAD_W(w), RAD_W(bias), RAD_NONE,
                        brow_slice(y, r0, k, nout_c));
        }
    };
    heads(m.op_aout, m.w_aout, m.b_aout, aseg, arows, kAudioC, m.b_avel);
    heads(m.op_vout, m.w_vout, m.b_vout, vseg, vrows, kPatchVals, m.b_vel);
    /* The sampler's step (RadVideoGenBatch::sched and sampler). */
    if (V->sampler == RAD_SAMPLER_RES_MULTISTEP) {
        RAD_ISSUE_N(c, m.op_rmv, vrows, brows(m.b_vel, vrows),
                    brow_slice(m.b_vx0, lane * m.maxV, vrows, kPatchVals),
                    praw(V->sched + 6, RAD_F32, 1), praw(V->sched + 7, RAD_F32, 1),
                    praw(V->sched + 8, RAD_F32, 1), praw(V->sched + 9, RAD_F32, 1),
                    praw(V->sched + 10, RAD_F32, 1),
                    brow_slice(m.b_vstate, lane * m.maxV, vrows, kPatchVals));
        RAD_ISSUE_N(c, m.op_rma, arows, brows(m.b_avel, arows),
                    brow_slice(m.b_ax0, lane * m.maxA, arows, kAudioC),
                    praw(V->sched + 11, RAD_F32, 1), praw(V->sched + 12, RAD_F32, 1),
                    praw(V->sched + 13, RAD_F32, 1), praw(V->sched + 14, RAD_F32, 1),
                    praw(V->sched + 15, RAD_F32, 1),
                    brow_slice(m.b_astate, lane * m.maxA, arows, kAudioC));
    } else {
        RAD_ISSUE_N(c, m.op_ev, vrows, brows(m.b_vel, vrows), praw(V->sched, RAD_F32, 1),
                    praw(V->sched + 1, RAD_F32, 1), praw(V->sched + 2, RAD_F32, 1),
                    brow_slice(m.b_vstate, lane * m.maxV, vrows, kPatchVals));
        RAD_ISSUE_N(c, m.op_ea, arows, brows(m.b_avel, arows), praw(V->sched + 3, RAD_F32, 1),
                    praw(V->sched + 4, RAD_F32, 1), praw(V->sched + 5, RAD_F32, 1),
                    brow_slice(m.b_astate, lane * m.maxA, arows, kAudioC));
    }
}

/* The soundtrack: both channels, each a sequence, through the audio decoder, then interleaved. */
static void step_audio(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    const int64_t lane = V->lane, na = V->geom.n_audio, r0 = kAudioCh * na;
    const RadOperand nlen = praw(V->geom_d + RAD_VGEOM_N_AUDIO, RAD_I32, 1);
    auto conv = [&](const AConv& k, rad_buf x, rad_buf res, rad_buf y, int64_t R, rad_buf cu) {
        RAD_ISSUE_N(c, k.op, R, brows(x, R), RAD_W(k.w), brows(cu, 3), res ? brows(res, R) : RAD_NONE,
                    brows(y, R), k.b ? RAD_W(k.b) : RAD_NONE);
    };
    auto act = [&](const AAct& a, rad_buf x, rad_buf y, int64_t R, rad_buf cu) {
        RAD_ISSUE_N(c, a.op, R, brows(x, R), RAD_W(a.alpha), RAD_W(a.beta), RAD_W(a.up),
                    RAD_W(a.down), brows(cu, 3), brows(y, R));
    };
    RAD_ISSUE_N(c, m.op_adenorm, r0, brow_slice(m.b_astate, lane * m.maxA, r0, kAudioC),
                brows(m.b_aden, r0));
    RAD_ISSUE_N(c, m.op_acu0, 3, nlen, brows(m.b_acu[0], 3));
    conv(m.a_in, m.b_aden, 0, m.b_ain0, r0, m.b_acu[0]);
    conv(m.a_pre, m.b_ain0, 0, m.b_aavg[0], r0, m.b_acu[0]);
    int64_t rows = r0;
    for (int s = 0; s < 7; ++s) {
        const AStage& St = m.ast[s];
        const int64_t R = rows * St.rate;
        RAD_ISSUE_N(c, St.op_cu, 3, nlen, brows(m.b_acu[s + 1], 3));
        RAD_ISSUE_N(c, St.up.op, rows, brows(m.b_aavg[s], rows), RAD_W(St.up.w), RAD_W(St.up.b),
                    brows(m.b_acu[s], 3), brows(m.b_ah[s + 1], R));
        for (int j = 0; j < 3; ++j)
            for (int d = 0; d < 3; ++d) {
                const AmpUnit& U = St.amp[j].u[d];
                const rad_buf x = d == 0 ? m.b_ah[s + 1] : m.b_aamp[j][s + 1];
                act(U.a1, x, m.b_at1[s + 1], R, m.b_acu[s + 1]);
                conv(U.c1, m.b_at1[s + 1], 0, m.b_at2[s + 1], R, m.b_acu[s + 1]);
                act(U.a2, m.b_at2[s + 1], m.b_at1[s + 1], R, m.b_acu[s + 1]);
                conv(U.c2, m.b_at1[s + 1], x, m.b_aamp[j][s + 1], R, m.b_acu[s + 1]);
            }
        RAD_ISSUE_N(c, St.op_sum, R, brows(m.b_aamp[0][s + 1], R), brows(m.b_aamp[1][s + 1], R),
                    brows(m.b_aavg[s + 1], R));
        RAD_ISSUE_N(c, St.op_avg, R, brows(m.b_aavg[s + 1], R), brows(m.b_aamp[2][s + 1], R),
                    brows(m.b_aavg[s + 1], R));
        rows = R;
    }
    act(m.a_actpost, m.b_aavg[7], m.b_at1[7], rows, m.b_acu[7]);
    conv(m.a_post, m.b_at1[7], 0, m.b_aout0, rows, m.b_acu[7]);
    const int64_t n = rows / kAudioCh;
    RAD_ISSUE_N(c, m.op_vmix, n, brow_slice(m.b_aout0, 0, n, 1), brow_slice(m.b_aout0, n, n, 1),
                brows(m.b_audio, n));
}

/* One chunk of frames: the latents, this rank's tiles through the ViT, every rank's gathered, and
 * the stitch into the chunk's frames. */
static void step_video(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    const int64_t lane = V->lane, lh = V->geom.lat_h, lw = V->geom.lat_w;
    const int64_t ppf = (lh / kPatch) * (lw / kPatch), vox = (int64_t)kDecLat * lh * lw;
    const int64_t dim = 2048, px = 3 * 4 * kScale * kScale;
    const int64_t nt = tiles_of(lh, lw), tpr = (nt + m.W - 1) / m.W;
    const int64_t t0 = (int64_t)m.gt.rank * tpr;
    const int64_t mine = std::max<int64_t>(0, std::min<int64_t>(tpr, nt - t0));
    const int64_t R = mine * kPeriod, a = t0 * kPeriod;

    RAD_ISSUE_N(c, m.op_unfold, vox,
                brow_slice(m.b_vstate, lane * m.maxV + (int64_t)V->chunk * kLpc * ppf,
                           (int64_t)kDecLat * ppf, kPatchVals),
                praw(V->geom_d + RAD_VGEOM_PATCH_FRAMES, RAD_I32, 3), brows(m.b_vox, vox));
    RAD_ISSUE_N(c, m.op_pqc, vox, brows(m.b_vox, vox), RAD_W(m.w_pqc), RAD_W(m.b_pqc), RAD_NONE,
                brows(m.b_vox2, vox));
    RAD_ISSUE_N(c, m.op_vpin, vox, brows(m.b_vox2, vox), RAD_W(m.w_vpin2), RAD_W(m.b_vpin2),
                RAD_NONE, brows(m.b_vpin, vox));
    RAD_ISSUE_N(c, m.op_tiles, nt * kPeriod, brows(m.b_vpin, vox), RAD_W(m.w_reg),
                praw(V->geom_d + RAD_VGEOM_CHUNK_FRAMES, RAD_I32, 3),
                brows(m.b_vx, nt * kPeriod));
    if (mine > 0) {
        const RadOperand x = brow_slice(m.b_vx, a, R, dim);
        const RadOperand pos = bcol(m.b_dpos, 0, R, 3);
        RAD_ISSUE_N(c, m.op_dpos, R, pos);
        RAD_ISSUE_N(c, m.op_dcu, mine + 1, brows(m.b_dcu, mine + 1));
        for (const VitBlock& B : m.vit) {
            RAD_ISSUE_N(c, B.op_n1, R, x, RAD_W(B.w_n1), brows(m.b_vn, R));
            RAD_ISSUE_N(c, B.op_qkv, R, brows(m.b_vn, R), RAD_W(B.w_qkv), RAD_W(B.b_qkv), RAD_NONE,
                        brows(m.b_vqkv, R));
            RAD_ISSUE_N(c, B.op_qkr, R, brows(m.b_vqkv, R), pos, RAD_NONE, RAD_NONE);
            RAD_ISSUE_N(c, B.op_attn, R, bcol(m.b_vqkv, 0, dim, R), bcol(m.b_vqkv, dim, dim, R),
                        bcol(m.b_vqkv, 2 * dim, dim, R), brows(m.b_dcu, mine + 1),
                        brows(m.b_vattn, R));
            RAD_ISSUE_N(c, B.op_o, R, brows(m.b_vattn, R), RAD_W(B.w_o), RAD_W(B.b_o), RAD_NONE,
                        brows(m.b_vo, R));
            RAD_ISSUE_N(c, B.op_s1, R, brows(m.b_vo, R), RAD_W(B.s1), brows(m.b_vt, R));
            RAD_ISSUE_N(c, B.op_r1, R, x, brows(m.b_vt, R), x);
            RAD_ISSUE_N(c, B.op_n2, R, x, RAD_W(B.w_n2), brows(m.b_vn, R));
            RAD_ISSUE_N(c, B.op_up, R, brows(m.b_vn, R), RAD_W(B.w_up), RAD_W(B.b_up), RAD_NONE,
                        brows(m.b_vup, R));
            RAD_ISSUE_N(c, B.op_act, R, brows(m.b_vup, R), brows(m.b_vact, R));
            RAD_ISSUE_N(c, B.op_down, R, brows(m.b_vact, R), RAD_W(B.w_down), RAD_W(B.b_down),
                        RAD_NONE, brows(m.b_vo, R));
            RAD_ISSUE_N(c, B.op_s2, R, brows(m.b_vo, R), RAD_W(B.s2), brows(m.b_vt, R));
            RAD_ISSUE_N(c, B.op_r2, R, x, brows(m.b_vt, R), x);
        }
        RAD_ISSUE_N(c, m.op_vnorm, R, x, RAD_W(m.w_vno), RAD_W(m.b_vno), brows(m.b_vn, R));
        RAD_ISSUE_N(c, m.op_vproj, R, brows(m.b_vn, R), RAD_W(m.w_vpo), RAD_W(m.b_vpo), RAD_NONE,
                    brows(m.b_vpix, R));
    }
    rad_buf pix = m.b_vpix;
    if (m.W > 1) {
        RAD_ISSUE_N(c, m.op_vgather, tpr * kPeriod * px, brows(m.b_vpix, tpr * kPeriod),
                    brows(m.b_vpix_all, m.W * tpr * kPeriod));
        pix = m.b_vpix_all;
    }
    const int64_t hp = lh * kScale, wp = lw * kScale, n = (int64_t)V->n_frames_out * hp * wp;
    RAD_ISSUE_N(c, m.op_stitch, n, brows(pix, nt * kPeriod),
                praw(V->geom_d + RAD_VGEOM_CHUNK_FRAMES, RAD_I32, 5),
                brow_slice(m.b_prev, lane * kFtail * m.max_area, (int64_t)kFtail * hp * wp, 3),
                brows(m.b_frames, n));
}

/* The latent upscale: the lane's clean small latent through the upscaler to the full size, then
 * both latents re-noised to the next step's sigmas with streams of their own (stream 1 of each
 * seed; the first step drew stream 0). */
static void step_upscale(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    const RadVideoGeom& s = V->geom_from;
    const RadVideoGeom& d = V->geom;
    const int64_t lane = V->lane, F = d.lat_frames, C = m.uC;
    const int64_t vs = F * s.lat_h * s.lat_w, vd = F * d.lat_h * d.lat_w;
    const int64_t pd = vd / (kPatch * kPatch), ps = vs / (kPatch * kPatch);
    const int64_t arows = (int64_t)kAudioCh * d.n_audio;
    const int32_t* gd = V->geom_d;
    const RadOperand gsrc = praw(gd + RAD_VGEOM_UP_SRC, RAD_I32, 3);
    const RadOperand gdst = praw(gd + RAD_VGEOM_UP_DST, RAD_I32, 3);
    const RadOperand st = RAD_B(m.b_ust);

    RAD_ISSUE_N(c, m.op_uunf, vs, brow_slice(m.b_vstate, lane * m.maxV, ps, kPatchVals),
                praw(gd + RAD_VGEOM_UP_PSRC, RAD_I32, 3), brows(m.b_uin32, vs));
    RAD_ISSUE_N(c, m.op_ucast, vs, brows(m.b_uin32, vs), brows(m.b_uin, vs));
    RAD_ISSUE_N(c, m.op_ufac, 1, praw(V->upscale, RAD_F32, 1), brows(m.b_ufac, 1));
    RAD_ISSUE_N(c, m.op_ue0, 1, brows(m.b_ufac, 1), RAD_W(m.u_e0w), RAD_W(m.u_e0b), RAD_NONE,
                brows(m.b_ue0, 1));
    RAD_ISSUE_N(c, m.op_ues0, 1, brows(m.b_ue0, 1), brows(m.b_ue0s, 1));
    RAD_ISSUE_N(c, m.op_ue2, 1, brows(m.b_ue0s, 1), RAD_W(m.u_e2w), RAD_W(m.u_e2b), RAD_NONE,
                brows(m.b_ue1, 1));
    RAD_ISSUE_N(c, m.op_ues2, 1, brows(m.b_ue1, 1), brows(m.b_ue1s, 1));
    for (const auto* list : { &m.u_in, &m.u_out })
        for (const UpsBlock& B : *list)
            if (!B.temporal)
                RAD_ISSUE_N(c, B.op_e, 1, brows(m.b_ue1s, 1), RAD_W(B.ew), RAD_W(B.eb), RAD_NONE,
                            brow_slice(m.b_uemb, B.emb, 1, 2 * C));
    RAD_ISSUE_N(c, m.op_ucin, vs, brows(m.b_uin, vs), RAD_W(m.u_cinw), RAD_W(m.u_cinb), gsrc,
                RAD_NONE, brows(m.b_ua, vs));
    auto blocks = [&](const std::vector<UpsBlock>& list, rad_buf X, rad_buf T, rad_buf H,
                      int64_t v, const RadOperand& g) {
        for (const UpsBlock& B : list) {
            RAD_ISSUE_N(c, B.op_n1, v, brows(X, v), RAD_W(B.n1w), RAD_W(B.n1b), RAD_NONE, st,
                        brows(T, v));
            if (B.temporal) {
                RAD_ISSUE_N(c, B.op_c1, v, brows(T, v), RAD_W(B.c1w), RAD_W(B.c1b), g, brows(H, v));
                RAD_ISSUE_N(c, B.op_c2, v, brows(H, v), RAD_W(B.c2w), RAD_W(B.c2b), brows(X, v),
                            brows(X, v));
                continue;
            }
            RAD_ISSUE_N(c, B.op_c1, v, brows(T, v), RAD_W(B.c1w), RAD_W(B.c1b), g, RAD_NONE,
                        brows(H, v));
            RAD_ISSUE_N(c, B.op_n2, v, brows(H, v), RAD_W(B.n2w), RAD_W(B.n2b),
                        brow_slice(m.b_uemb, B.emb, 1, 2 * C), st, brows(T, v));
            RAD_ISSUE_N(c, B.op_c2, v, brows(T, v), RAD_W(B.c2w), RAD_W(B.c2b), g, brows(X, v),
                        brows(X, v));
        }
    };
    blocks(m.u_in, m.b_ua, m.b_ub, m.b_uc, vs, gsrc);
    RAD_ISSUE_N(c, m.op_ursz, vd, brows(m.b_ua, vs), praw(gd + RAD_VGEOM_UP_SRC, RAD_I32, 5),
                brows(m.b_ub, vd));
    blocks(m.u_out, m.b_ub, m.b_ua, m.b_uc, vd, gdst);
    RAD_ISSUE_N(c, m.op_unout, vd, brows(m.b_ub, vd), RAD_W(m.u_now), RAD_W(m.u_nob), RAD_NONE, st,
                brows(m.b_ua, vd));
    RAD_ISSUE_N(c, m.op_ucout, vd, brows(m.b_ua, vd), RAD_W(m.u_coutw), RAD_W(m.u_coutb), gdst,
                RAD_NONE, brows(m.b_uout, vd));
    RAD_ISSUE_N(c, m.op_ufold, pd, brows(m.b_uout, vd), praw(gd + RAD_VGEOM_UP_PDST, RAD_I32, 3),
                brow_slice(m.b_vstate, lane * m.maxV, pd, kPatchVals));
    RAD_ISSUE_N(c, m.op_urnv, pd, praw(V->sched + 1, RAD_F32, 1), praw(V->seed, RAD_U32, 2),
                brow_slice(m.b_vstate, lane * m.maxV, pd, kPatchVals));
    RAD_ISSUE_N(c, m.op_urna, arows, praw(V->sched + 4, RAD_F32, 1), praw(V->seed + 2, RAD_U32, 2),
                brow_slice(m.b_astate, lane * m.maxA, arows, kAudioC));
}

/* A picture larger than a tile, encoded as ComfyUI's vae.py tiled_encode does: overlapping tiles,
 * each through the trunk alone (GroupNorm sees only its tile), and each tile's moments blended
 * with its unblended top and left neighbours into the canvas (vtile_join). A tile's geometries are
 * its row of b_tplan, which vtile_plan writes as vae.py splits. Returns the slot row past the
 * media's latents. */
static int64_t step_venc_tiled(RadCtx* c, const RadVideoGenBatch* V, const Model& m,
                               const int32_t* gd, int64_t H, int64_t W, int64_t ppc,
                               const RadOperand st, int64_t row0) {
    const int64_t lat_h = H / kVaeRatio, lat_w = W / kVaeRatio;
    const std::vector<int64_t> ty = tile_starts(lat_h), tx = tile_starts(lat_w);
    const int64_t nh = (int64_t)ty.size(), nw = (int64_t)tx.size(), nt = nh * nw;
    if (nt > m.enc_tiles) {
        rad_step_fail(c, "minimaxh3: a Vencode pass whose picture splits into more tiles than "
                         "the encoder declares room for");
        return row0;
    }
    /* vtile_plan's split on the host: each tile's side in latents and its first row in the two
     * tile-major planes. */
    struct Tile { int64_t prow, mrow, ny, nx; };
    std::vector<Tile> tl;
    tl.reserve((size_t)nt);
    int64_t spx = 0, smom = 0;
    for (int64_t i = 0; i < nh; ++i)
        for (int64_t j = 0; j < nw; ++j) {
            const int64_t ny = std::min((int64_t)kTile, lat_h - ty[(size_t)i]),
                          nx = std::min((int64_t)kTile, lat_w - tx[(size_t)j]);
            tl.push_back({ spx, smom, ny, nx });
            spx += ny * nx * (int64_t)kVaeRatio * kVaeRatio;
            smom += ny * nx;
        }
    RAD_ISSUE_N(c, m.op_vt_plan, nt, praw(gd, RAD_I32, 3), brows(m.b_tplan, nt));
    RAD_ISSUE_N(c, m.op_vt_crop, spx, praw2(V->enc_pixels, RAD_F32, H * W, 3),
                praw(gd, RAD_I32, 3), brows(m.b_vtile, spx));
    const int32_t* tp = static_cast<const int32_t*>(rad_buf_ptr(c, m.b_tplan));
    for (size_t t = 0; t < tl.size(); ++t) {
        const Tile& T = tl[t];
        const int64_t th = T.ny * kVaeRatio, tw = T.nx * kVaeRatio;   /* the tile, in pixels */
        const int64_t Rpix = th * tw, R5 = T.ny * T.nx;
        int64_t R[6];
        for (int L = 0; L < 6; ++L) {
            const int64_t d = 1 << std::min(L, 4);   /* the level's spatial divisor */
            R[L] = (th / d) * (tw / d);
        }
        /* The tile's plan row: its six {1, side, side} geometry entries, one frame deep. */
        const int32_t* tr = tp + 22 * (int64_t)t;
        const RadOperand gg[6] = { praw(tr + 4, RAD_I32, 3), praw(tr + 7, RAD_I32, 3),
                                   praw(tr + 10, RAD_I32, 3), praw(tr + 13, RAD_I32, 3),
                                   praw(tr + 16, RAD_I32, 3), praw(tr + 19, RAD_I32, 3) };
        RAD_ISSUE_N(c, m.op_ve_pix, Rpix, brow_slice(m.b_vtile, T.prow, Rpix, 3),
                    brows(m.b_epix32, Rpix));
        RAD_ISSUE_N(c, m.op_ve_pcast, Rpix, brows(m.b_epix32, Rpix), brows(m.b_epix, Rpix));
        RAD_ISSUE_N(c, m.op_ve_cin, Rpix, brows(m.b_epix, Rpix), RAD_W(m.ve_cinw),
                    RAD_W(m.ve_cinb), gg[0], RAD_NONE, brows(m.v_lv[0].a, Rpix));
        for (int L = 0; L < 6; ++L) {
            const Model::VEncLevel& E = m.v_lv[(size_t)L];
            const bool sc = E.op_sc;
            for (int blk = 0; blk < 2; ++blk) {
                const Model::VEncBlock& B = E.blk[blk];
                const bool f0 = blk == 0 && sc;
                const rad_buf xa = f0 ? E.in : E.a, nb = f0 ? E.in : E.b;
                /* The shortcut convolves the raw input, before norm1 (as in the plain loop). */
                if (f0)
                    RAD_ISSUE_N(c, E.op_sc, R[L], brows(E.in, R[L]), RAD_W(E.scw), RAD_W(E.scb),
                                gg[L], RAD_NONE, brows(E.a, R[L]));
                RAD_ISSUE_N(c, B.op_n1, R[L], brows(xa, R[L]), RAD_W(B.n1w), RAD_W(B.n1b),
                            RAD_NONE, st, brows(nb, R[L]));
                RAD_ISSUE_N(c, B.op_c1, R[L], brows(nb, R[L]), RAD_W(B.c1w), RAD_W(B.c1b),
                            gg[L], RAD_NONE, brows(E.c, R[L]));
                RAD_ISSUE_N(c, B.op_n2, R[L], brows(E.c, R[L]), RAD_W(B.n2w), RAD_W(B.n2b),
                            RAD_NONE, st, brows(E.b, R[L]));
                RAD_ISSUE_N(c, B.op_c2, R[L], brows(E.b, R[L]), RAD_W(B.c2w), RAD_W(B.c2b),
                            gg[L], brows(E.a, R[L]), brows(E.a, R[L]));
            }
            if (L < 4) {
                const Model::VEncLevel& X = m.v_lv[(size_t)L + 1];
                const rad_buf y = X.op_sc ? X.in : X.a;
                RAD_ISSUE_N(c, E.op_dn, R[L + 1], brows(E.a, R[L]), RAD_W(E.dnw), RAD_W(E.dnb),
                            gg[L], RAD_NONE, brows(y, R[L + 1]));
            }
        }
        RAD_ISSUE_N(c, m.op_ve_nout, R5, brows(m.v_lv[5].a, R5), RAD_W(m.ve_now), RAD_W(m.ve_nob),
                    RAD_NONE, st, brows(m.v_lv[5].b, R5));
        RAD_ISSUE_N(c, m.op_ve_cout, R5, brows(m.v_lv[5].b, R5), RAD_W(m.ve_coutw),
                    RAD_W(m.ve_coutb), gg[5], RAD_NONE, brows(m.b_eq48, R5));
        RAD_ISSUE_N(c, m.op_ve_q, R5, brows(m.b_eq48, R5), RAD_W(m.ve_qw), RAD_W(m.ve_qb), gg[5],
                    RAD_NONE, brow_slice(m.b_eqt, T.mrow, R5, 48));
    }
    /* The blend over the canvas (gd's level-4/5 entry, in latents), then the plain path's fold:
     * the canvas has a one-clip encode's b_eq2 layout. */
    RAD_ISSUE_N(c, m.op_vt_join, lat_h * lat_w, brows(m.b_eqt, smom), praw(gd + 12, RAD_I32, 3),
                brows(m.b_eq2, lat_h * lat_w));
    RAD_ISSUE_N(c, m.op_ve_fold, ppc, bcol(m.b_eq2, 0, kC, lat_h * lat_w),
                praw(gd + 18, RAD_I32, 3), brow_slice(m.b_evc, row0, ppc, kPatchVals));
    return row0 + ppc;
}


/* A Vencode pass: each clip of `enc_pixels` through the levels, its latents folded into the
 * media's slot of b_evc. The geometry is derived again from the batch, which checks it. */
static void step_venc(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    int32_t g[25];
    if (!V->enc_pixels ||
        !venc_geom(V->enc_frames, V->enc_height, V->enc_width, g)) {
        rad_step_fail(c, "minimaxh3: a Vencode pass with no pixels, or frames, height or width "
                         "the video encoder does not take");
        return;
    }
    const int64_t row0 = venc_slot_row(V->enc_kind, V->enc_slot);
    if (row0 < 0) {
        rad_step_fail(c, "minimaxh3: a Vencode pass naming an encode kind or slot this plugin "
                         "does not know");
        return;
    }
    if ((int64_t)V->enc_frames != g[23]) {
        rad_step_fail(c, "minimaxh3: a Vencode pass whose frames are not the clip-padded count "
                         "rad_arch_encode_geom reports");
        return;
    }
    const int32_t* gd = V->enc_geom_d;
    const int64_t H = V->enc_height, W = V->enc_width;
    const int64_t ppc = (H / (kVaeRatio * kPatch)) * (W / (kVaeRatio * kPatch));
    const int64_t n_clips = g[22], full = g[15], short_last = g[18];
    if ((full * (n_clips - 1) + short_last) * ppc > kEncSlotRowsV) {
        rad_step_fail(c, "minimaxh3: a Vencode pass whose latents overflow the lane's slot");
        return;
    }
    const RadOperand st = RAD_B(m.b_egn);
    /* Each level's volume in the geometry; the last two levels have no downsampler between
     * them. */
    static const int kVOrg[6] = { 0, 3, 6, 9, 12, 12 };
    int64_t frow = row0;
    /* A one-frame media larger than a tile goes through the tiled path, as vae.py tiles its image
     * encodes; a clip never does, since venc_geom bounds its sides to kEncMaxSide. */
    const bool tiled = V->enc_frames == 1 && (H / kVaeRatio > kTile || W / kVaeRatio > kTile);
    if (tiled)
        frow = step_venc_tiled(c, V, m, gd, H, W, ppc, st, row0);
    else
    for (int64_t cc = 0; cc < n_clips; ++cc) {
        const bool last = cc == n_clips - 1;
        const int64_t R0 = (int64_t)g[0] * H * W;
        const RadOperand gg[6] = { praw(gd + kVOrg[0], RAD_I32, 3), praw(gd + kVOrg[1], RAD_I32, 3),
                                   praw(gd + kVOrg[2], RAD_I32, 3), praw(gd + kVOrg[3], RAD_I32, 3),
                                   praw(gd + kVOrg[4], RAD_I32, 3), praw(gd + kVOrg[5], RAD_I32, 3) };
        RAD_ISSUE_N(c, m.op_ve_pix, R0, praw2(V->enc_pixels + cc * R0 * 3, RAD_F32, R0, 3),
                    brows(m.b_epix32, R0));
        RAD_ISSUE_N(c, m.op_ve_pcast, R0, brows(m.b_epix32, R0), brows(m.b_epix, R0));
        RAD_ISSUE_N(c, m.op_ve_cin, R0, brows(m.b_epix, R0), RAD_W(m.ve_cinw), RAD_W(m.ve_cinb),
                    gg[0], RAD_NONE, brows(m.v_lv[0].a, R0));
        for (int L = 0; L < 6; ++L) {
            const Model::VEncLevel& E = m.v_lv[(size_t)L];
            const int64_t R = (int64_t)g[kVOrg[L]] * g[kVOrg[L] + 1] * g[kVOrg[L] + 2];
            const bool sc = E.op_sc;
            for (int blk = 0; blk < 2; ++blk) {
                const Model::VEncBlock& B = E.blk[blk];
                const bool f0 = blk == 0 && sc;
                const rad_buf xa = f0 ? E.in : E.a, nb = f0 ? E.in : E.b;
                /* The shortcut convolves the raw input, so it runs before norm1 normalises `in` in
                 * place. */
                if (f0)
                    RAD_ISSUE_N(c, E.op_sc, R, brows(E.in, R), RAD_W(E.scw), RAD_W(E.scb), gg[L],
                                RAD_NONE, brows(E.a, R));
                RAD_ISSUE_N(c, B.op_n1, R, brows(xa, R), RAD_W(B.n1w), RAD_W(B.n1b), RAD_NONE, st,
                            brows(nb, R));
                RAD_ISSUE_N(c, B.op_c1, R, brows(nb, R), RAD_W(B.c1w), RAD_W(B.c1b), gg[L],
                            RAD_NONE, brows(E.c, R));
                RAD_ISSUE_N(c, B.op_n2, R, brows(E.c, R), RAD_W(B.n2w), RAD_W(B.n2b), RAD_NONE, st,
                            brows(E.b, R));
                RAD_ISSUE_N(c, B.op_c2, R, brows(E.b, R), RAD_W(B.c2w), RAD_W(B.c2b), gg[L],
                            brows(E.a, R), brows(E.a, R));
            }
            if (L < 4) {
                const Model::VEncLevel& X = m.v_lv[(size_t)L + 1];
                const int64_t RX = (int64_t)g[3 * (L + 1)] * g[3 * (L + 1) + 1] * g[3 * (L + 1) + 2];
                const rad_buf y = X.op_sc ? X.in : X.a;
                RAD_ISSUE_N(c, E.op_dn, RX, brows(E.a, R), RAD_W(E.dnw), RAD_W(E.dnb), gg[L],
                            RAD_NONE, brows(y, RX));
            }
        }
        const int64_t R5 = (int64_t)g[12] * g[13] * g[14];
        RAD_ISSUE_N(c, m.op_ve_nout, R5, brows(m.v_lv[5].a, R5), RAD_W(m.ve_now), RAD_W(m.ve_nob),
                    RAD_NONE, st, brows(m.v_lv[5].b, R5));
        RAD_ISSUE_N(c, m.op_ve_cout, R5, brows(m.v_lv[5].b, R5), RAD_W(m.ve_coutw),
                    RAD_W(m.ve_coutb), gg[5], RAD_NONE, brows(m.b_eq48, R5));
        RAD_ISSUE_N(c, m.op_ve_q, R5, brows(m.b_eq48, R5), RAD_W(m.ve_qw), RAD_W(m.ve_qb), gg[5],
                    RAD_NONE, brows(m.b_eq2, R5));
        const int64_t flat = last ? short_last : full;
        RAD_ISSUE_N(c, m.op_ve_fold, flat * ppc,
                    bcol(m.b_eq2, 0, kC, flat * (H / kVaeRatio) * (W / kVaeRatio)),
                    praw(gd + (last ? 18 : 15), RAD_I32, 3),
                    brow_slice(m.b_evc, frow, flat * ppc, kPatchVals));
        frow += flat * ppc;
    }
    /* The condition noise over the item's rows, numbered from its first row (ComfyUI reseeds each
     * item). b_evc persists, so once an encode is the blend every step reads. */
    if (frow > row0)
        RAD_ISSUE_N(c, m.op_eaug, frow - row0, praw(V->seed, RAD_U32, 2),
                    brow_slice(m.b_evc, row0, frow - row0, kPatchVals));
}

/* An Aencode pass: each channel of `enc_samples` through the encoder, its latents into the media's
 * slot of b_enca. The stages' lengths are the staged geometry's cus. */
static void step_aenc(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadVideoGenBatch* V = batch->vgen_b;
    int32_t g[13];
    if (!V->enc_samples || !aenc_geom(V->enc_n_samples, V->enc_channels, g)) {
        rad_step_fail(c, "minimaxh3: an Aencode pass with no samples, or a length and channel "
                         "count the audio encoder does not take");
        return;
    }
    const int64_t row0 = aenc_slot_row(V->enc_kind, V->enc_slot);
    if (row0 < 0) {
        rad_step_fail(c, "minimaxh3: an Aencode pass naming an encode kind or slot this plugin "
                         "does not know");
        return;
    }
    const int64_t n_pad = g[0];
    if ((int64_t)V->enc_n_samples != n_pad) {
        rad_step_fail(c, "minimaxh3: an Aencode pass whose samples are not padded to the length "
                         "rad_arch_encode_geom reports");
        return;
    }
    const int32_t* gd = V->enc_geom_d;
    const int64_t ld = 2048, hd = ld / 8;
    const RadOperand cu6 = praw(gd + 11, RAD_I32, 2);
    for (int32_t ch = 0; ch < V->enc_channels; ++ch) {
        RAD_ISSUE_N(c, m.op_ae_cin, n_pad, praw2(V->enc_samples + ch * n_pad, RAD_F32, n_pad, 1),
                    RAD_W(m.ae_cinw), praw(gd + 1, RAD_I32, 2), RAD_NONE,
                    brows(m.b_ax[0], n_pad), RAD_W(m.ae_cinb));
        int64_t rows = n_pad;
        for (int blk = 0; blk < 5; ++blk) {
            const Model::AEncBlk& A = m.ae_blk[blk];
            const RadOperand cu = praw(gd + 1 + 2 * blk, RAD_I32, 2);
            const int64_t R = rows, no = g[2 + 2 * (blk + 1)];
            const rad_buf S = m.b_ax[blk];
            for (int r = 0; r < 3; ++r) {
                const Model::AEncUnit& U = A.u[r];
                RAD_ISSUE_N(c, U.op_a1, R, brows(S, R), RAD_W(U.a1), brows(m.b_et1[blk], R),
                            RAD_NONE);
                RAD_ISSUE_N(c, U.op_c1, R, brows(m.b_et1[blk], R), RAD_W(U.c1w), cu,
                            RAD_NONE, brows(m.b_et2[blk], R), RAD_W(U.c1b));
                RAD_ISSUE_N(c, U.op_a2, R, brows(m.b_et2[blk], R), RAD_W(U.a2),
                            brows(m.b_et1[blk], R), RAD_NONE);
                RAD_ISSUE_N(c, U.op_c2, R, brows(m.b_et1[blk], R), RAD_W(U.c2w), cu,
                            brows(S, R), brows(S, R), RAD_W(U.c2b));
            }
            RAD_ISSUE_N(c, A.op_al, R, brows(S, R), RAD_W(A.al), brows(m.b_et1[blk], R),
                        RAD_NONE);
            RAD_ISSUE_N(c, A.op_ci, R, brows(m.b_et1[blk], R), brows(A.bi, R));
            RAD_ISSUE_N(c, A.op_dn, no, brows(A.bi, R), RAD_W(A.dnw), cu, RAD_NONE,
                        brows(A.bo, no), RAD_W(A.dnb));
            RAD_ISSUE_N(c, A.op_co, no, brows(A.bo, no), brows(m.b_ax[blk + 1], no));
            rows = no;
        }
        RAD_ISSUE_N(c, m.op_ae_snk6, rows, brows(m.b_ax[5], rows), RAD_W(m.ae_al6),
                    brows(m.b_a6t, rows), RAD_NONE);
        RAD_ISSUE_N(c, m.op_ae_fin, rows, brows(m.b_a6t, rows), RAD_W(m.ae_finw), cu6,
                    RAD_NONE, brows(m.b_ax6, rows), RAD_W(m.ae_finb));
        RAD_ISSUE_N(c, m.op_ae_cast6, rows, brows(m.b_ax6, rows), brows(m.b_apb, rows));
        /* The attention block. AttnProjection.forward feeds the attention from norm1 and the
         * projection from norm3, and the two differ: each reads its own through b_aln3. */
        RAD_ISSUE_N(c, m.p_op_ln1, rows, brows(m.b_apb, rows), RAD_W(m.p_ln1w), RAD_W(m.p_ln1b),
                    brows(m.b_aln3, rows));
        RAD_ISSUE_N(c, m.p_op_qkv, rows, brows(m.b_aln3, rows), RAD_W(m.p_qkvw), RAD_W(m.p_qkvb),
                    RAD_NONE, brows(m.b_aqkv, rows));
        for (int h = 0; h < 8; ++h) {
            /* The first head writes the sum, the next six add to it and the last divides it. */
            const rad_buf y = h == 0 ? m.b_ahsum : m.b_aatt;
            RAD_ISSUE_N(c, m.p_op_attn, rows, bcol(m.b_aqkv, h * hd, hd, rows),
                        bcol(m.b_aqkv, ld + h * hd, hd, rows),
                        bcol(m.b_aqkv, 2 * ld + h * hd, hd, rows), cu6, brows(y, rows));
            if (h > 0)
                RAD_ISSUE_N(c, h == 7 ? m.p_op_hsd : m.p_op_hs, rows, brows(m.b_ahsum, rows),
                            brows(m.b_aatt, rows), brows(m.b_ahsum, rows));
        }
        RAD_ISSUE_N(c, m.p_op_pool, rows, brows(m.b_ahsum, rows), brows(m.b_apool, rows));
        RAD_ISSUE_N(c, m.p_op_aproj, rows, brows(m.b_apool, rows), RAD_W(m.p_apw), RAD_W(m.p_apb),
                    RAD_NONE, brows(m.b_a32a, rows));
        RAD_ISSUE_N(c, m.p_op_ln3, rows, brows(m.b_apb, rows), RAD_W(m.p_ln3w), RAD_W(m.p_ln3b),
                    brows(m.b_aln3, rows));
        RAD_ISSUE_N(c, m.p_op_proj, rows, brows(m.b_aln3, rows), RAD_W(m.p_prw), RAD_W(m.p_prb),
                    RAD_NONE, brows(m.b_eh, rows));
        RAD_ISSUE_N(c, m.p_op_hadd, rows, brows(m.b_eh, rows), brows(m.b_a32a, rows),
                    brows(m.b_a32b, rows));
        RAD_ISSUE_N(c, m.p_op_ln2, rows, brows(m.b_a32b, rows), RAD_W(m.p_ln2w), RAD_W(m.p_ln2b),
                    brows(m.b_a32c, rows));
        RAD_ISSUE_N(c, m.p_op_mln, rows, brows(m.b_a32c, rows), RAD_W(m.p_mlw), RAD_W(m.p_mlb),
                    brows(m.b_a32d, rows));
        RAD_ISSUE_N(c, m.p_op_gu, rows, brows(m.b_a32d, rows), RAD_W(m.p_guw), RAD_W(m.p_gub),
                    RAD_NONE, brows(m.b_agu, rows));
        RAD_ISSUE_N(c, m.p_op_act, rows, brows(m.b_agu, rows), brows(m.b_agu64, rows));
        RAD_ISSUE_N(c, m.p_op_w2, rows, brows(m.b_agu64, rows), RAD_W(m.p_w2w), RAD_W(m.p_w2b),
                    RAD_NONE, brows(m.b_a32c, rows));
        RAD_ISSUE_N(c, m.p_op_xadd, rows, brows(m.b_a32b, rows), brows(m.b_a32c, rows),
                    brows(m.b_ax2, rows));
        RAD_ISSUE_N(c, m.op_ae_castx, rows, brows(m.b_ax2, rows), brows(m.b_amb, rows));
        RAD_ISSUE_N(c, m.op_ae_mp, rows, brows(m.b_amb, rows), RAD_W(m.ae_mpw), cu6,
                    RAD_NONE, brows(m.b_amb2, rows), RAD_W(m.ae_mpb));
        RAD_ISSUE_N(c, m.op_ae_norm, rows, brows(m.b_amb2, rows),
                    brow_slice(m.b_enca, row0 + (int64_t)ch * (kEncSlotRowsA / (int64_t)kAudioCh),
                               rows, kAudioC));
    }
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_slot(c)];
    if (!batch->gen || !batch->vgen_b) {
        rad_step_fail(c, "minimaxh3 is a video generator and runs its generator passes only");
        return;
    }
    switch (batch->gen) {
    case RAD_GEN_PASS_VCOND:   step_cond(c, batch, m); return;
    case RAD_GEN_PASS_VSTEP:   step_dit(c, batch, m); return;
    case RAD_GEN_PASS_ADECODE: step_audio(c, batch, m); return;
    case RAD_GEN_PASS_VDECODE: step_video(c, batch, m); return;
    case RAD_GEN_PASS_VUPSCALE:
        if (!m.ups) {
            rad_step_fail(c, "minimaxh3: an upscale pass, and this container has no latent "
                             "upscaler");
            return;
        }
        step_upscale(c, batch, m);
        return;
    case RAD_GEN_PASS_VENCODE:
        if (!m.venc) {
            rad_step_fail(c, "minimaxh3: a Vencode pass, and this container has no video VAE "
                             "encoder");
            return;
        }
        step_venc(c, batch, m);
        return;
    case RAD_GEN_PASS_AENCODE:
        if (!m.aenc) {
            rad_step_fail(c, "minimaxh3: an Aencode pass, and this container has no audio VAE "
                             "encoder");
            return;
        }
        step_aenc(c, batch, m);
        return;
    default:
        rad_step_fail(c, "minimaxh3: a generator pass of a kind this plugin does not know");
    }
}

/* ================================================================== the probe */
/* The rows of the largest pass: a step of the longest video at the largest size, or a decode of
 * the most tiles. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    const int64_t area = video_area(out->gen_max_pixels);
    if (area < kMinSide * kMinSide) return RAD_E_INVAL;
    out->gen_rows = std::max<int64_t>(max_step_rows(area), max_tiles(area) * kPeriod);
    out->gen_video = 1;
    return RAD_OK;
}

}  /* namespace minimaxh3 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(minimaxh3)
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(minimaxh3, "minimaxh3", "", "0.1.0",
                "MiniMax-H3 (t2va, i2va, l2va, fl2va, ref2va): Qwen3-VL-32B conditioning, a 33B "
                "single-stream diffusion transformer over text, soundtrack and video, a tiled ViT "
                "video decoder, a BigVGAN audio decoder and the VAEs' encoders; block-fp8 or "
                "int8 linears")

extern "C" int64_t rad_arch_video_positions(int which, const RadVideoGeom* g, float* pos,
                                            int64_t cap) {
    if (!g || g->n_text < 1 || g->n_text > minimaxh3::kMaxText) return RAD_E_INVAL;
    if (which == RAD_VPOS_STEP) return minimaxh3::positions(g, pos, cap);
    if (which == RAD_VPOS_DECODE) return minimaxh3::decode_rows(g);
    return RAD_E_INVAL;
}

/* The encode passes' geometry (kVEncGeomN, kAEncGeomN above); the steps derive it again from the
 * batch's own fields. */
extern "C" int64_t rad_arch_encode_geom(int32_t kind, const int64_t* in, int32_t* out,
                                        int64_t cap) {
    using namespace minimaxh3;
    if (!in) return RAD_E_INVAL;
    const bool video = kind == RAD_GEN_PASS_VENCODE;
    if (!video && kind != RAD_GEN_PASS_AENCODE) return RAD_E_INVAL;
    const int64_t n = video ? kVEncGeomN : kAEncGeomN;
    if (out && cap < n) return RAD_E_INVAL;
    const bool ok = video ? venc_geom(in[0], in[1], in[2], out) : aenc_geom(in[0], in[3], out);
    return ok ? n : (int64_t)RAD_E_INVAL;
}

extern "C" int64_t rad_arch_video_pack(const RadVideoGeom* g, const int32_t* ref_tab,
                                       int32_t* seg, int32_t* n_seg, int32_t* mod,
                                       int32_t* n_mod, float* pos, int64_t cap_pos) {
    using namespace minimaxh3;
    if (!g || !ref_tab || g->n_text < 1 || g->n_text > kMaxText || g->lat_frames < 1 ||
        g->lat_h < kPatch || g->lat_w < kPatch || g->n_audio < 1)
        return RAD_E_INVAL;
    Ref2VAPack p;
    if (!ref_pack_of(g, ref_tab, &p, pos != nullptr)) return RAD_E_INVAL;
    Ref2VAModGroups grp;
    ref2va_mod_groups(p, 1, ref_pads_of(ref_tab), &grp);
    const int32_t ns = (int32_t)p.segs.size() + 1, nb = (int32_t)grp.bound.size();
    const int32_t nm = nb + (int32_t)grp.col.size();
    if ((seg && (!n_seg || *n_seg < ns)) || (mod && (!n_mod || *n_mod < nm)) ||
        (pos && cap_pos < 3 * p.rows))
        return RAD_E_INVAL;
    if (seg) {
        for (int32_t k = 0; k + 1 < ns; ++k) seg[k] = (int32_t)p.segs[(size_t)k].start;
        seg[ns - 1] = (int32_t)p.rows;
    }
    if (mod) {
        for (int32_t k = 0; k < nb; ++k) mod[k] = (int32_t)grp.bound[(size_t)k];
        for (size_t k = 0; k < grp.col.size(); ++k) mod[nb + k] = grp.col[k];
    }
    if (pos) std::copy(p.pos.begin(), p.pos.end(), pos);
    if (n_seg) *n_seg = ns;
    if (n_mod) *n_mod = nm;
    return p.rows;
}
#endif
