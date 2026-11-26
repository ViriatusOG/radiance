/* fishomni -- Fish Audio drama-3 (FishOmniModel): speech from text, 44.1 kHz mono.
 *
 * ============================== THE MODEL ==============================
 *
 * Three networks (fish-speech's fish_speech/models/text2semantic/fish_omni/, Fish Audio's own
 * reference, and its S2 codec):
 *
 *   Backbone: the language model of Qwen3.5-35B-A3B, unchanged -- 40 layers of 2048, three gated
 *   delta-net layers to one gated attention layer (16 query heads over 2 KV heads of 256, NeoX
 *   rotary on 64 dims at theta 1e7), a routed feed-forward in every layer (256 experts of 512, 8 a
 *   token, and a gated shared expert of 512), (1 + w) norms. Only the routed experts use fp8 in
 *   the source checkpoint (`quant_method: bsgemm-moe`, E4M3 with one f32 scale per 128 x 128
 *   block). The int8 recipe converts these experts and keeps the rest of the backbone in bf16.
 *   A prompt row is a token; an audio row is one frame of the codec's ten codes, embedded as
 *   (E_lm[248099 + c0] + E_cb[4096 + c1]) / sqrt 2: the first code through its semantic token,
 *   the second through section 1 of the codebook table.
 *
 *   Tower: 13 MTP-style layers that write a frame's codes 1..9 from the backbone's hidden state:
 *   four "shared" layers that inject code 0, then nine "per-codebook" layers, layer k injecting
 *   code k - 1 and its head drawing code k. Each layer is m = eh_proj([enorm(e); hnorm(h)]), one
 *   pre-norm block (gated attention with (1 + w) q/k norms and GPT-J rotary on 64 dims at 1e7, a
 *   SwiGLU of 6144), and a final norm. The attention is causal across time, over a cache of its
 *   own at the backbone's positions: the layers are bound as layers 40..52 of the backbone's
 *   full-attention KV group. The first shared layer also adds the summed embeddings of the input
 *   frame's codes 2..9, the ones the backbone never saw.
 *
 *   Codec: Fish S2's modified DAC decoder: ten codebooks of dimension 8 (one of 4096, nine of
 *   1024), each projected to 1024 and summed; an 8-layer transformer (16 heads of 64, a causal
 *   window of 128 frames); two 2x upsamplings with ConvNeXt blocks; four transposed convolutions
 *   (8, 8, 4, 2) with snake residual units; 2048 samples a frame. Its encoder, for a reference
 *   clip, mirrors it: four blocks of snake residual units and strided causal convolutions (2, 4,
 *   8, 8) with a 4-layer transformer (window 512 rows) after the last, two 2x downsamplings with
 *   ConvNeXt blocks, an 8-layer transformer (window 128 frames), and a residual quantiser,
 *   nearest code by cosine, the semantic codebook first.
 *
 * ============================== THE PASSES ==============================
 *
 * The core's audio generator in its chunked form, one branch a request (no guidance):
 *
 *   PROMPT  the backbone over the whole prompt; the last row's hidden state into the lane; the
 *           tower teacher-forced over every row. A text prompt's labels are all text, so every
 *           layer's code embedding is zero; the tower still runs, because its attention reads
 *           those rows' keys and values from every later frame. With references the codec's
 *           encoder first turns their clips into frames, each a prompt row embedded as a made
 *           frame is; a row before a frame is labelled with it, a frame's row carries its unseen
 *           codes, and the window of main picks starts with the prompt's last frames. The last
 *           row's tower pass is the first frame's, and FRAME 0 overwrites it.
 *   FRAME   frame j of every lane in flight. The main token is drawn from the hidden state of the
 *           row before the frame's (the prompt's last row for frame 0) over 4097 candidates (the
 *           end of speech and the 4096 semantic tokens), with Fish's repetition-aware choice
 *           between two draws; then the tower at that row's position draws codes 1..9; then the
 *           frame goes into the lane's history and the backbone runs over its embedding at the
 *           frame's own row.
 *   VOCODE  a few kept frames of one lane through the codec, re-decoding the frames its outputs
 *           reach back to (every layer is causal), so no state is carried between chunks.
 *
 * ============================== TENSOR PARALLEL ==============================
 *
 * Two ranks. The backbone splits as Qwen3.6-35B-A3B does (heads, delta-net heads, expert width).
 * The tower splits by heads and feed-forward width, an all-reduce after each output projection.
 * The candidate head, the code tables, the nine heads, the samplers and the codec are
 * replicated: every rank draws the same codes from the same logits and seeds, and plays the same
 * samples.
 */
#include <arch/rad_arch.h>
#include <arch/rad_block_attn_gated_q.h>
#include <arch/rad_block_gdn_q.h>
#include <arch/rad_block_moe_q.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace fishomni {

using namespace rad::arch;

/* ================================================================== the model's constants */
constexpr int64_t kMaxLanes = 4;
/* References (voice cloning): at most 8 clips of 30 s together. fish-speech's server takes each
 * up to 60 s; here the encoder's first stages, 64 and 128 channels at the samples' rate, hold
 * about 170 MB a buffer at 30 s. A clip's frames are prompt rows. */
constexpr int64_t kMaxRefFrames = 646;
constexpr int32_t kMaxRefs = 8;
/* The longest prompt and speech. fish-speech's default request is 2048 frames (95 s); a plain
 * prompt is a few dozen rows a sentence, and the references add a row a frame. */
constexpr int64_t kMaxPrompt = 3072, kMaxFrames = 2048;
/* A frame: ten codes, code 0 of 4096 (the semantic band), codes 1..9 drawn from heads of 4096 and
 * read by the codec through 1024-entry codebooks. */
constexpr int64_t kCodes = 10, kVocab = 4096, kResVocab = 1024;
/* Token ids: the end of speech (<|im_end|>) and the semantic band <|semantic:0..4095|>. The main
 * token is drawn over those 4097, in id order, so candidate 0 is the end and candidate c + 1 is
 * semantic code c. */
constexpr int64_t kEos = 248046, kSemBegin = 248099, kSemEnd = 252194, kCand = kVocab + 1;
/* The tower: four shared layers, then one a codebook after the first. */
constexpr int kShared = 4, kTower = kShared + (int)kCodes - 1;
/* The draws: temperature 1.0 and top-p 0.9, the settings of Fish Audio's model card, for the main
 * token and every head; the main token uncapped (256, cfg_topk_sample's widest) and each head at
 * top-k 30, as SGLang-Omni draws them. fish-speech's server defaults (0.7, 0.7, 30 for both) make
 * the model fall silent after a tag or a speaker's line and never come back: over 40 takes of
 * four short scripts, 8 ran to their length in silence against none here
 * (docs/ARCHITECTURES.md). */
constexpr int64_t kMainTopK = 256, kHeadTopK = 30;
static const double kTemp = 1.0, kTopP = 0.9;
/* The repetition-aware redraw, fixed by the training recipe (tower.py): when the first draw is
 * semantic and among the last 16 main tokens, a second draw at temperature 1, top-p 0.95,
 * top-k 30 is taken instead. */
constexpr int64_t kRasWin = 16, kRasTopK = 30;
static const double kRasTemp = 1.0, kRasTopP = 0.95;

/* The codec: 2048 samples a frame at 44.1 kHz. A chunk is at most kVocChunk frames; its outputs
 * reach back about nine frames through the convolutions after the transformer (kConvCtx, rounded
 * up), whose rows reach back kAttnWin - 1 frames. The dequantiser is per frame. */
constexpr int64_t kSamplesPerFrame = 2048, kSampleRate = 44100;
constexpr int64_t kMaxRefSamples = kMaxRefFrames * kSamplesPerFrame;   /* 30.0 s */
constexpr int64_t kVocChunk = 16, kConvCtx = 12, kAttnWin = 128;
constexpr int64_t kVocContext = (kAttnWin - 1) + kConvCtx;
constexpr int64_t kConvFrames = kConvCtx + kVocChunk;
constexpr int64_t kCbDim = 8, kLat = 1024, kDec = 1536;
/* The codec transformer: 8 layers of 1024, 16 heads of 64, a SwiGLU of 3072. */
constexpr int64_t kCtLayers = 8, kCtHeads = 16, kCtHeadDim = 64, kCtFf = 3072;
static const double kCtTheta = 10000.0, kCtEps = 1e-5;
/* The encoder (modded_dac.py's Encoder): four blocks, each three residual units and a strided
 * causal conv (kernel two strides), a 4-layer transformer over a window of 512 rows after the
 * last; the quantiser then halves the rate twice to the frames (rvq.py). Its rows are 1, 2, 8, 64
 * and 512 samples, then 1024 and 2048. */
static const int64_t kEncStride[4] = { 2, 4, 8, 8 };
constexpr int64_t kEncWin = 512, kEncLayers = 4;
/* The codec's sequence rates, rows a frame: the transformer's frames, the convolutions' frames,
 * the two upsamplings', and each decoder block's output (8, 8, 4, 2 times). */
constexpr int kRates = 8;
static const int64_t kRate[kRates] = { 1, 1, 2, 4, 32, 256, 1024, 2048 };

/* The prompt (reference-bracket-v1, the plain form: generate.py's build_drama3_prompt with no
 * references). Every literal there is tokenised on its own; at these seams the joined text
 * tokenises the same, since each one falls at a special token or a newline. The input gets its
 * speaker tag in text_prepare. */
static const char* kPlain =
    "<|im_start|>system\nconvert the provided text to speech<|im_end|>\n"
    "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n<|voice|>";
/* With references: the system turn's instruction for them, then each one's numbered header, its
 * transcript and its frames (the core tokenises the instruction apart from the first header, as
 * fish-speech does: joined, ":\n\n" is one token), then the user turn as kPlain's. The
 * transcript's speaker tag comes first in it, from text_prepare. */
static const char* kRefPlain =
    "<|im_start|>system\nconvert the provided text to speech reference to the following:"
    "{references}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
    "<|voice|>";
static const char* kRefItem = "\n\n[Reference {index}]\nText: {text}\nSpeech: {audio}";

/* Weight groups (movement units): the backbone's layers are 0..39 (the blocks' own), the tower's
 * 40..52 like their KV layers, then the frame's tables and the codec. */
enum : int { kGrpTower = 40, kGrpFrame = 53, kGrpCodec = 100 };

/* ================================================================== the networks */
struct Layer {
    int          full = 0;
    AttnGatedQ attn;
    GdnQ       gdn;
    MoeQ       mlp;
};

/* A tower layer (tower.py's TowerLayer): the code embedding and the two norms into eh_proj, the
 * gated attention block over the paged cache, the SwiGLU, the final norm; a per-codebook layer's
 * head and its draw. q, gate, k and v are four projections of the checkpoint's [Q | G | K | V]
 * rows (paged attention reads a contiguous query, and the gate is not interleaved per head as the
 * backbone's is). */
struct TowerLayer {
    rad_weight w_emb = 0, w_en = 0, w_hn = 0, w_eh = 0, w_an = 0;
    rad_weight w_q = 0, w_g = 0, w_k = 0, w_v = 0, w_qn = 0, w_kn = 0, w_o = 0;
    rad_weight w_fn = 0, w_gu = 0, w_down = 0, w_fin = 0, w_head = 0;
    rad_op op_emb = 0, op_eadd = 0, op_en = 0, op_hn = 0, op_eh = 0, op_an = 0;
    rad_op op_q = 0, op_g = 0, op_k = 0, op_v = 0, op_qn = 0, op_kn = 0, op_rq = 0, op_rk = 0;
    rad_op op_attn = 0, op_kv = 0, op_sig = 0, op_mul = 0, op_o = 0, op_ar1 = 0, op_add1 = 0;
    rad_op op_fn = 0, op_gu = 0, op_act = 0, op_down = 0, op_ar2 = 0, op_add2 = 0, op_fin = 0;
    rad_op op_head = 0, op_samp = 0;
    /* PROMPT with references: the label embedding over the prompt's rows (and layer 0's unseen
     * codes), and its norm. */
    rad_op op_pemb = 0, op_peadd = 0, op_pen = 0;
};

/* The codec transformer's layer: fused q|k|v with rotary at the frames' positions, causal
 * attention over a 128-frame window, layer scales on both residual branches. */
struct CodecLayer {
    rad_weight w_n1 = 0, w_n2 = 0, w_qkv = 0, w_o = 0, w_gu = 0, w_down = 0, w_ls1 = 0, w_ls2 = 0;
    rad_op     op_n1 = 0, op_qkv = 0, op_rope = 0, op_attn = 0, op_o = 0, op_ls1 = 0, op_add1 = 0;
    rad_op     op_n2 = 0, op_gu = 0, op_act = 0, op_down = 0, op_ls2 = 0, op_add2 = 0;
};

/* A 2x upsampling and its ConvNeXt block. */
struct Upsample {
    rad_weight w_t = 0, b_t = 0, w_dw = 0, b_dw = 0, w_ln = 0, b_ln = 0, w_p1 = 0, b_p1 = 0;
    rad_weight w_p2 = 0, b_p2 = 0, w_g = 0;
    rad_op     op_t = 0, op_dw = 0, op_ln = 0, op_p1 = 0, op_gelu = 0, op_p2 = 0, op_g = 0;
    rad_op     op_add = 0;
};

/* A causal weight-normalised conv with its bias. */
struct VConv {
    rad_weight w = 0, b = 0;
    rad_op     op = 0;
};
struct VRes {
    rad_weight a1 = 0, a2 = 0;
    rad_op     op_s1 = 0, op_s2 = 0;
    VConv      c1{}, c2{};
};
struct VBlock {
    rad_weight a = 0, wt = 0, bt = 0;
    rad_op     op_s = 0, op_t = 0;
    VRes       res[3];
};
/* An encoder block: three residual units, a snake, the strided conv. */
struct EBlock {
    VRes       res[3];
    rad_weight a = 0;
    rad_op     op_s = 0;
    VConv      down{};
};
/* The quantiser's 2x downsampling: a strided causal conv (plain weights) and a ConvNeXt block. */
struct EDown {
    VConv      c{};
    rad_weight w_dw = 0, b_dw = 0, w_ln = 0, b_ln = 0, w_p1 = 0, b_p1 = 0, w_p2 = 0, b_p2 = 0;
    rad_weight w_g = 0;
    rad_op     op_dw = 0, op_ln = 0, op_p1 = 0, op_gelu = 0, op_p2 = 0, op_g = 0, op_add = 0;
};
/* A residual quantiser step's weights, f32: the in-projection, the codebook, the out-projection. */
struct EQuant {
    rad_weight in_w = 0, in_b = 0, cb = 0, out_w = 0, out_b = 0;
    rad_op     op = 0;
};
/* A codec transformer's working set: its residual stream and scratch. */
struct CtSet { rad_buf x = 0, h = 0, qkv = 0, attn = 0, o = 0, gu = 0, ff = 0; };

struct Model {
    Names nm{""};
    Geom  g{};                      /* the backbone's */
    Geom  gt{};                     /* the tower's */
    GdnQ::Config gcfg{};
    MoeQ::Config moecfg{};
    int64_t n_rot = 0, n_rot_t = 0, n_main = 0, lanes = 0;
    rad_kvgroup kv_full = 0, kv_state = 0, kv_conv = 0;

    /* ---- the backbone: the residual stream, its normed copy, and the blocks' working set */
    rad_buf b_x = 0;
    ActQ  a_h{};
    rad_buf b_qg = 0, b_k = 0, b_v = 0, b_q = 0, b_rope_cs = 0;
    ActQ  a_attn{};
    ActQ  a_in{}, a_go{};
    rad_buf b_ab = 0, b_gq = 0, b_gk = 0, b_gv = 0, b_gdec = 0, b_beta = 0, b_kkt = 0;
    rad_buf b_rlogits = 0, b_eids = 0, b_ew = 0, b_sorted = 0, b_eoff = 0, b_ecnt = 0;
    rad_buf b_egu = 0, b_edn = 0, b_emoe = 0;
    ActQ  a_eff{};
    rad_buf b_sgu = 0, b_sout = 0, b_sgate = 0;
    ActQ  a_sff{};
    rad_weight w_tok = 0, w_out_norm = 0;
    rad_op  op_embed = 0, op_embed_ar = 0, op_final_norm = 0, op_rope_cs = 0;
    std::vector<Layer> layers;

    /* ---- a lane's state (PERSIST) */
    rad_buf b_hid = 0;            /* [lanes, 2048] the last row's final-norm hidden state */
    rad_buf b_status = 0;         /* [lanes] i32 */
    rad_buf b_hist = 0;           /* [lanes * 10, kMaxFrames] i32: each lane's codes, by codebook */
    rad_buf b_ras = 0;            /* [lanes, 16] i32: the last 16 main picks (candidate indices) */
    rad_buf b_u = 0;              /* [lanes, 2048] the last frame's codes 2..9, embedded, summed */

    /* ---- the frame */
    rad_weight w_cand = 0;        /* [4097, 2048] the lm_head's candidate rows */
    rad_weight w_cemb = 0;        /* [4097, 2048] the embedding's candidate rows */
    rad_weight w_cb = 0;          /* [9 * 4096, 2048] codebook table sections 1..9 */
    rad_buf b_plane = 0, b_slog = 0, b_first = 0, b_alt = 0, b_codes = 0, b_ug = 0, b_unew = 0;
    rad_buf b_last = 0, b_hlog = 0;
    rad_op  op_hsel = 0, op_head = 0, op_samp = 0, op_samp_ras = 0, op_ras = 0, op_status = 0;
    rad_op  op_ugather = 0, op_plast = 0, op_pstore = 0, op_uzero = 0, op_rasinit = 0;
    rad_op  op_efill = 0, op_store = 0, op_u = 0, op_uscat = 0, op_fb = 0, op_fstore = 0;

    /* ---- the tower */
    TowerLayer tw[kTower];
    rad_buf b_te = 0, b_tcat = 0, b_tm = 0, b_th = 0, b_tq = 0, b_tg = 0, b_tk = 0, b_tv = 0;
    rad_buf b_tattn = 0, b_to = 0, b_tgu = 0, b_tff = 0, b_tho = 0;

    /* ---- the codec (VOCODE) */
    rad_weight w_cbk[kCodes] = {};
    rad_op  op_rq[kCodes] = {};
    VConv   qo[kCodes];
    std::vector<CodecLayer> ct;
    rad_weight w_cn = 0;
    rad_op  op_cn = 0, op_cnst = 0, op_ccu[kRates] = {}, op_fcast = 0;
    Upsample up[2];
    VConv   dec0{};
    VBlock  vb[4];
    int64_t v_ups[4] = { 8, 8, 4, 2 };
    rad_weight w_fa = 0;
    rad_op  op_fs = 0;
    VConv   fin{};
    rad_buf b_rq = 0, b_cx = 0, b_ch = 0, b_cqkv = 0, b_cattn = 0, b_co = 0, b_cgu = 0, b_cff = 0;
    rad_buf b_cconst = 0, b_ccu[kRates] = {};
    rad_buf b_ux[3] = {}, b_ua = 0, b_ub = 0, b_uh = 0, b_ug2 = 0;
    rad_buf b_vs[5] = {}, b_vt[5] = {}, b_va[5] = {}, b_vb[5] = {};
    rad_buf b_vout = 0, b_audio = 0;

    /* ---- the codec's encoder (PROMPT with references): the clips in, their frames' codes out
     * into b_pc, the prompt rows' codes ([10, kMaxPrompt + 1] code-major, -2 and -1 at a token's
     * row), from which the rows' embeddings and the tower's labels are read */
    rad_op  op_ecast = 0, op_epos = 0, op_e5 = 0, op_etn = 0, op_pqn = 0;
    VConv   e0{}, e6{};
    EBlock  eb[4];
    std::vector<CodecLayer> et, pq;
    rad_weight w_etn = 0, w_e5 = 0, w_pqn = 0;
    EDown   ed[2];
    EQuant  eq[kCodes];
    rad_op  op_pc0 = 0, op_pc1 = 0, op_pfb = 0, op_pufill = 0, op_pu = 0, op_ptefill = 0;
    rad_op  op_rascopy = 0;
    rad_buf b_ein = 0, b_ex[4] = {}, b_et[4] = {}, b_ea[4] = {}, b_ex4 = 0, b_et4 = 0, b_ea4 = 0;
    CtSet   ecs{};
    rad_buf b_epos = 0, b_ed1 = 0, b_ez = 0, b_dua = 0, b_dub = 0, b_duh = 0, b_dug = 0;
    rad_buf b_pc = 0, b_pte = 0, b_pu = 0;
};

/* The width of the prompt rows' code plane: every row and one past the last, a token's (the last
 * row's label). */
constexpr int64_t kPcW = kMaxPrompt + 1;

static Model g_model[MAX_SLOTS];

/* ================================================================== declare helpers */
static rad_op norm_op(RadBuilder* b, int64_t rows, int64_t n, double eps, double wadd,
                      rad_weight w, std::initializer_list<rad_buf> x, rad_buf y) {
    return rw(b, RAD_OP(b, "rmsnorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                   RAD_F64("wadd", wadd), RAD_STR("dtype", "bf16")),
                        RAD_WEIGHTS(w)),
              x, { y });
}

static rad_op cu_op(RadBuilder* b, int64_t mult, rad_buf len, rad_buf y) {
    return rw(b, RAD_OP(b, "cu_repeat",
                        RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("mult", mult)),
                        RAD_NOWEIGHTS),
              { len }, { y });
}

static rad_op ar_op(RadBuilder* b, const Geom& g, rad_buf x) {
    if (g.world <= 1) return 0;
    return decl_all_reduce(b, g, x);
}

/* A frame draw: `k` largest, Fish's nucleus over the whole candidate set first (rad OPS.md,
 * cfg_topk_sample's top_p), one row a lane, no guidance. */
static rad_op draw_op(RadBuilder* b, int64_t lanes, int64_t n, int64_t k, double temp,
                      double top_p, int stream, rad_buf logits, rad_buf ids) {
    return rw(b, RAD_OP(b, "cfg_topk_sample",
                        RAD_PARAMS(RAD_RANGE("M", 1, lanes), RAD_INT("n_vocab", n),
                                   RAD_INT("k", k), RAD_INT("rep", 1), RAD_F64("cfg", 1.0),
                                   RAD_F64("temp", temp), RAD_INT("stream", stream),
                                   RAD_STR("dtype", "f32"), RAD_F64("top_p", top_p)),
                        RAD_NOWEIGHTS),
              { logits }, { ids });
}

/* A plain weight copied from the checkpoint, declared at `dtype`, replicated. */
static rad_weight wcopy(RadBuilder* b, const char* dn, const char* src, uint32_t dtype,
                        std::initializer_list<int64_t> shape, int group) {
    if (map_copy(b, dn, src) < 0) return 0;
    return decl_w(b, dn, dtype, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(group));
}

/* ================================================================== check_format */
/* The container's statement of what it is. The metadata describes Fish's source checkpoint,
 * which a conversion carries over unchanged; the converted experts' encoding is read from the
 * weights in declare_trunk. Every mismatch below would still make sound -- a different template,
 * another codebook layout, another release's experts read as this one -- so each is a refusal by
 * name rather than a silent default. */
static int check_format(const RadModelMeta* meta) {
    const char* method = rad_meta_gets(meta, "quantization_config.quant_method", nullptr);
    const char* block  = rad_meta_gets(meta, "quantization_config.weight_block_size", nullptr);
    if (!method || std::strcmp(method, "bsgemm-moe") != 0) {
        fprintf(stderr, "radiance: fishomni serves Fish Audio's released drama-3, whose source "
                        "experts use quant_method 'bsgemm-moe'; this checkpoint "
                        "says '%s'\n", method ? method : "(none)");
        return RAD_E_UNSUPPORTED;
    }
    if (block) {
        long long b0 = 0, b1 = 0;
        if (std::sscanf(block, "%lld %lld", &b0, &b1) != 2) { b0 = 0; b1 = 0; }
        if (b0 != RAD_SCALE_GROUP || b1 != RAD_SCALE_GROUP) {
            fprintf(stderr, "radiance: fishomni: source weight_block_size [%lld, %lld] differs "
                            "from drama-3's %d in both dimensions\n",
                    b0, b1, RAD_SCALE_GROUP);
            return RAD_E_UNSUPPORTED;
        }
    }
    const char* tv = rad_meta_gets(meta, "speech_template_version", "");
    if (std::strcmp(tv, "reference-bracket-v1") != 0) {
        fprintf(stderr, "radiance: fishomni builds the reference-bracket-v1 prompt and this "
                        "checkpoint asks for '%s'\n", tv);
        return RAD_E_UNSUPPORTED;
    }
    const char* ad = "audio_decoder_config";
    char k[96];
    auto ai = [&](const char* key, long long dflt) {
        std::snprintf(k, sizeof k, "%s.%s", ad, key);
        return rad_meta_geti(meta, k, dflt);
    };
    auto as = [&](const char* key) {
        std::snprintf(k, sizeof k, "%s.%s", ad, key);
        return rad_meta_gets(meta, k, "");
    };
    if (rad_meta_geti(meta, "semantic_start_token_id", 0) != kSemBegin ||
        rad_meta_geti(meta, "semantic_end_token_id", 0) != kSemEnd ||
        rad_meta_geti(meta, "eos_token_id", 0) != kEos ||
        std::strcmp(as("decoder_type"), "codebook_mtp_shared") != 0 ||
        ai("num_codebooks", 0) != kCodes || ai("vocab_size", 0) != kVocab ||
        ai("vq_backbone_codebooks", 0) != 1 || ai("shared_enrichment_layers", 0) != kShared ||
        std::strcmp(as("vq_feature_placement"), "backbone_partial") != 0 ||
        std::strcmp(as("vq_feature_scope"), "all_vq") != 0) {
        fprintf(stderr, "radiance: fishomni: not the released drama-3 layout (semantic tokens "
                        "%lld..%lld, end %lld, a codebook_mtp_shared tower of %d shared layers "
                        "over %lld codebooks of %lld, one codebook into the backbone)\n",
                (long long)kSemBegin, (long long)kSemEnd, (long long)kEos, kShared,
                (long long)kCodes, (long long)kVocab);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* ================================================================== the backbone */
/* qwen35moe_fp8's trunk, written out (arch/qwen35_fp8/qwen35_fp8.cpp, QWEN35_MOE): the same
 * blocks, metadata and names. Its bf16 linears (attention, delta net and shared expert) read
 * the plain activations. Only the routed experts use the converted encoding. */
static int declare_trunk(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                         Model& m) {
    Geom& g = m.g;
    Names& nm = m.nm;
    if (meta->n_expert <= 0) {
        fprintf(stderr, "radiance: fishomni: the backbone declares no experts; drama-3's is "
                        "routed\n");
        return RAD_E_UNSUPPORTED;
    }
    bool have_interval = false;
    const int64_t interval = meta_int2(meta, "qwen35.full_attention_interval",
                                       "full_attention_interval", 4, &have_interval);
    if (interval <= 0) return RAD_E_INVAL;
    RAD_ARCH_TRY(check_layer_schedule("fishomni", meta_text_gets(meta, "layer_types", nullptr),
                                      g.n_layer, interval, {"full_attention"}));
    m.n_main = g.n_layer;

    /* GemmaRMSNorm everywhere but the delta net's output norm (rad_block_gdn_q.h). */
    g.wadd    = 1.0f;
    g.wadd_qk = 1.0f;

    double prf = rad_meta_getf(meta, "partial_rotary_factor", 0.0);
    if (!(prf > 0.0)) prf = rad_meta_getf(meta, "rope_parameters.partial_rotary_factor", 0.0);
    m.n_rot = prf > 0.0 ? (int64_t)(g.head_dim * prf) : 0;
    if (m.n_rot <= 0 || m.n_rot > g.head_dim || (m.n_rot & 1)) {
        fprintf(stderr, "radiance: fishomni: no usable rotary width (head_dim %lld)\n",
                (long long)g.head_dim);
        return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(yarn_from(g, meta, "", m.n_rot));

    GdnQ::Config& gc = m.gcfg;
    const int64_t n_head_k_all = meta_int2(meta, "qwen35.ssm.group_count",
                                           "linear_num_key_heads", 0);
    const int64_t n_head_v_all = meta_int2(meta, "qwen35.ssm.time_step_rank",
                                           "linear_num_value_heads", 0);
    gc.head_k     = meta_int2(meta, "qwen35.ssm.state_size", "linear_key_head_dim", 0);
    gc.head_v     = meta_int2(meta, "qwen35.ssm.value_head_dim", "linear_value_head_dim", 0);
    gc.conv_width = meta_int2(meta, "qwen35.ssm.conv_kernel", "linear_conv_kernel_dim", 0);
    if (n_head_k_all <= 0 || n_head_v_all <= 0 || gc.head_k <= 0 || gc.head_v <= 0 ||
        gc.conv_width <= 0) {
        fprintf(stderr, "radiance: fishomni: the container does not describe the delta-net "
                        "layers\n");
        return RAD_E_INVAL;
    }
    gc.chunk = 64;
    RAD_ARCH_TRY(divide_or_fail(n_head_k_all, g.world, "linear_num_key_heads", &gc.n_head_k));
    RAD_ARCH_TRY(divide_or_fail(n_head_v_all, g.world, "linear_num_value_heads", &gc.n_head_v));

    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();
    const int64_t v_dim = gc.v_dim(), conv_dim = gc.conv_dim();

    /* ---- the KV groups. The full-attention group also holds the tower's 13 layers, bound below
     * at 40..52 (declare_tower). */
    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    g.kv_dtype = rad_dtype_name(fd.dtype);

    RadKVGroupDecl sd{};
    sd.kind         = RAD_KV_LINEAR;
    sd.dtype        = RAD_F32;
    sd.n_head_kv    = gc.n_head_v;
    sd.state_dim[0] = gc.head_v;
    sd.state_dim[1] = gc.head_k;
    m.kv_state = rad_decl_kv_group(b, nm.f("kv_gdn_state"), &sd);

    RadKVGroupDecl cd{};
    cd.kind       = RAD_KV_CONV;
    cd.dtype      = RAD_BF16;
    cd.n_head_kv  = 2 * gc.n_head_k + gc.n_head_v;
    cd.head_dim   = gc.head_k;
    cd.conv_width = gc.conv_width;
    m.kv_conv = rad_decl_kv_group(b, nm.f("kv_gdn_conv"), &cd);
    if (!m.kv_full || !m.kv_state || !m.kv_conv) return RAD_E_INVAL;

    /* ---- activations (qwen35_fp8.cpp says what each is) */
    m.b_x       = decl_b(b, nm.f("x"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.a_h.x     = decl_b(b, nm.f("h"),        g.act_dtype, {g.max_tok, g.n_embd});
    RAD_ARCH_TRY(m.a_h.declare_qs(b, nm, g, "h", g.n_embd));
    m.b_qg      = decl_b(b, nm.f("attn_qg"),  g.act_dtype, {g.max_tok, g.n_head, 2 * g.head_dim});
    m.b_k       = decl_b(b, nm.f("attn_k"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_v       = decl_b(b, nm.f("attn_v"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_q       = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, q_dim});
    m.a_attn.x  = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, q_dim});
    m.b_rope_cs = decl_b(b, nm.f("rope_cs"), RAD_F32, {rope_table_rows(g), m.n_rot},
                         RAD_BUF_PERSIST);
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "attn_out", q_dim));
    m.a_in.x    = decl_b(b, nm.f("gdn_in"),   g.act_dtype, {g.max_tok, conv_dim + v_dim});
    RAD_ARCH_TRY(m.a_in.declare_qs(b, nm, g, "gdn_in", conv_dim + v_dim));
    m.b_ab      = decl_b(b, nm.f("gdn_ab"),   g.act_dtype, {g.max_tok, gc.ab_dim()});
    m.b_gq      = decl_b(b, nm.f("gdn_q"),    g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    m.b_gk      = decl_b(b, nm.f("gdn_k"),    g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    m.b_gv      = decl_b(b, nm.f("gdn_v"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});
    m.b_gdec    = decl_b(b, nm.f("gdn_g"),    RAD_F32,     {g.max_tok, gc.n_head_v});
    m.b_beta    = decl_b(b, nm.f("gdn_beta"), RAD_F32,     {g.max_tok, gc.n_head_v});
    m.b_kkt     = decl_b(b, nm.f("gdn_kkt"),  RAD_BF16,    {g.max_tok, gc.n_head_v, gc.chunk});
    m.a_go.x    = decl_b(b, nm.f("gdn_o"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});
    RAD_ARCH_TRY(m.a_go.declare_qs(b, nm, g, "gdn_o", v_dim));

    /* ---- the routing geometry; `n_ff` is already the per-rank expert width */
    {
        MoeQ::Config& mc = m.moecfg;
        mc.n_expert = meta->n_expert;
        mc.top_k    = meta->n_expert_used > 0 ? meta->n_expert_used
                                              : meta_int2(meta, "qwen35.expert_used_count",
                                                          "num_experts_per_tok", 0);
        mc.n_ff_exp = g.n_ff;
        mc.norm_topk = (int)meta_int2(meta, "qwen35.expert_weights_norm", "norm_topk_prob", 1);
        const int64_t sh_all = meta_int2(meta, "qwen35.expert_shared_feed_forward_length",
                                         "shared_expert_intermediate_size", 0);
        if (sh_all > 0)
            RAD_ARCH_TRY(divide_or_fail(sh_all, g.world, "shared_expert_intermediate_size",
                                        &mc.n_ff_shared));
        const int64_t moe_rows = 2048;
        mc.rows = g.max_tok < moe_rows ? g.max_tok : moe_rows;
        if (mc.top_k <= 0 || mc.top_k > mc.n_expert) {
            fprintf(stderr, "radiance: fishomni: top_k %lld is not in [1, %lld]\n",
                    (long long)mc.top_k, (long long)mc.n_expert);
            return RAD_E_INVAL;
        }
        const int64_t R = g.max_tok, K = mc.rows * mc.top_k;
        m.b_rlogits = decl_b(b, nm.f("moe_logits"), g.act_dtype, {R, mc.n_expert});
        m.b_eids    = decl_b(b, nm.f("moe_ids"),    RAD_I32,     {R, mc.top_k});
        m.b_ew      = decl_b(b, nm.f("moe_w"),      g.act_dtype, {R, mc.top_k});
        m.b_sorted  = decl_b(b, nm.f("moe_sorted"), RAD_I32,     {K});
        m.b_eoff    = decl_b(b, nm.f("moe_off"),    RAD_I32,     {mc.n_expert + 1});
        m.b_ecnt    = decl_b(b, nm.f("moe_cnt"),    RAD_I32,     {mc.n_expert}, RAD_BUF_PERSIST);
        m.b_egu     = decl_b(b, nm.f("moe_gate_up"), g.act_dtype, {K, 2 * mc.n_ff_exp});
        m.a_eff.x   = decl_b(b, nm.f("moe_ffn"),    g.act_dtype, {K, mc.n_ff_exp});
        m.b_edn     = decl_b(b, nm.f("moe_down"),   g.act_dtype, {K, g.n_embd});
        if (mc.n_ff_shared > 0) {
            m.b_emoe  = decl_b(b, nm.f("moe_out"),   g.act_dtype, {R, g.n_embd});
            m.b_sgu   = decl_b(b, nm.f("shexp_gate_up"), g.act_dtype, {R, 2 * mc.n_ff_shared});
            m.a_sff.x = decl_b(b, nm.f("shexp_ffn"),     g.act_dtype, {R, mc.n_ff_shared});
            m.a_sff.q = decl_b(b, nm.f("shexp_ffn.q"),   RAD_F8E4M3,  {R, mc.n_ff_shared});
            m.a_sff.s = decl_b(b, nm.f("shexp_ffn.s"),   RAD_F32,
                              {R, scale_groups(mc.n_ff_shared)});
            m.b_sout  = decl_b(b, nm.f("shexp_out"),     g.act_dtype, {R, g.n_embd});
            m.b_sgate = decl_b(b, nm.f("shexp_gate"),    g.act_dtype, {R, 1});
        }
    }

    /* ---- the vocabulary edge. The embedding is row-sharded with the vocabulary and summed by the
     * all-reduce below; there is no lm_head here, only its 4097 candidate rows (declare_frame). */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"),
                          nm.ckpt("model.language_model.norm.weight")));
    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    /* Always issued by PROMPT, never folded into layer 0's norm: a frame's embedding is the same
     * on every rank (replicated tables), and a reduction there would sum it world times. Exact:
     * it sums disjoint rows. */
    if (g.world > 1)
        m.op_embed_ar = decl_all_reduce_exact(b, g, m.b_x);
    m.op_rope_cs = rw(b, decl_rotary(b, g, "rope_table",
                         RAD_PARAMS(RAD_RANGE("M", 1, qk_fuse_rows(g)), RAD_INT("rot", m.n_rot),
                                    RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                    RAD_STR("dtype", "f32")),
                         RAD_NOWEIGHTS),
                     {}, {m.b_rope_cs});

    AttnGatedQ::Wire aw{};
    aw.x = m.b_x; aw.h = m.a_h; aw.qg = m.b_qg; aw.k = m.b_k; aw.v = m.b_v;
    aw.q = m.b_q; aw.attn = m.a_attn;
    aw.cos_sin = m.op_rope_cs ? m.b_rope_cs : 0;

    GdnQ::Wire gw{};
    gw.x = m.b_x; gw.h = m.a_h; gw.in = m.a_in; gw.ab = m.b_ab;
    gw.q = m.b_gq; gw.k = m.b_gk; gw.v = m.b_gv;
    gw.gdec = m.b_gdec; gw.beta = m.b_beta; gw.kkt = m.b_kkt; gw.o = m.a_go;

    MoeQ::Wire mw{};
    mw.x = m.b_x; mw.h = m.a_h;
    mw.logits = m.b_rlogits; mw.ids = m.b_eids; mw.ew = m.b_ew;
    mw.sorted = m.b_sorted;  mw.eoff = m.b_eoff; mw.ecnt = m.b_ecnt;
    mw.egu = m.b_egu; mw.eff = m.a_eff; mw.edn = m.b_edn; mw.moe = m.b_emoe;
    mw.sgu = m.b_sgu; mw.sff = m.a_sff; mw.sout = m.b_sout; mw.sgate = m.b_sgate;

    m.layers.resize((size_t)m.n_main);
    for (int64_t l = 0; l < m.n_main; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.full = is_full_attention(l, interval) ? 1 : 0;
        /* embed_ar false: layer 0's norm takes no reduction (see op_embed_ar). */
        const LayerFold f = layer_fold(g, l, m.n_main, nullptr, 0, false);
        const char* pre = "model.language_model.layers";
        if (lay.full) {
            AttnGatedQ::Src as{};
            as.norm   = nm.ckpt("%s.%lld.input_layernorm.weight",  pre, (long long)l);
            as.qg     = nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l);
            as.k      = nm.ckpt("%s.%lld.self_attn.k_proj.weight", pre, (long long)l);
            as.v      = nm.ckpt("%s.%lld.self_attn.v_proj.weight", pre, (long long)l);
            as.q_norm = nm.ckpt("%s.%lld.self_attn.q_norm.weight", pre, (long long)l);
            as.k_norm = nm.ckpt("%s.%lld.self_attn.k_norm.weight", pre, (long long)l);
            as.o      = nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_full));
            RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv_full, m.n_rot, aw, as,
                                          f.mix_fold, false, f.mix_ar_in, f.mix_ar_out));
        } else {
            GdnQ::Src gs{};
            gs.norm     = nm.ckpt("%s.%lld.input_layernorm.weight",         pre, (long long)l);
            gs.in_qkv   = nm.ckpt("%s.%lld.linear_attn.in_proj_qkv.weight", pre, (long long)l);
            gs.in_a     = nm.ckpt("%s.%lld.linear_attn.in_proj_a.weight",   pre, (long long)l);
            gs.in_b     = nm.ckpt("%s.%lld.linear_attn.in_proj_b.weight",   pre, (long long)l);
            gs.in_z     = nm.ckpt("%s.%lld.linear_attn.in_proj_z.weight",   pre, (long long)l);
            gs.conv1d   = nm.ckpt("%s.%lld.linear_attn.conv1d.weight",      pre, (long long)l);
            gs.a_log    = nm.ckpt("%s.%lld.linear_attn.A_log",              pre, (long long)l);
            gs.dt_bias  = nm.ckpt("%s.%lld.linear_attn.dt_bias",            pre, (long long)l);
            gs.out_norm = nm.ckpt("%s.%lld.linear_attn.norm.weight",        pre, (long long)l);
            gs.out      = nm.ckpt("%s.%lld.linear_attn.out_proj.weight",    pre, (long long)l);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_state));
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_conv));
            RAD_ARCH_TRY(lay.gdn.declare(b, nm, g, gc, (int)l, m.kv_state, m.kv_conv, gw, gs,
                                         f.mix_fold, false, f.mix_ar_in, f.mix_ar_out));
        }
        MoeQ::Src ms{};
        ms.cfg     = m.moecfg;
        ms.norm    = nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l);
        ms.router  = nm.ckpt("%s.%lld.mlp.gate.weight",                 pre, (long long)l);
        ms.experts = nm.ckpt("%s.%lld.mlp.experts",                     pre, (long long)l);
        if (m.moecfg.n_ff_shared > 0) {
            ms.shared      = nm.ckpt("%s.%lld.mlp.shared_expert",             pre, (long long)l);
            ms.shared_gate = nm.ckpt("%s.%lld.mlp.shared_expert_gate.weight", pre, (long long)l);
        }
        RAD_ARCH_TRY(MoeQ::probe(b, nm, (int)l, ms, &ms.cfg));
        /* Int8 gate/up needs a rotated copy of the bf16 norm output used by the shared expert.
         * The down projection needs one activation row per routed slot in this pass. */
        if (ms.cfg.expert_i8 && !mw.hr.q8) {
            RAD_ARCH_TRY(mw.hr.declare_qs(b, nm, g, "moe_hr", g.n_embd, g.max_tok, true, false));
            mw.hr.q8_fed = mw.hr.q8_rot = true;
            RAD_ARCH_TRY(mw.eff.declare_qs(b, nm, g, "moe_ffn", ms.cfg.n_ff_exp,
                                           ms.cfg.rows * ms.cfg.top_k, true, false));
        } else if (!ms.cfg.expert_i8 && !ms.cfg.expert_bf16 && !mw.eff.q) {
            RAD_ARCH_TRY(mw.eff.declare_qs(b, nm, g, "moe_ffn", ms.cfg.n_ff_exp,
                                           ms.cfg.rows * ms.cfg.top_k));
        }
        mw.eff.q8_fed = mw.eff.q8_rot = ms.cfg.expert_i8 != 0;
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms, true, f.mlp_add,
                                     f.mlp_ar_in, f.mlp_ar_out));
    }
    for (int64_t l = 0; l + 1 < m.n_main; ++l)
        m.layers[(size_t)l].mlp.xp_next = &m.layers[(size_t)l + 1].mlp;

    m.op_final_norm = norm_op(b, g.max_tok, g.n_embd, g.eps, 1.0, m.w_out_norm, {m.b_x},
                              m.a_h.x);
    if (!have_interval)
        rad_note(b, "the container did not state full_attention_interval; assuming %lld, which "
                    "makes layers 3,7,... full attention", (long long)interval);
    return m.op_embed && m.op_final_norm ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the frame's head */
/* The lane state, the candidate head and the main token's two draws (FRAME, before the tower),
 * and what PROMPT leaves in the lane. Declared in the order a frame issues them: buffer liveness
 * follows declaration order. */
static int declare_frame_head(RadBuilder* b, Model& m) {
    Names& nm = m.nm;
    const int64_t S = m.lanes, D = m.g.n_embd, T = m.g.max_tok;
    m.b_hid    = decl_b(b, nm.f("frame.hid"), RAD_BF16, { S, D }, RAD_BUF_PERSIST);
    m.b_status = decl_b(b, nm.f("frame.status"), RAD_I32, { S }, RAD_BUF_PERSIST);
    m.b_hist   = decl_b(b, nm.f("frame.history"), RAD_I32, { S * kCodes, kMaxFrames },
                        RAD_BUF_PERSIST);
    m.b_ras    = decl_b(b, nm.f("frame.ras"), RAD_I32, { S, kRasWin }, RAD_BUF_PERSIST);
    m.b_u      = decl_b(b, nm.f("frame.unseen"), RAD_BF16, { S, D }, RAD_BUF_PERSIST);
    m.b_plane  = decl_b(b, nm.f("frame.plane"), RAD_BF16, { S, D });
    m.b_slog   = decl_b(b, nm.f("frame.slog"), RAD_F32, { S, kCand });
    m.b_first  = decl_b(b, nm.f("frame.first"), RAD_I32, { S });
    m.b_alt    = decl_b(b, nm.f("frame.alt"), RAD_I32, { S });
    m.b_codes  = decl_b(b, nm.f("frame.codes"), RAD_I32, { kCodes, S });
    m.b_ug     = decl_b(b, nm.f("frame.ug"), RAD_BF16, { S, D });
    m.b_last   = decl_b(b, nm.f("frame.last"), RAD_BF16, { S, D });

    /* The main token's candidates: the lm_head's rows for the end of speech and the semantic band,
     * in id order, replicated (the full head is vocab-sharded and 1 GB). The same rows of the
     * embedding feed a frame back to the backbone, indexed by the same candidate. */
    const char* lm = nm.ckpt("lm_head.weight");
    const char* et = nm.ckpt("model.language_model.embed_tokens.weight");
    RAD_ARCH_TRY(map_rows(b, nm.f("frame.cand.weight"),
                          { { lm, (int)kEos, 1 }, { lm, (int)kSemBegin, kVocab } }));
    m.w_cand = decl_w(b, nm.f("frame.cand.weight"), RAD_BF16, { kCand, D }, RAD_ACCESS_PER_TOKEN,
                      RAD_SHARD_NONE, grp_layer(kGrpFrame));
    RAD_ARCH_TRY(map_rows(b, nm.f("frame.cemb.weight"),
                          { { et, (int)kEos, 1 }, { et, (int)kSemBegin, kVocab } }));
    m.w_cemb = decl_w(b, nm.f("frame.cemb.weight"), RAD_BF16, { kCand, D }, RAD_ACCESS_PER_TOKEN,
                      RAD_SHARD_NONE, grp_layer(kGrpFrame));
    /* The codebook table's sections 1..9; section 0 is never read (tower.py, embedding_sum). */
    RAD_ARCH_TRY(map_rows(b, nm.f("frame.cb.weight"),
                          { { nm.ckpt("audio_decoder.codebook_embeddings.weight"), (int)kVocab,
                              (kCodes - 1) * kVocab } }));
    m.w_cb = decl_w(b, nm.f("frame.cb.weight"), RAD_BF16, { (kCodes - 1) * kVocab, D },
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(kGrpFrame));
    if (!m.w_cand || !m.w_cemb || !m.w_cb) return RAD_E_INVAL;

    /* ---- FRAME: the hidden states, the candidates' logits, the two draws and the choice */
    m.op_hsel = rows_op(b, "gather_rows", S, D, { m.b_hid }, { m.b_plane });
    m.op_head = rw(b, RAD_OP(b, "logits_gemm",
                             RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_vocab", kCand),
                                        RAD_INT("n_embd", D), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(m.w_cand)),
                   { m.b_plane }, { m.b_slog });
    m.op_samp = draw_op(b, S, kCand, kMainTopK, kTemp, kTopP, 0, m.b_slog, m.b_first);
    m.op_samp_ras = draw_op(b, S, kCand, kRasTopK, kRasTemp, kRasTopP, 1, m.b_slog, m.b_alt);
    /* The redraw: only a semantic first draw (candidates 1..4096), and only one the lane's last
     * 16 picks hold; the prompt's rows are text, so its window starts empty (op_rasinit). */
    m.op_ras = rw(b, RAD_OP(b, "ras_pick",
                            RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("win", kRasWin),
                                       RAD_INT("lo", 1), RAD_INT("hi", kCand), RAD_INT("rep", 1)),
                            RAD_NOWEIGHTS),
                  { m.b_first, m.b_alt, m.b_ras }, { m.b_codes, m.b_ras });
    m.op_status = rw(b, RAD_OP(b, "end_status",
                               RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("end", 0),
                                          RAD_INT("rep", 1)),
                               RAD_NOWEIGHTS),
                     { m.b_codes }, { m.b_status });
    m.op_ugather = rows_op(b, "gather_rows", S, D, { m.b_u }, { m.b_ug });

    /* ---- PROMPT: the last row's hidden state into the lane, an empty window and no unseen codes
     * (the prompt's last row is text), and the tower's code embeddings, all zero for text labels */
    m.op_plast = rows_op(b, "gather_rows", S, D, { m.a_h.x }, { m.b_last });
    m.op_pstore = rw(b, RAD_OP(b, "scatter_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", D),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_last }, { m.b_hid });
    m.op_uzero = rw(b, RAD_OP(b, "fill",
                              RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("n", D),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    {}, { m.b_u });
    m.op_rasinit = iota_op(b, kRasWin, -1, 0, m.b_ras);
    m.b_tcat = decl_b(b, nm.f("tower.cat"), RAD_BF16, { T, 2 * D });
    m.op_efill = rw(b, RAD_OP(b, "fill",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    {}, { m.b_tcat });
    return m.op_hsel && m.op_head && m.op_samp && m.op_samp_ras && m.op_ras && m.op_status &&
                   m.op_ugather && m.op_plast && m.op_pstore && m.op_uzero && m.op_rasinit &&
                   m.op_efill
               ? RAD_OK
               : RAD_E_INVAL;
}

/* ================================================================== the tower */
static int declare_tower(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    Geom& g = m.gt;
    g = m.g;
    const char* ad = "audio_decoder_config";
    char key[96];
    auto ai = [&](const char* k, long long d) {
        std::snprintf(key, sizeof key, "%s.%s", ad, k);
        return (int64_t)rad_meta_geti(meta, key, d);
    };
    auto af = [&](const char* k, double d) {
        std::snprintf(key, sizeof key, "%s.%s", ad, k);
        return rad_meta_getf(meta, key, d);
    };
    g.n_head_all = ai("n_head", 0);
    g.n_head_kv_all = ai("n_local_heads", 0);
    g.head_dim = ai("head_dim", 0);
    g.n_ff_all = ai("intermediate_size", 0);
    const double rot_frac = af("rotary_percent", 0.0);
    g.eps = (float)af("norm_eps", 1e-6);
    g.theta = (float)af("rope_base", 0.0);
    if (ai("dim", 0) != g.n_embd || g.n_head_all <= 0 || g.n_head_kv_all <= 0 ||
        g.head_dim <= 0 || g.n_ff_all <= 0 || !(g.theta > 0.0f) || !(rot_frac > 0.0)) {
        fprintf(stderr, "radiance: fishomni: the audio decoder config does not describe the "
                        "tower (dim %lld, %lld heads over %lld of %lld, ffn %lld)\n",
                (long long)ai("dim", 0), (long long)g.n_head_all, (long long)g.n_head_kv_all,
                (long long)g.head_dim, (long long)g.n_ff_all);
        return RAD_E_UNSUPPORTED;
    }
    RAD_ARCH_TRY(divide_or_fail(g.n_head_all, g.world, "the tower's heads", &g.n_head));
    RAD_ARCH_TRY(divide_or_fail(g.n_head_kv_all, g.world, "the tower's KV heads", &g.n_head_kv));
    RAD_ARCH_TRY(divide_or_fail(g.n_ff_all, g.world, "the tower's feed-forward", &g.n_ff));
    g.rope_scale = 1.0f;
    g.yarn = RadYarn{};
    g.rope_mc = false;
    m.n_rot_t = (int64_t)(g.head_dim * rot_frac);
    /* One KV group for the backbone's ten attention layers and these: the cache rows must match. */
    if (g.n_head_kv != m.g.n_head_kv || g.head_dim != m.g.head_dim) {
        fprintf(stderr, "radiance: fishomni: the tower's KV heads (%lld of %lld) differ from the "
                        "backbone's (%lld of %lld), and they share one cache group\n",
                (long long)g.n_head_kv, (long long)g.head_dim, (long long)m.g.n_head_kv,
                (long long)m.g.head_dim);
        return RAD_E_UNSUPPORTED;
    }

    const int64_t T = g.max_tok, S = m.lanes, D = g.n_embd, hd = g.head_dim;
    const int64_t qd = g.n_head * hd, kd = g.n_head_kv * hd, ff = g.n_ff;
    const int64_t qd_all = g.n_head_all * hd, kd_all = g.n_head_kv_all * hd;
    m.b_te    = decl_b(b, nm.f("tower.e"), RAD_BF16, { S, D });
    m.b_tm    = decl_b(b, nm.f("tower.m"), RAD_BF16, { T, D });
    m.b_th    = decl_b(b, nm.f("tower.h"), RAD_BF16, { T, D });
    m.b_tq    = decl_b(b, nm.f("tower.q"), RAD_BF16, { T, qd });
    m.b_tg    = decl_b(b, nm.f("tower.g"), RAD_BF16, { T, qd });
    m.b_tk    = decl_b(b, nm.f("tower.k"), RAD_BF16, { T, kd });
    m.b_tv    = decl_b(b, nm.f("tower.v"), RAD_BF16, { T, kd });
    m.b_tattn = decl_b(b, nm.f("tower.attn"), RAD_BF16, { T, qd });
    m.b_to    = decl_b(b, nm.f("tower.o"), RAD_BF16, { T, D });
    m.b_tgu   = decl_b(b, nm.f("tower.gu"), RAD_BF16, { T, 2 * ff });
    m.b_tff   = decl_b(b, nm.f("tower.ff"), RAD_BF16, { T, ff });
    m.b_tho   = decl_b(b, nm.f("tower.out"), RAD_BF16, { T, D });
    m.b_hlog  = decl_b(b, nm.f("tower.hlog"), RAD_F32, { S, kVocab });
    /* PROMPT with references: every row's label embedding, and its unseen codes (2..9 of the
     * row's own frame, zero at a token's row), summed in f32 and rounded once as op_u's. */
    m.b_pte   = decl_b(b, nm.f("tower.pe"), RAD_BF16, { T, D });
    m.b_pu    = decl_b(b, nm.f("tower.pu"), RAD_BF16, { T, D });
    m.op_ptefill = rw(b, RAD_OP(b, "fill",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                      {}, { m.b_pte });
    m.op_pufill = rw(b, RAD_OP(b, "fill",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", D),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     {}, { m.b_pu });
    m.op_pu = rw(b, RAD_OP(b, "rvq_embed",
                           RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", D),
                                      RAD_INT("n_codes", kCodes - 2), RAD_INT("vocab", kVocab),
                                      RAD_INT("off0", 0), RAD_F64("scale", 1.0),
                                      RAD_STR("dtype", "bf16"), RAD_INT("round", 1)),
                           RAD_WEIGHTS(m.w_cb, m.w_cb)),
                 { m.b_pc }, { m.b_pu });
    if (!m.op_ptefill || !m.op_pufill || !m.op_pu) return RAD_E_INVAL;

    for (int i = 0; i < kTower; ++i) {
        TowerLayer& L = m.tw[i];
        const bool shared = i < kShared;
        const int k = i - kShared;                 /* the per-codebook index, -1.. for shared */
        const int grp = kGrpTower + i, kvl = (int)m.n_main + i;
        const char* pre = shared ? nm.ckpt("audio_decoder_mtp_shared.layers.%d", i)
                                 : nm.ckpt("audio_decoder_mtp.layers.%d", k);
        auto src = [&](const char* s) { return nm.ckpt("%s.%s", pre, s); };
        auto dn = [&](const char* s) { return nm.f("tower.%d.%s", i, s); };
        auto gain = [&](const char* d, const char* s, int64_t n) {
            return wcopy(b, dn(d), src(s), RAD_F32, { n }, grp);
        };

        /* The code embedding. A layer that injects code 0 reads it by candidate (row 0 the end of
         * speech, row c + 1 code c), so its table is the checkpoint's with row 0 again in front:
         * fish-speech clamps the end's c0 to 0. The others read the code a head drew. */
        const char* tsrc = src(shared ? "semantic_embedding.weight" : "codebook_embedding.weight");
        const bool by_cand = shared || k == 0;
        if (by_cand) RAD_ARCH_TRY(map_rows(b, dn("emb.weight"), { { tsrc, 0, 1 },
                                                                  { tsrc, 0, kVocab } }));
        else RAD_ARCH_TRY(map_copy(b, dn("emb.weight"), tsrc));
        L.w_emb = decl_w(b, dn("emb.weight"), RAD_BF16, { by_cand ? kCand : kVocab, D },
                         RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(grp));
        L.w_en  = gain("enorm.weight", "enorm.weight", D);
        L.w_hn  = gain("hnorm.weight", "hnorm.weight", D);
        L.w_an  = gain("attn_norm.weight", "transformer_layer.attention_norm.weight", D);
        L.w_fn  = gain("ffn_norm.weight", "transformer_layer.ffn_norm.weight", D);
        L.w_fin = gain("final_norm.weight", "final_layernorm.weight", D);
        L.w_qn  = gain("q_norm.weight", "transformer_layer.attention.q_norm.weight", hd);
        L.w_kn  = gain("k_norm.weight", "transformer_layer.attention.k_norm.weight", hd);
        L.w_eh  = wcopy(b, dn("eh.weight"), src("eh_proj.weight"), RAD_BF16, { D, 2 * D }, grp);

        /* wqkv rows [Q | G | K | V], each a projection of its own, split by heads. */
        const char* wqkv = src("transformer_layer.attention.wqkv.weight");
        auto proj = [&](const char* d, int64_t first, int64_t rows, int64_t local) -> rad_weight {
            if (map_rows(b, dn(d), { { wqkv, (int)first, rows } }) < 0) return 0;
            return decl_w(b, dn(d), RAD_BF16, { local, D }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW,
                          grp_layer(grp));
        };
        L.w_q = proj("q.weight", 0, qd_all, qd);
        L.w_g = proj("g.weight", qd_all, qd_all, qd);
        L.w_k = proj("k.weight", 2 * qd_all, kd_all, kd);
        L.w_v = proj("v.weight", 2 * qd_all + kd_all, kd_all, kd);
        RAD_ARCH_TRY(map_copy(b, dn("o.weight"), src("transformer_layer.attention.wo.weight")));
        L.w_o = decl_w(b, dn("o.weight"), RAD_BF16, { D, qd }, RAD_ACCESS_PER_TOKEN,
                       RAD_SHARD_COL, grp_layer(grp));
        RAD_ARCH_TRY(map_concat(b, dn("gate_up.weight"),
                                { src("transformer_layer.feed_forward.w1.weight"),
                                  src("transformer_layer.feed_forward.w3.weight") }));
        L.w_gu = decl_w(b, dn("gate_up.weight"), RAD_BF16, { 2 * ff, D }, RAD_ACCESS_PER_TOKEN,
                        RAD_SHARD_ROW, grp_layer(grp), 0, { ff, ff });
        RAD_ARCH_TRY(map_copy(b, dn("down.weight"),
                              src("transformer_layer.feed_forward.w2.weight")));
        L.w_down = decl_w(b, dn("down.weight"), RAD_BF16, { D, ff }, RAD_ACCESS_PER_TOKEN,
                          RAD_SHARD_COL, grp_layer(grp));
        if (!L.w_emb || !L.w_en || !L.w_hn || !L.w_an || !L.w_fn || !L.w_fin || !L.w_qn ||
            !L.w_kn || !L.w_eh || !L.w_q || !L.w_g || !L.w_k || !L.w_v || !L.w_o || !L.w_gu ||
            !L.w_down)
            return RAD_E_INVAL;

        /* ---- the ops, in the order a pass issues them. The embedding and its norm run in FRAME
         * only; PROMPT's labels are text, whose embedding is zero, and so is its norm (op_efill
         * zeroes that half of the concatenation once a prompt). */
        L.op_emb = rw(b, RAD_OP(b, "rvq_embed",
                                RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_embd", D),
                                           RAD_INT("n_codes", 1), RAD_INT("vocab", kVocab),
                                           RAD_INT("off0", 0), RAD_F64("scale", 1.0),
                                           RAD_STR("dtype", "bf16")),
                                RAD_WEIGHTS(L.w_emb, L.w_emb)),
                      { m.b_codes }, { m.b_te });
        if (i == 0) L.op_eadd = add_op(b, S, D, m.b_te, m.b_ug, m.b_te);
        L.op_en = norm_op(b, S, D, g.eps, 1.0, L.w_en, { m.b_te }, m.b_tcat);
        /* PROMPT with references: the label is the next row's frame, off the prompt rows' code
         * plane a row on. Code 0 there is the frame's own c0, so a by-candidate table reads it
         * one row on; a token row's code (-2 for code 0, -1 for the others) reads nothing, and
         * its embedding stays the zero the fill leaves. */
        L.op_pemb = rw(b, RAD_OP(b, "rvq_embed",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", D),
                                            RAD_INT("n_codes", 1), RAD_INT("vocab", kVocab),
                                            RAD_INT("off0", by_cand ? 1 : 0),
                                            RAD_F64("scale", 1.0), RAD_STR("dtype", "bf16")),
                                 RAD_WEIGHTS(L.w_emb, L.w_emb)),
                       { m.b_pc }, { m.b_pte });
        if (i == 0) L.op_peadd = add_op(b, T, D, m.b_pte, m.b_pu, m.b_pte);
        L.op_pen = norm_op(b, T, D, g.eps, 1.0, L.w_en, { m.b_pte }, m.b_tcat);
        if (!L.op_pemb || !L.op_pen || (i == 0 && !L.op_peadd)) return RAD_E_INVAL;
        /* Layer 0's hidden state is the backbone's: the prompt's rows (PROMPT) or the frame's
         * plane (FRAME); both stay live to here. */
        L.op_hn = i == 0 ? norm_op(b, T, D, g.eps, 1.0, L.w_hn, { m.a_h.x, m.b_plane }, m.b_tcat)
                         : norm_op(b, T, D, g.eps, 1.0, L.w_hn, { m.b_tho }, m.b_tcat);
        L.op_eh = gemm_op(b, T, D, 2 * D, L.w_eh, m.b_tcat, m.b_tm);
        L.op_an = norm_op(b, T, D, g.eps, 1.0, L.w_an, { m.b_tm }, m.b_th);
        L.op_q = gemm_op(b, T, qd, D, L.w_q, m.b_th, m.b_tq);
        L.op_g = gemm_op(b, T, qd, D, L.w_g, m.b_th, m.b_tg);
        L.op_k = gemm_op(b, T, kd, D, L.w_k, m.b_th, m.b_tk);
        L.op_v = gemm_op(b, T, kd, D, L.w_v, m.b_th, m.b_tv);
        /* (1 + w) per head, always: the exporter never wrote attention_qk_norm (tower.py). */
        L.op_qn = norm_op(b, T * g.n_head, hd, g.eps, 1.0, L.w_qn, { m.b_tq }, m.b_tq);
        L.op_kn = norm_op(b, T * g.n_head_kv, hd, g.eps, 1.0, L.w_kn, { m.b_tk }, m.b_tk);
        /* Adjacent pairs (GPT-J), unlike the backbone's halves, over the first 64 dims. */
        L.op_rq = rw(b, decl_rope(b, g, T, hd, g.n_head, 0, m.n_rot_t, "gptj"), { m.b_tq },
                     { m.b_tq });
        L.op_rk = rw(b, decl_rope(b, g, T, hd, g.n_head_kv, 0, m.n_rot_t, "gptj"), { m.b_tk },
                     { m.b_tk });
        RAD_ARCH_TRY(rad_bind_layer_kv(b, kvl, m.kv_full));
        /* Declared before kv_store, which reads back the resolved block size. */
        L.op_attn = rw(b, RAD_OP(b, "attn_paged",
                           RAD_PARAMS(RAD_RANGE("q_len", 1, T), RAD_INT("head_dim", hd),
                                      RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                      RAD_INT("window", 0), RAD_INT("n_head", g.n_head),
                                      RAD_INT("block_size", RAD_KV_BLOCK),
                                      RAD_INT("max_seqs", g.max_seqs),
                                      RAD_INT("max_ctx", g.max_ctx),
                                      RAD_STR("q_dtype", g.dtype),
                                      RAD_STR("kv_dtype", m.g.kv_dtype)),
                           RAD_NOWEIGHTS),
                       { m.b_tq }, { m.b_tattn });
        L.op_kv = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, m.kv_full)),
                                    RAD_STR("kv_dtype", m.g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     { m.b_tk, m.b_tv }, {});
        L.op_sig = rows_op(b, "sigmoid", T, qd, { m.b_tg }, { m.b_tg });
        L.op_mul = rows_op(b, "mul", T, qd, { m.b_tattn, m.b_tg }, { m.b_tattn });
        L.op_o = gemm_op(b, T, D, qd, L.w_o, m.b_tattn, m.b_to);
        L.op_ar1 = ar_op(b, g, m.b_to);
        L.op_add1 = add_op(b, T, D, m.b_tm, m.b_to, m.b_tm);
        L.op_fn = norm_op(b, T, D, g.eps, 1.0, L.w_fn, { m.b_tm }, m.b_th);
        L.op_gu = gemm_op(b, T, 2 * ff, D, L.w_gu, m.b_th, m.b_tgu);
        L.op_act = rows_op(b, "silu_mul", T, ff, { m.b_tgu }, { m.b_tff });
        L.op_down = gemm_op(b, T, D, ff, L.w_down, m.b_tff, m.b_to);
        L.op_ar2 = ar_op(b, g, m.b_to);
        L.op_add2 = add_op(b, T, D, m.b_tm, m.b_to, m.b_tm);
        L.op_fin = norm_op(b, T, D, g.eps, 1.0, L.w_fin, { m.b_tm }, m.b_tho);
        if (!L.op_emb || !L.op_en || !L.op_hn || !L.op_eh || !L.op_an || !L.op_q || !L.op_g ||
            !L.op_k || !L.op_v || !L.op_qn || !L.op_kn || !L.op_rq || !L.op_rk || !L.op_attn ||
            !L.op_kv || !L.op_sig || !L.op_mul || !L.op_o || !L.op_add1 || !L.op_fn ||
            !L.op_gu || !L.op_act || !L.op_down || !L.op_add2 || !L.op_fin ||
            (i == 0 && !L.op_eadd) || (g.world > 1 && (!L.op_ar1 || !L.op_ar2)))
            return RAD_E_INVAL;

        /* A per-codebook layer's head draws code k + 1 (FRAME only). */
        if (!shared) {
            L.w_head = wcopy(b, nm.f("tower.head.%d.weight", k),
                             nm.ckpt("audio_decoder_mtp.heads.%d.weight", k), RAD_BF16,
                             { kVocab, D }, grp);
            L.op_head = rw(b, RAD_OP(b, "logits_gemm",
                                     RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_vocab", kVocab),
                                                RAD_INT("n_embd", D), RAD_STR("dtype", "bf16")),
                                     RAD_WEIGHTS(L.w_head)),
                           { m.b_tho }, { m.b_hlog });
            L.op_samp = draw_op(b, S, kVocab, kHeadTopK, kTemp, kTopP, 2 + k, m.b_hlog, m.b_codes);
            if (!L.w_head || !L.op_head || !L.op_samp) return RAD_E_INVAL;
        }
    }
    /* The codes stay live through the last layer: every layer's embedding reads them. */
    RAD_ARCH_TRY(rad_op_reads(b, m.tw[kTower - 1].op_samp, &m.b_codes, 1));
    return RAD_OK;
}

/* ================================================================== the frame's tail */
/* The frame into the lane's history, its unseen codes into the lane, and its embedding into the
 * backbone's stream; after the backbone, the frame row's hidden state into the lane. */
static int declare_frame_tail(RadBuilder* b, Model& m) {
    Names& nm = m.nm;
    const int64_t S = m.lanes, D = m.g.n_embd;
    m.b_unew = decl_b(b, nm.f("frame.unew"), RAD_BF16, { S, D });
    m.op_store = rw(b, RAD_OP(b, "frame_store",
                              RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_codes", kCodes),
                                         RAD_INT("stride", kMaxFrames), RAD_INT("rep", 1),
                                         RAD_INT("skip", 0)),
                              RAD_NOWEIGHTS),
                    { m.b_codes, m.b_status }, { m.b_hist });
    /* Codes 2..9 summed in f32 and rounded once, as embedding_bag does: sections 2..9. */
    m.op_u = rw(b, RAD_OP(b, "rvq_embed",
                          RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_embd", D),
                                     RAD_INT("n_codes", kCodes - 2), RAD_INT("vocab", kVocab),
                                     RAD_INT("off0", 0), RAD_F64("scale", 1.0),
                                     RAD_STR("dtype", "bf16"), RAD_INT("round", 1)),
                          RAD_WEIGHTS(m.w_cb, m.w_cb)),
                { m.b_codes }, { m.b_unew });
    m.op_uscat = rw(b, RAD_OP(b, "scatter_rows",
                              RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", D),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_unew }, { m.b_u });
    /* (E_lm[candidate] + E_cb[4096 + c1]) / sqrt 2, each sum rounded to bf16 as the reference's
     * tensors are (tower.py, embed_backbone). */
    m.op_fb = rw(b, RAD_OP(b, "rvq_embed",
                           RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_embd", D),
                                      RAD_INT("n_codes", 2), RAD_INT("vocab", kVocab),
                                      RAD_INT("off0", 0), RAD_F64("scale", 1.0 / std::sqrt(2.0)),
                                      RAD_STR("dtype", "bf16")),
                           RAD_WEIGHTS(m.w_cemb, m.w_cb)),
                 { m.b_codes }, { m.b_x });
    m.op_fstore = rw(b, RAD_OP(b, "scatter_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", D),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.a_h.x }, { m.b_hid });
    /* PROMPT with references: a frame's row embedded as a made frame's is, off the rows' code
     * plane (code 0 its own c0, so the candidate table is read one row on); a token's row, whose
     * code 0 is -2, keeps its token's embedding. */
    m.op_pfb = rw(b, RAD_OP(b, "rvq_embed",
                            RAD_PARAMS(RAD_RANGE("M", 1, m.g.max_tok), RAD_INT("n_embd", D),
                                       RAD_INT("n_codes", 2), RAD_INT("vocab", kVocab),
                                       RAD_INT("off0", 1), RAD_F64("scale", 1.0 / std::sqrt(2.0)),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(m.w_cemb, m.w_cb)),
                  { m.b_pc }, { m.b_x });
    /* ... and the lane's window of main picks starts with the prompt's last 16 rows', a frame's
     * as its candidate (c0 + 1) and a token's as no pick (-2 + 1). */
    std::string src, one;
    for (int64_t i = 0; i < kRasWin; ++i) {
        src += (i ? " " : "") + std::to_string(i);
        one += i ? " 1" : "1";
    }
    m.op_rascopy = rw(b, RAD_OP(b, "int_affine",
                                RAD_PARAMS(RAD_RANGE("M", 1, kRasWin), RAD_STR("src", src.c_str()),
                                           RAD_STR("mul", one.c_str()),
                                           RAD_STR("add", one.c_str())),
                                RAD_NOWEIGHTS),
                      { m.b_pc }, { m.b_ras });
    return m.op_store && m.op_u && m.op_uscat && m.op_fb && m.op_fstore && m.op_pfb &&
                   m.op_rascopy
               ? RAD_OK
               : RAD_E_INVAL;
}

/* ================================================================== the codec */
/* A causal conv whose weight the checkpoint keeps weight-normalised (folded at convert). */
static int vconv(RadBuilder* b, Names& nm, VConv& v, const char* dn, const char* src, int64_t cin,
                 int64_t cout, int64_t k, int64_t dil, int64_t rows, rad_buf x, rad_buf res,
                 rad_buf y, rad_buf cu, const char* act = "none") {
    const char* wn = nm.f("%s.weight", dn);
    RAD_ARCH_TRY(map_wnorm(b, wn, nm.ckpt("%s.parametrizations.weight.original0", src),
                           nm.ckpt("%s.parametrizations.weight.original1", src)));
    v.w = decl_w(b, wn, RAD_BF16, { cout, cin, k }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                 grp_layer(kGrpCodec));
    v.b = wcopy(b, nm.f("%s.bias", dn), nm.ckpt("%s.bias", src), RAD_BF16, { cout }, kGrpCodec);
    if (!v.w || !v.b) return RAD_E_INVAL;
    const RadParam ps[] = { RAD_RANGE("M", 1, rows), RAD_INT("cin", cin), RAD_INT("cout", cout),
                            RAD_INT("k", k), RAD_INT("dil", dil), RAD_STR("act", act),
                            RAD_STR("dtype", "bf16"), RAD_INT("causal", 1) };
    const rad_weight ws[] = { v.w, v.b };
    v.op = res ? rw(b, rad_decl_op(b, "conv1d", ps, 8, ws, 2), { x, cu, res }, { y })
               : rw(b, rad_decl_op(b, "conv1d", ps, 8, ws, 2), { x, cu }, { y });
    return v.op ? RAD_OK : RAD_E_INVAL;
}

static rad_weight valpha(RadBuilder* b, Names& nm, const char* dn, const char* src, int64_t n) {
    return wcopy(b, nm.f("%s.alpha", dn), nm.ckpt("%s.alpha", src), RAD_F32, { 1, n, 1 },
                 kGrpCodec);
}

static rad_op snake_op(RadBuilder* b, int64_t rows, int64_t n, rad_weight a, rad_buf x, rad_buf y) {
    return rw(b, RAD_OP(b, "snake",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                   RAD_STR("dtype", "bf16")),
                        RAD_WEIGHTS(a)),
              { x }, { y });
}

/* An encoder's causal conv over the pass's clips: their bounds in samples, an input row holding
 * `rate` of them (cu_div), `stride` input rows an output row. */
static rad_op econv_op(RadBuilder* b, int64_t rows, int64_t cin, int64_t cout, int64_t k,
                       int64_t dil, int64_t stride, int64_t rate, rad_weight w, rad_weight bias,
                       rad_buf x, rad_buf res, rad_buf y) {
    const RadParam ps[] = { RAD_RANGE("M", 1, rows), RAD_INT("cin", cin), RAD_INT("cout", cout),
                            RAD_INT("k", k), RAD_INT("dil", dil), RAD_STR("act", "none"),
                            RAD_STR("dtype", "bf16"), RAD_INT("causal", 1),
                            RAD_INT("cu_div", rate), RAD_INT("stride", stride) };
    const int np = stride > 1 ? 10 : 9;
    const rad_weight ws[] = { w, bias };
    return res ? rw(b, rad_decl_op(b, "conv1d", ps, np, ws, 2), { x, res }, { y })
               : rw(b, rad_decl_op(b, "conv1d", ps, np, ws, 2), { x }, { y });
}

/* An encoder conv, weight-normalised as vconv's. */
static int econv(RadBuilder* b, Names& nm, VConv& v, const char* dn, const char* src, int64_t cin,
                 int64_t cout, int64_t k, int64_t dil, int64_t stride, int64_t rate, int64_t rows,
                 rad_buf x, rad_buf res, rad_buf y) {
    const char* wn = nm.f("%s.weight", dn);
    RAD_ARCH_TRY(map_wnorm(b, wn, nm.ckpt("%s.parametrizations.weight.original0", src),
                           nm.ckpt("%s.parametrizations.weight.original1", src)));
    v.w = decl_w(b, wn, RAD_BF16, { cout, cin, k }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                 grp_layer(kGrpCodec));
    v.b = wcopy(b, nm.f("%s.bias", dn), nm.ckpt("%s.bias", src), RAD_BF16, { cout }, kGrpCodec);
    if (!v.w || !v.b) return RAD_E_INVAL;
    v.op = econv_op(b, rows, cin, cout, k, dil, stride, rate, v.w, v.b, x, res, y);
    return v.op ? RAD_OK : RAD_E_INVAL;
}

/* A codec transformer of `n` layers (modded_dac.py's WindowLimitedTransformer): plain-gain RMS
 * norms at eps 1e-5, fused q|k|v with adjacent-pair rotary over all 64 dims at the rows'
 * positions, causal attention over a window of `win` rows within each sequence, layer scales on
 * both residual branches, over `rows` rows of B.x in place. Layer l's weights are `dn`.l.* from
 * `src`.layers.l. The sequences are the buffer `cu`, or with none the pass's clips in samples,
 * `cdiv` of them a row; the positions the buffer `pos`, or with none the pass's own. */
static int declare_ct(RadBuilder* b, Names& nm, std::vector<CodecLayer>& ls, const char* dn,
                      const char* src, int n, int64_t rows, int64_t win, int64_t cdiv,
                      const CtSet& B, rad_buf cu, rad_buf pos) {
    const int64_t qd = kCtHeads * kCtHeadDim;
    ls.assign((size_t)n, CodecLayer{});
    for (int l = 0; l < n; ++l) {
        CodecLayer& L = ls[(size_t)l];
        auto sr = [&](const char* x) { return nm.ckpt("%s.layers.%d.%s", src, l, x); };
        auto d = [&](const char* x) { return nm.f("%s.%d.%s", dn, l, x); };
        L.w_n1 = wcopy(b, d("n1.weight"), sr("attention_norm.weight"), RAD_F32, { kLat },
                       kGrpCodec);
        L.w_n2 = wcopy(b, d("n2.weight"), sr("ffn_norm.weight"), RAD_F32, { kLat }, kGrpCodec);
        L.w_ls1 = wcopy(b, d("ls1.weight"), sr("attention_layer_scale.gamma"), RAD_F32,
                        { kLat }, kGrpCodec);
        L.w_ls2 = wcopy(b, d("ls2.weight"), sr("ffn_layer_scale.gamma"), RAD_F32, { kLat },
                        kGrpCodec);
        L.w_qkv = wcopy(b, d("qkv.weight"), sr("attention.wqkv.weight"), RAD_BF16,
                        { 3 * qd, kLat }, kGrpCodec);
        L.w_o = wcopy(b, d("o.weight"), sr("attention.wo.weight"), RAD_BF16, { kLat, qd },
                      kGrpCodec);
        if (map_concat(b, d("gate_up.weight"),
                       { sr("feed_forward.w1.weight"), sr("feed_forward.w3.weight") }) < 0)
            return RAD_E_INVAL;
        L.w_gu = decl_w(b, d("gate_up.weight"), RAD_BF16, { 2 * kCtFf, kLat },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(kGrpCodec));
        L.w_down = wcopy(b, d("down.weight"), sr("feed_forward.w2.weight"), RAD_BF16,
                         { kLat, kCtFf }, kGrpCodec);
        /* Plain-gain RMS norms at eps 1e-5 (the codec's own RMSNorm, not the backbone's). */
        L.op_n1 = norm_op(b, rows, kLat, kCtEps, 0.0, L.w_n1, { B.x }, B.h);
        L.op_qkv = gemm_op(b, rows, 3 * qd, kLat, L.w_qkv, B.h, B.qkv);
        /* Adjacent pairs over all 64 dims, positions the rows' own: a piece's frames as a whole
         * decode restarting at 0 would see them, a clip's rows in the pass (attention reads only
         * their differences). */
        L.op_rope = rw(b, RAD_OP(b, "rope",
                                 RAD_PARAMS(RAD_RANGE("M", 1, rows),
                                            RAD_INT("head_dim", kCtHeadDim),
                                            RAD_INT("n_head", kCtHeads),
                                            RAD_INT("n_head_kv", kCtHeads),
                                            RAD_F64("theta", kCtTheta), RAD_F64("scale", 1.0),
                                            RAD_STR("mode", "gptj")),
                                 RAD_NOWEIGHTS),
                       { B.qkv }, { B.qkv });
        const RadParam ap[] = { RAD_RANGE("M", 1, rows), RAD_INT("head_dim", kCtHeadDim),
                                RAD_INT("gqa", 1), RAD_INT("causal", 1), RAD_STR("dtype", "bf16"),
                                RAD_F64("scale", 1.0 / std::sqrt((double)kCtHeadDim)),
                                RAD_INT("max_seqlen", rows), RAD_INT("window", win),
                                RAD_INT("cu_div", cdiv) };
        L.op_attn = rw(b, rad_decl_op(b, "attn_dense", ap, cdiv > 1 ? 9 : 8, nullptr, 0),
                       { B.qkv }, { B.attn });
        L.op_o = gemm_op(b, rows, kLat, qd, L.w_o, B.attn, B.o);
        L.op_ls1 = mulw_op(b, rows, kLat, L.w_ls1, B.o, B.o);
        L.op_add1 = add_op(b, rows, kLat, B.x, B.o, B.x);
        L.op_n2 = norm_op(b, rows, kLat, kCtEps, 0.0, L.w_n2, { B.x }, B.h);
        L.op_gu = gemm_op(b, rows, 2 * kCtFf, kLat, L.w_gu, B.h, B.gu);
        L.op_act = rows_op(b, "silu_mul", rows, kCtFf, { B.gu }, { B.ff });
        L.op_down = gemm_op(b, rows, kLat, kCtFf, L.w_down, B.ff, B.o);
        L.op_ls2 = mulw_op(b, rows, kLat, L.w_ls2, B.o, B.o);
        L.op_add2 = add_op(b, rows, kLat, B.x, B.o, B.x);
        if (!L.w_n1 || !L.w_n2 || !L.w_ls1 || !L.w_ls2 || !L.w_qkv || !L.w_o || !L.w_gu ||
            !L.w_down || !L.op_n1 || !L.op_qkv || !L.op_rope || !L.op_attn || !L.op_o ||
            !L.op_ls1 || !L.op_add1 || !L.op_n2 || !L.op_gu || !L.op_act || !L.op_down ||
            !L.op_ls2 || !L.op_add2)
            return RAD_E_INVAL;
        if (cu) RAD_ARCH_TRY(rad_op_reads(b, L.op_attn, &cu, 1));
        if (pos) RAD_ARCH_TRY(rad_op_reads(b, L.op_rope, &pos, 1));
    }
    return RAD_OK;
}

static int declare_codec(RadBuilder* b, Model& m) {
    Names& nm = m.nm;
    const int64_t NL = kVocContext + kVocChunk;       /* frames through the transformer */
    const int64_t NC = kConvFrames;                    /* frames through the convolutions */
    const int64_t qd = kCtHeads * kCtHeadDim;
    const char* q = "codec.quantizer";

    /* ---- the dequantiser: each codebook's raw row (dimension 8) through its own 1x1 projection
     * to 1024, summed into the transformer's stream. Code 0 is a candidate index (c0 + 1); codes
     * 1..9 come from heads of 4096 and read 1024-entry codebooks clamped at 1023 (rvq.py). */
    m.b_rq = decl_b(b, nm.f("codec.rq"), RAD_BF16, { NL, kCbDim });
    m.b_cx = decl_b(b, nm.f("codec.x"), RAD_BF16, { NL, kLat });
    m.b_cconst = decl_b(b, nm.f("codec.nconv"), RAD_I32, { 1 });
    m.op_cnst = iota_op(b, 1, NC, 0, m.b_cconst);
    for (int s = 0; s < kRates; ++s) {
        m.b_ccu[s] = decl_b(b, nm.f("codec.cu%d", s), RAD_I32, { 2 });
        m.op_ccu[s] = s == 0 ? cu_op(b, 1, 0, m.b_ccu[0])
                             : cu_op(b, kRate[s], m.b_cconst, m.b_ccu[s]);
        if (!m.op_ccu[s]) return RAD_E_INVAL;
    }
    for (int k = 0; k < kCodes; ++k) {
        const char* qs = k == 0 ? nm.ckpt("%s.semantic_quantizer.quantizers.0", q)
                                : nm.ckpt("%s.quantizer.quantizers.%d", q, k - 1);
        const int64_t bins = k == 0 ? kVocab : kResVocab;
        m.w_cbk[k] = wcopy(b, nm.f("codec.cb%d.weight", k), nm.ckpt("%s.codebook.weight", qs),
                           RAD_BF16, { bins, kCbDim }, kGrpCodec);
        m.op_rq[k] = rw(b, RAD_OP(b, "rvq_embed",
                                  RAD_PARAMS(RAD_RANGE("M", 1, NL), RAD_INT("n_embd", kCbDim),
                                             RAD_INT("n_codes", 1), RAD_INT("vocab", bins),
                                             RAD_INT("off0", k == 0 ? -1 : 0),
                                             RAD_F64("scale", 1.0), RAD_STR("dtype", "bf16"),
                                             RAD_INT("clamp", k == 0 ? 0 : kResVocab)),
                                  RAD_WEIGHTS(m.w_cbk[k], m.w_cbk[k])),
                        { m.b_hist }, { m.b_rq });
        /* The out-projection is the old-style weight norm (weight_g, weight_v). */
        VConv& v = m.qo[k];
        const char* wn = nm.f("codec.q%d.weight", k);
        RAD_ARCH_TRY(map_wnorm(b, wn, nm.ckpt("%s.out_proj.weight_g", qs),
                               nm.ckpt("%s.out_proj.weight_v", qs)));
        v.w = decl_w(b, wn, RAD_BF16, { kLat, kCbDim, 1 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                     grp_layer(kGrpCodec));
        v.b = wcopy(b, nm.f("codec.q%d.bias", k), nm.ckpt("%s.out_proj.bias", qs), RAD_BF16,
                    { kLat }, kGrpCodec);
        if (!m.w_cbk[k] || !m.op_rq[k] || !v.w || !v.b) return RAD_E_INVAL;
        const RadParam ps[] = { RAD_RANGE("M", 1, NL), RAD_INT("cin", kCbDim),
                                RAD_INT("cout", kLat), RAD_INT("k", 1), RAD_INT("dil", 1),
                                RAD_STR("act", "none"), RAD_STR("dtype", "bf16"),
                                RAD_INT("causal", 1) };
        const rad_weight ws[] = { v.w, v.b };
        v.op = k == 0 ? rw(b, rad_decl_op(b, "conv1d", ps, 8, ws, 2), { m.b_rq, m.b_ccu[0] },
                           { m.b_cx })
                      : rw(b, rad_decl_op(b, "conv1d", ps, 8, ws, 2),
                           { m.b_rq, m.b_ccu[0], m.b_cx }, { m.b_cx });
        if (!v.op) return RAD_E_INVAL;
    }

    /* ---- the post-quantiser transformer (modded_dac.py, WindowLimitedTransformer) */
    const char* pt = "codec.quantizer.post_module";
    m.b_ch    = decl_b(b, nm.f("codec.h"), RAD_BF16, { NL, kLat });
    m.b_cqkv  = decl_b(b, nm.f("codec.qkv"), RAD_BF16, { NL, 3 * qd });
    m.b_cattn = decl_b(b, nm.f("codec.attn"), RAD_BF16, { NL, qd });
    m.b_co    = decl_b(b, nm.f("codec.o"), RAD_BF16, { NL, kLat });
    m.b_cgu   = decl_b(b, nm.f("codec.gu"), RAD_BF16, { NL, 2 * kCtFf });
    m.b_cff   = decl_b(b, nm.f("codec.ff"), RAD_BF16, { NL, kCtFf });
    {
        CtSet B;
        B.x = m.b_cx; B.h = m.b_ch; B.qkv = m.b_cqkv; B.attn = m.b_cattn; B.o = m.b_co;
        B.gu = m.b_cgu; B.ff = m.b_cff;
        RAD_ARCH_TRY(declare_ct(b, nm, m.ct, "codec", pt, (int)kCtLayers, NL, kAttnWin, 1, B,
                                m.b_ccu[0], 0));
    }
    /* The final norm over only the frames the convolutions read: the last nc. */
    m.w_cn = wcopy(b, nm.f("codec.norm.weight"), nm.ckpt("%s.norm.weight", pt), RAD_F32, { kLat },
                   kGrpCodec);
    m.b_ux[0] = decl_b(b, nm.f("codec.ux0"), RAD_BF16, { NC, kLat });
    m.op_cn = norm_op(b, NC, kLat, kCtEps, 0.0, m.w_cn, { m.b_cx }, m.b_ux[0]);

    /* ---- two 2x upsamplings, each a transposed conv of kernel 2 and a ConvNeXt block */
    m.b_ua  = decl_b(b, nm.f("codec.ua"), RAD_BF16, { 4 * NC, kLat });
    m.b_ub  = decl_b(b, nm.f("codec.ub"), RAD_BF16, { 4 * NC, kLat });
    m.b_uh  = decl_b(b, nm.f("codec.uh"), RAD_BF16, { 4 * NC, 4 * kLat });
    m.b_ug2 = decl_b(b, nm.f("codec.ug"), RAD_BF16, { 4 * NC, 4 * kLat });
    for (int i = 0; i < 2; ++i) {
        Upsample& U = m.up[i];
        const int64_t rin = NC << i, rout = NC << (i + 1);
        const char* us = nm.ckpt("%s.upsample.%d", q, i);
        auto w = [&](const char* dn, const char* s, uint32_t dt,
                     std::initializer_list<int64_t> sh) {
            return wcopy(b, nm.f("codec.up%d.%s", i, dn), nm.ckpt("%s.%s", us, s), dt, sh,
                         kGrpCodec);
        };
        U.w_t  = w("t.weight", "0.conv.weight", RAD_BF16, { kLat, kLat, 2 });
        U.b_t  = w("t.bias", "0.conv.bias", RAD_BF16, { kLat });
        U.w_dw = w("dw.weight", "1.dwconv.conv.weight", RAD_BF16, { kLat, 1, 7 });
        U.b_dw = w("dw.bias", "1.dwconv.conv.bias", RAD_BF16, { kLat });
        U.w_ln = w("ln.weight", "1.norm.weight", RAD_F32, { kLat });
        U.b_ln = w("ln.bias", "1.norm.bias", RAD_F32, { kLat });
        U.w_p1 = w("p1.weight", "1.pwconv1.weight", RAD_BF16, { 4 * kLat, kLat });
        U.b_p1 = w("p1.bias", "1.pwconv1.bias", RAD_BF16, { 4 * kLat });
        U.w_p2 = w("p2.weight", "1.pwconv2.weight", RAD_BF16, { kLat, 4 * kLat });
        U.b_p2 = w("p2.bias", "1.pwconv2.bias", RAD_BF16, { kLat });
        U.w_g  = w("gamma", "1.gamma", RAD_F32, { kLat });
        m.b_ux[i + 1] = decl_b(b, nm.f("codec.ux%d", i + 1), RAD_BF16, { rout, kLat });
        U.op_t = rw(b, RAD_OP(b, "conv_t1d",
                              RAD_PARAMS(RAD_RANGE("M", 1, rin), RAD_INT("cin", kLat),
                                         RAD_INT("cout", kLat), RAD_INT("stride", 2),
                                         RAD_STR("dtype", "bf16"), RAD_INT("pad", 0),
                                         RAD_INT("k", 2)),
                              RAD_WEIGHTS(U.w_t, U.b_t)),
                    { m.b_ux[i], m.b_ccu[1 + i] }, { m.b_ux[i + 1] });
        U.op_dw = rw(b, RAD_OP(b, "conv1d_dw",
                               RAD_PARAMS(RAD_RANGE("M", 1, rout), RAD_INT("n", kLat),
                                          RAD_INT("k", 7), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(U.w_dw, U.b_dw)),
                     { m.b_ux[i + 1], m.b_ccu[2 + i] }, { m.b_ua });
        U.op_ln = rw(b, RAD_OP(b, "layernorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, rout), RAD_INT("n", kLat),
                                          RAD_F64("eps", 1e-6), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(U.w_ln, U.b_ln)),
                     { m.b_ua }, { m.b_ub });
        U.op_p1 = gemm_bias_op(b, rout, 4 * kLat, kLat, U.w_p1, U.b_p1, m.b_ub, m.b_uh);
        U.op_gelu = rows_op(b, "gelu", rout, 4 * kLat, { m.b_uh }, { m.b_ug2 });
        U.op_p2 = gemm_bias_op(b, rout, kLat, 4 * kLat, U.w_p2, U.b_p2, m.b_ug2, m.b_ua);
        U.op_g = mulw_op(b, rout, kLat, U.w_g, m.b_ua, m.b_ua);
        U.op_add = add_op(b, rout, kLat, m.b_ux[i + 1], m.b_ua, m.b_ux[i + 1]);
        if (!U.w_t || !U.b_t || !U.w_dw || !U.b_dw || !U.w_ln || !U.b_ln || !U.w_p1 || !U.b_p1 ||
            !U.w_p2 || !U.b_p2 || !U.w_g || !U.op_t || !U.op_dw || !U.op_ln || !U.op_p1 ||
            !U.op_gelu || !U.op_p2 || !U.op_g || !U.op_add)
            return RAD_E_INVAL;
    }

    /* ---- the decoder: a conv into 1536 channels, four upsampling blocks, the output conv */
    const int64_t R0 = 4 * NC;   /* rows at 4 a frame */
    int64_t rows[5], ch[5];
    rows[0] = R0;
    ch[0] = kDec;
    for (int i = 0; i < 4; ++i) {
        rows[i + 1] = rows[i] * m.v_ups[i];
        ch[i + 1] = kDec >> (i + 1);
    }
    for (int s = 0; s < 5; ++s) {
        m.b_vs[s] = decl_b(b, nm.f("codec.s%d", s), RAD_BF16, { rows[s], ch[s] });
        m.b_vt[s] = decl_b(b, nm.f("codec.t%d", s), RAD_BF16, { rows[s], ch[s] });
        if (s > 0) {
            m.b_va[s] = decl_b(b, nm.f("codec.a%d", s), RAD_BF16, { rows[s], ch[s] });
            m.b_vb[s] = decl_b(b, nm.f("codec.b%d", s), RAD_BF16, { rows[s], ch[s] });
        }
    }
    const char* dd = "codec.decoder.model";
    RAD_ARCH_TRY(vconv(b, nm, m.dec0, "codec.dec0", nm.ckpt("%s.0.conv", dd), kLat, kDec, 7, 1,
                       R0, m.b_ux[2], 0, m.b_vs[0], m.b_ccu[3]));
    static const int64_t kDil[3] = { 1, 3, 9 };
    for (int i = 0; i < 4; ++i) {
        VBlock& B = m.vb[i];
        const int64_t ci = ch[i], co = ch[i + 1], s = m.v_ups[i];
        const char* bn = nm.f("codec.b%d", i);
        const char* bs = nm.ckpt("%s.%d.block", dd, i + 1);
        B.a = valpha(b, nm, nm.f("%s.snake", bn), nm.ckpt("%s.0", bs), ci);
        B.op_s = snake_op(b, rows[i], ci, B.a, m.b_vs[i], m.b_vt[i]);
        /* The transposed conv's weight norm is per input channel: its weight is [in, out, k]. */
        const char* wn = nm.f("%s.t.weight", bn);
        RAD_ARCH_TRY(map_wnorm(b, wn, nm.ckpt("%s.1.conv.parametrizations.weight.original0", bs),
                               nm.ckpt("%s.1.conv.parametrizations.weight.original1", bs)));
        B.wt = decl_w(b, wn, RAD_BF16, { ci, co, 2 * s }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                      grp_layer(kGrpCodec));
        B.bt = wcopy(b, nm.f("%s.t.bias", bn), nm.ckpt("%s.1.conv.bias", bs), RAD_BF16, { co },
                     kGrpCodec);
        B.op_t = rw(b, RAD_OP(b, "conv_t1d",
                              RAD_PARAMS(RAD_RANGE("M", 1, rows[i]), RAD_INT("cin", ci),
                                         RAD_INT("cout", co), RAD_INT("stride", s),
                                         RAD_STR("dtype", "bf16"), RAD_INT("pad", 0),
                                         RAD_INT("k", 2 * s)),
                              RAD_WEIGHTS(B.wt, B.bt)),
                    { m.b_vt[i], m.b_ccu[3 + i] }, { m.b_vs[i + 1] });
        if (!B.a || !B.wt || !B.bt || !B.op_t || !B.op_s) return RAD_E_INVAL;
        /* Three residual units (dilations 1, 3, 9): x + conv2(snake2(conv1(snake1(x)))), x the
         * stage buffer, written back in place by conv2's residual. */
        for (int r = 0; r < 3; ++r) {
            VRes& U = B.res[r];
            const char* un = nm.f("%s.res%d", bn, r);
            const char* ub = nm.ckpt("%s.%d.block", bs, r + 2);
            const int64_t R = rows[i + 1];
            U.a1 = valpha(b, nm, nm.f("%s.act1", un), nm.ckpt("%s.0", ub), co);
            U.a2 = valpha(b, nm, nm.f("%s.act2", un), nm.ckpt("%s.2", ub), co);
            U.op_s1 = snake_op(b, R, co, U.a1, m.b_vs[i + 1], m.b_vt[i + 1]);
            RAD_ARCH_TRY(vconv(b, nm, U.c1, nm.f("%s.conv1", un), nm.ckpt("%s.1.conv", ub), co, co,
                               7, kDil[r], R, m.b_vt[i + 1], 0, m.b_va[i + 1], m.b_ccu[4 + i]));
            U.op_s2 = snake_op(b, R, co, U.a2, m.b_va[i + 1], m.b_vb[i + 1]);
            RAD_ARCH_TRY(vconv(b, nm, U.c2, nm.f("%s.conv2", un), nm.ckpt("%s.3.conv", ub), co, co,
                               1, 1, R, m.b_vb[i + 1], m.b_vs[i + 1], m.b_vs[i + 1],
                               m.b_ccu[4 + i]));
            if (!U.a1 || !U.a2 || !U.op_s1 || !U.op_s2) return RAD_E_INVAL;
        }
    }
    m.w_fa = valpha(b, nm, "codec.snake_out", nm.ckpt("%s.5", dd), ch[4]);
    m.op_fs = snake_op(b, rows[4], ch[4], m.w_fa, m.b_vs[4], m.b_vt[4]);
    m.b_vout = decl_b(b, nm.f("codec.out"), RAD_BF16, { rows[4], 1 });
    RAD_ARCH_TRY(vconv(b, nm, m.fin, "codec.conv_out", nm.ckpt("%s.6.conv", dd), ch[4], 1, 7, 1,
                       rows[4], m.b_vt[4], 0, m.b_vout, m.b_ccu[7], "tanh"));
    m.b_audio = decl_b(b, nm.f("codec.audio"), RAD_F32, { kVocChunk * kSamplesPerFrame, 1 },
                       RAD_BUF_PERSIST);
    m.op_fcast = cast_op(b, kVocChunk * kSamplesPerFrame, 1, "bf16", "f32", m.b_vout, m.b_audio);
    return m.w_cn && m.op_cn && m.w_fa && m.op_fs && m.op_fcast && m.op_cnst ? RAD_OK
                                                                             : RAD_E_INVAL;
}

/* ================================================================== the codec's encoder */
/* Voice cloning: the PROMPT pass's reference clips to the codec's ten codes a frame
 * (modded_dac.py's Encoder, then rvq.py's DownsampleResidualVectorQuantize), every clip at once,
 * each a sequence of the pass's bounds, which are in samples and read at each rate through cu_div.
 * In bf16, as fish-speech's server encodes a reference, every quantiser step in f32 (vq_encode)
 * into the prompt rows' code plane at the clip's rows.
 *
 * Declared first: the clips are encoded before anything else of the pass runs, so the encoder's
 * buffers are dead by the backbone's, and only the code plane, which the tower reads, lives on. */
static int declare_encoder(RadBuilder* b, Model& m) {
    Names& nm = m.nm;
    const int64_t L = kMaxRefSamples, D = kLat, qd = kCtHeads * kCtHeadDim;
    const char* es = "codec.encoder.block";
    static const int64_t kDil[3] = { 1, 3, 9 };

    m.b_ein = decl_b(b, nm.f("enc.in"), RAD_BF16, { L, 1 });
    m.op_ecast = rw(b, RAD_OP(b, "cast",
                              RAD_PARAMS(RAD_RANGE("M", 1, L), RAD_INT("n", 1),
                                         RAD_STR("from", "f32"), RAD_STR("to", "bf16")),
                              RAD_NOWEIGHTS),
                    {}, { m.b_ein });
    int64_t rows = L, ch = 64, rate = 1;
    for (int s = 0; s < 4; ++s) {
        m.b_ex[s] = decl_b(b, nm.f("enc.x%d", s), RAD_BF16, { rows, ch });
        m.b_et[s] = decl_b(b, nm.f("enc.t%d", s), RAD_BF16, { rows, ch });
        m.b_ea[s] = decl_b(b, nm.f("enc.a%d", s), RAD_BF16, { rows, ch });
        rows /= kEncStride[s];
        ch *= 2;
    }
    const int64_t R4 = rows, R5 = R4 / 2, F = R5 / 2;   /* 512, 1024 and 2048 samples a row */
    m.b_ex4 = decl_b(b, nm.f("enc.x4"), RAD_BF16, { R4, D });
    m.b_et4 = decl_b(b, nm.f("enc.t4"), RAD_BF16, { R4, D });
    m.b_ea4 = decl_b(b, nm.f("enc.a4"), RAD_BF16, { R4, D });
    RAD_ARCH_TRY(econv(b, nm, m.e0, "codec.enc.0", nm.ckpt("%s.0.conv", es), 1, 64, 7, 1, 1, 1, L,
                       m.b_ein, 0, m.b_ex[0]));
    if (!m.op_ecast) return RAD_E_INVAL;

    /* ---- four blocks: three residual units (dilations 1, 3, 9: x + conv2(snake2(conv1(snake1
     * (x))))), a snake and the strided conv into the next block's rows */
    rows = L;
    ch = 64;
    for (int s = 0; s < 4; ++s) {
        EBlock& B = m.eb[s];
        const char* bn = nm.f("codec.enc.b%d", s);
        const char* bs = nm.ckpt("%s.%d.block", es, s + 1);
        const rad_buf X = m.b_ex[s], T = m.b_et[s], A = m.b_ea[s];
        for (int r = 0; r < 3; ++r) {
            VRes& U = B.res[r];
            const char* un = nm.f("%s.res%d", bn, r);
            const char* ub = nm.ckpt("%s.%d.block", bs, r);
            U.a1 = valpha(b, nm, nm.f("%s.act1", un), nm.ckpt("%s.0", ub), ch);
            U.a2 = valpha(b, nm, nm.f("%s.act2", un), nm.ckpt("%s.2", ub), ch);
            U.op_s1 = snake_op(b, rows, ch, U.a1, X, T);
            RAD_ARCH_TRY(econv(b, nm, U.c1, nm.f("%s.conv1", un), nm.ckpt("%s.1.conv", ub), ch, ch,
                               7, kDil[r], 1, rate, rows, T, 0, A));
            U.op_s2 = snake_op(b, rows, ch, U.a2, A, T);
            RAD_ARCH_TRY(econv(b, nm, U.c2, nm.f("%s.conv2", un), nm.ckpt("%s.3.conv", ub), ch, ch,
                               1, 1, 1, rate, rows, T, X, X));
            if (!U.a1 || !U.a2 || !U.op_s1 || !U.op_s2) return RAD_E_INVAL;
        }
        B.a = valpha(b, nm, nm.f("%s.snake", bn), nm.ckpt("%s.3", bs), ch);
        B.op_s = snake_op(b, rows, ch, B.a, X, T);
        if (!B.a || !B.op_s) return RAD_E_INVAL;
        const int64_t st = kEncStride[s];
        RAD_ARCH_TRY(econv(b, nm, B.down, nm.f("%s.down", bn), nm.ckpt("%s.4.conv", bs), ch,
                           2 * ch, 2 * st, 1, st, rate, rows, T, 0,
                           s < 3 ? m.b_ex[s + 1] : m.b_ex4));
        rows /= st;
        ch *= 2;
        rate *= st;
    }

    /* ---- the last block's transformer (window 512 rows) and its norm, a snake, a conv */
    m.ecs.x = m.b_ex4;
    m.ecs.h = decl_b(b, nm.f("enc.h"), RAD_BF16, { R4, D });
    m.ecs.qkv = decl_b(b, nm.f("enc.qkv"), RAD_BF16, { R4, 3 * qd });
    m.ecs.attn = decl_b(b, nm.f("enc.attn"), RAD_BF16, { R4, qd });
    m.ecs.o = decl_b(b, nm.f("enc.o"), RAD_BF16, { R4, D });
    m.ecs.gu = decl_b(b, nm.f("enc.gu"), RAD_BF16, { R4, 2 * kCtFf });
    m.ecs.ff = decl_b(b, nm.f("enc.ff"), RAD_BF16, { R4, kCtFf });
    m.b_epos = decl_b(b, nm.f("enc.pos"), RAD_I32, { R4 });
    m.op_epos = iota_op(b, R4, 0, 1, m.b_epos);
    const char* tsrc = nm.ckpt("%s.4.block.5", es);
    RAD_ARCH_TRY(declare_ct(b, nm, m.et, "codec.enc.t", tsrc, (int)kEncLayers, R4, kEncWin, 512,
                            m.ecs, 0, m.b_epos));
    m.w_etn = wcopy(b, nm.f("codec.enc.t.norm.weight"), nm.ckpt("%s.norm.weight", tsrc), RAD_F32,
                    { D }, kGrpCodec);
    m.op_etn = norm_op(b, R4, D, kCtEps, 0.0, m.w_etn, { m.b_ex4 }, m.b_et4);
    m.w_e5 = valpha(b, nm, "codec.enc.snake_out", nm.ckpt("%s.5", es), D);
    m.op_e5 = snake_op(b, R4, D, m.w_e5, m.b_et4, m.b_ea4);
    RAD_ARCH_TRY(econv(b, nm, m.e6, "codec.enc.conv_out", nm.ckpt("%s.6.conv", es), D, D, 3, 1, 1,
                       512, R4, m.b_ea4, 0, m.b_ex4));
    if (!m.op_epos || !m.w_etn || !m.op_etn || !m.w_e5 || !m.op_e5) return RAD_E_INVAL;

    /* ---- the quantiser's two 2x downsamplings: a strided conv (plain weights), a ConvNeXt
     * block (causal depthwise conv, layer norm, GELU feed-forward, layer scale, residual) */
    m.b_ed1 = decl_b(b, nm.f("enc.d1"), RAD_BF16, { R5, D });
    m.b_ez = decl_b(b, nm.f("enc.z"), RAD_BF16, { F, D });
    m.b_dua = decl_b(b, nm.f("enc.dua"), RAD_BF16, { R5, D });
    m.b_dub = decl_b(b, nm.f("enc.dub"), RAD_BF16, { R5, D });
    m.b_duh = decl_b(b, nm.f("enc.duh"), RAD_BF16, { R5, 4 * D });
    m.b_dug = decl_b(b, nm.f("enc.dug"), RAD_BF16, { R5, 4 * D });
    for (int i = 0; i < 2; ++i) {
        EDown& E = m.ed[i];
        const int64_t rin = i ? R5 : R4, rout = rin / 2, rr = i ? 1024 : 512;
        const rad_buf xin = i ? m.b_ed1 : m.b_ex4, y = i ? m.b_ez : m.b_ed1;
        const char* ds = nm.ckpt("codec.quantizer.downsample.%d", i);
        auto w = [&](const char* dn, const char* sr, uint32_t dt,
                     std::initializer_list<int64_t> sh) {
            return wcopy(b, nm.f("codec.enc.down%d.%s", i, dn), nm.ckpt("%s.%s", ds, sr), dt, sh,
                         kGrpCodec);
        };
        E.c.w = w("weight", "0.conv.weight", RAD_BF16, { D, D, 2 });
        E.c.b = w("bias", "0.conv.bias", RAD_BF16, { D });
        E.w_dw = w("dw.weight", "1.dwconv.conv.weight", RAD_BF16, { D, 1, 7 });
        E.b_dw = w("dw.bias", "1.dwconv.conv.bias", RAD_BF16, { D });
        E.w_ln = w("ln.weight", "1.norm.weight", RAD_F32, { D });
        E.b_ln = w("ln.bias", "1.norm.bias", RAD_F32, { D });
        E.w_p1 = w("p1.weight", "1.pwconv1.weight", RAD_BF16, { 4 * D, D });
        E.b_p1 = w("p1.bias", "1.pwconv1.bias", RAD_BF16, { 4 * D });
        E.w_p2 = w("p2.weight", "1.pwconv2.weight", RAD_BF16, { D, 4 * D });
        E.b_p2 = w("p2.bias", "1.pwconv2.bias", RAD_BF16, { D });
        E.w_g = w("gamma", "1.gamma", RAD_F32, { D });
        if (!E.c.w || !E.c.b || !E.w_dw || !E.b_dw || !E.w_ln || !E.b_ln || !E.w_p1 || !E.b_p1 ||
            !E.w_p2 || !E.b_p2 || !E.w_g)
            return RAD_E_INVAL;
        E.c.op = econv_op(b, rin, D, D, 2, 1, 2, rr, E.c.w, E.c.b, xin, 0, y);
        E.op_dw = rw(b, RAD_OP(b, "conv1d_dw",
                               RAD_PARAMS(RAD_RANGE("M", 1, rout), RAD_INT("n", D), RAD_INT("k", 7),
                                          RAD_INT("cu_div", 2 * rr), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(E.w_dw, E.b_dw)),
                     { y }, { m.b_dua });
        E.op_ln = rw(b, RAD_OP(b, "layernorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, rout), RAD_INT("n", D),
                                          RAD_F64("eps", 1e-6), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(E.w_ln, E.b_ln)),
                     { m.b_dua }, { m.b_dub });
        E.op_p1 = gemm_bias_op(b, rout, 4 * D, D, E.w_p1, E.b_p1, m.b_dub, m.b_duh);
        E.op_gelu = rows_op(b, "gelu", rout, 4 * D, { m.b_duh }, { m.b_dug });
        E.op_p2 = gemm_bias_op(b, rout, D, 4 * D, E.w_p2, E.b_p2, m.b_dug, m.b_dua);
        E.op_g = mulw_op(b, rout, D, E.w_g, m.b_dua, m.b_dua);
        E.op_add = add_op(b, rout, D, y, m.b_dua, y);
        if (!E.c.op || !E.op_dw || !E.op_ln || !E.op_p1 || !E.op_gelu || !E.op_p2 || !E.op_g ||
            !E.op_add)
            return RAD_E_INVAL;
    }

    /* ---- the pre-quantiser transformer (8 layers, window 128 frames) and its norm, over the
     * encoder's working set */
    const char* psrc = "codec.quantizer.pre_module";
    CtSet P = m.ecs;
    P.x = m.b_ez;
    RAD_ARCH_TRY(declare_ct(b, nm, m.pq, "codec.enc.pre", nm.ckpt("%s", psrc), (int)kCtLayers, F,
                            kAttnWin, kSamplesPerFrame, P, 0, m.b_epos));
    m.w_pqn = wcopy(b, nm.f("codec.enc.pre.norm.weight"), nm.ckpt("%s.norm.weight", psrc),
                    RAD_F32, { D }, kGrpCodec);
    m.op_pqn = norm_op(b, F, D, kCtEps, 0.0, m.w_pqn, { m.b_ez }, m.b_ez);
    if (!m.w_pqn || !m.op_pqn) return RAD_E_INVAL;

    /* ---- the prompt rows' codes: -2 for code 0 and -1 for the others at every row, then each
     * clip's frames at their rows, the semantic step first and the nine residual ones after it,
     * every step quantising what the last left of z */
    m.b_pc = decl_b(b, nm.f("enc.codes"), RAD_I32, { kCodes, kPcW });
    m.op_pc0 = iota_op(b, kPcW, -2, 0, m.b_pc);
    m.op_pc1 = iota_op(b, (kCodes - 1) * kPcW, -1, 0, m.b_pc);
    for (int k = 0; k < kCodes; ++k) {
        EQuant& Q = m.eq[k];
        const char* qs = k == 0 ? nm.ckpt("codec.quantizer.semantic_quantizer.quantizers.0")
                                : nm.ckpt("codec.quantizer.quantizer.quantizers.%d", k - 1);
        const int64_t bins = k == 0 ? kVocab : kResVocab;
        const char* iw = nm.f("codec.enc.vq%d.in.weight", k);
        const char* ow = nm.f("codec.enc.vq%d.out.weight", k);
        RAD_ARCH_TRY(map_wnorm(b, iw, nm.ckpt("%s.in_proj.weight_g", qs),
                               nm.ckpt("%s.in_proj.weight_v", qs)));
        RAD_ARCH_TRY(map_wnorm(b, ow, nm.ckpt("%s.out_proj.weight_g", qs),
                               nm.ckpt("%s.out_proj.weight_v", qs)));
        Q.in_w = decl_w(b, iw, RAD_F32, { kCbDim, D, 1 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                        grp_layer(kGrpCodec));
        Q.in_b = wcopy(b, nm.f("codec.enc.vq%d.in.bias", k), nm.ckpt("%s.in_proj.bias", qs),
                       RAD_F32, { kCbDim }, kGrpCodec);
        Q.cb = wcopy(b, nm.f("codec.enc.vq%d.codebook", k), nm.ckpt("%s.codebook.weight", qs),
                     RAD_F32, { bins, kCbDim }, kGrpCodec);
        Q.out_w = decl_w(b, ow, RAD_F32, { D, kCbDim, 1 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                         grp_layer(kGrpCodec));
        Q.out_b = wcopy(b, nm.f("codec.enc.vq%d.out.bias", k), nm.ckpt("%s.out_proj.bias", qs),
                        RAD_F32, { D }, kGrpCodec);
        if (!Q.in_w || !Q.in_b || !Q.cb || !Q.out_w || !Q.out_b) return RAD_E_INVAL;
        Q.op = rw(b, RAD_OP(b, "vq_encode",
                            RAD_PARAMS(RAD_RANGE("M", 1, F), RAD_INT("n", D),
                                       RAD_INT("dim", kCbDim), RAD_INT("bins", bins),
                                       RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(Q.in_w, Q.in_b, Q.cb, Q.out_w, Q.out_b)),
                  { m.b_ez }, { m.b_ez, m.b_pc });
        if (!Q.op) return RAD_E_INVAL;
    }
    return m.op_pc0 && m.op_pc1 ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== declare */
static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(check_format(meta));
    if (ctx->world_size > 2) {
        fprintf(stderr, "radiance: fishomni: drama-3 is served on one or two cards (--tp 1 or 2): "
                        "its two KV heads and the tower's split stop at two ranks\n");
        return RAD_E_UNSUPPORTED;
    }
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));
    m.lanes = std::min<int64_t>(kMaxLanes, std::max<int64_t>(1, ctx->max_seqs));
    if (m.g.max_tok < kVocContext + kVocChunk) {
        fprintf(stderr, "radiance: fishomni: a pass carries %lld rows; a codec chunk needs %lld\n",
                (long long)m.g.max_tok, (long long)(kVocContext + kVocChunk));
        return RAD_E_INVAL;
    }

    /* Declared in the order the passes issue them where it matters for liveness: the codec's
     * encoder first (PROMPT runs it before everything else when the request has references), the
     * backbone, then the frame head, the tower, the frame's tail, and the codec. The rows' code
     * plane the encoder leaves is named by the tower's ops too, so it lives across the backbone.
     * FRAME runs the head and the tower before the backbone; the buffers that cross that seam
     * (`x`, `h`, the codes, the plane) are named by the ops on both sides, so their live ranges
     * span it. */
    RAD_ARCH_TRY(declare_encoder(b, m));
    RAD_ARCH_TRY(declare_trunk(b, meta, ctx, m));
    RAD_ARCH_TRY(declare_frame_head(b, m));
    RAD_ARCH_TRY(declare_tower(b, meta, m));
    RAD_ARCH_TRY(declare_frame_tail(b, m));
    RAD_ARCH_TRY(declare_codec(b, m));

    if (!ctx->shape_probe) {
        RadAudioGenDecl d{};
        d.name = "drama-3";
        d.tmpl = nullptr;
        d.tmpl_plain = kPlain;
        d.prompt_whole = 1;
        d.first_frame = 0;
        d.max_prompt = kMaxPrompt;
        d.max_frames = kMaxFrames;
        d.frame_rate_num = kSampleRate;
        d.frame_rate_den = kSamplesPerFrame;
        d.lat_num = 1;
        d.lat_den = 1;
        d.samples_per_latent = kSamplesPerFrame;
        d.sample_rate = kSampleRate;
        d.channels = 1;
        d.default_cfg = 1.0;
        d.max_lanes = m.lanes;
        d.status = m.b_status;
        d.audio_out = m.b_audio;
        d.voc_chunk = kVocChunk;
        d.voc_context = kVocContext;
        d.branches = 1;
        d.input_special = 1;
        d.tmpl_ref = kRefPlain;
        d.ref_item = kRefItem;
        d.ref_rate = kSampleRate;
        d.ref_hop = kSamplesPerFrame;
        d.ref_max_samples = kMaxRefSamples;
        d.ref_max = kMaxRefs;
        RAD_ARCH_TRY(rad_declare_audio_generator(b, &d));
    }
    rad_note(b, "rank %d/%d: drama-3 -- backbone %lld layers (%lld heads, %lld KV a rank; %lld "
                "experts of %lld a rank, %lld a token), tower %d layers at KV layers %lld..%lld "
                "(%lld heads a rank, ffn %lld), codec transformer %d layers, %lld lanes",
             m.g.rank, m.g.world, (long long)m.n_main, (long long)m.g.n_head,
             (long long)m.g.n_head_kv, (long long)m.moecfg.n_expert, (long long)m.moecfg.n_ff_exp,
             (long long)m.moecfg.top_k, kTower, (long long)m.n_main,
             (long long)(m.n_main + kTower - 1), (long long)m.gt.n_head, (long long)m.gt.n_ff,
             (int)kCtLayers, (long long)m.lanes);
    rad_note(b, "sampling: temperature %.2f, top-p %.2f for every draw, top-k %lld for the main "
                "token and %lld for each head; a semantic first draw among the lane's last %lld "
                "picks is redrawn at temperature %.2f, top-p %.2f, top-k %lld",
             kTemp, kTopP, (long long)kMainTopK, (long long)kHeadTopK, (long long)kRasWin,
             kRasTemp, kRasTopP, (long long)kRasTopK);
    return RAD_OK;
}

/* ================================================================== step */
/* A codec transformer (declare_ct) over R rows of B.x. */
static void step_ct(RadCtx* c, const std::vector<CodecLayer>& ls, int64_t R, const CtSet& B,
                    RadOperand pos, RadOperand cu) {
    const int64_t qd = kCtHeads * kCtHeadDim;
    for (const CodecLayer& Ly : ls) {
        RAD_ISSUE_N(c, Ly.op_n1, R, brows(B.x, R), RAD_W(Ly.w_n1), brows(B.h, R));
        RAD_ISSUE_N(c, Ly.op_qkv, R, brows(B.h, R), RAD_W(Ly.w_qkv), brows(B.qkv, R));
        RAD_ISSUE_N(c, Ly.op_rope, R, brows(B.qkv, R), pos);
        RAD_ISSUE_N(c, Ly.op_attn, R, bcol(B.qkv, 0, qd, R), bcol(B.qkv, qd, qd, R),
                    bcol(B.qkv, 2 * qd, qd, R), cu, brows(B.attn, R));
        RAD_ISSUE_N(c, Ly.op_o, R, brows(B.attn, R), RAD_W(Ly.w_o), brows(B.o, R));
        RAD_ISSUE_N(c, Ly.op_ls1, R, brows(B.o, R), RAD_W(Ly.w_ls1), brows(B.o, R));
        RAD_ISSUE_N(c, Ly.op_add1, R, brows(B.x, R), brows(B.o, R), brows(B.x, R));
        RAD_ISSUE_N(c, Ly.op_n2, R, brows(B.x, R), RAD_W(Ly.w_n2), brows(B.h, R));
        RAD_ISSUE_N(c, Ly.op_gu, R, brows(B.h, R), RAD_W(Ly.w_gu), brows(B.gu, R));
        RAD_ISSUE_N(c, Ly.op_act, R, brows(B.gu, R), brows(B.ff, R));
        RAD_ISSUE_N(c, Ly.op_down, R, brows(B.ff, R), RAD_W(Ly.w_down), brows(B.o, R));
        RAD_ISSUE_N(c, Ly.op_ls2, R, brows(B.o, R), RAD_W(Ly.w_ls2), brows(B.o, R));
        RAD_ISSUE_N(c, Ly.op_add2, R, brows(B.x, R), brows(B.o, R), brows(B.x, R));
    }
}

/* A PROMPT pass's clips through the encoder (declare_encoder), their frames' codes into the
 * prompt rows' code plane at the rows that name them. */
static void step_encode(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadAudioGenBatch* A = batch->agen_b;
    const int64_t L = A->ref_samples, D = kLat;
    const RadOperand cu = praw(A->ref_cu, RAD_I32, A->n_ref + 1);
    auto conv = [&](const VConv& v, rad_buf x, rad_buf res, rad_buf y, int64_t R, int64_t Ry) {
        RAD_ISSUE_N(c, v.op, R, brows(x, R), RAD_W(v.w), cu, res ? brows(res, R) : RAD_NONE,
                    brows(y, Ry), RAD_W(v.b));
    };
    RAD_ISSUE_N(c, m.op_ecast, L, praw(A->ref_audio, RAD_F32, L), brows(m.b_ein, L));
    conv(m.e0, m.b_ein, 0, m.b_ex[0], L, L);
    int64_t R = L;
    for (int s = 0; s < 4; ++s) {
        const EBlock& B = m.eb[s];
        const rad_buf X = m.b_ex[s], T = m.b_et[s], Ab = m.b_ea[s];
        for (const VRes& U : B.res) {
            RAD_ISSUE_N(c, U.op_s1, R, brows(X, R), RAD_W(U.a1), brows(T, R), RAD_NONE);
            conv(U.c1, T, 0, Ab, R, R);
            RAD_ISSUE_N(c, U.op_s2, R, brows(Ab, R), RAD_W(U.a2), brows(T, R), RAD_NONE);
            conv(U.c2, T, X, X, R, R);
        }
        RAD_ISSUE_N(c, B.op_s, R, brows(X, R), RAD_W(B.a), brows(T, R), RAD_NONE);
        conv(B.down, T, 0, s < 3 ? m.b_ex[s + 1] : m.b_ex4, R, R / kEncStride[s]);
        R /= kEncStride[s];
    }
    /* The last block's transformer, at the clips' rows of 512 samples; the snake and the conv. */
    RAD_ISSUE_N(c, m.op_epos, R, brows(m.b_epos, R));
    step_ct(c, m.et, R, m.ecs, brows(m.b_epos, R), cu);
    RAD_ISSUE_N(c, m.op_etn, R, brows(m.b_ex4, R), RAD_W(m.w_etn), brows(m.b_et4, R));
    RAD_ISSUE_N(c, m.op_e5, R, brows(m.b_et4, R), RAD_W(m.w_e5), brows(m.b_ea4, R), RAD_NONE);
    conv(m.e6, m.b_ea4, 0, m.b_ex4, R, R);
    /* The quantiser's two downsamplings. */
    for (int i = 0; i < 2; ++i) {
        const EDown& E = m.ed[i];
        const rad_buf xin = i ? m.b_ed1 : m.b_ex4, y = i ? m.b_ez : m.b_ed1;
        const int64_t ro = R / 2;
        conv(E.c, xin, 0, y, R, ro);
        RAD_ISSUE_N(c, E.op_dw, ro, brows(y, ro), RAD_W(E.w_dw), cu, brows(m.b_dua, ro),
                    RAD_W(E.b_dw));
        RAD_ISSUE_N(c, E.op_ln, ro, brows(m.b_dua, ro), RAD_W(E.w_ln), RAD_W(E.b_ln),
                    brows(m.b_dub, ro));
        RAD_ISSUE_N(c, E.op_p1, ro, brows(m.b_dub, ro), RAD_W(E.w_p1), RAD_W(E.b_p1), RAD_NONE,
                    brows(m.b_duh, ro));
        RAD_ISSUE_N(c, E.op_gelu, ro, brows(m.b_duh, ro), brows(m.b_dug, ro));
        RAD_ISSUE_N(c, E.op_p2, ro, brows(m.b_dug, ro), RAD_W(E.w_p2), RAD_W(E.b_p2), RAD_NONE,
                    brows(m.b_dua, ro));
        RAD_ISSUE_N(c, E.op_g, ro, brows(m.b_dua, ro), RAD_W(E.w_g), brows(m.b_dua, ro));
        RAD_ISSUE_N(c, E.op_add, ro, brows(y, ro), brows(m.b_dua, ro), brows(y, ro));
        R = ro;
    }
    /* The pre-quantiser transformer and its norm at the frames: z. */
    const int64_t F = R;
    CtSet P = m.ecs;
    P.x = m.b_ez;
    RAD_ISSUE_N(c, m.op_epos, F, brows(m.b_epos, F));
    step_ct(c, m.pq, F, P, brows(m.b_epos, F), cu);
    RAD_ISSUE_N(c, m.op_pqn, F, brows(m.b_ez, F), RAD_W(m.w_pqn), brows(m.b_ez, F));

    /* The rows' codes: a token's everywhere, then each run of rows naming consecutive frames
     * quantised step by step into its columns. */
    RAD_ISSUE_N(c, m.op_pc0, kPcW, brow_slice(m.b_pc, 0, 1, kPcW));
    RAD_ISSUE_N(c, m.op_pc1, (kCodes - 1) * kPcW, brow_slice(m.b_pc, 1, kCodes - 1, kPcW));
    const int64_t T = batch->n_tok;
    for (int64_t t = 0; t < T;) {
        const int32_t f = A->ref_frame_h[t];
        if (f < 0 || f >= F) { ++t; continue; }
        int64_t n = 1;
        while (t + n < T && A->ref_frame_h[t + n] == f + n && f + n < F) ++n;
        for (int k = 0; k < kCodes; ++k) {
            const EQuant& Q = m.eq[k];
            RAD_ISSUE_N(c, Q.op, n, brow_slice(m.b_ez, f, n, D), RAD_W(Q.in_w), RAD_W(Q.in_b),
                        RAD_W(Q.cb), RAD_W(Q.out_w), RAD_W(Q.out_b),
                        bcol_at(m.b_pc, k, kPcW, t, n, 1));
        }
        t += n;
    }
}

static void step_trunk(RadCtx* c, const RadBatch* batch, const Model& m) {
    const int64_t T = batch->n_tok;
    if (m.op_rope_cs && T <= qk_fuse_rows(m.g) && !rope_mixed(batch))
        RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));
    for (const Layer& l : m.layers) {
        if (l.full) l.attn.step(c, batch);
        else        l.gdn.step(c, batch);
        l.mlp.step(c, batch);
    }
    RAD_ISSUE_N(c, m.op_final_norm, T, brows(m.b_x, T), RAD_W(m.w_out_norm), brows(m.a_h.x, T));
}

/* Where the tower runs: its rows' positions and cache slots, the sequences' block tables,
 * context lengths and bounds, and its first layer's hidden state. FRAME also embeds codes. */
struct TowerAt {
    int64_t    R = 0;
    int64_t    max_q = 0;
    RadOperand pos{}, slot{}, bt{}, used{}, cu{};
    rad_buf    h0 = 0;
    bool       frame = false;
    RadOperand seed{}, fidx{};
    bool       labels = false;   /* PROMPT with references: labels off the rows' code plane */
};

static void step_tower(RadCtx* c, const Model& m, const TowerAt& at) {
    const Geom& g = m.gt;
    const int64_t R = at.R, D = g.n_embd, Sx = m.lanes;
    /* Row k of the code-major [10, lanes] plane, narrowed to this pass's lanes. */
    auto code = [&](int k) {
        RadOperand o = brow_slice(m.b_codes, k, 1, Sx);
        o.cols = R;
        return o;
    };
    for (int i = 0; i < kTower; ++i) {
        const TowerLayer& L = m.tw[i];
        const int k = i - kShared;
        if (at.frame) {
            RAD_ISSUE_N(c, L.op_emb, R, code(k < 0 ? 0 : k), RAD_W(L.w_emb), RAD_W(L.w_emb),
                        brows(m.b_te, R));
            if (L.op_eadd)
                RAD_ISSUE_N(c, L.op_eadd, R, brows(m.b_te, R), brows(m.b_ug, R), brows(m.b_te, R));
            RAD_ISSUE_N(c, L.op_en, R, brows(m.b_te, R), RAD_W(L.w_en), bcol(m.b_tcat, 0, D, R));
        }
        if (at.labels) {
            /* Row r's label: the code plane's column r + 1. */
            RAD_ISSUE_N(c, m.op_ptefill, R, brows(m.b_pte, R));
            RAD_ISSUE_N(c, L.op_pemb, R, bcol_at(m.b_pc, k < 0 ? 0 : k, kPcW, 1, R, 1),
                        RAD_W(L.w_emb), RAD_W(L.w_emb), brows(m.b_pte, R));
            if (L.op_peadd)
                RAD_ISSUE_N(c, L.op_peadd, R, brows(m.b_pte, R), brows(m.b_pu, R),
                            brows(m.b_pte, R));
            RAD_ISSUE_N(c, L.op_pen, R, brows(m.b_pte, R), RAD_W(L.w_en),
                        bcol(m.b_tcat, 0, D, R));
        }
        RAD_ISSUE_N(c, L.op_hn, R, brows(i == 0 ? at.h0 : m.b_tho, R), RAD_W(L.w_hn),
                    bcol(m.b_tcat, D, D, R));
        RAD_ISSUE_N(c, L.op_eh, R, brows(m.b_tcat, R), RAD_W(L.w_eh), brows(m.b_tm, R));
        RAD_ISSUE_N(c, L.op_an, R, brows(m.b_tm, R), RAD_W(L.w_an), brows(m.b_th, R));
        RAD_ISSUE_N(c, L.op_q, R, brows(m.b_th, R), RAD_W(L.w_q), brows(m.b_tq, R));
        RAD_ISSUE_N(c, L.op_g, R, brows(m.b_th, R), RAD_W(L.w_g), brows(m.b_tg, R));
        RAD_ISSUE_N(c, L.op_k, R, brows(m.b_th, R), RAD_W(L.w_k), brows(m.b_tk, R));
        RAD_ISSUE_N(c, L.op_v, R, brows(m.b_th, R), RAD_W(L.w_v), brows(m.b_tv, R));
        RAD_ISSUE_N(c, L.op_qn, R * g.n_head, brows(m.b_tq, R), RAD_W(L.w_qn), brows(m.b_tq, R));
        RAD_ISSUE_N(c, L.op_kn, R * g.n_head_kv, brows(m.b_tk, R), RAD_W(L.w_kn),
                    brows(m.b_tk, R));
        RAD_ISSUE_N(c, L.op_rq, R, brows(m.b_tq, R), at.pos);
        RAD_ISSUE_N(c, L.op_rk, R, brows(m.b_tk, R), at.pos);
        RAD_ISSUE_N(c, L.op_kv, R, brows(m.b_tk, R), brows(m.b_tv, R), at.slot,
                    kv_cache(m.kv_full, (int)m.n_main + i));
        RAD_ISSUE_N(c, L.op_attn, at.max_q, brows(m.b_tq, R),
                    kv_cache(m.kv_full, (int)m.n_main + i), at.bt, at.used, RAD_NONE, RAD_NONE,
                    at.cu, brows(m.b_tattn, R));
        RAD_ISSUE_N(c, L.op_sig, R, brows(m.b_tg, R), brows(m.b_tg, R));
        RAD_ISSUE_N(c, L.op_mul, R, brows(m.b_tattn, R), brows(m.b_tg, R), brows(m.b_tattn, R));
        RAD_ISSUE_N(c, L.op_o, R, brows(m.b_tattn, R), RAD_W(L.w_o), brows(m.b_to, R));
        if (L.op_ar1) RAD_ISSUE_N(c, L.op_ar1, R * D, brows(m.b_to, R), RAD_NONE);
        RAD_ISSUE_N(c, L.op_add1, R, brows(m.b_tm, R), brows(m.b_to, R), brows(m.b_tm, R));
        RAD_ISSUE_N(c, L.op_fn, R, brows(m.b_tm, R), RAD_W(L.w_fn), brows(m.b_th, R));
        RAD_ISSUE_N(c, L.op_gu, R, brows(m.b_th, R), RAD_W(L.w_gu), brows(m.b_tgu, R));
        RAD_ISSUE_N(c, L.op_act, R, brows(m.b_tgu, R), brows(m.b_tff, R));
        RAD_ISSUE_N(c, L.op_down, R, brows(m.b_tff, R), RAD_W(L.w_down), brows(m.b_to, R));
        if (L.op_ar2) RAD_ISSUE_N(c, L.op_ar2, R * D, brows(m.b_to, R), RAD_NONE);
        RAD_ISSUE_N(c, L.op_add2, R, brows(m.b_tm, R), brows(m.b_to, R), brows(m.b_tm, R));
        RAD_ISSUE_N(c, L.op_fin, R, brows(m.b_tm, R), RAD_W(L.w_fin), brows(m.b_tho, R));
        if (at.frame && L.op_head) {
            RAD_ISSUE_N(c, L.op_head, R, brows(m.b_tho, R), RAD_W(L.w_head), brows(m.b_hlog, R));
            RAD_ISSUE_N(c, L.op_samp, R, brows(m.b_hlog, R), at.seed, at.fidx, code(k + 1),
                        RAD_NONE, RAD_NONE, RAD_NONE);
        }
    }
}

static void step_prompt(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadAudioGenBatch* A = batch->agen_b;
    const int64_t T = batch->n_tok, D = m.g.n_embd;
    /* References: their clips encoded into the rows' code plane first. */
    const bool refs = A->n_ref > 0;
    if (refs) step_encode(c, batch, m);
    RAD_ISSUE_N(c, m.op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok),
                brows(m.b_x, T));
    if (m.op_embed_ar) RAD_ISSUE_N(c, m.op_embed_ar, T * D, brows(m.b_x, T), RAD_NONE);
    RadOperand sec1 = RAD_W(m.w_cb), sec2 = RAD_W(m.w_cb), sec3 = RAD_W(m.w_cb);
    sec1.rows = kVocab;
    sec2.offset = kVocab * D;
    sec2.rows = kVocab;
    sec3.offset = 2 * kVocab * D;
    sec3.rows = (kCodes - 3) * kVocab;
    if (refs)
        RAD_ISSUE_N(c, m.op_pfb, T, bcol_at(m.b_pc, 0, kPcW, 0, T, 2), RAD_W(m.w_cemb), sec1,
                    brows(m.b_x, T));
    step_trunk(c, batch, m);

    /* The lane: the prompt's last hidden state, an empty window, no unseen codes. */
    const int64_t S = batch->n_seq, lane = A->lane_h[0];
    RAD_ISSUE_N(c, m.op_plast, S, brows(m.a_h.x, T), praw(A->last_row, RAD_I32, S),
                brows(m.b_last, S));
    RAD_ISSUE_N(c, m.op_pstore, S, brows(m.b_last, S), praw(A->slot, RAD_I32, S),
                brows(m.b_hid, m.lanes));
    RAD_ISSUE_N(c, m.op_uzero, 1, brow_slice(m.b_u, lane, 1, D));
    {
        RadOperand w = brow_slice(m.b_ras, lane, 1, kRasWin);
        RAD_ISSUE_N(c, m.op_rasinit, kRasWin, w);
        /* A reference's frames among the last rows are picks the window has seen. */
        if (refs && T >= kRasWin)
            RAD_ISSUE_N(c, m.op_rascopy, kRasWin,
                        bcol_at(m.b_pc, 0, kPcW, T - kRasWin, kRasWin, 1), w);
    }

    /* The tower over every prompt row. A text prompt's labels are all text, so every code
     * embedding is 0; with references a row before a frame is labelled with it, and a frame's row
     * also carries its unseen codes (tower.py, prefill). The last row's pass is FRAME 0's, which
     * writes its cache rows again. */
    const RadKVGroupBatch* kvb = kv_batch(batch, m.kv_full);
    if (!refs) {
        RAD_ISSUE_N(c, m.op_efill, T, bcol(m.b_tcat, 0, D, T));
    } else {
        RAD_ISSUE_N(c, m.op_pufill, T, brows(m.b_pu, T));
        RAD_ISSUE_N(c, m.op_pu, T, bcol_at(m.b_pc, 2, kPcW, 0, T, kCodes - 2), sec2, sec3,
                    brows(m.b_pu, T));
    }
    TowerAt at{};
    at.R = T;
    at.max_q = batch->max_q_len;
    at.pos = praw(batch->positions, RAD_I32, T);
    at.slot = praw(kvb->slot_mapping, RAD_I32, T);
    at.bt = praw2(kvb->block_table, RAD_I32, batch->n_seq, kvb->block_table_pitch);
    at.used = praw(kvb->seqused, RAD_I32, batch->n_seq);
    at.cu = praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1);
    at.h0 = m.a_h.x;
    at.labels = refs;
    step_tower(c, m, at);
}

static void step_frame(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadAudioGenBatch* A = batch->agen_b;
    const int64_t S = A->n_lane, Sx = m.lanes;
    const RadOperand seed = praw(A->seed, RAD_U32, 2 * S), frame = praw(A->frame, RAD_I32, S);
    const RadOperand lanes = praw(A->lane, RAD_I32, S), slot = praw(A->slot, RAD_I32, S);
    auto code = [&](int k) {
        RadOperand o = brow_slice(m.b_codes, k, 1, Sx);
        o.cols = S;
        return o;
    };

    /* 1. the main token, from the hidden state of the row before the frame's */
    RAD_ISSUE_N(c, m.op_hsel, S, brows(m.b_hid, m.lanes), slot, brows(m.b_plane, S));
    RAD_ISSUE_N(c, m.op_head, S, brows(m.b_plane, S), RAD_W(m.w_cand), brows(m.b_slog, S));
    RAD_ISSUE_N(c, m.op_samp, S, brows(m.b_slog, S), seed, frame, brows(m.b_first, S), RAD_NONE,
                RAD_NONE, RAD_NONE);
    RAD_ISSUE_N(c, m.op_samp_ras, S, brows(m.b_slog, S), seed, frame, brows(m.b_alt, S),
                RAD_NONE, RAD_NONE, RAD_NONE);
    RAD_ISSUE_N(c, m.op_ras, S, brows(m.b_first, S), brows(m.b_alt, S), lanes,
                brows(m.b_ras, m.lanes), code(0));
    RAD_ISSUE_N(c, m.op_status, S, code(0), lanes, brows(m.b_status, m.lanes));
    RAD_ISSUE_N(c, m.op_ugather, S, brows(m.b_u, m.lanes), slot, brows(m.b_ug, S));

    /* 2. the tower at that row: its position, slot and context, the frame's own sequences */
    const RadKVGroupBatch* kvb = kv_batch(batch, m.kv_full);
    TowerAt at{};
    at.R = S;
    at.max_q = 1;
    at.pos = praw(A->prev_pos, RAD_I32, S);
    at.slot = praw(A->prev_slot, RAD_I32, S);
    at.bt = praw2(kvb->block_table, RAD_I32, batch->n_seq, kvb->block_table_pitch);
    at.used = praw(A->prev_used, RAD_I32, S);
    at.cu = praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1);
    at.h0 = m.b_plane;
    at.frame = true;
    at.seed = seed;
    at.fidx = frame;
    step_tower(c, m, at);

    /* 3. the frame into the lane's history and the lane's unseen codes; the backbone over its
     * embedding at the frame's own row */
    RAD_ISSUE_N(c, m.op_store, S, bcol_at(m.b_codes, 0, Sx, 0, S, kCodes), lanes, frame,
                brows(m.b_status, m.lanes), brows(m.b_hist, m.lanes * kCodes));
    RadOperand sec2 = RAD_W(m.w_cb), sec3 = RAD_W(m.w_cb), sec1 = RAD_W(m.w_cb);
    const int64_t D = m.g.n_embd;
    sec1.rows = kVocab;
    sec2.offset = kVocab * D;
    sec2.rows = kVocab;
    sec3.offset = 2 * kVocab * D;
    sec3.rows = (kCodes - 3) * kVocab;
    RAD_ISSUE_N(c, m.op_u, S, bcol_at(m.b_codes, 2, Sx, 0, S, kCodes - 2), sec2, sec3,
                brows(m.b_unew, S));
    RAD_ISSUE_N(c, m.op_uscat, S, brows(m.b_unew, S), slot, brows(m.b_u, m.lanes));
    RAD_ISSUE_N(c, m.op_fb, S, bcol_at(m.b_codes, 0, Sx, 0, S, 2), RAD_W(m.w_cemb), sec1,
                brows(m.b_x, S));
    step_trunk(c, batch, m);
    RAD_ISSUE_N(c, m.op_fstore, S, brows(m.a_h.x, S), slot, brows(m.b_hid, m.lanes));
}

static void step_vocode(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadAudioGenBatch* A = batch->agen_b;
    const int64_t nl = A->voc_latents, n = nl - A->voc_left, L = A->voc_lane;
    /* The convolutions run over the last kConvFrames frames, or all of them when fewer. */
    const int64_t nc = std::min<int64_t>(nl, kConvFrames), c0 = nl - nc;
    const RadOperand nlen = praw(batch->cu_seqlens + 1, RAD_I32, 1);
    const RadOperand clen = nc == nl ? nlen : brows(m.b_cconst, 1);

    /* 1. the bounds at each rate: the transformer's frames, then the convolutions' */
    if (nc != nl) RAD_ISSUE_N(c, m.op_cnst, 1, brows(m.b_cconst, 1));
    RAD_ISSUE_N(c, m.op_ccu[0], 2, nlen, brows(m.b_ccu[0], 2));
    for (int s = 1; s < kRates; ++s) RAD_ISSUE_N(c, m.op_ccu[s], 2, clen, brows(m.b_ccu[s], 2));

    /* 2. the codes, dequantised: the lane's frames [voc_frame0, + nl) of each codebook */
    for (int k = 0; k < kCodes; ++k) {
        RadOperand codes = bcol_at(m.b_hist, L * kCodes + k, kMaxFrames, A->voc_frame0, nl, 1);
        RAD_ISSUE_N(c, m.op_rq[k], nl, codes, RAD_W(m.w_cbk[k]), RAD_W(m.w_cbk[k]),
                    brows(m.b_rq, nl));
        RAD_ISSUE_N(c, m.qo[k].op, nl, brows(m.b_rq, nl), RAD_W(m.qo[k].w), brows(m.b_ccu[0], 2),
                    k == 0 ? RAD_NONE : brows(m.b_cx, nl), brows(m.b_cx, nl), RAD_W(m.qo[k].b));
    }

    /* 3. the transformer over every frame of the pass */
    {
        CtSet B;
        B.x = m.b_cx; B.h = m.b_ch; B.qkv = m.b_cqkv; B.attn = m.b_cattn; B.o = m.b_co;
        B.gu = m.b_cgu; B.ff = m.b_cff;
        step_ct(c, m.ct, nl, B, praw(batch->positions, RAD_I32, nl), brows(m.b_ccu[0], 2));
    }
    RAD_ISSUE_N(c, m.op_cn, nc, brow_slice(m.b_cx, c0, nc, kLat), RAD_W(m.w_cn),
                brows(m.b_ux[0], nc));

    /* 4. the upsamplings, at 2 and 4 rows a frame */
    for (int i = 0; i < 2; ++i) {
        const Upsample& U = m.up[i];
        const int64_t rin = nc << i, rout = nc << (i + 1);
        RAD_ISSUE_N(c, U.op_t, rin, brows(m.b_ux[i], rin), RAD_W(U.w_t), RAD_W(U.b_t),
                    brows(m.b_ccu[1 + i], 2), brows(m.b_ux[i + 1], rout));
        RAD_ISSUE_N(c, U.op_dw, rout, brows(m.b_ux[i + 1], rout), RAD_W(U.w_dw),
                    brows(m.b_ccu[2 + i], 2), brows(m.b_ua, rout), RAD_W(U.b_dw));
        RAD_ISSUE_N(c, U.op_ln, rout, brows(m.b_ua, rout), RAD_W(U.w_ln), RAD_W(U.b_ln),
                    brows(m.b_ub, rout));
        RAD_ISSUE_N(c, U.op_p1, rout, brows(m.b_ub, rout), RAD_W(U.w_p1), RAD_W(U.b_p1), RAD_NONE,
                    brows(m.b_uh, rout));
        RAD_ISSUE_N(c, U.op_gelu, rout, brows(m.b_uh, rout), brows(m.b_ug2, rout));
        RAD_ISSUE_N(c, U.op_p2, rout, brows(m.b_ug2, rout), RAD_W(U.w_p2), RAD_W(U.b_p2),
                    RAD_NONE, brows(m.b_ua, rout));
        RAD_ISSUE_N(c, U.op_g, rout, brows(m.b_ua, rout), RAD_W(U.w_g), brows(m.b_ua, rout));
        RAD_ISSUE_N(c, U.op_add, rout, brows(m.b_ux[i + 1], rout), brows(m.b_ua, rout),
                    brows(m.b_ux[i + 1], rout));
    }

    /* 5. the decoder */
    int64_t rows[5];
    rows[0] = 4 * nc;
    for (int i = 0; i < 4; ++i) rows[i + 1] = rows[i] * m.v_ups[i];
    auto conv = [&](const VConv& v, rad_buf x, rad_buf res, rad_buf y, int64_t R, rad_buf cub) {
        RAD_ISSUE_N(c, v.op, R, brows(x, R), RAD_W(v.w), brows(cub, 2),
                    res ? brows(res, R) : RAD_NONE, brows(y, R), RAD_W(v.b));
    };
    conv(m.dec0, m.b_ux[2], 0, m.b_vs[0], rows[0], m.b_ccu[3]);
    for (int i = 0; i < 4; ++i) {
        const VBlock& B = m.vb[i];
        const int64_t r0 = rows[i], r1 = rows[i + 1];
        RAD_ISSUE_N(c, B.op_s, r0, brows(m.b_vs[i], r0), RAD_W(B.a), brows(m.b_vt[i], r0),
                    RAD_NONE);
        RAD_ISSUE_N(c, B.op_t, r0, brows(m.b_vt[i], r0), RAD_W(B.wt), RAD_W(B.bt),
                    brows(m.b_ccu[3 + i], 2), brows(m.b_vs[i + 1], r1));
        for (int r = 0; r < 3; ++r) {
            const VRes& U = B.res[r];
            RAD_ISSUE_N(c, U.op_s1, r1, brows(m.b_vs[i + 1], r1), RAD_W(U.a1),
                        brows(m.b_vt[i + 1], r1), RAD_NONE);
            conv(U.c1, m.b_vt[i + 1], 0, m.b_va[i + 1], r1, m.b_ccu[4 + i]);
            RAD_ISSUE_N(c, U.op_s2, r1, brows(m.b_va[i + 1], r1), RAD_W(U.a2),
                        brows(m.b_vb[i + 1], r1), RAD_NONE);
            conv(U.c2, m.b_vb[i + 1], m.b_vs[i + 1], m.b_vs[i + 1], r1, m.b_ccu[4 + i]);
        }
    }
    RAD_ISSUE_N(c, m.op_fs, rows[4], brows(m.b_vs[4], rows[4]), RAD_W(m.w_fa),
                brows(m.b_vt[4], rows[4]), RAD_NONE);
    conv(m.fin, m.b_vt[4], 0, m.b_vout, rows[4], m.b_ccu[7]);
    /* 6. the chunk's own samples: its last n frames */
    const int64_t ns = n * kSamplesPerFrame;
    RAD_ISSUE_N(c, m.op_fcast, ns, brow_slice(m.b_vout, rows[4] - ns, ns, 1), brows(m.b_audio, ns));
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_slot(c)];
    if (!batch->gen || !batch->agen_b) {
        rad_step_fail(c, "fishomni is an audio generator and runs generator passes only");
        return;
    }
    switch (batch->gen) {
    case RAD_GEN_PASS_PROMPT: step_prompt(c, batch, m); return;
    case RAD_GEN_PASS_FRAME:  step_frame(c, batch, m); return;
    case RAD_GEN_PASS_VOCODE: step_vocode(c, batch, m); return;
    default:
        rad_step_fail(c, "fishomni: a generator pass of a kind this plugin does not know");
    }
}

/* ================================================================== the probe */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->gen_rows = kMaxPrompt;
    out->gen_audio = 1;
    out->gen_ctx = kMaxPrompt + kMaxFrames + 1;
    return RAD_OK;
}

/* ================================================================== the request's text */
/* A Unicode White_Space code point's length at `s`, else 0. Python's str.strip() without
 * arguments also strips U+001C..U+001F, which fish-speech avoids (generate.py,
 * UNICODE_WHITE_SPACE); so does this. */
static size_t white_space_at(const unsigned char* s, size_t n) {
    if (n >= 1 && ((s[0] >= 0x09 && s[0] <= 0x0D) || s[0] == 0x20)) return 1;
    if (n >= 2 && s[0] == 0xC2 && (s[1] == 0x85 || s[1] == 0xA0)) return 2;
    if (n >= 3 && s[0] == 0xE1 && s[1] == 0x9A && s[2] == 0x80) return 3;           /* U+1680 */
    if (n >= 3 && s[0] == 0xE2 && s[1] == 0x80 &&
        ((s[2] >= 0x80 && s[2] <= 0x8A) || s[2] == 0xA8 || s[2] == 0xA9 || s[2] == 0xAF))
        return 3;                                          /* U+2000..200A, 2028, 2029, 202F */
    if (n >= 3 && s[0] == 0xE2 && s[1] == 0x81 && s[2] == 0x9F) return 3;           /* U+205F */
    if (n >= 3 && s[0] == 0xE3 && s[1] == 0x80 && s[2] == 0x80) return 3;           /* U+3000 */
    return 0;
}

/* The length of a canonical speaker tag at `s` (`<|speaker:N|>`, N 0..15 without a leading
 * zero), else 0. */
static size_t speaker_tag_at(const char* s, size_t n) {
    static const char kHead[] = "<|speaker:";
    const size_t h = sizeof kHead - 1;
    if (n < h || std::strncmp(s, kHead, h) != 0) return 0;
    size_t i = h;
    while (i < n && s[i] >= '0' && s[i] <= '9') ++i;
    const size_t digits = i - h;
    if (digits == 0 || digits > 2 || i + 1 >= n || s[i] != '|' || s[i + 1] != '>') return 0;
    if (digits == 2 && s[h] == '0') return 0;
    const int v = digits == 1 ? s[h] - '0' : (s[h] - '0') * 10 + (s[h + 1] - '0');
    return v <= 15 ? i + 2 : 0;
}

static int64_t refuse(char* out, int64_t cap, const char* why) {
    if (out && cap > 0) std::snprintf(out, (size_t)cap, "%s", why);
    return RAD_E_INVAL;
}

/* `s` without the Unicode White_Space at either end. */
static std::string strip_white(const std::string& s) {
    const unsigned char* u = (const unsigned char*)s.data();
    size_t a = 0, z = s.size();
    for (size_t w; a < z && (w = white_space_at(u + a, z - a)) > 0;) a += w;
    /* From the back: a White_Space code point is one to three bytes, so try each ending. */
    for (bool more = true; more && z > a;) {
        more = false;
        for (size_t w = 1; w <= 3 && w <= z - a; ++w)
            if (white_space_at(u + z - w, w) == w) {
                z -= w;
                more = true;
                break;
            }
    }
    return s.substr(a, z - a);
}

/* What fish-speech's server hands the model (chunking.py, split_drama3_text): each speaker's turn
 * as "<|speaker:N|> " and its text stripped of white space at both ends, turns joined by a newline,
 * text before the first tag speaker 0's; untagged text stripped and given speaker 0
 * (generate.py, with_speaker_tag). Its model was trained on that form: the same two turns joined
 * by a space instead are spoken through less often. `r` is validated already. */
static std::string speaker_turns(const std::string& r) {
    if (r.find("<|speaker") == std::string::npos) return "<|speaker:0|> " + strip_white(r);
    std::string outs, id = "0";
    size_t at = 0;
    auto turn = [&](const std::string& who, size_t a, size_t z) {
        const std::string body = strip_white(r.substr(a, z - a));
        if (body.empty()) return;
        if (!outs.empty()) outs += "\n";
        outs += "<|speaker:" + who + "|> " + body;
    };
    for (size_t t = r.find("<|speaker"); t != std::string::npos;) {
        turn(id, at, t);
        const size_t len = speaker_tag_at(r.data() + t, r.size() - t);
        id = r.substr(t + 10, len - 12);      /* the digits between "<|speaker:" and "|>" */
        at = t + len;
        t = r.find("<|speaker", at);
    }
    turn(id, at, r.size());
    return outs;
}

/* generate.py's build_drama3_prompt for a reference's transcript: speaker 0's tag and a space in
 * front unless a tag leads after white space (with_speaker_tag), and white space ahead of a
 * leading tag moved behind it, so the tag follows "Text: " directly. `r` is validated already. */
static std::string reference_text(const std::string& r) {
    const unsigned char* u = (const unsigned char*)r.data();
    size_t lead = 0;
    for (size_t w; lead < r.size() && (w = white_space_at(u + lead, r.size() - lead)) > 0;)
        lead += w;
    const size_t tag = speaker_tag_at(r.data() + lead, r.size() - lead);
    if (!tag) return "<|speaker:0|> " + r;
    return r.substr(lead, tag) + r.substr(0, lead) + r.substr(lead + tag);
}

/* generate.py's validate_drama3_text: the input, or a reference's transcript, is not empty, holds
 * none of the tokens only the prompt builder writes, and names speakers only as <|speaker:0|> to
 * <|speaker:15|>; then the input's speaker turns are written as the reference's server writes
 * them (speaker_turns), and a transcript is tagged as its prompt builder tags it. */
static int64_t text_prepare(int field, const char* in, char* out, int64_t cap) {
    if (!in) return RAD_E_INVAL;
    std::string r = in;
    if (field == RAD_TEXT_INPUT || field == RAD_TEXT_REFERENCE) {
        const bool ref = field == RAD_TEXT_REFERENCE;
        const unsigned char* u = (const unsigned char*)r.data();
        const size_t n = r.size();
        size_t lead = 0;
        for (size_t w; lead < n && (w = white_space_at(u + lead, n - lead)) > 0;) lead += w;
        bool blank = true;
        for (size_t i = lead; i < n && blank;) {
            const size_t w = white_space_at(u + i, n - i);
            if (w == 0) blank = false;
            else i += w;
        }
        if (blank)
            return refuse(out, cap, ref ? "the reference's text is empty: it is what its clip says"
                                        : "the input is empty: there is nothing to speak");
        static const char* kStructural[] = { "<|im_start|>", "<|im_end|>", "<|voice|>",
                                             "<|text|>", "<|interleave|>", "<|semantic:" };
        for (const char* t : kStructural)
            if (r.find(t) != std::string::npos) {
                char why[128];
                std::snprintf(why, sizeof why, "%s is written by the prompt builder, not %s", t,
                              ref ? "a reference's text" : "the input");
                return refuse(out, cap, why);
            }
        for (size_t at = r.find("<|speaker"); at != std::string::npos;
             at = r.find("<|speaker", at + 1))
            if (!speaker_tag_at(r.data() + at, n - at))
                return refuse(out, cap, "speaker tags are <|speaker:0|> to <|speaker:15|>");
        r = ref ? reference_text(r) : speaker_turns(r);
        if (r.empty()) return refuse(out, cap, "the input is empty: there is nothing to speak");
    }
    if ((int64_t)r.size() < cap) {
        std::memcpy(out, r.data(), r.size());
        out[r.size()] = 0;
    }
    return (int64_t)r.size();
}

}  /* namespace fishomni */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(fishomni)
extern "C" int64_t rad_arch_text_prepare(int field, const char* in, char* out, int64_t cap) {
    return fishomni::text_prepare(field, in, out, cap);
}
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(fishomni, "fishomni", "bsgemm-moe", "0.1.0",
                "Fish Audio drama-3: the Qwen3.5-35B-A3B backbone with converted routed experts, "
                "a 13-layer codebook tower over its full-attention cache, and Fish S2's modified "
                "DAC; 44.1 kHz mono speech")
#endif
