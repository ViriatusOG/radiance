/* breeze -- Breeze-TTS-2: speech from text, voice design or a reference recording; 24 kHz mono.
 *
 * ============================== THE MODEL ==============================
 *
 * Four generation networks (BreezeBlue's BreezeForConditionalGeneration and its audio tokenizer),
 * plus the reference-audio encoder in reference.h for voice cloning:
 *
 *   Text encoder: T5Gemma 2's text encoder, Gemma 3 1B's shape (26 layers of 1152, 4 query heads
 *   over one KV head of 256, gelu-tanh GLU 6912, sandwich RMS norms with a (1 + w) gain), read
 *   bidirectionally over the whole prompt: sliding layers see 255 keys behind and 256 ahead, every
 *   sixth layer sees all of it. Its output, projected 1152 -> 2048, is the backbone's prompt: the
 *   backbone never embeds a text token.
 *
 *   Backbone: Qwen3 (28 layers of 2048, 16 query heads over 8 KV heads of 128, q and k RMS-normed,
 *   theta 1e6). From its last hidden state a head of 2052 writes each frame's first code: one of
 *   2048, or the end of the speech (2051). A frame's sixteen codes, embedded from one shared table
 *   and summed, are its next input.
 *
 *   Depth decoder: 12 layers of 1024 (8 query heads over 2 KV heads of 128, Llama 3's rotary
 *   scaling, no q/k norms) writes the frame's other fifteen codes one at a time from the backbone's
 *   hidden state and the codes before them, each from its own head. Sampling everywhere: guidance
 *   u + g (c - u) when the request asks for it (an unconditional prompt without the voice
 *   description), temperature 0.9, top 50; the backbone's pick also takes a repetition penalty of
 *   1.1 over the piece's earlier first codes.
 *
 *   Codec: the Qwen3-TTS 12 Hz tokenizer's decoder: the codes dequantised through two residual
 *   quantisers (the first code, the other fifteen), a causal 3-tap conv, an 8-layer transformer of
 *   512 (causal, 72 frames of attention), two 2x upsamplings with ConvNeXt blocks, then four
 *   transposed convolutions (8, 5, 4, 3) with snake-beta residual units: 1920 samples a frame.
 *
 * The passes are the core's audio generator in its chunked form (abi/rad_builder.h,
 * rad_declare_audio_generator with voc_chunk; core/gen/audio_driver.h): PROMPT runs the encoder
 * and the backbone over both branches' whole prompts, FRAME samples one frame of every speech in
 * flight and runs the backbone over its embedding, VOCODE plays a few frames of one lane as soon as
 * they are made. Every layer of the codec is causal, so a chunk is decoded exactly by running the
 * codec again over the frames its outputs reach back to (kVocContext) and keeping only its own
 * samples; no state is carried between chunks.
 *
 * ============================== STATE ==============================
 *
 * A request is a lane: its branches' last hidden states, its first codes so far (the penalty's
 * history), and every frame's sixteen codes, which the vocoder reads back. The depth decoder's
 * sixteen positions are computed incrementally, each from the keys and values the earlier
 * positions of its sequence stored (step_depth_inc).
 *
 * One card: the model is 3.6B parameters, and a speech in flight is bound by the step's latency,
 * which tensor parallelism would only lengthen.
 */
#include <arch/rad_arch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace breeze {

using namespace rad::arch;

/* ================================================================== the model's constants */
constexpr int64_t kMaxLanes = 4;
/* The longest prompt and speech: the reference runs at most 1500 frames (120 s) in a 2048-row
 * context. */
constexpr int64_t kMaxPrompt = 512, kMaxFrames = 1500;
/* The frame's codes: 16 codebooks of 2048, rows of 2051 in the shared table (2048..2050 are never
 * sampled); the backbone's head has a 2052nd row, the end of the speech. */
constexpr int64_t kCodes = 16, kVocab = 2051, kHeadRows = 2052, kEnd = 2051;
constexpr int64_t kSupLo = 2048, kSupHi = 2051, kTopK = 50;
static const double kTemp = 0.9;
static const double kPenalty = 1.1;

/* The text encoder: a sliding layer sees 255 keys behind and 256 ahead (transformers' sliding mask
 * at 512); full layers see the whole prompt, with linear rotary scaling. */
constexpr int64_t kEncWin = 256, kEncWinRight = 257;
static const double kEncThetaSliding = 10000.0, kEncThetaFull = 1000000.0, kEncFullScale = 8.0;
/* bf16(sqrt(1152)), as the reference holds it. */
static const double kEncEmbedScale = 34.0;

/* The codec: 12.5 frames a second of 1920 samples. A chunk is at most kVocChunk frames; its
 * outputs reach back kConvCtx frames through the convolutions after the transformer, whose rows
 * reach back kAttnWin - 1 frames, whose inputs reach back 2 through the first conv. */
constexpr int64_t kSamplesPerFrame = 1920, kSampleRate = 24000;
constexpr int64_t kVocChunk = 4, kConvCtx = 12, kAttnWin = 72;
constexpr int64_t kVocContext = 2 + (kAttnWin - 1) + kConvCtx;
constexpr int64_t kConvFrames = kConvCtx + kVocChunk;
constexpr int64_t kCbBins = 2048, kCbDim = 256;
/* One clean reference of at most ten seconds (rounded to complete codec frames). */
constexpr int64_t kMaxRefFrames = 125, kMaxRefSamples = kMaxRefFrames * kSamplesPerFrame;
/* The codec's sequence rates, rows a frame: the transformer's frames, the convolutions' frames,
 * the two upsamplings', and each decoder block's output (8, 5, 4, 3 times). */
constexpr int kRates = 8;
static const int64_t kRate[kRates] = { 1, 1, 2, 4, 32, 160, 640, 1920 };

/* The prompt (the reference's tts_instruction and tts_plain templates, speaker S0). */
static const char* kTemplate = "<bos>[S0]<ins_bos>{instructions}<ins_eos>{input}";
static const char* kPlain = "<bos>[S0]{input}";

/* Weight groups (movement units) by network. */
enum : int {
    kGrpEnc   = 0,      /* the text encoder's layers: 0.. */
    kGrpLm    = 100,
    kGrpDepth = 200,
    kGrpCodec = 300,
};

/* ================================================================== the networks */
/* A Gemma 3 encoder layer: fused q|k|v, per-head (1 + w) norms and rotary, bidirectional
 * attention, sandwich norms around attention and the gelu-tanh GLU. */
struct EncLayer {
    rad_weight w_n1 = 0, w_n2 = 0, w_n3 = 0, w_n4 = 0, w_qn = 0, w_kn = 0;
    rad_weight w_qkv = 0, w_o = 0, w_gu = 0, w_down = 0;
    rad_op     op_n1 = 0, op_qkv = 0, op_qknr = 0, op_attn = 0, op_o = 0, op_n2 = 0, op_add1 = 0;
    rad_op     op_n3 = 0, op_gu = 0, op_act = 0, op_down = 0, op_n4 = 0, op_add2 = 0;
};

/* A Qwen3 layer (the backbone, over the paged cache) or a Llama layer (the depth decoder, dense,
 * no q/k norms): separate q, k, v (paged attention reads a contiguous query), rotary, SwiGLU. */
struct LmLayer {
    rad_weight w_n1 = 0, w_n2 = 0, w_qn = 0, w_kn = 0;
    rad_weight w_q = 0, w_k = 0, w_v = 0, w_o = 0, w_gu = 0, w_down = 0;
    rad_op     op_n1 = 0, op_q = 0, op_k = 0, op_v = 0, op_qn = 0, op_kn = 0, op_rq = 0, op_rk = 0;
    rad_op     op_kv = 0, op_attn = 0, op_attn_p = 0, op_o = 0, op_add1 = 0;
    rad_op     op_n2 = 0, op_gu = 0, op_act = 0, op_down = 0, op_add2 = 0;
};

/* The codec transformer's layer: fused q|k|v with rotary at the frames' positions, causal
 * attention over a 72-frame window, layer scales on both residual branches. */
struct CodecLayer {
    rad_weight w_n1 = 0, w_n2 = 0, w_qkv = 0, w_o = 0, w_gu = 0, w_down = 0, w_ls1 = 0, w_ls2 = 0;
    rad_op     op_n1 = 0, op_qkv = 0, op_rope = 0, op_attn = 0, op_o = 0, op_ls1 = 0, op_add1 = 0;
    rad_op     op_n2 = 0, op_gu = 0, op_act = 0, op_down = 0, op_ls2 = 0, op_add2 = 0;
};

/* A ConvNeXt block after a 2x upsampling. */
struct Upsample {
    rad_weight w_t = 0, b_t = 0, w_dw = 0, b_dw = 0, w_ln = 0, b_ln = 0, w_p1 = 0, b_p1 = 0;
    rad_weight w_p2 = 0, b_p2 = 0, w_g = 0;
    rad_op     op_t = 0, op_dw = 0, op_ln = 0, op_p1 = 0, op_gelu = 0, op_p2 = 0, op_g = 0;
    rad_op     op_add = 0;
};

/* A causal conv with its bias. */
struct VConv {
    rad_weight w = 0, b = 0;
    rad_op     op = 0;
};
struct VRes {
    rad_weight a1 = 0, b1 = 0, a2 = 0, b2 = 0;
    rad_op     op_s1 = 0, op_s2 = 0;
    VConv      c1{}, c2{};
};
struct VBlock {
    rad_weight a = 0, b = 0, wt = 0, bt = 0;
    rad_op     op_s = 0, op_t = 0;
    int64_t    cin = 0, cout = 0, stride = 1;
    VRes       res[3];
};

/* The depth decoder's rotary scaling (rope_scaling, rope_type "llama3"). */
struct Llama3 {
    double factor = 0, low = 0, high = 0, orig = 0;
};

struct RefConv {
    rad_weight w = 0, b = 0;
    rad_op op = 0;
};
struct RefLayer {
    rad_weight n1 = 0, b1 = 0, n2 = 0, b2 = 0, qkv = 0, o = 0, fc1 = 0, fc2 = 0;
    rad_weight ls1 = 0, ls2 = 0;
    rad_op norm1 = 0, norm2 = 0, proj = 0, rope = 0, attn = 0, out = 0;
    rad_op f1 = 0, act = 0, f2 = 0, scale1 = 0, scale2 = 0, add1 = 0, add2 = 0;
};
struct Reference {
    RefConv first{}, res1[4]{}, res2[4]{}, down[4]{}, last{}, sample{};
    rad_buf x[5]{}, a[5]{}, h[4]{}, z = 0, pad = 0, sampled = 0;
    rad_op elu[5]{}, elu_h[4]{};
    rad_buf th = 0, qkv = 0, attn = 0, o = 0, ff = 0, act = 0;
    RefLayer layer[8]{};
    rad_buf index = 0, cu = 0, text_pos = 0, text_cu = 0;
    rad_op indices = 0, copy_i = 0, copy_f = 0, text_indices = 0;
    rad_buf q[2]{}, codes = 0, eos_codes = 0;
    rad_weight qi[2]{}, cb[16]{};
    rad_op project[2]{}, quant[16]{}, embed = 0, eos = 0;
};

struct Model {
    Reference ref{};
    /* The model holds the reference encoder (reference_stored), whose first weight is then already
     * in the name map. */
    bool  ref_stored = false;
    Names nm{""};
    Llama3 l3{};
    Geom  ge{}, gl{}, gd{};         /* the text encoder's, the backbone's, the depth decoder's */
    int64_t lanes = 0;
    int64_t S = 0;                  /* a FRAME pass's rows at most: two a lane */
    rad_kvgroup kv = 0;

    /* ---- the text encoder (PROMPT) */
    rad_weight w_etok = 0, w_enorm = 0, w_eproj = 0;
    rad_op  op_eembed = 0, op_enorm = 0, op_eproj = 0;
    rad_buf b_ex = 0, b_eh = 0, b_eqkv = 0, b_eattn = 0, b_eo = 0, b_egu = 0, b_eff = 0;
    std::vector<EncLayer> enc;

    /* ---- the backbone */
    rad_weight w_norm = 0, w_head = 0;
    rad_buf b_x = 0, b_h = 0, b_q = 0, b_k = 0, b_v = 0, b_attn = 0, b_h2 = 0, b_gu = 0, b_ff = 0;
    rad_buf b_last = 0;
    std::vector<LmLayer> lm;

    /* ---- the frame */
    rad_buf b_hid = 0;            /* [2 lanes, 2048] PERSIST: row 2 lane + branch */
    rad_buf b_status = 0;         /* [lanes] i32 PERSIST */
    rad_buf b_hist = 0;           /* [lanes, 2052] i32 PERSIST: the first codes picked so far */
    rad_buf b_codes_all = 0;      /* [lanes * 16, kMaxFrames] i32 PERSIST: each lane's codes */
    rad_buf b_plane = 0;          /* [S, 2048] the lanes' hidden states */
    rad_buf b_slog = 0;           /* [S, 2052] f32 */
    rad_buf b_codes = 0;          /* [16, S] i32: code-major, each code twice (both branches) */
    rad_op  op_hsel = 0, op_head = 0, op_samp0 = 0, op_status = 0, op_store = 0, op_fb = 0;
    rad_op  op_plast = 0, op_pnorm = 0, op_pstore = 0;
    rad_weight w_cb = 0;          /* [16 * 2051, 2048] the shared codebook table */

    /* ---- the depth decoder */
    rad_weight w_din = 0, w_dnorm = 0, w_dheads = 0;
    rad_op  op_din[16] = {}, op_demb[16] = {}, op_dscat = 0, op_dgather = 0;
    rad_op  op_dnorm = 0, op_dhead = 0, op_dsamp[16] = {}, op_iota[16] = {};
    rad_op  op_dpos = 0, op_dcu = 0, op_dzero = 0;
    rad_buf b_dseq = 0, b_dx = 0, b_dh = 0, b_dq = 0, b_dk = 0, b_dv = 0, b_dattn = 0, b_dh2 = 0;
    rad_buf b_dgu = 0, b_dff = 0, b_didx = 0, b_dpos = 0, b_dcu = 0, b_demb = 0, b_drow = 0;
    rad_buf b_dsel = 0, b_dplane = 0, b_dlog = 0;
    std::vector<LmLayer> dep;
    /* The incremental form (step_depth_inc): each layer's keys and values at the frame's
     * positions, [n_layer * R, kv width] (sequence s's position j at row l R + 16 s + j); the
     * bounds {0, 1} of a one-row query run and {0, p + 1} of pass p's key run; the pass's
     * positions. */
    rad_buf b_dkc = 0, b_dvc = 0, b_dcuq = 0, b_dcuk = 0, b_dposi = 0;
    rad_op  op_dcuq = 0, op_dcuk[16] = {}, op_dposi[16] = {}, op_dnorm_i = 0;
    std::vector<rad_op> op_dsk, op_dsv, op_dxa;
    /* Its first pass, positions 0 and 1 together (2 S rows, a sequence's pair adjacent): the
     * pairs' rows {2 s} and {2 s + 1}, their positions {0, 1, 0, 1, ...}, a pair's store rows
     * {0, 1}, the pair's query bounds {0, 2}, and the causal attention over a pair. */
    rad_buf b_dev = 0, b_dodd = 0, b_dpos2 = 0, b_d01 = 0, b_dcuq2 = 0;
    rad_op  op_dev = 0, op_dodd = 0, op_dpos2 = 0, op_d01 = 0, op_dcuq2 = 0;
    std::vector<rad_op> op_dxa2;

    /* ---- the codec (VOCODE) */
    rad_weight w_cb0 = 0, w_cbr = 0, w_qo0 = 0, w_qor = 0;
    rad_op  op_rq0 = 0, op_rqr = 0, op_qp0 = 0, op_qpr = 0, op_qadd = 0;
    VConv   pre{};
    rad_weight w_ip = 0, b_ip = 0, w_cn = 0, w_op = 0, b_op = 0;
    rad_op  op_ip = 0, op_cn = 0, op_op = 0;
    std::vector<CodecLayer> ct;
    Upsample up[2];
    VConv   dec0{};
    VBlock  vb[4];
    int64_t v_ups[4] = { 8, 5, 4, 3 };
    rad_weight w_fa = 0, w_fb = 0;
    rad_op  op_fs = 0, op_fcast = 0, op_cnst = 0, op_ccu[8] = {};
    VConv   fin{};
    rad_buf b_rq0 = 0, b_rqr = 0, b_qz = 0, b_qt = 0, b_pre = 0, b_cx = 0, b_ch = 0, b_cqkv = 0;
    rad_buf b_cattn = 0, b_co = 0, b_cgu = 0, b_cff = 0, b_cy = 0;
    rad_buf b_cconst = 0, b_ccu[8] = {};
    rad_buf b_ux[3] = {}, b_ua = 0, b_ub = 0, b_uh = 0, b_ug = 0;
    rad_buf b_vs[5] = {}, b_vt[5] = {}, b_va[5] = {}, b_vb[5] = {};
    rad_buf b_vout = 0, b_audio = 0;
};

static Model g_model[MAX_SLOTS];

/* ================================================================== declare helpers */
static rad_op norm_op(RadBuilder* b, int64_t rows, int64_t n, double eps, rad_weight w, rad_buf x,
                      rad_buf y, double wadd = 0.0) {
    return rw(b, RAD_OP(b, "rmsnorm",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                   RAD_STR("dtype", "bf16"), RAD_F64("wadd", wadd)),
                        RAD_WEIGHTS(w)),
              { x }, { y });
}

static rad_op cu_op(RadBuilder* b, int64_t mult, rad_buf len, rad_buf y) {
    return rw(b, RAD_OP(b, "cu_repeat",
                        RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("mult", mult)),
                        RAD_NOWEIGHTS),
              { len }, { y });
}

/* A plain weight copied from the checkpoint, declared at `dtype`. */
static rad_weight wcopy(RadBuilder* b, Names& nm, const char* dn, const char* src, uint32_t dtype,
                        std::initializer_list<int64_t> shape, int group) {
    (void)nm;
    if (map_copy(b, dn, src) < 0) return 0;
    return decl_w(b, dn, dtype, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(group));
}

/* Two or three checkpoint tensors stacked along dim 0 into one declared weight. */
static rad_weight wcat(RadBuilder* b, const char* dn, std::initializer_list<const char*> src,
                       std::initializer_list<int64_t> shape, int group) {
    if (map_concat(b, dn, src) < 0) return 0;
    return decl_w(b, dn, RAD_BF16, shape, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(group));
}

/* ================================================================== the text encoder */
static int declare_encoder(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& g = m.ge;
    Names& nm = m.nm;
    const char* tc = "text_encoder_config";
    g.n_layer = rad_meta_geti(meta, nm.f("%s.num_hidden_layers", tc), 26);
    g.n_embd = rad_meta_geti(meta, nm.f("%s.hidden_size", tc), 1152);
    g.n_head = rad_meta_geti(meta, nm.f("%s.num_attention_heads", tc), 4);
    g.n_head_kv = rad_meta_geti(meta, nm.f("%s.num_key_value_heads", tc), 1);
    g.head_dim = rad_meta_geti(meta, nm.f("%s.head_dim", tc), 256);
    g.n_ff = rad_meta_geti(meta, nm.f("%s.intermediate_size", tc), 6912);
    const int64_t V = rad_meta_geti(meta, nm.f("%s.vocab_size", tc), 262158);
    const int64_t qpas = rad_meta_geti(meta, nm.f("%s.query_pre_attn_scalar", tc), 256);
    const char* lt = rad_meta_gets(meta, nm.f("%s.layer_types", tc), "");
    if (g.n_embd != 1152 || g.head_dim != 256 || g.n_head != 4 || g.n_head_kv != 1 ||
        rad_meta_geti(meta, nm.f("%s.sliding_window", tc), 512) != 512 || !lt || !*lt) {
        fprintf(stderr, "radiance: breeze: the text encoder is not the released one (width %lld, "
                        "%lld x %lld heads over %lld)\n", (long long)g.n_embd,
                (long long)g.n_head, (long long)g.head_dim, (long long)g.n_head_kv);
        return RAD_E_UNSUPPORTED;
    }
    /* Which layers see the whole prompt, from layer_types. */
    std::vector<bool> full;
    {
        std::string s(lt);
        size_t at = 0;
        while (at < s.size()) {
            size_t e = s.find(' ', at);
            if (e == std::string::npos) e = s.size();
            const std::string w = s.substr(at, e - at);
            if (!w.empty()) full.push_back(w == "full_attention");
            at = e + 1;
        }
    }
    if ((int64_t)full.size() != g.n_layer) {
        fprintf(stderr, "radiance: breeze: layer_types names %zu layers for %lld\n", full.size(),
                (long long)g.n_layer);
        return RAD_E_UNSUPPORTED;
    }
    const int64_t T = g.max_tok, D = g.n_embd, hd = g.head_dim;
    const int64_t qd = g.n_head * hd, kd = g.n_head_kv * hd, Wq = qd + 2 * kd;
    const double eps = 1e-6;

    m.b_ex = decl_b(b, nm.f("enc.x"), RAD_BF16, { T, D });
    m.b_eh = decl_b(b, nm.f("enc.h"), RAD_BF16, { T, D });
    m.b_eqkv = decl_b(b, nm.f("enc.qkv"), RAD_BF16, { T, Wq });
    m.b_eattn = decl_b(b, nm.f("enc.attn"), RAD_BF16, { T, qd });
    m.b_eo = decl_b(b, nm.f("enc.o"), RAD_BF16, { T, D });
    m.b_egu = decl_b(b, nm.f("enc.gu"), RAD_BF16, { T, 2 * g.n_ff });
    m.b_eff = decl_b(b, nm.f("enc.ff"), RAD_BF16, { T, g.n_ff });

    m.w_etok = wcopy(b, nm, nm.f("enc.tok.weight"), "text_encoder.embed_tokens.weight", RAD_BF16,
                     { V, D }, kGrpEnc);
    m.op_eembed = rw(b, RAD_OP(b, "embed_lookup",
                               RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", D),
                                          RAD_INT("n_vocab", V), RAD_STR("dtype", "bf16"),
                                          RAD_F64("mult", kEncEmbedScale)),
                               RAD_WEIGHTS(m.w_etok)),
                     {}, { m.b_ex });

    m.enc.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l) {
        EncLayer& L = m.enc[(size_t)l];
        const int li = (int)l, grp = kGrpEnc + 1 + li;
        auto src = [&](const char* s) { return nm.ckpt("text_encoder.layers.%d.%s", li, s); };
        auto gain = [&](const char* dn, const char* s, int64_t n) {
            return wcopy(b, nm, nm.f("enc.%d.%s", li, dn), src(s), RAD_F32, { n }, grp);
        };
        L.w_n1 = gain("n1.weight", "pre_self_attn_layernorm.weight", D);
        L.w_n2 = gain("n2.weight", "post_self_attn_layernorm.weight", D);
        L.w_n3 = gain("n3.weight", "pre_feedforward_layernorm.weight", D);
        L.w_n4 = gain("n4.weight", "post_feedforward_layernorm.weight", D);
        L.w_qn = gain("q_norm.weight", "self_attn.q_norm.weight", hd);
        L.w_kn = gain("k_norm.weight", "self_attn.k_norm.weight", hd);
        L.w_qkv = wcat(b, nm.f("enc.%d.qkv.weight", li),
                       { src("self_attn.q_proj.weight"), src("self_attn.k_proj.weight"),
                         src("self_attn.v_proj.weight") }, { Wq, D }, grp);
        L.w_o = wcopy(b, nm, nm.f("enc.%d.o.weight", li), src("self_attn.o_proj.weight"), RAD_BF16,
                      { D, qd }, grp);
        L.w_gu = wcat(b, nm.f("enc.%d.gate_up.weight", li),
                      { src("mlp.gate_proj.weight"), src("mlp.up_proj.weight") },
                      { 2 * g.n_ff, D }, grp);
        L.w_down = wcopy(b, nm, nm.f("enc.%d.down.weight", li), src("mlp.down_proj.weight"),
                         RAD_BF16, { D, g.n_ff }, grp);
        L.op_n1 = norm_op(b, T, D, eps, L.w_n1, m.b_ex, m.b_eh, 1.0);
        L.op_qkv = gemm_op(b, T, Wq, D, L.w_qkv, m.b_eh, m.b_eqkv);
        const bool fl = full[(size_t)l];
        L.op_qknr = rw(b, RAD_OP(b, "qk_norm_rope",
                                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                            RAD_INT("n_head", g.n_head),
                                            RAD_INT("n_head_kv", g.n_head_kv),
                                            RAD_F64("theta", fl ? kEncThetaFull : kEncThetaSliding),
                                            RAD_F64("eps", eps),
                                            RAD_F64("scale", fl ? kEncFullScale : 1.0),
                                            RAD_STR("mode", "neox"), RAD_F64("wadd", 1.0)),
                                 RAD_WEIGHTS(L.w_qn, L.w_kn)),
                       { m.b_eqkv }, { m.b_eqkv });
        const double scale = 1.0 / std::sqrt((double)qpas);
        L.op_attn = fl ? rw(b, RAD_OP(b, "attn_dense",
                                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                                 RAD_INT("gqa", g.n_head / g.n_head_kv),
                                                 RAD_INT("causal", 0), RAD_STR("dtype", "bf16"),
                                                 RAD_F64("scale", scale),
                                                 RAD_INT("max_seqlen", kMaxPrompt)),
                                      RAD_NOWEIGHTS),
                            { m.b_eqkv }, { m.b_eattn })
                       : rw(b, RAD_OP(b, "attn_dense",
                                      RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", hd),
                                                 RAD_INT("gqa", g.n_head / g.n_head_kv),
                                                 RAD_INT("causal", 0), RAD_STR("dtype", "bf16"),
                                                 RAD_F64("scale", scale),
                                                 RAD_INT("max_seqlen", kMaxPrompt),
                                                 RAD_INT("window", kEncWin),
                                                 RAD_INT("window_right", kEncWinRight)),
                                      RAD_NOWEIGHTS),
                            { m.b_eqkv }, { m.b_eattn });
        L.op_o = gemm_op(b, T, D, qd, L.w_o, m.b_eattn, m.b_eo);
        L.op_n2 = norm_op(b, T, D, eps, L.w_n2, m.b_eo, m.b_eh, 1.0);
        L.op_add1 = add_op(b, T, D, m.b_ex, m.b_eh, m.b_ex);
        L.op_n3 = norm_op(b, T, D, eps, L.w_n3, m.b_ex, m.b_eh, 1.0);
        L.op_gu = gemm_op(b, T, 2 * g.n_ff, D, L.w_gu, m.b_eh, m.b_egu);
        L.op_act = rows_op(b, "gelu_mul", T, g.n_ff, { m.b_egu }, { m.b_eff });
        L.op_down = gemm_op(b, T, D, g.n_ff, L.w_down, m.b_eff, m.b_eo);
        L.op_n4 = norm_op(b, T, D, eps, L.w_n4, m.b_eo, m.b_eh, 1.0);
        L.op_add2 = add_op(b, T, D, m.b_ex, m.b_eh, m.b_ex);
        if (!L.w_n1 || !L.w_n2 || !L.w_n3 || !L.w_n4 || !L.w_qn || !L.w_kn || !L.w_qkv || !L.w_o ||
            !L.w_gu || !L.w_down || !L.op_qknr || !L.op_attn || !L.op_act)
            return RAD_E_INVAL;
    }
    m.w_enorm = wcopy(b, nm, nm.f("enc.norm.weight"), "text_encoder.norm.weight", RAD_F32, { D },
                      kGrpEnc);
    m.op_enorm = norm_op(b, T, D, eps, m.w_enorm, m.b_ex, m.b_eh, 1.0);
    m.w_eproj = wcopy(b, nm, nm.f("enc.proj.weight"), "text_encoder_proj.weight", RAD_BF16,
                      { m.gl.n_embd, D }, kGrpEnc);
    m.op_eproj = gemm_op(b, T, m.gl.n_embd, D, m.w_eproj, m.b_eh, m.b_x);
    return m.w_etok && m.w_enorm && m.w_eproj && m.op_eembed ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the backbone, the depth */
/* A layer's weights and ops over `rows` rows. The backbone's (paged) has q/k norms and NeoX rotary
 * at theta; the depth decoder's (dense) has Llama 3's rotary and no norms. */
static int declare_layer(RadBuilder* b, Names& nm, const Geom& g, const Llama3& l3, LmLayer& L,
                         int li, const char* dn, const char* pre, int group, int64_t rows,
                         bool paged, rad_kvgroup kv, rad_buf x, rad_buf h, rad_buf bq, rad_buf bk,
                         rad_buf bv, rad_buf attn, rad_buf h2, rad_buf gu, rad_buf ff,
                         rad_buf x_in, rad_buf pos = 0) {
    const int64_t D = g.n_embd, hd = g.head_dim, qd = g.n_head * hd, kd = g.n_head_kv * hd;
    auto src = [&](const char* s) { return nm.ckpt("%s.%d.%s", pre, li, s); };
    L.w_n1 = wcopy(b, nm, nm.f("%s.%d.attn_norm.weight", dn, li), src("input_layernorm.weight"),
                   RAD_F32, { D }, group);
    L.w_n2 = wcopy(b, nm, nm.f("%s.%d.ffn_norm.weight", dn, li),
                   src("post_attention_layernorm.weight"), RAD_F32, { D }, group);
    L.w_q = wcopy(b, nm, nm.f("%s.%d.q.weight", dn, li), src("self_attn.q_proj.weight"), RAD_BF16,
                  { qd, D }, group);
    L.w_k = wcopy(b, nm, nm.f("%s.%d.k.weight", dn, li), src("self_attn.k_proj.weight"), RAD_BF16,
                  { kd, D }, group);
    L.w_v = wcopy(b, nm, nm.f("%s.%d.v.weight", dn, li), src("self_attn.v_proj.weight"), RAD_BF16,
                  { kd, D }, group);
    L.w_o = wcopy(b, nm, nm.f("%s.%d.o.weight", dn, li), src("self_attn.o_proj.weight"), RAD_BF16,
                  { D, qd }, group);
    L.w_gu = wcat(b, nm.f("%s.%d.gate_up.weight", dn, li),
                  { src("mlp.gate_proj.weight"), src("mlp.up_proj.weight") }, { 2 * g.n_ff, D },
                  group);
    L.w_down = wcopy(b, nm, nm.f("%s.%d.down.weight", dn, li), src("mlp.down_proj.weight"),
                     RAD_BF16, { D, g.n_ff }, group);
    if (!L.w_n1 || !L.w_n2 || !L.w_q || !L.w_k || !L.w_v || !L.w_o || !L.w_gu || !L.w_down)
        return RAD_E_INVAL;
    /* Layer 0 of the depth stack reads its input from the rebuilt sequence, which must survive the
     * stack; every other layer reads and writes the residual. */
    L.op_n1 = norm_op(b, rows, D, g.eps, L.w_n1, x_in, h);
    L.op_q = gemm_op(b, rows, qd, D, L.w_q, h, bq);
    L.op_k = gemm_op(b, rows, kd, D, L.w_k, h, bk);
    L.op_v = gemm_op(b, rows, kd, D, L.w_v, h, bv);
    auto rope = [&](int64_t heads, rad_buf t) {
        if (paged)
            return rw(b, RAD_OP(b, "rope",
                                RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", hd),
                                           RAD_INT("n_head", heads), RAD_INT("n_head_kv", 0),
                                           RAD_F64("theta", g.theta), RAD_F64("scale", 1.0),
                                           RAD_STR("mode", "neox")),
                                RAD_NOWEIGHTS),
                      { t }, { t });
        return rw(b, RAD_OP(b, "rope",
                            RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", hd),
                                       RAD_INT("n_head", heads), RAD_INT("n_head_kv", 0),
                                       RAD_F64("theta", g.theta), RAD_F64("scale", 1.0),
                                       RAD_STR("mode", "neox"),
                                       RAD_F64("llama3_factor", l3.factor),
                                       RAD_F64("llama3_low", l3.low),
                                       RAD_F64("llama3_high", l3.high),
                                       RAD_F64("llama3_orig", l3.orig)),
                            RAD_NOWEIGHTS),
                  { t, pos }, { t });
    };
    if (paged) {
        L.w_qn = wcopy(b, nm, nm.f("%s.%d.q_norm.weight", dn, li), src("self_attn.q_norm.weight"),
                       RAD_F32, { hd }, group);
        L.w_kn = wcopy(b, nm, nm.f("%s.%d.k_norm.weight", dn, li), src("self_attn.k_norm.weight"),
                       RAD_F32, { hd }, group);
        if (!L.w_qn || !L.w_kn) return RAD_E_INVAL;
        /* Each norm in place: a row's scale is computed before the row is written. */
        L.op_qn = norm_op(b, rows * g.n_head, hd, g.eps, L.w_qn, bq, bq);
        L.op_kn = norm_op(b, rows * g.n_head_kv, hd, g.eps, L.w_kn, bk, bk);
    }
    L.op_rq = rope(g.n_head, bq);
    L.op_rk = rope(g.n_head_kv, bk);
    if (paged) {
        /* A frame is one row a sequence, decode's band; a prompt attends to its own rows. */
        L.op_attn = rw(b, RAD_OP(b, "attn_paged",
                           RAD_PARAMS(RAD_RANGE("q_len", 1, 1), RAD_INT("head_dim", hd),
                                      RAD_INT("gqa", g.n_head / g.n_head_kv), RAD_INT("causal", 1),
                                      RAD_INT("window", 0), RAD_INT("n_head", g.n_head),
                                      RAD_INT("block_size", RAD_KV_BLOCK),
                                      RAD_INT("max_seqs", g.max_seqs),
                                      RAD_INT("max_ctx", g.max_ctx),
                                      RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", g.kv_dtype)),
                           RAD_NOWEIGHTS),
                       { bq }, { attn });
        L.op_attn_p = rw(b, RAD_OP(b, "attn_prefix_self",
                             RAD_PARAMS(RAD_INT("q_len", rows), RAD_INT("head_dim", hd),
                                        RAD_INT("gqa", g.n_head / g.n_head_kv),
                                        RAD_INT("block_size", RAD_KV_BLOCK), RAD_INT("causal", 1),
                                        RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", g.kv_dtype),
                                        RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                        RAD_INT("max_ctx", g.max_ctx), RAD_INT("n_head", g.n_head),
                                        RAD_INT("max_seqs", g.max_seqs), RAD_STR("qk", "bf16")),
                             RAD_NOWEIGHTS),
                         { bq, bk, bv }, { attn });
        L.op_kv = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", hd),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     { bk, bv }, {});
    } else {
        L.op_attn = rw(b, RAD_OP(b, "attn_dense",
                           RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("head_dim", hd),
                                      RAD_INT("gqa", g.n_head / g.n_head_kv), RAD_INT("causal", 1),
                                      RAD_STR("dtype", "bf16"),
                                      RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                      RAD_INT("max_seqlen", kCodes)),
                           RAD_NOWEIGHTS),
                       { bq, bk, bv }, { attn });
    }
    L.op_o = gemm_op(b, rows, D, qd, L.w_o, attn, h2);
    L.op_add1 = add_op(b, rows, D, x_in, h2, x);
    L.op_n2 = norm_op(b, rows, D, g.eps, L.w_n2, x, h);
    L.op_gu = gemm_op(b, rows, 2 * g.n_ff, D, L.w_gu, h, gu);
    L.op_act = rows_op(b, "silu_mul", rows, g.n_ff, { gu }, { ff });
    L.op_down = gemm_op(b, rows, D, g.n_ff, L.w_down, ff, h2);
    L.op_add2 = add_op(b, rows, D, x, h2, x);
    return L.op_rq && L.op_rk && L.op_attn && L.op_act ? RAD_OK : RAD_E_INVAL;
}

static void step_layer(RadCtx* c, const Model& m, const Geom& g, const LmLayer& L,
                       const RadBatch* batch, int64_t T, rad_buf x, rad_buf x_in, rad_buf h,
                       rad_buf bq, rad_buf bk, rad_buf bv, rad_buf attn, rad_buf h2, rad_buf gu,
                       rad_buf ff, const RadOperand& pos, const RadOperand& cu, int layer) {
    RAD_ISSUE(c, L.op_n1, brows(x_in, T), RAD_W(L.w_n1), brows(h, T));
    RAD_ISSUE(c, L.op_q, brows(h, T), RAD_W(L.w_q), brows(bq, T));
    RAD_ISSUE(c, L.op_k, brows(h, T), RAD_W(L.w_k), brows(bk, T));
    RAD_ISSUE(c, L.op_v, brows(h, T), RAD_W(L.w_v), brows(bv, T));
    if (L.op_qn) {
        RAD_ISSUE_N(c, L.op_qn, T * g.n_head, brows(bq, T), RAD_W(L.w_qn), brows(bq, T));
        RAD_ISSUE_N(c, L.op_kn, T * g.n_head_kv, brows(bk, T), RAD_W(L.w_kn), brows(bk, T));
    }
    RAD_ISSUE(c, L.op_rq, brows(bq, T), pos);
    RAD_ISSUE(c, L.op_rk, brows(bk, T), pos);
    if (L.op_kv) {
        const RadKVGroupBatch* kvb = kv_batch(batch, m.kv);
        const RadOperand bt =
            praw2(kvb->block_table, RAD_I32, batch->n_seq, kvb->block_table_pitch);
        const RadOperand bcu = praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1);
        if (batch->gen == RAD_GEN_PASS_PROMPT) {
            /* The prefix first, read from the cache before this pass's rows go in. */
            RAD_ISSUE(c, L.op_attn_p, brows(bq, T), brows(bk, T), brows(bv, T),
                      kv_cache(m.kv, layer), bt,
                      praw(batch->agen_b->prefix, RAD_I32, batch->n_seq), bcu, RAD_NONE, RAD_NONE,
                      RAD_NONE, RAD_NONE, brows(attn, T), RAD_NONE);
            RAD_ISSUE(c, L.op_kv, brows(bk, T), brows(bv, T), praw(kvb->slot_mapping, RAD_I32, T),
                      kv_cache(m.kv, layer));
        } else {
            RAD_ISSUE(c, L.op_kv, brows(bk, T), brows(bv, T), praw(kvb->slot_mapping, RAD_I32, T),
                      kv_cache(m.kv, layer));
            RAD_ISSUE_N(c, L.op_attn, batch->max_q_len, brows(bq, T), kv_cache(m.kv, layer), bt,
                        praw(kvb->seqused, RAD_I32, batch->n_seq), RAD_NONE, RAD_NONE, bcu,
                        brows(attn, T));
        }
    } else {
        RAD_ISSUE_N(c, L.op_attn, T, brows(bq, T), brows(bk, T), brows(bv, T), cu, brows(attn, T));
    }
    RAD_ISSUE(c, L.op_o, brows(attn, T), RAD_W(L.w_o), brows(h2, T));
    RAD_ISSUE(c, L.op_add1, brows(x_in, T), brows(h2, T), brows(x, T));
    RAD_ISSUE(c, L.op_n2, brows(x, T), RAD_W(L.w_n2), brows(h, T));
    RAD_ISSUE(c, L.op_gu, brows(h, T), RAD_W(L.w_gu), brows(gu, T));
    RAD_ISSUE(c, L.op_act, brows(gu, T), brows(ff, T));
    RAD_ISSUE(c, L.op_down, brows(ff, T), RAD_W(L.w_down), brows(h2, T));
    RAD_ISSUE(c, L.op_add2, brows(x, T), brows(h2, T), brows(x, T));
}

static int declare_backbone(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                            Model& m) {
    Geom& g = m.gl;
    Names& nm = m.nm;
    const int64_t T = g.max_tok, D = g.n_embd;
    (void)meta;
    {
        RadKVGroupDecl kd{};
        kd.kind = RAD_KV_FULL;
        kd.dtype = rad_kv_cache_dtype(ctx, RAD_BF16);
        kd.n_head_kv = g.n_head_kv;
        kd.head_dim = g.head_dim;
        m.kv = rad_decl_kv_group(b, nm.f("kv_lm"), &kd);
        if (!m.kv) return RAD_E_INVAL;
        for (int64_t l = 0; l < g.n_layer; ++l) RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv));
        g.kv_dtype = rad_dtype_name(kd.dtype);
    }
    m.b_x  = decl_b(b, nm.f("lm.x"), RAD_BF16, { T, D });
    m.b_h  = decl_b(b, nm.f("lm.h"), RAD_BF16, { T, D });
    m.b_q  = decl_b(b, nm.f("lm.q"), RAD_BF16, { T, g.n_head * g.head_dim });
    m.b_k  = decl_b(b, nm.f("lm.k"), RAD_BF16, { T, g.n_head_kv * g.head_dim });
    m.b_v  = decl_b(b, nm.f("lm.v"), RAD_BF16, { T, g.n_head_kv * g.head_dim });
    m.b_attn = decl_b(b, nm.f("lm.attn"), RAD_BF16, { T, g.n_head * g.head_dim });
    m.b_h2 = decl_b(b, nm.f("lm.h2"), RAD_BF16, { T, D });
    m.b_gu = decl_b(b, nm.f("lm.gu"), RAD_BF16, { T, 2 * g.n_ff });
    m.b_ff = decl_b(b, nm.f("lm.ff"), RAD_BF16, { T, g.n_ff });
    m.b_last = decl_b(b, nm.f("lm.last"), RAD_BF16, { m.S, D });
    m.lm.resize((size_t)g.n_layer);
    for (int64_t l = 0; l < g.n_layer; ++l)
        RAD_ARCH_TRY(declare_layer(b, nm, g, m.l3, m.lm[(size_t)l], (int)l, "lm",
                                   "backbone_model.layers",
                                   kGrpLm + (int)l, T, true, m.kv, m.b_x, m.b_h, m.b_q, m.b_k,
                                   m.b_v, m.b_attn, m.b_h2, m.b_gu, m.b_ff, m.b_x));
    m.w_norm = wcopy(b, nm, nm.f("lm.norm.weight"), "backbone_model.norm.weight", RAD_F32, { D },
                     kGrpLm);
    m.w_head = wcopy(b, nm, nm.f("lm.head.weight"), "lm_head.weight", RAD_BF16, { kHeadRows, D },
                     kGrpLm);
    return m.w_norm && m.w_head ? RAD_OK : RAD_E_INVAL;
}

/* ================================================================== the frame */
static int declare_frame(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Geom& gd = m.gd;
    Names& nm = m.nm;
    const int64_t S = m.S, D = m.gl.n_embd, lanes = m.lanes, R = (int64_t)kCodes * S;
    m.b_hid    = decl_b(b, nm.f("frame.hid"), RAD_BF16, { 2 * lanes, D }, RAD_BUF_PERSIST);
    m.b_status = decl_b(b, nm.f("frame.status"), RAD_I32, { lanes }, RAD_BUF_PERSIST);
    m.b_hist   = decl_b(b, nm.f("frame.hist"), RAD_I32, { lanes, kHeadRows }, RAD_BUF_PERSIST);
    m.b_codes_all = decl_b(b, nm.f("frame.history"), RAD_I32, { lanes * kCodes, kMaxFrames },
                           RAD_BUF_PERSIST);
    m.b_plane  = decl_b(b, nm.f("frame.plane"), RAD_BF16, { S, D });
    m.b_slog   = decl_b(b, nm.f("frame.slog"), RAD_F32, { S, kHeadRows });
    m.b_codes  = decl_b(b, nm.f("frame.codes"), RAD_I32, { kCodes, S });

    /* The shared codebook table: the backbone embeds a frame from it, the depth decoder each
     * code. */
    m.w_cb = wcopy(b, nm, nm.f("frame.codebooks.weight"), "depth_decoder.model.embed_tokens.weight",
                   RAD_BF16, { (int64_t)kCodes * kVocab, D }, kGrpDepth);

    m.op_hsel = rows_op(b, "gather_rows", S, D, { m.b_hid }, { m.b_plane });
    m.op_head = rw(b, RAD_OP(b, "logits_gemm",
                             RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_vocab", kHeadRows),
                                        RAD_INT("n_embd", D), RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(m.w_head)),
                   { m.b_plane }, { m.b_slog });
    m.op_samp0 = rw(b, RAD_OP(b, "cfg_topk_sample",
                              RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_vocab", kHeadRows),
                                         RAD_INT("k", kTopK), RAD_INT("rep", 2),
                                         RAD_F64("cfg", 1.0),
                                         RAD_F64("temp", kTemp), RAD_INT("stream", 0),
                                         RAD_STR("dtype", "f32"), RAD_INT("sup_lo", kSupLo),
                                         RAD_INT("sup_hi", kSupHi), RAD_F64("penalty", kPenalty)),
                              RAD_NOWEIGHTS),
                    { m.b_slog, m.b_hist }, { m.b_codes, m.b_hist });
    m.op_status = rw(b, RAD_OP(b, "end_status",
                               RAD_PARAMS(RAD_RANGE("M", 1, lanes), RAD_INT("end", kEnd),
                                          RAD_INT("rep", 2)),
                               RAD_NOWEIGHTS),
                     { m.b_codes }, { m.b_status });

    /* ---- the depth decoder */
    gd = m.gl;
    const char* dc = "depth_decoder_config";
    gd.n_layer = rad_meta_geti(meta, nm.f("%s.num_hidden_layers", dc), 12);
    gd.n_embd = rad_meta_geti(meta, nm.f("%s.hidden_size", dc), 1024);
    gd.n_head = rad_meta_geti(meta, nm.f("%s.num_attention_heads", dc), 8);
    gd.n_head_kv = rad_meta_geti(meta, nm.f("%s.num_key_value_heads", dc), 2);
    gd.head_dim = rad_meta_geti(meta, nm.f("%s.head_dim", dc), 128);
    gd.n_ff = rad_meta_geti(meta, nm.f("%s.intermediate_size", dc), 8192);
    gd.eps = (float)rad_meta_getf(meta, nm.f("%s.rms_norm_eps", dc), 1e-5);
    gd.theta = (float)rad_meta_getf(meta, nm.f("%s.rope_theta", dc), 500000.0);
    m.l3.factor = rad_meta_getf(meta, nm.f("%s.rope_scaling.factor", dc), 32.0);
    m.l3.low = rad_meta_getf(meta, nm.f("%s.rope_scaling.low_freq_factor", dc), 0.001953125);
    m.l3.high = rad_meta_getf(meta, nm.f("%s.rope_scaling.high_freq_factor", dc), 0.0078125);
    m.l3.orig = rad_meta_getf(meta, nm.f("%s.rope_scaling.original_max_position_embeddings", dc),
                              16.0);
    gd.max_tok = R;
    const int64_t Dd = gd.n_embd;
    if (Dd != 1024 || gd.head_dim != 128 ||
        std::strcmp(rad_meta_gets(meta, nm.f("%s.rope_scaling.rope_type", dc), "llama3"),
                    "llama3") != 0 ||
        rad_meta_geti(meta, nm.f("%s.num_codebooks", dc), kCodes) != kCodes ||
        rad_meta_geti(meta, nm.f("%s.vocab_size", dc), kVocab) != kVocab) {
        fprintf(stderr, "radiance: breeze: the depth decoder is not the released one (width %lld, "
                        "head %lld)\n", (long long)Dd, (long long)gd.head_dim);
        return RAD_E_UNSUPPORTED;
    }
    m.w_din = wcopy(b, nm, nm.f("dep.in.weight"),
                    "depth_decoder.model.inputs_embeds_projector.weight", RAD_BF16, { Dd, D },
                    kGrpDepth);
    m.w_dnorm = wcopy(b, nm, nm.f("dep.norm.weight"), "depth_decoder.model.norm.weight", RAD_F32,
                      { Dd }, kGrpDepth);
    /* The fifteen heads, stored (in, out) [15, 1024, 2051]: read transposed, one [2051, 1024]
     * each. */
    RAD_ARCH_TRY(map_transpose(b, nm.f("dep.heads.weight"), "depth_decoder.codebooks_head.weight"));
    m.w_dheads = decl_w(b, nm.f("dep.heads.weight"), RAD_BF16, { (kCodes - 1) * kVocab, Dd },
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(kGrpDepth));

    m.b_dseq = decl_b(b, nm.f("dep.seq"), RAD_BF16, { R, Dd });
    m.b_dx   = decl_b(b, nm.f("dep.x"), RAD_BF16, { R, Dd });
    m.b_dh   = decl_b(b, nm.f("dep.h"), RAD_BF16, { R, Dd });
    m.b_dq   = decl_b(b, nm.f("dep.q"), RAD_BF16, { R, gd.n_head * gd.head_dim });
    m.b_dk   = decl_b(b, nm.f("dep.k"), RAD_BF16, { R, gd.n_head_kv * gd.head_dim });
    m.b_dv   = decl_b(b, nm.f("dep.v"), RAD_BF16, { R, gd.n_head_kv * gd.head_dim });
    m.b_dattn = decl_b(b, nm.f("dep.attn"), RAD_BF16, { R, gd.n_head * gd.head_dim });
    m.b_dh2  = decl_b(b, nm.f("dep.h2"), RAD_BF16, { R, Dd });
    m.b_dgu  = decl_b(b, nm.f("dep.gu"), RAD_BF16, { R, 2 * gd.n_ff });
    m.b_dff  = decl_b(b, nm.f("dep.ff"), RAD_BF16, { R, gd.n_ff });
    m.b_didx = decl_b(b, nm.f("dep.idx"), RAD_I32, { kCodes, S });
    m.b_dpos = decl_b(b, nm.f("dep.pos"), RAD_I32, { R });
    m.b_dcu  = decl_b(b, nm.f("dep.cu"), RAD_I32, { S + 1 });
    m.b_demb = decl_b(b, nm.f("dep.emb"), RAD_BF16, { S, D });
    m.b_drow = decl_b(b, nm.f("dep.row"), RAD_BF16, { S, Dd });
    m.b_dsel = decl_b(b, nm.f("dep.sel"), RAD_BF16, { S, Dd });
    m.b_dplane = decl_b(b, nm.f("dep.plane"), RAD_BF16, { S, Dd });
    m.b_dlog = decl_b(b, nm.f("dep.log"), RAD_F32, { S, kVocab });

    /* Declared in the order a frame issues them, so each buffer's live range is its use: the
     * sequence's indices, positions and bounds, and the inputs of positions 0 and p; then the
     * stack; then the read-out. The stack runs fifteen times, and what the read-out writes must
     * not land on what the next run reads (see the reads added at the end). */
    m.op_dcu = rw(b, RAD_OP(b, "cu_repeat",
                            RAD_PARAMS(RAD_RANGE("M", 1, S + 1), RAD_INT("mult", kCodes)),
                            RAD_NOWEIGHTS),
                  {}, { m.b_dcu });
    for (int p = 0; p < kCodes; ++p) m.op_iota[p] = iota_op(b, S, p, kCodes, m.b_didx);
    /* Every sequence's rows at positions 0..15. */
    m.op_dpos = rw(b, RAD_OP(b, "iota",
                             RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("start", 0),
                                        RAD_INT("step", 1), RAD_INT("period", kCodes)),
                             RAD_NOWEIGHTS),
                   {}, { m.b_dpos });
    /* The sequences start each frame as zeros: causal attention never reads a row past the step's
     * position, and zeros keep it finite as well (a buffer's first contents are arbitrary). */
    m.op_dzero = rw(b, RAD_OP(b, "fill",
                              RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", Dd),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    {}, { m.b_dseq });
    for (int p = 1; p < kCodes; ++p)
        /* Code p - 1's embedding (position p's input): codebook p - 1's rows of the table. */
        m.op_demb[p] = rw(b, RAD_OP(b, "rvq_embed",
                                    RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_embd", D),
                                               RAD_INT("n_codes", 1), RAD_INT("vocab", kVocab),
                                               RAD_INT("off0", (int64_t)(p - 1) * kVocab),
                                               RAD_F64("scale", 1.0), RAD_STR("dtype", "bf16")),
                                    RAD_WEIGHTS(m.w_cb, m.w_cb)),
                          { m.b_codes }, { m.b_demb });
    for (int p = 0; p < kCodes; ++p)
        m.op_din[p] = gemm_op(b, S, Dd, D, m.w_din, p == 0 ? m.b_plane : m.b_demb, m.b_drow);
    m.op_dscat = rw(b, RAD_OP(b, "scatter_rows",
                              RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", Dd),
                                         RAD_STR("dtype", "bf16")),
                              RAD_NOWEIGHTS),
                    { m.b_drow, m.b_didx }, { m.b_dseq });

    m.dep.resize((size_t)gd.n_layer);
    for (int64_t l = 0; l < gd.n_layer; ++l)
        RAD_ARCH_TRY(declare_layer(b, nm, gd, m.l3, m.dep[(size_t)l], (int)l, "dep",
                                   "depth_decoder.model.layers", kGrpDepth + 1 + (int)l, R, false,
                                   0, m.b_dx, m.b_dh, m.b_dq, m.b_dk, m.b_dv, m.b_dattn, m.b_dh2,
                                   m.b_dgu, m.b_dff, l == 0 ? m.b_dseq : m.b_dx, m.b_dpos));
    /* Each layer's attention also reads the frame's sequence offsets. */
    for (const LmLayer& L : m.dep) RAD_ARCH_TRY(rad_op_reads(b, L.op_attn, &m.b_dcu, 1));

    /* The incremental form (step_depth_inc): pass p runs only position p, one row a sequence,
     * through the stack's own ops, and its attention reads the keys and values the earlier passes
     * stored. Its tables are PERSIST, so the planner never lays them over anything; the stack's
     * buffers it uses in another order are carried through the read-out below. */
    {
        const int64_t kd = gd.n_head_kv * gd.head_dim, hd = gd.head_dim;
        m.b_dkc = decl_b(b, nm.f("dep.kc"), RAD_BF16, { gd.n_layer * R, kd }, RAD_BUF_PERSIST);
        m.b_dvc = decl_b(b, nm.f("dep.vc"), RAD_BF16, { gd.n_layer * R, kd }, RAD_BUF_PERSIST);
        m.b_dcuq = decl_b(b, nm.f("dep.cuq"), RAD_I32, { 2 }, RAD_BUF_PERSIST);
        m.b_dcuk = decl_b(b, nm.f("dep.cuk"), RAD_I32, { 2 * kCodes }, RAD_BUF_PERSIST);
        m.b_dposi = decl_b(b, nm.f("dep.posi"), RAD_I32, { S }, RAD_BUF_PERSIST);
        m.op_dcuq = iota_op(b, 2, 0, 1, m.b_dcuq);
        for (int p = 0; p < kCodes; ++p) {
            m.op_dcuk[p] = iota_op(b, 2, 0, p + 1, m.b_dcuk);
            m.op_dposi[p] = iota_op(b, S, p, 0, m.b_dposi);
        }
        for (int64_t l = 0; l < gd.n_layer; ++l) {
            auto scat = [&](rad_buf src, rad_buf cache) {
                return rw(b, RAD_OP(b, "scatter_rows",
                                    RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", kd),
                                               RAD_STR("dtype", "bf16")),
                                    RAD_NOWEIGHTS),
                          { src, m.b_didx }, { cache });
            };
            m.op_dsk.push_back(scat(m.b_dk, m.b_dkc));
            m.op_dsv.push_back(scat(m.b_dv, m.b_dvc));
            m.op_dxa.push_back(rw(b, RAD_OP(b, "attn_cross",
                                            RAD_PARAMS(RAD_RANGE("M", 1, 1), RAD_INT("head_dim", hd),
                                                       RAD_INT("gqa", gd.n_head / gd.n_head_kv),
                                                       RAD_INT("causal", 0), RAD_STR("dtype", "bf16"),
                                                       RAD_F64("scale", 1.0 / std::sqrt((double)hd))),
                                            RAD_NOWEIGHTS),
                                  { m.b_dq, m.b_dkc, m.b_dvc, m.b_dcuq, m.b_dcuk }, { m.b_dattn }));
            if (!m.op_dsk.back() || !m.op_dsv.back() || !m.op_dxa.back()) return RAD_E_INVAL;
        }
        m.b_dev = decl_b(b, nm.f("dep.ev"), RAD_I32, { S }, RAD_BUF_PERSIST);
        m.b_dodd = decl_b(b, nm.f("dep.odd"), RAD_I32, { S }, RAD_BUF_PERSIST);
        m.b_dpos2 = decl_b(b, nm.f("dep.pos2"), RAD_I32, { 2 * S }, RAD_BUF_PERSIST);
        m.b_d01 = decl_b(b, nm.f("dep.01"), RAD_I32, { 2 }, RAD_BUF_PERSIST);
        m.b_dcuq2 = decl_b(b, nm.f("dep.cuq2"), RAD_I32, { 2 }, RAD_BUF_PERSIST);
        m.op_dev = iota_op(b, S, 0, 2, m.b_dev);
        m.op_dodd = iota_op(b, S, 1, 2, m.b_dodd);
        m.op_dpos2 = rw(b, RAD_OP(b, "iota",
                                  RAD_PARAMS(RAD_RANGE("M", 1, 2 * S), RAD_INT("start", 0),
                                             RAD_INT("step", 1), RAD_INT("period", 2)),
                                  RAD_NOWEIGHTS),
                        {}, { m.b_dpos2 });
        m.op_d01 = iota_op(b, 2, 0, 1, m.b_d01);
        m.op_dcuq2 = iota_op(b, 2, 0, 2, m.b_dcuq2);
        for (int64_t l = 0; l < gd.n_layer; ++l) {
            m.op_dxa2.push_back(rw(b, RAD_OP(b, "attn_cross",
                                             RAD_PARAMS(RAD_RANGE("M", 1, 2), RAD_INT("head_dim", hd),
                                                        RAD_INT("gqa", gd.n_head / gd.n_head_kv),
                                                        RAD_INT("causal", 1), RAD_STR("dtype", "bf16"),
                                                        RAD_F64("scale", 1.0 / std::sqrt((double)hd))),
                                             RAD_NOWEIGHTS),
                                   { m.b_dq, m.b_dkc, m.b_dvc, m.b_dcuq2, m.b_dcuk }, { m.b_dattn }));
            if (!m.op_dxa2.back()) return RAD_E_INVAL;
        }
        if (!m.b_dev || !m.b_dodd || !m.b_dpos2 || !m.b_d01 || !m.b_dcuq2 || !m.op_dev ||
            !m.op_dodd || !m.op_dpos2 || !m.op_d01 || !m.op_dcuq2)
            return RAD_E_INVAL;
        m.op_dnorm_i = norm_op(b, S, Dd, gd.eps, m.w_dnorm, m.b_dx, m.b_dplane);
        if (!m.b_dkc || !m.b_dvc || !m.b_dcuq || !m.b_dcuk || !m.b_dposi || !m.op_dcuq ||
            !m.op_dnorm_i)
            return RAD_E_INVAL;
    }

    m.op_dgather = rows_op(b, "gather_rows", S, Dd, { m.b_dx, m.b_didx }, { m.b_dsel });
    m.op_dnorm = norm_op(b, S, Dd, gd.eps, m.w_dnorm, m.b_dsel, m.b_dplane);
    m.op_dhead = rw(b, RAD_OP(b, "logits_gemm",
                              RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_vocab", kVocab),
                                         RAD_INT("n_embd", Dd), RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(m.w_dheads)),
                    { m.b_dplane }, { m.b_dlog });
    for (int p = 1; p < kCodes; ++p)
        m.op_dsamp[p] = rw(b, RAD_OP(b, "cfg_topk_sample",
                                     RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_vocab", kVocab),
                                                RAD_INT("k", kTopK), RAD_INT("rep", 2),
                                                RAD_F64("cfg", 1.0), RAD_F64("temp", kTemp),
                                                RAD_INT("stream", p), RAD_STR("dtype", "f32"),
                                                RAD_INT("sup_lo", kSupLo),
                                                RAD_INT("sup_hi", kSupHi)),
                                     RAD_NOWEIGHTS),
                           { m.b_dlog }, { m.b_codes });
    /* The buffers the next run of the stack reads stay live through this one's read-out; the
     * incremental form also reads the stack's row buffers in its own order (layer 0's residual is
     * the projected row, the read-out reads the residual itself). */
    {
        const rad_buf carried[] = { m.b_dseq, m.b_dpos, m.b_dcu, m.b_didx, m.b_drow, m.b_demb,
                                    m.b_dx, m.b_dh, m.b_dq, m.b_dk, m.b_dv, m.b_dattn, m.b_dh2,
                                    m.b_dgu, m.b_dff };
        RAD_ARCH_TRY(rad_op_reads(b, m.op_dsamp[kCodes - 1], carried,
                                  (int)(sizeof carried / sizeof carried[0])));
    }

    /* ---- the frame's codes into the lane's history; the next input of the backbone */
    m.op_store = rw(b, RAD_OP(b, "frame_store",
                              RAD_PARAMS(RAD_RANGE("M", 1, lanes), RAD_INT("n_codes", kCodes),
                                         RAD_INT("stride", kMaxFrames), RAD_INT("rep", 2),
                                         RAD_INT("skip", 0)),
                              RAD_NOWEIGHTS),
                    { m.b_codes, m.b_status }, { m.b_codes_all });
    m.op_fb = rw(b, RAD_OP(b, "rvq_embed",
                           RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_embd", D),
                                      RAD_INT("n_codes", kCodes), RAD_INT("vocab", kVocab),
                                      RAD_INT("off0", 0), RAD_F64("scale", 1.0),
                                      RAD_STR("dtype", "bf16"), RAD_INT("round", 1)),
                           RAD_WEIGHTS(m.w_cb, m.w_cb)),
                 { m.b_codes }, { m.b_x });
    /* The backbone's last rows: normed (a prompt's last row a branch, or a frame's row) and into
     * the lane's state. */
    m.op_plast = rows_op(b, "gather_rows", S, D, { m.b_x }, { m.b_plane });
    m.op_pnorm = norm_op(b, S, D, m.gl.eps, m.w_norm, m.b_plane, m.b_last);
    m.op_pstore = rw(b, RAD_OP(b, "scatter_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", D),
                                          RAD_STR("dtype", "bf16")),
                               RAD_NOWEIGHTS),
                     { m.b_last }, { m.b_hid });
    return m.w_cb && m.w_din && m.w_dnorm && m.w_dheads && m.op_samp0 && m.op_store && m.op_fb
               ? RAD_OK
               : RAD_E_INVAL;
}

/* ================================================================== the codec */
static int vconv(RadBuilder* b, Names& nm, VConv& v, const char* dn, const char* src, int64_t cin,
                 int64_t cout, int64_t k, int64_t dil, int64_t rows, rad_buf x, rad_buf res,
                 rad_buf y, rad_buf cu) {
    v.w = wcopy(b, nm, nm.f("%s.weight", dn), nm.ckpt("%s.weight", src), RAD_BF16, { cout, cin, k },
                kGrpCodec);
    v.b = wcopy(b, nm, nm.f("%s.bias", dn), nm.ckpt("%s.bias", src), RAD_BF16, { cout }, kGrpCodec);
    if (!v.w || !v.b) return RAD_E_INVAL;
    const RadParam ps[] = { RAD_RANGE("M", 1, rows), RAD_INT("cin", cin), RAD_INT("cout", cout),
                            RAD_INT("k", k), RAD_INT("dil", dil), RAD_STR("act", "none"),
                            RAD_STR("dtype", "bf16"), RAD_INT("causal", 1) };
    const rad_weight ws[] = { v.w, v.b };
    v.op = res ? rw(b, rad_decl_op(b, "conv1d", ps, 8, ws, 2), { x, cu, res }, { y })
               : rw(b, rad_decl_op(b, "conv1d", ps, 8, ws, 2), { x, cu }, { y });
    return v.op ? RAD_OK : RAD_E_INVAL;
}

/* A snake-beta activation's two log-scale vectors. */
static int snake_beta(RadBuilder* b, Names& nm, const char* dn, const char* src, int64_t n,
                      rad_weight* a, rad_weight* be) {
    *a = wcopy(b, nm, nm.f("%s.alpha", dn), nm.ckpt("%s.alpha", src), RAD_F32, { n }, kGrpCodec);
    *be = wcopy(b, nm, nm.f("%s.beta", dn), nm.ckpt("%s.beta", src), RAD_F32, { n }, kGrpCodec);
    return *a && *be ? RAD_OK : RAD_E_INVAL;
}

static rad_op snake_op(RadBuilder* b, int64_t rows, int64_t n, rad_weight a, rad_weight be,
                       rad_buf x, rad_buf y) {
    return rw(b, RAD_OP(b, "snake",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n),
                                   RAD_STR("dtype", "bf16"), RAD_INT("log", 1)),
                        RAD_WEIGHTS(a, be)),
              { x }, { y });
}

static int declare_codec(RadBuilder* b, const RadModelMeta* meta, Model& m) {
    Names& nm = m.nm;
    const char* pc = "audio_tokenizer.decoder_config";
    const int64_t Hc = rad_meta_geti(meta, nm.f("%s.hidden_size", pc), 512);
    const int64_t Lat = rad_meta_geti(meta, nm.f("%s.latent_dim", pc), 1024);
    const int64_t Dec = rad_meta_geti(meta, nm.f("%s.decoder_dim", pc), 1536);
    const int64_t Cd = rad_meta_geti(meta, nm.f("%s.codebook_dim", pc), 512);
    const int64_t nq = rad_meta_geti(meta, nm.f("%s.num_quantizers", pc), kCodes);
    const int64_t nl_c = rad_meta_geti(meta, nm.f("%s.num_hidden_layers", pc), 8);
    const int64_t heads = rad_meta_geti(meta, nm.f("%s.num_attention_heads", pc), 16);
    const int64_t hd = rad_meta_geti(meta, nm.f("%s.head_dim", pc), 64);
    const int64_t ffc = rad_meta_geti(meta, nm.f("%s.intermediate_size", pc), 1024);
    if (std::strcmp(rad_meta_gets(meta, "audio_tokenizer.model_type", ""),
                    "qwen3_tts_tokenizer_12hz") != 0 ||
        Hc != 512 || Lat != 1024 || Dec != 1536 || Cd != 512 || nq != kCodes || hd != 64 ||
        rad_meta_geti(meta, nm.f("%s.sliding_window", pc), kAttnWin) != kAttnWin ||
        rad_meta_geti(meta, "audio_tokenizer.output_sample_rate", kSampleRate) != kSampleRate) {
        fprintf(stderr, "radiance: breeze: the checkpoint has no audio_tokenizer/ folder holding "
                        "the released Qwen3-TTS 12 Hz codec\n");
        return RAD_E_UNSUPPORTED;
    }
    const int64_t NL = kVocContext + kVocChunk;       /* frames through the transformer */
    const int64_t NC = kConvFrames;                    /* frames through the convolutions */
    const int64_t qd = heads * hd;
    const char* q = "audio_tokenizer.decoder.quantizer";

    /* ---- the dequantiser: two residual quantisers, each projected 256 -> 512 */
    RAD_ARCH_TRY(map_ema(b, nm.f("codec.cb0.weight"),
                         nm.ckpt("%s.rvq_first.vq.layers.0._codebook.embedding_sum", q),
                         nm.ckpt("%s.rvq_first.vq.layers.0._codebook.cluster_usage", q)));
    m.w_cb0 = decl_w(b, nm.f("codec.cb0.weight"), RAD_BF16, { kCbBins, kCbDim },
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(kGrpCodec));
    for (int j = 0; j < kCodes - 1; ++j)
        RAD_ARCH_TRY(map_ema(b, nm.f("codec.cbr.weight"),
                             nm.ckpt("%s.rvq_rest.vq.layers.%d._codebook.embedding_sum", q, j),
                             nm.ckpt("%s.rvq_rest.vq.layers.%d._codebook.cluster_usage", q, j)));
    m.w_cbr = decl_w(b, nm.f("codec.cbr.weight"), RAD_BF16, { (kCodes - 1) * kCbBins, kCbDim },
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(kGrpCodec));
    m.w_qo0 = wcopy(b, nm, nm.f("codec.q0.weight"), nm.ckpt("%s.rvq_first.output_proj.weight", q),
                    RAD_BF16, { Cd, kCbDim }, kGrpCodec);
    m.w_qor = wcopy(b, nm, nm.f("codec.qr.weight"), nm.ckpt("%s.rvq_rest.output_proj.weight", q),
                    RAD_BF16, { Cd, kCbDim }, kGrpCodec);
    m.b_rq0 = decl_b(b, nm.f("codec.rq0"), RAD_BF16, { NL, kCbDim });
    m.b_rqr = decl_b(b, nm.f("codec.rqr"), RAD_BF16, { NL, kCbDim });
    m.b_qz  = decl_b(b, nm.f("codec.qz"), RAD_BF16, { NL, Cd });
    m.b_qt  = decl_b(b, nm.f("codec.qt"), RAD_BF16, { NL, Cd });
    auto rvq = [&](rad_weight t0, rad_weight t1, int64_t K, rad_buf y) {
        return rw(b, RAD_OP(b, "rvq_embed",
                            RAD_PARAMS(RAD_RANGE("M", 1, NL), RAD_INT("n_embd", kCbDim),
                                       RAD_INT("n_codes", K), RAD_INT("vocab", kCbBins),
                                       RAD_INT("off0", 0), RAD_F64("scale", 1.0),
                                       RAD_STR("dtype", "bf16"), RAD_INT("round", 1)),
                            RAD_WEIGHTS(t0, t1)),
                  { m.b_codes_all }, { y });
    };
    m.op_rq0 = rvq(m.w_cb0, m.w_cb0, 1, m.b_rq0);
    m.op_rqr = rvq(m.w_cbr, m.w_cbr, kCodes - 1, m.b_rqr);
    m.op_qp0 = gemm_op(b, NL, Cd, kCbDim, m.w_qo0, m.b_rq0, m.b_qz);
    m.op_qpr = gemm_op(b, NL, Cd, kCbDim, m.w_qor, m.b_rqr, m.b_qt);
    m.op_qadd = add_op(b, NL, Cd, m.b_qz, m.b_qt, m.b_qz);

    /* ---- the sequence bounds, one sequence at each rate: the transformer's frames (the pass's
     * own), then the convolutions' (the last kConvFrames of them, or all when fewer) at the frame
     * rate, 2x, 4x, and after each of the decoder's four blocks */
    m.b_cconst = decl_b(b, nm.f("codec.nconv"), RAD_I32, { 1 });
    m.op_cnst = iota_op(b, 1, NC, 0, m.b_cconst);
    for (int s2 = 0; s2 < kRates; ++s2) {
        m.b_ccu[s2] = decl_b(b, nm.f("codec.cu%d", s2), RAD_I32, { 2 });
        m.op_ccu[s2] = s2 == 0 ? cu_op(b, 1, 0, m.b_ccu[0])
                               : cu_op(b, kRate[s2], m.b_cconst, m.b_ccu[s2]);
    }

    /* ---- the causal 3-tap conv and the transformer */
    m.b_pre = decl_b(b, nm.f("codec.pre"), RAD_BF16, { NL, Lat });
    RAD_ARCH_TRY(vconv(b, nm, m.pre, "codec.pre", "audio_tokenizer.decoder.pre_conv.conv", Cd, Lat,
                       3, 1, NL, m.b_qz, 0, m.b_pre, m.b_ccu[0]));
    const char* pt = "audio_tokenizer.decoder.pre_transformer";
    m.w_ip = wcopy(b, nm, nm.f("codec.in.weight"), nm.ckpt("%s.input_proj.weight", pt), RAD_BF16,
                   { Hc, Lat }, kGrpCodec);
    m.b_ip = wcopy(b, nm, nm.f("codec.in.bias"), nm.ckpt("%s.input_proj.bias", pt), RAD_BF16,
                   { Hc }, kGrpCodec);
    m.b_cx   = decl_b(b, nm.f("codec.x"), RAD_BF16, { NL, Hc });
    m.b_ch   = decl_b(b, nm.f("codec.h"), RAD_BF16, { NL, Hc });
    m.b_cqkv = decl_b(b, nm.f("codec.qkv"), RAD_BF16, { NL, 3 * qd });
    m.b_cattn = decl_b(b, nm.f("codec.attn"), RAD_BF16, { NL, qd });
    m.b_co   = decl_b(b, nm.f("codec.o"), RAD_BF16, { NL, Hc });
    m.b_cgu  = decl_b(b, nm.f("codec.gu"), RAD_BF16, { NL, 2 * ffc });
    m.b_cff  = decl_b(b, nm.f("codec.ff"), RAD_BF16, { NL, ffc });
    m.op_ip = gemm_bias_op(b, NL, Hc, Lat, m.w_ip, m.b_ip, m.b_pre, m.b_cx);
    m.ct.resize((size_t)nl_c);
    for (int64_t l = 0; l < nl_c; ++l) {
        CodecLayer& L = m.ct[(size_t)l];
        const int li = (int)l;
        auto src = [&](const char* s) { return nm.ckpt("%s.layers.%d.%s", pt, li, s); };
        L.w_n1 = wcopy(b, nm, nm.f("codec.%d.n1.weight", li), src("input_layernorm.weight"),
                       RAD_F32, { Hc }, kGrpCodec);
        L.w_n2 = wcopy(b, nm, nm.f("codec.%d.n2.weight", li),
                       src("post_attention_layernorm.weight"), RAD_F32, { Hc }, kGrpCodec);
        L.w_ls1 = wcopy(b, nm, nm.f("codec.%d.ls1.weight", li), src("self_attn_layer_scale.scale"),
                        RAD_F32, { Hc }, kGrpCodec);
        L.w_ls2 = wcopy(b, nm, nm.f("codec.%d.ls2.weight", li), src("mlp_layer_scale.scale"),
                        RAD_F32, { Hc }, kGrpCodec);
        L.w_qkv = wcat(b, nm.f("codec.%d.qkv.weight", li),
                       { src("self_attn.q_proj.weight"), src("self_attn.k_proj.weight"),
                         src("self_attn.v_proj.weight") }, { 3 * qd, Hc }, kGrpCodec);
        L.w_o = wcopy(b, nm, nm.f("codec.%d.o.weight", li), src("self_attn.o_proj.weight"),
                      RAD_BF16, { Hc, qd }, kGrpCodec);
        L.w_gu = wcat(b, nm.f("codec.%d.gate_up.weight", li),
                      { src("mlp.gate_proj.weight"), src("mlp.up_proj.weight") }, { 2 * ffc, Hc },
                      kGrpCodec);
        L.w_down = wcopy(b, nm, nm.f("codec.%d.down.weight", li), src("mlp.down_proj.weight"),
                         RAD_BF16, { Hc, ffc }, kGrpCodec);
        L.op_n1 = norm_op(b, NL, Hc, 1e-5, L.w_n1, m.b_cx, m.b_ch);
        L.op_qkv = gemm_op(b, NL, 3 * qd, Hc, L.w_qkv, m.b_ch, m.b_cqkv);
        L.op_rope = rw(b, RAD_OP(b, "rope",
                                 RAD_PARAMS(RAD_RANGE("M", 1, NL), RAD_INT("head_dim", hd),
                                            RAD_INT("n_head", heads), RAD_INT("n_head_kv", heads),
                                            RAD_F64("theta",
                                                    rad_meta_getf(meta, nm.f("%s.rope_theta", pc),
                                                                  10000.0)),
                                            RAD_F64("scale", 1.0), RAD_STR("mode", "neox")),
                                 RAD_NOWEIGHTS),
                       { m.b_cqkv }, { m.b_cqkv });
        L.op_attn = rw(b, RAD_OP(b, "attn_dense",
                                 RAD_PARAMS(RAD_RANGE("M", 1, NL), RAD_INT("head_dim", hd),
                                            RAD_INT("gqa", 1), RAD_INT("causal", 1),
                                            RAD_STR("dtype", "bf16"),
                                            RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                            RAD_INT("max_seqlen", NL), RAD_INT("window", kAttnWin)),
                                 RAD_NOWEIGHTS),
                       { m.b_cqkv }, { m.b_cattn });
        L.op_o = gemm_op(b, NL, Hc, qd, L.w_o, m.b_cattn, m.b_co);
        L.op_ls1 = mulw_op(b, NL, Hc, L.w_ls1, m.b_co, m.b_co);
        L.op_add1 = add_op(b, NL, Hc, m.b_cx, m.b_co, m.b_cx);
        L.op_n2 = norm_op(b, NL, Hc, 1e-5, L.w_n2, m.b_cx, m.b_ch);
        L.op_gu = gemm_op(b, NL, 2 * ffc, Hc, L.w_gu, m.b_ch, m.b_cgu);
        L.op_act = rows_op(b, "silu_mul", NL, ffc, { m.b_cgu }, { m.b_cff });
        L.op_down = gemm_op(b, NL, Hc, ffc, L.w_down, m.b_cff, m.b_co);
        L.op_ls2 = mulw_op(b, NL, Hc, L.w_ls2, m.b_co, m.b_co);
        L.op_add2 = add_op(b, NL, Hc, m.b_cx, m.b_co, m.b_cx);
        if (!L.w_n1 || !L.w_n2 || !L.w_ls1 || !L.w_ls2 || !L.w_qkv || !L.w_o || !L.w_gu ||
            !L.w_down || !L.op_rope || !L.op_attn || !L.op_ls1 || !L.op_ls2)
            return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_op_reads(b, L.op_attn, &m.b_ccu[0], 1));
    }
    m.w_cn = wcopy(b, nm, nm.f("codec.norm.weight"), nm.ckpt("%s.norm.weight", pt), RAD_F32, { Hc },
                   kGrpCodec);
    m.op_cn = norm_op(b, NL, Hc, 1e-5, m.w_cn, m.b_cx, m.b_ch);
    m.w_op = wcopy(b, nm, nm.f("codec.out.weight"), nm.ckpt("%s.output_proj.weight", pt), RAD_BF16,
                   { Lat, Hc }, kGrpCodec);
    m.b_op = wcopy(b, nm, nm.f("codec.out.bias"), nm.ckpt("%s.output_proj.bias", pt), RAD_BF16,
                   { Lat }, kGrpCodec);
    m.b_cy = decl_b(b, nm.f("codec.y"), RAD_BF16, { NL, Lat });
    m.op_op = gemm_bias_op(b, NL, Lat, Hc, m.w_op, m.b_op, m.b_ch, m.b_cy);

    /* ---- two 2x upsamplings, each a transposed conv of kernel 2 and a ConvNeXt block */
    m.b_ua = decl_b(b, nm.f("codec.ua"), RAD_BF16, { 4 * NC, Lat });
    m.b_ub = decl_b(b, nm.f("codec.ub"), RAD_BF16, { 4 * NC, Lat });
    m.b_uh = decl_b(b, nm.f("codec.uh"), RAD_BF16, { 4 * NC, 4 * Lat });
    m.b_ug = decl_b(b, nm.f("codec.ug"), RAD_BF16, { 4 * NC, 4 * Lat });
    m.b_ux[0] = m.b_cy;
    for (int i = 0; i < 2; ++i) {
        Upsample& U = m.up[i];
        const int64_t rin = NC << i, rout = NC << (i + 1);
        const char* us = nm.ckpt("audio_tokenizer.decoder.upsample.%d", i);
        auto w = [&](const char* dn, const char* s, uint32_t dt,
                     std::initializer_list<int64_t> sh) {
            return wcopy(b, nm, nm.f("codec.up%d.%s", i, dn), nm.ckpt("%s.%s", us, s), dt, sh,
                         kGrpCodec);
        };
        U.w_t = w("t.weight", "0.conv.weight", RAD_BF16, { Lat, Lat, 2 });
        U.b_t = w("t.bias", "0.conv.bias", RAD_BF16, { Lat });
        U.w_dw = w("dw.weight", "1.dwconv.conv.weight", RAD_BF16, { Lat, 1, 7 });
        U.b_dw = w("dw.bias", "1.dwconv.conv.bias", RAD_BF16, { Lat });
        U.w_ln = w("ln.weight", "1.norm.weight", RAD_F32, { Lat });
        U.b_ln = w("ln.bias", "1.norm.bias", RAD_F32, { Lat });
        U.w_p1 = w("p1.weight", "1.pwconv1.weight", RAD_BF16, { 4 * Lat, Lat });
        U.b_p1 = w("p1.bias", "1.pwconv1.bias", RAD_BF16, { 4 * Lat });
        U.w_p2 = w("p2.weight", "1.pwconv2.weight", RAD_BF16, { Lat, 4 * Lat });
        U.b_p2 = w("p2.bias", "1.pwconv2.bias", RAD_BF16, { Lat });
        U.w_g = w("gamma", "1.gamma", RAD_F32, { Lat });
        m.b_ux[i + 1] = decl_b(b, nm.f("codec.ux%d", i + 1), RAD_BF16, { rout, Lat });
        U.op_t = rw(b, RAD_OP(b, "conv_t1d",
                              RAD_PARAMS(RAD_RANGE("M", 1, rin), RAD_INT("cin", Lat),
                                         RAD_INT("cout", Lat), RAD_INT("stride", 2),
                                         RAD_STR("dtype", "bf16"), RAD_INT("pad", 0),
                                         RAD_INT("k", 2)),
                              RAD_WEIGHTS(U.w_t, U.b_t)),
                    { m.b_ux[i], m.b_ccu[1 + i] }, { m.b_ux[i + 1] });
        U.op_dw = rw(b, RAD_OP(b, "conv1d_dw",
                               RAD_PARAMS(RAD_RANGE("M", 1, rout), RAD_INT("n", Lat),
                                          RAD_INT("k", 7), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(U.w_dw, U.b_dw)),
                     { m.b_ux[i + 1], m.b_ccu[2 + i] }, { m.b_ua });
        U.op_ln = rw(b, RAD_OP(b, "layernorm",
                               RAD_PARAMS(RAD_RANGE("M", 1, rout), RAD_INT("n", Lat),
                                          RAD_F64("eps", 1e-6), RAD_STR("dtype", "bf16")),
                               RAD_WEIGHTS(U.w_ln, U.b_ln)),
                     { m.b_ua }, { m.b_ub });
        U.op_p1 = gemm_bias_op(b, rout, 4 * Lat, Lat, U.w_p1, U.b_p1, m.b_ub, m.b_uh);
        U.op_gelu = rows_op(b, "gelu", rout, 4 * Lat, { m.b_uh }, { m.b_ug });
        U.op_p2 = gemm_bias_op(b, rout, Lat, 4 * Lat, U.w_p2, U.b_p2, m.b_ug, m.b_ua);
        U.op_g = mulw_op(b, rout, Lat, U.w_g, m.b_ua, m.b_ua);
        U.op_add = add_op(b, rout, Lat, m.b_ux[i + 1], m.b_ua, m.b_ux[i + 1]);
        if (!U.w_t || !U.b_t || !U.w_dw || !U.b_dw || !U.w_ln || !U.b_ln || !U.w_p1 || !U.b_p1 ||
            !U.w_p2 || !U.b_p2 || !U.w_g || !U.op_t || !U.op_dw || !U.op_ln || !U.op_gelu)
            return RAD_E_INVAL;
    }

    /* ---- the decoder: a conv into 1536 channels, four upsampling blocks, the output conv */
    const int64_t R0 = 4 * NC;   /* rows at the 50 Hz latent rate */
    int64_t rows[5], ch[5];
    rows[0] = R0;
    ch[0] = Dec;
    for (int i = 0; i < 4; ++i) {
        rows[i + 1] = rows[i] * m.v_ups[i];
        ch[i + 1] = Dec >> (i + 1);
    }
    for (int s = 0; s < 5; ++s) {
        m.b_vs[s] = decl_b(b, nm.f("codec.s%d", s), RAD_BF16, { rows[s], ch[s] });
        m.b_vt[s] = decl_b(b, nm.f("codec.t%d", s), RAD_BF16, { rows[s], ch[s] });
        if (s > 0) {
            m.b_va[s] = decl_b(b, nm.f("codec.a%d", s), RAD_BF16, { rows[s], ch[s] });
            m.b_vb[s] = decl_b(b, nm.f("codec.b%d", s), RAD_BF16, { rows[s], ch[s] });
        }
    }
    const char* dd = "audio_tokenizer.decoder.decoder";
    RAD_ARCH_TRY(vconv(b, nm, m.dec0, "codec.dec0", nm.ckpt("%s.0.conv", dd), Lat, Dec, 7, 1, R0,
                       m.b_ux[2], 0, m.b_vs[0], m.b_ccu[3]));
    static const int64_t kDil[3] = { 1, 3, 9 };
    for (int i = 0; i < 4; ++i) {
        VBlock& B = m.vb[i];
        const int64_t ci = ch[i], co = ch[i + 1], s = m.v_ups[i];
        B.cin = ci; B.cout = co; B.stride = s;
        const char* bn = nm.f("codec.b%d", i);
        const char* bs = nm.ckpt("%s.%d.block", dd, i + 1);
        RAD_ARCH_TRY(snake_beta(b, nm, nm.f("%s.snake", bn), nm.ckpt("%s.0", bs), ci, &B.a, &B.b));
        B.op_s = snake_op(b, rows[i], ci, B.a, B.b, m.b_vs[i], m.b_vt[i]);
        B.wt = wcopy(b, nm, nm.f("%s.t.weight", bn), nm.ckpt("%s.1.conv.weight", bs), RAD_BF16,
                     { ci, co, 2 * s }, kGrpCodec);
        B.bt = wcopy(b, nm, nm.f("%s.t.bias", bn), nm.ckpt("%s.1.conv.bias", bs), RAD_BF16, { co },
                     kGrpCodec);
        B.op_t = rw(b, RAD_OP(b, "conv_t1d",
                              RAD_PARAMS(RAD_RANGE("M", 1, rows[i]), RAD_INT("cin", ci),
                                         RAD_INT("cout", co), RAD_INT("stride", s),
                                         RAD_STR("dtype", "bf16"), RAD_INT("pad", 0),
                                         RAD_INT("k", 2 * s)),
                              RAD_WEIGHTS(B.wt, B.bt)),
                    { m.b_vt[i], m.b_ccu[3 + i] }, { m.b_vs[i + 1] });
        if (!B.wt || !B.bt || !B.op_t || !B.op_s) return RAD_E_INVAL;
        for (int r = 0; r < 3; ++r) {
            VRes& U = B.res[r];
            const char* un = nm.f("%s.res%d", bn, r);
            const char* us = nm.ckpt("%s.%d", bs, r + 2);
            const int64_t R = rows[i + 1];
            RAD_ARCH_TRY(snake_beta(b, nm, nm.f("%s.act1", un), nm.ckpt("%s.act1", us), co, &U.a1,
                                    &U.b1));
            RAD_ARCH_TRY(snake_beta(b, nm, nm.f("%s.act2", un), nm.ckpt("%s.act2", us), co, &U.a2,
                                    &U.b2));
            U.op_s1 = snake_op(b, R, co, U.a1, U.b1, m.b_vs[i + 1], m.b_vt[i + 1]);
            RAD_ARCH_TRY(vconv(b, nm, U.c1, nm.f("%s.conv1", un), nm.ckpt("%s.conv1.conv", us), co,
                               co, 7, kDil[r], R, m.b_vt[i + 1], 0, m.b_va[i + 1], m.b_ccu[4 + i]));
            U.op_s2 = snake_op(b, R, co, U.a2, U.b2, m.b_va[i + 1], m.b_vb[i + 1]);
            RAD_ARCH_TRY(vconv(b, nm, U.c2, nm.f("%s.conv2", un), nm.ckpt("%s.conv2.conv", us), co,
                               co, 1, 1, R, m.b_vb[i + 1], m.b_vs[i + 1], m.b_vs[i + 1],
                               m.b_ccu[4 + i]));
            if (!U.op_s1 || !U.op_s2) return RAD_E_INVAL;
        }
    }
    RAD_ARCH_TRY(snake_beta(b, nm, "codec.snake_out", nm.ckpt("%s.5", dd), ch[4], &m.w_fa,
                            &m.w_fb));
    m.op_fs = snake_op(b, rows[4], ch[4], m.w_fa, m.w_fb, m.b_vs[4], m.b_vt[4]);
    m.b_vout = decl_b(b, nm.f("codec.out"), RAD_BF16, { rows[4], 1 });
    RAD_ARCH_TRY(vconv(b, nm, m.fin, "codec.conv_out", nm.ckpt("%s.6.conv", dd), ch[4], 1, 7, 1,
                       rows[4], m.b_vt[4], 0, m.b_vout, m.b_ccu[7]));
    m.b_audio = decl_b(b, nm.f("codec.audio"), RAD_F32, { kVocChunk * kSamplesPerFrame, 1 },
                       RAD_BUF_PERSIST);
    m.op_fcast = cast_op(b, (int64_t)kVocChunk * kSamplesPerFrame, 1, "bf16", "f32", m.b_vout,
                         m.b_audio);
    return m.w_cb0 && m.w_cbr && m.w_qo0 && m.w_qor && m.w_ip && m.b_ip && m.w_cn && m.w_op &&
                   m.b_op && m.op_fs && m.op_fcast && m.op_cnst
               ? RAD_OK
               : RAD_E_INVAL;
}

/* ================================================================== check_format */
static int check_format(const RadModelMeta* meta) {
    if (std::strcmp(rad_meta_gets(meta, "backbone_model_type", ""), "qwen3") != 0 ||
        rad_meta_geti(meta, "num_codebooks", 0) != kCodes ||
        rad_meta_geti(meta, "vocab_size", 0) != kVocab ||
        rad_meta_geti(meta, "audio_token_id", 0) != 262144) {
        fprintf(stderr, "radiance: breeze: not the released Breeze-TTS-2 layout (a Qwen3 backbone "
                        "over 16 codebooks of 2051)\n");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

#include "reference.h"

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS || rad_build_slot(ctx) >= MAX_SLOTS)
        return RAD_E_INVAL;
    Model& m = g_model[rad_build_slot(ctx)];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(check_format(meta));
    if (ctx->world_size > 1) {
        fprintf(stderr, "radiance: breeze: Breeze-TTS-2 is served on one card (--tp 1): a speech's "
                        "pace is the step's latency, which tensor parallelism only lengthens\n");
        return RAD_E_UNSUPPORTED;
    }
    RAD_ARCH_TRY(geom_from(m.gl, meta, ctx, "bf16", RAD_BF16));
    /* The backbone's own configuration: the top level's rotary and eps are the depth decoder's
     * family's, not the Qwen3 backbone's. */
    m.gl.eps = (float)rad_meta_getf(meta, "backbone_config.rms_norm_eps", 1e-6);
    m.gl.theta = (float)rad_meta_getf(meta, "backbone_config.rope_theta", 1000000.0);
    m.lanes = std::min<int64_t>(kMaxLanes, std::max<int64_t>(1, ctx->max_seqs / 2));
    m.S = 2 * m.lanes;
    m.ge = m.gl;
    if (m.gl.max_tok < 2 * (int64_t)kMaxPrompt || m.gl.max_tok < kVocContext + kVocChunk) {
        fprintf(stderr, "radiance: breeze: a pass carries %lld rows; a prompt pass needs %lld\n",
                (long long)m.gl.max_tok, (long long)(2 * kMaxPrompt));
        return RAD_E_INVAL;
    }

    RAD_ARCH_TRY(declare_backbone(b, meta, ctx, m));
    RAD_ARCH_TRY(declare_encoder(b, meta, m));
    RAD_ARCH_TRY(declare_frame(b, meta, m));
    RAD_ARCH_TRY(declare_codec(b, meta, m));
    /* A container converted before the reference encoder was in the recipe holds the generator
     * alone: it serves speech as before, and with no reference templates declared below the
     * server refuses a request that brings references, saying the model clones no voice. */
    m.ref_stored = reference_stored(b, m);
    if (m.ref_stored) RAD_ARCH_TRY(declare_reference(b, meta, m));

    if (!ctx->shape_probe) {
        RadAudioGenDecl d{};
        d.name = "breeze-tts-2";
        d.tmpl = kTemplate;
        d.tmpl_plain = kPlain;
        d.neg_tmpl = kPlain;
        d.branches = 2;
        if (m.ref_stored) {
            d.tmpl_ref = "{references}<bos>[S0]{input}";
            d.tmpl_ref_instruct = "{references}<bos>[S0]<ins_bos>{instructions}<ins_eos>{input}";
            d.neg_tmpl_ref = d.tmpl_ref;
            d.ref_item = "<bos>[S0]{text}{audio}<|audio_eos|>";
            d.ref_rate = kSampleRate;
            d.ref_hop = kSamplesPerFrame;
            d.ref_max_samples = kMaxRefSamples;
            d.ref_max = 1;
        }
        d.prompt_whole = 1;
        d.first_frame = 0;
        d.max_prompt = kMaxPrompt;
        d.max_frames = kMaxFrames;
        d.frame_rate_num = 25;
        d.frame_rate_den = 2;
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
        RAD_ARCH_TRY(rad_declare_audio_generator(b, &d));
    }
    rad_note(b, "Breeze-TTS-2 -- text encoder %lld layers, backbone %lld layers (%lld heads over "
                "%lld KV), depth decoder %lld layers, codec transformer %zu layers, %lld lanes",
             (long long)m.ge.n_layer, (long long)m.gl.n_layer, (long long)m.gl.n_head,
             (long long)m.gl.n_head_kv, (long long)m.gd.n_layer, m.ct.size(), (long long)m.lanes);
    return RAD_OK;
}

/* ================================================================== step */
static void store_hidden(RadCtx* c, const Model& m, const RadAudioGenBatch* A, int64_t S,
                         const int32_t* last_rows, int64_t T) {
    RAD_ISSUE_N(c, m.op_plast, S, brows(m.b_x, T), praw(last_rows, RAD_I32, S),
                brows(m.b_plane, S));
    RAD_ISSUE_N(c, m.op_pnorm, S, brows(m.b_plane, S), RAD_W(m.w_norm), brows(m.b_last, S));
    RAD_ISSUE_N(c, m.op_pstore, S, brows(m.b_last, S), praw(A->slot, RAD_I32, S),
                brows(m.b_hid, 2 * m.lanes));
}

static void step_backbone(RadCtx* c, const RadBatch* batch, const Model& m) {
    const int64_t T = batch->n_tok;
    const RadOperand cu = praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1);
    const RadOperand pos = praw(batch->positions, RAD_I32, T);
    int layer = 0;
    for (const LmLayer& L : m.lm)
        step_layer(c, m, m.gl, L, batch, T, m.b_x, m.b_x, m.b_h, m.b_q, m.b_k, m.b_v, m.b_attn,
                   m.b_h2, m.b_gu, m.b_ff, pos, cu, layer++);
}

static void step_text(RadCtx* c, const Model& m, int64_t T, RadOperand tokens,
                       RadOperand pos, RadOperand cu, RadOperand output) {
    const int64_t qd = m.ge.n_head * m.ge.head_dim, kd = m.ge.n_head_kv * m.ge.head_dim;
    /* The encoder over each branch's prompt, bidirectional within it; its projection is the
     * backbone's input. */
    RAD_ISSUE_N(c, m.op_eembed, T, tokens, RAD_W(m.w_etok),
              brows(m.b_ex, T));
    for (const EncLayer& L : m.enc) {
        RAD_ISSUE_N(c, L.op_n1, T, brows(m.b_ex, T), RAD_W(L.w_n1), brows(m.b_eh, T));
        RAD_ISSUE_N(c, L.op_qkv, T, brows(m.b_eh, T), RAD_W(L.w_qkv), brows(m.b_eqkv, T));
        RAD_ISSUE_N(c, L.op_qknr, T, brows(m.b_eqkv, T), pos, RAD_W(L.w_qn), RAD_W(L.w_kn));
        RAD_ISSUE_N(c, L.op_attn, T, bcol(m.b_eqkv, 0, qd, T), bcol(m.b_eqkv, qd, kd, T),
                    bcol(m.b_eqkv, qd + kd, kd, T), cu, brows(m.b_eattn, T));
        RAD_ISSUE_N(c, L.op_o, T, brows(m.b_eattn, T), RAD_W(L.w_o), brows(m.b_eo, T));
        RAD_ISSUE_N(c, L.op_n2, T, brows(m.b_eo, T), RAD_W(L.w_n2), brows(m.b_eh, T));
        RAD_ISSUE_N(c, L.op_add1, T, brows(m.b_ex, T), brows(m.b_eh, T), brows(m.b_ex, T));
        RAD_ISSUE_N(c, L.op_n3, T, brows(m.b_ex, T), RAD_W(L.w_n3), brows(m.b_eh, T));
        RAD_ISSUE_N(c, L.op_gu, T, brows(m.b_eh, T), RAD_W(L.w_gu), brows(m.b_egu, T));
        RAD_ISSUE_N(c, L.op_act, T, brows(m.b_egu, T), brows(m.b_eff, T));
        RAD_ISSUE_N(c, L.op_down, T, brows(m.b_eff, T), RAD_W(L.w_down), brows(m.b_eo, T));
        RAD_ISSUE_N(c, L.op_n4, T, brows(m.b_eo, T), RAD_W(L.w_n4), brows(m.b_eh, T));
        RAD_ISSUE_N(c, L.op_add2, T, brows(m.b_ex, T), brows(m.b_eh, T), brows(m.b_ex, T));
    }
    RAD_ISSUE_N(c, m.op_enorm, T, brows(m.b_ex, T), RAD_W(m.w_enorm), brows(m.b_eh, T));
    RAD_ISSUE_N(c, m.op_eproj, T, brows(m.b_eh, T), RAD_W(m.w_eproj), output);
}

/* rvq_embed reads codebook 0 from its first table and codebooks 1..15 from its
 * second table, whose row zero must therefore be the start of codebook 1. */
static RadOperand residual_codebooks(const Model& m) {
    RadOperand t = RAD_W(m.w_cb);
    t.offset = (int64_t)kVocab * m.gl.n_embd;
    t.rows = (int64_t)(kCodes - 1) * kVocab;
    return t;
}

static void step_prompt(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadAudioGenBatch* A = batch->agen_b;
    const int64_t T = batch->n_tok;
    if (A->n_ref) {
        if (!A->prompt_cu_h || A->prompt_cu_h[0] != 0 ||
            A->prompt_cu_h[batch->n_seq] != T) {
            rad_step_fail(c, "breeze: reference prompt needs its host branch boundaries");
            return;
        }
        step_reference(c, batch, m);
        const Reference& E = m.ref;
        RAD_ISSUE_N(c, E.text_indices, kMaxPrompt, brows(E.text_pos, kMaxPrompt));
        RAD_ISSUE_N(c, E.eos, 16, brows(E.eos_codes, 16));
        /* Each branch has two independent text-encoder segments around its audio. The
         * backbone alone attends across the concatenation. Never encode audio placeholders
         * or the audio EOS as text, and reset rotary positions for each text segment. */
        for (int64_t branch = 0; branch < batch->n_seq; ++branch) {
            const int64_t end = A->prompt_cu_h[branch + 1];
            for (int64_t t = A->prompt_cu_h[branch]; t < end;) {
                const int32_t f = A->ref_frame_h[t];
                if (f >= 0) {
                    int64_t n = 1;
                    while (t + n < end && A->ref_frame_h[t + n] == f + n) ++n;
                    if (t + n >= end) {
                        rad_step_fail(c, "breeze: reference frames need their audio EOS row");
                        return;
                    }
                    RAD_ISSUE_N(c, E.embed, n, bcol(E.codes, f, n, 16), RAD_W(m.w_cb),
                                residual_codebooks(m), brow_slice(m.b_x, t, n, m.gl.n_embd));
                    t += n;
                    RAD_ISSUE_N(c, E.embed, 1, brows(E.eos_codes, 16), RAD_W(m.w_cb),
                                residual_codebooks(m), brow_slice(m.b_x, t, 1, m.gl.n_embd));
                    ++t;
                } else {
                    int64_t n = 1;
                    while (t + n < end && A->ref_frame_h[t + n] < 0) ++n;
                    ref_bounds(c, E, E.text_cu, n);
                    step_text(c, m, n, praw(batch->token_ids + t, RAD_I32, n),
                              brows(E.text_pos, n), brows(E.text_cu, 2),
                              brow_slice(m.b_x, t, n, m.gl.n_embd));
                    t += n;
                }
            }
        }
    } else {
        step_text(c, m, T, praw(batch->token_ids, RAD_I32, T),
                  praw(batch->positions, RAD_I32, T),
                  praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1), brows(m.b_x, T));
    }
    step_backbone(c, batch, m);
    bool any_last = false;
    for (int64_t i = 0; i < A->n_lane; ++i) any_last = any_last || A->last_h[i];
    if (any_last) store_hidden(c, m, A, batch->n_seq, A->last_row, T);
}

/* The depth decoder's sixteen positions of every sequence, one pass for positions 0 and 1 (each
 * sequence's pair causal over itself) and one pass for each position p >= 2 alone (S rows). Each
 * pass stores its keys and values in the per-layer store and attends over positions 0..p from it,
 * one sequence at a time: the stack is causal, so a position's states never change once computed,
 * and every GEMM runs at decode height instead of re-running the stack over all 16 S rows
 * fifteen times a frame. On the 8060S that took a frame from 131 ms to 77. */
static void step_depth_inc(RadCtx* c, const Model& m, int64_t S, const RadOperand& seed,
                           const RadOperand& frame, const RadOperand& cfg,
                           const std::function<RadOperand(int)>& codes,
                           const std::function<RadOperand(int)>& idx) {
    const Geom& g = m.gd;
    const int64_t Rx = (int64_t)kCodes * m.S;       /* a layer's rows in the store */
    const int64_t kd = g.n_head_kv * g.head_dim, qd = g.n_head * g.head_dim;
    const RadOperand cuq = brows(m.b_dcuq, 2), cuq2 = brows(m.b_dcuq2, 2);
    RAD_ISSUE_N(c, m.op_dcuq, 2, cuq);
    RAD_ISSUE_N(c, m.op_dcuq2, 2, cuq2);
    RAD_ISSUE_N(c, m.op_d01, 2, brows(m.b_d01, 2));
    RAD_ISSUE_N(c, m.op_dev, S, brows(m.b_dev, S));
    RAD_ISSUE_N(c, m.op_dodd, S, brows(m.b_dodd, S));
    RAD_ISSUE_N(c, m.op_dpos2, 2 * S, brows(m.b_dpos2, 2 * S));
    /* Pass 1 runs positions 0 and 1 (T = 2 S rows, sequence s's pair at rows 2 s and 2 s + 1, in
     * b_dseq); pass p >= 2 position p alone (S rows, layer 0's input the projected row). */
    for (int p = 1; p < kCodes; ++p) {
        const bool pair = p == 1;
        const int64_t T = pair ? 2 * S : S;
        const RadOperand cuk = brow_slice(m.b_dcuk, p, 2, 2);
        const RadOperand pos = pair ? brows(m.b_dpos2, T) : brows(m.b_dposi, S);
        RAD_ISSUE_N(c, m.op_dcuk[p], 2, cuk);
        if (pair) {
            RAD_ISSUE_N(c, m.op_din[0], S, brows(m.b_plane, S), RAD_W(m.w_din), brows(m.b_drow, S));
            RAD_ISSUE_N(c, m.op_dscat, S, brows(m.b_drow, S), brows(m.b_dev, S), brows(m.b_dseq, T));
        } else {
            RAD_ISSUE_N(c, m.op_dposi[p], S, pos);
        }
        RAD_ISSUE_N(c, m.op_demb[p], S, codes(p - 1), RAD_W(m.w_cb), RAD_W(m.w_cb),
                    brows(m.b_demb, S));
        RAD_ISSUE_N(c, m.op_din[p], S, brows(m.b_demb, S), RAD_W(m.w_din), brows(m.b_drow, S));
        if (pair)
            RAD_ISSUE_N(c, m.op_dscat, S, brows(m.b_drow, S), brows(m.b_dodd, S), brows(m.b_dseq, T));
        for (size_t l = 0; l < m.dep.size(); ++l) {
            const LmLayer& L = m.dep[l];
            const rad_buf x_in = l > 0 ? m.b_dx : pair ? m.b_dseq : m.b_drow;
            RAD_ISSUE(c, L.op_n1, brows(x_in, T), RAD_W(L.w_n1), brows(m.b_dh, T));
            RAD_ISSUE(c, L.op_q, brows(m.b_dh, T), RAD_W(L.w_q), brows(m.b_dq, T));
            RAD_ISSUE(c, L.op_k, brows(m.b_dh, T), RAD_W(L.w_k), brows(m.b_dk, T));
            RAD_ISSUE(c, L.op_v, brows(m.b_dh, T), RAD_W(L.w_v), brows(m.b_dv, T));
            RAD_ISSUE(c, L.op_rq, brows(m.b_dq, T), pos);
            RAD_ISSUE(c, L.op_rk, brows(m.b_dk, T), pos);
            const int64_t l0 = (int64_t)l * Rx;
            if (pair) {
                /* A pair's two rows into its sequence's first two store rows, then causal over
                 * them. */
                for (int64_t sq = 0; sq < S; ++sq) {
                    const RadOperand kc = brow_slice(m.b_dkc, l0 + kCodes * sq, kCodes, kd);
                    const RadOperand vc = brow_slice(m.b_dvc, l0 + kCodes * sq, kCodes, kd);
                    RAD_ISSUE_N(c, m.op_dsk[l], 2, brow_slice(m.b_dk, 2 * sq, 2, kd),
                                brows(m.b_d01, 2), kc);
                    RAD_ISSUE_N(c, m.op_dsv[l], 2, brow_slice(m.b_dv, 2 * sq, 2, kd),
                                brows(m.b_d01, 2), vc);
                    RAD_ISSUE_N(c, m.op_dxa2[l], 2, brow_slice(m.b_dq, 2 * sq, 2, qd), kc, vc, cuq2,
                                cuk, brow_slice(m.b_dattn, 2 * sq, 2, qd));
                }
            } else {
                RAD_ISSUE_N(c, m.op_dsk[l], S, brows(m.b_dk, S), idx(p),
                            brow_slice(m.b_dkc, l0, Rx, kd));
                RAD_ISSUE_N(c, m.op_dsv[l], S, brows(m.b_dv, S), idx(p),
                            brow_slice(m.b_dvc, l0, Rx, kd));
                for (int64_t sq = 0; sq < S; ++sq)
                    RAD_ISSUE_N(c, m.op_dxa[l], 1, brow_slice(m.b_dq, sq, 1, qd),
                                brow_slice(m.b_dkc, l0 + kCodes * sq, kCodes, kd),
                                brow_slice(m.b_dvc, l0 + kCodes * sq, kCodes, kd), cuq, cuk,
                                brow_slice(m.b_dattn, sq, 1, qd));
            }
            RAD_ISSUE(c, L.op_o, brows(m.b_dattn, T), RAD_W(L.w_o), brows(m.b_dh2, T));
            RAD_ISSUE(c, L.op_add1, brows(x_in, T), brows(m.b_dh2, T), brows(m.b_dx, T));
            RAD_ISSUE(c, L.op_n2, brows(m.b_dx, T), RAD_W(L.w_n2), brows(m.b_dh, T));
            RAD_ISSUE(c, L.op_gu, brows(m.b_dh, T), RAD_W(L.w_gu), brows(m.b_dgu, T));
            RAD_ISSUE(c, L.op_act, brows(m.b_dgu, T), brows(m.b_dff, T));
            RAD_ISSUE(c, L.op_down, brows(m.b_dff, T), RAD_W(L.w_down), brows(m.b_dh2, T));
            RAD_ISSUE(c, L.op_add2, brows(m.b_dx, T), brows(m.b_dh2, T), brows(m.b_dx, T));
        }
        if (pair) {
            /* Position 1's rows are the read-out's. */
            RAD_ISSUE_N(c, m.op_dgather, S, brows(m.b_dx, T), brows(m.b_dodd, S), brows(m.b_dsel, S));
            RAD_ISSUE_N(c, m.op_dnorm, S, brows(m.b_dsel, S), RAD_W(m.w_dnorm), brows(m.b_dplane, S));
        } else {
            RAD_ISSUE_N(c, m.op_dnorm_i, S, brows(m.b_dx, S), RAD_W(m.w_dnorm), brows(m.b_dplane, S));
        }
        RadOperand head = RAD_W(m.w_dheads);
        head.offset = (int64_t)(p - 1) * kVocab * g.n_embd;
        head.rows = kVocab;
        RAD_ISSUE_N(c, m.op_dhead, S, brows(m.b_dplane, S), head, brows(m.b_dlog, S));
        RAD_ISSUE_N(c, m.op_dsamp[p], S, brows(m.b_dlog, S), seed, frame, codes(p), cfg, RAD_NONE,
                    RAD_NONE);
    }
}

static void step_frame(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadAudioGenBatch* A = batch->agen_b;
    const int64_t n = A->n_lane, S = 2 * n;
    const int64_t Sx = m.S;   /* the codes' and indices' row pitch */
    const RadOperand seed = praw(A->seed, RAD_U32, 2 * n), frame = praw(A->frame, RAD_I32, n);
    const RadOperand lanes = praw(A->lane, RAD_I32, n), slot = praw(A->slot, RAD_I32, S);
    const RadOperand cfg = praw(A->cfg, RAD_F32, n);
    /* Row p of a code-major [16, Sx] plane, narrowed to this pass's S entries. */
    auto row_of = [&](rad_buf h, int p) {
        RadOperand o = brow_slice(h, p, 1, Sx);
        o.cols = S;
        return o;
    };
    auto codes = [&](int p) { return row_of(m.b_codes, p); };
    auto idx = [&](int p) { return row_of(m.b_didx, p); };

    /* 1. the first code, from the lanes' hidden states */
    RAD_ISSUE_N(c, m.op_hsel, S, brows(m.b_hid, 2 * m.lanes), slot, brows(m.b_plane, S));
    RAD_ISSUE_N(c, m.op_head, S, brows(m.b_plane, S), RAD_W(m.w_head), brows(m.b_slog, S));
    RAD_ISSUE_N(c, m.op_samp0, S, brows(m.b_slog, S), seed, frame, codes(0), cfg, lanes,
                brows(m.b_hist, m.lanes));
    RAD_ISSUE_N(c, m.op_status, n, codes(0), lanes, brows(m.b_status, m.lanes));

    /* 2. the depth decoder's sequences: 16 rows each, the hidden state's projection at position 0
     * and each code's from position 1 */
    for (int p = 0; p < kCodes; ++p) RAD_ISSUE_N(c, m.op_iota[p], S, idx(p));
    step_depth_inc(c, m, S, seed, frame, cfg, codes, idx);

    /* 3. the frame into its lane's history, and the backbone over the frame's embedding */
    RAD_ISSUE_N(c, m.op_store, n, brows(m.b_codes, kCodes), lanes, frame,
                brows(m.b_status, m.lanes), brows(m.b_codes_all, m.lanes * kCodes));
    RAD_ISSUE_N(c, m.op_fb, S, brows(m.b_codes, kCodes), RAD_W(m.w_cb), residual_codebooks(m),
                brows(m.b_x, S));
    step_backbone(c, batch, m);
    store_hidden(c, m, A, S, A->last_row, S);
}

static void step_vocode(RadCtx* c, const RadBatch* batch, const Model& m) {
    const RadAudioGenBatch* A = batch->agen_b;
    const int64_t nl = A->voc_latents, n = nl - A->voc_left, L = A->voc_lane;
    const int64_t Hc = 512, qd = 1024;
    /* The convolutions run over the last kConvFrames frames, or all of them when fewer. */
    const int64_t nc = std::min<int64_t>(nl, kConvFrames), c0 = nl - nc;
    const RadOperand nlen = praw(batch->cu_seqlens + 1, RAD_I32, 1);
    const RadOperand clen = nc == nl ? nlen : brows(m.b_cconst, 1);

    /* 1. the codes, dequantised: the lane's frames [voc_frame0, + nl) of each codebook */
    RadOperand codes = bcol_at(m.b_codes_all, L * kCodes, kMaxFrames, A->voc_frame0, nl, kCodes);
    RadOperand rest = bcol_at(m.b_codes_all, L * kCodes + 1, kMaxFrames, A->voc_frame0, nl,
                              kCodes - 1);
    RadOperand t1 = RAD_W(m.w_cbr);
    t1.offset = (int64_t)kCbBins * kCbDim;
    t1.rows = (int64_t)(kCodes - 2) * kCbBins;
    RAD_ISSUE_N(c, m.op_rq0, nl, codes, RAD_W(m.w_cb0), RAD_W(m.w_cb0), brows(m.b_rq0, nl));
    RAD_ISSUE_N(c, m.op_rqr, nl, rest, RAD_W(m.w_cbr), t1, brows(m.b_rqr, nl));
    RAD_ISSUE_N(c, m.op_qp0, nl, brows(m.b_rq0, nl), RAD_W(m.w_qo0), brows(m.b_qz, nl));
    RAD_ISSUE_N(c, m.op_qpr, nl, brows(m.b_rqr, nl), RAD_W(m.w_qor), brows(m.b_qt, nl));
    RAD_ISSUE_N(c, m.op_qadd, nl, brows(m.b_qz, nl), brows(m.b_qt, nl), brows(m.b_qz, nl));

    /* 2. the causal conv and the transformer over every frame of the pass */
    /* In declaration order: the constant, then the bounds that read it. */
    if (nc != nl) RAD_ISSUE_N(c, m.op_cnst, 1, brows(m.b_cconst, 1));
    RAD_ISSUE_N(c, m.op_ccu[0], 2, nlen, brows(m.b_ccu[0], 2));
    for (int s2 = 1; s2 < kRates; ++s2)
        RAD_ISSUE_N(c, m.op_ccu[s2], 2, clen, brows(m.b_ccu[s2], 2));
    RAD_ISSUE_N(c, m.pre.op, nl, brows(m.b_qz, nl), RAD_W(m.pre.w), brows(m.b_ccu[0], 2), RAD_NONE,
                brows(m.b_pre, nl), RAD_W(m.pre.b));
    RAD_ISSUE_N(c, m.op_ip, nl, brows(m.b_pre, nl), RAD_W(m.w_ip), RAD_W(m.b_ip), RAD_NONE,
                brows(m.b_cx, nl));
    const RadOperand pos = praw(batch->positions, RAD_I32, nl);
    const RadOperand cu = brows(m.b_ccu[0], 2);
    for (const CodecLayer& Ly : m.ct) {
        RAD_ISSUE_N(c, Ly.op_n1, nl, brows(m.b_cx, nl), RAD_W(Ly.w_n1), brows(m.b_ch, nl));
        RAD_ISSUE_N(c, Ly.op_qkv, nl, brows(m.b_ch, nl), RAD_W(Ly.w_qkv), brows(m.b_cqkv, nl));
        RAD_ISSUE_N(c, Ly.op_rope, nl, brows(m.b_cqkv, nl), pos);
        RAD_ISSUE_N(c, Ly.op_attn, nl, bcol(m.b_cqkv, 0, qd, nl), bcol(m.b_cqkv, qd, qd, nl),
                    bcol(m.b_cqkv, 2 * qd, qd, nl), cu, brows(m.b_cattn, nl));
        RAD_ISSUE_N(c, Ly.op_o, nl, brows(m.b_cattn, nl), RAD_W(Ly.w_o), brows(m.b_co, nl));
        RAD_ISSUE_N(c, Ly.op_ls1, nl, brows(m.b_co, nl), RAD_W(Ly.w_ls1), brows(m.b_co, nl));
        RAD_ISSUE_N(c, Ly.op_add1, nl, brows(m.b_cx, nl), brows(m.b_co, nl), brows(m.b_cx, nl));
        RAD_ISSUE_N(c, Ly.op_n2, nl, brows(m.b_cx, nl), RAD_W(Ly.w_n2), brows(m.b_ch, nl));
        RAD_ISSUE_N(c, Ly.op_gu, nl, brows(m.b_ch, nl), RAD_W(Ly.w_gu), brows(m.b_cgu, nl));
        RAD_ISSUE_N(c, Ly.op_act, nl, brows(m.b_cgu, nl), brows(m.b_cff, nl));
        RAD_ISSUE_N(c, Ly.op_down, nl, brows(m.b_cff, nl), RAD_W(Ly.w_down), brows(m.b_co, nl));
        RAD_ISSUE_N(c, Ly.op_ls2, nl, brows(m.b_co, nl), RAD_W(Ly.w_ls2), brows(m.b_co, nl));
        RAD_ISSUE_N(c, Ly.op_add2, nl, brows(m.b_cx, nl), brows(m.b_co, nl), brows(m.b_cx, nl));
    }
    RAD_ISSUE_N(c, m.op_cn, nl, brows(m.b_cx, nl), RAD_W(m.w_cn), brows(m.b_ch, nl));
    /* Only the frames the convolutions read: the last nc. */
    RAD_ISSUE_N(c, m.op_op, nc, brow_slice(m.b_ch, c0, nc, Hc), RAD_W(m.w_op), RAD_W(m.b_op),
                RAD_NONE, brows(m.b_cy, nc));

    /* 3. the upsamplings, at 25 and 50 rows a second */
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
        RAD_ISSUE_N(c, U.op_gelu, rout, brows(m.b_uh, rout), brows(m.b_ug, rout));
        RAD_ISSUE_N(c, U.op_p2, rout, brows(m.b_ug, rout), RAD_W(U.w_p2), RAD_W(U.b_p2), RAD_NONE,
                    brows(m.b_ua, rout));
        RAD_ISSUE_N(c, U.op_g, rout, brows(m.b_ua, rout), RAD_W(U.w_g), brows(m.b_ua, rout));
        RAD_ISSUE_N(c, U.op_add, rout, brows(m.b_ux[i + 1], rout), brows(m.b_ua, rout),
                    brows(m.b_ux[i + 1], rout));
    }

    /* 4. the decoder */
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
                    RAD_W(B.b));
        RAD_ISSUE_N(c, B.op_t, r0, brows(m.b_vt[i], r0), RAD_W(B.wt), RAD_W(B.bt),
                    brows(m.b_ccu[3 + i], 2), brows(m.b_vs[i + 1], r1));
        for (int r = 0; r < 3; ++r) {
            const VRes& U = B.res[r];
            RAD_ISSUE_N(c, U.op_s1, r1, brows(m.b_vs[i + 1], r1), RAD_W(U.a1),
                        brows(m.b_vt[i + 1], r1), RAD_W(U.b1));
            conv(U.c1, m.b_vt[i + 1], 0, m.b_va[i + 1], r1, m.b_ccu[4 + i]);
            RAD_ISSUE_N(c, U.op_s2, r1, brows(m.b_va[i + 1], r1), RAD_W(U.a2),
                        brows(m.b_vb[i + 1], r1), RAD_W(U.b2));
            conv(U.c2, m.b_vb[i + 1], m.b_vs[i + 1], m.b_vs[i + 1], r1, m.b_ccu[4 + i]);
        }
    }
    RAD_ISSUE_N(c, m.op_fs, rows[4], brows(m.b_vs[4], rows[4]), RAD_W(m.w_fa),
                brows(m.b_vt[4], rows[4]), RAD_W(m.w_fb));
    conv(m.fin, m.b_vt[4], 0, m.b_vout, rows[4], m.b_ccu[7]);
    /* 5. the chunk's own samples: its last n frames */
    const int64_t ns = n * kSamplesPerFrame;
    RAD_ISSUE_N(c, m.op_fcast, ns, brow_slice(m.b_vout, rows[4] - ns, ns, 1), brows(m.b_audio, ns));
}

static void step(RadCtx* c, const RadBatch* batch) {
    const Model& m = g_model[rad_slot(c)];
    if (!batch->gen || !batch->agen_b) {
        rad_step_fail(c, "breeze is an audio generator and runs generator passes only");
        return;
    }
    switch (batch->gen) {
    case RAD_GEN_PASS_PROMPT: step_prompt(c, batch, m); return;
    case RAD_GEN_PASS_FRAME:  step_frame(c, batch, m); return;
    case RAD_GEN_PASS_VOCODE: step_vocode(c, batch, m); return;
    default:
        rad_step_fail(c, "breeze: a generator pass of a kind this plugin does not know");
    }
}

/* ================================================================== the probe */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->gen_rows = 2 * kMaxPrompt;
    out->gen_audio = 1;
    out->gen_ctx = (int64_t)kMaxPrompt + kMaxFrames + 1;
    return RAD_OK;
}

/* ================================================================== the request's texts */
/* The reference strips the instructions; the text goes in as written. */
static int64_t text_prepare(int field, const char* in, char* out, int64_t cap) {
    if (!in) return RAD_E_INVAL;
    std::string r = in;
    if (field == RAD_TEXT_REFERENCE) {
        for (const char* marker : {"<bos>", "<eos>", "<|AUDIO|>", "<|audio_eos|>",
                                   "<ins_bos>", "<ins_eos>"})
            if (r.find(marker) != std::string::npos) return RAD_E_INVAL;
    }
    if (field == RAD_TEXT_INSTRUCTIONS) {
        const size_t a = r.find_first_not_of(" \t\r\n"), z = r.find_last_not_of(" \t\r\n");
        r = a == std::string::npos ? std::string() : r.substr(a, z - a + 1);
    }
    if ((int64_t)r.size() < cap) {
        std::memcpy(out, r.data(), r.size());
        out[r.size()] = 0;
    }
    return (int64_t)r.size();
}

}  /* namespace breeze */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(breeze)
extern "C" int64_t rad_arch_text_prepare(int field, const char* in, char* out, int64_t cap) {
    return breeze::text_prepare(field, in, out, cap);
}
/* Every per-instance state above is indexed by slot (rad_build_slot / rad_slot). */
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PLUGIN(breeze, "breeze", "", "0.1.0",
                "Breeze-TTS-2: a T5Gemma 2 text encoder over a Qwen3 backbone with a 16-codebook "
                "depth decoder and the Qwen3-TTS 12 Hz codec; 24 kHz mono speech")
#endif
